"""Lossless per-step evidence capture for pinned VLFM evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
from typing import Any

import cv2
import numpy as np


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size <= 32:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _array(value: Any, dtype: Any | None = None) -> np.ndarray:
    """Copy an array-like value without retaining a live Torch tensor."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=dtype)
    return np.array(array, copy=True)


def _safe(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unknown"


def _episode_meta(episode: Any) -> dict[str, Any]:
    goals = []
    for goal in getattr(episode, "goals", []) or []:
        goals.append(
            {
                "position": _plain(getattr(goal, "position", None)),
                "object_id": _plain(getattr(goal, "object_id", None)),
                "object_name": _plain(getattr(goal, "object_name", None)),
            }
        )
    return {
        "scene_id": str(getattr(episode, "scene_id", "unknown")),
        "episode_id": str(getattr(episode, "episode_id", "unknown")),
        "object_category": str(getattr(episode, "object_category", "unknown")),
        "start_position": _plain(getattr(episode, "start_position", None)),
        "start_rotation": _plain(getattr(episode, "start_rotation", None)),
        "goals": goals,
    }


class EvidenceRecorder:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.signature: str | None = None
        self.episode_dir: pathlib.Path | None = None
        self.step = 0
        self.detection_call = 0
        self.event_call = 0
        self.raw_video = None
        self.boxed_video = None

    def _start_episode(self, episode: Any) -> None:
        self.close_videos()
        meta = _episode_meta(episode)
        signature = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(signature.encode()).hexdigest()[:12]
        scene = pathlib.Path(meta["scene_id"]).stem
        self.episode_dir = self.root / "episodes" / f"{_safe(scene)}__{_safe(meta['episode_id'])}__{digest}"
        for name in ("rgb", "depth", "sensors", "detections", "maps"):
            (self.episode_dir / name).mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "episode.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.signature = signature
        self.step = 0
        self.detection_call = 0
        self.event_call = 0

    def record_observation(self, episode: Any, observation: dict[str, Any]) -> None:
        signature = json.dumps(_episode_meta(episode), sort_keys=True, separators=(",", ":"))
        if signature != self.signature:
            self._start_episode(episode)
        assert self.episode_dir is not None
        rgb = np.asarray(observation.get("rgb"))
        depth = np.asarray(observation.get("depth"))
        stem = f"{self.step:04d}"
        if rgb.size:
            rgb = np.ascontiguousarray(rgb.astype(np.uint8))
            cv2.imwrite(str(self.episode_dir / "rgb" / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if self.raw_video is None:
                h, w = rgb.shape[:2]
                self.raw_video = cv2.VideoWriter(
                    str(self.episode_dir / "raw_rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h)
                )
            self.raw_video.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if depth.size:
            depth = np.asarray(depth, dtype=np.float32)
            np.save(self.episode_dir / "depth" / f"{stem}.npy", depth, allow_pickle=False)
            preview = np.clip(np.squeeze(depth), 0.0, 1.0)
            cv2.imwrite(str(self.episode_dir / "depth" / f"{stem}.png"), (preview * 65535).astype(np.uint16))
        sensors = {}
        for key, value in observation.items():
            if key not in {"rgb", "depth"}:
                try:
                    sensors[key] = np.asarray(value)
                except Exception:
                    pass
        if sensors:
            np.savez_compressed(self.episode_dir / "sensors" / f"{stem}.npz", **sensors)
        self.detection_call = 0

    def record_detections(self, detections: Any, target: str, stage: str = "filtered") -> None:
        if self.episode_dir is None or detections is None:
            return
        image = getattr(detections, "image_source", None)
        boxes = _array(getattr(detections, "boxes", []), dtype=float)
        logits = _array(getattr(detections, "logits", []), dtype=float)
        phrases = list(getattr(detections, "phrases", []))
        call_dir = self.episode_dir / "detections" / f"{self.step:04d}_{self.detection_call:02d}_{_safe(stage)}"
        call_dir.mkdir(parents=True, exist_ok=True)
        payload = {"step": self.step, "call": self.detection_call, "stage": stage, "target": target,
                   "boxes_xyxy": boxes.tolist(), "scores": logits.tolist(), "phrases": phrases}
        (call_dir / "detections.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if image is not None:
            image = np.asarray(image, dtype=np.uint8)
            h, w = image.shape[:2]
            # Draw independently instead of accessing detections.annotated_frame.
            # Upstream chooses random box colors there, which advances NumPy's RNG
            # and could perturb a baseline run.  The recorder must be observational.
            annotated = image.copy()
            for idx, box in enumerate(boxes):
                b = box.copy()
                if b.size == 4 and np.max(b) <= 1.0:
                    b *= np.array([w, h, w, h])
                x1, y1, x2, y2 = np.rint(b).astype(int)
                x1, x2 = sorted((max(0, min(w - 1, x1)), max(0, min(w - 1, x2))))
                y1, y2 = sorted((max(0, min(h - 1, y1)), max(0, min(h - 1, y2))))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = phrases[idx] if idx < len(phrases) else "object"
                score = float(np.ravel(logits[idx])[0]) if idx < len(logits) else 0.0
                cv2.putText(annotated, f"{label}:{score:.3f}", (x1, max(12, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(call_dir / "boxed.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            for idx, box in enumerate(boxes):
                b = box.copy()
                if b.size == 4 and np.max(b) <= 1.0:
                    b *= np.array([w, h, w, h])
                x1, y1, x2, y2 = np.rint(b).astype(int)
                x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
                y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
                if x2 > x1 and y2 > y1:
                    label = _safe(phrases[idx] if idx < len(phrases) else "object")
                    crop = image[y1:y2, x1:x2]
                    cv2.imwrite(str(call_dir / f"crop_{idx:03d}_{label}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if stage == "filtered_target":
                if self.boxed_video is None:
                    self.boxed_video = cv2.VideoWriter(
                        str(self.episode_dir / "detections.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h)
                    )
                self.boxed_video.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        self.detection_call += 1

    def record_model_event(
        self,
        kind: str,
        request: dict[str, Any],
        response: Any,
        image: Any | None = None,
        array: Any | None = None,
    ) -> None:
        """Record an already-completed model call without changing its value."""
        if self.episode_dir is None:
            return
        event_dir = self.episode_dir / "model_events" / f"{self.step:04d}_{self.event_call:03d}_{_safe(kind)}"
        event_dir.mkdir(parents=True, exist_ok=False)
        payload = {
            "step": self.step,
            "call": self.event_call,
            "kind": kind,
            "request": _plain(request),
            "response": _plain(response),
        }
        (event_dir / "event.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if image is not None:
            frame = _array(image, dtype=np.uint8)
            if frame.ndim == 3 and frame.shape[2] >= 3:
                cv2.imwrite(str(event_dir / "image.png"), cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR))
        if array is not None:
            value = _array(array)
            np.save(event_dir / "array.npy", value, allow_pickle=False)
            if value.ndim in (2, 3):
                preview = np.squeeze(value)
                if preview.ndim == 2:
                    preview = (preview.astype(bool) * 255).astype(np.uint8)
                    cv2.imwrite(str(event_dir / "array.png"), preview)
        self.event_call += 1

    def record_transition(self, action: Any, reward: Any, done: Any, info: dict[str, Any], policy_info: dict[str, Any]) -> None:
        if self.episode_dir is None:
            return
        record = {"step": self.step, "action": _plain(action), "reward": _plain(reward), "done": bool(done),
                  "info": _plain(info), "policy_info": _plain(policy_info)}
        with (self.episode_dir / "steps.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        maps = self.episode_dir / "maps"
        top = info.get("top_down_map") if isinstance(info, dict) else None
        if isinstance(top, dict) and isinstance(top.get("map"), np.ndarray):
            np.save(maps / f"{self.step:04d}_top_down.npy", top["map"], allow_pickle=False)
        for key in ("obstacle_map", "value_map", "target_point_cloud"):
            value = policy_info.get(key) if isinstance(policy_info, dict) else None
            if isinstance(value, np.ndarray):
                np.save(maps / f"{self.step:04d}_{key}.npy", value, allow_pickle=False)
        self.step += 1

    def finish_episode(self, stats: dict[str, Any], failure_cause: str) -> None:
        if self.episode_dir is None:
            return
        payload = {"steps_recorded": self.step, "failure_cause": failure_cause, "metrics": _plain(stats)}
        (self.episode_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.close_videos()
        # Force the next reset to initialize a fresh episode even if a dataset
        # happens to repeat the same metadata.
        self.signature = None
        self.episode_dir = None

    def close_videos(self) -> None:
        for attr in ("raw_video", "boxed_video"):
            writer = getattr(self, attr)
            if writer is not None:
                writer.release()
                setattr(self, attr, None)


_RECORDER: EvidenceRecorder | None = None


def get_evidence_recorder() -> EvidenceRecorder:
    global _RECORDER
    if _RECORDER is None:
        root = pathlib.Path(os.environ["VLFM_EVIDENCE_DIR"])
        _RECORDER = EvidenceRecorder(root)
    return _RECORDER
