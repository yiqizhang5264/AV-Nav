#!/usr/bin/env python3
"""Run pinned VLFM with lossless evidence capture enabled."""

from __future__ import annotations

import argparse
import os
import pathlib
import runpy
import sys

from vlfm_evidence_adapter import create_vlfm_evidence_overlay
from vlfm_evidence_recorder import get_evidence_recorder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlfm-root", required=True)
    args, hydra_args = parser.parse_known_args()
    root = pathlib.Path(args.vlfm_root).resolve()
    overlay = create_vlfm_evidence_overlay(root)
    sys.path.insert(0, overlay.name)
    sys.path.insert(1, str(root))

    from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
    from vlfm.vlm.blip2 import BLIP2Client
    from vlfm.vlm.blip2itm import BLIP2ITMClient
    from vlfm.vlm.grounding_dino import GroundingDINOClient
    from vlfm.vlm.sam import MobileSAMClient
    from vlfm.vlm.yolov7 import YOLOv7Client

    original = BaseObjectNavPolicy._get_object_detections

    def capture(self, image):
        detections = original(self, image)
        get_evidence_recorder().record_detections(detections, str(self._target_object), "filtered_target")
        return detections

    BaseObjectNavPolicy._get_object_detections = capture

    def wrap_detector(cls, stage):
        method = cls.predict

        def wrapped(self, image, *method_args, **method_kwargs):
            result = method(self, image, *method_args, **method_kwargs)
            caption = method_kwargs.get("caption", method_args[0] if method_args else "")
            get_evidence_recorder().record_detections(result, str(caption), stage)
            return result

        cls.predict = wrapped

    wrap_detector(GroundingDINOClient, "grounding_dino_raw")
    wrap_detector(YOLOv7Client, "yolov7_raw")

    original_segment = MobileSAMClient.segment_bbox

    def capture_segment(self, image, bbox):
        result = original_segment(self, image, bbox)
        get_evidence_recorder().record_model_event(
            "mobile_sam", {"bbox": bbox}, {"shape": list(result.shape)}, image=image, array=result
        )
        return result

    MobileSAMClient.segment_bbox = capture_segment

    original_ask = BLIP2Client.ask

    def capture_ask(self, image, prompt=None):
        result = original_ask(self, image, prompt)
        get_evidence_recorder().record_model_event(
            "blip2_vqa", {"prompt": prompt}, result, image=image
        )
        return result

    BLIP2Client.ask = capture_ask

    original_cosine = BLIP2ITMClient.cosine

    def capture_cosine(self, image, txt):
        result = original_cosine(self, image, txt)
        get_evidence_recorder().record_model_event(
            "blip2_itm", {"text": txt}, result, image=image
        )
        return result

    BLIP2ITMClient.cosine = capture_cosine
    os.chdir(root)
    sys.argv = [str(root / "vlfm" / "run.py"), *hydra_args]
    try:
        runpy.run_path(str(root / "vlfm" / "run.py"), run_name="__main__")
    finally:
        get_evidence_recorder().close_videos()
        overlay.cleanup()


if __name__ == "__main__":
    main()
