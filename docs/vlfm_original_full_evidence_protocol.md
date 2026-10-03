# Original VLFM full-evidence evaluation protocol

The evaluated policy is the unmodified upstream VLFM submodule at commit `584ed56008754fde7997d904983607def8328322`.
The historical checkout `/home/zyq/vlfm` contains local policy and trainer edits and must not be used as evaluation code. It is used only as a source for immutable weights and dataset assets.

Evidence capture is passive: it copies observations, final filtered detections, actions, metrics and visualization products to disk and returns every upstream value unchanged. Before full evaluation, run the same one-episode seed with and without capture and compare action traces and metrics.

Validation sets:

- HM3Dv1: 20 scenes, 2000 episodes.
- HM3Dv2: 36 scenes, 1000 episodes.
- MP3D: 11 scenes, 2195 episodes.

Each scene is an independent shard. A shard is complete only when upstream writes its `DONE` marker. Existing run directories are immutable and must never be overwritten.

Per episode capture includes lossless RGB PNGs, float32 depth arrays, 16-bit depth previews, remaining sensor tensors, per-step actions/rewards/scalars, raw and final-filtered detection JSON, boxed frames, every detection crop, SAM masks, VQA calls, image-text similarity calls, top-down and policy map arrays, raw and boxed MP4 videos, the normal VLFM combined video, final metrics, failure classification and console logs.
