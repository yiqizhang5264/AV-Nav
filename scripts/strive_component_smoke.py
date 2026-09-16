#!/usr/bin/env python3
"""Load and execute STRIVE's GroundingDINO and SAM components on a blank frame."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "strive"))

from cv_utils.sam import MMDINO_Grounded_SAM  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    model = MMDINO_Grounded_SAM(["tv", "mirror"], device="cuda")
    detections = model._run_grounding(image, "tv . mirror .")
    model.sam_predictor.set_image(image)
    embedding = model.sam_predictor.get_image_embedding()
    print("grounding_dino_boxes", len(detections["bboxes"]))
    print("sam_embedding_shape", tuple(embedding.shape))
    print("cuda_peak_memory_bytes", torch.cuda.max_memory_allocated())


if __name__ == "__main__":
    main()
