# SAP-Nav category-only reproduction on HM3Dv1 ObjectNav

Requested benchmark: **HM3Dv1 ObjectNav val**, category-name goals only.
HM3D-OVON and LangMap are not evaluated. No room/region/attribute instruction
parser or QSSR spatial-constraint gate is needed for this task adaptation.

## Source and scope

- Paper: https://arxiv.org/html/2608.12707v1
- Author repository: https://github.com/XuetongPei/SAP-Nav
- Checked 2026-10-04: author repository contains the project page and says
  "Code will be released soon." This is an independent paper-based implementation,
  not a reproduction of released author code or a claim of matching paper scores.
- VLFM stays pinned at `584ed56008754fde7997d904983607def8328322`.
  All new behavior is in `av_nav/sap_*`; original upstream files are untouched.
- Off-the-shelf models and pretrained VLFM PointNav are used without ObjectNav
  task-specific training. Zero-shot does not mean all component networks are untrained.

## Implemented method

The adapter retains VLFM exploration and category detection. Candidate crop and
full RGB image are passed to a VLM for visibility and perspective scores, each
in [1,5]. Insufficient views trigger viewpoint sampling at 24 angles on radii
0.8/1.2/1.6/2.0/2.4 m. Candidates must be reachable in explored free space and
satisfy the vertical-FOV distance constraint. An online maximum-height map
scores ray visibility to observed footprint cells. The highest visibility
view is selected; after at most three repositioning attempts, the best scored
observation is used for category verification. Rejected spatial clusters are
blacklisted for the episode. Only verified candidates may issue a target STOP.

## Explicit reconstruction choices / deviations

- Category-only adaptation omits QSSR, whose specified role is evaluating
  spatial constraints absent from a bare category goal. This is not the complete
  hierarchical SAP-Nav system.
- Existing VLFM YOLOv7/GroundingDINO and MobileSAM are retained. The exact
  detector/segmenter versions used by the authors are not available as code.
- Configured VLM is local Qwen3.5-9B with thinking disabled; paper HM3D-OVON
  results use GPT-4o and paper Qwen3.5 experiments enable thinking. Model choice
  and prompting therefore differ and must accompany every result.
- Exact prompts and sufficiency threshold are not supplied in the paper text.
  Reconstructed prompts are versioned in `sap_vlm.py`; threshold 7 is an explicit
  uncalibrated initial choice, not a published paper parameter.
- Engineering limits: association radius 0.75 m, 60 actions per relocation,
  reachable-path search limit 20 m, arrival tolerance 0.25 m. Missing feasible
  views fall back to the best recorded observation. A missing re-detection is
  not scored as negative evidence. Stale unverified map targets cannot stop.
- Existing robust RGB-D candidate backprojection trims depth tails. Footprint
  height uses the maximum retained observed point per cell, not ground truth.
- Unknown space is occluding unless already mapped as free. Equal visibility
  scores resolve by fixed ring/angle enumeration, without motion-cost scoring.

## Evaluation protocol

Run training smoke tests before final val. Freeze code/config/model before val;
do not tune on final val. Full runs use 500 steps and the inherited HM3Dv1
Habitat success measure. Do not import the paper's different benchmark success
threshold. Both variants use the same pinned VLFM, sensors, episodes and services.

`scripts/run_sap_eval.py` requires `.../hm3d/v1`, rejects unexpected categories,
and archives source hashes, exact source IDs, frozen config, commits, environment,
commands, per-episode metrics and verification events. Existing output directories
are rejected. A run is complete only on zero process exit and an exact unique
episode-set match. Shortened/limited runs are diagnostic only. VLM errors fail
the run instead of silently accepting candidates.

```bash
/home/zyq/miniconda3/envs/vlfm/bin/python scripts/run_sap_eval.py \
  --dataset-root /home/zyq/vlfm/data/datasets/objectnav/hm3d/v1 \
  --scenes-dir /home/zyq/vlfm/data/scene_datasets \
  --output /home/zyq/AV-Nav/runs/UNIQUE_RUN_NAME --gpu 1 --variant sap
```

Use a dedicated fixed-commit worktree for each long evaluation. All RGB evidence
and run files remain untracked. Compare SR, SPL, soft-SPL, per-episode success,
VLM calls, repositions and zero-trigger frequency against `--variant baseline`.
No efficacy result is claimed before a complete evaluation.
