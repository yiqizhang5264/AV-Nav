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
- Per user instruction, configured VLM is local Qwen3.5-9B with thinking enabled
  and an 8192-token completion budget. Paper HM3D-OVON results use GPT-4o.
  Model choice and reconstructed prompting must accompany every result.
  Qwen uses the existing isolated `strive-qwen` server environment; Habitat and
  VLFM use `/home/zyq/miniconda3/envs/vlfm/bin/python`.
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
- The height map keeps observed surfaces at or below camera height, avoiding
  projecting ceilings into solid floor-to-ceiling obstacle columns. This is an
  explicit reconstruction choice aligned with VLFM's obstacle-height band;
  overhangs and above-camera geometry are not fully modeled by this 2.5D map.

## Evaluation protocol

Run training smoke tests before final val. Freeze code/config/model before val;
do not tune on final val. Full runs use 500 steps and the inherited HM3Dv1
Habitat success measure. Do not import the paper's different benchmark success
threshold. Both variants use the same pinned VLFM, sensors, episodes and services.

`scripts/run_sap_eval.py` requires `.../hm3d/v1`, rejects unexpected categories,
and archives source hashes, original IDs plus unique source-hash/row identities
(HM3Dv1 repeats original IDs within scenes), frozen config, commits, environment,
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

## Engineering repair on 2026-10-05

Empty or malformed final VLM content now retries up to three calls with thinking
still enabled. Completion budgets are 8192, 10240 and 12288 tokens; each failed
response is archived in `vlm_errors.jsonl`. Exhaustion still fails the attempt;
it never automatically approves an unverified candidate. This changes the
inference protocol and must be disclosed with results; category prompts and
navigation thresholds are unchanged.
Explanatory prose or Markdown surrounding exactly one complete JSON object is
accepted after the same strict schema validation; ambiguous multiple objects
are rejected. A captured failure showed a valid scored object preceded by prose,
which the previous parser rejected.

`--resume-from` explicitly imports a previous suite's completed deterministic
prefix. Source rows, source and episode hashes, configuration and benchmark must
match. Each attempt writes a new directory and resumes only remaining episodes;
each row retains its originating commit and run path. This is an engineering
continuation across commits, not a homogeneous fixed-commit rerun. Global RNG
state is restarted, so resumed trajectories are not claimed to be identical to
an uninterrupted run. Final comparison must disclose this lineage. Old failed
attempts and their evidence remain intact.

## Thinking exhaustion repair, 2026-10-05 evening

The second scene stalled on an actual repeated reasoning loop. Raising the
completion budget to 12288 still produced no final content. When a request ends
with `finish_reason=length` and nonempty reasoning but no final answer, the
verifier now continues the same multimodal context with the actual reasoning,
an explicit closing `</think>` marker and a JSON opening brace. A separate
512-token budget is reserved for the final answer. `enable_thinking=True`
remains set; no answer is extracted from the initial thought trace.

This follows Qwen's two-stage thinking-budget approach, adapted to the existing
multimodal chat endpoint. vLLM 0.18.1 mislabels the resulting final continuation
as reasoning in non-streaming responses; only in this explicitly closed final
phase is that field decoded as the answer, with the original strict schema
validation. Both response stages and usage are archived. `sap_vlm_calls` now
counts actual HTTP inference requests, including retries and continuations;
older inherited metrics counted logical verifier calls, a disclosed difference.

This is a frozen engineering protocol change, not navigation parameter tuning
on validation. Thresholds, target prompts, models and thinking mode stay fixed.
Reference: https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md
