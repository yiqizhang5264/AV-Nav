# SAP category navigation: HM3Dv1 val execution record

User scope: zero-shot category-name ObjectNav on **HM3Dv1 val only**.
VLM: **Qwen3.5-9B with thinking enabled**. No HM3D-OVON evaluation.
This is a paper-based independent reconstruction, since author code is not released.
Method and deviations: [reproduction protocol](sap_category_reproduction.md).

## Frozen implementation and environment

- SAP code: `5cf1d2f`; pinned VLFM: `584ed56008754fde7997d904983607def8328322`.
- SAP worktree: `/home/zyq/AV-Nav-worktrees/sap-hm3dv1-thinking-final`.
- Baseline code: `b23ab56`, worktree `/home/zyq/AV-Nav-worktrees/sap-hm3dv1-sourceids`.
  Subsequent SAP-only geometry/diagnostic changes do not alter baseline policy.
- Habitat Python: `/home/zyq/miniconda3/envs/vlfm/bin/python`.
- Qwen service uses the existing `strive-qwen` environment, vLLM 0.18.1,
  bfloat16, `--reasoning-parser qwen3`, context 16384, completion limit 8192,
  explicit `enable_thinking=true`. Model snapshot:
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Qwen command, environment and all model file hashes are archived at
  `/home/zyq/AV-Nav/runs/sap_qwen_thinking_20261004`; tmux `sap_qwen_thinking`, GPU 2.
- Existing VLFM perception services: GPU 0, ports 12181--12184.
- Latest server unit tests: 60 passed, log
  `/home/zyq/AV-Nav/runs/sap_server_tests_final.txt`.

## Dataset integrity

The source is `/home/zyq/vlfm/data/datasets/objectnav/hm3d/v1/val`:
20 content shards, 2000 episodes, bed/chair/plant/sofa/toilet/tv_monitor.
Original IDs are heavily repeated: there are only 27 distinct scene/original-ID
pairs. Every source row is retained; identity is source-file SHA256 plus row index.
Per-episode records preserve original IDs, row indices, source hashes and episode
hashes. This prevents accidental deduplication and supports exact paired comparisons.

## Completed engineering checks (not efficacy results)

- `sap_baseline_train_smoke_03`: one training episode, 100-step limit; completed,
  success 1, SPL 0.6593985.
- `sap_thinking_train_smoke_01`: same training episode, 200-step limit; completed,
  success 1, SPL 0.6593985. Two VLM calls, **zero active triggers/repositions**.
  Both VLM responses contained nonempty reasoning (4769 and 643 characters).
- `sap_identity_train_smoke_04`: two training rows sharing original ID, 15-step
  limit; both recorded independently, successful process exit. Both timed out
  without success; this validates identity plumbing only.
- `sap_forced_train_diagnostic_01` and `_02`: forced threshold 11; zero feasible
  sampled views at first detection, fallback verification worked. `_02` completed
  two episodes, with no repositioning. Not an active-perception efficacy result.
- `sap_near_forced_train_diagnostic_03`: separate forced diagnostic additionally
  defers review until within 2 m. Observed 12 feasible views at step 80, selected
  a 1.31066 m path, reached a new view and re-scored at step 89. A subsequent
  missing detection triggered another view rather than a false negative.
  Completed with zero process exit, 3 reposition attempts, 1 trigger and 3 VLM
  calls. At step 114, the attempt limit caused verification from the best
  observation (score 10). The episode succeeded, diagnostic SPL 0.5651323.

## Failed attempts retained

- `sap_baseline_train_smoke_a751cee_01`: zero episodes; multiprocessing entry
  lacked a main guard. Fixed with an import regression test.
- `sap_baseline_train_smoke_02`: simulator completed one episode, but strict
  completion check failed because Habitat expands scene paths. Fixed by
  canonicalizing scene paths; original failed summary was not overwritten.
- `sap_hm3dv1_val_baseline_038428b_launcher.log`: full-launch preflight rejected
  duplicate original IDs before evaluating episodes. Fixed using source-row identity.
- Initial general bootstrap failed on locally configured submodule URLs; only
  pinned VLFM was subsequently initialized with per-command local transport permission.

## Full evaluation

SAP suite started in tmux `sap_hm3dv1_thinking`, GPU 1, output:
`/home/zyq/AV-Nav/runs/sap_hm3dv1_val_thinking_5cf1d2f`.
After discovering the user's resumed original VLFM evaluator on physical GPU 1,
the SAP attempt using that same GPU was stopped and preserved as attempt 02.
SAP resumed as attempt 03 on physical GPU 3; the user's VLFM process remains on
GPU 1. The failed attempt 01 and interrupted attempt 02 remain archived.
The full VLFM baseline launched during this work duplicated a user-managed run.
It was stopped after two partially evaluated scene shards. The partial directory
`/home/zyq/AV-Nav/runs/sap_hm3dv1_val_baseline_b23ab56` is not a complete result.
At shutdown, no other active HM3Dv1 VLFM evaluator process was visible. The
archived original VLFM run currently contains 10/20 completed HM3Dv1 scene shards
and no active process, so it cannot yet provide a complete val comparison.
SAP uses the normal `configs/sap_category.json`, not the forced diagnostic config.
Each scene runs all its episodes at 500 steps. Three separate attempts maximum
per scene; failed attempts are preserved. A full aggregate is emitted only after
all 2000 unique source rows complete with successful process exits.

No final SR/SPL or improvement claim is available yet. Training smoke/forced
diagnostic results above must not appear in an efficacy table. Final validation
results are not used to tune parameters.

Read progress without changing running checkouts:

```bash
cd /home/zyq/AV-Nav-worktrees/sap-hm3dv1-thinking-final
/home/zyq/miniconda3/envs/vlfm/bin/python scripts/summarize_sap_progress.py \
  /home/zyq/AV-Nav/runs/sap_hm3dv1_val_thinking_5cf1d2f
```
