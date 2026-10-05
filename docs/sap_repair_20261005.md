# SAP evaluation repair and continuation, 2026-10-05

Runtime commit: `d0d8a2a293b8b087a29e1f637f7ff2aa1a442ffa`.
Pinned VLFM: `584ed56008754fde7997d904983607def8328322`.

The old suite's third attempt failed with `JSONDecodeError` after 48 completed
episodes, 43 successes. The failed request's raw output was not recorded by the
old code, so whether its final content was empty, truncated or malformed cannot
be determined from that traceback alone.

The repair retries invalid/empty VLM results and transport errors three times,
with completion budgets 8192, 10240 and 12288. Thinking stays enabled. Failed
responses are now archived. Exhausted retries still fail rather than approving
the candidate. Navigation scores, prompts and threshold are unchanged.

Completed episodes can be imported as an exact source-checked prefix. The new
attempt inherits the old third attempt's 48 rows and executes only the remaining
51 episodes in the first scene. Source/configuration mismatch is rejected. Each
row retains its original commit and output path. Global RNG state is restarted;
continuation is not claimed to reproduce uninterrupted stochastic trajectories.

Validation:

- Local unit suite: 68 tests, one OpenCV-dependent skip.
- Server unit suite: all 68 tests passed.
- First training smoke failed before evaluation because the fresh worktree lacked
  local weight links; this is not an efficacy result. Links were added without
  changing upstream source.
- Second training smoke completed one episode at a 100-step limit, success 1,
  SPL 0.6593985, two VLM calls, zero repositions. Diagnostic only.
- Live Qwen API check returned valid negative category JSON and nonempty reasoning.

Execution uses a new immutable worktree:
`/home/zyq/AV-Nav-worktrees/sap-hm3dv1-resume-d0d8a2a`.
New output:
`/home/zyq/AV-Nav/runs/sap_hm3dv1_val_thinking_resume_d0d8a2a_20261005`.
Old outputs remain intact at
`/home/zyq/AV-Nav/runs/sap_hm3dv1_val_thinking_5cf1d2f`.
Session `sap_hm3dv1_thinking`: SAP on physical GPU 3; Qwen thinking on GPU 2.
The existing VLFM evaluation continues on GPU 1. No baseline was started.

The result lineage contains two code commits and different VLM retry budgets;
report it as an engineering continuation, not a single-commit rerun. Full-val
performance remains unavailable until exact coverage of all 2000 episodes.
