# STRIVE Qwen3.5-9B HM3Dv2 aggregation (2026-10-03)

The launcher stopped after producing valid metrics for 988 of the expected 1000 episodes. This directory therefore records an incomplete aggregation, not a final 1000-episode score.

- Valid episodes: 988/1000
- Successes/failures: 711/277
- Observed SR: 71.96%
- Observed SPL: 31.81%
- Missing-as-failure SR: 71.10%
- Missing-as-zero SPL: 31.43%
- Best possible final SR if all 12 missing episodes succeed: 72.30%
- Missing: 817-821 and 987-993

Episodes 817-821 are blocked by the deterministic Open3D CUDA OOM at the initial point-cloud downsample. Episodes 987-993 are missing because episode 987 exceeded the Qwen 16,384-token context window by one token on the retry request, aborting the shard after episode 986.

The `Found Goal` flag is only a coarse diagnostic. Among the 277 failed episodes, 126 ended with `Found Goal=True` and 151 with `Found Goal=False`; the former must not be reported as confirmed false positives without video or trajectory review.
