# Forced-review diagnostic

This single training episode used high=1, low=-1, a 200-action task cap,
48 verification actions per attempt, and a 5 m verification path budget.
It is deliberately excluded from efficacy comparisons.

Observed: no-feasible-view fallback, a reachable viewpoint selected at step 87,
new visual evidence sampled at step 108 and fused at step 113, then a missing
view followed by unresolved fallback at step 132. That completed attempt used
46 actions and 3.25 m actual travel. A later attempt was interrupted by the task
step limit; check av_active_at_step in the final episode record.

The source episode ID is 67; runtime_episode_id is 0. Raw evidence crops are
retained under runs/diagnostic_force_review_plant1_v2/evidence on the server and
are not committed. Successful process completion does not mean task success or
an improvement in navigation accuracy.
