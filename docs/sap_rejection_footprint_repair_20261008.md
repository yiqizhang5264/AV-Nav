# SAP rejected-candidate footprint repair (2026-10-08)

The HM3Dv1 bed episode at source row 33 in `p53SfW6mjZe` repeatedly verified
an already rejected candidate. Fourteen observations had identical image
hashes; each cycle took 169 seconds. The selected navigation point was
0.752076 metres from the rejection center, outside the existing 0.75 metre
center-only exclusion disk. This is a candidate identity/exclusion defect,
not evidence for changing validation thresholds.

The explicit SAP adapter now records all associated observed XY point clouds,
the candidate center and navigation anchor on rejection. Subsequent object-map
selection excludes points within the existing association radius of this
footprint. A reobserved rejected candidate is excluded before scoring or
category verification, extends the footprint with its current observation,
and returns to exploration. Rejection state resets at each episode.

Qwen3.5-9B thinking, token budgets, detector thresholds, view planning and the
pinned upstream VLFM source remain unchanged. This does not fix Qwen's repeated
thinking on new observations; it prevents repeated calls for rejected ones.
Spatial association retains the existing radius, so nearby observations within
that radius may share the rejection footprint, as with the existing association
scheme. No radius was tuned from validation outcomes.

Regression tests cover the recorded goal outside the center disk, extended
candidate geometry, detector reinsertion, observations from multiple views,
preservation of a separate candidate and gaps between observed regions, empty
clouds, and episode reset. The full local unittest suite passed 75 tests with
one OpenCV-dependent test skipped pending the server environment.

Deployment uses a new fixed worktree and a new run directory, with video saving
enabled and the previous suite supplied through `--resume-from`. Completed
source identities and their original commit provenance are inherited. The old
incomplete episode is interrupted and restarted; it is not a completed failure
and is not added to efficacy metrics. The repair commit changes policy behavior,
so aggregated results across the resume boundary must retain their mixed-commit
provenance. Server regression/smoke checks are diagnostic checks, not efficacy
measurements. Existing VLFM evaluation and model services are left running.

## Server validation and deployment

The server ran all 75 tests successfully, including the OpenCV-dependent test.
The first Habitat smoke attempt failed before navigation because the new
worktree lacked its untracked `dummy_policy.pth` resource link. That failed
attempt was preserved; missing dataset, scene and dummy-checkpoint links were
added without changing the upstream checkout's tracked files.

The second smoke attempt replayed only the pending source-row-33 episode with
a 30-step cap, real Qwen thinking, and video saving. At step 12, category
verification rejected the candidate and saved a 510-point footprint. It reached
the 30-step limit with only two VLM API calls, one rejection and no further
verification of that candidate. It exited normally and encoded its video.
There were zero AVV reposition triggers in this smoke episode. Its shortened
unsuccessful episode and inherited rows are diagnostic data, not efficacy
results, and are not used to resume the formal evaluation.

The formal suite resumed on GPU 3 at fixed policy commit `dc53468` in
`/home/zyq/AV-Nav-worktrees/sap-hm3dv1-rejections-dc53468`, with videos enabled,
using the old video's suite as its source of completed prefixes:

`/home/zyq/AV-Nav/runs/sap_hm3dv1_val_video_e13dd1e_20261006`.

The new formal output is
`/home/zyq/AV-Nav/runs/sap_hm3dv1_val_video_rejections_dc53468_20261008`.
Source row 33 restarts from the old scene's durable prefix of 33 completed
episodes (rows 0–32). Previous metrics, evidence and videos remain in their
original directories. Commands, server test output, diagnostic attempts and
deployment metadata are stored untracked under
`/home/zyq/AV-Nav/runs/sap_rejection_checks_dc53468_20261008`.
