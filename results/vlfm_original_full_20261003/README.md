# Original VLFM full-evidence evaluation

This run evaluates the unmodified upstream VLFM code at commit
`584ed56008754fde7997d904983607def8328322`.  Evidence hooks are supplied by
AV-Nav commit `1224d7e` through a temporary package overlay; the pinned VLFM
submodule is not edited.

Server locations:

- Fixed worktree: `/home/zyq/AV-Nav-worktrees/vlfm-original-1224d7e`
- Full results: `/home/zyq/vlfm_evidence_runs/original_vlfm_584ed56_avnav_1224d7e_20261003`
- Equivalence check: `/home/zyq/vlfm_evidence_runs/equivalence_1224d7e`
- Model services: tmux session `vlm_servers_20141`, physical GPU 0
- Evaluation queue: tmux session `vlfm_full_original`, physical GPU 1
- MP3D asset download: tmux session `mp3d_assets_download`

The full-configuration equivalence check used HM3Dv1 scene `4ok3usBNeis`,
episode `0057`, seed 100, one environment, and official disk-video rendering.
Both original and instrumented runs completed 69 steps, succeeded, and produced
the same action trace SHA-256:
`06280d44a7464a23a12e9af3c0446c3347b79ba078e08bda426b8bb497762a07`.
Reward, success, SPL, soft-SPL, distance, detection, and stop metrics also match.

HM3Dv1 and HM3Dv2 smoke evaluations passed.  The first MP3D smoke evaluation
was rejected before any episode result because the existing scene folders held
only render GLBs and lacked the semantic assets required by ObjectNav metrics.
The incomplete official `mp3d_habitat.zip` download is being resumed; the queue
will test the archive, extract it to a new directory, hash the assets, and only
then start MP3D.

The queue runs HM3Dv1, HM3Dv2, then MP3D.  Each scene is an independent shard
with up to three non-overwriting attempts.  A shard is complete only when the
original trainer writes its `DONE` marker.  Raw evidence remains untracked on
the server.
