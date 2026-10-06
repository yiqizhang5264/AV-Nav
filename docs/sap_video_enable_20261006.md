# SAP video continuation, 2026-10-06

User requested stopping after the current episode, enabling video saving, and
continuing the remaining validation episodes.

The old evaluator had completed 97 episodes in `DYehNKdT76V` when the boundary
watch was armed. An inotify close-write watcher waited for episode 98 to be
durably recorded, paused the isolated SAP process group, then interrupted it.
The receipt timestamp is 11:39:23 Singapore time. The last completed source row
is 97; cumulative unique completion count is 395 (three scenes of 99 plus 98).
No unfinished episode is counted. The old code checkout was never modified.
Receipt: `sap_hm3dv1_val_thinking_budget_2c7b7e5_20261005/video_boundary_stop_20261006.json`.

Runtime commit: `e13dd1e`. Dedicated worktree:
`/home/zyq/AV-Nav-worktrees/sap-hm3dv1-video-e13dd1e`.
Pinned VLFM stays `584ed56008754fde7997d904983607def8328322`.

`--save-video` enables Habitat's disk MP4 output. Each new episode has an RGB,
depth and map visualization. `videos.jsonl` maps the MP4's runtime episode ID
back to the original scene, source row, source and episode hashes. It also
records frame count and FPS. Metrics are committed after successful encoding,
so retries cannot silently skip an episode whose video failed to save.

A display-only adapter bounds projected point-cloud pixels to the actual
rendered map dimensions, preventing the known upstream map-edge IndexError.
Navigation maps and coordinates are not clipped. Upstream source is untouched.

The new suite imports the old completed prefixes and executes only remaining
episodes; old rows retain their original provenance and do not acquire videos
retroactively. Videos/raw imagery stay on the server and untracked. This remains
an engineering continuation across commits; global RNG state restarts at resume.

Server unit suite: all 72 tests passed. The video integration smoke is performed
on one training episode with a 100-step cap and is diagnostic only.
The smoke completed successfully (success 1, SPL 0.6593985). Its MP4 is
1,053,895 bytes with 87 frames, 10 FPS, and 1520 by 960 resolution; the first
and last frames were both decoded using OpenCV. The source identity index was
verified against source row 0. This is not a validation efficacy result.

Resumed output:
`/home/zyq/AV-Nav/runs/sap_hm3dv1_val_video_e13dd1e_20261006`.
Videos: `<scene>/attempt_XX/videos/*.mp4`.
Index: `<scene>/attempt_XX/videos.jsonl`.
SAP remains on GPU 3, Qwen3.5-9B thinking on GPU 2. The original VLFM evaluation
remains on GPU 1.
