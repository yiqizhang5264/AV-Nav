# VLFM MP3D evaluation environment repair — 2026-10-08

The original VLFM evaluation completed HM3Dv1 and HM3Dv2, then failed on
MP3D ObjectNav val. MP3D attempts 002–008 stopped at the initial observation
of episode 22 (target `cabinet`); only episode 149 had a completed result.
These failed attempts are retained and are not counted as completed episodes.

## Cause

GroundingDINO on GPU 0, port 12181, returned HTTP 500. Its PyTorch
`1.12.1+cu113` bundled NVRTC 11.3 rejected the RTX 6000 Ada GPU architecture:
`nvrtc: error: invalid value for --gpu-architecture (-arch)`.
The failing operation was `spatial_shapes.prod(1)` in GroundingDINO's
transformer. Increasing the client timeout would not fix this compiler error.
GroundingDINO is used for this MP3D target, whereas the six HM3D target
categories use YOLO in the original VLFM policy.

## Repair

Only the GroundingDINO service was restarted. Its original source, weights,
thresholds, Python environment and GPU assignment were preserved. An isolated
directory prepended to `LD_LIBRARY_PATH` supplies the existing CUDA 11.8
NVRTC libraries from `/home/zyq/miniconda3/envs/apexnav/lib`:

- `libnvrtc-1ea278b5.so.11.2` and `libnvrtc.so.11.2` link to that environment's
  `libnvrtc.so.11.2` (verified by `nvrtcVersion` as 11.8).
- `libnvrtc-builtins.so.11.8` links to the matching builtins library.

No shared environment libraries or pinned upstream files were replaced.
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` load the existing cached BERT
resources without unreachable Hugging Face network requests.

Service source remains
`/home/zyq/AV-Nav-worktrees/vlfm-original-95df3f9/external/vlfm`.
Service Python remains `/home/zyq/miniconda3/envs/vlfm/bin/python`.
The repaired service runs in tmux `vlfm_gdino_nvrtc_repaired` on GPU 0.
Other model services and the SAP evaluation were left running.

The repair artifacts, exact launch commands, library/config/weight hashes,
pip freeze, old traceback, CUDA probe and HTTP probe are retained untracked at:

`/home/zyq/AV-Nav/runs/vlfm_gdino_nvrtc_repair_20261008`.

## Validation and resumption

A fresh CUDA subprocess correctly evaluated integer shape products to
`[6400, 1600]` with the isolated NVRTC 11.8 loader override. Three requests
using the actual previously failing cabinet observation returned HTTP 200
and the same cabinet detection. The first warm-up request took 3.393 seconds;
subsequent requests took 0.153 and 0.132 seconds.

The original MP3D suite resumed in tmux `vlfm_full_original_resume` on GPU 1
at its unchanged fixed adapter commit
`5596d7a6059569d0ddadbc62c624589a0ea3bd26`, with upstream VLFM pinned to
`584ed56008754fde7997d904983607def8328322`. Attempt 009 skips completed
episode 149 and requests the remaining 199 episodes in the first scene.
The maximum attempt index was extended to 16 because attempts 001–008 already
exist. It is the same original run, with no duplicate evaluator launched.

The original dataset hashes and run manifest remain intact. The first scene
content hash recorded in the resume plan is
`7fa5fbe963fc36e14db59592ca1579b6d0844b69ca2e2452b549c47b2201d780`.
Full per-scene hashes and evaluation configuration remain in the run manifest
and each attempt's command/Hydra configuration under:

`/home/zyq/vlfm_evidence_runs/original_vlfm_584ed56_avnav_1224d7e_20261003/mp3d`.

This is an environment compatibility repair, not a change to VLFM's navigation
algorithm, checkpoint or detector thresholds. Successful service probes are
integration checks, not navigation efficacy results.

At 14:20:44 server time, the resumed episode 22 had advanced past step 30,
reported `target_object=cabinet` and `target_detected=true`, and continued
receiving HTTP 200 GroundingDINO responses. This verifies progress past the
original initial-observation failure; it does not imply the full suite has
finished.
