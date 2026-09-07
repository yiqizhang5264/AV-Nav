# AV-Nav development rules

- Develop code locally in this repository. Run Habitat and GPU inference on the server in `/home/zyq/AV-Nav` using `/home/zyq/miniconda3/envs/vlfm/bin/python`.
- The research order is VLFM first, DSVR-Nav integration after demonstrated VLFM efficacy. Do not silently change the main backbone.
- Keep `external/vlfm` pinned and preserve upstream behavior for the baseline. Add explicit adapters rather than mixing untracked changes from `/home/zyq/vlfm`.
- Exchange code and small result summaries through `origin`. Use normal pushes and fast-forward pulls. Never force-push, reset, or auto-stash another machine's work.
- Do not update a code checkout while its experiment is running. Long evaluations need a dedicated worktree at a fixed commit.
- Never commit passwords, tokens, private keys, datasets, weights, videos, or raw evidence images. `runs/` and local resource links stay untracked.
- Record exact configurations, commits, dataset hashes, environment and per-episode metrics. Do not overwrite a run directory.
- Calibrate on training scenes and keep development scenes separate. Never tune using final validation results.
- Diagnostic forced-review configurations and shortened smoke episodes are not efficacy results. State zero-trigger findings and failed runs explicitly.
- Run `python -m unittest discover -s tests -v` for changes to geometry, planning, split identity or metrics. Use server smoke tests for Habitat integration changes.
- The Word plan is generated from `docs/experiment_plan.md`; when changing it, preserve the original backup and verify rendered pages before replacement.
