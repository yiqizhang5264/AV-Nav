# HM3Dv2 resume incident: Qwen context window

## Failure

- The `59db4e0` run completed episodes 97-101 and continued through multiple full shards between 132 and 221.
- `shard_0182_0191` wrote metrics for episodes 182-184, then failed deterministically on episode 185 in all three attempts.
- At step 451, the room-selection prompt contained at least 15,361 input tokens. Requesting 1,024 completion tokens exceeded the local Qwen service's 16,384-token context window and returned HTTP 400.

## Repair

- Commit: `c08d58148626aff659ddb1e21a5210df9c1aebf0`
- Only an OpenAI-compatible context-window rejection triggers the new recovery path.
- Recovery retries once with a 512-token limit, a final-fields-only response schema, deterministic temperature, and a concise-output instruction.
- Normal VLM requests retain the existing 1,024-token completion limit.
- The pinned STRIVE submodule remains unchanged at `1872d73b7db297705d251df73bf5f08ffed0d749`.
- All 38 unit tests passed locally and on the server.

## Resume

The metrics union across the earlier fixed worktrees contains 215 unique completed episodes. The exact missing ranges are:

```text
185-191
222-999
```

Episode 185 runs first as the repair validation, followed by 186-191 and 222-999. The new fixed worktree is `/home/zyq/AV-Nav-worktrees/strive-full-c08d581`; launcher PID at start is `787147`.
