# HM3Dv2 resume incident: fully explored with no waypoint

## Failure

- The `c1f9e3f` resume completed episodes 120, 121, 45, and 61.
- The next suite stopped at episode 97 after three deterministic failures.
- Initial planning correctly reported that the scene was fully explored and no unvisited node remained, but the upstream benchmark ignored the returned `False` flag and called `step_mod()` with `self.waypoint = None`.
- STRIVE then raised `TypeError` while adding that missing waypoint to the mapper origin.

## Repair and validation

- Commit: `59db4e0d6b2126c0c37b0cfdf1de182168228d65`
- The runtime adapter now ends an episode cleanly when no waypoint exists. The episode is still recorded as a navigation failure, preserving the evaluation semantics.
- The pinned STRIVE submodule remains unchanged at `1872d73b7db297705d251df73bf5f08ffed0d749`.
- All 37 unit tests passed locally and on the server.
- Fixed server worktree: `/home/zyq/AV-Nav-worktrees/strive-full-59db4e0`
- Episode 97, which reproduced the crash in every prior attempt, completed on the first repaired attempt with return code 0 and a written metrics row.

## Resume

The new fixed-checkout launcher evaluates only the remaining ranges:

```text
97
98-101
132-999
```

Episode 97 is isolated as the repair validation. After it completes, the launcher automatically continues with 98-101 and then 132-999. Launcher PID at start: `3518168`.
