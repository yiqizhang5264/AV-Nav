# BeliefMapNav Qwen yes/no parser repair (2026-11-09)

## Incident

The single-worker HM3Dv1 evaluation
`hm3dv1_full_single_gpu2_dino_qwen35_novideo_20261109` stopped in the tenth
episode. Nine episodes had complete `episode_stats` records. The tenth episode
returned a Qwen response whose last non-empty line was `No`, but the upstream
parser accepted only an explicit `Answer: yes/no` label and raised
`ValueError`.

This is a runtime compatibility repair, not an efficacy change or validation
set parameter tuning. The nine completed episodes remain in the failed run and
were not overwritten.

## Repair

- Adapter: `adapters/beliefmapnav/official_c9800d0_vqa_answer_parser.patch`
- Adapter commit: `ca97a0ef8a3d5a07b6e1836db1d967b61e8376d3`
- Adapter SHA256 on the server: `1cfa477ca00f0126bb14fa0459af00f1bc6e261508a217b43d1bc893ee633f66`
- BeliefMapNav upstream snapshot: `c9800d09c2e0f0203a0efbee976d421afade7456`

The parser first accepts an explicit `Answer: yes/no` label. If that label is
absent, it accepts only a final non-empty line consisting solely of `Yes` or
`No`, with optional Markdown emphasis and terminal punctuation. It does not
search for unanchored yes/no words inside reasoning text. Both duplicated VQA
verification branches call the same parser.

## Deployment and verification

The adapter was applied to the dedicated server snapshot at
`/home/zyq/BeliefMapNav-repro/c9800d0`. The active `/home/zyq/AV-Nav` checkout
was not updated because the Qwen and GroundingDINO services were running from
it. The adapter was fetched into the detached worktree
`/home/zyq/AV-Nav-worktrees/bmn-vqa-parser-d931282` instead.

- Original server source SHA256: `46b91989a08ad81f91d60759dde3a3ce5b45ca2755ad734cef546df0ff9738b5`
- Repaired server source SHA256: `26be5f865f16cf797a6b1049daf3d4c7fe030ec7d7d224da39fb123691db9112`
- Original backup: `vlfm/policy/base_objectnav_policy.py.pre_vqa_parser_ca97a0e`
- Local full test suite: 81 tests passed, 1 environment-dependent test skipped
- Server syntax compilation: passed with
  `/home/zyq/miniconda3/envs/vlfm/bin/python`
- Server behavior checks: explicit answer accepted, final-line `No` accepted,
  and an ambiguous reasoning-only response rejected

Evaluation was not restarted immediately after deployment. Physical GPU 2 was
occupied by another user's process (PID 2248090, approximately 11.6 GiB and
100% CPU at inspection time). Restarting Habitat there would have caused
resource contention. A future resume must use a new run directory, preserve
the original 0--8 episode records, and restart from episode index 9.

## Follow-up repair (2026-11-10)

The first parser adapter still required an explicit `Answer:` label to begin a
line. GPU 0 completed episode indices 9--33, then stopped while processing index
34 when Qwen returned an otherwise unambiguous inline label ending in
`Answer: yes`. This exposed a gap in the first adapter rather than a model or
HTTP service failure.

The v2 repair accepts an explicit `Answer: yes/no` anywhere in the response and
uses the last explicit label if more than one is present. A final standalone
`Yes` or `No` remains supported. If no unambiguous answer is parseable, both VQA
call sites now log the condition and conservatively treat the detection as
`no`, rather than terminating the episode and worker.

- Final adapter source commit: `fa7d56bc56743765426f63ad0ff2d70e46cfbe24`
- Incremental adapter:
  `adapters/beliefmapnav/official_c9800d0_vqa_answer_parser_v2.patch`
- Dedicated GPU 0 code root:
  `/home/zyq/BeliefMapNav-repro/c9800d0-gpu0-vqa-v2-fa7d56b`
- Repaired policy SHA256:
  `c2e2e89429adff7d9dff88646612b09a66606cde574fae798d9b31e24f3d7a0d`
- Pre-v2 policy SHA256:
  `26be5f865f16cf797a6b1049daf3d4c7fe030ec7d7d224da39fb123691db9112`
- Local verification: 82 tests passed, 1 environment-dependent test skipped
- Server verification: syntax compilation passed; the actual inline failure
  response parsed as `yes`; multiple labels selected the last answer; ambiguous
  reasoning still raised inside the parser and was covered by the nonfatal
  caller fallback

GPU 2 and GPU 3 continued using the unchanged shared snapshot while this repair
was deployed. GPU 0 recovery uses a new run directory and starts at index 34;
the 25 completed indices 9--33 remain in the failed attempt. At queue time GPU 0
was occupied by a separate root-owned audit process, so the guarded monitor was
left waiting rather than preempting it.
