# SAP thinking exhaustion recovery

Runtime commit: `2c7b7e5`. Qwen3.5-9B thinking remains enabled on the existing
GPU 2 service; the navigation evaluator uses GPU 3.

The previous suite stopped after 130 distinct episodes: 99 in `4ok3usBNeis`
and 31 in `5cdEh9F2hJL`. The second scene's next episode repeatedly produced
the same reasoning paragraph. Every completion budget (8192, 10240, 12288)
ended with `finish_reason=length` and no final content. Completed prefixes
remain intact and are imported with their original commit provenance.

Recovery:

1. Execute the normal multimodal thinking request.
2. On length exhaustion with reasoning but no final content, retain the actual
   reasoning and all original images/user text, close `</think>` explicitly,
   and prefill `{` for a final-answer continuation.
3. Generate at most 512 additional tokens with `continue_final_message=True`,
   `add_generation_prompt=False`, and `enable_thinking=True`.
4. Decode and validate the final JSON using the existing strict schema. The
   initial thinking trace is never treated as an answer. Invalid final answers
   still cause a bounded retry and ultimately fail rather than bypassing review.

The installed vLLM 0.18.1 has no native `thinking_token_budget` parameter.
Its non-streaming Qwen parser reports final continuation text in the `reasoning`
field even when the prompt has closed the thought block. The adapter handles
that field only for this explicitly prefixed final phase. The original thought
trace, final response, both usage records and failed calls are archived.

This adapts Qwen's documented two-stage budget mechanism to a multimodal chat
endpoint: https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md

Validation:

- Local suite: 71 tests passed, with the existing OpenCV-dependent test skipped.
- Server suite: all 71 tests passed.
- A synthetic blank-image diagnostic with a 16-token initial limit produced
  an invalid string boolean in the final phase. Strict validation rejected it;
  the normal retry recovered, using three API calls. The diagnostic's assertion
  for a two-call continuation failed. This is not an efficacy experiment.
- A synthetic blank-image diagnostic with a 128-token initial limit exercised
  actual thinking exhaustion and successfully produced boolean false in the
  final phase with two API calls and 535 characters of recorded reasoning.
- Habitat training smoke: one episode, 100-step cap, successful zero-exit
  completion, success 1, SPL 0.6593985, two VLM calls and zero repositions.
  This shortened episode is a diagnostic, not an efficacy result.

No navigation threshold, target prompt, model weights, task categories, or
validation success criterion is changed. The inference continuation protocol
and actual HTTP call accounting are disclosed changes. Inherited old rows
counted logical verifier calls; new rows count HTTP calls, including recovery.

Dedicated worktree:
`/home/zyq/AV-Nav-worktrees/sap-hm3dv1-thinking-2c7b7e5`.
Parent suite:
`/home/zyq/AV-Nav/runs/sap_hm3dv1_val_thinking_resume_b747e29_20261005`.
New suite:
`/home/zyq/AV-Nav/runs/sap_hm3dv1_val_thinking_budget_2c7b7e5_20261005`.
Resume manifests confirm 99 inherited and zero pending in the first scene;
31 inherited and 68 pending in the second scene. Session
`sap_hm3dv1_thinking` uses the fixed runtime commit. The original VLFM
evaluation remains separate on GPU 1.

At 20:12 Singapore time, the actual previously failing request in scene
`5cdEh9F2hJL`, source row 31, completed through thinking-budget recovery.
The final validated scores were visibility 2 and perspective 2. The
`sufficiency` event records the continuation response; subsequent viewpoint
selection and navigation advanced from step 103 to step 109. This verifies
recovery past the observed stalled request. The episode is still in progress;
the retained count remains 130 completed distinct episodes, 97 successes.
