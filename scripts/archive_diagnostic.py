"""Archive a completed forced-trigger engineering check separately from benchmarks."""
import json
import shutil
from pathlib import Path

root=Path(__file__).resolve().parents[1]
name='diagnostic_force_review_plant1_v2'
run=root/'runs'/name
if json.loads((run/'exit.json').read_text())['returncode']!=0:
    raise SystemExit('Diagnostic has not completed successfully')
out=root/'results/diagnostic_20260907'
out.mkdir(parents=True,exist_ok=True)
for filename in ['manifest.json','episodes.jsonl','verification.jsonl','exit.json']:
    shutil.copyfile(run/filename,out/filename)
events=[json.loads(line) for line in (run/'verification.jsonl').read_text().splitlines()]
summary={kind:sum(e['event']==kind for e in events) for kind in ['trigger','no_feasible_view','view_selected','evidence','fusion','target_missing','finish']}
summary['purpose']='Forced-review engineering check only; not efficacy evidence'
summary['observed_complete_chain']=all(summary[k]>0 for k in ['view_selected','evidence','fusion','finish'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'README.md').write_text('''# Forced-review diagnostic

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
''',encoding='utf-8')
print(json.dumps(summary,indent=2))
