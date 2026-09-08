"""Read-only summary of a running/completed suite, with missing and failed cells."""
import argparse,collections,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--suite-id',required=True);a=p.parse_args()
jobs=a.root/'runs'/a.suite_id/'jobs.jsonl'
records=[json.loads(s) for s in jobs.read_text().splitlines()] if jobs.exists() else []
rows=[]
for job in records:
    r=a.root/'runs'/job['run_id'];row=job.copy()
    if job['returncode']!=0:
        row['status']='failed';rows.append(row);continue
    episodes=[json.loads(s) for s in (r/'episodes.jsonl').read_text().splitlines()]
    traces=[json.loads(s) for s in (r/'trace.jsonl').read_text().splitlines()]
    steps=[t for t in traces if t['event']=='step']
    assert len(episodes)==1
    e=episodes[0];m=e['metrics']
    assert all(t['episode_id']==e['episode_id'] and e['scene_id'].endswith(t['scene_id']) for t in traces)
    assert [t['step_completed'] for t in steps]==list(range(len(steps)))
    v=r/'verification.jsonl';events=[json.loads(s) for s in v.read_text().splitlines()] if v.exists() else []
    counts=collections.Counter(x['event'] for x in events)
    statuses=collections.Counter(x.get('status') for x in events if x['event'] in ('decision','finish'))
    row.update(status='complete',scene_id=e['scene_id'],episode_id=e['episode_id'],success=m['success'],spl=m['spl'],distance=m['distance_to_goal'],steps=len(steps),
        path_m=m.get('trace_path_m'),target_detected=m.get('target_detected'),stop_called=m.get('stop_called'),
        triggers=counts['trigger'],views=counts['view_selected'],evidence=counts['evidence'],
        rejections=statuses['rejected'],confirmations=statuses['confirmed'],suppressed=counts['candidate_suppressed'],
        resumes=counts['exploration_resume'],no_feasible_view=counts['no_feasible_view'],
        av_actions=m.get('av_actions',0),av_calls=m.get('av_model_calls',0),av_path=m.get('av_path',0),
        decisions=dict(statuses),failure_cause=e['failure_cause'])
    rows.append(row)
pairs=[]
for num in sorted({r['num'] for r in rows}):
    by={r['strategy']:r for r in rows if r['num']==num and r['status']=='complete'}
    if 'baseline' not in by:continue
    for name in ['passive','active']:
        if name in by:
            b=by['baseline'];v=by[name]
            pairs.append(dict(num=num,strategy=name,recovered=b['success']==0 and v['success']==1,
                regressed=b['success']==1 and v['success']==0,delta_success=v['success']-b['success'],
                delta_steps=v['steps']-b['steps']))
print(json.dumps(dict(jobs_recorded=len(records),expected_jobs=36,complete=sum(r['status']=='complete' for r in rows),
    failed=sum(r['status']=='failed' for r in rows),rows=rows,pairs=pairs),indent=2))
