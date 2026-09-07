"""Aggregate real episode logs; paired comparison refuses mismatched episode sets."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    records=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    keyed={(r['scene_id'],r['episode_id']):r for r in records}
    if len(keyed)!=len(records) or not records:
        raise ValueError('Empty logs or duplicate episode IDs')
    return keyed


def summarize(records):
    rows=list(records.values())
    stops=[r for r in rows if r['metrics'].get('stop_called',False)]
    failed_stops=[r for r in stops if not r['metrics'].get('success',0)]
    result=dict(episodes=len(rows),target_stop_count=len(stops),unsuccessful_target_stop_count=len(failed_stops),
                unsuccessful_target_stops_per_episode=len(failed_stops)/len(rows),
                unsuccessful_target_stops_per_stop=len(failed_stops)/len(stops) if stops else None)
    for key in ('success','spl','distance_to_goal','av_actions','av_path','av_model_calls','av_triggers'):
        values=[r['metrics'][key] for r in rows if r['metrics'].get(key) is not None]
        result[key]=float(np.mean(values)) if values else None
    # Unsuccessful STOP is broader than semantic false-positive; do not conflate.
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('episodes')
    p.add_argument('--baseline')
    p.add_argument('--output')
    args=p.parse_args()
    records=read(args.episodes)
    result=summarize(records)
    if args.baseline:
        base=read(args.baseline)
        if set(base)!=set(records):
            raise ValueError('Paired comparison requires identical scene/episode sets')
        pairs=np.array([[base[k]['metrics']['success'],records[k]['metrics']['success']] for k in sorted(base)])
        delta=pairs[:,1]-pairs[:,0]
        rng=np.random.default_rng(17)
        boot=[float(rng.choice(delta,size=len(delta),replace=True).mean()) for _ in range(2000)]
        result['paired']=dict(recovered=int((delta>0).sum()),regressed=int((delta<0).sum()),
                               sr_delta=float(delta.mean()),episode_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist())
    text=json.dumps(result,indent=2)
    print(text)
    if args.output:
        out=Path(args.output)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(text+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
