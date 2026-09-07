"""Select stable, category-stratified episodes without changing scene/episode IDs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,help='train or val directory containing content/*.json.gz')
    p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,required=True)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--partition',choices=['all','calibration','development'],default='all',
                   help='Stable scene split: 75%% calibration / 25%% development; training data only')
    args=p.parse_args()
    source=Path(args.source)
    destination=Path(args.output)
    if destination.exists():
        p.error('Output already exists; splits are immutable')
    if args.count<1:
        p.error('Count must be positive')
    by_class,meta={},{}
    total=0
    def priority(episode):
        value=f"{args.seed}|{episode['scene_id']}|{episode['episode_id']}"
        return hashlib.sha256(value.encode()).hexdigest()
    for path in sorted((source/'content').glob('*.json.gz')):
        scene_bucket=int(hashlib.sha256(('scene-v1|'+path.stem).encode()).hexdigest()[:8],16)%4
        if args.partition=='calibration' and scene_bucket==0:
            continue
        if args.partition=='development' and scene_bucket!=0:
            continue
        with gzip.open(path,'rt') as f:
            data=json.load(f)
        total+=len(data.get('episodes',[]))
        meta.update({k:v for k,v in data.items() if k not in {'episodes','goals_by_category','content_scenes_path'}})
        scene_classes={}
        for e in data.get('episodes',[]):
            scene_classes.setdefault(e['object_category'],[]).append(e)
        for category,items in scene_classes.items():
            items.sort(key=priority)
            for e in items[:args.count]:
                goal_key=Path(e['scene_id']).name+'_'+category
                by_class.setdefault(category,[]).append((priority(e),e,goal_key,data.get('goals_by_category',{}).get(goal_key,[])))
            by_class[category].sort(key=lambda item:item[0])
            by_class[category]=by_class[category][:args.count]
    if total<args.count:
        p.error(f'Only {total} episodes available')
    chosen=[]
    selected_goals={}
    while len(chosen)<args.count:
        for c in sorted(by_class):
            if by_class[c] and len(chosen)<args.count:
                _,episode,key,goal=by_class[c].pop(0)
                chosen.append(episode)
                selected_goals[key]=goal
    chosen.sort(key=lambda e:(e['scene_id'],str(e['episode_id'])))
    keys=[(e['scene_id'],str(e['episode_id'])) for e in chosen]
    if len(set(keys))!=len(keys):
        raise ValueError('Duplicate scene/episode IDs')
    result=dict(meta,episodes=chosen,goals_by_category=selected_goals,
                content_scenes_path='{data_path}/__no_external_content__/{scene}.json.gz')
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Stable gzip header makes file hashes reproducible.
    destination.write_bytes(gzip.compress(json.dumps(result,sort_keys=True).encode(),mtime=0))
    manifest=dict(source=str(source),seed=args.seed,partition=args.partition,
                  selection_algorithm='sha256-priority-v2',count=len(chosen),keys=keys,
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='keys'},indent=2))


if __name__=='__main__':
    main()
