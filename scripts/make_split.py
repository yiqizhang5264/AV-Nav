"""Select stable, category-stratified episodes without changing scene/episode IDs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,help='train or val directory containing content/*.json.gz')
    p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,required=True)
    p.add_argument('--seed',type=int,default=17)
    args=p.parse_args()
    source=Path(args.source)
    destination=Path(args.output)
    if destination.exists():
        p.error('Output already exists; splits are immutable')
    if args.count<1:
        p.error('Count must be positive')
    episodes,goals,meta=[],{},{}
    for path in sorted((source/'content').glob('*.json.gz')):
        with gzip.open(path,'rt') as f:
            data=json.load(f)
        episodes.extend(data.get('episodes',[]))
        goals.update(data.get('goals_by_category',{}))
        meta.update({k:v for k,v in data.items() if k not in {'episodes','goals_by_category','content_scenes_path'}})
    if len(episodes)<args.count:
        p.error(f'Only {len(episodes)} episodes available')
    by_class={}
    for e in episodes:
        by_class.setdefault(e['object_category'],[]).append(e)
    rng=random.Random(args.seed)
    for category in sorted(by_class):
        by_class[category].sort(key=lambda e:(e['scene_id'],str(e['episode_id'])))
        rng.shuffle(by_class[category])
    chosen=[]
    while len(chosen)<args.count:
        for c in sorted(by_class):
            if by_class[c] and len(chosen)<args.count:
                chosen.append(by_class[c].pop())
    chosen.sort(key=lambda e:(e['scene_id'],str(e['episode_id'])))
    keys=[(e['scene_id'],str(e['episode_id'])) for e in chosen]
    if len(set(keys))!=len(keys):
        raise ValueError('Duplicate scene/episode IDs')
    selected_goals={}
    for e in chosen:
        key=Path(e['scene_id']).name+'_'+e['object_category']
        if key in goals:
            selected_goals[key]=goals[key]
    result=dict(meta,episodes=chosen,goals_by_category=selected_goals,
                content_scenes_path='{data_path}/__no_external_content__/{scene}.json.gz')
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Stable gzip header makes file hashes reproducible.
    destination.write_bytes(gzip.compress(json.dumps(result,sort_keys=True).encode(),mtime=0))
    manifest=dict(source=str(source),seed=args.seed,count=len(chosen),keys=keys,
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='keys'},indent=2))


if __name__=='__main__':
    main()
