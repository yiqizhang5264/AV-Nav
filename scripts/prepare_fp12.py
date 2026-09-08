"""Build fixed single-episode splits from audited source enumeration IDs."""
import gzip,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
out=ROOT/'runs/splits/hm3dv1_fp12';out.mkdir(parents=True,exist_ok=False)
selection=json.loads((ROOT/'configs/diagnostics/hm3dv1_fp12.json').read_text())
manifest=[]
for r in selection['cases']:
    source=pathlib.Path('/home/zyq/vlfm/data/datasets/objectnav/hm3d/v1/val/content')/(r['scene']+'.json.gz')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==r['dataset_sha256']
    d=json.load(gzip.open(source,'rt'));e=d['episodes'][int(r['episode_id'])].copy()
    assert pathlib.Path(e['scene_id']).name.split('.')[0]==r['scene']
    e['episode_id']=r['episode_id'];d['episodes']=[e]
    d['content_scenes_path']='{data_path}/__no_external_content__/{scene}.json.gz'
    key=pathlib.Path(e['scene_id']).name+'_'+e['object_category']
    d['goals_by_category']={key:d['goals_by_category'][key]}
    f=out/(str(r['num'])+'.json.gz');f.write_bytes(gzip.compress(json.dumps(d,sort_keys=True).encode(),mtime=0))
    manifest.append(dict(num=r['num'],scene_id=e['scene_id'],episode_id=e['episode_id'],sha256=hashlib.sha256(f.read_bytes()).hexdigest(),source_sha256=r['dataset_sha256']))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Prepared',len(manifest),'single-episode splits')
