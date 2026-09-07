"""Publish small, auditable results from completed engineering runs, not raw assets."""
import json
import shutil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from summarize import read, summarize

out=ROOT/'results/smoke_20260907'
out.mkdir(parents=True,exist_ok=True)
names=['baseline_train_smoke5_v3','active_train_smoke5_v1']
reports={}
for name in names:
    run=ROOT/'runs'/name
    if json.loads((run/'exit.json').read_text())['returncode']!=0:
        raise SystemExit(f'Run did not succeed: {name}')
    data=read(run/'episodes.jsonl')
    if len(data)!=5:
        raise SystemExit('Incomplete smoke run')
    reports[name]=summarize(data)
    for filename in ['manifest.json','episodes.jsonl','exit.json']:
        shutil.copyfile(run/filename,out/(name+'_'+filename))
shutil.copyfile(ROOT/'runs/splits/train_smoke5.json.manifest.json',out/'split_manifest.json')
(out/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
note='''# First server smoke evaluation

Both runs completed five training episodes with a 100-action cap. This is an
engineering smoke test, not a paper benchmark or a calibrated method evaluation.
The active configuration produced zero active-verification triggers; therefore
these results do not establish any benefit from viewpoint selection.

The archived logs predate source-ID restoration: Habitat rewrote episode IDs to
0..4 when loading this combined split. `split_manifest.json` maps each index in
its `keys` list back to the original scene and source episode ID. Both runs used
the exact same split hash. Subsequent runs preserve source IDs in output logs.

The forced-review diagnostic is separate and must never be included in efficacy
tables. Calibration and full-budget evaluation are still required.
'''
(out/'README.md').write_text(note,encoding='utf-8')
print(json.dumps(reports,indent=2))
