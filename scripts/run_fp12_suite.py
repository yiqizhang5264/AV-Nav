"""Sequential fixed-commit diagnostic suite; failures recorded, never overwritten."""
import argparse,datetime,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--suite-id',required=True);p.add_argument('--nums',nargs='*',type=int);a=p.parse_args()
out=ROOT/'runs'/a.suite_id;out.mkdir(exist_ok=False)
cases=json.loads((ROOT/'configs/diagnostics/hm3dv1_fp12.json').read_text())['cases']
priority=[309,1850,1340,1894,1600,1,625,1517,638,1558,5,1796]
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
(out/'manifest.json').write_text(json.dumps(dict(commit=commit,seed=100,max_steps=500,priority=priority,strategies=['baseline','passive','active'],purpose='diagnostic; no validation tuning'),indent=2))
for num in priority:
    if a.nums and num not in a.nums:continue
    for strategy in ['baseline','passive','active']:
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==commit
        run_id=f'{a.suite_id}_{num}_{strategy}'
        command=[sys.executable,str(ROOT/'scripts/run_experiment.py'),'--config',str(ROOT/f'configs/diagnostics/{strategy}_seed100.json'),'--run-id',run_id,'--episodes','1','--max-steps','500','--dataset',str(ROOT/f'runs/splits/hm3dv1_fp12/{num}.json.gz'),'--trace']
        start=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (out/(run_id+'.launch.log')).open('w') as f:code=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
        record=dict(num=num,strategy=strategy,run_id=run_id,returncode=code,started=start,finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
        with (out/'jobs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
(out/'exit.json').write_text(json.dumps(dict(completed=True)))
