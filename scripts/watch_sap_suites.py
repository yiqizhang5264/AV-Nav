"""Detached completion watcher. Does not modify, restart or tune evaluations."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--sap',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'inputs.json').write_text(json.dumps(vars(args),indent=2))
    while True:
        status={}
        for name,root,session in [('baseline',args.baseline,'sap_hm3dv1_baseline'),
                                  ('sap',args.sap,'sap_hm3dv1_thinking')]:
            path=Path(root)
            done=(path/'summary.json').exists()
            running=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
            status[name]=dict(complete=done,session_running=running,
                logged_episode_lines=sum(sum(1 for line in f.open() if line.strip())
                                         for f in path.glob('*/attempt_*/episodes.jsonl')))
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),**status))+'\n')
        if all(item['complete'] for item in status.values()):
            command=[sys.executable,str(Path(__file__).with_name('compare_sap_suites.py')),
                     '--baseline',args.baseline,'--sap',args.sap,'--output',str(output/'final')]
            result=subprocess.run(command,capture_output=True,text=True)
            (output/'comparison.log').write_text(result.stdout+result.stderr)
            (output/('DONE' if result.returncode==0 else 'FAILED')).write_text(str(result.returncode)+'\n')
            raise SystemExit(result.returncode)
        if any(not item['complete'] and not item['session_running'] for item in status.values()):
            (output/'FAILED').write_text('Evaluation session exited before a full summary. Inspect preserved attempts.\n')
            raise SystemExit(2)
        time.sleep(60)


if __name__ == '__main__':
    main()
