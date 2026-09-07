"""Launch a bounded experiment without keeping an SSH shell alive."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
(root/'runs').mkdir(exist_ok=True)
with (root/'runs/launcher.log').open('a') as log:
    child=subprocess.Popen([sys.executable,str(root/'scripts/run_experiment.py'),*sys.argv[1:]],
                           cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True)
print(f'Experiment launcher PID: {child.pid}')
