set -euo pipefail
cd /home/zyq/AV-Nav-worktrees/room-online-9301f80
PY=/home/zyq/miniconda3/envs/vlfm/bin/python
$PY -u scripts/run_room_capture.py --sample-dir runs/fixed_online_cases/00 --output-dir runs/smoke_v2_capture --vlfm-root /home/zyq/AV-Nav-worktrees/vlfm-original-1224d7e/external/vlfm --gpu 3 --max-steps 2 --online-settings runs/smoke_v2_settings.json > runs/smoke_v2_launch.log 2>&1
$PY - <<'PY'
import json
from pathlib import Path
case=Path('runs/online_smoke_v2/inputs/00')
rows=[json.loads(s) for s in (case/'online_events.jsonl').read_text().splitlines()]
assert len(rows)==2
previous=0
for row in rows:
 t=json.loads((case/f'transition_{row["frame"]:04d}.json').read_text())
 assert previous < row['published_ns'] < row['both_methods_completed_ns'] < t['env_step_finished_ns']
 assert all(row['published_ns'] < ack['completed_ns'] <= row['both_methods_completed_ns'] for ack in row['methods'])
 previous=t['env_step_finished_ns']
print('ONLINE_SMOKE_ORDER_VERIFIED',flush=True)
PY
for n in 00 01 02 03 04; do
 (
  set +e
  $PY -u scripts/run_room_capture.py --sample-dir "runs/fixed_online_cases/$n" --output-dir "runs/online_capture_$n" --vlfm-root /home/zyq/AV-Nav-worktrees/vlfm-original-1224d7e/external/vlfm --gpu 3 --max-steps 500 --online-settings "runs/settings_$n.json" > "runs/launch_$n.log" 2>&1
  code=$?
  echo ONLINE_CASE_$n=$code
 ) &
done
wait
echo ONLINE_ALL_EVALUATIONS_FINISHED
