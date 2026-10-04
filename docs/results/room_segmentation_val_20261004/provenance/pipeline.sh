set -euo pipefail
cd /home/zyq/AV-Nav-worktrees/room-val-0029f40
PY=/home/zyq/miniconda3/envs/vlfm/bin/python
CAP=/home/zyq/AV-Nav-worktrees/room-val-7b876c4/runs/val_capture5_20261004
RES=/home/zyq/AV-Nav-worktrees/room-6cdffd1/runs/resources
while [ ! -f "$CAP/exit.json" ]; do sleep 20; done
$PY -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["returncode"])' "$CAP/exit.json"
$PY -u scripts/replay_room_capture.py --capture-dir "$CAP" --output-dir runs/val_replay5 --gpu 3 > runs/replay.log 2>&1
echo REPLAY_DONE
mkdir runs/val_active5 runs/val_occusg5
docker run -d --name avnav_room_val_occ_20261004 --entrypoint bash \
 -v /home/zyq/AV-Nav-worktrees/room-val-0029f40:/val \
 -v /home/zyq/AV-Nav-worktrees/room-6cdffd1:/task \
 avnav-room-occ-val:20261004 -lc 'sleep infinity'
(
 for n in 00 01 02 03 04; do
  (
  set +e
  PYTHONPATH=/home/zyq/AV-Nav-worktrees/room-6cdffd1/runs/python_deps:$PWD $PY -u scripts/run_active_room_replay.py --upstream "$RES/active_proxy" --detr-source "$RES/detr" --weights "$RES/door_model.pth" --case-dir "runs/val_replay5/$n" --output-dir "runs/val_active5/$n" > "runs/active_$n.log" 2>&1
  code=$?
  set -e
  echo "{\"returncode\":$code}" > "runs/val_active5/$n/exit.json"
  echo ACTIVE_DONE_$n=$code
  ) &
 done
 wait
) &
active_pid=$!
for n in 00 01 02 03 04; do
 set +e
 docker exec -e ROS_DOMAIN_ID=$((200+10#$n)) -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp avnav_room_val_occ_20261004 bash -lc "source /opt/ros/humble/setup.bash; source /task/runs/occusg_build_v3/install/setup.bash; export PYTHONPATH=/usr/lib/python3/dist-packages:/val:\$PYTHONPATH; python3 -u /val/scripts/run_occusg_room_replay.py --upstream /task/runs/occusg_build_v3 --case-dir /val/runs/val_replay5/$n --output-dir /val/runs/val_occusg5/$n --frame-timeout 120 --debug" > "runs/occusg_$n.log" 2>&1
 code=$?
 set -e
 echo "{\"returncode\":$code}" > "runs/val_occusg5/$n/exit.json"
 echo OCCUSG_DONE_$n=$code
done
wait "$active_pid"
docker stop avnav_room_val_occ_20261004
$PY scripts/collect_room_comparison.py --replay-dir runs/val_replay5 --active-dir runs/val_active5 --occusg-dir runs/val_occusg5 --output-dir runs/val_comparison5
echo COMPARISON_DONE
