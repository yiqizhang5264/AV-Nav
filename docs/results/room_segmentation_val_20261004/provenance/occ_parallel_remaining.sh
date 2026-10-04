set -uo pipefail
cd /home/zyq/AV-Nav-worktrees/room-val-0029f40
for n in 02 03 04; do
 (
 docker exec -e ROS_DOMAIN_ID=$((210+10#$n)) -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp avnav_room_val_occ_20261004 bash -lc "source /opt/ros/humble/setup.bash; source /task/runs/occusg_build_v2/install/setup.bash; source /task/runs/occusg_build_v3/install/local_setup.bash; export PYTHONPATH=/usr/lib/python3/dist-packages:/val:\$PYTHONPATH; python3 -u /val/scripts/run_occusg_room_replay.py --upstream /task/runs/occusg_build_v3 --case-dir /val/runs/val_replay5/$n --output-dir /val/runs/val_occusg5_v2/$n --frame-timeout 120 --debug" > "runs/occusg_v2_$n.log" 2>&1
 code=$?
 docker exec avnav_room_val_occ_20261004 mkdir -p "/val/runs/val_occusg5_v2/$n"
 docker exec avnav_room_val_occ_20261004 chown -R "$(id -u):$(id -g)" "/val/runs/val_occusg5_v2/$n"
 echo "{\"returncode\":$code}" > "runs/val_occusg5_v2/$n/exit.json"
 echo OCCUSG_DONE_$n=$code
 ) &
done
wait
while [ ! -f runs/val_occusg5_v2/01/exit.json ]; do sleep 10; done
docker exec avnav_room_val_occ_20261004 chown -R "$(id -u):$(id -g)" /val/runs/val_occusg5_v2
echo OCCUSG_ALL_DONE
