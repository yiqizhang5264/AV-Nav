"""ROS2 shared-input bridge for official OccuSG occupancy and DuDe nodes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', required=True)
    p.add_argument('--case-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--frame-timeout', type=float, default=20)
    args = p.parse_args()
    import cv2
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from nav_msgs.msg import OccupancyGrid
    from incremental_dude_msgs.msg import Region2DArray
    from tf2_ros import TransformBroadcaster
    from scipy.spatial.transform import Rotation
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from av_nav.room_geometry import ros_sensor_pose
    case, output, upstream = Path(args.case_dir), Path(args.output_dir), Path(args.upstream)
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((case / 'manifest.json').read_text())
    params = yaml.safe_load((upstream / 'src/scene_graph_ros/config/scene_graph_pipeline_params_mp3d.yaml').read_text())
    for node_name in ['point_cloud_generator_node', 'octomap_server', 'map_conversion_node']:
        params[node_name]['ros__parameters']['use_sim_time'] = False
        params[node_name]['ros__parameters'].update(enable_profiling=True, profiling_output_path=str(output / 'profiles'),
                                                    profiling_run_name='room_case', profiling_discard_first_n=5)
    pc = params['point_cloud_generator_node']['ros__parameters']
    pc.update(depth_image_topic='/depth', camera_info_topic='/depth/camera_info', output_frame='depth_optical', target_hz=0.0)
    parameter_file = output / 'pipeline_params.yaml'
    parameter_file.write_text(yaml.safe_dump(params))
    dude = yaml.safe_load((upstream / 'src/incremental_dude_ros2/incremental_dude_ros2/config/inc_dude_params.yaml').read_text())
    dude['/**']['ros__parameters'].update(use_sim_time=False, enable_profiling=True,
                                         profiling_output_path=str(output / 'profiles'), profiling_run_name='room_case')
    dude_file = output / 'dude_params.yaml'
    dude_file.write_text(yaml.safe_dump(dude))
    commands = [
        ['ros2', 'run', 'point_cloud_generator', 'depth_to_pointcloud', '--ros-args', '--params-file', str(parameter_file)],
        ['ros2', 'run', 'octomap_server', 'octomap_server_node', '--ros-args', '--params-file', str(parameter_file),
         '-r', 'cloud_in:=/pointcloud', '-r', 'octomap_full:=/octomap'],
        ['ros2', 'run', 'mapconversion', 'map_conversion_oct_node', '--ros-args', '--params-file', str(parameter_file)],
        ['ros2', 'run', 'incremental_dude_ros2', 'inc_dude', '--ros-args', '--params-file', str(dude_file)]]
    (output / 'commands.json').write_text(json.dumps(commands, indent=2))
    processes, logs = [], []
    rclpy.init()
    node = Node('room_replay_bridge')
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    map_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    depth_pub = node.create_publisher(Image, '/depth', qos)
    info_pub = node.create_publisher(CameraInfo, '/depth/camera_info', qos)
    tf_pub = TransformBroadcaster(node)
    received = dict(pc_stamp=None, map=None, map_generation=0, regions=None, region_generation=0)
    def stamp(s):
        return s.sec * 1000000000 + s.nanosec
    def cloud_callback(msg):
        received['pc_stamp'] = stamp(msg.header.stamp)
    def map_callback(msg):
        received['map'] = msg
        received['map_generation'] += 1
    def region_callback(msg):
        received['regions'] = msg
        received['region_generation'] += 1
    node.create_subscription(PointCloud2, '/pointcloud', cloud_callback, qos)
    node.create_subscription(OccupancyGrid, '/mapUAV', map_callback, map_qos)
    node.create_subscription(Region2DArray, '/dude/regions', region_callback, qos)
    records = []
    def spin_until(condition, timeout):
        deadline = time.monotonic() + timeout
        while not condition():
            if any(proc.poll() is not None for proc in processes):
                raise RuntimeError('OccuSG node exited; inspect node logs')
            if time.monotonic() > deadline:
                raise TimeoutError('OccuSG pipeline did not acknowledge the input frame')
            rclpy.spin_once(node, timeout_sec=.02)
    def snapshot(index):
        grid, region_msg = received['map'], received['regions']
        if grid is None or region_msg is None:
            raise ValueError('Missing occupancy/region output')
        occupancy = np.asarray(grid.data, np.int16).reshape(grid.info.height, grid.info.width)
        labels = np.zeros(occupancy.shape, np.int32)
        ox, oy = grid.info.origin.position.x, grid.info.origin.position.y
        res = grid.info.resolution
        polygons = []
        for region in region_msg.regions:
            points = [[float(pt.x), float(pt.y)] for pt in region.polygon.points]
            polygons.append(dict(id=region.id, area=region.area, points=points))
            if len(points) >= 3:
                xy = np.rint((np.asarray(points) - [ox, oy]) / res).astype(np.int32)
                cv2.fillPoly(labels, [xy], int(region.id) + 1)
        labels[occupancy != 0] = 0
        np.savez_compressed(output / f'checkpoint_{index:04d}.npz', labels=labels, occupancy=occupancy,
                            origin=[ox,oy], resolution=res)
        (output / f'regions_{index:04d}.json').write_text(json.dumps(polygons, indent=2))
    try:
        for index, command in enumerate(commands):
            log = (output / f'node_{index}.log').open('w')
            logs.append(log)
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        spin_until(lambda: depth_pub.get_subscription_count() > 0 and info_pub.get_subscription_count() > 0
                   and node.count_subscribers('/pointcloud') >= 2 and node.count_subscribers('/mapUAV') >= 2
                   and node.count_publishers('/dude/regions') > 0, 30)
        frames = sorted(case.glob('[0-9][0-9][0-9][0-9].npz'))
        floor_y = source['episode']['start_position'][1]
        for index, frame in enumerate(frames):
            started = time.perf_counter()
            data = np.load(frame)
            if abs(float(data['agent_position'][1] - floor_y)) > .3:
                raise ValueError('Multi-floor trajectory requires separate floor maps')
            sensor_r = Rotation.from_quat(data['sensor_rotation'][[1,2,3,0]]).as_matrix()
            position, rotation = ros_sensor_pose(data['sensor_position'], sensor_r, floor_y)
            now = node.get_clock().now().to_msg()
            transform = TransformStamped()
            transform.header.stamp, transform.header.frame_id = now, 'map'
            transform.child_frame_id = 'depth_optical'
            transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = map(float, position)
            q = Rotation.from_matrix(rotation).as_quat()
            transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = map(float, q)
            tf_pub.sendTransform(transform)
            depth = np.ascontiguousarray(data['depth'], dtype=np.float32)
            h, w = depth.shape
            info = CameraInfo()
            info.header.stamp, info.header.frame_id = now, 'depth_optical'
            info.height, info.width = h, w
            f = w / (2 * np.tan(np.deg2rad(source['hfov']) / 2))
            info.k = [f,0.,(w-1)/2,0.,f,(h-1)/2,0.,0.,1.]
            info.p = [f,0.,(w-1)/2,0.,0.,f,(h-1)/2,0.,0.,0.,1.,0.]
            image = Image()
            image.header = info.header
            image.height, image.width, image.encoding, image.step = h, w, '32FC1', w * 4
            image.data = depth.tobytes()
            previous_map, previous_region = received['map_generation'], received['region_generation']
            info_pub.publish(info)
            depth_pub.publish(image)
            spin_until(lambda: received['pc_stamp'] == stamp(now) and received['map_generation'] > previous_map
                       and received['region_generation'] > previous_region, args.frame_timeout)
            records.append(dict(frame=index, seconds=time.perf_counter()-started,
                                region_count=len(received['regions'].regions), map_updates=received['map_generation']))
            if index + 1 in [100,250,500] or index == len(frames)-1:
                snapshot(index + 1)
            if index % 20 == 0:
                print('OCCUSG',index,records[-1],flush=True)
        summary = dict(source=source, records=records, final_room_count=len(received['regions'].regions),
                       frames=len(frames), accuracy=None, accuracy_reason='No verified room footprint GT',
                       adapter=dict(ros_frame='x=world_x,y=-world_z,z=world_y-start_y', pose_source='simulator',
                                    native_nodes=True, object_graph_disabled=True, floor_policy='reject transitions >0.3m'))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        (output / 'exit.json').write_text(json.dumps(dict(returncode=0)))
    finally:
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for log in logs:
            log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
