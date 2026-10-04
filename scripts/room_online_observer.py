"""Submit each current observation and wait for both segmenters before stepping."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from room_frame_stream import atomic_json


class OnlineRooms:
    def __init__(self, settings, config):
        self.settings = settings
        self.root = Path(settings['output_dir'])
        self.root.mkdir(parents=True, exist_ok=False)
        self.sample = Path(settings['sample_dir'])
        import gzip
        self.selection = json.loads((self.sample / 'selection.json').read_text())
        payload = (self.sample / 'episodes.json.gz').read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.selection['sha256']:
            raise ValueError('Online sample hash mismatch')
        self.episodes = json.loads(gzip.decompress(payload))['episodes']
        self.config = config
        self.processes, self.logs, self.cases = [], [], []
        self.current = None
        self.records = []
        self.terminal = False
        self.atomic_json = atomic_json
        atomic_json(self.root / 'settings.json', settings)

    def start(self, episode):
        matches = [i for i, e in enumerate(self.episodes) if episode.scene_id.endswith(e['scene_id'])]
        if len(matches) != 1 or str(episode.episode_id) != str(matches[0]):
            raise ValueError('Online episode identity mismatch')
        self.current = matches[0]
        self.case = self.root / 'inputs' / f'{self.current:02d}'
        self.case.mkdir(parents=True, exist_ok=False)
        hc = self.config.habitat.simulator
        camera = next(iter(hc.agents.values())).sim_sensors.depth_sensor
        self.source = dict(selection=self.selection['cases'][self.current], episode=self.episodes[self.current],
                           frames=0, poses=[], hfov=float(camera.hfov), sensor_height=float(camera.position[1]),
                           depth_units='meters', result=None, online=True)
        atomic_json(self.case / 'manifest.json', self.source)
        self.records = []
        self.terminal = False
        self.processes, self.logs = [], []
        repo = Path(__file__).resolve().parents[1]
        s = self.settings
        active = [s['python'], '-u', str(repo / 'scripts/run_active_room_replay.py'),
                  '--upstream', s['active_upstream'], '--detr-source', s['detr_source'], '--weights', s['weights'],
                  '--case-dir', str(self.case), '--output-dir', str(self.root / 'active' / f'{self.current:02d}'), '--stream']
        relative = self.root.relative_to(repo)
        container_root = Path('/online') / relative
        occ = ['docker', 'exec', '-e', f'ROS_DOMAIN_ID={s.get("domain_base",170)+self.current}', '-e', 'RMW_IMPLEMENTATION=rmw_cyclonedds_cpp',
               s['container'], 'bash', '-lc',
               'source /opt/ros/humble/setup.bash; source /task/runs/occusg_build_v2/install/setup.bash; '
               'source /task/runs/occusg_build_v3/install/local_setup.bash; '
               'export PYTHONPATH=/usr/lib/python3/dist-packages:/online:$PYTHONPATH; '
               f'python3 -u /online/scripts/run_occusg_room_replay.py --upstream /task/runs/occusg_build_v3 '
               f'--case-dir {container_root}/inputs/{self.current:02d} '
               f'--output-dir {container_root}/occusg/{self.current:02d} --stream --frame-timeout 120 --debug']
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([s['extra_python_path'], str(repo / 'scripts'), str(repo)])
        atomic_json(self.case / 'worker_commands.json', dict(active=active, occusg=occ))
        for name, command in [('active', active), ('occusg', occ)]:
            (self.root / name).mkdir(exist_ok=True)
            log = (self.root / f'{name}_{self.current:02d}.log').open('w')
            self.logs.append(log)
            self.processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))

    def observe(self, envs, episode, observation, action):
        import numpy as np
        if self.current is None:
            self.start(episode)
        if self.terminal:
            raise ValueError('Observation submitted after final online frame')
        index = len(self.records)
        data = envs.call_at(0, 'room_observation')
        if not np.array_equal(data['rgb'], observation['rgb']):
            raise ValueError('Online raw RGB does not match policy observation')
        camera = next(iter(self.config.habitat.simulator.agents.values())).sim_sensors.depth_sensor
        expected = np.clip(data['depth'], float(camera.min_depth), float(camera.max_depth))
        if camera.normalize_depth:
            expected = (expected - float(camera.min_depth)) / (float(camera.max_depth) - float(camera.min_depth))
        if not np.allclose(expected, np.asarray(observation['depth']).squeeze(), atol=1e-6, rtol=0):
            raise ValueError('Online raw depth does not match policy observation')
        last = int(action) == 0 or index + 1 == int(self.config.habitat.environment.max_episode_steps)
        self.source['frames'] = index + 1
        self.source['poses'].append(dict(step=index, position=np.asarray(data['agent_position']).tolist()))
        atomic_json(self.case / 'manifest.json', self.source)
        temporary = self.case / f'{index:04d}.tmp.npz'
        np.savez_compressed(temporary, **data)
        temporary.replace(self.case / f'{index:04d}.npz')
        published = time.time_ns()
        atomic_json(self.case / f'{index:04d}.ready.json', dict(frame=index, last=last, published_ns=published))
        acks = []
        for name, proc in zip(['active', 'occusg'], self.processes):
            ack = self.root / name / f'{self.current:02d}' / f'ack_{index:04d}.json'
            deadline = time.monotonic() + 1800
            while not ack.exists():
                if proc.poll() is not None:
                    raise RuntimeError(f'{name} worker exited {proc.returncode} before frame {index} ack')
                if time.monotonic() > deadline:
                    raise TimeoutError(f'{name} online frame {index} timed out')
                time.sleep(.02)
            acks.append(json.loads(ack.read_text()))
        record = dict(frame=index, action=int(action), published_ns=published,
                      both_methods_completed_ns=time.time_ns(), methods=acks, last=last)
        self.records.append(record)
        with (self.case / 'online_events.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        self.terminal = last
        if (index + 1) % 10 == 0 or last:
            from room_online_visualization import render_live
            render_live(self.root, self.current, self.source, data, index)
        if index % 20 == 0 or last:
            print('ONLINE_ROOMS', self.current, index, 'both updated before env.step', flush=True)

    def finish(self, metrics, failure_cause):
        if not self.terminal:
            raise ValueError('Episode ended without final stream marker')
        self.source['result'] = dict(steps_recorded=len(self.records), metrics=metrics, failure_cause=failure_cause)
        atomic_json(self.case / 'manifest.json', self.source)
        for name, proc in zip(['active', 'occusg'], self.processes):
            code = proc.wait(timeout=90)
            if code != 0:
                raise RuntimeError(f'{name} online worker failed: {code}')
            folder = self.root / name / f'{self.current:02d}'
            summary = json.loads((folder / 'summary.json').read_text())
            summary['source'] = self.source
            if name == 'occusg':
                subprocess.run(['docker', 'exec', self.settings['container'], 'chown', '-R',
                                f'{os.getuid()}:{os.getgid()}', str(Path('/online') / folder.relative_to(Path(__file__).resolve().parents[1]))], check=True)
            atomic_json(folder / 'summary.json', summary)
        self.cases.append(self.source)
        atomic_json(self.root / 'inputs/manifest.json', dict(cases=self.cases, online=True, diagnostic=self.settings.get('diagnostic', False)))
        for log in self.logs:
            log.close()
        self.current = None

    def transition(self, done):
        atomic_json(self.case / f'transition_{len(self.records)-1:04d}.json',
                    dict(frame=len(self.records)-1, env_step_finished_ns=time.time_ns(), done=bool(done)))


_observer = None


def get_online_rooms(config):
    global _observer
    if _observer is None:
        settings = json.loads(Path(os.environ['AVNAV_ROOM_ONLINE_SETTINGS']).read_text())
        _observer = OnlineRooms(settings, config)
    return _observer
