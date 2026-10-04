"""Read current raw RGB-D and poses through a wrapped Habitat worker."""
from functools import partial


def make_online_env(original, *args, **kwargs):
    import gym
    import numpy as np
    import quaternion

    class RoomObservationWrapper(gym.Wrapper):
        def room_observation(self):
            simulator = self.unwrapped.habitat_env.sim
            state = simulator.get_agent_state()
            camera = state.sensor_states['depth']
            raw = simulator.get_sensor_observations()
            return dict(rgb=np.asarray(raw['rgb'])[:, :, :3], depth=np.asarray(raw['depth']),
                        agent_position=state.position, sensor_position=camera.position,
                        agent_rotation=quaternion.as_float_array(state.rotation),
                        sensor_rotation=quaternion.as_float_array(camera.rotation))

    return RoomObservationWrapper(original(*args, **kwargs))


def install_online_env():
    from habitat import VectorEnv
    original = VectorEnv.__init__

    def initialize(self, *args, **kwargs):
        if 'make_env_fn' not in kwargs:
            raise ValueError('Online adapter requires explicit Habitat environment factory')
        kwargs['make_env_fn'] = partial(make_online_env, kwargs['make_env_fn'])
        original(self, *args, **kwargs)

    VectorEnv.__init__ = initialize
