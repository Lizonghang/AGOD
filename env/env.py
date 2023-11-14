import gym
from gym.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv
from .swarm_manager import SwarmManager
from .config import *


class AIGCEnv(gym.Env):

    def __init__(self):
        self._swarm_manager = SwarmManager()
        self._observation_space = Box(shape=self.state.shape, low=0, high=1)
        self._action_space = Discrete(NUM_SERVICE_PROVIDERS)

        self._num_steps = 0
        self._terminated = False
        self._global_clock = 0

    @property
    def swarm_manager(self):
        return self._swarm_manager

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def state(self):
        """Generate state (i.e., observation)"""
        return self._swarm_manager.vector

    def reset(self):
        self._swarm_manager.reset()
        self._num_steps = 0
        self._terminated = False
        self._global_clock = 0
        return self.state, {'num_steps': self._num_steps}

    def step(self, action):
        assert not self._terminated, "One episodic has terminated"

        reward = self._swarm_manager.assign(action, self._global_clock)
        self._global_clock, self._terminated = self._swarm_manager.next_user_task()
        self._num_steps += 1
        info = {'num_steps': self._num_steps, 'curr_time': self._global_clock}
        return self.state, reward, self._terminated, info

    def render(self, mode=""):
        self._swarm_manager.monitor()

    def seed(self, seed=None):
        np.random.seed(seed)


def make_env(task, training_num=0, test_num=0):
    """Wrapper function for env.
    :return: a tuple of (single env, training envs, test envs).
    """

    def _select_env(task_name):
        return AIGCEnv() if task_name == 'AaaS' else gym.make(task_name)

    env = _select_env(task)
    if task == 'AaaS': env.seed(SEED)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv(
            [lambda: _select_env(task) for _ in range(training_num)])
        if task == 'AaaS': train_envs.seed(SEED)

    if test_num:
        test_envs = DummyVectorEnv(
            [lambda: _select_env(task) for _ in range(test_num)])
        if task == 'AaaS': test_envs.seed(SEED)

    return env, train_envs, test_envs
