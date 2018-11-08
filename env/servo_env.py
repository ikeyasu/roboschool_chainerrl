# Copyright (c) 2018 ikeyasu (http://ikeyasu.com)
import gym
import numpy as np
from gym.envs.registration import register


# TODO: not used yet
class PhysicalRobotServo(gym.Env):
    def __init__(self, action_dim, obs_dim):
        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


class PhysicalWrapper(gym.Env):
    env = None

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self._warn_double_wrap()

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        env = self.env
        while True:
            if isinstance(env, PhysicalWrapper):
                if env.class_name() == self.class_name():
                    raise Exception("Attempted to double wrap with Wrapper: {}".format(self.__class__.__name__))
                env = env.env
            else:
                break

    def step(self, action):
        if hasattr(self, "_step"):
            self.step = self._step
            return self.step(action)
        else:
            return self.env.step(action)

    def reset(self, **kwargs):
        if hasattr(self, "_reset"):
            self.reset = self._reset
            return self._reset(**kwargs)
        else:
            return self.env.reset(**kwargs)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        if self.env:
            return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def spec(self):
        return self.env.spec


def _init(self, simenv):
    PhysicalWrapper.__init__(self, simenv)


def make(simenv=None):
    robot = type("Robo", (PhysicalWrapper,), {
        "__init__": lambda self: _init(self, simenv),
    })
    register(
        id='PhysicalWithSimRobotServo-v1',
        entry_point=robot,
        max_episode_steps=10000000,
        tags={"pg_complexity": 200 * 1000000, "simenv": simenv}
    )
    if simenv is not None:
        return gym.make("PhysicalWithSimRobotServo-v1")
    else:
        raise Exception("Not implemented yet")
