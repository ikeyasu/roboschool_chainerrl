# Copyright (c) 2019 ikeyasu (http://ikeyasu.com)
import gym
import numpy as np
from gym.envs.registration import register
import pybullet
from .pybullet_urdf_base_env import URDFWalkerBaseBulletEnv
from .pybullet_waker_base import URDFWalkerBase


def random_action(action_space):
    a = action_space.sample()
    if isinstance(a, np.ndarray):
        a = a.astype(np.float32)
    return a


def zero_action(action_space):
    a = action_space.sample()
    a = np.zeros(a.shape, np.float32)
    return a


def sin_action(old_rads):
    rads = old_rads + 0.01
    a = np.sin(rads)
    if isinstance(a, np.ndarray):
        a = a.astype(np.float32)
    return a, rads


def _robo_init(self, model_xml, robot_name, action_dim, obs_dim):
    URDFWalkerBase.__init__(self, model_xml, robot_name, action_dim, obs_dim, power=0.40)


def _robo_apply_action(self, a):
    assert (np.isfinite(a).all())
    for n, j in enumerate(self.ordered_joints):
        # j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
        target_position = float(np.clip(a[n], -1, +1))
        j._p.setJointMotorControl2(j.bodies[j.bodyIndex],
                                   j.jointIndex,
                                   pybullet.POSITION_CONTROL,
                                   targetPosition=target_position,
                                   force=50.0)


def _alive_bonus(self, z, pitch):
    return +1 if z > 0.1 else -1  # 0.1 is position of base_link


def make(model_xml, robot_name, footlist, action_dim, obs_dim, render=False,
         max_episode_steps=1000, reward_threshold=2500.0):
    robot = type("Robo", (URDFWalkerBase,), {
        "foot_list": footlist,
        "__init__": lambda self: _robo_init(self, model_xml, robot_name, action_dim, obs_dim),
        "apply_action": _robo_apply_action,
        "alive_bonus": _alive_bonus,
    })
    env = type("Env", (URDFWalkerBaseBulletEnv,), {
        "__init__": lambda self: URDFWalkerBaseBulletEnv.__init__(self, robot(), render)
    })
    register(
        id='PyBulletUrdf-v0',
        entry_point=env,
        max_episode_steps=max_episode_steps,
        reward_threshold=reward_threshold
    )
    return gym.make("PyBulletUrdf-v0")
