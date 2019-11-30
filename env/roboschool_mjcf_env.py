# Copyright (c) 2018 ikeyasu (http://ikeyasu.com)
import gym
import numpy as np
# noinspection PyUnresolvedReferences
import roboschool
from gym.envs.registration import register
from roboschool import cpp_household
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv

from env.roboschool_gym_forward_walker_servo import RoboschoolForwardWalkerServo


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
    RoboschoolForwardWalkerServo.__init__(self, power=0.30)
    RoboschoolMujocoXmlEnv.__init__(self, model_xml, robot_name, action_dim=action_dim, obs_dim=obs_dim)


def _robot_specific_reset(self):
    RoboschoolForwardWalkerServo.robot_specific_reset(self)
    self.set_initial_orientation(yaw_center=0, yaw_random_spread=np.pi)
    # self.head = self.parts["head"]


def _set_initial_orientation(self, yaw_center, yaw_random_spread):
    # noinspection PyArgumentList
    cpose = cpp_household.Pose()
    yaw = yaw_center

    cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 1.0)
    cpose.set_rpy(0, 0, yaw)  # just face random direction, but stay straight otherwise
    self.cpp_robot.set_pose_and_speed(cpose, 0, 0, 0)
    self.initial_z = 1.5


def _alive_bonus(self, z, pitch):
    return +1 if z > 0.1 else -1  # 0.1 is position of base_link


def make(model_xml, robot_name, footlist, action_dim, obs_dim):
    robot = type("Robo", (RoboschoolForwardWalkerServo, RoboschoolMujocoXmlEnv,), {
        "foot_list": footlist,
        "__init__": lambda self: _robo_init(self, model_xml, robot_name, action_dim, obs_dim),
        "alive_bonus": _alive_bonus,
        "robot_specific_reset": _robot_specific_reset,
        "set_initial_orientation": _set_initial_orientation
    })
    register(
        id='RoboschoolRoboUrdf-v1',
        entry_point=robot,
        max_episode_steps=1000,
        tags={"pg_complexity": 200 * 1000000}
    )
    return gym.make("RoboschoolRoboUrdf-v1")
