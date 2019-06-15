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


def _reset(self):
    if self.scene is None:
        self.scene = self.create_single_player_scene()
    if not self.scene.multiplayer:
        self.scene.episode_restart()
    self.mjcf = self.scene.cpp_world.load_mjcf(self.model_xml)
    self.ordered_joints = []
    self.jdict = {}
    self.parts = {}
    self.frame = 0
    self.done = 0
    self.reward = 0
    dump = 0
    for r in self.mjcf:
        if dump: print("ROBOT '%s'" % r.root_part.name)
        if r.root_part.name == self.robot_name:
            self.cpp_robot = r
            self.robot_body = r.root_part
        for part in r.parts:
            if dump: print("\tPART '%s'" % part.name)
            self.parts[part.name] = part
            if part.name == self.robot_name:
                self.cpp_robot = r
                self.robot_body = part
        for j in r.joints:
            if dump: print(
                "\tALL JOINTS '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()))
            if j.name[:6] == "ignore":
                j.set_motor_torque(0)
                continue
            j.power_coef = 100.0
            self.ordered_joints.append(j)
            self.jdict[j.name] = j
    assert (self.cpp_robot)
    self.robot_specific_reset()
    for r in self.mjcf:
        r.query_position()
    s = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
    self.potential = self.calc_potential()
    self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
    return s


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
    # env = gym.make("RoboschoolHumanoidFlagrun-v1")
    robot = type("Robo", (RoboschoolForwardWalkerServo, RoboschoolMujocoXmlEnv,), {
        "foot_list": footlist,
        "__init__": lambda self: _robo_init(self, model_xml, robot_name, action_dim, obs_dim),
        "alive_bonus": _alive_bonus,
        # "_reset": _reset,
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
