import math
import random
import numpy as np
from gym import Wrapper
from roboschool.gym_forward_walker import RoboschoolForwardWalker
# noinspection PyUnresolvedReferences
import roboschool

from window import PygletInteractiveWindow


class GymFPS(Wrapper):
    """Gym environment with first person camera.
    """
    # TODO: This reward implementation needs to be evaluated

    def __init__(self, env: RoboschoolForwardWalker, fps_window: bool=False, cam_size: tuple=(3, 64, 64)):
        super().__init__(env)
        self.cam_width = cam_size[1]
        self.cam_height = cam_size[2]
        self.is_showing_window = fps_window
        self.target_theta = 0
        self.window = PygletInteractiveWindow(env.unwrapped, self.cam_width, self.cam_height) if fps_window else None
        env.reset()
        self.camera = env.unwrapped.scene.cpp_world.new_camera_free_float(self.cam_width, self.cam_height , "camera")
        x, y = self._calc_walk_target(5.0, random.uniform(0, 2.0 * math.pi))
        self.flag_pos = (x, y)
        x, y, _ = env.unwrapped.body_xyz
        self.init_pos = (x, y)
        self.init_len = GymFPS._calc_length(self.flag_pos, self.init_pos)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # For detailed reward, you can see
        # alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost = self.env.unwrapped.rewards
        rgb_array, rgb_ary_len = self._render_fps()
        x, y, _ = self.env.unwrapped.body_xyz
        reward = self.init_len  - GymFPS._calc_length((x, y), self.flag_pos)
        obs = np.concatenate((rgb_array.reshape(rgb_ary_len), obs))
        return obs, reward, done, info

    def _calc_walk_target(self, scale=2.0, target_theta=None):
        eu = self.env.unwrapped
        x, y, z = eu.body_xyz
        # 1.0 or less will trigger flag reposition by env itself
        target_theta = self.target_theta if target_theta is None else target_theta
        return x + scale * np.cos(target_theta), y + scale * np.sin(target_theta)

    @staticmethod
    def _calc_length(pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return math.sqrt((math.fabs(x2 - x1) ** 2 + math.fabs(y2 - y1) ** 2))

    def _render_fps(self):
        eu = self.env.unwrapped
        eu.walk_target_x, eu.walk_target_y = self._calc_walk_target()
        eu.flag = eu.scene.cpp_world.debug_sphere(self.flag_pos[0], self.flag_pos[1], 0.5, 0.5, 0xFF8080)
        eu.flag_timeout = 100500
        x, y, z = eu.body_xyz
        r, p, yaw = eu.body_rpy
        cam_x = x + 0.2 * np.cos(yaw)
        cam_y = y + 0.2 * np.sin(yaw)
        tx = x + 2.0 * np.cos(yaw)
        ty = y + 2.0 * np.sin(yaw)
        tz = z + 2.0 * np.sin(p)
        self.camera.move_and_look_at(cam_x, cam_y, z, tx, ty, tz)
        rgb, depth, depth_mask, labeling, labeling_mask = self.camera.render(False, False, False)
        rgb_array = np.fromstring(rgb, dtype=np.uint8).reshape((self.cam_height, self.cam_width, 3))
        if self.window:
            self.window.imshow(rgb_array)
            self.target_theta += 0.05 * (
                    (1 if self.window.left_pressed else 0) - (1 if self.window.right_pressed else 0))
            # img = self.env.render("rgb_array")
            # self.window.imshow(img)
        rgb_ary_len = np.array(rgb_array.shape).prod()
        return rgb_array, rgb_ary_len

    def reset(self):
        obs = self.env.reset()
        rgb_array, rgb_ary_len = self._render_fps()
        return np.concatenate((rgb_array.reshape(rgb_ary_len), obs))
