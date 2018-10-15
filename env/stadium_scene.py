# This files is forked from
# https://github.com/openai/roboschool/blob/911f7cd55e6169097fad8735e5424cf436e3c0ad/roboschool/scene_stadium.py
# LICENSE: https://github.com/openai/roboschool/blob/c1cb98386f73035d02700b592c54501c80c348f9/LICENSE.md
import os
from roboschool.scene_abstract import Scene, cpp_household


class CustomStadiumScene(Scene):
    "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
    multiplayer = False
    zero_at_running_strip_start_line = True   # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    stadium_halflen   = 105*0.25    # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50*0.25     # FOOBALL_FIELD_HALFWID

    def episode_restart(self):
        Scene.episode_restart(self)   # contains cpp_world.clean_everything()
        stadium_pose = cpp_household.Pose()
        if self.zero_at_running_strip_start_line:
            stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants
        self.stadium = self.cpp_world.load_thingy(
            os.path.join(os.path.dirname(__file__), "models_outdoor/stadium/stadium1.obj"),
            stadium_pose, 1.0, 0, 0xFFFFFF, True)
        self.ground_plane_mjcf = self.cpp_world.load_mjcf(os.path.join(os.path.dirname(__file__), "mujoco_assets/ground_plane.xml"))

