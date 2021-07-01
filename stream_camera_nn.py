import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import os
from pybullet_env import CutterEnvBase
from stable_baselines3 import PPO

class RealCutterEnv(CutterEnvBase):
    def __init__(self, width, height, cutter_mask, use_seg=True, use_depth=True):
        super(RealCutterEnv, self).__init__(width, height, use_seg=use_seg, use_depth=use_depth)

        # Setup the camera
        self.pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        profile = self.pipe.start(config)

        # Config stuff
        self.cutter_mask = cutter_mask
        self.nearval = 0.01
        self.farval = 3.0

        # For rendering
        self.last_rgb = None
        self.last_depth = None

    def get_obs(self):
        frames = self.align.process(self.pipe.wait_for_frames())
        depth_img = frames.get_depth_frame()
        rgb_img = frames.get_color_frame()



        if not depth_img:
            raise Exception('No depth image obtained from camera!')
        if not rgb_img:
            raise Exception('No RGB image obtained from camera!')

        self.last_rgb = np.asanyarray(rgb_img.get_data())
        self.last_depth = np.asanyarray(depth_img.get_data())

    def step(self, action):
        horizontal, vertical, terminate = action
        done = terminate > 0
        return self.get_obs(), 0.0, done, {}

    def shutdown(self):
        self.pipe.stop()

    def render(self, mode='human'):
        plt.imshow(self.last_rgb)
        plt.show()
        plt.imshow(self.last_depth, cmap='gray')
        plt.show()

if __name__ == '__main__':
    try:
        env = RealCutterEnv(159, 90, use_seg=True, use_depth=True, cutter_mask=None)
        model = PPO("CnnPolicy", env, verbose=1)
        model_file = '{}.model'.format(env.model_name)
        if os.path.exists(model_file):
            model = model.load(model_file)

        for i in range(0, 100):

            env.get_obs()
            env.render()

    finally:
        env.shutdown()