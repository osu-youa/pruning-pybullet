import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import os
from pybullet_env import CutterEnv
from stable_baselines3 import PPO

env = CutterEnv(159, 90, use_seg=True, use_depth=True, use_gui=True, max_elapsed_time=2.5, max_vel=0.05, debug=True)
model = PPO("CnnPolicy", env, verbose=1)
model_file = '{}.model'.format(env.model_name)
if os.path.exists(model_file):
    model = model.load(model_file)

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 30)
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
align = rs.align(rs.stream.color)

profile = pipe.start(config)
try:
    for i in range(0, 100):
        frames = align.process(pipe.wait_for_frames())
        depth_img = frames.get_depth_frame()
        rgb_img = frames.get_color_frame()
        if not depth_img or not rgb_img:
            continue



        depth_img = np.asanyarray(depth_img.get_data())
        rgb_img = np.asanyarray(rgb_img.get_data())


        #
        plt.imshow(rgb_img)
        plt.show()
        plt.imshow(depth_img, cmap='gray')
        plt.show()
finally:
    pipe.stop()