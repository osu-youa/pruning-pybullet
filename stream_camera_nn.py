import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import os
from pybullet_env import CutterEnvBase, check_input_variance
from stable_baselines3 import PPO
import cv2
import time


class RealCutterEnv(CutterEnvBase):
    def __init__(self, width, height, grayscale=False, use_seg=False, use_depth=False, use_flow=False, use_last_frame=False,
                 use_gui=False):
        super(RealCutterEnv, self).__init__(width, height, grayscale=grayscale, use_seg=use_seg, use_depth=use_depth,
                                            use_flow=use_flow, use_last_frame=use_last_frame, use_gui=use_gui)

        # Setup the camera
        self.pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 30)
        if self.use_depth:
            config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
            self.align = rs.align(rs.stream.color)
        self.pipe.start(config)

    def get_images(self):
        frames = self.pipe.wait_for_frames()
        depth_img = None
        if self.use_depth:
            frames = self.align.process(frames)
            depth_img = self.resize(np.asanyarray(frames.get_depth_frame().get_data()))
        rgb_img = self.resize(np.asanyarray(frames.get_color_frame().get_data()))

        seg_img = None
        if self.use_seg:
            raise NotImplementedError("Segmentation masks not defined")

        return rgb_img, depth_img, seg_img

    def resize(self, img):
        if tuple(img.shape[:2]) == (self.height, self.width):
            return img
        return cv2.resize(img, (self.width, self.height))


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

    # action = 'server'
    action = 'eval'
    forward_speed = 0.05

    width = 424
    height = 240
    grayscale = False
    use_seg = False
    use_depth = False
    use_flow = False
    use_last_frame = True


    try:
        env = RealCutterEnv(width, height, grayscale=grayscale, use_seg=use_seg, use_depth=use_depth,
                            use_flow=use_flow, use_last_frame=use_last_frame, use_gui=True)
        model = PPO("CnnPolicy", env, verbose=1)
        # model_file = '{}.model'.format(env.model_name)
        model_file = 'best_model.zip'
        if os.path.exists(model_file):
            model = model.load(model_file)


        if action == 'eval':
            num_frames = 500
            start = time.time()

            action_hist = []

            for i in range(num_frames):
                obs = env.get_obs()
                action = model.predict(obs)[0]
                action_hist.append(action)
                print(action)
                env.set_title('Frame {}'.format(i))

                check_input_variance(model, obs, output=True)

            end = time.time()
            print('Ran at {:.2f} fps'.format(num_frames / (end - start)))

            plt.ioff()
            plt.figure()
            import pdb
            pdb.set_trace()
            plt.plot(np.arange(len(action_hist)), np.array(action_hist))
            plt.show()
        elif action == 'server':

            import socket

            ADDRESS = '192.168.2.227'
            PORT = 10000

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            address = (ADDRESS, PORT)
            print('Starting up server on {}:{}'.format(*address))
            sock.bind(address)
            sock.listen(1)

            while True:
                print('Waiting for connection')
                connection, client_address = sock.accept()
                print('Connection accepted!')

                try:
                    while True:
                        obs = env.get_obs()
                        action = model.predict(obs, deterministic=True)[0]
                        array = np.array([action[0], action[1], 0]) * forward_speed
                        connection.sendall(array.tostring())
                        time.sleep(0.1)
                finally:
                    connection.close()
                    print('Connection terminated, waiting for new connection...')
        else:
            raise NotImplementedError

    finally:
        env.shutdown()