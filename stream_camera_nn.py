import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import os
from pybullet_env import CutterEnvBase, check_input_variance
from stable_baselines3 import PPO
# import cv2
import time
from PIL import Image


class RealCutterEnv(CutterEnvBase):
    def __init__(self, width, height, grayscale=False, use_net=False, use_seg=False, use_depth=False, use_flow=False, use_last_frame=False,
                 use_gui=False):
        super(RealCutterEnv, self).__init__(width, height, grayscale=grayscale, use_net=use_net, use_seg=use_seg, use_depth=use_depth,
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
        raise Exception('Resizing temporarily disabled')
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

class CutterEnvFromFiles(CutterEnvBase):
    def __init__(self, width, height, img_src, grayscale=False, use_net=False, use_seg=False, use_depth=False, use_flow=False,
                 use_last_frame=False, use_gui=False):
        super(CutterEnvFromFiles, self).__init__(width, height, grayscale=grayscale, use_net=use_net, use_seg=use_seg,
                                            use_depth=use_depth,
                                            use_flow=use_flow, use_last_frame=use_last_frame, use_gui=use_gui)
        self.img_files = [os.path.join(img_src, f) for f in sorted(os.listdir(img_src), key=lambda x: int(x.split('.')[0])) if f.endswith('.png')]
        self.counter = 0


    def get_images(self):
        rgb_img = np.array(Image.open(self.img_files[self.counter]))
        depth_img = None
        seg_img = None

        self.counter = min(self.counter + 1, len(self.img_files) - 1)

        return rgb_img, depth_img, seg_img

    def step(self, action):
        return self.get_obs(), 0.0, False, {}









if __name__ == '__main__':

    action = 'server'
    # action = 'eval'
    save_img = False
    forward_speed = 0.03

    width = 424
    height = 240
    grayscale = False
    use_net = True
    use_seg = False
    use_depth = False
    use_flow = False
    use_last_frame = False

    # # TESTING
    # img_src = 'images'
    # env = CutterEnvFromFiles(width, height, img_src, grayscale=grayscale, use_net=use_net, use_seg=use_seg, use_depth=use_depth,
    #                     use_flow=use_flow, use_last_frame=use_last_frame, use_gui=True)
    # counter = 0
    # while True:
    #     counter += 1
    #     if counter < 10:
    #         env.counter = 0
    #     env.get_obs()
    #     time.sleep(0.1)
    #
    # raise Exception()

    try:
        env = RealCutterEnv(width, height, grayscale=grayscale, use_net=use_net, use_seg=use_seg, use_depth=use_depth,
                            use_flow=use_flow, use_last_frame=use_last_frame, use_gui=True)
        model = PPO("CnnPolicy", env, verbose=1)
        # model_file = '{}.model'.format(env.model_name)
        model_file = 'best_model_0_6.zip'
        if os.path.exists(model_file):
            model = model.load(model_file)
        else:
            raise Exception()


        if action == 'eval':
            num_frames = 5000
            start = time.time()
            file_id = str(int(start))

            action_hist = []
            try:
                for i in range(num_frames):

                    obs = env.get_obs()
                    action = model.predict(obs, deterministic=True)[0]
                    env.set_action(action)
                    action_hist.append(action)
                    print(action)
                    env.set_title('Frame {}'.format(i))

                    if save_img:
                        obs = obs[:,:,:3]
                        im = Image.fromarray(obs)
                        save_path = os.path.join('images', f'{i}.png')
                        im.save(save_path)
                        time.sleep(0.5)
            finally:
                # np.save('data/hist_{}.npy'.format(int(file_id)), np.array(action_hist))
                print('Saved!')


            # check_input_variance(model, obs, output=True)

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

            ADDRESS = '169.254.63.255'
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

                action_history = []

                try:
                    while True:
                        obs = env.get_obs()
                        action = model.predict(obs, deterministic=True)[0]
                        env.set_action(action)
                        action_history.append(action)
                        # array = np.array([0, 0, -0.01])
                        # TEMPORARILY APPLY NEGATIVE DUE TO FRAME ISSUE
                        array = np.array([-action[0], -action[1], 1]) * forward_speed
                        connection.sendall(array.tostring())
                finally:
                    connection.close()
                    ts = time.time()
                    np.save('data/hist_real_{}.npy'.format(int(ts)), np.array(action_history))
                    print('Connection terminated, waiting for new connection...')
        else:
            raise NotImplementedError

    finally:
        env.shutdown()