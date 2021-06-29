import pybullet as pb
import time
import pybullet_data
import numpy as np
import os
from helloworld import URDFRobot
from utils import PerlinNoiseBuffer, overlay_noise

from stable_baselines3 import PPO

import gym
from gym import spaces


class CutterEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, use_seg=False, use_depth=False, max_vel=0.075, action_freq=24, max_elapsed_time=3.0,
                 min_reward_dist=0.10, use_gui=False, debug=False):
        super(CutterEnv, self).__init__()

        # Initialize gym parameters
        num_channels = (2 if use_seg else 3) + (1 if use_depth else 0)
        self.action_space = spaces.Box(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]), dtype=np.float32)    # LR, UD, Terminate
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, num_channels), dtype=np.uint8)
        self.model_name = 'model_{mode}{use_depth}'.format(mode='seg' if use_seg else 'rgb', use_depth='_depth' if use_depth else '')

        # Configuration parameters
        self.debug = debug
        self.width = width
        self.height = height
        self.use_seg = use_seg
        self.use_depth = use_depth
        self.action_freq = action_freq
        self.max_elapsed_time = max_elapsed_time
        self.min_reward_dist = min_reward_dist
        self.max_vel = max_vel
        self.current_camera_tf = np.identity(4)
        self.mesh_num_points = 100

        # State parameters
        self.target_id = 0
        self.target_tf = np.identity(4)
        self.elapsed_time = 0.0
        self.mesh_points = {}

        # Simulation tools - Some are only for seg masks!
        self.noise_buffer = PerlinNoiseBuffer(width, height, rectangle_size=30, buffer_size=50)
        self.max_depth_sigma = 5.0
        self.max_tree_sigma = 5.0
        self.max_noise_alpha = 0.3
        self.current_depth_noise = (None, None, None)       # Sigma, noise image, noise alpha
        self.current_tree_noise = (None, None, None)


        # Setup Pybullet simulation

        self.gui = use_gui
        self.client_id = pb.connect(pb.GUI if use_gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        pb.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        plane_id = pb.loadURDF("plane.urdf", physicsClientId=self.client_id)
        dirt_texture = pb.loadTexture('textures/dirt.png')
        pb.changeVisualShape(objectUniqueId=plane_id, linkIndex=-1, textureUniqueId=dirt_texture, physicsClientId=self.client_id)


        arm_location = os.path.join('robots', 'ur5e_cutter_new_calibrated_precise.urdf')
        home_joints = [-1.5708, -2.2689, -1.3963, 0.52360, 1.5708, 3.14159]
        self.robot = URDFRobot(arm_location, [0, 0, 0.02], pb.getQuaternionFromEuler([0, 0, 0]))
        self.robot.reset_joint_states(home_joints)
        self.start_orientation = self.robot.get_link_kinematics('cutpoint')[1]
        self.proj_mat = pb.computeProjectionMatrixFOV(
            fov=60.0, aspect = width / height, nearVal=0.01,
            farVal=3.0)
        self.robot.attach_ghost_body_from_file('robots/ur5e/collision/cutter-mouth-collision.stl',
                                               'mouth', 'cutpoint', rpy=[0, 0, 3.1416])

        self.tool_to_camera_offset = np.array([0.0, 0.075, 0.0, 1.0])
        tool_tf = self.robot.get_link_kinematics('wrist_3_link-tool0_fixed_joint', as_matrix=True)
        self.ideal_camera_pos = (tool_tf @ self.tool_to_camera_offset)[:3]
        ideal_view_matrix = np.reshape(pb.computeViewMatrix(cameraEyePosition = self.ideal_camera_pos,
                                                            cameraTargetPosition=self.robot.get_link_kinematics('cutpoint')[0],
                                                            cameraUpVector=[0,0,1]), (4,4)).T
        self.ideal_tool_camera_tf = np.linalg.inv(tool_tf) @ np.linalg.inv(ideal_view_matrix)

        scaling = 1.35
        tree_y = 0.875
        self.tree = URDFRobot('models/trellis-model.urdf', basePosition=[0, tree_y, 0.02 * scaling],
                              baseOrientation=[0, 0, 0.7071, 0.7071], globalScaling=scaling,
                              useFixedBase=True)

        wall_id = pb.loadURDF("models/wall.urdf", physicsClientId=self.client_id, basePosition=[0, tree_y + 2.0, 0])
        pb.changeVisualShape(objectUniqueId=wall_id, linkIndex=-1, textureUniqueId=pb.loadTexture('/textures/trees.png'),
                             physicsClientId=self.client_id)

        self.start_state = pb.saveState(physicsClientId=self.client_id)

    def step(self, action, realtime=False):
        # Execute one time step within the environment

        self.target_tf = self.robot.get_link_kinematics('cutpoint', as_matrix=True)

        horizontal, vertical, terminate = action
        terminate = terminate > 0
        if terminate:
            return self.get_obs(), self.get_reward(True), True, {}

        step = self.action_freq * self.max_vel / 240
        delta = np.array([horizontal * step, vertical * step, step, 1.0], dtype=np.float32)
        prev_target_pos = self.target_tf[:3,3]
        new_target_pos = (self.target_tf @ delta)[:3]
        diff = new_target_pos - prev_target_pos

        # Move the arm in the environment
        self.elapsed_time += self.action_freq / 240
        for i in range(self.action_freq):
            target_pos = prev_target_pos + (i + 1) / self.action_freq * diff
            self.target_tf[:3, 3] = target_pos
            ik = self.robot.solve_end_effector_ik('cutpoint', target_pos, self.start_orientation)
            self.robot.set_control_target(ik)
            pb.stepSimulation(physicsClientId=self.client_id)
            if realtime:
                time.sleep(1.0/240)
                # self.get_obs()

        done = self.is_done()
        return self.get_obs(), self.get_reward(done), done, {}


    def get_obs(self):
        # Grab the new image observation
        view_matrix = self.robot.get_link_kinematics('wrist_3_link-tool0_fixed_joint', as_matrix=True) @ self.current_camera_tf

        _, _, rgb_img, raw_depth_img, seg_img = pb.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=np.linalg.inv(view_matrix).T.reshape(-1),
            projectionMatrix=self.proj_mat,
            renderer=pb.ER_TINY_RENDERER,
            physicsClientId=self.client_id
        )


        # import matplotlib.pyplot as plt
        # from scipy.ndimage import sobel
        # from skimage.color import rgb2hsv
        # rgb_img = rgb_img[:,:,:3]
        # greyscale = rgb_img.mean(axis=2)
        #
        # sobel_1 = sobel(greyscale, axis=0)
        # sobel_2 = sobel(greyscale, axis=1)
        #
        # final = sobel_1 + sobel_2
        # final = (255 * (final - np.min(final)) / (np.max(final) - np.min(final))).astype(np.uint8)
        #
        # plt.imshow(final, cmap='gray')
        # plt.show()
        #
        # import pdb
        # pdb.set_trace()
        #
        # hsv_image = rgb2hsv(rgb_img)
        # final = np.zeros(hsv_image.shape[:2])
        # for channel in range(3):
        #     final = final + sobel(hsv_image[:,:,channel], axis=0)
        #     final = final + sobel(hsv_image[:, :, channel], axis=1)
        # final = (255 * (final - np.min(final)) / (np.max(final) - np.min(final))).astype(np.uint8)
        # plt.imshow(final, cmap='gray')
        # plt.show()
        #
        # import pdb
        # pdb.set_trace()

        depth_img = None
        if self.use_depth:
            depth_img = overlay_noise(raw_depth_img, *self.current_depth_noise, convert_to_uint8=True)

        if self.use_seg:
            tree_layer_raw = (seg_img == self.tree.robot_id).astype(np.float64)
            tree_layer = overlay_noise(tree_layer_raw, *self.current_tree_noise, convert_to_uint8=True)
            robot_layer = ((seg_img == self.robot.robot_id) * 255).astype(np.uint8)

            layers = [tree_layer, robot_layer]
            if depth_img is not None:
                layers.append(depth_img)
            base_img = np.stack(layers, axis=2)
        else:
            if depth_img is not None:
                rgb_img[:, :, 3] = depth_img
            else:
                rgb_img = rgb_img[:,:,:3]


            base_img = rgb_img[:, :, :3]

        return base_img

    def get_cutter_dist(self):
        cutter_loc = self.robot.get_link_kinematics('cutpoint', use_com_frame=False)[0]
        target_loc = self.tree.get_link_kinematics(self.target_id)[0]

        d = np.linalg.norm(np.array(target_loc) - np.array(cutter_loc))
        return d

    def target_collision_points(self, link_id=None):

        """
        WARNING! THIS ONLY WORKS ON CONVEX SHAPES!
        """

        if link_id is None:
            link_id = self.target_id
        else:
            link_id = self.tree.convert_link_name(link_id)

        try:
            return self.mesh_points[link_id]
        except KeyError:

            vertices = np.array(pb.getMeshData(self.tree.robot_id, link_id)[1])
            choice_1 = np.random.choice(len(vertices), self.mesh_num_points)
            choice_2 = np.random.choice(len(vertices), self.mesh_num_points)
            wgt = np.random.uniform(0, 1, size=self.mesh_num_points)
            pts = vertices[choice_1] + (vertices[choice_2] - vertices[choice_1]) * wgt[:, np.newaxis]
            self.mesh_points[link_id] = pts
            return pts


    def get_reward(self, done=False):

        link_tf = self.tree.get_link_kinematics(self.target_id, as_matrix=True)
        in_mouth = self.robot.query_ghost_body_collision('mouth', self.target_collision_points(),
                                                         point_frame_tf=link_tf, plot_debug=False)

        if self.debug and in_mouth:
            print('[DEBUG] Branch is in mouth!')

        d = self.get_cutter_dist()
        dist_proportion = max(1 - d / self.min_reward_dist, 0.0)
        if done:
            if not in_mouth:
                reward = 0.0
            else:
                reward = (self.max_elapsed_time - self.elapsed_time) * dist_proportion
        else:
            ts = self.action_freq / 240.0
            reward = dist_proportion * ts * (1.0 if in_mouth else 0.25)

        if self.debug:
            print('Obtained reward: {:.3f}'.format(reward))

        return reward


    def is_done(self):
        return self.elapsed_time >= self.max_elapsed_time

    def reset(self):

        if self.elapsed_time:
            print('Reset! (Elapsed time {:.2f}s)'.format(self.elapsed_time))

        self.elapsed_time = 0.0
        pb.restoreState(stateId=self.start_state, physicsClientId=self.client_id)

        # Reset the image noise parameters

        self.current_depth_noise = (np.random.uniform(0, self.max_depth_sigma), self.noise_buffer.get_random(), np.random.uniform(0, self.max_noise_alpha))
        self.current_tree_noise = (np.random.uniform(0, self.max_tree_sigma), self.noise_buffer.get_random(), np.random.uniform(0, self.max_noise_alpha))

        # Compute a random for the camera transform

        xyz_noise = np.random.uniform(-0.0025, 0.0025, size=3)
        rpy_noise = np.random.uniform(-np.radians(2.5), np.radians(2.5), size=3)
        quat = pb.getQuaternionFromEuler(rpy_noise)
        noise_tf = np.identity(4)
        noise_tf[:3,3] = xyz_noise
        noise_tf[:3,:3] = np.reshape(pb.getMatrixFromQuaternion(quat), (3,3))
        self.current_camera_tf = self.ideal_tool_camera_tf @ noise_tf

        # Pick a target on the tree
        self.target_id = np.random.randint(len(self.tree.joint_names_to_ids))
        target_pos = self.tree.get_link_kinematics(self.target_id)[0]

        # Find valid IKs for it
        robot_start_pos = target_pos - np.array([0, np.random.uniform(0.08, 0.12), 0])
        robot_start_pos += np.random.uniform(-0.05, 0.05, 3) * np.array([1, 0, 1])

        ik = self.robot.solve_end_effector_ik('cutpoint', robot_start_pos, self.start_orientation)
        self.robot.reset_joint_states(ik)

        self.target_tf = self.robot.get_link_kinematics('cutpoint', as_matrix=True)

        return self.get_obs()

    def render(self, mode='human', close=False):
        print('Last dist: {:.3f}'.format(self.get_cutter_dist()))

if __name__ == '__main__':

    action = 'eval'
    # action = 'train'
    use_seg = True
    use_depth = True

    if action == 'train':
        env = CutterEnv(159, 90, use_seg=use_seg, use_depth=use_depth, use_gui=False, max_elapsed_time=2.5, max_vel=0.05, debug=False)
        model_file = '{}.model'.format(env.model_name)
        model = PPO("CnnPolicy", env, verbose=1, device='auto')

        print('Learning...')

        model.learn(total_timesteps=50000)
        print('Done learning!')
        model.save(model_file)

    elif action == 'eval':
        env = CutterEnv(159, 90, use_seg=use_seg, use_depth=use_depth, use_gui=True, max_elapsed_time=2.5, max_vel=0.05, debug=True)
        model = PPO("CnnPolicy", env, verbose=1)
        model_file = '{}.model'.format(env.model_name)
        if os.path.exists(model_file):
            model = model.load(model_file)
        obs = env.reset()
        all_dists = []
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            # action = env.action_space.sample()
            # if (i + 1) % 10:
            #     action = np.array([0.0, 0.0, -1.0])
            # else:
            #     print('Terminating')
            #     action = np.array([0.0, 0.0, 1.0])

            obs, reward, done, info = env.step(action, realtime=True)
            # env.render()
            if done:
                all_dists.append(env.get_cutter_dist())
                obs = env.reset()

        print('Average terminal dist: {:.3f}'.format(np.mean(all_dists)))
        env.close()
    else:
        raise NotImplementedError("Unknown action {}".format(action))
