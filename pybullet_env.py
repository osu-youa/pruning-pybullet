import pybullet as pb
import time
import pybullet_data
import numpy as np
import os
from helloworld import URDFRobot
# from utils import PerlinNoiseBuffer, overlay_noise
import random
import pickle
from scipy.spatial.transform import Rotation
from itertools import product
from PIL import Image, ImageEnhance

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecTransposeImage, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from functools import partial

import matplotlib.pyplot as plt

import gym
from gym import spaces


def homog_tf(tf, pt):
    homog_pt = np.ones(4)
    homog_pt[:3] = pt
    return (tf @ homog_pt)[:3]

def pose_to_tf(pos, quat):
    tf = np.identity(4)
    tf[:3,3] = pos
    tf[:3,:3] = Rotation.from_quat(quat).as_matrix()
    return tf


class CutterEnvBase(gym.Env):
    def __init__(self, width, height, grayscale=False, use_seg=False, use_depth=False, use_flow=False, use_last_frame=False,
                 use_gui=False):
        super(CutterEnvBase, self).__init__()

        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.use_seg = use_seg
        self.use_depth = use_depth
        self.use_flow = use_flow
        self.use_last_frame = use_last_frame
        self.use_gui = use_gui

        self.last_grayscale = None

        # Initialize gym parameters
        num_channels = (2 if use_seg else (1 if grayscale else 3)) + (1 if use_depth else 0) + (1 if use_flow else 0) + (1 if use_last_frame else 0)
        self.action_space = spaces.Box(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]),
                                       dtype=np.float32)  # LR, UD, Terminate
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, num_channels), dtype=np.uint8)
        self.model_name = 'model_{mode}{use_depth}{flow}{uselast}'.format(mode='seg' if use_seg else ('gray' if grayscale else 'rgb'),
                                                           use_depth='_depth' if use_depth else '', flow='_flow' if use_flow else '',
                                                                          uselast='_uselastframe' if use_last_frame else '')

        if self.use_gui:
            plt.ion()
            self.fig = plt.figure()
            self.axs = [self.fig.add_subplot(ind) for ind in [121, 122]]
            self.imgs = [ax.imshow(np.zeros((height, width), dtype=np.uint8)) for ax in self.axs]
            self.fig.canvas.draw()
            plt.pause(0.01)

    def step(self, action):
        raise NotImplementedError()

    def get_obs(self):
        # Grab the new image observation
        rgb_img, depth_img, seg_img = self.get_images()
        grayscale = rgb_img.mean(axis=2).astype(np.uint8)
        layers = []

        if self.grayscale:
            layers.append(grayscale)
        else:
            layers.append(rgb_img)

        if depth_img is not None:
            layers.append(depth_img)

        if seg_img is not None:
            layers.append(seg_img)

        if self.use_flow:
            if self.last_grayscale is None:
                flow_img = np.zeros((self.height, self.width), dtype=np.uint8)
            else:
                import cv2
                flow = cv2.calcOpticalFlowFarneback(prev=self.last_grayscale, next=grayscale, flow=None,
                                                    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                                    poly_n=5, poly_sigma=1.1, flags=0)
                flow_mag = np.linalg.norm(flow, axis=2)
                flow_img = (255 * flow_mag / flow_mag.max()).astype(np.uint8)

                # if self.debug:
                #     import matplotlib.pyplot as plt
                #     # plt.imshow(grayscale, cmap='gray')
                #     # plt.show()
                #     plt.imshow(mask, cmap='gray')
                #     plt.show()

            layers.append(flow_img)
            if self.use_gui:
                self.imgs[1].set_data(np.dstack([flow_img] * 3))

        if self.use_last_frame:
            layers.append(self.last_grayscale if self.last_grayscale is not None else grayscale)

        self.last_grayscale = grayscale

        if self.use_gui:
            self.imgs[0].set_data(rgb_img)
            self.fig.canvas.draw()
            plt.pause(0.01)

        return np.dstack(layers)

    def reset(self):
        raise NotImplementedError()

    def render(self, mode='human'):
        raise NotImplementedError()

    def get_images(self):
        # Returns RGB, depth, and segmentation images
        raise NotImplementedError()



class CutterEnv(CutterEnvBase):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, grayscale=False, use_seg=False, use_depth=False, use_flow=False, use_last_frame=False, max_vel=0.075, action_freq=24, max_elapsed_time=3.0,
                 min_reward_dist=0.10, use_gui=False, debug=False):
        super(CutterEnv, self).__init__(width, height, grayscale=grayscale, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow, use_last_frame=use_last_frame,
                                        use_gui=use_gui)

        # Configuration parameters
        self.debug = debug
        self.action_freq = action_freq
        self.max_elapsed_time = max_elapsed_time
        self.min_reward_dist = min_reward_dist
        self.max_vel = max_vel
        self.current_camera_tf = np.identity(4)
        self.mesh_num_points = 100
        self.accel_threshold = 0.50         # Vector magnitude where [X,Y] are in range [-1, 1]
        self.accel_penalty = 0.50           # For every unit accel exceeding the threshold, reduce base reward by given proportion
        self.frames_per_img = 8             # Corresponds to 30 Hz

        # State parameters
        self.target_pose = None             # What pose should the cutter end up in?
        self.target_tree = None             # Which tree model is in front?
        self.target_id = None               # Which of the side branches are we aiming at on the target tree?
        self.target_tf = np.identity(4)     # What is the next waypoint for the cutter?
        self.elapsed_time = 0.0
        self.in_mouth = False               # Is the branch currently inside of the cutter mouth?
        self.failure = False                # Is the branch currently in the failure region?
        self.mesh_points = {}
        self.last_command = np.zeros(2)
        self.lighting = None                # Location of light source
        self.contrast = None                # Contrast adjustment factor

        # Simulation tools - Some are only for seg masks!
        # self.noise_buffer = PerlinNoiseBuffer(width, height, rectangle_size=30, buffer_size=50)
        self.max_depth_sigma = 5.0
        self.max_tree_sigma = 5.0
        self.max_noise_alpha = 0.3
        # self.current_depth_noise = (None, None, None)       # Sigma, noise image, noise alpha
        # self.current_tree_noise = (None, None, None)

        # Setup Pybullet simulation

        self.client_id = pb.connect(pb.GUI if self.use_gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        pb.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        self.plane_id = pb.loadURDF("plane.urdf", physicsClientId=self.client_id)
        plane_texture_root = os.path.join('textures', 'floor')
        self.plane_textures = [pb.loadTexture(os.path.join(plane_texture_root, file), physicsClientId=self.client_id) for file in os.listdir(plane_texture_root)]

        arm_location = os.path.join('robots', 'ur5e_cutter_new_calibrated_precise.urdf')
        home_joints = [-1.5708, -2.2689, -1.3963, 0.52360, 1.5708, 3.14159]
        self.robot = URDFRobot(arm_location, [0, 0, 0.02], [0, 0, 0, 1], physicsClientId=self.client_id)
        self.robot.reset_joint_states(home_joints)
        self.start_orientation = self.robot.get_link_kinematics('cutpoint')[1]
        self.proj_mat = pb.computeProjectionMatrixFOV(
            fov=60.0, aspect = width / height, nearVal=0.01,
            farVal=3.0)
        self.robot.attach_ghost_body_from_file('robots/ur5e/collision/cutter-mouth-collision.stl',
                                               'mouth', 'cutpoint', rpy=[0, 0, 3.1416])
        self.robot.attach_ghost_body_from_file('robots/ur5e/collision/cutter-failure-zone.stl',
                                               'failure', 'cutpoint', rpy=[0, 0, 3.1416])

        self.tool_to_camera_offset = np.array([0.0, 0.075, 0.0, 1.0])
        tool_tf = self.robot.get_link_kinematics('wrist_3_link-tool0_fixed_joint', as_matrix=True)
        self.ideal_camera_pos = (tool_tf @ self.tool_to_camera_offset)[:3]
        ideal_view_matrix = np.reshape(pb.computeViewMatrix(cameraEyePosition = self.ideal_camera_pos,
                                                            cameraTargetPosition=self.robot.get_link_kinematics('cutpoint')[0],
                                                            cameraUpVector=[0,0,1]), (4,4)).T
        self.ideal_tool_camera_tf = np.linalg.inv(tool_tf) @ np.linalg.inv(ideal_view_matrix)

        # Create pose database
        target_xs = np.linspace(-0.3, 0.3, num=25, endpoint=True)
        target_ys = np.linspace(0.75, 0.95, num=17, endpoint=True)
        target_zs = np.linspace(0.8, 1.3, num=25, endpoint=True)
        all_poses = [[x, y, z] + list(self.start_orientation) for x, y, z in product(target_xs, target_ys, target_zs)]
        self.poses = self.robot.determine_reachable_target_poses('cutpoint', all_poses, home_joints)

        # wall_id = pb.loadURDF("models/wall.urdf", physicsClientId=self.client_id, basePosition=[0, tree_y + 2.0, 0])
        # pb.changeVisualShape(objectUniqueId=wall_id, linkIndex=-1, textureUniqueId=pb.loadTexture('/textures/trees.png'),
        #                      physicsClientId=self.client_id)

        # Load trees - Put them in background out of sight of the camera

        self.tree_model_metadata = {}
        tree_models_directory = os.path.join('models', 'trees')
        tree_model_files = [x for x in os.listdir(tree_models_directory) if x.endswith('.obj') and not '-' in x]
        if len(tree_model_files) < 3:
            tree_model_files = tree_model_files * 3

        for file in tree_model_files:
            path = os.path.join(tree_models_directory, file)
            viz = pb.createVisualShape(shapeType=pb.GEOM_MESH, fileName=path, physicsClientId=self.client_id)
            col = pb.createCollisionShape(shapeType=pb.GEOM_MESH, fileName=path.replace('.obj', '-collision.obj'), physicsClientId=self.client_id)

            tree_id = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=viz, baseCollisionShapeIndex=col, basePosition=[-10, 5, 0],
                                         baseOrientation=[0.7071, 0, 0, 0.7071], physicsClientId=self.client_id)
            annotation_path = path.replace('.obj', '.annotations')
            with open(annotation_path, 'rb') as fh:
                annotations = pickle.load(fh)

            self.tree_model_metadata[tree_id] = annotations

        self.start_state = pb.saveState(physicsClientId=self.client_id)


    def step(self, action, realtime=False):
        # Execute one time step within the environment

        self.target_tf = self.robot.get_link_kinematics('cutpoint', as_matrix=True)

        horizontal, vertical, terminate = action
        vel_command = np.array([horizontal, vertical])
        terminate = terminate > 0
        if terminate:
            return self.get_obs(), self.get_reward(vel_command, True), True, {}

        step = self.action_freq * self.max_vel / 240
        delta = np.array([horizontal * step, vertical * step, step, 1.0], dtype=np.float32)
        prev_target_pos = self.target_tf[:3,3]
        new_target_pos = (self.target_tf @ delta)[:3]
        diff = new_target_pos - prev_target_pos

        # Move the arm in the environment
        self.elapsed_time += self.action_freq / 240
        img_update_frames = range(self.action_freq, 0, -self.frames_per_img)
        for i in range(self.action_freq):
            target_pos = prev_target_pos + (i + 1) / self.action_freq * diff
            self.target_tf[:3, 3] = target_pos
            ik = self.robot.solve_end_effector_ik('cutpoint', target_pos, self.start_orientation)
            self.robot.set_control_target(ik)
            pb.stepSimulation(physicsClientId=self.client_id)

            if i in img_update_frames:
                self.get_obs()

            if realtime:
                time.sleep(1.0/240)

        # Compute collisions with ghost bodies
        base_pos, base_quat = pb.getBasePositionAndOrientation(self.target_tree, physicsClientId=self.client_id)
        base_tf = pose_to_tf(base_pos, base_quat)
        tree_pts = self.tree_model_metadata[self.target_tree][self.target_id]['points']
        self.in_mouth = self.robot.query_ghost_body_collision('mouth', tree_pts,
                                                         point_frame_tf=base_tf, plot_debug=False)
        self.failure = not self.in_mouth and self.robot.query_ghost_body_collision('failure', tree_pts,
                                                                         point_frame_tf=base_tf, plot_debug=False)

        done = self.is_done()
        reward = self.get_reward(vel_command, done)

        self.last_command = vel_command
        return self.get_obs(), reward, done, {}

    def get_images(self):
        view_matrix = self.robot.get_link_kinematics('wrist_3_link-tool0_fixed_joint',
                                                     as_matrix=True) @ self.current_camera_tf
        _, _, rgb_img, raw_depth_img, raw_seg_img = pb.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=np.linalg.inv(view_matrix).T.reshape(-1),
            projectionMatrix=self.proj_mat,
            renderer=pb.ER_TINY_RENDERER,
            lightDirection=self.lighting,
            shadow=True,
            physicsClientId=self.client_id
        )

        rgb_img = rgb_img[:, :, :3]
        pil_img = Image.fromarray(rgb_img, 'RGB')
        enhancer = ImageEnhance.Contrast(pil_img)
        rgb_img = np.asarray(enhancer.enhance(self.contrast))

        depth_img = None
        if self.use_depth:
            depth_img = raw_depth_img
            # depth_img = overlay_noise(raw_depth_img, *self.current_depth_noise, convert_to_uint8=True)

        seg_img = None
        if self.use_seg:
            raise NotImplementedError("Segmentation logic needs to be reimplemented")
            tree_layer_raw = (raw_seg_img == self.tree.robot_id).astype(np.float64)
            # tree_layer = overlay_noise(tree_layer_raw, *self.current_tree_noise, convert_to_uint8=True)
            tree_layer = (tree_layer_raw * 255).astype(np.uint8)
            robot_layer = ((raw_seg_img == self.robot.robot_id) * 255).astype(np.uint8)
            seg_img = np.dstack([tree_layer, robot_layer])

        return rgb_img, depth_img, seg_img

    @property
    def current_tree_base_tf(self):
        base_pos, base_quat = pb.getBasePositionAndOrientation(self.target_tree, physicsClientId=self.client_id)
        return pose_to_tf(base_pos, base_quat)

    @property
    def current_target_position(self):
        target_loc = self.tree_model_metadata[self.target_tree][self.target_id]['position']
        return homog_tf(self.current_tree_base_tf, target_loc)

    def get_cutter_dist(self):
        cutter_loc = self.robot.get_link_kinematics('cutpoint', use_com_frame=False)[0]
        target_loc = self.current_target_position
        d = np.linalg.norm(np.array(target_loc) - np.array(cutter_loc))
        return d

    def get_reward(self, command, done=False):

        in_mouth = self.in_mouth
        failure = not in_mouth and self.failure

        if self.debug:
            if in_mouth:
                print('[DEBUG] Branch is in mouth!')
            if failure:
                print('[DEBUG] Branch has reached failure point!')

        if failure:
            reward = -(self.max_elapsed_time - self.elapsed_time)
        else:
            d = self.get_cutter_dist()
            dist_proportion = max(1 - d / self.min_reward_dist, 0.0)

            if done:
                if not in_mouth:
                    reward = 0.0
                else:
                    reward = (self.max_elapsed_time - self.elapsed_time) * dist_proportion
            else:
                accel = np.linalg.norm(command - self.last_command)
                accel_multiplier = 1.0 - max(0, accel - self.accel_threshold) * self.accel_penalty

                ts = self.action_freq / 240.0
                reward = dist_proportion * ts * (1.0 if in_mouth else 0.25) * accel_multiplier

        if self.debug:
            print('Obtained reward: {:.3f}'.format(reward))

        return reward


    def is_done(self):
        return self.failure or (self.elapsed_time >= self.max_elapsed_time)

    def reset(self):

        if self.elapsed_time:
            print('Reset! (Elapsed time {:.2f}s)'.format(self.elapsed_time))

        self.elapsed_time = 0.0
        self.failure = False
        self.in_mouth = False
        pb.restoreState(stateId=self.start_state, physicsClientId=self.client_id)

        self.last_command = np.zeros(2)

        # Modify the scenery
        self.reset_trees()
        self.randomize()

        # Reset the image noise parameters

        self.last_grayscale = None
        # self.current_depth_noise = (np.random.uniform(0, self.max_depth_sigma), self.noise_buffer.get_random(), np.random.uniform(0, self.max_noise_alpha))
        # self.current_tree_noise = (np.random.uniform(0, self.max_tree_sigma), self.noise_buffer.get_random(), np.random.uniform(0, self.max_noise_alpha))

        # Compute a random for the camera transform

        xyz_noise = np.random.uniform(-0.0025, 0.0025, size=3)
        rpy_noise = np.random.uniform(-np.radians(2.5), np.radians(2.5), size=3)
        noise_tf = np.identity(4)
        noise_tf[:3,3] = xyz_noise
        noise_tf[:3,:3] = Rotation.from_euler('xyz', rpy_noise, degrees=False).as_matrix()
        self.current_camera_tf = self.ideal_tool_camera_tf @ noise_tf

        # From the selected target on the tree (computed in reset_trees()), figure out the offset for the cutters
        # Convert to world pose and then solve for the IKs
        tf = pose_to_tf(self.target_pose[:3], self.target_pose[3:])
        offset = np.array([np.random.uniform(-0.05, 0.05),
                           np.random.uniform(-0.05, 0.05),
                           np.random.uniform(-0.02, 0.02) - 0.15]) * np.array([0.0, 0.0, 1.0])
        cutter_start_pos = homog_tf(tf, offset)
        ik = self.robot.solve_end_effector_ik('cutpoint', cutter_start_pos, self.start_orientation)
        self.robot.reset_joint_states(ik)

        self.target_tf = self.robot.get_link_kinematics('cutpoint', as_matrix=True)

        return self.get_obs()

    def render(self, mode='human', close=False):
        print('Last dist: {:.3f}'.format(self.get_cutter_dist()))

    def reset_trees(self):

        # Select one of the trees from the tree model metadata
        all_trees = list(self.tree_model_metadata)
        self.target_tree = all_trees[np.random.choice(len(all_trees))]
        all_trees.remove(self.target_tree)
        random.shuffle(all_trees)

        # From the tree, select one of the targets
        self.target_id = np.random.randint(len(self.tree_model_metadata[self.target_tree]))

        # Select a random pose from the list of random poses
        self.target_pose = self.poses[np.random.randint(len(self.poses))]

        # Based on the target pose and the corresponding target, figure out where the base of the tree is (may clip through floor)
        tree_frame_pt = self.tree_model_metadata[self.target_tree][self.target_id]['position']
        target_tree_target_tf = np.identity(4)
        target_tree_target_tf[:3,3] = self.target_pose[:3]
        target_tree_target_tf[:3,:3] = Rotation.from_quat([0.7071, 0, 0, 0.7071]).as_matrix()
        base_loc = homog_tf(target_tree_target_tf, -tree_frame_pt)

        pb.resetBasePositionAndOrientation(bodyUniqueId=self.target_tree, posObj=base_loc, ornObj=[0.7071, 0, 0, 0.7071], physicsClientId=self.client_id)

        ROW_SPACING = 2.0
        offsets = np.arange(len(all_trees))
        offsets = (offsets - offsets.mean()) * 2
        bg_offset = base_loc[1] + ROW_SPACING
        for bg_tree, offset in zip(all_trees, offsets):
            base_offset = np.array([np.random.uniform(-0.10, 0.10) + offset, np.random.uniform(-0.10, 0.10) + bg_offset, 0])
            pb.resetBasePositionAndOrientation(bodyUniqueId=bg_tree, posObj=base_offset, ornObj=[0.7071, 0, 0, 0.7071], physicsClientId=self.client_id)

    def randomize(self):
        # Resets ground texture
        pb.changeVisualShape(objectUniqueId=self.plane_id, linkIndex=-1, textureUniqueId=self.plane_textures[np.random.choice(len(self.plane_textures))],
                             physicsClientId=self.client_id)



        self.lighting = np.random.uniform(-1.0, 1.0, 3)
        self.lighting[2] = np.abs(self.lighting[2])
        self.lighting *= np.random.uniform(8.0, 16.0) / np.linalg.norm(self.lighting)
        self.contrast = np.random.uniform(0.5, 2.0)

        # TODO: Randomize exposure, loaded robot model, etc.



if __name__ == '__main__':

    # action = 'eval'
    action = 'train'
    width = 318
    height = 180
    grayscale = False
    use_seg = False
    use_depth = False
    use_flow = False
    use_last_frame = True
    num_envs = 3

    model_name = 'model_{w}_{h}{g}{s}{d}{f}{l}.zip'.format(w=width, h=height, g='_grayscale' if grayscale else '',
                                                       s='_seg' if use_seg else '', d='_depth' if use_depth else '',
                                                       f='_flow' if use_flow else '', l='_uselast' if use_last_frame else '')

    if action == 'train':


        def make_env(monitor=False, with_gui=False):
            env = CutterEnv(width, height, grayscale=grayscale, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow,
                             use_last_frame=use_last_frame, use_gui=with_gui, max_elapsed_time=2.5, max_vel=0.05, debug=False)
            if monitor:
                env = Monitor(env)
            return env

        env = VecTransposeImage(SubprocVecEnv([make_env] * num_envs))
        # env = (VecTransposeImage(DummyVecEnv([partial(make_env, monitor=True)])))
        eval_env = (VecTransposeImage(DummyVecEnv([partial(make_env, monitor=True)])))

        n_steps = 600 // num_envs
        batch_size = 60
        eval_callback = EvalCallback(eval_env, best_model_save_path='./', log_path='./', eval_freq=n_steps, n_eval_episodes=10,
                                     deterministic=True, render=False)

        # model_file = '{}.model'.format(eval_env.model_name)
        model = PPO("CnnPolicy", env, batch_size=batch_size, n_steps=n_steps, verbose=1, device='auto')

        print('Learning...')

        model.learn(total_timesteps=50000, callback=eval_callback)
        print('Done learning!')
        # model.save(model_file)

    elif action == 'eval':
        # env = CutterEnv(159, 90, use_seg=use_seg, use_depth=use_depth, use_gui=True, max_elapsed_time=2.5, max_vel=0.05, debug=True)
        env = CutterEnv(width, height, grayscale=grayscale, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow, use_last_frame=use_last_frame,
                        use_gui=True, max_elapsed_time=1.0, max_vel=0.05, debug=True)
        model = PPO("CnnPolicy", env, verbose=1)
        # model_file = '{}.model'.format(env.model_name)
        model_file = 'best_model.zip'
        if os.path.exists(model_file):
            model = model.load(model_file)
        obs = env.reset()
        all_dists = []
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            # action = env.action_space.sample()
            # if (i + 1) % 8:
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
