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

import camera_util
import matplotlib.pyplot as plt

import gym
from gym import spaces

BASE_ROT = [0.7071, 0, 0, 0.7071]

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
                 use_gui=False, img_buffer_size=0):
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
        self.action_space = spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), dtype=np.float32)  # LR, UD
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, num_channels), dtype=np.uint8)
        self.model_name = 'model_{mode}{use_depth}{flow}{uselast}'.format(mode='seg' if use_seg else ('gray' if grayscale else 'rgb'),
                                                           use_depth='_depth' if use_depth else '', flow='_flow' if use_flow else '',
                                                                          uselast='_uselastframe' if use_last_frame else '')

        # For visualization
        self.image_buffer_index = 0
        self.image_buffer = None
        if img_buffer_size > 0:
            self.image_buffer = np.zeros((img_buffer_size, self.height, self.width, 3), dtype=np.uint8)

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
        if self.image_buffer is not None:
            self.image_buffer[self.image_buffer_index] = rgb_img
            self.image_buffer_index = (self.image_buffer_index + 1) % (self.image_buffer.shape[0])

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

    def set_title(self, title):
        self.axs[0].set_title(title)


class CutterEnv(CutterEnvBase):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, grayscale=False, use_seg=False, use_depth=False, use_flow=False, use_last_frame=False, max_vel=0.075, action_freq=24, max_elapsed_time=3.0,
                 min_reward_dist=0.10, difficulty=0.0, eval=False, use_gui=False, debug=False, img_buffer_size=0):
        super(CutterEnv, self).__init__(width, height, grayscale=grayscale, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow, use_last_frame=use_last_frame,
                                        use_gui=use_gui, img_buffer_size=img_buffer_size)

        # Configuration parameters
        self.debug = debug
        self.difficulty = difficulty
        self.eval = eval
        self.action_freq = action_freq
        self.max_elapsed_time = max_elapsed_time
        self.min_reward_dist = min_reward_dist
        self.max_vel = max_vel
        self.current_camera_tf = np.identity(4)
        self.mesh_num_points = 100
        self.accel_threshold = 0.50         # Vector magnitude where [X,Y] are in range [-1, 1]
        self.accel_penalty = 0.50           # For every unit accel exceeding the threshold, reduce base reward by given proportion
        self.frames_per_img = 8             # Corresponds to 30 Hz
        self.pass_threshold = 0.015         # Experiment ends positively if Z-ax distance to target is less than threshold
        self.fail_threshold = 0.02          # Experiment ends negatively if cutter passes Z-ax distance behind target

        # State parameters
        self.target_pose = None             # What pose should the cutter end up in?
        self.target_tree = None             # Which tree model is in front?
        self.target_id = None               # Which of the side branches are we aiming at on the target tree?
        self.target_tf = np.identity(4)     # What is the next waypoint for the cutter?
        self.approach_vec = None            # Unit vector pointing towards the target from the cutter start position
        self.approach_history = []          # Keeps track of best approach distances
        self.speed = max_vel
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
        self.home_joints = [-1.5708, -2.2689, -1.3963, 0.52360, 1.5708, 3.14159]
        self.robot = URDFRobot(arm_location, [0, 0, 0.02], [0, 0, 0, 1], flags=pb.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                               physicsClientId=self.client_id)
        self.robot.reset_joint_states(self.home_joints)
        self.start_orientation = self.robot.get_link_kinematics('cutpoint')[1]
        self.proj_mat = pb.computeProjectionMatrixFOV(
            fov=42.0, aspect = width / height, nearVal=0.01,
            farVal=10.0)
        self.robot.attach_ghost_body_from_file('robots/ur5e/collision/cutter-mouth-collision.stl',
                                               'mouth', 'cutpoint', rpy=[0, 0, 3.1416])
        self.robot.attach_ghost_body_from_file('robots/ur5e/collision/cutter-failure-zone.stl',
                                               'failure', 'cutpoint', rpy=[0, 0, 3.1416])

        # Create pose database
        self.poses = self.load_pose_database()

        # Load trellis
        self.trellis_id = self.load_mesh(os.path.join('models', 'trellis-setup.obj'), pos=[0, 2.0, 0],
                                         orientation=BASE_ROT)

        # Load wall and wall textures
        wall_folder = os.path.join('models', 'wall_textures')
        self.wall_textures = [pb.loadTexture(os.path.join(wall_folder, file)) for file in os.listdir(wall_folder) if file.endswith('.png')]
        wall_viz = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=[10, 0.01, 7.5], physicsClientId=self.client_id)
        wall_col = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=[10, 0.01, 7.5], physicsClientId=self.client_id)
        self.wall_id = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=wall_viz, baseCollisionShapeIndex=wall_col, basePosition=[0, 15, 7.5],
                                          baseOrientation=[0,0,0,1], physicsClientId=self.client_id)

        side_wall_viz = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=[0.01, 10.0, 7.5], physicsClientId=self.client_id)
        side_wall_col = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=[0.01, 10.0, 7.5],
                                           physicsClientId=self.client_id)
        self.side_wall_id = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=side_wall_viz, baseCollisionShapeIndex=side_wall_col, basePosition=[5, 0, 7.5],
                                          baseOrientation=[0,0,0,1], physicsClientId=self.client_id)

        # Load trees - Put them in background out of sight of the camera

        self.tree_model_metadata = {}
        tree_models_directory = os.path.join('models', 'trees')
        tree_model_files = [x for x in os.listdir(tree_models_directory) if x.endswith('.obj') and not '-' in x]
        if len(tree_model_files) < 3:
            tree_model_files = tree_model_files * 3

        for file in tree_model_files:

            path = os.path.join(tree_models_directory, file)
            col_path = path.replace('.obj', '-collision.obj')
            tree_id = self.load_mesh(path, col_path, pos=[-10, 5, 0], orientation=BASE_ROT)
            annotation_path = path.replace('.obj', '.annotations')
            with open(annotation_path, 'rb') as fh:
                annotations = pickle.load(fh)
            self.tree_model_metadata[tree_id] = annotations

        self.start_state = pb.saveState(physicsClientId=self.client_id)

    def load_mesh(self, mesh_file, col_file=None, mass=0, pos=None, orientation=None):
        viz = pb.createVisualShape(shapeType=pb.GEOM_MESH, fileName=mesh_file, physicsClientId=self.client_id)
        if col_file is None:
            col = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=[0.001, 0.001, 0.001], physicsClientId=self.client_id)
        else:
            col = pb.createCollisionShape(shapeType=pb.GEOM_MESH, fileName=col_file, physicsClientId=self.client_id)
        if pos is None:
            pos = [0, 0, 0]
        if orientation is None:
            orientation = [0, 0, 0, 1]
        return pb.createMultiBody(baseMass=mass, baseVisualShapeIndex=viz, baseCollisionShapeIndex=col,
                                  basePosition=pos, baseOrientation=orientation,
                                  physicsClientId=self.client_id)


    def update_difficulty(self, difficulty):
        self.difficulty = difficulty
        return difficulty

    def step(self, action, realtime=False):
        # Execute one time step within the environment

        self.target_tf = self.robot.get_link_kinematics('cutpoint', as_matrix=True)

        horizontal, vertical = action
        vel_command = np.array([horizontal, vertical])

        step = self.action_freq * self.speed / 240
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

        # Compute approach distance with respect to original approach vector
        base_pos, base_quat = pb.getBasePositionAndOrientation(self.target_tree, physicsClientId=self.client_id)
        base_tf = pose_to_tf(base_pos, base_quat)
        current_target = homog_tf(base_tf, self.tree_model_metadata[self.target_tree][self.target_id]['position'])
        current_pos = np.array(self.robot.get_link_kinematics('cutpoint', as_matrix=False)[0])
        approach_dist = (current_pos - current_target).dot(
            self.approach_vec)  # Positive is before target, neg is past target

        no_improvement = False
        if len(self.approach_history) >= 3:
            if all(map(lambda x: x < approach_dist, self.approach_history[-3:])):
                no_improvement = True
                if self.debug:
                    print('[DEBUG] No improvement!')
        self.approach_history.append(abs(approach_dist))

        # Compute collisions with ghost bodies, as well as current target position
        tree_pts = self.tree_model_metadata[self.target_tree][self.target_id]['points']
        self.in_mouth = self.robot.query_ghost_body_collision('mouth', tree_pts,
                                                         point_frame_tf=base_tf, plot_debug=False)
        self.failure = not self.in_mouth and \
                       (approach_dist < -self.fail_threshold or self.robot.query_ghost_body_collision('failure', tree_pts, point_frame_tf=base_tf, plot_debug=False))



        # Done conditions: Failure, time elapsed, cutter is in mouth and within threshold
        done =  self.failure or (self.elapsed_time >= self.max_elapsed_time) or (self.in_mouth and approach_dist < self.pass_threshold) or no_improvement
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
                    reward = -(self.max_elapsed_time - self.elapsed_time)
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

    def reset(self):

        if self.elapsed_time:
            print('Reset! (Elapsed time {:.2f}s)'.format(self.elapsed_time))

        self.elapsed_time = 0.0
        self.failure = False
        self.in_mouth = False
        self.speed = self.max_vel if self.eval else np.random.uniform(self.max_vel * 0.5, self.max_vel)
        pb.restoreState(stateId=self.start_state, physicsClientId=self.client_id)

        self.last_command = np.zeros(2)

        # Modify the scenery
        self.reset_trees()
        self.randomize()

        # Reset the image noise parameters

        self.last_grayscale = None
        # self.current_depth_noise = (np.random.uniform(0, self.max_depth_sigma), self.noise_buffer.get_random(), np.random.uniform(0, self.max_noise_alpha))
        # self.current_tree_noise = (np.random.uniform(0, self.max_tree_sigma), self.noise_buffer.get_random(), np.random.uniform(0, self.max_noise_alpha))

        # Initialize an ideal camera transform, but with noise added to the parameters
        tool_tf = self.robot.get_link_kinematics('wrist_3_link-tool0_fixed_joint', as_matrix=True)
        deg_noise = 0 if self.eval else (0.5 + 4.5 * self.difficulty)

        pan = np.radians(np.random.uniform(-45-deg_noise, -45+deg_noise))
        tilt = np.radians(np.random.uniform(45-deg_noise, 45+deg_noise))
        ideal_view_matrix = camera_util.get_view_matrix(pan, tilt, base_tf=tool_tf)
        ideal_tool_camera_tf = np.linalg.inv(tool_tf) @ np.linalg.inv(ideal_view_matrix)

        # Perturb the tool-camera TF
        xyz_noise = np.random.uniform(-0.0025, 0.0025, size=3)
        rpy_noise = np.random.uniform(-np.radians(2.5), np.radians(2.5), size=3)
        noise_tf = np.identity(4)
        noise_tf[:3,3] = xyz_noise
        noise_tf[:3,:3] = Rotation.from_euler('xyz', rpy_noise, degrees=False).as_matrix()
        self.current_camera_tf = ideal_tool_camera_tf @ noise_tf

        # From the selected target on the tree (computed in reset_trees()), figure out the offset for the cutters
        # The offset has a schedule where at the lowest difficulty, the cutters start out right in front of the
        # target, and gradually move back as the difficulty increases

        easy_dist = -0.06
        hard_dist = -0.15
        easy_dev = 0.005
        hard_dev = 0.03

        dist_center = easy_dist + (hard_dist - easy_dist) * self.difficulty
        dev = easy_dev + (hard_dev - easy_dev) * self.difficulty
        if self.eval:
            dev = dev / 3.0
        dist_bounds = (dist_center - dev, dist_center + dev)

        easy_offset = 0.01
        hard_offset = 0.075
        offset = easy_offset + (hard_offset - easy_offset) * self.difficulty
        if self.eval:
            offset = offset / 3.0
        offset_bounds = (-offset, offset)

        # Convert to world pose and then solve for the IKs
        tf = pose_to_tf(self.target_pose[:3], self.target_pose[3:])
        offset = np.array([np.random.uniform(*offset_bounds),
                           np.random.uniform(*offset_bounds),
                           np.random.uniform(*dist_bounds)])
        cutter_start_pos = homog_tf(tf, offset)

        approach_vec = cutter_start_pos - self.target_pose[:3]
        approach_vec /= np.linalg.norm(approach_vec)
        self.approach_vec = approach_vec
        self.approach_history = []
        self.robot.move_end_effector_ik('cutpoint', cutter_start_pos, self.start_orientation, threshold=0.005,
                                        retries=3)

        self.target_tf = self.robot.get_link_kinematics('cutpoint', as_matrix=True)

        return self.get_obs()

    def render(self, mode='human', close=False):
        print('Last dist: {:.3f}'.format(self.get_cutter_dist()))

    def reset_trees(self):

        # Determine the spacing for the wall
        wall_range = (0.5, 1)
        side_wall_range = (2, 10)
        wall_offset = np.random.uniform(*wall_range)
        side_wall_offset = np.random.uniform(*side_wall_range)

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
        target_tree_target_tf[:3,:3] = Rotation.from_quat(BASE_ROT).as_matrix()
        base_loc = homog_tf(target_tree_target_tf, -tree_frame_pt)

        pb.resetBasePositionAndOrientation(bodyUniqueId=self.target_tree, posObj=base_loc, ornObj=BASE_ROT, physicsClientId=self.client_id)

        # Determine the spacing of the three trees, and select the index to associate with the branch
        # Then move the trellis into position accordingly
        trellis_base_pos = np.array([-0.43, 0, 0.43]) + np.random.uniform(-0.03, 0.03, size=3)
        trellis_height = 0.70
        chosen_idx = np.random.randint(3)
        other_idx = [0, 1, 2]
        other_idx.remove(chosen_idx)

        trellis_loc = np.array([base_loc[0] - trellis_base_pos[chosen_idx], base_loc[1], trellis_height])
        pb.resetBasePositionAndOrientation(bodyUniqueId=self.trellis_id, posObj=trellis_loc, ornObj=BASE_ROT, physicsClientId=self.client_id)

        # Move walls and the other trees into position, and any excess trees off into the background
        for tree_id, leader_idx in zip(all_trees[:len(other_idx)], other_idx):
            pb.resetBasePositionAndOrientation(bodyUniqueId=tree_id, posObj=[trellis_loc[0] + trellis_base_pos[leader_idx], trellis_loc[1], trellis_height],
                                               ornObj=BASE_ROT, physicsClientId=self.client_id)

        for tree_id in all_trees[len(other_idx):]:
            pb.resetBasePositionAndOrientation(bodyUniqueId=tree_id, posObj=[-10, -10, 0], ornObj=BASE_ROT, physicsClientId=self.client_id)

        pb.resetBasePositionAndOrientation(bodyUniqueId=self.wall_id, posObj=[0, base_loc[1] + wall_offset, 7.5],
                                           ornObj=[0, 0, 0, 1], physicsClientId=self.client_id)

        pb.resetBasePositionAndOrientation(bodyUniqueId=self.side_wall_id, posObj=[side_wall_offset, 0, 7.5], ornObj=[0,0,0,1],
                                           physicsClientId=self.client_id)

        import pdb
        pdb.set_trace()

    def randomize(self):
        # Resets ground and wall texture
        pb.changeVisualShape(objectUniqueId=self.plane_id, linkIndex=-1, textureUniqueId=self.plane_textures[np.random.choice(len(self.plane_textures))],
                             physicsClientId=self.client_id)
        pb.changeVisualShape(objectUniqueId=self.wall_id, linkIndex=-1, textureUniqueId=self.wall_textures[np.random.choice(len(self.wall_textures))],
                             physicsClientId=self.client_id)
        pb.changeVisualShape(objectUniqueId=self.side_wall_id, linkIndex=-1,
                             textureUniqueId=self.wall_textures[np.random.choice(len(self.wall_textures))],
                             physicsClientId=self.client_id)

        self.lighting = np.random.uniform(-1.0, 1.0, 3)
        self.lighting[2] = np.abs(self.lighting[2])
        self.lighting *= np.random.uniform(8.0, 16.0) / np.linalg.norm(self.lighting)
        self.contrast = np.random.uniform(0.5, 2.0)

    def load_pose_database(self):
        db_file = 'pose_database.pickle'
        try:
            with open(db_file, 'rb') as fh:
                return pickle.load(fh)
        except FileNotFoundError:
            target_xs = np.linspace(-0.3, 0.3, num=25, endpoint=True)
            target_ys = np.linspace(0.75, 0.95, num=17, endpoint=True)
            target_zs = np.linspace(0.8, 1.3, num=25, endpoint=True)
            all_poses = [[x, y, z] + list(self.start_orientation) for x, y, z in
                         product(target_xs, target_ys, target_zs)]
            poses = self.robot.determine_reachable_target_poses('cutpoint', all_poses, self.home_joints, max_iters=100)
            with open(db_file, 'wb') as fh:
                pickle.dump(poses, fh)
            return poses

def check_input_variance(model, obs, samples=10, output=False):
    rez = np.array([model.predict(obs, deterministic=False)[0] for _ in range(samples)])
    std = rez.std(axis=0)
    if output:
        print('Stdevs: ' + ', '.join(['{:.4f}'.format(x) for x in std]))

    return std


if __name__ == '__main__':

    # action = 'eval'
    action = 'train'
    use_trained = True
    width = 424
    height = 240
    grayscale = False
    use_seg = False
    use_depth = False
    use_flow = False
    use_last_frame = True
    num_envs = 3
    record = False
    variance_debug = False

    model_name = 'model_{w}_{h}{g}{s}{d}{f}{l}.zip'.format(w=width, h=height, g='_grayscale' if grayscale else '',
                                                       s='_seg' if use_seg else '', d='_depth' if use_depth else '',
                                                       f='_flow' if use_flow else '', l='_uselast' if use_last_frame else '')

    if action == 'train':
        def make_env(monitor=False, with_gui=False, eval=False):
            env = CutterEnv(width, height, grayscale=grayscale, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow,
                             use_last_frame=use_last_frame, use_gui=with_gui, max_elapsed_time=1.5, max_vel=0.05, difficulty=0.0, debug=False,
                            eval=eval)
            if monitor:
                env = Monitor(env)
            return env

        env = VecTransposeImage(SubprocVecEnv([make_env] * num_envs))
        eval_env = (VecTransposeImage(DummyVecEnv([partial(make_env, monitor=True, eval=True)])))

        n_steps = 600 // num_envs
        batch_size = 60
        eval_callback = EvalCallback(eval_env, best_model_save_path='./', log_path='./', eval_freq=n_steps, n_eval_episodes=12,
                                     deterministic=True, render=False)

        model = PPO("CnnPolicy", env, batch_size=batch_size, n_steps=n_steps, verbose=1, device='auto')

        print('Learning...')
        difficulties = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        steps_per_difficulty = 8040
        for difficulty in difficulties:
            difficulty_str = str(difficulty).replace('.', '_')
            model_file = f'best_model_{difficulty_str}.zip'
            if not os.path.exists(model_file):

                print('STARTING LEARNING FOR DIFFICULTY {}'.format(difficulty))
                env.env_method('update_difficulty', difficulty)
                eval_env.env_method('update_difficulty', difficulty)
                model.learn(total_timesteps=steps_per_difficulty, callback=eval_callback)
                os.rename('evaluations.npz', f'evaluations_{difficulty_str}.npz')
                os.rename('best_model.zip', f'best_model_{difficulty_str}.zip')
                raise Exception('Done, restart')        # BUG WHERE NEXT DIFFICULTY WON'T OUTPUT MODEL
            else:
                print('Difficulty {} has already been learned!'.format(difficulty))
            model = model.load(f'best_model_{difficulty_str}.zip', env=env)


    elif action == 'eval':
        timesteps, buffer_size = (150, 450) if record else (1000, 0)

        # env = CutterEnv(159, 90, use_seg=use_seg, use_depth=use_depth, use_gui=True, max_elapsed_time=2.5, max_vel=0.05, debug=True)
        env = CutterEnv(width, height, grayscale=grayscale, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow, use_last_frame=use_last_frame,
                        use_gui=True, max_elapsed_time=1.0, max_vel=0.05, debug=True, img_buffer_size=buffer_size,
                        eval=True, difficulty=1.0)
        model = PPO("CnnPolicy", env, verbose=1)
        # model_file = '{}.model'.format(env.model_name)
        if use_trained:
            model_file = 'best_model_1_0.zip'
            if os.path.exists(model_file):
                model = model.load(model_file)
                print('Using best model!')
        obs = env.reset()
        all_dists = []
        for i in range(timesteps):
            if use_trained:
                action, _states = model.predict(obs, deterministic=True)
                if variance_debug:
                    check_input_variance(model, obs, output=True)
            else:
                # action = env.action_space.sample()
                action = np.array([1.0, 0.0])

            obs, reward, done, info = env.step(action, realtime=True)
            # env.render()
            if done:
                all_dists.append(env.get_cutter_dist())
                obs = env.reset()

        if record:
            from matplotlib import animation
            fig = plt.figure(figsize=(8.0, 6.4))
            img_ax = fig.add_subplot(111)
            img_obj = img_ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))

            plt.ion()
            plt.show()

            def animate(i):
                rgb_data = env.image_buffer[i]
                img_obj.set_data(rgb_data)

            ani = animation.FuncAnimation(fig, animate, frames=range(buffer_size), interval=1000/20)
            ani.save('test_output.mp4')

        print('Average terminal dist: {:.3f}'.format(np.mean(all_dists)))
        env.close()
    else:
        raise NotImplementedError("Unknown action {}".format(action))
