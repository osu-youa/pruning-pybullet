import sys
import os
model_path = os.path.join(os.path.expanduser('~'), 'install', 'pytorch-CycleGAN-and-pix2pix')
if os.path.exists(model_path):
    sys.path.append(model_path)
    import torch
    from data.base_dataset import get_transform
    from util.util import tensor2im

import pybullet as pb
import time
import pybullet_data
import numpy as np

from helloworld import URDFRobot
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
    def __init__(self, width, height, grayscale=False, use_net=False, use_seg=False, use_depth=False, use_flow=False, use_last_frame=False,
                 use_gui=False, use_plot_gui=True, crop=None, downscale = None, img_buffer_size=0):
        super(CutterEnvBase, self).__init__()

        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.use_net = use_net
        self.use_seg = use_seg
        self.use_depth = use_depth
        self.use_flow = use_flow
        self.use_last_frame = use_last_frame
        self.use_gui = use_gui
        self.use_plot_gui = use_plot_gui
        self.crop = crop
        self.downscale = downscale

        self.last_grayscale = None

        # Initialize gym parameters
        num_channels = (2 if use_seg else (1 if grayscale else 3)) + (1 if use_depth else 0) + (1 if use_flow else 0) + (1 if use_last_frame else 0)
        final_shape = (height, width, num_channels)
        if self.crop is not None:
            if isinstance(self.crop[0], int):
                width = self.crop[0] * 2
                height = self.crop[1] * 2
            else:
                width = self.crop[0][1] - self.crop[0][0]
                height = self.crop[1][1] - self.crop[1][0]

            final_shape = (height, width, num_channels)
        if self.downscale is not None:
            if isinstance(self.downscale, int):
                final_shape = (final_shape[0] // self.downscale, final_shape[1] // self.downscale, num_channels)
            else:
                final_shape = (self.downscale[1], self.downscale[0], num_channels)

        self.action_space = spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]), dtype=np.float32)  # LR, UD
        self.observation_space = spaces.Box(low=0, high=255, shape=final_shape, dtype=np.uint8)
        self.model_name = 'model_{mode}{use_depth}{flow}{uselast}'.format(mode='seg' if use_seg else ('gray' if grayscale else 'rgb'),
                                                           use_depth='_depth' if use_depth else '', flow='_flow' if use_flow else '',
                                                                          uselast='_uselastframe' if use_last_frame else '')

        self.model = None
        self.model_tf = None
        if self.use_net:

            import pickle
            from options.test_options import TestOptions
            from models import create_model
            with open('default_options.pickle', 'rb') as fh:
                opt = pickle.load(fh)
            opt.checkpoints_dir = os.path.join(model_path, 'checkpoints')
            opt.name = 'pruning_front_pix2pix'
            self.model = create_model(opt)
            self.model.setup(opt)
            # self.model.eval()

            self.model_tf = get_transform(opt)

            # For testing, but also to deal with overhead of loading model into GPU
            test_input = {'A': torch.rand(1,3,256,256), 'A_paths': ''}
            self.model.set_input(test_input)
            self.model.test()

        # For visualization
        self.image_buffer_index = 0
        self.image_buffer = None
        if img_buffer_size > 0:
            self.image_buffer = np.zeros((img_buffer_size, self.height, self.width, 3), dtype=np.uint8)

        if self.use_gui and self.use_plot_gui:
            plt.ion()
            self.fig = plt.figure()
            self.img_ax = self.fig.add_subplot(131)
            self.img = self.img_ax.imshow(np.zeros((self.height, self.width), dtype=np.uint8))

            self.img_ax_2 = self.fig.add_subplot(132)
            self.img_2 = self.img_ax_2.imshow(np.zeros((self.height, self.width), dtype=np.uint8))


            self.action_ax = self.fig.add_subplot(133)
            self.action_ax.set_xlim(-1, 1)
            self.action_ax.set_ylim(-1, 1)
            self.arrow = None

            self.fig.canvas.draw()
            plt.pause(0.01)

    def step(self, action):
        raise NotImplementedError()

    def process_rgb_image(self, rgb_img):

        if self.use_net:

            if self.use_gui and self.use_plot_gui:
                self.img_2.set_data(rgb_img)

            img_tensor = self.model_tf(Image.fromarray(rgb_img))
            img_input = {'A': img_tensor.view(-1, *img_tensor.shape) , 'A_paths': ''}
            self.model.set_input(img_input)
            self.model.test()
            output = self.model.get_current_visuals()['fake']

            if output.shape[-2] != (self.height, self.width):
                output = torch.nn.functional.interpolate(output, size=(self.height, self.width))

            rgb_img = tensor2im(output)

        if self.crop is not None:

            if isinstance(self.crop[0], int):
                crop_h, crop_v = self.crop
                midpoint_v = self.height // 2
                midpoint_h = self.width // 2
                rgb_img = rgb_img[midpoint_v - crop_v:midpoint_v + crop_v,
                                  midpoint_h - crop_h:midpoint_h + crop_h]
            else:
                rgb_img = rgb_img[self.crop[1][0]:self.crop[1][1],
                                  self.crop[0][0]:self.crop[0][1]]

        if self.downscale:
            img = Image.fromarray(rgb_img)
            if isinstance(self.downscale, int):
                current_w = rgb_img.shape[1]
                current_h = rgb_img.shape[0]
                rgb_img = np.array(img.resize((current_w//self.downscale, current_h //self.downscale))).astype(np.uint8)
            else:
                rgb_img = np.array(img.resize(self.downscale)).astype(np.uint8)

        return rgb_img


    def get_obs(self):
        # Grab the new image observation
        rgb_img, depth_img, seg_img = self.get_images()
        rgb_img = self.process_rgb_image(rgb_img)

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
            # if self.use_gui:
            #     self.imgs[1].set_data(np.dstack([flow_img] * 3))

        if self.use_last_frame:
            layers.append(self.last_grayscale if self.last_grayscale is not None else grayscale)

        self.last_grayscale = grayscale

        if self.use_gui and self.use_plot_gui:
            self.img.set_data(rgb_img)
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
        self.img_ax.set_title(title)

    def set_action(self, action):
        if self.use_gui and self.use_plot_gui:
            if self.arrow is not None:
                self.arrow.remove()
            self.arrow = self.action_ax.arrow(0, 0, action[0], action[1])
            self.action_ax.set_title('{:.3f}, {:.3f}'.format(*action))


class CutterEnv(CutterEnvBase):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, grayscale=False, use_net=False, use_seg=False, use_depth=False, use_flow=False, use_last_frame=False, max_vel=0.075, action_freq=24, max_elapsed_time=3.0,
                 min_reward_dist=0.10, difficulty=0.0, eval=False, eval_count=None, use_gui=False, crop=None,
                 downscale=None, debug=False, img_buffer_size=0):
        super(CutterEnv, self).__init__(width, height, grayscale=grayscale, use_net=use_net, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow, use_last_frame=use_last_frame,
                                        use_gui=use_gui, downscale=downscale, crop=crop, img_buffer_size=img_buffer_size)

        # Configuration parameters
        self.debug = debug
        self.difficulty = difficulty
        self.eval = eval
        self.eval_count = eval_count
        self.action_freq = action_freq
        self.max_elapsed_time = max_elapsed_time
        self.min_reward_dist = min_reward_dist
        self.max_vel = max_vel
        self.current_camera_tf = np.identity(4)
        self.accel_threshold = 0.50         # Vector magnitude where [X,Y] are in range [-1, 1]
        self.accel_penalty = 0.0           # For every unit accel exceeding the threshold, reduce base reward by given proportion
        self.frames_per_img = 8             # Corresponds to 30 Hz
        self.fail_threshold = 0.0           # Experiment ends negatively if cutter passes Z-ax distance behind target

        # State parameters
        self.target_pose = None             # What pose should the cutter end up in?
        self.target_tree = None             # Which tree model is in front?
        self.target_id = None               # Which of the side branches are we aiming at on the target tree?
        self.target_tf = np.identity(4)     # What is the next waypoint for the cutter?
        self.approach_vec = None            # Unit vector pointing towards the target from the cutter start position
        self.approach_history = []          # Keeps track of best approach distances
        self.speed = max_vel
        self.elapsed_time = 0.0
        self.mesh_points = {}
        self.last_command = np.zeros(2)
        self.lighting = None                # Location of light source
        self.lighting_params = {}           # For feeding into getCameraImage (diffuse, color, etc.)
        self.contrast = None                # Contrast adjustment factor

        self.eval_counter = 0               # [EVAL only] After the specified number of evaluations, reset the Numpy random seed

        # Simulation tools - Some are only for seg masks!
        # self.noise_buffer = PerlinNoiseBuffer(width, height, rectangle_size=30, buffer_size=50)
        self.max_depth_sigma = 5.0
        self.max_tree_sigma = 5.0
        self.max_noise_alpha = 0.3

        # Setup Pybullet simulation

        self.client_id = pb.connect(pb.GUI if self.use_gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        pb.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        self.plane_id = pb.loadURDF("plane.urdf", physicsClientId=self.client_id)
        plane_texture_root = os.path.join('textures', 'floor')
        self.plane_textures = [pb.loadTexture(os.path.join(plane_texture_root, file), physicsClientId=self.client_id) for file in os.listdir(plane_texture_root)]

        # robot_name = 'ur5e_cutter_new_calibrated_precise.urdf'
        self.robot_name = 'ur5e_cutter_new_calibrated_precise_level.urdf'

        arm_location = os.path.join('robots', self.robot_name)
        self.home_joints = [-1.5708, -2.2689, -1.3963, 0.52360, 1.5708, 3.14159]
        self.robot = URDFRobot(arm_location, [0, 0, 0.02], [0, 0, 0, 1], flags=pb.URDF_USE_MATERIAL_COLORS_FROM_MTL,
                               physicsClientId=self.client_id)
        self.robot.reset_joint_states(self.home_joints)
        self.start_orientation = self.robot.get_link_kinematics('cutpoint')[1]
        self.proj_mat = pb.computeProjectionMatrixFOV(
            fov=42.0, aspect = width / height, nearVal=0.01,
            farVal=10.0)
        # self.robot.attach_ghost_body_from_file('robots/ur5e/collision/cutter-mouth-collision.stl',
        #                                        'mouth', 'cutpoint', rpy=[0, 0, 3.1416])
        # self.robot.attach_ghost_body_from_file('robots/ur5e/collision/cutter-mouth-collision-shrunk.stl',
        self.robot.attach_ghost_body_from_file('robots/ur5e/collision/mouth-full-extended.stl',
                                               'mouth', 'cutpoint', rpy=[0, 0, 0])
        # self.robot.attach_ghost_body_from_file('robots/ur5e/collision/mouth-bonus.stl',
        #                                        'bonus', 'cutpoint', rpy=[0, 0, 0])
        self.robot.attach_ghost_body_from_file('robots/ur5e/collision/cutter-failure-zone-new.stl',
                                               'failure', 'cutpoint', rpy=[0, 0, 0])

        # Create pose database
        self.poses = self.load_pose_database()

        # Load trellis
        self.trellis_frame_id = self.load_mesh(os.path.join('models', 'trellis-wood-frame.obj'),
                                               col_file=os.path.join('models', 'trellis-setup-collision.obj'),
                                               pos=[0, 2.0, 0], orientation=BASE_ROT)
        self.trellis_base_id = self.load_mesh(os.path.join('models', 'trellis-base-plane.obj'), pos=[0, 2.0, 0], orientation=BASE_ROT)
        self.trellis_wires_id = self.load_mesh(os.path.join('models', 'trellis-wires.obj'), pos=[0, 2.0, 0], orientation=BASE_ROT)

        # Load wall and wall textures
        wall_folder = os.path.join('models', 'wall_textures')
        self.wall_textures = [pb.loadTexture(os.path.join(wall_folder, file)) for file in os.listdir(wall_folder) if file.endswith('.png')]
        self.wall_dim = [4, 0.01, 1.5]
        wall_viz = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=self.wall_dim, physicsClientId=self.client_id)
        wall_col = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=self.wall_dim, physicsClientId=self.client_id)

        self.wall_id = pb.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # self.wall_id = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=wall_viz, baseCollisionShapeIndex=wall_col, basePosition=[0, 15, 7.5],
        #                                   baseOrientation=[0,0,0,1], physicsClientId=self.client_id)
        #
        side_wall_viz = pb.createVisualShape(shapeType=pb.GEOM_BOX, halfExtents=[0.01, 10.0, 7.5], physicsClientId=self.client_id)
        side_wall_col = pb.createCollisionShape(shapeType=pb.GEOM_BOX, halfExtents=[0.01, 10.0, 7.5],
                                           physicsClientId=self.client_id)
        self.side_wall_id = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=side_wall_viz, baseCollisionShapeIndex=side_wall_col, basePosition=[5, 0, 7.5],
                                          baseOrientation=[0,0,0,1], physicsClientId=self.client_id)

        # Load trees - Put them in background out of sight of the camera

        self.tree_model_metadata = {}
        self.tree_textures = []
        tree_models_directory = os.path.join('models', 'trees')
        tree_textures_directory = os.path.join('models', 'trees', 'textures')
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

        textures = [x for x in os.listdir(tree_textures_directory) if x.endswith('.png') and not x.endswith('N.png')]
        for texture_file in textures:
            self.tree_textures.append(pb.loadTexture(os.path.join(tree_textures_directory, texture_file), physicsClientId=self.client_id))

        # OTHER TEXTURES LIBRARY - LOADED IN THEIR RESPECTIVE FUNCTIONS
        self.canonical_textures = {}
        self.random_textures = []


        # TESTING
        self.robot.enable_force_torque_readings('wrist_3_link-tool0_fixed_joint')

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
        prev_target_pos = self.target_tf[:3,3].copy()
        new_target_pos = (self.target_tf @ delta)[:3].copy()
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

            if self.use_last_frame and i in img_update_frames:
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


        in_mouth = False
        failure = False

        if approach_dist < 0.065:        # Based on model size, this is the minimum dist before a collision can happen
            tree_pts = self.tree_model_metadata[self.target_tree][self.target_id]['points']
            in_mouth = self.robot.query_ghost_body_collision('mouth', tree_pts, point_frame_tf=base_tf, plot_debug=False)
            if not in_mouth and self.robot.query_ghost_body_collision('failure', tree_pts, point_frame_tf=base_tf):
                failure = True
                if self.debug:
                    print('[DEBUG] Entered failure region')

        done = in_mouth or failure or (approach_dist < -self.fail_threshold) or (self.elapsed_time >= self.max_elapsed_time) or no_improvement
        reward = self.get_reward(done, in_mouth)

        self.last_command = vel_command
        return self.get_obs(), reward, done, {}

    def get_images(self, canonical=False):
        view_matrix = self.robot.get_link_kinematics('wrist_3_link-tool0_fixed_joint',
                                                     as_matrix=True) @ self.current_camera_tf

        flags = {}
        if not (canonical or self.use_seg):
            flags['flags'] = pb.ER_NO_SEGMENTATION_MASK


        params = self.lighting_params
        params['shadow'] = True
        if canonical:
            params = {'lightColor': [1,1,1], 'shadow': False}

        _, _, rgb_img, raw_depth_img, raw_seg_img = pb.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=np.linalg.inv(view_matrix).T.reshape(-1),
            projectionMatrix=self.proj_mat,
            renderer=pb.ER_TINY_RENDERER,
            lightDirection=[0, 0, 5] if canonical else self.lighting,
            lightDistance=10,
            physicsClientId=self.client_id,
            **flags, **params
        )

        rgb_img = rgb_img[:, :, :3]
        # if not canonical:
        #     pil_img = Image.fromarray(rgb_img, 'RGB')
        #     enhancer = ImageEnhance.Contrast(pil_img)
        #     rgb_img = np.asarray(enhancer.enhance(self.contrast))

        depth_img = None
        if self.use_depth:
            depth_img = raw_depth_img
            # depth_img = overlay_noise(raw_depth_img, *self.current_depth_noise, convert_to_uint8=True)


        # By default, output a segmentation mask showing the tree mask
        seg_img = None
        if canonical:
            seg_img = np.zeros(raw_seg_img.shape, dtype=bool)
            for tree_id in self.tree_model_metadata:
                seg_img |= raw_seg_img == tree_id

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

    def get_reward(self, done=False, success=False):

        if self.debug and done:
            if success:

                print('[DEBUG] Episode success!')
            else:
                print('[DEBUG] Episode has failed!')

        d = self.get_cutter_dist()
        dist_proportion = max(1 - d / self.min_reward_dist, 0.0)
        if done:
            if success:
                reward = self.max_elapsed_time - self.elapsed_time
            else:
                reward = -self.max_elapsed_time
        else:
            ts = self.action_freq / 240.0
            reward = ts * dist_proportion

        if self.debug:
            print('Obtained reward: {:.3f}'.format(reward))

        return reward

    def reset(self):

        if self.elapsed_time:
            print('Reset! (Elapsed time {:.2f}s)'.format(self.elapsed_time))

        if self.eval and self.eval_count is not None:
            self.eval_counter = (self.eval_counter + 1) % self.eval_count
            np.random.seed(self.eval_counter)
            print('Using seed {}'.format(self.eval_counter))
        # else:
        #     new_seed = np.random.randint(65536)
        #     np.random.seed(new_seed)
        #     print('Using seed {}'.format(new_seed))

        self.elapsed_time = 0.0
        self.speed = self.max_vel if self.eval else np.random.uniform(self.max_vel * 0.5, self.max_vel)
        pb.restoreState(stateId=self.start_state, physicsClientId=self.client_id)

        self.last_command = np.zeros(2)

        # Modify the scenery
        self.lighting_params = {}
        self.reset_trees()
        self.randomize_scenery()

        # Reset the image noise parameters

        self.last_grayscale = None

        # From the selected target on the tree (computed in reset_trees()), figure out the offset for the cutters
        # The offset has a schedule where at the lowest difficulty, the cutters start out right in front of the
        # target, and gradually move back as the difficulty increases

        easy_dist = -0.06
        hard_dist = -0.175
        easy_dev = 0.005
        hard_dev = 0.025

        dist_center = easy_dist + (hard_dist - easy_dist) * self.difficulty
        dev = 0 if self.eval else easy_dev + (hard_dev - easy_dev) * self.difficulty
        dist_bounds = (dist_center - dev, dist_center + dev)

        easy_offset = 0.01
        hard_offset = 0.025
        offset = easy_offset + (hard_offset - easy_offset) * self.difficulty
        if self.eval:
            offset = offset * 0.75
        offset_bounds = (-offset - 0.015, offset)

        # Convert to world pose and then solve for the IKs
        tf = pose_to_tf(self.target_pose[:3], self.target_pose[3:])
        offset = np.array([np.random.uniform(*offset_bounds),
                           np.random.uniform(*offset_bounds) * 1.5,
                           # np.random.uniform(-0.02, -0.01),
                           np.random.uniform(*dist_bounds)])
        cutter_start_pos = homog_tf(tf, offset)

        approach_vec = cutter_start_pos - self.target_pose[:3]
        approach_vec /= np.linalg.norm(approach_vec)
        self.approach_vec = approach_vec
        self.approach_history = []
        self.robot.move_end_effector_ik('cutpoint', cutter_start_pos, self.start_orientation, threshold=0.005,
                                        retries=3)

        self.target_tf = self.robot.get_link_kinematics('cutpoint', as_matrix=True)
        self.randomize_camera()

        return self.get_obs()

    def render(self, mode='human', close=False):
        print('Last dist: {:.3f}'.format(self.get_cutter_dist()))


    def randomize_camera(self):
        tool_tf = self.robot.get_link_kinematics('wrist_3_link-tool0_fixed_joint', as_matrix=True)
        deg_noise = 0 if self.eval else (0.5 + 4.5 * self.difficulty)

        # pan_offset_pairs = [
        #     (-30, np.array([0,0,0])),
        #     (-22.5, np.array([-0.0338, 0, -0.025])),
        #     (-15, np.array([-0.0676, 0, -0.05]))
        # ]
        #
        # base_pan, base_offset = pan_offset_pairs[2]

        #
        #
        # # base_pan, base_offset = pan_offset_pairs[np.random.choice(len(pan_offset_pairs))]
        # xyz_offset = base_offset + np.random.uniform(-1, 1, 3) * np.array([0.01, 0.01, 0.02])
        #
        # pan = np.radians(np.random.uniform(base_pan - deg_noise, base_pan + deg_noise))

        pan = np.radians(np.random.uniform(-3, 3))
        # tilt = np.radians(np.random.uniform(-2.5, 7.5))
        tilt = np.radians(10.0 + np.random.uniform(-2.0, 2.0))
        # xyz_offset = np.random.uniform(-1, 1, 3) * np.array([0.01, 0.005, 0.01 ])
        xyz_offset = np.zeros(3)

        ideal_view_matrix = camera_util.get_view_matrix(pan, tilt, xyz_offset, base_tf=tool_tf)
        ideal_tool_camera_tf = np.linalg.inv(tool_tf) @ np.linalg.inv(ideal_view_matrix)

        # Perturb the tool-camera TF
        xyz_noise = np.random.uniform(-0.0025, 0.0025, size=3)
        rpy_noise = np.random.uniform(-np.radians(2.5), np.radians(2.5), size=3)
        noise_tf = np.identity(4)
        noise_tf[:3, 3] = xyz_noise
        noise_tf[:3, :3] = Rotation.from_euler('xyz', rpy_noise, degrees=False).as_matrix()
        self.current_camera_tf = ideal_tool_camera_tf @ noise_tf

    def randomize_tree_textures(self):
        all_trees = list(self.tree_model_metadata)
        use_same_texture = np.random.uniform() < 0.3
        if use_same_texture:
            random_texture = self.tree_textures[np.random.randint(len(self.tree_textures))]
            for tree_id in all_trees:
                self.set_texture(tree_id, random_texture)
        else:
            for tree_id in all_trees:
                self.set_texture(tree_id, self.tree_textures[np.random.randint(len(self.tree_textures))])

    def reset_trees(self):

        # Determine the spacing for the wall
        wall_range = (0.5, 1)
        side_wall_range = (2, 10)
        wall_offset = np.random.uniform(*wall_range)
        side_wall_offset = np.random.uniform(*side_wall_range)

        # Select one of the trees from the tree model metadata
        self.randomize_tree_textures()
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

        trellis_loc = np.array([base_loc[0] - trellis_base_pos[chosen_idx], self.target_pose[1] + np.random.uniform(0, 0.02), trellis_height])
        for trellis_id in [self.trellis_frame_id, self.trellis_wires_id]:
            pb.resetBasePositionAndOrientation(bodyUniqueId=trellis_id, posObj=trellis_loc, ornObj=BASE_ROT, physicsClientId=self.client_id)

        pb.resetBasePositionAndOrientation(bodyUniqueId=self.trellis_base_id, posObj=trellis_loc + np.random.uniform(-1, 1,size=3)* [0.05, 0.05, 0],
                                           ornObj=BASE_ROT, physicsClientId=self.client_id)

        # Move walls and the other trees into position, and any excess trees off into the background
        for tree_id, leader_idx in zip(all_trees[:len(other_idx)], other_idx):
            pb.resetBasePositionAndOrientation(bodyUniqueId=tree_id, posObj=[trellis_loc[0] + trellis_base_pos[leader_idx], trellis_loc[1] + np.random.uniform(-0.02, 0.02), trellis_height],
                                               ornObj=BASE_ROT, physicsClientId=self.client_id)

        for tree_id in all_trees[len(other_idx):]:
            pb.resetBasePositionAndOrientation(bodyUniqueId=tree_id, posObj=[-20, -20, 0], ornObj=BASE_ROT, physicsClientId=self.client_id)

        pb.resetBasePositionAndOrientation(bodyUniqueId=self.wall_id, posObj=[0, base_loc[1] + wall_offset, 0],
                                           ornObj=BASE_ROT, physicsClientId=self.client_id)

        pb.resetBasePositionAndOrientation(bodyUniqueId=self.side_wall_id, posObj=[side_wall_offset, 0, 7.5], ornObj=[0,0,0,1],
                                           physicsClientId=self.client_id)

    def randomize_scenery(self):
        # Resets ground and wall texture
        pb.changeVisualShape(objectUniqueId=self.plane_id, linkIndex=-1, textureUniqueId=self.plane_textures[np.random.choice(len(self.plane_textures))],
                             physicsClientId=self.client_id)
        pb.changeVisualShape(objectUniqueId=self.wall_id, linkIndex=-1, textureUniqueId=self.wall_textures[np.random.choice(len(self.wall_textures))],
                             physicsClientId=self.client_id)
        pb.changeVisualShape(objectUniqueId=self.side_wall_id, linkIndex=-1,
                             textureUniqueId=self.wall_textures[np.random.choice(len(self.wall_textures))],
                             physicsClientId=self.client_id)

        self.randomize_lighting_and_contrast()

    def randomize_lighting_and_contrast(self, include_color=False):
        self.lighting = np.random.uniform(-1.0, 1.0, 3)
        self.lighting[2] = np.abs(self.lighting[2])
        self.lighting *= np.random.uniform(2.0, 10.0) / np.linalg.norm(self.lighting)
        self.contrast = np.random.uniform(0.5, 2.0)
        if include_color:
            self.lighting_params['lightColor'] = np.random.uniform(size=3)
            self.lighting_params['lightAmbientCoeff'] = np.random.uniform(0.25, 0.9)
            self.lighting_params['lightDiffuseCoeff'] = np.random.uniform(0, 2)
            self.lighting_params['lightSpecularCoeff'] = np.random.uniform(0, 0.5)

    def load_pose_database(self):
        db_file = 'pose_database_{}.pickle'.format(self.robot_name)
        try:
            with open(db_file, 'rb') as fh:
                return pickle.load(fh)
        except FileNotFoundError:
            target_xs = np.linspace(-0.3, 0.3, num=25, endpoint=True)
            target_ys = np.linspace(0.75, 0.95, num=17, endpoint=True)
            target_zs = np.linspace(0.9, 1.3, num=25, endpoint=True)
            all_poses = [[x, y, z] + list(self.start_orientation) for x, y, z in
                         product(target_xs, target_ys, target_zs)]
            poses = self.robot.determine_reachable_target_poses('cutpoint', all_poses, self.home_joints, max_iters=100)
            with open(db_file, 'wb') as fh:
                pickle.dump(poses, fh)
            return poses

    def set_texture(self, obj_id, texture, link_id=-1):
        pb.changeVisualShape(objectUniqueId=obj_id, textureUniqueId=texture, linkIndex=link_id,
                             physicsClientId=self.client_id)

    def set_random(self):

        self.randomize_lighting_and_contrast(include_color=True)

        if not self.random_textures:
            print('Loading random textures...')
            texture_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'TexturesProcessed')
            for file in os.listdir(texture_dir):
                self.random_textures.append(pb.loadTexture(os.path.join(texture_dir, file), physicsClientId=self.client_id))
            print('Done loading random textures!')


        use_same_tree_texture = np.random.uniform() < 0.30
        if use_same_tree_texture:
            tree_texture = self.random_textures[np.random.randint(len(self.random_textures))]
            for tree_id in self.tree_model_metadata:
                self.set_texture(tree_id, tree_texture)
        else:
            for tree_id in self.tree_model_metadata:
                self.set_texture(tree_id, self.random_textures[np.random.randint(len(self.random_textures))])

        link_id = self.robot.convert_link_name('cutter')
        self.set_texture(self.robot.robot_id, self.random_textures[np.random.randint(len(self.random_textures))], link_id=link_id)

        for other_id in [self.trellis_base_id, self.trellis_frame_id, self.trellis_wires_id, self.wall_id, self.side_wall_id, self.plane_id]:
            self.set_texture(other_id, self.random_textures[np.random.randint(len(self.random_textures))])

        return self.get_images(canonical=False)


    def set_canonical(self):

        if not self.canonical_textures:
            print('Loading canonical textures...')
            for file in filter(lambda x: x.startswith('canonical-') and x.endswith('.png'), os.listdir('textures')):
                desc = file.replace('canonical-', '').replace('.png', '')
                self.canonical_textures[desc] = pb.loadTexture(os.path.join('textures', file),
                                                               physicsClientId=self.client_id)

        link_id = self.robot.convert_link_name('cutter')
        self.set_texture(self.robot.robot_id, self.canonical_textures['robot'], link_id=link_id)
        self.set_texture(self.trellis_frame_id, self.canonical_textures['wood'])
        self.set_texture(self.trellis_wires_id, self.canonical_textures['wires'])
        self.set_texture(self.trellis_base_id, self.canonical_textures['wall'])

        for tree_id in self.tree_model_metadata:
            self.set_texture(tree_id, self.canonical_textures['tree'])
        for other_id in [self.wall_id, self.side_wall_id, self.plane_id]:
            self.set_texture(other_id, self.canonical_textures['wall'])

        return self.get_images(canonical=True)

def check_input_variance(model, obs, samples=10, output=False):
    rez = np.array([model.predict(obs, deterministic=False)[0] for _ in range(samples)])
    std = rez.std(axis=0)
    if output:
        print('Stdevs: ' + ', '.join(['{:.4f}'.format(x) for x in std]))

    return std


if __name__ == '__main__':

    action = 'eval'
    # action = 'train'
    use_trained = False
    train_use_pretrained = True
    difficulty = 1.0
    model_difficulty = 1.0
    # model_difficulty = difficulty
    width = 424
    height = 240
    grayscale = False
    use_net = True
    use_seg = False
    use_depth = False
    use_flow = False
    use_last_frame = False
    crop = (120, 60)
    downscale = 2
    num_envs = 3
    record = False
    variance_debug = False
    train_eval_count = 12

    model_name = 'model_{w}_{h}{g}{s}{d}{f}{l}.zip'.format(w=width, h=height, g='_grayscale' if grayscale else '',
                                                       s='_seg' if use_seg else '', d='_depth' if use_depth else '',
                                                       f='_flow' if use_flow else '', l='_uselast' if use_last_frame else '')

    if action == 'train':
        def make_env(monitor=False, with_gui=False, eval=False, eval_count=None):
            env = CutterEnv(width, height, grayscale=grayscale, use_net=use_net, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow,
                             use_last_frame=use_last_frame, use_gui=with_gui, max_elapsed_time=1.0, max_vel=0.75, difficulty=0.0, debug=False,
                            eval=eval, eval_count=eval_count, crop=crop, downscale=downscale)
            if monitor:
                env = Monitor(env)
            return env

        env = VecTransposeImage(SubprocVecEnv([make_env] * num_envs))
        eval_env = (VecTransposeImage(DummyVecEnv([partial(make_env, monitor=True, eval=True, eval_count=train_eval_count)])))

        n_steps = 600 // num_envs
        batch_size = 60

        model = PPO("CnnPolicy", env, batch_size=batch_size, n_steps=n_steps, verbose=1, device='auto')
        if train_use_pretrained:
            print('Loading pre-trained network...')
            net = model.policy
            net.load_state_dict(torch.load('pretrained_agent.weights'))

        print('Learning...')
        # difficulties = [0.0, 0.25, 0.5, 0.75, 1.0]
        difficulties = [1.0]
        steps_per_difficulty = 100020
        for i, difficulty in enumerate(difficulties):
            difficulty_str = str(difficulty).replace('.', '_')
            model_file = f'best_model_{difficulty_str}.zip'
            if not os.path.exists(model_file):

                print('STARTING LEARNING FOR DIFFICULTY {}'.format(difficulty))
                env.env_method('update_difficulty', difficulty)
                eval_env.env_method('update_difficulty', difficulty)

                eval_callback = EvalCallback(eval_env, best_model_save_path='./', log_path='./', eval_freq=n_steps,
                                             n_eval_episodes=train_eval_count,
                                             deterministic=True, render=False)
                model.learn(total_timesteps=steps_per_difficulty, callback=eval_callback)
                os.rename('evaluations.npz', f'evaluations_{difficulty_str}.npz')
                os.rename('best_model_1_0_fullsize.zip', f'best_model_{difficulty_str}.zip')
            else:
                print('Difficulty {} has already been learned!'.format(difficulty))
            model = model.load(f'best_model_{difficulty_str}.zip', env=env)


    elif action == 'eval':
        timesteps, buffer_size = (150, 450) if record else (1000, 0)

        # env = CutterEnv(159, 90, use_seg=use_seg, use_depth=use_depth, use_gui=True, max_elapsed_time=2.5, max_vel=0.05, debug=True)
        env = CutterEnv(width, height, grayscale=grayscale, use_net=use_net, use_seg=use_seg, use_depth=use_depth, use_flow=use_flow, use_last_frame=use_last_frame,
                        use_gui=True, max_elapsed_time=1.0, max_vel=0.30, debug=True, img_buffer_size=buffer_size,
                        eval=True, eval_count=None, difficulty=difficulty, crop=crop, downscale=downscale)
        model = PPO("CnnPolicy", env, verbose=1)
        if use_trained:
            diff_str = str(model_difficulty).replace('.', '_')
            model_file = 'best_model_{}_midpoint.zip'.format(diff_str)
            if os.path.exists(model_file):
                model = model.load(model_file)
                print('Using best model!')
        obs = env.reset()
        all_dists = []
        action_hist = []
        try:
            for i in range(timesteps):
                if use_trained:
                    action, _states = model.predict(obs, deterministic=True)
                    if variance_debug:
                        check_input_variance(model, obs, output=True)
                else:
                    # action = env.action_space.sample()
                    action = np.array([0.0, -0.1])

                env.set_action(action)
                action_hist.append(action)
                obs, reward, done, info = env.step(action, realtime=True)
                # env.render()
                if done:
                    all_dists.append(env.get_cutter_dist())
                    obs = env.reset()
        finally:
            file_id = int(time.time())
            np.save('data/hist_sim_{}.npy'.format(file_id), np.array(action_hist))

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
