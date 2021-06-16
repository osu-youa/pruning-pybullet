import pybullet as pb
import time
import pybullet_data
import numpy as np
import os
from helloworld import URDFRobot

from stable_baselines3 import PPO

import gym
from gym import spaces


class CutterEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, use_depth=False, max_vel=0.075, action_freq=24, max_elapsed_time=3.0,
                 min_reward_dist=0.10, use_gui=False):
        super(CutterEnv, self).__init__()

        # Initialize gym parameters
        self.action_space = spaces.Box(np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, max_vel]), dtype=np.float32)    # LR, UD, Forward
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 4 if use_depth else 3), dtype=np.uint8)

        # Configuration parameters
        self.width = width
        self.height = height
        self.use_depth = use_depth
        self.action_freq = action_freq
        self.max_elapsed_time = max_elapsed_time
        self.min_reward_dist = min_reward_dist

        # State parameters
        self.target_id = 0
        self.target_tf = np.identity(4)
        self.elapsed_time = 0.0

        # Setup Pybullet simulation
        arm_location = os.path.join('robots', 'ur5e_cutter_new_calibrated_precise.urdf')
        home_joints = [-1.5708, -2.2689, -1.3963, 0.52360, 1.5708, 3.14159]
        self.robot = URDFRobot(arm_location, [0, 0, 0.02], pb.getQuaternionFromEuler([0, 0, 0]))
        self.robot.reset_joint_states(home_joints)
        self.start_orientation = self.robot.get_link_kinematics('cutpoint')[1]
        self.proj_mat = pb.computeProjectionMatrixFOV(
            fov=47.5, aspect = width / height, nearVal=0.01,
            farVal=3.0)

        scaling = 1.35
        self.tree = URDFRobot('models/trellis-model.urdf', basePosition=[0, 0.875, 0.02 * scaling],
                              baseOrientation=[0, 0, 0.7071, 0.7071], globalScaling=scaling)

    def step(self, action):
        # Execute one time step within the environment
        horizontal, vertical, forward = action
        step = self.action_freq / 240 * forward
        delta = np.array([horizontal * step, vertical * step, step, 1.0], dtype=np.float32)
        target_pos = (self.target_tf @ delta)[:3]
        self.target_tf[:3,3] = target_pos

        # Move the arm in the environment
        self.elapsed_time += self.action_freq / 240
        for _ in range(self.action_freq):
            pb.stepSimulation()

        return self.get_obs(), self.get_reward(), self.is_done(), {}


    def get_obs(self):
        # Grab the new image observation
        view_matrix = self.robot.get_z_offset_view_matrix('camera_mount')
        _, _, rgb_img, depth_img, seg_img = pb.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=self.proj_mat,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL
        )

        if self.use_depth:
            rgb_img[:, :, 3] = (depth_img * 255).astype(np.uint8)
        else:
            rgb_img = rgb_img[:, :, :3]

        return rgb_img

    def get_cutter_dist(self):
        cutter_loc = self.robot.get_link_kinematics('cutpoint', use_com_frame=False)[0]
        target_loc = self.tree.get_link_kinematics(self.target_id)[0]

        d = np.linalg.norm(np.array(target_loc) - np.array(cutter_loc))
        return d

    def get_reward(self):

        d = self.get_cutter_dist()

        if d > self.min_reward_dist:
            return 0.0
        else:
            return 1 - d / self.min_reward_dist


    def is_done(self):
        return self.elapsed_time >= self.max_elapsed_time

    def reset(self):

        self.elapsed_time = 0.0

        # Pick a target on the tree
        self.target_id = np.random.randint(len(self.tree.joint_names_to_ids))
        target_pos = self.tree.get_link_kinematics(self.target_id)[0]

        # Find valid IKs for it
        robot_start_pos = target_pos - np.array([0, np.random.uniform(0.08, 0.12), 0])
        robot_start_pos += np.random.uniform(-0.05, 0.05, 3) * np.array([1, 0, 1])

        ik = self.robot.solve_end_effector_ik('cutpoint', robot_start_pos, self.start_orientation)
        self.robot.reset_joint_states(ik)

    def render(self, mode='human', close=False):
        print('Last dist: {:.3f}'.format(self.get_cutter_dist()))

if __name__ == '__main__':

    physicsClient = pb.connect(pb.GUI)
    # pb.connect(pb.GUI if use_gui else pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.8)
    pb.loadURDF("plane.urdf")

    env = CutterEnv(212, 120, use_depth=True, use_gui=True)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()
