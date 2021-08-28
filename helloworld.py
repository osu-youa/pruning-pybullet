import pybullet as pb
import time
import pybullet_data
import numpy as np
import trimesh
import os
from scipy.spatial.transform import Rotation

# This is a suboptimal hack involving some sort of Numpy installation issue which was causing crashes when doing matrix operations!
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# For debugging
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class URDFRobot:
    def __init__(self, urdf_path, *args, **kwargs):
        self.robot_id = pb.loadURDF(urdf_path, *args, **kwargs)
        self.client_kwarg = {}
        if 'physicsClientId' in kwargs:
            self.client_kwarg['physicsClientId'] = kwargs['physicsClientId']

        self.joint_names_to_ids = {}
        self.num_joints = pb.getNumJoints(self.robot_id, **self.client_kwarg)
        self.joint_limits = {}
        self.revolute_joints = []
        self.revolute_and_prismatic_joints = []

        for i_joint in range(self.num_joints):
            joint_info = pb.getJointInfo(self.robot_id, i_joint, **self.client_kwarg)
            joint_name = joint_info[1].decode('utf8')
            joint_type = joint_info[2]
            joint_limits = (joint_info[8], joint_info[9])

            self.joint_names_to_ids[joint_name] = i_joint
            self.joint_limits[i_joint] = joint_limits

            if joint_type in {pb.JOINT_PRISMATIC, pb.JOINT_REVOLUTE}:
                self.revolute_and_prismatic_joints.append(i_joint)
                if joint_type == pb.JOINT_REVOLUTE:
                    self.revolute_joints.append(i_joint)

        self.ghost_bodies = {}
        self.ghost_body_tfs = {}


    def convert_link_name(self, name):
        if isinstance(name, int):
            return name
        return self.joint_names_to_ids[name]

    def reset_joint_states(self, positions, velocities=None, include_prismatic=False):
        joint_ids = self.revolute_and_prismatic_joints if include_prismatic else self.revolute_joints
        if velocities is None:
            velocities = [0.0] * len(joint_ids)
        for joint_id, position, velocity in zip(joint_ids, positions, velocities):
            limit_low, limit_high = self.joint_limits[joint_id]
            if not limit_low <= position <= limit_high:
                raise ValueError("For joint {}, desired position {:.3f} is not in limits [{:.3f}, {:.3f}]".format(joint_id, position, limit_low, limit_high))
            pb.resetJointState(self.robot_id, joint_id, position, velocity, **self.client_kwarg)


    def get_force_reading(self, joint):
        return pb.getJointState(self.robot_id, self.joint_names_to_ids[joint], **self.client_kwarg)[2]

    def get_link_kinematics(self, link_name_or_id, use_com_frame=False, as_matrix=False):

        link_id = self.convert_link_name(link_name_or_id)

        pos_idx = 4
        quat_idx = 5
        if use_com_frame:
            pos_idx = 0
            quat_idx = 1

        rez = pb.getLinkState(self.robot_id, link_id, computeForwardKinematics=True, **self.client_kwarg)
        position, orientation = rez[pos_idx], rez[quat_idx]

        if as_matrix:
            tf = np.identity(4)
            tf[:3, :3] = Rotation.from_quat(orientation).as_matrix()
            tf[:3, 3] = position
            return tf
        else:
            return position, orientation

    def get_z_offset_view_matrix(self, link_name_or_id, apply_offset=None):
        tf = self.get_link_kinematics(link_name_or_id, use_com_frame=False, as_matrix=True)
        if apply_offset is not None:
            tf = tf @ apply_offset
        target_position = (tf @ np.array([0, 0, 1, 1]))[:3]
        return pb.computeViewMatrix(cameraEyePosition=tf[:3,3], cameraTargetPosition=target_position,
                                    cameraUpVector=[0, 0, 1])


    def move_end_effector_ik(self, link_name_or_id, target_position, target_orientation=None, threshold=0.005, retries=3):
        # Hack for the fact that somehow the IK doesn't get solved properly
        for _ in range(retries):
            ik = self.solve_end_effector_ik(link_name_or_id, target_position, target_orientation=target_orientation,
                                            threshold=threshold)
            self.reset_joint_states(ik)
            pos, _ = self.get_link_kinematics(link_name_or_id)
            offset = np.linalg.norm(np.array(target_position) - pos)
            if offset < threshold:
                return True
        return False

    def solve_end_effector_ik(self, link_name_or_id, target_position, target_orientation=None, threshold=None, max_iters=20):

        link_id = self.convert_link_name(link_name_or_id)
        kwargs = {'targetPosition': target_position}
        if target_orientation is not None:
            kwargs['targetOrientation'] = target_orientation
        if threshold is not None:
            kwargs['residualThreshold'] = threshold
        return pb.calculateInverseKinematics(self.robot_id, link_id, **kwargs, **self.client_kwarg)

    def set_control_target(self, targets, control_type=pb.POSITION_CONTROL, include_prismatic=False):
        joint_ids = self.revolute_and_prismatic_joints if include_prismatic else self.revolute_joints
        pb.setJointMotorControlArray(self.robot_id, joint_ids, control_type, targetPositions=targets, **self.client_kwarg)

    def enable_force_torque_readings(self, joint_name):
        joint_id = self.joint_names_to_ids[joint_name]
        pb.enableJointForceTorqueSensor(self.robot_id, joint_id, True, **self.client_kwarg)

    def determine_reachable_target_poses(self, frame, pose_list, base_ik, max_iters=20):
        """
        Given a list of poses, determines whether there exists an IK solution
        that is sufficiently close, and which also has an offset position with a sufficiently close IK solution.
        :param pose_list: A list of poses in [px py pz qx qy qz qw] form
        :param base_ik: A starting solution for the IK solver.
        :return:
        """

        POSITION_TOLERANCE = 0.005
        ORIENTATION_TOLERANCE = 0.0025
        SUM_ABS_DIFF_TOLERANCE = np.pi * 2/3

        valid = []
        for pose in pose_list:
            pose = np.array(pose)
            pos_target = pose[:3]
            quat_target = pose[3:7]

            self.reset_joint_states(base_ik)
            self.move_end_effector_ik(frame, target_position=pos_target, target_orientation=quat_target)
            pos, quat = self.get_link_kinematics(frame, as_matrix=False)
            d_pos = np.linalg.norm(pos_target - pos)
            d_quat = np.linalg.norm(quat_target - quat)
            if d_pos > POSITION_TOLERANCE or d_quat > ORIENTATION_TOLERANCE:
                continue

            #
            # pose_mat = np.identity(4)
            # pose_mat[:3,3] = pos_target
            # pose_mat[:3,:3] = Rotation.from_quat(quat_target).as_matrix()
            # homog = np.ones(4)
            # homog[:3] = vector_offset
            # pos_app = (pose_mat @ homog)[:3]
            #
            #
            #
            # ik_app = np.array(self.solve_end_effector_ik(frame, pos_app, quat_target, threshold=POSITION_TOLERANCE))
            # self.reset_joint_states(ik_app)
            # pos, quat = self.get_link_kinematics(frame, as_matrix=False)
            # d_pos = np.linalg.norm(pos_app - pos)
            # d_quat = np.linalg.norm(quat_target - quat)
            # d_iks = np.abs(ik_app - ik).sum()
            #
            # if d_pos > POSITION_TOLERANCE or d_quat > ORIENTATION_TOLERANCE or d_iks > SUM_ABS_DIFF_TOLERANCE:
            #     continue
            #
            #

            valid.append(pose)

        self.reset_joint_states(base_ik)

        return valid


    def attach_ghost_body_from_file(self, file_name, name, joint_attachment, xyz=None, rpy=None):
        """
        Attaches a ghost collision body from a model file to a given joint.
        A really hacky solution for the fact that Pybullet's Python wrapper doesn't support ghost bodies.
        You should verify visually in the URDF that the params input here match what you expect.
        """

        mesh = trimesh.load(file_name)
        assert mesh.is_watertight
        self.ghost_bodies[name] = (mesh, trimesh.proximity.ProximityQuery(mesh))
        if xyz is None:
            xyz = np.zeros(3)
        if rpy is None:
            rpy = np.zeros(3)

        tf = np.identity(4)
        tf[:3,3] = xyz
        tf[:3,:3] = Rotation.from_euler('xyz', rpy, degrees=False).as_matrix()

        self.ghost_body_tfs[name] = (joint_attachment, tf)

        if name == 'mouth':
            mins = mesh.vertices.min(axis=0)
            maxs = mesh.vertices.max(axis=0)
            rand = mins + np.random.uniform(0, 1, size=(1000, 3)) * (maxs - mins)
            dists = self.ghost_bodies[name][1].signed_distance(rand)
            to_show = rand[dists > 0]
            self.debug_test_points = to_show

    def query_ghost_body_collision(self, name, points, point_frame_tf=None, plot_debug=False):
        tf = (self.get_link_kinematics(self.ghost_body_tfs[name][0], as_matrix=True) @ self.ghost_body_tfs[name][1])
        if point_frame_tf is not None:
            if not isinstance(point_frame_tf, np.ndarray):
                point_frame_tf = self.get_link_kinematics(point_frame_tf, as_matrix=True)
            tf = np.linalg.inv(tf) @ point_frame_tf
        pts_homog = np.ones((len(points), 4))
        pts_homog[:,:3] = points
        pts_tfed = (tf @ pts_homog.T)[:3].T

        if plot_debug:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            to_show = self.debug_test_points
            ax.scatter(to_show[:, 0], to_show[:, 1], to_show[:, 2], color='red', label='Target')
            ax.scatter(pts_tfed[:,0], pts_tfed[:,1], pts_tfed[:,2], color='blue', label='Queried')
            ax.legend()
            plt.show()


        return (self.ghost_bodies[name][1].signed_distance(pts_tfed) > 0).any()