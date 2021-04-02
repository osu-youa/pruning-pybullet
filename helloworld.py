import pybullet as pb
import time
import pybullet_data
import numpy as np
import os
# from ipdb import set_trace

joint_indexes = []

class URDFRobot:
    def __init__(self, urdf_path, *args, **kwargs):
        self.robot_id = pb.loadURDF(urdf_path, *args, **kwargs)
        self.joint_names_to_ids = {}
        self.num_joints = pb.getNumJoints(self.robot_id)
        self.joint_limits = {}

        self.revolute_joints = []
        self.revolute_and_prismatic_joints = []


        for i_joint in range(self.num_joints):
            joint_info = pb.getJointInfo(self.robot_id, i_joint)
            joint_name = joint_info[1].decode('utf8')
            joint_type = joint_info[2]
            joint_limits = (joint_info[8], joint_info[9])

            self.joint_names_to_ids[joint_name] = i_joint
            self.joint_limits[i_joint] = joint_limits

            if joint_type in {pb.JOINT_PRISMATIC, pb.JOINT_REVOLUTE}:
                self.revolute_and_prismatic_joints.append(i_joint)
                if joint_type == pb.JOINT_REVOLUTE:
                    self.revolute_joints.append(i_joint)

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
            pb.resetJointState(self.robot_id, joint_id, position, velocity)


    def get_link_kinematics(self, link_name_or_id, use_com_frame=False, as_matrix=False):

        link_id = self.convert_link_name(link_name_or_id)

        pos_idx = 4
        quat_idx = 5
        if use_com_frame:
            pos_idx = 0
            quat_idx = 1

        rez = pb.getLinkState(self.robot_id, link_id, computeForwardKinematics=True)
        position, orientation = rez[pos_idx], rez[quat_idx]

        if as_matrix:
            tf = np.identity(4)
            tf[:3, :3] = np.array(pb.getMatrixFromQuaternion(orientation)).reshape((3, 3))
            tf[:3, 3] = position
            return tf
        else:
            return position, orientation

    def get_z_offset_view_matrix(self, link_name_or_id):
        tf = self.get_link_kinematics(link_name_or_id, use_com_frame=False, as_matrix=True)
        target_position = (tf @ np.array([0, 0, 1, 1]))[:3]
        return pb.computeViewMatrix(cameraEyePosition=tf[:3,3], cameraTargetPosition=target_position,
                                    cameraUpVector=[0, 0, 1])


    def solve_end_effector_ik(self, link_name_or_id, target_position, target_orientation=None):
        link_id = self.convert_link_name(link_name_or_id)
        kwargs = {'targetPosition': target_position}
        if target_orientation is not None:
            kwargs['targetOrientation'] = target_orientation
        return pb.calculateInverseKinematics(self.robot_id, link_id, **kwargs)


    # def solve_end_effector_change_ik(self, link_name_or_id, desired_change):
    #
    #     link_id = self.convert_link_name(link_name_or_id)
    #
    #     tf = self.get_link_kinematics(link_id, use_com_frame=False, as_matrix=True)
    #     homog = np.array([0, 0, 0, 1])
    #     homog[:3] = desired_change
    #     target_position = (tf @ homog)[:3]
    #     return pb.calculateInverseKinematics(self.robot_id, link_id, targetPosition=target_position)

    def set_control_target(self, targets, control_type=pb.POSITION_CONTROL, include_prismatic=False):
        joint_ids = self.revolute_and_prismatic_joints if include_prismatic else self.revolute_joints
        pb.setJointMotorControlArray(self.robot_id, joint_ids, control_type, targetPositions=targets)


if __name__ == '__main__':

    DEBUG = False
    TRELLIS_DEPTH = 0.875
    CAMERA_OFFSET = np.array([0.6, 0.00, 0.0])

    # arm_location = '/home/main/catkin_ws/src/FREDS-MP/fredsmp_utils/robots/ur5/ur5e_cutter_new_mounted_calibrated_precise.urdf'
    arm_location = os.path.join('robots', 'ur5e_cutter_new_calibrated_precise.urdf')


    # Find camera parameters
    projectionMatrix = pb.computeProjectionMatrixFOV(
        fov=47.5,
        aspect=848/480,
        nearVal=0.1,
        farVal=3.1)
    # TODO: Confirm Intrinsic Parameters for D435

    # Environment setup
    physicsClient = pb.connect(pb.GUI)#or p.DIRECT for non-graphical version
    pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    pb.setGravity(0, 0, -9.8)
    planeId = pb.loadURDF("plane.urdf")

    # Load in the arm and configure the joint control
    startPos = [0,0,0.02]
    startOrientation = pb.getQuaternionFromEuler([0, 0, 0])

    home_joints = [-1.5708, -2.2689, -1.3963, 0.52360, 1.5708, 3.14159]
    robot = URDFRobot(arm_location, startPos, startOrientation)
    robot.reset_joint_states(home_joints)

    # Load in other objects in the environment
    scaling = 1.35
    tree_id = pb.loadURDF('models/trellis-model.urdf', [0, TRELLIS_DEPTH, 0.02 * scaling], [0, 0, 0.7071, 0.7071], globalScaling=scaling)
    # branch_id = pb.loadURDF('models/test_branch.urdf', [0, 0.85, 1.8])
    # pb.createSoftBodyAnchor(branch_id, 0, -1, -1)

    # Initialize stuff for simulation
    DESIRED_VELOCITY = np.array([0, 0, 0.05])
    DESIRED_STEP = DESIRED_VELOCITY / 240


    #     homog = np.array([0, 0, 0, 1])
    #     homog[:3] = desired_change
    #     target_position = (tf @ homog)[:3]

    # Starting
    start_pos, start_orientation = robot.get_link_kinematics('cutpoint')
    state_id = pb.saveState()

    start_base, _ = robot.get_link_kinematics('mount_base_joint')

    for z_offset in np.linspace(0, 0.20, 50, endpoint=False):

        print('Working on Z-Offset {:.4f}'.format(z_offset))
        pb.restoreState(stateId=state_id)
        desired_start_pos = np.array(start_pos) + np.array([0, 0, z_offset])
        ik = robot.solve_end_effector_ik('cutpoint', desired_start_pos, start_orientation)
        robot.reset_joint_states(ik)
        tf = robot.get_link_kinematics('cutpoint', use_com_frame=False, as_matrix=True)

        camera_look_pos = tf[:3,3].copy()
        camera_look_pos[1] = TRELLIS_DEPTH

        # Main simulation loop
        for i in range (500):


            # print('{} steps elapsed'.format(i))

            # Compute the camera view matrix and update the corresponding image
            # view_matrix = robot.get_z_offset_view_matrix('camera_mount')
            view_matrix = pb.computeViewMatrix(cameraEyePosition=start_base + CAMERA_OFFSET, cameraTargetPosition=camera_look_pos, cameraUpVector=[0, 0, 1])

            width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                width=212,
                height=120,
                viewMatrix=view_matrix,
                projectionMatrix=projectionMatrix)

            # Compute the new IKs to move the robot
            frame_goal = DESIRED_STEP * i
            homog = np.array([0., 0., 0., 1.])
            homog[:3] = frame_goal
            target_position = (tf @ homog)[:3]

            ik = robot.solve_end_effector_ik('cutpoint', target_position, start_orientation)
            robot.set_control_target(ik)
            if not i % 240 and DEBUG:
                start_pos, _ = robot.get_link_kinematics('cutpoint')
                print('Step {}:\nTarget: ({:.3f}, {:.3f}, {:.3f})\nActual: ({:.3f}, {:.3f}, {:.3f})'.format(i, *target_position, *start_pos))

            time.sleep(1. / 240.)
            pb.stepSimulation()


    pb.disconnect()

