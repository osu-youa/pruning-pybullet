import pybullet as pb
import time
import pybullet_data
import numpy as np
import os

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


    def get_force_reading(self, joint):
        return pb.getJointState(self.robot_id, self.joint_names_to_ids[joint])[2]

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

    def set_control_target(self, targets, control_type=pb.POSITION_CONTROL, include_prismatic=False):
        joint_ids = self.revolute_and_prismatic_joints if include_prismatic else self.revolute_joints
        pb.setJointMotorControlArray(self.robot_id, joint_ids, control_type, targetPositions=targets)

    def enable_force_torque_readings(self, joint_name):
        joint_id = self.joint_names_to_ids[joint_name]
        pb.enableJointForceTorqueSensor(self.robot_id, joint_id, True)

if __name__ == '__main__':

    DEBUG = False
    TRELLIS_DEPTH = 0.875
    arm_location = os.path.join('robots', 'ur5e_cutter_new_calibrated_precise.urdf')


    # Find camera parameters
    projectionMatrix = pb.computeProjectionMatrixFOV(
        fov=47.5,
        aspect=848/480,
        nearVal=0.01,
        farVal=3.1)
    # TODO: Confirm Intrinsic Parameters for D435

    # Environment setup
    physicsClient = pb.connect(pb.GUI)#or p.DIRECT for non-graphical version

    # # Process concave model files as necessary
    # file_path = 'robots/ur5e/collision/new-cutter.obj'
    # name_out = 'new-cutter-convexified.obj'
    # name_log = 'log.txt'
    # pb.vhacd(file_path, name_out, name_log)

    pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    pb.setGravity(0, 0, -9.8)
    planeId = pb.loadURDF("plane.urdf")

    # Load in the arm and configure the joint control
    startPos = [0,0,0.02]
    startOrientation = pb.getQuaternionFromEuler([0, 0, 0])

    home_joints = [-1.5708, -2.2689, -1.3963, 0.52360, 1.5708, 3.14159]
    robot = URDFRobot(arm_location, startPos, startOrientation)
    robot.reset_joint_states(home_joints)
    robot.enable_force_torque_readings('wrist_3_link-tool0_fixed_joint')

    # Load in other objects in the environment
    scaling = 1.35
    tree = URDFRobot('models/trellis-model.urdf', basePosition=[0, TRELLIS_DEPTH, 0.02 * scaling],
                        baseOrientation=[0, 0, 0.7071, 0.7071], globalScaling=scaling)
    # tree_id = pb.loadURDF('models/trellis-model.urdf', [0, TRELLIS_DEPTH, 0.02 * scaling], [0, 0, 0.7071, 0.7071], globalScaling=scaling)
    # branch_id = pb.loadURDF('models/test_branch.urdf', [0, 0.85, 1.8])
    # pb.createSoftBodyAnchor(branch_id, 0, -1, -1)

    # Initialize stuff for simulation
    FORWARD_VELOCITY = 0.05
    VERTICAL_CONTROL_VELOCITIES = [0.0, 0.01, -0.01]
    ACTION_STEP = 24


    #     homog = np.array([0, 0, 0, 1])
    #     homog[:3] = desired_change
    #     target_position = (tf @ homog)[:3]

    # Starting
    start_pos, start_orientation = robot.get_link_kinematics('cutpoint')
    state_id = pb.saveState()

    start_base, _ = robot.get_link_kinematics('mount_base_joint')

    for _ in range(10):
        # for z_offset in np.linspace(0.06, 0.20, 50, endpoint=False):

        pb.restoreState(stateId=state_id)

        target_id = np.random.randint(len(tree.joint_names_to_ids))
        target_pos = tree.get_link_kinematics(target_id)[0]

        robot_start_pos = target_pos - np.array([0, np.random.uniform(0.08, 0.12), 0])
        robot_start_pos += np.random.uniform(-0.05, 0.05, 3) * np.array([1, 0, 1])

        ik = robot.solve_end_effector_ik('cutpoint', robot_start_pos, start_orientation)
        robot.reset_joint_states(ik)

        target_tf = robot.get_link_kinematics('cutpoint', use_com_frame=False, as_matrix=True)
        control_action = 0

        # Main simulation loop
        for i in range (640):

            if not i % 240:
                print('{} steps elapsed'.format(i))

            if not i % 24:
                control_action = np.random.randint(3)
                cutter_loc = robot.get_link_kinematics('cutpoint', use_com_frame=False)[0]
                target_loc = tree.get_link_kinematics(target_id)[0]

                d = np.linalg.norm(np.array(target_loc) - np.array(cutter_loc))
                print('Dist: {:.3f}m'.format(d))



            # Compute the camera view matrix and update the corresponding image
            view_matrix = robot.get_z_offset_view_matrix('camera_mount')
            width, height, rgbImg, depthImg, segImg = pb.getCameraImage(
                width=212,
                height=120,
                viewMatrix=view_matrix,
                projectionMatrix=projectionMatrix,
                renderer=pb.ER_BULLET_HARDWARE_OPENGL
            )

            # Compute the new IKs to move the robot
            delta = np.array([0., VERTICAL_CONTROL_VELOCITIES[control_action] / 240, FORWARD_VELOCITY / 240, 1.])
            target_position = (target_tf @ delta)[:3]
            target_tf[:3,3] = target_position

            ik = robot.solve_end_effector_ik('cutpoint', target_position, start_orientation)
            robot.set_control_target(ik)
            if not i % 240 and DEBUG:
                start_pos, _ = robot.get_link_kinematics('cutpoint')
                print('Step {}:\nTarget: ({:.3f}, {:.3f}, {:.3f})\nActual: ({:.3f}, {:.3f}, {:.3f})'.format(i, *target_position, *start_pos))

            time.sleep(1. / 240.)
            pb.stepSimulation()

    pb.disconnect()

