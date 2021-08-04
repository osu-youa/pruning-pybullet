import numpy as np
from scipy.optimize import fmin
from scipy.spatial.transform import Rotation

# NEW PARAMS
# PARAMS ARE IN FLIPPED FRAME WHERE X POINTS LEFT, Y POINTS UP

CAMERA_MOUNT_WIDTH = 0.01014
MOUNT_PAN_OFFSET = np.array([0.0, 0.07220, CAMERA_MOUNT_WIDTH / 2])
PAN_TILT_OFFSET = np.array([0.0, 0.06131, 0.002])
TILT_EYE_OFFSET = np.array([0.0325, 0.02682 + 0.01236, 0.01200])

def create_tf(xyz=None, rpy=None):
    if xyz is None:
        xyz = np.zeros(3)
    if rpy is None:
        rpy = np.zeros(3)

    tf = np.identity(4)
    tf[:3,3] = xyz
    tf[:3,:3] = Rotation.from_euler('xyz', rpy, degrees=False).as_matrix()
    return tf


def compute_eye_transform(pan, tilt, base_tf=None):

    adjustment_tf = np.identity(4)
    # adjustment_tf = create_tf(rpy=[0, 0, np.pi])
    pan_tf = create_tf(xyz=MOUNT_PAN_OFFSET, rpy=[0, pan, 0])
    tilt_tf = create_tf(xyz=PAN_TILT_OFFSET, rpy=[tilt, 0, 0])
    eye_tf = create_tf(xyz=TILT_EYE_OFFSET)

    final_tf = adjustment_tf @ pan_tf @ tilt_tf @ eye_tf
    if base_tf is not None:
        final_tf = base_tf @ final_tf

    return final_tf


def get_view_matrix(pan, tilt, base_tf=None):
    import pybullet as pb

    tf = compute_eye_transform(pan, tilt, base_tf=base_tf)
    viewpoint = tf[:3,3]
    lookat_point = (tf @ np.array([0, 0, 0.1, 1]))[:3]

    return np.reshape(pb.computeViewMatrix(cameraEyePosition=viewpoint,
                                   cameraTargetPosition=lookat_point,
                                   cameraUpVector=[0, 0, 1]), (4, 4)).T

#
#
#
#
#
#
#
#
#
#
#
#
# # PARAMS
# TARGET = np.array([-0.09286, -0.00511, 0.09707])          # Flipped for simplicity
# CAMERA_RAISE = 0.075                                     # Positive for simplicity
# EYE_CAMERA_BASE_TF = np.array([0.0325, 0, 0])           # Dist from center of camera to RGB Module
#
# def dist_func(rot_array):
#
#     eye_tf = get_eye_tf(rot_array)
#     eye_origin = eye_tf[:3,3]
#     eye_z_axis = eye_tf[:3,2]
#
#     return get_ray_point_dist(TARGET, eye_origin, eye_z_axis)
#
# def get_eye_tf(rot_array):
#     y_rot, x_rot = rot_array
#
#     y_rot_mat = np.identity(4)
#     y_rot_mat[1, 3] = CAMERA_RAISE
#     y_rot_mat[:3, :3] = Rotation.from_euler('xyz', [0, y_rot, 0], degrees=False).as_matrix()
#
#     x_rot_mat = np.identity(4)
#     x_rot_mat[:3, 3] = EYE_CAMERA_BASE_TF
#     x_rot_mat[:3, :3] = Rotation.from_euler('xyz', [x_rot, 0, 0], degrees=False).as_matrix()
#
#     eye_tf = y_rot_mat @ x_rot_mat
#     return eye_tf
#
# def get_ray_point_dist(point, ray_origin, ray_vector):
#     return np.linalg.norm((point - ray_origin) - (point - ray_origin).dot(ray_vector) * ray_vector)
#
# def solve():
#     xopt, fopt, _, _, _ = fmin(dist_func, np.array([0.0, 0.0]), full_output=True)
#     tf = get_eye_tf(xopt)
#     return tf
#
# if __name__ == '__main__':
#     xopt, fopt, _, _, _ = fmin(dist_func, np.array([0.0, 0.0]), full_output=True)
#     tf = get_eye_tf(xopt)
#     print(tf)
#     print(np.degrees(xopt))