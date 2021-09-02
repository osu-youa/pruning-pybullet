import numpy as np
from scipy.optimize import fmin
from scipy.spatial.transform import Rotation

# NEW PARAMS
# PARAMS ARE IN FLIPPED FRAME WHERE X POINTS LEFT, Y POINTS UP

# CAMERA_BASE_OFFSET = np.array([-0.063175 - 0.0325, 0.06, 0.082929])
CAMERA_BASE_OFFSET = np.array([-0.063179, 0.077119, 0.0420027])
EYE_BASE_OFFSET = np.array([0.0325, 0, 0])

def create_tf(xyz=None, rpy=None):
    if xyz is None:
        xyz = np.zeros(3)
    if rpy is None:
        rpy = np.zeros(3)

    tf = np.identity(4)
    tf[:3,3] = xyz
    tf[:3,:3] = Rotation.from_euler('xyz', rpy, degrees=False).as_matrix()
    return tf


def compute_eye_transform(pan, tilt, xyz_offset=None, base_tf=None):

    if xyz_offset is None:
        xyz_offset = np.zeros(3)

    eye_tf = create_tf(xyz=EYE_BASE_OFFSET)

    camera_tf = create_tf(xyz=CAMERA_BASE_OFFSET + xyz_offset, rpy=[0, pan, 0])
    tilt_tf = create_tf(xyz=np.zeros(3), rpy=[tilt, 0, 0])


    final_tf = camera_tf @ eye_tf @ tilt_tf
    if base_tf is not None:
        final_tf = base_tf @ final_tf

    return final_tf


def get_view_matrix(pan, tilt, xyz_offset, base_tf=None):
    import pybullet as pb

    tf = compute_eye_transform(pan, tilt, xyz_offset, base_tf=base_tf)
    viewpoint = tf[:3,3]
    lookat_point = (tf @ np.array([0, 0, 0.1, 1]))[:3]

    return np.reshape(pb.computeViewMatrix(cameraEyePosition=viewpoint,
                                   cameraTargetPosition=lookat_point,
                                   cameraUpVector=[0, 0, 1]), (4, 4)).T
