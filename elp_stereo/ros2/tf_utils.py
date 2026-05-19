"""TF helpers for the detection publisher."""

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped


def static_transform_from_config(tf_cfg):
    """Build a ``TransformStamped`` from a config dict.

    Expected keys:
        parent           – parent frame_id
        child            – child frame_id
        translation_m    – [x, y, z] meters
        rotation_xyzw    – [qx, qy, qz, qw]
    """
    msg = TransformStamped()
    msg.header.frame_id = str(tf_cfg["parent"])
    msg.child_frame_id = str(tf_cfg["child"])
    t = tf_cfg.get("translation_m", [0.0, 0.0, 0.0])
    q = tf_cfg.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
    msg.transform.translation.x = float(t[0])
    msg.transform.translation.y = float(t[1])
    msg.transform.translation.z = float(t[2])
    msg.transform.rotation.x = float(q[0])
    msg.transform.rotation.y = float(q[1])
    msg.transform.rotation.z = float(q[2])
    msg.transform.rotation.w = float(q[3])
    return msg


def _quat_to_matrix(qx, qy, qz, qw):
    """Quaternion -> 3x3 rotation matrix (no external dep)."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx = qx * qx * s; yy = qy * qy * s; zz = qz * qz * s
    xy = qx * qy * s; xz = qx * qz * s; yz = qy * qz * s
    wx = qw * qx * s; wy = qw * qy * s; wz = qw * qz * s
    return np.array([
        [1.0 - (yy + zz),       xy - wz,             xz + wy],
        [      xy + wz,    1.0 - (xx + zz),          yz - wx],
        [      xz - wy,         yz + wx,        1.0 - (xx + yy)],
    ], dtype=np.float64)


def lookup_camera_to_robot(tf_buffer, target_frame, source_frame, stamp=None):
    """Look up ``source_frame -> target_frame`` as a 4x4 homogeneous matrix (meters).

    Returns ``None`` if the transform is not yet available.
    """
    try:
        t = rclpy.time.Time() if stamp is None else stamp
        tf = tf_buffer.lookup_transform(target_frame, source_frame, t)
    except Exception:  # tf2 raises various — treat all as "not available"
        return None
    tr = tf.transform.translation
    ro = tf.transform.rotation
    R = _quat_to_matrix(ro.x, ro.y, ro.z, ro.w)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tr.x, tr.y, tr.z]
    return M


def transform_point_mm(M, xyz_mm):
    """Apply a 4x4 meters-frame transform to a (X,Y,Z) point in mm.

    Returns the transformed point in meters: ``(x_m, y_m, z_m)``.
    """
    x, y, z = xyz_mm
    p = np.array([x / 1000.0, y / 1000.0, z / 1000.0, 1.0])
    q = M @ p
    return (float(q[0]), float(q[1]), float(q[2]))
