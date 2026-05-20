"""Eye-to-hand calibration helpers for ELP camera -> MG400 robot transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def solve_rigid_transform(src_pts, dst_pts):
    """Solve ``dst = R @ src + t`` with Umeyama SVD, no scale.

    Points are expected in meters. ``src_pts`` are camera-frame points and
    ``dst_pts`` are robot-frame points for the ELP/MG400 eye-to-hand setup.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src_pts and dst_pts must both be Nx3 arrays")
    if len(src) < 4:
        raise ValueError("At least 4 point pairs are required")

    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)

    D = np.eye(3)
    D[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ D @ U.T
    t = dst_c - R @ src_c
    return R, t


def reprojection_errors_mm(cam_pts, robot_pts, R, t):
    """Return per-point transform errors in millimeters."""
    cam = np.asarray(cam_pts, dtype=np.float64)
    rob = np.asarray(robot_pts, dtype=np.float64)
    pred = (np.asarray(R, dtype=np.float64) @ cam.T).T + np.asarray(t, dtype=np.float64)
    return (np.linalg.norm(pred - rob, axis=1) * 1000.0).tolist()


def matrix_to_quat_xyzw(R):
    """Convert a 3x3 rotation matrix to ROS quaternion order ``[x,y,z,w]``."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = (tr + 1.0) ** 0.5 * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = (1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5 * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = (1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5 * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = (1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5 * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    return (q / n).tolist() if n > 0 else [0.0, 0.0, 0.0, 1.0]


def transform_matrix(R, t):
    """Build a 4x4 homogeneous transform from ``R`` and ``t``."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def save_hand_eye_yaml(
    path,
    R,
    t,
    cam_pts,
    robot_pts,
    mean_error_mm,
    max_error_mm,
    parent_frame="robot_base",
    child_frame="camera_optical_frame",
):
    """Save calibration in a format usable by the detection ROS2 static TF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "parent_frame": str(parent_frame),
        "child_frame": str(child_frame),
        "transform_4x4": transform_matrix(R, t).tolist(),
        "R": np.asarray(R, dtype=float).tolist(),
        "t_m": np.asarray(t, dtype=float).tolist(),
        "translation_m": np.asarray(t, dtype=float).tolist(),
        "rotation_xyzw": matrix_to_quat_xyzw(R),
        "num_points": int(len(cam_pts)),
        "mean_error_mm": float(mean_error_mm),
        "max_error_mm": float(max_error_mm),
        "camera_points_m": [list(map(float, p)) for p in cam_pts],
        "robot_points_m": [list(map(float, p)) for p in robot_pts],
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return data


def load_hand_eye_yaml(path):
    """Load hand-eye YAML and return ``(data, T_robot_camera)``."""
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if "transform_4x4" in data:
        T = np.asarray(data["transform_4x4"], dtype=np.float64)
    else:
        R = np.asarray(data["R"], dtype=np.float64)
        t = np.asarray(data.get("t_m", data.get("translation_m")), dtype=np.float64)
        T = transform_matrix(R, t)
    return data, T


def static_tf_config(data):
    """Return the ``ros2.static_tf`` config block for ``config/detect.yaml``."""
    return {
        "parent": data.get("parent_frame", "robot_base"),
        "child": data.get("child_frame", "camera_optical_frame"),
        "translation_m": data["translation_m"],
        "rotation_xyzw": data["rotation_xyzw"],
    }
