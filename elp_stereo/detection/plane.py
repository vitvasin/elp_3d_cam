"""Tray-plane fitter and ray-plane intersection for depth fallback.

The tray sits roughly fronto-parallel to the camera but its exact pose
varies. We fit ``n . X + d = 0`` (X in camera frame, mm) once from a
depth-map snapshot using RANSAC and refit on inliers with SVD. The fitted
plane lets us recover Z at any pixel even when stereo depth is invalid
inside a tiny worm bbox.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    n: np.ndarray   # unit normal, shape (3,)
    d: float        # plane equation: n . X + d = 0  (X in mm)
    rms: float      # mm, RMS inlier distance to plane
    mean_z_mm: float = 0.0
    tilt_deg: float = 0.0


def _depth_to_xyz(depth_map, P1):
    """Back-project a depth map (mm) to (N, 3) camera-frame points (mm).

    Returns ``(xyz, mask)`` where ``mask`` is the bool index of finite
    samples and ``xyz`` has shape ``(mask.sum(), 3)``.
    """
    fx = float(P1[0, 0]); fy = float(P1[1, 1])
    cx = float(P1[0, 2]); cy = float(P1[1, 2])
    h, w = depth_map.shape
    mask = np.isfinite(depth_map) & (depth_map > 0)
    if not mask.any():
        return np.empty((0, 3), dtype=np.float32), mask
    ys, xs = np.nonzero(mask)
    z = depth_map[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1), mask


def _fit_plane_svd(points):
    """Least-squares plane fit. Returns ``(n, d, rms)``."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n = vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(np.dot(n, centroid))
    dist = points @ n + d
    rms = float(np.sqrt(np.mean(dist * dist)))
    return n.astype(np.float64), d, rms


def fit_tray_plane(depth_map, P1, max_samples=20000, iters=200,
                   thresh_mm=5.0, min_inlier_ratio=0.3):
    """RANSAC plane fit on the back-projected depth map.

    ``max_samples`` caps the number of points used (random subsample of all
    finite-depth pixels). ``thresh_mm`` is the inlier distance threshold.
    Raises ``ValueError`` if not enough finite samples or no plane found.
    """
    xyz, _ = _depth_to_xyz(depth_map, P1)
    n_pts = xyz.shape[0]
    if n_pts < 100:
        raise ValueError(f"Not enough finite depth samples for plane fit: {n_pts}")

    if n_pts > max_samples:
        rng = np.random.default_rng(0)
        sel = rng.choice(n_pts, max_samples, replace=False)
        xyz = xyz[sel]
        n_pts = xyz.shape[0]

    rng = np.random.default_rng(1)
    best_inliers = None
    best_count = 0
    for _ in range(int(iters)):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = xyz[idx]
        v1 = p1 - p0; v2 = p2 - p0
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        d = -float(np.dot(n, p0))
        dist = np.abs(xyz @ n + d)
        inliers = dist < thresh_mm
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < max(50, int(min_inlier_ratio * n_pts)):
        raise ValueError(
            f"RANSAC failed: best inlier count {best_count} "
            f"of {n_pts} (need >= {int(min_inlier_ratio * n_pts)})"
        )

    inlier_pts = xyz[best_inliers]
    n, d, rms = _fit_plane_svd(inlier_pts)
    # Convention: normal should point toward the camera (negative Z), so the
    # plane is in front. Flip if it points away.
    if n[2] > 0:
        n = -n
        d = -d
    # Tilt = angle between plane normal and -Z (camera optical axis pointing forward).
    cos_t = float(np.clip(np.dot(n, np.array([0.0, 0.0, -1.0])), -1.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(cos_t)))
    mean_z = float(inlier_pts[:, 2].mean())
    return Plane(n=n, d=d, rms=rms, mean_z_mm=mean_z, tilt_deg=tilt_deg)


def plane_z_at(plane, u, v, P1):
    """Z (mm) at pixel ``(u, v)`` along the camera ray, intersecting ``plane``.

    Returns NaN if the ray is parallel to the plane or hits behind the camera.
    """
    fx = float(P1[0, 0]); fy = float(P1[1, 1])
    cx = float(P1[0, 2]); cy = float(P1[1, 2])
    dx = (float(u) - cx) / fx
    dy = (float(v) - cy) / fy
    direction = np.array([dx, dy, 1.0], dtype=np.float64)
    denom = float(plane.n @ direction)
    if abs(denom) < 1e-9:
        return float("nan")
    t = -plane.d / denom
    if t <= 0:
        return float("nan")
    return float(t)  # Z = t since direction[2] = 1
