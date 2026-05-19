"""Depth-from-bbox helpers and pinhole back-projection."""

import numpy as np


def bbox_median_depth(depth_map, bbox):
    """Median of finite depth values inside ``bbox`` (x1,y1,x2,y2).

    Returns ``(z_mm, valid_count)``. ``z_mm`` is NaN if the bbox contains no
    finite depth samples.
    """
    if depth_map is None:
        return (float("nan"), 0)
    h, w = depth_map.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return (float("nan"), 0)
    patch = depth_map[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch)]
    if valid.size == 0:
        return (float("nan"), 0)
    return (float(np.median(valid)), int(valid.size))


def bbox_robust_depth(depth_map, bbox, shrink=0.6):
    """Median of finite depth values inside a *shrunken* bbox.

    Avoids background bleed at the bbox edges that hurts tiny targets.
    ``shrink`` is the inner fraction kept (0 < shrink <= 1); 0.6 keeps the
    central 60% in each axis. Returns ``(z_mm, valid_count)``; NaN if empty.
    """
    if depth_map is None:
        return (float("nan"), 0)
    h, w = depth_map.shape
    x1, y1, x2, y2 = bbox
    bw = max(1, int(x2) - int(x1))
    bh = max(1, int(y2) - int(y1))
    s = float(shrink)
    if s <= 0 or s >= 1.0:
        s = max(1e-3, min(1.0, s))
    cx = (int(x1) + int(x2)) / 2.0
    cy = (int(y1) + int(y2)) / 2.0
    hw = bw * s / 2.0
    hh = bh * s / 2.0
    sx1 = max(0, int(round(cx - hw)))
    sy1 = max(0, int(round(cy - hh)))
    sx2 = min(w, int(round(cx + hw)))
    sy2 = min(h, int(round(cy + hh)))
    if sx2 <= sx1 or sy2 <= sy1:
        return (float("nan"), 0)
    patch = depth_map[sy1:sy2, sx1:sx2]
    valid = patch[np.isfinite(patch)]
    if valid.size == 0:
        return (float("nan"), 0)
    return (float(np.median(valid)), int(valid.size))


def pixel_to_camera_xyz(u, v, z_mm, P1):
    """Back-project rectified-left pixel + depth to camera-frame XYZ (mm).

    Uses the rectified projection matrix ``P1`` from ``CalibrationResult``
    (3x4). Camera frame: Z forward, X right, Y down (OpenCV optical).
    """
    fx = float(P1[0, 0])
    fy = float(P1[1, 1])
    cx = float(P1[0, 2])
    cy = float(P1[1, 2])
    X = (float(u) - cx) * z_mm / fx
    Y = (float(v) - cy) * z_mm / fy
    return (X, Y, float(z_mm))
