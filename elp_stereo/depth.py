"""Rectification, disparity (StereoSGBM) and depth with an uncertainty model.

Depth values come from ``cv2.reprojectImageTo3D`` using the rectification ``Q``
matrix, which encodes the full rectified geometry. The scalar pinhole relation
``z = f*B/d`` is used only to derive the depth *uncertainty* and the detectable
range, not the depth value itself.

SGBM disparity is 16-subpixel fixed-point; it is divided by 16 before any use,
and non-positive disparities are treated as invalid.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# cv2.ximgproc is a contrib module; absent in some OpenCV builds. The WLS
# post-filter is disabled gracefully when it is missing.
WLS_AVAILABLE = hasattr(cv2, "ximgproc")


class Rectifier:
    """Builds and applies the stereo rectification remap from a calibration."""

    def __init__(self, calib):
        w, h = calib.image_size
        self.image_size = (w, h)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            calib.K1, calib.D1, calib.R1, calib.P1, (w, h), cv2.CV_32FC1
        )
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            calib.K2, calib.D2, calib.R2, calib.P2, (w, h), cv2.CV_32FC1
        )
        self.Q = calib.Q
        # Rectified focal length (px) and baseline (mm) from the projection mats.
        self.fx = float(calib.P1[0, 0])
        self.baseline = float(-calib.P2[0, 3] / calib.P2[0, 0])

    def rectify(self, left, right):
        """Return rectified ``(left, right)`` images."""
        lr = cv2.remap(left, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rr = cv2.remap(right, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return lr, rr


@dataclass
class PixelInfo:
    valid: bool
    depth_mm: float = 0.0
    error_mm: float = 0.0          # +/- band from disparity quantization
    range_min_mm: float = 0.0      # nearest detectable depth (global)
    range_max_mm: float = 0.0      # farthest detectable depth (global)


class DepthEngine:
    """Computes disparity/depth and answers per-pixel depth queries."""

    def __init__(self, cfg, rectifier):
        self.rectifier = rectifier
        self._cfg = cfg["depth"]
        self.delta_d = self._cfg["subpixel_delta_disparity"]
        self._build_matchers()
        # Last computed frame state (set by compute()).
        self.disparity = None      # float32, true disparity in px, NaN = invalid
        self.depth_map = None      # float32 (h, w), Z in mm, NaN = invalid

    def _build_matchers(self):
        s = self._cfg["sgbm"]
        bs = s["block_size"]
        ch = 1  # disparity computed on grayscale
        p1 = s["p1"] if s["p1"] is not None else 8 * ch * bs * bs
        p2 = s["p2"] if s["p2"] is not None else 32 * ch * bs * bs
        self.min_disparity = s["min_disparity"]
        self.num_disparities = s["num_disparities"]
        self._left_matcher = cv2.StereoSGBM_create(
            minDisparity=s["min_disparity"],
            numDisparities=s["num_disparities"],
            blockSize=bs,
            P1=p1, P2=p2,
            disp12MaxDiff=s["disp12_max_diff"],
            uniquenessRatio=s["uniqueness_ratio"],
            speckleWindowSize=s["speckle_window_size"],
            speckleRange=s["speckle_range"],
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.use_wls = self._cfg["use_wls_filter"] and WLS_AVAILABLE
        if self.use_wls:
            self._right_matcher = cv2.ximgproc.createRightMatcher(self._left_matcher)
            self._wls = cv2.ximgproc.createDisparityWLSFilter(self._left_matcher)
            self._wls.setLambda(self._cfg["wls_lambda"])
            self._wls.setSigmaColor(self._cfg["wls_sigma"])
        else:
            self._right_matcher = None
            self._wls = None

    def update_params(self, params):
        """Apply changed SGBM/WLS params (from the GUI sliders) and rebuild."""
        self._cfg["sgbm"].update(params.get("sgbm", {}))
        if "use_wls_filter" in params:
            self._cfg["use_wls_filter"] = params["use_wls_filter"]
        self._build_matchers()

    def compute(self, left_rect, right_rect):
        """Compute disparity + depth from a rectified pair. Returns the depth map."""
        gl = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        raw_left = self._left_matcher.compute(gl, gr)  # int16, fixed-point *16
        if self.use_wls:
            raw_right = self._right_matcher.compute(gr, gl)
            raw = self._wls.filter(raw_left, gl, disparity_map_right=raw_right)
        else:
            raw = raw_left

        disp = raw.astype(np.float32) / 16.0
        invalid = disp <= 0
        disp[invalid] = np.nan
        self.disparity = disp

        # Depth via Q. Feed the true (un-scaled) disparity as float32.
        points3d = cv2.reprojectImageTo3D(
            np.nan_to_num(disp, nan=0.0).astype(np.float32), self.rectifier.Q
        )
        depth = points3d[:, :, 2].astype(np.float32)
        depth[invalid] = np.nan
        # Guard against the divide-by-tiny-disparity blow-ups Q can produce.
        depth[~np.isfinite(depth)] = np.nan
        self.depth_map = depth
        return depth

    def detection_range(self):
        """Global ``(min_mm, max_mm)`` detectable depth for current SGBM params."""
        f = self.rectifier.fx
        b = self.rectifier.baseline
        d_far = self.min_disparity + self.num_disparities  # largest disparity
        d_near = max(self.min_disparity, self.delta_d)     # smallest reliable
        range_min = f * b / d_far
        range_max = f * b / d_near
        return range_min, range_max

    def pixel_info(self, x, y):
        """Depth + uncertainty band + detection range at pixel ``(x, y)``."""
        rmin, rmax = self.detection_range()
        if self.depth_map is None:
            return PixelInfo(valid=False, range_min_mm=rmin, range_max_mm=rmax)
        h, w = self.depth_map.shape
        if not (0 <= x < w and 0 <= y < h):
            return PixelInfo(valid=False, range_min_mm=rmin, range_max_mm=rmax)

        z = float(self.depth_map[y, x])
        if not np.isfinite(z) or z <= 0:
            return PixelInfo(valid=False, range_min_mm=rmin, range_max_mm=rmax)

        # dz = z^2 / (f*B) * delta_d  -- error grows with the square of depth.
        f = self.rectifier.fx
        b = self.rectifier.baseline
        error = (z * z) / (f * b) * self.delta_d
        return PixelInfo(
            valid=True,
            depth_mm=z,
            error_mm=error,
            range_min_mm=rmin,
            range_max_mm=rmax,
        )

    def colorized(self):
        """BGR colormap of the last depth map for display."""
        if self.depth_map is None:
            return None
        rmin, rmax = self.detection_range()
        depth = self.depth_map
        norm = np.clip((depth - rmin) / max(rmax - rmin, 1e-6), 0, 1)
        norm = np.nan_to_num(norm, nan=0.0)
        vis = (norm * 255).astype(np.uint8)
        color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        color[~np.isfinite(depth)] = (0, 0, 0)  # invalid -> black
        return color
