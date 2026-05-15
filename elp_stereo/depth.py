"""Rectification, disparity (StereoSGBM) and depth with an uncertainty model.

Depth values come from ``cv2.reprojectImageTo3D`` using the rectification ``Q``
matrix, which encodes the full rectified geometry. The scalar pinhole relation
``z = f*B/d`` is used only to derive the depth *uncertainty* and the detectable
range, not the depth value itself.

SGBM disparity is 16-subpixel fixed-point; it is divided by 16 before any use,
and non-positive disparities are treated as invalid.
"""

from collections import deque
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
        # User-adjustable depth cap (mm). Floats so GIL-atomic for cross-thread
        # write from the GUI sliders.
        self.min_depth_mm = float(self._cfg.get("min_depth_mm", 50.0))
        self.max_depth_mm = float(self._cfg.get("max_depth_mm", 3000.0))
        # Temporal averaging over N frames (1 = disabled). Noise drops as
        # 1/sqrt(N) for static scenes. Higher = cleaner depth, more latency.
        self.temporal_frames = int(self._cfg.get("temporal_frames", 1))
        self._depth_buffer = deque(maxlen=max(1, self.temporal_frames))
        self._build_matchers()
        # Last computed frame state (set by compute()).
        self.disparity = None      # float32, true disparity in px, NaN = invalid
        self.depth_map = None      # float32 (h, w), Z in mm, NaN = invalid
        # Colormap used by colorized(). Write from GUI thread; int assign is
        # GIL-atomic so the worker reading it on the next frame is safe.
        self.colormap = cv2.COLORMAP_TURBO

    def _build_matchers(self):
        s = self._cfg["sgbm"]
        bs = s["block_size"]
        ch = 1  # disparity computed on grayscale
        p1 = s["p1"] if s["p1"] is not None else 8 * ch * bs * bs
        p2 = s["p2"] if s["p2"] is not None else 32 * ch * bs * bs
        self.min_disparity = s["min_disparity"]
        self.num_disparities = s["num_disparities"]
        # HH mode = full 8-direction aggregation. Higher quality than 3WAY,
        # ~2× slower but worth it for fine-detail close-range work.
        self._left_matcher = cv2.StereoSGBM_create(
            minDisparity=s["min_disparity"],
            numDisparities=s["num_disparities"],
            blockSize=bs,
            P1=p1, P2=p2,
            disp12MaxDiff=s["disp12_max_diff"],
            uniquenessRatio=s["uniqueness_ratio"],
            speckleWindowSize=s["speckle_window_size"],
            speckleRange=s["speckle_range"],
            mode=cv2.STEREO_SGBM_MODE_HH,
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
        """Apply changed SGBM/WLS/depth-cap params (from the GUI) and rebuild."""
        self._cfg["sgbm"].update(params.get("sgbm", {}))
        if "use_wls_filter" in params:
            self._cfg["use_wls_filter"] = params["use_wls_filter"]
        if "min_depth_mm" in params:
            self.min_depth_mm = float(params["min_depth_mm"])
        if "max_depth_mm" in params:
            self.max_depth_mm = float(params["max_depth_mm"])
        if "temporal_frames" in params:
            n = max(1, int(params["temporal_frames"]))
            self.temporal_frames = n
            self._depth_buffer = deque(maxlen=n)
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
        # Apply user-adjustable depth cap — anything outside [min, max] is
        # treated as invalid for both visualization and pixel readout.
        out_of_range = (depth < self.min_depth_mm) | (depth > self.max_depth_mm)
        depth[out_of_range] = np.nan

        # 3×3 median smoothing to kill isolated speckle pixels. NaN-aware:
        # fill NaN with 0 for the filter, then restore original NaN mask so
        # invalid regions don't bleed into valid ones (valid pixels near a
        # NaN edge get a small bias toward 0, acceptable for speckle removal).
        finite_mask = np.isfinite(depth)
        if finite_mask.any():
            filled = np.where(finite_mask, depth, 0.0).astype(np.float32)
            depth = cv2.medianBlur(filled, 3)
            depth[~finite_mask] = np.nan

        # Temporal averaging — noise/sqrt(N) reduction for slow/static scenes.
        if self.temporal_frames > 1:
            self._depth_buffer.append(depth)
            if len(self._depth_buffer) >= 2:
                stack = np.stack(self._depth_buffer, axis=0)
                with np.errstate(invalid="ignore"):
                    depth = np.nanmean(stack, axis=0).astype(np.float32)

        self.depth_map = depth
        return depth

    def detection_range(self):
        """Active depth window ``(min_mm, max_mm)``.

        Intersects the SGBM theoretical range (from disparity settings) with
        the user-set cap. The user cap usually narrows the SGBM range to a
        useful subset for the scene.
        """
        f = self.rectifier.fx
        b = self.rectifier.baseline
        d_far = self.min_disparity + self.num_disparities
        d_near = max(self.min_disparity, self.delta_d)
        sgbm_min = f * b / d_far
        sgbm_max = f * b / d_near
        return (max(sgbm_min, self.min_depth_mm),
                min(sgbm_max, self.max_depth_mm))

    def pixel_info_from_map(self, depth_map, x, y):
        """Depth + uncertainty at ``(x, y)`` using an externally supplied depth map.

        Use this from the GUI thread with a snapshot copy from the worker so
        reads never race with the worker writing ``self.depth_map``.
        """
        rmin, rmax = self.detection_range()
        if depth_map is None:
            return PixelInfo(valid=False, range_min_mm=rmin, range_max_mm=rmax)
        h, w = depth_map.shape
        if not (0 <= x < w and 0 <= y < h):
            return PixelInfo(valid=False, range_min_mm=rmin, range_max_mm=rmax)
        z = float(depth_map[y, x])
        if not np.isfinite(z) or z <= 0:
            return PixelInfo(valid=False, range_min_mm=rmin, range_max_mm=rmax)
        f = self.rectifier.fx
        b = self.rectifier.baseline
        error = (z * z) / (f * b) * self.delta_d
        return PixelInfo(valid=True, depth_mm=z, error_mm=error,
                         range_min_mm=rmin, range_max_mm=rmax)

    def pixel_info(self, x, y):
        """Depth + uncertainty band + detection range at pixel ``(x, y)``."""
        return self.pixel_info_from_map(self.depth_map, x, y)

    def colorized(self):
        """BGR colormap of the last depth map for display.

        Gradient spans the user-adjustable [min_depth_mm, max_depth_mm] range
        so the same color = same physical depth across frames. Pixels outside
        the range were already NaN-masked in compute() and render as black.
        """
        if self.depth_map is None:
            return None
        depth = self.depth_map
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            return np.zeros((*depth.shape, 3), np.uint8)
        lo = self.min_depth_mm
        hi = self.max_depth_mm
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        norm = np.nan_to_num(norm, nan=0.0)
        vis = (norm * 255).astype(np.uint8)
        color = cv2.applyColorMap(vis, self.colormap)
        color[~valid] = (0, 0, 0)  # invalid -> black
        return color
