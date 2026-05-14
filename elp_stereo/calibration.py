"""Stereo calibration: accumulate board detections, run the OpenCV pipeline,
and persist the result.

For ChArUco targets the per-frame corner sets differ between the two cameras,
so only corners with IDs seen in *both* images are kept. For a chessboard every
corner is always present in a fixed order, so the full grid is used.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    image_size: tuple        # (w, h)
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    roi1: tuple
    roi2: tuple
    rms: float


def _match(det_left, det_right):
    """Return ``(objp, imgp_l, imgp_r)`` for corners visible in both views."""
    if det_left.ids is None or det_right.ids is None:
        # Chessboard: fixed full-grid order, every corner present.
        return (det_left.object_points,
                det_left.image_points,
                det_right.image_points)

    # ChArUco: intersect by corner id.
    ids_l = det_left.ids.flatten()
    ids_r = det_right.ids.flatten()
    common = np.intersect1d(ids_l, ids_r)
    if len(common) < 4:
        return None
    idx_l = {cid: i for i, cid in enumerate(ids_l)}
    idx_r = {cid: i for i, cid in enumerate(ids_r)}
    sel_l = [idx_l[c] for c in common]
    sel_r = [idx_r[c] for c in common]
    return (det_left.object_points[sel_l],
            det_left.image_points[sel_l],
            det_right.image_points[sel_r])


class StereoCalibrator:
    """Accumulates matched stereo detections and runs the calibration."""

    def __init__(self, image_size):
        self.image_size = tuple(image_size)  # (w, h)
        self._objpoints = []
        self._imgpoints_l = []
        self._imgpoints_r = []
        self._coverage_pts = []  # left-image points, for the spread heatmap

    @property
    def pair_count(self):
        return len(self._objpoints)

    def add_pair(self, det_left, det_right):
        """Add one stereo frame. Returns matched-corner count, or 0 if rejected."""
        if det_left is None or det_right is None:
            return 0
        matched = _match(det_left, det_right)
        if matched is None:
            return 0
        objp, imgp_l, imgp_r = matched
        self._objpoints.append(objp.astype(np.float32))
        self._imgpoints_l.append(imgp_l.astype(np.float32))
        self._imgpoints_r.append(imgp_r.astype(np.float32))
        self._coverage_pts.append(imgp_l.copy())
        return len(objp)

    def reset(self):
        self._objpoints.clear()
        self._imgpoints_l.clear()
        self._imgpoints_r.clear()
        self._coverage_pts.clear()

    def coverage_heatmap(self):
        """Grayscale heatmap (h, w) of where corners landed in the left image."""
        w, h = self.image_size
        heat = np.zeros((h, w), np.float32)
        for pts in self._coverage_pts:
            for x, y in pts:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < w and 0 <= yi < h:
                    heat[yi, xi] += 1.0
        if heat.max() > 0:
            heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=15)
            heat /= heat.max()
        return (heat * 255).astype(np.uint8)

    def calibrate(self):
        """Run mono + stereo calibration and rectification. Returns a result."""
        if self.pair_count < 5:
            raise RuntimeError(
                f"Need at least 5 stereo pairs, have {self.pair_count}."
            )

        rms_l, K1, D1, _, _ = cv2.calibrateCamera(
            self._objpoints, self._imgpoints_l, self.image_size, None, None
        )
        rms_r, K2, D2, _, _ = cv2.calibrateCamera(
            self._objpoints, self._imgpoints_r, self.image_size, None, None
        )

        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            self._objpoints, self._imgpoints_l, self._imgpoints_r,
            K1, D1, K2, D2, self.image_size,
            flags=flags, criteria=criteria,
        )

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, self.image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )

        return CalibrationResult(
            image_size=self.image_size,
            K1=K1, D1=D1, K2=K2, D2=D2,
            R=R, T=T, E=E, F=F,
            R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
            roi1=tuple(roi1), roi2=tuple(roi2),
            rms=float(rms),
        )


def save_yaml(path, result):
    """Persist a ``CalibrationResult`` to an OpenCV YAML file."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    w, h = result.image_size
    fs.write("image_width", w)
    fs.write("image_height", h)
    fs.write("K1", result.K1)
    fs.write("D1", result.D1)
    fs.write("K2", result.K2)
    fs.write("D2", result.D2)
    fs.write("R", result.R)
    fs.write("T", result.T)
    fs.write("E", result.E)
    fs.write("F", result.F)
    fs.write("R1", result.R1)
    fs.write("R2", result.R2)
    fs.write("P1", result.P1)
    fs.write("P2", result.P2)
    fs.write("Q", result.Q)
    fs.write("roi1", np.array(result.roi1, np.int32))
    fs.write("roi2", np.array(result.roi2, np.int32))
    fs.write("rms", result.rms)
    fs.release()


def load_yaml(path):
    """Load a ``CalibrationResult`` from an OpenCV YAML file."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration file: {path}")

    def mat(name):
        return fs.getNode(name).mat()

    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    roi1 = tuple(mat("roi1").flatten().astype(int))
    roi2 = tuple(mat("roi2").flatten().astype(int))
    rms = fs.getNode("rms").real()
    result = CalibrationResult(
        image_size=(w, h),
        K1=mat("K1"), D1=mat("D1"), K2=mat("K2"), D2=mat("D2"),
        R=mat("R"), T=mat("T"), E=mat("E"), F=mat("F"),
        R1=mat("R1"), R2=mat("R2"), P1=mat("P1"), P2=mat("P2"), Q=mat("Q"),
        roi1=roi1, roi2=roi2, rms=rms,
    )
    fs.release()
    return result
