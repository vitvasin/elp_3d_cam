"""Calibration target abstraction: chessboard, ChArUco, and circle-grid boards.

Each target detects corners in a grayscale image and returns a ``Detection``
holding matched 3D object points and 2D image points. ``ids`` is ``None`` for a
chessboard (corner order is fixed) and the charuco corner IDs otherwise, which
lets the stereo calibrator intersect corners visible in both cameras.

The ChArUco code supports both OpenCV aruco APIs:
  * legacy (<= 4.6, the system package on Ubuntu 22.04 / 24.04):
    ``Dictionary_get`` / ``CharucoBoard_create`` / ``interpolateCornersCharuco``
  * modern (>= 4.7, e.g. pip ``opencv-contrib-python``):
    ``getPredefinedDictionary`` / ``CharucoBoard`` / ``CharucoDetector``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# OpenCV 4.7 introduced the CharucoDetector class and removed the *_create /
# interpolateCornersCharuco free functions.
_ARUCO_MODERN = hasattr(cv2.aruco, "CharucoDetector")


@dataclass
class Detection:
    object_points: np.ndarray            # (N, 3) float32, in mm
    image_points: np.ndarray             # (N, 2) float32
    ids: Optional[np.ndarray]            # (N, 1) int32 charuco corner ids, or None

    @property
    def count(self):
        return len(self.object_points)


class CalibrationTarget:
    """Base interface for calibration targets."""

    def detect(self, gray):
        """Return a ``Detection`` or ``None`` if the board was not found."""
        raise NotImplementedError

    def draw(self, bgr, detection):
        """Draw the detected board onto a BGR image (in place)."""
        raise NotImplementedError


class ChessboardTarget(CalibrationTarget):
    """Standard OpenCV checkerboard. ``cols``/``rows`` are inner-corner counts."""

    def __init__(self, cols, rows, square_size_mm):
        self.cols = cols
        self.rows = rows
        self.square_size_mm = square_size_mm
        # Fixed object-point grid, scaled to physical mm.
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= square_size_mm
        self._objp = objp

    def detect(self, gray):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        found, corners = cv2.findChessboardCornersSB(
            gray, (self.cols, self.rows), flags=flags
        )
        if not found:
            found, corners = cv2.findChessboardCorners(
                gray, (self.cols, self.rows),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if not found:
                return None
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA
            )
        return Detection(
            object_points=self._objp.copy(),
            image_points=corners.reshape(-1, 2).astype(np.float32),
            ids=None,
        )

    def draw(self, bgr, detection):
        cv2.drawChessboardCorners(
            bgr, (self.cols, self.rows),
            detection.image_points.reshape(-1, 1, 2), True,
        )


class CharucoTarget(CalibrationTarget):
    """ChArUco board (chessboard + ArUco markers), robust to partial views."""

    def __init__(self, squares_x, squares_y, square_len_mm, marker_len_mm,
                 dictionary="DICT_4X4_50"):
        self.squares_x = squares_x
        self.squares_y = squares_y
        dict_id = getattr(cv2.aruco, dictionary)

        if _ARUCO_MODERN:
            self._dict = cv2.aruco.getPredefinedDictionary(dict_id)
            self._board = cv2.aruco.CharucoBoard(
                (squares_x, squares_y), square_len_mm, marker_len_mm, self._dict
            )
            self._detector = cv2.aruco.CharucoDetector(self._board)
            self._all_objp = self._board.getChessboardCorners().astype(np.float32)
        else:
            self._dict = cv2.aruco.Dictionary_get(dict_id)
            self._board = cv2.aruco.CharucoBoard_create(
                squares_x, squares_y, square_len_mm, marker_len_mm, self._dict
            )
            self._detector = None
            self._all_objp = self._board.chessboardCorners.astype(np.float32)

    def detect(self, gray):
        if _ARUCO_MODERN:
            ch_corners, ch_ids, _, _ = self._detector.detectBoard(gray)
            if ch_ids is None or len(ch_ids) < 4:
                return None
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dict)
            if ids is None or len(ids) == 0:
                return None
            retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self._board
            )
            if retval is None or retval < 4:
                return None

        flat_ids = ch_ids.flatten()
        return Detection(
            object_points=self._all_objp[flat_ids].copy(),
            image_points=ch_corners.reshape(-1, 2).astype(np.float32),
            ids=ch_ids.reshape(-1, 1).astype(np.int32),
        )

    def draw(self, bgr, detection):
        cv2.aruco.drawDetectedCornersCharuco(
            bgr, detection.image_points.reshape(-1, 1, 2), detection.ids
        )


class CircleGridTarget(CalibrationTarget):
    """Symmetric or asymmetric circle grid (``cv2.findCirclesGrid``).

    For asymmetric grids, ``cols`` is the number of circles per row and
    ``rows`` is the number of rows; ``spacing_mm`` is the distance between
    a circle and its neighbour in the same row of an adjacent row (OpenCV
    convention — see ``cv2.findCirclesGrid`` docs).
    """

    def __init__(self, cols, rows, spacing_mm, asymmetric=True):
        self.cols = int(cols)
        self.rows = int(rows)
        self.spacing_mm = float(spacing_mm)
        self.asymmetric = bool(asymmetric)
        self._flag = (cv2.CALIB_CB_ASYMMETRIC_GRID if self.asymmetric
                      else cv2.CALIB_CB_SYMMETRIC_GRID)
        self._objp = self._build_object_points()

    def _build_object_points(self):
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        s = self.spacing_mm
        if self.asymmetric:
            for i in range(self.rows):
                for j in range(self.cols):
                    objp[i * self.cols + j] = ((2 * j + i % 2) * s, i * s, 0.0)
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    objp[i * self.cols + j] = (j * s, i * s, 0.0)
        return objp

    def detect(self, gray):
        found, centers = cv2.findCirclesGrid(
            gray, (self.cols, self.rows), flags=self._flag
        )
        if not found or centers is None:
            return None
        return Detection(
            object_points=self._objp.copy(),
            image_points=centers.reshape(-1, 2).astype(np.float32),
            ids=None,
        )

    def draw(self, bgr, detection):
        cv2.drawChessboardCorners(
            bgr, (self.cols, self.rows),
            detection.image_points.reshape(-1, 1, 2), True,
        )


def build_target(kind, cfg):
    """Construct a target from the ``calibration`` config block.

    ``kind`` is ``"chessboard"``, ``"charuco"``, or ``"circle_grid"``.
    """
    if kind == "chessboard":
        c = cfg["chessboard"]
        return ChessboardTarget(c["cols"], c["rows"], c["square_size_mm"])
    if kind == "charuco":
        c = cfg["charuco"]
        return CharucoTarget(
            c["squares_x"], c["squares_y"], c["square_len_mm"],
            c["marker_len_mm"], c["dictionary"],
        )
    if kind == "circle_grid":
        c = cfg["circle_grid"]
        return CircleGridTarget(
            c["cols"], c["rows"], c["spacing_mm"],
            asymmetric=c.get("asymmetric", True),
        )
    raise ValueError(f"Unknown target kind: {kind}")
