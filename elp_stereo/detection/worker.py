"""Background thread: rectify -> depth -> (CLAHE) -> tile/infer -> top-1 ->
robust depth + plane fallback -> tracker -> emit result.
"""

import math
import threading

from PyQt5.QtCore import QThread, pyqtSignal

from .depth_pick import bbox_trimmed_depth, pixel_to_camera_xyz
from .plane import plane_z_at
from .preproc import clahe_bgr
from .tiler import TiledDetector


class DetectionWorker(QThread):
    """Latest-only worker. Emits a single dict per processed frame.

    Result dict keys:
        ``left``        – rectified left BGR ndarray (no CLAHE)
        ``left_infer``  – BGR sent to detector (post-CLAHE if enabled)
        ``depth_color`` – colorized depth BGR or None
        ``depth_map``   – float32 mm, NaN = invalid (snapshot copy)
        ``detections``  – list of items (all dets, full-frame coords)
        ``top``         – smoothed top-1 item or None
        ``error``       – str (failure only)

    Each item: ``{"det": Detection, "uv": (u,v), "xyz_mm": (X,Y,Z),
                  "valid_pixels": int, "depth_source": "stereo"|"plane"}``.
    """

    result_ready = pyqtSignal(object)

    def __init__(self, rectifier, depth_engine, detector,
                 min_valid_pixels=5, bbox_shrink=0.6,
                 fallback_to_plane=True, parent=None):
        super().__init__(parent)
        self._rectifier = rectifier
        self._engine = depth_engine
        self._detector = detector
        self._min_valid = int(min_valid_pixels)
        self._shrink = float(bbox_shrink)
        self._fallback_to_plane = bool(fallback_to_plane)
        self._depth_trim = 0.2
        self._P1 = None
        self._plane = None
        self._tracker = None
        self._clahe_on = False
        self._lock = threading.Lock()
        self._pending = None
        self._stop = False

    # --------------------------------------------------------------- setters
    def set_projection(self, P1):
        self._P1 = P1

    def set_min_valid_pixels(self, n):
        self._min_valid = max(1, int(n))

    def set_bbox_shrink(self, s):
        self._shrink = float(s)

    def set_depth_trim(self, trim):
        self._depth_trim = max(0.0, min(0.45, float(trim)))

    def set_detector(self, detector):
        self._detector = detector

    def set_plane(self, plane):
        self._plane = plane

    def get_plane(self):
        return self._plane

    def set_tracker(self, tracker):
        self._tracker = tracker

    def reset_tracker(self):
        if self._tracker is not None:
            self._tracker.reset()

    def set_clahe(self, on):
        self._clahe_on = bool(on)

    def set_fallback_to_plane(self, on):
        self._fallback_to_plane = bool(on)

    def set_tile_params(self, enabled, grid=(2, 2), overlap=0.2):
        """Enable/disable tiled inference around the current base detector."""
        if self._detector is None:
            return
        if enabled:
            if isinstance(self._detector, TiledDetector):
                self._detector._grid = (int(grid[0]), int(grid[1]))  # noqa: SLF001
                self._detector._overlap = float(overlap)              # noqa: SLF001
            else:
                self._detector = TiledDetector(
                    self._detector, grid=grid, overlap=overlap,
                )
        else:
            if isinstance(self._detector, TiledDetector):
                self._detector = self._detector.base

    # -------------------------------------------------------------- ingress
    def submit(self, left, right):
        with self._lock:
            self._pending = (left, right)

    # ---------------------------------------------------------------- loop
    def run(self):
        self._stop = False
        while not self._stop:
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                self.msleep(5)
                continue

            left, right = frame
            try:
                lr, rr = self._rectifier.rectify(left, right)
                depth_map = self._engine.compute(lr, rr)
                depth_color = self._engine.colorized()

                left_infer = clahe_bgr(lr) if self._clahe_on else lr
                detections = []
                if self._detector is not None and self._P1 is not None:
                    raw = self._detector.infer(left_infer)
                    detections = self._build_items(raw, depth_map)

                top = None
                if detections:
                    top = max(detections, key=lambda it: it["det"].score)
                if self._tracker is not None:
                    top = self._tracker.update(top)

            except Exception as exc:  # noqa: BLE001
                self.result_ready.emit({"error": str(exc)})
                continue

            dm_copy = depth_map.copy() if depth_map is not None else None
            self.result_ready.emit({
                "left": lr,
                "left_infer": left_infer,
                "depth_color": depth_color,
                "depth_map": dm_copy,
                "detections": detections,
                "top": top,
            })

    # ----------------------------------------------------------- pipeline
    def _build_items(self, raw_detections, depth_map):
        items = []
        for det in raw_detections:
            x1, y1, x2, y2 = det.bbox
            u = (x1 + x2) // 2
            v = (y1 + y2) // 2
            z, n, std = bbox_trimmed_depth(
                depth_map, det.bbox, self._shrink, self._depth_trim
            )
            source = "stereo"
            valid = n
            if (math.isnan(z) or n < self._min_valid):
                if self._fallback_to_plane and self._plane is not None:
                    z = plane_z_at(self._plane, u, v, self._P1)
                    source = "plane"
                    valid = 0
            if not (z == z) or z <= 0:  # NaN or non-positive
                continue
            xyz = pixel_to_camera_xyz(u, v, z, self._P1)
            items.append({
                "det": det,
                "uv": (int(u), int(v)),
                "xyz_mm": xyz,
                "valid_pixels": int(valid),
                "depth_std_mm": float(std) if std == std else float("nan"),
                "depth_source": source,
            })
        return items

    def stop(self):
        self._stop = True
        with self._lock:
            self._pending = None
        self.wait(2000)
