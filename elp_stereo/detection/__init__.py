"""Object detection on rectified stereo frames + 3D back-projection from depth."""

from .detector import Detection, Detector, build_detector
from .depth_pick import bbox_median_depth, bbox_robust_depth, pixel_to_camera_xyz
from .plane import Plane, fit_tray_plane, plane_z_at
from .preproc import clahe_bgr
from .tiler import TiledDetector, make_tiles
from .tracker import TopPickTracker
from .worker import DetectionWorker

__all__ = [
    "Detection",
    "Detector",
    "build_detector",
    "bbox_median_depth",
    "bbox_robust_depth",
    "pixel_to_camera_xyz",
    "Plane",
    "fit_tray_plane",
    "plane_z_at",
    "clahe_bgr",
    "TiledDetector",
    "make_tiles",
    "TopPickTracker",
    "DetectionWorker",
]
