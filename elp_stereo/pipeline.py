"""High-level API for ELP 3D stereo camera depth processing."""

from .camera import StereoCamera
from .calibration import load_yaml
from .depth import DepthEngine, Rectifier
from .config import load_config


class StereoPipeline:
    """Headless pipeline for stereo capture, rectification, and depth.

    This class encapsulates the camera, rectifier, and depth engine into a
    single object for easy integration without needing the PyQt5 GUI.
    """

    def __init__(self, config_path=None):
        """Initialize the pipeline with an optional config file path."""
        self.cfg = load_config(config_path)
        self.camera = StereoCamera(self.cfg)
        self.rectifier = None
        self.depth_engine = None

    def load_calibration(self, calibration_path):
        """Initialize rectification and depth engine from a calibration file."""
        calib = load_yaml(calibration_path)
        self.rectifier = Rectifier(calib)
        self.depth_engine = DepthEngine(self.cfg, self.rectifier)

    def start(self):
        """Open the camera capture."""
        self.camera.open()

    def stop(self):
        """Release the camera capture."""
        self.camera.release()

    def grab_depth(self):
        """Capture a frame pair and compute depth.

        Returns:
            tuple: (left_rect, right_rect, depth_map) where depth_map is float32
            or None on capture failure. If no calibration is loaded, depth_map
            will be None.
        """
        pair = self.camera.grab()
        if pair is None:
            return None

        left, right = pair
        if self.rectifier is None or self.depth_engine is None:
            return left, right, None

        lr, rr = self.rectifier.rectify(left, right)
        depth = self.depth_engine.compute(lr, rr)
        return lr, rr, depth

    def get_colorized_depth(self):
        """Return a BGR colorized version of the last computed depth map."""
        if self.depth_engine:
            return self.depth_engine.colorized()
        return None
