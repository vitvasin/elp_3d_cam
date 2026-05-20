#!/usr/bin/env python3
"""Entry point for the ELP 3D object-detection + ROS2 publisher app.

Loads ``config/default.yaml`` (camera + depth) and deep-merges
``config/detect.yaml`` (detection + ros2) on top. Reuses the calibration file
produced by ``app.py``.
"""

import sys
from pathlib import Path

import yaml
from PyQt5.QtWidgets import QApplication

from elp_stereo.config import load_config, project_root
from elp_stereo.gui.detect_main_window import DetectMainWindow
from elp_stereo.hand_eye import load_hand_eye_yaml, static_tf_config


def _deep_merge(base, overlay):
    """Recursive dict merge: overlay values win, nested dicts merged in place."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_detect_config():
    cfg = load_config()  # camera + depth + calibration blocks
    overlay_path = project_root() / "config" / "detect.yaml"
    if overlay_path.is_file():
        with open(overlay_path, "r") as f:
            overlay = yaml.safe_load(f) or {}
        _deep_merge(cfg, overlay)
    hand_eye_path = project_root() / cfg.get("hand_eye_path", "config/hand_eye.yaml")
    if hand_eye_path.is_file():
        hand_eye, _ = load_hand_eye_yaml(hand_eye_path)
        ros_cfg = cfg.setdefault("ros2", {})
        ros_cfg["camera_frame"] = hand_eye.get("child_frame", ros_cfg.get("camera_frame", "camera_optical_frame"))
        ros_cfg["robot_frame"] = hand_eye.get("parent_frame", ros_cfg.get("robot_frame", "robot_base"))
        ros_cfg["use_tf_lookup"] = True
        ros_cfg["static_tf"] = static_tf_config(hand_eye)
    return cfg


def main():
    cfg = load_detect_config()
    app = QApplication(sys.argv)
    win = DetectMainWindow(cfg)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
