#!/usr/bin/env python3
"""MG400 pick controller for the ELP detection pipeline.

Consumes ``vision_msgs/Detection3DArray`` from ``app_detect.py`` and executes a
simple pick/place sequence with the MG400 ROS2 action/services.
"""

import sys
import threading
import time
import subprocess
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from elp_stereo.calibration import load_yaml
from elp_stereo.camera import CaptureThread, StereoCamera
from elp_stereo.config import load_config, project_root
from elp_stereo.depth import DepthEngine, Rectifier, build_depth_engine
from elp_stereo.detection.depth_pick import pixel_to_camera_xyz
from elp_stereo.gui.widgets import ImagePanel
from elp_stereo.hand_eye import load_hand_eye_yaml
from elp_stereo.worker import DepthWorker

try:
    import rclpy
    from vision_msgs.msg import Detection3DArray
    from elp_stereo.ros2.mg400 import MG400Node, RosSpinThread
    ROS_OK = True
except Exception as exc:  # noqa: BLE001
    rclpy = None
    Detection3DArray = None
    MG400Node = object
    RosSpinThread = None
    ROS_OK = False
    ROS_IMPORT_ERROR = str(exc)


def detection_label(det):
    if not det.results:
        return "object", 0.0
    hyp = det.results[0]
    if hasattr(hyp, "hypothesis"):
        return hyp.hypothesis.class_id, float(hyp.hypothesis.score)
    return getattr(hyp, "id", "object"), float(getattr(hyp, "score", 0.0))


PELLET_PROJECT_ROOT = Path("/home/admin01/workspace/pelletprototype")
PELLET_VISION_PATH = PELLET_PROJECT_ROOT / "pellet_vision.py"
PELLET_BOX_MODEL = PELLET_PROJECT_ROOT / "model_Onnx" / "box_best.onnx"
PELLET_SEG_MODEL = PELLET_PROJECT_ROOT / "model_Onnx" / "seg_best.onnx"
ROBOT_MIN_Z_M = -0.068


def load_pellet_detector_class():
    """Load PelletDetector from the prototype project without changing cwd."""
    if not PELLET_VISION_PATH.is_file():
        raise RuntimeError(f"pellet_vision.py not found: {PELLET_VISION_PATH}")
    spec = importlib.util.spec_from_file_location("pelletprototype_pellet_vision", PELLET_VISION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pellet detector from {PELLET_VISION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PelletDetector


class PickNode(MG400Node):
    """MG400 facade plus a subscriber for ELP detections."""

    def __init__(self, on_detection):
        super().__init__("elp_mg400_pick_controller")
        self._on_detection = on_detection
        self.create_subscription(Detection3DArray, "/elp/detections", self._detections_cb, 10)

    def _detections_cb(self, msg):
        items = []
        for det in msg.detections:
            label, score = detection_label(det)
            p = det.bbox.center.position
            items.append({
                "label": label,
                "score": score,
                "xyz_m": (float(p.x), float(p.y), float(p.z)),
                "frame_id": msg.header.frame_id,
            })
        self._on_detection(items)


class RobotControlWindow(QMainWindow):
    detections_received = pyqtSignal(object)
    pellet_result_received = pyqtSignal(object)
    pellet_loop_status_received = pyqtSignal(str)
    log_received = pyqtSignal(str)
    pose_received = pyqtSignal(object, object)
    place_points_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ELP Detection -> MG400 Pick Control")
        self.resize(720, 680)
        self.node = None
        self.spin = None
        self.busy = False
        self.detections = []
        self.hand_eye_T = None
        self.robot_frame = "robot_base"
        self.camera_frame = "camera_optical_frame"
        self.place_points = []
        self.place_index = 0
        self.bringup_proc = None
        self.config_path = project_root() / "config" / "robot_control.yaml"
        self.home_pose = None
        self.camera_cfg = load_config()
        self.camera = None
        self.capture = None
        self.depth_worker = None
        self.cam_calib = None
        self.cam_rectifier = None
        self.cam_depth_engine = None
        self.latest_depth = None
        self.latest_left_rect = None
        self.clicked_robot = None
        self.pellet_detector = None
        self.pellet_loaded_paths = None
        self.pellet_detections = []
        self.pellet_selected = None
        self.pellet_busy = False
        self.pick_roi = None
        self.roi_selecting = False
        self.roi_first_point = None
        self.pellet_loop_running = False
        self.pellet_loop_stop = threading.Event()
        self.pellet_loop_thread = None
        self.bringup_ready_checks_remaining = 0

        self.list = QListWidget()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.robot_ip = QLineEdit("192.168.1.6")
        self.bringup_status = QLabel("Bringup: not started")
        self.bringup_status.setWordWrap(True)

        self.approach_z = self._spin(ROBOT_MIN_Z_M, 1.0, 0.150, 0.005)
        self.pick_z_offset = self._spin(-0.100, 0.100, 0.000, 0.001)
        self.place_x = self._spin(-1.0, 1.0, 0.300, 0.005)
        self.place_y = self._spin(-1.0, 1.0, 0.000, 0.005)
        self.place_z = self._spin(ROBOT_MIN_Z_M, 1.0, 0.150, 0.005)
        self.r_deg = self._spin(-180.0, 180.0, 0.0, 1.0)
        self.do_index = QSpinBox()
        self.do_index.setRange(1, 16)
        self.do_index.setValue(1)
        self.do_type = QComboBox()
        self.do_type.addItems(["Base DO", "Tool DO"])
        self.close_high = QCheckBox("HIGH closes gripper")
        self.close_high.setChecked(True)
        self.auto_pick = QCheckBox("Auto pick first detection")
        self.auto_loop = QCheckBox("Auto loop first detection to place list")

        self.manual_x = self._spin(-1.0, 1.0, 0.300, 0.005)
        self.manual_y = self._spin(-1.0, 1.0, 0.000, 0.005)
        self.manual_z = self._spin(ROBOT_MIN_Z_M, 1.0, 0.150, 0.005)
        self.manual_r = self._spin(-180.0, 180.0, 0.0, 1.0)
        self.jog_step = self._spin(0.001, 0.100, 0.010, 0.001)
        self.auto_place_x = self._spin(-1.0, 1.0, 0.300, 0.005)
        self.auto_place_y = self._spin(-1.0, 1.0, 0.000, 0.005)
        self.auto_place_z = self._spin(ROBOT_MIN_Z_M, 1.0, 0.150, 0.005)
        self.place_list = QListWidget()
        self.place_list.currentRowChanged.connect(self.on_place_selected)
        self.place_index_label = QLabel("Next place: 0")
        self.home_label = QLabel("Home: not set")
        self.click_x_offset = self._spin(-300.0, 300.0, 0.0, 1.0, decimals=1)
        self.click_y_offset = self._spin(-300.0, 300.0, 0.0, 1.0, decimals=1)
        self.click_z_offset = self._spin(-300.0, 300.0, 0.0, 1.0, decimals=1)
        self.click_r = self._spin(-180.0, 180.0, 40.0, 1.0, decimals=1)
        self.click_approach_z = self._spin(0.0, 500.0, 100.0, 1.0, decimals=1)
        self.click_near_approach_z = self._spin(0.0, 500.0, 30.0, 1.0, decimals=1)
        self.click_speed = QSpinBox()
        self.click_speed.setRange(1, 100)
        self.click_speed.setValue(50)
        self.click_near_speed = QSpinBox()
        self.click_near_speed.setRange(1, 100)
        self.click_near_speed.setValue(20)
        self.click_accel = QSpinBox()
        self.click_accel.setRange(1, 100)
        self.click_accel.setValue(50)
        self.click_near_accel = QSpinBox()
        self.click_near_accel.setRange(1, 100)
        self.click_near_accel.setValue(20)
        self.click_move_type = QComboBox()
        self.click_move_type.addItems(["MovJ", "MovL"])
        self.click_x_offset.valueChanged.connect(self.update_clicked_robot_offset)
        self.click_y_offset.valueChanged.connect(self.update_clicked_robot_offset)
        self.click_z_offset.valueChanged.connect(self.update_clicked_robot_offset)
        self.click_label = QLabel("Clicked target: -")
        self.pellet_box_model = QLineEdit(str(PELLET_BOX_MODEL))
        self.pellet_seg_model = QLineEdit(str(PELLET_SEG_MODEL))
        self.pellet_box_conf = self._spin(0.01, 1.0, 0.50, 0.05, decimals=2)
        self.pellet_seg_conf = self._spin(0.01, 1.0, 0.50, 0.05, decimals=2)
        self.pellet_seg_iou = self._spin(0.01, 1.0, 0.80, 0.05, decimals=2)
        self.pellet_r_offset = self._spin(-180.0, 180.0, 40.0, 1.0, decimals=1)
        self.pellet_device = QComboBox()
        self.pellet_device.addItems(["Auto", "GPU 0", "CPU"])
        self.pellet_use_angle = QCheckBox("Use pellet orientation for R yaw")
        self.pellet_use_angle.setChecked(True)
        self.pellet_show_confidence = QCheckBox("Show confidence labels")
        self.pellet_show_confidence.setChecked(False)
        self.pick_roi_enabled = QCheckBox("Use pickup ROI")
        self.pick_roi_crop = QCheckBox("Crop detection to ROI")
        self.pick_roi_crop.setChecked(True)
        self.pick_roi_label = QLabel("Pickup ROI: full image")
        self.pick_roi_label.setWordWrap(True)
        self.pellet_status = QLabel("Pellet detector: not loaded")
        self.pellet_status.setWordWrap(True)
        self.pellet_table = QTableWidget(0, 5)
        self.pellet_table.setHorizontalHeaderLabels(["#", "score", "color", "angle", "uv"])
        self.pellet_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.pellet_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.pellet_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pellet_table.verticalHeader().setVisible(False)
        self.pellet_table.setMinimumHeight(120)
        self.pellet_table.itemSelectionChanged.connect(self.on_pellet_table_selection_changed)
        self.pellet_loop_delay = self._spin(0.0, 30.0, 1.0, 0.5, decimals=1)
        self.pellet_home_settle = self._spin(0.0, 10.0, 0.5, 0.1, decimals=1)
        self.pellet_loop_max_cycles = QSpinBox()
        self.pellet_loop_max_cycles.setRange(0, 9999)
        self.pellet_loop_max_cycles.setValue(0)
        self.pellet_loop_status = QLabel("Pellet loop: stopped")
        self.pellet_loop_status.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Approach Z (m)", self.approach_z)
        form.addRow("Pick Z offset (m)", self.pick_z_offset)
        form.addRow("Place X (m)", self.place_x)
        form.addRow("Place Y (m)", self.place_y)
        form.addRow("Place Z (m)", self.place_z)
        form.addRow("R yaw (deg)", self.r_deg)
        form.addRow("DO type", self.do_type)
        form.addRow("DO index", self.do_index)
        form.addRow("", self.close_high)

        btn_pick = QPushButton("Pick Selected")
        btn_reload = QPushButton("Reload Hand-Eye")
        btn_pick.clicked.connect(self.pick_selected)
        btn_reload.clicked.connect(self.load_hand_eye)

        buttons = QHBoxLayout()
        buttons.addWidget(btn_pick)
        buttons.addWidget(btn_reload)

        pick_tab = QWidget()
        pick_layout = QVBoxLayout(pick_tab)
        pick_layout.addWidget(QLabel("Detections from /elp/detections"))
        pick_layout.addWidget(self.list, 1)
        pick_layout.addLayout(form)
        pick_layout.addWidget(self.auto_pick)
        pick_layout.addLayout(buttons)

        manual_tab = self.scrollable(self.build_manual_tab())
        camera_tab = self.build_camera_pick_tab()
        auto_tab = self.scrollable(self.build_auto_tab())

        tabs = QTabWidget()
        tabs.addTab(pick_tab, "Pick")
        tabs.addTab(camera_tab, "Camera Pick")
        tabs.addTab(manual_tab, "Manual")
        tabs.addTab(auto_tab, "Auto Loop")

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addWidget(self.build_robot_state_box())
        layout.addWidget(tabs, 3)
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log, 1)
        self.setCentralWidget(body)
        self.setStatusBar(QStatusBar())

        self.detections_received.connect(self.on_detections)
        self.pellet_result_received.connect(self.on_pellet_result)
        self.pellet_loop_status_received.connect(self.pellet_loop_status.setText)
        self.log_received.connect(self.append_log)
        self.pose_received.connect(self.on_pose_received)
        self.place_points_changed.connect(self.refresh_place_points)
        self.load_robot_config()
        self.load_hand_eye()
        self.start_ros()

    def scrollable(self, widget):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    def _spin(self, lo, hi, value, step, decimals=4):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(value)
        return s

    def robot_config_data(self):
        data = {
            "robot_ip": self.robot_ip.text().strip(),
            "home_pose": self.home_pose,
            "pick": {
                "approach_z_m": self.approach_z.value(),
                "pick_z_offset_m": self.pick_z_offset.value(),
                "place_m": [self.place_x.value(), self.place_y.value(), self.place_z.value()],
                "r_deg": self.r_deg.value(),
            },
            "manual": {
                "jog_step_m": self.jog_step.value(),
            },
            "camera_pick": {
                "x_offset_mm": self.click_x_offset.value(),
                "y_offset_mm": self.click_y_offset.value(),
                "z_offset_mm": self.click_z_offset.value(),
                "r_deg": self.click_r.value(),
                "approach_z_mm": self.click_approach_z.value(),
                "near_approach_z_mm": self.click_near_approach_z.value(),
                "speed": self.click_speed.value(),
                "near_speed": self.click_near_speed.value(),
                "accel": self.click_accel.value(),
                "near_accel": self.click_near_accel.value(),
                "move_type": self.click_move_type.currentText(),
            },
            "pellet_detection": {
                "box_model": self.pellet_box_model.text().strip(),
                "seg_model": self.pellet_seg_model.text().strip(),
                "box_conf": self.pellet_box_conf.value(),
                "seg_conf": self.pellet_seg_conf.value(),
                "seg_iou": self.pellet_seg_iou.value(),
                "use_angle": self.pellet_use_angle.isChecked(),
                "show_confidence": self.pellet_show_confidence.isChecked(),
                "r_offset_deg": self.pellet_r_offset.value(),
                "device": self.pellet_device.currentText(),
                "roi_enabled": self.pick_roi_enabled.isChecked(),
                "roi_crop": self.pick_roi_crop.isChecked(),
                "roi": list(self.pick_roi) if self.pick_roi is not None else None,
            },
            "pellet_loop": {
                "delay_s": self.pellet_loop_delay.value(),
                "home_settle_s": self.pellet_home_settle.value(),
                "max_cycles": self.pellet_loop_max_cycles.value(),
            },
            "gripper": {
                "do_type": self.do_type.currentText(),
                "do_index": self.do_index.value(),
                "high_closes": self.close_high.isChecked(),
            },
            "auto_place_points_m": [list(p) for p in self.place_points],
        }
        return data

    def save_robot_config(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.robot_config_data(), f, sort_keys=False)
        self.append_log(f"Saved {self.config_path}")

    def load_robot_config(self):
        if not self.config_path.is_file():
            self.update_home_label()
            return
        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"Failed to load {self.config_path}: {exc}")
            return

        if data.get("robot_ip"):
            self.robot_ip.setText(str(data["robot_ip"]))

        home = data.get("home_pose")
        if isinstance(home, dict):
            self.home_pose = {
                "x": float(home.get("x", 0.3)),
                "y": float(home.get("y", 0.0)),
                "z": float(home.get("z", 0.15)),
                "r": float(home.get("r", 0.0)),
            }

        pick = data.get("pick", {})
        if isinstance(pick, dict):
            self.approach_z.setValue(float(pick.get("approach_z_m", self.approach_z.value())))
            self.pick_z_offset.setValue(float(pick.get("pick_z_offset_m", self.pick_z_offset.value())))
            place = pick.get("place_m")
            if isinstance(place, (list, tuple)) and len(place) >= 3:
                self.place_x.setValue(float(place[0]))
                self.place_y.setValue(float(place[1]))
                self.place_z.setValue(float(place[2]))
            self.r_deg.setValue(float(pick.get("r_deg", self.r_deg.value())))

        manual = data.get("manual", {})
        if isinstance(manual, dict):
            self.jog_step.setValue(float(manual.get("jog_step_m", self.jog_step.value())))

        camera_pick = data.get("camera_pick", {})
        if isinstance(camera_pick, dict):
            self.click_x_offset.setValue(float(camera_pick.get("x_offset_mm", self.click_x_offset.value())))
            self.click_y_offset.setValue(float(camera_pick.get("y_offset_mm", self.click_y_offset.value())))
            if "z_offset_mm" in camera_pick:
                self.click_z_offset.setValue(float(camera_pick["z_offset_mm"]))
            elif "z_offset_m" in camera_pick:
                self.click_z_offset.setValue(float(camera_pick["z_offset_m"]) * 1000.0)
            self.click_r.setValue(float(camera_pick.get("r_deg", self.click_r.value())))
            self.click_approach_z.setValue(
                float(camera_pick.get("approach_z_mm", self.click_approach_z.value()))
            )
            self.click_near_approach_z.setValue(
                float(camera_pick.get("near_approach_z_mm", self.click_near_approach_z.value()))
            )
            self.click_speed.setValue(int(camera_pick.get("speed", self.click_speed.value())))
            self.click_near_speed.setValue(
                int(camera_pick.get("near_speed", self.click_near_speed.value()))
            )
            self.click_accel.setValue(int(camera_pick.get("accel", self.click_accel.value())))
            self.click_near_accel.setValue(
                int(camera_pick.get("near_accel", self.click_near_accel.value()))
            )
            move_type = str(camera_pick.get("move_type", self.click_move_type.currentText()))
            mt_idx = self.click_move_type.findText(move_type)
            if mt_idx >= 0:
                self.click_move_type.setCurrentIndex(mt_idx)

        pellet = data.get("pellet_detection", {})
        if isinstance(pellet, dict):
            self.pellet_box_model.setText(str(pellet.get("box_model", self.pellet_box_model.text())))
            self.pellet_seg_model.setText(str(pellet.get("seg_model", self.pellet_seg_model.text())))
            self.pellet_box_conf.setValue(float(pellet.get("box_conf", self.pellet_box_conf.value())))
            self.pellet_seg_conf.setValue(float(pellet.get("seg_conf", self.pellet_seg_conf.value())))
            self.pellet_seg_iou.setValue(float(pellet.get("seg_iou", self.pellet_seg_iou.value())))
            self.pellet_use_angle.setChecked(bool(pellet.get("use_angle", self.pellet_use_angle.isChecked())))
            self.pellet_show_confidence.setChecked(bool(pellet.get("show_confidence", self.pellet_show_confidence.isChecked())))
            self.pellet_r_offset.setValue(float(pellet.get("r_offset_deg", self.pellet_r_offset.value())))
            device = str(pellet.get("device", self.pellet_device.currentText()))
            device_i = self.pellet_device.findText(device)
            if device_i >= 0:
                self.pellet_device.setCurrentIndex(device_i)
            self.pick_roi_enabled.setChecked(bool(pellet.get("roi_enabled", self.pick_roi_enabled.isChecked())))
            self.pick_roi_crop.setChecked(bool(pellet.get("roi_crop", self.pick_roi_crop.isChecked())))
            roi = pellet.get("roi")
            if isinstance(roi, (list, tuple)) and len(roi) >= 4:
                self.pick_roi = tuple(int(v) for v in roi[:4])
                self.update_pick_roi_label()

        pellet_loop = data.get("pellet_loop", {})
        if isinstance(pellet_loop, dict):
            self.pellet_loop_delay.setValue(float(pellet_loop.get("delay_s", self.pellet_loop_delay.value())))
            self.pellet_home_settle.setValue(float(pellet_loop.get("home_settle_s", self.pellet_home_settle.value())))
            self.pellet_loop_max_cycles.setValue(int(pellet_loop.get("max_cycles", self.pellet_loop_max_cycles.value())))

        gripper = data.get("gripper", {})
        if isinstance(gripper, dict):
            do_type = str(gripper.get("do_type", self.do_type.currentText()))
            idx = self.do_type.findText(do_type)
            if idx >= 0:
                self.do_type.setCurrentIndex(idx)
            self.do_index.setValue(int(gripper.get("do_index", self.do_index.value())))
            self.close_high.setChecked(bool(gripper.get("high_closes", self.close_high.isChecked())))

        points = data.get("auto_place_points_m", [])
        if isinstance(points, list):
            self.place_points = [
                (float(p[0]), float(p[1]), float(p[2]))
                for p in points
                if isinstance(p, (list, tuple)) and len(p) >= 3
            ]
            self.refresh_place_points()

        self.update_home_label()
        self.append_log(f"Loaded {self.config_path}")

    def update_home_label(self):
        if not self.home_pose:
            self.home_label.setText("Home: not set")
            return
        h = self.home_pose
        self.home_label.setText(
            f"Home: X={h['x']:.4f} Y={h['y']:.4f} Z={h['z']:.4f} R={h['r']:.1f}"
        )

    def build_manual_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        move_box = QGroupBox("Manual Cartesian Move")
        form = QFormLayout(move_box)
        form.addRow("X (m)", self.manual_x)
        form.addRow("Y (m)", self.manual_y)
        form.addRow("Z (m)", self.manual_z)
        form.addRow("R yaw (deg)", self.manual_r)
        form.addRow("Jog step (m)", self.jog_step)

        row = QHBoxLayout()
        btn_read = QPushButton("Read Pose")
        btn_movj = QPushButton("MoveJ")
        btn_movl = QPushButton("MoveL")
        btn_read.clicked.connect(self.read_pose)
        btn_movj.clicked.connect(lambda: self.manual_move(False))
        btn_movl.clicked.connect(lambda: self.manual_move(True))
        row.addWidget(btn_read)
        row.addWidget(btn_movj)
        row.addWidget(btn_movl)
        form.addRow(row)

        jog_box = QGroupBox("Jog")
        grid = QGridLayout(jog_box)
        for label, axis, sign, row_i, col_i in (
            ("X-", "x", -1, 1, 0), ("X+", "x", 1, 1, 2),
            ("Y+", "y", 1, 0, 1), ("Y-", "y", -1, 2, 1),
            ("Z+", "z", 1, 0, 3), ("Z-", "z", -1, 2, 3),
        ):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, a=axis, s=sign: self.jog(a, s))
            grid.addWidget(b, row_i, col_i)

        do_box = QGroupBox("Gripper / DO")
        do_row = QHBoxLayout(do_box)
        btn_open = QPushButton("Open")
        btn_close = QPushButton("Close")
        btn_open.clicked.connect(lambda: self.set_gripper_from_ui(open_gripper=True))
        btn_close.clicked.connect(lambda: self.set_gripper_from_ui(open_gripper=False))
        do_row.addWidget(btn_open)
        do_row.addWidget(btn_close)

        layout.addWidget(move_box)
        layout.addWidget(jog_box)
        layout.addWidget(do_box)
        layout.addStretch(1)
        return tab

    def build_robot_state_box(self):
        state_box = QGroupBox("Robot State")
        state_layout = QVBoxLayout(state_box)
        bringup_row = QGridLayout()
        btn_launch = QPushButton("Launch Bringup")
        btn_stop = QPushButton("Stop Bringup")
        btn_check = QPushButton("Check Services")
        btn_launch.clicked.connect(self.launch_bringup)
        btn_stop.clicked.connect(self.stop_bringup)
        btn_check.clicked.connect(self.check_services)
        bringup_row.addWidget(QLabel("IP"), 0, 0)
        bringup_row.addWidget(self.robot_ip, 0, 1, 1, 2)
        bringup_row.addWidget(btn_launch, 1, 0)
        bringup_row.addWidget(btn_stop, 1, 1)
        bringup_row.addWidget(btn_check, 1, 2)
        state_layout.addLayout(bringup_row)
        state_layout.addWidget(self.bringup_status)

        home_row = QHBoxLayout()
        btn_set_home = QPushButton("Save Home")
        btn_home = QPushButton("Go Home")
        btn_load_cfg = QPushButton("Load YAML")
        btn_set_home.clicked.connect(self.save_home_from_manual)
        btn_home.clicked.connect(self.go_home)
        btn_load_cfg.clicked.connect(self.load_robot_config)
        home_row.addWidget(btn_set_home)
        home_row.addWidget(btn_home)
        home_row.addWidget(btn_load_cfg)
        state_layout.addWidget(self.home_label)
        state_layout.addLayout(home_row)

        state_row = QHBoxLayout()
        btn_clear = QPushButton("Clear Error")
        btn_enable = QPushButton("Enable")
        btn_disable = QPushButton("Disable")
        btn_clear.clicked.connect(lambda: self.robot_state_command("clear"))
        btn_enable.clicked.connect(lambda: self.robot_state_command("enable"))
        btn_disable.clicked.connect(lambda: self.robot_state_command("disable"))
        state_row.addWidget(btn_clear)
        state_row.addWidget(btn_enable)
        state_row.addWidget(btn_disable)
        state_layout.addLayout(state_row)
        return state_box

    def build_camera_pick_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        self.pick_image = ImagePanel("Camera Pick - Rectified Left")
        self.pick_image.clicked.connect(self.on_camera_pick_click)
        layout.addWidget(self.pick_image, 2)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        form = QFormLayout()
        form.addRow("Robot X offset (mm)", self.click_x_offset)
        form.addRow("Robot Y offset (mm)", self.click_y_offset)
        form.addRow("Robot Z offset (mm)", self.click_z_offset)
        form.addRow("R yaw (deg)", self.click_r)
        form.addRow("Pre-approach Z above target (mm)", self.click_approach_z)
        form.addRow("Near-approach Z above target (mm)", self.click_near_approach_z)
        form.addRow("Speed (%)", self.click_speed)
        form.addRow("Near-approach speed (%)", self.click_near_speed)
        form.addRow("Accel (%)", self.click_accel)
        form.addRow("Near-approach accel (%)", self.click_near_accel)
        form.addRow("Move type", self.click_move_type)
        controls_layout.addLayout(form)
        controls_layout.addWidget(self.click_label)

        detect_box = QGroupBox("Pellet Detection")
        detect_layout = QVBoxLayout(detect_box)
        detect_form = QFormLayout()
        detect_form.addRow("Box model", self.pellet_box_model)
        detect_form.addRow("Seg model", self.pellet_seg_model)
        detect_form.addRow("Box conf", self.pellet_box_conf)
        detect_form.addRow("Seg conf", self.pellet_seg_conf)
        detect_form.addRow("Seg IoU", self.pellet_seg_iou)
        detect_form.addRow("R offset (deg)", self.pellet_r_offset)
        detect_form.addRow("Inference device", self.pellet_device)
        detect_layout.addLayout(detect_form)
        detect_layout.addWidget(self.pellet_use_angle)
        detect_layout.addWidget(self.pellet_show_confidence)
        detect_layout.addWidget(self.pick_roi_enabled)
        detect_layout.addWidget(self.pick_roi_crop)
        detect_layout.addWidget(self.pick_roi_label)
        roi_row = QHBoxLayout()
        btn_set_roi = QPushButton("Set ROI")
        btn_clear_roi = QPushButton("Clear ROI")
        btn_full_roi = QPushButton("Full ROI")
        btn_set_roi.clicked.connect(self.start_pick_roi_selection)
        btn_clear_roi.clicked.connect(self.clear_pick_roi)
        btn_full_roi.clicked.connect(self.set_full_pick_roi)
        roi_row.addWidget(btn_set_roi)
        roi_row.addWidget(btn_clear_roi)
        roi_row.addWidget(btn_full_roi)
        detect_layout.addLayout(roi_row)
        detect_layout.addWidget(self.pellet_status)
        detect_layout.addWidget(self.pellet_table)
        detect_row = QHBoxLayout()
        btn_load_detector = QPushButton("Load Detector")
        btn_detect = QPushButton("Detect Current")
        btn_pick_detected = QPushButton("Pick Detected")
        btn_select_detected = QPushButton("Select Row")
        btn_load_detector.clicked.connect(self.load_pellet_detector)
        btn_detect.clicked.connect(self.detect_current_pellet)
        btn_pick_detected.clicked.connect(self.pick_detected_target)
        btn_select_detected.clicked.connect(self.select_current_pellet_row)
        detect_row.addWidget(btn_load_detector)
        detect_row.addWidget(btn_detect)
        detect_row.addWidget(btn_select_detected)
        detect_row.addWidget(btn_pick_detected)
        detect_layout.addLayout(detect_row)
        loop_box = QGroupBox("Pellet Home Loop")
        loop_layout = QVBoxLayout(loop_box)
        loop_form = QFormLayout()
        loop_form.addRow("Loop delay (s)", self.pellet_loop_delay)
        loop_form.addRow("Home settle (s)", self.pellet_home_settle)
        loop_form.addRow("Max cycles (0=forever)", self.pellet_loop_max_cycles)
        loop_layout.addLayout(loop_form)
        loop_layout.addWidget(self.pellet_loop_status)
        loop_row = QHBoxLayout()
        btn_loop_start = QPushButton("Start Loop")
        btn_loop_stop = QPushButton("Stop Loop")
        btn_loop_start.clicked.connect(self.start_pellet_home_loop)
        btn_loop_stop.clicked.connect(self.stop_pellet_home_loop)
        loop_row.addWidget(btn_loop_start)
        loop_row.addWidget(btn_loop_stop)
        loop_layout.addLayout(loop_row)
        detect_layout.addWidget(loop_box)
        controls_layout.addWidget(detect_box)

        row = QGridLayout()
        btn_start = QPushButton("Start Camera")
        btn_stop = QPushButton("Stop Camera")
        btn_reload = QPushButton("Reload Calib + Hand-Eye")
        btn_move = QPushButton("Move To Click")
        btn_pick = QPushButton("Pick Clicked")
        btn_uncover = QPushButton("Uncover")
        btn_save_offset = QPushButton("Save Offset")
        self.btn_heatmap = QPushButton("Depth Heatmap")
        self.btn_heatmap.setCheckable(True)
        btn_start.clicked.connect(self.start_camera_pick)
        btn_stop.clicked.connect(self.stop_camera_pick)
        btn_reload.clicked.connect(self.reload_camera_pick_calibration)
        btn_move.clicked.connect(self.move_to_clicked_target)
        btn_pick.clicked.connect(self.pick_clicked_target)
        btn_uncover.clicked.connect(self.go_home)
        btn_save_offset.clicked.connect(self.save_camera_pick_offset)
        row.addWidget(btn_start, 0, 0)
        row.addWidget(btn_stop, 0, 1)
        row.addWidget(btn_reload, 0, 2)
        row.addWidget(btn_move, 1, 0)
        row.addWidget(btn_pick, 1, 1)
        row.addWidget(btn_uncover, 1, 2)
        row.addWidget(btn_save_offset, 2, 0, 1, 2)
        row.addWidget(self.btn_heatmap, 2, 2)
        controls_layout.addLayout(row)
        controls_layout.addStretch(1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls)
        controls_scroll.setMinimumWidth(360)
        layout.addWidget(controls_scroll, 1)

        return tab

    def reload_camera_pick_calibration(self):
        self.load_hand_eye()
        try:
            path = project_root() / self.camera_cfg["calibration"]["output_path"]
            calib = load_yaml(path)
            self._apply_camera_pick_calibration(calib)
            self.append_log(f"Loaded camera-pick calibration: {path}")
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"Camera-pick calibration failed: {exc}")

    def _apply_camera_pick_calibration(self, calib):
        w, h = calib.image_size
        self.camera_cfg["camera"]["frame_width"] = int(w) * 2
        self.camera_cfg["camera"]["frame_height"] = int(h)
        self.cam_calib = calib
        self.cam_rectifier = Rectifier(calib)
        try:
            self.cam_depth_engine = build_depth_engine(self.camera_cfg, self.cam_rectifier)
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"Camera-pick depth fallback to SGBM: {exc}")
            self.camera_cfg.setdefault("depth", {})["engine"] = "sgbm"
            self.cam_depth_engine = DepthEngine(self.camera_cfg, self.cam_rectifier)
        if self.capture is not None:
            self._start_camera_pick_worker()

    def start_camera_pick(self):
        if self.capture is not None:
            return
        if self.cam_rectifier is None or self.cam_depth_engine is None:
            self.reload_camera_pick_calibration()
        if self.cam_rectifier is None or self.cam_depth_engine is None:
            QMessageBox.warning(self, "Camera Pick", "Load stereo calibration first.")
            return
        try:
            self.camera = StereoCamera(self.camera_cfg)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Camera Pick", f"Camera startup failed: {exc}")
            return
        self.capture = CaptureThread(self.camera, self)
        self.capture.frames_ready.connect(self.on_camera_pick_frames)
        self.capture.error.connect(lambda e: self.log_received.emit(f"Camera pick: {e}"))
        self.capture.start()
        self._start_camera_pick_worker()
        self.append_log("Camera pick stream started.")

    def _start_camera_pick_worker(self):
        if self.depth_worker:
            self.depth_worker.stop()
        self.depth_worker = DepthWorker(self.cam_rectifier, self.cam_depth_engine, self)
        self.depth_worker.result_ready.connect(self.on_camera_pick_depth)
        self.depth_worker.start()

    def stop_camera_pick(self):
        self.stop_pellet_home_loop()
        if self.depth_worker:
            self.depth_worker.stop()
            self.depth_worker = None
        if self.capture:
            self.capture.stop()
            self.capture = None
        self.camera = None
        self.append_log("Camera pick stream stopped.")

    def on_camera_pick_frames(self, left, right):
        if self.depth_worker:
            self.depth_worker.submit(left, right)

    def on_camera_pick_depth(self, result):
        if "error" in result:
            self.append_log(f"Camera pick depth error: {result['error']}")
            return
        self.latest_depth = result["depth_map"]
        self.latest_left_rect = result["left"].copy()
        if self.btn_heatmap.isChecked() and result.get("color") is not None:
            display = result["color"].copy()
        else:
            display = result["left"].copy()
        self.draw_pick_roi_overlay(display)
        self.draw_pellet_overlay(display)
        if self.clicked_robot is not None:
            xy = self.clicked_robot.get("uv")
            if xy is not None:
                cv2.drawMarker(display, tuple(xy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        self.pick_image.show_image(display)

    def on_camera_pick_click(self, x, y):
        if self.roi_selecting:
            self.handle_pick_roi_click(int(x), int(y))
            return
        self.set_camera_pick_target_from_uv(int(x), int(y), source="click")

    def set_camera_pick_target_from_uv(self, x, y, source="click", pellet=None):
        if self.pick_roi_enabled.isChecked() and not self.point_in_pick_roi(x, y):
            self.click_label.setText(f"{source.title()} target: outside pickup ROI")
            self.clicked_robot = None
            return False
        if self.cam_calib is None or self.cam_depth_engine is None or self.latest_depth is None:
            QMessageBox.information(self, "Camera Pick", "Need live calibrated depth first.")
            return False
        if self.hand_eye_T is None:
            QMessageBox.warning(self, "Camera Pick", "Load config/hand_eye.yaml first.")
            return False
        info = self.cam_depth_engine.pixel_info_from_map(self.latest_depth, x, y)
        if not info.valid:
            self.click_label.setText(f"{source.title()} target: no valid depth")
            self.clicked_robot = None
            return False
        cam_mm = pixel_to_camera_xyz(int(x), int(y), info.depth_mm, self.cam_calib.P1)
        cam_m = np.asarray(cam_mm, dtype=np.float64) / 1000.0
        robot = self.hand_eye_T @ np.array([cam_m[0], cam_m[1], cam_m[2], 1.0])
        robot_base = (
            float(robot[0]),
            float(robot[1]),
            float(robot[2]),
        )
        self.clicked_robot = {
            "uv": (int(x), int(y)),
            "camera_m": tuple(float(v) for v in cam_m),
            "robot_base_m": robot_base,
            "depth_mm": float(info.depth_mm),
            "std_mm": float(info.sample_std_mm),
            "source": source,
            "pellet": pellet,
        }
        if pellet is not None and self.pellet_use_angle.isChecked():
            r = float(pellet.get("orientation", 0.0)) + self.pellet_r_offset.value()
            while r > 180.0:
                r -= 360.0
            while r < -180.0:
                r += 360.0
            self.click_r.setValue(r)
        self.update_clicked_robot_offset()
        return True

    def load_pellet_detector(self):
        box_path = self.pellet_box_model.text().strip()
        seg_path = self.pellet_seg_model.text().strip()
        paths = (box_path, seg_path)
        if self.pellet_detector is not None and self.pellet_loaded_paths == paths:
            self.pellet_status.setText("Pellet detector: loaded")
            return True
        if not box_path or not seg_path:
            QMessageBox.warning(self, "Pellet Detection", "Set both model paths first.")
            return False
        try:
            detector_cls = load_pellet_detector_class()
            self.pellet_detector = detector_cls(box_path, seg_path)
            self.pellet_loaded_paths = paths
        except Exception as exc:  # noqa: BLE001
            self.pellet_detector = None
            self.pellet_loaded_paths = None
            self.pellet_status.setText(f"Pellet detector load failed: {exc}")
            self.append_log(f"Pellet detector load failed: {exc}")
            return False
        self.pellet_status.setText("Pellet detector: loaded")
        self.append_log("Pellet detector loaded.")
        return True

    def pellet_device_arg(self):
        mode = self.pellet_device.currentText()
        if mode == "GPU 0":
            return "0"
        if mode == "CPU":
            return "cpu"
        return None

    def configure_pellet_device(self):
        device = self.pellet_device_arg()
        if self.pellet_detector is None or device is None:
            return
        if device != "cpu":
            try:
                import torch
                if not torch.cuda.is_available():
                    self.append_log("GPU requested, but torch CUDA is not available; inference may fail or stay on CPU.")
            except Exception as exc:  # noqa: BLE001
                self.append_log(f"GPU requested, but torch CUDA check failed: {exc}")
        for attr in ("box_model", "seg_model"):
            model = getattr(self.pellet_detector, attr, None)
            overrides = getattr(model, "overrides", None)
            if isinstance(overrides, dict):
                overrides["device"] = device
            predictor = getattr(model, "predictor", None)
            args = getattr(predictor, "args", None)
            if args is not None and hasattr(args, "device"):
                args.device = device

    def run_pellet_detector_on_frame(self, frame):
        roi = self.normalized_pick_roi() if self.pick_roi_enabled.isChecked() else None
        roi_offset = (0, 0)
        detect_frame = frame
        if roi is not None and self.pick_roi_crop.isChecked():
            x, y, w, h = roi
            detect_frame = frame[y:y + h, x:x + w].copy()
            roi_offset = (x, y)
        result = self.pellet_detector.predict(
            image_source=detect_frame,
            box_conf=self.pellet_box_conf.value(),
            seg_conf=self.pellet_seg_conf.value(),
            seg_iou=self.pellet_seg_iou.value(),
        )
        if roi_offset != (0, 0) and result.get("status") == "success":
            ox, oy = roi_offset
            for det in result.get("data", []):
                det["centroid_x"] = float(det["centroid_x"]) + ox
                det["centroid_y"] = float(det["centroid_y"]) + oy
        return result

    def camera_pick_target_from_pellet(self, pellet, depth_map):
        x = int(round(float(pellet["centroid_x"])))
        y = int(round(float(pellet["centroid_y"])))
        if self.pick_roi_enabled.isChecked() and not self.point_in_pick_roi(x, y):
            return None, "outside pickup ROI"
        if self.cam_calib is None or self.cam_depth_engine is None or depth_map is None:
            return None, "need calibrated depth"
        if self.hand_eye_T is None:
            return None, "need hand-eye calibration"
        info = self.cam_depth_engine.pixel_info_from_map(depth_map, x, y)
        if not info.valid:
            return None, "no valid depth"
        cam_mm = pixel_to_camera_xyz(x, y, info.depth_mm, self.cam_calib.P1)
        cam_m = np.asarray(cam_mm, dtype=np.float64) / 1000.0
        robot = self.hand_eye_T @ np.array([cam_m[0], cam_m[1], cam_m[2], 1.0])
        robot_base = (float(robot[0]), float(robot[1]), float(robot[2]))
        ox = self.click_x_offset.value() / 1000.0
        oy = self.click_y_offset.value() / 1000.0
        oz = self.click_z_offset.value() / 1000.0
        robot_m = (robot_base[0] + ox, robot_base[1] + oy, robot_base[2] + oz)
        r = self.click_r.value()
        if self.pellet_use_angle.isChecked():
            r = float(pellet.get("orientation", 0.0)) + self.pellet_r_offset.value()
            while r > 180.0:
                r -= 360.0
            while r < -180.0:
                r += 360.0
        target = {
            "uv": (x, y),
            "robot_m": robot_m,
            "robot_base_m": robot_base,
            "depth_mm": float(info.depth_mm),
            "std_mm": float(info.sample_std_mm),
            "r": r,
            "pellet": pellet,
        }
        return target, None

    def current_image_size(self):
        if self.latest_left_rect is None:
            return None
        h, w = self.latest_left_rect.shape[:2]
        return w, h

    def normalized_pick_roi(self):
        if self.pick_roi is None:
            return None
        size = self.current_image_size()
        if size is None:
            return self.pick_roi
        w, h = size
        x, y, rw, rh = self.pick_roi
        x = max(0, min(int(x), w - 1))
        y = max(0, min(int(y), h - 1))
        rw = max(1, min(int(rw), w - x))
        rh = max(1, min(int(rh), h - y))
        return x, y, rw, rh

    def update_pick_roi_label(self):
        roi = self.normalized_pick_roi()
        if roi is None:
            self.pick_roi_label.setText("Pickup ROI: full image")
            return
        x, y, w, h = roi
        self.pick_roi_label.setText(f"Pickup ROI: x={x} y={y} w={w} h={h}")

    def start_pick_roi_selection(self):
        if self.latest_left_rect is None:
            QMessageBox.information(self, "Pickup ROI", "Start camera and wait for an image first.")
            return
        self.roi_selecting = True
        self.roi_first_point = None
        self.pick_roi_enabled.setChecked(True)
        self.pick_roi_label.setText("Pickup ROI: click first corner")

    def handle_pick_roi_click(self, x, y):
        if self.roi_first_point is None:
            self.roi_first_point = (int(x), int(y))
            self.pick_roi_label.setText("Pickup ROI: click opposite corner")
            return
        x0, y0 = self.roi_first_point
        x1, y1 = int(x), int(y)
        rx = min(x0, x1)
        ry = min(y0, y1)
        rw = abs(x1 - x0) + 1
        rh = abs(y1 - y0) + 1
        self.pick_roi = (rx, ry, rw, rh)
        self.roi_selecting = False
        self.roi_first_point = None
        self.update_pick_roi_label()
        self.append_log(f"Pickup ROI set: {self.pick_roi_label.text()}")

    def clear_pick_roi(self):
        self.pick_roi = None
        self.roi_selecting = False
        self.roi_first_point = None
        self.pellet_detections = []
        self.pellet_selected = None
        self.refresh_pellet_table()
        self.update_pick_roi_label()

    def set_full_pick_roi(self):
        size = self.current_image_size()
        if size is None:
            self.clear_pick_roi()
            return
        w, h = size
        self.pick_roi = (0, 0, int(w), int(h))
        self.pick_roi_enabled.setChecked(True)
        self.roi_selecting = False
        self.roi_first_point = None
        self.update_pick_roi_label()

    def point_in_pick_roi(self, x, y):
        roi = self.normalized_pick_roi()
        if roi is None:
            return True
        rx, ry, rw, rh = roi
        return rx <= int(x) < rx + rw and ry <= int(y) < ry + rh

    def filter_detections_to_pick_roi(self, detections):
        if not self.pick_roi_enabled.isChecked() or self.normalized_pick_roi() is None:
            return detections
        out = []
        for det in detections:
            try:
                x = int(round(float(det["centroid_x"])))
                y = int(round(float(det["centroid_y"])))
            except Exception:  # noqa: BLE001
                continue
            if self.point_in_pick_roi(x, y):
                out.append(det)
        return out

    def refresh_pellet_table(self):
        self.pellet_table.blockSignals(True)
        self.pellet_table.setRowCount(len(self.pellet_detections))
        for row, det in enumerate(self.pellet_detections):
            try:
                x = int(round(float(det["centroid_x"])))
                y = int(round(float(det["centroid_y"])))
            except Exception:  # noqa: BLE001
                x, y = 0, 0
            values = [
                str(row),
                f"{float(det.get('score', 0.0)):.3f}",
                str(det.get("color", "unknown")),
                f"{float(det.get('orientation', 0.0)):.1f}",
                f"{x},{y}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, row)
                self.pellet_table.setItem(row, col, item)
        self.pellet_table.resizeColumnsToContents()
        if self.pellet_selected in self.pellet_detections:
            self.pellet_table.selectRow(self.pellet_detections.index(self.pellet_selected))
        else:
            self.pellet_table.clearSelection()
        self.pellet_table.blockSignals(False)

    def select_pellet_detection(self, index):
        if index < 0 or index >= len(self.pellet_detections):
            return False
        pellet = self.pellet_detections[index]
        self.pellet_selected = pellet
        self.refresh_pellet_table()
        x = int(round(float(pellet["centroid_x"])))
        y = int(round(float(pellet["centroid_y"])))
        ok = self.set_camera_pick_target_from_uv(x, y, source="pellet", pellet=pellet)
        color = pellet.get("color", "unknown")
        self.pellet_status.setText(
            f"Pellet selected: row={index} uv=({x},{y}) "
            f"score={float(pellet.get('score', 0.0)):.2f} "
            f"angle={float(pellet.get('orientation', 0.0)):.1f} color={color}"
        )
        return ok

    def selected_pellet_row(self):
        rows = self.pellet_table.selectionModel().selectedRows()
        if not rows:
            return -1
        return int(rows[0].row())

    def select_current_pellet_row(self):
        row = self.selected_pellet_row()
        if row < 0:
            QMessageBox.information(self, "Pellet Detection", "Select a target row first.")
            return
        self.select_pellet_detection(row)

    def on_pellet_table_selection_changed(self):
        row = self.selected_pellet_row()
        if row >= 0:
            self.select_pellet_detection(row)

    def detect_current_pellet(self):
        if self.latest_left_rect is None or self.latest_depth is None:
            QMessageBox.information(self, "Pellet Detection", "Start camera and wait for a depth frame first.")
            return
        if self.hand_eye_T is None:
            QMessageBox.warning(self, "Pellet Detection", "Load config/hand_eye.yaml first.")
            return
        if self.pellet_busy:
            return
        if not self.load_pellet_detector():
            return
        self.configure_pellet_device()
        frame = self.latest_left_rect.copy()
        self.pellet_busy = True
        self.pellet_status.setText("Pellet detector: running ...")

        def run():
            try:
                result = self.run_pellet_detector_on_frame(frame)
            except Exception as exc:  # noqa: BLE001
                result = {"status": "error", "message": str(exc), "data": []}
            self.pellet_result_received.emit(result)

        threading.Thread(target=run, daemon=True).start()

    def on_pellet_result(self, result):
        self.pellet_busy = False
        if result.get("status") != "success":
            self.pellet_detections = []
            self.pellet_selected = None
            self.refresh_pellet_table()
            msg = result.get("message", "no pellet detected")
            self.pellet_status.setText(f"Pellet detector: {msg}")
            self.append_log(f"Pellet detection: {msg}")
            return

        raw_detections = list(result.get("data", []))
        detections = self.filter_detections_to_pick_roi(raw_detections)
        self.pellet_detections = detections
        if not detections:
            self.pellet_selected = None
            self.refresh_pellet_table()
            if raw_detections:
                self.pellet_status.setText("Pellet detector: no pellets inside pickup ROI")
            else:
                self.pellet_status.setText("Pellet detector: no pellets")
            return

        # Prototype API treats the first pellet as priority 0.
        ok = self.select_pellet_detection(0)
        pellet = self.pellet_detections[0]
        color = pellet.get("color", "unknown")
        self.pellet_status.setText(
            f"Pellet detector: {len(detections)} found, selected "
            f"uv=({int(round(float(pellet['centroid_x'])))},{int(round(float(pellet['centroid_y'])))}) "
            f"score={float(pellet.get('score', 0.0)):.2f} "
            f"angle={float(pellet.get('orientation', 0.0)):.1f} color={color}"
        )
        if ok:
            self.append_log(
                f"Pellet target uv=({int(round(float(pellet['centroid_x'])))},"
                f"{int(round(float(pellet['centroid_y'])))}) "
                f"score={float(pellet.get('score', 0.0)):.2f} "
                f"angle={float(pellet.get('orientation', 0.0)):.1f} color={color}"
            )

    def draw_pellet_overlay(self, display):
        for det in self.pellet_detections:
            try:
                x = int(round(float(det["centroid_x"])))
                y = int(round(float(det["centroid_y"])))
            except Exception:  # noqa: BLE001
                continue
            color = (0, 180, 255)
            if det is self.pellet_selected:
                color = (0, 255, 255)
            radius = 5 if det is self.pellet_selected else 4
            cv2.circle(display, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(display, (x, y), radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
            if self.pellet_show_confidence.isChecked():
                label = f"{det.get('color', '?')} {float(det.get('score', 0.0)):.2f}"
                cv2.putText(
                    display, label, (x + 8, max(16, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )

    def draw_pick_roi_overlay(self, display):
        roi = self.normalized_pick_roi()
        if roi is None:
            return
        x, y, w, h = roi
        color = (255, 180, 0)
        cv2.rectangle(display, (x, y), (x + w - 1, y + h - 1), color, 2)
        cv2.putText(
            display, "pickup ROI", (x + 6, max(18, y + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

    def update_clicked_robot_offset(self):
        if self.clicked_robot is None or "robot_base_m" not in self.clicked_robot:
            return
        bx, by, bz = self.clicked_robot["robot_base_m"]
        ox = self.click_x_offset.value() / 1000.0
        oy = self.click_y_offset.value() / 1000.0
        oz = self.click_z_offset.value() / 1000.0
        robot_xyz = (float(bx) + ox, float(by) + oy, float(bz) + oz)
        self.clicked_robot["robot_m"] = robot_xyz
        uv = self.clicked_robot.get("uv", ("-", "-"))
        depth = self.clicked_robot.get("depth_mm", float("nan"))
        std = self.clicked_robot.get("std_mm", float("nan"))
        source = self.clicked_robot.get("source", "click")
        pellet = self.clicked_robot.get("pellet") or {}
        pellet_text = ""
        if pellet:
            pellet_text = (
                f"  pellet={pellet.get('color', 'unknown')} "
                f"score={float(pellet.get('score', 0.0)):.2f} "
                f"angle={float(pellet.get('orientation', 0.0)):.1f}"
            )
        self.click_label.setText(
            f"{source.title()} target: "
            f"uv=({uv[0]},{uv[1]}) depth={depth:.1f} +/-{std:.1f} mm  "
            f"offset=({self.click_x_offset.value():+.1f}, {self.click_y_offset.value():+.1f}, "
            f"{self.click_z_offset.value():+.1f}) mm  "
            f"robot=({robot_xyz[0]:+.4f}, {robot_xyz[1]:+.4f}, {robot_xyz[2]:+.4f}) m"
            f"{pellet_text}"
        )

    def save_camera_pick_offset(self):
        self.save_robot_config()
        self.append_log(
            f"Saved camera-pick offset: "
            f"X={self.click_x_offset.value():+.1f} "
            f"Y={self.click_y_offset.value():+.1f} "
            f"Z={self.click_z_offset.value():+.1f} mm  "
            f"R={self.click_r.value():+.1f} deg  "
            f"approachZ={self.click_approach_z.value():.1f}/"
            f"{self.click_near_approach_z.value():.1f} mm  "
            f"speed={self.click_speed.value()}/{self.click_near_speed.value()}% "
            f"accel={self.click_accel.value()}/{self.click_near_accel.value()}% "
            f"{self.click_move_type.currentText()}"
        )

    def move_to_clicked_target(self):
        if self.node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        if self.clicked_robot is None:
            QMessageBox.information(self, "Camera Pick", "Click a valid depth point first.")
            return
        if self.busy:
            return
        x, y, z = self.clicked_robot["robot_m"]
        r = self.click_r.value()
        linear = self.click_move_type.currentText() == "MovL"
        speed = self.click_speed.value()
        accel = self.click_accel.value()
        threading.Thread(
            target=self.manual_move_sequence,
            args=(x, y, z, r, linear, speed, accel),
            daemon=True,
        ).start()

    def pick_clicked_target(self):
        if self.node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        if self.clicked_robot is None:
            QMessageBox.information(self, "Camera Pick", "Click a valid depth point first.")
            return
        if self.busy:
            return
        params = self.pick_params(use_place_list=False)
        if params is None:
            return
        params["r"] = self.click_r.value()
        target = self.clicked_robot["robot_m"]
        params["approach_z"] = target[2] + self.click_approach_z.value() / 1000.0
        params["release_at_approach"] = True
        params["linear"] = self.click_move_type.currentText() == "MovL"
        params["speed"] = self.click_speed.value()
        params["accel"] = self.click_accel.value()
        params["near_approach_z"] = target[2] + self.click_near_approach_z.value() / 1000.0
        params["near_speed"] = self.click_near_speed.value()
        params["near_accel"] = self.click_near_accel.value()
        threading.Thread(
            target=self.pick_sequence,
            args=(target, params),
            daemon=True,
        ).start()

    def pick_detected_target(self):
        if self.clicked_robot is None or self.clicked_robot.get("source") != "pellet":
            QMessageBox.information(self, "Pellet Detection", "Run Detect Current first.")
            return
        self.pick_clicked_target()

    def start_pellet_home_loop(self):
        if self.node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        if not self.home_pose:
            QMessageBox.information(self, "Pellet Loop", "Save a home pose first.")
            return
        if self.latest_left_rect is None or self.latest_depth is None:
            QMessageBox.information(self, "Pellet Loop", "Start camera and wait for depth first.")
            return
        if self.hand_eye_T is None:
            QMessageBox.warning(self, "Pellet Loop", "Load config/hand_eye.yaml first.")
            return
        if self.pellet_loop_running:
            return
        if not self.load_pellet_detector():
            return
        self.configure_pellet_device()
        self.pellet_loop_stop.clear()
        self.pellet_loop_running = True
        self.pellet_loop_status_received.emit("Pellet loop: running")
        self.save_robot_config()
        self.pellet_loop_thread = threading.Thread(
            target=self.pellet_home_loop_sequence,
            daemon=True,
        )
        self.pellet_loop_thread.start()

    def stop_pellet_home_loop(self):
        self.pellet_loop_stop.set()
        if self.pellet_loop_running:
            self.pellet_loop_status.setText("Pellet loop: stopping")
        else:
            self.pellet_loop_status.setText("Pellet loop: stopped")

    def pellet_pick_params_for_target(self, target):
        params = self.pick_params(use_place_list=False)
        if params is None:
            return None
        params["place"] = (
            float(self.home_pose["x"]),
            float(self.home_pose["y"]),
            float(self.home_pose["z"]),
        )
        params["place_i"] = None
        params["r"] = float(target["r"])
        params["approach_z"] = target["robot_m"][2] + self.click_approach_z.value() / 1000.0
        params["release_at_approach"] = False
        params["linear"] = self.click_move_type.currentText() == "MovL"
        params["speed"] = self.click_speed.value()
        params["accel"] = self.click_accel.value()
        params["near_approach_z"] = target["robot_m"][2] + self.click_near_approach_z.value() / 1000.0
        params["near_speed"] = self.click_near_speed.value()
        params["near_accel"] = self.click_near_accel.value()
        return params

    def pellet_home_loop_sequence(self):
        cycles = 0
        max_cycles = int(self.pellet_loop_max_cycles.value())
        delay_s = float(self.pellet_loop_delay.value())
        settle_s = float(self.pellet_home_settle.value())
        home = dict(self.home_pose)
        try:
            while not self.pellet_loop_stop.is_set():
                if max_cycles > 0 and cycles >= max_cycles:
                    self.log_received.emit("Pellet loop reached max cycles.")
                    break
                if self.busy:
                    time.sleep(0.1)
                    continue
                self.busy = True
                should_delay = True
                try:
                    self.pellet_loop_status_received.emit(f"Pellet loop: cycle {cycles + 1}")
                    self.log_received.emit(f"Pellet loop cycle {cycles + 1}: go home")
                    if not self._move(home["x"], home["y"], home["z"], home["r"], linear=False):
                        break
                    if not self._set_do(
                        self.do_type.currentText() == "Tool DO",
                        self.do_index.value(),
                        1,
                    ):
                        break
                    if settle_s > 0:
                        time.sleep(settle_s)
                    frame = self.latest_left_rect.copy() if self.latest_left_rect is not None else None
                    depth = self.latest_depth.copy() if self.latest_depth is not None else None
                    if frame is None or depth is None:
                        self.log_received.emit("Pellet loop: no camera frame/depth yet.")
                        continue
                    result = self.run_pellet_detector_on_frame(frame)
                    if result.get("status") != "success":
                        self.pellet_result_received.emit(result)
                        self.log_received.emit(f"Pellet loop detect: {result.get('message', 'no target')}")
                        continue
                    detections = self.filter_detections_to_pick_roi(list(result.get("data", [])))
                    self.pellet_result_received.emit(result)
                    if not detections:
                        self.log_received.emit("Pellet loop: no target inside ROI.")
                        continue
                    pellet = max(detections, key=lambda det: float(det.get("score", 0.0)))
                    target, err = self.camera_pick_target_from_pellet(pellet, depth)
                    if target is None:
                        self.log_received.emit(f"Pellet loop target skipped: {err}")
                        continue
                    params = self.pellet_pick_params_for_target(target)
                    if params is None:
                        break
                    uv = target["uv"]
                    self.log_received.emit(
                        f"Pellet loop pick uv=({uv[0]},{uv[1]}) "
                        f"score={float(pellet.get('score', 0.0)):.2f}"
                    )
                    self.pick_sequence(target["robot_m"], params)
                    cycles += 1
                finally:
                    self.busy = False
                    if should_delay and delay_s > 0 and not self.pellet_loop_stop.is_set():
                        time.sleep(delay_s)
        finally:
            self.pellet_loop_running = False
            self.pellet_loop_stop.set()
            self.log_received.emit("Pellet loop stopped.")
            self.pellet_loop_status_received.emit("Pellet loop: stopped")

    def build_auto_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        info = QLabel("Auto loop picks the first current detection and places it at the next point in this list.")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addWidget(self.auto_loop)
        layout.addWidget(self.place_index_label)
        layout.addWidget(self.place_list, 1)

        place_form = QFormLayout()
        place_form.addRow("Place X (m)", self.auto_place_x)
        place_form.addRow("Place Y (m)", self.auto_place_y)
        place_form.addRow("Place Z (m)", self.auto_place_z)
        layout.addLayout(place_form)

        row1 = QHBoxLayout()
        btn_add = QPushButton("Add Place Point")
        btn_update = QPushButton("Update Selected")
        btn_delete = QPushButton("Delete Selected")
        btn_add.clicked.connect(self.add_place_point)
        btn_update.clicked.connect(self.update_place_point)
        btn_delete.clicked.connect(self.delete_place_point)
        row1.addWidget(btn_add)
        row1.addWidget(btn_update)
        row1.addWidget(btn_delete)

        row2 = QHBoxLayout()
        btn_clear = QPushButton("Clear")
        btn_reset = QPushButton("Reset Next")
        btn_pick_next = QPushButton("Pick Selected -> Next Place")
        btn_clear.clicked.connect(self.clear_place_points)
        btn_reset.clicked.connect(self.reset_place_index)
        btn_pick_next.clicked.connect(lambda: self.pick_selected(use_place_list=True))
        row2.addWidget(btn_clear)
        row2.addWidget(btn_reset)
        row2.addWidget(btn_pick_next)

        layout.addLayout(row1)
        layout.addLayout(row2)
        return tab

    def start_ros(self):
        if not ROS_OK:
            self.statusBar().showMessage(f"ROS2 unavailable: {ROS_IMPORT_ERROR}")
            return
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.node = PickNode(lambda items: self.detections_received.emit(items))
            self.spin = RosSpinThread(self.node)
            self.spin.start()
        except Exception as exc:  # noqa: BLE001
            self.node = None
            self.spin = None
            self.statusBar().showMessage(f"ROS2 startup failed: {exc}")
            self.append_log(f"ROS2 startup failed: {exc}")
            return
        self.statusBar().showMessage("Listening on /elp/detections")

    def launch_bringup(self):
        if self.bringup_proc and self.bringup_proc.poll() is None:
            self.append_log("MG400 bringup already running from this app.")
            return
        ip = self.robot_ip.text().strip() or "192.168.1.6"
        cmd = [
            "ros2", "launch", "mg400_bringup", "mg400_gui.launch.py",
            f"ip_address:={ip}",
        ]
        try:
            self.bringup_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            self.append_log(f"Launch bringup failed: {exc}")
            self.bringup_status.setText(f"Bringup: launch failed ({exc})")
            return
        self.bringup_status.setText("Bringup: starting, waiting for MG400 services ...")
        self.append_log("Started: " + " ".join(cmd))
        threading.Thread(target=self.drain_bringup_output, daemon=True).start()
        if self.node is None:
            QTimer.singleShot(3000, self.start_ros)
        self.bringup_ready_checks_remaining = 25
        QTimer.singleShot(1000, self.poll_bringup_ready)

    def drain_bringup_output(self):
        proc = self.bringup_proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if line:
                self.log_received.emit("[bringup] " + line)

    def stop_bringup(self, silent=False):
        if not self.bringup_proc or self.bringup_proc.poll() is not None:
            if not silent:
                self.append_log("No bringup process started by this app.")
            return
        self.bringup_proc.terminate()
        try:
            self.bringup_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.bringup_proc.kill()
        self.bringup_ready_checks_remaining = 0
        self.bringup_status.setText("Bringup: stopped")
        self.append_log("Stopped MG400 bringup process.")

    def mg400_service_status(self):
        if self.node is None:
            return False, "ROS2 node is not running", []
        names = sorted(name for name, _types in self.node.get_service_names_and_types())
        needed = [
            "/mg400/clear_error",
            "/mg400/enable_robot",
            "/mg400/disable_robot",
            "/mg400/get_pose",
            "/mg400/do_execute",
            "/mg400/tool_do_execute",
        ]
        missing = [name for name in needed if name not in names]
        if not missing:
            return True, "MG400 services ready", names
        return False, "Missing: " + ", ".join(missing), names

    def poll_bringup_ready(self):
        if self.bringup_ready_checks_remaining <= 0:
            return
        if self.node is None:
            self.bringup_status.setText("Bringup: waiting for ROS2 node ...")
            self.bringup_ready_checks_remaining -= 1
            QTimer.singleShot(1000, self.poll_bringup_ready)
            return
        ready, msg, _names = self.mg400_service_status()
        if ready:
            self.bringup_ready_checks_remaining = 0
            self.bringup_status.setText("Bringup: ready")
            self.append_log("MG400 bringup ready.")
            return
        self.bringup_status.setText(f"Bringup: waiting ({msg})")
        self.bringup_ready_checks_remaining -= 1
        if self.bringup_ready_checks_remaining > 0:
            QTimer.singleShot(1000, self.poll_bringup_ready)
        else:
            self.append_log(f"MG400 bringup not ready: {msg}")

    def check_services(self):
        ready, msg, names = self.mg400_service_status()
        if self.node is None:
            QMessageBox.warning(self, "ROS2", msg)
            return
        self.bringup_status.setText(f"Bringup: {'ready' if ready else msg}")
        needed = [
            "/mg400/clear_error",
            "/mg400/enable_robot",
            "/mg400/disable_robot",
            "/mg400/get_pose",
            "/mg400/do_execute",
            "/mg400/tool_do_execute",
        ]
        for name in needed:
            self.append_log(f"{name}: {'OK' if name in names else 'MISSING'}")
        visible = [n for n in names if "mg400" in n or n in {"/enable_robot", "/disable_robot", "/clear_error"}]
        if visible:
            self.append_log("Visible MG400-like services: " + ", ".join(visible))
        else:
            self.append_log("No MG400 services visible. Start bringup or source the ROS workspace.")

    def load_hand_eye(self):
        path = project_root() / "config" / "hand_eye.yaml"
        try:
            data, T = load_hand_eye_yaml(path)
            self.hand_eye_T = T
            self.robot_frame = data.get("parent_frame", "robot_base")
            self.camera_frame = data.get("child_frame", "camera_optical_frame")
            self.append_log(f"Loaded {path}")
        except Exception as exc:  # noqa: BLE001
            self.hand_eye_T = None
            self.append_log(f"No hand-eye fallback loaded: {exc}")

    def on_detections(self, items):
        self.update_place_index_label()
        self.detections = items
        self.list.clear()
        for i, item in enumerate(items):
            x, y, z = self.robot_xyz(item)
            frame = item["frame_id"]
            warn = "" if self.can_use_detection(item) else "  NOT ROBOT FRAME"
            self.list.addItem(
                f"{i}: {item['label']} {item['score']:.2f}  "
                f"({x:+.4f}, {y:+.4f}, {z:+.4f}) m  frame={frame}{warn}"
            )
        if self.auto_loop.isChecked() and items and not self.busy:
            self.list.setCurrentRow(0)
            self.pick_selected(use_place_list=True)
        elif self.auto_pick.isChecked() and items and not self.busy:
            self.list.setCurrentRow(0)
            self.pick_selected(use_place_list=False)

    def robot_xyz(self, item):
        xyz = np.asarray(item["xyz_m"], dtype=np.float64)
        frame = item.get("frame_id", "")
        if frame == self.robot_frame or self.hand_eye_T is None:
            return tuple(float(v) for v in xyz)
        if frame == self.camera_frame:
            q = self.hand_eye_T @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
            return (float(q[0]), float(q[1]), float(q[2]))
        return tuple(float(v) for v in xyz)

    def can_use_detection(self, item):
        frame = item.get("frame_id", "")
        return frame == self.robot_frame or (frame == self.camera_frame and self.hand_eye_T is not None)

    def read_pose(self):
        if self.node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        self.node.request_pose_async(lambda pose, err: self.pose_received.emit(pose, err))

    def on_pose_received(self, pose, err):
        if pose is None:
            self.append_log(f"Read pose failed: {err}")
            return
        x, y, z = pose
        self.manual_x.setValue(float(x))
        self.manual_y.setValue(float(y))
        self.manual_z.setValue(float(z))
        self.append_log(f"Pose read ({x:.4f}, {y:.4f}, {z:.4f})")

    def manual_move(self, linear):
        if self.node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        if self.busy:
            return
        x = self.manual_x.value()
        y = self.manual_y.value()
        z = self.manual_z.value()
        r = self.manual_r.value()
        threading.Thread(target=self.manual_move_sequence, args=(x, y, z, r, linear), daemon=True).start()

    def manual_move_sequence(self, x, y, z, r, linear, speed=None, accel=None):
        self.busy = True
        try:
            self._move(x, y, z, r, linear=linear, speed=speed, accel=accel)
        finally:
            self.busy = False

    def save_home_from_manual(self):
        self.home_pose = {
            "x": self.manual_x.value(),
            "y": self.manual_y.value(),
            "z": self.manual_z.value(),
            "r": self.manual_r.value(),
        }
        self.update_home_label()
        self.save_robot_config()

    def go_home(self):
        if self.node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        if not self.home_pose:
            QMessageBox.information(self, "Home", "No home pose saved yet.")
            return
        if self.busy:
            return
        h = dict(self.home_pose)
        threading.Thread(
            target=self.manual_move_sequence,
            args=(h["x"], h["y"], h["z"], h["r"], False),
            daemon=True,
        ).start()

    def jog(self, axis, sign):
        step = self.jog_step.value() * float(sign)
        if axis == "x":
            self.manual_x.setValue(self.manual_x.value() + step)
        elif axis == "y":
            self.manual_y.setValue(self.manual_y.value() + step)
        elif axis == "z":
            self.manual_z.setValue(self.manual_z.value() + step)
        self.manual_move(linear=False)

    def set_gripper_from_ui(self, open_gripper):
        if self.node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        # DO polarity hard-inverted: open = HIGH(1), close = LOW(0).
        state = 1 if open_gripper else 0
        use_tool = self.do_type.currentText() == "Tool DO"
        idx = self.do_index.value()
        threading.Thread(target=self._set_do, args=(use_tool, idx, state), daemon=True).start()

    def robot_state_command(self, command):
        if self.node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return

        def done(ok, msg):
            self.log_received.emit(f"{command}: {'ok' if ok else msg}")

        if command == "clear":
            self.node.clear_error_async(done)
        elif command == "enable":
            self.node.enable_robot_async(done)
        elif command == "disable":
            self.node.disable_robot_async(done)

    def add_place_point(self):
        self.place_points.append((
            self.auto_place_x.value(),
            self.auto_place_y.value(),
            self.auto_place_z.value(),
        ))
        self.refresh_place_points()
        self.save_robot_config()

    def update_place_point(self):
        row = self.place_list.currentRow()
        if row < 0 or row >= len(self.place_points):
            return
        self.place_points[row] = (
            self.auto_place_x.value(),
            self.auto_place_y.value(),
            self.auto_place_z.value(),
        )
        self.refresh_place_points()
        self.place_list.setCurrentRow(row)
        self.save_robot_config()

    def delete_place_point(self):
        row = self.place_list.currentRow()
        if row < 0 or row >= len(self.place_points):
            return
        self.place_points.pop(row)
        if self.place_points:
            self.place_index %= len(self.place_points)
        else:
            self.place_index = 0
            self.auto_loop.setChecked(False)
        self.refresh_place_points()
        self.save_robot_config()

    def clear_place_points(self):
        self.place_points.clear()
        self.place_index = 0
        self.auto_loop.setChecked(False)
        self.refresh_place_points()
        self.save_robot_config()

    def reset_place_index(self):
        self.place_index = 0
        self.refresh_place_points()

    def refresh_place_points(self):
        self.place_list.clear()
        for i, (x, y, z) in enumerate(self.place_points):
            prefix = "-> " if i == self.place_index % max(1, len(self.place_points)) else "   "
            self.place_list.addItem(f"{prefix}{i}: ({x:+.4f}, {y:+.4f}, {z:+.4f}) m")
        self.update_place_index_label()

    def update_place_index_label(self):
        n = len(self.place_points)
        if n:
            self.place_index_label.setText(f"Next place: {self.place_index % n} / {n}")
        else:
            self.place_index_label.setText("Next place: none")

    def on_place_selected(self, row):
        if row < 0 or row >= len(self.place_points):
            return
        x, y, z = self.place_points[row]
        self.auto_place_x.setValue(x)
        self.auto_place_y.setValue(y)
        self.auto_place_z.setValue(z)

    def pick_selected(self, use_place_list=False):
        if self.node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        row = self.list.currentRow()
        if row < 0 or row >= len(self.detections):
            QMessageBox.information(self, "Pick", "Select a detection first.")
            return
        if self.busy:
            return
        item = self.detections[row]
        if not self.can_use_detection(item):
            QMessageBox.warning(
                self,
                "Frame mismatch",
                f"Detection frame is {item.get('frame_id')!r}. "
                f"Need {self.robot_frame!r}, or {self.camera_frame!r} with config/hand_eye.yaml loaded.",
            )
            return
        pick = self.robot_xyz(item)
        params = self.pick_params(use_place_list=use_place_list)
        if params is None:
            return
        threading.Thread(target=self.pick_sequence, args=(pick, params), daemon=True).start()

    def pick_params(self, use_place_list=False):
        place = None
        place_i = None
        if use_place_list:
            if not self.place_points:
                QMessageBox.warning(self, "Auto Loop", "Add at least one place point first.")
                self.auto_loop.setChecked(False)
                return None
            place_i = self.place_index % len(self.place_points)
            place = self.place_points[place_i]
        else:
            place = (self.place_x.value(), self.place_y.value(), self.place_z.value())

        # DO polarity hard-inverted: open = HIGH(1), close = LOW(0).
        return {
            "approach_z": self.approach_z.value(),
            "pick_z_offset": self.pick_z_offset.value(),
            "place": place,
            "place_i": place_i,
            "r": self.r_deg.value(),
            "use_tool": self.do_type.currentText() == "Tool DO",
            "idx": self.do_index.value(),
            "close_state": 0,
            "open_state": 1,
        }

    def pick_sequence(self, pick, params):
        self.busy = True
        px, py, pz = pick
        pz += params["pick_z_offset"]
        approach_z = params["approach_z"]
        place = params["place"]
        r = params["r"]
        use_tool = params["use_tool"]
        idx = params["idx"]
        close_state = params["close_state"]
        open_state = params["open_state"]
        linear = params.get("linear", False)
        speed = params.get("speed")
        accel = params.get("accel")
        near_z = params.get("near_approach_z")
        near_speed = params.get("near_speed", speed)
        near_accel = params.get("near_accel", accel)
        descend_speed = near_speed if near_z is not None else speed
        descend_accel = near_accel if near_z is not None else accel
        try:
            self._log(f"Pick start ({px:.4f}, {py:.4f}, {pz:.4f})")
            if not self._set_do(use_tool, idx, open_state):
                return
            if not self._move(px, py, approach_z, r, linear=linear, speed=speed, accel=accel):
                return
            if near_z is not None:
                if not self._move(px, py, near_z, r, linear=linear,
                                  speed=near_speed, accel=near_accel):
                    return
            if not self._move(px, py, pz, r, linear=linear,
                              speed=descend_speed, accel=descend_accel):
                return
            if not self._set_do(use_tool, idx, close_state):
                return
            time.sleep(0.2)
            if not self._move(px, py, approach_z, r, linear=linear, speed=speed, accel=accel):
                return
            if not params.get("release_at_approach"):
                if not self._move(place[0], place[1], place[2], r,
                                  linear=linear, speed=speed, accel=accel):
                    return
            self._set_do(use_tool, idx, open_state)
            self._log("Pick/place complete")
            if params["place_i"] is not None:
                self.place_index = (params["place_i"] + 1) % len(self.place_points)
                self.log_received.emit(f"Next place index: {self.place_index}")
                self.place_points_changed.emit()
        finally:
            self.busy = False

    def _move(self, x, y, z, r, linear=False, speed=None, accel=None):
        evt = threading.Event()
        out = {"ok": False, "msg": "timeout"}
        if z < ROBOT_MIN_Z_M:
            self._log(f"Z limited from {z:.4f} to {ROBOT_MIN_Z_M:.4f} m")
            z = ROBOT_MIN_Z_M

        def done(ok, msg):
            out["ok"] = bool(ok)
            out["msg"] = msg
            evt.set()

        self._log(
            f"{'MovL' if linear else 'MovJ'} ({x:.4f}, {y:.4f}, {z:.4f}) "
            f"s={speed} a={accel}"
        )
        self.node.move_cartesian_async(
            x, y, z, r, is_linear=linear, on_done=done, speed=speed, accel=accel
        )
        evt.wait(timeout=30.0)
        if not out["ok"]:
            self._log(f"Move failed: {out['msg']}")
        return out["ok"]

    def _set_do(self, use_tool, idx, state):
        evt = threading.Event()
        out = {"ok": False, "msg": "timeout"}

        def done(ok, msg):
            out["ok"] = bool(ok)
            out["msg"] = msg
            evt.set()

        self._log(f"DO {'tool' if use_tool else 'base'}[{idx}]={state}")
        self.node.set_do_async(use_tool, idx, state, on_done=done)
        evt.wait(timeout=5.0)
        if not out["ok"]:
            self._log(f"DO failed: {out['msg']}")
        return out["ok"]

    def _log(self, msg):
        self.log_received.emit(msg)

    def append_log(self, msg):
        self.log.append(msg)
        self.statusBar().showMessage(msg)

    def closeEvent(self, event):
        self.stop_camera_pick()
        self.stop_bringup(silent=True)
        if self.spin:
            self.spin.stop()
        if self.node:
            self.node.destroy_node()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = RobotControlWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
