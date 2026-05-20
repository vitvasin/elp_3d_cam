#!/usr/bin/env python3
"""MG400 pick controller for the ELP detection pipeline.

Consumes ``vision_msgs/Detection3DArray`` from ``app_detect.py`` and executes a
simple pick/place sequence with the MG400 ROS2 action/services.
"""

import sys
import threading
import time
import subprocess

import numpy as np
import yaml
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
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
    QSpinBox,
    QStatusBar,
    QTabWidget,
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
        self.clicked_robot = None

        self.list = QListWidget()
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.robot_ip = QLineEdit("192.168.1.6")

        self.approach_z = self._spin(-0.500, 1.0, 0.150, 0.005)
        self.pick_z_offset = self._spin(-0.100, 0.100, 0.000, 0.001)
        self.place_x = self._spin(-1.0, 1.0, 0.300, 0.005)
        self.place_y = self._spin(-1.0, 1.0, 0.000, 0.005)
        self.place_z = self._spin(-0.500, 1.0, 0.150, 0.005)
        self.r_deg = self._spin(-180.0, 180.0, 0.0, 1.0)
        self.do_index = QSpinBox()
        self.do_index.setRange(1, 16)
        self.do_index.setValue(1)
        self.do_type = QComboBox()
        self.do_type.addItems(["Tool DO", "Base DO"])
        self.close_high = QCheckBox("HIGH closes gripper")
        self.close_high.setChecked(True)
        self.auto_pick = QCheckBox("Auto pick first detection")
        self.auto_loop = QCheckBox("Auto loop first detection to place list")

        self.manual_x = self._spin(-1.0, 1.0, 0.300, 0.005)
        self.manual_y = self._spin(-1.0, 1.0, 0.000, 0.005)
        self.manual_z = self._spin(-0.500, 1.0, 0.150, 0.005)
        self.manual_r = self._spin(-180.0, 180.0, 0.0, 1.0)
        self.jog_step = self._spin(0.001, 0.100, 0.010, 0.001)
        self.auto_place_x = self._spin(-1.0, 1.0, 0.300, 0.005)
        self.auto_place_y = self._spin(-1.0, 1.0, 0.000, 0.005)
        self.auto_place_z = self._spin(-0.500, 1.0, 0.150, 0.005)
        self.place_list = QListWidget()
        self.place_list.currentRowChanged.connect(self.on_place_selected)
        self.place_index_label = QLabel("Next place: 0")
        self.home_label = QLabel("Home: not set")
        self.click_z_offset = self._spin(-0.300, 0.300, 0.000, 0.001)
        self.click_z_offset.valueChanged.connect(self.update_clicked_robot_offset)
        self.click_label = QLabel("Clicked target: -")

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
        buttons.addWidget(self.auto_pick)

        pick_tab = QWidget()
        pick_layout = QVBoxLayout(pick_tab)
        pick_layout.addWidget(QLabel("Detections from /elp/detections"))
        pick_layout.addWidget(self.list, 1)
        pick_layout.addLayout(form)
        pick_layout.addLayout(buttons)

        manual_tab = self.build_manual_tab()
        camera_tab = self.build_camera_pick_tab()
        auto_tab = self.build_auto_tab()

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
        self.log_received.connect(self.append_log)
        self.pose_received.connect(self.on_pose_received)
        self.place_points_changed.connect(self.refresh_place_points)
        self.load_robot_config()
        self.load_hand_eye()
        self.start_ros()

    def _spin(self, lo, hi, value, step):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(4)
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
                "z_offset_m": self.click_z_offset.value(),
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
            self.click_z_offset.setValue(float(camera_pick.get("z_offset_m", self.click_z_offset.value())))

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
        form.addRow(self.home_label)
        form.addRow(home_row)

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
        bringup_row = QHBoxLayout()
        btn_launch = QPushButton("Launch Bringup")
        btn_stop = QPushButton("Stop Bringup")
        btn_check = QPushButton("Check Services")
        btn_launch.clicked.connect(self.launch_bringup)
        btn_stop.clicked.connect(self.stop_bringup)
        btn_check.clicked.connect(self.check_services)
        bringup_row.addWidget(QLabel("IP"))
        bringup_row.addWidget(self.robot_ip)
        bringup_row.addWidget(btn_launch)
        bringup_row.addWidget(btn_stop)
        bringup_row.addWidget(btn_check)
        state_layout.addLayout(bringup_row)

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
        layout = QVBoxLayout(tab)

        self.pick_image = ImagePanel("Camera Pick - Rectified Left")
        self.pick_image.clicked.connect(self.on_camera_pick_click)
        layout.addWidget(self.pick_image, 1)

        form = QFormLayout()
        form.addRow("Robot Z offset (m)", self.click_z_offset)
        layout.addLayout(form)
        layout.addWidget(self.click_label)

        row = QHBoxLayout()
        btn_start = QPushButton("Start Camera")
        btn_stop = QPushButton("Stop Camera")
        btn_reload = QPushButton("Reload Calib + Hand-Eye")
        btn_move = QPushButton("Move To Click")
        btn_pick = QPushButton("Pick Clicked")
        btn_start.clicked.connect(self.start_camera_pick)
        btn_stop.clicked.connect(self.stop_camera_pick)
        btn_reload.clicked.connect(self.reload_camera_pick_calibration)
        btn_move.clicked.connect(self.move_to_clicked_target)
        btn_pick.clicked.connect(self.pick_clicked_target)
        row.addWidget(btn_start)
        row.addWidget(btn_stop)
        row.addWidget(btn_reload)
        row.addWidget(btn_move)
        row.addWidget(btn_pick)
        layout.addLayout(row)

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
        left = result["left"].copy()
        if self.clicked_robot is not None:
            xy = self.clicked_robot.get("uv")
            if xy is not None:
                import cv2
                cv2.drawMarker(left, tuple(xy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        self.pick_image.show_image(left)

    def on_camera_pick_click(self, x, y):
        if self.cam_calib is None or self.cam_depth_engine is None or self.latest_depth is None:
            QMessageBox.information(self, "Camera Pick", "Need live calibrated depth first.")
            return
        if self.hand_eye_T is None:
            QMessageBox.warning(self, "Camera Pick", "Load config/hand_eye.yaml first.")
            return
        info = self.cam_depth_engine.pixel_info_from_map(self.latest_depth, x, y)
        if not info.valid:
            self.click_label.setText("Clicked target: no valid depth")
            self.clicked_robot = None
            return
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
        }
        self.update_clicked_robot_offset()

    def update_clicked_robot_offset(self):
        if self.clicked_robot is None or "robot_base_m" not in self.clicked_robot:
            return
        bx, by, bz = self.clicked_robot["robot_base_m"]
        robot_xyz = (float(bx), float(by), float(bz) + self.click_z_offset.value())
        self.clicked_robot["robot_m"] = robot_xyz
        uv = self.clicked_robot.get("uv", ("-", "-"))
        depth = self.clicked_robot.get("depth_mm", float("nan"))
        std = self.clicked_robot.get("std_mm", float("nan"))
        self.click_label.setText(
            "Clicked target: "
            f"uv=({uv[0]},{uv[1]}) depth={depth:.1f} +/-{std:.1f} mm  "
            f"baseZ={bz:+.4f} offset={self.click_z_offset.value():+.4f}  "
            f"robot=({robot_xyz[0]:+.4f}, {robot_xyz[1]:+.4f}, {robot_xyz[2]:+.4f}) m"
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
        r = self.r_deg.value()
        threading.Thread(target=self.manual_move_sequence, args=(x, y, z, r, False), daemon=True).start()

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
        threading.Thread(
            target=self.pick_sequence,
            args=(self.clicked_robot["robot_m"], params),
            daemon=True,
        ).start()

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
            return
        self.append_log("Started: " + " ".join(cmd))
        threading.Thread(target=self.drain_bringup_output, daemon=True).start()
        if self.node is None:
            QTimer.singleShot(3000, self.start_ros)
        QTimer.singleShot(5000, self.check_services)

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
        self.append_log("Stopped MG400 bringup process.")

    def check_services(self):
        if self.node is None:
            QMessageBox.warning(self, "ROS2", "ROS2 node is not running.")
            return
        names = sorted(name for name, _types in self.node.get_service_names_and_types())
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

    def manual_move_sequence(self, x, y, z, r, linear):
        self.busy = True
        try:
            self._move(x, y, z, r, linear=linear)
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
        close_state = 1 if self.close_high.isChecked() else 0
        state = (0 if close_state == 1 else 1) if open_gripper else close_state
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

        close_state = 1 if self.close_high.isChecked() else 0
        return {
            "approach_z": self.approach_z.value(),
            "pick_z_offset": self.pick_z_offset.value(),
            "place": place,
            "place_i": place_i,
            "r": self.r_deg.value(),
            "use_tool": self.do_type.currentText() == "Tool DO",
            "idx": self.do_index.value(),
            "close_state": close_state,
            "open_state": 0 if close_state == 1 else 1,
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
        try:
            self._log(f"Pick start ({px:.4f}, {py:.4f}, {pz:.4f})")
            if not self._set_do(use_tool, idx, open_state):
                return
            if not self._move(px, py, approach_z, r):
                return
            if not self._move(px, py, pz, r):
                return
            if not self._set_do(use_tool, idx, close_state):
                return
            time.sleep(0.2)
            if not self._move(px, py, approach_z, r):
                return
            if not self._move(place[0], place[1], place[2], r):
                return
            self._set_do(use_tool, idx, open_state)
            self._log("Pick/place complete")
            if params["place_i"] is not None:
                self.place_index = (params["place_i"] + 1) % len(self.place_points)
                self.log_received.emit(f"Next place index: {self.place_index}")
                self.place_points_changed.emit()
        finally:
            self.busy = False

    def _move(self, x, y, z, r, linear=False):
        evt = threading.Event()
        out = {"ok": False, "msg": "timeout"}

        def done(ok, msg):
            out["ok"] = bool(ok)
            out["msg"] = msg
            evt.set()

        self._log(f"{'MovL' if linear else 'MovJ'} ({x:.4f}, {y:.4f}, {z:.4f})")
        self.node.move_cartesian_async(x, y, z, r, is_linear=linear, on_done=done)
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
