#!/usr/bin/env python3
"""ELP 3D camera <-> MG400 eye-to-hand calibration tool.

Click a visible TCP/calibration point in the rectified-left image, collect the
matching robot pose from ``/mg400/get_pose``, then solve ``robot = R*camera+t``.
The result is saved to ``config/hand_eye.yaml`` and can update App 2's static TF.
"""

import sys
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app_detect import _deep_merge
from elp_stereo.calibration import load_yaml
from elp_stereo.camera import CaptureThread, StereoCamera
from elp_stereo.config import load_config, project_root
from elp_stereo.depth import DepthEngine, Rectifier, build_depth_engine
from elp_stereo.detection.depth_pick import pixel_to_camera_xyz
from elp_stereo.gui.widgets import ImagePanel
from elp_stereo.hand_eye import (
    reprojection_errors_mm,
    save_hand_eye_yaml,
    solve_rigid_transform,
    static_tf_config,
)
from elp_stereo.worker import DepthWorker

try:
    import rclpy
    from elp_stereo.ros2.mg400 import MG400Node, RosSpinThread
    ROS_OK = True
except Exception as exc:  # noqa: BLE001
    rclpy = None
    MG400Node = None
    RosSpinThread = None
    ROS_OK = False
    ROS_IMPORT_ERROR = str(exc)


MIN_POINTS = 4


def load_handeye_config():
    cfg = load_config()
    overlay_path = project_root() / "config" / "detect.yaml"
    if overlay_path.is_file():
        with open(overlay_path, "r") as f:
            overlay = yaml.safe_load(f) or {}
        _deep_merge(cfg, overlay)
    return cfg


def finite_depth_near(depth_map, x, y, radius=6):
    """Return nearest finite depth sample in mm around ``x,y``."""
    if depth_map is None:
        return None, (x, y)
    h, w = depth_map.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    for r in range(radius + 1):
        x1 = max(0, x - r)
        x2 = min(w, x + r + 1)
        y1 = max(0, y - r)
        y2 = min(h, y + r + 1)
        patch = depth_map[y1:y2, x1:x2]
        ys, xs = np.where(np.isfinite(patch))
        if xs.size:
            d2 = (xs + x1 - x) ** 2 + (ys + y1 - y) ** 2
            i = int(np.argmin(d2))
            sx = int(xs[i] + x1)
            sy = int(ys[i] + y1)
            return float(depth_map[sy, sx]), (sx, sy)
    return None, (x, y)


class HandEyeWindow(QMainWindow):
    pose_received = pyqtSignal(object, object)
    manual_pose_received = pyqtSignal(object, object)
    sequence_move_done = pyqtSignal(bool, str)
    sequence_pose_received = pyqtSignal(object, object)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("ELP 3D -> MG400 Hand-Eye Calibration")
        self.resize(1180, 760)

        self.calib = None
        self.rectifier = None
        self.depth_engine = None
        self.capture = None
        self.depth_worker = None
        self.ros_node = None
        self.ros_spin = None

        self.latest_depth = None
        self.latest_left = None
        self.click_xy = None
        self.cam_pts = []
        self.robot_pts = []
        self.last_R = None
        self.last_t = None
        self.marker_centroid = None
        self.marker_corners = None
        self.marker_tvec = None
        self.marker_stable_count = 0
        self.marker_collecting = False
        self.auto_seq_running = False
        self.auto_seq_positions = []
        self.auto_seq_idx = 0
        self.auto_seq_collected = 0
        self.seq_waiting_for_marker = False
        self.seq_collect_idx = None
        self.seq_collect_pending = False
        self.bringup_proc = None

        self.robot_ip = QLineEdit("192.168.1.6")
        self.manual_x = self._dspin(-1.0, 1.0, 0.300, 0.005)
        self.manual_y = self._dspin(-1.0, 1.0, 0.000, 0.005)
        self.manual_z = self._dspin(-0.500, 1.0, 0.150, 0.005)
        self.manual_r = self._dspin(-180.0, 180.0, 0.0, 1.0)

        self.left_panel = ImagePanel("rectified left")
        self.depth_panel = ImagePanel("depth")
        self.left_panel.clicked.connect(self.on_image_click)

        self.info = QLabel(
            "Load stereo calibration, click TCP/calibration point, collect pose. "
            "app_detect.py auto-loads config/hand_eye.yaml at startup."
        )
        self.info.setWordWrap(True)
        self.point_list = QListWidget()
        self.auto_update_detect = QCheckBox("Also rewrite config/detect.yaml static TF")
        self.auto_update_detect.setChecked(False)
        self._init_aruco_detector()

        controls = QTabWidget()
        controls.addTab(self.build_click_tab(), "Click")
        controls.addTab(self.build_aruco_tab(), "ArUco")
        controls.addTab(self.build_sequence_tab(), "Auto")

        btn_load = QPushButton("Load Stereo Calibration")
        btn_collect = QPushButton("Collect Point")
        btn_delete = QPushButton("Delete Last")
        btn_clear = QPushButton("Clear")
        btn_solve = QPushButton("Solve && Save")
        btn_load.clicked.connect(self.load_calibration_dialog)
        btn_collect.clicked.connect(self.collect_point)
        btn_delete.clicked.connect(self.delete_last)
        btn_clear.clicked.connect(self.clear_points)
        btn_solve.clicked.connect(self.solve_and_save)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(self.build_robot_control_box())
        rv.addWidget(self.info)
        rv.addWidget(self.point_list, 1)
        rv.addWidget(controls)
        rv.addWidget(self.auto_update_detect)
        for b in (btn_load, btn_collect, btn_delete, btn_clear, btn_solve):
            rv.addWidget(b)
        rv.addStretch(1)

        image_box = QWidget()
        grid = QGridLayout(image_box)
        grid.addWidget(self.left_panel, 0, 0)
        grid.addWidget(self.depth_panel, 0, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(image_box)
        split.addWidget(right)
        split.setSizes([850, 330])
        self.setCentralWidget(split)
        self.setStatusBar(QStatusBar())

        self.pose_received.connect(self.on_pose_received)
        self.manual_pose_received.connect(self.on_manual_pose_received)
        self.sequence_move_done.connect(self.on_sequence_move_done)
        self.sequence_pose_received.connect(self.on_sequence_pose_received)
        self.start_ros()
        self.start_camera()
        self.try_load_calibration()

    def _init_aruco_detector(self):
        self.aruco_dict = self.get_aruco_dictionary(cv2.aruco.DICT_4X4_50)
        params = self.create_aruco_params()
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params = params
        self.aruco_detector = (
            cv2.aruco.ArucoDetector(self.aruco_dict, params)
            if hasattr(cv2.aruco, "ArucoDetector") else None
        )

    def get_aruco_dictionary(self, dict_id):
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            return cv2.aruco.getPredefinedDictionary(dict_id)
        return cv2.aruco.Dictionary_get(dict_id)

    def create_aruco_params(self):
        if hasattr(cv2.aruco, "DetectorParameters"):
            return cv2.aruco.DetectorParameters()
        return cv2.aruco.DetectorParameters_create()

    def build_click_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        msg = QLabel("Click the TCP/calibration point in the rectified-left image, then Collect Point.")
        msg.setWordWrap(True)
        layout.addWidget(msg)
        layout.addStretch(1)
        return tab

    def build_aruco_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        form = QFormLayout()
        self.aruco_dict_combo = QComboBox()
        self.aruco_dict_combo.addItems(["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50"])
        self.aruco_dict_combo.currentTextChanged.connect(self.on_aruco_dict_changed)
        self.aruco_id = QSpinBox()
        self.aruco_id.setRange(0, 999)
        self.aruco_id.setValue(0)
        self.aruco_stable_frames = QSpinBox()
        self.aruco_stable_frames.setRange(3, 120)
        self.aruco_stable_frames.setValue(15)
        self.aruco_marker_size = QDoubleSpinBox()
        self.aruco_marker_size.setRange(5.0, 300.0)
        self.aruco_marker_size.setDecimals(1)
        self.aruco_marker_size.setSingleStep(5.0)
        self.aruco_marker_size.setValue(40.0)
        self.aruco_use_pnp = QCheckBox("Use PnP from marker size")
        self.aruco_auto_collect = QCheckBox("Auto collect when stable")
        form.addRow("Dictionary", self.aruco_dict_combo)
        form.addRow("Marker ID", self.aruco_id)
        form.addRow("Stable frames", self.aruco_stable_frames)
        form.addRow("Marker size (mm)", self.aruco_marker_size)
        form.addRow("", self.aruco_use_pnp)
        form.addRow("", self.aruco_auto_collect)
        layout.addLayout(form)

        self.aruco_status = QLabel("No marker detected")
        self.aruco_status.setWordWrap(True)
        layout.addWidget(self.aruco_status)

        btn_collect = QPushButton("Collect Current Marker")
        btn_collect.clicked.connect(self.collect_aruco_marker)
        layout.addWidget(btn_collect)
        layout.addStretch(1)
        return tab

    def build_sequence_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        msg = QLabel("Moves the robot through a grid. With ArUco auto-collect enabled, each stable marker pose is collected automatically.")
        msg.setWordWrap(True)
        layout.addWidget(msg)

        box = QGroupBox("Robot Grid")
        form = QFormLayout(box)
        self.seq_cx = self._dspin(-1.0, 1.0, 0.250, 0.005)
        self.seq_cy = self._dspin(-1.0, 1.0, 0.000, 0.005)
        self.seq_cz = self._dspin(0.0, 1.0, 0.100, 0.005)
        self.seq_step_xy = self._dspin(0.001, 0.200, 0.040, 0.005)
        self.seq_step_z = self._dspin(0.0, 0.200, 0.030, 0.005)
        self.seq_cols = QSpinBox(); self.seq_cols.setRange(1, 7); self.seq_cols.setValue(3)
        self.seq_rows = QSpinBox(); self.seq_rows.setRange(1, 7); self.seq_rows.setValue(3)
        self.seq_z_levels = QSpinBox(); self.seq_z_levels.setRange(1, 5); self.seq_z_levels.setValue(2)
        self.seq_r = self._dspin(-180.0, 180.0, 0.0, 1.0)
        self.seq_settle = self._dspin(0.1, 10.0, 1.0, 0.1)
        form.addRow("Center X (m)", self.seq_cx)
        form.addRow("Center Y (m)", self.seq_cy)
        form.addRow("Center Z (m)", self.seq_cz)
        form.addRow("Step XY (m)", self.seq_step_xy)
        form.addRow("Step Z (m)", self.seq_step_z)
        form.addRow("Cols", self.seq_cols)
        form.addRow("Rows", self.seq_rows)
        form.addRow("Z levels", self.seq_z_levels)
        form.addRow("R yaw (deg)", self.seq_r)
        form.addRow("Settle (s)", self.seq_settle)
        layout.addWidget(box)

        row = QHBoxLayout()
        btn_current = QPushButton("Use Current Pose")
        self.seq_start_btn = QPushButton("Start Auto")
        self.seq_stop_btn = QPushButton("Stop")
        self.seq_stop_btn.setEnabled(False)
        btn_current.clicked.connect(self.sequence_use_current_pose)
        self.seq_start_btn.clicked.connect(self.sequence_start)
        self.seq_stop_btn.clicked.connect(self.sequence_stop)
        row.addWidget(btn_current)
        row.addWidget(self.seq_start_btn)
        row.addWidget(self.seq_stop_btn)
        layout.addLayout(row)

        self.seq_status = QLabel("Sequence idle")
        self.seq_status.setWordWrap(True)
        layout.addWidget(self.seq_status)
        layout.addStretch(1)
        return tab

    def _dspin(self, lo, hi, value, step):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(4)
        s.setSingleStep(step)
        s.setValue(value)
        return s

    def build_robot_control_box(self):
        box = QGroupBox("Robot Control")
        layout = QVBoxLayout(box)

        bringup = QHBoxLayout()
        btn_launch = QPushButton("Launch Bringup")
        btn_stop = QPushButton("Stop Bringup")
        btn_check = QPushButton("Check Services")
        btn_launch.clicked.connect(self.launch_bringup)
        btn_stop.clicked.connect(self.stop_bringup)
        btn_check.clicked.connect(self.check_services)
        bringup.addWidget(QLabel("IP"))
        bringup.addWidget(self.robot_ip)
        bringup.addWidget(btn_launch)
        bringup.addWidget(btn_stop)
        bringup.addWidget(btn_check)
        layout.addLayout(bringup)

        state = QHBoxLayout()
        btn_clear = QPushButton("Clear Error")
        btn_enable = QPushButton("Enable")
        btn_disable = QPushButton("Disable")
        btn_clear.clicked.connect(lambda: self.robot_state_command("clear"))
        btn_enable.clicked.connect(lambda: self.robot_state_command("enable"))
        btn_disable.clicked.connect(lambda: self.robot_state_command("disable"))
        state.addWidget(btn_clear)
        state.addWidget(btn_enable)
        state.addWidget(btn_disable)
        layout.addLayout(state)

        form = QFormLayout()
        form.addRow("X (m)", self.manual_x)
        form.addRow("Y (m)", self.manual_y)
        form.addRow("Z (m)", self.manual_z)
        form.addRow("R yaw (deg)", self.manual_r)
        layout.addLayout(form)

        move = QHBoxLayout()
        btn_read = QPushButton("Read Pose")
        btn_movej = QPushButton("MoveJ")
        btn_read.clicked.connect(self.read_manual_pose)
        btn_movej.clicked.connect(self.manual_movej)
        move.addWidget(btn_read)
        move.addWidget(btn_movej)
        layout.addLayout(move)
        return box

    def launch_bringup(self):
        if self.bringup_proc and self.bringup_proc.poll() is None:
            self.statusBar().showMessage("MG400 bringup already running from this app.")
            return
        ip = self.robot_ip.text().strip() or "192.168.1.6"
        cmd = [
            "ros2", "launch", "mg400_bringup", "mg400_gui.launch.py",
            f"ip_address:={ip}",
        ]
        try:
            self.bringup_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Launch bringup failed: {exc}")
            return
        self.statusBar().showMessage("Started MG400 bringup.")
        if self.ros_node is None:
            QTimer.singleShot(3000, self.start_ros)
        QTimer.singleShot(5000, self.check_services)

    def stop_bringup(self, silent=False):
        if not self.bringup_proc or self.bringup_proc.poll() is not None:
            if not silent:
                self.statusBar().showMessage("No bringup process started by this app.")
            return
        self.bringup_proc.terminate()
        try:
            self.bringup_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.bringup_proc.kill()
        self.statusBar().showMessage("Stopped MG400 bringup process.")

    def check_services(self):
        if self.ros_node is None:
            QMessageBox.warning(self, "ROS2", "ROS2 node is not running.")
            return
        names = sorted(name for name, _types in self.ros_node.get_service_names_and_types())
        needed = [
            "/mg400/clear_error",
            "/mg400/enable_robot",
            "/mg400/disable_robot",
            "/mg400/get_pose",
        ]
        msg = "  ".join(f"{name}: {'OK' if name in names else 'MISSING'}" for name in needed)
        self.statusBar().showMessage(msg)

    def robot_state_command(self, command):
        if self.ros_node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return

        def done(ok, msg):
            self.statusBar().showMessage(f"{command}: {'ok' if ok else msg}")

        if command == "clear":
            self.ros_node.clear_error_async(done)
        elif command == "enable":
            self.ros_node.enable_robot_async(done)
        elif command == "disable":
            self.ros_node.disable_robot_async(done)

    def read_manual_pose(self):
        if self.ros_node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        self.ros_node.request_pose_async(
            lambda pose, err: self.manual_pose_received.emit(pose, err)
        )

    def on_manual_pose_received(self, pose, err):
        if pose is None:
            self.statusBar().showMessage(f"Read pose failed: {err}")
            return
        x, y, z = pose
        self.manual_x.setValue(float(x))
        self.manual_y.setValue(float(y))
        self.manual_z.setValue(float(z))
        self.statusBar().showMessage(f"Pose read ({x:.4f}, {y:.4f}, {z:.4f})")

    def manual_movej(self):
        if self.ros_node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        self.ros_node.move_cartesian_async(
            self.manual_x.value(),
            self.manual_y.value(),
            self.manual_z.value(),
            self.manual_r.value(),
            is_linear=False,
            on_done=lambda ok, msg: self.statusBar().showMessage(f"MoveJ: {'ok' if ok else msg}"),
        )

    def start_ros(self):
        if not ROS_OK:
            self.statusBar().showMessage(f"ROS2 unavailable: {ROS_IMPORT_ERROR}")
            return
        try:
            if not rclpy.ok():
                rclpy.init(args=None)
            self.ros_node = MG400Node("elp_handeye_calibrator")
            self.ros_spin = RosSpinThread(self.ros_node)
            self.ros_spin.start()
        except Exception as exc:  # noqa: BLE001
            self.ros_node = None
            self.ros_spin = None
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
            self.statusBar().showMessage(f"ROS2 startup failed: {exc}")

    def start_camera(self):
        try:
            camera = StereoCamera(self.cfg)
        except Exception as exc:  # noqa: BLE001
            self.capture = None
            self.statusBar().showMessage(f"Camera startup failed: {exc}")
            return
        self.capture = CaptureThread(camera, self)
        self.capture.frames_ready.connect(self.on_frames)
        self.capture.error.connect(lambda e: self.statusBar().showMessage(e))
        self.capture.start()

    def restart_camera(self):
        if self.depth_worker:
            self.depth_worker.stop()
            self.depth_worker = None
        if self.capture:
            self.capture.stop()
            self.capture = None
        self.start_camera()

    def try_load_calibration(self):
        path = self.cfg.get("calibration_path") or self.cfg["calibration"]["output_path"]
        full = project_root() / path
        if full.is_file():
            try:
                self.apply_calibration(load_yaml(full))
                self.statusBar().showMessage(f"Loaded stereo calibration: {full}")
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"Calibration load failed: {exc}")

    def load_calibration_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load stereo calibration", str(project_root() / "config"), "YAML (*.yaml *.yml)"
        )
        if path:
            try:
                self.apply_calibration(load_yaml(path))
                self.statusBar().showMessage(f"Loaded stereo calibration: {path}")
            except Exception as exc:  # noqa: BLE001
                self.statusBar().showMessage(f"Calibration load failed: {exc}")

    def apply_calibration(self, calib):
        expected = (
            int(self.cfg["camera"]["frame_width"]) // 2,
            int(self.cfg["camera"]["frame_height"]),
        )
        if tuple(calib.image_size) != expected:
            w, h = calib.image_size
            self.cfg["camera"]["frame_width"] = int(w) * 2
            self.cfg["camera"]["frame_height"] = int(h)
            self.statusBar().showMessage(
                f"Camera resolution changed to {w}x{h} per eye for calibration."
            )
            self.restart_camera()
        self.calib = calib
        self.rectifier = Rectifier(calib)
        if self.capture is None:
            self.depth_engine = None
            self.depth_worker = None
            return
        try:
            self.depth_engine = build_depth_engine(self.cfg, self.rectifier)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Depth engine fallback to SGBM: {exc}")
            self.depth_engine = DepthEngine(self.cfg, self.rectifier)
        if self.capture is not None:
            if self.depth_worker:
                self.depth_worker.stop()
            self.depth_worker = DepthWorker(self.rectifier, self.depth_engine, self)
            self.depth_worker.result_ready.connect(self.on_depth_result)
            self.depth_worker.start()

    def on_frames(self, left, right):
        if self.depth_worker:
            self.depth_worker.submit(left, right)
        else:
            self.left_panel.show_image(left)

    def on_depth_result(self, result):
        if "error" in result:
            self.statusBar().showMessage(result["error"])
            return
        self.latest_left = result["left"]
        self.latest_depth = result["depth_map"]
        left_vis = result["left"].copy()
        self.update_aruco(left_vis)
        self.left_panel.show_image(left_vis)
        self.depth_panel.show_image(result["color"])

    def on_image_click(self, x, y):
        self.click_xy = (int(x), int(y))
        z, sample = finite_depth_near(self.latest_depth, x, y)
        if z is None:
            self.info.setText(f"Clicked ({x}, {y}); no finite depth nearby.")
            return
        xyz_mm = pixel_to_camera_xyz(sample[0], sample[1], z, self.calib.P1)
        self.info.setText(
            f"Clicked ({x}, {y}); depth sample ({sample[0]}, {sample[1]}) "
            f"camera=({xyz_mm[0]/1000:.4f}, {xyz_mm[1]/1000:.4f}, {xyz_mm[2]/1000:.4f}) m"
        )

    def on_aruco_dict_changed(self, name):
        mapping = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        }
        self.aruco_dict = self.get_aruco_dictionary(mapping.get(name, cv2.aruco.DICT_4X4_50))
        self.aruco_detector = (
            cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            if hasattr(cv2.aruco, "ArucoDetector") else None
        )
        self.marker_stable_count = 0
        self.marker_centroid = None

    def detect_aruco(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None:
            self.marker_corners = None
            self.marker_tvec = None
            return None
        target_id = int(self.aruco_id.value())
        for i, marker_id in enumerate(ids.flatten()):
            if int(marker_id) != target_id:
                continue
            c = corners[i][0].astype(np.float64)
            cx, cy = c.mean(axis=0)
            self.marker_corners = c.copy()
            cv2.aruco.drawDetectedMarkers(bgr, [corners[i]], ids[i:i + 1])
            self.marker_tvec = self.solve_aruco_pnp(c) if self.aruco_use_pnp.isChecked() else None
            if self.marker_tvec is not None:
                cv2.putText(
                    bgr,
                    f"PnP z={self.marker_tvec[2] * 1000:.0f}mm",
                    (int(cx) + 14, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 200, 0),
                    1,
                )
            return (float(cx), float(cy))
        self.marker_corners = None
        self.marker_tvec = None
        return None

    def solve_aruco_pnp(self, corners_2d):
        if self.calib is None:
            return None
        half = self.aruco_marker_size.value() / 2000.0
        obj = np.array([
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)
        K = np.array([
            [float(self.calib.P1[0, 0]), 0.0, float(self.calib.P1[0, 2])],
            [0.0, float(self.calib.P1[1, 1]), float(self.calib.P1[1, 2])],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        try:
            ok, _, tvec = cv2.solvePnP(
                obj, corners_2d.astype(np.float64), K, np.zeros(5),
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error:
            return None
        return tvec.flatten() if ok else None

    def update_aruco(self, bgr):
        centroid = self.detect_aruco(bgr)
        if centroid is None:
            self.marker_centroid = None
            self.marker_stable_count = 0
            self.aruco_status.setText("No marker detected")
            return

        cx, cy = centroid
        threshold = int(self.aruco_stable_frames.value())
        if self.marker_centroid is not None:
            dx = abs(cx - self.marker_centroid[0])
            dy = abs(cy - self.marker_centroid[1])
            self.marker_stable_count = self.marker_stable_count + 1 if dx <= 2.0 and dy <= 2.0 else 1
        else:
            self.marker_stable_count = 1
        self.marker_stable_count = min(self.marker_stable_count, threshold)
        self.marker_centroid = centroid

        frac = self.marker_stable_count / max(threshold, 1)
        color = (0, int(255 * frac), int(255 * (1.0 - frac)))
        cv2.circle(bgr, (int(round(cx)), int(round(cy))), 12, color, 2)
        cv2.putText(
            bgr,
            f"{self.marker_stable_count}/{threshold}",
            (int(cx) + 14, int(cy) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        mode = "PnP" if self.aruco_use_pnp.isChecked() else "depth"
        self.aruco_status.setText(f"Marker stable {self.marker_stable_count}/{threshold}, mode={mode}")

        if (
            self.marker_stable_count >= threshold
            and self.aruco_auto_collect.isChecked()
            and not self.marker_collecting
            and (not self.auto_seq_running or self.seq_waiting_for_marker)
        ):
            if self.auto_seq_running:
                self.seq_waiting_for_marker = False
                self.seq_collect_idx = self.auto_seq_idx
                self.seq_collect_pending = True
                self.seq_status.setText(
                    f"Collecting {self.auto_seq_idx + 1}/{len(self.auto_seq_positions)} ..."
                )
            self.marker_collecting = True
            QTimer.singleShot(0, self.collect_aruco_marker)

    def _auto_collect_retry(self, msg):
        self.marker_collecting = False
        self.seq_collect_pending = False
        self.seq_collect_idx = None
        if self.auto_seq_running:
            self.seq_waiting_for_marker = True
            self.seq_status.setText(msg)
            return True
        return False

    def collect_aruco_marker(self):
        if self.marker_collecting and not self.aruco_auto_collect.isChecked():
            return
        if self.marker_centroid is None:
            if self._auto_collect_retry("No marker detected; waiting for stable ArUco."):
                return
            QMessageBox.warning(self, "ArUco", "No target marker detected.")
            self.marker_collecting = False
            return
        if self.aruco_use_pnp.isChecked():
            if self.marker_tvec is None:
                if self._auto_collect_retry("PnP pose unavailable; waiting for stable ArUco."):
                    return
                QMessageBox.warning(self, "ArUco", "PnP is enabled but marker pose is unavailable.")
                self.marker_collecting = False
                return
            self.collect_camera_point(tuple(float(v) for v in self.marker_tvec), "ArUco PnP")
            return

        if self.calib is None or self.latest_depth is None:
            if self._auto_collect_retry("Need stereo calibration and depth frame; waiting."):
                return
            QMessageBox.warning(self, "ArUco", "Need stereo calibration and depth frame.")
            self.marker_collecting = False
            return
        cx, cy = self.marker_centroid
        z, sample = finite_depth_near(self.latest_depth, int(round(cx)), int(round(cy)), radius=10)
        if z is None:
            if self._auto_collect_retry("No finite depth near marker; waiting for stable depth."):
                return
            QMessageBox.warning(self, "ArUco", "No finite depth near marker centroid.")
            self.marker_collecting = False
            return
        xyz_mm = pixel_to_camera_xyz(sample[0], sample[1], z, self.calib.P1)
        cam_m = tuple(float(v) / 1000.0 for v in xyz_mm)
        self.collect_camera_point(cam_m, f"ArUco depth pixel {sample}")

    def collect_point(self):
        if self.ros_node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        if self.calib is None or self.latest_depth is None or self.click_xy is None:
            QMessageBox.warning(self, "Collect", "Need stereo calibration, depth frame, and image click.")
            return
        x, y = self.click_xy
        z, sample = finite_depth_near(self.latest_depth, x, y)
        if z is None:
            QMessageBox.warning(self, "Collect", "No finite depth near the clicked point.")
            return
        xyz_mm = pixel_to_camera_xyz(sample[0], sample[1], z, self.calib.P1)
        cam_m = tuple(float(v) / 1000.0 for v in xyz_mm)
        self.statusBar().showMessage("Reading /mg400/get_pose ...")
        self.ros_node.request_pose_async(
            lambda pose, err: self.pose_received.emit((cam_m, f"depth pixel {sample}"), (pose, err))
        )

    def collect_camera_point(self, cam_m, source):
        if self.ros_node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        self.statusBar().showMessage("Reading /mg400/get_pose ...")
        seq_idx = self.seq_collect_idx if self.auto_seq_running else None
        self.ros_node.request_pose_async(
            lambda pose, err, seq_idx=seq_idx: self.pose_received.emit(
                (tuple(cam_m), source, seq_idx), (pose, err)
            )
        )

    def on_pose_received(self, cam_payload, pose_payload):
        if len(cam_payload) == 3:
            cam_m, source, seq_idx = cam_payload
        else:
            cam_m, source = cam_payload
            seq_idx = None
        pose, err = pose_payload
        self.marker_collecting = False
        if self.auto_seq_running:
            self.seq_collect_pending = False
            self.seq_waiting_for_marker = False
            if seq_idx != self.auto_seq_idx:
                self.seq_collect_idx = None
                self.statusBar().showMessage(
                    f"Ignored stale pose callback for sequence point {seq_idx}; "
                    f"current is {self.auto_seq_idx}."
                )
                return
            self.seq_collect_idx = None
        if pose is None:
            self.statusBar().showMessage(f"Pose read failed: {err}")
            if self.auto_seq_running:
                QTimer.singleShot(500, self.sequence_advance)
            return
        self.cam_pts.append(cam_m)
        self.robot_pts.append(tuple(pose))
        self.refresh_list()
        if self.auto_seq_running:
            self.auto_seq_collected += 1
        self.statusBar().showMessage(
            f"Collected {len(self.cam_pts)} points from {source}; need >= {MIN_POINTS}."
        )
        if self.auto_seq_running:
            QTimer.singleShot(300, self.sequence_advance)

    def refresh_list(self, errors=None):
        self.point_list.clear()
        for i, (cp, rp) in enumerate(zip(self.cam_pts, self.robot_pts), start=1):
            err = "" if errors is None else f"  err={errors[i-1]:.1f}mm"
            self.point_list.addItem(
                f"{i:02d} cam=({cp[0]:+.4f},{cp[1]:+.4f},{cp[2]:+.4f})  "
                f"robot=({rp[0]:+.4f},{rp[1]:+.4f},{rp[2]:+.4f}){err}"
            )

    def delete_last(self):
        if self.cam_pts:
            self.cam_pts.pop()
            self.robot_pts.pop()
            self.refresh_list()

    def clear_points(self):
        self.cam_pts.clear()
        self.robot_pts.clear()
        self.refresh_list()

    def solve_and_save(self):
        if len(self.cam_pts) < MIN_POINTS:
            QMessageBox.warning(self, "Solve", f"Need at least {MIN_POINTS} points.")
            return
        R, t = solve_rigid_transform(self.cam_pts, self.robot_pts)
        errors = reprojection_errors_mm(self.cam_pts, self.robot_pts, R, t)
        mean_e = float(np.mean(errors))
        max_e = float(np.max(errors))
        out = project_root() / "config" / "hand_eye.yaml"
        data = save_hand_eye_yaml(out, R, t, self.cam_pts, self.robot_pts, mean_e, max_e)
        self.last_R = R
        self.last_t = t
        self.refresh_list(errors)
        if self.auto_update_detect.isChecked():
            self.update_detect_yaml(data)
        msg = f"Saved {out}: mean={mean_e:.1f}mm max={max_e:.1f}mm"
        self.statusBar().showMessage(msg)
        QMessageBox.information(self, "Hand-Eye Saved", msg)

    def update_detect_yaml(self, hand_eye_data):
        path = project_root() / "config" / "detect.yaml"
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        ros_cfg = cfg.setdefault("ros2", {})
        ros_cfg["camera_frame"] = hand_eye_data.get("child_frame", "camera_optical_frame")
        ros_cfg["robot_frame"] = hand_eye_data.get("parent_frame", "robot_base")
        ros_cfg["use_tf_lookup"] = True
        ros_cfg["static_tf"] = static_tf_config(hand_eye_data)
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def sequence_use_current_pose(self):
        if self.ros_node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        self.seq_status.setText("Reading current robot pose ...")
        self.ros_node.request_pose_async(
            lambda pose, err: self.sequence_pose_received.emit(pose, err)
        )

    def on_sequence_pose_received(self, pose, err):
        if pose is None:
            msg = f"Current pose read failed: {err}"
            self.seq_status.setText(msg)
            self.statusBar().showMessage(msg)
            return
        x, y, z = pose
        self.set_sequence_center(x, y, z)

    def set_sequence_center(self, x, y, z):
        self.seq_cx.setValue(float(x))
        self.seq_cy.setValue(float(y))
        self.seq_cz.setValue(float(z))
        self.seq_status.setText(f"Center set to ({x:.4f}, {y:.4f}, {z:.4f}) m")

    def sequence_positions(self):
        cx = self.seq_cx.value()
        cy = self.seq_cy.value()
        cz = self.seq_cz.value()
        step_xy = self.seq_step_xy.value()
        step_z = self.seq_step_z.value()
        cols = self.seq_cols.value()
        rows = self.seq_rows.value()
        levels = self.seq_z_levels.value()
        r = self.seq_r.value()
        z_offsets = [step_z * (i - (levels - 1) / 2.0) for i in range(levels)]
        out = []
        for dz in z_offsets:
            for row in range(rows):
                col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
                for col in col_range:
                    x = cx + step_xy * (col - (cols - 1) / 2.0)
                    y = cy + step_xy * (row - (rows - 1) / 2.0)
                    out.append((x, y, cz + dz, r))
        return out

    def sequence_start(self):
        if self.ros_node is None:
            QMessageBox.warning(self, "ROS2", "ROS2/MG400 messages are not available.")
            return
        if self.auto_seq_running:
            return
        if not self.aruco_auto_collect.isChecked():
            self.aruco_auto_collect.setChecked(True)
        self.auto_seq_positions = self.sequence_positions()
        if not self.auto_seq_positions:
            return
        self.auto_seq_idx = 0
        self.auto_seq_collected = 0
        self.auto_seq_running = True
        self.seq_waiting_for_marker = False
        self.seq_collect_idx = None
        self.seq_collect_pending = False
        self.seq_start_btn.setEnabled(False)
        self.seq_stop_btn.setEnabled(True)
        self.sequence_move_next()

    def sequence_stop(self):
        self.auto_seq_running = False
        self.seq_waiting_for_marker = False
        self.seq_collect_idx = None
        self.seq_collect_pending = False
        self.marker_collecting = False
        self.aruco_auto_collect.setChecked(False)
        self.seq_start_btn.setEnabled(True)
        self.seq_stop_btn.setEnabled(False)
        self.seq_status.setText(
            f"Stopped at {self.auto_seq_idx}/{len(self.auto_seq_positions)}; "
            f"collected {self.auto_seq_collected}."
        )

    def sequence_move_next(self):
        if not self.auto_seq_running:
            return
        if self.auto_seq_idx >= len(self.auto_seq_positions):
            self.auto_seq_running = False
            self.seq_waiting_for_marker = False
            self.seq_collect_idx = None
            self.seq_collect_pending = False
            self.marker_collecting = False
            self.aruco_auto_collect.setChecked(False)
            self.seq_start_btn.setEnabled(True)
            self.seq_stop_btn.setEnabled(False)
            self.seq_status.setText(f"Done; collected {self.auto_seq_collected}. Press Solve && Save.")
            self.statusBar().showMessage("Auto calibration sequence complete.")
            return

        x, y, z, r = self.auto_seq_positions[self.auto_seq_idx]
        n = len(self.auto_seq_positions)
        self.marker_stable_count = 0
        self.marker_centroid = None
        self.marker_collecting = False
        self.seq_waiting_for_marker = False
        self.seq_collect_idx = None
        self.seq_collect_pending = False
        self.seq_status.setText(
            f"Moving {self.auto_seq_idx + 1}/{n}: ({x:.4f}, {y:.4f}, {z:.4f}) m"
        )
        self.ros_node.move_cartesian_async(
            x, y, z, r, is_linear=True,
            on_done=lambda ok, msg: self.sequence_move_done.emit(bool(ok), str(msg)),
        )

    def on_sequence_move_done(self, ok, msg):
        if not self.auto_seq_running:
            return
        if not ok:
            self.seq_status.setText(f"Move failed: {msg}; skipping.")
            QTimer.singleShot(500, self.sequence_advance)
            return
        wait_ms = int(self.seq_settle.value() * 1000)
        self.seq_status.setText(
            f"Settling {self.auto_seq_idx + 1}/{len(self.auto_seq_positions)}; waiting for stable ArUco."
        )
        QTimer.singleShot(wait_ms, self.sequence_ready_for_marker)

    def sequence_ready_for_marker(self):
        if not self.auto_seq_running:
            return
        self.marker_stable_count = 0
        self.marker_centroid = None
        self.marker_collecting = False
        self.seq_waiting_for_marker = True
        self.seq_collect_idx = None
        self.seq_collect_pending = False
        self.seq_status.setText(
            f"At {self.auto_seq_idx + 1}/{len(self.auto_seq_positions)}; hold marker stable."
        )

    def sequence_advance(self):
        if not self.auto_seq_running:
            return
        self.auto_seq_idx += 1
        self.sequence_move_next()

    def closeEvent(self, event):
        self.stop_bringup(silent=True)
        if self.capture:
            self.capture.stop()
        if self.depth_worker:
            self.depth_worker.stop()
        if self.ros_spin:
            self.ros_spin.stop()
        if self.ros_node:
            self.ros_node.destroy_node()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = HandEyeWindow(load_handeye_config())
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
