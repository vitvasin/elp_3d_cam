"""Main application window: 3-panel live view + calibration/depth tabs."""

import os

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAction, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..calibration import load_yaml
from ..camera import CaptureThread, StereoCamera
from ..config import project_root
from ..depth import DepthEngine, Rectifier
from .calib_widget import CalibWidget
from .depth_widget import DepthWidget
from .widgets import ImagePanel


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("ELP 3D Stereo Camera Tool")
        self.resize(1400, 800)

        self.camera = None
        self.capture_thread = None
        self.rectifier = None
        self.depth_engine = None
        self.latest_raw = None        # (left, right) most recent grabbed pair
        self.latest_rect = None       # (left_rect, right_rect)
        self._depth_click = None      # (x, y) crosshair on the depth panel

        self._build_ui()
        self._try_autoload_calibration()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.left_panel = ImagePanel("Left")
        self.right_panel = ImagePanel("Right")
        self.depth_panel = ImagePanel("Depth")
        self.depth_panel.clicked.connect(self.on_depth_click)

        panels = QHBoxLayout()
        for p in (self.left_panel, self.right_panel, self.depth_panel):
            panels.addWidget(p, 1)
        panels_box = QWidget()
        panels_box.setLayout(panels)

        self.tabs = QTabWidget()
        self.calib_widget = CalibWidget(self)
        self.depth_widget = DepthWidget(self)
        self.tabs.addTab(self.calib_widget, "Calibration")
        self.tabs.addTab(self.depth_widget, "Depth")
        self.tabs.setMaximumWidth(380)

        body = QHBoxLayout()
        body.addWidget(panels_box, 1)
        body.addWidget(self.tabs)
        central = QWidget()
        central.setLayout(body)
        self.setCentralWidget(central)

        toolbar = self.addToolBar("Main")
        self.act_start = QAction("Start", self)
        self.act_start.triggered.connect(self.toggle_camera)
        toolbar.addAction(self.act_start)
        act_load = QAction("Load Calibration...", self)
        act_load.triggered.connect(self.load_calibration_dialog)
        toolbar.addAction(act_load)

        self.status = QLabel("Idle")
        self.statusBar().addWidget(self.status)

        self.timer = QTimer(self)
        self.timer.setInterval(33)  # ~30 Hz processing cap
        self.timer.timeout.connect(self.process_tick)

    # -------------------------------------------------------------- camera
    def toggle_camera(self):
        if self.capture_thread is None:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        try:
            self.camera = StereoCamera(self.cfg)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Camera error: {exc}")
            return
        self.capture_thread = CaptureThread(self.camera)
        self.capture_thread.frames_ready.connect(self.on_frames_ready)
        self.capture_thread.error.connect(self.on_capture_error)
        self.capture_thread.start()
        self.timer.start()
        self.act_start.setText("Stop")
        msg = f"Capturing ({self.camera.mode})"
        if self.camera.unsynced:
            msg += "  -  WARNING: frames may be unsynchronized"
        self.status.setText(msg)

    def stop_camera(self):
        self.timer.stop()
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread = None
        self.camera = None
        self.latest_raw = None
        self.act_start.setText("Start")
        self.status.setText("Idle")

    def on_frames_ready(self, left, right):
        self.latest_raw = (left, right)

    def on_capture_error(self, msg):
        self.status.setText(f"Capture: {msg}")

    # ------------------------------------------------------------ pipeline
    def process_tick(self):
        if self.latest_raw is None:
            return
        left, right = self.latest_raw

        if self.rectifier is not None:
            lr, rr = self.rectifier.rectify(left, right)
            self.latest_rect = (lr, rr)
            self.left_panel.show_image(lr)
            self.right_panel.show_image(rr)
            self.depth_engine.compute(lr, rr)
            color = self.depth_engine.colorized()
            if color is not None:
                if self._depth_click is not None:
                    x, y = self._depth_click
                    cv2.drawMarker(color, (x, y), (255, 255, 255),
                                   cv2.MARKER_CROSS, 16, 1)
                self.depth_panel.show_image(color)
            # Refresh the readout for the held crosshair pixel.
            if self._depth_click is not None:
                info = self.depth_engine.pixel_info(*self._depth_click)
                self.depth_widget.show_pixel_info(info)
        else:
            self.latest_rect = None
            self.left_panel.show_image(left)
            self.right_panel.show_image(right)
            self.depth_panel.setText("Load calibration\nto enable depth")

    # --------------------------------------------------------- calibration
    def _try_autoload_calibration(self):
        path = project_root() / self.cfg["calibration"]["output_path"]
        if os.path.isfile(path):
            try:
                self.apply_calibration(load_yaml(path))
                self.status.setText(f"Loaded calibration: {path}")
            except Exception as exc:  # noqa: BLE001
                self.status.setText(f"Calibration load failed: {exc}")

    def load_calibration_dialog(self):
        start_dir = str(project_root() / "config")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", start_dir, "YAML (*.yaml *.yml)"
        )
        if path:
            try:
                self.apply_calibration(load_yaml(path))
                self.status.setText(f"Loaded calibration: {path}")
            except Exception as exc:  # noqa: BLE001
                self.status.setText(f"Calibration load failed: {exc}")

    def apply_calibration(self, calib):
        """Install a calibration result: build the rectifier + depth engine."""
        self.rectifier = Rectifier(calib)
        self.depth_engine = DepthEngine(self.cfg, self.rectifier)
        self.depth_widget.set_depth_engine(self.depth_engine)

    # --------------------------------------------------------------- depth
    def on_depth_click(self, x, y):
        self._depth_click = (x, y)
        if self.depth_engine is not None:
            info = self.depth_engine.pixel_info(x, y)
            self.depth_widget.show_pixel_info(info)

    # ----------------------------------------------------------- shutdown
    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)
