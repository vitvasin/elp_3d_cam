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
from ..worker import DepthWorker
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
        self.depth_worker = None

        self.latest_raw = None        # (left, right) most recent grabbed pair
        self._latest_depth_map = None # snapshot from last worker result
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

        # Lightweight timer: only used to push raw frames when no calibration
        # is loaded. With calibration, the depth worker drives all 3 panels.
        self.timer = QTimer(self)
        self.timer.setInterval(33)
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

        if self.rectifier is not None:
            self._start_depth_worker()

        self.timer.start()
        self.act_start.setText("Stop")
        msg = f"Capturing ({self.camera.mode})"
        if self.camera.unsynced:
            msg += "  ⚠  frames may be unsynchronized"
        self.status.setText(msg)

    def stop_camera(self):
        self.timer.stop()
        self._stop_depth_worker()
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread = None
        self.camera = None
        self.latest_raw = None
        self.act_start.setText("Start")
        self.status.setText("Idle")

    def on_frames_ready(self, left, right):
        self.latest_raw = (left, right)
        if self.depth_worker is not None:
            self.depth_worker.submit(left, right)

    def on_capture_error(self, msg):
        self.status.setText(f"Capture: {msg}")

    # ---------------------------------------------------------- depth worker
    def _start_depth_worker(self):
        self._stop_depth_worker()
        self.depth_worker = DepthWorker(self.rectifier, self.depth_engine)
        self.depth_worker.result_ready.connect(self.on_depth_result)
        self.depth_worker.start()

    def _stop_depth_worker(self):
        if self.depth_worker is not None:
            self.depth_worker.stop()
            self.depth_worker = None

    def on_depth_result(self, result):
        """Called in GUI thread when a depth frame is ready."""
        self._latest_depth_map = result["depth_map"]
        self.left_panel.show_image(result["left"])
        self.right_panel.show_image(result["right"])

        color = result["color"]
        if color is not None:
            if self._depth_click is not None:
                x, y = self._depth_click
                cv2.drawMarker(color, (x, y), (255, 255, 255),
                               cv2.MARKER_CROSS, 16, 1)
            self.depth_panel.show_image(color)

        if self._depth_click is not None:
            self._update_pixel_readout(*self._depth_click)

    # ------------------------------------------------------------ raw display
    def process_tick(self):
        """Show raw (unrectified) frames when no calibration is loaded."""
        if self.rectifier is not None or self.latest_raw is None:
            return
        left, right = self.latest_raw
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
        """Install calibration: rebuild rectifier + depth engine + worker."""
        self.rectifier = Rectifier(calib)
        self.depth_engine = DepthEngine(self.cfg, self.rectifier)
        self.depth_widget.set_depth_engine(self.depth_engine)
        # Restart worker only if camera is running.
        if self.capture_thread is not None:
            self._start_depth_worker()

    # --------------------------------------------------------------- depth
    def on_depth_click(self, x, y):
        self._depth_click = (x, y)
        self._update_pixel_readout(x, y)

    def _update_pixel_readout(self, x, y):
        if self.depth_engine is None:
            return
        # Use the snapshot depth_map stored from the last worker result so
        # pixel_info never races with the worker writing a new depth_map.
        info = self.depth_engine.pixel_info_from_map(self._latest_depth_map, x, y)
        self.depth_widget.show_pixel_info(info)

    # ----------------------------------------------------------- shutdown
    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)
