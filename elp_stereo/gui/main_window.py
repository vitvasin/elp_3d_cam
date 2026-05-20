"""Main application window: 3-panel live view + calibration/depth tabs."""

import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAction, QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QTabWidget, QVBoxLayout, QWidget,
)

# Per-eye resolution presets shown in the toolbar.
# Stored as (label, per_eye_width, per_eye_height).
# Full SBS frame sent to the camera = (2*w, h).
_RESOLUTIONS = [
    ("1920 × 1080", 1920, 1080),
    ("1280 × 720",  1280, 720),
    ("960 × 540",    960, 540),
    ("640 × 480",    640, 480),
    ("640 × 360",    640, 360),
    ("320 × 240",    320, 240),
]

_FPS_PRESETS = [5, 10, 15, 20, 25, 30]

# (label, show_left, show_right, show_depth)
_VIEW_MODES = [
    ("All",   True,  True,  True),
    ("Left",  True,  False, False),
    ("Right", False, True,  False),
    ("Both",  True,  True,  False),
    ("Depth", False, False, True),
    ("Raw",   True,  True,  True),   # unrectified feed; bypasses depth worker for L/R
]

from ..calibration import load_yaml
from ..camera import CaptureThread, StereoCamera
from ..config import project_root, save_config
from ..depth import Rectifier, build_depth_engine
from ..worker import DepthWorker
from .calib_widget import CalibWidget
from .depth_widget import DepthWidget
from .widgets import ImagePanel


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("ELP 3D Stereo Camera Tool")
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.resize(min(1400, int(geo.width() * 0.92)),
                        min(800, int(geo.height() * 0.92)))
        else:
            self.resize(1200, 720)

        self.camera = None
        self.capture_thread = None
        self.rectifier = None
        self.depth_engine = None
        self.depth_worker = None
        self._reported_capture_format = False

        self.latest_raw = None        # (left, right) most recent grabbed pair
        self._latest_rectified = None  # (left, right) most recent rectified pair
        self._latest_depth_map = None # snapshot from last worker result
        self._latest_disparity = None # left-to-right disparity snapshot
        self._depth_click = None      # (x, y) crosshair on the depth panel
        self._depth_label = None      # rendered text for the latest depth click
        self._stereo_click = None     # matched left/right RGB crosshairs

        self._build_ui()
        self._try_autoload_calibration()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.left_panel = ImagePanel("Left")
        self.right_panel = ImagePanel("Right")
        self.depth_panel = ImagePanel("Depth")
        self.left_panel.clicked.connect(self.on_left_click)
        self.right_panel.clicked.connect(self.on_right_click)
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
        act_save_params = QAction("Save Parameters", self)
        act_save_params.triggered.connect(self.save_parameters)
        toolbar.addAction(act_save_params)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Resolution: "))
        self.res_combo = QComboBox()
        for label, *_ in _RESOLUTIONS:
            self.res_combo.addItem(label)
        cam_cfg = self.cfg.get("camera", {})
        current = (int(cam_cfg.get("frame_width", 0)) // 2,
                   int(cam_cfg.get("frame_height", 0)))
        for i, (_, w, h) in enumerate(_RESOLUTIONS):
            if (w, h) == current:
                self.res_combo.setCurrentIndex(i)
                break
        self.res_combo.currentIndexChanged.connect(self._on_resolution_changed)
        toolbar.addWidget(self.res_combo)

        toolbar.addWidget(QLabel(" FPS: "))
        self.fps_combo = QComboBox()
        for fps in _FPS_PRESETS:
            self.fps_combo.addItem(str(fps), fps)
        current_fps = int(cam_cfg.get("fps", 30))
        if current_fps not in _FPS_PRESETS:
            self.fps_combo.addItem(str(current_fps), current_fps)
        self.fps_combo.setCurrentText(str(current_fps))
        self.fps_combo.currentIndexChanged.connect(self._on_fps_changed)
        toolbar.addWidget(self.fps_combo)

        self.act_swap_lr = QAction("Swap L/R", self)
        self.act_swap_lr.setCheckable(True)
        self.act_swap_lr.setChecked(bool(self.cfg.get("camera", {}).get("swap_left_right", False)))
        self.act_swap_lr.setToolTip(
            "Swap returned left/right camera images. Recalibrate after changing."
        )
        self.act_swap_lr.toggled.connect(self._on_swap_left_right_toggled)
        toolbar.addAction(self.act_swap_lr)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" View: "))
        self.view_combo = QComboBox()
        for label, *_ in _VIEW_MODES:
            self.view_combo.addItem(label)
        self.view_combo.currentIndexChanged.connect(self._apply_view_mode)
        toolbar.addWidget(self.view_combo)

        toolbar.addSeparator()
        act_zoom_in = QAction("Zoom +", self)
        act_zoom_in.triggered.connect(lambda: self._zoom_panels("in"))
        toolbar.addAction(act_zoom_in)
        act_zoom_out = QAction("Zoom -", self)
        act_zoom_out.triggered.connect(lambda: self._zoom_panels("out"))
        toolbar.addAction(act_zoom_out)
        act_zoom_reset = QAction("Reset Zoom", self)
        act_zoom_reset.triggered.connect(lambda: self._zoom_panels("reset"))
        toolbar.addAction(act_zoom_reset)

        self.status = QLabel("Idle")
        self.statusBar().addWidget(self.status)

        # Lightweight timer: only used to push raw frames when no calibration
        # is loaded. With calibration, the depth worker drives all 3 panels.
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.process_tick)

    def _on_resolution_changed(self, index):
        _, w, h = _RESOLUTIONS[index]
        self.cfg["camera"]["frame_width"] = w * 2   # full SBS frame
        self.cfg["camera"]["frame_height"] = h
        # Calibration remap maps are tied to the image size they were built at.
        # Clear them so the user is not silently applying wrong maps.
        was_calibrated = self.rectifier is not None
        if was_calibrated:
            self._stop_depth_worker()
            self.rectifier = None
            self.depth_engine = None
            self.depth_widget.set_depth_engine(None)
            self.depth_panel.setText("Resolution changed\nReload calibration to enable depth")
        # Restart capture at new resolution if camera is running.
        if self.capture_thread is not None:
            self._stop_depth_worker()
            self.capture_thread.stop()
            self.capture_thread = None
            self.camera = None
            self.start_camera()
        msg = f"Resolution set to {w}×{h} per eye"
        if was_calibrated:
            msg += " — calibration cleared, please reload"
        self.status.setText(msg)

    def _on_fps_changed(self, index):
        fps = int(self.fps_combo.itemData(index))
        self.cfg.setdefault("camera", {})["fps"] = fps
        if self.capture_thread is not None:
            self._stop_depth_worker()
            self.capture_thread.stop()
            self.capture_thread = None
            self.camera = None
            self.start_camera()
        self.status.setText(f"FPS set to {fps}")

    def _on_swap_left_right_toggled(self, checked):
        self.cfg.setdefault("camera", {})["swap_left_right"] = bool(checked)
        was_calibrated = self.rectifier is not None
        if was_calibrated:
            self._stop_depth_worker()
            self.rectifier = None
            self.depth_engine = None
            self.depth_widget.set_depth_engine(None)
            self.depth_panel.setText("Left/right swapped\nRecalibrate or reload matching calibration")
        if self.capture_thread is not None:
            self._stop_depth_worker()
            self.capture_thread.stop()
            self.capture_thread = None
            self.camera = None
            self.start_camera()
        msg = "Left/right swap enabled" if checked else "Left/right swap disabled"
        if was_calibrated:
            msg += " — calibration cleared"
        self.status.setText(msg)

    def _apply_view_mode(self, index=None):
        if index is None:
            index = self.view_combo.currentIndex()
        _, show_l, show_r, show_d = _VIEW_MODES[index]
        self.left_panel.setVisible(show_l)
        self.right_panel.setVisible(show_r)
        self.depth_panel.setVisible(show_d)

    def _zoom_panels(self, mode):
        for panel in (self.left_panel, self.right_panel, self.depth_panel):
            if mode == "in":
                panel.zoom_in()
            elif mode == "out":
                panel.zoom_out()
            else:
                panel.reset_zoom()

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
        self._reported_capture_format = False

        if self.rectifier is not None:
            self._start_depth_worker()

        self.timer.start()
        self.act_start.setText("Stop")
        msg = f"Capturing ({self.camera.mode})"
        actual = self.camera.actual_per_eye_size()
        if actual is not None:
            aw, ah = actual
            msg += f" {aw}×{ah}"
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
        if not self._reported_capture_format:
            h, w = left.shape[:2]
            msg = f"Capturing ({self.camera.mode}) {w}×{h}"
            if self.camera.unsynced:
                msg += "  ⚠  frames may be unsynchronized"
            self.status.setText(msg)
            self._reported_capture_format = True
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
        if "error" in result:
            self.status.setText(f"Depth worker error: {result['error']}")
            return
        self._latest_depth_map = result["depth_map"]
        self._latest_disparity = result.get("disparity")
        self._latest_rectified = (result["left"], result["right"])
        if not self._is_raw_mode():
            left = result["left"].copy()
            right = result["right"].copy()
            self._draw_stereo_clicks(left, right)
            self.left_panel.show_image(left)
            self.right_panel.show_image(right)
        # Depth panel always updated from worker regardless of mode.

        color = result["color"]
        if color is not None:
            if self._depth_click is not None:
                x, y = self._depth_click
                cv2.drawMarker(color, (x, y), (255, 255, 255),
                               cv2.MARKER_CROSS, 16, 1)
                self._draw_click_label(color, (x, y), self._depth_label, (255, 255, 255))
            self.depth_panel.show_image(color)

        if self._depth_click is not None:
            self._update_pixel_readout(*self._depth_click)

    # ------------------------------------------------------------ raw display
    def _is_raw_mode(self):
        return self.view_combo.currentText() == "Raw"

    def process_tick(self):
        """Show raw (unrectified) frames when no calibration or Raw mode selected."""
        if self.latest_raw is None:
            return
        raw_mode = self._is_raw_mode()
        if self.rectifier is not None and not raw_mode:
            return  # depth worker drives L/R panels
        left, right = self.latest_raw
        left_vis = left.copy()
        right_vis = right.copy()
        if self.rectifier is None:
            self._draw_stereo_clicks(left_vis, right_vis)
        self.left_panel.show_image(left_vis)
        self.right_panel.show_image(right_vis)
        if self.rectifier is None:
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

    def save_parameters(self):
        try:
            self.cfg.setdefault("camera", {})["swap_left_right"] = self.act_swap_lr.isChecked()
            self.cfg.setdefault("camera", {})["fps"] = int(self.fps_combo.currentData())
            if self.depth_widget is not None:
                self.depth_widget.sync_config_from_ui()
            path = save_config(self.cfg)
            self.status.setText(f"Saved parameters: {path}")
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Save failed: {exc}")
            QMessageBox.warning(self, "Save Parameters", f"Could not save parameters:\n{exc}")

    def apply_calibration(self, calib):
        """Install calibration: rebuild rectifier + depth engine + worker."""
        self.rectifier = Rectifier(calib)
        try:
            self.depth_engine = build_depth_engine(self.cfg, self.rectifier)
        except Exception as exc:  # noqa: BLE001
            engine_name = self.cfg.get("depth", {}).get("engine", "sgbm")
            self.status.setText(
                f"Depth engine '{engine_name}' failed ({exc}); falling back to SGBM"
            )
            self.cfg.setdefault("depth", {})["engine"] = "sgbm"
            self.depth_engine = build_depth_engine(self.cfg, self.rectifier)
        self.depth_widget.set_depth_engine(self.depth_engine)
        # Restart worker only if camera is running.
        if self.capture_thread is not None:
            self._start_depth_worker()

    # --------------------------------------------------------------- depth
    def on_depth_click(self, x, y):
        self._depth_click = (x, y)
        self._update_pixel_readout(x, y)

    def on_left_click(self, x, y):
        if self._is_raw_mode() and self.rectifier is not None:
            self.status.setText("Stereo match uses rectified view; switch from Raw to Left/Both/All")
            return
        self._update_stereo_match_from_left(x, y)

    def on_right_click(self, x, y):
        if self._is_raw_mode() and self.rectifier is not None:
            self.status.setText("Stereo match uses rectified view; switch from Raw to Right/Both/All")
            return
        self._update_stereo_match_from_right(x, y)

    def _update_stereo_match_from_left(self, x, y):
        valid = False
        left_xy = (int(x), int(y))
        right_xy = None
        disp = float("nan")
        if self._latest_disparity is not None:
            h, w = self._latest_disparity.shape
            if 0 <= x < w and 0 <= y < h:
                disp = self._robust_disparity_sample(x, y)
                disp = self._refine_left_match(x, y, disp)
                rx = int(round(x - disp)) if np.isfinite(disp) else -1
                valid = np.isfinite(disp) and disp > 0 and 0 <= rx < w
                if valid:
                    right_xy = (rx, int(y))
        self._set_stereo_match(left_xy, right_xy, disp, valid)
        self._depth_click = left_xy
        self._update_pixel_readout(*left_xy)

    def _update_stereo_match_from_right(self, x, y):
        valid = False
        right_xy = (int(x), int(y))
        left_xy = None
        disp = float("nan")
        if self._latest_disparity is not None:
            h, w = self._latest_disparity.shape
            if 0 <= x < w and 0 <= y < h:
                row = self._latest_disparity[y]
                xs = np.arange(w, dtype=np.float32)
                mapped_right = xs - row
                err = np.abs(mapped_right - float(x))
                err[~np.isfinite(err)] = np.inf
                lx = int(np.argmin(err))
                disp = float(row[lx])
                valid = (
                    np.isfinite(disp) and disp > 0
                    and np.isfinite(err[lx]) and err[lx] <= 1.5
                )
                if valid:
                    lx = self._refine_right_match(x, y, lx)
                    disp = float(lx - x)
                    valid = disp > 0
                if valid:
                    left_xy = (lx, int(y))
        self._set_stereo_match(left_xy, right_xy, disp, valid)
        if valid:
            self._depth_click = left_xy
            self._update_pixel_readout(*left_xy)

    def _robust_disparity_sample(self, x, y):
        if self._latest_disparity is None:
            return float("nan")
        h, w = self._latest_disparity.shape
        if not (0 <= x < w and 0 <= y < h):
            return float("nan")
        radius = 3
        trim = 0.2
        if self.depth_engine is not None:
            radius = int(getattr(self.depth_engine, "sample_radius_px", radius))
            trim = float(getattr(self.depth_engine, "sample_trim", trim))
        x1 = max(0, int(x) - radius)
        x2 = min(w, int(x) + radius + 1)
        y1 = max(0, int(y) - radius)
        y2 = min(h, int(y) + radius + 1)
        vals = self._latest_disparity[y1:y2, x1:x2]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return float("nan")
        vals = np.sort(vals.astype(np.float32))
        cut = int(vals.size * max(0.0, min(0.45, trim)))
        if cut > 0 and vals.size > 2 * cut:
            vals = vals[cut:-cut]
        if vals.size == 0:
            return float("nan")
        return float(np.median(vals))

    def _refine_left_match(self, x, y, initial_disp):
        """Refine a clicked left pixel with local texture matching on the same row."""
        if self._latest_rectified is None or not np.isfinite(initial_disp):
            return initial_disp
        left, right = self._latest_rectified
        h, w = left.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return initial_disp
        patch_radius = 7
        search_radius = 24
        if (
            x - patch_radius < 0 or x + patch_radius >= w
            or y - patch_radius < 0 or y + patch_radius >= h
        ):
            return initial_disp

        rx0 = int(round(x - initial_disp))
        rx1 = max(patch_radius, rx0 - search_radius)
        rx2 = min(w - patch_radius - 1, rx0 + search_radius)
        if rx2 < rx1:
            return initial_disp

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        tpl = left_gray[
            y - patch_radius:y + patch_radius + 1,
            x - patch_radius:x + patch_radius + 1,
        ]
        strip = right_gray[
            y - patch_radius:y + patch_radius + 1,
            rx1 - patch_radius:rx2 + patch_radius + 1,
        ]
        if strip.shape[1] < tpl.shape[1]:
            return initial_disp

        scores = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(scores)
        if score < 0.45:
            return initial_disp
        best_rx = rx1 + loc[0]
        refined = float(x - best_rx)
        if refined <= 0:
            return initial_disp
        return refined

    def _refine_right_match(self, x, y, initial_left_x):
        """Refine a clicked right pixel by matching its patch into the left image."""
        if self._latest_rectified is None:
            return int(initial_left_x)
        left, right = self._latest_rectified
        h, w = right.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return int(initial_left_x)
        patch_radius = 7
        search_radius = 24
        lx0 = int(round(initial_left_x))
        if (
            x - patch_radius < 0 or x + patch_radius >= w
            or y - patch_radius < 0 or y + patch_radius >= h
        ):
            return lx0
        lx1 = max(patch_radius, lx0 - search_radius)
        lx2 = min(w - patch_radius - 1, lx0 + search_radius)
        if lx2 < lx1:
            return lx0

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        tpl = right_gray[
            y - patch_radius:y + patch_radius + 1,
            x - patch_radius:x + patch_radius + 1,
        ]
        strip = left_gray[
            y - patch_radius:y + patch_radius + 1,
            lx1 - patch_radius:lx2 + patch_radius + 1,
        ]
        if strip.shape[1] < tpl.shape[1]:
            return lx0

        scores = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(scores)
        if score < 0.45:
            return lx0
        return lx1 + loc[0]

    def _set_stereo_match(self, left_xy, right_xy, disparity_px, valid):
        if valid:
            self._stereo_click = {"left": left_xy, "right": right_xy}
        else:
            self._stereo_click = None
        self.depth_widget.show_stereo_match(left_xy, right_xy, disparity_px, valid)

    def _draw_stereo_clicks(self, left, right):
        if self._stereo_click is None:
            return
        points = ((left, self._stereo_click["left"]), (right, self._stereo_click["right"]))
        for img, xy in points:
            x, y = xy
            cv2.drawMarker(img, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.circle(img, (x, y), 5, (0, 255, 255), 1)
            self._draw_click_label(img, (x, y), self._depth_label, (0, 255, 255))

    def _draw_click_label(self, img, xy, text, color):
        if not text:
            return
        x, y = xy
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        tx = min(max(4, x + 10), max(4, w - tw - 8))
        ty = y - 10 if y - th - base - 14 >= 0 else y + th + base + 14
        ty = min(max(th + base + 4, ty), h - 4)
        cv2.rectangle(
            img,
            (tx - 4, ty - th - base - 4),
            (tx + tw + 4, ty + base + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(img, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

    def _update_pixel_readout(self, x, y):
        if self.depth_engine is None:
            return
        # Use the snapshot depth_map stored from the last worker result so
        # pixel_info never races with the worker writing a new depth_map.
        info = self.depth_engine.pixel_info_from_map(self._latest_depth_map, x, y)
        if info.valid:
            self._depth_label = f"{info.depth_mm:.1f} +/-{info.sample_std_mm:.1f} mm"
        else:
            self._depth_label = "no depth"
        self.depth_widget.show_pixel_info(info)

    # ----------------------------------------------------------- shutdown
    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)
