"""Main window for the detection + ROS2 publisher app."""

import os

import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QAction, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QWidget,
)

from ..calibration import load_yaml
from ..camera import CaptureThread, StereoCamera
from ..config import project_root
from ..depth import Rectifier, build_depth_engine
from ..detection import (
    DetectionWorker, TiledDetector, TopPickTracker, build_detector,
    fit_tray_plane,
)
from ..ros2 import ROS2_AVAILABLE, _IMPORT_ERROR
from .detect_widget import DetectionWidget
from .widgets import ImagePanel


if ROS2_AVAILABLE:
    import rclpy
    from ..ros2 import DetectionPublisher, RosSpinThread


_BBOX_COLOR = (0, 255, 0)
_TOP_COLOR = (0, 255, 255)
_TEXT_COLOR = (255, 255, 255)


def _draw_overlay(bgr, detections, top=None):
    out = bgr.copy()
    top_id = id(top) if top is not None else None
    for item in detections:
        det = item["det"]
        x1, y1, x2, y2 = det.bbox
        z = item["xyz_mm"][2]
        color = _TOP_COLOR if (top_id and id(item) == top_id) else _BBOX_COLOR
        thick = 3 if color is _TOP_COLOR else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        src = item.get("depth_source", "stereo")
        label = f"{det.cls_name} {det.score:.2f} Z={z:.0f}mm [{src}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(0, y1 - th - 4)
        cv2.rectangle(out, (x1, y_top), (x1 + tw + 4, y_top + th + 4), color, -1)
        cv2.putText(out, label, (x1 + 2, y_top + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        u, v = item["uv"]
        cv2.drawMarker(out, (u, v), _TEXT_COLOR, cv2.MARKER_CROSS, 12, 1)
    if top is not None and id(top) not in {id(d) for d in detections}:
        # Smoothed top may have a different uv than any raw det; draw a marker.
        u, v = top["uv"]
        cv2.drawMarker(out, (u, v), _TOP_COLOR, cv2.MARKER_TILTED_CROSS, 20, 2)
    return out


class DetectMainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("ELP 3D — Worm Detection + ROS2")
        self.resize(1500, 850)

        self.camera = None
        self.capture_thread = None
        self.rectifier = None
        self.depth_engine = None
        self.calib = None
        self.detector = None
        self.det_worker = None
        self._latest_depth_map = None

        # Optimization state (mirrors config but mutable from UI).
        det_cfg = cfg.get("detection", {})
        pre_cfg = cfg.get("preproc", {})
        tile_cfg = cfg.get("tiling", {})
        smooth_cfg = cfg.get("smoothing", {})
        pick_cfg = cfg.get("pick_strategy", {})
        self._clahe_on = bool(pre_cfg.get("clahe", True))
        self._tiling_on = bool(tile_cfg.get("enabled", True))
        self._tile_grid = tuple(tile_cfg.get("grid", [2, 2]))
        self._tile_overlap = float(tile_cfg.get("overlap", 0.2))
        self._smoothing_on = bool(smooth_cfg.get("enabled", True))
        self._publish_top1 = pick_cfg.get("mode", "top_score") == "top_score"
        self._fallback_plane = bool(det_cfg.get("depth_fallback_to_plane", True))
        self._bbox_shrink = float(det_cfg.get("bbox_shrink", 0.6))
        self._smooth_cfg = smooth_cfg

        # ROS2 state
        self._ros_node = None
        self._ros_spin = None
        self._ros_initialized = False

        self._build_ui()
        self._try_autoload_calibration()
        self._maybe_start_ros()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        self.left_panel = ImagePanel("Left + Detections")
        self.depth_panel = ImagePanel("Depth")

        panels = QHBoxLayout()
        panels.addWidget(self.left_panel, 1)
        panels.addWidget(self.depth_panel, 1)
        panels_box = QWidget(); panels_box.setLayout(panels)

        self.dock = DetectionWidget(self.cfg)
        self.dock.setMaximumWidth(440)
        self.dock.model_load_requested.connect(self.load_detector)
        self.dock.model_clear_requested.connect(self.clear_detector)
        self.dock.ros_toggle_requested.connect(self.toggle_ros)
        self.dock.clahe_toggled.connect(self.on_clahe_toggled)
        self.dock.tiling_toggled.connect(self.on_tiling_toggled)
        self.dock.smoothing_toggled.connect(self.on_smoothing_toggled)
        self.dock.publish_top1_toggled.connect(self.on_publish_top1_toggled)
        self.dock.fallback_plane_toggled.connect(self.on_fallback_plane_toggled)
        self.dock.fit_plane_requested.connect(self.on_fit_plane)
        self.dock.clear_plane_requested.connect(self.on_clear_plane)
        self.dock.bbox_shrink_changed.connect(self.on_bbox_shrink_changed)

        body = QHBoxLayout()
        body.addWidget(panels_box, 1)
        body.addWidget(self.dock)
        central = QWidget(); central.setLayout(body)
        self.setCentralWidget(central)

        toolbar = self.addToolBar("Main")
        self.act_start = QAction("Start Camera", self)
        self.act_start.triggered.connect(self.toggle_camera)
        toolbar.addAction(self.act_start)
        act_load_calib = QAction("Load Calibration...", self)
        act_load_calib.triggered.connect(self.load_calibration_dialog)
        toolbar.addAction(act_load_calib)

        self.status = QLabel("Idle")
        self.statusBar().addWidget(self.status)

        self._ros_tick = QTimer(self)
        self._ros_tick.setInterval(500)
        self._ros_tick.timeout.connect(self._refresh_ros_status)
        self._ros_tick.start()

    # ------------------------------------------------------------ camera
    def toggle_camera(self):
        if self.capture_thread is None:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        if self.rectifier is None:
            self.status.setText("Load calibration first")
            return
        try:
            self.camera = StereoCamera(self.cfg)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Camera error: {exc}")
            return
        self.capture_thread = CaptureThread(self.camera)
        self.capture_thread.frames_ready.connect(self.on_frames_ready)
        self.capture_thread.error.connect(lambda m: self.status.setText(f"Capture: {m}"))
        self.capture_thread.start()
        self._start_worker()
        self.act_start.setText("Stop Camera")
        self.status.setText(f"Capturing ({self.camera.mode})")

    def stop_camera(self):
        self._stop_worker()
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread = None
        self.camera = None
        self.act_start.setText("Start Camera")
        self.status.setText("Idle")

    def on_frames_ready(self, left, right):
        if self.det_worker is not None:
            self.det_worker.submit(left, right)

    # ------------------------------------------------------------ worker
    def _start_worker(self):
        self._stop_worker()
        if self.rectifier is None or self.depth_engine is None:
            return
        det_cfg = self.cfg.get("detection", {})
        self.det_worker = DetectionWorker(
            self.rectifier, self.depth_engine,
            self._wrap_detector(self.detector),
            min_valid_pixels=int(det_cfg.get("min_valid_pixels", 5)),
            bbox_shrink=self._bbox_shrink,
            fallback_to_plane=self._fallback_plane,
        )
        if self.calib is not None:
            self.det_worker.set_projection(self.calib.P1)
        self.det_worker.set_clahe(self._clahe_on)
        if self._smoothing_on:
            self.det_worker.set_tracker(self._build_tracker())
        self.det_worker.result_ready.connect(self.on_result)
        self.det_worker.start()

    def _stop_worker(self):
        if self.det_worker is not None:
            self.det_worker.stop()
            self.det_worker = None

    def _build_tracker(self):
        return TopPickTracker(
            alpha=float(self._smooth_cfg.get("alpha", 0.4)),
            reset_px_dist=float(self._smooth_cfg.get("reset_px_dist", 30.0)),
            max_miss_frames=int(self._smooth_cfg.get("max_miss_frames", 10)),
        )

    def _wrap_detector(self, base):
        """Wrap base detector in TiledDetector if tiling is enabled."""
        if base is None:
            return None
        if self._tiling_on:
            return TiledDetector(base, grid=self._tile_grid,
                                 overlap=self._tile_overlap)
        return base

    def on_result(self, result):
        if "error" in result:
            self.status.setText(f"Worker error: {result['error']}")
            return
        detections = result.get("detections", [])
        top = result.get("top")
        left = result["left"]
        if detections or top is not None:
            left = _draw_overlay(left, detections, top)
        self.left_panel.show_image(left)
        depth_color = result.get("depth_color")
        if depth_color is not None:
            self.depth_panel.show_image(depth_color)
        self._latest_depth_map = result.get("depth_map")
        self.dock.show_detections(detections, top)

        if self._ros_node is None:
            return
        to_publish = []
        if self._publish_top1:
            if top is not None:
                to_publish = [top]
        else:
            to_publish = detections
        if not to_publish:
            return
        try:
            n = self._ros_node.publish_detections(to_publish)
            self.dock.set_ros_status(f"ROS2: published {n} detection(s)")
        except Exception as exc:  # noqa: BLE001
            self.dock.set_ros_status(f"ROS2 publish failed: {exc}")

    # -------------------------------------------------------- calibration
    def _try_autoload_calibration(self):
        path = self.cfg.get("calibration_path") \
            or self.cfg["calibration"]["output_path"]
        full = project_root() / path
        if os.path.isfile(full):
            try:
                self.apply_calibration(load_yaml(full))
                self.status.setText(f"Loaded calibration: {full}")
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
        self.calib = calib
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
        if self.capture_thread is not None:
            self._start_worker()

    # ---------------------------------------------------------- detector
    def load_detector(self, path, conf, iou, min_valid):
        det_cfg = self.cfg.get("detection", {})
        try:
            self.detector = build_detector(
                path, conf=conf, iou=iou,
                classes=det_cfg.get("classes"),
                class_names=det_cfg.get("onnx_class_names"),
                input_size=int(det_cfg.get("onnx_input_size", 960)),
            )
        except Exception as exc:  # noqa: BLE001
            self.dock.set_model_status(f"Load failed: {exc}")
            return
        nclasses = len(self.detector.class_names)
        self.dock.set_model_status(
            f"Loaded {os.path.basename(path)} ({nclasses} classes)"
        )
        if self.det_worker is not None:
            self.det_worker.set_detector(self._wrap_detector(self.detector))
            self.det_worker.set_min_valid_pixels(min_valid)
        self.cfg.setdefault("detection", {})["min_valid_pixels"] = int(min_valid)

    def clear_detector(self):
        self.detector = None
        if self.det_worker is not None:
            self.det_worker.set_detector(None)
        self.dock.set_model_status("No model loaded")

    # ------------------------------------------------------- optimizations
    def on_clahe_toggled(self, on):
        self._clahe_on = bool(on)
        if self.det_worker is not None:
            self.det_worker.set_clahe(self._clahe_on)

    def on_tiling_toggled(self, on):
        self._tiling_on = bool(on)
        if self.det_worker is not None and self.detector is not None:
            self.det_worker.set_detector(self._wrap_detector(self.detector))

    def on_smoothing_toggled(self, on):
        self._smoothing_on = bool(on)
        if self.det_worker is not None:
            self.det_worker.set_tracker(self._build_tracker() if on else None)

    def on_publish_top1_toggled(self, on):
        self._publish_top1 = bool(on)

    def on_fallback_plane_toggled(self, on):
        self._fallback_plane = bool(on)
        if self.det_worker is not None:
            self.det_worker.set_fallback_to_plane(self._fallback_plane)

    def on_bbox_shrink_changed(self, val):
        self._bbox_shrink = float(val)
        if self.det_worker is not None:
            self.det_worker.set_bbox_shrink(self._bbox_shrink)

    def on_fit_plane(self):
        if self._latest_depth_map is None or self.calib is None:
            self.dock.set_plane_status("Plane fit: need a depth frame first")
            return
        plane_cfg = self.cfg.get("tray_plane", {})
        try:
            plane = fit_tray_plane(
                self._latest_depth_map, self.calib.P1,
                iters=int(plane_cfg.get("ransac_iters", 200)),
                thresh_mm=float(plane_cfg.get("ransac_threshold_mm", 5.0)),
            )
        except Exception as exc:  # noqa: BLE001
            self.dock.set_plane_status(f"Plane fit failed: {exc}")
            return
        if self.det_worker is not None:
            self.det_worker.set_plane(plane)
        self.dock.set_plane_status(
            f"Plane: meanZ={plane.mean_z_mm:.1f}mm  tilt={plane.tilt_deg:.1f}°  "
            f"RMS={plane.rms:.2f}mm"
        )

    def on_clear_plane(self):
        if self.det_worker is not None:
            self.det_worker.set_plane(None)
        self.dock.set_plane_status("Plane: cleared")

    # ---------------------------------------------------------------- ROS2
    def _maybe_start_ros(self):
        ros_cfg = self.cfg.get("ros2", {})
        if not ros_cfg.get("enabled", True):
            self.dock.set_ros_status("ROS2: disabled in config")
            return
        if not ROS2_AVAILABLE:
            self.dock.set_ros_status(f"ROS2 unavailable: {_IMPORT_ERROR}")
            return
        self._start_ros(ros_cfg)

    def _start_ros(self, ros_cfg):
        try:
            if not self._ros_initialized:
                rclpy.init()
                self._ros_initialized = True
            self._ros_node = DetectionPublisher(ros_cfg)
            self._ros_spin = RosSpinThread(self._ros_node)
            self._ros_spin.start()
            self.dock.set_ros_status(
                f"ROS2: publishing to {ros_cfg.get('topic')}"
            )
        except Exception as exc:  # noqa: BLE001
            self._ros_node = None
            self._ros_spin = None
            self.dock.set_ros_status(f"ROS2 start failed: {exc}")

    def _stop_ros(self):
        if self._ros_spin is not None:
            self._ros_spin.stop()
            self._ros_spin = None
        if self._ros_node is not None:
            try:
                self._ros_node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
            self._ros_node = None
        if self._ros_initialized:
            try:
                rclpy.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._ros_initialized = False

    def toggle_ros(self, enabled):
        if enabled:
            if self._ros_node is None:
                self._start_ros(self.cfg.get("ros2", {}))
        else:
            self._stop_ros()
            self.dock.set_ros_status("ROS2: stopped")

    def _refresh_ros_status(self):
        if self._ros_node is None:
            return
        if not self._ros_node.use_tf_lookup:
            return
        ok = self._ros_node.last_lookup_ok
        if ok is None:
            return
        self.dock.set_ros_status(
            f"ROS2: lookup {'OK' if ok else 'FAIL'} "
            f"(camera->{self._ros_node.robot_frame})"
        )

    # ----------------------------------------------------------- shutdown
    def closeEvent(self, event):
        self.stop_camera()
        self._stop_ros()
        super().closeEvent(event)
