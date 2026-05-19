"""Detection tab UI: model picker, confidence, optimization toggles,
tray-plane fit, detection table, ROS status.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


class DetectionWidget(QWidget):
    """Right-hand control panel for the detection app."""

    model_load_requested = pyqtSignal(str, float, float, int)  # path, conf, iou, min_valid
    model_clear_requested = pyqtSignal()
    ros_toggle_requested = pyqtSignal(bool)

    # New optimization toggles
    clahe_toggled = pyqtSignal(bool)
    tiling_toggled = pyqtSignal(bool)
    smoothing_toggled = pyqtSignal(bool)
    publish_top1_toggled = pyqtSignal(bool)
    fallback_plane_toggled = pyqtSignal(bool)
    fit_plane_requested = pyqtSignal()
    clear_plane_requested = pyqtSignal()
    bbox_shrink_changed = pyqtSignal(float)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._build_ui()

    def _build_ui(self):
        det_cfg = self._cfg.get("detection", {})
        ros_cfg = self._cfg.get("ros2", {})
        preproc_cfg = self._cfg.get("preproc", {})
        tiling_cfg = self._cfg.get("tiling", {})
        plane_cfg = self._cfg.get("tray_plane", {})
        smooth_cfg = self._cfg.get("smoothing", {})
        pick_cfg = self._cfg.get("pick_strategy", {})

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        # ---- Detector group
        layout.addWidget(self._section("Detector"))
        form = QFormLayout()
        self.model_edit = QLineEdit(str(det_cfg.get("model_path", "")))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._on_browse)
        row = QHBoxLayout()
        row.addWidget(self.model_edit, 1)
        row.addWidget(browse)
        row_w = QWidget(); row_w.setLayout(row)
        form.addRow("Model:", row_w)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99); self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(float(det_cfg.get("confidence", 0.25)))
        form.addRow("Confidence:", self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.1, 0.95); self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(float(det_cfg.get("iou", 0.45)))
        form.addRow("NMS IoU:", self.iou_spin)

        self.min_valid_spin = QSpinBox()
        self.min_valid_spin.setRange(1, 100000)
        self.min_valid_spin.setValue(int(det_cfg.get("min_valid_pixels", 5)))
        form.addRow("Min valid px:", self.min_valid_spin)

        self.shrink_spin = QDoubleSpinBox()
        self.shrink_spin.setRange(0.1, 1.0); self.shrink_spin.setSingleStep(0.05)
        self.shrink_spin.setValue(float(det_cfg.get("bbox_shrink", 0.6)))
        self.shrink_spin.valueChanged.connect(
            lambda v: self.bbox_shrink_changed.emit(float(v))
        )
        form.addRow("BBox shrink:", self.shrink_spin)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load / Reload")
        self.load_btn.clicked.connect(self._emit_load)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.model_clear_requested.emit)
        btn_row.addWidget(self.load_btn); btn_row.addWidget(self.clear_btn)
        layout.addLayout(btn_row)

        self.model_status = QLabel("No model loaded")
        self.model_status.setWordWrap(True)
        layout.addWidget(self.model_status)

        # ---- Optimization group
        layout.addWidget(self._section("Optimizations"))
        self.clahe_chk = QCheckBox("CLAHE pre-processing (low light)")
        self.clahe_chk.setChecked(bool(preproc_cfg.get("clahe", True)))
        self.clahe_chk.toggled.connect(self.clahe_toggled.emit)
        layout.addWidget(self.clahe_chk)

        self.tile_chk = QCheckBox("Tiled inference (2x2, 20% overlap)")
        self.tile_chk.setChecked(bool(tiling_cfg.get("enabled", True)))
        self.tile_chk.toggled.connect(self.tiling_toggled.emit)
        layout.addWidget(self.tile_chk)

        self.smooth_chk = QCheckBox("Smooth top-1 (centroid EMA)")
        self.smooth_chk.setChecked(bool(smooth_cfg.get("enabled", True)))
        self.smooth_chk.toggled.connect(self.smoothing_toggled.emit)
        layout.addWidget(self.smooth_chk)

        self.top1_chk = QCheckBox("Publish only top-1 detection")
        self.top1_chk.setChecked(pick_cfg.get("mode", "top_score") == "top_score")
        self.top1_chk.toggled.connect(self.publish_top1_toggled.emit)
        layout.addWidget(self.top1_chk)

        self.fallback_chk = QCheckBox("Fallback to tray plane Z")
        self.fallback_chk.setChecked(bool(det_cfg.get("depth_fallback_to_plane", True)))
        self.fallback_chk.toggled.connect(self.fallback_plane_toggled.emit)
        layout.addWidget(self.fallback_chk)

        # ---- Tray plane group
        layout.addWidget(self._section("Tray plane"))
        plane_row = QHBoxLayout()
        self.fit_plane_btn = QPushButton("Fit Tray Plane")
        self.fit_plane_btn.setEnabled(bool(plane_cfg.get("enabled", True)))
        self.fit_plane_btn.clicked.connect(self.fit_plane_requested.emit)
        self.clear_plane_btn = QPushButton("Clear")
        self.clear_plane_btn.clicked.connect(self.clear_plane_requested.emit)
        plane_row.addWidget(self.fit_plane_btn)
        plane_row.addWidget(self.clear_plane_btn)
        layout.addLayout(plane_row)
        self.plane_status = QLabel("Plane: not fit")
        self.plane_status.setWordWrap(True)
        layout.addWidget(self.plane_status)

        # ---- ROS2 group
        layout.addWidget(self._section("ROS2"))
        self.ros_enable = QCheckBox("Publish to ROS2")
        self.ros_enable.setChecked(bool(ros_cfg.get("enabled", True)))
        self.ros_enable.toggled.connect(self.ros_toggle_requested.emit)
        layout.addWidget(self.ros_enable)

        topic = str(ros_cfg.get("topic", "/elp/detections"))
        cam_f = str(ros_cfg.get("camera_frame", "camera_optical_frame"))
        rob_f = str(ros_cfg.get("robot_frame", "robot_base"))
        lookup = "ON" if ros_cfg.get("use_tf_lookup", False) else "OFF"
        layout.addWidget(QLabel(f"Topic: {topic}"))
        layout.addWidget(QLabel(f"camera_frame: {cam_f}"))
        layout.addWidget(QLabel(f"robot_frame:  {rob_f}"))
        layout.addWidget(QLabel(f"TF lookup:    {lookup}"))

        self.ros_status = QLabel("ROS2: not started")
        self.ros_status.setWordWrap(True)
        layout.addWidget(self.ros_status)

        # ---- Detections table
        layout.addWidget(self._section("Detections (latest frame)"))
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["class", "score", "u", "v", "X mm", "Y mm", "Z mm", "src"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table, 1)

    def _section(self, title):
        lab = QLabel(title)
        f = lab.font(); f.setBold(True); lab.setFont(f)
        wrap = QFrame()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 8, 0, 2)
        v.addWidget(lab)
        return wrap

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model", "", "Models (*.pt *.onnx)"
        )
        if path:
            self.model_edit.setText(path)

    def _emit_load(self):
        path = self.model_edit.text().strip()
        if not path:
            self.set_model_status("Pick a .pt or .onnx file first")
            return
        self.model_load_requested.emit(
            path, float(self.conf_spin.value()),
            float(self.iou_spin.value()),
            int(self.min_valid_spin.value()),
        )

    def set_model_status(self, text):
        self.model_status.setText(text)

    def set_ros_status(self, text):
        self.ros_status.setText(text)

    def set_plane_status(self, text):
        self.plane_status.setText(text)

    def show_detections(self, items, top=None):
        self.table.setRowCount(len(items))
        top_id = id(top) if top is not None else None
        for i, item in enumerate(items):
            det = item["det"]
            u, v = item["uv"]
            x, y, z = item["xyz_mm"]
            src = item.get("depth_source", "stereo")
            cells = [
                det.cls_name,
                f"{det.score:.2f}",
                str(u), str(v),
                f"{x:.1f}", f"{y:.1f}", f"{z:.1f}",
                src,
            ]
            for j, val in enumerate(cells):
                w = QTableWidgetItem(val)
                if top_id is not None and id(item) == top_id:
                    f = w.font(); f.setBold(True); w.setFont(f)
                self.table.setItem(i, j, w)
