"""Depth tab: SGBM parameter controls and the pixel-click depth readout.

Spinboxes are used instead of raw sliders because two SGBM parameters have hard
constraints -- numDisparities must be a multiple of 16 and blockSize must be odd
-- which a free slider cannot express cleanly.
"""

import cv2
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..depth import WLS_AVAILABLE

# Ordered list of (label, cv2 constant). TURBO/MAGMA/INFERNO/PLASMA require OpenCV >= 4.1.
_COLORMAPS = [
    ("TURBO",            cv2.COLORMAP_TURBO),
    ("AUTUMN (warm)",    cv2.COLORMAP_AUTUMN),
    ("HOT",              cv2.COLORMAP_HOT),
    ("JET",              cv2.COLORMAP_JET),
    ("MAGMA",            cv2.COLORMAP_MAGMA),
    ("INFERNO",          cv2.COLORMAP_INFERNO),
    ("PLASMA",           cv2.COLORMAP_PLASMA),
    ("VIRIDIS",          cv2.COLORMAP_VIRIDIS),
    ("PARULA",           cv2.COLORMAP_PARULA),
    ("CIVIDIS",          cv2.COLORMAP_CIVIDIS),
    ("RAINBOW",          cv2.COLORMAP_RAINBOW),
    ("OCEAN",            cv2.COLORMAP_OCEAN),
    ("BONE",             cv2.COLORMAP_BONE),
    ("PINK",             cv2.COLORMAP_PINK),
    ("SPRING",           cv2.COLORMAP_SPRING),
    ("SUMMER",           cv2.COLORMAP_SUMMER),
    ("WINTER",           cv2.COLORMAP_WINTER),
    ("COOL",             cv2.COLORMAP_COOL),
    ("HSV",              cv2.COLORMAP_HSV),
    ("TWILIGHT",         cv2.COLORMAP_TWILIGHT),
    ("TWILIGHT SHIFTED", cv2.COLORMAP_TWILIGHT_SHIFTED),
    ("DEEPGREEN",        cv2.COLORMAP_DEEPGREEN),
]


class DepthWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.cfg = main_window.cfg
        self.depth_engine = None
        self._build_ui()
        self.setEnabled(False)  # enabled once a calibration is applied

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Colormap selector -------------------------------------------
        cmap_box = QGroupBox("Depth colormap")
        cmap_form = QFormLayout(cmap_box)
        self.cmap_combo = QComboBox()
        for label, _ in _COLORMAPS:
            self.cmap_combo.addItem(label)
        self.cmap_combo.setCurrentText("TURBO")
        self.cmap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        cmap_form.addRow("Colormap", self.cmap_combo)
        layout.addWidget(cmap_box)

        # --- Depth range cap ---------------------------------------------
        range_box = QGroupBox("Depth range (mm)")
        range_form = QFormLayout(range_box)
        cap_cfg = self.cfg["depth"]
        self.sp_min_depth = QDoubleSpinBox()
        self.sp_min_depth.setRange(1.0, 100000.0)
        self.sp_min_depth.setDecimals(0)
        self.sp_min_depth.setSingleStep(10.0)
        self.sp_min_depth.setSuffix(" mm")
        self.sp_min_depth.setValue(float(cap_cfg.get("min_depth_mm", 50.0)))
        self.sp_max_depth = QDoubleSpinBox()
        self.sp_max_depth.setRange(10.0, 100000.0)
        self.sp_max_depth.setDecimals(0)
        self.sp_max_depth.setSingleStep(100.0)
        self.sp_max_depth.setSuffix(" mm")
        self.sp_max_depth.setValue(float(cap_cfg.get("max_depth_mm", 3000.0)))
        # Live-apply on change (no Apply button needed for these — fast path).
        self.sp_min_depth.valueChanged.connect(self._on_depth_range_changed)
        self.sp_max_depth.valueChanged.connect(self._on_depth_range_changed)
        range_form.addRow("min", self.sp_min_depth)
        range_form.addRow("max", self.sp_max_depth)
        layout.addWidget(range_box)

        s = self.cfg["depth"]["sgbm"]
        params = QGroupBox("StereoSGBM parameters")
        form = QFormLayout(params)

        self.sp_min_disp = self._spin(0, 512, 1, s["min_disparity"])
        self.sp_num_disp = self._spin(16, 512, 16, s["num_disparities"])
        self.sp_block = self._spin(1, 21, 2, s["block_size"])
        self.sp_uniq = self._spin(0, 50, 1, s["uniqueness_ratio"])
        self.sp_speckle_win = self._spin(0, 300, 10, s["speckle_window_size"])
        self.sp_speckle_range = self._spin(0, 10, 1, s["speckle_range"])
        self.sp_disp12 = self._spin(-1, 50, 1, s["disp12_max_diff"])
        self.sp_temporal = self._spin(
            1, 32, 1, int(self.cfg["depth"].get("temporal_frames", 1))
        )

        form.addRow("min disparity", self.sp_min_disp)
        form.addRow("num disparities", self.sp_num_disp)
        form.addRow("block size", self.sp_block)
        form.addRow("uniqueness ratio", self.sp_uniq)
        form.addRow("speckle window", self.sp_speckle_win)
        form.addRow("speckle range", self.sp_speckle_range)
        form.addRow("disp12 max diff", self.sp_disp12)
        form.addRow("temporal frames", self.sp_temporal)

        self.chk_wls = QCheckBox("WLS post-filter (ximgproc)")
        if WLS_AVAILABLE:
            self.chk_wls.setChecked(self.cfg["depth"]["use_wls_filter"])
        else:
            self.chk_wls.setChecked(False)
            self.chk_wls.setEnabled(False)
            self.chk_wls.setText("WLS post-filter (ximgproc not installed)")
        form.addRow(self.chk_wls)

        layout.addWidget(params)

        self.btn_apply = QPushButton("Apply Parameters")
        self.btn_apply.clicked.connect(self.apply_params)
        layout.addWidget(self.btn_apply)

        # --- pixel readout -----------------------------------------------
        readout = QGroupBox("Pixel depth (click the depth panel)")
        rv = QVBoxLayout(readout)
        self.lbl_depth = QLabel("depth: -")
        self.lbl_error = QLabel("error band: -")
        self.lbl_range = QLabel("detection range: -")
        for lbl in (self.lbl_depth, self.lbl_error, self.lbl_range):
            rv.addWidget(lbl)
        self.lbl_depth.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(readout)

        layout.addStretch(1)

    def _spin(self, lo, hi, step, val):
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setValue(val)
        return sp

    def set_depth_engine(self, engine):
        self.depth_engine = engine
        if engine is None:
            self.setEnabled(False)
            return
        self.setEnabled(True)
        # Push current widget state into the fresh engine (calibration load
        # must not reset user-tuned params or colormap choice).
        self.apply_params()
        self._on_colormap_changed()
        self._on_depth_range_changed()

    def apply_params(self):
        if self.depth_engine is None:
            return
        # blockSize must be odd.
        block = self.sp_block.value()
        if block % 2 == 0:
            block += 1
            self.sp_block.setValue(block)
        params = {
            "sgbm": {
                "min_disparity": self.sp_min_disp.value(),
                "num_disparities": self.sp_num_disp.value(),
                "block_size": block,
                "uniqueness_ratio": self.sp_uniq.value(),
                "speckle_window_size": self.sp_speckle_win.value(),
                "speckle_range": self.sp_speckle_range.value(),
                "disp12_max_diff": self.sp_disp12.value(),
                "p1": None,
                "p2": None,
            },
            "use_wls_filter": self.chk_wls.isChecked(),
            "temporal_frames": self.sp_temporal.value(),
        }
        self.depth_engine.update_params(params)
        rmin, rmax = self.depth_engine.detection_range()
        self.lbl_range.setText(
            f"detection range: {rmin:.0f} - {rmax:.0f} mm"
        )

    def _on_colormap_changed(self, _index=None):
        if self.depth_engine is None:
            return
        idx = self.cmap_combo.currentIndex()
        self.depth_engine.colormap = _COLORMAPS[idx][1]

    def _on_depth_range_changed(self, _value=None):
        """Live-push depth cap to the engine; no SGBM rebuild needed."""
        if self.depth_engine is None:
            return
        lo = self.sp_min_depth.value()
        hi = self.sp_max_depth.value()
        if hi <= lo:
            return  # ignore inverted range; user is mid-edit
        # GIL-atomic float assigns — safe without locking.
        self.depth_engine.min_depth_mm = lo
        self.depth_engine.max_depth_mm = hi
        rmin, rmax = self.depth_engine.detection_range()
        self.lbl_range.setText(f"detection range: {rmin:.0f} - {rmax:.0f} mm")

    def show_pixel_info(self, info):
        rmin = info.range_min_mm
        rmax = info.range_max_mm
        self.lbl_range.setText(f"detection range: {rmin:.0f} - {rmax:.0f} mm")
        if not info.valid:
            self.lbl_depth.setText("depth: - (no valid disparity)")
            self.lbl_error.setText("error band: -")
            return
        self.lbl_depth.setText(f"depth: {info.depth_mm:.1f} mm")
        self.lbl_error.setText(
            f"error band: ±{info.error_mm:.1f} mm   "
            f"({info.depth_mm - info.error_mm:.1f} - "
            f"{info.depth_mm + info.error_mm:.1f} mm)"
        )
