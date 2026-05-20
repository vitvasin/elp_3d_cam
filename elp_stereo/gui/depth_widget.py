"""Depth tab: backend controls and the pixel-click depth readout.

Spinboxes are used instead of raw sliders because stereo matcher parameters
have hard constraints -- disparity range must be a multiple of 16 and window
size must be odd -- which a free slider cannot express cleanly.
"""

import cv2
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
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
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # --- Common depth/display controls -------------------------------
        common_box = QGroupBox("Common depth / readout")
        common_form = QFormLayout(common_box)
        self.cmap_combo = QComboBox()
        for label, _ in _COLORMAPS:
            self.cmap_combo.addItem(label)
        self.cmap_combo.setCurrentText("TURBO")
        self.cmap_combo.currentIndexChanged.connect(self._on_colormap_changed)

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

        self.sp_temporal = self._spin(
            1, 32, 1, int(self.cfg["depth"].get("temporal_frames", 1))
        )
        self.sp_sample_radius = self._spin(
            0, 25, 1, int(self.cfg["depth"].get("sample_radius_px", 3))
        )
        self.sp_sample_trim = QDoubleSpinBox()
        self.sp_sample_trim.setRange(0.0, 0.45)
        self.sp_sample_trim.setDecimals(2)
        self.sp_sample_trim.setSingleStep(0.05)
        self.sp_sample_trim.setValue(float(self.cfg["depth"].get("sample_trim", 0.2)))

        self._add_row(common_form, "colormap", self.cmap_combo,
                      "Color palette used to render the depth map.")
        self._add_row(common_form, "min depth", self.sp_min_depth,
                      "Depths nearer than this are masked invalid.")
        self._add_row(common_form, "max depth", self.sp_max_depth,
                      "Depths farther than this are masked invalid.")
        self._add_row(common_form, "temporal frames", self.sp_temporal,
                      "Average this many recent depth frames. Higher is smoother but adds latency.")
        self._add_row(common_form, "sample radius px", self.sp_sample_radius,
                      "Radius of the local window used for clicked depth/disparity readout.")
        self._add_row(common_form, "sample trim", self.sp_sample_trim,
                      "Fraction of low/high local samples discarded before median depth.")
        layout.addWidget(common_box)

        s = self.cfg["depth"]["sgbm"]
        v = self.cfg["depth"].get("vpi", {})
        self.vpi_box = QGroupBox("VPI parameters")
        vpi_form = QFormLayout(self.vpi_box)
        self.cmb_vpi_backend = QComboBox()
        for backend in ("OFA", "CUDA"):
            self.cmb_vpi_backend.addItem(backend)
        self.cmb_vpi_backend.setCurrentText(str(v.get("backend", "OFA")).upper())
        self.sp_vpi_quality = self._spin(1, 8, 1, int(v.get("quality", 6)))
        self.sp_vpi_maxdisp = self._spin(
            16, 512, 16, int(v.get("max_disparity", s["num_disparities"]))
        )
        self.sp_vpi_window = self._spin(
            3, 21, 2, int(v.get("window_size", s["block_size"]))
        )
        self.sp_vpi_min_disp = self._spin(0, 512, 1, int(v.get("min_disparity", 0)))
        self.sp_vpi_conf = self._spin(0, 65280, 256, int(v.get("confthreshold", 32767)))
        self.sp_vpi_p1 = self._spin(0, 4096, 1, int(v.get("p1", 3)))
        self.sp_vpi_p2 = self._spin(0, 8192, 1, int(v.get("p2", 48)))
        self.sp_vpi_p2alpha = self._spin(0, 4096, 1, int(v.get("p2alpha", 0)))
        self.sp_vpi_uniqueness = QDoubleSpinBox()
        self.sp_vpi_uniqueness.setRange(-1.0, 100.0)
        self.sp_vpi_uniqueness.setDecimals(2)
        self.sp_vpi_uniqueness.setSingleStep(0.5)
        self.sp_vpi_uniqueness.setValue(float(v.get("uniqueness", -1.0)))
        self.chk_vpi_diag = QCheckBox()
        self.chk_vpi_diag.setChecked(bool(v.get("include_diagonals", True)))
        self.sp_vpi_passes = self._spin(1, 8, 1, int(v.get("num_passes", 3)))
        self._add_row(vpi_form, "backend", self.cmb_vpi_backend,
                      "VPI execution engine. OFA uses dedicated Jetson stereo hardware; CUDA uses GPU compute.")
        self._add_row(vpi_form, "quality", self.sp_vpi_quality,
                      "VPI quality level 1..8. Higher may improve disparity but can be slower.")
        self._add_row(vpi_form, "max disparity", self.sp_vpi_maxdisp,
                      "Maximum horizontal search distance in pixels. Must cover near objects.")
        self._add_row(vpi_form, "window size", self.sp_vpi_window,
                      "Matching/median window size. Larger smooths noise but loses detail at edges.")
        self._add_row(vpi_form, "min disparity", self.sp_vpi_min_disp,
                      "Minimum disparity. Usually 0; mainly useful with CUDA or shifted rectification.")
        self._add_row(vpi_form, "confidence threshold", self.sp_vpi_conf,
                      "Reject disparities below this VPI confidence threshold. Lower keeps more pixels; higher is stricter.")
        self._add_row(vpi_form, "p1", self.sp_vpi_p1,
                      "Penalty for +/-1 disparity changes between neighbors. Lower follows slanted surfaces more.")
        self._add_row(vpi_form, "p2", self.sp_vpi_p2,
                      "Penalty for larger disparity jumps. Higher smooths surfaces but can blur depth edges.")
        self._add_row(vpi_form, "p2 alpha", self.sp_vpi_p2alpha,
                      "Adaptive P2 control. 0 disables adaptive large-penalty behavior.")
        self._add_row(vpi_form, "uniqueness", self.sp_vpi_uniqueness,
                      "Ambiguity filter. -1 disables; higher rejects less-distinct matches.")
        self._add_row(vpi_form, "include diagonals", self.chk_vpi_diag,
                      "Include diagonal/oblique SGM paths. Can improve smoothness at extra cost.")
        self._add_row(vpi_form, "num passes", self.sp_vpi_passes,
                      "Number of stereo aggregation passes when supported by the installed VPI version.")
        layout.addWidget(self.vpi_box)

        self.sgbm_box = QGroupBox("SGBM parameters")
        sgbm_form = QFormLayout(self.sgbm_box)
        self.sp_sgbm_min_disp = self._spin(0, 512, 1, s["min_disparity"])
        self.sp_sgbm_num_disp = self._spin(16, 512, 16, s["num_disparities"])
        self.sp_sgbm_block = self._spin(1, 21, 2, s["block_size"])
        self.sp_uniq = self._spin(0, 50, 1, s["uniqueness_ratio"])
        self.sp_speckle_win = self._spin(0, 300, 10, s["speckle_window_size"])
        self.sp_speckle_range = self._spin(0, 10, 1, s["speckle_range"])
        self.sp_disp12 = self._spin(-1, 50, 1, s["disp12_max_diff"])
        self._add_row(sgbm_form, "min disparity", self.sp_sgbm_min_disp,
                      "OpenCV SGBM minimum disparity. Usually 0 for rectified ELP stereo.")
        self._add_row(sgbm_form, "num disparities", self.sp_sgbm_num_disp,
                      "OpenCV SGBM disparity search range. Must be a multiple of 16.")
        self._add_row(sgbm_form, "block size", self.sp_sgbm_block,
                      "OpenCV SGBM matching block size. Larger smooths noise but loses fine detail.")
        self._add_row(sgbm_form, "uniqueness ratio", self.sp_uniq,
                      "Reject ambiguous SGBM matches. Higher is stricter and may create more invalid pixels.")
        self._add_row(sgbm_form, "speckle window", self.sp_speckle_win,
                      "Maximum connected speckle region size to remove. 0 disables.")
        self._add_row(sgbm_form, "speckle range", self.sp_speckle_range,
                      "Allowed disparity variation inside speckle filtering regions.")
        self._add_row(sgbm_form, "disp12 max diff", self.sp_disp12,
                      "Left-right consistency check threshold. -1 disables.")
        self.chk_wls = QCheckBox("WLS post-filter (ximgproc)")
        if WLS_AVAILABLE:
            self.chk_wls.setChecked(self.cfg["depth"]["use_wls_filter"])
        else:
            self.chk_wls.setChecked(False)
            self.chk_wls.setEnabled(False)
            self.chk_wls.setText("WLS post-filter (ximgproc not installed)")
        self.chk_wls.setToolTip("OpenCV ximgproc WLS disparity post-filter. Can smooth SGBM output if ximgproc is installed.")
        sgbm_form.addRow(self.chk_wls)
        layout.addWidget(self.sgbm_box)

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

        match_box = QGroupBox("Stereo match (click left/right RGB)")
        mv = QVBoxLayout(match_box)
        self.lbl_match_left = QLabel("left: -")
        self.lbl_match_right = QLabel("right: -")
        self.lbl_match_disp = QLabel("disparity: -")
        for lbl in (self.lbl_match_left, self.lbl_match_right, self.lbl_match_disp):
            mv.addWidget(lbl)
        layout.addWidget(match_box)

        layout.addStretch(1)

    def _spin(self, lo, hi, step, val):
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setValue(val)
        return sp

    def _add_row(self, form, label_text, widget, tooltip):
        label = QLabel(label_text)
        label.setToolTip(tooltip)
        widget.setToolTip(tooltip)
        form.addRow(label, widget)

    def set_depth_engine(self, engine):
        self.depth_engine = engine
        if engine is None:
            self.setEnabled(False)
            return
        self.setEnabled(True)
        self._sync_backend_groups()
        # Push current widget state into the fresh engine (calibration load
        # must not reset user-tuned params or colormap choice).
        self.apply_params()
        self._on_colormap_changed()
        self._on_depth_range_changed()

    def apply_params(self):
        if self.depth_engine is None:
            return
        params = self.sync_config_from_ui()
        self.depth_engine.update_params(params)
        rmin, rmax = self.depth_engine.detection_range()
        self.lbl_range.setText(
            f"detection range: {rmin:.0f} - {rmax:.0f} mm"
        )

    def sync_config_from_ui(self):
        """Copy current widget values into ``self.cfg`` and return engine params."""
        vpi_window = self._odd_value(self.sp_vpi_window)
        sgbm_block = self._odd_value(self.sp_sgbm_block)
        engine_name = self.cfg.get("depth", {}).get("engine", "sgbm").lower()
        if engine_name == "vpi":
            min_disp = int(self.sp_vpi_min_disp.value())
            num_disp = self._multiple_of_16(self.sp_vpi_maxdisp)
            block = vpi_window
        else:
            min_disp = self.sp_sgbm_min_disp.value()
            num_disp = self._multiple_of_16(self.sp_sgbm_num_disp)
            block = sgbm_block
        depth_cfg = self.cfg.setdefault("depth", {})
        depth_cfg["min_depth_mm"] = float(self.sp_min_depth.value())
        depth_cfg["max_depth_mm"] = float(self.sp_max_depth.value())
        depth_cfg["temporal_frames"] = int(self.sp_temporal.value())
        depth_cfg["sample_radius_px"] = int(self.sp_sample_radius.value())
        depth_cfg["sample_trim"] = float(self.sp_sample_trim.value())
        depth_cfg["use_wls_filter"] = bool(self.chk_wls.isChecked())
        vpi_cfg = depth_cfg.setdefault("vpi", {})
        vpi_cfg["backend"] = self.cmb_vpi_backend.currentText()
        vpi_cfg["quality"] = int(self.sp_vpi_quality.value())
        vpi_cfg["max_disparity"] = int(self._multiple_of_16(self.sp_vpi_maxdisp))
        vpi_cfg["window_size"] = int(vpi_window)
        vpi_cfg["min_disparity"] = int(self.sp_vpi_min_disp.value())
        vpi_cfg["confthreshold"] = int(self.sp_vpi_conf.value())
        vpi_cfg["p1"] = int(self.sp_vpi_p1.value())
        vpi_cfg["p2"] = int(self.sp_vpi_p2.value())
        vpi_cfg["p2alpha"] = int(self.sp_vpi_p2alpha.value())
        vpi_cfg["uniqueness"] = float(self.sp_vpi_uniqueness.value())
        vpi_cfg["include_diagonals"] = bool(self.chk_vpi_diag.isChecked())
        vpi_cfg["num_passes"] = int(self.sp_vpi_passes.value())
        depth_cfg.setdefault("sgbm", {}).update({
            "min_disparity": int(self.sp_sgbm_min_disp.value()),
            "num_disparities": int(self._multiple_of_16(self.sp_sgbm_num_disp)),
            "block_size": int(sgbm_block),
            "uniqueness_ratio": int(self.sp_uniq.value()),
            "speckle_window_size": int(self.sp_speckle_win.value()),
            "speckle_range": int(self.sp_speckle_range.value()),
            "disp12_max_diff": int(self.sp_disp12.value()),
            "p1": None,
            "p2": None,
        })
        params = {
            "sgbm": {
                "min_disparity": min_disp,
                "num_disparities": num_disp,
                "block_size": block,
                "uniqueness_ratio": self.sp_uniq.value(),
                "speckle_window_size": self.sp_speckle_win.value(),
                "speckle_range": self.sp_speckle_range.value(),
                "disp12_max_diff": self.sp_disp12.value(),
                "p1": None,
                "p2": None,
            },
            "vpi": {
                "backend": self.cmb_vpi_backend.currentText(),
                "quality": self.sp_vpi_quality.value(),
                "max_disparity": self.sp_vpi_maxdisp.value(),
                "window_size": vpi_window,
                "min_disparity": self.sp_vpi_min_disp.value(),
                "confthreshold": self.sp_vpi_conf.value(),
                "p1": self.sp_vpi_p1.value(),
                "p2": self.sp_vpi_p2.value(),
                "p2alpha": self.sp_vpi_p2alpha.value(),
                "uniqueness": self.sp_vpi_uniqueness.value(),
                "include_diagonals": self.chk_vpi_diag.isChecked(),
                "num_passes": self.sp_vpi_passes.value(),
            },
            "use_wls_filter": self.chk_wls.isChecked(),
            "temporal_frames": self.sp_temporal.value(),
            "sample_radius_px": self.sp_sample_radius.value(),
            "sample_trim": self.sp_sample_trim.value(),
        }
        return params

    def _sync_backend_groups(self):
        engine_name = self.cfg.get("depth", {}).get("engine", "sgbm").lower()
        vpi_active = engine_name == "vpi"
        self.vpi_box.setEnabled(vpi_active)
        self.sgbm_box.setEnabled(not vpi_active)
        self.vpi_box.setTitle("VPI parameters" + (" (active)" if vpi_active else ""))
        self.sgbm_box.setTitle("SGBM parameters" + (" (active)" if not vpi_active else ""))

    def _odd_value(self, spinbox):
        val = spinbox.value()
        if val % 2 == 0:
            val += 1
            spinbox.setValue(val)
        return val

    def _multiple_of_16(self, spinbox):
        val = max(16, int(round(spinbox.value() / 16)) * 16)
        if val != spinbox.value():
            spinbox.setValue(val)
        return val

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
        self.cfg.setdefault("depth", {})["min_depth_mm"] = float(lo)
        self.cfg.setdefault("depth", {})["max_depth_mm"] = float(hi)
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
            f"{info.depth_mm + info.error_mm:.1f} mm)   "
            f"local σ={info.sample_std_mm:.2f} mm n={info.sample_count}"
        )

    def show_stereo_match(self, left_xy, right_xy, disparity_px, valid):
        if not valid:
            self.lbl_match_left.setText("left: -")
            self.lbl_match_right.setText("right: -")
            self.lbl_match_disp.setText("disparity: - (no valid match)")
            return
        lx, ly = left_xy
        rx, ry = right_xy
        self.lbl_match_left.setText(f"left: ({lx}, {ly})")
        self.lbl_match_right.setText(f"right: ({rx}, {ry})")
        self.lbl_match_disp.setText(f"disparity: {disparity_px:.2f} px")
