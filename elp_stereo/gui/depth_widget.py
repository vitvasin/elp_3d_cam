"""Depth tab: SGBM parameter controls and the pixel-click depth readout.

Spinboxes are used instead of raw sliders because two SGBM parameters have hard
constraints -- numDisparities must be a multiple of 16 and blockSize must be odd
-- which a free slider cannot express cleanly.
"""

from PyQt5.QtWidgets import (
    QCheckBox, QFormLayout, QGroupBox, QLabel, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..depth import WLS_AVAILABLE


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

        s = self.cfg["depth"]["sgbm"]
        params = QGroupBox("StereoSGBM parameters")
        form = QFormLayout(params)

        self.sp_min_disp = self._spin(0, 256, 1, s["min_disparity"])
        self.sp_num_disp = self._spin(16, 512, 16, s["num_disparities"])
        self.sp_block = self._spin(1, 21, 2, s["block_size"])
        self.sp_uniq = self._spin(0, 50, 1, s["uniqueness_ratio"])
        self.sp_speckle_win = self._spin(0, 300, 10, s["speckle_window_size"])
        self.sp_speckle_range = self._spin(0, 10, 1, s["speckle_range"])
        self.sp_disp12 = self._spin(-1, 50, 1, s["disp12_max_diff"])

        form.addRow("min disparity", self.sp_min_disp)
        form.addRow("num disparities", self.sp_num_disp)
        form.addRow("block size", self.sp_block)
        form.addRow("uniqueness ratio", self.sp_uniq)
        form.addRow("speckle window", self.sp_speckle_win)
        form.addRow("speckle range", self.sp_speckle_range)
        form.addRow("disp12 max diff", self.sp_disp12)

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
        self.setEnabled(True)
        # Push the current slider values into the fresh engine so a calibration
        # load (including startup autoload) does not reset user-tuned params.
        self.apply_params()

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
        }
        self.depth_engine.update_params(params)
        rmin, rmax = self.depth_engine.detection_range()
        self.lbl_range.setText(
            f"detection range: {rmin:.0f} - {rmax:.0f} mm"
        )

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
