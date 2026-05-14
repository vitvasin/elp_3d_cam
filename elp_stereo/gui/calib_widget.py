"""Calibration tab: live capture or folder load, run calibration, save result."""

import glob
import os

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ..calibration import StereoCalibrator, save_yaml
from ..config import project_root
from ..targets import build_target
from .widgets import ImagePanel


class CalibWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.cfg = main_window.cfg
        self.calibrator = None
        self.result = None
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Board type + parameters --------------------------------------
        self.board_combo = QComboBox()
        self.board_combo.addItems(["chessboard", "charuco"])
        self.board_combo.currentTextChanged.connect(self._swap_params)

        cb = self.cfg["calibration"]["chessboard"]
        self.cb_cols = self._spin(2, 30, cb["cols"])
        self.cb_rows = self._spin(2, 30, cb["rows"])
        self.cb_square = self._dspin(1.0, 500.0, cb["square_size_mm"])
        self.cb_box = QGroupBox("Chessboard (inner corners)")
        f = QFormLayout(self.cb_box)
        f.addRow("cols", self.cb_cols)
        f.addRow("rows", self.cb_rows)
        f.addRow("square mm", self.cb_square)

        ch = self.cfg["calibration"]["charuco"]
        self.ch_sx = self._spin(2, 30, ch["squares_x"])
        self.ch_sy = self._spin(2, 30, ch["squares_y"])
        self.ch_square = self._dspin(1.0, 500.0, ch["square_len_mm"])
        self.ch_marker = self._dspin(1.0, 500.0, ch["marker_len_mm"])
        self.ch_dict = QComboBox()
        self.ch_dict.addItems([
            "DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_100", "DICT_6X6_250",
        ])
        self.ch_dict.setCurrentText(ch["dictionary"])
        self.ch_box = QGroupBox("ChArUco")
        f = QFormLayout(self.ch_box)
        f.addRow("squares X", self.ch_sx)
        f.addRow("squares Y", self.ch_sy)
        f.addRow("square mm", self.ch_square)
        f.addRow("marker mm", self.ch_marker)
        f.addRow("dictionary", self.ch_dict)

        layout.addWidget(QLabel("Board type"))
        layout.addWidget(self.board_combo)
        layout.addWidget(self.cb_box)
        layout.addWidget(self.ch_box)
        self._swap_params(self.board_combo.currentText())

        # --- Live capture -------------------------------------------------
        live = QGroupBox("Live capture")
        lv = QVBoxLayout(live)
        self.btn_grab = QPushButton("Grab Pair")
        self.btn_grab.clicked.connect(self.grab_pair)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset)
        row = QHBoxLayout()
        row.addWidget(self.btn_grab)
        row.addWidget(self.btn_reset)
        lv.addLayout(row)
        self.count_label = QLabel("pairs: 0")
        lv.addWidget(self.count_label)
        self.coverage_panel = ImagePanel("coverage")
        self.coverage_panel.setMinimumSize(280, 160)
        self.coverage_panel.setMaximumHeight(180)
        lv.addWidget(self.coverage_panel)
        layout.addWidget(live)

        # --- Folder load --------------------------------------------------
        folder = QGroupBox("Load image folder")
        fo = QVBoxLayout(folder)
        self.btn_folder = QPushButton("Load Folder...")
        self.btn_folder.clicked.connect(self.load_folder)
        fo.addWidget(self.btn_folder)
        fo.addWidget(QLabel("left/ + right/ subdirs, or *_left.* / *_right.*"))
        layout.addWidget(folder)

        # --- Run + save ---------------------------------------------------
        self.btn_calibrate = QPushButton("Calibrate")
        self.btn_calibrate.clicked.connect(self.run_calibration)
        layout.addWidget(self.btn_calibrate)
        self.rms_label = QLabel("RMS: -")
        layout.addWidget(self.rms_label)
        self.btn_save = QPushButton("Save Calibration")
        self.btn_save.clicked.connect(self.save_calibration)
        self.btn_save.setEnabled(False)
        layout.addWidget(self.btn_save)

        layout.addStretch(1)

    def _spin(self, lo, hi, val):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    def _dspin(self, lo, hi, val):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setDecimals(2)
        return s

    def _swap_params(self, kind):
        self.cb_box.setVisible(kind == "chessboard")
        self.ch_box.setVisible(kind == "charuco")

    # -------------------------------------------------------------- target
    def _current_target(self):
        kind = self.board_combo.currentText()
        cfg = {
            "chessboard": {
                "cols": self.cb_cols.value(),
                "rows": self.cb_rows.value(),
                "square_size_mm": self.cb_square.value(),
            },
            "charuco": {
                "squares_x": self.ch_sx.value(),
                "squares_y": self.ch_sy.value(),
                "square_len_mm": self.ch_square.value(),
                "marker_len_mm": self.ch_marker.value(),
                "dictionary": self.ch_dict.currentText(),
            },
        }
        return build_target(kind, cfg)

    def _ensure_calibrator(self, image_size):
        if self.calibrator is None:
            self.calibrator = StereoCalibrator(image_size)

    # --------------------------------------------------------- live capture
    def grab_pair(self):
        raw = self.main.latest_raw
        if raw is None:
            self.count_label.setText("pairs: 0  (no camera frame)")
            return
        left, right = raw
        h, w = left.shape[:2]
        self._ensure_calibrator((w, h))
        target = self._current_target()
        det_l = target.detect(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
        det_r = target.detect(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
        n = self.calibrator.add_pair(det_l, det_r)
        if n == 0:
            self.count_label.setText(
                f"pairs: {self.calibrator.pair_count}  (board not in both views)"
            )
            return
        self.count_label.setText(
            f"pairs: {self.calibrator.pair_count}  (+{n} corners)"
        )
        self._update_coverage()

    def reset(self):
        if self.calibrator is not None:
            self.calibrator.reset()
        self.result = None
        self.btn_save.setEnabled(False)
        self.count_label.setText("pairs: 0")
        self.rms_label.setText("RMS: -")
        self.coverage_panel.setText("coverage")

    def _update_coverage(self):
        heat = self.calibrator.coverage_heatmap()
        color = cv2.applyColorMap(heat, cv2.COLORMAP_HOT)
        self.coverage_panel.show_image(color)

    # --------------------------------------------------------- folder load
    def load_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not path:
            return
        pairs = self._collect_pairs(path)
        if not pairs:
            self.count_label.setText("pairs: 0  (no L/R pairs found)")
            return
        cam = self.cfg["camera"]
        expected = (cam["frame_width"] // 2, cam["frame_height"])
        target = self._current_target()
        added = 0
        rejected_size = 0
        for lp, rp in pairs:
            left = cv2.imread(lp)
            right = cv2.imread(rp)
            if left is None or right is None:
                continue
            h, w = left.shape[:2]
            if (w, h) != expected:
                # Rectify maps built from a mismatched size would silently
                # corrupt live frames -- skip these pairs.
                rejected_size += 1
                continue
            self._ensure_calibrator((w, h))
            det_l = target.detect(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
            det_r = target.detect(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
            if self.calibrator.add_pair(det_l, det_r) > 0:
                added += 1
        msg = f"pairs: {self.calibrator.pair_count if self.calibrator else 0}  ({added} from folder)"
        if rejected_size:
            msg += f"  -  {rejected_size} skipped: size != {expected[0]}x{expected[1]}"
        self.count_label.setText(msg)
        if self.calibrator is not None and self.calibrator.pair_count:
            self._update_coverage()

    @staticmethod
    def _collect_pairs(path):
        left_dir = os.path.join(path, "left")
        right_dir = os.path.join(path, "right")
        pairs = []
        if os.path.isdir(left_dir) and os.path.isdir(right_dir):
            for lp in sorted(glob.glob(os.path.join(left_dir, "*"))):
                rp = os.path.join(right_dir, os.path.basename(lp))
                if os.path.isfile(rp):
                    pairs.append((lp, rp))
            return pairs
        for lp in sorted(glob.glob(os.path.join(path, "*_left.*"))):
            rp = lp.replace("_left.", "_right.")
            if os.path.isfile(rp):
                pairs.append((lp, rp))
        return pairs

    # ------------------------------------------------------------ calibrate
    def run_calibration(self):
        if self.calibrator is None or self.calibrator.pair_count < 5:
            self.rms_label.setText("RMS: need >= 5 pairs")
            return
        try:
            self.result = self.calibrator.calibrate()
        except Exception as exc:  # noqa: BLE001
            self.rms_label.setText(f"RMS: error - {exc}")
            return
        self.rms_label.setText(f"RMS: {self.result.rms:.4f} px")
        self.btn_save.setEnabled(True)
        self.main.apply_calibration(self.result)
        self.main.status.setText("Calibration applied")

    def save_calibration(self):
        if self.result is None:
            return
        out = project_root() / self.cfg["calibration"]["output_path"]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        save_yaml(out, self.result)
        self.main.status.setText(f"Saved calibration: {out}")
