#!/usr/bin/env python3
"""Unified launcher for the ELP 3D Stereo Camera tool suite."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QProcess, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPainter, QColor, QLinearGradient, QBrush, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


ROOT_DIR = Path(__file__).resolve().parent


@dataclass
class AppEntry:
    key: str
    title: str
    subtitle: str
    description: str
    script: str
    accent: str  # hex color
    glyph: str   # short text or unicode glyph for the icon tile


APPS: list[AppEntry] = [
    AppEntry(
        key="calib",
        title="Stereo Calibration",
        subtitle="app.py",
        description=(
            "Calibrate the ELP stereo camera with chessboard, ChArUco, or "
            "circle-grid targets. Live depth viewer with SGBM or NVIDIA VPI."
        ),
        script="app.py",
        accent="#4FC3F7",
        glyph="CAL",
    ),
    AppEntry(
        key="detect",
        title="Object Detection",
        subtitle="app_detect.py",
        description=(
            "Tiled YOLO inference on rectified-left frames. Publishes "
            "Detection3DArray + static TF to ROS2."
        ),
        script="app_detect.py",
        accent="#81C784",
        glyph="DET",
    ),
    AppEntry(
        key="handeye",
        title="Hand-Eye Calibration",
        subtitle="app_handeye.py",
        description=(
            "ELP <-> MG400 eye-to-hand calibration. Manual click, ArUco "
            "centroid, ArUco PnP, or auto grid collection."
        ),
        script="app_handeye.py",
        accent="#FFB74D",
        glyph="H-E",
    ),
    AppEntry(
        key="robot",
        title="Robot Pick Control",
        subtitle="app_robot_control.py",
        description=(
            "MG400 pick controller. Consumes /elp/detections, runs "
            "approach/pick/place sequences, manual jog, and auto loop."
        ),
        script="app_robot_control.py",
        accent="#E57373",
        glyph="BOT",
    ),
]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

GLOBAL_QSS = """
QMainWindow, QWidget#root {
    background-color: #1B1F27;
}
QLabel#title {
    color: #FFFFFF;
    font-size: 28px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
QLabel#subtitle {
    color: #8A93A6;
    font-size: 13px;
}
QLabel#footer {
    color: #5A6479;
    font-size: 11px;
}
QFrame#card {
    background-color: #252B36;
    border: 1px solid #2F3645;
    border-radius: 14px;
}
QFrame#card:hover {
    border: 1px solid #4A5468;
    background-color: #2A3140;
}
QLabel#cardTitle {
    color: #FFFFFF;
    font-size: 17px;
    font-weight: 600;
}
QLabel#cardSubtitle {
    color: #7C8499;
    font-size: 11px;
    font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
}
QLabel#cardDesc {
    color: #BCC4D6;
    font-size: 12px;
}
QPushButton#launchBtn {
    color: #FFFFFF;
    background-color: #3B4456;
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton#launchBtn:hover {
    background-color: #4A5468;
}
QPushButton#launchBtn:pressed {
    background-color: #2F3645;
}
QStatusBar {
    background-color: #161A21;
    color: #7C8499;
}
"""


def make_glyph_pixmap(text: str, accent_hex: str, size: int = 64) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)

    grad = QLinearGradient(0, 0, size, size)
    grad.setColorAt(0.0, QColor(accent_hex))
    grad.setColorAt(1.0, QColor(accent_hex).darker(160))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(0, 0, size, size, 14, 14)

    p.setPen(QPen(QColor("#FFFFFF")))
    font = QFont("DejaVu Sans", int(size * 0.28))
    font.setBold(True)
    p.setFont(font)
    p.drawText(pm.rect(), Qt.AlignCenter, text)
    p.end()
    return pm


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class AppCard(QFrame):
    def __init__(self, entry: AppEntry, on_launch, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumHeight(190)

        self.entry = entry
        self._on_launch = on_launch
        self._process: Optional[QProcess] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(14)

        glyph = QLabel()
        glyph.setPixmap(make_glyph_pixmap(entry.glyph, entry.accent))
        glyph.setFixedSize(64, 64)
        header.addWidget(glyph)

        titles = QVBoxLayout()
        titles.setSpacing(2)
        t = QLabel(entry.title)
        t.setObjectName("cardTitle")
        s = QLabel(entry.subtitle)
        s.setObjectName("cardSubtitle")
        titles.addWidget(t)
        titles.addWidget(s)
        titles.addStretch(1)
        header.addLayout(titles, 1)
        layout.addLayout(header)

        desc = QLabel(entry.description)
        desc.setObjectName("cardDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.status_lbl = QLabel("")
        self.status_lbl.setObjectName("cardSubtitle")
        btn_row.addWidget(self.status_lbl)
        btn_row.addSpacing(10)

        self.btn = QPushButton("Launch")
        self.btn.setObjectName("launchBtn")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setMinimumWidth(110)
        self.btn.clicked.connect(self._launch)
        btn_row.addWidget(self.btn)
        layout.addLayout(btn_row)

    def _launch(self):
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            self.status_lbl.setText("already running")
            return

        script_path = ROOT_DIR / self.entry.script
        if not script_path.exists():
            self.status_lbl.setText(f"missing: {self.entry.script}")
            return

        proc = QProcess(self)
        proc.setWorkingDirectory(str(ROOT_DIR))
        proc.setProgram(sys.executable)
        proc.setArguments([str(script_path)])
        proc.setProcessChannelMode(QProcess.ForwardedChannels)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        proc.start()
        self._process = proc

        self.status_lbl.setText("running...")
        self.btn.setText("Running")
        self._on_launch(self.entry, started=True)

    def _on_finished(self, code, status):
        self.status_lbl.setText(f"exited ({code})")
        self.btn.setText("Launch")
        self._on_launch(self.entry, started=False)

    def _on_error(self, err):
        self.status_lbl.setText(f"error: {err}")
        self.btn.setText("Launch")
        self._on_launch(self.entry, started=False)


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ELP 3D Stereo Camera — Launcher")
        self.setMinimumSize(960, 640)

        icon_pm = make_glyph_pixmap("ELP", "#4FC3F7", 128)
        self.setWindowIcon(QIcon(icon_pm))

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(36, 28, 36, 24)
        outer.setSpacing(18)

        # Header
        header = QHBoxLayout()
        header.setSpacing(16)
        logo = QLabel()
        logo.setPixmap(make_glyph_pixmap("ELP", "#4FC3F7", 56))
        logo.setFixedSize(56, 56)
        header.addWidget(logo)

        h_text = QVBoxLayout()
        h_text.setSpacing(2)
        title = QLabel("ELP 3D Stereo Camera")
        title.setObjectName("title")
        subtitle = QLabel(f"App Suite Launcher  -  {ROOT_DIR}")
        subtitle.setObjectName("subtitle")
        h_text.addWidget(title)
        h_text.addWidget(subtitle)
        header.addLayout(h_text, 1)
        outer.addLayout(header)

        # Cards grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(18)

        self.cards: list[AppCard] = []
        for i, entry in enumerate(APPS):
            card = AppCard(entry, self._on_card_event)
            self.cards.append(card)
            grid.addWidget(card, i // 2, i % 2)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        outer.addLayout(grid, 1)

        outer.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        footer = QLabel(
            "Tip: run scripts/install_desktop_icon.sh to add this launcher "
            "to the application menu and desktop."
        )
        footer.setObjectName("footer")
        footer.setAlignment(Qt.AlignCenter)
        outer.addWidget(footer)

        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage(f"Python: {sys.executable}")

        self.setStyleSheet(GLOBAL_QSS)

    def _on_card_event(self, entry: AppEntry, started: bool):
        if started:
            self.statusBar().showMessage(f"Started {entry.script}")
        else:
            self.statusBar().showMessage(f"{entry.script} exited")


def main():
    os.chdir(ROOT_DIR)
    app = QApplication(sys.argv)
    app.setApplicationName("ELP 3D Launcher")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    win = LauncherWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
