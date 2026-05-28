#!/usr/bin/env python3
"""Unified launcher for the ELP 3D Stereo Camera tool suite."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QProcess, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
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
WORM_IMG = ROOT_DIR / "assets" / "worm.png"

# Neon palette
NEON_CYAN = "#00E5FF"
NEON_GREEN = "#3DF5A1"
NEON_AMBER = "#FFC24B"
NEON_RED = "#FF4D6D"
WORM_BODY = "#5CFF8F"
WORM_TIP = "#00E5FF"


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
        accent=NEON_CYAN,
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
        accent=NEON_GREEN,
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
        accent=NEON_AMBER,
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
        accent=NEON_RED,
        glyph="BOT",
    ),
]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

GLOBAL_QSS = """
QMainWindow {
    background-color: #06090F;
}
QLabel#title {
    color: #EAFEFF;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 2px;
    font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
}
QLabel#subtitle {
    color: #4FD8E8;
    font-size: 12px;
    letter-spacing: 1px;
    font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
}
QLabel#footer {
    color: #3C586A;
    font-size: 11px;
    letter-spacing: 1px;
    font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
}
QFrame#card {
    background-color: rgba(13, 22, 33, 0.85);
    border: 1px solid #16303C;
    border-radius: 14px;
}
QFrame#card:hover {
    border: 1px solid #00E5FF;
    background-color: rgba(16, 30, 44, 0.95);
}
QLabel#cardTitle {
    color: #EAFEFF;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
QLabel#cardSubtitle {
    color: #4FD8E8;
    font-size: 11px;
    font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
}
QLabel#cardDesc {
    color: #9FB8C6;
    font-size: 12px;
}
QPushButton#launchBtn {
    color: #06090F;
    background-color: #00E5FF;
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}
QPushButton#launchBtn:hover {
    background-color: #5CF6FF;
}
QPushButton#launchBtn:pressed {
    background-color: #00B4CC;
}
QPushButton#launchAll {
    color: #06090F;
    background-color: #3DF5A1;
    border: none;
    border-radius: 10px;
    padding: 10px 26px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 2px;
    font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
}
QPushButton#launchAll:hover {
    background-color: #6BFFBC;
}
QPushButton#launchAll:pressed {
    background-color: #21C77F;
}
QStatusBar {
    background-color: #04070B;
    color: #3FE0B0;
    font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
}
"""


def _glow(widget: QWidget, color_hex: str, radius: int = 24) -> None:
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(radius)
    eff.setColor(QColor(color_hex))
    eff.setOffset(0, 0)
    widget.setGraphicsEffect(eff)


def worm_pixmap(size: int = 72) -> QPixmap:
    """Worm mascot: load assets/worm.png, fall back to procedural drawing."""
    if WORM_IMG.exists():
        pm = QPixmap(str(WORM_IMG))
        if not pm.isNull():
            return pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return make_worm_pixmap(size)


def make_worm_pixmap(size: int = 72) -> QPixmap:
    """Procedural neon worm mascot — segmented body on a sine curve."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)

    margin = size * 0.16
    span = size - 2 * margin
    n = 28
    pts = []
    for i in range(n + 1):
        t = i / n
        x = margin + t * span
        y = size * 0.5 + math.sin(t * math.pi * 1.6) * size * 0.18
        pts.append(QPointF(x, y))

    path = QPainterPath(pts[0])
    for q in pts[1:]:
        path.lineTo(q)

    body_w = size * 0.20

    # Fake outer glow: stroke several translucent passes, widest first.
    for w_mul, alpha in ((2.4, 26), (1.8, 40), (1.35, 70)):
        glow = QColor(WORM_TIP)
        glow.setAlpha(alpha)
        gp = QPen(glow, body_w * w_mul)
        gp.setCapStyle(Qt.RoundCap)
        gp.setJoinStyle(Qt.RoundJoin)
        p.setPen(gp)
        p.drawPath(path)

    # Main body with green->cyan gradient.
    grad = QLinearGradient(pts[0], pts[-1])
    grad.setColorAt(0.0, QColor(WORM_TIP))
    grad.setColorAt(0.5, QColor(WORM_BODY))
    grad.setColorAt(1.0, QColor(WORM_TIP))
    body = QPen(QBrush(grad), body_w)
    body.setCapStyle(Qt.RoundCap)
    body.setJoinStyle(Qt.RoundJoin)
    p.setPen(body)
    p.drawPath(path)

    # Segment ticks.
    seg = QColor("#06200F")
    seg.setAlpha(150)
    p.setPen(QPen(seg, max(1, size // 48)))
    for i in range(3, n, 3):
        a, b = pts[i - 1], pts[i + 1]
        dx, dy = b.x() - a.x(), b.y() - a.y()
        ln = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / ln, dx / ln
        c = pts[i]
        r = body_w * 0.42
        p.drawLine(QPointF(c.x() - nx * r, c.y() - ny * r),
                   QPointF(c.x() + nx * r, c.y() + ny * r))

    # Head + eyes at the right tip.
    head = pts[-1]
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(WORM_TIP))
    hr = body_w * 0.62
    p.drawEllipse(head, hr, hr)
    eye_r = hr * 0.30
    eo = hr * 0.34
    for ey in (-eo, eo):
        ec = QPointF(head.x() + eo * 0.4, head.y() + ey)
        p.setBrush(QColor("#06090F"))
        p.drawEllipse(ec, eye_r, eye_r)
        p.setBrush(QColor("#EAFEFF"))
        p.drawEllipse(QPointF(ec.x() + eye_r * 0.35, ec.y() - eye_r * 0.35),
                      eye_r * 0.45, eye_r * 0.45)
    p.end()
    return pm


def make_glyph_pixmap(text: str, accent_hex: str, size: int = 64) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)

    grad = QLinearGradient(0, 0, size, size)
    grad.setColorAt(0.0, QColor(accent_hex))
    grad.setColorAt(1.0, QColor(accent_hex).darker(220))
    p.setBrush(QBrush(grad))
    p.setPen(QPen(QColor(accent_hex).lighter(140), 1.5))
    p.drawRoundedRect(QRectF(1, 1, size - 2, size - 2), 14, 14)

    p.setPen(QPen(QColor("#06090F")))
    font = QFont("DejaVu Sans Mono", int(size * 0.26))
    font.setBold(True)
    p.setFont(font)
    p.drawText(pm.rect(), Qt.AlignCenter, text)
    p.end()
    return pm


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class GridBackground(QWidget):
    """Futuristic dark canvas: faint grid + corner glow vignette."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("root")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.fillRect(self.rect(), QColor("#06090F"))

        # Radial glow top-left and bottom-right.
        for cx, cy, col in ((w * 0.12, h * 0.08, QColor(0, 229, 255, 38)),
                            (w * 0.92, h * 0.95, QColor(61, 245, 161, 28))):
            rg = QRadialGradient(cx, cy, max(w, h) * 0.6)
            rg.setColorAt(0.0, col)
            rg.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.fillRect(self.rect(), QBrush(rg))

        # Grid lines.
        step = 42
        grid = QColor(0, 229, 255, 16)
        p.setPen(QPen(grid, 1))
        x = 0
        while x < w:
            p.drawLine(x, 0, x, h)
            x += step
        y = 0
        while y < h:
            p.drawLine(0, y, w, y)
            y += step
        p.end()


class AppCard(QFrame):
    def __init__(self, entry: AppEntry, on_launch, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumHeight(190)

        self.entry = entry
        self._on_launch = on_launch
        self._process: Optional[QProcess] = None

        _glow(self, entry.accent, radius=18)

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

        self.btn = QPushButton("LAUNCH")
        self.btn.setObjectName("launchBtn")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setMinimumWidth(110)
        self.btn.clicked.connect(self.launch)
        btn_row.addWidget(self.btn)
        layout.addLayout(btn_row)

    def is_running(self) -> bool:
        return self._process is not None and self._process.state() != QProcess.NotRunning

    def launch(self):
        if self.is_running():
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
        self.btn.setText("RUNNING")
        self._on_launch(self.entry, started=True)

    def _on_finished(self, code, status):
        self.status_lbl.setText(f"exited ({code})")
        self.btn.setText("LAUNCH")
        self._on_launch(self.entry, started=False)

    def _on_error(self, err):
        self.status_lbl.setText(f"error: {err}")
        self.btn.setText("LAUNCH")
        self._on_launch(self.entry, started=False)


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Worm Sorter — Launcher")
        self.setMinimumSize(980, 680)

        self.setWindowIcon(QIcon(worm_pixmap(128)))

        root = GridBackground()
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(40, 30, 40, 24)
        outer.setSpacing(20)

        # Header
        header = QHBoxLayout()
        header.setSpacing(18)
        logo = QLabel()
        logo.setPixmap(worm_pixmap(72))
        logo.setFixedSize(72, 72)
        _glow(logo, NEON_GREEN, radius=28)
        header.addWidget(logo)

        h_text = QVBoxLayout()
        h_text.setSpacing(3)
        title = QLabel("WORM SORTER")
        title.setObjectName("title")
        _glow(title, NEON_CYAN, radius=18)
        subtitle = QLabel(f"// ELP 3D STEREO · APP SUITE LAUNCHER  ·  {ROOT_DIR}")
        subtitle.setObjectName("subtitle")
        h_text.addWidget(title)
        h_text.addWidget(subtitle)
        header.addLayout(h_text, 1)

        self.launch_all_btn = QPushButton("LAUNCH ALL")
        self.launch_all_btn.setObjectName("launchAll")
        self.launch_all_btn.setCursor(Qt.PointingHandCursor)
        self.launch_all_btn.clicked.connect(self.launch_all)
        _glow(self.launch_all_btn, NEON_GREEN, radius=22)
        header.addWidget(self.launch_all_btn, 0, Qt.AlignTop)
        outer.addLayout(header)

        # Cards grid
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(20)

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
            "TIP: run scripts/install_desktop_icon.sh to add this launcher "
            "to the application menu and desktop."
        )
        footer.setObjectName("footer")
        footer.setAlignment(Qt.AlignCenter)
        outer.addWidget(footer)

        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage(f"PYTHON · {sys.executable}")

        self.setStyleSheet(GLOBAL_QSS)

    def launch_all(self):
        started = 0
        for card in self.cards:
            if not card.is_running():
                card.launch()
                started += 1
        self.statusBar().showMessage(f"Launched {started} app(s)")

    def _on_card_event(self, entry: AppEntry, started: bool):
        if started:
            self.statusBar().showMessage(f"Started {entry.script}")
        else:
            self.statusBar().showMessage(f"{entry.script} exited")


def main():
    os.chdir(ROOT_DIR)
    app = QApplication(sys.argv)
    app.setApplicationName("Worm Sorter")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    win = LauncherWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
