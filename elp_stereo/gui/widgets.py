"""Shared GUI widgets."""

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy


def ndarray_to_qpixmap(img):
    """Convert a BGR or grayscale uint8 ndarray to a QPixmap."""
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ImagePanel(QLabel):
    """Displays an ndarray scaled to fit, and reports clicks in image coords."""

    clicked = pyqtSignal(int, int)  # (x, y) in source-image pixels

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#202020; color:#aaa;")
        self.setText(title or "no signal")
        self._src_size = None       # (w, h) of the source image
        self._draw_rect = None      # (x0, y0, w, h) of the pixmap inside label

    def show_image(self, img):
        """Display a BGR/grayscale ndarray, scaled to fit while keeping aspect."""
        if img is None:
            return
        h, w = img.shape[:2]
        self._src_size = (w, h)
        pix = ndarray_to_qpixmap(np.ascontiguousarray(img))
        scaled = pix.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        sw, sh = scaled.width(), scaled.height()
        x0 = (self.width() - sw) // 2
        y0 = (self.height() - sh) // 2
        self._draw_rect = (x0, y0, sw, sh)
        self.setPixmap(scaled)

    def mousePressEvent(self, event):
        if self._draw_rect is None or self._src_size is None:
            return
        x0, y0, sw, sh = self._draw_rect
        px = event.x() - x0
        py = event.y() - y0
        if not (0 <= px < sw and 0 <= py < sh):
            return
        sx = int(px / sw * self._src_size[0])
        sy = int(py / sh * self._src_size[1])
        self.clicked.emit(sx, sy)
