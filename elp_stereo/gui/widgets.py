"""Shared GUI widgets."""

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
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
        self._title = title or ""
        self._src_size = None       # (w, h) of the source image
        self._draw_rect = None      # (x0, y0, w, h) of the pixmap inside label
        self._pixmap_src = None
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._drag_start = None
        self._drag_pan = None

    def show_image(self, img):
        """Display a BGR/grayscale ndarray, scaled to fit while keeping aspect."""
        if img is None:
            return
        h, w = img.shape[:2]
        self._src_size = (w, h)
        self._pixmap_src = ndarray_to_qpixmap(np.ascontiguousarray(img))
        self._update_draw_rect()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
            self._drag_pan = tuple(self._pan)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is None or self._drag_pan is None:
            return
        delta = event.pos() - self._drag_start
        if self._zoom > 1.0:
            self._pan[0] = self._drag_pan[0] + delta.x()
            self._pan[1] = self._drag_pan[1] + delta.y()
            self._clamp_pan()
            self._update_draw_rect()
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._drag_start is None:
            super().mouseReleaseEvent(event)
            return
        moved = event.pos() - self._drag_start
        self._drag_start = None
        self._drag_pan = None
        if abs(moved.x()) <= 3 and abs(moved.y()) <= 3:
            self._emit_click(event.x(), event.y())
        event.accept()

    def wheelEvent(self, event):
        if self._pixmap_src is None:
            return
        before = self._widget_to_image(event.x(), event.y())
        step = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.set_zoom(self._zoom * step, anchor=(event.x(), event.y()), image_anchor=before)
        event.accept()

    def resizeEvent(self, event):
        self._clamp_pan()
        self._update_draw_rect()
        super().resizeEvent(event)

    def paintEvent(self, event):
        if self._pixmap_src is None or self._draw_rect is None:
            super().paintEvent(event)
            self._draw_title_overlay(QPainter(self))
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        x0, y0, sw, sh = self._draw_rect
        painter.drawPixmap(x0, y0, sw, sh, self._pixmap_src)
        self._draw_title_overlay(painter)

    def _draw_title_overlay(self, painter):
        if not self._title:
            return
        painter.save()
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad_x = 8
        pad_y = 5
        text_w = metrics.horizontalAdvance(self._title)
        text_h = metrics.height()
        rect_w = text_w + pad_x * 2
        rect_h = text_h + pad_y * 2
        painter.fillRect(8, 8, rect_w, rect_h, QColor(0, 0, 0, 170))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(8 + pad_x, 8 + pad_y + metrics.ascent(), self._title)
        painter.restore()

    def zoom_in(self):
        self.set_zoom(self._zoom * 1.25)

    def zoom_out(self):
        self.set_zoom(self._zoom * 0.8)

    def reset_zoom(self):
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._update_draw_rect()
        self.update()

    def set_zoom(self, zoom, anchor=None, image_anchor=None):
        old_anchor = image_anchor
        if old_anchor is None and anchor is not None:
            old_anchor = self._widget_to_image(anchor[0], anchor[1])
        self._zoom = max(1.0, min(zoom, 12.0))
        self._update_draw_rect()
        if anchor is not None and old_anchor is not None and self._draw_rect is not None:
            x0, y0, sw, sh = self._draw_rect
            ix, iy = old_anchor
            self._pan[0] += anchor[0] - (x0 + ix / self._src_size[0] * sw)
            self._pan[1] += anchor[1] - (y0 + iy / self._src_size[1] * sh)
        if self._zoom <= 1.0:
            self._pan = [0.0, 0.0]
        self._clamp_pan()
        self._update_draw_rect()
        self.update()

    def _emit_click(self, x, y):
        if self._draw_rect is None or self._src_size is None:
            return
        img_xy = self._widget_to_image(x, y)
        if img_xy is None:
            return
        self.clicked.emit(*img_xy)

    def _widget_to_image(self, x, y):
        if self._draw_rect is None or self._src_size is None:
            return None
        x0, y0, sw, sh = self._draw_rect
        px = x - x0
        py = y - y0
        if not (0 <= px < sw and 0 <= py < sh):
            return None
        sx = int(px / sw * self._src_size[0])
        sy = int(py / sh * self._src_size[1])
        sx = max(0, min(self._src_size[0] - 1, sx))
        sy = max(0, min(self._src_size[1] - 1, sy))
        return sx, sy

    def _update_draw_rect(self):
        if self._pixmap_src is None or self._src_size is None:
            self._draw_rect = None
            return
        src_w, src_h = self._src_size
        if src_w <= 0 or src_h <= 0 or self.width() <= 0 or self.height() <= 0:
            self._draw_rect = None
            return
        scale = min(self.width() / src_w, self.height() / src_h) * self._zoom
        sw = max(1, int(round(src_w * scale)))
        sh = max(1, int(round(src_h * scale)))
        x0 = int(round((self.width() - sw) / 2 + self._pan[0]))
        y0 = int(round((self.height() - sh) / 2 + self._pan[1]))
        self._draw_rect = (x0, y0, sw, sh)

    def _clamp_pan(self):
        if self._pixmap_src is None or self._src_size is None:
            self._pan = [0.0, 0.0]
            return
        src_w, src_h = self._src_size
        scale = min(self.width() / src_w, self.height() / src_h) * self._zoom
        sw = src_w * scale
        sh = src_h * scale
        max_x = max(0.0, (sw - self.width()) / 2)
        max_y = max(0.0, (sh - self.height()) / 2)
        self._pan[0] = max(-max_x, min(max_x, self._pan[0]))
        self._pan[1] = max(-max_y, min(max_y, self._pan[1]))
