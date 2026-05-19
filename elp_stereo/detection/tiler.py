"""Sliced inference wrapper for small-object detection.

Splits an image into an overlapping grid of tiles, runs a base detector on
each, remaps the per-tile bboxes back to full-image coordinates, and
performs a single global NMS so duplicate detections in overlap regions
collapse to one. Keeps tiny targets at their native pixel resolution
instead of letting them shrink under the detector's letterbox.
"""

import cv2
import numpy as np

from .detector import Detection, Detector


def make_tiles(W, H, grid=(2, 2), overlap=0.2):
    """Return list of ``(x0, y0, x1, y1)`` tiles covering a ``W x H`` image.

    ``grid`` = (cols, rows). ``overlap`` is the fractional overlap between
    adjacent tiles relative to the base (non-overlapping) tile width / height.
    The outermost tile edges are clamped to the image bounds so the union of
    tiles exactly covers the image.
    """
    cols, rows = int(grid[0]), int(grid[1])
    if cols < 1 or rows < 1:
        raise ValueError("grid must have positive cols and rows")
    if cols == 1 and rows == 1:
        return [(0, 0, W, H)]
    o = float(overlap)

    base_w = W / cols
    base_h = H / rows
    pad_w = int(round(base_w * o / 2.0))
    pad_h = int(round(base_h * o / 2.0))

    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * base_w)) - pad_w
            y0 = int(round(r * base_h)) - pad_h
            x1 = int(round((c + 1) * base_w)) + pad_w
            y1 = int(round((r + 1) * base_h)) + pad_h
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(W, x1); y1 = min(H, y1)
            if x1 - x0 > 1 and y1 - y0 > 1:
                tiles.append((x0, y0, x1, y1))
    return tiles


class TiledDetector(Detector):
    """Wraps a base :class:`Detector`, runs it on grid tiles, NMS-merges."""

    def __init__(self, base, grid=(2, 2), overlap=0.2, iou=0.45):
        self._base = base
        self._grid = (int(grid[0]), int(grid[1]))
        self._overlap = float(overlap)
        self._iou = float(iou)

    @property
    def class_names(self):
        return self._base.class_names

    @property
    def base(self):
        return self._base

    def set_base(self, base):
        self._base = base

    def infer(self, bgr):
        H, W = bgr.shape[:2]
        tiles = make_tiles(W, H, self._grid, self._overlap)
        if len(tiles) == 1:
            return self._base.infer(bgr)

        merged = []
        for (x0, y0, x1, y1) in tiles:
            sub = bgr[y0:y1, x0:x1]
            for det in self._base.infer(sub):
                bx1, by1, bx2, by2 = det.bbox
                merged.append(Detection(
                    det.cls_id, det.cls_name, det.score,
                    (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0),
                ))
        if not merged:
            return []

        # Global NMS across all tile-merged detections (xywh format).
        boxes_xywh = []
        scores = []
        for d in merged:
            x1, y1, x2, y2 = d.bbox
            boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            scores.append(float(d.score))
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.0, self._iou)
        if idxs is None or len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()
        return [merged[i] for i in idxs]
