"""Object detection backends.

Auto-selects backend by model extension:
    .pt   -> Ultralytics YOLO
    .onnx -> onnxruntime + YOLOv8 head decode

Both backends import lazily so the unused one need not be installed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    score: float
    bbox: tuple  # (x1, y1, x2, y2) ints in source-image coordinates


class Detector(ABC):
    @abstractmethod
    def infer(self, bgr):
        """Run detection on a BGR ndarray. Returns list[Detection]."""

    @property
    @abstractmethod
    def class_names(self):
        """Return list[str] of class names indexed by cls_id."""


class PtDetector(Detector):
    """Ultralytics YOLO (.pt) backend."""

    def __init__(self, model_path, conf=0.35, iou=0.45, classes=None):
        from ultralytics import YOLO  # lazy
        self._model = YOLO(str(model_path))
        self._conf = float(conf)
        self._iou = float(iou)
        self._classes = classes  # list[int] or None
        names = self._model.names
        if isinstance(names, dict):
            self._names = [names[i] for i in sorted(names.keys())]
        else:
            self._names = list(names)

    @property
    def class_names(self):
        return self._names

    def infer(self, bgr):
        results = self._model.predict(
            bgr, conf=self._conf, iou=self._iou,
            classes=self._classes, verbose=False,
        )
        out = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        h, w = bgr.shape[:2]
        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
            x1i = max(0, int(round(x1))); y1i = max(0, int(round(y1)))
            x2i = min(w - 1, int(round(x2))); y2i = min(h - 1, int(round(y2)))
            if x2i <= x1i or y2i <= y1i:
                continue
            name = self._names[c] if 0 <= c < len(self._names) else str(c)
            out.append(Detection(int(c), name, float(s), (x1i, y1i, x2i, y2i)))
        return out


class OnnxDetector(Detector):
    """YOLOv8 ONNX backend (640x640 letterboxed input, anchor-free head)."""

    def __init__(self, model_path, conf=0.35, iou=0.45, classes=None,
                 class_names=None, input_size=640):
        import onnxruntime as ort  # lazy
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # Filter providers list to those actually available; ORT errors otherwise.
        avail = set(ort.get_available_providers())
        providers = [p for p in providers if p in avail] or ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        self._conf = float(conf)
        self._iou = float(iou)
        self._classes = set(classes) if classes else None
        self._names = list(class_names) if class_names else []
        self._input_size = int(input_size)

    @property
    def class_names(self):
        return self._names

    def _letterbox(self, bgr):
        H, W = bgr.shape[:2]
        s = self._input_size
        scale = min(s / W, s / H)
        nw, nh = int(round(W * scale)), int(round(H * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        dx = (s - nw) // 2
        dy = (s - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, scale, dx, dy

    def infer(self, bgr):
        canvas, scale, dx, dy = self._letterbox(bgr)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None]  # 1x3xHxW
        out = self._sess.run(None, {self._input_name: blob})[0]
        # YOLOv8 export: shape (1, 4+nc, N) or (1, N, 4+nc). Normalize layout.
        if out.ndim == 3:
            if out.shape[1] < out.shape[2]:
                pred = out[0].T  # -> (N, 4+nc)
            else:
                pred = out[0]
        else:
            return []

        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]
        if scores_all.shape[1] == 0:
            return []
        cls_ids = scores_all.argmax(axis=1)
        scores = scores_all.max(axis=1)
        keep = scores >= self._conf
        if not keep.any():
            return []
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        # xywh (center) -> xyxy in letterbox space, then undo letterbox to source.
        xy = boxes_xywh[:, :2]
        wh = boxes_xywh[:, 2:]
        x1y1 = xy - wh / 2.0
        x2y2 = xy + wh / 2.0
        x1 = (x1y1[:, 0] - dx) / scale
        y1 = (x1y1[:, 1] - dy) / scale
        x2 = (x2y2[:, 0] - dx) / scale
        y2 = (x2y2[:, 1] - dy) / scale

        H, W = bgr.shape[:2]
        # NMS expects xywh in pixels.
        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), self._conf, self._iou)
        if idxs is None or len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()

        out_dets = []
        for i in idxs:
            c = int(cls_ids[i])
            if self._classes is not None and c not in self._classes:
                continue
            x1i = max(0, int(round(float(x1[i]))))
            y1i = max(0, int(round(float(y1[i]))))
            x2i = min(W - 1, int(round(float(x2[i]))))
            y2i = min(H - 1, int(round(float(y2[i]))))
            if x2i <= x1i or y2i <= y1i:
                continue
            name = self._names[c] if 0 <= c < len(self._names) else str(c)
            out_dets.append(Detection(c, name, float(scores[i]), (x1i, y1i, x2i, y2i)))
        return out_dets


def build_detector(model_path, conf=0.35, iou=0.45, classes=None,
                   class_names=None, input_size=640):
    """Construct a detector from a model file path. Backend chosen by extension."""
    ext = Path(model_path).suffix.lower()
    if ext == ".pt":
        return PtDetector(model_path, conf=conf, iou=iou, classes=classes)
    if ext == ".onnx":
        return OnnxDetector(
            model_path, conf=conf, iou=iou, classes=classes,
            class_names=class_names, input_size=input_size,
        )
    raise ValueError(
        f"Unsupported model extension '{ext}'. Use .pt (Ultralytics) or .onnx."
    )
