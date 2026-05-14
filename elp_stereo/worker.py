"""Background depth computation thread.

Receives the latest stereo pair via ``submit()``, rectifies, computes disparity
and depth, then emits ``result_ready`` with the processed images. Frames that
arrive while a compute is in progress are dropped (latest-only policy) so the
GUI is never blocked waiting for a stale result.
"""

import threading

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class DepthWorker(QThread):
    """Runs rectify + StereoSGBM in a background thread.

    Emits ``result_ready`` with a dict:
        ``left``      – rectified left  BGR ndarray
        ``right``     – rectified right BGR ndarray
        ``color``     – colorized depth BGR ndarray
        ``depth_map`` – float32 depth in mm, NaN = invalid (snapshot copy)
    """

    result_ready = pyqtSignal(object)

    def __init__(self, rectifier, depth_engine, parent=None):
        super().__init__(parent)
        self._rectifier = rectifier
        self._engine = depth_engine
        self._lock = threading.Lock()
        self._pending = None      # latest (left, right) waiting to be processed
        self._stop = False

    def submit(self, left, right):
        """Offer a new frame pair. Overwrites any unprocessed pending pair."""
        with self._lock:
            self._pending = (left, right)

    def run(self):
        self._stop = False
        while not self._stop:
            with self._lock:
                frame = self._pending
                self._pending = None

            if frame is None:
                self.msleep(5)
                continue

            left, right = frame
            lr, rr = self._rectifier.rectify(left, right)
            self._engine.compute(lr, rr)
            color = self._engine.colorized()

            # Copy depth_map so the GUI holds a stable snapshot even if the
            # next compute starts immediately after this signal is queued.
            dm = self._engine.depth_map
            dm_copy = dm.copy() if dm is not None else None

            self.result_ready.emit({
                "left": lr,
                "right": rr,
                "color": color,
                "depth_map": dm_copy,
            })

    def stop(self):
        self._stop = True
        with self._lock:
            self._pending = None
        self.wait(2000)
