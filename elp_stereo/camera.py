"""Stereo capture for the ELP 3D USB camera.

The ELP 3D enumerates as a single V4L2 device delivering a wide side-by-side
frame (``left | right``) that is hardware-synchronized. This is the only mode
that gives synced pairs. If only separate devices are available the capture
falls back to opening two of them independently and flags the stream as
unsynchronized (acceptable for monitoring, not for calibration/depth).
"""

import glob
import os

import cv2
from PyQt5.QtCore import QThread, pyqtSignal


def find_elp_device(device_match, device_override=None):
    """Resolve the ELP capture device.

    Returns a tuple ``(spec, mode)`` where ``mode`` is ``"sbs"`` for a single
    side-by-side device or ``"dual"`` for a two-device fallback. ``spec`` is a
    device path string (sbs) or a ``(left, right)`` path pair (dual).
    """
    if device_override:
        return device_override, "sbs"

    by_id = "/dev/v4l/by-id"
    matches = []
    if os.path.isdir(by_id):
        for link in sorted(glob.glob(os.path.join(by_id, "*"))):
            if device_match in os.path.basename(link):
                matches.append(os.path.realpath(link))

    # De-duplicate while preserving order (index0/index1 may point to one cam).
    seen = []
    for m in matches:
        if m not in seen:
            seen.append(m)

    if len(seen) >= 1:
        # ELP 3D: first node is the capture node, side-by-side.
        return seen[0], "sbs"

    raise RuntimeError(
        f"No ELP device found in {by_id} matching '{device_match}'. "
        f"Set camera.device_override in the config."
    )


class StereoCamera:
    """Opens the ELP camera and yields rectified-ready ``(left, right)`` pairs."""

    def __init__(self, cfg):
        cam = cfg["camera"]
        self.spec, self.mode = find_elp_device(
            cam["device_match"], cam.get("device_override")
        )
        self.frame_width = cam["frame_width"]
        self.frame_height = cam["frame_height"]
        self.fourcc = cam.get("fourcc", "MJPG")
        self.fps = cam.get("fps", 30)
        self.unsynced = self.mode == "dual"
        self._caps = []

    def open(self):
        if self.mode == "sbs":
            cap = cv2.VideoCapture(self.spec, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {self.spec}")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._caps = [cap]
        else:
            left, right = self.spec
            for dev in (left, right):
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open {dev}")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width // 2)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                self._caps.append(cap)
        return self

    def grab(self):
        """Return ``(left, right)`` BGR frames, or ``None`` on read failure."""
        if self.mode == "sbs":
            ok, frame = self._caps[0].read()
            if not ok or frame is None:
                return None
            mid = frame.shape[1] // 2
            return frame[:, :mid].copy(), frame[:, mid:].copy()
        else:
            # Grab both first to minimize the inter-device time skew.
            self._caps[0].grab()
            self._caps[1].grab()
            ok_l, left = self._caps[0].retrieve()
            ok_r, right = self._caps[1].retrieve()
            if not (ok_l and ok_r) or left is None or right is None:
                return None
            return left, right

    def release(self):
        for cap in self._caps:
            cap.release()
        self._caps = []


class CaptureThread(QThread):
    """Background thread that continuously grabs stereo frames."""

    frames_ready = pyqtSignal(object, object)  # (left, right) BGR ndarrays
    error = pyqtSignal(str)

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self._camera = camera
        self._running = False

    def run(self):
        self._running = True
        try:
            self._camera.open()
        except Exception as exc:  # noqa: BLE001 - surface to GUI
            self.error.emit(str(exc))
            return
        while self._running:
            pair = self._camera.grab()
            if pair is None:
                self.error.emit("Frame grab failed")
                self.msleep(50)
                continue
            self.frames_ready.emit(pair[0], pair[1])
        self._camera.release()

    def stop(self):
        self._running = False
        self.wait(2000)
