"""Centroid-EMA smoother for the top-1 detection across frames.

Worms wiggle locally so per-frame XYZ jitters even when the same target is
chosen. ``TopPickTracker`` smooths (u, v, X, Y, Z) with an EMA, but resets
if the new pick jumps further than ``reset_px_dist`` from the last smoothed
position (new target) or no detection is seen for ``max_miss_frames``.
"""

import math


class TopPickTracker:
    def __init__(self, alpha=0.4, reset_px_dist=30.0, max_miss_frames=10):
        self.alpha = float(alpha)
        self.reset_px_dist = float(reset_px_dist)
        self.max_miss_frames = int(max_miss_frames)
        self._state = None  # dict with u,v,X,Y,Z and original "det"
        self._miss = 0

    def reset(self):
        self._state = None
        self._miss = 0

    def update(self, item):
        """``item`` is a worker detection dict, or None for a miss frame.

        Returns the smoothed item (with updated ``uv`` and ``xyz_mm``) or
        None if no current track.
        """
        if item is None:
            self._miss += 1
            if self._miss > self.max_miss_frames:
                self.reset()
                return None
            return self._state

        self._miss = 0
        u, v = item["uv"]
        x, y, z = item["xyz_mm"]

        if self._state is None:
            self._state = {**item, "uv": (int(u), int(v)),
                           "xyz_mm": (float(x), float(y), float(z))}
            return self._state

        pu, pv = self._state["uv"]
        if math.hypot(u - pu, v - pv) > self.reset_px_dist:
            # Treat as a new target — reset smoother.
            self._state = {**item, "uv": (int(u), int(v)),
                           "xyz_mm": (float(x), float(y), float(z))}
            return self._state

        a = self.alpha
        nu = a * u + (1 - a) * pu
        nv = a * v + (1 - a) * pv
        px, py, pz = self._state["xyz_mm"]
        nx = a * x + (1 - a) * px
        ny = a * y + (1 - a) * py
        nz = a * z + (1 - a) * pz
        self._state = {
            **item,
            "uv": (int(round(nu)), int(round(nv))),
            "xyz_mm": (float(nx), float(ny), float(nz)),
        }
        return self._state
