"""VPI-backed depth engine — GPU/OFA stereo on Jetson devices.

NVIDIA VPI provides hardware-accelerated stereo disparity. On Orin Nano the
OFA (Optical Flow Accelerator) backend uses dedicated stereo silicon, which
frees both CPU and GPU and avoids the GPU memory pressure that the CUDA
backend hits on the 4 GB Orin Nano at HD.

Output disparity is in Q10.5 fixed point (divide by 32 for px). The
subpixel uncertainty (``delta_d``) is set to 1/32 px to match.

Public API mirrors :class:`elp_stereo.depth.DepthEngine`, so :class:`DepthWorker`,
GUI panels and :class:`StereoPipeline` see no difference.

Buffer reuse is mandatory: ``vpi.asimage`` / ``Image.convert`` without an
``out=`` target leak into VPI's allocator pool and exhaust NvMap after a
handful of frames at HD. All buffers are pre-allocated in ``_build_matchers``
and reused via ``lock_cpu`` / ``out=`` each frame.
"""

import cv2
import inspect
import numpy as np

try:
    import vpi  # type: ignore
    VPI_AVAILABLE = True
except ImportError:
    vpi = None
    VPI_AVAILABLE = False

from .depth import DepthEngine

# VPI stereo output is Q10.5 unsigned fixed-point.
_VPI_DISP_SCALE = 1.0 / 32.0

_VPI_BACKENDS = {
    "OFA":  "OFA",
    "CUDA": "CUDA",
}


class VPIDepthEngine(DepthEngine):
    """Stereo via NVIDIA VPI. Drop-in for :class:`DepthEngine`."""

    def __init__(self, cfg, rectifier):
        if not VPI_AVAILABLE:
            raise RuntimeError(
                "VPI not available. `import vpi` failed. "
                "Use `engine: sgbm` in config or install VPI."
            )
        super().__init__(cfg, rectifier)
        # VPI disparity has 1/32 px subpixel resolution (Q10.5).
        self.delta_d = _VPI_DISP_SCALE

    def _build_matchers(self):
        s = self._cfg["sgbm"]  # reuse min_disp / num_disp / block_size fields
        vcfg = self._cfg.get("vpi", {}) or {}
        backend_name = vcfg.get("backend", "OFA").upper()
        if backend_name not in _VPI_BACKENDS:
            raise ValueError(
                f"Unknown vpi.backend '{backend_name}'. Valid: {sorted(_VPI_BACKENDS)}"
            )
        self._backend = getattr(vpi.Backend, backend_name)
        self.min_disparity = s["min_disparity"]
        self.num_disparities = int(vcfg.get("max_disparity", s["num_disparities"]))
        self._window = max(3, int(vcfg.get("window_size", s.get("block_size", 5))))
        self._quality = int(vcfg.get("quality", 6))
        self._confthreshold = int(vcfg.get("confthreshold", 32767))
        self._mindisp = int(vcfg.get("min_disparity", 0))
        self._p1 = int(vcfg.get("p1", 3))
        self._p2 = int(vcfg.get("p2", 48))
        self._p2alpha = int(vcfg.get("p2alpha", 0))
        self._uniqueness = float(vcfg.get("uniqueness", -1.0))
        self._include_diagonals = bool(vcfg.get("include_diagonals", True))
        self._num_passes = int(vcfg.get("num_passes", 3))
        self._stereodisp_params = self._supported_stereodisp_params()
        # SGBM-only knobs ignored on VPI.
        self.use_wls = False
        self._left_matcher = None
        self._right_matcher = None
        self._wls = None

        W, H = self.rectifier.image_size
        # Pre-allocate every VPI image used per frame. asimage / convert
        # without `out=` would otherwise allocate new buffers each call and
        # exhaust the NvMap pool.
        self._vl_u8 = vpi.Image(size=(W, H), format=vpi.Format.U8)
        self._vr_u8 = vpi.Image(size=(W, H), format=vpi.Format.U8)
        self._vl_y16 = vpi.Image(size=(W, H), format=vpi.Format.Y16_ER)
        self._vr_y16 = vpi.Image(size=(W, H), format=vpi.Format.Y16_ER)
        self._vl_bl = vpi.Image(size=(W, H), format=vpi.Format.Y16_ER_BL)
        self._vr_bl = vpi.Image(size=(W, H), format=vpi.Format.Y16_ER_BL)
        # OFA outputs block-linear; reconvert to pitch-linear S16 for numpy read.
        self._vd_bl = vpi.Image(size=(W, H), format=vpi.Format.S16_BL)
        self._vd = vpi.Image(size=(W, H), format=vpi.Format.S16)

    def _supported_stereodisp_params(self):
        """Return supported keyword params; older VPI versions expose fewer knobs."""
        try:
            return set(inspect.signature(vpi.stereodisp).parameters)
        except (TypeError, ValueError):
            return {
                "out", "backend", "window", "maxdisp",
                "confthreshold", "quality",
            }

    def _stereodisp_kwargs(self, out):
        kwargs = {
            "out": out,
            "backend": self._backend,
            "window": self._window,
            "maxdisp": self.num_disparities,
            "confthreshold": self._confthreshold,
            "quality": self._quality,
            "mindisp": self._mindisp,
            "p1": self._p1,
            "p2": self._p2,
            "p2alpha": self._p2alpha,
            "uniqueness": self._uniqueness,
            "includediagonals": self._include_diagonals,
            "numpasses": self._num_passes,
        }
        return {
            key: value for key, value in kwargs.items()
            if key in self._stereodisp_params
        }

    def compute(self, left_rect, right_rect):
        gl = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        # Copy current frames into pre-allocated U8 buffers.
        with self._vl_u8.lock_cpu() as arr:
            arr[:] = gl
        with self._vr_u8.lock_cpu() as arr:
            arr[:] = gr

        # U8 -> Y16_ER on CUDA (scale 256 = << 8 to fill 16-bit range).
        self._vl_u8.convert(out=self._vl_y16, backend=vpi.Backend.CUDA, scale=256)
        self._vr_u8.convert(out=self._vr_y16, backend=vpi.Backend.CUDA, scale=256)
        # Y16_ER -> Y16_ER_BL block-linear via VIC (required by OFA).
        self._vl_y16.convert(out=self._vl_bl, backend=vpi.Backend.VIC)
        self._vr_y16.convert(out=self._vr_bl, backend=vpi.Backend.VIC)

        # Stereo on the configured backend (OFA preferred on Orin). OFA emits
        # S16_BL; CUDA can emit S16 directly. Use block-linear out then VIC
        # convert for OFA, direct out for CUDA.
        if self._backend == vpi.Backend.OFA:
            vpi.stereodisp(
                self._vl_bl, self._vr_bl,
                **self._stereodisp_kwargs(self._vd_bl),
            )
            self._vd_bl.convert(out=self._vd, backend=vpi.Backend.VIC)
        else:
            vpi.stereodisp(
                self._vl_bl, self._vr_bl,
                **self._stereodisp_kwargs(self._vd),
            )

        with self._vd.rlock_cpu() as arr:
            raw = np.array(arr, copy=True)

        disp = raw.astype(np.float32) * _VPI_DISP_SCALE
        return self._finalize_depth(disp)

    def update_params(self, params):
        """Apply param changes. SGBM-specific keys are ignored (no matcher)."""
        # VPI has its own tunables; keep SGBM config independent in the UI/file.
        if "min_depth_mm" in params:
            self.min_depth_mm = float(params["min_depth_mm"])
        if "max_depth_mm" in params:
            self.max_depth_mm = float(params["max_depth_mm"])
        if "temporal_frames" in params:
            from collections import deque
            n = max(1, int(params["temporal_frames"]))
            self.temporal_frames = n
            self._depth_buffer = deque(maxlen=n)
        if "sample_radius_px" in params:
            self.sample_radius_px = max(0, int(params["sample_radius_px"]))
        if "sample_trim" in params:
            self.sample_trim = max(0.0, min(0.45, float(params["sample_trim"])))
        if "vpi" in params:
            self._cfg.setdefault("vpi", {}).update(params["vpi"])
        self._build_matchers()
