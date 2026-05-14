# CLAUDE.md — ELP 3D Stereo Camera Tool

Context file for AI sessions integrating with or extending this codebase.

## What this is

Desktop app (PyQt5) for the **ELP 3D USB stereo camera** — a single-USB device
that streams a hardware-synchronized side-by-side (SBS) wide frame
(e.g. 2560×720 = left 1280×720 | right 1280×720). The app provides:

1. **Live visualization** — synchronized left / right / depth panels.
2. **Stereo calibration** — chessboard or ChArUco board, live capture or folder load.
3. **Depth measurement** — per-pixel depth (mm) with uncertainty band and detectable range.

## Hardware

- **Camera device:** auto-detected via `/dev/v4l/by-id/` symlink containing
  `3D_USB_Camera` → resolves to `/dev/video2` on this machine.
- **Frame format:** single capture, split at `width // 2`. Left = `frame[:, :w//2]`.
- **Override:** `config/default.yaml` → `camera.device_override: /dev/videoN`.

## Environment

- Python 3.12 (Ubuntu 24.04), also tested 3.10 (Ubuntu 22.04).
- OpenCV 4.6.0 system build (`python3-opencv`) — includes `aruco` + `ximgproc`.
- PyQt5 system package.
- No virtualenv needed on this machine; `pip install -r requirements.txt` for
  other machines.

## File map

```
app.py                          Entry point. Loads config, launches MainWindow.
config/default.yaml             All runtime config (device, board, SGBM params).
config/stereo_calib.yaml        Calibration output. Auto-loaded on startup. In .gitignore.

elp_stereo/
  config.py                     load_config(path=None) → dict. project_root() → Path.
  camera.py                     StereoCamera, CaptureThread (QThread).
  targets.py                    ChessboardTarget, CharucoTarget, build_target(kind, cfg).
  calibration.py                StereoCalibrator, CalibrationResult, save_yaml, load_yaml.
  depth.py                      Rectifier, DepthEngine, PixelInfo.
  worker.py                     DepthWorker (QThread) — background SGBM compute.
  gui/
    widgets.py                  ImagePanel (QLabel + click→image coords signal).
    main_window.py              MainWindow — top-level, wires all components.
    calib_widget.py             Calibration tab UI.
    depth_widget.py             Depth tab UI (SGBM sliders + pixel readout).
```

## Data / signal flow

```
CaptureThread  ──frames_ready(left, right)──►  MainWindow.on_frames_ready
                                                    │
                                                    ▼
                                               DepthWorker.submit(left, right)
                                                    │  (drops frame if busy)
                                                    ▼
                                               rectify → SGBM → reprojectImageTo3D
                                                    │
                                         result_ready({left, right, color, depth_map})
                                                    │
                                                    ▼
                                               MainWindow.on_depth_result
                                               → update 3 ImagePanels (GUI thread)
                                               → store _latest_depth_map (snapshot copy)

ImagePanel.clicked(x,y)  ──►  MainWindow.on_depth_click
                               → DepthEngine.pixel_info_from_map(_latest_depth_map, x, y)
                               → DepthWidget.show_pixel_info(info)
```

## Key classes and APIs

### `StereoCamera` (`camera.py`)
```python
cam = StereoCamera(cfg)           # cfg = load_config()
cam.open()                        # opens /dev/videoN
left, right = cam.grab()          # returns (H,W,3) BGR pair or None
cam.release()
cam.mode        # "sbs" | "dual"
cam.unsynced    # True if dual-device fallback (warn user)
```

### `CalibrationTarget` (`targets.py`)
```python
target = build_target("chessboard", cfg["calibration"])
target = build_target("charuco",    cfg["calibration"])
det = target.detect(gray_image)   # Detection | None
det.object_points   # (N,3) float32 mm
det.image_points    # (N,2) float32 px
det.ids             # (N,1) int32 charuco ids, or None (chessboard)
target.draw(bgr, det)
```

### `StereoCalibrator` (`calibration.py`)
```python
cal = StereoCalibrator((width, height))
cal.add_pair(det_left, det_right)   # → matched corner count
cal.pair_count
result = cal.calibrate()            # CalibrationResult (raises if < 5 pairs)
save_yaml("config/stereo_calib.yaml", result)
result = load_yaml("config/stereo_calib.yaml")
```

### `CalibrationResult` fields
`image_size, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q, roi1, roi2, rms`

### `Rectifier` (`depth.py`)
```python
rect = Rectifier(calib_result)
left_rect, right_rect = rect.rectify(left_bgr, right_bgr)
rect.fx         # focal length px (from P1)
rect.baseline   # baseline mm (from P2)
rect.Q          # 4×4 reprojection matrix
```

### `DepthEngine` (`depth.py`)
```python
engine = DepthEngine(cfg, rectifier)
depth_map = engine.compute(left_rect, right_rect)   # float32 (H,W) mm, NaN=invalid
color     = engine.colorized()                       # BGR COLORMAP_JET, black=invalid
rmin, rmax = engine.detection_range()               # (mm, mm)
info = engine.pixel_info_from_map(depth_map, x, y) # PixelInfo (use snapshot copy)
engine.update_params({"sgbm": {...}, "use_wls_filter": bool})
```

### `PixelInfo` fields
`valid: bool, depth_mm, error_mm, range_min_mm, range_max_mm`

Depth uncertainty: `error_mm = z² / (fx * baseline) * delta_d`
where `delta_d` = 1/16 px (SGBM subpixel resolution, configurable).

### `DepthWorker` (`worker.py`)
```python
worker = DepthWorker(rectifier, depth_engine)
worker.result_ready.connect(slot)   # slot receives dict
worker.start()
worker.submit(left, right)          # call from any thread; drops stale frames
worker.stop()
```

Result dict: `{"left": ndarray, "right": ndarray, "color": ndarray, "depth_map": ndarray}`

## Threading rules

- `CaptureThread` and `DepthWorker` are `QThread`s. **Never call Qt widget methods
  from them.** Use signals only.
- `depth_map` passed in `result_ready` is a `.copy()` snapshot — safe to hold in
  GUI thread while worker starts next compute.
- `DepthWorker` uses a single pending-frame slot (latest-only): slow compute never
  queues up stale frames.

## Config structure (`config/default.yaml`)

```yaml
camera:
  device_match: "3D_USB_Camera"   # matched against /dev/v4l/by-id/ symlink names
  device_override: null            # override with /dev/videoN
  frame_width: 2560                # full SBS frame width (2 × per-eye width)
  frame_height: 720
  fourcc: "MJPG"
  fps: 30

calibration:
  output_path: "config/stereo_calib.yaml"
  chessboard: {cols: 9, rows: 6, square_size_mm: 25.0}
  charuco: {squares_x: 5, squares_y: 7, square_len_mm: 30.0, marker_len_mm: 22.0, dictionary: DICT_4X4_50}

depth:
  sgbm: {min_disparity: 0, num_disparities: 128, block_size: 5, ...}
  use_wls_filter: false
  subpixel_delta_disparity: 0.0625   # 1/16 px
```

## OpenCV version compatibility

| OpenCV | aruco API path | flag |
|--------|---------------|------|
| ≤ 4.6  | `Dictionary_get` / `CharucoBoard_create` / `interpolateCornersCharuco` | `_ARUCO_MODERN = False` |
| ≥ 4.7  | `getPredefinedDictionary` / `CharucoBoard` / `CharucoDetector` | `_ARUCO_MODERN = True` |

`WLS_AVAILABLE = hasattr(cv2, "ximgproc")` — WLS filter disabled gracefully if absent.

## Integration notes for future sessions

- **ROS2 output:** `DepthWorker.result_ready` dict provides rectified BGR pairs and
  float32 depth map ready to wrap in `sensor_msgs/Image` and `sensor_msgs/CameraInfo`.
  Calibration `K1,D1,P1,R1` → left `CameraInfo`. `Q` matrix encodes stereo geometry.
- **Point cloud:** `depth_map` + `Q` → `cv2.reprojectImageTo3D` already done inside
  `DepthEngine.compute`. The 3D points are in the rectified left camera frame, Z forward, units mm.
- **Adding a new panel/view:** subclass or reuse `ImagePanel` from `gui/widgets.py`.
  Connect to `MainWindow.depth_worker.result_ready` or add a new signal to `DepthWorker`.
- **Custom depth algorithm:** replace `DepthEngine.compute` or subclass `DepthEngine`.
  The rest of the pipeline (Rectifier, DepthWorker, GUI) is algorithm-agnostic.
- **Saving captures:** hook into `DepthWorker.result_ready` or `CaptureThread.frames_ready`.
  Both fire in background threads — copy arrays before storing.

## Run

```bash
python app.py
```

## Smoke test (no display needed)

```bash
python3 -c "
import os; os.environ['QT_QPA_PLATFORM']='offscreen'
from elp_stereo.config import load_config
from PyQt5.QtWidgets import QApplication
from elp_stereo.gui.main_window import MainWindow
from elp_stereo.calibration import load_yaml
app = QApplication([])
w = MainWindow(load_config())
w.apply_calibration(load_yaml('config/stereo_calib.yaml'))
print('OK', w.depth_engine is not None)
"
```
