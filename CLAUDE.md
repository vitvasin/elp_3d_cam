# CLAUDE.md — ELP 3D Stereo Camera Tool

Context file for AI sessions integrating with or extending this codebase.

## What this is

Two PyQt5 desktop apps for the **ELP 3D USB stereo camera** — a single-USB
device that streams a hardware-synchronized side-by-side (SBS) wide frame
(e.g. 2560×720 = left 1280×720 | right 1280×720).

**App 1 — `app.py` (calibration + live depth viewer)**
1. Synchronized left / right / depth panels.
2. Stereo calibration — chessboard, ChArUco, or circle-grid (symmetric / asymmetric).
3. Per-pixel depth (mm) with uncertainty band. CPU SGBM or NVIDIA VPI (Jetson).

**App 2 — `app_detect.py` (object detection + ROS2 publisher)**
1. Loads calibration from App 1, runs depth pipeline.
2. Object detection on rectified-left frame — `.pt` (Ultralytics) or `.onnx`
   (onnxruntime, YOLOv8 head), auto-picked by extension.
3. Tiled inference (2×2 + 20% overlap), CLAHE pre-proc, tray-plane Z fallback,
   shrink-bbox depth pick, centroid-EMA smoothing of top-1.
4. Publishes top-1 (or all) detections to `vision_msgs/Detection3DArray` on
   `/elp/detections` + static TF `robot_base → camera_optical_frame`.

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
app.py                          App 1 entry. Calibration + depth viewer.
app_detect.py                   App 2 entry. Detection + ROS2 publisher. Deep-merges detect.yaml.
config/default.yaml             Camera + depth + calibration board params.
config/detect.yaml              Detection + ROS2 + optimization params (App 2 overlay).
config/stereo_calib.yaml        Calibration output. Auto-loaded by both apps. In .gitignore.

elp_stereo/
  config.py                     load_config(path=None) → dict. project_root() → Path.
  camera.py                     StereoCamera, CaptureThread (QThread).
  targets.py                    ChessboardTarget, CharucoTarget, CircleGridTarget, build_target(kind, cfg).
  calibration.py                StereoCalibrator, CalibrationResult, save_yaml, load_yaml.
  depth.py                      Rectifier, DepthEngine (SGBM), build_depth_engine, PixelInfo.
  depth_vpi.py                  VPIDepthEngine (NVIDIA VPI on Jetson, OFA/CUDA backends).
  pipeline.py                   StereoPipeline (Headless API).
  worker.py                     DepthWorker (QThread) — background depth compute.

  detection/                    App 2 detection package.
    detector.py                 Detector ABC, PtDetector, OnnxDetector, build_detector(path,...).
    tiler.py                    make_tiles(W,H,grid,overlap), TiledDetector (wraps any Detector).
    preproc.py                  clahe_bgr(bgr) — LAB-L CLAHE.
    plane.py                    Plane dataclass, fit_tray_plane (RANSAC+SVD), plane_z_at.
    depth_pick.py               bbox_median_depth, bbox_robust_depth (shrink), pixel_to_camera_xyz.
    tracker.py                  TopPickTracker (centroid-EMA, reset on jump, miss counter).
    worker.py                   DetectionWorker (QThread) — full pipeline.

  ros2/                         App 2 ROS2 publisher (guarded; degrades if rclpy/vision_msgs missing).
    __init__.py                 ROS2_AVAILABLE probe.
    publisher.py                DetectionPublisher (Node) + RosSpinThread.
    tf_utils.py                 static_transform_from_config, lookup_camera_to_robot, transform_point_mm.

  gui/
    widgets.py                  ImagePanel (QLabel + click→image coords signal).
    main_window.py              App 1 MainWindow.
    calib_widget.py             Calibration tab UI.
    depth_widget.py             Depth tab UI (SGBM sliders + pixel readout).
    detect_main_window.py       App 2 DetectMainWindow.
    detect_widget.py            App 2 DetectionWidget (model, opt toggles, plane, ROS status).

examples/
  headless_depth.py             Stand-alone integration example using StereoPipeline.
```

## Data / signal flow

```
CaptureThread  ──frames_ready(left, right)──►  MainWindow.on_frames_ready
                                                    │
                                                    ▼
                                               DepthWorker.submit(left, right)
                                                    │  (drops frame if busy)
                                                    ▼
                                               rectify → engine.compute (SGBM or VPI) → reprojectImageTo3D
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

### `StereoPipeline` (`pipeline.py`) — Headless API
```python
from elp_stereo import StereoPipeline
pipe = StereoPipeline(config_path=None)
pipe.load_calibration("config/stereo_calib.yaml")
pipe.start()
left, right, depth = pipe.grab_depth()  # rectified BGRs + float32 depth map
color = pipe.get_colorized_depth()       # BGR color visualization
pipe.stop()
```

### `CalibrationTarget` (`targets.py`)
```python
target = build_target("chessboard",  cfg["calibration"])
target = build_target("charuco",     cfg["calibration"])
target = build_target("circle_grid", cfg["calibration"])  # sym + asym
det = target.detect(gray_image)   # Detection | None
det.object_points   # (N,3) float32 mm
det.image_points    # (N,2) float32 px
det.ids             # (N,1) int32 charuco ids, or None (chessboard / circle_grid)
target.draw(bgr, det)
```

`CircleGridTarget` uses `cv2.findCirclesGrid`. For asymmetric: `cols` = circles per row, `rows` = total rows; `spacing_mm` = `s` where same-row horizontal pitch = `2*s` and vertical row pitch = `s` (OpenCV convention). Standard board: 4×11 asymmetric.

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

### `DepthEngine` (`depth.py`) — CPU SGBM
```python
engine = DepthEngine(cfg, rectifier)
depth_map = engine.compute(left_rect, right_rect)   # float32 (H,W) mm, NaN=invalid
color     = engine.colorized()                       # BGR colormap, black=invalid
rmin, rmax = engine.detection_range()               # (mm, mm)
info = engine.pixel_info_from_map(depth_map, x, y) # PixelInfo (use snapshot copy)
engine.update_params({"sgbm": {...}, "min_depth_mm": 50, "temporal_frames": 5})
```

- **SGBM Mode:** Selectable via `sgbm.mode` config: `HH` (8-dir, slowest, best), `SGBM` (5-dir, default OpenCV), `SGBM_3WAY` (3-dir, ~3× faster — current default), `HH4` (4-dir, OpenCV ≥ 4.5).
- **Range Capping:** `min_depth_mm` and `max_depth_mm` act as a depth mask.
- **Temporal Averaging:** `temporal_frames > 1` enables `1/sqrt(N)` noise reduction.
- **Speckle Removal:** 3×3 median blur applied to the depth map.

The compute path is split into `compute()` (matcher-specific disparity) and `_finalize_depth(disp_px)` (shared Q-reproject + capping + median + temporal averaging). `VPIDepthEngine` reuses `_finalize_depth`.

### `VPIDepthEngine` (`depth_vpi.py`) — NVIDIA VPI (Jetson)
```python
from elp_stereo.depth import build_depth_engine
engine = build_depth_engine(cfg, rectifier)   # picks engine from cfg["depth"]["engine"]
# same API as DepthEngine: compute / colorized / detection_range / pixel_info_from_map
```

- **Backend:** `vpi.backend` config = `"OFA"` (dedicated stereo HW on Orin, frees CPU + GPU; default) or `"CUDA"`. On 4 GB Orin Nano avoid CUDA at ≥ 720p — NvMap OOM.
- **Subpixel:** VPI disparity is Q10.5 (1/32 px) — `delta_d = 1/32` (overrides config).
- **Buffer reuse:** All VPI images are pre-allocated in `_build_matchers`; `compute()` copies new frames in via `lock_cpu` and runs conversions with `out=` targets. Without this, the NvMap allocator pool exhausts after ~10 frames at HD.
- **Pipeline:** U8 → Y16_ER (CUDA scale ×256) → Y16_ER_BL (VIC) → `stereodisp` (OFA, S16_BL out) → S16 (VIC) → numpy → `_finalize_depth`.
- **Factory:** `build_depth_engine(cfg, rectifier)` dispatches on `cfg["depth"]["engine"]` (`"sgbm"` | `"vpi"`). `MainWindow.apply_calibration` falls back to SGBM if VPI import fails.

### `PixelInfo` fields
`valid: bool, depth_mm, error_mm, range_min_mm, range_max_mm`

Depth uncertainty: `error_mm = z² / (fx * baseline) * delta_d`
where `delta_d` = 1/16 px for SGBM (Q4.4) and 1/32 px for VPI (Q10.5).

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
  chessboard:  {cols: 9, rows: 6, square_size_mm: 25.0}
  charuco:     {squares_x: 5, squares_y: 7, square_len_mm: 30.0, marker_len_mm: 22.0, dictionary: DICT_4X4_50}
  circle_grid: {cols: 4, rows: 11, spacing_mm: 20.0, asymmetric: true}

depth:
  engine: "vpi"                     # "sgbm" (CPU) | "vpi" (Jetson, OFA/CUDA)
  vpi:    {backend: "OFA", quality: 6}
  sgbm:   {min_disparity: 0, num_disparities: 128, block_size: 5, mode: "SGBM_3WAY", ...}
  use_wls_filter: false
  subpixel_delta_disparity: 0.0625  # 1/16 px (SGBM only; VPI uses 1/32 internally)
```

## OpenCV version compatibility

| OpenCV | aruco API path | flag |
|--------|---------------|------|
| ≤ 4.6  | `Dictionary_get` / `CharucoBoard_create` / `interpolateCornersCharuco` | `_ARUCO_MODERN = False` |
| ≥ 4.7  | `getPredefinedDictionary` / `CharucoBoard` / `CharucoDetector` | `_ARUCO_MODERN = True` |

`WLS_AVAILABLE = hasattr(cv2, "ximgproc")` — WLS filter disabled gracefully if absent.
`VPI_AVAILABLE` — `import vpi` guarded in `depth_vpi.py`; on non-Jetson installs the factory transparently falls back to SGBM.

## Integration notes for future sessions

- **Headless Usage:** See `examples/headless_depth.py` for how to use `StereoPipeline`
  without the GUI. Ideal for automation or background processing.
- **ROS2 output:** `DepthWorker.result_ready` (GUI) or `StereoPipeline.grab_depth`
  (Headless) provide rectified BGR pairs and float32 depth maps ready to wrap in
  `sensor_msgs/Image`. Calibration `K1,D1,P1,R1` → left `CameraInfo`.
- **Point cloud:** `depth_map` + `Q` → `cv2.reprojectImageTo3D` already done inside
  `DepthEngine.compute`. The 3D points are in the rectified left camera frame, Z forward, units mm.
- **Adding a new panel/view:** subclass or reuse `ImagePanel` from `gui/widgets.py`.
  Connect to `MainWindow.depth_worker.result_ready` or add a new signal to `DepthWorker`.
- **Custom depth algorithm:** subclass `DepthEngine`, override `_build_matchers` and `compute`, reuse `_finalize_depth` for Q reproject + capping + median + temporal averaging. Add a kind to `build_depth_engine` to wire it. `VPIDepthEngine` is the reference example.
- **Saving captures:** hook into `DepthWorker.result_ready` or `CaptureThread.frames_ready`.
  Both fire in background threads — copy arrays before storing.

## App 2 — Detection + ROS2

### Detector (`elp_stereo.detection.detector`)
```python
det = build_detector("models/worm.pt", conf=0.25, iou=0.45,
                     classes=None, class_names=None, input_size=960)
# Detection(cls_id, cls_name, score, bbox=(x1,y1,x2,y2))
results = det.infer(left_rect_bgr)
```
Backend auto-picked: `.pt` → `PtDetector` (ultralytics), `.onnx` → `OnnxDetector`
(onnxruntime, YOLOv8 head decode + cv2.dnn.NMSBoxes, 640/960 letterbox).

### Tiled inference (`tiler.py`)
```python
td = TiledDetector(det, grid=(2,2), overlap=0.2, iou=0.45)
# splits bgr -> per-tile infer -> remap bbox -> global NMS
```
Keeps tiny targets at native px resolution; mandatory for sub-10-px objects.

### Tray-plane fallback (`plane.py`)
```python
plane = fit_tray_plane(depth_map, P1, iters=200, thresh_mm=5.0)
# Plane(n, d, rms, mean_z_mm, tilt_deg)
z = plane_z_at(plane, u, v, P1)   # mm
```
Used when bbox stereo depth has too few valid pixels (worms texture-poor).

### Robust depth pick (`depth_pick.py`)
```python
z, n = bbox_robust_depth(depth_map, (x1,y1,x2,y2), shrink=0.6)
```
Shrinks bbox to inner 60% → median → avoids background bleed at bbox edges.

### Top-1 smoothing (`tracker.py`)
```python
trk = TopPickTracker(alpha=0.4, reset_px_dist=30.0, max_miss_frames=10)
smoothed = trk.update(item)   # or None
```
Centroid-EMA in pixel + 3D space; resets on jumps > `reset_px_dist`.

### Worker (`detection/worker.py`) pipeline
```
left,right  → rectify → depth (raw input!) → CLAHE? → detector (tiled?) →
top-1 by score → bbox_robust_depth → fallback to plane Z if sparse →
pixel_to_camera_xyz → tracker → result_ready(dict)
```
Setters: `set_clahe`, `set_tile_params(enabled, grid, overlap)`,
`set_plane(plane)`, `set_tracker(t)`, `set_fallback_to_plane`,
`set_bbox_shrink`, `set_projection(P1)`.

### ROS2 publisher (`elp_stereo.ros2`)
```python
node = DetectionPublisher(cfg["ros2"])
n = node.publish_detections([item, ...])   # vision_msgs/Detection3DArray
```
- Topic default: `/elp/detections` (Reliable, depth 10).
- Static TF: `cfg["ros2"]["static_tf"]` → `StaticTransformBroadcaster` at init.
- `use_tf_lookup: true` → pre-transforms XYZ into `robot_frame` via tf2 lookup;
  message `header.frame_id` flips from `camera_frame` to `robot_frame`.
- All ROS imports guarded; `ROS2_AVAILABLE` flag exposed at package root.

### App 2 config (`config/detect.yaml`) — defaults tuned for 4–7 px worms

```yaml
detection:   {confidence: 0.25, iou: 0.45, min_valid_pixels: 5,
              bbox_shrink: 0.6, depth_fallback_to_plane: true,
              onnx_input_size: 960}
preproc:     {clahe: true, clahe_clip: 2.0, clahe_grid: [8,8]}
tiling:      {enabled: true, grid: [2,2], overlap: 0.2}
tray_plane:  {enabled: true, ransac_iters: 200, ransac_threshold_mm: 5.0}
smoothing:   {enabled: true, alpha: 0.4, reset_px_dist: 30.0, max_miss_frames: 10}
pick_strategy: {mode: "top_score"}        # or "all"
ros2:        {topic, camera_frame, robot_frame, use_tf_lookup, static_tf{...}}
```
`app_detect.py` loads `config/default.yaml` (camera + depth) and deep-merges
this overlay on top, so per-eye/SGBM/VPI params stay single-sourced.

## Run

```bash
python app.py            # App 1 (calibration + depth viewer)

# App 2 — ROS2 env must be sourced (humble or jazzy):
#   sudo apt install ros-<distro>-vision-msgs ros-<distro>-tf2-ros
#   source /opt/ros/<distro>/setup.bash
python app_detect.py
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
