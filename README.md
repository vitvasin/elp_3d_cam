# ELP 3D Stereo Camera Tool

Desktop app for the ELP 3D USB stereo camera:

1. **Visualize** synchronized left / right / depth views live.
2. **Calibrate** the stereo pair (chessboard or ChArUco target).
3. **Measure** per-pixel depth with an uncertainty band and detectable range.

## Requirements

Tested on **Ubuntu 22.04** (Python 3.10, OpenCV 4.5.4) and **Ubuntu 24.04**
(Python 3.12, OpenCV 4.6.0).

- Python 3.10+
- OpenCV with the `aruco` module — the Ubuntu `python3-opencv` package has it.
  Both the legacy aruco API (OpenCV <= 4.6) and the modern one (>= 4.7, e.g.
  `pip install opencv-contrib-python`) are supported automatically.
- `cv2.ximgproc` is optional — used only for the WLS depth post-filter. If it
  is missing the filter is disabled and the GUI checkbox is greyed out.
- `PyQt5`, `numpy`, `pyyaml`.

Install on either Ubuntu:

```bash
sudo apt install python3-opencv python3-pyqt5 python3-numpy python3-yaml
```

Or in a virtualenv:

```bash
pip install -r requirements.txt
```

Add your user to the `video` group (or rely on the device ACL) so the camera
node is accessible:

```bash
sudo usermod -aG video "$USER"   # then log out / back in
```

## Run

```bash
python app.py             # stereo calibration + depth viewer
python app_detect.py      # detection + ROS2 Detection3DArray publisher
python app_handeye.py     # ELP camera <-> MG400 hand-eye calibration
python app_robot_control.py  # MG400 pick controller for /elp/detections
```

Optional shell aliases:

```bash
./scripts/add_app_aliases.sh
source ~/.bashrc

elp-calib
elp-detect
elp-handeye
elp-robot
```

The ELP 3D is auto-detected via `/dev/v4l/by-id/` (single-device side-by-side).
Override the device in `config/default.yaml` if needed.

## Workflow

1. **Start** the camera — left/right panels show the synced split frames.
   The toolbar resolution selector includes 1080p per-eye capture
   (`3840×1080` side-by-side) when the camera/USB mode supports it, and the
   FPS selector offers 5/10/15/20/25/30 FPS presets.
   If the physical camera order is reversed, enable *Swap L/R* before
   calibrating; changing it later requires a matching calibration.
2. **Calibration** tab — pick a board type, either *Live capture* ~15-20 poses
   (covering the frame edges) or *Load folder* of stereo pairs, then *Calibrate*
   and *Save*. The result is auto-loaded on next start. Calibration files are
   resolution/order-specific; do not reuse a 720p calibration after switching
   to 1080p, or a non-swapped calibration after enabling *Swap L/R*. App 1,
   App 2, and the hand-eye app validate calibration image size before use.
3. **Depth** tab — with calibration loaded, the depth panel is colorized.
   The controls are split into common readout/display settings, VPI settings,
   and SGBM settings. Click the depth panel or the rectified RGB panels to read
   `depth (±error)` in mm plus the global detection range. RGB clicks also show
   the matched left/right pixel and disparity. Use mouse wheel or the toolbar
   zoom buttons to zoom, then drag to pan for accurate clicks.
   *Save Parameters* writes current camera/depth tuning to
   `config/saved_params.yaml`, which is merged over `config/default.yaml` on
   the next startup.
4. **Hand-eye** — source the MG400 ROS2 environment, run `app_handeye.py`,
   click the robot TCP/calibration point in the rectified-left image, collect
   at least 4 spread-out robot poses, then solve. Output is
   `config/hand_eye.yaml`. The hand-eye app also supports ArUco collection
   using stereo depth or marker-size PnP, plus an auto-calibration sequence that
   moves the robot through a configurable grid and auto-collects stable ArUco
   observations. *Use Current Pose* reads the live robot pose into the sequence
   center, and auto-collection accepts only one ArUco observation per grid pose.
   When ArUco PnP is enabled, stereo depth is disabled for marker collection and
   the marker pose comes from the calibrated camera intrinsics plus marker size.
   Its right-side Robot Control panel can launch/stop MG400
   bringup, clear/enable/disable the robot, read pose, and send a basic MoveJ.
5. **Runtime pick flow** — `app_detect.py` auto-loads `config/hand_eye.yaml`
   and publishes detections in `robot_base` when ROS2 is enabled. Then run
   `app_robot_control.py` to consume `/elp/detections` and execute MG400
   pick/place moves. Its **Manual** tab provides direct Cartesian MoveJ/MoveL,
   jog, pose readback, clear error, enable/disable, and gripper DO controls.
   It can save a home pose to `config/robot_control.yaml`. Its **Auto Loop**
   tab lets you add multiple place points and repeatedly pick the first
   detection into the next place point. Its **Camera Pick** tab starts the
   calibrated stereo camera directly, lets you click a rectified-left image
   point, transforms the clicked depth point through `config/hand_eye.yaml`,
   applies a robot Z offset, then moves to or picks that clicked point for
   end-to-end calibration testing.

## Notes

- Only the single-device **side-by-side** mode is hardware-synced. If the camera
  enumerates as two devices, the app falls back to dual capture and shows an
  "unsynchronized" warning — depth/calibration accuracy will be degraded.
- Depth uses rectified stereo disparity and `cv2.reprojectImageTo3D` with the
  calibration `Q` matrix. The default Jetson backend is NVIDIA VPI `OFA`;
  OpenCV `StereoSGBM` is the CPU fallback. VPI `CUDA` is available mainly for
  testing or when OFA is unavailable.
