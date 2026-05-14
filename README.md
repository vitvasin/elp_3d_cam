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
python app.py
```

The ELP 3D is auto-detected via `/dev/v4l/by-id/` (single-device side-by-side).
Override the device in `config/default.yaml` if needed.

## Workflow

1. **Start** the camera — left/right panels show the synced split frames.
2. **Calibration** tab — pick a board type, either *Live capture* ~15-20 poses
   (covering the frame edges) or *Load folder* of stereo pairs, then *Calibrate*
   and *Save*. The result is auto-loaded on next start.
3. **Depth** tab — with calibration loaded, the depth panel is colorized.
   Tune SGBM sliders, click a pixel to read `depth (±error)` in mm plus the
   global detection range.

## Notes

- Only the single-device **side-by-side** mode is hardware-synced. If the camera
  enumerates as two devices, the app falls back to dual capture and shows an
  "unsynchronized" warning — depth/calibration accuracy will be degraded.
- Disparity from `StereoSGBM` is 16-subpixel fixed-point; depth comes from
  `cv2.reprojectImageTo3D` using the rectification `Q` matrix.
