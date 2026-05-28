#!/usr/bin/env bash
# Install desktop entries for the ELP 3D Stereo Camera launcher and apps.
#
# Usage:
#   scripts/install_desktop_icon.sh            # launcher only, menu + Desktop
#   scripts/install_desktop_icon.sh --all      # launcher + 4 apps
#   scripts/install_desktop_icon.sh --no-desktop  # menu only (skip ~/Desktop)
#   scripts/install_desktop_icon.sh --uninstall

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS_DIR="${HOME}/.local/share/applications"
ICONS_DIR="${HOME}/.local/share/icons"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "${HOME}/Desktop")"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

INSTALL_ALL=0
SKIP_DESKTOP=0
UNINSTALL=0

for arg in "$@"; do
  case "${arg}" in
    --all) INSTALL_ALL=1 ;;
    --no-desktop) SKIP_DESKTOP=1 ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help)
      sed -n '1,12p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown arg: ${arg}" >&2; exit 1 ;;
  esac
done

ENTRY_IDS=(
  "elp-3d-launcher"
  "elp-3d-calib"
  "elp-3d-detect"
  "elp-3d-handeye"
  "elp-3d-robot"
)

uninstall_all() {
  for id in "${ENTRY_IDS[@]}"; do
    rm -f "${APPS_DIR}/${id}.desktop"
    rm -f "${DESKTOP_DIR}/${id}.desktop"
    rm -f "${ICONS_DIR}/${id}.png"
  done
  echo "Removed ELP desktop entries from:"
  echo "  ${APPS_DIR}"
  echo "  ${DESKTOP_DIR}"
  echo "  ${ICONS_DIR}"
  command -v update-desktop-database >/dev/null && \
    update-desktop-database "${APPS_DIR}" >/dev/null 2>&1 || true
}

if [[ "${UNINSTALL}" -eq 1 ]]; then
  uninstall_all
  exit 0
fi

mkdir -p "${APPS_DIR}" "${ICONS_DIR}"
[[ "${SKIP_DESKTOP}" -eq 0 ]] && mkdir -p "${DESKTOP_DIR}"

# Generate icons via the launcher's pixmap helper, so the icon style stays
# in sync with the in-app glyphs.
gen_icon() {
  local out="$1" glyph="$2" accent="$3"
  ELP_ROOT="${ROOT_DIR}" "${PYTHON_BIN}" - "$out" "$glyph" "$accent" <<'PYEOF'
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.environ["ELP_ROOT"])
from PyQt5.QtGui import QGuiApplication
_app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
from launcher import make_glyph_pixmap
out, glyph, accent = sys.argv[1], sys.argv[2], sys.argv[3]
pm = make_glyph_pixmap(glyph, accent, 256)
if not pm.save(out, "PNG"):
    raise SystemExit(f"failed to save {out}")
print(f"  {out}")
PYEOF
}

write_entry() {
  local id="$1" name="$2" comment="$3" script="$4" icon="$5"
  local exec_cmd
  if [[ -n "${script}" ]]; then
    exec_cmd="${PYTHON_BIN} ${ROOT_DIR}/${script}"
  else
    exec_cmd="${PYTHON_BIN} ${ROOT_DIR}/launcher.py"
  fi

  local desktop_file="${APPS_DIR}/${id}.desktop"
  cat > "${desktop_file}" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=${name}
Comment=${comment}
Exec=${exec_cmd}
Path=${ROOT_DIR}
Icon=${icon}
Terminal=false
Categories=Development;Science;Robotics;
StartupNotify=true
StartupWMClass=${id}
EOF
  chmod +x "${desktop_file}"
  echo "  ${desktop_file}"

  if [[ "${SKIP_DESKTOP}" -eq 0 ]]; then
    cp "${desktop_file}" "${DESKTOP_DIR}/${id}.desktop"
    chmod +x "${DESKTOP_DIR}/${id}.desktop"
    # gio set sets the "trusted" flag on GNOME-based desktops so the icon
    # works without the user right-clicking "Allow Launching".
    command -v gio >/dev/null && \
      gio set "${DESKTOP_DIR}/${id}.desktop" metadata::trusted true 2>/dev/null || true
    echo "  ${DESKTOP_DIR}/${id}.desktop"
  fi
}

echo "Generating icons in ${ICONS_DIR} ..."
# Launcher icon: prefer the worm mascot image, fall back to a generated glyph.
if [[ -f "${ROOT_DIR}/assets/worm.png" ]]; then
  cp "${ROOT_DIR}/assets/worm.png" "${ICONS_DIR}/elp-3d-launcher.png"
  echo "  ${ICONS_DIR}/elp-3d-launcher.png (worm)"
else
  gen_icon "${ICONS_DIR}/elp-3d-launcher.png" "WS" "#3DF5A1"
fi
# App icons: prefer the generated assets/icon_*.png, fall back to glyphs.
app_icon() {
  local out="$1" asset="$2" glyph="$3" accent="$4"
  if [[ -f "${ROOT_DIR}/assets/${asset}" ]]; then
    cp "${ROOT_DIR}/assets/${asset}" "${out}"
    echo "  ${out} (${asset})"
  else
    gen_icon "${out}" "${glyph}" "${accent}"
  fi
}
if [[ "${INSTALL_ALL}" -eq 1 ]]; then
  app_icon "${ICONS_DIR}/elp-3d-calib.png"   "icon_calib.png"   "CAL" "#00E5FF"
  app_icon "${ICONS_DIR}/elp-3d-detect.png"  "icon_detect.png"  "DET" "#3DF5A1"
  app_icon "${ICONS_DIR}/elp-3d-handeye.png" "icon_handeye.png" "H-E" "#FFC24B"
  app_icon "${ICONS_DIR}/elp-3d-robot.png"   "icon_robot.png"   "BOT" "#FF4D6D"
fi

echo "Writing .desktop entries:"
write_entry "elp-3d-launcher" \
  "Worm Sorter" \
  "Worm Sorter launcher for the ELP 3D stereo camera app suite" \
  "launcher.py" \
  "${ICONS_DIR}/elp-3d-launcher.png"

if [[ "${INSTALL_ALL}" -eq 1 ]]; then
  write_entry "elp-3d-calib" \
    "ELP Stereo Calibration" \
    "Calibrate ELP stereo camera and view live depth" \
    "app.py" \
    "${ICONS_DIR}/elp-3d-calib.png"

  write_entry "elp-3d-detect" \
    "ELP Object Detection" \
    "Tiled YOLO detection with ROS2 publisher" \
    "app_detect.py" \
    "${ICONS_DIR}/elp-3d-detect.png"

  write_entry "elp-3d-handeye" \
    "ELP Hand-Eye Calibration" \
    "ELP <-> MG400 eye-to-hand calibration" \
    "app_handeye.py" \
    "${ICONS_DIR}/elp-3d-handeye.png"

  write_entry "elp-3d-robot" \
    "ELP Robot Pick Control" \
    "MG400 pick controller consuming /elp/detections" \
    "app_robot_control.py" \
    "${ICONS_DIR}/elp-3d-robot.png"
fi

command -v update-desktop-database >/dev/null && \
  update-desktop-database "${APPS_DIR}" >/dev/null 2>&1 || true

echo
echo "Done."
echo "Launcher available in your application menu as 'ELP 3D Camera Launcher'."
if [[ "${SKIP_DESKTOP}" -eq 0 ]]; then
  echo "Desktop shortcut placed in: ${DESKTOP_DIR}"
fi
if [[ "${INSTALL_ALL}" -eq 0 ]]; then
  echo "Run with --all to also install individual app icons."
fi
