#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASHRC="${HOME}/.bashrc"
MARKER_BEGIN="# >>> elp_3d_cam app aliases >>>"
MARKER_END="# <<< elp_3d_cam app aliases <<<"

if [[ ! -f "${BASHRC}" ]]; then
  touch "${BASHRC}"
fi

tmp="$(mktemp)"
awk -v begin="${MARKER_BEGIN}" -v end="${MARKER_END}" '
  $0 == begin {skip=1; next}
  $0 == end {skip=0; next}
  skip != 1 {print}
' "${BASHRC}" > "${tmp}"

cat >> "${tmp}" <<EOF
${MARKER_BEGIN}
alias elp-calib='cd "${ROOT_DIR}" && python app.py'
alias elp-detect='cd "${ROOT_DIR}" && python app_detect.py'
alias elp-handeye='cd "${ROOT_DIR}" && python app_handeye.py'
alias elp-robot='cd "${ROOT_DIR}" && python app_robot_control.py'
${MARKER_END}
EOF

mv "${tmp}" "${BASHRC}"

echo "Added ELP app aliases to ${BASHRC}:"
echo "  elp-calib"
echo "  elp-detect"
echo "  elp-handeye"
echo "  elp-robot"
echo
echo "Run: source ${BASHRC}"
