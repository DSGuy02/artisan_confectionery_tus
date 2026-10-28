#!/usr/bin/env bash
# ==============================================================================
# Install the controller's Python dependencies for the Webots FLATPAK.
#
# Why this is needed:
#   The Flatpak version of Webots (com.cyberbotics.webots) runs controllers with
#   its OWN sandboxed Python, not your system Python or ./venv. That Python has
#   no numpy/OpenCV and no pip, so cell_supervisor.py fails on "import cv2".
#
#   The sandbox's Python looks for user packages in:
#     ~/.var/app/com.cyberbotics.webots/data/python   (PYTHONUSERBASE=/var/data/python)
#   This script downloads matching wheels with the project venv's pip and
#   copies them there.
#
# Prerequisite: run ./setup_env.sh first (it creates ./venv, which provides pip).
#
# Works from a normal terminal AND from inside a Flatpak editor terminal
# (e.g. VSCodium), where ~/.var/app is hidden and commands must go via the host.
#
# Only needed for the Flatpak install. Native (.deb / tarball) Webots uses your
# system python3, so `pip install --user -r requirements.txt` is enough there.
#
# Re-run this after a Webots Flatpak update that changes the Python version.
# To undo: rm -rf ~/.var/app/com.cyberbotics.webots/data/python
# ==============================================================================

set -euo pipefail

APP_ID="com.cyberbotics.webots"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If we are running inside a Flatpak sandbox, run host-side commands via flatpak-spawn
if [ -f /.flatpak-info ]; then
    HOST=(flatpak-spawn --host)
else
    HOST=()
fi

# pip comes from the project venv (the host / Webots Python may not have pip)
if [ -x "${PROJECT_DIR}/venv/bin/python" ]; then
    PIP_PY="${PROJECT_DIR}/venv/bin/python"
else
    echo "[ERROR] ${PROJECT_DIR}/venv not found. Run ./setup_env.sh first."
    exit 1
fi

# Ask the Webots sandbox which Python version it uses (e.g. 3.12)
PY_VER="$("${HOST[@]}" flatpak run --command=python3 "${APP_ID}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
TARGET="${HOME}/.var/app/${APP_ID}/data/python/lib/python${PY_VER}/site-packages"

echo "[INFO] Webots Flatpak Python version: ${PY_VER}"
echo "[INFO] Target folder: ${TARGET}"

# Stage inside the project folder (visible to both the host and any editor sandbox)
STAGE="$(mktemp -d "${PROJECT_DIR}/.webots_pydeps.XXXXXX")"
trap 'rm -rf "${STAGE}"' EXIT

# opencv-python-headless (not opencv-python) avoids Qt library clashes inside Webots
"${PIP_PY}" -m pip install --quiet \
    --target "${STAGE}" \
    --python-version "${PY_VER}" \
    --only-binary=:all: \
    "numpy>=1.24.0" "opencv-python-headless>=4.8.0"

"${HOST[@]}" mkdir -p "${TARGET}"
"${HOST[@]}" cp -a "${STAGE}/." "${TARGET}/"

echo "[INFO] Verifying imports inside the Webots sandbox..."
"${HOST[@]}" flatpak run --command=python3 "${APP_ID}" -c 'import numpy, cv2; print("  numpy", numpy.__version__, "| cv2", cv2.__version__)'

# Ensure HiDPI displays under Linux/Wayland render the full 3D viewport (prevents 1/4 viewport bug)
echo "[INFO] Configuring HiDPI scaling override for Webots..."
"${HOST[@]}" flatpak override --user --env=XDG_CURRENT_DESKTOP=MATE "${APP_ID}"

echo "[SUCCESS] Webots Flatpak configured: dependencies installed and full-viewport rendering enabled."
