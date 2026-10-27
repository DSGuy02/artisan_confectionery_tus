#!/usr/bin/env bash
# ==============================================================================
# Setup Environment for Artisan Confectionery Robotics Project
# Technological University of the Shannon (TUS Athlone) - Integrated Robotics 4
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

echo "=================================================================="
echo " Setting up Python Virtual Environment for Confectionery Project "
echo "=================================================================="

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment in ${PROJECT_DIR}/venv..."
    python3 -m venv venv
else
    echo "[INFO] Existing virtual environment found."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install requirements
echo "[INFO] Installing required Python libraries..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=================================================================="
echo " Setup Completed Successfully!"
echo "=================================================================="
echo "To activate this environment in your terminal, run:"
echo "  source ${PROJECT_DIR}/venv/bin/activate"
echo ""
echo "Note for Webots Python Controllers:"
echo "Webots uses its own bundled Python controller library."
echo "If running scripts outside Webots, ensure PYTHONPATH points to:"
echo "  export WEBOTS_HOME=/usr/local/webots  # Adjust to your install path"
echo "  export PYTHONPATH=\$WEBOTS_HOME/lib/controller/python:\$PYTHONPATH"
echo "=================================================================="
