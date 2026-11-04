#!/usr/bin/env bash
# ==============================================================================
# One-click Dependency Installer for herculens_wrapper & Herculens
#
# Features:
#   - Installs all core numerical, JAX, modeling, sampling, and astronomy deps.
#   - Excludes web GUI packages (Flask, etc.).
#   - Does NOT install herculens_wrapper into site-packages.
#   - Auto-detects active Python/Conda/Venv environment and NVIDIA GPU.
#
# Usage:
#   bash install_dependencies.sh            # Default auto-detect
#   bash install_dependencies.sh --cpu      # Force CPU-only JAX
#   bash install_dependencies.sh --cuda12   # Install JAX with CUDA 12
#   bash install_dependencies.sh --cuda13   # Install JAX with CUDA 13
#
# Custom Python:
#   PYTHON=/path/to/python bash install_dependencies.sh
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

# Select Python interpreter: default to PYTHON env var, or current python in PATH
PYTHON="${PYTHON:-$(which python3 || which python)}"

if [ -z "${PYTHON}" ] || ! command -v "${PYTHON}" &> /dev/null; then
    echo "[ERROR] Python interpreter not found. Please activate your environment or specify PYTHON=/path/to/python."
    exit 1
fi

echo "======================================================================"
echo " Starting Dependency Installation for Herculens & Lensing Workflow"
echo "======================================================================"
echo "[INFO] Using Python: $("${PYTHON}" -c 'import sys; print(sys.executable)')"
echo "[INFO] Python Version: $("${PYTHON}" -c 'import sys; print(sys.version.split()[0])')"

if [ -n "${CONDA_DEFAULT_ENV}" ]; then
    echo "[INFO] Active Conda Env: ${CONDA_DEFAULT_ENV}"
elif [ -n "${VIRTUAL_ENV}" ]; then
    echo "[INFO] Active Virtualenv: ${VIRTUAL_ENV}"
fi

if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo "[ERROR] requirements.txt not found at: ${REQUIREMENTS_FILE}"
    exit 1
fi

# Ensure pip, setuptools, and wheel are up to date
echo ""
echo "--> Upgrading pip, setuptools, and wheel..."
"${PYTHON}" -m pip install --upgrade pip setuptools wheel

# Optional JAX CUDA handling
TARGET_CUDA=""
for arg in "$@"; do
    case "$arg" in
        --cuda12) TARGET_CUDA="jax[cuda12]" ;;
        --cuda13) TARGET_CUDA="jax[cuda13]" ;;
        --cuda)   TARGET_CUDA="jax[cuda12]" ;;
        --cpu)    TARGET_CUDA="CPU" ;;
    esac
done

if [ -z "${TARGET_CUDA}" ]; then
    # Auto-detection
    if command -v nvidia-smi &> /dev/null; then
        HAS_GPU_JAX=$("${PYTHON}" -c "import jax; print(jax.default_backend() == 'gpu')" 2>/dev/null || echo "False")
        if [ "${HAS_GPU_JAX}" != "True" ]; then
            echo "[INFO] NVIDIA GPU detected and JAX GPU backend not active."
            echo "[INFO] Installing JAX with CUDA 12 support (use --cpu or --cuda13 to override)..."
            TARGET_CUDA="jax[cuda12]"
        fi
    fi
fi

if [ -n "${TARGET_CUDA}" ] && [ "${TARGET_CUDA}" != "CPU" ]; then
    echo ""
    echo "--> Installing / upgrading ${TARGET_CUDA}..."
    "${PYTHON}" -m pip install --upgrade "${TARGET_CUDA}"
fi

# Install requirements (excluding web GUI)
echo ""
echo "--> Installing dependencies from requirements.txt..."
"${PYTHON}" -m pip install -r "${REQUIREMENTS_FILE}"

# Explicitly ensure herculens is installed and satisfied
echo ""
echo "--> Verifying herculens..."
"${PYTHON}" -m pip install --upgrade "herculens>=0.3.0"

echo ""
echo "======================================================================"
echo " Verifying Installation"
echo "======================================================================"
"${PYTHON}" -c '
import sys
import numpy as np
import scipy
import astropy
import jax
import herculens
import jax_lensing_profiles
import numpyro
import optax
import h5py
import matplotlib
import corner
import arviz

jlp_ver = getattr(jax_lensing_profiles, "__version__", "installed")
print("✓ All core packages imported successfully!")
print(f"  - JAX version:           {jax.__version__} (backend: {jax.default_backend()})")
print(f"  - JAX devices:           {jax.devices()}")
print(f"  - Herculens version:     {herculens.__version__}")
print(f"  - Jax-Lensing-Profiles:  {jlp_ver}")
print(f"  - NumPyro version:       {numpyro.__version__}")
print(f"  - Optax version:         {optax.__version__}")
print(f"  - Astropy version:       {astropy.__version__}")
'

echo ""
echo "======================================================================"
echo " Installation Complete!"
echo "======================================================================"
echo ""
echo "NOTE: 'herculens_wrapper' does NOT need to be installed into site-packages."
echo "You can import and use it in your scripts via sys.path.insert, for example:"
echo ""
echo "----------------------------------------------------------------------"
cat << 'EOF'
import sys
from pathlib import Path

# Locate herculens_wrapper directory relative to your script
SCRIPT_DIR = Path(__file__).resolve().parent
WRAPPER_DIR = SCRIPT_DIR.parents[1] / "herculens_wrapper"  # or direct absolute path

if WRAPPER_DIR.is_dir() and str(WRAPPER_DIR) not in sys.path:
    sys.path.insert(0, str(WRAPPER_DIR))

# Now import herculens_wrapper
import herculens_wrapper
from herculens_wrapper.api import (
    SingleBandData,
    SingleBandModel,
    MassProfile,
    LightProfile,
    LensProfileCollection,
    SamplerConfig,
)
EOF
echo "----------------------------------------------------------------------"
