#!/bin/bash
# =============================================================================
# LEONARDO Setup Script
# =============================================================================
# Run this on the LEONARDO login node to prepare everything for training.
#
# Uses native cineca-ai modules + Python venv instead of containers,
# following CINECA best practices for A100 GPU workloads.
#
# Storage layout:
#   $FAST (1 TB) — code, checkpoints, wandb runs
#   $WORK (2 TB) — Python venv, OXE datasets, HuggingFace model cache
#
# Prerequisites:
#   - Set LEONARDO_FAST and LEONARDO_WORK, e.g.:
#       export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047
#       export LEONARDO_WORK=/leonardo_work/AIFAC_P01_047
#   - Have gsutil installed/available (for OXE data download)
#
# Usage:
#   export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047
#   export LEONARDO_WORK=/leonardo_work/AIFAC_P01_047
#   bash scripts/leonardo/setup.sh
# =============================================================================

set -euo pipefail

if [ -z "${LEONARDO_FAST:-}" ]; then
    echo "ERROR: LEONARDO_FAST is not set."
    echo "  export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047"
    exit 1
fi

if [ -z "${LEONARDO_WORK:-}" ]; then
    echo "ERROR: LEONARDO_WORK is not set."
    echo "  export LEONARDO_WORK=/leonardo_work/AIFAC_P01_047"
    exit 1
fi

FAST_PROJECT="${LEONARDO_FAST}/project"
WORK_PROJECT="${LEONARDO_WORK}/project"

echo "=== LEONARDO Setup ==="
echo "FAST directory: ${FAST_PROJECT}  (code, checkpoints, wandb)"
echo "WORK directory: ${WORK_PROJECT}  (venv, datasets, HF cache)"
echo ""

# --- 1. Create directory structure ---
echo "[1/5] Creating directory structure..."
# $FAST: code, output
mkdir -p "${FAST_PROJECT}/flower_vla_pret"
mkdir -p "${FAST_PROJECT}/output/checkpoints"
mkdir -p "${FAST_PROJECT}/output/wandb_runs"
# $WORK: venv, data
mkdir -p "${WORK_PROJECT}/venvs"
mkdir -p "${WORK_PROJECT}/data/tensorflow_datasets"
mkdir -p "${WORK_PROJECT}/data/huggingface_cache"
echo "  Done."

# --- 2. Check that the code repo is in place ---
echo ""
echo "[2/5] Checking code repository..."
CODE_DIR="${FAST_PROJECT}/flower_vla_pret"
if [ ! -f "${CODE_DIR}/setup.py" ]; then
    echo "  WARNING: Code not found at ${CODE_DIR}/"
    echo "  Clone or rsync the repo there:"
    echo "    git clone <repo_url> ${CODE_DIR}"
    echo "    # or"
    echo "    rsync -avz --exclude='.git' /local/path/flower_vla_pret/ ${CODE_DIR}/"
else
    echo "  Found code repository."
fi

# --- 3. Create Python venv with cineca-ai ---
echo ""
echo "[3/5] Creating Python virtual environment..."
VENV_DIR="${WORK_PROJECT}/venvs/flowervla"

module purge
module load profile/deeplrn
module load cineca-ai/4.3.0

if [ ! -d "${VENV_DIR}" ]; then
    echo "  Creating venv at ${VENV_DIR} (with --system-site-packages)..."
    python -m venv "${VENV_DIR}" --system-site-packages
    echo "  Venv created."
else
    echo "  Venv already exists at ${VENV_DIR}."
fi

echo "  Activating venv and installing requirements..."
source "${VENV_DIR}/bin/activate"
cd "${CODE_DIR}"
pip install -r requirements_leonardo.txt
echo "  Done."

# --- 4. Download OXE datasets ---
echo ""
echo "[4/5] Downloading OXE datasets from GCS..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LEONARDO_DATA_DIR="${WORK_PROJECT}/data/tensorflow_datasets"
bash "${SCRIPT_DIR}/download_oxe_data.sh"
echo "  Done."

# --- 5. Pre-download HuggingFace models ---
echo ""
echo "[5/5] Pre-downloading HuggingFace models..."
export HF_HOME="${WORK_PROJECT}/data/huggingface_cache"
python "${SCRIPT_DIR}/download_hf_models.py"
echo "  Done."

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Verify cineca-ai version: module load profile/deeplrn && module av cineca-ai"
echo "  2. If LIBERO data is needed, transfer modified_libero_rlds/ from HoreKa:"
echo "       rsync -avz horeka:<path>/modified_libero_rlds/ ${WORK_PROJECT}/data/tensorflow_datasets/modified_libero_rlds/"
echo "  3. Submit a test job:"
echo "       export LEONARDO_FAST=${LEONARDO_FAST}"
echo "       export LEONARDO_WORK=${LEONARDO_WORK}"
echo "       sbatch scripts/leonardo/sbatch_train.sh"
