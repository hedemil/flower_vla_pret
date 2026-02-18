#!/bin/bash
# =============================================================================
# LEONARDO Setup Script
# =============================================================================
# Run this on the LEONARDO login node to prepare everything for training.
#
# Prerequisites:
#   - Set LEONARDO_BASE to your scratch/project directory, e.g.:
#       export LEONARDO_BASE=/leonardo_scratch/fast/<account>
#   - Have gsutil installed/available (for OXE data download)
#   - Have python3 with transformers installed (for HF model caching)
#
# Usage:
#   export LEONARDO_BASE=/leonardo_scratch/fast/<your_account>
#   bash scripts/leonardo/setup.sh
# =============================================================================

set -euo pipefail

if [ -z "${LEONARDO_BASE:-}" ]; then
    echo "ERROR: LEONARDO_BASE is not set."
    echo "  export LEONARDO_BASE=/leonardo_scratch/fast/<your_account>"
    exit 1
fi

LEONARDO_PROJECT="${LEONARDO_BASE}/project"

echo "=== LEONARDO Setup ==="
echo "Base directory: ${LEONARDO_BASE}"
echo "Project directory: ${LEONARDO_PROJECT}"
echo ""

# --- 1. Create directory structure ---
echo "[1/5] Creating directory structure..."
mkdir -p "${LEONARDO_PROJECT}/flower_vla_pret"
mkdir -p "${LEONARDO_PROJECT}/containers"
mkdir -p "${LEONARDO_PROJECT}/data/tensorflow_datasets"
mkdir -p "${LEONARDO_PROJECT}/data/huggingface_cache"
mkdir -p "${LEONARDO_PROJECT}/output/checkpoints"
mkdir -p "${LEONARDO_PROJECT}/output/wandb_runs"
echo "  Done."

# --- 2. Check that the code repo is in place ---
echo ""
echo "[2/5] Checking code repository..."
if [ ! -f "${LEONARDO_PROJECT}/flower_vla_pret/setup.py" ]; then
    echo "  WARNING: Code not found at ${LEONARDO_PROJECT}/flower_vla_pret/"
    echo "  Clone or rsync the repo there:"
    echo "    git clone <repo_url> ${LEONARDO_PROJECT}/flower_vla_pret"
    echo "    # or"
    echo "    rsync -avz --exclude='.git' /local/path/flower_vla_pret/ ${LEONARDO_PROJECT}/flower_vla_pret/"
else
    echo "  Found code repository."
fi

# --- 3. Download OXE datasets ---
echo ""
echo "[3/5] Downloading OXE datasets from GCS..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LEONARDO_DATA_DIR="${LEONARDO_PROJECT}/data/tensorflow_datasets"
bash "${SCRIPT_DIR}/download_oxe_data.sh"
echo "  Done."

# --- 4. Pre-download HuggingFace models ---
echo ""
echo "[4/5] Pre-downloading HuggingFace models..."
export HF_HOME="${LEONARDO_PROJECT}/data/huggingface_cache"
python3 "${SCRIPT_DIR}/download_hf_models.py"
echo "  Done."

# --- 5. Check for Singularity container ---
echo ""
echo "[5/5] Checking for Singularity container..."
SIF_PATH="${LEONARDO_PROJECT}/containers/flower_vla_train.sif"
if [ ! -f "${SIF_PATH}" ]; then
    echo "  WARNING: Container not found at ${SIF_PATH}"
    echo "  Build it on a machine with Docker, then convert:"
    echo "    docker build -f Dockerfile.train -t flower_vla_train ."
    echo "    docker save flower_vla_train -o flower_vla_train.tar"
    echo "    # Transfer tar to LEONARDO, then:"
    echo "    singularity build ${SIF_PATH} docker-archive://flower_vla_train.tar"
else
    echo "  Found container: ${SIF_PATH}"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. If container is missing, build and transfer it (see above)"
echo "  2. If LIBERO data is needed, transfer modified_libero_rlds/ from HoreKa:"
echo "       rsync -avz horeka:<path>/modified_libero_rlds/ ${LEONARDO_DATA_DIR}/modified_libero_rlds/"
echo "  3. Submit a test job:"
echo "       export LEONARDO_BASE=${LEONARDO_BASE}"
echo "       sbatch scripts/leonardo/sbatch_train.sh"
