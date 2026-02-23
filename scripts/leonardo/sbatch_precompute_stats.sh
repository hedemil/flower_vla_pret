#!/bin/bash
# =============================================================================
# Pre-compute DROID dataset statistics (CPU-only, no GPU needed)
# =============================================================================
# Iterates through the DROID dataset to compute and cache action/proprio
# statistics. This avoids wasting GPU time during the actual training job.
#
# Usage:
#   export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047
#   export LEONARDO_WORK=/leonardo_work/AIFAC_P01_047
#   sbatch scripts/leonardo/sbatch_precompute_stats.sh
# =============================================================================

#SBATCH --job-name=droid-stats
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# --- Paths ---
if [ -z "${LEONARDO_FAST:-}" ]; then
    echo "ERROR: LEONARDO_FAST is not set."
    exit 1
fi
if [ -z "${LEONARDO_WORK:-}" ]; then
    echo "ERROR: LEONARDO_WORK is not set."
    exit 1
fi

FAST_PROJECT="${LEONARDO_FAST}/project"
WORK_PROJECT="${LEONARDO_WORK}/project"

CODE_DIR="${FAST_PROJECT}/flower_vla_pret"
VENV_DIR="${WORK_PROJECT}/venvs/flowervla"
DATA_DIR="${WORK_PROJECT}/data/tensorflow_datasets"
HF_CACHE="${WORK_PROJECT}/data/huggingface_cache"

# --- Load modules and activate venv ---
module purge
module load profile/deeplrn
module load cuda/12.1
source "${VENV_DIR}/bin/activate"

# --- Environment ---
export HYDRA_FULL_ERROR=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_CACHE}"
export OXE_DATA_DIR="${DATA_DIR}"
export OXE_LIBERO_DIR="${DATA_DIR}/modified_libero_rlds"

# --- Diagnostics ---
echo "=== Pre-computing DROID dataset statistics ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Data: ${DATA_DIR}"
echo "Python: $(which python)"
echo ""

# --- Run ---
cd "${CODE_DIR}"
python scripts/leonardo/precompute_droid_stats.py --data_dir "${DATA_DIR}" --dataset droid
