#!/bin/bash
# =============================================================================
# SLURM job script for FlowerVLA training on LEONARDO
# =============================================================================
# Usage:
#   export LEONARDO_BASE=/leonardo_scratch/fast/<your_account>
#   sbatch scripts/leonardo/sbatch_train.sh
#
# All paths are derived from LEONARDO_BASE. Set it before submitting.
# =============================================================================

#SBATCH --job-name=flowervla
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# --- Paths (all derived from LEONARDO_BASE) ---
if [ -z "${LEONARDO_BASE:-}" ]; then
    echo "ERROR: LEONARDO_BASE is not set."
    exit 1
fi

LEONARDO_PROJECT="${LEONARDO_BASE}/project"
CODE_DIR="${LEONARDO_PROJECT}/flower_vla_pret"
SIF_PATH="${LEONARDO_PROJECT}/containers/flower_vla_train.sif"
DATA_DIR="${LEONARDO_PROJECT}/data/tensorflow_datasets"
HF_CACHE="${LEONARDO_PROJECT}/data/huggingface_cache"
OUTPUT_DIR="${LEONARDO_PROJECT}/output/checkpoints"
WANDB_DIR="${LEONARDO_PROJECT}/output/wandb_runs"

# --- Environment variables for offline operation ---
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_CACHE}"
export OXE_DATA_DIR="${DATA_DIR}"
export OXE_LIBERO_DIR="${DATA_DIR}/modified_libero_rlds"

# --- Verify prerequisites ---
if [ ! -f "${SIF_PATH}" ]; then
    echo "ERROR: Container not found: ${SIF_PATH}"
    exit 1
fi

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

echo "=== FlowerVLA Training on LEONARDO ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-4}"
echo "Code: ${CODE_DIR}"
echo "Container: ${SIF_PATH}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# --- Launch training via Singularity ---
singularity exec --nv \
    --bind "${CODE_DIR}:/app" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${HF_CACHE}:${HF_CACHE}" \
    --bind "${OUTPUT_DIR}:${OUTPUT_DIR}" \
    --bind "${WANDB_DIR}:${WANDB_DIR}" \
    --pwd /app \
    --env WANDB_MODE=offline \
    --env WANDB_DIR="${WANDB_DIR}" \
    --env TRANSFORMERS_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env HF_HOME="${HF_CACHE}" \
    --env OXE_DATA_DIR="${DATA_DIR}" \
    --env OXE_LIBERO_DIR="${DATA_DIR}/modified_libero_rlds" \
    "${SIF_PATH}" \
    accelerate launch --num_processes 4 \
        flower_vla/training.py \
        datamodule.datasets.DATA_PATH="${DATA_DIR}" \
        log_dir="${OUTPUT_DIR}" \
        wandb.entity=null \
        wandb.mode=offline
