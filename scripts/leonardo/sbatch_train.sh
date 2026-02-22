#!/bin/bash
# =============================================================================
# SLURM job script for FlowerVLA training on LEONARDO
# =============================================================================
# Uses pip-installed venv (no cineca-ai module).
#
# Usage:
#   export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047
#   export LEONARDO_WORK=/leonardo_work/AIFAC_P01_047
#   sbatch scripts/leonardo/sbatch_train.sh
#
# Paths are split across two storage tiers:
#   $FAST — code, checkpoints, wandb (I/O-intensive)
#   $WORK — venv, datasets, HF cache (large, read-heavy)
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

# --- Paths (split across FAST and WORK) ---
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

# FAST: code, output
CODE_DIR="${FAST_PROJECT}/flower_vla_pret"
OUTPUT_DIR="${FAST_PROJECT}/output/checkpoints"
WANDB_DIR="${FAST_PROJECT}/output/wandb_runs"

# WORK: venv, data
VENV_DIR="${WORK_PROJECT}/venvs/flowervla"
DATA_DIR="${WORK_PROJECT}/data/tensorflow_datasets"
HF_CACHE="${WORK_PROJECT}/data/huggingface_cache"

# --- Load modules and activate venv ---
module purge
module load profile/deeplrn
module load cuda/12.1
source "${VENV_DIR}/bin/activate"

# --- Environment variables for offline operation ---
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_CACHE}"
export OXE_DATA_DIR="${DATA_DIR}"
export OXE_LIBERO_DIR="${DATA_DIR}/modified_libero_rlds"

# --- NCCL / InfiniBand settings for Leonardo A100 nodes ---
export NCCL_NET=IB
export NCCL_IB_ENABLE=1
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=WARN

# --- Distributed training settings ---
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# --- Verify prerequisites ---
if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: Venv not found: ${VENV_DIR}"
    echo "  Run setup.sh first."
    exit 1
fi

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

# --- Diagnostics ---
echo "=== FlowerVLA Training on LEONARDO ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-4}"
echo "Code: ${CODE_DIR}"
echo "Venv: ${VENV_DIR}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# --- Launch training ---
cd "${CODE_DIR}"

python -m accelerate.commands.launch --num_processes 4 \
    flower_vla/training.py \
    datamodule.datasets.DATA_PATH="${DATA_DIR}" \
    log_dir="${OUTPUT_DIR}" \
    wandb.entity=null \
    wandb.mode=offline
