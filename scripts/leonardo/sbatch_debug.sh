#!/bin/bash
# =============================================================================
# SLURM debug job for FlowerVLA training on LEONARDO
# =============================================================================
# Uses boost_qos_dbg QOS for higher scheduling priority (30 min max).
# Good for verifying the setup works end-to-end before submitting long jobs.
#
# Usage:
#   export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047
#   export LEONARDO_WORK=/leonardo_work/AIFAC_P01_047
#   sbatch scripts/leonardo/sbatch_debug.sh
#
# Limits (boost_qos_dbg):
#   - Max walltime: 30 minutes
#   - Max nodes: 4
#   - Max 1 job running and/or pending per user account
# =============================================================================

#SBATCH --job-name=flowervla-dbg
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
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
export TORCHELASTIC_ERROR_FILE="${OUTPUT_DIR}/elastic_error_${SLURM_JOB_ID}.json"
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
export NCCL_DEBUG=INFO

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

# --- Quick data integrity check: verify no git-lfs pointer files ---
for ds in libero_10_no_noops libero_goal_no_noops; do
    ds_dir="${DATA_DIR}/modified_libero_rlds/${ds}"
    if [ ! -d "$ds_dir" ]; then
        echo "WARNING: Dataset directory not found: ${ds_dir}"
        continue
    fi
    first_tfrecord=$(find "$ds_dir" -name "*.tfrecord*" -print -quit 2>/dev/null || true)
    if [ -n "$first_tfrecord" ]; then
        size=$(stat --format=%s "$first_tfrecord" 2>/dev/null || echo 0)
        if [ "$size" -lt 1000 ]; then
            echo "WARNING: ${ds} tfrecord is only ${size} bytes — may be an LFS pointer file"
            echo "  Run: cd ${DATA_DIR}/modified_libero_rlds && git lfs pull"
        fi
    else
        echo "WARNING: No tfrecord files found for ${ds}"
    fi
done

# --- Diagnostics ---
echo "=== FlowerVLA DEBUG Job on LEONARDO ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-4}"
echo "Code: ${CODE_DIR}"
echo "Venv: ${VENV_DIR}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# List visible GPUs
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# --- Launch training (short run for validation) ---
cd "${CODE_DIR}"

python -m accelerate.commands.launch --num_processes 4 \
    flower_vla/training.py \
    datamodule.datasets.DATA_PATH="${DATA_DIR}" \
    log_dir="${OUTPUT_DIR}" \
    wandb.entity=null \
    wandb.mode=offline \
    max_train_steps=50
