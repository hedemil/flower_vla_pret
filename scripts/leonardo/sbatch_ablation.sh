#!/bin/bash
# =============================================================================
# SLURM ablation script for iMF training on LEONARDO
# =============================================================================
# Launches 3 ablation runs with Hydra overrides.
#
# Usage:
#   export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047
#   export LEONARDO_WORK=/leonardo_work/AIFAC_P01_047
#   sbatch scripts/leonardo/sbatch_ablation.sh <ablation>
#
# Ablations:
#   ratio05      — ratio=0.5 (default heads)
#   heads12      — imf_head_depth=12 (default ratio)
#   both         — ratio=0.5 + imf_head_depth=12
# =============================================================================

#SBATCH --job-name=imf-ablation
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=AIFAC_F02_024

set -euo pipefail

ABLATION="${1:?Usage: sbatch sbatch_ablation.sh <ratio05|heads12|both>}"

# --- Set overrides per ablation ---
case "${ABLATION}" in
    ratio05)
        OVERRIDES="++agent.agent.ratio=0.5"
        RUN_NAME="imf_ablation_ratio05"
        ;;
    heads12)
        OVERRIDES="++agent.agent.imf_head_depth=12"
        RUN_NAME="imf_ablation_heads12"
        ;;
    both)
        OVERRIDES="++agent.agent.ratio=0.5 ++agent.agent.imf_head_depth=12"
        RUN_NAME="imf_ablation_ratio05_heads12"
        ;;
    *)
        echo "ERROR: Unknown ablation '${ABLATION}'. Choose: ratio05, heads12, both"
        exit 1
        ;;
esac

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

CODE_DIR="${FAST_PROJECT}/flower_vla_pret"
OUTPUT_DIR="${FAST_PROJECT}/output/checkpoints"
WANDB_DIR="${FAST_PROJECT}/output/wandb_runs"

VENV_DIR="${WORK_PROJECT}/venvs/flowervla"
DATA_DIR="${WORK_PROJECT}/data/tensorflow_datasets"
HF_CACHE="${WORK_PROJECT}/data/huggingface_cache"

# --- Load modules and activate venv ---
module purge
module load profile/deeplrn
module load cuda/12.1
source "${VENV_DIR}/bin/activate"

# --- Environment variables ---
export HYDRA_FULL_ERROR=1
export TORCHELASTIC_ERROR_FILE="${OUTPUT_DIR}/elastic_error_${SLURM_JOB_ID}.json"
export WANDB_MODE=offline
export WANDB_DIR="${WANDB_DIR}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_CACHE}"
export OXE_DATA_DIR="${DATA_DIR}"
export OXE_LIBERO_DIR="${DATA_DIR}/modified_libero_rlds"

# --- NCCL / InfiniBand ---
export NCCL_NET=IB
export NCCL_IB_ENABLE=1
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=WARN

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# --- Verify prerequisites ---
if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: Venv not found: ${VENV_DIR}"
    exit 1
fi
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

# --- Diagnostics ---
echo "=== iMF Ablation: ${ABLATION} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Run name: ${RUN_NAME}"
echo "Overrides: ${OVERRIDES}"
echo ""

# --- Launch training ---
cd "${CODE_DIR}"
git log --oneline -1

python -m accelerate.commands.launch --num_processes 4 --mixed_precision=bf16 \
    flower_vla/training.py \
    --config-name="meanflower_training.yaml" \
    datamodule.datasets.DATA_PATH="${DATA_DIR}" \
    log_dir="${OUTPUT_DIR}" \
    wandb.name="${RUN_NAME}_${SLURM_JOB_ID}" \
    wandb.entity=null \
    wandb.mode=offline \
    ${OVERRIDES}
