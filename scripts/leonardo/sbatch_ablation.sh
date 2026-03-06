#!/bin/bash
# =============================================================================
# Submit iMF ablation grid (Phase 1) on LEONARDO
# =============================================================================
# Submits 4 independent SLURM jobs, one per ablation config.
# Each job uses 1 node / 4 GPUs, 15k steps, eval every 5k.
#
# Usage:
#   export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047
#   export LEONARDO_WORK=/leonardo_work/AIFAC_P01_047
#   bash scripts/leonardo/sbatch_ablation.sh
#
# Optional: override steps/time
#   MAX_STEPS=15000 WALL_TIME=08:00:00 bash scripts/leonardo/sbatch_ablation.sh
# =============================================================================

set -euo pipefail

# --- Defaults (override via env) ---
MAX_STEPS="${MAX_STEPS:-15000}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
WALL_TIME="${WALL_TIME:-08:00:00}"

# --- Paths ---
if [ -z "${LEONARDO_FAST:-}" ] || [ -z "${LEONARDO_WORK:-}" ]; then
    echo "ERROR: Set LEONARDO_FAST and LEONARDO_WORK before running."
    exit 1
fi

FAST_PROJECT="${LEONARDO_FAST}/project"
CODE_DIR="${FAST_PROJECT}/flower_vla_pret"
OUTPUT_DIR="${FAST_PROJECT}/output/checkpoints"

# --- Ablation grid (Phase 1) ---
# Format: "name|max_dudt_norm|norm_eps"
ABLATIONS=(
    "A1_baseline|50.0|0.01"
    "A2_noclip|1e10|0.01"
    "A3_bigeps|50.0|1.0"
    "A4_noclip_bigeps|1e10|1.0"
)

echo "=== iMF Ablation Grid (Phase 1) ==="
echo "Steps: ${MAX_STEPS}, Eval every: ${EVAL_EVERY}, Wall time: ${WALL_TIME}"
echo ""

for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r NAME MAX_DUDT NORM_EPS <<< "$entry"

    JOB_NAME="abl_${NAME}"
    echo "Submitting ${JOB_NAME}: max_dudt_norm=${MAX_DUDT}, norm_eps=${NORM_EPS}"

    sbatch --job-name="${JOB_NAME}" \
           --partition=boost_usr_prod \
           --qos=normal \
           --nodes=1 \
           --ntasks-per-node=1 \
           --gpus-per-node=4 \
           --cpus-per-task=32 \
           --mem=256G \
           --time="${WALL_TIME}" \
           --output="${JOB_NAME}_%j.out" \
           --error="${JOB_NAME}_%j.err" \
           --wrap="$(cat <<WRAP_EOF
set -euo pipefail

FAST_PROJECT="${FAST_PROJECT}"
WORK_PROJECT="${LEONARDO_WORK}/project"

CODE_DIR="\${FAST_PROJECT}/flower_vla_pret"
OUTPUT_DIR="\${FAST_PROJECT}/output/checkpoints"
WANDB_DIR="\${FAST_PROJECT}/output/wandb_runs"
VENV_DIR="\${WORK_PROJECT}/venvs/flowervla"
DATA_DIR="\${WORK_PROJECT}/data/tensorflow_datasets"
HF_CACHE="\${WORK_PROJECT}/data/huggingface_cache"

module purge
module load profile/deeplrn
module load cuda/12.1
source "\${VENV_DIR}/bin/activate"

export HYDRA_FULL_ERROR=1
export TORCHELASTIC_ERROR_FILE="\${OUTPUT_DIR}/elastic_error_\${SLURM_JOB_ID}.json"
export WANDB_MODE=offline
export WANDB_DIR="\${WANDB_DIR}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="\${HF_CACHE}"
export OXE_DATA_DIR="\${DATA_DIR}"
export OXE_LIBERO_DIR="\${DATA_DIR}/modified_libero_rlds"

export NCCL_NET=IB
export NCCL_IB_ENABLE=1
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=WARN

export MASTER_ADDR=\$(hostname)
export MASTER_PORT=29500

echo "=== Ablation ${NAME} ==="
echo "Job ID: \${SLURM_JOB_ID}, Node: \$(hostname)"
echo "max_dudt_norm=${MAX_DUDT}, norm_eps=${NORM_EPS}"

cd "\${CODE_DIR}"

python -m accelerate.commands.launch --num_processes 4 \\
    flower_vla/training.py \\
    datamodule.datasets.DATA_PATH="\${DATA_DIR}" \\
    log_dir="\${OUTPUT_DIR}" \\
    wandb.name=${NAME}_\${SLURM_JOB_ID} \\
    wandb.entity=null \\
    wandb.mode=offline \\
    max_train_steps=${MAX_STEPS} \\
    eval_every_n_steps=${EVAL_EVERY} \\
    save_every_n_steps=${SAVE_EVERY} \\
    trainer.agent.agent.max_dudt_norm=${MAX_DUDT} \\
    trainer.agent.agent.norm_eps=${NORM_EPS}
WRAP_EOF
)"

done

echo ""
echo "All ${#ABLATIONS[@]} jobs submitted. Use 'squeue -u \$USER' to monitor."
