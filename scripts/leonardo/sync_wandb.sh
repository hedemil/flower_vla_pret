#!/bin/bash
# =============================================================================
# Sync offline wandb runs from LEONARDO to wandb.ai
# =============================================================================
# Run on a login node (has internet access).
#
# Usage:
#   export LEONARDO_FAST=/leonardo_scratch/fast/AIFAC_P01_047
#   bash scripts/leonardo/sync_wandb.sh
# =============================================================================

set -euo pipefail

if [ -z "${LEONARDO_FAST:-}" ]; then
    echo "ERROR: LEONARDO_FAST is not set."
    exit 1
fi

WANDB_DIR="${LEONARDO_FAST}/project/output/wandb_runs"

if [ ! -d "${WANDB_DIR}" ]; then
    echo "ERROR: wandb directory not found: ${WANDB_DIR}"
    exit 1
fi

echo "Syncing offline wandb runs from: ${WANDB_DIR}"
echo ""

# Find and sync all offline runs
for run_dir in "${WANDB_DIR}"/wandb/offline-run-*; do
    if [ -d "${run_dir}" ]; then
        echo "Syncing: $(basename "${run_dir}")"
        wandb sync "${run_dir}"
        echo ""
    fi
done

echo "Done. Check https://wandb.ai for synced runs."
