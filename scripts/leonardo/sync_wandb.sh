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

synced=0
skipped=0
failed=0

for run_dir in "${WANDB_DIR}"/wandb/offline-run-*; do
    if [ -d "${run_dir}" ]; then
        run_name="$(basename "${run_dir}")"
        echo "Syncing: ${run_name}"

        # Capture output to detect deleted-run errors
        output=$(wandb sync "${run_dir}" 2>&1) || true

        if echo "${output}" | grep -q "previously created and deleted"; then
            echo "  SKIPPED (deleted on wandb.ai) — removing local dir"
            rm -rf "${run_dir}"
            skipped=$((skipped + 1))
        elif echo "${output}" | grep -qi "error"; then
            echo "  WARNING: sync had errors:"
            echo "${output}" | grep -i "error" | head -3
            failed=$((failed + 1))
        else
            echo "  OK"
            synced=$((synced + 1))
        fi
        echo ""
    fi
done

echo "Done. Synced: ${synced}, Skipped (deleted): ${skipped}, Failed: ${failed}"
echo "Check https://wandb.ai for synced runs."
