#!/bin/bash
# =============================================================================
# Download OXE datasets from Google Cloud Storage
# =============================================================================
# Run on login node (requires internet access).
# Downloads only the datasets used in the CROSS_X mix.
#
# Requires:
#   - gsutil (Google Cloud SDK)
#   - LEONARDO_DATA_DIR set to the target directory
#
# Usage:
#   export LEONARDO_DATA_DIR=/leonardo_scratch/fast/<account>/project/data/tensorflow_datasets
#   bash scripts/leonardo/download_oxe_data.sh
# =============================================================================

set -euo pipefail

if [ -z "${LEONARDO_DATA_DIR:-}" ]; then
    echo "ERROR: LEONARDO_DATA_DIR is not set."
    exit 1
fi

GCS_BASE="gs://gresearch/robotics"

# Datasets in CROSS_X mix (from mixes.py)
DATASETS=(
    "bridge_dataset"
    "fractal20220817_data"
    "dobbe"
    "bc_z"
    "cmu_play_fusion"
    "stanford_hydra_dataset_converted_externally_to_rlds"
    "droid"
    "robo_set"
    "kit_irl_real_kitchen_lang"
)

echo "Downloading OXE datasets to: ${LEONARDO_DATA_DIR}"
echo "Source: ${GCS_BASE}"
echo ""

for ds in "${DATASETS[@]}"; do
    echo "--- Downloading ${ds} ---"
    if [ -d "${LEONARDO_DATA_DIR}/${ds}" ]; then
        echo "  Already exists, skipping (delete to re-download)"
        continue
    fi
    gsutil -m cp -r "${GCS_BASE}/${ds}" "${LEONARDO_DATA_DIR}/"
    echo "  Done: ${ds}"
    echo ""
done

echo ""
echo "All CROSS_X datasets downloaded."
echo ""
echo "NOTE: LIBERO datasets (libero_10_no_noops, libero_goal_no_noops) are NOT on GCS."
echo "Transfer them manually from HoreKa:"
echo "  rsync -avz horeka:<path>/modified_libero_rlds/ ${LEONARDO_DATA_DIR}/modified_libero_rlds/"
