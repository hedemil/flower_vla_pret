#!/bin/bash
# =============================================================================
# Download OXE datasets from Google Cloud Storage
# =============================================================================
# Run on login node (requires internet access and gsutil in PATH).
# Downloads only the CROSS_X mix datasets that are available on GCS.
#
# Datasets NOT on GCS (must be transferred manually):
#   - bridge_dataset  (custom copy from rail.eecs.berkeley.edu)
#   - kit_irl_real_kitchen_lang  (KIT IRL dataset, not on GCS)
#   - libero_10_no_noops, libero_goal_no_noops  (LIBERO, transfer from HoreKa)
#
# Usage:
#   export LEONARDO_DATA_DIR=/leonardo_work/AIFAC_P01_047/project/data/tensorflow_datasets
#   bash scripts/leonardo/download_oxe_data.sh
# =============================================================================

set -euo pipefail

if [ -z "${LEONARDO_DATA_DIR:-}" ]; then
    echo "ERROR: LEONARDO_DATA_DIR is not set."
    exit 1
fi

GCS_BASE="gs://gresearch/robotics"

# CROSS_X mix datasets available on GCS (verified with gsutil ls)
DATASETS=(
    "fractal20220817_data"
    "dobbe"
    "bc_z"
    "cmu_play_fusion"
    "stanford_hydra_dataset_converted_externally_to_rlds"
    "droid"
    "robo_set"
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
echo "All GCS datasets downloaded."
echo ""
echo "=== Manual transfers still needed ==="
echo ""
echo "1. bridge_dataset (custom copy, not the GCS 'bridge'):"
echo "   Download from https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/"
echo "   Place in: ${LEONARDO_DATA_DIR}/bridge_dataset/"
echo ""
echo "2. kit_irl_real_kitchen_lang (KIT IRL dataset):"
echo "   Transfer from HoreKa or original source"
echo "   Place in: ${LEONARDO_DATA_DIR}/kit_irl_real_kitchen_lang/"
echo ""
echo "3. LIBERO datasets (libero_10_no_noops, libero_goal_no_noops):"
echo "   rsync -avz horeka:<path>/modified_libero_rlds/ ${LEONARDO_DATA_DIR}/modified_libero_rlds/"
