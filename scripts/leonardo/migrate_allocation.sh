#!/bin/bash
# =============================================================================
# Migrate data between Leonardo allocations
# =============================================================================
# Copies datasets, venv, HF cache, checkpoints, and code from one allocation
# to another without re-downloading anything.
#
# Usage:
#   bash scripts/leonardo/migrate_allocation.sh \
#       AIFAC_P01_047 AIFAC_F02_024
#
# This will copy:
#   WORK: venvs, datasets (OXE + LIBERO), HF cache
#   FAST: checkpoints, wandb runs
#
# Dry-run first (recommended):
#   DRY_RUN=1 bash scripts/leonardo/migrate_allocation.sh AIFAC_P01_047 AIFAC_F02_024
# =============================================================================

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <OLD_ALLOCATION> <NEW_ALLOCATION>"
    echo "  e.g. $0 AIFAC_P01_047 AIFAC_F02_024"
    exit 1
fi

OLD_ALLOC="$1"
NEW_ALLOC="$2"
DRY_RUN="${DRY_RUN:-0}"

# --- Source and destination paths ---
OLD_FAST="/leonardo_scratch/fast/${OLD_ALLOC}/project"
OLD_WORK="/leonardo_work/${OLD_ALLOC}/project"
NEW_FAST="/leonardo_scratch/fast/${NEW_ALLOC}/project"
NEW_WORK="/leonardo_work/${NEW_ALLOC}/project"

echo "=== Leonardo Allocation Migration ==="
echo "FROM: ${OLD_ALLOC}"
echo "  FAST: ${OLD_FAST}"
echo "  WORK: ${OLD_WORK}"
echo ""
echo "TO:   ${NEW_ALLOC}"
echo "  FAST: ${NEW_FAST}"
echo "  WORK: ${NEW_WORK}"
echo ""

# --- Verify source directories exist ---
for dir in "${OLD_FAST}" "${OLD_WORK}"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Source directory not found: $dir"
        exit 1
    fi
done

# --- rsync helper ---
RSYNC_OPTS="-avh --progress"
if [ "$DRY_RUN" = "1" ]; then
    RSYNC_OPTS="${RSYNC_OPTS} --dry-run"
    echo "*** DRY RUN MODE — no files will be copied ***"
    echo ""
fi

sync_dir() {
    local src="$1"
    local dst="$2"
    local label="$3"

    if [ ! -d "$src" ]; then
        echo "  SKIP (not found): ${src}"
        return
    fi

    local size
    size=$(du -sh "$src" 2>/dev/null | cut -f1)
    echo "  ${label}: ${src} (${size})"
    echo "    -> ${dst}"

    mkdir -p "$(dirname "$dst")"
    rsync ${RSYNC_OPTS} "${src}/" "${dst}/"
    echo ""
}

# --- Create destination directory structure ---
echo "[1/5] Creating directory structure..."
if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "${NEW_FAST}/output/checkpoints"
    mkdir -p "${NEW_FAST}/output/wandb_runs"
    mkdir -p "${NEW_WORK}/venvs"
    mkdir -p "${NEW_WORK}/data/tensorflow_datasets"
    mkdir -p "${NEW_WORK}/data/huggingface_cache"
fi
echo "  Done."
echo ""

# --- Copy WORK tier (large, read-heavy) ---
echo "[2/5] Copying datasets (OXE + LIBERO)..."
sync_dir "${OLD_WORK}/data/tensorflow_datasets" \
         "${NEW_WORK}/data/tensorflow_datasets" \
         "Datasets"

echo "[3/5] Copying HuggingFace cache..."
sync_dir "${OLD_WORK}/data/huggingface_cache" \
         "${NEW_WORK}/data/huggingface_cache" \
         "HF cache"

echo "[4/5] Copying Python venv..."
sync_dir "${OLD_WORK}/venvs" \
         "${NEW_WORK}/venvs" \
         "Venvs"

# --- Copy FAST tier (I/O-intensive) ---
echo "[5/5] Copying checkpoints and wandb runs..."
sync_dir "${OLD_FAST}/output" \
         "${NEW_FAST}/output" \
         "Output (checkpoints + wandb)"

echo "=== Migration complete ==="
echo ""
echo "Next steps:"
echo "  1. Clone the code repo into the new FAST directory:"
echo "       git clone <repo_url> ${NEW_FAST}/flower_vla_pret"
echo ""
echo "  2. Update your environment variables:"
echo "       export LEONARDO_FAST=/leonardo_scratch/fast/${NEW_ALLOC}"
echo "       export LEONARDO_WORK=/leonardo_work/${NEW_ALLOC}"
echo ""
echo "  3. Update the checkpoint path in sbatch_train.sh if resuming:"
echo "       +continue_training=${NEW_FAST}/output/checkpoints/runs/2026-04-01/04-23-29/checkpoint_270000"
echo ""
echo "  4. If SLURM requires an account flag, add to sbatch scripts:"
echo "       #SBATCH --account=${NEW_ALLOC}"
echo ""
echo "  5. Submit a job:"
echo "       sbatch scripts/leonardo/sbatch_train.sh iMF"
