"""Pre-compute and cache DROID dataset statistics.

Run this once before training so the expensive statistics computation
doesn't block 4 GPUs during the actual training job.

Usage (on Leonardo via sbatch):
    sbatch scripts/leonardo/sbatch_precompute_stats.sh

Or directly:
    python scripts/leonardo/precompute_droid_stats.py \
        --data_dir $LEONARDO_WORK/project/data/tensorflow_datasets
"""

import argparse
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flower_vla.dataset.dataset import make_dataset_from_rlds
from flower_vla.dataset.oxe.configs import OXE_DATASET_CONFIGS


def precompute_stats(data_dir: str, dataset_name: str = "droid"):
    """Compute and cache dataset statistics for the given dataset."""
    config = OXE_DATASET_CONFIGS[dataset_name]

    print(f"Pre-computing statistics for dataset '{dataset_name}'")
    print(f"Data directory: {data_dir}")
    print(f"Config: {config}")
    print()

    # Call make_dataset_from_rlds with train=False just to trigger
    # the statistics computation and caching. The returned dataset
    # object itself is not needed.
    make_dataset_from_rlds(
        name=dataset_name,
        data_dir=data_dir,
        train=True,
        image_obs_keys=config.get("image_obs_keys", {}),
        depth_obs_keys=config.get("depth_obs_keys", {}),
        proprio_obs_key=config.get("proprio_obs_key"),
        language_key=config.get("language_key"),
        standardize_fn=config.get("standardize_fn"),
        ignore_errors=config.get("ignore_errors", False),
    )

    print(f"\nDone! Statistics for '{dataset_name}' are now cached.")
    print("Future training runs will load them instantly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute dataset statistics")
    parser.add_argument("--data_dir", required=True, help="Path to tensorflow_datasets directory")
    parser.add_argument("--dataset", default="droid", help="Dataset config name (default: droid)")
    args = parser.parse_args()

    precompute_stats(args.data_dir, args.dataset)
