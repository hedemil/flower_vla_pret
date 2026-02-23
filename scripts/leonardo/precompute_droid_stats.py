"""Pre-compute and cache DROID dataset statistics.

Run this once before training so the expensive statistics computation
doesn't block 4 GPUs during the actual training job.

Usage (on Leonardo):
    python scripts/leonardo/precompute_droid_stats.py \
        --data_dir $LEONARDO_WORK/project/data/tensorflow_datasets
"""

import argparse
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tensorflow as tf
import tensorflow_datasets as tfds

from flower_vla.dataset.utils.rlds_utils import compute_dataset_statistics
from flower_vla.dataset.utils.spec import ModuleSpec
from flower_vla.dataset.oxe.configs import OXE_DATASET_CONFIGS


def precompute_stats(data_dir: str, dataset_name: str = "droid"):
    """Compute and cache dataset statistics for the given dataset."""
    config = OXE_DATASET_CONFIGS[dataset_name]

    # Resolve the tfds builder name (same logic as dataset.py line 417)
    builder_name = dataset_name
    if dataset_name in ("eef_droid", "delta_droid"):
        builder_name = "droid"
    elif dataset_name == "bc_z":
        builder_name = "bc_z:0.1.0"
    elif dataset_name == "kit_irl_real_kitchen_lang_delta":
        builder_name = "kit_irl_real_kitchen_lang"

    print(f"Loading builder '{builder_name}' from {data_dir}")
    builder = tfds.builder(builder_name, data_dir=data_dir)

    # Build the restructure function (same as dataset.py)
    from flower_vla.dataset.dataset import DatasetUtils
    proprio_obs_key = config.get("proprio_obs_key")
    standardize_fn = config.get("standardize_fn")

    image_obs_keys = config.get("image_obs_keys", {})
    depth_obs_keys = config.get("depth_obs_keys", {})
    language_key = config.get("language_key")

    restructure = DatasetUtils.create_restructure_fn(
        builder_name.split(":")[0],
        image_obs_keys=image_obs_keys,
        depth_obs_keys=depth_obs_keys,
        proprio_obs_key=proprio_obs_key,
        language_key=language_key,
        standardize_fn=standardize_fn,
    )

    print(f"Computing statistics for '{dataset_name}' (builder: '{builder_name}')...")
    print("This will iterate through all trajectories — may take a while for large datasets.")

    stats = compute_dataset_statistics(
        builder=builder,
        ignore_errors=False,
        restructure_fn=restructure,
        proprio_obs_key=proprio_obs_key,
        standardize_fn=standardize_fn,
        force_recompute=False,  # Will skip if cache already exists
    )

    print(f"Done! Statistics: {stats.get('num_trajectories', '?')} trajectories, "
          f"{stats.get('num_transitions', '?')} transitions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute dataset statistics")
    parser.add_argument("--data_dir", required=True, help="Path to tensorflow_datasets directory")
    parser.add_argument("--dataset", default="droid", help="Dataset config name (default: droid)")
    args = parser.parse_args()

    precompute_stats(args.data_dir, args.dataset)
