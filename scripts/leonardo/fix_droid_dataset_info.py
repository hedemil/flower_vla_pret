"""Truncate DROID dataset_info.json to match available shards on Leonardo.

Leonardo only has 1061 of 2048 DROID shards (files 00000-01060).
TFDS validates all declared shards exist before loading, so we must
edit dataset_info.json to declare only the available shards.

Usage:
    python scripts/leonardo/fix_droid_dataset_info.py /path/to/droid/dataset_info.json

The script:
1. Backs up the original file to dataset_info.json.bak
2. Truncates shardLengths from 2048 to 1061 entries
3. Recalculates numBytes proportionally
4. Writes the modified file
"""

import argparse
import json
import shutil
from pathlib import Path


NUM_AVAILABLE_SHARDS = 1061


def fix_dataset_info(dataset_info_path: Path) -> None:
    backup_path = dataset_info_path.with_suffix(".json.bak")
    if not backup_path.exists():
        shutil.copy2(dataset_info_path, backup_path)
        print(f"Backed up original to {backup_path}")
    else:
        print(f"Backup already exists at {backup_path}, skipping backup")

    with open(dataset_info_path) as f:
        info = json.load(f)

    splits = info["splits"]
    train_split = None
    for split in splits:
        if split["name"] == "train":
            train_split = split
            break

    if train_split is None:
        raise ValueError("No 'train' split found in dataset_info.json")

    shard_lengths = train_split["shardLengths"]
    original_num_shards = len(shard_lengths)
    original_num_bytes = int(train_split["numBytes"])

    if original_num_shards <= NUM_AVAILABLE_SHARDS:
        print(
            f"Dataset already has {original_num_shards} shards "
            f"(<= {NUM_AVAILABLE_SHARDS}), nothing to do"
        )
        return

    truncated_shard_lengths = shard_lengths[:NUM_AVAILABLE_SHARDS]

    original_num_examples = sum(int(s) for s in shard_lengths)
    truncated_num_examples = sum(int(s) for s in truncated_shard_lengths)

    ratio = truncated_num_examples / original_num_examples
    new_num_bytes = int(original_num_bytes * ratio)

    train_split["shardLengths"] = truncated_shard_lengths
    train_split["numBytes"] = str(new_num_bytes)

    with open(dataset_info_path, "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    print(f"Shards: {original_num_shards} -> {NUM_AVAILABLE_SHARDS}")
    print(f"Examples: {original_num_examples} -> {truncated_num_examples}")
    print(f"Bytes: {original_num_bytes} -> {new_num_bytes}")
    print(f"Data retained: {ratio:.1%}")
    print(f"Written to {dataset_info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Truncate DROID dataset_info.json to match available shards"
    )
    parser.add_argument(
        "dataset_info_path",
        type=Path,
        help="Path to dataset_info.json",
    )
    args = parser.parse_args()

    path = args.dataset_info_path
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    fix_dataset_info(path)


if __name__ == "__main__":
    main()
