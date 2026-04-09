#!/usr/bin/env python3
"""Merge multiple per-shape Policy 2 datasets into a single diverse dataset.

Uses LeRobot's aggregate_datasets() to concatenate per-shape datasets collected
during Phase 3 (shape diversity) into one unified dataset for training.

Each per-shape dataset must already have spatial features added via
add_spatial_features.py before merging — the merged dataset must have consistent
feature schemas across all inputs.

Usage
-----
    conda activate lerobot

    # Basic usage — merge all per-shape datasets:
    python scripts/merge_datasets.py \\
        --datasets \\
            rgragulraj/policy2_core \\
            rgragulraj/policy2_shape_round \\
            rgragulraj/policy2_shape_dshape \\
            rgragulraj/policy2_shape_triangle \\
        --output_repo_id rgragulraj/policy2_diverse \\
        --output_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_diverse

    # With explicit per-dataset roots (if datasets are local-only and not on the Hub):
    python scripts/merge_datasets.py \\
        --datasets \\
            rgragulraj/policy2_core \\
            rgragulraj/policy2_shape_round \\
        --roots \\
            ~/.cache/huggingface/lerobot/rgragulraj/policy2_core \\
            ~/.cache/huggingface/lerobot/rgragulraj/policy2_shape_round \\
        --output_repo_id rgragulraj/policy2_diverse \\
        --output_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_diverse

After merging, verify the episode count:
    python -c "
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset('rgragulraj/policy2_diverse',
                        root='~/.cache/huggingface/lerobot/rgragulraj/policy2_diverse')
    print(f'Total episodes: {ds.num_episodes}')
    print(f'Total frames:   {ds.num_frames}')
    print(f'Features:       {list(ds.features.keys())}')
    "
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset repo IDs to merge (e.g. rgragulraj/policy2_core rgragulraj/policy2_shape_round).",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=None,
        help=(
            "Optional local root directories for each dataset, in the same order as --datasets. "
            "Required when datasets are local-only (not on the Hub). "
            "If provided, must have the same length as --datasets."
        ),
    )
    parser.add_argument(
        "--output_repo_id",
        required=True,
        help="Repo ID for the merged output dataset (e.g. rgragulraj/policy2_diverse).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Local directory to write the merged dataset (e.g. ~/.cache/huggingface/lerobot/rgragulraj/policy2_diverse).",
    )
    args = parser.parse_args()

    if args.roots is not None and len(args.roots) != len(args.datasets):
        parser.error(
            f"--roots must have the same length as --datasets ({len(args.datasets)} datasets, {len(args.roots)} roots)."
        )

    # Import here so the script fails fast if lerobot isn't installed.
    from lerobot.datasets.aggregate import aggregate_datasets

    output_root = args.output_root.expanduser()
    roots = None
    if args.roots is not None:
        roots = [Path(r).expanduser() for r in args.roots]

    logger.info(f"Merging {len(args.datasets)} dataset(s) into '{args.output_repo_id}':")
    for i, repo_id in enumerate(args.datasets):
        root_label = f" (root: {roots[i]})" if roots else ""
        logger.info(f"  [{i + 1}] {repo_id}{root_label}")
    logger.info(f"Output: {output_root}")

    aggregate_datasets(
        repo_ids=args.datasets,
        output_repo_id=args.output_repo_id,
        roots=roots,
        output_root=str(output_root),
    )

    logger.info(f"\nMerge complete. Dataset written to: {output_root}")
    logger.info(
        "Verify with:\n"
        f'  python -c "\n'
        f"  from lerobot.datasets.lerobot_dataset import LeRobotDataset\n"
        f"  ds = LeRobotDataset('{args.output_repo_id}', root='{output_root}')\n"
        f"  print('Episodes:', ds.num_episodes, 'Frames:', ds.num_frames)\n"
        f'  "'
    )


if __name__ == "__main__":
    main()
