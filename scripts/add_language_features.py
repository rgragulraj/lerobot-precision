#!/usr/bin/env python
"""Add observation.language (CLIP text embeddings) as a tensor feature to a LeRobot dataset.

Each episode has a task description stored in the dataset's episodes table (column "tasks"
or "task_description") or in info.json["tasks"]. This script maps each unique task string
to a CLIP ViT-B/32 pooled text embedding (512 floats), then writes one embedding per frame
into the per-frame parquet files as a new column "observation.language".

The CLIP encoder is NOT stored in the ACT checkpoint — only the linear projection layer
lives in the model. Pre-caching here keeps the model small and allows sharing CLIP across
multiple policies.

Run once per dataset, after collection, before training with --policy.use_language_conditioning=true.

Example
-------
    # Policy 1 multi-shape dataset:
    python scripts/add_language_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_multishape

    # Dry run to see what task strings will be encoded:
    python scripts/add_language_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_multishape \\
        --dry_run

    # Override model (e.g. larger CLIP):
    python scripts/add_language_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_multishape \\
        --model_name openai/clip-vit-large-patch14

Verify after running
--------------------
    python -c "
    from pathlib import Path
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(
        'rgragulraj/policy1_multishape',
        root=Path.home() / '.cache/huggingface/lerobot/rgragulraj/policy1_multishape',
    )
    assert 'observation.language' in ds.meta.features, 'feature not registered'
    item = ds[0]
    print('language embedding shape:', item['observation.language'].shape)
    print('OK')
    "

Task description format
-----------------------
Keep task descriptions short and consistent across episodes. Examples:
    "hover above the square slot"
    "hover above the round slot"
    "hover above the hex slot"
    "hover above the triangle slot"

Each shape variant should use exactly the same phrasing — only the shape word changes.
This ensures CLIP's embedding space encodes shape identity rather than phrasing variation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure the lerobot package is importable.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lerobot.datasets.utils import INFO_PATH, write_info

LANGUAGE_FEATURE_KEY = "observation.language"
EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"


# ---------------------------------------------------------------------------
# CLIP embedding helper
# ---------------------------------------------------------------------------


def _compute_clip_embeddings(task_strings: list[str], model_name: str, device: str) -> dict[str, np.ndarray]:
    """Compute L2-normalised CLIP text embeddings for each unique task string.

    Args:
        task_strings: List of unique task description strings.
        model_name: HuggingFace model ID (e.g. "openai/clip-vit-base-patch32").
        device: Device to run CLIP on ("cpu" or "cuda").

    Returns:
        dict mapping task string → float32 numpy array of shape (embedding_dim,).
    """
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

    print(f"Loading CLIP model: {model_name} ...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings: dict[str, np.ndarray] = {}
    for task in tqdm(task_strings, desc="Computing CLIP embeddings"):
        inputs = processor(text=[task], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            features = model.get_text_features(**inputs)  # (1, dim)
            features = features / features.norm(dim=-1, keepdim=True)
        embeddings[task] = features[0].cpu().float().numpy()

    return embeddings


# ---------------------------------------------------------------------------
# Task string resolution
# ---------------------------------------------------------------------------


def _resolve_episode_tasks(info: dict, dataset_root: Path, total_episodes: int) -> list[str]:
    """Return a list of task strings, one per episode.

    Resolution order:
      1. info.json["tasks"] — list of {task_index, task} dicts (new LeRobot format)
      2. meta/episodes/*.parquet column "task_description" (fallback)
      3. meta/episodes/*.parquet column "tasks" (older convention)
      4. Raises if none found.
    """
    import pandas as pd  # noqa: PLC0415

    # --- Option 1: info.json tasks list
    if "tasks" in info and isinstance(info["tasks"], list) and len(info["tasks"]) > 0:
        # Build task_index → task_string map
        idx_to_task = {t["task_index"]: t["task"] for t in info["tasks"] if "task_index" in t and "task" in t}

        # Episodes parquet must have a "task_index" column
        ep_parquet_files = sorted((dataset_root / EPISODES_DIR).glob("**/*.parquet"))
        if ep_parquet_files:
            ep_dfs = [pd.read_parquet(f) for f in ep_parquet_files]
            ep_df = pd.concat(ep_dfs, ignore_index=True).sort_values("episode_index").reset_index(drop=True)
            if "task_index" in ep_df.columns:
                tasks = [
                    idx_to_task.get(int(ep_df.loc[ep_df["episode_index"] == i, "task_index"].iloc[0]), "")
                    for i in range(total_episodes)
                ]
                if all(tasks):
                    return tasks

    # --- Options 2 & 3: column in episodes parquet
    ep_parquet_files = sorted((dataset_root / EPISODES_DIR).glob("**/*.parquet"))
    if ep_parquet_files:
        ep_dfs = [pd.read_parquet(f) for f in ep_parquet_files]
        ep_df = pd.concat(ep_dfs, ignore_index=True).sort_values("episode_index").reset_index(drop=True)
        for col in ("task_description", "tasks"):
            if col in ep_df.columns:
                tasks = [
                    str(ep_df.loc[ep_df["episode_index"] == i, col].iloc[0]) for i in range(total_episodes)
                ]
                if all(tasks):
                    return tasks

    raise ValueError(
        "Could not resolve task strings for this dataset. Expected one of:\n"
        "  • info.json['tasks'] list with 'task_index' column in episodes parquet\n"
        "  • 'task_description' column in episodes parquet\n"
        "  • 'tasks' column in episodes parquet\n"
        "Record with --dataset.single_task='hover above the square slot' (per shape) to populate tasks."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add observation.language (CLIP embeddings) to a LeRobot dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Root directory of the LeRobot dataset (contains info.json).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model ID (default: openai/clip-vit-base-patch32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for CLIP inference: 'cpu' or 'cuda' (default: cpu).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print which task strings would be encoded — write nothing.",
    )
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root

    if not dataset_root.is_dir():
        print(f"ERROR: dataset_root does not exist: {dataset_root}")
        sys.exit(1)

    info_path = dataset_root / INFO_PATH
    if not info_path.exists():
        print(f"ERROR: info.json not found at {info_path}")
        sys.exit(1)

    with open(info_path) as f:
        info = json.load(f)

    if LANGUAGE_FEATURE_KEY in info.get("features", {}):
        print(f"ERROR: Feature '{LANGUAGE_FEATURE_KEY}' already exists in this dataset.")
        print("Remove the 'observation.language' column from data parquets and reset info.json manually.")
        sys.exit(1)

    total_episodes: int = info["total_episodes"]

    # -----------------------------------------------------------------------
    # Resolve task strings
    # -----------------------------------------------------------------------
    episode_tasks = _resolve_episode_tasks(info, dataset_root, total_episodes)

    unique_tasks = sorted(set(episode_tasks))
    print(f"Found {len(unique_tasks)} unique task string(s) across {total_episodes} episodes:")
    for t in unique_tasks:
        count = episode_tasks.count(t)
        print(f"  [{count:3d} episodes] {t!r}")

    if args.dry_run:
        print("\n[dry run] No files written.")
        return

    # -----------------------------------------------------------------------
    # Compute CLIP embeddings
    # -----------------------------------------------------------------------
    task_to_embedding = _compute_clip_embeddings(unique_tasks, args.model_name, args.device)
    embedding_dim: int = next(iter(task_to_embedding.values())).shape[0]
    print(f"Embedding dimension: {embedding_dim}")

    # -----------------------------------------------------------------------
    # Build episode_index → embedding map
    # -----------------------------------------------------------------------
    ep_to_embedding: dict[int, np.ndarray] = {
        ep_idx: task_to_embedding[episode_tasks[ep_idx]] for ep_idx in range(total_episodes)
    }

    # -----------------------------------------------------------------------
    # Write observation.language into per-frame data parquet files
    # -----------------------------------------------------------------------
    import pandas as pd  # noqa: PLC0415

    data_parquet_files = sorted((dataset_root / DATA_DIR).glob("**/*.parquet"))
    if not data_parquet_files:
        print(f"ERROR: No data parquet files found in {dataset_root / DATA_DIR}")
        sys.exit(1)

    print(f"\nUpdating {len(data_parquet_files)} data parquet file(s)...")
    for parquet_file in tqdm(data_parquet_files, desc="Writing embeddings"):
        df = pd.read_parquet(parquet_file)
        # Map each frame's episode_index to the corresponding embedding.
        embeddings = df["episode_index"].map(lambda ep: ep_to_embedding[ep].tolist())
        df[LANGUAGE_FEATURE_KEY] = embeddings
        df.to_parquet(parquet_file, index=False)

    # -----------------------------------------------------------------------
    # Update info.json
    # -----------------------------------------------------------------------
    print("Updating info.json...")
    info["features"][LANGUAGE_FEATURE_KEY] = {
        "dtype": "float32",
        "shape": [embedding_dim],
        "names": None,
    }
    write_info(info, dataset_root)

    print(f"\n✓ Done. Added '{LANGUAGE_FEATURE_KEY}' to dataset at {dataset_root}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Unique tasks: {len(unique_tasks)}")
    print(f"  Embedding dim: {embedding_dim}")
    print("\nNext step: train with")
    print("  lerobot-train ... --policy.use_language_conditioning=true --policy.language_dim=512")


if __name__ == "__main__":
    main()
