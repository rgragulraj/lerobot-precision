#!/usr/bin/env python3
"""Add spatial conditioning features to an existing Policy 2 dataset.

Runs the WristCameraDetector on every frame of the wrist-camera video for each episode
and writes the resulting 10-float spatial token as `observation.environment_state` into
the dataset's parquet files. Updates info.json and recomputes stats.json.

This script must be run AFTER collecting data and BEFORE training. It does NOT require
re-recording. The detector calibration must already exist at scripts/wrist_calibration.json
— run `python scripts/detect_block_slot.py --calibrate` first.

For Phase 3+ multi-shape datasets, use --shape to select the correct calibration profile
(e.g. --shape=round). Each per-shape dataset must have its own calibration profile.

Usage
-----
    conda activate lerobot

    # Phase 2 (single shape — no --shape flag):
    python scripts/add_spatial_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_core \\
        --wrist_camera_key wrist \\
        --dry_run

    python scripts/add_spatial_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_core \\
        --wrist_camera_key wrist

    # Phase 3+ (multi-shape — use --shape for each per-shape dataset):
    python scripts/add_spatial_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_shape_round \\
        --wrist_camera_key wrist \\
        --shape round

After this script, verify with:
    python -c "
    import pandas as pd, glob
    files = glob.glob('~/.cache/huggingface/lerobot/rgragulraj/policy2_core/data/**/*.parquet', recursive=True)
    df = pd.read_parquet(files[0])
    print(df.columns.tolist())
    print(df['observation.environment_state'].head())
    "
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add the scripts directory to path so we can import detect_block_slot.
sys.path.insert(0, str(Path(__file__).parent))

from detect_block_slot import CALIBRATION_PATH, WristCameraDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _extract_frames_from_video(video_path: Path) -> list[np.ndarray]:
    """Extract all frames from an mp4 as a list of BGR numpy arrays."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def _get_video_path(dataset_root: Path, info: dict, ep_idx: int, video_key: str) -> Path:
    """Resolve the video file path for a given episode and camera key."""
    # Load episodes metadata to get chunk/file indices.
    episodes_dir = dataset_root / "meta" / "episodes"
    # Find all episode parquet files and load them.
    ep_files = sorted(episodes_dir.rglob("*.parquet"))
    all_eps = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
    ep = all_eps[all_eps["episode_index"] == ep_idx].iloc[0]

    vid_chunk = ep[f"videos/{video_key}/chunk_index"]
    vid_file = ep[f"videos/{video_key}/file_index"]

    video_path_template = info["video_path"]
    video_path = dataset_root / video_path_template.format(
        video_key=video_key, chunk_index=vid_chunk, file_index=vid_file
    )
    return video_path


def _get_data_parquet_paths(dataset_root: Path, info: dict) -> list[Path]:
    """Return all data parquet files sorted by path."""
    data_dir = dataset_root / "data"
    return sorted(data_dir.rglob("*.parquet"))


def process_dataset(
    dataset_root: Path,
    wrist_camera_key: str,
    calibration_path: Path = CALIBRATION_PATH,
    dry_run: bool = False,
    shape: str | None = None,
) -> None:
    """Add observation.environment_state to all parquet files in the dataset.

    Args:
        dataset_root: Root directory of the LeRobotDataset.
        wrist_camera_key: Camera key for the wrist camera (e.g. "wrist").
        calibration_path: Path to wrist_calibration.json.
        dry_run: If True, run detection and print stats but don't write anything.
        shape: Optional shape name for multi-shape (Phase 3+) calibration profiles
            (e.g. 'square', 'round'). When None, uses the flat Phase 2 format.
    """
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No info.json found at {info_path}. Is this a valid LeRobotDataset root?")

    with open(info_path) as f:
        info = json.load(f)

    obs_key = f"observation.images.{wrist_camera_key}"
    if obs_key not in info["features"]:
        raise ValueError(
            f"Camera key '{wrist_camera_key}' not found in dataset features. "
            f"Available image features: {[k for k in info['features'] if 'observation.images' in k]}"
        )

    detector = WristCameraDetector(calibration_path, shape=shape)
    shape_label = f" (shape={shape})" if shape else ""
    logger.info(f"Loaded calibration from {calibration_path}{shape_label}")

    parquet_files = _get_data_parquet_paths(dataset_root, info)
    logger.info(f"Found {len(parquet_files)} parquet file(s) to process.")

    env_state_key = "observation.environment_state"
    total_frames = 0
    zero_block_frames = 0
    zero_slot_frames = 0

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        episode_indices = df["episode_index"].unique()
        logger.info(f"Processing {pq_path.name}: {len(df)} frames, episodes {episode_indices.tolist()}")

        env_states = np.zeros((len(df), 10), dtype=np.float32)

        for ep_idx in episode_indices:
            ep_mask = df["episode_index"] == ep_idx
            ep_frame_indices = df[ep_mask]["frame_index"].values

            try:
                video_path = _get_video_path(dataset_root, info, int(ep_idx), wrist_camera_key)
            except Exception as e:
                logger.warning(f"Could not resolve video path for episode {ep_idx}: {e}. Skipping.")
                continue

            if not video_path.exists():
                logger.warning(f"Video not found: {video_path}. Skipping episode {ep_idx}.")
                continue

            logger.info(f"  Episode {ep_idx}: extracting frames from {video_path.name}")
            frames = _extract_frames_from_video(video_path)

            for _i, (df_row_idx, frame_idx) in enumerate(
                zip(df.index[ep_mask], ep_frame_indices, strict=False)
            ):
                if frame_idx >= len(frames):
                    logger.warning(
                        f"  frame_index {frame_idx} out of range for video ({len(frames)} frames). "
                        "Using zero vector."
                    )
                    continue

                token = detector.detect(frames[int(frame_idx)])
                env_states[df.index.get_loc(df_row_idx)] = token

                if np.all(token[:5] == 0):
                    zero_block_frames += 1
                if np.all(token[5:] == 0):
                    zero_slot_frames += 1
                total_frames += 1

        # Add the new column to the dataframe.
        # Store as list-of-lists so parquet can encode it as a repeated float column.
        df[env_state_key] = [row.tolist() for row in env_states]

        if not dry_run:
            # Write back using pyarrow to preserve schema.
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, pq_path)
            logger.info(f"  Written: {pq_path}")
        else:
            logger.info(f"  [DRY RUN] Would write {pq_path}")

    logger.info(
        f"\nDetection summary ({total_frames} frames total):\n"
        f"  Block detection failures: {zero_block_frames} ({100 * zero_block_frames / max(total_frames, 1):.1f}%)\n"
        f"  Slot detection failures:  {zero_slot_frames} ({100 * zero_slot_frames / max(total_frames, 1):.1f}%)\n"
    )

    if zero_slot_frames / max(total_frames, 1) > 0.10:
        logger.warning(
            "Slot detection failure rate >10%. Check calibration with:\n"
            "  python scripts/detect_block_slot.py --verify --camera_index=7"
        )

    if not dry_run:
        _update_info_json(info_path, info)
        _recompute_stats(dataset_root, info, env_state_key)


def _update_info_json(info_path: Path, info: dict) -> None:
    """Add observation.environment_state to the features dict in info.json."""
    env_state_key = "observation.environment_state"
    if env_state_key in info["features"]:
        logger.info(f"info.json already has {env_state_key}, skipping update.")
        return

    info["features"][env_state_key] = {
        "dtype": "float32",
        "shape": [10],
        "names": [
            "cx_block",
            "cy_block",
            "w_block",
            "h_block",
            "angle_block",
            "cx_slot",
            "cy_slot",
            "w_slot",
            "h_slot",
            "angle_slot",
        ],
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info(f"Updated info.json with {env_state_key} feature.")


def _recompute_stats(dataset_root: Path, info: dict, env_state_key: str) -> None:
    """Recompute and overwrite stats.json with updated mean/std for the new feature."""
    stats_path = dataset_root / "meta" / "stats.json"
    if not stats_path.exists():
        logger.info("No stats.json found — skipping stats update.")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    # Collect all env_state values across the entire dataset.
    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    all_values = []
    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path, columns=[env_state_key])
        all_values.extend(df[env_state_key].tolist())

    arr = np.array(all_values, dtype=np.float32)  # (N, 10)
    mean = arr.mean(axis=0).tolist()
    std = arr.std(axis=0).clip(min=1e-6).tolist()  # avoid division by zero during normalisation
    minimum = arr.min(axis=0).tolist()
    maximum = arr.max(axis=0).tolist()

    stats[env_state_key] = {
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Updated stats.json with {env_state_key} statistics.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Root directory of the LeRobotDataset (e.g. ~/.cache/huggingface/lerobot/rgragulraj/policy2_core).",
    )
    parser.add_argument(
        "--wrist_camera_key",
        type=str,
        default="wrist",
        help="Camera key for the wrist camera in the dataset (default: wrist).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=CALIBRATION_PATH,
        help=f"Path to wrist_calibration.json (default: {CALIBRATION_PATH}).",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help=(
            "Shape name for multi-shape (Phase 3+) calibration profiles (e.g. 'square', 'round'). "
            "Must match a profile created with: "
            "python scripts/detect_block_slot.py --calibrate --shape=<name>. "
            "When omitted, uses the flat Phase 2 format (backward compatible)."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run detection and print stats but don't write anything to disk.",
    )
    args = parser.parse_args()

    process_dataset(
        dataset_root=args.dataset_root.expanduser(),
        wrist_camera_key=args.wrist_camera_key,
        calibration_path=args.calibration,
        dry_run=args.dry_run,
        shape=args.shape,
    )


if __name__ == "__main__":
    main()
