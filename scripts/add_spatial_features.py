#!/usr/bin/env python3
"""Add spatial conditioning features to an existing LeRobot dataset.

Supports two detector backends:

  wrist_shape (Policy 2 default)
    Runs the WristCameraDetector (CLAHE + Canny + shape template matching) on every
    frame of the wrist-camera video. Outputs a 10-float spatial token per frame.
    Calibration: scripts/wrist_calibration.json (from detect_block_slot.py).

  hsv_top (Policy 1)
    Runs the TopCameraDetector (HSV colour masking) on every frame of the top-down
    camera video. Outputs an 8-float token (or 10-float with --include_angle).
    Calibration: scripts/top_calibration.json (from detect_block_slot_hsv.py).

Both write `observation.environment_state` into the dataset's parquet files and update
info.json and stats.json. This script must be run AFTER data collection and BEFORE training.

Usage
-----
    conda activate lerobot

    # Policy 2 — wrist camera, shape matching (default):
    python scripts/add_spatial_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_core \\
        --camera_key wrist \\
        --detector_type wrist_shape \\
        --dry_run

    python scripts/add_spatial_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_core \\
        --camera_key wrist \\
        --detector_type wrist_shape

    # Policy 2 Phase 3+ (multi-shape):
    python scripts/add_spatial_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_shape_round \\
        --camera_key wrist \\
        --detector_type wrist_shape \\
        --shape round

    # Policy 1 — top-down camera, HSV (8-float):
    python scripts/add_spatial_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_diverse \\
        --camera_key top \\
        --detector_type hsv_top \\
        --dry_run

    python scripts/add_spatial_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_diverse \\
        --camera_key top \\
        --detector_type hsv_top

    # Policy 1 with angles (10-float):
    python scripts/add_spatial_features.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_diverse \\
        --camera_key top \\
        --detector_type hsv_top \\
        --include_angle

After this script, verify with:
    python -c "
    import pandas as pd, glob
    files = glob.glob(
        '~/.cache/huggingface/lerobot/rgragulraj/policy1_diverse/data/**/*.parquet',
        recursive=True
    )
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
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add the scripts directory to path so we can import the detector modules.
sys.path.insert(0, str(Path(__file__).parent))

from detect_block_slot import (  # noqa: E402  # noqa: E402
    CALIBRATION_PATH as WRIST_CALIBRATION_PATH,  # noqa: E402
    TOP_CALIBRATION_PATH,
    TopCameraShapeDetector,
    WristCameraDetector,
)
from detect_block_slot_hsv import (
    CALIBRATION_PATH as HSV_TOP_CALIBRATION_PATH,  # noqa: E402
    TopCameraDetector,  # noqa: E402
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Backwards-compatible alias used in the original Policy 2 script.
CALIBRATION_PATH = WRIST_CALIBRATION_PATH


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


def _build_detector(
    detector_type: str,
    calibration_path: Path | None,
    include_angle: bool,
    shape: str | None,
    match_threshold: float | None,
    min_area: int | None,
):
    """Instantiate the correct detector and return (detector, token_dim).

    Args:
        detector_type: 'wrist_shape', 'shape_top', or 'hsv_top'.
        calibration_path: Override for the calibration file path.
        include_angle: If True, output 10-float token (only relevant for hsv_top).
        shape: Shape profile name for multi-shape datasets.
        match_threshold: Override match threshold (shape_top and wrist_shape only).
        min_area: Override minimum contour area (shape_top only).

    Returns:
        Tuple of (detector_instance, token_dim).
    """
    if detector_type == "wrist_shape":
        cal_path = calibration_path or WRIST_CALIBRATION_PATH
        wrist_kwargs: dict[str, Any] = {"shape": shape}
        if match_threshold is not None:
            wrist_kwargs["match_threshold"] = match_threshold
        detector_w = WristCameraDetector(cal_path, **wrist_kwargs)
        return detector_w, 10
    elif detector_type == "shape_top":
        cal_path = calibration_path or TOP_CALIBRATION_PATH
        top_kwargs: dict[str, Any] = {"shape": shape}
        if match_threshold is not None:
            top_kwargs["match_threshold"] = match_threshold
        if min_area is not None:
            top_kwargs["min_area"] = min_area
        detector_t = TopCameraShapeDetector(cal_path, **top_kwargs)
        return detector_t, 10  # always 10 floats (same token format as wrist_shape)
    elif detector_type == "hsv_top":
        cal_path = calibration_path or HSV_TOP_CALIBRATION_PATH
        detector_h = TopCameraDetector(cal_path, include_angle=include_angle)
        token_dim = 10 if include_angle else 8
        return detector_h, token_dim
    else:
        raise ValueError(
            f"Unknown detector_type: '{detector_type}'. Must be 'wrist_shape', 'shape_top', or 'hsv_top'."
        )


def process_dataset(
    dataset_root: Path,
    camera_key: str,
    detector_type: str = "wrist_shape",
    calibration_path: Path | None = None,
    dry_run: bool = False,
    shape: str | None = None,
    include_angle: bool = False,
    match_threshold: float | None = None,
    min_area: int | None = None,
    # Legacy alias kept for backwards compatibility with existing Policy 2 call sites.
    wrist_camera_key: str | None = None,
) -> None:
    """Add observation.environment_state to all parquet files in the dataset.

    Args:
        dataset_root: Root directory of the LeRobotDataset.
        camera_key: Camera key to run detection on (e.g. "wrist" for Policy 2, "top" for Policy 1).
        detector_type: 'wrist_shape' (Policy 2 — CLAHE+Canny+shape template) or
            'hsv_top' (Policy 1 — HSV colour masking).
        calibration_path: Override for the calibration JSON path. When None, uses the
            default path for the selected detector type.
        dry_run: If True, run detection and print stats but don't write anything.
        shape: Shape profile name for Policy 2 multi-shape (Phase 3+) datasets.
        include_angle: If True, output a 10-float token with rotation angles.
            Only applies to hsv_top; wrist_shape always outputs 10 floats.
        wrist_camera_key: Deprecated alias for camera_key. Kept for backwards compatibility.
    """
    # Backwards-compatible alias.
    if wrist_camera_key is not None:
        camera_key = wrist_camera_key

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"No info.json found at {info_path}. Is this a valid LeRobotDataset root?")

    with open(info_path) as f:
        info = json.load(f)

    obs_key = f"observation.images.{camera_key}"
    if obs_key not in info["features"]:
        raise ValueError(
            f"Camera key '{camera_key}' not found in dataset features. "
            f"Available image features: {[k for k in info['features'] if 'observation.images' in k]}"
        )

    detector, token_dim = _build_detector(
        detector_type, calibration_path, include_angle, shape, match_threshold, min_area
    )
    shape_label = f" (shape={shape})" if shape else ""
    logger.info(f"Detector: {detector_type} | token_dim={token_dim} | camera_key={camera_key}{shape_label}")

    parquet_files = _get_data_parquet_paths(dataset_root, info)
    logger.info(f"Found {len(parquet_files)} parquet file(s) to process.")

    env_state_key = "observation.environment_state"
    total_frames = 0
    zero_block_frames = 0
    zero_slot_frames = 0
    half = token_dim // 2  # first half = block, second half = slot

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        episode_indices = df["episode_index"].unique()
        logger.info(f"Processing {pq_path.name}: {len(df)} frames, episodes {episode_indices.tolist()}")

        env_states = np.zeros((len(df), token_dim), dtype=np.float32)

        for ep_idx in episode_indices:
            ep_mask = df["episode_index"] == ep_idx
            ep_frame_indices = df[ep_mask]["frame_index"].values

            try:
                video_path = _get_video_path(dataset_root, info, int(ep_idx), camera_key)
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

                if np.all(token[:half] == 0):
                    zero_block_frames += 1
                if np.all(token[half:] == 0):
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
        _update_info_json(info_path, info, token_dim)
        _recompute_stats(dataset_root, info, env_state_key)


# Feature name lists per token dimension.
_NAMES_8 = ["cx_block", "cy_block", "w_block", "h_block", "cx_slot", "cy_slot", "w_slot", "h_slot"]
_NAMES_10 = [
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
]


def _update_info_json(info_path: Path, info: dict, token_dim: int = 10) -> None:
    """Add observation.environment_state to the features dict in info.json.

    Args:
        info_path: Path to info.json.
        info: Loaded info dict (modified in place).
        token_dim: Dimension of the spatial token (8 or 10).
    """
    env_state_key = "observation.environment_state"
    if env_state_key in info["features"]:
        logger.info(f"info.json already has {env_state_key}, skipping update.")
        return

    names = _NAMES_10 if token_dim == 10 else _NAMES_8
    info["features"][env_state_key] = {
        "dtype": "float32",
        "shape": [token_dim],
        "names": names,
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info(f"Updated info.json with {env_state_key} feature (dim={token_dim}).")


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
        help="Root directory of the LeRobotDataset.",
    )
    parser.add_argument(
        "--camera_key",
        type=str,
        default=None,
        help=(
            "Camera key to run detection on. "
            "Policy 2 (wrist_shape): 'wrist'. Policy 1 (hsv_top): 'top'. "
            "Defaults to 'wrist' for wrist_shape and 'top' for hsv_top."
        ),
    )
    # Backwards-compatible alias.
    parser.add_argument(
        "--wrist_camera_key",
        type=str,
        default=None,
        help="Deprecated alias for --camera_key. Use --camera_key instead.",
    )
    parser.add_argument(
        "--detector_type",
        type=str,
        choices=["wrist_shape", "shape_top", "hsv_top"],
        default="wrist_shape",
        help=(
            "Detector backend. "
            "'wrist_shape': Policy 2 — CLAHE+Canny+shape template, wrist camera (default). "
            "'shape_top': Policy 1 — CLAHE+Canny+shape template, top-down camera. "
            "'hsv_top': Policy 1 fallback — HSV colour masking, top-down camera."
        ),
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=None,
        help=(
            "Override the match threshold for shape_top or wrist_shape detectors. "
            "Defaults to 0.30 (shape_top) or 0.25 (wrist_shape). "
            "Raise if detection rate is too low. Lower if wrong objects are matched. "
            "See tuning guide in detect_block_slot.py."
        ),
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=None,
        help=(
            "Override the minimum contour area in pixels for shape_top. "
            "Default: 500. Raise if background clutter is being detected. "
            "Lower if the object is not detected at normal working distance."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Override calibration file path. "
            "Defaults to scripts/wrist_calibration.json for wrist_shape, "
            "scripts/top_calibration.json for hsv_top."
        ),
    )
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help=(
            "Shape profile name for Policy 2 multi-shape (Phase 3+) datasets "
            "(e.g. 'square', 'round'). "
            "Must match a profile created with: "
            "python scripts/detect_block_slot.py --calibrate --shape=<name>. "
            "Not used for hsv_top."
        ),
    )
    parser.add_argument(
        "--include_angle",
        action="store_true",
        help=(
            "Output 10-float token with rotation angles instead of 8-float. "
            "Only applies to hsv_top. wrist_shape always outputs 10 floats."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run detection and print stats but don't write anything to disk.",
    )
    args = parser.parse_args()

    # Resolve camera_key default based on detector_type.
    camera_key = args.camera_key or args.wrist_camera_key
    if camera_key is None:
        camera_key = "wrist" if args.detector_type == "wrist_shape" else "top"

    process_dataset(
        dataset_root=args.dataset_root.expanduser(),
        camera_key=camera_key,
        detector_type=args.detector_type,
        calibration_path=args.calibration,
        dry_run=args.dry_run,
        shape=args.shape,
        include_angle=args.include_angle,
        match_threshold=args.match_threshold,
        min_area=args.min_area,
    )


if __name__ == "__main__":
    main()
