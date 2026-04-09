#!/usr/bin/env python
"""Add observation.images.goal as a video feature to a LeRobot dataset.

Goal images must have been captured during recording with:
    lerobot-record --dataset.record_goal_image=true --dataset.goal_image_camera_key=<key>

They are saved as:
    {dataset_root}/goal_images/episode_XXXXXX.png

This script converts each PNG into a constant-frame mp4 (every frame = the goal image,
duration = episode length) and registers it as a proper LeRobot video feature named
'observation.images.goal'. The video files follow LeRobot's chunked format so the
standard dataset loader picks them up transparently.

Run once per dataset, after collection, before training with --policy.use_goal_image=true.

Example
-------
    # Policy 2 core dataset:
    python scripts/add_goal_image_feature.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_core

    # Policy 1 goal dataset:
    python scripts/add_goal_image_feature.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_goal_batch1

    # Dry run to check for missing images before committing:
    python scripts/add_goal_image_feature.py \\
        --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_core \\
        --dry_run

Verify after running
--------------------
    python -c "
    from pathlib import Path
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(
        'rgragulraj/policy2_core',
        root=Path.home() / '.cache/huggingface/lerobot/rgragulraj/policy2_core',
    )
    assert 'observation.images.goal' in ds.meta.video_keys, 'feature not registered'
    item = ds[0]
    print('goal image shape:', item['observation.images.goal'].shape)
    print('OK')
    "
"""

import argparse
import json
import shutil
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Ensure the lerobot package is importable.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lerobot.datasets.utils import (
    DEFAULT_VIDEO_PATH,
    INFO_PATH,
    get_file_size_in_mb,
    write_info,
)
from lerobot.datasets.video_utils import (
    concatenate_video_files,
    get_video_duration_in_s,
    get_video_info,
)

GOAL_FEATURE_KEY = "observation.images.goal"
EPISODES_DIR = "meta/episodes"
DEFAULT_VIDEO_FILE_SIZE_MB = 512  # matches LeRobot default


# ---------------------------------------------------------------------------
# Video creation helper
# ---------------------------------------------------------------------------


def _create_constant_video(goal_path: Path, output_path: Path, num_frames: int, fps: int) -> None:
    """Encode an mp4 where every frame is identical to *goal_path*.

    Args:
        goal_path: Path to the source goal image PNG.
        output_path: Destination mp4 path (parent directory must exist).
        num_frames: How many frames to write (= episode length in frames).
        fps: Frame rate of the output video — must match the dataset fps.
    """
    img = Image.open(goal_path).convert("RGB")
    width, height = img.size

    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("libsvtav1", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        # Use the same encoding options as LeRobot's encode_video_frames.
        stream.options = {"g": "2", "crf": "30"}

        time_base = Fraction(1, fps)
        for i in range(num_frames):
            frame = av.VideoFrame.from_ndarray(np.array(img), format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts = i
            frame.time_base = time_base
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add observation.images.goal video feature to a LeRobot dataset.",
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
        "--video_file_size_mb",
        type=int,
        default=DEFAULT_VIDEO_FILE_SIZE_MB,
        help=f"Max video file size in MB before starting a new chunk file (default: {DEFAULT_VIDEO_FILE_SIZE_MB}).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Verify goal images exist and print what would happen — write nothing.",
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

    # -----------------------------------------------------------------------
    # Load dataset info
    # -----------------------------------------------------------------------
    with open(info_path) as f:
        info = json.load(f)

    fps: int = info["fps"]
    total_episodes: int = info["total_episodes"]
    chunks_size: int = info.get("chunks_size", 1000)

    if GOAL_FEATURE_KEY in info.get("features", {}):
        print(f"ERROR: Feature '{GOAL_FEATURE_KEY}' already exists in this dataset.")
        print("Remove the existing videos and reset info.json manually if you want to re-run.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Verify goal images
    # -----------------------------------------------------------------------
    goal_images_dir = dataset_root / "goal_images"
    if not goal_images_dir.is_dir():
        print(f"ERROR: goal_images/ directory not found at {goal_images_dir}")
        print("Record with --dataset.record_goal_image=true to generate goal images.")
        sys.exit(1)

    missing = [
        ep_idx
        for ep_idx in range(total_episodes)
        if not (goal_images_dir / f"episode_{ep_idx:06d}.png").exists()
    ]
    if missing:
        count = len(missing)
        preview = missing[:10]
        suffix = "..." if count > 10 else ""
        print(f"ERROR: Missing goal images for {count} episode(s): {preview}{suffix}")
        sys.exit(1)

    print(f"✓ All {total_episodes} goal images found.")

    # -----------------------------------------------------------------------
    # Load episode lengths from episodes parquet
    # -----------------------------------------------------------------------
    ep_parquet_files = sorted((dataset_root / EPISODES_DIR).glob("**/*.parquet"))
    if not ep_parquet_files:
        print(f"ERROR: No episode parquet files found in {dataset_root / EPISODES_DIR}")
        sys.exit(1)

    # Load with pandas to preserve all columns (including stats/).
    ep_dfs = [pd.read_parquet(f) for f in ep_parquet_files]
    ep_df_combined = pd.concat(ep_dfs, ignore_index=True).sort_values("episode_index").reset_index(drop=True)

    if len(ep_df_combined) != total_episodes:
        print(
            f"ERROR: info.json says {total_episodes} episodes but episodes parquet has "
            f"{len(ep_df_combined)} rows."
        )
        sys.exit(1)

    if args.dry_run:
        total_frames = ep_df_combined["length"].sum()
        print(f"[dry run] Would process {total_episodes} episodes, {total_frames} total frames at {fps} fps.")
        print("[dry run] Goal videos would be written to:")
        print(
            f"  {dataset_root / DEFAULT_VIDEO_PATH.format(video_key=GOAL_FEATURE_KEY, chunk_index=0, file_index=0)}"
        )
        print("[dry run] No files written.")
        return

    # -----------------------------------------------------------------------
    # Create goal image videos and collect episode metadata
    # -----------------------------------------------------------------------
    episode_video_meta: list[dict] = []

    # Current video file state
    chunk_idx = 0
    file_idx = 0
    current_file_path: Path | None = None
    current_file_duration_s: float = 0.0

    with tempfile.TemporaryDirectory() as _tmp_root_str:
        tmp_root = Path(_tmp_root_str)

        for ep_idx in tqdm(range(total_episodes), desc="Encoding goal videos"):
            goal_png = goal_images_dir / f"episode_{ep_idx:06d}.png"
            ep_length: int = int(
                ep_df_combined.loc[ep_df_combined["episode_index"] == ep_idx, "length"].iloc[0]
            )

            # Create per-episode goal video in a temp location.
            tmp_mp4 = tmp_root / f"ep_{ep_idx:06d}.mp4"
            _create_constant_video(goal_png, tmp_mp4, ep_length, fps)

            ep_size_mb: float = get_file_size_in_mb(tmp_mp4)
            ep_duration_s: float = get_video_duration_in_s(tmp_mp4)

            if current_file_path is None:
                # Very first episode — initialise the first video file.
                current_file_path = dataset_root / DEFAULT_VIDEO_PATH.format(
                    video_key=GOAL_FEATURE_KEY,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )
                current_file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(tmp_mp4), str(current_file_path))
                from_ts = 0.0
            else:
                current_size_mb: float = get_file_size_in_mb(current_file_path)

                if current_size_mb + ep_size_mb >= args.video_file_size_mb:
                    # Current file would exceed size limit — start a new file.
                    file_idx += 1
                    if file_idx >= chunks_size:
                        chunk_idx += 1
                        file_idx = 0
                    current_file_path = dataset_root / DEFAULT_VIDEO_PATH.format(
                        video_key=GOAL_FEATURE_KEY,
                        chunk_index=chunk_idx,
                        file_index=file_idx,
                    )
                    current_file_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(tmp_mp4), str(current_file_path))
                    from_ts = 0.0
                    current_file_duration_s = 0.0
                else:
                    # Append to the current file.
                    from_ts = current_file_duration_s
                    concatenate_video_files([current_file_path, tmp_mp4], current_file_path)

            current_file_duration_s = from_ts + ep_duration_s

            episode_video_meta.append(
                {
                    f"videos/{GOAL_FEATURE_KEY}/chunk_index": chunk_idx,
                    f"videos/{GOAL_FEATURE_KEY}/file_index": file_idx,
                    f"videos/{GOAL_FEATURE_KEY}/from_timestamp": from_ts,
                    f"videos/{GOAL_FEATURE_KEY}/to_timestamp": from_ts + ep_duration_s,
                }
            )

    # -----------------------------------------------------------------------
    # Update episodes parquet files
    # -----------------------------------------------------------------------
    print("Updating episode metadata...")

    # Build a combined DataFrame of the new columns, indexed by episode_index.
    new_cols_df = pd.DataFrame(
        episode_video_meta,
        index=ep_df_combined["episode_index"].values,
    )
    new_cols_df.index.name = "episode_index"

    # Rewrite each original parquet file with the new columns added.
    row_cursor = 0
    for parquet_file in ep_parquet_files:
        df = pd.read_parquet(parquet_file)
        n_rows = len(df)
        for col in new_cols_df.columns:
            # Match by episode_index to be safe (avoids ordering issues across files).
            df[col] = df["episode_index"].map(new_cols_df[col])
        df.to_parquet(parquet_file, index=False)
        row_cursor += n_rows

    # -----------------------------------------------------------------------
    # Update info.json
    # -----------------------------------------------------------------------
    print("Updating info.json...")

    first_video_path = dataset_root / DEFAULT_VIDEO_PATH.format(
        video_key=GOAL_FEATURE_KEY, chunk_index=0, file_index=0
    )
    video_info = get_video_info(first_video_path)

    # Infer shape from video info.
    h = video_info.get("video.height", video_info.get("height", None))
    w = video_info.get("video.width", video_info.get("width", None))
    if h is None or w is None:
        print("WARNING: Could not determine video dimensions from video_info; using 480×640 as fallback.")
        h, w = 480, 640

    info["features"][GOAL_FEATURE_KEY] = {
        "dtype": "video",
        "shape": [3, h, w],
        "names": ["channels", "height", "width"],
        "info": video_info,
    }
    write_info(info, dataset_root)

    print(f"\n✓ Done. Added '{GOAL_FEATURE_KEY}' to dataset at {dataset_root}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Video files: videos/{GOAL_FEATURE_KEY}/chunk-*/file-*.mp4")
    print("\nNext step: train with")
    print("  lerobot-train ... --policy.use_goal_image=true --policy.use_shared_goal_backbone=true")


if __name__ == "__main__":
    main()
