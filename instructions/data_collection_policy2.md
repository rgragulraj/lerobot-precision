# Policy 2 — Data Collection Instructions

**Author:** LENS Lab
**Hardware:** SO-101 follower + leader arm, wrist camera (index 7), top-down webcam (index 5)

---

## Overview

Policy 2 is built across four data collection phases. Phases 0 and 1 use the same hardware setup
and the same slot/block. Phase 2 (spatial conditioning) does **not** require re-recording — its
features are extracted offline from Phase 1 videos. Phases 3 and 4 extend the diversity of the
training set.

| Phase   | Dataset                                                                          | Episodes    | Goal images | Purpose                                                           |
| ------- | -------------------------------------------------------------------------------- | ----------- | ----------- | ----------------------------------------------------------------- |
| Phase 0 | `rgragulraj/policy2_baseline`                                                    | 50          | No          | Vanilla ACT baseline — measure what breaks before adding anything |
| Phase 1 | `rgragulraj/policy2_core`                                                        | 100         | Yes         | Goal image conditioning — the core Phase 1 model                  |
| Phase 2 | Same as Phase 1 (add features offline)                                           | —           | —           | Spatial conditioning — no re-recording needed                     |
| Phase 3 | `rgragulraj/policy2_shape_<name>` × N shapes → `policy2_diverse`                 | 20–25/shape | Yes         | Shape diversity — 6–8 structurally distinct shapes                |
| Phase 4 | `rgragulraj/policy2_depth_<depth_mm>` × 3 depths → merged into `policy2_diverse` | 8–12/depth  | Yes         | Slot depth variation — prevents fixed-depth descent               |

**Collection order:** Phase 0 → Phase 1 → (Phase 2 offline) → Phase 3 → Phase 4.
Phases 3 and 4 both merge into `policy2_diverse` — retrain after each phase is added.

---

## Hardware setup (do this before any phase)

### 1. Verify camera indices

```bash
conda activate lerobot
python -c "import cv2; [print(f'index {i}: OK' if cv2.VideoCapture(i).read()[0] else f'index {i}: no camera') for i in range(10)]"
```

Wrist camera should be index 7, top-down index 5. If different, update `--robot.cameras` in the
recording commands below.

### 2. Define and save the canonical hover pose

The canonical hover pose is the fixed starting arm configuration for every Policy 2 episode:

- Gripper directly above slot centre, ±0 mm
- Gripper height: ~4 cm above the top of the slot opening
- Block orientation aligned with the slot's insertion axis

If you haven't saved this pose yet:

```bash
# Teleoperate the arm to the desired hover position manually, then capture it:
python scripts/capture_start_position.py --name insert_above_slot
```

To load it at the start of each session:

```bash
python scripts/go_to_start_position.py --name insert_above_slot
```

### 3. Verify wrist camera view at hover pose

With the arm at canonical hover pose and block in gripper:

- Slot opening should be clearly visible and centred in the wrist frame.
- Block should be visible from above, roughly aligned.
- No overexposure. Consistent lighting — do not change lights between episodes.

---

## Phase 0 — Baseline

**Purpose:** Measure raw vanilla ACT precision before any architectural changes. This tells you
where the baseline fails (lateral vs. rotational vs. depth) and confirms that the wrist camera
alone provides enough signal. The baseline is required before Phase 1 — do not skip it.

**Start conditions:** Always start from the exact canonical hover pose. No deliberate variation.
Phase 0 tests the insertion task, not the alignment task.

### Recording command

```bash
conda activate lerobot

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/policy2_baseline \
    --dataset.single_task="Insert the block into the slot" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=30 \
    --dataset.push_to_hub=false \
    --display_data=true
```

### Per-episode procedure (Phase 0)

**Step 1 — Reset to canonical hover pose**

```bash
python scripts/go_to_start_position.py --name insert_above_slot
```

Place the block in the gripper by hand. Align it carefully to match the slot's insertion axis —
Phase 0 tests the policy's ability to insert from a clean start, so the manual reset should be
as consistent as possible.

**Step 2 — Record the insertion**

When the terminal prompts for the next episode, press **→** to begin recording.

Perform the insertion:

- Descend smoothly from canonical hover pose to fully seated.
- The block should seat cleanly in 2–6 seconds from a perfect start.
- Press **→** as soon as the block is fully seated.

Episode time limit is 20 seconds. If you miss or it takes longer, let the timer expire —
that episode still counts as a (failed) demonstration. Do not press → on failed insertions if
you want a clean dataset; press **←** to discard and re-record instead.

**Step 3 — Reset environment**

During the 30-second reset window:

- Remove block from slot.
- Return arm to canonical hover pose using `go_to_start_position.py`.
- Re-place block in gripper, carefully aligned.

### After Phase 0 collection

Train vanilla ACT on this dataset:

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy2_baseline \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy2_baseline \
  --policy.push_to_hub=false \
  --training.num_train_steps=50000 \
  --output_dir=outputs/policy2_baseline
```

Evaluate using the protocol in Policy2.md section 4.5 (60 trials across 7 offset/rotation
conditions). Classify failures into types (a)–(d). **Proceed to Phase 1 regardless of results.**

---

## Phase 1 — Core dataset with goal images

**Purpose:** Collect the 100-episode dataset that Phase 1 and Phase 2 will both train on.
This is the main data collection effort. Every episode requires a goal image.

**Start conditions:** Vary within the canonical envelope on every episode:

- Lateral offset: ±0.5 cm in x/y
- Rotation: ±5° from canonical
- Height: within ±0.5 cm of canonical hover height

Do not always start from the exact canonical pose — pure fixed-pose training is brittle.
Natural variation from hand-placing the block is fine; don't over-correct for it.

> **You collect data once.** Phase 2 (spatial conditioning) uses this same dataset.
> Spatial features are added offline after collection — no re-recording.

### Recording command

```bash
conda activate lerobot

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/policy2_core \
    --dataset.single_task="Insert the block into the slot" \
    --dataset.num_episodes=100 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=30 \
    --dataset.record_goal_image=true \
    --dataset.goal_image_camera_key=wrist \
    --dataset.push_to_hub=false \
    --display_data=true
```

### Per-episode procedure (Phase 1)

**Step 1 — Reset to start position**

```bash
python scripts/go_to_start_position.py --name insert_above_slot
```

Place block in gripper. Apply deliberate variation — shift laterally by up to ±0.5 cm or rotate
up to ±5° before closing the gripper. Vary across episodes so the full envelope is covered.

**Step 2 — Record the insertion**

Press **→** when the terminal prompts for the next episode.

Perform the insertion from hover pose:

- Align block to slot.
- Descend until fully seated.
- Total time should be 3–8 seconds. Press **→** as soon as seated.

Episode time limit is 20 seconds.

**Step 3 — Capture the goal image**

After **→**, the recorder pauses and prints:

```
[Goal Image] Episode recorded.
  1. Ensure block is fully seated in the slot.
  2. Open gripper.
  3. Retract elbow — arm must be clear of the wrist camera view.
  4. Wait for arm to fully stop (no motion blur).
  5. Press G to capture.
```

Follow these exactly:

1. Confirm block is flush with slot surface — no rocking, no tilt.
2. Open gripper fully.
3. Retract elbow slightly upward and back so the arm is not visible in the wrist camera frame.
4. **Wait 2–3 seconds.** Do not press G until the arm has completely stopped. Motion blur destroys
   the goal image and will degrade training.
5. Press **G**.

The terminal confirms: `[Goal Image] Saved → .../goal_images/episode_XXXXXX.png`

**What makes a good goal image:**

- Block fully seated, flush with slot surface.
- No arm or gripper visible — only slot and inserted block.
- Same lighting as the episode (do not move lamps between episodes).
- No motion blur.

**Step 4 — Re-record if needed**

If the insertion failed (missed, slipped, erratic motion): press **←** immediately after **→**.
This discards both the episode and the goal image capture — start fresh.

If the goal image was bad (blur, arm in frame): press **←** after pressing G. The episode
and goal image are both discarded.

**Step 5 — Reset environment**

During the 30-second reset window:

- Remove block from slot.
- Return arm to hover pose (`go_to_start_position.py` in a separate terminal if needed).
- Re-place block in gripper with the next planned variation.

### Session targets

- Aim for 20+ episodes per session.
- Check wrist camera alignment at the start of each session — camera mount can drift between sessions.
- **Target: 100 total.** Quality over speed. One bad goal image is worse than one fewer episode.

### After Phase 1 collection — verify goal images

```bash
python -c "
import glob
from pathlib import Path
imgs = sorted(Path.home().glob('.cache/huggingface/lerobot/rgragulraj/policy2_core/goal_images/*.png'))
print(f'{len(imgs)} goal images found (expected 100)')
"
```

Open a sample to verify visually:

```bash
eog ~/.cache/huggingface/lerobot/rgragulraj/policy2_core/goal_images/episode_000000.png
```

Every episode must have a corresponding goal image. If any are missing, append to the dataset
using `--resume=true` and re-collect those specific episodes.

### After verifying — register goal images as a dataset feature

Goal images are saved as `.png` files during recording but are not yet part of the LeRobot dataset
schema. Run `add_goal_image_feature.py` to convert them into a proper `observation.images.goal`
video feature so the standard dataset loader picks them up at training time.

```bash
# Dry run first — checks all goal images exist:
python scripts/add_goal_image_feature.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_core \
    --dry_run

# Full run:
python scripts/add_goal_image_feature.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_core
```

Verify:

```bash
python -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    'rgragulraj/policy2_core',
    root=Path.home() / '.cache/huggingface/lerobot/rgragulraj/policy2_core',
)
assert 'observation.images.goal' in ds.meta.video_keys
item = ds[0]
print('goal image shape:', item['observation.images.goal'].shape)
print('OK')
"
```

Run this once per dataset. Re-run Phase 3 and Phase 4 batches the same way before merging.

---

## Phase 2 — Spatial conditioning (offline, no re-recording)

Run this after Phase 1 collection is complete, before Phase 2 training.

### Step 1 — Calibrate the wrist-camera detector

The detector uses lighting-independent shape matching (CLAHE + Canny + Hu moment invariants).
No HSV sliders. You capture the shape contour once and it works under any lighting.

With the arm at canonical hover pose and the block/slot clearly visible in the wrist frame:

```bash
conda activate lerobot

# Slot shape template:
python scripts/detect_block_slot.py --calibrate --target=slot --camera_index=7

# Block face shape template:
python scripts/detect_block_slot.py --calibrate --target=block --camera_index=7
```

Point the camera at each target so it is clearly in frame. Press **C** to capture the largest
detected contour as the template. Confirm the green overlay tightly outlines the shape.
Press **S** to save. Calibration is written to `scripts/wrist_calibration.json`.

### Step 2 — Verify detection quality

```bash
python scripts/detect_block_slot.py --verify --camera_index=7
```

Hover the arm at canonical pose with block in gripper and check:

- Green box correctly tracks the block face.
- Red box correctly tracks the slot opening.
- Angles are stable frame-to-frame (not flipping randomly).

Re-calibrate if detection is noisy. >10% failure rate on either object degrades Phase 2.

### Step 3 — Run the offline detector

```bash
# Dry run first — detection stats without writing:
python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_core \
    --wrist_camera_key wrist \
    --dry_run

# Full run when satisfied:
python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_core \
    --wrist_camera_key wrist
```

This writes `observation.environment_state` (10-float spatial token) into every parquet frame
and updates `info.json` and `stats.json`.

### Step 4 — Verify the new feature

```bash
python -c "
import pandas as pd, glob
files = sorted(glob.glob('$HOME/.cache/huggingface/lerobot/rgragulraj/policy2_core/data/**/*.parquet', recursive=True))
df = pd.read_parquet(files[0])
print('Columns:', [c for c in df.columns if 'observation' in c])
print(df['observation.environment_state'].head())
"
```

---

## Phase 3 — Shape diversity (multi-shape datasets)

**Purpose:** Train the policy on 6–8 structurally diverse shape/slot pairs so it learns the
general principle of insertion rather than memorising specific geometries. Phase 3 datasets
are collected after Phase 2 and merged before retraining.

> **You collect data once per shape.** Spatial features are added offline after each shape's
> collection using `add_spatial_features.py --shape=<name>`, then all datasets are merged.

### Shape targets

Aim for at least 4 structurally distinct shapes (up to 8 if fixtures are available):

| Shape                                  | Dataset name             | Fixture needed     |
| -------------------------------------- | ------------------------ | ------------------ |
| Square (canonical — already collected) | `policy2_core`           | Already have it    |
| Round (cylinder)                       | `policy2_shape_round`    | Drilled hole       |
| Asymmetric/keyed (D-shape, T-slot)     | `policy2_shape_dshape`   | Fabricate or print |
| Triangular                             | `policy2_shape_triangle` | Fabricate or print |
| Cross/star                             | `policy2_shape_cross`    | Fabricate or print |
| Tall narrow rectangle                  | `policy2_shape_narrow`   | Fabricate or print |

Structural diversity matters — do not collect 20 variants of a square. Each new shape should
be geometrically distinct in insertion terms.

### Per-shape workflow

Repeat the following for each new shape.

#### Step A — Calibrate detector for this shape

```bash
conda activate lerobot

# Slot shape template (stored under data["<shape>"]["slot"]):
python scripts/detect_block_slot.py --calibrate --target=slot --shape=round --camera_index=7

# Block face shape template:
python scripts/detect_block_slot.py --calibrate --target=block --shape=round --camera_index=7

# Verify detection before recording:
python scripts/detect_block_slot.py --verify --shape=round --camera_index=7
```

Replace `round` with the actual shape name. Templates are stored alongside existing
calibration in `scripts/wrist_calibration.json` — no data is overwritten.

#### Step B — Record the shape dataset

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/policy2_shape_round \
    --dataset.single_task="Insert the round block into the slot" \
    --dataset.num_episodes=25 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=30 \
    --dataset.record_goal_image=true \
    --dataset.goal_image_camera_key=wrist \
    --dataset.push_to_hub=false \
    --display_data=true
```

**Per-episode procedure:** Same as Phase 1 — canonical hover → insert → goal image (G) → reset.
Vary start within ±1.5 cm lateral and ±15° rotation. Capture a goal image for every episode.

#### Step C — Add spatial features for this shape

```bash
# Dry run first:
python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_shape_round \
    --wrist_camera_key wrist \
    --shape round \
    --dry_run

# Full run when satisfied with detection quality (>90% detection rate):
python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_shape_round \
    --wrist_camera_key wrist \
    --shape round
```

Repeat Steps A–C for each new shape before merging.

### Merging all shape datasets

After all per-shape datasets have spatial features added:

```bash
python scripts/merge_datasets.py \
    --datasets \
        rgragulraj/policy2_core \
        rgragulraj/policy2_shape_round \
        rgragulraj/policy2_shape_dshape \
        rgragulraj/policy2_shape_triangle \
    --roots \
        ~/.cache/huggingface/lerobot/rgragulraj/policy2_core \
        ~/.cache/huggingface/lerobot/rgragulraj/policy2_shape_round \
        ~/.cache/huggingface/lerobot/rgragulraj/policy2_shape_dshape \
        ~/.cache/huggingface/lerobot/rgragulraj/policy2_shape_triangle \
    --output_repo_id rgragulraj/policy2_diverse \
    --output_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_diverse
```

Verify the merged dataset:

```bash
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('rgragulraj/policy2_diverse',
                    root='~/.cache/huggingface/lerobot/rgragulraj/policy2_diverse')
print('Episodes:', ds.num_episodes)
print('Frames:  ', ds.num_frames)
print('Features:', list(ds.features.keys()))
"
```

Expected: ~150–200 episodes total across all shapes.

---

## Phase 4 — Slot depth variation

**Purpose:** Without depth variation in training, the policy learns a fixed-depth descent and
fails on slots shallower or deeper than the training slot. With goal image conditioning, the
policy _can_ learn to descend until the goal image state is reached — but only if it has seen
varying depths during training, so it learns that descent distance is determined by visual
feedback, not a fixed step count.

**Start conditions:** Same canonical hover pose as Phase 1. Vary laterally ±1.5 cm and ±15°
rotation as usual — this is depth-variation data, not position-variation data, so do not
tighten the start envelope.

> **No new detector calibration needed.** The spatial token is 2D geometry only.
> Depth is handled entirely by goal image conditioning.
> Use the same `wrist_calibration.json` already set up in Phase 2/3.

### Slot depths to collect

| Depth           | Dataset name       | Episodes | Fixture                                  |
| --------------- | ------------------ | -------- | ---------------------------------------- |
| 2 cm (shallow)  | `policy2_depth_20` | 8–10     | Shallow insert stop or thin block spacer |
| 4 cm (standard) | `policy2_depth_40` | 10–12    | Standard slot (same as Phase 1)          |
| 6 cm (deep)     | `policy2_depth_60` | 8–10     | Extended slot or tube                    |

Minimum viable: all 3 depths on the square (canonical shape). Cover 2 depths on each
additional shape if you can fabricate the fixtures; skip if not feasible.

> The goal image is what teaches the policy what "fully seated" looks like at each depth.
> A shallow slot produces a goal image where the block protrudes further; a deep slot
> produces one where it sits flush or recessed. The policy learns to match that image.

### Recording commands

Run once per depth. The only change between runs is `repo_id`, `single_task`, and `num_episodes`.

**2 cm (shallow):**

```bash
conda activate lerobot

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/policy2_depth_20 \
    --dataset.single_task="Insert the block into the slot (depth 20mm)" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=30 \
    --dataset.record_goal_image=true \
    --dataset.goal_image_camera_key=wrist \
    --dataset.push_to_hub=false \
    --display_data=true
```

**4 cm (standard):**

```bash
lerobot-record \
    ... \
    --dataset.repo_id=rgragulraj/policy2_depth_40 \
    --dataset.single_task="Insert the block into the slot (depth 40mm)" \
    --dataset.num_episodes=12 \
    ...
```

**6 cm (deep):**

```bash
lerobot-record \
    ... \
    --dataset.repo_id=rgragulraj/policy2_depth_60 \
    --dataset.single_task="Insert the block into the slot (depth 60mm)" \
    --dataset.num_episodes=10 \
    ...
```

### Per-episode procedure (Phase 4)

**Step 1 — Install the correct depth fixture**

Swap to the target depth slot before starting that depth's recording session. Do not mix
depths within a single dataset — each `policy2_depth_XX` must contain only episodes at that depth.

Return arm to canonical hover pose:

```bash
python scripts/go_to_start_position.py --name insert_above_slot
```

**Step 2 — Place block and apply start variation**

Place block in gripper. Apply the same deliberate variation as Phase 1:

- Lateral offset: up to ±1.5 cm in x/y
- Rotation: up to ±15° from canonical alignment

**Step 3 — Record the insertion**

Press **→** when the terminal prompts for the next episode.

Perform the insertion:

- Align block to slot opening.
- Descend until the block is **fully seated** at this depth — do not stop early.
- For a 2 cm slot this is a short descent. For 6 cm it is a longer descent. Let the block
  tell you when it is seated, not a fixed arm position.
- Press **→** immediately when seated.

Episode time limit is 20 seconds. If the insertion takes longer, let the timer expire — that
episode still counts as a failed demonstration. Press **←** to discard if you do not want it.

**Step 4 — Capture the goal image**

This step is identical to Phase 1, but the seating depth is different for each dataset.
The goal image must show the block at the correct seated depth for this fixture — this is
the signal that teaches the policy what "done" means for this depth.

After **→**, the recorder pauses. Follow these steps exactly:

1. Confirm the block is fully seated at this depth — no partial insertion.
2. Open gripper fully.
3. Retract elbow upward and back — arm must not appear in the wrist camera frame.
4. **Wait 2–3 seconds** until the arm has completely stopped moving.
5. Press **G** to capture.

Terminal confirms: `[Goal Image] Saved → .../goal_images/episode_XXXXXX.png`

**What to check:**

- Block seated at the correct depth (not halfway in).
- No arm or gripper in frame.
- No motion blur.
- Same lighting as during the episode.

**Step 5 — Re-record if needed**

Press **←** to discard and re-record if:

- The insertion was partial — block did not fully seat.
- The goal image had motion blur or the arm was in frame.
- The descent trajectory was erratic.

**Step 6 — Reset environment**

During the 30-second reset window:

- Remove block from slot.
- Return arm to canonical hover pose.
- Re-place block in gripper with next planned start variation.
- Do not swap depth fixtures during a recording session.

### Session targets

- Complete one full depth dataset (8–12 episodes) in a single session where possible.
- Do not mix depths within a session — swap fixtures between sessions, not mid-session.
- **Check the goal images** at the end of each session before moving on.

### After collection — verify goal images per depth dataset

```bash
# Replace policy2_depth_20 with the dataset you just collected:
python -c "
import glob
from pathlib import Path
imgs = sorted(Path.home().glob('.cache/huggingface/lerobot/rgragulraj/policy2_depth_20/goal_images/*.png'))
print(f'{len(imgs)} goal images found')
"

eog ~/.cache/huggingface/lerobot/rgragulraj/policy2_depth_20/goal_images/episode_000000.png
```

Confirm the goal image shows the block fully seated at 20 mm depth — the seating depth should
be visually distinct from the 40 mm and 60 mm goal images. If the images look identical across
depths, the fixtures are not producing distinct enough visual states.

### Add spatial features per depth dataset

```bash
conda activate lerobot

# No --shape flag needed if using canonical square block:
python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_depth_20 \
    --wrist_camera_key wrist

python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_depth_40 \
    --wrist_camera_key wrist

python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_depth_60 \
    --wrist_camera_key wrist
```

If collecting depth variations for non-square shapes, add `--shape=<name>` to each command.

### Merge depth datasets into `policy2_diverse`

Re-run the merge to include the depth datasets alongside all shape datasets. This overwrites
the previous `policy2_diverse`:

```bash
python scripts/merge_datasets.py \
    --datasets \
        rgragulraj/policy2_core \
        rgragulraj/policy2_shape_round \
        rgragulraj/policy2_shape_dshape \
        rgragulraj/policy2_shape_triangle \
        rgragulraj/policy2_depth_20 \
        rgragulraj/policy2_depth_40 \
        rgragulraj/policy2_depth_60 \
    --roots \
        ~/.cache/huggingface/lerobot/rgragulraj/policy2_core \
        ~/.cache/huggingface/lerobot/rgragulraj/policy2_shape_round \
        ~/.cache/huggingface/lerobot/rgragulraj/policy2_shape_dshape \
        ~/.cache/huggingface/lerobot/rgragulraj/policy2_shape_triangle \
        ~/.cache/huggingface/lerobot/rgragulraj/policy2_depth_20 \
        ~/.cache/huggingface/lerobot/rgragulraj/policy2_depth_40 \
        ~/.cache/huggingface/lerobot/rgragulraj/policy2_depth_60 \
    --output_repo_id rgragulraj/policy2_diverse \
    --output_root ~/.cache/huggingface/lerobot/rgragulraj/policy2_diverse
```

Verify the merged dataset episode count before training:

```bash
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('rgragulraj/policy2_diverse',
                    root='~/.cache/huggingface/lerobot/rgragulraj/policy2_diverse')
print('Episodes:', ds.num_episodes)
print('Frames:  ', ds.num_frames)
"
```

---

## Keyboard shortcuts

| Key             | Action                                                     |
| --------------- | ---------------------------------------------------------- |
| → (right arrow) | End current episode and proceed                            |
| ← (left arrow)  | Discard episode and re-record                              |
| G               | Capture goal image (active only during goal capture phase) |
| Esc             | Stop recording session                                     |

---

## Training

### Phase 0 — vanilla baseline (before Phase 1 collection)

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy2_baseline \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy2_baseline \
  --policy.push_to_hub=false \
  --training.num_train_steps=50000 \
  --output_dir=outputs/policy2_baseline
```

### Phase 1 — goal image conditioning

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy2_core \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy2_core \
  --policy.chunk_size=20 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.1 \
  --policy.kl_weight=20.0 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_lr_backbone=5e-6 \
  --policy.use_goal_image=true \
  --policy.use_shared_goal_backbone=true \
  --policy.push_to_hub=false \
  --training.num_train_steps=50000 \
  --output_dir=outputs/policy2_phase1
```

Validation targets (see Policy2.md §5.5):

- Tier 1 (±0 cm, ±0°): >90% success
- Tier 2 (±1 cm lateral): >80%
- Tier 3 (±1.5 cm, ±10° rotation): >60%

### Phase 2 — goal image + spatial conditioning

After running `add_spatial_features.py`:

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy2_core \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy2_core \
  --policy.chunk_size=20 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.1 \
  --policy.kl_weight=20.0 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_lr_backbone=5e-6 \
  --policy.use_goal_image=true \
  --policy.use_shared_goal_backbone=true \
  --policy.use_spatial_conditioning=true \
  --policy.spatial_conditioning_dim=10 \
  --policy.push_to_hub=false \
  --training.num_train_steps=50000 \
  --output_dir=outputs/policy2_phase2
```

Validation: rerun Tier 1/2/3. Also test with a ±12° rotated start — the policy should visibly
correct the rotation before descending. If it descends straight into the slot edge, the angular
signal is not being used (see Policy2.md §6 for debugging steps).

### Phase 3 — shape diversity

After merging all shape datasets into `policy2_diverse`:

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy2_diverse \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy2_diverse \
  --policy.chunk_size=20 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.1 \
  --policy.kl_weight=20.0 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_lr_backbone=5e-6 \
  --policy.use_goal_image=true \
  --policy.use_shared_goal_backbone=true \
  --policy.use_spatial_conditioning=true \
  --policy.spatial_conditioning_dim=10 \
  --policy.push_to_hub=false \
  --training.num_train_steps=80000 \
  --output_dir=outputs/policy2_phase3
```

Validation: test on a novel shape not in training. Target >50% Tier 1 success on a novel shape.
If novel shape success is below 30%, increase shape diversity (add more structurally distinct shapes).
See Policy2.md §10.2 for the full novel-shape evaluation protocol.

### Phase 4 — slot depth variation

After merging depth datasets into `policy2_diverse` (re-run merge, same config, higher steps):

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy2_diverse \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy2_diverse \
  --policy.chunk_size=20 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.1 \
  --policy.kl_weight=20.0 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_lr_backbone=5e-6 \
  --policy.use_goal_image=true \
  --policy.use_shared_goal_backbone=true \
  --policy.use_spatial_conditioning=true \
  --policy.spatial_conditioning_dim=10 \
  --policy.push_to_hub=false \
  --training.num_train_steps=80000 \
  --output_dir=outputs/policy2_phase4
```

Validation: test on a slot depth not seen in training (e.g. 5 cm if trained on 2/4/6 cm).
The policy should descend until the block is seated, not stop at the nearest trained depth.
See Policy2.md §10.3 for the full depth generalisation evaluation protocol.

---

## Inference

Policy 2 requires both a goal image and (Phase 2+) spatial conditioning at inference time.
Both are provided via processor steps in the pre-processing pipeline.

### Capture the goal image for a novel slot

1. Bring the arm to the canonical hover pose above the slot.
2. Place the block in the gripper, correctly aligned.
3. Wait for the arm to stabilise (2–3 seconds — no motion blur).
4. Capture the wrist-camera frame:

   ```bash
   python -c "
   import cv2; cap = cv2.VideoCapture(7)
   ret, frame = cap.read(); cap.release()
   cv2.imwrite('goal_novel_slot.png', frame)
   print('Saved goal_novel_slot.png')
   "
   ```

### Wire up the processors in your inference script

```python
from lerobot.policies.act.processor_act import GoalImageProcessorStep, SpatialConditioningProcessorStep

# Create once at script startup:
goal_step = GoalImageProcessorStep(
    goal_image_path="goal_novel_slot.png",
    target_size=(480, 640),  # must match training resolution
)
spatial_step = SpatialConditioningProcessorStep(
    camera_key="observation.images.wrist",
    detector_type="shape_wrist",
    calibration_path="scripts/wrist_calibration.json",
    ema_alpha=0.5,
)

# Both steps go before AddBatchDimensionProcessorStep in the pre-processing pipeline.
# Call reset_goal() at the start of each episode when the target slot changes:
goal_step.reset_goal("goal_novel_slot_v2.png")
```
