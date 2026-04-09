# Policy 1 — Data Collection Instructions

**Author:** LENS Lab
**Hardware:** SO-101 follower + leader arm, top-down webcam (index 5), gripper camera (index 7)

---

## Overview

Policy 1 is built across four phases. The core challenge is **data diversity** — the policy must generalise to block positions, slot positions, objects, and lighting conditions it has never seen during training. Each phase adds a new dimension of coverage.

| Phase    | Dataset                                                         | Episodes | Goal images | Purpose                                                          |
| -------- | --------------------------------------------------------------- | -------- | ----------- | ---------------------------------------------------------------- |
| Phase 0  | `rgragulraj/policy1_baseline`                                   | 50       | No          | Vanilla ACT baseline — measure what fails before adding anything |
| Phase 1a | Same as Phase 0 (retrain only)                                  | —        | —           | Aggressive augmentation — code change only, no new data needed   |
| Phase 1b | `rgragulraj/policy1_diverse_<batch_label>` → `policy1_merged`   | 300–500  | No          | Systematic diversity — grid, objects, slot positions, lighting   |
| Phase 1c | Same as Phase 1b (retrain only)                                 | —        | —           | Selective backbone unfreeze — only if >500 episodes              |
| Phase 2  | Same as Phase 1b (add features offline)                         | —        | —           | Spatial conditioning — no re-recording needed                    |
| Phase 3  | `rgragulraj/policy1_goal_<batch_label>` → `policy1_goal_merged` | 200+     | Yes         | Goal image conditioning — orientation awareness                  |

**Collection order:** Phase 0 → Phase 1a (retrain) → Phase 1b → Phase 1c (optional) → Phase 2 (offline) → Phase 3.

Phase 2 does not require re-recording. Spatial features are extracted offline from the Phase 1b videos.

---

## Hardware setup (do this before any phase)

### 1. Verify camera indices

```bash
conda activate lerobot
python -c "import cv2; [print(f'index {i}: OK' if cv2.VideoCapture(i).read()[0] else f'index {i}: no camera') for i in range(10)]"
```

Top-down webcam should be index 5, gripper camera index 7. If different, update `--robot.cameras` in all recording commands below.

### 2. Define and save the canonical hover pose

The canonical hover pose is the fixed arm configuration that is the **target end state** for every Policy 1 episode:

- Gripper directly above slot centre, ±0 mm in x/y
- Gripper height: ~4 cm above the top of the slot opening
- Block orientation: aligned with the slot's insertion axis

If you haven't saved this pose yet, teleoperate the arm to the hover position manually, then:

```bash
python scripts/capture_start_position.py --name insert_above_slot
```

To move the arm back to it between episodes:

```bash
python scripts/go_to_start_position.py --name insert_above_slot
```

### 3. Mark the 3×3 workspace grid

Policy 1b requires collecting from a **3×3 grid** of block starting positions. Mark cell boundaries with tape before starting:

```
+-------+-------+-------+
|  TL   |  TC   |  TR   |
|       |       |       |
+-------+-------+-------+
|  ML   |  MC   |  MR   |
|       |       |       |
+-------+-------+-------+
|  BL   |  BC   |  BR   |
|       |       |       |
+-------+-------+-------+
(T=top, M=mid, B=bottom, L=left, C=centre, R=right from robot's perspective)
```

Cell size: ~15–20 cm per side. The grid should span the full reachable workspace.

### 4. Verify the top-down camera view

Confirm that at hover pose the top-down camera shows:

- The entire 3×3 workspace grid in frame.
- The slot is clearly visible.
- The block in the gripper is visible from above.

Do not move the camera mount between sessions. Consistent framing is required for the shape detector (Phase 2) to work reliably.

---

## Phase 0 — Baseline

**Purpose:** Measure the raw precision of vanilla ACT before any architectural changes. This tells you exactly which failure mode is the bottleneck (wrong area, grasp slip, or wrong orientation at hover). Do not skip — the baseline is required to make the right Phase 1b collection decisions.

**Start conditions:** Always start from the **exact canonical hover pose** for the block start, not the block placement. Place the block at a **single fixed location** (centre cell `MC`). Use a **single fixed slot position**. No variation.

### Recording command

```bash
conda activate lerobot

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/policy1_baseline \
    --dataset.single_task="Pick up the block and hover above the slot" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=false \
    --display_data=true
```

### Per-episode procedure (Phase 0)

**Step 1 — Reset arm to home**

The arm starts from home position for every episode. Use teleoperation or a home position script:

```bash
python scripts/go_to_start_position.py --name home
```

**Step 2 — Place the block at the fixed position**

Place the block at the centre cell (`MC`) in the same orientation and same spot every time. Phase 0 tests the insertion approach task, not position generalisation.

**Step 3 — Record the pick-and-hover**

When the terminal prompts for the next episode, press **→** to begin recording.

Perform the pick-and-hover:

- Pick up the block.
- Transport it to the canonical hover pose: gripper directly above the slot, ~4 cm height, block aligned with the insertion axis.
- Press **→** as soon as the hover pose is reached cleanly.

Episode time limit is 30 seconds. A well-executed demo should complete in 8–15 seconds.

**Step 4 — Reset environment**

During the 15-second reset window:

- Return arm to home.
- Return block to the fixed starting position.

### After Phase 0 collection

Train vanilla ACT:

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_baseline \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_baseline \
  --policy.push_to_hub=false \
  --training.num_train_steps=80000 \
  --output_dir=outputs/policy1_baseline
```

Evaluate using the protocol in Policy1.md §4.4. Classify every failure into:

- **(a) Wrong approach** — arm goes to wrong area → location generalisation failure → prioritise Phase 2
- **(b) Grasp slip** — correct area, wrong grip → prioritise more object diversity in Phase 1b
- **(c) Wrong orientation at hover** — correct pick, wrong block orientation above slot → prioritise Phase 3

**Proceed to Phase 1a regardless of results.**

---

## Phase 1a — Augmentation (no new data)

Phase 1a is a **code change and retrain only**. No new data collection.

Run the Phase 0 training command with augmentation flags added:

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_baseline \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_baseline \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.brightness.min_max="[0.5, 1.5]" \
  --dataset.image_transforms.contrast.min_max="[0.5, 2.0]" \
  --dataset.image_transforms.hue.min_max="[-0.1, 0.1]" \
  --policy.push_to_hub=false \
  --training.num_train_steps=80000 \
  --output_dir=outputs/policy1_phase1a
```

Rerun Phase 0 evaluation. If location ±10 cm hover success improves by 15%+ over the Phase 0 baseline, augmentation is contributing. Then proceed to Phase 1b.

---

## Phase 1b — Systematic Data Diversity

**Purpose:** This is the main Policy 1 data collection effort. Cover the workspace, object space, slot position space, and lighting conditions systematically. The policy cannot generalise to what it has not seen.

**Target: 300–500 episodes.** Aim for at least 3 dedicated collection sessions.

**Start conditions:** Vary the block start position across the 3×3 grid — no two consecutive episodes should start from the same cell. Slot position varies across sessions (see §1b.3 below). Arm always starts from home.

> **You can collect Phase 1b data before training Phase 1a — the datasets are compatible.** Train Phase 1a and Phase 1b together once you have enough episodes.

### 1b.1 Workspace grid coverage

Collect **15–20 episodes per cell** per object. Vary the exact block position **within** each cell — do not always place it at the cell centre. This ensures the policy sees the full continuous distribution of positions, not just 9 discrete points.

| Cell          | Label | Target episodes (primary object)      |
| ------------- | ----- | ------------------------------------- |
| Top-left      | TL    | 15–20                                 |
| Top-centre    | TC    | 15–20                                 |
| Top-right     | TR    | 15–20                                 |
| Mid-left      | ML    | 15–20                                 |
| Mid-centre    | MC    | 15–20 (already have ~50 from Phase 0) |
| Mid-right     | MR    | 15–20                                 |
| Bottom-left   | BL    | 15–20                                 |
| Bottom-centre | BC    | 15–20                                 |
| Bottom-right  | BR    | 15–20                                 |

### 1b.2 Object diversity

Collect across **at least 5 objects**. For each new object, cover at least 5 of the 9 grid cells:

| Object                                  | Target episodes          | Notes                             |
| --------------------------------------- | ------------------------ | --------------------------------- |
| Original square block                   | 60 (from Phase 0 + grid) | Training baseline                 |
| Object 2: different colour, same shape  | 40                       | Tests colour generalisation       |
| Object 3: similar shape, different size | 40                       | Tests scale generalisation        |
| Object 4: different shape (e.g. round)  | 40                       | Tests shape generalisation        |
| Object 5: visually distinct (wildcard)  | 40                       | Tests broad visual generalisation |

### 1b.3 Slot position diversity

Move the slot to **at least 5 different positions** across sessions. For each slot position, collect 40 episodes spanning multiple grid cells:

| Position          | Setup                            | Episodes |
| ----------------- | -------------------------------- | -------- |
| Centre (standard) | Default position                 | 40       |
| Left              | Shift slot 10 cm left            | 40       |
| Right             | Shift slot 10 cm right           | 40       |
| Near robot        | Shift slot 10 cm toward robot    | 40       |
| Away from robot   | Shift slot 10 cm away from robot | 40       |

### 1b.4 Lighting variation

Vary lighting **between sessions**, not within sessions. Each session must have consistent lighting:

| Condition     | Setup                                   | Target episodes |
| ------------- | --------------------------------------- | --------------- |
| Overhead on   | Lab overhead lighting, no other sources | 80–100          |
| Side lamp     | Overhead off, desk lamp from one side   | 80–100          |
| Natural light | Window light, overhead off              | 80–100          |

Mix objects and slot positions within each lighting session.

### 1b.5 Camera tilt variation

Between sessions, vary the top-down webcam tilt by ±5 degrees (physically adjust the mount). Keep the gripper camera mount fixed. This prevents the policy from depending on a single camera angle.

### Recording command

Use separate `repo_id` labels per collection batch so you can track what was collected. Merge all batches after collection (see below).

```bash
conda activate lerobot

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/policy1_diverse_<batch_label> \
    --dataset.single_task="Pick up the block and hover above the slot" \
    --dataset.num_episodes=<number> \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=false \
    --display_data=true
```

Replace `<batch_label>` with a descriptive name, e.g.:

- `policy1_diverse_grid_obj1` — square block, full grid
- `policy1_diverse_obj2` — second object, 5 cells
- `policy1_diverse_slot_right` — slot shifted right, mixed objects
- `policy1_diverse_lighting_side` — side lamp session

### Per-episode procedure (Phase 1b)

**Step 1 — Set up this episode's conditions**

Before pressing → to start, confirm:

- Block object for this batch
- Target cell for this episode
- Slot position for this session

**Step 2 — Start from home**

Return arm to home position. Place the block in the target cell with a natural variation — do not always use the exact cell centre or the same orientation.

**Step 3 — Record the pick-and-hover**

When the terminal prompts for the next episode, press **→** to begin.

Perform the pick-and-hover:

- Pick up the block from the target cell.
- Transport to the canonical hover pose: gripper directly above the slot, ~4 cm height, block aligned with insertion axis.
- Press **→** as soon as the hover pose is cleanly reached.

Episode time limit is 30 seconds.

**Step 4 — Reset environment**

During the 15-second reset window:

- Return arm to home.
- Place block in the next planned cell (keep a written log of cells per session to ensure coverage).

### After Phase 1b — merge and train

After each collection batch, merge all datasets into `policy1_merged`:

```python
from lerobot.datasets.aggregate import aggregate_datasets

aggregate_datasets(
    repo_ids=[
        "rgragulraj/policy1_baseline",
        "rgragulraj/policy1_diverse_grid_obj1",
        "rgragulraj/policy1_diverse_obj2",
        # ... add all batch names
    ],
    output_repo_id="rgragulraj/policy1_merged",
    local_dir="~/.cache/huggingface/lerobot/rgragulraj/policy1_merged",
)
```

**Always recompute normalisation stats after every merge:**

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import compute_episode_stats, aggregate_stats

dataset = LeRobotDataset(
    "rgragulraj/policy1_merged",
    root="~/.cache/huggingface/lerobot/rgragulraj/policy1_merged",
)
compute_episode_stats(dataset)
aggregate_stats(dataset)
```

Stale normalisation stats are a common silent failure mode — always rerun after merging.

---

## Phase 1c — Selective Backbone Unfreeze (optional, code change only)

**Only attempt this if you have >500 Policy 1 episodes.**

No new data collection. Retrain on `policy1_merged` with the `--policy.unfreeze_backbone_layers` flag:

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_merged \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_merged \
  --dataset.image_transforms.enable=true \
  --policy.unfreeze_backbone_layers='["layer4"]' \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_lr_backbone=1e-6 \
  --policy.push_to_hub=false \
  --training.num_train_steps=100000 \
  --output_dir=outputs/policy1_phase1c
```

With fewer than 500 episodes, unfrozen batch norm statistics tend to overfit to specific objects and lighting. Skip this phase if you don't meet the threshold.

---

## Phase 2 — Spatial Conditioning (offline, no re-recording)

Run this after Phase 1b collection is complete, before Phase 2 training.

Phase 2 adds a **10-float spatial token** (`observation.environment_state`) to every frame in the existing dataset:

```
[cx_block, cy_block, w_block, h_block, angle_block,
 cx_slot,  cy_slot,  w_slot,  h_slot,  angle_slot]
```

Values are computed from the **top-down camera** using **shape template matching** (CLAHE + Canny + Hu moment invariants — the same pipeline as Policy 2's wrist detector). No new episodes needed, and no HSV colour sliders.

**Why shape matching instead of colour (HSV):**

- Lighting-independent — no re-calibration needed between sessions (you varied lighting in Phase 1b)
- Works with multiple slots in the workspace — the `--shape` flag selects which slot to track by geometry, not colour
- Directly feeds into Phase 4 language conditioning: "hover above the square slot" maps to `--shape=square`, the detector returns that slot's position, no architecture change needed

Token format is 10 floats — identical to Policy 2. `spatial_conditioning_dim=10` for both.

Calibration saved to `scripts/top_shape_calibration.json`.

---

### Step 1 — Calibrate the top-down shape detector

Run once for the block face, then once per slot shape. The calibration captures the geometric contour of each object — no colour sliders.

```bash
conda activate lerobot

# Block face (viewed from above at hover height):
python scripts/detect_block_slot.py --calibrate --target=block --camera top

# Slot (single shape, flat format):
python scripts/detect_block_slot.py --calibrate --target=slot --camera top

# OR — if you have multiple slot shapes in the workspace:
python scripts/detect_block_slot.py --calibrate --target=slot --shape=square --camera top
python scripts/detect_block_slot.py --calibrate --target=slot --shape=round  --camera top
```

**What to do during calibration:**

1. Bring the arm to the canonical hover pose. Place the block in the gripper. The top-down camera should show the block face clearly from above.
2. A live window opens showing all detected contours in blue.
3. The status line at the bottom shows `Contours: N  min_area=500`. If N is very large (>20) with the block not yet in frame, the `min_area` is too low — see tuning below.
4. Press **C** to capture the largest detected contour as the template. A green bounding box appears over it.
5. Confirm the green box tightly outlines the block face (or slot opening). If it is capturing background clutter instead, re-calibrate with a higher `--min_area` (see tuning).
6. Press **S** to save.

Repeat for the slot: remove the block from the gripper, bring the arm to the side so it is not in frame, then calibrate.

---

### Step 2 — Verify detection quality

```bash
# Single-shape setup:
python scripts/detect_block_slot.py --verify --camera top

# Multi-shape — verify each shape profile:
python scripts/detect_block_slot.py --verify --camera top --shape=square
python scripts/detect_block_slot.py --verify --camera top --shape=round
```

With the arm at hover pose and block in gripper, check:

- **Green box** tracks the block face correctly.
- **Red box** tracks the target slot correctly.
- Boxes are **stable frame-to-frame** — not jumping or swapping to background objects.
- Match scores shown in the overlay are **below `match_threshold`** (default 0.30 for top camera).
- The bottom-left token readout shows **non-zero values** for both block and slot.

Move the arm across the full workspace (all 9 cells from Phase 1b) during verification. The detector must be robust to scale changes as the arm moves to different distances from the camera.

**Minimum acceptable quality before proceeding:**

- > 90% detection rate for both block and slot across the workspace
- Bounding boxes stable (not flickering) when the arm is stationary
- No false matches on background objects when block and slot are not in the frame

---

### Step 3 — Tuning `match_threshold` and `min_area`

These are the only two parameters you need to adjust. The calibration UI shows both on screen during `--verify`.

**`match_threshold` (default: 0.30)**

Controls how similar a contour must be to the template. Lower = stricter, higher = looser.

| Symptom                                                         | Likely cause                                    | Fix                                                                           |
| --------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------- |
| Correct object visible but not detected (red/green box missing) | Threshold too strict                            | Raise by 0.05: `--match_threshold 0.35`                                       |
| Block detection works near camera but fails at far end of grid  | Perspective/scale change too large for template | Raise to 0.40; or re-calibrate from the mid-distance position                 |
| Wrong object detected (background edge highlighted)             | Threshold too loose                             | Lower by 0.05: `--match_threshold 0.25`                                       |
| Block and slot detections swap or overlap                       | Both templates match the same contour           | Lower threshold; or re-calibrate one template with the object better isolated |

To verify a threshold override without re-running offline processing:

```bash
python scripts/detect_block_slot.py --verify --camera top --match_threshold 0.35
```

Once the right value is found, use it in all subsequent `add_spatial_features.py` calls via `--match_threshold`.

**`min_area` (default: 500 px)**

Minimum contour area in pixels. Anything smaller is ignored. Filters out shadows, cable edges, and table surface texture.

| Symptom                                                       | Likely cause                                                       | Fix                                                                                              |
| ------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Live view shows many small spurious blue contours everywhere  | min_area too low                                                   | Note the area printed when you press C on a spurious contour. Set `--min_area` to ~2× that value |
| Object not detected even when clearly in frame                | min_area too high                                                  | Lower by 100 px increments                                                                       |
| Works at close range (arm near camera) but fails at far range | Object appears smaller at far end of grid; min_area filters it out | Lower to 300 and re-verify                                                                       |

To test a min_area override during verification:

```bash
python scripts/detect_block_slot.py --verify --camera top --min_area 350
```

**Typical starting values for this setup:**

| Working distance                | `min_area` starting point |
| ------------------------------- | ------------------------- |
| ~40 cm (arm close to camera)    | 600–800 px                |
| ~60 cm (arm mid-distance)       | 400–600 px                |
| ~80 cm (arm at far end of grid) | 250–400 px                |

If the working distance varies significantly across the grid, set `min_area` to work at the farthest point — false positives at close range are less damaging than missed detections.

---

### Step 4 — Run the offline detector

```bash
conda activate lerobot

# Dry run first — prints detection stats per episode without writing anything:
python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_merged \
    --camera_key top \
    --detector_type shape_top \
    --dry_run

# If using tuned parameters:
python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_merged \
    --camera_key top \
    --detector_type shape_top \
    --match_threshold 0.35 \
    --dry_run
```

The dry run prints a detection summary:

```
Detection summary (XXXX frames total):
  Block detection failures: XX (X.X%)
  Slot detection failures:  XX (X.X%)
```

**Target: <10% failure rate for both block and slot.** If either is above 10%:

1. Run `--verify` to visually identify what is failing.
2. Adjust `--match_threshold` or `--min_area` and re-dry-run.
3. Do not proceed to the full run until both are below 10%.

Full run once satisfied:

```bash
python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_merged \
    --camera_key top \
    --detector_type shape_top
```

For multi-slot workspaces — specify the target shape (must match a calibrated profile):

```bash
python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_merged \
    --camera_key top \
    --detector_type shape_top \
    --shape square
```

This writes `observation.environment_state` (10-float token) into every parquet frame and updates `info.json` and `stats.json`.

---

### Step 5 — Verify the written feature

```bash
python -c "
import pandas as pd, glob
files = sorted(glob.glob('$HOME/.cache/huggingface/lerobot/rgragulraj/policy1_merged/data/**/*.parquet', recursive=True))
df = pd.read_parquet(files[0])
print('Columns:', [c for c in df.columns if 'observation' in c])
print(df['observation.environment_state'].head())
"
```

Confirm `observation.environment_state` is present and the values are non-zero for the majority of rows.

---

### Common failure modes

| Symptom                                       | Likely cause                                                                 | Fix                                                                                     |
| --------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| >10% block failures, slot fine                | Block template captured at wrong distance or angle                           | Re-calibrate block at mid-distance position; try `--match_threshold 0.35`               |
| >10% slot failures, block fine                | Slot partially occluded by arm during training episodes                      | Expected for frames mid-trajectory; if slot fails even at episode start, re-calibrate   |
| Both >10%                                     | `min_area` filters out objects at far end of grid                            | Lower `--min_area` to 300–350                                                           |
| Token values erratic frame-to-frame in verify | Background contour matches template; threshold too loose                     | Lower `--match_threshold` to 0.25; or raise `--min_area` to filter the spurious contour |
| Works in verify but high failure in dry run   | Training videos have different lighting or camera angle than current session | Re-calibrate template under representative lighting; or use `--match_threshold 0.40`    |

---

## Phase 3 — Goal Image Conditioning

**Purpose:** Enable orientation awareness. When the arm arrives at hover pose with the wrong block orientation, the goal image (showing the correct orientation from above) gives the policy a visual target to correct toward.

**Start conditions:** Same as Phase 1b — vary block position, object, slot position, and lighting. Additionally vary block orientation within ±15° of the correct insertion axis on each episode start. This builds the orientation-correction capability.

**You collect a separate dataset.** Phase 3 does not replace Phase 1b data — both are merged into `policy1_goal_merged` for training.

> **Collect at least 200 episodes for goal image conditioning to be effective.** With fewer episodes the policy has not seen enough (current view, goal image) pairs to learn to close the gap.

### Recording command

```bash
conda activate lerobot

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/policy1_goal_<batch_label> \
    --dataset.single_task="Pick up the block and hover above the slot" \
    --dataset.num_episodes=<number> \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=30 \
    --dataset.record_goal_image=true \
    --dataset.goal_image_camera_key=top \
    --dataset.push_to_hub=false \
    --display_data=true
```

Reset time is 30 seconds (longer than Phase 1b) to allow time for the goal image capture step.

### Per-episode procedure (Phase 3)

**Step 1 — Return to home and place block**

Return arm to home. Place the block in the target cell. Apply deliberate variation:

- Lateral position: anywhere within the cell (±5–10 cm from cell centre).
- Rotation: ±15° from correct insertion axis. Vary across episodes.

**Step 2 — Record the pick-and-hover**

When the terminal prompts for the next episode, press **→** to begin.

Perform the pick-and-hover:

- Pick up the block.
- Orient the block to align correctly with the slot insertion axis.
- Arrive at the canonical hover pose: gripper directly above slot, ~4 cm height, block aligned.
- Press **→** as soon as the hover pose is cleanly reached.

Episode time limit is 30 seconds.

**Step 3 — Capture the goal image**

After **→**, the recorder pauses and prints:

```
[Goal Image] Episode recorded.
  1. Hold arm steady at the hover pose — block must be in gripper, ~4 cm above slot.
  2. Block must be correctly aligned with the slot insertion axis.
  3. Wait for arm to fully stop (no motion blur).
  4. Press G to capture.
```

Follow these exactly:

1. Confirm block is in gripper at the canonical hover pose — correctly aligned with the insertion axis.
2. **Wait 2–3 seconds.** Do not press G until the arm has completely stopped. Motion blur destroys the goal image.
3. Press **G**.

The terminal confirms: `[Goal Image] Saved → .../goal_images/episode_XXXXXX.png`

**What makes a good goal image:**

- Block in gripper, held at hover pose, correctly aligned with slot insertion axis.
- Slot clearly visible below the block.
- No arm motion blur — the top-down camera must be still.
- Same lighting as the episode.

**Step 4 — Re-record if needed**

Press **←** immediately after **→** to discard the episode before capturing the goal image.

Press **←** after pressing **G** to discard both the episode and the goal image if:

- The pick was a grasp slip — block not securely held.
- The hover pose was not cleanly reached.
- The goal image had motion blur or the block was not aligned.

**Step 5 — Reset environment**

During the 30-second reset window:

- Return arm to home.
- Remove block from gripper, place at next planned cell.

### After Phase 3 collection — verify goal images

```bash
python -c "
import glob
from pathlib import Path
imgs = sorted(Path.home().glob('.cache/huggingface/lerobot/rgragulraj/policy1_goal_<batch_label>/goal_images/*.png'))
print(f'{len(imgs)} goal images found')
"
```

Open a sample to verify:

```bash
eog ~/.cache/huggingface/lerobot/rgragulraj/policy1_goal_<batch_label>/goal_images/episode_000000.png
```

Every episode must have a corresponding goal image. The image must show the block clearly held at hover pose, correctly aligned, with the slot visible below. If any are missing or blurry, append using `--resume=true` and recollect those episodes.

### After Phase 3 collection — offline processing and merge

Two offline steps are required on each Phase 3 batch before merging:

1. Add spatial features (top-down shape detector → `observation.environment_state`)
2. Register goal images as a proper video feature (`observation.images.goal`)

Both must run on the Phase 3 batch before it can be merged with Phase 1b.

#### Step 1 — Add spatial features

```bash
# Dry run first:
python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_goal_<batch_label> \
    --camera_key observation.images.top \
    --detector_type shape_top \
    --dry_run

# Full run:
python scripts/add_spatial_features.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_goal_<batch_label> \
    --camera_key observation.images.top \
    --detector_type shape_top
```

#### Step 2 — Register goal images as a dataset feature

Goal images are saved as `.png` files during recording but are not yet part of the LeRobot dataset
schema. This script converts them into a proper video feature so the training loader picks them
up automatically as `observation.images.goal`.

```bash
# Dry run first — checks all goal images exist:
python scripts/add_goal_image_feature.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_goal_<batch_label> \
    --dry_run

# Full run:
python scripts/add_goal_image_feature.py \
    --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_goal_<batch_label>
```

Verify the new feature exists:

```bash
python -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    'rgragulraj/policy1_goal_<batch_label>',
    root=Path.home() / '.cache/huggingface/lerobot/rgragulraj/policy1_goal_<batch_label>',
)
assert 'observation.images.goal' in ds.meta.video_keys, 'MISSING: goal feature not registered'
assert 'observation.environment_state' in ds.meta.features, 'MISSING: spatial features not added'
item = ds[0]
print('goal image shape:', item['observation.images.goal'].shape)
print('spatial token:', item['observation.environment_state'])
print('OK')
"
```

#### Step 3 — Merge Phase 1b + Phase 3 batches

Run steps 1 and 2 on every Phase 3 batch before merging.

```python
from lerobot.datasets.aggregate import aggregate_datasets

aggregate_datasets(
    repo_ids=[
        "rgragulraj/policy1_merged",           # Phase 1b (already has spatial features, no goal images)
        "rgragulraj/policy1_goal_batch1",       # Phase 3 (spatial features + goal images)
        "rgragulraj/policy1_goal_batch2",
        # ... add all Phase 3 batches
    ],
    output_repo_id="rgragulraj/policy1_goal_merged",
    local_dir="~/.cache/huggingface/lerobot/rgragulraj/policy1_goal_merged",
)
```

> **Note:** Phase 1b episodes do not have goal images, but the policy handles this gracefully —
> the `use_goal_image` forward pass is skipped when `observation.images.goal` is absent from the
> batch. This is correct behaviour: Phase 1b episodes train the spatial-conditioning path while
> Phase 3 episodes additionally train the orientation-correction path.

Recompute normalisation stats after merging:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import compute_episode_stats, aggregate_stats

dataset = LeRobotDataset(
    "rgragulraj/policy1_goal_merged",
    root="~/.cache/huggingface/lerobot/rgragulraj/policy1_goal_merged",
)
compute_episode_stats(dataset)
aggregate_stats(dataset)
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

### Phase 0 — vanilla baseline

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_baseline \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_baseline \
  --policy.push_to_hub=false \
  --training.num_train_steps=80000 \
  --output_dir=outputs/policy1_baseline
```

### Phase 1a — augmentation only (retrain on Phase 0 data)

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_baseline \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_baseline \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.brightness.min_max="[0.5, 1.5]" \
  --dataset.image_transforms.contrast.min_max="[0.5, 2.0]" \
  --dataset.image_transforms.hue.min_max="[-0.1, 0.1]" \
  --policy.push_to_hub=false \
  --training.num_train_steps=80000 \
  --output_dir=outputs/policy1_phase1a
```

### Phase 1b — diverse dataset (merged)

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_merged \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_merged \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.brightness.min_max="[0.5, 1.5]" \
  --dataset.image_transforms.contrast.min_max="[0.5, 2.0]" \
  --dataset.image_transforms.hue.min_max="[-0.1, 0.1]" \
  --policy.push_to_hub=false \
  --training.num_train_steps=100000 \
  --output_dir=outputs/policy1_phase1b
```

Validation targets:

- Location ±10 cm hover success: >40% (up from baseline)
- Location ±15 cm hover success: >20%

### Phase 1c — backbone unfreeze (optional, >500 episodes only)

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_merged \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_merged \
  --dataset.image_transforms.enable=true \
  --policy.unfreeze_backbone_layers='["layer4"]' \
  --policy.optimizer_lr=1e-4 \
  --policy.optimizer_lr_backbone=1e-6 \
  --policy.push_to_hub=false \
  --training.num_train_steps=100000 \
  --output_dir=outputs/policy1_phase1c
```

### Phase 2 — spatial conditioning

After running `add_spatial_features.py` on `policy1_merged`:

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_merged \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_merged \
  --policy.use_spatial_conditioning=true \
  --policy.spatial_conditioning_dim=10 \
  --dataset.image_transforms.enable=true \
  --policy.push_to_hub=false \
  --training.num_train_steps=80000 \
  --output_dir=outputs/policy1_phase2
```

Validation: hover-above-slot tolerance should expand from ±10 cm to ±20 cm+. If not, check detector output for noisy angle values.

### Phase 3 — goal image + spatial conditioning

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_goal_merged \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_goal_merged \
  --policy.use_goal_image=true \
  --policy.use_shared_goal_backbone=true \
  --policy.use_spatial_conditioning=true \
  --policy.spatial_conditioning_dim=10 \
  --dataset.image_transforms.enable=true \
  --policy.push_to_hub=false \
  --training.num_train_steps=150000 \
  --output_dir=outputs/policy1_phase3
```

Validation: test on 3 novel slot geometries with a goal image provided at inference. Orientation failure rate should drop vs Phase 2 levels.

### Phase 4 — language + goal image + spatial conditioning

See the Phase 4 section below for data labelling, offline steps, and the full training command. The key flags added over Phase 3:

```bash
lerobot-train \
  ... \
  --policy.use_language_conditioning=true \
  --policy.language_dim=512 \
  --training.num_train_steps=200000 \
  --output_dir=outputs/policy1_phase4
```

### Phase 3 — inference

At deployment, the goal image must be provided per episode. Use `GoalImageProcessorStep` in
the pre-processing pipeline alongside `SpatialConditioningProcessorStep`.

**Capture the goal image for a novel slot:**

1. Bring the arm to the canonical hover pose above the new slot (by hand or with
   `go_to_start_position.py`).
2. Place the block in the gripper, correctly aligned with the slot's insertion axis.
3. Wait for the arm to stabilise (2–3 seconds — no motion blur).
4. Capture the top-down frame:

   ```bash
   python -c "
   import cv2; cap = cv2.VideoCapture(5)
   ret, frame = cap.read(); cap.release()
   import cv2; cv2.imwrite('goal_novel_slot.png', frame)
   print('Saved goal_novel_slot.png')
   "
   ```

**Wire up the processor in your inference script:**

```python
from lerobot.policies.act.processor_act import GoalImageProcessorStep, SpatialConditioningProcessorStep

# Create once at script startup:
goal_step = GoalImageProcessorStep(
    goal_image_path="goal_novel_slot.png",
    target_size=(480, 640),  # must match training resolution
)
spatial_step = SpatialConditioningProcessorStep(
    camera_key="observation.images.top",
    detector_type="shape_top",
    calibration_path="scripts/top_shape_calibration.json",
    ema_alpha=0.5,
)

# Insert both steps before the normaliser in the pre-processing pipeline:
# [..., goal_step, spatial_step, AddBatchDimensionProcessorStep(), ...]

# At the start of each episode with a new goal, call:
goal_step.reset_goal("goal_novel_slot_v2.png")
```

The goal image is reused for all timesteps in the episode — call `reset_goal()` only when the
target slot changes, not every step.

---

## Phase 4 — Language Conditioning (multi-shape routing)

**Purpose:** Allow a single Policy 1 checkpoint to handle multiple slot shapes via natural-language commands. "Hover above the square slot" → the policy selects the square slot; "hover above the round slot" → the round slot. Internally the language command is encoded as a CLIP embedding and fed as an extra encoder token; the spatial detector is simultaneously filtered to the correct slot geometry by shape name.

**No re-recording required** — Phase 4 is an offline step on existing data plus a configuration change. Collect the **multi-shape dataset** described below before running it.

### Phase 4 data collection — multi-shape episodes

Collect separate batches per shape using a consistent task description per batch. The task description is the CLIP input — keep it short and identical across all episodes of the same shape.

```bash
conda activate lerobot

# Square slot:
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/policy1_lang_square \
    --dataset.single_task="hover above the square slot" \
    --dataset.num_episodes=100 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=false \
    --display_data=true

# Round slot (change repo_id, single_task, and physically swap the slot insert):
lerobot-record \
    ... \
    --dataset.repo_id=rgragulraj/policy1_lang_round \
    --dataset.single_task="hover above the round slot" \
    ...
```

**Task description format — rules:**

| Rule                                                                  | Rationale                                                                    |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Lowercase, no punctuation                                             | CLIP tokenisation is case-sensitive in rare edge cases — lowercase is safest |
| Same phrasing across episodes                                         | Embedding space encodes shape identity, not phrasing variation               |
| Include the shape word exactly once                                   | Parser matches first shape word; ambiguous phrases cause routing errors      |
| Examples: "hover above the square slot", "hover above the round slot" |                                                                              |

Minimum 80–100 episodes per shape. Vary block position, lighting, and approach trajectory within each shape's batch — but keep the task string fixed.

### After Phase 4 collection — offline steps

Run all three steps on each Phase 4 batch before merging:

1. **Spatial features** (same as Phase 2 — one calibrated shape profile per slot):

   ```bash
   python scripts/add_spatial_features.py \
       --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_lang_square \
       --camera_key observation.images.top \
       --detector_type shape_top \
       --shape square

   python scripts/add_spatial_features.py \
       --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_lang_round \
       --camera_key observation.images.top \
       --detector_type shape_top \
       --shape round
   ```

2. **Goal images** (optional but recommended — reuse Phase 3 infrastructure):

   ```bash
   python scripts/add_goal_image_feature.py \
       --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_lang_square
   python scripts/add_goal_image_feature.py \
       --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_lang_round
   ```

3. **Language embeddings** — converts the `single_task` description into a 512-float CLIP embedding per frame:

   ```bash
   # Dry run first — shows which task strings will be encoded:
   python scripts/add_language_features.py \
       --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_lang_square \
       --dry_run

   # Full run:
   python scripts/add_language_features.py \
       --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_lang_square

   python scripts/add_language_features.py \
       --dataset_root ~/.cache/huggingface/lerobot/rgragulraj/policy1_lang_round
   ```

   The CLIP encoder (ViT-B/32) is NOT stored in the model checkpoint — only the linear projection layer lives in ACT. This keeps the checkpoint ~150 MB smaller.

4. **Merge all Phase 4 batches** with the Phase 1b / Phase 3 data:

   ```python
   from lerobot.datasets.aggregate import aggregate_datasets

   aggregate_datasets(
       repo_ids=[
           "rgragulraj/policy1_goal_merged",   # Phase 1b + Phase 3 (no language labels)
           "rgragulraj/policy1_lang_square",    # Phase 4 — square slot
           "rgragulraj/policy1_lang_round",     # Phase 4 — round slot
       ],
       output_repo_id="rgragulraj/policy1_lang_merged",
       local_dir="~/.cache/huggingface/lerobot/rgragulraj/policy1_lang_merged",
   )
   ```

   > **Note:** Phase 1b/Phase 3 episodes do not have `observation.language`. The language token injection
   > in `modeling_act.py` is guarded with `and "observation.language" in batch` — unlabelled episodes
   > silently skip the language token. This is correct: they train the spatial + goal image paths while
   > Phase 4 episodes additionally train the language routing path.

5. **Recompute normalisation stats:**

   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   from lerobot.datasets.utils import compute_episode_stats, aggregate_stats

   dataset = LeRobotDataset(
       "rgragulraj/policy1_lang_merged",
       root="~/.cache/huggingface/lerobot/rgragulraj/policy1_lang_merged",
   )
   compute_episode_stats(dataset)
   aggregate_stats(dataset)
   ```

### Phase 4 — verification

```bash
python -c "
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    'rgragulraj/policy1_lang_merged',
    root=Path.home() / '.cache/huggingface/lerobot/rgragulraj/policy1_lang_merged',
)
# Check language feature is present
has_lang = 'observation.language' in ds.meta.features
has_spatial = 'observation.environment_state' in ds.meta.features
print(f'observation.language: {has_lang}')
print(f'observation.environment_state: {has_spatial}')
item = ds[0]
if has_lang:
    lang = item.get('observation.language')
    print(f'language embedding shape: {lang.shape if lang is not None else None}')
print('OK')
"
```

### Phase 4 training

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_lang_merged \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_lang_merged \
  --policy.use_goal_image=true \
  --policy.use_shared_goal_backbone=true \
  --policy.use_spatial_conditioning=true \
  --policy.spatial_conditioning_dim=10 \
  --policy.use_language_conditioning=true \
  --policy.language_dim=512 \
  --dataset.image_transforms.enable=true \
  --policy.push_to_hub=false \
  --training.num_train_steps=200000 \
  --output_dir=outputs/policy1_phase4
```

### Phase 4 — inference

At deployment, pass a natural-language command via `LanguageConditioningProcessorStep`. The step:

1. Encodes the command with CLIP ViT-B/32 (loaded lazily once).
2. Parses the shape word and calls `spatial_step.update_shape()` so the top-camera detector filters to the correct slot geometry.
3. Injects the embedding as `observation.language` per step.

```python
from lerobot.policies.act.processor_act import (
    GoalImageProcessorStep,
    LanguageConditioningProcessorStep,
    SpatialConditioningProcessorStep,
)

# Create once at script startup:
spatial_step = SpatialConditioningProcessorStep(
    camera_key="observation.images.top",
    detector_type="shape_top",
    calibration_path="scripts/top_shape_calibration.json",
    ema_alpha=0.5,
)
goal_step = GoalImageProcessorStep(
    goal_image_path="goal_square_slot.png",
    target_size=(480, 640),
)
lang_step = LanguageConditioningProcessorStep(
    model_name="openai/clip-vit-base-patch32",
    spatial_step=spatial_step,  # bridge: parsed shape → detector shape filter
)

# At the start of each episode, set command and goal:
lang_step.set_command("hover above the square slot")
# ^ automatically calls spatial_step.update_shape("square")
goal_step.reset_goal("goal_square_slot.png")

# Insert all three steps before AddBatchDimensionProcessorStep in the pipeline:
# [..., lang_step, goal_step, spatial_step, AddBatchDimensionProcessorStep(), ...]
```

**Shape word routing:**

| Command word                  | Canonical shape | Detector profile                              |
| ----------------------------- | --------------- | --------------------------------------------- |
| "square"                      | "square"        | `top_shape_calibration.json` → square profile |
| "round", "circular", "circle" | "round"         | → round profile                               |
| "hex", "hexagon", "hexagonal" | "hex"           | → hex profile                                 |
| "triangle", "triangular"      | "triangle"      | → triangle profile                            |

If the command contains no recognised shape word, `parsed_shape` is `None` and `update_shape()` is not called — the detector retains its current shape filter.

---

## Session targets and tracking

Use this table to track collection progress across sessions:

| Session | Date | Lighting  | Objects        | Slot position | Cells covered              | Episodes | Notes                  |
| ------- | ---- | --------- | -------------- | ------------- | -------------------------- | -------- | ---------------------- |
| 1       | —    | Overhead  | Square block   | Centre        | TL TC TR ML MC MR BL BC BR | 50       | Phase 0 baseline       |
| 2       | —    | Overhead  | Square block   | Centre        | TL TC TR ML MC MR BL BC BR | 90       | Phase 1b grid coverage |
| 3       | —    | Side lamp | Square + Obj 2 | Left          | TL ML BL TC MC BC          | 80       |                        |
| ...     |      |           |                |               |                            |          |                        |

Target breakdown for a minimum viable 300-episode dataset:

| Dimension                       | Minimum             |
| ------------------------------- | ------------------- |
| Block positions across 3×3 grid | All 9 cells covered |
| Objects                         | 3+ distinct objects |
| Slot positions                  | 3+ positions        |
| Lighting conditions             | 2+ conditions       |
| Total episodes                  | 300                 |
