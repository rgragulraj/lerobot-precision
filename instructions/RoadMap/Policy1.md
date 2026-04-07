# Policy 1: Coarse Generalisation Policy — Instructions Manual

**Author:** LENS Lab
**Last updated:** 2026-04-07
**Hardware:** SO-101 (follower + leader arm), top-down webcam (index 5), gripper camera (index 7)

---

## 1. What Policy 1 Does

Policy 1 is the **generalisation policy**. Its job is to pick up a block from anywhere in the workspace, orient it correctly relative to the slot, and deliver it to a **canonical hover pose** — a fixed position directly above the slot at approximately 4 cm height. It does **not** insert the block. That is Policy 2's job.

```
Start state:   Arm at home position. Block placed anywhere in the 3×3 workspace grid.
               Slot may be at any trained slot position.

End state:     Block held by gripper, hovering ~4 cm above the slot centre,
               roughly aligned in orientation (±1.5 cm, ±15°).

Success criterion (Tier 3 handoff):
               Block centred within ±1.5 cm of slot centre.
               Block rotated within ±15° of correct insertion orientation.
               Gripper height within ±1 cm of canonical hover height.
```

This handoff envelope (±1.5 cm, ±15°) is what Policy 2 is trained to accept. Policy 1's only obligation is to deliver within this envelope. It does not need millimetre precision — that is Policy 2's domain.

---

## 2. Development Phases Overview

Policy 1 is developed in four phases, each adding capability on top of the last. **Do not skip phases.** Each phase addresses a specific failure mode, and running the Phase 0 baseline tells you which failure mode is the actual bottleneck.

| Phase    | What it adds                              | Effort                        | When to do it                      |
| -------- | ----------------------------------------- | ----------------------------- | ---------------------------------- |
| Phase 0  | Baseline — establishes what fails and why | 2 days                        | Always first                       |
| Phase 1a | Aggressive image augmentation             | ~30 min code change           | After Phase 0                      |
| Phase 1b | Systematic data diversity                 | 2–3 days collection           | After Phase 1a                     |
| Phase 1c | Selective backbone unfreeze               | ~1 hour code change           | Only if >500 episodes              |
| Phase 2  | Spatial conditioning via keypoints        | ~50 lines of code, 1–2 weeks  | If location fails after Phase 1    |
| Phase 3  | Goal image conditioning                   | ~80 lines of code, 2–3 weeks  | If orientation fails after Phase 2 |
| Phase 4  | Language conditioning                     | ~150 lines of code, 3–4 weeks | Last — enables multi-task routing  |

---

## 3. The Canonical Hover Pose

The canonical hover pose is a single, fixed arm configuration that serves as the interface between Policy 1 and Policy 2. It must be defined, saved, and used consistently throughout all Policy 1 data collection.

**Definition:**

- Gripper is directly above the slot centre (±0 mm in x/y)
- Gripper height: ~4 cm above the top of the slot opening
- Block orientation: aligned with the slot's insertion axis
- This position is saved in `instructions/start_positions/insert_above_slot.json`

**To set the canonical hover pose:**

1. Teleoperate the arm to the desired hover position manually
2. Save the joint positions:
   ```bash
   python scripts/go_to_start_position.py --save --file instructions/start_positions/insert_above_slot.json
   ```
3. Verify it by running:
   ```bash
   python scripts/go_to_start_position.py --file instructions/start_positions/insert_above_slot.json
   ```

**Important:** This pose is the ground truth target for all Policy 1 demonstrations. Every episode must end at or very near this pose. Inconsistent hover poses across episodes create contradictory training signal and degrade performance.

---

## 4. Phase 0: Baseline Data Collection and Training

**Do not skip Phase 0. It tells you which failure mode to invest in.**

### 4.1 What to collect

Collect 50 demonstrations under fixed, controlled conditions:

- Block: one fixed object (e.g. the square block used in earlier experiments)
- Block start position: one fixed location in the centre of the workspace
- Slot: one fixed position and orientation
- Camera: both gripper camera (index 7) and top-down webcam (index 5)
- Arm start: home position every episode

Each demo ends when the arm reaches the canonical hover pose with the block in the gripper. **Do not insert. Stop at hover.**

### 4.2 Recording command

```bash
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

**Episode time:** 30 seconds is enough. Most well-executed pick-and-hover demos should complete in 8–15 seconds. Press `→` as soon as the arm reaches the hover pose — don't pad with dead time.

**Reset between episodes:** Move the arm back to home, place the block back at the fixed start position.

### 4.3 Training the baseline

Copy to remote PC via USB, then train:

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

**80k steps** for a 50-episode dataset. This is the Phase 0 baseline training budget.

### 4.4 Phase 0 evaluation protocol

After training, run 30 trials in each of the following conditions. **Record pass/fail for each trial. Note the specific failure mode.**

| Test condition                | How to set it up                               | Target (baseline)  |
| ----------------------------- | ---------------------------------------------- | ------------------ |
| Fixed (training distribution) | Exact same block position and slot as training | >70% hover success |
| Location ±5 cm                | Shift block 5 cm in any direction              | —                  |
| Location ±10 cm               | Shift block 10 cm                              | —                  |
| Location ±15 cm               | Shift block 15 cm                              | —                  |
| Novel object (similar)        | Swap to object of similar size/shape           | —                  |
| Novel object (different)      | Swap to visually different object              | —                  |
| Novel slot position           | Move slot to 3 different positions             | —                  |

**At each failure, classify which sub-step failed:**

- **(a) Wrong approach** — arm moves to wrong area entirely → **location generalisation is the bottleneck → prioritise Phase 2**
- **(b) Grasp slip** — arm arrives at correct area but wrong grip/pick → **more object diversity in data + augmentation (Phase 1a/1b)**
- **(c) Wrong orientation at hover** — picked correctly but wrong orientation when arriving above slot → **Phase 3 (goal image conditioning)**

**Decision rule after Phase 0:**

- If failure mode (a) dominates: proceed through Phase 1 → Phase 2
- If failure mode (b) dominates: prioritise Phase 1a + 1b (diverse objects, augmentation)
- If failure mode (c) dominates: prioritise Phase 3 after Phase 1

---

## 5. Phase 1a: Aggressive Image Augmentation

**Effort:** ~30 minutes. **Highest ROI change available.** Do this before collecting more data.

### 5.1 What to change

**File:** `src/lerobot/datasets/transforms.py`

The `ImageTransformsConfig` has augmentation disabled by default (`enable: False`). Enable and expand the augmentation ranges.

The transforms are applied in `make_transform_from_config()`. The current defaults are conservative. For generalisation across lighting and workspace positions, they need to be expanded:

| Transform          | Current default | Phase 1a value      |
| ------------------ | --------------- | ------------------- |
| `enable`           | `False`         | `True`              |
| `brightness`       | `(0.8, 1.2)`    | `(0.5, 1.5)`        |
| `contrast`         | `(0.8, 1.2)`    | `(0.5, 2.0)`        |
| `saturation`       | `(0.5, 1.5)`    | `(0.5, 1.5)` (keep) |
| `hue`              | `(-0.05, 0.05)` | `(-0.1, 0.1)`       |
| `affine translate` | `(0.05, 0.05)`  | `(0.1, 0.1)`        |

The `affine translate=(0.1, 0.1)` corresponds to ~5–8 cm of effective workspace shift at 640px — this is the cheapest location generalisation available without collecting new data.

### 5.2 Additional transforms to add

Add these two transforms by adding new `elif` branches in `make_transform_from_config()` in `transforms.py`:

**RandomResizedCrop** — simulates small zoom changes and camera position variation:

```python
transforms.RandomResizedCrop(
    size=(config.image_size, config.image_size),
    scale=(0.85, 1.0),
    ratio=(0.9, 1.1),
)
```

**RandomErasing** — partial occlusion robustness (block partially hidden by arm or other objects):

```python
transforms.RandomErasing(
    p=0.3,
    scale=(0.02, 0.1),
    ratio=(0.3, 3.3),
    value=0,
)
```

### 5.3 How to enable in training config

Pass augmentation flags when launching training:

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_baseline \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_baseline \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.brightness.weight=1 \
  --dataset.image_transforms.brightness.min_max="[0.5, 1.5]" \
  --dataset.image_transforms.contrast.weight=1 \
  --dataset.image_transforms.contrast.min_max="[0.5, 2.0]" \
  --dataset.image_transforms.hue.weight=1 \
  --dataset.image_transforms.hue.min_max="[-0.1, 0.1]" \
  --policy.push_to_hub=false \
  --training.num_train_steps=80000 \
  --output_dir=outputs/policy1_phase1a
```

**Validation:** Rerun the Phase 0 evaluation protocol. If location ±10 cm hover success improves by 15%+ over the Phase 0 baseline, augmentation is contributing. If not, the bottleneck is data diversity (Phase 1b).

---

## 6. Phase 1b: Systematic Data Diversity

**Effort:** 2–3 days of data collection. **Do Phase 1a first, then collect this data.**

This is the main data collection effort for Policy 1. The goal is to systematically cover the workspace, object space, slot position space, and lighting conditions. The total target is **300–500 episodes**.

### 6.1 Workspace grid coverage

Divide the reachable workspace into a **3×3 grid** (9 cells). Collect **15–20 episodes per cell** for each object. Mark cell boundaries with tape on the table.

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

Place the block **within the cell**, not at its centre. Vary the exact position within the cell between episodes — don't place it at the same spot every time.

### 6.2 Object diversity

Collect episodes across **at least 5 objects**:

| Object                                  | Quantity                              | Notes                             |
| --------------------------------------- | ------------------------------------- | --------------------------------- |
| Original square block                   | 60 eps (from Phase 0 + grid coverage) | Training baseline                 |
| Object 2: different colour, same shape  | 40 eps                                | Tests colour generalisation       |
| Object 3: similar shape, different size | 40 eps                                | Tests scale generalisation        |
| Object 4: different shape (e.g. round)  | 40 eps                                | Tests shape generalisation        |
| Object 5: visually distinct (wildcard)  | 40 eps                                | Tests broad visual generalisation |

For each object, distribute the 40 episodes across at least 5 of the 9 grid cells. Cover all 9 cells for the primary object.

### 6.3 Slot position diversity

Collect with the slot at **at least 5 different positions** in the workspace. For each slot position, collect 40 episodes:

- Position 1: Standard centre position (current default)
- Position 2: Shifted 10 cm left
- Position 3: Shifted 10 cm right
- Position 4: Shifted 10 cm toward robot
- Position 5: Shifted 10 cm away from robot

This teaches Policy 1 to approach the slot correctly regardless of where it is, not just memorise a single slot location.

### 6.4 Lighting variation

Vary lighting **between sessions**, not within sessions. Each session should have consistent lighting. Three lighting conditions:

| Condition     | Setup                                   |
| ------------- | --------------------------------------- |
| Overhead on   | Lab overhead lighting, no other sources |
| Side lamp     | Overhead off, desk lamp from one side   |
| Natural light | Window light, overhead off              |

Collect at least 80–100 episodes in each lighting condition. Mix objects and slot positions within each lighting session.

### 6.5 Camera tilt variation

Between sessions, vary the top-down webcam tilt by ±5 degrees (physically adjust the mount). This teaches the policy to be robust to minor camera misalignments between sessions. Keep the gripper camera mount fixed and mechanically stable.

### 6.6 Recording commands for Phase 1b

Use the same base command as Phase 0, updating the dataset name and task description per batch:

```bash
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

Label batches clearly, e.g. `policy1_diverse_grid_obj1`, `policy1_diverse_obj2`, `policy1_diverse_slot_right`, etc.

### 6.7 Merging datasets

After collecting multiple batches, merge them into a single training dataset:

```python
from lerobot.datasets.aggregate import aggregate_datasets

datasets = [
    "rgragulraj/policy1_baseline",
    "rgragulraj/policy1_diverse_grid_obj1",
    "rgragulraj/policy1_diverse_obj2",
    "rgragulraj/policy1_diverse_slot_right",
    # ... add all batch names
]

aggregate_datasets(
    repo_ids=datasets,
    output_repo_id="rgragulraj/policy1_merged",
    local_dir="~/.cache/huggingface/lerobot/rgragulraj/policy1_merged",
)
```

**Critical:** After every merge, recompute normalisation stats before training:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import compute_episode_stats, aggregate_stats

dataset = LeRobotDataset("rgragulraj/policy1_merged",
                          root="~/.cache/huggingface/lerobot/rgragulraj/policy1_merged")
compute_episode_stats(dataset)
aggregate_stats(dataset)
```

Verify the stats file exists at `~/.cache/huggingface/lerobot/rgragulraj/policy1_merged/meta/stats.json` before training. Stale normalisation stats are a common silent failure mode that produces nonsensical actions.

### 6.8 Training with merged dataset

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

**100k steps** for the merged dataset. Monitor validation loss — if it plateaus early, check for data quality issues before adding more steps.

**Validation:** Rerun Phase 0 evaluation. Target: hover-above-slot success at ±10 cm offset goes from ~20% to ~60%+. If achieved, Phase 2 can wait. If location still fails beyond ±10 cm, proceed to Phase 2.

---

## 7. Phase 1c: Selective Backbone Unfreeze (Optional)

**Only attempt this if you have >500 Policy 1 episodes.** With fewer episodes, unfrozen batch norm statistics tend to overfit to your specific objects and lighting conditions, making things worse.

### 7.1 Code changes

**File:** `src/lerobot/policies/act/configuration_act.py`

Add a new field to `ACTConfig`:

```python
from dataclasses import field

# In ACTConfig class:
unfreeze_backbone_layers: list[str] = field(default_factory=list)
# Set to ["layer4"] to unfreeze the final ResNet block only.
# Unfreezes feature extractor fine-tuning for manipulation-relevant visual cues.
```

**File:** `src/lerobot/policies/act/modeling_act.py`

In the model `__init__`, after the backbone is created, add:

```python
if config.unfreeze_backbone_layers:
    # Freeze everything first
    for param in self.backbone.parameters():
        param.requires_grad = False
    # Selectively unfreeze the specified layers
    for layer_name in config.unfreeze_backbone_layers:
        layer = getattr(self.backbone, layer_name, None)
        if layer is not None:
            for param in layer.parameters():
                param.requires_grad = True
    # Important: keep FrozenBatchNorm2d — do NOT switch to regular BatchNorm
    # and do NOT call model.train() on frozen layers (BN stats must stay fixed)
```

### 7.2 Training config for Phase 1c

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

The `optimizer_lr_backbone=1e-6` is 100× lower than the head LR. This prevents catastrophic forgetting in the backbone while still allowing it to shift toward manipulation-relevant features.

---

## 8. Phase 2: Spatial Conditioning via Keypoints

**When to do this:** Phase 0 evaluation shows the arm consistently goes to the wrong area of the workspace (failure mode (a)) even after Phase 1 data diversity.

**Estimated effort:** 1–2 weeks, ~50 lines of code.

### 8.1 What it does

Instead of relying purely on visual features, the policy gets an explicit 8-float spatial token:

```
[cx_block, cy_block, w_block, h_block, cx_slot, cy_slot, w_slot, h_slot]
```

All values are normalised to [0, 1] in image space. This directly tells the policy where the block and slot are, making location generalisation dramatically easier. The values are computed from the top-down camera at each timestep using a colour-based OpenCV detector.

For insertion tasks where orientation matters, extend to 10 floats by adding `angle_block` and `angle_slot` (computed via `cv2.minAreaRect()`):

```
[cx_block, cy_block, w_block, h_block, angle_block, cx_slot, cy_slot, w_slot, h_slot, angle_slot]
```

### 8.2 Shape detector script

Create `scripts/detect_block_slot.py`:

```python
import cv2
import numpy as np
import json
from pathlib import Path

def detect_block_and_slot(frame_bgr, block_hsv_range, slot_hsv_range, include_angle=False):
    """
    Detect block and slot bounding boxes from a BGR camera frame.

    Args:
        frame_bgr: BGR camera frame from top-down webcam.
        block_hsv_range: Tuple of (lower_hsv, upper_hsv) numpy arrays for block colour.
        slot_hsv_range: Tuple of (lower_hsv, upper_hsv) numpy arrays for slot colour.
        include_angle: If True, returns 10 floats including rotation angles.

    Returns:
        (block_vec, slot_vec) each as (cx, cy, w, h) or (cx, cy, w, h, angle),
        normalised to [0, 1]. Returns None for each if not detected.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, w = frame_bgr.shape[:2]

    def find_bbox(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 200:  # noise threshold (pixels)
            return None
        if include_angle:
            rect = cv2.minAreaRect(c)
            (cx, cy), (bw, bh), angle = rect
            return cx / w, cy / h, bw / w, bh / h, angle / 180.0  # normalise angle to [-0.5, 0.5]
        else:
            x, y, bw, bh = cv2.boundingRect(c)
            return (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    block_mask = cv2.inRange(hsv, *block_hsv_range)
    block_mask = cv2.morphologyEx(block_mask, cv2.MORPH_OPEN, kernel)

    slot_mask = cv2.inRange(hsv, *slot_hsv_range)
    slot_mask = cv2.morphologyEx(slot_mask, cv2.MORPH_OPEN, kernel)

    return find_bbox(block_mask), find_bbox(slot_mask)


def calibrate_hsv_range(camera_index=5):
    """Interactive HSV calibration tool. Saves result to a JSON file."""
    cap = cv2.VideoCapture(camera_index)
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 255])

    def nothing(x):
        pass

    cv2.namedWindow("HSV Calibration")
    for name, max_val, default in [("H_low", 180, 0), ("S_low", 255, 0), ("V_low", 255, 0),
                                    ("H_high", 180, 180), ("S_high", 255, 255), ("V_high", 255, 255)]:
        cv2.createTrackbar(name, "HSV Calibration", default, max_val, nothing)

    print("Adjust sliders to isolate your object. Press 's' to save, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([cv2.getTrackbarPos("H_low", "HSV Calibration"),
                          cv2.getTrackbarPos("S_low", "HSV Calibration"),
                          cv2.getTrackbarPos("V_low", "HSV Calibration")])
        upper = np.array([cv2.getTrackbarPos("H_high", "HSV Calibration"),
                          cv2.getTrackbarPos("S_high", "HSV Calibration"),
                          cv2.getTrackbarPos("V_high", "HSV Calibration")])
        mask = cv2.inRange(hsv, lower, upper)
        cv2.imshow("HSV Calibration", cv2.bitwise_and(frame, frame, mask=mask))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            result = {"lower": lower.tolist(), "upper": upper.tolist()}
            path = Path("instructions/hsv_ranges")
            path.mkdir(exist_ok=True)
            name_in = input("Save as (e.g. 'square_block'): ")
            with open(path / f"{name_in}.json", "w") as f:
                json.dump(result, f)
            print(f"Saved to {path / name_in}.json")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

Run calibration for each new object/slot colour:

```bash
python scripts/detect_block_slot.py --calibrate --camera_index=5
```

This takes 5–10 minutes per new object. Save HSV ranges to `instructions/hsv_ranges/<object_name>.json`.

### 8.3 ACT config changes

**File:** `src/lerobot/policies/act/configuration_act.py`

Add to `ACTConfig`:

```python
# Spatial conditioning — injects block/slot keypoints as an extra encoder token
use_spatial_conditioning: bool = False
spatial_conditioning_dim: int = 8
# Set to 8 for (cx, cy, w, h) × 2. Set to 10 if including rotation angles.
```

In `validate_features()`, add:

```python
if self.use_spatial_conditioning:
    if "observation.environment_state" not in self.input_features:
        raise ValueError(
            "use_spatial_conditioning=True requires 'observation.environment_state' "
            f"with shape ({self.spatial_conditioning_dim},) in input_features."
        )
```

**File:** `src/lerobot/policies/act/modeling_act.py`

The existing `env_state_feature` path in `ACTPolicy` already handles injecting a flat vector as a token into the encoder sequence. When `use_spatial_conditioning=True` and `observation.environment_state` is present in the batch, it will be picked up automatically — no Transformer architecture changes are needed.

### 8.4 Processor changes

**File:** `src/lerobot/policies/act/processor_act.py`

Add a `SpatialConditioningProcessorStep` class:

```python
class SpatialConditioningProcessorStep:
    """
    Runs the OpenCV block/slot detector on each top-down camera frame
    and injects the keypoint vector as observation.environment_state.

    Uses exponential moving average (EMA) over 3 frames to smooth noisy detections.
    Falls back to a zero vector if detection fails.
    """

    def __init__(self, hsv_ranges_path: str, include_angle: bool = False, ema_alpha: float = 0.5):
        import json
        from pathlib import Path
        ranges = json.loads(Path(hsv_ranges_path).read_text())
        import numpy as np
        self.block_hsv = (np.array(ranges["block"]["lower"]), np.array(ranges["block"]["upper"]))
        self.slot_hsv  = (np.array(ranges["slot"]["lower"]),  np.array(ranges["slot"]["upper"]))
        self.include_angle = include_angle
        self.ema_alpha = ema_alpha
        self._ema = None

    def __call__(self, batch: dict) -> dict:
        import numpy as np
        frame = batch["observation.images.top"]  # shape: (H, W, 3) or (C, H, W)
        if frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)  # CHW → HWC
        frame_bgr = frame[..., ::-1].copy()  # RGB → BGR

        block, slot = detect_block_and_slot(
            frame_bgr, self.block_hsv, self.slot_hsv, include_angle=self.include_angle
        )
        vec = np.zeros(10 if self.include_angle else 8, dtype=np.float32)
        if block is not None:
            vec[:len(block)] = block
        if slot is not None:
            offset = 5 if self.include_angle else 4
            vec[offset:offset + len(slot)] = slot

        # EMA smoothing over 3 frames
        if self._ema is None:
            self._ema = vec
        else:
            self._ema = self.ema_alpha * vec + (1 - self.ema_alpha) * self._ema

        batch["observation.environment_state"] = self._ema.copy()
        return batch
```

### 8.5 Processing training data offline

Run the detector on all existing Policy 1 training videos to add keypoints as a feature:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# Use dataset_tools.add_features() to add keypoints as observation.environment_state
# This writes a new float feature column to the parquet files.
# See lerobot/datasets/dataset_tools.py for the add_features() API.
```

After adding the feature, recompute normalisation stats (same as Section 6.7).

### 8.6 Training with spatial conditioning

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_spatial \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_spatial \
  --policy.use_spatial_conditioning=true \
  --policy.spatial_conditioning_dim=8 \
  --dataset.image_transforms.enable=true \
  --policy.push_to_hub=false \
  --training.num_train_steps=80000 \
  --output_dir=outputs/policy1_phase2
```

**Validation:** Rerun Phase 0 location test. If hover-above-slot tolerance expands from ±10 cm to ±20 cm+, spatial conditioning is working. If not, add attention logging on `ACTEncoderLayer.self_attn` and check whether the spatial token has high attention weights — if the policy is ignoring the spatial token, the detector output is likely too noisy.

---

## 9. Phase 3: Goal Image Conditioning

**When to do this:** Failure mode (c) dominates — arm picks correctly and approaches the hover zone but arrives with wrong block orientation relative to the slot.

**Estimated effort:** 2–3 weeks, ~80 lines of code.

### 9.1 What it does

The policy receives a **goal image** alongside each observation — a clean image showing the block correctly oriented above the slot (the target hover state). The policy learns to bring the current camera view toward the goal view. For a novel slot geometry, as long as you provide a goal image at inference, the policy has everything it needs to figure out the correct orientation.

### 9.2 Collecting goal images during data collection

After each Policy 1 demo reaches the hover pose:

1. Hold the arm still at the hover pose
2. Capture a clean frame from the top-down camera with no arm movement
3. Save this frame as the goal image for that episode

This adds ~10 seconds per episode. The goal image is stored as `observation.images.goal` in the dataset.

Modify the recording workflow to capture this frame. The simplest approach is to press a dedicated key (e.g. `g`) to capture the goal frame at the end of each episode before pressing `→` to save.

**What makes a good goal image:**

- Block is in the gripper, held cleanly at the hover pose
- Slot is visible in the frame below the block
- No arm blur — wait for the arm to stabilise before capturing
- Same lighting as the recording session

### 9.3 ACT config changes

**File:** `src/lerobot/policies/act/configuration_act.py`

Add to `ACTConfig`:

```python
# Goal image conditioning — provide a goal image showing the target state
use_goal_image: bool = False
goal_image_feature_key: str = "observation.images.goal"
use_shared_goal_backbone: bool = True
# If True, goal image uses the same ResNet backbone weights as the current-frame images.
# Recommended: True — forces current and goal frames into the same feature space,
# which helps the policy compute the delta between current state and goal state.
```

### 9.4 Architecture changes

**File:** `src/lerobot/policies/act/modeling_act.py`

In `ACTPolicy.__init__`, after the main camera backbone setup, add:

```python
if self.config.use_goal_image:
    if self.config.use_shared_goal_backbone:
        # Share weights with the main backbone
        self.goal_backbone = self.backbone
    else:
        # Separate backbone (heavier, more capacity, only if you have >300 goal episodes)
        self.goal_backbone = _make_backbone(config)

    self.encoder_goal_feat_input_proj = nn.Conv2d(
        backbone_out_channels, config.dim_model, kernel_size=1
    )
    self.encoder_goal_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(
        config.dim_model // 2
    )
```

In `ACTPolicy.forward()`, after the current-camera tokens are computed and appended to `encoder_in_tokens`, add a symmetric block for the goal image:

```python
if self.config.use_goal_image and self.config.goal_image_feature_key in batch:
    goal_img = batch[self.config.goal_image_feature_key]  # (B, C, H, W)
    goal_feat = self.goal_backbone(goal_img)               # (B, C', H', W')
    goal_feat = self.encoder_goal_feat_input_proj(goal_feat)  # (B, dim_model, H', W')
    goal_pos = self.encoder_goal_cam_feat_pos_embed(goal_feat).flatten(2).permute(2, 0, 1)
    goal_feat = goal_feat.flatten(2).permute(2, 0, 1)
    encoder_in_tokens = torch.cat([encoder_in_tokens, goal_feat], dim=0)
    encoder_in_pos_embed = torch.cat([encoder_in_pos_embed, goal_pos], dim=0)
    # The decoder already cross-attends to all encoder tokens — no decoder changes needed.
```

### 9.5 Training with goal image conditioning

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_goal \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_goal \
  --policy.use_goal_image=true \
  --policy.use_shared_goal_backbone=true \
  --policy.use_spatial_conditioning=true \
  --policy.spatial_conditioning_dim=10 \
  --dataset.image_transforms.enable=true \
  --policy.push_to_hub=false \
  --training.num_train_steps=150000 \
  --output_dir=outputs/policy1_phase3
```

**150k steps** — goal image conditioning adds more parameters and requires more training.

**Validation:** Test on 3 novel slot geometries (each with a goal image provided at inference). If orientation failure rate drops below Phase 2 levels, goal conditioning is working. If not, check whether the goal image is clean (arm stable, correct pose, clear view of slot).

---

## 10. Phase 4: Language Conditioning

**When to do this:** Last. After Phases 1–3 are validated. This enables multi-task routing — a single language command selects which Policy 1 specialisation to run.

**Estimated effort:** 3–4 weeks, ~150 lines of code. Full details are in `generalisation_roadmap.md` Phase 4 section.

**Summary of changes:**

- Add CLIP ViT-B/32 text encoder (frozen, ~150 MB) to `ACTPolicy`
- Add `encoder_language_input_proj` linear layer to project 512-dim CLIP embeddings to `dim_model`
- Prepend language token to encoder sequence
- Add `LanguageConditioningProcessorStep` to `processor_act.py`
- Label each episode with a short task description during data collection

This phase is not required for Policy 1 to work independently. It becomes relevant when connecting Policy 1 to a multi-task routing system.

---

## 11. ACT Configuration Summary Per Phase

| Parameter                  | Phase 0 (default) | Phase 1a    | Phase 1b                 | Phase 2                  | Phase 3      |
| -------------------------- | ----------------- | ----------- | ------------------------ | ------------------------ | ------------ |
| `chunk_size`               | 100               | 100         | 100                      | 100                      | 50           |
| `n_action_steps`           | 100               | 100         | 100                      | 100                      | 50           |
| `temporal_ensemble_coeff`  | None              | None        | None                     | 0.1                      | 0.1          |
| `kl_weight`                | 10.0              | 10.0        | 10.0                     | 10.0                     | 10.0         |
| `optimizer_lr`             | 1e-4              | 1e-4        | 1e-4                     | 1e-4                     | 1e-4         |
| `optimizer_lr_backbone`    | 1e-5              | 1e-5        | 1e-5                     | 1e-5                     | 1e-5         |
| `unfreeze_backbone_layers` | `[]`              | `[]`        | `["layer4"]` if >500 eps | `["layer4"]` if >500 eps | `["layer4"]` |
| `use_spatial_conditioning` | `False`           | `False`     | `False`                  | `True`                   | `True`       |
| `spatial_conditioning_dim` | —                 | —           | —                        | `8` or `10`              | `10`         |
| `use_goal_image`           | `False`           | `False`     | `False`                  | `False`                  | `True`       |
| Image augmentation         | disabled          | **enabled** | enabled                  | enabled                  | enabled      |
| Training steps             | 80k               | 80k         | 100k                     | 80k                      | 150k         |

Note: Policy 1 uses larger `chunk_size` (100) than Policy 2 (20) because coarse, generalised motions benefit from longer prediction horizons. Reduce to 50 only in Phase 3 when goal image makes the task more reactive.

---

## 12. Full Evaluation Protocol

Run this protocol after each phase to measure progress.

### 12.1 Individual component tests

**Pick success rate** — does the arm pick up the block correctly?

- 10 trials × 5 block positions = 50 trials total

**Hover success rate** — given a successful pick, does the arm reach the canonical hover pose within tolerance?

- Tier 1: Within ±0.5 cm, ±5° → target >90% (in-distribution)
- Tier 2: Within ±1 cm, ±10° → target >75%
- Tier 3: Within ±1.5 cm, ±15° → target >60% ← **Policy 2's minimum requirement**

**Location generalisation** — 10 trials per offset:

| Block offset from training position | Target after Phase 1 | Target after Phase 2 |
| ----------------------------------- | -------------------- | -------------------- |
| ±5 cm                               | >60%                 | >80%                 |
| ±10 cm                              | >40%                 | >70%                 |
| ±15 cm                              | >20%                 | >50%                 |
| ±20 cm                              | baseline             | >35%                 |

**Novel object generalisation** — 10 trials per object:

| Object type                   | Target after Phase 1 |
| ----------------------------- | -------------------- |
| Training object               | >80%                 |
| Novel object, similar shape   | >60%                 |
| Novel object, different shape | >40%                 |

**Novel slot position** — 10 trials per position:

- 5 trained positions: >70% each
- 2 untrained positions: >40% each

### 12.2 Reporting

Record all results in `instructions/data_and_models_log.md` under the corresponding model entry. Include:

- Phase number
- Number of training episodes
- Training steps
- Success rates per condition
- Observed failure modes

### 12.3 Decision gate

Only proceed to the full pipeline (Policy 1 → Policy 2) when Policy 1 achieves **Tier 3 handoff (±1.5 cm, ±15°) at >60%** on the evaluation conditions that match your deployment scenario.

---

## 13. Common Failure Modes and Fixes

| Failure                                              | Likely cause                            | Fix                                                      |
| ---------------------------------------------------- | --------------------------------------- | -------------------------------------------------------- |
| Arm goes to wrong area entirely                      | No location generalisation              | Phase 2 (spatial conditioning)                           |
| Grasp slip — correct approach, wrong grip            | Not enough object diversity             | Phase 1b — more objects                                  |
| Correct pick but wrong orientation at hover          | No orientation signal                   | Phase 3 (goal image) + spatial conditioning with angle   |
| Stale normalisation stats after merge                | Forgot to rerun `aggregate_stats()`     | Always rerun stats after every merge                     |
| Policy ignores spatial token                         | Detector output too noisy               | Add EMA smoothing, check detector with live feed         |
| Backbone gradient explosion during unfreeze          | LR too high for backbone                | Keep `optimizer_lr_backbone` at 1e-6, 100× below head LR |
| Good on fixed slot but fails on novel slot positions | Only trained on one slot position       | Phase 1b — collect across 5+ slot positions              |
| Performance degrades after augmentation              | Augmentation too aggressive             | Reduce brightness/contrast range first                   |
| Goal image conditioning makes things worse           | Goal images captured with arm in motion | Recapture goal images with arm stabilised                |

---

## 14. File Reference

| File                                                  | Purpose                                                                                              |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `src/lerobot/policies/act/configuration_act.py`       | ACT config dataclass — add `use_spatial_conditioning`, `use_goal_image`, `unfreeze_backbone_layers`  |
| `src/lerobot/policies/act/modeling_act.py`            | ACT model — add goal image encoder block, backbone unfreeze logic                                    |
| `src/lerobot/policies/act/processor_act.py`           | Inference processor — add `SpatialConditioningProcessorStep`                                         |
| `src/lerobot/datasets/transforms.py`                  | Image augmentation — extend `make_transform_from_config()` with `RandomResizedCrop`, `RandomErasing` |
| `src/lerobot/datasets/aggregate.py`                   | Dataset merging — `aggregate_datasets()`                                                             |
| `scripts/detect_block_slot.py`                        | OpenCV HSV detector + calibration tool (to be created)                                               |
| `instructions/hsv_ranges/`                            | HSV range JSON files per object/slot colour (to be created)                                          |
| `instructions/start_positions/insert_above_slot.json` | Canonical hover pose joint positions                                                                 |
| `instructions/data_and_models_log.md`                 | Record all datasets, models, and eval results here                                                   |

---

## 15. Next Step: Policy 2

Once Policy 1 achieves Tier 3 handoff at >60%, see `Policy2.md` for the precision insertion policy. Policy 2 starts from the canonical hover pose and completes the insertion. Its training distribution must cover the full output distribution of Policy 1's terminal states — whatever offsets and orientations Policy 1 produces at handoff must appear in Policy 2's training data.
