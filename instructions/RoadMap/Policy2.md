# Policy 2: Precision Insertion Policy — Instructions Manual

**Author:** LENS Lab
**Last updated:** 2026-04-07
**Hardware:** SO-101 (follower arm), gripper/wrist camera (index 7), top-down webcam (index 5)
**Dependency:** Policy 1 must be validated (Tier 3 handoff: ±1.5 cm, ±15°) before running the full pipeline. Policy 2 can be developed independently in parallel.

---

## 1. What Policy 2 Does

Policy 2 is the **precision insertion policy**. It takes over from Policy 1 at the canonical hover pose and completes the task: align the block precisely to the slot and descend until the block is fully seated.

```
Start state:   Arm at canonical hover pose — block in gripper ~4 cm above slot centre,
               roughly aligned (within ±1.5 cm, ±15° — Policy 1's Tier 3 handoff envelope).

End state:     Block fully seated in slot. Gripper releases. Task complete.

Success criterion:
               Block physically seated — no rocking, flush with slot surface.
               The wrist camera view matches the goal image (block-in-slot state).
```

Policy 2 operates entirely on **wrist camera feedback**. At 4 cm hover height, the wrist camera is looking almost directly at the slot opening. This is the only camera view that has enough resolution to resolve the millimetre-level misalignment that insertion requires. The top-down webcam is largely uninformative at this scale and should not be the primary observation for this policy.

---

## 2. Development Phases Overview

Policy 2 is built up in five phases. Unlike Policy 1 (which is primarily a data diversity problem), Policy 2's main challenges are:

1. Getting the right ACT configuration for reactive, precise control
2. Goal image conditioning — **required from Phase 1 onwards, not optional**
3. Enough shape and depth diversity for zero-shot transfer to novel geometries

| Phase   | What it adds                                                           | Effort                  | When to do it            |
| ------- | ---------------------------------------------------------------------- | ----------------------- | ------------------------ |
| Phase 0 | Baseline — establishes raw precision of vanilla ACT on wrist camera    | 1 day                   | Always first             |
| Phase 1 | Precision ACT config + goal image conditioning + wrist camera emphasis | 1–2 days code, ~100 eps | After Phase 0            |
| Phase 2 | Spatial conditioning with orientation angle                            | ~30 lines code, 1 week  | After Phase 1            |
| Phase 3 | Shape diversity (6–8 structurally different shapes)                    | 2–3 days collection     | After Phase 2            |
| Phase 4 | Slot depth variation (3+ depths)                                       | 1 day collection        | After Phase 3            |
| Phase 5 | Selective backbone unfreeze                                            | ~1 hour code change     | Only after 150+ episodes |

**Recommended execution order:** Do Phase 0 → Phase 1 → Phase 2 together as the foundational setup. Phase 3 and 4 are the main data collection effort. Phase 5 is optional but beneficial.

---

## 3. The Handoff Contract

Policy 2 is trained on start conditions that exactly mirror what Policy 1 delivers at handoff. This is the most critical design constraint.

**The rule:** Policy 2's training start-state distribution must cover the full output distribution of Policy 1's terminal states. If Policy 1 occasionally hands off with the block 15° rotated, that 15° case must appear in Policy 2's training data or Policy 2 will fail on it.

**Handoff envelope (defined by Policy 1's Tier 3 validation target):**

- Lateral offset: ±1.5 cm from slot centre (in x/y)
- Rotation: ±15° from correct insertion orientation
- Height: ±1 cm from canonical hover height (approximately 3–5 cm above slot)

Every Policy 2 training episode must start from within this envelope. Deliberately vary your start positions to span this entire range — do not always start from the exact canonical hover pose.

**Measuring Policy 1's actual terminal distribution:** Before collecting most Policy 2 data, measure where Policy 1 actually hands off by running 20 Policy 1 trials and recording the terminal arm pose (joint positions or end-effector xyz+angle) at each handoff. This gives you the empirical distribution. Policy 2's training starts should span at least that range, ideally slightly larger (by 20%).

---

## 4. Phase 0: Baseline Data Collection and Training

### 4.1 What to collect

Collect 50 demonstrations from a **fixed canonical hover pose**, inserting into a **fixed slot**, with a **single shape**. No variation in start position — perfect start conditions only.

This baseline tells you:

- How much of the insertion problem vanilla ACT already solves
- What the baseline offset and rotation tolerance are before any architectural changes
- Whether the wrist camera alone gives enough signal

### 4.2 Start position setup

Before each episode:

1. Run `go_to_start_position.py` to move the arm to the canonical hover pose
2. The block should already be in the gripper (placed by hand during reset)
3. Verify the block is aligned before starting — the baseline should test the insertion task, not the alignment task

```bash
python scripts/go_to_start_position.py --file instructions/start_positions/insert_above_slot.json
```

### 4.3 Recording command

Policy 2 uses the **gripper/wrist camera as primary** and top-down as secondary. Record both, but Policy 2 will train primarily on the wrist feed:

```bash
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

**Note:** Camera key is `wrist` not `gripper` for Policy 2 datasets. This is intentional — it keeps Policy 1 and Policy 2 dataset schemas distinct and avoids confusion when training.

**Episode time:** 20 seconds. A well-executed insertion from hover should take 3–8 seconds. Press `→` as soon as the block is fully seated.

**Reset:** Move arm back to canonical hover pose using `go_to_start_position.py`, then reposition the block by hand (either in the gripper, or fully remove and re-seat for the next episode start).

### 4.4 Phase 0 training

Use vanilla ACT with default config. The goal is to measure the raw baseline, not optimise it yet:

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

### 4.5 Phase 0 evaluation protocol

Run 10 trials per condition, starting from exact canonical hover pose (no offset):

| Condition                   | Trials | Metric              |
| --------------------------- | ------ | ------------------- |
| Canonical pose (±0 cm, ±0°) | 10     | Insertion success % |
| ±0.5 cm lateral offset      | 10     | Insertion success % |
| ±1 cm lateral offset        | 10     | Insertion success % |
| ±2 cm lateral offset        | 10     | Insertion success % |
| ±5° rotation                | 10     | Insertion success % |
| ±10° rotation               | 10     | Insertion success % |
| ±15° rotation               | 10     | Insertion success % |

Record actual success and classify failures:

- **(a) Jammed** — block hits slot edge and stops without seating → needs better lateral correction (Phase 2: spatial conditioning)
- **(b) Stops short** — block descends but stops before fully seated → slot depth issue or not enough descent signal (Phase 4: depth variation)
- **(c) Wrong trajectory** — erratic descent, doesn't track slot at all → wrist camera features insufficient without goal image (Phase 1: goal image conditioning)
- **(d) Works on canonical but fails on any offset** → insufficient start-state variation in training data (Phase 3/4 data diversity)

**Decision rule after Phase 0:** Regardless of results, proceed to Phase 1. The goal image and precision ACT config are not optional — they are the correct foundation for this policy. The Phase 0 baseline is purely to measure starting point, not to decide whether to apply Phase 1.

---

## 5. Phase 1: Precision ACT Config + Goal Image Conditioning + Wrist Camera Emphasis

This is the foundational setup for Policy 2. All three components go in together from Phase 1 onward.

### 5.1 Precision ACT configuration

Policy 2 requires a fundamentally different ACT config from Policy 1. The default ACT config is tuned for coarser pick-and-place tasks. For millimetre-precision insertion, every parameter needs to shift toward reactivity and conservatism.

**File:** `src/lerobot/policies/act/configuration_act.py`

```python
# --- Policy 2 precision config ---

# Smaller chunk size: predict 20 steps at a time instead of 100.
# Replanning more frequently lets the policy correct misalignment on the fly.
chunk_size: int = 20           # default: 100

# Replan every single step for maximum reactivity.
# At 30fps this means the policy re-evaluates the wrist camera every frame.
n_action_steps: int = 1        # default: 100

# Temporal ensembling: average recent predictions weighted by recency.
# This smooths out jitter from noisy wrist-camera observations.
# 0.1 means the most recent prediction gets 10× weight over the oldest.
temporal_ensemble_coeff: float = 0.1    # default: None (disabled)

# Higher KL weight: forces the policy's latent variable to encode
# more information about the current state (alignment + depth).
# Higher value → less noisy latents → more consistent fine trajectories.
kl_weight: float = 20.0        # default: 10.0

# Lower learning rate: precision tasks need conservative gradient updates.
# The policy is learning to stop mid-trajectory and correct; high LR causes overshooting.
optimizer_lr: float = 5e-5     # default: 1e-4
optimizer_lr_backbone: float = 5e-6    # default: 1e-5
```

Pass these at training time:

```bash
--policy.chunk_size=20 \
--policy.n_action_steps=1 \
--policy.temporal_ensemble_coeff=0.1 \
--policy.kl_weight=20.0 \
--policy.optimizer_lr=5e-5 \
--policy.optimizer_lr_backbone=5e-6
```

### 5.2 Wrist camera as primary observation

Configure `ACTConfig.image_features` to include only the wrist camera for Policy 2:

```python
# In ACTConfig — set this in the training command:
# observation.images.wrist  → primary (always included)
# observation.images.top    → omit for Policy 2 (or include as secondary with lower weight)
```

At 4 cm hover height, the top-down camera sees the arm and block from far away — it cannot resolve the millimetre-level slot alignment needed here. Using only the wrist camera keeps the observation space focused on the signal that matters.

In practice, specify this via `dataset.image_features` at training time to use only `observation.images.wrist`.

### 5.3 Goal image conditioning (required for Policy 2)

This is the most important mechanism for Policy 2 and is not optional. Without a goal image, the policy has no representation of what "inserted" looks like for a shape it has never seen.

**What the goal image shows:** The block fully seated in the slot, gripper open, block flush with the slot surface. The arm has been moved slightly out of frame so only the slot + fully-inserted block is visible. This image is captured at the end of every demo.

**Why it enables generalisation:** The policy learns "make the current wrist-camera view look like this goal image." For a novel shape, providing a goal image of the correct final state gives the policy everything it needs — it does not need to have memorised that geometry during training.

#### 5.3.1 Capturing goal images during data collection

At the end of every demo, before pressing `→` to save:

1. Complete the insertion (block fully seated)
2. Open the gripper
3. Move the arm slightly out of the camera frame (retract elbow slightly)
4. Press `g` (or the designated goal-capture key) to capture a clean wrist-camera frame as the goal image
5. Then press `→` to save the episode

This adds ~10–15 seconds per episode.

The goal image is saved as `observation.images.goal` in the dataset — a single image per episode, not a per-frame feature.

**What makes a good goal image:**

- Block is fully seated, flush with slot surface
- Arm is out of frame or minimally visible
- No motion blur — wait for the arm to fully stop
- Slot geometry clearly visible around the block
- Same lighting as the rest of the episode

#### 5.3.2 Architecture changes for goal image

These are the same changes described in Policy1.md Phase 3 — they apply to Policy 2 first (Policy 2 needs goal images; Policy 1 uses them for orientation).

**File:** `src/lerobot/policies/act/configuration_act.py`

Add to `ACTConfig`:

```python
use_goal_image: bool = False
goal_image_feature_key: str = "observation.images.goal"
use_shared_goal_backbone: bool = True
# Shared backbone: goal and current frames are embedded in the same feature space.
# The policy can then directly compute the visual delta (current → goal).
```

**File:** `src/lerobot/policies/act/modeling_act.py`

In `ACTPolicy.__init__`, after the main backbone setup:

```python
if self.config.use_goal_image:
    if self.config.use_shared_goal_backbone:
        self.goal_backbone = self.backbone  # shared weights
    else:
        self.goal_backbone = _make_backbone(config)  # separate (only if >300 episodes)

    self.encoder_goal_feat_input_proj = nn.Conv2d(
        backbone_out_channels, config.dim_model, kernel_size=1
    )
    self.encoder_goal_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(
        config.dim_model // 2
    )
```

In `ACTPolicy.forward()`, after the current-frame tokens are appended to `encoder_in_tokens`:

```python
if self.config.use_goal_image and self.config.goal_image_feature_key in batch:
    goal_img = batch[self.config.goal_image_feature_key]          # (B, C, H, W)
    goal_feat = self.goal_backbone(goal_img)                       # (B, C', H', W')
    goal_feat = self.encoder_goal_feat_input_proj(goal_feat)       # (B, dim_model, H', W')
    goal_pos = self.encoder_goal_cam_feat_pos_embed(goal_feat).flatten(2).permute(2, 0, 1)
    goal_feat = goal_feat.flatten(2).permute(2, 0, 1)
    encoder_in_tokens = torch.cat([encoder_in_tokens, goal_feat], dim=0)
    encoder_in_pos_embed = torch.cat([encoder_in_pos_embed, goal_pos], dim=0)
    # Decoder cross-attends to all encoder tokens — no decoder changes needed.
```

### 5.4 Phase 1 data collection

Collect 100 episodes with goal images, from the canonical hover pose:

```bash
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
    --dataset.push_to_hub=false \
    --display_data=true
```

**Important:** Slightly vary your starting position within the canonical envelope (±0.5 cm, ±5°) even for this initial batch. Pure fixed-pose training produces a brittle policy. Small natural variation from placing the block by hand is helpful — don't over-correct for it.

### 5.5 Phase 1 training

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

**Validation targets after Phase 1:**

- Tier 1 (±0 cm, ±0°): >90% insertion success
- Tier 2 (±1 cm lateral): >80%
- Tier 3 (±1.5 cm, ±10° rotation): >60%

If Tier 1 is not met, the policy is failing on its own training distribution. Check:

1. Goal images were captured correctly (arm out of frame, block fully seated)
2. Wrist camera is physically stable — even slight camera mount movement between episodes destroys the visual signal

---

## 6. Phase 2: Spatial Conditioning with Orientation Angle

**When to do this:** After Phase 1. Spatial conditioning with rotation angle is required for Policy 2 — it directly encodes the angular misalignment the policy must correct.

**Estimated effort:** 1 week, ~30 lines of code (builds on Policy 1's Phase 2 detector infrastructure).

### 6.1 What the spatial token encodes for Policy 2

For Policy 2, the wrist camera is looking downward at the slot opening from ~4 cm above. The 10-float spatial token encodes:

```
[cx_block_face, cy_block_face, w_block_face, h_block_face, angle_block,
 cx_slot,       cy_slot,       w_slot,       h_slot,       angle_slot]
```

All values normalised to [0, 1] (angles normalised to [-0.5, 0.5] representing [-90°, 90°]).

- `cx_slot, cy_slot` — where the slot centre is in the wrist camera frame. When these are both 0.5, the block is directly above the slot centre.
- `angle_block - angle_slot` — the angular misalignment the policy must correct before descending.

The policy learns: when `(cx_slot, cy_slot) ≈ (0.5, 0.5)` and `angle_block ≈ angle_slot`, descend. When misaligned, correct first. This generalises to novel geometries because it encodes the relationship, not a specific shape.

### 6.2 Detector for wrist camera

The wrist camera view is different from the top-down view. At hover height, the wrist camera looks directly at the slot opening from above.

**Detection approach: lighting-independent shape template matching**

The detector works purely on geometry, not colour. This is important because:

- Lighting changes (lab overhead vs. window light vs. different sessions) shift HSV values unpredictably.
- The block and slot are geometric shapes — their _edges_ are stable under any lighting.
- Phase 3 (shape diversity) requires the detector to work on novel shapes without re-calibration.

**Pipeline per frame:**

```
BGR → Grayscale → CLAHE (local contrast normalisation)
    → Gaussian blur → Canny edges (Otsu auto-threshold)
    → Morphological close (seal edge gaps)
    → findContours → matchShapes vs. saved templates
    → minAreaRect → (cx, cy, w, h, angle)
```

1. **CLAHE** normalises local contrast — the frame looks uniformly lit regardless of ambient conditions.
2. **Otsu auto-threshold** automatically selects the Canny threshold from the image histogram — no manual tuning, adapts to any lighting.
3. **Canny edges** respond to intensity _gradients_, not absolute brightness — robust to global lighting shifts.
4. **`cv2.matchShapes`** compares contour geometry using Hu moment invariants, which are invariant to scale, rotation, and lighting. A template captured once works across sessions.

**Calibration (one time per shape):**

Point the wrist camera at the target object (block face or slot opening). Run:

```bash
# Slot template:
python scripts/detect_block_slot.py --calibrate --target=slot --camera_index=7

# Block face template:
python scripts/detect_block_slot.py --calibrate --target=block --camera_index=7
```

Press **C** to capture the largest detected contour as the shape template. Press **S** to save.
No HSV sliders. The template is the contour itself — it generalises to any shape you point it at.

**Why this generalises to Phase 3 (novel shapes):**
When you introduce a new shape in Phase 3, run calibration once to capture that shape's contour template. The detector immediately works on the new shape — no re-tuning.

**Verify before running on dataset:**

```bash
python scripts/detect_block_slot.py --verify --camera_index=7
```

Green box = block face | Red box = slot. Check that angles are stable frame-to-frame.

### 6.3 ACT config changes

**File:** `src/lerobot/policies/act/configuration_act.py`

Policy 2 always uses `spatial_conditioning_dim=10` (includes rotation angles):

```python
use_spatial_conditioning: bool = True   # Always True for Policy 2
spatial_conditioning_dim: int = 10      # cx, cy, w, h, angle for both block and slot
```

### 6.4 Data processing and training

Run the wrist-camera detector offline on all existing Policy 2 training videos to add `observation.environment_state` (10-float keypoint vector) to each frame.

After adding the feature, recompute normalisation stats:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import compute_episode_stats, aggregate_stats

dataset = LeRobotDataset("rgragulraj/policy2_spatial",
                          root="~/.cache/huggingface/lerobot/rgragulraj/policy2_spatial")
compute_episode_stats(dataset)
aggregate_stats(dataset)
```

Training with both goal image and spatial conditioning:

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy2_spatial \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy2_spatial \
  --policy.chunk_size=20 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.1 \
  --policy.kl_weight=20.0 \
  --policy.optimizer_lr=5e-5 \
  --policy.optimizer_lr_backbone=5e-6 \
  --policy.use_goal_image=true \
  --policy.use_spatial_conditioning=true \
  --policy.spatial_conditioning_dim=10 \
  --policy.push_to_hub=false \
  --training.num_train_steps=50000 \
  --output_dir=outputs/policy2_phase2
```

**Validation:** Rerun Tier 1/2/3 tests. Also test with a deliberately misaligned start (±12° rotation). If the policy now corrects the misalignment before descending (visible in the wrist feed), spatial conditioning is working. If it still descends straight into the slot edge, the angular signal is not being used — check the detector output for noisy angle values.

---

## 7. Phase 3: Shape Diversity

**Goal:** Zero-shot insertion on novel shapes — shapes not seen in training.

**Theory:** Policy 2 cannot generalise to a shape it has never seen unless it has learned the underlying principle of insertion (align visual edges → descend until seated), rather than memorising shape-specific motions. The combination of goal image conditioning and spatial conditioning creates the conditions for this — but only if the training data spans enough structural diversity that the policy learns the principle rather than the specific geometries.

### 7.1 How many shapes and which ones

Collect **6–8 structurally diverse shapes**, approximately **20–25 episodes per shape**. Total: ~150–200 episodes.

Structural diversity means shapes that are geometrically distinct in insertion terms:

| Shape category         | Example                       | Why it's structurally distinct                                          |
| ---------------------- | ----------------------------- | ----------------------------------------------------------------------- |
| **Square/rectangular** | Square peg, rectangular block | Symmetric, 4 valid orientations (90° multiples)                         |
| **Round**              | Cylindrical peg               | No orientation constraint — tests whether policy learns to ignore angle |
| **Asymmetric/keyed**   | D-shaped peg, T-slot key      | Only 1–2 valid orientations — maximum orientation sensitivity           |
| **Triangular**         | Triangular prism              | 3-fold symmetry, unusual geometry                                       |
| **Star/cross**         | Cross-shaped peg              | Multi-lobed, requires precision on all axes simultaneously              |
| **Tall narrow**        | Thin rectangular peg          | Tests depth sensitivity — tips easily                                   |

Do not train on 20 slight variations of a square — that is not diversity. The goal is to teach the policy that "insertion" as a concept applies to many geometries. Novel shapes at test time should be structurally similar to at least one training category.

### 7.2 Slot diversity

For each shape, you need a slot that matches it. This requires having physical slot fixtures for each shape. If you cannot fabricate 6–8 slots, prioritise:

1. Square (already have it)
2. Round (easy to make — drilled hole)
3. Asymmetric/keyed (highest information content for orientation learning)
4. Triangular or rectangular (add one more structural class)

Minimum viable: 4 structurally distinct shape/slot pairs.

### 7.3 Recording per shape

For each shape, create a separate dataset and merge later. Keep dataset names consistent:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/policy2_shape_<shape_name> \
    --dataset.single_task="Insert the <shape_name> block into the slot" \
    --dataset.num_episodes=25 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=30 \
    --dataset.push_to_hub=false \
    --display_data=true
```

Replace `<shape_name>` with e.g. `square`, `round`, `dshape`, `triangle`, etc.

**For each shape's recording session:**

- Use the correct slot fixture for that shape
- Vary the start orientation within ±15° to cover the handoff envelope
- Vary lateral start position within ±1.5 cm
- **Capture a goal image at the end of every episode** (same as Phase 1)

### 7.4 Merging shape datasets

```python
from lerobot.datasets.aggregate import aggregate_datasets

datasets = [
    "rgragulraj/policy2_core",           # Phase 1 (square, canonical starts)
    "rgragulraj/policy2_shape_square",   # More square, varied starts
    "rgragulraj/policy2_shape_round",
    "rgragulraj/policy2_shape_dshape",
    "rgragulraj/policy2_shape_triangle",
    # ... add all shapes
]

aggregate_datasets(
    repo_ids=datasets,
    output_repo_id="rgragulraj/policy2_diverse",
    local_dir="~/.cache/huggingface/lerobot/rgragulraj/policy2_diverse",
)
```

Always recompute stats after merging (see Phase 1b in Policy1.md for the stats recomputation snippet).

### 7.5 Training with shape diversity

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
  --policy.use_spatial_conditioning=true \
  --policy.spatial_conditioning_dim=10 \
  --policy.push_to_hub=false \
  --training.num_train_steps=100000 \
  --output_dir=outputs/policy2_phase3
```

**Validation:** Test on a held-out shape not in the training set (select one shape and exclude it from the `policy2_diverse` dataset). Target: >50% insertion success on the novel shape with a goal image provided.

---

## 8. Phase 4: Slot Depth Variation

**Goal:** Policy 2 should descend until the block is seated regardless of how deep the slot is. Without depth variation in training, the policy learns a fixed-depth descent and fails on slots of different depths.

### 8.1 Why depth variation is necessary

If all training slots are 3 cm deep, the policy trains on trajectories that always descend exactly 3 cm. On a 6 cm deep slot, it will stop at 3 cm regardless of visual feedback — because visual feedback of "block partially inserted" was always the terminal state in training. The wrist camera sees "block partially inserted" and stops because that's what it saw at the end of every training episode.

With the goal image + visual feedback approach, the policy _can_ learn to descend until the goal image state is reached — but only if it has seen varying depths during training so it learns that "how far" is determined by the goal image, not a fixed step count.

### 8.2 Slot depths to cover

Collect demos with **at least 3 slot depths**:

| Depth           | Episodes per shape |
| --------------- | ------------------ |
| 2 cm (shallow)  | 8–10               |
| 4 cm (standard) | 10–12              |
| 6 cm (deep)     | 8–10               |

If fabricating physical slots of 3 depths per shape is infeasible, prioritise the shapes you have and cover 3 depths on the square/primary shape plus 2 depths on each other shape.

### 8.3 Recording for depth variation

Add a depth label to dataset names for clarity:

```bash
lerobot-record \
    ... \
    --dataset.repo_id=rgragulraj/policy2_depth_<depth_mm> \
    --dataset.single_task="Insert the block into the slot (depth <depth_mm>mm)" \
    --dataset.num_episodes=30 \
    ...
```

Example: `policy2_depth_20`, `policy2_depth_40`, `policy2_depth_60`.

Merge these into the main `policy2_diverse` dataset using `aggregate_datasets()` and retrain.

**Validation:** Test on a slot depth not seen in training (e.g. 5 cm if you trained on 2/4/6). The policy should descend until the block is seated, not stop at the nearest trained depth.

---

## 9. Phase 5: Selective Backbone Unfreeze (Optional)

**When:** After 150+ Policy 2 episodes. Backbone unfreeze is more justified for Policy 2 than for Policy 1 — the visual features needed for slot alignment (edge detection, shadow geometry, depth-from-defocus cues in the wrist camera) are very different from ImageNet features. ImageNet-trained ResNets have not been trained to detect the subtle visual cues of a block edge approaching a slot edge.

### 9.1 Code changes

Same changes as Policy 1 Phase 1c — add `unfreeze_backbone_layers` to `ACTConfig` and selectively unfreeze `["layer4"]` in `modeling_act.py`. See Policy1.md Section 7 for the exact code.

### 9.2 Training config

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
  --policy.optimizer_lr_backbone=5e-7 \
  --policy.use_goal_image=true \
  --policy.use_spatial_conditioning=true \
  --policy.spatial_conditioning_dim=10 \
  --policy.unfreeze_backbone_layers='["layer4"]' \
  --policy.push_to_hub=false \
  --training.num_train_steps=100000 \
  --output_dir=outputs/policy2_phase5
```

Note: `optimizer_lr_backbone=5e-7` — 100× lower than the head LR. This is intentionally more conservative than Policy 1 because Policy 2's backbone needs to learn fine-grained slot-edge features without forgetting its general edge detection capability.

---

## 10. Full Evaluation Protocol

### 10.1 Tier-based insertion success (per-shape)

Run 10 trials per tier per shape:

| Tier       | Start condition                 | Target |
| ---------- | ------------------------------- | ------ |
| **Tier 1** | Canonical hover (±0 cm, ±0°)    | >90%   |
| **Tier 2** | ±1 cm lateral offset            | >80%   |
| **Tier 3** | ±1.5 cm lateral + ±10° rotation | >60%   |

Tier 3 is the minimum Policy 2 must achieve for the full pipeline to work. It must work at the same offsets that Policy 1 hands off at.

### 10.2 Novel shape generalisation

Test on a held-out shape (never in training):

- Provide a goal image showing the novel shape fully seated
- Run 10 trials from canonical hover
- Target: >50% success (Tier 1 only for novel shapes)

If novel shape success is below 30%, the policy has memorised geometries rather than learning insertion principles. Increase shape diversity (add more structurally distinct shapes to training).

### 10.3 Novel slot depth generalisation

Test on a slot depth not in training:

- Run 10 trials
- Target: policy descends until seated, does not stop at a trained depth
- If it stops short: depth variation is insufficient (add more depth diversity to training)

### 10.4 Rotation tolerance heatmap

Run 10 trials per (offset × rotation) condition and plot as a heatmap:

```
           | 0°   | ±5°  | ±10° | ±15°
-----------+------+------+------+------
±0.0 cm   | [%]  | [%]  | [%]  | [%]
±0.5 cm   | [%]  | [%]  | [%]  | [%]
±1.0 cm   | [%]  | [%]  | [%]  | [%]
±1.5 cm   | [%]  | [%]  | [%]  | [%]
```

This heatmap directly shows the Policy 2 operating envelope and tells you precisely what Policy 1 must achieve at handoff. Record this in `instructions/data_and_models_log.md`.

### 10.5 Reporting

For each trained model, record in `instructions/data_and_models_log.md`:

- Phase number and dataset used
- Training steps and config parameters (chunk_size, kl_weight, etc.)
- Tier 1/2/3 success rates
- Novel shape success rate
- Observed failure modes

---

## 11. ACT Configuration Summary Per Phase

| Parameter                  | Phase 0 (vanilla ACT) | Phase 1+ (precision)                  | Phase 5 (+ unfreeze) |
| -------------------------- | --------------------- | ------------------------------------- | -------------------- |
| `chunk_size`               | 100 (default)         | **20**                                | 20                   |
| `n_action_steps`           | 100 (default)         | **1**                                 | 1                    |
| `temporal_ensemble_coeff`  | None                  | **0.1**                               | 0.1                  |
| `kl_weight`                | 10.0                  | **20.0**                              | 20.0                 |
| `optimizer_lr`             | 1e-4                  | **5e-5**                              | 5e-5                 |
| `optimizer_lr_backbone`    | 1e-5                  | **5e-6**                              | **5e-7**             |
| `use_goal_image`           | False                 | **True**                              | True                 |
| `use_spatial_conditioning` | False                 | False (Phase 1) / **True** (Phase 2+) | True                 |
| `spatial_conditioning_dim` | —                     | — / **10**                            | 10                   |
| `unfreeze_backbone_layers` | `[]`                  | `[]`                                  | **`["layer4"]`**     |
| Primary camera             | wrist                 | wrist                                 | wrist                |
| Training steps             | 50k                   | 50k                                   | 100k                 |

**Key difference from Policy 1:** Policy 2 uses `n_action_steps=1` (replan every frame) vs. Policy 1's larger chunk horizon. Insertion requires the policy to stop and correct mid-trajectory — this is impossible with a 100-step action chunk that cannot be interrupted.

---

## 12. Wrist Camera Setup and Calibration

The wrist camera mount is the most mechanically critical component of Policy 2. Even a 2mm shift in the camera mount between training and deployment will shift every pixel of the wrist view, causing systematic insertion failure.

**Rules:**

1. The wrist camera must be mounted identically for every recording session and every deployment. Use a mechanical fixture, not tape.
2. After any camera remounting, collect 10 new episodes from canonical hover and verify Tier 1 success before trusting the policy.
3. Do not try to augment away camera mounting inconsistency with `RandomAffine` — this creates training-deployment distribution shift. Fix the mount mechanically.
4. Record the camera mount position/angle in `instructions/data_and_models_log.md` for each dataset batch.

**Verifying camera alignment before each session:**

```bash
# Quick sanity check: does the wrist view look the same as during training?
python scripts/go_to_start_position.py --file instructions/start_positions/insert_above_slot.json
# Then capture a frame and visually compare to a reference frame from training
```

---

## 13. Common Failure Modes and Fixes

| Failure                                                       | Likely cause                                                           | Fix                                                                                   |
| ------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Block jams into slot edge on every attempt                    | Not enough lateral correction before descent                           | Add spatial conditioning (Phase 2), vary lateral start positions in training          |
| Stops before fully seated                                     | Policy learned fixed descent depth                                     | Phase 4 — add slot depth variation                                                    |
| Works on training shape but fails completely on novel shape   | Policy memorised geometry, not insertion principle                     | Phase 3 — add more structurally diverse shapes                                        |
| Erratic trajectory on wrist camera                            | chunk_size too large, no temporal smoothing                            | Phase 1 — set chunk_size=20, n_action_steps=1, temporal_ensemble_coeff=0.1            |
| Tier 1 good, Tier 3 fails completely                          | Start-state variation not in training data                             | Deliberately collect varied-start episodes spanning ±1.5 cm, ±15°                     |
| Rotates block to correct orientation but descends at an angle | Angle in spatial token is noisy / not detected cleanly                 | Check detector output with live feed, add EMA smoothing                               |
| Good in lab, fails when connected to Policy 1                 | Policy 1's terminal distribution wider than Policy 2's training starts | Measure Policy 1's terminal distribution, add more Policy 2 data at those extremes    |
| Goal image conditioning makes things worse                    | Goal images captured while arm was moving                              | Recapture goal images with arm fully stopped, block fully seated, arm out of frame    |
| Wrist camera features insufficient for depth estimation       | All training slots same depth                                          | Phase 4 — add 3+ slot depths                                                          |
| Backbone gradient explosion during Phase 5                    | `optimizer_lr_backbone` too high                                       | Keep at 5e-7 — 100× below head LR                                                     |
| Works on shallow slots, fails on deep ones                    | Depth variation only at bottom of depth range                          | Include depths both shallower and deeper than deployment target                       |
| Insertion fails when lighting changes                         | No augmentation                                                        | Enable brightness/contrast augmentation in transforms.py (moderate range: (0.7, 1.3)) |

---

## 14. File Reference

| File                                                  | Purpose                                                                                                                      |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `src/lerobot/policies/act/configuration_act.py`       | Add `use_goal_image`, `use_spatial_conditioning`, `unfreeze_backbone_layers`; set chunk_size/kl_weight defaults for Policy 2 |
| `src/lerobot/policies/act/modeling_act.py`            | Add goal image encoder block; backbone unfreeze logic                                                                        |
| `src/lerobot/policies/act/processor_act.py`           | Add `SpatialConditioningProcessorStep` for wrist camera detector                                                             |
| `src/lerobot/datasets/transforms.py`                  | Enable augmentation (moderate ranges for wrist-camera view)                                                                  |
| `scripts/detect_block_slot.py`                        | OpenCV wrist-view slot detector (calibrate on camera index 7)                                                                |
| `instructions/hsv_ranges/wrist_slot.json`             | HSV range for slot detection from wrist camera                                                                               |
| `instructions/hsv_ranges/wrist_block_face.json`       | HSV range for block-face detection from wrist camera                                                                         |
| `instructions/start_positions/insert_above_slot.json` | Canonical hover pose (Policy 2's start position)                                                                             |
| `instructions/data_and_models_log.md`                 | Record all datasets, models, and eval results here                                                                           |

---

## 15. After Policy 2 Is Validated

Once Policy 2 achieves Tier 3 success at >60%, it is ready to receive handoffs from Policy 1. The next steps are:

1. **Measure Policy 1's actual terminal distribution** — run 20 Policy 1 trials and record exact terminal arm poses. Verify they fall within Policy 2's validated operating envelope.
2. **Implement the handoff trigger** — start with Option A (fixed waypoint): Policy 1 runs until the arm reaches the canonical hover pose (detected by checking Z-height threshold or joint position match). Then hand control to Policy 2.
3. **Run the full pipeline** — 20 trials per condition. Classify failures as Policy 1 failures (didn't reach handoff) vs Policy 2 failures (reached handoff but failed insertion).

See `generalisation_roadmap.md` Section "The Handoff Problem" for the handoff implementation options.
