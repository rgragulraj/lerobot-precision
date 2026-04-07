# Precision + Generalisation Roadmap: Two-Policy Block Insertion

## Core Architecture: Why Two Policies

A single policy cannot simultaneously be general and precise. These goals impose conflicting requirements:

- **Generalisation** requires training data that spans many objects, locations, and orientations. The policy learns coarse, robust mappings from visual observations to movement.
- **Precision** requires dense training data covering a narrow, well-defined sub-task with consistent start conditions. The policy learns fine-grained corrections in a tight distribution.

Asking one policy to pick up a novel object from an arbitrary location AND precisely insert it into a tight slot is asking it to solve both problems at once. It will compromise on both.

**The split:**

|                   | Policy 1 (Coarse)                                           | Policy 2 (Precision)                                                                                |
| ----------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Task**          | Pick block → orient → hover above slot                      | Hover position → insert into slot                                                                   |
| **Start state**   | Arm at home, block anywhere in workspace                    | Block held ~3–5 cm above slot, roughly aligned                                                      |
| **End state**     | Block approximately above slot, roughly correct orientation | Block fully seated in slot                                                                          |
| **Key challenge** | Generalisation to novel objects, locations, slot positions  | Precision AND generalisation: millimetre-level alignment on shapes/slots never seen in training     |
| **Data needed**   | Many episodes, high diversity                               | 100–200 episodes, structurally diverse shapes, varied slot depths                                   |
| **Architecture**  | ACT with spatial + goal conditioning (phases below)         | ACT with goal image conditioning (required), spatial conditioning with angle, wrist camera emphasis |

---

## The Handoff Problem

The hardest design question is: **when and how does Policy 1 hand off to Policy 2?**

Policy 2's training distribution must cover the full output distribution of Policy 1's terminal states. If Policy 1 sometimes leaves the block 6 cm above the slot and sometimes 2 cm, your Policy 2 training data must span that range.

**Option A — Fixed waypoint (recommended to start):**
Policy 1's goal is always to bring the block to a canonical hover pose: a fixed position and orientation directly above the slot centre, at a fixed height (e.g. 4 cm above). Policy 2 always starts from this canonical pose. Clean separation, easy to validate independently.

- Pro: You can train and test the policies completely independently.
- Pro: Policy 2's training distribution is narrow and well-defined.
- Con: Policy 1 must learn to hit a precise waypoint, which is itself a precision task. Mitigate by defining the waypoint loosely (±1.5 cm tolerance) and training Policy 2 to handle that range.

**Option B — Wrist-camera trigger:**
Policy 2 activates when the wrist camera sees the slot centred and fills more than X% of the frame. Requires a simple classifier or OpenCV detector on the wrist feed.

**Option C — Z-height threshold:**
Transition when end-effector Z is within N cm of the slot surface, based on robot state. Simple to implement.

Start with Option A. It lets you develop and validate the two policies independently before worrying about the interface.

---

## Data Collection Strategy

### Policy 1 data

- Collect diverse demonstrations: block anywhere in the 3×3 workspace grid, multiple objects, multiple slot positions
- Each episode ends when the arm reaches the canonical hover pose above the slot — not when the block is inserted
- Focus: breadth over depth. 300–500 episodes across diverse conditions.
- Camera: top-down camera is the primary view for location generalisation

### Policy 2 data

- Each episode starts from the canonical hover pose (or within the handoff tolerance envelope)
- Vary starting conditions systematically: ±1.5 cm lateral offset from slot centre, ±15° rotation, ±1 cm height
- Each episode ends when the block is fully seated
- Focus: high quality and structural diversity. 100–200 episodes across ~6–8 structurally diverse shapes.
- Camera: **wrist camera is the primary view**. The wrist camera sees the slot geometry directly and provides the fine-grained visual feedback needed for precise alignment. The top-down camera is largely uninformative at this scale.

**Shape diversity for generalisation:** Don't train Policy 2 on one shape — don't train it on 50 shapes either. You need ~6–8 shapes that are _structurally diverse_: round, square, rectangular, keyed/asymmetric. The goal is for the policy to learn the _principle_ of insertion (align → descend → seat) rather than memorising shape-specific motions. Zero-shot transfer to a novel shape works if it's structurally similar to something in training. Collecting 20–25 demos per shape covers this.

**Slot depth variation:** Include slots of different depths (e.g. 2 cm, 4 cm, 6 cm) in training. The policy uses wrist-camera visual feedback and will naturally descend until the seated state is reached — but only if it has seen varying depths during training. If all training slots are the same depth, it will stop at that depth regardless.

**Critical:** Policy 2's training data must span the full handoff envelope. If Policy 1 occasionally leaves the block 15° rotated, that 15° case must appear in Policy 2's training data or it will fail on it.

---

## Phase 0: Establish Baselines for Both Policies (2 days, no code changes)

**Do not skip this. It will save weeks of misdirected effort.**

**Policy 1 baseline:**
Collect 50 episodes with fixed object, fixed block position, fixed slot position. Train for 80k steps. Measure:

1. **Location tolerance**: shift block/slot by ±5, ±10, ±15 cm. Plot reach-and-hover success rate at each offset.
2. **Novel object**: swap to 3 visually similar + 3 visually different objects. Record pick success and hover-above-slot success separately.
3. **Novel slot position**: move the slot to 5 different positions. Record hover success.

At each failure, classify the failure mode:

- (a) Wrong approach — arm goes to wrong area entirely → location generalisation is the bottleneck → prioritise Phase 2
- (b) Grasp slip — correct approach, wrong grip → more object diversity data + augmentation
- (c) Wrong orientation on hover — picked up correctly but wrong orientation when arriving above slot → Phase 3 (goal image)

**Policy 2 baseline:**
Collect 50 episodes from a fixed canonical hover pose. Train for 50k steps. Measure:

1. **Offset tolerance**: start at ±0.5, ±1, ±2 cm from centre. Plot insertion success rate.
2. **Rotation tolerance**: start at ±5°, ±10°, ±15°. Plot insertion success rate.
3. **Novel slot geometry**: test on 2–3 slot sizes/shapes.

This baseline tells you how much of the handoff envelope Policy 2 already handles. If it fails beyond ±0.5 cm, you need more diverse Policy 2 training data before worrying about Policy 1's handoff accuracy.

---

## Policy 2: Precision Configuration

Before addressing generalisation (Policy 1), get Policy 2 right. A Policy 2 that reliably inserts within ±1.5 cm is more valuable than a Policy 1 that generalises perfectly but hands off to a brittle inserter.

### ACT config for precision

```python
# Smaller chunk size — predict fewer steps at once, replan more frequently
chunk_size: int = 20          # vs default 100; more frequent replanning aids precision
n_action_steps: int = 1       # replan every step for maximum reactivity

# Temporal ensembling — average recent predictions to smooth jitter
temporal_ensemble_coeff: float = 0.1

# Higher KL weight — forces the latent to be more informative
kl_weight: float = 20.0       # vs default 10.0

# Lower LR — precision tasks benefit from slower, more conservative updates
optimizer_lr: float = 5e-6
optimizer_lr_backbone: float = 5e-7
```

### Wrist camera emphasis

If you have both a top-down camera and a wrist camera, train Policy 2 on **wrist camera only** (or wrist-primary with top-down as secondary). The wrist camera sees the slot directly at the scale where precision matters. Top-down at 4 cm hover height provides almost no useful signal for millimetre-level alignment.

In your dataset, this means recording with the wrist camera feed as `observation.images.wrist` and configuring `image_features` in `ACTConfig` to use only that camera for Policy 2.

### Selective backbone unfreeze for Policy 2

For Policy 2, unfreezing `layer4` of the ResNet backbone (see 1c below) is more justified than for Policy 1 — the visual features needed for slot alignment (edge detection, depth cues, shadow geometry) are very different from ImageNet features. Do this after you have 150+ Policy 2 episodes.

```python
# In ACTConfig for Policy 2
optimizer_lr_backbone: float = 5e-7  # 10× lower than head LR
```

Add `unfreeze_backbone_layers: list[str] = field(default_factory=list)` to `ACTConfig` and unfreeze `["layer4"]` only. Keep `FrozenBatchNorm2d` — do not update BN stats.

### Validation protocol for Policy 2

Define success tiers:

- **Tier 1:** Successful insertion from canonical hover (±0 cm, ±0°) — target: >90%
- **Tier 2:** Insertion from ±1 cm lateral offset — target: >80%
- **Tier 3:** Insertion from ±1.5 cm lateral + ±10° rotation — target: >60%

Tier 3 defines the maximum handoff tolerance Policy 1 must achieve.

---

## Policy 2: Achieving Generalisation

Pure behavioral cloning cannot generalise to novel shapes on its own. If Policy 2 has never seen a shape, it has no information about how to align it — more demos of other shapes don't help unless the policy learns the underlying _principle_ of insertion. Three things together solve this.

### The mental model

Policy 2 should not be a specialist memorising each shape. It should be:

> A visually-guided insertion policy conditioned on (a) a goal image showing the target seated state and (b) the spatial relationship — including rotation angle — between block and slot. It generalises because it is told what "done" looks like and where things are, not because it has memorised every geometry.

### Goal image conditioning (required, not optional)

This is the most important mechanism for novel-shape generalisation and is **not optional** for Policy 2. Without a goal image, the policy has no information about the target geometry for a shape it has never seen.

A goal image showing the block correctly seated in the slot encodes the target geometry implicitly. The policy learns "make the current wrist-camera view look like this goal view." For a novel shape, as long as the goal image shows the correct final state, the policy has everything it needs.

**For Policy 2 data collection:** after each demo, move the arm slightly out of frame and capture a clean goal image of the fully-seated block. This adds ~10 seconds per episode. See Phase 3 (goal image conditioning) for the architecture implementation — those changes apply to Policy 2 first.

### Spatial conditioning with orientation angle (required)

The 8-float keypoint vector from Phase 2 must include the in-plane rotation angles:

```
[cx_block, cy_block, w_block, h_block, angle_block, cx_slot, cy_slot, w_slot, h_slot, angle_slot]
```

Set `spatial_conditioning_dim: int = 10` for Policy 2. The policy learns to correct for the delta between `angle_block` and `angle_slot` — this generalises to novel orientations because it encodes the _relationship_, not a specific angle. Use `cv2.minAreaRect()` in the detector (already handles this).

### Slot height and depth

Slot height (how tall the slot walls are, how deep the block must descend) is handled naturally by visual feedback — the wrist camera shows when the block is seated. The policy descends until the goal image state is reached rather than executing a fixed-depth trajectory.

What you must do is **include slot depth variation in training data**. Collect demos with slots of at least 3 different depths (e.g. 2 cm, 4 cm, 6 cm). If all training slots are 3 cm deep, the policy stops at 3 cm on a 6 cm slot.

### Angled slots (deferred — requires hardware)

A slot angled in 3D (e.g. tilted 30° toward the robot) means the insertion axis is not vertical. The 2D keypoint detector handles in-plane rotation (block/slot twist around Z-axis) but cannot estimate a slot's 3D surface normal.

Solving fully angled slots requires one of:

- A depth camera or stereo rig to estimate 3D slot pose
- ArUco markers on the slot face (gives full 6-DOF pose cheaply with `cv2.aruco`)

**Do not attempt angled slots with the current top-down webcam setup.** Cover in-plane rotation (the `angle` field above) first. Revisit when a depth sensor is added to the hardware stack.

---

## Policy 1 Generalisation: Phased Roadmap

The goal of Policy 1 is to reliably deliver the block to within Tier 3 handoff tolerance. The phases below address the three ways Policy 1 fails on novel scenes.

### Phase 1: Data-Centric Hardening (1–2 weeks, minimal code changes)

**This has the highest ROI. Do not invest engineering time in Phases 2–4 until this is saturated.**

#### 1a. Enable Aggressive Image Augmentation

**File:** `src/lerobot/datasets/transforms.py` + training config

The `ImageTransformsConfig` has augmentation disabled by default (`enable: False`). Enable it:

| Transform        | Current default | Recommended |
| ---------------- | --------------- | ----------- |
| Brightness       | (0.8, 1.2)      | (0.5, 1.5)  |
| Contrast         | (0.8, 1.2)      | (0.5, 2.0)  |
| Saturation       | (0.5, 1.5)      | (0.5, 1.5)  |
| Hue              | (-0.05, 0.05)   | (-0.1, 0.1) |
| Affine translate | (0.05, 0.05)    | (0.1, 0.1)  |

`affine translate=(0.1, 0.1)` corresponds to ~5–8 cm of workspace shift at 640px — the cheapest location generalisation available.

Also add:

- `RandomResizedCrop(scale=(0.85, 1.0))` — simulates small zoom/camera-position changes
- `RandomErasing(p=0.3, scale=(0.02, 0.1))` — partial occlusion robustness

Both require adding a new `elif` branch in `make_transform_from_config()` in `transforms.py`.

#### 1b. Systematic Data Variation

Collect more diverse Policy 1 demonstrations:

- Divide the workspace into a **3×3 grid**. Collect 15–20 episodes per cell → ~300 total episodes
- Collect **40 episodes per object** across 5 different objects (vary shape, colour, size)
- Collect **40 episodes per slot position** across at least 5 positions in the workspace
- **Vary lighting** between sessions (overhead on, overhead off + side lamp, afternoon window light)
- **Vary camera tilt ±5 degrees** between sessions

Merge datasets using `aggregate_datasets()` in `src/lerobot/datasets/aggregate.py`.

> **Critical:** After merging, always rerun `compute_episode_stats()` + `aggregate_stats()` on the merged dataset and verify the new stats file exists before training. Stale normalisation stats are a frequent silent failure mode.

#### 1c. Selective Backbone Unfreeze (Optional, needs >500 Policy 1 episodes)

**File:** `src/lerobot/policies/act/configuration_act.py`, `modeling_act.py`

Add `unfreeze_backbone_layers: list[str] = field(default_factory=list)` to `ACTConfig`. Unfreeze `layer4` only. Set `optimizer_lr_backbone` to `1e-6` (10× lower than head LR). This fine-tunes the final feature extractor toward manipulation-relevant features without destabilising batch norm statistics.

Only attempt with >500 episodes — unfrozen BN with small datasets tends to overfit visual features to your specific objects.

**Validation:** Rerun Phase 0 protocol after 1a+1b. If hover-above-slot success at ±10 cm goes from ~20% to ~60%+, the data-centric approach has solved the problem and Phases 2–4 can wait.

---

### Phase 2: Spatial Conditioning via Keypoints (1–2 weeks, ~50 lines of code)

**Target:** Give Policy 1 an explicit spatial token telling it where the block and slot are. This addresses location generalisation more directly than augmentation alone.

#### 2a. Shape Detection Script (OpenCV)

Rather than physical markers (AprilTags) or large ML models (GroundingDINO), a simple OpenCV shape detector is the right first approach. The block and slot are known geometric shapes under controlled lab lighting — contour matching is fast, requires no training, and works directly on the actual geometry.

**What it produces:** `[cx_obj, cy_obj, w_obj, h_obj, cx_slot, cy_slot, w_slot, h_slot]` — 8 floats, normalised to [0, 1] in image space.

```python
import cv2
import numpy as np

def detect_block_and_slot(frame_bgr, block_hsv_range, slot_hsv_range):
    """
    Returns (block_bbox, slot_bbox) as (cx, cy, w, h) normalised to [0,1],
    or None for each if not detected.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, w = frame_bgr.shape[:2]

    def find_largest_contour_bbox(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 200:  # noise threshold in pixels
            return None
        x, y, bw, bh = cv2.boundingRect(c)
        return (x + bw/2) / w, (y + bh/2) / h, bw / w, bh / h

    block_mask = cv2.inRange(hsv, *block_hsv_range)
    slot_mask  = cv2.inRange(hsv, *slot_hsv_range)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    block_mask = cv2.morphologyEx(block_mask, cv2.MORPH_OPEN, kernel)
    slot_mask  = cv2.morphologyEx(slot_mask,  cv2.MORPH_OPEN, kernel)

    return find_largest_contour_bbox(block_mask), find_largest_contour_bbox(slot_mask)
```

**Calibration per object/slot:** Write a small calibration script that shows a live camera feed with HSV sliders (`cv2.createTrackbar`) to dial in the correct range for each colour under your lab lighting. Save ranges as JSON. Takes 5–10 minutes per new object.

**Handling orientation:** `cv2.minAreaRect()` gives the rotation angle of the detected contour. For insertion tasks where block orientation matters, extend the output to 10 floats: `[cx_obj, cy_obj, w_obj, h_obj, angle_obj, cx_slot, cy_slot, w_slot, h_slot, angle_slot]` and update `spatial_conditioning_dim` accordingly.

**Fallback for novel objects:** Fall back to a zero vector or last valid detection (via EMA smoothing). Policy 1 should learn to rely on visual features when the spatial token is uninformative.

**Upgrade path:** Once validated, switching to GroundingDINO (text-prompted, no per-object calibration) is straightforward — it produces the same 8-float output, so only the `SpatialConditioningProcessorStep` needs updating.

#### 2b. Inject Spatial Conditioning via the Existing `env_state_feature` Path

No changes to the Transformer architecture are needed. The existing `env_state_feature` path in `modeling_act.py` already injects a flat vector as a token into the encoder sequence.

**Changes needed:**

`configuration_act.py` — add:

```python
use_spatial_conditioning: bool = False
spatial_conditioning_dim: int = 8  # 8 for (cx,cy,w,h) x2; 10 if including rotation angle
```

In `validate_features()`: when `use_spatial_conditioning=True`, require that input features include `observation.environment_state` with the matching shape.

`processor_act.py` — add a `SpatialConditioningProcessorStep` that:

1. Runs the detector (or loads a pre-cached result) at inference time
2. Applies an exponential moving average over 3 frames to smooth noisy detections
3. Injects the keypoint vector into the batch dict as `observation.environment_state`

**Data pipeline for training:**

1. Run detector offline on all Policy 1 training videos (batch process)
2. Use `add_features()` from `dataset_tools.py` to add the keypoint vector as a new float feature
3. Recompute normalisation stats to cover the new feature

**Validation:** Rerun Phase 0 location protocol. If hover-above-slot tolerance expands from ±10 cm to ±20 cm+, spatial conditioning is working. If not, add a hook on `ACTEncoderLayer.self_attn` and check whether the spatial token has high attention weights.

---

### Phase 3: Goal-Image Conditioning (2–3 weeks, ~80 lines of code)

**Target:** Novel slot geometry generalisation — show Policy 1 what the final state looks like (block correctly oriented above slot) so it can plan approach and orientation without having seen that slot before.

#### 3a. Collecting Goal Images

For each Policy 1 training episode, you need a "goal image" showing the desired hover state. Options:

1. **Canonical hover frame** (recommended) — after the robot reaches the canonical hover pose, capture a clean image with the arm still (block visible in gripper, slot visible below). This directly shows the target state for Policy 1.
2. **Last-frame goals** — use the final frame of each Policy 2 episode (block fully inserted). Provides a stronger goal signal but further from Policy 1's decision horizon.
3. **Last frame raw** — use the final frame as-is. Works as a starting point but the model may attend to arm pose rather than object/slot relationship.

Start with option 1. It creates a clean supervised signal: "get the scene to look like this."

#### 3b. Architecture Changes

**Files:** `src/lerobot/policies/act/modeling_act.py`, `configuration_act.py`

```python
use_goal_image: bool = False
goal_image_feature_key: str = "observation.images.goal"
use_shared_goal_backbone: bool = True  # share weights with main backbone initially
```

**Model additions:**

- When `use_shared_goal_backbone=True`: `self.goal_backbone = self.backbone` — ties weights so both branches embed current and goal frames in the same feature space, which helps the policy compute "what is the delta from here to goal"
- Add `self.encoder_goal_feat_input_proj = nn.Conv2d(backbone_out_channels, config.dim_model, kernel_size=1)`
- Add `self.encoder_goal_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)`
- In `forward()`: after adding current camera tokens, add a symmetric block that processes the goal image through the goal backbone and appends its tokens to `encoder_in_tokens` + `encoder_in_pos_embed`. The decoder already cross-attends to all encoder tokens — no decoder changes needed.

**Validation:** Test on 3 novel slot geometries (with a goal image at inference). If this improves over Phase 2 alone, goal conditioning is contributing to orientation handling.

---

### Phase 4: Language Conditioning (3–4 weeks, ~150 lines of code)

**Target:** Full task specification via natural language — "pick up the blue cylinder and insert into the square slot." This enables the multi-policy routing described in the project overview: a language command selects which Policy 1 specialisation to run, then hands off to Policy 2.

#### 4a. Text Encoder

**Recommended: CLIP ViT-B/32 text encoder, frozen.**

- 512-dim text embeddings
- Pre-cache all embeddings offline — zero inference overhead at evaluation time
- Good at associating visual-semantic descriptions ("red cube", "square slot")
- Lightweight (~150 MB)

#### 4b. Architecture Changes

**Files:** `modeling_act.py`, `configuration_act.py`, `processor_act.py`

```python
use_language_conditioning: bool = False
language_dim: int = 512
freeze_language_encoder: bool = True
```

**Model changes:**

- `self.language_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")`, frozen
- `self.encoder_language_input_proj = nn.Linear(config.language_dim, config.dim_model)`
- Increment `n_1d_tokens` by 1 in `encoder_1d_feature_pos_embed` for the pooled language token
- In `forward()`: project language embedding and prepend to `encoder_in_tokens` after the latent token

**Processor addition:** `LanguageConditioningProcessorStep` in `processor_act.py` — takes a string, tokenizes, looks up pre-cached embedding, injects as `observation.language`.

**Data labeling:** For each Policy 1 episode, write a task description. Keep it short and consistent:

- "pick up the [colour] [shape] and move above the [shape] slot"
- ~5 seconds per episode × 300 episodes = ~25 minutes total

---

## Recommended Execution Order

```
Policy 2 first:
  Collect 100 Policy 2 episodes (canonical hover → insert)
  Train Policy 2, validate Tier 1/2/3 tolerance
  → Defines the handoff envelope Policy 1 must achieve

Then Policy 1:
  Phase 0  →  Phase 1a  →  Phase 1b  →  evaluate
                                            │
                ┌───────────────────────────┼───────────────────────────┐
                ▼                           ▼                           ▼
         Location still             Object appearance             Slot geometry /
         failing (wrong area)       failing (grasp slip)          orientation failing
                │                           │                           │
         Phase 2                     Phase 1c +                   Phase 3
         (keypoints)                 diverse data                  (goal image)
                │
         Phase 4 (language) — last, subsumes all, enables multi-task routing
```

**Run both policies together only after each is validated independently.** The most common mistake is debugging both simultaneously when they have separate failure modes.

---

## Training Budgets (Approximate)

| Policy / Phase                         | Steps | GPU Time  |
| -------------------------------------- | ----- | --------- |
| Policy 2 (precision baseline)          | 50k   | ~2 hours  |
| Policy 1 Phase 1 (augmentation + data) | 100k  | ~4 hours  |
| Policy 1 Phase 2 (keypoints)           | 80k   | ~3 hours  |
| Policy 1 Phase 3 (goal image)          | 150k  | ~7 hours  |
| Policy 1 Phase 4 (language)            | 200k  | ~10 hours |

---

## Evaluation Protocol

**Policy 2:**

- 10 trials per offset condition (0, ±0.5, ±1, ±1.5 cm) × 3 rotation conditions (0°, ±10°, ±15°)
- Report: insertion success rate per condition, plot as heatmap (offset × rotation)
- Also test on 2–3 novel slot geometries

**Policy 1:**

- 10 trials per location condition (±5, ±10, ±15, ±20 cm from training centre)
- 10 trials per novel object (3 similar + 3 different)
- Report separately: pick success, hover-above-slot success, handoff-within-tolerance success
- Plot: success rate vs. workspace offset as a curve

**Full pipeline (Policy 1 → Policy 2):**

- Run only after both policies pass their individual evaluations
- 20 trials per condition; report full-task success (block fully seated)
- Failures: classify as Policy 1 failure (didn't reach handoff) or Policy 2 failure (reached handoff but failed insertion)

---

## Common Failure Modes to Watch For

1. **Stale normalisation stats** after dataset merges — always run `aggregate_stats()` after `aggregate_datasets()` and verify the stats file before training
2. **Handoff envelope mismatch** — Policy 2 was trained on ±0.5 cm offsets but Policy 1 hands off at ±2 cm. Always measure Policy 1's actual terminal state distribution and ensure Policy 2's training data covers it.
3. **Policy 2 overfitting to a single orientation** — if all Policy 2 demos start with the block perfectly aligned, it will fail when Policy 1 leaves it rotated. Deliberately collect varied-orientation Policy 2 starts.
4. **Backbone gradient explosion** during unfreeze — keep `optimizer_lr_backbone` at least 10× lower than head LR
5. **Goal image leakage** — ensure the goal image comes from after the arm has reached the hover pose cleanly, otherwise the model learns to copy the arm pose rather than the object/slot relationship
6. **Detector noise at inference** — keypoint detectors are noisier at inference than in offline-processed training data; smooth with exponential moving average over 3 frames in the processor
7. **Wrist camera miscalibration** — if the wrist camera is even slightly misaligned between Policy 2 training and deployment, insertion success drops sharply. Fix the camera mount mechanically; don't try to augment your way out of it.
8. **Policy 2 trained on one shape only** — a Policy 2 with no shape diversity will fail on novel geometries regardless of how good the goal image is. You need ~6–8 structurally diverse shapes in training for zero-shot transfer to work.
9. **Slot depth not varied in Policy 2 training** — a policy trained on a single slot depth will execute a fixed-depth descent and either stop short or jam on slots of different depths. Always include at least 3 depths in training.
10. **Attempting angled slots without a depth sensor** — in-plane rotation (Z-axis twist) is tractable with the 2D keypoint detector. Full 3D slot tilt is not. Don't conflate these two problems; solve rotation first.
