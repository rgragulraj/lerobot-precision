# Generalising ACT to Novel Objects, Slots, and Locations

## The Problem

The SO-101 ACT policy is trained via pure behavioral cloning — it learns a direct mapping from visual observations to action chunks. It has no explicit understanding of object identity, slot geometry, or absolute workspace position. It fails when:

- An object or slot geometry was not in the training distribution
- The object/slot is placed at a different absolute location (even slightly)

The goal is to extend the policy progressively so it generalises across three axes:

1. **Novel objects** — pick up shapes/objects not seen during training
2. **Novel slots** — insert into slots with different geometries not seen during training
3. **Novel locations** — perform the task when object/slot have been moved to a different absolute position in the workspace

---

## Architecture Background (What ACT Currently Does)

Before modifying anything, it helps to understand what the policy actually does with visual input.

**Token flow into the Transformer encoder:**

```
[latent_token, robot_state_token?, env_state_token?, *flattened_camera_feature_maps]
```

1. Images go through **ResNet18** → feature maps
2. **2D sinusoidal positional embeddings** are added per pixel of the feature map (each spatial location gets a unique embedding). This is what gives ACT some spatial awareness already.
3. All tokens (1D state tokens + 2D image tokens) go into a **Transformer encoder** (4 layers, 512 hidden dim)
4. The **decoder** cross-attends to encoder outputs and generates an action chunk of size 100

**What this means for generalisation:**

- The 2D positional embeddings give the policy some ability to reason about where things are in the image — but only for positions it has seen during training
- The ResNet18 features capture visual appearance — but only for appearances seen during training
- There is no language/goal/task conditioning — it is purely "see this → do this"

**Key existing hook:** There is an `env_state_feature` path in the model (currently unused in most setups) that injects an arbitrary flat vector as a token into the encoder. This is the cleanest insertion point for spatial conditioning (Phase 2) without changing the transformer architecture.

---

## Phase 0: Establish Baselines First (2 days, no code changes)

**Do not skip this. It will save weeks of misdirected effort.**

Collect 50 episodes with fixed object, fixed slot, fixed workspace centre. Train for 100k steps. Then measure:

1. **Location tolerance bubble**: shift object/slot by ±5, ±10, ±15 cm from training centre. Plot success rate at each offset.
2. **Novel object**: swap to 3 visually similar + 3 visually different objects. Record pick success and insert success separately.
3. **Novel slot**: swap 2–3 different slot geometries. Record insertion success.

At each failure, watch the wrist-camera footage and classify the failure mode:

- (a) Wrong approach trajectory — the arm goes to the wrong place entirely
- (b) Grasp slip — correct approach, arm closes but misses
- (c) Wrong insertion angle — grasped correctly but can't seat into the slot

This diagnosis determines which phase to prioritise. If most failures are (a), location conditioning (Phase 2) is the priority. If (c), goal-image conditioning (Phase 3) will help more.

---

## Phase 1: Data-Centric Hardening (1–2 weeks, minimal code changes)

**This has the highest ROI. Do not invest engineering time in Phase 2/3 until this is saturated.**

### 1a. Enable Aggressive Image Augmentation

**File:** `src/lerobot/datasets/transforms.py` + your training config

The `ImageTransformsConfig` has augmentation disabled by default (`enable: False`). Enable it and use broader ranges:

| Transform        | Current default | Recommended |
| ---------------- | --------------- | ----------- |
| Brightness       | (0.8, 1.2)      | (0.5, 1.5)  |
| Contrast         | (0.8, 1.2)      | (0.5, 2.0)  |
| Saturation       | (0.5, 1.5)      | (0.5, 1.5)  |
| Hue              | (-0.05, 0.05)   | (-0.1, 0.1) |
| Affine translate | (0.05, 0.05)    | (0.1, 0.1)  |

The `affine translate=(0.1, 0.1)` means up to 10% of image width translation — at 640px that is ~64px, corresponding to roughly 5–8 cm of workspace shift. This is the cheapest location generalisation trick available.

Also add:

- `RandomResizedCrop(scale=(0.85, 1.0))` — simulates small zoom/camera-position changes
- `RandomErasing(p=0.3, scale=(0.02, 0.1))` — partial occlusion robustness

Both require adding a new `elif` branch in `make_transform_from_config()` in `transforms.py`.

### 1b. Systematic Data Variation

Collect more diverse demonstrations using `lerobot-record`:

- Divide the workspace into a **3×3 grid**. Collect 15–20 episodes per cell for both object placement and slot placement → ~300 total episodes
- Collect **40 episodes per object** across 5 different objects
- Collect **40 episodes per slot geometry** across 3 slot types
- **Vary lighting** between sessions (overhead on, overhead off + side lamp, afternoon window light)
- **Vary camera tilt ±5 degrees** between sessions — the affine augmentation will cover the resulting gap

Merge datasets using `aggregate_datasets()` in `src/lerobot/datasets/aggregate.py`.

> **Critical:** After merging, always rerun `compute_episode_stats()` + `aggregate_stats()` on the merged dataset and verify the new stats file exists before training. Stale normalisation stats are a frequent silent failure mode.

### 1c. Selective Backbone Unfreeze (Optional, needs >500 episodes)

**File:** `src/lerobot/policies/act/configuration_act.py`, `modeling_act.py`

By default, `FrozenBatchNorm2d` is used and backbone gradients may not flow to the final layers. The `get_optim_params()` in `modeling_act.py` already handles backbone LR separately via `optimizer_lr_backbone`.

Add `unfreeze_backbone_layers: list[str] = field(default_factory=list)` to `ACTConfig`. Unfreeze `layer4` only. Set `optimizer_lr_backbone` to `1e-6` (10× lower than head LR). This fine-tunes the final feature extractor toward manipulation-relevant features without destabilising batch norm statistics.

Only attempt with >500 episodes — unfrozen BN with small datasets tends to overfit the visual features to your specific objects.

**Validation:** Rerun Phase 0 protocol after 1a+1b. If location success at ±10 cm goes from ~20% to ~60%+, the data-centric approach has solved the problem and Phases 2–4 can wait.

---

## Phase 2: Spatial Conditioning via Keypoints (1–2 weeks, ~50 lines of code)

**Target:** Give the policy an explicit spatial token telling it where the object and slot are. This addresses location generalisation more directly than augmentation alone.

### 2a. Shape Detection Script (OpenCV)

Rather than physical markers (AprilTags) or large ML models (GroundingDINO), a simple OpenCV shape detector is the right first approach. The block and slot are known geometric shapes under controlled lab lighting — contour matching is fast, requires no training, and works directly on the actual geometry rather than a fiducial.

**What it produces:** `[cx_obj, cy_obj, w_obj, h_obj, cx_slot, cy_slot, w_slot, h_slot]` — 8 floats, normalised to [0, 1] in image space.

**Implementation sketch:**

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

**Calibration per object/slot:** Write a small calibration script that shows a live camera feed with HSV sliders (`cv2.createTrackbar`) to dial in the correct range for each colour under your lab lighting. Save ranges as JSON — one file per object/slot type. Takes 5–10 minutes per new object.

**Handling orientation:** `cv2.minAreaRect()` gives the rotation angle of the detected contour. If slot orientation matters for insertion, extend the output to 10 floats: `[cx_obj, cy_obj, w_obj, h_obj, angle_obj, cx_slot, cy_slot, w_slot, h_slot, angle_slot]` and update `spatial_conditioning_dim` in `ACTConfig` accordingly.

**Fallback for novel objects:** For objects without a calibrated HSV range, fall back to a zero vector or last valid detection (via EMA smoothing). The policy should learn to rely on visual features when the spatial token is uninformative.

**Upgrade path:** Once validated, switching to GroundingDINO (text-prompted, no per-object calibration) is straightforward — it produces the same 8-float output, so only the `SpatialConditioningProcessorStep` needs updating.

### 2b. Inject Spatial Conditioning via the Existing `env_state_feature` Path

The key insight is that **no changes to the Transformer architecture are needed**. The existing `env_state_feature` path in `modeling_act.py` already injects a flat vector as a token into the encoder sequence.

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

1. Run detector offline on all training videos (batch process)
2. Use `add_features()` from `dataset_tools.py` to add the keypoint vector as a new float feature to the existing dataset
3. Recompute normalisation stats to cover the new feature

**Validation:** Rerun Phase 0 location protocol. If the location tolerance bubble expands from ±10 cm to ±20 cm+, spatial conditioning is working. If not, inspect what the model attends to by adding a hook on `ACTEncoderLayer.self_attn` and check whether the spatial token has high attention weights.

---

## Phase 3: Goal-Image Conditioning (2–3 weeks, ~80 lines of code)

**Target:** Novel slot geometry generalisation — show the policy what the final state looks like, so it can plan the insertion angle without having seen that slot before.

### 3a. Collecting Goal Images

For each training episode, you need a "goal image" showing the final state. Three options:

1. **Last-frame goals** (easiest) — use the final frame of each recorded video. After the robot completes the task, move the arm out of frame and capture a clean goal image. Adds ~10 seconds per episode.
2. **Synthetic goals** — use SAM2 to mask out the arm from the final frame, then LaMa inpainting to fill in the background. More generalizable but more effort.
3. **Last frame raw** — use the final frame as-is (arm still in view). Works as a starting point but the model may attend to the arm pose rather than the object/slot.

Start with option 1 for new collection sessions. For existing data, option 3 is the fastest.

### 3b. Architecture Changes

**Files:** `src/lerobot/policies/act/modeling_act.py`, `configuration_act.py`

**Config additions:**

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

**Validation:** Test insertion success on 3 novel slot geometries (with a goal image provided at inference). If this improves over Phase 2 alone, goal conditioning is contributing.

---

## Phase 4: Language Conditioning (3–4 weeks, ~150 lines of code)

**Target:** Full task specification via natural language — "pick up the blue cylinder and insert into the square slot." This is the most powerful generalisation mechanism but also the most infrastructure.

### 4a. Text Encoder

**Recommended: CLIP ViT-B/32 text encoder, frozen.**

- 512-dim text embeddings
- Pre-cache all embeddings offline — zero inference overhead at evaluation time
- Good at associating visual-semantic descriptions ("red cube", "square slot")
- Lightweight (~150 MB)

Alternative: T5-small/base — better for compositional instructions, but heavier.

### 4b. Architecture Changes

**Files:** `modeling_act.py`, `configuration_act.py`, `processor_act.py`

**Config additions:**

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

**Data labeling:** For each episode, type a task description. Keep it short and consistent:

- "pick up the [colour] [shape] and insert into the [shape] slot"
- Example: "pick up the red cube and insert into the round slot"

At ~5 seconds per episode × 300 episodes = ~25 minutes total labeling.

---

## Recommended Execution Order

```
Phase 0  →  Phase 1a  →  Phase 1b  →  evaluate
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
             Location still         Object appearance       Slot geometry
             failing                failing                 failing
                    │                     │                     │
             Phase 2                Phase 1c +            Phase 3
             (keypoints)            diverse data           (goal image)
                    │
             Phase 4 (language) — last, subsumes all, highest infrastructure cost
```

---

## Training Budgets (Approximate)

| Phase                         | Steps | GPU Time  |
| ----------------------------- | ----- | --------- |
| Phase 1 (augmentation + data) | 100k  | ~4 hours  |
| Phase 2 (keypoints)           | 80k   | ~3 hours  |
| Phase 3 (goal image)          | 150k  | ~7 hours  |
| Phase 4 (language)            | 200k  | ~10 hours |

---

## Evaluation Protocol

For each phase:

- **10 episodes per condition** (location offset level / novel object type / novel slot type)
- **Report separately**: pick success rate, insert success rate, full-task success rate
- **Plot**: success rate vs. workspace offset distance as a curve (not just a single number)
- **Run**: `lerobot-eval --policy.path=<checkpoint>` with a fixed random seed

---

## Common Failure Modes to Watch For

1. **Stale normalisation stats** after dataset merges — always run `aggregate_stats()` after `aggregate_datasets()` and verify the stats file before training
2. **Backbone gradient explosion** during unfreeze — keep `optimizer_lr_backbone` at least 10× lower than head LR
3. **Goal image leakage** — ensure the goal image comes from after the arm has moved out of frame, otherwise the model learns to copy the final arm pose rather than the final object state
4. **Detector noise at inference** — keypoint detectors are noisier at inference than in offline-processed training data; smooth with exponential moving average over 3 frames in the processor
