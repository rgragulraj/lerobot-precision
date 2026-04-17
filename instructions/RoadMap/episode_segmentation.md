# Episode Segmentation: Unified Collection Strategy

## 1. The Core Concept: "Collect Once, Train Twice"

Instead of collecting data for Policy 1 (Pick-to-Hover) and Policy 2 (Hover-to-Insert) in separate sessions, we perform a single, continuous teleoperation sequence. This sequence is then programmatically "cut" at the **Canonical Hover Pose** to generate training data for both policies simultaneously.

### Why this is better:

- **Efficiency:** You reach your 500-episode target 2x faster.
- **Distribution Matching:** Policy 2 is trained on the _exact_ terminal states (including human error/offsets) that Policy 1 actually produces, making the handoff significantly more robust.
- **Flow Consistency:** The transition between policies is grounded in real physical motion rather than two artificially separated tasks.

---

## 2. The "Handoff Marker" Protocol

To make the segmentation work, the teleop session must follow a strict three-phase rhythm.

### Phase A: Coarse Approach (Policy 1)

- **Action:** Pick up the block from any workspace cell.
- **Target:** Move to the **Canonical Hover Pose** (directly above the slot, ~4cm height).
- **The Trigger:** Once at the hover pose, the teleoperator **pauses for 1 second** and hits the **Marker Key (M)**.
- **Outcome:** The recording script logs this `segment_index`. This frame is the **End State** for Policy 1 and the **Start State** for Policy 2.

### Phase B: Precision Insertion (Policy 2)

- **Action:** From the hover pose, perform the final alignment and insertion.
- **Target:** Block fully seated and flush.
- **The Trigger:** Once seated, hit the **End Key (→)**.

### Phase C: Goal Image Capture

- **Action:** Open the gripper and retract the arm entirely from the workspace.
- **The Trigger:** Hit the **Goal Key (G)**.
- **Outcome:** This captures the "Success State" for the Precision Policy (wrist camera) and the Coarse Policy (top-down camera).

---

## 3. Technical Requirements

### 1. The Handoff Pause (Critical)

The 1-second pause at the hover pose is non-negotiable.

- **Without it:** Policy 1 learns to "fly through" the target at high speed (residual velocity), causing it to overshoot during autonomous inference.
- **With it:** The policy learns a zero-velocity stabilization, which is required for a clean handoff to Policy 2.

### 2. Dual Goal Images

A single episode now requires two distinct goal representations:

- **Policy 1 Goal:** Top-down image of the block held at hover (captured at the Marker frame).
- **Policy 2 Goal:** Wrist-camera image of the block seated with the arm removed (captured at the End frame).

### 3. Metadata Logging

The recording script must be modified to save a `handoff_frame` field in the episode metadata. This allows the training loader to know exactly where Policy 1 ends and Policy 2 begins.

---

## 4. The Processing Pipeline

Once a "Unified" dataset is collected (e.g., `policy_unified_v1`), it goes through the following offline steps:

1. **Spatial Feature Extraction:**
   - Run `add_spatial_features.py` using the **Top-Down** camera for frames `0` to `M`.
   - Run `add_spatial_features.py` using the **Wrist** camera for frames `M` to `End`.
2. **Goal Image Registration:**
   - Register the frame at `M` as the `observation.images.goal` for Policy 1.
   - Register the final retracted frame as the `observation.images.goal` for Policy 2.
3. **Dataset Splitting:**
   - A utility script `scripts/split_unified_dataset.py` generates two "virtual" datasets:
     - `policy1_data`: Episodes trimmed to `[0 : M]`.
     - `policy2_data`: Episodes trimmed to `[M : End]`.

---

## 5. Summary Table: Unified vs. Separate

| Feature               | Separate Collection        | Unified + Segmentation             |
| :-------------------- | :------------------------- | :--------------------------------- |
| **Collection Time**   | High (2 separate sessions) | **Low (1 session)**                |
| **Handoff Quality**   | Brittle (Idealized starts) | **Robust (Natural starts)**        |
| **Data Cleaning**     | Easy                       | **Requires Marker (M) precision**  |
| **Storage**           | Duplicated frames          | **Shared frames (efficient)**      |
| **Teleop Complexity** | Easy                       | **Medium (requires rhythm/pause)** |

---

## 6. Next Steps for Implementation

1. **Modify `lerobot-record`** to support the `M` key and `handoff_frame` metadata.
2. **Update `add_spatial_features.py`** to support frame-range specific detection.
3. **Create `split_unified_dataset.py`** to automate the training set preparation.
