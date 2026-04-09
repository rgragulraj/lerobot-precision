#!/usr/bin/env python3
"""Block/slot shape detector for Policy 1 (top-down camera) and Policy 2 (wrist camera).

Outputs a 10-float spatial token per frame:
    [cx_block, cy_block, w_block, h_block, angle_block,
     cx_slot,  cy_slot,  w_slot,  h_slot,  angle_slot]

All values normalised to [0, 1]. Angles normalised to [-0.5, 0.5] (representing [-90°, 90°]).
On detection failure the corresponding 5 values are zero.

Detection approach: lighting-independent shape matching
-------------------------------------------------------
Both WristCameraDetector (Policy 2) and TopCameraShapeDetector (Policy 1) use the same
geometry-based pipeline — no HSV colour tuning required:

  1. CLAHE — normalises local contrast regardless of ambient lighting.
  2. Canny edge detection — responds to intensity gradients, not absolute brightness.
  3. Otsu auto-threshold — selects the optimal Canny threshold from the image histogram.
  4. Contour extraction — finds closed geometric boundaries.
  5. cv2.matchShapes — compares shapes using Hu moment invariants (scale/rotation/lighting
     invariant). A template captured once works across sessions and lighting changes.

Key tuning parameters (see tuning guide below):
  match_threshold  — maximum acceptable matchShapes score (0.0 = identical, higher = looser).
  min_area         — minimum contour area in pixels to consider (filters background noise).

Tuning guide
------------
These two parameters control detection quality. Adjust them when:

  match_threshold (default wrist: 0.25, top: 0.30)
    SYMPTOMS of too-low threshold (too strict):
      - Detection rate <90% even when object is clearly in frame.
      - Correct object is visible but not highlighted in --verify mode.
      - Dry-run shows high block/slot failure rate.
    FIX: Raise by 0.05 increments. Upper limit is ~0.50 — above that, unrelated shapes
    start matching.
    SYMPTOMS of too-high threshold (too loose):
      - Wrong object is highlighted (a background edge instead of the block/slot).
      - Token values jump erratically frame-to-frame in --verify mode.
      - Block and slot bounding boxes overlap or swap.
    FIX: Lower by 0.05 increments. If even 0.10 is too loose, the template is likely
    poor — re-calibrate with the object better isolated.

  min_area (default wrist: 100 px, top: 500 px)
    SYMPTOMS of too-low min_area:
      - Tiny background contours (shadows, cable edges) being detected instead of the object.
      - Token jumps to wrong positions when the arm is in motion.
    FIX: Open --verify, move the arm around, observe which spurious contours appear.
      Measure the approximate pixel area of the spurious contour (printed in --calibrate mode).
      Set min_area to ~2× that area.
    SYMPTOMS of too-high min_area:
      - Object not detected even when clearly in frame.
      - Detection works when object fills the frame but fails at normal working distance.
    FIX: Lower by 100 px increments until the object is reliably detected.

Workflow for a new setup
------------------------
  1. Run --calibrate for block, then slot.
  2. Run --verify — check that boxes are stable across 30+ seconds of arm movement.
  3. Run add_spatial_features.py --dry_run — target >90% detection rate for both objects.
  4. If detection rate is low: adjust match_threshold, re-verify, re-dry-run.
  5. If spurious detections: adjust min_area, re-verify.
  6. Run add_spatial_features.py (without --dry_run) once satisfied.

Calibration file format
-----------------------
Wrist camera (Policy 2): scripts/wrist_calibration.json
Top-down camera (Policy 1): scripts/top_shape_calibration.json

Both use the same JSON format:

  Single-shape (flat format — backward-compatible):
    { "block": {"contour": [[x,y], ...]}, "slot": {"contour": [[x,y], ...]} }

  Multi-shape (nested format — Policy 1 Phase 4, Policy 2 Phase 3+):
    { "square": { "block": {...}, "slot": {...} },
      "round":  { "block": {...}, "slot": {...} } }

Both formats can coexist in the same file.

Usage
-----
# Policy 2 — wrist camera (--camera wrist is the default):
python scripts/detect_block_slot.py --calibrate --target=slot  --camera wrist --camera_index=7
python scripts/detect_block_slot.py --calibrate --target=block --camera wrist --camera_index=7
python scripts/detect_block_slot.py --verify --camera wrist

# Policy 2 Phase 3+ multi-shape:
python scripts/detect_block_slot.py --calibrate --target=slot --shape=square --camera wrist
python scripts/detect_block_slot.py --calibrate --target=slot --shape=round  --camera wrist
python scripts/detect_block_slot.py --verify --shape=square --camera wrist

# Policy 1 — top-down camera:
python scripts/detect_block_slot.py --calibrate --target=slot  --camera top --camera_index=5
python scripts/detect_block_slot.py --calibrate --target=block --camera top --camera_index=5
python scripts/detect_block_slot.py --verify --camera top

# Policy 1 with named shapes (for multi-slot workspaces and Phase 4 routing):
python scripts/detect_block_slot.py --calibrate --target=slot --shape=square --camera top
python scripts/detect_block_slot.py --calibrate --target=slot --shape=round  --camera top
python scripts/detect_block_slot.py --verify --shape=square --camera top
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Per-camera constants
# ---------------------------------------------------------------------------

# Policy 2 — wrist camera (close-up, ~4 cm from objects, minimal background clutter).
CALIBRATION_PATH = Path(__file__).parent / "wrist_calibration.json"
MATCH_THRESHOLD = 0.25  # strict — close-up view has clean, well-defined contours
_WRIST_MIN_AREA = 100  # small contours are meaningful at close range

# Policy 1 — top-down camera (~50 cm from objects, more background clutter, greater
# scale variation across the 3×3 workspace grid).
TOP_CALIBRATION_PATH = Path(__file__).parent / "top_shape_calibration.json"
TOP_MATCH_THRESHOLD = 0.30  # slightly looser — perspective and scale vary more
TOP_MIN_AREA = 500  # filter out cable edges, shadows, and small background contours

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration data helpers
# ---------------------------------------------------------------------------


def load_calibration(calibration_path: Path = CALIBRATION_PATH) -> dict:
    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_path}\n"
            "Run calibration first — see script usage at the top of this file."
        )
    with open(calibration_path) as f:
        return json.load(f)


def save_calibration(
    target: str,
    contour: np.ndarray,
    shape: str | None = None,
    calibration_path: Path = CALIBRATION_PATH,
) -> None:
    """Save a contour template to the calibration file.

    Args:
        target: 'slot' or 'block'.
        contour: numpy array of shape (N, 1, 2) or (N, 2).
        shape: Optional shape name (e.g. 'square', 'round'). When provided, the template
            is stored under data[shape][target] for multi-shape usage.
            When None, stored flat as data[target] for single-shape backward compat.
        calibration_path: JSON file to write to. Defaults to wrist_calibration.json.
    """
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if calibration_path.exists():
        with open(calibration_path) as f:
            existing = json.load(f)

    pts = contour.reshape(-1, 2).tolist()
    template = {"contour": pts}

    if shape is not None:
        if shape not in existing:
            existing[shape] = {}
        existing[shape][target] = template
        print(f"[Calibration] Saved {shape}/{target} template ({len(pts)} points) → {calibration_path}")
    else:
        existing[target] = template
        print(f"[Calibration] Saved {target} template ({len(pts)} points) → {calibration_path}")

    with open(calibration_path, "w") as f:
        json.dump(existing, f, indent=2)


def list_calibrated_shapes() -> list[str]:
    """Return a list of named shape profiles in the calibration file.

    Returns only keys that have a nested block/slot structure (Phase 3+ format).
    The flat 'block'/'slot' entries (Phase 2 format) are not returned.
    """
    if not CALIBRATION_PATH.exists():
        return []
    with open(CALIBRATION_PATH) as f:
        cal = json.load(f)
    return [
        k
        for k, v in cal.items()
        if isinstance(v, dict) and "block" in v or "slot" in v and k not in ("block", "slot")
    ]


def _load_contour(cal_section: dict, key: str) -> np.ndarray | None:
    """Load a saved contour template as a numpy array (N, 1, 2) int32.

    Args:
        cal_section: The calibration dict section to look in. For flat format this is
            the root calibration dict. For nested format it is cal[shape_name].
        key: 'block' or 'slot'.
    """
    if key not in cal_section or "contour" not in cal_section[key]:
        return None
    pts = np.array(cal_section[key]["contour"], dtype=np.int32).reshape(-1, 1, 2)
    return pts


# ---------------------------------------------------------------------------
# Core preprocessing and detection
# ---------------------------------------------------------------------------


def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert frame to a lighting-robust edge/contour binary mask.

    Pipeline:
      BGR → Grayscale → CLAHE (local contrast normalisation) →
      Gaussian blur (noise reduction) → Canny edges (Otsu auto-threshold) →
      Morphological close (seal contour gaps)

    The output is a binary image suitable for cv2.findContours.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE: clip limit 2.0, 8×8 tile grid — balances local contrast enhancement
    # without amplifying noise in flat regions.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Gaussian blur to reduce high-frequency noise before edge detection.
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # Otsu's method: automatically finds the optimal threshold from the bimodal
    # histogram — adapts to any ambient lighting without manual tuning.
    otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blurred, otsu_val * 0.5, otsu_val)

    # Morphological close: seals small gaps in detected edges so contours are closed.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed


def _extract_contours(binary: np.ndarray, min_area: int = _WRIST_MIN_AREA) -> list:
    """Find all external contours above the minimum area threshold."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def _best_match(
    contours: list, template: np.ndarray, threshold: float = MATCH_THRESHOLD
) -> np.ndarray | None:
    """Return the contour most similar to the template by Hu moment shape matching.

    cv2.matchShapes returns 0.0 for identical shapes and larger values for increasingly
    different shapes. CONTOURS_MATCH_I2 is the most robust mode (normalised Hu moments).

    Args:
        contours: list of contour arrays to search.
        template: reference contour (N, 1, 2) int32.
        threshold: maximum acceptable match score (default 0.25).

    Returns:
        The best-matching contour, or None if no contour scores below threshold.
    """
    best_contour = None
    best_score = threshold
    for c in contours:
        score = cv2.matchShapes(template, c, cv2.CONTOURS_MATCH_I2, 0)
        if score < best_score:
            best_contour = c
            best_score = score
    return best_contour


def _contour_to_token(contour: np.ndarray, frame_shape: tuple) -> tuple:
    """Extract (cx, cy, w, h, angle) from a contour, all normalised to [0, 1].

    Uses cv2.minAreaRect which also returns the in-plane rotation angle of the
    minimum bounding rectangle — this is the angle value for spatial conditioning.

    Args:
        contour: detected contour array.
        frame_shape: (height, width, ...) of the source frame.

    Returns:
        (cx_n, cy_n, w_n, h_n, angle_n) where cx/cy/w/h are in [0, 1] and
        angle is in [-0.5, 0.5] (representing [-90°, 90°]).
    """
    h, w = frame_shape[:2]
    rect = cv2.minAreaRect(contour)
    (cx, cy), (bw, bh), angle = rect
    return cx / w, cy / h, bw / w, bh / h, angle / 180.0


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


class WristCameraDetector:
    """Detects block face and slot opening in wrist-camera BGR frames.

    Uses lighting-independent shape template matching:
    CLAHE + Canny edges + Otsu auto-threshold + cv2.matchShapes (Hu moments).

    The detector requires one-time calibration per shape — run
    `python scripts/detect_block_slot.py --calibrate --target=slot` and
    `... --target=block` to capture shape templates. For Phase 3+ multi-shape
    datasets, pass a shape name to select the correct template profile.

    Args:
        calibration_path: Path to wrist_calibration.json.
        match_threshold: Maximum cv2.matchShapes score to accept a match.
            Lower = stricter. 0.25 is a good default.
        shape: Optional shape name (e.g. 'square', 'round'). When provided,
            loads templates from cal[shape]['block'] / cal[shape]['slot'].
            When None (default), uses the flat cal['block'] / cal['slot'] format
            for backward compatibility with Phase 2 single-shape datasets.
    """

    def __init__(
        self,
        calibration_path: Path = CALIBRATION_PATH,
        match_threshold: float = MATCH_THRESHOLD,
        shape: str | None = None,
    ):
        cal = load_calibration()
        self.shape = shape
        self.match_threshold = match_threshold

        if shape is not None:
            # Multi-shape format: look up cal[shape]
            if shape not in cal:
                raise KeyError(
                    f"Shape '{shape}' not found in calibration file {calibration_path}.\n"
                    f"Available shapes: {[k for k in cal if k not in ('block', 'slot')]}\n"
                    f"Run: python scripts/detect_block_slot.py --calibrate --target=block --shape={shape}"
                )
            cal_section = cal[shape]
        else:
            # Flat format (Phase 2 / backward compat)
            cal_section = cal

        self.block_template = _load_contour(cal_section, "block")
        self.slot_template = _load_contour(cal_section, "slot")

        if self.block_template is None:
            logger.warning("No block template in calibration. Block detection disabled.")
        if self.slot_template is None:
            logger.warning("No slot template in calibration. Slot detection disabled.")

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Detect block face and slot opening and return the 10-float spatial token.

        Args:
            frame_bgr: BGR image as numpy array (H, W, 3), uint8.

        Returns:
            np.ndarray of shape (10,), float32:
                [cx_block, cy_block, w_block, h_block, angle_block,
                 cx_slot,  cy_slot,  w_slot,  h_slot,  angle_slot]
            All spatial values in [0, 1]; angles in [-0.5, 0.5].
            Undetected objects produce zeros in their 5-value slot.
        """
        token = np.zeros(10, dtype=np.float32)
        binary = _preprocess(frame_bgr)
        contours = _extract_contours(binary)

        if not contours:
            return token

        if self.block_template is not None:
            matched = _best_match(contours, self.block_template, self.match_threshold)
            if matched is not None:
                token[0:5] = _contour_to_token(matched, frame_bgr.shape)

        if self.slot_template is not None:
            # Exclude the contour already claimed by the block to avoid double-detection.
            remaining = contours
            if self.block_template is not None:
                block_match = _best_match(contours, self.block_template, self.match_threshold)
                if block_match is not None:
                    remaining = [c for c in contours if not np.array_equal(c, block_match)]
            matched = _best_match(remaining, self.slot_template, self.match_threshold)
            if matched is not None:
                token[5:10] = _contour_to_token(matched, frame_bgr.shape)

        return token

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw detection overlay on a copy of the frame for visual verification."""
        vis = frame_bgr.copy()
        binary = _preprocess(frame_bgr)
        contours = _extract_contours(binary)

        shape_label = f" [{self.shape}]" if self.shape else ""

        if not contours:
            cv2.putText(
                vis,
                f"No contours detected{shape_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            return vis

        def _draw_match(template, colour, label):
            if template is None:
                return
            matched = _best_match(contours, template, self.match_threshold)
            if matched is None:
                return
            rect = cv2.minAreaRect(matched)
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(vis, [box], 0, colour, 2)
            cx, cy = int(rect[0][0]), int(rect[0][1])
            angle = rect[2]
            cv2.putText(
                vis,
                f"{label}{shape_label} {angle:.1f}°",
                (cx - 40, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1,
            )

        _draw_match(self.block_template, (0, 255, 0), "block")
        _draw_match(self.slot_template, (0, 0, 255), "slot")

        # Print the spatial token values at the bottom of the frame.
        token = self.detect(frame_bgr)
        labels = ["cx_b", "cy_b", "w_b", "h_b", "a_b", "cx_s", "cy_s", "w_s", "h_s", "a_s"]
        text = "  ".join(f"{lbl}:{v:.2f}" for lbl, v in zip(labels, token, strict=False))
        cv2.putText(
            vis, text[:80], (5, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1
        )
        return vis


# ---------------------------------------------------------------------------
# Top-down camera detector (Policy 1 Phase 2)
# ---------------------------------------------------------------------------


class TopCameraShapeDetector:
    """Detects block face and slot opening in top-down camera BGR frames using
    the same lighting-independent shape template matching pipeline as WristCameraDetector.

    Use this for Policy 1 spatial conditioning (camera index 5, ~50 cm above workspace).

    Differences from WristCameraDetector:
      - Default calibration: top_shape_calibration.json (separate from wrist_calibration.json)
      - Default match_threshold: 0.30 (looser — more scale/perspective variation from above)
      - Default min_area: 500 px (higher — filters background clutter at working distance)

    Multi-slot support: pass shape='square' to target the square slot out of multiple slots
    in the workspace. The detector finds the contour best matching that shape template and
    ignores others. This is the mechanism Phase 4 language conditioning builds on.

    Calibration:
        python scripts/detect_block_slot.py --calibrate --target=block --camera top
        python scripts/detect_block_slot.py --calibrate --target=slot  --camera top

    Multi-shape calibration (for multi-slot workspaces):
        python scripts/detect_block_slot.py --calibrate --target=slot --shape=square --camera top
        python scripts/detect_block_slot.py --calibrate --target=slot --shape=round  --camera top

    Args:
        calibration_path: Path to top_shape_calibration.json.
        match_threshold: Maximum cv2.matchShapes score to accept a match.
            See tuning guide at the top of this file.
        min_area: Minimum contour area in pixels. Raise this if background clutter
            (shadows, cables, table edges) is being detected instead of the target object.
        shape: Shape profile name for multi-slot workspaces (e.g. 'square', 'round').
            When None, uses the flat single-shape format.
    """

    def __init__(
        self,
        calibration_path: Path = TOP_CALIBRATION_PATH,
        match_threshold: float = TOP_MATCH_THRESHOLD,
        min_area: int = TOP_MIN_AREA,
        shape: str | None = None,
    ):
        cal = load_calibration(calibration_path)
        self.match_threshold = match_threshold
        self.min_area = min_area
        self.shape = shape

        if shape is not None:
            if shape not in cal:
                available = [k for k in cal if k not in ("block", "slot")]
                raise KeyError(
                    f"Shape '{shape}' not found in {calibration_path}.\n"
                    f"Available shapes: {available}\n"
                    f"Run: python scripts/detect_block_slot.py --calibrate --target=slot "
                    f"--shape={shape} --camera top"
                )
            cal_section = cal[shape]
        else:
            cal_section = cal

        self.block_template = _load_contour(cal_section, "block")
        self.slot_template = _load_contour(cal_section, "slot")

        if self.block_template is None:
            logger.warning("No block template in calibration. Block detection disabled.")
        if self.slot_template is None:
            logger.warning("No slot template in calibration. Slot detection disabled.")

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Detect block face and slot opening and return the 10-float spatial token.

        Args:
            frame_bgr: BGR image as numpy array (H, W, 3), uint8.

        Returns:
            np.ndarray of shape (10,), float32. Undetected objects produce zeros.
        """
        token = np.zeros(10, dtype=np.float32)
        binary = _preprocess(frame_bgr)
        contours = _extract_contours(binary, min_area=self.min_area)

        if not contours:
            return token

        if self.block_template is not None:
            matched = _best_match(contours, self.block_template, self.match_threshold)
            if matched is not None:
                token[0:5] = _contour_to_token(matched, frame_bgr.shape)

        if self.slot_template is not None:
            remaining = contours
            if self.block_template is not None:
                block_match = _best_match(contours, self.block_template, self.match_threshold)
                if block_match is not None:
                    remaining = [c for c in contours if not np.array_equal(c, block_match)]
            matched = _best_match(remaining, self.slot_template, self.match_threshold)
            if matched is not None:
                token[5:10] = _contour_to_token(matched, frame_bgr.shape)

        return token

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw detection overlay on a copy of the frame for visual verification."""
        vis = frame_bgr.copy()
        binary = _preprocess(frame_bgr)
        contours = _extract_contours(binary, min_area=self.min_area)

        shape_label = f" [{self.shape}]" if self.shape else ""

        # Show all contours above min_area in dim blue so you can see what the detector sees.
        cv2.drawContours(vis, contours, -1, (100, 60, 0), 1)

        if not contours:
            cv2.putText(
                vis,
                f"No contours (min_area={self.min_area}){shape_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            return vis

        def _draw_match(template, colour, label):
            if template is None:
                return
            matched = _best_match(contours, template, self.match_threshold)
            if matched is None:
                cv2.putText(
                    vis,
                    f"{label}: NO MATCH",
                    (10, 60 if label == "slot" else 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                return
            rect = cv2.minAreaRect(matched)
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(vis, [box], 0, colour, 2)
            cx, cy = int(rect[0][0]), int(rect[0][1])
            score = cv2.matchShapes(template, matched, cv2.CONTOURS_MATCH_I2, 0)
            cv2.putText(
                vis,
                f"{label}{shape_label} score={score:.3f}",
                (cx - 50, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
            )

        _draw_match(self.block_template, (0, 255, 0), "block")
        _draw_match(self.slot_template, (0, 0, 255), "slot")

        token = self.detect(frame_bgr)
        labels = ["cx_b", "cy_b", "w_b", "h_b", "a_b", "cx_s", "cy_s", "w_s", "h_s", "a_s"]
        text = "  ".join(f"{lbl}:{v:.2f}" for lbl, v in zip(labels, token, strict=False))
        cv2.putText(
            vis, text[:80], (5, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1
        )

        # Print tuning parameters so they're visible during verification.
        cv2.putText(
            vis,
            f"threshold={self.match_threshold}  min_area={self.min_area}  contours={len(contours)}",
            (5, vis.shape[0] - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
        )
        return vis


# ---------------------------------------------------------------------------
# Calibration UI
# ---------------------------------------------------------------------------


def run_calibration(
    target: str,
    camera_index: int,
    shape: str | None = None,
    calibration_path: Path = CALIBRATION_PATH,
    min_area: int = _WRIST_MIN_AREA,
) -> None:
    """Capture a shape template for the target object.

    Shows a live camera feed overlaid with currently detected contours.
    Press C to capture the largest contour as the template. Press S to save.

    Args:
        target: 'slot' or 'block'.
        camera_index: OpenCV camera index.
        shape: Optional shape name. When provided, saved under data[shape][target].
            When None, saved flat as data[target] (single-shape backward compat).
        calibration_path: JSON file to save to.
        min_area: Minimum contour area to show during calibration. Raise this if the
            live view shows too many small spurious contours cluttering the display.
    """
    assert target in ("slot", "block"), f"target must be 'slot' or 'block', got '{target}'"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    shape_label = f"  shape: {shape}" if shape else "  (no shape name — flat format)"
    win = f"Calibrate {target} — C: capture, S: save & exit, Q: quit"
    cv2.namedWindow(win)

    pending_contour = None

    print(f"\n[Calibration] Target: {target} | Camera: {camera_index}{shape_label}")
    print(f"  Saving to: {calibration_path}")
    print("  1. Position the camera so only the TARGET OBJECT is clearly in frame.")
    print("  2. Press C to capture the largest detected contour as the template.")
    print("  3. Inspect the green overlay — it should tightly outline the target shape.")
    print("  4. Press C again to re-capture if needed, then S to save.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()
        binary = _preprocess(frame)
        contours = _extract_contours(binary, min_area=min_area)

        # Show all contours above min_area in blue.
        cv2.drawContours(vis, contours, -1, (255, 100, 0), 1)

        if pending_contour is not None:
            rect = cv2.minAreaRect(pending_contour)
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)
            label = "TEMPLATE CAPTURED"
            if shape:
                label += f" [{shape}/{target}]"
            cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "Press C to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.putText(
            vis,
            f"Contours: {len(contours)}  min_area={min_area}",
            (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if not contours:
                print("[Calibration] No contours detected — reposition the object or lower --min_area.")
                continue
            pending_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(pending_contour)
            print(f"[Calibration] Template captured: {len(pending_contour)} points, area={area:.0f}px²")

        elif key == ord("s"):
            if pending_contour is None:
                print("[Calibration] Nothing captured yet. Press C first.")
                continue
            save_calibration(target, pending_contour, shape=shape, calibration_path=calibration_path)
            break

        elif key == ord("q"):
            print("[Calibration] Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


def run_verify(
    camera_index: int,
    shape: str | None = None,
    camera: str = "wrist",
    match_threshold: float | None = None,
    min_area: int | None = None,
) -> None:
    """Show live detection overlay to verify template quality.

    Args:
        camera_index: OpenCV camera index.
        shape: Optional shape profile name to verify.
        camera: 'wrist' (Policy 2) or 'top' (Policy 1). Selects the correct
            calibration file and default tuning parameters.
        match_threshold: Override the default match threshold for this camera.
        min_area: Override the default min_area for this camera.
    """
    detector: TopCameraShapeDetector | WristCameraDetector
    if camera == "top":
        detector = TopCameraShapeDetector(
            match_threshold=match_threshold or TOP_MATCH_THRESHOLD,
            min_area=min_area or TOP_MIN_AREA,
            shape=shape,
        )
    else:
        detector = WristCameraDetector(
            match_threshold=match_threshold or MATCH_THRESHOLD,
            shape=shape,
        )

    shape_label = f" [{shape}]" if shape else ""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print(f"\n[Verify — {camera} camera{shape_label}]")
    print("  Green = block | Red = slot")
    print("  Bottom line shows match score per detection and current tuning parameters.")
    print("  Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vis = detector.draw_overlay(frame)
        cv2.imshow(f"Verify [{camera}{shape_label}] — Q to quit", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--calibrate", action="store_true", help="Run shape template calibration.")
    parser.add_argument("--verify", action="store_true", help="Show live detection overlay.")
    parser.add_argument(
        "--camera",
        choices=["wrist", "top"],
        default="wrist",
        help=(
            "Which camera to calibrate/verify. "
            "'wrist' = Policy 2 (index 7, wrist_calibration.json, threshold=0.25). "
            "'top' = Policy 1 (index 5, top_shape_calibration.json, threshold=0.30). "
            "(default: wrist)"
        ),
    )
    parser.add_argument(
        "--target",
        choices=["slot", "block"],
        default="slot",
        help="Which object to calibrate (only with --calibrate).",
    )
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help=(
            "Shape profile name for multi-shape workspaces (e.g. 'square', 'round'). "
            "Templates stored/loaded under data[shape][target] in the JSON. "
            "When omitted, uses flat single-shape format."
        ),
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=None,
        help="OpenCV camera index override. Defaults to 7 for wrist, 5 for top.",
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=None,
        help=(
            "Override match threshold for --verify. "
            "See tuning guide at the top of this file. "
            "Defaults to 0.25 (wrist) or 0.30 (top)."
        ),
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=None,
        help=(
            "Override minimum contour area in pixels for --calibrate and --verify. "
            "See tuning guide at the top of this file. "
            "Defaults to 100 (wrist) or 500 (top)."
        ),
    )
    args = parser.parse_args()

    # Resolve defaults based on selected camera.
    if args.camera == "top":
        cal_path = TOP_CALIBRATION_PATH
        default_index = 5
        default_min_area = TOP_MIN_AREA
    else:
        cal_path = CALIBRATION_PATH
        default_index = 7
        default_min_area = _WRIST_MIN_AREA

    camera_index = args.camera_index if args.camera_index is not None else default_index
    min_area = args.min_area if args.min_area is not None else default_min_area

    if args.calibrate:
        run_calibration(
            args.target,
            camera_index,
            shape=args.shape,
            calibration_path=cal_path,
            min_area=min_area,
        )
    elif args.verify:
        run_verify(
            camera_index,
            shape=args.shape,
            camera=args.camera,
            match_threshold=args.match_threshold,
            min_area=args.min_area,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
