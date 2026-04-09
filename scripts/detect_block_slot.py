#!/usr/bin/env python3
"""Wrist-camera block/slot detector for Policy 2 Phase 2+ spatial conditioning.

Detects the block face and slot opening in the wrist-camera view and outputs a 10-float
spatial token per frame:
    [cx_block, cy_block, w_block, h_block, angle_block,
     cx_slot,  cy_slot,  w_slot,  h_slot,  angle_slot]

All values normalised to [0, 1]. Angles normalised to [-0.5, 0.5] (representing [-90°, 90°]).
On detection failure for either object the corresponding 5 values are zero — the policy must
be robust to this (zero fallback is the correct behaviour, not an error).

Detection approach: lighting-independent shape matching
-------------------------------------------------------
Rather than HSV colour thresholding (which requires re-calibration when lighting changes),
this detector works purely on geometry:

  1. CLAHE — normalises local contrast so the image looks uniformly lit regardless of
     ambient conditions. Does not change shape information.
  2. Canny edge detection — responds to intensity *gradients*, not absolute brightness.
     Lighting changes shift all pixel values together, so gradients remain stable.
  3. Otsu auto-threshold — automatically computes the optimal Canny threshold from the
     image histogram. No manual threshold tuning.
  4. Contour extraction — finds closed geometric boundaries.
  5. cv2.matchShapes — compares contour shapes using Hu moment invariants, which are
     invariant to scale, rotation, and lighting. A template captured once works under
     any lighting.

Calibration (one time per object, per shape):
  Point the camera at the target object (block face or slot opening) so only that object
  is prominently in frame. Press C to capture. The largest contour is saved as the shape
  template. No sliders, no colour tuning.

Calibration file format
-----------------------
When using --shape (Phase 3+, multi-shape):
    {
      "square": {
        "block": {"contour": [[x, y], ...]},
        "slot":  {"contour": [[x, y], ...]}
      },
      "round": {
        "block": {"contour": [[x, y], ...]},
        "slot":  {"contour": [[x, y], ...]}
      }
    }

When using without --shape (Phase 2, single shape, backward-compatible):
    {
      "block": {"contour": [[x, y], ...]},
      "slot":  {"contour": [[x, y], ...]}
    }

Both formats can coexist in the same file.

Modes
-----
--calibrate  : Capture a shape template for the target object.
               Point camera at the object, press C to capture, S to save.
               Run once for slot, once for block (use --target=slot|block).
               Use --shape=<name> to store under a named shape profile.
--verify     : Show live detection overlay to confirm template quality.
               Use --shape=<name> to verify a specific shape profile.

Usage
-----
# Phase 2 (single shape — no --shape flag needed):
python scripts/detect_block_slot.py --calibrate --target=slot --camera_index=7
python scripts/detect_block_slot.py --calibrate --target=block --camera_index=7
python scripts/detect_block_slot.py --verify --camera_index=7

# Phase 3+ (multi-shape — use --shape flag):
python scripts/detect_block_slot.py --calibrate --target=slot --shape=square --camera_index=7
python scripts/detect_block_slot.py --calibrate --target=block --shape=square --camera_index=7
python scripts/detect_block_slot.py --calibrate --target=slot --shape=round --camera_index=7
python scripts/detect_block_slot.py --calibrate --target=block --shape=round --camera_index=7
python scripts/detect_block_slot.py --verify --shape=square --camera_index=7

Calibration saved to scripts/wrist_calibration.json.
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_PATH = Path(__file__).parent / "wrist_calibration.json"

# Maximum cv2.matchShapes score to accept a contour as a match.
# 0.0 = identical shape. Lower values = stricter matching.
# 0.25 is a good starting point — tight enough to reject noise, loose enough for scale/pose variation.
MATCH_THRESHOLD = 0.25

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration data helpers
# ---------------------------------------------------------------------------


def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_PATH}\n"
            "Run: python scripts/detect_block_slot.py --calibrate --target=slot --camera_index=7\n"
            "Then: python scripts/detect_block_slot.py --calibrate --target=block --camera_index=7"
        )
    with open(CALIBRATION_PATH) as f:
        return json.load(f)


def save_calibration(target: str, contour: np.ndarray, shape: str | None = None) -> None:
    """Save a contour template to the calibration file.

    Args:
        target: 'slot' or 'block'.
        contour: numpy array of shape (N, 1, 2) or (N, 2).
        shape: Optional shape name (e.g. 'square', 'round'). When provided, the template
            is stored under data[shape][target] for multi-shape (Phase 3+) usage.
            When None, stored flat as data[target] for single-shape (Phase 2) backward compat.
    """
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            existing = json.load(f)

    pts = contour.reshape(-1, 2).tolist()
    template = {"contour": pts}

    if shape is not None:
        # Nested format: data[shape][target] = template
        if shape not in existing:
            existing[shape] = {}
        existing[shape][target] = template
        print(f"[Calibration] Saved {shape}/{target} template ({len(pts)} points) → {CALIBRATION_PATH}")
    else:
        # Flat format (backward compat): data[target] = template
        existing[target] = template
        print(f"[Calibration] Saved {target} template ({len(pts)} points) → {CALIBRATION_PATH}")

    with open(CALIBRATION_PATH, "w") as f:
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


def _extract_contours(binary: np.ndarray, min_area: int = 100) -> list:
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
# Calibration UI
# ---------------------------------------------------------------------------


def run_calibration(target: str, camera_index: int, shape: str | None = None) -> None:
    """Capture a shape template for the target object.

    Shows a live camera feed overlaid with the currently detected contours.
    Press C to capture the largest contour as the template.
    Press S to save and exit.

    Args:
        target: 'slot' or 'block'.
        camera_index: OpenCV camera index.
        shape: Optional shape name (e.g. 'square', 'round'). When provided, the template
            is saved under data[shape][target] for multi-shape (Phase 3+) usage.
            When None, saved flat as data[target] for backward compatibility.
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
        contours = _extract_contours(binary)

        # Show all detected contours in blue.
        cv2.drawContours(vis, contours, -1, (255, 100, 0), 1)

        # Show the pending (captured) template in green.
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
            f"Contours: {len(contours)}",
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
                print("[Calibration] No contours detected — reposition the object and try again.")
                continue
            # Take the largest contour as the template.
            pending_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(pending_contour)
            print(f"[Calibration] Template captured: {len(pending_contour)} points, area={area:.0f}px²")

        elif key == ord("s"):
            if pending_contour is None:
                print("[Calibration] Nothing captured yet. Press C first.")
                continue
            save_calibration(target, pending_contour, shape=shape)
            break

        elif key == ord("q"):
            print("[Calibration] Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


def run_verify(camera_index: int, shape: str | None = None) -> None:
    """Show live detection overlay to verify template quality.

    Args:
        camera_index: OpenCV camera index.
        shape: Optional shape name to verify. When None, uses the flat (Phase 2) format.
    """
    detector = WristCameraDetector(shape=shape)
    shape_label = f" [{shape}]" if shape else ""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print(f"\n[Verify{shape_label}] Green = block | Red = slot | Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vis = detector.draw_overlay(frame)
        cv2.imshow(f"Detection verify{shape_label} — Q to quit", vis)
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
            "Shape name for multi-shape (Phase 3+) calibration profiles "
            "(e.g. 'square', 'round', 'dshape'). "
            "When provided, templates are stored/loaded under data[shape][target] in the JSON. "
            "When omitted, uses the flat format for backward compatibility with Phase 2."
        ),
    )
    parser.add_argument("--camera_index", type=int, default=7, help="Camera index for wrist camera.")
    args = parser.parse_args()

    if args.calibrate:
        run_calibration(args.target, args.camera_index, shape=args.shape)
    elif args.verify:
        run_verify(args.camera_index, shape=args.shape)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
