#!/usr/bin/env python3
"""Top-down camera block/slot detector for Policy 1 Phase 2 spatial conditioning.

Detects the block and slot opening in the top-down camera view (index 5) using HSV colour
masking and outputs an 8-float (or 10-float with angles) spatial token per frame:

    8-float (default):
        [cx_block, cy_block, w_block, h_block, cx_slot, cy_slot, w_slot, h_slot]

    10-float (--include_angle):
        [cx_block, cy_block, w_block, h_block, angle_block,
         cx_slot,  cy_slot,  w_slot,  h_slot,  angle_slot]

All values normalised to [0, 1]. Angles normalised to [-0.5, 0.5] (representing [-90°, 90°]).
On detection failure for either object the corresponding values are zero.

Detection approach: HSV colour masking
---------------------------------------
Policy 1 uses the top-down camera (index 5) which has a wide, stable field of view.
At this scale, HSV colour thresholding is effective and fast. The top-down view does
not suffer from the lighting instability that makes HSV unreliable at close range
(which is why Policy 2 uses shape template matching instead).

Workflow:
  1. Interactive slider calibration — run once per object colour.
  2. Calibration saved to scripts/top_calibration.json (shared block + slot ranges).
  3. Run offline on recorded videos via add_spatial_features.py --detector_type=hsv_top.
  4. Use TopCameraDetector in processor_act.py for online inference.

Modes
-----
--calibrate  : Interactive HSV slider tool to capture colour range for block or slot.
               Press S to save, Q to quit without saving.
--verify     : Show live detection overlay to confirm calibration quality.

Usage
-----
python scripts/detect_block_slot_hsv.py --calibrate --target=block --camera_index=5
python scripts/detect_block_slot_hsv.py --calibrate --target=slot  --camera_index=5
python scripts/detect_block_slot_hsv.py --verify --camera_index=5

Calibration saved to scripts/top_calibration.json.
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

CALIBRATION_PATH = Path(__file__).parent / "top_calibration.json"

# Minimum contour area in pixels — filters out noise and shadows.
MIN_CONTOUR_AREA = 200

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_PATH}\n"
            "Run:\n"
            "  python scripts/detect_block_slot_hsv.py --calibrate --target=block --camera_index=5\n"
            "  python scripts/detect_block_slot_hsv.py --calibrate --target=slot  --camera_index=5"
        )
    with open(CALIBRATION_PATH) as f:
        return json.load(f)


def save_calibration(target: str, lower: np.ndarray, upper: np.ndarray) -> None:
    """Save HSV range for block or slot to the shared calibration file.

    Args:
        target: 'block' or 'slot'.
        lower: HSV lower bound array [H, S, V].
        upper: HSV upper bound array [H, S, V].
    """
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            existing = json.load(f)

    existing[target] = {"lower": lower.tolist(), "upper": upper.tolist()}

    with open(CALIBRATION_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[Calibration] Saved {target} HSV range → {CALIBRATION_PATH}")


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------


def detect_block_and_slot(
    frame_bgr: np.ndarray,
    block_hsv_range: tuple[np.ndarray, np.ndarray],
    slot_hsv_range: tuple[np.ndarray, np.ndarray],
    include_angle: bool = False,
) -> tuple:
    """Detect block and slot bounding boxes from a BGR top-down camera frame.

    Args:
        frame_bgr: BGR camera frame from the top-down webcam.
        block_hsv_range: Tuple of (lower_hsv, upper_hsv) numpy arrays for block colour.
        slot_hsv_range: Tuple of (lower_hsv, upper_hsv) numpy arrays for slot colour.
        include_angle: If True, returns 5 floats per object (cx, cy, w, h, angle)
            instead of 4. Angle uses cv2.minAreaRect and is normalised to [-0.5, 0.5].

    Returns:
        (block_vec, slot_vec) where each is a tuple of floats normalised to [0, 1],
        or None if the object was not detected.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, w = frame_bgr.shape[:2]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def find_bbox(mask: np.ndarray):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            return None
        if include_angle:
            rect = cv2.minAreaRect(c)
            (cx, cy), (bw, bh), angle = rect
            return cx / w, cy / h, bw / w, bh / h, angle / 180.0  # angle → [-0.5, 0.5]
        else:
            x, y, bw, bh = cv2.boundingRect(c)
            return (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h

    block_mask = cv2.inRange(hsv, *block_hsv_range)
    slot_mask = cv2.inRange(hsv, *slot_hsv_range)

    return find_bbox(block_mask), find_bbox(slot_mask)


# ---------------------------------------------------------------------------
# Detector class (for use in processor and offline scripts)
# ---------------------------------------------------------------------------


class TopCameraDetector:
    """Detects block and slot in top-down camera BGR frames using HSV colour masking.

    Requires calibration via:
        python scripts/detect_block_slot_hsv.py --calibrate --target=block --camera_index=5
        python scripts/detect_block_slot_hsv.py --calibrate --target=slot  --camera_index=5

    Args:
        calibration_path: Path to top_calibration.json.
        include_angle: If True, return 10-float token; otherwise 8-float.
    """

    def __init__(self, calibration_path: Path = CALIBRATION_PATH, include_angle: bool = False):
        cal = load_calibration()
        self.include_angle = include_angle
        self._dim = 10 if include_angle else 8

        if "block" not in cal or "slot" not in cal:
            raise ValueError(
                f"Calibration file {calibration_path} is missing 'block' or 'slot' entry.\n"
                "Run calibration for both targets first."
            )

        self.block_hsv = (
            np.array(cal["block"]["lower"], dtype=np.uint8),
            np.array(cal["block"]["upper"], dtype=np.uint8),
        )
        self.slot_hsv = (
            np.array(cal["slot"]["lower"], dtype=np.uint8),
            np.array(cal["slot"]["upper"], dtype=np.uint8),
        )

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Detect block and slot and return the spatial token.

        Args:
            frame_bgr: BGR image as numpy array (H, W, 3), uint8.

        Returns:
            np.ndarray of shape (8,) or (10,), float32.
            Undetected objects produce zeros in their slot.
        """
        token = np.zeros(self._dim, dtype=np.float32)
        block, slot = detect_block_and_slot(
            frame_bgr, self.block_hsv, self.slot_hsv, include_angle=self.include_angle
        )
        stride = 5 if self.include_angle else 4
        if block is not None:
            token[:stride] = block
        if slot is not None:
            token[stride : stride * 2] = slot
        return token

    def draw_overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw detection overlay on a copy of the frame for visual verification."""
        vis = frame_bgr.copy()
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = frame_bgr.shape[:2]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        def draw_detection(mask, colour, label):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                return
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(vis, [box], 0, colour, 2)
            cx, cy = int(rect[0][0]), int(rect[0][1])
            angle = rect[2]
            cv2.putText(
                vis, f"{label} {angle:.1f}°", (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1
            )

        block_mask = cv2.inRange(hsv, *self.block_hsv)
        slot_mask = cv2.inRange(hsv, *self.slot_hsv)
        draw_detection(block_mask, (0, 255, 0), "block")
        draw_detection(slot_mask, (0, 0, 255), "slot")

        token = self.detect(frame_bgr)
        labels = (
            ["cx_b", "cy_b", "w_b", "h_b", "a_b", "cx_s", "cy_s", "w_s", "h_s", "a_s"]
            if self.include_angle
            else ["cx_b", "cy_b", "w_b", "h_b", "cx_s", "cy_s", "w_s", "h_s"]
        )
        text = "  ".join(f"{lbl}:{v:.2f}" for lbl, v in zip(labels, token, strict=False))
        cv2.putText(
            vis, text[:90], (5, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1
        )
        return vis


# ---------------------------------------------------------------------------
# Interactive calibration UI
# ---------------------------------------------------------------------------


def run_calibration(target: str, camera_index: int) -> None:
    """Interactive HSV slider calibration for a block or slot colour.

    Shows a live masked preview while you adjust H/S/V sliders. Press S to save
    the current range, Q to quit without saving.

    Args:
        target: 'block' or 'slot'.
        camera_index: OpenCV camera index for the top-down webcam.
    """
    assert target in ("block", "slot"), f"target must be 'block' or 'slot', got '{target}'"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    win = f"HSV Calibration — {target} | Camera {camera_index} | S: save, Q: quit"
    cv2.namedWindow(win)

    sliders = [
        ("H_low", 180, 0),
        ("S_low", 255, 50),
        ("V_low", 255, 50),
        ("H_high", 180, 180),
        ("S_high", 255, 255),
        ("V_high", 255, 255),
    ]
    for name, max_val, default in sliders:
        cv2.createTrackbar(name, win, default, max_val, lambda _: None)

    print(f"\n[Calibration] Target: {target} | Camera: {camera_index}")
    print("  1. Point the top-down camera at the target object.")
    print("  2. Adjust H/S/V sliders until only the target is shown in white.")
    print("  3. Press S to save, Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lower = np.array(
            [
                cv2.getTrackbarPos("H_low", win),
                cv2.getTrackbarPos("S_low", win),
                cv2.getTrackbarPos("V_low", win),
            ],
            dtype=np.uint8,
        )
        upper = np.array(
            [
                cv2.getTrackbarPos("H_high", win),
                cv2.getTrackbarPos("S_high", win),
                cv2.getTrackbarPos("V_high", win),
            ],
            dtype=np.uint8,
        )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        masked = cv2.bitwise_and(frame, frame, mask=mask)

        # Side-by-side: original | masked
        display = np.hstack([frame, masked])
        cv2.imshow(win, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            save_calibration(target, lower, upper)
            break
        elif key == ord("q"):
            print("[Calibration] Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


def run_verify(camera_index: int, include_angle: bool = False) -> None:
    """Show live detection overlay to verify calibration quality.

    Args:
        camera_index: OpenCV camera index for the top-down webcam.
        include_angle: If True, show angle values in the overlay.
    """
    detector = TopCameraDetector(include_angle=include_angle)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print("\n[Verify] Green = block | Red = slot | Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vis = detector.draw_overlay(frame)
        cv2.imshow(f"Detection verify — camera {camera_index} | Q to quit", vis)
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
    parser.add_argument("--calibrate", action="store_true", help="Run interactive HSV slider calibration.")
    parser.add_argument("--verify", action="store_true", help="Show live detection overlay.")
    parser.add_argument(
        "--target",
        choices=["block", "slot"],
        default="block",
        help="Which object to calibrate (only used with --calibrate).",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=5,
        help="Camera index for the top-down webcam (default: 5).",
    )
    parser.add_argument(
        "--include_angle",
        action="store_true",
        help="Show/compute rotation angles in verify mode (10-float token instead of 8).",
    )
    args = parser.parse_args()

    if args.calibrate:
        run_calibration(args.target, args.camera_index)
    elif args.verify:
        run_verify(args.camera_index, include_angle=args.include_angle)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
