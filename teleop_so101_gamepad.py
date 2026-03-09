#!/usr/bin/env python
"""
Gamepad teleoperation for SO-101 follower arm.

Each analog stick axis maps to one joint. Use --map to discover
your controller's axis/button layout before driving the arm.

Default mapping (generic gamepad):
  Left  stick X  (axis 0) → shoulder_pan
  Left  stick Y  (axis 1) → shoulder_lift  (push up = positive)
  Right stick X  (axis 2) → elbow_flex
  Right stick Y  (axis 3) → wrist_flex     (push up = positive)
  LB   (button 4)         → wrist_roll CCW
  RB   (button 5)         → wrist_roll CW
  LT   (button 6)         → gripper close
  RT   (button 7)         → gripper open

Usage:
  # Step 1 — discover your controller layout (arm stays still):
  python teleop_so101_gamepad.py --port /dev/ttyACM0 --id my_follower_arm --map

  # Step 2 — drive the arm:
  python teleop_so101_gamepad.py --port /dev/ttyACM0 --id my_follower_arm
"""

import argparse
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # no display needed
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

# ── Tuning constants ──────────────────────────────────────────────────────────
STEP     = 3.0   # max position units moved per second at full stick deflection
DEADZONE = 0.1   # ignore stick values below this magnitude
FPS      = 30

# ── Default axis → joint mapping ──────────────────────────────────────────────
# Format: axis_index → (joint_name, direction_sign)
# Invert sign (-1) if the joint moves the wrong way.
AXIS_MAP = {
    0: ("shoulder_pan",  +1),
    1: ("shoulder_lift", -1),   # Y axes are usually inverted
    2: ("elbow_flex",    +1),
    3: ("wrist_flex",    -1),
}

# ── Button → joint mapping ────────────────────────────────────────────────────
BTN_WRIST_CCW     = 4   # LB
BTN_WRIST_CW      = 5   # RB
BTN_GRIPPER_CLOSE = 6   # LT (often a button on generic pads)
BTN_GRIPPER_OPEN  = 7   # RT

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    # Rescale so motion starts at 0 just outside the deadzone
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def init_gamepad() -> pygame.joystick.JoystickType:
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count == 0:
        print("No gamepad detected. Plug it in and try again.")
        sys.exit(1)
    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"Controller: {joy.get_name()}")
    print(f"  Axes: {joy.get_numaxes()}   Buttons: {joy.get_numbuttons()}\n")
    return joy


def map_mode(joy: pygame.joystick.JoystickType):
    """Print live axis and button values so the user can identify their controller layout."""
    print("MAP MODE — move sticks and press buttons to discover their indices.")
    print("Press Ctrl+C to exit.\n")
    try:
        while True:
            pygame.event.pump()
            axes   = [round(joy.get_axis(i), 2) for i in range(joy.get_numaxes())]
            buttons = [joy.get_button(i) for i in range(joy.get_numbuttons())]
            active_axes    = {i: v for i, v in enumerate(axes) if abs(v) > DEADZONE}
            active_buttons = [i for i, v in enumerate(buttons) if v]
            parts = []
            if active_axes:
                parts.append("Axes: " + ", ".join(f"{i}={v:+.2f}" for i, v in active_axes.items()))
            if active_buttons:
                parts.append("Buttons: " + str(active_buttons))
            if parts:
                print("\r" + "  |  ".join(parts) + "          ", end="", flush=True)
            time.sleep(1 / FPS)
    except KeyboardInterrupt:
        print("\nDone mapping.")


def run(port: str, robot_id: str):
    joy = init_gamepad()

    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
    robot.connect()

    obs = robot.get_observation()
    current_pos = {motor: obs[f"{motor}.pos"] for motor in MOTORS}

    print("Gamepad teleoperation active. Press Ctrl+C to quit.\n")
    print("  Left  stick → shoulder_pan / shoulder_lift")
    print("  Right stick → elbow_flex  / wrist_flex")
    print("  LB / RB     → wrist_roll (CCW / CW)")
    print("  LT / RT     → gripper (close / open)\n")
    print("  Tip: edit AXIS_MAP in the script if joints move the wrong way.\n")

    try:
        while True:
            loop_start = time.perf_counter()
            pygame.event.pump()

            dt = 1.0 / FPS  # seconds per tick

            # ── Analog stick → joint increments ──────────────────────────────
            for axis_idx, (joint, sign) in AXIS_MAP.items():
                if axis_idx < joy.get_numaxes():
                    raw = joy.get_axis(axis_idx)
                    val = apply_deadzone(raw, DEADZONE)
                    current_pos[joint] += sign * val * STEP * dt * FPS
                    current_pos[joint] = max(-100.0, min(100.0, current_pos[joint]))

            # ── Buttons → wrist_roll ──────────────────────────────────────────
            if BTN_WRIST_CCW < joy.get_numbuttons() and joy.get_button(BTN_WRIST_CCW):
                current_pos["wrist_roll"] -= STEP
            if BTN_WRIST_CW < joy.get_numbuttons() and joy.get_button(BTN_WRIST_CW):
                current_pos["wrist_roll"] += STEP
            current_pos["wrist_roll"] = max(-100.0, min(100.0, current_pos["wrist_roll"]))

            # ── Buttons → gripper ─────────────────────────────────────────────
            if BTN_GRIPPER_CLOSE < joy.get_numbuttons() and joy.get_button(BTN_GRIPPER_CLOSE):
                current_pos["gripper"] -= STEP
            if BTN_GRIPPER_OPEN < joy.get_numbuttons() and joy.get_button(BTN_GRIPPER_OPEN):
                current_pos["gripper"] += STEP
            current_pos["gripper"] = max(0.0, min(100.0, current_pos["gripper"]))

            # ── Send to robot ─────────────────────────────────────────────────
            action = {f"{motor}.pos": val for motor, val in current_pos.items()}
            robot.send_action(action)

            # ── Live display ──────────────────────────────────────────────────
            status = "  ".join(f"{m[:4]}={current_pos[m]:+6.1f}" for m in MOTORS)
            print(f"\r{status}", end="", flush=True)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1.0 / FPS - dt_s, 0.0))

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.disconnect()
        pygame.quit()
        print("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Robot USB port, e.g. /dev/ttyACM0")
    parser.add_argument("--id", default="my_follower_arm", help="Robot ID (must match calibration)")
    parser.add_argument("--map", action="store_true",
                        help="Print axis/button values without moving the arm")
    args = parser.parse_args()

    joy = init_gamepad()

    if args.map:
        map_mode(joy)
    else:
        run(args.port, args.id)
