#!/usr/bin/env python
"""
Keyboard teleoperation for SO-101 follower arm. No recording.

Controls:
  q/e = shoulder_pan      w/s = shoulder_lift   a/d = elbow_flex
  r/f = wrist_flex        t/g = wrist_roll       z/x = gripper
  Esc = quit

Usage:
    python teleop_so101_keyboard.py --port /dev/ttyACM0 --id my_follower_arm
"""

import argparse
import time

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

STEP = 2.0  # position units per tick — reduce to 0.5 if arm moves too fast
FPS = 30

KEY_TO_JOINT_DELTA = {
    "q": {"shoulder_pan": +STEP},
    "e": {"shoulder_pan": -STEP},
    "w": {"shoulder_lift": +STEP},
    "s": {"shoulder_lift": -STEP},
    "a": {"elbow_flex": +STEP},
    "d": {"elbow_flex": -STEP},
    "r": {"wrist_flex": +STEP},
    "f": {"wrist_flex": -STEP},
    "t": {"wrist_roll": +STEP},
    "g": {"wrist_roll": -STEP},
    "z": {"gripper": +STEP},
    "x": {"gripper": -STEP},
}

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def run(port, robot_id):
    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
    teleop = KeyboardTeleop(KeyboardTeleopConfig())

    robot.connect()
    teleop.connect()

    obs = robot.get_observation()
    current_pos = {motor: obs[f"{motor}.pos"] for motor in MOTORS}

    print("\nKeyboard teleoperation active. Press Esc to quit.\n")
    print("  q/e = shoulder_pan    w/s = shoulder_lift   a/d = elbow_flex")
    print("  r/f = wrist_flex      t/g = wrist_roll      z/x = gripper\n")

    try:
        while teleop.is_connected:
            loop_start = time.perf_counter()

            pressed = teleop.get_action()

            for key in pressed:
                if key in KEY_TO_JOINT_DELTA:
                    for motor, delta in KEY_TO_JOINT_DELTA[key].items():
                        current_pos[motor] += delta
                        if motor == "gripper":
                            current_pos[motor] = max(0.0, min(100.0, current_pos[motor]))
                        else:
                            current_pos[motor] = max(-100.0, min(100.0, current_pos[motor]))

            action = {f"{motor}.pos": val for motor, val in current_pos.items()}
            robot.send_action(action)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1.0 / FPS - dt_s, 0.0))

    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Robot USB port, e.g. /dev/ttyACM0")
    parser.add_argument("--id", default="my_follower_arm", help="Robot ID (must match calibration)")
    args = parser.parse_args()
    run(args.port, args.id)
