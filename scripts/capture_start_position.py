"""
Capture and save the SO-101 follower arm's current joint positions as a named starting position.

Workflow:
  1. Connects to the follower arm using the SO101Follower class (loads calibration automatically).
  2. Disables torque so you can physically move the arm to the desired pose.
  3. Reads and displays the joint positions when you press ENTER.
  4. Saves the positions to instructions/start_positions/<name>.json.

Usage:
  python scripts/capture_start_position.py --name insert_above_slot
  python scripts/capture_start_position.py --name insert_above_slot --port /dev/ttyACM0 --id vellai_kunjan
"""

import argparse
import json
from pathlib import Path

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower


def capture(port: str, robot_id: str, name: str) -> None:
    # No cameras needed for position capture
    config = SO101FollowerConfig(port=port, id=robot_id)
    robot = SO101Follower(config)

    # connect() loads calibration from ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/<id>.json
    robot.connect(calibrate=False)
    print(f"Connected to {port} (calibration loaded for '{robot_id}').")

    robot.bus.disable_torque()
    print("\nTorque DISABLED — arm is free to move.")
    print("Manually position the arm to the desired starting pose:")
    print("  → above the insertion slot")
    print("  → gripper closed around the object")
    print()
    input("Press ENTER when the arm is in position...")

    # sync_read returns calibrated/normalised values (same units used by send_action)
    positions = robot.bus.sync_read("Present_Position")
    print(f"\nCaptured joint positions:\n{json.dumps(positions, indent=4)}")

    # Save
    out_dir = Path(__file__).parent.parent / "instructions" / "start_positions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"

    payload = {
        "name": name,
        "robot_id": robot_id,
        "port": port,
        "positions": positions,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nStarting position '{name}' saved to:\n  {out_path}")

    robot.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture a named starting position for the SO-101 follower arm."
    )
    parser.add_argument(
        "--name", required=True, help="Short label for this position (e.g. insert_above_slot)"
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for the follower arm")
    parser.add_argument(
        "--id", default="vellai_kunjan", dest="robot_id", help="Robot ID (used to find calibration file)"
    )
    args = parser.parse_args()

    capture(args.port, args.robot_id, args.name)
