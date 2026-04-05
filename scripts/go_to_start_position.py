"""
Move the SO-101 follower arm to a previously saved starting position.

Use this during the reset phase between recording episodes to return
the arm to the exact starting pose before the next episode begins.

Usage:
  python scripts/go_to_start_position.py --name insert_above_slot
  python scripts/go_to_start_position.py --name insert_above_slot --port /dev/ttyACM0 --id vellai_kunjan
"""

import argparse
import json
import time
from pathlib import Path

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower


def go_to_start(port: str, robot_id: str, name: str, wait_s: float = 4.0) -> None:
    positions_path = Path(__file__).parent.parent / "instructions" / "start_positions" / f"{name}.json"
    if not positions_path.exists():
        raise FileNotFoundError(
            f"No saved position '{name}' found at {positions_path}.\n"
            f"Run capture_start_position.py --name {name} first."
        )

    payload = json.loads(positions_path.read_text())
    target = payload["positions"]
    print(f"Loaded start position '{name}': {target}")

    config = SO101FollowerConfig(port=port, id=robot_id, disable_torque_on_disconnect=False)
    robot = SO101Follower(config)
    robot.connect(calibrate=False)
    print(f"Connected to {port}.")

    # send_action expects keys like "shoulder_pan.pos"
    action = {f"{motor}.pos": val for motor, val in target.items()}
    print(f"Moving to start position... (waiting {wait_s}s)")
    robot.send_action(action)
    time.sleep(wait_s)

    final = robot.bus.sync_read("Present_Position")
    print(f"Reached: {final}")

    robot.disconnect()
    print("Done. Torque left ON to hold position.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move SO-101 follower to a saved starting position.")
    parser.add_argument("--name", required=True, help="Name of the saved starting position")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="vellai_kunjan", dest="robot_id")
    parser.add_argument("--wait", type=float, default=4.0, help="Seconds to wait for arm to reach position")
    args = parser.parse_args()

    go_to_start(args.port, args.robot_id, args.name, args.wait)
