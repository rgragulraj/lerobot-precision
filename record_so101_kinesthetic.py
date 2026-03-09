#!/usr/bin/env python
"""
Kinesthetic (hand-guided) dataset recording for SO-101 follower arm.
No camera, no keyboard/gamepad required.

The arm's torque is disabled so you can physically move it by hand.
Joint positions are read continuously and saved as the dataset.
The recorded positions become both the observation AND the action
(i.e. the policy replays the exact joint trajectory you demonstrated).

Usage:
    python record_so101_kinesthetic.py \
        --port /dev/ttyACM0 \
        --id vellai_kunjan \
        --dataset-name so101_kinesthetic_task \
        --num-episodes 20 \
        --task "Pick up the red cube and place it in the box" \
        --episode-time 30 \
        --reset-time 10

Controls:
    Enter      → save episode and start reset timer
    Backspace  → discard episode and re-record
    Esc        → exit

NOTE: Without a camera the ACT policy learns joint trajectories only and
cannot generalise to different object positions. Best for tasks where the
object is always in the same place.
"""

import argparse
import logging
import time
import threading
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FPS = 30
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


# ── Non-blocking key listener using pynput ────────────────────────────────────

class KeyListener:
    """Tracks the last special key pressed (Enter / Backspace / Esc)."""

    def __init__(self):
        self._last = None
        self._lock = threading.Lock()
        self._listener = None

    def start(self):
        from pynput import keyboard

        def on_press(key):
            from pynput.keyboard import Key
            with self._lock:
                if key == Key.enter:
                    self._last = "enter"
                elif key in (Key.backspace, Key.delete):
                    self._last = "backspace"
                elif key == Key.esc:
                    self._last = "esc"

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

    def pop(self):
        """Return and clear the last key press (or None)."""
        with self._lock:
            key = self._last
            self._last = None
        return key

    def stop(self):
        if self._listener:
            self._listener.stop()


# ── Main recording logic ───────────────────────────────────────────────────────

def run(port, robot_id, dataset_name, num_episodes, task, episode_time_s, reset_time_s):
    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
    robot.connect()

    # Build dataset features (state-only, no images)
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset_root = Path("data") / dataset_name
    repo_id = f"local/{dataset_name}"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        root=dataset_root,
        use_videos=False,
    )

    logger.info(f"Dataset will be saved to: {dataset_root}")
    logger.info(f"Task: {task}")
    print("\nControls:")
    print("  Enter      = save episode")
    print("  Backspace  = discard episode")
    print("  Esc        = quit\n")

    keys = KeyListener()
    keys.start()

    episode_idx = 0

    try:
        while episode_idx < num_episodes:
            print(f"\n{'='*50}")
            print(f"Episode {episode_idx + 1}/{num_episodes}")
            print("Arrange the scene, then physically move the arm to demonstrate the task.")
            print("The arm is now PASSIVE (torque off). Press Enter when done.\n")

            # Disable torque — arm goes limp
            robot.bus.disable_torque(None)

            frames = []
            recording = True
            episode_start_t = time.perf_counter()

            # ── Episode recording loop ─────────────────────────────────────────
            while recording:
                loop_start = time.perf_counter()

                key = keys.pop()
                if key == "esc":
                    print("\nExit requested.")
                    robot.bus.enable_torque(None)
                    return

                if key == "enter":
                    logger.info("Saving episode...")
                    recording = False
                    save = True
                elif key == "backspace":
                    logger.info("Episode discarded.")
                    recording = False
                    save = False
                else:
                    save = None  # still recording

                # Read joint positions
                obs = robot.get_observation()

                # For kinesthetic teaching: action = observation
                obs_frame  = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
                act_frame  = build_dataset_frame(dataset.features, obs, prefix=ACTION)
                frames.append({**obs_frame, **act_frame})

                # Live readout
                status = "  ".join(f"{m[:4]}={obs[f'{m}.pos']:+6.1f}" for m in MOTORS)
                print(f"\r{status}  frames={len(frames):4d}", end="", flush=True)

                # Auto-save when time limit reached
                if time.perf_counter() - episode_start_t >= episode_time_s:
                    logger.info("\nEpisode time limit reached. Saving.")
                    recording = False
                    save = True

                dt_s = time.perf_counter() - loop_start
                precise_sleep(max(1.0 / FPS - dt_s, 0.0))

            # ── Save or discard ────────────────────────────────────────────────
            print()  # newline after live readout

            # Re-enable torque so arm holds position during reset
            robot.bus.enable_torque(None)

            if not save or not frames:
                continue

            logger.info(f"Saving {len(frames)} frames for episode {episode_idx}...")
            for frame in frames:
                dataset.add_frame({**frame, "task": task})
            dataset.save_episode()
            logger.info("Episode saved.")
            episode_idx += 1

            if episode_idx < num_episodes:
                print(f"\nReset the scene. Next episode starts in {reset_time_s}s...")
                time.sleep(reset_time_s)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        keys.stop()
        # Make sure torque is re-enabled before disconnecting
        try:
            robot.bus.enable_torque(None)
        except Exception:
            pass
        dataset.consolidate()
        robot.disconnect()
        logger.info(f"Done. {episode_idx} episodes saved to: {dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Robot USB port, e.g. /dev/ttyACM0")
    parser.add_argument("--id", default="vellai_kunjan", help="Robot ID (must match calibration)")
    parser.add_argument("--dataset-name", default="so101_kinesthetic_task")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--task", default="Pick up the red cube and place it in the box")
    parser.add_argument("--episode-time", type=float, default=30.0, dest="episode_time_s",
                        help="Max seconds per episode before auto-save")
    parser.add_argument("--reset-time", type=float, default=10.0, dest="reset_time_s",
                        help="Seconds to wait between episodes for scene reset")
    args = parser.parse_args()
    run(
        port=args.port,
        robot_id=args.id,
        dataset_name=args.dataset_name,
        num_episodes=args.num_episodes,
        task=args.task,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
    )
