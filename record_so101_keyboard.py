#!/usr/bin/env python
"""
Keyboard-controlled dataset recording for SO-101 follower arm with camera.

Usage:
    python record_so101_keyboard.py \
        --port /dev/ttyACM0 \
        --id my_follower_arm \
        --dataset-name so101_pick_place \
        --num-episodes 20 \
        --task "Pick up the red cube and place it in the box" \
        --episode-time 60 \
        --reset-time 15 \
        --camera /dev/video4
"""

import argparse
import logging
import time
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STEP = 2.0   # normalised position units per control tick — reduce if arm shakes
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


def run(port, robot_id, dataset_name, num_episodes, task, episode_time_s, reset_time_s, camera):
    # ── Robot & camera setup ──────────────────────────────────────────────────
    # Accept either an integer index or a device path like /dev/video4
    camera_id = int(camera) if camera.isdigit() else Path(camera)
    camera_config = {
        "front": OpenCVCameraConfig(
            index_or_path=camera_id,
            width=640,
            height=480,
            fps=FPS,
        )
    }
    robot_config = SO101FollowerConfig(
        port=port,
        id=robot_id,
        cameras=camera_config,
    )
    robot = SO101Follower(robot_config)

    teleop_config = KeyboardTeleopConfig()
    teleop = KeyboardTeleop(teleop_config)

    robot.connect()
    teleop.connect()

    # ── Dataset setup ─────────────────────────────────────────────────────────
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
        use_videos=True,
        image_writer_threads=4,
    )

    logger.info(f"Dataset will be saved to: {dataset_root}")
    logger.info(f"Task: {task}")
    print("\nControls:")
    print("  q/e = shoulder_pan    w/s = shoulder_lift   a/d = elbow_flex")
    print("  r/f = wrist_flex      t/g = wrist_roll      z/x = gripper")
    print("  Enter = save episode  Backspace = discard   Esc = quit\n")

    episode_idx = 0

    try:
        while episode_idx < num_episodes and teleop.is_connected:
            print(f"\n{'='*50}")
            print(f"Episode {episode_idx + 1}/{num_episodes}")
            print("Arrange the scene, then press any movement key to start recording.")

            obs = robot.get_observation()
            current_pos = {motor: obs[f"{motor}.pos"] for motor in MOTORS}

            frames = []
            recording = False
            save_episode = False
            discard_episode = False
            episode_start_t = None

            # ── Episode loop ──────────────────────────────────────────────────
            while teleop.is_connected:
                loop_start = time.perf_counter()

                pressed = teleop.get_action()

                # Control keys
                if "\r" in pressed or "\n" in pressed:
                    save_episode = True
                    break
                if "\x7f" in pressed or "\x08" in pressed:
                    discard_episode = True
                    break

                # Apply joint deltas
                moved = False
                for key in pressed:
                    if key in KEY_TO_JOINT_DELTA:
                        moved = True
                        for motor, delta in KEY_TO_JOINT_DELTA[key].items():
                            current_pos[motor] += delta
                            if motor == "gripper":
                                current_pos[motor] = max(0.0, min(100.0, current_pos[motor]))
                            else:
                                current_pos[motor] = max(-100.0, min(100.0, current_pos[motor]))

                if moved and not recording:
                    recording = True
                    episode_start_t = time.perf_counter()
                    logger.info("Recording started.")

                action = {f"{motor}.pos": val for motor, val in current_pos.items()}
                sent_action = robot.send_action(action)

                if recording:
                    obs = robot.get_observation()
                    obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
                    action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
                    frames.append({**obs_frame, **action_frame})

                    if time.perf_counter() - episode_start_t >= episode_time_s:
                        logger.info("Episode time limit reached. Saving.")
                        save_episode = True
                        break

                dt_s = time.perf_counter() - loop_start
                precise_sleep(max(1.0 / FPS - dt_s, 0.0))

            # ── Save or discard ───────────────────────────────────────────────
            if discard_episode or not frames:
                logger.info("Episode discarded.")
                continue

            if save_episode and frames:
                logger.info(f"Saving episode {episode_idx} with {len(frames)} frames...")
                for frame in frames:
                    dataset.add_frame({**frame, "task": task})
                dataset.save_episode()
                logger.info("Episode saved.")
                episode_idx += 1

                if episode_idx < num_episodes:
                    print(f"\nReset the scene. Resuming in {reset_time_s}s...")
                    time.sleep(reset_time_s)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        teleop.disconnect()
        robot.disconnect()
        logger.info(f"Done. {episode_idx} episodes saved to: {dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="my_follower_arm")
    parser.add_argument("--dataset-name", default="so101_pick_place")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--task", default="Pick up the red cube and place it in the box")
    parser.add_argument("--episode-time", type=float, default=60.0, dest="episode_time_s")
    parser.add_argument("--reset-time", type=float, default=15.0, dest="reset_time_s")
    parser.add_argument("--camera", default="/dev/video4",
                        help="Camera device path (e.g. /dev/video4) or integer index")
    args = parser.parse_args()
    run(
        port=args.port,
        robot_id=args.id,
        dataset_name=args.dataset_name,
        num_episodes=args.num_episodes,
        task=args.task,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        camera=args.camera,
    )