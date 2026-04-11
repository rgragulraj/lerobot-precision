#!/bin/bash
# Policy 1b — Session C (50 episodes)
# Vary: block position, slot location, lighting, camera tilt, shapes between episodes
# Reset time: 15s

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/policy1_diverse_session_c \
    "--dataset.single_task=Pick up the block and hover above the slot" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=false \
    --display_data=true
