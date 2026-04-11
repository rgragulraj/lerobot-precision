#!/bin/bash
# Eval Policy 1 Session A — autonomous run, no leader arm needed
# Press → to end episode early (save), ← to discard, Ctrl+C to stop

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --policy.path=/home/rgragulraj/models/policy1_session_a \
    --dataset.repo_id=rgragulraj/eval_session_a \
    "--dataset.single_task=Pick up the block and hover above the slot" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=false \
    --display_data=true
