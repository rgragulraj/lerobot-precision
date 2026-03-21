#!/bin/bash

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}}' \
    --policy.path="$HOME/models/act_lenslab_square_pickplace/pretrained_model" \
    --dataset.repo_id=rgragulraj/eval_lenslab_square_pickplace \
    --dataset.single_task="Pick up the square and place it in the target zone" \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=30 \
    --dataset.push_to_hub=false \
    --display_data=true
