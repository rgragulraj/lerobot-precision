#!/bin/bash
# Record block insertion episodes.
# Usage: bash scripts/record_insert.sh [num_episodes]
# Defaults to 30 episodes. Pass 2 for a test run.

NUM_EPISODES=${1:-30}
DATASET="rgragulraj/lenslab_block_insert"

if [ "$NUM_EPISODES" -le 3 ]; then
    DATASET="rgragulraj/lenslab_block_insert_test"
fi

echo "Recording $NUM_EPISODES episodes into $DATASET"

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=keyboard_joint \
    --teleop.start_position_file=instructions/start_positions/insert_above_slot.json \
    --dataset.repo_id="$DATASET" \
    --dataset.single_task="Insert the block into the slot" \
    --dataset.num_episodes="$NUM_EPISODES" \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=120 \
    --dataset.push_to_hub=false \
    --display_data=true
