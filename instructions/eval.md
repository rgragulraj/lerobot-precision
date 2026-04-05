# Running a Trained Policy on the Follower Arm

How to run an ACT policy on the SO-101 follower arm without a leader arm.

---

## Overview

There is no separate "run only" command in LeRobot for real hardware. Use `lerobot-record` with a `--policy.path` argument — the policy drives the arm autonomously and the session is recorded locally. Ignore or delete the dataset after if you don't need it.

---

## 1. Check hardware is connected

```bash
ls /dev/ttyACM* /dev/ttyUSB*
```

Expected output:

```
/dev/ttyACM0    ← follower arm
```

If nothing shows, the arm is not connected or not powered on.

---

## 2. Find the camera index

The camera index can change between sessions. Run:

```bash
ls /dev/video*
```

Or use the LeRobot helper:

```bash
lerobot-find-cameras
```

The gripper camera is typically `/dev/video5` or `/dev/video7`. Confirm visually with:

```bash
ffplay /dev/video5
```

---

## 3. Verify the model checkpoint exists

```bash
ls ~/models/act_lenslab_square_pickplace/pretrained_model/
```

Expected files:

```
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
train_config.json
```

---

## 4. Run the policy

Replace `index_or_path` with the correct camera index found in step 2.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}}' \
    --policy.path=/home/rgragulraj/models/act_lenslab_square_pickplace/pretrained_model \
    --dataset.repo_id=rgragulraj/eval_throwaway \
    --dataset.single_task="Pick up the square and place it in the target zone" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=false \
    --display_data=true
```

---

## 5. What happens during a session

1. The arm connects and the camera feed opens on screen
2. Press **Enter** to start each episode
3. The policy reads the camera + joint positions and moves the arm autonomously
4. After `episode_time_s` seconds, the episode ends and the reset timer starts
5. Reset the scene (move the square back to start position) during reset time
6. Repeat for `num_episodes` total

---

## 6. Keyboard controls

| Key      | Action                           |
| -------- | -------------------------------- |
| `→`      | End episode early and save it    |
| `←`      | End episode early and discard it |
| `Ctrl+C` | Stop the session entirely        |

---

## 7. After the session

The recorded data is saved locally at:

```
~/.cache/huggingface/lerobot/rgragulraj/eval_throwaway/
```

Delete it if you don't need it:

```bash
rm -rf ~/.cache/huggingface/lerobot/rgragulraj/eval_throwaway
```

---

## Notes

- No leader arm is needed — the policy drives the follower arm on its own
- Dataset repo*id \*\*must start with `eval*`\*\* when a policy is provided — LeRobot enforces this
- `~` in `--policy.path` is not expanded by the argument parser — always use the full path (`/home/rgragulraj/...`)
- If the arm behaves erratically, collect more episodes and retrain
- `episode_time_s=30` is a starting point — adjust to how long your task actually takes
- Success is judged manually by watching the arm
