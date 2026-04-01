# Data Collection Guide

How to record teleoperation datasets using the SO-101 arms and gripper camera for ACT training.

---

## 1. Dataset naming convention

Dataset names follow the format: `<hf_username>/<dataset_name>`

**Naming guidelines:**

- Use lowercase with underscores, no spaces
- Be descriptive: include the lab, object, and task
- Examples:
  - `rgragulraj/lenslab_square_pickplace` — pick and place a square at lens lab
  - `rgragulraj/lenslab_cube_stack` — stacking cubes at lens lab
  - `rgragulraj/lenslab_peg_insert` — peg insertion at lens lab

A consistent naming pattern makes it easy to identify datasets when training.

---

## 2. Where data is stored

Datasets are saved locally at:

```
~/.cache/huggingface/lerobot/<hf_username>/<dataset_name>/
```

Example:

```
~/.cache/huggingface/lerobot/rgragulraj/lenslab_square_pickplace/
```

Inside, the structure is:

```
<dataset_name>/
├── meta/                  # Dataset metadata and stats
├── data/                  # Parquet files (joint positions, actions per episode)
└── videos/                # MP4 files (one per camera per episode)
```

---

## 3. Record a dataset

Replace the values in `< >` for each new task.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/<dataset_name> \
    --dataset.single_task="<Short description of the task>" \
    --dataset.num_episodes=<number> \
    --dataset.episode_time_s=<seconds per episode> \
    --dataset.reset_time_s=<seconds to reset between episodes> \
    --dataset.push_to_hub=false \
    --display_data=true
```

**Current task — lenslab square pick and place:**

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --dataset.repo_id=rgragulraj/lenslab_square_pickplace \
    --dataset.single_task="Pick up the square and place it in the target zone" \
    --dataset.num_episodes=25 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=false \
    --display_data=true
```

---

## 4. Key parameters explained

| Parameter                  | What it does                                              |
| -------------------------- | --------------------------------------------------------- |
| `--dataset.repo_id`        | Dataset name in `username/dataset` format                 |
| `--dataset.single_task`    | Short text description of the task (used during training) |
| `--dataset.num_episodes`   | Total number of episodes to record                        |
| `--dataset.episode_time_s` | Max seconds per episode before auto-stopping              |
| `--dataset.reset_time_s`   | Seconds given to reset the scene between episodes         |
| `--dataset.push_to_hub`    | Upload to Hugging Face Hub when done (`true`/`false`)     |
| `--display_data`           | Show live camera feed and joint data during recording     |
| `--resume`                 | Continue recording into an existing dataset               |

---

## 5. Keyboard controls during recording

| Key             | Action                                                   |
| --------------- | -------------------------------------------------------- |
| `→` Right arrow | End current episode early and **save** it                |
| `←` Left arrow  | End current episode early and **discard** it (re-record) |
| `Esc`           | Stop recording entirely                                  |

Use `→` as soon as the task is complete — short, clean episodes train better than episodes padded with dead time.

---

## 6. Episode workflow

Each episode follows this cycle automatically:

1. **Recording phase** (`episode_time_s`) — teleoperate the task. Press `→` when done.
2. **Reset phase** (`reset_time_s`) — move objects back to start position. The arm is still live.
3. Repeat until `num_episodes` are saved.

---

## 7. Tips for good demonstrations

- Start every episode from the **same resting arm pose**.
- Place the object in the **same starting position** each time.
- Move **smoothly and deliberately** — no jerky or hesitant motions.
- Aim for each episode to take **5–15 seconds** of actual task time.
- If an episode goes wrong at any point, press `←` to discard and redo it.
- 25–50 episodes is a good starting point for ACT.

---

## 8. Resume interrupted recording

If recording is stopped before all episodes are done:

```bash
lerobot-record \
    ... <same arguments as before> ... \
    --resume=true
```

This picks up from the last saved episode without overwriting existing data.

---

## 9. Verify the dataset after recording

Check how many episodes were saved and inspect a sample:

```bash
python - <<'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "rgragulraj/lenslab_square_pickplace",
    root="~/.cache/huggingface/lerobot/rgragulraj/lenslab_square_pickplace"
)
print(f"Episodes: {dataset.num_episodes}")
print(f"Frames:   {dataset.num_frames}")
print(f"Features: {list(dataset.features.keys())}")
EOF
```

---

## 10. Upload to Hugging Face Hub (optional)

If you recorded with `--dataset.push_to_hub=false` and want to upload later:

```bash
python - <<'EOF'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "rgragulraj/lenslab_square_pickplace",
    root="~/.cache/huggingface/lerobot/rgragulraj/lenslab_square_pickplace"
)
dataset.push_to_hub()
EOF
```

---

## 11. Next step — Training

Once recording is done, follow `train_and_eval.md` to train ACT on the dataset.
