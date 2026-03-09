# SO-101 Kinesthetic Teaching + ACT Training

**Your setup:**
- SO-101 follower arm (no leader arm, no camera, no keyboard)
- NVIDIA GPU (CUDA) for training
- Everything stored locally

You physically move the arm by hand to record demonstrations. The arm's torque is disabled so it goes limp and you can guide it freely. Joint positions are recorded at 30 FPS and used to train an ACT policy.

> **Limitation:** Without a camera the policy cannot see the scene — it replays the learned joint trajectory regardless of where objects are. Keep the object in the **exact same position** for every episode and during evaluation.

---

## Prerequisites

```bash
cd ~/lerobot
pip install -e ".[feetech]"
pip install pynput
```

---

## Step 1: Find the Robot USB Port

Connect the arm to your PC via USB and power, then run:

```bash
lerobot-find-port
```

Unplug the USB cable when prompted, press Enter. Note the port (e.g. `/dev/ttyACM0`).

Grant port access if needed:

```bash
sudo chmod 666 /dev/ttyACM0
```

To make it permanent:

```bash
sudo usermod -aG dialout $USER
# then log out and back in
```

---

## Step 2: Set Up Motors (One-Time)

Assigns unique IDs and baudrates to each motor. Only needed once per arm.

```bash
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0
```

Connect one motor at a time starting from the gripper and press Enter after each prompt:

```
Connect the controller board to the 'gripper' motor only and press enter.
'gripper' motor id set to 6
Connect the controller board to the 'wrist_roll' motor only and press enter.
...
```

---

## Step 3: Calibrate the Arm (One-Time)

Use `--robot.id=vellai_kunjan` — this name must be the same in every command below.

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan
```

When prompted:

1. Move the arm to the **middle of its range** (roughly upright, relaxed) and press `Enter`.
2. **Slowly move every joint through its full range** (min → max), then press `Enter`.

---

## Step 4: Record Kinesthetic Demonstrations

```bash
python record_so101_kinesthetic.py \
    --port /dev/ttyACM0 \
    --id vellai_kunjan \
    --dataset-name so101_kinesthetic_task \
    --num-episodes 20 \
    --task "Pick up the red cube and place it in the box" \
    --episode-time 30 \
    --reset-time 10
```

| Argument | What it does |
|----------|-------------|
| `--num-episodes` | Total demonstrations to collect |
| `--task` | Text description of the task (stored in dataset metadata) |
| `--episode-time` | Max seconds per episode before auto-save |
| `--reset-time` | Seconds to wait between episodes for scene reset |

### What happens when you run it

1. The arm connects and **torque is disabled immediately** — you will feel it go limp.
2. Arrange the scene, then **physically guide the arm** through the task.
3. Press a control key when finished.

### Controls

| Key | Action |
|-----|--------|
| `Enter` | Save episode, wait reset timer, start next |
| `Backspace` | Discard episode and re-record |
| `Esc` | Exit |

### Tips for good data

- Aim for **at least 20–50 episodes**. More is better for ACT.
- Place the object in **exactly the same position** every episode.
- Move **smoothly and deliberately** — slow, clean demonstrations train better.
- Use `Backspace` to throw away shaky or failed attempts immediately.
- After `Enter`, the arm re-enables torque briefly (holds position) during reset, then goes limp again for the next episode.

Dataset is saved to `data/so101_kinesthetic_task/`.

---

## Step 5: Verify the Dataset

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/so101_kinesthetic_task", root="data/so101_kinesthetic_task")
print(f"Episodes : {dataset.num_episodes}")
print(f"Frames   : {dataset.num_frames}")
print(f"Features : {list(dataset.features.keys())}")
print(f"FPS      : {dataset.fps}")
```

Save as `check_kinesthetic_dataset.py` and run:

```bash
python check_kinesthetic_dataset.py
```

Expected output:

```
Episodes : 20
Frames   : <frames>
Features : ['observation.state', 'action']
FPS      : 30
```

You should see `observation.state` and `action` — **no image features** (that is correct).

---

## Step 6: Train ACT

```bash
lerobot-train \
    --dataset.repo_id=local/so101_kinesthetic_task \
    --dataset.root=data/so101_kinesthetic_task \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_kinesthetic \
    --job_name=act_so101_kinesthetic \
    --policy.device=cuda \
    --policy.push_to_hub=false
```

| Argument | Purpose |
|----------|---------|
| `--dataset.repo_id` | Matches the `repo_id` used during recording |
| `--dataset.root` | Local path where the dataset was saved |
| `--policy.type=act` | Action Chunking with Transformers |
| `--output_dir` | Where checkpoints are saved |
| `--policy.device=cuda` | Use your NVIDIA GPU |
| `--policy.push_to_hub=false` | Keep everything local |

Training is faster than with image data (no vision encoder to train).
Checkpoints are saved to:

```
outputs/train/act_so101_kinesthetic/checkpoints/last/
```

### Resume training from a checkpoint

```bash
lerobot-train \
    --config_path=outputs/train/act_so101_kinesthetic/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

### Optional: Monitor with Weights & Biases

```bash
pip install wandb && wandb login
```

Add `--wandb.enable=true` to the training command.

---

## Step 7: Evaluate the Policy on the Robot

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --policy.path=outputs/train/act_so101_kinesthetic/checkpoints/last/pretrained_model \
    --dataset.repo_id=local/eval_act_so101_kinesthetic \
    --dataset.root=data/eval_act_so101_kinesthetic \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick up the red cube and place it in the box" \
    --dataset.push_to_hub=false
```

Place the object in the **same position** you used during recording.
The arm will run the learned policy autonomously. Press `Ctrl+C` to stop if it moves dangerously.

---

## Full Workflow Summary

```
1. lerobot-find-port                          → note /dev/ttyACM0
2. lerobot-setup-motors                       → configure motor IDs (once)
3. lerobot-calibrate --robot.id=vellai_kunjan → calibrate arm (once)
4. python record_so101_kinesthetic.py         → physically guide arm, 20–50 demos
5. python check_kinesthetic_dataset.py        → verify data
6. lerobot-train                              → train ACT policy
7. lerobot-record --policy.path=...          → evaluate on robot
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied: /dev/ttyACM0` | `sudo chmod 666 /dev/ttyACM0` |
| Arm does not go limp | Check that calibration `--robot.id` matches exactly: `vellai_kunjan` |
| Arm moves on its own when limp | Normal — gravity. Support the arm with your hand before torque is disabled |
| `pynput` not capturing keys | Run from a desktop terminal with a display, not over SSH |
| Calibration ID mismatch | Use `--robot.id=vellai_kunjan` in every command |
| Training OOM (out of GPU memory) | Add `--training.batch_size=4` to the train command |
| Policy replays but misses the object | Object was not in the same position as during recording — reposition it |
| `module not found` errors | Run `pip install -e ".[feetech]"` from `~/lerobot` |
