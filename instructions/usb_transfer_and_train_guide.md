# Policy 1 Phase 1b — USB Transfer & Training Guide

**Dataset:** `rgragulraj/policy1_diverse_all` (150 episodes, ~1.3 GB)
**Phase:** Phase 1b — systematic data diversity training
**Training steps:** 100,000
**Hub push:** NEVER — all steps use `--policy.push_to_hub=false`

---

## Overview

```
[This laptop]                          [Remote PC]
  Copy dataset → USB → Copy dataset
  Copy repo    → USB → Clone / copy repo
                                        Install deps
                                        Train ACT
  Copy model   ← USB ← Copy checkpoint
  Run eval
```

---

## Step 1 — On this laptop: copy dataset and repo to USB

Plug in the USB. Find its mount name:

```bash
ls /media/rgragulraj/
```

Copy the merged dataset (~1.3 GB):

```bash
cp -r ~/.cache/huggingface/lerobot/rgragulraj/policy1_diverse_all \
    /media/rgragulraj/<your-usb-name>/policy1_diverse_all
```

Copy this instructions file to USB root so you can read it on the remote PC:

```bash
cp ~/lerobot-precision/instructions/usb_transfer_and_train_guide.md \
    /media/rgragulraj/<your-usb-name>/
```

> **Why not just clone lerobot from HuggingFace on the remote PC?**
> This project (`lerobot-precision`) has custom code changes that are required for
> Phase 1b training — expanded augmentation ranges, RandomResizedCrop, RandomErasing,
> and the backbone unfreeze scaffolding. The remote PC must use this exact repo.

If the remote PC has internet access, you can clone from GitHub there (skip the repo copy).
If it does NOT have internet, also copy the repo:

```bash
# Only needed if remote PC has no internet:
cp -r ~/lerobot-precision \
    /media/rgragulraj/<your-usb-name>/lerobot-precision
```

Safely eject:

```bash
sync && udisksctl unmount -b /dev/sdX   # replace sdX with your USB device
```

---

## Step 2 — On the remote PC: set up the environment

Plug in the USB. Find its mount name:

```bash
ls /media/<remote-user>/
# or: ls /run/media/<remote-user>/
```

**Option A — Remote PC has internet:**

```bash
git clone https://github.com/<your-github-username>/lerobot-precision.git
cd lerobot-precision
```

**Option B — Remote PC has no internet (copy from USB):**

```bash
cp -r /media/<remote-user>/<your-usb-name>/lerobot-precision ~/lerobot-precision
cd ~/lerobot-precision
```

Create and activate the conda environment:

```bash
conda create -n lerobot python=3.11 -y
conda activate lerobot
pip install -e ".[act]"
```

Verify the install worked:

```bash
lerobot-train --help
```

---

## Step 3 — On the remote PC: copy dataset from USB

```bash
mkdir -p ~/.cache/huggingface/lerobot/rgragulraj/
cp -r /media/<remote-user>/<your-usb-name>/policy1_diverse_all \
    ~/.cache/huggingface/lerobot/rgragulraj/policy1_diverse_all
```

Verify it landed correctly (should show 150 episodes, ~101 617 frames):

```bash
python3 -c "
import json, pathlib
p = pathlib.Path.home() / '.cache/huggingface/lerobot/rgragulraj/policy1_diverse_all'
info = json.loads((p / 'meta' / 'info.json').read_text())
print('episodes:', info['total_episodes'])
print('frames:  ', info['total_frames'])
"
```

Verify the stats file is present (stale or missing stats cause silent training failures):

```bash
ls ~/.cache/huggingface/lerobot/rgragulraj/policy1_diverse_all/meta/stats.json
```

If the file is missing, recompute stats before training:

```bash
python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
ds = LeRobotDataset(
    'rgragulraj/policy1_diverse_all',
    root=Path.home() / '.cache/huggingface/lerobot/rgragulraj/policy1_diverse_all',
)
print('Dataset loaded OK, features:', list(ds.meta.features.keys()))
"
```

---

## Step 4 — On the remote PC: train

Make sure the conda environment is active:

```bash
conda activate lerobot
cd ~/lerobot-precision
```

Run Phase 1b training. **Do not remove `--policy.push_to_hub=false`.**

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_diverse_all \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/policy1_diverse_all \
  --dataset.image_transforms.enable=true \
  --policy.push_to_hub=false \
  --training.num_train_steps=100000 \
  --output_dir=outputs/policy1_phase1b
```

> **What the augmentation flag does:** `--dataset.image_transforms.enable=true` activates the
> Phase 1a augmentation ranges already baked into the source code in `transforms.py`:
> brightness (0.5–1.5), contrast (0.5–2.0), hue (±0.1), affine translate (0.1),
> RandomResizedCrop (scale 0.85–1.0), RandomErasing (p=0.3).
> No need to pass individual min_max flags — they are already at Phase 1a values.

> **Why NOT `--policy.unfreeze_backbone_layers`:** This option (Phase 1c) is only safe with
>
> > 500 episodes. This dataset has 150 episodes — skip it.

Checkpoints save to:

```
outputs/policy1_phase1b/checkpoints/
```

The final model lands at:

```
outputs/policy1_phase1b/checkpoints/last/pretrained_model/
```

Training progress is logged to the terminal. Expect ~2–4 hours on a modern GPU (RTX 3090/4090).
If you need to resume after an interruption:

```bash
lerobot-train \
  --config_path=outputs/policy1_phase1b/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

---

## Step 5 — On the remote PC: copy model to USB

Plug in the USB.

```bash
cp -r outputs/policy1_phase1b/checkpoints/last/pretrained_model \
    /media/<remote-user>/<your-usb-name>/policy1_phase1b_model
```

Also copy the full checkpoint folder if you want to resume training later:

```bash
cp -r outputs/policy1_phase1b/checkpoints/last \
    /media/<remote-user>/<your-usb-name>/policy1_phase1b_checkpoint_last
```

Safely eject:

```bash
sync && udisksctl unmount -b /dev/sdX
```

---

## Step 6 — Back on this laptop: copy model from USB

Plug the USB back into the laptop.

```bash
mkdir -p ~/models/
cp -r /media/rgragulraj/<your-usb-name>/policy1_phase1b_model \
    ~/models/policy1_phase1b
```

Verify the model files are present:

```bash
ls ~/models/policy1_phase1b/
# Should contain: config.json, model.safetensors (or pytorch_model.bin), train_config.json
```

---

## Step 7 — Run eval on this laptop

Autonomous eval — follower arm only, no leader arm needed.

```bash
conda activate lerobot
cd ~/lerobot-precision

lerobot-eval \
  --policy.path=~/models/policy1_phase1b \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=vellai_kunjan \
  --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}, "top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 640, "height": 480}}'
```

Place the block at the start position, run the command. Press `Ctrl+C` to stop.

---

## Checklist

### On this laptop (before leaving)

- [ ] USB plugged in and mounted
- [ ] `policy1_diverse_all` copied to USB (verify: ~1.3 GB)
- [ ] This guide file copied to USB root
- [ ] `lerobot-precision` repo copied to USB (if remote PC has no internet)
- [ ] USB safely ejected

### On remote PC

- [ ] USB mounted
- [ ] Dataset copied to `~/.cache/huggingface/lerobot/rgragulraj/policy1_diverse_all`
- [ ] Dataset verified: 150 episodes, 101 617 frames
- [ ] `meta/stats.json` present
- [ ] `lerobot-precision` cloned or copied
- [ ] `conda activate lerobot` confirmed
- [ ] Training command runs without errors
- [ ] `outputs/policy1_phase1b/checkpoints/last/pretrained_model/` exists after training
- [ ] Model copied to USB
- [ ] USB safely ejected

### Back on this laptop

- [ ] `~/models/policy1_phase1b/` exists
- [ ] `config.json` and `model.safetensors` present in that folder
- [ ] Eval runs

---

## Notes

- Replace `<your-usb-name>` with the actual mount name shown by `ls /media/rgragulraj/`
- Replace `<remote-user>` with the username on the remote PC
- Replace `sdX` in udisksctl commands with your actual USB device node (find it with `lsblk`)
- The `lerobot` conda env must be active whenever you run `lerobot-train` or `lerobot-eval`
- **Never** add `--push_to_hub=true` — all data and models stay local
- If training crashes on OOM: reduce batch size with `--training.batch_size=8` (default is 16 or 32 depending on config)
