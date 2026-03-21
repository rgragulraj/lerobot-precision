# Train on Remote PC, Evaluate on Laptop

Full workflow: copy dataset via USB → train on remote PC → copy model back → run autonomous eval.

---

## Overview

```
[This laptop]                        [Remote PC]
Record dataset          →  USB  →    Copy dataset
                                     Train ACT
Copy trained model      ←  USB  ←    Save checkpoint
Run eval (arm moves autonomously)
```

---

## 1. Copy dataset to USB (on this laptop)

Plug in your USB drive, then find its mount name:

```bash
ls /media/rgragulraj/
```

Copy the dataset:

```bash
cp -r ~/.cache/huggingface/lerobot/rgragulraj/lenslab_square_pickplace \
    /media/rgragulraj/<your-usb-name>/lenslab_square_pickplace
```

---

## 2. Set up LeRobot on the remote PC

On the remote PC, clone and install:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[act]"
```

---

## 3. Copy dataset from USB to remote PC

Plug the USB into the remote PC, then:

```bash
mkdir -p ~/.cache/huggingface/lerobot/rgragulraj/
cp -r /media/<user>/<your-usb-name>/lenslab_square_pickplace \
    ~/.cache/huggingface/lerobot/rgragulraj/lenslab_square_pickplace
```

---

## 4. Train on the remote PC

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/lenslab_square_pickplace \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/lenslab_square_pickplace \
  --policy.push_to_hub=false \
  --output_dir=outputs/act_lenslab_square_pickplace
```

Checkpoints are saved to:
```
outputs/act_lenslab_square_pickplace/checkpoints/
```

Training will log progress to the terminal. The final model is at:
```
outputs/act_lenslab_square_pickplace/checkpoints/last/pretrained_model/
```

---

## 5. Copy trained model to USB (on remote PC)

```bash
cp -r outputs/act_lenslab_square_pickplace/checkpoints/last/pretrained_model \
    /media/<user>/<your-usb-name>/act_lenslab_square_pickplace
```

---

## 6. Copy trained model from USB to this laptop

Plug the USB back into the laptop, then:

```bash
mkdir -p ~/models/
cp -r /media/rgragulraj/<your-usb-name>/act_lenslab_square_pickplace \
    ~/models/act_lenslab_square_pickplace
```

---

## 7. Run autonomous eval on this laptop

No leader arm needed — the policy drives the follower arm on its own.

```bash
lerobot-eval \
  --policy.path=~/models/act_lenslab_square_pickplace \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=vellai_kunjan \
  --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}}'
```

Place the square in the start position, run the command, and the arm will attempt the task autonomously.
Press `Ctrl+C` to stop.

---

## Notes

- Replace `<your-usb-name>` with the actual mount name from `ls /media/rgragulraj/`
- Replace `<user>` on the remote PC with that machine's username
- Make sure the `lerobot` conda environment is activated on the remote PC before training
- The follower arm and gripper camera must be connected to this laptop before running eval
- If the arm behaves erratically during eval, collect more episodes and retrain
