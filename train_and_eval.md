# Train on PC, Evaluate on Laptop

---

## 1. Copy dataset to USB (on this laptop)

Plug in your USB drive, then:

```bash
cp -r ~/.cache/huggingface/lerobot/rgragulraj/LENS1_square /media/rgragulraj/<your-usb-name>/LENS1_square
```

---

## 2. Set up lerobot on the training PC

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[act]"
```

---

## 3. Copy dataset from USB to training PC

```bash
mkdir -p ~/.cache/huggingface/lerobot/rgragulraj/
cp -r /media/<user>/<your-usb-name>/LENS1_square ~/.cache/huggingface/lerobot/rgragulraj/LENS1_square
```

---

## 4. Train

```bash
lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/LENS1_square \
  --dataset.root=~/.cache/huggingface/lerobot/rgragulraj/LENS1_square \
  --policy.push_to_hub=false \
  --output_dir=outputs/act_LENS1_square
```

Training will save checkpoints to `outputs/act_LENS1_square/checkpoints/`.

---

## 5. Copy trained model back to USB

```bash
cp -r outputs/act_LENS1_square/checkpoints/last/pretrained_model /media/<user>/<your-usb-name>/act_LENS1_square
```

---

## 6. Evaluate on this laptop

Plug the USB back into the laptop, then:

```bash
~/miniforge3/envs/lerobot/bin/lerobot-eval \
  --policy.path=/media/rgragulraj/<your-usb-name>/act_LENS1_square \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower_arm \
  --robot.cameras='{"top": {"type": "opencv", "index_or_path": 5, "fps": 30, "width": 1280, "height": 720, "fourcc": "MJPG"}}'
```

The follower arm will run autonomously — no leader arm needed.
Press `Ctrl+C` to stop.

---

## Notes

- Replace `<your-usb-name>` with the actual mount name (check with `ls /media/rgragulraj/`)
- Replace `<user>` on the training PC with that machine's username
- Make sure the `lerobot` conda environment is activated on the training PC before training
- The follower arm and camera must be connected to the laptop before running eval
