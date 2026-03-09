# SO-101 Full Pipeline: Keyboard Teleop + Camera + Dataset Recording + ACT Training

**Your setup:**
- SO-101 follower arm (no leader arm)
- 1× iPhone camera via DroidCam app + OBS Studio (network)
- NVIDIA GPU (CUDA) for training
- Everything stored locally (no Hugging Face Hub upload)

This guide follows the [LeRobot IL Robots documentation](https://huggingface.co/docs/lerobot/il_robots),
the [SO-101 setup guide](https://huggingface.co/docs/lerobot/so101), and the
[LeRobot cameras guide](https://huggingface.co/docs/lerobot/cameras) as closely as possible.

---

## Prerequisites

### Install LeRobot

```bash
cd ~/lerobot
pip install -e ".[feetech]"
pip install pynput
```

---

## Step 0: Set Up iPhone Camera via DroidCam + OBS

> This section follows the LeRobot cameras guide exactly.

### 0.1 — Install v4l2loopback (Linux virtual camera driver)

```bash
sudo apt install v4l2loopback-dkms v4l-utils
```

This creates a virtual `/dev/videoX` device that OBS will stream into.

To load it now and on every boot:

```bash
sudo modprobe v4l2loopback
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
```

### 0.2 — Install DroidCam on your iPhone

Install the **[DroidCam](https://droidcam.app)** app from the App Store.
Make sure your iPhone and laptop are on the **same Wi-Fi network**.

### 0.3 — Install OBS Studio

```bash
sudo add-apt-repository ppa:obsproject/obs-studio
sudo apt update
sudo apt install obs-studio
```

### 0.4 — Install the DroidCam OBS Plugin

Download and install the plugin from **[droidcam.app/obs](https://droidcam.app/obs)**.

Follow the installation instructions on that page for Linux.

### 0.5 — Configure OBS

1. Open OBS Studio.
2. In the **Sources** panel, click `+` → select **DroidCam OBS**.
3. Follow the setup at [droidcam.app/obs/usage](https://droidcam.app/obs/usage):
   - Open the DroidCam app on your iPhone — it will show an **IP address and port**.
   - Enter that IP and port into the DroidCam OBS source settings.
   - Set resolution to **640×480** (important — avoids the free-tier watermark).
4. Go to `File > Settings > Video` and set **both** resolutions to `640x480`:
   - `Base (Canvas) Resolution` → `640x480`
   - `Output (Scaled) Resolution` → `640x480`
5. Click **OK**.

### 0.6 — Start the OBS Virtual Camera

In OBS Studio, click **`Start Virtual Camera`** (bottom-right panel).

### 0.7 — Find the Virtual Camera Device

```bash
v4l2-ctl --list-devices
```

The v4l2loopback virtual camera appears as **"Dummy video device"**. Example output:

```
Dummy video device (0x0000) (platform:v4l2loopback-000):
    /dev/video4

Integrated RGB Camera: Integrat (usb-...):
    /dev/video0
    /dev/video1
    ...
```

Your OBS virtual camera is at **`/dev/video4`**. Confirm the resolution:

```bash
v4l2-ctl -d /dev/video4 --get-fmt-video
```

You should see `width=640, height=480`.

Confirm lerobot can see it:

```bash
lerobot-find-cameras opencv
```

Your virtual camera will show `Id: /dev/video4`. Note: lerobot uses the **device path** as the ID, not an integer index. Use `/dev/video4` as `index_or_path` in all commands below.

> **Troubleshooting — wrong resolution error:**
> If you see `frame width/height do not match configured width/height`, the OBS virtual camera is
> advertising a different resolution than 640×480. Delete the source in OBS and recreate it.
> Resolution cannot be changed after creation.

---

## Step 1: Find the Robot USB Port

Connect the follower arm's controller board to your PC via USB and power. Run:

```bash
lerobot-find-port
```

Disconnect the USB cable when prompted and press Enter. Note the reported port (e.g. `/dev/ttyACM0`).

> **Linux:** You may need to grant port access:
> ```bash
> sudo chmod 666 /dev/ttyACM0
> ```
> To make it permanent: `sudo usermod -aG dialout $USER` then log out and back in.

---

## Step 2: Set Up Motors (One-Time)

Assigns unique IDs and baudrates to each motor. Only needed once per arm.

```bash
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0
```

Follow the prompts — connect one motor at a time starting from the gripper, press Enter after each:

```
Connect the controller board to the 'gripper' motor only and press enter.
'gripper' motor id set to 6
Connect the controller board to the 'wrist_roll' motor only and press enter.
...
```

---

## Step 3: Calibrate the Follower Arm (One-Time)

Choose an `--robot.id` name and **use the exact same name in every command in this guide** —
calibration is stored and looked up by this ID.

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan
```

When prompted:

1. **Move the arm to the middle of its range** (roughly upright, relaxed pose) and press `Enter`.
2. **Slowly move every joint through its full range of motion** (min → max), then press `Enter`.

---

## Step 4: Verify Camera + Arm Together

This step confirms the robot and camera both connect successfully.

> Make sure OBS Virtual Camera is running before this step.

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras="{front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
    --teleop.type=keyboard \
    --display_data=true
```

**What to look for in the output:**

```
INFO OpenCVCamera(/dev/video4) connected.     ← camera OK
INFO my_follower_arm SO101Follower connected. ← robot OK
```

If you see both lines, your setup is confirmed. The script will then immediately crash with `StopIteration` — **this is expected**. The built-in `--teleop.type=keyboard` sends raw key characters which are not valid joint commands, so `send_action` receives an empty action and errors out. The Rerun viewer will open but stay empty for the same reason.

Arm joint control is handled by the custom script in Step 5. Proceed once you see both connected lines.

---

## Step 4b: Teleoperate the Arm (No Recording)

Use these scripts to freely move the arm without recording a dataset — useful for testing,
practising movements, or checking that all joints respond correctly.

### Option A — Keyboard

```bash
python teleop_so101_keyboard.py \
    --port /dev/ttyACM0 \
    --id vellai_kunjan
```

#### Keyboard Controls

| Key | Joint | Direction |
|-----|-------|-----------|
| `q` | shoulder_pan | left |
| `e` | shoulder_pan | right |
| `w` | shoulder_lift | up |
| `s` | shoulder_lift | down |
| `a` | elbow_flex | bend |
| `d` | elbow_flex | extend |
| `r` | wrist_flex | up |
| `f` | wrist_flex | down |
| `t` | wrist_roll | CW |
| `g` | wrist_roll | CCW |
| `z` | gripper | open |
| `x` | gripper | close |
| `Esc` | — | quit |

> If the arm moves too fast, open `teleop_so101_keyboard.py` and reduce `STEP = 2.0` to `0.5` or `1.0`.

---

### Option B — Gamepad / Controller

First install pygame if you haven't already:

```bash
pip install pygame
```

#### Step 1 — Discover your controller layout (arm stays still)

Plug in your gamepad, then run:

```bash
python teleop_so101_gamepad.py --port /dev/ttyACM0 --id vellai_kunjan --map
```

Move each stick and press each button. The terminal will print which axis/button index is active.
Note the indices that correspond to your sticks and shoulder/trigger buttons.

#### Step 2 — Drive the arm

```bash
python teleop_so101_gamepad.py --port /dev/ttyACM0 --id vellai_kunjan
```

#### Default Gamepad Controls (generic controller)

| Input | Joint | Direction |
|-------|-------|-----------|
| Left stick X (axis 0) | shoulder_pan | left / right |
| Left stick Y (axis 1) | shoulder_lift | up / down |
| Right stick X (axis 2) | elbow_flex | bend / extend |
| Right stick Y (axis 3) | wrist_flex | up / down |
| LB (button 4) | wrist_roll | CCW |
| RB (button 5) | wrist_roll | CW |
| LT (button 6) | gripper | close |
| RT (button 7) | gripper | open |
| `Ctrl+C` | — | quit |

> **If a joint moves the wrong way:** open `teleop_so101_gamepad.py` and flip the sign in `AXIS_MAP`.
> For example, change `("shoulder_pan", +1)` to `("shoulder_pan", -1)`.
>
> **If your controller uses different axis/button indices:** update the numbers in `AXIS_MAP` and the
> `BTN_*` constants at the top of the script to match what `--map` mode reported.

---

## Step 5: Record a Dataset

The built-in `lerobot-record` CLI with keyboard teleoperation does not map keys to individual
arm joints. The script below stays fully within the LeRobot framework while adding that mapping.

### Key Controls During Recording

| Key | Robot action |
|-----|-------------|
| `q` / `e` | Shoulder pan — left / right |
| `w` / `s` | Shoulder lift — up / down |
| `a` / `d` | Elbow flex — bend / extend |
| `r` / `f` | Wrist flex — up / down |
| `t` / `g` | Wrist roll — rotate CW / CCW |
| `z` / `x` | Gripper — open / close |
| `Enter` | **Save episode** and start reset timer |
| `Backspace` | **Discard episode** and re-record |
| `Esc` | Exit |

### Recording Script

Save this as `record_so101_keyboard.py` in the lerobot root:

```python
#!/usr/bin/env python
"""
Keyboard-controlled dataset recording for SO-101 follower arm with camera.

Usage:
    python record_so101_keyboard.py \
        --port /dev/ttyACM0 \
        --id vellai_kunjan \
        --dataset-name so101_pick_place \
        --num-episodes 20 \
        --task "Pick up the red cube and place it in the box" \
        --episode-time 60 \
        --reset-time 15 \
        --camera /dev/video4
"""

import argparse
import logging
import time
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STEP = 2.0   # normalised position units per control tick — reduce if arm shakes
FPS = 30

KEY_TO_JOINT_DELTA = {
    "q": {"shoulder_pan": +STEP},
    "e": {"shoulder_pan": -STEP},
    "w": {"shoulder_lift": +STEP},
    "s": {"shoulder_lift": -STEP},
    "a": {"elbow_flex": +STEP},
    "d": {"elbow_flex": -STEP},
    "r": {"wrist_flex": +STEP},
    "f": {"wrist_flex": -STEP},
    "t": {"wrist_roll": +STEP},
    "g": {"wrist_roll": -STEP},
    "z": {"gripper": +STEP},
    "x": {"gripper": -STEP},
}

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def run(port, robot_id, dataset_name, num_episodes, task, episode_time_s, reset_time_s, camera):
    # ── Robot & camera setup ──────────────────────────────────────────────────
    # Accept either an integer index or a device path like /dev/video4
    camera_id = int(camera) if camera.isdigit() else Path(camera)
    camera_config = {
        "front": OpenCVCameraConfig(
            index_or_path=camera_id,
            width=640,
            height=480,
            fps=FPS,
        )
    }
    robot_config = SO101FollowerConfig(
        port=port,
        id=robot_id,
        cameras=camera_config,
    )
    robot = SO101Follower(robot_config)

    teleop_config = KeyboardTeleopConfig()
    teleop = KeyboardTeleop(teleop_config)

    robot.connect()
    teleop.connect()

    # ── Dataset setup ─────────────────────────────────────────────────────────
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset_root = Path("data") / dataset_name
    repo_id = f"local/{dataset_name}"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        root=dataset_root,
        use_videos=True,
        image_writer_threads=4,
    )

    logger.info(f"Dataset will be saved to: {dataset_root}")
    logger.info(f"Task: {task}")
    print("\nControls:")
    print("  q/e = shoulder_pan    w/s = shoulder_lift   a/d = elbow_flex")
    print("  r/f = wrist_flex      t/g = wrist_roll      z/x = gripper")
    print("  Enter = save episode  Backspace = discard   Esc = quit\n")

    episode_idx = 0

    try:
        while episode_idx < num_episodes and teleop.is_connected:
            print(f"\n{'='*50}")
            print(f"Episode {episode_idx + 1}/{num_episodes}")
            print("Arrange the scene, then press any movement key to start recording.")

            obs = robot.get_observation()
            current_pos = {motor: obs[f"{motor}.pos"] for motor in MOTORS}

            frames = []
            recording = False
            save_episode = False
            discard_episode = False
            episode_start_t = None

            # ── Episode loop ──────────────────────────────────────────────────
            while teleop.is_connected:
                loop_start = time.perf_counter()

                pressed = teleop.get_action()

                # Control keys
                if "\r" in pressed or "\n" in pressed:
                    save_episode = True
                    break
                if "\x7f" in pressed or "\x08" in pressed:
                    discard_episode = True
                    break

                # Apply joint deltas
                moved = False
                for key in pressed:
                    if key in KEY_TO_JOINT_DELTA:
                        moved = True
                        for motor, delta in KEY_TO_JOINT_DELTA[key].items():
                            current_pos[motor] += delta
                            if motor == "gripper":
                                current_pos[motor] = max(0.0, min(100.0, current_pos[motor]))
                            else:
                                current_pos[motor] = max(-100.0, min(100.0, current_pos[motor]))

                if moved and not recording:
                    recording = True
                    episode_start_t = time.perf_counter()
                    logger.info("Recording started.")

                action = {f"{motor}.pos": val for motor, val in current_pos.items()}
                sent_action = robot.send_action(action)

                if recording:
                    obs = robot.get_observation()
                    obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
                    action_frame = build_dataset_frame(dataset.features, sent_action, prefix=ACTION)
                    frames.append({**obs_frame, **action_frame})

                    if time.perf_counter() - episode_start_t >= episode_time_s:
                        logger.info("Episode time limit reached. Saving.")
                        save_episode = True
                        break

                dt_s = time.perf_counter() - loop_start
                precise_sleep(max(1.0 / FPS - dt_s, 0.0))

            # ── Save or discard ───────────────────────────────────────────────
            if discard_episode or not frames:
                logger.info("Episode discarded.")
                continue

            if save_episode and frames:
                logger.info(f"Saving episode {episode_idx} with {len(frames)} frames...")
                for frame in frames:
                    dataset.add_frame({**frame, "task": task})
                dataset.save_episode()
                logger.info("Episode saved.")
                episode_idx += 1

                if episode_idx < num_episodes:
                    print(f"\nReset the scene. Resuming in {reset_time_s}s...")
                    time.sleep(reset_time_s)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        teleop.disconnect()
        robot.disconnect()
        logger.info(f"Done. {episode_idx} episodes saved to: {dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="my_follower_arm")
    parser.add_argument("--dataset-name", default="so101_pick_place")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--task", default="Pick up the red cube and place it in the box")
    parser.add_argument("--episode-time", type=float, default=60.0, dest="episode_time_s")
    parser.add_argument("--reset-time", type=float, default=15.0, dest="reset_time_s")
    parser.add_argument("--camera", default="/dev/video4",
                        help="Camera device path (e.g. /dev/video4) or integer index")
    args = parser.parse_args()
    run(
        port=args.port,
        robot_id=args.id,
        dataset_name=args.dataset_name,
        num_episodes=args.num_episodes,
        task=args.task,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        camera=args.camera,
    )
```

### Run the Recording

> Make sure OBS Virtual Camera is running before this step.

```bash
python record_so101_keyboard.py \
    --port /dev/ttyACM0 \
    --id vellai_kunjan \
    --dataset-name so101_pick_place \
    --num-episodes 20 \
    --task "Pick up the red cube and place it in the box" \
    --episode-time 60 \
    --reset-time 15 \
    --camera /dev/video4
```

### Tips for Good Data

- Aim for **at least 20–50 episodes**. More is better for ACT.
- Keep the phone/camera position **fixed** throughout — tape it down if needed.
- Use `Backspace` to discard failed or messy attempts immediately.
- Vary the object position slightly between episodes so the policy generalises.
- Move **smoothly and deliberately** — slow, clean demonstrations train better policies.

Dataset is saved to `data/so101_pick_place/`.

---

## Step 6: Verify the Recorded Dataset

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/so101_pick_place", root="data/so101_pick_place")
print(f"Episodes : {dataset.num_episodes}")
print(f"Frames   : {dataset.num_frames}")
print(f"Features : {list(dataset.features.keys())}")
print(f"FPS      : {dataset.fps}")
```

Save as `check_dataset.py` and run:

```bash
python check_dataset.py
```

You should see your episode count, frame count, and features including `observation.images.front`,
`observation.state`, and `action`.

---

## Step 7: Train an ACT Policy

```bash
lerobot-train \
    --dataset.repo_id=local/so101_pick_place \
    --dataset.root=data/so101_pick_place \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_pick_place \
    --job_name=act_so101_pick_place \
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

Training takes several hours. Checkpoints are saved to:
```
outputs/train/act_so101_pick_place/checkpoints/last/
```

### Resume Training from a Checkpoint

```bash
lerobot-train \
    --config_path=outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

### Optional: Monitor with Weights & Biases

```bash
pip install wandb && wandb login
```

Add `--wandb.enable=true` to the training command.

---

## Step 8: Evaluate the Policy on the Robot

> Make sure OBS Virtual Camera is running before this step.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --robot.cameras="{front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
    --policy.path=outputs/train/act_so101_pick_place/checkpoints/last/pretrained_model \
    --dataset.repo_id=local/eval_act_so101_pick_place \
    --dataset.root=data/eval_act_so101_pick_place \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick up the red cube and place it in the box" \
    --dataset.push_to_hub=false \
    --display_data=true
```

The arm will run the learned policy autonomously. Press `Ctrl+C` to stop if it moves dangerously.

---

## Full Workflow Summary

```
0. iPhone → DroidCam app → OBS Virtual Camera → /dev/video4
1. lerobot-find-port                 → note /dev/ttyACM0
2. lerobot-setup-motors              → configure motor IDs (once)
3. lerobot-calibrate                 → calibrate arm (once)
4. lerobot-teleoperate               → verify camera + arm
5. python record_so101_keyboard.py   → collect 20–50 demonstrations
6. python check_dataset.py           → verify data
7. lerobot-train                     → train ACT policy (hours)
8. lerobot-record --policy.path=...  → evaluate on robot
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied: /dev/ttyACM0` | `sudo chmod 666 /dev/ttyACM0` |
| v4l2loopback not found | `sudo modprobe v4l2loopback` |
| OBS Virtual Camera not in device list | Click "Start Virtual Camera" in OBS first |
| Wrong resolution error from OpenCV | Delete and recreate the DroidCam source in OBS at 640×480 |
| Camera index not found | Run `lerobot-find-cameras opencv` to get the correct index |
| DroidCam shows no connection in OBS | Phone and laptop must be on same Wi-Fi; check IP in DroidCam app |
| Arm shakes or overshoots | Reduce `STEP = 2.0` to `0.5` or `1.0` in the script |
| Training OOM (out of GPU memory) | Add `--training.batch_size=4` to the train command |
| `pynput` not capturing keys | Must run from a desktop terminal with a display, not over SSH |
| Calibration ID mismatch | Use the exact same `--robot.id` string in every command |

---

---

## Alternative: Kinesthetic Teaching (No Camera, No Keyboard)

Kinesthetic teaching lets you **physically move the arm by hand** to record demonstrations.
No camera, no OBS, no input device needed — just your hands and the robot.

The arm's torque is disabled so it goes limp and you can guide it freely.
Joint positions are sampled at 30 FPS and saved as the dataset.
When the policy replays, it commands the robot to follow the exact same joint trajectory.

> **Limitation:** Without a camera the ACT policy cannot see the scene. It replays the
> learned joint trajectory regardless of where objects are placed. Best for tasks where
> the object is always in the same position. Add a camera later for visual generalisation.

### Prerequisites

```bash
pip install pynput   # already installed if you followed Step 0
```

No camera setup (OBS / DroidCam / v4l2loopback) needed.

---

### K-Step 1: Find Port & Setup Motors (same as Steps 1–2 above)

```bash
lerobot-find-port
sudo chmod 666 /dev/ttyACM0

lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0
```

---

### K-Step 2: Calibrate (same as Step 3 above)

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan
```

---

### K-Step 3: Record Kinesthetic Demonstrations

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

#### Controls

| Key | Action |
|-----|--------|
| `Enter` | Save episode and start reset timer |
| `Backspace` | Discard episode and re-record |
| `Esc` | Exit |

When the script starts, the arm goes **passive immediately** — you will feel the torque release.
Move it through the task, then press `Enter`. After the reset timer the arm stays stiff briefly,
then goes limp again for the next episode.

Dataset is saved to `data/so101_kinesthetic_task/` (state-only, no video files).

---

### K-Step 4: Verify the Dataset

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("local/so101_kinesthetic_task", root="data/so101_kinesthetic_task")
print(f"Episodes : {dataset.num_episodes}")
print(f"Frames   : {dataset.num_frames}")
print(f"Features : {list(dataset.features.keys())}")
```

You should see `observation.state` and `action` — no image features.

---

### K-Step 5: Train ACT on Kinesthetic Data

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

Training is faster than with image data (no vision encoder). Checkpoints saved to:
```
outputs/train/act_so101_kinesthetic/checkpoints/last/
```

---

### K-Step 6: Evaluate

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

---

### Kinesthetic Workflow Summary

```
1. lerobot-find-port + lerobot-setup-motors   → configure (once)
2. lerobot-calibrate                          → calibrate arm (once)
3. python record_so101_kinesthetic.py         → physically guide arm, collect 20–50 demos
4. verify dataset (check features)
5. lerobot-train                              → train ACT (faster, no vision)
6. lerobot-record --policy.path=...          → evaluate
```

---

## References

- [LeRobot IL Robots tutorial](https://huggingface.co/docs/lerobot/il_robots)
- [SO-101 setup guide](https://huggingface.co/docs/lerobot/so101)
- [LeRobot cameras guide](https://huggingface.co/docs/lerobot/cameras)
- [DroidCam OBS plugin](https://droidcam.app/obs)
- [OBS Virtual Camera guide](https://obsproject.com/kb/virtual-camera-guide)
