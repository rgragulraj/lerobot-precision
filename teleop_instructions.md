# SO-101 Keyboard Teleoperation Instructions

This guide follows the [LeRobot IL Robots documentation](https://huggingface.co/docs/lerobot/il_robots) and [SO-101 setup guide](https://huggingface.co/docs/lerobot/so101) as closely as possible, adapted for **keyboard-only teleoperation** (no leader arm required).

---

## Prerequisites

Install LeRobot with the Feetech motor driver:

```bash
pip install -e ".[feetech]"
```

Also install `pynput` for keyboard input:

```bash
pip install pynput
```

---

## Step 1: Find Your USB Port

Connect the follower arm's controller board to your computer via USB and power. Then run:

```bash
lerobot-find-port
```

When prompted, disconnect the USB cable and press Enter. The script will identify your port.

**Linux example output:**
```
Finding all available ports for the MotorBus.
['/dev/ttyACM0', '/dev/ttyACM1']
Remove the usb cable from your MotorsBus and press Enter when done.
The port of this MotorsBus is /dev/ttyACM0
Reconnect the USB cable.
```

> **Linux only:** You may need to grant port access first:
> ```bash
> sudo chmod 666 /dev/ttyACM0
> ```

Note the port — you will use it in every subsequent command.

---

## Step 2: Set Up Motors

Each motor needs a unique ID and baudrate configured. This only needs to be done **once**.

```bash
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0
```

The script will prompt you to connect one motor at a time, starting from the gripper and working back to the shoulder. Follow each prompt:

```
Connect the controller board to the 'gripper' motor only and press enter.
'gripper' motor id set to 6
Connect the controller board to the 'wrist_roll' motor only and press enter.
...
```

Repeat until all 6 motors are configured.

---

## Step 3: Calibrate the Follower Arm

Calibration maps the motor's raw encoder values to a consistent position range. This must be done **once per arm**, and the same `id` must be used across teleoperate, record, and evaluate.

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm
```

When prompted:

1. **Move the arm to the middle of its range of motion** (roughly upright, relaxed pose) and press `Enter`.
2. **Slowly move every joint through its full range of motion** (min to max for each joint), then press `Enter` to stop recording.

The calibration file is saved automatically and reused in future sessions via `--robot.id`.

---

## Step 4: Keyboard Teleoperation

The LeRobot library includes keyboard teleop support, but for arm joint control (rather than rover or end-effector control) you need a custom key-to-joint mapping. The script below maps keys directly to individual joint increments on the SO-101.

### Key Mapping

| Key | Action |
|-----|--------|
| `q` / `e` | Shoulder pan — left / right |
| `w` / `s` | Shoulder lift — up / down |
| `a` / `d` | Elbow flex — bend / extend |
| `r` / `f` | Wrist flex — up / down |
| `t` / `g` | Wrist roll — rotate CW / CCW |
| `z` / `x` | Gripper — open / close |
| `Esc` | Stop and disconnect |

### Keyboard Teleop Script

Save the following as `teleop_so101_keyboard.py` in the root of the lerobot repo (it is already listed there if you cloned it):

```python
#!/usr/bin/env python
"""
Keyboard teleoperation for SO-101 follower arm.
Keys map to individual joint position increments.

Usage:
    python teleop_so101_keyboard.py --port /dev/ttyACM0 --id my_follower_arm
"""

import argparse
import time

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

# Degrees to move per key press (per loop tick)
STEP = 2.0

# Map pressed key chars to {motor: delta} increments
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

FPS = 30


def run(port: str, robot_id: str):
    robot_config = SO101FollowerConfig(port=port, id=robot_id)
    robot = SO101Follower(robot_config)

    teleop_config = KeyboardTeleopConfig(type="keyboard")
    teleop = KeyboardTeleop(teleop_config)

    robot.connect()
    teleop.connect()

    # Read current joint positions as starting targets
    obs = robot.get_observation()
    current_pos = {
        motor: obs[f"{motor}.pos"]
        for motor in ["shoulder_pan", "shoulder_lift", "elbow_flex",
                      "wrist_flex", "wrist_roll", "gripper"]
    }

    print("Keyboard teleoperation active. Press ESC to quit.")
    print("Controls: q/e=shoulder_pan  w/s=shoulder_lift  a/d=elbow_flex")
    print("          r/f=wrist_flex    t/g=wrist_roll     z/x=gripper")

    try:
        while teleop.is_connected:
            loop_start = time.perf_counter()

            # Get currently pressed keys
            pressed = teleop.get_action()  # e.g. {'w': None, 'a': None}

            # Apply increments for each pressed key
            for key in pressed:
                if key in KEY_TO_JOINT_DELTA:
                    for motor, delta in KEY_TO_JOINT_DELTA[key].items():
                        current_pos[motor] += delta
                        # Clamp gripper to [0, 100], others to [-100, 100]
                        if motor == "gripper":
                            current_pos[motor] = max(0.0, min(100.0, current_pos[motor]))
                        else:
                            current_pos[motor] = max(-100.0, min(100.0, current_pos[motor]))

            # Send target positions to the robot
            action = {f"{motor}.pos": val for motor, val in current_pos.items()}
            robot.send_action(action)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1.0 / FPS - dt_s, 0.0))

    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="USB port, e.g. /dev/ttyACM0")
    parser.add_argument("--id", default="my_follower_arm", help="Robot ID (must match calibration)")
    args = parser.parse_args()
    run(args.port, args.id)
```

### Run It

```bash
python teleop_so101_keyboard.py --port /dev/ttyACM0 --id my_follower_arm
```

> **Important:** The `--id` must match the one used during `lerobot-calibrate`. The calibration file is looked up by this ID.

---

## Step 5: Record a Dataset (Optional — for Imitation Learning)

Once comfortable with teleoperation, you can record demonstrations. First, log in to Hugging Face:

```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
HF_USER=$(huggingface-cli whoami | head -1)
echo $HF_USER
```

Then record episodes. Since there is no leader arm, use `lerobot-record` with `--teleop.type=keyboard`. Note: the built-in `lerobot-record` keyboard integration records key events, not joint trajectories — for full joint recording use the script above and adapt it with `lerobot.datasets.lerobot_dataset.LeRobotDataset`.

For the standard CLI record (captures whatever the keyboard teleop provides):

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm \
    --teleop.type=keyboard \
    --dataset.repo_id=${HF_USER}/so101_keyboard_test \
    --dataset.num_episodes=5 \
    --dataset.single_task="Pick up the block"
```

---

## Step 6: Train a Policy

After recording a dataset:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_keyboard_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101 \
  --job_name=act_so101 \
  --policy.device=cuda
```

Use `--policy.device=mps` on Apple Silicon, or `cpu` if no GPU is available.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied: /dev/ttyACM0` | Run `sudo chmod 666 /dev/ttyACM0` |
| Arm does not move | Check calibration `--robot.id` matches across commands |
| Keyboard not responding | Ensure a `DISPLAY` is set (pynput requires a display on Linux) |
| Motors shaking or overshooting | Reduce `STEP` value in the script (try `0.5` or `1.0`) |
| `pynput` import error | Run `pip install pynput` |

---

## References

- [LeRobot IL Robots tutorial](https://huggingface.co/docs/lerobot/il_robots)
- [SO-101 setup guide](https://huggingface.co/docs/lerobot/so101)
- [LeRobot GitHub](https://github.com/huggingface/lerobot)
