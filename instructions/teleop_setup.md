# Leader-Follower Teleoperation Setup

Setup guide for the SO-101 follower arm (`vellai_kunjan`) and a new SO-101 leader arm.

---

## 1. Install the package

```bash
cd ~/lerobot-precision
pip install -e ".[feetech]"
```

---

## 2. Find USB ports

Plug in one arm at a time and run:

```bash
ls /dev/ttyACM*
```

Note which port belongs to the follower and which to the leader. For example:

- Follower → `/dev/ttyACM0`
- Leader → `/dev/ttyACM1`

---

## 3. Calibrate the follower arm

Skip this step if `vellai_kunjan` is already calibrated.

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan
```

Follow the on-screen prompts to move the arm through its range of motion.

---

## 4. Calibrate the leader arm

```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm
```

Follow the on-screen prompts to move the leader arm through its range of motion.

---

## 5. Run teleoperation

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm
```

Move the leader arm — the follower arm mirrors it in real time.
Press `Ctrl+C` to stop and disconnect both arms.

---

## Notes

- Replace `/dev/ttyACM0` and `/dev/ttyACM1` with the actual ports from Step 2.
- Replace `my_leader_arm` with whatever ID you chose during leader calibration.
- Calibration files are saved to `~/.cache/huggingface/lerobot/calibration/`.
- If an arm moves in the wrong direction after calibration, re-run the calibration for that arm.
