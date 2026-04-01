# Gripper Camera Check

Guide for verifying the gripper camera works before recording datasets.

---

## 1. Find the camera device index

List all video devices:

```bash
ls /dev/video*
```

Plug in the gripper camera and run it again — the new entry is your camera. Typically `/dev/video0` or `/dev/video2`, etc.

To get more details on each device:

```bash
v4l2-ctl --list-devices
```

> **Note:** On modern Ubuntu, apps like Cheese access cameras via **PipeWire** rather than V4L2 directly.
> If your gripper camera shows up in Cheese but not under `/dev/video*`, see [Section 1b](#1b-camera-visible-in-cheese-but-not-in-devvideo) below.

---

## 1b. Camera visible in Cheese but not in /dev/video*

This happens when the camera is accessed through PipeWire instead of V4L2 directly. Skip `ls /dev/video*` and find the index by scanning with OpenCV:

```python
import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"index {i}: {'OK - ' + str(frame.shape) if ret else 'opened but no frame'}")
        cap.release()
    else:
        print(f"index {i}: not available")
```

Save this as `/tmp/find_camera.py` and run:

```bash
python /tmp/find_camera.py
```

The index that returns `OK` and a valid frame shape is your gripper camera. Use that index in all subsequent steps.

To also check via PipeWire directly:

```bash
wpctl status | grep -i camera
```

---

## 2. Test the camera with a quick Python script

Replace `0` with your actual device index:

```python
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("gripper_test.jpg", frame)
        print(f"Frame captured: {frame.shape}")
    else:
        print("ERROR: Cannot read frame")
    cap.release()
```

Run it:

```bash
python /tmp/camera_test.py
```

If `gripper_test.jpg` is saved and shows a valid image, the camera works.

---

## 3. Test live feed (optional)

```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Gripper Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

Press `q` to quit.

---

## 4. Test with LeRobot's teleoperate (with camera)

Once you know the device index, pass it to `lerobot-teleoperate` using the `--robot.cameras` argument:

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=vellai_kunjan \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm \
    --robot.cameras='{"gripper": {"type": "opencv", "index_or_path": 7, "fps": 30, "width": 640, "height": 480}}' \
    --display_data=true
```

Set `--display_data=true` to see the live camera feed during teleoperation.

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open camera` | Try a different index (0, 2, 4...) |
| Black frame / no image | Camera may need a second to warm up; add `time.sleep(1)` before `cap.read()` |
| Low FPS | Use MJPG format: pass `"fourcc": "MJPG"` in the camera config |
| Image rotated | Add `"rotation": 90` (or 180, 270) to the camera config |
| Camera not listed in `/dev/video*` | Camera may be PipeWire-only — scan with OpenCV indices (see Section 1b) |

---

## Notes

- LeRobot uses OpenCV (`opencv` camera type) for USB cameras like the gripper cam.
- Calibration files for the arms are in `~/.cache/huggingface/lerobot/calibration/`.
- Camera config is passed inline as JSON or defined in a robot config file.
