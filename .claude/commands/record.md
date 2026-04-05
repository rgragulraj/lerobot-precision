# lerobot-record command builder

Interview the user to build a `lerobot-record` CLI command. Ask questions one at a time and build toward a final command.

## Interview Flow

Ask the following questions **one at a time**, in order. Wait for each answer before asking the next. Keep questions short and conversational.

---

### Question 1 — Hugging Face repo ID

"What's your Hugging Face repo ID for this dataset? (format: `username/dataset-name`, e.g. `rgragulraj/pick-cube-so101`)"

### Question 2 — Task description

"Describe the task in one sentence for the `single_task` label. This is what gets stored as ground truth. (e.g. `Pick up the red cube and place it in the box`)"

### Question 3 — Robot type

"Which robot are you using? Options:

- `so101_follower` (SO-101, your primary)
- `so100_follower`
- `bi_so100_follower` (bimanual SO-100)
- `koch_follower`
- Other (tell me)"

### Question 4 — Robot port

"What USB port is the robot follower arm on? (e.g. `/dev/ttyUSB0`, `/dev/ttyACM0`) — check with `ls /dev/tty*` if unsure."

### Question 5 — Teleoperator type

"How will you control the robot during data collection?

- `so101_leader` (SO-101 leader arm — your primary)
- `so100_leader`
- `keyboard`
- Policy only (no teleop, specify a policy path)
- Other (tell me)"

### Question 6 — Teleop port (skip if keyboard or policy-only)

"What port is the leader arm on? (e.g. `/dev/ttyUSB1`)"

### Question 7 — Cameras

"How many cameras are you using, and what type?

- `opencv` (USB webcam — most common for SO-101 setup)
- `realsense` (Intel RealSense)
- None

For each camera, I'll need: a name (e.g. `top`, `gripper`), index or path (e.g. `0`, `1`, or `/dev/video2`), and optionally resolution/fps (default: 640x480 @ 30fps)."

Collect camera info for each camera. For example: "Camera 1: name=`top`, index=`0`". Ask follow-ups if needed.

### Question 8 — Number of episodes

"How many episodes do you want to record? (default: 50)"

### Question 9 — Episode and reset times

"How long should each episode be in seconds? (default: 60s)
And how long for the reset period between episodes? (default: 60s)"

### Question 10 — Push to Hub?

"Push to Hugging Face Hub after recording? (yes/no, default: yes)"

### Question 11 — Resume?

"Are you resuming a previously interrupted recording? (yes/no, default: no)"

### Question 12 — Any tags for the Hub dataset? (optional)

"Any tags to add to the dataset on the Hub? (e.g. `so101`, `precision`, `insertion`) — press Enter to skip."

---

## After collecting all answers

Build the full `lerobot-record` command using this structure:

```
lerobot-record \
  --robot.type=<robot_type> \
  --robot.port=<robot_port> \
  --robot.cameras='{<name>: {"type": "<cam_type>", "index_or_path": <idx>, "width": <w>, "height": <h>, "fps": <fps>}, ...}' \
  --teleop.type=<teleop_type> \
  --teleop.port=<teleop_port> \
  --dataset.repo_id=<repo_id> \
  --dataset.single_task="<task_description>" \
  --dataset.num_episodes=<n> \
  --dataset.episode_time_s=<t> \
  --dataset.reset_time_s=<r> \
  --dataset.push_to_hub=<true|false> \
  [--dataset.tags='["tag1","tag2"]'] \
  [--resume=true]
```

Omit any optional flags that weren't set or that match their defaults (to keep the command clean).

**Display the final command in a code block**, then briefly explain any non-obvious flags.

Also mention: "Run `lerobot-calibrate --robot.type=<type> --robot.port=<port>` first if this is your first time with this robot."
