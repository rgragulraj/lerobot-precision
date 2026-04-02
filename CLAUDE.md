# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**lerobot-precision** is a research project by LENS Lab built on top of LeRobot (v0.4.3). The goal is to develop a high-precision, high-orientation-aware manipulation system on low-cost hardware (SO-101), targeting tasks like insertion, screwing, plug/unplug, and connector alignment — with the broader ambition of a generalised robot that can handle objects and perspectives it has never seen before.

Key research directions:

- Modifying the ACT policy source code to improve precision and orientation awareness
- Multi-policy routing: automatic task detection + language command input to select the appropriate policy at runtime
- Language-conditioned control (text input to guide policy selection — planned, not yet implemented)
- Generalisation to novel objects and viewpoints
- Benchmarking success rate on precision manipulation tasks (benchmarking platform TBD)

This project is in its **initial/exploration phase**. Data collection with the SO-101 is the primary path — Hub datasets are used minimally.

LeRobot (v0.4.3) is a Hugging Face library for real-world robotics in PyTorch. It provides imitation learning and reinforcement learning policies, pretrained models/datasets on Hugging Face Hub, simulation environments (ALOHA, PushT, Libero, MetaWorld), and real hardware robot support (SO-100, SO-101, Koch, Reachy2, Unitree G1, etc.).

## Installation

```bash
pip install -e ".[dev]"
# Or with uv (preferred):
uv pip install -e ".[dev]"
```

## Commands

### Linting & Formatting

```bash
pre-commit run --all-files        # Run all pre-commit hooks (ruff, typos, mypy, etc.)
ruff check src/                   # Lint only
ruff format src/                  # Format only
```

### Tests

```bash
pytest tests/                     # Run all unit tests
pytest tests/test_datasets.py     # Run a single test file
pytest tests/ -k "test_name"      # Run a specific test by name
```

### End-to-End Tests (via Make)

```bash
make test-end-to-end              # Run full E2E pipeline (all policies)
make test-act-ete-train           # ACT policy train E2E
make test-diffusion-ete-train     # Diffusion policy train E2E
make DEVICE=cuda test-end-to-end  # Run on GPU
```

### Training & Evaluation

```bash
lerobot-train --policy.type=act --env.type=aloha --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
lerobot-train --config_path=<checkpoint>/train_config.json --resume=true
lerobot-eval --policy.path=<checkpoint>/pretrained_model --env.type=aloha
```

### Data Collection

```bash
lerobot-record --robot.type=so101 --repo_id=<hf_user>/<dataset_name>
lerobot-teleoperate --robot.type=so101
lerobot-calibrate --robot.type=so101
```

## Code Architecture

The main package lives in `src/lerobot/`. All CLI entry points are in `src/lerobot/scripts/` and registered in `pyproject.toml` under `[project.scripts]`.

### Key Design Patterns

**Config-driven via draccus**: All major components use typed Python dataclasses as configs, which draccus exposes as CLI arguments. Configs live in `src/lerobot/configs/`. The pattern is `--component.type=<name>` to select an implementation, then `--component.param=<value>` to override defaults.

**Factory functions**: `make_policy()`, `make_dataset()`, `make_env()`, `make_robot_from_config()` instantiate components from their configs. When adding new implementations, register them in the corresponding factory.

**PreTrainedPolicy base class** (`src/lerobot/policies/pretrained.py`): All policies inherit from this. Key methods are `select_action(batch)` for inference and `forward(batch)` for training. Policies are saved/loaded with Hugging Face Hub conventions.

### Layer Structure

- **`policies/`** — ACT, Diffusion, TDMPC, VQBeT, SmolVLA, PI0, GROOT, X-VLA, RTC, SAC. Each policy has a `<name>.py` + `configuration_<name>.py` pair.
- **`datasets/`** — `LeRobotDataset` (parquet + video files), `StreamingLeRobotDataset`, `OnlineBuffer` (for live collection). Dataset features are normalized via the `Processor` pipeline.
- **`robots/`** — Abstract `Robot` base class; concrete implementations per hardware platform. Calibration stored as JSON files.
- **`motors/`** — Low-level Dynamixel and Feetech servo control.
- **`cameras/`** — OpenCV, RealSense, Reachy2 camera interfaces.
- **`envs/`** — Gymnasium-compatible simulation wrappers; `GymManipulator` bridges real robots to the gym interface.
- **`teleoperators/`** — Interfaces for human-in-the-loop data collection (gamepad, keyboard, leader arm).
- **`configs/`** — `TrainPipelineConfig`, `EvalPipelineConfig`, policy/env/dataset config dataclasses.
- **`processor/`** — Observation/action normalization and feature extraction pipelines.
- **`rl/`** — RL-specific components: `Actor`, `Learner`, `ReplayBuffer`, `GymManipulator`.
- **`async_inference/`** + **`transport/`** — Asynchronous policy inference and inter-process communication layer.

### Dataset Format

Datasets store observations as parquet files (non-image features) and mp4 video files (image features). The `LeRobotDataset` class handles both local and Hub-hosted datasets transparently. Episodes are indexed by `episode_index`; frames by `frame_index`.

## Hardware Setup

- **Robot**: SO-101 (leader + follower arm pair)
- **Teleop**: SO-101 leader arm (primary), keyboard (secondary)
- **Cameras**: Top-down webcam (fixed mount) + gripper camera
- **Policy focus**: ACT (primary), custom policies (future)

## Research Context

- **Lab**: LENS Lab
- **Repo**: lerobot-precision
- **Audience**: Solo developer, but code must be structured clearly enough for other lab members to understand
- **Phase**: Initial — exploring architecture changes to ACT, no custom modifications yet
- **Benchmarking**: Success rate on precision manipulation tasks; specific benchmark platform TBD

## Code Style

- Line length: 110 characters (ruff)
- Target: Python 3.10+
- Docstrings: Google style convention
- Quotes: Black-compatible (double quotes)
- Keep changes modular and well-commented — other lab members need to follow the work
- Prefer extending existing LeRobot abstractions (factory functions, config dataclasses) over bypassing them
