# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

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

## Code Style

- Line length: 110 characters (ruff)
- Target: Python 3.10+
- Docstrings: Google style convention
- Quotes: Black-compatible (double quotes)
