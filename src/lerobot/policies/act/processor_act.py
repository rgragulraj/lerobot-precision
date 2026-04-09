#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor as tv_to_tensor

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


class SpatialConditioningProcessorStep:
    """Runs an OpenCV block/slot detector on a camera frame and injects the keypoint
    vector as observation.environment_state for online inference.

    Applies exponential moving average (EMA) smoothing over frames to reduce detection
    noise. Falls back to the last known good estimate (or zero on first frame) if
    detection fails.

    Use this in the pre-processing pipeline when deploying a policy trained with
    use_spatial_conditioning=True. For offline dataset processing (training data),
    use scripts/add_spatial_features.py instead.

    Args:
        camera_key: Observation key for the camera frame to run detection on.
            Policy 1 (top-down, HSV): "observation.images.top"
            Policy 2 (wrist, shape-matching): "observation.images.wrist"
        detector_type: Which detector to use.
            "shape_top" — CLAHE + Canny + shape template matching, top-down camera (Policy 1).
            "shape_wrist" — CLAHE + Canny + shape template matching, wrist camera (Policy 2).
            "hsv_top" — HSV colour masking, top-down camera (simple fallback, single-slot only).
        calibration_path: Path to the calibration JSON.
            Policy 1: scripts/top_calibration.json (produced by detect_block_slot_hsv.py)
            Policy 2: scripts/wrist_calibration.json (produced by detect_block_slot.py)
        include_angle: If True, output a 10-float token with rotation angles.
            Must match the spatial_conditioning_dim used at training time.
        ema_alpha: EMA smoothing coefficient. Higher = more weight on the current frame
            (more responsive but noisier). Lower = smoother but laggier. 0.5 is a good default.
        shape: Shape profile name for Policy 2 multi-shape (Phase 3+) datasets.
            Leave None for Policy 1 or Policy 2 single-shape.
    """

    def __init__(
        self,
        camera_key: str,
        detector_type: str = "hsv_top",
        calibration_path: str | Path | None = None,
        include_angle: bool = False,
        ema_alpha: float = 0.5,
        shape: str | None = None,
    ):
        self.camera_key = camera_key
        self.ema_alpha = ema_alpha
        self._ema: np.ndarray | None = None

        # Lazy import — scripts/ is not on the package path; only needed at inference time.
        import sys

        sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))

        cal_path = Path(calibration_path) if calibration_path else None

        if detector_type == "shape_top":
            from detect_block_slot import TopCameraShapeDetector  # noqa: PLC0415

            kwargs = {}
            if cal_path is not None:
                kwargs["calibration_path"] = cal_path
            if shape is not None:
                kwargs["shape"] = shape
            self._detector = TopCameraShapeDetector(**kwargs)

        elif detector_type == "shape_wrist":
            from detect_block_slot import WristCameraDetector  # noqa: PLC0415

            kwargs = {}
            if cal_path is not None:
                kwargs["calibration_path"] = cal_path
            if shape is not None:
                kwargs["shape"] = shape
            self._detector = WristCameraDetector(**kwargs)

        elif detector_type == "hsv_top":
            from detect_block_slot_hsv import TopCameraDetector  # noqa: PLC0415

            kwargs = {"include_angle": include_angle}
            if cal_path is not None:
                kwargs["calibration_path"] = cal_path
            self._detector = TopCameraDetector(**kwargs)

        else:
            raise ValueError(
                f"Unknown detector_type: '{detector_type}'. "
                "Must be 'shape_top' (Policy 1), 'shape_wrist' (Policy 2), or 'hsv_top' (fallback)."
            )

    def update_shape(self, shape: str) -> None:
        """Update the shape filter on the underlying detector at runtime.

        Called by LanguageConditioningProcessorStep when the language command is parsed to
        a specific shape name (e.g. "square", "round"), so the spatial detector filters to
        the correct slot geometry without needing a new processor instance.

        Args:
            shape: Shape profile name recognised by the detector (e.g. "square", "round").
        """
        if hasattr(self._detector, "shape"):
            self._detector.shape = shape
        else:
            raise AttributeError(
                f"Detector {type(self._detector).__name__} does not have a 'shape' attribute. "
                "Only TopCameraShapeDetector / WristCameraDetector support runtime shape updates."
            )

    def __call__(self, batch: dict) -> dict:
        frame = batch[self.camera_key]

        # Accept torch tensors or numpy arrays.
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()

        # Strip batch dimension if present: (B, C, H, W) → (C, H, W)
        if frame.ndim == 4:
            frame = frame[0]

        # CHW → HWC
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            frame = frame.transpose(1, 2, 0)

        # Normalise to uint8 if the frame is in [0, 1] float range.
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)

        # RGB → BGR for OpenCV.
        frame_bgr = frame[..., ::-1].copy()

        token = self._detector.detect(frame_bgr)

        # EMA smoothing.
        if self._ema is None:
            self._ema = token
        else:
            self._ema = self.ema_alpha * token + (1 - self.ema_alpha) * self._ema

        batch["observation.environment_state"] = self._ema.copy()
        return batch


class GoalImageProcessorStep:
    """Loads a goal image from disk and injects it as observation.images.goal for online inference.

    The goal image is loaded once at construction, preprocessed to match the training format
    (uint8 HWC → float32 CHW in [0,1]), and injected into every inference batch. It is not
    re-loaded per step — the same goal image is reused for the entire episode.

    Use this in the pre-processing pipeline when deploying a policy trained with
    use_goal_image=True. Set a new instance at the start of each episode by calling
    reset_goal(path) with the path to the new goal image.

    Args:
        goal_image_path: Path to the goal image PNG captured at the canonical hover pose.
        target_size: (H, W) to resize the goal image to. Must match the resolution used
            during training (i.e., the same as the observation camera images). If None,
            uses the image as-is.
        goal_image_key: Key under which the goal image is injected into the batch.
            Must match ACTConfig.goal_image_feature_key (default: "observation.images.goal").
    """

    def __init__(
        self,
        goal_image_path: str | Path,
        target_size: tuple[int, int] | None = None,
        goal_image_key: str = "observation.images.goal",
    ):
        self.goal_image_key = goal_image_key
        self.target_size = target_size
        self._goal_tensor: torch.Tensor | None = None
        self.reset_goal(goal_image_path)

    def reset_goal(self, goal_image_path: str | Path) -> None:
        """Load a new goal image. Call this at the start of each episode.

        Args:
            goal_image_path: Path to the new goal image PNG.
        """
        path = Path(goal_image_path)
        img = Image.open(path).convert("RGB")
        if self.target_size is not None:
            h, w = self.target_size
            img = img.resize((w, h), Image.BILINEAR)
        # HWC uint8 → CHW float32 in [0, 1]
        self._goal_tensor = tv_to_tensor(img)  # (3, H, W), float32

    def __call__(self, batch: dict) -> dict:
        if self._goal_tensor is None:
            raise RuntimeError("GoalImageProcessorStep: no goal image loaded. Call reset_goal() first.")

        # The pre-processor runs before AddBatchDimensionProcessorStep, so the goal tensor
        # needs to match the batch structure at that point.  Inject as (C, H, W) float32 —
        # AddBatchDimensionProcessorStep will add the batch dimension together with all other keys.
        batch[self.goal_image_key] = self._goal_tensor
        return batch


class LanguageConditioningProcessorStep:
    """Computes a CLIP text embedding from a natural-language command and injects it as
    observation.language for online inference.

    CLIP is loaded lazily on the first call to set_command() and is not stored in the model
    checkpoint — only the linear projection layer (encoder_language_input_proj) lives in ACT.
    This keeps the model checkpoint small (~150 MB smaller) and allows the CLIP encoder to be
    shared across policies.

    When the command mentions a known shape word ("square", "round", "hex", "triangle", "circle"),
    the parsed shape is optionally forwarded to a SpatialConditioningProcessorStep via
    update_shape() so the top-camera detector filters to the correct slot geometry.

    Use this in the pre-processing pipeline when deploying a policy trained with
    use_language_conditioning=True. Call set_command() once per episode (or when the task
    changes). The same embedding is reused for every step until the command changes.

    Args:
        model_name: HuggingFace model ID for the CLIP model.
            Default matches ACTConfig.language_model_name: "openai/clip-vit-base-patch32".
        language_key: Key under which the embedding is injected into the batch.
            Must match the feature key used at training time: "observation.language".
        spatial_step: Optional SpatialConditioningProcessorStep to forward the parsed shape
            to at runtime. When provided, calling set_command("... square slot ...") will
            also call spatial_step.update_shape("square").
        device: Device for the CLIP model ("cpu" or "cuda"). Embeddings are stored on CPU
            and moved to the model device by DeviceProcessorStep later in the pipeline.
    """

    # Canonical shape words to match in language commands.
    _SHAPE_VOCAB: dict[str, str] = {
        "square": "square",
        "round": "round",
        "circular": "round",
        "circle": "round",
        "hex": "hex",
        "hexagon": "hex",
        "hexagonal": "hex",
        "triangle": "triangle",
        "triangular": "triangle",
    }

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        language_key: str = "observation.language",
        spatial_step: "SpatialConditioningProcessorStep | None" = None,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.language_key = language_key
        self.spatial_step = spatial_step
        self.device = device

        # Set at first set_command() call.
        self._embedding: torch.Tensor | None = None
        self.parsed_shape: str | None = None

        # CLIP model and tokenizer — loaded lazily.
        self._clip_model = None
        self._clip_processor = None

    def _load_clip(self) -> None:
        """Load CLIP model and processor from HuggingFace. Called once on first use."""
        from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

        self._clip_processor = CLIPProcessor.from_pretrained(self.model_name)
        self._clip_model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._clip_model.eval()

    def set_command(self, command: str) -> None:
        """Tokenize *command*, compute its CLIP embedding, and cache it for inference.

        Also parses a shape word from the command (if present) and forwards it to the
        linked SpatialConditioningProcessorStep so the spatial detector filters to the
        correct slot geometry.

        Args:
            command: Natural-language task description, e.g.
                "hover above the square slot" or "pick up the block and insert into round slot".
        """
        if self._clip_model is None:
            self._load_clip()

        inputs = self._clip_processor(text=[command], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self._clip_model.get_text_features(**inputs)  # (1, 512)
            # L2-normalise to match CLIP's standard embedding space.
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Store on CPU — DeviceProcessorStep will move the whole batch to the model device.
        self._embedding = text_features[0].cpu()

        # Parse shape word and update spatial detector if linked.
        command_lower = command.lower()
        self.parsed_shape = None
        for word, canonical in self._SHAPE_VOCAB.items():
            if word in command_lower:
                self.parsed_shape = canonical
                break

        if self.parsed_shape is not None and self.spatial_step is not None:
            self.spatial_step.update_shape(self.parsed_shape)

    def __call__(self, batch: dict) -> dict:
        if self._embedding is None:
            raise RuntimeError("LanguageConditioningProcessorStep: no command set. Call set_command() first.")
        batch[self.language_key] = self._embedding
        return batch


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], PolicyProcessorPipeline[PolicyAction, PolicyAction]]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
