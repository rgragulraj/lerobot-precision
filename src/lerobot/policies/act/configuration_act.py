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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and 'output_shapes`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Policy 2 (precision insertion) overrides — pass these at training time via CLI:
        --policy.chunk_size=20
        --policy.n_action_steps=1
        --policy.temporal_ensemble_coeff=0.1
        --policy.kl_weight=20.0
        --policy.optimizer_lr=5e-5
        --policy.optimizer_lr_backbone=5e-6

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
            Default (100) suits coarse pick-and-place (Policy 1). Policy 2 uses 20 — predicting fewer steps
            at a time lets the policy replan more frequently and correct millimetre-level misalignment on
            the fly.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
            Policy 2 uses 1 — the policy re-evaluates the wrist camera every single frame (at 30fps), giving
            maximum reactivity for fine alignment. Requires temporal_ensemble_coeff to be set.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        use_spatial_conditioning: Whether to condition on a 10-float spatial keypoint vector
            (observation.environment_state). Flows through the existing env_state_feature path —
            no architecture changes. Features extracted offline via scripts/detect_block_slot.py.
        spatial_conditioning_dim: Expected dimension of the spatial token. Used for validation only.
        use_goal_image: Whether to condition the policy on a goal image (observation.images.goal). Required
            for Policy 2 — enables generalisation to novel shapes by giving the policy a visual target state.
        goal_image_feature_key: Dataset key for the per-episode goal image. Default matches the recording
            convention used during Policy 2 data collection.
        use_shared_goal_backbone: Whether the goal image shares the ResNet backbone with the current
            observation images. Recommended until >300 episodes — keeps both frames in the same embedding
            space so the policy can compute the visual delta (current → goal) directly.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
            Policy 2 uses 0.1 — the most recent prediction gets ~10x weight over the oldest, smoothing
            out jitter from noisy wrist-camera observations without introducing lag.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
            Policy 2 uses 20.0 (vs default 10.0) — higher KL weight forces the latent variable to encode
            more information about the current alignment state, producing more consistent fine trajectories.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100  # Policy 2: 20
    n_action_steps: int = 100  # Policy 2: 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Selective backbone unfreeze (Policy 1 Phase 1c).
    # Only use this after >500 Policy 1 episodes. With fewer episodes the unfrozen BN stats tend to
    # overfit to specific objects and lighting conditions.
    # Set to ["layer4"] to unfreeze the final ResNet block only.
    # IMPORTANT: FrozenBatchNorm2d is kept — do not switch to regular BatchNorm when unfreezing.
    unfreeze_backbone_layers: list[str] = field(default_factory=list)

    # Spatial conditioning (Policy 2 Phase 2).
    # When True, the policy expects observation.environment_state to be a 10-float keypoint vector:
    #   [cx_block, cy_block, w_block, h_block, angle_block,
    #    cx_slot,  cy_slot,  w_slot,  h_slot,  angle_slot]
    # All values in [0,1]; angles in [-0.5, 0.5] (representing [-90°, 90°]).
    # This is fed through the existing env_state_feature path in ACT — no architecture change needed.
    # The spatial features are extracted offline from recorded wrist-camera videos using
    # scripts/detect_block_slot.py + scripts/add_spatial_features.py.
    use_spatial_conditioning: bool = False
    spatial_conditioning_dim: int = 10  # must match the detector output dimension

    # Goal image conditioning (Policy 2 Phase 1).
    # When enabled, a single goal image (what "success" looks like from the wrist camera) is fed as an
    # additional encoder token alongside the current observation. The policy learns to close the gap between
    # the current wrist view and the goal view, which is the key mechanism for generalising to novel shapes.
    # Set use_goal_image=True at training time when the dataset contains observation.images.goal.
    # Use shared backbone (default) until you have >300 episodes — it keeps the current and goal frames in
    # the same embedding space so the policy can directly compute the visual delta (current → goal).
    use_goal_image: bool = False
    goal_image_feature_key: str = "observation.images.goal"
    use_shared_goal_backbone: bool = True

    # Language conditioning (Policy 1 Phase 4).
    # When enabled, a CLIP ViT-B/32 text embedding (512-float) is projected into the encoder token sequence
    # as an additional 1D token. This lets a single policy handle multiple task variants selected by natural
    # language ("hover above the square slot" vs "hover above the round slot").
    #
    # The model itself does NOT contain the CLIP encoder — embeddings are pre-cached offline via
    # scripts/add_language_features.py and stored as observation.language (shape [language_dim]) in the
    # dataset. At inference, LanguageConditioningProcessorStep computes the embedding from a command string.
    #
    # Bridge to spatial conditioning: the parsed shape name (e.g. "square") from the language command is
    # passed to SpatialConditioningProcessorStep.update_shape() so the spatial detector filters to the
    # correct slot geometry — no architecture changes needed for the spatial path.
    use_language_conditioning: bool = False
    language_dim: int = 512  # CLIP ViT-B/32 pooled text embedding dimension
    language_model_name: str = "openai/clip-vit-base-patch32"

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None  # Policy 2: 0.1

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0  # Policy 2: 20.0

    # Training preset
    optimizer_lr: float = 1e-5  # Policy 2: 5e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5  # Policy 2: 5e-6

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature and not self.robot_state_feature:
            raise ValueError(
                "You must provide at least one image, environment state, or robot state among the inputs."
            )
        if self.use_spatial_conditioning and not self.env_state_feature:
            raise ValueError(
                "use_spatial_conditioning=True requires 'observation.environment_state' "
                f"with shape ({self.spatial_conditioning_dim},) in input_features. "
                "Run add_spatial_features.py on the dataset before training."
            )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
