#!/bin/bash

lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --dataset.repo_id=rgragulraj/policy1_diverse_session_a \
  --dataset.root=/home/asurite.ad.asu.edu/rrangasa/.cache/huggingface/lerobot/rgragulraj/policy1_diverse_s \
  --dataset.image_transforms.enable=true \
  --policy.push_to_hub=false \
  --steps=80000 \
  --output_dir=outputs/policy1_session_a
