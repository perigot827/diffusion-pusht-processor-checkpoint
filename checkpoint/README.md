---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
base_model: lerobot/diffusion_pusht
datasets:
- lerobot/pusht
tags:
- robotics
- diffusion
- lerobot
- community-conversion
---
# Diffusion PushT checkpoint with saved processors

Community format conversion by AI-D. This is not a newly trained model or an official Hugging Face release.

Original trained weights: [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653), revision `84a7c23178445c6bbf7e1a884ff497017910f653`, Apache-2.0. The upstream model card is preserved in `UPSTREAM_MODEL_CARD.md`, with its original authorship and evaluation context. Dataset: `lerobot/pusht`. Method: Diffusion Policy (Chi et al.); implementation: LeRobot, The Hugging Face team and contributors.

## Conversion

All 213 core model tensors are unchanged. Eight legacy normalization buffers were removed from the model and their statistics saved in the pre/post processors. The converter combined LeRobot PR4457 tuple coercion and PR4635 offline card rendering on base `89236ea0f4f81a81ca566081e20dd1ff5f823cbe`. No retraining or quantization was performed.

This package sets the model and input processor's default device to CPU for portability. The dataset attribution above corrects the converter's `unknown` card field. These are packaging metadata/device changes; learned weights and normalization statistics remain unchanged. `SHA256SUMS` covers every package file except itself.

## Verification

The converted weights and saved normalization processors passed tensor equality and numerical normalization checks. Before CPU-default packaging, one fixed-seed MPS PushT smoke episode succeeded (seed 100000, 251 steps, reward 1.0, 100 diffusion steps). This is not a reproduction of the upstream 500-episode benchmark. The packaged checkpoint also strictly loaded and produced one finite CPU action with unmodified LeRobot at the base revision.

This checkpoint is for the simulated 2D PushT environment. Real-robot compatibility, third-party adoption and human labor savings have not been demonstrated. AI-D prepared and tested the conversion using Codex; no human review or upstream endorsement is claimed.
