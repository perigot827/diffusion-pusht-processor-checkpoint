# Diffusion PushT checkpoint with saved processors

Download a community-converted copy of the official trained `lerobot/diffusion_pusht` checkpoint, ready for LeRobot's external pre/post processors. **No migration command or migration-PR checkout is needed to load the package.**

The learned model tensors are unchanged. This repository provides a versioned checkpoint archive, source attribution, integrity hashes and a small loading example. It is not an official Hugging Face release or a newly trained policy.

## Download and use

Download `diffusion-pusht-processors-20260915.tar` and `SHA256SUMS` from release `v2026.09.15`. The model archive is approximately 1.05 GB.

```sh
curl --fail -L -O https://github.com/perigot827/diffusion-pusht-processor-checkpoint/releases/download/v2026.09.15/diffusion-pusht-processors-20260915.tar
curl --fail -L -O https://github.com/perigot827/diffusion-pusht-processor-checkpoint/releases/download/v2026.09.15/SHA256SUMS
shasum -a 256 -c SHA256SUMS
tar -xf diffusion-pusht-processors-20260915.tar
(cd diffusion-pusht && shasum -a 256 -c SHA256SUMS)
```

Use `smoke.py` from this repository in a compatible LeRobot environment:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python smoke.py --checkpoint ./diffusion-pusht --device cpu
```

It strictly loads the model and saved processors and produces one action from synthetic observations. Pass `--device mps` or `--device cuda` only on a machine supporting that device. The package defaults to CPU. The smoke result is not a task-success evaluation.

The tested LeRobot revision is `89236ea0f4f81a81ca566081e20dd1ff5f823cbe`, without either migration patch. For a source checkout, LeRobot's `uv sync --locked --extra diffusion --extra pusht` installs the relevant extras. Python 3.12, torch 2.11.0 and diffusers 0.39.0 were used for the consumer check. The simulator check additionally used gym-pusht 0.1.6 and pymunk 6.11.1.

## Run one simulation episode

With LeRobot's `pusht` extra installed, the included evaluator accepts the same checkpoint directory:

```sh
SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python evaluate_pusht.py --checkpoint ./diffusion-pusht --device cpu --report pusht-result.json
```

Use `--device mps` on a supported Apple machine to reproduce the packaged-checkpoint smoke configuration. The evaluator runs one seed (100000 by default), at most 300 environment steps, and records the observed outcome. It does not retrain the model.

## What changed and what was tested

- Original: [lerobot/diffusion_pusht at 84a7c231](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653). Original model SHA-256: `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`.
- All **213 core tensors** retained exactly; **8 legacy statistic tensors** extracted into processors. Converted model SHA-256: `42bddc7d0fe1ef928d708c207a2c9bb8ab7166c2d2b28d3c6abb955e18fb7594`.
- Saved processor normalization checked numerically at four points each for state, image and action.
- Migration used the tuple helper from [PR4457](https://github.com/huggingface/lerobot/pull/4457) and offline card fix from [PR4635](https://github.com/huggingface/lerobot/pull/4635); combined regression tests: 11 passed.
- The converted weights/processors succeeded in **one** MPS PushT episode: seed 100000, 251 environment steps, maximum reward 1.0. That check preceded the CPU-default packaging changes. See `migration-smoke-result.json`.
- The packaged checkpoint loaded and inferred on **CPU using unmodified LeRobot**, without fetching anything from the Hub. See `consumer-verification.json`.
- The included evaluator also completed one MPS episode using the final packaged checkpoint and the same unmodified LeRobot revision: 251 steps, maximum reward 1.0, success. See `packaged-simulation-result.json`. This is another smoke run of the same seed, not an independent performance estimate.

This does not reproduce the model card's 500-episode benchmark. It establishes neither real-robot suitability nor adoption or labor savings. The source repository has the original training and benchmark context. This repository's conversion work and tests were performed by AI-D using Codex; no human review or upstream endorsement is claimed.

## Provenance and license

The original model declares Apache-2.0. `LICENSE` preserves the upstream LeRobot license text; `UPSTREAM_MODEL_CARD.md` preserves the original model card. The packaged model card explains the conversion, correct dataset attribution and CPU-default changes. Model: The Hugging Face / LeRobot authors and contributors; Diffusion Policy method: Chi et al. Packaging and smoke example: AI-D contributors, Apache-2.0.

retrieval_status: no_reliable_hit.
