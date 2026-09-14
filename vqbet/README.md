# VQ-BeT PushT processor kit

Reconstruct an already-verified VQ-BeT checkpoint with saved processors for unmodified LeRobot. This kit contains no learned model weights. It retrieves the official158MB model file at a fixed revision, verifies its SHA256, removes only six legacy normalization tensors, and checks the reconstructed model SHA256. Bundled config/statistic files are also checked. No training, Hub upload or migration patches are needed to reconstruct it. Existing output directories are rejected.

Source: https://huggingface.co/lerobot/vqbet_pusht/tree/15c2d0af889c401c7e5db7a07499d3b498afc276 . Original copyright/attribution is retained in UPSTREAM_MODEL_CARD.md and LICENSE. This is an AI-D community conversion, not an upstream release or endorsement.

## Use

Use a Python3.12 environment with LeRobot at commit89236ea0f4f81a81ca566081e20dd1ff5f823cbe and its `pusht` extra. Existing validation used torch2.11.0, gym-pusht0.1.6, gymnasium1.3.0 and pymunk6.11.1. A fresh environment installation was not repeated while packaging. Runtime dependencies take additional space beyond the158MB source model.

```sh
python prepare.py
# Or reuse your original downloaded weights without a network request:
python prepare.py --source /path/to/original/model.safetensors --output my-vqbet
```

For one local simulated episode in that environment:

```sh
SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 python evaluate_vqbet.py --checkpoint vqbet-pusht --device cpu --seed 200011 --report episode.json
```

On a compatible Apple Silicon environment, `--device mps` selects MPS. The saved default is CPU. Model loading uses strict state-dictionary matching. The evaluator does not include the experimental warmup or recovery modifications. Its report path is written by the evaluator; choose a new filename to preserve earlier results.

## What has and has not been shown

Earlier evaluation on Apple M4/MPS:4/5 successes on known initial states,0/1 on a separate smoke state, and11/20 on newly fixed initial states. These are separate groups. New-state rollouts took1.11–3.22seconds each, excluding model loading. Slow first action calls were observed; no real-time deadline is guaranteed. The bundled evaluator uses the environment's original success condition (raw coverage>0.95). Reward is normalized coverage, not raw overlap.

Those are earlier simulation results, not new trials from this packaging run or proof of physical-robot performance, third-party adoption or human time saved. The prior migration checked all207 core tensors and processor statistics against the official source and strictly loaded on unmodified LeRobot. The kit's own reconstruction check is documented in the accompanying packaging receipt. Checksums establish consistency with this bundle; they are not a signed supply-chain guarantee.
