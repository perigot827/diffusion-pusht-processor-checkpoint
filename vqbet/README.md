# VQ-BeT PushT processor kit

Reconstruct an already-verified VQ-BeT checkpoint with saved processors for unmodified LeRobot. This kit contains no learned model weights. It retrieves the official158MB model file at a fixed revision, verifies its SHA256, removes only six legacy normalization tensors, and checks the reconstructed model SHA256. Bundled config/statistic files are also checked. No training, Hub upload or migration patches are needed to reconstruct it. Existing output directories are rejected.

Source: https://huggingface.co/lerobot/vqbet_pusht/tree/15c2d0af889c401c7e5db7a07499d3b498afc276 . Original copyright/attribution is retained in UPSTREAM_MODEL_CARD.md and LICENSE. This is an AI-D community conversion, not an upstream release or endorsement.

## Use

In a new directory, extract `vqbet-pusht-kit.zip` and enter its `vqbet-pusht-kit` directory. With `uv` and Git installed, these commands create a Python 3.12 environment and select the verified LeRobot source commit and key runtime versions:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  "lerobot[pusht] @ git+https://github.com/huggingface/lerobot.git@89236ea0f4f81a81ca566081e20dd1ff5f823cbe" \
  'torch==2.11.0' 'torchvision==0.26.0' \
  'gym-pusht==0.1.6' 'gymnasium==1.3.0' 'pymunk==6.11.1'
.venv/bin/python prepare.py
SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
  .venv/bin/python evaluate_vqbet.py --checkpoint vqbet-pusht \
  --device cpu --seed 200011 --report episode.json
```

These shell paths are for macOS/Linux. This release was checked on macOS 26.6.2, Apple M4, Python 3.12.13; Linux, Windows and CUDA were not tested. Runtime dependencies require additional download space beyond the 158 MB source weights. `diffusers` is not needed by this VQ-BeT evaluator.

The fresh-environment check installed a local `git archive` of exactly the commit above with `pusht`, gym-pusht 0.1.6 and pymunk 6.11.1, fetching dependencies online. It resolved to the runtime versions pinned above. The Git URL install transport itself was not re-exercised. Model reconstruction reused a previously downloaded original weight file:

```sh
.venv/bin/python prepare.py --source /path/to/original/model.safetensors --output my-vqbet
```

The fixed evaluator completed one CPU setup smoke (seed 200011, 95 steps, success) and wrote a finite report without `diffusers` installed. This is setup validation on a previously used seed, not a new success-rate benchmark. See `setup-verification.json`. The earlier `reconstruction-verification.json` records the original packaging run; its no-fresh-install/no-new-inference fields apply to that earlier run.

### Fix in vqbet-v2026.09.15.1

The original companion evaluator asked for the installed `diffusers` version when constructing the final report. A clean `pusht` installation does not install that optional library, so this lookup could raise `PackageNotFoundError` after the episode. This patch removes only that unused metadata lookup. Policy execution, checkpoint files and the original evaluation groups are unchanged. The historical release remains available.

On a compatible Apple Silicon environment, `--device mps` selects MPS. The saved default is CPU. Model loading uses strict state-dictionary matching. The evaluator does not include the experimental warmup or recovery modifications. Its report path is written by the evaluator; choose a new filename to preserve earlier results.

## What has and has not been shown

Earlier evaluation on Apple M4/MPS:4/5 successes on known initial states,0/1 on a separate smoke state, and11/20 on newly fixed initial states. These are separate groups. New-state rollouts took1.11–3.22seconds each, excluding model loading. Slow first action calls were observed; no real-time deadline is guaranteed. The bundled evaluator uses the environment's original success condition (raw coverage>0.95). Reward is normalized coverage, not raw overlap.

Those are earlier simulation results, not new trials from this packaging run or proof of physical-robot performance, third-party adoption or human time saved. The prior migration checked all207 core tensors and processor statistics against the official source and strictly loaded on unmodified LeRobot. The kit's own reconstruction check is documented in the accompanying packaging receipt. Checksums establish consistency with this bundle; they are not a signed supply-chain guarantee.
