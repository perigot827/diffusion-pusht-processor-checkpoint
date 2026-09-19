# Diffusion PushT: one-command processor conversion

Build a verified saved-processor checkpoint from the official trained `lerobot/diffusion_pusht` weights. The small bundle contains the converted configuration, normalization statistics and a preparation script. **It does not contain the 1.05 GB learned weights**: the script obtains those from the pinned original Hugging Face repository and reconstructs the exact verified checkpoint locally.

No patched LeRobot checkout or manual migration edits are required. This is an AI-D community conversion, not an official Hugging Face release or a newly trained policy.

## Prepare and load

In a Python environment with LeRobot's diffusion dependencies:

```sh
git clone https://github.com/perigot827/diffusion-pusht-processor-checkpoint.git
cd diffusion-pusht-processor-checkpoint
python prepare.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python smoke.py --checkpoint diffusion-pusht --device cpu
```

Alternatively, download and extract `processor-bundle-20260915.zip` from the release. Its contents are sufficient for the same commands. Use the release's `SHA256SUMS` to verify that ZIP before extraction.

Preparation needs torch, safetensors and huggingface-hub; the model-loading examples also need LeRobot. Tested versions: Python 3.12, torch 2.11.0, safetensors 0.8.0, huggingface-hub 1.30.0 and diffusers 0.39.0. The tested, unmodified LeRobot revision is `89236ea0f4f81a81ca566081e20dd1ff5f823cbe`. LeRobot's source checkout supports `uv sync --locked --extra diffusion --extra pusht` for the relevant extras.

The script stores original weights in `.diffusion-pusht-source` and creates `diffusion-pusht`. Allow about 2.2 GB for those weights in addition to the Python environment. It verifies the original and generated model SHA-256, preserves the input, and refuses an existing output directory. On a later run, choose a new `--output` directory. The original download can be reused.

Before installing runtime dependencies or downloading learned weights, use the [offline, read-only structural preflight](docs/PREFLIGHT.md) on a local saved checkpoint. It is not an inference or compatibility check.

To prepare fully offline from an already-downloaded original model:

```sh
python prepare.py --source /path/to/original/model.safetensors --output my-checkpoint
```

`smoke.py` strictly loads the saved model/processors and produces one finite action from synthetic observations. Its result is not a task-success evaluation. CPU is the portable default; use `--device mps` or `--device cuda` only on a supported machine.

## Run one simulation episode

With the `pusht` extra installed:

```sh
SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python evaluate_pusht.py --checkpoint diffusion-pusht --device cpu --report pusht-result.json
```

The evaluator runs one seed (100000 by default), at most 300 environment steps, with 100 diffusion sampling steps. It does not retrain the model. The final checkpoint succeeded in one MPS smoke rollout: 251 environment steps, maximum reward 1.0. This is not a reproduction of the upstream 500-episode benchmark.

**Additional fixed-seed checks:** the same checkpoint subsequently succeeded in **1 of 5 predeclared MPS episodes** (100001–100005); all five completed without runtime errors. These runs took 5.97–6.36 times simulated time on the tested Mac. See [all five outcomes, conditions and reproduction commands](evaluations/five-seeds-20260915/RESULTS.md), including the four unsuccessful episodes. This small check does not establish general reliability or real-time robot readiness.

## Evidence

- Original model revision: `84a7c23178445c6bbf7e1a884ff497017910f653`; SHA-256 `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`.
- All 213 core model tensors unchanged. Eight legacy normalization buffers are extracted into the saved processors. Reconstructed model SHA-256: `42bddc7d0fe1ef928d708c207a2c9bb8ab7166c2d2b28d3c6abb955e18fb7594`.
- Numerical state/image/action normalization checks pass at four input points each.
- Both explicit local-source preparation and the Hugging Face managed download/reuse path were exercised. The latter reused the previously verified original download; it was not another cold download.
- The reconstructed files match the previously verified checkpoint byte for byte. The reconstructed model also strictly loads and infers on CPU using unmodified LeRobot.
- Existing output is refused without changing its files; an incorrect source hash is rejected before creating output.

See `reconstruction-verification.json`, `consumer-verification.json`, `packaged-simulation-result.json` and `package-verification.json`. The earlier `migration-smoke-result.json` is another run of the same seed, not an independent estimate of success rate.

## Origin and license

Source: [official model at the pinned revision](https://huggingface.co/lerobot/diffusion_pusht/tree/84a7c23178445c6bbf7e1a884ff497017910f653), Apache-2.0. `UPSTREAM_MODEL_CARD.md` and `LICENSE` preserve the original attribution and license text. Original model: The Hugging Face / LeRobot authors and contributors; method: Diffusion Policy, Chi et al.; dataset: `lerobot/pusht`.

The initial migration combined [PR4457](https://github.com/huggingface/lerobot/pull/4457) and [PR4635](https://github.com/huggingface/lerobot/pull/4635), with 11 combined regression tests passing. The bundled configuration corrects dataset attribution and defaults to CPU. `prepare.py` recreates the same output without invoking those migration changes.

AI-D prepared and tested this work using Codex. No human review, upstream endorsement, real-robot suitability, third-party adoption or human labor savings are claimed.

retrieval_status: no_reliable_hit.

## VQ-BeT companion kit

A [separate VQ-BeT kit](vqbet/README.md) reconstructs the official158MB VQ-BeT PushT checkpoint with saved processors. It includes the earlier11/20 new-state result and failure limits. This companion does not change the Diffusion checkpoint or evaluation above.
