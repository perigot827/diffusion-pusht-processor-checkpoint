"""Load the local converted checkpoint and generate one finite action.

Copyright 2026 AI-D contributors. Licensed under Apache-2.0.
This checks loading and inference; it does not evaluate task success.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("diffusion-pusht"))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(100000)
    start = time.monotonic()
    config = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    config.device = args.device
    policy = DiffusionPolicy.from_pretrained(
        args.checkpoint, config=config, local_files_only=True, strict=True
    ).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    batch = pre({
        "observation.state": torch.tensor([256.0, 256.0]),
        "observation.image": torch.zeros(3, 96, 96),
    })
    policy.reset()
    with torch.inference_mode():
        action = post(policy.select_action(batch))
    assert action.shape == (1, 2) and torch.isfinite(action).all()
    print(json.dumps({
        "status": "PASS", "device": args.device,
        "action_shape": list(action.shape), "action": action.cpu().tolist(),
        "elapsed_seconds": time.monotonic() - start,
        "strict_model_and_saved_processors_loaded": True,
        "task_success_evaluated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
