"""One fixed-seed simulation after official Diffusion checkpoint migration.

This is a smoke rollout, not a reproduction of the 500-episode benchmark.
Copyright 2026 AI-D contributors. Licensed under Apache-2.0.
Run in a compatible LeRobot environment; no migration patches are required.
"""
import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

parser = argparse.ArgumentParser(description="Evaluate one fixed-seed PushT episode with a local checkpoint.")
parser.add_argument("--checkpoint", type=Path, default=Path("diffusion-pusht"))
parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
parser.add_argument("--seed", type=int, default=100000)
parser.add_argument("--report", type=Path)
args = parser.parse_args()
DEST = args.checkpoint
SEED = args.seed
torch.set_num_threads(4)
torch.manual_seed(SEED)
np.random.seed(SEED)
cfg = PreTrainedConfig.from_pretrained(DEST, local_files_only=True)
cfg.device = args.device
policy = DiffusionPolicy.from_pretrained(DEST, config=cfg, local_files_only=True, strict=True).eval()
pre, post = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=str(DEST),
    preprocessor_overrides={"device_processor": {"device": cfg.device}},
)
policy.reset()
trajectory = []
start = time.monotonic()
with gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
              observation_width=96, observation_height=96, max_episode_steps=300) as env:
    obs, _ = env.reset(seed=SEED)
    print(f"Loaded official migrated policy; device={cfg.device}; steps={cfg.num_inference_steps or cfg.num_train_timesteps}", flush=True)
    with torch.inference_mode():
        for step in range(300):
            batch = pre({
                "observation.image": torch.from_numpy(obs["pixels"].copy()).permute(2, 0, 1).float() / 255,
                "observation.state": torch.as_tensor(obs["agent_pos"], dtype=torch.float32),
            })
            action = post(policy.select_action(batch)).squeeze(0).cpu().numpy()
            assert action.shape == (2,) and np.isfinite(action).all()
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory.append({"step": step, "action": action.tolist(), "reward": float(reward),
                               "terminated": bool(terminated), "truncated": bool(truncated),
                               "is_success": bool(info.get("is_success", False))})
            if step % 8 == 0 or terminated or truncated:
                print(json.dumps(trajectory[-1] | {"elapsed_seconds": round(time.monotonic()-start, 2)}), flush=True)
            if terminated or truncated:
                break
result = {
    "retrieval_status": "no_reliable_hit", "seed": SEED, "episodes": 1,
    "device": cfg.device, "steps": len(trajectory), "all_actions_finite": True,
    "maximum_reward": max(row["reward"] for row in trajectory),
    "success": any(row["is_success"] for row in trajectory),
    "terminated": trajectory[-1]["terminated"], "truncated": trajectory[-1]["truncated"],
    "elapsed_seconds": time.monotonic()-start,
    "diffusion_inference_steps": cfg.num_inference_steps or cfg.num_train_timesteps,
    "versions": {name: version(name) for name in ("torch", "diffusers", "gym-pusht", "gymnasium", "pymunk")},
    "benchmark_reproduction": False, "hardware_tested": False,
    "third_party_adoption_or_human_labor_saving_verified": False,
}
if args.report:
    args.report.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
