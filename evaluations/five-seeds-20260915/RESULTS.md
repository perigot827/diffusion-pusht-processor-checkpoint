# Five fixed-seed simulation checks

The converted Diffusion PushT checkpoint succeeded in **1 of 5 predeclared episodes** on MPS. All five completed without a runtime error and all actions were finite. Unsuccessful episodes are retained in this report.

| Seed | Success | Steps | Maximum normalized reward | Seconds |
|---|---|---:|---:|---:|
| 100001 | No | 300 | 0.948016 | 183.7 |
| 100002 | No | 300 | 0.983288 | 179.0 |
| 100003 | Yes | 129 | 1.000000 | 82.1 |
| 100004 | No | 300 | 0.995482 | 182.2 |
| 100005 | No | 300 | 0.995721 | 188.4 |

Seeds 100001–100005 were fixed before execution. The previously reported successful seed 100000 is excluded. Each episode ran in a separate process, with the existing evaluator unchanged, at most 300 environment steps and 100 diffusion sampling steps. The model was not trained or retuned between episodes.

The environment advances 0.1 simulated seconds per step (its configured control rate is 10 Hz). On this Mac/MPS run, elapsed rollout time was 5.97–6.36 times simulated time. The elapsed timer excludes initial model loading. This is unpaced headless simulation, not real-time robot control; these measurements do not establish that control deadlines can be met.

Success uses the installed gym-pusht environment's `info.is_success`: target coverage must exceed 0.95. Its reward is `clip(coverage / 0.95, 0, 1)`, so the reward column is **not raw overlap coverage**. No direct numerical comparison to the upstream model card's raw-overlap metric is made.

These are five fixed initial states, not an estimate of general reliability, a reproduction of the upstream 500-episode benchmark, or a real-robot test. The outcome does not establish third-party adoption or human labor savings.

## Reproduce

Use the checkpoint and environment described in the repository README. For each predeclared seed, run the existing evaluator:

```sh
SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python evaluate_pusht.py --checkpoint diffusion-pusht --device mps --seed 100001 --report seed-100001.json
```

Repeat for 100002, 100003, 100004 and 100005. CPU and CUDA outcomes were not tested in this five-seed check. Cross-device or cross-version bitwise reproducibility is not claimed.

LeRobot source: `89236ea0f4f81a81ca566081e20dd1ff5f823cbe`, unchanged. Checkpoint model SHA-256: `42bddc7d0fe1ef928d708c207a2c9bb8ab7166c2d2b28d3c6abb955e18fb7594`. Evaluator SHA-256 and complete checkpoint-file hashes were recorded in `plan.json` before execution. Each seed report records package versions and termination status.

Prepared and executed by AI-D. No human review or upstream endorsement is claimed.

retrieval_status: no_reliable_hit.
