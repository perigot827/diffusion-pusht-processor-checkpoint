---
license: apache-2.0
datasets:
- lerobot/pusht
tags:
- vqbet-policy
- model_hub_mixin
- pytorch_model_hub_mixin
- robotics
pipeline_tag: robotics
---
# Model Card for VQ-BeT/PushT

VQ-BeT (as per [Behavior Generation with Latent Actions](https://arxiv.org/abs/2403.03181)) trained for the `PushT` environment from [gym-pusht](https://github.com/huggingface/gym-pusht).

## How to Get Started with the Model

See the [LeRobot library](https://github.com/huggingface/lerobot) (particularly the [evaluation script](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/eval.py)) for instructions on how to load and evaluate this model.

## Training Details

Trained with [LeRobot@3c0a209](https://github.com/huggingface/lerobot/tree/3c0a209f9fac4d2a57617e686a7f2a2309144ba2).

The model was trained using [LeRobot's training script](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/train.py) and with the [pusht](https://huggingface.co/datasets/lerobot/pusht) dataset, using this command:

```bash
python lerobot/scripts/train.py \
    --output_dir=outputs/train/vqbet_pusht \
    --policy.type=vqbet \
    --dataset.repo_id=lerobot/pusht \
    --env.type=pusht \
    --seed=100000 \
    --batch_size=64 \
    --steps=250000 \
    --eval_freq=25000 \
    --save_freq=25000 \
    --wandb.enable=true
```

The training curves may be found at https://wandb.ai/aliberts/lerobot/runs/3i7zs94u.
The current model corresponds to the checkpoint at 200k steps.


## Model Size

<blank>|Number of Parameters
-|-
RGB Encoder | 11.2M
Remaining VQ-BeT Parts | 26.3M

## Evaluation

The model was evaluated on the `PushT` environment from [gym-pusht](https://github.com/huggingface/gym-pusht). There are two evaluation metrics on a per-episode basis:

- Maximum overlap with target (seen as `eval/avg_max_reward` in the charts above). This ranges in [0, 1].
- Success: whether or not the maximum overlap is at least 95%.

Here are the metrics for 500 episodes worth of evaluation.

Metric|Value
-|-
Average max. overlap ratio for 500 episodes | 0.895
Success rate for 500 episodes (%) | 63.8

The results of each of the individual rollouts may be found in [eval_info.json](eval_info.json). It was produced after training with this command:
```bash
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/vqbet_pusht/checkpoints/200000/pretrained_model \
    --output_dir=outputs/eval/vqbet_pusht/200000 \
    --env.type=pusht \
    --seed=100000 \
    --eval.n_episodes=500 \
    --eval.batch_size=50 \
    --device=cuda \
    --use_amp=false
```