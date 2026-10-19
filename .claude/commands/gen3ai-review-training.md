---
description: Reviews a model's TensorBoard training plots from a folder of PNG images and produces a structured training health report. Use this skill whenever the user wants to analyze training results, review training charts, understand how a model is progressing, check win rates, diagnose training instability, or assess PPO health. Triggers on phrases like "review training", "analyze training plots", "look at the training images", "how is training going", or when the user provides a path that looks like a tb_imgs folder. If the user pastes a folder path and asks anything about the training run, use this skill.
---

# /gen3ai-review-training

Given a folder of TensorBoard PNGs (written by `src/utils/plot_tb.py`, one PNG per metric group, paged as `<group>_<n>.png` when large), read every image with the Read tool and write a training health report. Run `ls <path>/*.png` first. Groups are the tag prefixes: `eval`, `rollout`, `train`, `time`, `hparams`, plus any `pinned` page. Each subplot's top-right badge is the final smoothed value: quote it, with the x-axis step count and the curve shape (climbing, levelling, plateaued, spiking).

Report, in this order:

1. **Win rates**: each opponent's current rate, trend, and any that is low or stagnant. An ELO is quoted only from `ladder.json` by the rules in `designs/ops/training_runbook.md`, never from these plots.
2. **PPO health**, rules of thumb (check the run's own settings before calling one wrong): `approx_kl` 0.005-0.02 (below: LR too low; above: unstable), `clip_fraction` 0.05-0.15, `value_loss` without spikes (> 3x baseline), `explained_variance` above 0.7 (below 0.5: the value function is not fitting), learning rate and `n_epochs` changes, `entropy_loss` rising gradually, `loss` spikes.
3. **Rollout**: `ep_rew_mean` and `ep_len_mean` trends.
4. **Hyperparameters**: a rising `ent_coef` means the system is fighting entropy collapse; flag any other schedule change.
5. **Anomalies**: spikes, drops and discontinuities with approximate steps.
6. **Overall**: 2-3 sentences, then the top 1-2 suggestions specific to these charts (flat win rates with stable, low KL suggests a local optimum, not generic advice).
