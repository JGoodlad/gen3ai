---
description: Reviews a model's TensorBoard training plots from a folder of PNG images and produces a structured training health report. Use this skill whenever the user wants to analyze training results, review training charts, understand how a model is progressing, check win rates, diagnose training instability, or assess PPO health. Triggers on phrases like "review training", "analyze training plots", "look at the training images", "how is training going", or when the user provides a path that looks like a tb_imgs folder. If the user pastes a folder path and asks anything about the training run, use this skill.
---

# /gen3ai-review-training

The user has provided a path to a folder of PNG training charts exported from TensorBoard. Your job is to read every image in that folder and produce a clear, actionable training health report.

## Step 1 — List the images

Run `ls <path>/*.png` (or equivalent) to see what files are present before reading anything. Note the filenames — they tell you which metric group each image covers.

## Step 2 — Read every PNG

Use the Read tool on each PNG file. It renders images natively so you can see the charts directly. Read them all before writing your report — you need the full picture.

## Image naming conventions

The plot script (plot_tb.py) writes one PNG per metric group, splitting large groups into pages:

| Filename pattern | Contents |
|---|---|
| `eval.png` / `eval_1.png`, `eval_2.png` … | Win rates, episode lengths, mean rewards per opponent |
| `train.png` / `train_1.png`, `train_2.png` … | PPO internals: KL, clip fraction, losses, explained variance, LR, n_epochs |
| `rollout.png` | Episode reward mean and episode length mean during training |
| `hparams.png` / `hparams_1.png` … | Hyperparameter schedules: ent_coef, gae_lambda, gamma, vf_coef |

Each subplot has a **badge in the top-right corner** showing the final smoothed value — read these for exact current numbers without squinting at axes.

Win rate subplots are pinned to a 0–1 y-axis. All others use auto-scaling.

## Step 3 — Write the report

Structure your report exactly as follows. Be specific: quote the badge values, note the step count on the x-axis, call out the shape of the curve (still climbing, levelling off, plateauing, spiking).

---

### Win Rates
For each opponent (Random, Heuristic, Aggressive, Staller, SetupSweep, Bots), report:
- Current win rate (from the badge)
- Trend: climbing / slowing / plateaued / declining
- Flag any opponent where win rate is unexpectedly low or stagnant

### PPO Health
Report on each of these, using badge values where available:

- **approx_kl** — healthy range 0.005–0.02. Below 0.005: too conservative, LR probably too low. Above 0.02: unstable updates.
- **clip_fraction** — healthy 0.05–0.15. Sustained spikes suggest large policy jumps.
- **value_loss** — should be stable. Spikes (>3× baseline) indicate value function instability.
- **explained_variance** — above 0.7 is healthy; below 0.5 means the value function isn't fitting well.
- **learning_rate** — current level and whether it's still changing.
- **entropy_loss** — should rise gradually as the policy becomes more confident (more negative → less negative over time).
- **n_epochs** — note if it has changed; the adaptive KL callback reduces this when KL runs hot.
- **loss** — note any spikes.

### Rollout
- Current `ep_rew_mean` and trend
- Current `ep_len_mean` — shorter episodes suggest more decisive play

### Hyperparameters
- `ent_coef`: is it stable, rising, or falling? A rising ent_coef means the system is fighting entropy collapse — worth flagging.
- Note any other hparam changes.

### Anomalies
List any spikes, sudden drops, or discontinuities with approximate step counts where they occurred.

### Overall Assessment
2–3 sentences on training health overall. Then list the top 1–2 concrete, actionable suggestions — not generic advice, but things specific to what you actually observed in these charts.

---

## What good looks like (for context)

This is a Gen3OU Pokémon RL agent trained with PPO against a fixed pool of scripted opponents. Healthy training looks like:
- Win rates vs non-random bots slowly climbing toward 70–80%
- approx_kl staying in the 0.007–0.013 band
- ep_rew_mean trending upward over the full run
- explained_variance above 0.75
- ent_coef roughly stable (not needing to keep rising)

A rising ent_coef, flat win rates, and stable-but-low KL together suggest the policy may be converging toward a local optimum driven by shaped rewards rather than win/loss.
