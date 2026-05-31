# Implementation: Step 7 — KL-Reactive LR Band Widening

This step widens the "happy band" of the KL-reactive learning-rate controller from
`[0.007, 0.013]` to **`[0.005, 0.02]`** by replacing the symmetric `kl_tolerance` knob
with the asymmetric `kl_factor` convention used by the modern standard KL-adaptive-LR
scheduler (skrl / RL Games `KLAdaptive`). It is a training-infrastructure tuning change:
**reward-neutral, observation-neutral, and not weight-relevant** — no `ARCH_SIGNATURE`
bump, no obs-dim change. The motivation is research-grounded: the old band was the
*narrowest* of every reference checked, including this repo's own training-review skill,
and it fought the naturally-low KL that the current run's tight `clip_range` produces.

---

## Background — the KL-reactive LR controller

Learning rate is driven by `approx_kl` in `adaptive_lr_callback.py`. Two callbacks share
the same Phase-1 logic:

- **`TwoPhaseLRCallback`** — used when `--anneal-lr-start-steps` is set. **Phase 1**
  (`num_timesteps < anneal_start_steps`) is KL-reactive; **Phase 2** is deterministic
  cosine decay that ignores KL entirely.
- **`AdaptivePPOCallback`** — pure KL-reactive mode (no cosine phase).

Each rollout, Phase 1:

1. Reads `train/approx_kl` (logged by `InstrumentedMaskablePPO`, the k3 estimator
   `(exp(logratio) − 1) − logratio`).
2. Folds it into an EMA (`ema_alpha = 0.20`, half-life ~3 rollouts).
3. If the EMA leaves the no-op band, nudges LR by `lr_factor` (×1.2 up / ÷1.2 down),
   bounded by `[min_lr, max_lr]`, then arms a `cooldown_rollouts = 7` suppression window.

Only `lr_factor`, `ema_alpha`, `cooldown_rollouts`, and the **band** govern behaviour. This
step changes only the band.

---

## The problem — the band was the narrowest in the literature

The old band came from a symmetric `kl_tolerance = 0.3`: `target_kl × (1 ± 0.3)` →
`[0.007, 0.013]`. That treats as "unhappy" KL values that every reference — including the
repo's own `gen3ai-review-training` skill — considers healthy:

| Source | Happy / no-op band | Target |
|---|---|---|
| **Old controller** | **[0.007, 0.013]** (symmetric ±30%) | 0.01 |
| skrl `KLAdaptive` (RL Games / Isaac Gym standard) | **[t/2, 2t]** → [0.005, 0.02] at t=0.01 (literal default t=0.008 → [0.004, 0.016]) | 0.008 |
| `gen3ai-review-training` skill (repo's own "safe" range) | [0.005, 0.02] | — |
| Classic PPO adaptive-KL deadband (Schulman 2017) | [t/1.5, 1.5t] → [0.0067, 0.015] | 0.01 |
| "37 Implementation Details of PPO" (ICLR 2022) | "approx_kl generally stays below 0.02" | 0.01 |
| OpenAI Spinning Up | — | 0.01 ("usually 0.01 or 0.05") |

The modern standard (skrl / RL Games `KLAdaptive`, the same controller *family* used here)
is an **asymmetric band of `[target_kl / kl_factor, target_kl × kl_factor]`**. A single
symmetric tolerance cannot express `[0.005, 0.02]` — a 0.005 floor symmetric about 0.01
caps the ceiling at 0.015 — so the parameterization itself had to change.

### Why it mattered for the current run

The active command uses `--clip-range 0.10` (half the PPO default of 0.2) and
`--n-epochs 8`. A tight clip caps the policy ratio at `[0.9, 1.1]` regardless of LR, so
per-update KL is *structurally suppressed* and naturally sits low — often below the old
0.007 floor. The controller then read "KL too low" and ratcheted LR **up** by 1.2× every
~8 rollouts toward `max_lr`, fighting a low KL that was simply a consequence of the clip.
A wider lower bound stops that fight.

---

## What changed

In **both** `TwoPhaseLRCallback` and `AdaptivePPOCallback`, the symmetric `kl_tolerance`
parameter/attribute was replaced with an asymmetric `kl_factor`, and the band computation
in `_adapt_lr_from_kl` / `_on_rollout_end` changed from

```python
lo = self.target_kl * (1.0 - self.kl_tolerance)   # 0.007
hi = self.target_kl * (1.0 + self.kl_tolerance)   # 0.013
```

to

```python
lo = self.target_kl / self.kl_factor              # 0.005
hi = self.target_kl * self.kl_factor              # 0.020
```

### Constants (single source of truth: callback defaults)

| Constant | Before | After | Note |
|---|---|---|---|
| band parameter | `kl_tolerance = 0.3` | **`kl_factor = 2.0`** | symmetric → asymmetric multiplicative |
| `target_kl` | 0.01 | 0.01 (unchanged) | keeps centre at the Spinning-Up / 37-details value |
| **no-op band** | **[0.007, 0.013]** | **[0.005, 0.02]** | ~4× wider |
| `lr_factor` | 1.2 | 1.2 (unchanged) | wider band already fires less often |
| `ema_alpha` | 0.20 | 0.20 (unchanged) | |
| `cooldown_rollouts` | 7 | 7 (unchanged) | |

`target_kl = 0.01` was kept (rather than skrl's literal 0.008) because it matches Spinning
Up / the 37-details example **and** makes `[t/2, 2t]` land exactly on `[0.005, 0.02]` — the
range the repo's review skill already documents as "safe," so controller and docs stay
self-consistent.

The clean replacement (no `kl_tolerance` compat shim) follows the project's rapid-iteration
rule — callback args are reconstructed each run from CLI defaults and are not serialized
into checkpoint weights, so there is nothing to migrate.

---

## Scope — what was deliberately left unchanged

- **No new CLI flags.** `target_kl` / `kl_factor` remain hardcoded callback defaults; the
  band changes for every run on the next launcher start/restart. (The callback is built
  once at startup, so a *running* child does not change mid-flight.)
- **`target_kl` is still not passed to the PPO model.** The model's `target_kl` stays
  `None`, so the early-stopping path in `InstrumentedMaskablePPO`
  (`continue_training = False` when a batch exceeds `1.5 × target_kl`) remains dead code —
  LR control is the only KL-reactive mechanism. Wiring that up is out of scope.
- **Phase 2 cosine is untouched.** The band governs only Phase 1; the deterministic cosine
  decay after `anneal_start_steps` ignores KL.

---

## Test status

- **`adaptive_lr_callback_test.py` — 42 passed.** The fixture and band-edge tests were
  re-pointed from the old `[0.007, 0.013]` / `kl_tolerance` to the new
  `[0.005, 0.02]` (and the `AdaptivePPOCallback` fixture's `target_kl = 0.015 → [0.0075,
  0.03]`); "high KL" probes moved from `0.030` (no longer above the wider ceiling) to
  `0.05`, and the EMA-arithmetic assertion was updated to the new seed.
- **Full unit suite — 1141 passed, 2 skipped** on the rebased tree (combined with the
  `gen3_turn_delta_v2` obs change), confirming no cross-interaction.
- Smoke test not run in this worktree (Showdown submodule/server not provisioned); the
  change is pure band arithmetic and exercises no pipeline the smoke test would uniquely
  cover.

The production-default construction (mirroring the live command) was verified to yield the
band `[0.005, 0.02]`.

---

## Files Changed

| File | Change |
|---|---|
| `src/agents/training/adaptive_lr_callback.py` | Both callbacks: `kl_tolerance` → `kl_factor` (default 2.0); band `[target_kl/kl_factor, target_kl*kl_factor]`; docstrings cite the skrl / RL Games convention and the [0.005, 0.02] band |
| `src/agents/training/adaptive_lr_callback_test.py` | Fixture + band-edge tests updated to the new band; high-KL probes 0.030 → 0.05; EMA assertion re-seeded |
| `.claude/commands/gen3ai-review-training.md` | "What good looks like" band synced 0.007–0.013 → 0.005–0.02 (the "safe range" line at 0.005–0.02 already agreed) |

*Shipped as commit `e4c305d` (`feat(training): widen KL-reactive LR band to [0.005, 0.02]
via kl_factor convention`).*
