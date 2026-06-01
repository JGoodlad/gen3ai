# Design: Reward Annealing

Gradually scale the **shaping** reward terms toward zero (or a floor) over a step window,
leaving the **outcome** terms intact. This is the gating prerequisite for league play and
the precondition for the v6 MCTS value head.

## Motivation

Two independent reasons, both load-bearing:

1. **Anti-reward-hacking.** The shaping terms (`switch_base`, `pivot_*`, `matchup_penalty`,
   `se_switch`, …) are human strategic priors. Early in training they accelerate learning;
   late in training a strong policy can farm them at the margins in ways that are orthogonal
   to — or mildly against — actually winning (e.g. taking a "free" subsidised switch that
   loses tempo). Annealing them removes the incentive once the policy no longer needs the
   scaffolding.

2. **The value head must estimate win probability, not shaped return.** In v6, MCTS uses the
   PPO value head `V_θ(s)` as the **leaf evaluator** for PIMC search (Wang 2024). If `V_θ`
   was trained against a heavily shaped reward, it estimates "expected discounted shaped
   return", which is not the quantity PUCT wants at a leaf. The closer `V_θ` is to win
   probability, the better the search. Annealing shaping toward zero before the value head
   is frozen for MCTS is what makes the head usable as a leaf evaluator.

This is why the Step 2 league prerequisite is "reward annealing ≥ 50% complete" — the value
head needs most of its late-training gradient to come from the outcome, not the scaffolding.

---

## The term taxonomy — what anneals, what doesn't

The reward components are defined in `RewardBreakdown` (`reward_manager.py`) and grouped by
`_GROUPS`. Annealing splits them into **three tiers**, which is the central design decision
of this document:

### Tier A — Strategic priors → anneal to **0**

These are pure heuristics that encode "good Gen 3 play" and carry the reward-hacking risk.
They should reach exactly zero by `anneal_end`.

| Group | Fields |
|-------|--------|
| attack | `roar`, `futile_attack`, `futile_setup`, `setup_low_hp`, `boost_utilized`, `status_wasted` |
| switch | `switch_base`, `escape_threat_switch`, `pivot_protect`, `pivot_status`, `pivot_damage`, `se_switch`, `sleep_out`, `sleep_in` |
| field | `spikes`, `matchup_penalty`, `dead_matchup_tax`, `status` |

### Tier B — Outcome proxies → **keep (never anneal)**

The `base` group is the dense, outcome-correlated signal. `win_loss` (`VICTORY_VALUE = 30`)
is the terminal truth; HP and faint deltas are a *material* proxy that is monotone-aligned
with winning and provides the credit-assignment density PPO needs over 30+ turn games.
Annealing these to zero would collapse the problem to a single ±30 terminal reward — the
classic sparse-reward catastrophe — and is **not** recommended.

| Group | Fields |
|-------|--------|
| base | `hp_ours`, `hp_opp`, `faint_ours`, `faint_opp`, `win_loss`, `explosion`, `explosion_block`, `finishing_blow` |

Because `VICTORY_VALUE` (30) already dwarfs per-turn HP/faint magnitudes, keeping Tier B
still yields a value head dominated by the outcome — close enough to win probability for MCTS
leaf evaluation, and far more stable to train than pure-terminal. If a later v6 ablation shows
`V_θ` is too material-biased for good search, add an **optional final outcome-only phase** that
anneals Tier B as well (a second, later window).

### Tier C — Anti-degenerate taxes → anneal to a **floor**, not zero

These do not encode strategy — they guard against *environment exploits* (infinite stalling,
struggle loops, switch-bouncing to dodge a clock). If annealed fully to zero, a strong policy
may rediscover the degenerate behaviour to force a turn-limit draw. They anneal toward a small
floor (a fraction of their initial weight), **not** zero, unless terminal turn-limit handling
already makes stalling a guaranteed loss.

| Field | Guards against |
|-------|----------------|
| `repetition_tax` | same attack spammed (floored by `REPETITION_TAX_FLOOR`) |
| `struggle_tax` | struggle loop |
| `stall_tax` | progressive non-action stalling |
| `switch_bouncing_tax` | A↔B switch oscillation to burn the clock |

> **Open decision:** if turn-limit handling is changed so a stall-to-cap is scored as a loss
> (not a draw), Tier C can fold into Tier A and anneal to zero. Until then, keep the floor.

---

## Annealing schedule

A single scalar coefficient `λ(t) ∈ [0, 1]` per tier, computed from the **global** timestep
`t = num_timesteps`:

```python
def anneal_coef(t, start, end, floor=0.0):
    if t <= start:
        return 1.0
    if t >= end:
        return floor
    x = (t - start) / (end - start)          # 0 → 1
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * x))   # cosine 1 → floor
```

- **Tier A:** `floor = 0.0`.
- **Tier C:** `floor = TIER_C_FLOOR` (default `0.25`).
- **Tier B:** not annealed (`λ ≡ 1.0`).

Cosine (not linear) matches `adaptive_lr_callback.py` and keeps the rate-of-change small at
both ends, minimising value-target shock. `t` is the global step counter, which is preserved
across checkpoint loads — so annealing resumes correctly after a launcher restart, exactly
like the LR anneal.

---

## Implementation design

The hard part is plumbing: reward shaping runs **inside the env subprocesses**
(`SubprocVecEnv`), which do not know the global training step. The model and its
`num_timesteps` live in the main process.

**Use SB3's `set_attr` push (the idiomatic VecEnv path), not shared memory.** A callback in
the main process recomputes the coefficients each rollout and pushes them into every worker:

### New: `src/agents/training/reward_anneal_callback.py`

```python
class RewardAnnealCallback(BaseCallback):
    def __init__(self, start, end, tier_c_floor=0.25):
        ...
    def _on_rollout_start(self) -> None:
        t = self.num_timesteps
        coefs = {
            "A": anneal_coef(t, self.start, self.end, 0.0),
            "C": anneal_coef(t, self.start, self.end, self.tier_c_floor),
        }
        # Dispatches to all SubprocVecEnv workers; each Gen3Env stores it and
        # forwards to its Gen3RewardManager on the next process_turn_reward().
        self.training_env.set_attr("_reward_anneal_coefs", coefs)
        self.logger.record("train/reward_anneal_A", coefs["A"])
        self.logger.record("train/reward_anneal_C", coefs["C"])
```

### Modified: `Gen3RewardManager`

`process_turn_reward()` scales each Tier-A field by `coefs["A"]` and each Tier-C field by
`coefs["C"]` when assembling the `RewardBreakdown.total`. The per-field values stored in the
breakdown (for logging/replay) remain **un-scaled** so the TensorBoard/JSON breakdown still
shows the raw shaping magnitudes — only the summed `total` returned to PPO is scaled. Tier B
is untouched. Default coefs are `{"A": 1.0, "C": 1.0}` so behaviour is identical when the
callback is absent (debug runs, unit tests).

The tier membership lives next to `_GROUPS` as an explicit `_ANNEAL_TIER` map so the taxonomy
above is the single source of truth and a new reward field must declare its tier.

---

## Schedule coupling with self-play / league

| Phase | Step window (example) | Anneal state |
|-------|----------------------|--------------|
| Fixed-bot + early self-play | `t < anneal_start` | full shaping (`λ_A = 1`) |
| Late self-play | `[anneal_start, anneal_end]` | ramping down |
| League play (Step 2) | `t ≥ anneal_end` | Tier A at 0, Tier C at floor |

Trigger `anneal_start` when `eval/win_rate_vs_bots` has been **flat for ≥ 10M steps** (the
self-play curriculum has saturated — see `designs/ai_v5/todo.md` Step 1 deferred items).
`anneal_end` should land **before** league play begins so the league trains a
win-probability value head from its first step. The Step 2 CLI's
`--reward-anneal-start 50000000 --reward-anneal-end 70000000` is consistent with a 75M-step
self-play run.

---

## Stability notes

- Annealing shifts the reward scale → value targets drift downward over the window. PPO
  normalises advantages per batch, which absorbs most of this, but anneal **slowly** (a
  ≥ 15–20M step window) so the value loss does not spike.
- Watch `train/value_loss` and `train/explained_variance` across the window — a sustained
  `explained_variance` drop means the value head is struggling to track the moving target;
  widen the window.
- The KL-band adaptive LR (`adaptive_lr_callback.py`) and reward annealing can overlap. They
  are independent (LR responds to KL, reward to step), but if both are active during a rocky
  window, prefer to let LR annealing trail reward annealing slightly.

---

## Metrics

| Metric | Interpretation |
|--------|----------------|
| `train/reward_anneal_A` | Tier-A coefficient (1 → 0) — confirms the schedule is running |
| `train/reward_anneal_C` | Tier-C coefficient (1 → floor) |
| `train/value_loss`, `train/explained_variance` | Stability guard across the window |
| reward-breakdown JSON (`base`/`attack`/`switch`/`field`) | Raw (un-scaled) magnitudes for diagnosis |

---

## Files

| File | Change |
|------|--------|
| `src/agents/training/reward_anneal_callback.py` | **New.** `RewardAnnealCallback` + `anneal_coef()` |
| `src/agents/training/reward_manager.py` | `_ANNEAL_TIER` map; scale Tier-A/C in `total`; `_reward_anneal_coefs` default |
| `src/agents/training/gen3_env.py` | Hold `_reward_anneal_coefs`; forward to `Gen3RewardManager` |
| `src/main/train_rl_agent.py` | `--reward-anneal-start` / `--reward-anneal-end` / `--reward-anneal-tier-c-floor`; wire the callback |
| `src/main/launcher/*` | Forward the three flags (already verbatim-forwarded) |
| `src/agents/training/reward_anneal_callback_test.py` | **New.** Schedule unit tests (boundaries, cosine shape, floor) |
| `src/agents/training/reward_invariants_e2e_test.py` | Assert that at `λ_A = 0` only Tier B + floored Tier C survive in `total` |

---

## CLI

```bash
export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py \
  --model models/<run>/checkpoint.zip \
  --self-play \
  --reward-anneal-start 50000000 \
  --reward-anneal-end 70000000 \
  --reward-anneal-tier-c-floor 0.25 \
  --steps 75000000 \
  --device cuda
```

Omitting the flags leaves shaping at full strength for the whole run (current behaviour).

---

## Verification

1. **Schedule unit test**: `anneal_coef` returns 1.0 before start, the floor at/after end, and
   a monotone cosine between — for both Tier A (floor 0) and Tier C (floor 0.25).
2. **Invariant test**: with `λ_A = 0`, `reward_invariants_e2e_test.py` confirms `total` equals
   the Tier-B sum plus floored Tier-C, and that no attack/switch/field shaping leaks in.
3. **Push test**: in `--debug`, set a short window and confirm `set_attr` reaches the worker —
   `train/reward_anneal_A` falls and the env's effective reward `total` shrinks accordingly.
4. **Smoke**: `--debug --steps 20000 --reward-anneal-start 5000 --reward-anneal-end 15000`;
   confirm no value-loss explosion and episodes still complete.
