# Implementation: Step 5 — Hyperparameter Tuning

This step addresses training hyperparameters identified as likely root causes of the
20–30% win-rate ceiling observed after Step 4. It is directly inspired by the findings
in Jett Wang's 2024 MIT M.Eng thesis, *"Winning at Pokémon Random Battles Using
Reinforcement Learning"* (available at https://dspace.mit.edu/handle/1721.1/153888).

---

## Motivation

After Step 4, `eval/mean_reward` plateaued around −23 to −25 and win rate vs
SimpleHeuristicsPlayer sat at 20–30%. Wang (2024), training a PPO agent for Gen 4
Random Battles, reached ~80% vs SimpleHeuristicsPlayer by ~40M steps and peaked at
rank 8 on the online ladder (1693 Elo). Their setup used the same SB3/PPO framework,
making it a direct comparison point.

Two root causes were identified:

1. **`gamma=0.99` is too low for Gen 3's long games.** With games averaging 30–100
   turns, the terminal win signal (`±30.0`) is discounted by `0.99^T` at turn `T`:

   | Turn | Discounted win value |
   |------|---------------------|
   | 30 | 22.2 |
   | 50 | 18.2 |
   | 75 | 14.2 |
   | 100 | 11.0 |

   By turn 100, a win is worth only 11 — barely more than a single faint (±2.0).
   Per-turn shaping bonuses firing in the first 20 turns are nearly undiscounted,
   causing the model to optimise shaping noise over the actual game outcome. Wang
   used `gamma=0.9999` for 25-turn Gen 4 games; Gen 3's longer episodes make the
   case even stronger.

2. **Constant LR.** Wang found that a constant learning rate caused the validation
   win rate to plateau at ~55%. Their annealing schedule reached ~80%. This is
   documented as a future item (see `todo.md` §1).

---

## What Was Changed

### 1. Gamma: 0.99 → 0.9999

**`src/main/train_rl_agent.py`**

```python
# Before
gamma=0.99,

# After
gamma=0.9999,
```

At `gamma=0.9999`, the terminal win signal retains >99% of its value even at turn 100:

| Turn | Discounted win value (0.9999) |
|------|-------------------------------|
| 30 | 29.9 |
| 50 | 29.9 |
| 100 | 29.7 |

The win signal now dominates the shaping bonuses throughout the full episode length,
including games that hit the stall-tax window (turn 125+).

### 2. GAE Lambda: 0.95 → 0.85

**`src/main/train_rl_agent.py`**

```python
gae_lambda=0.85,
```

SB3's default of 0.95 leans toward Monte Carlo returns (low bias, high variance). For
100-turn Gen 3 games with `gamma=0.9999`, far-future rewards contribute almost fully to
every advantage estimate, amplifying that variance considerably. Lowering to 0.85 trusts
the value function a bit more, reducing noise in advantage estimates. The thesis used 0.754;
0.85 is a conservative step in that direction without fully committing to a value tuned
for a different format.

### 3. clip_range: 0.2 → 0.15 and lr: 3e-4 → 1.5e-4

**`src/main/train_rl_agent.py`**

Motivated by live TensorBoard diagnostics at 33M steps of the first run with the
gamma/gae_lambda changes:

- `clip_fraction = 0.23–0.26` — policy was regularly hitting the clip boundary,
  meaning it was trying to take larger steps than clip_range=0.2 allowed. Healthy
  target is ~0.1–0.15. Clear sign the LR was too aggressive.
- `approx_kl = 0.028–0.030` — slightly above the typical 0.01–0.02 stable range.

`clip_range` lowered from 0.2 to 0.15 and default `lr` halved from 3e-4 to 1.5e-4.
Run was resumed from the 33M-step checkpoint rather than restarted from scratch.

---

## Reward Signal Summary (unchanged from Step 4)

No reward constants were modified in this step. The table below is reproduced for
reference — the gamma change alters the *effective weight* of each signal in the
discounted return without touching the constants themselves.

| Signal | Scale | Notes |
|--------|-------|-------|
| Faint | ±2.0 | base |
| Victory | ±30.0 | terminal |
| Switch subsidy | +0.5 | one-time per voluntary switch (× spam_mult) |
| Status infliction/cure | ±0.3 | one-time on the event turn |
| Sleep swap out | +0.25 | one-time, voluntary only |
| Sleep swap in | −0.25 | one-time, unless phazing |
| SE switch-in | +0.2 | one-time; skipped if phazing |
| Roar bonus | +0.2 | one-time on Roar turn |
| Failed Roar penalty | −0.2 | one-time on Roar turn |
| Pivot bonus | +0.1–0.15 | one-time; skipped if phazing |
| Spikes layer set | +0.5 | per layer added |
| Spikes waste | −0.2 | used Spikes at 3-layer cap |
| Matchup penalty | −0.15 | per turn in a known-bad matchup |
| Futile attack | −0.05 | damaging move, opp net-healed this turn |

---

## Files Changed

| File | Change |
|------|--------|
| `src/main/train_rl_agent.py` | `gamma` 0.99 → 0.9999; `gae_lambda` 0.95 → 0.85; `clip_range` 0.2 → 0.15; default `lr` 3e-4 → 1.5e-4 |
| `designs/ai_v3/todo.md` | LR annealing added as §1; clip_range + LR reduction added as §2 |

---

## What's Next

See `designs/ai_v3/todo.md`. Priority items after this step:

1. **LR annealing** — implement the Wang schedule `ℓ(x) = peak / (8x+1)^1.5` via SB3's
   callable `learning_rate` interface. Expected to lift win rate significantly based on
   the thesis result (55% → 80% from annealing alone).
2. **Stat-stage deltas in `TurnDelta`** — Calm Mind / Dragon Dance boosts; Intimidate
   on switch-in. Needed to reward setup moves and penalise unchecked opponent boosts.
3. **Turn-history memory** — sliding window of K `TurnDelta` blocks, then GRU.
