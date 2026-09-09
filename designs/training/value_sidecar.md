# The training-side VALUE SIDECAR (`gen3_value_sidecar_v1`)

> Owned by `src/agents/training/CLAUDE.md` → *The training-side value sidecar*. Always current;
> update it in the same pass as `value_sidecar.py`, `value_sidecar_read.py` or either gate.

Code: `src/agents/training/value_sidecar.py` (writer) · `src/main/ops/value_sidecar_read.py`
(reader) · `src/agents/training/value_sidecar_benchmark.py` (cost) ·
`src/agents/training/value_sidecar_test.py` (both).

---

## 1. The hole it closes

**Every instrument this project owns that reads the critic reads EVAL battles** — `main.critic_gate`,
`main.ops.critic_read`, the prober's `calibration`, the scaffolding gauge, the whole win-prob
ladder. Nothing had ever logged the value estimate against **its own training target**. Three
reasons that is a real hole rather than a cosmetic one:

1. **The eval slice is not the training distribution.** Eval plays a *greedy* trainee against a
   *fixed* roster under a quota that **prefers losses** (`trace_selection`, rule 17). Training plays
   a *stochastic* policy against a *self-play curriculum* whose mix moves with `self_play_fraction`.
   A critic can be well calibrated on one and badly calibrated on the other.
2. **The eval read is 8 cycles wide.** A 10M arm evaluates 5 times. The training side produces a
   labelled state every step of every episode, so the same question can be asked per rollout.
3. **The training target is what the loss actually minimises.** `--critic winprob` fits
   `V = sigmoid(win logit)` by BCE against `win_target`. Calibration against THAT is the
   objective's own residual; calibration against an eval battle is a *generalisation* question.

🚨 **`critic_read` and `value_sidecar_read` answer different questions and neither supersedes the
other.** A disagreement between them is a finding about generalisation, not a defect in either
instrument — report it as one.

⚠️ **The sidecar cannot be reconstructed after the fact.** It reads the rollout buffer, which is
gone the moment `train()` returns. A run launched without it has no training-side read available at
any later date.

---

## 2. The flags

| Flag | Default | Meaning |
|---|---|---|
| `--value-sidecar {auto,on,off}` | `auto` | `auto` = **ON under `--critic winprob`**, off otherwise. `on` forces it under any critic (the file then records the mode, and the reader refuses to Brier-decompose a shaped return without `--allow-shaped`). |
| `--value-sidecar-fraction` | `1/64` (0.015625) | Share of buffer states sampled per rollout. |
| `--value-sidecar-seed` | `0` | Sampler seed. The sample is a function of **(seed, rollout index)**, never of a running stream. |

**`auto` is off under `--critic shaped` because `v` is not a probability there** — it is a
PopArt-normalised shaped return whose scale moves over the run, so a Brier decomposition of it is a
category error rather than a loose reading.

**None of these reaches `model_config.json`.** That file is an explicit whitelist of
weight-shape/architecture keys, not an argv dump, so the sidecar needs no `MODEL_CONFIG_VERSION`
bump and cannot make a checkpoint incompatible.

---

## 3. The file

`<run>/value_sidecar/rows.jsonl` — append-only, one JSON object per line, never rotated. Line 1 is
a `{"kind": "header", …}` row (schema, tag, critic mode, `v_is_probability`, fraction, seed,
`max_turns`) so the file stays plain JSONL a consumer can `for line in f` with no special case.

| Column | Provenance |
|---|---|
| `step` · `rollout` · `env` · `t` | READ — `num_timesteps`, the rollout counter, the buffer's own axes |
| `episode` | DERIVED — `"<step>:<env>:<index within rollout>"`, unique within the run |
| `turn` | DERIVED — inverted from the observation's **linear** deadline-clock channel |
| `v` | READ — `rollout_buffer.values`, the value PPO actually used |
| `win_logit` | DERIVED — the exact inverse link of `v` under winprob, `null` under shaped |
| `target` / `target_known` | READ — the back-filled `win_target` / `win_mask` obs keys |
| `opp_class` | READ — the `opp_class` obs key (bot / pool / stable / exploiter) |
| `win_margin` | READ — the `win_margin` obs key |
| `ep_len` / `ep_complete` | DERIVED — folded from `episode_starts` WITHIN this rollout |
| `timeout` | DERIVED — a completed episode whose clock reached `MAX_TURNS` |

### The hazards a reader must carry

🚨 **CALLBACK ORDER IS LOAD-BEARING AND SILENT IF WRONG.** The sidecar must run **after**
`WinProbLabelCallback`, whose `_on_rollout_end` overwrites the `win_target` / `win_mask`
placeholders with the Monte-Carlo label. Registered earlier it reads placeholder ZEROS and writes a
file full of `target: 0.0` — which looks exactly like a critic scoring an unbroken run of losses.
`main.train.callbacks` appends them in that order and the test pins it; at runtime an all-zero mask
over a whole rollout is **reported** (`labels_unfilled`), never written as data.

⚠️ **`target` IS the episode's final outcome**, back-filled to every state of the episode that
produced it. There is no separate outcome column because it would be the same number.
`target_known` marks exactly the episodes that FINISHED inside the buffer.

⚠️ **`ep_len` is the length WITHIN this rollout.** An episode straddling a rollout boundary has its
head in the previous buffer. `ep_complete` is false for exactly those, and every length statistic
must filter on it. Carrying per-env state across rollouts was rejected: a restart would silently
reset it and produce a short-episode spike indistinguishable from a stall regression.

⚠️ **A 250-turn timeout is not a draw, and `win_draw` does not flag one.** The cap forfeits,
Showdown answers `|win|<opponent>`, and it arrives as a plain LOSS. `timeout` is therefore inferred
from the deadline clock reaching `MAX_TURNS` on a completed episode.

⚠️ **The clock SATURATES at the deadline**, and **a battle turn is not a decision index** — one turn
can carry several decisions (a forced switch after a KO), so consecutive rows can share a turn.
Bucketing by it is a statement about game phase, not about decision count.

### What the opponent column is, and is not

`opp_class` is one of four codes. The finer identities are deliberately absent:

* the **bot's archetype name** and the **pool snapshot's step** are chosen per EPISODE inside
  `MaskableAgentWrapper._select_episode_opponent` and never reach the observation;
* an opponent **ladder rating does not exist at training time at all** — bots are unrated by
  construction and a snapshot's Elo is a POST-HOC quantity `main.elo` derives from the finished
  run's `snapshot_ladder/ladder.json`. A rating column would be null on every row of every run,
  which is a worse artifact than its absence. Look a rating up by snapshot afterwards.

🚨 **`opp_class` rides a gate this subsystem WIDENED.** It used to be declared only under the
opponent-intent labels, and a win-prob arm normally runs with no intent loss — so the by-class slice
was empty on exactly the runs the sidecar exists for. `gen3_env` now declares it under the win-prob
label gate too (`designs/ARCHITECTURE.md` §7). It stays a LABEL key the network never reads, and
`train()`'s one-ahead intent SHIFT is still gated on `opp_intent_coef > 0` **and** runs after every
`_on_rollout_end` — so what the sidecar reads is the env's own per-episode value, unshifted, in both
regimes. A row with `opp_class: null` means a PRE-widening checkpoint was resumed under its own
saved observation space; such rows are **absent** from the by-class table, never pooled into `bot`
(0 is a real class, so a default would invent a curriculum).

---

## 4. The cost

Once per rollout at `_on_rollout_end`: a numpy slice over arrays the buffer already holds, a
`json.dumps` per sampled row, and **one** append. `_on_step` is empty — there is no per-step
capture to do, which is what keeps the cost a per-rollout slice rather than a branch on 131,072
steps.

**Measured 2026-09-08** at production shape (`--n-steps 2048 --n-envs 64`, fraction 1/64 → 2,048
rows of 131,072 states), on a box carrying a live training arm:

| | |
|---|---|
| median | **19.3 ms** per rollout |
| worst of 20 | 32.8 ms |
| bytes | 0.57 MB per rollout |
| overhead vs a **120 s** rollout | **0.016%** median, 0.027% worst — **within** the 1% budget |

Reproduce: `python3 src/agents/training/value_sidecar_benchmark.py [--rollout-seconds S]`.

🚨 **A SMOKE A/B CANNOT MEASURE THIS.** Two `--debug --steps 10000` runs differing only in
`--value-sidecar` came out at **2:39.10 (ON)** and **2:45.60 (OFF)** — the arm *with* the sidecar
was 6.5 s faster. That is not a negative cost; it is the run-to-run spread of a
battle-simulation-bound smoke, one to two orders of magnitude wider than the quantity being
measured. The honest decomposition is to measure the numerator directly and divide by a rollout
time quoted with its own provenance. The benchmark's default denominator is deliberately the most
**hostile** plausible rollout (120 s ≈ 1,090 steps/s), because a fast rollout makes the overhead
look worse and a pass there passes everywhere slower.

Per the root `CLAUDE.md`, the benchmark **warns on contention and does not rescale** — a benchmark's
output IS the measurement.

---

## 5. The read

```bash
python -m main.ops.value_sidecar_read <run> [--out DIR] [--bootstrap-draws N] [--allow-shaped]
```

Mean V vs mean target, the Murphy decomposition (reliability / resolution / uncertainty), the Brier
score and skill — pooled, then sliced four ways: **by turn bucket**, **by opponent class**, **by
outcome**, **by training step (1M buckets)**. Writes `value_sidecar_read.{json,md}` under `--out`.

🚨 **`critic_resolution` is the meter; `reliability` alone is not.** A base-rate forecaster scores a
perfect 0 reliability and a useless 0 resolution, so a change that improves reliability while
resolution stays flat has moved the number that was never the disease. The report prints resolution
first and says so.

🚨 **The bootstrap clusters by EPISODE, never by row.** Decisions inside one battle share a board, a
team matchup and a dice stream; sampling is uniform over buffer *cells*, so a long episode
contributes proportionally more rows and a row-level interval would understate its width by exactly
that correlation. `agents.training.stats.cluster_bootstrap_ci` is imported, never re-implemented.

🚨 **Slicing by OUTCOME is not a calibration check, and the report says so.** Conditioning on the
outcome makes the target constant within each slice by construction, so `mean error` there is a
resolution component, not a bias. Read the **asymmetry** between the two rows — an optimistic critic
reads high in both.

**Sign convention:** positive `mean_error` = the critic is **OPTIMISTIC** on that slice. A CI that
straddles 0 reads `UNRESOLVED`, never "no effect".

### Refusals over silence

| Condition | Why it refuses |
|---|---|
| no sidecar at all | a run trained without one; it cannot be reconstructed |
| no header row | the critic mode — and so whether `v` is a probability — is unknown |
| a `shaped` sidecar without `--allow-shaped` | `v` is a moving-scale shaped return; a Brier decomposition of it is a category error |
| nothing labelled | a callback-ORDER defect, or every episode still in progress |
| a slice under `MIN_CELL_N` | marked `UNDER THE CELL FLOOR`, never averaged into a confident number |

A torn final line (a killed run) is **skipped**, never guessed.
