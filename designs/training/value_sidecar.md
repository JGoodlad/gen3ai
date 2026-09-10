# The training-side VALUE SIDECAR (`gen3_value_sidecar_v1`)

> Owned by `src/agents/training/CLAUDE.md` → *The training-side value sidecar*. Always current;
> update it in the same pass as `value_sidecar.py`, `value_sidecar_read.py` or either gate.

Code: `src/agents/training/value_sidecar.py` (writer) · `src/main/ops/value_sidecar_read.py`
(reader) · `src/agents/training/win_prob_callback.py` (the pre-λ outcome stash the writer reads) ·
`src/agents/training/value_sidecar_benchmark.py` (cost) ·
`src/agents/training/value_sidecar_test.py` (writer + the pooled read) ·
`src/main/ops/value_sidecar_read_test.py` (the schema guard).

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

`<run>/value_sidecar/rows.jsonl` — append-only, one JSON object per line, never rotated. A
`{"kind": "header", …}` row (schema, tag, critic mode, `v_is_probability`, `win_prob_lambda`,
`win_prob_lambda_truncated`, fraction, seed, `max_turns`, `resumed`) opens each WRITER SESSION, so
the file stays plain JSONL a consumer can `for line in f` with no special case.

🚨 **ONE HEADER PER PROCESS, NOT PER FILE.** It used to be skipped whenever the file was already
non-empty, so a RESUME appended its rows under the FIRST process's header. A run resumed across a
flag change then held one header saying `win_prob_lambda: 1.0` above a tail of rows whose `target`
is a λ-return, and **nothing on disk said so** — every consumer would pool a 0/1 outcome with a soft
return and report the average as a calibration. A header per session makes the change visible at the
exact row it happens; `resumed` is `false` on the first and `true` on every later one, and
`read_sidecar_segments` is what turns that into a refusal by ROW INDEX.

| Column | Provenance |
|---|---|
| `step` · `rollout` · `env` · `t` | READ — `num_timesteps`, the rollout counter, the buffer's own axes |
| `episode` | DERIVED — `"<step>:<env>:<index within rollout>"`, unique within the run |
| `turn` | DERIVED — inverted from the observation's **linear** deadline-clock channel |
| `v` | READ — `rollout_buffer.values`, the value PPO actually used |
| `win_logit` | DERIVED — the exact inverse link of `v` under winprob, `null` under shaped |
| `target` / `target_known` | READ — the back-filled `win_target` / `win_mask` obs keys. 🚨 **HEADER-DEPENDENT** — see *What `target` IS* below |
| `outcome` / `outcome_known` | READ — the PRE-λ terminal bit and its mask. Identical to `target` at λ = 1.0; the only route to the outcome below it; `null` when λ < 1 and the writer's stash is absent |
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

### 🚨 What `target` IS — one NAME, two MEANINGS, and only the header says which

| header `win_prob_lambda` | `target` holds | `target_known` marks |
|---|---|---|
| **1.0** (every schema-1 file; every unflagged run) | the episode's **terminal 0/1 OUTCOME**, back-filled to every state of the episode that produced it — constant within an episode | exactly the episodes that FINISHED inside the buffer |
| **< 1.0** (`gen3_winprob_lambda_v1`, schema 2 — arm 8) | the **λ-RETURN**: a per-state SOFT probability that VARIES within an episode, blending the outcome with the collector's own recorded `V(s)` | rows the λ target covers — which under the default `bootstrap` truncation **includes trailing rows whose episode never finished** |

🚨 **THIS IS THE DEFECT THE SCHEMA GUARD EXISTS FOR.** Three arms have each landed on exactly
155,137 rows, which invites treating the files as interchangeable. They are not: a reader that
pools schema-1 and schema-2 rows on `target`, or compares two runs' calibration tables across the
boundary without saying so, has averaged two different quantities under one column name. **Matching
row counts are not evidence of a matching quantity.**

🚨 **SO THE OUTCOME IS ITS OWN COLUMN, and it has to be.** `WinProbLabelCallback._apply_lambda`
overwrites `win_target` **in place**, and afterwards the terminal bit exists nowhere else in the
buffer. It is **not** recoverable from the rows:

* the λ-return carries the outcome at weight **λ^d** for a distance `d` to the terminal that no
  column records;
* **`win_margin` is NOT an outcome.** It is the per-turn normalised MATERIAL margin ∈ [−1,1], a
  by-product of `Φ_mat` stashed by the reward manager — its sign is a material lead at that turn.
  Reading it as a win is the circular move this project has a standing rule against;
* `target_known` under `bootstrap` marks rows whose episode never finished at all.

So that callback **publishes the pre-overwrite `(y, mask)` on the model** immediately before
destroying it (cleared at `_on_rollout_start`, so a stale array cannot label the next rollout's
states), and the sidecar — which runs immediately after — writes it as `outcome` / `outcome_known`.
At λ = 1.0 no recursion runs and the columns are the target itself. When λ < 1 and the stash is
absent or the wrong shape, they are `null`: **never inferred.**

🚨 **`ep_complete` FOLLOWS THE TERMINAL MASK, NOT `target_known`.** Under `bootstrap` the λ
recursion UNMASKS the trailing in-progress episode, so `win_mask` stops meaning "this episode
finished inside the buffer". Derived from it, `ep_complete` would call a straddling episode
complete and every length statistic that filters on it would quietly include a truncated head.

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
| bytes | 0.57 MB per rollout — **0.64 MB since `outcome` / `outcome_known` landed** (2026-09-09; a deterministic count, not a timing, so it is comparable across boxes) |
| overhead vs a **120 s** rollout | **0.016%** median, 0.027% worst — **within** the 1% budget |

Reproduce: `python3 src/agents/training/value_sidecar_benchmark.py [--rollout-seconds S]`.

⚠️ **The 19.3 ms median has NOT been re-measured since the outcome columns landed** — the re-run on
2026-09-09 read 41.8 ms on a box at load average 27.9/16 cpus, which the benchmark itself flags as
CPU-STARVED and refuses to rescale. That number is a fact about a busy box and is not an A/B against
the 19.3 ms one; the honest statement is that the byte cost rose ~12% and the timing is unmeasured
on a quiet box. It still verdicted WITHIN the 1% budget at 0.035% of a hostile 120 s rollout.

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
python -m main.ops.value_sidecar_read <run> [--compare RUN2] [--out DIR] [--bootstrap-draws N] [--allow-shaped]
```

Mean V vs mean target, the Murphy decomposition (reliability / resolution / uncertainty), the Brier
score and skill — pooled, then sliced four ways: **by turn bucket**, **by opponent class**, **by
outcome**, **by training step (1M buckets)**. Writes `value_sidecar_read.{json,md}` under `--out`.

🚨 **THE HEADER IS READ FIRST, before a single statistic is computed**, and the report opens with
it: `schema`, `critic_mode`, `win_prob_lambda`, `win_prob_lambda_truncated`, the writer-segment
count, and one sentence saying **in words** what `target` IS for the file in hand.

**Then every table carries the quantity it scored, in its own heading.** On a λ file there are TWO
labelled readings, and the split is the point:

| reading | scores `v` against | answers |
|---|---|---|
| *Calibration against the λ-RETURN target (NOT the outcome)* | `target` | is the critic calibrated to **its objective**? |
| *Calibration against the OUTCOME* | `outcome` | is the critic calibrated to **winning**? — the only reading that compares across the λ boundary |

⚠️ **The two readings' `n` columns are NOT meant to match.** A `bootstrap`-unmasked row has a λ
target and no outcome, so it is scored in the first and absent from the second. Each table uses its
OWN known-flag; neither borrows the other's.

⚠️ **The λ reading OMITS the `won` / `lost` split.** A soft target has no such rows — splitting on
`target == 1.0` under λ < 1 selects only the rows whose return happened to land on an endpoint, i.e.
a reading of proximity to a terminal dressed as a reading of the win/loss asymmetry. The split lives
in the outcome reading, where it is well defined.

**When the outcome is UNRECOVERABLE** (λ < 1 and no `outcome` column — any λ file written before
that column landed) the report says so **in place of the table**, naming all three reasons (the
λ^d weight, `win_margin` being a material margin, `bootstrap` coverage) rather than leaving a
silence for a reader to fill in.

### `--compare RUN2`, and what it refuses

`--compare` reads a second run beside the first and prints the two pooled cells plus **the delta
with its OWN episode-clustered CI** (`cluster_bootstrap_diff_ci`) — two overlapping per-side
intervals would say nothing about whether the difference is resolved, and an equivalence claim needs
this interval inside a pre-stated bar, never a bar against a point estimate.

🚨 **It REFUSES two runs that do not mean the same thing by `target`**, naming both headers, both
target descriptions and the fix. The comparison is on the **QUANTITY**, not the version number:
`critic_mode`, `win_prob_lambda` and `win_prob_lambda_truncated` must agree, and the two schemas
must both be in the declared `SCHEMA_EQUIVALENCE` set. **A schema-1 file and a schema-2 file at
λ = 1.0 compare EQUAL on purpose** — `SIDECAR_SCHEMA`'s own rule is that they are byte-identical
apart from two header fields, so refusing them would be a false alarm on every arm before arm 8. A
schema nobody has declared equivalent refuses even at an identical λ; that is the direction this
subsystem errs in everywhere else.

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
| **`target` changes MEANING mid-file** | a resume across the λ boundary: outcomes in the head, λ-returns in the tail. Refused **by ROW INDEX**, naming every segment's header and what its `target` is. The fix is to split the file at the header rows and read each segment on its own |
| **rows before any header** | the critic mode and λ regime that produced them are unknown; attributing them to the header that FOLLOWS would be a guess |
| **`--compare` across a quantity boundary** | the two `target` columns hold different quantities; a delta between them is mostly the difference between the two DEFINITIONS |

A torn final line (a killed run) is **skipped**, never guessed. A plain restart at the same λ is
**not** a refusal — the pooled read succeeds and a note states how many writer sessions the file
holds.
