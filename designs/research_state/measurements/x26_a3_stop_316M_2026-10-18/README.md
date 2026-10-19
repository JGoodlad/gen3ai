# X26's Amendment-3 stop at 316.8M (2026-10-18): NUMERICAL, and the mechanism is named

## Verdict

**NUMERICAL. No fault was found, and no checkpoint is void.** Both events are fp32 rounding on a row where one
computation is badly conditioned. The computation is X5's fixed-size species shift τ (`hypothesis_set.fixed_size_tau`,
the species call). Each violator's state has exactly one unseen opponent species left to place (k = 1), and the
model is almost certain which species it is. The constraint Σ σ(a + τ) = 1 is then almost flat at its root:

- its slope is 8e-6 to 1.7e-5, against a median of about 0.6 for the dump's other rows;
- an fp32 sum near 1 is only known to within ulp(1), so fp32 pins τ only to ulp(1) / slope, which is **7e-3 to
  1.5e-2**, against 2e-7 for the other rows.

τ shifts the log-presence of every tail hypothesis. That shift reaches log π through the attention bias.

Two equally valid fp32 programs sum the constraint in different orders, so they land at different points inside
that band. T2 runs an inductor-compiled CUDA program; the learner runs eager. Reordering the sum alone moves the
violator's τ by 4.4e-3 to 1.4e-2. Multiplied by the row's d log π / d τ, that predicts a gap of 2.5e-4 to 1.9e-3,
and the violator ranks **1st of 33** in all six weight sets tried. The observed gaps are 2.97e-3 and 8.1e-4.

**The decision is drawn from the row dumps plus the nearest checkpoints, because the exact weights are gone** (F1).
§5 states what that can and cannot establish.

**Recommendation (the orchestrator decides; §6):**
- RESUME X26 from its newest state, `final_model_interrupted.zip` (320,139,480; at the pin, the same argv, `warn`).
  If the startup gate refuses it, follow the §4.2 procedure and use `checkpoint_320000071`. The void range is EMPTY,
  as for u13900.
- Before the resume, register the judged-max handling, because this mechanism WILL cross 1e-3 again (§6). The
  proposal: a STOP at 1e-2 and a RE-DIAGNOSIS (not a stop) at 1e-3, using this directory's scripts. p99 and
  non-finite stay unchanged.
- The conservative alternative, if the orchestrator does not accept a classification made from neighbour weights:
  resume from `checkpoint_316000231`, voiding (316,000,231, 320,139,480], which is 4.14M steps.

| | |
|---|---|
| run | `rb_x26_s2601`, pin `9c033489`, `--behaviour-check warn` (Amendment 3). Stopped by the launcher's PID at 320,139,480 after the 320M read |
| events | **u32180** @ 316,796,508: judged max **2.974e-3** (bar 1e-4, A3 ceiling 1e-3), p99 2.30e-5, 974 / 1,024 judged, policy_version 646, T2 trainee slot (rust collector, backend graph). Its seeded 4,096-row scan read max 8.38e-5, over_bar 0. **u32320** @ 318,171,536: 8.12e-4 (A4 W2 ≥ 5e-4), p99 2.52e-5, version 660 |
| inputs (read only, copied out first) | `~/.cache/gen3ai/x26_a3_stop/` (SHA256SUMS verified): the jsonl (70 records), the two `_rows.npz`, checkpoints 316000231 / 317000041 / 320000071, `final_model_interrupted.zip`. CONTROL inputs from the run (copied): the four weight-carrying dumps u13900 / u19300 / u20490 / u20630 (from the refresh's copies) and their bracketing checkpoints 136000153, 137000325, 190000228, 191000457, 201000180, 202000230, 203000130, 204000126. The tb events were also copied. Checksums: `inputs_SHA256SUMS` |
| code | X26's PINNED tree (`~/.cache/gen3ai/x26/pin_9c033489/src`, read only, `PYTHONDONTWRITEBYTECODE=1`), CPU only, `CUDA_VISIBLE_DEVICES=""`, under `nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 12`, one job at a time (peak 1.35 GB) |

## 1. What the row dumps show (the record level; no weights needed)

| check | u32180 (row 140) | u32320 (row 143) |
|---|---|---|
| dump vs JSON record (stored log μ, action, mask, buffer t/e) | identical; the stored-vs-learner gap reproduces to the bit | identical |
| non-finite anywhere | none (TB: 0 non-finite in 3,260 probes) | none |
| legal set, stored vs recomputed | equal (`00000011110`: the 4 moves, no switch) | equal (`00001111110`) |
| argmax, stored vs recomputed | 8 = 8 (margin 0.0815 → 0.0756) | 9 = 9 |
| policy version vs current | 646 = current (rows_current 1,024 / 1,024) | 660 = current |
| shift over the legal actions (log μ − log π) | +3.27e-3, −2.11e-3, +2.97e-3, −2.92e-3: the whole row, coherent | max 8.1e-4, spread over all 6 |
| the sampling draw (u vs the cumulative p) | draw margin 0.0188, far above the 1.3e-3 shift in p | 0.0168 |
| nearest declared selection site (learner margin, relative) | 0.0039 at `hypothesis_set.py:214 argsort`, 20× the 2e-4 exclusion, not excluded | 0.0258, same site |
| other rows of the same game | none recorded above 4.5e-6 | the NEXT two decisions are elevated too: 9.25e-5 (t 1, the same hypothesis state, the identical tie margin 0.02577…), then 1.24e-5 (t 2) |

**Reading.**
- A wrong row, a wrong action, a mask fault or a stale version moves a row by ≥ 1e-2, and usually by about a nat.
  Along the weight path, one update moves THIS row by ≈ 0.12 nats. Both events instead shift the whole distribution
  by 1e-3.
- In u32320 the elevation decays over three consecutive decisions of one game. Those are three separate T2 calls,
  so this is a property of the STATE, not a one-call transient.
- The record also shows that **1e-3-scale whole-row shifts are not new**: over the 70 records' top-200 rows,
  non-taken actions have already moved by 1.38e-3 (u31430, 309.4M) and 9.4e-4 (u30880, 304.0M). Only the TAKEN
  action is judged, and it happened to be the big mover this time.

**Code reading** (a read-only sub-agent on the pinned tree):
- **Weight publication** is an in-place copy on the main stream. Every T2 lane does `wait_stream`, and the copy is
  bit-verified by `verify_copy` plus a re-gate (`slots.py:227-257`, `service.py:260-272`, `engine.py:276-287`).
- **The version tag** is assigned when the result is read, and changes only after `load` returns
  (`collector.py:235, 447-448`).
- **The stored obs** is the same array that was submitted (`store.py:447`, `service.py:413`).
- **T2's replica** is forced to fp32 (`slots.py:64-69`). Pad rows are discarded.
- **Torn weights, a stale tag, an obs carry-over at t = 0 and bf16 on T2 are RULED OUT by the code.**
- **buffer_t = 0 is a selection artefact.** `choose_rows` takes the first current rows in t-major order, so about
  24 % of judged rows sit at t = 0 (both violators there: chance ≈ 6 %).
- The run log holds no undeclared-op, flip-divergence or flip-forward error in the window.

## 2. The neighbour replay and its calibration control

**Method.** `replay_neighbours.py` runs d29f781c's `replay_dumps.one_dump`, unchanged, on these weight sets:
- each event's bracketing checkpoints;
- their step-linear interpolation at the event (α 0.796 for u32180, 0.390 for u32320).

Each run compares a float64 reference, the fp32 learner and decision paths, and a 32-draw one-ulp dither. Each
dump holds 33 rows: the violator plus the scan's 32 worst. `control_neighbours.py` runs the same thing on the four
NUMERICAL dumps that DO carry weights, on their true weights and on their own bracketing checkpoints.

**Weight distance.** Neighbour weights move the rows by 0.2–0.6 nats (u32180: 0.27 at the interpolation), which is
100× the gaps. So the replay cannot reproduce a gap. It can only read the row's CONDITIONING near the true weights.

| | violator: fp32 error vs float64, CPU (rank / 33; × others' median) | violator: one-ulp dither std (rank / 33) |
|---|---|---|
| u32180 · ckpt 316000231 | 2.66e-4 (**1**; 44×) | 9.8e-6 (10) |
| u32180 · ckpt 317000041 | 1.22e-3 (**1**; 234×) | 4.2e-6 (24) |
| u32180 · interp 0.796 | 1.89e-3 (**1**; 274×) | 8.1e-6 (13) |
| u32320 · ckpt 317000041 | 1.13e-4 (**1**; 16×) | 6.4e-6 (17) |
| u32320 · ckpt 320000071 | 8.7e-5 (**1**; 15×) | 6.8e-6 (15) |
| u32320 · interp 0.390 | 4.9e-5 (**1**; 7×) | 3.8e-6 (28) |
| control (u13900 / u19300 / u20490 / u20630), true weights | ranks 2 / 1 / 2 / 1 | ranks 1 / 1 / 1 / 1 |
| control, neighbour weights (3 each) | ranks 1–6 for three of the four; u13900 9–33 | ranks 1–7 for three of the four; u13900 17–28 |

**Reading.**
- **On fp32 error, both violators rank FIRST in all six weight sets, by 7–274× the other rows.** The neighbour
  replay keeps that property, and keeps it more strongly than for the calibrated controls.
- Every fp32 CPU path agrees with the others to 2e-6 – 1.5e-5. They all miss float64 by the same amount: the error
  is SYSTEMATIC.
- **The dither criterion of d29f781c does NOT flag these rows.** It perturbs weights, but the error here is born in
  ACTIVATION arithmetic (§3), which a weight dither barely moves. So the class test is extended, not failed:
  - the dither rank still identifies the earlier mechanisms;
  - the fp32-error rank and the root band (§3) identify this one.
- **A flip of the violator's NEAREST declared selection element does not reproduce the gap** (`flip_test.py`, the
  probe's own `TieFlip`):
  - u32180's nearest pair moves the row by only 3–5e-6;
  - u32320's moves it by 7e-3 – 1.3e-2, in a different direction (cos −0.87, −0.71, −0.02).

  The controls behave as recorded: u20630's flip is 0.0176, matching the refresh's 0.0146.

## 3. The mechanism: X5's species shift τ is ill-conditioned on a saturated k = 1 state

**Where the error is born.** It is traced forward in three steps:
- `locate.py`: the edge-bias family maps are clean, with input and output errors around 1e-7, as on the other rows.
- `locate_bias.py`: the error is ALREADY in the attention bias entering `edge_bias`. It sits in ONE key column
  (key 31, a hypothesis key) at a log-presence near −12. The fp32 error there is 2.0–5.6e-3; on the other rows it is
  about 1e-6.
- `locate_tau.py`: that column is fed by **call 0 of `fixed_size_tau`** (the species presence over 400 candidates).
  The violator's |τ| is 19.2–20.0, against an other-row median of about 7. Its fp32 τ error is 2.0e-3 to 5.6e-3,
  against an other-row median of about 2e-6. The other four calls (moves) are clean.

**Why τ is ill-conditioned there.** The violator has k = 1, and one candidate holds π ≈ 1 − 1e-5. So the root's
slope S'(τ) = Σ π(1 − π) is tiny. An fp32 sum near k = 1 resolves only ulp(1), so τ is fixed only to about
ulp(1) / S'(τ):

| | k | slope S'(τ) at the root | band ulp(k) / S'(τ) (rank / 33) | measured fp32 τ error |
|---|---|---|---|---|
| u32180 (three weight sets) | 1 | 8.1e-6 – 1.04e-5 | 1.1e-2 – 1.5e-2 (rank 2) | 4.2e-3 – 5.6e-3 |
| u32320 (three weight sets) | 1 | 1.6e-5 – 1.7e-5 | 7.1e-3 – 7.5e-3 (rank 1) | 2.0e-3 – 4.2e-3 |
| the other rows, median | — | ≈ 0.6 | ≈ 2e-7 | ≈ 2e-6 |

The saturation is a stable property of the STATE: the slope barely moves across the three weight sets. That is
why the neighbour replay can see it without the exact weights.

**Why the violator, among the saturated rows.** Each dump holds other k = 1 rows. One of them has a WIDER band
(2.7e-2) than the violator, but its log π barely reads τ: d log π / d τ ≈ 0.001–0.003, and its gap is 8.4e-5. An
event needs BOTH a wide band and a log π that reads τ. The violator is the only row with both: d log π / d τ =
0.064–0.335 for u32180 and 0.023–0.041 for u32320.

**Two fp32 programs, emulated** (`tau_programs.py`). The violator's scores, cast to fp32, go through the pinned
`fixed_size_tau` under 64 random orders of the candidate axis. That is the same arithmetic with a different
summation order, which is the difference between T2's fused CUDA reduction and the learner's eager one.

| | τ spread over orders (others' median) | d log π(taken) / d τ | predicted gap scale (rank / 33; others' max) | observed / predicted | direction: cos(d log π / d τ, observed Δ) over legal actions |
|---|---|---|---|---|---|
| u32180 · ckpt 316000231 | 1.36e-2 (0) | 0.064 | 8.7e-4 (**1**; 1.6e-4) | 3.4 | −0.74 |
| u32180 · ckpt 317000041 | 4.4e-3 (5e-7) | 0.235 | 1.0e-3 (**1**; 8.7e-5) | 2.9 | +0.62 |
| u32180 · interp 0.796 | 5.6e-3 (0) | 0.335 | 1.9e-3 (**1**; 2.9e-6) | 1.6 | **+0.94** (implied τ difference 1.25e-2 = the band 1.23e-2) |
| u32320 · ckpt 317000041 | 8.5e-3 (1e-6) | 0.041 | 3.5e-4 (**1**; 4.1e-6) | 2.3 | −0.87 |
| u32320 · ckpt 320000071 | 1.06e-2 (1e-6) | 0.023 | 2.5e-4 (**1**; 8.8e-6) | 3.3 | −0.89 |
| u32320 · interp 0.390 | 1.41e-2 (1e-6) | 0.023 | 3.2e-4 (**1**; 2.9e-6) | 2.5 | −0.70 |

**Reading the table.**
- Reordering the sum, and nothing else, moves the violator's τ by up to its band. On the other rows it moves τ by
  nothing.
- The predicted gap ranks the violator first in every weight set. The observed gaps sit 1.6–3.4× above the
  64-order spread, and inside the band × sensitivity bound at the event-step weights (u32180: 1.2e-2 × 0.335 =
  4.1e-3 ≥ 2.97e-3).
- The observed mismatch points along the τ direction: |cos| is 0.62–0.94, with 3 and 5 free dimensions. The SIGN
  depends on which program lands higher in the band, so it carries no information.

## 4. The record against the fitted rounding tail (`surprise.py`, `forecast.py`)

**`surprise.py`** refits the d29f781c model (trend + empirical body + generalised-Pareto tail) on the probes from
100M to the last probe BEFORE u32180:
- u32180's judged max is **7.5×** the largest judged max of the run before it (3.95e-4, u28980). Its residual sits
  ≈ 1.9 log units above the largest earlier residual.
- It lies AT or BEYOND the fitted tail's upper endpoint (ξ ≈ −0.09 to −0.10), so P ≤ 3e-18 per probe under every
  form.
- u32320 is 2.1× the previous max, at P ≈ 2–5e-6 per probe.

**So the gate statistic's earlier tail did not predict these events.** The reason is the mechanism, not a fault.
k = 1 saturation is a NEW, stronger amplifier: up to ≈ 1e-2 in τ, against the weight-and-logit conditioning
(≤ 5e-5) that produced the earlier tail. The taken-action statistic had only seen it on rows where log π barely
reads τ. The full-distribution record had already shown it (§1).

**`forecast.py`** was refreshed at 320.16M (3,260 probes; `forecast.json`; the fits now INCLUDE the events):
- **The event count (≥ 1e-4) now runs well ahead of 177f61c0's table:** 69 observed after 137M against 26.4
  expected, and 22 in the last 20M against 5.3.
- **The p99 is NOT growing:** its 10M-window medians are 2.14e-5, 2.22e-5 and 2.20e-5 from 290M to 320M, below W1's
  3.0e-5. P(p99 ≥ 1e-4 before 500M) ≤ 0.06 under every form.
- **With the events in, P(judged max ≥ 1e-3 before 500M) is 0.31–0.52** (bootstrap mixture, fit ≥ 100M). The
  first-crossing 5 % point is ≈ 345M.
- The many-row statistic is calm while the single-row tail fattens. That is the signature of a rare-state
  mechanism, not of drift across the board.

## 5. What this replay can and cannot establish

**CAN establish:**
- the violators' states are k = 1-saturated, with a τ band of 7e-3 to 1.5e-2 that is stable across the neighbours;
- their fp32 error is born in τ and ranks first of 33;
- a change of summation order alone reproduces τ differences of the band's size;
- the predicted gap ranks the violator first, within 1.6–3.4× of the observed gap;
- every structural fault class is ruled out, by the record and by the code.

**CANNOT establish:**
- **The exact gap.** The weights that played the rows are gone (F1), and d log π / d τ varies 5× across the
  neighbours (0.064–0.335 for u32180).
- **u32320's implied τ difference fits only within the neighbour spread.** It is 1.3e-2 to 3.2e-2, 2–4× its
  neighbour band of 7e-3. The slope at the true weights would have to be 2–4× flatter; the neighbours sit 0.2–0.6
  nats away.
- **T2's exact program.** The inductor CUDA graph was not run (no GPU lease; 177f61c0 F2). The reordering is an
  emulation of a different program, not that program.
- **What would make it exact:**
  - the next event's own weights, which no event in this lineage will dump (F1);
  - or a GPU replay of T2's compiled decision forward on these rows, at a checkpoint, compared with the eager learner
    on the same weights. That would show that the two GPU programs disagree on τ by the band.

## 6. Recommendation, in the registered rules' terms

**1. Classification: NUMERICAL** under §A2.7(3) and the A3-F2 / A4 re-diagnosis. A4.2 says NUMERICAL ⇒ continue
under the ceilings, and A3.3's precedent is that a NUMERICAL stop voids nothing. Therefore:
- **Resume from the newest state:** `final_model_interrupted.zip` (320,139,480), at the pin `9c033489`, with the run's
  own argv (`warn`), toward the Amendment-5 stop.
- The stop is at **380M per the brief (P = 280M; not re-verified here)**.
- If the startup gate refuses the save, follow §4.2's procedure (as on 2026-10-17) and use `checkpoint_320000071`.
- **The void range is EMPTY.**

**2. Before the resume: register the judged-max handling (an Amendment 6; the orchestrator's call).** Under the
unchanged 1e-3 ceiling, this mechanism will stop X26 again:
- it reaches band × sensitivity ≈ 1.5e-2 × 0.34 ≈ 5e-3;
- the refreshed forecast gives a 0.31–0.52 chance of another crossing before 500M; the last 20M held one
  crossing of 1e-3 and two of 5e-4;
- each stop costs a GPU idle and a re-diagnosis.

The proposal:
- judged max **≥ 1e-2 ⇒ STOP**. That is 2× the mechanism's measured reach, and still ≥ 10× below the smallest
  weight-version fault on such a row (one update ≈ 0.12 nats);
- **1e-3 ≤ judged max < 1e-2 ⇒ the A3-F2 RE-DIAGNOSIS, not a stop.** It runs this directory's `locate_tau.py` and
  `tau_programs.py` on the dump with the nearest checkpoint, about 3 CPU-min.
  - NUMERICAL-by-saturation (k = 1, the violator ranking first on the band × sensitivity, τ the error's origin) ⇒
    continue;
  - anything else ⇒ HOLD per §A2.7(3);
- **p99 ≥ 1e-4 and any non-finite value stay STOPS, unchanged.** The p99 is the many-row fault detector, and it is
  flat.

The alternative is to keep 1e-3 and accept the stops. Each one would be diagnosed in about 30 minutes with these
scripts.

**3. The per-process weight-dump cap: do NOT change it at the pin.** It is not per process (F1). This class can now
be classified WITHOUT weights, from the rows plus the nearest checkpoint, so the cap is no longer the blocker.

**4. Watch.**
- A4 W2 (judged max ≥ 5e-4) will fire on this class. Fold it into the proposed 1e-3 re-diagnosis.
- W1 (p99 windows) and the A3 p99 read are unchanged.

**5. The conservative alternative**, if the orchestrator does not accept a neighbour-weight classification:
resume from `checkpoint_316000231`, the last checkpoint before the first over-ceiling update. That voids
(316,000,231, 320,139,480] (4.14M steps, ≈ 1.3 % of the run). Every probe up to u32170 is below the ceilings (the
320M full-precision read).

**Not recommended: END X26.** No fault was found. The single-row effect on PPO is nil: an importance ratio of
1.003 on 1 of ≈ 98k rows. And the owner's Amendment-5 overrun is the registered stop.

## FINDINGS (standing rule 7)

- **F1 (MAJOR for ops). The weight-dump cap is per LINEAGE, not per process, so A4-F1's "the next restart resets the
  count" is FALSE.**
  - `_behaviour_weight_dumps` is a plain attribute on the model, so SB3 pickles it into every save.
  - It reads **3** in the `data` of both `checkpoint_316000231_steps.zip` and `final_model_interrupted.zip`. Every
    resume restores it.
  - That is why no event since u20630 (203M) has dumped weights, across restarts, the 10-18 07:56 one included.
  - At the pin, no future X26 event will dump weights.
  - Fixes for a FUTURE pin: exclude the attribute from the save, or reset it at load.
- **F2 (MAJOR, design). `selection_sites.REASONS["BISECT"]` is wrong in this regime.** It claims a flip of the
  bisection's direction test "moves the converged root by at most the final bracket width — a continuous, ulp-scale
  change".
  - The bisection is fine. The fp32 EVALUATION of the constraint is the problem: it resolves the root only to
    ulp(k) / S'(τ), up to 1.5e-2 here. The change is continuous, but it is not ulp-scale.
  - The same text is at HEAD (`src/agents/model/selection_sites.py:174`).
  - Fix for a future pin: run `fixed_size_tau`'s constraint in float64 (400 candidates; negligible cost), or solve
    the complementary constraint when the mass saturates.
  - Not done here; out of scope.
- **F3. The growth driver of the K9(b) tail (the refresh's F4, "unidentified") has a candidate.** The species belief
  saturates as the model sharpens: score ranges of about 100 on these rows, against about 74 for the others. More
  states land at k = 1 with a near-certain candidate. **UNVERIFIED** as the driver of the whole tail.
- **F4. The brief's premise needed a correction.** "Every earlier event ≤ 2.6e-4" is wrong: u28980 (285.3M) read
  **3.95e-4**. The conclusion is unchanged: 2.97e-3 is still 7.5× the earlier max.
- **F5.** The earlier class criterion (the dither rank) misses this mechanism. A re-diagnosis must read the
  fp32-error rank and the τ band too (`locate_tau.py`).
- **F6.** The pinned tree `~/.cache/gen3ai/x26/pin_9c033489` shows a MODIFIED `designs/ops/slow_tier_status.json` and
  `__pycache__` dirs. These were pre-existing, were not made by this unit, and were not touched. The pin's `src`
  code was read only.
- **F7.** The "u22430 row 255, 2.64e-3" whole-row shift in the record is a FLIP-JUDGED row (tie margin set to inf).
  Its two distributions are two resolutions of a tie, so it is excluded from §1's continuity point.
- **F8.** No GPU was touched. Nothing under `models/`, `data/`, `~/.cache/gen3ai/x26_plateau/` or
  `x26_a3_stop/` was written (all reads were copied out first). X26 was not relaunched.

## Files

- **Scripts:**
  - `replay_neighbours.py`: the neighbour replay, the weight-path sweep and the tie margins;
  - `control_neighbours.py`: the calibration control;
  - `flip_test.py`: the nearest-element flip;
  - `locate.py`, `locate_bias.py`, `locate_tau.py`: the error's origin and the root conditioning;
  - `tau_programs.py`: the reorder emulation, the sensitivity and the direction;
  - `surprise.py`: the event against the pre-event fitted tail.
- **Reused unchanged:** d29f781c's `replay_dumps.py`, `tb_extract.py` and `forecast.py`.
- **Outputs:** the matching `.json` files. `replay_neighbours.json` is slimmed: the per-module error arrays are
  dropped, and their top-5 summary is kept. `forecast.json` is the 320M refresh.
- **Inputs:** `inputs_SHA256SUMS`.
- **The weight-path sweep** (`replay_neighbours.json` `sweep`, 101 points across each bracket): neither violator
  shows a jump. Several scan rows do cross declared sites, jumping 0.1–1.2 nats. At this resolution a 3e-3 step on
  the violator would be invisible, so the sweep is reported, not used.
