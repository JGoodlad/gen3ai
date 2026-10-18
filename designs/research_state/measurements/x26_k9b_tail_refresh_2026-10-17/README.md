# X26's K9(b) tail, refreshed at 211.9M steps (2026-10-17): still NUMERICAL, growth faster than linear

**Verdict.**
1. **Mechanism: NUMERICAL, for every dump that can be checked.** Four dumps carry their weights (u13900 as the
   control, plus u19300, u20490 and u20630). In all four, the violating row fits the class that 177f61c0 found.
   It is the most or second-most fp32-ill-conditioned of the dump's 33 rows, and its gate gap is ordinary for its
   conditioning. No dump falls outside the class, so there is **no MAJOR FINDING**. Two events cannot be
   replayed: u21290 and u21440 have no weights (F1).
2. **Forecast: the gaps grow FASTER THAN LINEARLY, so the ceilings' margin is smaller than Amendment 3 assumed.**
   The event COUNT still matches the 177f61c0 rate table: 5 events after 137M against 5.4 expected. But the
   SCALE of the gaps fits exponential growth in steps better than the linear growth 177f61c0 assumed. Under that
   form:
   - the p99 ceiling (1e-4) has a **≈ 2-in-3 chance of being crossed before the 500M cap**, most likely near
     490M;
   - the judged-max ceiling (1e-3) has a **≈ 1-in-4 chance**.

   Under linear or power growth, each ceiling's chance is ≤ 4 %. The data cannot yet tell exponential from linear
   growth with confidence.
3. **Recommendation, in the registered rule's terms: a PRE-REGISTERED WATCH TRIGGER.** This is neither a stop nor
   a ceiling change (§4). Nothing is near a ceiling today:
   - the p99 window max is 2.06e-5, 4.9× below its ceiling;
   - the judged max is 1.67e-4, 6× below its ceiling.

   A3-F2's re-diagnosis should be TRIGGERED by the scale before a rounding event forces a stop. It should not wait
   for that stop.

| | |
|---|---|
| run | `rb_x26_s2601` (LIVE; pin `9c033489`; `--behaviour-check warn` under Amendment 3) |
| inputs (read only, COPIED out first) | the run's `behaviour_violations.jsonl` (6 records, to u21440), the 6 `_rows.npz` and 4 `_policy.pt` dumps, `final_model_exception.zip` (an architecture template only; its weights are replaced by each dump's), `model_config.json`, the `tb/` events. Copies at `~/.cache/gen3ai/x26_k9b_refresh/in/`; checksums in `inputs_SHA256SUMS` |
| code | the forward passes use X26's PINNED tree (`~/.cache/gen3ai/x26/pin_9c033489/src`, read only, `PYTHONDONTWRITEBYTECODE=1`), CPU only, `CUDA_VISIBLE_DEVICES=""`, under `nice -n 19 taskset -c 5-7,13-15 scripts/ops/mem_cap.sh 8` |
| scripts | `tb_extract.py` → `tb_series.json` (trimmed to the tags read); `replay_dumps.py` → `replay_dumps.json` (it imports `../x26_k9b_stop_2026-10-17/replay.py`'s forwards unchanged); `forecast.py` → `forecast.json` |

## 1. The mechanism check, per dump (`replay_dumps.json`)

**Method.** This is 177f61c0's replay, unchanged. Its functions are imported:
- **ref64** is the learner forward in float64: a `TorchFunctionMode` upcasts every operand. Two float64 paths agree
  to ≤ 7.5e-14 on every dump.
- **The fp32 CPU paths** are the learner (`evaluate_actions`) at full batch and at batch 1, and T2's
  `DecisionModule` at buckets 8 / 64 / 256.
- **The dither** is 16 one-ulp random roundings of every parameter. Its spread (`pert_std`) measures how far two
  equally valid fp32 evaluations of THAT row sit apart.

Each dump's rows are the probe's violator (`obs__*`, the row the gate judged) plus the 32 worst rows of the scan
(`scan_obs__*`). Under `warn` the scan is a seeded 4,096-row sample, so in the new dumps the violator is NOT among
the scan rows. Its comparison set is therefore 32 rows that are below the bar.

**Integrity.** On every dump the violator's stored log μ, action, mask and buffer (t, e) match its JSON record
exactly. The dump's stored and GPU-learner values reproduce the record's |Δ| to the bit.

**u13900 control.** It reproduces 177f61c0 to the digit:
- stored μ − ref64 is +6.41e-5;
- GPU learner − ref64 is +1.65e-4;
- the CPU fp32 paths miss ref64 by 1.45–1.52e-4.

| dump | step | violator: CPU fp32 − ref64 (others' median \|·\|) | stored μ − ref64 / GPU learner − ref64 | fp32-error rank / 33 | dither std (rank / 33; × others' median) | gate \|Δ\| ÷ dither std (others: median, max) | fits the class |
|---|---|---|---|---|---|---|---|
| u13900 | 136.97M | +1.45…1.52e-4 (1.8e-5) | +6.4e-5 / +1.65e-4 | 2 | 2.8e-5 (1; 2.5×) | 3.6 (3.1, 6.8) | yes (control) |
| u19300 | 190.07M | +6.3…8.0e-5 (7.1e-6) | −7.1e-5 / +5.6e-5 | **1** | 3.9e-5 (**1**; 6.4×) | 3.2 (3.3, 10.3) | **yes** |
| u20490 | 201.77M | +2.4…3.3e-5 (1.0e-5) | **−1.01e-4** / +2.7e-5 | 2 | 3.6e-5 (2; 3.6×) | 3.5 (2.5, 4.9) | **yes**, see note |
| u20630 | 203.15M | +1.07…1.14e-4 (9.7e-6) | −6.0e-5 / +1.08e-4 | **1** | 4.5e-5 (**1**; 4.4×) | 3.7 (2.3, 7.5) | **yes** |
| u21290 | 209.64M | — no weights dumped (F1) | — | — | — | — | UNVERIFIED (record-level checks pass) |
| u21440 | 211.11M | — no weights dumped (F1) | — | — | — | — | UNVERIFIED (record-level checks pass) |

Reading the table:
- **Every replayable violator is the best- or second-best-conditioned candidate for a rounding gap among its
  dump's rows.** It ranks 1st or 2nd of 33 on BOTH the fp32 error and the dither spread, at 2.5–6.4× the other
  rows' median dither.
- **Its gate gap is 3.2–3.7 dither standard deviations, inside the other rows' range** (medians 2.3–3.3, maxima
  4.9–10.3). Every row crosses the bar because of its CONDITIONING, never because its gap is unusual for that
  conditioning. That is the 177f61c0 class.
- **The error grows in smooth layers, not at a cutoff.** The violators' fp32 error concentrates in the same kind of
  layers 177f61c0 named:
  - u19300: `op_worst_proj` 12×, then `switch_score` 9×;
  - u20490: `move_resolution_cell.status_move_proj` 10×;
  - u20630: `win_head` 29×, then `move_score` 19×.

  Each figure is the module's relative fp32 error over the other rows' median.
- **u20630's violator was FLIP-JUDGED.** Its near tie sits at the `hypothesis_set.py:214` argsort, and its other
  resolution would be 0.0146 away. The float64 forward lands 6.0e-5 from stored μ, on the SAME resolution, so the
  gap is rounding inside one branch, not a branch choice.
- **Note on u20490.** Here the far side of the gap is the STORED μ: T2's inductor-compiled CUDA-graph program
  misses ref64 by −1.01e-4, while every CPU fp32 path misses it by only 2.4–3.3e-5. That miss is 2.8 of the row's
  dither std and about one dither RANGE (1.1e-4). It sits at the edge of the row's fp32 envelope, not outside it.
  It is also the dump whose violator is least ill-conditioned (fp32 error 3.7× the others' median).

  T2's exact GPU program cannot be re-run on this box (177f61c0 F2; no GPU lease). So the stored side is
  bounded, not reproduced (F3).
- **177f61c0's "the stored μ is the closest fp32 path" does NOT generalise.** It holds on 2 of the 4 dumps. In
  3 of the 4, stored μ and the learner sit on OPPOSITE sides of ref64. Both orderings are what two independent
  fp32 programs of one ill-conditioned function produce, so this weakens no conclusion. It does retire that
  sentence as a signature of the class.

**Record-level checks on all six violators** (`forecast.json` `records`). These checks are what the two
weight-less events can still be judged on:
- the stored and recomputed legal sets are equal;
- the argmax is equal;
- every legal action's |Δ log p| is ≤ 1.9e-4, which is a rounding-sized shift of the whole distribution (a wrong
  row or wrong action moves it by ≥ 1e-2, 177f61c0 §2);
- the probe's p99 stays at 9.2e-6…1.44e-5, so the event is not a many-row shift.

u21290's tie margin is 0.16 at `damage_kinds.py:130`, 800× the 2e-4 exclusion margin. u21440 is flip-judged at
the same `hypothesis_set.py:214` argsort as u20630. Both are CONSISTENT with the class, but neither can be
replayed.

## 2. The forecast (`forecast.py` → `forecast.json`)

**Data.** The probe runs every 10th update (98.3k steps). The series has 2,159 probes up to 211.9M, read at full
precision from the TensorBoard events. Two adjustments:
- The run resumed after the u13900 FATAL from `checkpoint_136000153` (a startup-gate refusal of the exception save,
  `design_x26_readiness.md` §4.2), so 136.2–137.0M is logged twice. The series keeps the LIVE lineage, which is the
  later file.
- The u13900 event sits on the abandoned branch. It is carried only as a sensitivity.

There are 0 non-finite values.

| 10M window | judged max: median | judged max: max | p99: median | p99: max | events ≥ 1e-4 |
|---|---|---|---|---|---|
| 130–140M | 2.4e-5 | 9.7e-5 | 9.5e-6 | 1.37e-5 | (u13900, abandoned branch) |
| 160–170M | 2.6e-5 | 9.4e-5 | 1.06e-5 | 1.71e-5 | 0 |
| 190–200M | 3.4e-5 | 1.26e-4 | 1.36e-5 | 1.81e-5 | 1 |
| 200–210M | 3.2e-5 | 1.67e-4 | 1.36e-5 | 2.06e-5 | 3 |
| 210–211.9M | 4.1e-5 | 1.19e-4 | 1.39e-5 | 1.79e-5 | 1 |

### 2a. The event rate (the 1e-4 bar): what 5 events can and cannot say

- **Against 177f61c0's rate table** (0.37 % per probe at 137M, rising to 1.0 % at 200M), the table predicts 5.4
  events after 137M and 5 were observed (Poisson P(≥ 5) = 0.62). Over the last 20M it predicts 2.1 events and 4
  were observed (P(≥ 4) = 0.16). **The event count is not running ahead of the 177f61c0 forecast.** The 4-in-20M
  cluster is unremarkable at the forecast's own 200M rate.
- **A binomial model per probe** has three forms: constant, logit-linear in steps, and logit-linear in log steps.
  - A trend is detected: the likelihood ratio against the constant is 22.1, p ≈ 3e-6. With u13900 added it is
    16.6.
  - **The two trend forms are indistinguishable** (Δ log-likelihood 0.03).
  - The 90 % interval on the slope spans an expected 3 to 2,900 events by the cap.

  **Five events cannot forecast the ceilings.** The trajectories below use all 2,159 probes and do.

### 2b. The trajectories against the ceilings

The model is **log x = T(s) + r**, where x is each probe's judged max or p99, s is the step, and T(s) takes three
forms:
- linear (177f61c0's form);
- exponential in steps;
- power in steps.

It is fitted on s ≥ 100M, the regime since the early ramp, with s ≥ 20M as a sensitivity. The residual r is
stationary: its 99th percentile before vs after 137M is 1.01 vs 1.04–1.06 for the max and 0.26–0.28 vs 0.29 for
the p99. Its body is empirical, and its tail above the 95th percentile is a generalised Pareto fit.

Each future probe to the cap gets a crossing probability. Uncertainty comes from a moving-block bootstrap (blocks of
50 probes ≈ 5M steps, B = 1,000), which refits the trend AND the tail. The intervals are the mixture of the
replicates' first-crossing distributions. **A ceiling 5–30× above the data is reached only through the fitted
tail and the trend's extrapolation, and every number below is an ESTIMATE of that kind.**

| ceiling | growth form (fit ≥ 100M) | AIC | median at the cap | first crossing: 5 % / 50 % / 95 % | **P(cross before 500M)** [5–95 % over replicates] |
|---|---|---|---|---|---|
| **p99 ≥ 1e-4** | linear | −4811.6 | 3.1e-5 | > cap / > cap / > cap | **0.03** [0, 0.15] |
| | power | −4807.3 | 2.9e-5 | > cap / > cap / > cap | **0.02** [0, 0.11] |
| | **exponential** (best AIC) | **−4814.6** | 7.3e-5 | **439M / 492M / > cap** | **0.66** [0.18, 1.0] |
| **judged max ≥ 1e-3** | linear | −2307.6 | 7.9e-5 | > cap / > cap / > cap | **0.04** [0, 0.29] |
| | power | −2306.9 | 7.7e-5 | > cap / > cap / > cap | **0.04** [0, 0.26] |
| | **exponential** (best AIC) | **−2312.2** | 2.1e-4 | **409M / > cap / > cap** | **0.25** [0.003, 0.68] |

**Fit-window sensitivity (s ≥ 20M).** Exponential is again the best AIC, by 305 for p99 and 60 for the max over
linear. It gives P(cross) of 0.02 for p99 and 0.19 for the max (first-crossing 5 % point 407M). Linear and power
give ≤ 0.01.

**What the data cannot distinguish.**
- Since 100M, exponential beats linear by ΔAIC 3.0 (p99) and 4.6 (max). That is a preference, not a decision.
- The p99 forecast swings from 0.03 to 0.66 on that choice alone. Linear vs power is a tie.
- **Hold-out check** (fit on s ≤ 137M, the data 177f61c0 had, then predict the 140–210M window medians in log
  units):
  - from 20M, linear under-predicts by +0.06 … +0.23 (up to 26 % low) and power by up to +0.39, while exponential
    errs by −0.03 … +0.14;
  - from 100M, every form is within ±0.12.

  So growth since ~140M is **faster than 177f61c0's linear form**, which is the A3-F2 case. Between linear and
  exponential, only the next ~50–100M of data decides. At 300M the two forms' p99 medians differ by ≈ 24 %, which
  is resolvable.

**What this does to Amendment 3's stated margins.** A3.1 justifies each ceiling as "≥ 4× from its forecast at the
cap" (p99 cap forecast ≈ 2.4e-5, judged max cap median ≈ 6e-5, with P(max ≥ 1e-3) ≈ 5e-9 per update):
- **Refreshed under linear**, the p99 cap median is 3.1e-5 (3.2× margin) and the judged max cap median is 7.9e-5
  (12.7×).
- **Under exponential**, the p99 cap median is 7.3e-5 (**1.4×**) and the judged max cap median is 2.1e-4 (4.7×).
  The per-probe tail then makes a NUMERICAL crossing of either ceiling before the cap more likely than not for the
  p99.

**Sharpness is not the driver.** `train/entropy_loss` is flat from 40M to 211M (medians −1.064 → −1.048), so a
"policy sharpness" form has no moving covariate to fit. The gaps grow while the mean policy entropy does not
(F4).

## 3. Interpretation

The tail's MECHANISM is unchanged: every replayable event is fp32 rounding on the most ill-conditioned row in reach.
What has changed is the SCALE: the whole distribution of gaps is moving up, about 1.7× in the median from 100M to
210M. A NUMERICAL crossing of a FIXED ceiling therefore becomes a live possibility before the cap. Under Amendment 3
that would be a GIGO stop, a false alarm in the sense 177f61c0 argued against for the 1e-4 bar.

## 4. Recommendation, in the registered rule's terms

**(b) A pre-registered WATCH TRIGGER.** The two other options are rejected:
- **(a) "no change"** ignores that A3.1's margin argument no longer holds under the best-fitting form;
- **(c) "re-diagnosis now"** has nothing to diagnose. The mechanism holds on every replayable dump, and both
  statistics sit 5–6× below their ceilings.

The ceilings themselves stay as registered. They are the fault detector, and changing them is an amendment, not
this unit's call.

The trigger proposed for the orchestrator to register (as an Amendment 3 addendum) follows. Each part is a fixed
threshold on the logged statistics, read at full precision through `main.ops.tb_read.load`. Per rule 8, a value
within 1e-9 (relative) of a threshold is not a crossing.

- **W1 (scale).** Any CLOSED 10M window [k·10M, (k+1)·10M) whose median p99 is ≥ **3.0e-5** (2.2× today's
  1.39e-5). Under the exponential fit this fires at ≈ 341M, about 100M before the forecast's 5 % first-crossing
  point (439M). Under the linear fit it fires at ≈ 482M, so it stays quiet if growth is linear.
- **W2 (single probe, halfway to a ceiling).** Any probe with p99 ≥ **5e-5** or judged max ≥ **5e-4**.
- **Either one ⇒ a re-diagnosis (A3-F2), not a stop.** That means:
  1. this directory's `replay_dumps.py` on the newest dumps that carry weights;
  2. `forecast.py` refreshed;
  3. a decision BEFORE the forecast's 5 % first-crossing step. Candidates are 177f61c0 §5's per-row
     sensitivity-scaled bar, or re-registered ceilings.
- **Routine.** Re-run `tb_extract.py` + `forecast.py` at every 25M read point (≈ 1 min of CPU). By ≈ 300M the
  growth form should be decided.

Until W1 or W2 fires, nothing about X26 changes.

## FINDINGS (standing rule 7)

- **F1. Weights are dumped for at most 3 violations PER PROCESS** (`consistency.py` `_DUMP_WEIGHTS_MAX = 3`, pin
  `9c033489`). The current process spent them on u19300 / u20490 / u20630. **u21290 and u21440 have rows but no
  weights**, so they cannot be replayed, and no further event can be until the next restart resets the count.
  With the 24 h restart interval, most events between restarts will be weight-less. If the re-diagnosis needs
  fresh weights, the nearest checkpoint is NOT a substitute: weights ten updates apart move these rows by ~0.8
  nats (177f61c0 §2).
- **F2. The run resumed from `checkpoint_136000153`, not `final_model_exception.zip` as A3.3 registered.** The
  launcher log shows two attempts from the exception save (06:49, 07:12) and then the live child (07:45) from
  136,000,153. `design_x26_readiness.md` §4.2 records it as a startup-gate refusal, so it is documented. But
  **Amendment 3 §A3.3 still says the run resumed from the exception save**, so that clause is stale. I did not edit
  the preregistration, which is out of scope. The effects:
  - TensorBoard holds two lineages over 136.2–137.0M;
  - the u13900 event sits on the abandoned branch;
  - `forecast.py` keeps the live lineage, and its self-check refuses a record it cannot place.
- **F3. T2's exact behaviour program** (inductor CUDA + a CUDA graph) was not re-run: there was no GPU lease, and
  the CPU-inductor build fails (177f61c0 F2). On u20490 that program IS the far side of the gap. The stored side is
  bounded by the row's dither envelope, not reproduced.
- **F4. The growth driver is unidentified.** Mean entropy is flat, so "sharpening" in the entropy sense is not
  it. Weight or activation magnitude growth, or the policy moving into more ill-conditioned regions, are
  candidates. **UNVERIFIED.**
- **F5. The exponential and linear forms are not yet distinguishable** (ΔAIC 3.0 / 4.6 since 100M). The
  headline 0.66 / 0.25 is conditional on the exponential form. Under linear or power it is ≤ 0.04.
- **F6. The 1e-3 ceiling is reached only by extrapolating a fitted tail** (generalised Pareto shape ξ ≈ −0.12 for
  the max since 100M), 4.7–29× above the data. The interval is wide by construction.
- **F7. The brief's premise needed correcting.** "The tail is running ahead of the forecast" is true of the SCALE
  (the median grows about 2× the 177f61c0 linear slope since 137M) and false of the event COUNT (5 vs 5.4
  expected).
- **F8. A sixth record (u21440, 211.11M) landed during this measurement.** It is included above. The run is live,
  so `forecast.json` is a snapshot at 211.9M.
