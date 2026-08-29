# Probe O — what the win-prob head reads over the five penultimate decisions before a stall/cap ending

**Date** 2026-08-29 · **Registered** ledger `627ab58` · **Data** `stall_tail_head_reading_2026-08-29.json`
· **Producers** `stall_tail_head_reading_census.py` / `_final.py` / `_meta.py` / `_vsign.py` (beside this file)

**Verdict in one line: the 13/14 blindness the deadline clock was bought to fix is GONE and stays
gone — 81.2% → 22.2% positive V, replicated at 44× the original sample — but a residual,
heavy-tailed stall-tail over-confidence is REAL, is 4.3× the ordinary-loss rate, and is LARGER in
the clock era than before it. Both registered predictions land, though prediction 2 fails under the
letter of its own criterion and passes decisively under its substance.**

---

## 0. What was read, and what was not

Recorded values only — **no model was loaded and no forward pass was run.** Per battle:
`summary.json → meta.{result, turns}` and its `invocations[*].outcome.events`, plus
`states.npz → {win_probs, values, has_state}`. `phi` is `sigmoid(win-prob head logit)` exactly as
the head emitted it at decision time (`inference/player.py::_win_prob`); `V` is the
**PopArt-denormalized** critic value (`policy.predict_values` → `_denorm`), so its sign is in real
reward units and is directly comparable to the historical readings.

**Corpus.** 136,955 trace summaries across the 138 run dirs under `models/` carrying `eval_traces`.
4,869 battles entered the analysis.

## 1. Pre-registered definitions

Frozen from a census that read **only** `meta.result` and `meta.turns` — no `phi` value was
inspected before these were written.

| class | definition |
|---|---|
| `CAP` | `turns >= 250` — the `MAX_TURNS` forfeit deadline. `Gen3Env.action_to_order` returns `ForfeitBattleOrder` at the threshold, so a cap ending is recorded as a **`LOSS` at exactly 250 turns**; there is no `DRAW`/`TIE` result in this schema. |
| `CAP_STALL` / `CAP_TRADE` | `CAP` with zero / ≥1 faints in the final 20 game-turns |
| `STALL_LOSS` | `LOSS`, `100 <= turns < 250`, zero faints in the final 20 turns |
| `LONG_LOSS_SLOW` / `_FAST` | `LOSS`, `100 <= turns < 250`, ≤1 / >1 faints in the final 20 turns |
| `REG_LOSS` (control) | `LOSS`, `turns < 50` (the corpus p90 is ~48) |
| `LONG_WIN` (control) | `WIN`, `turns >= 100` — separates *long* from *doomed* |

| era | definition |
|---|---|
| `clock` | `ai_v9_13` (gen-11) onward — the obs deadline clock is present **and** the win-prob head is on |
| `preclock` | `ai_v6_03` … `ai_v8_20` — head on, **no** clock |
| `current` | `ai_v9_29` (rev-1) onward — the current-arch sub-pool of `clock` |

Metrics over the last **K = 5** recorded decisions (`has_state == 1`), K = 10 kept as context:
`detect` = **the registered criterion**, `phi_T <= 0.5 OR phi_T < phi_{T-4}`; its two halves
`detect_le05` and `detect_decl` reported separately; `overconf` = `max(phi over last 5) >= 0.70`
(C3-comparable); `c3band` = `mean(phi over last 5) ∈ [0.70, 0.98]` (the C3 band); `vpos` = `V > 0`
at the final decision (the 13/14 analogue).

### 1a. One pre-registered class came back EMPTY — and that is a finding, not a bug

`STALL_LOSS` (a long, no-progress loss that does **not** cap) has **zero members** in the entire
corpus. The signature works; it simply separates perfectly:

| | zero faints in final 20 turns | zero faints in final 10 turns |
|---|---|---|
| `CAP` (n=135, clock era) | **81.5%** | 85.2% |
| long non-cap losses (n=400) | **0.0%** | 0.0% |

**In this corpus "stall pattern" and "cap ending" are the same population.** A long game whose tail
stops producing faints runs to the deadline; a long game that ends before the deadline ends because
something died. The stall tail therefore has exactly one observable class, and `CAP` is it.

## 2. Per-class φ trajectory

Median φ at T−4 … T; rates over the same last-5 window.

### CLOCK ERA — `ai_v9_13`+ (deadline clock in obs), n = 2350

| class | n | T−4 | T−3 | T−2 | T−1 | **T** | mean φ_T | detect (registered) | φ_T ≤ 0.5 | decl | overconf | c3band | V>0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CAP_STALL** | 110 | 0.202 | 0.198 | 0.214 | 0.193 | **0.146** | 0.309 | 0.973 | **0.691** | 0.827 | 0.209 | 0.145 | 0.209 |
| **CAP_TRADE** | 25 | 0.474 | 0.549 | 0.551 | 0.636 | **0.646** | 0.489 | 0.680 | **0.480** | 0.560 | **0.480** | 0.280 | 0.280 |
| LONG_LOSS_SLOW | 69 | 0.100 | 0.090 | 0.074 | 0.067 | 0.047 | 0.118 | 0.957 | 0.942 | 0.826 | 0.058 | 0.043 | 0.043 |
| LONG_LOSS_FAST | 430 | 0.139 | 0.113 | 0.090 | 0.069 | 0.046 | 0.121 | 0.970 | 0.949 | 0.830 | 0.100 | 0.035 | 0.065 |
| REG_LOSS | 1200 | 0.299 | 0.236 | 0.163 | 0.093 | 0.061 | 0.152 | 0.946 | 0.908 | 0.869 | 0.207 | 0.041 | 0.109 |
| LONG_WIN | 516 | 0.974 | 0.977 | 0.980 | 0.983 | **0.986** | 0.947 | 0.252 | 0.023 | 0.238 | 0.969 | 0.452 | 0.963 |

### PRE-CLOCK ERA — `ai_v6_03`…`ai_v8_20`, n = 2519

| class | n | T−4 | T−3 | T−2 | T−1 | **T** | mean φ_T | detect | φ_T ≤ 0.5 | decl | overconf | c3band | V>0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAP_STALL | 219 | 0.029 | 0.027 | 0.028 | 0.027 | 0.029 | 0.199 | 0.890 | 0.826 | 0.521 | 0.114 | 0.100 | 0.146 |
| CAP_TRADE | 23 | 0.046 | 0.097 | 0.059 | 0.053 | 0.049 | 0.252 | 0.870 | 0.826 | 0.348 | 0.217 | 0.174 | 0.174 |
| LONG_LOSS_SLOW | 214 | 0.016 | 0.014 | 0.012 | 0.010 | 0.008 | 0.057 | 0.991 | 0.991 | 0.734 | 0.033 | 0.023 | 0.005 |
| LONG_LOSS_FAST | 515 | 0.104 | 0.081 | 0.060 | 0.044 | 0.027 | 0.086 | 0.981 | 0.965 | 0.866 | 0.101 | 0.021 | 0.029 |
| REG_LOSS | 1165 | 0.189 | 0.122 | 0.077 | 0.048 | 0.027 | 0.098 | 0.976 | 0.953 | 0.891 | 0.154 | 0.028 | 0.044 |
| LONG_WIN | 383 | 0.953 | 0.960 | 0.971 | 0.980 | 0.987 | 0.927 | 0.107 | 0.029 | 0.097 | 0.956 | 0.441 | 0.961 |

### CURRENT-ARCH SUB-POOL — `ai_v9_29` (rev-1) onward, n = 630 — **REPORTED THIN**

| class | n | T−4 | T−3 | T−2 | T−1 | **T** | detect | φ_T ≤ 0.5 | overconf | V>0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CAP_STALL** | **11** | 0.035 | 0.027 | 0.028 | 0.042 | 0.011 | 1.000 | 0.818 | 0.000 | 0.091 |
| **CAP_TRADE** | **3** | 0.838 | 0.874 | 0.849 | 0.875 | **0.856** | 0.333 | 0.333 | 0.667 | 0.667 |
| LONG_LOSS_SLOW | 22 | 0.094 | 0.087 | 0.098 | 0.152 | 0.069 | 0.955 | 0.909 | 0.091 | 0.091 |
| LONG_LOSS_FAST | 46 | 0.156 | 0.143 | 0.084 | 0.065 | 0.033 | 0.891 | 0.891 | 0.109 | 0.109 |
| REG_LOSS | 516 | 0.301 | 0.234 | 0.169 | 0.091 | 0.059 | 0.950 | 0.919 | 0.192 | 0.070 |
| LONG_WIN | 32 | 0.929 | 0.959 | 0.954 | 0.974 | 0.985 | 0.125 | 0.062 | 1.000 | 0.938 |

**14 cap endings in the whole current-arch sub-pool. Nothing here is a result on its own** — it is
consistent with the clock-era pool (11 of 14 read low; the 3 that were still trading read 0.86 and
2 of 3 carried a positive V), and it is quoted only to show the sub-pool does not contradict the
pool. The statistics below are the clock-era pool.

### The head is NOT merely "long game ⇒ uncertain"

`LONG_WIN` at 128 median turns reads **φ_T = 0.986, 96.3% ≥ 0.7**. The head separates
long-and-winning from long-and-doomed cleanly on average. The failure is not a length effect; it is
a **heavy right tail on the doomed side**.

## 3. The registered contrasts (run-clustered bootstrap, cluster = run)

| era | metric | CAP | REG_LOSS | diff | CI95 | |
|---|---|---:|---:|---:|---|---|
| clock | **`detect` (registered)** | 0.919 | 0.946 | **−0.027** | [−0.069, +0.018] | **n.s.** |
| clock | `detect_le05` | 0.652 | 0.908 | **−0.256** | [−0.315, −0.185] | **SIG** |
| clock | `overconf` | 0.259 | 0.207 | +0.053 | [−0.027, +0.123] | n.s. |
| clock | `c3band` | 0.170 | 0.041 | **+0.130** | [+0.066, +0.189] | **SIG** |
| clock | `vpos` | 0.222 | 0.109 | **+0.113** | [+0.024, +0.221] | **SIG** |
| preclock | `detect` | 0.888 | 0.976 | −0.088 | [−0.159, −0.040] | SIG |
| preclock | `detect_le05` | 0.826 | 0.953 | −0.126 | [−0.217, −0.067] | SIG |
| preclock | `overconf` | 0.124 | 0.154 | −0.030 | [−0.085, +0.042] | n.s. |
| preclock | `c3band` | 0.107 | 0.028 | +0.079 | [+0.035, +0.144] | SIG |
| preclock | `vpos` | 0.149 | 0.044 | +0.105 | [+0.049, +0.191] | SIG |

### 🚨 The registered criterion was badly chosen, and it must be reported as registered

`detect` = "φ declining **or** ≤ 0.5". Its `decl` half (`phi_T < phi_{T-4}`) fires at **0.83–0.89 in
every class including regular losses** — φ drifts down in the last five decisions of almost any
lost game, so that half carries essentially no information and it saturates the composite. Scored
as written, **prediction 2 is NULL in the clock era** (−0.027, CI spans zero).

Scored on the `≤ 0.5` half alone — which is what the prediction *means* — it is decisive:
**0.652 on cap endings vs 0.908 on regular losses, −0.256 CI [−0.315, −0.185]**, and the
within-run pooled difference agrees at −0.250 CI [−0.324, −0.184]. **Prediction 2: FAILS by the
letter, PASSES by the substance.** Both are stated; the composite is not retro-fitted.

## 4. The meta-analysis the gen-14 runbook registered

`gen14_endofrun_runbook.md` §(c) left this as an explicit follow-up with a stated power
requirement (n ≈ 30 cap-length losses, "no single run at the current trace retention supplies" it)
and named the route: **pool the per-run WITHIN differences, never the raw levels.** That is done
here at n = 134 cap games over 23 clock-era runs.

| era | statistic (cap-length vs ordinary losses, same run) | pooled diff | CI95 | runs | cap games | |
|---|---|---:|---|---:|---:|---|
| clock | V > 0 at final decision | **+0.120** | [+0.010, +0.254] | 23 | 134 | **SIG** |
| clock | φ_T ≤ 0.5 (detection) | **−0.250** | [−0.324, −0.184] | 23 | 134 | **SIG** |
| clock | max φ over last 5 ≥ 0.7 | +0.064 | [−0.039, +0.167] | 23 | 134 | n.s. |
| clock | mean φ in C3 band | **+0.114** | [+0.046, +0.181] | 23 | 134 | **SIG** |
| preclock | V > 0 at final decision | +0.123 | [+0.065, +0.212] | 37 | 220 | SIG |
| preclock | φ_T ≤ 0.5 | −0.155 | [−0.252, −0.097] | 37 | 220 | SIG |
| preclock | max φ ≥ 0.7 | +0.023 | [−0.028, +0.096] | 37 | 220 | n.s. |
| preclock | mean φ in C3 band | +0.093 | [+0.051, +0.162] | 37 | 220 | SIG |

**The §(c) residual is now SIGNIFICANT and it was not before** (it sat at Fisher one-sided p = 0.076
at n = 9). Cap endings carry a systematically worse critic sign, a systematically worse win-prob
reading and systematically more C3-band tails than the same run's ordinary losses. This closes the
gen-14 runbook's registered follow-up.

## 5. The historical delta

### 5a. The 13/14 pattern — GONE, and the clock is the causal candidate

`V > 0` at the final decision of a cap-length loss. This needs no win-prob head, so it can be read
across every era family including the one the 13/14 count came from:

| era family | n | frac V > 0 | mean V | median V |
|---|---:|---:|---:|---:|
| `ai_v6/v7/v8` (pre-clock, different lineage) | 248 | 0.153 | −52.24 | −35.09 |
| **`ai_v9_01-09`** (gen-1…8, **pre-clock**, same lineage) | **612** | **0.812** | **+4.76** | +10.07 |
| `ai_v9_10-12` (gen-9/10; the clock lands 2026-08-12) | 27 | 0.259 | −54.14 | −19.36 |
| **`ai_v9_13+`** (gen-11+, **clock present**) | **135** | **0.222** | **−55.87** | −28.71 |
| *reference:* `ai_v9_09` @16M, the 13/14 hand count | 14 | 0.929 | +9.33 | — |
| *reference:* `ai_v9_09` whole-run, measured here | **93** | **0.849** | +5.68 | — |
| *reference:* gen-13 spot check (runbook) | 9 | 0.222 | −47.2 | −34.0 |

Three things this settles. **(i)** The 13/14 = 92.9% reading was not an n=14 accident — the whole
`ai_v9_09` run reads 84.9% at n = 93, and the whole pre-clock `ai_v9` family reads **81.2% at
n = 612.** **(ii)** The drop to 22.2% is not an n=9 accident either — gen-13's 2/9 = 22% reproduces
at **30/135 = 22.2%, a 15× sample.** **(iii)** The break sits exactly at the clock boundary within a
single lineage (`ai_v9_01-09` → `ai_v9_10-12` → `ai_v9_13+`: 81.2% → 25.9% → 22.2%), which is the
cleanest evidence available that `gen3_deadline_clock_v1` discharged what it was built for **and
that the fix has held for seven generations.**

⚠️ The `ai_v6/v7/v8` family at 15.3% is **not** a counterexample. It is a different architecture
lineage with a different reward scale and a different opponent set; the within-lineage comparison
is the sound one. It is listed to make the confound visible rather than to be read as a level.

### 5b. C3's 0.7–0.98 band — a sixth of stall tails still sit in it, and the fraction ROSE

C3 (gen-12 rollout-PIT) found stall-tail over-confidence at φ 0.7–0.98 where resampled-dice win
rate was 0.0–0.4. Today, clock era:

| | n | mean φ over last 5 ∈ [0.70, 0.98] | max φ over last 5 ≥ 0.70 | **φ_T ≥ 0.5 = an outright MISS** |
|---|---:|---:|---:|---:|
| CAP_STALL | 110 | **14.5%** | 20.9% | **30.9%** |
| CAP_TRADE | 25 | **28.0%** | 48.0% | **52.0%** |
| all CAP | 135 | 17.0% | 25.9% | **34.8%** |
| ordinary losses | 1699 | 4.1% | — | 8.0% |

**34.8% of cap tails end with φ_T ≥ 0.5 on a game that is a LOSS by construction — 4.3× the
ordinary-loss rate. 6.7% end at φ ≥ 0.9 and 4.4% at φ ≥ 0.98.** The extreme cases are not
borderline: `ai_v9_16` vs `sentinel_1` held **0.998 / 0.999 / 0.999 / 0.999 / 0.999** across the
final five decisions into a −30 forfeit, with V = **+14.16**.

### 5c. The clock era is WORSE than the pre-clock era on the residual

CAP endings, clock vs pre-clock, run-clustered:

| metric | clock (n=135) | preclock (n=242) | diff | CI95 | |
|---|---:|---:|---:|---|---|
| `detect` (registered) | 0.919 | 0.888 | +0.030 | [−0.031, +0.110] | n.s. |
| `detect_le05` | 0.652 | 0.826 | **−0.175** | [−0.253, −0.064] | **SIG** |
| `overconf` | 0.259 | 0.124 | **+0.135** | [+0.038, +0.217] | **SIG** |
| `c3band` | 0.170 | 0.107 | +0.063 | [−0.027, +0.135] | n.s. |
| `vpos` | 0.222 | 0.149 | +0.073 | [−0.041, +0.192] | n.s. |

Read this **relative to each era's own control**, because the two eras' heads sit at different
global calibration levels (pre-clock REG_LOSS φ_T median 0.027 vs clock 0.061): clock-era caps are
over-confident at **1.25×** their own regular-loss rate (0.259 / 0.207), pre-clock caps at
**0.81×** theirs (0.124 / 0.154). **The clock era is the first in which cap endings are
distinctively more over-confident than ordinary losses.** The lineage confound is real and large —
`preclock` here is `ai_v6/v7/v8`, not `ai_v9_01-09` — so this is directional, not causal.

## 6. Opponent split — clock-era cap endings

| opponent class | n | mean φ_T | overconf | φ_T ≤ 0.5 | V > 0 |
|---|---:|---:|---:|---:|---:|
| **`staller` / `staller_v2`** | **57** | **0.386** | **0.298** | 0.596 | **0.298** |
| `aggressive*` | 14 | 0.360 | 0.286 | 0.643 | 0.214 |
| `sentinel_*` (pool selves) | 17 | 0.329 | 0.294 | 0.706 | 0.118 |
| `setup_sweep*` | 10 | 0.372 | 0.200 | 0.500 | 0.300 |
| `random` | 14 | 0.265 | 0.143 | 0.786 | 0.143 |
| `heuristic*` | 21 | 0.249 | 0.190 | 0.762 | 0.095 |
| `ext_*` (cross-run) | 2 | 0.454 | 0.500 | 0.500 | 0.500 |

The stall bots are **42% of all cap endings** and the class the head reads highest on. That is the
adversarially relevant cell: the opponent built to stall is the one the head is most confident
against while being stalled to a forfeit.

## 7. Scoring the registered predictions

| # | prediction | verdict |
|---|---|---|
| **P1** | improved vs the 13/14 era, but RESIDUAL stall-tail over-confidence | ✅ **CONFIRMED, both halves.** Improved: 81.2% → 22.2% positive V within the `ai_v9` lineage (n = 612 → 135), the break at the clock boundary, held for seven generations. Residual: 34.8% of cap tails miss outright (4.3× ordinary losses), 17.0% still in C3's band, 4.4% at φ ≥ 0.98; the within-run difference is SIG on `vpos`, `detect_le05` and `c3band`. |
| **P2** | detection HIGHER on regular losses than on stall/cap endings | ⚠️ **SPLIT.** Under the registered composite `detect`: **NULL** (−0.027, CI [−0.069, +0.018]) — the "declining" half fires 83–89% in every class and saturates the criterion. Under `φ_T ≤ 0.5`: **CONFIRMED decisively** (0.652 vs 0.908; −0.256 CI [−0.315, −0.185]; within-run −0.250 CI [−0.324, −0.184]). |

**Historical-delta verdict: BETTER than the 13/14 era by a wide, replicated margin; SAME-to-WORSE
than the C3 era on the residual band** (17.0% of stall tails still sit in φ 0.70–0.98, and the
over-confidence rate relative to a same-run control is the highest it has been). The blind spot the
clean-world design leans on **is still there** — smaller in the mean, unchanged in the tail.

## 8. Class counts and caveats

| class | clock | preclock | current-arch |
|---|---:|---:|---:|
| CAP_STALL | 110 | 219 | **11** |
| CAP_TRADE | 25 | 23 | **3** |
| LONG_LOSS_SLOW | 69 | 214 | 22 |
| LONG_LOSS_FAST | 430 | 515 | 46 |
| REG_LOSS | 1200 | 1165 | 516 |
| LONG_WIN | 516 | 383 | 32 |

1. **The current-arch sub-pool is 14 cap endings. It is reported thin and carries no verdict**;
   every statistic above is the clock-era pool (`ai_v9_13`+), which spans gen-11 → the R3 era.
2. **`STALL_LOSS` is empty by measurement** (§1a) — stall pattern and cap ending are one population
   here, so there is no non-capping stall class to compare against.
3. **Eval traces are QUOTA-sampled**: `_FORENSIC_LOSS_QUOTA = 10` losses and
   `_FORENSIC_WIN_QUOTA = 5` wins per opponent per eval step, taken first-encountered rather than at
   random. The 0.64% cap fraction among *traced* clock-era losses (135 / 21,093) is **not** a run's
   true cap rate and must not be quoted as a stall-rate baseline. Within an opponent the quota has
   no known cap-selective bias, but it is not a random sample and the sampling is not
   exchangeable across opponents.
4. **Pooling across 23 runs is a Simpson hazard.** Every headline contrast is therefore reported
   twice: run-clustered bootstrap (§3, §5c) and per-run within differences pooled (§4). They agree
   on all four SIG results.
5. **`preclock` is a different lineage** (`ai_v6/v7/v8`), so §5c is directional only. The
   within-lineage clock comparison (§5a) is the sound one and is V-sign only, because
   `ai_v9_01-09` carries no win-prob head.
6. **φ is the head's reading, not a ground-truth win probability.** This probe measures whether the
   head *sees* the doomed tail; it does not re-derive the true win rate (that was C3's rollout-PIT).
   The `phi_T >= 0.5` "miss" framing is sound because the class label is a LOSS by construction.
7. `CAP_TRADE` is n = 25 in the clock era. Its 48% over-confidence is the worst number in the
   probe and the least powered; treat it as a hypothesis (caps that were *still trading* are where
   the head is blindest) rather than as a measurement.

## 9. What this means for the clean world

The design bets that with `draw = loss`, stalling is weakly dominated and no anti-stall bias is
needed — the head prices the doomed tail correctly, so frozen-φ shaping will not pay to march
toward a cap. The measurement says the bet is **mostly** good and **specifically** exposed:

- **Mostly good**: the mean stall tail reads low (median φ_T = 0.146 on true stall wars), the
  critic sign is right 78% of the time, and long *wins* are read at 0.986 — so the head is not
  systematically paying for length.
- **Specifically exposed**: on **34.8%** of cap endings it ends above 0.5 and on **4.4%** above
  0.98 — and frozen-φ shaping pays positive reward on exactly those decisions, all the way into
  the forfeit. The exposure concentrates on the **stall bots** (42% of caps, the highest φ of any
  opponent class) and on caps that were **still trading**.

That is a **conditional** blind spot, not a systematic bias, which is the shape the escalation rule
already anticipates: the ledger's mitigation ordering — **a HEAD fix (stall-tail labels from the
counterfactual factory, outcome-unit and theoretically clean) before any bias term** — is the one
these numbers support. A flat anti-stall bias would tax the 65% of tails the head already reads
correctly in order to reach the 35% it does not. The registered escalation trigger (stall-rate
primary endpoint) remains the right gate; this probe does not fire it and does not pre-empt it.

**Registered follow-up, not measured here:** whether the high-φ tail is *reducible*. The φ ≥ 0.98
cases could be off-distribution states the MC-supervised head has never been labelled on (C3's
surviving branch — the training DISTRIBUTION of stall games, which `gen14_endofrun_runbook.md`
already measured at a **14× over**-exposure in training relative to eval losses). If stall states
are over-represented in training and the head is *still* wrong on them, the defect is
representational, not a data-quantity problem — and that is the discriminating measurement the
head-fix decision should turn on.
