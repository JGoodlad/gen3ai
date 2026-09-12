# THE STEP CURVE — does the win-prob critic's opponent conditioning keep improving past 10M?

**Status: COMPLETE.** 8 cycles, 6 pair reads, 8 representation extractions. `PREDICTION.md`
(registered before any cycle, amended before any cycle was read) holds the hypotheses and the bar.

> ## The answer, in one line
> **Seven-fold more training steps buy CALIBRATION, not DISCRIMINATION.** `gate.ece.all` improves
> monotonically past its own bar at every step on both draws (0.102 → 0.017, a 6× reduction);
> resolution on the fixed bot panel is flat to four decimal places (0.0190 → 0.0185, Δ −0.0004);
> and the representation's opponent-class information does not rise at any step — **hypothesis (i)
> is refuted directionally**, on a row that the strength confound cannot produce.

---

## 1. The question and why it is the next one

Ten 10M arms on the critic ladder returned **zero detected registered rows**. The N-curve
(ledger 2026-09-10 *THE N-CURVE*) then showed the conditioning HEAD saturates within ~4,000
battles **for a fixed representation** — so more eval buys no more conditioning, and the ladder's
null is not a power problem on that axis any more.

One axis nobody varied: every arm was 10M steps. **Does the REPRESENTATION keep improving with
training STEPS?** If it does, the ladder measured 10M rather than the critic, and *more steps* is
the cheapest untested lever. If it does not, 10M is where this configuration's opponent
conditioning lands.

## 2. The instrument, and the pin diff

`models/ai_v12_02_winprob_critic` — the completed 75M run — read at four checkpoints against its
own 10M checkpoint. It was chosen because it is the **same argv lineage** as the ladder's control.

**`original_command` diff (`ai_v12_02_winprob_critic` vs `ai_v12_11_ladder_ctrl10M`), by execution,
not by eye:** both argvs are **228 tokens**; **124 of 127 flags are identical**. The three that
differ:

| flag | 75M run | ladder control |
|---|---|---|
| `--pin-commit` | `f971caf2` (Sun 2026-09-06) | `f3502568` (Tue 2026-09-08) |
| `--run-name` | `ai_v12_02_winprob_critic` | `ai_v12_11_ladder_ctrl10M` |
| `--steps` | 75,000,000 | 10,000,000 |

**The RESOLVED configs (`metadata.json` `cli_args`, 278 keys) differ on 12**, and this is where the
argv diff stops being the whole story. Nine are run identity (`run_name`, `run_dir`, `model`,
`steps`) or a `*_source` provenance field. **Three are substantive, and two of those are a DEFAULT
that moved between the pins:**

| key | 75M (`f971caf2`) | control (`f3502568`) | why |
|---|---|---|---|
| `eval_sentinel_greedy` | **False** | **True** | 🚨 **the default flipped at the 2026-09-07 opponent-regime boundary.** Neither run passed the flag. |
| `promote_threshold` | `None` → 0.65 | 0.55 | follows the regime by design |
| `teacher_scan_limit` | *absent* | 60 | flag did not exist at the older pin |
| `arch`, `allow_nonproduction_arch` | *absent* | `None`, `False` | flags did not exist at the older pin (verified by `git grep` at each pin) |

**Architecture is NOT a difference**: both runs record `arch_signature =
gen3_critic_route_wave_v1`, which is also the signature the current tree builds, so every
checkpoint here loads at HEAD. `config_version` differs (110 vs 113) — that is the config
RECORD's version, not the weights'.

## 3. The three structural facts that reshaped the design

Each was discovered before any cycle was generated, and each is a finding in its own right.

### 3.1 The 75M run is PRE-BOUNDARY and its `model_config.json` records no regime at all

`--eval-sentinel-greedy` defaulted **False** at `f971caf2` and **True** at `f3502568`. The key
itself only entered `model_config.json` at config_version 111 — the boundary commit — so the 75M
run (cv 110) records the regime **nowhere `eval_trace_gen` looked**, and the tool refused it
outright. Its own `metadata.json` `cli_args` does record it (`False`), consistent with the pin's
default and with the run's launch date.

**Consequence, landed as `bab72428`:** `eval_trace_gen` now takes
`--eval-sentinel-greedy` / `--no-eval-sentinel-greedy` to DECLARE the regime for a run that records
none, marks the cycle `eval_sentinel_greedy_source: declared`, and — this is the part that matters
— **names the value recoverable from the run's own `metadata.json` in the refusal**, so the
declaration is informed rather than invented. A run that DOES record the regime still wins: a
contradicting declaration is refused. A run that records it nowhere is refused with no flag
offered.

**🚨 A declaration is not a licence to difference across the boundary** — `spec_of` keys on the
regime VALUE, not on how it was learned. See §3.3.

### 3.2 The run's retained pool starts at 36M, so the specified sentinel spec cannot produce the curve

The 75M run's bounded self-play pool evicted everything below **36,000,000** during training; 23
snapshots survive (36M–74M). So `--sentinels 3` — which draws snapshots *below* the read step —
**refuses at 10M and at 20M** ("no pool snapshot below step 9969408"), and at 40M would draw a
different, weaker trio than at 73M. A bots-only cycle is not a fallback either: the decision row
`cond.opp_class_auc.*` is the **pool-vs-bot** class AUC and does not exist without pool cells.

**Resolution (registered in PREDICTION.md's amendment, before any read):**
`--include-current-snapshot --sentinels 3`, verified to select the **identical panel at all four
checkpoints** —

| checkpoint | sentinel_0 | sentinel_1 | sentinel_2 |
|---|---|---|---|
| 10M · 20M · 40M · 73M | 36,000,000 | 56,000,016 | 74,000,016 |

**This ELIMINATES the sentinel-strength-drift confound the task asked to be named** — the panel no
longer moves with the x-axis — and replaces it with one that is **conservative for the hypothesis
under test**: the panel sits at late-run strength, so the 10M checkpoint faces opponents far
stronger than itself and the 73M checkpoint faces near-peers. Pool-cell outcomes converge toward
the bot cells as steps rise, which makes the two classes *harder* to separate, pushing the decision
row **DOWN with steps**. A rise seen anyway is therefore stronger than the registered design would
have given; a fall is confounded and is reported UNDECIDED, never as evidence for "flat".

The nine scripted bots are fixed and identical at every checkpoint under either design, so
**`gate.resolution.bot` carries the reading no sentinel choice touches.**

### 3.3 The cross-run control read is REFUSED, correctly

`critic_read`'s `check_comparable` requires two offline frames to share games, opponent set,
capture rule and **sentinel regime**. Mine and the ladder control's specs differ on exactly one
key:

```
ctrl10M : {..., 'capture': 'ALL', 'n_games': 400, 'eval_sentinel_greedy': True,  'complete': True}
75M     : {..., 'capture': 'ALL', 'n_games': 400, 'eval_sentinel_greedy': False, 'complete': True}
```

The verbatim refusal is recorded in §6. **The within-run curve is unaffected** — all eight cycles
share one run, one regime, one panel, one spec — and the within-run curve is the question. Any
comparison to the ladder roster is reported as LEVELS side by side and **never as a delta**
(rule 20).

## 4. What was generated and read

**Eight cycles**: 4 checkpoints × 2 independent draws (seeds 20260909 / 20260910), 400 games ×
12 opponents, full capture, `--concurrency 1` (reproducible), `--workers 4 --nice 15`, rust
bridge, no server. Nothing written under `models/`; the shadow run dirs live under the session's
job tmp.

**Six reads**: {draw1, draw2} × {20M, 40M, 73M} vs the run's OWN 10M **of the same draw**,
`main.ops.critic_read` at the registered defaults (`--states 800 --anchors 150 --rollouts 8`,
the ladder's own).

**The bar** (PREDICTION.md): a rise counts only if every consecutive gap exceeds the larger of the
two checkpoints' own two-draw spreads `W(s)`, on both draws. The ladder's `hp400_floor.json` is
passed so the report prints it, and it **decides nothing here** — it was measured on a different
run, a different panel and the other side of the regime boundary.

---



## 5. The table

All eight cycles completed at 4,800/4,800 battles, one identical sentinel panel, one regime, two
seeds (`json/cycle_provenance.json` carries every cycle's sentinel steps, checkpoint sha, seed and
completeness). Every quantity below is **Δ against the run's OWN 10M checkpoint on the SAME draw**;
`W(s)` is that checkpoint's two-draw spread and the bar is `max(W)` over the pair.

Full per-row tables: [`stepcurve_table.md`](stepcurve_table.md) · numbers in
[`json/stepcurve_table.json`](json/stepcurve_table.json).

### 5.1 The decision rows

| row | Δ(10M→73M) d1 / d2 | max W | verdict vs PREDICTION.md |
|---|---|---|---|
| `cond.opp_class_auc.t4_10` | −0.0596 / −0.0372 | 0.0264 | **UNDECIDED** — falls, and the fall is confounded (§5.3) |
| **`gate.resolution.bot`** | **−0.0004 / −0.0010** | 0.0069 | **(ii) SUPPORTED — FLAT** |
| `gate.resolution.all` | −0.0114 / −0.0207 | 0.0093 | UNDECIDED — but see the base-rate cap (§5.3) |
| `cond.opp_class_auc.t1_3` | +0.0024 / −0.0061 | 0.0154 | **(ii) SUPPORTED — FLAT** |

**Not one decision row rises.** Two are flat by the bar; two fall. `gate.resolution.bot` is the
cleanest row in the measurement — nine FIXED scripted bots, a weighted mean over cell means (so
neither quota-matched nor frame-sensitive), untouched by any sentinel choice — and it ends at
**−0.0004** with a CI straddling zero. Levels: 0.0190 → 0.0185.

### 5.2 The strength-robust row — the primary reading

Added after the confound in §5.3 was measured but **before the curve was complete**, at the
coordinator's direction and for the reason §5.3 gives: `cond.opp_class_auc.*` is the AUC of **V**,
and V's ability to separate pool from bot MUST fall as the two classes' outcomes converge. That row
therefore cannot answer a question about the REPRESENTATION. This one can.

🚨 **A correction that changed what got measured.** The v6 tool does **not** fit its
`cond.spread_ratio_optimal` posterior from `value_pooled` — it fits `q(o | V)` from **V**, the
scalar output, and does no model forward at all (which is exactly why the tool declares that row a
bound at the head's OUTPUT, not at its information set). Reading it as the strength-robust row
would have reintroduced the very sensitivity being escaped. The `0.82` reference figure comes from
the N-curve's `frame_check.py`, which DOES forward the model. So this row is produced by reusing
**`measurements/winprob_refit_ncurve_2026-09-10/extract.py` + `frame_check.py` in place, unchanged**
— out-of-fold, battle-grouped (`GroupedRidgeCV`), weighted AUC. The extractor's own QC reports
`max|V_fwd − V_rec| = 1.01e-06`, i.e. the forward reproduces the recorded win-probs exactly.

**Three readings on IDENTICAL states, turns 4–10** (`scripts/pooled_table.py`):

| checkpoint | draw | own-team leak (chance = 0.50) | **pooled → class** | V → class |
|---|---|---|---|---|
| 10M | draw1 | 0.4953 | **0.8981** | 0.7976 |
| 10M | draw2 | 0.4752 | **0.8857** | 0.7835 |
| 20M | draw1 | 0.4939 | **0.8957** | 0.7547 |
| 20M | draw2 | 0.4741 | **0.8922** | 0.7703 |
| 40M | draw1 | 0.4963 | **0.8572** | 0.7397 |
| 40M | draw2 | 0.4739 | **0.8673** | 0.7515 |
| 73M | draw1 | 0.4906 | **0.8624** | 0.7363 |
| 73M | draw2 | 0.4764 | **0.8548** | 0.7382 |

**The frame is clean.** The own-team → opponent-class leak — the artefact that made the original
probe read's 0.846 spurious — sits at **0.474–0.496 at every checkpoint**. It is reported so a
reader sees it rather than takes it on trust.

**The bar, applied:**

| gap | draw1 | draw2 | bar | clears on BOTH draws? |
|---|---|---|---|---|
| 10M → 20M | −0.0024 | **+0.0065** | 0.0124 | no — signs disagree, both inside |
| 20M → 40M | **−0.0385** | **−0.0249** | 0.0101 | **YES — DOWN on both** |
| 40M → 73M | +0.0052 | −0.0125 | 0.0101 | no |

**10M → 73M: −0.0357 / −0.0309** against max W = 0.0124.

**⇒ (i) is REFUTED directionally; (ii) is not met either (|Δ| ≈ 3× max W); formally UNDECIDED
between the registered pair.** No gap is up-and-past-the-bar. The one gap that clears the bar
clears it **downward on both independent draws**. And it is a **step change, not a slope** — flat
to 20M, a real drop in 20M–40M, flat again to 73M — so a smooth step-response is refuted alongside
(i).

### 5.2b 🚨 A PEER's independently measured FLOOR makes the one bar-clearing gap MARGINAL

`measurements/repr_class_decode_2026-09-11` (landed on main while this was generating, and found
on the rebase) ran the SAME `pooled → class` row on five ladder checkpoints and priced its
**run-level control-pair floor at 0.023 (hp800) / 0.031 (hp800b)** — larger than the `max W = 0.0124`
this measurement derives from its own re-draws.

**Both bars are right about different things, and the difference is rule 19.** `W(s)` is the
EVAL-DRAW component, fixed by re-drawing one checkpoint — which is what the registered bar names and
what two draws buy. The peer's 0.023–0.031 is the RUN-TO-RUN component, measured between two
DIFFERENT control runs. Four checkpoints of ONE run are not four runs, so the run-level floor does
not straightforwardly apply here; but it is the honest upper bar, and against it:

| | the 20M→40M gap | vs `max W` = 0.0124 | vs the peer's run-level floor 0.023 / 0.031 |
|---|---|---|---|
| draw1 | −0.0385 | clears (3.1×) | clears 0.023, clears 0.031 |
| draw2 | −0.0249 | clears (2.0×) | **marginal at 0.023, INSIDE 0.031** |

**So the fall is unambiguous on the registered eval-draw bar and MARGINAL on the stricter
run-level one.** The direction is unaffected — nothing here rises on any bar, so **(i) stays
refuted either way** — but the strength of "a real drop between 20M and 40M" is bar-dependent and
is stated as such rather than at its most favourable reading. Pricing it properly would need a
second 75M run, which is not scheduled.

**An unlooked-for cross-instrument validation, reproduced here independently.** The peer notes
`frame_check.py`'s `V_to_opp_class_AUC` agreeing with `critic_read`'s tool-v6
`cond.opp_class_auc.t4_10` to ±0.005 on five ladder checkpoints. The same holds on these frames,
between two differently-built fitters on differently-matched frames:

| | `frame_check` V→class | `critic_read` v6 | diff |
|---|---|---|---|
| 10M draw1 | 0.7976 | 0.7965 | 0.0011 |
| 73M draw1 | 0.7363 | 0.7370 | 0.0007 |
| 10M draw2 | 0.7835 | 0.7815 | 0.0020 |
| 73M draw2 | 0.7382 | 0.7443 | 0.0061 |

Note the peer's finding that the representation row is **noisier, slower and tells the same story**
as the V row on the ladder's arms. That is consistent with what happens here — except that on THIS
question the two rows do NOT tell the same story, and that is the whole point of §5.3: across a
strength gradient the V row acquires a confound the representation row does not have. The rows are
redundant when comparing equal-strength arms and are not redundant when comparing checkpoints of
one run.

### 5.3 The mechanism — why the V-based rows fall, measured not assumed

The panel is fixed, so the panel does not move; the TRAINEE does, and two registered rows are
bounded by quantities that move with it (`scripts/strength_drift.py`, read off trace filenames — no
model forward, no fitted decoder):

| step | draw1 win rate | p(1−p) | bot wr | **pool wr** | **bot−pool outcome gap** |
|---|---|---|---|---|---|
| 10M | 0.7990 | 0.1606 | 0.9056 | 0.4792 | 0.4264 |
| 20M | 0.8327 | 0.1393 | 0.9217 | 0.5658 | 0.3558 |
| 40M | 0.8325 | 0.1394 | 0.9142 | 0.5875 | 0.3267 |
| 73M | 0.8383 | 0.1355 | 0.9131 | 0.6142 | 0.2989 |

1. **The bots are saturated at 10M.** Bot win rate is flat across seven-fold more training
   (0.906 → 0.913); every gain lands on the pool (0.479 → 0.614). The fixed scripted panel cannot
   see whatever the extra 63M steps bought.
2. **The class-AUC fall is mechanically driven.** `cond.opp_class_auc.t4_10` falls 0.796 → 0.737
   while the bot−pool OUTCOME gap falls 0.426 → 0.299. Two classes whose outcomes converge are
   harder to separate whatever the representation holds — exactly the conservative confound
   registered in the PREDICTION amendment, which is why that row is UNDECIDED and not evidence.
3. **Most of the resolution fall is the cap moving.** Murphy resolution is bounded by base-rate
   variance, which drops 16% (0.1606 → 0.1355) while `gate.resolution.all` drops 21%. The
   strength-normalised row, `identity.resolution_cap_share`, is NOT DETECTED (−0.064 / −0.055,
   both CIs covering zero).
4. **Arithmetic, offered as arithmetic:** over 10M→73M the V row falls 0.061 and the representation
   row falls 0.036. Consistent with ~0.025 of the V fall being outcome convergence and the rest
   the representation — but this is a subtraction, not a proven decomposition.

### 5.4 What DOES improve with steps — calibration

| row | 10M → 73M level (d1) | Δ d1 / d2 | every gap past its bar? |
|---|---|---|---|
| **`gate.ece.all`** | **0.1023 → 0.0170** | −0.0853 / −0.0909 | **YES — DOWN on all three gaps, both draws** |
| `identity.bias.ALL` | 0.1125 → 0.0062 | −0.1063 / −0.0891 | (run-level, rule 19 — reported) |
| `cond.calibration_slope.all` | 1.0213 → 1.0850 | +0.0637 / +0.0303 | slope stays near 1 at both ends |

**`gate.ece.all` is the only row in the measurement that moves monotonically past its own bar at
every single step, on both draws** — a 6× reduction in expected calibration error. Its formal label
is UNDECIDED only because the registered (i)/(ii) test is written about the CONDITIONING direction
and ECE is a REPORTED row; the movement itself is unambiguous and replicated.

In Murphy terms: **reliability improves substantially, resolution does not move.** For a project
whose stated goal is a win-prob value function, that is the operative finding — steps buy a
better-calibrated critic, not a more discriminating one.

🚨 **A bug caught in this file's own tooling.** The first `tabulate.py` tested gaps with `g > bar`
(rise-only), which marks every downward move "no" — and would have hidden §5.4 entirely, since ECE
improving IS a downward move. It now tests `|Δ| > bar` and reports direction separately;
`rising`, which hypothesis (i) needs, still requires the gaps to be UP.

### 5.5 Reported, not decided

`cond.spread_ratio.t1_3` (+0.054 / +0.070) and `.t4_10` (+0.115 / +0.184) both rise, and t4–10
clears its bar on the first gap — but both carry run-level floors of 0.19–0.22 from the ladder
(ledger 2026-09-11) that a single run cannot price, so they are REPORTED. `cond.own_team_r2.t1` is
FLAT and remains PROVISIONAL under rule 18. `gate.skill.bot` is UNDECIDED with the widest W of any
row (0.0587).

---

## 6. The reading, and every confound named

**The question was: does the REPRESENTATION keep improving with steps past 10M? The answer is no.**
Not on the strength-robust row (refuted directionally, with the only bar-clearing gap pointing
down), not on the clean fixed-bot resolution row (flat to four decimals), not on either class-AUC
row. What improves is calibration, and it improves a lot.

**The confounds, each as a verified fact:**

1. **Sentinel-strength drift — ELIMINATED by design, at a price.** The registered spec could not
   run (§3.2); the fixed panel removes the drift entirely and replaces it with a bias that is
   **conservative for (i)** — pool/bot outcome convergence pushes the decision row DOWN with steps.
   A rise seen anyway would have been stronger than the original design could deliver; the fall is
   confounded and is treated as UNDECIDED, never as evidence for (ii). The representation row
   exists precisely because it is immune to this.
2. **The pin difference, and a default that moved.** `f971caf2` vs `f3502568` straddle the
   2026-09-07 boundary: `--eval-sentinel-greedy` defaulted **False** then **True**, neither run
   passed it, and three flags (`--arch`, `--allow-nonproduction-arch`, `--teacher-scan-limit`)
   did not exist at the older pin. Architecture is NOT a difference (same `arch_signature`).
3. **The cross-run control read is REFUSED, and correctly.** Verbatim:
   > `REFUSING: both cycles are offline-generated, but to DIFFERENT specs — they are not two draws from one population.`
   > `  eval_sentinel_greedy: arm=False  control=True`

   No delta against `ai_v12_11_ladder_ctrl10M` or any ladder arm is formed or quoted. Note the
   LEVELS are not comparable either: this run's `cond.opp_class_auc.t4_10` reads 0.74–0.80 against
   the ladder roster's 0.688–0.714, which is the fixed late-strength panel making class separation
   an easier problem — not this run being better conditioned.
4. **ONE RUN.** Every point is one trajectory; steps and this run's path are not separable. The
   decision rows are eval-draw-level and get their width from the re-draws; the run-level rows
   (identity bias, the spread ratios, the calibration slope) are REPORTED and carry no verdict.
5. **The quota match matches on the OUTCOME MIX — which is what steps change.** `critic_read`
   subsamples both sides to the elementwise minimum of their W/L/D profiles (20M vs 10M: arm
   400/181/8, control 399/215/4 → 399/181/4). On a ladder of equal-strength 10M arms this is
   nearly a no-op; across a 10M→73M strength gradient it is matching away part of the treatment's
   own footprint. It touches the frame-sensitive fitted rows, **including the V-based decision
   row**, and NOT `gate.resolution.bot`. It also makes the 10M control's own LEVEL pair-dependent
   (measured wobble 0.0017–0.0048), which is why the curve is built from deltas.
6. **A curriculum explanation was looked for and NOT found.** A step change in one interval invites
   one, so the run's own TensorBoard was read over the three intervals
   (`scripts/curriculum_read.py`, read-only):

   | | 0–10M | 10M–20M | 20M–40M | 40M–73M |
   |---|---|---|---|---|
   | `train/selfplay_fraction` | 0.675 | 0.900 | 0.900 | 0.900 |
   | **bot share of training** (`1 − nonbot_fraction`) | 0.325 | **0.100** | **0.100** | **0.100** |
   | `eval/pool_snapshot_count` | 1 | 5 | 13 | 19.9 |

   **Nothing changes over 20M–40M that does not change elsewhere.** The whole curriculum shift
   (0.325 → 0.100 bot share) happened BEFORE the curve begins. The pool reaches its cap of 20 at
   step **44,000,016**, so snapshot eviction starts in 40M–73M — the interval where the
   representation row is FLAT (and this is why `snapshots/` retains only 36M upward). *Hypothesis,
   stated as one:* the strata result suggests class-balanced weighting raises V's class
   discrimination, and bot rows becoming rare in the buffer is one way a representation could lose
   class information — the bot share IS a low 0.100 throughout, but it does not CHANGE in the
   right window. **One run and one interval cannot separate a curriculum change from anything else
   moving in the same window, and this negative does not license "therefore it was something
   else".**
7. **EVAL-vs-TRAINING population.** A fixed 12-opponent panel is not the training mix.
8. **γ = 0.99 vs the run's 1.0.** `eval_trace_gen` reads gamma from `model_config.json`, which
   records none for EITHER run, so every offline cycle in this campaign — mine and the landed
   ladder's — is generated at 0.99 while both runs trained at 1.0. It reaches only
   `BattleRecorder._td_residuals`, which no row in `main.ops` consumes, and it is constant across
   all eight cycles. Reported, not corrected, not load-bearing.

### Hazards worth landing as findings

- **`critic_read`'s cache degenerates on a WITHIN-RUN pair.** `stamp_dir` keys on the run NAME
  while the cache FINGERPRINT correctly keys on read-root and cycle, so when arm and control are
  the same run the arm's stamp clobbers the control's and the identity + gate blocks recompute
  every pair (the separately-keyed `cond_readout.json` reuses correctly). It cannot serve a wrong
  readout; it roughly doubles the cost — ~1.7–2.1 h per read here. Remedy: give `stamp_dir` the
  key the fingerprint already uses.
- **`signal/outcome_n_bots` / `_pool` are a FIXED DIAGNOSTIC SAMPLE CAP, not buffer counts.** Both
  read exactly 200 at every one of their ~750 points, so their ratio is 0.5 BY CONSTRUCTION. A
  first draft of `curriculum_read.py` derived "bot share = 0.5000" from them — a fabricated number
  pointing the opposite way from the truth (0.100). The script now carries the real quantity and
  stamps a `CAVEAT` on those two tags in its JSON.

---

## 7. Ready-to-append ledger paragraph

> ### 2026-09-12 · MEASUREMENT · THE STEP CURVE — the win-prob critic's opponent conditioning does NOT improve past 10M; seven-fold more steps buy CALIBRATION, not DISCRIMINATION
>
> `measurements/winprob_step_curve_2026-09-11/`. The completed 75M run `ai_v12_02_winprob_critic`
> read at 10M / 20M / 40M / 73M against its OWN 10M checkpoint, **two independent offline draws per
> checkpoint** (400 games × 12 opponents, full capture, seeds 20260909/20260910, all 8 cycles
> complete at 4,800 battles), so every point carries its own eval-draw width `W(s)` and the
> registered bar is "the between-checkpoint gap must exceed the within-checkpoint draw spread on
> both draws" (`PREDICTION.md`, registered before generation; amended before any read).
> **Registered decision rows: not one rises.** `gate.resolution.bot` — the clean row (nine FIXED
> bots, a weighted mean over cell means, not quota-matched, untouched by the sentinel panel) — is
> **FLAT**: 0.0190 → 0.0185, Δ −0.0004 / −0.0010 against max W 0.0069, both CIs covering zero.
> `cond.opp_class_auc.t1_3` flat; `cond.opp_class_auc.t4_10` and `gate.resolution.all` fall, and
> both falls are CONFOUNDED by the trainee getting stronger against a fixed panel (the bot−pool
> outcome gap closes 0.426 → 0.299; base-rate variance drops 16% while `resolution.all` drops 21%;
> `identity.resolution_cap_share` NOT DETECTED). **The strength-robust row settles it:**
> `value_pooled → opponent class` at turns 4–10 (out-of-fold, battle-grouped, reusing the N-curve's
> own `extract.py` + `frame_check.py`; forward QC `max|V_fwd−V_rec| = 1.01e-06`; own-team leak at
> chance 0.474–0.496 at every checkpoint) reads **0.898/0.886 → 0.896/0.892 → 0.857/0.867 →
> 0.862/0.855**. No gap is up-and-past-the-bar; the ONE gap that clears the bar is **20M→40M, DOWN
> on both draws** (−0.0385 / −0.0249 against 0.0101); 10M→73M = −0.036 / −0.031 against max W
> 0.0124 — though against the PEER's independently measured run-level floor for this row (0.023/0.031, `measurements/repr_class_decode_2026-09-11`) the 20M→40M fall is MARGINAL on draw2, so its STRENGTH is bar-dependent while its DIRECTION is not. **(i) "the representation keeps improving with steps" is REFUTED directionally; (ii)
> "flat" is not met either; formally UNDECIDED between the registered pair — and it is a STEP
> CHANGE in one interval, not a slope, so a smooth step-response is refuted too.** 🚨 **What DOES
> improve is CALIBRATION:** `gate.ece.all` 0.1023 → 0.0170 (−0.085 / −0.091), the only row moving
> monotonically past its own bar at EVERY step on BOTH draws, with the calibration slope near 1 at
> both ends (1.02 → 1.09) and identity bias 0.1125 → 0.0062. In Murphy terms reliability improves
> ~6× while resolution does not move. **Consequence for the ladder: its ten 10M arms' null is NOT a
> statement about 10M only — more steps do not move the conditioning rows either, so "train
> longer" is not the untested lever it looked like.** A curriculum explanation for the 20M–40M step
> was sought in the run's own TB and NOT found: the bot share of training is a constant 0.100
> across all three intervals (the 0.325 → 0.100 shift predates the curve) and the pool cap/eviction
> begins at 44,000,016, inside the interval where the row is flat. **Confounds:** the registered
> `--sentinels 3` spec was IMPOSSIBLE (the run's pool retains only ≥36M, so 10M/20M have no
> sentinel below them) and was replaced by `--include-current-snapshot`, giving an identical panel
> (36,000,000 / 56,000,016 / 74,000,016) at all four checkpoints — this ELIMINATES sentinel drift
> and introduces a bias CONSERVATIVE for (i); one run; the quota match matches on the outcome mix,
> which is what steps change; eval-vs-training population. **The cross-run read vs
> `ai_v12_11_ladder_ctrl10M` is REFUSED and no delta is quoted** — the two pins straddle the
> 2026-09-07 boundary (`--eval-sentinel-greedy` default False→True, neither run passing it) and
> `spec_of` differs on exactly that key; the LEVELS are not comparable either (0.74–0.80 here vs
> the roster's 0.688–0.714, the fixed late panel being an easier separation). **Tooling:**
> `bab72428` lets a PRE-BOUNDARY run's regime be DECLARED (its `model_config.json`, cv 110, records
> none; the refusal now names the value recoverable from `metadata.json` and a contradicting
> declaration is refused) — without it this run could not be re-read at all. **Hazards:**
> `critic_read`'s `stamp_dir` keys on the run NAME so a WITHIN-RUN pair recomputes the identity +
> gate blocks every pair (~2× cost, never a wrong readout); `signal/outcome_n_bots`/`_pool` are a
> fixed sample cap at exactly 200 and their ratio is 0.5 by construction, NOT the training mix;
> every offline cycle in this campaign is generated at γ 0.99 while the runs trained at γ 1.0
> (reaches only `_td_residuals`, which no `main.ops` row consumes). Tag: **MEASURED · (i) REFUTED
> directionally · calibration IMPROVES ~6× · resolution FLAT · steps are NOT the ladder's missing
> lever**.
