# THE 75M RUN-LEVEL FLOOR — `ai_v13_04_flywheel_winprob_b` (W_b) against arm W, 2026-09-18

**For the first time, a 75M row has a floor of its own.** `ai_v13_04_flywheel_winprob_b`
(**W_b**, `--seed 1002`) is `ai_v13_02_flywheel_winprob`'s (**arm W**, `--seed 1001`)
**token-exact seed replicate**: the argv multiset difference is exactly `1001 → 1002` plus the
run name (232 → 232 tokens, verified here as a multiset), same pin `6eb9c776` for all
75,005,952 steps, same declared dose, same `--ent-coef 0.05`, same `--eval-sentinel-greedy`
(`source=argv` on both), both `role: fresh`, both crossed self-play at the **identical
4,128,768**, both retain **20 ladder nodes**.

Every number below was produced **by exactly the code paths that produced arm W's and arm S's**
([`flywheel_pair_read_2026-09-15/`](../flywheel_pair_read_2026-09-15/),
[`flywheel_armS_reads_2026-09-14/`](../flywheel_armS_reads_2026-09-14/)) — the same scripts with
the third arm added, the same harness, the same seeds, the same cells, the same ports. The bars
were pre-registered in [`PREDICTION.md`](PREDICTION.md) and **committed before any W_b number in
this directory existed** (`12836d5f`).

🚨 **THE ONE SENTENCE THAT GOVERNS EVERY ROW: ONE PAIR BOUNDS A FLOOR; IT DOES NOT ESTIMATE ONE.**
Rules 19 and 22. `|W − W_b|` is a single realisation of run-to-run variance. There is no CI on it,
the true floor may be larger or smaller, and a row that clears it has survived **one** replicate —
it has not become a family verdict. Every verdict below is labelled **at n = 2**.

---

## 0. THE FLOOR TABLE — every registered row, finding beside floor

**Sign convention.** The FINDING is the pair read's own `Δ = arm S − arm W`; the FLOOR is
`|arm W − W_b|`, printed unsigned with the signed `W_b − W` in the text. The decision rule was
fixed in `PREDICTION.md` §2 before any number existed:

> **OUTSIDE THE 75M FLOOR** iff `|S − W| > |W − W_b|` **AND** the `S − W` CI **excludes the point**
> `|W − W_b|`. Otherwise **WITHIN FLOOR at n = 2** — which is **never** "equivalent" (rule 6).

| # | row | **finding \|S − W\|** + CI | **floor \|W − W_b\|** | **verdict** |
|---|---|---|---|---|
| 1 | **ladder Elo**, 20 refit nodes, newest (both recipes current, rule 24) | **17.5** [−6.9, +41.9] | **25.2** (W_b ahead) | **WITHIN FLOOR at n = 2** |
| 1b | ladder Elo, second-newest node | 10.2 [−14.2, +34.6] | **17.6** | **WITHIN FLOOR at n = 2** |
| 1c | **ladder Elo, COMMON-STEP refit**, newest shared node | 6.7 [−20.2, +33.6] | **53.1** | **WITHIN FLOOR at n = 2** |
| 2 | late slope (newest node dropped) — *no bar attaches* | S +1.14 ± 0.44 vs W +0.78 ± 1.33 Elo/M | W_b **+1.82 ± 0.89**, a **1.04 Elo/M** seed spread | DESCRIPTION — the pair's slope contrast is inside the seed spread |
| 3 | **H post-crossing slope** (the FLAT-vs-DECAYING finding) | **0.00133** nats/M [0.00118, 0.00148] | **0.00103** — **77 % of the finding** | **OUTSIDE THE 75M FLOOR** — but see §3.1: the *qualitative form is refuted*, W_b decays too |
| 3b | H_end level (never a finding) | 0.0356 nats | 0.0203 nats | both far inside the 0.074-nat imported floor |
| 4 | **`cond.opp_class_auc.t4_10`** — the registered guard, matched frame | 0.0254 [0.0120, 0.0396] | **0.0014** (draw 1) / **0.0172** (draw 2) | **SPLIT ⇒ NOT CONFIRMED** (rule 21). It was **NOT DETECTED** against the imported floor to begin with — no verdict moves |
| 4b | **`gate.ece.all`** — the pair's DETECTED calibration row | **0.0510** [0.0415, 0.0596] | **0.0587** (draw 1) / **0.0493** (draw 2) | 🚨 **WITHIN FLOOR at n = 2, on BOTH draws — the finding does not survive** |
| 4c | `gate.ece.bot` | 0.0614 [0.0506, 0.0714] | 0.0467 / 0.0452 | **OUTSIDE**, at 1.3× the floor |
| 4d | `gate.reliability.all` — the pair's other DETECTED row | 0.0046 [0.0035, 0.0060] | **0.0050 / 0.0051** | 🚨 **WITHIN FLOOR at n = 2, both draws** |
| 4e | `gate.resolution.all` (was NOT DETECTED) | 0.0043 [−0.0021, 0.0108] | 0.00001 / 0.0043 | WITHIN FLOOR — **and the resolution row is the one that reproduces across seeds** |
| 4f | calibration INTERCEPT, all states | 0.5469 [0.4404, 0.6550] | 0.2044 / 0.3123 | **OUTSIDE** both draws — the *sign* of the pessimism survives, its size does not |
| 5 | **untaught meter** (pp), registry opponent, both config resolutions | **8.31** [+5.69, +11.19], 8/8 teams | **3.69** | **OUTSIDE THE 75M FLOOR at n = 2** — but the same contrast vs W_b is **+4.62 pp on 5/8 teams** |
| 6 | **Metamon `SmallRL` greedy · home**, 100 games | 0.020 [−0.151, +0.111] | **0.060** | **WITHIN FLOOR at n = 2** |
| 6b | **Metamon `SmallRL` greedy · away**, 100 games (`main.anchors`) | 0.020 [−0.117, +0.155] | **0.090** — inside the SOP's own three-seed 0.110 | **WITHIN FLOOR at n = 2** |
| 7 | **Foul Play** @ `--search-time-ms 1000`, 80 games | 0.025 [−0.175, +0.127] at widths 1.249 M vs 1.153 M (**unmatched**) | **0.075** at 1.153 M vs 1.134 M — **ratio 0.984, the first WIDTH-MATCHED pair in the programme** | **WITHIN FLOOR at n = 2**; rule 23 overrides either way |
| 8 | **realized dose** (the pair's hazard H-F) | **1.44×** (lr_median 4.32e-4 vs 3.00e-4) | **1.10×** (3.30e-4 vs 3.00e-4) | the KL controller's annealing is itself draw-level — a new floor on a hazard |
| 9 | the **G7 excursion at ~30M** | — | peaks at **exactly 30,000,000 on all three arms** | DESCRIPTOR — and it is not win-prob-specific |

### The whole read in one paragraph

*Of the four rows the pair read singled out, **two fall inside the floor their own seed replicate
measures and two clear it**. **Strength was never a claim and is now firmly inside a 25-Elo seed
floor** — on the common-step frame the seed pair is **8×** the treatment pair, and **W_b leads arm W
at 15 of 15 shared steps**, which is the exact shape the pair read had described for arm S over arm
W and correctly refused to claim. **The `gate.ece.all` row — the pair's one decisive DETECTED
finding — does not survive: two seeds of one config move ECE by 0.049–0.059 against the treatment's
0.051, and W_b's critic is better calibrated than arm S's auxiliary head.** **The untaught meter
survives** at 8.31 pp against a 3.69 pp floor, though against the other seed the same contrast is
4.62 pp. **The entropy trajectory passes the rule by a 23 % margin and fails as a description**: all
three arms decay, the ordering is a magnitude ordering (S −0.002, W_b −0.023, W −0.096 nats over
71M), and "flat vs decaying" should not be carried forward. Every anchor cell is WITHIN FLOOR, and
the away cell's 0.090 corroborates the anchors SOP's imported 0.110 at 75M for the first time.*

### 🚨 What this read does NOT license

* **It does not establish the floor.** One pair BOUNDS it. The true run-to-run spread may be larger
  or smaller and two runs is n = 2 (rules 19, 22).
* **It does not make a row that CLEARS the floor a family verdict** — it makes it a candidate that
  has survived ONE replicate, and on the three findings a candidate that still carries the
  **1.44× realized-dose gap** the floor pair does not measure.
* **It does not turn the pair's ECE row into a critic-vs-critic comparison.** That row was a CRITIC
  against a DIAGNOSTIC and no floor changes what was compared; the floor simply says the comparison
  was not resolvable at n = 1 either way.
* **It says nothing about the critic objective in general**, at 277M, in a fold, or after a
  distillation.

---

## 1. Provenance — the three arms side by side

| | **arm S** `ai_v13_01_flywheel_shaped` | **arm W** `ai_v13_02_flywheel_winprob` | **W_b** `ai_v13_04_flywheel_winprob_b` |
|---|---|---|---|
| steps | 75,005,952 | 75,005,952 | 75,005,952 |
| `--seed` | 1001 | **1001** | **1002** |
| pin (`pin_history`) | `6eb9c776…` 0 → 75,005,952 | `6eb9c776…` 0 → 75,005,952 | `6eb9c776…` 0 → 75,005,952, **one row** |
| `lineage.role` | `fresh` | `fresh` | `fresh` |
| `config_version` / `arch_signature` | 119 / `gen3_critic_route_wave_v1` | 119 / same | 119 / same |
| `critic` | `shaped` | `winprob` | `winprob` |
| `eval_sentinel_greedy` | true, RECORDED | true, RECORDED | true, RECORDED |
| self-play crossing | 4,128,768 | **4,128,768** | **4,128,768** |
| ladder nodes | 20, 22.0M → 72.0M | 20, 26.0M → 72.0M | 20, **36.0M → 74.0M** |
| `bots8` final | 0.9188 | 0.9100 | **0.9212** |
| wall clock / FPS | 37 h 36 m / 410 | 39 h 23 m / 567 | 39 h 11 m / 547 |
| restarts / crashes | 13 / 1 (teardown SIGTERM) | 13 / 1 (teardown SIGTERM) | **12 / 0** |
| **declared dose** | 4.5776e-8 | 4.5776e-8 | 4.5776e-8 — identical tokens |
| **realized dose** (`python -m main.dose`) | 6.592e-08 (lr_median **4.32e-4**) | 4.578e-08 (lr_median **3.00e-4**) | **5.035e-08** (lr_median **3.30e-4**) |

*Provenance of the rows above: steps, pin, lineage, config version, crossing, node count, `bots8` and the realized dose are read here from each run's own `metadata.json` / `model_config.json` / `snapshot_ladder` / TensorBoard and `python -m main.dose`. Wall clock, FPS and restart counts are quoted from the pair read (arm S, arm W) and the completion entry (W_b) — the `crashes/` directories on disk agree: one entry each on S and W, **none on W_b**, which never created the directory.*

**The argv multiset difference W → W_b, verified here rather than asserted:** 232 tokens on each
side; **only in W** `{1001, ai_v13_02_flywheel_winprob}`, **only in W_b**
`{1002, ai_v13_04_flywheel_winprob_b}`. Nothing else moved. (The completion entry counted
231 → 231; the count here is of `metadata.json`'s `original_command`, and the conclusion — the
difference IS the seed and the run name — is the same.)

### 🚨 The first thing the floor pair measures is a floor on the HAZARD the pair read found

Hazard **H-F** of the pair read was that the realized dose came out **unmatched** even though the
four dose inputs are identical tokens in both argvs: the KL controller annealed arm S to
`lr_median` 4.32e-4 against arm W's 3.00e-4, a **1.44×** realized-dose gap in arm S's favour.

**W_b's `lr_median` is 3.30e-4, a 1.10× gap against arm W's 3.00e-4 — on the SAME argv.** So the
KL controller's annealing is itself a **draw-level quantity**: about a quarter of the S-vs-W
realized-dose gap, on a log scale (ln 1.10 / ln 1.44 = 0.26), is reproduced by changing nothing but
the seed. That does **not** dissolve H-F — 1.44× is still materially larger than 1.10× — but it is
the first number that says how much of a realized-dose gap a matched pair can produce by itself,
and it is a new floor of its own.

---

## 2. STRENGTH — the floor is 25.2 Elo, and the pair's +17.5 sits inside it

**Instrument:** `<run>/snapshot_ladder/ladder.json` (dense, ±10), never `eval/elo` (±29)
[§3.2 rule 1]. Every refit goes through `fit_ladder(run_dir, first_n=N | steps=COMMON,
write=False)` — nothing under `models/` was written.

### 2.1 Rule 24 holds on all three arms

| run | committed computed_at | pairs | `eval_sentinel_edges_dropped` (the recipe stamp) | committed newest | current-code refit | committed − refit |
|---|---|---|---|---|---|---|
| arm S | 2026-09-14 | 190 of 190 | 51 | 2036.6 | 2036.6 | **0.0** |
| arm W | 2026-09-16 | 190 of 190 | 48 | 2019.1 | 2019.1 | **0.0** |
| **W_b** | 2026-09-18 | **190 of 190** | **47** | **2044.3** | **2044.3** | **0.0** |

All three carry the stamp and all three reproduce exactly, so no part of the floor is stale recipe.

### 2.2 The headline rows

| frame | arm S | arm W | **W_b** | **Δ(S − W) — THE FINDING** | **\|W − W_b\| — THE FLOOR** | verdict |
|---|---|---|---|---|---|---|
| **matched COUNT (20 nodes), newest** | 2036.6 @72.0M (se 8.9) | 2019.1 @72.0M (se 8.7) | **2044.3 @74.0M** (se 8.9) | **+17.5** [−6.9, +41.9] | **25.2** (W_b ahead; CI of the floor pair [−49.6, −0.8]) | **WITHIN FLOOR at n = 2** |
| matched COUNT, second-newest | 2042.3 @68.0M | 2032.1 @70.0M | **2049.7 @72.0M** | +10.2 [−14.2, +34.6] | **17.6** | **WITHIN FLOOR at n = 2** |
| **COMMON-STEP refit**, newest shared node | 2032.3 @72.0M (16 shared) | 2025.6 @72.0M | **2064.7 @72.0M** (15 shared with W) | +6.7 [−20.2, +33.6] | **53.1** | **WITHIN FLOOR at n = 2** |

🚨 **The pre-registered prediction (§3.1: strength WITHIN FLOOR, ~70/30) is confirmed on all three
frames, and the margin is not close.** On the frame the registration itself called the honest one —
the common-step refit — **the seed pair is 8× the treatment pair.**

### 2.3 🚨 THE SHAPE THE PAIR READ ASKED TO BE SHOWN IS REPRODUCED BY A SEED

The pair read's most quoted description was that *arm S leads arm W at **every one of the 16 shared
steps**, by +25 to +57 Elo through 26–42M, narrowing to +6.7 at the shared 72.0M node* — a pattern
it correctly refused to claim and named as the reason to run a seed replicate. Here is that
replicate's own common-step table:

| step (M) | arm W | **W_b** | W − W_b |
|---:|---:|---:|---:|
| 36.0 | 1960.1 | 1998.5 | −38.4 |
| 38.0 | 1967.5 | 2013.7 | −46.2 |
| 40.0 | 1978.4 | 2018.6 | −40.2 |
| 42.0 | 1987.0 | 2013.7 | −26.7 |
| 44.0 | 1996.1 | 2041.6 | −45.5 |
| 46.0 | 1998.9 | 2037.2 | −38.3 |
| 48.0 | 1998.1 | 2019.8 | −21.7 |
| 50.0 | 1998.9 | 2029.5 | −30.6 |
| 52.0 | 1996.5 | 2054.8 | −58.3 |
| 54.0 | 1985.1 | 2033.5 | −48.4 |
| 56.0 | 2000.5 | 2048.6 | −48.1 |
| 60.0 | 1995.3 | 2049.8 | −54.5 |
| 64.0 | 1973.3 | 2043.7 | −70.4 |
| 70.0 | 2023.7 | 2046.5 | −22.8 |
| **72.0** | **2011.6** | **2064.7** | **−53.1** |

**W_b leads arm W at 15 of 15 shared steps, by 22 to 70 Elo, and the lead does NOT narrow at the
end.** Two runs of the *identical* configuration produce a larger, more persistent and
more monotone ladder separation than the critic-objective treatment did. 🚨 **A sign that holds at
every shared step is therefore NOT evidence of a treatment** — it is what two seeds of one config
do on this instrument at 75M, and the pair read's own instinct to withhold the claim was right.

### 2.4 The node-by-node curves, all three arms, at matched COUNT

⚠️ **This frame is matched in COUNT and NOT in STEP** — worse here than in the pair, because W_b's
retained pool starts at 36.0M against arm W's 26.0M and arm S's 22.0M, so ordinal *k* is up to 14M
of training later on W_b. The common-step tables above are the honest frames; this one is printed
because the registration asks for matched count and because the shape is worth seeing.

| k | arm S step / Elo | arm W step / Elo | W_b step / Elo | S − W | W − W_b |
|---:|---|---|---|---:|---:|
| 1 | 22.0M / 1982.7 | 26.0M / 1972.8 | 36.0M / 1991.5 | +9.9 | −18.7 |
| 2 | 24.0M / 1967.8 | 28.0M / 1950.6 | 38.0M / 1999.9 | +17.2 | −49.3 |
| 3 | 26.0M / 2005.6 | 30.0M / 1950.0 | 40.0M / 2004.6 | +55.6 | −54.6 |
| 4 | 28.0M / 1991.3 | 32.0M / 1967.6 | 42.0M / 2009.9 | +23.7 | −42.3 |
| 5 | 30.0M / 1988.9 | 34.0M / 1978.9 | 44.0M / 2023.3 | +10.0 | −44.4 |
| 6 | 32.0M / 2016.2 | 36.0M / 1965.2 | 46.0M / 2025.5 | +51.0 | −60.3 |
| 7 | 34.0M / 2025.2 | 38.0M / 1971.9 | 48.0M / 2013.9 | +53.3 | −42.0 |
| 8 | 36.0M / 2020.9 | 40.0M / 1977.1 | 50.0M / 2014.2 | +43.8 | −37.1 |
| 9 | 38.0M / 2038.5 | 42.0M / 1999.2 | 52.0M / 2038.3 | +39.3 | −39.1 |
| 10 | 40.0M / 2020.2 | 44.0M / 2000.4 | 54.0M / 2020.1 | +19.8 | −19.7 |
| 11 | 42.0M / 2040.0 | 46.0M / 1999.2 | 56.0M / 2033.9 | +40.8 | −34.7 |
| 12 | 44.0M / 2017.7 | 48.0M / 2004.8 | 58.0M / 2040.5 | +12.9 | −35.7 |
| 13 | 46.0M / 2025.6 | 50.0M / 2000.4 | 60.0M / 2044.6 | +25.2 | −44.2 |
| 14 | 48.0M / 2022.4 | 52.0M / 2007.3 | 62.0M / 2030.8 | +15.1 | −23.5 |
| 15 | 50.0M / 2019.6 | 54.0M / 1993.9 | 64.0M / 2035.2 | +25.7 | −41.3 |
| 16 | 54.0M / 2039.1 | 56.0M / 2008.2 | 66.0M / 2028.0 | +30.9 | −19.8 |
| 17 | 56.0M / 2041.9 | 60.0M / 2000.7 | 68.0M / 2050.7 | +41.2 | −50.0 |
| 18 | 62.0M / 2043.5 | 64.0M / 1975.2 | 70.0M / 2038.6 | +68.3 | −63.4 |
| 19 | **68.0M / 2042.3** | **70.0M / 2032.1** | **72.0M / 2049.7** | +10.2 | −17.6 |
| 20 | **72.0M / 2036.6** | **72.0M / 2019.1** | **74.0M / 2044.3** | **+17.5** | **−25.2** |

**The three-way COMMON step set is only n = 11** (36 / 38 / 40 / 42 / 44 / 46 / 48 / 50 / 54 / 56 /
72 M) — **below the registered n ≥ 12 report floor**, so it carries no verdict. For description
only, W_b is the highest of the three at 10 of those 11 steps (at the shared 72.0M: S 2031.7,
W 2018.5, W_b 2069.1).

### 2.5 Resolution, late slopes and the within-run spread

| run | se at n = 20 | se(Δ) | CI95(Δ) | late third (newest node dropped) | middle third | adjacent-node \|Δ\| max / median |
|---|---|---|---|---|---|---|
| arm S | 8.9 | 12.59 | ±24.7 | **+1.14 ± 0.44** Elo/M (48–68M), t = 2.62 | +0.02 ± 1.30 (34–44M) | 37.8 / 9.0 |
| arm W | 8.7 | 12.30 | ±24.1 | +0.78 ± 1.33 (52–70M), t = 0.59 | **+3.31 ± 0.83** (38–48M) | 56.9 / 11.3 |
| **W_b** | **8.9** | **12.59** | **±24.7** | **+1.82 ± 0.89** (62–72M), t = 2.04 | +2.48 ± 1.05 (48–58M), t = 2.37 | **24.1 / 8.4** |

**All three arms are still gaining at 75M**, and no arm's late−middle difference is distinguishable
from zero. 🚨 The late slope is one of the rows the registration attached **no bar** to, and this
read shows why that was right: two seeds of one config give **+0.78 ± 1.33** and **+1.82 ± 0.89**,
a 1.04 Elo/M seed spread against arm S's +1.14 — so the "arm S's slope is 2.6 se clear of zero and
arm W's is 0.6 se" contrast of the pair read is **inside the seed floor** and is a description of
two trajectories, not of two objectives.

---

## 3. ENTROPY — the FLAT-vs-DECAYING dichotomy does NOT survive, though the rule's verdict is OUTSIDE

**Instrument:** the TensorBoard events via `main.ops.tb_read`, never the child log's table.
🚨 SB3 logs `train/entropy_loss` as the **negative** mean policy entropy; every H below is its
negation, in nats, positive. All three arms: 740–741 points over 196,608 → 75,005,952, one fresh
run each, and **all three crossed at the identical 4,128,768**, so every post-crossing window is
genuinely matched across all three.

| run | H_start (med-20) | H_peak | **H_end (med-20)** | **post-crossing slope, 4.2M → 75.0M** | total H change over the span | late-third slope |
|---|---|---|---|---|---|---|
| **arm S** | 1.4679 | 1.6724 | **1.0292** | **−0.00003 ± 0.00005** (t = −0.57) | **−0.002 nats** | −0.00211 ± 0.00019 |
| **arm W** | 1.5365 | 1.6885 | **1.0648** | **−0.00136 ± 0.00006** (t = −23.6) | **−0.096 nats** | −0.00169 ± 0.00022 |
| **W_b** | 1.5619 | 1.6836 | **1.0445** | **−0.00033 ± 0.00006** (t = −5.62) | **−0.023 nats** | −0.00119 ± 0.00023 |
| the 0.02 leg | 1.4950 | 1.6725 | 0.7132 | −0.00069 ± 0.00007 | −0.049 nats | −0.00034 ± 0.00024 |
| `ent05` @10M | 1.5487 | 1.6762 | 1.0861 | −0.01415 ± 0.00274 | — | +0.03911 ± 0.00815 |

### 3.1 What the decision rule says, and what the numbers say

| | **finding** \|slope_S − slope_W\| | **floor** \|slope_W − slope_W_b\| | clause (a) | clause (b) | verdict |
|---|---|---|---|---|---|
| post-crossing slope, nats/M | **0.00133** [0.00118, 0.00148] | **0.00103** | ✅ | ✅ | **OUTSIDE THE 75M FLOOR** |
| H_end level, nats | 0.0356 | 0.0203 | ✅ | — | (both far inside the 0.074-nat imported floor; **this row was never a finding**) |

🚨 **The verdict is OUTSIDE and the honest reading is still that the dichotomy broke.** Three
things have to be said together:

1. **The floor is 77 % of the finding.** `|W − W_b| = 0.00103` against `|S − W| = 0.00133`. A margin
   of 23 % supported by **one** replicate is not a margin; rule 19 says a row like this needs
   replication at the level that dominates, and one pair is the minimum, not a sufficiency.
2. **Clause (b) is close to vacuous on this row, and that is a hazard of the rule, not a result.**
   The CI on the slope delta comes from an OLS residual se over ~700 points **on two single
   trajectories** — points that are strongly autocorrelated and are not independent draws. It
   measures how well a line fits one run, not how much the line moves between runs. The floor point
   is the only run-to-run quantity in that row, and it sits at 77 % of the effect.
3. 🚨 **The QUALITATIVE form of the finding is refuted. Arm W does not "decay" while arm S is "flat":
   all three arms decay, and the ordering is a MAGNITUDE ordering — S −0.002 nats, W_b −0.023 nats,
   W −0.096 nats over 71M steps.** W_b's slope is **t = −5.62**, statistically a decay, and one
   quarter of arm W's. Against the *other* seed the treatment contrast nearly vanishes:
   **|slope_S − slope_W_b| = 0.00030**, a *third* of the floor.

**The defensible restatement:** *on this pair of seeds, the win-prob arms lose more entropy over
the post-crossing span than the shaped arm does, but how much more is a seed-level quantity that
moves by 77 % of the treatment effect.* The pair read's sentence — *"arm S holds entropy FLAT for
71M steps; arm W decays"* — should not be carried forward in that form.

### 3.2 Matched-step medians, one point per 5M steps

| step (M) | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 | 65 | 70 | 75 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **arm S** | 1.179 | 1.029 | 1.051 | 1.070 | 1.055 | 1.081 | 1.055 | 1.047 | 1.066 | 1.075 | 1.063 | 1.073 | 1.047 | 1.038 | 1.029 | 1.038 |
| **arm W** | 1.310 | 1.181 | 1.104 | 1.133 | 1.109 | 1.091 | 1.102 | 1.107 | 1.079 | 1.091 | 1.085 | 1.089 | 1.071 | 1.051 | 1.069 | 1.031 |
| **W_b** | 1.307 | 1.101 | 1.049 | 1.060 | 1.053 | 1.065 | 1.095 | 1.081 | 1.066 | 1.083 | 1.074 | 1.057 | 1.052 | 1.048 | 1.051 | 1.036 |
| S − W | −0.131 | **−0.153** | −0.053 | −0.064 | −0.054 | −0.010 | −0.046 | −0.060 | −0.013 | −0.016 | −0.022 | −0.016 | −0.024 | −0.012 | −0.040 | +0.007 |
| **W − W_b (the floor)** | +0.003 | **+0.080** | +0.055 | +0.073 | +0.057 | +0.026 | +0.007 | +0.026 | +0.013 | +0.008 | +0.011 | +0.032 | +0.019 | +0.003 | +0.017 | −0.005 |

**W_b sits below arm W at 15 of the 16 buckets** (the exception is the last, 75M, by 0.005) and
**between arm S and arm W at 11 of 16**. The mean post-crossing gap |W − W_b| = **0.0288** nats
against the finding's |S − W| = **0.0394** — **73 %** of it, the same ratio the slopes gave.
Neither the finding's nor the floor's largest bucket clears the 0.074-nat three-seed replicate
floor imported from 10M, except the 5M one on the finding side (−0.153).

---

## 4. THE CRITIC ROWS — 🚨 the ECE finding does NOT survive its own floor

### 4.1 The frame — and why THIS pair is the like-for-like one the pair could not be

| | **W_b** (ARM) | **arm W** (CONTROL) |
|---|---|---|
| cycle | `step_74000016`, PINNED by `--step` | `step_74000016`, PINNED by `--step` |
| frame | offline full-capture, 12 opponents × 800 games | offline full-capture, 12 opponents × 800 games |
| battles traced | **9,600**, complete, shortfall 0 | **9,600**, complete, shortfall 0 |
| sentinel regime | GREEDY, **recorded** | GREEDY, **recorded** |
| draw share | 0.5 % (50 / 9,600) | 0.4 % — rule 12's 25 % threshold is nowhere near |
| checkpoint | `eval_traces/step_74000016/snapshot.zip`, sha `ef1575f3dd140b24` | sha `7edc9f8e6aa82825` |
| seeds | **20260910 / 20260911** | **20260910 / 20260911** — the SAME two |
| `max|values − win_probs|` | **0.0** | **0.0** |
| anchors reproduced | 150/150 | 150/150 |

🚨 **All three arms are read at 74,000,016 — 1,005,936 steps (1.34 %) short of each one's end, an
exactly matched distance on all three.** No run in this family has an `eval_traces` cycle at its
final 75,005,952 and `eval_trace_gen` refuses a step with no weights of its own.

🚨 **`max|values − win_probs|` is 0.0 on BOTH sides, so every row here — the whole `gate.*`
calibration family included — is a CRITIC-vs-CRITIC contrast.** The pair read's calibration rows
were necessarily **ROW A**: arm S's AUXILIARY win-prob head at `--win-prob-coef 0.05` against arm
W's value function, a DIAGNOSTIC against a CRITIC, because arm S's real critic is a distributional
`E[Z]` on a shaped-return scale where ECE is not defined. **That caveat belongs to the finding and
no floor removes it.** What this read supplies is the run-to-run floor the finding had to be judged
against, and it is measured on the cleanest pair available.

⚖️ Frames were quota-MATCHED before any frame-sensitive row was labelled (rule 17): draw 1 caps
799/392/7 (arm cut from 800/399/8), draw 2 caps 797/395/9 (both sides cut), capture rates
recomputed per subsample over 21 seeded draws.

### 4.2 🚨 THE HEADLINE — `gate.ece.all`

| | arm S (AUX head) | arm W (CRITIC) | **W_b (CRITIC)** |
|---|---|---|---|
| `gate.ece.all`, draw 1 | **0.0178** | **0.0688** | **0.0101** |
| `gate.ece.all`, draw 2 | — | 0.0715 | **0.0222** |
| eval-draw spread | 0.0024 | 0.0028 | **0.0121** |

| | matched-frame Δ | 95 % CI | vs the 0.0245 imported 10M floor |
|---|---|---|---|
| **THE FINDING** \|arm W − arm S\|, draw 1 | **0.0510** | [+0.0415, +0.0596] | **DETECTED** (the pair read's headline) |
| **THE FLOOR** \|arm W − W_b\|, draw 1 | **0.0587** | [−0.0669, −0.0455] | larger than the finding |
| **THE FLOOR** \|arm W − W_b\|, draw 2 | **0.0493** | [−0.0592, −0.0387] | |

> **clause (a)** |S − W| = 0.0510 > |W − W_b| = 0.0587? ❌ (draw 1). On draw 2 the floor is 0.0493
> and clause (a) passes, but **clause (b)** fails — the finding's CI [+0.0415, +0.0596] contains
> 0.0493.
>
> ## 🚨 **VERDICT: WITHIN FLOOR at n = 2, on BOTH independent draws.**

**Two seeds of the identical win-prob configuration differ in offline ECE by as much as, or more
than, the critic-objective treatment did.** W_b's critic reads **0.010–0.022** against arm W's
**0.069–0.072** — and W_b's critic is therefore **better calibrated than arm S's auxiliary
win-prob head** (0.0178), which is the exact opposite of the direction the pair read's row pointed.

**What this retires, and what it does not.**

* ❌ *"Arm W's critic carries ~4× the ECE of arm S's spare head at indistinguishable resolution"* is
  a true statement about **those two checkpoints** and is **not** a statement about the objectives.
  The pair read's own hedge — CRITIC-vs-DIAGNOSTIC, never critic-vs-critic — turns out to have been
  the smaller of the two problems with the row.
* ✅ The **dissociation** the pair read drew survives and is if anything sharper: **RESOLUTION is
  stable and CALIBRATION is not.** `gate.resolution.all` reads 0.0593 (W) / 0.0593 (W_b) / 0.0551
  (S) — a seed floor of **0.00001** on draw 1 — while `gate.ece.all` moves 0.059 between seeds. The
  win-prob critic separates outcomes reproducibly and **calibrates them at a level that is a
  run-level lottery at 75M.**
* 🚨 **`--critic winprob`'s calibration is a RUN-LEVEL quantity with a floor of ≈ 0.05 ECE at 75M.**
  That is the single most consequential number in this read: any future claim about win-prob critic
  calibration — a temperature fix, a `--vf-coef` retune, a distillation gate — must clear it, and
  none of the reads banked so far does.

### 4.3 The full matched-frame table, finding beside floor

Sign: all entries unsigned magnitudes; the finding is `|arm W − arm S|` (the pair read's own
`critic_read` run, `--v-column win_probs`, draw 1) and the floor is `|arm W − W_b|` on each draw.

| row | **finding \|W − S\|** + CI | **floor draw 1** | **floor draw 2** | verdict |
|---|---|---|---|---|
| `gate.ece.all` | **0.0510** [0.0415, 0.0596] | **0.0587** | 0.0493 | 🚨 **WITHIN FLOOR** (both draws) |
| `gate.ece.bot` | 0.0614 [0.0506, 0.0714] | 0.0467 | 0.0452 | **OUTSIDE** (both draws) — but only 1.3× the floor |
| `gate.reliability.all` | 0.0046 [0.0035, 0.0060] | **0.0050** | **0.0051** | 🚨 **WITHIN FLOOR** (both draws) |
| `gate.resolution.all` | 0.0043 [−0.0021, 0.0108] | 0.00001 | 0.0043 | WITHIN FLOOR (it was NOT DETECTED anyway) |
| `gate.resolution.bot` | 0.0028 [−0.0021, 0.0084] | 0.0058 | 0.0046 | WITHIN FLOOR |
| `gate.resolution.pool` | 0.0046 [−0.0043, 0.0134] | 0.0118 | 0.0023 | WITHIN FLOOR |
| `gate.skill.bot` | 0.1391 [0.0576, 0.2209] | **0.1744** | 0.0732 | WITHIN FLOOR (both draws) |
| `gate.skill.all` | 0.0106 [−0.0394, 0.0177] | 0.0041 | 0.0349 | WITHIN FLOOR |
| **`cond.opp_class_auc.t4_10`** (the registered guard) | 0.0254 [0.0120, 0.0396] | **0.0014** | 0.0172 | **SPLIT** — OUTSIDE on draw 1, WITHIN on draw 2 ⇒ **NOT CONFIRMED** (rule 21). It was **NOT DETECTED** against the imported floor to begin with, so no verdict moves |
| `cond.opp_class_auc.t1_3` | 0.0151 [−0.0327, +0.0011] | 0.0093 | 0.0137 | WITHIN FLOOR — and its own eval-draw spread already disqualified it |
| `cond.opp_class_auc.t1` | 0.0109 [−0.0073, +0.0300] | 0.0031 | 0.0124 | turn 1 reads ≈ 0.50 on every arm and draw; nothing to read |
| calibration SLOPE, all | 0.0041 [−0.0896, 0.0906] | 0.0668 | 0.0453 | WITHIN FLOOR (was inside the imported floor too) |
| calibration INTERCEPT, all | 0.5469 [0.4404, 0.6550] | 0.2044 | 0.3123 | **OUTSIDE** both draws — the level shift is the one calibration row that survives, at 1.8–2.7× its floor |

**The one calibration statement that survives the floor:** *arm S's auxiliary head is calibrated in
the large (intercept +0.03) where both win-prob critics are systematically pessimistic (+0.58 on
arm W, +0.38 / +0.27 on W_b), at a dispersion (calibration slope 1.28–1.39) the three share.* The
size of that pessimism, and the ECE it produces, is a seed-level quantity; its **sign** is not.

### 4.4 The eval-draw spread — and rule 19 confirmed at 75M for the first time

Two independent draws of the same checkpoint bound a floor at the eval-draw level; they never give
it a CI (rule 19).

| row | arm W eval-draw spread | **W_b** eval-draw spread | **the RUN-level floor measured here** | ratio run/draw |
|---|---|---|---|---|
| `gate.ece.all` | 0.0028 | 0.0121 | **0.0493–0.0587** | **4–21×** |
| `gate.reliability.all` | 0.0005 | 0.0005 | 0.0050 | **10×** |
| `gate.resolution.all` | 0.0014 | 0.0029 | 0.00001–0.0043 | ≈ 1× |
| `cond.opp_class_auc.t4_10` | 0.0165 | **0.0013** | 0.0014–0.0172 | ≈ 1× |
| `gate.ece.bot` | 0.0008 | 0.0008 | 0.0452–0.0467 | **56×** |

🚨 **Rule 19 says to identify which variance component dominates a row and replicate at THAT level.
For the calibration family the answer at 75M is now measured: the RUN component is 4–56× the
eval-draw component, so more draws of one checkpoint buy nothing there.** For the resolution family
and the registered conditioning guard the two components are the same size, which is why those rows
were stable across draws in the pair read and are stable across seeds here.

---

## 5. THE UNTAUGHT METER — the floor is 3.69 pp, the finding is 8.31 pp, and it SURVIVES

`python -m main.untaught_meter` — the win rate of a checkpoint **piloting** a fixed 8-team slice it
never trained on, against ONE fixed opponent, cluster-bootstrapped over TEAMS. The opponent is the
registry name `untaught_meter_opponent` (= `ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip`
@ 24,000,000). 200 games/team, seed 0, **concurrency 1**, **W_b and arm W measured in ONE
invocation** so they saw the identical 8 teams and the identical games. **3,200 battles per config,
0 timeouts.**

### Levels — identical under BOTH config resolutions, on both arms

| ref | registry `untaught_meter_config` | `--config auto` | wins / finished |
|---|---:|---:|---:|
| arm S (banked 2026-09-16) | **54.50 pp** | 54.50 pp | 872 / 1600 |
| arm W | **46.19 pp** | 46.19 pp | 739 / 1600 |
| **W_b** | **49.88 pp** | **49.88 pp** | **798 / 1600** |

✅ **Arm W reproduces its banked 2026-09-16 level EXACTLY — Δ = 0.00 pp, 739/1600 both times.** That
is the second independent confirmation that the meter is deterministic at seed 0 / concurrency 1,
and it is what licenses joining arm S's level in from the pair read's file.

### The paired team-clustered contrasts (rule 10: the team is the unit; 20,000 draws, one shared index set)

| contrast | Δ (pp) | CI95 (pp) | teams favouring the first |
|---|---:|---|---:|
| **arm S − arm W — THE FINDING** | **+8.31** | [+5.69, +11.19] | **8 of 8** |
| **arm W − W_b — THE 75M RUN-LEVEL FLOOR** | **−3.69** | [−8.19, +0.81] | 2 of 8 |
| arm S − W_b — the treatment against the OTHER seed | **+4.62** | [+0.69, +8.56] | **5 of 8** |

| team | `U_61590463` | `U_92832108` | `U_ce35b736` | `U_9909f2e9` | `U_9d5f8458` | `U_f7ba5702` | `U_90b94599` | `U_dbf81d8e` |
|---|---|---|---|---|---|---|---|---|
| arm S | 56.50 | 49.50 | 56.00 | 54.50 | 55.50 | 55.00 | 49.50 | 59.50 |
| arm W | 48.50 | 39.00 | 49.50 | 38.00 | 47.00 | 48.00 | 48.00 | 51.50 |
| **W_b** | **58.00** | **53.00** | **42.50** | **48.00** | **46.00** | **48.00** | **51.50** | **52.00** |

### Verdict — **OUTSIDE THE 75M FLOOR**, with the caveat printed in the same breath

| | value |
|---|---|
| finding \|S − W\| | **8.31 pp**, CI [+5.69, +11.19] |
| floor \|W − W_b\| | **3.69 pp** |
| clause (a) 8.31 > 3.69 | ✅ |
| clause (b) CI excludes 3.69 | ✅ |
| **verdict** | **OUTSIDE THE 75M FLOOR at n = 2** |

🚨 **And the caveat that must travel with it: against the OTHER seed the same treatment contrast is
+4.62 pp, on 5 of 8 teams — 1.25× the floor rather than 2.25×.** The pair happened to draw the
win-prob seed that scores lowest on this meter. The defensible statement is *"the shaped arm is
ahead of both win-prob seeds on the untaught meter, by between 4.6 and 8.3 pp, against a
seed-to-seed floor of 3.7 pp"* — a **run-level CANDIDATE that has now survived one replicate**, and
still a **DESCRIPTOR and not an endpoint** (registration §8.5: the meter's axes and floors are
established at ~1M-step FOLD depths and none of these arms is a fold).

**This is also the meter's first RUN-LEVEL floor at any depth.** Every floor it had before was a
fold floor (1.19 pp END-depth replicate, 1.66 pp frozen-dose, 4.27 pp controller-live, 1.00 pp G5
continuation), quoted rather than applied. **3.69 pp at 75M sits between the frozen-dose and
controller-live fold floors** — so the fold floors were not badly wrong in magnitude, which is worth
knowing on its own.

---

## 6. EXTERNAL ANCHORS — every cell WITHIN FLOOR, and the away floor lands on the SOP's own number

### 6.1 Metamon `SmallRL`, greedy-vs-greedy, both team sets

Harness: the pair read's own `run_cell.sh`, **verbatim**, with the same team seeds (`BASE_OURS
20260914`, `BASE_THEIRS 20270914`) on our own pinned Showdown at **:9450** for the HOME cell; the
AWAY cell through the tool of record, `python -m main.anchors --opponent metamon:SmallRL --regime
greedy --teamset away --games 100`, which starts and stops its own server (:9500) and verifies the
regime per decision. Metamon `@0a00a759`, `SmallRL` ckpt 40 (13.9M), `VanillaAttention`, CPU,
`OMP_NUM_THREADS=1`. Our side is each arm's `final_model.zip` @ 75,005,952.

| cell | arm S | arm W | **W_b** | **\|S − W\| (finding)** | **\|W − W_b\| (floor)** | verdict |
|---|---|---|---|---|---|---|
| **greedy · home**, 100 games | 0.630 [0.532, 0.718] | 0.650 [0.553, 0.736] | **0.590** [0.492, 0.681] | 0.020 [−0.151, +0.111] | **0.060** [−0.074, +0.191] | **WITHIN FLOOR at n = 2** |
| **greedy · away**, 100 games (`main.anchors`) | 0.520 [0.423, 0.615] | 0.500 [0.404, 0.596] | **0.590** [0.492, 0.681] | 0.020 [−0.117, +0.155] | **0.090** [−0.223, +0.047] | **WITHIN FLOOR at n = 2** |

✅ **The away floor lands INSIDE the SOP's own three-seed run-level floor for this exact cell —
0.090 against 0.110** [`EXTERNAL_ANCHORS_SOP.md` amendment 2026-09-16, measured on `ctrl10M` /
`_b` / `_c` at 10M]. The SOP warned that the budget effect (~0.38 at 10M → ~0.64 at 75M) dwarfs
every lever the ladder measured, so a 10M floor imported to 75M was an import rather than a match.
**It now has one 75M corroboration, and the imported number was not wrong.** (The eval-draw floor on
the same cell is 0.020; this pair's 0.090 is 4.5× it, which is rule 19's point — the variance on
this row is RUN-level.)

**Per half-cell (role balance), W_b:** greedy·home **0.72** (Metamon challenges) / **0.46** (we
challenge); greedy·away **0.54** (peer challenges) / **0.64** (we challenge).

🚨 **W_b's home role split is 26 pp with a Newcombe CI of [−0.428, −0.067] that EXCLUDES zero** —
against arm W's 0.66 / 0.64 and arm S's 0.66 / 0.60. Role balance inside a cell is exactly the
design that stops this from becoming a level error, and the size of the split here is a standing
reason the 100-game cell's run-level floor is ~0.09–0.11 rather than ~0.02. It is reported as a
descriptor; at n = 50 per half nothing else may be taken from it.

**Regime verified per decision:** our `stochastic` kwarg `[false]` and temperature 0.0 in every
half-cell; Metamon's `argmax_match_rate` **1.0000** in the half-cell whose record survived, and
**1.0000** in both away halves. 1 tie in 100 home games, 0 in 100 away. `team_source_asymmetry`
false; `distinct_our_teams` 19 of 20 on the away set; `model_loader` `bare`, `model_rung`
`explicit_zip`, Showdown pin `e0551883f`.

### 6.2 Foul Play at `--search-time-ms 1000` — and the FIRST width-matched pair in the campaign

Foul Play `@6c467c08` + poke-engine `0.0.48` (`--features poke-engine/gen3 --no-default-features`),
`--search-time-ms 1000 --search-parallelism 1 --search-threads 1`, our side pinning **the same 8
pool teams** arm S and arm W used (same sorted export, same stride), on our pinned Showdown at
**:9417**. 8 sessions × 10 games.

| campaign | our win rate | Wilson 95 % | **realized FP visits/decision** | session range | our think ms |
|---|---|---|---:|---|---:|
| the 0.02 leg (de-risk, 2026-09-14) | 0.388 | [0.288, 0.497] | 1.400 M | 1.21–1.53 M | ~37 |
| arm S (2026-09-14) | 0.450 | [0.346, 0.559] | 1.249 M | 1.05–1.65 M | 481 |
| arm W (2026-09-16) | 0.475 | [0.369, 0.583] | 1.153 M | 0.86–1.33 M | 474 |
| **W_b (2026-09-18)** | **0.400** | **[0.300, 0.510]** | **1.134 M** | **0.71–1.56 M** | **459** |

| | value |
|---|---|
| finding \|S − W\| | 0.025 [−0.175, +0.127], widths 1.249 M vs 1.153 M — **ratio 0.923, NOT matched** |
| **floor \|W − W_b\|** | **0.075** [−0.077, +0.223], widths 1.153 M vs 1.134 M — **ratio 0.984, MATCHED** |
| verdict | **WITHIN FLOOR at n = 2** |

🚨 **Two things this row buys, and one it destroys.**

1. **It is the first WIDTH-MATCHED Foul Play pair in the programme** (ratio 0.984, inside the
   0.95–1.05 band that `pair_foulplay.py` has always tested and nothing has ever passed). At matched
   width, **the seed alone moves the cell by 7.5 pp** — three times the treatment difference the
   pair read reported at unmatched width.
2. 🚨 **It destroys the "win rates order exactly inversely to realized width" observation.** The pair
   read noted 1.400 M → 0.388, 1.249 M → 0.450, 1.153 M → 0.475 and called the monotone inversion
   *"exactly what rule 23 predicts a width meter does"*. W_b is the fourth point: **1.134 M →
   0.400**, the narrowest search and the second-lowest win rate. The monotone relation was a
   three-point coincidence. **Rule 23 itself is untouched** — a meter whose value depends on the
   box's throughput still needs a contemporaneous control or width matching — but the specific
   inverse-ordering evidence offered for it here should not be carried forward.
3. Nothing about a model may be taken from any of these four numbers. All four Wilson intervals
   overlap heavily; **all four 75M policies sit BELOW Foul Play at this budget** and the external
   anchors continue to bracket us (Metamon's search-free `SmallRL` below, Foul Play's search above).

---

## 7. THE G7 EXCURSION AT ~30M — same step, same magnitude, and it is not even win-prob-specific

Registration §8.4b made `rollout/ep_len_mean` and the stall/timeout fraction PRIMARY on the win-prob
arm by its own `[CRITIC]` banner's instruction (*a `[0,1]` critic cannot express "a timeout is worse
than a loss"*). The completion entry banked that **the excursion reproduced across seeds**. This
section reads the SERIES underneath that claim.

**Instrument:** `eval/mean_ep_len_vs_bots` from the TensorBoard events. AMENDMENT 6: each arm is
referenced to its OWN first two eval cycles, frozen, bar 1.25×, and a breach is a REPORT, never a
kill. ✅ **The references computed here reproduce the banked ones to the third decimal — arm W
25.327, W_b 22.330 — which is the check that this is the series the tool quoted.**

| | arm S | arm W | **W_b** |
|---|---|---|---|
| frozen reference (mean of cycles at 2.0M / 4.0M), turns | 23.323 | **25.327** | **22.330** |
| run's WORST ratio, anywhere | 1.070 @ **30.0M** | 1.131 @ 6.0M | **1.160 @ 30.0M** |
| % of the 1.25 bar at that worst | 85.6 % | 90.5 % | 92.8 % |

### The ~30M excursion, read inside a 24M–34M window pinned before either arm was looked at

| | arm S | arm W | **W_b** |
|---|---|---|---|
| rises from | 28.0M, 22.82 turns | 26.0M, 26.51 turns | 28.0M, 25.20 turns |
| **peak step** | **30,000,000** | **30,000,000** | **30,000,000** |
| peak turns / ratio | 24.95 / **1.070** | 28.14 / **1.111** | 25.90 / **1.160** |
| recovered at | 32.0M, 22.38 turns (0.959) | 34.0M, 25.03 turns (0.988) | 32.0M, 23.61 turns (1.057) |
| amplitude over the arm's own median turns | **+2.18** | **+1.79** | **+2.27** |

**The raw turn series, 24M → 34M** (reference-free, which is the only cross-arm-comparable form):

| step (M) | 24 | 26 | 28 | **30** | 32 | 34 |
|---|---|---|---|---|---|---|
| arm S | 22.77 | 23.01 | 22.82 | **24.95** | 22.38 | 21.89 |
| arm W | 26.58 | 26.51 | 27.52 | **28.14** | 26.69 | 25.03 |
| **W_b** | 23.68 | 25.38 | 25.20 | **25.90** | 23.61 | 23.59 |

**The descriptor, stated exactly:** *the excursion peaks at **exactly 30,000,000 on all three
arms**, rises over the preceding 2–4M, and is back inside each arm's own median by 32–34M. Its
magnitude in raw turns is the same on all three to within half a turn — +1.8 to +2.3 turns over the
arm's own median — and no arm came within 7 % of the 1.25 bar.*

🚨 **The completion entry called it "a property of the 75M win-prob trajectory, not of either
seed". The third arm extends that: arm S — a different critic objective, a different reward
composition — does the same thing at the same step with the same amplitude.** It is a property of
this recipe's curriculum at ~30M, not of the win-prob critic. What it is *not* is a pathology: no
arm breached G7, the stall half-peaks are 0.0237 (S) / 0.0145 (W), and rule 12's 25 % inconclusive
threshold is nowhere near.

⚠️ **The RATIO is comparable only WITHIN an arm and the raw turns only across arms.** The frozen
reference moved **3.0 turns between two seeds of one config** (25.327 → 22.330), which is the third
measurement in this campaign that the G7 reference is a draw-level quantity. W_b's 1.160 reads as a
larger excursion than arm W's 1.111 **entirely because its reference is 3 turns lower**; in turns
the two peaks are 28.14 and 25.90 and the amplitudes over each arm's own median are 1.79 and 2.27.

### `rollout/ep_len_mean` — the pair's operational row, with its floor

| row | arm S | arm W | **W_b** | \|S − W\| | **\|W − W_b\| (floor)** |
|---|---|---|---|---|---|
| post-crossing mean (4.13M → 75.0M) | 40.48 | 44.53 | **43.75** | 4.05 turns | **0.78 turns** |
| post-crossing max | 53.32 | 63.92 | **60.39** | 10.60 | **3.53** |
| median-20 at the end | 43.31 | 45.43 | **45.47** | 2.12 | **0.04** |
| last value (a single point, not a measurement) | 46.51 | 40.90 | 49.58 | — | — |

**The pair read's operational observation survives:** arm W's rollout episodes ran ~4 turns longer
than arm S's, and **W_b reproduces arm W to within 0.8 turns** — a floor about a fifth of the
difference. ⚠️ `rollout/ep_len_mean` is ~90 % self-play on all three arms and is never comparable to
the bot-eval series; the arm-to-arm comparison above is rollout-vs-rollout, which is matched, and no
bar attaches to it.

---

## 8. THE RE-READ — the three pre-designated findings and strength, against THIS floor

The registration named three rows that could not be endpoints but were reported as findings, plus
the primary endpoint. Here is each one, restated against its own 75M run-level floor, with the
honest caveats attached rather than appended.

### 8.1 STRENGTH (+17.5 Elo) — **WITHIN FLOOR at n = 2**, on all three frames

| frame | \|S − W\| | \|W − W_b\| | ratio |
|---|---|---|---|
| matched count, newest | 17.5 | **25.2** | 0.69 |
| matched count, second-newest | 10.2 | **17.6** | 0.58 |
| common-step, newest shared | 6.7 | **53.1** | 0.13 |

**The pair read said NOT DETECTED and was right for a reason it could not yet demonstrate.** What is
new is the magnitude of the thing it could not see: on the frame the registration itself called the
honest one, the seed pair moves **eight times** what the treatment pair did, in the opposite
direction. 🚨 **And the SHAPE argument collapses with it** — "arm S leads at all 16 shared steps" was
the pattern the pair read carried forward as a reason to replicate; the replicate produces "W_b
leads arm W at all 15 shared steps" with a *larger* gap that *does not narrow*. A monotone sign
across a run's shared nodes is **not** evidence of a treatment at this depth.

⚠️ **The imported 45.0-Elo control floor is not replaced by this number.** 45.0 was the MAX pairwise
|Δ| over three same-argv **10M four-node** controls; 25.2 is one pairwise |Δ| between two same-argv
**75M twenty-node** arms. They are different depths, different node counts, and — decisively —
**one pair is not a MAX over replicates** (rule 3: a floor is the MAX over the replicates in hand).
Both are reported; neither supersedes the other, and the smallest claimable |Δ| on the pair's own
fit stays **69.4**.

### 8.2 ENTROPY, "flat vs decaying" — **OUTSIDE by the rule, REFUTED as a description**

| | value |
|---|---|
| finding \|slope_S − slope_W\| | **0.00133** nats/M, CI [0.00118, 0.00148] |
| floor \|slope_W − slope_W_b\| | **0.00103** nats/M — **77 %** of the finding |
| clause (a) / clause (b) | ✅ / ✅ ⇒ **OUTSIDE THE 75M FLOOR at n = 2** |
| the contrast against the OTHER seed, \|slope_S − slope_W_b\| | **0.00030** — *a third of the floor* |

**Both halves have to be said.** The rule's verdict is OUTSIDE. But the margin is 23 % on one
replicate; clause (b)'s CI is an OLS residual se over ~700 autocorrelated points on two single
trajectories, so it measures line-fit uncertainty and not run-to-run movement; and the sentence the
pair read wrote — *"arm S holds entropy FLAT for 71M steps; arm W decays"* — is **not true of the
config**, only of that seed. **All three arms decay. The honest statement is a magnitude ordering:
S −0.002 nats, W_b −0.023 nats, arm W −0.096 nats over the 71M post-crossing span**, with W_b's
decay statistically real (t = −5.62) and a quarter of arm W's.

### 8.3 THE UNTAUGHT METER (+8.31 pp) — **OUTSIDE THE 75M FLOOR at n = 2**

| | value |
|---|---|
| finding \|S − W\| | **8.31 pp** [+5.69, +11.19], **8 of 8** teams |
| floor \|W − W_b\| | **3.69 pp**, 6 of 8 teams favour W_b |
| clause (a) / clause (b) | ✅ / ✅ ⇒ **OUTSIDE** |
| the contrast against the OTHER seed | **+4.62 pp** [+0.69, +8.56], **5 of 8** teams |

**The strongest surviving row in the pair, and it survives with its magnitude halved against the
other seed.** *The shaped arm is ahead of BOTH win-prob seeds on this meter, by 4.6 to 8.3 pp,
against a seed-to-seed floor of 3.7 pp.* It remains a **DESCRIPTOR, not an endpoint** (§8.5), and
the **1.44× realized-dose gap** still confounds arm S: this read measures a floor on a pair whose
own realized-dose gap is 1.10×, so clearing the floor does not clear the dose.

### 8.4 CALIBRATION — ECE 0.069 vs 0.018 — 🚨 **WITHIN FLOOR at n = 2. Overturned.**

| | value |
|---|---|
| finding \|W − S\|, `gate.ece.all`, matched frame | **0.0510** [0.0415, 0.0596] |
| floor \|W − W_b\|, draw 1 / draw 2 | **0.0587** / **0.0493** |
| clause (a) draw 1 | ❌ (the floor is LARGER than the finding) |
| clause (b) draw 2 | ❌ (the CI contains the floor) |
| **verdict** | 🚨 **WITHIN FLOOR at n = 2, on both draws** |

**The levels make the point better than the arithmetic:** arm S's auxiliary head 0.0178, arm W's
critic 0.0688 / 0.0715, **W_b's critic 0.0101 / 0.0222**. The win-prob seed the pair happened to
draw is the badly-calibrated one; the other seed's critic is better calibrated than the shaped arm's
diagnostic head. `gate.reliability.all` falls the same way (finding 0.0046, floor 0.0050 / 0.0051).

**What replaces the retired claim.** ✅ The **resolution/calibration DISSOCIATION survives and
sharpens**: `gate.resolution.all` is 0.0593 / 0.0593 / 0.0551 across W / W_b / S — a seed floor of
0.00001 on draw 1 — while ECE moves 0.059 between the two seeds. 🚨 **`--critic winprob`'s
calibration is a RUN-LEVEL quantity with a 75M floor of ≈ 0.05 ECE**, and any future claim about
win-prob critic calibration must clear it.

---

## 9. HAZARDS — every one a finding

| # | hazard | why it matters, and what was done |
|---|---|---|
| **F-A** | 🚨 **THE PAIR'S ONE DECISIVE DETECTED ROW IS INSIDE ITS OWN SEED FLOOR.** `gate.ece.all` moves 0.049–0.059 between two seeds of one config against a treatment difference of 0.051 — and the direction flips: W_b's critic (0.0101 / 0.0222) is better calibrated than arm S's auxiliary head (0.0178). | The row is restated as WITHIN FLOOR at n = 2 on both draws, the dissociation it supported is re-derived from `gate.resolution.*` (which DOES reproduce across seeds), and the standing consequence is recorded: **win-prob critic calibration has a ≈ 0.05 ECE run-level floor at 75M** and no banked read clears it. §4.2, §8.4 |
| **F-B** | 🚨 **A MONOTONE SIGN ACROSS A RUN'S SHARED LADDER NODES IS NOT EVIDENCE OF A TREATMENT.** The pair read described arm S leading arm W at 16 of 16 shared steps as the pattern worth replicating. **W_b leads arm W at 15 of 15 shared steps, by a larger margin that does not narrow.** | Reported as the central strength result. The shape argument is retired at this depth; the 25.2 / 53.1 Elo floors are reported beside the imported 45.0 rather than replacing it (one pair is not a MAX over replicates — rule 3). §2.3, §8.1 |
| **F-C** | 🚨 **"FLAT vs DECAYING" IS A SEED DESCRIPTION, NOT A CONFIG ONE.** W_b decays at −3.3e-4 nats/M (t = −5.62), a quarter of arm W's rate; the floor is 77 % of the finding and the contrast against W_b is a third of the floor. | The rule's verdict (OUTSIDE) is reported *with* the refutation of the qualitative form, and clause (b)'s near-vacuity on an OLS fit over autocorrelated points is stated rather than hidden. §3.1, §8.2 |
| **F-D** | 🚨 **THE METAMON `RecursionError` IS NOT ABOUT SAMPLING — IT IS ABOUT WHO CHALLENGES.** Hazard H-H of the pair read recorded three occurrences, all in *ours greedy / Metamon SAMPLING / Metamon challenging*, and concluded it was a property of that configuration. **This read's occurrence is in a GREEDY-vs-GREEDY cell** — the fourth, with the same 988-frame signature out of `metamon_to_amago.py::step` after its last game. The common factor across all four is **Metamon CHALLENGING**, not Metamon sampling. | Our side's 50 results are complete and feed the win rate; that half-cell's positional join is UNRELIABLE (`sides_disagree` 22, `metamon_row_missing` 2) against `sides_disagree` = 1 in the other half. The characterisation in H-H should be narrowed to the challenge role. §6.1 |
| **F-E** | 🚨 **THE FOUL PLAY "WIN RATES ORDER EXACTLY INVERSELY TO WIDTH" OBSERVATION IS BROKEN BY THE FOURTH POINT.** 1.400 M → 0.388, 1.249 M → 0.450, 1.153 M → 0.475, and now **1.134 M → 0.400**. | Rule 23 itself is untouched — a width meter still needs width matching — but the specific three-point inverse ordering must not be carried forward as evidence for it. **The compensating gain: W and W_b are width-matched to 1.6 % (ratio 0.984), the first such pair in the programme, and at matched width the SEED moves the cell 7.5 pp.** §6.2 |
| **F-F** | 🚨 **A FOUL PLAY SESSION DIED SILENTLY AND WOULD HAVE COST A TEAM.** Session 5 ran its full 2400 s timeout with Foul Play logged in, challenging, and **zero battles initialised** (`Initialized battle` count = 0) — the H10 class. Left alone the campaign would have been **70 games over 7 of the 8 pinned teams**, not team-matched with arm S's and arm W's. | Detected by counting rows per team file, **re-run to completion** at a lower box load (the failed attempt's logs are kept as `fp_s5.FAILED_attempt1.log`), and the campaign is the full 80 games over all 8 teams. ⚠️ The retried session therefore ran at a materially lighter load than the other seven — its realized width is 1.314 M against the campaign's 1.134 M mean — which is recorded rather than smoothed. |
| **F-G** | ⚠️ **THE THREE ARMS' POOL SENTINELS ARE DIFFERENT CHECKPOINTS** — W_b's offline draws face its own 36.0M / 54.0M / 72.0M snapshots, arm W's face 26.0M / 46.0M / 72.0M, arm S's 22.0M / 42.0M / 72.0M — because the retained pools differ. | The instrument's design (a sentinel is the run's own snapshot), so the frames are matched in KIND but not in the identity of three of twelve opponents. Recorded because every conditioning row is a statistic OF the traced frame and the `pool` stratum is the one it touches. This is the pair read's H-I, unchanged. |
| **F-H** | ⚠️ **W_b's LADDER POOL STARTS 10M LATER THAN ARM W's** (36.0M → 74.0M against 26.0M → 72.0M), so the matched-COUNT table compares ordinal *k* at steps up to 14M apart — worse than the pair's own H-J. | The COMMON-STEP refits (15 shared steps for W/W_b, 16 for S/W) are the honest frames and are the ones the verdicts use; the matched-count table is printed beside them and labelled. The three-way common set is **n = 11, below the registered n ≥ 12 report floor**, and carries no verdict. §2.3, §2.4 |
| **F-I** | ⚠️ **W_b's home anchor cell has a 26 pp ROLE SPLIT whose CI excludes zero** (0.72 when Metamon challenges, 0.46 when we do; Newcombe [−0.428, −0.067]), against 0.66/0.64 on arm W and 0.66/0.60 on arm S. | Role balance inside the cell is what stops this becoming a level error, and it is a standing reason this cell's run-level floor is ~0.09–0.11 rather than ~0.02. Reported as a descriptor; at n = 50 per half nothing else may be taken from it. §6.1 |
| **F-J** | ⚠️ **Six `Decision context is missing at turn 1` tracebacks fired on OUR side in the away cell**, and 2 of 100 away games hit the 250-turn forfeit. | The same class the pair read logged (its M4). No game was lost from the record (100/100 complete, `status: OK`), and 2 % is far under rule 12's 25 % INCONCLUSIVE threshold. Recorded, not investigated here. |
| **F-K** | ⚠️ **The live GPU arm `ai_v13_05_exploit_big5starmie` COMPLETED on its own at 15:10 PT** during this read (final aggregate 99.8 %, `final_model.zip` written 15:08). **It was never touched.** | Recorded because this read ran CPU-only beside it and because a completion is an orchestrator-visible event. The GPU was idle from 15:10; nothing in this read used it. |
| **(context)** | The box carried the live arm plus four concurrent CPU jobs; load average ran **13–43** across the session, peaking at 43 while the trace generation, the untaught meter, the Metamon cells and the Foul Play campaign overlapped. | Every timing-sensitive row carries its realized cost: trace draws 118.7 min and 86.0 min, Foul Play's realized width per session, and the Metamon cells' wall time. Only Foul Play's number is load-sensitive by construction (rule 23). |

### What was NOT done, and why

* **No third seed.** Two runs is n = 2 and the floor is a bound, not an estimate. A third arm is
  ~35 GPU-h and is not proposed here.
* **The Metamon MIXED cell was not run.** Anchors-SOP rule 2: the recurring read is greedy-vs-greedy,
  verified per decision; "temperature 1.0" is not a fixed yardstick across policies, and the mixed
  cell is the one whose Metamon half has now died four times.
* **`--v-column values` was not run.** On two `--critic winprob` arms it reads the same tensor
  (`max|values − win_probs|` = 0.0 on both), so a second invocation would return identical rows.
* **The mirror battery was not run** (rule 25 retired the 100-pair cells; a ≥ 400-pair leaf read is a
  separate job).
* **No code under `src/` was changed**, and nothing was written under `models/`.
* **`main.ops.quota_match`'s `--v-column` defect (the pair read's H-L) was not fixed** — it remains a
  tech-debt row. It does not touch this read: both columns here are `win_probs` by construction.

---

## 10. What is in this directory

| path | what |
|---|---|
| `PREDICTION.md` | the pre-registration, committed at `12836d5f` before any W_b number here existed |
| `README.md` | this note |
| `scripts/floor_strength_read.py` | the ladder read for all three arms — committed-vs-refit (rule 24), the resolution curve, the newest-node-dropped slopes, the adjacent-node spread, the COMMON-STEP refits, the three-way common set, the node-by-node table, and the floor arithmetic. Every helper VERBATIM from the pair read's `pair_strength_read.py` |
| `scripts/floor_entropy_read.py` | `train/entropy_loss` from the TB events on all three arms plus the two comparators, sign-corrected to H, median-20, each arm's crossing, the slopes, the matched-step buckets, and the floor verdict. Helpers VERBATIM from `pair_entropy_read.py` |
| `scripts/floor_critic_levels.py` | the critic rows on both arms and both draws through the same library calls `critic_read` makes, plus the re-read of the pair's own banked ROW A deltas against the floor measured here |
| `scripts/floor_untaught_delta.py` | the paired team-clustered contrasts off the meter's own per-team rows under both config resolutions, with arm W's reproduction check against the pair read's file |
| `scripts/floor_metamon_cells.py` | W_b's HOME half-cells, Wilson + Newcombe, against arm W / arm S / the 0.02 leg, each in its own regime |
| `scripts/floor_away_anchor.py` | the AWAY cell through `python -m main.anchors`, with the SOP's own three-seed and eval-draw floors printed beside the measured one |
| `scripts/floor_foulplay.py` | the Foul Play campaign summary and the rule-23 width comparison across all four campaigns |
| `scripts/floor_eplen_read.py` | the G7 excursion descriptor — `eval/mean_ep_len_vs_bots` on all three arms with AMENDMENT 6's frozen reference REPRODUCED to the banked third decimal, plus `rollout/ep_len_mean` |
| `scripts/harness/` | the run scripts as executed: the W_b draw generator, the untaught driver, the Metamon cell runner (the pair read's `run_cell.sh`, verbatim) and W_b's plan, the away-cell driver, and the `critic_read` floor invocation |
| `out/floor_strength_read.json` | every ladder number, all three arms |
| `out/floor_entropy_read.json` | every entropy number, all five runs, plus the matched-step buckets and the floor verdict |
| `out/floor_critic_levels.json` | W_b and arm W levels on both draws, the frames, the floor deltas, the eval-draw spread, and the re-read |
| `out/floor_untaught_delta.json`, `out/untaught/` | the meter under both config resolutions and every paired contrast |
| `out/floor_eplen_read.json` | the ep_len series, the frozen references, the 24M–34M excursion on all three arms |
| `out/critic_read_floor/` | `main.ops.critic_read`'s own reports for W_b (ARM) vs arm W (CONTROL), both draws |
| `out/metamon/` | `armWb_cells.json`, the merged `games.jsonl`, the harness `summary.json` with the per-decision regime instruments |
| `out/away/` | the `main.anchors` away cell (`summary.json` + `games.jsonl`) and `floor_away_anchor.json` |
| `out/foul_play/` | the merged `games.jsonl`, `summary.json`, the session table, and the failed session-5 logs |
| `out/traces_manifests/` | the two W_b draw manifests and the generator log |

---

## 11. Ledger paragraph — ready to append (nothing in `ledger.md`, `UNDERSTANDING.md` or the registration was edited from here)

> ### 2026-09-18 · MEASUREMENT (MAJOR) · **THE 75M RUN-LEVEL FLOOR IS MEASURED**, and it swallows two of the flywheel pair's four headline rows: strength (+17.5 Elo vs a **25.2** seed floor, **53.1** on the common-step frame) and **`gate.ece.all` (0.051 vs a 0.049–0.059 seed floor — the pair's ONE decisive DETECTED row, retired)**; the untaught meter (+8.31 vs **3.69 pp**) and the entropy trajectory (0.00133 vs **0.00103** nats/M) clear it, the latter by 23 % and with its "FLAT vs DECAYING" form REFUTED — all three arms decay; plus **W_b leads arm W at 15 of 15 shared ladder steps**, which is the same monotone shape the pair read had described for arm S and refused to claim
>
> `designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/`. Bars pre-registered in
> `PREDICTION.md` and committed BEFORE any W_b number existed. **THE FLOOR ARM.**
> `ai_v13_04_flywheel_winprob_b` (**W_b**, `--seed 1002`) is `ai_v13_02_flywheel_winprob`'s
> (**arm W**, `--seed 1001`) TOKEN-EXACT seed replicate — the argv multiset difference verified here
> as exactly `1001 → 1002` plus the run name (232 → 232 tokens), same pin `6eb9c776` for all
> 75,005,952 steps, `role: fresh`, config v119 / `gen3_critic_route_wave_v1`, `--ent-coef 0.05`,
> `--eval-sentinel-greedy` RECORDED, crossing at the **identical 4,128,768**, 20 ladder nodes.
> Every row produced by exactly the code paths that produced arm W's and arm S's — same scripts with
> the third arm added, same harness, same seeds (20260910 / 20260911), same cells, same team seeds
> (`BASE_OURS 20260914`), same ports. Everything CPU-only (`CUDA_VISIBLE_DEVICES=""`), `nice`,
> `OMP_NUM_THREADS=1` for peers, from the main checkout, **nothing written under `models/`**; Showdown
> on :9450 and :9417 started and stopped by their own **node** PIDs and `main.anchors` on its own
> :9500, **:8000 and :8001 never touched**. 🚨 **THE RULE, fixed in advance: a finding is OUTSIDE THE
> 75M FLOOR only if `|S − W| > |W − W_b|` AND the `S − W` CI excludes the `|W − W_b|` POINT; else
> WITHIN FLOOR at n = 2 — and ONE PAIR BOUNDS A FLOOR, IT DOES NOT ESTIMATE ONE (rules 19/22), so
> there is no CI on the floor, WITHIN FLOOR is never "equivalent" (rule 6), and a row that CLEARS it
> is a candidate that has survived ONE replicate, not a family verdict.** **STRENGTH.** All three
> committed `ladder.json` files carry the recipe stamp (`eval_sentinel_edges_dropped` 51 / 48 / 47)
> and reproduce their current-code refits to **0.0** (rule 24 satisfied three times). At matched
> COUNT, newest node: arm S 2036.6 @72.0M, arm W 2019.1 @72.0M, **W_b 2044.3 @74.0M** (se 8.9 / 8.7 /
> 8.9) ⇒ **finding 17.5 [−6.9, +41.9] against a floor of 25.2 ⇒ WITHIN FLOOR**; second-newest 10.2
> against 17.6 ⇒ WITHIN. On the **COMMON-STEP refits** — the frame the registration itself calls the
> honest one — the finding is 6.7 [−20.2, +33.6] and **the floor is 53.1, eight times it**.
> 🚨 **AND THE SHAPE ARGUMENT COLLAPSES: W_b leads arm W at 15 of 15 shared steps, by 22–70 Elo, and
> the lead does NOT narrow at the end** — the pair read's "arm S leads at all 16 shared steps,
> narrowing to +6.7" was offered as the pattern worth replicating, and the replicate reproduces it
> larger and more monotone with the treatment held fixed. **A monotone sign across a run's shared
> ladder nodes is not evidence of a treatment at this depth.** ⚠️ The imported 45.0-Elo control floor
> is NOT superseded: 45.0 is a MAX over three same-argv 10M four-node controls, 25.2/53.1 is one
> pairwise |Δ| at 75M/20 nodes, and rule 3 makes a floor a MAX over replicates — both are reported,
> the smallest claimable |Δ| stays 69.4. Late slopes (no bar attaches): S **+1.14 ± 0.44**, W
> **+0.78 ± 1.33**, **W_b +1.82 ± 0.89** — a **1.04 Elo/M seed spread**, so the pair's slope contrast
> is inside its own floor; all three still gaining at 75M. W_b's node-to-node adjacent spread is the
> tightest of the three (max 24.1 vs 37.8 / 56.9). ⚠️ W_b's pool starts 10M later (36.0M → 74.0M), so
> matched-COUNT ordinals differ by up to 14M and the three-way common set is **n = 11, below the
> registered n ≥ 12 report floor** and carries no verdict. **ENTROPY.** All three arms 740–741 points
> over 196,608 → 75,005,952 and all three crossed at 4,128,768, so every post-crossing window is
> matched. Post-crossing slopes: arm S **−0.00003 ± 0.00005** (t = −0.57, **−0.002 nats** total), arm
> W **−0.00136 ± 0.00006** (t = −23.6, **−0.096 nats**), **W_b −0.00033 ± 0.00006 (t = −5.62,
> −0.023 nats)**. Finding 0.00133 [0.00118, 0.00148] vs floor **0.00103 ⇒ OUTSIDE THE 75M FLOOR —
> by 23 %.** 🚨 **The rule passes and the description fails: W_b DECAYS TOO, at a quarter of arm W's
> rate, and the contrast against that seed (0.00030) is a THIRD of the floor. "Arm S holds entropy
> FLAT while arm W decays" is a statement about one seed; the config-level statement is a MAGNITUDE
> ordering S < W_b < W.** Clause (b) is near-vacuous on this row and is reported as such — its CI is
> an OLS residual se over ~700 autocorrelated points on two single trajectories, i.e. line-fit
> uncertainty, not run-to-run movement. H_end 1.0292 / 1.0648 / **1.0445**: finding 0.0356, floor
> 0.0203, both far inside the 0.074-nat imported three-seed floor, and W_b sits BETWEEN S and W at 14
> of 16 matched-step buckets. **CRITIC ROWS — the first LIKE-FOR-LIKE pair in the campaign.** Both
> sides are `--critic winprob`, so `max|values − win_probs|` = **0.0 on both** and every row
> including the whole calibration family is CRITIC-vs-CRITIC; the pair read's calibration rows were
> necessarily ROW A (arm S's AUXILIARY win-prob head at coef 0.05 against arm W's value function),
> and **that caveat belongs to the finding — no floor removes it.** Two offline full-capture draws
> per arm, **9,600 battles each, complete, shortfall 0**, sentinels GREEDY (RECORDED), draws 0.4–0.5 %,
> anchors 150/150, all three arms read at **74,000,016 — 1,005,936 steps (1.34 %) short of each end,
> an exactly matched distance**; frames quota-MATCHED (caps 799/392/7 and 797/395/9) before any
> frame-sensitive row was labelled. 🚨 **`gate.ece.all`: arm S's aux head 0.0178, arm W's critic
> 0.0688 / 0.0715, W_b's critic 0.0101 / 0.0222. Finding 0.0510 [0.0415, 0.0596]; FLOOR 0.0587
> (draw 1) / 0.0493 (draw 2) ⇒ WITHIN FLOOR at n = 2 ON BOTH DRAWS. The pair's one decisive DETECTED
> row does not survive its own seed floor, and the direction flips — W_b's critic is BETTER
> calibrated than arm S's diagnostic head.** `gate.reliability.all` falls the same way (finding
> 0.0046, floor 0.0050 / 0.0051, WITHIN on both draws). `gate.ece.bot` clears at 1.3× (0.0614 vs
> 0.0467 / 0.0452); the calibration-in-the-large INTERCEPT clears at 1.8–2.7× (0.5469 vs 0.2044 /
> 0.3123), so **the SIGN of the win-prob critics' pessimism survives and its SIZE does not**
> (+0.58 on arm W, +0.38 / +0.27 on W_b, +0.03 on arm S, at a shared slope 1.28–1.39). ✅ **The
> dissociation the pair drew SURVIVES and sharpens: RESOLUTION reproduces across seeds and
> CALIBRATION does not** — `gate.resolution.all` 0.0593 / 0.0593 / 0.0551 (a 0.00001 seed floor on
> draw 1) while ECE moves 0.059 between seeds. 🚨 **STANDING CONSEQUENCE: `--critic winprob`'s
> calibration is a RUN-LEVEL quantity with a ≈ 0.05 ECE floor at 75M, and no banked read clears it.**
> The registered guard `cond.opp_class_auc.t4_10` (finding 0.0254, **NOT DETECTED** against the
> imported 0.02451 floor to begin with) has a floor of 0.0014 / 0.0172 — **SPLIT across draws ⇒ NOT
> CONFIRMED** (rule 21); no verdict moves. **Rule 19 is now measured at 75M**: on the calibration
> family the RUN component is **4–56×** the eval-draw component (ECE draw spreads 0.0028 / 0.0121
> against a 0.049–0.059 run floor), while on resolution and the conditioning guard the two are the
> same size — so more DRAWS of one checkpoint buy nothing on calibration and everything must be an
> arm. **UNTAUGHT METER.** Registry opponent `untaught_meter_opponent` (= `ai_v9_29_rev1_0823
> @24,000,000`), 200 games/team over the untaught 8, seed 0, concurrency 1, **W_b and arm W in ONE
> invocation**, 3,200 battles per config, **0 timeouts**, levels IDENTICAL under BOTH config
> resolutions: arm S **54.50**, arm W **46.19**, **W_b 49.88 pp**. ✅ **Arm W reproduces its banked
> 2026-09-16 level EXACTLY (Δ 0.00 pp, 739/1600 both times)** — a second confirmation that the meter
> is deterministic at seed 0 / concurrency 1. Paired team-clustered: finding **+8.31 pp
> [+5.69, +11.19], 8/8 teams**; **FLOOR |W − W_b| = 3.69 pp** (6 of 8 teams favour W_b) ⇒ **OUTSIDE
> THE 75M FLOOR at n = 2** — 🚨 **but the same contrast against the OTHER seed is +4.62 pp
> [+0.69, +8.56] on 5 of 8 teams**, so the defensible statement is *the shaped arm is ahead of BOTH
> win-prob seeds by 4.6–8.3 pp against a 3.7 pp seed floor*, still a DESCRIPTOR and not an endpoint
> (§8.5). **This is the meter's first RUN-LEVEL floor at any depth, and at 3.69 pp it sits between
> the 1.66 pp frozen-dose and 4.27 pp controller-live FOLD floors** — the fold floors were not badly
> wrong in magnitude. **EXTERNAL ANCHORS, each in its own regime, all WITHIN FLOOR.** vs Metamon
> `SmallRL` (ckpt 40, 13.9M, `VanillaAttention`, CPU, `@0a00a759`), greedy-vs-greedy verified per
> decision (`argmax_match_rate` 1.0000 everywhere it was recorded), 100 games per cell as two
> role-balanced half-cells: **home 0.630 / 0.650 / W_b 0.590** ⇒ finding 0.020, **floor 0.060**;
> **away (`python -m main.anchors`) 0.520 / 0.500 / W_b 0.590** ⇒ finding 0.020, **floor 0.090**.
> ✅ **The away floor lands INSIDE the anchors SOP's own three-seed run-level floor for this exact
> cell — 0.090 against 0.110 — the first 75M corroboration of a bar the SOP had to import from 10M**,
> and 4.5× the SOP's 0.020 eval-draw floor, which is rule 19's point again. vs **Foul Play**
> (`6c467c08` + poke-engine 0.0.48, `--features gen3`) at `--search-time-ms 1000
> --search-parallelism 1`, 80 games over the same 8 pinned pool teams: **W_b 0.400 [0.300, 0.510] at
> 1.134 M realized visits/decision** against arm W's 0.475 at 1.153 M and arm S's 0.450 at 1.249 M ⇒
> finding 0.025, **floor 0.075 ⇒ WITHIN FLOOR**. 🚨 **W and W_b are WIDTH-MATCHED to 1.6 % (ratio
> 0.984) — the first width-matched Foul Play pair in the programme — and at matched width the SEED
> alone moves the cell 7.5 pp.** 🚨 **And the "win rates order exactly inversely to realized width"
> observation is BROKEN by the fourth point: 1.400 M → 0.388, 1.249 M → 0.450, 1.153 M → 0.475,
> 1.134 M → 0.400.** Rule 23 is untouched — a width meter still needs width matching — but that
> specific three-point inverse ordering must not be carried forward as evidence for it. All four 75M
> policies remain BELOW Foul Play at this budget; the external anchors continue to bracket us.
> **THE G7 EXCURSION.** The AMENDMENT-6 frozen references computed here REPRODUCE the banked ones to
> the third decimal (arm W 25.327, W_b 22.330), which is the check that this is the series the tool
> quoted. Inside a 24M–34M window pinned before either arm was read, **the excursion peaks at
> EXACTLY 30,000,000 on ALL THREE ARMS** — arm S 24.95 turns (ratio 1.070, 85.6 % of bar), arm W
> 28.14 (1.111, 88.9 %), W_b 25.90 (1.160, 92.8 %) — rises over the preceding 2–4M and is back inside
> each arm's own median by 32–34M, with amplitudes over each arm's own median of **+2.18 / +1.79 /
> +2.27 turns**. 🚨 **The completion entry called it a property of the 75M WIN-PROB trajectory; the
> third arm extends it — arm S, a different critic objective and a different reward composition, does
> the same thing at the same step with the same amplitude. It is a property of this recipe's
> curriculum at ~30M.** It is not a pathology: no arm breached G7, stall half-peaks 0.0237 / 0.0145,
> rule 12's 25 % threshold nowhere near. ⚠️ The frozen reference moved **3.0 turns between two seeds**
> (25.327 → 22.330), so a RATIO is comparable only WITHIN an arm and the raw turns only across them —
> W_b's 1.160 reads larger than arm W's 1.111 entirely because its reference is lower.
> `rollout/ep_len_mean` post-crossing mean 40.48 (S) / 44.53 (W) / **43.75 (W_b)**: the pair's
> "+4 turns on the win-prob arm" survives with a **0.78-turn** floor. **THE REALIZED DOSE — a floor on
> the pair's own hazard H-F.** The pair found the KL controller had annealed arm S to `lr_median`
> 4.32e-4 against arm W's 3.00e-4, a **1.44×** realized-dose gap on identical dose tokens. **W_b's is
> 3.30e-4 — a 1.10× gap on the SAME argv**, so about a quarter of the S-vs-W gap (on a log scale) is
> reproduced by changing nothing but the seed. H-F is not dissolved (1.44× > 1.10×) but it now has a
> floor, and **clearing any floor in this read does not clear the dose**, which W_b does not measure.
> **HAZARDS, each a finding.** (1) 🚨 **The Metamon `RecursionError` is about WHO CHALLENGES, not
> about sampling.** The pair's H-H recorded three occurrences, all in *ours greedy / Metamon SAMPLING
> / Metamon challenging*, and called it a property of that configuration; **this one is in a
> GREEDY-vs-GREEDY cell** — the fourth, same 988-frame signature out of `metamon_to_amago.py::step`
> after its last game. The common factor is **Metamon CHALLENGING**. Our side's 50 results are
> complete and feed the win rate; that half-cell's positional join is unreliable (`sides_disagree` 22,
> `metamon_row_missing` 2) against 1 in the other half. (2) 🚨 **A Foul Play session died silently and
> would have cost a pinned team**: session 5 ran its full 2400 s timeout logged in and challenging with
> **zero battles initialised** (the H10 class); undetected the campaign would have been 70 games over
> 7 of 8 teams and not team-matched with arm S's and arm W's. Caught by counting rows per team file and
> **re-run to completion** (failed logs kept); the retry ran at a lighter load and its realized width
> (1.314 M) is recorded rather than smoothed. (3) ⚠️ W_b's home anchor cell carries a **26 pp role
> split whose Newcombe CI excludes zero** (0.72 Metamon-challenges vs 0.46 we-challenge), against
> 0.66/0.64 on arm W — a standing reason this cell's run floor is ~0.09–0.11 and not ~0.02. (4) ⚠️ The
> three arms' pool SENTINELS are different checkpoints (36/54/72M vs 26/46/72M vs 22/42/72M), so the
> offline frames are matched in KIND but not in the identity of three of twelve opponents. (5) ⚠️ Six
> `Decision context is missing at turn 1` tracebacks on our side in the away cell and 2/100 games at
> the 250-turn forfeit; `status: OK`, 100/100 recorded, far under rule 12's threshold. (6) ⚠️ The live
> GPU arm `ai_v13_05_exploit_big5starmie` **completed on its own at 15:10 PT** during this read (final
> aggregate 99.8 %) and was never touched; this read was CPU-only throughout, box load 13–43.
> **NOTHING UNDER `src/` WAS CHANGED**, and the pair read's `--v-column` quota-match defect (H-L) is
> untouched and irrelevant here (both columns are `win_probs` by construction). **WHAT IS NOT
> CLAIMED:** that the floor is known (one pair bounds it; n = 2); that a row clearing it is a family
> verdict; that WITHIN FLOOR means equivalent (rule 6); that the ECE row was ever a critic-vs-critic
> comparison (it was a CRITIC against a DIAGNOSTIC and the floor only says it was unresolvable at
> n = 1); anything about the critic objective in general, at 277M, in a fold, or after a
> distillation. Tag: **MEASURED · THE 75M RUN-LEVEL FLOOR · strength WITHIN FLOOR (17.5 vs 25.2; 6.7
> vs 53.1) · `gate.ece.all` WITHIN FLOOR on both draws — the pair's one DETECTED row RETIRED ·
> `gate.reliability.all` WITHIN FLOOR · untaught +8.31 vs 3.69 pp OUTSIDE (but +4.62 vs the other
> seed) · entropy slope OUTSIDE by 23 % with "flat vs decaying" REFUTED · anchors ×3 WITHIN FLOOR,
> away 0.090 corroborating the SOP's 0.110 · first WIDTH-MATCHED Foul Play pair, seed moves it 7.5 pp
> · G7 excursion peaks at exactly 30.0M on ALL THREE arms · realized-dose gap has a 1.10× seed floor ·
> Metamon RecursionError is the CHALLENGE role, not sampling**.
