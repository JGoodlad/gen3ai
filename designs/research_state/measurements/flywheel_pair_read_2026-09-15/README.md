# THE FLYWHEEL-ERA PAIR — the S-vs-W read, 2026-09-15/16

**The pair is complete.** `ai_v13_01_flywheel_shaped` (**arm S — the era's SHAPED configuration**:
`--critic shaped`, PopArt ON, a 51-atom distributional `E[Z]`, the win-prob head kept as an
AUXILIARY at `--win-prob-coef 0.05`) and `ai_v13_02_flywheel_winprob` (**arm W — the WIN-PROB
critic**: `V(s) = sigmoid(win-prob logit) ∈ [0,1]`, the value loss IS that head's BCE against the
terminal outcome) both finished at **75,005,952 steps**, both pinned to **`6eb9c776`** for every one
of their 75M steps, both `--seed 1001`, `--ent-coef 0.05`, `--eval-sentinel-greedy` (RECORDED true),
`--no-value-true-team`, both `config_version` 119 / `arch_signature gen3_critic_route_wave_v1`, both
crossed into self-play at the **identical 4,128,768**, both retain **20 ladder nodes**.
The treatment is the critic objective and, on the resolved config, nothing else
(16 differing keys of 283; 15 the treatment or forced by it, 1 the run name
— [the registration](../../flywheel_era_pair_2026-09-12.md) §4).

This note produces **arm W's run-end reads by exactly the code paths that produced arm S's**
([`flywheel_armS_reads_2026-09-14/`](../flywheel_armS_reads_2026-09-14/) — same scripts, same
harness, same seeds, same cells, same ports) and then the **S-vs-W diff on every registered row**,
with the §9 bars applied and nothing claimed past them.

🚨 **The binding limitation, stated before any number: ONE SEED PER ARM.** Registration §9.1 — there
is no run-to-run floor at 75M and none is affordable (~35 GPU-h each). Rule 22 therefore binds on
every row: **a single-arm direction is a CANDIDATE, never a family verdict**, and a null is
**NOT DETECTED, never "equivalent"** (rule 6: equivalence needs the DELTA's own CI *inside* a bar,
and every bar in reach here is imported from a 10M four-node depth).

**Everything ran CPU-only** (`CUDA_VISIBLE_DEVICES=""`), `nice`, from the main checkout, with
nothing written under `models/`. A training arm — `ai_v13_03_fork`, launched 04:30 PT — held the GPU
throughout and **was not touched**. Two Showdown servers were started and stopped by their own
recorded PIDs on **:9450** (Metamon) and **:9417** (Foul Play), and `main.anchors` started and
stopped its own on **:9500**; **:8000 and :8001 were never touched.**

---

## 0. THE PAIR TABLE — every registered row, with its verdict

Sign convention throughout: **Δ = arm S − arm W**, so a positive number means the SHAPED arm is
ahead.

| # | row | arm S | arm W | Δ (S − W) | bar / floor, with its provenance | verdict under the registered rule |
|---|---|---|---|---|---|---|
| 1 | **ladder Elo**, 20 nodes, current-recipe refit, newest node (both at **72.0M**) | **2036.6** (se 8.9) | **2019.1** (se 8.7) | **+17.5** [−6.9, +41.9] | claimable \|Δ\| > 45.0 + 1.96·se(Δ) = **69.4** (45.0 = MAX pairwise \|Δ\| over three same-argv **10M four-node** controls, imported) | **NOT DETECTED** |
| 1b | same, second-newest node (68.0M vs 70.0M) | 2042.3 | 2032.1 | **+10.2** [−14.2, +34.6] | 69.4 | **NOT DETECTED** |
| 1c | **COMMON-STEP refit**, 16 shared steps, newest (72.0M both) | 2032.3 | 2025.6 | **+6.7** [−20.2, +33.6] | 71.9 | **NOT DETECTED** |
| 2 | **late slope**, newest node dropped | **+1.14 ± 0.44** Elo/M (48–68M) | **+0.78 ± 1.33** Elo/M (52–70M) | — | **no bar attaches** (registration §8.2) | DESCRIPTION: arm S's is 2.6 se clear of zero, arm W's is 0.6 se — *both point up, only arm S's is distinguishable from flat* |
| 3 | **H_end** (median-20, nats) | **1.0292** | **1.0648** | **−0.0356** | 0.074-nat three-seed replicate floor, measured at **10M** | **NOT DETECTED** (inside the floor) |
| 3b | **H slope over the whole post-crossing span** (4.2M → 75.0M) | **−3e-5 ± 5e-5** nats/M (t = −0.57) | **−1.36e-3 ± 6e-5** nats/M (t = −23.6) | — | no bar | **FINDING** (§8.3 makes an H difference a finding, not a bar): **arm S holds entropy FLAT for 71M steps; arm W decays**, ≈ **−0.097 nats** over the span, which clears the 0.074 floor as a level change |
| 4 | **`cond.opp_class_auc.t4_10`** — ROW A, `--v-column win_probs`, MATCHED frame (arm S's AUXILIARY head vs arm W's CRITIC) | **0.7686** | **0.7431** | **+0.0254** [+0.0120, +0.0396] | 0.02451 (MAX of the two v6 **10M control-vs-control** floors, imported) | **NOT DETECTED** — the CI does not clear the floor; reproduces on both draws |
| 4b | **`cond.opp_class_auc.t4_10`** — ROW B, rank-only, AS TRACED (arm S's ACTUAL critic `values` vs arm W's CRITIC) | 0.7683 / 0.7579 (two draws) | 0.7449 / 0.7284 | **+0.0236** / **+0.0297** | 0.02451, same import | **NOT CONFIRMED** (rule 21) — inside the floor on one draw, outside on the other; no matched-frame read exists (hazard H-L) |
| 4c | **`gate.resolution.all`** — ROW A (probability columns) | +0.0551 | +0.0593 | **−0.0043** [−0.0108, +0.0021] | 0.0024, imported 10M | **NOT DETECTED** (CI covers zero) — *the two heads separate outcomes equally well* |
| 4d | **`gate.ece.all`** — ROW A (probability columns) | **+0.0178** | **+0.0688** | **−0.0510** [−0.0595, −0.0415] | 0.0245, imported 10M | 🚨 **DETECTED on both draws — arm W's critic is ~4× less calibrated**, at indistinguishable resolution. 🚨 A **CRITIC vs a DIAGNOSTIC**, never critic-vs-critic (§4.4) |
| 4e | **`gate.reliability.all`** — ROW A | +0.0005 | +0.0051 | **−0.0046** [−0.0060, −0.0035] | 0.0026, imported 10M | **DETECTED on both draws**, same direction and same caveat |
| 4f | **calibration SLOPE**, all states — ROW A | +1.3463 | +1.3504 | **−0.0041** [−0.0906, +0.0896] | 0.2001, imported 10M | **WITHIN FLOOR** — the same dispersion; what differs is the LEVEL (intercept +0.03 vs +0.58) |
| 5 | **untaught meter** (pp), registry opponent, IDENTICAL under both config resolutions | **54.50** [52.25, 56.62] | **46.19** [42.81, 49.12] | **+8.31** [+5.69, +11.19], **8 of 8 teams** | fold floors 1.19 / 1.66 / 4.27 pp, all at **~1M fold depth**; neither arm is a fold, so they are quoted, not applied | **run-level CANDIDATE** — clears every floor in evidence and the CI excludes zero, but §8.5 makes this row a **DESCRIPTOR, not an endpoint**, and rule 22 binds at n = 1 per arm |
| 6 | **Metamon `SmallRL`**, greedy-v-greedy, **home** pool, 100 games | **0.630** [0.532, 0.718] | **0.650** [0.553, 0.736] | **−0.020** [−0.151, +0.111] | no bar exists for this row | **NOT DETECTED** |
| 6b | **Metamon `SmallRL`**, greedy-v-greedy, **away** set, 100 games (`main.anchors`) | **0.520** [0.423, 0.615] | **0.500** [0.404, 0.596] | **+0.020** [−0.117, +0.155] | no bar | **NOT DETECTED** |
| 6c | **Metamon `SmallRL`**, ours greedy / theirs **t = 1.0**, home, 100 games | **0.840** [0.756, 0.899] | **0.760** [0.668, 0.833] | **+0.080** [−0.032, +0.190] | no bar | **NOT DETECTED** |
| 7 | **Foul Play** @ `--search-time-ms 1000`, 80 games | **0.450** [0.346, 0.559] at **1.249 M** visits/decision | **0.475** [0.369, 0.583] at **1.153 M** visits/decision | **−0.025** [−0.175, +0.127] | no bar; **rule 23 — a WIDTH meter**, and the widths are **NOT matched** (arm W's opponent 7.7 % narrower, in arm W's favour) | **NOT DETECTED**, and the width gap runs the way that flatters arm W |

🚨 **NOT DETECTED is never "equivalent."** Rule 6: equivalence needs the delta's own CI *inside* a
bar. Row 1's CI95 is ±24.4 and sits inside a 45.0-Elo floor imported from another depth, so the
equivalence clause is **unavailable** — as the registration said in advance it would be.

**One sentence for the whole read:** *on the PRIMARY endpoint (strength), on the registered critic
guard, and on all four external-anchor cells, the two critic objectives are **NOT DETECTED** apart
at 75M with n = 1 per arm; the rows that do move past a floor are ones the registration had already
declared could not be endpoints — the entropy TRAJECTORY (arm S flat across 71M steps, arm W
decaying), the untaught meter (+8.31 pp to arm S, 8 of 8 teams, a DESCRIPTOR by §8.5), and the
calibration family, where arm W's critic carries ~4× the ECE of arm S's spare win-prob head at
indistinguishable resolution — a CRITIC-vs-DIAGNOSTIC contrast, not a critic-vs-critic one.*

**And the one shape worth carrying forward:** arm S leads at **every one of the 16 shared ladder
steps**, by +25 to +57 Elo through 26–42M, narrowing to **+6.7 at the shared 72.0M node**. No
single gap is claimable (the smallest claimable |Δ| is 71.9 on that fit) and the 16 points are not
independent draws — but "a lead that is large in mid-run and gone by the end" is exactly the pattern
the registration asked to be shown, and it is the reason the named next increment is a **seed
replicate of arm W**, not a further offline read of these two.

---

## 1. Provenance — both arms, side by side

| | **arm S** `ai_v13_01_flywheel_shaped` | **arm W** `ai_v13_02_flywheel_winprob` |
|---|---|---|
| steps | 75,005,952, `latest.txt` → `final_model.zip` | 75,005,952, `latest.txt` → `final_model.zip` |
| pin (`pin_history`, one row each) | `6eb9c776…` for steps 0 → 75,005,952 | `6eb9c776…` for steps 0 → 75,005,952 |
| `lineage.role` | `fresh` (no parent) | `fresh` (no parent) |
| `config_version` / `arch_signature` | 119 / `gen3_critic_route_wave_v1` | 119 / `gen3_critic_route_wave_v1` |
| `critic` | `shaped` | `winprob` |
| `use_popart` / `value_dist_mode` | true / `shaping` | false / `none` |
| `terminal_indicator` / `victory_value` / `draw_penalty` | false / 30.0 / −35.0 | true / 1.0 / 0.0 |
| `hand_shaping` / `win_prob_coef` | true / 0.05 (AUXILIARY) | false / 1.0 (the value loss itself) |
| `vf_coef` | 0.5 (a PopArt-normalised MSE) | 0.5 (a BCE) — hazard H4: matched in NUMBER, not in UNITS |
| `eval_sentinel_greedy` | **true, RECORDED** | **true, RECORDED** |
| self-play crossing | **4,128,768** | **4,128,768** |
| ladder nodes | 20, 22.0M → 72.0M | 20, 26.0M → 72.0M |
| wall clock / FPS | 37 h 36 m / 410 | 39 h 23 m / 567 |
| **declared dose** | 4.5776e-8 | 4.5776e-8 — identical tokens in both argvs |
| **realized dose** (`python -m main.dose`) | **6.592e-08** (lr_median 4.32e-4), 3.07× ref | **4.578e-08** (lr_median 3.00e-4), 2.13× ref |

🚨 **THE ONE MATCHED VARIABLE THAT CAME OUT UNMATCHED — and it is a consequence of the treatment,
not a confound.** The four dose inputs (`--lr`, `--n-epochs`, `--batch-size`, `--grad-accum-steps`)
are the same tokens in both argvs, so the dose is equal **by declaration**, which is the
registration's own convention (§5). But the **KL controller anneals the realized LR**, and it
annealed the two arms differently: arm S's `lr_median` ran up to **4.32e-4** against arm W's
**3.00e-4**, a **1.44× realized-dose gap in arm S's favour**. The registration anticipated the
mechanism in words (*"the realized `lr_median` is set by the KL controller's annealing and is not a
variable either argv controls"*) but no arm-to-arm number existed until now. **The honest reading:
step size is downstream of the objective here, in the same class as `gamma` and PopArt — it is
FORCED, not chosen — but any row on which arm S leads must be read knowing arm S also took
1.44× the realized step-size dose.** It is listed as hazard **H-F** below.

Other provenance: reads ran at `5ff03aae` + this branch's scripts; Showdown `deps/pokemon-showdown`
@ `e0551883f`; Metamon `@0a00a759`, `SmallRL` ckpt 40 (13.9M), `VanillaAttention`, CPU; Foul Play
`@6c467c08` + poke-engine `0.0.48` (`--features poke-engine/gen3 --no-default-features`).
Box: 16 cores carrying the live `ai_v13_03_fork` arm throughout; load average 15–25 across the
session. The fork arm was verified alive and untouched at the end of the session.

---

## 2. STRENGTH — the primary endpoint

**Instrument:** `<run>/snapshot_ladder/ladder.json` (dense, ±10), never `eval/elo` (±29)
[UNDERSTANDING §3.2 rule 1]. Every refit goes through
`fit_ladder(run_dir, first_n=N | steps=COMMON, write=False)`; `first_n` forces `write=False`, so
nothing under `models/` was written.

### 2.1 Rule 24 is satisfied on BOTH sides of the pair — and still bites the third leg

The arm-S read's most transferable finding was that **a committed `ladder.json` carries the RECIPE
it was fitted with**. It is checked here for every run in the table:

| run | committed computed_at | pairs in the committed fit | `eval_sentinel_edges_dropped` (the recipe stamp) | committed newest | current-code refit | committed − refit |
|---|---|---|---|---|---|---|
| **arm S** | 2026-09-14 | 190 of 190 possible | **51** | 2036.6 | **2036.6** | **0.0** |
| **arm W** | 2026-09-16 | 190 of 190 possible | **48** | 2019.1 | **2019.1** | **0.0** |
| the 0.02 leg `ai_v12_02` | 2026-09-08 | **494** of 190 possible | `null` — NO STAMP | 2057.3 | **1984.2** | **+73.1** |
| the v8 control | 2026-07-22 | 45 of 45 | `null` | 2026.7 | 2022.3 | +4.4 |

**Both arms of the pair are current-recipe and their committed files reproduce exactly**, so the
pair's own Δ is unaffected by the trap. The 0.02 leg is REFIT everywhere it appears below
(registration §9b amendment 1; rule 24), and **its committed 2057.3 is never quoted.**

### 2.2 The two arms' nodes, at matched COUNT (n = 20 each)

| k | arm S step | arm S Elo | arm W step | arm W Elo | S − W (same ordinal) |
|---:|---:|---:|---:|---:|---:|
| 1 | 22.0M | 1982.7 | 26.0M | 1972.8 | +9.9 |
| 2 | 24.0M | 1967.8 | 28.0M | 1950.6 | +17.2 |
| 3 | 26.0M | 2005.6 | 30.0M | 1950.0 | +55.6 |
| 4 | 28.0M | 1991.3 | 32.0M | 1967.6 | +23.7 |
| 5 | 30.0M | 1988.9 | 34.0M | 1978.9 | +10.0 |
| 6 | 32.0M | 2016.2 | 36.0M | 1965.2 | +51.0 |
| 7 | 34.0M | 2025.2 | 38.0M | 1971.9 | +53.3 |
| 8 | 36.0M | 2020.9 | 40.0M | 1977.1 | +43.8 |
| 9 | 38.0M | 2038.5 | 42.0M | 1999.2 | +39.3 |
| 10 | 40.0M | 2020.2 | 44.0M | 2000.4 | +19.8 |
| 11 | 42.0M | 2040.0 | 46.0M | 1999.2 | +40.8 |
| 12 | 44.0M | 2017.7 | 48.0M | 2004.8 | +12.9 |
| 13 | 46.0M | 2025.6 | 50.0M | 2000.4 | +25.2 |
| 14 | 48.0M | 2022.4 | 52.0M | 2007.3 | +15.1 |
| 15 | 50.0M | 2019.6 | 54.0M | 1993.9 | +25.7 |
| 16 | 54.0M | 2039.1 | 56.0M | 2008.2 | +30.9 |
| 17 | 56.0M | 2041.9 | 60.0M | 2000.7 | +41.2 |
| 18 | 62.0M | 2043.5 | 64.0M | 1975.2 | +68.3 |
| 19 | **68.0M** | **2042.3** | **70.0M** | **2032.1** | **+10.2** |
| 20 | **72.0M** | **2036.6** | **72.0M** | **2019.1** | **+17.5** |

⚠️ **This table is at matched COUNT, which is the rule [§3.2 rule 3] — but for ordinals 1–19 it is
NOT at matched STEP**, because arm W's retained pool starts 4M later. That is exactly the hazard the
registration's §8.1 correction 1 names, so the honest version is §2.3.

### 2.3 🚨 THE COMMON-STEP REFIT — the registration's own correction, and the shape it shows

Both arms refit on the **16 steps they share** (`fit_ladder(steps=COMMON)`). 16 ≥ the registered
n ≥ 12 floor, so the row is reportable; per-node se is 9.7 on each side, se(Δ) 13.72, CI95 ±26.9,
and the smallest claimable |Δ| is **71.9**.

| step (M) | arm S | arm W | S − W |
|---:|---:|---:|---:|
| 26.0 | 2003.9 | 1974.7 | +29.2 |
| 28.0 | 1985.4 | 1955.6 | +29.8 |
| 30.0 | 1987.7 | 1962.6 | +25.1 |
| 32.0 | 2012.2 | 1977.7 | +34.5 |
| 34.0 | 2020.5 | 1981.8 | +38.7 |
| 36.0 | 2023.9 | 1972.2 | +51.7 |
| 38.0 | 2039.3 | 1982.5 | +56.8 |
| 40.0 | 2012.2 | 1982.9 | +29.3 |
| 42.0 | 2039.3 | 2005.0 | +34.3 |
| 44.0 | 2022.4 | 2009.2 | +13.2 |
| 46.0 | 2020.5 | 2000.1 | +20.4 |
| 48.0 | 2018.6 | 2012.2 | +6.4 |
| 50.0 | 2011.4 | 2000.5 | +10.9 |
| 54.0 | 2033.5 | 1993.3 | +40.2 |
| 56.0 | 2040.4 | 2021.7 | +18.7 |
| **72.0** | **2032.3** | **2025.6** | **+6.7** |

**This is the "transient lead that fades" pattern the registration asked to see.** Arm S is ahead at
**every one of the 16 shared steps**, by +25 to +57 Elo through 26–42M, and the lead narrows to
**+6.7 at the shared 72.0M node**. 🚨 **Not one of those per-step gaps is claimable**: the smallest
claimable |Δ| on this fit is 71.9 and the largest gap in the table is 56.8. **The SHAPE is a
description, and the endpoint is NOT DETECTED.** A sign that holds at 16 of 16 shared steps is the
kind of pattern rule 22 calls a CANDIDATE at best and names the next increment for — a seed
replicate of arm W — and it is emphatically not evidence that the shaped critic is better, because
the arms are not independent draws at those 16 steps (they are 16 points on two single
trajectories).

### 2.4 Resolution — the registration's node-count table reproduces on arm W too

| nodes | 0.02 leg (registration) | 0.02 leg (refit here) | arm S (refit) | **arm W (refit)** |
|---|---|---|---|---|
| 4 | 15.70 / 22.20 / ±43.5 | 15.70 / 22.20 / ±43.5 | 16.4 / 23.19 / ±45.5 | 15.8 / 22.34 / ±43.8 |
| 8 | 12.50 / 17.68 / ±34.6 | 12.50 / 17.68 / ±34.6 | 13.1 / 18.53 / ±36.3 | 12.3 / 17.39 / ±34.1 |
| 12 | 10.70 / 15.13 / ±29.7 | 10.70 / 15.13 / ±29.7 | 11.0 / 15.56 / ±30.5 | 10.8 / 15.27 / ±29.9 |
| 16 | 9.30 / 13.15 / ±25.8 | 9.30 / 13.15 / ±25.8 | 9.8 / 13.86 / ±27.2 | 9.6 / 13.58 / ±26.6 |
| **20** | **8.50 / 12.02 / ±23.6** | **8.50 / 12.02 / ±23.6** | **8.9 / 12.59 / ±24.7** | **8.7 / 12.30 / ±24.1** |

(se / se(Δ) / CI95.) On the pair's own two se's, se(Δ) = **12.45** and CI95 = **±24.4**, so the
smallest claimable |Δ| against the imported 45.0 floor is **69.4 Elo**. **Both arms clear the
registered n = 20 and the n ≥ 12 report floor.**

### 2.5 The within-run late slope (§8.2 — registered, and NO bar attaches)

Fit on each run's own nodes with the **newest node dropped** [§3.2 rule 2]; se from the residual
spread. "Late" is the last third of the surviving nodes, "middle" the middle third.

| run | nodes used | middle third | late third | late − middle |
|---|---|---|---|---|
| **arm S** | 19 | +0.02 ± 1.30 Elo/M (34–44M) | **+1.14 ± 0.44** Elo/M (48–68M), t = 2.62 | +1.12 ± 1.37 — CI covers zero |
| **arm W** | 19 | **+3.31 ± 0.83** Elo/M (38–48M), t = 3.97 | **+0.78 ± 1.33** Elo/M (52–70M), t = 0.59 | −2.53 ± 1.57 — CI covers zero |
| the 0.02 leg (refit) | 19 | +0.75 ± 1.67 (48–58M) | **−0.32 ± 1.44** (62–72M), t = −0.22 | −1.07 ± 2.20 — CI covers zero |
| the v8 control (refit) | 9 | −0.12 ± 0.46 (192–208M) | **−0.10 ± 0.15** (210–242M), t = −0.71 | +0.02 ± 0.49 — CI covers zero |

⚠️ **The last two rows do NOT match the prose of the arm-S note, and the arm-S note is the one in
error** — hazard **H-G**. `flywheel_armS_reads_2026-09-14/README.md` §2.4 prints the 0.02 leg's late
slope as `+0.15 ± 1.23` and the v8 control's as `+0.32 ± 0.11`; its OWN committed
`out/strength_read.json` says `−0.32 ± 1.4378` and `−0.1044 ± 0.1476`, and a fresh refit here
reproduces the JSON to the digit (the v8 late window is three nodes at 210.0 / 224.0 / 242.0M
reading 2010.0 / 2012.8 / 2007.0 — a downward fit). **Arm S's own slope row is unaffected and
reproduces exactly** (+1.1435 ± 0.4357 over 48–68M), so the arm-S note's headline stands; what is
wrong is its comparator row, which is a table this note must not copy forward.

**Both arms end with a positive late slope and neither late−middle difference is distinguishable
from zero.** What separates them is *where* the gaining happened: arm W's steepest fitted segment is
its MIDDLE third (+3.31 ± 0.83, 4 se clear of zero) and its late third is flat-with-a-wide-error;
arm S's is the other way round. **Reported as a description** — the registration attaches no bar
here, the two segment windows are at different steps, and a residual-se slope on six nodes is a
weak instrument (arm W's late-third residual RMS is 20.1 Elo against arm S's 7.3, which is what the
wide error bar is).

🚨 **The v8 row is SHAPE ONLY** [§3.2 rule 4]: different architecture, obs dim, action-space era,
reward composition, clip range (0.10 vs 0.15) and ecology; 20 nodes over 22–72M (S) and 26–72M (W)
against 10 nodes over 150.9–248.0M — a different DEPTH *and* a different COUNT, both far above the
bot anchors, and **no direct match was played**. §3.2 rule 6 adds a second reason to distrust any
cross-run gap on this frame: the bot frame is PRECISE and WRONG near the frontier. What can be
said: **the v8 line was still gaining
at a QUARTER-BILLION steps (+0.32 ± 0.11 Elo/M over 210–242M) and both of our 75M arms are still
gaining at 75M** (+1.14 ± 0.44 and +0.78 ± 1.33). The Elo gaps to it (arm S +14.3, arm W −3.2 at the
newest node) are in `pair_strength_read.json` for completeness and **must not be quoted as
differences.**

### 2.6 The third leg — `ai_v12_02_winprob_critic` at `--ent-coef 0.02`, REFIT

| | arm S | arm W | the 0.02 leg (REFIT) |
|---|---|---|---|
| newest node | 2036.6 @72.0M (se 8.9) | 2019.1 @72.0M (se 8.7) | **1984.2** @74.0M (se 8.5) |
| second-newest | 2042.3 @68.0M | 2032.1 @70.0M | 1980.3 @72.0M |
| Δ vs the 0.02 leg, newest | **+52.4** [+28.3, +76.5] | **+34.9** [+10.9, +58.9] | — |
| smallest claimable | 69.1 | 68.9 | — |
| verdict | **NOT DETECTED** | **NOT DETECTED** | — |

**Confounds on BOTH of those legs, named:** different pin (`6eb9c776` vs `f971caf2`), different
`--ent-coef` (0.05 vs 0.02), different step span at the same node count, and — for the arm-S leg
only — a different critic objective as well. **The arm-W leg is the cleaner one**: same critic
objective on both sides, so it isolates the entropy-coefficient axis at full length, at the cost of
the pin. Both are **ladder-only** reads: the 0.02 leg spans the 2026-09-07 opponent-regime boundary,
so `eval/elo` and `win_rate_vs_pool` are NOT comparable across it (+8.9 pp [+7.0, +10.7] to the
trainee under the old asymmetry) while `ladder.json` and every bot edge ARE unaffected. Neither
delta is claimable, and quoting the 0.02 leg's COMMITTED file instead of the refit would have handed
it 73 free Elo and flipped both signs.

### 2.7 The within-run floor proxy (§9.1)

| run | adjacent-node \|Δ\| max | median | mean |
|---|---|---|---|
| arm S | 37.8 | 9.0 | 12.2 |
| **arm W** | **56.9** | 11.3 | 13.1 |
| the 0.02 leg | 30.6 | 11.6 | 12.6 |

This bounds **within-run wobble only**, is dominated by early LEARNING, and is **not** a run-to-run
floor — the registration rejects it as one explicitly. Arm W's 56.9 is the 62M→64M→70M excursion
(2000.7 → 1975.2 → 2032.1) and is the largest single-node swing in either arm; it is one more reason
the pair's ±24 CI is the wrong intuition for how much a 75M arm's rating can move.

---

## 3. ENTROPY — `train/entropy_loss`, each arm on its own span (§8.3)

**Instrument:** the TENSORBOARD EVENTS via `main.ops.tb_read`, never the child log's table.
🚨 **SB3 logs `train/entropy_loss` as the NEGATIVE mean policy entropy**; every H below is its
negation, in nats, positive. Both spans were verified: arm S 741 points and arm W 740 points, both
over **196,608 → 75,005,952**, one fresh run each, no inherited prefix. **Both crossed at
4,128,768** (first `*_pool` scalar; promotion logged at 4,000,032), so — unlike most pairs in this
programme — rule 15's boundary falls at the identical step on both sides and every post-crossing
window below is genuinely matched.

| run | H_start (median-20) | H_peak | **H_end (median-20)** | slope 4.2M→75.0M | late-third slope |
|---|---|---|---|---|---|
| **arm S** (`--ent-coef 0.05`) | 1.4679 | 1.6724 | **1.0292** | **−0.00003 ± 0.00005** (t = −0.57) | −0.00211 ± 0.00019 (t = −11.3), 51.7–75.0M |
| **arm W** (`--ent-coef 0.05`) | 1.5365 | 1.6885 | **1.0648** | **−0.00136 ± 0.00006** (t = −23.6) | −0.00169 ± 0.00022 (t = −7.5), 51.7–75.0M |
| the 0.02 leg | 1.4950 | 1.6725 | **0.7132** | −0.00069 ± 0.00007 (t = −9.6) | −0.00034 ± 0.00024 (t = −1.4) |
| `ent05` @10M (the registered comparator) | 1.5487 | 1.6762 | **1.0861** | −0.01415 ± 0.00274 | +0.03911 ± 0.00815, 8.4–10.0M |

Matched-step medians, one point per 5M steps:

| step (M) | 0 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 | 65 | 70 | 75 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **arm S** | 1.179 | 1.029 | 1.051 | 1.070 | 1.055 | 1.082 | 1.055 | 1.047 | 1.066 | 1.075 | 1.063 | 1.074 | 1.047 | 1.038 | 1.029 | 1.038 |
| **arm W** | 1.310 | 1.181 | 1.104 | 1.133 | 1.110 | 1.091 | 1.102 | 1.107 | 1.079 | 1.091 | 1.085 | 1.090 | 1.071 | 1.051 | 1.069 | 1.031 |
| **S − W** | −0.131 | **−0.153** | −0.053 | −0.064 | −0.054 | −0.010 | −0.046 | −0.060 | −0.013 | −0.016 | −0.022 | −0.012 | −0.024 | −0.012 | −0.040 | +0.007 |

### What this says

* **The LEVELS are NOT DETECTED apart.** H_end differs by **−0.0356 nats**, well inside the
  0.074-nat three-seed replicate floor. Against the registered comparators, arm S reads −0.057 from
  `ent05`'s 1.0861 and arm W −0.021; both are inside the floor, so **neither arm's distance from
  `ent05` is readable either**, and the pre-registered prediction (*both arms sit near v8's
  1.07–1.11 plateau, because `ent05` PASSED at this coefficient*) **is not contradicted by either
  arm** — arm W's 1.0648 is 0.005 below the plateau's bottom edge and arm S's 1.0292 is 0.041 below
  it, both inside the floor.
* 🚨 **The row that IS a finding: the TRAJECTORIES differ.** Over the whole post-crossing span arm S
  is statistically flat (−3e-5 ± 5e-5 nats/M, t = −0.57 over 71M steps) while **arm W decays at
  −1.36e-3 nats/M, t = −23.6** — a total of **≈ −0.097 nats** across the span, which clears the
  0.074-nat floor *as a level change* even though the endpoint difference does not. The entropy
  coefficient is matched at 0.05 and is LIVE on a resume, so it held for all 13 restarts on both
  arms: **the difference is downstream of the reward composition, which is exactly the localisation
  §8.3 said this row could buy and the ladder could not.** No PASS/FAIL attaches.
* **The matched-step gaps are largest early and shrink monotonically.** The only bucket whose
  |S − W| clears the 0.074 floor is the 5M one (−0.153), immediately after the shared crossing —
  i.e. arm W starts its post-crossing life with more entropy and spends it, while arm S starts lower
  and holds. By 25M the gap is inside the floor and it stays there.
* Both arms' late thirds decay at a similar rate (−0.0021 vs −0.0017 nats/M); what separates them is
  the 4M–50M stretch, where arm S is flat and arm W is not.

---

## 4. THE CRITIC ROWS — and the control the arm-S read could not have

### 4.1 🚨 The registered control EXISTS this time, and the tool ACCEPTED it

Arm S's standalone read had to report LEVELS ONLY: `main.ops.critic_read` REFUSED its one available
control (the 73M win-prob trees) on two counts at once — `eval_sentinel_greedy: arm=True
control=False` and `n_games: arm=800 control=400`. **The pair is comparable by construction and the
tool takes it without a murmur:**

| | **arm W** (ARM) | **arm S** (CONTROL) |
|---|---|---|
| cycle | `step_74000016`, PINNED by `--step` | `step_74000016`, PINNED by `--step` |
| frame | offline full-capture, 12 opponents × 800 games | offline full-capture, 12 opponents × 800 games |
| battles traced | 9,600 (7872 W / 1685 L / 43 D) | 9,600 (7955 W / 1594 L / 51 D) |
| sentinel regime | GREEDY, **recorded** | GREEDY, **recorded** |
| draw/timeout share | 0.4 % | 0.5 % — rule 12's 25 % threshold is nowhere near |
| anchors reproduced | 150/150 (100.0 %) | 150/150 (100.0 %) |
| checkpoint | `eval_traces/step_74000016/snapshot.zip`, sha `7edc9f8e6aa82825` | sha `1f08931fb04ad151` |
| seeds | 20260910 / 20260911 | 20260910 / 20260911 — the SAME two |

🚨 **Both arms are read at 74,000,016, and it is the SAME step on both.** Neither run has an
`eval_traces` cycle at its final 75,005,952 and `eval_trace_gen` refuses a step with no weights of
its own, so both are **1,005,936 steps (1.34 %) short of the end — an exactly matched distance.**
That is the check the arm-S note asked the pair read to make, and it passes: the two critic rows are
the same distance from their arms' ends. Every other row in this note is at the true final model.

⚖️ **The frames were MATCHED before the frame-sensitive rows were labelled.** The realized
per-opponent caps differed slightly (arm 799/392/7, control 800/403/9 traced W/L/D), so `critic_read`
subsampled the control to 799/392/7 over 21 seeded draws with the capture rates RECOMPUTED per
subsample (rule 17). **The AUC rows carry the MATCHED label**; the as-traced value is printed beside
them and never labelled.

### 4.2 The three columns, and why the labels are load-bearing

| | what it is | `max|values − win_probs|` |
|---|---|---|
| **W-V** | arm W's **ACTUAL critic** — `V(s) = sigmoid(win-prob logit) ∈ [0,1]`. On a `--critic winprob` run `values` and `win_probs` are the SAME TENSOR | **0.0** |
| **S-WP** | arm S's **AUXILIARY win-prob head** at `--win-prob-coef 0.05` — a DIAGNOSTIC, not the value function, and **the column the v6 tool reads by DEFAULT** | **75.12** (draw1) / **119.06** (draw2) |
| **S-V** | arm S's **ACTUAL critic** — the distributional `E[Z]` in raw shaped-return units, PopArt ON, γ = 0.9999 | as above |

🚨 **The large QC scalar on arm S is the TREATMENT, not a defect** — hazard H5 registered it in
advance, and it reproduces arm S's own 2026-09-14 values exactly (75.12 / 119.06).

**ROW A** = (S-WP) − (W-V): head-vs-head, same readout family, a *diagnostic* on one side and the
*value function* on the other. Both columns are probabilities, so every row is defined — calibration
and the whole `gate.*` family live here. **ROW B** = (S-V) − (W-V): critic-vs-critic, the treatment's
own contrast, and 🚨 **RANK-BASED ROWS ONLY** — an AUC is invariant to a monotone rescaling of `V`,
but the calibration family regresses on `logit(V)` and every `gate.*` row is a Murphy decomposition
of a PROBABILITY forecast; neither is defined on a raw shaped-return column, and the scaffolding
gauge computes `gate.*` from `win_probs` regardless.

### 4.3 THE REGISTERED GUARD — `cond.opp_class_auc.t4_10`

Floor **0.02451** — the MAX of the two v6 two-pair floors (`hp800_floor_v6.json` 0.02189,
`hp800b_floor_v6.json` 0.02451), rule 3. 🚨 **Both are 10M, four-node-era, CONTROL-vs-CONTROL
floors, imported to 75M because there is no 75M floor and none is affordable** (§9.1).

| contrast | frame | arm S | arm W | **Δ (S − W)** | 95% CI | verdict |
|---|---|---|---|---|---|---|
| **ROW A**, `critic_read`'s own matched read (draw1) | **MATCHED 799/392/7** | 0.7686 | 0.7431 | **+0.0254** | [+0.0120, +0.0396] | **NOT DETECTED** — the CI does not clear the floor |
| ROW A, as traced, draw1 | as traced | 0.7671 | 0.7449 | +0.0223 | [+0.0087, +0.0357] | **NOT DETECTED** |
| ROW A, as traced, draw2 | as traced | 0.7494 | 0.7284 | +0.0211 | [+0.0066, +0.0353] | **NOT DETECTED** |
| **ROW B**, `critic_read` as traced (draw1) | as traced | 0.7721 | 0.7431 | **+0.0290** | [+0.0136, +0.0435] | **CANDIDATE on this draw only — see below** |
| ROW B, as traced, draw1 (`pair_critic_levels.py`) | as traced | 0.7683 | 0.7449 | +0.0236 | [+0.0102, +0.0370] | **NOT DETECTED** |
| ROW B, as traced, draw2 (`pair_critic_levels.py`) | as traced | 0.7579 | 0.7284 | **+0.0297** | [+0.0155, +0.0440] | clears the floor on THIS draw only — see below |

**Verdict on the registered guard: NOT DETECTED.** The label-bearing read is `critic_read`'s own
matched-frame ROW A, whose CI [+0.0120, +0.0396] straddles the 0.0245 floor rather than clearing it;
ROW A reproduces as NOT DETECTED on both independent draws and on the as-traced frame as well.

🚨 **ROW B has NO matched-frame reading, because the quota-match path ignores `--v-column`** —
hazard **H-L**, found here. `quota_match._scan` calls `conditioning_meters.conditioning_block(...)`
without the `v_column` argument, so every MATCHED row is computed on `win_probs` whatever the flag
says; running `critic_read --v-column values` returns a matched `cond.opp_class_auc.t4_10` delta
**byte-identical to the `win_probs` run's** (−0.0254, CI [−0.0396, −0.0120]) while its own
as-traced table correctly reads the shaped column. **ROW B is therefore reported AS TRACED only,
and labelled so.**

🚨 **And ROW B splits across the two draws**: +0.0236 (draw1) and +0.0297 (draw2) against a 0.0245
floor — inside it on one draw and outside it on the other, with `critic_read`'s independent
as-traced computation of the same draw-1 contrast landing at +0.0290. That is precisely the shape
rule 19 assigns to a row sitting AT its own eval-draw noise, and rule 21 says a candidate appearing
on only one of two draws is NOT CONFIRMED — never "refuted", since a small true effect both draws
are underpowered for fits equally well. **It must not carry a claim.** The honest statement is that
at turns 4–10 arm S's critic and arm W's critic rank opponent CLASS to within a difference this
instrument cannot resolve at 75M.

⚠️ **A third-decimal reconciliation, stated rather than smoothed.** Two independent computations of
the same as-traced S-V level differ slightly — this note's `pair_critic_levels.py` reads 0.7683 and
`critic_read` reads 0.7721 on draw1. The former reproduces arm S's own banked 2026-09-14 number
exactly (0.7683 / 0.7579), which is why it is the one quoted in the S-V column above. The gap does
not move any verdict: every version of the ROW B delta sits within 0.006 of the 0.0245 floor.

The sibling rows, for completeness: `cond.opp_class_auc.t1` reads ≈ 0.50 on every column and every
draw (0.4908–0.5206) — turn 1 knows nothing, the established shape — and `cond.opp_class_auc.t1_3`
reads Δ +0.0154 / +0.0076 on ROW A and +0.0292 / +0.0204 on ROW B against a 0.0153 floor, but its
own eval-draw spread is **0.0218–0.0227, i.e. AT or ABOVE that floor**, so this sibling cannot carry
a claim either — the same conclusion arm S's note reached about it.

### 4.4 The gate rows — where the two objectives DO come apart

`critic_read`'s own deltas, **arm − control (W − S)**, reported in that orientation because that is
the tool's; a POSITIVE `ece`/`reliability` delta means arm W is WORSE, a positive `resolution`/
`skill` delta means arm W is better. Every row below is **S-WP vs W-V — ROW A**, because the gate
family is undefined on arm S's shaped-return column.

| row | arm W | arm S | **Δ (W − S)** | 95% CI | floor (10M, imported) | verdict |
|---|---|---|---|---|---|---|
| `gate.resolution.all` | +0.0593 | +0.0551 | **+0.0043** | [−0.0021, +0.0108] | 0.0024 | **NOT DETECTED** (CI covers zero) |
| `gate.resolution.bot` | +0.0217 | +0.0189 | **+0.0028** | [−0.0020, +0.0084] | 0.0074 | **WITHIN FLOOR** |
| `gate.resolution.pool` | +0.0634 | +0.0588 | **+0.0046** | [−0.0043, +0.0134] | 0.0141 | **WITHIN FLOOR** |
| **`gate.ece.all`** | **+0.0688** | **+0.0178** | **+0.0510** | **[+0.0415, +0.0595]** | 0.0245 | 🚨 **DETECTED** |
| `gate.ece.bot` | +0.1338 | +0.0724 | **+0.0614** | [+0.0506, +0.0714] | 0.0449 | 🚨 **DETECTED** |
| `gate.ece.pool` | +0.0528 | +0.1185 | −0.0657 | [−0.0889, −0.0369] | 0.1097 | WITHIN FLOOR |
| `gate.reliability.all` | +0.0051 | +0.0005 | **+0.0046** | [+0.0035, +0.0060] | 0.0026 | 🚨 **DETECTED** |
| `gate.reliability.bot` | +0.0234 | +0.0082 | **+0.0152** | [+0.0120, +0.0186] | 0.0053 | 🚨 **DETECTED** |
| `gate.skill.all` | +0.2980 | +0.3086 | −0.0106 | [−0.0394, +0.0177] | 0.0454 | WITHIN FLOOR |
| `gate.skill.bot` | −0.0172 | +0.1219 | −0.1391 | [−0.2209, −0.0576] | 0.1004 | **NOT DETECTED** (CI does not clear the floor) |
| calibration SLOPE, all states | +1.3504 | +1.3463 | **+0.0041** | [−0.0896, +0.0906] | 0.2001 | WITHIN FLOOR |
| calibration INTERCEPT, all states | +0.5804 | +0.0335 | +0.5469 | [+0.4404, +0.6550] | 1.0247 | WITHIN FLOOR |
| calibration slope, `turn 1-3` | +0.7333 | +0.9992 | −0.2659 | [−0.4167, −0.1225] | 0.1915 | **NOT DETECTED** (CI does not clear the floor) |

**Every one of those DETECTED rows reproduces on the second independent draw** (`gate.ece.all`
+0.0507 / +0.0551; `gate.ece.bot` +0.0614 / +0.0588; `gate.reliability.all` +0.0047 / +0.0052 — all
in the W-worse direction, all clear of the imported floor on both draws), and the per-arm eval-draw
spreads on those rows are tiny (0.0008–0.0028) against floors of 0.0245–0.0449.

🚨 **The dissociation, stated carefully.** **RESOLUTION is NOT DETECTED apart on any stratum — the
two heads separate outcomes about equally well — while CALIBRATION comes apart decisively: arm W's
`V` has ~4× arm S's auxiliary head's ECE overall (0.069 vs 0.018) and ~2× against bots, with a
calibration-in-the-large intercept of +0.58 against +0.03.** Both heads have essentially the same
dispersion (calibration slope 1.350 vs 1.346, Δ inside the floor); what differs is the LEVEL — arm W
is systematically pessimistic (`V − p̂` = −0.110 against arm S's −0.049 on the ipw-weighted identity
row), so its probabilities are shifted rather than mis-spread.

⚠️ **And the comparison is a CRITIC against a DIAGNOSTIC, which is the whole reason ROW A must be
labelled.** Arm S's `win_probs` head carries 5 % of the value weight, is trained by a pure BCE
against the terminal outcome, and is *not* the baseline any advantage is taken against. Arm W's head
IS the value function: it is simultaneously fit to the outcome and consumed by GAE, and `--vf-coef
0.5` multiplies its BCE rather than a PopArt-normalised MSE (hazard H4 — matched in number, not in
units, and deliberately not re-tuned). **So "the win-prob critic is worse calibrated than the shaped
arm's spare head" is the measurement; "a win-prob critic is worse calibrated than a shaped critic"
is NOT, and cannot be had from this pair** — arm S's actual critic lives on a shaped-return scale
where ECE is not defined.

### 4.5 The eval-draw spread — the only floor proxy this read owns

Two independent draws of the same checkpoint (rule 19: two draws BOUND a floor at the eval-draw
level; they never give it a CI).

| row | arm W (W-V) | arm S (S-WP) | arm S (S-V) | vs the imported floor |
|---|---|---|---|---|
| `cond.opp_class_auc.t4_10` | **0.0165** | **0.0177** | **0.0104** | all **inside** 0.0245 |
| `cond.opp_class_auc.t1_3` | 0.0142 | 0.0218 | 0.0227 | at/above the 0.0153 floor — **this sibling cannot carry a claim** |
| `cond.opp_class_auc.t1` | 0.0185 | 0.0221 | 0.0126 | at/inside 0.0227 |
| `gate.resolution.all` | 0.0014 | 0.0031 | — | above the 0.0024 floor on arm S |
| `gate.resolution.bot` | 0.0032 | 0.0024 | — | inside 0.0074 |
| `gate.ece.all` | 0.0028 | 0.0024 | — | far inside 0.0245 |
| `gate.skill.bot` | 0.0358 | 0.0199 | — | inside 0.1004 |
| calibration slope, all | 0.0008 | 0.0354 | 0.0020 | far inside 0.2001 |

**The decision row is stable across draws on all three columns** — which is what makes it usable —
and it is stable enough that its NOT DETECTED is a statement about the two arms rather than about
the instrument.

---

## 5. EXTERNAL ANCHOR 1 — Metamon `SmallRL`, three cells, each quoted inside its own regime

**Harness:** arm S's own cell scripts, reused **verbatim** for the two HOME cells — same
`run_cell.sh`, same half-cell structure, **the same team seeds** (`BASE_OURS 20260914`,
`BASE_THEIRS 20270914`), so the two arms drew the identical team pairs — on our own pinned Showdown
at **:9450**. The AWAY cell is the SOP's tier-B row and was taken on BOTH arms through the tool of
record, `python -m main.anchors --opponent metamon:SmallRL --regime greedy --teamset away
--games 100`, which starts and stops its own server (:9500) and verifies the regime per decision.

Metamon `@0a00a759`, conda env `metamon`, torch CPU, `OMP_NUM_THREADS=1`, `VanillaAttention`,
`SmallRL` ckpt 40 (13.9M). Our side is each arm's `final_model.zip` @ **75,005,952**, CPU,
through `main.play`'s own client path. 100 games per cell, split into two 50-game role-balanced
half-cells (Showdown makes the challenger p1, and p1/p2 is not a priori neutral).

### 5.1 The three-way table — arm S, arm W, and the 75M win-prob run, each anchor in its own regime

| cell | **arm S** | **arm W** | Δ (S − W), Newcombe 95% | the 0.02 leg (75M win-prob), same cell | verdict on the PAIR row |
|---|---|---|---|---|---|
| **greedy · home** (both sides greedy, our 719-team pool) | **0.630** [0.532, 0.718] | **0.650** [0.553, 0.736] | **−0.020** [−0.151, +0.111] | **0.520** [0.423, 0.615] | **NOT DETECTED** |
| **greedy · away** (both greedy, Metamon's own 20-team `competitive` set) | **0.520** [0.423, 0.615] | **0.500** [0.404, 0.596] | **+0.020** [−0.117, +0.155] | not measured in this cell | **NOT DETECTED** |
| **mixed · home** (ours greedy, Metamon at ITS OWN eval default t = 1.0) | **0.840** [0.756, 0.899] | **0.760** [0.668, 0.833] | **+0.080** [−0.032, +0.190] | **0.742** [0.657, 0.812] (n = 120) | **NOT DETECTED** |

🚨 **The three rows are NOT comparable to each other**, and two separate reasons say so.
(i) The 21-pp gap between a greedy row and a mixed row is the *opponent's* regime moving, not ours
— the matched-regime 2×2 convicted "temperature 1.0" as not a comparable setting across models
(`SmallRL` plays its own argmax only ~64.7 % of the time at t = 1.0), and this campaign reproduces
that: `argmax_match_rate` **1.0000** in every greedy half-cell and **0.6408** in the sampling one.
(ii) The home and away team sets disagreed in SIGN for `SmallRL` before and they disagree here too:
both arms drop ~13–15 pp going from home to away. The away set is the one **no dial of ours is set
to**, which is why the SOP says report both, always.

Per half-cell (role balance), arm W: greedy·home **0.66** (Metamon challenges) / **0.64** (we
challenge); mixed·home **0.78** / **0.74**; greedy·away **0.52** (peer challenges) / **0.48** (we
challenge). Arm S's were greedy·home 0.66 / 0.60 and mixed·home 0.90 / 0.78. **Neither arm shows a
role imbalance large enough to matter at these n**, which is the reason the cells are role-balanced
in the first place.

### 5.2 The regime was VERIFIED per decision, not assumed

| cell / half | our `stochastic` kwarg | our temperature | Metamon `sample` kwarg | Metamon `argmax_match_rate` |
|---|---|---|---|---|
| greedy·home · Metamon challenges | `[false]` | 0.0 | `[false]` | **1.0000** |
| greedy·home · we challenge | `[false]` | 0.0 | `[false]` | **1.0000** |
| mixed·home · we challenge | `[false]` | 0.0 | `[true]` | **0.6408** |
| mixed·home · Metamon challenges | `[false]` | 0.0 | *n/a* — hazard **M1**, below | *n/a* |
| greedy·away · both halves (`main.anchors`) | greedy (stamped per row) | 0.0 | greedy | **1.0000** |

The instrument has power: it reads 1.0000 **exactly** in every greedy half-cell and materially below
1.0 in the sampling one — inside the 2×2's 0.623–0.680 band for `SmallRL`. Ties count in the
denominator and not the numerator; **1 tie in 300 home games, 0 in 200 away games.**

### 5.3 Hazards — and every one of arm S's Metamon hazards REPRODUCED

* **M1 — the `mixed · Metamon challenges` half-cell lost its Metamon-side record again**, with the
  identical signature: a `RecursionError` out of Metamon's own `metamon_to_amago.py::step`
  (988 traceback frames), *after* its last game. **Our side's 50 results are complete** and the win
  rate above is ours. 🚨 **This is now the THIRD occurrence** (the de-risk's H-B, arm S's M1, and
  this one) and all three are in the same cell — *ours greedy, Metamon sampling, Metamon
  challenging*. A failure that recurs in one cell across three independent campaigns is a property
  of that configuration, not noise.
* **M2 — the positional join in that half-cell is therefore UNRELIABLE.** `sides_disagree` = 6 and
  `metamon_row_missing` = 2 there; **every other half-cell reads `sides_disagree` = 0.** Only OUR
  side's record feeds the win rate.
* **M3 — one 250-turn forfeit fired** in that same half-cell (1 of 300 home games, 0.3 %) and one in
  the arm-S away cell (1 of 100) — far under rule 12's 25 % INCONCLUSIVE threshold, bucketed as a
  loss on our side's record.
* **M4 — two of our own tracebacks** (`Decision context is missing at turn 1. get_mask() /
  embed_battle() must run before action_to_order()`) appeared in that half-cell, coincident with
  Metamon's crash storm — again exactly as on arm S. No game was lost (50/50 recorded).
* **M5 — the `production` registry name still cannot be used** (resolves to
  `ai_v9_21_gen17_pfspoff_0820` @ `config_version` 97, which no longer loads at HEAD). Every
  comparator here is an explicit `.zip`.

---

## 6. THE UNTAUGHT METER at run end (§8.5) — the one row that moves past every floor in evidence

`python -m main.untaught_meter` — the win rate of a checkpoint **piloting** a fixed 8-team slice it
never trained on, against ONE fixed opponent, cluster-bootstrapped over TEAMS. Offline, CPU, no
server; nothing was written under `models/`.

The opponent is the **registry name** `untaught_meter_opponent`, which the tool resolved and
stamped: `ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` @ 24,000,000 steps. 200
games/team, seed 0, **concurrency 1** (above 1 is REFUSED — seeded at concurrency 3 two runs of the
same measurement have differed by +0.043 in level). **Both arms were measured in ONE invocation**,
so they saw the identical 8 teams and the identical games. **3,200 battles per config, 0 timeouts**
(rule 12's INCONCLUSIVE threshold is 25 %).

### Levels — identical under BOTH config resolutions, on both arms

| ref | registry `untaught_meter_config` | `--config auto` | CI95 | wins / finished | timeouts |
|---|---:|---:|---|---:|---:|
| **arm S** | **54.50 pp** | **54.50 pp** | [52.25, 56.62] | 872 / 1600 | 0 |
| **arm W** | **46.19 pp** | **46.19 pp** | [42.81, 49.12] | 739 / 1600 | 0 |
| the 0.02 leg (banked 2026-09-14) | **58.25 pp** | **58.25 pp** | [56.44, 60.44] | 932 / 1600 | 0 |

**The two resolutions agree to the last win on every ref**, which is the condition §8.5 set for the
level counting as a level — neither arm is architecture-identical to the meter's registry config, so
a level that existed under only one resolution would not be a level. ✅ **And arm S reproduces its
banked 2026-09-14 level EXACTLY (Δ = 0.00 pp, 872/1600 both times)** — an independent confirmation
that the meter is deterministic at seed 0 / concurrency 1, which is what licenses joining the 0.02
leg in from the earlier file.

### The paired contrasts (rule 10: the team is the unit)

One resampled team index set per draw, shared by both refs — the paired convention that removes the
team-difficulty component the two refs share. 20,000 draws.

| contrast | Δ (pp) | CI95 (pp) | teams favouring the first | reading |
|---|---:|---|---:|---|
| **arm S − arm W** (THE PAIR) | **+8.31** | **[+5.69, +11.19]** | **8 of 8** | past **every** floor in evidence, including the 4.27 pp controller-live one |
| the 0.02 leg − arm W | **+12.06** | [+8.81, +15.50] | — | the ENTROPY-COEFFICIENT axis on a matched objective, confounded by pin |
| arm S − the 0.02 leg | **−3.75** | [−6.37, −1.12] | — | reproduces arm S's own 2026-09-14 number exactly |

| team | `U_61590463` | `U_92832108` | `U_ce35b736` | `U_9909f2e9` | `U_9d5f8458` | `U_f7ba5702` | `U_90b94599` | `U_dbf81d8e` |
|---|---|---|---|---|---|---|---|---|
| arm S | 56.50 | 49.50 | 56.00 | 54.50 | 55.50 | 55.00 | 49.50 | 59.50 |
| **arm W** | **48.50** | **39.00** | **49.50** | **38.00** | **47.00** | **48.00** | **48.00** | **51.50** |
| the 0.02 leg | 61.00 | 59.00 | 64.50 | 55.50 | 57.50 | 57.00 | 55.00 | 56.50 |

### What may and may not be said about this row

* 🚨 **It is a DESCRIPTOR, not an endpoint** — the registration says so in advance (§8.5), because
  the meter's axes and floors are established at **~1M-step FOLD depths** and *neither of these arms
  is a fold*. The floors quoted (1.19 pp END-depth replicate, 1.66 pp frozen-dose, 4.27 pp
  controller-live, 1.00 pp G5 continuation) are all fold floors, quoted rather than applied.
* **Within that framing it is the largest and cleanest movement in the whole read**: +8.31 pp with
  8 of 8 teams in the same direction, a CI well clear of zero, and a magnitude ~2× the widest fold
  floor in evidence. Under rule 22 that is a **run-level CANDIDATE at n = 1 per arm**, and the named
  next increment is a seed replicate of arm W.
* **On this row the pair AGREES in sign with the ladder** (arm S ahead on both) — unlike arm S's own
  2026-09-14 read, where the ladder and the meter pointed opposite ways against the 0.02 leg. That
  disagreement survives here: **the 0.02 leg is LOWEST of the three on the ladder (1984.2 refit) and
  HIGHEST of the three on the untaught meter (58.25 pp)**, which is a standing reminder that the two
  instruments are not measuring one quantity. Arm W is the only one of the three that is *low on
  both*.
* ⚠️ **Arm W sits below 50 % against a 24M-step opponent from two generations ago** (46.19 pp,
  CI upper bound 49.12) while arm S sits above it. On the dense ladder the same two arms are 17.5
  Elo apart with a CI covering zero. **Both statements are true of the same pair of checkpoints**;
  which one generalises is exactly what one seed per arm cannot say.

---

## 7. EXTERNAL ANCHOR 2 — Foul Play (search) at `--search-time-ms 1000`

**Harness:** arm S's own Foul Play scripts, verbatim but for the model, the usernames and the
output root. Foul Play `@6c467c08` + poke-engine `0.0.48` built `--features poke-engine/gen3
--no-default-features`, its own conda env, `--search-time-ms 1000 --search-parallelism 1
--search-threads 1`. Our side is arm W's `final_model.zip` @ **75,005,952**, CPU, through
`main.play`'s own client path, on our pinned Showdown at **:9417**. 8 sessions × 10 games, our side
pinning **the same 8 pool teams arm S used** (same sorted export, same stride), Foul Play drawing a
random pool team every battle from the same export.

**Our win rate 0.475 (38/80), Wilson 95 % [0.369, 0.583].** Mean 37.3 turns, median 32, max 119;
**0/80 reached the 250-turn forfeit**, 0 error lines on Foul Play's side, 0 games without a matched
Foul Play log.

| session | our team | games | our wins | win rate | mean turns | our think ms | FP think s | **FP visits/decision** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `009e3d0244` | 10 | 5 | 0.50 | 21.4 | 599 | 2.20 | 1.33 M |
| 1 | `4239fc5ba2` | 10 | 4 | 0.40 | 36.7 | 524 | 2.15 | 1.23 M |
| 2 | `64a691c473` | 10 | 6 | 0.60 | 31.4 | 413 | 2.07 | 1.27 M |
| 3 | `7d0337af97` | 10 | 6 | 0.60 | 34.1 | 465 | 2.04 | 1.00 M |
| 4 | `9b454d9ea7` | 10 | 3 | 0.30 | 57.6 | 603 | 1.95 | 0.86 M |
| 5 | `a185b2d193` | 10 | 3 | 0.30 | 32.0 | 519 | 2.09 | 1.09 M |
| 6 | `b904dbe059` | 10 | 6 | 0.60 | 41.8 | 558 | 1.98 | 1.14 M |
| 7 | `e11829f0561ef5a9` | 10 | 5 | 0.50 | 43.1 | 110 | 2.06 | 1.31 M |
| **all** | 8 pool teams | **80** | **38** | **0.475** | **37.3** | **474** | **2.07** | **1.15 M** |

### 🚨 Rule 23 — this is a WIDTH meter, and the three campaigns are not width-matched

Foul Play has **NO iteration budget**: `--search-time-ms` is wall clock only
(`fp/search/main.py:53`). So the opponent's strength is set by the box, and every read must carry
its realized visit count.

| campaign | our win rate | Wilson 95 % | **realized FP visits/decision** | session range | our think ms |
|---|---|---|---:|---|---:|
| the 0.02 leg (de-risk, 2026-09-14) | **0.388** | [0.288, 0.497] | **1.40 M** | 1.21–1.53 M | ~37 |
| **arm S** (2026-09-14) | **0.450** | [0.346, 0.559] | **1.249 M** | 1.05–1.65 M | 481 |
| **arm W** (2026-09-16) | **0.475** | [0.369, 0.583] | **1.153 M** | 0.86–1.33 M | 474 |

* **Δ (S − W) = −0.025, Newcombe 95 % [−0.175, +0.127] — NOT DETECTED**, and never "equal" (rule 6).
* **The widths are NOT matched: arm W's opponent searched 7.7 % NARROWER than arm S's** (ratio
  0.923), i.e. a weaker opponent, **in the direction that flatters arm W** — the mirror image of the
  arm-S campaign, whose opponent was 11 % narrower than the de-risk's and flattered arm S.
* 🚨 **The tell that this row is measuring the box.** Across the three campaigns the win rates order
  **exactly inversely to the realized search width**: 1.40 M → 0.388, 1.249 M → 0.450, 1.153 M →
  0.475. A monotone relation across three points is not proof, but it is exactly what rule 23
  predicts a width meter does when the machine's load moves under it, and it is the reason **no
  difference between these three numbers may be attributed to a model.**
* The honest statement: **at a 1000 ms nominal budget all three 75M policies sit BELOW Foul Play**
  (every Wilson interval's upper bound is at or under 0.583), **by a margin these experiments cannot
  separate**, and the two external anchors continue to bracket our policies — Metamon's search-free
  `SmallRL` below us, Foul Play's search above.
* The box this campaign ran on carried **the live `ai_v13_03_fork` training arm** plus the arm-W
  trace draws and the untaught meter; load average 15–25, against the arm-S campaign's 26–37 and the
  de-risk's 22.6. **The lighter load is why arm W's own decision time (474 ms) matched arm S's
  (481 ms) while Foul Play's visit count fell** — Foul Play is the side whose budget is a clock.

---

## 8. THE OPERATIONAL ROWS — read from the arms, not from the ledger's summary

Registration §8.4b promoted `rollout/ep_len_mean` and the stall/timeout fraction to PRIMARY on arm
W, by the `[CRITIC]` banner's own instruction (*a `[0,1]` critic cannot express "a timeout is worse
than a loss"*). The G7 and stall rows were banked at completion
[ledger 2026-09-16 · *`ai_v13_02_flywheel_winprob` (ARM W) COMPLETE*]; `rollout/ep_len_mean` is
re-read here from the TensorBoard events on both arms, on the matched post-crossing span.

| row | arm S | arm W | note |
|---|---|---|---|
| `rollout/ep_len_mean`, post-crossing mean (4.13M → 75.0M) | **40.48** | **44.53** | +4.05 turns on arm W |
| … post-crossing max | 53.32 | **63.92** | |
| … median-20 at the end | 43.31 | 45.43 | |
| … last value | 46.51 | 40.90 | a single point; not a measurement |
| G7 worst ratio (banked) | 1.070 (85.6 % of bar) | 1.131 (90.5 % of bar) | both **under bar, both halves**, off in-family references |
| stall half peak / last (banked) | 0.0237 / 0.0046 | 0.0145 / 0.0047 | |
| `bots8` final (banked) | 0.9188 | 0.9100 | |

**Reading, under rule 12 (a timeout is never a semantic outcome).** Arm W's rollout episodes are
**~4 turns longer on average and its worst post-crossing excursion is 10.6 turns higher**, which is
the direction the mode's own banner warned about — and **neither arm came anywhere near the 25 %
inconclusive threshold, neither breached G7, and arm W's stall half-peak is in fact LOWER than arm
S's.** So the warning's mechanism is visible as a shift in episode length and is *not* visible as a
pathology. `--arm-no-progress-tax`, the registered contingency, stays OFF and there is nothing here
that argues for turning it on. ⚠️ `rollout/ep_len_mean` is ~90 % self-play on both arms and is
never comparable to the bot-eval series; the S-vs-W comparison above is rollout-vs-rollout, which
is matched, and no bar attaches to it.

---

## 9. HAZARDS — every one a finding

| # | hazard | why it matters, and what was done |
|---|---|---|
| **H-A** | ✅ **Rule 24 satisfied on BOTH sides of the pair.** A committed `ladder.json` carries the RECIPE it was fitted with, and arm S's read found +73 Elo of stale-recipe inflation in `ai_v12_02`'s file. | Both arms' committed files carry the recipe stamp (`eval_sentinel_edges_dropped` 51 / 48) and **reproduce their refits to 0.0**. The third leg is REFIT everywhere it appears and its committed 2057.3 is never quoted. §2.1 |
| **H-B** | ✅ **The critic-row step deviation is MATCHED this time.** Arm S's note flagged that its critic rows sat at 74,000,016 rather than the final 75,005,952, and asked the pair read to check the two arms were comparably far from their ends. | They are at the **same step**, 1,005,936 (1.34 %) short on both. Neither run has an eval cycle at its final step; `eval_trace_gen` refuses a step with no weights of its own. §4.1 |
| **H-C** | ✅ **The control the arm-S read could not have now exists.** `critic_read` REFUSED arm S's only available control on two counts (regime and game count). | The pair is comparable by construction — same offline full-capture spec, same 800 games × 12 opponents, same recorded GREEDY regime, same two seeds — and **the tool accepted it and produced the registered pair report.** §4.1 |
| **H-D** | 🚨 **"Read arm S both ways" is only partly possible, and the limit is structural** — reconfirmed. `cond.opp_class_auc.*` is an AUC, invariant to a monotone rescaling of `V`, so it reads on `values` and `win_probs` alike. The calibration family regresses on `logit(V)` and every `gate.*` row is a Murphy decomposition of a PROBABILITY forecast, and the gauge computes `gate.*` from `win_probs` unconditionally. | ROW A (head-vs-head) carries every row; ROW B (critic-vs-critic) is rank-based rows ONLY, and is marked so wherever it appears. §4.2 |
| **H-E** | 🚨 **Foul Play is a WIDTH meter and the three campaigns are NOT width-matched — this time in arm W's favour.** Arm W's opponent searched **1.153 M** visits/decision against arm S's **1.249 M** (−7.7 %) and the de-risk's **1.40 M**, because this campaign ran at load 15–25 against arm S's 26–37. | Rule 23. Across the three campaigns the win rates order **exactly inversely to realized width** (1.40 M → 0.388, 1.249 M → 0.450, 1.153 M → 0.475). **No difference among the three may be attributed to a model.** §7 |
| **H-F** | 🚨 **THE REALIZED DOSE CAME OUT UNMATCHED, and no arm-to-arm number for it existed before.** The four dose inputs are identical tokens in both argvs, so the dose is equal BY DECLARATION (§5) — but the KL controller annealed the two arms differently: arm S's `lr_median` **4.32e-4**, arm W's **3.00e-4**, a **1.44× realized-dose gap in arm S's favour** (6.592e-08 vs 4.578e-08 by `python -m main.dose`). | The registration anticipated the mechanism in words but not the size. Step size here is downstream of the objective, in the same class as `gamma` and PopArt — **FORCED, not chosen** — so it is not a confound in the design sense; it IS a reason no row on which arm S leads may be read as if the optimiser had been held fixed. §1 |
| **H-G** | 🚨 **The arm-S note's late-slope row for its two COMPARATOR runs does not reproduce from its own committed JSON.** `flywheel_armS_reads_2026-09-14/README.md` §2.4 prints the 0.02 leg's late slope as `+0.15 ± 1.23` and the v8 control's as `+0.32 ± 0.11`; its own `out/strength_read.json` says **−0.32 ± 1.4378** and **−0.1044 ± 0.1476**, and a fresh refit here reproduces the JSON to the digit. | **Arm S's OWN slope row is unaffected and reproduces exactly** (+1.1435 ± 0.4357 over 48–68M), so the arm-S headline stands. What is wrong is a comparator row, and this note uses the JSON-reproducible numbers and says so rather than copying the table forward. §2.5 |
| **H-H** | 🚨 **The Metamon `mixed · Metamon challenges` half-cell failed for the THIRD time, in the same cell, with the same signature** — a `RecursionError` out of Metamon's own `metamon_to_amago.py::step` (988 frames) *after* its last game (the de-risk's H-B, arm S's M1, and now this). A failure that recurs in one configuration across three independent campaigns is a property of that configuration, not noise. | Our side's 50 results are complete and feed the win rate; that half-cell's positional join is UNRELIABLE (`sides_disagree` 6, `metamon_row_missing` 2) while **every other half-cell reads `sides_disagree` = 0**. Also 1 forfeit in 300 home games and two of our own `Decision context is missing at turn 1` tracebacks, again in that cell. §5.3 |
| **H-I** | ⚠️ **The two arms' three pool SENTINELS are different checkpoints.** Each arm's offline draw faces three of its OWN snapshots — arm W's at 26.0M / 46.0M / 72.0M, arm S's at 22.0M / 42.0M / 72.0M — because the retained pools differ. | This is the instrument's design (a sentinel is the run's own snapshot), so the frames are matched in KIND but not in the identity of three of twelve opponents. It is recorded because every conditioning row is a statistic OF the traced frame, and the `pool` stratum is the one it touches. |
| **H-J** | ⚠️ **The pair's ladder node sets overlap in only 16 of 20 steps**, because arm W's retained pool starts 4M later. A matched-COUNT table therefore compares arm S's node *k* against arm W's node *k* at a DIFFERENT step for 19 of 20 ordinals. | The registration's §8.1 correction 1 names exactly this. The **COMMON-STEP refit** (§2.3) is the honest read and is reported beside the matched-count one; 16 ≥ the registered n ≥ 12 report floor. |
| **H-K** | ⚠️ **`critic_read`'s DELTA orientation is arm − control (W − S); every other row in this note is S − W.** Mixing the two silently would flip a sign. | §4.4's table is printed in the tool's own orientation and labelled in its header; §0 and §4.3 are in S − W. Both are stated wherever a number appears. |
| **H-L** | 🚨 **A TOOL DEFECT FOUND BY THIS READ: `--v-column` does not reach the quota-matched rows.** `main.ops.quota_match._scan` and its bootstrap pass call `conditioning_meters.conditioning_block(run_dir, step, boot=…, seed=…, ladder="off", frame=(sarr, smeta))` **without `v_column`**, so it defaults to `win_probs`. On a `--critic shaped` side that is the AUXILIARY head, not the critic. `critic_read --v-column values` therefore returns MATCHED rows byte-identical to the `win_probs` run (`cond.opp_class_auc.t4_10` Δ −0.0254, CI [−0.0396, −0.0120] in both) while its own AS-TRACED table correctly reads the shaped column — the two halves of one report disagree about which tensor `V` is, and nothing in the output says so. | **The failure is silent and it is label-bearing**: the frame-sensitive rows take their verdict from the MATCHED delta by design. **ROW B is reported AS TRACED only and marked so**; no matched-frame critic-vs-critic number is quoted anywhere in this note. The fix is one keyword argument in two call sites, and it belongs on the tech-debt backlog rather than in this read — nothing under `src/` was changed here. The `win_probs` default means **every banked read is unaffected**, because they all wanted `win_probs`. §4.3 |
| **(banked)** | Arm W's crash-at-teardown — SIGTERM during the final eval's shutdown, after `Training complete` and after the full final evaluation; the FIFTH occurrence. | **Already banked** [ledger 2026-09-16 · *ARM W COMPLETE*]; not re-investigated here. |

### What was NOT done, and why

* **The mirror battery was not run.** Rule 25 retired the 100-pair cells, and a ≥ 400-pair leaf read
  is a separate job. The registration's §6 exclusion stands.
* **No seed replicate of either arm**, and none is affordable (~35 GPU-h each). Rule 22 binds on
  every row: a single-arm direction is a CANDIDATE, never a family verdict.
* **`critic_read` was not paid for twice on the identity half.** The registered pair report was
  produced once per `--v-column` (`win_probs` for ROW A, `values` for ROW B); the second invocation
  REUSED both runs' cached identity and gate readouts — `_cond_fingerprint` is deliberately separate
  from `_fingerprint`, so only the conditioning block recomputed. The per-draw levels and the
  draw-2 deltas come from the same library functions `critic_read` itself calls.
* **No claim about the critic objective in general.** Every row is one arm against one arm at n = 1.

---

## 10. What is in this directory

| path | what |
|---|---|
| `README.md` | this note |
| `scripts/pair_strength_read.py` | the ladder read for all four runs: committed-vs-refit (rule 24) on each, the resolution curve, the slopes with the newest node dropped, the adjacent-node spread, the cross-run deltas, **the COMMON-STEP refit** and the node-by-node table |
| `scripts/pair_entropy_read.py` | `train/entropy_loss` from the TB events on both arms, sign-corrected to H, median-20, each arm's crossing, the slopes, and the MATCHED-STEP S-vs-W buckets |
| `scripts/pair_critic_levels.py` | the critic rows on both arms and both draws: `conditioning_block(v_column=…)` + `gate_block`, ROW A and ROW B with the delta's own CI from the two independent bootstraps, and the eval-draw spread |
| `scripts/pair_untaught_delta.py` | the paired team-clustered contrasts off the meter's own per-team rows, under both config resolutions, with the 0.02 leg joined from arm S's banked run |
| `scripts/pair_metamon_cells.py` | arm W's Metamon half-cells, Wilson + Newcombe, against arm S and the 0.02 leg, each in its own regime |
| `scripts/pair_away_anchor.py` | the AWAY-set cell on both arms through `python -m main.anchors` |
| `scripts/pair_foulplay.py` | the Foul Play campaign summary and the **rule-23 width comparison** against arm S |
| `scripts/harness/` | the run scripts as executed: the arm-W draw generator, the untaught driver, the Metamon cell runner (arm S's `run_cell.sh`, verbatim) and arm W's plan, the away-cell driver, the Foul Play session/campaign scripts, and the `critic_read` pair invocation |
| `out/pair_strength_read.json` | every ladder number, all four runs |
| `out/pair_entropy_read.json` | every entropy number, all four runs, plus the matched-step buckets |
| `out/pair_critic_levels.json` | W-V / S-WP / S-V levels on both draws, the frames, ROW A and ROW B deltas, the eval-draw spread |
| `out/critic_read_pair/` | `main.ops.critic_read`'s own report for arm W (ARM) vs arm S (CONTROL) at `--v-column win_probs` |
| `out/pair_untaught_delta.json`, `out/untaught/` | the meter under both config resolutions and every paired contrast |
| `out/metamon/` | `armW_cells.json`, the merged `games.jsonl`, the harness `summary.json` with the per-decision regime instruments |
| `out/away/` | the two `main.anchors` away cells (`summary.json` + `games.jsonl` per arm) and `pair_away_anchor.json` |
| `out/foul_play/` | the merged `games.jsonl`, `summary.json`, `session_table.md` |
| `out/traces_manifests/` | the two arm-W draw manifests and the generator log |

**No code under `src/` was changed by this read.** The one tool change the registration named —
`--v-column {win_probs,values}` on `main.ops.critic_read` — was built and landed with arm S's read
on 2026-09-14 and is used here as it stands.

---

## 11. Ledger paragraph — ready to append (nothing in `ledger.md`, `UNDERSTANDING.md` or the registration was edited from here)

> ### 2026-09-16 · MEASUREMENT (MAJOR) · THE FLYWHEEL-ERA PAIR READ — `--critic shaped` vs `--critic winprob` at 75M on one pin, one dose-by-declaration, one seed slot and one eval regime: **NOT DETECTED on strength (+17.5 Elo, claimable ≈ 69), on the registered critic guard, and on all four external-anchor cells**; the rows that DO move are ones the registration had already ruled could not be endpoints — entropy **FLAT vs DECAYING**, the untaught meter **+8.31 pp to arm S (8 of 8 teams)**, and **~4× the ECE on arm W's critic at indistinguishable resolution**; plus a **realized-dose gap of 1.44× the registration did not price** and a **tool defect that makes `--v-column` silently miss every quota-matched row**
>
> `designs/research_state/measurements/flywheel_pair_read_2026-09-15/`. **THE PAIR.**
> `ai_v13_01_flywheel_shaped` (**arm S — the era's SHAPED reward/critic composition transplanted onto
> the current pin**) and `ai_v13_02_flywheel_winprob` (**arm W — the WIN-PROB critic, V(s) = P(win|s)
> with the value loss the win-prob head's BCE**), both COMPLETE at **75,005,952**, both pin
> `6eb9c776` for all 75M steps (`pin_history` one row each), both `role: fresh`, both config v119 /
> `gen3_critic_route_wave_v1`, both `--seed 1001` / `--ent-coef 0.05` /
> `--eval-sentinel-greedy` (RECORDED) / `--no-value-true-team`, both crossed self-play at the
> **identical 4,128,768**, both 20 ladder nodes. Everything CPU-only, `nice`, from the main checkout,
> nothing written under `models/`; the live `ai_v13_03_fork` arm held the GPU throughout and was not
> touched; Showdown on :9450 / :9417 / :9500 started and stopped by their own PIDs, **:8000 and
> :8001 untouched**. **STRENGTH (the primary endpoint).** Both committed `ladder.json` files carry
> the recipe stamp (`eval_sentinel_edges_dropped` 51 / 48) and **reproduce their current-code refits
> to 0.0** — rule 24 satisfied on both sides, unlike the third leg. Headline at matched COUNT (20
> nodes), newest node **at the same 72,000,000 on both**: arm S **2036.6** (se 8.9), arm W **2019.1**
> (se 8.7), **Δ(S − W) = +17.5 [−6.9, +41.9]**; second-newest +10.2 [−14.2, +34.6]. se(Δ) = 12.45,
> CI95 ±24.4, so the smallest claimable |Δ| against the imported 45.0 Elo floor is **69.4** ⇒
> **NOT DETECTED, never "equivalent"** (rule 6: ±24.4 inside a floor imported from 10M four-node
> depth makes the equivalence clause unavailable). 🚨 **The registration's §8.1 correction 1 bites:
> the two node sets share only 16 of 20 STEPS**, so the matched-COUNT table compares different steps
> at 19 of 20 ordinals; the **COMMON-STEP refit** (16 nodes ≥ the registered n ≥ 12 report floor)
> gives **arm S 2032.3 vs arm W 2025.6, Δ +6.7 [−20.2, +33.6], NOT DETECTED**. **The SHAPE the
> registration asked to see is there: arm S leads at ALL 16 shared steps, by +25 to +57 Elo through
> 26–42M, narrowing to +6.7 at the shared 72.0M node** — no single gap claimable (smallest claimable
> 71.9 on that fit), the 16 points not independent, reported as a description. Late slopes, newest
> node dropped, **no bar attaches**: arm S **+1.14 ± 0.44** Elo/M (48–68M, t = 2.62), arm W
> **+0.78 ± 1.33** (52–70M, t = 0.59) with its steepest segment in the MIDDLE third
> (+3.31 ± 0.83); both still gaining at 75M, neither late−middle difference clear of zero. Third leg
> `ai_v12_02_winprob_critic` **REFIT** (1984.2; its committed 2057.3 never quoted): arm S +52.4
> [+28.3, +76.5], arm W +34.9 [+10.9, +58.7], **both NOT DETECTED**, both carrying pin and
> `--ent-coef` confounds, ladder-only across the 2026-09-07 regime boundary. v8 line **SHAPE ONLY**
> (rule 4) and the bot frame is precise-and-wrong near the frontier (rule 6): its own late third
> refits to **−0.10 ± 0.15** over 210–242M, not distinguishable from flat. **ENTROPY.** Both spans
> verified own (741 / 740 points, 196,608 → 75,005,952), both crossings at 4,128,768 so every
> post-crossing window is genuinely matched. H_end (median-20) **1.0292 (S) vs 1.0648 (W)**, Δ
> −0.0356 — **inside the 0.074-nat three-seed floor, NOT READ**; both arms' distances from `ent05`'s
> 1.0861 (−0.057 / −0.021) and from v8's 1.07–1.11 plateau (−0.041 / −0.005) are inside it too, so
> **the pre-registered prediction is not contradicted by either arm**. 🚨 **The row that is a
> FINDING (§8.3 makes an H difference a finding, never a bar): the TRAJECTORIES differ. Arm S holds
> entropy FLAT across the whole post-crossing span — −3e-5 ± 5e-5 nats/M, t = −0.57 over 71M steps —
> while arm W DECAYS at −1.36e-3 nats/M, t = −23.6, ≈ −0.097 nats total, which clears the floor as a
> level change even though the endpoint difference does not.** The coefficient is matched at 0.05 and
> is live on a resume, so it held across all 13 restarts on both: the difference is downstream of the
> reward composition, which is the localisation §8.3 said this row could buy. The matched-step gap is
> largest at the 5M bucket (−0.153) and inside the floor from 25M on. **CRITIC ROWS.** 🚨 **The
> control arm S could not have now exists and `main.ops.critic_read` ACCEPTED it** — where arm S's
> read was REFUSED on two counts at once (`eval_sentinel_greedy` arm=True/control=False, `n_games`
> 800/400), the pair is comparable by construction: two independent offline full-capture draws per
> arm, seeds 20260910 / 20260911, **9,600 battles each, complete, shortfall 0**, sentinels GREEDY
> (regime RECORDED), draw/timeout share 0.4 % / 0.5 %, anchors 150/150 on both. 🚨 **Both arms are
> read at 74,000,016 and it is the SAME step — 1,005,936 (1.34 %) short of each arm's end, an exactly
> matched distance**, which is the check arm S's note asked the pair read to make; neither run has an
> eval cycle at its final step. Frames quota-MATCHED to caps 799/392/7 before any frame-sensitive row
> was labelled. `max|values − win_probs|` = **0.0 on arm W** (one tensor) and **75.12 / 119.06 on arm
> S** (the treatment, hazard H5, reproducing arm S's own numbers). **The registered guard
> `cond.opp_class_auc.t4_10`, ROW A (arm S's AUXILIARY head at coef 0.05 vs arm W's CRITIC), matched
> frame: 0.7686 vs 0.7431, Δ(S − W) +0.0254 [+0.0120, +0.0396] against the imported 0.02451 floor —
> NOT DETECTED**, reproducing on both draws (+0.0223 / +0.0211 as traced). **ROW B (arm S's ACTUAL
> shaped critic vs arm W's critic, rank rows ONLY) SPLITS: +0.0236 on draw1 and +0.0297 on draw2** —
> inside the floor on one and outside on the other, with `critic_read`'s own as-traced computation at
> +0.0290 — the shape rule 19 assigns to a row at its eval-draw noise and rule 21 calls **NOT
> CONFIRMED, never refuted**. Turn 1 reads ≈0.50 on every column and draw; the `t1_3` sibling's own
> draw spread (0.0218–0.0227) sits at or above its 0.0153 floor and cannot carry a claim. **Where the
> objectives DO come apart is CALIBRATION, not resolution.** `gate.resolution.all / bot / pool` are
> NOT DETECTED on every stratum (Δ(W − S) +0.0043 / +0.0028 / +0.0046), while **`gate.ece.all`
> +0.0510 [+0.0415, +0.0595] against a 0.0245 floor, `gate.ece.bot` +0.0614, `gate.reliability.all`
> +0.0046 and `gate.reliability.bot` +0.0152 are all DETECTED in the arm-W-is-worse direction and all
> reproduce on the second draw** (per-arm eval-draw spreads 0.0008–0.0028 against floors of
> 0.0245–0.0449). Arm W's ECE is ~4× arm S's head's overall (0.069 vs 0.018) and ~2× against bots,
> with calibration-in-the-large +0.58 vs +0.03 and `V − p̂` −0.110 vs −0.049 — **the same dispersion**
> (calibration slope 1.350 vs 1.346, Δ inside floor) **shifted in LEVEL: arm W is systematically
> pessimistic.** 🚨 **This is a CRITIC against a DIAGNOSTIC and must never be quoted as
> critic-vs-critic**: arm S's `win_probs` head carries 5 % of the value weight, is fit by a pure BCE
> and is not the baseline any advantage is taken against, while arm W's head IS the value function,
> consumed by GAE, with `--vf-coef 0.5` multiplying a BCE rather than a PopArt-normalised MSE
> (hazard H4, deliberately not re-tuned). Arm S's actual critic lives on a shaped-return scale where
> ECE is not defined. **UNTAUGHT METER.** Registry opponent `untaught_meter_opponent` (=
> `ai_v9_29_rev1_0823@24,000,000`), 200 games/team over the untaught 8, seed 0, concurrency 1, both
> arms in ONE invocation, **3,200 battles per config, 0 timeouts**: arm S **54.50 pp [52.25, 56.62]**,
> arm W **46.19 pp [42.81, 49.12]**, **IDENTICAL to the last win under BOTH config resolutions**
> (the condition §8.5 sets for a level counting as a level), and **arm S reproduces its banked
> 2026-09-14 level EXACTLY (Δ 0.00 pp, 872/1600 both times)**, confirming the meter is deterministic
> at seed 0 / concurrency 1. Paired team-clustered **Δ(S − W) = +8.31 pp [+5.69, +11.19], 8 of 8
> teams** — past **every** floor in evidence including the 4.27 pp controller-live one, but **a
> DESCRIPTOR and not an endpoint** (§8.5: the meter's axes and floors are established at ~1M FOLD
> depths and neither arm is a fold) and a **run-level CANDIDATE at n = 1 per arm**. The 0.02 leg
> joins at 58.25 pp (0.02 leg − arm W = +12.06 pp), so the three-way order is **0.02 leg > arm S >
> arm W on the meter while the ladder orders arm S > arm W > 0.02 leg** — the two instruments are not
> measuring one quantity, and **arm W is the only one of the three that is low on both**. **EXTERNAL
> ANCHORS, each inside its own regime.** vs Metamon `SmallRL` (ckpt 40, 13.9M, VanillaAttention, CPU,
> `@0a00a759`), arm S's own cell scripts reused verbatim with **the same team seeds**, 100 games per
> cell as two role-balanced half-cells: **greedy·home 0.630 vs 0.650, Δ(S − W) −0.020 [−0.151,
> +0.111]**; **greedy·AWAY (`python -m main.anchors`, the tool of record, on BOTH arms) 0.520 vs
> 0.500, Δ +0.020 [−0.117, +0.155]**; **mixed (ours greedy / theirs t = 1.0) ·home 0.840 vs 0.760,
> Δ +0.080 [−0.032, +0.190]** — **all three NOT DETECTED**. Regime VERIFIED per decision:
> `argmax_match_rate` **1.0000** in every greedy half-cell and **0.6408** in the sampling one (inside
> the 2×2's 0.623–0.680 band), our `stochastic` kwarg `[false]` everywhere; 1 tie in 300 home games,
> 0 in 200 away. Both arms drop ~13–15 pp from the home set to the away set, so the SOP's
> report-both-team-sets rule earns itself again. vs **Foul Play** (`6c467c08` + poke-engine 0.0.48,
> `--features gen3`) at `--search-time-ms 1000 --search-parallelism 1`, 80 games over the same 8
> pinned pool teams: **arm W 0.475 [0.369, 0.583] against arm S's 0.450 [0.346, 0.559], Δ(S − W)
> −0.025 [−0.175, +0.127], NOT DETECTED**; 0/80 forfeits, 0 error lines. 🚨 **AND THE WIDTHS ARE NOT
> MATCHED — this time in ARM W's favour**: realized **1.153 M MCTS visits/decision (sessions
> 0.86–1.33 M)** against arm S's **1.249 M** and the de-risk's **1.40 M**, because this campaign ran
> at load 15–25 against arm S's 26–37. **Across the three campaigns the win rates order EXACTLY
> inversely to realized search width (1.40 M → 0.388, 1.249 M → 0.450, 1.153 M → 0.475)** — rule 23's
> prediction for a WIDTH meter under a moving machine, and the reason **no difference among the three
> may be attributed to a model**. The honest statement: all three 75M policies sit **BELOW** Foul
> Play by a margin these experiments cannot separate, and the two external anchors continue to
> bracket the policy. **OPERATIONAL ROWS** (§8.4b, PRIMARY on arm W by its own `[CRITIC]` banner):
> `rollout/ep_len_mean` post-crossing mean **40.48 (S) vs 44.53 (W)**, post-crossing max 53.3 vs
> 63.9 — arm W's episodes run ~4 turns longer, the direction the banner warned of — **but neither arm
> is near rule 12's 25 % threshold, neither breached G7 (worst 1.070 vs 1.131, both under bar off
> in-family references) and arm W's stall half-peak is LOWER (0.0145 vs 0.0237)**. The mode's warning
> is visible as a shift in episode length and not as a pathology; `--arm-no-progress-tax` stays OFF.
> 🚨 **TWO HAZARDS THE REGISTRATION DID NOT PRICE.** (1) **THE REALIZED DOSE CAME OUT UNMATCHED.**
> The four dose inputs are identical tokens in both argvs so the dose is equal BY DECLARATION (§5),
> but the KL controller annealed the arms differently: `lr_median` **4.32e-4 (S)** vs **3.00e-4 (W)**,
> dose_rate **6.592e-08 vs 4.578e-08** — a **1.44× realized-dose gap in arm S's favour**. The
> registration anticipated the mechanism in words and no arm-to-arm number existed until now. Step
> size here is downstream of the objective, in the same class as `gamma` and PopArt — FORCED, not
> chosen — so it is not a design confound; it IS a reason no row on which arm S leads may be read as
> if the optimiser had been held fixed. (2) **A TOOL DEFECT: `--v-column` does not reach the
> quota-matched rows.** `main.ops.quota_match._scan` and its bootstrap pass call
> `conditioning_meters.conditioning_block(...)` **without `v_column`**, so every MATCHED row is
> computed on `win_probs` whatever the flag says — on a shaped arm, the AUXILIARY head rather than
> the critic. `critic_read --v-column values` returns matched rows byte-identical to the `win_probs`
> run while its own as-traced table correctly reads the shaped column: **the two halves of one report
> disagree about which tensor `V` is, and nothing in the output says so**, and the matched delta is
> the label-bearing one by design. ROW B is therefore reported AS TRACED only. **The `win_probs`
> default means no banked read is affected**; the fix is one keyword argument at two call sites and
> belongs on the tech-debt backlog. **Nothing under `src/` was changed by this read.** **Other
> hazards, each a finding:** the arm-S note's late-slope row for its two COMPARATOR runs does not
> reproduce from its own committed `strength_read.json` (it prints +0.15 / +0.32 where the JSON and a
> fresh refit both say −0.32 / −0.104; **arm S's own row reproduces exactly at +1.1435 ± 0.4357**, so
> its headline stands and only the comparator row is wrong); the Metamon `mixed · Metamon challenges`
> half-cell failed **for the third time in three independent campaigns, in the same cell, with the
> same `RecursionError` signature** *after* its last game, making that half-cell's positional join
> unreliable (`sides_disagree` 6, `metamon_row_missing` 2, **0 in every other half-cell**) — only our
> own complete 50-game record feeds the win rate; the two arms' three pool SENTINELS are different
> checkpoints (26/46/72M vs 22/42/72M) because the retained pools differ, so the frames are matched
> in KIND but not in the identity of three of twelve opponents; and `critic_read`'s delta orientation
> is arm − control (W − S) where every other row here is S − W. **The mirror battery was NOT run**
> (rule 25 retired the 100-pair cells; a ≥400-pair leaf read is a separate job). **NOTHING IS
> CLAIMED ABOUT THE CRITIC OBJECTIVE IN GENERAL**: n = 1 per arm, no seed replicate exists and none
> is affordable (~35 GPU-h), rule 22 binding — and the named next increment is **a seed replicate of
> arm W**, not a further offline read of these two. Tag: **MEASURED · the flywheel pair read COMPLETE
> · strength NOT DETECTED (+17.5, claimable ≈ 69; +6.7 on the common-step refit) · critic guard NOT
> DETECTED · anchors NOT DETECTED ×4, Foul Play's widths unmatched in arm W's favour · entropy FLAT
> vs DECAYING (a FINDING) · untaught +8.31 pp to arm S, 8/8 teams (a DESCRIPTOR) · ECE ~4× on arm W
> at equal resolution, CRITIC-vs-DIAGNOSTIC · realized dose 1.44× unmatched · `--v-column` misses
> every quota-matched row**.
