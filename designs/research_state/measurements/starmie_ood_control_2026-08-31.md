# Starmie/Tyranitar OOD control — the constructed risk probe repeated at COMMON faint counts

**2026-08-31 · owner-ordered.** The objection under test, verbatim in substance: the
[v1](starmie_ttar_risk_probe_2026-08-30.md) / [v2](starmie_ttar_risk_probe_v2_2026-08-31.md)
constructed risk probe reads the policy's mask at a decision with **five fainted mons on each
side** — a 1v1 endgame, the extreme tail of the attrition distribution — reached by an
**engineered prelude** (v2: a nine-turn Protect/pivot reveal phase followed by an
explosion-into-Protect parade, 22 turns in all), so the observation's history blocks carry a
pattern no natural game produces. The probe's findings may therefore be out-of-distribution
artifacts rather than general properties.

Three measurements answer it: the **base rate** of the probed state class over real eval traces,
a **control sweep** repeating the identical engineered gamble at faint counts F = 1…5 under a new
and much more ordinary prelude, and a **history-block OOD distance** placing the constructed
observations against the trace distribution block by block.

**v1's and v2's records and artifacts are NOT modified.** This is a new record that
cross-references them.

Artifacts (beside this file):
`starmie_ood_control_2026-08-31.json` (every control-sweep number) ·
`starmie_ood_control_probe.py` (the construction; phases `smoke|capture|sweep|v2obs|seedcheck|analyze`,
resumable) · `starmie_ood_control_obs.npz` (the 79 captured observation vectors) ·
`starmie_ood_control_traces.py` + `starmie_ood_control_traces_2026-08-31.json` (the base rate) ·
`starmie_ood_control_distance.py` + `starmie_ood_control_distance_2026-08-31.json` (the OOD
distances). `starmie_ood_control_traces_stats.npz` (4.3 MB of trace obs statistics + subsamples) is
regenerable scratch and is deliberately **not committed**.

Checkpoint: `models/ai_v9_70_R3ACTION_0828/final_model.zip` (v1/v2's), lineage
`models/ai_v9_29_rev1_0823/final_model.zip`. Bridge: node, `gen3customgame`, CPU.

---

## 1. Verdict

**The bias numbers HOLD at common faint counts — and are LARGER there, not smaller. The
out-of-distribution objection is answered: the constructed 5-5 numbers were, if anything,
CONSERVATIVE.** Every headline of v1/v2 replicates under a different prelude, a different filler
roster, and a different board:

- the safe-move mass at true KO-indifference is **0.79 → 0.83 → 0.91–0.93** going from v2's 5-5 to
  a fresh 5-5 to the common 2-2 / 3-3 / 4-4 boards (unbiased = 0.5);
- the **argmax never flips** at F ≥ 2, censored at the same > 0.7375 KO-probability deficit;
- the **zero excess response at the KO-roll boundary** replicates exactly, at every faint count.

**One claim does NOT generalize and must be restated.** At **F = 1** (1-1 faints, five alive per
side) the bias vanishes — P(Surf) at true indifference is **0.4937**, the argmax flips at the
**first** truly-Pump-favouring cell, and the policy orders **22 of 22** sweep points correctly
where every F ≥ 2 arm orders **6 of 22**. So "the argmax never flips" and "the mask carries a large
safe-move bias" are properties of the **attrited** board, not of the policy at an equal-EV gamble
in general.

**And the history-block half of the objection is REFUTED on its own terms.** Against the real
trace distribution, the constructed observation's **event window and pair history are the LEAST
anomalous blocks in it** — the event window's diagonal Mahalanobis distance sits *inside* the
traces' own 99th percentile in every arm (1.11 vs a p99 of 1.57 at F = 5; 0.89, *below the trace
median*, at F = 2). What is massively out of distribution is **our own roster** (6.07 vs a p99 of
1.71 — 3.6× past it), which the objection did not name and which is unavoidable in any hand-built
scenario.

---

## 2. Base rate — how rare is the probed state class?

`starmie_ood_control_traces.py`, five current-era runs (`ai_v9_70_R3ACTION_0828`,
`ai_v9_72_R3SELF_0828`, `ai_v9_75_R4S3c_0829`, `ai_v9_76_R4ACTION_0830`, `ai_v9_77_G1LEAN_0830`),
all obs dim 2501: **2,512 battles / 90,202 decisions**. Faint counts are decoded from the obs
itself (`species_known == 1` and `hp == 0`, per 122-dim slot); zero monotonicity violations across
all 2,512 battles.

### The diagonal — the exact family the control arms use

| F = our faints = opp faints | decisions | fraction |
|---|---|---|
| 0 | 15,165 | 0.1681 |
| 1 | 6,088 | 0.0675 |
| 2 | 4,204 | 0.0466 |
| 3 | 3,389 | 0.0376 |
| 4 | 2,676 | 0.0297 |
| **5** | **2,568** | **0.0285** |

**P(5-5) = 0.0285.** State it plainly: the probed class is **2.85% of decisions**, at the
**97.15th percentile** of total faints. `P(5-5)`, `P(1v1 endgame)` and `P(total faints ≥ 10)` are
numerically identical because with `opp_alive = 6 − opp_faints` they are the *same predicate* —
reported as one number rather than three, deliberately. By opponent class: **bots 0.0321**
(n = 59,238) vs **model opponents** (sentinels + frozen `ext_` snapshots) **0.0216** (n = 30,964).
Per-run P(5-5) spans 0.0188–0.0451, so the pooled figure is not one run's artifact, but the spread
is ~2.4× and should not be quoted to three digits.

The 5-5 cell is a rare tail — the objection's premise is correct — though **less rare than the
registered prediction guessed** (2.85%, not < 2%), and the diagonal is remarkably flat from F = 1
to F = 5 (6.75% → 2.85%), so "common" only buys a factor of ~1.6–2.4 over "rare" on this axis.

### The accuracy / KO-race decision class — flat, and if anything RARER at 5-5

Defined per decision as: our active mon's move set contains two damaging moves, one with
accuracy ≥ 95 and one with accuracy ≤ 85 and strictly higher base power (the Surf/Hydro-Pump
shape). **11.20% of all decisions** (10,106 / 90,202).

| F | 0 | 1 | 2 | 3 | 4 | **5** |
|---|---|---|---|---|---|---|
| class frequency | 0.1127 | 0.1247 | 0.1263 | 0.1301 | 0.1413 | **0.0884** |

**The decision shape the probe constructs is no more available at 5-5 than at common faint counts —
it is less.** A matched control at F = 2/3 therefore loses nothing on carrier availability, which
is the fact that licenses §4.

⚠️ **Reconcile this with
[`risk_capstone_baseline_gen15_2026-08-31.md`](risk_capstone_baseline_gen15_2026-08-31.md), which
reports 0 pair-legal decisions out of 224,995 — the two are not in conflict.** The capstone's
registered instrument requires a **same-type** power/accuracy pair (Surf/HydroPump,
IceBeam/Blizzard, Tbolt/Thunder), and only 9 of 4,566 pool mons carry one. The class counted here
is **cross-type** (top witnesses: icebeam/hydropump 23.9%, pursuit/fireblast 20.8%,
hiddenpowerfire/meteormash 17.4%), which is a genuinely available risk trade but is *not* the
clean same-type contrast the capstone registered. Both numbers are right about different questions.
Only four risky (acc ≤ 85) damaging moves exist in this eval team pool at all — hydropump,
meteormash, fireblast, megahorn — present on 16.3% of decisions.

**Caveat carried from the measuring script:** the 85–95 accuracy dead band excludes Rock Slide
and Overheat (both 90) by the spec's own definition, and the accurate side of the pair is not
required to be a serious attacking option (Rapid Spin witnesses 3.5%). A first pass of this
measurement read **4.76%** with Hydro Pump as 100% of witnesses; that was a float32 artifact
(`float32(0.85) × 100 == 85.0000024` fails `≤ 85`), which silently deleted every 85%-accuracy move.
Rounding to integers before thresholding moved the class to 11.20% and the "only Hydro Pump"
finding evaporated. Recorded because a spot check with a 0.75 tolerance passed straight through it.

---

## 3. The control construction — one script, five faint counts, F the only manipulated variable

`starmie_ood_control_probe.py`. The gamble is v1's, unchanged: our frail Starmie (Def IV 0,
262 HP) vs their Choice Band Tyranitar at an Endeavor-set exact HP `H`, where **Surf** (100%
accuracy) KOs on `k` of 16 damage rolls and **Hydro Pump** (80% accuracy) KOs on every roll — so
`E[KO|Surf] = 1/16 + (15/16)·k/16` against a flat `E[KO|Pump] = 0.80`. Roll tables, crit fraction
and miss fraction are v1's damage-probe measurements, read from its JSON and not re-derived.

The prelude is **new, and deliberately ordinary**:

```
T1  Marshtomp (max HP == H, faster) ENDEAVORS Tyranitar -> Ttar HP := H exactly; CB EQ chips it
T2  Marshtomp Growls; TYRANITAR SWITCHES OUT to Koffing (Choice lock clears; Ttar is never
    touched again). Nothing our side controls can reach Tyranitar after T1.
T3  Koffing EXPLODES -> Marshtomp and Koffing both faint          faints 1/1
T4  Weezing EXPLODES  -> our filler and Weezing both faint        faints 2/2
T5  Graveler   -> 3/3      T6  Pineco  -> 4/4      T7  Exeggutor -> 5/5
```

Every step is one Explosion that kills its target, so **faints advance 1/1 per turn** and F is set
by where the chain stops. On the F-th step BOTH sides force-switch in the same turn — we choose
Starmie, they choose Tyranitar — so the **entry mechanism is identical at every F** (at F = 5 it is
the only legal choice), Starmie always enters after residuals at exactly 262 HP with zero sandstorm
ticks, and Tyranitar always enters unlocked at exactly H. Decision-state asserts fire on every
battle: faints F/F, Starmie 262/262, Tyranitar exactly H/342, decision reached exactly once.

**There is no Protect anywhere**, which is what makes this chain deterministic where v1's was not:
gen3's stall counter gives a second consecutive Protect only a 1/3 success chance
(`conditions.ts::stall`, `randomChance(1, counter)` with `counter = 3`), and v1's parity rule does
call for consecutive Protects — its chain self-heals, but its middle is stochastic. Their exploders
carry 252 Atk / Adamant so every Explosion is guaranteed overkill on the chipped Marshtomp and on
our deliberately frail fillers (Rattata / Sentret / Zigzagoon / Poochyena). Prelude length: 4 turns
at F = 1 to 8 at F = 5, against v2's 22.

**Two things change at F < 5 that are properties of the question, not defects, and both are handled
explicitly:**

1. **Switches are legal** (up to 4 alive bench mons), so mass could leak out of the two moves. The
   headline statistic is therefore the renormalized `P(Surf | Surf or Hydro Pump)`. **It turned out
   not to matter: the policy puts EXACTLY 0.000000 on every switch action at all 110 sweep points**,
   at every faint count, with up to four legal switches — so raw and renormalized agree to 5 decimal
   places everywhere. That zero is itself a finding (§6).
2. **A KO is no longer a win.** The x-axis is `E[KO|Surf] − E[KO|Pump]`, which equals the
   win-probability delta only at F = 5. Every truth column below is a **KO** probability and is
   labelled as one; §9 states what that costs the interpretation.

---

## 4. The bias numbers, side by side with v2's

All statistics recomputed by the SAME code (`analyze_arm`) over 22 sweep points per arm, including
v2's banked hidden-condition sweep, so nothing is compared across two implementations.

| arm | prelude | board | **P(Surf) at true KO-indifference** | slope global | slope local (\|δ\|≤0.16) | span ratio | P at best Surf (k=16) | P at worst Surf (k=0) | argmax flip |
|---|---|---|---|---|---|---|---|---|---|
| **v2 F5 (reference)** | v2, 22 turns | 5-5, 1v1 | **0.7872** | 0.403 | 0.410 | 0.347 | 0.854 | 0.528 | **no** — censored > 0.7375 |
| **F5** | new, 8 turns | 5-5, 1v1 | **0.8272** | 0.339 | 0.363 | 0.280 | 0.883 | 0.620 | **no** — censored > 0.7375 |
| **F4** | new, 7 turns | 4-4, 2v2 | **0.9194** | 0.281 | 0.240 | 0.226 | 0.954 | 0.743 | **no** — censored > 0.7375 |
| **F3** | new, 6 turns | 3-3, 3v3 | **0.9290** | 0.268 | 0.227 | 0.216 | 0.961 | 0.759 | **no** — censored > 0.7375 |
| **F2** | new, 5 turns | 2-2, 4v4 | **0.9142** | 0.337 | 0.284 | 0.268 | 0.954 | 0.703 | **no** — censored > 0.7375 |
| **F1** | new, 4 turns | 1-1, 5v5 | **0.4937** | 0.540 | 0.900 | 0.506 | 0.662 | 0.188 | **YES** at H = 296, deficit **0.0344**, 16/22 points |

Base cell (H = 295, k = 13, truth Surf 0.8242 vs Pump 0.8000), full value stack:

| arm | turn | alive | legal switches | P(Surf) | P(Pump) | P(switch) | V(s) | win-prob |
|---|---|---|---|---|---|---|---|---|
| F1 | 4 | 5/5 | 4 | 0.4998 | 0.4995 | 0.000000 | 6.354 | 0.7904 |
| F2 | 5 | 4/4 | 3 | 0.9162 | 0.0838 | 0.000000 | 4.588 | 0.7351 |
| F3 | 6 | 3/3 | 2 | 0.9306 | 0.0694 | 0.000000 | 4.330 | 0.7166 |
| F4 | 7 | 2/2 | 1 | 0.9211 | 0.0789 | 0.000000 | 10.236 | 0.8037 |
| F5 | 8 | 1/1 | 0 | 0.8297 | 0.1702 | — | 21.178 | 0.9189 |
| v2 prelude, F5 | 22 | 1/1 | 0 | 0.7901 | 0.2098 | — | 20.820 | 0.8977 |

**The two effects, separated.**

- **HISTORY (+ roster) — the prelude change alone, at a matched 5-5 board: 0.7872 → 0.8272,
  Δ = +0.040.** That is the whole cost of replacing a 22-turn Protect/Explosion parade with an
  8-turn ordinary one *and* swapping four filler species. It is real, it is small, and it points
  the wrong way for the objection (the engineered prelude produced the *lower* bias, not the
  higher).
- **BOARD — faint count, at a matched prelude: 0.8272 (F5) → 0.9194 / 0.9290 / 0.9142 (F4/F3/F2),
  Δ = +0.09 to +0.10; then 0.4937 at F1, Δ = −0.33.** The level is strongly faint-count dependent
  and **non-monotone**, with a discontinuity between F = 1 and F = 2 (§5).

**Construction noise is ~0.001 and cannot explain any of it.** `--phase seedcheck` re-runs the base
cell under five distinct dice streams (five distinct observation SHAs — the turn-1 Earthquake
magnitude rides in the event window, so the histories genuinely differ): spread in
`P(Surf | Surf∪Pump)` is **0.0011 at F1, 0.0007 at F2, 0.0013 at F5**. The faint-count effect is
**~320× the noise floor**; the prelude effect is ~30×.

**Lineage (rev-1, `ai_v9_29_rev1_0823`, base cell): 0.9643 at F5, 0.9833 at F2** — near-flat across
the same manipulation (Δ = +0.019), and sharp at both. So the faint-count sensitivity measured here
is a property of the **newer** checkpoint, not of the construction. (rev-1's value stack disagrees
sharply too: V = 16.0 / wp 0.891 at F5 but V = −6.15 / wp 0.525 at F2, where R3ACTION reads 0.735.)

---

## 5. The F = 1 discontinuity — the one place the v1/v2 claims do NOT generalize

At 1-1 faints the policy behaves like a different agent on the identical gamble:

| H | k | E[KO\|Surf] | truth-better move | P(Surf \| Surf∪Pump) | argmax |
|---|---|---|---|---|---|
| 284 | 16 | 1.0000 | Surf | 0.6619 | surf ✓ |
| 289 | 14 | 0.8828 | Surf | 0.5915 | surf ✓ |
| 293 | 13 | 0.8242 | Surf | 0.5310 | surf ✓ |
| 294 | 13 | 0.8242 | Surf | 0.5157 | surf ✓ |
| **295** | **13** | **0.8242** | **Surf** | **0.5002** | **surf ✓** |
| **296** | **12** | **0.7656** | **Pump** | **0.4846** | **hydropump ✓** |
| 299 | 12 | 0.7656 | Pump | 0.4386 | hydropump ✓ |
| 306 | 9 | 0.5898 | Pump | 0.3367 | hydropump ✓ |
| 337 | 0 | 0.0625 | Pump | 0.1877 | hydropump ✓ |

**22 of 22 points ordered correctly**, with the crossing landing in the single 1-HP window
(295 → 296) where the truth crosses — the finest resolution gen3's discrete rolls allow. Every
F ≥ 2 arm orders **6 of 22** correctly (the k ≥ 13 points only) and is wrong at the remaining 16,
including H = 337 where Surf's true KO probability is 0.0625 against Pump's 0.80.

**Do not over-read this as boundary evaluation.** The unit-HP steps at F = 1 are
−1.538, −1.550, **−1.554**, −1.541, −1.527, −1.535, −1.537 pp across H = 293…300: the 295 → 296
step, where the true Surf value drops 5.9pp, is **indistinguishable from its neighbours**, exactly
as in v2. The mechanism is still a smooth function of the displayed HP bar; at F = 1 the curve's
level and slope simply happen to put its 0.5 crossing on the right cell. What changed is the
**level** (0.49 vs 0.83–0.93) and the **amplitude** (local slope 0.900 vs 0.24–0.36), not the
mechanism.

Whether landing on the right cell is skill or coincidence is **not settled by one arm**. The
crossing could fall anywhere in a ~50 HP sweep; landing within one HP point of truth is ~4% by
chance, which is suggestive and no more. What *is* settled is the behavioural fact: at 1-1 the
argmax responds to the KO boundary and at 2-2 through 5-5 it does not.

---

## 6. What replicates at EVERY faint count

1. **Zero excess response at the KO-roll boundary.** The 295 → 296 unit-HP step (where the truth
   moves 5.9pp) against its neighbours (where the truth does not move at all):

   | arm | 293→294 | 294→295 | **295→296** | 296→297 | 297→298 | 298→299 |
   |---|---|---|---|---|---|---|
   | v2 F5 | −0.666 | −0.687 | **−0.709** | −0.731 | −0.749 | −0.774 |
   | F5 | −0.580 | −0.605 | **−0.630** | −0.654 | −0.683 | −0.709 |
   | F4 | −0.369 | −0.385 | **−0.406** | −0.435 | −0.463 | −0.492 |
   | F3 | −0.341 | −0.362 | **−0.385** | −0.409 | −0.437 | −0.464 |
   | F2 | −0.419 | −0.459 | **−0.491** | −0.524 | −0.558 | −0.593 |
   | F1 | −1.538 | −1.550 | **−1.554** | −1.541 | −1.527 | −1.535 |

   pp of `P(Surf | Surf∪Pump)` per HP point. Not one arm shows a kink. **v2's sharpest finding —
   the policy prices the HP BAR, not the roll table — is fully general.** This is the finding the
   [`ko_boundary_decodability`](ko_boundary_decodability_2026-08-31.md) programme and the GLU
   framing rest on, and it is untouched by the objection.

2. **The argmax is censored at F ≥ 2**, at the identical bound: no flip anywhere in the sweep, so
   the flip threshold exceeds a true KO-probability deficit of **0.7375** (Surf 0.0625 vs Pump
   0.80). Same number as v1 and v2, at four different boards.

3. **Switch probability is exactly zero.** At F = 1…4 the mask offers 1–4 legal switches and the
   policy assigns them **0.000000** combined at every one of the 88 points — with 5, 4, 3 and 2
   alive mons behind Starmie. Under an equal-EV do-or-die gamble the policy never considers not
   taking it. Two readings, both worth banking: it makes the renormalization in §3 vacuous (good for
   the comparison), and it is itself an OOD tell (real decisions carry meaningful switch mass — the
   human-agreement probe measured a 28.96% human switch share).

4. **Direction is always right; magnitude never is.** `P(Surf)` falls monotonically with the true
   KO fraction in every arm, at 0.22–0.90 local gain against the ~1.0 the truth demands.

---

## 7. History-block OOD — which blocks are anomalous, and by how much

`starmie_ood_control_distance.py`. Each constructed observation is scored per obs block against
4,000 random real trace observations (and against two matched strata), with the block's diagonal
Mahalanobis distance (RMS z over dims the traces actually vary in) compared to **the distribution
of the same statistic over the trace rows themselves** — so "how far out" is expressed in units the
traces supply.

Reference: pooled traces (n = 4,000). `ref median` 0.92 / `ref p99` for each block in its own row.

| block | dims | trace median | trace **p99** | F2 | F5 | **v2 prelude F5** | reading |
|---|---|---|---|---|---|---|---|
| **our_team** | 732 | 0.924 | 1.708 | **6.072** | **6.072** | **5.985** | **3.6× past p99 — massively OOD** |
| opp_team | 732 | 0.792 | 2.296 | 0.548 | 0.670 | 0.817 | inside, below median |
| active_ctx | 116 | 0.111 | 4.910 | 0.111 | 0.111 | 0.111 | exact trace match exists |
| global_env | 20 | 0.650 | 2.704 | 0.553 | 0.482 | 0.411 | inside, below median |
| board | 17 | 0.823 | 3.196 | 0.526 | 0.844 | 0.836 | inside |
| **pair_history** | 180 | 0.830 | 2.199 | **0.660** | **0.795** | **1.137** | **inside p99 in every arm** |
| **event_window** | 704 | 0.962 | 1.571 | **0.889** | **1.109** | **1.129** | **inside p99 in every arm** |
| per-mon last action | 72 | 0.717 | 2.672 | 0.942 | 0.942 | 0.656 | inside |
| per-mon recency | 36 | 0.969 | 1.359 | 1.031 | 0.944 | 1.113 | inside |
| FULL_OBS | 2501 | 0.940 | 1.628 | 3.146 | 3.180 | 3.158 | driven by `our_team` |

**The objection named the wrong blocks.** The 32-event window and the 6×6×5 pair history — the two
it identified — are the *least* anomalous parts of the constructed observation, both sitting inside
the traces' own 99th percentile at every arm, and the event window at F = 2 sits **below the trace
median**. The v2 prelude is the most anomalous of the three on pair history (1.137 vs F5's 0.795)
and essentially tied on the event window (1.129 vs 1.109) — consistent with, and about the size of,
the +0.040 mask shift §4 measures for the prelude change.

What *is* out of distribution is **our own team block**: mean |z| 1.61, max |z| 116, 7.95% of live
dims past |z| > 3, and **7.35% of dims that are CONSTANT across all 4,000 trace rows take a
different value** — Marshtomp / Rattata / Sentret / Zigzagoon / Poochyena / a Def-IV-0 Modest
Starmie is a roster no eval team resembles. That is intrinsic to any hand-built scenario and cannot
be engineered away; it is also the block whose content the two moves' comparison does not turn on.

Against the **matched F2–F3 stratum** (n = 2,000) the ordering is unchanged and the event window
looks *better* still (nearest-neighbour percentile 20.1 at F2, 14.9 at F1); against the
**F ≥ 4 stratum** (n = 381) every block rises, as a smaller and more homogeneous reference set
forces it to. Full tables in the JSON.

⚠️ **Two limits on this instrument, stated rather than buried.** (a) A *diagonal* Mahalanobis
standardizes each dimension separately and is blind to a combination of individually-ordinary
values; the nearest-neighbour statistic was included to cover that, but it is degenerate on the
blocks whose trace-to-trace NN distance is ~0 (`our_team`, `board`, `active_ctx`, `global_env` —
the eval team pool is small, so many decisions share byte-identical blocks), which is why the
`nn_ratio` column reads 558× for `our_team` and is not quoted here. (b) All of it is a *marginal*
comparison against a 4,000-row sample; it bounds anomaly, it does not certify in-distribution-ness.

---

## 8. Registered predictions, scored

| # | prediction | outcome |
|---|---|---|
| 1 | 5-5 faint states are **< 2%** of trace decisions | **MISSED, narrowly.** 2.85% (2,568 / 90,202). The direction was right — it is the 97.15th percentile of attrition — but the threshold was wrong. |
| 2 | the safe-bias at equality **PERSISTS but is SMALLER**, between 0.60 and 0.78 vs the 0.79–0.80 measured at 5-5 | **REFUTED on the magnitude, confirmed on the persistence.** At the common counts it is **0.914 / 0.929 / 0.919** (F2/F3/F4) — *larger*, not smaller, and outside the predicted band in the opposite direction. And the prediction did not anticipate F = 1 at **0.494**, which is below the band on the other side. |

Both misses are recorded rather than smoothed. The second is the informative one: the prior
expectation was "the constructed number is inflated by the tail", and the measurement says the tail
is where the bias is *smallest* among attrited boards.

---

## 9. Verdict applied, and what must be restated

Against the pre-registered rule:

> *Bias numbers HOLD at common faint counts ⇒ the v1/v2 findings generalize; the OOD objection is
> answered and the constructed probe stands as measured.*

**This is the branch that fires**, with two qualifications that the rule's third branch anticipated
("numbers hold on the board but move with the PRELUDE ⇒ history-block sensitivity, itself a notable
finding"): the prelude moves the number by **+0.040**, small but real and measurable above a 0.001
noise floor; and the board moves it by **−0.33 to +0.10**, so the *level* is a property of the
board, not a constant of the policy.

### Claims that stand, unrestated

- v1 §3 / v2 §1 **bias #1** — a large safe-move mass at true indifference. Replicated at four
  boards; the 5-5 value is the *low* end of the attrited range.
- v2 §3 **the micro-step finding** — the mask prices the HP bar, not the roll table, with zero
  excess response at the KO boundary. Replicated at six arms including F = 1.
- v2 §1 **bias #3** — the amplitude is a fraction of what the truth demands. Range 0.22–0.90 local
  across arms, all below 1.0.
- [`ko_boundary_decodability_2026-08-31.md`](ko_boundary_decodability_2026-08-31.md) and the
  **GLU / gated-mechanism framing** it serves. Its motivating fact is the no-kink result, which is
  the single most robust finding here. **No restatement needed.**
- [`exploiter_fingerprint_truthcheck_2026-08-31.md`](exploiter_fingerprint_truthcheck_2026-08-31.md).
  Population-scale, trace-derived, never touches this construction. Unaffected, as the mission
  stated.
- The capstone's **constructed-carrier plan**. Strengthened, not weakened: §2 shows the carrier
  decision class is *at least as available* at common faint counts, so a constructed carrier can be
  built at F = 2/3 and need not sit in the tail.

### Claims that MUST be restated

1. **"The argmax never flips" is faint-count-scoped, not general.** v1 §3 reading 3 and v2 §1 bias
   #2 must carry "at F ≥ 2"; at 1-1 faints the same policy on the same gamble orders 22/22 points
   correctly. Anywhere that fact was used to argue the policy cannot act on the KO boundary *at
   all*, it now argues that it stops doing so once the board is attrited. Ledger entries
   `41d2d60` and the v2 entry are the two places to annotate — **by appending, never by editing
   them** (append-only discipline).
2. **"P(Surf) at true equality = 0.787 / 0.803" is one board's number, not the policy's.** The same
   quantity spans **0.49 → 0.93** across faint counts, a range 320× the construction noise. Quote it
   with its board.
3. **The v1/v2 amplitude figures ("~⅓", 0.403 / 0.187) are board-scoped too**, ranging 0.216–0.506
   (span ratio) across arms. v2's revealed-vs-hidden amplitude *halving* was measured at 5-5 only
   and this probe does not re-test it — the reveal manipulation was not repeated here.
4. **A caveat the constructed frame owes at F < 5, and which cuts against this record's own
   headline: the KO-equal anchor is not the win-equal anchor once a bench survives.** A failed Surf
   leaves Tyranitar chipped; a missed Hydro Pump leaves it at full HP. With mons left to fight, that
   residual has value, so the win-optimal Surf lean at F < 5 sits **above** 0.5 by an unmeasured
   amount and part of the 0.91–0.93 at F = 2/3/4 may be *correct*. Only the **F = 5** number is
   anchored exactly (there, KO ⇔ win). The claims that survive this caveat untouched are the
   censored argmax at H = 337 (where Pump's 0.80 outright KO beats any chip story), the no-kink
   result, and the F = 5 numbers.

---

## 10. Honest caveats

1. **One checkpoint family, one scenario.** `ai_v9_70_R3ACTION_0828` plus a rev-1 lineage read at
   two cells. rev-1 shows almost no faint-count sensitivity (Δ 0.019), so §4's F-dependence is not
   a property of "the architecture" — it is a property of this checkpoint, and n = 2 checkpoints
   cannot say more.
2. **The prelude comparison is prelude + roster, not prelude alone.** v2's F5 and this F5 differ in
   the filler species and EV spreads as well as the choreography; the +0.040 is their sum.
3. **F = 1 rests on one arm of 22 points plus a 5-seed base-cell check.** Its crossing landing on
   the correct 1-HP cell is suggestive, not established (§5).
4. **The bench mons are junk on purpose.** Rattata / Sentret / Zigzagoon / Poochyena keep the
   Explosion kills deterministic, but they mean the "4 alive mons" at F = 2 are worth very little —
   which is one reason the model may value the KO nearly as decisively at F = 2 as at F = 5, and a
   reason the F < 5 arms understate how different a *realistic* mid-game board would look. A control
   with a plausible bench is the obvious next iteration and is **not** run here.
5. **Every "truth" column is a KO probability** (§3 point 2, §9 point 4).
6. **The OOD distance is marginal and diagonal** (§7's two limits).
7. **The base rate is five runs' eval traces** — bot and sentinel opponents at a fixed team pool,
   not ladder play. Per-run P(5-5) spans 2.4×.

---

## 11. Rerun

```bash
export PYTHONPATH=$PYTHONPATH:src
D=designs/research_state/measurements
nice -n 15 python $D/starmie_ood_control_traces.py                       # base rate (~7 min, 90k decisions)
nice -n 15 python $D/starmie_ood_control_probe.py --phase all            # F=5,3,2 sweeps + base cells (~25 min)
nice -n 15 python $D/starmie_ood_control_probe.py --phase sweep --faints 1,4
nice -n 15 python $D/starmie_ood_control_probe.py --phase analyze
nice -n 15 python $D/starmie_ood_control_probe.py --phase seedcheck      # the noise floor
nice -n 15 python $D/starmie_ood_control_distance.py                     # history-block OOD (~1 min)
```

Node bridge, no server, CPU-only, 2 torch threads. Every phase is resumable and saves
incrementally. v1's and v2's scripts and JSONs are read and never written.
