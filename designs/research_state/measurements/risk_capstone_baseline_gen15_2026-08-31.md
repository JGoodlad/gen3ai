# Risk-capstone baseline — the trace-only instruments on gen-15-era traces

**2026-08-31 · the pre-baseline registered in
[`designs/ai_v12/probe_risk_modulation_capstone.md`](../../ai_v12/probe_risk_modulation_capstone.md) §3**
("instruments 1–2 can be BASELINED NOW on gen-15 traces — the shaped-world row costs nothing and
pins the pre-clean-world slope"). Trace-only: no model loads, no battles. Theory:
`designs/learning/temperature_mixing_and_risk.md` §3. The constructed-scenario result these curves
should rhyme with:
[`starmie_ttar_risk_probe_v2_2026-08-31.md`](starmie_ttar_risk_probe_v2_2026-08-31.md)
(right sign, ~⅓-and-worse amplitude, no flip, ~0.79 safe-bias at equality).

Artifacts: `risk_capstone_baseline_gen15_2026-08-31.json` (every number) ·
`risk_capstone_baseline_gen15_2026-08-31_{explosion,accclass}.png` ·
**`risk_capstone_curves.py`** (the reusable script — run-dirs are a parameter; the ai_v12 arm rows
are the same invocation with different `--runs`).

**Sample:** 17 current-era runs (`models/ai_v9_60…76`, the R2/R3/R4 distillation-ladder arms off
the rev-1 shaped-world lineage), 8,405 trace battles, **224,995 move-selection decisions**, 296
distinct movesets. Win-prob = the recorded `win_probs` head at each decision. All slopes are OLS in
probability per unit win-prob, **cluster-bootstrapped over battles** (2,000 resamples). "FE" =
moveset fixed effects (within-moveset slope — controls the roster/composition confound).

## 1. Headline

| instrument | slope dP/d(wp) | 95% CI | n | verdict |
|---|---|---|---|---|
| **1. accuracy-tradeoff pairs (registered primary)** | — | — | **0** | **UNSCOREABLE — zero support in the population** |
| 1b. accuracy-class companion (confounded proxy) | −0.036 (FE −0.052) | [−0.061, −0.011] (FE [−0.075, −0.029]) | 43,289 | **near-flat** — consistent with prediction |
| **2. explosion timing** | **−0.085** (FE −0.088) | **[−0.106, −0.063]** (FE [−0.110, −0.068]) | 35,324 | **clearly falling — the registered SURPRISE condition fires (with a two-regime qualifier, §4)** |

## 2. Instrument 1 is UNDEFINED on this trace population — and will be on the ai_v12 arms too

229 same-type power/accuracy pairs resolve from `gen3_moves.json` (criteria: same type; inaccurate
member strictly higher basePower AND strictly lower accuracy; both ≥50 BP, numeric accuracy, no
never_miss/isCharge/recoil/selfDrops — Surf/HydroPump, IceBeam/Blizzard, Thunderbolt/Thunder among
them, resolved not hardcoded). Across all 224,995 decisions, the number where any pair is
simultaneously legal is **zero**. Per the registration's honesty rule: every canonical pair —
Surf/HydroPump n=0, IceBeam/Blizzard n=0, Thunderbolt/Thunder n=0 — is too rare to bound because it
never occurs at all.

This is **structural, not sampling**: a moveslot spent on the accurate twin is a wasted slot, so
real teams don't carry both. Measured in the team pool itself: **9 of 4,566 pool mons** carry any
same-type power/acc pair — 4× Mach Punch/Sky Uppercut (confounded by priority), 4×
Flamethrower/Overheat (confounded by Overheat's SpA −2; the only pair that surfaced in traces at
all, 17 decisions, excluded by the confound filter), 1× Thunderbolt/Thunder (the single clean
carrier; its mon never appeared in these traces).

**Implication for the capstone (flagged now, before the arms run):** the ai_v12 arm traces draw
from the same pool/slate, so instrument 1 will read n=0 there too. The per-arm accuracy-tradeoff
row must come from **constructed scenarios** (the Starmie/Ttar probe pattern — which is exactly
what `starmie_ttar_risk_probe_v2.py` already is) or from an eval-team slate deliberately augmented
with a pair-carrying set. The population instruments that CAN be compared arm-vs-arm are 1b and 2
below — this script, different `--runs`.

## 3. Instrument 1b — the accuracy-class companion (explicitly confounded)

Defined wherever the active mon has BOTH a legal 100-acc damaging move and a legal sub-100-acc
damaging move (confound-filtered as above), conditioned on choosing a damaging move. This drops
the same-type control — the two classes differ in type, so matchup/effectiveness confounds the
level — which is why the within-moveset FE slope is the honest read and the LEVEL is not the
endpoint.

P(chose sub-100-acc) by win-prob decile: 0.448 · 0.409 · 0.413 · 0.399 · 0.428 · 0.416 · 0.416 ·
0.421 · 0.403 · 0.393 (all bins n≥1,588). **Slope −0.036 [−0.061, −0.011]; FE −0.052
[−0.075, −0.029]** — right sign, but ~5pp across the entire wp range against a ~41% base rate:
**near-flat**. At equality (wp 0.45–0.55, n=2,623): P(inaccurate)=0.431 ⇒ safe-share 0.569 —
directionally the same safe-bias the constructed probe measured (0.79 at engineered TRUE equality;
not numerically comparable, the classes here are not equal-EV).

Sensitivity: on the `ai_v9_7*` runs alone the slope is −0.015 [−0.049, +0.020] — flat. The mild
pooled negative comes mostly from the `ai_v9_6*` arms; between-arm heterogeneity within one
lineage, worth re-reading per-arm when the ai_v12 rows exist.

## 4. Instrument 2 — explosion timing: falling, but in one regime only

P(boom | Explosion/Selfdestruct legal), 35,324 boom-legal decisions across 6,369 battles:

| wp decile | n | P(boom) | policy mass on boom |
|---|---|---|---|
| 0.0–0.1 | 625 | **0.330** | 0.321 |
| 0.1–0.2 | 653 | 0.306 | 0.279 |
| 0.2–0.3 | 942 | 0.269 | 0.254 |
| 0.3–0.4 | 1,259 | 0.257 | 0.237 |
| 0.4–0.5 | 1,775 | 0.209 | 0.203 |
| 0.5–0.6 | 2,746 | 0.184 | 0.184 |
| 0.6–0.7 | 4,304 | 0.175 | 0.172 |
| 0.7–0.8 | 6,505 | **0.165** | 0.161 |
| 0.8–0.9 | 6,832 | 0.169 | 0.166 |
| 0.9–1.0 | 9,683 | **0.208** | 0.209 |

Pooled slope **−0.085 [−0.106, −0.063]**; FE −0.088 [−0.110, −0.068] (FE ≈ pooled, so the fall is
within-moveset, not roster composition); probability-mass form −0.065 [−0.082, −0.049]. Per move:
Explosion −0.055 [−0.079, −0.031] (n=28,228), Selfdestruct −0.178 [−0.222, −0.136] (n=7,096).

**Two regimes.** All of the fall lives below wp≈0.5 (0.330 → 0.184); above 0.5 the curve is a
plateau at ~0.17 with an **uptick to 0.208 in the top decile**. So the shaped world does gamble
more from behind (sign-correct), but it **never consolidates when ahead** — it booms on ~1 in 6
boom-legal decisions even at wp 0.7–0.9, and MORE in near-certain wins. The correct-play criterion
("almost never comfortably ahead") is clearly violated on the ahead side; the top-decile uptick is
the self-KO-floor-leak signature (booming when it can no longer cost the game — cheap, but exactly
the risk-neutral-in-material behavior the capstone predicts shaping buys).

## 5. The registered prediction, scored

**Prediction (capstone §2): the shaped world reads SHALLOWEST — near-flat slope.**

- **Registered primary (accuracy pairs): UNSCOREABLE** — the instrument has zero support on any
  trace population drawn from this team pool (§2). This is the loudest finding of the baseline:
  the capstone's primary endpoint needs a constructed-scenario or slate-augmented carrier before
  any arm comparison, or it will silently score every arm as n=0.
- **Accuracy-class companion: CONSISTENT with the prediction** — −0.036 pooled / −0.052 FE, ~5pp
  across the whole range; near-flat.
- **Explosion: the pre-registered SURPRISE condition fires** — the slope is clearly falling
  (−0.085, CI excludes zero at every parameterization tried). Flagged loudly as the registration
  requires. **The honest qualifier:** the fall is entirely the behind-side half; the ahead-side —
  where the clean-world argument predicts shaping hurts most (concavity above the midpoint ⇒
  consolidate) — is flat-to-rising. So the shaped world is NOT risk-blind from behind, which
  weakens the strongest form of "shaping destroys risk modulation," but the consolidation failure
  the theory predicts is present and measured. The ai_v12 sparse arm's testable delta is now
  sharp: **steepen the behind-side slope AND suppress the ahead-side plateau** (cw1 should boom
  <<0.17 at wp>0.7).

## 6. Rhyme with the constructed-scenario probe

Right sign: yes — both instruments fall, as the Starmie/Ttar mask tracked truth with the right
sign. No flip / weak amplitude: yes — the boom rate never crosses 0.5 anywhere and the accuracy
companion moves ~5pp over the full range (the population analog of the probe's ~⅓-and-worse
amplitude). Safe-bias at equality: same direction (0.569 population safe-share on a confounded
class split vs 0.79 at engineered true equality). The population curves and the constructed probe
agree: the shaped-world policy carries a weak, sign-correct, level-biased risk response.

## 7. Honest caveats

1. **Instrument 1b has no type control** — the inaccurate class skews toward specific
   move/type mixes and matchup drives damaging-move choice through effectiveness, not risk. FE
   removes the roster confound only. Slope, not level; direction, not magnitude.
2. **Boom-legality is state-correlated**: a boom-legal decision at low wp is a different board
   (last mon, forced positions) than at high wp. FE ≈ pooled rules out roster composition, not
   board-state composition. The behind-side fall may partly BE correct play being available —
   which is not a confound for the capstone's use (arm-vs-arm at matched populations) but bars
   reading the −0.085 as a calibrated risk-response gain.
3. **The win-prob axis is the model's own head** — a miscalibrated head compresses/stretches the
   x-axis per regime. Arm comparisons inherit each arm's own head; the capstone's cross-arm read
   should check head calibration per arm (`python -m main.prober.query calibration`) beside these
   slopes.
4. **Eval traces are greedy** (`stochastic=False`), so the indicator form is P(argmax = risky);
   the probability-mass form is reported beside it everywhere and agrees (−0.065 vs −0.085).
5. **These 17 runs are experiment arms, not one policy** — R2/R3/R4 distillation-ladder forks of
   one lineage. Pooling is deliberate (the shaped-world ROW, maximal support); the 7x-only
   sensitivity (§3, and `/tmp`-reproducible via `--runs 'models/ai_v9_7*'`) shows the companion
   slope is family-sensitive while the explosion slope is not (−0.102 on 7x alone).

## 8. Rerun

```bash
export PYTHONPATH=$PYTHONPATH:src
# gen-15 baseline (this document):
python designs/research_state/measurements/risk_capstone_curves.py \
  --runs 'models/ai_v9_6*' 'models/ai_v9_7*' \
  --out designs/research_state/measurements/risk_capstone_baseline_gen15_2026-08-31.json \
  --bootstrap 2000 --plots
# the ai_v12 arm rows, later — same script, different runs:
python designs/research_state/measurements/risk_capstone_curves.py \
  --runs 'models/<cw1_sparse_run>' --out <arm_row>.json --bootstrap 2000 --plots
```
