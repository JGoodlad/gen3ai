# KO-boundary decodability — where the knockout-roll structure actually stops

**2026-08-31 · owner-ordered.** `starmie_ttar_risk_probe_v2_2026-08-31.md` measured the win-prob
head pricing the displayed HP bar smoothly with **zero excess response at the KO-roll boundary**,
over a ~40× compressed range; `exploiter_fingerprint_truthcheck_2026-08-31.md` replicated the
defect at population scale (per-state |error| **0.278** against an aggregate bias of **+0.036**,
AUC 0.679 vs truth's 0.970). Both measure the HEAD. This measurement asks the layer beneath it:
**is the knockout information in the representation at all, and if so where does it stop?**

The owner's question behind the mission — *"is this evidence that we need GLU to produce those
sharp changes?"* — is answered by the three-way rule in §0, applied in §6.

Artifacts (beside this file): `ko_boundary_decodability_2026-08-31.json` (every population
number) · `ko_boundary_constructed_2026-08-31.json` (the constructed sweep) ·
`ko_boundary_decodability_probe.py` (population: `labels|features|probe|report`, resumable) ·
`ko_boundary_constructed_probe.py` (constructed; imports `starmie_ttar_risk_probe`'s construction
verbatim and does not modify it). Checkpoint: `models/ai_v9_59_R2ACTION_0827/final_model.zip` —
the common fork base, FROZEN, `torch.no_grad()` throughout.

## HEADLINE

**The knockout information is present, in closed form, and the head does not use it.**

| where | what it says about P(KO) |
|---|---|
| the raw 2501-dim observation | linear R² **+0.066** — the input does NOT carry it |
| the `DamageOperator`'s `pko` channel | tracks the exact 16-roll truth at **slope +1.166, r = 0.995**; reads Hydro Pump's 0.8000 exactly |
| **`value_pooled` — the win-prob head's own input** | a LINEAR probe gets **AUC 0.845** (0.798 on open-race states) |
| **the win-prob head's output** | **AUC 0.588** — and **0.517, chance,** on open-race states |

**Verdict: EXPRESSIVENESS IS NOT THE CONSTRAINT — the GLU program is not supported by this
evidence.** The `WinProbHead` is a 2-layer MLP, strictly more expressive than the linear map that
beats it by 0.28 AUC on its own input. A gated probe adds **+0.10 R² on raw physics scalars** and
**≤ +0.05, often negative, on the learned tensors a GLU would be inserted into**. What binds is
SUPERVISION (one Bernoulli episode outcome copied onto every step) and, for our own knockouts,
DELIVERY (the outgoing KO channel has no route into the critic at all).

---

## 0. The three-way rule, stated before it is applied

| result | verdict |
|---|---|
| a LINEAR probe on frozen features recovers true P(KO) well | expressiveness is NOT the constraint — the defect is SUPERVISION (route-2 labels). **The GLU program is not supported by this evidence.** |
| linear fails, a small NONLINEAR/GATED probe succeeds | the information is present but needs multiplicative extraction — **evidence FOR the gated-mechanism program** (quantify the gap) |
| neither recovers it, and the raw-obs floor also fails | the roll/variance structure never reaches the representation — **OBS / DamageOperator COVERAGE work**; GLU would be solving the wrong layer |

## 0.1 Registered predictions (scored in §7, never tuned)

1. **P1** — the DamageOperator delivers a point-estimate-like damage quantity, not a roll
   distribution, so the variance needed for P(KO) is **NOT** explicitly present.
2. **P2** — despite that, a linear probe on trunk features recovers P(KO) at **R² ≥ 0.5**
   ⇒ supervision, not expressiveness, is the binding constraint.
3. **P3** — registered mid-probe, after the routing code in §2 was read and the constructed sweep
   in §3 was run, but **before any population number was computed** (text pinned as git blob
   `5ea97e01026780dbbe5b2d5c1c9f6e3a555d1aee`): the decodability gap between the op tap and
   `value_pooled` will be **larger for the outgoing target** (P(we KO them), unrouted to the
   critic) **than for the incoming target** (P(they KO us), routed).

---

## 1. THE DAMAGE-OPERATOR FACT — asked for by name, and it is not what P1 assumed

**`src/agents/model/damage_op.py:484`, the whole of it:**

```python
ko = acc * torch.clamp((dmg - cur_hp) / (0.15 * dmg + eps), 0.0, 1.0)
```

with the docstring one line above (`damage_op.py:471-475`) calling it *"the accuracy-discounted
P(KO this turn) vs CURRENT HP (`acc·P(KO|hit)`, the exact realized KO probability — accuracy and
the roll are independent events)"*.

**Read it as physics.** gen3's damage roll is uniform over the 16 values
`(85..100)/100 × dmg` (`_DMG_ROLL_MIN = 0.85`, `damage_op_layout.py:50`). The fraction of that
band lying at or above the defender's current HP is exactly `(dmg − cur_hp) / (0.15·dmg)`. So the
expression above **is** the continuous-uniform limit of the 16-roll KO fraction, multiplied by
accuracy. It is not a threshold on a point estimate and it is not a heuristic — it is the closed
form of the quantity.

So the honest headline is a **both**:

- **NO DISTRIBUTION.** There is no variance channel, no per-roll enumeration, no second moment
  anywhere in the block. What ships is three scalars — `high` (the 1.00 roll), `low` (the 0.85
  roll) and `crit` (×2 pre-screen) — plus `pko` and `acc`, five per type channel
  (`_DMG_CHANNEL_FEATS = 5`). Nothing downstream can compute a spread the op did not give it.
- **BUT THE ONE FUNCTIONAL OF THE SPREAD THAT P(KO) NEEDS IS COMPUTED IN CLOSED FORM AND SHIPPED
  AS ITS OWN CHANNEL** — per (our request-slot move × their active) in the outgoing direction
  (`_DMG_OUT_PER_MOVE = 4` = `[low, high, crit, pko]`, `damage_op_layout.py:109`) and per (their
  believed move × our mon) in the incoming direction (`_DMG_IDX_PHYS_PKO`/`_DMG_IDX_SPEC_PKO`).

P1's *observation* is right and P1's *consequence* is wrong. The variance is not present as a
variance; the KO probability the variance implies is present as a number. Three known deviations
from the exact truth, each a bias and none a missing capability: the continuous limit smooths the
1/16 discreteness; **crit is excluded from `ko`** (`_DMG_CRIT_P` at `damage_op_layout.py:75` is
documentation — it is never multiplied into any channel), so the op's outgoing pko has no
`+1/16` floor; and the opponent's bulk/max-HP is a hidden quantity the outgoing block estimates
with a neutral 0-EV spread.

---

## 2. THE ROUTING FACT — the outgoing KO channel has no path to the critic

Since the v96 critic-route deletion wave, `vf_combined` **is** `value_pooled`, and exactly one
route injects into it (`extractor_forward.py:766-796`, `_value_pooled_routes`, one live member).
That member is `UnifiedValueReadout`, and the op tensor it is handed is named at
**`extractor_forward.py:786`**:

```python
_op_rows = (self.damage_op.last_tensors.incoming_rows
            if (self.damage_op is not None and damage_block is not None) else None)
```

`incoming_rows` is `[B, 6, 12]` — **their** threat to **our** six mons. The readout's own
signature (`value_readouts.py:69-78`) accepts `op_rows [B,6,per_mon]` and nothing else from the
op.

**Consequence, stated as a fact about the graph and not as an interpretation:** the win-prob head
reads `value_pooled` (`extractor_forward.py:727`), `value_pooled` receives from the op only the
INCOMING rows, and the OUTGOING block — the one that carries "does my Surf kill their Tyranitar"
— reaches the policy through the pointer MOVE cell (`damage_op_layout.py:181`) and reaches the
critic **not at all**, except as attention *biases* on the trunk (the `d1`/`d2` edge families),
which are softmax-normalised ratios and structurally cannot carry a magnitude.

That asymmetry is what P3 predicts should show up as a measured decodability gap.

---

## 3. THE CONSTRUCTED ARM — the same sweep, one layer lower

`ko_boundary_constructed_probe.py` re-runs v1's Starmie/Tyranitar construction verbatim (same
teams, same parity-rule choreography, same decision-state asserts, same capture seed
`[7,11,13,17]`, same measured 16-roll Surf table) across v1's 17 sweep points ∪ v2's five
crossover micro-steps — **22 points, seven of them consecutive H across the k=13→12 boundary** —
and reads the DamageOperator's own outgoing `pko` for the Surf slot beside the head.

`op pko` below is the **PRE-GAIN** cell (`DamageOperator.last_out_cells`); the flat block's copy
is multiplied by a learned per-feature gain, measured here at ×1.3505 on this channel, and
quoting that would be quoting a scale factor as a probability.

| H | k | true E[KO\|Surf] | **op pko (Surf)** | true E[KO\|Pump] | **op pko (Pump)** | P(surf) | win-prob |
|---|---|---|---|---|---|---|---|
| 284 | 16 | 1.000 | 0.6809 | 0.800 | **0.8000** | 0.9829 | 0.9168 |
| 286 | 15 | 0.941 | 0.6388 | 0.800 | 0.8000 | 0.9808 | 0.9161 |
| 289 | 14 | 0.883 | 0.5756 | 0.800 | 0.8000 | 0.9771 | 0.9149 |
| 293 | 13 | 0.824 | 0.4915 | 0.800 | 0.8000 | 0.9709 | 0.9135 |
| 294 | 13 | 0.824 | 0.4704 | 0.800 | 0.8000 | 0.9690 | 0.9131 |
| **295** | **13** | **0.824** | **0.4494** | 0.800 | 0.8000 | 0.9671 | 0.9127 |
| **296** | **12** | **0.766** | **0.4283** | 0.800 | 0.8000 | 0.9649 | 0.9123 |
| 297 | 12 | 0.766 | 0.4073 | 0.800 | 0.8000 | 0.9627 | 0.9119 |
| 298 | 12 | 0.766 | 0.3862 | 0.800 | 0.8000 | 0.9603 | 0.9115 |
| 299 | 12 | 0.766 | 0.3652 | 0.800 | 0.8000 | 0.9577 | 0.9111 |
| 300 | 11 | 0.707 | 0.3441 | 0.800 | 0.8000 | 0.9549 | 0.9107 |
| 303 | 10 | 0.648 | 0.2810 | 0.800 | 0.8000 | 0.9453 | 0.9095 |
| 306 | 9 | 0.590 | 0.2179 | 0.800 | 0.8000 | 0.9334 | 0.9082 |
| 310 | 8 | 0.531 | 0.1337 | 0.800 | 0.8000 | 0.9111 | 0.9065 |
| 313 | 7 | 0.473 | 0.0705 | 0.800 | 0.8000 | 0.8892 | 0.9053 |
| 316 | 6 | 0.414 | 0.0074 | 0.800 | 0.8000 | 0.8628 | 0.9040 |
| 320 | 5 | 0.355 | **0.0000** | 0.800 | 0.8000 | 0.8551 | 0.9026 |
| 323 | 4 | 0.297 | 0.0000 | 0.800 | 0.8000 | 0.8516 | 0.9015 |
| 326 | 3 | 0.238 | 0.0000 | 0.800 | 0.8000 | 0.8481 | 0.9005 |
| 330 | 2 | 0.180 | 0.0000 | 0.800 | 0.8000 | 0.8435 | 0.8991 |
| 333 | 1 | 0.121 | 0.0000 | 0.800 | 0.8000 | 0.8400 | 0.8980 |
| 337 | 0 | 0.062 | 0.0000 | 0.800 | 0.8000 | 0.8354 | 0.8967 |

**Amplitude against the same truth, over the same 22 states:**

| quantity | span | OLS slope vs true E[KO\|Surf] | pearson r |
|---|---|---|---|
| truth | 0.9375 | 1.000 | 1.000 |
| **op pko (Surf), pre-gain** | **0.6809** | **+0.795** | **+0.952** |
| op pko, restricted to its live range (16 pts, pko > 0) | — | **+1.166** | **+0.995** |
| P(surf) — the action mask | 0.1475 | +0.193 | +0.979 |
| **win-prob head** | **0.0201** | **+0.0218** | **+0.9987** |

**Three readings.**

1. **The KO quantity is right there, at essentially unit gain.** Over the range where the op's
   pko is live it tracks the exact 16-roll truth at slope **+1.166** with **r = 0.995**. On the
   Hydro Pump slot — where every roll kills, so the whole answer is the accuracy — the op reads
   **0.8000 at all 22 points**, the exact truth to four decimals. The op is not approximating the
   KO question; it is answering it.
2. **The head, on the same forward, moves at 0.0218 of the required slope.** The win-prob head's
   rank correlation with the truth is nearly perfect (**r = 0.9987**) and its amplitude is
   **~1/46th** of it. The v2 "~40× compression" is reproduced here as a slope ratio of
   **0.0218 / 1.166 = 1/53** *between the head and the op channel one layer below it*. The
   compression is not at the input.
3. **The op has an OFFSET defect, and it is a different defect from the head's.** The op's pko
   hits exactly 0 from H = 320 upward, where the truth still runs 0.355 → 0.062 — because its
   estimate of Surf's max roll is 0.925 of the target's max HP against a true 336/342 = 0.982
   (a ~5.8% low damage estimate, from the neutral-0-EV defender bulk), and because `ko` carries
   no crit term so it cannot express the 1/16 = 0.0625 floor. That is a **calibration/coverage
   bug in the physics**, fixable in the operator; it is not a missing capability, and it is not
   what makes the head flat.

**The no-kink finding, correctly attributed.** Neither the op nor the head shows a kink at the
295→296 boundary where the truth steps 5.9pp. For the **op** that is *correct behaviour*: the
continuous-uniform limit has no kink, and its smooth ramp is the right smooth answer (its local
slope is 1.17× the truth's average slope). For the **head** it is the defect v2 named. v2's
"the mask prices the HP bar, not the roll table" therefore sharpens: **the op prices the roll
table and hands the head a number; the head prices the HP bar anyway.**

---

## 4. THE POPULATION ARM — 9,119 real decisions with Monte-Carlo ground truth

### 4.1 Construction

**States.** Every `move_selection` decision at turn ≥ 2 in all **474** `ai_v9_59_R2ACTION_0827`
eval-trace battles carrying a `*_reconstruction.json` (both eval steps, all 14 opponent
directories). 10,379 decisions were labelled; the 9,119 whose recorded action was a MOVE (so the
outgoing move cell is defined) form the frame. **Zero re-roll errors.**

**Ground truth.** For each state the recorded turn is re-rolled **R = 64** times through
`reroll_many(..., impl="rust")` with **both** sides' actions held at `"recorded"` — only the dice
change. The label is the fraction of re-rolls in which that side's alive-count fell:

- **`p_out` = P(the opponent loses a mon this turn)** — our knockout,
- **`p_in` = P(we lose a mon this turn)** — theirs.

664,256 rollouts in 15 minutes on 2 niced cores. R = 64 puts a single label's own sd at ≤ 0.0625,
which sets a **binomial-noise R² ceiling of 0.997** — i.e. label noise is not what limits any row
below.

**The population is not a disguised HP sweep.** Opponent HP fraction explains only **R² = 0.141**
of `p_out` (and our HP **0.195** of `p_in`) in-sample: unlike the constructed cell, the KO
fraction here is not a monotone function of one displayed bar, so the degenerate-probe risk the
mission flagged does not apply to this arm.

| | `p_out` | `p_in` |
|---|---|---|
| mean · variance | 0.259 · 0.159 | 0.209 · 0.141 |
| fraction ≈ 0 (< 0.02) | 51.7% | 63.4% |
| fraction ≈ 1 (> 0.98) | 15.2% | 12.5% |
| **"live" (0.02 – 0.98)** | **33.2%** (3,023) | **24.1%** |

**Features.** One frozen `torch.no_grad()` forward per recorded observation, nine taps read off
the same forward. Grouped **5-fold CV over BATTLES** — a battle never appears in both train and
test — with per-fold standardisation fit on the train half only, ridge `alpha` chosen on an inner
80/20 split, and the two neural probes early-stopped on their own inner 15% validation split.
Predictions are clipped to [0,1] before scoring, uniformly.

### 4.2 `p_out` — P(we knock one out) — the tap × probe-class table

**Full frame, n = 9,119 over 474 battles.** Out-of-fold R²; `sd` is across the 5 folds.

| tap | dim | **linear R²** | MLP R² | **GLU R²** | gated gain | linear AUC | GLU AUC |
|---|---|---|---|---|---|---|---|
| `obs_raw` — the coverage FLOOR | 2501 | +0.066 (sd .041) | +0.028 | +0.008 | −0.058 | 0.704 | 0.699 |
| `hp_only` — the HP-bar control | 2 | +0.148 | +0.186 | +0.185 | +0.037 | 0.705 | 0.723 |
| **`op_pko` — ONE scalar** | **1** | **+0.398** | +0.397 | +0.397 | −0.001 | **0.843** | 0.845 |
| `op_out_move` | 5 | +0.414 | +0.455 | +0.451 | +0.036 | 0.872 | 0.892 |
| `op_flat` | 138 | +0.308 | +0.405 | +0.410 | **+0.102** | 0.847 | 0.884 |
| **`op_move_cell` (policy's cell)** | 62 | **+0.450** | +0.489 | **+0.490** | +0.040 | 0.890 | **0.905** |
| `pi_features` | 512 | +0.375 | +0.390 | +0.389 | +0.014 | 0.868 | 0.871 |
| `vf_features` | 512 | +0.309 | +0.293 | +0.290 | −0.019 | 0.840 | 0.832 |
| **`value_pooled` — what the win-prob head reads** | 128 | **+0.314** | +0.314 | +0.327 | +0.013 | **0.845** | 0.846 |
| | | | | | | | |
| *the win-prob head's OWN output, unfitted* | 1 | *r = +0.138* | — | — | — | ***0.588*** | — |
| *the scalar value head's output, unfitted* | 1 | *r = +0.161* | — | — | — | *0.596* | — |

**Live subset (0.02 < p_out < 0.98), n = 3,023 over 474 battles** — the states where the knockout
race is genuinely open, i.e. exactly the boundary states the mission is about:

| tap | linear R² | MLP R² | GLU R² | linear AUC | GLU AUC |
|---|---|---|---|---|---|
| `obs_raw` | +0.044 | −0.005 | −0.014 | 0.665 | 0.659 |
| `hp_only` | +0.046 | +0.117 | +0.110 | 0.605 | 0.680 |
| `op_pko` (1 scalar) | +0.441 | +0.453 | +0.444 | 0.836 | 0.842 |
| `op_out_move` | +0.455 | +0.507 | +0.472 | 0.861 | 0.868 |
| `op_flat` | +0.341 | +0.435 | +0.437 | 0.843 | 0.873 |
| **`op_move_cell`** | **+0.498** | **+0.527** | +0.518 | 0.880 | **0.889** |
| `pi_features` | +0.332 | +0.328 | +0.331 | 0.831 | 0.836 |
| `vf_features` | +0.188 | +0.141 | +0.137 | 0.757 | 0.735 |
| **`value_pooled`** | **+0.259** | +0.233 | +0.241 | **0.798** | 0.790 |
| | | | | | |
| *win-prob head's own output* | *r = +0.034* | — | — | ***0.517*** | — |
| *scalar value head's output* | *r = +0.057* | — | — | *0.536* | — |

### 4.3 What those two tables say

1. **The head is at chance on the states in question, and a LINEAR read of the tensor it consumes
   is not.** On the live subset the win-prob head's output correlates with the true KO probability
   at **r = +0.034, AUC 0.517** — indistinguishable from a coin. A ridge regression on
   `value_pooled`, the *same 128 numbers the head is handed*, reaches **AUC 0.798 / R² 0.259**.
   The head is not missing the information; it is not using it.
2. **A single unfitted scalar beats every trunk tap.** `op_pko` — one number, the op's own
   outgoing KO probability for the chosen move — carries **R² 0.398 / AUC 0.843** on the full
   frame and **0.441 / 0.836** on the live subset, and a nonlinear probe on it adds **nothing**
   (−0.001). Its raw correlation with the MC truth, with no fitting at all, is **+0.631** against
   the win-prob head's **+0.138**.
3. **The RAW OBSERVATION does not carry it.** `obs_raw` — 2,501 dimensions, the whole input —
   manages **R² 0.066** linearly and *worse* nonlinearly (overfit at 2501 dims on ~7,300 training
   rows). P(KO) is a ratio of a product of stats to a hidden HP total; it is not a linear or
   easily-learned function of the input. **This is the DamageOperator earning its existence**: the
   quantity appears in the representation only because the op computes it.
4. **The gated form's gain is real, small, and lands in the wrong place for the proposal.** The
   largest GLU-over-linear gain in either table is **+0.102 R² on `op_flat`** — the 138 RAW
   PHYSICS numbers, where a multiplicative unit is doing exactly what one would expect (forming
   damage÷HP-type combinations the linear map cannot). On the network's own learned
   representations the gain is **+0.013 on `value_pooled`** (full frame) and **−0.018** (live
   subset); on `vf_features` it is negative in both. A gated mechanism inserted into the learned
   pathway — which is what the GLU program proposes — is not where the measured headroom is.
5. **The two heads fail differently, and §2 says why.** The POLICY's own cell
   (`op_move_cell`, 62 dims) is the best tap in every table (R² 0.450 / 0.498), because the
   outgoing pko is routed into it; the policy's realized use of it is the constructed arm's
   slope 0.193. The CRITIC's tensor is 0.19–0.16 R² *behind* that best op tap, consistent with
   §2's finding that the outgoing block has no route into `value_pooled` at all.

### 4.4 `p_in` — P(they knock one out) — the other direction, and the routing test

Same 9,119 states, same taps, same folds; only the target changes. `p_in` is the quantity the op
routes to the critic (`incoming_rows`), so P3 predicted the op-tap-vs-`value_pooled` gap would be
*smaller* here.

| tap | dim | linear R² | MLP R² | GLU R² | linear AUC | GLU AUC |
|---|---|---|---|---|---|---|
| `obs_raw` | 2501 | +0.155 | +0.089 | +0.087 | 0.775 | 0.766 |
| `hp_only` | 2 | +0.207 | +0.259 | +0.258 | 0.769 | 0.800 |
| `op_pko` *(the OUTGOING pko — wrong direction, a sanity check)* | 1 | **+0.007** | +0.011 | +0.010 | **0.546** | 0.558 |
| `op_out_move` *(also outgoing)* | 5 | +0.083 | +0.128 | +0.100 | 0.685 | 0.705 |
| `op_flat` *(contains `incoming_rows`)* | 138 | +0.290 | +0.393 | +0.416 | 0.847 | 0.896 |
| **`op_move_cell`** | 62 | +0.486 | **+0.580** | +0.580 | 0.912 | **0.934** |
| **`pi_features`** | 512 | **+0.499** | +0.520 | +0.515 | **0.920** | 0.918 |
| `vf_features` | 512 | +0.293 | +0.342 | +0.328 | 0.853 | 0.862 |
| **`value_pooled`** | 128 | +0.322 | +0.370 | +0.369 | 0.859 | 0.878 |
| | | | | | | |
| *win-prob head's own output* | 1 | *r = **−0.254*** | — | — | *0.347 → **0.653** inverted* | — |
| *scalar value head's output* | 1 | *r = −0.272* | — | — | *0.342 → 0.658 inverted* | — |

**The sanity check passes cleanly**: the outgoing `op_pko` — a genuinely informative scalar for
`p_out` (R² 0.398, AUC 0.843) — collapses to **R² 0.007, AUC 0.546** on `p_in`. The taps are
measuring what they claim to measure, and no tap is winning on a nuisance correlate.

**The head reads the two directions differently.** Its output separates "we are about to lose a
mon" at **AUC 0.653** (sign-inverted, as it must be: higher win probability ⇒ lower P(we lose
one)) and "we are about to KO one" at **AUC 0.588**, dropping to chance (0.517) on the open-race
subset. Some of the KO structure does reach the head; it is the *incoming* half, which is the
half §2 shows is wired to it.

**Live subset (0.02 < p_in < 0.98), n = 2,199 over 456 battles:** `op_move_cell` +0.378 lin /
+0.461 glu (AUC 0.848 / 0.867) · `pi_features` +0.340 (0.828) · `value_pooled` +0.192 lin
(**AUC 0.758**) · `op_flat` +0.223 (0.770) · `obs_raw` −0.012 (0.637) · *win-prob head r = −0.092,
AUC 0.453 → **0.547** inverted*.

### 4.5 The head against a linear read of its own input — all four cells

| cell | n | **win-prob head, AUC** | **ridge on `value_pooled`, AUC** | head R²-equivalent (pearson r) |
|---|---|---|---|---|
| `p_out`, full frame | 9,119 | 0.588 | **0.845** | +0.138 |
| `p_out`, live | 3,023 | **0.517** *(chance)* | **0.798** | +0.034 |
| `p_in`, full frame | 9,119 | 0.653 *(inverted)* | **0.859** | −0.254 |
| `p_in`, live | 2,199 | 0.547 *(inverted)* | **0.758** | −0.092 |

The head is beaten by a linear map of its own 128 inputs by **0.21 – 0.28 AUC in every cell**, and
on the open-race states of the outgoing direction it is at chance while that linear map is at
0.798. This is the single comparison the verdict rests on, and it holds dimension, states, folds
and information set fixed — only the readout changes.

### 4.6 The routing test (P3), reported both ways because the registration was under-specified

Gap = (best probe R² at an op tap) − (best probe R² at `value_pooled`). P3 predicted a **larger**
gap for the unrouted outgoing target.

| comparator | `p_out` (outgoing, **not** routed to the critic) | `p_in` (incoming, routed) | P3 |
|---|---|---|---|
| `op_flat` — the op's own output block, full frame | 0.410 − 0.327 = **0.082** | 0.416 − 0.370 = **0.046** | ✓ |
| `op_flat`, live | 0.437 − 0.259 = **0.178** | 0.225 − 0.192 = **0.033** | ✓ |
| `op_move_cell` — the policy's pointer cell, full frame | 0.490 − 0.327 = 0.163 | 0.580 − 0.370 = **0.211** | ✗ |
| `op_move_cell`, live | 0.527 − 0.259 = 0.268 | 0.461 − 0.192 = 0.269 | tie |

**The `op_move_cell` reversal has a known cause and it is a second routing gap, not a
counterexample.** `--intent-threshold` (v84) puts the α-weighted **`p_KO` — "am I about to die"**
— into the pointer MOVE cell, and the v96 critic-route deletion wave **deleted its vf route**
(dV 0.155/0.136 against a 0.39 bar) while the policy cell kept it. So on `p_in` the move cell is
carrying a channel the critic no longer receives, which inflates that comparator on exactly the
target P3 expected to be the routed one. That is not a contradiction of v96 — a **low dV means
the critic was not LEANING on the route**, which is what a head that ignores the information it
has would produce, and it is the standing lesson of this tree restated: *dV measures dependence,
not coverage.*

---

## 5. Limits, stated before the verdict

1. **"P(KO)" here is a TURN outcome, not a move outcome.** The label is "did that side's
   alive-count fall during this turn" under both sides' recorded actions. It therefore also
   counts a knockout by Explosion, recoil, sandstorm or a switched-in mon dying — and it counts
   0 when the opponent switched and our move hit somebody else. That definition is *natural* (it
   is the event the critic should price) but it is not identical to "does my move kill their
   active", which is what `op_pko` computes. Every mis-alignment of that kind **penalises the
   `op_pko` row**, so its numbers are, if anything, a floor.
2. **Decodability is not use, and this measurement cannot show use.** A probe says the
   information is linearly present in a tensor. It does not show that the downstream head's
   weights read it, nor that training on better labels would make them. The verdict below is
   about which of three constraints binds, not a promise that route-2 labels will work.
3. **One checkpoint.** `ai_v9_59_R2ACTION_0827`, one architecture (v100-era). The constructed arm
   additionally re-uses one hand-built state family whose *distribution* is off-policy
   (gen3customgame, an explosion-parade prelude, a 4-attack Starmie vs a 0-Spe CB Tyranitar); its
   obs pipeline is genuine, its state distribution is not. **The population arm exists precisely
   because the constructed arm cannot carry a population claim**, and the two agree.
4. **The traces were recorded by a nearby checkpoint, not the probed one.** The eval traces come
   from `step_26000016` / `step_28000032` snapshots; features here are from `final_model.zip`
   (recorded vs recomputed win-prob correlate at **0.981**, max |Δ| 0.388). Every head number
   quoted in §4 is the RECOMPUTED one from the probed network, so the comparison is internally
   consistent; the recorded `win_probs` are used nowhere in the verdict.
5. **High-dimensional taps are handicapped for the NEURAL probes, not for the linear one.** Ridge
   is well-behaved at p > n; a 64-unit MLP on 2,501 inputs with ~7,300 training rows is not, and
   `obs_raw`'s negative gated R² is that, not a statement about the input. The load-bearing
   comparisons here are **within a tap** (linear vs gated) and **head-vs-probe on the same
   tensor**, both of which hold dimension fixed.
6. **The `op_pko` and `op_out_move` taps are read at the RECORDED action's move slot**, so they
   inherit whatever selection the policy's own behaviour imposes. `value_pooled`, `pi_features`,
   `vf_features`, `op_flat` and `obs_raw` are action-independent and carry no such selection —
   and they are the taps the verdict rests on.
7. **The subset cut is post-hoc but classifier-blind.** "Live" is defined on the LABEL, never on
   any model output, and the identical rows go to every tap and every probe class, so it cannot
   favour one row of the table over another.

---

## 6. The three-way verdict

### Branch 1 — **EXPRESSIVENESS IS NOT THE CONSTRAINT. The GLU program is NOT supported by this
### evidence.**

The rule's first branch fires, and it fires on an argument that does not need an arbitrary R² bar.

**The `WinProbHead` is `LayerNorm → Linear(128,128) → ReLU → Linear(128,1)`
(`aux_value_heads.py:16-47`) — strictly more expressive than a linear map of the same input.** On
its own input tensor, a *linear* map reaches **AUC 0.845 / R² 0.314** against the true KO
probability (0.798 / 0.259 on the open-race states); the head itself reaches **AUC 0.588**, and
**0.517 — chance — on the open-race states**. A readout with *less* capacity than the head, on
*exactly* the head's input, outperforms it by 0.26–0.28 AUC. Whatever is stopping the head, it is
not the shape of the function it is allowed to represent, and adding a multiplicative gate would
enlarge a hypothesis class that is already too large rather than too small.

**The direct measurement of the gate agrees, and it localises where a gate DOES pay.** GLU R²
minus linear R², all nine taps × all four cells:

| tap | `p_out` all | `p_out` live | `p_in` all | `p_in` live |
|---|---|---|---|---|
| `obs_raw` | −0.058 | −0.058 | −0.068 | −0.079 |
| `hp_only` (raw scalars) | +0.037 | +0.064 | +0.051 | +0.080 |
| `op_pko` (1 scalar, already the answer) | −0.001 | +0.003 | +0.002 | +0.000 |
| `op_out_move` (raw physics) | +0.036 | +0.017 | +0.017 | +0.034 |
| **`op_flat` (138 raw physics scalars)** | **+0.102** | **+0.096** | **+0.127** | +0.002 |
| `op_move_cell` | +0.040 | +0.021 | +0.093 | +0.082 |
| **`pi_features` (learned)** | +0.014 | −0.001 | +0.016 | −0.005 |
| **`vf_features` (learned)** | −0.019 | −0.051 | +0.034 | −0.040 |
| **`value_pooled` (learned — the head's input)** | +0.013 | −0.018 | +0.047 | −0.027 |

**A gate pays where you are combining RAW numbers and does not pay on the network's own learned
representations.** `op_flat` (+0.10 to +0.13 in three of four cells) is a multiplicative unit
doing exactly what one would expect on 138 unmixed physics scalars — forming the damage÷HP-type
ratios a linear map cannot. On `pi_features` / `vf_features` / `value_pooled` the gate buys
**≤ +0.047 and is NEGATIVE in 5 of those 12 cells**. And on `op_pko` — one scalar that already
*is* the ratio — it buys **±0.003**, which is the cleanest possible statement of the principle:
once the multiplicative work is done, gating adds nothing.

The GLU proposal is to add a gate inside the learned pathway. That is the column where the
measured gain is zero-to-negative.

### Branch 2 — the actual defect is **SUPERVISION and DELIVERY**, in two separable halves

**(a) Supervision.** The quantity is linearly present in the head's own input and the head does
not use it. Nothing about the architecture prevents the current head from reading `value_pooled`
the way the ridge does; what would change its weights is a target that rewards resolving the
boundary — and **the target it currently has provably cannot**. `WinProbHead`'s own docstring
(`aux_value_heads.py:22-24`) states the supervision: *"the Monte-Carlo episode OUTCOME (win=1 /
loss=0) propagated to every step of the episode."* One Bernoulli draw per game, copied onto every
decision in it. That label is unbiased for the LEVEL and carries almost no information about how
this state differs from its neighbour — which is the definition of a target that produces
aggregate calibration with no resolution. §4's labels are the opposite object: a per-state
probability from 64 independent draws of the same state. This is the same conclusion `exploiter_fingerprint_truthcheck`'s §"what this newly
convicts" reached from the opposite direction — *aggregate-calibrated, resolution-blind* — and
this probe adds the missing half: **the resolution the head lacks is available to it.** The
labels manufactured for §4 (664,256 rollouts, 15 minutes, 2 cores) are literally a route-2 label
factory output; the R1/R2 counterfactual-grounding line in
[`design_counterfactual_value_grounding.md`](../../ai_v10/design_counterfactual_value_grounding.md)
is the named vehicle.

**(b) Delivery.** For the OUTGOING half — "does my move kill theirs", the half the Starmie cell is
about — supervision alone may not be enough, because §2 shows the channel is **not wired to the
critic at all**: `_value_pooled_routes` hands `UnifiedValueReadout` only `incoming_rows`
(`extractor_forward.py:786`). This is a one-line-shaped structural gap of exactly the kind the
v89 `gen3_value_pooled_routes_v1` seam was built to make auditable, and it is testable before any
training: **route the outgoing rows into the value pool as a sixth source and re-run this probe's
`value_pooled` column.**

### Branch 3 — coverage work exists, but it is CALIBRATION of a channel that exists, not a missing capability

The rule's third branch does **not** fire — the raw-obs floor failing (R² 0.066) while the op tap
succeeds (0.450–0.498) is exactly the signature of *a representation that supplies what the input
does not*. But two real coverage defects were measured on the way and should be recorded as
their own work item, because both are biases in a shipped number:

1. **`ko` carries no crit term.** `_DMG_CRIT_P` (`damage_op_layout.py:75`) is documentation and is
   never multiplied in, so the op's KO probability cannot express gen3's 1/16 = 0.0625 floor.
   Measured cost in §3: the op reads exactly 0.0000 at six sweep points whose true KO probability
   runs 0.355 → 0.062.
2. **The outgoing block prices the defender with a neutral 0-EV bulk estimate**, measured in §3 at
   a **5.8% low** max-roll (0.925 of max HP against a true 0.982) — which is what pushes the whole
   ramp ~20 HP too early and zeroes the top third of the sweep.

Neither needs a new mechanism. Both are arithmetic inside `_rolls` and `_outgoing_block`.

### One sentence

**The knockout-roll structure reaches the DamageOperator at essentially unit gain (r = 0.995
against the exact 16-roll truth), survives into the win-prob head's own input tensor at AUC 0.845,
and comes out of that head at AUC 0.517 — so the defect is what the head is TAUGHT and (for our
own knockouts) what it is WIRED to, not what it is CAPABLE of.**

---

## 7. Predictions scored

**P1 — "the DamageOperator delivers a point estimate, not a roll distribution, so the variance
needed for P(KO) is NOT explicitly present": HALF RIGHT, and the half that mattered is WRONG.**
The observation is correct — there is no distribution, no variance channel, no per-roll
enumeration anywhere in the block (§1). The *consequence* is false: the operator ships the exact
closed form of the KO probability the roll spread implies, `acc·(dmg − cur_hp)/(0.15·dmg)`
(`damage_op.py:484`), as its own named channel in both directions. Measured against exact
ground truth it reads Hydro Pump's KO probability as **0.8000 at all 22 sweep points** (truth
0.8000) and tracks Surf's 16-roll KO fraction at **slope +1.166, r = 0.995** over its live range.
Had P1's consequence held, the verdict would have been branch 3; it is branch 1. *The lesson is
the mission's own framing turned back on it: "is the variance present?" and "is the functional of
the variance we need present?" are different questions, and only the second one decides.*

**P2 — "a linear probe on trunk features recovers P(KO) at R² ≥ 0.5": FAILS on the number,
holds on the conclusion.** Measured linear R² on the trunk taps: `pi_features` **+0.375**,
`value_pooled` **+0.314**, `vf_features` **+0.309** (live subset: 0.332 / 0.259 / 0.188). None
reaches 0.5; the best tap in the whole table is the op's own **`op_move_cell` at 0.450 / 0.498**,
also short of it. The 0.5 bar was set without knowing that ~half of the KO variance is
undecodable from *any* tap — part genuinely hidden information (the opponent's exact spread and
item are unknown, so P(KO) is not a function of the agent's information set), part the operator
calibration defects in §6 branch 3. **The conclusion P2 was making — supervision, not
expressiveness — stands, but it stands on the head-vs-linear-probe comparison on the SAME tensor
(AUC 0.845 vs 0.588), which is a fair test, and not on an absolute R² threshold, which was not.**

**P3 — "the op-tap-vs-`value_pooled` decodability gap will be larger for the outgoing target than
for the incoming one": PARTIAL, and the registration is at fault.** On `op_flat` — the
DamageOperator's own output block, which is the right comparator for a claim about the op→critic
path — the gap is **0.082 vs 0.046** (full frame) and **0.178 vs 0.033** (live), a clean pass in
both subsets. On `op_move_cell` it reverses (0.163 vs 0.211) for the reason given in §4.6: the
pointer cell carries an α-weighted `p_KO` whose *critic* route v96 deleted, so that comparator is
measuring a second, different routing gap. **P3 did not name its comparator and therefore does not
cleanly resolve; it is scored PARTIAL rather than claimed as a pass.** The routing story is
better supported by a reading P3 did not register: the head's own separation of the two
directions, **AUC 0.653 incoming (routed) vs 0.588 / 0.517 outgoing (unrouted)** — reported here
as an observation, not as a confirmed prediction.

### What the scoring cost

Two of three registered predictions failed on their stated terms and the verdict came out
*stronger* than any of them, from a comparison none of them named — the head against a linear map
of its own input. P1's failure is the instructive one: it asked whether the VARIANCE is present,
the answer was "no", and the answer to the question that actually decides — *is the functional of
the variance we need present?* — was "yes, in closed form, at unit gain."

---

## 7a. What this licenses, in order of cost

Ranked by (evidence strength × cheapness), not by appeal. None of these is a launch decision;
each is stated so the next person can pick one up.

1. **DO NOT build the GLU/gated-mechanism program on this evidence.** §6 branch 1. If it is
   built, it must be justified by something other than "the head can't produce sharp changes" —
   the head demonstrably *could*, from what it is already handed.
2. **Route the outgoing rows into `UnifiedValueReadout` as a sixth source and re-run this
   probe's `value_pooled` column** (§2 / §4.6). Pure offline test, needs no training run, and the
   `_value_pooled_routes` seam plus `value_route_gradient_test.py` already exist to make it
   auditable the day it is written. The prediction is registered by construction: `p_out`'s
   `value_pooled` decodability should rise toward `op_flat`'s.
3. **Fix the two operator calibration defects** (§6 branch 3): add the crit term to `ko`, and
   revisit the neutral-0-EV defender bulk (measured 5.8% low). Arithmetic inside `_rolls` /
   `_outgoing_block`; both are shipped biases in a number several consumers read.
4. **Feed the head a per-state label instead of an episode outcome** — the R1/R2 line in
   `design_counterfactual_value_grounding.md`. This measurement is a working label factory:
   **664,256 rollouts in 15 minutes on 2 niced cores** produced 10,379 per-state probabilities.
   Cost is not the obstacle.

## 7b. What this does NOT say

- It does not show that better labels **will** fix the head. Decodability is not use (§5.2).
- It does not exonerate the representation everywhere: **40–54%** of the KO variance is undecodable
  from *every* tap tried, part of it the genuine hidden-information floor (the opponent's exact
  spread and item are unknown, so P(KO) is not a function of the agent's information set) and
  part of it the §6-branch-3 calibration bias. This probe cannot separate those two.
- It does not touch the POLICY's failure, which is a different failure at a different head: the
  policy *is* handed the outgoing `pko` in its pointer cell (`op_move_cell` is the best tap in
  three of four cells) and still realises only **slope 0.193** of the truth on the constructed
  sweep, with the argmax never flipping. That is the knows≠uses shape the bait verdict convicted,
  and nothing here moves it.

---

## 8. Rerun

```bash
export PYTHONPATH=$PYTHONPATH:src
D=designs/research_state/measurements
# population arm (~8 min labels on 2 workers, ~1 min features, ~N min probe)
nice -n 15 python $D/ko_boundary_decodability_probe.py --phase labels --worker 0 --workers 2 &
nice -n 15 python $D/ko_boundary_decodability_probe.py --phase labels --worker 1 --workers 2 &
nice -n 15 python $D/ko_boundary_decodability_probe.py --phase features
nice -n 15 python $D/ko_boundary_decodability_probe.py --phase probe
nice -n 15 python $D/ko_boundary_decodability_probe.py --phase report
# constructed arm (~90 s, node bridge, no server)
nice -n 15 python $D/ko_boundary_constructed_probe.py
```

Reads `models/` via `main_models_dir()` (worktree-safe), rust re-roll driver for the labels, node
bridge for the constructed battles, CPU-only, 2 torch threads. Every table in §3 and §4 is
rendered from the JSON by `--phase report`; no number in them was copied by hand.

**The expensive artifact ships**: `..._labels_w{0,1}.jsonl.gz` (10,379 per-state MC probabilities,
60 KB each) are committed and `label_rows()` reads them directly, so `features` / `probe` re-run
from a fresh checkout in ~20 minutes **without re-paying the 664,256 rollouts** — and a resumed
`labels` phase seeds its done-set from them too. The 29 MB `_features.npz` is **not** committed:
it is one `--phase features` (70 s) away from the labels plus the checkpoint.

Approximate cost of the whole thing: labels 15 min (2 workers), features 70 s, probe ~45 min
(the 2501-dim `obs_raw` tap dominates), constructed arm 90 s.
