# THE HEAD REFIT — what the win head would have to be PAID to do before it conditions

Follow-up to [`winprob_probe_read_2026-09-09`](../winprob_probe_read_2026-09-09/README.md), which
established that `stash.value_pooled` — the tensor the win head literally reads — decodes the
opponent's class at AUC 0.846 and the trainee's own-team win rate at R² 0.671 on turn-1 states,
while `V = sigmoid(win_head(value_pooled))` reads 0.532 and 0.010. One MLP (LayerNorm → 128 → ReLU
→ 1) sits between those numbers. That measurement said the head has the information and does not
use it. It could not say whether the fault is in **how the head was optimised**, in **what it was
paid to predict**, or in **what the head can express** — and those three have three different
treatments. This one holds the features frozen and varies only the target and the optimiser.

Run 2026-09-09, offline, CPU only, nothing written under `models/`. Reproduce with `./run.sh`.

**VERDICT: (2), on BOTH substrates, and PARTIAL.** Refitting the same head architecture from
scratch on frozen `value_pooled` against the **terminal 0/1 outcome — the target the online head
actually trains on — reproduces the online head's failure exactly**: the turn-1–3 between-opponent
spread ratio moves from 0.149 to 0.000 on arm A (Δ **[−0.055, +0.234]**, straddles zero) and from
0.066 to 0.068 on the control (Δ **[−0.044, +0.042]**), and the prediction's turn-1 opponent-class
decode moves from AUC 0.532 to 0.546 on A (Δ **[−0.049, +0.081]**) and *falls* on the control.
Swap ONLY the target for the same label with its per-episode variance removed — the per-(cycle,
opponent) × own-team leave-one-battle-out empirical win rate — and the same head, the same
features, the same folds and the same optimiser recover a **DETECTED** part of the separation on
both: spread ratio 0.149 → **0.323** on A (Δ **[+0.063, +0.296]**) and 0.066 → **0.259** on CTRL
(Δ **[+0.051, +0.192]**), turn-1 opponent-class AUC 0.532 → **0.646** on A (Δ **[+0.059, +0.170]**),
own-team win-rate R² 0.010 → **0.569** on A and 0.028 → **0.467** on CTRL against `value_pooled`'s
own 0.671 / 0.628. **Fine-tuning from the online head's own weights lands in the same place as
training from scratch, under BOTH targets and on both substrates** — so the online head is *not*
trapped in a basin, and it is the target and not the optimisation history that decides where the
head ends up. And the recovery is partial: `mlp_cond` reaches 46 % (A) and 29 % (CTRL) of its own
target's ceiling — closing 31 % and 23 % of the gap from the online head to that ceiling, and under the conditional target — and only under it — the MLP beats the linear
floor with the delta's CI clear of zero, so a smaller **capacity** term survives underneath.

The mechanism is arithmetic and is in [`variance_shares.json`](variance_shares.json): only
**10.2 %** (A) / **14.4 %** (CTRL) of the terminal label's variance lies BETWEEN (cycle, opponent)
cells. A head minimising BCE buys its resolution far more cheaply from the board and from its own
team, and it does — the terminal refit lifts held-out Brier from 0.1328 to **0.1148** on A while
moving the opponent meter not at all. Removing the per-episode draw raises the between-cell share
to **24.0 %** (A) / **58.8 %** (CTRL), and the head then conditions. **The win head is not failing
to learn; it is being paid, correctly, to predict something else.**

---

## 1. The outcomes and their decisions — stated BEFORE the fit

| | what the numbers would look like | what it would mean | the treatment |
|---|---|---|---|
| **1** | (a) the TERMINAL-target refit closes the spread ratio toward 1 and its prediction decodes the opponent far above the online head | the target is learnable from these features at this data scale and the ONLINE run did not learn it — non-stationarity under a moving policy, the head's effective lr / epochs / gradient share, buffer reuse | how the head is **TRAINED**: a value replay, a periodic head refit, a head-specific lr |
| **2** | (a) fails and (b), the CONDITIONAL target, recovers | the per-episode label VARIANCE is what the head cannot see through at its effective sample size | **TARGET-VARIANCE** levers: MC / counterfactual labels, search leaves, outcome balancing |
| **3** | both fail while `value_pooled` still linearly decodes the opponent | the MAPPING features→outcome is what the head lacks | head **CAPACITY**, the LayerNorm-128 bottleneck, a separate value trunk |

and one guard the registration did not name but the data can force, checked FIRST:

| | | |
|---|---|---|
| **0** | `cond_oracle` — the conditional target emitted verbatim as a prediction — does not itself reach a ratio near 1 | (b)'s failure would say nothing about the head, because the label it was handed did not contain the separation either | (b) is **INCONCLUSIVE**, never folded into 3 |

`summarize.py` applies this rule **in code**, so the verdict per substrate is a function of the
numbers rather than of the author. A condition RECOVERS only when its turn-1–3 spread ratio
improves on the online head with the **delta's own** CI clear of zero AND the ratio's own CI lower
bound clears 0.80 AND its turn-1 opponent-class decode clears both its permutation null and the
online head's, again on the delta's own CI. Satisfying the first pair but not the ratio floor is
**PARTIAL** — a different treatment recommendation from either 1 or 2, and what both substrates
returned.

---

## 2. The frame

| | |
|---|---|
| substrate A | `ai_v12_02_winprob_critic` @74M, last four trace cycles — **29,495 states · 951 battles · 216 teams · 14 opponents** |
| substrate CTRL | `ai_v12_11_ladder_ctrl10M` @10M, all five cycles — **25,564 states · 873 battles · 180 teams · 12 opponents** (the substrate the ladder arms actually run on) |
| features | **NOT re-derived.** `pooled.npy` (`stash.value_pooled`) and `meta.npy` come verbatim from the probe read's ONE frozen forward per substrate. Re-deriving them would have risked a second, subtly different tensor and made the two measurements incomparable for no gain |
| the identity under test | `dump_head.py` lifts `features_extractor.win_head` out of each snapshot, re-applies it to `pooled.npy` in numpy, and **REFUSES** unless it reproduces the frozen forward's own `V`. It does: max \|Δ\| **2.3e-07** (A) and **2.8e-07** (CTRL). Without that check "refit the head" and "replace the online head" are not the same experiment |
| head architecture | `WinProbHead` verbatim — `LayerNorm(128) → Linear(128,128) → ReLU → Linear(128,1)`, asserted against the checkpoint's own parameter shapes. The linear floor keeps the LayerNorm and drops only the hidden layer, so "the MLP beat the linear head" cannot be a statement about feature scaling |
| folds | outer 5-fold **grouped by battle**, and the early-stopping split is a grouped split of the TRAINING battles. States in a battle share the team, the opponent AND the outcome |
| every prediction | **out of fold**, the online head included (it is a fixed function, so its OOF prediction is itself). The in-fold columns are kept and read separately — that difference IS the overfitting counter-hypothesis |
| weights | every fit and every score Horvitz–Thompson weighted by the manifest capture rates. Arm A's traced sample wins 0.42 of its battles against a true 0.83; an unweighted refit is a refit for a population that does not exist |
| optimiser | Adam, lr 1e-3, batch 1024, weight decay 0, max 1500 epochs, early stop on the grouped validation fold with patience 60. Median epochs run: 104 (`mlp_term`), 298 (`mlp_cond`) — no arm is capped except one `lin_cond` fold, and the linear floor loses on both targets anyway |

### The eight conditions

| condition | head | target | what it is for |
|---|---|---|---|
| `online` | the checkpoint's own | — | the reference row (c). ONE frozen snapshot, so the same weights and the same features as every refit |
| `online_rec` | — | — | the **conservative** reference: the RECORDED per-cycle win prob (each cycle's own model). Reproduces the mixture diagnostic's published numbers exactly |
| `lin_term` | linear | terminal 0/1 | the capacity FLOOR for (a) |
| `mlp_term` | `WinProbHead`, scratch | terminal 0/1 | **(a)** |
| `mlp_term_ft` | `WinProbHead`, init FROM the online weights | terminal 0/1 | the **ORDER hazard** |
| `lin_cond` | linear | conditional | the capacity floor for (b) |
| `mlp_cond` | `WinProbHead`, scratch | conditional | **(b)** |
| `mlp_cond_ft` | `WinProbHead`, from the online weights | conditional | the order hazard under (b) |
| `cond_oracle` | — | — | the conditional TARGET itself, emitted as a prediction: the **CEILING** |

### The conditional target, and why it is not a per-(opponent, team) cell mean

951 battles over 14 opponents and 216 teams leaves ~0.3 battles in the average joint cell, so the
literal cell mean is undefined almost everywhere. The target is therefore **additive in the two
factors on the win-rate scale**, every ingredient **leave-one-battle-out** and IPW-weighted:
`p = clip(base_LOO + s_o·(opp_LOO − base) + s_t·(team_LOO − base))`, each factor shrunk by
`n/(n+k)` with the two `k` chosen by minimising the HT-weighted battle-level log loss of the LOO
prediction itself — an honest out-of-sample criterion, because the label of battle *i* is built
without battle *i*. Selected: `k_opp` 16 / `k_team` 4 (A), 4 / 8 (CTRL); LOO log loss **0.357** vs
the marginal's 0.454 (A) and **0.393** vs 0.492 (CTRL), i.e. the conditional structure is real and
estimable at this sample size.

🚨 **The opponent factor is keyed by (cycle, opponent), not by opponent.** The identity is scored
over (cycle, opponent) cells — that is the unit the manifest reports a 100-game win rate for — and
a trainee's own strength moves between cycles. On the ladder control it moves *enormously*: its
per-cycle mean true win rate runs **0.547 at 2M → 0.896 at 8M**. A target with no cycle term
cannot reproduce those cell means however well it is fitted, and with the opponent-only key CTRL's
ceiling reads **0.420** for a reason that has nothing to do with the head. Keying by the cell
raises it to **0.898 [0.822, 0.984]** and leaves (a) untouched (the terminal target has no factors
at all). Both keyings are run; §10(e) reports the difference.

---

## 3. Arm A @74M — the headline rows

**The mixture identity** — between-opponent spread of the prediction ÷ between-opponent spread of
the outcome, each corrected for its own sampling noise. For any calibrated critic the two are
EQUAL. **Bold** = the delta's own battle-clustered CI is clear of zero.

| condition | turn 1 | turns 1–3 | all states | Δ ratio vs online @t1–3 |
|---|---|---|---|---|
| online (frozen forward) | 0.000 [0.098, 0.307] | 0.149 [0.190, 0.339] | 0.613 [0.545, 0.730] | — |
| online (recorded per-cycle) | 0.308 [0.297, 0.428] | 0.334 [0.317, 0.448] | 0.573 [0.507, 0.690] | — |
| linear · terminal | 0.315 | 0.150 | 0.566 | +0.001 [−0.031, +0.213] |
| **MLP · terminal, scratch (a)** | 0.000 | **0.000** | 0.585 | −0.149 [−0.055, +0.234] · *not detected* |
| MLP · terminal, fine-tuned | 0.000 | 0.057 | 0.601 | −0.092 [−0.040, +0.228] · *not detected* |
| linear · conditional | 0.000 | 0.000 | 0.239 | −0.149 [−0.153, +0.036] · *not detected* |
| **MLP · conditional, scratch (b)** | 0.152 | **0.323** [0.336, 0.549] | 0.381 | **+0.174 [+0.063, +0.296]** |
| MLP · conditional, fine-tuned | 0.171 | 0.291 | 0.386 | **+0.142 [+0.045, +0.272]** |
| **the conditional TARGET (ceiling)** | 0.702 | **0.702 [0.652, 0.894]** | 0.702 | **+0.553 [+0.383, +0.641]** |

**What the PREDICTION decodes** (the probe read's own rows, same grouped-CV ridge, same nulls,
same HT weights). `value_pooled` is the reference: it is what the head is handed.

| turn-1 decoder | opponent class (AUC) | own-team LOO win rate (R²) |
|---|---|---|
| `value_pooled` — the head's input | **0.846** [0.812, 0.876] | **0.671** [0.529, 0.789] |
| online (frozen forward) | 0.532 [0.488, 0.574] · *inside its null* | 0.010 [−0.011, 0.027] · *inside its null* |
| linear · terminal | **0.624**, Δ vs online [+0.024, +0.161] | **0.152**, Δ [+0.091, +0.191] |
| **MLP · terminal (a)** | 0.546, Δ **[−0.049, +0.081]** · *not detected* | **0.318**, Δ **[+0.240, +0.376]** |
| MLP · terminal, fine-tuned | 0.540, Δ [−0.059, +0.074] · *not detected* | **0.340**, Δ [+0.259, +0.399] |
| **MLP · conditional (b)** | **0.646**, Δ **[+0.059, +0.170]** | **0.569**, Δ **[+0.464, +0.648]** |
| MLP · conditional, fine-tuned | **0.649**, Δ [+0.058, +0.175] | **0.539**, Δ [+0.422, +0.634] |
| the conditional TARGET (ceiling) | **0.901** | **0.768** |

**Read the two columns against each other, because they do not say the same thing.** On the
OWN-TEAM axis the terminal refit already recovers a third of what the tensor carries (0.010 →
0.318 against `value_pooled`'s 0.671) — the head *can* be made to condition on something from 951
battles of 0/1 labels. On the OPPONENT axis it recovers **nothing** (0.532 → 0.546, CI through
zero), and at turns 1–3 it is *worse* than the online head (0.501 vs 0.671, Δ **[−0.214, −0.125]**).
The conditional target moves both, and moves the opponent axis to within 0.20 of the tensor.

**Brier on held-out battles, turns 1–3** — the window where the mixture defect is sharpest:

| condition | Brier | Δ vs online | reliability | resolution | skill vs base rate |
|---|---|---|---|---|---|
| online (frozen forward) | 0.1608 | — | 0.0231 | 0.0054 | **−0.129** |
| online (recorded) | 0.1536 | — | 0.0150 | 0.0036 | −0.078 |
| MLP · terminal (a) | **0.1335** | **[−0.0362, −0.0186]** | 0.0041 | 0.0141 | +0.063 |
| MLP · conditional (b) | **0.1190** | **[−0.0495, −0.0344]** | 0.0017 | 0.0237 | **+0.164** |
| the conditional TARGET (ceiling) | **0.1093** | **[−0.0607, −0.0423]** | 0.0049 | 0.0369 | +0.233 |

🚨 **On turns 1–3 the online win-prob critic's Brier SKILL is −0.129: it forecasts WORSE than the
constant base rate**, and its reliability term (0.0231) is four times its resolution (0.0054). Every
refit beats it, the linear floor included. Over ALL states the online head is skill +0.195 and the
terminal refit +0.304 (Brier 0.1328 → **0.1148**, Δ **[−0.0286, −0.0082]**) — so the head is
recoverable as a *forecaster* from these same frozen features; what it does not recover from the
terminal label is the *opponent*.

---

## 4. CTRL @10M — the substrate the ladder arms actually run on

The same shape, and on the opponent axis if anything cleaner.

| condition | ratio @turn 1 | @turns 1–3 | @all | Δ ratio vs online @t1–3 |
|---|---|---|---|---|
| online (frozen forward) | 0.000 [0.108, 0.234] | 0.066 [0.130, 0.257] | 0.487 | — |
| **MLP · terminal (a)** | 0.000 | 0.068 | 0.548 | +0.002 **[−0.044, +0.042]** · *not detected* |
| MLP · terminal, fine-tuned | 0.029 | 0.118 | 0.536 | +0.052 [−0.020, +0.046] · *not detected* |
| **MLP · conditional (b)** | 0.214 | **0.259** [0.260, 0.379] | 0.392 | **+0.193 [+0.051, +0.192]** |
| MLP · conditional, fine-tuned | 0.186 | 0.212 | 0.373 | **+0.146 [+0.006, +0.155]** |
| the conditional TARGET (ceiling) | 0.898 | **0.898 [0.822, 0.984]** | 0.898 | **+0.832 [+0.619, +0.797]** |

Own-team LOO win rate at turn 1, R²: `value_pooled` **0.628** · online **0.028** · terminal refit
**0.053** [0.009, 0.093] · **conditional refit 0.467 [0.387, 0.543]**. Brier on turns 1–3: online
0.1611 (skill +0.037) → conditional refit **0.1415**, Δ **[−0.0260, −0.0131]** (skill +0.155).

⚠️ **CTRL's opponent-CLASS axis is the one cell that does not move, and the reason is in this
measurement.** Because CTRL's cells span 2M→10M and its own strength quintuples across them, the
(cycle, opponent) cell means are dominated by the CYCLE, so the conditional target encodes
"how strong was the trainee at this cycle" far more than "is this opponent a sentinel" — and its
own class decode is 0.463, *inside its null*. A target that does not carry the class cannot teach
it, and no condition on CTRL decodes class above the online head. **The cell-level identity and the
own-team axis are the readable ones on CTRL; read its `opp_class` row on arm A instead**, where the
four cycles are flat (per-cycle true win rate 0.848 / 0.826 / 0.832 / 0.821) and the axis is clean.

⚠️ **`online_rec` reads a ratio of 1.079 on CTRL and that is NOT evidence the control's head
conditions.** The recorded per-cycle `V` tracks the trainee's own improvement (mean `V_rec` 0.386 at
2M → 0.837 at 8M) and the between-CELL variance it explains is overwhelmingly the cycle term, not
the opponent. The internally consistent comparator — one frozen snapshot over all five cycles,
which is what the refits' features come from — reads **0.066**. This is why every headline delta
here is against `online` and `online_rec` is reported beside it rather than instead of it.

---

## 5. The reading, per substrate

| substrate | (a) terminal | (b) conditional | ceiling reaches the identity | reading |
|---|---|---|---|---|
| **A** | **FAILS** (spread Δ and class-decode Δ both straddle zero; class Δ at t1–3 is NEGATIVE) | **PARTIAL** — DETECTED on spread and on both decodes, reaching 46 % of its own ceiling (31 % of the gap to it) | 0.702 [0.652, 0.894] — below the 0.80 floor, so the code emits an (b)-INCONCLUSIVE guard | **2, PARTIAL** |
| **CTRL** | **FAILS** (spread Δ straddles zero; class decode falls) | **PARTIAL** — DETECTED on spread and on own-team, reaching 29 % of its own ceiling (23 % of the gap to it) | **0.898 [0.822, 0.984]** — clears it | **2, PARTIAL** |

**The code's own words.** On CTRL `summarize.py` emits *"MIXED — (a) FAILS, (b) PARTIAL"*; on A it
emits the ceiling guard, because A's shrunk conditional target reaches only 0.702 and the rule
refuses to read (b) against a ceiling below 0.80. **Both are the same finding stated under
different ceilings, and the ceiling is a property of the TARGET's estimation noise, not of the
head**: with the opponent-only key (less shrinkage: 14 opponents × ~68 battles instead of 56 cells
× ~17) arm A's ceiling is **1.028 [0.919, 1.195]** and its conditional refit reaches **0.517** —
the same conclusion with the guard not tripped. Reported as the rule emitted it, and reconciled
here rather than quietly overridden. **The half of the reading that needs no ceiling at all is (a),
and (a) FAILS identically on both substrates and under every sensitivity.**

---

## 6. The mechanism — why the terminal target loses the opponent

`variance_shares.json`, HT-weighted over the traced state population:

| | total variance of the label | share BETWEEN (cycle, opponent) cells | share between own TEAMS |
|---|---|---|---|
| A · terminal 0/1 | 0.1649 | **10.2 %** | 46.0 % |
| A · conditional | 0.0367 | **24.0 %** | 93.1 % |
| CTRL · terminal 0/1 | 0.1595 | **14.4 %** | 38.2 % |
| CTRL · conditional | 0.0341 | **58.8 %** | 71.7 % |

Nine-tenths of the terminal label is per-episode noise plus within-cell board state. A head
minimising BCE buys its resolution wherever it is cheapest, and the board and the trainee's own
team are far cheaper than the opponent — which is exactly what the terminal refit does: it lifts
held-out Brier resolution from 0.0323 to 0.0513 on arm A while leaving the opponent meter
untouched. Removing the per-episode draw does not add any information the features did not
already carry; it **re-prices the objective**, raising the between-cell share 2.3× (A) and 4.1×
(CTRL). That is the whole difference between (a) and (b), and it is why the treatment class is
target-side.

This also sharpens the probe read's counter-hypothesis (f) — *"a calibrated head SHOULD shrink
toward the prior under a noisy target"* — from a caveat into the mechanism. It is not that the
between-opponent signal is weak; the outcome moves through 0.365 of win rate across opponents. It
is that the between-opponent component is 10 % of a proper scoring rule the head can reduce more
cheaply elsewhere.

---

## 7. The ORDER hazard — is the online head in a basin? No.

| contrast (scratch − fine-tuned) | Δ ratio @t1 | Δ ratio @t1–3 | Δ ratio @all | Δ Brier @all |
|---|---|---|---|---|
| A · terminal | 0.000 [−0.080, +0.090] | −0.057 [−0.074, +0.064] | −0.016 [−0.033, +0.017] | [−0.0029, +0.0021] |
| A · conditional | −0.019 [−0.066, +0.035] | +0.032 [−0.021, +0.063] | −0.005 [−0.024, +0.017] | [−0.0018, +0.0017] |
| CTRL · terminal | −0.029 [−0.030, +0.033] | −0.050 [−0.042, +0.013] | +0.013 [−0.003, +0.022] | [−0.0003, +0.0031] |
| CTRL · conditional | +0.028 [−0.009, +0.066] | **+0.046 [+0.016, +0.070]** | **+0.019 [+0.001, +0.039]** | [−0.0043, +0.0021] |

Seven of the eight ratio cells straddle zero, and the one that does not favours SCRATCH by 0.046 —
the opposite of a basin, and small. **A head initialised from the online weights and re-optimised
to convergence on the terminal target goes straight back to the online head's non-separating
solution; a head initialised from those same weights and re-optimised on the conditional target
separates.** The online weights are not a trap. The registered order hazard is therefore
ELIMINATED, and its elimination is itself the argument against treatment class 1: if the failure
were an optimisation-history artefact, scratch and fine-tune would part company. They do not.

## 8. Capacity — a real second-order term, and only under the conditional target

| contrast (MLP − linear, same target) | Δ ratio @t1–3 | Δ Brier @all |
|---|---|---|
| A · terminal | −0.150 [−0.118, +0.117] · *not detected* | [−0.0070, +0.0045] |
| A · conditional | **+0.323 [+0.136, +0.337]** | [−0.0120, −0.0036] |
| CTRL · terminal | −0.000 [−0.016, +0.034] · *not detected* | [−0.0018, +0.0030] |
| CTRL · conditional | **+0.136 [+0.095, +0.201]** | [−0.0069, +0.0005] |

Under the terminal target the hidden layer buys nothing on the opponent meter — there is nothing
extra to express. Under the conditional target it buys a lot, on both substrates, with the delta's
CI clear of zero. **Capacity is not the binding constraint, but it becomes one as soon as the
target stops being noise**, which is exactly the ordering "target first, head width second".

---

## 9. Does the reading survive the offline/online sample-size gap?

It has to be asked, because it is the obvious objection to outcome 2. The two exposures:

| | distinct episodes (0/1 labels) | states | passes over each label |
|---|---|---|---|
| ONLINE, arm A to 74M | ~2.4 M (74 M steps ÷ ~31 states/battle) | 74 M | `--n-epochs 10` |
| ONLINE, CTRL to 10M | ~0.32 M | 10 M | 10 |
| OFFLINE, this measurement | **951** (A) / **873** (CTRL) | 29,495 / 25,564 | median 104 (`mlp_term`) / 298 (`mlp_cond`) |

The online head saw roughly **2,500×** (A) and **370×** (CTRL) more distinct outcome labels.

**The half of the reading that the gap threatens is (a)'s failure, and the gap makes (a)'s failure
CONSERVATIVE in the wrong direction — so the conclusion has to be stated carefully.** Offline, with
951 battles, "the per-episode variance is binding" is exactly what you would expect on sample size
alone. It cannot be *only* sample size, because the conditional target — built from those same 951
battles and nothing else — recovers the separation, i.e. the between-cell signal IS estimable at
n = 951 (a cell mean has SE ≈ 0.045 against a between-cell spread of 0.11; the LOO log loss beats
the marginal's by 0.097 nats). But it does mean the offline (a) row is not, by itself, proof about
the online head.

**What closes it is that the online head, with 2,500× the labels, reads 0.149 / 0.066 anyway.** A
pure sample-size account predicts the online head should have averaged the per-episode noise down
long ago and conditioned; it has not. So the online mechanism is not "too few episodes", and the
variance-share arithmetic of §6 says what it is instead: the between-opponent component is ~10 % of
the objective at *any* sample size, and the head reduces the other 90 % more cheaply. **That
statement is scale-free** — it is a property of the loss, not of n — which is why it transfers
across the gap and why the recommended lever (re-price the target so the between-cell component is
a larger share) is the one that does.

**What does NOT survive the gap**, and is stated as a limit rather than a result: the *size* of
(b)'s recovery. 0.323 / 0.259 against ceilings of 0.702 / 0.898 is a statement about a head fitted
on ~1k battles. An online arm with a lower-variance target has 2,500× the data and could go
further — or could fail for the separate reason that its target is produced by the same
non-stationary loop. Nothing here predicts which.

---

## 10. Counter-hypotheses

**(a) Overfitting to ~1k battles.** Every headline number is out-of-fold; the in-fold columns are
kept and read as a whole second pass (`read_A_train.json`, tables §"IN-FOLD columns"). The gap is
real and is exactly where you would expect it: `mlp_term`'s in-fold Brier is **0.0842** against a
held-out **0.1148**, `mlp_cond`'s 0.1260 against 0.1285. Two things follow. First, the terminal
refit *does* have spare capacity to memorise — it burns 0.031 of Brier on the training battles —
which is why "it could not fit the opponent" is not available as an explanation of (a)'s failure.
Second, the spread ratios barely move between the two passes (A, in-fold: online 0.149,
`lin_term` 0.136 vs 0.150 held out), because the identity is computed over 56 cell MEANS and cell
means are not what a head overfits. **SURVIVES as a caveat on the Brier column only; it does not
touch the reading.**

**(b) Label leakage through the conditional target.** Every ingredient is leave-one-battle-out in
closed form: the group's weighted sum minus this battle's own contribution, over the group's weight
minus this battle's weight. So the label of battle *i* cannot contain the outcome of battle *i*.
Its correlation with its own outcome is 0.464 (A) / 0.320 (CTRL) — that number is the target's
genuine out-of-sample skill, not a leak, and it is corroborated by the LOO log loss beating the
marginal by 0.097 / 0.099 nats. Re-run with the LITERAL per-(cycle, opponent, team) LOO cell mean
wherever the cell has ≥ 3 battles (19.7 % of battles; `read_A_cell.json`), the conditional refit
reads **0.291** against the additive target's 0.323 and the ceiling 0.783 against 0.702 — same
conclusion. **ELIMINATED.**

**(c) The HT weights.** The whole pipeline — fit and readout — was re-run unweighted
(`read_A_raw.json`). Unweighted, the TERMINAL refit's ratio reads 0.217 against the online head's
0.149 and its delta reads clear of zero, i.e. **without reweighting, (a) looks like a partial
success.** With reweighting it is not. This is the same direction the mixture diagnostic recorded
(§6(a) there: the naive read overstates the effect 6×), and it is why every headline here is the
reweighted one. **SURVIVES as a hazard for anyone reading the raw tree; it does not threaten the
reading, because the reweighted read is the conservative one.**

**(d) "The conditional target is just an easier regression."** It is — and that is not exculpatory,
because the head is scored on the SAME held-out terminal outcome either way. On turns 1–3 the
conditional refit's Brier against the real 0/1 outcome is **0.1190** on A and **0.1415** on CTRL,
beating the online head by **[−0.0495, −0.0344]** and **[−0.0260, −0.0131]**. It separates opponents
*and* forecasts the actual outcome better in the window where the defect lives. Over ALL states on
CTRL it forecasts worse (0.1321 vs 0.1086, Δ **[+0.0149, +0.0325]**) — the late-game states where
the board itself says who is winning are where a shrunk conditional label costs you, and that is a
real trade-off to price in any arm that adopts this target, not a reason to reject it.

**(e) The cycle key.** Both keyings of the conditional target's opponent factor are run in full.
On arm A, whose four cycles sit within 0.027 of win rate, the choice barely matters for the reading
((cycle, opponent): refit 0.323, ceiling 0.702; opponent-only: refit 0.517, ceiling 1.028 — the
*fraction of the ceiling reached* is 46 % vs 50 %). On CTRL, whose cycles span 0.547→0.896, the
opponent-only key collapses the ceiling to 0.420 and makes (b) unreadable. **The cell key is the
correct one and the sensitivity says so; (a) is untouched by the choice, since the terminal target
has no factors.**

**(f) Is `online` the right comparator?** Two are reported. `online` is the ONE frozen snapshot that
also produced the features — the internally consistent row, and the only one for which "same
weights, same features, only the target changed" is true. `online_rec` is the recorded per-cycle
`V`, which reproduces the mixture diagnostic's published 0.334 / 0.308 / 0.573 and its +0.0171 slope
exactly. They differ (A: 0.149 vs 0.334 at t1–3, Δ **[−0.186, −0.048]**), and on arm A the
difference flatters the refits. Against the CONSERVATIVE row, (b)'s turn-1–3 gain is
**+0.183 [+0.103, +0.332]** under the opponent-only key and straddles zero under the cell key
(−0.011 [−0.051, +0.179]) — so **on arm A the (b) recovery is detected against the frozen
comparator under both keys and against the recorded comparator under one of them.** On CTRL,
`online_rec`'s 1.079 is a cycle artefact (§4) and cannot be used as a comparator at all.

---

## 11. What this means for the treatment

**The supported class is TARGET-SIDE, and two other classes are now ruled out rather than merely
deprioritised.**

- **RULED OUT — head-side optimisation (treatment class 1).** A "value replay", a periodic head
  refit, or a head-specific learning rate all assume the target is learnable and the online loop
  failed to learn it. This measurement ran the idealised version of that arm — the same head, on
  the same features, refit from scratch to convergence with early stopping and no non-stationarity
  at all — and it reproduced the online head's failure on both substrates. Fine-tuning from the
  online weights lands in the same place, so there is no basin to escape either. **Do not build it.**
- **RULED OUT (reconfirmed) — a conditioning INPUT.** The mixture diagnostic's §12 FiLM route was
  already deprecated by the probe read on the grounds that `value_pooled` carries the opponent
  code. This adds a second, independent reason: a head handed a *perfect* conditional target still
  only reaches 0.32 / 0.26, so a cleaner INPUT cannot be the binding lever when a cleaner TARGET is
  only partly sufficient.
- **BUILD NEXT — `ai_v12_12_ladder_cflabels`, the counterfactual win-probability label arm**
  (`--cf-records --cf-winprob-coef 0.5` on the ladder control substrate). It is the only QUEUED arm
  in the supported class and it is already argv-ready. Its meters are named by this measurement,
  not by strength: the **turn-1–3 spread ratio** (0.066 on the control today) and the **turn-1
  own-team win-rate R²** (0.028 today, 0.467 with a conditional target, 0.628 in the tensor). A
  cf-label arm that does not move those two has not done the thing it exists to do, whatever its
  ELO.
- **DEMOTE — `ai_v12_13_ladder_tdaux`, the TD-bootstrap auxiliary** (`--td-aux-coef 1.0`). It is in
  the right class (a bootstrap target has lower variance than a Monte-Carlo draw) but this
  measurement gives a specific reason to run it second: at turn 1 the bootstrap's regression target
  is `V(s′)` — the head's own, currently unconditioned, estimate — so it lowers variance without
  necessarily raising the between-cell SHARE, which §6 identifies as the operative quantity. Run it
  after the cf arm, and read it on the same two meters.
- **DEMOTE — `ai_v12_14_ladder_truevalue`** (`--value-true-team`). A privileged-information
  representation probe, and both this measurement and the probe read say the representation is not
  the binding gap.
- **CHEAP SECOND LEVER, one line — widen the win head.** §8 shows the hidden layer earns its keep
  only once the target carries the conditional signal, and then earns it on both substrates. Any
  arm that changes the target should carry the wider head, not test it separately.
- **The un-queued arm this measurement actually points at, and the reason to build it:**
  **opponent-stratified (or cell-balanced) weighting of the win-prob loss.** §6 says the operative
  quantity is the between-cell SHARE of the objective, and a stratified weight raises that share
  directly, needing no new labels, no new machinery and no extra rollout cost — where cf labels buy
  the same re-pricing at the cost of producing them. It is not built; it is a small change at
  `value_terms._win_prob_loss`, and it is the highest ratio of expected effect to cost on this list.

**And one thing this measurement does NOT license.** It says a differently-paid head conditions and
forecasts better on early states. It does not say the resulting agent plays better, and on CTRL the
conditional refit forecasts *worse* over all states (§10(d)). The identity test against an MC
continuation remains the arbiter for any arm that moves the ratio, exactly as the mixture
diagnostic already required.

---

## 12. Hazards hit (a hazard is a finding)

1. 🚨 **A NOISE-CORRECTED, CLAMPED SPREAD RATIO CAN SIT BELOW ITS OWN BOOTSTRAP INTERVAL.**
   `spread_corrected` subtracts each side's sampling variance and clamps a negative result to zero —
   a biased, non-monotone operator — so a point estimate of 0.000 routinely carries a CI of
   [0.211, 0.533]. The mixture diagnostic has the same shape and reported it. It is NOT the
   cluster-bootstrap defect that measurement recorded as its hazard 2 (a resampled selection paired
   with unresampled cell codes); that one is excluded here by construction, because the cell code
   travels with the SLOT. An unclamped `ratio_raw` is computed in the same pass as the monotone
   companion. **Read the interval, and never read a 0.000 as "no spread".**
2. 🚨 **A CONDITIONAL TARGET WITH NO CYCLE TERM IS UNREADABLE ON A RAPIDLY IMPROVING RUN.** The
   identity is scored over (cycle, opponent) cells, and CTRL's own strength runs 0.547 → 0.896
   across its five cycles. An opponent-keyed target cannot reproduce those cell means and its
   CEILING reads 0.420 — which the reading rule correctly refused to interpret, and which would
   otherwise have been reported as "the conditional target does not carry the separation on CTRL".
   Keying by the cell raises the ceiling to 0.898. **The first revision of this measurement got
   exactly that wrong number.**
3. 🚨 **THE RECORDED `V` AND THE FROZEN FORWARD'S `V` ARE DIFFERENT OBJECTS, AND ON A YOUNG RUN THEY
   ARE VERY DIFFERENT.** `V_rec` is each cycle's own model; `V_fwd` is one snapshot over every
   cycle, and it is what the refits' features come from. On arm A the two spread ratios are 0.334
   and 0.149; **on CTRL they are 1.079 and 0.066**, because `V_rec` tracks the trainee's own
   improvement across cycles and that variance lands between cells. Quoting `V_rec` as the
   comparator would have said the ladder control's head conditions perfectly. Both are reported;
   the frozen one is the comparator, the recorded one is the conservative check.
4. ⚠️ **THE CEILING IS A PROPERTY OF THE TARGET'S SHRINKAGE, NOT OF THE HEAD.** A well-estimated
   conditional label must shrink toward the base rate, and shrinkage attenuates exactly the
   between-cell spread the identity measures. Arm A's ceiling is 0.702 under the cell key and 1.028
   under the opponent key for that reason alone. Any "(b) only reached X" claim must be read as a
   fraction of the ceiling, and the ceiling row must be run — without it a failed (b) is
   uninterpretable. This is why `cond_oracle` exists and why the reading rule checks it FIRST.
5. ⚠️ **WITHOUT HT REWEIGHTING, (a) LOOKS LIKE A PARTIAL SUCCESS.** Unweighted, the terminal refit's
   turn-1–3 ratio reads 0.217 against the online head's 0.149 with a delta clear of zero; weighted,
   it reads 0.000 and the delta straddles. Same direction as the mixture diagnostic's §6(a).
6. ⚠️ **THE OFFLINE HEAD HAS SPARE CAPACITY TO MEMORISE AND STILL DOES NOT LEARN THE OPPONENT.**
   `mlp_term` burns 0.031 of Brier between its in-fold (0.0842) and held-out (0.1148) scores. That
   closes "it lacked the capacity" as an explanation of (a)'s failure, and it is the reason the
   in-fold pass is a whole re-run rather than a footnote.
7. ⚠️ Feature extraction is NOT repeated here — `pooled.npy` and `meta.npy` are the probe read's,
   so probe-read hazard 5 is inherited verbatim: off-cycle rows are a probe of the FINAL model on
   earlier states, not a replay.

---

## 13. Ledger paragraph (for the orchestrator to append — this file does NOT edit the ledger)

> **2026-09-09 · HEAD REFIT on the win-prob critic — reading (2), PARTIAL, on BOTH substrates: the
> head is not failing to learn, it is being paid to predict something else.** Following the probe
> read, the win head alone was refit on a FROZEN `value_pooled` (the probe read's own extraction;
> `dump_head.py` REFUSES unless `sigmoid(win_head(value_pooled))` reproduces the frozen forward's V
> — 2.3e-07 on A, 2.8e-07 on CTRL). Seven conditions on one tensor, every prediction OUT OF FOLD
> under battle-grouped 5-fold CV with a grouped early-stopping split, every fit and score HT-
> reweighted by the manifest capture rates, every interval a battle-clustered bootstrap resampling
> battles WITHIN their (cycle, opponent) cell with the outcome side's 100-game binomial redrawn, and
> every claim on the DELTA's own CI. Substrates `ai_v12_02_winprob_critic` @74M (29,495 states / 951
> battles / 216 teams / 14 opponents) and the ladder control `ai_v12_11_ladder_ctrl10M` @10M (25,564
> / 873 / 180 / 12). **(a) THE TERMINAL 0/1 TARGET — the one the online head actually trains on —
> REPRODUCES THE ONLINE FAILURE EXACTLY**, from scratch, to convergence, with no non-stationarity:
> turn-1–3 between-opponent spread ratio 0.149 → 0.000 on A (Δ [−0.055, +0.234]) and 0.066 → 0.068
> on CTRL (Δ [−0.044, +0.042]); turn-1 opponent-class decode of the prediction 0.532 → 0.546 on A
> (Δ [−0.049, +0.081]) against `value_pooled`'s 0.846, and FALLING on CTRL. **(b) THE SAME HEAD ON
> THE SAME FEATURES, TRAINED ON THE PER-(CYCLE, OPPONENT) × OWN-TEAM LEAVE-ONE-BATTLE-OUT WIN RATE,
> RECOVERS A DETECTED PART OF IT**: ratio 0.149 → 0.323 on A (Δ **[+0.063, +0.296]**) and 0.066 →
> 0.259 on CTRL (Δ **[+0.051, +0.192]**); turn-1 class AUC 0.532 → 0.646 on A (Δ [+0.059, +0.170]);
> own-team LOO win-rate R² 0.010 → 0.569 on A and 0.028 → 0.467 on CTRL, against `value_pooled`'s
> 0.671 / 0.628 — and it forecasts the REAL outcome better where the defect lives (turns 1–3 Brier
> 0.1608 → 0.1190 on A, Δ [−0.0495, −0.0344]; 0.1611 → 0.1415 on CTRL, Δ [−0.0260, −0.0131]). 🚨 **The
> online head's turn-1–3 Brier SKILL against the base rate is −0.129 on arm A: in the window where
> the mixture defect lives it forecasts worse than a constant, reliability 0.0231 against resolution
> 0.0054, and every refit including the linear floor beats it.** **THE ORDER HAZARD IS ELIMINATED:**
> a head initialised FROM the online weights lands in the same place as one trained from scratch
> under BOTH targets on both substrates (7 of 8 ratio deltas straddle zero; the one that does not
> favours scratch by 0.046) — the online head is not in a basin, so treatment class 1 (a value
> replay / periodic head refit / head-specific lr) is RULED OUT, not merely deprioritised.
> **CAPACITY is a real but second-order term**: the MLP beats the linear floor only under the
> conditional target (A Δ +0.323 [+0.136, +0.337]; CTRL +0.136 [+0.095, +0.201]) and not under the
> terminal one. **The mechanism is arithmetic**: only 10.2 % (A) / 14.4 % (CTRL) of the terminal
> label's variance lies between (cycle, opponent) cells, so a head minimising BCE buys resolution
> from the board and its own team instead — the terminal refit lifts held-out Brier resolution
> 0.0323 → 0.0513 while moving the opponent meter not at all; the conditional target raises the
> between-cell share to 24.0 % / 58.8 % and the head then conditions. That statement is scale-free,
> which is what carries it across the 2,500× gap between the offline 951 battles and the online
> head's ~2.4M episodes: a pure sample-size account predicts the online head should have averaged
> the noise down and it did not. The recovery is PARTIAL — 46 % (A) and 29 % (CTRL) of the
> conditional target's OWN ceiling, i.e. 31 % and 23 % of the gap from the online head to it (0.702 [0.652, 0.894] and 0.898 [0.822, 0.984]), and the ceiling
> is set by the shrinkage a well-estimated conditional label requires, not by the head.
> Counter-hypotheses: overfitting SURVIVES on the Brier column only (in-fold 0.0842 vs held-out
> 0.1148 for the terminal refit — which also closes "it lacked capacity" as an account of (a));
> leakage ELIMINATED (closed-form LOO; the literal per-(cycle, opponent, team) cell target at ≥3
> battles reads 0.291 against 0.323); the HT weights SURVIVE as a hazard in the usual direction —
> UNWEIGHTED, (a) reads 0.217 with a delta clear of zero, i.e. the naive read manufactures a
> success. **Consequence:** the mixture diagnostic's §12 FiLM conditioning arm is doubly ruled out,
> the head-side-optimisation class is ruled out, and the next ladder arm is the QUEUED cf-label arm
> `ai_v12_12_ladder_cflabels` (`--cf-records --cf-winprob-coef 0.5`), read on the turn-1–3 spread
> ratio and the turn-1 own-team R² rather than on strength; `ai_v12_13_ladder_tdaux` is DEMOTED (a
> bootstrap lowers variance but at turn 1 regresses on the head's own unconditioned V, so it need
> not raise the between-cell share) and `ai_v12_14_ladder_truevalue` is DEMOTED as a representation
> probe. The highest-value UNBUILT lever this identifies is opponent-stratified weighting of the
> win-prob loss — it raises the between-cell share directly, with no new labels and no rollout cost.
> Three tooling hazards recorded, each of which produced a wrong number before it was caught: a
> clamped noise-corrected ratio sits BELOW its own bootstrap interval (0.000 with CI [0.211, 0.533]);
> a conditional target with no CYCLE term is unreadable on a rapidly improving run (CTRL's ceiling
> read 0.420 instead of 0.898); and the RECORDED per-cycle V is a different object from the frozen
> forward's — on CTRL they read 1.079 and 0.066, so quoting the recorded one would have said the
> ladder control's head conditions perfectly.
> Measurement: `designs/research_state/measurements/winprob_head_refit_2026-09-09/`.

---

## 14. Proposed amendment to UNDERSTANDING.md §4.2b (text only — not applied here)

> Append to §4.2b, after the probe-read paragraph:
>
> **The head's failure is a TARGET failure, and the target's defect is the SHARE of the objective
> the opponent holds.** [MEASURED, `winprob_head_refit_2026-09-09`] Refitting the win head alone on
> a frozen `value_pooled`, out of fold under battle-grouped CV and HT-reweighted, against the
> TERMINAL 0/1 outcome the online head actually trains on reproduces the online failure exactly on
> both `ai_v12_02_winprob_critic` @74M and the ladder control `ai_v12_11_ladder_ctrl10M` @10M: the
> turn-1–3 between-opponent spread ratio moves 0.149 → 0.000 (Δ [−0.055, +0.234]) and 0.066 → 0.068
> (Δ [−0.044, +0.042]), and the prediction's turn-1 opponent-class decode moves 0.532 → 0.546
> (Δ [−0.049, +0.081]). Swapping ONLY the target for the per-(cycle, opponent) × own-team
> leave-one-battle-out win rate recovers a DETECTED part: 0.149 → 0.323 (Δ [+0.063, +0.296]) and
> 0.066 → 0.259 (Δ [+0.051, +0.192]), class AUC 0.532 → 0.646 (Δ [+0.059, +0.170]), own-team
> win-rate R² 0.010 → 0.569 and 0.028 → 0.467 against `value_pooled`'s 0.671 / 0.628 — while
> forecasting the real outcome BETTER on turns 1–3 (Brier 0.1608 → 0.1190, Δ [−0.0495, −0.0344]).
> The mechanism is that only 10.2 % / 14.4 % of the terminal label's variance lies between (cycle,
> opponent) cells, so a head minimising a proper scoring rule buys its resolution from the board and
> its own team instead; the conditional target raises that share to 24.0 % / 58.8 %. **A head
> initialised from the online weights lands where a scratch head lands, under both targets and on
> both substrates**, so the online head is NOT in a basin and the head-side-optimisation treatment
> class (value replay, periodic head refit, head-specific lr) is RULED OUT. Capacity is a real
> second-order term: the MLP beats a linear head only under the conditional target. **The online
> win-prob critic's turn-1–3 Brier skill against the base rate is −0.129 on arm A** — in the window
> where the mixture defect lives it forecasts worse than a constant. **Consequence:** the treatment
> is target-side (counterfactual / MC labels, search leaves, and above all opponent-stratified
> weighting of the value loss, which raises the between-cell share with no new labels). **Open:**
> whether an online arm with a lower-variance target moves the same two meters, and whether a head
> that conditions plays better — the identity test against an MC continuation remains the arbiter.
> **Caveat:** the recovery is PARTIAL (46 % / 29 % of the conditional target's own ceiling; 31 % / 23 % of the gap to it), and
> the offline head saw ~2,500× fewer distinct episodes than the online one, so the size of the
> recovery does not transfer even though the variance-share mechanism, being scale-free, does.

---

## 15. Files

`dump_head.py` (lift the online head; REFUSE unless it reproduces `V`) · `refit.py` (the seven
conditions, the conditional target, the out-of-fold predictions) · `readout.py` (the mixture
identity, the bias slope, the probe decodes, the Brier decomposition, every delta's CI) ·
`summarize.py` (the reading rule, applied by code) · `run.sh` (end to end, including the four
counter-hypothesis re-runs) · `refit_stats.json` + `tables.md` (the committed summary) ·
`variance_shares.json` (the §6 arithmetic). The prediction columns and the 128-dim feature tensor
are NOT committed; `run.sh` regenerates them from the probe read's extraction in about an hour.
