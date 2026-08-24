# R1 FIRST READ — `ai_v9_29_rev1_0823`, the §2 paired head difference

**This is the honest first read, NOT the verdict.** The label corpus is **6,600 rows** (the
duty-cycle incident capped the night; the warm producer path landed ~4 h before run end), so per
the runbook's own §5 amendment a *flat* primary at this dosage would have been "a coverage/dosage
reading, not the kill". **The primary is not flat** — every contrast is significant and the
replication reproduces it — so the dosage caveat does not excuse the result; it bounds how far the
result generalises. Read every number below with the dosage line in §4 attached.

Instrument: `agents/training/cf_audit.py` as built, run per the runbook's §2 command. Data:
`r1_first_read_ai_v9_29.json`. Box: 16 cores, **idle** (load 0.10 at start, no training run live).

---

## 0. What was audited, and which node

| | |
|---|---|
| run | `models/ai_v9_29_rev1_0823` (READ-ONLY; every audit product written to the worktree, nothing added under `models/`) |
| trace step | `eval_traces/step_24000000` — the audit's default (`--step` omitted ⇒ latest) |
| **checkpoint** | **`eval_traces/step_24000000/snapshot.zip`, tier `exact`, step 24,000,000** |
| convention | `cf_audit`'s own: no `--checkpoint`, so `ProbeSession`'s exact→nearest→recent ladder resolves from the first labelled battle. It landed on the **exact eval snapshot**, *not* the endpoint checkpoint `checkpoint_24988992_steps.zip`. Stated because the tool does not record it — see §7. |
| arch | `git d78aa810`, `arch_signature gen3_critic_route_wave_v1`, `config_version 101` — matches this worktree's HEAD, so all three heads load and score |
| held-out | by construction: `cf_producer` reads `<run>/cf_records/`, `cf_audit` reads `<run>/eval_traces/`. Structurally disjoint. |
| label trust | **anchors 20/20 (100%)** primary · **19/20 (95%)** replication — both ≥ the 0.9 refusal threshold |

**Two launch deviations from the signed §0 block, both recorded rather than argued away.**
`--win-prob-mode` ran as **`shaping`**, not the block's `read_only`. §0b contemplates exactly this
("A's own `shaping` mode (if used) shapes it for all three equally"), so head A remains a valid
control for a *paired* read — but head A is not the inert control the block describes, and the
trunk it shapes is the trunk B and C read detached. `--cf-winprob-coef` was correctly left at
**0.0**. Everything else in the §0 block is present in `model_config.json` (`cf_twin_heads: true`,
`cf_twin_coef: 0.1`, `cf_head_only: true`, `cf_label_likelihood: binomial`,
`cf_label_lag_steps: 150000`, `cf_shadow_critic: true`, `cf_evidential: true`).

---

## 1. §2 PRIMARY — the paired head difference on held-out states

🔒 **These are ERROR scores. A NEGATIVE difference means the first-named head is BETTER.**

### Pre-registered run — `--rollouts 8 --states 400`, seed 20260822, **388 labels / 170 battles**

| contrast | isolates | **Brier Δ** | CI95 (battle-clustered) | abs-err Δ | CI95 | `mean_abs_pred_diff` |
|---|---|---|---|---|---|---|
| **`B_minus_A`** | coverage / prioritization | **+0.0650** | **[+0.0424, +0.0891]** | +0.0738 | [+0.0458, +0.1012] | 0.190 |
| **`C_minus_B`** | precision (tight-MC vs one draw) | **−0.0358** | **[−0.0559, −0.0155]** | −0.0354 | [−0.0609, −0.0102] | 0.174 |
| **`C_minus_A`** | the total effect (the original R1 claim) | **+0.0292** | **[+0.0190, +0.0405]** | +0.0384 | [+0.0244, +0.0526] | 0.082 |

Per head: **A** Brier 0.0772 · abs-err 0.1958 · mean pred 0.7368 — **B** 0.1422 · 0.2697 · 0.8747 —
**C** 0.1064 · 0.2342 · 0.7718. Labels' mean MC on these states: **0.6601**.

### Replication — different seed (20260824) and 2× audit states, **788 labels / 204 battles**

| contrast | Brier Δ | CI95 | abs-err Δ | CI95 | `mean_abs_pred_diff` |
|---|---|---|---|---|---|
| `B_minus_A` | +0.0609 | [+0.0447, +0.0785] | +0.0736 | [+0.0543, +0.0945] | 0.180 |
| `C_minus_B` | −0.0364 | [−0.0538, −0.0199] | −0.0424 | [−0.0630, −0.0223] | 0.167 |
| `C_minus_A` | +0.0245 | [+0.0174, +0.0317] | +0.0312 | [+0.0222, +0.0398] | 0.075 |

*(Not pre-registered — a stability check on the METER, not on the arm. It doubles the audit's
sample, which buys precision on the comparison; it does **not** double the label dosage, so it
cannot address §5's "flat at 2× sample" clause. Every sign and magnitude reproduces.)*

### Reading it

**`mean_abs_pred_diff` first, as §2 demands: 0.17–0.19.** The heads separated, decisively, and the
live counterpart agrees (`cf/twin_b_vs_c_abs` ranged 0.078–0.897, mean of the last 50 points
**0.248**). So this is a divergence that HAPPENED — the §5 dosage escape does not apply, and the
contrasts mean what they say.

1. **`B_minus_A` is significantly POSITIVE — the coverage / prioritization arm made the head
   WORSE**, by +0.065 Brier on held-out states, in both draws. Labelling the sampler's
   high-critic-surprise states *at all*, with the same single-outcome loss form head A already
   eats, cost calibration rather than buying it.
2. **`C_minus_B` is significantly NEGATIVE — the tight-MC target is better than the single draw**,
   by −0.036 Brier. This is the amendment's whole point and it fires. But see §2: a large part of
   it is a re-centring, not a resolution gain.
3. **`C_minus_A` is significantly POSITIVE — the TOTAL effect is a net HARM**, +0.029 Brier.
   Precision recovers roughly **55%** of what prioritization cost, and no more.

This is precisely the shape §5 pre-registered as "a decomposition is a result even when the total
is null" — *"the prioritization is COSTING what the precision buys, which is an actionable finding
about the sampler and not about the labels"* — except that here the total is not null but
negative, because the recovery is partial.

---

## 2. Why B is worse, and why the C−B win is smaller than it looks

Two descriptive reads, both flagged as *descriptive* (neither is pre-registered):

**(a) Most of B's penalty is a mean shift, not a resolution failure.** Against the audit's label
mean (0.6601):

| head | mean pred | bias | bias² | Brier | bias² as a share of Brier |
|---|---|---|---|---|---|
| A | 0.7368 | +0.0767 | 0.0059 | 0.0772 | **7.6%** |
| B | 0.8747 | +0.2146 | 0.0461 | 0.1422 | **32.4%** |
| C | 0.7718 | +0.1117 | 0.0125 | 0.1064 | **11.7%** |

**(b) The two label streams are not the same estimand on the same states.** Over all 6,600 rows:

| head | its target | mean | |
|---|---|---|---|
| **B** | `outcome_label` — the RECORDED training episode's realized outcome | **0.7327** | (4824 wins / 1752 losses / 24 draws) |
| **C** | tight-MC, R=8, `label_regime: self_current_stochastic_both_sides` | **0.6017** | sd 0.2917 |

**Offset +0.1310 in the mean on identical states, with Pearson r = 0.240.** For reference the
run's own on-policy `win_prob/label_mean` is 0.7354 — i.e. B's target is the training ecology's
realized win rate, and C's target is what the *current* snapshot playing *both sides* scores those
same states at. The runbook's header ecology note predicted this direction qualitatively ("a
bot-opponent episode gets a stronger, self-like opponent, so that label is biased LOW"); **this is
the first measurement of its size.**

Three consequences, and they are the most load-bearing findings of this read:

- **`B_minus_A` is a clean read and it convicts the SAMPLER.** B's label is by design the same
  estimand as A's (`cf_producer.py`'s own comment: *"the SAME quantity the on-policy BCE eats, on
  the states the sampler selected: that identity is what makes B−A a read of COVERAGE alone"*). So
  the +0.065 is attributable to the state population, not the label form: up-weighting a
  high-critic-surprise slice whose realized win rate is 0.733 pushed the head optimistic against an
  eval frame whose true MC mean is 0.660.
- **`C_minus_B` is NOT budget-matched on estimand, contrary to §2's claim that it is
  "budget-matched by construction".** It differs in precision *and* in a −0.131 mean shift that
  happens to move C toward the audit's population. That is a real caveat on the pre-registration,
  not a quibble.
- **The resolution block confirms the deflation.** `twin_resolution` (UNWEIGHTED — shape only,
  never comparable to `population_weighted_sd_true_excess`) gives sample-weighted `sd_true_excess`
  of **A 0.235 · B 0.281 · C 0.286** on the primary and **A 0.234 · B 0.281 · C 0.272** on the
  replication. **C is no sharper than B on either draw, and both are blurrier than the control.**
  So C's Brier win over B is a re-centring — exactly the failure mode G0's amendment warned about
  ("a re-centred head would score a success on the wrong meter").

---

## 3. §6 HEALTH CERTIFICATION — from the run's own logged scalars

Five launcher segments, 249 logged points (129 for the fold-conditional scalars). **VERDICT: the
plumbing was healthy; the DOSAGE was not.**

| scalar | healthy reading | observed | verdict |
|---|---|---|---|
| **`cf/twin_b_coverage`** *(read first of all)* | ~1.0 | **1.000** — min = max = 1.0 in every segment | ✅ **PASS** — head B trained on real `outcome_label`s; `C_minus_B` is not silently `C_minus_A` |
| `cf/outcome_label_coverage` | ~1.0 | 1.0 on every poll with a non-empty buffer | ✅ PASS |
| `cf/mc_return_coverage` | ~1.0 | 1.0, same | ✅ PASS |
| `cf/shadow_coverage` | ~1.0 | **1.000** — min = max | ✅ PASS |
| **`train/cf_twin_grad_share`** | **0.0** | **exactly 0.0**, all 249 points | ✅ PASS — the unconditional detach held; the "identical trunk" claim stands |
| **`train/cf_shadow_grad_share`** | **0.0** | **exactly 0.0** | ✅ PASS — the shadow stayed passive |
| `train/cf_evidential_grad_share` | 0.0 | exactly 0.0 | ✅ PASS |
| **`cf/labels_future_total`** | **0** | **0** throughout | ✅ PASS — no crash-restart rollback |
| `cf/labels_replaced_total` | small | **0** | ✅ PASS — the sampler never re-labelled covered ground |
| `cf/labels_skipped_total` | 0 | **0** | ✅ PASS — no GIGO |
| `cf/labels_field_skipped_total` | 0 | **0** | ✅ PASS |
| `cf/labels_mc_return_rejected_total` | **0** | **0** | ✅ PASS — one reward composition throughout |
| `cf/label_age_steps_p50` | 0 < age < lag | max **145,730** vs the 150,000 bound; **never negative** | ⚠️ PASS, riding the bound |
| `cf/labels_ingested_total` | rising | rising within each segment: 6 · 285 · 924 · 837 · 594 | ✅ PASS (per-process counter; it resets at each restart) |
| `cf/rows_sampled` | > 0 every `train()` | **0 on 120/249 (48.2%) of logged points** | ⚠️ **DOSAGE** — the buffer was empty, so ~half of all logged training points ran no cf fold at all |
| **`cf/buffer_fill`** | at/near capacity | mean **11.2**, median **3.0**, max **120** against a capacity of **2048** | 🔴 **DOSAGE — 0.5% of capacity at its peak** |
| `cf/labels_expired_total` | ~flat after warmup | 6,537 cumulative | ⚠️ see below — **not** producer lag |
| **`cf/twin_b_vs_c_abs`** | rising off 0 | 0.078 – 0.897, mean of last 50 = **0.248** | ✅ **PASS — the label streams DID separate the heads** |

**Guards (§4).** Belief-bank canaries are live and improving, not starved: `belief/species_acc`
0.147→0.638, `item_acc` 0.910→0.968, `hptype_acc` 0.667→0.867, `natureev_nature_acc` 0.602→0.779;
all 17 declared edge families report non-zero `edge/*_grad_norm` at run end. **The ELO
non-inferiority guard is NOT EVALUABLE within-run and must not be claimed either way** — §0b/§7
make the acting path bit-identical across A/B/C by construction, so there is no ELO contrast here
at all. For the record only, the dense ladder's endpoint node reads **2098.4** at step 24,000,000.

### The two dosage numbers, stated honestly

- **6,600 rows on disk, but ~2,646 row-ingestions into training buffers.** The buffer's counters
  are per-process and each of the five launcher segments re-scans the whole on-disk archive,
  expiring every row outside the 150k lag bound. That is the dominant term in
  `labels_expired_total` = 6,537 — **expected behaviour of a per-process buffer, not the producer
  lagging.** The honest consequence is that the *effective* training dosage is smaller than the
  6,600 headline, not larger.
- **Production rate:** 150–330 rows/h through 21:00, then **528–678 rows/h** once the warm path
  landed (~22:00 onward), 6,600 rows over 13.3 h. The producer ran concurrently with training the
  whole way; only 3 rows were written after the last TB flush.

---

## 4. §2a SECONDARY — `sd_true_excess`, and the floor

| | primary (388) | replication (788) |
|---|---|---|
| `population_weighted_gap` | +0.0330 | +0.0376 |
| **`population_weighted_sd_true_excess`** | **0.1993** | **0.1965** |

Per-decile (primary): dec 7 **0.249** · dec 8 **0.233** · dec 9 **0.146**. The §2 amendment's
measured floor is **sd 0.151 [0.119, 0.186] in deciles 7–9** — but that was measured on **gen-17
@24M**, a *different run*, so it is a reference point and not this run's floor. Taken at face value
the excess over it in deciles 7–8 is ~0.08–0.10 and **decile 9 sits at or below it**. Per the
amendment, no target may be expressed as a fraction of the raw meter.

**The floor does not enter §1 at all**: the paired difference scores every head on the same state,
so the floor is the same additive constant in both terms and cancels *exactly*.

**Conviction class** (the +0.23 G0 class's descendant): `wp≥0.75 & LOST` gap **+0.174 [+0.110,
+0.246]** (n=129 / 53 battles), confidence-matched `WON` control **−0.032 [−0.055, −0.012]**;
replication +0.196 [+0.133, +0.273] and −0.029 [−0.047, −0.012]. **Do not quote the tool's
`loss_minus_win_ci`** — see §7.

---

## 5. The evidential read (`--cf-evidential` was ON)

**NULL, and the sign flips between draws.**

| draw | `width_vs_blur_spearman` | CI95 (battle-clustered) | strata | usable draws |
|---|---|---|---|---|
| primary (388) | **+0.286** | [−0.524, +0.886] | 8 | 1000 |
| replication (788) | **−0.273** | [−0.576, +0.133] | 10 | 1000 |

**On "per stratum": the tool computes ONE rank correlation ACROSS the decile strata, with the CI
bootstrapped over battles — there is no per-stratum correlation in `cf_audit`, and I did not invent
one.** §3's phrasing reads as if there were; what the instrument actually produces is the table
below, which *is* the correlation's input.

| decile (primary) | n | mean predicted | `evid_width_mean` | `sd_true_excess` |
|---|---|---|---|---|
| 0 | 12 | 0.037 | 0.174 | 0.000 |
| 3 | 12 | 0.355 | 0.176 | 0.264 |
| 4 | 15 | 0.450 | 0.155 | 0.278 |
| 5 | 22 | 0.550 | 0.153 | 0.279 |
| 6 | 26 | 0.659 | 0.141 | 0.202 |
| 7 | 97 | 0.748 | 0.128 | 0.249 |
| 8 | 78 | 0.849 | 0.113 | 0.233 |
| 9 | 106 | 0.966 | 0.089 | 0.146 |

The shape is the finding: **the confessed width falls monotonically with the prediction decile
(0.174 → 0.089) while the measured blur is roughly flat at 0.20–0.28 across deciles 3–8.** The head
is confessing *confidence*, not *blur* — which is the null this meter exists to detect, and the
falling `cf/evid_nll` (0.688 → 0.574) beside it is the standing learns≠helps kill, not a result.

## 5b. The shadow critic (orthogonal; §0c — may not license a route change on its own)

`shadow_vs_live_v` = **+5.52 [+3.35, +7.53]** shaped-return units on the audit's states
(replication +5.32 [+3.67, +6.77]); `shadow_vs_live_v_abs` 9.82. **But the live trainer-side
`cf/shadow_shadow_vs_live_v` reads −4.05 (mean of last 50) on the producer's own states.** The two
populations give **opposite signs**, and the standing caveat applies with full force: this compares
two *fitted* heads with no external anchor, so it says nothing about which of them moved.
`cf/shadow_live_v_vs_label` — the arm that does have ground truth — reads **+4.08**, i.e. the live
critic sits above the MC return on the factory's states. `cf/shadow_coverage` was 1.0 throughout,
so the head was genuinely trained and its divergence is not initialisation noise.

---

## 6. What this read does and does not license

**Does not** (§7, unchanged): say anything about a better TRUNK (the twins are detached from it —
`train/cf_twin_grad_share` exactly 0.0), or anything about the POLICY (the acting path is
bit-identical across A/B/C by construction), or a critic route change on the shadow's evidence
(that owes the C4 offline gate).

**Does**, at this dosage: report that on this run's held-out eval frame the tight-MC label stream
delivered to a detached win-prob head produced a **net worse-calibrated head than the control**,
that the loss is attributable to the **sampler's state population** rather than to label precision,
and that the precision term — the amendment's whole reason for existing — **fires in the predicted
direction but is mostly a re-centring**, because it buys no resolution (§2).

---

## 7. Instrument defects found during this read

1. 🔴 **`conviction_class.loss_minus_win_ci` does not bracket its own point estimate** and is
   computed on a different estimand. `cf_audit.py:590-595` takes the point as
   `mean(diff) − mean(dctl)` but bootstraps the mean of the *pooled, sign-flipped* vector
   `diff + [−dctl]` — which equals the difference of means only when the two groups are the same
   size. Here 129 vs 102, giving `+0.205` with a CI of `[+0.070, +0.158]` (replication: `+0.225`
   with `[+0.084, +0.169]`). **The primary meter is unaffected** — `paired_head_read` bootstraps a
   single genuinely paired per-row vector, which is correct. Do not quote the conviction class's
   difference CI until this is fixed.
2. ⚠️ **The audit records no resolved checkpoint.** `render_markdown(..., ckpt=args.checkpoint)`
   passes the *flag*, which is `None` on the documented invocation, and `bias_map.json` carries
   `policy_step` (the trace step) but not the path the twin/evidential/shadow forwards actually
   used. §2's "record the audited `step_N` anyway" is therefore satisfiable only by hand — as done
   in §0 here.
3. ℹ️ **§2's claim that `C_minus_B` is "budget-matched by construction" is too strong** — matched on
   states and folds, yes; not matched on estimand (§2).

---

## 8. Conclusion — three sentences, dosage-honest

At a **6,600-row corpus of which only ~2,646 row-ingestions reached a training buffer** (median
resident rows 3, against a capacity of 2048, with 48% of logged training points running no cf fold
at all), the within-run paired primary is **not** flat: `B_minus_A` **+0.065 Brier [+0.042,
+0.089]**, `C_minus_B` **−0.036 [−0.056, −0.016]**, `C_minus_A` **+0.029 [+0.019, +0.041]** on 388
held-out states / 170 battles, every sign reproduced at a second seed and 2× audit sample, with
`mean_abs_pred_diff` 0.17–0.19 confirming the label streams genuinely separated the heads — so this
is a divergence that happened and bought a net harm, not the dosage-null the pre-registered caveat
anticipated. **It is still dosage-limited in exactly one direction**: it cannot tell whether the
precision term's −0.036 would keep growing and eventually overcome the sampler's +0.065 at a larger
corpus, because both effects were measured at ~½% buffer occupancy and a 48% fold-skip rate, and it
cannot rule out that the sampler's cost is an artifact of so thin a slice being seen so often. **At
the producer's post-warm-path rate of ~600–900 rows/h a full-length run yields ~15–20k labels
(≈2.5–3× this corpus, and with a buffer that stays populated the effective dosage rises by more
than the row count does)**; the reading that would count as SIGNAL is `C_minus_A` crossing to
significantly negative *together with* `twin_resolution` showing head C's `sd_true_excess` falling
below head A's — a resolution gain, not a re-centring — while the reading that would count as a
dosage-null is `C_minus_B` staying at ≈−0.036 with `B_minus_A` still ≈+0.06, which would say the
sampler's population cost is structural and the next move is the **sampler**
(`cf_producer_priority_v1`) and the **label estimand** (§2's +0.131 self-play offset), not more
labels.
