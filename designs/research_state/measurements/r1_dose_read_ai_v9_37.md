# R1 DOSE READ — `ai_v9_37_tick1_dosext_0825`, the §2 paired head difference at the extended dose

**The pre-registered dosage replication of [`r1_first_read_ai_v9_29.md`](r1_first_read_ai_v9_29.md).**
Run `ai_v9_37_tick1_dosext_0825` extended tick-1 by +5M steps with the label producer running at
the warm rate; the tick-1 lineage's cumulative on-disk corpus is **11,370 rows** (tick-1 7,344 +
extension 4,026) = **76% of the 15k figure** the first read named, which the pre-registration
recorded as *"dosage-null branch WEAKENED, not retired"*.

**🚨 READ §2 BEFORE ANY NUMBER BELOW.** This endpoint's trunk is a TICK-1 DESCENDANT, i.e. a
distill-collapsed trunk. The within-run paired design is unaffected — it holds trunk, states and
seeds identical across A/B/C by construction — but **every cross-read comparison with the first
read confounds dose with trunk state**, and the confound is large enough to change what a
magnitude difference means.

Instrument: `agents/training/cf_audit.py` at `d87393d` (which carries the §7 CI fix `1c8d784`),
run per the runbook's §2 command. Box: 16 cores, **idle** (load 0.1 at start, 2.2–2.5 with the two
audits themselves running; no training run live, no eval competing). Data:
`r1_dose_read_ai_v9_37.json` (all three bias maps, the health scalars, the corpus and dosage
accounting, and the discrimination table §3 computes outside the tool).

---

## 0. What was audited, and which node

| | |
|---|---|
| run | `models/ai_v9_37_tick1_dosext_0825` (**READ-ONLY**; every audit product written to the worktree via `--out`, nothing added under `models/`) |
| trace step | `eval_traces/step_40000032` — the audit's default (`--step` omitted ⇒ latest) |
| **checkpoint** | **`eval_traces/step_40000032/snapshot.zip`, tier `exact`, step 40,000,032** |
| convention | `cf_audit`'s own: no `--checkpoint`, so `ProbeSession`'s exact→nearest→recent ladder resolves from the first labelled battle. Verified directly (`ModelChoice(tier='exact', detail='exact eval snapshot · step 40,000,032')`) because the tool still does not record it — the first read's §7 defect 2 is **unfixed** |
| arch | `git 530db54`, `arch_signature gen3_critic_route_wave_v1`, `config_version 101` — matches this worktree's HEAD, so all three heads load and score |
| lineage | `--model models/ai_v9_34_tick1_0824/final_model.zip`, which itself forked `models/ai_v9_29_rev1_0823/final_model.zip`; `--distill-coef 1.0`, teachers tock-1a/1b, `--stable-opponents` on |
| held-out | by construction: `cf_producer` reads `<run>/cf_records/`, `cf_audit` reads `<run>/eval_traces/`. Structurally disjoint |
| label trust | **anchors 19/20 (95%)** primary · **19/20 (95%)** replication — both ≥ the 0.9 refusal threshold; 0 label errors in either draw |

**The same two launch deviations as the first read, and they are the same two.** `--win-prob-mode`
ran as **`shaping`**, not §0's `read_only` (contemplated by §0b: A's shaping mode shapes the trunk
for all three equally, so A remains a valid *paired* control but is not the inert control §0
describes). `--cf-winprob-coef` was correctly **0.0**. Everything else in the §0 block is present in
`model_config.json`: `cf_twin_heads: true`, `cf_twin_coef: 0.1`, `cf_head_only: true`,
`cf_label_likelihood: binomial`, `cf_label_lag_steps: 150000`, `cf_shadow_critic: true`,
`cf_shadow_coef: 0.1`, `cf_evidential: true`, `cf_records_keep: 4096`.

**INSTRUMENT RE-VERIFIED BEFORE THE READ.** The first read's exact protocol was re-run on
`ai_v9_29_rev1_0823 --step 24000000 --states 400 --seed 20260822` on today's code, and it
reproduced the banked numbers **to four decimal places** (`B−A +0.0650`, `C−B −0.0358`,
`C−A +0.0292`, `population_weighted_gap +0.0330`, `sd_true_excess 0.1993`,
`width_vs_blur_spearman +0.286`, 388 labels / 170 battles). So the instrument is deterministic
across the `1c8d784` change and **every difference reported below is a property of the model, not
of the tool.**

---

## 1. §2 PRIMARY — the paired head difference on held-out states

🔒 **These are ERROR scores. A NEGATIVE difference means the first-named head is BETTER.**

### Pre-registered run — `--rollouts 8 --states 400`, seed 20260822, **400 labels / 155 battles**

| contrast | isolates | **Brier Δ** | CI95 (battle-clustered) | abs-err Δ | CI95 | `mean_abs_pred_diff` |
|---|---|---|---|---|---|---|
| **`B_minus_A`** | coverage / prioritization | **+0.1597** | **[+0.1049, +0.2036]** | +0.1997 | [+0.1368, +0.2526] | 0.409 |
| **`C_minus_B`** | precision (tight-MC vs one draw) | **−0.1128** | **[−0.1502, −0.0642]** | −0.1231 | [−0.1660, −0.0664] | 0.292 |
| **`C_minus_A`** | the total effect (the original R1 claim) | **+0.0469** | **[+0.0298, +0.0625]** | +0.0765 | [+0.0533, +0.0981] | 0.184 |

Per head: **A** Brier 0.0960 · abs-err 0.2307 · mean pred 0.6960 — **B** 0.2557 · 0.4304 · 0.5162 —
**C** 0.1429 · 0.3073 · 0.5969. Labels' mean MC on these states: **0.6144**.

### Replication — different seed (20260824) and 2× audit states, **800 labels / 175 battles**

| contrast | Brier Δ | CI95 | abs-err Δ | CI95 | `mean_abs_pred_diff` |
|---|---|---|---|---|---|
| `B_minus_A` | +0.1563 | [+0.1197, +0.1911] | +0.1910 | [+0.1468, +0.2335] | 0.410 |
| `C_minus_B` | −0.1209 | [−0.1525, −0.0900] | −0.1295 | [−0.1655, −0.0939] | 0.284 |
| `C_minus_A` | +0.0354 | [+0.0220, +0.0489] | +0.0614 | [+0.0424, +0.0799] | 0.186 |

Per head: **A** 0.0939 · 0.2341 · 0.6952 — **B** 0.2503 · 0.4251 · 0.5078 — **C** 0.1294 · 0.2955 ·
0.6015. Mean MC: **0.6184**.

*(The replication is a stability check on the METER, not on the arm — it doubles the audit sample,
not the label dosage. Every sign and magnitude reproduces.)*

### Reading it

**`mean_abs_pred_diff` first, as §2 demands: 0.18–0.41** — more than double the first read's
0.08–0.19, and the live counterpart agrees (`cf/twin_b_vs_c_abs` ranged 0.056–0.553, mean of the
last 50 points **0.246**, last value 0.305). The heads separated decisively. **§5's dosage escape
does not apply**; the contrasts mean what they say.

1. **`B_minus_A` is significantly POSITIVE — the coverage / prioritization arm made the head
   WORSE** by +0.16 Brier, in both draws, with CIs that clear zero by a wide margin.
2. **`C_minus_B` is significantly NEGATIVE — the tight-MC target beats the single draw** by −0.11
   to −0.12 Brier. It recovers **~71%** (primary) / **~77%** (replication) of what prioritization
   cost — better than the first read's ~55%, and still not all of it.
3. **`C_minus_A` is significantly POSITIVE — the TOTAL effect remains a net HARM**, +0.047 /
   +0.035 Brier.

**The registered SIGNAL condition requires BOTH halves and neither is met**: `C_minus_A` is
significantly *positive*, not negative, and head C's `sd_true_excess` is **above** head A's, not
below (§3).

---

## 2. 🚨 THE CONFOUND THIS READ HAS AND THE FIRST READ DID NOT

**The dosext endpoint sits on a distill-collapsed trunk.** It is a tick-1 descendant, trained with
`--distill-coef 1.0` against tock-1a/1b, which is precisely the regime the factorial arms convicted
(ledger 2026-08-25, *"THE LOSS CHANNEL"*): `pi_features` participation ratio **12.50 at both
nonzero distill doses vs 21.87 at zero**, an all-or-nothing switch, with tick-1's own end-of-run
capacity row reading `pi_features 19.6→12.4` and `team_tokens 16.5→12.0`. *(For provenance: the
IN-RUN telemetry, a different population from that offline battery, reads `rank/policy_pr` tail-5
**27.04 rev-1 → 26.37 tick-1 → 25.62 dosext** and `rank/trunk_pr` 17.89 → 17.69 → 17.00 — a mild
monotone decline. The two instruments disagree in magnitude; both agree in sign. Neither is quoted
here as the collapse's size.)*

Three consequences, and they must travel with every number above:

- **The WITHIN-RUN paired design is intact and is the reason this read is worth taking at all.**
  Heads A, B and C read the *same* trunk, on the *same* rows, at the *same* seeds. Whatever the
  trunk's condition, it is an identical additive property of every term in the difference — exactly
  as the hidden-information floor is. So `B−A`, `C−B` and `C−A` remain clean reads of the **label
  effect on a fixed representation**.
- **CROSS-READ MAGNITUDE COMPARISONS ARE CONFOUNDED, and this is the one thing not to do casually.**
  The first read's `B−A +0.065` and this read's `+0.160` differ by 2.5×, and **dose and trunk moved
  together**: rev-1's corpus was 6,600 rows on an uncollapsed trunk; this is 11,370 on a collapsed
  one. Nothing in this design separates them. **Do not report "the harm scales with dose"** — the
  honest statement is *"at 1.7× the corpus on a damaged trunk, the harm is larger and the sign is
  unchanged"*.
- **One within-run fact does survive the confound and is worth more than the magnitudes.** Head A
  reads the **same damaged trunk** and retains most of its discrimination against ground truth
  (Pearson r with the tight-MC label **+0.562 / +0.566**, vs **+0.647** for A on rev-1's healthy
  trunk). Heads B and C, on that identical trunk, do not: **B is ANTI-correlated with ground truth
  (r = −0.213 / −0.194; Spearman −0.174 / −0.147)** and C is barely correlated (**+0.180 / +0.274**)
  — against B **+0.463** and C **+0.493** on rev-1. A trunk defect shared by all three heads cannot
  produce a spread that large *between* them. **The label streams are doing this, not the trunk** —
  the trunk confound bounds how far the finding generalises to a healthy representation, it does not
  explain the finding away.

**A fourth, smaller accounting caveat.** The twin heads were **not re-initialised** at either fork,
so head B's weights carry rev-1's exposure too: the lineage corpus is really 6,600 + 7,344 + 4,026 =
**17,970 rows on disk** (≈6,250 effective row-ingestions), of which the pre-registration's "11,370 /
76%" counts only the tick-1 lineage's own two runs. This is another reason the endpoint is not a
clean dose ladder: it is a cumulative-lineage quantity spread across three trunks.

---

## 3. Why the contrasts are so much larger — the failure changed KIND

Descriptive (not pre-registered), and the most load-bearing part of this read after §2.

**In the first read most of B's penalty was a MEAN SHIFT. Here almost none of it is.** Against each
draw's own label mean:

| head | mean pred | bias | bias² | Brier | **bias² as a share of Brier** | sd of error | **Pearson r vs MC** |
|---|---|---|---|---|---|---|---|
| A | 0.6960 | +0.0816 | 0.0067 | 0.0960 | **6.9%** | 0.299 | **+0.562** |
| B | 0.5162 | −0.0982 | 0.0096 | 0.2557 | **3.8%** | 0.496 | **−0.213** |
| C | 0.5969 | −0.0175 | 0.0003 | 0.1429 | **0.2%** | 0.378 | **+0.180** |

*(primary draw; the replication gives 6.3% / 4.9% / 0.2% and r = +0.566 / −0.194 / +0.274.)*

Three readings:

- **The re-centring already happened, and it bought nothing.** Head C is now essentially
  **unbiased** — mean prediction 0.597 against a label mean of 0.614, bias² **0.2%** of its Brier —
  and it is still nearly 50% worse than the control by Brier. The first read warned that C's win
  over B was "a re-centring, not a resolution gain"; at this dose the centring is complete and the
  Brier gap to A has *widened*. **This is the cleanest possible demonstration that the meter
  amendment was right**: a head that scores a perfect mean-gap and a terrible proper score.
- **B's ORDERING has inverted.** A head whose rank correlation with ground truth is negative is not
  a blurry head, it is an anti-informative one on this frame. Its bias flipped sign too (rev-1
  **+0.215** optimistic, here **−0.098** pessimistic) — so "the single-outcome stream teaches
  inflated optimism", the first read's mechanism, does **not** carry forward unchanged; what
  carries forward is that the stream teaches something that does not transfer to the eval frame,
  and its direction is not stable.
- **The run's own live meters agree and are not the same measurement.** On the *on-policy* stream
  the ordering is A ≤ C < B throughout: `win_prob/brier` last-10 **0.1590**, `cf/twin_c_onpolicy_brier`
  **0.1698**, `cf/twin_b_onpolicy_brier` **0.1909** (accuracies 0.761 / 0.738 / 0.710). Same sign as
  the audit, much smaller spread — the twins are worst exactly where the audit looks (the
  quota-selected eval frame), and merely worse on-policy.

---

## 4. §6 HEALTH CERTIFICATION — from the extension's own logged scalars

Two launcher segments, 49 logged points (33 for the fold-conditional scalars).
**VERDICT: the plumbing was healthy; the DOSAGE improved but is still the binding limit.**

| scalar | healthy reading | observed | verdict |
|---|---|---|---|
| **`cf/twin_b_coverage`** *(read first of all)* | ~1.0 | **1.000** — min = max = 1.0 on all 33 fold points | ✅ **PASS** — head B trained on real `outcome_label`s; `C_minus_B` is not silently `C_minus_A` |
| `cf/outcome_label_coverage` | ~1.0 | **1.0 on every one of the 33 non-empty-buffer polls** (the 16 zeros are exactly the empty-buffer polls) | ✅ PASS |
| `cf/mc_return_coverage` | ~1.0 | 1.0, same 33/33 | ✅ PASS |
| `cf/shadow_coverage` | ~1.0 | **1.000** — min = max | ✅ PASS |
| **`train/cf_twin_grad_share`** | **0.0** | **exactly 0.0**, all 49 points | ✅ PASS — the unconditional detach held; the "identical trunk" claim stands |
| **`train/cf_shadow_grad_share`** | **0.0** | **exactly 0.0** | ✅ PASS — the shadow stayed passive |
| `train/cf_evidential_grad_share` | 0.0 | exactly 0.0 | ✅ PASS |
| **`cf/labels_future_total`** | **0** | **0** throughout | ✅ PASS — no crash-restart rollback |
| `cf/labels_replaced_total` | small | max **1** | ✅ PASS |
| `cf/labels_skipped_total` / `_field_skipped_total` | 0 | **0** / **0** | ✅ PASS — no GIGO |
| `cf/labels_mc_return_rejected_total` | **0** | **0** | ✅ PASS — one reward composition throughout |
| `cf/label_age_steps_p50` | 0 < age < lag | median 99,120; max **145,728** vs the 150,000 bound; **never negative** | ⚠️ PASS, riding the bound (as in the first read) |
| `cf/labels_ingested_total` | rising | rising within each segment: **1,185 · 183** (per-process counters) | ✅ PASS |
| `cf/rows_sampled` | > 0 every `train()` | **0 on 16/49 (32.7%)** of logged points — and **never** zero while the buffer was non-empty | ⚠️ **DOSAGE** — improved from the first read's 48.2%, still a third of training points running no cf fold |
| **`cf/buffer_fill`** | at/near capacity | mean **28.4**, median **15.0**, max **93** against a capacity of **2048** | 🔴 **DOSAGE — 4.5% of capacity at its peak** (first read: 0.5%) |
| `cf/labels_expired_total` | ~flat after warmup | 3,624 cumulative | ⚠️ per-process rescan expiry, **not** producer lag — see below |
| **`cf/twin_b_vs_c_abs`** | rising off 0 | 0.056 – 0.553, mean of last 50 = **0.246** | ✅ **PASS — the label streams DID separate the heads** |

**Guards (§4).** Belief-bank canaries are live and improving, not starved: `belief/species_acc`
0.572→0.691, `item_acc` 0.949→0.965, `hptype_acc` 0.840→0.886, `natureev_nature_acc` 0.753→0.821;
all **17/17** declared edge families report non-zero `edge/*_grad_norm` at run end. **The ELO
non-inferiority guard is NOT EVALUABLE within-run and must not be claimed either way** — §0b/§7 make
the acting path bit-identical across A/B/C by construction, so there is no ELO contrast here. For
the record only, the dense ladder carries **two** nodes (1947.0 ± 15.6 at 36,000,032; **2020.0 ±
18.3** at 40,000,032) — far too few for the matched-count convention, so it is not a generation
verdict and must not be quoted as one.

### The dosage numbers, stated honestly

- **4,026 rows on disk this run, but ~1,368 row-ingestions** into training buffers (per-process
  counters; each launcher segment re-scans the on-disk archive and expires everything outside the
  150k lag bound). Cumulatively across the tick-1 lineage: **11,370 rows on disk ⇒ ~3,606 effective
  row-ingestions** (tick-1 ~2,238 + extension ~1,368), against the first read's ~2,646. **So the
  corpus grew 1.72× while the effective exposure grew only ~1.36×** — the headline "76% of 15k"
  overstates what the heads actually ate, in the same direction the first read flagged.
- **Production rate: 1,122 rows/h** over 3.59 h (4,026 rows, steps 35,244,768 → 40,066,752) —
  above the first read's post-warm-path 528–678 rows/h, i.e. the producer really did run at the
  warm rate the whole way.

### The estimand mismatch REPLICATES on both corpora

| corpus | tight-MC label mean | `outcome_label` mean | **offset** | Pearson r |
|---|---|---|---|---|
| rev-1 (first read, 6,600) | 0.6017 | 0.7327 | **+0.1310** | 0.240 |
| tick-1 (7,344) | 0.5493 | 0.6503 (W4761/L2553/D30) | **+0.1010** | 0.273 |
| extension (4,026) | 0.5655 | 0.6744 (W2706/L1302/D18) | **+0.1089** | 0.289 |
| **tick-1 lineage combined (11,370)** | 0.5550 | 0.6588 | **+0.1038** | — |

The self-play-ecology offset the runbook's header note predicted qualitatively is now measured
three times on three corpora at **+0.10 to +0.13**, with a per-state correlation of only ~0.24–0.29.
**This is the single most reproducible quantity in the whole arm**, and it is the v2 factory's
first work item.

---

## 5. §2a SECONDARY — `sd_true_excess`, and the floor

| | primary (400) | replication (800) | *(rev-1 first read, for reference only)* |
|---|---|---|---|
| `population_weighted_gap` | +0.0284 | +0.0195 | +0.0330 |
| **`population_weighted_sd_true_excess`** | **0.2242** | **0.2151** | 0.1993 |

Per-decile (primary, head A's bins): dec 6 **0.294** · dec 7 **0.305** · dec 8 **0.301** · dec 9
**0.149**. The §2 amendment's measured floor is **sd 0.151 [0.119, 0.186] in deciles 7–9** — measured
on **gen-17 @24M**, a *different run*, so it is a reference point and not this run's floor. Taken at
face value the excess in deciles 7–8 is ~0.15 and **decile 9 sits at the floor**. Per the amendment,
no target may be expressed as a fraction of the raw meter.

**`twin_resolution`** — UNWEIGHTED, shape only, never comparable with
`population_weighted_sd_true_excess` (the block says so in its own `weighting` field):

| head | primary | replication | *(rev-1)* |
|---|---|---|---|
| A | **0.2669** | **0.2545** | 0.2352 |
| B | 0.3085 | 0.3205 | 0.2814 |
| C | **0.3262** | **0.3188** | 0.2857 |

**C is BLURRIER than B on the primary and level with it on the replication, and both are blurrier
than the control on both draws — the same ordering the first read found, at a wider gap.** This is
the second half of the registered SIGNAL condition and it fails outright: the reading that would
count as signal is C's `sd_true_excess` falling *below* A's.

**The floor does not enter §1 at all**: the paired difference scores every head on the same state,
so the floor is the same additive constant in both terms and cancels *exactly*.

**Conviction class** (the +0.23 G0 class's descendant) — now quotable, since the first read's §7
defect 1 is fixed at `1c8d784` and the interval brackets its own point in both draws:

| draw | `wp≥0.75 & LOST` gap | CI95 | matched `WON` control | CI95 | **loss − win** | CI95 |
|---|---|---|---|---|---|---|
| primary (n=116 / 40 battles) | **+0.2276** | [+0.0992, +0.3688] | −0.0700 | [−0.0946, −0.0424] | **+0.2976** | [+0.1709, +0.4424] |
| replication (n=233 / 52 battles) | **+0.2163** | [+0.1347, +0.3027] | −0.0709 | [−0.0923, −0.0493] | **+0.2872** | [+0.2041, +0.3731] |

The 0.827 class's descendant is alive and unchanged in size from the first read (+0.174 / +0.196).

---

## 6. The evidential read (`--cf-evidential` was ON) — NULL again, sign flips again

| draw | `width_vs_blur_spearman` | CI95 (battle-clustered) | strata | usable draws |
|---|---|---|---|---|
| primary (400) | **−0.048** | [−0.405, +0.714] | 8 | 1000 |
| replication (800) | **+0.770** | [+0.251, +0.834] | 10 | 1000 |

**Same verdict as the first read, and for the same reason: the sign is not stable across seeds.** The
first read gave +0.286 / −0.273; this one gives −0.048 / +0.770. One draw's interval excluding zero
while the other spans it, with the two disagreeing in sign, is a null with a small-strata sampling
artifact on top — the correlation is computed over 8–10 decile strata, which is a tiny n for a rank
statistic. Mean confessed width is **0.2304** in both draws (precision 3.57), and `cf/evid_nll` fell
0.68 → 0.58 across the run: the standing learns≠helps kill, not a result.

## 6b. The shadow critic (orthogonal; §0c — may not license a route change on its own)

`shadow_vs_live_v` = **−5.06 [−6.15, −3.86]** shaped-return units on the audit's states
(replication **−5.08 [−6.13, −3.94]**), `shadow_vs_live_v_abs` 10.73 / 10.44. The live trainer-side
`cf/shadow_shadow_vs_live_v` reads **−4.24** (mean of last 50, last value −5.78) on the producer's own
states. **Unlike the first read, the two populations now AGREE in sign** — there, the audit said
+5.52 and the trainer said −4.05. The direction that has ground truth behind it is unchanged:
`cf/shadow_live_v_vs_label` = **+4.20**, i.e. the live critic sits above the MC return on the
factory's states. `cf/shadow_coverage` was 1.0 throughout, so the head was genuinely trained and its
divergence is not initialisation noise. The standing caveat still applies with full force: the
shadow-vs-live comparison is two *fitted* heads with no external anchor, and says nothing about
which of them moved.

---

## 7. Which pre-registered branch fired

The runbook (§5) plus the ledger entry `2638e07` registered two readings:

| branch | condition | fired? |
|---|---|---|
| **SIGNAL** | `C−A` crossing **significantly negative** **AND** C's `sd_true_excess` below A's | ❌ **NO, on both halves.** `C−A` = **+0.047 [+0.030, +0.063]** / **+0.035 [+0.022, +0.049]** — significantly POSITIVE. C's blur **0.326 / 0.319** vs A's **0.267 / 0.255** — ABOVE, not below |
| **DOSAGE-NULL** | `C−B` stuck ≈ −0.036 with `B−A` ≈ +0.06 ⇒ *"the v2 factory (estimand + sampler) is the move, not more labels"* | ⚠️ **Its DECISION fires; its literal numbers do not.** Neither contrast stayed put: `C−B` is −0.113 / −0.121 and `B−A` is +0.160 / +0.156. But the direction is unchanged, the net is still harm, and at 1.7× the corpus the arm is further from signal, not closer |

**The operative reading is the dosage-null branch's decision, reached by a different route than the
one registered.** The registered dosage-null anticipated a *stuck* contrast — the case where more
labels might still have moved it. What happened instead is that the contrasts moved substantially
and **moved against the arm**: more of the current label stream did not convert into a
better-calibrated head, it coincided with a worse one. That is a stronger licence for the same
decision (**v2 factory: estimand + sampler, not more labels**) — with one honest limit, below.

**What the 76%-of-15k shortfall still buys the dosage-null branch.** The pre-registration recorded
this dose as *"WEAKENED, not retired"*, and that stands: the 15k threshold was never reached, and
this read cannot exclude a non-monotone dose curve that turns over somewhere past 11,370 rows. What
it *can* say is that nothing in the interval 6,600 → 11,370 rows trends toward the SIGNAL condition
on either of its two halves. **And the amplification itself must not be read as dose-response** —
see §2; dose and trunk moved together and this design does not separate them.

---

## 8. What this read does and does not license

**Does not** (§7, unchanged): say anything about a better TRUNK (the twins are detached from it —
`train/cf_twin_grad_share` exactly 0.0), or anything about the POLICY (the acting path is
bit-identical across A/B/C by construction), or a critic route change on the shadow's evidence (that
owes the C4 offline gate). **Additionally does not**, and this is new to this read: attribute any of
the magnitude change since the first read to dose rather than to trunk state (§2), or treat the
ladder's two-node endpoint as a strength result.

**Does**: report that on a distill-collapsed tick-1 descendant, at a cumulative 11,370-row lineage
corpus (≈3,606 effective row-ingestions), the tight-MC label stream delivered to a detached win-prob
head produced a **significantly worse-calibrated head than the control on held-out states, in both
draws**, with the loss again attributable to the **sampler's state population** and the precision
term again recovering only part of it; that head C is now **essentially unbiased and still much
worse** by proper score, which is the re-centring failure mode carried to its endpoint; and that head
B's ordering against ground truth has **inverted** while head A on the identical trunk has not.

---

## 9. Instrument defects — carried forward

1. ✅ **FIXED**: `conviction_class.loss_minus_win_ci` now brackets its own point estimate (`1c8d784`,
   `cluster_bootstrap_diff_ci` returns point and interval from one call). Verified in both draws
   (+0.2976 in [+0.1709, +0.4424]; +0.2872 in [+0.2041, +0.3731]). The class's difference is
   quotable again.
2. ⚠️ **STILL OPEN — the audit records no resolved checkpoint.**
   `render_markdown(..., ckpt=args.checkpoint)` passes the *flag*, which is `None` on the documented
   invocation, and `bias_map.json` carries `policy_step` but not the path the twin/evidential/shadow
   forwards actually used. §0 above satisfies it by hand again, this time by asking `ProbeSession`
   directly. This is the second read in a row to pay that cost.
3. ℹ️ **`twin_paired` carries no per-head DISCRIMINATION statistic**, which is what made §3's
   headline (B anti-correlated with ground truth) invisible to the instrument — it had to be
   computed outside the tool, from the emitted label file plus a re-forward. A per-head
   `pearson_vs_mc` / `spearman_vs_mc` beside `by_head`'s mean_pred/brier/abs_err would have surfaced
   it automatically, and would have surfaced it in the first read too (rev-1: A +0.647, B +0.463,
   C +0.493 — a real 0.15-point deficit that was in the data and went unread).

---

## 10. Conclusion — three sentences, keyed to the registered branches

At a cumulative tick-1-lineage corpus of **11,370 rows** (76% of the pre-registered 15k, ≈3,606
effective row-ingestions, buffer fill median 15 of 2048, 32.7% of training points running no cf
fold), the within-run paired primary on the `ai_v9_37_tick1_dosext_0825` endpoint is
**`B_minus_A` +0.160 Brier [+0.105, +0.204]**, **`C_minus_B` −0.113 [−0.150, −0.064]**,
**`C_minus_A` +0.047 [+0.030, +0.063]** on 400 held-out states / 155 battles, every sign and
magnitude reproduced at a second seed and 2× audit sample (+0.156 / −0.121 / +0.035), with
`mean_abs_pred_diff` 0.18–0.41 confirming the streams genuinely separated the heads. **The
registered SIGNAL branch did NOT fire on either of its two halves** — `C_minus_A` is significantly
*positive*, and head C's `sd_true_excess` (0.326 / 0.319) sits *above* head A's (0.267 / 0.255),
with C now essentially unbiased (bias² 0.2% of its Brier) and still ~50% worse by proper score,
which is the re-centring failure mode taken to its endpoint; **the DOSAGE-NULL branch's decision
fires instead, by a stronger route than the one registered** — the contrasts did not stay stuck at
≈−0.036 / ≈+0.06, they grew and grew *against* the arm, so the pre-registered move (the **v2 factory:
the estimand fix for the measured +0.10 to +0.13 self-play offset, and the sampler, not more labels
of the current kind**) is the licensed one, while the 76% shortfall keeps that branch formally
*weakened rather than retired*. **Every cross-read magnitude comparison in this document is
confounded**: the dosext trunk is a distill-collapsed tick-1 descendant, so dose and trunk state
moved together and neither the growth of the contrasts nor their absolute size may be attributed to
dose — what survives the confound is the within-run fact that on one identical trunk and identical
rows, head A retains r = +0.56 against ground truth while head B has **inverted to r = −0.21** and
head C has fallen to +0.18, a spread between heads that no shared trunk defect can produce.
