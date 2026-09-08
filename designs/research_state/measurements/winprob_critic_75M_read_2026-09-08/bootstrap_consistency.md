# TEST 3 — BOOTSTRAP CONSISTENCY on `ai_v12_02_winprob_critic` (75M read, 2026-09-08)

**Question (from the D2 verdict, ledger 2026-09-08, commit `6c4ae35d`).** The head is optimistic
against its own Monte-Carlo continuation by **+0.0965 [+0.0671, +0.1268]**, and the offset is
LATE-GAME (late ≥25 turns +0.169 vs early ≤10 +0.002). Is that late optimism

- **the γ = 1 terminal-label TARGET** — every late state inherits the final label undiscounted, so a
  head reading the material lead over-commits; or
- **a MOVING-POLICY target** — V tracks what the improving policy wins, and lags it?

**Registered discriminator (44e3a7f5):** a CONSISTENT bootstrap (mean TD residual ≈ 0 at every turn
bucket) with late optimism ⇒ the TARGET; a WITHIN-GAME DRIFT (residual mean ≠ 0 late, V rising then
losing) ⇒ the MOVING POLICY. **Registered bar: none — this test is DESCRIPTIVE.**

**Direction: it reads as the TARGET.** See §7.

---

## 0. What was measured, and the one thing that decides how to read it

Model-free throughout: the recorded per-decision `values` are read through
`ProbeSession.battle_overview` — no checkpoint is loaded. Four eval cycles of
`models/ai_v12_02_winprob_critic/eval_traces`: **50000016, 60000000, 70000032, 74000016** (the last
being the run's final cycle at 75,005,952 steps).

Two invariants were **asserted, not assumed**, and hold exactly on all four cycles:

| invariant | measured |
|---|---|
| off-terminal reward is exactly 0.0 (terminal-only reward) | **0** decisions with non-zero off-terminal reward, of 28,536 |
| therefore δ_t = r_t + γV(s_{t+1}) − V(s_t) ≡ ΔV | **0** violations of \|δ − ΔV\| > 1e-9 |
| γ | **1.0** (read from the run) |
| terminal reward by class | WIN → `[1.0]`, LOSS_decisive → `[0.0]`, TIMEOUT_as_LOSS → `[0.0]` |
| critic currency | `winprob`, V ∈ [0,1], G(s) = the terminal win indicator |

🚨 **THE SELECTION CONFOUND IS THE WHOLE METHODOLOGICAL STORY, AND IT IS FIXABLE HERE.** The trace
tree is **loss-enriched by design** — `eval_manifest.selection_rule`: *"per-opponent per-cycle
OUTCOME QUOTA: the first 10 losses and the first 5 wins of each opponent are persisted as traces …
LOSS-ENRICHED BY DESIGN — the traces are a loss-forensics sample, never a random subsample of the
cycle."* Selecting battles on their **outcome** — a FUTURE event — breaks the martingale property
E[δ_t | V_t] = 0 *even for a perfectly consistent critic*, so the raw trace numbers cannot answer
this question at all.

The same manifest records `capture_rate_win` / `capture_rate_loss` **per opponent**, so every traced
battle can be reweighted by **w = 1 / capture_rate(opponent, outcome)**. That recovers the population
of battles actually played. **Validation that the weights are right: the reweighted battle count
comes to exactly 1400.0 on every cycle** = 14 opponents × 100 games, and the reweighted win rate
lands at 0.821 against the captured 0.463.

**Everything headline in this document is the population-reweighted (IPW) read.** The raw
captured-sample numbers are kept in §5 because they are what the CLI reports, and the gap between
the two is itself the finding.

Intervals are a **2,000-resample cluster bootstrap over BATTLES** — never over pooled decisions.
Decisions inside one battle share the same label y and are strongly dependent; pooling them would
shrink every interval by roughly √(decisions per battle) ≈ √29.

**Strata.** `bot` = the 9 scripted bots; `pool_sentinel` = `sentinel_0…4`. 🚨 **The pool stratum was
played under the greedy-vs-stochastic handicap** (A ran stochastic sentinels; +8.9 pp, D2 verdict) —
**the BOT stratum carries the verdict.** These traces are also **pre-draw-bucket**
(`result_vocabulary: gen3_trace_result_v1`, outcomes only `win`/`loss`), so a LOSS mixes decisive
losses with timeouts; split at `meta.turns ≥ 250`. At 74M that is **1** timeout in 242 battles, so
the split changes nothing here — but it was applied, and the timeout is excluded from every 0/1
label because no binary outcome exists for it.

---

## 1. Summary table — 74M (step 74000016), population-reweighted

| quantity | value [95% cluster-bootstrap CI] | reads as |
|---|---|---|
| **mean TD residual δ, all decisions** | **+0.0072 [+0.0058, +0.0085]** | small upward drift, DETECTED |
| — early ≤10 | +0.0147 [+0.0126, +0.0166] | DETECTED |
| — mid 11–24 | +0.0017 [−0.0003, +0.0036] | **not detected** |
| — **late ≥25** | **+0.0021 [−0.0024, +0.0062]** | **not detected — the bootstrap is CONSISTENT late** |
| **terminal \|V(s_T) − outcome\|, all** | **0.0719 [0.0564, 0.0913]** | |
| — on WINs | 0.0212 [0.0110, 0.0343] | V ends at 0.979 |
| — **on decisive LOSSES** | **0.3050 [0.2587, 0.3558]** | **V ends a lost game at 0.305 — the head never concedes** |
| calibration bias E[V − y], all | +0.0060 [−0.0319, +0.0501] | **NOT DETECTED — unbiased on the population** |
| — early ≤10 | −0.0585 [−0.0946, −0.0147] | **PESSIMISTIC**, detected |
| — late ≥25 | +0.0732 [+0.0030, +0.1611] | **OPTIMISTIC**, detected |
| overdispersion (Brier / E[V(1−V)]) | 0.9661 [0.8550, 1.1002] | **1.0 inside — confidence matches error** |
| Murphy: Brier = REL − RES + UNC | 0.1261 = 0.0019 − 0.0454 + 0.1700 | REL≈0 (calibrated), **RES = 27% of UNC** |
| base-rate cap (UNC), p̄ = 0.783 | 0.1700 | skill vs base rate **+0.2582** |

**Read in one line:** on the population the head is *calibrated in the aggregate and consistent as a
bootstrap late in the game*, but it is **pessimistic early, optimistic late, badly under-resolved,
and its terminal anchor on losses is broken by 0.305.**

---

## 2. (a) TD residual δ_t = V(s_{t+1}) − V(s_t)

### 2.1 By turn bucket and stratum — 74M, IPW

| slice | δ [CI] | n dec / battles |
|---|---|---|
| all | +0.0072 [+0.0058, +0.0085] | 7,019 / 242 |
| early ≤10 | +0.0147 [+0.0126, +0.0166] | 2,746 / 242 |
| mid 11–24 | +0.0017 [−0.0003, +0.0036] | 2,942 / 238 |
| late ≥25 | +0.0021 [−0.0024, +0.0062] | 1,331 / 95 |
| **bot** | +0.0113 [+0.0099, +0.0129] | 3,618 / 142 |
| pool_sentinel *(handicapped stratum)* | +0.0023 [−0.0001, +0.0041] | 3,401 / 100 |

### 2.2 The martingale test — E[δ | V-level], the confound-free instrument

Conditioning on V_t conditions on the PAST only, so under a consistent bootstrap this is **0 at
every level**, for every level. This is the sharpest form of the test, and unlike the pooled mean it
carries no ratio bias from unequal battle lengths.

| V-level | ALL (IPW) | BOT only (IPW) |
|---|---|---|
| [0.00, 0.20) | −0.0039 [−0.0098, +0.0038] | −0.0089 [−0.0277, +0.0081] |
| [0.20, 0.40) | +0.0046 [−0.0080, +0.0186] | −0.0021 [−0.0177, +0.0145] |
| [0.40, 0.60) | **+0.0182 [+0.0115, +0.0253]** | **+0.0300 [+0.0171, +0.0432]** |
| [0.60, 0.80) | **+0.0119 [+0.0085, +0.0153]** | **+0.0223 [+0.0181, +0.0268]** |
| [0.80, 0.95) | **+0.0065 [+0.0034, +0.0095]** | **+0.0123 [+0.0093, +0.0154]** |
| [0.95, 1.01) | −0.0021 [−0.0042, −0.0005] | +0.0002 [−0.0012, +0.0013] |

**The bootstrap is NOT perfectly consistent: from mid-confidence states (0.4–0.95) V drifts
systematically UP**, by +0.006 to +0.030 per decision, with every interval clear of zero. It is
consistent at the extremes. Note the direction — the drift is **upward**, i.e. the head reads
mid-confidence positions **too low** and corrects toward its own later, higher estimate. That is the
opposite sign to the late optimism, and it is what makes the compression account (§7) the one that
fits both.

### 2.3 Sign-run structure (74M, all)

Mean run length **2.206** against the iid reference **2.0**, max run 45, mean lag-1 autocorrelation
**−0.042**, mean positive fraction 0.568. Residual signs are very close to iid: **there is no
persistent within-battle drift signature in the sign sequence** — the drift in §2.2 is a small mean
offset, not a long one-way run.

### 2.4 By result (unweighted — a within-class read, so the quota mix does not touch it)

| result | δ [CI] | battles |
|---|---|---|
| WIN | +0.0121 [+0.0110, +0.0133] | 112 |
| LOSS_decisive | −0.0112 [−0.0128, −0.0095] | 129 |
| TIMEOUT_as_LOSS | +0.0004 (single battle) | 1 |

These are near-tautological (Σ_t δ_t telescopes to V(s_T) − V(s_0) inside a battle, so a win must
drift up and a loss down) and are reported only to make the telescoping explicit: **a pooled
residual mean is dominated by the win/loss mix of whatever sample you hold**, which is exactly why
§1 and §2.1–2.2 are reweighted.

---

## 3. (b) Calibration of V against the terminal win indicator

### 3.1 74M, population-reweighted, by bucket and stratum

Murphy: **Brier = REL − RES + UNC**, ten equal-count V-bins. REL = 0 is perfect calibration; RES is
resolution; **UNC = p̄(1 − p̄) is the base-rate cap** — the Brier a constant forecast of the slice's
own base rate would score. Overdispersion = E[(V−y)²] / E[V(1−V)]: **1.0 means the head's stated
confidence exactly matches its realized error rate**; >1 over-confident, <1 under-confident.

| slice | bias E[V−y] [CI] | Brier | REL | RES | UNC (cap) | p̄ | skill | overdispersion [CI] |
|---|---|---|---|---|---|---|---|---|
| **OVERALL** | +0.0060 [−0.0319, +0.0501] | 0.1261 | 0.0019 | 0.0454 | 0.1700 | 0.783 | +0.258 | 0.966 [0.855, 1.100] |
| early ≤10 | **−0.0585 [−0.0946, −0.0147]** | 0.1403 | 0.0061 | 0.0165 | 0.1513 | 0.814 | +0.073 | 0.845 [0.754, 0.948] |
| mid 11–24 | **+0.0430 [+0.0041, +0.0868]** | 0.1158 | 0.0047 | 0.0590 | 0.1707 | 0.782 | +0.321 | 1.095 [0.940, 1.258] |
| **late ≥25** | **+0.0732 [+0.0030, +0.1611]** | 0.1172 | 0.0123 | 0.1040 | 0.2088 | 0.703 | +0.439 | 1.111 [0.803, 1.455] |
| **bot** (carries the verdict) | −0.0280 [−0.0566, +0.0095] | 0.0861 | 0.0038 | 0.0234 | 0.1069 | 0.878 | +0.195 | 0.834 [0.702, 1.017] |
| pool_sentinel *(handicapped)* | +0.0475 [−0.0258, +0.1356] | 0.1750 | 0.0038 | 0.0505 | 0.2224 | 0.666 | +0.213 | 1.068 [0.901, 1.281] |
| **bot / early ≤10** | **−0.0960 [−0.1270, −0.0572]** | 0.1054 | 0.0137 | 0.0074 | 0.0996 | 0.888 | −0.058 | 0.713 [0.622, 0.826] |
| **bot / mid 11–24** | **+0.0321 [+0.0039, +0.0672]** | 0.0674 | 0.0094 | 0.0502 | 0.1086 | 0.876 | +0.380 | 1.086 [0.827, 1.401] |
| **bot / late ≥25** | +0.0575 [−0.0112, +0.1842] | 0.0725 | 0.0154 | 0.0835 | 0.1409 | 0.830 | +0.486 | 1.247 [0.592, 2.524] |
| pool / early ≤10 | +0.0108 [−0.0672, +0.1018] | 0.2047 | 0.0048 | 0.0166 | 0.2182 | 0.678 | +0.062 | 1.025 [0.877, 1.202] |
| pool / mid 11–24 | +0.0566 [−0.0190, +0.1420] | 0.1767 | 0.0051 | 0.0503 | 0.2234 | 0.663 | +0.209 | 1.100 [0.916, 1.316] |
| pool / late ≥25 | +0.0791 [−0.0067, +0.1938] | 0.1338 | 0.0120 | 0.1030 | 0.2258 | 0.656 | +0.407 | 1.087 [0.787, 1.471] |

**Three readings.**

1. **The bias flips sign across the game, and the flip is detected on the stratum that counts.**
   BOT early **−0.0960 [−0.1270, −0.0572]** (pessimistic) → BOT mid **+0.0321 [+0.0039, +0.0672]**
   (optimistic) → BOT late +0.0575 [−0.0112, +0.1842] (not detected, only 31 battles reach turn 25
   against bots). Pooled over strata the late bucket IS detected: +0.0732 [+0.0030, +0.1611]. **This
   replicates test 1's late-minus-early gradient in direction** (+0.167 [+0.086, +0.251] there), at
   roughly half the magnitude once the population weighting is applied.
2. **Overdispersion is ~1 everywhere.** Every interval covers 1.0 except `bot / early` (0.713
   [0.622, 0.826], **under**-confident — the head hedges early against bots it beats 88.8% of the
   time). There is no over-confidence in the distributional sense; the head is not claiming more
   certainty than it earns.
3. **Resolution is the defect, as D2 said.** RES 0.0454 against a base-rate cap UNC 0.1700 — the head
   captures **27%** of the available uncertainty. `bot / early` has RES 0.0074 against a cap of
   0.0996: over the first ten turns against a scripted bot, the head separates almost nothing.

### 3.2 The `calibration` CLI on the same traces (raw captured sample)

`python -m main.prober.query calibration … --step 74000016 --limit 10 --worst 2 --seeds 16
--concurrency 2 --bins 10`:

```
overall_calibration: n 7261, bias +0.3113, mae 0.4450, ev 0.1936, slope 0.9716,
                     v_mean 0.7162, g_mean 0.4049,
                     bias_on_wins −0.1651, bias_on_losses +0.6354,
                     captured_win_fraction 0.4628, n_draw_excluded 0
gate: policy_reducible 0.0530 · aleatoric 0.3186 · unattributed 0.6284
      unattributed_critic_overvalued 0.6284 · unattributed_lost_position 0.0
      critic_mean_reducible_upper_bound 0.6284       (overvalue_tau 0.0833, per-currency default)
```

🚨 **The CLI's own dominant caveat is the reason those numbers are not the headline**, verbatim:
*"the unconditional E[V−G] ≈ P(win)·bias_win + P(loss)·bias_loss, which is ~0 for a calibrated critic
at the true win rate. So critic_mean_reducible is a LOOSE upper bound until reweighted to the true
win rate."* **This measurement performs that reweighting.** Check the CLI's own arithmetic against
the reweighted win rate 0.8213:

> 0.8213 × (−0.1651) + 0.1787 × (+0.6354) = **−0.0220**

against the IPW bias **+0.0060 [−0.0319, +0.0501]** — the same answer within the interval (the small
gap is per-opponent weighting versus a single global rate, and the excluded timeout). **So the CLI's
`critic_mean_reducible_upper_bound = 0.6284` and its raw bias +0.3113 are the loose bound the tool
warns about, and they collapse to NOT DETECTED under the reweighting.** The CLI is not wrong; it is
reporting the captured sample and labelling it correctly.

---

## 4. (c) The terminal anchor and the 0.5 crossing

### 4.1 Terminal error |V(s_T) − outcome| — 74M

| slice | IPW \|V−y\| [CI] | signed E[V−y] | mean V(s_T) | raw (captured) \|V−y\| |
|---|---|---|---|---|
| all | 0.0719 [0.0564, 0.0913] | +0.0371 | 0.858 | 0.1799 [0.1500, 0.2132] |
| WIN | 0.0212 [0.0110, 0.0343] | −0.0212 | 0.979 | 0.0239 [0.0124, 0.0388] |
| **LOSS_decisive** | **0.3050 [0.2587, 0.3558]** | **+0.3050** | **0.305** | 0.3155 [0.2701, 0.3601] |
| bot | 0.0470 [0.0329, 0.0662] | +0.0260 | 0.923 | 0.1791 [0.1370, 0.2226] |
| pool_sentinel *(handicapped)* | 0.1168 [0.0841, 0.1552] | +0.0571 | 0.743 | 0.1811 [0.1385, 0.2264] |

**This is the largest single defect in the head and it is quota-immune** — it is a within-class
statement (conditional on the battle being a decisive loss), so the between-class capture mix cannot
move it, and indeed the IPW and raw columns agree to within 0.011. **On the last decision of a game
it is about to lose, the critic still says it wins 30.5% of the time.** On wins it commits properly
(0.979).

### 4.2 The 0.5 crossing (74M, unweighted; all are within-class reads)

| slice | n | never crossed 0.5 | last crossing turn (median / mean) | final side matches outcome | mean max V in battle | median turn of max V |
|---|---|---|---|---|---|---|
| all | 241 | 112 | 20.0 / 21.2 | 0.884 | 0.921 | 17 |
| WIN | 112 | 96 | 18.0 / 15.6 | **1.000** | 0.992 | 20 |
| **LOSS_decisive** | 129 | 16 | 21.0 / 22.1 | **0.783** | **0.860** | **12** |
| **bot / LOSS_decisive** | 69 | 12 | 17.0 / 18.4 | **0.739** | 0.875 | 10 |

**"V rising then losing" is real and quantified.** In a lost game the head's V *peaks at 0.860 around
turn 12* and then falls; **21.7% of decisive losses (26.1% against bots) end with V still above
0.5** — the head is on the wrong side of even at the moment the game ends. Meanwhile every win ends
on the right side. The asymmetry is total: the head knows how to commit to a win and cannot commit
to a loss.

---

## 5. Raw captured-sample numbers, for the record

Kept because they are what any un-reweighted tool reports on this tree, and the gap is the finding.
74M, no weighting (captured win fraction 0.463 against a true 0.821):

| slice | δ [CI] | bias E[V−y] [CI] | Brier | REL | RES | UNC | skill | overdispersion |
|---|---|---|---|---|---|---|---|---|
| all | −0.0014 [−0.0031, +0.0002] | +0.2898 [+0.2330, +0.3461] | 0.2716 | 0.0930 | 0.0654 | 0.2435 | −0.116 | 1.734 [1.550, 1.916] |
| early ≤10 | +0.0092 [+0.0072, +0.0113] | +0.2696 [+0.2097, +0.3309] | 0.2944 | 0.0763 | 0.0288 | 0.2475 | −0.190 | 1.624 [1.464, 1.797] |
| mid 11–24 | −0.0084 [−0.0111, −0.0055] | +0.3083 [+0.2519, +0.3639] | 0.2678 | 0.1124 | 0.0880 | 0.2431 | −0.102 | 1.877 [1.662, 2.112] |
| late ≥25 | −0.0078 [−0.0130, −0.0034] | +0.2887 [+0.1945, +0.3780] | 0.2297 | 0.1055 | 0.1053 | 0.2293 | −0.002 | 1.685 [1.276, 2.067] |
| bot | — | +0.2945 [+0.2230, +0.3668] | 0.2728 | 0.1020 | 0.0782 | 0.2489 | −0.096 | 1.983 [1.700, 2.278] |

**Every sign of interest is different.** Unweighted, the residual reads *negative* late (−0.0078,
interval clear of zero) and the head reads *hopelessly over-confident* (overdispersion 1.73–1.98,
skill NEGATIVE against the base-rate cap). Reweighted, the late residual is not detected, the
overdispersion is 0.97 and the skill is +0.26. **A reader who took the raw numbers would have
reported "within-game drift ⇒ MOVING POLICY" with confidence, from an artifact of the capture
quota.** That is the strongest practical result of this test.

---

## 6. The four-cycle trend (50M → 60M → 70M → 74M), all population-reweighted

| step | reweighted win rate | δ all | δ late ≥25 | bias all | bias early ≤10 | bias late ≥25 |
|---|---|---|---|---|---|---|
| 50000016 | 0.848 | +0.0049 [+0.0037, +0.0060] | −0.0037 [−0.0084, −0.0009] | +0.0420 [+0.0073, +0.0850] | −0.0065 [−0.0374, +0.0304] | +0.0834 [+0.0089, +0.1963] |
| 60000000 | 0.826 | +0.0070 [+0.0056, +0.0084] | −0.0011 [−0.0065, +0.0029] | −0.0089 [−0.0501, +0.0370] | −0.0604 [−0.0966, −0.0187] | +0.0006 [−0.0786, +0.1355] |
| 70000032 | 0.833 | +0.0053 [+0.0040, +0.0065] | −0.0008 [−0.0050, +0.0035] | +0.0285 [−0.0121, +0.0741] | −0.0433 [−0.0757, −0.0045] | +0.1398 [+0.0418, +0.2522] |
| 74000016 | 0.821 | +0.0072 [+0.0058, +0.0085] | +0.0021 [−0.0024, +0.0062] | +0.0060 [−0.0319, +0.0501] | −0.0585 [−0.0946, −0.0147] | +0.0732 [+0.0030, +0.1611] |

| step | terminal V(s_T) on decisive LOSSES | terminal \|V−y\| all | REL | RES | UNC | skill | overdispersion |
|---|---|---|---|---|---|---|---|
| 50000016 | 0.3003 [0.2456, 0.3574] | 0.0641 [0.0497, 0.0806] | 0.0042 | 0.0401 | 0.1520 | +0.239 | 1.185 [0.985, 1.435] |
| 60000000 | 0.2365 [0.1971, 0.2784] | 0.0653 [0.0533, 0.0794] | 0.0013 | 0.0342 | 0.1629 | +0.207 | 0.974 [0.841, 1.120] |
| 70000032 | 0.2298 [0.1845, 0.2776] | 0.0538 [0.0420, 0.0683] | 0.0018 | 0.0429 | 0.1676 | +0.252 | 1.138 [0.957, 1.363] |
| 74000016 | 0.3050 [0.2587, 0.3558] | 0.0719 [0.0564, 0.0913] | 0.0019 | 0.0454 | 0.1700 | +0.258 | 0.966 [0.855, 1.100] |

E[δ | V-level] at [0.60, 0.80) across the four cycles: **+0.0177, +0.0132, +0.0147, +0.0119** — every
interval clear of zero, no trend. Losses ending on the wrong side of 0.5: 25.2%, 13.0%, 14.8%, 21.7%.

**Trend verdict: FLAT.** Over 25M steps nothing moves outside noise — not the terminal anchor on
losses (0.300 → 0.237 → 0.230 → 0.305), not resolution (RES 0.040 → 0.034 → 0.043 → 0.045 against a
rising cap), not the skill score (+0.239 → +0.207 → +0.252 → +0.258), not the mid-V drift. **The
reweighted win rate is flat-to-declining over the same window (0.848 → 0.821).**

---

## 7. Direction — TARGET, and the reasoning

> **The late optimism reads as the γ = 1 TERMINAL-LABEL TARGET, not a moving-policy target.**

**Against the registered discriminator, read literally.**

- *"CONSISTENT bootstrap (mean TD residual ≈ 0 at every turn bucket) with late optimism ⇒ TARGET."*
  Late δ **+0.0021 [−0.0024, +0.0062] — NOT DETECTED**; mid **+0.0017 [−0.0003, +0.0036] — NOT
  DETECTED**; late optimism present, **+0.0732 [+0.0030, +0.1611]**. The clause holds in the buckets
  where the optimism lives. It fails only in the EARLY bucket (+0.0147, detected).
- *"WITHIN-GAME DRIFT (residual mean ≠ 0 late …) ⇒ MOVING POLICY."* Its own stated condition —
  residual ≠ 0 **late** — is **not met** on any of the four cycles under the population weighting.
  The other half of that clause, *"V rising then losing"*, **is** met (§4.2), but it is met by the
  terminal anchor rather than by a late residual.

**Three pieces of positive evidence for the target account.**

1. **The optimism enters at the anchor and propagates back through a locally consistent bootstrap.**
   V(s_T) on a decisive loss is **0.305 [0.259, 0.356]** — the supervision the head gets is the
   undiscounted final label at γ = 1, and it never learns to drive a losing terminal state to 0. With
   the late bootstrap consistent (δ_late not detected), a mis-set anchor is exactly what makes every
   late state inherit an inflated value. This is the mechanism the D2 question named, observed
   directly.
2. **The drift that does exist has the WRONG SIGN to explain the optimism.** The detected within-game
   drift is **upward from mid-confidence states** (+0.018/+0.012/+0.007; bot +0.030/+0.022/+0.012) and
   the head is **pessimistic early** (bot −0.0960 [−0.1270, −0.0572]). A moving-policy lag predicts a
   head that under-predicts and corrects upward — which is what the early/mid picture shows — but that
   account then has to explain a head that ends up *over*-predicting late. It cannot do both.
3. **The trend is flat while the policy is not moving.** A moving-target artifact must shrink as the
   policy's velocity decays. Over 50M → 74M this arm's reweighted win rate is flat-to-declining
   (0.848 → 0.821) and its ELO was flat (D2), yet the mid-V drift, the resolution and the broken
   anchor are unchanged. **A lag that does not shrink when the thing it lags stops moving is not a
   lag.**

**What actually unifies every number, and it is neither hypothesis in its pure form: the head is
COMPRESSED toward its own base rate.** V sits near 0.75–0.79 while the truth runs from 0.814 early
to 0.703 late, so the same shrinkage reads as pessimism early and optimism late; the same shrinkage
stops V(s_T) reaching 0 on a loss; and it is measured directly as RES 0.0454 against a base-rate cap
of 0.1700 — **27% of the available uncertainty**. That is the D2 verdict's G1 resolution failure seen
from a second instrument. The γ = 1 terminal label is the *reason* for the compression — a single
undiscounted 0/1 per battle is a maximally high-variance target, and a head fit to it under-fits
toward the mean — so the direction stands as **TARGET**, with the mechanism named as under-resolution
rather than as a naive "reads the material lead and over-commits".

**One thing that contradicted the registration, and one that qualifies it.** (i) `td_resid_tails` is
**not a CLI and has no `--help`** — it is an eval-time metric name (the per-opponent TD-residual CVaR
written into snapshot rows by `src/agents/training/eval_callback.py` and
`src/agents/model/snapshot.py`); the TD-residual instrument for this test had to be built, and it is
`bootstrap_consistency.py` beside this file. (ii) The registration asked for the numbers "by turn
bucket and by opponent stratum … with the Murphy decomposition, the base-rate cap per bucket, and
overdispersion" from `calibration`; that CLI reports none of those three and stratifies by neither,
so §3.1 computes them here **on the project's own primitives** (`_reliability_curve`,
`_calibration_stats` from `main.prober.session.stats`) and §3.2 reports what the CLI does return.
(iii) Most consequentially: **the registration's discriminator is not evaluable on the raw traces at
all.** The quota selects battles on their outcome, which breaks the martingale property the test
depends on; without the manifest reweighting this test would have returned the opposite answer with
a confident interval (§5).

---

## 8. Commands and paths

All read-only against `/home/goodlad/dev/gen3ai/models/…`; nothing was written under `models/`.
CPU only, `nice -n 10`, on a box carrying arm C.

```bash
export PYTHONPATH=$PYTHONPATH:src
P=/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3

# (a) + (c) + the raw-sample calibration fold
nice -n 10 $P designs/research_state/measurements/winprob_critic_75M_read_2026-09-08/\
bootstrap_consistency.py 50000016 60000000 70000032 74000016

# the population (inverse-capture-probability) reweighting — the headline read
nice -n 10 $P designs/research_state/measurements/winprob_critic_75M_read_2026-09-08/\
ipw_pass.py 50000016 60000000 70000032 74000016

# (b) the registered CLI, on the same traces
nice -n 10 $P -m main.prober.query calibration \
  /home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic \
  --step 74000016 --limit 10 --worst 2 --seeds 16 --concurrency 2 --bins 10
```

| artifact | holds |
|---|---|
| `bootstrap_consistency.json` (316 KB) | the raw captured-sample fold: δ by bucket/result/stratum/V-level, sign runs, Murphy, reliability bins, terminal error, crossings, per-opponent — all four cycles |
| `bootstrap_consistency_ipw.json` (49 KB) | the population-reweighted fold — every headline number in this document |
| `calibration_cli_74M.json` (14 KB) | the `main.prober.query calibration` output at 74M |
| `bootstrap_consistency.py`, `ipw_pass.py` | the two instruments, kept beside their output |

Total raw JSON **380 KB** — under the 5 MB budget, nothing sampled or truncated.

**Inputs:** `models/ai_v12_02_winprob_critic/eval_traces/step_{50000016,60000000,70000032,74000016}`,
`git_hash` of the 74M cycle `f971caf2cb9b56877f609a47bf38211b34efcb7b`, `arch_signature`
`gen3_critic_route_wave_v1`, `config_version` 110. Companion reads: test 1 `identity_test.md` and
test 2 `critic_gate.md` in this directory; the verdict they feed is ledger 2026-09-08 *D2 VERDICT*
(`6c4ae35d`).
