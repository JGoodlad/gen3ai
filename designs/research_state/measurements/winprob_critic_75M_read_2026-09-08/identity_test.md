# TEST 1 — THE IDENTITY TEST · `ai_v12_02_winprob_critic` @ 75M

**Is V(s) the Monte-Carlo win fraction from s, under the policy that produced s?**
Registered 2026-09-07 (`44e3a7f5`), run 2026-09-08 10:11–10:31 PT by an opus agent on the main
box, offline, CPU only, nothing written under `models/`.

On this arm the question is about the CRITIC itself, not a side head: `--critic winprob` makes
V = sigmoid(win logit) ∈ [0,1], BCE against the terminal win indicator, γ = 1, no shaping. Measured
here on all 800 labelled states: **`|value − win_prob| = 0.0` exactly**, so the recorded `win_prob`
the bias map compares against IS the critic's output. (G0's map, the comparator below, was of the
win-prob head on a SHAPED-critic run, where the two were different objects.)

---

## 1. Summary

| | value | interval | note |
|---|---|---|---|
| **Anchor reproduction** (label trust) | **142 / 142 = 100.0 %**, 0 errors | Wilson [0.974, 1.000] | tolerance 0.90 → **PASSED**, labels emitted (`label_trust_passed: true`) |
| Labels / battles / rollouts | 800 / 214 / 6 400 | — | 0 errors, 0 timeouts, 17.79 core-min |
| Frame it was drawn from | 6 083 decisions / 242 battles | — | skipped: `turn_1_unopenable` 242, `forced_switch_rounds` 936 |
| **Bias V − p̂, POPULATION-weighted** | **+0.0965** | **[+0.0671, +0.1268]** | battle-clustered bootstrap, 10 000 draws |
| Bias V − p̂, as-drawn sample | +0.1717 | [+0.1367, +0.2073] | the sampler's own number — do not quote as the critic's |
| `cf_audit`'s own population-weighted gap | +0.0979 | (tool prints no CI on the headline) | quoted verbatim from `cf_audit_bias_map.md` |
| **`sd_true_excess`, population-weighted** (THE REGISTERED METER) | **0.2550** | per-decile 0.221 – 0.304 | verbatim from the tool; G0's comparator range was 0.11 – 0.36 |
| **Brier (population-weighted)** | **0.1943** | [0.1749, 0.2143] | rollout-level, forecast = that state's V |
| ├ reliability (lower better) | 0.0106 | [0.0063, 0.0192] | 4.6 % of the Brier score |
| ├ resolution (higher better) | 0.0476 | [0.0348, 0.0635] | |
| ├ uncertainty = base-rate cap p(1−p) | 0.2331 | [0.2193, 0.2425] | base rate p̂ = 0.6302 |
| └ skill score 1 − BS/UNC | +0.166 | [+0.097, +0.231] | |
| **Registered pass/fail bar** | **NONE** — see §7 verbatim | — | the registration declares this read DESCRIPTIVE |
| Contention | factor **1.00** (16 cpus) | loadavg over the run: min 2.0 / median 18.0 / max 35.3 | a second arm (`ai_v12_04_pfsp_fork25M`) launched during the run; cost is wall-clock only |

**The three sentences the numbers support, each with its interval:**

1. **The replay is exact.** Every bot battle in the 75M-era frame anchored — 142/142, 0 errors. The
   audit is not INCONCLUSIVE and the labels are quotable. (The dry run's 80–87.5 % refusal was the
   `--checkpoint` defect, root-caused `1a1ad063`; dropping the flag reproduces its 100 % result.)
2. **V exceeds the MC win fraction by +0.0965 [+0.0671, +0.1268]** on the frame population — the CI
   excludes 0, so an offset IS detected, at roughly 1.4–1.9× the |0.05|–|0.07| G0 measured on the
   shaped run's head.
3. **The offset is not where the head is worst.** Reliability is 0.0106 [0.0063, 0.0192] against a
   0.2331 base-rate cap — the miscalibration costs ~4.6 % of the Brier score, while the head buys
   back only 0.0476 [0.0348, 0.0635] of resolution. The within-decile true spread the head does not
   resolve, `sd_true_excess = 0.2550`, is **2.6×** the offset. The G0 reframe (defect = RESOLUTION,
   not offset) reproduces on a run whose critic IS the win-prob head.

---

## 2. What was run, and how it complies with the registration

| registration says | what ran |
|---|---|
| `cf_audit <run> --step <75M step dir> --impl rust --rollouts 8 --states N --out <outside models/>` with **NO `--checkpoint`** (runbook change, ledger `1a1ad063`) | exactly that; `--checkpoint` never passed, so the anchor AND the label rollouts both use the trace dir's own `snapshot.zip` (tier `exact`) |
| `--rollouts 8` (registered) | `--rollouts 8` |
| `--states N` (the runbook suggested 400) | **`--states 800`** — a superset, taken because the box was quiet and the cost is linear (0.17 s/rollout at load ≤ 8). Same sampler, same seed; only n changed. **Declared deviation, in the direction of precision.** |
| `--anchors` (default 20) | **`--anchors 150`**, which takes ALL 142 bot battles in the frame — the reproduction rate is then a census of the bot pool, not a sample of it |
| reproducibility: fixed seeds, `concurrency = 1` | `--seed 0`; `cf_audit` has **no concurrency knob** — it is a single serial process and pins `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = 1` and `torch.set_num_threads(1)` itself, which is the CLAUDE.md rule satisfied by construction |
| CPU, never cuda | the prober session loads the snapshot on CPU; the run never touched the GPU (a training arm held it throughout) |
| nothing written under `models/` | every output under `/home/goodlad/.claude/jobs/9ab51de6/tmp/identity_75M/`, then copied here |

**Trace dir.** `eval_traces/step_74000016` — the LAST COMPLETE eval cycle. **The final eval at
75M wrote no trace dir**: `eval_results.jsonl` ends at step 74000016 (37 rows) and the newest
`eval_traces/` entry is `step_74000016` (written 09:06), while `final_model.zip` is 09:41. So the
75M read is taken at 74.0M, 1.0M steps (1.3 %) before the final weights, on the snapshot that
actually played those battles.

**Anchor coverage bound, stated because it is not obvious.** `cf_audit`'s anchor pool is
`opp_class == "bot"` only (142 of the frame's 242 battles). The 100 sentinel battles are **not
anchored by construction** — their label trust is inherited from the same replay driver and the same
`snapshot.zip`, not separately measured.

---

## 3. Reliability curve — 10 bins on V

Two columns per bin, and the difference between them is the sampler, not the critic:
**sample** = the states as drawn (stratified, 4× boost on high-confidence-lost states, on top of
loss-enriched traces); **population** = the same states recombined at the frame's own
(decile × outcome) mass inside the stratum, which is what `cf_audit` reports. Wilson is on the
bin's raw rollout counts; the population columns carry battle-clustered bootstrap intervals
(4 000 draws).

| bin (V) | states | battles | rollouts | V̄ (sample) | p̂ (sample) [Wilson] | V̄ (pop) | p̂ (pop) [boot] | gap V−p̂ (pop) [boot] |
|---|---|---|---|---|---|---|---|---|
| [0.0,0.1) | 8 | 6 | 64 | 0.031 | 0.016 [0.003, 0.083] | 0.031 | 0.016 [0.000, 0.042] | +0.015 [−0.018, +0.044] |
| [0.1,0.2) | 8 | 8 | 64 | 0.154 | 0.094 [0.044, 0.190] | 0.154 | 0.094 [0.000, 0.219] | +0.060 [−0.068, +0.159] |
| [0.2,0.3) | 9 | 8 | 72 | 0.247 | 0.111 [0.057, 0.204] | 0.247 | 0.111 [0.056, 0.172] | +0.136 [+0.066, +0.206] |
| [0.3,0.4) | 13 | 12 | 104 | 0.348 | 0.183 [0.120, 0.268] | 0.348 | 0.183 [0.000, 0.367] | +0.165 [−0.022, +0.341] |
| [0.4,0.5) | 24 | 22 | 192 | 0.455 | 0.286 [0.227, 0.354] | 0.454 | 0.280 [0.176, 0.398] | +0.174 [+0.059, +0.275] |
| [0.5,0.6) | 37 | 29 | 296 | 0.551 | 0.456 [0.400, 0.513] | 0.551 | 0.457 [0.357, 0.565] | +0.093 [−0.011, +0.192] |
| [0.6,0.7) | 65 | 53 | 520 | 0.651 | 0.613 [0.571, 0.654] | 0.651 | 0.614 [0.548, 0.685] | +0.037 [−0.033, +0.103] |
| [0.7,0.8) | 232 | 118 | 1 856 | 0.749 | 0.587 [0.564, 0.609] | 0.750 | 0.663 [0.612, 0.713] | +0.086 [+0.037, +0.138] |
| [0.8,0.9) | 202 | 101 | 1 616 | 0.845 | 0.663 [0.639, 0.685] | 0.846 | 0.744 [0.688, 0.797] | +0.102 [+0.049, +0.158] |
| [0.9,1.0) | 202 | 96 | 1 616 | 0.957 | 0.714 [0.692, 0.736] | 0.967 | 0.849 [0.784, 0.904] | +0.117 [+0.065, +0.180] |

Reading it: the curve is **monotone** (population p̂ rises with V̄ at every step) and sits **below
the diagonal in every bin** — the gap point estimate is positive 10 times out of 10. **Five of the
ten gap CIs exclude 0** ([0.2,0.3), [0.4,0.5), [0.7,0.8), [0.8,0.9), [0.9,1.0)); the five that
straddle are the four thinnest bins (8–37 states) and [0.6,0.7). The gap is widest in the sparse
middle (0.2–0.5) and settles at **+0.086 to +0.117** across the three bins that hold 636 of the 800
drawn states.

⚠️ **The 0.9 bin is the one to be careful with.** Its population p̂ is 0.849 [0.784, 0.904]: on the
frame population, a state the critic calls ≥ 0.9 wins ~85 % of the time. The as-drawn column
(0.714) is far lower because the sampler deliberately over-draws lost battles from exactly that
region — quoting 0.714 would be quoting the sampler.

---

## 4. Brier decomposition (Murphy), population-weighted

Every MC rollout is one Bernoulli outcome y; the forecast is its state's V. Because V is not
constant inside a bin, the exact identity carries a within-bin forecast-variance term, reported
rather than hidden. Intervals: battle-clustered bootstrap, 4 000 draws, each draw recomputing the
whole decomposition.

| term | population-weighted | 95 % CI | as-drawn sample |
|---|---|---|---|
| Brier score | **0.1943** | [0.1749, 0.2143] | 0.2512 |
| reliability (REL, ↓) | **0.0106** | [0.0063, 0.0192] | 0.0329 |
| resolution (RES, ↑) | **0.0476** | [0.0348, 0.0635] | 0.0197 |
| uncertainty (UNC = base-rate cap p(1−p)) | **0.2331** | [0.2193, 0.2425] | 0.2395 |
| base rate p̂ | 0.6302 | — | 0.6027 |
| within-bin forecast variance | 0.0008 | — | 0.0008 |
| residual (BS − (REL − RES + UNC + WBV)) | −0.0025 | — | −0.0023 |
| skill score 1 − BS/UNC | **+0.166** | [+0.097, +0.231] | −0.049 |

The residual is the within-bin covariance between V and y (the resolution the binning throws away);
at −0.0025 it is 1.3 % of the Brier score.

**Base-rate cap.** A constant forecast at the base rate scores 0.2331; the critic scores 0.1943, so
it captures **16.6 % [9.7, 23.1] of the available Brier improvement** on this population. On the
as-drawn (loss-enriched, conviction-boosted) sample the same critic scores WORSE than the constant
(skill −0.049) — the clearest single illustration of why the population weighting is not optional.

---

## 5. By turn bucket and by opponent stratum

Buckets are the ones the read registered (early ≤ 10 / mid 11–24 / late ≥ 25), NOT `cf_audit`'s
data-driven terciles (its edges here are 10 / 19; its own table is in §6).

| stratum | states | battles | V̄ | MC | **bias (pop)** | 95 % CI | REL | RES | UNC | skill | sd_true_excess† |
|---|---|---|---|---|---|---|---|---|---|---|---|
| early (turn ≤ 10) | 279 | 156 | 0.762 | 0.696 | **+0.0015** | **[−0.0335, +0.0385]** | 0.0042 | 0.0157 | 0.1987 | +0.060 | 0.227 |
| mid (11–24) | 355 | 163 | 0.785 | 0.559 | **+0.1412** | [+0.1024, +0.1815] | 0.0227 | 0.0618 | 0.2403 | +0.174 | 0.351 |
| late (turn ≥ 25) | 166 | 63 | 0.772 | 0.539 | **+0.1687** | [+0.0963, +0.2474] | 0.0388 | 0.0877 | 0.2490 | +0.209 | 0.385 |
| **bot** | 406 | 121 | 0.814 | 0.619 | **+0.1043** | [+0.0556, +0.1570] | 0.0148 | 0.0412 | 0.2169 | +0.135 | 0.354 |
| **pool (sentinel)** | 394 | 93 | 0.734 | 0.586 | **+0.1025** | [+0.0667, +0.1381] | 0.0120 | 0.0495 | 0.2460 | +0.155 | 0.295 |
| won battles | 182 | 88 | 0.846 | 0.877 | −0.0311 | [−0.0606, −0.0000] | 0.0028 | 0.0116 | 0.1082 | +0.090 | 0.185 |
| lost battles | 618 | 126 | 0.753 | 0.522 | +0.1876 | [+0.1471, +0.2278] | 0.0444 | 0.0304 | 0.2479 | −0.060 | 0.295 |

† whole-stratum spread of the true p with the R = 8 binomial floor subtracted — **not** the
registered within-decile meter (§6's table is that one). It is larger by construction because it
also contains the across-decile spread.

**Contrasts, with the DIFFERENCE's interval** (two overlapping bars are not a comparison):

| contrast | Δ bias (pop) | 95 % CI | reading |
|---|---|---|---|
| late − early | **+0.1672** | [+0.0863, +0.2511] | the offset is a LATE-GAME phenomenon; excludes 0 |
| mid − early | +0.1397 | [+0.0845, +0.1931] | it has already appeared by turn 11 |
| **bot − pool (sentinel)** | **+0.0018** | **[−0.0597, +0.0674]** | **NOT DETECTED** — the two ecologies carry the same offset to within ±0.06 |
| lost − won battles | +0.2186 | [+0.1668, +0.2681] | the G0 "conviction class" shape, reproduced |

🚨 **The pool stratum is cross-regime and DESCRIPTIVE.** Arm A ran WITHOUT `--eval-sentinel-greedy`
(confirmed in its `original_command`), i.e. the old greedy-trainee-vs-stochastic-sentinel regime —
the handicap measured at **+8.9 pp [+7.0, +10.7] in the trainee's favour**. Its sentinel games are
therefore easier than a symmetric pool game, and both V and the MC label are taken in that same
handicapped regime (`prober.replay.build_opponent` plays a ckpt opponent stochastic, matching what
`eval_worker` recorded). The bot−pool contrast above is a like-for-like comparison of the OFFSET
within each regime; the pool stratum's absolute level is not comparable to arm C's, which runs
greedy-symmetric.

---

## 6. `cf_audit`'s own output, quoted

Full file: `cf_audit_bias_map.md` (and `.json`). The registered meter and the strata the
registration named:

**RESOLUTION — within-decile true spread vs the binomial floor** (the registered primary meter):

| decile | n | predicted | MC | sd(MC) | binomial floor | **sd_true_excess** | % variance real |
|---|---|---|---|---|---|---|---|
| 3 | 13 | 0.348 | 0.183 | 0.317 | 0.090 | **0.304** | 92.0 % |
| 4 | 24 | 0.454 | 0.280 | 0.284 | 0.133 | **0.251** | 78.0 % |
| 5 | 37 | 0.551 | 0.457 | 0.272 | 0.159 | **0.221** | 66.0 % |
| 6 | 65 | 0.651 | 0.614 | 0.284 | 0.150 | **0.242** | 72.2 % |
| 7 | 232 | 0.750 | 0.663 | 0.296 | 0.140 | **0.261** | 77.8 % |
| 8 | 202 | 0.846 | 0.744 | 0.279 | 0.127 | **0.249** | 79.3 % |
| 9 | 202 | 0.967 | 0.849 | 0.282 | 0.084 | **0.269** | 91.2 % |

Headline: `population-weighted gap +0.0979` · `population-weighted sd_true_excess 0.2550` ·
800 labels over 214 battles.

**The conviction class** (verbatim):

```
n=379 over 92 battles
predicted 0.861 (median 0.849)  vs tight-MC 0.579 (median 0.625)
gap +0.2818  CI [+0.2288, +0.3394]
LOSS - WIN difference +0.2928  CI [+0.2254, +0.3547]
MC >= 0.75 (the critic was RIGHT; the dice lost it)  40.9%
MC <  0.50 (the critic was genuinely wrong)          30.9%
MC <  0.25 (badly wrong)                             14.5%
```

**By opponent** (the tool's own cells, unweighted within cell, battle-clustered CIs):

| stratum | n | n_battles | mean_predicted | mean_mc | mean_gap | 95 % CI |
|---|---|---|---|---|---|---|
| aggressive | 17 | 10 | +0.8095 | +0.5515 | +0.2581 | [+0.0758, +0.4228] |
| aggressive_v2 | 57 | 17 | +0.7812 | +0.6184 | +0.1628 | [+0.0488, +0.2741] |
| heuristic | 69 | 18 | +0.8095 | +0.5761 | +0.2334 | [+0.1145, +0.3649] |
| heuristic2 | 46 | 13 | +0.8218 | +0.7065 | +0.1152 | [−0.0307, +0.2497] |
| random | 18 | 7 | +0.9075 | +1.0000 | −0.0925 | [−0.1199, −0.0665] |
| sentinel_0 | 94 | 17 | +0.7720 | +0.5545 | +0.2175 | [+0.1394, +0.2935] |
| sentinel_1 | 66 | 18 | +0.7025 | +0.4735 | +0.2290 | [+0.1337, +0.3189] |
| sentinel_2 | 78 | 18 | +0.6853 | +0.5913 | +0.0939 | [+0.0381, +0.1565] |
| sentinel_3 | 60 | 20 | +0.7227 | +0.6208 | +0.1019 | [+0.0287, +0.1564] |
| sentinel_4 | 96 | 20 | +0.7632 | +0.6693 | +0.0940 | [+0.0169, +0.1729] |
| setup_sweep | 45 | 16 | +0.8210 | +0.6333 | +0.1877 | [+0.0785, +0.2942] |
| setup_sweep_v2 | 46 | 15 | +0.8193 | +0.5380 | +0.2812 | [+0.0992, +0.3812] |
| staller | 51 | 13 | +0.8286 | +0.6103 | +0.2183 | [+0.0364, +0.3559] |
| staller_v2 | 57 | 12 | +0.7941 | +0.5592 | +0.2349 | [−0.0286, +0.4663] |

**`random` is the only negative cell** (−0.0925 [−0.1199, −0.0665]): the MC win rate against it is
1.0000 and the critic still says 0.9075 — under-confidence where the game is already won.

**The tool's own turn terciles** (its edges, 10 / 19; loss and win rows kept apart as it emits them):

| stratum | n | n_battles | mean_predicted | mean_mc | mean_gap | 95 % CI |
|---|---|---|---|---|---|---|
| turn ≤ 10 / win | 73 | 57 | +0.7860 | +0.8613 | −0.0753 | [−0.1148, −0.0301] |
| turn ≤ 10 / loss | 206 | 99 | +0.7538 | +0.6377 | +0.1160 | [+0.0700, +0.1625] |
| turn 10–19 / win | 61 | 49 | +0.8793 | +0.8689 | +0.0104 | [−0.0413, +0.0714] |
| turn 10–19 / loss | 190 | 93 | +0.7534 | +0.4605 | +0.2929 | [+0.2363, +0.3496] |
| turn > 19 / win | 48 | 35 | +0.8934 | +0.9115 | −0.0181 | [−0.0569, +0.0353] |
| turn > 19 / loss | 222 | 64 | +0.7528 | +0.4668 | +0.2860 | [+0.2048, +0.3745] |

**By material: ABSENT.** The registration's read asked for "the bias map by turn bucket and by
material"; `cf_audit` emits no material stratum (its cells are decile × outcome × turn tercile ×
opponent). Nothing was suppressed — the column does not exist. **Evidential and twin-head columns
are ABSENT too**, not zero: this checkpoint carries neither `cf_evid_head` nor
`cf_twin_head_b`/`cf_shadow_head`, and the tool says so on stderr.

---

## 7. The registered bar

Verbatim, from the registration (`44e3a7f5`, ledger 2026-09-07 · *THREE VALUE-FUNCTION TESTS at
75M*, test 1):

> **No pass/fail bar is registered — the reading is DESCRIPTIVE, with the shaped-critic G0 map as
> the comparator where strata match.**

and, for what to read:

> Read: the bias map by turn bucket and by material; `sd_true_excess` (resolution) as the meter, per
> the G0 verdict's rule that the defect class on the shaped critic was RESOLUTION, not offset.

**So there is nothing to be "met", and no delta-inside-a-bar statement is available to make.** The
one threshold this test does carry is the LABEL-TRUST gate, `--anchor-tolerance 0.9`, and it is
**met: 142/142 = 100.0 %, Wilson [0.974, 1.000], the whole interval above 0.90.**

**Against the G0 comparator, where the strata match** (G0: gen-17 @24M, shaped critic, win-prob
head, 2 204 labels / 216 battles, same R = 8) — a comparison of two independently measured
constants, with only THIS read's interval available, so it is descriptive and no delta CI can be
formed from published summaries:

| quantity | G0 (2026-08-22, shaped run) | here (75M, win-prob critic) |
|---|---|---|
| population-mean gap | \|0.05\|–\|0.07\|, sign flips with the weighting | +0.0965 [+0.0671, +0.1268] (pop), +0.0979 (tool) |
| within-decile `sd_true_excess` | 0.11–0.36, 80–95 % real | 0.221–0.304, 66–92 % real; pop-weighted 0.2550 |
| per-state error vs aggregate offset | 2–6× | **2.6×** |
| conviction class (V ≥ 0.75, lost) gap | +0.231 [+0.154, +0.313] | +0.2818 [+0.2288, +0.3394] |
| … of which genuinely winning (MC ≥ 0.75) | 53.1 % [42.0, 64.5] | 40.9 % |
| bot vs pool sign | bots −0.065 (true-play) / pool +0.106 — **sign flips** | bots +0.1043, pool +0.1025 — **same sign**, Δ +0.0018 [−0.0597, +0.0674] |

---

## 8. Caveats, and anything that contradicted the registration

- **Nothing contradicted the registration's shape.** The `1a1ad063` runbook change (no
  `--checkpoint`) is confirmed correct at 75M: the anchor rate went from the dry run's 80–87.5 %
  (with the override) to 100 % (without it), on the same instrument.
- **The 75M read is at 74.0M.** The final eval cycle wrote no trace dir; the newest is
  `step_74000016`. Every number here is the 74.0M snapshot against itself.
- **Deviation from the dry-run runbook, declared:** `--states 800` (runbook suggested 400) and
  `--anchors 150` (default 20). Both increase n only. Same seed 0, same sampler
  `cf_audit_strata_v1`, same R = 8.
- **The sampler is not the population, and the traces are not the cycle.** The eval quota persists
  the first 10 losses and first 5 wins per opponent (`selection_schema 1`, recorded in
  `eval_manifest.json`), and `cf_audit` then boosts high-confidence-lost states 4×. Population
  columns recombine at frame mass and cover **99.9 %** of it; sample columns are reported beside
  them only so the difference is visible.
- **Structural coverage bounds, from the tool's own accounting:** turn-1 decisions (242, one per
  battle) and forced-switch rounds (936) are outside the frame. Sentinel battles are outside the
  ANCHOR pool.
- **R = 8 per state.** A single label's own sd is ≤ 0.177 (95 % half-width ±0.35). Bin and stratum
  aggregates are honest; no single state's MC is a point value. `sd_true_excess` subtracts exactly
  this floor, which is why it is the meter.
- **The MC label is measured on the EVAL distribution, played greedy**, while the head was trained
  on a mostly-self-play mixture with a stochastic actor. Per the standing rule, never quote a gap
  without naming the population — the sign of the G0 read depended on it.
- **Not INCONCLUSIVE:** 0 errors on 800 label tasks and 0 anchor errors — a 0 % timeout rate against
  the 25 % ceiling. Contention factor 1.00 at the check; loadavg ranged 2.0–35.3 as a second
  training arm launched mid-run, which costs wall-clock and nothing else.

---

## 9. Commands, exactly as run

```bash
# worktree: /home/goodlad/dev/gen3ai-wt/read75-identity (branch read75-identity-0908, from main b697e82a)
export PYTHONPATH=$PYTHONPATH:src
cargo build --release --bin sim_bridge --bin search_driver --manifest-path src/rust_sim/Cargo.toml

# TEST 1 — the identity test.  NO --checkpoint (ledger 1a1ad063).  17.79 core-min, exit 0.
nohup nice -n 10 /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
  -m agents.training.cf_audit /home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic \
  --step 74000016 --rollouts 8 --states 800 --anchors 150 --impl rust \
  --out /home/goodlad/.claude/jobs/9ab51de6/tmp/identity_75M/cf_audit \
  --seed 0 --deadline-min 90 \
  > /home/goodlad/.claude/jobs/9ab51de6/tmp/identity_75M/cf_audit.log 2>&1 &

# the readout: join the labels to the traces, then every table in this file
python identity_readout.py --build \
  --labels /home/goodlad/.claude/jobs/9ab51de6/tmp/identity_75M/cf_audit/cf_labels/labels_cf_audit_74000016.jsonl \
  --traces /home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic/eval_traces/step_74000016
python identity_readout.py            # -> identity_stats.json
python -c "import json, identity_readout as R; ..."   # -> identity_contrasts.json (see §5)
```

A pilot at `--states 8 --anchors 6` (31 s, anchors 6/6) preceded the run; its output is not quoted.

## 10. Files

Everything below is committed beside this report; nothing is read from a job/tmp directory at run
time.

| file | what |
|---|---|
| `identity_test.md` | this report |
| `identity_readout.py` | the readout — `--build` joins labels to traces; a bare run recomputes every table; `--check` resolves inputs |
| `identity_labels.json` | 800 joined rows (V, MC, wins/n, turn, opponent, opp_class, outcome, battle) + the frame's mass tables — 276 KB, the raw input for every number here |
| `identity_stats.json` | every computed statistic, with its bootstrap interval |
| `identity_contrasts.json` | the four stratum contrasts of §5 |
| `cf_audit_bias_map.md` / `cf_audit_bias_map.json` | the tool's own output, verbatim |
| `cf_labels_74000016.jsonl` | the emitted label file (shared v1 schema), 456 KB |
| `cf_audit_run.log` | the run's stdout, including the anchor line |

Source artifacts (NOT committed, and not needed to reproduce the tables):
`/home/goodlad/dev/gen3ai/models/ai_v12_02_winprob_critic/eval_traces/step_74000016/` (242 traces +
`snapshot.zip`, git hash `f971caf2`, arch `gen3_critic_route_wave_v1`, config version 110) and the
job tmp dir `/home/goodlad/.claude/jobs/9ab51de6/tmp/identity_75M/`.
