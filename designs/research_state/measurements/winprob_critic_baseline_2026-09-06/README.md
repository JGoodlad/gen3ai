# WIN-PROB HEAD CALIBRATION BASELINE — `ai_v9_59_R2ACTION_0827`, 2026-09-06

**Why this exists.** The next generation is to make the win-probability head the critic's only
signal ([`designs/ai_v12/design_winprob_only_critic.md`](../../../ai_v12/design_winprob_only_critic.md)).
That design's calibration gate needs a number the new critic must beat, measured with the
instruments that already exist, on the generation it replaces. This is that number.

**The subject.** `models/ai_v9_59_R2ACTION_0827` — the gen-era parent every rev-4 arm forks from
— at its two recorded eval-trace steps, **26,000,016** and **28,000,032**. 474 battles, 12,694
decisions, 14 opponents (9 scripted bots + 5 pool sentinels). Read-only; nothing was written into
`models/`.

**The head under test** is `win_head` at `--win-prob-mode shaping --win-prob-coef 0.05`: one logit
off `value_pooled`, BCE against the episode's own outcome broadcast undiscounted to every decision
of that episode. It is a SIDE readout — it never enters `pi` or `vf`, and the live critic is the
51-atom `value_dist_head`. So this measures the barometer the new design proposes to promote, as
that barometer exists today.

---

## The commands

```bash
export PYTHONPATH=$PYTHONPATH:src
# (1) the RAW capture quota — what the traces literally hold
python -m main.scaffolding_gauge models/ai_v9_59_R2ACTION_0827 \
  --reliability --boot 2000 --seed 0 --out <this dir>/raw_capture_quota.json
# (2) SELECTION-REWEIGHTED to the eval cycle's own recorded win rates — the quotable one
python -m main.scaffolding_gauge models/ai_v9_59_R2ACTION_0827 \
  --reliability --reliability-reweight --boot 2000 --seed 0 \
  --out <this dir>/selection_reweighted.json
```

Console output is committed verbatim as `console_raw.txt` / `console_reweighted.txt`; the JSON
carries the per-bin tables, the per-opponent strata and the `units` block stating what each number
cannot claim. Tree at `407b27c0`. `--reliability` / `--reliability-reweight` were built in the same
pass as this baseline — see *What was missing*, below.

---

## 🚨 READ THIS FIRST — the raw table is not a calibration of this head

**The eval recorder's quota is loss-enriched, and by a lot.** The captured slice's outcome rate is
**0.456 / 0.463**; the same cycles' own `eval_results.jsonl` records **0.901 vs bots and 0.702 vs
the pool**. So the raw table scores the head against a population it was never deployed against,
and its large positive ECE is mostly the quota, not the head. Every headline below is therefore the
**REWEIGHTED** one — each opponent's rows importance-weighted back to the win/loss mix that
opponent's own eval row recorded, weights constant within a battle so the cluster bootstrap stays
valid. The raw table is committed beside it because the size of the correction is itself the
finding.

| | raw quota | reweighted |
|---|---|---|
| base rate (26M / 28M) | 0.456 / 0.463 | 0.761 / 0.791 |
| ECE (26M / 28M) | **0.237 / 0.281** | **0.025 / 0.035** |
| skill (26M / 28M) | 0.071 / **−0.080** | **+0.336 / +0.265** |

Read raw-first and you conclude the head is grossly optimistic and, at 28M, worse than a coin
weighted to the base rate. Both readings are artifacts of the quota.

---

## THE BASELINE (selection-reweighted; 95% cluster bootstrap over BATTLES, 2000 resamples)

`skill` = 1 − Brier/Brier_base — 0 means no better than the slice's own base rate.
`rel` (Murphy reliability, lower better) and `res` (resolution, **higher** better) sum with
`unc` to the Brier: `BS = REL − RES + UNC`.

| step | stratum | n | ess | battles | base | **Brier** [95% CI] | **skill** [95% CI] | **ECE** | MCE | rel | res | unc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 26,000,016 | **all** | 6487 | 4247 | 240 | 0.761 | **0.1207** [0.1022, 0.1427] | **+0.336** [0.266, 0.391] | **0.0249** | 0.147 | 0.0013 | 0.0618 | 0.1817 |
| 26,000,016 | bot | 3208 | 1732 | 143 | 0.877 | 0.0767 [0.0612, 0.0973] | +0.291 [0.185, 0.374] | 0.0395 | 0.178 | 0.0027 | 0.0337 | 0.1082 |
| 26,000,016 | pool | 3279 | 2592 | 97 | 0.661 | 0.1591 [0.1277, 0.1950] | +0.290 [0.189, 0.368] | 0.0667 | 0.137 | 0.0064 | 0.0711 | 0.2242 |
| 28,000,032 | **all** | 6207 | 4188 | 234 | 0.791 | **0.1216** [0.0979, 0.1473] | **+0.265** [0.170, 0.343] | **0.0349** | 0.122 | 0.0020 | 0.0445 | 0.1653 |
| 28,000,032 | bot | 3179 | 1857 | 141 | 0.896 | 0.0723 [0.0565, 0.0920] | +0.222 [0.101, 0.306] | 0.0228 | 0.156 | 0.0012 | 0.0215 | 0.0929 |
| 28,000,032 | pool | 3028 | 2396 | 93 | 0.687 | 0.1705 [0.1269, 0.2203] | +0.208 [0.067, 0.319] | 0.0875 | 0.221 | 0.0103 | 0.0530 | 0.2152 |

Per-opponent rows (14 per step) are in the JSON and the console transcripts.

### The reliability curve, 26M, stratum `all` (reweighted)

| forecast bin | n | p̄ | realized | gap (p̄ − y) |
|---|---|---|---|---|
| 0.0–0.1 | 366 | 0.042 | 0.000 | +0.042 |
| 0.1–0.2 | 227 | 0.148 | 0.044 | +0.104 |
| 0.2–0.3 | 240 | 0.253 | 0.106 | +0.147 |
| 0.3–0.4 | 246 | 0.355 | 0.251 | +0.104 |
| 0.4–0.5 | 320 | 0.452 | 0.471 | −0.019 |
| 0.5–0.6 | 441 | 0.554 | 0.599 | −0.044 |
| 0.6–0.7 | 635 | 0.653 | 0.630 | +0.022 |
| 0.7–0.8 | 1046 | 0.758 | 0.727 | +0.031 |
| 0.8–0.9 | 1316 | 0.848 | 0.818 | +0.030 |
| 0.9–1.0 | 1650 | 0.966 | 0.969 | −0.003 |

---

## What it says

**1. In the MEAN the head is very nearly calibrated, and that is not the good news it sounds
like.** Murphy reliability is **0.0013 / 0.0020** — a hundredth of the Brier. Every large gap in
the table above sits in the three sparse low-forecast bins (0.1–0.4, 713 of 6487 rows) where the
head says 15–35% and wins 4–25%; across the mass of the distribution it is within 0.05.

**2. RESOLUTION is the deficit, exactly as `win_prob_decomposition.md` axis 2 predicts.** Against
an available uncertainty of 0.182 the head resolves **0.062 — 34%** at 26M, and **0.045 — 27%** at
28M. The head separates far less of the outcome variance than is there to separate. This is the
blur the standing critic-calibration plan names as the disease and `sd_true_excess` meters, now
measured for this run in Murphy coordinates on existing traces at zero GPU cost.

**3. It got WORSE over 2M steps, on the resolution axis only.** Skill 0.336 → 0.265, resolution
0.062 → 0.045 (bot class 0.034 → 0.022) while reliability stayed at ~0.002. The two steps are
different eval quotas so this is **not a controlled comparison** — but the direction agrees with
the rank gauge, which moved the same way over the same interval (ρ 0.915 → 0.880).

**4. The ecology split reproduces.** The pool class carries **2.4× / 8.4×** the bot class's
reliability error (0.0064 vs 0.0027; 0.0103 vs 0.0012) and 2–4× the ECE. The head is BCE-trained on
a ~90% self-play mixture and is nonetheless *less* calibrated against recent selves than against
the scripted bots. A pooled number averages the two and describes neither
(`win_prob_decomposition.md` axis 3).

**5. An affine map of the SHAPED CRITIC out-predicts the win-prob head on the raw slice** —
Brier 0.1835 vs 0.2304 at 26M, 0.2011 vs 0.2686 at 28M (`affine_brier_v_affine` vs
`affine_brier_head`, both in `raw_capture_quota.json`'s `curve`). The head is a *worse outcome
forecaster than a one-parameter readout of the critic it is supposed to be a cleaner alternative
to*, on the population as captured. **UNVERIFIED whether this survives reweighting** — the affine
gauge has no weighted form, so this comparison exists only on the raw quota, and the raw quota is
the regime where the head's optimism is most inflated. Closing that is the first item on the
design's gap list.

---

## What the numbers cannot claim

- **Not a random sample of play.** These are the recorder's quota; the reweighting corrects the
  win/loss MIX per opponent, not any other selection the quota applies (turn depth, battle length,
  which battles within an outcome class got kept).
- **`n` is not a sample size.** Labels are per-battle and broadcast to every decision, so the
  clusters are 234–240 battles, not 6207 states. Only the cluster CIs are honest; `ess` reports
  what the reweighting cost (4188 of 6207 at 28M).
- **The two steps are not an A/B.** The quota moves under them.
- **This is the head as a BAROMETER.** It has never carried a gradient into `pi`, so nothing here
  predicts how it calibrates once it becomes the critic and its own errors start steering the
  policy that generates its labels.
- **Draws are labelled as losses** (`wrappers.py:491-493` — `won is None` ⇒ 0.0), so every drawn
  episode's decisions are scored against a loss label here as in training.
- The `random` opponent at 26M was captured as 8 wins of 8 battles: one outcome class, nothing to
  reweight, **weight 0**, and it is named as such in the console output rather than silently
  dropped.

---

## What was MISSING (and is now built)

Neither existing instrument could produce this table:

- `python -m main.prober.query calibration` reads the **scalar critic** `values` against the
  realized shaped return `G(s)`. It never touches `win_probs` and has no notion of outcome
  probability, so it cannot score this head.
- `python -m main.scaffolding_gauge` reads `win_probs`, but as one side of a **divergence** gauge
  against `V`. Its `affine_gauge` computed `brier_head` as a by-product disclaimer; there was no
  reliability curve, no ECE, no Murphy split, no opponent-class stratification, and no correction
  for the capture quota.

Built in this pass, and what produced the tables above:

| piece | file |
|---|---|
| `reliability_table(win_probs, outcomes, bins=, weights=)` — Brier / skill / ECE / MCE / Murphy REL-RES-UNC + `ess` | `src/agents/training/scaffolding.py` |
| `--reliability` / `--reliability-bins` — per-step, stratified `all`/`bot`/`pool`/per-opponent, cluster-bootstrap CIs | `src/main/scaffolding_gauge.py` |
| `--reliability-reweight` + `true_win_rates()` / `selection_weights()` — the quota correction, REFUSING rather than falling back | `src/main/scaffolding_gauge.py` |
| tests | `src/agents/training/scaffolding_test.py`, `src/main/scaffolding_gauge_test.py` |

The default JSON and the default render are unchanged: both flags are opt-in, and a test pins that
the `reliability` block is absent unless asked for.
