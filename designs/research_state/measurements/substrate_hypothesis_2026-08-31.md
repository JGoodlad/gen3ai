# M3 — THE SUBSTRATE HYPOTHESIS: does an untaught gift need prior competence to attach to?

**Status: COMPLETE.** Pure re-analysis of two committed cell-level artifacts plus the run
archive's step counts — **no battles, no models, no traces.** Predictions were registered in the
dispatch and are scored below without adjustment. Script:
`designs/research_state/measurements/substrate_hypothesis.py`; full numbers in the sibling
`.json`.

---

## 0. Headline

**REFUTED at its own mechanism.** On the ability-additive (log-odds) scale — the scale this
codebase's own Bradley-Terry ELO subsystem already assumes — a fold's untaught gift is **FLAT in
the parent's prior competence** in both eras:

| era | fold | n teams | gift | **prior-competence correlation** (split-half, logit) |
|---|---|---|---|---|
| **v8** | `ai_v8_14` − `ai_v8_04` | 16 | **+5.42pp** | **+0.013 [−0.283, +0.323]** |
| **gen (rev-3)** | `R3ACTION` − `R2ACTION` | 8 | −0.75pp | **−0.174 [−0.521, +0.130]** |
| pooled (within-fold z-scored) | | 24 | | **−0.050 [−0.260, +0.160]**, P(r>0) = 0.31 |

The hypothesis predicts a POSITIVE correlation. Two eras with **opposite-signed aggregate
effects** — v8 gifted, rev-3 was null — produce the same within-fold answer: nothing.

**So the between-era claim collapses without needing to be measured.** Even granting the
hypothesis's antecedent (v8's parent trained **277.18M** steps against the gen-era parent's
**28.07M**, a **9.88×** exposure ratio — the one half that IS decidable from artifacts), higher
prior competence cannot explain v8's gift, because *within* v8 the gift does not track competence.

⚠️ **The most important finding is methodological, and it nearly went the other way.** Read
against zero, the headroom-normalised correlation at the tightest ceiling is
**+0.397 [+0.061, +0.660]** — an interval excluding zero, which would have "confirmed" the
registered prediction. **Its own null is +0.944.** A ceiling control must be scored against the
correlation that control MANUFACTURES under a no-effect null, never against zero (§2.3).

---

## 1. The question, and the two nuisances that decide how it must be asked

**The hypothesis.** A 277M-step parent has played most of the 719-team pool many times, so v8's
"untaught" teams were teams the base already had real competence on and the teacher's content
could COMPOSE with existing knowledge. Gen-era untaught teams are ones the base barely knows.
**Prediction: within a fold, the gift on an untaught team should be LARGER where the parent was
already more competent.**

That is a within-fold correlation between `x = parent WR on team i` and
`y = fold WR − parent WR on team i`. Two nuisances corrupt it, and **both push it NEGATIVE**:

| # | nuisance | mechanism | sign |
|---|---|---|---|
| 1 | **ceiling headroom** | a team the parent wins 0.51 on has less room to gain than one at 0.22 | − |
| 2 | **shared measurement noise** | `p̂` appears in BOTH axes: `cov_noise(x,y) = cov(p̂,f̂) − var(p̂) < 0` | − |

Nuisance 2 is regression to the mean and **no transform fixes it** — it is a property of the
estimator, not of the scale. It is not small here: the parent's win rate is measured on 240
(v8) / 200 (rev) battles, so its measurement sd is **2.88pp** against an across-team sd of
**9.06pp** — **10.1%** of the x-axis variance is noise (v8), **27.1%** (rev-3).

Because both nuisances are negative, the test is **one-sided-informative**: a positive controlled
correlation would be conservative evidence FOR the hypothesis; a null is ambiguous between "no
substrate effect" and "an effect swamped by nuisances". That asymmetry is stated up front rather
than discovered in the discussion.

### Registered predictions (scored in §6, never adjusted)

1. Within v8, the headroom-controlled correlation between parent prior competence and gift is
   **POSITIVE but WEAK** (n=16; expect a wide interval).
2. The between-era comparison is **NOT cleanly decidable** from these artifacts (different
   reference opponents); the honest deliverable is a bound plus a named measurement.

---

## 2. Method

### 2.1 Data — cell-level, both probes

| source | teams | design |
|---|---|---|
| `v8_redistribution_pfsp_2026-08-30_cells.jsonl.gz` | 16 untaught + 6 taught | 8 opponent cells × 30 CRN-paired battles × 2 arms (parent `ai_v8_04`, fold `ai_v8_14`), reference opponent `ai_v8_03` |
| `rev3_untaught_pulldown_2026-08-30_cells.jsonl.gz` | 8 untaught | 200 CRN-paired battles × 3 arms (`R3ACTION` / `R2ACTION` / `REV1`), per-battle win vectors, reference opponent rev-1 final |

The rev artifact gives **two** fold events: rev-3 (`A−B`, parent `R2ACTION`) and rev-2 (`B−C`,
parent `REV1`). Only rev-3 is usable — see §3.3.

### 2.2 Nuisance 2 — the SPLIT-HALF instrument

`x` is measured on one half of the parent's battles and `y` on the **disjoint** other half, so
the two carry independent noise and the `−var(p̂)` term vanishes.

- **rev era:** the arms are index-aligned per-battle vectors, so the split is exact — `x` from
  index set `S`, `y` from its complement. `cov_noise(x,y) = 0` identically.
- **v8:** only per-cell win COUNTS survive, so each 30-battle cell is split 15/15 by a
  hypergeometric draw on its win count (opponent-balanced). The CRN pairing means a residual leak
  survives, priced at **~0.9% of the x-axis signal variance** against the 10.1% bias it removes.
  A second, **exactly** independent variant splits the 8 OPPONENT cells 4/4 (opponent-disjoint ⇒
  disjoint battles ⇒ CRN irrelevant, at the cost of opponent heterogeneity in `x`). Both are
  reported; they agree.

400 split replicates; cluster bootstrap over **teams** (20,000 reps, 40 splits averaged per
replicate, so split noise is inside the interval rather than hidden).

### 2.3 Nuisance 1 — and why "control for the ceiling" is not one thing

Two controls are conventional and **they are opposites**:

- **LOGIT.** `y = logit(f) − logit(p)`, `x = logit(p)`. Under Bradley-Terry,
  `logit(p_i) = θ_agent,i − θ_ref,i`, so a fold that improves the agent by `Δ_i` on team `i`
  gives `logit-gift = Δ_i` exactly. This is the scale `main.elo` already fits.
- **HEADROOM.** `y = (f − p) / (C − p)` — the fraction of the remaining distance to a ceiling `C`
  that the fold captured, at `C ∈ {0.60, 0.6881, 0.80, 1.00}` (the exploitability decomposition's
  own sensitivity ladder: 0.5742 coverage / 0.69–0.70 meter bracket the first two).

**Each is neutral under a DIFFERENT null, and manufactures a near-±1 correlation under the
other's.** That is not a nuance; it is the whole result:

| null | free parameter pinned to reproduce the fold's observed mean gift | `r_logit` | `r_headroom` |
|---|---|---|---|
| **log-odds-additive** (constant `Δ` for every team) | v8: `Δ = +0.232` | **0 by construction** | **+0.944 / +0.983 / +0.997 / +1.000** at C = 0.60 / 0.6881 / 0.80 / 1.00 |
| **headroom-additive** (constant fraction of `C − p`) | v8: `k` per C | **−0.997 … −0.986** | **0 by construction** |

Both nulls are computed noise-free on the observed prior-competence vector. So an observed value
is only interpretable against its own null. **Every headroom reading below is far BELOW its null;
the logit reading sits AT its null. The two agree: the data are a constant log-odds increment.**

---

## 3. Results — within fold

### 3.1 v8 (`ai_v8_14` − `ai_v8_04`), 16 untaught teams — the primary dataset

Parent WR **0.3828** (sd 0.0906, range 0.221–0.513); gift **+5.42pp** (sd 4.17pp).

| estimator | r (95% CI, cluster bootstrap over teams) | its own null |
|---|---|---|
| naive, raw pp | −0.070 [−0.477, +0.364] | 0 |
| naive, logit | −0.183 [−0.567, +0.271] | 0 |
| **split-half, raw pp** | **+0.088 [−0.214, +0.378]** | 0 |
| **split-half, LOGIT — primary** | **+0.013 [−0.283, +0.323]** | **0** |
| split-half, logit SLOPE | +0.004 [−0.194, +0.217] logits per logit | 0 |
| split-half, opponent-disjoint, raw | +0.124 [−0.092, +0.333] | 0 |
| split-half, opponent-disjoint, logit | +0.092 [−0.128, +0.306] | 0 |
| split-half, headroom C = 0.60 | +0.397 [+0.061, +0.660] | **+0.944** |
| split-half, headroom C = 0.6881 | +0.324 [−0.006, +0.595] | **+0.983** |
| split-half, headroom C = 0.80 | +0.266 [−0.052, +0.539] | **+0.997** |
| split-half, headroom C = 1.00 | +0.212 [−0.102, +0.490] | **+1.000** |

**Reading.**

1. **The shared-noise correction works and is the size predicted.** naive → split-half moves raw
   `r` by **+0.158**, against a −0.219 bias predicted analytically if the two arms were
   uncorrelated (they are CRN-paired, so the true bias is smaller — the observed shift sits inside
   the bound). x-axis reliability **0.899**; the disattenuated naive logit r is −0.193, i.e.
   attenuation is not what produced the naive negative — the shared noise is.
2. **The primary estimate is a precise zero.** +0.013 with a ±0.30 half-width. The two
   independent split constructions (hypergeometric-within-cell and opponent-disjoint) bracket it
   at +0.013 and +0.092, both null.
3. **The headroom column is NOT the positive it looks like.** Every one of the four ceilings sits
   0.55–0.79 BELOW its own null. Sign is stable across `C` (all positive) and magnitude declines
   monotonically with `C`, but the null declines the other way — so **the verdict does not flip
   with the ceiling assumption once each `C` is read against its own null**, and it is the same
   verdict as the logit column.

### 3.2 gen era, rev-3 (`R3ACTION` − `R2ACTION`), 8 untaught teams

Parent WR **0.4975** (sd 0.069); gift **−0.75pp** (sd 5.76pp) — the probe-Q null.

| estimator | r | its own null |
|---|---|---|
| naive, logit | **−0.462 [−0.889, −0.021]** | 0 |
| **split-half, LOGIT — primary** | **−0.174 [−0.521, +0.130]** | **0** |
| split-half, headroom C = 0.60 / 0.6881 / 0.80 / 1.00 | −0.174 / −0.192 / −0.188 / −0.182 | **−0.940 / −0.927 / −0.986 / −1.000** |

**The naive value is significant and it is an artifact.** Measurement noise is **27.1%** of this
x-axis's variance (only 8 teams, 200 battles each), and removing it takes `r` from −0.462 (CI
excludes zero) to −0.174 (CI spans it). *A naive correlation between a win rate and a delta built
on that same win rate will report a spurious negative, and here it reported one at the 95% level.*

Under the log-odds-additive null the headroom correlations would be −0.94 to −1.00 (the null is
negative here because the mean gift is negative); observed is −0.18. Same conclusion as v8: the
gift is a constant log-odds increment, flat in prior competence.

### 3.3 rev-2 (`R2ACTION` − `REV1`) — **UNIDENTIFIED, reported so it is not quoted**

Split-half logit `r = +0.366 [−0.118, +0.723]`. **Do not read this.** Two independent defects:

- **`x` has no signal.** rev-1's across-team WR sd on these 8 teams is **3.24pp** while its own
  measurement sd is **3.54pp** — measurement noise is **119%** of the x-axis variance, i.e. the
  estimated true between-team variance is **zero**. Reliability 0.000; disattenuation undefined.
- **`x`'s arm is a near-mirror.** `REV1` is also the fixed reference opponent. Probe P's own
  method note says exactly this biases the delta.

### 3.4 v8 taught teams (n = 6) — exploratory

Split-half logit `r = −0.300 [−0.858, +0.672]` on a **+26.18pp** gift. No competence dependence
there either, at a sample too small to carry weight. Recorded, not read.

---

## 4. The between-era claim

### 4.1 The levels, and why they are not comparable

| | v8 parent (`ai_v8_04`) | gen parent (`R2ACTION`) |
|---|---|---|
| untaught WR mean | **0.3828** (sd 0.0906, 0.221–0.513) | **0.4975** (sd 0.0691, 0.405–0.620) |
| reference opponent | `ai_v8_03` final | `ai_v9_29_rev1` final |
| opponent teams | 8 FIXED teams | 719-pool draw |
| battles per team | 240 | 200 |

Taken at face value the **gen-era parent looks MORE competent, not less** — the opposite of the
hypothesis's premise. **That reading is worthless**, and the reason is structural rather than
fixable by re-weighting: the two eras have different observation dimensions and different
`ARCH_SIGNATURE`s, so **no learned model can referee both**. There is no common opponent, no
common team-draw regime, and no common scale.

### 4.2 The half that IS decidable: exposure

The hypothesis's ANTECEDENT needs no common opponent — step counts are comparable directly. From
the run archive's own checkpoints:

| run | max checkpoint |
|---|---|
| `ai_v8_04_distill_4teacher_0722` (v8 parent) | **277,178,472** |
| `ai_v8_14_distill3_0725` (v8 fold) | 292,100,648 |
| `ai_v9_29_rev1_0823` | 24,988,992 |
| `ai_v9_59_R2ACTION_0827` (gen parent) | **28,067,760** |
| `ai_v9_70_R3ACTION_0828` (gen fold) | 32,621,088 |

**Ratio 9.88×.** Both folds ran `--distill-team-bias 0.4`, so ~60% of episodes draw over the
719-team pool; at ~50 env steps per episode that is **~4,626 vs ~468 episodes per pool team**
(estimates, caveated in the `.json`: the step-per-episode figure is nominal and lineage before
each run's own first step is not summed; v8_04's `--team-pfsp onesided` is measured near-inert,
TV distance from uniform ≤ 0.049).

**So the antecedent HOLDS — and it does not rescue the hypothesis**, because the consequent's
mechanism is absent inside both folds (§3). A ~10× exposure difference that buys no
competence-dependent gift within either era cannot be what makes v8's gift positive and rev-3's
null.

### 4.3 The named measurement that WOULD decide it

**Both parents vs the FIXED HEURISTIC BOTS, on a common untaught team set.** The 9 bots are
scripted, architecture-independent, and exist unchanged in both eras — they are the **only**
common reference constructible across an obs-dim change. Per-team WR vs a common bot panel puts
v8_04's and R2ACTION's untaught competence on one scale.

- **Design:** 2 arms × 16 untaught teams × 3 bots (Random / MaxDamage / Heuristic) × 100 games =
  **9,600 bridge battles**, CRN-paired on team + seed, greedy both sides, no server.
- **Price:** ~4–7 h on 2 cores at the rates probe Q measured (1.5–4.3 s/battle under load).
- **Prerequisite, and it is the real cost:** loading `ai_v8_04` needs an era-pinned worktree at
  `b13b30b289c5eaba136a930a4ab63451e209fbe5` (probe P's era pin) plus the **node** bridge, since
  the era predates the seedless-seed fix `bc00d4d`.
- **What it can and cannot say:** it settles whether v8's parent was absolutely more competent on
  untaught teams. It does **not** revive the substrate hypothesis on its own — §3 already shows
  the gift is competence-independent *within* each fold, so a level difference would have to work
  through some other mechanism, which would have to be named before the battles are worth paying
  for.

**Recommendation: do not run it yet.** The within-fold mechanism is dead in both eras; the
between-era level is downstream of a mechanism that does not exist.

---

## 5. The rival: archetype proximity to the taught set

The taught sets, derived (never hand-copied) from each fold's recorded `--trainee-teams`:

| fold | distinct taught | archetypes |
|---|---|---|
| v8 (`ai_v8_14`) | 22 | **10 semi_stall · 10 stall · 2 balance** |
| rev-3 fleet | 12 | 5 offense · 4 hyper_offense · 1 balance · 1 semi_stall · 1 stall |

Proximity is scored three ways per untaught team: share of the taught set with its archetype, and
mean / max Jaccard over the `gen3_team_archetypes.json` style tags.

| predictor of logit-gift | v8 (n=16) | rev-3 (n=8) |
|---|---|---|
| archetype share of taught | +0.137 [−0.369, +0.622] | +0.194 [−0.077, +0.639] |
| tag Jaccard, mean | −0.046 [−0.491, +0.477] | −0.281 [−0.896, +0.302] |
| tag Jaccard, max | +0.135 [−0.185, +0.474] | **−0.597** [−0.964, +0.084] |
| prior competence (logit) | −0.183 [−0.572, +0.266] | −0.462 [−0.890, −0.005] |

*(The competence row's point estimates are identical to §3's naive-logit values; its intervals
come from an independently-seeded bootstrap of the same statistic, hence the ±0.01 jitter. Both
are in the `.json`. This is the NAIVE row — the shared-noise-corrected one is §3's, and the rival
comparison is deliberately made against the naive value so the rival is judged against
competence at its most favourable.)*

Two-predictor OLS (competence + proximity, standardised) on v8 reaches **R² 0.036–0.067** — the
pair explains essentially none of the between-team variance in the gift.

**And the motivating observation does not survive the correct taught set.** Probe P noted
semi_stall — "the taught set's own archetype" — was the most positive untaught cut at **+8.33pp**.
It is, but the taught set is **10 semi_stall AND 10 stall**, and the untaught **stall** cut is the
**LEAST** positive of the five:

| untaught archetype cut (v8) | n | mean gift | mean parent WR |
|---|---|---|---|
| semi_stall | 3 | **+8.33pp** | 0.374 |
| hyper_offense | 3 | +5.97pp | 0.464 |
| offense | 3 | +5.14pp | 0.432 |
| balance | 4 | +4.48pp | 0.366 |
| **stall** | 3 | **+3.47pp** | 0.285 |

The two archetypes with **identical** representation in the taught set (10 and 10) sit at the two
ENDS of the untaught ranking. At rev-3 the sharpest proximity coefficient has the **wrong sign**
for the rival (closer to taught ⇒ *smaller* gift), and its 2-predictor R² of 0.747 on n=8 with two
predictors is overfit, not evidence.

**Verdict: the archetype-proximity rival is not supported either.** It priced out at roughly the
same nothing as prior competence.

---

## 6. Predictions scored

| # | registered | outcome |
|---|---|---|
| 1 | within v8, the headroom-controlled competence↔gift correlation is **POSITIVE but WEAK** | **NOT SUPPORTED.** Primary (logit) **+0.013 [−0.283, +0.323]** — a null, not a weak positive. The headroom columns *are* positive (+0.21…+0.40) and the tightest even excludes zero (+0.397 [+0.061, +0.660]) — **but every one sits far below its own null (+0.94…+1.00), so on that reading the data show LESS competence-dependence than no effect at all would produce.** The "wide interval" half of the prediction is confirmed. |
| 2 | the between-era comparison is **not cleanly decidable**; deliver a bound + a named measurement | **CONFIRMED**, and the reason is stronger than anticipated: the two eras' architectures differ, so **no learned common reference is constructible at all** — not merely unavailable. Bound and named measurement in §4. |

**Broken link named** (standing rule): prediction 1's broken link is
*prior competence → something for content to compose with*. It fails in **both** eras and at both
signs of aggregate effect, so it is not a rev-3-specific floor artifact.

---

## 7. Verdict

**Transfer does not require a substrate — at least not one that shows up as prior competence on
the untaught team.**

- The untaught gift is, to measurement precision, a **constant log-odds increment** across teams,
  independent of how well the parent already played them. Replicated in two eras with
  opposite-signed aggregate effects.
- Pooled over the 24 teams of the two clean folds: **r = −0.050 [−0.260, +0.160]**, P(r > 0) = 0.31.
- **What is excluded:** any moderate-or-stronger positive dependence — `r ≳ 0.32` on v8 alone,
  and `r ≳ 0.16` pooled (the pooled interval's upper bound; its lower bound is −0.26, so a
  *negative* dependence of moderate size is not excluded either).
- **What is NOT excluded:** a weak dependence; or a substrate that WR-against-one-reference-opponent
  does not proxy (e.g. representational coverage of a team's mons rather than outcome competence).
  Naming a better proxy is the only way this line reopens.
- **What that means for the v8-vs-gen reconciliation:** whatever made v8's fold gift +5.42pp to
  teams it never taught while rev-2's robbed −7.06pp, **it is not that v8 had a substrate to
  compose with.** The remaining live differences between the two folds are breadth
  (23 vs 9 distinct taught teams; 7.67 vs 2 per teacher) and lineage — and probe Q already showed
  breadth-per-teacher alone does not order the three points. The reconciliation is still open;
  this probe closes one of its candidate explanations.

### The durable methodological finding

**A ceiling control must be scored against the correlation it MANUFACTURES under a no-effect
null, never against zero.** Headroom normalisation at C = 0.60 returned +0.397 with a CI excluding
zero on a dataset whose true effect is nil, because that transform's own null on this
x-distribution is +0.944. The complement is equally sharp: the logit transform's null under a
headroom-additive world is −0.99. Reporting either against zero — the conventional move, and the
one the registered prediction assumed — produces a confident result of the analyst's choosing.

Same family as this tree's vacuous-guard taxonomy: *a number that cannot be told apart from a
measurement will eventually be read as one.* Here the number was a transform artifact wearing a
confidence interval.

Second, smaller: **a naive correlation between a rate and a delta built on that same rate reports
a spurious negative, and here it cleared the 95% bar** (rev-3, −0.462 [−0.889, −0.021] → −0.174
[−0.521, +0.130] after split-half). Any future "did X predict the gain?" analysis on this tree
needs the split-half instrument, and every probe artifact that ships per-battle or per-cell counts
(both of these do) makes it free.

---

## 8. Cuts and limits

- **No new battles.** The two cell-level artifacts already carry everything the within-fold
  question needs; the only gap (the between-era level) is blocked by an architecture change, not
  by battle budget, so buying battles would not have closed it.
- **n = 16 and n = 8 teams.** The cluster bootstrap is over teams because that is the unit the
  claim generalises over, and it produces honest ±0.30 / ±0.33 half-widths. This probe can
  exclude a strong effect and cannot resolve a weak one; §7 says so rather than reporting the
  point estimate as the answer.
- **rev-2's row is unidentified** (§3.3) and is published as such rather than dropped, so nobody
  re-derives it later from the same artifact.
- **v8's split is hypergeometric-within-cell**, not battle-index-aligned, because probe P's
  artifact stores counts and not per-battle vectors. The residual CRN leak is priced (~0.9% of
  signal variance) and a second, exactly-independent opponent-disjoint split agrees.
  *(Cheap fix for the future: probe artifacts should store per-battle win vectors, as probe Q's
  does — it costs kilobytes and it is what makes an exact split-half instrument available at all.)*
- **The taught sets are derived from recorded `--trainee-teams`** (v8 via probe P's own resolution
  at `/tmp/probeP_taught_paths.json`, rev-3 via the artifact's `taught_union`), never hand-copied.
  v8's 23 recorded paths resolve to **22 distinct** shas — two paths are the same team.
- **Archetype labels are the committed `gen3_team_archetypes.json`** (a pool-derived calibration
  artifact, not a prior the model reads).

## Reproduce

```
export PYTHONPATH=$PYTHONPATH:src
nice -n 15 python designs/research_state/measurements/substrate_hypothesis.py \
  --splits 400 --out designs/research_state/measurements/substrate_hypothesis_2026-08-31
```

~55 s, one core, `nice -n 15`, BLAS pinned to one thread, `models/` read-only (step counts only).
Seed `20260831`; 400 split replicates, 20,000 bootstrap replicates.
