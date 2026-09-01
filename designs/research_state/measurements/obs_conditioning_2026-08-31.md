# PROBE M8 — is the OBSERVATION badly conditioned? (2026-08-31)

**Question.** M2 (`representational_richness_transfer_2026-08-31.md` §3/§3.1) measured, model-free,
that the observation matrix's own participation ratio collapsed inside the gen lineage on a datable
schedule — **obs PR 37.76 at gen-12 → 22.88 at gen-13 → 16.20 at gen-14** — bracketing the H-B
event-window addition and the frame deletion, while a scale-invariant reading put the v8 era and
rev-1 at an identical 0.0463 effective directions per live dimension. M2's own recommended next
test was free and unrun: z-score the obs per column and re-read the PR, attribute the change to
blocks ADDED / REMOVED / a change in the SCALE of surviving columns, and establish whether the
trainer sees raw obs.

**Companion data:** `obs_conditioning_2026-08-31.json`.
**Scripts:** `obs_conditioning_probe.py` (PR × normalization × generation, per-block attribution)
and `obs_conditioning_idsplit.py` (the embedding-index vs scalar split), this directory.

---

## 0. Headline

**Both registered predictions score, and a third finding supersedes the question they were asked
about.**

1. **The gap M2 measured is not the observation's conditioning. It is the DEX NUMBERING.** In the
   current 2501-dim observation **433–437 of the ~2023 live columns are raw dex numbers** that the
   extractor casts with `.long()` and feeds to `nn.Embedding` — species num (1..386), move num,
   item num, type / ability / status / cant / faint-cause / item-transition ids. Those columns
   carry **99.993% of the raw observation variance**, and `PR(full obs) ≈ PR(ID columns only)` to
   two decimals in every generation measured. Every one of the top-20 variance columns, in every
   generation, is one of them.
2. **On the ~1,590 SCALAR columns — the ones that actually reach a weight matrix as magnitudes —
   there is no collapse and never was.** Scalar-only PR runs **45.14 (gen-12) → 44.62 (gen-14) →
   45.70 (rev-1 @24M)**, flat, with overlapping bootstrap spreads; per live dimension it *rises*
   (0.0236 → 0.0281 → 0.0286). And those columns are **already well conditioned**: at rev-1 @24M
   the max scalar column std is 0.985 against a median of 0.175 — **max/median = 5.6×** — with
   everything inside [0, 1] (or ±1 for the one signed flag) by construction.
3. **The trainer does see raw obs** (§4, cited) — there is no `VecNormalize`, no input norm, and
   SB3's `preprocess_obs` is a pass-through cast for our Box space. That fact is now established;
   it just has no defect attached to it.

**So: this is NOT a conditioning problem and NOT an information problem. It is a METRIC problem** —
`PR(covariance of the raw obs)` treats an embedding index as a magnitude, and on this observation
that one modelling error accounts for essentially all of the measured signal.

⚠️ **What this does NOT license.** It does not say the observation is beyond criticism, and it
does not touch the v8 gift. M2's own R5 already argued against the "more richness = more gift"
story from the other side (v8's *gifting* fold compressed `pi_features` by −4.60 against the gen
folds' −1.00). This probe removes the input-side leg of that account as well: there was no input
richness collapse to be the cause of anything. The v8 question is untouched and still open.

---

## 1. Method

**Read-only.** No battles, no training, no checkpoint loaded — every number is a statistic over
each generation's own `eval_traces/step_*/**/*_states.npz`. CPU, `nice -n 15`, 2 BLAS threads.

**Estimator, verbatim-compatible with M2.** `PR = (Σλ)² / Σλ²` over the column covariance,
computed as `(trace C)² / ‖C‖_F²` (exact, no eigendecomposition) and **asserted equal to the
project's canonical `agents.training.rank_metrics.effective_rank`** before use — agreement
**1.07e-14**.

**Sampler.** Files sorted, all rows pooled, then an evenly-strided n = 3000 subsample —
deterministic across processes, per the fixture rule. M2's sampler is not recoverable (its obs-PR
script was never committed), so the reproduction is stated as a tolerance rather than an identity:
this probe returns 36.22 / 38.39 / 23.39 / 16.11 / 15.45 / 17.88 / 17.11 against M2's 36.19 /
37.76 / 22.88 / 16.20 / 15.59 / 19.04 / 17.06 — max |Δ| **1.16**, against a measured
sampling spread of ±0.4–1.0 between sampler variants on the same trace directory. **The 2.4×
gen-12-to-gen-14 effect is 20× that spread**, so nothing here depends on which sampler is used.

**Era layouts.** gen-12 (2921) and gen-13 (3529) cannot be described by current code. Their
layouts and their embedded-ID manifests are dumped from git worktrees pinned to each run's own
`metadata.json` `git_hash` (`ede5a88` / `1fa4733`), so **every offset and every ID position is that
era's own declaration** — nothing is projected backwards and nothing is hardcoded.

**Uncertainty.** Cluster bootstrap over the source trace file (one file ≈ one battle), 400
resamples. ⚠️ **The estimator is biased DOWNWARD under resampling with replacement** — a duplicated
battle is a perfectly correlated row pair, which concentrates the covariance — so the point
estimate can sit above its own interval. **Read the WIDTH as the sampling scale, not the location
as a bound.**

---

## 2. The PR table — three normalizations × the generation ladder

`zscore` = per-column z-score using the same sample's mean/std, dead columns dropped.
`signedlog` = `sign(x)·log1p(|x|)`.

| generation | run | obs dim | live | **PR raw** | boot spread | **PR z-score** | boot spread | PR signed-log | PR_z / live |
|---|---|---|---|---|---|---|---|---|---|
| v8_04 (v8 era) | `ai_v8_04` @276M | 2992 | 2250 | **36.22** | [30.0, 36.3] | **105.89** | [54.9, 106.1] | 23.35 | 0.0471 |
| **gen-12** (frames LIVE) | `ai_v9_14` @8M | 2921 | 2177 | **38.39** | [34.3, 38.7] | **129.78** | [85.3, 125.3] | 26.68 | **0.0596** |
| **gen-13** (frames + events) | `ai_v9_15` @8M | 3529 | 2796 | **23.39** | [20.9, 24.7] | **134.28** | [83.5, 127.1] | 18.10 | 0.0480 |
| **gen-14** (frames DELETED) | `ai_v9_16` @8M | 2501 | 2023 | **16.11** | [14.6, 16.9] | **94.16** | [70.5, 87.4] | 12.19 | 0.0465 |
| gen-15 | `ai_v9_18` @8M | 2501 | 1988 | 15.45 | [14.3, 16.3] | 84.62 | [52.0, 84.0] | 10.98 | 0.0426 |
| gen-17 | `ai_v9_21` @8M | 2501 | 2024 | 17.88 | [16.2, 18.8] | 97.41 | [70.0, 89.1] | 12.47 | 0.0481 |
| rev-1 | `ai_v9_29` @8M | 2501 | 2007 | 17.11 | [15.8, 18.0] | 92.63 | [70.0, 87.5] | 11.71 | 0.0462 |
| rev-1 | `ai_v9_29` @24M | 2501 | 2036 | 17.19 | [15.6, 18.4] | 92.67 | [62.1, 88.7] | 12.40 | 0.0455 |
| **CURRENT** COMPFOLD | `ai_v9_91` @32M | 2501 | 1977 | 15.60 | [14.4, 16.5] | 72.29 | [60.4, 69.1] | 11.43 | 0.0366 |

**Two identities, asserted as VALUES rather than assumed** (both in the JSON, per generation):

- **z-score PR *is* correlation PR.** `|PR(z-scored) − PR(np.corrcoef)| ≤ 2.8e-14` everywhere. The
  mission listed these as two normalizations; they are one, and saying so is the honest reading —
  M2's "PR (correlation)" column and "z-score the obs" recommendation were the same test.
- **Dead columns do not change PR.** `|PR(all cols) − PR(live cols)| ≤ 1.1e-14`. A zero-variance
  column contributes a zero eigenvalue; it moves `live_dims` and nothing else.

**What z-scoring does and does not recover:**

| comparison | raw | z-score | z / live dim |
|---|---|---|---|
| v8_04 vs rev-1 @24M | 36.22 vs 17.19 (**2.11×**) | 105.89 vs 92.67 (1.14×) | 0.0471 vs 0.0455 (**1.04×**) |
| gen-12 vs gen-14 | 38.39 vs 16.11 (**2.38×**, −58%) | 129.78 vs 94.16 (1.38×, **−27%**) | 0.0596 vs 0.0465 (1.28×, **−22%**) |

So on the **cross-era** question z-scoring recovers parity outright, exactly as M2 predicted. On the
**within-lineage** collapse it removes roughly two thirds of the effect and leaves a ~22–27%
residual — and gen-12's 0.0596 sat *above* v8's 0.0471 in the first place, so "gen-12 matched v8"
was true on the variance-weighted reading and an *overshoot* on the scale-invariant one. §5 shows
that the surviving residual is also an ID-column artifact.

---

## 3. Per-block attribution — and the control that decides it

Blocks come from each era's own `get_layout()`; the spans tile the vector exactly
(`uncovered = 0` in every row). `var_share` = the block's share of the total column variance;
`PR block` = the block's own PR in isolation; `ΔPR if removed` = PR of everything else, minus the
full PR.

### gen-12 (2921 dims, PR 38.39)

| block | dims | live | var share | PR block | PR block (z) | **ΔPR if removed** |
|---|---|---|---|---|---|---|
| our_team | 732 | 527 | 19.84% | 14.38 | 41.95 | −9.25 |
| opp_team | 732 | 563 | 18.27% | 9.94 | 34.38 | −2.30 |
| active_context | 116 | 24 | 0.00% | 3.87 | 15.70 | −0.00 |
| global_env | 20 | 9 | 0.00% | 1.46 | 3.75 | −0.00 |
| board_reactive | 17 | 17 | 3.07% | 3.77 | 7.19 | −1.75 |
| pair_history | 180 | 180 | 0.00% | 26.81 | 36.96 | −0.00 |
| prev_action_mask | 11 | 11 | 0.00% | 6.06 | 7.98 | −0.00 |
| **turn_frames** | 1113 | 846 | **58.82%** | 23.17 | **198.90** | **−14.76** |

### gen-13 (3529 dims, PR 23.39) — the event window arrives

| block | dims | live | var share | PR block | PR block (z) | **ΔPR if removed** |
|---|---|---|---|---|---|---|
| our_team | 732 | 533 | 11.18% | 15.38 | 42.35 | −4.31 |
| opp_team | 732 | 567 | 10.22% | 10.45 | 35.10 | −1.81 |
| board_reactive | 17 | 17 | 1.66% | 3.82 | 8.07 | −0.69 |
| pair_history | 180 | 180 | 0.00% | 26.81 | 35.65 | −0.00 |
| **event_window** | 608 | 605 | **45.52%** | **9.09** | 44.20 | **+17.41** |
| turn_frames | 1113 | 850 | 31.42% | 23.03 | 200.60 | −6.97 |
| *(context / global / prev_mask)* | 147 | 44 | 0.00% | — | — | −0.00 |

### gen-14 (2501 dims, PR 16.11) — the frames leave

| block | dims | live | var share | PR block | PR block (z) | **ΔPR if removed** |
|---|---|---|---|---|---|---|
| our_team | 732 | 536 | 16.36% | 14.09 | 40.23 | −4.29 |
| opp_team | 732 | 558 | 13.84% | 9.88 | 38.46 | −1.30 |
| board_reactive | 17 | 17 | 2.54% | 3.75 | 7.30 | −0.73 |
| pair_history | 180 | 180 | 0.00% | 32.84 | 45.21 | −0.00 |
| **event_window** | 704 | 696 | **67.26%** | **9.54** | 58.14 | **+7.89** |
| *(context / global)* | 136 | 36 | 0.00% | — | — | −0.00 |

### CURRENT (rev-1 @24M, 2501 dims, PR 17.19)

| block | dims | live | var share | PR block | **ΔPR if removed** |
|---|---|---|---|---|---|
| our_team | 732 | 528 | 16.70% | 14.41 | −4.59 |
| opp_team | 732 | 572 | 14.89% | 10.46 | −1.52 |
| board_reactive | 17 | 17 | 2.53% | 3.73 | −0.77 |
| pair_history | 180 | 180 | 0.00% | 30.60 | −0.00 |
| **event_window** | 704 | 703 | **65.89%** | 10.02 | **+7.39** |

**Three blocks are ≥97% of the raw variance in every generation** — `{event_window, our_team,
opp_team}` today (98.5%), `{turn_frames, our_team, opp_team}` at gen-12 (96.9%). `pair_history`,
`active_context` and `global_env` are jointly **0.00%**: they are the well-behaved [0,1] blocks,
and the variance-weighted PR is structurally blind to them.

### 3.1 The COMMON-BLOCK control — the test that separates the two hypotheses

PR restricted to the six blocks *every* generation has (our_team, opp_team, active_context,
global_env, board_reactive, pair_history = 1797 dims):

| generation | common-6 PR (raw) | common-6 PR (z) | full-obs PR (raw) |
|---|---|---|---|
| gen-12 | **23.63** | 64.68 | 38.39 |
| gen-13 | **25.10** | 65.99 | 23.39 |
| gen-14 | **24.00** | 72.42 | 16.11 |
| gen-15 | 24.93 | 65.13 | 15.45 |
| gen-17 | 25.93 | 74.40 | 17.88 |
| rev-1 @8M | 24.91 | 70.24 | 17.11 |
| rev-1 @24M | 24.59 | 69.29 | 17.19 |
| CURRENT COMPFOLD @32M | *16.20* | *47.03* | 15.60 |

**The surviving columns did not change.** Common-6 PR is flat at 23.6 → 25.9 from gen-12 through
rev-1 while the full-obs PR falls by 2.4×. So the attribution is unambiguous:

- **gen-12 → gen-13 (−15.0)** is the event window being **ADDED**: it lands at 45.5% of the
  variance with a block PR of **9.09**, and removing it from gen-13 gives **40.80** — i.e. gen-13
  *without* the event window is gen-12.
- **gen-13 → gen-14 (−7.3)** is the frames being **REMOVED**: removing them from gen-13 gives
  **16.42**, which is gen-14's 16.11. The frames were a 1113-dim, PR-198.9-under-z block; deleting
  them left the low-PR event window as a larger share of what remained.
- **Scale of surviving columns: no change.**

*(The COMPFOLD row is italicised because it is a different question. Its eval traces cover **12**
distinct our-team species sets against 81–100 for the generation runs — it is a distillation fold
on a pinned team set. Its low common-6 PR and low `our_team` block PR are a **state-distribution**
effect, and it is a live demonstration of the confound M2 flagged in its own §3.1.)*

---

## 4. Does the trainer see raw obs? — YES, established with citations

| link in the chain | verdict | citation |
|---|---|---|
| observation encoder output | raw `float32`, block-assembled, no global scaling or clipping | `src/agents/observation/state_encoder.py:157-434` (`encode`) |
| `Gen3Env.embed_battle` | returns `encode(...)` unchanged — `gen3_frame_deletion_v1` made the encoder's output *be* the observation | `src/agents/training/gen3_env.py:340-392` |
| observation space | `Box(-inf, +inf, (2501,), float32)` generated from the schema | `src/agents/training/gen3_env.py:112` (`build_schema(...).gym_space()`), `:219-221` |
| env wrapper chain | `Gen3Env` → `MaskableAgentWrapper` → `Monitor`. Nothing else. | `src/main/train/env_factory.py:226`, `:252` |
| vec env | `env = EnvClass(env_factories)` where `EnvClass ∈ {SubprocVecEnv, AsyncSubprocVecEnv, DummyVecEnv}`. **No wrapper is applied to the result.** | `src/main/train_rl_agent.py:236-243`, `:341` |
| `VecNormalize` anywhere in the tree | **ZERO occurrences** under `src/` | `grep -rn "VecNormalize\|obs_rms\|RunningMeanStd" src/` → no matches |
| SB3 policy preprocessing | `preprocess_obs` on a non-image `Box` returns `obs.float()` — a pass-through cast. `is_image_space(Box(-inf,inf,(2501,),f32))` is **False** (verified by running it). | `site-packages/stable_baselines3/common/preprocessing.py:117-120` |
| first module to touch the raw vector | `ObsUnpack.forward` — pure slicing/reshape, no normalization | `src/agents/model/extractor_ctx.py:264-329`, `:359-372` |
| first Linear on the per-mon scalar path | `role_encoder[0]`, with **no** LayerNorm/BatchNorm before it | `src/agents/model/encoders.py:149-150` |

**The network consumes the encoder's bytes.** The conditioning question is therefore a real one to
have asked; §5 is why the answer is "there is nothing wrong here".

**And the per-column scaling is deliberate and already done, at encode time.** Base stats `/255`
(`src/agents/observation/species.py:59-64`), IVs `/31` and EVs `/252`, boosts `/6`
(`active_context.py:47-48`), spikes `/MAX_SPIKES` and all three clock scalars `/log(1+MAX_TURNS)`
(`global_env.py:62-85`), sleep/toxic counters `/4` and `/8`, pair-history cells log-saturated to
[0,1]. *(A subagent's first pass reported base stats as "RAW [4–183]"; that is wrong — they are
`/255` at `species.py:59`. Corrected here because the whole diagnosis would have inverted on it.)*

---

## 5. 🔴 THE DECIDING MEASUREMENT — embedding indices vs scalars

A covariance PR treats every column as a magnitude. This observation is not all magnitudes. A
declared set of positions carries **raw dex numbers** that the extractor casts with `.long()` and
hands to `nn.Embedding`; no Linear ever reads them as numbers — `slice_pokemon_categoricals` even
**zeroes** the raw last-action column after extracting it, under the comment *"a raw dex num must
never reach a Linear; the manifest rule"* (`src/agents/model/extractor_ctx.py:101-107`; the same rule is restated at `:141`), and
`ARCHITECTURE.md` §1.1 says the same of the whole event window (*"Ids are embedding ids; no Linear
reads the block raw"*) and of the board (*"the embedding-ID block is excluded from
`non_matchup_rest` by construction"*).

The ID set is read from each era's own declarations — the `pokemon` sub-layout fields
`slice_pokemon_categoricals` reads, `reactive_layout['active_req_moves']`, the `EventCol` members
`EventSeats.forward` casts, and `turn_delta_encoder.TURN_DELTA_EMBEDDED_IDS`.

| generation | live ID cols | live scalar cols | **ID var share** | scalar var share | **PR full** | **PR IDs only** | **PR scalars only** | boot spread | PR scalars / live | top-20 that are IDs |
|---|---|---|---|---|---|---|---|---|---|---|
| gen-12 | 266 | 1911 | **0.99993** | 6.9e-05 | 38.39 | 38.39 | **45.14** | [34.1, 47.3] | 0.0236 | **20 / 20** |
| gen-13 | 426 | 2370 | **0.99994** | 5.8e-05 | 23.39 | 23.39 | **58.98** | [46.6, 58.3] | 0.0249 | **20 / 20** |
| gen-14 | 433 | 1590 | **0.99994** | 6.5e-05 | 16.11 | 16.11 | **44.62** | [38.4, 44.5] | 0.0281 | **20 / 20** |
| gen-15 | 437 | 1551 | 0.99993 | 6.6e-05 | 15.45 | 15.45 | 39.54 | [29.4, 41.6] | 0.0255 | 20 / 20 |
| gen-17 | 438 | 1586 | 0.99993 | 6.8e-05 | 17.88 | 17.88 | 47.82 | [40.8, 47.0] | 0.0302 | 20 / 20 |
| rev-1 @8M | 436 | 1571 | 0.99993 | 6.7e-05 | 17.11 | 17.11 | 44.50 | [38.9, 44.8] | 0.0283 | 20 / 20 |
| rev-1 @24M | 437 | 1599 | 0.99993 | 6.9e-05 | 17.19 | 17.19 | **45.70** | [39.0, 45.4] | 0.0286 | 20 / 20 |
| CURRENT COMPFOLD | 433 | 1544 | 0.99994 | 6.1e-05 | 15.60 | 15.59 | *35.89* | [32.8, 36.3] | *0.0232* | 20 / 20 |

`PR full ≈ PR IDs only` to two decimals in every row: **the observation's covariance participation
ratio is a statistic about 433 dex-number columns and nothing else.**

**Which ID family** (share of total raw variance, current obs): species **62.4%** · move **33.5%** ·
item 3.8% · ability 0.25% · type 0.08% · everything else ≤0.01%. Species dex numbers span 1–386 and
move nums 1–370, so a std near 140 is simply the spread of the numbering.

**The top-20 variance columns are 100% ID columns, every generation.** In the current obs they are
`opp_team[k].species.species_id` (std ≈ 144) and, dominating by count, **event-window column 3 =
`TARGET_SPECIES` across the 32 token slots** (std ≈ 138 each) — 22 columns per token, of which
these are the ones the PR sees.

### 5.1 Are the SCALAR columns badly conditioned? No.

| generation | scalar std max | p99 | median | **max / median** | top scalar columns |
|---|---|---|---|---|---|
| gen-12 | 0.928 | 0.749 | 0.129 | **7.2×** | `our_team[k].moves[·]` |
| gen-14 | 0.985 | 0.909 | 0.158 | **6.2×** | `event_window[·]` (col 2 = `ACTOR_SIDE`, ±1) |
| rev-1 @24M | 0.985 | 0.908 | 0.175 | **5.6×** | same |
| CURRENT COMPFOLD | 0.990 | 0.891 | 0.163 | 6.1× | same |

Every scalar column sits inside [0, 1] (or [−1, +1] for the signed side flag) and the whole dynamic
range is under 6×. **There is no per-column scale pathology in the input the network reads as
numbers, and there is nothing for a normalizer to normalize.** For contrast, the ID columns'
max/median std ratio is **635–882×** — and that number is a fact about Showdown's dex, not about
the state.

---

## 6. Predictions, scored

| # | prediction | verdict |
|---|---|---|
| **1** | Z-scoring RECOVERS era parity — the gap is per-column scale, not lost information | **PASS on the cross-era comparison, PARTIAL within the lineage, and SUPERSEDED as a framing.** v8 vs rev-1: PR_z/live 0.0471 vs 0.0455 (1.04×) against a raw 2.11× — parity recovered. gen-12 vs gen-14: raw −58% becomes z −27% (per live dim −22%) — two thirds recovered, a real residual left. But §5 shows both readings are ≈100% determined by 433 embedding-index columns, so "per-column scale" is the right answer for the wrong reason: the scale in question is the DEX NUMBERING, and on the columns that carry magnitude there was never a gap (45.14 → 44.62 → 45.70, flat). |
| **2** | ≤3 blocks dominate the raw variance in the CURRENT obs, and ≥1 was added after gen-12 | **PASS, cleanly.** `event_window` 65.9% + `our_team` 16.7% + `opp_team` 14.9% = **97.5%**, and `event_window` landed at gen-13. Same shape at gen-12 with `turn_frames` (58.8%) in the event window's place: 96.9% in three blocks. |

---

## 7. What would REFUTE this probe's conclusion, and what did

| # | refuter | fired? |
|---|---|---|
| R1 | z-scoring leaves the full collapse intact ⇒ a structural change, not scale | **partially** — 22–27% survives z-scoring; §5 then attributes that residual to the ID columns too |
| R2 | the common-6 blocks' PR moves across generations ⇒ surviving columns changed scale | **no** — flat 23.6 → 25.9 while full PR fell 2.4× |
| R3 | the scalar-only PR shows the same collapse ⇒ a real conditioning defect | **no** — 45.14 / 44.62 / 45.70, overlapping spreads, per-live-dim *rising* |
| R4 | the scalar columns span a large dynamic range ⇒ a real optimization liability | **no** — max/median std 5.6×, all in [0,1] |
| R5 | the trainer normalizes the obs ⇒ the question was moot | **no** — it does not (§4), so the question was legitimate; the answer is just "nothing wrong" |

---

## 8. The fix menu, priced — and the recommendation is DO NOTHING TO THE OBSERVATION

Ordered by cost. Nothing here is implemented; every one is an owner decision.

| # | candidate | cost | arch-breaking? | verdict |
|---|---|---|---|---|
| **1** | **Fix the METRIC, not the obs** — compute obs PR on scalar columns only (and report ID-column diversity separately, if wanted). The ID manifest is already layout-declared, so this is the `obs_conditioning_idsplit.py` code path, ~2 min per generation. | **zero** | **no** | ✅ **RECOMMENDED.** It is the only change this evidence licenses. |
| 2 | **Per-column z-score / `VecNormalize`** over the whole vector | one flag | **yes** (retrain-class) | ❌ **REJECT — it is a CORRECTNESS BUG, not a trade-off.** 433 live columns are `.long()`-cast embedding indices; z-scoring maps them to signed floats near 0, `.long()` truncates, `.clamp(min=0)` collapses most of them to index 0, and the model silently loses species/move/item identity. The measurement above is exactly what makes this look attractive and exactly why it must not be done. |
| 3 | **MASKED normalization** — z-score the 1,590 scalar columns only, leave IDs alone | one flag + the manifest (which exists) | **yes** — changes the input distribution the weights were fit to ⇒ fresh weights | ❌ **NOT WARRANTED.** Technically clean, but §5.1 says the scalar columns are already in [0,1] with a 5.6× range. There is no defect to fix, and the price is a generation. |
| 4 | **Input LayerNorm on the scalar slices** before the first Linear | new parameters | **yes** — `ARCH_SIGNATURE` bump, and LayerNorm is *not* identity at init, so there is no byte-identical-OFF default | ❌ **NOT WARRANTED** on this evidence. Same reason as 3, at higher cost. |
| 5 | **RUNNING observation normalization** (`VecNormalize`-style, masked or not) | one flag | **yes, and worse** | ❌ **REJECT on a second, structural ground.** It makes the observation a function of run history rather than of the event log. This tree's forensic stack asserts the opposite as an invariant — `obs_roundtrip_fuzz_test` (offline obs == live obs, bit-for-bit), `reroll_many_parity_fuzz_test`, `search_clone_parity_fuzz_test`, and the ai_v9 roadmap's stated reason for ruling out recurrence (*"the obs must stay a pure function of the event log for the forensic stack"*). A running normalizer breaks the prober, the materializer, and every replay/clone parity gate at once. |
| 6 | **Rescale the offending ID columns** (e.g. dex num / 400) | one line | — | ❌ **REJECT.** They are `.long()`-cast immediately; dividing makes every index 0. Same failure as 2. |

**M2 §12's step 2 is now KNOWN-BAD and should not be run as written.** It proposed forwarding rev-1
with the obs z-scored per column and re-reading `projection` / `pi_features` PR. That forward would
feed z-scored dex numbers into `.long()` → `clamp(min=0)` → mostly index 0, i.e. it would measure a
model that has lost species, move and item identity, and the resulting low PR would look like a
confirmation. **A masked version (scalar columns only) is runnable and is what step 2 should have
been — but §5.1 has already answered the question it was asking**, so it would cost a forward pass
to learn nothing. Step 1 (per-block contribution) is done: §3.

**The one genuinely open question this probe surfaces is a DESIGN question, not a conditioning
one**, and it is stated here only so it is not lost: the event window spends 7 of its 22 columns per
token on embedding ids, and repeats `TARGET_SPECIES` / `ACTOR_SPECIES` / `MOVE` across 32 slots that
draw from ~12 species. That is a fact about token layout and embedding-table reuse, has nothing to
do with input scale, and would need its own measurement (an `EventSeats` ablation, not a PR) before
anyone should touch it.

---

## 9. Limits, stated plainly

- **This says nothing about the v8 gift.** It removes one candidate explanation (an input richness
  collapse) by showing the measured collapse was a metric artifact. M2's R5 already pointed the
  other way independently. The gift is untouched and still unexplained.
- **The generation ladder confounds architecture with state distribution** — six runs, six
  policies, six different team/opponent mixes. M2 flagged this; this probe **measures** it (§3.1's
  COMPFOLD row: 12 distinct team sets vs 81–100) rather than only warning about it. The
  **common-block control and the ID/scalar split are the two readings that survive the confound**,
  because both compare a *within-vector* decomposition, not a level.
- **The bootstrap is downward-biased** (§1). Widths are usable; locations are not bounds.
- **M2's exact sampler is unrecoverable** — its obs-PR script was never committed. Reproduction is
  a tolerance (max |Δ| 1.16), not an identity.
- **v8 has no block or ID/scalar row.** Its layout was not reconstructed (its obs family differs
  and no era worktree was built for it here); the v8 row in §2 is the headline PR only.
- **"Scalar" means "not routed to an embedding table".** A scalar column can still be a one-hot or
  a flag; the split is by *consumption*, not by information type.
- **PR is one statistic.** A flat scalar-only PR does not certify the observation as adequate — it
  says the *variance is not concentrated*, which is the specific claim M2's number was read as
  making. Coverage, resolution and encoding quality are separate questions with separate
  instruments (the frame-deletion coverage audit is the standing example of why).

---

## Provenance

| claim | source |
|---|---|
| every PR, per generation, per normalization | `obs_conditioning_probe.py` over each run's own `eval_traces/step_*/**/*_states.npz`; estimator asserted equal to `agents.training.rank_metrics.effective_rank` at 1.07e-14 |
| the M2 numbers being reproduced | `representational_richness_transfer_2026-08-31.json` §3.1 |
| per-block spans, gen-12 / gen-13 | `get_layout()` inside git worktrees at `ede5a88` / `1fa4733` (each run's own `metadata.json` `git_hash`) |
| per-block spans, gen-14 onward | live `Gen3ObservationEncoder(load_mappings()).get_layout()` |
| the embedded-ID column set | each era's own `slice_pokemon_categoricals` fields, `reactive_layout['active_req_moves']`, `EventCol` (gen-13: its own documented literal block, asserted against `EVENT_TOKEN_DIM == 19`), `TURN_DELTA_EMBEDDED_IDS` |
| the trainer-sees-raw chain | the file:line table in §4; `is_image_space` verdict executed, not assumed |
| team diversity of each state set | distinct sorted our-team species-id 6-tuples in the n=3000 sample |
| every raw number | the sibling `.json` |
