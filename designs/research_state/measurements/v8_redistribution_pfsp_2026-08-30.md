# PROBE P — did v8's fold ALSO redistribute, and was team-PFSP the treadmill-killer?

**Date:** 2026-08-30 · **Registered:** ledger `8bab2f2` (predictions P1/P2/P3 fixed before any data)
· **Era-pinned** to v8_14's commit `b13b30b289c5eaba136a930a4ab63451e209fbe5`.

Companion data: `v8_redistribution_pfsp_2026-08-30.json`.

---

## 0. The headline, before the detail

The probe was dispatched to test a reconciliation candidate: *our* folds concentrate a FIXED
`--distill-team-bias 0.4` on the taught slices while *v8*'s fold ran ADAPTIVE one-sided
team-PFSP, so v8's ecology was the anti-treadmill and the fixed bias is the unreplicated
ingredient.

**The premise is false on its own terms, and the archaeology killed it before any battle was
played.** v8_14's fold ran `--distill-team-bias 0.4` — *the same fixed bias, at the same value*
— **and** `--team-pfsp onesided` on top. The two are not alternatives: in the era's teambuilder
the bias branch is tested first and short-circuits, so the episode budget splits **40% pinned
uniformly over the 22 taught teams (untracked, PFSP-free) / 60% PFSP-weighted over the pool**.
The contrast with rev-2/rev-3 is therefore not "fixed bias vs adaptive PFSP" but "fixed bias
alone vs fixed bias plus a PFSP layer" — and that PFSP layer measures as **near-inert**.

---

## 1. What v8 actually ran (verified from metadata + era code)

| run | role | `--distill-team-bias` | `--team-pfsp` | teachers | distinct taught teams |
|---|---|---|---|---|---|
| `ai_v8_04_distill_4teacher_0722` | the PARENT (fork source) | **0.4** | `onesided` | 4 | 4 |
| `ai_v8_14_distill3_0725` | **the FOLD** | **0.4** | `onesided` | 3 | **22** (23 paths, 2 identical) |
| `ai_v9_59_R2ACTION_0827` | rev-2 fold | **0.4** | `off` | 5 | 9 |
| `ai_v9_70_R3ACTION_0828` | rev-3 fold | **0.4** | `off` | 6 | 12 |

v8_14's `--model` names `ai_v8_04.../final_model_interrupted.zip`, confirming the parent.
Taught teams resolve by `sha1(team_str.strip())[:10]`; **all 22 are members of the 719-team
pool**, so "untaught" = the remaining 697.

Lineage: `v8_01` (root) → `v8_03` (from `v8_01`'s *init*) → `v8_04` → `v8_14`; `v8_02` is an
off-lineage sibling of `v8_03`.

---

## 2. P2 — the adaptive-repair signature: **NOT SUPPORTED** (mechanism refuted, not merely unobserved)

### 2.1 The sampler cannot express the prediction

`src/agents/training/team_pfsp_callback.py` @ `b13b30b` (byte-identical at the parent's commit
`ce8bbc9`):

```python
raw = [floor + (0.25 if (onesided and p < 0.5) else p * (1.0 - p)) for p in emas]
mean = sum(raw) / n
cap_val = cap * mean
return [min(r, cap_val) for r in raw]
```

- **No trend input exists.** Only the level EMA enters the weight. "Draw mass concentrating on
  *weakening* teams" is not a representable behaviour of this function.
- Under `onesided` with `floor=0.05`, **every team at or below 50% WR gets the identical
  weight 0.30** — a flat maximum across the whole losing half. A 5%-WR team and a 49%-WR team
  are indistinguishable to the sampler.
- The `cap=3.0` is **inert**: `raw ∈ [0.05, 0.30]` ⇒ `cap·mean ≈ 0.83`, never binding.

### 2.2 Why it was near-inert *in practice* — the durable lesson

The theoretical steering range is 6× (0.30/0.05), but `p(1-p)` is **maximally flat exactly
where self-play win rates live**. In self-play against a pool of your own snapshots per-team WR
sits near 0.5, so nearly every team scored ≈0.28-0.30 and the realized weight vector was almost
uniform:

- TB `team_pfsp/weight_spread` (= `max(w)/mean(w)`): **v8_14 max 1.088**, v8_04 max 1.052.
- Reconstructed weight vectors: **Gini(w) ≤ 0.061**, total-variation distance from uniform
  **≤ 0.049** — i.e. **at most ~5% of the draw mass was moved relative to uniform**, over the
  whole fold.
- The little steering that happened points the *other* way from "repair": `min/mean` bottoms at
  0.40 while `max/mean` tops at 1.088, i.e. it **retired mastered teams** rather than boosting
  weak ones. Weakest-decile share rose 0.0989 → 0.1063 (+7.5% rel.); strongest-decile fell to
  0.0725 (−27%).

**A variance-shaped sampler is a no-op on a distribution that is already at its variance
maximum.** That is the transferable finding: `p(1-p)` weighting buys steering only when win
rates are *spread*, and self-play by construction keeps them near 0.5.

### 2.3 The empirical tests

Bootstrap over teams, 4000 resamples. `ai_v8_15_retention_A_frozen_0726` ran `--team-pfsp
measure` (weights computed, never pushed) and is the natural no-steering control.

| run | r(draws, WR) | r(draws, recent slope) | partial r(draws, slope \| WR) |
|---|---|---|---|
| v8_14 (onesided) | −0.026 [−0.190, +0.134] | −0.092 [−0.258, +0.069] | −0.090 [−0.250, +0.074] |
| v8_04 (onesided) | +0.004 [−0.086, +0.093] | +0.058 [−0.028, +0.141] | +0.062 [−0.024, +0.147] |
| v8_15 (**measure** ctrl) | +0.070 [−0.039, +0.169] | +0.095 [+0.025, +0.182] | +0.069 [+0.003, +0.142] |

**Every CI in both PFSP runs spans zero**, and the trend correlation — the adaptive-repair claim
proper — is the weaker of the two, with the wrong sign in v8_04.

Draw-mass concentration is likewise **not** distinguishable from uniform blocked sampling: with
`--team-block-episodes 64`, v8_14's Gini 0.444 sits at the blk≈8 null (0.434) and v8_04's 0.499
between the blk16/blk32 nulls — while the *asymptotic* concentration the weight vector could
ever produce is Gini 0.03-0.06, an order of magnitude smaller. The observed Gini is counting
noise.

"Did it repair?" is **MISSING for v8_14** (its committed snapshot covers a single update
window), null for v8_04 (+0.004 [−0.014, +0.021]), and *larger in the no-steering control*
(+0.018 [+0.008, +0.027]) — which is dispositive that the statistic measures an
exposure/episode-length confound, not repair.

### 2.4 Confounds named

- `games` counts **episodes** while the budget is **timesteps**: teams the agent wins with
  finish faster and accrue more episodes, biasing draw mass toward *strong* teams independent
  of any sampler (visible as the control's +0.146 Spearman).
- `_cum_games` and the EMA both **reset at every launcher restart**; the committed snapshots
  cover 1 / 4 / 27 update rows for v8_14 / v8_04 / v8_15, so cross-run count statistics are
  partly window-length comparisons.
- Per-team WR is measured against a **moving self-play opponent**, so a rising WR can be the
  opponent weakening.
- The 22 taught teams are a **non-random subset** (selected by prior exploiter runs) and rise
  faster than the pool even in the no-distill control.

### 2.5 MISSING

- `team_winrates_history.jsonl` records only `{step, wr}` — **no per-row `games`**. Per-team
  draw mass over time does not exist in any artifact; only the final overwritten snapshot has
  it, and only for the post-restart window. *(Cheap fix for the future: add a `games` vector to
  `_persist_snapshot`.)*
- **No per-team TensorBoard series exist** (v8_14 has 243 scalar tags; the only team-PFSP ones
  are the four aggregates `team_pfsp/{min_wr,max_wr,n_measured,weight_spread}`).
- The 40% pinned-teacher episodes are **structurally invisible** to `games` (bias draws set
  `_last_pool_idx = None` and are never counted). The largest single concentration of draw mass
  in the fold is a declared constant, not an adaptive response.

---

## 3. P1 — did v8's fold regress on UNTAUGHT pool teams? **NO — it GIFTED them.**

**352/352 cells, 10,560 battles, zero errors, zero short cells.** (Scored by the parent
session: the shard fleet completed 21:57 PDT 08-29 and the agent's completion watcher never
fired — the data sat finished ~10 h. Second dead watcher of the night; same repair as
iter-3.)

| slice | teams | battles | parent WR | fold WR | Δ (fold−parent) | cluster-boot 95% CI | pooled z |
|---|---|---|---|---|---|---|---|
| **UNTAUGHT** | 16 | 7,680 | 0.3828 | 0.4370 | **+5.42pp** | **[+3.44, +7.42]** | **+4.83** |
| TAUGHT | 6 | 2,880 | 0.4306 | 0.6924 | **+26.18pp** | [+20.28, +32.85] | +14.68 |

**The registered bar (frozen table `91d5125`, row "P-final v8-untaught ≥ 0") PASSES
decisively.** v8's fold is the sign-flip of rev-2's −5.9/−7.1pp robbery: a genuine positive
externality on teams the fold never trained on. Per-team: 14/16 untaught positive (the two
negatives are −2.1 and −1.7pp — noise-scale); every archetype cut positive, and
**semi_stall — the taught set's own archetype, where redistribution should bite hardest — is
the MOST positive untaught cut (+8.33pp)**. The equal-weight per-team mean equals the pooled
delta to the basis point, so no team-size weighting artifact is present.

**The floor confound (probe Q's) does NOT apply here in reverse:** the parent's untaught WR
is 0.383 against the fixed reference opponent — far from any ceiling or floor — so there was
room to move in both directions and the gift is a measurement, not an artifact of exhausted
headroom. Transfer is still mostly local (taught +26.2 vs untaught +5.4, a ~4.8:1 ratio) —
breadth did not abolish locality, it changed the SIGN of what leaks out.

---

## 4. P3 — pull-down vs coverage (EXPLORATORY, never a verdict)

The registered x-axis (share-taken-from-untaught) was shown DEGENERATE by probe Q
(`rev3_untaught_pulldown_2026-08-30.md` §P3): all three folds ran bias 0.4 with K≪719, so
the share spans 0.8pp against ~18pp of y. On the axis that does vary — distinct taught teams
per teacher (2 / 2 / 7.33) — the ordering is consistent (rev-2 −7.1 · rev-3 −0.75 · v8
**+5.4**) but with only three points, two of which disagree at equal breadth, this section
remains exploratory. The cross-probe row for v8 is now the MEASURED +5.42 [+3.44, +7.42]
(this file), superseding the +10.4 interim probe Q cited.

---

## 5. Method (P1)

- **Arms:** `parent` = `ai_v8_04.../final_model_interrupted.zip` (the exact fork source);
  `fold` = `ai_v8_14.../final_model_interrupted.zip`.
- **Fixed reference opponent:** `ai_v8_03.../final_model_interrupted.zip` — an ancestor of
  *both* arms and equal to neither. Using the parent as opponent would make the parent arm a
  self-mirror, biasing the delta toward the fold (i.e. toward the "no regression" reading the
  probe was registered to test).
- **Design:** 16 untaught probe teams (archetype-stratified, deliberately over-sampling
  stall/semi_stall so the "same archetype as the taught set" cut is powered) + 6 taught controls
  (2 per teacher cluster) × 8 fixed opponent teams × 30 games × 2 arms = **10,560 battles**.
  Both arms play the **same (team, opponent) cells with the same per-battle sim seeds** (common
  random numbers), so the comparison is paired at the battle level.
- Greedy on both sides (`stochastic=False`), matching the era's in-run `vs_ext` eval regime.
- **Node bridge, no server, CPU.** The era's rust bridge predates the seedless-seed fix
  (`bc00d4d`, 2026-08-03) and would have replayed a single dice stream — node is mandatory at
  this commit.
- **Era-code load validated** by reproducing recorded trace logits: the fold's checkpoint
  reproduces the fold's own `step_292000005` traces at centred-logit r = **0.982**, argmax
  agreement **0.864**, V r = **0.992**; the parent's checkpoint scores **0.844 / 0.610 / 0.937**
  on the same traces. The load discriminates correctly.
- Statistics: battle-level z for the pooled contrast, plus an **8,000-resample cluster bootstrap
  over teams** (the unit the claim generalises over) and an equal-weight per-team mean.
