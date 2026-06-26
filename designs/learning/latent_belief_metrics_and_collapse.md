# Reading the Latent-Belief SimSiam Metrics (and What Collapse Looks Like)

> **What this is.** A durable explainer for how to read the `belief/latent_*` metrics our
> latent-belief head emits — specifically why `latent_cosine_above_chance` rises *continuously*
> while `latent_std` traces an *inverted-U*, and how to tell a healthy run from a collapsing one.
> Self-supervised (SimSiam-style) representation learning has a famous failure mode — **collapse**
> — and these two metrics are the two complementary detectors for it. Written to teach: intuitive
> first, then technical, no code. Grounded in the v18 latent head (`--opp-belief-latent-coef`) and
> the live `ai_v6_13_outgoing_dmg_0620` run.

---

## TL;DR

- The **latent-belief head** (v18) is a SimSiam-style self-supervised predictor: for each hidden
  opponent slot it predicts a vector and is trained to make that vector **cosine-match the
  stop-grad `pokemon_encoder` role-token of the *true* hidden mon**. This grades identity in
  role-space — the soft "Suicune ≈ other bulky waters" signal the discrete species cross-entropy
  can't give.
- Self-supervised cosine training has a **cheat**: output the *same* vector for everything →
  perfect cosine, zero information. That's **collapse**. We run two guards against it.
- **`latent_std`** (per-dim spread of the predictions) is the **dimensional-collapse** detector —
  it must not go to 0. A **VICReg variance floor** (`relu(1.0 − std)`) actively pushes it up off 0.
- **`latent_cosine_above_chance` = matched − baseline** is the **discriminability** detector. The
  role-token manifold is *non-orthogonal* (two random mons already sit at cosine ~0.5–0.8), so the
  raw matched cosine has a large non-zero null. Subtracting the cosine to a *mismatched* target
  isolates the real "this prediction picks the RIGHT mon, not just a typical mon" signal.
- **In the live run the rise of `above_chance` is driven entirely by the baseline *falling*** —
  matched cosine is flat ~0.71, baseline drops 0.78 → 0.46. That fall is **de-collapse**: the head
  moving from "predict the average mon's role" to "predict THIS mon's role."
- **`latent_std`'s inverted-U** = *inflate then sharpen*: first the prediction cloud expands off
  the near-collapsed floor (de-collapse), then it contracts as predictions lock onto the
  structured target manifold. **Falling std while `above_chance` rises is the *opposite* of
  collapse** — it's sharpening. The collapse alarm is std→0 **with** `above_chance`→0.
- **The theory lens (Part 3):** this is textbook **non-contrastive self-supervised learning**. Our
  two metrics map onto **alignment vs uniformity** (Wang & Isola); the non-zero baseline is
  **representation anisotropy**; `above_chance` is a **contrastive / mutual-information readout over
  a non-contrastive loss**. Collapse resistance comes from **stop-grad + a task-anchored target + a
  VICReg variance floor** — and we honestly use only 1 of VICReg's 3 terms (see §3.6).

---

## Part 1 — What the head is and why collapse is the danger (intuitive)

### The latent-belief head in one sentence

Gen 3 has no team preview, so ~3 opponent mons are hidden. The **species head** guesses each
hidden mon's identity discretely (a softmax over species, trained with cross-entropy). The
**latent head** does something subtler: it predicts each hidden mon's identity *in role-space* —
a continuous vector — and is graded on how close that vector points to the **role-token the model's
own `pokemon_encoder` produces for the true hidden mon**. This gives *graded* credit: predicting
"some bulky Water type" when the truth is Suicune is mostly right and should score mostly right,
which a hard species label can't express.

### Why it's self-supervised, and why that invites cheating

The target isn't an external human label — it's the model's *own* encoder output for the true mon
(a **stop-gradient** target). That makes it **SimSiam-style**: predict your own representation of
the answer. Self-supervised cosine objectives have a trivial, useless optimum: **emit the same
constant vector for every input.** Cosine similarity to the target is then high on average, the
loss looks great, and the head has learned *nothing* — every hidden mon "looks identical." This is
**representation collapse**, the central failure mode of BYOL/SimSiam/VICReg-family methods.

### Two ways to collapse → two detectors

Collapse has two flavors, and one metric alone misses one of them:

1. **Dimensional collapse** — all predictions literally squeeze onto one point (or a
   low-dimensional sliver). Detected by **`latent_std`**: the spread of predictions → 0. We defend
   with a **variance floor** (VICReg) that adds loss whenever per-dim std drops below a target.
2. **"Predict the centroid" collapse** — predictions keep *some* spread but all hug the *average*
   role, so a prediction is no closer to its own mon than to a random other mon. `latent_std` can
   look healthy here, yet the head is non-discriminative. Detected by **`above_chance`**: matched
   cosine ≈ mismatched cosine ⟹ `above_chance` ≈ 0.

This is exactly why our test suite asserts both: a "predict the same shared target for all"
degenerate case has fine std but `above_chance ≈ 0` (`belief_aux_loss_test.py::
test_latent_above_chance_discriminates_identity_vs_mean`). The two metrics are **complementary
collapse detectors**, not redundant.

---

## Part 2 — The metrics, decomposed (technical)

### Definitions (all under the `belief/latent_*` TB prefix)

For each matched (prediction, true-mon role-token) pair in a minibatch:

- **`latent_cosine`** = mean cosine similarity to the **matched** (correct mon's) stop-grad target.
  Higher = closer match. (The loss trains `1 − cosine` + the VICReg floor.)
- **`latent_cosine_baseline`** = mean cosine to a **mismatched** target — the within-batch
  roll-by-1, i.e. each prediction scored against *some other* mon's role-token. This is the
  "non-zero null" of a non-orthogonal manifold: the cosine a predictor gets for free by regressing
  toward a *typical* role-token.
- **`latent_cosine_above_chance`** = `latent_cosine − latent_cosine_baseline`. The **discriminative**
  signal: how much better the prediction matches the *right* mon than a *random* mon. This is the
  latent analog of `species_acc_above_chance` (which anchors species accuracy to `1/n_species`).
- **`latent_std`** = mean over dims of the per-dim std of the predictions across the batch. The
  **dimensional-collapse monitor**.
- **`latent_vicreg`** = `relu(_LATENT_STD_TARGET − std).mean()`, with `_LATENT_STD_TARGET = 1.0`,
  weighted by `_LATENT_VICREG_WEIGHT = 1.0`. A hinge: it adds loss only while some dim's std is
  *below* 1.0; once std clears the floor it's exactly 0 and stops pushing.

Why a **stop-grad, task-anchored** target (no EMA momentum encoder, unlike BYOL): the target is the
model's own `pokemon_encoder` role-token, which is *already being trained* by the main RL + species
losses. Because it's anchored to a real task it can't drift to a trivial constant, so we get
collapse resistance without BYOL's momentum-encoder machinery — the stop-grad + VICReg floor are
the "belt and braces."

### The live decomposition (ai_v6_13, 0 → 73M steps)

| Metric | Early | Late | Shape |
|---|---|---|---|
| `latent_cosine` (matched) | ~0.79 @4M | ~0.71 | jumps, then **flat** (slight decay) |
| `latent_cosine_baseline` (mismatched) | ~0.78 @3M | **0.46** | jumps, then **monotonic fall** |
| `latent_cosine_above_chance` | ~0.00 | **0.25** (still rising) | **continuous monotonic rise** |
| `latent_std` | 0.54 | 1.57 (falling) | **inverted-U**, peak 2.62 @ ~27M |
| `latent_vicreg` | 0.46 | 0.00 (since ~7.4M) | floor cleared early |

### Why `above_chance` rises *continuously*: it's the baseline falling

The arithmetic is exact: matched ≈ flat 0.71, baseline falls 0.78→0.46, so the difference rises
0→0.25. **The rise is almost entirely the baseline dropping, not the match improving.**

Interpretation: early on the predictor is **near the "predict the centroid" collapse** — it emits
roughly the average believed-mon role for *every* slot, so a prediction is about as close to a
*wrong* mon's target as to its own (baseline ≈ matched ≈ 0.78, `above_chance` ≈ 0). As training
proceeds the head learns to point different slots in **different, mon-specific** directions. Its
cosine to a *random other* target therefore falls, while it keeps matching its own → the gap
widens. So **`above_chance` is literally a de-collapse / diversification curve**, and its still
climbing at 73M means the head is *still* getting more identity-specific (not converged, still
extracting signal).

**Is the continuous rise useful?** Two honest senses:
- **As a health / interpretability signal — yes, strongly.** It's the clean evidence the SimSiam
  head is doing its job: encoding *graded* hidden-mon identity in role-space. Still rising ⟹ live
  signal, not a dead/collapsed head.
- **As proof the feature helps the policy win — no.** Per the latent-head **honesty gate**
  (`project_latent_belief_built`): *learns ≠ helps*. `above_chance` rising tells you the aux is
  healthy and non-collapsed; only a win-rate / loss-crater A/B (coef on vs off) shows whether it
  improves play. "Decodable ≠ helps" is the standing caveat for every belief head.

### Why `latent_std` is an inverted-U: inflate, then sharpen

- **Rising half (0 → ~27M, 0.54 → 2.62).** The cloud de-collapses off the floor. The first lift
  (0.54 → ~1.0) is partly the **VICReg floor** doing its job — but `latent_vicreg` hits 0 by ~7.4M
  and std keeps climbing to 2.6, *far* above the 1.0 floor. So the overshoot is **the cosine loss
  itself**: to match a *diverse* set of targets you must emit *diverse* predictions, so dispersion
  grows well past the floor.
- **Peak + falling half (~27M → 73M, 2.62 → 1.57, still falling).** Predictions stop merely
  spreading and start **sharpening onto the structured target manifold**. As each prediction homes
  onto its specific mon's role-token, the within-prediction noise shrinks and total dispersion
  contracts toward the *natural* spread of the true role-tokens. (Mechanistic reading, not a logged
  number — we don't log the target manifold's own std; but it's consistent with everything else.
  Confirmed *not* a value-scale artifact: `popart/sigma` is flat ~15 across this window.)

### The crucial decoupling: dispersion ≠ discriminability after the peak

Early, std and `above_chance` rise *together* (both are de-collapse). **After ~27M they diverge:
`above_chance` keeps rising while `latent_std` falls.** This is not a contradiction — it's the
signature of healthy convergence. The head is separating the clusters (more discriminative) while
tightening *within* each cluster (less dispersed): predictions getting **sharper AND
better-targeted** at once. The same sharpening that shrinks total spread is what's driving
`above_chance` up. Falling std here is the *mechanism* of improvement, not a warning.

### The collapse alarm (the actual watch condition)

- **Healthy (current state):** `above_chance` ↑ (or holds) while `latent_std` ↓ → sharpening. Fine.
- **Collapse:** `latent_std` → ~0 **AND** `above_chance` → 0 (or turns *down*) → everything mashing
  to one point / one centroid. That pairing is the NO-GO (`src/agents/model/CLAUDE.md`: "std→0
  while cosine→1 = collapse").
- **Structural backstop:** the VICReg hinge re-engages the moment per-dim std drops below the 1.0
  target, so a *hard* collapse to 0 is actively prevented by design. At a mean std of 1.57 and
  falling, there's ~0.5 of headroom before the floor starts pushing back. If you ever see `latent_std`
  flatten near ~1.0 *with* `latent_vicreg` lifting off 0 again, that's the floor catching it — read
  `above_chance` at that point to decide if the head is sharpening (fine) or genuinely collapsing
  (coef too high / target too weak).

---

## Part 3 — The general theory (zoom out: what your two curves are an instance of)

Everything above is one concrete instance of a well-studied area: **non-contrastive self-supervised
representation learning**. The theory here explains *why* our head is built the way it is and *why*
the two metrics behave as they do — and it generalizes to any SimSiam/BYOL/VICReg-style head you
might add later.

### 3.1 The collapse landscape — why a self-supervised head needs guards at all

Self-supervised methods split on one question: **do you use negatives?**

- **Contrastive (InfoNCE / SimCLR, CPC).** The loss has an explicit denominator of *negative* pairs
  (other samples). Matched pairs are pulled together, everything else pushed apart. Collapse (all
  vectors identical) is *impossible* at the optimum — it would make the negatives maximally close,
  blowing up the loss. The cost is you need many negatives (large batches / memory banks).
- **Non-contrastive (BYOL, SimSiam, Barlow Twins, VICReg).** No negatives — the loss only asks
  matched pairs to agree. **The trivial constant `f(x) ≡ c` is now a global optimum** (perfect
  agreement, zero loss). So *something else* must break that symmetry, or the head collapses.

Our latent head is **non-contrastive** (the loss only pulls each prediction toward its own true
mon's role-token — no "push away from other mons" term). That is exactly why collapse is a live
danger and why we run explicit guards. The four known anti-collapse mechanisms, and which we use:

| Mechanism | Method | Do we use it? |
|---|---|---|
| Predictor MLP + **stop-grad** asymmetry | SimSiam | **Yes** — the asymmetric predictor + stop-grad target |
| **Momentum / EMA** target encoder | BYOL | **No** — replaced by a *task-anchored* target (below) |
| Explicit **variance** floor | VICReg | **Yes** — `relu(1.0 − std)` |
| Explicit **covariance** decorrelation | VICReg / Barlow Twins | **No** — honest gap, see §3.6 |

So our head ≈ **SimSiam + a VICReg variance floor + a task-anchored target** — belt and braces, no
momentum encoder needed.

### 3.2 Why stop-grad breaks the collapse symmetry (the EM view)

If gradients flowed through *both* sides of `cosine(f(x), f(y))`, the smoothest way to minimize it
is to send `f` toward a constant — both sides move toward each other and meet at collapse.
**Stop-grad freezes the target**, converting the objective from "make two moving things agree" into
"**chase a currently-fixed point**." Chen & He (SimSiam, 2021) argue this makes training behave like
**alternating optimization / Expectation-Maximization**: treat the target as a constant, fit the
predictor; then let the target update; repeat. The asymmetry (one side has a learnable **predictor**
head, the other is stopped) is what makes the fixed point informative instead of trivial — the
predictor is *load-bearing*, not decorative.

BYOL adds a **momentum (EMA) target encoder** to make that fixed point extra stable. We get the same
stability *for free* a different way: our target is the model's **own `pokemon_encoder` role-token**,
which is continuously trained by the **main RL loss + the species cross-entropy**. Because it's
anchored to a real task, it *cannot* drift to a constant (a constant encoder would tank species
accuracy and the value head), so no EMA is required. The phrase in the code — "task-anchored target,
no EMA/collapse" — is this argument in five words.

### 3.3 Alignment vs uniformity — the lens that unifies the two metrics

Wang & Isola (2020) showed contrastive representations optimize **two** properties on the unit
hypersphere:

- **Alignment** — matched pairs land close together.
- **Uniformity** — features spread out to fill the sphere (use all the space / all directions).

This is the cleanest frame for our two metrics:

- `latent_cosine` (matched) is essentially **alignment**.
- `latent_std` and a *low* baseline are essentially **uniformity** (spread / coverage).
- **Collapse = perfect alignment, zero uniformity** (everything aligned because everything is the
  same point).
- `above_chance` = alignment **net of the anti-uniformity leakage** (matched minus the shared
  direction that wrong pairs also share).

Read the live curves through this lens and they snap into place: **early training, alignment and
uniformity rise together** (escape collapse — std up, above_chance up). **Late training they trade
off** — raw uniformity (std) comes *down* as predictions sharpen, but **discriminative alignment**
(above_chance) keeps going *up*. A healthy run doesn't maximize both forever; it escapes collapse,
then converts loose spread into tight, well-separated, identity-specific clusters.

### 3.4 Why the baseline isn't zero: anisotropy and concentration of measure

A natural question: in a high-dimensional space, two *random* unit vectors are nearly orthogonal
(cosine → 0 as dimension grows — **concentration of measure**). So why is our `latent_cosine_baseline`
(cosine between a prediction and a *wrong* mon's target) sitting at **0.46–0.78**, nowhere near 0?

Because the role-tokens are **not random** — they live on a low-dimensional, task-shaped manifold
with a strong **shared component** (a dominant "mean role" direction). This is **representation
anisotropy** — the "cone effect," well documented for learned embeddings (e.g. representation
*degeneration* in NLP; Gao et al. 2019; Ethayarajh 2019): trained embeddings occupy a narrow cone
rather than the whole sphere, so even unrelated items have high cosine.

That reframes `above_chance` precisely: **it is a mean-centered / cone-subtracted similarity** —
strip out the shared anisotropic direction and measure only the **discriminative residual**. And it
explains the otherwise-odd fact that the **matched cosine itself *decays* (0.79 → 0.71) while
above_chance *rises***: as the head de-collapses, the target/prediction manifold **de-anisotropizes**
(the cone opens up), so absolute cosine to *anything* — including the right answer — drifts down,
*yet the right-vs-wrong gap widens*. **Absolute alignment ↓, discriminative alignment ↑.** Tracking
the raw cosine alone would have read that as "getting worse"; the baseline subtraction reveals it as
"getting sharper."

### 3.5 `above_chance` is a contrastive diagnostic over a non-contrastive loss

A subtlety worth savoring: our **loss** has no negatives (§3.1), but our **diagnostic** does — the
`baseline` is the cosine to a *mismatched* target (the within-batch roll-by-1), i.e. a negative
pair. So `above_chance = matched − one_negative` is effectively a **1-negative contrastive readout**.
Contrastive scores of this form are the basis of the InfoNCE bound, which **lower-bounds the mutual
information** between the prediction and the identity (van den Oord et al. 2018). Informally:
`above_chance` is asking *"does this prediction carry information about **which** hidden mon it is?"*
— a soft retrieval / MI signal — which is exactly the property the bare cosine loss does **not**
explicitly optimize. We monitor the thing we don't directly train, which is why it's the honest
health check rather than a restatement of the loss.

### 3.6 Dimensional collapse and our honest gap (VICReg has 3 terms; we use 1)

Collapse isn't binary. Jing et al. (2022) describe **dimensional collapse**: representations can
collapse along a *subset* of directions — the embedding **covariance matrix becomes low-rank**, a
chunk of its eigenvalues vanish — even while the overall scale looks fine. Two dimensions can each
have healthy std yet be **perfectly correlated**, in which case they carry one dimension of
information, not two.

Our `latent_std` is the **mean per-dim std** — a *coarse* proxy. It catches the gross "everything
shrank to a point" failure (and the VICReg floor actively prevents it), but it is **blind to
off-diagonal (correlated-dimension) collapse**: a low-rank-but-spread cloud can show a healthy
`latent_std`. The full **VICReg** objective has three terms — **V**ariance (per-dim std floor),
**I**nvariance (the matched agreement), **C**ovariance (drive off-diagonal covariances to 0 to
decorrelate dims). We use **Variance + Invariance** but **not Covariance**.

> **Open / unverified caveat.** A stricter monitor would track the **effective rank** /
> eigenspectrum of the prediction covariance (or add the VICReg covariance penalty). We don't
> currently log or regularize that, so a slow *dimensional* collapse with healthy `latent_std` and
> healthy `above_chance` is a blind spot in principle. In practice `above_chance` rising to 0.25
> argues the representation is *not* degenerate (a low-rank collapse would cap discriminability),
> but we have not measured the covariance rank directly — flagged as a known-unknown, not asserted.

### Key references (general theory)

- **InfoNCE / CPC** — van den Oord et al. 2018 (contrastive ≈ MI lower bound).
- **SimCLR** — Chen et al. 2020 (negatives + augmentation).
- **BYOL** — Grill et al. 2020 (predictor + momentum target, no negatives).
- **SimSiam** — Chen & He 2021 (stop-grad alone suffices; the EM view).
- **Barlow Twins** — Zbontar et al. 2021 (cross-correlation → identity, decorrelation).
- **VICReg** — Bardes, Ponce, LeCun 2022 (explicit Variance-Invariance-Covariance).
- **Alignment & Uniformity** — Wang & Isola 2020 (the two-property hypersphere lens).
- **Dimensional collapse** — Jing et al. 2022 (low-rank covariance failure).
- **Anisotropy / representation degeneration** — Gao et al. 2019; Ethayarajh 2019 (the cone effect).

---

## Where this lives in our architecture

- **Model side (`src/agents/model/CLAUDE.md` → LATENT belief, v18).** `BeliefHead` carries an
  asymmetric SimSiam predictor MLP; each believed slot's refined transformer token is regressed
  (cosine) toward the **stop-grad `pokemon_encoder` role-token** of the true hidden mon. Gated by
  `--opp-belief-latent-coef > 0`, which **requires `--opp-belief-aux-coef > 0`** (it rides the
  species head's believed slots + Hungarian slot↔mon assignment). OFF is byte-identical; the toggle
  is version-checked (v18) and threaded through `arch_toggles` at all opponent-load sites so a
  latent-ON self-play run doesn't FATAL on its own sentinels.
- **Target plumbing (leak-safe).** `Gen3Env` emits a training-only privileged Dict-obs key
  `belief_target_slots` [6,107]: a fresh per-mon encode of each hidden mon at its believed slot
  (same assignment as the species label). Read **only** by the loss — the model forward reads only
  `obs["observation"]`, so the omniscient target can't leak. Fuzz-guarded
  (`poke_env_gaps/belief_target_fuzz_test.py`: target == an independent fresh encode of the actual
  hidden mon).
- **Loss + metrics (`src/agents/training/CLAUDE.md` → Latent-belief loss;
  `instrumented_ppo._belief_aux_loss`).** On the species-CE Hungarian assignment, the latent term =
  mean cosine distance to the matched stop-grad target + the VICReg floor; folded at
  `opp_belief_latent_coef`. Constants `_LATENT_STD_TARGET = 1.0`, `_LATENT_VICREG_WEIGHT = 1.0`. All
  `belief/latent_*` metrics defined above; the trunk pull is broken out as **`grad/latent_share`**
  (+ `_norm_shared` / `_policy_cosine`) on the common-denominator grad-balance probe — watch it sit
  small (a few %); a spike with a degrading policy ⟹ lower the coef. `opp_belief_latent_coef` is
  training-only (read back on a flagless resume); the `opp_belief_latent` arch bool is
  version-checked.
- **Honesty gate.** The latent head **learns** (above_chance climbs to 0.25, non-collapsed) but it
  is **unmeasured whether it helps the policy** — that needs a fresh-run A/B, and the role-probe's
  "decodable ≠ helps" plus the unrevealed-inference probe remain the real gates
  (`project_latent_belief_built`, `project_belief_latent_role_probe`).

---

## The synthesis

The latent-belief head is a SimSiam-style self-supervised predictor, and its two headline metrics
are the **two complementary collapse detectors** that family always needs: `latent_std` guards
*dimensional* collapse (the VICReg floor keeps it off 0), and `latent_cosine_above_chance` guards
*"predict-the-centroid"* collapse (matched minus mismatched cosine, because the role-token manifold
is non-orthogonal so raw cosine has a big null). In the live run, `above_chance` rises *continuously*
because the **baseline falls** — the predictions de-correlate from a shared centroid and become
mon-specific — which is exactly the healthy "learning identity" signal (and it's still climbing, so
the head isn't done). `latent_std` is an inverted-U because the prediction cloud first **inflates**
off the near-collapsed floor (de-collapse, partly VICReg-lifted) and then **sharpens** onto the
structured target manifold; crucially, after the peak std *falls while* `above_chance` *rises* —
sharper and better-targeted at once, the **opposite** of collapse. The alarm to watch is the
*pairing* std→0 **with** above_chance→0; falling std on its own, alongside rising discriminability,
is convergence working. In the language of the self-supervised literature (Part 3), you're watching
**alignment and uniformity co-develop**: a non-contrastive head escapes collapse (uniformity up),
then trades raw spread for sharper, **anisotropy-reduced** discriminability (`above_chance` = the
cone-subtracted, MI-flavored residual) — held off the trivial constant solution by **stop-grad + a
task-anchored target + a VICReg variance floor**. And as always: this proves the head **learns**,
not that it **helps** — only an A/B settles that.

---

## See also

- `src/agents/model/CLAUDE.md` → **LATENT belief (v18)** — the SimSiam predictor, stop-grad
  task-anchored target, VICReg floor, the `std→0 while cosine→1 = collapse` NO-GO.
- `src/agents/training/CLAUDE.md` → **Latent-belief loss (`--opp-belief-latent-coef`)** — the loss,
  every `belief/latent_*` metric, the `cosine_baseline` / `above_chance` interpretability anchor,
  and `grad/latent_share`.
- Root `CLAUDE.md` → Feature Extractor Architecture (belief heads, the `--opp-belief-latent-coef`
  SimSiam latent predictor) and Model Versioning (v18).
- `designs/learning/marginalization_and_uncertainty.md` — the *other* belief-head concept (how a
  net carries/marginalizes uncertainty); this note is the *diagnostics* companion (how to read
  whether a self-supervised belief head is actually learning vs collapsing).
- Memory: `project_latent_belief_built.md` (the build + the honesty gate),
  `project_belief_latent_role_probe.md` (role-geometry validation; decodable ≠ helps),
  `project_hidden_team_belief_built.md` (the species head this rides on).
