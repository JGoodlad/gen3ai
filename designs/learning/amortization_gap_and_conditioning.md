# The amortization gap: why one shared policy under-performs its specialists, and why a perfect critic can't fix it

**TL;DR.** A single self-play generalist plays every team with **one** shared policy head, so the
per-team optimal strategies get **averaged in weight space** — team A's plan and team B's plan partially
cancel wherever they conflict, and the net lands at a compromise that's mediocre on *every* team relative
to a specialist. That's the **amortization gap**. It has two sub-problems — **extraction** (get the small
per-team signal out of the noise) and **storage** (hold conflicting per-team behaviors in one weight set
without cancellation). A better critic (and distillation) win *extraction*; they are **powerless on
storage**, because the cancellation happens in the *shared parameters*, downstream of the advantage. The
storage fix is **conditioning** (FiLM on a learned archetype latent `z_arch`), an architectural change,
not a critic change.

**Status (2026-07-17): FiLM is justified by evidence.** Probes this session showed (a) the base model
plays greedy-local and trades away its counters (the averaging is real), (b) distillation *fixes it on
distilled teams* but does **not** generalize to neighbors and **interferes** with the rest of the
distribution, and (c) the critic-richness route did not convert to strength. Per-team distillation is
expensive *and* doesn't scale (no generalization, plus interference) — conditioning is the mechanism that
makes the per-team fix *spread* (to neighbors) and *stop colliding* (with the rest). See **Empirical
evidence** below.

---

## What "amortization" means here

*Amortized* optimization = using ONE network to represent the solution to MANY tasks, instead of solving
each task separately. Our generalist is an amortized solver: one `π_θ` for all ~700 pool teams. The
**amortization gap** on team A = `performance(specialist π_A*) − performance(generalist π_θ on team A)`.
It's why the trap-exploiter beats the generalist *on the trap team* even after the generalist distilled
from it.

## The gap splits into extraction and storage

- **Extraction** — the per-team refinement is a *small* true advantage on a few decisions; getting it out
  of the noise is hard when the tendency is to average play-styles together. **The critic helps here**
  (lower-variance, unbiased advantage ⇒ higher signal-to-noise ⇒ the small signal survives), and
  **distillation helps even more** (the specialist teacher hands the student the per-team optimal *action
  distribution* directly — a full-distribution supervised target, far higher-SNR than any RL advantage;
  this is why offense *transfers*, [[project_double_sided_recipe]]).
- **Storage** — holding team A's "stay and trap" and team B's "switch to preserve the core" in **one
  shared head** without them overwriting each other. **This is the part the critic cannot touch.**

## Why a perfect critic cannot fix storage

The policy gradient over a mixture of teams is

```
g = Σ_team p(team) · E_{s,a|team}[ ∇_θ log π_θ(a|s) · A(s,a) ]
```

A *perfect* critic makes every per-`(s,a)` term correct and zero-variance. But SGD moves the **shared**
`θ` by the **sum** of the per-team gradient vectors. Where the per-team optima conflict, `g_A` and `g_B`
point in **opposing** directions in the subspace that encodes the team-specific behavior, so their sum is
small *there* and that behavior stays at the compromise. **The critic set each gradient's content
correctly; it did nothing about the fact that you add them into one `θ`.** The conflict is a property of
*parameter sharing + the geometry of the two tasks' optima*, independent of how well `A` is estimated.

Two sharper statements of the same point:

- **The linear-regressor picture.** Train one linear model to fit two datasets that want opposite slopes.
  *Perfect labels* for every point don't help — the single slope must compromise. You need two parameters
  (or a conditioning input that selects a different effective slope per regime). Perfect *targets* don't
  buy you two *slopes*.
- **A perfect critic reports the small signal as small — it doesn't amplify it.** You can't make `A`
  team-selectively huge without making it *wrong* (a biased critic). Scaling `A` up scales *every* team's
  advantage, so the conflict scales too and the *relative* cancellation is unchanged. So a perfect critic
  rescues the small signal from **noise** (extraction) but **not from cancellation** (storage) — small-
  but-correct signals cancel under weight-sharing exactly like noisy ones do, wherever they conflict.

## The concrete Pokémon version

On the **trap** team, in a given board the right play is "**stay in** and trap the switch-in with
Magneton." On the **stall** team, at a similar-looking board the right play is "**switch out** to preserve
the core." The policy *sees its team*, so it *could* learn the conditional — but the gradient for "stay"
(trap) and "switch" (stall) at similar active-state features push the shared *board→(stay-vs-switch logit)*
mapping in **opposite** directions. A perfect critic tells you *exactly* how good each is; both correct
updates then hit the same shared weights and fight. The net *can* carve out a team-conditional region, but
the **simplicity bias** resists it (the shared "play generally-good Pokémon" solution is lower-complexity
— see [[objective_richness_and_representation]]) and averaging blurs it. Nothing about the advantage
estimate changes that pressure.

## Three complementary levers (not substitutes)

| Sub-problem | Lever | Mechanism | Status |
|---|---|---|---|
| **Extraction** — get the small per-team signal | good critic + **distillation** | raise the SNR / hand over a supervised per-team target | distillation done; critic improved (FitNets un-crystallized it) |
| **Storage** — hold conflicting behaviors without cancelling | **conditioning: FiLM on `z_arch`** | per-archetype affine gain/bias on the head features ⇒ per-team gradients modulate *different* subspaces instead of overwriting one shared weight | **v1 BUILT (v44, `gen3_zarch_film_v1`, `--zarch-film heads`) — not yet run** |
| **Surpassing** — improve *beyond* the distilled teacher | good critic again | advantage SNR for the non-imitation RL climb | gated by the critic |

They're aligned in **goal** (close the gap) but act on **different mechanisms**, so a perfect critic maxes
out extraction and leaves storage wide open. The "generalist surpasses its specialists" dream needs **all
three**: the critic is genuinely necessary (extraction + surpassing), but **not sufficient** — even
perfectly-extracted, perfectly-scored per-team signals cancel when one shared head tries to store them.

## FiLM as the storage fix — and how we'd know if it's needed

FiLM (Feature-wise Linear Modulation) conditions the trunk on a learned archetype latent `z_arch`: it
applies `γ(z_arch) ⊙ h + β(z_arch)` to the trunk features, so "this kind of board" is *computed
differently* under trap-mode vs stall-mode. The per-team updates then land in different modulated
subspaces instead of fighting over the same weights — soft per-archetype parameter routing. Identity-at-
init (`γ≈1, β≈0`) so ON starts == baseline, like every structural toggle we ship.

**Alternatives to FiLM for storage** (be honest): per-archetype policy heads / mixture-of-experts,
gradient surgery (PCGrad — delete the conflicting gradient component), or simply more capacity. FiLM is the
parameter-efficient, elegant one, and the rich distillation we built is the prerequisite that makes
conditioning on a *learned* `z_arch` tractable.

**The falsifiable precondition — now RESOLVED (see the Empirical evidence section below).** The test was:
if the per-team optima don't conflict much (the gap is just extraction/SNR), a better critic closes it and
FiLM buys little; if the gap persists with a rich critic *and* the per-team fix neither generalizes nor
composes, storage (FiLM) is the wall. The 2026-07-17 probes ran this: the critic-richness route did **not**
convert to strength (storage, not extraction, is binding), and per-team distillation was shown to *not
generalize to neighbors* and to *interfere with the rest of the distribution* — the amortization/cancellation
made visible. **Conclusion: FiLM is justified** as the scalable escape from per-team distillation.

## Empirical evidence (2026-07-17 probes) — why FiLM is justified

Four probes this session settled "is *storage* the binding wall," and the answer is yes:

1. **The averaging is REAL — the base model plays greedy-local ("6 1v1s, not 6v6").**
   `tmp/counter_preservation_probe.py` measures how often our team's *unique defensive answer to a live
   threat gets traded away* (the "trades away the only counter" failure). The **un-distilled base `_14`
   loses its counters +0.236 more than the specialists**, biggest on the patient archetypes (stall +0.364,
   where the *specialist loses ZERO*; cmpass +0.346). That is the greedy-local averaging, quantified.
   Distillation *fixes it on the distilled teams* (`gap_21 ≈ +0.010`) — so the shared head *can* store the
   planning **when handed a strong per-team target**. (`tmp/amortization_gap_eval.py` agreed: `_21` matches
   the specialists on the 4 distilled teams, gap ≈ 0.)

2. **But distilling one team does NOT generalize, and it INTERFERES — the decisive result.**
   `tmp/distill_generalization_probe.py` (100 games, WR vs a fixed `_14`-on-pool opponent) distilled the
   Starmie-flex TSS team, then measured `_21` (distilled) vs `_14` (base) on a similarity gradient:
   - EXACT (Starmie, distilled): **+0.160** — distillation helped a lot, *on the exact team*.
   - SIM (same core, Zapdos flex / Moltres flex, *not* distilled): **−0.09 / −0.05** — **no** lift on the
     close neighbors.
   - DISSIM (different core): **−0.130** — the broader distribution **regressed**.

   So per-team distillation into ONE *unconditioned* head has **no within-archetype generalization** (a few
   anchors can't cover the space) **and interferes** (the distilled teams win the gradient tug-of-war at the
   expense of the rest). **That −0.13 regression is the amortization/cancellation, made visible.**

3. **The critic-richness route is NOT the lever.** The `_21` FitNets strength gate: un-crystallizing the
   critic (rank ↑) did *not* convert to strength (`td_resid_tail` flat, ELO flat); V_pub / human-replay
   value went NULL. So the wall is *storage*, not extraction/critic.

**The economic point that makes FiLM the answer.** Per-team distillation is *unbelievably expensive* (a full
exploiter run per team) and probe (2) shows it *neither generalizes to neighbors nor composes without
interference*. Conditioning fixes both: a neighbor maps to a *nearby* `z_arch` (so the +0.16 spreads to the
core's neighborhood), and each team's play routes to its own `z_arch` subspace (so distilling one can't drag
down the rest — no −0.13 regression). **FiLM turns "one exploiter helps one team, hurts the rest" into "a few
anchors lift whole neighborhoods without collateral damage" — the scalable version of a fix that otherwise
doesn't scale.**

**Honest caveats.** (a) At 100 games the neighbor deltas sit within ±7% noise — "no help" is solid, "actively
hurts" is suggestive; (b) `_21`'s general regression could be *partly* arch drift since `_14` (FitNets etc.),
not purely distillation interference — the clean control is `_18` (distill-only) vs `_14` on the same teams
(run before committing engineering); (c) the counter heuristic (type-resist × bulk) is coarse. But the
*pattern across probes* is coherent and points one way.

## The trilemma of a *self-created* archetype latent — and its resolution

The hard part isn't FiLM the mechanism; it's getting a good **`z_arch`** to condition on. Three
requirements look mutually exclusive:

1. **Self-created** — no hand labels (archetypes are fluid/dynamic; a hand-drawn bucket mis-fits the
   edge cases, and our PACE labels probed *weak*: pool-wide silhouette **+0.031**).
2. **No biased ground-truth label** — the very hand-labeling we're avoiding.
3. **Rich** — but to *enrich* a latent you must *force the model to use it*, and the only label-free
   forcing signal ("how should this team be played") is the **weak, sparse RL** signal — not a strong
   supervised one like the privileged belief labels.

**The hidden assumption is "strong supervision = a hand label." It doesn't have to be.** Two label-free
*strong* signals dissolve the trilemma:

- **The specialists ARE the label for "how to play the team."** An exploiter's policy — a *full action
  distribution per state* (dark knowledge) — is a **strong, dense, supervised** answer to "how do you play
  team X," and it is **not hand-drawn** (it was *learned* to play the team well). So `z_arch`'s job becomes
  *"the latent that lets FiLM reproduce the specialists."* That reconciles all three: self-created (learned;
  its coordinates emerge from *which specialists play alike*), unbiased (grounded in actual good play, not a
  bucket — an edge-case team's specialist plays it however it should be played, so there's nothing to
  mis-bucket), and richly-supervised (multi-specialist distillation is the pressure that *forces* `z_arch`
  to carry the routing info — strong, not the weak RL signal). This is why the whole distillation arc was
  the prerequisite for FiLM: **the specialists are the archetype supervision itself.** It is also elegant
  against the gap — the specialists are the *un-amortized* ground truth, the gap *is* "can't reproduce
  them," so supervision and goal coincide.
- **Unsupervised composition clustering (label-free structural scaffold).** The mon-role *atoms* are clean
  (per-mon probe: Starmie≈Zapdos, role silhouette **+0.185**, human-aligned sub-roles incl. Moltres≈Charizard),
  so a composition encoding gives `z_arch` rich structure *everywhere*, with **zero hand-labeling** — the
  model draws the map. This covers the teams without specialists and breaks the bootstrap (below).

So `z_arch` = *discovered* as the coordinate system that best compresses the specialists' collective play,
scaffolded by composition, refined by RL — never hand-labeled.

## Building `z_arch` robustly — the anti-over-fixation construction

The fear (the LLM-label failure mode): one mon/move flips an offense team to "stall." The over-fixation
**robustness probe** (`tmp/composition_robustness_probe.py`, OFF vs STALL teams) shows the FLIP RATE
(states nearer the *wrong* archetype centroid) is **0.030 for the pooled team summary / 0.041 for a DeepSets
mean of the 6 mon tokens / 0.092 for the fragile per-decision active-mon token** — i.e. **pooled/static is
~3× more robust than a per-decision read**, and the residual ~3% is *per-state wobble* that vanishes the
moment `z_arch` is **team-static**. (Caveat recorded: the raw `spread/gap` *ratio* is a red herring — in
128-D the total per-state scatter dwarfs a one-axis gap even when archetypes separate cleanly; **flip rate**
is the metric.) The robust construction, each choice aimed at a cause of over-fixation:

- **Team-static** — `z_arch = f(roster)`, computed **once per battle**, not per state/action ⇒ a *move*
  structurally cannot touch it (kills the residual → flip rate 0).
- **Continuous, not a discrete label** — a continuous code *drifts*, never *flips* (no argmax to tip).
- **Permutation-invariant pool of the 6 mon-role atoms (DeepSets)** — one swap changes 1/6 of the sum = a
  small *twist*, not a flip; no single mon dominates.
- **VIB bottleneck (β)** — compresses onto the *distributed, informative* composition axes, so it can't
  over-fit one salient mon/move (β is also the LUT-vs-style knob).
- **Smoothness/consistency loss** — require a team and its one-mon variant to get *nearby* codes: turns the
  "one swap = a twist" fear into a training constraint.
- **Soft manifold, never hard buckets** — fluid/edge-case teams get an *interpolated* position; no bucket to
  mis-fit into. (The "archetypes" are just dense regions; we cluster only for *our* visualization.)

Note the team CLS pool alone is **core-dominated** and washes out the flex nuance (team-archetype probe:
core silhouette +0.19, but the flex "twist" was ~chance) — so build `z_arch` by **aggregating the mon-role
atoms** (where role is clean), not by reading the existing team pool.

## Keeping `z_arch` alive — the anti-collapse regularizer

Is the "play teams well" RL pressure enough to prevent trivial collapse? **No — it works *against* you.**
The RL objective has a degenerate optimum where `z_arch` collapses to a constant and the shared policy
plays generically-well (the amortized solution) — and that collapsed solution is *simpler*, so the
simplicity bias favors it and the weak RL gradient to `z_arch` won't reliably escape it. Two distinct
failure modes, two fixes:

- **Collapse** (`z_arch` → constant / uninformative). Guard: **the composition-reconstruction objective
  IS the anti-collapse anchor** — if `z_arch` must reconstruct the team composition, it *cannot* collapse
  to a constant (a constant can't reconstruct different teams). This is the *same* objective as the
  day-0 composition prior, so anti-collapse comes free with the structure, label-free, no exploiters.
  Belt-and-suspenders: **VICReg** (a per-dim variance floor kills constant-collapse; a covariance term
  decorrelates dims) — we already have this from the belief-latent head (`aux/belief_latent_std` is the
  monitor). **The VIB β is NOT a collapse guard — it *causes* collapse if too high** (it compresses
  toward the prior mean); β is the rate-distortion / LUT-vs-style knob, balanced *against*
  reconstruction+variance. Don't reach for β to prevent collapse.
- **Dead** (`z_arch` informative but FiLM ignores it — γ→1, β→0 downstream). Guard: the *use* pressure
  from RL + specialist distillation (weak without exploiters → `z_arch` under-used but *harmless*, since
  FiLM is identity-at-init; strong with exploiters → FiLM routes). Monitor: FiLM γ/β deviation from
  identity + the gradient share into `z_arch`.

So "play well" alone is insufficient, but **reconstruction + VICReg** guarantee an informative latent from
day 0, and RL + specialists supply the *use* pressure that enriches it. Because reconstruction = the prior,
the no-exploiter phase is non-degenerate by construction.

## LUT vs style, and annealing β

The LUT (low β, per-team memorization) vs style (high β, compressed generalizing axes) failure modes are
**asymmetric**, which sets the safe default: **over-LUT fails gracefully** (per-team routing that *works*
on-distribution, just doesn't generalize to unseen teams + more params) — **over-style fails
catastrophically** (collapse → single code → no conditioning → back to the amortized baseline, the very
problem). So **bias toward LUT** (low β). For a *fixed* pool a LUT is genuinely fine — generalization to
unseen teams is a nice-to-have, not a requirement. If we want generalization, **anneal β LOW→HIGH
(LUT→style)** as a curriculum (learn to condition first with distinct codes, then compress to force the
transferable style axes), *slowly and monitored* — better still, **GECO-style dynamic β** (raise
compression only while the reconstruction/task quality holds), which self-regulates away from the collapse
cliff. The `z_arch` variance/rank monitor is the guardrail: if raising β drops variance toward collapse,
back off.

## Bootstrapping (the chicken-and-egg)

"Can't make an exploiter on a new arch without a generalist; FiLM needs exploiters; circular." Untangled:

1. **Adding FiLM does NOT invalidate the existing specialists** — it's computed from the *existing obs*
   (team tokens), so it doesn't change the obs vector; the 4 specialists consume the same obs and we distill
   their (arch-agnostic) *action distributions*. So there is **no chicken-and-egg for building FiLM now** —
   we have 4 specialists on the current arch.
2. **The circularity only bites at a full *obs*-retrain** (old checkpoints die), and even there it breaks:
   the **composition prior needs zero specialists** (structure from day 0), and **exploiters are single-team
   so they escape amortization regardless of the generalist's quality** (a mediocre bootstrap generalist
   still spawns *good* single-team exploiters).
3. **The loop:** Phase 0 — bootstrap generalist, FiLM **identity-at-init** (byte-identical) + `z_arch`
   **composition-supervised** (no specialists); Phase 1 — spawn N exploiters from it; Phase 2 — turn on
   `z_arch` **behavioral** supervision (distill the specialists) ⇒ FiLM routes ⇒ de-amortize; Phase 3 —
   better generalist → more exploiters → richer `z_arch`.

**So FiLM is not "off until N exploiters."** It is structural + identity-at-init + composition-supervised
from day 0 (never harmful); only the *behavioral* enrichment ramps up as the league grows. The composition
prior is what breaks the circularity — you don't need specialists to *start* `z_arch`, only to *sharpen* it.

## Synthesis

The reason a self-play generalist plateaus below its own specialists is not (only) that its advantages are
noisy — it's that one policy is being asked to be *all archetypes at once*, and the gradients for
incompatible team strategies partially cancel in the shared weights. A perfect critic makes each of those
gradients *correct*; it cannot make them *compatible*. Extraction (critic + distillation) and storage
(conditioning) are orthogonal, and the dream needs both — which is exactly why the whole rich-distillation
arc was the *prerequisite* for the FiLM/`z_arch` step, not a substitute for it.

The 2026-07-17 probes close the loop empirically: distillation *proves the shared head can store the
per-team planning* (it fixes the greedy-local counter-trading on the teams it distills), but it does so
*expensively, per-team, without generalizing to neighbors, and while interfering with the rest* — the
literal signature of one head trying to hold conflicting strategies. Conditioning is the mechanism that
keeps the per-team fixes from colliding and lets them spread across a core's neighborhood, turning an
un-scalable per-team distillation loop into a few-anchors-plus-`z_arch` one.

**Status (2026-07-17, later the same day): FiLM v1 is BUILT** — `gen3_zarch_film_v1` (v44,
`--zarch-film heads` + `--zarch-dim 32`): a team-static, permutation-invariant DeepSets `z_arch` over
OUR team's invariant facts (species/item/ability/moves/spread; detached embedding reads → zero trunk
interference), DETERMINISTIC in v1 (no VIB — per-forward sampling would break team-static, PPO's
ratio recompute, and eval determinism; LUT-first needs no rate limiter), conditioning BOTH root heads
post-projection pre-ReLU with zero-init generators (identity-at-init, OFF byte-identical).
Anti-collapse = species-recon BCE + a VICReg variance floor (auto-zeroed on single-team runs). Smoke:
recon acc rises, z variance healthy, FiLM norms grow off zero. Details:
`src/agents/model/CLAUDE.md` → Team-archetype latent + head FiLM; the as-built deviations from the
original sketch are recorded in [[self_discovered_archetype_latent]]. **Remaining next steps: run the
`_18`-vs-`_14` interference control (still outstanding — gates the strength of the "interference"
claim), then the FiLM fresh run with at least one seed-anchor distillation**, measuring the payoff as:
does distilling an anchor now *lift its neighbors without regressing the rest*.

**The Phase-0 verdict (2026-07-19, the ablation probe — the capstone sentence, measured).** After
~28M steps of pure-RL FiLM across ai_v8_01/02/03: zeroing the FiLM generators on the live checkpoint
(exact identity — the multiplicative design makes the ablation surgical, `tmp/film_ablation_probe.py`)
costs NOTHING — ablated beat intact 0.567 ± 0.08 over 150 games, and a WRONG-Z model (mismatched team
code) drew intact at 0.500, even though FiLM flips ~16% of greedy actions (KL 0.048). So the learned
modulation is behaviorally present but strength-neutral and team-content-free — and the ai_v8_03
plateau-break (first promotions + ELO 2009 vs the ~1970 baseline) is attributable to the CURRICULUM
(dropping the stable exploiters + one-sided team-PFSP), NOT the conditioning. Four independent
instruments now agree: `film/noise_scale` ≈ 8–9× the batch (the per-team RL gradient is starved),
the post-FiLM value features are THINNER than the no-FiLM baseline (PR 2.9–3.0 vs 3.24, between-team
share +2pp — the critic never became archetype-conditional), per-team evals flat, and the ablation.
**This is "FiLM is the ENABLER, not the cause," measured**: the architecture is built, alive, and
EMPTY under the weak RL diet — Phase-2 distillation (policy side) + the categorical-critic /
value-feature-hint work (value side) are the specifically-indicated fill, and this ablation probe is
their acceptance test (ablated-vs-intact must drop below 0.5; wrong-z must start costing).

**REVISION (2026-07-19, later — the owner's fixed-matchup design overturns "content-free").** The
pool-vs-pool head-to-heads let BOTH sides' team draws vary, and that matchup variance washed the
signal out. Freezing the matchup (`tmp/film_fixed_matchup_probe.py`: pilot TSS-starmie vs the
intact model on the stall team, 250 games/arm, @186M) revealed: **INTACT 0.536 ± 0.062, ABLATED
0.484 ± 0.062, WRONG-Z (stall code) 0.368 ± 0.060** — feeding the style-mismatched code craters
play by ~17pp (~4σ vs intact). So the policy DOES condition on z's content — the earlier
"wrong-z costs nothing" was a measurement artifact of pooled matchups (a pinned code is only badly
wrong for a subset of drawn teams, and the harm concentrates on style-contrast matchups). The
ASYMMETRY is the signature of early-phase conditioning: removing the signal degrades gracefully to
near-marginal play (−5pp, ~1σ — real-but-small benefit, not yet significant), while a WRONG signal
actively misleads (−17pp) — harm-sensitivity precedes benefit, like a student who has started
trusting a textbook: take it away and they cope from memory; hand them the wrong one and they follow
it off a cliff. Implication: pure RL HAS partially filled the conditioning pathway (the discovery
barriers — gradient competition, starvation, exploration collapse — slow it but don't zero it), which
STRENGTHENS the case for the in-loop discovery boosters (per-group FiLM grad accumulation,
team-blocked episodes, style compression) as alternatives to heavy distillation. Measurement lesson
(general): when hunting a small conditional effect, FREEZE every non-treatment factor — pooled
designs average a concentrated effect into invisibility.

**REVISION (2026-07-20 — the two-direction probe @~205M catches the LAZY mode red-handed).** After
~15M steps of the booster stack (team-block-64, global accum-16, film-accum-4) the modulation grew
2–3× (`vf_dev` 0.56→0.85, `team_std` ~3×) and the SAME probe run in BOTH directions now reads:
TSS pilot — INTACT 0.348, ABLATED 0.524, WRONG-Z(stall) 0.396; stall pilot — INTACT 0.696, ABLATED
0.648, WRONG-Z(TSS) 0.700 (250 games/arm; the two intact-vs-intact mirrors agree, 65.2% vs 69.6%
for the stall side; TSS-starmie verified IN the 719-team trainee pool, so this is in-distribution).
Decomposition: modulation ON-vs-OFF moves play a lot, but the CODE CONTENT moves it ~nil in both
directions (intact ≈ wrong-z) — i.e. FiLM's behavioral effect at this checkpoint is dominated by its
team-SHARED component, a global style tilt (helps the stall pilot +5pp, hurts the TSS pilot −18pp on
this matchup pair; the model's stall play also strengthened outright — the intact mirror flipped
from 54/46 TSS-favored @186M to ~2:1 stall-favored). This is EXACTLY the lazy mode §(1) below
predicted — dev is growing ~3.4× faster than the differential (`dev` 0.85 vs `team_std` 0.25) and
the probe confirms the differential is behaviorally inert. The 186M content-sensitivity (wrong-z
−17pp) did not persist: under the boosters the shared component grew fastest and swamped it.
Candidate fix (not yet built): **center the FiLM output across teams** — subtract the running
team-mean modulation so FiLM can only express per-team DEVIATIONS and the shared tilt is forced back
into the trunk (the PopArt pattern applied to conditioning); alternatively freeze/shrink film LR and
let the trunk absorb the shared shift. Global signals (4 promotions, ELO high-water ~2010) say the
shared tilt is not net-harmful to strength — but it is not de-amortization either.

**REVISION (2026-07-20, later — the owner's emphasis critique + the stratified GRID probe).** The
owner asked whether the two frozen matchups were themselves unrepresentative — out-of-distribution
w.r.t. what team-PFSP currently EMPHASIZES. Checked: only 321/719 pool teams have live win-rate
EMAs; the TSS twin is mid-pack (0.540) and the stall anchor is COLD (unmeasured) — neither sits in
the emphasis. The grid probe (`tmp/film_grid_probe.py`, @208.6M: 7 pilots stratified by live EMA —
HIGH = the 3 weakest/most-drawn under onesided PFSP, all defensive archetypes; MID = the TSS twin;
LOW = 2 de-emphasized strong teams; COLD = the stall anchor — × 2 median-EMA opponents ×
intact/ablated, 100 games/cell) found: **HIGH intact 0.178 vs ablated 0.087 (δ +9.2pp, ~4σ — FiLM
DOUBLES the win rate exactly where training is grinding); MID −3.5, LOW −4.0, COLD −7.5pp**
(net over all 14 pairs ≈ +1pp). So the earlier "modulation hurts in-distribution" was a SAMPLING
artifact — both old anchors sat outside the emphasis. But the content channel stays inert: wrong-z
(a balance team's code) matched or BEAT intact on every HIGH cell it ran (0.117 vs 0.090 on the
balance defender). The sharpest mechanism read is the HIGH-vs-COLD contrast — both defensive-
archetype pilots, own codes, modulation ON: the heavily-TRAINED one gains +9.2, the UNTRAINED one
loses 7.5. So the benefit is tied to per-team training EXPOSURE (PFSP grinds a weak team →
modulation becomes the delivery vehicle for what was learned there), while the z code that unlocks
it is interchangeable among in-distribution codes. FiLM + onesided team-PFSP are functioning as a
COUPLED improvement loop (point at the weakest teams, store the gains in the fast modulation
channel, EMAs rise, PFSP moves on) — which retracts the centering recommendation in its "do it
soon" form: centering would amputate the demonstrably active shared component mid-loop. Untested
cell that would settle the mechanism: wrong-z on COLD/MID/LOW (does generic modulation help
independent of exposure?). Measurement lesson (general): when a treatment interacts with the
TRAINING DISTRIBUTION, stratify the eval by the sampler's own emphasis weights — a "representative"
probe pair chosen by salience (well-known teams) can still be OOD w.r.t. where learning is
happening.

**Two refinements to the Collapse-vs-Dead section above (2026-07-17, owner discussion — same
conclusion, sharper mechanics).** The section already splits the failure modes correctly (Collapse
killed by reconstruction, no exploiters needed; Dead = under-used-but-harmless until specialists
supply the use pressure — the Phase-0/Phase-2 bootstrapping plan). Two things sharpened since:
(1) *why* pure RL still grows FiLM, not just "weakly": the generator gradient is an **outer product
with z**, so per-team components write to different z-directions and don't cancel the way shared-head
gradients do (the storage mechanism acting on FiLM's own weights; the smoke shows the norms rising
immediately). The honest residual risk is therefore the LAZY mode — growing on z's team-SHARED
component (a generic scale/shift = free capacity) while the per-team DIFFERENTIAL, the actual
de-amortization, stays weak (fed by the small extraction-limited per-team advantage). Distillation =
the *sharpener* of the differential, not an aliveness prerequisite. (2) The monitor this section
prescribed ("FiLM γ/β deviation from identity") **cannot distinguish those two modes** — the shipped
upgrade is the split `film/{side}_dev` (mean |modulation|) vs `film/{side}_team_std` (modulation
spread ACROSS teams): `team_std`≈0 while `dev` grows = lazy generic mode, both rising = real routing.

## Why FiLM "adds capacity" — three distinct claims (2026-07-20 owner discussion)

"FiLM adds capacity" tangles three claims with different evidence:

1. **Representational capacity — only the DIFFERENTIAL is new.** The team-SHARED part of the
   modulation (a constant scale/shift on head features) is mathematically absorbable into the
   projection Linear the old model already had — zero new expressiveness. The team-differential
   part IS new: the head goes from one function of the trunk features to a FAMILY of functions
   indexed by team. The trunk always saw the team in the obs but compressed it away (the value
   features run in ~3–5 effective dims — the minimal-sufficient-statistic bias); FiLM re-injects
   team identity at the last layer, where using it is cheapest.
2. **Optimization capacity — the escape hatch (the mechanism that carried the plateau break).**
   A converged trunk is fired clay: every weight load-bearing, Adam second moments calibrated to
   tiny gradients — the model isn't out of things to learn, it's out of CHEAP DIRECTIONS to learn
   them in. FiLM's fresh zero-init weights (identity at init → moving them initially breaks
   nothing, clean Adam state → large permissible steps) are a new low-interference direction —
   the LoRA-on-a-frozen-LLM effect. The measured shared "defensive tilt" is a shift the trunk
   could always REPRESENT but was too converged to REACH.
3. **Gradient-coherence capacity — why K same-team episodes "update the same weights."** The
   generator gradient is an outer product (head error ⊗ z), so a team-block's 64 consecutive
   episodes all write into the SAME z-slice — and the next block (different team) writes into a
   DIFFERENT slice. In the shared head both blocks hit identical weights: coherent within a
   block, then partially cancelled by the next team's pull (the amortization gap's storage
   failure). The outer product routes per-team lessons into separate subspaces so they accumulate
   across the pool instead of averaging out.

**The counterfactual (would old-arch + new opponent selection have grown too?) — unmeasured,
bracketed honestly:** no zarch-off fork on the new diet exists. Some growth WAS architecture-free
(the boosters cut global gradient noise 1.7→1.07 for the trunk too; onesided PFSP concentrates
signal regardless of arch). But the eval-time ablation shows the learning CHOSE the modulation
pathway as storage (removing it halves the win rate on exactly the emphasized teams — if the trunk
could have absorbed those lessons as easily, ablation would be harmless), and the HIGH-vs-COLD
contrast shows the break needed BOTH legs: emphasis without storage did nothing historically (v7
plateaued with weak teams always in the pool), storage without emphasis does nothing now (the cold
stall team gains nothing from the same machinery). Selection supplies concentrated signal; FiLM
supplies a place to put it that other teams' gradients can't erase — a PRODUCT, not a sum. Honest
weighting today: escape hatch + PFSP targeting first (team_std still ~3.5× below dev; wrong-z codes
interchangeable), subspace routing second and growing. The clean falsifier = an old-arch fork on
the identical diet (a full run's cost for one attribution bit; the ablation buys most of it cheap).

## The FiLM family — what else exploits the same three mechanisms (2026-07-20 owner discussion)

Each mechanism FiLM exploited generalizes into a family; each family has a member matched to a
measured disease of this system:

1. **More conditioning axes** (the representational win). **z_opp / matchup FiLM** — condition on
   the REVEALED/believed opponent team (the belief heads already infer it); sharpens within-episode
   (deterministic per state → PPO-legal); ~5 opponent archetype clusters ⇒ ~100× the sample density
   of per-team codes; unlocks adapting *to who you fight*, the exploiter skill team-conditioning
   can't express. **Privileged-critic conditioning** (asymmetric actor-critic, the sleeper) — FiLM
   the TRUE hidden opponent team (training-only labels we already extract) onto the VALUE head:
   the baseline may condition on anything without biasing the policy gradient, and our critic's
   diseases (tail-blindness, PR ~2.9 thin features, the falsify-scan `unattributed` bucket) are
   partly UNLEARNABLE-TARGET problems — returns depend on hidden state the critic can't see. Keep
   the public-input critic as the deployed readout (prober/search); the privileged one is the GAE
   baseline only.
2. **Fresh identity-at-init capacity** (the escape-hatch / optimization win). The literature name
   for the disease is PLASTICITY LOSS. Members: **trunk adapters** (zero-init residuals between
   transformer layers — the damage-refine/re-attend pattern, generalized; a `--zarch-film trunk`
   A/B), **periodic capacity grafts** (Net2Net/progressive-networks lineage on our version-gated
   playbook), and the nuclear **reincarnation** (distill the converged policy into a re-initialized
   net at an epoch boundary — every piece already built). NOT from this shelf: dropout, batch norm
   (unmodeled nonstationarity), noisy-nets (off-policy exploration tool).
3. **Stronger gradient routing** (the coherence win). The ladder: FiLM (diagonal) → conditional
   LoRA (z-generated low-rank deltas) → hypernetworks → MoE heads (router keyed by z). All
   deterministic/PPO-safe — but **storage is no longer the binding constraint; SIGNAL is**
   (team_std 3.5× below dev because the per-team advantage is extraction-limited, not because FiLM
   lacks rank). Climb the ladder only when team_std saturates while per-team probes still show
   headroom.

Shortlist (in order): privileged critic (gate: the falsify-scan unattributed bucket shrinks) →
z_opp (gate: per-archetype win-rate spread narrows) → trunk adapters (gate: the escape-hatch
fast-early-movement signature). Every candidate must pass the FiLM shippability checklist:
deterministic, OFF-byte-identical, identity-at-init, version-gated, with a named falsifying metric.

**The scouting caveat on the privileged critic (owner catch, 2026-07-20).** A NAIVE oracle critic
(true state INSTEAD of the observation) devalues information gathering: on a Swampert
Protect-scout for HP Grass the oracle sees tempo burned and nothing learned (it already knew), so
the bootstrapped TD error on the scouting move is NEGATIVE while the real benefit lands later,
diffusely. (With pure MC returns any baseline is unbiased; with GAE the critic's VALUES enter the
advantages, and a critic without uncertainty assigns no value to reducing uncertainty.) The fix
(Baisero & Amato, unbiased asymmetric actor-critic): condition on the agent's INFORMATION STATE
AND the true state — the full public obs (belief features, threat provenance) PLUS a privileged
FiLM — so a reveal changes the critic's input and information gain is priceable. Residual risk =
shortcut learning (the privileged wire atrophies the belief-pricing); the falsifying gate is a
scouting probe: V(s) must still jump on reveal events with the privileged branch ABLATED.

**The conditioning ladder, precisely.** Head = W·h + b; the ladder is "how much of that
computation does z command": **FiLM** h′=h·(1+γ(z))+β(z) — a mixing board, per-dim gain/offset on
FIXED channels, no feature mixing (the DIAGONAL of a hypernetwork); **conditional LoRA**
W′=W+U(z)·Vᵀ — z generates a low-rank weight UPDATE, so each team gets r NEW feature
combinations (a low-rank-truncated hypernetwork; still zero-init-able → identity); **hypernetwork**
W(z) generated whole — every team its own head, smooth in team space, costs |W| outputs + delicate
init; **MoE** = a hypernetwork QUANTIZED to N choices + a router.

**Signal extraction (the actual binding constraint).** "Signal" = advantage differences between
team-specific best play and generalist play REACHING the gradient. On-policy PPO cannot see the
advantage of a line the collapsed policy never samples — counterfactual invisibility at any batch
size (blocking/batch fixed density and noise; this barrier remains). Ranked by yield: (1)
**search-as-teacher** (BUILT, coef-0) — the CRN beam finds VERIFIED better lines and distils them
(AWR × confirmed Δwin): manufactures counterfactual signal instead of waiting to sample it; (2)
**exploiter distillation** (validated, ai_v7_19 double-sided recipe) — a dedicated per-team
signal-mining run, and FiLM now provides per-team storage for the transfer (the Phase-2 pairing);
(3) the fixed **privileged critic** — variance reduction IS extraction (same samples, cleaner
advantages); (4) **tail-weighted/distributional value objectives** — per-team differences
concentrate in tail events. Ordering logic: (1)/(2) CREATE signal absent from the on-policy
stream; (3)/(4) conserve what's present. Creation beats conservation while exploration collapse
binds.

**Signal per unit compute (2026-07-20, compute-limited prioritization).** At the margin,
signal/FLOP is about TARGETING, not volume: near convergence almost all rollout signal is
REDUNDANT (the policy already plays those states as the gradient would push — that redundancy IS
the plateau), yet ~100% of the box runs rollouts. Ranking: (1) **free riders** (~2–5% overhead) —
privileged critic + tail-weighted vf loss reweight samples already paid for (variance reduction ≡
a larger effective batch); (2) **the mining assembly line** — triage/falsify-scan (offline, idle
CPU, converts existing LOSSES into a ranked list of provably-thrown states; the
`policy_reducible` bucket is extracted signal sitting unused) → better-line search verifies ONLY
those states (~tens of s/state; yields counterfactual signal on-policy sampling cannot produce at
ANY volume) → AWR distill folds corrections at ~zero GPU cost. Filter-then-verify economics:
every expensive FLOP is spent where cheap FLOPs proved there's something to learn — orders of
magnitude better per bit than rollouts at a plateau; (3) **exploiter distillation** — ~100–1000×
cost per improvement but extracts STRATEGIC (multi-turn closed-loop) signal that depth-2 search
can't see; use for archetype-level holes; (4) **more generic self-play** — worst at a plateau,
but do NOT cannibalize a run that's climbing (ai_v8_03 @full-LR is): STAGE the pipeline on spare
CPU (traces are already written, corrections bank at coef-0) so the reallocation is pre-loaded
when the slope flattens.

**Correction (owner: "we are CPU-flops limited").** The scarce currency is CPU-HOURS (~86% of
wall is rollout, obs-build ~88% of per-decision CPU, encode ~80%); the GPU is ~86% IDLE. This
re-sorts the list: **(0) engineering multipliers on the CPU bottleneck** — a Rust
`state_encoder` hot path (the Rust bridge exists; "encoder is the bottleneck, not the
transformer") and verifying whether `--unified-obs` still COMPUTES the masked blocks env-side (if
so, skipping them is a pure CPU refund) — a 2–3× encoder speedup buys 2–3× of EVERY signal
source; **(1) GPU-side free riders are even freer** (privileged critic, tail-weighted vf, bigger
critic — they spend the idle resource); **(2) mining vs rollouts is a REALLOCATION of the same
saturated CPU** (search/falsify re-rolls ARE bridge battles + obs builds; "spare CPU" ≈ cycle
tails only; trace-reading triage/scan is the near-free exception) — the exchange rate is
slope-dependent: rollouts pay while climbing, mining dominates at plateau; **(3) exploiters =
bulk CPU for strategic-depth signal.**

## How much per-team divergence does the game actually reward? (2026-07-22, the ground-truth probe)

The "team_std is stuck at 5:1 (dev:team_std)" concern rested on an unexamined assumption — that
per-team differentiation *should* be large. `tmp/specialist_divergence_probe.py` measured the
ground truth: a per-team specialist (exploiter) IS the optimal per-team policy, so the BEHAVIORAL
divergence between each exploiter and the generalist (ai_v7_14), on the exploiter's own team, is
the target amount of per-team conditioning. Basis-independent (independently-trained nets → feature
L2 meaningless), so KL(π_spec‖π_gen) + action agreement + a **control** (a second, non-specialized
generalist ai_v7_02 vs the same reference — the baseline "two generalists disagree" from arbitrary
tie-breaking). 4 teams, ~1000-2000 states each:

- Raw KL(spec‖gen) ≈ 0.65 and ~50% action agreement LOOK huge — but the CONTROL generalist already
  diverges KL ≈ 0.42 / agrees 59%. So ~2/3 of the apparent divergence is baseline noise, not
  specialization. **Specialist-attributable EXCESS = +0.23 KL (≈21% of the policy's 1.10 entropy);
  the action-agreement gap is only +6pp.** → the true per-team play the generalist misses is
  MODEST. **5:1 team_std is NOT clearly wrong** — the owner's "maybe it's optimal" challenge is
  substantially borne out.
- Concentrated on STRATEGY-DEFINING teams: trap (+0.32) / cmpass (+0.30) carry ~2× the excess of
  TSS (+0.12) / stall (+0.18) — mechanic-driven plans (trap, Baton-Pass) reward per-team play;
  generically-defensive teams barely do.
- Moderately spread (top-10% of states hold ~38% of KL — some concentration, not pivotal-only).
- Both caveats push the SAME way (→ +0.23 is an UPPER BOUND): the exploiters were trained to beat
  ai_v7_14 *specifically* (some excess = opponent-exploitation, not team-optimal), and states come
  from the specialist's own steered trajectory. True headroom < +0.23.

**Implication:** the per-team FiLM prize is real but MODEST and team-specific (gimmick teams), best
captured by targeted distillation for those archetypes — NOT a broad push to inflate team_std. This
explains why team_std stays small under pure RL (the per-team advantage signal is genuinely small →
extraction-limited, not architecture-limited) and DE-prioritizes the FiLM-differential work vs the
broader levers (privileged critic, categorical critic, search-teacher). The metric wasn't lying;
the assumption that it should be bigger was.

## CORRECTION: the per-team prize is LARGE, not modest — the KL proxy misled (2026-07-22 mirror)

The "How much per-team divergence does the game reward?" section above concluded (from the
divergence probe's ~0.23 excess KL) that the prize was MODEST and "5:1 team_std is fine." **That
was wrong — KL magnitude was a poor proxy for the prize.** The equal-pilot mirror
(`tmp/piloting_mirror_eval.py`, 60 games/side vs the current ai_v8_03) measures the OUTCOME
directly: exploiter-pilots-team vs current-on-pool MINUS current-pilots-SAME-team vs current-on-pool
= the pure PILOTING edge (team advantage cancels). Result: mean **+0.183 win-rate** (trap +0.300,
stall +0.166, TSS +0.134, cmpass +0.133), team-only baseline ~0.53. So the exploiters' ~71% win vs
current is almost ENTIRELY piloting, not team. **trap is the smoking gun: the current model pilots
the trap team to 0.400 — it LOSES with a legal team — while the exploiter pilots it to 0.700**, a
+30pp purely-piloting gap. Reconciliation: a MODEST policy change (0.23 KL) LEVERAGED at pivotal
decisions produces a LARGE win-rate swing — the two are consistent; I conflated "how differently
they play" with "how much better the outcome." So team_std being low is a REAL deficit (the
washing), NOT the optimal, and distillation-into-FiLM is WELL-JUSTIFIED (~18pp distillable,
gimmick-team-concentrated). Exploiters are good teachers AS-IS (they beat current on piloting, not
anti-v7_14 gimmicks). CI caveat: individual per-team edges ±0.17 at 60g (cmpass/TSS suggestive);
the mean and trap are solid. This is the washing-limited story made concrete: trap's per-team play
washed out of the generalist entirely; the dedicated exploiter has it; distill it.

## "Signal-limited" was imprecise — it's WASHING-limited (2026-07-22 owner catch)

The owner pushed back on "team_std stays small because the per-team signal is small/extraction-
limited": **an exploiter trained against ourselves is NOT signal-limited** — it converges to strong
per-team play, proving the per-team signal EXISTS and is EXTRACTABLE. The precise diagnosis is
WASHING, not absence. The generalist loses the (real) signal to three things the exploiter doesn't
face: (1) sample DILUTION (~0.14%/team vs the exploiter's 100%), (2) gradient INTERFERENCE in the
shared trunk (team A's "stay" vs team B's "switch" at similar features sum to ~0), (3) FiLM ROUTING
to the shared tilt (simplicity bias). The exploiter avoids all three by DEDICATING params+samples
to one team.

**The fix is therefore not more RL or more storage — it's transferring the already-extracted
exploiter signal into isolated storage:** (Move 1) distill the exploiter's FULL action-distribution
target (replayable → no dilution; supervised → no extraction fight; higher-SNR than a scalar
advantage); (Move 2) distill INTO FiLM, not the shared head — the shared head washes out exactly
like RL (this is why the earlier shared-head distillation "interfered with the rest"). **The
mechanism that forces team_std up where RL couldn't:** N DISTINCT per-team targets CANNOT be fit by
one shared modulation, so the shared component can only capture the common part and everything
team-specific is FORCED into the differential channel — the supervised targets deny the shared
solution its low-complexity escape hatch. (Move 3, optional: center FiLM to structurally forbid the
shared absorption.) Bounded by the modest measured per-team divergence (~0.23 excess KL, gimmick
teams) — so a targeted lift, not a transformation. This IS the Phase-2 exploiter-distill-into-FiLM
pairing: both halves are BUILT (exploiters exist, FiLM shipped v44); the never-run step is feeding
FiLM the concentrated exploiter signal instead of the washed RL advantage.

## The distinct-target mechanism CONFIRMED (2026-07-22, ai_v8_04 distillation run)

Launched the 4-teacher distill-into-FiLM run (ai_v8_04, fork from ai_v8_03, --distill-coef 1.0,
--distill-team-bias 0.4, the 4 exploiters as TEACHER:TEAM pairs). ~3M steps in, all gates move
COHERENTLY (not a fork transient — a transient reverts; a dropping KL confirms real convergence):
distill/kl 0.41→0.15 (converging), agree_rate 0.59→0.73+ (rising→1), **film/pi_team_std 0.29→0.43
AND film/vf_dev 1.18→0.94** — the differential GROWS while the shared tilt SHRINKS, the dev:team_std
ratio collapsing ~4:1→~2:1. This is the predicted mechanism made real: N DISTINCT teacher targets
can't be fit by one shared modulation, so FiLM is FORCED to move capacity from the generic tilt into
per-team routing. FIRST TIME team_std moved off ~0.25 in the whole program — RL never could (washing);
distillation does (the exploiters already extracted the signal). Expected transient: self-play ELO
dipped 2016→1980 (policy in flux + fresh pool). MECHANISM confirmed; PAYOFF pending — "team_std grew"
≠ "per-team win rate improved" until the per-team eval is re-run at settle (~agree plateau), esp trap
(the 0.40 deficit). learns≠helps still applies, though the mirror says the teacher targets are good.

**PAYOFF CONFIRMED (2026-07-22, @275M, agreement settled ~0.75-0.79).** `tmp/distill_payoff_eval.py`
(distilled trainee vs pre-distill baseline, each piloting the 4 teams vs the SAME fixed opponent,
60g): **per-team piloting +0.129 mean, ALL 4 teams up** — TSS 0.60→0.73 (80% of the exploiter gap
closed), stall 0.52→0.68 (63%), trap 0.47→0.60 (57% — the sub-even team now WINS), cmpass 0.45→0.53
(56%). So the mechanism (team_std up) CONVERTED to play (~64% of the exploiter's piloting edge
captured; the double-sided recipe's self-play half caps it below 100%). NOT learns≠helps — the whole
program validated end-to-end: washing-limited barrier → exploiter has the signal → distill into
head-FiLM → per-team win rate rises. **head-FiLM is SUFFICIENT** (the distilled signal converts to
play at 56-80%), so PLACEMENT was NOT the limiter and trunk-FiLM stays PREMATURE — the barrier was
always signal (washing), now fixed. CI caveat: per-team deltas ±0.17 at 60g (individually noisy), but
4/4 positive + the +0.13 mean (~240g/condition) is solid. Self-play ELO dipped 2016→1985 (distribution
shift + fresh pool — a confounded proxy during distillation; the per-team WIN RATE is the real read).
NEXT: scale distillation to more archetypes (4 teams helps those 4; broad lift needs broad coverage).

## See also
- [[objective_richness_and_representation]] — the simplicity bias / minimal-sufficient-statistic backbone (why the shared solution wins by default) + the distillation bits-ladder
- `src/agents/model/CLAUDE.md` → dual-head value readout, the shared trunk (where the interference lives)
- `src/agents/training/CLAUDE.md` → Exploiter distillation (extraction: policy KL + FitNets + scalar/value distill), grad-balance (per-head trunk-pull cosines — the interference instrument)
- `designs/learning/self_discovered_archetype_latent.md` — z_arch pooling → VIB (the β = LUT-vs-style knob) → FiLM (the v8 capstone design)
- Memory: [[project_value_distill_fitnet]], [[project_double_sided_recipe]], `project_model_frontier_roadmap`, `project_archetype_competence_gradient` (the flat ~30% switch-axis = the gap in the wild)
