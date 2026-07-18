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

## See also
- [[objective_richness_and_representation]] — the simplicity bias / minimal-sufficient-statistic backbone (why the shared solution wins by default) + the distillation bits-ladder
- `src/agents/model/CLAUDE.md` → dual-head value readout, the shared trunk (where the interference lives)
- `src/agents/training/CLAUDE.md` → Exploiter distillation (extraction: policy KL + FitNets + scalar/value distill), grad-balance (per-head trunk-pull cosines — the interference instrument)
- `designs/learning/self_discovered_archetype_latent.md` — z_arch pooling → VIB (the β = LUT-vs-style knob) → FiLM (the v8 capstone design)
- Memory: [[project_value_distill_fitnet]], [[project_double_sided_recipe]], `project_model_frontier_roadmap`, `project_archetype_competence_gradient` (the flat ~30% switch-axis = the gap in the wild)
