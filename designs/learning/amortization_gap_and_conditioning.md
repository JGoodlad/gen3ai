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
| **Storage** — hold conflicting behaviors without cancelling | **conditioning: FiLM on `z_arch`** | per-archetype affine gain/bias on the trunk features ⇒ per-team gradients modulate *different* subspaces instead of overwriting one shared weight | **the gap — not built (v8)** |
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

**The falsifiable precondition.** If the per-team optima *don't actually conflict much* (the specialists
mostly agree; the gap is really just extraction/SNR), then a better critic *would* mostly close it and FiLM
would buy little. This is **testable, now**: `tmp/subspace_overlap_probe.py` measures how orthogonal the
archetype representation subspaces are (orthogonal ⇒ storage conflict is real ⇒ FiLM has room; overlapping
⇒ less conflict ⇒ FiLM buys less), and the `_21` **strength gate** tells us whether the un-crystallized
critic *alone* starts closing the generalist-vs-specialist gap. If it does, storage isn't binding; if it
stalls with a rich critic, storage (FiLM) is the wall.

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

## See also
- [[objective_richness_and_representation]] — the simplicity bias / minimal-sufficient-statistic backbone (why the shared solution wins by default) + the distillation bits-ladder
- `src/agents/model/CLAUDE.md` → dual-head value readout, the shared trunk (where the interference lives)
- `src/agents/training/CLAUDE.md` → Exploiter distillation (extraction: policy KL + FitNets + scalar/value distill), grad-balance (per-head trunk-pull cosines — the interference instrument)
- `designs/learning/self_discovered_archetype_latent.md` — z_arch pooling → VIB (the β = LUT-vs-style knob) → FiLM (the v8 capstone design)
- Memory: [[project_value_distill_fitnet]], [[project_double_sided_recipe]], `project_model_frontier_roadmap`, `project_archetype_competence_gradient` (the flat ~30% switch-axis = the gap in the wild)
