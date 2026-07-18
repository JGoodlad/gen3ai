# Self-discovered team-archetype latent (z_arch, VIB, β, UMAP)

**TL;DR.** We want the policy/value to condition on *what style of team it's piloting* — a low-dim
"team vibe" vector **z_arch** — so it stops playing the team-blind *marginal* optimum (see
[[generalist_specialist_amortization_gap]]). The hard part isn't building the token, it's making it
*mean* something without hand-labels or a fixed number of archetypes. The lever set: **pool** the team
tokens into a distribution, **VIB** compresses it (squeeze out incidental detail, keep only what helps
win), **β** tunes how hard you squeeze, and **UMAP** (diagnostic only) lets you *see* whether it
self-organized by play-style. The purest version has **no target at all** — the RL task + a
compression penalty are the only pressures, so the model carves its own archetypes.

**Status (2026-07-17): v1 is BUILT as `gen3_zarch_film_v1` (v44, `--zarch-film heads`), with three
deliberate deviations from this note's original spec, each evidence-driven** (see
[[amortization_gap_and_conditioning]] for the probes): **(1) the z source is NOT a CLSPool query over
the trunk's team tokens** — the team pool probed core-dominated (flex twist ~chance) and a
per-decision trunk read has 3× the archetype flip rate of a static code; v1 builds z from a dedicated
DeepSets encoder over the obs's INVARIANT per-mon facts (species/item/ability/moves/spread, detached
embedding reads → zero trunk gradient interference). **(2) v1 is DETERMINISTIC — no VIB sampling**:
a per-forward reparam sample breaks team-static (per-decision z jitter), adds noise to PPO's
epoch-recomputed ratio, and breaks eval determinism; and the chosen LUT-first operating point (β→0)
needs no rate limiter — anti-collapse comes from composition-reconstruction + a VICReg variance floor
instead (VIB is the *style-generalization phase's* tool, and needs per-EPISODE sampling when it
lands). **(3) FiLM is applied at the two root heads only** (post-projection pre-ReLU, zero-init) —
trunk-token FiLM is the follow-up A/B (`--zarch-film trunk`, not yet built). The β/rate-distortion
theory below is kept as the design record for that later phase.

## Why a latent at all (the connection to the marginal trap)

Self-play averages the gradient over teams, so a *team-conditional* strategy gets washed out by the
universally-applicable "click the effective move" signal → the policy amortizes to a team-blind
marginal optimum. A z_arch that the heads are *forced* to read is the **explicit** conditional
representation the league only builds *implicitly*. Handing it a "what style am I piloting" signal
attacks team-blindness directly. (The league forces the representation by covering diverse
teachers/opponents; the latent gives it a place to live. Complementary.)

## The self-discovery spectrum (how much *we* specify vs how much *emerges*)

Ranked from most-specified to most-emergent:

- **Behavioral-signature grounding** — a head predicts the team's *observed* stats (switch/setup/heal
  rates, tempo, win-rate vs pool), regressed toward the EMA'd measurements. Self-supervised (targets
  are observed, not labeled) but *we* pick the stats. Reuses the team-PFSP per-team accumulator.
- **Self-predictive / successor (recommended grounding)** — z_arch predicts the team's *own future*:
  future latent states (SPR/BYOL, stop-grad target) or expected discounted future features (successor
  representation). No hand-picked stats — the model discovers "what *game* this team produces." Grounds
  in **dynamics/style** while staying emergent.
- **Pure task-driven bottleneck (maximally self-discovered)** — *no target.* z_arch is shaped only by
  (1) it must help the policy/value (via FiLM) and (2) it must be compressed (VIB). The task decides
  what team-info is worth keeping; archetypes emerge as density. Risk: may carve along an axis you
  wouldn't name "archetype" (speed? phazer-present?), or not organize at all if raw team tokens already
  suffice. That's the price — and arguably the point — of authenticity.
- **Reconstruction (autoencode the team)** — self-supervised, but captures *composition* not *play*.
  Warm-start prior at best.
- **Diversity-only (VICReg/Barlow) + FiLM** — force z_arch high-variance + decorrelated across teams +
  useful. Very emergent, least controlled (may latch onto spurious differences).

**Recommendation:** pure task+VIB+FiLM *backbone* + a *light* self-predictive grounding, so it
self-discovers along the **dynamics** axis rather than a random one — then inspect (UMAP) and be open
to its carving differing from human archetypes.

## The concepts

### z_arch — the latent
A short vector (d≈32–64) = a *style fingerprint* of the team (aggressive vs grindy, fast vs bulky,
setup vs pivot), NOT the 6 mons in detail. It's a *learned coordinate*: its axes mean whatever the
training pressure makes them mean — which is why the objective choice dominates the design.

### Pooling ("packing" 6 tokens → 1 vector)
The transformer emits 6 per-mon tokens; a learned **query token** (like our `our_cls`/`value_cls` in
`CLSPool`) does **attention pooling** — it attends over the 6 and pulls a weighted summary. Learns
*which* mons/features matter for style (vs mean-pooling, which just averages). z_arch is one more
`CLSPool` query. (Linear inspection cousin: **PCA**.)

### VIB — Variational Information Bottleneck (the compressor)
*Telephone game:* force the team info through a **noisy, narrow pipe** so only information worth
transmitting (because it helps the task) survives; incidental detail is squeezed out. That squeezing
turns "all the team's info" into "the team's *essential style*."
*Mechanics:* encode the team to a **distribution** `N(μ,σ)`, **sample** z_arch (reparameterization
trick → gradients flow), add `KL(N(μ,σ) ‖ N(0,I))` to the loss. KL = **rate** (bits used); task loss =
**distortion**. Minimize both → fewest bits that still help → a compressed, smooth, *continuous*
manifold (no fixed K; archetypes = density).

### β — the squeeze knob (β-VAE / β-VIB)
`loss = distortion + β·KL`.
- **β high** → squeeze hard → tight/disentangled latent, but too far → **collapse** (z_arch carries
  nothing, μ→0 σ→1).
- **β low** → keep lots → rich but barely compressed (little archetype structure; it just copies the
  team).
The single most important hyperparameter: anneal/sweep β, watch (a) latent stays alive (σ not →1
everywhere) and (b) per-archetype play improves.

### UMAP — seeing whether it worked (diagnostic ONLY)
z_arch is 32–64-D; UMAP squashes it to 2-D **preserving neighborhoods** so you can *look* at whether
teams cluster by style. (Manifold learning; t-SNE-like but better global structure; PCA = the linear
version.) **Never a training signal** — color the 2-D points by the *known* PACE labels *after*
training to check "did it self-organize by style without ever seeing labels?" If stall clusters apart
from hyper-offense unsupervised, the self-discovery worked.

### FiLM — forcing the heads to USE it
z_arch generates `(γ,β_film)` that *modulate* the pi/vf features: `h' = γ(z_arch)⊙h + β_film(z_arch)`.
**Apply it to BOTH heads (separate γ/β generators, shared z_arch) — and the VALUE head is arguably the
better target.** Value is intrinsically archetype-conditional (the SAME board is "winning" for stall,
"losing" for offense), so an unconditioned critic averages them into the blurry ~4-effective-dim marginal
we measured (`value_cls` rank flat `_14→_18→_19`). Value-FiLM is the escape-the-marginal move for the
critic, and it's the **architectural complement to VALUE DISTILLATION** (`--distill-value-coef`):
distillation supplies the per-team value TARGET, FiLM supplies the CAPACITY to represent it — do only one
and it under-delivers (distill w/o capacity can't de-blur; FiLM w/o supervision = learns≠helps); together
they de-blur the critic per archetype. Value-FiLM also helps the policy indirectly (better critic → better
GAE). Gate: policy-FiLM by per-archetype win-rate, value-FiLM by per-archetype value CALIBRATION +
`value_cls` rank rising off ~4.
Multiplicative + in the main path → can't be ignored (a *concat* input can be zeroed out and washed
away; a multiplicative γ puts z in the **control path** — the same board features produce different
downstream activations per z). Identity-at-init (γ→1, β_film→0) → OFF byte-identical. (Not the VIB β.)

**FiLM vs the marginal trap (the point).** FiLM does **not** change the objective — per-team-optimal is
already the objective's optimum; the marginal is a *stuck point*. What FiLM changes is the gradient:
without conditioning, one policy π(a|s) has its gradient **averaged over teams** (team A wants X, team B
wants Y in the same state → the update is the *blend*). With FiLM, the policy is a **family indexed by
z_arch** — π(·|s,z_A) updates toward X and π(·|s,z_B) toward Y *separately*, so per-team gradients stop
collapsing into a marginal blend. That is the escape mechanism.
**But FiLM is the ENABLER, not the cause** — it stays marginal unless (1) z_arch actually *separates*
teams (the VIB grounding — else same z → same modulation → one policy) and (2) there is *pressure* to
differentiate (diverse opponents + distillation — else it learns γ≈1 and ignores z). FiLM makes
per-team the *easy* path; the league/distillation supply the reason to take it.
**β = the memorize-vs-generalize dial (and a LUT is NOT inherently bad).** A rich z_arch lets FiLM
condition on team *identity* (≈ a per-team lookup table). For a FIXED pool you only ever pilot, with
enough data per team, a LUT *is* the per-team optimum — same seen-team ceiling as a style code, and it
still escapes the marginal. It's a spectrum, not a binary; low-β "LUT with a little generalization" is a
legitimate operating point. A LUT costs you only where **sharing/generalization** matters — and those
are exactly our goals: (1) the distillation point — distil ONE stall exploiter and lift ALL stall teams;
a LUT makes each distilled team an island (one exploiter per team). (2) The opponent side — its team is
hidden/partial/always-novel, so there's no entry to look up; you *must* infer style. (3) Sample
efficiency even closed-world — a LUT learns 719 separate problems and starves the rare-team tail; a style
code amortizes. (4) Novel/meta teams. So β isn't "avoid a LUT" — it's "how much do I need to *share*";
our use case (distil→archetype, opponent inference, tail coverage) is share-heavy → lean style.

## How they compose
Team tokens → **attention-pool** into `N(μ,σ)` → **VIB** samples z_arch, **β** squeezes → the **task**
(+ optional self-prediction) shapes *what* survives → **FiLM** forces the heads to condition on it →
**UMAP** reveals the emergent archetypes afterward. No labels, no fixed K, self-discovered.

## Where this lives in our stack (v1 as-built — supersedes the original CLSPool sketch)
**Shipped (v44, `gen3_zarch_film_v1`):** `ZArchEncoder` (a dedicated static-atom DeepSets encoder in
`features_extractor.py` — NOT a CLSPool query; see the TL;DR status note for why) → deterministic z →
zero-init `film_pi`/`film_vf` on the root heads. Flags `--zarch-film {off,heads}` + `--zarch-dim`
(default 32), coefs `--zarch-recon-coef`/`--zarch-vicreg-coef` (training-only, auto-zeroed on
single-team runs), OFF byte-identical, version-gated in `check_compatible`
(`MODEL_CONFIG_VERSION` 44), **leak-trivial** (our own public roster). Monitors: `zarch/std`
(collapse), `zarch/recon_topk_acc`, `film/{pi,vf}_{gamma,beta}_norm` (deviation-from-identity = is
FiLM alive). Validation gate = *does conditioning improve per-archetype win-rate* (the team-PFSP
win-rate history) and *does distilling an anchor now lift its neighbors without regressing the rest* —
NOT "does z_arch predict archetype." Model details: `src/agents/model/CLAUDE.md` → Team-archetype
latent + head FiLM.

## Implementation: FiLM via VIB + low-rank
- **VIB:** the `archetype_cls` pool emits `(μ, logσ²)`; sample `z_arch = μ + σ⊙ε` (reparam), penalize
  `β·KL(N(μ,σ)‖N(0,I))`. z_arch is deliberately **low-dim** (~16–32) — that IS the bottleneck.
- **Low-rank FiLM:** generate the modulation with a **single linear** map `z_arch → (γ,β_film)`.
  Because z_arch is low-dim, the per-feature scale/shift lives in a **≤dim(z_arch) subspace** of the
  feature space — i.e. the conditioning is intrinsically **low-rank** (LoRA-flavored): few params, and
  the modulation can only move features within a small subspace → strong regularization against
  memorizing per-team quirks. `γ = 1 + Δγ(z_arch)`, `β_film = Δβ(z_arch)`, both zero-init → identity
  at start (OFF byte-identical). Two knobs compose cleanly: **VIB β** sets *what* z_arch encodes; the
  **rank** (= dim z_arch) sets *how expressively* it can reshape the heads.

## Measuring the right compression (tuning β / latent dim)
"Correct compression" = z_arch encodes *style* at the granularity that **generalizes**, without
collapsing (too much) or memorizing team-ID (too little). Three tiers, cheap → definitive:

- **Geometry (cheap, per-checkpoint proxy).** Embed all pool teams; take canonical style anchors from
  the clean sample teams.
  - *Separation* (anchor-to-anchor distance) is necessary but **incomplete** — it says how far styles
    sit, NOT how tightly teams cluster (the "how many nearby" gap).
  - The fix is a **ratio**: between-style separation / within-style spread (**silhouette score**, or
    Davies-Bouldin, or the LDA F-ratio) — captures both "how far apart" (numerator) AND "how tight /
    how many nearby" (denominator).
  - **kNN purity above chance:** per team, the fraction of its k nearest neighbors sharing its
    (held-out) style label, minus the style's base rate — directly answers "are the nearby teams the
    same style" (mirrors our belief `*_above_chance` metrics).
  - *Density per anchor* (# teams within radius r + neighborhood overlap) → how populated & distinct
    each style is. *Effective modes* (participation ratio / effective rank of the z_arch cloud) → the
    **emergent number of styles**, with no K imposed.
  - **TODO when we build the archetype token:** add a `rank/archetype_cls_*` metric (participation
    ratio / effrank / n90-n95) alongside the existing `rank/{trunk,value_cls,policy}_*` probes
    (`rank_metrics.py`) — the effective rank of the z_arch cloud IS the live "emergent number of
    styles" read, and watching it vs β is the cheapest per-checkpoint compression gauge.
- **Rate-distortion elbow (info-theoretic, from training).** VIB gives rate (KL = bits) + distortion
  (task loss). Sweep β, plot rate vs distortion; the **knee** — where more bits stop cutting task loss
  — is the natural compression (beyond it you're memorizing). Read straight off training metrics.
- **Downstream (definitive, expensive).** Gold standard: sweep β, measure **per-archetype win-rate**
  (team-PFSP history) + a **held-out generalization** test (hold out teams; do they land near their
  style / does conditioning help them?). The β that maximizes downstream conditional performance IS the
  correct compression; everything above is a proxy for it.

Recipe: geometry (silhouette + kNN-purity) to narrow β *fast* per checkpoint; confirm finalists with
the downstream A/B.

### Dynamically tuning β (and why you can't tune it on the training fit)
β is a **regularizer**, so the iron rule applies: **you cannot tune it on the training loss** — a LUT
*fits training teams better* than a style code, so minimizing training distortion slides β *down toward
LUT* every time. The signal that pulls toward *style* must be a **generalization** signal (held-out /
transfer) — the only place LUT loses. The knob and the metric are inseparable, and the metric has to be
held-out.
- **Static (MVP):** fix β at the **rate-distortion elbow**; watch `rank/archetype_cls_*` (extend the
  `rank_metrics.py` probe) as the live LUT-vs-style gauge — high effective rank ≈ LUT, low-but-alive ≈
  style, ~0 = collapsed.
- **Dynamic:** a **Lagrangian/PID controller (GECO, "Taming VAEs")** makes β a *dual variable* that
  targets a level: slack → raise β (squeeze to style), hurting → lower β (keep more). Drive it with a
  **generalization** signal — cheapest = a target **effective rank** of z_arch (self-catches collapse),
  truer = held-out **kNN-purity-above-chance**, gold = **downstream per-archetype win-rate on held-out
  teams** (EMA heavily; a PID on a noisy non-stationary RL signal needs strong smoothing). **Never
  target the training distortion.**

One sentence: β sets a point on the rate-distortion curve, but "correct" is defined only by a *held-out*
signal — fit and generalization pull β in opposite directions, and only generalization tells style from
a lookup table.

## Synthesis
z_arch is the model's *own* answer to "what am I piloting." Grounding it with hand-picked stats works
but imposes our ontology; the more self-discovered route (task+VIB backbone, self-predictive grounding)
lets the model carve its own style-space and is the more honest fix for team-blindness — it's the
policy self-organizing the conditional structure the marginal self-play never demanded. β is the dial
that decides whether that space is a crisp disentangled map or a soft copy of the team; UMAP is how you
check it landed on *style* and not noise.

## See also
- [[generalist_specialist_amortization_gap]] — the marginal trap this latent attacks.
- [[latent_belief_metrics_and_collapse]] — collapse/std diagnostics + VICReg, reused here as the
  anti-collapse guard.
- `src/agents/model/CLAUDE.md` — `CLSPool`, `ProjectionAssembler`, the belief-head arch-toggle pattern.
- `src/agents/training/CLAUDE.md` → Team-side PFSP — the per-team accumulator the behavioral grounding
  (and the validation win-rate history) reuse.
