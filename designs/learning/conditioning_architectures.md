# Conditioning architectures — FiLM, LoRA, MoE, what to condition on, and the sample economics

**TL;DR.** Conditioning is how ONE network holds MANY behaviours. There are three independent design
axes, and confusing them wastes runs:

1. **HOW you modulate** — FiLM (per-feature gain/shift, diagonal), LoRA (low-rank feature *mixing*),
   MoE (select an expert). FiLM and LoRA are **mathematically complementary** — neither contains the
   other — and can be summed. MoE is the wrong family for us (it splits samples K ways).
2. **WHERE you modulate** — head (reweight an already-computed summary) vs trunk (change what gets
   computed at all). Our FiLM is head-only; that is a real expressivity limit, and a more compelling
   argument for trunk conditioning than the diagonal-vs-low-rank one.
3. **WHAT you condition on** — our team (current `z_arch`), the *matchup* (our team ⊗ believed opp
   team), the board phase. Finer conditioning is more expressive but **quadratically more
   sample-hungry** — there is a bias/variance tradeoff in the conditioning granularity itself.

And the unifying economic fact: **conditioning does not add samples, it improves the SNR of the
per-team gradient by giving it its own subspace so it stops fighting the shared gradient.** Batch size
is a *noise* lever, not a *coverage* lever — and we already have the surgical version
(`--film-grad-accum-steps`) plus the metric that says when to use it (`film/noise_scale_ratio`).

---

## 1. The intuitive picture

Imagine one chef who must cook 719 different dishes.

- **Amortized (no conditioning)** — one recipe, averaged over all dishes. Edible, excellent at nothing.
  This is the generalist, and the averaging is the amortization gap.
- **FiLM** — the same recipe, but the chef gets a dial per ingredient: *more salt, less heat*. Cheap,
  and surprisingly powerful. What it **cannot** do is say "replace the salt with miso" — it can only
  turn existing ingredients up and down.
- **LoRA** — the chef gets a small set of *substitutions* ("swap dairy for coconut", "swap sear for
  braise") and z picks how much of each to apply. It can genuinely *mix* ingredients, but only along
  the few substitution directions you gave it.
- **MoE** — K entirely separate chefs; a router picks one. Maximum capacity per dish, but each chef
  only ever sees 1/K of the training, so each learns slowly. **This is exactly our binding
  constraint**, which is why MoE is a poor fit here.

The deeper point: FiLM adjusts *quantities*, LoRA adjusts *relationships*. Those are different
operations, which is why "LoRA is just a bigger FiLM" is false.

---

## 2. FiLM vs conditional LoRA — the actual mathematics

Write the modulation as a linear map applied to the feature vector `h`.

| | form | structure | what it can express |
|---|---|---|---|
| **FiLM** | `h ↦ diag(γ(z))·h + β(z)` | **diagonal**, full-rank | independent per-feature gain + shift |
| **LoRA** | `h ↦ h + B(z)A(z)·h` | **dense**, rank ≤ r | rotation / mixing inside an r-dim subspace |

**Neither is a special case of the other.** A diagonal matrix is generically *full rank*, so a rank-r
LoRA (r ≪ d) cannot reproduce it. A rank-r dense matrix has off-diagonal entries, which FiLM's
diagonal structurally cannot produce. They cut the space of linear modulations along different grains
— FiLM is *wide but axis-aligned*, LoRA is *narrow but free-form*.

**So "both" is a real option**, and cheap: `h ↦ diag(γ(z))·h + β(z) + B(z)A(z)·h`. You get per-feature
gain across *all* d features **and** genuine mixing in r directions.

### The generator cost — the practical constraint

Conditioning means a **hypernetwork** emitting the modulation from `z ∈ ℝ³²` (`ZARCH_DIM`):

- **FiLM** emits `2d` numbers. At d = 128: **256 outputs**. Small, well-conditioned, easy to learn.
- **Fully-conditional LoRA** emits `r(d_in + d_out)` numbers. At r = 8, d = 128: **2048 outputs** —
  8× larger, and generating a *matrix* from a 32-dim vector is an ill-conditioned regression. This is
  the real reason full conditional LoRA is awkward, not any deep principle.

### The cheap middle ground — and it may be the *right* architecture for us

Keep `A` and `B` **shared and learned** (not generated), and condition only the r-dim scaling between
them:

> `h ↦ h + B · diag(s(z)) · A · h`

Now the generator emits **r numbers** (say 8) instead of 2048. Interpretation: the network learns a
small basis of **adaptation directions** once, and `z` merely picks *how much of each* a given team
needs. It is LoRA-expressive (real mixing) at below-FiLM generator cost.

**Why this fits our evidence specifically:** the count sweep showed tight z-clusters *share gradient*
— 10 tight teams cost **less per team** than 3 loose ones. That is precisely the signature of teams
living on a **low-dimensional manifold of strategy directions**. A shared basis + conditional mixing
coefficients is the architecture that assumption implies. Worth remembering if the saturation
diagnostics ever say head-FiLM is the constraint.

---

## 3. WHERE you condition matters more than WHICH modulation

This is the argument I would rank *above* the diagonal-vs-low-rank one.

Our FiLM sits at the **two projection heads** — after the transformer, after CLS pooling. By then the
trunk has already computed one board representation and pooled it to a summary. Head conditioning can
only **reweight a summary that was computed team-blind**.

Consider two teams needing genuinely different *attention*: a trapper team must attend to the
opponent's escape options; a stall team must attend to PP, status, and recovery. If the trunk never
computed the escape-option feature (because it was optimising an average over all teams), **no amount
of head modulation can recover it** — the information was discarded upstream.

That is the real capacity argument for trunk conditioning (FiLM inside the transformer blocks, or
conditional LoRA on the attention projections, or — the most surgical — a **z-conditioned attention
edge bias** that changes *what attends to what*).

Ranking of interventions by how deep they reach:

| intervention | changes | cost |
|---|---|---|
| head FiLM (current) | how the summary is weighted | tiny |
| trunk FiLM | which features survive each block | small |
| conditional LoRA in trunk | how features are mixed | medium |
| z-conditioned attention bias | **what attends to what** | medium, most targeted |
| separate models | everything | no sharing — the thing we are avoiding |

---

## 4. `z_arch` over the OPPONENT belief — the matchup latent

Currently `z_arch = f(our 6 mons)`: team-static, deterministic, LUT-like. The natural extension is a
**matchup** latent `z(our team, believed opponent team)`, since we already run a belief over the
opponent's species/moves/spread (`BeliefSlots` / `BeliefHead`).

**Why it is well-motivated.** Piloting stall *vs offense* is a different game plan from stall *vs
stall*. Today one modulation per our-team must cover every opponent. And the "it is already in the
trunk" objection does not save us: the opp tokens *are* attended, but after CLS pooling the matchup
identity is compressed away — exactly the same reason our-team conditioning was needed at the head.

**Four honest problems, in order of severity.**

1. **Sample dilution is quadratic.** With K our-team clusters and K opponent clusters you have up to
   **K² matchups**. The entire difficulty we have been fighting is samples-per-conditioning-cell. This
   axis makes that dramatically worse. *Conditioning granularity is itself a bias/variance knob*:
   finer = less averaging bias, more estimation variance.
2. **The information arrives when it is least useful.** Early in a battle the belief ≈ the prior, so
   `z_opp` is near-constant across opponents — little signal exactly when a game plan must be chosen.
   By the time the belief is sharp, the game is often decided. (Testable: measure belief entropy vs
   turn, and the mutual information between the belief and the eventual outcome.)
3. **It is no longer team-static.** Our `z` is one vector per battle — cheap and stable. `z_opp`
   changes every reveal. Not fatal (it is still a deterministic function of the observation, so PPO's
   ratio recompute is fine — the v44 concern was VIB *sampling*, not time-variation), but it forfeits
   caching and the clean LUT interpretation.
4. **Compounding errors.** A wrong belief produces a wrong modulation; two learned systems now feed
   each other. GIGO risk.

**The cheap version to try first** (and the one I would actually run): condition on the opponent's
**archetype**, not a full latent — a handful of bits (offense / balance / semi-stall / stall) rather
than 32 continuous dimensions. It captures most of the matchup signal at a fraction of the sample
cost, and we already have the labels (`data/teams/gen3_team_archetypes.json`). **Coarse conditioning
first; refine only when the coarse version proves it pays.** That is the general rule this whole
programme keeps re-teaching.

---

## 5. The sample economics — and why "just raise the batch size" is not quite the answer

This is the part most worth internalising, because the intuition is subtly wrong.

### What actually determines per-team coverage

Naively: with 719 teams and a 2048 minibatch, each team gets ~3 samples — hopeless. But that is not
the structure. Teams are drawn **per episode per env**, and an episode is ~50–100 decisions:

- rollout = `n_steps × n_envs` = 2048 × 48 ≈ **98k samples**
- each env holds one team for an episode (or N episodes under `--team-block-episodes`)
- so a rollout covers roughly **48–200 distinct teams with ~500–2000 samples each** — and **zero** for
  the other ~500 teams

So the real picture is not "every team gets a trickle" but **"a subset of teams gets a lot, and a
given team is revisited only every ~15 rollouts."** That changes which lever matters:

| lever | what it actually changes |
|---|---|
| `--n-envs` ↑ | **more distinct teams covered** per rollout (parallel team draws) |
| `--team-block-episodes` ↑ | **fewer teams, more samples each** — concentration + the team is replayed right after its gradient lands |
| `--batch-size` / `--grad-accum-steps` ↑ | **gradient noise only** — does *not* change which teams are covered |
| `--team-pfsp` | **which** teams get covered (targeted at the weak ones) |

**The key correction: batch size is a noise lever, not a coverage lever.** Enlarging the batch averages
over the same team distribution more precisely; it does not visit more teams per unit of experience.

### The SNR view — the unifying idea

Per-team learning is a signal-in-noise problem:

- **signal** = the true per-team gradient direction
- **noise** = sampling variance, plus interference from the other 718 teams sharing the trunk
- SNR grows like `√(samples per team)` — so **doubling SNR costs 4× the samples**

This single relation explains everything we have observed:

- **Why exploiters work so well** — 100% of samples on one team is maximal SNR. Our 2M-step exploiter
  beat a 20M-step from-scratch net partly for this reason.
- **Why the generalist has an amortization gap** — 0.14% of samples per team, and we *measured* the
  consequence: `film/noise_scale` showed the per-team FiLM gradient sitting **below its own noise
  floor**.
- **What conditioning actually buys** — it does not add samples. **It reduces the interference term**
  by giving each team its own subspace, so the same samples produce a usable gradient. Conditioning is
  an SNR intervention, not a data intervention.

### The principled batch-size answer we already have

McCandlish's **gradient noise scale** `B_simple = tr(Σ)/|G|²` is the critical batch size — below it,
more batch buys ~linear progress; above it, diminishing returns. We log it:

- `train/noise_scale_ratio` — global. ≫1 = noise-limited (raise the effective batch); ≪1 = over-batched.
- `film/noise_scale_ratio` — restricted to the **FiLM generator parameter group**, because the global
  number cannot resolve whether a ~33k-parameter conditioning gradient is signal or noise inside a
  ~10M-parameter total.
- **`--film-grad-accum-steps`** — the surgical response: accumulate *only* the FiLM group's gradient
  across N optimizer steps so each applied conditioning update sees N× the effective batch, while
  everything else updates normally.

That is strictly better than raising the global batch: the conditioning path gets its big batch, and
the other ~10M parameters do not pay the memory/compute bill for a batch they did not need.

---

## 6. How to tell if FiLM has saturated

Ordered cheapest → most decisive.

**The principled ceiling first.** Every modulation is `(γ, β) = f(z)` with `z ∈ ℝ³²`, so the manifold
of achievable modulations is at most **32-dimensional**. That bounds the number of *mutually
independent* per-team adaptations at ~32 — **not** the number of teams. Teams needing similar
adaptations share a direction (which is why tight clusters are cheap). **The currency is independent
strategy directions, not team count.**

1. **Conditioning-ablation gap (best single test; offline, minutes).** For each team, run the model
   with its own `z` and again with a neutral/mean `z`; measure the policy divergence. A gap ≈ 0 means
   FiLM has *abandoned* that team. **The fraction of teams with a real gap is literally "how many
   teams FiLM is serving."**
2. **Modulation effective rank vs N.** Collect `{(γᵢ, βᵢ)}` over N teams and take the participation
   ratio. Plot against N; where it plateaus is the ceiling (≤ 32 by the argument above).
3. **z-collision.** Pair z-distance against the policy divergence the teams' specialists require. The
   saturation signature is the **"close z, far required policy"** quadrant. (Close-z-close-policy is
   the *healthy* clustering we want — the distinction matters.)
4. **`--zarch-dim` 32 vs 64 A/B.** Definitive, costs a run. Only worth it if 1–2 say we are near.

Live monitors: `zarch/pr` (z-cloud participation ratio), `film/{pi,vf}_team_std` (per-team
differential — if adding teams stops raising it, the generators are out of room).

---

## 7. What we actually know, and what to do next

**Established empirically (2026-07-22→24):**

| finding | evidence |
|---|---|
| head-FiLM has capacity to spare at N ≤ 10 | 1 team 0.84, 3 teams 0.835, 10 teams 0.825 — near-parity, and the residual closed with **more budget**, so it was sample-limited, not capacity-limited |
| multi-team is **sub-linear**, not linear | 10 tight teams reached parity at **1.24M steps/team** vs the 3-team's 2.1M/team — tighter cluster ⇒ more shared gradient |
| conditioning yields **true specialization**, not averaging | the 3-team model matches the dedicated `564` specialist at **92% agreement on confident decisions**, JS 0.048 |
| warm fork ≫ from scratch | 0.84 @2M vs plateau ~0.65 @20M, and the fork was *less* overcommitted (entropy 1.60 vs 1.22) |

**Therefore `z`-LoRA / wider `z` is NOT indicated yet.** We have no evidence of a capacity ceiling.
Reaching for more expressivity now would be solving a problem we have not demonstrated.

**Priority order (expected gain ÷ cost):**

1. **Distill the multi-team teachers → the generalist, then the retention ablation.** Converts the
   exploiter work into generalist strength *and* answers whether teachers must stay forever.
2. **`--team-pfsp onesided` in generalist self-play.** The concentration lever, already shipped; it is
   also a *homeostat* against per-team regression.
3. **Run the saturation diagnostics (§6).** Cheap, offline, and they are the *gate* for everything
   below. We keep being wrong about capacity in both directions — measure before building.
4. **Coarse matchup conditioning** (opponent archetype bits, not a full `z_opp`) — only after the
   sample-cost question in §4 is understood.
5. **Trunk conditioning** (FiLM-in-trunk, then shared-basis conditional LoRA) — only if §6 says
   head-FiLM is the constraint.

---

## Synthesis

Conditioning is not one knob but three: **how** you modulate (FiLM's per-feature gains vs LoRA's
low-rank mixing vs MoE's selection — the first two complementary and summable, the third
sample-hostile), **where** you modulate (head reweights a team-blind summary; trunk changes what is
computed at all — usually the bigger constraint), and **what** you condition on (our team today; the
matchup tomorrow, at quadratic sample cost).

Underneath all three sits one economic law: **conditioning buys SNR, not samples.** It works by giving
the per-team gradient its own subspace so it stops being drowned by the other teams — which is why
FiLM produced *true* specialization (92% agreement with a dedicated specialist) rather than a blurred
average, and why tighter clusters are *cheaper* per team rather than merely equal.

Which means the discipline is always the same: **measure the constraint before adding expressivity.**
Our own record is the argument — we predicted dilution at 10 teams and got parity; we predicted a
warm-fork bias and found the fork strictly better. Both times the architecture had more room than we
guessed, and the real limit was signal.

---

## See also

- `designs/learning/amortization_gap_and_conditioning.md` — *why* conditioning is needed (the
  generalist/specialist gap, storage vs signal, the FiLM design rationale).
- `designs/learning/self_discovered_archetype_latent.md` — the `z_arch` latent itself.
- `designs/learning/regularization_and_noise_in_ppo.md` — noise scale, batch sizing, PPO specifics.
- `designs/learning/on_policy_self_distillation.md` — dark knowledge, why distillation transfers so
  much more per sample than RL.
- `designs/ai_v8/exploiter_batch_strategy.md` — the z-cluster targeting + the **retention ablation**
  protocol (acquisition vs retention).
- `src/agents/model/CLAUDE.md` → Team-archetype latent + head FiLM (v44); `src/agents/training/CLAUDE.md`
  → z_arch aux, team-PFSP, gradient noise scale.
- Memory: `project_exploiter_fork_vs_scratch` (the count sweep + fork-vs-scratch results).
