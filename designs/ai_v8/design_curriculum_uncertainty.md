# Design Note — Value-Free Curriculum + Uncertainty (escaping the value-function bootstrap)

**Status:** PROPOSAL / post-Rust-sim, post-balanced-base. Not built. Captures the curriculum +
uncertainty thread (2026-07-05 discussion). Companion to `design_opponent_system_parity.md` and the
archetype-conditioning direction.

## The problem this solves

Measured (ai_v7_02 + ai_v6_13, replicated): the model has **no archetype-conditioned game plan** — the
critic is **blind to defensive positions** (AUC ≈ 0.50, vs ~0.63–0.71 on offense) and the policy plays a
single board-reactive style regardless of team. Root cause is the PPO objective: it maximizes *average*
return, the offense-heavy + floor-carried distribution is satisfiable with a low-rank board summary, and the
archetype floor **zeros the advantage signal** on defensive games (win regardless of action → A≈0 → no
gradient → never learns to read them). See the calibration + behavioral analyses (`tmp/` scripts, gitignored).

The fix stack is: **(1) contested opponents to un-zero the advantage → (2) balanced/DRO exposure to aim it →
(3) archetype conditioning (FiLM/routing) for capacity.** This note covers the *curriculum* that drives (1)+(2)
and the uncertainty machinery it needs.

## Principle 1 — sample by REGRET / LEARNING PROGRESS, not difficulty

"Play harder matchups" done naively = maximize difficulty = **ai_v7_05** (fixed 1998-ELO opponent → unwinnable
→ stalled). Wrong target. The right target is the **frontier**: games won ~40–60%, where outcome depends on
the action so **advantage variance (∝ p(1−p)) is maximal**. This is the zone-of-proximal-development / PLR /
PFSP principle. Sample where **learning is happening**, not where difficulty is highest.

## Principle 2 — the curriculum circularity, and the value-free escape

**The trap (owner-identified):** to sample "contested" games you must *estimate* contestedness — and if that
estimate is the critic's P(win), the curriculum **inherits the critic's blindness** (it calls every defensive
game ~0.5, can't separate contested from unwinnable). You'd need a good value function to build the curriculum
that fixes the value function.

**Escape — drive the curriculum with VALUE-FREE signals:**
- **Empirical win-rate** (measured outcomes, not predicted) — what PFSP already uses. Value-function-free at
  the opponent/team/**style** level.
- **Learning progress** (the empirical *derivative* of win-rate per style over time) — the rigorous ACL answer.
  Never queries the critic. Distinguishes contested (rising) from unwinnable (flat) that the blind critic can't.
- **AUC-by-style is a MONITOR, not a driver** (it's value-based → circular if used to drive).

The value function becomes a *consequence* of the curriculum, not a prerequisite. **State-level** targeting
(restart-from-hard-positions) *does* need per-state difficulty → genuinely circular → deferred until the critic
is bootstrapped partway (or fed by the ensemble/search below).

## Component A — value-free uncertainty via a randomized-prior / diverse-TRUNK ensemble

Ensemble disagreement is a value-free "we don't know if this is winnable / this is novel" signal — for the
curriculum *and* as an exploration bonus (disagreement-based exploration, Pathak).

- **Diversity must reach the TRUNK.** Cheap fakes (multi-head on a shared trunk, MC-dropout) inherit the blind
  trunk → members *falsely agree* "defensive = 0.5" → zero disagreement exactly where we need it. Design
  constraint: diversity deep enough to include the trunk.
- **Better than a naive N-net ensemble: Randomized Prior Functions (Osband) / epinets.** Each member =
  trainable net + a *fixed random prior*; priors differ → members disagree *by construction* in unvisited
  regions (= surprise), agree where visited. RL-tested, principled, far cheaper than N heavy models.
- **Cost control:** 3–5 members with diverse trunks + random priors on a **distilled light critic** (NOT the
  DamageOp-heavy model). Keeps diversity deep, cost bounded.

## Component B — failure-latent + KNN retrieval curriculum ("family of surprise")

When the model is surprised/beaten, retrieve the *neighborhood* of that failure and drill the whole family so
the fix **generalizes to the vulnerability class**, not the one team.

- **Embed the FAILURE, not the roster** (reframe). Two different teams can produce the same surprise; the same
  team can produce different surprises. KNN in **failure/threat-space** (`(position, surprise-mechanism)`) or the
  model's own belief/value representation ("teams it treats alike") — not pure composition.
- **Value-free surprise triggers:** belief-error (predicted opp move X, got Y) and surprise-OHKO. NOT
  value-prediction-error (circular — the blind critic won't register defensive surprises).
- **Hard prerequisite: the pool must CONTAIN the family.** KNN only retrieves what exists; the 3:1-offense pool
  can't surface an underrepresented defensive/gimmick family → depends on **pool diversity** (balanced data /
  human replays).

## The synthesis loop

**Detect** (ensemble disagreement / value-free surprise trigger) → **Retrieve** (failure-latent KNN → the
family) → **Drill** (fast Rust sim makes drilling the family affordable) → **Generalize** (patch the class) →
the advantage signal reappears → critic co-improves → curriculum becomes progressively value-trustworthy. Both
the detector and the retriever are value-independent, so the loop **sidesteps the bootstrap** rather than trying
to escape it from inside a broken critic.

## Prerequisites, acceptance metrics, timeline

- **Prereqs:** diverse team pool (balanced data / human replays); team/failure latent + retrieval index;
  value-free surprise detector; Rust sim (clone/determinize + cheap drilling); a distilled light critic for the
  ensemble.
- **Acceptance metrics** (all floor-robust): fraction of games at ~40–60% WR (are we on the frontier?);
  **defensive AUC** climbing 0.50 → 0.65+ (critic un-blinding); **learning-progress per style** (is the
  curriculum finding live gradients?). NOT raw difficulty or raw win-rate (floor-confounded).
- **Timeline:** post-Rust-sim, after the balanced-styles base + archetype conditioning. This is the mature-system
  curriculum, not an immediate lever. The immediate levers remain: restore the advantage (contested opponents),
  balanced/DRO exposure, and exogenous data (human replays) as the value-independent break-in.
