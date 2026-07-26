# On-Policy Self-Distillation, Search-as-Teacher, and the Subset Exploiter

> **What this is.** A durable explainer for one concept cluster: **on-policy distillation (OPD)**
> as a training regime, why it's information-dense vs PPO, and how it maps onto *our* tooling — the
> `better-line` beam (a policy-improvement operator), the `search-teacher` AWR loss (which we
> upgrade to a full-distribution KL), cheap Gumbel-style search under our expensive
> `DamageOperator` critic, and the **exploiter-on-a-team-subset** as the place OPD pays off most.
> Intuitive first, then technical, no code. Grounded in `--exploiter`, `--search-teacher`, the
> `CorrectionBuffer`, `main/prober/better_line.py`, and `designs/ai_v6/design_search_teacher.md`.

---

## TL;DR

- **PPO gives ~1 bit per game** (win/loss + a few shaped scalars). **OPD gives a full target
  distribution at every state the student visits** — many bits per state. That density is the
  entire ~7-10× step-efficiency story (Thinking Machines Lab 2025; Qwen3 ~1/10 GPU-hours).
- **"On-policy" = the student grades itself on its OWN trajectories** — the exact inference-time
  states — so no train/test distribution shift, no exposure bias.
- **Our `better-line` beam IS a policy-improvement operator** (ExIt/AlphaZero lens; Grill 2020: the
  search step is an approximate KL-regularized policy improvement around the prior).
- **Upgrade `search-teacher`:** today it distills toward a single `A*` (AWR-weighted CE). OPD
  distills the **whole improved distribution `π'`** via `KL(π' || π_student)` on the student's own
  states — transfers ties/coverage, not just the argmax.
- **This dissolves the `V^{π*}`/GAE-bias pitfall we flagged:** the policy KL is a *classification*
  target off the on-policy path (never GAE); the critic is regressed only on **confirmed** returns.
- **Cheap search under our expensive critic:** Gumbel top-k over OUR actions (eval ~3-4, not 11) ×
  collapse the OPP axis to the believed-opp policy (expectimax over ~2-3 opp moves, not the 121
  matrix) × depth 1-2 + learned leaf value → `≈ k_our·k_opp ≈ 8` critic calls/node, not `121·depth`.
  Completed-Q credits the un-searched actions → guaranteed improvement even at ~2 sims.
- **Subset exploiter = the compute multiplier:** a narrow team slice ⇒ narrow state distribution ⇒
  critic + beam only need local accuracy ⇒ higher CI-gate yield, cheaper/deeper search, dense
  targets on a recurring state set ⇒ fast specialization. On-policy self-distilled specialist beats
  a pure-PPO exploiter *per unit compute* for exactly the OPD reason.

---

## Part 1 — What OPD is, and why it's information-dense vs PPO

### The 2×2 that makes it click

Two axes define the design space (Thinking Machines Lab's framing):

- **WHERE do trajectories come from?** student (on-policy) vs teacher/replay (off-policy).
- **HOW DENSE is the signal?** a full target distribution per step (dense) vs one scalar per
  trajectory (sparse).

| Regime | Trajectories | Signal |
|---|---|---|
| SFT / off-policy KD | teacher | dense |
| **RL / PPO** | student | **sparse (1 scalar/episode)** |
| **On-policy distillation** | **student** | **dense (per-step distribution)** |

OPD takes the best cell: the student trains on the exact states it hits at inference (kills
exposure bias — the SFT/off-policy-KD failure where the student drifts off the teacher's support),
and the teacher returns a **full probability vector** at each of those states.

### Why "information-dense" is not hand-waving

RL delivers a fixed, tiny number of bits per episode (~`log2` of the outcome space) no matter how
long the game was. **A whole Gen3 battle collapses to ~1 bit** (win/loss); even our shaped/PBRS
reward is a handful of scalars, on-policy but sparse and credit-assignment-hard (the recovery blind
spot below is exactly a credit-assignment failure). A per-state teacher distribution delivers *many*
bits per state — the whole vector over ~11 legal actions — so each trajectory teaches vastly more.
Consistent independent numbers:

- **Thinking Machines Lab** (Kevin Lu, Oct 2025): ~7-10× fewer gradient steps than matched RL,
  ~50-100× total compute; ~70% AIME at ~1,800 GPU-hr vs ~17,920 for RL. *(High confidence — fetched
  blog.)*
- **Qwen3 Technical Report** (arXiv:2505.09388): distillation matching/beating RL at ~1/10 GPU-hours
  via a two-phase off-policy-then-on-policy recipe. *(Verified.)*
- **GKD / "On-Policy Distillation of LMs"** (Agarwal et al., DeepMind, ICLR 2024, arXiv:2306.13649)
  and **MiniLLM** (Gu et al., ICLR 2024, arXiv:2306.08543) are the founding methods; MiniLLM
  established **reverse KL** (mode-seeking) as the default. *(Verified primary sources.)*

**Why popular *lately*:** the reasoning-model era made strong teachers cheap to sample; DeepSeek-R1
(arXiv:2501.12948) showed "distill a strong teacher into a small model" beats "RL a small model from
scratch"; and mechanically OPD is a **one-line objective swap** on existing RL/PPO infra (replace the
sparse env reward with a per-step teacher-KL) that reuses everything.

### The honest transfer caveat

LLM OPD is single-agent, autoregressive, perfect-info-per-token. **Ours is simultaneous-move,
imperfect-info, two-player.** A "teacher policy" is only well-defined relative to a
belief/information set, and a *fixed* teacher is exploitable. So we inherit the **principle** (dense
per-state targets on the student's own states) and the **default** (reverse-KL when a weaker student
imitates a stronger self), not the method wholesale. **Our teacher is our own beam search over the
critic** — a textbook "same model at higher test-time compute" OPD teacher.

---

## Part 2 — Apply OPD to our beam search

### The beam as a policy-improvement operator (ExIt / AlphaZero)

Given a base policy `π` (our PPO actor) and a critic `V`, a local search (MCTS, or our shallow
CRN-anchored beam in `main/prober/better_line.py`) produces a **locally better policy `π'`** + a
better value at the searched state, because it spends compute unrolling consequences and backing up
the critic. You then **distill `π'` back into the net** by supervised regression, and iterate =
approximate policy iteration. Grill et al. (ICML 2020, arXiv:2007.12509) proved the search step is an
**approximate KL-regularized policy improvement** around the prior — the same "improve-then-project
under a trust region" skeleton as MPO (Abdolmaleki et al. 2018) and the exp-advantage weighting of
AWR (Peng et al. 2019, arXiv:1910.00177). **Our `search-teacher` is the AWR branch of this exact
literature.**

### Upgrade: from single-`A*` AWR → KL to the full `π'`

Current Phase-2 policy term (`design_search_teacher.md` §1): `coef_pi · CE(π(·|s), A*)`, weighted by
confirmed Δwin. That's **mode-seeking on a single action** — it discards everything else the search
learned. The OPD move distills the **whole search-induced distribution** on the student's own visited
states:

```
L_opd = coef · E_{s ~ student rollouts} [ KL( π'(·|s) || π_student(·|s) ) ]
```

- **AWR-toward-`A*`** = "do this one move." Cheap, no explicit posterior, but discards the beam's
  distributional info, mode-collapsing / higher-variance.
- **KL-to-`π'`** = "here's the shape of good play here." Transfers second/third options, the beam's
  *confidence*, and its **ties** — which is what stops the argmax under-commitment we already measured
  (soft switch-prob 0.28 ≈ human 0.30, `project_opp_action_head_falsified`).
- **Forward vs reverse KL:** forward (`π'` as target, mode-covering) to transfer the whole improved
  distribution + uncertainty — the default when the beam is a *stronger* teacher and you want
  coverage; reverse (mode-seeking, MiniLLM) if you want the student to concentrate on the beam's peak
  and not chase its low-prob tail (weaker-student case). Start **forward-KL to `π'`**, keep
  AWR-toward-`A*` as the conservative fallback.

### Building `π'` from a beam (an improved *distribution*, not an argmax)

1. **Softmax over backed-up per-action values (cheap).** Our `lookahead` tool already yields `V(s')`
   per legal action. `π'(a|s) = softmax_a( logits(a) + β·Â(a) )`, `Â(a)` = backed-up advantage,
   `logits(a)` = prior actor logit. Anchored to the prior → a trust-region step. **This is the Gumbel
   completed-Q update** (Part 3) / Grill's exact regularized solution.
2. **Visit-analog (AlphaZero-faithful).** AlphaZero's target is `N(s,a)/ΣN`; the beam's
   retained-candidate mass × backed-up value is the analog. Construction (1) is cleaner for a beam and
   is what to ship.

**Completed-Q matters for us:** we only evaluate `k` actions (the critic is expensive), so fill `Q`
for the un-searched legal actions from the policy prior; then `π' = softmax(logits + σ(completedQ))`
is a **guaranteed improvement even at ~2 evals** (Danihelka et al., Gumbel MuZero, ICLR 2022) and
credits the actions you skipped.

### The value target — and the `V^{π*}`/GAE-bias pitfall, dissolved

**The pitfall (we flagged it):** the beam's backed-up value is `V^{π'}` — optimistic
(`V^{π*}`-flavored) *and* a best-response-to-a-fixed-opponent value, hence biased/exploitable. **Feed
it into GAE or the critic bootstrap and you bias the advantages** — GAE assumes the value is `V^π` for
the *behavior* policy, not `V^{π'}`. `design_search_teacher.md` names it: "search V is `V^{π*}` →
biases GAE," which is why it weights AWR by **confirmed Δwin**, not critic advantage.

**How KL-distilling the POLICY sidesteps it, by construction:**

- The policy term is a **supervised classification target** (`KL(π' || π_student)`) that never touches
  GAE, the rollout buffer, the clip objective, or the advantage. It's an aux loss beside the
  belief/win-prob/value-dist heads (the `instrumented_ppo._*_loss` pattern). No importance-sampling
  correction to get wrong — you're regressing a distribution, not doing off-policy policy-gradient.
- The value side stays where `search-teacher` already put it: regress the critic **only** toward the
  **rollout-CONFIRMED** return (Wilson-CI-gated realized win-rate of `A*` vs the EXACT reloaded
  opponent), *never* the beam's optimistic backed-up value (the demo: critic 95% vs confirmed 62% on
  Spore — distill 62%).

**Division of labor:** KL-distill the improved *policy* `π'` (safe — off the on-policy path); regress
the critic on *confirmed* returns (safe — the realized outcome). Neither injects `V^{π'}` into GAE.

### How this fixes the recovery blind spot

The critic **undervalues Recover/Wish** because the payoff is delayed/tail-risk — a credit-assignment
problem a sparse-reward on-policy critic learns slowly. **Search fixes it because search rolls it
out:** the beam / confirm-rollout plays `heal → survive → win` and *sees* the delayed return, then
hands back (a) a `π'` with real mass on Recover, and (b) a confirmed value for the healed state above
the critic's pessimistic estimate. Distill **both** — the dense per-state target carries the
credit-assignment info the scalar reward delivers too slowly. And distill the **confirmed** heal→win
value, not the beam's optimistic one.

---

## Part 3 — Better-than-naive-MCTS at the `11×11` cost

Our constraint: a heavy differentiable `DamageOperator` ⇒ **expensive critic/value forward**. Naive
PUCT-MCTS is the wrong tool — it minimizes *cumulative* regret, so at low budget it (a) may never
visit some root actions (breaks the visit-count target — no target for an unsampled action) and
(b) spreads its few evals wastefully. The fix: reframe "improve the policy from search" as a
**one-shot policy-improvement operator** costing `O(k)` critic calls.

### Four cheap levers (all liftable into our tooling)

1. **Gumbel top-k over OUR actions — eval `k`, not 11.** Nominate a small candidate set with the
   policy prior via **Gumbel-top-k without replacement** (`g(a)+logits(a)`, top `k≈3-4`), spend the
   critic budget only there. Guaranteed improvement at ~2 sims where MuZero fails below ~16
   (Danihelka 2022). `LegalActions` mask + actor logits give the prior free. Branching 11 → ~3-4.
2. **Completed-Q so un-searched actions get credit.** `q(a)=r+γV(s')` for the `k` searched (our
   `lookahead`); prior-fill the rest; `π' = softmax(logits + σ(completedQ))`, `σ` monotone. That's the
   `π'` KL-distill target, built from a handful of calls.
3. **Collapse the OPP axis — the big `121 → few` win.** The turn is a genuine `~11×11 = 121` joint
   matrix under hidden info. Don't solve it. Condition on a **believed-opponent policy** (from the
   move-belief head / `SPECIES_EXP_MULT` marginalized threat), take the opp's top `k_opp≈2-3` believed
   moves, expectimax over *our* action only. Our beam already **records the opponent at the divergence
   ply** + reloads-policy at interior plies — that IS the collapse (PokeChamp-style `b|a`, arXiv:
   2503.04094; their opp-model is only 13-16% top-1, so full-matrix solving is pointless anyway —
   *gen9/LLM, domain-shifted, directional only*). `k_opp=1` (single believed move) is the cheapest.
4. **Depth 1-2 + learned leaf value, CRN + state-clone caching.** Stop at depth 1 (`lookahead`) or 2
   (`better-line` default); the critic summarizes the rest (SoG `d=2`; ReBeL depth-limited). The
   `serializeBattle` clone-and-branch search server (`utils/bridge/search_session.py`, ~1.7ms/clone,
   *constant* in depth) + CRN anchoring is the depth>1 enabler.

### Eval-count math (per node)

- **Naive full matrix, per ply:** `11×11 = 121` critic evals; over depth `d`, `~121^d`.
- **Pruned (Gumbel top-k × opp-collapse):** `k_our × k_opp ≈ 4×2 = 8` per ply (or `4×1 = 4` with a
  single believed opp move). Depth-2, beam width `w≈4` → **~tens** of calls, not hundreds-thousands.

A **~15-30× reduction per node** before depth compounds — and completed-Q makes those ~8-12 calls a
*guaranteed* improvement worth distilling.

### Honest caveats for our setting

- The completed-Q **guarantee is for perfect-info, sequential, single-agent-per-node MDPs.** Here `π'`
  improves our critic's **belief-conditioned `V`**, not a true game value — a **belief-MDP
  improvement**; any "search value = improved value" claim must be **belief-averaged**, not a point.
- Collapsing the opp axis means the search **best-responds to a fixed model** → value is `V^π` vs
  *that* opponent, exploitable, **not the equilibrium value.** A sound per-node solve needs a
  regret-matching/Exp3 pass over the matrix with the critic as payoff oracle (SMMCTS arXiv:1310.8613;
  NN-CCE arXiv:2406.10411) — correct but 121 calls/node. Top-k × believed-opp is the right
  cost/soundness trade; **just don't claim Nash.** **SePoT** (Kubicek/Burch/Lisy, IJCAI 2024,
  arXiv:2312.15220) is the honest precedent (cheap depth-limited search on a trained net, explicit
  about non-best-response value error); **Student of Games** (Schmid et al., Science Advances 2023,
  arXiv:2112.03178) is the "if you want soundness under hidden info" reference.

---

## Part 4 — The exploiter-on-a-team-subset lens

The claim: **restricting the exploiter to a subset of teams makes search-distillation dramatically
more compute-efficient**, for four compounding reasons.

1. **Narrow slice = narrow state distribution.** Full Gen3 OU is enormous. (The former
   "~⅔ team-draw, uncoachable" claim is **RETRACTED** — it was model-judged, hence circular; see
   `feedback_no_circular_unwinnable_claims`.) Fix the teams ⇒ the reachable manifold collapses.
2. **Critic + beam only need LOCAL accuracy.** The critic doesn't have to be globally calibrated,
   only right on *this* matchup family — easier to reach, stays accurate ⇒ the beam's leaf values +
   confirm-rollouts are more trustworthy ⇒ **more searches pass the CI gate** (`teacher/yield` ↑) ⇒
   denser, higher-quality distillation coverage per unit compute.
3. **Cheaper/deeper search per situation.** On a narrow slice you can spend more of the fixed critic
   budget per node (wider beam, depth 2 vs 1, more confirm-rollouts) — you re-visit a small recurring
   set instead of paying for breadth. The beam deepens where it compounds.
4. **Fast specialization.** OPD's step-efficiency applies *per state visited*; when states repeat
   (narrow slice) the dense per-state targets accumulate fast.

### The loop

```
1. Exploiter plays ON-POLICY games vs the frozen strong target, restricted to the team subset
     (--exploiter, sole fixed target, init from a strong ckpt; --run-name models/<name>/).
2. On the exploiter's OWN visited crater states, run the cheap beam improvement
     (Gumbel top-k × believed-opp collapse × depth 1-2), building π' + confirmed values.
3. KL-distill π' into the exploiter + regress the critic on confirmed returns
     (rides the CorrectionBuffer + non-blocking frozen-snapshot workers already built).
4. The exploiter specializes fast on the slice → win_rate_vs_target climbs.
5. Its corrections REVEAL the target's blind spots on that matchup family.
6. Fold the specialist back into a league (baked corrections → league states/opponents).
```

Steps 2-3 are the **only new wiring**; `--exploiter`, `CorrectionBuffer`, frozen-snapshot workers,
and the beam all exist.

### Why the specialist beats a pure-PPO exploiter per unit compute

Pure-PPO exploiter: student trajectories, **sparse** scalar. Self-distilled specialist: same
trajectories, same state distribution, **dense** per-state `π'` targets. The *only* difference is
signal density — the axis where OPD reports ~7-10× fewer steps — and the subset makes it more extreme
because the dense targets land on a small recurring state set. **You don't change what it explores;
you change how many bits each explored state teaches** (a distribution/state vs ~1 bit/game).

Regime reading: DeepNash/R-NaD (Science 2022, arXiv:2206.15378) and **Metamon** (arXiv:2504.04395,
*our exact domain*) argue the plateau is a **regime problem** fixed by amortized-NE training + **team
diversity**, not test-time solving. The subset-exploiter is the surgical version: narrow diversity to
specialize + surface blind spots, then fold back (the league re-introduces diversity). It composes
with our one proven plateau lever (critic-shaping via win-prob + value-dist heads in shaping mode) —
OPD densifies the *policy* signal the way critic-shaping densified the *value* signal. **BC/human
replay is dead** (our agent exceeds the corpus), so a self-teacher, not a human teacher, is the play.

---

## Part 5 — A concrete, minimal first experiment

**Reuse:** `better-line` beam, `--exploiter`, `--run-name`, `CorrectionBuffer`, the non-blocking
frozen-snapshot workers, the 3-tier CI gate. **New wiring: one KL aux term + the `π'` construction.**

1. **Narrow slice.** 1-3 teams for the exploiter, one fixed strong `ckpt` as the target. `--exploiter
   --run-name opd_subset_01`, restricted to the subset, init from the **LATEST** ckpt (not a stale
   `final_model.zip` — `project_exploiter_league_tooling`).
2. **On the exploiter's own crater states,** run the beam but construct `π'` via completed-Q softmax
   (Part 3) instead of only extracting `A*`. Store `(obs, action_mask, π'_target, confirmed_value,
   weight)` in the `CorrectionBuffer` (one extra field: the full `π'` beside `A*`).
3. **Add ONE aux term** to `instrumented_ppo.train()`: `L_opd = coef · KL(π'_target || π_student)` on
   buffer minibatches; keep value = MSE to `confirmed_value`. Wire the grad-balance probe
   (`grad/opd_share`, `grad/opd_policy_cosine`) like every other aux head.
4. **A/B on the slice:** `coef=0` control (= current AWR-toward-`A*`) vs `coef>0` KL-to-`π'`. Success =
   `win_rate_vs_target` climbs **faster per gradient step** AND the recovery blind spot shrinks (probe:
   does Recover/Wish get real mass at safe-heal states?). Watch `teacher/yield` (should be *higher* on
   the narrow slice — the Part-4 point).

### When this hurts (honest failure modes)

- **KL-distill hurts if the beam is WORSE than the policy at a state.** OPD assumes a *stronger*
  teacher; a shallow depth-1-2 beam over an already-strong converged policy isn't uniformly better.
  Guard: distill only CI-gated confirmed-better `π'` (existing 3-tier gate); keep AWR-toward-`A*` for
  states where only the argmax is verified. If `grad/opd_policy_cosine` goes strongly negative, the
  teacher is fighting the actor — lower the coef.
- **Simultaneous-move / imperfect-info:** `π'` is a **believed-opp expectimax** improvement, **not**
  Nash. Exploitable; belief-conditioned. Confirm vs the exact reloaded opponent (we do); optionally
  confirm vs ≥2 pool opponents and down-weight lines that beat only one.
- **Imperfect-info belief in the rollout:** the interior-ply opponent past divergence is approximate
  (greedy, not the real stochastic sample). Distill the **first (divergence) action distribution** at
  full weight; treat deeper PV as soft/value-only (§9 edge cases).
- **Teacher cost:** the beam + confirm-rollouts are the expense; the subset restriction is what makes
  it affordable. Budget (`--teacher-search-budget`), prioritize by `|ΔV|·P(reducible mistake)`.
- **No known uncoachable ceiling (RETRACTED 2026-07-24):** the old "~⅔ of grind losses are
  uncoachable team-draw" bound was CIRCULAR (our own policy judged it) and is dead. The subset lens is
  still right — fixed teams shrink the state manifold — but do NOT budget around a matchup-lost floor.

### Shaky references to flag

- Danihelka completed-Q exact formula (`σ` definition, unvisited-action estimator) paraphrased from
  secondary summaries — verify against the primary PDF before quoting verbatim; the **shape**
  `softmax(logits + monotone(completedQ))` is well-attested. The "~2 simulations" figure is
  qualitatively attested, exact table not pinned.
- PokeChamp 13-16% opp-action accuracy is gen9 + LLM, domain-shifted from us.
- The self-distillation / Mean-Teacher (Tarvainen & Valpola 2017) / BYOL (Grill et al. 2020) lineage
  is background synthesis, **not** an LLM-OPD source.
- Value-guided-decoding IDs (arXiv:2503.02368, 2406.10858) and any 2025-2026 OPD follow-up IDs from
  the briefs were **not opened** — don't cite them load-bearing without a click-through.

---

## The synthesis

OPD is the best cell of the trajectory×density grid: **the student's own states (no distribution
shift) × a full target distribution per state (dense).** For us the teacher isn't an LLM — it's **our
own beam search over the critic**, a textbook "same model at higher test-time compute" OPD teacher.
The upgrade over `search-teacher` is to distill the **whole improved distribution `π'`** (KL) rather
than a single `A*` (AWR), transferring the beam's ties/coverage and keeping the `V^{π'}`/GAE-bias out
of training by construction (the policy KL is a classification target off the on-policy path; the
critic sees only confirmed returns). You get `π'` cheaply from a **handful** of expensive-critic calls
via Gumbel top-k over our actions × collapsing the opponent axis to the believed-opp policy
(≈`k_our·k_opp ≈ 8`, not 121), with completed-Q crediting the un-searched actions — a guaranteed
improvement even at ~2 sims. And the **team-subset exploiter** is where it compounds: a narrow slice
means the critic and beam only need local accuracy, so more searches pass the CI gate, the search goes
deeper per situation, and the dense per-state targets land on a small recurring state set. An
on-policy self-distilled specialist beats a pure-PPO exploiter for exactly the OPD reason — one
distribution/state vs one scalar/game. Just don't call the believed-opp improvement a Nash
equilibrium, and don't distill a beam that's worse than the policy.

---

## See also

- `designs/ai_v6/design_search_teacher.md` — the AWR search-as-teacher we upgrade (the two heads, the
  3-tier CI gate, the EXACT-opponent interior+confirm, the `V^{π*}`/GAE-bias rationale, selection,
  metrics, edge cases). **This note's Part 2 is the OPD generalization of that doc's §1 policy head.**
- Root `CLAUDE.md` → Prober (`better-line` beam, `lookahead`, `replay-counterfactual`), Training →
  Bot evaluation / self-play (frozen-snapshot workers, eval sharding), `--exploiter` / `--run-name`.
- `src/main/prober/better_line.py`, `utils/bridge/search_session.py` (clone-and-branch search server),
  `utils/bridge/counterfactual.py` (rollout-confirm), `agents/training/teacher/` + the
  `instrumented_ppo._*_loss` aux pattern + `CorrectionBuffer`.
- `designs/learning/marginalization_and_uncertainty.md` — why the believed-opp collapse must be
  belief-averaged (marginalize, not point-estimate); the `DamageOperator` that makes the critic
  expensive.
- Memory: `project_search_teacher.md` (selective ExIt, AWR toward A*, confirm-gate),
  `project_better_line_search.md` (the beam + clone-and-branch server),
  `project_exploiter_league_tooling.md` (`--exploiter` / `--run-name` / fork-on-resume gotchas),
  `project_plateau_research_2026_06_25.md` (regime-change hope: Metamon, team diversity),
  `project_positional_grind_decomposition.md` (whose "uncoachable team-draw ceiling" is RETRACTED — see `feedback_no_circular_unwinnable_claims`).
