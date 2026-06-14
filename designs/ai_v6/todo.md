# AI v6 — Todo

---

## Step 1 — Replay Collection ✓ DONE

Passive spectator client that connects to Showdown, discovers active Gen 3 OU battles,
and saves the complete battle logs for offline use. See
`designs/ai_v6/impl_step1_replay_collection.md`.

**Deliverables:**
- `src/poke_env/spectator/` — `BattleSpectator` async generator, `SpectatedBattle` pure data object
- `src/main/collect_replays.py` — long-running daemon with Rich dashboard
- `src/poke_env/spectator/spectated_battle_test.py` — 9 unit tests
- `src/poke_env/spectator/spectator_e2e_test.py` — live server test + round-trip log parse

**Status:** Daemon actively collecting at `replays/showdown/1/`.

---

## Step 2 — Behavioural Cloning

Pre-train the policy on (observation, action) pairs extracted from human replay logs, then
hand off to RL fine-tuning. Gives the agent a strong human prior so RL exploration starts
from a competent baseline rather than random. See `designs/ai_v6/impl_step2_bc.md`.

**Design questions to resolve:**
- **Mask synthesis**: spectated logs lack `|request|` JSON (no PP data). Options: skip
  turns where exact options are unknown, or synthesise a best-effort mask from the known
  moveset and infer PP from move-use history.
- **Class imbalance**: switch actions are far less frequent than move actions — weighted
  sampling or focal loss needed.
- **Stopping criterion**: monitor validation loss to avoid mode collapse (overfit to the
  most common move). Early stopping when val loss plateaus or rises.

---

## Step 3 — Team Completion Model

A masked-slot prediction model (BERT-style) that, given the opponent's revealed Pokémon
mid-game, outputs a distribution over the unrevealed slots. This is the world-sampling
step for MCTS: at the start of each search trajectory, sample one complete team hypothesis
from this distribution. See `designs/ai_v6/impl_step3_team_completion.md` and the
detailed architecture and data-pipeline notes in
`designs/ai_v6/design_team_completion_detail.md`.
Ladder data sources and opponent-sampling rationale are in
`designs/ai_v6/design_ladder_sampling_and_prediction.md`.

**Design questions to resolve:**
- **Backbone freezing**: freeze the PPO role encoder + embeddings and train only the new
  transformer head, or fine-tune the backbone jointly once the head has converged?
- **Inference mode**: sample one hypothesis per MCTS trajectory (stochastic, diverse) vs.
  argmax (deterministic, biased toward common teams). Stochastic is correct for PIMC.
- **Data sources**: 770 curated teams + self-play JSONL (from `--team-log`) + scraped
  ladder replays. Which combination is needed before MCTS quality is acceptable?

---

## Step 5 — MCTS

> **Superseded as the "anticipation" route by Step 6.** The owner's standing constraint
> (`designs/research_state/README.md`) is **no search/MCTS on the model** (inference OR training
> loop) — search is an offline teacher / diagnostic only. Step 6 delivers the same goal (an
> anticipatory policy/value) as a feedforward L3 lever. MCTS below is retained as the L4
> ceiling-setter (an offline, distilled teacher), not a runtime tree.

MCTS used at **inference time only** as a policy improvement operator on top of the
trained PPO network. The neural network is never used to generate MCTS training data —
environment stepping is the rollout bottleneck (~10ms per step), making MCTS-based data
generation ~1000× too slow to collect 150M training steps. Following Wang (2024) exactly:
20 workers, 10 rollouts per sync cycle, Showdown as the game simulator, policy+value
networks for action selection and leaf evaluation, team completion model for hidden info
sampling. See `designs/ai_v6/impl_step5_mcts.md`.

**Design questions to resolve:**
- **α and β hyperparameters**: control exploration weight and policy trust in the tree
  policy `U(s,a) = P[s,a]^β · sqrt(M[s]) / (N[s,a] + 1)`. Tune on a held-out set of
  games; start with α=1.0, β=1.0 (standard PUCT) and ablate.
- **Parallelism**: 20 workers × 10 rollouts per sync cycle. How much of the 10-second
  decision clock survives after poke-env WebSocket latency on our hardware?

### Baby Step: Sim Bridge

The first concrete implementation piece of Step 5. See
`designs/ai_v6/impl_step5_sim_bridge.md`.

Covers `new` / `fork` / `step` / `inject` / `free` only. Uses a hybrid API:
BattleStream for initial session setup (`new`), Direct Battle API for all fork/step
operations (`Battle.fromJSON()`, `battle.makeChoices()`). Does NOT include root sync
(`advance`), PIMC sampling, action sampler, or any tree logic.

**Go/no-go gate:** Step 2 of the baby step (injection test) must pass before any
further MCTS code is written. If `Battle.fromJSON()` rejects modified state, the
state-extraction approach for PIMC needs rethinking.

### Deferred from Baby Step

**`advance` (root sync)** — keeps the bridge root session in lock-step with the live
poke-env game by replaying real move choices after each turn via BattleStream. Not
needed until the bridge is wired into `choose_move`.

**`battle_serializer.py`** — converts a live poke-env `AbstractBattle` + team
hypothesis into Showdown `toJSON()` format. Requires careful handling of: Gen 3 stat
formula (EVs/IVs/nature → `baseStoredStats`), move PP tracking for opponent mons,
volatiles/boosts/side-conditions dict formats, and the `"[Species:name]"` reference
format. Depends on the injection test passing.

**`hypothesis.py`** — fills unrevealed opponent slots for PIMC. Initially uniform
random from Gen3OU usage stats; later replaced by the team completion model (Step 3/4).

**`action_sampler.py`** (Phase 1) — K rollouts per legal action, mean return, argmax.
The first search layer above the bridge. No tree needed.

**`tree.py` + `rollout.py`** (Phase 2) — full UCB tree with Q/N/M/P/F dicts, backup,
and fainted-slot pruning. Deferred until Phase 1 shows win-rate signal.

**`search_player.py` / `play_mcts.py`** — inference player wrapper and eval entry point.

**Parallel workers (20 workers + aggregator)** — deferred until single-worker search
is validated end-to-end.

**Rust sim (v8)** — replaces `sim_bridge.js` with a PyO3 Rust sim (~50× faster). The
`SimClient` interface is unchanged; only the bridge implementation swaps out.

---

## Step 6 — Latent Predictive Representation (Meaning B)

The **search-free** route to an anticipatory agent: a latent predictive auxiliary objective that
shapes the shared trunk so the single forward pass acts as if it had looked one ply ahead — **no
runtime simulator, no tree**. The simulator is used only as a *supervision oracle* (the env
already folds the realized `TurnDelta` for the reward; we reuse it as a free, on-policy target).
Culminates in the **per-action outcome-token injection** idea: a learned `g(trunk, action)`
produces one predicted-outcome latent token per legal action, injected so policy (and, via the
shared trunk, value) can attend over the imagined consequence of each option.

Full design + grounded attach points + the L1–L4 / leakage / collapse analysis:
**`designs/ai_v6/design_latent_predictive_representation.md`**.

**Incremental ladder (cheapest/safest first; the injection is the *payoff*, Stage 4 — not Stage 1):**

- **Stage 0 — FREE offline kill-gates** (zero training): prober probes on the existing checkpoint
  — is the next-turn outcome anticipatable, and is there headroom below the ~0.79 incoming-belief
  ceiling? `falsify-scan` whether loss-craters are anticipatable-but-unacted (a credit problem) vs
  genuinely surprising. **GO/STOP before spending a retrain.**
- **Stage 1 — plumbing, no learning** (`aux_coef=0`): the `outcome_target` Dict obs key +
  gated-construction `OutcomePredictor` reproducing the baseline byte-for-byte (no `ARCH_SIGNATURE`
  bump). Gate: `no_op_equivalence` + obs-build benchmark + bridge fuzz of the target pairing.
- **Stage 2 — aux loss ON, no injection** (`aux_coef~0.1`, 7-field collapse-proof discrete
  target): does a predictive objective shape the trunk? Gate: `next_ko_auc > 0.70`,
  `grad/aux_share < 0.4`.
- **Stage 3 — behavioural A/B** of the shaped trunk (anchored ELO + prober). The
  decorate-the-trunk gate: a NAMED behavioural metric (surprise-OHKO read-rate / under-switching)
  must move, else escalate to injection.
- **Stage 4 — inject per-action outcome tokens** (the centerpiece): concat-read → CLSPool
  cross-attention → sequence injection, in escalating risk; per-class generators +
  inverse-propensity weighting; **policy-pool-only to avoid the value-path leak.** `ARCH_SIGNATURE`
  bump. Gate: beats the loss-only arm beyond ELO CI + non-trivial token attention mass + clean
  leak ablation.
- **Stage 5 — opt-in richer targets** (never required for v1): 5a SPR/BYOL latent target (full
  collapse stack); 5b offline counterfactual reroll (the principled fix if switch behaviour
  doesn't move).

**Design questions to resolve:**
- **Stage-0 outcome:** does the trunk *already* anticipate as well as the incoming-damage belief
  (AUC ~0.79)? If so, kill the lever and spend the cycle on the reward/credit levers
  (`--switch-bias-weight`, `--self-ko-hp-penalty`).
- **The honest null:** the incoming-belief precedent shaped the trunk yet the policy still
  under-switched (a credit-assignment gap, not representation). Stage 4 (injection) is the bet
  that *attending over imagined consequences* is what converts anticipation into action — is it?
- **Re-home `team_completion_model.py`:** the orphaned masked-slot predictor shares the decode-head
  pattern; restructuring it onto the shared trunk is the natural second aux head (predict the
  opponent's hidden party). ai_v7.

---

## Step 7 — In-place opponent belief, move reinjection ✓ SHIPPED (A/B pending)

The in-trunk realization of the "predict the opponent's hidden party" aux head above — built on the
shared PPO trunk, not the offline team-completion model. Two halves shipped:

- **Species** (`opp_belief_slots` / `--opp-belief-aux-coef`, v16): `BeliefSlots` fills the un-revealed
  opp slots with learned unknown-mon tokens (refined in-lineup, attended by both heads); a `BeliefHead`
  aux-supervises species + moves (Hungarian). A readout — predicts but doesn't feed back.
  Record: `designs/ai_v5/belief_aux_as_built.md`.
- **Move reinjection** (`--move-belief-mode` / `--move-belief-coef`, v17): `MoveBelief` predicts each opp
  slot's moveset, soft-embeds it, and REINJECTS it into the token before the CLS pools — so the belief
  flows to both heads (not a dead-end readout). `--move-belief-mode {off,revealed,unrevealed,both}`;
  `revealed` (seen mons' unrevealed moves — defensible, surprise-OHKO) vs `unrevealed` (hidden mons,
  Hungarian; requires the species head) is the defensible-vs-omniscient A/B.
  Design + as-built: **`designs/ai_v6/impl_step7_move_belief_reinjection.md`**.

**Open gate:** UNMEASURED whether it helps the policy — fresh-run A/B (revealed vs unrevealed at matched
coef + a coef=0 control) where a NAMED behavioural metric (surprise-OHKO read-rate / crater share) moves
and win-rate is non-regressing. Same honesty discipline as Step 6 Stage 3; risk = "learnable but
inconsequential" (the credit-assignment gap the incoming-belief precedent showed).
