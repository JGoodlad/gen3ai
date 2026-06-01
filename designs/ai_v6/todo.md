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
