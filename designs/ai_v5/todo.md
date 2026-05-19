# AI v5 — Todo

---

## Step 1 — Replay Collection ✓ DONE

Passive spectator client that connects to Showdown, discovers active Gen 3 OU battles,
and saves the complete battle logs for offline use. See
`designs/ai_v5/impl_step1_replay_collection.md`.

**Deliverables:**
- `src/poke_env/spectator/spectated_battle.py` — pure data object; accumulates log lines,
  tracks current turn, player names, join time
- `src/poke_env/spectator/spectator_client.py` — `BattleSpectator` async generator; rate
  control, ghost-battle prevention, auto-reconnect on TCP reset
- `src/poke_env/spectator/__init__.py` — package exports
- `src/poke_env/ps_client/ps_client.py` — `QueryResponseCallback`, proxy support,
  `ConnectionClosedError` demoted to WARNING, wire logs at DEBUG
- `src/poke_env/battle/abstract_battle.py` — `":"` / `"t:"` added to `MESSAGES_TO_IGNORE`
- `src/main/collect_replays.py` — daemon with Rich full-screen dashboard (active battles,
  recent completions, live logs, PROXIED/DIRECT indicator); proxy + verbose CLI flags
- `src/poke_env/spectator/spectated_battle_test.py` — 9 unit tests
- `src/poke_env/spectator/spectator_e2e_test.py` — live server test + round-trip log parse

**Status:** Daemon is actively collecting at `replays/showdown/1/`. Running on POKE_LOOP
via `asyncio.run_coroutine_threadsafe`. Reconnects automatically on connection drop.

---

## Step 2 — Replay Parsing Pipeline

Convert raw `.log` files into (observation, action) pairs that can feed a behavioural
cloning loss. Requires:

- A **log reader** that instantiates a `Battle` and replays log lines through
  `parse_message` / `won_by` / `tied`, producing the sequence of battle states.
- A **label extractor** that reads the `|move|` / `|switch|` lines to determine what
  action each player took on each turn, maps those to the RL action space (via
  `Gen3ActionMapper`), and aligns them with the observation vector at that turn.
- An **observation reconstructor** that runs `Gen3ObservationEncoder` on each replayed
  `Battle` state to produce the float32 obs vectors.

Key challenge: the spectated log is from the spectator's perspective (public channel),
so the opponent's moves are visible but the `|request|` JSON (which lists available
moves with PP) is absent. The observation will be missing the `prev_mask` and exact PP
values for the player whose turn it is. Decide whether to:
  - Skip turns where the acting player's exact options are unknown, or
  - Synthesise a best-effort mask from the known moveset.

---

## Step 3 — Behavioural Cloning Pre-training

Pre-train (or fine-tune) the PPO policy network on the (obs, action) pairs from Step 2
using a supervised cross-entropy loss before handing off to RL. Goal: give the agent a
strong prior over human move selection so RL exploration starts from a sensible baseline
rather than random.

Design questions to resolve:
- **Dataset split:** hold out a fraction of replays for validation loss tracking.
- **Class imbalance:** switch actions are far less frequent than move actions; consider
  weighted sampling or focal loss.
- **Architecture compatibility:** the existing `Gen3FeaturesExtractor` + PPO policy head
  should be usable directly; no architecture change needed.
- **Stopping criterion:** monitor validation loss and stop before the policy collapses to
  mode-seeking (overfit to the most common action).

---

## Step 4 — RL Fine-tuning

Resume PPO training from the BC-pre-trained checkpoint. The pre-trained prior should
reduce the number of steps needed to reach competency against the heuristic opponent,
and may push the ceiling higher.

Metrics to watch vs. a cold-start RL baseline:
- Win rate vs. `RandomPlayer` at 100K steps
- Win rate vs. `MaxDamagePlayer` at 500K steps
- Entropy collapse rate (BC pre-training can over-reduce entropy — watch `ent_coef` tuning)
