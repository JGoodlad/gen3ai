# Implementation: Step 1 — Clean Training Pipeline

This step refactors the training pipeline to eliminate state tattooing on battle objects,
establish clean separation of concerns across env, reward, and wrapper layers, and harden
several reward signal correctness issues discovered during v2 training.

## Motivation

The v2 pipeline had accumulated several coupling problems:

- **`battle._latched_mask` / `battle._latched_turn`** — mask and turn were tattooed directly
  onto poke-env battle objects, creating hidden shared state between embed_battle, action
  mapping, and reward computation.
- **Reward function entangled with env** — `calc_reward` called `super()` internally,
  meaning the reward function couldn't be understood or tested in isolation.
- **Stall termination was hacky** — turn 250 would short-circuit `step()` and return a
  hardcoded `-50.0` penalty without ever sending a forfeit to the server. The battle never
  actually ended, so `battle.lost` was never set and the normal reward path was bypassed.
- **Explosion detection was noisy** — scanned all currently-fainted opponent mons for
  explosion in their moveset, not just the one that acted this turn, causing false positives
  from mons that fainted in prior turns.
- **Struggle leaked into move slots** — the reactive encoder filled move slots 0-3 with
  Struggle's stats when PP hit zero, creating an ambiguous alias between slot 0 (action 6)
  and the dedicated Struggle action (10).
- **Struggle not visible to role encoder** — the `is_forced_struggle` flag existed in the
  reactive block but was never broadcast to the per-Pokémon role encoder, so the model
  couldn't learn "Skarmory is in struggle, switch it out."

---

## What Was Built

### New Types: `BattleContext` and `TurnDelta`

**`src/agents/training/battle_context.py`**

`BattleContext` is a frozen per-turn snapshot built once in `embed_battle()` and stored on
the env as `self._last_ctx`. It replaces all uses of `battle._latched_mask` and
`battle._latched_turn`.

```
BattleContext
  turn, phase, mask (11,), obs (OBS_DIM,)
  our_slot_map, opp_slot_map   — stable species→slot dicts
  our_hp, opp_hp               — (6,) HP fractions in slot order
  our_active, opp_active       — species name of active mon ("NONE" if absent)
  our_fainted_count, opp_fainted_count
```

`TurnDelta` is the diff between two consecutive `BattleContext`s. It carries everything the
reward function needs without touching the battle object for HP or faint signals.

```
TurnDelta
  our_move_id, our_switch_to, our_prev_active
  opp_move_id, opp_switch_to, opp_prev_active
  our_hp_delta, opp_hp_delta   — (6,) deltas; negatives = damage taken
  we_fainted, opp_fainted
```

`TurnDelta.build(prev_ctx, curr_ctx, action)` is the canonical constructor.
`TurnDelta.empty()` is used for the first turn of an episode.

### New Type: `SlotRegistry`

**`src/agents/training/slot_registry.py`**

Assigns stable 0-5 slot indices to species on first-seen, preventing the observation vector
from re-ordering between turns when poke-env's insertion-order dict changes.

### `RewardFunction` Protocol

**`src/agents/training/reward_function.py`**

```python
class RewardFunction(Protocol):
    def record_action(self, ctx: BattleContext, action: int) -> None: ...
    def process_turn_reward(self, battle, delta: TurnDelta) -> float: ...
    def reset(self) -> None: ...
    def report_episode(self, battle) -> None: ...
```

Injected into `Gen3Env` at construction time. The env gates reward calls to agent1's battle
only — opponent battles fall back to `reward_computing_helper`.

### `Gen3RewardManager` Rewrite

**`src/agents/training/reward_manager.py`**

- `record_action(ctx, action)` — uses `ctx.our_active`, `ctx.opp_active`, `ctx.turn`,
  `ctx.mask` instead of reaching into the battle object.
- `compute_base_reward(delta, battle)` — HP deltas and faint events from `TurnDelta`;
  win/loss still from `battle.won/lost/finished` (terminal signal only).
- `process_turn_reward(battle, delta)` — assembles base reward, explosion signal, pivot
  bonus, switch subsidy, and stall tax. No `super()` call anywhere.
- Removed `SinglePlayerRewardManager` wrapper — the env gates correctly now.

### `MaskableAgentWrapper`

**`src/agents/training/wrappers.py`**

Moved out of `train_rl_agent.py` to its own module. Two responsibilities:

1. **`action_masks()`** — delegates to the inner env's `BattleContext` for MaskablePPO.
2. **Wait-turn absorption** — loops while `not env.agent1_to_move` to absorb ghost steps
   before returning to SB3, preventing credit assignment pollution.

### Stall Termination Fix

**`src/main/train_rl_agent.py` — `action_to_order()`**

The forfeit is now issued via `ForfeitBattleOrder()` inside `action_to_order` when
`battle.turn >= 250`. The server processes it as a real forfeit, `battle.lost = True`, and
the reward flows through the normal `-VICTORY_VALUE` path. No more hardcoded `-50`.

### Explosion Detection Fix

**`src/agents/training/reward_manager.py` — `process_turn_reward()`**

Now checks only `delta.opp_prev_active` (the mon that was active this turn) rather than all
currently-fainted mons. Rewards:

| Scenario | Signal |
|---|---|
| Opponent explodes, we survive | `+2.0` bonus |
| Opponent explodes, mutual KO or we lose | `-3.0` penalty |

Mutual KO (tie) also hits `elif battle.finished: reward -= VICTORY_VALUE` for a total of
`-33.0` — meaningfully worse than a clean loss.

### Struggle Observation Fixes

**`src/agents/observation/reactive.py`**

When `is_forced_struggle` is true, move slots 0-3 are zeroed out. Previously Struggle's
stats were filling slot 0, creating ambiguity with action 6. Now `vec[15] = 1.0` is the
sole signal.

**`src/agents/model/features_extractor.py`**

The `is_forced_struggle` flag is now extracted from the reactive block and included in the
global context broadcast to every Pokémon role encoder:

```python
struggle_feature = remaining_part[:, reactive_start + 15 : reactive_start + 16]
global_context = torch.cat([turn_feature, weather_feature, fainted_feature,
                             spikes_feature, struggle_feature], dim=1)  # [B, 12]
```

Role encoder input dim: `237 → 238`. Every per-Pokémon role token now sees whether the
active mon is in Struggle, enabling the model to learn "switch out the useless mon."

---

### Replay Recorder (`callbacks.py` → `replay_recorder.py`)

Rewrote the replay and summary system. The old `callbacks.py` had `StatTrackingRLPlayer`
doing retroactive result patching, duplicated switch detection, and manual HP subtraction —
all the patterns we eliminated in the reward pipeline. The new design uses the same
primitives.

**`src/agents/training/battle_recorder.py`** — new

`BattleRecorder` owns all tracking logic for a single battle:
- `record(battle, action_idx, probs, mask)` — called once per model invocation; builds a
  `BattleContext` using `SlotRegistry` for stable HP indexing, uses
  `battle._gen3_decision_context` (latched by `Gen3ActionMasker`) for consistent action
  labels, stores the invocation as pending.
- `_complete_pending(curr_ctx, battle)` — called at the next invocation; calls
  `TurnDelta.build(prev_ctx, curr_ctx, action)` to get HP deltas, switch events, and faint
  events for free. No manual state diffing.
- `finalize(battle)` — completes the last pending invocation at battle end.
- `to_summary(battle, step)` — exports a JSON-serializable dict.

Each invocation entry is self-contained — action labels are inlined, not referenced from a
legend — so a human can read any turn in isolation:

```json
{
  "i": 3, "turn": 5, "phase": "move_selection",
  "our": { "species": "tyranitar", "hp": "82%" },
  "opp": { "species": "zapdos",    "hp": "65%" },
  "outcome": {
    "we":   { "action": "rockslide",   "hp_delta": "-18%" },
    "they": { "action": "thunderbolt", "hp_delta": "-35%" },
    "events": []
  },
  "chosen": "rockslide",
  "actions": {
    "switch:skarmory": { "prob": "12.5%", "valid": true },
    "rockslide":       { "prob": "18.3%", "valid": true },
    ...
  }
}
```

**`src/agents/training/replay_recorder.py`** (renamed from `callbacks.py`)

`StatTrackingRLPlayer` is now 8 lines — just hooks `choose_move` to call
`recorder.record()`. `StatTrackingHeuristicPlayer` and `init_stats` are removed entirely.
Summary files drop from ~30 KB to ~3-5 KB per battle.

---

## Infrastructure Fix: Git Worktree + Submodule

The original `CLAUDE.md` worktree setup used a symlink for `deps/pokemon-showdown`:

```bash
rmdir deps/pokemon-showdown
ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown deps/pokemon-showdown
```

This caused `git status` to fail with `error: expected submodule path
'deps/pokemon-showdown' not to be a symbolic link`, breaking VS Code's git integration.

**Fix — two steps**:

1. Initialize the submodule (gets source files, fixes VS Code git integration):
   ```bash
   git submodule update --init
   ```
   Git creates a real checkout with a `.git` file pointing back to the shared objects store.

2. Symlink the build artifacts (`dist/` and `node_modules/` are gitignored, so not part of the checkout, but the main repo already has them built):
   ```bash
   ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/dist \
         deps/pokemon-showdown/dist
   ln -s /home/goodlad/dev/gen3ai/deps/pokemon-showdown/node_modules \
         deps/pokemon-showdown/node_modules
   ```
   Without step 2, training fails with `Cannot find module '.../dist/sim/index.js'`.

`CLAUDE.md` updated accordingly.

---

## Tests Added

| File | Coverage |
|---|---|
| `src/agents/training/slot_registry_test.py` | SlotRegistry assignment, stability, overflow |
| `src/agents/training/battle_context_test.py` | BattleContext fields, TurnDelta.build(), faint detection, switch detection |
| `src/agents/observation/state_encoder_test.py` | Added `test_encoder_and_features_extractor_are_compatible` — full forward pass through both encoder and features extractor; catches obs-dim/architecture mismatches before they surface at checkpoint load |

---

## Phase 2: Pipeline Hardening

### Stall Detection Refactor

**`src/agents/training/stall.py`** — new

Extracted stall detection into `StallConfig` (dataclass) and `StallLogger`:
- `StallConfig(threshold=250, output_dir=...)` — configures threshold and HTML output dir
- `StallLogger.log_once(battle, suffix)` — idempotent per episode; writes HTML on first call
- `StallLogger.reset()` — called at episode start

Used by `Gen3Env`, `RLPlayer`, and `StatTrackingRLPlayer` with per-battle reset via
`_last_battle_tag` tracking on inference players.

### Action Mapping Fixes

**`src/agents/action/mask_generator.py`**

Struggle was being enabled at both slot 6 (move slot) and slot 10 (dedicated struggle slot)
when PP hit zero, because `request_moves[0]` holds the struggle entry. Fixed by filtering
`move.id == "struggle"` from the move loop. Regression test added.

### poke-env Stability Fixes

**`src/poke_env/environment/env.py`**

Three bugs fixed in the env lifecycle:

1. **Stale battle queue after forfeit** — when `action_to_order()` returns
   `ForfeitBattleOrder()`, `race_get()` for agent2 could return None without consuming its
   queue item. On the next `reset()`, this stale entry was picked up as the new battle.
   Fixed by draining non-empty queues in the `elif battle1.finished` branch of `reset()`.

2. **Popup after forfeit** — after agent1 forfeited, agent2's order was still queued and
   sent to the server after the battle room closed. Fixed by detecting
   `agent1_forfeited = isinstance(order1, ForfeitBattleOrder)` in `step()` and skipping
   agent2's order entirely.

3. **Stale `_choose_move()` hang** — skipping agent2's order left its `_choose_move()`
   coroutine suspended forever awaiting `order_queue`. On the next episode, the stale
   coroutine competed with the fresh one, stealing the first order and causing `step()` to
   hang indefinitely. Fixed by sending `_EmptyBattleOrder()` (null message, no server popup)
   to unblock the stale coroutine, plus draining agent2's `order_queue` in `reset()` in
   case poke-env cancelled the task before it consumed the empty order.

### Inference Player Hardening

**`src/agents/inference/player.py`**

- `Gen3Player` now owns `StallLogger` and `_last_battle_tag` for per-battle reset
- `action_to_order()` guards against missing decision context — raises `RuntimeError` if
  called without a preceding `embed_battle()`, catching turn-mismatch bugs early
- `RLPlayer.choose_move()` wrapped in try/except returning `ForfeitBattleOrder()` on any
  exception, preventing poke-env from hanging with no response to the server

### EpisodeTracker Extraction

**`src/agents/training/episode_tracker.py`** — new

Pulled the scattered per-episode mutable state (`_our_slots`, `_opp_slots`, `_last_ctx`,
`_prev_ctx`, `_last_action`) out of `Gen3Env` into a dedicated `EpisodeTracker`:

- `record(battle, mask, obs)` — builds and stores a `BattleContext` for the current turn
- `advance(action)` — records the chosen action before the game steps forward
- `build_delta()` — returns `TurnDelta.build(history[-2], history[-1], last_action)`,
  or `TurnDelta.empty()` at episode start
- `reset()` — clears all episode state

`Gen3Env` holds zero episode-scoped mutable state directly. The `_history: list[BattleContext]`
is the natural extension point for turn history: cap it to a `deque(maxlen=N)` and expose
`get_history() -> list[BattleContext]`.

---

## Final State

Step 1 is complete. The smoke test (`--debug --steps 10000`, ~1 min) passes reliably
including under stall conditions. 92 unit tests pass.

**Ready for Step 2: Turn History + Action Mask as Observation Features**

- `EpisodeTracker._history` → cap to `deque(maxlen=N)` to retain last N frames
- `Gen3ObservationEncoder` → `encode_with_history(current_obs, history: list[BattleContext])`
- Each `BattleContext` already carries `mask` and `obs` — no new fields needed
- `Gen3Env.embed_battle()` passes `tracker.get_history()` to the encoder

**Remaining `todo.md` items**

- Populate `opp_move_id` from the battle log (explosion detection precision)
- Status and stat-stage deltas (Aromatherapy, Calm Mind, Curse)
