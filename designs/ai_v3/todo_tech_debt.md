# Tech Debt Backlog

Items from the May 2026 refactoring audit. Grouped by when they're safe to attempt.

---

## Safe With Care — Do manually, one at a time, when training is paused ⚠️

These touch live call sites but behavior is identical. Update all sites atomically and run
smoke test after each.

### #5 — Lazy encoder init in `embed_battle()`

**File:** `src/agents/inference/player.py:51–60`

**Problem:** `observation_encoder` and `_turn_delta_encoder` initialize on the first `embed_battle()` call, hiding dependencies from the constructor. On a 64-env setup, 64 concurrent first calls to `load_mappings()` with no lock. `RLPlayer(model, team, ...)` looks complete at construction but crashes on first step if JSON files are missing.

**Fix:** Make `observation_encoder` and `turn_delta_encoder` required constructor args of `Gen3Player`. Keep a deprecated lazy fallback with a warning during transition.

**Call sites to update:**
- `src/main/train_rl_agent.py:306, 420`
- `src/agents/training/eval_callback.py:117`
- `src/agents/training/selfplay_callback.py:118, 130`
- `src/agents/training/replay_recorder.py:94`

---

### #1 — Implicit decision context latch

**File:** `src/agents/action/mask_generator.py:47–52`, `src/agents/action/mapper.py:35–51`, `src/agents/inference/player.py`

**Problem:** `get_mask()` mutates `battle._gen3_decision_context` as a side effect. The call-order contract (always mask before map) is implicit — wrong order gives a runtime error at action time, not a type error. Hard to test in isolation.

**Fix:** Return `(mask, DecisionContext)` from `get_mask()`. Thread `ctx` explicitly through `action_to_order(action, battle, ctx)`. Keep writing `battle._gen3_decision_context` for one release for backward compat.

**Call sites to update:**
- `src/agents/training/gen3_env.py:49, 86`
- `src/agents/observation/state_encoder.py:173`
- `src/agents/inference/player.py:84`

---

## Requires a Clean Run Boundary — Do not touch during active training 🔴

### #3 — TurnDelta / observation slice offsets hardcoded

**File:** `src/agents/model/features_extractor.py:631–635`

**Problem:** `td_strategic[:, 0:4]`, `[:, 4:8]`, `[:, 8:9]`, `[:, 9:10]` in the perspective-flip code; `N_HISTORY_TURNS`, `TURN_DELTA_DIM` imported across module boundaries. Layout changes silently break slices.

**Fix:** Add `dimensions.py` with named slices (`OUR_EFF_SLICE`, `OPP_FIRST_SLICE`) and all dimension constants. Both encoder and extractor import from it. Requires model reload smoke test after.

---

### #4 — `order_to_action()` returns `0` on 3 silent failure modes

**File:** `src/agents/action/mapper.py:94–124`

**Problem:** Action index `0` (switch to slot 0) is returned on unknown switch, unknown move, and non-SingleBattleOrder — conflating success with error. Other call paths silently depend on this return value.

**Fix:** Return `Optional[int]` (None on failure) or raise typed `ActionMappingError`. Update 5 call sites. Risk: may expose silent misbehavior that's currently masked by the fallback.

---

### #8 — `PokemonEncoder` god aggregator

**File:** `src/agents/observation/pokemon.py`

**Problem:** Orchestrates 5 sub-encoders, hardcodes status mapping inline, implements decoding, does layout offset arithmetic — all in one class. Adding any new per-Pokémon feature requires editing multiple methods.

**Fix:** Move status encode/decode to `StatusEncoder`; declare offsets in a layout schema. `PokemonEncoder` only orchestrates. Touches observation encoding — test that obs vector dimension and values are unchanged after.

---

### #9 — `RewardTracker` / `RewardTrackingMixin` near-duplicates

**File:** `src/agents/training/reward_tracker.py:9–134`

**Problem:** Two classes do essentially the same `begin_turn/complete_pending/finalize` lifecycle for different callers (env vs. eval player). The mixin accesses private `tracker._our_slots` (lines 109–110), breaking encapsulation. Bug fixes in one don't propagate to the other.

**Fix:** Unify into `BattleRewardTracker`; add thin adapter for env vs. player call sites. Risk: reward computation changes; run reward invariant tests after.

---

### #10 — `Gen3RewardManager` god object (600 lines)

**File:** `src/agents/training/reward_manager.py`

**Problem:** Mixes snapshot state (10+ instance vars), 20+ signal functions, subsidy/tax management, orchestration, and episode reporting. Adding a signal touches `record_action`, `process_turn_reward`, and the signal function — 3 places in a 600-line class. Called from 6 production sites.

**Fix:** Extract signal computation to `RewardSignalComputer`; move move-category sets (`BOOST_MOVES`, etc.) to `MoveCategories` module separate from `RewardMagnitudes` scalars. Must preserve `record_action → process_turn_reward` call ordering at all 6 sites.

---

## Medium Priority (non-critical, do opportunistically) 

| # | File | Issue |
|---|------|-------|
| 11 | `inference/player.py:28–49` | Per-battle state (tracker + stall logger) split across two parallel dicts; combine into `BattleState` dataclass |
| 12 | `training/episode_tracker.py:61–75` | `prev_N_delta_vecs()` uses `-1-i`, `-2-i` reverse indexing; extract `get_deltas_newest_first(n)` helper |
| 13 | `model/features_extractor.py:40–44` | `mappings` param now only used by `ObservationDebugger`; remove it from extractor constructor |
| 14 | `model/features_extractor.py:407–490` | 80-line ID extraction block repeats offset→`.long()`→`.unsqueeze(2)` for 5 types; extract helper |
| 15 | `observation/pokemon.py:64–72, 156–161` | `status_map` / `names` defined inline twice; extract `STATUS_TO_IDX` / `IDX_TO_STATUS` constants |
| 16 | `observation/state_encoder.py:32–73` | `load_mappings()` hard-codes file I/O; extract `MappingRegistry.from_json_files()` / `from_dict()` |
| 17 | `observation/state_encoder.py:107–109` | `TurnDeltaEncoder` imported inside `__init__()` to break circular import; fix with `dimensions.py` |
| 18 | `training/battle_recorder.py:86–128, 317–369` | `_fill_pending_outcome` and `finalize` share 60% of outcome-building code; extract `_build_outcome()` |
| 19 | `training/battle_context.py:257–351` | `TurnDelta.build` has 6-level nesting for phaze/faint/voluntary detection; extract `_determine_opponent_action()` |
| 20 | `training/gen3_env.py:46–60` | `embed_battle` uses `if battle is self.battle1` inline; delegate to `AgentObsBuilder` |
| 21 | `training/eval_callback.py:74–247` | Schedule and opponent construction hardcoded; inject `schedule_fn` and pre-built opponents |
| 22 | `training/battle_context.py:10–107` | 26 fields; `opp_all_last_move_ids` only populated in edge cases but lives on every snapshot |

---

## Low Priority (fix during nearby edits)

| # | File | Issue |
|---|------|-------|
| 23 | `observation/pokemon.py:50–62` | Stale comments claim `Items (16+1)`, `Types (8)`, `Abilities (25)`; actual dims are 3, 2, 2 |
| 24 | `observation/state_encoder.py:104–108` | `ActiveContextEncoder` receives `move_to_id` it never uses; remove the parameter |
| 25 | `observation/` encoders | `__init__` signatures all differ; define common `EncoderConfig` / `MappingRegistry` arg |
| 26 | `model/features_extractor.py:600–615, 660–668` | `torch.where(flags.any(dim=1), argmax, zeros)` repeated 4 times; extract `_find_active_index()` |
| 27 | `action/mapper.py:79–80, 121–122` | `startswith("hiddenpower")` matching duplicated; extract `_move_ids_match(a, b) -> bool` |
| 28 | `action/mask_generator.py:30–32, mapper.py:57–58` | Duplicate species check at mask time and map time; remove from mapper |
| 29 | `training/battle_context.py:20` | `phase: Literal["move_selection", "forced_switch"]` is a decision type not a game phase; rename to `decision_type` |
| 30 | `training/reward_manager.py` | `_last_` prefix on 5+ unrelated scopes; use `_prev_` / `_current_` / `_episode_` prefixes |
