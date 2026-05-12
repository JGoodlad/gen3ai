# Implementation Plan: Step 2.1 - Pipeline Hardening & Stall Diagnostics

This plan focuses on stabilizing the Gen3 RL training pipeline by enforcing state integrity, implementing proactive stall capture, and upgrading diagnostic reporting for long-term battle analysis.

## Goal
Eliminate cross-episode reward leakage, suppress redundant environment spam, and implement a "one-shot" turn-800 stall logging system to capture high-fidelity Showdown replays for passive-play investigation.

## Proposed Changes

### `src/main/train_rl_agent.py`

#### 1. Consolidate `Gen3Env` Logic
Merge duplicate `step()` and `reset()` definitions into a single, robust lifecycle handler that ensures:
- Reward state is cleared via `reward_manager.reset()` on every episode.
- Battle objects are pre-captured before `super().step()` to prevent cleanup race conditions.

#### 2. Proactive Stall Logger
Implement a turn-locked snapshot trigger in `step()`:
```python
# Trigger EXACTLY once at turn 800
if battle and battle.turn == 800 and not self._stall_logged:
    self._save_stall_html(battle, suffix="STALL")
    self._stall_logged = True
```
Utilize the official `poke_env` `battle.save_replay(path)` method for maximum fidelity.

### `src/poke_env/player/player.py`

#### 1. Selective Warning Suppression
Refactor the "bigerror" handler to silence the server's Turn 1000 auto-tie countdown while preserving critical protocol errors:
```python
elif split_message[1] == "bigerror":
    msg = "|".join(split_message)
    if "auto-tie" in msg or "turn 1000" in msg:
        self.logger.debug("Stall countdown silenced.")
    else:
        self.logger.warning(msg)
```

### `src/agents/training/callbacks.py`

#### 1. Transition-Based Switch Tracking
Upgrade `StatTrackingRLPlayer` to monitor actual field transitions (active Pokémon changes) rather than just action choices. This captures both voluntary swaps and forced faints.

#### 2. Nested JSON Summaries
Refactor the summary output to provide a structured `switches` object:
```json
"switches": {
    "total": 25,
    "voluntary": 18,
    "forced": 7,
    "log": { "12": "Blissey -> Tyranitar (voluntary)" }
}
```


### `src/agents/training/reward_manager.py`

#### 1. Metric Isolation (Decorator Pattern)
Prevent metric cross-talk and "double counting" of switches caused by shared state in the `Gen3RewardManager` by introducing a `SinglePlayerRewardManager` wrapper.
- `SinglePlayerRewardManager` acts as a guard, only delegating `process_turn_reward` to the internal manager if `is_trainee=True`.
- This removes the need for `Gen3RewardManager` to manage multi-perspective state, ensuring its internal `switch_count` and `total_reward` are always agent-pure.
- Standardize on `.species` tracking and implement the Turn 1 deployment safety check.

## Verification Plan
1. **Stall Verification:** Verify that `models/[RUN_NAME]/stalls/` contains exactly one `.html` file per 800-turn battle.
2. **Spam Reduction:** Confirm the terminal only displays Env #0 heartbeat logs and no Turn 1000 warnings.
3. **Report Accuracy:** Verify `total_switches` in `summary.json` correctly tallies the entries in the new `switches.log` dictionary.
4. **Console Integrity:** Confirm the "🏁 Episode Finished" console log shows realistic `Voluntary Switches` counts (typically < 30) instead of inflated turn-count multiples.
