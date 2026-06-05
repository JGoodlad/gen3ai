# Design: Env-Worker Memory Leak — Root Cause & Fix

The `--use-showdown-bridge --self-play` training runs exhibited a severe memory climb: the
SubprocVecEnv env-worker forks grew **~30 GB/hr**, hit an **~82 GB ceiling**, then swap-thrashed
until the launcher's 3 h restart reclaimed everything — every cycle, identically. This document
records the investigation that root-caused it, the fix, and how it was verified. **Status:
FIXED** (`Gen3Player.reset_battles` + `selfplay_opponent_leak_fuzz_test.py`).

## Symptom

A 12 h PSS-bucketed monitor over a live run isolated the growth to a single bucket:

| Bucket | Over 12 h | Verdict |
|---|---|---|
| **env_worker** (24 forks) | 10 → **82 GB** each 3 h cycle, ~30 GB/hr | the leak |
| ui_launcher (the TUI) | 34–40 MB | flat — not the UI |
| main_trainer | 0.4–2.0 GB | flat (dips = swapped out) |
| bridge_node | 1.9–5.1 GB | flat (spikes = eval bursts) |

The growth was **uniform across all 24 workers** and **~100% private anonymous heap**. The
pattern repeated across 5 restart cycles with no variation. The launcher's restart — documented
as reclaiming *"pymalloc fragmentation"* — was in fact masking a real retained-object leak (a
`malloc_trim(0)` reclaimed only ~10%, so ~90% was genuinely live).

## Why it was hard to see

Three properties sent naive profiling the wrong way:

1. **A `gc` object-COUNT histogram looked flat.** `gc.get_objects()` does not return `str`,
   `bytes`, `int`, or `float`, and the leak's bytes live inside a *few* container objects whose
   *count* barely moves while their *contents* grow. So "list count flat, dict count flat" was
   true and misleading.
2. **`tracemalloc` showed almost nothing.** Much of the retained memory is the obs/history
   arrays and strings reached through those containers; the Python-level allocation deltas were
   hundreds of KB against hundreds of MB of RSS growth.
3. **It is self-play-specific.** A heuristic-opponent rollout does **not** leak. Reproducing it
   required the `--self-play` opponent path.

A methodological hazard also surfaced: a **reporter thread** walking `gc.get_objects()`
concurrently with the rollout crashes the process with a C-level
`SystemError: bad argument to internal function` (a tuple being built on the main thread while
the gc graph is traversed). **All heap introspection must be synchronous** with the rollout.

## Root cause

`Gen3Player` (base of the inference `RLPlayer`) keeps two caches keyed by battle tag:

```python
self._stall_loggers: dict[str, StallLogger] = {}   # player.py
self._trackers:      dict[str, EpisodeTracker] = {} # player.py — holds obs + turn-delta history
```

`_get_tracker()` / `_handle_stall()` create one entry per battle tag. The **only** eviction
point is `_battle_finished_callback`:

```python
def _battle_finished_callback(self, battle):
    super()._battle_finished_callback(battle)
    tag = battle.strict_view().battle_tag
    self._stall_loggers.pop(tag, None)
    self._trackers.pop(tag, None)
```

But `_battle_finished_callback` only fires for a **networked** player — one with a live poke-env
message loop. In self-play, the opponent (`pool_player`) is built **`start_listening=False`** and
used as a **pure decision function** over the env's `battle2` (`env.agent1`/`agent2` do the
networking; the opponent only supplies move choices). It has no message loop, so **it never
receives `_battle_finished_callback`.**

Two further facts complete the leak:

- `reset_battles()` — which the training wrapper calls on the opponent **every episode**
  (`single_agent_wrapper.reset` → `self.opponent.reset_battles()`) — cleared poke-env's
  `_battles` dict but **not** our `_trackers` / `_stall_loggers`.
- The in-process bridge mints a **process-unique battle tag per battle**
  (see `design_local_sim_bridge_transport.md` / the unique-tag guard). So every battle is a fresh
  key that is never the same twice.

Net: **every self-play battle leaves one permanent `EpisodeTracker` (and its obs + turn-delta
history — `BattleContext`s, ~3357-float obs vectors × `N_HISTORY_TURNS`, delta vectors, a
`HiddenPowerTracker`) plus one `StallLogger` behind, keyed by a tag that never recurs.** At
roughly ~1 MB+ per battle × thousands of battles per 3 h × 24 workers, that is the 30 GB/hr → 82 GB
climb. The trainee side never leaked: `Gen3Env` uses a **single reused** `EpisodeTracker`
(`self._tracker`), not a tag-keyed dict.

### Reproduction

An isolated harness rebuilt one env-worker rollout in-process via the bridge and drove it with
random legal actions, watching `len(opponent._trackers)` per episode:

```
# heuristic opponent — NO leak (519 MB flat over 60 eps)
# self-play RLPlayer opponent — LEAK:
eps=5   OPP _trackers=5    _stall_loggers=5    RSS=601MB
eps=10  OPP _trackers=10   _stall_loggers=10   RSS=606MB
...
eps=45  OPP _trackers=45   _stall_loggers=45   RSS=652MB   (+1 per battle, monotonic RSS)
```

`_trackers` grows exactly +1 per battle while `_battles` stays at 1 — the smoking gun.

## Fix

Override `reset_battles()` on `Gen3Player` to prune the per-tag caches to the still-live battle
tags. The wrapper already calls this on the opponent every episode, so it is the correct, cheap
eviction point that the missing `_battle_finished_callback` cannot be:

```python
def reset_battles(self) -> None:
    super().reset_battles()                       # clears _battles
    live = set(self._battles)
    self._trackers = {t: v for t, v in self._trackers.items() if t in live}
    self._stall_loggers = {t: v for t, v in self._stall_loggers.items() if t in live}
```

`super().reset_battles()` empties `_battles`, so in the per-episode case this clears both caches
entirely; writing it as a **prune to live tags** (rather than a blind `.clear()`) keeps it safe
for any caller that resets while battles are still in flight (e.g. concurrent eval). Networked
players (eval) are unaffected — their `_battle_finished_callback` still does the eviction; this
is belt-and-suspenders on an already-called per-episode hook.

### Why not "make `_battle_finished_callback` fire"?

The opponent is intentionally a decision function with no message loop (that's the throughput
design — `env.agent1`/`agent2` own the networking). Giving it a loop would undo that. Pruning on
the existing per-episode `reset_battles()` is the minimal, on-design fix.

## Verification

- **Repro, post-fix:** `_trackers`/`_stall_loggers` stay at 0 at every checkpoint; RSS flat at
  ~600 MB over 45 episodes (was climbing 601 → 652).
- **Regression guard:** `src/agents/training/selfplay_opponent_leak_fuzz_test.py` — runs real
  self-play battles in-process via the bridge (trainee = random legal actions, opponent = an
  untrained `RLPlayer` through the actual `MaskableAgentWrapper` self-play path) and asserts the
  opponent's per-tag caches stay bounded (`≤ 3`). Trips after **4 battles** on the pre-fix code;
  passes on the fix.
- **Unit suite:** 595 `inference/` + `training/` unit tests pass; the existing
  `selfplay_opponent_fuzz_test.py` still passes (opponent plays normally).

## Files

- `src/agents/inference/player.py` — `Gen3Player.reset_battles()` override (the fix).
- `src/agents/training/selfplay_opponent_leak_fuzz_test.py` — regression fuzz test.

## Implications

- The launcher's periodic restart can return to being **fragmentation hygiene** rather than
  leak containment. Env-worker RSS should now hold flat across a full 3 h window.
- General lesson for this codebase: any `Gen3Player`/`RLPlayer` used as a **decision-function
  opponent** (`start_listening=False`) must have its per-battle-tag state pruned on
  `reset_battles()`, because `_battle_finished_callback` will not fire for it.
- Diagnostic lesson: to root-cause a per-worker leak, reproduce **one** env-worker rollout
  in-process via the bridge and introspect **synchronously** (never from a reporter thread).
  Compare a heuristic-opponent rollout (control) against a self-play rollout to localize.
