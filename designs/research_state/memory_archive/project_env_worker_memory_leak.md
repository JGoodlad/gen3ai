---
name: project_env_worker_memory_leak
description: Root cause of the env-worker memory explosion (self-play opponent leaked an EpisodeTracker per battle); found + fixed 2026-06-05
metadata: 
  node_type: memory
  type: project
  originSessionId: f6666107-f284-4812-8475-58be2ebc41e0
---

The training "memory explosion" (env-worker forks climbing ~30 GB/hr to an ~82 GB ceiling each
3 h restart cycle, on a `--use-showdown-bridge --self-play` run) was a **real retained-object
leak**, NOT pymalloc fragmentation (the launcher's restart rationale) and NOT the UI/main
trainer/bridge (all measured flat — UI was ~40 MB the whole time).

**Root cause:** the self-play OPPONENT (`pool_player`, a `Gen3Player`/`RLPlayer` built
`start_listening=False` and used as a pure decision function over the env's battle) keeps two
per-battle-tag caches — `self._trackers` (an `EpisodeTracker` + its obs/turn-delta history) and
`self._stall_loggers`. Their only eviction point is `_battle_finished_callback`, which **only
fires for a networked player** — a decision-function opponent never gets it. `reset_battles()`
(called every episode by the wrapper) cleared `_battles` but not these caches, and the bridge
mints a process-unique tag per battle ([[project_bridge_unique_battle_tags]]), so **every battle
left a permanent tracker behind** (~1 MB+ each) → ×thousands of battles ×24 workers = the climb.
This is the answer to the "RAM-over-time" puzzle on the Python side (distinct from the node
server growth in [[project_showdown_server_memory_growth]]).

**Why it hid:** the trainee env uses ONE reused `EpisodeTracker` (no leak); only the `RLPlayer`
opponent keys them by tag. A heuristic opponent has no such cache, so it only reproduces under
`--self-play`. The leaked bytes were invisible to a naive `gc` object-COUNT histogram (the growth
is inside a few containers' history + strings) and to `tracemalloc` (much is non-Python-level),
and `malloc_trim` reclaimed ~0% — all consistent with live retained objects.

**Fix:** override `Gen3Player.reset_battles()` (src/agents/inference/player.py) to also prune
`_trackers`/`_stall_loggers` to the still-live `_battles` tags. Regression guard:
`src/agents/training/selfplay_opponent_leak_fuzz_test.py` (real bridge self-play battles, asserts
the opponent caches stay bounded; trips after 4 battles on pre-fix code). Verified: repro RSS went
from climbing 601→652 MB/45 eps to flat ~600 MB; 595 inference+training unit tests pass.

**How to reproduce/diagnose this class of leak:** isolate ONE env-worker rollout in-process via
the bridge and instrument it (the throwaway `/tmp/mem_monitor/leak_*.py` harnesses did this).
Watch domain-object/cache counts per episode SYNCHRONOUSLY — a reporter thread walking
`gc.get_objects()` concurrently crashes the rollout with a C-level `SystemError`. Heuristic-only
rollout was flat (519 MB/60 eps), which is what localized the leak to the self-play opponent path.
