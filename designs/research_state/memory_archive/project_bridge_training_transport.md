---
name: project_bridge_training_transport
description: "In-process BattleStream bridge as a flag-guarded RL TRAINING transport (--use-showdown-bridge) — poke-env issue #907; built & validated 2026-06-03"
metadata: 
  node_type: memory
  type: project
  originSessionId: be1aa377-9fc7-4359-b791-9d4288eebe29
---

> **Archived 2026-09-08** — --use-showdown-bridge era; superseded by --use-bridge rust as the DEFAULT transport (root CLAUDE.md). Preserved verbatim; nothing below is current.

`--use-showdown-bridge` (default **False**, training envs only) swaps the **training** transport
from a websocket Showdown server to an in-process `BattleStream` subprocess per env — no server,
no port, no `/challenge` connection storm, deterministic delivery (poke-env issue #907; the
issue author's bottleneck was per-episode battle startup/reset, not per-turn logic). Built &
validated 2026-06-03 (this is the implementation of the plan in [[project_local_sim_bridge]]).

**Design = bridge as a drop-in transport, NOT a new env.** `PokeEnv` already inverts control via
two `_EnvPlayer` agents + `battle_queue`/`order_queue`; the websocket is only the byte transport.
`src/utils/bridge/bridge_session.py` (`BridgeSession` + `attach_bridge_transport`) reassigns both
agents' `ps_client` to a `BattleStreamClient` and intercepts the ONE battle-start seam —
`agent1.battle_against` (the `/challenge` call in `PokeEnv.reset`) — with a coroutine that spawns
the bridge, sends `START` + both packed teams, and launches a background reader. The whole
obs/reward/mask/wrapper/reward-manager stack is unchanged.

**Key correctness insight:** the reader mirrors poke-env's websocket `listen()` — it FIRES each
chunk's `feed()` as a task (never `await`s). Mandatory: `_EnvPlayer._choose_move` blocks on
`order_queue` awaiting SB3's action, so awaiting feed inline stalls the reader and deadlocks the
OTHER side (whose request never reaches `reset()`). Safe because `_handle_message` serializes
same-battle handling under a per-battle `asyncio.Lock` (`ps_client.py` ~line 200) and the two
sides use independent clients. Everything binds to `env._loop` (PokeEnv's per-env loop, NOT the
global `POKE_LOOP` the runner uses).

**Status: training + eval both bridged, persistent child default, validated.**

- **Training transport** — `BridgeSession` (background pump). Child is **persistent by default**
  (`attach_bridge_transport(persistent=True)`): ONE long-lived Node child per env reused across
  episodes; each `reset()` sends a fresh `START` (the JS rebuilds a clean `BattleStream`; opt-in
  `"persistent":true` in START, backward-compatible — `run_local_battles` still spawn-per-battle).
  A single reader owns stdout; `__END__`→`_battle_ended` Event gates the next `START`/tag-swap.
  `spawn-per-battle` kept as fallback/A-B.
- **Eval** — same `--use-showdown-bridge` flag, but via the *synchronous* `run_local_battles`
  (eval is greedy-vs-bot, no SB3 action → no inversion-of-control needed): `eval_one_matchup`
  swaps `battle_against`→`run_local_battles`; threaded as a `use_showdown_bridge` cfg key through
  `PerOpponentEvalCallback`/`SelfPlayCallback`→`eval_worker` + the final `evaluate_model_random`.
  So a whole run needs **no Showdown server at all**.
- **Validated:** integration tests (real `Gen3Env`, both modes, forced-switch path clean; one
  reused child PID across 4 episodes), unit transport-swap guards, full serverless
  `--debug --use-showdown-bridge` smoke (Training complete → final eval win rates, no server),
  22 `src/utils/bridge/` tests + 55 eval + 35 selfplay tests pass.
- **Perf (single-env latency A/B, `bridge_vs_websocket_latency_benchmark.py`, RandomPlayer, no
  GPU, loaded box):** websocket ~13 ms/step → persistent bridge ~6 ms/step (**~2.1×**);
  spawn-per-battle only ~11 ms/step — **reusing the child is the win** (re-loading the Showdown
  sim per episode otherwise eats it; confirms #907's "reset/startup overhead is the bottleneck").
- **Bug fixed:** `BridgeSession.close()` ran `proc.kill()` from the MAIN thread while the asyncio
  subprocess belongs to `env._loop` → silently leaked the Node child. Now `os.kill(pid, SIGKILL)`
  (thread-agnostic). **Lesson: another job on this box (`/tmp/distill/gen_data.py`) also spawns
  `local_sim_bridge.js` children — NEVER `pkill -f local_sim_bridge.js` system-wide; kill only your
  own PIDs.**

**Fuzz/soak — DONE, zero failures.** `bridge_session_fuzz_test.py` (per-phase timing + a
`--workers N` multi-env soak). Single-env: **32,400 episodes on ONE reused child, 0 failures**.
64-child soak: 64 concurrent persistent children, thousands of episodes, **0 failures, 0 child
respawns**. **The 42s-episode "outlier" is pure CPU contention, NOT a transport stall** — the
instrumentation (`BridgeSession._last_end_wait_s`, the inter-battle `__END__` wait) showed
`end_wait` ≤ 9ms on EVERY episode even under 64-way saturation; slow episodes are 100%
`step_loop` (the sim round-trip starved of CPU). Transport is correct under extreme concurrency.

**Cleanup gotchas (learned):** (a) `pkill -f "bridge_session_fuzz_test"` kills only the soak
PARENT — the multiprocessing **spawn** workers have a bootstrap cmdline (no script name), so they
survive; kill them by PID (parents of the node children) or by process group. (b) A 64-worker
soak does NOT OOM an 89GB box (peaked ~37GB), but it heavily loads CPU — be considerate of any
live training. (c) The `:8001` server/training were already down before the soak (box idle at
~7GB) — the soak didn't crash them.

**Multi-env end-to-end FPS — MEASURED (2026-06-04): only ~5%.** Full `train_rl_agent` A/B on an
idle box, `n_envs=64`, production hyperparams, CUDA, vs-bots, 3 rollouts each, sequential:
steady-state **~1192 fps bridge vs ~1140 fps websocket (1.05×)**; bridge also started ~17% faster
(87s vs 105s — node spawns beat the `/challenge` connection storm) and ran steadier. So the
single-env 2.1× does NOT translate to training scale: at 64 envs the 16-thread box is
CPU-saturated and the SubprocVecEnv barrier waits on the slowest worker, so oversubscription hides
the per-step transport latency (the [[project_throughput_profile]] "n_envs is not the FPS lever"
prediction, confirmed). **The bridge's value is OPERATIONAL, not FPS** (no server → no
[[project_showdown_server_memory_growth]] leak, no connection storm, no port tuning, deterministic,
faster restarts). Default stays websocket. Benchmark: `train_rl_agent` A/B via /tmp/fps_bench.sh
pattern (read SB3 rollout `total_timesteps`/`time_elapsed` for steady-state incremental fps).

**Persistent-child lifecycle — MEASURED + OPTIMIZED (2026-06-04).** Two rules (both tested):
(1) a **dead** child **crashes** the env (reader latches `_child_crashed` on stdout EOF → next
`reset()` raises → launcher restart; NO in-place recovery — resuming risks a corrupted PPO
transition; user-confirmed crash-over-corruption). (2) a **healthy** child is **recycled** every
`recycle_every` battles — a BACKSTOP, not routine: `bridge_heap_growth_benchmark.py` measured RSS
**flat** (~189 MB fresh → +36 MB one-time V8 warmup → ~229 MB, ~0 growth over thousands of
battles; plateaus ~battle 100). A child plays only ~2150 battles in the launcher's 3h window, so
the bridge does NOT reintroduce [[project_showdown_server_memory_growth]] and needs **no recycle
within 3h** — the 3h restart owns the lifecycle. Default `recycle_every=5000` (was 1000, which
fired ~2×/window needlessly) never fires under the launcher; only caps marathon/no-launcher runs.
Recycle is crash-safe (spawn-new-first, reader captures own event, `_teardown` doesn't touch the
shared `_stderr_task`). Tests: `test_persistent_child_recycles_after_n_battles`,
`test_dead_persistent_child_crashes_no_recovery`. 24 bridge tests pass.

**FPS at 48 envs: +20%** (bridge 1214 vs ws 1013) — vs +5% at 64 and +114% single-env, a clean
trend (more envs → more oversubscription hides transport latency).

**Concurrent bridge eval — DONE (2026-06-04).** `run_local_battles(concurrency=N)` overlaps N
games (each its own sim subprocess) but serializes each battle's team→creation under a `start_lock`
released the instant both battle objects exist — mirrors the server's per-battle semaphore
(`_send_challenges` gates on `_battle_semaphore`, `_create_battle` reads `_current_packed_team`
BEFORE releasing it), so the shared team can't race. `concurrency=1` = unchanged sequential path
(zero fuzz-suite risk). ~1.8× for sim-bound RandomPlayer matchups (more for model-bound). Threaded
through `eval_one_matchup`/`run_eval`/`eval_worker` (=`_EVAL_SUBPROCESS_CONCURRENCY`=5) + final eval
(cap 8). Guard: `test_concurrent_local_battles_complete` (asserts valid 6-mon own-team per battle).

**Open:** (1) Upstream PR to poke-env #907 (design: `designs/ai_v5/design_local_sim_bridge_transport.md`).
(2) True ws-vs-bridge obs parity (hard to seed the server battle; largely redundant with existing
fuzz stream-parity). (3) spawn-per-battle mode in `bridge_session.py` is now vestigial (persistent
is the proven default) — candidate for removal to cut complexity, but kept as A-B baseline/fallback.
