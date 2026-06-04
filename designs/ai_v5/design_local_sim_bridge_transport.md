# Design: Local Sim Bridge as a Training/Eval Transport (+ poke-env upstream PR)

The in-process **BattleStream bridge** lets training and eval run with **no Pokémon Showdown
server** — each env owns a local sim subprocess and feeds poke-env the identical `|...|` protocol
stream over stdio. This is the poke-env [issue #907](https://github.com/hsahovic/poke-env/issues/907)
idea: a *transport swap* under an unchanged `Player`/`Battle`/parser stack. Gated by
`--use-showdown-bridge` (default off).

This doc has two halves:

1. **As-built (local).** What shipped in this repo — an *adapter* that swaps the transport from
   outside vendored poke-env, why it's shaped the way it is, and the measured win.
2. **Upstream design.** The cleaner abstraction we would contribute back to poke-env #907, how it
   differs from our adapter, and the generalization gap. (Issue #907 is open with **no maintainer
   reply and no reference implementation** — this doc *is* the design we'd attach to it.)

Relevant because the self-play / league chapter (ai_v5) is **throughput-bound** and
**server-coupled**: the live self-play run spends ~86% of wall on rollout collection, carries a
persistent Showdown node server whose RAM grows unbounded over a run, and pays a `/challenge`
connection storm at every restart (see `[[project_throughput_profile]]`,
`[[project_showdown_server_memory_growth]]`). The bridge removes the server entirely.

---

## Motivation

poke-env is a **state tracker over a websocket transport**. For local RL we don't need the
server, the websocket, the matchmaking, or the login handshake — only the protocol stream the
server would have sent. Issue #907's author (whose framing we adopt) identifies the real
bottleneck as **battle startup / reset overhead**, not per-turn decision logic.

What the server transport costs us today:

- **A persistent node server** per training session (the `:8001` instance) whose RSS grows
  unbounded over a multi-day run; launcher restarts recycle only the Python child, never the node
  server.
- **A `/challenge` connection storm** at startup and at every restart — `challenge_timeout` had
  to be tuned to 120 s so parallel envs don't crash `reset()` with "Agent is not challenging".
- **Env-size guessing.** `--n-envs` is implicitly capped by one shared server's capacity (2
  websockets/env, the connection storm). With each env owning an in-process sim, sizing becomes a
  pure local-resource question (RAM/CPU).
- **Async-delivery nondeterminism** — the whole `race_get` / stale-decision / force-switch
  deadlock saga exists to paper over the server's async message timing.

The bridge removes all four. It is **not** a free FPS multiplier (the dominant per-step CPU cost
in self-play is the opponent NN forward, transport-independent — see throughput profile), but it
deletes the server-coupling tax and tightens step-latency variance, which *is* the lever in a
latency-bound system.

---

## As-built (local)

### Two consumers, two integration patterns

The bridge has **two** distinct callers because they have two distinct control-flow needs:

| Pattern | File | Used by | Control flow |
|---|---|---|---|
| **Synchronous driver** | `local_battle_runner.py` → `run_local_battles(p1, p2, n)` | fuzz tests, **eval** | Owns the battle loop; *pulls* each decision from a self-contained `choose_move`. The issue's simple "feed protocol" model. |
| **Async background-pump transport** | `bridge_session.py` → `BridgeSession` | **training env** | A background reader feeds battle state into the env's queues; the action arrives from *outside* via SB3's `step()`. |

The split matters: eval is a greedy-trainee-vs-bot matchup (the decision is internal, synchronous)
so `run_local_battles` is a faithful drop-in for `battle_against`. **Training is not** — SB3
supplies the action asynchronously, so the synchronous driver can't host it. That gap is the
engineering #907 leaves unspecified, and `BridgeSession` is the answer.

### Shared substrate

- **`local_sim_bridge.js`** — a Node relay: `START`/`CHOOSE`/`FORCELOSE`/`END` on stdin; per-side
  base64-framed protocol chunks (`p1 <b64>` / `p2 <b64>`) on stdout. Uses `getPlayerStreams()` so
  Showdown does the per-side channel demux. A fresh `BattleStream` per `START` → no cross-battle
  state. **Persistent mode** (opt-in `"persistent":true` in START): on battle end the process
  emits `__END__` and *resets* instead of exiting, so one child runs many sequential battles.
  Backward-compatible — `run_local_battles` and the seed-repro test never set the flag and exit on
  `__END__` as before.
- **`battle_stream_client.py`** — `BattleStreamClient(PSClient)`: subclasses the **vendored**
  `PSClient` *from outside* `src/poke_env/`. No `.websocket`; translates `/choose …` onto the
  bridge stdin and no-ops websocket ceremony (`/utm`, `/timer on`, `/leave`, `/challenge`, …).

### `BridgeSession` — the gym transport (the hard part)

`PokeEnv` already inverts control: two `_EnvPlayer` agents block on `order_queue.async_get()`
inside `_choose_move` while a background `listen()` loop feeds `battle_queue`. The websocket is
only the byte layer under that. `attach_bridge_transport(env, battle_format=…, persistent=True)`
swaps the byte layer on a freshly-built env (built `start_listening=False`):

1. Reassigns both `_EnvPlayer.ps_client` to a `BattleStreamClient` (sides p1/p2).
2. Intercepts the **one** battle-start seam — `agent1.battle_against` (the `/challenge` call inside
   `PokeEnv.reset`) — with a coroutine that spawns/uses the bridge child, sends `START` + both
   packed teams, and (re)starts the reader.

Three design points that are load-bearing:

- **Fire-and-forget feed, never awaited.** The reader reads a chunk, frames it with the room
  header, and `ensure_future(client.feed(...))` — it does **not** await. Awaiting would stall the
  reader on `_choose_move`'s `order_queue` wait and deadlock the *other* side (whose request would
  never reach `reset()`). This mirrors poke-env's own websocket `listen()` (`ps_client.py`, which
  `create_task`s each `_handle_message`), and is **safe + order-preserving** because
  `_handle_message` serializes same-battle handling under a per-battle `asyncio.Lock`
  (`ps_client.py`), and the two sides use independent clients.
- **Everything on `env._loop`.** `PokeEnv` builds its **own** per-env loop (`env.py`, not the
  global `POKE_LOOP` the synchronous runner uses); the clients, reader, and choice-writes all bind
  to it, matching the queues `_EnvPlayer` created against it.
- **Persistent child reuse + the `__END__`→`_battle_ended` gate.** A single long-lived reader owns
  stdout for the child's whole life. Between battles, `_send_start` swaps the battle tag *before*
  writing `START` (the child can't emit the new battle's chunks until it receives `START`), and
  the next `_battle_against` awaits a `_battle_ended` Event the reader sets on `__END__` — so a
  finished battle's tail can never race the next battle's tag swap. Unique process-global tags
  (`itertools.count`) prevent the "phantom 7th pokemon" stale-battle reuse class of bug
  (`[[project_bridge_unique_battle_tags]]`).

### Flag wiring

`--use-showdown-bridge` (default `False`) threads like `server_config`:

- **Training** — the env factory builds `Gen3Env(..., start_listening=not use_bridge)` then calls
  `attach_bridge_transport`.
- **Eval** — `eval_one_matchup` swaps `battle_against` → `run_local_battles`; eval players built
  `start_listening=False`; the flag rides as a `use_showdown_bridge` config key through
  `PerOpponentEvalCallback` / `SelfPlayCallback` → `eval_worker`, plus the end-of-training
  `evaluate_model_random`. So **a whole run needs no Showdown server**.

### Measured win (single-env transport A/B)

`bridge_vs_websocket_latency_benchmark.py` — single env, `RandomPlayer` opponent (no GPU, no
policy net), so it isolates the transport. On a loaded box (the protected run was live):

| Transport | ms/step | vs websocket |
|---|---|---|
| websocket (server) | 13.0 | 1.0× |
| bridge, **spawn-per-battle** | 11.3 | 1.15× |
| bridge, **persistent** | 6.1 | **2.14×** |

**The persistent child is the whole win.** Spawn-per-battle barely beats the websocket because it
re-loads the Showdown sim into a fresh Node process every episode — exactly #907's "startup/reset
overhead is the bottleneck," confirmed. Reusing the child (load the sim once) is what realizes it.

**Caveat (do not over-claim):** this isolates the *transport*. The end-to-end training-FPS gain at
production scale is far smaller — see below.

### End-to-end FPS at production scale (the honest number)

A full `train_rl_agent` A/B on an idle box — `n_envs=64`, production hyperparams, CUDA, vs-bots,
3 rollouts each, sequential — measured **steady-state ~1192 fps bridge vs ~1140 fps websocket =
~5% (1.05×)**. The bridge also **started ~17% faster** (87 s vs 105 s to first rollout — node
spawns beat the `/challenge` connection storm) and ran steadier (1111→1192 vs the websocket's
noisier 978→1140).

So the single-env 2.1× **does not** translate to training-scale FPS: at 64 envs the box (16
threads) is CPU-saturated and the `SubprocVecEnv` barrier waits on the slowest worker, so
oversubscription hides the per-step transport latency (exactly the throughput-profile prediction
that *n_envs is not the FPS lever*). The bridge's FPS edge grows as `n_envs` drops (less
oversubscription to hide latency), but at the n_envs training actually uses it's ~5%.

**The case for the bridge is therefore operational, not throughput:** no server at all → no
`[[project_showdown_server_memory_growth]]` RAM-growth leak, no connection storm, no port /
`challenge_timeout` tuning, no "never kill :8001" footgun, deterministic delivery, faster
restarts. Default stays websocket (opt-in).

### Persistent-child lifecycle (measured + optimized)

Two rules govern the long-lived child:

- **Death ⇒ crash, no in-place recovery.** A child that exits mid-run means lost / inconsistent
  battle state; resuming could feed PPO a corrupted `(obs, action) → (reward, next_obs)`. So the
  reader latches `_child_crashed` on stdout EOF and the next `reset()` raises — the launcher
  restarts from checkpoint. Same crash-over-corruption rule as the trainee's stale-decision path.
  (Robust to returncode-reap timing: the latch is set on the EOF break, not via `returncode`, and
  recycle/close *cancel* the reader so they don't trip it.)
- **Healthy recycle ⇒ a backstop, not a routine need.** `bridge_heap_growth_benchmark.py` measured
  one child's RSS **flat**: ~189 MB fresh → a one-time ~+36 MB V8 warmup → **~229 MB with ~0
  growth** over thousands of battles (plateaus by ~battle 100; V8 GC reclaims the per-`START`
  `BattleStream`). At production (64 envs, ~1200 fps, ~94 steps/episode) a child plays only ~2150
  battles in the launcher's **3h restart window**, over which heap growth ≈ 0. So the bridge does
  **not** reintroduce the server's RAM-creep, and **no recycle is needed within 3h** — the 3h
  restart owns the lifecycle. `recycle_every` default **5000** sits comfortably above a 3h window,
  so it never fires under the launcher; it only bounds marathon / no-launcher direct runs. (The old
  1000 default would have fired ~2×/window needlessly, churning a healthy child — corrected.)

The recycle swap is itself crash-safe: it spawns the fresh child FIRST, then cancels the old reader
(which captured its OWN `ended_event`, so it can't set the new child's) and tears the old child
down explicitly; `_teardown` no longer touches the shared `_stderr_task` (which would kill the live
child's drain mid-recycle).

### Validation

- `bridge_session_test.py` — transport-swap contract (no server, no battle).
- `bridge_session_integration_test.py` — a real `Gen3Env` plays full episodes over the bridge,
  **both modes parametrized**; persistent mode asserts one reused child PID across episodes.
- `bridge_session_fuzz_test.py` — long-running edge-case fuzz: thousands of episodes, asserting
  per-episode invariants (finished/turn>0/role, finite obs, valid mask), one reused child, no
  deadlock (per-episode wall + step caps), with injected mid-battle forfeit-resets and natural
  force-switch / 250-turn stall coverage.
- Serverless `--debug --use-showdown-bridge` smoke: training → final eval win rates, no server.
- **Bug fixed:** `BridgeSession.close()` ran `proc.kill()` from the main thread while the asyncio
  subprocess belongs to `env._loop` → silently leaked the child. Now `os.kill(pid, SIGKILL)`
  (thread-agnostic). *Operational note:* other jobs on the box also spawn `local_sim_bridge.js`
  children — kill only your own PIDs, never a system-wide `pkill`.

---

## Upstream design (the poke-env PR for #907)

Our local code is a deliberate **adapter**: it swaps `ps_client` *after* construction and
monkeypatches `battle_against`. That's fine for us (rapid iteration, gen3-shaped) but is **not how
you'd design it for upstream**. The PR-quality version replaces both hacks with a first-class
**transport backend** chosen at construction.

### Proposed abstraction

A `PSClient`-level transport interface, selected by a `backend=` (or `server_configuration` of a
new kind) parameter on `Player` / the env:

```
class Backend(Protocol):
    async def start(self) -> None: ...          # open websocket OR spawn the sim child
    async def send(self, room: str, msg: str) -> None:   # "/choose ...", "/forfeit", ...
    async def listen(self) -> None: ...         # pump inbound protocol → _handle_message
    async def start_battle(self, p1, p2, fmt, *, seed=None) -> None:  # challenge OR START
    async def close(self) -> None: ...

class WebsocketBackend(Backend): ...   # the current PSClient internals, unchanged behaviour
class LocalSimBackend(Backend): ...    # BattleStream child + the background pump
```

- `PSClient` keeps the **websocket backend as default** → byte-for-byte unchanged for every
  existing user.
- The **env**'s `reset()` calls `backend.start_battle(...)` instead of hardcoding
  `agent1.battle_against(agent2)` — so the same `PokeEnv`/`SinglesEnv` hosts either transport with
  no `if websocket:` branches in user code. This is the generalization of our `battle_against`
  intercept.
- The **gym/RL integration is the contribution that #907 omits**: the `LocalSimBackend.listen()`
  pump (fire-and-forget feed under the per-battle lock, on the env loop) is what makes
  `PokeEnv`/`SinglesEnv` trainable serverless. That, plus persistent-child reuse, is the novel,
  upstreamable core.

### Generalization gap (adapter → PR)

| Concern | Our adapter | PR-ready |
|---|---|---|
| Injection | reassign `ps_client` + monkeypatch `battle_against` post-construction | `backend=` parameter; no post-construction surgery |
| Formats | gen3ou only (no teampreview) | teampreview (`/team`), other gens, doubles/VGC order serialization |
| Concurrency | one battle per child (env uses `max_concurrent_battles=1`) | N concurrent battles per backend, or document the 1-per-child model |
| Lifecycle | `os.kill` from main thread; gen3 env owns it | backend-owned `close()`, context-manager, crash/respawn policy |
| Tests/style | our pytest + fuzz conventions | poke-env's test suite + CI + type conventions |
| Scope | training + eval glue is gen3-specific | generic `Player.battle_against`/env path works for any user |

### Decision

Keep the validated local adapter; prepare the **clean backend** as a *separate* branch shaped for
a poke-env PR. They're not mutually exclusive, and the community-spirited move is worth making now
that the approach is proven. The honest risk: **#907 has no maintainer reply**, so a PR may sit —
in which case we maintain a fork *anyway* (we already patch vendored poke-env: the `race_get`
force-switch deadlock fix, the Snatch handler). Sequencing — prototype-and-validate locally first,
then generalize and upstream — is the right order regardless of how fast review moves.

---

## Open items / next local improvements

1. ~~Multi-env end-to-end FPS run on an idle box~~ — **DONE: ~5% at `n_envs=64`** (see the FPS
   section above). The transport is not the FPS lever at production scale; default stays websocket.
2. ~~Concurrent bridge eval~~ — **DONE.** `run_local_battles(concurrency=N)` overlaps N games
   (each its own sim subprocess) but serializes each battle's team→creation under a `start_lock`
   released the instant both battle objects exist — mirroring the server's per-battle semaphore, so
   the shared `_current_packed_team` can't race. ~1.8× for sim-bound matchups (more for model-bound).
   `concurrency=1` is the unchanged sequential path (zero risk to the fuzz suite). Threaded through
   `eval_one_matchup` / `run_eval` / `eval_worker` (= `_EVAL_SUBPROCESS_CONCURRENCY`) + the final
   eval (capped at 8). Guard: `test_concurrent_local_battles_complete` (12 games × 4, all finish
   with a valid 6-mon own-team).
3. **Multi-env persistent-child stress** — the fuzz covers one env × many episodes; add a
   `SubprocVecEnv`-shaped soak (many envs × many episodes) to stress 64 concurrent children.
4. **Retire race machinery where the bridge makes it dead code** — deterministic in-process
   delivery should let us shrink `_settle_opponent_battle` / `race_get` timeout handling *on the
   bridge path* (keep it on the websocket path).
5. **Upstream PR branch** — the `Backend` abstraction above, generalized + tested to poke-env
   conventions, attached to #907.

---

## Files

- `src/utils/bridge/bridge_session.py` — `BridgeSession`, `attach_bridge_transport` (NEW)
- `src/utils/bridge/local_sim_bridge.js` — persistent-mode relay (MODIFIED)
- `src/utils/bridge/battle_stream_client.py` — `BattleStreamClient(PSClient)` (pre-existing)
- `src/utils/bridge/local_battle_runner.py` — `run_local_battles` synchronous driver (pre-existing)
- `src/utils/bridge/bridge_vs_websocket_latency_benchmark.py` — transport A/B (NEW)
- `src/utils/bridge/bridge_session_{test,integration_test,fuzz_test}.py` — guards + fuzz (NEW)
- `src/main/train_rl_agent.py` — `--use-showdown-bridge`; bridge-mode env + final-eval wiring
- `src/agents/training/eval_callback.py`, `selfplay_callback.py`, `src/main/eval_worker.py` —
  `use_showdown_bridge` cfg key + `run_local_battles` eval path

## Verification

- `pytest src/utils/bridge/ -q` (22, both modes parametrized) + eval/selfplay suites (90+).
- `bridge_session_fuzz_test.py <N>|<N>m` — long edge-case soak (both modes via `--spawn`).
- `bridge_vs_websocket_latency_benchmark.py` — transport A/B (websocket arm needs a `9XXX` server).
- Serverless `--debug --use-showdown-bridge` — full train→eval with no server bound.
