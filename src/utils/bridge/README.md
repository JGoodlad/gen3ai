# Pokémon Showdown Bridge

This directory contains the bridge logic used to access the Pokémon Showdown simulation library directly from Python, without requiring a running server.

## Overview
By bridging Python to Node.js, we can utilize the actual `pokemon-showdown` codebase (located in `deps/`) as a library. This allows us to perform complex operations like **Team Validation** using the official Smogon rules and logic, but with the performance of a local function call.

## Components

### Team validation (request/response)
1.  **`validate_team.js`**: A Node.js script that imports the Showdown `TeamValidator` and `Teams` modules. It reads a team and format from `stdin` and outputs a validation result as JSON.
2.  **`team_validator.py`**: The Python wrapper that manages the subprocess communication with the Node bridge.

### Local battle (streaming) — run whole battles with no Showdown server
A second bridge runs *entire battles* in-process via Showdown's `BattleStream`, so a
poke-env `Player` can play with **no websocket server, no port, no usernames, no
matchmaking**. Unlike the team validator, it relays the **protocol text** each side sees,
so it feeds poke-env's parser exactly as the live server would — the parsing/encoder
pipeline runs byte-for-byte the same; only the transport changes (poke-env issue #907).

1.  **`local_sim_bridge.js`**: A Node streaming relay. `START`/`CHOOSE`/`FORCELOSE`/`END`
    on `stdin`; per-side, base64-framed protocol chunks (`p1 <b64>` / `p2 <b64>`) on
    `stdout`. One battle per process. Uses `getPlayerStreams()` so Showdown does the
    per-side channel demux. Accepts an optional fixed PRNG `seed` for reproducible battles.
2.  **`battle_stream_client.py`**: `BattleStreamClient(PSClient)` — a poke-env transport
    that subclasses the (vendored) `PSClient` **from outside** `src/poke_env/` (it modifies
    no poke_env file). No websocket; translates poke-env's `/choose …` onto the bridge and
    no-ops websocket ceremony (`/utm`, `/timer on`, `/leave`, `/challenge`, …).
3.  **`local_battle_runner.py`**: `run_local_battles(player1, player2, n_battles, *,
    seed=None)` — a drop-in for `player1.battle_against(player2, n_battles=…)`. It spawns one
    bridge per battle, fabricates the `>battle-…`/`|init|` room header the sim does not emit,
    and routes each side's protocol to the right player's client. Build the players with
    `start_listening=False` (no websocket opens); the runner swaps in the bridge transport.

This powers the `*_fuzz_test.py` suite. (A few timing-sensitive checks stay on the live
server as `*_fuzz_e2e_test.py` — e.g. `effectiveness_fuzz_e2e_test`.)

### RL-training transport (`BridgeSession`) — the bridge as a gym env transport

`run_local_battles` is a *synchronous driver*: it owns the battle loop and pulls each decision
from a self-contained `choose_move`. That shape can't host SB3, whose action arrives from
*outside* the env via `step()`. `BridgeSession` (`bridge_session.py`) closes that gap: it makes
the bridge a **drop-in transport for poke-env's `PokeEnv`**, so the *exact same* obs / reward /
mask / wrapper stack trains with no websocket, no server, no port, no `/challenge` connection
storm.

`PokeEnv` already inverts control with its two `_EnvPlayer` agents and their
`battle_queue` / `order_queue` handshake — the websocket is only the byte transport underneath.
`attach_bridge_transport(env, battle_format=…)` swaps that transport on a freshly-built env
(built `start_listening=False`):

1.  Reassigns both `_EnvPlayer.ps_client` to a `BattleStreamClient` (sides p1 / p2).
2.  Intercepts the one battle-start seam — `agent1.battle_against` (the `/challenge` handshake
    `PokeEnv.reset()` calls) — with a coroutine that spawns the bridge subprocess, sends `START`
    with both packed teams, and launches a **background reader**.
3.  The reader mirrors poke-env's own websocket `listen()`: it reads one protocol chunk, frames
    it with the room header, and **fires** the feed as a task (never `await`s it). That is
    mandatory — `_EnvPlayer._choose_move` blocks on `order_queue` awaiting SB3's action, so
    awaiting the feed inline would stall the reader and deadlock the *other* side. It is safe
    because `_handle_message` serializes same-battle handling under a per-battle `asyncio.Lock`,
    so chunks for one battle stay strictly ordered, and the two sides use independent clients.

Two child-lifecycle modes (`attach_bridge_transport(persistent=…)`): **persistent** (default)
reuses ONE long-lived Node child per env across every episode (a fresh `START` rebuilds a clean
`BattleStream`); **spawn-per-battle** spawns a fresh child per battle. Persistent is the win — a
single-env transport-latency A/B (`bridge_vs_websocket_latency_benchmark.py`, RandomPlayer
opponent, no GPU) measured **~13.0 ms/step websocket → ~6.1 ms/step persistent bridge (~2.1×)**,
while spawn-per-battle was only ~11.3 ms/step: re-loading the Showdown sim into a fresh Node
process every episode eats nearly all the savings, so reusing the child is what unlocks the gain
(matching issue #907's "reset/startup overhead is the bottleneck"). It is **flag-guarded**:
`python -m main.launcher … --use-showdown-bridge` (or `train_rl_agent.py --use-showdown-bridge`),
default off (websocket). Guarded by `bridge_session_test.py` (transport-swap contract, no server)
and `bridge_session_integration_test.py` (a real `Gen3Env` plays full episodes over the bridge).

**Persistent-child lifecycle (two rules, both guarded):**
- **A dead child CRASHES the env, no in-place recovery.** If the Node child exits mid-run, lost /
  inconsistent battle state means resuming could feed PPO a corrupted transition — so the reader
  latches `_child_crashed` on stdout EOF and the next `reset()` raises (the launcher restarts from
  checkpoint). Same crash-over-corruption rule as the trainee's stale-decision path.
- **A healthy child is RECYCLED every `recycle_every` battles — a backstop, not a routine need.**
  `bridge_heap_growth_benchmark.py` measured a child's RSS **flat**: ~189 MB fresh → a one-time
  ~+36 MB V8 warmup → **~229 MB with ~0 growth over thousands of battles** (V8 GC reclaims the
  per-battle `BattleStream`). At production scale a child plays only ~2150 battles in the
  launcher's 3h restart window, so the default `recycle_every=5000` **never fires under the
  launcher** (the 3h restart owns the lifecycle); it only caps marathon / no-launcher direct runs.

**Eval rides the same flag, via the *synchronous* driver, not `BridgeSession`.** Eval is a pure
synchronous-decision matchup (a greedy trainee vs a bot/sentinel — no SB3-supplied action), so it
doesn't need the inversion-of-control machinery: the eval worker's `_play_unit` just calls
`run_local_battles` instead of `battle_against` when `use_showdown_bridge` is set (players built
`start_listening=False`). The flag threads as a `use_showdown_bridge` config key through
`PerOpponentEvalCallback` / `SelfPlayCallback` → `eval_worker`, plus the end-of-training
`evaluate_model_random`. So `--use-showdown-bridge` makes a whole run — training **and** eval —
need no Showdown server.

## Why use a Bridge?
- **Serverless**: No need to start or manage a Pokémon Showdown server process.
- **No contention**: Each call/battle is fully isolated — no shared server lifecycle, no
  username collisions, no port. Multiple fuzz tests can run at once without fighting.
- **Speed**: Local function calls beat websocket round-trips (the fuzz suite is *faster*
  on the bridge than on the server).
- **Accuracy**: It uses the *exact* same Showdown code as the live server (same `deps/` files).
- **Reproducible**: the battle bridge accepts a fixed PRNG seed.

## Usage

### Team validation
```python
from src.utils.bridge.team_validator import validate_team_locally

result = validate_team_locally("gen3ou", team_text)
if result["valid"]:
    print("Team is valid!")
else:
    print(f"Errors: {result['errors']}")
```

### Local battle (used by the `*_fuzz_test.py` suite)
```python
from utils.bridge.local_battle_runner import run_local_battles

# Players must be built with start_listening=False so no websocket is opened.
await run_local_battles(my_player, opponent, n_battles=40)   # no `npm run showdown`
```

### RL-training transport (used by `train_rl_agent.py --use-showdown-bridge`)
```python
from utils.bridge.bridge_session import attach_bridge_transport

# Build the env with start_listening=False, then swap in the bridge transport.
env = Gen3Env(mappings, battle_format="gen3ou", team=teambuilder, start_listening=False)
attach_bridge_transport(env, battle_format="gen3ou")
# env now trains with no websocket / server — everything above the transport is unchanged.
```
