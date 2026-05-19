# Implementation: Step 1 — Replay Collection

This step adds a passive spectator client that connects to Pokémon Showdown, discovers
active Gen 3 OU battles, and saves the complete battle logs to disk for offline use.
The primary motivation is building a human-expert dataset for imitation learning
pre-training in later steps.

---

## Motivation

The current RL agent learns entirely from self-play, which means it has to rediscover
basic Gen 3 OU fundamentals (common cores, speed tiers, hazard play) from scratch every
run. Human expert replay logs are a cheap source of strong prior behaviour that can
bootstrap the policy before RL fine-tuning.

The Showdown spectator protocol is well-suited to this: a guest connection (no login
required) can query the roomlist for active battles in any format and join them as a
read-only observer. Crucially, joining mid-battle gives the full history from turn 1 —
the server replays the entire log on `onConnect` — so no battles are partially captured.

---

## Protocol Details

**Finding battles:** `/query roomlist gen3ou` returns a JSON payload via
`|queryresponse|roomlist|{...}` with up to 100 active rooms that have both players
present. Format filtering is an exact match on the format ID. No authentication is
required in normal server mode (`Config.emergency = false` means all connections are
trustable).

**Joining:** `/join battle-gen3ou-123456` triggers the server's `onConnect` handler,
which sends the full battle scrollback prepended with a `|t:|{timestamp}` header line.
The spectator then receives all subsequent turns in real time until `|win|` or `|tie`.

**Rate limits:** No limit on roomlist queries or concurrent spectators per room. The
only relevant throttle is 600 ms between outbound messages (all `/join` commands share
the same per-user queue). The implementation enforces a 10-second join interval
(well above this floor) and caps at 10 concurrent rooms to stay well within server
limits.

---

## What Was Built

### `src/poke_env/ps_client/ps_client.py` — small addition

Added `QueryResponseCallback` type and `on_query_response` optional parameter, matching
the existing `on_battle_message` callback pattern. `_handle_message` now dispatches
`|queryresponse|` to this callback instead of silently dropping it.

### `src/poke_env/spectator/spectated_battle.py`

`SpectatedBattle` is a pure data object — no file I/O, no dependency on the battle state
machine. It accumulates raw Showdown protocol lines and exposes them as `log_text`. The
caller decides what to do with a completed battle.

```
SpectatedBattle
  battle_tag: str
  finished: bool
  winner: str | None        — None means tie
  log_text: str             — full Showdown-format log, one protocol line per line
```

`finish()` is idempotent. `add_lines()` is a no-op after `finish()`. Neither method
touches the filesystem.

### `src/poke_env/spectator/spectator_client.py`

`BattleSpectator` is an async generator that yields `SpectatedBattle` objects
indefinitely as battles complete. All state-machine logic lives here; the caller only
sees a stream of finished battles.

```
BattleSpectator
  max_concurrent: int = 10   — rooms watched simultaneously
  join_interval:  float = 10 — seconds between /join commands
  poll_interval:  float = 30 — seconds between /query roomlist calls

  async def watch(format_id: str) -> AsyncGenerator[SpectatedBattle, None]
```

**Internal loop:**
1. Connect as anonymous guest (no login)
2. Send `/query roomlist {format_id}` immediately
3. `_on_query_response` enqueues unseen room IDs into `_pending` (a `_seen` set prevents
   rejoining rooms from a prior poll cycle)
4. `_join_loop` drains `_pending` one room at a time, gated by `max_concurrent` and
   `join_interval`
5. `_poll_loop` re-queries every `poll_interval` seconds
6. On `|win|` or `|tie|`: `battle.finish(winner)`, `/leave` the room, put the battle on
   `_done` queue
7. `watch()` yields from `_done` indefinitely

`BattleSpectator` runs on `POKE_LOOP` (poke-env's background event loop) so that
`PSClient` callbacks, asyncio queues, and the generator all share the same loop.
The daemon script uses `asyncio.run_coroutine_threadsafe(..., POKE_LOOP).result()`
rather than `asyncio.run()`.

### `src/main/collect_replays.py`

Long-running daemon script. Saves each replay to disk the moment it finishes. Safe to
stop and restart — already-saved files are not overwritten (checked via `path.exists()`).

```
python src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou
python src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou --local
```

Stops cleanly on Ctrl+C.

### `src/poke_env/battle/abstract_battle.py` — one-line addition

Added `":"` and `"t:"` to `MESSAGES_TO_IGNORE`. These are timestamp header lines
(`|:|{ts}` / `|t:|{ts}`) that the Showdown server prepends to the scrollback when
sending the full battle history to a newly-joined spectator. They have no battle-state
meaning. Poke-env was never exposed to these before (participants receive messages in
real time, not as a replay dump), so the parser raised `NotImplementedError` on them.

---

## Log Format

Each `.log` file is the raw Showdown protocol, one line per message:

```
|t:|1779116750
|init|battle
|player|p1|Alice|60|1200
|player|p2|Bob|113|1300
|teamsize|p1|6
|teamsize|p2|6
|gen|3
|tier|[Gen 3] OU
...
|turn|1
|move|p1a: Snorlax|Body Slam|p2a: Gengar
|-damage|p2a: Gengar|61/100
...
|win|Alice
```

This is the same format poke-env's `AbstractBattle.parse_message()` consumes. The
round-trip is confirmed by the e2e test. Two dispatch rules for consumers:
- `|win|username` → call `battle.won_by(username)` directly (not via `parse_message`)
- `|tie` → call `battle.tied()` directly
All other lines go through `battle.parse_message(line.split("|"))`.

---

## Files

| File | Change |
|------|--------|
| `src/poke_env/ps_client/ps_client.py` | `QueryResponseCallback` type + `on_query_response` param + dispatch |
| `src/poke_env/battle/abstract_battle.py` | `":"` and `"t:"` added to `MESSAGES_TO_IGNORE` |
| `src/poke_env/spectator/__init__.py` | New — exports `BattleSpectator`, `SpectatedBattle` |
| `src/poke_env/spectator/spectated_battle.py` | New — pure data object |
| `src/poke_env/spectator/spectator_client.py` | New — async generator, rate control |
| `src/poke_env/spectator/spectated_battle_test.py` | New — 9 unit tests |
| `src/poke_env/spectator/spectator_e2e_test.py` | New — live server test + round-trip parse |
| `src/main/collect_replays.py` | New — daemon script |
