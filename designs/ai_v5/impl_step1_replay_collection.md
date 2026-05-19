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
the same per-user queue). The implementation enforces a 1-second join interval and
caps at 20 concurrent rooms by default; slots are reclaimed immediately when a battle
finishes so the next queued room joins within ~0.1 s.

---

## What Was Built

### `src/poke_env/ps_client/ps_client.py` — proxy support + query callback

- Added `QueryResponseCallback` type and `on_query_response` optional parameter,
  matching the existing `on_battle_message` callback pattern. `_handle_message` now
  dispatches `|queryresponse|` to this callback instead of silently dropping it.
- Added `proxy_url: Optional[str]` parameter. When set, WebSocket connections are
  established through a SOCKS5 proxy via `python-socks[asyncio]`, and HTTP auth
  requests use `requests` proxy support. Accepts `socks5h://` (normalised internally
  to `socks5://` — python-socks resolves hostnames remotely automatically).
- `_create_logger` no longer adds a `StreamHandler`; it just sets the level and
  propagates to the root logger, so callers control output formatting.
- Raw `<<<`/`>>>` wire messages are logged at `DEBUG` (not `INFO`) to keep INFO-level
  output clean for application events only.
- `ConnectionClosedError` (TCP reset — server drops connection without WebSocket
  close frame) is now caught at `WARNING` level alongside `ConnectionClosedOK`, so
  unclean disconnects produce a one-line warning instead of a full traceback at ERROR.

### `src/poke_env/spectator/spectated_battle.py`

`SpectatedBattle` is a pure data object — no file I/O, no dependency on the battle
state machine. It accumulates raw Showdown protocol lines and exposes them as
`log_text`. The caller decides what to do with a completed battle.

```
SpectatedBattle
  battle_tag: str
  finished: bool
  winner: str | None        — None means tie
  turn: int                 — most recent |turn| number seen (0 before first turn line)
  players: dict[str, str]   — {"p1": username, "p2": username} from |player| lines
  joined_at: float          — time.time() when the battle object was created
  log_text: str             — full Showdown-format log, one protocol line per line
```

`finish()` is idempotent. `add_lines()` is a no-op after `finish()`. Neither method
touches the filesystem.

`turn`, `players`, and `joined_at` are used by the dashboard to show age, current turn,
and player names for each active room.

### `src/poke_env/spectator/spectator_client.py`

`BattleSpectator` is an async generator that yields `SpectatedBattle` objects
indefinitely as battles complete. All state-machine logic lives here; the caller only
sees a stream of finished battles.

```
BattleSpectator
  max_concurrent: int = 20   — rooms watched simultaneously (default raised from 10)
  join_interval:  float = 1  — seconds between /join commands (reduced from 10)
  poll_interval:  float = 30 — seconds between /query roomlist calls
  proxy_url: str | None      — SOCKS5 proxy URL, e.g. socks5h://127.0.0.1:1080

  async def watch(format_id: str) -> AsyncGenerator[SpectatedBattle, None]
```

**Internal loop:**
1. Connect as anonymous guest (no login)
2. Send `/query roomlist {format_id}` immediately
3. `_on_query_response` enqueues unseen room IDs into `_pending` (a `_seen` set prevents
   rejoining rooms from a prior poll cycle)
4. `_join_loop` drains `_pending` one room at a time, gated by `max_concurrent` and
   `join_interval`; slot check polls every 0.1 s so new rooms join immediately when
   a battle finishes
5. `_poll_loop` re-queries every `poll_interval` seconds
6. On `|win|` or `|tie|`: `battle.finish(winner)`, `/leave` the room, put the battle on
   `_done` queue
7. `watch()` yields from `_done` indefinitely; auto-reconnects on connection drop

**Ghost-battle prevention:** `_finished_tags` tracks rooms that have been finished and
left within a session. Late server messages arriving after `/leave` are silently ignored,
preventing a finished room from being re-created as a ghost `_active` entry. Without
this, a battle could appear simultaneously in Active Battles (as an empty "loading…"
entry) and in Recent Completions.

**Auto-reconnect:** `watch()` wraps `_watch_once()` in an outer retry loop.
`_watch_once()` checks `client._listening_coroutine.done()` every 5 seconds (via
`asyncio.wait_for` timeout on `_done.get()`). When `listen()` exits for any reason
(TCP reset, server restart, etc.), `_watch_once` raises `ConnectionError` and `watch()`
catches it, logs a warning, waits `reconnect_delay` (10 s), and reconnects.
`_seen` is preserved across reconnects so old rooms are not re-joined.
`_total_joined` and the `_seen` set persist across reconnects; `_active`,
`_pending`, `_done`, and `_finished_tags` are reset each session.

`BattleSpectator` runs on `POKE_LOOP` (poke-env's background event loop) so that
`PSClient` callbacks, asyncio queues, and the generator all share the same loop.
The daemon script uses `asyncio.run_coroutine_threadsafe(..., POKE_LOOP)` rather than
`asyncio.run()`.

### `src/main/collect_replays.py`

Long-running daemon script with a Rich terminal dashboard. Saves each replay to disk
the moment it finishes. Safe to stop and restart — already-saved files are not
overwritten (checked via `path.exists()`).

```bash
# Real Showdown via SOCKS5 proxy (recommended — hides your home IP)
export PYTHONPATH=$PYTHONPATH:src
python3 src/main/collect_replays.py \
  --format gen3ou \
  --save-dir replays/gen3ou \
  --max-concurrent 20 \
  --proxy socks5h://127.0.0.1:1080

# Local server (development / no proxy needed)
python3 src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou --local
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `gen3ou` | Showdown format ID |
| `--save-dir` | `replays` | Output directory for `.log` files |
| `--local` | off | Connect to `localhost:8000` instead of real Showdown |
| `--max-concurrent` | `20` | Max rooms watched simultaneously |
| `--proxy` | none | SOCKS5 proxy URL, e.g. `socks5h://127.0.0.1:1080` |
| `--verbose` | off | Show DEBUG-level logs in the UI |

**Dashboard:** Uses Rich `Live(screen=True)` — takes over the alternate terminal buffer
like `htop`, so resizing is clean and the normal scroll history is untouched. Three
sections: Active Battles (with room #, age, turn, player names), Recent Completions
(last 5), and Logs (last 5 INFO+ lines captured from the Python logging system). The
stats row shows connection mode: **PROXIED** (green) or **DIRECT** (yellow). Active
battle rows shrink dynamically to fit the terminal height; if space is very tight,
recent completions drops to 3 rows. Exits cleanly on Ctrl+C, restoring the terminal.

Stops cleanly on Ctrl+C.

### `src/poke_env/battle/abstract_battle.py` — one-line addition

Added `":"` and `"t:"` to `MESSAGES_TO_IGNORE`. These are timestamp header lines
(`|:|{ts}` / `|t:|{ts}`) that the Showdown server prepends to the scrollback when
sending the full battle history to a newly-joined spectator. They have no battle-state
meaning. Poke-env was never exposed to these before (participants receive messages in
real time, not as a replay dump), so the parser raised `NotImplementedError` on them.

---

## Proxy Setup

All traffic to real Showdown is routed through a GCP e2-micro VM (`proxy.g5d.io`,
`136.109.158.194`) via an SSH SOCKS5 tunnel managed by a systemd user service. See
`scripts/PROXY_TUNNEL.md` for full setup and usage instructions.

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
| `src/poke_env/ps_client/ps_client.py` | Proxy support, `QueryResponseCallback`, clean logger |
| `src/poke_env/battle/abstract_battle.py` | `":"` and `"t:"` added to `MESSAGES_TO_IGNORE` |
| `src/poke_env/spectator/__init__.py` | New — exports `BattleSpectator`, `SpectatedBattle` |
| `src/poke_env/spectator/spectated_battle.py` | New — pure data object |
| `src/poke_env/spectator/spectator_client.py` | New — async generator, rate control, proxy |
| `src/poke_env/spectator/spectated_battle_test.py` | New — 9 unit tests |
| `src/poke_env/spectator/spectator_e2e_test.py` | New — live server test + round-trip parse |
| `src/main/collect_replays.py` | New — daemon script with Rich dashboard |
| `scripts/proxy_tunnel.sh` | New — SSH tunnel keepalive script |
| `scripts/get_proxy_ip.sh` | New — prints GCP static IP and tunnel command |
| `scripts/PROXY_TUNNEL.md` | New — proxy setup and usage documentation |
