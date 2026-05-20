# Implementation: Battle Spectator & Replay Collection

This doc covers the passive battle observer added to harvest human expert game logs
from live Pokémon Showdown. The spectator connects as an anonymous guest, discovers
active gen3ou rooms, joins them, and saves complete protocol logs to disk for offline
analysis.

Primary themes: anonymous WebSocket spectating without authentication, full-history
room replay on join, concurrency control to stay within server rate limits, and a Rich
live dashboard for monitoring collection progress.

---

## Motivation

Behavioral cloning pre-training requires a corpus of expert-play trajectories. The
Showdown ladder generates thousands of gen3ou games daily. Rather than downloading
static replay files (which omit timing and some intermediate state), the spectator
captures the live Showdown protocol stream — the same format that poke-env's
`AbstractBattle.parse_message()` consumes. Saved logs can be round-tripped through
the parser to reconstruct full game states, including all damage events, switch-ins,
and weather/status transitions.

---

## Architecture

Three independent layers:

```
SpectatedBattle (pure data)
    ↑ add_lines(), finish()
BattleSpectator (async protocol client)
    ↑ yields completed SpectatedBattle objects
collect_replays.py (CLI daemon + Rich dashboard)
    saves to disk, refreshes TUI at 2 Hz
```

### `SpectatedBattle` (`src/poke_env/spectator/spectated_battle.py`)

Accumulates raw Showdown protocol lines with no file I/O or parsing side-effects.

| Field | Type | Content |
|---|---|---|
| `_lines` | `list[str]` | Raw protocol messages in order |
| `_turn` | `int` | Current turn (updated from `\|turn\|N` messages) |
| `_players` | `dict[str, str]` | `{"p1": username, "p2": username}` |
| `_joined_at` | `float` | Unix timestamp of room join |
| `_finished` | `bool` | True after `\|win\|` or `\|tie\|` |
| `_winner` | `Optional[str]` | Winner's username, or None for ties |

`add_lines(lines)` and `finish()` are both idempotent — safe to call after the battle
is already marked finished (late server messages are dropped harmlessly).

`log_text` property returns the complete log as a single string (one protocol line per
newline), ready to write to disk.

### `BattleSpectator` (`src/poke_env/spectator/spectator_client.py`)

An async generator that yields completed `SpectatedBattle` objects indefinitely.
Runs on poke-env's background `POKE_LOOP` thread so PSClient callbacks, asyncio queues,
and the generator share the same event loop.

**Connection flow:**

1. Connect to Showdown as anonymous guest (no login). The server assigns a name like
   "Guest 12345". The client detects this via `nametaken` / guest-name assignment
   messages and marks the connection ready without authentication.
2. Query `/query roomlist gen3ou` to get the list of currently active battle rooms.
3. For each room not in `_seen`: add to a pending queue.
4. Slot management: maintain at most `max_concurrent` (default 20) simultaneous joins.
   When a slot opens, immediately pop the next room from the queue and send `/join`.
5. On join: the Showdown server replays the entire battle history from move 1 (via its
   `onConnect` handler). No battle is partially captured, regardless of when we joined.
6. Process incoming messages: route by `battle_tag` to the corresponding `SpectatedBattle`.
7. On `|win|` or `|tie|`: mark the battle finished, send `/leave`, put the object on
   the `_done` asyncio queue.
8. The generator awaits `_done.get()` and yields each completed battle.

**Rate control:**

| Parameter | Default | Meaning |
|---|---|---|
| `max_concurrent` | 20 | Simultaneous rooms watched |
| `join_interval` | 1.0 s | Minimum time between `/join` commands |
| `poll_interval` | 30 s | Time between roomlist re-queries |

`join_interval` is well above the 600 ms server message throttle. The slot-check loop
polls at 100 ms — when a slot opens, the next join fires within 100 ms.

**Reconnection (`59258e1`):**

If the WebSocket drops, the background listening coroutine exits. A watchdog detects
this and reconnects after a 10 s delay. `_seen` (already-visited rooms) is preserved
across reconnect to avoid duplicate captures; `_active` (in-flight battles) is cleared
since their mid-game state is lost.

**Ghost battle prevention (`3a57f72`):**

`_finished_tags` tracks rooms that have been left. Late server messages for departed
rooms are silently dropped rather than recreating zombie `SpectatedBattle` entries.

### poke-env protocol extensions

**`PSClient` additions (`77ef01c`):**

- `on_query_response` callback parameter added. `_handle_message` dispatches
  `|queryresponse|` to this callback (previously silently ignored) so the spectator
  can receive roomlist JSON.
- `nametaken` handler: if the requested username is taken, the client continues with
  the server-assigned guest name rather than failing.
- Guest login: detects "Guest XXXXX" name assignment and marks the connection ready
  without waiting for authentication.

**`AbstractBattle.MESSAGES_TO_IGNORE` additions (`77ef01c`):**

Added `":"` and `"t:"` to the ignore set. These are timestamp header lines
(`|:|{ts}` and `|t:|{ts}`) that Showdown prepends to battle history when replaying
to a newly-joined spectator. The parser was raising `NotImplementedError` on these
before this fix.

---

## Collection Daemon

**`src/main/collect_replays.py`**

```bash
# Live Showdown server
python src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou

# Local server
python src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou --local

# Via SOCKS5 proxy (e.g. SSH tunnel to a GCP instance)
python src/main/collect_replays.py --format gen3ou --save-dir replays/gen3ou \
  --proxy socks5h://127.0.0.1:1080 --max-concurrent 20 --verbose
```

**Proxy support (`0933b46`):**

`PSClient` accepts a `proxy_url` parameter using `python-socks[asyncio]`.
`socks5h://` is normalised to `socks5://` for library compatibility. Useful for
running the collector from a machine that needs to reach Showdown's servers via an
SSH tunnel.

**File I/O:**

Each completed battle is written immediately to `{save_dir}/{battle_tag}.log`. Existence
check at write time means the daemon is safe to restart — already-saved logs are skipped
and counted in the "skipped" stat rather than overwritten.

**Threading model:**

`asyncio.run_coroutine_threadsafe(_run(...), POKE_LOOP).result()` schedules the async
spectator on poke-env's background event loop from the main thread. This lets the Rich
dashboard refresh at 2 Hz in the main thread while the spectator handles all I/O
concurrently.

Root logger handlers are cleaned of the default `StreamHandler` to avoid terminal
pollution from the live Rich layout; captured log records are injected into the dashboard
instead.

---

## Rich Live Dashboard

The dashboard is full-screen (alternate buffer, `screen=True`) and refreshes at 2 Hz.

**Stats row (always visible):**

| Field | Content |
|---|---|
| Collected | Total battles saved to disk |
| Watching | Active rooms / max_concurrent |
| Queued | Rooms discovered but not yet joined |
| Seen | Distinct rooms observed this session |
| Elapsed | Wall-clock time |
| Connection | `PROXIED` or `DIRECT` |

**Active Battles table:**

One row per in-progress battle: room number, age in seconds, current turn, player names.
Capped at available terminal height; overflow count shown if truncated.

**Recent Completions:**

Last 5 finished battles with winner name and final turn count.

**Logs section:**

INFO+ records captured from the spectator, colourised by WARNING/ERROR.

---

## Saved Log Format

Each `.log` file contains one raw Showdown protocol message per line, identical to
what `AbstractBattle.parse_message()` expects. Example:

```
|player|p1|PikalaxALT|pikachu|1800
|player|p2|Finchinator|pikachu|1750
|teamsize|p1|6
|teamsize|p2|6
|gametype|singles
|gen|3
|tier|[Gen 3] OU
|turn|1
|move|p1a: Tyranitar|Rock Slide|p2a: Clefable
|-supereffective|p2a: Clefable
|-damage|p2a: Clefable|55/100
...
|win|PikalaxALT
```

The log can be fed back through a `Battle` object's `parse_message()` to reconstruct
the full game state, including revealed moves, HP fractions, status, weather, and all
other fields tracked by poke-env.

---

## Files Changed

| File | Change |
|---|---|
| `src/poke_env/spectator/spectated_battle.py` | New — pure data accumulator |
| `src/poke_env/spectator/spectator_client.py` | New — async generator protocol client |
| `src/poke_env/spectator/__init__.py` | New — package init |
| `src/poke_env/ps_client/ps_client.py` | `on_query_response` callback; guest login; `nametaken` handler; proxy support |
| `src/poke_env/battle/abstract_battle.py` | Add `":"` and `"t:"` to `MESSAGES_TO_IGNORE` |
| `src/main/collect_replays.py` | New — daemon script with Rich dashboard |

## Commits

| Hash | Summary |
|---|---|
| `77ef01c` | feat(spectator): add battle spectator and replay collection daemon |
| `e09f591` | feat(spectator): add live Rich dashboard to replay collector |
| `59258e1` | fix(spectator): auto-reconnect on connection drop |
| `3a57f72` | fix(spectator): ghost battle entries; full-screen live UI |
| `0933b46` | feat(spectator): proxy support, rich dashboard, improved rate control |
