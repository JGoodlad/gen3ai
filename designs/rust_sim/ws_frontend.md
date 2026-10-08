# The websocket front end — the protocol surface, and everything it does NOT implement

**This file OWNS the protocol surface of `src/utils/bridge/ws_frontend.py`.** It is ALWAYS-CURRENT:
state the truth, never narrate a change. The module keeps the mechanism and the hazards; this keeps
the inventory, the deferral list and the evidence.

---

## What it is

An asyncio **websocket server** that speaks just enough of the Showdown *client* protocol for a
third-party poke-env-style client to log in, challenge or accept, and play complete gen3ou battles
— with each battle backed by ONE `sim_bridge` child instead of a Showdown server.

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m utils.bridge.ws_frontend --port 9601           # ws://127.0.0.1:9601/showdown/websocket
python -m utils.bridge.ws_frontend --port 9601 --impl rust --seed-base 914001 \
    --capture-dir /tmp/caps                              # reproducible + replayable
```

**Why it exists.** Every external gen3ou opponent measured so far (Metamon, Foul Play) is a
websocket CLIENT, so playing one has meant `npm run showdown -- 9XXX`. The serverless transports we
already have — `bridge_session.py` for training (deleted in U3; training is the Rust env core), `local_battle_runner.py` for offline series — are
*library* seams: they assign a `BattleStreamClient` onto a poke-env `Player` **in this process**.
That is closed to a third party for the reason
[`metamon_derisk_2026-09-14/README.md`](../research_state/measurements/metamon_derisk_2026-09-14/README.md)
records as verdict (d): their player subclasses **upstream** poke-env, `BattleStreamClient`
subclasses the **vendored fork**, and one process resolves `import poke_env` to exactly one of
them. The integration point therefore cannot be an import; it has to be a socket, with the
opponent in its own process and its own poke-env.

| | transport | who drives | opponent lives |
|---|---|---|---|
| training | the Rust env core (`src/rust_env/`; formerly `bridge_session.py`) | the collector | in-process |
| offline series | `local_battle_runner.py` | `choose_move` | in-process |
| **external opponent** | **`ws_frontend.py`** | **the opponent's own loop** | **its own process** |

### Who uses it

🚨 **`python -m main.anchors` — every EXTERNAL-ANCHOR read — runs this module by default**
(`--server rust`, a 9500–9599 port). With a checkpoint our-side (the default
`--our-transport auto` → `core`, P3 of the poke-env retirement) it runs **IN the anchors process**
(`main.anchors.server.InProcessFrontEnd`; rows say `server_version =
ws_frontend[in-process]@<gen3ai head>+rust:<bridge binary>`) and OUR side is an **in-process
slot** of it (`main.anchors.core_side`, see "The in-process slot" below), so the only websocket
client is the opponent. A `bot:` our-side plays as a websocket CLIENT on the Rust stack
(`--our-transport live`, P6: `main.live.client` with the Rust port of the bot over the `bot_reader`
session) and gets the SUBPROCESS form (`FrontEndServer`, stopped by PID; `server_version =
ws_frontend@…`); the legacy poke-env client is deleted (P6). No Node server is involved in an anchor read unless `--server node` is passed. The promotion's evidence is
[`anchors_rust_frontend_2026-09-20`](../research_state/measurements/anchors_rust_frontend_2026-09-20/README.md)
(100 games each way: Δ +0.010 [−0.124, +0.144], NOT DETECTED; 40 MB mean server-tree RSS against
Node's 3,227 MB; 126 s against 203 s), on top of the 200-battle side-by-side in
[`foulplay_axes_and_frontend_validation_2026-09-16`](../research_state/measurements/foulplay_axes_and_frontend_validation_2026-09-16/README.md).
⚠️ **`--server node` stays one flag away** — the deferral list below is exactly what it is for,
and a differential needs a reference transport that is not ours.

---

## The protocol surface — everything it implements

### Server → client

| line | shape | who reads it |
|---|---|---|
| `\|updateuser\| Guest N\|0\|1\|{}` | sent on CONNECT, **before** `\|challstr\|` | poke-env's fork documents this order at length — a guest greeting before `/trn` must not read as a login |
| `\|challstr\|4\|<64 hex>` | sent on connect | Foul Play blocks on it (`get_id_and_challstr`); poke-env triggers `log_in` on it |
| `\|updateuser\| <Name>\|1\|<avatar>\|<settings json>` | the `/trn` answer | **the leading space is load-bearing** — upstream poke-env compares against `" " + username` |
| `\|nametaken\|\|Your name must be 18 characters or shorter.` | userid > 18 chars | the refusal; see hazard H1 |
| `\|nametaken\|<name>\|Someone is already using the name "<n>".` | duplicate login | |
| `\|popup\|…` | team rejection, absent challenger, `/search`, a crashed battle | both clients log it; poke-env's fork warns |
| `\|pm\| <from>\| <to>\|/challenge <fmt>\|<fmt>\|\|\|` | a challenge, **exactly 9 fields** | poke-env `_handle_challenge_request` (field 5 == format); Foul Play `accept_challenge` (**requires len == 9**) |
| `\|pm\| <from>\| <to>\|/challenge` | a cancelled/consumed challenge | |
| `\|queryresponse\|userdetails\|<json>` | the `/cmd userdetails` answer | Foul Play's `avatar()` blocks on it |
| `>battle-<fmt>-<n>` + `\|init\|battle` + `\|title\|<A> vs. <B>` | the room, fabricated | poke-env creates the battle on a frame whose **second line** is `\|init\|`; Foul Play takes the opponent name out of **field 4** |
| the per-side protocol chunks | relayed **verbatim** from the bridge child, inside `>battle-…` | everything |
| `\|request\|{…,"rqid":N}` | the sim's own JSON **plus** an appended `rqid` | **the decision trigger** |
| `\|deinit` | the `/leave` answer | Foul Play's `leave_battle` loops until it sees it |

`|player|`, `|teamsize|`, `|start`, `|turn|`, `|win|`, `|tie`, `|error|` and every battle line come
from the sim as ordinary chunk content — the front end adds nothing to them and removes nothing.

### Client → server

| command | effect |
|---|---|
| `/trn <name>,0,<assertion>` | a NO-OP login: any assertion is accepted; only the length and duplicate-name refusals are reproduced |
| `/utm <packed team>` / `/utm null` | registers the team, validated through `validate_team.js` (cached per team) |
| `/challenge <user>, <format>` | records the challenge (challenger's team snapshotted) + PMs both sides |
| `/accept <user>` | starts the battle — **the challenger is p1**, as on a real server |
| `/reject` / `/cancelchallenge` | drops it, PMs the empty `/challenge` update |
| `<tag>\|/choose <choice>[\|<rqid>]` | forwarded as `CHOOSE <slot> <choice>` |
| `<tag>\|/team <order>[\|<rqid>]` | forwarded as a choice (team preview — gen3 has none) |
| `<tag>\|/forfeit` | forwarded as `FORCELOSE <slot>` |
| `/leave [<tag>]` | answers `\|deinit` |
| `/search <format>` | **refused** with a popup naming `/challenge` as the alternative |
| `/timer`, `/undo`, `/join`, `/savereplay`, `/avatar`, `/cancelsearch`, chat | accepted, no effect |

### The in-process slot — a connection with no socket (P3, `core_obs`)

A `_Conn` with `core_obs = True` is a user that lives in the front end's own process
(`main.anchors.core_side`'s slot). Three hooks, and nothing a websocket client sees changes:

| hook | what it does |
|---|---|
| `client_line(conn, line)` / `drop(conn)` / `next_guest()` | the slot speaks the SAME client protocol (`/trn`, `/utm`, `/challenge`, `/accept`, `<tag>\|/choose <token>\|<rqid>`, `<tag>\|/forfeit`) by function call, so every server-side rule above applies to it |
| `START` carries `core_obs: {"sides": [<the slot's side>]}` | **only** when a connection in the battle has `core_obs`; otherwise the child's stdout is byte-identical to a plain battle's (`sim_bridge`'s `gen3_bridge_core_obs_v1`) |
| `__OBS__ <side> <json>` lines go to `conn.on_obs(tag, side, payload)` | never relayed to any websocket; the child writes the frame BEFORE the chunk carrying that side's `\|request\|`, so the slot holds the row when the request (with its spliced `rqid`) arrives |

The slot decides on the frame's row + mask (the training reader — the same `BattleVersion` parse
chain and encoder training runs) and answers with the frame's own choice TOKEN for the chosen
index. Pinned by `src/utils/bridge/ws_frontend_test.py` (the START gating, the routing) and
`src/main/anchors/core_side_test.py` (the slot's protocol rules).

### The two things the front end ADDS to the sim's stream

1. **The room framing** — `>battle-…`, `|init|battle`, `|title|`. The same fabrication
   `local_battle_runner._frame` does for the in-process path; the sim never emits it.
2. **The `rqid`** — see below. Nothing else is added, removed or reordered.

---

## 🚨 The `rqid`, and why it is the whole integration

A `|request|` is what makes a client move — `fp/battle/protocol.py:2279` returns `action_required`
only on one, and poke-env's `_handle_battle_request` fires on one. **The sim does not emit an
`rqid`.** Measured, not assumed: `Side.emitRequest` sends the bare object, and
`server/room-battle.ts:796-801` injects the id — `this.rqid++`, `request.rqid = this.rqid`,
re-stringify — from **one counter shared by both slots**, so the two sides' ids interleave into a
single `1..N` sequence. Every choice Foul Play sends echoes it back
(`"<tag>|/choose move X|<rqid>"`), and the server refuses a stale one.

The front end therefore does exactly that, at exactly that point, plus the two `sideupdate` state
transitions the same handler performs:

| event | effect |
|---|---|
| a `\|request\|` reaches a slot | shared counter `+= 1`; splice `,"rqid":N` in; slot is `isWait=false` (or `'cantUndo'` for a `wait` request) |
| a `\|error\|[Invalid choice]` reaches a slot | slot returns to `isWait=false` — poke-env re-chooses off the SAME request, and a slot left closed would wedge the battle |
| a `\|error\|[Unavailable choice]` | **nothing** — Showdown follows it with a fresh `\|request\|`, which is what reopens the slot |
| a `/choose` with a stale `rqid` | refused: `[Invalid choice] Sorry, too late to make a different move…` |
| a `/choose` with an empty `rqid` | the staleness check is skipped — poke-env sends none, and the server skips it too |

The injection is a **string splice**, not a JSON round trip: everything before the closing brace
stays the sim's own bytes. That is what makes the byte-differential gate legible — its only
request-related normalization is removing one appended field.

---

## The byte-differential gate

`src/utils/bridge/ws_frontend_replay.py` + `ws_frontend_byte_identity_integration_test.py`.

Play a **seeded** battle through the front end; replay the exact command stream it produced into
the Node `local_sim_bridge.js`; compare the per-side protocol TEXT byte for byte. A replay rather
than a live A/B because the decisions are already DISCOVERED (the front end recorded what its
clients actually chose), which removes the decision-segmentation artifact `gen_sim_bridge_diff.js`
documents.

**Exactly two normalizations, and nothing else:**

| normalized | why it is legitimate |
|---|---|
| `\|t:\|<anything>` → `\|t:\|` | wall clock — the ONE protocol exception poke-env ignores. The whole payload, not an epoch: Node emits `\|t:\|1789430503`, the Rust port emits the literal `\|t:\|<NORMALIZED>` |
| `,"rqid":N` dropped from a `\|request\|` | the front end's (and the real server's) deliberate injection; a Node bridge that never saw a server cannot have it |

Separately asserted, so stripping a field can never hide the field being wrong: every relayed
`|request|` carries an rqid, and the two slots' ids merge into exactly `1..N`.

An **unseeded** capture RAISES rather than reporting a divergence — a replay of an unseeded battle
is a different battle, and an inconclusive comparison must never read as either verdict.

**Measured 2026-09-14** (front end on `--impl rust`, reference `local_sim_bridge.js`):

| series | battles | protocol lines | `\|request\|`s | verdict |
|---|---:|---:|---:|---|
| Metamon `SmallRL` greedy vs `ai_v12_02_winprob_critic`@75M | 20 / 20 | 22,728 | 1,919 | **byte-identical** |
| Foul Play `--search-time-ms 300` vs the same checkpoint | 20 / 20 | 21,306 | 1,678 | **byte-identical** |
| two poke-env `RandomPlayer`s, the SHIPPED code re-verified through the CLI | 5 / 5 | 11,960 | 948 | **byte-identical** |

---

## Throughput — and why it is not the reason to use this

`ws_frontend_throughput_benchmark.py` (deleted in P6 slice 6c with the poke-env `RandomPlayer`s it drove; the figures below are its dated measurement). Two poke-env `RandomPlayer`s played the
identical workload over a real websocket; the only difference is what is listening.

**Measured 2026-09-14, `--battles 24 --impl rust`, on a box carrying a live training arm**
(load average 35.27/28.24/25.89 on 16 cpus — contention factor ~2.2, WARNED and **not** scaled):

| concurrency | ws_frontend (battles/s) | `npm run showdown` (battles/s) | ratio |
|---:|---:|---:|---:|
| 1 | **5.78** | 1.87 | **3.09x** |
| 4 | 6.21 | 7.54 | 0.82x |
| 8 | 7.07 | 10.78 | 0.66x |

A 8-battle-per-cell run the same hour read 10.75 / 9.83 / 11.10 against 2.06 / 5.39 / 19.33 — the
same SHAPE (front end far ahead at 1, server ahead by 8), with the absolute numbers moving with the
load, which is exactly what the contention warning says to expect.

🚨 **Read the concurrency-1 row and treat the rest as bounds.** The harness is SINGLE-PROCESS —
both clients and (in the front-end arm) the server share one Python process and one event loop — so
above one concurrent battle the front-end arm is capped by that process's CPU share rather than by
the transport, while the server arm offloads its sim work to a separate process. Real use puts the
opponent in its own process, which is the entire point of the front end. The crossover is therefore
a property of this harness, not a verdict.

The honest headline is the foul-play de-risk's, unchanged: **a shim buys determinism and the
removal of a server process, not throughput.** At a realistic external-opponent budget the OPPONENT
dominates by orders of magnitude — Foul Play at `--search-time-ms 1000` costs ~88 s/game against
~1.2 s of server overhead. What this table rules out is the failure that would matter: a front end
that serialized what a server overlaps. It does not — 8-way concurrency is faster than 1-way in
both benchmark runs.

---

## What it does NOT implement — the deferral list

Each row is a deliberate non-goal, not an oversight. **This front end is not a Showdown server and
must never be presented as one.**

| deferred | consequence | why |
|---|---|---|
| **authentication** | any assertion is accepted; no password is ever checked | a local harness. The clients still fetch their assertion from the real login server (Foul Play always, poke-env when a password is set) — that is their code, and the front end neither needs nor inspects the result |
| **a battle timer** | a client that never chooses hangs forever | the trainer's forfeit is CLIENT-side (`StallConfig.threshold`, 250 turns). A second, server-side clock would be an unrecorded second way to lose |
| **`/search` (laddering)** | refused with a popup | there is no matchmaker and no rating |
| **ratings / `\|updatesearch\|` / ELO** | absent | no ladder |
| **chat, rooms, `/join`, user lists** | swallowed | nothing reads them |
| **replays (`/savereplay`)** | swallowed | the bridge's `__RECON__` is the forensic record, and it is referee-view — see below |
| **`\|updatechallenges\|`** | only the `\|pm\|` form is sent | both clients accept the PM, and sending BOTH would double-queue the challenge in poke-env's `_challenge_queue` and make the acceptor accept twice |
| **`/undo`** | swallowed | no client in scope sends it |
| **team preview** | `/team` is forwarded, untested | gen3 has none |
| **spectators / the omniscient stream** | never served | the one-sided wall: only a slot's own chunks reach that slot |
| **`__RECON__`** | dropped in the pump | full-information (both teams + the seed). It must never reach a player, and the front end has no forensic sink |
| **multiple formats at once** | `--format` is global | every consumer is gen3ou |
| **reconnection / `/rejoin`** | a dropped client FORFEITS its live battles | a half-finished battle with an absent player is worse than a recorded loss |

---

## Hazards

**H1 🚨 A username longer than 18 characters is a HANG, not an error.** `server/users.ts:745`
refuses a >18-char userid with `|nametaken||Your name must be 18 characters or shorter.` and does
NOT log the user in; Foul Play ignores `|nametaken|` entirely, logs "Successfully logged in", and
waits forever (foul-play de-risk hazard 1 — and there the refusal came from `action.php`, which
even a `--no-security` local server consults). The front end reproduces the server's message AND
logs an ERROR naming the cause, so the hang is diagnosable from the server log. **Keep usernames
short.**

**H2 🚨 A blasted command stream makes the Node bridge emit NOTHING.** Node's stdin `data` handler
runs `handleLine` synchronously for every line in the chunk it receives, while the per-side pumps
that produce output are `for await` loops that have not run yet — so a replay that writes the whole
stream at once reaches the recorded `END`, calls `exitWhenDrained(0)`, and exits having written
zero protocol chunks, with no error and no stderr. Measured here: 0 chunks for a 106-command
battle. `replay_through` therefore paces each write behind a quiescence settle and drops a recorded
`END`.

**H3 ⚠️ A capture without `--seed-base` is not replayable.** A seedless `START` makes the child mint
its own seed, so the gate would compare two different battles. The server warns at startup when
`--capture-dir` is given without `--seed-base`, and the gate raises rather than reporting a
divergence.

**H4 ⚠️ The challenger is p1.** As on a real server — and the matched-regime harness balances roles
on exactly that fact, so reversing it would silently flip a measurement's sides.

**H5 ⚠️ A team file handed to a third-party poke-env must be nickname-free.** Unchanged from the
metamon de-risk (H1 there): upstream poke-env packs `Airmure (Skarmory)` with an EMPTY species
field, the sim rejects the team, and the match STALLS rather than erroring. The front end's `/utm`
validation turns that into a loud popup plus an ERROR log on OUR side, but it cannot fix the
opponent's packer — export nickname-free team files.

**H6 ⚠️ A bare TCP connect is not a free readiness probe.** `websockets` answers a connect that
closes without an HTTP request with `ERROR opening handshake failed` and a three-deep traceback —
one per probe, in the log whose emptiness is the protocol criterion every validation of this front
end has read ("0 ERROR, 0 WARNING over 200 battles"). The `[ws_frontend] READY <uri>` line is
printed after the listener is bound and is therefore the probe to use; `main.anchors` waits for it
and never dials the port. Measured 2026-09-20.

**H7 ⚠️ The clients' postures on an unknown protocol keyword are OPPOSITE.** Our
`battle_event.classify` raises by design; Foul Play silently ignores. A front end that relays the
pinned submodule's bytes exercises neither, exactly as a pinned local server does not —
`ladder_drift_scan` (on the Rust reader since P6) remains the instrument for live drift.
