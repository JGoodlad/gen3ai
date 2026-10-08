# The live reader — a FOREIGN stream through the training chain (`live_reader`, `side_reader`)

**This file OWNS the reader session of live websocket play** (poke-env retirement P4, 2026-10-07): the Rust
`pokesim::side_reader::SideReader`, the `live_reader` binary around it, and the framing contract the Python client
(`src/main/live/`) keeps. It is ALWAYS-CURRENT: state the truth, never narrate a change.

---

## What it is

ONE side of ONE battle, read from a stream the reader did not produce — a real Showdown server's messages, our
`--server rust` front end's (`ws_frontend.md`), a public replay's — into exactly the frame training reads:

```
__OBS__ p1 {"frame":{"dtype":"<f4","shape":[OBS_DIM],"b64":…},"mask":[11 ints],"tokens":{…},"turn":…,"line":…,"rqid":…,"n":…}
```

**There is ONE implementation.** `SideReader` (`src/rust_sim/src/side_reader.rs`) is the per-side state
`sim_bridge`'s core observation mode used to keep inline (`encoder.md` §5a): the chain
`BattleVersion::parse_root_unrecorded(side, name, packed team, ClockConfig::default())`, advanced write by write by
`parse_advance_lean`, the choice noted BEFORE the next write (`note_choice`), one frame per decision written by
`encoder::wire::obs_json_into`, the alignment rules (one decision per write; the decision's `|request|` the write's
LAST line) and the sticky failure. `sim_bridge` now holds one `SideReader` per requested side, and so does
`live_reader` — so the live reader IS the training reader by construction, and gate (a) below checks it anyway.

**Its four front ends** (P6 of the poke-env retirement merged the second one-side reader into it, F-P5-8):
`sim_bridge`'s core observation mode (training's rows), `live_reader` (a live websocket client's session),
`rust_env`'s `bot_reader` (a roster bot on the same reading, below) and `core_events --obs-stream` (the PROBER's
batch reader: many streams per process, each batch ending at a `|request|` line; it calls
`SideReader::advance_fold` — the same fold, rules and sticky failure, no frame — and encodes only at the decisions
it was asked for, `core_events.md` §9). Before P6 `--obs-stream` folded its own `parse_advance_lean` loop without
the alignment check; old vs new on 1,200 streams of 240 banked core-trace battles (43,432 decisions, 18,839 rows)
were byte-identical, and `sim_bridge_core_obs_test.rs::obs_stream_reads_the_bridge_rows_from_each_sides_text` holds
its rows to the bridge's (`designs/research_state/measurements/pokeenv_p6_reader_merge_2026-10-08/`). The env core
(`rust_env`'s `pool::Env` / `search::game::Game`) still folds its OWN copy of the rule, held equal by gate ① and
`tests/search_game_test.rs` — the one fold P6 did not merge (training's hot path).

| layer | file | job |
|---|---|---|
| the chain | `src/rust_sim/src/side_reader.rs` | text in, frame out, alignment, sticky failure |
| the session | `src/rust_sim/src/bin/live_reader.rs` | a stdio process around one `SideReader` |
| the pipe | `src/main/live/reader.py` (`LiveReader`, `Frame`, `ROOM_SKIP`) | spawn, `OPEN` / `FEED` / `CHOOSE`, every failure a `ReaderRefusal` |
| the client | `src/main/live/client.py` (`LiveBattle`, `LiveClient`) | the websocket, the framing, the rules (T28) |
| the policy | `src/main/live/policy.py` | `ModelPolicy` (CPU, greedy or seeded T) / `RandomPolicy` |

## The protocol (stdin / stdout, one reply per command)

| command | reply |
|---|---|
| `OPEN {"side":"p1"\|"p2","name":…,"team":<packed>\|null}` | `__OK__` — a new battle (the previous reader is dropped) |
| `FEED [<line>, …]` — one WRITE: the lines one server message carried, battle protocol only | `__OBS__ pN {…}` when the write ended at a decision, then `__FED__ {"folded","decided","turn"}` |
| `CHOOSE <token>` — the choice sent for the open decision | `__OK__` |
| `PROBE` — the drift scan's encoder check (never the live path) | `__OK__`, or `__ERR__ {"kind":"probe",…}` (the reader stays alive) |
| `CLOSE` / `END` | `__OK__` / exit |

Any other failure is ONE `__ERR__ {"kind","message"}` in place of the reply, and the reader is FAILED for the rest of
the battle. A panic is caught and reported the same way. The Python side turns every `__ERR__` (and a dead process)
into `ReaderRefusal`, and the client turns that into a T28 halt.

## The framing contract (what the caller owes the reader)

1. **Battle protocol only.** Dropped before the reader, by a DECLARED list (`main.live.reader.ROOM_SKIP`): the room
   framing (`init`, `title`, `deinit`, `noinit`, `expire`), presence (`j`/`J`/`join`, `l`/`L`/`leave`,
   `n`/`N`/`name`), chat (`c`, `c:`, `chat`), server HTML and notices (`raw`, `html`, `uhtml`, `uhtmlchange`,
   `notify`, `tempnotify`, `tempnotifyoff`, `hidelines`, `unlink`, `badge`) and the timer's `inactive` /
   `inactiveoff`; plus the empty string `""` a message's trailing newline leaves (measured on Showdown master
   `51ad80fa`). The simulator emits none of them on the gen-3 path, so the stream training reads holds none.
   **Every other line is fed — and an unknown keyword is the READER's to refuse**, never a skip.
2. **One message = one write.** A real server forwards the sim's `sendUpdates` in order — the battle chunk, then each
   side's `|request|` as its own `sideupdate` message (measured on master: the request arrives alone, after the message
   ending `|turn|N`). Feeding each message as a write therefore ends a write at our request exactly when the bridge's
   write would; a message whose request is NOT its last line is an `[ALIGN]` refusal (T28), never a quietly late row.
   The chunk itself may be split over several messages (master sent `|t:|` + `|gametype|`, then `|player|p1`, then
   the rest): harmless, since only a request opens a decision.
3. **Lines before our side is known are held** and fed with the first write after the `|player|pN|<our name>|` line
   names our seat.
4. **After `|win|` / `|tie`** nothing more is read: the battle is over and no decision can follow (master sends the
   bare `|` + `|win|X` as its own message after a `|-message|X forfeited.`; no `|request|null`, `|raw|` or `|deinit`
   arrived in a 6 s window).
5. **The `rqid`** is the server's (injected into the request JSON; `ws_frontend.md` "The rqid"); the frame carries it
   and the client echoes it in `/choose <token>|<rqid>`. The training wire's frames carry `null`.

## The gates (P4)

| gate | instrument | proves |
|---|---|---|
| (a) TWO ROADS, ONE ROW | `python -m main.live.two_roads` | live frames over a real websocket (`--server rust`) == `sim_bridge` core_obs frames of the replayed battle, byte for byte, every decision (rqid excluded and checked present) |
| (b) the public replay corpus | `python -m main.live.replay_scan` | the corpus reads from both seats with zero refusals; the per-turn encoder probe finds no unclassified effect |
| (c) a Node Showdown at MASTER | `main.live` client vs bots / itself on a local master server | real framing, real `rqid`, zero T28 halts |
| (d) SHADOW | the poke-env `RLPlayer` + a shadow `LiveReader` on the same messages | poke-env's row / mask / action == the reader's, decision by decision |

Results: `designs/research_state/measurements/pokeenv_p4_live_2026-10-07/`.

## The BOT reader — a roster bot on the same chain (`bot_reader`, P6)

A `bot:<name>` our-side of an external-anchor read (`main.anchors`) is the Rust port of that roster bot
(`src/rust_env/src/bots/`, gated action-equal to the Python bots per decision) deciding on OUR side's reading of a
foreign stream. `src/rust_env/src/bot_side.rs` (`BotSide`) is one `SideReader` + one `bots::Bot`: each write goes
through `SideReader::advance` exactly as above, and at a decision the bot decides on that side's `BoardReading` over
the decision's own choice tokens (`present::choice_tokens`, the env core's mapper) — the env core's bot on the env
core's reading. The binary `src/rust_env/src/bin/bot_reader.rs` is this file's protocol plus one command and one reply:

| command | reply |
|---|---|
| `BOT {"bot":<name>,"seed":S,"env":h}` — BEFORE the first `OPEN`; the bot OUTLIVES every battle | `__OK__` |
| `FEED [...]` | at a decision `__BOT__ {"index","token","choice_words","protect_words","branch"}` BEFORE the `__OBS__` frame; then `__FED__` |

The bot's streams are `random.Random(opponents::stream_seed(S, h, k))` (choice k = 0, protect k = 1) — the env core's
bot-route rule — and run across the session's battles (one bot per half-series). The Python pipe is
`src/main/live/bot_reader.py` (`BotReader`, `BotFrame`, `BotPolicy`; a died process is REFUSED, never respawned —
a respawn would restart the streams); the binary is resolved / built by `src/utils/rust_env/bot_reader_bin.py`
(`$POKESIM_BOT_READER_BIN` overrides). Gate: `src/rust_env/tests/bot_side_test.rs` (a bot over its side stream ==
the env core's bot on a `search::game::Game`, token, index and both stream offsets, every decision, two battles
per bot without a rebuild); the identity against the deleted Python-bot client:
`designs/research_state/measurements/pokeenv_p6_anchors_2026-10-07/`.

## Limits (each a reason, not an oversight)

- **The probe cannot see the event window's cant-reason classification**: the trackers fold at DECISIONS, and a
  replay has none. On the live path an unknown cant reason IS refused at the decision's encode (a T28 halt) — the
  probe is a pre-detector, not the guarantee.
- **No reconnect inside a battle.** A rejoined room replays its log without our own choice notes, so the tracker rows
  after a rejoin could not be the training rows. The client ends the run (`ConnectionLost`, a crash — not a T28 halt).
- **Server-originated battle lines** reach the reader as protocol: `|-message|X forfeited.` (cosmetic), the turn-500
  `|bigerror|` countdown (ignored, as in training), a reconnect's `|player|pN|name|avatar|` (never on our path).
