# P4 — live websocket play on the Rust stack: the four gates (2026-10-07)

**Unit:** poke-env retirement **P4** (`../pokeenv_and_hotpath_survey_2026-10-06/README.md` §A4.5) + the owner's
**T28** parse-panic halt (`designs/endstate/design_ladder_campaign.md` Decision record, 2026-10-07).
**Code:** `src/main/live/` (the reader pipe, the client, the policies, the gate instruments, the halt),
`src/rust_sim/src/side_reader.rs` + `src/rust_sim/src/bin/live_reader.rs` (contract: `designs/rust_sim/live_reader.md`).
Landed at `25ea2cc6`.

**Owner policy honoured: ZERO contact with humans.** Every game below ran on 127.0.0.1. The public Showdown server was
never contacted (GitHub and the npm registry were, to build master). The public self-vs-self acceptance is a gated
command (`main.play --server official --public-acceptance`, both accounts in `$PS_OWN_ACCOUNTS`, `--proxy` required)
that was NOT run.

**Two code points.** The X5 version break (config v144, the OBS-FACTS block appended, `OBS_DIM` 2761 → 2845) landed on
main while P4 ran. Gates (a), (c) and (d) were run on BOTH sides of it: before it on the P4 commit as first built
(2761-dim rows, checkpoint `rb_x5ab_blob_s1008/final_model.zip`), and after it on the rebased tree (2845-dim rows).
Every archived checkpoint is behind the v144 floor, so the post-break model cells use a fresh v144 checkpoint from a
CPU `--debug` smoke (4,096 steps, written OUTSIDE `models/`): an untrained policy, which is what a shadow needs (any
policy at the current architecture) and which plays to the turn limit (so the stall forfeit is exercised). Gate (b)
reads no checkpoint.

## Verdict

| gate | instrument | n | result |
|---|---|---|---|
| (a) two roads, one row | `python -m main.live.two_roads` | 2,224 battles / 318,465 decisions | **PASS — 0 differing frames** |
| (b) the public replay corpus | `python -m main.live.replay_scan` | 376,410 replays × 2 seats / 128.4 M lines / 19.8 M encoder probes | **PASS — 0 unclassified refusals, 0 encoder failures; 1 classified refusal (spectator-only, root-caused)** |
| (c) a Node Showdown at MASTER | `python -m main.live.master_series` | 315 games / 36,290 of our decisions | **PASS — 0 T28 halts** |
| (d) shadow | `master_series` + `main.live.gate_peer --role shadow` | 120 games / 16,523 decisions | **PASS — 0 row / mask / action / token differences** |

**What this licenses:** P6 may rely on the Rust reader as the live reader. Reading a live game no longer needs the
poke-env client or the Python encoder. What still runs on poke-env is listed under "What remains for P6".

## (a) Two roads, one row

Each battle is played over a real websocket against our `--server rust` front end (`utils.bridge.ws_frontend`, seeded,
capturing), with two `main.live` clients. Each decision's frame is what `live_reader` built from the server's messages,
with the room framing (`|init|`, `|title|`) and the injected `rqid` present. Each captured command stream is then
replayed into a fresh `sim_bridge` with `core_obs` on for both sides: the chain whose rows are held byte-equal to the
env core (gate ①). Compared per side and per decision:

- the row BYTES (the base64 frame), the mask, the tokens, the turn, the stream line and the frame index `n`;
- the frame COUNT;
- the server's `rqid`, present on every live frame.

The replay must also reproduce the captured protocol chunks byte for byte (with `rqid` stripped). Every battle did, so
the two roads were provably the same battle.

| run | code | policy (both seats) | battles | decisions | failing |
|---|---|---|---:|---:|---:|
| `random_s11` | P4 as built (2761) | seeded uniform over the legal actions | 1,000 | 153,605 | 0 |
| `model_s12` | P4 as built (2761) | `rb_x5ab_blob_s1008`, greedy, CPU | 24 | 2,394 | 0 |
| `model_s13` | P4 as built (2761) | the same | 200 | 18,119 | 0 |
| `random_s14_v144` | rebased, post-break (2845) | seeded uniform | 1,000 | 144,347 | 0 |

The teeth are pinned by `live_integration_test.py::test_gate_a_two_roads_one_row_in_miniature`. One flipped base64
character in one live row is reported as `{"n": 3, "fields": ["frame"]}`.

## (b) The public replay corpus through the Rust reader

The corpus is the project's own download of the public replay archive: `<main checkout>/replays/showdown/gen3ou/`,
2026-05-18 … 2026-09-23, **376,410** gen3ou logs. The downloader is outside this repo; `ladder_drift_scan.py` and
`human_agreement.py` read the same tree. The scan downloaded nothing.

Every replay was read from BOTH seats as a spectator-style chain (no team, no request). It used the live client's exact
line filter and was fed turn by turn. After every turn, the seat's reading was ENCODED (`PROBE`).

| | count |
|---|---:|
| replays clean on both seats | 376,409 |
| protocol lines | 128,438,902 |
| distinct keywords | 63 |
| encoder probes | 19,804,295 |
| unclassified refusals | **0** |
| encoder (probe) failures | **0** |
| classified: `spectator-only: the viewer's seat was renamed mid-battle` | **1** |

**The one refusal, root-caused.** The replay is `battle-gen3ou-2678693343` (2026-09-10); the anonymized fixture is
`src/main/live/testdata/viewer_seat_rename.log`. The players' names are anonymized there and in the committed
`gate_b.*` files (`PlayerA`, `Guest 1`, `PlayerB`).

- **What happened.** A guest logged in mid-battle, so the server re-announced the seat as `|player|p2|<new name>|1|`.
- **Why it refused.** The reading's identity is the player NAME (`BoardReading::player`, poke-env's `_player_role` rule).
  Read AS the renamed seat, the role flips and the opponent's mons land in "our" team:
  `p1's team already has 6 pokemons: cannot add p1: Charizard`.
- **Why it cannot happen live.** A live client is never the renamed seat: its own name never changes inside a battle,
  and the client has no reconnect and no rename. An OPPONENT's rename leaves the role alone; the same replay read as p1
  reads to the end.
- **Where it is pinned.** `live_integration_test.py::test_a_mid_battle_rename_flips_only_a_spectator_reading_never_a_players`.
  The scan classifies the case deterministically from the seat's names, never from the message text.

**The probe's teeth, and its limit.**

- An injected unknown effect (`|-start|p2a: X|zzzunknowneffect`) fails the probe from either seat with
  `volatile "unknown" has no gen3 encoding slot`.
- An unknown `|cant|` REASON is NOT seen by the probe. Cant reasons reach the encoder through the event window, and
  the trackers fold the window only at decisions; a replay has none.
- On the live path that reason IS refused at the decision's encode (`cant reason "zzzreason" … is not a known gen3
  cause`), which is a T28 halt. So the limit is the drift scan's, not live safety's.
- `ladder_drift_scan.py`'s source-derived check (`gen3_effect_sources`) still covers the Python tables until P6.

## (c) A Node Showdown at MASTER

The server is Showdown master `51ad80fa5b264a28416b0d78667519a97ba2d2ee` (2026-10-07). It is a fresh download plus
`npm install` and `node build` under `~/.cache/gen3ai/p4/showdown-master/`, NOT the pinned `deps/`. It was started
with `--no-security` on a 9XXX port and stopped by PID. All 719 pool teams pass master's own validator. Our side is
the checkpoint on the Rust client, greedy, CPU, with a pool team per game. The opponents' poke-env clients ran in their
own processes. Metamon ran in its own interpreter through the anchors' peer plan (upstream poke-env, CPU, greedy).

| cell | games, pre-break (`run1` + `metamon1`) | games, post-break (`run2_v144` + `run3`) | T28 halts |
|---|---:|---:|---:|
| 7 scripted bots (`agents.opponents`: HeuristicV2, StallerV2, AggressiveV2, SetupSweepV2, Staller, Aggressive, SetupSweep) | 70 | 35 | 0 |
| Metamon `SmallRL` (greedy) | 20 | 10 | 0 |
| self-vs-self (two Rust clients: the acceptance's shape) | 40 | 20 | 0 |
| vs the shadow peer (gate (d)) | 60 | 60 | 0 |
| **total** | **190** | **125** | **0** |

These were also measured on master, in the series and in the build probe:

- **Framing.** A request arrives ALONE in its own message, after the message ending `|turn|N`. The sim's chunk may be
  split over several messages: `|t:|` + `|gametype|`, then `|player|p1`, then the rest. A message may end with a
  newline; the empty line it leaves is dropped as framing.
- **`|player|` lines** carry the avatar and an empty rating field.
- **The end of a battle.**
  - After a forfeit, `|-message|X forfeited.` and then `|` + `|win|Y` arrive as two messages.
  - The post-break untrained policy played to the turn limit: 26 of the 30 run-2 shadow games ended in the stall
    forfeit, often on both sides at once.
  - The room then sends `|l|`, `|player|p1|` (empty) and `|deinit`. The client reads none of them after `|win|`.
- **No halt** at any point, including after our own stall forfeit.

## (d) Shadow

Our checkpoint ran on the LEGACY poke-env client (`RLPlayer`, the path of `main.play --client poke-env`), accepting
challenges from the Rust client. A shadow `LiveReader` was fed every websocket frame the poke-env client received, in
arrival order, before poke-env handled it. The fork concatenates frames after a request ("SMART FLUSH") and handles
each message in its own task, so the shadow taps the connection, not the handler.

At each poke-env decision the shadow compares:

- poke-env's row against the reader's frame (bytes);
- the two masks;
- the model's argmax on each row;
- the token poke-env SENT against the reader's token for that action.

The sent token is then noted on the reader (`CHOOSE`), as the live client notes its own.

| run | code | games | decisions | differences | other events |
|---|---|---:|---:|---:|---:|
| `run1` | pre-break (2761) | 60 | 3,053 | 0 | 0 |
| `run2_v144` | post-break (2845) | 30 | 6,817 | 0 | 125 (a harness artifact, below) |
| `run3_v144_shadow` | post-break, harness fixed | 30 | 6,653 | 0 | 0 |

**The run-2 artifact.** All 125 "race" records sat in turn-250 battles, after the stall forfeit. poke-env forfeits at
the limit BEFORE it encodes (`_handle_stall`), so the reader's open frame was never consumed. Every later frame of that
battle was then flagged as "arrived before poke-env answered". The fix: a stall forfeit closes the open frame, recorded
as `stall_forfeit`, and the battle's forfeit notices and end are not a race. It is pinned by
`gate_peer_test.py::test_a_stall_forfeit_closes_the_open_frame_and_the_end_is_not_a_race`; run 3 re-ran the cell clean.
The comparator's teeth are pinned by `gate_peer_test.py`: a one-ulp change in one cell, a mask, an action and a token
are each reported.

## T28 behaviour

- **Exit code.** `FATAL_LIVE_PARSE` = 7 (`main.exit_codes`, mapped by the class NAME `LiveParseHalt`). The launcher
  never restarts it.
- **Triggers.**
  - a reader refusal: an unknown keyword, an unparseable line, an alignment break, an encoder raise, a dead process;
  - a frame with no legal token;
  - a policy that raises or picks an illegal index;
  - an `|error|[Invalid choice]`, meaning the server refused what we sent.

  The client drops only a DECLARED list of lines (`ROOM_SKIP`) and the empty string. An unknown battle keyword is never
  skipped.
- **Marker.** `~/.local/state/gen3ai/live_play_halt.json`, or `$GEN3AI_LIVE_HALT_FILE`. It records the battle id, the
  offending lines, the tail of the battle's received stream, the trace, the commit, the argv and the entry point. A
  second halt is appended, never replacing the first. `main.play` (every mode) and `main.anchors` refuse to start while
  it exists, and exit 7.
- **Clear.** `python -m main.live.halt clear --fixed-by <commit>`. It is refused unless the commit is an ancestor of
  HEAD and touches a test file. The cleared marker moves into an append-only history with the fix recorded.
- **Pinned end to end.** `live_integration_test.py::test_a_planted_bad_line_halts_and_refuses_the_next_start`: a fake
  server plants an unknown keyword; `main.play` exits 7; the marker names the battle and the planted line; the next
  start is refused before any connection.
- **Gate-local markers.** `master_series` points the marker at its own `<out>/live_halt.json`. A halt found BY a local
  validation series is the gate's finding and does not block another agent's live tool. The real marker governs every
  real live session.

## What remains for P6

- `main.anchors`' our side still runs the poke-env `RLPlayer` (`--client poke-env`, P3's unit). Until P3 lands, its own
  parse panics are not T28 raises; it only refuses to START past a marker.
- `ladder_drift_scan.py` still runs the Python checks, and `main.live.replay_scan` is its Rust twin. The source-derived
  effect check (`gen3_effect_sources`) has no Rust twin yet (the probe limit above).
- The last live-play importers of poke-env are the legacy `main.play` path (`main_poke_env`, `build_model_player`,
  `resolve_server`'s poke-env config) and `gate_peer --role shadow`. Deleting them shrinks the import allowlist by
  `src/main/play.py`.
- The official login (`_official_assertion`) is built but never run. A proxy-capable HTTP client for that login is
  NOT built (it refuses under `--proxy`). The gated public acceptance needs it, and the orchestrator's go.
- There is no reconnect inside a battle, by design: a rejoined room's rows could not be the training rows.

## Files

- `gate_a_<run>.summary.json`, `gate_a_<run>.verdicts.jsonl`: gate (a), per run and per battle.
- `gate_b.summary.json` (the keyword census, the classified refusal) and `gate_b.findings.jsonl`.
- `gate_cd_<run>.summary.json`, `gate_c.battles.jsonl` (one row per our-side seat) and `gate_d.shadow_digest.json` (per
  battle: decisions, differences, forfeits, other events).
- The per-battle captures (≈ 430 MB per 1,000 battles) and the raw shadow logs stay under `~/.cache/gen3ai/p4/`. They
  are not committed.
