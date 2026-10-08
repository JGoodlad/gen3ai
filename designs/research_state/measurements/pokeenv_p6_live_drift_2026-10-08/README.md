# P6 slice 3 of the poke-env retirement — live play and the drift gate on the Rust stack only (2026-10-08)

Backlog T27, plan [`../pokeenv_and_hotpath_survey_2026-10-06/README.md`](../pokeenv_and_hotpath_survey_2026-10-06/README.md)
§A4.5 (P6). Slice 3 of 7: the last live-play poke-env users.

**Verdict: no check lost; one false failure of the retired path gone.** The pre-session drift gate's two reader
checks now run on the reader the live client runs, and judged the same inputs alike (below); P4's gate-(c) harness
plays the Rust ports of the roster bots on a master-built Node; `main.play` has one client.

## What moved

| surface | before | after |
|---|---|---|
| `main.play --client poke-env` | the legacy `RLPlayer` websocket client (vendored poke-env, the Python battle layer + encoder) | DELETED with `--avatar` / `--concurrency` (its own settings); a typed one is refused with its reason (`main.play.DELETED_FLAGS`); `main.live` is the only client |
| `main.live.gate_peer --role bot` | a Python `agents.opponents` bot on a poke-env client | the RUST port of the bot (`main.live.bot_reader`, slice 2's `bot_reader` session) on a Rust-stack client; `--bot` takes the env core's `bots::Kind` names, `--seed` the route seed |
| `main.live.gate_peer --role shadow`, `master_series --shadow-games` | P4 gate (d): the legacy client with the Rust reader beside it | DELETED — gate (d) is BANKED (16,523 decisions, 0 differences, [`../pokeenv_p4_live_2026-10-07/`](../pokeenv_p4_live_2026-10-07/)) and its client is gone |
| `ladder_drift_scan` read + encoder (replayed) | each replay replayed into a spectator `Gen3Battle` (`battle_event.classify`, `gen3_effects.encode_volatiles` after effect lines) | each replay read from BOTH seats by the live reader and encoded every turn (`main.live.replay_scan.scan_one`) |
| `ladder_drift_scan` encoder (source) | each source-derived effect line EXECUTED on a `Gen3Battle`, its ids checked against `gen3_effects` | each line read + encoded by the live reader after a fixed two-mon preamble (`main.live.effect_scan`); the TEXT scan of the Showdown source (`gen3_effect_sources.scan_emissions` / `concrete_lines`) is unchanged |
| format-spec check | `main.format_drift` | unchanged |

## Identity

**Encoder (source), line by line** (`scripts/effect_probe.py`): every concrete effect line the gen3 sim source can
announce, executed both ways at the same commit. On the pinned `deps/pokemon-showdown`: **93 / 93 lines judged alike
(93 clean on both)**; on a master checkout (`~/.cache/gen3ai/p4/showdown-master`, 51ad80fa): **93 / 93 alike**. The
new check adds the silent Minimize `|move|` line (94). Teeth (`scripts/teeth.py`): a fabricated `-start` /
`-activate` / `-singleturn` effect is an `UnknownVolatileError` on the Rust probe; Substitute, Heal Bell and Yawn
are clean.

**Read + encoder (replayed)** (`scripts/drift_ab.sh`): the retired scan (the `origin/main` file, run against the same
tree) and the new one on the same 200 public replays (2026-09-23, copied out of the main checkout's
`replays/showdown/gen3ou/`) and master's source:

| check | old (Python layer) | new (Rust reader) |
|---|---|---|
| encoder (source) | 68 derived ids, all classified | 94 lines, all read + encoded clean |
| format spec | ✓ | ✓ |
| replays read clean | **198 / 200** | **200 / 200** (both seats) |
| encode checks | 2,729 (after effect lines) | 11,226 (every turn, both seats) |
| exit | 1 | 0 |

The 2 old failures are the old scan's own artifact, not drift: both replays carry a player DISCONNECT / REJOIN
(`|player|p1|` with an empty name, then the name again), and the spectator `Gen3Battle` (username `p1`,
`_player_role` forced) re-derived its role and added the other side's mons to its team (`ValueError: p2's team
already has 6 pokemons`). The Rust reader's identity is the player NAME and an empty-name line renames nothing, so it
reads both seats clean. One of them is now committed as `src/main/live/testdata/player_rejoin.log`
(`effect_scan_test.py`: reads clean; the same replay with a planted unknown keyword FAILS). Counts differ by
construction (the live client's line filter; per-turn vs per-effect encodes; lines vs ids).

**Gate (c) on the Rust bot peers**: `main.live.master_series` on the master-built Node (port 9552, a fresh v145
debug checkpoint, CPU), run under `utils.poke_env_blocker`: heuristic2 / aggressive_v2 / staller × 3 games + 2
self-games, **788 decisions, 0 T28 halts, verdict PASS** (`master_series_blocked_summary.json`); an unblocked run
(heuristic2 / staller_v2 / setup_sweep × 3 + 2 self) also PASSED (615 decisions).

## FINDINGS

- **F-P6S3-1 — the retired drift scan misread a player rejoin as a structural failure** (above): any past drift read
  that reported `team already has 6 pokemons` was the scan, not the server.
- **F-P6S3-2 — the effect-source check now proves "reads and encodes", not "classified into the expected slot".** The
  Python check compared ids against `gen3_effects`' tables; the Rust probe passes any line the reader folds and the
  encoder writes without raising. An effect the Rust reader DROPS silently would pass both checks only if the reader
  drops it — the reader's crash-don't-drop rule (`UnknownVolatileError`) is what makes a pass meaningful.
  **UNVERIFIED:** that every volatile the Rust reader folds lands in its own slot (the encoder's gates — slice O, the
  engine audit — cover the corpus, not every source line).
- **F-P6S3-3 — gate (c)'s bots changed from the Python bots to their Rust ports.** They are gated action-equal per
  decision on a banked corpus (`tests/bots_gate_test.rs`); a P4-era gate-(c) number is not re-measured against them.
