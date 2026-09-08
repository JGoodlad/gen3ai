---
name: project_local_sim_bridge
description: "Local BattleStream bridge — run poke-env battles in-process with no Showdown server (issue #907 idea); powers the fuzz tests"
metadata: 
  node_type: memory
  type: project
  originSessionId: 354d210a-a811-420e-a488-f7b1e1bd8e98
---

> **Archived 2026-09-08** — the bridge is the default transport; root CLAUDE.md and src/utils/bridge/ are the docs of record. Preserved verbatim; nothing below is current.

`src/utils/bridge/` has a local-process-scoped Showdown bridge so a poke-env `Player` can run whole battles **in-process via `BattleStream` over stdio — no websocket server, no port, no usernames, no matchmaking**. This is the poke-env issue #907 idea: a transport swap that keeps `Player`/`Battle`/`parse_message` and feeds the identical `|...|` protocol stream, so the parsing/encoder pipeline runs byte-for-byte the same — only the byte source changes. (Distinct from the ai_v8 sim bridge, which is a `serializeBattle()` state oracle and bypasses the parser.)

**Built 2026-05-30. Zero edits to `src/poke_env/`** (it's vendored). The transport swap is done from our code: build players `start_listening=False`, then the runner reassigns `player.ps_client`. `BattleStreamClient` subclasses the vendored `PSClient` *from outside* poke_env.

Files:
- `local_sim_bridge.js` — Node relay: `START`/`CHOOSE`/`FORCELOSE`/`END` on stdin; per-side base64-framed protocol on stdout. One battle/process. Uses `getPlayerStreams()`; optional fixed PRNG seed.
- `battle_stream_client.py` — `BattleStreamClient(PSClient)`: no `.websocket`; routes `/choose`→bridge, no-ops `/utm`,`/timer on`,`/leave`,`/challenge`...; `log_in` no-op.
- `local_battle_runner.py` — `run_local_battles(p1, p2, n, *, seed=None)`, drop-in for `battle_against`. Fabricates the `>battle-…`/`|init|battle` room header the sim doesn't emit (poke-env keys `_create_battle` off it). Runs on `POKE_LOOP`; sequential per battle (process-per-battle teardown). The sim player name must equal `player.username` so poke-env resolves `player_role`.

All 8 fuzz/E2E tests migrated to `run_local_battles` and pass (event_log n=40, action/fuzz, transition_fuzz, poke_env_gaps/*, hidden_power) — *faster* than the websocket. See [[feedback_fuzz_tests]] and [[project_event_sourced_migration]].

**telemetry_e2e** stays on the websocket (unmigrated): empirically its "desync" assertion (request slot-order ≠ alphabetical `moves` keys) is deterministic — it reproduced ~84.5% on the local bridge, i.e. it is NOT a server-timing race. Kept on websocket since its stated purpose is real async-server timing.
