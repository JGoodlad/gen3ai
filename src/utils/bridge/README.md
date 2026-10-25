# Pokémon Showdown Bridge

`src/utils/bridge/` lets Python use the real `pokemon-showdown` library (the `deps/pokemon-showdown` submodule)
without a running server: team validation through Showdown's own validator, a streaming battle child, the
offline search / replay children, and a websocket front end that lets an outside client play our sim. Training
and eval do NOT go through this directory: they run on the Rust env core (`src/rust_env/`,
`src/utils/rust_env/`, `designs/training/rust_collector.md`). The Python RL transport, `run_local_battles`, the
poke-env `PSClient` transport and the Node offline drivers are deleted; what is left below is what exists.

The Rust side (the `sim_bridge` and `search_driver` binaries, their gates and the port itself) is owned by
`src/rust_sim/CLAUDE.md` and `designs/rust_sim/`; this file does not restate it.

| File | Role |
|---|---|
| `validate_team.js`, `team_validator.py`, `validate_stream.js` | Team validation (Node's one runtime job) |
| `hp_set_check.js` + `hp_set_check.py`, `get_hp.js` | Hidden Power oracles for `agents.gen3_data.hp_repair` and its tests |
| `local_sim_bridge.js` | The Node streaming battle relay: the reference arm of the differential harnesses and the websocket front end's dev child |
| `sim_bridge_bin.py` | Resolves every Rust binary (build, env override, build stamp, self-check build) and builds spawn argv |
| `seed_spec.py` | The one definition of the seed spellings a `START` / `resumeReseed` accepts |
| `reconstruction.py`, `search_session.py` | Offline replay / re-roll and the warm search session over the `search_driver` binary |
| `ws_frontend.py`, `ws_frontend_replay.py` | The bridge as a Showdown server for outside clients, and its byte-differential gate |
| `damage_probe.js` | Exact-HP ground truth from the omniscient `BattleStream` |
| `bridge_trace.py` | Opt-in raw wire trace (`POKESIM_BRIDGE_TRACE=<dir>`) for diagnosing a silent stall: whoever fell silent first is the tail of the log |

## Team validation

- `validate_team.js` imports Showdown's `TeamValidator` and `Teams`, reads a team and format from stdin and
  prints a JSON verdict. `team_validator.py` wraps it. `validate_teams_locally` is MEMOISED per process by
  (validator version, format, team text): the version is the submodule commit plus the built
  `team-validator.js` / `validate_team.js` stamps, so a Showdown bump cannot serve a stale verdict; a failure
  or malformed reply is never stored (`clear_memo()` forgets everything).
- `validate_stream.js` is the PERSISTENT form (one JSON request per stdin line, one reply per line; a team that
  does not import is `{"valid": null, "raised": …}`, never a verdict). Its caller is the search world builder's
  legality rule (`main/value_search/world_legality.py`). It reports a raise as `valid: null`, so it is not a
  drop-in for `validate_team.js`.
- `hp_set_check.py::check_teams` returns per set what `dex.getHiddenPower` gives the IVs, the IVs after the
  validator rewrote them, and the battle's stored stats. A Node failure raises.

```python
from utils.bridge.team_validator import validate_team_locally

result = validate_team_locally("gen3ou", team_text)
if result["valid"]:
    print("Team is valid!")
else:
    print(f"Errors: {result['errors']}")
```

## The battle child (`local_sim_bridge.js` and the Rust `sim_bridge`)

One battle per process (or many in sequence when a `START` carries `"persistent": true`). Commands on stdin:
`START <json>` (`{formatid, seed?, persistent?, resumeReseed?, p1:{name,team}, p2:{name,team}}`),
`CHOOSE <side> <choice>`, `FORCELOSE <side>`, `END`. Frames on stdout, base64 per chunk so one line is exactly
one side-tagged chunk: `p1 <b64>` / `p2 <b64>`, `__RECON__ <b64 json>` (the reconstruction record, once per
battle, just before `__END__`), `__END__`, `__ERR__ <b64>`. The header of `local_sim_bridge.js` and the module
doc of `src/rust_sim/src/bin/sim_bridge.rs` are the exact grammar.

The child has two implementations behind `impl={node,rust}`:

- **`rust`**: the std-only `src/rust_sim/src/bin/sim_bridge.rs`, a byte-for-byte protocol-compatible drop-in,
  validated at the chunk level by `src/rust_sim/harness/gen_sim_bridge_diff.js`. The only value the websocket
  front end's CLI accepts and the default everywhere.
- **`node`**: `local_sim_bridge.js`, the reference / oracle, reachable from code and from the harnesses.

Mechanics:

- `sim_bridge_bin.py::bridge_spawn_argv(impl)` gives the spawn argv: `node` is `["node", local_sim_bridge.js]`,
  `rust` is the resolved binary. `resolve_sim_bridge_bin()` takes `$POKESIM_SIM_BRIDGE_BIN` first, else runs
  `cargo build --release --bin sim_bridge` in `src/rust_sim` and caches the result; any failure raises
  `SimBridgeBinaryError` with the reason (never a silent fall-back to node).
- Every resolved binary must pass its BUILD STAMP (`<bin> --build-stamp` against this checkout's Rust sources,
  `utils.rust_env.stamp`): a stale binary raises `StampMismatch` naming the rebuild command
  (`src/rust_sim/CLAUDE.md` § Builds).
- Under `POKESIM_EMISSION_SELFCHECK=1` (set by the root `conftest.py`; never by production) every resolver
  builds the EMISSION SELF-CHECK build into `target/selfcheck/<bin>`, so it can never overwrite the binary a
  live run execs (`designs/rust_sim/emission_selfcheck.md`). A test that execs a pre-built binary directly asks
  `expected_bin_path(name)`.
- The node child exits drain-aware (`exitWhenDrained`): a bare `process.exit()` after a write discards Node's
  un-drained pipe output and truncated long `__RECON__` lines (the rust child blocks on the pipe per line). Python readers of a child accept long lines with
  `limit=BRIDGE_STREAM_LIMIT` (16 MiB, `ws_frontend`'s alone). Gate: `bridge_flush_on_exit_integration_test.py`.

### What the two children must agree on

An `__ERR__` is fatal to a caller, so **anything the node child tolerates on a CHOOSE, the rust child must
tolerate too**: a stricter parser there is a whole-run crash, not a better error. The contract, with its gates:

- `CHOOSE <side> default` (and `auto` / `pass` / `skip`) and a stray `CHOOSE` after `__END__` are tolerated
  (`default` resolves through `bridge::resolve_auto_choice`; `designs/rust_sim/search_and_replay_drivers.md`).
- A choice the request marks illegal gets the sim's own `|error|[Unavailable choice] …` (with a re-request) or
  `[Invalid choice] …` (with nothing after it), to the offending side ONLY
  (`src/rust_sim/tests/bridge_choice_reject_test.rs`; `designs/rust_sim/search_and_replay_drivers.md`).
  A classifier must never refuse the ONE action the request offered (a Struggle or move-locked mon).
- **An idle child means "waiting for a legal choice"**: after `[Invalid choice]` the sim emits nothing, so a
  quiet child is usually not wedged. Look at the last choice the DRIVER sent. Both children bound the spin:
  `REJECT_STREAK_CAP = 8` consecutive refusals at one boundary is a loud `__ERR__`, reset only by a COMMITTED
  decision (`node_reject_bound_integration_test.py` pins it; `src/rust_sim/harness/probe_illegal_choice_park.js` reproduces the park).
- `FORCELOSE <side>` ends the battle like Node: a real `win(otherSide)`, so BOTH sides receive `|win|<name>`
  before `__END__` (a bare `__END__` leaves a client's battle unfinished forever).
- A seeded battle is byte-for-byte with node, including a speed-tied lead and an unspecified-gender mon (the
  turn-0 construction window, `src/rust_sim/tests/turn0_construction_test.rs`). `seed_spec.py` lists the
  accepted seed spellings; an unparseable one is a loud `__ERR__`, never a silent fall-through.
- An unmodeled move or item fails LOUD (an `__ERR__ … is not modeled`), never desyncs. The move census is
  recounted, never quoted: `designs/rust_sim/move_census.md`.
- A "default" branch that no gate takes is untested however green the suite looks: a gate that is seeded can
  never exercise the SEEDLESS branch, so test both.

## Offline driver transport: the Rust `search_driver` binary only

The warm clone-and-branch search server and the one-shot replay / re-roll verbs are ONE Rust binary,
`src/rust_sim`'s `search_driver`. There is no engine choice and no `impl` argument for these children (the
Node drivers are deleted). Contract, verbs and gates: `designs/rust_sim/search_and_replay_drivers.md`.

| family | engine | resolver | env override |
|---|---|---|---|
| live transport | `sim_bridge` binary (`node local_sim_bridge.js` as the in-code oracle) | `resolve_sim_bridge_bin` / `bridge_spawn_argv(impl)` | `POKESIM_SIM_BRIDGE_BIN` |
| offline drivers | `search_driver` binary (both verb families) | `resolve_search_driver_bin` / `search_driver_spawn_argv()` (no argument) | `POKESIM_SEARCH_DRIVER_BIN` |

Both go through `_resolve_rust_bin(bin_name, env_var, selector)`: the env override first (no build, but the
stamp is still checked), else `cargo build --release --bin <name>` (the self-check build under
`POKESIM_EMISSION_SELFCHECK=1`), cached per bin name and build.

Callers:

- `SearchSession` (`search_session.py`): one process per `better_line` call, a synchronous request /
  one-line-response protocol over a background-drained queue (a wedged child fails ONE call, never hangs the
  prober). `open_root` / `expand_many` / `close`. A CORE root (`open_root(core="text", side=…, trackers=True)`)
  makes every arm a Rust-core `BattleVersion`, and `expand_many(rows=True)` returns each leaf's encoded row,
  mask and choice tokens (`designs/rust_sim/encoder.md`).
- `reconstruction.replay_battle / reroll_turn / reroll_many` through `_run_driver`, both verb families via
  `search_driver_spawn_argv()`.
- The prober: `ProbeSession`, `better_line_decision`, `lookahead_decision`, `falsify_decision`,
  `replay_counterfactual_battle` (in process on the Rust core), `core_trace.expand`.
- A typed `--impl` on `python -m main.prober.query`, `python -m main.prober.web`, `python -m main.prober` or
  `python -m main.harvest` is REFUSED at parse time with its reason (`sim_bridge_bin.refuse_search_impl_flag`);
  `ws_frontend --impl node` likewise.

Gates: `sim_bridge_bin_test.py` and `search_session_test.py` (the seam: the search driver takes no impl and
never names node, the overrides are independent, an unresolvable binary raises before spawning);
`src/node_free_search_integration_test.py` (under `utils/node_blocker.py`, which refuses and records any `node`
spawn: replay / reroll / reroll_many, a warm session, `better_line_decision`, `lookahead_decision`,
`team_details`, `query falsify`); `src/rust_sim/tests/{bridge_clone_branch,search_driver,replay_driver}_test.rs`;
`src/main/prober/better_line_integration_test.py`. **Run each parity gate on at least two fresh goldens before
calling it green**: one bug class appeared only on the seventh.

## Battle reconstruction (`reconstruction.py`)

Every bridge battle is reconstructable. The `__RECON__` frame carries the full-information record: the
RESOLVED PRNG seed (the sim mints one per battle when none is passed), both packed teams, the sim's own
`inputLog`, and the raw `commands` the child processed. The raw log is kept in addition to `inputLog` because
the sim logs only COMMITTED choices: a refused `[Unavailable choice]` probe never reaches `inputLog`, but its
`|error|` and re-request round is part of the protocol the agent saw.

- `ReconstructionRecord` (`from_dict`, `start_options()`, `players()`, `team_details(side)`) is the record.
- `replay_battle(record)` re-runs the battle verbatim through the `search_driver` one-shot `replay` verb and
  returns the regenerated per-side chunks and the final outcome, byte-identical to the live streams modulo the
  `|t:|` wall-clock line.
- `reroll_turn(record, t, seeds=…)` reconstructs to the start of turn `t`, then resolves that one turn under
  each fresh seed. Each side's start-of-turn action source is independently `recorded` / `random` / an explicit
  choice string; mid-turn follow-ups use a configurable `followup` policy.
- `reroll_many(record, t, arms)` is the BATCHED form: N independent arms (`{p1_action, p2_action, seed,
  label}`) of turn `t` in one driver process, each in its own fresh session, so a candidate sweep pays the
  spawn and rebuild once.
- `record.team_details(side)` / `decode_packed_team(packed)` is the ONE decode home (moves, EVs, IVs, nature,
  item, ability, level). It delegates to `utils.team_packing`'s `parse_packed_team`, resolves MOVE aliases
  through `agents.gen3_data.moves.move_aliases()`, and is validated field by field against the sim's own
  `Teams.unpack` over the whole team pool by `packed_team_decode_integration_test.py`. Replay and re-roll never
  decode: the packed strings go back to the sim verbatim, so counterfactuals run with the true hidden details.
- **The one-sided / omniscient wall (hard rule).** The record holds the opponent's team and the dice, which is
  referee-view data. It exists only at this layer and in the `*_reconstruction.json` artifact; the obs
  pipeline never reads it.
- A driver failure reports its reason as JSON on STDOUT and then exits non-zero, so `_run_driver` reads stdout
  FIRST and says "EMPTY stdout AND stderr" only when there is nothing.

## Warm clone-and-branch search

Re-replaying the prefix per node is infeasible at depth, so the search server clones a mid-battle state
in-process (`BridgeSession::snapshot`, a derived `Clone`) and applies one joint turn per `expand_many` arm via
the `resolve_turn` / `resolve_turn_exact` kernels of `src/rust_sim/src/search.rs`. `open_root {record, turn}`
reconstructs to turn T; `recorded_exact` (root only) reproduces the realized turn for the `value_crn` anchor.
Each expand returns THIS ply's one-sided suffix.

🚨 **A deepening caller must ACCUMULATE the per-ply suffixes; `expand_many` will not.** The composition is
`root-prefix + ply₁ + … + ply_d`. At depth 1 the two readings coincide, so a gate at depth 1 never sees a
missing ply; a hole is a different battle, not a coarser one.

## Single-turn damage oracle (`damage_probe.js`)

A batch request / response script that drives the OMNISCIENT (referee) `BattleStream`, which reports EXACT
both-side HP (`|-damage|p2a: X|389/461`) and exposes the sim's own `storedStats`, boosts, status, item,
ability, types, weather and side conditions. `stdin` is one JSON `{scenarios:[{id, formatid, seed?, p1:[sets],
p2:[sets], choices:[["p1","move 1"],…]}]}`; it prints one JSON line per scenario: `{id, weather, log, p1, p2}`
with a snapshot per side. Consumers: the Rust goldens' generators (`src/rust_sim/harness/gen_damage_golden.js`
and the state / stats / turn generators) and `src/agents/model/beatup_sim_parity_test.py`.

## Websocket front end (`ws_frontend.py`)

The transport for an opponent we do not own. An outside client's upstream poke-env cannot be imported into our
process, so the integration point is a socket: an asyncio websocket server speaking just enough of the Showdown
client protocol to log in, challenge and play, with each battle backed by ONE `sim_bridge` child. No Showdown
server, no port 8000 / 8001, no ladder.

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m utils.bridge.ws_frontend --port 9601 --impl rust        # ws://127.0.0.1:9601/showdown/websocket
python -m utils.bridge.ws_frontend --port 9601 --impl rust \
    --seed-base 914001 --capture-dir /tmp/caps                    # reproducible + replayable
```

🚨 **The sim emits no `rqid`, and the `rqid` is the decision trigger.** The real server injects it; a client
that never sees one never moves (every choice a client sends echoes it back). The front end splices
`,"rqid":N` onto each `|request|` as a string operation, so the rest of the line stays the sim's own bytes.

The gate is a byte differential, not "it ran": `ws_frontend_replay.py` replays a seeded battle's command
stream into the Node `local_sim_bridge.js` and compares the per-side protocol text byte for byte, with two
normalizations (`|t:|`, the injected `rqid`); `ws_frontend_byte_identity_integration_test.py` runs it. The
protocol surface, the deferrals (no timer, no ladder, no auth, no replays, no spectators) and the hazards (a
username over 18 characters is refused and a client that ignores `|nametaken|` then hangs; a `--capture-dir`
without a `--seed-base` is not replayable; the challenger is p1) are owned by
`designs/rust_sim/ws_frontend.md`.

## Timeouts and a busy box (`gen3_contention_robust_timeouts_v1`)

The bridge removes the server, not the CPU. This box normally carries a production training run, so every
wall-clock bound here is partly a measurement of load. Bounds are scaled by measured contention
(`utils.contention.scale_timeout`, read at CALL time: `max(1, loadavg/cpus)` clamped to 12x, so an idle box is
exactly 1.0); `GEN3AI_TIMEOUT_SCALE=N` forces the factor when you know the regime.

1. **A timeout is never a semantic outcome.** Count timeouts separately; more than 25% timed out is
   INCONCLUSIVE, not a verdict.
2. **Bound the IDLE gap, not the total duration,** wherever progress is observable
   (`contention.ProgressDeadline`): contention stretches a battle, only a real wedge stops the lines arriving.

Every timeout raised on this path appends `describe_contention()` (the load average and the top CPU
processes), so a starved failure says so itself.
