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

#### Node vs Rust sim bridge (the `impl={node,rust}` argument)

> **Deletion pass U3:** the Python RL transport (`bridge_session.py`, `BridgeSession` / `attach_bridge_transport`, the Python `Gen3Env` it fed) is DELETED, and the trainer's `--use-bridge` has ONE legal value, `rust` (`node` / `off` are refused at parse time). Training and eval run on the Rust env core. `BridgeSession` in the Rust-side passages below is `src/rust_sim/src/bridge.rs`'s struct and is live; passages that still describe the Python session (`_dispatch`, `_child_error`, `_recycle_child`, the fatal-report latch) are RECORDED INCIDENT HISTORY of the deleted code.

The bridge child that speaks the `local_sim_bridge.js` stdin/stdout protocol has **two
implementations**, selected by an `impl` argument (`run_local_battles`, the offline drivers, the harnesses and benchmarks):

- **`node`** (`local_sim_bridge.js`) — the default bridge impl, a relay over the real Showdown
  `BattleStream`. Handles the full gen3 move/ability set and produces the `__RECON__`
  reconstruction record + honors `resumeReseed` (the forensic / search / counterfactual layers).
  **Every exit path is drain-aware** (`gen3_bridge_flush_on_exit_v1`, `exitWhenDrained`): a bare
  `process.exit()` after `out()` discards Node's un-drained async pipe writes, which TRUNCATED
  large `__RECON__` lines whenever the reader drained slowly (224 `Incorrect padding` capture
  failures in run_20260807_135637's final eval at `--eval-concurrency 100`; the persistent
  training path never exits, so it never truncated — and the rust bridge was never affected,
  its `LineWriter` blocks on the kernel pipe per line). Gate:
  `bridge_flush_on_exit_integration_test.py` (deterministic — a 1 MB `__ERR__` payload with a
  deliberately lazy reader; fails at ~48 KB on a pre-fix bridge). **Part 2**: once long lines
  arrive COMPLETE, the Python readers must accept them — both spawn sites pass
  `limit=BRIDGE_STREAM_LIMIT` (16 MiB; asyncio's 64 KiB readline default turned a delivered
  1000-turn-battle `__RECON__` line into `LimitOverrunError` → a crashed battle; training's
  250-turn stall cap is why the persistent path never tripped it). Same test file pins both.
- **`rust`** — the std-only `src/rust_sim/src/bin/sim_bridge.rs` binary, a byte-for-byte
  protocol-compatible drop-in (validated at the chunk/stdout level by
  `src/rust_sim/harness/gen_sim_bridge_diff.js`). No Node needed for battle stepping.

`sim_bridge_bin.py::bridge_spawn_argv(impl)` turns the impl into the spawn argv both transport
seam (`local_battle_runner.py`) execs:
`node` → `["node", local_sim_bridge.js]`; `rust` → `[<resolved sim_bridge binary>]`.
`resolve_sim_bridge_bin()` honors `$POKESIM_SIM_BRIDGE_BIN` (absolute-path override) first, else
runs `cargo build --release --bin sim_bridge` in `src/rust_sim` and caches the resulting
`target/release/sim_bridge`; it raises a clear, actionable error (never a silent fall-back to
node) if cargo/crate/binary is unavailable. Under **`POKESIM_EMISSION_SELFCHECK=1`** — set by the
root `conftest.py` for every pytest session and automatically for a `*fuzz_test.py` run as a script,
never by production — every resolver builds the EMISSION SELF-CHECK build instead (`cargo build
--profile selfcheck --features emission-selfcheck` → `target/selfcheck/<bin>`, its own directory, so
it can never overwrite the binary a live run execs): every emitted protocol line is checked at the
moment of emission and a failure kills the child (`designs/rust_sim/emission_selfcheck.md`). A test
that execs a pre-built binary directly asks `expected_bin_path(name)`, so it runs the same build.

**Honest scope of `rust` (re-audited 2026-08-04) — what works, and the ONE real gap.**
`__RECON__` (`gen3_bridge_recon_record_v1`) and `resumeReseed` (`gen3_bridge_resume_reseed_v1`) BOTH
shipped in `2b826d4`; the older "no `__RECON__` / no `resumeReseed`" text here was stale. A
**seeded** rust battle passes the whole forensic stack — `reconstruction_fuzz_test`,
`reroll_many_parity_fuzz_test` and `search_clone_parity_fuzz_test` all PASS on rust-recorded
records, with the Node replay/clone drivers reproducing the rust-recorded winner, turn and next obs
bit-for-bit (a cross-impl parity result worth more than the fuzzes' nominal subject). The record's
`>start`/`>player` lines are exact — the only part `replay_kernels.js` reads; its committed-choice
`input_log` lines are re-rendered from the engine's own script (`>p1 move 1` slot form vs Node's
`>p1 move icebeam`), replay-equivalent but not byte-identical, and consumed by nothing.

**⚠️ The three SEED gaps once listed here are FIXED** (`bc00d4d`,
`gen3_bridge_seedless_fixed_seed_v1` + `gen3_bridge_seed_forms_v1`; re-verified against the live
tree and the built binary 2026-08-04). Do not plan from the struck-through text:
- ~~The SEEDLESS path emits nothing and is not random.~~ **FIXED.** A seedless `START` MINTS a fresh
  `sodium,<hex>` (`Prng::generate_seed`) instead of falling through to `DEFAULT_CONSTRUCT_SEED`, and
  emits `__RECON__` carrying that resolved seed. So a `rust` run DOES write per-trace
  `*_reconstruction.json` (prober `falsify` / `better-line` / `replay-counterfactual` work) and every
  training episode draws its own dice. Gate:
  `bridge_impl_parity_test::test_seedless_rust_battles_are_distinct_and_recorded`.
- ~~A STRING `seed` is silently dropped.~~ **FIXED.** One shared `parse_seed_field` accepts every
  form Node accepts (`[a,b,c,d]`, `"m,n,o,p"`, `"gen5,<hex16>"`, `"sodium,<hex>"`); a
  present-but-unparseable seed is a LOUD `__ERR__`, never a silent fall-through. Gate:
  `test_seed_forms_reproduce_the_same_battle_on_rust_and_node`.
- ~~`resumeReseed` accepts only the array form.~~ **FIXED** — same shared parser, so the
  counterfactual Monte-Carlo works on rust.

**THE COVERAGE-HOLE LESSON (the durable part).** Every gate on that path — `sim_bridge_bin_test`,
`gen_sim_bridge_diff.js`, and the parity test's own win-rate check — was inherently SEEDED or
compared only aggregates, so the **production SEEDLESS branch was never exercised**. A "default"
branch that no test takes is untested however green the suite looks.

~~Still genuinely deferred: no rust clone-and-branch search driver.~~ **CLOSED.** The snapshot
primitive (`gen3_bridge_clone_branch_v1` — `BridgeSession::snapshot()`, a deep `Clone`, plus the
`clear_chunks`/`request_kind`/`is_choice_done`/`active_request_json`/`battle_state`/`winner`
surface; the `Battle::serialize`/`deserialize` stubs are deleted, the port's state needing no byte
format) and the driver on top of it both shipped — see *Offline driver transport* below. The
old search-teacher + rust guard is **gone**, and the reason it used to give — the record's
`input_log` being replay-EQUIVALENT rather than byte-identical — is **WRONG and RETRACTED**: no
consumer reads the committed-choice lines at all (`replay_kernels.js::writeStart` and
`ReconstructionRecord.start_options()`/`players()` read only `>start`/`>player`, which the rust
record renders exactly). The blocker was always the missing driver. Do not re-derive a plan from
the retracted reason.

**The CHOOSE path is gated for `__ERR__` parity** (`gen3_bridge_choose_path_parity_v1`). An
`__ERR__` is not an in-band error: `_dispatch` raises on it, `_persistent_read_loop` retires the
reader and trips `_signal_transport_dead()`, and every in-flight `step()` raises
`ShowdownException`. So **anything the node child tolerates on a CHOOSE, the rust child must
tolerate too** — a stricter parser there is a whole-run crash, not a better error. Two divergences
of exactly that shape killed `--use-bridge=rust --n-envs 48` at ~8 minutes, twice, at load 31 and
at load 5 alike (it is a RATE, not a load effect):
- **`CHOOSE <side> default` / `pass`.** Node writes every token to the sim verbatim
  (`local_sim_bridge.js:278`), so Showdown's `Side.choose` handled `default`/`auto`/`pass`/`skip`;
  the port's `parse_choice` took only `move `/`switch ` and answered `__ERR__`. These are ordinary
  production tokens, not a tail event — `singles_env.py`'s `action == -2`, an inference player
  whose predict returns `None`, its redecide-budget exhaustion, and `Player`'s
  `DEFAULT_CHOICE_CHANCE` fallback all emit `/choose default`. `default` now resolves through
  `bridge::resolve_auto_choice` (the one `Side.autoChoose()` port); `pass`, never legal in gen3
  singles, becomes the sim's own in-band `|error|[Invalid choice] Can't pass: …`.
- **A stray CHOOSE after `__END__`.** A persistent child resets itself at `__END__`, and
  `BridgeSession._dispatch` fires poke-env's feeds as UN-AWAITED tasks, so a late answer to the
  ending battle's last `|request|` routinely arrives with no battle live. Node drops it
  (`if (streams && streams[side])`); the port fell through to `flush_new_chunks` and returned
  `no battle in progress (missing START)`.

Gate: `bridge_impl_parity_test.py::test_poke_env_fallback_choice_tokens_never_produce_a_fatal_err`
and `::test_stray_choose_after_battle_end_is_ignored_on_a_persistent_child`, both parametrized over
node AND rust — node is the reference arm. Repro (no training, ~5 s):
`src/rust_sim/harness/rust_bridge_stray_choose_repro.py`. **Why the existing gate missed it:**
the (since-deleted) `bridge_session_fuzz_test.py --impl rust` drove only masked-legal `move`/`switch` tokens and
never landed a CHOOSE after `__END__`; a 16-worker / ~22 000-episode soak passes clean either way.

**The CHOICE-REJECT framing is MODELED** (`gen3_choice_reject_framing_v1`). When a client sends a
choice the request marks illegal, `bridge.rs`'s `RejectClass` / `classify_reject` emit the sim's own
`|error|[Unavailable choice] …` — for a `disabled` move, plus a re-request differing in exactly two
ways (`"update":true` and `"disabledSource":""` on the offending slot) — or `[Invalid choice] …`
with nothing following, for an out-of-range move slot and a `switch` into an active / fainted /
non-existent slot. The line goes to THAT side ONLY; the non-offending side receives ZERO chunks,
which is the half that matters, because re-opening the whole boundary made the other side record a
phantom extra pick. The forms are node-MEASURED bytes
(`src/rust_sim/harness/probe_choice_reject_framing.js`, re-runnable) — only the switch-into-a-FAINTED
slot string is read from source. The rule modeled is the CONDITION, not a per-message verdict:
`Side.emitChoiceError` emits `[Unavailable choice]` + re-issues the request IFF its update callback
actually CHANGED the request. Both parity harnesses now compare the framing chunks strictly (their
reject allowlist entries are deleted; the only live entry in either is `.error` message TEXT), and
`src/rust_sim/tests/bridge_choice_reject_test.rs` pins one case per measured class plus a
forced-Struggle control — a Struggle substitution is NOT a refusal, and classifying it as one emits
an `|error|` the sim never sends. ⚠️ A classifier must never refuse the ONE action the request DID
offer; that shape (`gen3_locked_choice_never_rejected_v1`) killed two production launches.

Still genuinely deferred:
- **`pre_state` volatile NAMES** are reconstructed from the port's typed fields rather than read
  from a keyed map. The golden verifies exactly one fact about them — that duration-1 volatiles
  (`focuspunch`, `pursuit`, `protect`, …) must not leak into a move-request boundary, which really
  did diverge and was fixed. Every other name is unverified because all 12 golden `pre_state`s end
  up empty. `pre_state` has no consumer today.

## Offline driver transport — `impl={node,rust}` (`gen3_search_driver_impl_seam_v1`)

`impl={node,rust}` selects the LIVE battle child of the library drivers. The **offline** children — the warm
clone-and-branch search server and the replay/re-roll primitives — are selected the same way, by an
`impl` argument threaded from the caller. Same module owns both: `sim_bridge_bin.py`.

| family | node | rust | resolver | env override |
|---|---|---|---|---|
| live transport | `node local_sim_bridge.js` | `sim_bridge` binary | `resolve_sim_bridge_bin` / `bridge_spawn_argv(impl)` | `POKESIM_SIM_BRIDGE_BIN` |
| offline drivers | `node search_driver.js` **+** `node replay_driver.js` | `search_driver` binary (**both** verb families) | `resolve_search_driver_bin` / `search_driver_spawn_argv(impl)` | `POKESIM_SEARCH_DRIVER_BIN` |

Note the asymmetry: node splits the offline verbs across two scripts (`search_driver.js` =
`open_root`/`expand_many`; `replay_driver.js` = `replay`/`reroll`/`reroll_many`), while the rust
port serves both from ONE binary — so `reconstruction._run_driver` keeps its own node script but
routes the rust branch through `search_driver_spawn_argv`.

Both resolvers share `_resolve_rust_bin(bin_name, env_var, selector)`: env override first (no
build), else `cargo build --release --bin <name>` in `src/rust_sim` (the self-check build under
`POKESIM_EMISSION_SELFCHECK=1`, `build_argv`), cached per bin name AND build across the
process, and a clear actionable error on any failure. **Neither ever falls back to node** — a
"rust" run that silently became a node run would answer a different question than the one asked.

Threading (every default is `"node"`, so this is byte-identical for every existing caller):

- `SearchSession(record=None, timeout=…, impl="node")` — stores `self.impl`; the child-death /
  timeout / desync messages name the impl **and** argv[0], so a rust failure self-diagnoses.
- `reconstruction.replay_battle / reroll_turn / reroll_many (…, impl="node")` →
  `_run_driver(request, timeout, impl)`. (`_sim_aliases` stays node-only by design: it dumps
  Showdown's own `aliases.ts` — a data query against the reference sim, not a sim run.)
- `obs_materializer.materialize_from_record / infer_action_indices (…, impl="node")`.
- `counterfactual.replay_counterfactual(…, impl=…)` → `run_local_battles(impl=…)` — this leg plays a
  REAL game, so it rides the LIVE `bridge_spawn_argv` seam, not the driver one.
- Prober: `ProbeSession(root, …, impl="node")` holds it **session-wide** (like `compile_extractor`)
  and every re-roll-backed probe reads it; the CLI exposes a global
  `python -m main.prober.query --impl {node,rust} <cmd>`. `better_line` REFUSES an injected warm
  `SearchSession` whose impl differs from the session's, so a correction can't be half-searched on
  one engine and half-confirmed on the other.

Tests — the SEAM: `sim_bridge_bin_test.py` (node argv, the `POKESIM_SEARCH_DRIVER_BIN` override,
bad-impl `ValueError`, independence of the two overrides, and a missing rust binary raising instead
of returning a node argv) + `search_session_test.py` (the exact historical node argv, rust execs
only the resolved binary, unresolvable rust raises **before** spawning, errors name the impl).

Tests — the ENGINE EQUIVALENCE (the claim that actually matters: rust answers the same question):

| gate | what it pins |
|---|---|
| `src/rust_sim/harness/search_impl_parity.py` | node vs rust on `open_root`/`expand_many` — 6 cases / 60 arms / **18877 leaf fields**, only `\|t:\|` normalized. **Re-measured on 7 FRESHLY generated goldens 2026-08-23: PASS on every one**, ~37.6k leaf fields each, allowlist 0 hits |
| `src/rust_sim/harness/replay_impl_parity.py` | node vs rust on `replay`/`reroll`/`reroll_many` — 76 cases / 136 arms / **30703 leaf fields**, incl. 9 error classes, 2 ended arms, 1 stuck arm. **Re-measured on the same 7 fresh sets: PASS on every one**, ~45-47k leaf fields each |
| `search_clone_parity_fuzz_test.py --impl rust [--record-impl rust]` | the rust clone ≡ the rust `reroll_many` **at the OBS**, bit-for-bit, + the `value_crn` anchor + depth-2 |
| `counterfactual_fuzz_test.py --impl rust [--record-impl rust]` | the CONFIRM leg — scripted-prefix obs oracle, divergence-to-terminal, Monte-Carlo reseed determinism |
| `main/prober/better_line_integration_test.py` | parametrized over both impls, **plus a cross-impl test** asserting node and rust yield identical candidate V. The fake model is `V = obs.sum()`, so an exact match is an obs-level bit-identity claim at every ply of the beam |
| `src/rust_sim/tests/{bridge_clone_branch,search_driver,replay_driver}_test.rs` | node-free: clone independence, the aux-RNG draw table, the `guard > 40` off-by-one, `recorded_queues` refusal-pull, one-shot dispatch |

⚠️ **A GOLDEN IS THREE RANDOM BATTLES, so ONE green run is weak evidence.** Both harnesses were
un-runnable from `ede4c79` until `f2bec7d`, and the first fresh goldens after that reported
divergences whose count swung **1 / 0 / 0 / 6 / 8 / 0 / 6 across seven seeds** — four distinct rust
bugs, no one of which every golden contains (the Return alias needs a Return carrier; the
`substitutebroken` gap needs a Substitute to break near a sampled turn; the fire-thaw one needs a
battle to END on a fire KO of a frozen mon; the single-entry-request one needs a mon at 0 PP on a
sampled turn — that last showed up only on the SEVENTH golden). All four are FIXED
(`gen3_fresh_golden_parity_triage_v1`), and the durable rule is: **generate a NEW golden and run
each gate on at least two different seeds** before reporting either one green.

**Performance — `search_impl_throughput_benchmark.py` (MEASURED, node vs rust).** Interleaved
per-rep A/B (order flips each rep) so a drifting box load hits both arms equally; medians, ms,
lower is better. Taken beside a live training run at load 8–24/16 cores, which the benchmark
announces via `warn_if_contended()` — absolute ms are inflated, the RATIO is the load-stable
signal and the IQRs below are disjoint by an order of magnitude.

| operation | node | rust | speedup |
|---|---|---|---|
| COLD spawn → 1st root | 438.6 | 9.7 | **45×** (paid once per non-warm `better_line`) |
| `open_root` turn 5 / 15 / 30 | 8.3 / 15.0 / 32.1 | 1.1 / 1.7 / 2.9 | 7.8× / 8.7× / 11.2× |
| `expand_many` per arm, turn 5 / 15 / 30 | 2.3 / 4.4 / 3.0 | 0.18 / 0.21 / 0.18 | 12.9× / 20.2× / 16.3× |
| `reroll_many` per arm | 73.4 | 2.2 | **33.8×** (node respawns a one-shot child per call) |
| child RSS | 193 MB | 9.3 MB | **20.7× smaller** |

IQR of `expand_many`/arm: node 3.00–4.59 ms vs rust 0.18–0.25 ms — disjoint, so the ratio survives
the contention. `open_root` rises monotonically with turn on both, which is the shape the prefix
replay must have; an earlier cut that showed turn 5 slower than turn 15 was folding V8 startup into
the first request, and the benchmark now measures COLD separately for exactly that reason.

**But the driver is no longer the bottleneck.** End-to-end `better_line` (depth 2, beam 3, top_k 4)
is **1.89× faster on rust** (1454 → 771 ms median, disjoint ranges) — far less than the 13–20× the
hot path gets, and a cProfile of one call says why: blocking child-wait falls from **51% of the
call on node to 4% on rust**, leaving Python-side obs materialization as the dominant cost, and
that is impl-invariant. Two consequences: (1) further engine work on this path is largely spent
effort — the materializer is the next lever, same lesson as the `torch.compile` saturation; (2) the
1.89× is an UPPER bound for a real search, because it was measured with the integration test's
`V = obs.sum()` stub, so a real extractor's forward adds impl-invariant time on both arms.

The RSS and cold-start numbers matter more than they look for any caller that runs several search
workers concurrently (the deleted search teacher ran `--teacher-search-workers` of them and respawned
per cycle): 5 workers is ~0.97 GB of node children vs ~47 MB of rust ones.

Both `--impl` and `--record-impl` exist on the two fuzz scripts on purpose: a MIXED run (train on
rust, run forensics on node) is the realistic deployment, so a record produced by either engine must
replay and search identically on both. The parity harnesses normalize `|t:|` and **nothing else
silently** — every unclosed divergence is an explicit printed allowlist entry with a reason and a
hit count, and both were fault-injection-proven (a one-line `pick_uniform` off-by-one → 2716
divergences; a dropped `|split|` shared line + a disabled `fnt` token → 1461).

**Coverage (MEASURED).** The port fail-louds (`__ERR__ … is not modeled` → `RuntimeError` → env
crash → launcher restart; the child itself survives via `catch_unwind`) rather than desync. On the
**training pool this is a non-issue**: 719/719 `data/teams/` teams construct and 1500 random-play
rust battles produced **zero** coverage errors. **⚠️ The randbats figures once quoted here (14.0% of
teams / 27.0% of battles) are STALE — every blocker they named is now CLOSED**, so do not plan from
them: the Deoxys/Unown forme DATA gap is fixed (`gen3_species_formes_v1`, ROUND 38 — 419 species rows
incl. the 33 gen-3 formes), `transform` is modeled (`gen3_transform_v1`, ROUND 33), the wrap family
too (`gen3_partial_trap_v1`, ROUND 32), and **Forecast/Castform — the last construction blocker — is
modeled** (`gen3_forecast_v1`, ROUND 35). Arbitrary ladder `gen3ou` remains ~5% of battles (INFERRED
from Smogon usage weights, not measured).

The old counter-hazard — unmodeled items/moves running as **silent no-ops** — is likewise CLOSED by
the ROUND 39/40 silent-no-op audits: the 5 genuinely-effectful unmodeled items (`shellbell`,
`berryjuice`, `mentalherb`, `machobrace`, `mail` — `gen3_unmodeled_item_failloud_v1`) FAIL LOUD at
construction. The 16 silent-desync moves that audit found (`fakeout`, `rollout`, the lock-in family,
…, `gen3_unmodeled_move_failloud_v2`) are now all **MODELED**, so `UNMODELED_FAILLOUD_MOVES` is an
EMPTY list; what keeps that seam honest is its negative controls
(`a_ditto_without_transform_builds_fine`, `transform_carriers_build_now_that_transform_is_modeled`),
which fail if an over-broad guard silently returns. The rest fail loud at RUNTIME instead (the
`run_status_move` / fixed-damage / charge guards). Full-universe census, re-run **2026-09-08** with
the ENGINE as the oracle (`src/rust_sim/src/bin/scan_move_probe.rs`): **369 gen3-legal moves → 312
modeled, 57 fail-loud, 0 MISMODELED**; **abilities 76/76** and **species 392/392** CLOSED;
**items 102/106**. Pool report: **762/762** teams fully engine-playable, out of 813 `.txt` files —
the 719-team TRAINING pool `TeamLoader` returns is a DIFFERENT filter and the two must not be
reconciled. The ranked remaining gap:
`designs/rust_sim/gen3_coverage_census_2026-09-08.md`. ⚠️ **Recount rather than quote** — the modeled
count moves with every coverage round, and this line has already gone stale three times (281/88 from
2026-08-04 while the tree was at 309/60; then 309/60 while the tree was at 312/57). 🚨 **Recount with
the PROBE, not the JS scan** — the JS scan's universe verdict is a hand-mirrored copy of `turn.rs`
and it is what went stale by 3 in 2026-09; the JS scan's own jobs are the pool report and the
invariant gate: `SCAN_UNIVERSE=1 SCAN_UNIVERSE_LIST=1 node src/rust_sim/harness/scan_move_coverage.js`
(exits non-zero if any silent desync reappears). **The load-bearing half is the invariant, not the split**: 0 MISMODELED is what
makes an unmodeled move a loud construction failure instead of a silent desync, and that has held
across every round. Measured exposure of the guarded sets is ZERO on both surfaces — they are
latent-hazard guards, not live failures. (An earlier note here named *Aromatherapy* and *Wish* as
unmodeled examples — both are in fact MODELED; the pool carries nothing unmodeled at all.)
- **Turn-0 construction — MODELED (`gen3_turn0_construction_v1`).** The bridge builds via
  `BridgeSession::new_construct_turn0`, which runs the sim's full turn-0 construction window from the
  RAW `>start` seed — the per-mon gender `sample(['M','F'])` + the speed-tie insertChoice/eachEvent
  shuffles (incl. Magnet Pull's `onAny` trap shuffles + a weather-setter's WeatherChange) + the Quick
  Claw — so a *seeded* rust battle is byte-for-byte with node even on a **speed-tied lead** or an
  **unspecified-gender** mon. (Formerly `advance_seed_for_construction` modeled only the Quick Claw and
  the diff harness skipped speed ties; both are gone.) The bridge runs that window with logging OFF
  and re-emits the leads' switch-in ability lines afterwards, so it also RECORDS the order the two
  `runSwitch` actions resolved to (`gen3_turn0_construction_mirror_order_v1`) — at a raw-Speed TIE
  that order is the `insertChoice` PRNG draw, p2-first half the time, and re-deriving it as
  "faster-first, tie = side order" permuted the Intimidate / weather-setter block on a tied lead
  (the seed and the board stayed correct; only the emitted lines were out of order). Gated by
  `src/rust_sim/tests/turn0_construction_test.rs` + `harness/gen_sim_bridge_diff.js`.

**Forfeit parity (`gen3_bridge_forfeit_win_v1`) — was the training-wedge bug.** A `FORCELOSE
<side>` must end the battle the way Node does: Node writes `>forcelose` INTO the sim, so Showdown
runs a real `win(otherSide)` and BOTH players receive `|` + `|win|<name>` before `__END__`. The
Rust bridge used to emit a bare `__END__` with no win line, which left poke-env's `Battle.finished`
False forever — the env's next `reset()` then waited on a result that could never arrive. Because
the training seam forfeits whenever `reset()` lands mid-battle, *every* episode boundary could
wedge, which is what stalled the multi-env `--use-bridge=rust` runs (they logged episodes but
never completed a single PPO iteration). Fixed by `BridgeSession::forfeit`; pinned by
`src/rust_sim/src/bridge.rs::a_forfeit_emits_the_win_line_to_both_sides_not_a_bare_end` and gated
end-to-end, when it was written, by the since-deleted `bridge_session_fuzz_test.py --impl rust` (whose every-9th-episode forfeit-reset was
the reproducer).

**Illegal-choice parity — the two impls fail DIFFERENTLY (know this before debugging a hang).**
Showdown's `Side.emitChoiceError` (`sim/side.ts:510`) branches on whether the refusal actually
CHANGED the request: `[Unavailable choice]` comes with a **re-request** (the client recovers),
while `[Invalid choice]` comes with **nothing at all** — the client is expected to re-pick from the
request it already holds. So a node child that goes quiet mid-battle is usually **not** wedged: it
refused an illegal choice and is waiting for a legal one (`probe_illegal_choice_park.js` drives
this deterministically: `move 4` on a 2-move mon → `[Invalid choice]`, 0 requests; a legal retry
then resolves the turn immediately). Two consequences:
- **BOTH impls now bound it** (`gen3_node_bridge_reject_bound_v1`): `REJECT_STREAK_CAP = 8`
  consecutive refusals at one boundary turns the spin into a loud `__ERR__` — rust in
  `BridgeSession`, node in `local_sim_bridge.js`. The counter resets **only on a COMMITTED
  decision** (node reads `battle.inputLog.length`, the one place an ACCEPTED choice lands);
  resetting on a received re-request instead would make the cap unreachable, which is the
  subtle way to get this wrong. The cap is generous on purpose — a handful of refusals is
  legitimate (the maybe-trapped probe is a normal two-exchange round; the max streak measured
  in normal play is **1**) — it exists to make an unbounded spin diagnosable, not to police
  ordinary refusals. Gate: `node_reject_bound_integration_test.py` (4 tests: the wedge pin,
  which HANGS pre-fix; an over-eager-cap guard; the reset-condition pin; and a cap-constant
  lockstep check), revert-verified — disabling the bound leaves the wedge pin spinning out its
  full 25 s budget with no `__ERR__` while the other three still pass.
  **MEASURED not-a-regression** (the real risk of a bound like this is a FALSE trip, not a missed
  one): the since-deleted `bridge_session_fuzz_test --impl node` 40 episodes clean; the bridge test package 91
  passed / 6 skipped; the python unit suite 3900 passed; and a 100-battle node-vs-rust
  `gen_sim_bridge_diff --mode randbats --persistent` soak came back **100/100 ended, 0
  divergences, 0 drain timeouts** while carrying **128 `trapped:true` frames** — i.e. the
  legitimate refusal round (the maybe-trapped probe) was exercised heavily and never tripped the
  cap. The max streak seen in normal play remains 1.
  **History (the deleted Python session):** the Python `BridgeSession` latched the `__ERR__` into `_child_error`, printed the real reason and the child's stderr tail the moment its reader retired (`gen3_bridge_fatal_report_now_v1`), and woke an in-flight `step()` by firing poke-env's `_disconnected` event captured off the `_EnvPlayer` queues (`gen3_bridge_child_error_wakes_step_v1`) — the lesson that outlives it: a dead bridge child must surface its OWN reason, never poke-env's generic websocket-dropped exception. `run_local_battles` (the eval driver) raises on the `__ERR__` frame directly.
- **Both impls validate the move INDEX against what the REQUEST OFFERED**, not against the moveset
  (`gen3_single_entry_request_slot_reject_v1`) — `Side.chooseMove`'s index check runs FIRST, ahead of
  the Struggle and Choice-lock substitution branches, so the two shapes that collapse the offered
  array to ONE entry (a 0-PP mon, a locked/charging mon) refuse `move 2`..`move 4` like any other
  out-of-range slot. An out-of-range `move N` is `[Invalid choice] Can't move: Your X doesn't have a
  move N` on node and rust alike, the turn does not advance, and either child can therefore park.
**Diagnostic rule: an idle bridge child means "waiting for a legal choice", so look at the last
choice the DRIVER sent, not at the child.** (Full diagnosis:
`designs/rust_sim/port_build_log.md` → ROUND 36.)

The move-name/switch-species transport parity (poke-env serializes choices by move-id + species
name, e.g. `move hiddenpowerice` / `switch Salamence` — not slot numbers) is exercised by
`bridge_impl_parity_test.py` (rust integration smoke + rust-vs-node win-rate parity at `seed=None`).

**Throughput (`bridge_impl_throughput_benchmark.py`, 16-core box, idle):** rust is FASTER than
node at every scale measured, and its child is an order of magnitude smaller:

| workers | node | rust | rust/node | node RSS/child | rust RSS/child |
|---|---|---|---|---|---|
| 8  | 1852 steps/s | 2182 steps/s | **1.18×** | 225 MB | 9 MB |
| 48 | 1942 steps/s | 2729 steps/s | **1.41×** | 223 MB | 9 MB |

The ~25× smaller child is the bigger operational win: at `--n-envs 48` the bridge children cost
~10.7 GB under node vs ~0.4 GB under rust. (An older note recording node 798 vs rust 427 fps at 8
envs was taken on a CPU-saturated box and is superseded — trust the ratio from a same-invocation
A/B on an idle machine.)

### Single-turn damage oracle (`damage_probe.js`) — exact ground truth, no poke-env

A third bridge mode drives the **OMNISCIENT** (referee) BattleStream directly — *not* the
per-side protocol poke-env consumes. The omniscient stream reports **EXACT both-side HP**
(`|-damage|p2a: X|389/461`, not the percent a player sees), and the live `battle` object exposes
the sim's **OWN computed stats** (`storedStats`), boosts, status, item, ability, types, weather, and
side conditions. So we can construct a battle with fully-specified teams, force a move sequence, and
read the **exact** damage a hit dealt — the clean ground truth for validating the differentiable
`DamageOperator`'s gen3 physics with **zero measurement confounds** (no percent-rounding, no
stale-HP, no overkill caps that plague scraping damage from random games).

- **`damage_probe.js`**: batch request/response over stdio (like `validate_team.js`). `stdin` = one
  JSON `{scenarios:[{id, formatid, seed?, p1:[sets], p2:[sets], choices:[["p1","move 1"],…]}]}`; it
  packs each team (`Teams.pack`), runs a fresh `BattleStream` per scenario, writes the choices to the
  omniscient stream, and emits one JSON line per scenario: `{id, weather, log:[omniscient lines],
  p1:<snap>, p2:<snap>}` where each snap carries `{species, maxhp, hp, stats, boosts, status, item,
  ability, types, sideConditions}`.
- Consumer: **`agents/training/poke_env_gaps/damage_op_probe_fuzz_test.py`** — the **authoritative**
  damage-op physics gate. It stages one modifier per scenario (type/STAB/super-effective/resisted/4×/
  type+ability immunity/Thick Fat/Choice Band/type-boost item/+Atk/+SpA boosts/burn/Reflect/Light
  Screen/rain/sun/defender +Def), measures p1's final hit on p2 with exact HP, and asserts the sim's
  damage lands inside the op's band (computed from the SIM's exact stats). The random-game
  `damage_op_fuzz_test.py` is a looser broad-coverage net by comparison (its per-side percent HP is
  inherently confounded — adjudicate any real physics question in the probe).

### RL-training transport — DELETED (deletion pass U3)

The Python gym-env transport (`bridge_session.py`: `BridgeSession`, `attach_bridge_transport`, persistent / spawn-per-battle child modes, `recycle_every`) is gone with the Python env core it fed; the RL rollout runs on the Rust env core (`src/rust_env/`, `src/utils/rust_env/`, `designs/training/rust_collector.md`). Its recorded measurements (persistent bridge ~2.1x over websocket per step, a node child's flat RSS) are history, not a current surface. What remains of the bridge on the Python side is the SYNCHRONOUS driver `run_local_battles` (eval opponents, the fuzz scripts), the offline search / replay children and the websocket front end below.

**The synchronous driver, `run_local_battles`.** Eval is a pure
synchronous-decision matchup (a greedy trainee vs a bot/sentinel — no SB3-supplied action): the eval worker's `_play_unit` calls
`run_local_battles` instead of `battle_against` when `use_showdown_bridge` is set (players built
`start_listening=False`). The config KEY `use_showdown_bridge` is a cross-process
worker-config contract, not a flag.

### Battle reconstruction (capture + offline replay / re-roll)

Every bridge battle is **fully reconstructable**. At battle end the child emits a
`__RECON__ <b64 json>` frame (just before `__END__`) carrying the full-information
reconstruction record: the **resolved** PRNG seed (the sim mints a fresh random one per battle
when none is passed — eval is reproducible-after-the-fact without pinning a seed), both packed
teams, the sim's own `inputLog`, and the raw `commands` the child processed. The raw command log
is kept *in addition to* `inputLog` because the sim logs only **committed** choices — a refused
`[Unavailable choice]` maybe-trapped probe never reaches `inputLog`, but its `|error|` +
re-request round *is* part of the protocol the agent saw, and replaying the raw commands
regenerates it exactly.

`reconstruction.py` owns the layer:

- **Capture join** — the `__RECON__` frame arrives *after* the `|win|` chunks (when the eval
  forensic trace is already written), so the two sides meet in a bounded registry keyed by battle
  tag: the demux calls `offer_record`, the forensic writer calls `register_trace_prefix`, and
  whichever lands second writes `<prefix>_reconstruction.json` next to the trace. (The counterfactual training half's record tap — the opt-in callable
  that wrote each episode's record into a `cf_records/` ring — was deleted in deletion pass L4, and the Python transport that owned it in U3, so a
  training episode's record is no longer persisted anywhere.)
- **`replay_battle(record)`** — re-runs the battle verbatim (`replay_driver.js`, batch
  JSON-over-stdio, no server) and returns the regenerated per-side protocol chunks +
  the final omniscient outcome. Byte-identical to the live streams modulo `|t:|` wall-clock
  lines (state/obs-invisible; in poke-env's `MESSAGES_TO_IGNORE`).
- **`reroll_turn(record, t, seeds=…)`** — reconstructs to the start of turn `t`, then resolves
  that one turn under each fresh seed by swapping `battle.prng` in place (every die routes
  through it). Each side's start-of-turn action source is independently `recorded` / `random` /
  an explicit choice string; mid-turn follow-ups in re-rolled timelines (forced switches that the
  original timeline may not contain) use a configurable `followup` policy.
- **`reroll_many(record, t, arms)`** — the BATCHED form: resolves N independent ARMS (each its own
  `{p1_action, p2_action, seed, label}`, with the EXACT per-side action semantics of `reroll_turn`)
  of turn `t` in **one Node process**. The dominant per-re-roll cost is the **~677 ms Node-spawn /
  pokemon-showdown `require`**, NOT the in-process `buildToTurn` (~26 ms warm), so a candidate sweep
  (the prober's one-ply lookahead) pays it ONCE instead of once per candidate — measured **~9×** on a
  full 9-legal-action sweep (5.5 s → 0.62 s). Each arm runs in its own fresh session, so its suffix
  chunks are **byte-identical** to the same single `reroll_turn` (modulo `|t:|`) and the materialized
  successor obs is **bit-for-bit identical** — pinned by `reroll_many_parity_fuzz_test.py`. (This is
  why `State.serializeBattle`/`deserializeBattle` is NOT wired here: the data shows the spawn dominates,
  not the rebuild, so cloning a battle snapshot wouldn't move the needle for these probes — batching
  does. `serializeBattle` is the lever for an in-process *tree* search, e.g. MCTS, which is out of scope.)

**The one-sided / omniscient wall (hard rule).** The record holds the opponent's team and the
dice — referee-view data. It exists only at this bridge layer and in the separate
`*_reconstruction.json` artifact; the obs pipeline never reads it. Offline obs come from
`agents.training.obs_materializer`, which is fed **only the per-side chunks** these primitives
regenerate and replays them through the real encoder (rebuilding tracker state). The round-trip
guarantee — materialized obs == the live `states.npz` rows **bit-for-bit** — is enforced by
`agents/training/obs_roundtrip_fuzz_test.py`; replay/re-roll invariants by
`reconstruction_fuzz_test.py`; the registry by `reconstruction_test.py`.

**Many arms of one decision share one prefix.** `obs_materializer.materialize_branches` replays a
decision's shared prefix ONCE and restores a snapshot of the player's battle/tracker state per arm,
instead of replaying from turn 1 for every arm (`arm_ms = 4.78 + 0.853·turn`, of which the prefix is
`2.53 + 0.855·turn`). It is defined to be exactly equivalent to a per-arm `materialize_decisions`,
**bit-for-bit** — measured 59/59 decisions / 452 arms byte-identical at **2.91×** (15.4 → 5.3 ms per
arm), gated by `agents/training/obs_materializer_branch_integration_test.py`. `lookahead` uses it
for its whole `(candidate × seed)` sweep.

⚠️ **A driver failure reports its reason as JSON on STDOUT, not on stderr**, and then exits
non-zero — so `_run_driver` reads stdout FIRST and says "EMPTY stdout AND stderr" when there is
genuinely nothing. Before that, the rust `search_driver` refusing turn 1 surfaced as literally
`failed (rc=1): ` with no reason at all, which reads like a crash and is not one. (That refusal is
**FIXED** — `gen3_search_turn1_open_v1`; the stdout-first read is what made it diagnosable.)

**Turn 1 opens on both impls** (`gen3_search_turn1_open_v1`). It did not until 2026-08-23: rust's
`at_turn_start` compared `BattleState::turn`, which the driver increments at `commitChoices` for
the first turn and then EAGERLY at every turn end — so from turn 2 on the field already names the
open boundary's turn, but at the FIRST boundary it still read `0` while the wire had already said
`|turn|1`. `build_to_turn` therefore walked the whole command log and reported "battle never
reached the start of turn 1", on BOTH verb families (`open_root` and the one-shot `reroll`
family). The predicate now maps a pre-commit `0` to `1` — exact rather than a fudge, because an
unbuilt battle (the other source of `turn() == 0`) has no boundary and is already excluded by the
`request_kind` conjunct, and the mapping is the identity for every `t >= 2`. Gates:
`src/rust_sim/tests/replay_driver_test.rs` (the predicate, `build_to_turn`, and both verbs through
the real binary) and `search_driver_turn1_integration_test.py` (`sim`: turn 1 opens on node AND
rust, with turn 2 as the identity control).

### Counterfactual replay-to-end (`counterfactual.py`)

Where `reroll_turn` re-rolls a SINGLE turn, **`replay_counterfactual`** picks up a recorded battle at
turn T, substitutes a different move for one side, and **plays the rest LIVE to a win/loss** — the
prober's "could it have won if it hadn't choked this turn?" (Feature 2). It reuses the
`run_local_battles` driver wholesale: both players are real poke-env players whose `choose_move` is
**scripted** (`install_scripted_prefix`) to replay the recorded commands until the divergence, then
handed back to the live policy. Faithful prefix: `START` uses the record's resolved seed + both packed
teams (turns 1..T-1 reproduce the real board), and each scripted `Gen3Player` decision runs
`embed_battle` + `tracker.advance(recorded_idx)` — the recorded index recovered by inverting the
recorded choice string through the real action mapper — so the **post-divergence turn-history stays
faithful**. At turn T our side plays the substitute and goes live; the opponent plays its recorded
turn-T move (it couldn't have reacted on the same turn) and goes live from T+1. The caller builds the
players (a greedy trainee + the RELOADED real opponent — a reproducible bot, a sentinel/stable
checkpoint, or a flagged self-model fallback; orchestrated by `src/main/prober/replay.py`).
`divergence_turn=None` scripts the whole game (the full-replay correctness oracle).
**Monte-Carlo**: `post_t_seed` (threaded into `START` as `resumeReseed: {turn, seed}`) swaps the sim
PRNG at the START of the divergence turn (mirroring `replay_driver.js`'s swap, but inside the live
`local_sim_bridge.js`), so the prefix keeps the recorded dice while each rollout resamples the
post-divergence dice → a win-rate ± CI. Faithfulness is proven by `counterfactual_fuzz_test.py`: a
full scripted replay reproduces the recorded **winner** AND the recorded one-sided obs **bit-for-bit**
(the `obs_roundtrip` guarantee carried through the live `run_local_battles` path), and the reseed keeps
the prefix fixed while varying the continuation. (`run_local_battles(..., start_extra=…)` is the generic
seam that merges extra `START` fields like `resumeReseed`.) For a human-readable **play-by-play**,
`run_local_battles(..., chunk_sink=[])` accumulates every `(side, chunk)` the bridge emits, and
`counterfactual.summarize_trajectory(side, sink)` parses OUR one-sided protocol into a per-turn
`{turn, events}` log (moves / switches / damage / faints / crits / status / win) — so a recovered
counterfactual win reads as an actual move-by-move line (`replay_counterfactual(..., capture_trajectory=True)`
→ the prober's `python -m main.prober.query replay-counterfactual --narrate`).

### Warm clone-and-branch search-server (`search_driver.js` + `search_session.py`)

Where `reroll_turn` / `reroll_many` rebuild the battle from turn 1 per call (O(commands-to-T)), a
multi-ply SEARCH (the prober's `better_line` beam) must branch a TREE from any explored node — so
re-replaying the prefix per node is infeasible at depth. The search-server clones a **mid-battle state
in-process** via Showdown's `State.serializeBattle` / `deserializeBattle` (`dist/sim/state.js`): a
verified round-trip of a battle paused with both move requests open (deserialize rebuilds the open
requests via `getRequests`; PRNG continuity restored from the live counter), at **~1.7 ms/clone —
~16× cheaper than the ~26 ms warm `buildToTurn`, and CONSTANT in depth**.

- **`search_driver.js`** — a WARM, persistent Node process (vs `replay_driver.js`'s one-shot batch)
  holding a node-snapshot cache. `open_root {record, turn}` reconstructs to turn T (via the shared
  `replay_kernels.js`), serializes the root, and returns the request/team-complete prefix chunks.
  `expand_many {arms}` clones a parent node, applies one joint turn (our action + the opponent's, via
  the same `resolveTurn`/`resolveTurnExact` kernels), and re-serializes the child. A deserialized
  battle can't re-emit the historical `|request|` lines (requests are out-of-band, not in `battle.log`),
  so each expand returns this ply's one-sided **suffix** (flushed-prefix baseline + the new turn) and
  the Python caller composes `root-prefix + each ply's suffix` — the same `(prefix + suffix)` shape
  `reroll_many` produces. `recorded_exact` (root only) reproduces the realized turn for the
  `value_crn` anchor. The per-side chunk splitting is REUSED verbatim (a deserialized battle is
  re-attached to a fresh `BattleStream` and `restart()`-ed with the same `send` wiring).
- **`search_session.py`** (`SearchSession`) — the Python wrapper: one process per `better_line` call
  (context-managed), a synchronous request→one-line-response protocol over a background-drained queue
  (a wedged child fails ONE call, never hangs the prober). `open_root` / `expand_many` / `close`.
  A CORE root (`open_root(core="text", side=..., trackers=True)`, rust only) makes every arm a
  Rust-core `BattleVersion`; `expand_many(rows=True)` then returns each leaf's ENCODED row + mask +
  choice tokens (`core_pN`), which is all a search leaf needs (`gen3_core_search_v1`,
  `gen3_core_encoder_v1`; contract `designs/rust_sim/encoder.md`). The one-sided `view_pN` payload
  and its Python consumers are deleted (program §4 M2).

🚨 **A DEEPENING caller must ACCUMULATE the per-ply suffixes; `expand_many` will not do it.** The
composition is `root-prefix + ply₁ + … + ply_d` — the same plies the branch's `actions` list names.
`ExpandedNode`'s docstring used to promise "the COMPLETE one-sided view, root → this node" and the
search-dividend deepener believed it, so a depth-`d` successor was replayed as `prefix` + ply `d`
with plies `1..d-1` MISSING (`gen3_search_depth2_chunk_gap_v1`). **At depth 1 the two readings
coincide**, which is why no gate saw it. A hole is a different battle, not a coarser one: poke-env
keeps applying lines to the board it last saw, so a switch in the gap logs `"Message thinks p1: X
is active, but it's not"` and an opponent reveal in the gap makes a later reference construct a
Pokémon whose *species* is the NICKNAME (`KeyError: 'ptãra'` — reported as an encoding bug, but the
mojibake is in the committed team file and any nickname raises). Gate:
`main/search_dividend/depth2_replay_integration_test.py`, over both impls.
🚨 **AN OFFLINE REPLAY MUST NOT SHARE A ROOM WITH A LIVE BATTLE.** `BattleStreamClient` writes
`CHOOSE <side> <choice>` keyed on the ROOM, and `search_dividend.record.install_choice_tap` is a
process-wide patch on that method whose only filter is the room — so a replay player running under
the live battle's tag had its `/choose default` orders recorded into the LIVE reconstruction
record. `obs_materializer._next_tag` now always mints a unique tag
(`gen3_recon_tag_collision_v1`); the cost of the collision is in that function's docstring.

- **`replay_kernels.js`** — the shared sim kernels (`buildSession` / `buildToTurn` / `resolveTurn` /
  `resolveTurnExact` / `recordedQueues` / `randomChoice` / `outcomeOf` / …) lifted out of
  `replay_driver.js` so the replay/re-roll path and the search-server use ONE implementation (no
  drift; the trusted reroll path is byte-for-byte unchanged).

**Faithfulness (a NEW path, so proven not asserted):** `search_clone_parity_fuzz_test.py` (real
bridge, no server) asserts over many battles that a depth-1 clone's successor obs equals the
`reroll_many` obs **bit-for-bit**, that the `recorded_exact` clone reproduces the recorded `states.npz`
next obs (the `value_crn` anchor), and that a depth-2 clone composes a valid chain. The omniscient
clone/outcome/teams/dice drive only the opponent + the dice — never the obs encoder (the one-sided
wall). Consumed by `main.prober.better_line`; see `src/main/prober/CLAUDE.md`.

**Reading the teams for review** — `record.team_details(side)` / `decode_packed_team(packed)`
is THE one decode home (moves, EVs, IVs, nature, item, ability, level; omission-defaults
applied; ids resolved through the sim's alias table — a pool export can say `wisp`, everything
downstream speaks `willowisp`). It delegates to poke-env's `parse_packed_team`, never a
reimplementation, and is validated **field-by-field against the sim's own `Teams.unpack` over
the ENTIRE team pool** by `packed_team_decode_integration_test.py` (719 teams / 4314 mons —
which caught the alias case on its first run). Replay/re-roll never decode at all: the packed
strings go back to the sim verbatim, so counterfactuals always run with the true hidden details
(exact Hidden Power types, speed stats, damage ranges).

### Websocket front end (`ws_frontend.py`) — the bridge as a Showdown *server* for OUTSIDE clients

**The transport for an opponent we do not own.** `run_local_battles` is a
*library* seam (the Python `BridgeSession` was another, deleted in U3): it assigns a `BattleStreamClient` onto a poke-env `Player` **in this process**.
That is closed to a third party for the reason the metamon de-risk recorded as its verdict (d) —
their player subclasses *upstream* poke-env, ours subclasses the *vendored fork*, and one process
resolves `import poke_env` to exactly one of them. So the integration point cannot be an import; it
has to be a **socket**, with the opponent in its own process and its own poke-env.

`ws_frontend.py` is that socket: an asyncio websocket server speaking just enough of the Showdown
*client* protocol (`|challstr|`, a no-op `/trn`, `|updateuser|`, `/utm`, `/challenge`+`/accept`,
the `>battle-…` room framing, `|request|` **with an `rqid`**, `/choose`, `|win|`/`|tie|`,
`|deinit|`, `|pm|`/`|popup|`) for a poke-env-style client to log in, challenge and play — with each
battle backed by ONE `sim_bridge` child. No Showdown server, no port 8000/8001, no ladder.

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m utils.bridge.ws_frontend --port 9601 --impl rust        # ws://127.0.0.1:9601/showdown/websocket
python -m utils.bridge.ws_frontend --port 9601 --impl rust \
    --seed-base 914001 --capture-dir /tmp/caps                    # reproducible + replayable
```

🚨 **The sim does NOT emit an `rqid`, and the `rqid` is the decision trigger.** `Side.emitRequest`
sends the bare request object; the SERVER injects the id (`server/room-battle.ts:796-801`, one
counter shared by both slots). A client that never sees one never moves — Foul Play literally
cannot, since every choice it sends echoes the id back. The front end therefore splices
`,"rqid":N` onto each `|request|` at exactly that point, and reproduces the two `sideupdate` state
transitions that go with it (an `[Invalid choice]` reopens the slot; an `[Unavailable choice]` does
not). The splice is a string operation, not a JSON round trip, so the rest of the line stays the
sim's own bytes.

**The gate is a byte differential, not "it ran".** `ws_frontend_replay.py` replays a seeded
battle's recorded command stream into the Node `local_sim_bridge.js` and compares the per-side
protocol TEXT byte for byte, with exactly two normalizations (`|t:|`, and the injected `rqid`).
Measured 2026-09-14 against `local_sim_bridge.js`, with the front end on `--impl rust`: **40 / 40
battles byte-identical, 44,034 protocol lines, 3,597 `|request|`s** — 20 vs Metamon `SmallRL`
(greedy) and 20 vs Foul Play (`--search-time-ms 300`), both against
`ai_v12_02_winprob_critic`@75M, with zero protocol errors on any side.

**The protocol surface and the full deferral list — no timer, no ladder, no auth, no replays, no
spectators — live in [`designs/rust_sim/ws_frontend.md`](../../../designs/rust_sim/ws_frontend.md),
which OWNS them.** Hazards worth knowing before you start one: a username longer than **18
characters** is refused and a client that ignores `|nametaken|` then HANGS; a `--capture-dir`
without a `--seed-base` is not replayable; and the **challenger is p1**, as on a real server.

## Why use a Bridge?
- **Serverless**: No need to start or manage a Pokémon Showdown server process.
- **No contention**: Each call/battle is fully isolated — no shared server lifecycle, no
  username collisions, no port. Multiple fuzz tests can run at once without fighting.
  (This means no *lifecycle* contention. It does NOT mean no *CPU* contention — see
  **Timeouts and a busy box** below.)
- **Speed**: Local function calls beat websocket round-trips (the fuzz suite is *faster*
  on the bridge than on the server).
- **Accuracy**: It uses the *exact* same Showdown code as the live server (same `deps/` files).
- **Reproducible**: the battle bridge accepts a fixed PRNG seed.

## Timeouts and a busy box (`gen3_contention_robust_timeouts_v1`)

The bridge removes the *server*, not the *CPU*. This box normally carries a production training
run, so every wall-clock bound here is partly a measurement of the load average — a healthy battle
that takes ~2 s idle takes ~6 s beside a 48-env run, for reasons that have nothing to do with the
sim.

All three bounds on this path are therefore scaled by measured contention
(`utils.contention.scale_timeout`, read at CALL time so the factor tracks load as it develops;
factor = `max(1, loadavg/cpus)` clamped to 12x, so an **idle box is exactly 1.0 and nothing
changes**):

| bound | baseline | where |
|---|---|---|
| per-battle | `_PER_BATTLE_TIMEOUT` 180 s (parity test overrides to 20 s) | `local_battle_runner._per_battle_timeout()` |
| silent-stall watchdog | `_RACE_GET_TIMEOUT_S` 120 s (`GEN3_RACE_GET_TIMEOUT_S`) | `poke_env.environment.env._race_get_timeout()` |

Two rules this encodes, both learned the hard way:

1. **A timeout is never a semantic outcome.** `bridge_impl_parity_test` used to fold a per-battle
   timeout into its "unmodeled move" SKIP bucket; beside a live trainer that turned 39/40 starved
   battles into a clean-looking pass that blamed the Rust port's move coverage for the box's load.
   Timeouts now have their own counter, and >25% timed out is INCONCLUSIVE, not a verdict.
2. **Bound the IDLE gap, not the total duration,** wherever progress is observable
   (`contention.ProgressDeadline`). Contention stretches how long a battle takes; only a genuine
   wedge stops the protocol lines arriving. A total-duration cap cannot tell those apart, so the
   only way to stop it flaking is to raise it until it stops catching the real bug too.

Every timeout raised on this path appends `describe_contention()` — the load average plus the
`ps -eo pcpu,pid,args --sort=-pcpu | head` command — so a starved failure says so itself.
`GEN3AI_TIMEOUT_SCALE=N` forces the factor when you already know the regime.

## Usage

### Team validation
```python
from utils.bridge.team_validator import validate_team_locally

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

### Impl selection (`run_local_battles(..., impl=…)`)
The trainer's `--use-bridge` has ONE legal value, `rust` (since U3 — `node` and `off` are refused at parse time, and the Python RL transport that took `node` is deleted). `node` remains an explicit `impl` for the A/B arm and the parity harness; the websocket server is only for `play.py` / the ladder. The deprecated `--use-showdown-bridge` boolean alias is DELETED.
`run_local_battles(..., impl=…)` takes the impl for the eval driver.
