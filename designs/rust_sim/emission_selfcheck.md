# The EMISSION SELF-CHECK — every line checked at the moment it is emitted (`gen3_core_emission_selfcheck_v1`)

<!-- ALWAYS-CURRENT (a `designs/rust_sim/` topic doc): state the truth, never narrate a change. The
leaf `src/rust_sim/CLAUDE.md` keeps the rule, the command and the hazard; this file keeps the
detail. Program context: `designs/endstate/program_rust_core.md` (§3, the parity gate). -->

The whole-battle parity harness (slices E and V, [`core_events.md`](core_events.md) §7,
[`present.md`](present.md) §5) proves a battle's typed events and views agree AFTER the battle. The
self-check is its per-emission complement: at the moment a line is emitted it asserts the line is
what every reader of it is owed, and a failure stops the build that found it — on the exact line,
with both viewers' renders in the message.

Code: `src/rust_sim/src/emission_check.rs` (the checks, the counters, the binaries' exit guard);
the call sites in `protocol.rs` (`ProtocolBuilder::emit` / `retro_edit`), `bridge.rs`
(`derive_side`, `BridgeChunks::push_chunk`, `split_log_lines`) and the three binaries
(`sim_bridge`, `search_driver`, `core_events`). Python: `utils.bridge.sim_bridge_bin` (the build
switch), the root `conftest.py`, `agents.battle.rust_core_parity` (the counts in its census).

---

## 1. What is asserted

| # | where | the assertion | failure tag |
|---|---|---|---|
| 1 | `ProtocolBuilder::emit` and every `attrLastMove` retro-edit | the omniscient line is CANONICAL: `Line::parse(render(l)) == l` and the committed text is `render(l)` | `round-trip` |
| 2 | `bridge::derive_side` — the one funnel the framing and every flushed batch go through, once per viewer | the viewer is sent the line iff it is not owner-only for the other side, and its text parses back to `side_view(l, viewer)` — the TYPED privacy fold (`core_events/side.rs`), an implementation independent of the text fold being checked | `per-viewer`, `withheld`, `owner-unknown` |
| 3 | `bridge::split_log_lines` (the replay family's `battle.log` shape) | each `\|split\|pN` triple: the secret half is the owner's line, the shared half the other viewer's (EMPTY for an owner-only line); an un-split line reads the same to both viewers | as 2 |
| 4 | `BridgeChunks::push_chunk` — every frame the bridge builds itself (`\|request\|`, `\|error\|`, the forced-Struggle `\|-activate\|`) | the frame parses under a known keyword and re-renders to its bytes; a `\|request\|` carries the id of the side it is shipped to, a Struggle announce names that side's mon | `unparseable-frame`, `frame-round-trip` |

**A secret never reaches the other viewer** — asserted as its own, spec-level invariant beside the
per-viewer equality, so a defect shared by BOTH folds (text and typed) is still caught: in a percent
format the non-owner's HP field is a `pct/100` (never the owner's `hp/max` when `max != 100`); an
owner-only line (gen-3 Pressure's `-ability …|[silent]`, the Intimidate-vs-Substitute `-hint`) is
absent from the other viewer's render and has an EMPTY shared half; a one-side frame is shipped only
to its side. Tag: `SECRET LEAK`.

The failure message: `EMISSION SELF-CHECK FAILED [<tag>] viewer=<p1|p2|omniscient>`, then the
omniscient line, both viewers' renders, what the failing viewer was owed and the typed values.

## 2. When it runs, and why production cannot pay for it

The predicate is `cfg(any(debug_assertions, feature = "emission-selfcheck"))`, on EVERY CALL SITE
(`#[cfg(…)]`, so in a build with neither the call is not compiled at all) and as the constant
`emission_check::ENABLED`. The checks themselves are always compiled and unit-tested, so they cannot
rot between the builds that use them (the reason M1 deleted its cfg-gated spike feature,
[`core_events.md`](core_events.md) §1, does not apply: only the calls are gated, not the code).

| build | self-check | who uses it |
|---|---|---|
| `cargo test` (the `test` profile has `debug_assertions`) | **ON** | every Rust test, and every binary a test spawns (`CARGO_BIN_EXE_*`) |
| `cargo build --profile selfcheck --features emission-selfcheck` → `target/selfcheck/` | **ON** | the four A/B fuzzers (`ab_fuzz.js` state + `--protocol`, `bridge_ab_fuzz.js`, `gen_sim_bridge_diff.js`), `search_impl_parity.py` / `replay_impl_parity.py`, and every Python process under `POKESIM_EMISSION_SELFCHECK=1` — every pytest session (root `conftest.py`) and every fuzz script run directly (`sim_bridge_bin._auto_selfcheck`: an entry script named `*fuzz_test.py` / `*fuzz_e2e_test.py`) — so slice E/V's `core_events`, the Python fuzz scripts' `sim_bridge` and the search tests' `search_driver` |
| `cargo build --release` → `target/release/` | **OFF — compiled out** | training (`utils.bridge.sim_bridge_bin` builds exactly this), eval, play, anchors |

* **The self-check build has its OWN directory** (`[profile.selfcheck] inherits = "release"` in
  `Cargo.toml`): it can never overwrite the `target/release/` binary a live run execs, and a
  production resolver never reads `target/selfcheck/`. `sim_bridge_bin.build_argv` /
  `expected_bin_path` are the one place the two are named; the tests that exec a pre-built binary
  directly (`bridge_impl_parity_test`, `ws_frontend_byte_identity_integration_test`,
  `search_teacher_composition_test`, `untaught_meter_reproducibility_integration_test`) ask
  `expected_bin_path`, so they run the build the suite is on.
* **An explicit `POKESIM_*_BIN` override still wins** — point it at a `target/selfcheck/` binary to
  keep the check on. `POKESIM_EMISSION_SELFCHECK=0` turns the switch off explicitly (it is never
  overridden). A process that PUBLISHES the bridge for training (`resolve_and_publish_sim_bridge_bin`)
  under the switch prints a loud warning: a run launched from a shell that exported it would
  otherwise train on the self-check build.
* **A test build never continues past a failure.** It panics. `sim_bridge` and `search_driver`,
  which catch a handler panic per request and keep serving (`__ERR__` / `ok:false`), EXIT with
  status `EXIT_STATUS` (86) on a self-check panic instead (`emission_check::exit_if_failure`,
  itself compiled out of `--release`); `ab_replay` / `bridge_replay` report it as a `panic`
  verdict, which fails their green gate; `core_events` dies and `rust_core_parity.run_core`
  raises.
* **The counts are the proof the check ran.** `emission_check::counts()` (`[omniscient,
  per_viewer, split, frame]`, relaxed atomics); `core_events` prints `summary()` to stderr at exit
  in a self-check build and `rust_core_parity`'s census carries it. **The parity gate REFUSES a
  run with no counts** (`rust_core_parity_test._assert_selfcheck_ran`) — a production
  `core_events` makes slice E/V fail rather than pass without the check.

## 3. The proof of zero production cost

Measurement record:
[`research_state/measurements/rust_core_emission_selfcheck_2026-09-23/`](../research_state/measurements/rust_core_emission_selfcheck_2026-09-23/README.md).

* **Compiled out**: the `--release` `sim_bridge` contains no `emission_check` symbol and no
  `EMISSION SELF-CHECK` string (the self-check build has 9 and 1); `tests/emission_check_test.rs::
  every_call_site_is_compiled_out_of_release` scans `src/` and fails any call without the `cfg`
  within the three lines above it.
* **Byte identity**: on the M1 record's recorded 41-battle / 6,078-`CHOOSE` training transcript,
  `sim_bridge` stdout is byte-identical (sha256 `4d0e37a6f80dcdd3…`, 20,216,559 bytes, 41
  `__END__`, 0 `__ERR__`) for the pre-change release build, this change's release build and the
  self-check build.
* **Cost** (the M1 record's replay bench, 30 interleaved triples, one binary per process): the
  production build's `sim_bridge` CPU is **0.9957×** the pre-change build's [0.9926, 0.9997] — no
  slowdown; the self-check build's is **1.308×** [1.303, 1.313] (+9.3 µs per `CHOOSE`), paid only by
  tests and fuzzers.
* **Milestone-scale run with the check ON** (same record): slice E/V MILESTONE (942 battles) +
  200 procedural battles — ≈ 1.05 M omniscient emissions, ≈ 2.1 M per-viewer renders, ≈ 203 k
  frames checked; the four A/B fuzzers (2,470 battles) and ten Python fuzz scripts — **0
  self-check failures**.

## 4. The gates

| gate | proves |
|---|---|
| `emission_check::tests` (`cargo test --lib`) | each check passes the truth and fails each defect class with its tag and both renders (a non-canonical line, an exact HP at the non-owner, a wrong percent, an owner-only line broadcast or withheld, an Intimidate hint with no owner, a `\|split\|` shared half that leaks, a frame shipped to the wrong side) |
| `tests/emission_check_test.rs` | ON in a test build; a real bridge battle reaches all four checks (counts); a self-check failure KILLS `sim_bridge` (exit 86, no `__ERR__`); every call site is compiled out of `--release` |
| every other `cargo test` | runs with the check on every emitted line |
| `utils/bridge/sim_bridge_bin_test.py` | the two builds never share a directory; the switch selects the build and keeps separate cache slots; a fuzz script turns it on, a production entry point does not, an explicit value wins |
| `rust_core_parity_test.py` (COMMIT + MILESTONE) | slice E/V ran on the self-check `core_events` (the counts) |
