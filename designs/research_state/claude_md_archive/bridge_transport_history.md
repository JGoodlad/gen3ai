# Bridge transport — the closed history

Lifted verbatim from the root `CLAUDE.md` § *In-process bridge transport* on **2026-09-07**, when
that section was cut from 225 lines to ~40. Everything here is **CLOSED**: the three seed defects,
the two CHOOSE-path divergences and the choice-reject entry are all FIXED, and the measurements
carry their original dates. The live rules stayed in `CLAUDE.md`; this is the evidence behind them.

Do **not** re-derive a plan from anything below without checking the current code first.

---

### In-process bridge transport (`--use-bridge {off,node,rust}`, **default `rust`**)

`--use-bridge` swaps **both training and eval** between a websocket Showdown server and an
in-process `BattleStream` subprocess — no server, no port, no `/challenge` connection storm,
deterministic delivery (poke-env issue #907). **THE DEFAULT IS `rust`, so a run needs no Showdown
server at all.** It reuses the *entire* obs/reward/mask/wrapper stack unchanged.

**The three values:**

| value | transport | when |
|---|---|---|
| **`rust`** | the std-only pokesim `src/rust_sim/src/bin/sim_bridge.rs` | **the DEFAULT** — fastest, smallest child, serverless |
| `node` | the Node `local_sim_bridge.js` | the explicit A/B arm; the parity harness and `gen_sim_bridge_diff.js` need it |
| `off` | websocket to a Showdown server on `--showdown-port` | the ladder / live-server path |

`rust` is a byte-for-byte protocol-compatible drop-in for `node` (validated by
`src/rust_sim/harness/gen_sim_bridge_diff.js`), so nothing above the transport changes. The
DEPRECATED `--use-showdown-bridge` boolean alias is **DELETED**: it meant `--use-bridge=node`,
which is no longer the default, so keeping it would have made the legacy spelling silently select
the slower impl. The flag resolves into one internal `bridge_enabled: bool` + `bridge_impl:
"node"|"rust"`; `bridge_impl` is threaded to `attach_bridge_transport(env, …, impl=…)` (training),
`run_local_battles(…, impl=…)` (eval driver), and the eval-worker shard config
(`bridge_impl` alongside `use_showdown_bridge`). The binary is resolved by
`src/utils/bridge/sim_bridge_bin.py::resolve_sim_bridge_bin()`: `$POKESIM_SIM_BRIDGE_BIN`
(absolute override) first, else `cargo build --release --bin sim_bridge` in `src/rust_sim`
(cached; a clear error, never a silent node fall-back, if cargo/crate/binary is missing).

**`rust` deferrals + coverage limit (honest, warned at startup; re-audited 2026-08-03).** The old
"the Rust bridge emits **no `__RECON__`**" claim is **STALE** — `2b826d4` shipped both
`gen3_bridge_recon_record_v1` and `gen3_bridge_resume_reseed_v1`, and a **seeded** rust battle
passes the whole forensic stack: `reconstruction_fuzz_test` (replay reproduced the rust-recorded
winner + turn), `reroll_many_parity_fuzz_test`, and `search_clone_parity_fuzz_test` (the clone's
successor obs equals the rust-recorded `states.npz` next obs **bit-for-bit**) all PASS when the
record is fed to the Node `replay_driver.js` / `search_driver.js` — a strong cross-impl parity
result, since the offline replay/clone layer is Node either way. What is **actually** still broken:

**⚠️ THE THREE SEED DEFECTS BELOW ARE FIXED — do not re-derive plans from them** (`bc00d4d`,
`gen3_bridge_seedless_fixed_seed_v1` + `gen3_bridge_seed_forms_v1`; re-verified against the live
tree + binary 2026-08-04). Kept as history because the *coverage-hole lesson* is the durable part.
- ~~The SEEDLESS path produces nothing / replays one dice stream.~~ **FIXED.** A seedless `START`
  now MINTS a fresh `sodium,<hex>` (`Prng::generate_seed`) instead of falling through to
  `DEFAULT_CONSTRUCT_SEED = "0,0,0,0"`, and still emits `__RECON__` (the resolved seed is the
  minted one). So eval traces DO get their `*_reconstruction.json` sibling and every training
  episode draws its own dice. Gate: `bridge_impl_parity_test::
  test_seedless_rust_battles_are_distinct_and_recorded` (three seedless battles must hash
  DIFFERENTLY and each carry a `__RECON__`).
- ~~A STRING `seed` is SILENTLY IGNORED.~~ **FIXED.** One `parse_seed_field` accepts every form
  Node accepts — `[a,b,c,d]`, `"m,n,o,p"`, `"gen5,<hex16>"`, `"sodium,<hex>"` — and a
  present-but-unparseable seed is a LOUD `__ERR__`, never a silent fall-through. Gate:
  `test_seed_forms_reproduce_the_same_battle_on_rust_and_node` (parametrized over all forms).
- ~~`resumeReseed` accepts ONLY the array form.~~ **FIXED** — same shared parser, so the
  counterfactual Monte-Carlo works on rust.
- ~~The clone-and-branch SEARCH server has no rust path.~~ **CLOSED — search runs on rust.** Three
  layers landed together:
  - **The snapshot primitive** (`gen3_bridge_clone_branch_v1`) — `BridgeSession::snapshot()`, a
    derived deep `Clone`, plus `clear_chunks` / `request_kind` / `is_choice_done` /
    `active_request_json` / `battle_state` / `winner`; gated by `tests/bridge_clone_branch_test.rs`.
    The `Battle::serialize`/`deserialize` stubs are **DELETED**: Showdown needs a byte format
    because its battle graph is cyclic, but the port's state is plain owned data and the snapshot
    never crosses a process boundary, so an in-process clone is the whole requirement.
  - **The drivers** — ONE binary, `src/rust_sim/src/bin/search_driver.rs`, serves BOTH offline verb
    families: `open_root` / `expand_many` / `close` over a persistent stdin loop
    (`gen3_rust_search_driver_v1`) and the one-shot `replay` / `reroll` / `reroll_many`
    (`gen3_rust_replay_driver_v1`; node splits these across `search_driver.js` +
    `replay_driver.js`). `src/rust_sim/src/search.rs` ports `replay_kernels.js` — the mulberry32
    aux-RNG, `random_choice`, `resolve_turn`/`resolve_turn_exact`, `recorded_queues`, `outcome_of`.
  - **The Python seam** (`gen3_search_driver_impl_seam_v1`) — `resolve_search_driver_bin()` /
    `search_driver_spawn_argv(impl)` mirror the `sim_bridge` pair (env override
    `$POKESIM_SEARCH_DRIVER_BIN`, else `cargo build --release --bin search_driver`, cached, clear
    error, **never** a node fall-back). `impl="node"|"rust"` threads through `SearchSession(impl=)`,
    `reconstruction.{replay_battle,reroll_turn,reroll_many}(impl=)`,
    `obs_materializer.{materialize_from_record,infer_action_indices}(impl=)`, the prober
    (`ProbeSession(impl=)` + `python -m main.prober.query --impl {node,rust}`), both search-teacher
    workers, `SearchTeacherCallback(impl=args.bridge_impl)` and `teacher/generate.py`'s
    `run_local_battles` (which used to take node silently). **Every default is `"node"`**, so this
    is byte-identical until someone asks for rust.

  **Gates** (all run): `src/rust_sim/harness/search_impl_parity.py` 6 cases / 60 arms / 18873 leaf fields and
  `src/rust_sim/harness/replay_impl_parity.py` 76 cases / 136 arms / 30689 leaf fields, both node-vs-rust with only
  `|t:|` normalized; `search_clone_parity_fuzz_test --impl rust [--record-impl rust]` (clone ≡
  `reroll_many` at the OBS, bit-for-bit); `counterfactual_fuzz_test --impl rust`;
  `better_line_integration_test`, parametrized over both impls plus a **cross-impl** test asserting
  node and rust produce identical candidate V — and since that fake model's `V = obs.sum()`, an
  exact match is an obs-level bit-identity claim at every ply of the beam.

  **The `--search-teacher` + rust guard is GONE.** For the record, the reason it used to give was
  **WRONG** — not `input_log` byte-identity: *nothing reads the record's committed-choice lines*
  (`replay_kernels.js::writeStart` and `ReconstructionRecord.start_options()`/`players()` read only
  `>start`/`>player`, which the rust record renders exactly). The real blocker was always the
  missing driver. **Do not re-derive a plan from the old reason.** Not yet gated: a full multi-cycle
  teacher run end-to-end on rust — every leg is gated, the composition is not.

  **The CHOOSE path is `__ERR__`-parity gated** (`gen3_bridge_choose_path_parity_v1`). An `__ERR__`
  is NOT an in-band error — it retires `BridgeSession`'s reader, trips `_signal_transport_dead()`,
  and raises `ShowdownException` in every in-flight `step()` — so **any CHOOSE node tolerates, rust
  must tolerate too**; a stricter parser there is a whole-run crash. Two such divergences killed
  `--use-bridge=rust --n-envs 48` at ~8 min, twice, at load 31 and at load 5 alike (a RATE, not a
  load effect): `CHOOSE <side> default`/`pass` (node passes every token to `Side.choose` verbatim;
  `parse_choice` took only `move `/`switch ` — and `/choose default` is routine, from
  `singles_env.py`'s `action == -2`, an inference player's `None` predict, its redecide exhaustion,
  and `DEFAULT_CHOICE_CHANCE`), and a **stray CHOOSE after `__END__`** on a persistent child (the
  child resets at `__END__` while `_dispatch` fires poke-env's feeds as un-awaited tasks). Gates:
  `bridge_impl_parity_test.py::test_poke_env_fallback_choice_tokens_never_produce_a_fatal_err` /
  `::test_stray_choose_after_battle_end_is_ignored_on_a_persistent_child`, both over node AND rust.
  **The existing fuzz gate could not catch either** — it drives only masked-legal tokens and never
  lands a post-`__END__` CHOOSE (22k episodes × 16 workers pass clean pre-fix). A related
  DIAGNOSTICS fix ships with it (`gen3_bridge_fatal_report_now_v1`): the `__ERR__` text used to be
  latched into `_child_error` and printed only by the NEXT `reset()`, which never runs, so the sole
  surviving evidence was poke-env's generic "websocket dropped" — `_report_fatal` now prints the
  reason + child stderr tail immediately. (And `race_trace.dump_recent()` is a no-op unless
  `GEN3_RACE_TRACE=1`; an empty dump means the buffer was off, not that nothing happened.)

  **⚠️ The old CHOICE-REJECT allowlist entry was FALSE ON BOTH HALVES and is DELETED — do not
  re-derive a plan from it.** It claimed rust "emits no `|error|` frame and re-opens the boundary to
  BOTH sides ... on a path poke-env never takes". The framing half was closed by
  `gen3_choice_reject_framing_v1` (the entry survived its own fix); the "never takes" half was
  falsified by poke-env taking it and killing **two production launches** at ~8 minutes. The real
  defect was `gen3_locked_choice_never_rejected_v1`: a MOVE-LOCKED mon (two-turn charging, or
  `must_recharge`) gets a single-entry request with `trapped:true`, and `classify_reject` — which
  never consulted `move_locked()` — could then REFUSE the only move that request offered, because
  `move_usable` models Choice-lock/Disable/Encore/Taunt/PP and knows nothing of lock-in. Rust
  contradicted ITSELF; it was not a stricter parser. Fixed with a `move_locked()` early-out beside
  the existing `must_struggle` one (same principle: **a forced choice is not a refusable one**), and
  one predicate covers the charge family and the recharge mirror, so fly/dig/bounce and hyperbeam go
  with it. Gated by `bridge_choice_reject_test::a_move_locked_mon_is_never_rejected_for_its_only_offered_move`
  — VERIFIED failing on revert. **The existing fuzz cannot catch this class by construction**: it
  drives masked-LEGAL tokens, and here the token IS masked-legal (the mask is built FROM the
  request), so 22k episodes passed clean while it was live. **The durable lesson: an allowlist entry
  can outlive its own fix and then mislead every reader after — including a subagent briefed from
  it.** One honest gap remains, allowlisted and printed by both harnesses, never silent: (2) `pre_state` volatile NAMES are reconstructed from the port's typed
  fields, and the golden verifies exactly one fact about them (duration-1 volatiles must not leak
  into a move boundary — that one really did diverge and was fixed). `pre_state` has no consumer.

**THE DURABLE LESSON from the seed defects** (why they shipped at all): every gate on that path —
`sim_bridge_bin_test`, `gen_sim_bridge_diff.js`, and the parity test's own check 2 — was inherently
SEEDED or compared only aggregates, so the **production SEEDLESS branch was never exercised**. When
a code path has a "default" branch nothing tests, that branch is untested no matter how green the
suite looks.

**Coverage (MEASURED, not asserted).** The port fail-louds (`__ERR__` → `RuntimeError` → env crash
→ launcher restart; the child survives via `catch_unwind`) rather than desync. On the **training
pool it is a non-issue**: 719/719 `data/teams/` teams construct (719 = the LOADED/deduped count
from 773 raw `.txt` files — both figures appear in this doc and mean different things), and 1500 random-play rust battles
hit **zero** coverage errors (the only 4 failures were the 1000-turn runaway cap). On
`gen3randombattle` **the construction blockers are CLOSED**: the Deoxys/Unown forme DATA gap is
fixed (`gen3_species_formes_v1`, ROUND 38), `transform` is modeled (`gen3_transform_v1`, ROUND 33),
the wrap family too (`gen3_partial_trap_v1`, ROUND 32), and **Forecast/Castform — the last blocker
(~2.6% of teams) — is modeled (`gen3_forecast_v1`, ROUND 35, with the hail/sandstorm weather-set
moves and the expiry-draw fix it flushed out)**. Arbitrary ladder gen3ou is ~5% of battles
(INFERRED from Smogon usage weights). The old inverse hazard — unmodeled items/moves running as
SILENT no-ops / generic hits — is CLOSED by the ROUND 39/40 silent-no-op audits: the 5
genuinely-effectful unmodeled items (`gen3_unmodeled_item_failloud_v1`) and the 16 silent-desync
moves (`fakeout`, `rollout`, the lock-in family, `eruption`, … —
`gen3_unmodeled_move_failloud_v2`; full-universe census **re-run 2026-08-23**: 369 gen3-legal
moves → **309 modeled, 60 fail-loud, 0 silent** — recount with `SCAN_UNIVERSE=1 node
src/rust_sim/harness/scan_move_coverage.js` rather than quoting this, the modeled count climbs
with every coverage round) now FAIL LOUD at construction, and every gen3 ability is modeled or
verified-no-op (none fail-loud). Measured exposure of the guarded sets is ZERO on both surfaces
(0 pool carriers; 0 in the entire curated randbats movepool) — latent-hazard guards. (The former
**seeded speed-tied-lead** / unspecified-gender divergence is FIXED —
`gen3_turn0_construction_v1` models the turn-0 construction window, so a seeded rust battle is
byte-for-byte with node.) See `src/utils/bridge/README.md`. Transport parity (poke-env sends
move-ids/species names, e.g. `move hiddenpowerice`) is guarded by
`src/utils/bridge/bridge_impl_parity_test.py`.

**`rust` is FASTER than node, and its child is ~25× smaller** (`bridge_impl_throughput_benchmark.py`,
a same-invocation A/B of N parallel env workers on an idle 16-core box): at 8 workers **1.18×**
(1852 → 2182 steps/s), at the production `--n-envs 48` **1.41×** (1942 → 2729 steps/s), with the
bridge child at **9 MB RSS vs node's ~224 MB** — so the children cost ~0.4 GB under rust vs ~10.7 GB
under node at `n_envs=48`. (An older note recording node 798 vs rust 427 fps at 8 envs was measured
on a CPU-saturated box and is superseded.) **`gen3_bridge_forfeit_win_v1` (2026-07-31) was the
blocker that made rust unusable for training**: `FORCELOSE` emitted a bare `__END__` with no `|win|`
line, so poke-env never marked the battle finished and the next `reset()` hung forever. Since the
training seam forfeits whenever `reset()` lands mid-battle, every episode boundary could wedge —
the multi-env soaks logged finished episodes yet completed ZERO PPO iterations. Fixed in
`BridgeSession::forfeit`; the durable gate is `bridge_session_fuzz_test.py --impl rust` (its
every-9th-episode forfeit-reset is the reproducer, and `--impl` exists because that fuzz previously
only ever tested node — the coverage hole that let this ship).

It reuses the *entire* obs/reward/mask/wrapper stack unchanged:

- **Training** — `attach_bridge_transport` (`src/utils/bridge/bridge_session.py`) swaps the two
  `_EnvPlayer` agents' transport for a background-pumped bridge subprocess per env. The child is
  **persistent by default** — one long-lived Node process reused across every episode (a fresh
  `START` rebuilds a clean `BattleStream`), which kills the per-episode Node-spawn cost. A
  single-env latency A/B (`bridge_vs_websocket_latency_benchmark.py`) measured ~13 ms/step
  websocket → ~6 ms/step persistent bridge (~2.1×); spawn-per-battle was only ~11 ms/step, so the
  reuse is the win. **But the 2.1× is single-env transport-only — at production `n_envs=64` the
  measured end-to-end training-FPS gain is just ~5%** (bridge 1192 vs websocket 1140 fps, vs-bots,
  CUDA), because oversubscription hides the per-step transport latency behind the SubprocVecEnv
  barrier (the box is CPU-saturated; n_envs is not the FPS lever — see
  `src/agents/training/CLAUDE.md` throughput notes). The bridge also started ~17% faster (no
  `/challenge` connection storm) and ran steadier. **The case for the bridge is operational, not
  FPS:** no server at all → no RAM-growth leak, no connection storm, no port tuning, deterministic.
- **Eval / self-play eval / final eval** — the eval players (built `start_listening=False`) play
  in-process via `run_local_battles` (the synchronous driver) instead of `battle_against`. Threaded
  as a `use_showdown_bridge` config key through `PerOpponentEvalCallback` / `SelfPlayCallback` →
  `eval_worker`. Each eval worker plays its opponents **one game at a time by default**
  (`--eval-concurrency-per-worker`, default `1`; threaded to `run_local_battles(concurrency=…)` /
  `max_concurrent_battles`). Overlapping battles within an opponent is single-thread asyncio
  latency-hiding, **not multi-core** — it nets negative under training contention (the old default's
  "measured slower"), but ~2× decisions/sec on spare cores (idle box / cycle tail); see
  `src/agents/training/CLAUDE.md` → intra-worker concurrency. Cross-opponent parallelism comes from the
  `--eval-workers` (5) subprocesses work-stealing the pool; this takes all eval load off the server.

**Persistent-child lifecycle (measured + optimized):** a child's RSS is **flat** — ~189 MB fresh
→ one-time ~+36 MB V8 warmup → ~229 MB with ~0 growth over thousands of battles
(`bridge_heap_growth_benchmark.py`). A child plays only ~2150 battles in the launcher's 3h restart
window, so the bridge does **not** reintroduce the server's RAM-creep and needs **no recycle within
3h** — the 3h restart owns the lifecycle. `recycle_every` (default 5000) is a backstop that never
fires under the launcher; it only caps marathon / no-launcher runs. A child that **dies** mid-run
**crashes** the env (no in-place recovery → launcher restart; resuming risks a corrupted PPO
transition).

**The default is `rust` (changed 2026-08-14; it was `off` = websocket).** The case was never
throughput — at production `n_envs` the end-to-end FPS gain over websocket is ~5% — it is
**operational**: no server to start, no port to tune, no connection storm, no RAM-creep, and
deterministic delivery. Those hold on every run, whereas the websocket path's costs land exactly
when a run is long. `rust` over `node` is the measured half (1.41x at `--n-envs 48`, a ~25x smaller
child); `node` stays a first-class explicit value because the parity harness and the A/B arm need
it, and `off` stays for the ladder. **The launcher agrees**: `child_uses_bridge` treats an ABSENT
`--use-bridge` as a bridge run, so it no longer injects a phantom `--showdown-port` (pinned by
`default_port_test.py` — a drift between the two defaults is what that file now catches). See
`src/utils/bridge/README.md` and `designs/ai_v5/design_local_sim_bridge_transport.md`.
