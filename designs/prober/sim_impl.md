# The prober's sim engine — one engine (the Rust `search_driver`), and `--compile`

Owned by this tree. There is ONE offline engine: the re-roll-backed probes run on the Rust
`search_driver` binary, always. This file holds what that engine is, what pins it, and the
`--compile` flag.

## One engine: the Rust `search_driver` binary

The re-roll-backed probes — `better-line` / `lookahead` / `falsify` / `falsify-scan` /
`calibration` — exec the `src/rust_sim` `search_driver` binary (`gen3_rust_search_driver_v1` +
`gen3_rust_replay_driver_v1`, over the `gen3_bridge_clone_branch_v1` snapshot primitive). One binary
serves both verb families: the warm clone-and-branch search (`utils/bridge/search_session.py`'s
`open_root` / `expand_many`) and the one-shot replay / reroll / reroll_many
(`utils/bridge/reconstruction.py`). It is resolved (and built, once) by
`utils/bridge/sim_bridge_bin.resolve_search_driver_bin`, overridable with
`$POKESIM_SEARCH_DRIVER_BIN`; an unbuildable binary is a clear error (`SimBridgeBinaryError`), never
a fallback. The model-free, no-replay commands (`summary`, `list`, `scan`, `triage`, `overview`,
`find`, `analyze`, `probe`, `decision-table`) spawn no sim child at all. `replay-counterfactual`
spawns no child either since P6 — its play-out runs in process on the Rust core (`engine:
rust_core` in its result).

**There is no engine selector.** The Node drivers (`search_driver.js`, `replay_driver.js`,
`replay_kernels.js`) and the `--impl {node,rust}` selector were DELETED 2026-10-10 (owner: "delete
node"). A typed `--impl` on `python -m main.prober.query`, `python -m main.prober.web` (and
`python -m main.prober`, which forwards to it) or `python -m main.harvest` is REFUSED at parse time
with its reason (`sim_bridge_bin.refuse_search_impl_flag` / `SEARCH_IMPL_DELETED_REASON`). The web
app's `/api/health` still reports `impl`, always `"rust"`. `ProbeSession(...)`, `SearchSession(...)`,
`better_line_decision`, `lookahead_decision` / `_battle`, `falsify_decision` / `_battle`,
`replay_counterfactual_battle` and `core_trace.expand` / `protocol_log` take no `impl` argument.

**What pins the engine now.** The cargo tests `src/rust_sim/tests/search_driver_test.rs` and
`tests/replay_driver_test.rs` are the node-free gates on the binary's wire behaviour
(`designs/rust_sim/search_and_replay_drivers.md`), and `src/node_free_search_integration_test.py`
RUNS replay / reroll / reroll_many, a warm `SearchSession`, `better_line_decision`,
`lookahead_decision`, `team_details` and `query falsify` / `falsify-scan` / `calibration` with Node
impossible (`utils/node_blocker.py`). The node-vs-rust differentials that once pinned equivalence
(18873 + 30689 leaf fields; `search_impl_parity.py` / `replay_impl_parity.py`) are deleted with the
Node drivers — their golden's generators had already gone in P6 slice 6c, so they had nothing left
to run on.

**Perf (measured while both engines existed):** per-op the Rust driver was 7–20× the Node one
(clone-and-branch 13–20×), but end-to-end `better_line` only ~1.9× — child-wait fell from 51% to 4%
of a call, so Python-side obs materialization is the bottleneck. See `src/utils/bridge/README.md` →
Offline driver transport.

## `--compile` (search-shaped commands)

`python -m main.prober.query --compile <cmd> …` `torch.compile`s the no-grad replay/rollout models
that `session._load` builds (`ProbeSession(..., compile_extractor=True)`), for a measured **~6.5×** per
B=1 CPU forward at a ~10-20 s one-time cost. It is a global flag placed BEFORE the subcommand.

**Off by default, and use it selectively.** A one-off `summary` / `list` / `analyze` does a handful of
forwards and would never amortize the compile. It pays for the SEARCH-shaped commands, which do
thousands: `better-line` (a CRN-anchored beam), `falsify` / `falsify-scan` (paired alternative-action
sweeps × seeds), `replay-counterfactual` (Monte-Carlo re-rolls to a win/loss), `lookahead`.

**Gradient saliency is unaffected.** `history-saliency` and the gradient paths backprop through this
same extractor, and the compiled artifact is inference-only (AOTAutograd's CPU backward codegen fails
on the model's scatter/`index_add`). `maybe_compile_extractor`'s wrapper routes any **grad-enabled**
call to the eager forward, so `--compile` cannot change or break a saliency result — it simply does
not apply there. Detail: `designs/training/compile_flags.md` → Compiled CPU opponents.
