# `--impl rust` — what is pinned, and what it buys

Owned by this tree. The leaf keeps the flag, its placement, its inertness on model-free commands,
the session-wide default and the never-falls-back-to-node rule. This file holds the equivalence
pins and the measured cost.

**The rust binary is BUILT and gated** (`gen3_rust_search_driver_v1` + `gen3_rust_replay_driver_v1`,
over the `gen3_bridge_clone_branch_v1` snapshot primitive), so `--impl rust` works today. Equivalence
is pinned node-vs-rust at 18873 + 30689 leaf fields, and — the claim that matters for a probe — the
cross-impl `better_line_integration_test` asserts node and rust yield IDENTICAL candidate V; since
that fake model is `V = obs.sum()`, an exact match is an obs-level bit-identity claim at every ply of
the beam. Two known divergences are printed by the parity harnesses rather than hidden: the
choice-reject framing (no `|error|` frame, boundary re-opens to both sides) and reconstructed
`pre_state` volatile names. **Perf: per-op the rust driver is 7–20× (clone-and-branch 13–20×), but
end-to-end `better_line` is only ~1.9×** — child-wait falls from 51% to 4% of a call, so Python-side
obs materialization is now the bottleneck and it is impl-invariant. See
`src/utils/bridge/README.md` → Offline driver transport.

## From the prober leaf (moved 2026-10-10)

Moved verbatim-ish out of `src/main/prober/CLAUDE.md` when that leaf was cut to rules, commands and the map.

### `--compile` (search-shaped commands)

`python -m main.prober.query --compile <cmd> …` `torch.compile`s the no-grad replay/rollout models
that `session._load` builds (`ProbeSession(..., compile_extractor=True)`), for a measured **~6.5×** per
B=1 CPU forward at a ~10-20 s one-time cost.

**Off by default, and use it selectively.** A one-off `summary` / `list` / `analyze` does a handful of
forwards and would never amortize the compile. It pays for the SEARCH-shaped commands, which do
thousands: `better-line` (a CRN-anchored beam), `falsify` / `falsify-scan` (paired alternative-action
sweeps × seeds), `replay-counterfactual` (Monte-Carlo re-rolls to a win/loss), `lookahead`.

**Gradient saliency is unaffected.** `history-saliency` and the gradient paths backprop through this
same extractor, and the compiled artifact is inference-only (AOTAutograd's CPU backward codegen fails
on the model's scatter/`index_add`). `maybe_compile_extractor`'s wrapper routes any **grad-enabled**
call to the eager forward, so `--compile` cannot change or break a saliency result — it simply does
not apply there. Detail: `designs/training/compile_flags.md` → Compiled CPU opponents.

### `--impl {node,rust}` (which sim engine the search/replay children run)

`python -m main.prober.query --impl rust <cmd> …` — the **offline analogue of the (now fixed)
training transport**, and like `--compile` it is a global flag placed BEFORE the subcommand.
Default `node` = today's behavior byte-for-byte.

It picks the child process the re-roll-backed probes exec — `better-line` / `lookahead` / `falsify`
/ `falsify-scan` / `calibration`. The model-free, no-replay commands
(`summary`, `list`, `scan`, `triage`, `overview`, `find`, `analyze`, `probe`, `decision-table`)
spawn no sim child, so the flag is inert for them. Under `node` the work is split across
`search_driver.js` (the clone-and-branch server) and `replay_driver.js` (replay / reroll); under
`rust` a single `src/rust_sim` `search_driver` binary serves both — resolved (and built, once) by
`utils/bridge/sim_bridge_bin.resolve_search_driver_bin`, overridable with
`$POKESIM_SEARCH_DRIVER_BIN`. `replay-counterfactual` spawns no child at all since P6 — its play-out
runs in process on the Rust core whatever `--impl` says, and an `--impl node` read says so in its
`caveats` (`engine: rust_core`). **It NEVER falls back to node** — an unbuildable binary is a clear error, because
a "rust" probe that silently ran on node would answer a different question than the one asked.

**The default lives on the SESSION, not the call**: `ProbeSession(root, …, impl="node")` stores it
and every probe reads it — the same shape as `compile_extractor`, and deliberate, since two probes
of one run answering under different engines would not be comparable. `better_line` REFUSES an
injected warm `SearchSession` whose `impl` differs from the session's (a caller's reuse
path), so a correction can't be half-searched on one engine and half-confirmed on the other.


Equivalence is pinned node-vs-rust at 18873 + 30689 leaf fields, and the cross-impl
`better_line_integration_test` asserts node and rust yield IDENTICAL candidate V — an obs-level
bit-identity claim at every ply of the beam. Per-op the rust driver is 7–20×; end-to-end
`better_line` is only ~1.9×, because Python-side obs materialization is the bottleneck and it is
impl-invariant. The two known divergences and the build/override path: `designs/prober/sim_impl.md`.
