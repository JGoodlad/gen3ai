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

