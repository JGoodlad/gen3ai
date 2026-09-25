"""The Rust Core CUTOVER stress (`gen3_core_cutover_stress_v1`, program M6 — `designs/endstate/
program_rust_core.md` §3, the CUTOVER tier).

A detached, INCREMENTAL, resumable driver (`designs/ops/ORCHESTRATOR_SOP.md` §2) that runs the
cutover tier's COVERAGE-and-COUNT targets on a box it SHARES with the training queue (owner option
2, 2026-09-24): every unit is minutes long, runs in its own process at `nice 19`, and writes ONE
durable row atomically; a restarted driver skips every unit on disk.

* :mod:`.plan` — the PRE-REGISTERED streams, their units and their targets (the numbers §3 states).
* :mod:`.units` — one unit of each kind: `parity` (slices E/V/T/O through
  `rust_core_parity.play(key, …)` + `check_battles`, both viewers, every decision), `corpora`
  (the protocol + byte-fuzz corpora, slice E), `fuzz` (one run of a Rust-vs-Node A/B fuzzer),
  `soak` (one long-lived training-transport bridge child, RSS sampled over its life).
* :mod:`.governor` — the live training arm's MARGINAL fps, read from its launcher child log
  (read-only), the pre-registered throttle, and the stress-OFF control windows.
* :mod:`.driver` — the scheduler (`run`, `status`, `unit`).

Run it from a PIN — a `git archive` export of one commit with its own `target/` (a binary compiled
in a worktree panics once the worktree is removed: its dex path is compile-time), never from a
worktree:

    python -m main.rust_core_cutover run --out ~/gen3ai_archive/cutover_stress_2026-09-24 --detach
    python -m main.rust_core_cutover status --out ~/gen3ai_archive/cutover_stress_2026-09-24
"""
