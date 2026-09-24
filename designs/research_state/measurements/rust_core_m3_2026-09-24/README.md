# Rust Core M3 — the engine/transport split, `CoreError`, the trackers on the version, slice T

<!-- A MEASUREMENT record (2026-09-24) for the Rust Core Program's M3
(designs/endstate/program_rust_core.md §2 M3 and its "Before M3 starts" hand-off). Code:
designs/rust_sim/present.md (the version + engine), designs/rust_sim/trackers.md (the trackers). -->

Each section is one question, answered by an interleaved A/B with one binary per process, the
1-minute load average recorded at every run's start and end, `nice -n 10`, CPU only, no port, no
`models/` write, no `data/` change. **Ratios are the claim** (the box carries a production training
arm); the statistic is the per-pair ratio's median with a paired bootstrap 95 % CI on that median
(20,000 resamples, seed 0).

## 1. The engine/transport split (`gen3_core_engine_split_v1`) — byte identity and the fork's clone cost

**Byte identity.** `sim_bridge` fed the M1 record's 41-battle transcript
(`../m1_transport_throughput_2026-09-23/transcript_env_seed0.txt`, 6,078 `CHOOSE`s) emits the SAME
**20,216,559 bytes** (sha256 `4d0e37a6f80dcdd3…`, 41 `__END__`, 0 `__ERR__`) before the split
(`efd3ee78`) and after it — the digest M2's record and the emission self-check recorded. `cargo test`
green (854 tests), including `bridge_test`'s incremental-vs-genesis parity and the new
`tests/engine_split_test.rs`.

**The question** the program left UNMEASURED: what does a fork pay for the transport's history
(`script` + `request_seeds`, both growing with battle length)? `fork_clone/` —
`examples/fork_clone_bench.rs` replays the same transcript through ONE session and, at every one of
the 3,155 decision boundaries, times each clone 101× (median kept per boundary):

| arm | what | build |
|---|---|---|
| before `snapshot` | `BridgeSession::snapshot()` of a `clear_chunks`ed session — what a version fork cloned pre-split | `efd3ee78` |
| before `script_seeds` | `script` + `request_seeds` alone | `efd3ee78` |
| after `engine` | `Engine::clone` | this change |
| after `resume` | `BridgeSession::resume(engine.clone())` — the whole fork a version now drives (`fork_session`) | this change |

**Result** — 8 interleaved pairs (order alternating), load1 7.3–31.1 (`fork_clone/analysis.txt`,
raw `fork_clone/rows.jsonl`):

| ratio | median [95 % CI] | slower in |
|---|---|---|
| **after `resume` / before `snapshot`** (the fork, before vs after) | **0.950** [0.921, 0.952] | 0 / 8 |
| after `engine` / before `snapshot` | 0.947 [0.917, 0.952] | 0 / 8 |
| before `script_seeds` / before `snapshot` (the transport history's share of a fork) | **0.039** [0.038, 0.039] | — |

Medians of the per-boundary medians: before snapshot 18.7 µs, script + seeds 0.73 µs (mean script
length 55.7 decisions), after engine clone 17.5 µs, after resume 17.6 µs.

**Reading.** The split takes the transport's history OFF the fork — 3.9 % of it at this corpus's
mean battle length, and a share that grows with the battle — and the fork is 5 % cheaper. The other
96 % is the ENGINE: the `Battle` (its protocol log and, on a core session, its typed source records,
both also growing with battle length) and the driver's decision records. That is the next fork-cost
term, and it is not the transport's (a FINDING for M5, which owns the in-process env shape).

**A superseded first build (kept: `fork_clone/rows_rerender_superseded.jsonl`).** The first split
rendered the fresh transport's two outstanding `|request|` JSONs from the engine's typed requests at
every fork: **2.17×** [2.02, 2.81] the pre-split snapshot (≈ 27 µs of rendering). The landed engine
keeps, beside each typed request, the bytes it was ISSUED as (an `Arc<str>`, rendered once at issue
and shared by every fork) — so a fork's transport re-renders nothing, and `Engine::request_json` (a
fresh render) is pinned equal to them at every boundary by `tests/engine_split_test.rs`.

```bash
cargo build --release --example fork_clone_bench          # at each commit, copied out
fork_clone/run_pairs.sh <before> <after> <transcript> 8 fork_clone/rows.jsonl   # resumable
python fork_clone/analyze.py fork_clone/rows.jsonl
```
