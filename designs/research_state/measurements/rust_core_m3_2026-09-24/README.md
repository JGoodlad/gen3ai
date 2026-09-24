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

## 2. `CoreError` (`gen3_core_error_v1`)

`Result<T, String>` in `core_events`, `present` and `version` (and the engine's fatal condition)
became `CoreResult<T>`: `Refusal { exc: PyExc }` (poke-env raises the same Python class on the same
input), `Malformed`, `Fault`. Byte identity: the transcript's 20,216,559 bytes, sha256
`4d0e37a6f80dcdd3…`, unchanged (messages convert verbatim at the transport). The parity gate compares
the refusal CLASS (`rust_core_present_test.py::test_refusals_raise_the_same_class`: an unknown
keyword → `UnknownMessageType`, `-mega` → `UnsupportedMessageType`, `|gen|4` → `RuntimeError`,
`|turn|two` → `ValueError`; malformed input never dressed as a poke-env class). Writing the pins
found one misclassification (the event Reader's `int()` sites are poke-env's `ValueError`).

## 3. The trackers on the version — slice T (`gen3_core_trackers_v1`)

**COMMIT** (`rust_core_parity_test.py::test_commit_tier_trackers_equal_episode_tracker`, first at
`28851dc6`, green in every routine gate since, last at `493db48f`): **0 divergences** — 13 in-scope
battles, 26 viewers, 1,838 decisions, 52,674 event-window rows, labels MOVE 943 / SWITCH 487 /
masked 408, 1,838 per-decision rewards + 26 terminal. Teeth: a residual folded into a move's
`hp_delta`, a clock whose PROGRESS clause never fires, a phaze labelled a choice, an opponent's denied
choice leaking into a viewer's record (`[BOUNDARY]`, M3 (3)) — each FAILS.

**MILESTONE** (at `493db48f`, `slow_tier_status.json`): **5 / 5 pass** — slices E + V + T (incl. the
information-boundary check) on 2 × 360 seeded-random and 2 × 50 production-policy battles played
live, plus the protocol and byte-fuzz corpora (19 min 53 s beside the catalogue; the per-battle
slice-T census is the catalogue's, §5).

**Fresh battles** (`rust_core_trackers_fuzz_test.py`, 6-minute runs, pool + mechanic-dense +
procedural teams): at `493db48f`, seed 1430675900 (`catalogue/tracker_fuzz_493db48f.log`), slices
E, V and T **0 divergences** over 176 battles, 39,110 decisions, 1,168,975 event-window rows; the
native record's coverage in that run: 827 denials, 4,900 refused actions, 276 Baton Pass entries,
1,490 drags, 230 multi-faint windows, 68 Pursuits on a switch, faints by Destiny Bond 4 and Perish
Song 14. An earlier run at `1b25086b` (measured pre-rebase as `c9f9c589`, the same tree; 185 battles, 35,660 decisions) was 0 on T and E; its slice V
flagged a procedural Rollout battle (the core's board audit: own Rollout PP read 30, engine 31) — the
port's missing `[from] lockedmove` on lock-in continuations, which the ladder-corpus agent found and
fixed the same hour (`f865e8ab`, `gen3_lockedmove_announce_v1`, not yet on main at this writing); an
independent reproduction, not a poke-env finding.

**The one rule the port had to take from poke-env rather than from the decision window**: the
Hidden-Power belief's input is `battle.opp_last_damaging_move` (pending at the damaging `|move|`,
promoted by the defender's effectiveness line, turn-gated). A window-scoped reading (the rule
`view_successor.view_context._to_dme` still uses for search) diverged from training on the HP belief at
57 of 1,838 COMMIT decisions. The mechanism that makes them differ is a window that does not hold the
whole previous turn (one that opened at a mid-turn forced switch); that each of the 57 is that shape
is **UNVERIFIED** (not broken down). On search, a depth-1 successor's window is its whole ply, so the
difference is reachable at depth ≥ 2.

## 4. The tracker FORK's cost in a searched decision (`fork_cost/`)

**Method.** M2's search benchmark (`search_decision_benchmark.py`, the real `SearchEngine.choose` on
10 decisions rebuilt from BANKED `ai_v12_02_winprob_critic` eval traces, read-only; the pure `obs.sum`
scorer), ONE road per process, interleaved (`fork_cost/run_roads.py`, resumable): `core` (today's
production core road — the successor's trackers are the PYTHON ones, forked by the pinned-pickle
THAW) and `core-trk` (the same road with the Rust core's trackers folded on every version,
`SearchConfig.core_trackers`). Frozen release binaries (`fork_cost/bins.sha256`). Two widths: **wide**
(honest arm, m_opp 3, k_worlds 4 → 684 successors over 10 decisions) and **B = 1** (10). 8 pairs each.

| | wide (load1 25.8–40.5) | B = 1 (load1 29.9–43.3) |
|---|---|---|
| decision wall, `core` road | 321 ms [261, 450] | 72 ms [64, 84] |
| **BEFORE — Python THAW / successor** | **0.46 ms** [0.40, 0.62] | 0.62 ms [0.56, 1.36] |
| **BEFORE — the thaw's share of the searched decision** | **9.9 %** [9.3, 10.3] | 0.8 % [0.7, 1.8] |
| BEFORE — thaw + Python `record_context` + `advance_window` / successor † | 0.69 ms [0.58, 0.82] | 3.9 ms [3.1, 4.5] |
| **AFTER — the Rust tracker fold / successor** (fork + record + advance, fold + render delta) | **0.089 ms** [0.027, 0.125] | 0.066 ms [0.047, 0.076] |
| **AFTER — its share of the searched decision** | **1.8 %** [0.7, 2.4] | 0.08 % [0.07, 0.10] |
| AFTER / the thaw alone, per successor | **0.17** [0.07, 0.30] | 0.10 [0.06, 0.13] |

† the Python-trackers phase also counts the ROOT prefix replay's own `record_context` /
`advance_window` calls (which dominate at B = 1); the thaw alone is purely per successor.

**Reading.** The Rust fork of the tracker state is an `Arc` clone (a pointer); the 0.089 ms is the
successor's own `record_context` + `advance_window` fold, copy-on-write — ≈ 1/6 of the thaw ALONE,
and the thaw is ≈ 10 % of a wide searched decision here (the program's 16.9 % was an earlier profile
on a different configuration). **Search does not get this yet**: its successors are still encoded by
the Python encoder, which reads the Python trackers, so production search still thaws. M4's encoder
reading the version's trackers is what removes the thaw; `core_trackers` stays OFF until then (it
would add the Rust fold without removing the Python one).

```bash
python fork_cost/run_roads.py --bin-dir <frozen release bins> --src <tree>/src --traces <eval_traces> --label wide --pairs 8 -- --decisions 10 --m-opp 3 --arm honest --k-worlds 4
python fork_cost/run_roads.py … --label b1 --pairs 8 -- --decisions 10 --n-actions 1 --m-opp 1 --k-worlds 1 --arm honest
python fork_cost/analyze.py fork_cost/rows.jsonl
```

## 5. The loss catalogue (`catalogue/`)

**In progress** — the MILESTONE-corpus run is detached and resumable (`catalogue/run.log`); this
section lands with its verdict at the registered n (820 battles).
