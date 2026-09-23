# RUST CORE PROGRAM — M1 measurements: the parity census, the byte gates, the cost of `parse`

<!-- A MEASUREMENT record (2026-09-23, branch `rust-core-m1`). Plan: `designs/endstate/program_rust_core.md`
(M1). Contract: `designs/rust_sim/core_events.md`. Do not rewrite the tables — a later milestone
appends its own. -->

**Box.** 16 cpus, a live training arm owning the GPU and most of the CPU; load average **14-24**
across this session. Everything ran on CPU at `nice 10`. Timings are reported with their load.

## 1. The parity census (slice E) — 0 residual, 0 refused

Per viewer, per event, `seq · turn · kind · side · actor · target · value · raw` of the core's
reading against a `Gen3Battle` fed the same per-side text; TYPE-strict; no allowlist
(`src/agents/battle/rust_core_parity_test.py`). Milestone verdicts are banked in
`designs/ops/slow_tier_status.json` at `b5dfc0a3`.

| tier / corpus | battles | viewers | per-side lines | events compared | divergences |
|---|---|---|---|---|---|
| COMMIT (8 recorded + 6 shape fixtures + 1 per protocol scenario + Forecast sweep + Ditto) | 45 | 90 | 33,524 | 16,980 | **0** |
| MILESTONE random, keys 0-199 | 200 | 400 | 478,408 | 249,088 | **0** |
| MILESTONE random, keys 5000-5199 | 200 | 400 | 464,622 | 239,720 | **0** |
| MILESTONE `production` policy, keys 100-149 | 50 | 100 | 83,681 | 43,823 | **0** |
| MILESTONE `production` policy, keys 6000-6049 | 50 | 100 | 52,079 | 26,171 | **0** |
| MILESTONE protocol corpus × 2 + all 77 byte-fuzz fixtures | 122 | 244 | 104,048 | 50,217 | **0** |
| **MILESTONE total** | **622** | **1,244** | **1,182,838** | **609,019** | **0** |

In every played milestone battle the LIVE players' `Gen3Battle` logs also equal the offline
feed's (the reference's own faithfulness). Keys 0-199 include the Phase-0 battles that reach
the turn limit (now a tie). Log: `logs/milestone_census.log`.

The Rust-internal gate (`cargo test`, `tests/core_events_test.rs`) on the protocol capture
corpus, the 77 byte-fuzz fixtures, the trapping golden and the turn-limit golden (220 battles):
every source record canonical and one per omniscient line; each side's step path re-derives the
shipped bytes with per-side conservation; `parse(side text) == step` on every line. 17 of the 138
protocol-corpus battles are TRUNCATED where the capture's blind script re-sent a rejected choice
past the bridge's 8-reject no-progress cap; every line before is checked.

## 2. Production bytes after the typed builder

| rung | seeds | result |
|---|---|---|
| `cargo test` (protocol capture 132 battles / 19,348 lines, `write_line` 44 / 2,377, bridge goldens, byte-fuzz corpus 77, e2e capstone, every pin) | — | 791 passed / 0 failed / 4 ignored |
| `ab_fuzz --mode pool --protocol --format gen3ou`, 150 battles | 4417, 88123 | GREEN — 149 ok + 1 standing A1 residual each |
| `bridge_ab_fuzz --mode pool`, 100 battles | 4417, 88123 | 100 / 100, 0 diverged each |
| `gen_sim_bridge_diff --mode pool --format gen3ou --persistent`, 60 battles | 4417, 88123 | 60 / 60, 0 diverged, `drain_timeouts` 0 |
| `ab_fuzz --mode random --protocol --format gen3customgame`, 150 battles | 5501, 91337 | 44 + 51 divergences (seed / protocol / boost / state) — **all 112 repros replay to BYTE-IDENTICAL `ab_replay` output on the pre-M1 binary** (`dfab2558`): pre-existing modeled-universe gaps, not M1 |

## 3. The cost of `parse` per decision, as a ratio to one env step

`parse_cost.py` (interleaved rounds, one process; `logs/parse_cost_r1.log`, `_r2.log`): the Rust
`Line::parse` + reading fold of ONE side's stream read incrementally per decision (**12.5 lines per
decision**, the `|request|` included), in-process, no IPC — against a `Gen3Env.step` wall on the
production rust bridge with a RandomPlayer opponent and random legal actions (NO policy forward,
NO PPO batch — both would lengthen the step, so the training ratio is smaller still).

| run | load | `parse` µs / decision (median of 5) | of which `\|request\|` | env step ms (median) | ratio (median; range) |
|---|---|---|---|---|---|
| r1 (seed 0) | 21 | 21.6 | 11-16 µs | 2.83 | **0.0076** (0.0057-0.0091) |
| r2 (seed 1, fuzzers running) | 19-21 | 30.8 | 18-20 µs | 4.63 | **0.0066** (0.0057-0.0071) |

**≈ 0.6-0.9 % of one random-action env step**; against the live arm's measured wall of ~108 ms per
env step (445 steps/s over 48 envs, Phase 0) it is ~0.02-0.03 %. The `|request|` JSON is ~60-70 %
of the cost (the only line whose content is a document); an FFI/IPC boundary, which a Python
training loop would add, is NOT included (M5 decides the binding).
