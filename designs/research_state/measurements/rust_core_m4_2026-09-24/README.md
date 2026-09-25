# Rust Core M4 — the encoder: slice O, the obs golden, the benchmark's core row

<!-- A MEASUREMENT record (2026-09-24) for the Rust Core Program's M4
(designs/endstate/program_rust_core.md §2 M4). Code: designs/rust_sim/encoder.md. -->

Box: 16 cores shared with a production training arm and other agents; every run `nice -n 10`, CPU
only, no port, no `models/` write, no `data/` change.

## 1. Slice O — COMMIT tier

`python3 -m pytest src/agents/battle/rust_core_parity_test.py -q` (the self-check build, NaN-prefilled
rows): **2,081 decisions, 15 in-scope battles, 30 viewers, 2,081 / 2,081 rows BYTE-equal, 0
divergences**, the mask equal at every decision; every obs block nonzero somewhere (our team, opp
team, active context 558 decisions, global, board, pair history, event window 2,051). Slice T on the
same pass: 0 divergences.

**The obs golden** (`training/golden_obs_fixture.json`): the core's rows for the trainee over the
golden's fixed battle set hash EXACTLY to the committed fixture and equal the Python capture's vectors
byte for byte (`test_the_obs_golden_is_reproduced_by_the_core`).

## 2. The benchmark's core row

`python src/agents/training/obs_build_benchmark.py --turn 25 --reps 400` (load1 17.4 at start; the
release `core_events`, zero-prefilled), one decision at turn 25, the core row asserted byte-equal to
the Python row:

| series | ms / decision |
|---|---|
| Python, full rebuild COLD | 0.606 |
| Python, production shape (assembler warm + view memo warm) | 0.157 |
| **core, encode (view memoized)** | **0.0257** (median 0.0255) |
| **core, `present()` + encode (cold)** | **0.0374** (median 0.0362) |

One decision, one run on a loaded box: ratios, not absolutes, are the claim (the core encode is ≈ 6×
the Python production shape and ≈ 16× the cold rebuild on this decision), UNVERIFIED across decisions.
