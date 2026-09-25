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

## 1b. Slice O — MILESTONE tier and fresh battles

**MILESTONE** (`python3 -m pytest src/agents/battle/rust_core_parity_test.py -m slow -q -n 2 -rP` at
`34da7225`, 13 min 32 s, contention 1.55; 10 / 10 pass, recorded in `designs/ops/slow_tier_status.json`).
Slice O per team source, every decision of both viewers, every row BYTE-equal, the mask and every choice
token equal, 0 divergences:

| source | battles | decisions | choice tokens |
|---|---|---|---|
| pool, seeded-random (keys 0–359 even / odd) | 2 × 360 | 64,991 + 67,022 = 132,013 | 763,757 |
| pool, `production` policy | 2 × 50 | 6,446 + 3,951 = 10,397 | 70,399 |
| LADDER-USAGE (Metamon `hl_05_26`, keys 0–299 minus the 3 named) | 150 + 147 | 25,975 + 25,620 = 51,595 | 316,591 |
| LADDER, the 3 named slice-V known divergences (slice O clean on them) | 3 | 299 | 2,040 |
| **total** | **1,020** | **194,304** | **1,152,787** |

Slices E, V and T clean on the same battles (V's three named ladder divergences still fire exactly).
The protocol + byte-fuzz corpora run slice E only (no slice O there).

**Fresh battles, procedural-heavy** (`rust_core_trackers_fuzz_test.py --minutes 8 --procedural 0.5`,
the self-check build, at `34da7225`; log `fuzz/tracker_obs_fuzz_34da7225.log`): **slice O 0 divergences
over 376 battles, 752 viewers, 77,475 decisions (449,894 choice tokens)** on pool, mechanic-dense and
PROCEDURAL teams; slice T 0. Slice V flagged 2 decisions of one pool battle — a pre-existing READING
finding, not the encoder's (§3).

**The FINAL tier, on the new training-input boundary** (`b6dfd7e8` — the M3 GIGO fix
`b9d74f65` rebased on M4, its event-window change mirrored in the encoder, the obs golden
regenerated): MILESTONE **10 / 10** (12 min 23 s; rows re-recorded in `slow_tier_status.json`), slice O
0 divergences on **194,606 decisions** — pool random 64,991 + 67,022, policy 6,748 + 3,951 (the policy's
own trajectories moved with its inputs), ladder 25,975 + 25,620, the 3 named ladder battles 299 — every
row byte-equal, **1,154,149** choice tokens equal. COMMIT tier and the regenerated obs golden green in
the routine gate. Fresh battles (`fuzz/tracker_obs_fuzz_b6dfd7e8.log`, seed 1451299227, 6 min, 50 %
procedural): slice O **0 over 352 battles, 66,818 decisions** (402,703 tokens), slice T 0. Its slice V
flagged one procedural battle with a TRANSFORMED own Smeargle (`procedural_50914`, turn 23): the view
road's opp PP / ability presentation (V3 / V8, the view road only) and the reading's `stats` vs the
engine's (`[BOARD] ours.stats`) — a pre-existing Transform (V14) class, `stats` not in the obs; which
side is right about a transformed mon's request stats is UNVERIFIED.

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

Three more decisions (`--seed 1/2/3`, load1 29–32, each row byte-asserted):

| seed | Python cold | Python production shape | core encode | core `present()` + encode |
|---|---|---|---|---|
| 1 | 0.557 | 0.159 | 0.0138 | 0.0195 |
| 2 | 0.418 | 0.116 | 0.0150 | 0.0213 |
| 3 | 0.404 | 0.126 | 0.0157 | 0.0231 |

The core encode is **8–12×** cheaper than the Python production shape and **18–29×** cheaper than the
cold rebuild on these four decisions (the seed-0 run's 6× was at its first, colder run). One decision per
run, a loaded box: the ratio band is the claim, not the absolutes; a search-decision wall A/B (rows vs
the pre-M4 Python successor) was NOT measured (it needs the banked `models/` traces; UNVERIFIED).

## 3. Findings

* **A Sleep Talk-called move charges Sleep Talk TWO PP in poke-env's reading** (and in the core's
  `present()`, which mirrors it); the sim charges one. Reproduction: `rust_core_parity.play(26085)`
  (pool, seeded-random), p2's Suicune, turn 101: `|cant|p2a: Suicune|slp` → `|move|p2a: Suicune|Sleep
  Talk|p2a: Suicune` → `|move|p2a: Suicune|Surf|p1a: Suicune|[from] Sleep Talk` into p1's Suicune
  (Pressure), then dragged out by Roar: poke-env's Sleep Talk PP 7 → 5, while the sim's next request
  (turn 104) says 6. The board audit catches it (`[BOARD] ours.[V15] moves[pp]`: view 5, engine 6 —
  V15 allows only a LAG). The mechanism (a Pressure charge applied to the caller for the called move)
  is **UNVERIFIED**. It reaches the obs (an own BENCHED mon's move PP until the next request resyncs
  it); the encoder reproduces it by design. Not fixed: a training-input change, the owner's call.
