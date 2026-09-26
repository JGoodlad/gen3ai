# The observation-architecture batch (`gen3_event_record_v2`, 2026-09-26) — the landing evidence

The owner's 2026-09-25 batch, one deliberate retrain boundary: the Mud Sport / Water Sport volatile
slots (`gen3_field_sport_slots_v1`), E4 (the refused-switch target), the parameter-free E10 hidden-slot
move mixture (`gen3_hidden_slot_move_mixture_v1`) and E12 (the event-row reshape). Obs 2501 → 2761,
`MODEL_CONFIG_VERSION` 121, `ARCH_SIGNATURE` `gen3_event_record_v2`, `MIGRATION_FLOOR` 121. Schema and
semantics: `designs/ARCHITECTURE.md` §1.5, §1.6, §2 (MoveBelief), §4, §5.

## 1. The golden obs change is CONFINED to what the batch intends (`golden_confinement_census.py`)

`golden_confinement_census.txt`: the base capture (56837827) reproduces the committed fixture 991/991;
decision count unchanged at 991; mapping HEAD back to the old layout reproduces every NON-event cell
bit-for-bit, the four field-sport columns are 0 in every decision; 500 / 991 decisions' event windows
differ in the old 22 columns, every one explained by the batch's intended changes alone (inserted
DENIED rows, a switch-in's entry chip and causing move, the RECEIVED item direction, the
destinybond / perishsong causes, the E4 target) — 0 unexplained. The fixture was regenerated after.

## 2. `obs_build_benchmark.py`, before (56837827) and after — NOT a clean A/B

`obs_build_benchmark_before_56837827.txt`, `obs_build_benchmark_after.txt`, plus an interleaved
base/batch/base/batch rerun (`--seed 0`, nice 10, a training run sharing the box):

| | base run 1 / 2 | batch run 1 / 2 |
|---|---|---|
| Python production shape (assembler WARM + view memo WARM) | 0.140 / 0.145 ms | 0.150 / 0.140 ms |
| Python COLD full build | 0.408 / 0.468 ms | 0.511 / 0.508 ms |
| CORE row, encode (view memoized) — what training runs | 0.0086 / 0.0093 ms | 0.0052 / 0.0051 ms |
| CORE row, present() + encode (COLD) | 0.0135 / 0.0088 ms | 0.0085 / 0.0084 ms |

⚠️ **The benchmark's decision is not reproducible run to run** (the same tree reported "opp mons w/
revealed moves 4/6" in one run and 5/6 in the next, `--seed 0` both times), so these are two samples
of DIFFERENT boards, and the box carried a live training run. What it supports: the production
Python shape is flat within noise (+0 to +7 %), the COLD Python build is up to ~+15 % (the wider event
rows are written in full), and the Rust core row — the training path since `--obs-source core` —
stays at ~5–9 µs, i.e. ≤ 0.3 % of a ~3–4 ms `Gen3Env.step` (the M6 bar is +3 % per decision). NOT
measured: the Rust tracker fold's added cost (the denial bookkeeping runs in `decide`, outside the
encode bench) and the model-side cost (the wider `EventSeats` projection, the third `r` cell, E10's
one `[B, S] @ [S, M]` matmul per forward).

## 3. The per-mechanic fixtures

`src/agents/battle/event_record_v2_fixture_test.py` — 15 tests, every battle run through the core
and the Python path with slices T and O on (0 divergences): denial fainted-first, turn cut by a
self-KO Explosion and by recoil, Destiny Bond and Perish Song trade KOs (cause + order), Roar into
Spikes, Baton Pass into Spikes, Thief / Trick / Knock Off, a Spikes sack and its free replacement,
Pursuit on a switching target, called moves, E4 under Arena Trap, and the row writer's columns. The
E4 fixture fails on revert (verified: slice T diverges on `.window.rows[0].target`).

## 4. E10 — the hidden-slot move PRIOR, flat sentinel row vs the parameter-free mixture (`e10_recall.py`)

Team-level, no battles: for 150 teams per source (every other key 0–298), reveal the first k species
(k = 1..5) and score each hidden mon's true moves against the top-4 of each prior (`e10_recall.txt`):

| source | hidden slots | recall@4, flat row | recall@4, mixture |
|---|---|---|---|
| pool | 2,250 | 0.000 | 0.325 |
| ladder (Metamon hl_05_26) | 2,250 | 0.000 | 0.295 |
| procedural | 2,250 | 0.000 | 0.265 |

This is the PRIOR alone, before the model's learned head delta (the 2026-09-24 calibration read's
0.10 for the constant posterior includes that delta); the flat row's top-4 is four arbitrary
floor-level moves, hence 0. It is the input the new lineage starts from, not a trained result.
