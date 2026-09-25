# The training-input GIGO fixes — the Rust core M3 loss catalogue's findings, fixed on both paths

<!-- A MEASUREMENT record (2026-09-24) for the TRAINING-INPUT change that fixed the event window
(W1–W5), the α/β intent label (L1–L5) and the progress clock's clause (i) (T1 / T2) found by
`../rust_core_m3_2026-09-24/` §5. Code + contract: designs/rust_sim/trackers.md §5,
designs/ARCHITECTURE.md §1.1 / §1.6. -->

Measured on the fix tree against its base `2eb5850a`, CPU only, `nice -n 10`, no port, no `models/`
write, no `data/` change, on a busy box (load1 5–32). The fix commit and the commit the MILESTONE
rows name are in the ledger paragraph and `designs/ops/slow_tier_status.json`.

| # | question | verdict |
|---|---|---|
| 1 | is every catalogued GIGO fixed BY CLASS, on the Python path training reads AND in the Rust core? | **YES** — three rules, below; slice T **0 divergences** at COMMIT and at MILESTONE (incl. both ladder tiers) |
| 2 | does every fix have a pin that FAILS on revert? | **YES** — Python: 13 / 14 unit pins and 11 / 11 constructed-battle pins fail on the base tree (the 14th pins the clock branch the TurnView fix feeds); Rust: each of the 11 fixes reverted alone fails its pin (`rust_revert_pins.txt`) |
| 3 | which obs cells moved, and only those? | golden: **698 / 991** decisions, ONLY the five event-window fields and the clock scalar, every cell resolved to its row and its event (`golden_census.txt`); the regenerated fixture == the censused FIX capture, the BASE capture == the committed fixture, 991 / 991 |
| 4 | which labels / rows / clock values moved, and at what rate? | 239 battles, 33,766 viewer decisions (`corpus_census.txt`): every change explained; rates agree with M3's catalogue |
| 5 | obs-build cost | **no measurable change** — production shape 0.986× [0.961, 1.030] |

## 1. The fixes

| rule | id | the class | the fix |
|---|---|---|---|
| `gen3_event_window_semantics_fixes_v1` | W1 | a stat DROP stored as a RISE (`update` negated an already-signed `amount`) | the BOOST row's magnitude is the signed amount |
| | W2 | a faint no damage line caused (Destiny Bond, Perish Song) read `attack` ("no `[from]` ⇒ attack") | a faint whose last damage since entry is KNOWN non-lethal, or absent, is `other` (`turn_view.damage_is_lethal`; an absent `hp_after` — the key is optional — keeps the classic reading) |
| | W3 | Trick's two `|-item|` lines read REVEALED ×2 | an item line `[from]` Trick / Thief / Covet / Switcheroo is SWAPPED on BOTH mons (the Thief taker's `|-item|` too — same class) |
| | W4 | a Protect / Detect block read OUT_HIT | `|-activate|<protector>|move: Protect` fails the MOVING side's open move (`turn_view.is_protect_block`; Endure is not a block) |
| | W5 | a Rapid Spin clear was identical to a hazard SET | the HAZARD row's magnitude is +1 on `sidestart`, −1 on `sideend`, written unscaled (only BOOST is `/ 6`) — no layout change |
| | (T1's class, on the row) | the OTHER side's own bare `-damage` (Substitute / Belly Drum cost) attached to our open move — the feature-coverage audit's strict xfail §3.5, live `machamp seismictoss hp_delta=-0.2493` | a bare `-damage` attaches only while the open move's user is the CURRENT MOVER; the xfail is now a plain pin |
| `gen3_intent_label_semantics_fixes_v1` | L1 / L2 | a Roar / Whirlwind-DRAGGED mon labelled as their chosen switch | `TurnDelta.opp_dragged` ⇒ masked |
| | L3 | the replacement for a faint in the PREVIOUS window labelled a chosen switch | `opp_switch_is_replacement` (the opening decision saw their active fainted) ⇒ masked |
| | L4 | a CALLED move labelled instead of its caller | `opp_called_via` ⇒ MOVE the caller (Sleep Talk, Mirror Move, Metronome, Assist, Nature Power, Magic Coat, Snatch) |
| | L5 | an Encore override labelled as the encored move | `opp_choice_overridden` (an Encore `-start` before their move, same turn) ⇒ masked |
| `gen3_progress_clock_attribution_fix_v1` | T1 | clause (i) credited sand / poison / recoil / Spikes chip to a status, failed or missed move | clause (i) also requires `our_move_hit_delta` (our moves' own hits) ≤ −3 % |
| | T2 | a blocked attack read outcome `hit`, so the "our attack blocked" freeze never fired | `TurnView` folds the block ⇒ outcome `fail` ⇒ the exogenous FREEZE |

No obs LAYOUT change: every fix is a value inside an existing column, a label mask, or a clock
decision. The new `TurnDelta` fields are not encoded. The Rust core (`src/rust_sim/src/trackers/`)
takes the same rules line for line (`designs/rust_sim/trackers.md` §5), and the Rust ENCODER (M4's,
which landed on main while this was in flight) writes the HAZARD magnitude unscaled as the
assembler does — slice O reproduces the regenerated obs golden byte for byte.

## 2. The pins

* `src/agents/training/training_input_semantics_test.py` — hand-built logs, one test per id.
  Against the BASE tree 13 / 14 fail; the passing one (`test_t2_…`) pins the clock's freeze branch,
  which always existed — T2's revert pin is the TurnView test beside it.
* `src/agents/battle/tracker_semantics_fixtures_test.py` — M3's constructed battles through slice T:
  0 divergences required AND the corrected value asserted. All 11 fail on the BASE tree.
* `src/rust_sim/tests/tracker_semantics_test.rs` — the Rust trackers on the same battles; each of the
  11 Rust fixes reverted alone fails its pin (`rust_revert_pins.py` / `.txt`).
* All of M3's 25 constructed fixtures and 5 GIGO repros replayed with M3's own script:
  **0 divergences on all 30**, the flipped values beside M3's originals (`e12_fixtures_after.txt`,
  `gigo_repros_after.txt` vs `../rust_core_m3_2026-09-24/catalogue/`).
* The event-window fuzz's INDEPENDENT oracle (`poke_env_gaps/event_window_fuzz_test.py`) was
  updated independently of the fold: 20 fresh battles, 36,534 checks, **0 failures**
  (`event_window_fuzz.txt`).

## 3. The golden census (`census.py`, `run_census.sh`)

Two captures of `golden_obs_capture`'s fixed battle set, one from the BASE tree and one from the FIX
tree, each with its own `src/` and its own `core_events`, the per-row provenance recorded beside each
vector (the seq of the event that appended the row, and the event log). Every changed obs cell is
resolved to its event-window SLOT → row ordinal → the key that changed, and that key's change must
be explained VALUE-AWARE by the events (`census.py`'s docstring lists the rules). The clock scalar
`reactive[2]` may change only where the two clocks' `n` differ, and a clock divergence must start at
a decision holding the T1 or T2 condition.

`golden_census.txt`: 991 decisions, **698 changed** — OUT_HIT / OUT_FAIL ← W4 (981 cells each),
MAGNITUDE ← W5 start (279), MAGNITUDE ← W1 (70), `turns_since_progress` (48), FAINT_CAUSE ← W2
(45: one Destiny Bond, three Perish Song — one of which the old rule called `weather`, its last
chip being non-lethal sand). The clock moved on T2 (32 transitions) plus 1 cascade. **Nothing
else.** The trajectory is identical (the golden policy reads the mask, not the obs).

## 4. The corpus census

The same instrument through slice T's corpora: COMMIT (49), seeded-random pool keys 0–119, the
`production` policy keys 100–109, ladder keys 0–59 — 239 battles, 82,327 event-window rows, 33,766
viewer decisions (slice T clean on every one, both trees). `policy.106` is EXCLUDED and named: the
policy reads the changed obs, so its trajectory branched after the first change (the other 9 policy
battles did not branch).

| change | /1,000 viewer decisions | M3's catalogue (MILESTONE corpus) |
|---|---|---|
| W1 stat drop, sign fixed | 54.40 | 58.39 |
| W4 Protect block → OUT_FAIL (per viewer row) | 24.34 | 17.50 blocks |
| W5 side condition start → +1 | 23.87 | — (every set row) |
| W5 side condition end → −1 | 2.61 | 8.67 Rapid Spin clears (+ screen ends) |
| T1-row foreign bare damage removed | 1.90 | — (new member of the class) |
| W2 faint → `other` | 0.36 | 0.27 |
| W3 transfer `|-item|` → SWAPPED | 0.06 | fixture only |
| L3 straddling replacement → masked | 9.95 | 8.38 |
| L1 / L2 drag → masked | 6.75 | 7.46 |
| L4 → the caller | 0.36 | 0.32 |
| L5 Encore override → masked | 0 | fixture only |
| clock: T2 blocked attack frozen | 9.89 | ≈ 8.75 |
| clock: T1 clause (i) without our own hit | 1.87 | 1.28 (decisive) |
| clock: cascade (`n` carries) | 2.28 | — |

## 5. Slice T and the gates

* COMMIT (`rust_core_parity_test.py`, unmarked): 14 / 14 pass, 0 divergences.
* MILESTONE (`-m slow`): **10 / 10 pass** before commit (`milestone_pre_commit.txt`, 12 min 23 s):
  seeded-random even / odd, `production` policy ×2, protocol + byte-fuzz, ladder keys 0–149 and
  150–299, and the three named ladder known-divergences still fire exactly as named. Re-run at the
  landed commit so `slow_tier_status.json`'s rows name the commit they measured.
* The routine gate: the only failures before the golden regeneration were the two obs-golden tests
  and the strict xfail this change fixes. `reward_golden_test` did NOT move (its six compositions
  over its 30 battles never hit a T1 or T2 window whose charge the reward reads — **UNVERIFIED**
  which of the two it is).

## 6. `obs_build_benchmark.py` (`obs_build_benchmark/`)

8 interleaved pairs, BASE vs FIX, load1 5.0–5.3: full (cold) 0.999× [0.989, 1.015], `live_view`
0.993× [0.973, 1.014], production shape 0.986× [0.961, 1.030] (`summary.txt`). No measurable change.

## 7. Findings

* **M3's `opp_double_edge_recoil_credits_our_status_move` repro never reaches its case** — the
  Double-Edge KOs the Blissey on turn 1, so that window closes on a forced switch and the clock sits
  out (M3's own `gigo_repros.txt` shows the same `n = 1` flat line). T1 is pinned on the sand repro,
  which does reach it.
* **Memento** (a no-damage self-faint) now reads `other`, not `selfko`: the W2 rule catches it, but
  `_SELF_KO_MOVES` is Explosion / Self-Destruct only. Not changed here: adding it would inherit the
  existing `used_selfko` flag's staleness (a Memento that FAILS leaves the flag set until the mon
  moves again). Unmeasured in the corpus.
* **A Ghost's Curse self-KO still reads `attack`** — its HP cost is a bare `-damage` on its own user
  while it moves, and the faint-cause classifier has no current-mover rule. Rare; not fixed.
* The label now masks a switch that its CHOOSER's own drag followed (L1: they chose Blissey, our Roar
  dragged Starmie). Their real choice (SWITCH Blissey) is visible in the window; masking forgoes it
  rather than risk the wrong mon. A later label change could recover it.

## Files

`census.py` + `run_census.sh` (§3–4) · `golden_census.txt` · `corpus_census.txt` ·
`e12_fixtures_after.txt` · `gigo_repros_after.txt` · `event_window_fuzz.txt` · `milestone_pre_commit.txt` ·
`rust_revert_pins.py` + `rust_revert_pins.txt` (paths are the fix worktree's) · `obs_build_benchmark/`.
