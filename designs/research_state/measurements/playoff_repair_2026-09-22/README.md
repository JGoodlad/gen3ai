# THE `playoff` ARM'S THREE BLOCKERS, ROOT-CAUSED AND CLOSED — and none of them was the prefix gate

*2026-09-22 · CPU only (`CUDA_VISIBLE_DEVICES=""`, `nice 15`), `--impl rust --search-impl rust`,
nothing written under `models/`, ports 8000/8001 untouched, no process this read did not start was
signalled. Checkpoint: `models/ai_v12_02_winprob_critic/final_model.zip` (the 2026-09-19 row names
only `<ckpt>`, so the choice is stated rather than inherited).*

> **The row said one thing, the 2026-09-22 re-measurement said another, and BOTH are refuted here.**
> The P1 row recorded `prefix_gate_failed` on 72 of 73 decisions. The re-measurement recorded
> `root_failed` on 51/63 (node driver) and 60/63 (rust). On a correctly-provisioned worktree at HEAD,
> the row's own repro produces **ZERO of either, on BOTH drivers**, and the two drivers agree
> decision for decision. What actually blocks the arm is three defects nobody had named.

---

## 0. WHAT THE REPRO ACTUALLY DOES AT HEAD

The row's own command, plus `--search-impl rust`, binaries pinned to the main checkout's
`target/release`:

```bash
export PYTHONPATH=$PYTHONPATH:src
export POKESIM_SIM_BRIDGE_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge
export POKESIM_SEARCH_DRIVER_BIN=/home/goodlad/dev/gen3ai/src/rust_sim/target/release/search_driver
python3 -m main.search_dividend models/ai_v12_02_winprob_critic/final_model.zip \
    --arm playoff --opponents self --games 1 --games-seed 7 --budget 60 \
    --playoff-rollouts 2 --playoff-min-pairs 2 --max-opp 1 --max-worlds 1 --max-dice 1 \
    --max-depth 1 --impl rust --search-impl rust
```

| search driver | o0 | decisions | searched | changed | fallbacks | o1 |
|---|---|---|---|---|---|---|
| **rust** (`--search-impl rust`) | loss, 102.2 s | 30 | 5 | 1 | `playoff_inconclusive:20` · `not_move_selection:5` | ProgressTimeout at 180.5 s |
| **node** (the repro's default) | loss, 99.7 s | 30 | 5 | 1 | `playoff_inconclusive:20` · `not_move_selection:5` | ProgressTimeout at 206.9 s |

**Zero `root_failed`. Zero `prefix_gate_failed`. Zero `worlds_gate_failed`. 21 playoffs RAN at a
realized R of exactly 2.0.** The two drivers produce the same row.

⚠️ **`root_failed` is easy to manufacture from the ENVIRONMENT, and that is the most likely account
of both earlier readings.** An unresolvable search-driver child fails every `open_root`: no
`POKESIM_SEARCH_DRIVER_BIN` and no `src/rust_sim/target/` in the worktree (rust), or no
`deps/pokemon-showdown/dist` symlink (node — and the `conftest.py` guard that refuses this does not
run for a CLI invocation). The arm then silently becomes its own `base` control. It is no longer
possible to read that as a clean cell: see defect 1.

---

## 1. DEFECT — `root_failed` did not say WHAT `open_root` raised · `c823b11c`

**Root cause.** The exception was written into `diagnostics["worlds"][i]["gate"]` as
`open_failed: <cls>: <msg>` and no results-file field carried it. `_log_decision` lifted
`diagnostics["error"]` into the row, and the two no-arm return sites never set it. So the counter
could read 60/63 and the only way to learn the reason was to patch the engine and re-run — which
the previous agent did, and the run died before reaching a searched decision (defects 2 and 3).

**Fix.** `search._no_arm_detail` collects the DISTINCT `open_failed:` messages (capped at three — a
decision opens K worlds and a dead driver raises K times) into `diagnostics["error"]`;
`battery.summarize_decisions` classes them into a new row field `fallback_errors` through the same
bounded `bump_error_class` the playoff's rollout errors use; `summary.per_cell` pools it per cell
and `format_report` prints an `err:` line under any cell whose fallbacks carried a message.
`playoff.root_failure_refusal` closes the hole the other two refusals left open — a decision whose
`open_root` raised never reaches a screen, so `playoff_error_refusal` and `short_r_refusal` both
see `attempted == 0` and return `None` on exactly the row that most needs a refusal. It fires on
the FIRST game at a third of SEARCHABLE decisions and names the classes.

**Test that fails on revert.** `search_test.py::test_a_dead_driver_becomes_root_failed_not_a_crash`
now asserts `res.diagnostics["error"] == "RuntimeError: driver died"`;
`battery_test.py::test_the_exception_behind_a_fallback_is_CLASSED_onto_the_row`;
`playoff_test.py::test_root_failure_refusal_fires_where_the_other_two_guards_are_STRUCTURALLY_BLIND`.

**One standing doc claim withdrawn in the same pass:** `search_dividend/__init__.py` stated that the
rust driver's `open_root` "cannot replay a LIVE-synthesized record at all: 43 of 44 decisions
returned `root_failed: battle never reached the start of turn 2`". It does not reproduce.

---

## 2. DEFECT — `ValueThreatInject shape mismatch: tokens (1, 6) vs rows (9, 6)` is a CROSS-THREAD STASH READ · `d5c465fd`

**It is not a batch-shape bug on the value-threat channel and has nothing to do with the successor
obs.** `Gen3FeaturesExtractor` keeps its whole per-forward contract on `self` — `forward_internal`
opens with `self.stash = ExtractorStashes()`, `DamageOperator.forward` with `self.stash =
OpStashes()` — and consumers read those attributes back LATER in the same forward (`CLSPool` reads
`damage_op.last_reduced_extra`; `RLPlayer` reads `last_win_prob_logits` and the α publication
AFTER `forward` returns). Every one of those reads assumes no other forward ran in between. True of
training, where each env worker is its own PROCESS. **False in the search battery's mirror**, which
runs `SearchEngine.choose` in a `run_in_executor` worker — it must, or the materializer deadlocks
POKE_LOOP against itself — while the unsearched side, and every playoff rollout player, decides on
POKE_LOOP against the SAME `model`. `PlayoffRunner._live_rollout` documents that arrangement in its
own docstring. Torch releases the GIL inside its kernels, so the two forwards interleave.

**MEASURED, not inferred** — two threads, one real extractor, B=1 against B=9, 2,400 interleaved
forwards:

| failures | class |
|---:|---|
| 637 | `IndexError: index 1 is out of bounds for dimension 0 with size 1` |
| 192 | `ValueError: value_threat_inject is built but the op supplied no reduced rows` |
| **172** | **`ValueError: ValueThreatInject shape mismatch: tokens (1, 6) vs rows (9, 6)`** |
| 30 | the same mismatch the other way round — `tokens (9, 6) vs rows (1, 6)` |
| 16 | `AttributeError: 'NoneType' object has no attribute 'incoming_rows'` |
| 11 | `RuntimeError: Sizes of tensors must match except in dimension 1` |
| 5 | `TypeError: 'NoneType' object is not subscriptable` |
| **1,063** | **total** |

🚨 **THE CRASH IS THE LUCKY CASE.** Two colliding forwards only RAISE when their batch sizes
differ. The mirror opponent's live decision and a playoff rollout player's decision are both B=1,
and there the identical race silently swaps one decision's belief logits, threat rows and P(win)
for another's with nothing anywhere saying so.

**Fix.** `agents/model/forward_guard.py` — an OPT-IN re-entrant lock per extractor, kept in a
weak-keyed registry (an `RLock` set as a module attribute breaks `copy.deepcopy(policy)`, which SB3
does). `Gen3FeaturesExtractor.forward` branches on `guard is None`, so the default path contains no
context manager at all and the compiled graph is byte-unchanged; training pays nothing.
`battery.build_players` installs it (that is the function that puts N players on one model) and
`SearchDividendPlayer.choose_move` holds it across its forward AND both stash reads — replacing a
comment that claimed "there is no await between our forward and this read, so nothing can land in
between", which is true of the LOOP and false of the process.

**Test that fails on revert.**
`forward_guard_test.py::test_a_GUARDED_extractor_is_byte_identical_to_the_SINGLE_THREADED_control`
— it asserts OUTPUT BYTES against a single-threaded control rather than "it did not crash", because
a corrupted forward with agreeable shapes returns wrong numbers. Reverting the `with guard:` fails
it with exactly `ValueError: ValueThreatInject shape mismatch: tokens (1, 6) vs rows (9, 6)`.

---

## 3. DEFECT — the "first-orientation livelock" is NOT a livelock · `4d75cfe3`

**Root-caused by raising the cap.** The same game the runner killed at 180 s with
`ProgressTimeout … livelock, not a stall` **FINISHES at 246.6 s** with nothing else changed:

```
[playoff@60 self g0/o0]  loss 148.9s  dec=30  searched=5  changed=1  fb=not_move_selection:5,playoff_inconclusive:20
[playoff@60 self g0/o1]  win  246.6s  dec=60  searched=3  changed=2  fb=not_move_selection:1,playoff_inconclusive:56
```

It was walking through turns the whole time — a progress event every ~0.5 s, each nested rollout
0.3–0.8 s — and the IDLE detector never fired, correctly, because nothing was ever wedged.
`_PER_BATTLE_TIMEOUT = 180.0` is a TOTAL-DURATION cap, and on a search arm the duration of a battle
is an ARGV PARAMETER (`--budget`, `--playoff-rollouts`, the width caps). **Half of every side-swapped
playoff cell was being deleted by a constant** — and the row it left behind read `dec=0`, because
`play_one_battle` let the exception escape and the caller's handler wrote `"decisions": []`.

**Three fixes.**
* **The message reports the frame instead of asserting a verdict** (`gen3_progress_frame_report_v1`).
  `BattleStreamClient.feed` records `t<turn>:<last protocol keyword>` per block and
  `ProgressDeadline` samples it on every sign of life. A total-budget expiry now ends either
  `LIVELOCK CONFIRMED: every one of the N signs of life was the same frame \`t3:error\`` or
  `NOT A LIVELOCK: the work ADVANCED through D distinct frames (\`t0:turn\` -> \`t7:turn\`) … raise
  the budget`. No sampler ⇒ `UNKNOWN from here`, never the old bare claim.
* **Both per-battle bounds are caller-sizable** (`gen3_caller_sized_battle_budget_v1`).
  `search_dividend.player.battle_bounds` sizes them from the cell's realized per-decision cost —
  the idle bound too, because on a search arm ONE DECISION is the longest gap between two protocol
  chunks. Never below the module defaults. `run_cell` seeds at a measured 5 s and carries the
  realized `wall_s / n_decisions` forward.
* **A timed-out battle keeps its decisions.** The outcome stays `unfinished` and leaves every rate
  alone — a timeout is never a semantic outcome — but the evidence reaches the results file.

**Tests that fail on revert.** 5 in `contention_test.py`, 2 in `local_battle_runner_test.py`, 1 in
`battle_stream_client_test.py`, 2 in `player_test.py`
(`test_battle_bounds_SCALE_…`, `test_a_timed_out_battle_KEEPS_the_decisions_it_made`).

---

## 4. DEFECT — the decision counters did not add up · `569f64ec`

`prefix_gate_failed: 9` beside `worlds_gate_failed: 0` is two statements that cannot both be true,
and the reason is one `continue`. Two defects in `summarize_decisions`:

1. **The WIDTH counters were summed over SEARCHED decisions only.** `worlds_gate_failed` and
   `deadline_truncated` sat after the `continue` every fallback takes — and a decision whose every
   world failed the gate IS a `prefix_gate_failed` fallback, so the one row shape the counter exists
   for was guaranteed to report zero. `worlds_open_failed` — the counter behind `root_failed`, which
   `search.py` has kept apart from the gate counter since the beginning — was folded NOWHERE.
2. **A decision with a NOTE but no fallback counted as SEARCHED.** `policy_default` and an
   `order_failed` whose search had succeeded both reach the row with `fallback` unset, inflating the
   denominator of `change_rate` with decisions the search's action was never played on.

**Fix.** One `decision_reason(d)` classifier (`None` = searched), `NOTE_AS_FALLBACK` naming the two
notes that are not, the width counters over every decision, `worlds_open_failed` folded and pooled,
and `check_decision_accounting(row)` stating `n_decisions == n_searched + Σ fallbacks` — which the
fold RAISES on. **Tests:** 4 in `battery_test.py`, including the identity over five row shapes and
the guard proven to fire on a hand-built row that does not add up.

---

## 5. THE MEASUREMENT THE ARM EXISTS FOR — **the arm ACTS**

One mirror game per orientation, `honest` against `playoff`, at JOB B's registered operating point
(`playoff_gate_operating_point_2026-09-20/` §82) but on the RUST search driver, which that battery
never ran:

```bash
python3 -m main.search_dividend models/ai_v12_02_winprob_critic/final_model.zip \
    --arm honest --arm playoff --opponents self --games 1 --games-seed 7 \
    --budget 120 --playoff-rollouts 4 --playoff-se-k 0.5 --playoff-min-pairs 4 \
    --max-worlds 4 --max-dice 2 --max-opp 6 --max-depth 1 \
    --device cpu --impl rust --search-impl rust
```

| arm | orient | result | decisions | **searched** | **changed** | change rate | wall | fallback classes | gate/open fail |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| honest | 0 | loss | 44 | 39 | 26 | 0.667 | 96 s | `not_move_selection:5` | 0 / 0 |
| honest | 1 | loss | 27 | 22 | 16 | 0.727 | 52 s | `not_move_selection:5` | 0 / 0 |
| **playoff** | 0 | loss | 47 | **31** | **13** | 0.419 | 933 s | `not_move_selection:5` · `playoff_inconclusive:11` | 0 / 0 |
| **playoff** | 1 | **win** | 33 | **20** | **9** | 0.450 | 416 s | `not_move_selection:5` · `playoff_inconclusive:8` | 0 / 0 |

**The playoff arm's second stage, pooled over the two games:** 70 decisions screened →
**18 screen-decisive (25.7 %)**, **33 PLAYED** (the playoff resolved and acted), **19
inconclusive (27.1 %)**, **0 `no_budget`**, **0 errored**, **0 failed rollout pairs**.
**Realized R = 4.00 against a requested 4**, at 25.1 s per adjudicated decision.

`fallback_errors` is **empty on every row** — no fallback on any of the 151 decisions carried an
exception message, which is the field's first live reading and the shape a healthy cell has.

**What this establishes, and what it does not.**
* ✅ **The arm ACTS.** 33 of 70 screened decisions were settled by paired rollouts, 22 of 151
  decisions were CHANGED away from the policy's pick, and `n_searched = 51`, not 0. The three
  readings that said otherwise (72/73 `prefix_gate_failed`; 51/63 and 60/63 `root_failed`) are
  all superseded.
* ✅ **`--search-impl rust` is clean here**, and this is the first playoff cell ever run on it: the
  2026-09-20 battery was `--impl node` throughout.
* ✅ **The livelock fix is load-bearing for this cell, not cosmetic.** Both playoff games exceed
  the old 180 s constant — one by **5.2×** (933 s). Under the old bound this measurement could not
  have been taken at ALL; every playoff row would have been an `unfinished` timeout.
* ✅ **`--budget 120` buys the requested R on this box.** JOB B's `n_playoff_no_budget = 15.2 %` at
  load 40 is 0 % here at load ~7-18, and `short_r_refusal` did not fire.
* ⛔ **NOTHING about strength.** n = 1 pair. The mirror reads `playoff` paired **0.5000** and
  `honest` paired **0.0000**, both over ONE swap-pair; neither interval is computable and neither
  number is a result. Read the 965-battle `playoff_gate_operating_point_2026-09-20/` battery for
  the strength question — this cell's job was to show the instrument works.
* ⚠️ **One number worth noting for whoever sizes the next battery:** at these widths the playoff
  arm costs **25.1 s per adjudicated decision** on the rust driver — 933 s for a 47-decision game.
  A 200-pair battery at this operating point is ~100 CPU-hours before any sharding.

---

## Ready-to-append ledger paragraph

> **2026-09-22 — THE `playoff` ARM IS UNBLOCKED, AND ITS RECORDED BLOCKER WAS WRONG TWICE.** The P1
> row said `prefix_gate_failed` on 72 of 73 decisions; the 2026-09-22 re-measurement said
> `root_failed` on 51/63 (node search driver) and 60/63 (rust). On a correctly-provisioned worktree
> at HEAD the row's own repro produces **ZERO of either, on BOTH drivers, which agree decision for
> decision** — and `root_failed` turns out to be easy to manufacture from the ENVIRONMENT (an
> unresolvable search-driver child: no `POKESIM_SEARCH_DRIVER_BIN` and no worktree `target/` on
> rust, no `deps/pokemon-showdown/dist` symlink on node, where `conftest.py`'s guard does not run
> for a CLI invocation), after which the arm silently becomes its own `base` control. **FOUR
> DEFECTS CLOSED, 14 gates, each failing on revert.** (1) `c823b11c` — `root_failed` now NAMES what
> `open_root` raised (`_no_arm_detail` → `diagnostics["error"]` → the row's CLASSED
> `fallback_errors`, pooled per cell and printed), and `root_failure_refusal` closes the hole the
> other two guards left open: a decision whose root raised never reaches a screen, so
> `playoff_error_refusal` and `short_r_refusal` both saw `attempted == 0` on exactly the row that
> needed a refusal. (2) `d5c465fd` — 🚨 **`ValueThreatInject shape mismatch: tokens (1, 6) vs rows
> (9, 6)` is a CROSS-THREAD STASH READ**, not a batch-shape bug on the successor obs.
> `Gen3FeaturesExtractor` keeps its whole per-forward contract on `self` and the mirror runs
> `SearchEngine.choose` in an executor worker while the unsearched side — and every playoff rollout
> player — decides on POKE_LOOP against the SAME model object. Measured: two threads, one real
> extractor, 2,400 interleaved forwards ⇒ **1,063 failures in seven classes**, and ⚠️ **the crash is
> the LUCKY case** — two B=1 forwards corrupt each other's belief logits, threat rows and P(win)
> silently. Fixed with an OPT-IN re-entrant forward guard installed where the second thread is made;
> training and the compiled graph are byte-unchanged, and the gate asserts OUTPUT BYTES against a
> single-threaded control. (3) `4d75cfe3` — 🚨 **the "first-orientation livelock" is NOT a
> livelock**: the same game FINISHES at **246.6 s** (a WIN, 60 decisions) with the cap raised and
> nothing else changed, so `_PER_BATTLE_TIMEOUT = 180.0` — a TOTAL-DURATION cap on a workload whose
> duration is an ARGV parameter — was deleting half of every side-swapped playoff cell, and the row
> it left read `dec=0` because a timed-out battle's decisions were discarded. Both per-battle bounds
> are now caller-sized from the cell's realized per-decision cost, `ProgressDeadline` REPORTS the
> repeating frame (`LIVELOCK CONFIRMED … the same frame t3:error` vs `NOT A LIVELOCK: the work
> ADVANCED through D distinct frames`) instead of asserting one, and a timed-out battle keeps its
> decisions. (4) `569f64ec` — `prefix_gate_failed:9` beside `worlds_gate_failed:0` was one
> `continue`: the width counters were summed over SEARCHED decisions only, and a gated-out decision
> is by definition a fallback. The counters now sum to the decision count by construction, and
> `worlds_open_failed` — folded nowhere at all — is carried beside the gate counter. **THE ARM NOW
> ACTS**: one mirror game per orientation at the registered operating point (`--playoff-se-k 0.5`,
> R = 4, `--budget 120`, widths 6/4/2) on the rust search driver — the first playoff cell ever run
> there — gives **51 searched decisions, 22 changed, 33 playoffs PLAYED of 70 screened, 19
> inconclusive, 0 no-budget, 0 errored, realized R = 4.00 exactly**, zero gate failures and zero
> driver failures. ⛔ **No strength claim**: n = 1 swap-pair. ⚠️ Both playoff games exceed the old
> 180 s bound, one by 5.2× (933 s), so under it this measurement could not have been taken at all;
> and the arm costs **25.1 s per adjudicated decision**, i.e. ~100 CPU-hours for a 200-pair battery.
> Record: `designs/research_state/measurements/playoff_repair_2026-09-22/`.
