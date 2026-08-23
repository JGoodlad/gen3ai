# CLAUDE.md — Observation Encoder (`src/agents/observation/`)

This directory builds the **2501-dim per-decision observation vector** (`Gen3ObservationEncoder.encode`;
the live value is `Gen3ObservationEncoder.dimension` — read it there, and see
`designs/ARCHITECTURE.md` § Observation for the full block table).
It runs once per agent decision across every training env, so it sits directly on the
training-throughput (FPS) critical path. Two independent things can regress here, and they
have **different** gates:

1. **Observation *values*** — if a change alters what the vector contains, it is
   **retrain-class**: bump `ARCH_SIGNATURE` in `src/agents/model/model_version.py` (see the
   root `CLAUDE.md` → Model Versioning). Value-neutral refactors do **not** bump it.
2. **Observation *build performance*** — if a change makes `encode` slower, training FPS
   drops for the entire run. **This file governs that gate.**

> **Off-hot-path exception — `belief_labels.py`.** This module (the pure builder of the
> hidden-opponent belief-aux labels) lives here for cohesion with the obs layer but is **NOT called
> by `encode`** — `Gen3Env.step/reset` invoke it only when `--opp-belief-aux-coef>0` or
> `--move-belief-mode != off`, to emit the privileged training-only `belief_species`/`belief_moves`
> (and, for move-belief known/both, `known_moves`; for `--spread-belief-coef>0`, `belief_spread`/`_mask` —
> and, under `--spread-belief-nature`, the inverted `belief_nature`/`belief_ev`(+masks) for the nature/EV
> decomposition, `gen3_nature_ev_belief_v1`, deterministically inverted from agent2's `mon.stats` so no leak;
> for `--hp-type-belief-coef>0`, `hp_type_label`/`hp_type_mask` — the opp Hidden-Power-type label,
> `gen3_typed_hp_belief_v1`) Dict keys (see
> `src/agents/training/CLAUDE.md`). So it adds **zero** cost to the default obs build (benchmark
> confirmed: `state_encoder.encode` unchanged, `belief_labels` absent from the profile). Changes to
> `encode` itself still trip the gate below.

---

## MANDATORY: run the obs-build benchmark on every change to this directory

**Any change to a file under `src/agents/observation/` — even a "pure refactor" or a
value-neutral one — MUST run the performance benchmark before and after the change and
confirm no meaningful regression.** This is not optional and applies to every edit, however
small. A one-line change to a hot loop (e.g. `reactive.py`, `pokemon.py`, `moves.py`,
`state_encoder.py`) can silently halve FPS.

The benchmark is `src/agents/training/obs_build_benchmark.py` (it lives in `training/` rather
than here only because a directly-run script puts its own dir on `sys.path[0]`, and
`observation/types.py` would shadow the stdlib `types` module — see the root `CLAUDE.md`
Benchmarks section).

### The workflow — do this for every change

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src

# 1. BEFORE you edit: capture a baseline on the CURRENT code (stash/commit your change away,
#    or run on a clean checkout), saving the full output.
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
    src/agents/training/obs_build_benchmark.py --turn 25 --reps 400 --top 22 | tee /tmp/obs_before.txt

# 2. Apply your change.

# 3. AFTER: re-run with the SAME flags and the SAME machine load, and diff.
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
    src/agents/training/obs_build_benchmark.py --turn 25 --reps 400 --top 22 | tee /tmp/obs_after.txt

diff /tmp/obs_before.txt /tmp/obs_after.txt
```

**"the SAME machine load" is now checked for you** (`gen3_contention_robust_timeouts_v1`): the
benchmark calls `warn_if_contended()` at entry and prints a loud "THE BOX IS BUSY" banner with the
load average when the box is not idle. It still WARNS rather than refuses — a before/after pair run
back-to-back under the *same* load is exactly the same-load A/B this gate asks for — but a banner on
only one of the two runs means the comparison is void, so check both outputs before believing a diff.

If you cannot easily get a "before" (the change is already applied), compare against the
**canonical baseline pasted below** — but prefer a same-session before/after, because
absolute timings are machine- and load-dependent.

### What counts as a "meaningful regression" — use the LOAD-STABLE signals

Absolute milliseconds scale with whatever else the box is doing (training alone pushes load
past the core count and can inflate the numbers 2–3×). **Do NOT judge by the `ms` line.**
Judge by these load-independent metrics, in priority order:

1. **Total function calls per encode** = `<N function calls>` line ÷ `--reps`. This is the
   single best regression detector — it does not move with machine load. Baseline ≈
   **~5.43k calls/encode** — measured 2026-08-23 on an idle box at `--turn 25 --reps 400`, obs
   2501, median of 3 runs (`designs/research_state/measurements/post_paydown_baselines_2026-08-23.json`).

   🚨 **This number is in FULL-PROTOCOL units. The older ~3.46k figure below is NAKED-ENCODE and
   is NOT comparable to what this benchmark prints today** — comparing today's output against it
   reads as a +57% regression that does not exist. The benchmark has threaded the whole env
   protocol since 2026-08-16 (`update_progress_clock`, recency, the H-A pair loop, the H-B
   event-window fold); `episode_tracker._pair_sat_norm` alone is ~164 calls/encode, and the
   tracker family is most of the difference. A threshold restated in the wrong units is worse
   than no threshold, because it fires.

   *History, in NAKED-ENCODE units — for the shape of past changes only, never as today's bar:*
   ~3.46k was the post-`gen3_entity_rehome_v1` (v60) reference: deleting the two
   144-dim matchup matrices removed the whole `_expected_multiplier`/`_joint_expectation` loop
   family (measured same-session before/after at `--turn 25 --reps 400`, seed-0 battle:
   6,332 → 3,462 calls/encode, −45%; wall 0.373 → 0.246 ms, −34% — the Stage-3 refund,
   confirmed in reverse). ~6.44k was the post-`gen3_cpu_damage_deleted_v1` (v48)
   reference, measured
   same-session before/after at `--turn 25 --reps 300` on the seed-0 battle (7,396 → 6,444, −12.9%,
   from deleting the incoming-damage / move-effect / active-move-scalar producers). History for
   context: ~6.36k pre-`gen3_incoming_damage_v1`, ~6.85k after it, ~7.4k after the `v2` belief
   recalibration (crit term + the wider candidate set). Always judge by a same-session before/after,
   not the absolute. A jump of **>10%** above this is a regression — investigate.
2. **cProfile `tottime` top-of-list structure.** A *new* function climbing into the top ~10,
   or a known hot function's **call count** ballooning, means you added work to a hot loop.
3. **Component ratios** (`state_encoder.encode` vs `live_view`). The turn-history component and
   its deque-cache multiplier are GONE with the lag frames (`gen3_frame_deletion_v1`), so the
   build is now `encode` + `live_view` only. Historically, if the turn-history "cached" line
   stopped being a
   single encode (`~12x saved` collapses toward `1x`), the deque memoization broke.

A value-neutral refactor that adds <10% calls/encode and doesn't reshuffle the tottime top is
fine. Anything larger needs justification (or a revert).

### 🚨 The build is INCREMENTAL now — there are FOUR series, and they answer different questions

`gen3_obs_assembler_v1` (`assembler.py`) makes `encode` a **scheduler** over a persistent
2501-dim buffer: it re-derives only the blocks an event, the request, or the HP tracker says have
moved, and the per-block writers are unchanged. So one number can no longer describe "the obs
build", and the benchmark prints all four:

| series | what it is | when production pays it |
|---|---|---|
| `full … [COLD]` | full rebuild, view memo dropped each rep | the comparable-to-history series |
| `encode with the view memo WARM` | full rebuild, view already built | what encode cost BEFORE this change |
| `cache invalidated each rep` | full rebuild THROUGH the assembler | the episode's first decision; after a re-decide rollback |
| **`cache WARM + view memo WARM`** | **the incremental encode** | **every other decision — THE PRODUCTION SHAPE** |

**Measured 2026-08-23, busy box (load 12–20 — absolute ms inflated, RATIOS are the claim),
`--turn 25 --reps 400`:** production-shape encode is **2.6–2.7× cheaper** than the same decision's
full rebuild, and warm `calls/encode` is **~1.33k against a cold ~4.6k (−72%)**. The
decision-matched end-to-end is `trainer_turn_benchmark` (below), which walks *consecutive*
decisions and therefore carries real dirty sets rather than one decision's: there the encode is
**1.79×** (0.302 → 0.169 ms, three same-load pairs, disjoint ranges) and worker CPU is
**1.19× (−16%)**.

⚠️ **The warm reps loop is an OPTIMISTIC bound by construction** — it re-encodes ONE decision, so
every rep has the same dirty set. It is a bound on the win, not the win. Quote
`trainer_turn_benchmark` for anything end-to-end; that script grew a **`--no-assembler`** arm for
exactly this, and a same-session pair is the only honest way to read absolute ms on this box.

⚠️ **The COLD `calls/encode` moved: ~5.43k → ~4.6k**, and that is a real reduction, not a rebase.
It is the saturation LUT (`assembler.SAT_LUT`, an 11-value codomain read as a table by
`RecencyTracker.values` / `PairHistoryTracker.pair_values`) plus the pair-history block becoming
ONE 180-float slice assignment instead of 36 five-float ones. Both are pinned bit-for-bit against
the arithmetic they replaced (`assembler_test::test_the_saturation_lut_is_bit_for_bit_…`).

**Byte-identity is asserted BY the benchmark**, not assumed: it encodes the profiled decision both
ways and raises before printing if they differ. A speedup measured against a different vector is
not a speedup.

🚨 **The reps loop re-encodes ONE decision, so ANY cache in the build is 100% warm from rep 2
— read the COLD series.** `battle.live_view()` is memoized per state-epoch
(`gen3_live_view_memo_v1`, `src/agents/battle/CLAUDE.md`), and production sees exactly one
COLD view build per decision — the memo's job is that the *other four* builds vanish, not that
the encode's own build gets cheaper across reps. Left alone, the benchmark would have reported
`live_view() alone : 0.000 ms (0%)` and a `calls/encode` that is bimodal by construction: a
fantasy speedup, and the primary regression metric silently rebased. It therefore **drops the
memo before every rep** (`_invalidate_view_memo`) for `full` / `enc_only` / `live_view`, and
prints one extra WARM line beside them for the honest cost of the encode's view read once the
mask/tracker path has already built it. **Judge calls/encode from the cold cProfile block.**
Measured 2026-08-23, quiet box, `--turn 25 --reps 400`: cold 5401 / 5401 / 5369 / 5562
calls/encode across four runs — the spread is which decision got profiled (`--seed` seeds
action selection only; the bridge mints its own sim seed), so a single-run before/after diff
of ±3% here is noise, not signal. Warm encode runs ~0.29–0.32 ms against cold ~0.39–0.42 ms.
**A cache added inside the encode must extend this pattern, not lean on it.**

---

## Canonical baseline (paste — the reference point for regressions)

Captured with `--turn 25 --reps 400`. Paths shown repo-relative. Absolute ms omitted from the
headline on purpose (load-dependent); the **call counts and ordering are the contract**.

> ⚠️ The pasted block below predates `gen3_entity_rehome_v1` (the matchup deletion): the
> matchup-era hot list (`effective_multiplier_by_types`, `reactive.py:encode` at ~44% of encode,
> `_joint_expectation`) no longer exists. The block is kept for the v48-era shape until the
> next full re-baseline.
>
> **Current headline (re-baselined 2026-08-16, idle box, `--reps 200`, obs 3529 — i.e. BEFORE
> `gen3_frame_deletion_v1` took the obs to 2437; the deletion only REMOVES work from this path,
> so the figures below are an upper bound until the next re-baseline)** — and a
> MEASUREMENT-HONESTY correction: until this date the benchmark (like the golden capture)
> never ran `update_progress_clock` and threaded none of the tracker-fed blocks, so every
> "encode ≈ 0.25 ms" figure timed the progress-clock/recency/H-A/H-B writes as SKIPPED —
> production always paid them. With the FULL env protocol threaded:
> **0.363 ms/decision** (encode 98%). Split: naked encode 0.246 ms; + recency/clock/H-A
> pair-loop ≈ +0.077 ms (paid since v79 — gen-11 trained at this cost); + the v81 H-B
> event-window write loop ≈ **+0.040 ms (+12%)** — the marginal cost of enabling nothing
> (the block is unconditional; the fold is ≤32 dict-row writes + 2 species and 1 move dex
> lookups per row). If the H-B tier survives its audit, vectorizing the row writes (numpy
> assembly in the tracker) is the obvious first optimization.
>
> **RE-BASELINED 2026-08-23** (idle box, no training run, `--turn 25 --reps 400`, median of 3):
> **0.373 ms/decision, ~5.43k calls/encode, obs 2501.** Note the obs figure — the `2437` above is
> stale; `gen3_frame_deletion_v1` took it there, later work brought it to 2501, and
> `Gen3ObservationEncoder.get_layout()` is the only figure worth trusting. The 2026-08-16
> headline's "upper bound" caveat is now DISCHARGED by a same-session A/B rather than assumed:
> against `bcdd868` (the frame deletion) the current tree is **+2.4% calls/encode and +3.6%
> wall**, tracking the +2.6% obs-dim growth — i.e. the frame-deletion era, v96–v100, the cf
> plumbing and the entry-point decomposition moved no hot path. Full record:
> `designs/research_state/measurements/post_paydown_baselines_2026-08-23.{json,md}`.

```
PER-DECISION OBS BUILD BENCHMARK  (obs dim <live>, turn 25, opp mons w/ revealed moves 5/6)

  full per-decision obs build  :  ~0.5–1.2 ms   (LOAD-DEPENDENT — not a regression signal)
    state_encoder.encode       :  ~79% of build
    live_view() alone          :  ~15% of build

  Total: ~2.74M function calls / 400 reps  ==>  ~6.85k calls per encode   <-- PRIMARY REGRESSION METRIC
  (the +0.49k vs the pre-feature 6.36k is the gen3_incoming_damage_v1 belief loop; per-species
   candidate/stat work is lru_cached, so only the per-defender damage/outspeed math is per-decision.
   gen3_incoming_damage_v2 adds ~+6.6% on top: the crit term doubles the per-candidate damage calc and
   the wider candidate pool fills more (defender, channel) pairs — _channel_threat goes ~10→12 calls/
   encode. Still no single dominant hot loop; the revealed-HP typed expansion is NOT lru_cached (it
   tracks the per-episode HP tracker) but only fires when a bare hiddenpower is revealed.)

  Top functions by tottime (no single dominant hot loop — the matchup work is now spread thin):
   ncalls  tottime  cumtime  function
    80800    0.048    0.088   agents/gen3_mechanics.py:effective_multiplier_by_types (memoized; chart lookup)
      400    0.046    0.261   agents/observation/reactive.py:encode                  (cumtime ≈ whole matchup block)
     4800    0.036    0.101   agents/observation/moves.py:encode
     4800    0.032    0.111   agents/battle/live_view.py:from_pokemon
    42800    0.028    0.042   poke_env/battle/move.py:entry                          (poke-env Move property)
    80800    0.028    0.116   agents/observation/reactive.py:_joint_expectation
   238800    0.026    0.035   enum.__hash__                                          (lru_cache key hashing)
     4800    0.025    0.189   agents/observation/pokemon.py:encode
    26400    0.012    0.040   poke_env/battle/move.py:max_pp
     4800    0.012    0.023   agents/observation/types.py:encode
   264800    0.011    0.011   {builtins.len}
      400    0.011    0.596   agents/observation/state_encoder.py:encode             (cumtime ≈ whole obs)
```

**Reading it:** there is **no single dominant hot loop** anymore — the matchup encoder's
per-cell poke-env property reads were hoisted to team level (`reactive._defender_terms` /
`_attacker_type_dist`, computed once per mon / per (attacker, move) instead of per cell), and
the per-mon move category is memoized by id (`moves._category_val`), so `move.entry` dropped
from ~158k to ~43k calls and `pokemon.ability` / `move.type` left the top list. Cost is now
spread across `effective_multiplier_by_types` (the memoized chart lookup — the irreducible
per-cell core), the matchup `encode` loop overhead itself, and the per-mon encoders. Type
effectiveness must stay a memoized chart lookup (`effective_multiplier_by_types` + `_eff_cached`
in `gen3_mechanics.py`) — `PokemonType.damage_multiplier` must **not** reappear here (if it
does, something bypassed the chart). The matchup block (`reactive.encode`) and the per-mon
`pokemon.encode` / `moves.encode` chain are the next-largest cumtime; `live_view.from_pokemon`
(rebuilt ×12/encode) is shared with reward/replay, so it carries a wider blast radius.

---

## Pitfalls that have caused regressions here

- **Calling `PokemonType.damage_multiplier` / `effective_multiplier(move_type, mon)` per cell.**
  Use the value-based `effective_multiplier_by_types(move_type, t1, t2, ability, status)` and
  read the mon's attributes once outside the loop. The object wrapper re-reads poke-env
  properties every call.
- **Re-reading poke-env properties inside the inner loop.** `move.type`, `mon.type_1/2`,
  `mon.ability`, `move.category` are properties that do real work (`move.entry`,
  `GenData.from_gen`); hoist them above the loop. The matchup matrices already do this
  (`reactive._defender_terms` / `_attacker_type_dist` read each mon / (attacker, move) once,
  not per cell) and the per-mon move category is memoized by id (`moves._category_val` — a
  process-global cache off the *live* `move.category`, NOT a `gen3_data.moves` re-derivation,
  which disagrees for fixed-power moves). Do not reintroduce a per-cell / per-slot property
  read.
- ~~**Breaking the turn-history deque cache** (`EpisodeTracker.prev_N_delta_vecs`)~~ — DELETED
  with the lag frames (`gen3_frame_deletion_v1`); kept struck through because the shape of the
  hazard recurs for any future memoized block. Historically: if the
  benchmark's "recompute all 10" multiplier collapses toward 1×, you've reintroduced the
  per-step O(N) re-encode.
- **Wrapping live mons in proxy objects** with `__getattr__` (the deleted
  `_AbilityOverrideMon`): `__getattr__` is slow and gets hit once per attribute per cell.

## ⚠️ A fuzz ORACLE binds to the layout too — by NAME, never by literal

The positional-binding sweep (2026-08-18) found two live misbinds on this directory's *readers*,
both silent, both in code whose job was to catch exactly this class:

- **`wish_floating_fuzz_test` read `OFFSET_REACTIVE + 17 / + 18`, and `REACTIVE_DIM` is 17** — so
  both literals pointed PAST the reactive block, at `OFFSET_PAIR_HISTORY + 0 / + 1`. The oracle
  compared its Wish expectation against a different block's (usually zero) content, so the
  completeness half could only ever read *"the encoder never floats a Wish."* They went stale the
  day `gen3_entity_rehome_v1` shrank the block. Fixed by resolving them from
  `ReactiveEncoder().get_layout()`; pinned by
  `reactive_test::test_the_wish_fuzz_reads_the_DECLARED_wish_columns`, which asserts the
  relationship (offset ↔ declared column ↔ inside the block bound), not a number.
- **`event_window_fuzz_test`'s independent fold guarded residual damage with
  `e.value.get("from")` on a DAMAGE event** — the key DAMAGE never carries (the parser writes the
  `[from]` clause to `value["reason"]` there and to `value["from"]` on the effect kinds). That is
  the *same* key drift the tracker was already fixed for, so the oracle could not have caught the
  bug coming back. Fixed to `e.from_clause`, behind the named `attributable_damage` predicate so
  the oracle is unit-testable against the trap
  (`event_window_test::test_the_fuzz_ORACLE_reads_the_from_clause_too`).

Same lesson, twice: **an oracle that mirrors its subject's key choice is not an independent
check.** Derive the oracle's addresses from the DECLARED layout (`get_layout()`,
`build_schema(layout).slices()`) or from the raw protocol — never from a literal, and never from
the consumer's own accessor.

The H-B event window's column-15 **status vocabulary** now lives here too, as
`constants.EVENT_STATUS_IDS` / `N_EVENT_STATUS`, for the reason `EVENT_T_*` does: it is the obs
contract, written by `episode_tracker` and embedded by `team_transformer.EventSeats`. The
producer CRASHES on an unrecognised status rather than coding it 0 = "none"
(`_event_status_id`, the `normalize_cant_reason` contract), and `EventSeats` asserts its table
covers the vocabulary and clamps from the table's own width — so growing the dict fails loud
instead of clamping a new id onto `tox`.

## The incremental cache (`assembler.py`) — what may and may not be cached

`ObsAssembler` is owned by the `EpisodeTracker` (so it resets with the episode and deep-copies
with a counterfactual arm) and is threaded into `encode(..., assembler=…)` by `Gen3Env` and the
inference player. **Every other caller passes nothing and gets the full rebuild** — which is also
the oracle. There is deliberately **no flag**: a launch flag would fork the obs path into two
long-lived variants and this tree has measured what happens to the branch nothing runs (the
seedless-seed lesson). The diagnostic escape hatch is `GEN3AI_OBS_VERIFY=1`, which shadow-encodes
both ways per decision and raises naming the offending block.

**Cached:** the twelve 122-dim per-mon slots, keyed by SPECIES (never by list position — the opp
team list grows as mons are revealed), and the encoded event-window rows.

**Never cached, and each for a stated reason:**

| block | why it is recomputed every decision |
|---|---|
| the two 58-dim active contexts | a switch clears boosts/volatiles with **no per-field event**, and a Baton Pass *keeps* them — "write zeros on SWITCH" is wrong in both directions |
| global / board (reactive) | cheap, and the board's Wish fold is now incremental anyway |
| the 180-dim pair history | every cell's `recency_of_last_pairing` ticks on every turn |
| per-mon recency triplets | same — turn-anchored, so they move under a mon that did nothing |
| `trapped` / `maybe_trapped` / `active` | request-sourced; a cached request bit that survives one decision too long is the `gen3_op_move_align_v1` misalignment class |
| BOTH actives' whole slots | unconditionally dirty — it costs ~2 slot encodes and shrinks the event→dirty map to the families that touch a BENCHED mon |

**Four dirty signals, and it takes all four** (the first three are the design's; the fourth is the
one the fuzz found):

1. the **event log** — `STATE_ONLY` is empty in gen3ou, so no protocol mutation bypasses it;
2. the **request**, per-mon (`StrictBattleView.request_change_seq`) — a `|request|` emits no event
   yet writes condition/item/ability/moves/stats. Per-mon because a request arrives every
   decision; an unchanged per-mon record *proves* no mutation, since `update_from_request` is a
   pure function of it;
3. **`HiddenPowerTracker.revision`** — 17 dims written by our own code, not by a line;
4. 🚨 **`|-cureteam|`** — `EventKind.CURESTATUS` covers two keywords and one is TEAM-WIDE (Heal
   Bell / Aromatherapy cures all six while naming only the active). This is the door the design's
   §2.2 map missed; the fuzz caught it as 11 stale status bits on benched opponents in 9,272
   decisions. Any CURESTATUS now dirties the whole side.

Two whole-log folds that used to run per encode — `build_wish_pending` and `build_sleep_sources`
— are incremental on the assembler. The full-fold functions **stay** and are what the non-cached
path (and the fuzz oracle) uses, so the two are compared rather than one reading the other back.

Gates: `assembler_test.py` (one named regression per design §2.3 trap; four of them scripted
because a random gen3ou corpus reports forme-change / Transform / partial-trap / Pain-Split as
NOT SEEN) and `training/poke_env_gaps/obs_assembler_fuzz_test.py` (real battles, byte-identity at
every decision, with a printed trigger census — a clean run that exercised no trap says so).

## Value-correctness (separate from perf, but also gated)

Changes to *what the vector contains* are validated by the bridge-backed fuzz tests
(`*_fuzz_test.py`, real battles, protocol-truth checks) and the unit tests in this directory.
If your change is meant to be **value-neutral**, prove it: the effectiveness fast-path, for
example, is pinned byte-for-byte by the exhaustive parity test in
`src/agents/gen3_mechanics_test.py`. Obs-value changes are retrain-class → bump
`ARCH_SIGNATURE`.

## Static typing (mypy)

This package is **type-checked at ZERO errors**, on the same config and the same strictness tier as
`src/agents/model/` — one `mypy.ini` at the repo root, one `files =` naming both. New code here must
pass before it lands:

```bash
# in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src
/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m mypy   # scope from mypy.ini; must be clean
```

The gate is `src/agents/model/mypy_gate_test.py`, which runs bare `python -m mypy` (no path) so the
scope is the config's, and separately asserts that `files =` still NAMES both packages — mypy exits
0 just as happily on a scope of nothing, and "checked less" is otherwise indistinguishable from
"everything is clean". **The two packages share one config, so a loosening to clear something here
silently de-tiers the model package.** Narrow with a targeted ignore instead.

**The obs layer's own idioms come first; types complement them, never replace them:**

- **The offset/layout constants discipline is unchanged.** A slice is still written from a named
  constant and read back through `get_layout()` — mypy types the *array*, not the *index*, so it
  cannot catch a wrong offset and must never be mistaken for a check that it does. The shape and
  block comments stay.
- **`np.ndarray` carries no shape.** Same rule as the model package's `[B, 6, K]` comments: the
  dimension lives in the comment and the `*_DIM` constant, the checker only knows "an array".
- **`typing.cast` at the `gen3_data` facade boundary**, where the facade's `.get()` is `Optional` but
  a guard has already proven presence. `incoming_damage_encoder` is the live case: `_is_damaging()`
  owns the not-None test, so no narrowing survives the call and the two use sites `cast` rather than
  re-test. `cast` returns its argument unchanged — the emitted vector is byte-identical. It IS a
  real function call, so keep it off the hottest loops; the current four per obs build are 0.0002%
  of the ~2.1M calls the benchmark counts and do not appear in the cProfile top-22.
- **`# type: ignore` always carries a code and a reason.** One cause dominates here and is worth
  knowing before reading one as a smell: **`ObservationEncoder.encode` still declares the pre-ai_v4
  `(item, battle)` signature**, while `StateEncoder` / `GlobalEnv` / `ActiveContext` / `Reactive`
  were migrated onto the LiveView read-model and take an entirely different subject. Nothing calls
  them through the base, so the divergence is declared at each override rather than paid for by
  widening the ABC to `*args` — which would delete the check for the encoders that DO conform. The
  same applies to the three compact-string `describe_vector` sub-encoders (types / items /
  abilities), whose output is embedded as a dict VALUE by `PokemonEncoder`.
- ⚠️ **A standalone comment that starts with `# type: ignore` IS a directive.** mypy parses it
  wherever it sits and rejects it as malformed, so an explanatory line above an ignore must not
  begin with those words — this file's convention is `# Why the \`type: ignore[...]\` below — …`.

**One latent mismatch is DECLARED rather than repaired**, at `incoming_damage_encoder._defender`:
`Defender.type1` is non-optional and `effective_multiplier_by_types` requires it, but the
expression yields `None` for a typeless `lm`. Believed unreachable (a `LivePokemon` always carries
≥1 type). It carries a `# type: ignore[arg-type]` naming itself; if it ever fires the fix belongs
in `_defender`, not in the annotation.

**Annotations are runtime-neutral, and this package's benchmark gate still applies to them.**
Measured over the typing pass (same-load A/B, busy box — absolute ms not comparable, ratios are):
`state_encoder.encode` 93% → 94% of the build, turn-history 9% → 9%, `live_view()` 22% → 20%, and
the cProfile top-22 ranking unchanged.

---

## Observation vector layout (per-block reference)

`designs/ARCHITECTURE.md` § Observation carries the top-level block table (block → dims → offset)
and the per-mon slot layout, derived from the live constants. **This** is the per-block detail:
what each field MEANS and where it is sourced from. All offsets are computed from named constants
— never hardcode indices.

**Per-Pokémon slot (122 dims):** the 110 below + the 3-dim recency block + the 1-dim
protect-odds field + the 6-dim last-action block + the 2 appended trapping bits + the
appended active flag.
**Recency block** at `POKEMON_RECENCY_OFFSET` (109) — [turns_since_seen, turns_since_acted,
turns_since_was_hit], TURN-ANCHORED (`cur_turn − event_turn`, clamped; on-field mon reads 0;
never-tracked reads 1.0 max staleness), log-saturated over a 10-turn cap, BOTH sides (public —
every reset derives from observed protocol events), sourced from the EpisodeTracker-owned
`RecencyTracker` (the same per-decision event window the TurnDelta fold reads) and threaded
into `encode(recency=…)` like the progress clock. Fuzz gate:
`poke_env_gaps/recency_fuzz_test.py` (encoded scalars == an independent full-log recount +
decision-time active log, per mon per decision). **Protect-odds field** at
`POKEMON_PROTECT_OFFSET` (112, gen3_entity_rehome_v1): P(a Protect/Detect/Endure by THIS mon
succeeds now) under the gen3 floored-doubling stall rule (100/50/25/12.5, floor 1/8), from the
LiveView `protect_counter` — EVERY mon owns its stall state (a benched mon truthfully reads 1.0;
the counter resets on switch). Pinned by `protect_success_prob_fuzz_test.py`. **Last-action block** at
`POKEMON_LAST_ACTION_OFFSET` (113, `gen3_pair_history_v1` — Tier H-A1 of
`designs/ai_v9/design_history_entity.md`): the SIDE's most recent executed action on its
ACTIVE mon's slot — `[last_move_id, was_switch, hit, miss, fail, crit]`, bench rows zero.
The move id is an EMBEDDING id (the model's `slice_pokemon_categoricals` routes it to the
move table and ZEROES its raw column — a dex num never reaches a Linear); outcome order
matches the turn-delta `_OUTCOME_ORDER`; CANT windows leave the previous action standing;
leads don't count (a placement, not an action). Folded by the EpisodeTracker-owned
`PairHistoryTracker` (same decision window as recency), threaded via
`encode(pair_history=…)`. Fuzz gate: `poke_env_gaps/pair_history_fuzz_test.py` (independent
full-log oracle; it caught a fainted-active-resurrection resync bug pre-ship). The SAME
tracker also feeds the **180-dim pair-history block** after reactive
(`OFFSET_PAIR_HISTORY`, 6×6×5 `h[i,j]` tendency counters — switch-ins/attacks/status-clicks
by their mon i while our mon j was active, shared-field turns, pairing recency; log-saturated
over the 10 cap; consumed by the opt-in `h` edge family). **Tier H-B follows it**
(`gen3_event_window_v1`, v81): the **event window** (`OFFSET_EVENT_WINDOW`,
`EVENT_WINDOW_N` × `EVENT_TOKEN_DIM` typed event records) closes base, and
**`gen3_frame_deletion_v1` made it the LAST block**:
the 11-dim prev-turn action mask and the 7 × 159 TurnDelta lag frames that used to follow are
DELETED, so `total_dim == base_dim` and `encode`'s output IS the observation. The window grew a
`cant_id` column in the same pass — the one lag-frame fact with no substitute. What that
deletion cost, and the three facts that ship WITHOUT a substitute, is
`designs/ai_v9/design_frame_deletion_coverage_gaps.md`. Folded by the EpisodeTracker-owned
`EventWindowTracker` (same window, same alive-filtered resync), threaded via
`encode(event_window=…)`; rows most-recent-LAST, front zero-padding; ids are embedding ids and
NO Linear reads the block raw (its only consumer is the opt-in `--history-events` event seats).

> ⚠️ **Feeding it takes the FULL three-step decision protocol** — `record` →
> `update_progress_clock` → `encode(event_window=…)`, in that order. `update_progress_clock` is
> the ONLY caller of `EventWindowTracker.update`, and `encode`'s `event_window=` is optional
> (`None` leaves the block zero), so a harness that skips either reads a structurally-ZERO
> block — which a presence check on any single row type reports as "the signal never reached
> the model", indistinguishable from a real miss. That is exactly how the trapping fuzz read
> FAIL 4/4 on a signal production delivers; `event_window_test::
> test_the_window_block_is_ZERO_without_update_progress_clock` pins the trap by name.
>
> An OUT-OF-BAND event is covered by the same window: `CHOICE_REJECTED` is the one kind
> recorded outside the parse pass (poke-env intercepts `|error|[Unavailable choice]` before
> `parse_message` and calls `Gen3Battle.record_choice_rejected`), and it still lands inside the
> NEXT decision's `[cursor, now)` slice, because the cursor is captured at `record()` time
> against the same log `_record` appends to. Pinned end to end by `event_window_test::
> test_an_out_of_band_choice_rejection_reaches_the_NEXT_decisions_obs` (and its CANT sibling,
> which asserts the ordinary parse-pass path has no such exposure).

> **The per-row COLUMN CONTRACT is `constants.EventCol`** (`gen3_event_col_names_v1`) — an
> `IntEnum`, ONE declaration that BOTH ends import: the producer (`state_encoder.encode`) and
> the consumer (`team_transformer.EventSeats.forward` + `_event_reference_cells`), plus the
> feature-coverage probe helper (`feature_coverage/_support.py::obs_with_event_row`) and every
> oracle that reads the block (`event_window_fuzz_test`, `trapping_signals_fuzz_test`,
> `hidden_power_typed_obs_fuzz_test`). It replaced a comment plus ~30 bare integer literals
> spread across five files — a producer/consumer pair bound by POSITION with nothing relating
> them, the class the 2026-08-18 positional-binding sweep convicted five times. **Never write a
> bare column index**; the members ARE ints, so `vec[_o + EventCol.CRIT]` is the same arithmetic
> and the same emitted bytes (byte-identity confirmed on the 991-decision golden capture).
> The two CONTIGUOUS one-hot groups have their own names (`EVENT_OUTCOME_GROUP`,
> `EVENT_EFF_GROUP`) because both are written by INDEXING (`EFF_NEUTRAL + eff`), so their order
> is load-bearing: reordering a member relabels every historical row with no shape change.
> `event_window_test.py` pins all of it — the members TILE `range(EVENT_TOKEN_DIM)` with no gaps
> or overlaps, the groups are contiguous and in the TurnDelta effectiveness-code order, the two
> ends resolve to the SAME object, and `EventSeats._N_SCALARS` (a weight shape) agrees with the
> map's id/scalar classification.
>
> **The event-window fuzz oracle models ALL 22 columns** — `_ORACLE_UNMODELED_COLS` is EMPTY,
> and the coverage assert keeps it that way (a new `EventCol` member must be modelled or
> declared, never silently unchecked). Two rounds got it here. First the three id columns were
> found UNCHECKED with nothing saying so (`_want_vec` returned a 19-tuple compared with `zip`
> against a 22-wide row, and `zip` stops at the shorter) and were declared unmodelled. Then the
> modelling landed: the oracle emits its own CANT row and derives the faint CAUSE and item
> TRANSITION — the semantic input independently (which event, which mon, and its own ledger of
> what last damaged each side / whether that side self-KO'd, cleared when a mon leaves the
> field), the label→id step through the declared vocabulary both sides must share.
>
> ⚠️ **The missing CANT ROW was the expensive half, and the failure shape is worth knowing.**
> One fewer record per `|cant|` than the tracker is invisible until the 32-row window
> SATURATES — after that the oracle's last-32 starts earlier in the timeline than the
> tracker's, so EVERY row compares against its neighbour: 8209 failures over 5 battles, one
> root. An independent fold must match the producer's row COUNT, not just its column values.

Unit gate: `training/event_window_test.py`; the event-fold FUZZ (the pair-history pattern) is
the pre-enable gate. **Appended tail**
(state_encoder): `POKEMON_TRAPPED_OFFSET` (119) + `POKEMON_MAYBE_TRAPPED_OFFSET` (120) — the
OUR-side LegalActions trapping bits, nonzero ONLY at our active slot (`maybe_trapped` is the
high-value trap-risk bit; fuzz gate `action/trapping_signals_fuzz_test.py`, which also asserts
bench slots stay zero) — then the ACTIVE flag at `POKEMON_ACTIVE_OFFSET` (121), deliberately
LAST in the slot (the model's `hp_and_active[:, :, -1]` convention is load-bearing).
Original 110: species ID + 6 base stats, item ID + known + consumed, 2 type
IDs, ability ID + known, 7-dim condition (status one-hot), 4 × 11-dim move slots, HP fraction,
species_known flag, sleep_counter_norm, toxic_counter_norm, **spread block (18 dims)**,
**HP-candidate block (17 dims)**, **sleep-wake belief (3 dims)**. The item block is 3 dims:
`[item_id, known, consumed]` — `consumed=1` when the item was spent this battle (Berry
activated, Knock Off, Trick, etc.) and `item_id` retains the identity of the consumed item so
the model knows what was lost. `species_known = 1.0` for all populated slots (own team and
revealed opponent mons), `0.0` for unseen opponent slots. Sleep counter:
`min(turns_slept, 4) / 4` (Gen 3 max 4 turns); toxic counter: `min(turns_poisoned, 8) / 8`
(practical max before fainting with Leftovers).

**Sleep-wake belief (3 dims, `gen3_sleep_wake_belief_v1`, layout in `sleep_belief.py`):** zeros
unless the mon is asleep, else `[sleep_is_deterministic, p_wake, sleep_counter_reliable]`. poke-env
exposes only `Status.SLP` + a noisy `status_counter`, NOT the rolled duration / remaining time / source
move, so a policy reading the raw counter would have to LEARN the gen3 sleep RNG and can't tell a
deterministic Rest from a random opp-sleep at the same counter. We **compute** the wake odds from the
adversarially-verified gen3 tables — opp `time = random(2,6)` ∈ {2,3,4,5} (the gen3 mod overrides the
modern `random(2,5)`), Rest `time = 3` fixed, Early Bird halves — `p_wake` = P(wake on the next move
attempt | observed counter K, source, Early Bird), **marginalising the opponent's Smogon Early-Bird
prior** (collapsing to exact 0/1 for our own mon or a revealed opp). `sleep_is_deterministic` (1.0 =
Rest) selects which table; it's read from our **event log's `[from]` clause** (poke-env discards it).
`sleep_counter_reliable` drops to 0.0 once a Sleep Talk / Snore turn has corrupted the counter (+3 per
turn, empirically verified) — instead of reconstructing Showdown's `skippedTime` switch refund. The
counter→p_wake mapping and the source/reliability bits are **fuzz-calibrated against the real sim RNG**
(`poke_env_gaps/sleep_wake_fuzz_test.py`: per-decision obs wiring exact + empirical wake-frequency ==
the computed table across well-sampled (K, source) buckets).

**Move slot (11 dims, layout in `moves.py`):** move ID, base power (/200), has_secondary,
has_recoil, type ID, category (0=status, 1=physical, 2=special), known flag, current PP
(/MAX_PP), max PP (/MAX_PP), accuracy (raw% / 100), never_miss bit. Accuracy is split into a
continuous scalar plus a categorical bit: never-miss moves carry accuracy=100 in the mapping →
encode as `[1.0, 1]`, while a genuine 100%-accuracy move is `[1.0, 0]` — same scalar,
distinguished only by the bit. A 100%-accuracy move can still miss into evasion (Double Team)
or after Sand-Attack; a never-miss move (Swift, Aerial Ace, all status/self moves) bypasses the
accuracy/evasion check entirely.

**Spread block (18 dims, appended at offset 71 within each slot):** IVs ×6 each/31 + EVs ×6
each/252 + spread_known (1.0 own, 0.0 opp) + nature modifiers ×5 [atk, def, spa, spd, spe] as
raw floats (0.9/1.0/1.1). Opponent slots have all 18 dims as zeros; `spread_known=0`
distinguishes "unknown opponent" from "own Pokémon with 0 EVs". Own-team `mon.ivs/evs/nature`
are populated by the poke-env fork's **`backfill_teambuilder_spread`** (`Battle.parse_request`):
gen3ou has no team preview, so `apply_teambuilder_team` never attaches the spread — the backfill
matches the declared teambuilder team to the request-built team by species and fills in
IVs/EVs/nature (spread only, never re-running `_update_from_teambuilder`). Without it this block
emitted a constant fallback (all-31 IVs, 0 EVs, neutral nature) for every own mon.
**Board (reactive) block — 17 dims, layout in `reactive.py`.** `REACTIVE_SCALAR_DIM` (5) raw
board scalars, then the 12-dim active-req-moves block. Offsets are `reactive_layout` entries —
read them, never hardcode. **gen3_entity_rehome_v1**: the two 144-dim matchup matrices are
DELETED (pair effectiveness is GPU-side — the D/V edge families compute
`[low, high, crit, pko, type_mult, revealed]` cells from real physics + the learned belief);
`active_status` (redundant with the per-mon condition one-hot) and `forced_struggle` (derivable
from the all-zero `active_req_moves` legal bits / the action mask) are deleted;
protect/trapped/maybe_trapped moved to the per-mon slots (above).

| Field | Offset | Dims |
|---|---|---|
| `fainted` (ours, theirs) | 0 | 2 |
| `turns_since_progress` | 2 | 1 |
| `wish_floating_our` / `wish_floating_opp` | 3 / 4 | 2 |
| `active_req_moves` | 5 | 12 |

The 5 scalars sit BEFORE `active_req_moves`, so the extractor picks them up in
`non_matchup_rest` automatically (it stops at the req-moves offset). Sources:

- `turns_since_progress` — the log-saturated no-progress clock (`log(1+min(n,10))/log(11)`), owned by
  the **EpisodeTracker's `ProgressClock`** (NOT LiveView — it is cross-turn state) and threaded into
  `encode()` like the HP tracker. The reward's `no_progress_tax` keys on the SAME clock instance, so
  obs and reward-key are one value. Lets the model state-condition on the penalty it is about to be
  charged.
**Do not confuse `turns_since_progress` with the DEADLINE clock** — they answer different
questions and live in different blocks. `turns_since_progress` (board block, above) is a
*resettable* stall counter: it measures how long since anything productive happened and it goes
back to 0 the moment it does. The forfeit deadline is the GLOBAL block's `clock` group
(`gen3_deadline_clock_v1`, `CLOCK_DIM` = 3): `[log_elapsed, remaining_linear, log_remaining]`,
where `remaining = MAX_TURNS − turn` clamped at 0 and `MAX_TURNS` (250) is the turn the trainee
actually forfeits on (`StallConfig.threshold` imports it — pinned by
`global_env_test.py::test_max_turns_is_the_forfeit_deadline`). Only the global group tells the
model how much game is LEFT; a reset progress clock says nothing about the cap.

The log-REMAINING channel exists because the old lone log-ELAPSED scalar put **58.6%** of its
range on turns 1–50 and **1.5%** across the last 20 — a 125× sensitivity gap at exactly the cliff
the critic has to price. Measured consequence before the fix (`ai_v9_09` @16M): a POSITIVE V(s)
on the final decision before a −30 forfeit in **13 of 14** timeout losses. log-remaining gives
those last 20 turns 55.1% of its range. Both remaining forms are raw facts, not a choice made for
the model.

- `wish_floating` — the pending-Wish heal: a flat `WISH_HEAL_FRACTION` (≈0.5; gen3 Wish heals the
  RECIPIENT's maxhp/2, so the fraction is constant and GIGO-proof) when a Wish cast last turn
  resolves at the end of this turn, else 0. Slot-keyed, so it survives faint / Roar-phaze / switch.
  poke-env tracks none of it → reconstructed from the event log (`wish_belief.py`).

**`active_req_moves` (12 dims):** OUR active mon's 4 moves in **REQUEST order** (slot *k* ↔ action
logit 6+*k*) — `[move_num ×4, resolved_type_id ×4, legal_now ×4]`, sourced from `legal.move_slots`,
the same source as the action mask. The `DamageOperator`'s OUTGOING per-move methods read THIS, so
their per-move output aligns with the action order instead of the per-mon block's sorted-by-id
order. `move_num` is the dex num (the opponent's HP stays bare 237; our own typed HP resolves);
`legal_now` is the current-decision choosability. These are embedding IDs, not scalars, so the block
sits AFTER the matchups and is EXCLUDED from `non_matchup_rest` — `ObsUnpack` slices it explicitly
into `ctx.our_active_req_move_{ids,type_ids,legal}` and it never enters the raw-scalar path.

**The per-mon move block stays sorted-by-id on purpose** — it feeds the role token, whose value is
order-sensitive (the 4 move encodings are concatenated), so it cannot be reordered without changing
the network. Both orders are therefore live at once; the pointer action head resolves this by
permuting on move-num IDENTITY, which makes a misaligned logit unrepresentable.

**Move-effect block and incoming-damage / OHKO belief block — BOTH DELETED from the observation.**
Their long descriptions moved verbatim to `designs/CHANGELOG.md` §5. Where the signal lives now:

| Deleted obs block | Dims | GPU home |
|---|---|---|
| action-aligned move-effect flags | 44 | static mechanics → the `MoveLatentEncoder` latent (`--move-latent`); board-conditional `status_will_land` → `DamageOperator._status_landing` (`--damage-outgoing`); `pp_fraction` → the per-mon move slot (unchanged) |
| per-our-mon incoming-damage / OHKO belief | 51 | the `DamageOperator`'s incoming block, off the LEARNED move belief instead of this block's FIXED usage prior — that substitution was the whole point of `--damage-op` |
| active-move scalars (base power ×4, type mult ×4) | 8 | the op's OUTGOING per-move block, request-ordered, with real gen3 physics rather than `bp/200` and `mult/4` |

**`agents/observation/incoming_damage.py` STAYS** — the reward PBRS (`reward_manager.py`) and the
prober import its math core, and its fuzz test now targets `encode_block` directly. Only the obs
write was removed.
> **Downstream reader:** the prober engine (`src/main/prober/engine.py`) resolves its obs
> offsets at runtime from `get_layout()` (`ObsOffsets`), with `0 = absent` for deleted blocks
> (`mm_off`, and since gen3_entity_rehome_v1 also `om_off`/`tm_off` — ThreatView/saliency
> no-op). Its pinned regression test (`prober/engine_test.py::
> test_offsets_resolve_matches_layout`) fails on any layout move; update the pins there.

**TurnDelta slot (159 dims, layout in `turn_delta_encoder.py`):** all offsets computed from
named `OFFSET_*` / `*_DIM` constants — never hardcode indices. TurnDelta is **folded from the
event log** (`Gen3Battle.events_since(cursor)` per decision window; see
`src/agents/battle/CLAUDE.md`) rather than diff-heuristics.

- **Base block (53 dims, indices 0–52)** — our/opp move features (5 each: raw move_id int,
  power_norm, has_secondary, has_recoil, raw type_id int — **our OWN Hidden Power carries its DISTINCT
  num + real type** here, `gen3_typed_hidden_power_ids_v1`: the fold restores the typed HP id from the
  decision-time `LegalActions.own_hp_typed_id`, which maps to its distinct dex num (355-370) so
  `_move_features` takes the typed-dex branch [real BP/type]; the **opponent's** HP stays bare num 237 /
  type-0, correctly unknown), switched/failed flags, cant onehots
  (`CANT_DIM` = 12 ea, from `gen3_effects.CANT_REASONS`:
  slp/frz/par/flinch/recharge/attract/disable/taunt/imprison/focuspunch/nopp/truant —
  source-derived, crash-don't-drop enforced in the encoder), summed HP deltas, faint flags,
  opp_move_known, effectiveness onehots (4 ea), move-order (2).
- **Extended block (106 dims, indices 53–158)** — our/opp boost deltas (7 each);
  `phase_is_forced_switch` (1); our/opp `target_hp_delta` (1 each); per-side HP-level vectors
  (6 each); our/opp target_status onehots (7 each, at move-fire time); our/opp move-outcome
  onehots (3 each: `[hit, miss, fail]`); our/opp move-crit (1 each); **gen3_turn_delta_v2
  additions**: `our_faint_causes`/`opp_faint_causes` (8 each, multi-hot over
  `attack/hazard/weather/status/recoil/selfko/leechseed/other`); **status-transition onehots**
  (4 × 7): `our/opp_status_applied` (status GAINED this window) + `our/opp_status_cured` (status
  LOST this window); **item-used bits** (2): `our/opp_item_used` — a single bit per side marking
  an item was consumed/removed this window (Berry/Knock Off/Trick). These (status transitions +
  item-used) are the per-turn *events*; the cause-**identity** (which item, which ability) lives
  in the per-mon item/ability block — the history carries the event, the block carries the what
  (parity with the collapsed `ability_activated` volatile). `our_attempted_move_id` (1, raw int
  embedded — the move we PRESSED, even if it never fired); **species block** (6 raw ints,
  embedded): `our_actor`/`opp_actor`/`our_target`/`opp_target`/`our_switch_to`/`opp_switch_to`;
  **gen3_trapping_signals_v1 additions**: `attempted_switch_rejected` (1 bit — the server
  REFUSED a switch we chose this window, `|error|[Unavailable choice]`, i.e. we tried to pivot
  and got trapped) + `our_attempted_switch_to` (1 raw int embedded — the mon we PRESSED a switch
  to).

`our_faint_causes` / `opp_faint_causes`: multi-hot — the rare 2-on-one-side window
(Pursuit/Future-Sight into a low-HP mon on hazards) can set 2 bits; the common Explosion
double-KO is one bit each side. Cause derived from the DAMAGE event's `[from]` clause
immediately preceding the FAINT in the event log. All-zeros when no faint.
`our_attempted_move_id`: decoded from the pressed action index at build time, preserved even
when the move never fired (cant / frozen / KO-before-acting). `attempted_switch_rejected` /
`our_attempted_switch_to` (gen3_trapping_signals_v1): the rejected-pivot history — folded from
the out-of-band `CHOICE_REJECTED` event (`TurnView.attempted_rejected`). On a rejected pivot
`our_switch_to` is the unknown sentinel (the switch never happened) while `attempted_switch_to`
names the mon we tried to bring in; both are zero on every turn with no rejection. Opp attempted
action is not observable. Faint *counts* are kept on the `TurnDelta` dataclass (for reward) but
not encoded — redundant with the faint flags + cause popcount.

**Embedded-ID manifest (the layout-driven contract):** which slot positions carry raw embedding
IDs and which table each routes to is declared once in `TURN_DELTA_EMBEDDED_IDS` (in
`turn_delta_encoder.py`). Both the encoder's layout and `Embeddings.embed_delta_slot` read it —
there are **no hardcoded positions in the extractor** and the embed-width formula derives from
the manifest, so a raw id can never silently leak through as a scalar. Adding an embedded ID is
a one-line manifest change. Current manifest: 3 move IDs (our/opp/attempted) → 3×16, 2 type IDs
→ 2×16, 7 species IDs (our/opp actor + our/opp target + our/opp switch_to + attempted_switch_to)
→ 7×32 = 48 + 32 + 224 = 304 embedded dims + 147 pass-through scalars = 451-dim per slot.
Positional encodings are added, one self-attention pass runs, and the last (most-recent) slot's
output flows into the projection block. All zeros on the first turn of each episode. Actor
species resolution prefers `damaging_event.user_species` (protocol-truth) and falls back to
`prev_active` for switches and non-damaging moves; target species comes from the OTHER side's
`damaging_event.target_species`. Species ID 0 is the unknown sentinel.
