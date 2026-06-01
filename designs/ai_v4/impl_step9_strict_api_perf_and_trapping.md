# Implementation: Step 9 — Strict-API Completion, Performance Pass, Trapping Signals

This is the as-built record of everything that landed **after** `impl_step8` on the
event-sourced battle layer. Three independent threads, documented together because they
close out ai_v4: (1) finishing the strict battle-API migration and the event-sourced
`TurnDelta` fold, (2) a ~2× obs-build performance pass, and (3) routing the trapping
signals into the observation.

| Thread | Commits | What landed | Arch impact |
|---|---|---|---|
| A — **Strict-API completion** | `b1d49af` `3418c26` `182b499` `8b41bb4` `0b23cd6` `021f2d3` `d56942d` `26d3d1e` `6f3c59f` `90739dd` `e1d9a94` | Phases 3–4 consumer migration + the `agents.enums` seam + the `strict_api_lock_test.py` lock; the TurnDelta full event-fold with `battle_context.py` deleted; reward `_read_live` retirement; legacy builder extraction. | `gen3_turn_delta_v2 → gen3_turn_delta_v3` (history-window fix, obs 3299 unchanged); the rest **value-neutral, no bump** |
| B — **Performance pass** | `ca5bac7` `96bfc3a` `63220dd` | Reactive matchup chart + `lru_cache`, turn-history deque memoization, matchup per-mon read hoisting + move-category memoization. | **none** — obs byte-identical, no `ARCH_SIGNATURE` bump |
| C — **Trapping signals** | `2f29240` | `trapped` / `maybe_trapped` reactive obs bits + a `CHOICE_REJECTED` rejected-switch history bit + restored `attempted_switch_to`. | `gen3_turn_delta_v3 → gen3_trapping_signals_v1` (obs **3299 → 3321**) |

Net at HEAD: `ARCH_SIGNATURE = gen3_trapping_signals_v1`, obs **3321-dim**,
`TURN_DELTA_DIM = 159`, `REACTIVE_DIM = 302`. The dim and signature chain in this doc is
the authoritative as-built record; numbers were queried from the live
`Gen3ObservationEncoder` / constants, not copied from another doc.

---

## Thread A — Strict battle-API completion (Phases 3–5a)

`impl_step8` landed Phases 1–2 (the widened `LiveView`, the new `LegalActions`, the
`StrictBattleView` boundary) but left every consumer still reading the raw poke-env
`Battle`/`Pokemon`. Thread A migrates them all, locks the boundary shut, and finishes the
event-sourced `TurnDelta` fold — retiring `battle_context.py` entirely.

### Phase 3 — migrate consumers (Wave 1, value-neutral)

Five file-disjoint tracks, each gated by an existing value-neutrality harness so the
migration changes *where* a value comes from, never *what* it is:

- **`observation/`** (`3418c26`) — `state_encoder` builds the read-model once via
  `battle.strict_view().live` and threads the matching `LivePokemon` into each sub-encoder
  (`pokemon`/`species`/`items`/`types`/`abilities`/`active_context`/`global_env`). The dead
  `_own_hp_type_index` (the last `mon.moves` read) was removed — the live request re-keys
  typed HP to bare `hiddenpower`, so that one-hot resolved to `None` on every real decision
  (17208/17208), making removal byte-identical. The effectiveness hot-loop in `reactive.py`
  deliberately stays on the raw battle (typed-HP id + enum-keyed loop, pinned by
  `alignment_test`). Byte-identical over 29.4k encodes.
- **`training/` display/replay/control-flow** (`182b499`) — `gen3_env`, `inference/player`,
  `stall` read meta via `strict_view()`; `battle_recorder` reads current-board state only
  through `LiveView` (the stale-`last_move` opponent fallback dropped — history now comes
  from the event-log `TurnDelta`). `replay_recorder.py` was **retired** (`664f3a5`) in
  favour of quota-gated eval forensic traces (Step 6's "lossless replay" goal, achieved a
  different way — see below).
- **`episode_tracker.py`** (`8b41bb4`) — current-board reads via `LiveView`; the HP-rule-out
  scan and Hidden-Power target lookup reconstruct `PokemonType` enums from `LivePokemon`.
  Event-log calls stay on the raw battle.
- **`action/`** (`021f2d3` + `0b23cd6`) — the masker/mapper read the `LegalActions` snapshot
  (`mask_from_legal`, `action_to_choice`); `action_to_order` split into the pure
  `action_to_choice(legal) → Choice` + the lone `serialize.choice_to_order(choice, battle)`
  adapter; the `_gen3_decision_context` stash deleted; struggle single-sourced via
  `legal.struggle`. An end-to-end send-conformance round-trip caught a latent reverse-map bug.
- **reward activation logic** (`d56942d`) — switch/pivot/attack helpers read current-board
  facts through `LiveView`, with move power/type from a new **`gen3_movedex`** concept module
  and effectiveness from `gen3_mechanics.effective_multiplier_by_types`. Equivalence harness
  0 diffs over 11,477 turns. (`e35df88` added `gen3_movedex` + an event-sourced switch-subsidy
  attribution fix in the same window.)

### Phase 4 — the lock (`6f3c59f`)

Two parts, the actual enforcement now that Phase 3 removed the raw reads:

- **4a — `agents.enums` seam.** `src/agents/enums.py` re-exports **only** the four accepted
  poke-env value-enums (`PokemonType`, `Status`, `MoveCategory`, `Weather`); `Effect` is
  excluded (replaced by `observation/gen3_effects.py`). Pure import-path indirection
  (`agents.enums.PokemonType is poke_env….PokemonType`). Every consumer under `src/agents/**`
  (outside `battle/`) now imports those four from the seam; `enums_test.py` pins `__all__`.
- **4b — `strict_api_lock_test.py`.** An **AST walk** over the production consumer modules
  fails CI on (a) a raw stateful `battle.<attr>` read (the 13 temporal fields; meta scalars
  are not guarded — the strict view re-exposes them verbatim) and (b) a `from poke_env …`
  import of one of the four value-enums. A small, inline-commented per-`(file, attr)`
  allowlist covers the intended residual seams (the effectiveness hot-loop + `base.py`, the
  `opponent_active_pokemon` identity check); self-tests prevent a vacuous pass.

### Phase 5a — TurnDelta full event-fold + `battle_context.py` deleted

Two commits:

- **`b1d49af` (history-window correctness fix).** `prev_N_delta_vecs` folded each of the N
  history slots over `events_since(cursor)` — that turn's cursor *through now* (no upper
  bound) — so every slot but the most-recent reported the **latest** turn's event-derived
  fields, and the per-step cost was O(N²). Now each slot folds exactly its own decision
  window (`events_between(cursors[-1-i], cursors[-i])`, `end=None` for the most-recent), so
  older slots carry their own turn and per-step event-folding drops ~5.5×. Same commit added
  an **incremental weather fold** (running weather state updated per `WEATHER` event, read
  O(1) by `LiveView` instead of rescanning the whole log) and **bounded-deque** episode
  history (caps the 250-turn-stall memory). `ARCH_SIGNATURE gen3_turn_delta_v2 →
  gen3_turn_delta_v3` — obs dim unchanged at 3299, only the older turn-history slot *values*
  change, so retrain-class (not weight-shape-incompatible).
- **`90739dd` (the fold + the deletion).** `TurnDelta` now folds **entirely** from the event
  log + `LiveView` on every production path; the diff-based heuristic detective and the
  `BattleContext` snapshot-diff layer are gone:
  - **Relocated.** `TurnDelta` → fold-only `training/turn_delta.py`; the per-decision snapshot
    `BattleContext` → `training/battle_snapshot.py`; **`training/battle_context.py` deleted**
    (~20 importers updated; the `strict_api_lock` documented-seam exclusion moved
    `battle_context.py` → `battle_snapshot.py`).
  - **One production builder.** `build_from_events(prev, curr, action, events)` on the training
    env, `episode_tracker`, `reward_tracker` (now captures the per-decision event cursor), and
    the forensic `battle_recorder`.
  - **What folds from the log:** moves/switches/cant/effectiveness/faints+causes/
    status-transitions/item-lost/outcome/crit/move-order, the **per-slot HP delta** (each
    `DAMAGE/HEAL/SETHP` `hp_after` + `FAINT`→0 — bit-identical to `curr_hp − prev_hp`, no
    float-sum noise), and **target-HP**.
  - **Findings (surfaced, not papered over).** A per-event *amount-sum* is NOT value-identical
    (float accumulation order flips the discrete `opp_hp_delta.sum() >= 0` futile-attack
    threshold by ~6e-8) — folding the **end HP** and subtracting `prev_hp` once is bit-exact
    instead. Self-KO Explosion/Selfdestruct emits NO `-damage` on the user (HP→0 from the
    `FAINT`). The **boost-stage delta** is not event-foldable (`SETBOOST`/`clearboost`/`invert`/
    `copy`/`swap` carry only an `op`, no realized stage amount) and **HP-after** is
    intrinsically current-board — both stay LiveView snapshot reads. **Obs byte-identical → no
    `ARCH_SIGNATURE` bump.**
- **`26d3d1e` (reward `_read_live` retired).** Post-fold the live view is always available, so
  `process_turn_reward` builds `live = battle.live_view()` once and every per-term helper reads
  only through it. The `_read_live=False` raw-battle fallback, the dead `compute_base_reward`,
  and the 5 reward allowlist entries were deleted — `reward_manager` now has **zero guarded raw
  reads** (only `battle.turn` stall-tax + `report_episode` meta remain). The moot equivalence
  harness became the single-path `reward_value_regression_fuzz_test.py`.
- **`e1d9a94` (legacy builder extracted).** The retired snapshot-diff `TurnDelta.build` (no
  production callers but ~50 fuzz/unit call-sites) moved with its four legacy-only helpers
  (`_moves_match`/`_align_effectiveness`/`_ko_before_acting`/`_derive_move_outcome`) into the
  test-support module `training/turn_delta_legacy.py` as `build_legacy(...)`. `turn_delta.py`
  shrank 877 → 567 lines and carries only the live fold path; the two shared helpers
  (`_fold_hp_deltas`/`_resolve_target_hp_delta`, also called by `build_from_events`) stayed.

### Phase 5b (the lone open strict-API item)

Only the **`LiveView` current-board event-fold** remains — making poke-env's tracker no
longer the source of current state (side-condition counters, ability/item inference,
type-change edge cases). FUTURE, not scheduled; behind the now-strict boundary so it needs no
consumer churn. Tracked in `todo_live_battle.md`.

---

## Thread B — Performance pass (~2× obs-build speedup, obs-neutral)

The obs encoder runs once per decision across every training env, so it sits on the FPS
critical path. Three commits halved its per-encode call count without changing a single
emitted value:

- **`ca5bac7`** — the reactive matchup effectiveness moved to a memoized chart lookup
  (`effective_multiplier_by_types` + `_eff_cached` `lru_cache` in `gen3_mechanics.py`),
  replacing the per-cell `PokemonType.damage_multiplier` object-property path; added the
  encoder perf gate.
- **`96bfc3a`** — deque-memoized the turn-history deltas (`EpisodeTracker.prev_N_delta_vecs`
  caches the per-slot encode instead of re-encoding all N every step) + added
  `obs_build_benchmark.py` and the fuzz.
- **`63220dd`** — hoisted the per-cell poke-env reads out of the 288-cell matchup loop to team
  level (`reactive._attacker_type_dist` per `(attacker, move)`, `reactive._defender_terms` per
  defender mon, `_joint_expectation` per cell), so the `type_1/type_2/status/ability` reads +
  ability distribution happen once per mon instead of on all 288 cells; memoized the per-mon
  move-category scalar by id (`moves._category_val`, off the **live** `move.category`, NOT a
  `gen3_movedex` re-derivation which disagrees for fixed-power moves). `move.entry` dropped from
  ~158k to ~43k calls/encode; `pokemon.ability` / `move.type` left the cProfile top list.

**Result (load-stable signal):** **~12.8k → ~6.36k calls/encode** (≈2×, −49%). Obs-neutral —
proven byte-identical by `alignment_test` + `turn_history_fuzz` (18,791 decisions, mismatch=0)
+ the unit suite; **no `ARCH_SIGNATURE` bump**. The canonical before/after baseline and the
load-stable regression criteria live in `src/agents/observation/CLAUDE.md`. There is no longer
a single dominant hot loop — cost is spread across the memoized chart lookup (the irreducible
per-cell core), the matchup loop overhead, and the per-mon encoders.

---

## Thread C — Trapping signals (`2f29240`, retrain-class)

Trapping (Arena Trap / Shadow Tag / Magnet Pull / Mean Look) is hidden-information play the
model was almost blind to: confirmed `trapped` reached it only as a masked logit,
`maybe_trapped` not at all, and a **rejected** switch (`|error|[Unavailable choice]`) was
silently lost. Three signals fix that, bundled under one `ARCH_SIGNATURE` so there is a single
retrain boundary:

1. **`trapped` reactive obs bit** — a scalar from the per-decision `LegalActions` snapshot
   (`legal.trapped`). Redundant with the mask (switch bits already zeroed) but explicit.
2. **`maybe_trapped` reactive obs bit** — the high-value one: switches stay legal there, so
   this is the only way the model sees the trap risk before attempting a blind pivot. Both bits
   sit **before** the matchups in the reactive block so the extractor picks them up in
   `non_matchup_rest`. `REACTIVE_DIM 300 → 302`. `legal` is threaded through
   `state_encoder.encode` (env/inference pass the already-built snapshot; eval/standalone
   derive it from the strict view).
3. **`attempted_switch_rejected` history bit** — the rejected pivot becomes a first-class,
   learnable history event. A new `EventKind.CHOICE_REJECTED` is recorded **out-of-band**
   (poke-env intercepts `|error|` *before* `parse_message`, so a duck-typed hook in
   `_handle_battle_message` calls `Gen3Battle.record_choice_rejected`); `TurnView` folds it
   (`attempted_rejected`); `TurnDelta` gains `attempted_switch_rejected` + the **restored**
   `attempted_switch_to` (dropped earlier on the false "switches always execute" assumption —
   false exactly in the trap-reveal case). Each TurnDelta slot gains 2 dims: the bit + the
   embedded attempted-switch species id (manifest entry #12). `TURN_DELTA_DIM 157 → 159`.

`ARCH_SIGNATURE gen3_turn_delta_v3 → gen3_trapping_signals_v1`; obs **3299 → 3321** (+2
reactive, + `N_HISTORY_TURNS × 2` history). `describe_vector` recovers all three. The permanent
`action/trapping_signals_fuzz_test.py` (Dugtrio + Arena Trap vs a grounded mon that always
tries to switch) validates all three at their absolute full-obs indices against an independent
raw-protocol re-derivation.

---

## Resulting observation vector (`gen3_trapping_signals_v1`, 3321-dim)

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 107) | 642 | 0 |
| Opp team (6 × 107) | 642 | 642 |
| Active context ×2 (`VOLATILE_DIM` = 44) | 116 | 1284 |
| Global env | 18 | 1400 |
| Reactive + matchups (`REACTIVE_DIM` = 302) | 302 | 1418 |
| Prev-turn action mask | 11 | 1720 |
| Turn history (`N_HISTORY_TURNS` 10 × `TURN_DELTA_DIM` 159) | 1590 | 1731 |
| **Total** | **3321** | |

Field-level layout is in the root `CLAUDE.md` "Observation Vector" section (kept current).

---

## Verification

- **Strict-API.** Full unit suite green at each wave (`6f3c59f` 1201 passed/2 skipped;
  `90739dd` fold-equivalence fuzz **158,326 decisions, 0 field diffs, 0 reward diffs**;
  `e1d9a94` 1228 passed, fold-equivalence 25,851 decisions 0 real diffs). `strict_api_lock`
  detectors confirmed to flag injected violations (not a vacuous pass).
- **Performance.** `alignment_test` byte-identical; `turn_history_fuzz` 18,791 decisions
  mismatch=0; obs benchmark ~6.36k calls/encode (≤ ~12.8k baseline), `PokemonType.damage_
  multiplier` absent from the profile (chart not bypassed).
- **Trapping.** `trapping_signals_fuzz_test.py` asserts the three signals fire at the right
  decisions against raw protocol; `turn_history_fuzz` stays consistent (the new history bit is
  zero on no-rejection turns); smoke round-trips `gen3_trapping_signals_v1`.

---

## Design rationale (carried from the retired handoff + todo docs)

The execution detail for Thread A originally lived in a cold-start handoff doc and the spec for
Thread C in a standalone trapping-signals todo; both are now retired and superseded by this
as-built record. The load-bearing rationale worth keeping:

- **Why fold the delta from the event log, not snapshot diffs.** The faint *cause* (attack /
  hazard / weather / status / recoil / self-KO / Leech Seed) and the rejected pivot exist
  **only** in the ordered event log — a snapshot diff sees "HP hit 0" but not *why*, and never
  sees a switch the server refused. Attribution is captured pre-mutation, so the gap=0 desync
  the old detective defended against can no longer corrupt it.
- **The decision window ≠ a protocol turn.** A forced switch after a faint splits one game turn
  into two decision windows; a faint window can span a `|turn|` boundary. The fold slices by
  `event_cursor` / `events_between`, not `events_for_turn`.
- **`maybe_trapped` is the high-value trapping signal** — it carries genuinely new information
  (switches stay legal, so the model can't infer it from the mask). Confirmed `trapped` is the
  least critical (the mask already enforces it); the rejected-switch history bit makes the
  "I tried to pivot and got trapped" event learnable for the first time.

---

## Pointers (impl_step1–9 as the single source of truth for ai_v4)

The Step-6 "lossless replay recorder" goal was reached a different way — quota-gated eval
forensic traces (`664f3a5`) replaced the retired `replay_recorder.py` — so no standalone replay
step is needed. With this doc, `impl_step1`–`impl_step9` are the complete as-built record of
ai_v4 (spread → HP inference → damaging-event attribution → unified transformer → move outcome →
next-run bundle → adaptive LR → strict battle-API + event-sourced TurnDelta → this step). The
one open ai_v4 tail is **pathology hunting** (eval-replay analysis); the one open strict-API
sub-item is Phase 5b above.
