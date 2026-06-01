# TODO: Lock down battle access behind the strict read-model API

**Status:** Phases 1–4 are **implemented and locked**. **Phase 3 is COMPLETE** — every consumer
cluster (`observation/`, `action/`, `training/` incl. `reward_manager.py`, `inference/`) reads
through `StrictBattleView` / `LiveView` / `LegalActions`, verified value-neutral and perf-neutral
(see "Wave 1 verification" below). **Phase 4 is COMPLETE** (Wave 2) — the four accepted value-enums
flow through the **`agents.enums`** re-export seam (4a), and a static AST guard
(`src/agents/strict_api_lock_test.py`) now fails CI on any new raw stateful `battle.<attr>` read or
direct poke-env value-enum import in those clusters (4b, **the lock**). **Phase 5 — HISTORY fold
is DONE** (`TurnDelta` now folds entirely from the event log + LiveView; `battle_context.py`
deleted; see "Phase 5" below). The only remaining work is the **`LiveView` event-fold** itself
(current-board independence).

Full approved plan (read first): `~/.claude/plans/can-you-explore-what-nested-pebble.md`
— it carries the complete dependency inventory and rationale this doc summarises.

---

## Wave 1 verification (Phase 3 complete) — 2026-05-31

Phase 3 was executed as five file-disjoint tracks (observation / training control+replay /
episode_tracker / action / reward), each gated by an existing value-neutrality harness so the
migration could be proven to change *where* a value comes from, never *what* it is.

**Landed commits:** observation `3418c26`; training control/replay `182b499` (+ `664f3a5`
retired `replay_recorder.py` for quota-gated forensic traces); episode_tracker `8b41bb4`;
action residual `0b23cd6`; reward activation logic `d56942d`.

**Gates, all green (on `d56942d`):**

| Gate | Result |
|------|--------|
| Unit suite (`-m "not integration and not e2e"`) | 1193 passed, 2 skipped |
| Action fuzz (`action/fuzz_test.py`) | 50 battles, 3710 turns, integrity verified |
| Reward equivalence (`reward_resourcing_equivalence_fuzz_test.py`) | 4778 turns, **0 field diffs** (live vs raw paths byte-identical) |
| Obs byte-identity (`turn_history_fuzz_test.py`) | 8681 decisions, **mismatch=0** |
| Obs-build perf (`obs_build_benchmark.py`) | **~12.3k calls/encode** (≤ ~12.8k baseline); `PokemonType.damage_multiplier` absent from the profile (chart not bypassed) — no regression |

**Documented, intentional residual reads** (NOT incomplete migration — these become the
Phase-4 guard allowlist):
- `observation/reactive.py` — the 288-cell effectiveness hot-loop stays on the raw battle:
  own Hidden Power keeps its *typed* id only on the raw `Move` (LiveView re-keys to bare
  `hiddenpower`, which would change the emitted effectiveness); the loop needs `PokemonType`
  enums not LiveView strings; pinned byte-for-byte by `alignment_test`.
- `observation/state_encoder.py` — `mon is battle.opponent_active_pokemon` identity check +
  `battle.battle_tag`/`battle.wait` in the obs error guard.
- `action/mapper.py` — `battle.turn` in the `assert_decision_current` staleness-guard error path.
- `training/reward_manager.py` — the `_read_live` dual-path is **retired** (impl_step8 §4 /
  `26d3d1e`): `process_turn_reward` builds `live = battle.live_view()` once and every per-term
  helper reads current-board state only through it, with no raw-`battle` fallback. reward_manager
  has **zero guarded raw reads**; the only residual is `battle.turn` (stall-tax ramp + episode
  report) and `report_episode` meta (`won`/`lost`).

---

## Why this matters (long-term direction)

poke-env was great for bootstrapping this Gen3 RL project, but over time its
semi-stateful `Battle`/`Pokemon` objects have become a liability. They carry temporal,
overwrite-on-every-line fields (`last_move`, `last_cant_reason`, `protect_counter`,
mutable `effects`, etc.) that are easy to misread and have caused real, hard-to-find bugs
(the Focus-Punch / Sleep-Talk class). Every consumer that reaches into the raw object is a
place where a future protocol edge case can silently corrupt an observation or reward.

The long-term standard we're moving toward: **our code never touches poke-env directly.**
All battle state flows through two vetted, extensively-fuzzed read-models we own and trust —
`LiveView` (current board) and `TurnView` (what happened, folded from the event log) — plus
a server-authoritative legality surface (`LegalActions`). poke-env keeps running underneath
as the state engine, but it becomes an implementation detail hidden behind our API, not
something ~15 files poke at. This is the encapsulation step that makes a later, fuller
decoupling (event-sourcing `LiveView` itself) possible without touching any consumer.

---

## What is already done (Phases 1–2, this branch)

**Phase 1 — the read-models are wide enough that nothing legitimately needs a raw read:**
- `LivePokemon` (`src/agents/battle/live_view.py`) gained: the **spread block**
  (`base_stats` — public, both sides; `ivs`/`evs`/`nature` — own side only, gated by
  `spread_known` mirroring the obs-encoder pattern), `consumed_item` (id-form), and
  `status_counter`. `moves` is now `Tuple[LiveMove]` (`id`, `current_pp`, `max_pp`) with a
  `move_ids` convenience accessor for id-only call-sites.
- `LiveView` gained meta: `turn`, `battle_tag`, `finished`, `won`, `lost`.
- New **`LegalActions`** dataclass (server-authoritative, sourced from poke-env's already-parsed
  request fields — `battle.py:parse_request`): per-slot `LegalMove(id, current_pp, max_pp,
  disabled, target)`, `LegalSwitch(species, slot)`, `force_switch`, `trapped`,
  `maybe_trapped`, `wait`, `struggle`, and a read-only (`MappingProxyType`) mirror of
  `last_request`.

**Phase 2 — the strict boundary object:**
- **`StrictBattleView`** (`src/agents/battle/strict_view.py`), built via
  `Gen3Battle.strict_view()`, exposes **only** `.live` (`LiveView`), `.turn_view(n)`/
  `.history` (`TurnView`), `.legal` (`LegalActions`), `.events_since(cursor)`/`.event_cursor`,
  and the scalar meta `turn`/`battle_tag`/`finished`/`won`/`lost`. `__getattr__` raises a
  helpful error naming the right accessor; the raw `Gen3Battle` is held privately and never
  returned.
  - **Naming note:** the plan listed both a `.turn(n)` history accessor *and* a `turn` meta
    field; a method and a scalar property can't share a name. Resolved to minimise surprise:
    **`turn` is the scalar current-turn `int`** (identical name/meaning to `battle.turn` and
    `LiveView.turn`, so migrating a consumer off `battle.turn` is a drop-in), and the
    `TurnView` accessor is **`turn_view(n)`** (mirrors `.live`→`LiveView`, `.history`→all
    `TurnView`s). When migrating consumers, map `battle.turn` → `view.turn` and any per-turn
    `TurnView` lookup → `view.turn_view(n)`.
- New types lazy-exported from `src/agents/battle/__init__.py`.

**Tests already in place:** `live_view_test.py` (new fields + minimal-set pin + boundary
invariant), `strict_view_test.py` (strict-view surfaces + forbidden-access guard +
`LegalActions` from-request unit tests), and the live e2e fuzz
(`event_log_fuzz_e2e_test.py`) cross-checks the new spread/PP/consumed/counter/meta fields
against poke-env per mon and validates the `LegalActions` legality surface against the live
server request at every decision.

**Obs-neutral:** Phases 1–2 only *add* APIs; they don't change the emitted observation
vector, so there is **no `ARCH_SIGNATURE` bump**. Phase 3 must preserve this — see the note
under Phase 3.

---

## Phase 3 — migrate consumers to the strict view ✅ COMPLETE

All consumers now build `battle.strict_view()` (or `battle.live_view()`) at the decision
boundary and read state through `.live` / `.legal` rather than the raw battle. This was a pure
*access* refactor — obs-neutral (vector byte-identical, no `ARCH_SIGNATURE` bump) and reward
value-neutral (equivalence harness 0-diff). See "Wave 1 verification" above for the gates.

### Consumer inventory (status)

1. **`observation/`** — ✅ **DONE** (`3418c26`). `state_encoder.py` builds the LiveView once
   and threads `.live` / `live_mon` to the sub-encoders.
   - `pokemon.py` / `species.py` — spread block + status_counter now from
     `LivePokemon.{base_stats,ivs,evs,nature,spread_known,status_counter}`.
   - `items.py` / `types.py` / `abilities.py` / `active_context.py` / `global_env.py` — all on
     `LiveView` pieces.
   - **HP / Hidden-Power resolution:** `_own_hp_type_index` was *removed* — it resolved to
     `None` on every real decision (17208/17208; live battles re-key typed HP to bare
     `hiddenpower`), so the one-hot was already dead. The effectiveness hot-loop in
     `reactive.py` deliberately stays on the raw battle (typed-HP id + enum-keyed loop +
     pinned by `alignment_test`) — see the residual-reads list above.
2. **`action/`** — ✅ **DONE** (`021f2d3` + `0b23cd6`). `mask_generator.py`, `mapper.py`,
   `ordering_integrity.py` read the `LegalActions` / `LiveView` snapshot
   (`mask_from_legal(legal)`, `action_to_choice(action, legal)`, `_sorted_move_ids` uses
   `LivePokemon.move_ids`, species-integrity check uses `live.ours.mons`). The
   `_gen3_decision_context` latch is **deleted**; lockstep is the immutable `LegalActions`
   captured at observation time and carried on the `BattleContext`. `action_to_order` is split
   into the pure `action_to_choice(legal) -> Choice` + the single
   `serialize.choice_to_order(choice, battle)` adapter; `assert_decision_current` fail-loud
   guards a mid-decision request shift (the only residual `battle.turn` read, error-path only).
   Struggle is single-sourced via `legal.struggle`.
3. **`training/`** — ✅ **DONE** (`182b499`, `8b41bb4`, `d56942d`). `gen3_env.py`,
   `battle_recorder.py`, `episode_tracker.py`, `stall.py` read meta + `.live` through the
   strict view; `replay_recorder.py` was retired (`664f3a5`) in favour of quota-gated forensic
   traces. `reward_manager.py` extends the `_read_live` dual-path to its per-term activation
   logic (switch/pivot/attack signals), proven value-neutral by the equivalence harness; the
   only remaining `battle.*` reads are the `_read_live=False` fallbacks + the acknowledged
   stall-tax / `report_episode` residue. **`battle_context.py` was deliberately NOT migrated**
   — it is the old diff-based heuristic reader that the deferred **Step 4 TurnDelta fold**
   retires (`designs/ai_v4/handoff_turn_delta_reward_replay.md`); let Step 4 delete it.
4. **`inference/player.py`** — ✅ **DONE.** Meta passthroughs via `strict_view()`.

---

## Phase 4 — enforce: enums seam + the static "no raw battle read" guard — ✅ DONE (Wave 2)

Two parts, the actual *lock* now that Phase 3 has removed the raw reads.

**4a — accepted-enums re-export seam — ✅ DONE.** `src/agents/enums.py` re-exports ONLY
`PokemonType`, `Status`, `MoveCategory`, `Weather` (`__all__` = exactly those four; `Effect`
intentionally excluded — replaced by `observation/gen3_effects.py`). The members are the same
objects poke-env uses (`agents.enums.PokemonType is poke_env….PokemonType`), so it is a pure
import-path indirection. Every consumer under `src/agents/**` (outside `battle/`, `enums.py`,
`*_test.py`) imports those four from `agents.enums` now; `src/agents/enums_test.py` pins the
accepted set so an accidental `Effect` re-export fails CI.

**4b — the guard — ✅ DONE.** `src/agents/strict_api_lock_test.py` **AST-walks** (not regex) the
production consumer modules under `observation/`, `action/`, `training/`, `inference/` and fails
CI on:
- a raw stateful `battle.<attr>` read — the 13 temporal fields (`active_pokemon`,
  `opponent_active_pokemon`, `available_moves`, `available_switches`, `last_request`, `team`,
  `opponent_team`, `weather`, `side_conditions`, `opponent_side_conditions`, `force_switch`,
  `trapped`, `_player_role`); meta scalars (`turn`/`battle_tag`/`finished`/`won`/`lost`/`wait`)
  are **not** guarded — the strict view re-exposes them verbatim. A read is matched only when the
  base is the `battle` identifier (so unrelated `.team`/`.weather` on other objects don't trip it).
- a `from poke_env … import` of one of the four value-enums (must come from `agents.enums`).

**Scope:** the walk skips test/fuzz/benchmark scaffolding (stem contains `test` or `benchmark` —
this also catches `action/fuzz_test_unit.py`) and the two documented seams, `action/serialize.py`
(the Choice→BattleOrder touch) and `training/battle_context.py` (Step 4 retires it).

**Allowlist** — a small, per-`(file, attr)`, inline-commented set of the intended residual reads
from "Wave 1 verification" (keyed by attribute, so a *new kind* of raw read in the same file still
fails): `observation/reactive.py` + `observation/base.py` (the 288-cell effectiveness hot-loop and
its `get_team_list` helper — typed-HP id only on the raw `Move`, enum-keyed loop, pinned by
`alignment_test`; plus reactive's `live is None` unit-test fallback), `observation/state_encoder.py`
(the `mon is battle.opponent_active_pokemon` identity check), and `training/reward_manager.py`
(the `_read_live=False` equivalence-harness `else` fallback + `report_episode` opponent-team
iteration). `base.py` was added beyond the original four-file list — it is the same intended seam
feeding reactive's matrices, not incomplete migration. `test_allowlist_has_no_dead_entries` prunes
stale entries; `test_walk_actually_covers_the_known_consumers` prevents a vacuous pass if an
exclusion ever over-broadens.

**Verified (Wave 2):** full unit suite **1201 passed / 2 skipped**; the lock's detectors confirmed
to flag injected violations (not a vacuous pass); obs-build benchmark **~12.3k calls/encode** (no
regression — the enum re-point is import-only).

---

## Phase 5

### 5a — HISTORY fold (`TurnDelta`) — ✅ DONE

`TurnDelta` now folds **entirely** from the event log + LiveView; the diff-based detective and
the `BattleContext` snapshot-diff layer are gone. Concretely:

- **Relocated.** `TurnDelta` → its own fold-only module **`training/turn_delta.py`**; the
  per-decision snapshot `BattleContext` → **`training/battle_snapshot.py`**;
  **`training/battle_context.py` deleted.** All ~20 importers updated (pure import-path change);
  `strict_api_lock_test`'s documented-seam exclusion moved `battle_context.py` →
  `battle_snapshot.py` (the snapshot still reads poke-env for the HP tracker's
  `opp_last_damaging_event` + the poke-env-gap-fuzz-probed flags).
- **One production path.** `build_from_events(prev, curr, action, events)` is the sole builder
  on every production path — training env, `episode_tracker.py`, `reward_tracker.py` (migrated to
  capture the per-decision event cursor), and the forensic `battle_recorder.py`. The legacy diff
  detective (formerly `TurnDelta.build`) is **retired from production** and has been **extracted
  out of `turn_delta.py` into the test-support module `training/turn_delta_legacy.py`** as the
  module-level `build_legacy` (with its four legacy-only helpers `_moves_match` /
  `_align_effectiveness` / `_ko_before_acting` / `_derive_move_outcome`) — kept for the
  poke-env-gap fuzz harnesses + crafted-context unit tests only, **not imported by any production
  module**. The shared helpers `_fold_hp_deltas` / `_resolve_target_hp_delta` (also called by
  `build_from_events`) stay in `turn_delta.py`, which now carries only the live fold path.
- **What folds from the log:** moves/switches/cant/effectiveness/faints+causes/status-transitions/
  item-lost/outcome/crit/move-order, the **per-slot HP delta** (each `DAMAGE/HEAL/SETHP` `hp_after`
  + `FAINT`→0 — bit-identical to `curr_hp − prev_hp`, no float-sum noise), and **target-HP**.
- **Findings (surfaced, not papered over) — what the event log canNOT fold value-identically,**
  so they stay sourced from the LiveView-projected decision snapshot (current-board, not
  poke-env-raw, not heuristic):
  - **HP magnitude needs care.** A per-event *amount-sum* is NOT value-identical: float
    accumulation order differs from a single endpoint subtraction (~6e-8) and that flips the
    discrete `opp_hp_delta.sum() >= 0` futile-attack reward threshold. Folding the **end HP** from
    each event's `hp_after` (last wins) and subtracting `prev_hp` once is bit-exact instead.
  - **Self-KO HP-incompleteness.** Explosion/Selfdestruct faints the user with NO `-damage` line
    (only `|faint|`), so a pure damage fold misses its HP→0; the `FAINT` event supplies it.
  - **Boost-stage delta is not event-foldable.** `SETBOOST` (Belly Drum), `clearboost`/`invert`/
    `copy`/`swap` (Haze etc.) carry only an `op`, no realized stage amount — so the boost delta is
    the LiveView stage diff. **HP-after** is likewise intrinsically current-board.
- **Verified.** Dedicated 15-min bridge fuzz `training/turn_delta_fold_equivalence_fuzz_test.py`
  (event-fold vs a frozen self-contained snapshot-diff reference, crash-on-first-divergence):
  **0 field diffs, 0 reward-breakdown diffs over 158k decisions**, every corner path (boost /
  CLEARBOOST / switch / Pain-Split-SETHP / double-KO / multi-hit / cant) covered. Plus
  `turn_delta_hp_fold_test.py` (HP/target/boost/self-KO units), the full unit suite (1228 passed),
  `reward_resourcing_equivalence` (0 diffs), `turn_history` 300 (mismatch 0), and the obs benchmark
  (~12.25k calls/encode, ≤ baseline).

### 5b — true `LiveView` independence (FUTURE, not scheduled)

Replace the per-field copy in `LiveView.from_battle` with an **event-fold**, so poke-env's
tracker is no longer the source of current-board state and we are genuinely decoupled. Most
per-mon fields are foldable from the event log. The hard residue (each effectively a poke-env
re-implementation, so do them deliberately, behind the now-strict boundary):

- **Side-condition counter semantics** — Spikes layers, screen turn counts, Toxic Spikes,
  etc., with the exact gen3 decrement/clear rules.
- **Ability / item inference** — the reveal-gating + uniquely-inferable-from-species logic
  (`LiveView` already does some of this for `ability`/`item`; folding needs all of it).
- **Type-change edge cases** — Conversion, Camouflage, forme/transform interactions.
- **Action legality** — already server-authoritative via `LegalActions`; this stays
  request-sourced even after a `LiveView` fold (it is not derivable from the event log alone).

Because Phases 1–4 put the strict boundary in place, Phase 5 can happen **entirely behind the
API** — no consumer changes — and be validated by the existing live e2e fuzz (event-fold
`LiveView` vs poke-env's tracker, per field, per turn).
