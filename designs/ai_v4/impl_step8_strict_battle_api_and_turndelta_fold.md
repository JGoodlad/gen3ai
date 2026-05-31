# impl_step8 — Strict Battle-API + Event-Sourced TurnDelta (ai_v4)

> **Note:** this file was previously corrupted (a tooling-under-load episode committed vim
> keystroke noise — `wq`/`wc`/`:q`/… — over the body). Rebuilt from the landed commits
> (`e35df88..26d3d1e`) and `todo_live_battle.md`. The forward-looking plan and per-phase
> rationale live in `todo_live_battle.md`; this is the as-built record.

This doc covers three interlocking ai_v4 migrations that landed together: encapsulate
poke-env behind read-models we own, event-source the per-turn history, and re-source the
reward onto that boundary. The end state: **our code reads battle state only through
`LiveView` / `TurnView` / `LegalActions` (via `StrictBattleView`)**, the per-turn `TurnDelta`
folds **entirely from the event log + LiveView**, `battle_context.py` is **deleted**, and the
reward manager reads **only LiveView** with no raw-`battle` fallback.

---

## 1. Strict read-model boundary

- **`LiveView` / `LiveSide` / `LivePokemon` / `LiveMove` / `LiveWeather`** — immutable
  current-board snapshot (primitives only, no back-reference to poke-env `Pokemon`).
  `LivePokemon` carries species/hp/status/types/moves(+PP)/item/ability/boosts/volatiles plus
  the spread block (`base_stats` both sides; `ivs`/`evs`/`nature` own-side, gated by
  `spread_known`), `consumed_item`, `status_counter`. `LiveView` carries meta
  `turn`/`battle_tag`/`finished`/`won`/`lost`.
- **`LegalActions` / `LegalMove` / `LegalSwitch`** — server-authoritative legality surface
  parsed from the request: per-slot move id+pp+disabled+target, switch species/slots,
  `force_switch`/`trapped`/`maybe_trapped`/`wait`/`struggle`, and a read-only `last_request`
  mirror. Struggle is single-sourced via the `struggle` flag (filtered out of `move_slots`).
- **`TurnView`** — pure per-turn fold over the `BattleEvent` log (history surface).
- **`StrictBattleView`** — the front door (`battle.strict_view()`): exposes only `.live`,
  `.turn_view(n)`/`.history`, `.legal`, `.events_since`/`.event_cursor`, and scalar meta;
  `__getattr__` raises and the raw `Gen3Battle` is never returned.
- **`agents.enums` seam** (`6f3c59f`) — the four accepted spec value-enums (`PokemonType`,
  `Status`, `MoveCategory`, `Weather`) are re-exported from one module; `Effect` is excluded
  (replaced by `observation/gen3_effects.py`).
- **`strict_api_lock_test.py`** (`6f3c59f`) — AST guard: fails CI on any raw stateful
  `battle.<attr>` read or direct poke-env value-enum import in the consumer clusters, with a
  small per-(file, attr) allowlist for intended seams and a `test_allowlist_has_no_dead_entries`
  guard so the allowlist can't rot.

## 2. Consumer migration (Wave 1)

All consumers re-sourced to the strict view, value-neutral (obs byte-identical, reward 0-diff):
`observation/` (`3418c26`, + opp-active via `live_mon.active` `803a94e`), `action/` mask+mapper
through `LegalActions` (`021f2d3`), `training/` display/replay/control-flow (`182b499`) and
`episode_tracker` (`8b41bb4`), `inference/player`. The reactive effectiveness hot-loop +
`action/serialize.py` remain documented, allowlisted seams.

## 3. Event-sourced TurnDelta fold + `battle_context.py` deletion (`90739dd`)

`TurnDelta` now folds entirely from `events_since(cursor)` + LiveView — HP deltas, faints,
boosts, target-HP, cant, effectiveness, status/item transitions all come from the event log
(boosts via the two-anchor approach, not naive event-replay, to handle `-setboost`/Haze).
The legacy snapshot-diff `build()` and the whole `BattleContext` snapshot layer are **deleted**;
`TurnDelta` lives in its own fold-only module. Verified by the fold-equivalence fuzz
(`turn_delta_fold_equivalence_fuzz_test.py`) + `turn_delta_hp_fold_test.py`.

## 4. Reward re-source onto LiveView → `_read_live` retirement (`d56942d` → `26d3d1e`)

- **`d56942d`** moved the per-term activation helpers (se_switch, pivot_*, dead_matchup,
  sleep_*, boost_utilized, finishing_blow, `_update_opp_se_threat`) onto LiveView +
  `gen3_movedex` + `effective_multiplier_by_types`, behind a `_read_live` dual-path proven
  value-neutral by the equivalence harness.
- **`26d3d1e` (this step) retires the dual-path.** Post-fold the live view is always
  available, so `process_turn_reward` builds `live = battle.live_view()` **once** and threads
  it to every helper. Deleted: every `_read_live=False` `else` branch reading
  `battle.active_pokemon`/`opponent_active_pokemon`/`team`/`opponent_team`/
  `opponent_side_conditions`; the `_read_live` flag; the dead `compute_base_reward`; now-dead
  imports (`SideCondition`, `effective_multiplier`, `is_status_move_immune`). `report_episode`
  alive-counts read `live.ours.mons`/`live.opp.mons`. The 5 `(training/reward_manager.py, …)`
  allowlist entries were removed — **reward_manager now has zero guarded raw reads.** The only
  residual raw `battle` access is `battle.turn` (stall-tax ramp + episode report) and
  `report_episode` meta (`won`/`lost`).
- The now-moot `reward_resourcing_equivalence_fuzz_test.py` was converted to a single-path
  `reward_value_regression_fuzz_test.py` (per decision: finite + `total == sum` + deterministic
  across two identically-driven managers; per-field activation report).

**Gates (26d3d1e):** unit+integration 1251 passed; lock+reward 146 passed;
reward_value_regression all fields finite/consistent/deterministic; CPU smoke (:9001)
round-trip PASS + 11 episodes + eval. No `ARCH_SIGNATURE` bump (pure re-source; obs and reward
values unchanged).

---

## Remaining (see `todo_live_battle.md` / `todo_trapping_signals.md`)
- **Phase 5b** — true `LiveView` independence (event-fold the current-board snapshot itself;
  side-condition counters, ability/item inference, type-change edge cases).
- **Performance pass** — the reactive 288-cell matchup hot loop (≈80% of obs CPU): hoist
  per-mon reads, carry borrowed enums + `LiveMove.move_type`/`type_known` on the read-model so
  the loop is enum-native and read-model-sourced, then vectorize.
- **Trapping signals** — route `trapped`/`maybe_trapped` + an `attempted_switch_rejected`
  history bit into the obs (retrain-class).
