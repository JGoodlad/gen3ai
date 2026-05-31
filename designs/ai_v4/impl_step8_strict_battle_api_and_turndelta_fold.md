# Implementation: Step 8 — Strict Battle-API + Event-Sourced TurnDelta Fold

This step lands two tightly-related advances on the event-sourced battle layer
(`design_event_sourced_battle.md`), plus the bug fix and doc resync that fell out of
bringing them together. They shipped to `main` across three commits:

| Commit | Thread | What landed | Arch impact |
|---|---|---|---|
| `10960c7` | A + B | **Strict battle-API** (Phases 1–2 of `todo_live_battle.md`): widened `LiveView`, new `LegalActions`, the `StrictBattleView` boundary. **+ own-team spread backfill** (poke-env fork). | `gen3_own_spread_v1` (obs 2823, spread *values* change → retrain-class; obs dim unchanged) |
| `c2dbee6` | C | **Event-sourced `TurnDelta` fold** — Step 4 / §9.5 of the design. Folds `TurnDelta` from the per-decision event window via `TurnView`, retires the diff heuristics, adds faint-cause / attempted-move / status-transition / item-used signals. | `gen3_turn_delta_v2` (obs 2823 → **3299**) — supersedes `gen3_own_spread_v1` |
| `d8c35f8` | docs | Resync the public `README` + `CLAUDE.md` obs-vector tables to the live encoder (`3299`-dim). | none (docs + one comment) |

Net: `ARCH_SIGNATURE` is now **`gen3_turn_delta_v2`**, obs **3299-dim**. The KL-reactive
LR-band widening (`e4c305d`) that landed in the same window is a separate
training-infra change — see `impl_step7_adaptive_lr_kl_band.md`, not repeated here.

Two threads (A/B = the strict-API encapsulation + spread fix; C = the TurnDelta fold) are
documented separately below because they are independent in motivation even though they
co-landed and both build on the §9.1–§9.4 event-sourced foundation.

---

## Thread A — Strict battle-API: a read-model boundary over poke-env

**Why.** poke-env's `Battle`/`Pokemon` mix *current* facts with *temporal*, overwrite-on-
every-line fields (`last_move`, `last_cant_reason`, `protect_counter`, mutable `effects`).
Every consumer that reads the raw object is a place a protocol edge case can silently
corrupt an obs/reward (the Focus-Punch / Sleep-Talk class). The long-term standard: our
code reads battle state **only** through vetted, fuzzed read-models we own — `LiveView`
(current board), `TurnView` (history fold), `LegalActions` (server-authoritative legality)
— with poke-env kept as the engine underneath, hidden behind the boundary. Full plan +
deferred phases: `todo_live_battle.md`.

### Phase 1 — widen the read-models (`src/agents/battle/live_view.py`)
So nothing legitimately needs a raw read:

- **`LivePokemon`** gained the **spread block** (`base_stats` — public, both sides;
  `ivs`/`evs`/`nature` — own-side only, gated by `spread_known`), `consumed_item` (id-form,
  normalised), and `status_counter`. `moves` became a tuple of `LiveMove(id, current_pp,
  max_pp)` with a `move_ids` accessor so id-only call-sites stay terse.
- **`LiveView`** gained meta: `turn` / `battle_tag` / `finished` / `won` / `lost`.
- New **`LegalActions` / `LegalMove` / `LegalSwitch`** — the **server-authoritative**
  legality surface, sourced from poke-env's already-parsed `|request|` (NOT derived):
  per-slot move id/pp/disabled/target, switch species + team-slot, `force_switch` /
  `trapped` / `maybe_trapped` / `wait` / `struggle`, and a read-only (`MappingProxyType`)
  mirror of `last_request` for the masker's request-echo path.

### Phase 2 — the strict boundary (`src/agents/battle/strict_view.py`)
- **`StrictBattleView`**, built via `Gen3Battle.strict_view()`, exposes **only** `.live`
  (`LiveView`), `.turn_view(n)` / `.history` (`TurnView`), `.legal` (`LegalActions`),
  `.events_since(cursor)` / `.event_cursor`, and scalar meta. `__getattr__` raises a
  helpful error naming the right accessor; the raw `Gen3Battle` is held privately and
  never returned.
- **Naming-collision resolution (worth remembering):** the plan listed both a `.turn(n)`
  history accessor and a `turn` scalar meta — a method and a property can't share a name.
  Resolved to minimise surprise: **`turn` is the scalar current-turn `int`** (identical in
  name and meaning to `battle.turn` / `LiveView.turn`, so migrating a consumer off
  `battle.turn` is a drop-in), and the `TurnView` accessor is **`turn_view(n)`** (mirrors
  `.live`→`LiveView`, `.history`→all `TurnView`s).
- New types lazy-exported from `src/agents/battle/__init__.py` (PEP 562, no import cycle).

### Deferred (Phases 3–5, `todo_live_battle.md`)
Consumers are **not yet migrated** to the strict view. Phase 3 = migrate the ~15
`observation/` / `action/` / `training/` / `inference/` consumers (skipping
`battle_context.py`, which the TurnDelta fold retires); Phase 4 = a static "no raw
`battle.`/`Pokemon` access outside `battle/`" guard test (the actual lock); Phase 5 =
event-fold `LiveView` itself for true poke-env independence. Phases 1–2 are additive and
obs-neutral on their own.

---

## Thread B — Own-team spread backfill (a dead obs feature, revived)

**The bug.** In a real gen3ou battle the own active Pokémon had `mon.ivs = None`,
`mon.evs = None`, `mon.nature = None` on the raw poke-env object — even though
`battle.teambuilder_team` carried the real declared spread. Consequence: the 18-dim spread
block (`impl_step1_spread_encoding.md`) read those `None`s and emitted its fallbacks
(IVs all-31, EVs all-0, neutral nature) for **all 6 own mons** — an identical constant
vector, i.e. *zero signal*, for the entire history of gen3ou training.

**Root cause.** poke-env's `_update_from_teambuilder` (which sets `_ivs/_evs/_nature`) is
only ever invoked by `apply_teambuilder_team`, which matches the teambuilder team against
**`teampreview_team`**. gen3ou has **no team preview**, so that list is empty → the loop
body never runs → the spread is never attached. `impl_step1` assumed
`apply_teambuilder_team` populates own spread; that holds only for formats *with* preview.
(The `_update_from_teambuilder` EV-guard fix from `impl_step1` was correct — it just was
never being *called*.)

**Fix (poke-env fork, minimal, format-general).** Invoke poke-env's own machinery for the
no-preview case instead of building a parallel read-model:

- `Pokemon.backfill_spread_from_teambuilder(tb)` (`pokemon.py`) — sets **only**
  `ivs`/`evs`/`nature`, never touching moves / PP / stats; idempotent (no-op once `_ivs`
  is set, so it never clobbers preview- or request-applied data).
- `AbstractBattle.backfill_teambuilder_spread()` (`abstract_battle.py`) — matches the
  declared `teambuilder_team` to the request-built team **by species** (unique per side in
  singles; `tb.species or tb.nickname` → id-form) and calls the above.
- `Battle.parse_request` calls it **after** `_update_team_from_request` builds the team.

Verified live: own active now reads e.g. `ivs=[31,0,31,31,31,31] evs=[252,0,8,0,248,0]
nature=careful`. The obs spread block and `LiveView.ivs` both read `mon.ivs` and so carry
real data automatically — no encoder change. Bumped `ARCH_SIGNATURE` →
`gen3_own_spread_v1` (obs dim unchanged at 2823; only the spread *values* change from
constant to real → retrain-class). This signature was subsequently superseded by the
TurnDelta fold's `gen3_turn_delta_v2`.

---

## Thread C — Event-sourced `TurnDelta` fold (Step 4 / §9.5)

This is the design's Step 4: `TurnDelta` is now **folded from the event log**
(`Gen3Battle.events_since(cursor)` — the per-*decision* window, NOT the protocol-turn
`events_for_turn`) via `TurnView`, retiring the old diff-of-two-`BattleContext`-snapshots
heuristics. `ARCH_SIGNATURE = gen3_turn_delta_v2`, obs **2823 → 3299**; per-slot
`TurnDelta` dim **155-era → 157** (base 53 + extended 104), `N_HISTORY_TURNS = 10`
(turn-history block 1570).

New per-window signals (the facts that exist *only* in the ordered event log, which is
what forces the fold):

- **Multi-KO faint-cause multi-hot** (`our/opp_faint_causes`, 8 each: `attack / hazard /
  weather / status / recoil / selfko / leechseed / other`) — derived from the `[from]`
  clause on the DAMAGE event preceding each FAINT; the rare 2-on-one-side window sets 2
  bits, Explosion double-KO sets one per side.
- **Attempted move** (`our_attempted_move_id`) — the move we *pressed*, preserved even when
  it never fired (cant / frozen / KO-before-acting). (Only the move: a pressed switch
  always executes, so `attempted_switch_to` would equal `switch_to`.)
- **Status transitions** (`our/opp_status_applied` + `_status_cured`, 7 each) and an
  **item-used bit** per side (Berry / Knock Off / Trick). These are the per-turn *events*;
  the cause-**identity** (which item, which ability) lives in the per-mon block — history
  carries the event, the block carries the what.
- **Cant vocab** moved to the authoritative source-derived `gen3_effects.CANT_REASONS`
  (`CANT_DIM = 12`), crash-don't-drop enforced in the encoder.
- **Embedded-ID manifest** (`TURN_DELTA_EMBEDDED_IDS`) — which slot positions carry raw
  embedding IDs and which table each routes to is declared once; both the encoder layout
  and `Embeddings.embed_delta_slot` read it, so there are **no hardcoded positions in the
  extractor** and a raw id can never silently leak through as a scalar.

**Crash-don't-drop tripwire earned its keep again.** The 1m/5m/15m soak (~21k+ battles)
surfaced two volatile classes the static `moves.ts` derivation can't see, both fixed:
- **future-move `-start` volatiles** `doomdesire` / `futuresight` (Doom Desire / Future
  Sight mechanically use `addSlotCondition(target,'futuremove')` but *also* emit
  `-start` on the user → `Effect.DOOM_DESIRE` / `FUTURE_SIGHT`) — added as binary
  volatiles in `gen3_effects.py`.
- **ability-activation `-activate` volatiles** — collapsed to one `ability_activated`
  hint slot, since the activation now reveals the opponent's ability *persistently* (an
  `abstract_battle` `-activate` handler change) and the identity lives in the per-mon
  block. `Effect.MAGMA_ARMOR` was added to the fork enum.

(See `c2dbee6`; full execution detail in `handoff_turn_delta_reward_replay.md`.)

---

## Resulting observation vector (`gen3_turn_delta_v2`, 3299-dim)

| Block | Dims | Offset |
|---|---|---|
| Our team (6 × 107) | 642 | 0 |
| Opp team (6 × 107) | 642 | 642 |
| Active context ×2 (`VOLATILE_DIM` = 44) | 116 | 1284 |
| Global env | 18 | 1400 |
| Reactive + matchups | 300 | 1418 |
| Prev-turn action mask | 11 | 1718 |
| Turn history (`N_HISTORY_TURNS` × 157) | 1570 | 1729 |
| **Total** | **3299** | |

Numbers queried from the live `Gen3ObservationEncoder` / constants (not from memory). The
per-Pokémon slot (107) carries the spread block (18, now real for own mons) + HP-candidate
block (17) + item consumed bit; full field-level layout is in the root `CLAUDE.md`
"Observation Vector" section, which Thread D resynced.

> Note: `design_event_sourced_battle.md`'s status table records the TurnDelta fold as
> "obs 2823 → 2997" — that was a pre-final estimate; the shipped value is **3299**
> (the design doc is forward-looking and explicit-edit-only, so it was left as-is).

---

## Thread D — Docs resync (`d8c35f8`)

The public `README.md` obs-vector tables were stale (`1107`-dim, 59-dim slots, 29-dim
TurnDelta) — they predated the spread, live-state, and TurnDelta-fold work. Resynced to
the live `3299`-dim layout. Also fixed two stale `CLAUDE.md` prose numbers (global env
`13 → 18`, TurnDelta slot `155 → 157`) and a stale `state_encoder.py` layout comment
(`N * 39 → N * 157`).

---

## Files changed (by thread)

**A + B — `10960c7` (14 files):**
`src/agents/battle/live_view.py` (widen), `strict_view.py` (new), `strict_view_test.py`
(new), `live_view_test.py`, `gen3_battle.py` (`strict_view()`), `__init__.py` (lazy
exports); poke-env fork: `pokemon.py` + `abstract_battle.py` + `battle.py` (spread
backfill); `model_version.py` (`gen3_own_spread_v1`); `event_log_fuzz_e2e_test.py` (widened
LiveView + LegalActions checks per decision, `--seconds` soak mode, `spread_data` required
coverage); `CLAUDE.md` + `model/CLAUDE.md`; `designs/ai_v4/todo_live_battle.md` (new,
Phases 3–5 deferral).

**C — `c2dbee6`:** `turn_delta_encoder.py`, `features_extractor.py`, `model_version.py`
(`gen3_turn_delta_v2`), `gen3_effects.py` (+ doomdesire/futuresight), `effect.py`,
`turn_view.py`, `battle_context.py`, `episode_tracker.py`, `event_log_fuzz_e2e_test.py`,
`abstract_battle.py` (`-activate` ability reveal), tests (`turn_delta_event_fold_test.py`
new, `turn_view_test.py`, `gen3_battle_test.py`, …), CLAUDE/README/design docs.

**D — `d8c35f8`:** `README.md`, `CLAUDE.md`, `src/agents/observation/state_encoder.py`.

---

## Verification

- **Unit:** 1141 passed, 2 skipped (combined tree). **Integration:** 20 passed, 1 skipped.
- **Smoke** (`train_rl_agent.py --debug --steps 10000`, combined tree): round-trip PASSED
  with `gen3_turn_delta_v2`, episodes completed, eval ran, **no `UnknownVolatileError`** —
  the doomdesire/futuresight classification holds.
- **Live e2e fuzz** (`event_log_fuzz_e2e_test.py`): the spine validates, per decision and
  per turn, the event log vs an independent re-derivation from raw protocol, plus the
  widened `LiveView` fields and the `LegalActions` legality surface. A `--seconds` time-
  budget soak (1m / 5m / 15m, **~25.8k battles / ~1.96M decisions**, zero mismatches)
  exercised the rare cases — `struggle`, `trapped`, `maybe_trapped` — and made
  **`spread_data` a *required* coverage flag** (the end-to-end tripwire proving own
  IV/EV/nature now reach `LiveView`; it was never seen across 200 battles before the fix).
- A focused review of the combined diff (spread backfill non-clobbering / idempotent /
  no preview-format regression; ARCH/obs-dim self-consistency) returned **ship-ready**.

---

## What's next

- **Event-sourced design Steps 5–6** (`handoff_turn_delta_reward_replay.md`): reward
  manager onto `TurnView` + `LiveView`; lossless replay recorder from the event log. Both
  remain.
- **Strict battle-API Phases 3–5** (`todo_live_battle.md`): migrate consumers to
  `StrictBattleView`; add the static no-raw-read guard test; event-fold `LiveView`.
- **Boost-delta** is captured in `TurnDelta` but **not yet folded** from the event log; a
  full-vector Layer-6 (turn-history) fuzz is still missing (see `todo.md`).
