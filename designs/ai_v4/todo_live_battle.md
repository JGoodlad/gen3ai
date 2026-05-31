# TODO: Lock down battle access behind the strict read-model API (Phases 3–5)

**Status:** Phases 1–2 are **implemented** (this branch). Phases 3–5 below are **deferred**.

Full approved plan (read first): `~/.claude/plans/can-you-explore-what-nested-pebble.md`
— it carries the complete dependency inventory and rationale this doc summarises.

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

## Phase 3 — migrate consumers to the strict view (DEFERRED)

Build `battle.strict_view()` once at the top of each decision and thread the strict view (or
its `.live` / `.legal` pieces) to the sub-consumers, replacing every raw `battle.<attr>` /
`Pokemon`-object read. Order smallest-blast-radius first; **run the unit suite + the relevant
e2e fuzz after each cluster.**

**Obs-neutrality is the bar:** this is a pure *access* refactor. If migrating an encoder to a
strict-view field changes any emitted value (e.g. PP or a stat now sourced differently), stop
and treat it as a retrain-class change — bump `ARCH_SIGNATURE` per `CLAUDE.md`. The intent is
that the diff changes *where* a value comes from, not *what* it is.

### Consumer inventory (raw reads to replace)

1. **`observation/`** — `state_encoder.py` already builds `battle.live_view()`; have it build
   the strict view once and thread `.live` (+ `.legal` where needed) to the sub-encoders.
   - `pokemon.py` / `species.py` — `mon.base_stats`, `mon.ivs`, `mon.evs`, `mon.nature`,
     `mon.status_counter` → `LivePokemon.{base_stats,ivs,evs,nature,spread_known,status_counter}`.
   - `moves.py` / `reactive.py` — `mon.moves[*].current_pp/max_pp` → `LiveMove`; the
     `get_sorted_moves` convention already matches `LivePokemon.moves` (sorted by id).
   - `items.py` — `mon.item` / `mon.consumed_item` → `LivePokemon.{item,consumed_item}`.
   - `types.py` / `abilities.py` / `active_context.py` / `global_env.py` — already consume
     `LiveView` pieces; finish any residual raw reads.
   - **HP / Hidden-Power note:** `pokemon.py`'s `_own_hp_type_index` reads `move.type` off the
     poke-env `Move` object. `LiveMove` does **not** carry type today — either add a `type` (and
     whatever else the HP path needs) to `LiveMove`/`LivePokemon` in a Phase-1-style widening
     *before* migrating this path, or keep the HP-type derivation in `battle/` and expose the
     result. Don't migrate `pokemon.py` until this field gap is closed.
2. **`action/`** — `mask_generator.py`, `mapper.py`, `ordering_integrity.py` read
   `battle.available_moves`, `battle.available_switches`, `battle.last_request`,
   `battle.team`, `battle.turn`, plus the `_gen3_decision_context` latch. Migrate to
   `.legal` (`move_ids`, `move_slots[*].disabled`, `switch_species`/`switch_slots`,
   `force_switch`, `trapped`, `struggle`) + `.legal.last_request` for the request-echo
   staleness check. The decision-context latch (`battle._gen3_decision_context`) keeps mask
   and mapper in lockstep — preserve that contract through the migration.
3. **`training/`** — `gen3_env.py`, `replay_recorder.py` / `battle_recorder.py`,
   `episode_tracker.py`, `reward_manager.py` / `reward_tracker.py`, `slot_registry.py` →
   meta (`turn`/`battle_tag`/`finished`/`won`/`lost`) + `.live`. **SKIP `battle_context.py`**
   — it is the old diff-based heuristic reader (`last_move`, `last_cant_reason`,
   `force_switch`, `last_request`, boosts) that the deferred **Step 4 TurnDelta fold** retires
   (`designs/ai_v4/handoff_turn_delta_reward_replay.md`). Migrating its heuristics to the
   strict view is wasted work; let Step 4 delete it. Sequence: run this migration *after*
   Step 4, or migrate everything *except* `battle_context.py` and let Step 4 finish it.
4. **`inference/player.py`, `training/stall.py`** — meta passthroughs
   (`turn`/`battle_tag`/`finished`/`won`/`lost`).

---

## Phase 4 — enforce: the static "no raw battle read" guard (DEFERRED)

This is the actual *lock*. Add a static test (sibling to the `_FORBIDDEN_HISTORY_FIELDS`
philosophy in `live_view_test.py`) that scans our consumer modules — everything under
`observation/`, `action/`, `training/`, `inference/` **except** the `battle/` package and
`battle_context.py` (until Step 4 deletes it) — for direct raw access:

- `battle.<attr>` for the temporal/stateful attrs (`active_pokemon`, `available_moves`,
  `last_request`, `team`, `weather`, …) outside `battle/`, and
- `Pokemon`-object attribute reads (`.last_move`, `.ivs`, `.effects`, `.current_pp`, …).

AST-walk the modules (not a regex grep — too many false positives) and fail on any raw read,
with an allowlist for the unavoidable seams. A new raw read then fails CI, which is what makes
the standard stick. Until Phase 4 lands, the standard is convention-only.

---

## Phase 5 — true `LiveView` independence (FUTURE, not scheduled)

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
