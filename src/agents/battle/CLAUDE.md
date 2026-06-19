# CLAUDE.md — Event-Sourced Battle Layer (`src/agents/battle/`)

poke-env is a **state tracker** — each `|...|` protocol line overwrites "current board"
fields. RL/reward/replay need the opposite: *what happened, in order*. The event-sourced
layer captures that without reimplementing poke-env (as-built record:
`designs/ai_v4/impl_step8_strict_battle_api_and_turndelta_fold.md`).

**Status: live and CONSUMED across training (ai_v4 is closed out).** The per-decision
`TurnDelta` history block folds entirely from the log (`build_from_events`, see "Per-decision
history fold" below) and feeds the obs turn-history; the action masker/mapper read the
LiveView / LegalActions / TurnView surfaces; and the reward manager reads `LiveView` too. Our
non-`battle/` code is held to the strict boundary by `src/agents/strict_api_lock_test.py` (the
lock) + the `src/agents/enums.py` re-export seam. The one remaining open item is the `LiveView`
**event-fold** itself (current-board independence), tracked in `designs/ai_v4/todo_live_battle.md`.

- **`Gen3Battle(Battle)`** (`gen3_battle.py`) — subclasses poke-env's singles `Battle`.
  Its `parse_message` override classifies each line, calls `super().parse_message`
  (state tracking is **verbatim** poke-env), then appends a `BattleEvent` with
  attribution resolved **before** the line mutates state. State-equivalence with the
  classic `Battle` is structural (every line still flows through `super()`).
- **`BattleEvent` / `EventKind` / `MESSAGE_POLICY`** (`battle_event.py`) — the immutable,
  ordered schema and the completeness registry. Every protocol keyword poke-env can emit
  is classified `EVENT` / `STATE_ONLY` / `CONTROL` / `COSMETIC` / `UNSUPPORTED`; an
  unclassified or non-gen3 keyword **raises** (a deliberate tripwire). The conservation
  invariant (`Gen3Battle.assert_conservation()`) proves no line is silently dropped.
- **`TurnView`** (`turn_view.py`) — the **history** read surface ("what happened, in
  order"). Folds one turn's events into per-side intent (`move_id`, `switched`,
  `cant_reason`/`cant_move`, `crit`/`missed`/`failed`, `effectiveness`, `damaging_move`,
  `status_applied`/`status_cured`, `item_lost`/`item_gained`) + turn-level facts
  (`move_order`, `we_moved_first`, `both_attacked`, `someone_fainted`,
  `damage_on(species, side=…)`). `TurnDelta.build_from_events` (`training/turn_delta.py`)
  folds this on every production path — the diff-based detective is retired (see below).
- **`LiveView` / `LiveSide` / `LivePokemon` / `LiveMove`** (`live_view.py`) — the
  **current-board** read surface ("what is true now"), built via `battle.live_view()`. An
  immutable snapshot of HP, status, boosts, revealed moves/item/ability, volatiles, hazards,
  weather, team sizes/reveal counts — holding **only primitives, no past-turn state** and no
  reference back to poke-env's `Pokemon`. A consumer literally cannot reach `last_move`
  through it, so current-state and history come from two disjoint, separately-fuzzed sources
  that can't drift. Opponent fields are reveal-gated (unknown item → `None`, only revealed
  moves listed; `ability` is `None` unless disclosed or uniquely inferable from species).
  `LivePokemon` also carries the **spread block** (`base_stats` — public, both sides;
  `ivs`/`evs`/`nature` — own side only, gated by `spread_known` mirroring the obs encoder),
  `consumed_item` (id-form), `status_counter`, `protect_counter` (the consecutive-successful-stall
  counter — a CURRENT-board fact like `status_counter`, NOT history; the obs `gen3_protect_odds_v1`
  scalars read it via `gen3_mechanics.protect_success_probability`), and the **incoming-damage belief
  inputs** (`stats` — the EV/IV/nature-computed {atk,def,spa,spd,spe}; integer `current_hp`/`max_hp` —
  own-side reliable, opp HP is %-based), so the incoming-KO belief reads the read-model instead
  of the raw `Pokemon` (shared by the obs `encode_block(live)` and the reward PBRS shaping);
  `moves` is a tuple of `LiveMove(id, current_pp, max_pp)` with a `move_ids` accessor for id-only
  call-sites. `LiveView` carries the meta `turn`/`battle_tag`/`finished`/`won`/`lost`.
- **`LegalActions` / `LegalMove` / `LegalSwitch`** (`live_view.py`) — the
  **server-authoritative** legality surface, built via `LegalActions.from_battle(battle)`
  (or `strict_view().legal`): per-slot `LegalMove(id, current_pp, max_pp, disabled, target)`,
  `LegalSwitch(species, slot)` (slot = the team index the 11-dim action space uses),
  `force_switch`/`trapped`/`maybe_trapped`/`wait`/`struggle`, and a read-only
  (`MappingProxyType`) mirror of `last_request`. **Hybrid sourcing (kept deliberately):**
  `move_slots` is **wire-truth** (straight from `last_request['active'][0]['moves']`), while
  the legality *flags* (`switches`/`force_switch`/`trapped`/`maybe_trapped`/`wait`/`struggle`)
  are poke-env's **derived** interpretation (`available_switches`/`force_switch`/
  `available_moves`/…) — kept byte-identical and flagged as the second poke-env-interpreted
  seam (alongside Choice→BattleOrder serialization) a future fully-owned `Player` would
  re-derive. **Struggle is single-sourced:** `from_battle` filters the lone `struggle` entry
  OUT of `move_slots` and surfaces it only as the `struggle` flag — so it can never set both a
  move slot and bit 10 (the historical "struggle double-enabling" class). **Typed own Hidden
  Power:** `from_battle` resolves `own_hp_typed_id` (our active mon's typed HP id off its moveset —
  gen3 has no team preview, so the wire `active` block keys our HP bare while the `Move` object
  keeps the IV-derived type, which we always know), and `display_move_ids` is `move_ids` with that
  bare `hiddenpower` shown typed. **`move_slots`/`move_ids` stay wire-truth** (the mask/mapper/
  serialization key on the bare id); `display_move_ids`/`own_hp_typed_id` are for human/forensic
  LABELS only (the recorder's action labels) + the turn-history fold (which restores our HP type),
  OUR side only (no opponent-HP leak). **The action
  masker/mapper read through this:** `Gen3ActionMasker.mask_from_legal(legal)` builds the
  mask, `Gen3ActionMapper.action_to_choice(action, legal)` is a pure decode to a poke-env-free
  `Choice`, and `serialize.choice_to_order(choice, battle)` is the lone serialization touch.
  The immutable snapshot captured at observation time (carried on the `BattleContext`) **is**
  the per-decision source, retiring the old `battle._gen3_decision_context` stash;
  `assert_decision_current` fail-loud-guards a mid-decision request shift.
- **`StrictBattleView`** (`strict_view.py`) — the **strict boundary** our non-`battle/` code
  reads through, built via `battle.strict_view()`. Exposes **only** `.live`
  (`LiveView`), `.turn_view(n)`/`.history` (`TurnView`), `.legal` (`LegalActions`),
  `.events_since(cursor)`/`.event_cursor`, and scalar meta
  (`turn`/`battle_tag`/`finished`/`won`/`lost`). `turn` is the current-turn **int** — same
  name/meaning as `battle.turn` and `LiveView.turn` so a consumer migrating off `battle.turn`
  is a drop-in; the per-turn `TurnView` accessor is `turn_view(n)` (mirrors `.live`→`LiveView`,
  `.history`→all `TurnView`s) to avoid the method-vs-property name clash. `__getattr__` raises a
  helpful error naming the right accessor; the raw `Gen3Battle` is held privately and never
  returned. Every consumer cluster (`observation/`, `action/`, `training/` incl.
  `reward_manager.py`, `inference/`) now reads through this boundary — verified value-neutral
  and pinned by `strict_api_lock_test.py`.
- **Per-decision history fold.** `TurnDelta` folds **entirely** from the event log + LiveView
  and lives in `training/turn_delta.py` (relocated out of the now-deleted `battle_context.py`).
  `build_from_events(prev, curr, action, events)` is the **sole production builder** — wired
  into the training env, `episode_tracker.py`, `reward_tracker.py`, and the forensic
  `battle_recorder.py`. What it folds from the event stream:
  moves/switches/cant/effectiveness/faints+causes/status-transitions/item-lost/outcome/crit/
  move-order, the **per-slot HP delta** (from each `DAMAGE/HEAL/SETHP` `hp_after` + `FAINT`→0 —
  bit-identical to `curr_hp − prev_hp`, no float-sum noise; a self-KO Explosion/Selfdestruct
  emits NO `-damage` on the user, so its HP→0 comes from the FAINT), and the **target-HP**
  attribution. **What the event log canNOT fold value-identically** (so these stay sourced from
  the LiveView-projected decision snapshot, which is current-board, not poke-env-raw and not a
  heuristic reconstruction): the per-slot **HP-after** (intrinsically current-board) and the
  **boost-stage delta** (`SETBOOST`/`clearboost`/`invert`/`copy`/`swap` carry no realized stage
  amount in the event payload — only an `op`). The per-decision snapshot itself relocated to
  `training/battle_snapshot.py` (`BattleContext`, still the reward `record_action` / mapper /
  `prev_mask` source; it is a documented seam exempted in `strict_api_lock_test`). The legacy
  diff detective `TurnDelta.build` is **retired from every production path** and lives in
  `training/turn_delta_legacy.py` (test-support only, retained for the poke-env-gap fuzz
  harnesses). Value-identity is pinned by the 15-min `turn_delta_fold_equivalence_fuzz_test.py`
  (event-fold vs a frozen snapshot-diff reference: 0 field diffs, 0 reward diffs over 150k+
  decisions, all corner paths covered).
- **Injection seam (wired into training):** `poke_env.player.Player.__init__` takes
  `battle_class=Battle` (default, with a `None`-guard since `PokeEnv` threads a `None`
  default to its `_EnvPlayer` agents). `Gen3Player` defaults `battle_class=Gen3Battle`
  (so RL / eval / replay / stat-tracking players inherit it), and `Gen3Env` defaults it
  too (both env agents track the log; the trainee `battle1` is what obs/reward/replay
  read). These + the (unchanged) parser are the only edits to the poke-env core.
- **Per-decision event window:** `Gen3Battle.event_cursor` + `events_since(cursor)` slice
  the log by "since the agent was last asked to act" — the granularity `TurnDelta` needs,
  which is NOT a protocol `|turn|N` boundary (a forced switch splits a turn into two
  decision windows; a faint window spans a turn boundary). `events_for_turn(N)` remains
  for protocol-turn slicing. Package re-exports are lazy (PEP 562 `__getattr__`) so
  importing one submodule doesn't force-load the others (avoids an import cycle).

**Verification:** unit tests in `src/agents/battle/*_test.py` (schema, registry audit,
scripted parse + state-equivalence, TurnView fold, `live_view_test.py` for the current-board
surface + widened fields, `strict_view_test.py` for the strict boundary + `LegalActions`
extraction + the forbidden-access guard). The spine is `event_log_fuzz_test.py` — real
`gen3ou` battles where both players run `Gen3Battle`; it independently re-derives each turn
from the intercepted raw protocol and asserts the event log matches, plus conservation +
event-kind coverage, the widened LiveView fields (spread/PP/consumed/counter/meta) per mon,
and the `LegalActions` legality surface against the live server request at every decision.
Run it as a script (no live server — bridge-backed): see the root `CLAUDE.md` Running Tests.
