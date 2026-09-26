# CLAUDE.md — Event-Sourced Battle Layer (`src/agents/battle/`)

poke-env is a **state tracker** — each `|...|` protocol line overwrites "current board"
fields. RL/reward/replay need the opposite: *what happened, in order*. The event-sourced
layer captures that without reimplementing poke-env (as-built record:
`designs/ai_v4/impl_step8_strict_battle_api_and_turndelta_fold.md`).

**Status: live and CONSUMED across training (ai_v4 is closed out).** The per-decision
`TurnDelta` history block folds entirely from the log (`build_from_events`, see "Per-decision
history fold" below). `gen3_frame_deletion_v1` deleted its OBS encoding (the lag frames) —
`TurnDelta` itself is unchanged and is still the reward manager's per-decision input, the reward
tracker's fold and the α/β intent-label source; the action masker/mapper read the
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
- **The `value` PAYLOAD schema has TWO halves, and both are enforced**
  (`gen3_event_value_schema_v1`). `EVENT_VALUE_KEYS` declares the **required** keys per kind —
  a consumer may rely on them — and is the guard against the Focus-Punch class of loss (a
  builder that FORGETS a key). `EVENT_OPTIONAL_KEYS` declares the rest of the vocabulary; the
  union is `declared_value_keys(kind)`, and `undeclared_value_keys(event)` must be EMPTY on
  every emitted event. **The second half exists because the first is a LOWER bound**: it is
  blind to a key that is invented or renamed, which is the direction that fails silently — a
  consumer reading a key the producer never writes just gets `None` and reads it as "absent".
  That is not hypothetical: the positional-binding sweep's fourth live site was the
  event-window fuzz guarding residual damage with `value.get("from")` on a DAMAGE event, a key
  DAMAGE has never carried (the `[from]` clause is `value["reason"]` there and `value["from"]`
  on the effect kinds). **Adding a payload key means declaring it in the same pass**, in one
  half or the other. Gates: `gen3_battle_test.py` (the canonical feed, both directions, the
  two halves covering the same kinds without overlap, and a PLANTED undeclared key proving the
  check can fail) + `event_log_fuzz_test.py` (the long tail of real gen3ou lines).
  Six keys were pruned when the second half landed, each verified unread: `hp_before`
  (DAMAGE/HEAL/SETHP — `amount` + `hp_after` already pin the transition), `prev_active` and
  `details` (SWITCH/DRAG — the fold takes `our_prev_active` from the decision snapshot, and the
  details string is verbatim in `raw`), `details` (FORMECHANGE), `detail` (SWAP), `move`
  (PREPARE — the typed accessor is `.move_id`, a different key), and the CONSTANT `op` on
  CRIT/MISS/FAIL (identical to the kind). ⚠️ **`op` was NOT dead in general** and is kept
  everywhere it discriminates — most sharply on CLEARBOOST, where seven protocol keywords
  collapse into one kind and `op` is the only thing separating Haze from Psych Up.
- **`TurnView`** (`turn_view.py`) — the **history** read surface ("what happened, in
  order"). Folds one turn's events into per-side intent (`move_id`, `switched`,
  `cant_reason`/`cant_move`, `crit`/`missed`/`failed`, `effectiveness`, `damaging_move`,
  `status_applied`/`status_cured`, `item_lost`/`item_gained`) + turn-level facts
  (`move_order`, `we_moved_first`, `both_attacked`, `someone_fainted`,
  `damage_on(species, side=…)`). Three facts need the CROSS-side event order and are folded in
  `_fold_attribution` (`gen3_event_window_semantics_fixes_v1` / `gen3_progress_clock_attribution_fix_v1`):
  `blocked` (the move was stopped by the target's Protect / Detect — `outcome` is then `"fail"`),
  `hit_dealt` (a bare `-damage` on the other side while THIS side is the current mover — the
  move's own hits, never sand / poison / recoil / Spikes / the other side's Substitute cost) and
  `choice_overridden` (an Encore landed before this side moved). `faint_details` calls a faint
  that no damage line took to 0 HP `other`, never `attack` (Destiny Bond / Perish Song; an absent
  `hp_after` keeps the classic reading); `is_protect_block` / `damage_is_lethal` are the shared
  predicates the event window uses too. `TurnDelta.build_from_events` (`training/turn_delta.py`)
  folds this on every production path — the diff-based detective is retired (see below).
- **`LiveView` / `LiveSide` / `LivePokemon` / `LiveMove`** (`live_view.py`) — the
  **current-board** read surface ("what is true now"), built via `battle.live_view()`. An
  immutable snapshot of HP, status, boosts, revealed moves/item/ability, volatiles, hazards,
  weather, team sizes/reveal counts — holding **only primitives, no past-turn state** and no
  reference back to poke-env's `Pokemon`. A consumer literally cannot reach `last_move`
  through it, so current-state and history come from two disjoint, separately-fuzzed sources
  that can't drift. Opponent fields are reveal-gated (unknown item → `None`, only revealed
  moves listed; `ability` is `None` unless disclosed or uniquely inferable from species).
  ⚠️ **It COPIES from poke-env's `Pokemon` rather than deriving from the event log, so a tracker
  bug down there IS an observation bug up here, and neither this layer's fuzzes nor the obs parity
  fuzzes can see it** — they all read the same `Pokemon`. That is exactly how every Baton-Passed
  stat stage stayed silently cleared for the fork's entire life (ledger 2026-08-23; pinned by
  `poke_env/battle/baton_pass_carryover_test.py` and the protocol-anchored
  `training/poke_env_gaps/baton_pass_obs_integration_test.py`).
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

  **`live_view()` is MEMOIZED — one build per state-epoch** (`gen3_live_view_memo_v1`). A
  production decision asked for this view **five** times over state that cannot change between
  them (the mask/legality snapshot, `tracker.record`, `update_progress_clock`,
  `state_encoder.encode`, `reward_manager.process_turn_reward`), each a 12-mon rebuild at ~25
  poke-env property reads plus two dict copies per mon. The view is frozen and holds only
  primitives, so *sharing* it is semantically free; the only question is staleness.

  The key is a single monotone `Gen3Battle._state_epoch`, bumped by **every** writer of the
  state `LiveView.from_battle` reads. That completeness argument is why the key is not
  `len(events)`:

  | Door | Covers | Why it is not redundant |
  |---|---|---|
  | `parse_message` | every protocol line, whatever its `Policy` | `\|turn\|` and `\|teamsize\|` mutate state while being `CONTROL`; `STATE_ONLY` is empty *today* but deliberately open |
  | `parse_request` | `_update_team_from_request` | a request is **never** an event, yet it writes HP / status / item / PP / stats and can flip the active mon — the stale view a `len(events)` key would serve |
  | `won_by` / `tied` | `finished` / `won` / `lost` | `\|win\|` / `\|tie\|` are intercepted by `Player._handle_battle_message` and never reach `parse_message` |
  | `_record` | every event append | keeps "an event was appended ⇒ the epoch moved" true for the out-of-band `CHOICE_REJECTED` too |

  The one write outside the doors is construction-time (`bc/log_reader` sets `_player_role` /
  `_format` before feeding a line); the memo is empty until the first read, so a
  pre-first-view write is invisible by construction.

  Two properties are load-bearing, and each has a named regression test in
  `live_view_memo_test.py` — all verified failing on revert of their own bump:

  * **The epoch is read BEFORE the build and the view stored under THAT epoch**, so a view
    built across a concurrent write lands under a key that is already dead and can never be
    served to a later reader. (Views are built on the env's main thread inside
    `embed_battle` / `calc_reward` while the protocol is parsed on `POKE_LOOP`; the
    battle-queue handshake means the two do not overlap in practice, but the memo does not
    depend on that.)
  * **The memo rides the object it describes.** Both the epoch and the view live on the
    battle, so the offline materializer's per-arm `deepcopy` restore (`_PlayerSnapshot`)
    carries a self-consistent pair. A cross-object cache keyed by `battle_tag` — the arms are
    *indistinguishable* by tag — would serve arm-1's forward state to a rewound arm-2; that
    shape is unrepresentable here. The re-decide rollback (`EpisodeTracker.restore`) rolls
    back tracker state and never touches the battle, so the memo stays correct across it by
    not participating.

  Measured over 589 real bridge decisions (`gen3ou`, the full `Gen3Env` path): **5.000 →
  1.000 `LiveView` builds per decision, 57.0 → 11.6 `LivePokemon.from_pokemon` calls per
  decision**; `trainer_turn_benchmark --decisions 300` (same session, back to back, quiet box)
  **0.923 → 0.666 ms of our controllable CPU per decision, −28%**. `mask_generator.get_mask`
  was the one caller reaching past the accessor to `LiveView.from_battle`; it now prefers
  `battle.live_view()` and falls back for a plain poke-env `Battle`, which has no such method.
  Byte-identity is gated by `live_view_memo_fuzz_test.py` (real bridge battles; memo'd view ==
  fresh rebuild, and the full 2501-dim obs encoded warm == encoded with the memo cleared, at
  every decision).

  **That ONE remaining build is now the largest single item in per-decision worker CPU — 17% —
  and it spent months wearing another stage's name** (`gen3_live_view_build_micros_v1`,
  2026-08-23). Because the memo serves whichever consumer asks FIRST, the whole 12-mon build was
  billed to `obs: legal + mask` in `trainer_turn_benchmark`, which therefore read 22% of worker
  CPU while the legality snapshot + 11-bit mask + two integrity checks measure **0.028 ms
  (2.8%)**. Measured by pre-building the view before the stage: the stage falls 0.222/0.243 ms →
  0.030/0.031 ms, i.e. **88% of that line was this build**. The benchmark now times
  `obs: live_view (shared build)` on its own line so the other five stages are honest. *A memo
  moves a cost onto whoever arrives first; from then on the profile names the wrong stage.*

  **What the build's cost decomposes into, and what was harvested.** There is NO redundancy left
  *within* a decision — the epoch memo already took it — so the remaining work is one honest
  12-mon rebuild whose cost is `LivePokemon.from_pokemon` × ~11.5. Three derivations in it were
  pure functions of IMMUTABLE inputs, re-evaluated every build: `Move.max_pp` (18% of the build,
  ~36 evaluations/decision of a dex constant), `Move.entry` (inside it), and `_enum_name`/`_id`
  (6.6%, the `.name` of a process-wide enum singleton, reached through a `DynamicClassAttribute`
  descriptor). All three are now memoized: `max_pp` per INSTANCE (`_id`, `_gen`,
  `_from_transform` are write-once in `Move.__init__`), the enum names per MEMBER, and **`entry`
  at MODULE scope keyed `(gen, id)` — deliberately NOT on the instance**, because
  `obs_materializer._PlayerSnapshot` deep-copies the whole battle graph per counterfactual arm on
  the stated ground that *"`Pokemon`/`Move` carry an int `_gen` and look entries up on demand"*;
  an instance-held dex row would be duplicated into every arm. Pinned by a deepcopy test. Plus
  two hot generator expressions turned into list comps and five defensive `getattr(mon, …,
  default)` calls turned into direct property reads (every one of those properties exists on
  every `Pokemon`, so the default was unreachable and could only have swallowed an
  AttributeError raised *inside* a property). Measured on a frozen real board, order-alternated
  same-process A/B against a verbatim copy of the old code, arms verified field-identical:
  **1.244× on the build** (six rounds, 1.235–1.255) and **−34.6% Python calls per build (1073 →
  702, `sys.setprofile`, load-free)**. End to end, seven alternated `trainer_turn_benchmark`
  pairs: the `live_view` stage **1.22× median (7/7 positive, 1.10–1.32)**, our controllable CPU
  ~1.06× median. Gates: `live_view_build_micros_test.py` (23 cases — the whole gen3 move
  universe against the spelled-out formula, every branch in seven gens, per-instance isolation,
  the synthetic `recharge` row, and the enum-key-safety property asserted on the enum classes
  themselves; three deliberate mutations verified failing).

  🔴 **The named next item, now sized: make the build PARTIAL.** ~9.5 of the ~11.5
  `from_pokemon` calls per decision rebuild a BENCHED mon that did not change — worth roughly
  13–16% of worker CPU, the largest remaining lever anywhere in the per-decision budget. It is
  not built here because it needs a per-mon dirty signal that is EXACT for every `LivePokemon`
  field, and the obs assembler's per-mon dirty set does not qualify: it is gated on the obs
  BYTES, so fields the obs slot never reads (`protect_counter`, `stats`, integer HP, the reward
  path's spread block) ride along unproven. `|-cureteam|` — one enum member unioning two
  protocol keywords, one of them side-wide — is the shape of what would go wrong, and it has
  already bitten the assembler once. A stale board here is silently wrong in the obs, the reward
  AND the mask at once.

  **The epoch is deliberately COARSE, and one consumer needed a finer signal.** A single monotone
  counter is what makes the completeness argument an enumeration of doors, but it also means "the
  battle changed" is all it can say. The incremental obs cache
  (`agents/observation/assembler.py`, `gen3_obs_assembler_v1`) needs *which mon* changed, and the
  event log answers that for every protocol line — except the `parse_request` door, which is not
  a line. So `Gen3Battle` additionally tracks the request at **per-mon granularity**:

  * `parse_request` diffs each `side.pokemon[i]` record against the last one it saw and stamps
    the mons whose record CHANGED with a monotone seq (`request_change_seq(species)`, re-exposed
    on `StrictBattleView` so the obs layer reads it through the boundary).
  * The proof that this is exact: `Pokemon.update_from_request` is a **pure function of that
    record** — it writes `active`, ability, condition, item, details, moves and stats, all read
    out of the dict and nothing else. An equal record therefore re-writes the same values, so an
    unchanged record proves the request did not mutate that mon.
  * Why per-mon and not "a request arrived": a request arrives on **every** decision, so a
    global signal would mark all six of our mons dirty every decision and delete the cache it
    exists to protect.
- **`offline_feed.py` — one side's recorded protocol into a `Gen3Battle`, no `Player`, no loop**
  (`gen3_offline_feed_v1`). `feed_chunk(battle, chunk)` is a DISPATCH MIRROR of
  `Player._handle_battle_message` (request → `parse_request`, win/tie → `won_by`/`tied`,
  `[Unavailable choice]` → `record_choice_rejected`, the player's `MESSAGES_TO_IGNORE` read from the
  class, `bigerror` dropped, the rest → `parse_message`). It is the Python half of the Rust Core
  parity harness. Gate: `offline_feed_test.py` (`sim`) — a live bridge battle's log == the offline
  feed's, every event, every field, both viewers, plus conservation (the transport's `|init|battle`
  room line is fed too).
- **`rust_core_parity.py` (+ `_test.py`) — the Rust Core parity harness, slice E (events)**
  (`gen3_core_parity_events_v1`, the Rust Core Program's M1). The Rust core's READING projection
  (`src/rust_sim/src/core_events/`) against THIS layer's `Gen3Battle`, per viewer, per event,
  `seq · turn · kind · side · actor · target · value · raw`, type-strict, NO allowlist — both fed the
  same per-side text (the core replays a recorded input log through the production bridge session;
  `Gen3Battle` is fed through `offline_feed`). COMMIT tier in the routine gate (~2 s: the recorded
  `rust_core_parity_fixtures/commit_tier.json.gz`, the four-shape byte-fuzz fixtures, one battle per
  protocol scenario, + the golden RECORD corpus's round trip); MILESTONE tier `slow` (2 × 360 random
  + 2 × 50 `production`-policy battles played live, 2 × 150 LADDER-USAGE battles, the protocol
  corpus × 2, every byte-fuzz fixture; pinned by `rust_core_parity_fixtures/manifest.json`). **Three
  team sources** (`utils.team_sources`): `play(key, source="ladder" | "procedural")` — slices T and O
  inherit them by calling `play`; the ladder tier's NAMED known divergences
  (`LADDER_KNOWN_DIVERGENCES`, each with its backlog row — EMPTY since the view projection's
  deletion) run in their own test and must still fire (`designs/ops/testing.md` → THREE TEAM SOURCES). 🚨 **This layer is now the ORACLE of a second
  implementation**: a change to `_build_event` / `_capture_pre` / the schema, or to a poke-env
  transition they read, fails the COMMIT tier the day it lands — mirror it in
  `src/rust_sim/src/core_events/reading.rs` in the same change (or behind a flag OFF in
  `production_config.json`). And `battle_event.py`'s tables are GENERATED into Rust:
  `python -m agents.battle.rust_core_schema --write` (`rust_core_schema_test.py` fails when stale).
  Contract: [`designs/rust_sim/core_events.md`](../../../designs/rust_sim/core_events.md).
- **`rust_core_parity_views.py` — slice V, the view + legality slice of the training observation
  path** (`gen3_core_parity_views_v1`). At EVERY decision of every recorded battle, both viewers, the
  `LiveView` + `LegalActions` training builds (a `Gen3Battle` fed through `offline_feed`, read at
  the exact chunk `Player._handle_battle_message` dispatches the decision on — `decision_points`)
  against the core column of `core_events --views`: every field classified SIM-FACT or PRESENTATION
  (a NAMED rule V1–V13). Rides `check_battles(…, views=ViewCensus())` — one harness, one core call
  per battle — at COMMIT (the recorded battles incl. two Baton Pass and two LADDER ones + the in-scope
  byte-fuzz fixtures, ~2 s) and MILESTONE (`slow`, the same played battles as slice E). gen3ou only.
  🚨 **This is the gate a Baton-Pass-class poke-env reading bug fails**: the core's view is audited
  against the ENGINE, so a reading that drops a sim fact diverges from it. Teeth: a re-introduced
  Baton Pass drop and a misread Spikes layer each FAIL a routine test. `offline_feed.new_battle(…,
  packed_team=)` mirrors the `Player`'s `_teambuilder_team`. (Until the Rust Core deletion pass,
  program §4 M2, it also compared the port's `one_sided_view` projection and ran reading-vs-engine
  truth checks off it; both went with `view.rs`. History and the rule table:
  [`designs/rust_sim/one_sided_view.md`](../../../designs/rust_sim/one_sided_view.md).)
  **The core column** (`core_events --views`):
  `present()` + `legal_actions()` + the 11-dim mask against the same `LiveView` / `LegalActions` /
  `Gen3ActionMasker`, type-strict, and the core's board audit (`check_view`, `[BOARD]`). The core
  reads the TRUTH, so a field where poke-env is WRONG is counted under its registered finding —
  value-aware, per decision — and never as a divergence (below).
- **`poke_env_findings.py` — the KNOWN poke-env READING findings** (`gen3_poke_env_findings_v1`).
  Where poke-env's reading is wrong about a sim fact (the Rust board / the pinned Showdown source /
  the request is the authority), the Rust core's `present()` carries the truth and the disagreement
  is registered here: ONE field, a VALUE-AWARE predicate, a minimal reproduction, the truth's source,
  whether it reaches the obs, and the obs blocks it may touch. Every comparison of the core against
  poke-env routes through `explain()`; a difference no entry explains is a divergence. 🚨 **Never a
  blanket tolerance, and never fixed from the core's side**: when the fork is fixed (a TRAINING-INPUT
  change, the owner's call), DELETE the entry and the check tightens. **Today the registry is
  EMPTY**: M2's three — PE-V10 (a fainted mon kept its stages), PE-R1b (the toxic count ticked at
  `|turn|`, not at the residual chip), PE-V16 (Flash Fire ended by its holder's Fire move), all
  reaching the obs — were FIXED in the fork as `gen3_pe_reading_fixes_v1` (2026-09-24, a
  TRAINING-INPUT change: `pokemon.py` `faint` / `note_residual_chip` / `moved`, `abstract_battle.py`
  `-damage`, `battle.py` `switch`), so every core-vs-reading comparison is exact. Pins:
  `poke_env/battle/reading_fixes_test.py` (protocol lines, each FAILS on upstream) +
  `training/poke_env_gaps/pe_reading_fixes_obs_integration_test.py` (three constructed real-Showdown
  battles — Curse + Self-Destruct, a Toxic pivot re-entering after the residual, a Flash Fire
  Houndoom — read at the obs). Contract + rates:
  [`designs/rust_sim/present.md`](../../../designs/rust_sim/present.md) §3.
- **`core_view.py` — a `LiveView` / `LegalActions` from the Rust core's `present()` JSON**
  (`gen3_core_present_v1`, the Rust Core Program's M2). A pure TRANSPORT: every presentation rule is
  applied in Rust (`src/rust_sim/src/present/`), so nothing here derives a field. Its consumers are
  slice V's core column (`rust_core_parity_views.py`) and the pins in
  `rust_core_present_test.py` (`sim`), which feed each reading rule's scenario to BOTH poke-env
  (through `offline_feed`) and the core (`core_events --present-stream`) and compare field by field.
  Slice T (`rust_core_parity_trackers.py`, `gen3_core_parity_trackers_v1`) holds the core's
  per-decision TRACKERS, α/β label and reward equal to the `EpisodeTracker` training drives over this
  package's `Gen3Battle` (`designs/rust_sim/trackers.md`).
  They also pin the core's REFUSALS (`gen3_core_error_v1`): a line poke-env raises on must fail in
  the core with the SAME exception class (`core_error.class`), and malformed input must never be
  reported as a poke-env class.
  🚨 **The poke-env data the core's reading consults is GENERATED from poke-env**:
  `python -m agents.battle.rust_core_present_tables --write` after any change to poke-env's pokedex,
  move table, `Effect` lifecycle sets or `SideCondition` (`rust_core_present_tables_test.py` fails
  when stale). Contract: [`designs/rust_sim/present.md`](../../../designs/rust_sim/present.md).
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
  `.events_since(cursor)`/`.event_cursor`, `.request_change_seq(species)` (the per-mon request
  door — the ONE mutation channel `events_since` cannot see, see above), and scalar meta
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
  action source; it is a documented seam exempted in `strict_api_lock_test`. Its `prev_mask`
  role went with the prev-turn action-mask obs block, `gen3_frame_deletion_v1`). The legacy
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
  for protocol-turn slicing. **An OUT-OF-BAND append is covered by the same slice**:
  `record_choice_rejected` is the only recorder outside `parse_message` (poke-env intercepts
  `|error|[Unavailable choice]` in `_handle_battle_message` and calls it directly), and its
  event still lands inside the NEXT decision's window, because the cursor is captured at
  decision time against the same log `_record` appends to — nothing about the parse pass is
  load-bearing for the window. Pinned end to end (protocol line → obs index) by
  `training/event_window_test.py`'s decision-cycle tests. Package re-exports are lazy
  (PEP 562 `__getattr__`) so importing one submodule doesn't force-load the others (avoids an
  import cycle).

**Verification:** unit tests in `src/agents/battle/*_test.py` (schema, registry audit,
scripted parse + state-equivalence, TurnView fold, `live_view_test.py` for the current-board
surface + widened fields, `live_view_memo_test.py` for the memo's four invalidation doors +
its store discipline + the clone-aliasing case, `live_view_build_micros_test.py` for the
build's three pure-function memos — the whole gen3 move universe against the spelled-out `max_pp`
formula, every branch in seven gens, per-instance isolation, and the enum-key-safety property
asserted on the ENUM CLASSES so an `IntEnum` conversion fails there rather than silently making
`Status.BRN` answer with another enum's name — `strict_view_test.py` for the strict boundary
+ `LegalActions` extraction + the forbidden-access guard **+ the two `__getattr__` branches**:
that hook fires both for a missing attribute AND for a property that raised AttributeError while
computing, and the boundary message used to assume the first, so a read-model field blowing up
inside `.live` surfaced as *"'StrictBattleView' has no attribute 'live'"* — a confident denial of
something that plainly exists, with the true cause four frames down and erased. It now tells the
two apart). The measurement instrument for the build is
`agents/training/live_view_build_benchmark.py` (order-alternated same-process A/B on ONE frozen
seeded board against a verbatim copy of the old code, with a field-identity check before timing
and a load-free `sys.setprofile` call count — the trainer-turn benchmark cannot do this job
because it walks a fresh random battle per invocation, so two runs profile two different boards).
The spine is
`event_log_fuzz_test.py` — real
`gen3ou` battles where both players run `Gen3Battle`; it independently re-derives each turn
from the intercepted raw protocol and asserts the event log matches, plus conservation +
event-kind coverage, the widened LiveView fields (spread/PP/consumed/counter/meta) per mon,
and the `LegalActions` legality surface against the live server request at every decision.
`live_view_memo_fuzz_test.py` is the memo's byte-identity spine: real bridge battles where
every decision asserts memo'd view == fresh rebuild AND obs-warm == obs-with-memo-cleared,
bit for bit. Its `--format gen3randombattle` arm is where Transform / Forecast forme change
live; check 2 is **skipped, loudly** there because the obs encoder is gen3ou-scoped and
fail-loud outside it (its effect vocabulary is derived for gen3 OU only — `observation/CLAUDE.md` → *The volatile vocabulary is SOURCE-DERIVED*).
Run both as scripts (no live server — bridge-backed): see the root `CLAUDE.md` Running Tests.
