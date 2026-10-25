# The ONE-SIDED VIEW — `gen3_one_sided_view_v1` — DELETED (history)

> 🗄️ **DELETED in the Rust Core deletion pass (program §4 M2 rows, 2026-09-26).** `view.rs`, the
> per-side reveal fold, `enable_view_fold`, the `view_pN` / `view_pN_at` payloads, `core_events
> --views`' `views` / `truth`, `view_adapter.py`, `view_successor.py`, `event_fold.py` and the
> search's `view` / `protocol` roads are gone: every search successor is a Rust-core version
> (`present.md`, `encoder.md`) and slice V compared the core's reading, audited against the engine,
> to the training `LiveView` (slice V and the Python `LiveView` / `Gen3Battle` side were themselves deleted in T27 P6 slices 6c / 6d-2). This document is kept as the RECORD of the projection, its reading
> rules (V1–V13, still the ids slice V and `present/` use) and the poke-env findings it surfaced;
> nothing below describes code that exists. The Node offline drivers (`search_driver.js`,
> `replay_driver.js`), the `impl="node"` / `search_impl` selector and the cross-impl harness
> `search_impl_parity.py` named below are deleted too (2026-10-10, owner: "delete node") — the
> Rust `search_driver` is the only offline driver ([`search_and_replay_drivers.md`](search_and_replay_drivers.md)).

<!-- HISTORY since the deletion pass: the code this described is deleted. -->

The Rust search server emits, per arm, the **projection of its own board onto what one side has
observed**, in exactly the shape `agents.battle.live_view.LiveView` holds — so a search
successor's observation can be built without replaying that ply's protocol text through poke-env.
The Python sub-encoders are **unchanged**: only the `LiveView`'s constructor differs.
What existed (all deleted in `97a30387a`, the Rust half; the Python half with it): a Rust projection
(`one_sided_view`, `SideObservation`) emitted by the search driver as `view_p1` / `view_p2` on `open_root` and on
every `expand_many` arm, a Python adapter that built a `LiveView` from it, and a differential, wall and truth-audit
gate set around them. Nothing below names a path that resolves: `git show 97a30387a^` has the code and its tests.


## 1. The wall

`pre_state` / `outcome` are the OMNISCIENT readouts and must never reach the encoder. The projection was the other
half of that wall, and a different object, not a filtered one:

* **`ours`** — the viewer's six mons in full. Nothing is hidden from its owner.
* **`opp`** — ONLY mons the viewer has seen switch in, each carrying only what the viewer was told: moves it watched
  being used, an item/ability the protocol disclosed, HP as the gen3ou `ceil%` fold rather than the true integer
  pair, no spread, no computed stats.

An unrevealed opposing mon had **no row at all** — not a redacted one — so the composition of the rest of the team
was not inferable from the payload's shape either. The Rust core's `present()` keeps this wall (`present.md`).

## 2. THE CENTRAL FINDING — `LiveView` is not a projection of sim state

Half of `LivePokemon` is **not a fact about the battle**; it is a fact about what poke-env's
tracker has folded out of the protocol, poke-env's own rules included. The port holds the board,
so the board half is free — but for the other half **the engine's value is the wrong answer**, and
in several places is wrong in a way that reads plausible:

| field | looks like sim state | is actually |
|---|---|---|
| `moves` (opponent) | the mon's moveset | the moves this side has WATCHED, each at `max_pp − sightings`, where a sighting against a Pressure holder costs **two** |
| `volatiles` | the sim's condition set | a fold over `\|-start\|`/`\|-end\|`/`\|-activate\|`/`\|-singleturn\|`/`\|-singlemove\|`, cleared on switch-out and faint (and by a `\|request\|` `0 fnt`, which re-runs `faint()`), with `ends_on_turn` effects DROPPED at the next `\|turn\|`, `is_turn_countable` ones counting up, and a Baton Pass copying `BATON_PASS_COPIED_EFFECTS` to the entrant |
| `status_counter` | the sim's sleep/toxic counter | `+1` per `\|move\|`-or-`\|cant\|` LINE while poke-env's OWN `_status` is asleep (a Sleep Talk turn is `+3`), `+1` per turn while badly poisoned AND active; frozen by a faint; the toxic one (only) reset by a switch-out; reset by `-curestatus` of the held status and by a NEW status (the fork's R1 fix, §4b) |
| `protect_counter` | the sim's `stall` denominator (0→2→4→8) | a plain consecutive-stall-move COUNT that resets on any non-stall move — a different quantity with different transitions |
| `revealed` | "is on the team" | set by the `\|switch\|` LINE, for BOTH sides alike |
| team ORDER | the team sheet | our own: the FIRST `\|request\|`'s roster order (Showdown floats the active mon to index 0 on every later one, poke-env's dict does not reorder). The opponent's: REVEAL order |
| `item` | the held item | disclosed only by `\|-item\|`/`\|-enditem\|` or a `[from] item:` clause on a `\|-damage\|`/`\|-heal\|` — **and on no other line**. Scanning every line resurrected a Salac Berry the preceding `\|-enditem\|` had consumed, because the `\|-boost\|` it triggers carries the same clause |
| `consumed_item` | the item the mon spent | set by `Pokemon.end_item` off `\|-enditem\|` and cleared only when a truthy item is set — **on BOTH sides**. The engine's `last_item` is a different quantity and read `None` where poke-env had a consumed Leftovers on an OWN mon |
| `ability` | the mon's ability | TWO slots whose getter prefers the temporary one: the setter fills `_ability` only while it is None and `_temporary_ability` after, `switch_out` clears the temporary, and the `-ability` handler's Trace case assigns TWICE. A Porygon2 that Traces Magnet Pull reads `magnetpull` while it is out and reverts to `trace` when it pivots |

Every one of these was found by the differential gate on real boards, not by reading code.

### The split this forces, and it is the contract

> **The port emits SIM FACTS and RAW PROTOCOL FACTS. The Python adapter applies every poke-env
> PRESENTATION RULE.**

The port therefore never needs poke-env's dex or its `Effect` enum, and a rule that is really
poke-env behaviour stays where the library it mirrors lives. Concretely the port sends a volatile's
announced NAME with its announcement history (never an id, never a counter), a move's SIGHTINGS
with each target's ability-event index at use time (never a `current_pp`), the stages a fainted
mon died with (never poke-env's stale ones), and the raw `|request|` bytes (never a parsed
legality).

🚨 **A corollary worth stating: the port must NOT send its own `protect_counter`, its own
`status`-counter, or an opponent's true PP even though it has all three.** Each is either a
different quantity or privileged information, and each read plausible enough to ship.

### 2b. The reading rules, named

Every field of the read-models is classified in `agents/battle/rust_core_parity_views.py`
(`MON_FIELDS` / `SIDE_FIELDS` / `VIEW_FIELDS` / `LEGAL_FIELDS`) as **SIM-FACT** — the projection
reads the ENGINE and a divergence is a reading bug on one side — or **PRESENTATION**, which names
one of these rules (`RULES` there, with the poke-env line each mirrors). A presentation field is
still compared EXACTLY; the rule is reproduced, never excused.

| rule | the reading | where the projection applies it |
|---|---|---|
| V1 | an opposing mon has a row once a `\|switch\|`/`\|drag\|` showed it; own `revealed` off the same line | `SideObservation.order` / `own_seen` |
| V2 | own slot order = the FIRST `\|request\|`'s roster; opponents' = reveal order | `own_order` / `order` |
| V3 | an opposing move's PP = `max_pp − sightings`, a sighting costing TWO when `_pressure_on` holds AT USE TIME (the target's ability as poke-env held it then; `_get_target_mon`'s default target for a target-less line and every `all`-target move; a `[from]move:` CALL reveals the called move with no use and charges the caller only its Pressure share) | `MoveObs.sightings` + `view_adapter._move` / `_AbilityAt` |
| V4 | the volatile fold, its lifecycle replayed from each effect's announcement history; Baton Pass carries | `VolObs.starts` / `bp_carried` + `view_adapter._volatiles` |
| V5 | poke-env's own `status_counter`, conditioned on poke-env's own `_status` — the sleep half per `\|move\|`/`\|cant\|` line, the toxic half the sim's stage (per residual `[from] psn` chip, reset at the switch-in; `gen3_pe_reading_fixes_v1`) | `MonObservation::pstatus` + `fold_status` |
| V6 | a consecutive-stall-move count; a faint keeps it, a switch-out zeroes it | `fold_status` |
| V7 | an opposing item from `-item`/`-enditem` or a `[from] item:` clause on `-damage`/`-heal` only | `observe` |
| V8 | the two ability slots, the single-ability inference, Trace, and the four non-`-ability` disclosures | `ability_disclosure` + `view_adapter._ability` |
| V9 | an opponent's spread / stats / exact HP are unknown | `mon_json(own=false)` |
| ~~V10~~ | RETIRED — a FAINTED mon holds NO stages, as the sim's faint `clearVolatile` leaves it; the fork was fixed to match (`gen3_pe_reading_fixes_v1`, PE-V10), so the payload's `boosts` stand and `MonState::faint_boosts`, `view_adapter`'s restore and `event_fold`'s boost ledger are deleted | — |
| V11 | a timed screen is stored as the TURN it started, Spikes as its layers | `SideObservation.screens` |
| V12 | weather `turns_active` = now − the `-weather` set turn | `weather_json` |
| V13 | the legality flags are poke-env's parse of the request | `legal_actions_from_payload` |

---

## 3. The payload — OBSOLETE

The `view_pN` / `view_pN_at` payloads, the reveal fold riding the bridge session and the lazy `enable_view_fold`
are deleted. The search driver's live payload contract is `search_and_replay_drivers.md`.

## 4. The GATE and what it reads — OBSOLETE

The search-road differential fuzz gate is deleted with the road. Its lesson stands: run a parity gate on at least two
fresh seeds before calling it green (three of that gate's findings appeared only on the second or later seed).

## 4a. The TRUTH AUDIT — slice V of the Rust Core parity harness (`gen3_core_parity_views_v1`)

Slice V compares, at EVERY decision of every recorded battle and for both viewers, the core's reading (audited
against the engine's truth by `present::check_view`) with the training `LiveView` field by field in §2b's classes,
with no allowlist and an `[ALIGN]` divergence, never a skip, where the two sides cannot be aligned. Its PROJECTION
column and the Python `LiveView` side are deleted; the core column and the board audit are `present.md`'s.

## 4b. The three read-model findings — CLOSED (2026-09-23), and what the truth audit found next

All three were reproduced on FIXED, recorded battles by the Rust Core parity harness's slice V
(§4a), which runs this projection at EVERY decision rather than at three roots per battle. Each
was established against the simulator's own board (the engine, cross-checked with the pinned
Showdown source where the engine and poke-env disagreed), and every one was the PROJECTION's error
— poke-env's reading was the rule to reproduce:

| finding | root cause (the poke-env line the fold failed to mirror) | fixed by | pinned by |
|---|---|---|---|
| own/opp `status_counter` drifting in BOTH directions | **Two mechanisms.** HIGHER: `-cureteam` (Aromatherapy / Heal Bell) → `cure_status()` clears `_status` and not the counter, but the fold kept counting `\|move\|` lines as asleep. LOWER: the `status` SETTER (`-status`) does NOT reset `_status_counter`, while the fold zeroed it — a Rest taken while badly poisoned starts poke-env's sleep count at the toxic count. The fold keyed its increments on its own `-status`-only flag instead of poke-env's `_status` | `view.rs` now mirrors poke-env's `_status` (`MonObservation::pstatus`) through every line that writes it: `-status`, `-curestatus` (only when it names the held status), `-cureteam`, `faint`, the HP token of `-damage` / `-heal` / `-sethp` / `switch`, and every own `\|request\|` roster `condition` | rule V5; `one_sided_view_test::v5_the_status_counter_follows_poke_envs_own_status` |
| `opp.<species>.ability` reading `None` where poke-env had it (a Snorlax's `immunity`) | poke-env ASSIGNS an ability off four non-`-ability` lines, each at an exact shape: `-immune\|X\|[from] ability: A` (4 fields), `_check_heal_message_for_ability` (6 fields, the healed mon), `_check_damage_message_for_ability` (6 fields, the `[of]` mon), and `-activate\|X\|ability: A` (only `if holder_mon.ability is None`). The Snorlax was `\|-immune\|p1a: Snorlax\|[from] ability: Immunity` | `view.rs::ability_disclosure` + `AbilityEvent::if_unknown`; 360 occurrences over 400 battles → 0 | rule V8; `v8_an_ability_is_disclosed_off_four_non_ability_lines`, `view_adapter_test::test_an_activate_disclosure_fills_the_ability_ONLY_while_it_is_unknown` |
| own `volatiles` missing `substitute` | **Baton Pass.** The sim's `copyVolatileFrom` and poke-env's `Pokemon.apply_baton_pass` carry `BATON_PASS_COPIED_EFFECTS` to the entrant; the fold wiped the entrant on its `\|switch\|`. The engine truth agrees with poke-env (the entrant holds the Substitute) | `view.rs::baton_pass_carry` flags the passer's volatiles `bp_carried`; the adapter keeps poke-env's copied set | rule V4; `v4_baton_pass_carries_the_passers_volatiles_to_the_entrant`, `view_adapter_test::test_a_BATON_PASSED_volatile_survives_only_if_poke_env_copies_it`, and the COMMIT tier's recorded battle `random_34` |

**The same audit closed eleven more PROJECTION classes** the three-root sweep never reached:
deferral **D6** (Pressure judged at READ time — each sighting now carries the target's
ability-event index at USE time, both sides, so an own Porygon2's Traced Pressure counts); a
move line with NO target charging our active (`_get_target_mon`); an `all`-target move (Perish
Song) taking the default target; a `[from]move: Sleep Talk` call revealing the called move with no
use and charging the caller only its Pressure share; a re-announced `ends_on_turn` effect being a
fresh one (the volatile fold now carries its announcement history and the adapter replays
poke-env's lifecycle); a fainted own mon's `|request|` `0 fnt` re-running `faint()` (clearing a
post-KO Destiny Bond); `Pokemon.faint` keeping the protect streak; a fainted mon keeping the stages
it died with until switched out (rule V10 — since retired: the fork now clears them at the faint,
`gen3_pe_reading_fixes_v1`); a timed screen stored as the
TURN it started (rule V11); the first decision reading turn 0 (`bs.turn` lags the framing's
`|turn|1` until the first commit); a fainted Traced mon's ability reverting to its base; and a
Transformed mon keeping its OWN species as its identity.

### What the truth audit found that is NOT the projection's — poke-env READING defects

These are cases where the ENGINE (the sim's truth) and poke-env disagree about a sim fact and
poke-env is wrong. Each fix belongs in the vendored fork and **changes what training reads**; R1–R3
are FIXED (a TRAINING-INPUT change, `designs/CHANGELOG.md` 2026-09-23) and the projection follows
them, R1b and R4 remain (measured per 1,000 decisions, both viewers, before the fixes):

| defect | truth (and how established) | per 1,000 decisions: pool random (73,605) · `production`-policy (10,291) · procedural (`ou_random_teams.js`, 33,298) |
|---|---|---|
| **R1** — the `status` setter never resets `_status_counter`, so a NEW status inherits the previous one's count (Rest while badly poisoned; a re-sleep after a cure the watcher saw only as a bare HP token). The obs's sleep counter AND the 3-dim sleep-wake belief (`K` = cant-turns) read it, so a fresh Rest can read "slept 3 turns, wakes next" with the reliability bit SET | for sleep, the `\|cant\|…\|slp` turns since the sleep began: as-is wrong on **886 of 12,201** asleep-mon decisions (Sleep Talk episodes excluded — the obs already flags those), a reset-on-change setter wrong on **0**. For toxic, the engine's `Toxic(stage)` | **11.09 · 7.48 · 9.22** decisions where an asleep / badly-poisoned mon's counter differs |
| **R1b** — the toxic counter ticks at every `\|turn\|` a badly-poisoned mon is active, so one that entered AFTER the residual (a post-faint replacement) reads one ahead of the sim's stage (**FIXED 2026-09-24**, `gen3_pe_reading_fixes_v1`, as M2's PE-R1b) | the engine's `Toxic(stage)`: 104 of 6,158 badly-poisoned-mon decisions with R1 fixed | ~1.2 (pool) |
| **R2** — `-copyboost` is read BACKWARDS: poke-env copies the FIRST ident's stages onto the second, the sim does the reverse (`data/mods/gen5/moves.ts` psychup, which gen 3 inherits: `source.boosts[i] = target.boosts[i]; this.add('-copyboost', source, target)`; `SIM-PROTOCOL.md`'s wording says the opposite and poke-env followed the doc). After a Psych Up BOTH mons' stages read wrong | the engine + the pinned Showdown source | 0 · 0 · **0.27** (1 of the 719 pool teams carries Psych Up) |
| **R3** — our OWN move PP is a sighting counter never synced to the `\|request\|`'s `pp`, so a PP the sim deducts without poke-env knowing (a foe's Pressure it cannot infer — gen 3 announces Pressure to its owner only, e.g. an Aerodactyl) drifts it HIGH; the old deferral D7, now a measured defect | the `\|request\|` (the sim's word) and the engine | 0 · 0 · **6.19** — and **7.15 on the WHOLE training pool** (465 of 64,991 decisions, 360 battles striding all 719 teams): the MILESTONE key range (below) never reaches the pool teams that trigger it |
| **R4** — a TRANSFORMED mon's ability / stats / watched moves are poke-env approximations the projection does not yet present (own ability reads the base one — a gen-3 request never states the copied ability) | the engine | 0 · 0 · 0.06 (no pool team carries Transform) |

**Also found by the procedural sweep, and it was the PORT's**: every move lock other than the
two-turn charge (Rollout, Ice Ball, Outrage, Thrash, Petal Dance, Uproar) shipped a `|request|`
naming **Solar Beam** on the training transport, which poke-env's live player asserts on (1 crash
in 200 procedural battles; zero pool exposure; live on the websocket front end every anchor read
plays over). Established against the pinned Showdown's own requests for all six; FIXED in
`bridge.rs` and pinned by `tests/locked_request_test.rs` (`gen3_locked_request_move_v1`).

**MILESTONE coverage hole**: keys 5000–5199 wrap (mod 719) onto pool teams 686–718 and 0–166,
overlapping keys 0–199 — the two random seeds cover ~234 of the 719 pool teams. Widening the recipe
makes R3 fire in the pool tier, so it lands with R3's fix, not before.

**R1–R3 are FIXED in the fork** (`pokemon.py` status setter, `abstract_battle.py` `-copyboost`,
`battle.py` `_sync_active_pp`), the projection's V5 following R1 and `event_fold`'s boost ledger
following R2 (their pins went with the fork, T27 P6); the
MILESTONE random recipe strides the whole pool (even keys 0–718 + odd keys 1–719) and is green. R1b
was fixed with M2's findings (below). Still open after it: R4, and the
part of R3 no client can see — a mon that leaves the field between its move and the next request
(Self-Destruct, a phaze) is never re-synced, so our BENCH PP is poke-env's count, a presentation
rule (V15) the projection does not yet reproduce; procedural teams only so far.

### Since M2: the core reads the TRUTH, and poke-env's remaining errors are FINDINGS

The Rust core's `present()` (`designs/rust_sim/present.md`) does not reproduce a poke-env mistake:
where poke-env is wrong about a sim fact, the core carries the truth and slice V's core column counts
the disagreement under a registered finding (the registry module is deleted with the fork). M2 registered
three — **PE-V10** (this projection's V10: a fainted mon's stages), **PE-R1b** (R1b above) and
**PE-V16** (poke-env's `moved` ended Flash Fire on its holder's own Fire move; the sim keeps it
until switch-out) — and all three are FIXED in the fork (`gen3_pe_reading_fixes_v1`, 2026-09-24, a
TRAINING-INPUT change), so the registry is empty. This projection (the view road) reproduces
poke-env, which is what its column is compared against, so it followed: V10 retired, V5's toxic half
moved to the residual chip; it never ended Flash Fire on a Fire move.

## 5. DEFERRALS — OBSOLETE

D1-D10 (sub-encoders reading the raw `Pokemon`, the Wish and sleep-wake logs, the trackers, Pressure at use time, own
PP, Mimic / Transform overlays, `Mist`, the intermediate-decision board) were the view road's deferrals; the road is
deleted. The reading defects they surfaced are in §4b.

## 5b. D10 — CLOSED (`gen3_view_at_intermediate_v1`) — OBSOLETE

An arm whose ply KO'd one of our mons (or refused a trapped switch) resolves a SECOND decision inside the same
`expand_many` arm, so the board the driver returned was one decision past the row the protocol road produced. The
close captured the board at the top of every loop iteration, cut the arm's events at the chunk that closed the first
decision, and fixed two poke-env presentation rules the gate found: the out-of-band `CHOICE_REJECTED` event, and a
fainted mon's stages (PE-V10, fixed in the fork). All of it is deleted with the road.

## 6. Cost — OBSOLETE

The view road's per-arm wall was measured and optimised (side elision, one fork per decision, lazy `action_choices`);
the record is `designs/research_state/measurements/search_profile_2026-09-22/README.md`. The side-elision contract
(`expand_many`'s `side`) survives in `search_and_replay_drivers.md`.

## 7. Transport status, and THE FLIP — OBSOLETE

`--materializer {protocol,view}` and its fallbacks are deleted; every search successor is a Rust-core row
(`present.md`, `encoder.md`).
