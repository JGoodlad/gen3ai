# The ONE-SIDED VIEW — `gen3_one_sided_view_v1`

<!-- ALWAYS-CURRENT (this tree OWNS what it holds; see the root CLAUDE.md). State the truth, never
narrate a change. The code is `src/rust_sim/src/view.rs` + `src/agents/battle/view_adapter.py`. -->

The Rust search server emits, per arm, the **projection of its own board onto what one side has
observed**, in exactly the shape `agents.battle.live_view.LiveView` holds — so a search
successor's observation can be built without replaying that ply's protocol text through poke-env.
The Python sub-encoders are **unchanged**: only the `LiveView`'s constructor differs.

| | |
|---|---|
| **Rust** | `src/rust_sim/src/view.rs` (`one_sided_view`, `SideObservation`); emitted by `src/bin/search_driver.rs` as `view_p1` / `view_p2` on `open_root` AND on every `expand_many` arm |
| **Python** | `src/agents/battle/view_adapter.py` (`live_view_from_payload`, `legal_actions_from_payload`, `ViewBattle`); reachable as `LiveView.from_view_json` |
| **Wall gate** | `src/rust_sim/tests/one_sided_view_test.rs` (22 tests — the wall, the D10 capture, and one pin per reading rule, §2b) |
| **Unit gate** | `src/agents/battle/view_adapter_test.py` (33 tests, the newer ones checked against poke-env fed the same protocol) |
| **Differential gate** | `src/agents/battle/one_sided_view_parity_fuzz_test.py` (`sim`) — the SEARCH road (roots + arms) |
| **Truth audit** | `src/agents/battle/rust_core_parity_views.py` — slice V of the Rust Core parity harness: this projection AND the engine truth against the `LiveView` training builds at EVERY decision, both viewers (§4a) |
| **Benchmark** | `src/agents/training/view_materialize_benchmark.py` (the two ROADS on a hand-built arm set) and `src/main/search_dividend/search_decision_benchmark.py` (the real `SearchEngine` over BANKED eval traces, broken into phases) — they answer different questions, keep both |

---

## 1. The wall

`pre_state` / `outcome` are the OMNISCIENT readouts and must never reach the encoder. This is the
other half of that wall, and it is a different object, not a filtered one:

* **`ours`** — the viewer's six mons in full. Nothing is hidden from its owner.
* **`opp`** — ONLY mons the viewer has seen switch in, each carrying only what the viewer was
  told: moves it watched being used, an item/ability the protocol disclosed, HP as the gen3ou
  `ceil%` fold rather than the true integer pair, no spread, no computed stats.

An unrevealed opposing mon has **no row at all** — not a redacted one — so the composition of the
rest of the team is not inferable from the payload's shape either.

**Proof, not assertion**: `one_sided_view_test::the_view_hides_every_unrevealed_opposing_mon_that_pre_state_shows`
takes a real mid-battle board, reads the omniscient `pre_state` for the species the view does not
list, and asserts each is absent from the view's **bytes**. It carries a non-vacuity guard (there
must BE an unrevealed mon) and a symmetric twin for p2.

---

## 2. THE CENTRAL FINDING — `LiveView` is not a projection of sim state

Half of `LivePokemon` is **not a fact about the battle**; it is a fact about what poke-env's
tracker has folded out of the protocol, poke-env's own rules included. The port holds the board,
so the board half is free — but for the other half **the engine's value is the wrong answer**, and
in several places is wrong in a way that reads plausible:

| field | looks like sim state | is actually |
|---|---|---|
| `moves` (opponent) | the mon's moveset | the moves this side has WATCHED, each at `max_pp − sightings`, where a sighting against a Pressure holder costs **two** |
| `volatiles` | the sim's condition set | a fold over `\|-start\|`/`\|-end\|`/`\|-activate\|`/`\|-singleturn\|`/`\|-singlemove\|`, cleared on switch-out and faint (and by a `\|request\|` `0 fnt`, which re-runs `faint()`), with `ends_on_turn` effects DROPPED at the next `\|turn\|`, `is_turn_countable` ones counting up, and a Baton Pass copying `BATON_PASS_COPIED_EFFECTS` to the entrant |
| `status_counter` | the sim's sleep/toxic counter | `+1` per `\|move\|`-or-`\|cant\|` LINE while poke-env's OWN `_status` is asleep (a Sleep Talk turn is `+3`), `+1` per turn while badly poisoned AND active; frozen by a faint; the toxic one (only) reset by a switch-out; reset by `-curestatus` of the held status and NOT by a new status (§4b, R1) |
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
| V5 | poke-env's own `status_counter`, conditioned on poke-env's own `_status` | `MonObservation::pstatus` + `fold_status` |
| V6 | a consecutive-stall-move count; a faint keeps it, a switch-out zeroes it | `fold_status` |
| V7 | an opposing item from `-item`/`-enditem` or a `[from] item:` clause on `-damage`/`-heal` only | `observe` |
| V8 | the two ability slots, the single-ability inference, Trace, and the four non-`-ability` disclosures | `ability_disclosure` + `view_adapter._ability` |
| V9 | an opponent's spread / stats / exact HP are unknown | `mon_json(own=false)` |
| V10 | a FAINTED mon keeps the stages it died with until switched out | `MonState::faint_boosts` + `view_adapter._mon` |
| V11 | a timed screen is stored as the TURN it started, Spikes as its layers | `SideObservation.screens` |
| V12 | weather `turns_active` = now − the `-weather` set turn | `weather_json` |
| V13 | the legality flags are poke-env's parse of the request | `legal_actions_from_payload` |

---

## 3. The payload

```text
{"side":"p1","turn":N,"finished":bool,"won":bool|null,"lost":bool|null,
 "weather":{"weather":id|null,"is_permanent":bool,"turns_active":n},
 "ours":SIDE,"opp":SIDE,
 "request":<the |request| payload verbatim>|null}

SIDE = {"team_size":n,"active":<species id>|null,"side_conditions":{id:n},"mons":[MON,…]}

MON  = {"species",                     # the IDENTITY species — its own, even while Transformed
        "active","fainted","revealed",
        "hp_fraction","current_hp","max_hp",
        "status","status_counter","protect_counter",
        "types":[…],"moves":[MOVE,…],
        "item","consumed_item",
        "ability",                       # OURS only — the engine's (the BASE one once fainted)
        "base_ability",                  # OURS only — the set's (poke-env's own base slot)
        "ability_events":[{"id","trace","if_unknown"},…],  # announcements in order, BOTH sides;
                                         #   id "" = left the field or fainted (temp slot cleared)
        "boosts":{stat:stage},          # NONZERO stages only, as LiveView keeps them
        "faint_boosts":{stat:stage}|null,   # the stages it DIED with (V10); null while alive
        "volatiles":[{"name","starts":[tick,…],"now":tick,"bp_carried":n},…],
        "base_stats":{…},
        "ivs"|null,"evs"|null,"nature"|null,"spread_known",
        "stats":{…}|null}               # null == poke-env's all-None dict (an opponent's)

MOVE = own:     {"id","move_id","current_pp","max_pp"}
       watched: {"id","move_id","uses","max_pp","sightings":[SIGHTING,…]}
SIGHTING = {"mv","called","t","t_own","t_k","d","d_k","n"}   # rule V3: the move (or CALLED
         # move) whose target type decides Pressure, the named target (species, its side, its
         # ability-event index at USE time), the default target, and the count of identical ones
```

`id` is the poke-env **moves-dict KEY** and `move_id` is the `Move.id` it maps to; they differ for
a typed Hidden Power (the wire re-keys it bare) and the two are sorted on by **different**
consumers — `LiveView.moves` by the key, `MovesEncoder` by `Move.id`.

### `view_p1_at` / `view_p2_at` — the boards at the decisions the ARM resolved itself

Every `expand_many` arm carries two more fields beside `view_p1` / `view_p2`
(`gen3_view_at_intermediate_v1`):

```text
"view_p1_at":[VIEW,…],"view_p2_at":[VIEW,…]      # same object as view_pN, ORDERED, usually EMPTY
```

A turn can contain more than one request round: our mon faints mid-turn and the replacement is a
SECOND request inside the same arm, or a trapped switch is refused and the request re-opens.
`resolve_turn_sourced` answers those from its follow-up policy, so `view_pN` is the board one
decision PAST the row a per-request consumer needs. Entry `k` of `view_pN_at` is
`one_sided_view(sess, N, dex)` rendered at the **top of the loop iteration that round opened** —
before any of its choices are committed — so `view_pN_at.len()` equals the number of `|request|`
lines that side's suffix carries **minus the last**, which is exactly what
`view_successor.intermediate_decisions` counts off the protocol.

🚨 **EMPTY is the normal case and an empty array is not a missing field.** The turn's final
request is never in here (the loop exits once the boundary turn moves) and an ordinary arm opens
no decision at all. It is also empty on a `recorded_exact` arm and under `impl="node"`; a consumer
that finds no entry falls back exactly as it did before the field existed. Pinned by
`one_sided_view_test::a_mid_turn_faint_captures_the_view_AT_its_replacement_request` and its
NEGATIVE twin `an_ordinary_turn_captures_no_intermediate_view` (the predicate was fault-injected:
capturing on iteration 0 fails both).

### The reveal fold rides the session, not the chunks

`SideObservation` lives on `BridgeChunks` (the one funnel every per-side line passes through) but
**survives `clear_chunks()`**: a branch drops its parent's chunk HISTORY so its own chunks are its
suffix, while what a side has SEEN is cumulative from turn 1. Resetting it would make every branch
claim the opponent's team is unrevealed. Pinned by
`one_sided_view_test::the_reveal_fold_survives_clear_chunks_and_the_snapshot`.

---

## 4. The GATE and what it reads

`one_sided_view_parity_fuzz_test.py` drives real gen3ou battles, opens a search root at
three turns per battle, expands five arms each, and compares BOTH roads at every point:

1. the whole `LiveView` graph, field by field, per mon;
2. `LegalActions`, field by field;
3. the **2501-dim observation vector**, `np.array_equal` on float32.

The obs comparison threads **no trackers and no assembler** on either side. That is what makes it
sharp rather than broad: the tracker-fed blocks are then zero in both vectors, so every surviving
difference is attributable to the read-models and to nothing else. There is **no allowlist** — the
run prints a CENSUS keyed by field path and fails on any non-empty census.

**Exactly TWO classes are DECLARED residual** (`DECLARED_RESIDUAL` in the gate), each naming its
deferral below, each matched by a NARROW path predicate, and each PRINTED with its count — or
printed as NOT SEEN — on every run. Everything else fails. The narrowness is enforced by
construction: the Wish entry matches only the two reactive columns the layout DECLARES, and the PP
entry only a `current_pp` disagreement on a watched moveset (the move id list and every `max_pp`
are still strict), so a wrong revealed-move SET cannot hide behind it. A decision at which a
declared residual fires has its VECTOR marked inconclusive rather than compared — a vector
downstream of a known input difference is not evidence either way.

🚨 **TWO ENTRY POINTS, and the split is the project's fuzz rule.** The SWEEP records fresh random
battles and runs as a script; the COLLECTED test takes its battle from
`obs_roundtrip_fuzz_test.record_fixture_battle(key=…)` — pinned teams, per-player RNG and sim seed
— so it is the same board every run and cannot ride main red on a draw nobody has seen (verified:
three runs, 54 comparisons each, identical).

🚨 **RUN THE SWEEP ON AT LEAST TWO FRESH SEEDS BEFORE CALLING IT GREEN.** Three of the findings
above appeared only on the SECOND or THIRD seed — the own-side `consumed_item` and the Traced
ability among them — and a single clean sweep would have read as a green gate.

**A D10 arm is COMPARED, not deferred** (§5b). It gets a SECOND protocol road, fed the prefix plus
the CUT and nothing else — the padded road above deliberately runs PAST the replacement, so its
read-models describe the next turn and say nothing about the board `view_pN_at[0]` carries. Both
the read-model census and the tracker-fed successor comparison run on that road, and the collected
test asserts `d10 >= 1` so a fixture whose arms stop KO-ing cannot make the gate vacuous.

---

## 4a. The TRUTH AUDIT — slice V of the Rust Core parity harness (`gen3_core_parity_views_v1`)

The search-road gate above compares at three roots and their arms per battle. The truth audit
compares at **every decision of every recorded battle, for both viewers**, on the TRAINING
observation path: `core_events --views` replays a recorded input log through the production
bridge session and captures BOTH `one_sided_view`s plus the ENGINE truth at the end of every write
that shipped a `|request|`; the reference is a `Gen3Battle` fed each viewer's text through
`offline_feed` and read at the exact chunk `Player._handle_battle_message` dispatches the decision
on (`decision_points`). Field by field (§2b's classes), plus the TRUTH checks no projection can
make: a revealed opposing item / ability / move / type against the OTHER viewer's engine-sourced
`ours` row, a consumed item no longer held, and ten sim-state volatiles (`TRUTH_VOLATILES`)
present on the reading exactly when the engine holds them. No allowlist; a decision the two sides
cannot align is an `[ALIGN]` divergence, never a skip. Scope: gen3ou (the training obs path is
gen3ou-only); the `gen3customgame` scenario corpora are counted as out of scope and printed.

**Why `one_sided_view` is the projection, not the M1 core**: M1's `CoreEvent`s carry the event
reading and no board; this is the only board projection that exists, and M2's `present()` replaces
it under the same slice (`designs/endstate/program_rust_core.md` M2).

| tier | corpus | wall (load ~15-20 / 16 cpus) |
|---|---|---|
| COMMIT (routine gate, `rust_core_parity_test.py`) | the 10 recorded battles (incl. the two **Baton Pass** battles `random_34` — a passed Substitute — and `random_177` — passed Calm Mind stages) + the 3 in-scope byte-fuzz fixtures: **13 battles, 1,838 decisions** | ~2 s (the file ~6 s) |
| MILESTONE (`slow`, verdicts in `designs/ops/slow_tier_status.json`) | 2 × 200 seeded-random + 2 × 50 `production`-policy battles played live — the same battles slice E runs: **83,896 decisions, 23.9M field comparisons, 1.94M truth checks** | ~70 s check per 200 battles + play |

**Teeth**, each a routine test: re-introducing the pre-2026-08-23 Baton Pass drop FAILS on the
entrant's SIM-FACT `boosts` and its engine-truth `volatiles`; a misread Spikes layer FAILS on
`side_conditions`; a dropped capture FAILS as `[ALIGN]`; a read-model field added without a class
FAILS `test_every_read_model_field_is_classified`.

---

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
it died with until switched out (`MonState::faint_boosts`, rule V10); a timed screen stored as the
TURN it started (rule V11); the first decision reading turn 0 (`bs.turn` lags the framing's
`|turn|1` until the first commit); a fainted Traced mon's ability reverting to its base; and a
Transformed mon keeping its OWN species as its identity.

### What the truth audit found that is NOT the projection's — poke-env READING defects

These are cases where the ENGINE (the sim's truth) and poke-env disagree about a sim fact and
poke-env is wrong. The projection reproduces the reading today so the gate stays exact; each fix
belongs in the vendored fork and **changes what training reads**, so each waits on the
orchestrator (measured per 1,000 decisions, both viewers):

| defect | truth (and how established) | per 1,000 decisions: pool random (73,605) · `production`-policy (10,291) · procedural (`ou_random_teams.js`, 33,298) |
|---|---|---|
| **R1** — the `status` setter never resets `_status_counter`, so a NEW status inherits the previous one's count (Rest while badly poisoned; a re-sleep after a cure the watcher saw only as a bare HP token). The obs's sleep counter AND the 3-dim sleep-wake belief (`K` = cant-turns) read it, so a fresh Rest can read "slept 3 turns, wakes next" with the reliability bit SET | for sleep, the `\|cant\|…\|slp` turns since the sleep began: as-is wrong on **886 of 12,201** asleep-mon decisions (Sleep Talk episodes excluded — the obs already flags those), a reset-on-change setter wrong on **0**. For toxic, the engine's `Toxic(stage)` | **11.09 · 7.48 · 9.22** decisions where an asleep / badly-poisoned mon's counter differs |
| **R1b** — the toxic counter ticks at every `\|turn\|` a badly-poisoned mon is active, so one that entered AFTER the residual (a post-faint replacement) reads one ahead of the sim's stage | the engine's `Toxic(stage)`: 104 of 6,158 badly-poisoned-mon decisions with R1 fixed | ~1.2 (pool) |
| **R2** — `-copyboost` is read BACKWARDS: poke-env copies the FIRST ident's stages onto the second, the sim does the reverse (`data/mods/gen5/moves.ts` psychup, which gen 3 inherits: `source.boosts[i] = target.boosts[i]; this.add('-copyboost', source, target)`; `SIM-PROTOCOL.md`'s wording says the opposite and poke-env followed the doc). After a Psych Up BOTH mons' stages read wrong | the engine + the pinned Showdown source | 0 · 0 · **0.27** (1 of the 719 pool teams carries Psych Up) |
| **R3** — our OWN move PP is a sighting counter never synced to the `\|request\|`'s `pp`, so a PP the sim deducts without poke-env knowing (a foe's Pressure it cannot infer — gen 3 announces Pressure to its owner only, e.g. an Aerodactyl) drifts it HIGH; the old deferral D7, now a measured defect | the `\|request\|` (the sim's word) and the engine | 0 · 0 · **6.19** — and seen on the POOL by the search-road sweep (2 of 24 battles): 3 pool teams carry an un-inferable Pressure Aerodactyl, none of them inside the MILESTONE key range (below) |
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

### Standing rule

**Run slice V's MILESTONE tier before trusting a change to `view.rs`, `view_adapter.py`,
`offline_feed.py`, or any poke-env reading.** The pool alone is not enough: R2, R4 and the locked
request have ZERO pool exposure and appeared only on the procedural generator's teams, and the
MILESTONE pool range does not reach R3's.

## 5. DEFERRALS — what is knowingly not carried

| id | what | why, and what it would take |
|---|---|---|
| **D1** | the obs SLOT order still comes from `battle.team` / `battle.opponent_team` | `state_encoder.encode` walks `base.get_team_list(battle, …)` to assign slots and only then looks the mon up in the `LiveView`. So an ordered identity list must exist whatever else happens; `ViewBattle` supplies it FROM the view |
| **D2** | four sub-encoders never took a `live_mon` | `items` (`mon.item`/`mon.consumed_item`), `abilities` (`mon.ability`), `types` (`mon.type_1`/`type_2`) and `moves` (`mon.moves`) read the raw `Pokemon` unconditionally, though `LivePokemon` carries every one of those fields and the rest of `pokemon.encode` is `if live_mon is not None:` throughout. `ViewMon` feeds them from the same read-model in the meantime. **Migrating those four onto `live_mon` is the right fix** and is a value-neutral refactor that owes the obs-build benchmark |
| **D3** | ✅ **CLOSED for a successor** — the reactive block's **Wish pair** | `reactive.encode` folds `battle.events` through `wish_belief.build_wish_pending`. The fix was not the port's `SideState::wish_pending` at all: the payload was never the problem, the missing LOG was. `ViewBattle` now carries the successor's whole-battle event log (the root's, plus the ply folded by `ViewEventFolder`) and the existing fold runs unchanged. Still OPEN for a BOARD-ONLY caller with no log, which is exactly what the gate's tracker-less comparison is — so the residual is still DECLARED there, and seen there |
| **D4** | ✅ **CLOSED for a successor** — the 3-dim **sleep-wake belief** | `build_sleep_sources(battle)` reads `battle.events` and `battle.turn` and nothing else, so the same log closes it. Same board-only caveat as D3 |
| **D5** | ✅ **CLOSED** — the per-decision **TRACKERS** — recency, pair history, the event window, the progress clock, the Hidden-Power block | `gen3_view_successor_v1` — see §7. The ply's events are folded from the arm's OWN one-sided protocol by `agents/battle/event_fold.py`, and the root's `EpisodeTracker` is carried forward and advanced through `record_context` / `advance_window` (the bodies of `record` / `update_progress_clock`, split out rather than copied) |
| ~~**D6**~~ | ✅ **CLOSED** — Pressure is judged at USE time | Each sighting carries its target's ability-event index at use time (both sides — our own Traced Pressure counts), and the adapter replays poke-env's two-slot ability rules up to it (rule V3). The un-fainted clause is still not re-checked at read time: a mon cannot be targeted while fainted |
| **D7** | our OWN pp is the wire's, not poke-env's counter | Now a measured poke-env READING defect (**R3**, §4b): poke-env never syncs our own move PP from the `\|request\|`, so a PP the sim deducts without poke-env knowing (an un-inferable foe Pressure) drifts it HIGH. The payload keeps sending the engine's `current_pp` — the truth — and slice V classifies own PP as SIM-FACT, so the drift FAILS the gate where it occurs (0 per 1,000 on the MILESTONE pool corpus, 6.19 on procedural teams, and present on pool teams outside that corpus). The fix belongs in the fork and waits on the orchestrator |
| **D8** | Mimic / Transform move overlays on our own side | The own moveset is rendered from `set.moves`; an overlay would need the same resolver the request path uses |
| **D9** | `Mist` as a side condition | The port models spikes / reflect / lightscreen / safeguard only |
| ~~**D10**~~ | ✅ **CLOSED** — an arm whose ply resolved an INTERMEDIATE decision | `gen3_view_at_intermediate_v1`. See §5b below |

## 5b. D10 — CLOSED (`gen3_view_at_intermediate_v1`)

**The mechanism.** When an arm's ply KOs one of our mons, the replacement round is a SECOND
request inside the same `expand_many` arm; a refused trapped switch does the same thing by
re-opening the request. `search.rs::resolve_turn_sourced` loops
`while !sess.is_ended() && open_boundary_turn(sess) == start_turn` and answers those rounds from
`followup_choice`, so the board it returns as `view_pN` is one decision PAST the row
`materialize_branches` produces. A leaf scored at the wrong one of them is not a rounding
difference. It fired on **16.8–17.1% of arms** on mid-game decisions banked from a real run's
`eval_traces` (and 6.9% of branch points on 24 fresh battles — the same mechanism at different
anchor distributions), and the per-arm fallback into `materialize_branches` cost **17.2% of the
view road's decision wall**.

**The close has THREE parts and only the first was the one on the design.**

| part | where | what it does |
|---|---|---|
| the BOARD | `search.rs` + `bin/search_driver.rs` | `Resolved::views_at` captures `one_sided_view` at the top of every loop iteration that answers a round the source could not, and the driver emits it as `view_pN_at` (§3). Nothing about poke-env enters the port |
| the EVENT CUT | `view_successor.split_at_intermediate` | the view road folds WHOLE chunks while `materialize_branches` stops at the `\|request\|` that opened the decision, so the arm's chunks are cut at the **chunk that closed the first decision** — a CHUNK boundary because that is poke-env's own rule (`Player._handle_battle_message` parses a whole message and only THEN dispatches the request), and a line-level cut would stop earlier than the protocol road does |
| the two poke-env RULES the port cannot supply | `event_fold.py` + `view_adapter.py` | below |

🚨 **THE TWO RULES WERE FOUND BY THE GATE, NOT BY READING CODE, AND NEITHER IS VISIBLE ON AN
ORDINARY ARM.** Both are poke-env PRESENTATION rules, so both are fixed in Python (§2's contract):

* **`|error|[Unavailable choice]` is intercepted but NOT dropped.** `Player._handle_battle_message`
  routes it to `Gen3Battle.record_choice_rejected`, an out-of-band hook that appends a
  `CHOICE_REJECTED` event, so a live battle's log carries the rejection even though
  `parse_message` never saw the line. `ViewEventFolder.fold` skipped it with the other six
  intercepted keywords; the missing event lands in the H-B event window and moved **~200 obs
  cells**. It can only appear on a ply whose reject re-opens the request — a D10 arm — which is
  why it survived every sweep until those arms stopped falling back.
* **`Pokemon.faint` does not clear boosts; the sim does.** Showdown's `clearVolatile` (and the
  port) zero a mon's stages at the faint, while poke-env drops them only at `switch_out`. On an
  ordinary arm the replacement switch happens inside the same ply and both roads end at zero; at
  an INTERMEDIATE decision the board sits exactly between, so the protocol road still shows the
  dead mon's stages and the payload cannot. **MEASURED: eleven cases over seven fixture battles,
  every one a mon FAINTED and still ACTIVE**, and the stages were the ply's OWN (an arm that
  Dragon Danced twice before dying read +2/+3 where its sibling read +1/+2) — so the light board
  keeps a boost LEDGER folded from this ply's lines (`ViewEventFolder.fold_boosts`, mirroring
  `AbstractBattle`'s boost branches one for one including the ±6 clamp) and
  `view_adapter._restore_fainted_boosts` rewrites **FAINTED mons only**. A live mon's stages stay
  the payload's, which is the sim fact and is right.

**What it does NOT close.** A D10-served leaf carries **`fork=None`**: the rust child `node_id` it
is paired with sits at the END of the arm's turn, not at the decision the leaf describes, so a
deeper ply expanded from it would branch from a state the leaf is not. Ply d+1 therefore falls
back to the protocol road — exactly what a D10 arm did at every depth before the close, so the
close is scoped to the depth-1 row it is evidenced at. `view_fallback_intermediate` also survives,
for an arm the port sent no entry for (`impl="node"`, a `recorded_exact` arm).

**The gates.** `one_sided_view_parity_fuzz_test` compares a D10 arm instead of deferring it, and
its collected test asserts `d10 >= 1` so the fixture cannot go vacuous; the comparison is made
against a SECOND protocol road stopped AT the cut (the padded one runs past the replacement and
its read-models say nothing about this board). `event_fold_parity_fuzz_test::run_split` asserts
the cut materializes exactly ONE further row on the protocol road and that the events over it
agree field by field. Both were fault-injected: an uncut fold and a one-chunk over-run each
produce 219/204 differing obs indices, and the wrong BOARD (`view_pN` instead of `view_pN_at[0]`)
is caught too. 🚨 **The cut gate alone cannot see an over-run that stops before the next request**
— both roads are fed the same bytes there — which is why the tracker-fed obs comparison is run
beside it and not instead of it.

---

**Not deferred, and worth saying so:** the volatile NAMES here are NOT the `pre_state`
reconstruction that `search_and_replay_drivers.md` marks UNVERIFIED. That set is the port's own
typed fields and includes conditions the sim never announces — feeding it to the obs layer raised
`UnknownVolatileError: volatile 'choicelock' has no gen3 encoding slot` on the first real board.
This fold reads the protocol instead.

---

## 6. Cost

`view_materialize_benchmark.py`, per successor, **the two PRODUCTION roads**, three battles, same
load, busy box (load ~50/16 cpus — ratios are the claim, absolute ms are inflated ~3x):

| B | arms | protocol ms | view ms | speedup |
|---|---|---|---|---|
| 1 | 3 | 93.628 | 95.706 | **0.98x** |
| 33 | 99 | 5.708 | 4.227 | **1.35x** |

🚨 **THIS TABLE REPLACES A 47x / 6.4x ONE, AND THE OLD NUMBERS WERE NOT WRONG — THEY WERE
MEASURING A DIFFERENT THING.** The first version of this benchmark timed *payload → read-models →
encode* against the production materializer. That leaf had **no trackers** (five obs blocks at
zero) and **no shared prefix** (the view column paid no replay at all), so it was not a leaf any
search scores. Closing D5 gave the view road both — the prefix replay through `open_view_fork`,
which is `materialize_branches`' own first half, and a per-arm tracker fork — and the honest
speedup is what is in the table. Both columns now also ask for `map_actions_at`, which production
always asks for and which is ~24% of the view road's per-arm wall.

At B=1 the whole cost IS the shared prefix, which both roads pay once, so there is nothing to win
and the table says so. The win is per-ARM and therefore appears as the ply widens.

**Where the view road's per-arm wall goes** — measured on the real `SearchEngine` over banked
eval traces, 864 arms of 12 decisions, with a stack-accounted wall timer
(`src/main/search_dividend/search_decision_benchmark.py`; the record is
`designs/research_state/measurements/search_profile_2026-09-22/README.md`): `expand_many` and its
JSON 15.7%, the tracker fork 10.9%, the **D10 fallback into `materialize_branches` 21.1%
inclusive**, encode 9.5%, `open_root` 7.8%, the shared prefix replay 13.7% inclusive, the
read-models 6.7%, `_choice_map` **1.2%**.

🚨 **THIS TABLE RETRACTS A cProfile ONE, AND THE RETRACTED NUMBER WAS `_choice_map` AT 24%.** It
is 1.2% (0.054 ms/arm) on the production engine, and **1.1% when the identical benchmark is re-run
under cProfile** — so the old figure is not explained by profiler overhead either; the two
measurements disagree about the denominator and the one taken on the road production drives is the
one to act on. What cProfile DOES distort is in the same pair: encode reads 9.5% un-profiled and
15.6% profiled, the action mask 14.7% against 2.2%. **A cProfile share is not a wall share.**

### THE SPAN IS SHIPPING, NOT SIMULATING — `gen3_expand_many_side_elision_v1`

`expand_many` and its JSON was the largest single span left (20.2%). Split with a driver-side
timer against a Python-side one, over the same 864 banked arms, it is **30.2% the rust child
(0.164 ms/arm), 4.1% the pipe, and 64.5% Python** — `json.loads` of a reply of which the port
rendered BOTH sides' `view_pN` + `pN_chunks` while the search reads one. That half was **43.0% of
the reply bytes**, and `expand_many` now takes a `side` on the request and omits it: **30,326 →
17,330 B/arm**, with the surviving `view_pN` byte-identical (a rust gate expands the same arm both
ways in one process and diffs the bytes). `requests` is never elided — both sides are read.

**Measured** — interleaved against a baseline worktree, load-matched, **one road per process**
(the benchmark's two roads in one interpreter are NOT independent and the first A/B of that
campaign pointed the wrong way because of it): **1.10x on the view road at wide B** (2.315 ->
2.110 and 2.293 -> 2.084 ms/arm, two warm pairs agreeing to three digits), **1.11x on the protocol
road** — which parses the same reply, so a result that moved only one road would have meant
something else — and **NOT RESOLVED at B = 1**, where the decision is its prefix and there are 31
arms to save on. `expand_many` is **31.1% -> 26.6% of the decision wall** and no longer the
largest single span.

**Four candidates were killed by measurement, each with its number**: a compact fixed-order
payload saves 0.024 ms/arm of parse and gives it all back re-keying for the existing adapter (~1%
of the decision wall); one request per DECISION is worth 0.9% and the call pattern is already one
per PLY (864 arms in 39 batches); caching the ply-invariant view parts in the driver is ~1% for a
cache, a merge and a risk to the byte-parity gate; and holding Python's cyclic collector off for
the reply parse measures **2.3x on a captured 454 KB reply and NOTHING end to end** — it was
built, gated, and reverted. 🚨 **A micro-benchmark share is not a wall share**, the sibling of this
file's cProfile retraction and for the same reason: the toy process lacks the heap the collector
has to walk. Full record, both instrument findings, and the 5% stop rule:
[`../research_state/measurements/expand_many_2026-09-22/README.md`](../research_state/measurements/expand_many_2026-09-22/README.md).

The tracker fork was **10.5 ms** by `deepcopy` and is **0.5 ms** by a pinned-pickle thaw — the
same 9x `_PlayerSnapshot._freeze` documents, and until it landed the view road was SLOWER than the
road it replaces (0.84x at B=33). `agents/training/clone_pins.py` is now the ONE home for both
mechanisms.

### ONE FORK PER DECISION — `gen3_one_fork_per_decision_v1`

The fork is a function of the one-sided PREFIX, our action history and OUR packed team, and a
determinized world changes only the OPPONENT's team — so the K worlds of one decision were each
replaying one identical prefix. `SearchEngine._root_fork` builds it once per decision and **keys
the cache on the prefix BYTES**: `determinize.prefix_matches` truncates its comparison at the
`|turn|` marker, so the gate alone does not license reuse of what follows it, and a world whose
prefix differs simply misses and replays its own (`RealizedWidths.fork_cache_hit` /
`fork_cache_miss` count both). What makes the factory reusable is that `successor()` mutates
none of its three carried objects — it branches the event folder, thaws the frozen tracker and
concatenates the prior-event list — which is the same property that already let one fork serve the
arms of one world.

**Gated by `src/main/search_dividend/fork_sharing_parity_integration_test.py`**: the real engine
over one seeded decision, sharing ON against a store-but-never-serve control, comparing every
successor's observation BYTES in order plus the scores, the chosen action and the widths, with
non-vacuity asserted on both sides. The failure it stands for is silent — a factory that mutated
would give world 2 world 1's ply and still produce a well-formed obs.

The same sharing is applied to the PROTOCOL road's own fork, which every **D10 fallback arm**
pays: `materialize_branches` is split into `open_branch_fork` (prefix replay + frozen
`_PlayerSnapshot`) and `materialize_branches_from` (the arms), with `materialize_branches` kept as
the composition so its gate is untouched, and `SearchEngine._branch_fork` caches it per decision.
Its key carries the prefix bytes **plus** `map_actions_at` / `stop_after_decision` /
`encode_only_at`, because those are baked into the player at construction and a deeper ply asks
about a different decision index. That one is a LIVE poke-env player every arm mutates, so
reusing it asserts `_PlayerSnapshot.restore` is COMPLETE — a separate test, not a second copy of
the first.

### `action_choices` IS LAZY NOW — `gen3_lazy_action_choices_v1`

The last of the two sized-but-unbuilt wins. A successor's token map is the real action mapper over
every legal index, and its only readers — `TreeNode.expandable`, `plan_beam`'s arm-count estimate
and `search._expand_ply`'s loop — are all inside `_score_world`'s `while ply < md`, so a **depth-1
decision built one per arm and read none**. `agents.training.view_successor.LazyTokens` builds on
first read; it is a `Mapping`, so `bool()` falls through to `__len__` and materializes, and
"is this branchable?" still gets a true answer. Gated by
`src/main/search_dividend/lazy_action_choices_integration_test.py`: **0 of 18 built** at
`max_depth=1`, **17 of 17 built and equal to their producer** at `max_depth=2`. VIEW road only —
the protocol road's map is built where the poke-env battle still stands, and deferring it there
is a second replay, not a lazy read.

**Measured — an INTERLEAVED A/B**, a baseline worktree against this branch, the same 10 banked
eval-trace decisions, back to back, twice. The NOW runs carried the HIGHER load in three of the
four pairs, so the ratios are conservative:

| B | BASE view ms/decision | NOW view ms/decision | speedup |
|---|---|---|---|
| wide (684 arms / 10 decisions) | 326.8, 361.9 | **250.5, 269.3** | **1.33x** |
| 1 (31 arms / 10 decisions) | 96.0, 98.9 | **58.1, 59.0** | **1.66x** |

The protocol road moved too (1.12x wide, 1.81x at B = 1) because it shares the second fork, which
is why protocol-vs-view (1.57x -> 1.74x at wide B) understates the change. Spans, before -> after:
the shared prefix replay **14.3% -> 7.8%**, the D10 fallback **24.0% -> 17.2%**, `_choice_map`
**1.2% -> 0.7%**; `expand_many` and its JSON is now the largest single span at **20.2%**.

**`obs_build_benchmark.py` must read UNCHANGED** — no file on the `encode` path was touched, and
that is the check rather than a hope.

### THE D10 CLOSE COST NOTHING AND BOUGHT CONSISTENCY — `gen3_view_at_intermediate_v1`

An interleaved A/B on the same 10 banked decisions, twice, base worktree at `48e767ba`, on a box
whose load rose monotonically from 20.8 to 33.2 on 16 cpus (so every NOW run carried the heavier
box; the protocol road run back to back is the control):

| B | `view_fallback_intermediate` | view ÷ protocol, BASE → NOW |
|---|---|---|
| wide (684 arms / 10 decisions) | **117 → 0** | 0.576, 0.574 → **0.562, 0.553** |
| 1 (10 arms) | **2 → 0** | 0.980, 0.917 → **0.673, 0.746** |

🚨 **D10's 17.2% SPAN WAS NOT RECOVERABLE WALL, and the §6 table above is where that belief was
written down.** The span was the cost of serving those arms *at all*: a D10 arm answered on the
view road costs about what it cost through the fallback, so moving 117 of 684 arms onto a
~4.2 ms/arm path refunds nearly the whole 16.4% the view road's breakdown attributed to
`open_branch_fork` + `materialize_branches_from` (both rows vanish — the branch fork is never
opened now). **A phase share is not a saving**: a span that disappears because the work MOVED is
an accounting change. At B = 1 the win IS real (~25%) and structural — there the fallback forced
an entire second poke-env prefix replay to serve ONE arm, and at B = 1 the decision is its prefix.
What wide B buys is ROAD CONSISTENCY: `--materializer view` no longer runs 17% of its arms on the
other road, so a cell measured on it is no longer a blend. Record:
`designs/research_state/measurements/search_profile_2026-09-22/README.md` §4b.

## 7. Transport status, and THE FLIP

`view_p1` / `view_p2` and `view_p1_at` / `view_p2_at` are **rust-only**; `search_driver.js` emits
no such field and `SearchSession` defaults them to `{}` / `[]`. The cross-impl parity harness
(`src/rust_sim/harness/search_impl_parity.py`) carries ONE allowlist entry per pair and each
allowlists exactly that absence and ONLY that absence — both are value-aware and refuse to forgive
anything but `<absent>` vs present, so neither can outlive its own fix the way two entries in that
harness once did.

**`--materializer {protocol,view}` is BUILT and `view` is the DEFAULT** (`SearchConfig.materializer`).
`search._materialize` is the one seam: it opens the fork once per ply through `open_view_fork` and
then answers each arm from the port's payload, falling back to `materialize_branches` per arm —
never per cell, and never silently — for the three classes it cannot answer:

| fallback | counter | why |
|---|---|---|
| no `view_pN` | `view_fallback_no_payload` | `search_impl="node"`, or a parent that had itself fallen back (a deeper ply cannot fork from a board the view road never built) |
| an intermediate decision with no `view_pN_at` entry | `view_fallback_intermediate` | `impl="node"`, or a `recorded_exact` arm. The ordinary intermediate arm is now ANSWERED — §5b — and counted by `view_arms_intermediate` |
| the arm opened no decision | — | both roads agree there is no row |

`RealizedWidths.view_arms` counts what the view road actually answered, and the parity gate asserts
it is non-zero: a run in which every arm fell back would compare the protocol road with itself and
pass.

### What licenses the flip

| gate | what it holds |
|---|---|
| `one_sided_view_parity_fuzz_test.py` | the FULL 2501-dim successor obs, trackers included, `np.array_equal` against the production `materialize_branches` row. **422 tracker-fed comparisons over 24 fresh battles, ZERO `successor.*` divergences**; 176 over a separate 16-battle sweep, also zero |
| `event_fold_parity_fuzz_test.py` | `ViewEventFolder`'s events == `Gen3Battle`'s on the same bytes, field by field including `raw` |
| `materializer_parity_integration_test.py` | the real `SearchEngine` decides identically on both roads — per-action scores, chosen action, fallback reason and `arms_scored` — with a scorer that is a PURE FUNCTION of the obs, which is strictly sharper than a trained net |

The remaining sweep failures are all READ-MODEL classes that predate this work and are listed in
§4b; none of them is on the successor path, and a read-model residual makes the gate skip the
successor comparison rather than pass it.
