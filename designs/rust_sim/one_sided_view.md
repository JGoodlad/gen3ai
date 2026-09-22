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
| **Wall gate** | `src/rust_sim/tests/one_sided_view_test.rs` (9 tests) |
| **Unit gate** | `src/agents/battle/view_adapter_test.py` (24 tests) |
| **Differential gate** | `src/agents/battle/one_sided_view_parity_fuzz_test.py` (`sim`) |
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
| `volatiles` | the sim's condition set | a fold over `\|-start\|`/`\|-end\|`/`\|-activate\|`/`\|-singleturn\|`/`\|-singlemove\|`, cleared on switch-out and faint, with `ends_on_turn` effects DROPPED at the next `\|turn\|` and `is_turn_countable` ones counting up |
| `status_counter` | the sim's sleep/toxic counter | `+1` per MOVE-or-CANT while asleep, `+1` per turn while badly poisoned AND active; frozen by a faint; the toxic one (only) reset by a switch-out |
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
announced NAME with its turn/restart counts (never an id, never a counter), a move's SIGHTING count
keyed by target species (never a `current_pp`), and the raw `|request|` bytes (never a parsed
legality).

🚨 **A corollary worth stating: the port must NOT send its own `protect_counter`, its own
`status`-counter, or an opponent's true PP even though it has all three.** Each is either a
different quantity or privileged information, and each read plausible enough to ship.

---

## 3. The payload

```text
{"side":"p1","turn":N,"finished":bool,"won":bool|null,"lost":bool|null,
 "weather":{"weather":id|null,"is_permanent":bool,"turns_active":n},
 "ours":SIDE,"opp":SIDE,
 "request":<the |request| payload verbatim>|null}

SIDE = {"team_size":n,"active":<species id>|null,"side_conditions":{id:n},"mons":[MON,…]}

MON  = {"species","active","fainted","revealed",
        "hp_fraction","current_hp","max_hp",
        "status","status_counter","protect_counter",
        "types":[…],"moves":[MOVE,…],
        "item","consumed_item",
        "ability",                       # OURS only — the engine's, i.e. our own |request|'s
        "ability_events":[{"id","trace"},…],   # THEIRS — announcements in order; id "" = left field
        "boosts":{stat:stage},          # NONZERO stages only, as LiveView keeps them
        "volatiles":[{"name","turns","restarts"},…],
        "base_stats":{…},
        "ivs"|null,"evs"|null,"nature"|null,"spread_known",
        "stats":{…}|null}               # null == poke-env's all-None dict (an opponent's)

MOVE = own:     {"id","move_id","current_pp","max_pp"}
       watched: {"id","move_id","uses","max_pp","uses_vs":{species:n}}
```

`id` is the poke-env **moves-dict KEY** and `move_id` is the `Move.id` it maps to; they differ for
a typed Hidden Power (the wire re-keys it bare) and the two are sorted on by **different**
consumers — `LiveView.moves` by the key, `MovesEncoder` by `Move.id`.

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

---

## 4b. ONE OPEN FINDING — an unexplained own-side sleep-counter drift

**Measured**: a 14-battle sweep (292 comparisons, 250 branch points) reported
`ours.<species>.status_counter` **one HIGHER than poke-env**, 9 times, all on ONE battle and one
mon; the other 13 battles were clean, and 14 targeted diagnostic battles did not reproduce it.

It is NOT declared residual and the gate is NOT relaxed for it — a systematic break in that field
must still fail. What is known: poke-env increments the sleep counter in exactly two places
(`Pokemon.moved` and `Pokemon.cant_move`), both are folded, `cure_status` zeroes it, a faint
freezes it, and a switch-out resets only the TOXIC one. An extra increment against that set is
unexplained. The likely shapes to check first are a `|move|` line poke-env routes somewhere that
does not reach `moved()`, and a sleep applied on the same line it is counted.

The COLLECTED test is a fixed battle and is green, so this does not ride main red; the SWEEP is
where it will reappear. **Run the sweep before trusting a change in this area.**

**Re-measured 2026-09-19** over 24 fresh battles (577 comparisons, 505 branch points): still
present, 15 occurrences on one battle and one mon — and this time reading one LOWER
(`protocol=3 view=2`), where the original 14-battle sweep read one HIGHER. A field that drifts in
BOTH directions is not an off-by-one in one branch; the likeliest remaining shape is a `|move|` or
`|cant|` line the port's fold attributes to a different sleep episode than poke-env does.

### TWO MORE OPEN FINDINGS from the 2026-09-19 sweeps, same status (failing, not declared)

| what | measured | what is known |
|---|---|---|
| `opp.<species>.ability` reads `None` on the view road where poke-env has it | 14 occurrences on ONE battle of 14 (a Snorlax reading `immunity`) | poke-env sets an ability from a `[from] ability:` CLAUSE on lines that are not `\|-ability\|` — `_check_damage_message_for_ability`, `_check_heal_message_for_ability`, and the `\|move\|` handler's trailing-clause branch. The port's `ability_events` folds the `\|-ability\|` line only. The fix is in `view.rs`, and it is the same SHAPE as the item finding in §2: scanning the wrong set of lines |
| own-side `volatiles` missing `substitute` | 1 occurrence over 24 battles (`protocol={'substitute': 0} view={}`) | Not reproduced. One occurrence is not a class; it is recorded so the next sweep can tell "still one" from "now many" |

## 5. DEFERRALS — what is knowingly not carried

| id | what | why, and what it would take |
|---|---|---|
| **D1** | the obs SLOT order still comes from `battle.team` / `battle.opponent_team` | `state_encoder.encode` walks `base.get_team_list(battle, …)` to assign slots and only then looks the mon up in the `LiveView`. So an ordered identity list must exist whatever else happens; `ViewBattle` supplies it FROM the view |
| **D2** | four sub-encoders never took a `live_mon` | `items` (`mon.item`/`mon.consumed_item`), `abilities` (`mon.ability`), `types` (`mon.type_1`/`type_2`) and `moves` (`mon.moves`) read the raw `Pokemon` unconditionally, though `LivePokemon` carries every one of those fields and the rest of `pokemon.encode` is `if live_mon is not None:` throughout. `ViewMon` feeds them from the same read-model in the meantime. **Migrating those four onto `live_mon` is the right fix** and is a value-neutral refactor that owes the obs-build benchmark |
| **D3** | ✅ **CLOSED for a successor** — the reactive block's **Wish pair** | `reactive.encode` folds `battle.events` through `wish_belief.build_wish_pending`. The fix was not the port's `SideState::wish_pending` at all: the payload was never the problem, the missing LOG was. `ViewBattle` now carries the successor's whole-battle event log (the root's, plus the ply folded by `ViewEventFolder`) and the existing fold runs unchanged. Still OPEN for a BOARD-ONLY caller with no log, which is exactly what the gate's tracker-less comparison is — so the residual is still DECLARED there, and seen there |
| **D4** | ✅ **CLOSED for a successor** — the 3-dim **sleep-wake belief** | `build_sleep_sources(battle)` reads `battle.events` and `battle.turn` and nothing else, so the same log closes it. Same board-only caveat as D3 |
| **D5** | ✅ **CLOSED** — the per-decision **TRACKERS** — recency, pair history, the event window, the progress clock, the Hidden-Power block | `gen3_view_successor_v1` — see §7. The ply's events are folded from the arm's OWN one-sided protocol by `agents/battle/event_fold.py`, and the root's `EpisodeTracker` is carried forward and advanced through `record_context` / `advance_window` (the bodies of `record` / `update_progress_clock`, split out rather than copied) |
| **D6** | Pressure is applied with the ability known at READ time | `_pressure_on` evaluates `target.ability == "pressure"` AT USE TIME, when it may still be undisclosed. The residual is a sighting made before the reveal. (The un-fainted clause is deliberately NOT re-checked at read time: a mon cannot be targeted while fainted, and checking it late lost the whole correction on any board whose Pressure holder had since died — 30 divergences over 80 comparisons) |
| **D7** | our OWN pp is the wire's, not poke-env's counter | The payload sends the engine's `current_pp`, which is exactly what the `\|request\|` states and what poke-env asserts its own counter against (`check_move_consistency`). When poke-env has not yet identified a Pressure holder its counter drifts one BELOW the wire per sighting, and it is the drifted value the protocol road encodes. Folding our own PP from sightings instead was measured **worse** (the `\|move\|` line names `Hidden Power` while the set token is `hiddenpowerfire`, so the slots do not key against each other) |
| **D8** | Mimic / Transform move overlays on our own side | The own moveset is rendered from `set.moves`; an overlay would need the same resolver the request path uses |
| **D9** | `Mist` as a side condition | The port models spikes / reflect / lightscreen / safeguard only |
| **D10** | 🔴 an arm whose ply resolved an INTERMEDIATE decision | When an arm's ply KOs one of our mons, the replacement round is a SECOND request inside the same `expand_many` arm. The port answers it through its own follow-up policy and returns the board AFTER it, while `materialize_branches` stops at the FIRST request its action list cannot answer — so the two roads describe DIFFERENT STATES, one decision apart, and a leaf scored at the wrong one of them is not a rounding difference. `view_successor.intermediate_decisions` detects it off the arm's own `\|request\|` lines (verified against the protocol road's realized row count on 104 arms over 6 fresh battles: exact agreement, 98 arms at 0 and 5 at 1), and the production road FALLS BACK to protocol, counted in `RealizedWidths.view_fallback_intermediate`. **Measured rate: 35 of 505 branch points (6.9%) over 24 fresh battles** — and **145 of 864 arms (16.8%)** on 12 mid-game decisions banked from a real run's `eval_traces`, where the fallback into `materialize_branches` costs **21.1% of the view road's decision wall** (`designs/research_state/measurements/search_profile_2026-09-22/README.md`). The two rates are the same mechanism at different anchor distributions; the second is the one production pays. Closing it needs either a driver mode that stops at the replacement, or a payload that carries the intermediate board too |

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

## 7. Transport status, and THE FLIP

`view_p1` / `view_p2` are **rust-only**; `search_driver.js` emits no such field and
`SearchSession` defaults them to `{}`. The cross-impl parity harness
(`src/rust_sim/harness/search_impl_parity.py`) allowlists exactly this absence and ONLY this
absence — the entry is value-aware and refuses to forgive anything but `<absent>` vs present, so it
cannot outlive its own fix the way two entries in that harness once did.

**`--materializer {protocol,view}` is BUILT and `view` is the DEFAULT** (`SearchConfig.materializer`).
`search._materialize` is the one seam: it opens the fork once per ply through `open_view_fork` and
then answers each arm from the port's payload, falling back to `materialize_branches` per arm —
never per cell, and never silently — for the three classes it cannot answer:

| fallback | counter | why |
|---|---|---|
| no `view_pN` | `view_fallback_no_payload` | `search_impl="node"`, or a parent that had itself fallen back (a deeper ply cannot fork from a board the view road never built) |
| an intermediate decision | `view_fallback_intermediate` | D10 |
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
