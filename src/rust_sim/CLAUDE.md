# CLAUDE.md — `src/rust_sim/` (pokesim)

A from-scratch Rust reimplementation of the Pokémon Showdown battle simulator,
scoped first to **Gen 3 OU singles**, whose hard requirement is **bit-for-bit
identical** output to upstream Showdown given the same seed + teams + choices.

The engine is **live and bit-for-bit through full battles**: every layer in the
module map below is differentially validated against the real Showdown (PRNG →
dex → team → stats → state → events → damage → turns → full battles with
switching/secondaries/status/setup/recovery/protect/spikes/phazing/leech/
substitute/explosion/fixed-damage/PP+Struggle/taunt+disable/trapping), capped by
the 220-battle real-team e2e capstone (STRICT `filtered_diverged == 0`) and a
byte-identical protocol-emission Phase 1+2+3 (132 battles / 19348 lines), and the
bridge-facing `BattleStream::write_line` streaming surface — per-write byte-gated
against the real Node `BattleStream` (44 battles / 2377 writes, `writeline_test.rs`).

## Why "bit-for-bit" is the hard part (read this first)

Reproducing *Gen 3's mechanics* is the easy half. The constraint that costs the
effort is matching Showdown's **control flow**, for two reasons:

1. **RNG-consumption-order equivalence.** All battle randomness funnels through
   one `PRNG` (no `Math.random()` in the battle path). Bit-identity therefore
   requires consuming the RNG in the *exact same order and count* — including the
   Fisher-Yates speed-tie shuffle, which itself draws from the PRNG. Roll
   accuracy-before-crit where Showdown rolls crit-before-accuracy and every
   downstream draw desyncs. So you must mirror Showdown's `runEvent`/`singleEvent`
   dispatch order (handlers sorted by order → priority → speed), not just the
   observable formula.
2. **Byte-identical protocol output.** Our poke-env fork parses the `|...|`
   lines; they must match exactly (tokens, order, HP-fraction formatting). The
   one documented exception is `|t:|` wall-clock lines, which poke-env ignores.

The verification answer to both is **differential testing against the real
Showdown**, layer by layer. Level 1 (the PRNG) is built; see below.

## Module map

| Module | State | Responsibility |
|---|---|---|
| `prng/` | **done, validated** | Bit-for-bit port of `sim/prng.ts`. `Prng` (high-level `random*`/`sample`/`shuffle`) over two backends: `SodiumRng` (ChaCha20, the default) + `Gen5Rng` (64-bit LCG, legacy seeds). |
| `json.rs` | **done** | Tiny std-only recursive-descent JSON reader (`Json`) so the dex parses with zero deps. Load-time only, never in the battle path. |
| `dex/` | **done, validated** | Static data over this repo's `data/pokemon/*.json` (the same source `agents.gen3_data` uses, NOT a poke-env re-derivation). `Dex::for_gen(3)` → `species`/`moves`/`item`/`ability`/`nature`/`type_chart`/`learnset`; `Type`/`MoveCategory`/`BaseStats`; `to_id` normalization. `moves()` resolves move-ID ALIASES (`gen3_move_alias_resolution_v1`, `gen3_move_aliases.json`: `wisp`→`willowisp`, `sd`→`swordsdance`, …) — mirroring Showdown's `dex.moves.get()`, so a packed team's shorthand token runs the SAME move the sim runs (the e2e_86 cascade fix). `item()` returns `ItemData` (`dex/items.rs`) carrying the `gen3_item_mechanics_v1` structured fields (`type_boost`/`stat_mods`/`only_species`/`choice`/`is_berry`) + the `gen3_accuracy_pipeline_v1` `acc_mod`; `ability()` returns `AbilityData` (`dex/abilities.rs`) carrying the `dmgMod` DMG_MOD params (`{num,den,fold,types,pinch,when_statused,direct}`) + `acc_mod` + the `status_immune` STATUS_IMMUNE params (`gen3_status_immune_v1`, `StatusImmune {statuses, phase: SetStatus\|Immunity}` — read by `try_set_status`) + the batch-1 `crit_immune`/`weather_speed`/`weather_negate` (`gen3_ability_batch1_v1`) + the **batch-2 reactive fields** (`gen3_ability_batch2_v1`): `contact_proc` (`ContactProc {statuses, chance, sample}` — Static/Poison Point/Flame Body/Effect Spore), `contact_recoil` (Rough Skin), `blocks_sound` (Soundproof), `blocks_explosion` (Damp), `blocks_phaze_drag` (Suction Cups), `synchronize` — read by `turn.rs::apply_contact_proc`/`damp_holder`/`move_is_sound`/the phaze arm/`try_set_status` — + the **batch-3 fields** (`gen3_berry_trace_shedskin_v1`): `ItemData.berry_effect` (`BerryEffect` — the 22 data-driven CURE/HEAL/PINCH/PP rows, read by `turn.rs::apply_berry_residual`/`berry_on_update`/the setStatus lum tail) and `AbilityData.trace`/`shed_skin` (read by `event.rs::trace_on_start` / the shed-skin residual); `moves()` returns `MoveData` carrying the batch-2 `contact` + `is_sound` (`flags.contact`/`flags.sound`) move flags. The first two ability fields feed the data-driven damage folds; the shared `dex/accmod.rs::AccMod` (`{op: Multiply\|Chain, side, weather_sand, physical_types_only}`, Bright Powder/Lax Incense/Compound Eyes/Sand Veil/Hustle) feeds the to-hit fold (`turn.rs::effective_accuracy`); see "## Data-driven mechanics". |
| `team.rs` | **done, validated** | Bit-faithful `Teams.unpack`/`Teams.pack` for one team — the packed string the bridge feeds `>player`. `PokemonSet` + `unpack`/`pack`; ingests both Showdown and poke-env (lowercase-id) forms, re-packs Showdown-canonical. |
| `stats.rs` | **done, validated** | Gen-3 in-battle stat computation: `compute_stats(&PokemonSet, &Dex) → [u16;6]`. Exact floor placement + integer nature math + the Shedinja `maxHP` hook. Validated vs the sim's OWN computed stats. |
| `state.rs` | **construction done, validated** | In-battle state: `BattleState`/`SideState`/`MonState`/`Field` + `Status`/`Weather` enums. `MonState` carries the major `status`, the `confusion: Option<u8>` counter, the `flinch: bool` volatile (the two volatiles the secondary/onBeforeMove step needs), and the **`cached_speed: u32`** (`pokemon.speed`) the `eachEvent` tie-shuffles + residual handler-sort read — refreshed para/boost-aware at turn-start / residual-start / switch-in, STALE between (the e2e-capstone draw-count fix). `SideState` carries the **`spikes: u8`** layer count (0..=3) — the **first SIDE CONDITION** (`gen3_entry_hazard_spikes_v1`), a per-side persistent state (NOT a mon volatile, so it PERSISTS across switches), reusable by future hazards/phazing; 0 at construction. `MonState` also carries the **`leech_seed: Option<usize>`** volatile (`gen3_leech_seed_v1`) — `Some(seeder_side)` when this mon is LEECH-SEEDED (the seeder side that owns the drain; gen-3 singles always heals the seeder's CURRENT active, so only the side is stored). It is set by the Leech Seed MOVE, drained each end-of-turn RESIDUAL (the seeded mon loses `⌊maxhp/8⌋`, the seeder's active heals it), and CLEARED on switch-out (`execute_switch`) AND on faint (`process_faints` → `clearVolatile`); `None` at construction. `MonState` also carries the **`substitute: Option<u16>`** volatile (`gen3_substitute_v1`) — `Some(hp)` is the SUBSTITUTE decoy's remaining HP (created at `floor(maxhp/4)`); it ABSORBS incoming foe damage (the sub HP drops, breaking at 0 with NO carry to the mon), is created by the Substitute MOVE (paying `floor(maxhp/4)` HP), and CLEARED on switch-out + faint; `None` at construction. `MonState` also carries the **`move_pp: Vec<u16>`** + `move_maxpp` per-move PP counters + the **`choice_locked_move: Option<usize>`** (`gen3_pp_tracking_v1`) — `move_pp[k]` is slot `k`'s current PP, INIT to `MoveData::max_pp()` = `pp*8/5` (the ctor's default 3 PP-ups) / raw `pp` for a `noPPBoosts` move; decremented −1 per USE (2 into a Pressure holder), DRAW-FREE, ONLY when the mon MOVES; PERSISTS across switch-out (gen-3, no reset). `choice_locked_move` is `Some(k)` when a Choice-Band mon has locked to slot `k` (set on its first move, cleared on switch-out/faint) — so `must_struggle()` (all usable slots at 0 PP, respecting the lock) forces Struggle. `MonState::pp_array()` → the fixed `[i16;4]` the per-decision differential asserts. `MonState` also carries the **`taunt: Option<u8>`** + **`disable: Option<(usize, u8)>`** selection-restriction volatiles + the **`last_move: Option<usize>`** slot record (`gen3_taunt_disable_v1`) — `taunt` is the remaining-turn counter (a FIXED 2 at apply), `disable` is `(disabled_slot, remaining_turns)`, `last_move` is the slot this mon last USED (set after BeforeMove passes; a Struggle stores `None`); `move_usable(k, dex)` folds the Choice lock + Disable + Taunt (per-slot derived-Status minus the fixed-damage family) + PP, and `must_struggle(dex)` forces Struggle when nothing is usable; all three clear on switch-out + faint. `MonState` also carries the **`flash_fire: bool`** activation volatile (`gen3_flashfire_boost_v1`) — `true` once this mon's Flash Fire ability has ABSORBED a Fire move (armed at the `acc_hit`-gated Fire-absorb site in `run_move`, DRAW-FREE — a MISSED Fire move does NOT arm it; skips a `frz`-status holder; cleared on switch-out + faint), so thereafter its OWN Fire moves get **×1.5** (the volatile's `onModifyDamagePhase1 chainModify(1.5)` — a DAMAGE-PHASE fold folded in `damage.rs::modify_damage`, NOT a stat mod; category-agnostic, NOT crit-bypassed; accumulated with any screen into ONE Phase1 chain modifier); `false` at construction. `MonState` also carries the **`curse: Option<usize>`** volatile (`gen3_move_coverage_batch3_v1`) — `Some(source_side)` when this mon is cursed by a GHOST Curse (the source side for the `[of]` clause), chipped `floor(maxhp/4)/turn` at the order-10 subOrder-8 residual, cleared on switch-out + faint (like `leech_seed`); `None` at construction. `SideState` also carries the **`wish_pending: Option<(u8, String)>`** slot condition (`gen3_move_coverage_batch3_v1`) — `(duration, wisher_name)` for a pending Wish (the slot-keyed order-7 delayed heal `floor(maxhp/2)` at N+1; survives a switch — a side/slot condition, NOT a mon volatile) — and the **`baton_pass_pending: bool`** marker — set when a Baton Pass resolves so `execute_switch` runs `copyVolatileFrom` (snapshot boosts + the copyable sub/leech/confusion/curse fields → the entrant, `[from] Baton Pass`). `BattleState::start` constructs from `>start`+teams (unpack → compute_stats → leads), runs NO events; `start_with_switchins` adds the `event.rs` switch-in sequence (post-event boosts + weather). Validated vs the sim's construction-time AND post-switch-in state. |
| `event.rs` | **switch-in done, validated** | The generic event-dispatch core: `single_event_ability_start` (`singleEvent` — no sort/RNG/modify) + the reusable `speed_sort` (`order→priority→speed→subOrder→effectOrder` selection sort with the **Fisher-Yates speed-tie shuffle** drawing from the `Prng` — the RNG-consumption crux) over `EventHandler<H>`. Wired for the `>start` switch-in abilities (Intimidate foe-Atk −1 clamp, **gated by the foe's `onTryBoost` immunity — Clear Body / White Smoke / Hyper Cutter — so Intimidate-into-Metagross is a no-op**, an e2e-capstone fix, **AND by the foe's SUBSTITUTE — the gen3 mod skips a subbed foe** (a mid-battle Intimidate switch-in must not drop a subbed foe's Atk; seed-neutral; the `gen3_trapping_v1` e2e regen surfaced it, pin `intimidate_into_a_substitute_is_a_noop`); Sand Stream/Drizzle/Drought permanent weather); `run_start_switchins` fires both leads in raw-Speed order (slower-last so weather overwrites). Validated by `tests/switchin_test.rs` (5 differential scenarios) + `speed_sort` unit tests. Move/turn-loop/residual dispatch NOT built. |
| `damage.rs` | **done, validated** | Gen-3 single-hit damage calc: `calc_damage(&DamageContext, &Dex) → DamageResult{base, rolls:[u16;16]}`. Self-contained (EXPLICIT inputs — stats/types/boosts/status/move/field/`crit` — no `BattleState`). Bit-faithful port of the shared `getDamage` base formula + gen3's OWN two-phase `modifyDamage` (burn-first / randomize-2nd-to-last); integer 4096-chain `modify`, crit ×2 + ignore-boosts/screens, STAB, type chart, weather, Choice Band / type-item / Sea Incense (stat modifiers) + the **ability DMG_MOD folds** (Huge/Pure Power ×2 Atk, Guts ×1.5 Atk statused, Marvel Scale ×1.5 Def statused, the pinch family ×1.5 BP — all data-driven from `AbilityData.dmg_mod`), **Thick Fat (a gen3 `onSourceBasePower` ×0.5, NOT a stat mod)**, the Guts burn-halve suppression, Explosion def-halve, immunity → 0. Validated vs the omniscient oracle with the MAX roll forced ⇒ EXACT (not banded). |
| `turn.rs` | **multi-turn + SWITCHING + post-faint + win/loss + secondaries + status moves + SETUP moves + RECOVERY moves + PROTECT + SPIKES + PHAZING + LEECH SEED + SUBSTITUTE + FIXED-DAMAGE + PP-TRACKING + STRUGGLE + TAUNT + DISABLE + TRAPPING + CONTACT_PROC/BLOCK/SYNCHRONIZE abilities + move-coverage BATCHES 1-5 (through Focus Punch/Pursuit, Beat Up/Thunder/Water Spout, Hyper Beam/Solar Beam/Doom Desire/Future Sight, and Counter/Mirror Coat/Endeavor + Return/Frustration/Flail/Reversal/Low Kick + Sleep Talk) done, validated** | `BattleState::run_turn(p1_slot, p2_slot, &dex) → TurnResult` runs ONE FULL turn cycle (both sides damaging); `BattleState::run_battle(scripted, &dex) → Vec<TurnRecord>` loops it, stopping at the first faint. The cycle, in Showdown's EXACT draw order: the **action-order** speed-tie shuffle (`event::speed_sort` on the queue), the per-action **`eachEvent('BeforeTurn'/'Update'/'Weather')`** speed-tie shuffles (the draws the single-turn step deferred — incl. the one INSIDE gen3 `tryMoveHit` that fires only on a LANDED move), each move accuracy `random(100) < effAcc` (skip iff `never_miss`; `gen3_accuracy_pipeline_v1` — `effAcc = move.accuracy × the acc/eva stage table × the accMod item/ability handlers`, via `effective_accuracy`/`roll_accuracy`; the empty path is byte-identical to `randomChance(acc,100)`) → crit `randomChance(1,critMult[critRatio])` → damage `random(16)`, the **end-of-turn RESIDUALS** in gen-3 residualOrder (weather chip Sandstorm/Hail `max(1,⌊maxhp/16⌋)` to non-Rock/Ground/Steel; Leftovers `+⌊maxhp/16⌋`; the major-status DoT — burn `max(1,⌊maxhp/8⌋)`, poison `max(1,⌊maxhp/8⌋)`, Toxic `max(1,⌊maxhp/16⌋)·stage` with the per-mon stage ramp on `Status::Toxic`), all DRAW-FREE except the handler-sort + nested-Weather tie-shuffles, then the **Quick Claw** `randomChance(1,5)`. The **deferred-faint protocol** (`apply_damage` zeroes HP; `process_faints` sets `fainted` AFTER the in-`tryMoveHit` shuffle, mirroring `faintMessages`) is the faint-turn draw-COUNT crux: a KO turn fires the in-tryMoveHit shuffle but NOT the trailing Update / second move / residual / Quick Claw. Residual order/values are the gen4-mod overrides gen3 INHERITS (burn **/8**, Leftovers order 10 sub 4, status DoT order 10 sub 6, sand field-residual order 8 — NOT the base-data values). `run_residuals` runs **`faintMessages` PER HANDLER** (sets `fainted` between handlers + `if (ended) return`s, mirroring `fieldEvent('Residual')`'s `while`-loop): a holder fainted by an earlier handler skips its later ones, a GAME-ENDING residual KO aborts the rest, but a non-ending faint does NOT abort (the other active still ticks) — and since `order→priority→SPEED→subOrder` puts SPEED above subOrder, a fast burned mon's DoT self-KO can end the battle before a slower foe's Leftovers (`run_turn`'s post-residual faint gate reads the STATE `any_active_fainted`, not the now-always-false newly-fainted return). The `eachEvent` tie-shuffles + the residual handler-sort read the **CACHED `pokemon.speed`** (`MonState::cached_speed`), refreshed para/boost-aware at turn-start, residual-start, and switch-in (`update_speed()` / `execute_switch`) and STALE between — so a mon paralyzed mid-turn ties on its FULL speed until the residual, while one that switches in paralyzed ties on its PARA speed (the e2e-capstone bit-for-bit fix). **`BattleState::run_full_battle(&[ScriptDecision], &dex) → BattleOutcome` plays a FULL battle to WIN/LOSS** with `Choice::Move`/`Choice::Switch`: voluntary switches (order 103 < move 200, so they resolve FIRST; the two-switch action-order tie-shuffle), the gen-3 draw-FREE switch-in (the entrant's ability `Start` via `single_event_ability_start`; the gen-4 `runSwitch` override has NO `speedSort(allActive)` — do NOT add a SwitchIn tie-shuffle), the `switchIn` POSITION SWAP (the entrant → active index, outgoing → its old bench slot; mon actions are keyed by a stable `MonState::uid`, not array slot), POST-FAINT replacement (single + DOUBLE; the double's `insertChoice` order-101 splice draw + the no-op fainted-mon move that `return false`s before its tail → NO trailing Update), the pause/resume of the saved turn tail (`makeRequest('switch')` returns the queue intact; the `peek===instaswitch` early-return SKIPs the first instaswitch's tail), the **reject-and-re-request boundary gate** (`move_decision_is_legal`, `gen3_forced_replacement_resume_v1`: a top-of-turn `move` decision whose `Move(K)` slot exceeds the CURRENT active mon's movepool is SKIPPED draw-free — run no turn, emit nothing, record nothing, re-pull the next decision — mirroring the sim's `side.choose` REJECTING an out-of-range slot after a replacement swapped in a mon with fewer moves; this is the "phantom zero-draw `move` decision" the omniscient capture records from a stale per-turn plan, and it un-deferred the last 2 protocol scenarios — pinned by `forced_replacement_resume_runs_the_post_replacement_move_decision`, ground truth `harness/probe_forced_replacement_resume_regression_rng.js`; VERIFIED zero-draw so the e2e/seed suites are byte-identical), and win/loss (`pokemon_left == 0` loses, foe wins; both 0 → a gen-3 TIE `win(None)`; the deciding faint draws NO Quick Claw). **EXPLOSION / SELF-DESTRUCT self-KO** (`useMoveInner` battle-actions.ts:501-503, `gen != 4 && selfdestruct == 'always'`: `this.battle.faint(pokemon)` BEFORE `trySpreadMoveHit`) is modeled bit-for-bit: the self-KO zeroes the user's HP + is DRAW-FREE + UNCONDITIONAL + PRECEDES the hit — sitting AFTER `on_before_move` (a fully-para/asleep/flinched user never reaches `useMoveInner`) but BEFORE the accuracy/protect-block/immunity/miss checks, so the USER FAINTS THROUGH a Protect (blocked, no foe damage), a Ghost (Normal-immune), a Substitute (the damage breaks the sub, no carry), or a miss (gen-3 Explosion accuracy is 100 → no self-accuracy miss, but a hypothetical miss would still faint the user). Explosion draws the SAME count as any damaging move (acc `randomChance(100,100)` + crit + damage; NO secondary); the resulting faint changes `pokemon_left` / cancels the foe's queued move (gen-3 singles) / draws NO trailing Quick Claw on a deciding faint. A mutual Explosion (both last mons) is a true double-faint gen-3 TIE. VALIDATED by `tests/explosion_test.rs` (the differential `harness/gen_explosion_golden.js`: 7 scenarios × 80 seeds, 3688 decision rows, 7376 FAINTED assertions, 544 self-KO rows, 294 sub-break boundaries, 341 wins + 59 ties — plain / into-a-sub / into-a-Protect / into-a-Ghost / mutual-TIE / double-replacement / into-a-real-battle) + 4 DETERMINISTIC `tests/regression_test.rs` pins E1-E4 (user-faints-through-Protect / -immunity / -a-sub-break-with-no-carry / the mutual double-faint TIE; ground-truth seeds from `harness/probe_explosion_regression_rng.js`, the probe `harness/probe_explosion_rng.js` nailed the draw model). The engine also carries a `pending_explosion_self_ko` flag → `DecisionRecord.explosion_self_ko` (a coverage/diagnostic signal only, no effect on any draw/state; the e2e capstone reads it to count explosion decisions). **e2e INCLUDED** (`EXPLOSION_E2E_EXCLUDED = false`, bit-for-bit — **544 explosion-move / self-KO decisions across the 220-battle strict gate**, `filtered_diverged == 0`, `explosion_decisions >= 50` coverage floor): admitting Explosion surfaced TWO STATEFUL desyncs in DIFFERENT layers (NOT the self-KO), both now FIXED. (1) A **double-faint → double-replacement → cascade `runSwitch` cancellation** (e2e_9): when a mutual double faint replaces BOTH sides and the FIRST runSwitch to run FAINTS its own entrant on its side's Spikes (the cascade), gen-3-singles `faintMessages` → `cancelAction(getAllActive())` (battle.ts:2606-2616 — "in gen 3, fainting skips all moves AND SWITCHES") REMOVES the OTHER side's still-pending `runSwitch` (a runSwitch's `action.pokemon` is the entrant, a getAllActive member) — so the foe entrant is NEVER chipped (stays FULL HP). The port's `cancel_active_actions` cancelled `Move`/`Switch` but NOT a pending `RunSwitch` → the stale foe runSwitch survived the cascade + re-chipped its already-settled entrant (e2e_9 dec43: 403 → 353). FIX: `cancel_active_actions` now also drops a `RunSwitch { side }` when `sides[side].active` is not fainted — DRAW-FREE (a queue splice, SEED untouched). VERIFIED vs the sim (`harness/probe_cascade_hazard_order.js` + `probe_double_replacement_spikes_rng.js`): with the FAINTING side's runSwitch FIRST the foe is UNCHIPPED; with the SURVIVING side's first it is chipped ONCE (its runSwitch already ran → nothing to cancel). (2) The **confusion self-hit dropped Choice Band** (e2e_194): gen-4 confusion (gen-3-inherited, conditions.ts:74-83) runs the FULL `getDamage(self,self,40)`, so Choice Band ×1.5 (physical) folds in; `apply_confusion_self_hit` passed no atk stat mods → it used the stored Atk not the CB-boosted Atk → the self-hit under-dealt. FIX: it now resolves `resolve_atk_stat_mods(item, None, Physical)` (typeless '???' → CB only, no type-item / Sea Incense) — DRAW-FREE (the self-hit still draws `random(1,2)` + `random(16)`). Both pinned by revert-verified `regression_test.rs` pins `double_replacement_cascade_does_not_rechip_the_other_sides_entrant` + `confusion_self_hit_applies_choice_band` (ground truth from `harness/probe_double_replacement_cascade_regression_rng.js` + `probe_confusion_choiceband_regression_rng.js`). See EDGE_CASES.md. **SECONDARY effects + onBeforeMove STATUS draws** (this step) bracket each move: (a) `on_before_move` fires the NEW LEADING draw BEFORE accuracy, priority-DESC with break-on-first-abort — sleep (DRAW-FREE counter decrement / wake), freeze `randomChance(1,5)` thaw, flinch (DRAW-FREE volatile), confusion (decrement → `randomChance(1,2)` → a typeless-40-BP self-hit one `random(16)`, NO crit, via `calc_damage`), paralysis `randomChance(1,4)` full-para; an abort draws NOTHING further; (b) `apply_secondaries` fires the NEW TRAILING `random(100)` per surviving secondary AFTER a landed hit (Body Slam par30 / Ice Beam frz10 / Thunderbolt par10 / Rock Slide flinch30 / Sludge Bomb psn30), applied if `roll<chance` via `try_set_status` (the onTrySetStatus gates: already-statused → no-op, gen-3 type immunity — **frz→Ice, brn→Fire, psn/tox→Poison&Steel; gen-3 has NO Electric→para immunity; **SUN → NO freeze** — `gen3_sun_freeze_immunity_v1`, the base `sunnyday` weather's `onImmunity('frz')` blocks a freeze while the field is Sun [Drought/Sunny Day], at `runStatusImmunity` before the SetStatus shuffle, DRAW-FREE; an already-frozen mon PERSISTS under sun — pinned FZ1**, verified vs sim) / the flinch volatile / **the CONFUSION arm** (`add_confusion`: a landed confusion secondary draws ONE EXTRA `random(2,6)` duration inside `addVolatile`'s onStart UNLESS gated — ALREADY-CONFUSED or OWN TEMPO draw the secondary `random(100)` but NOT the `random(2,6)`, the draw-COUNT gate) / **the STRUCTURED stat-boost arm** (`apply_secondary_boost`: the foe stat-DROP / self stat-RAISE the flat `secondaryEffects` `{col:percent}` loses — Crunch −1 SpD, Psychic/Shadow Ball −1 SpD, Meteor Mash +1 Atk SELF, Ancient Power +1 ALL — read from the additive **`secondaryBoosts`** dex field [`{chance,target:foe|self,boosts}`, only-when-present like `critRatio`], DRAW-FREE apply [`boost()` consumes no PRNG] clamped to ±6, with the Clear Body / White Smoke / Hyper Cutter / Keen Eye `onTryBoost` immunity gates); a **fail-loud guard** PANICS on any move with >1 secondary col except **Tri Attack** (`triattack`), which is SPECIAL-CASED to its true draw model — ONE `random(100)` (the 20% gate) then ON LAND ONE `random(3)` `sample(['brn','par','frz'])` → `try_set_status` (NOT three `random(100)`s the 3-col flatten would mis-draw). **Serene Grace ×2** the threshold (NOT the draw) and **Shield Dust** on the DEFENDER FILTERS foe-targeting secondaries out (a draw-COUNT effect — zero `random(100)`) — **but NOT behind a SUBSTITUTE** (`gen3_shielddust_sub_v1`: the filter is a TARGET-gathered ModifySecondaries handler and a sub-absorbed hit's target list is `null`, so the filter never gathers and the secondary / Tri-Attack-gate / King's-Rock `random(100)` STILL DRAWS, held AND breaking sub, while the effect stays sub-suppressed — the A/B fuzzer's #1 sub×secondary SEED cluster, probe `probe_sub_break_secondary_rng.js`, pin `shield_dust_behind_a_substitute_still_draws_the_secondary`); a DAMAGE-immune target short-circuits BEFORE the secondary (no draw), a status-immune-but-damaged target STILL draws it. The fire-move thaw cures the DEFENDER's freeze (draw-free, post-secondary). **gen-3 paralysis speed is ×0.25** (`modify(spe,1,4)` = `floor((spe·1024+2047)/4096)`, gen4-inherited `chainModify(0.25)` — NOT ×0.5; verified vs the sim's `getStat`). FLINCH (`duration:1`) clears at the top of each turn; switch-out clears confusion+flinch. Validated by `tests/turn_test.rs` (the single-turn 780-row gate), `tests/battle_test.rs` (a per-seed CROSS-TURN STATE+SEED differential over 12 scenarios × 40 seeds × several turns, no switching), `tests/fullbattle_test.rs` (a per-seed PER-DECISION STATE+SEED differential to GAME-END: ~2053 per-decision EXACT seed assertions over 8 scenarios × 50 seeds — both-switch distinct/tie, switch-vs-move, post-faint single + double replace, KO-to-win, last-mon double-KO TIE), AND `tests/secondary_test.rs` (a per-seed PER-DECISION STATE+**STATUS**+**BOOSTS**+**CONFUSION**+SEED+winner differential to GAME-END with REAL secondary moves: **~4328 per-decision EXACT seed assertions + ~7457 status-variant + ~7457 boost-stage + ~7457 confusion-counter assertions over 12 scenarios × 80 seeds** — full-para, freeze-thaw, flinch, psn-immune-but-damaged, Ground-damage-immune-no-secondary, **Crunch/Psychic/Shadow Ball −1 SpD foe, Meteor Mash +1 Atk self, Intimidate −1 Atk on entry, Water Pulse confusion + the random(2,6) duration**) + deterministic unit gates for the confusion-secondary draw (`random(100)`+`random(2,6)`), its already-confused / Own-Tempo no-`random(2,6)` gates, the Tri-Attack `random(100)`+`sample(3)` sequence, the >1-col fail-loud panic, the stat-drop/self-boost apply, and the Clear-Body block. **STANDALONE STATUS MOVES now BUILT** (`run_status_move`): the foe-targeting major-status moves — Thunder Wave/Stun Spore/Glare [par], Toxic [tox], Poison Powder/Poison Gas [psn], Will-O-Wisp [brn], Spore/Sleep Powder/Hypnosis/Sing/Lovely Kiss/Grass Whistle [slp] — drawing ONLY accuracy (`randomChance(acc,100)`, skip iff never_miss) then `try_set_status`, with the gen-3 MOVE-TYPE immunity (Thunder Wave→Ground, Glare→Ghost — the two `ignoreImmunity:false` moves; all others ignore type immunity, accuracy still drawn → `-immune`), the sleep `random(2,6)` onStart duration (Early Bird double-decrements the wake counter), Toxic at stage 0 (the residual ramps it), the **gen3ou SLEEP CLAUSE MOD** (a 2nd foe-sleep fails at the SetStatus event, draw-free) + the DATA-DRIVEN status-immunity ABILITIES (`gen3_status_immune_v1` — Limber par / Insomnia+Vital Spirit slp / Immunity psn,tox / Water Veil brn via `onSetStatus`; Magma Armor frz via `onImmunity` BEFORE the event — read from `AbilityData.status_immune`), and the **gen3ou-only `runEvent('SetStatus')` handler-sort shuffle** (the 2 `Standard` clauses tie → ONE `random(0,2)` on EVERY status APPLICATION reaching the SetStatus event — incl. a clause/ability block; gen3customgame has 0 handlers → no shuffle, gated by `BattleState::sleep_clause` from the format). NO crit/damage/secondary; `landed` always FALSE (a status `moveHit` returns `undefined` → the in-tryMoveHit Update is skipped). A fail-loud guard PANICS on any UNMODELED status move (and on a genuinely UNMODELED `onSetStatus` ability under a clause format via `ability_unmodeled_on_set_status` — the STATUS_IMMUNE members are MODELED, `gen3_status_immune_v1`, and sort into their own speed group so the 2-clause tie stays size-2, unchanged draw count). **The ALREADY-STATUSED `-fail` emission** (`gen3_forced_replacement_resume_v1`, `foe_status_move_fail` + `StatusMoveFail`): a foe-status MOVE into an already-statused foe emits a `|-fail|` line — draw-free past the accuracy roll (the fail is emitted at `setStatus` BEFORE `runEvent('SetStatus')`, so no clause shuffle). gen-3 `trySetStatus` re-passes the foe's OWN status to `setStatus` (`setStatus(this.status || status)`), so `status.id === this.status` (pokemon.ts:1699): **SAME status** as the move inflicts (Thunder Wave→par into par) → `|-fail|<target>|<status>` (fail on the TARGET, status token); **DIFFERENT status** (Thunder Wave→par into brn) → the move announce's `[still]` empty-target form + `|-fail|<user>` (fail on the USER, no token). Keyed on the move having a top-level `move.status` field so a SECONDARY status (Body Slam's par into a statused foe) correctly emits NOTHING (verified `harness/probe_status_move_fail_lines.js` + `probe_status_fail_accuracy.js` — the only per-move draw is the accuracy roll). **SELF-TARGETING SETUP / STAT-BOOST MOVES now BUILT** (`run_status_move` self-boost branch BEFORE the fail-loud): the PURE self-boost moves (category Status, bp 0, target self) — Calm Mind (+1 SpA/+1 SpD), Dragon Dance (+1 Atk/+1 Spe), Swords Dance (+2 Atk), Agility (+2 Spe), Bulk Up (+1 Atk/+1 Def), Amnesia (+2 SpD), Barrier/Acid Armor/Iron Defense (+2 Def), Cosmic Power (+1 Def/+1 SpD), Tail Glow (+2 SpA), Meditate/Sharpen/Howl (+1 Atk), Harden/Withdraw (+1 Def), Growth (+1 SpA) — resolved from the data-driven `selfBoosts` dex field (`self_boost_spec`, the GIGO-proof source). Their draw model: (1) ACCURACY — every modeled setup move is NEVER-MISS (`accuracy:true`) so NO accuracy draw (a non-never-miss setup move would draw it; none exist); (2) APPLY `boost()` on the USER, ±6 clamp, **DRAW-FREE** (`boost()` consumes no PRNG — like `apply_secondary_boost`); our OWN Clear Body / White Smoke etc. do NOT block our own self-boost (the `onTryBoost` immunity is for FOE drops); a boost into the +6 cap is a no-op-but-success that STILL draws nothing; (3) `landed` ALWAYS FALSE (the in-tryMoveHit Update is skipped). **The +SPEED cached-speed crux:** a Dragon Dance / Agility raises `boosts[4]` IMMEDIATELY but does NOT touch `cached_speed` — Showdown re-establishes `pokemon.speed` only at the next re-cache site (turn-start `commitChoices`, residual `updateSpeed`, switch-in), so THIS turn's eachEvent tie-shuffles read the PRE-boost cached speed and the NEXT turn's action order picks up the boosted speed at turn-start (the e2e + setup-golden first-mover FLIP proof). The EXCLUDED setup moves stay fail-loud: Defense Curl / Minimize (a `volatileStatus`), Double Team / Minimize (+evasion — the engine's accuracy roll ignores evasion → a silent desync), Belly Drum (HP-cost `onHit`, no declarative `boosts`), Curse (type-conditional). **WATER/VOLT ABSORB heal is now accuracy-gated** (the e2e-capstone fix this step surfaced): the absorb heal is an `onTryHit` that fires only when the move HITS, so a MISSED Water/Electric move (e.g. Hydro Pump's 80% accuracy fails) does NOT heal the Absorb holder — `run_move` only calls `apply_absorb_heal` when `acc_hit` (the draw count is accuracy-only either way, so a wrongly-applied heal desyncs the post-hit HP STATE but not the seed). **SELF-HEAL / RECOVERY MOVES now BUILT** (`run_status_move`'s recovery branch + `run_rest`, BEFORE the fail-loud): the self-targeting HP-recovery moves (category Status, bp 0, target self, isHeal). The flat-half recovers **Recover / Soft-Boiled / Slack Off / Milk Drink** heal `floor(maxhp/2)` (the `move.heal:[1,2]` path); the WEATHER-conditional **Moonlight / Synthesis / Morning Sun** heal — gen4-inherited PLAIN integer (NOT the 4096-`modify`) — `floor(maxhp/2)` (none) / `floor(maxhp*2/3)` (SUN) / `floor(maxhp/4)` (SAND/RAIN/HAIL); **Rest** FULL-heals + self-sleeps + cures the prior status. All NEVER-MISS (no accuracy draw) and the heal itself is DRAW-FREE (`apply_heal` mirrors `apply_leftovers`; a heal at FULL HP / heal-0 FAILS via the `heal`-returns-false path, draw-free); `landed` is FALSE (no in-tryMoveHit Update). **REST's draw subtlety (the verified-against-the-sim crux):** `setStatus('slp')` runs the gen-3 `slp.onStart`, which ALWAYS draws `random(2,6)` for the duration — so Rest **DOES consume one `random(2,6)`** (`run_rest` draws-then-DISCARDS it) — and Rest's `onHit` then OVERWRITES the time to a FIXED `Sleep(3)` (the stored counter is 3, NOT the rolled 1-4). The user wakes via the EXISTING `on_before_move` sleep counter (3→cant→cant→wake+move). A self-Rest sleep is EXEMPT from the Sleep Clause CAP, but its `setStatus('slp')` STILL draws the gen3ou **SetStatus handler-sort shuffle** in a clause format (`run_rest` does the shuffle THEN the `random(2,6)`, gated by `sleep_clause`; gen3customgame draws neither shuffle — only the `random(2,6)`). **`splash`** is modeled as a true DRAW-FREE no-op (the recovery golden's "do-nothing" filler). DEFERRED (fail-loud): **Wish** (a DELAYED slot-keyed end-of-next-turn heal — a pending-heal model), **Heal Bell / Aromatherapy / Refresh** (team/self STATUS cure, not HP), **Pain Split / Leech Seed / drain / Ingrain / Aqua Ring**, plus phaze/hazard/Substitute/field status moves, entry hazards (Spikes), Pursuit, Baton Pass, non-Leftovers items, the **top-level `move.self.boosts` `selfDrops` draw** (Overheat/Superpower/Psycho Boost), Thick Club, protocol emission. Validated by `tests/recovery_move_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS)+SEED+winner differential to GAME-END over 8 scenarios × 80 seeds: ~4468 decision rows, ~3500 seed + HP assertions, the heal floors + Rest sleep/wake + the weather fractions) + deterministic unit gates (`recover_heals_half_maxhp_draw_free`, `recover_at_full_hp_fails_draw_free`, `moonlight_weather_conditional_heal_amounts`, `rest_full_heal_fixed_sleep_three_and_cure_draws_one_random_2_6_customgame`, `rest_at_full_hp_fails_without_sleeping`, `rest_in_gen3ou_draws_the_setstatus_shuffle_then_random_2_6`). **PROTECT / DETECT now BUILT** (`run_protect` + the foe-move BLOCK in `run_move`; new `MonState` fields `protected`/`protect_counter`/`stall_duration`): NEVER-MISS + **priority 3** (resolves BEFORE the foe's attack so the volatile is up). The gen-3 consecutive-use STALL draw — the FIRST protect (counter 0, no `stall` volatile) SHORT-CIRCUITS with **NO draw** (always succeeds); a CONSECUTIVE one draws `onPrepareHit`'s `runEvent('StallMove')` = ONE `randomChance(1, counter)` at the floored denominator **2 → 4 → 8 → 8** (the gen4-inherited `stall` `counterMax: 8`); a SUCCESS (re)adds the volatile — `onStart` to 2, else `onRestart *= 2` capped at 8, refreshing `duration` to 2 — a FAILED roll draws nothing more and (the gen3 resolved **gen5-base** `onStallMove`, unlike gen8+) does **NOT delete** the volatile: the counter + duration PERSIST (so consecutive fails re-roll at the SAME denominator AND a `stall` residual handler still fires). The **`willAct()` gate** (`onPrepareHit`'s `!!this.queue.willAct() && …`): a Protect that resolves with NO foe action still pending — the foe **SWITCHED** (order 103 < the protect's move order 200, so the switch already ran) — FAILS draw-free, no volatile (the `&&`-short-circuit also skips the stall roll). The move-BLOCK: in gen-3 `tryMoveHit` accuracy (line 364) is drawn FIRST, then `if (accPass) runEvent('TryHit')` (line 369) where protect blocks — so a BLOCKED foe move TARGETING the protected mon DRAWS its accuracy roll (skip iff never_miss) then is blocked, drawing NO crit/damage/secondary/status (and BEFORE the immunity report — EQ into a Flying/Levitate protector shows `-activate Protect`, not `-immune`); a self-target move is never blocked. The stall counter resets after one non-protect/switch turn — the volatile's `duration: 2` expiry, modeled at the RESIDUAL via `stall_duration` countdown; switch-out clears all three. **The `protect`/`stall`/`flinch` DURATION-only volatiles register RESIDUAL duration handlers** (`findPokemonEventHandlers(..., 'duration')` gathers every duration-bearing volatile to count it down) at order NO_ORDER/subOrder 2 — they participate in the residual speed-sort (a protecting mon adds 2 tied handlers; a failed-protect-into-a-flinch adds stall+flinch tied → a tie-group shuffle COUNT the model must match; flinch was draw-free before protect but ties with stall now — confusion has NO `duration` so no handler). DEFERRED (fail-loud in `run_protect`): **Endure** (`volatileStatus:'endure'`, a survive-at-1-HP `onDamage`) + the gen4+ Quick/Wide Guard / King's Shield / Spiky Shield (none in gen3). Validated by `tests/protect_move_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+**STALL-COUNTER**)+SEED+winner differential to GAME-END over 6 scenarios × 80 seeds in gen3customgame: 480 runs, 2772 seed + 5544 HP + 4984 stall-counter assertions, 1102 blocks, 272 escalated counter≥4 — single protect block, CONSECUTIVE protects [the denominator BOTH ways], the counter reset, protect vs a STATUS move, Detect, protect-into-a-real-battle) + deterministic unit gates (`first_protect_draws_nothing_and_sets_counter_two`, `consecutive_protect_draws_one_stall_roll_and_escalates` [incl. the no-delete-on-fail persist], `protect_blocks_foe_move_after_its_accuracy_draw`, `protect_does_not_block_the_protectors_own_move`, `endure_panics_fail_loud`). The PRNG-draw crux was nailed by `harness/probe_protect_rng.js` (the instrumented probe). **SPIKES (the gen-3 ENTRY HAZARD — the first SIDE CONDITION) now BUILT** (`run_status_move`'s spikes arm + `apply_entry_hazards` in `run_switch`; new `SideState::spikes: u8`): the Spikes MOVE (`sideCondition:'spikes'`, `target:'foeSide'`) is NEVER-MISS + DRAW-FREE — it increments the CASTER's FOE side's `spikes` layer by 1, CAPPED at 3 (a 4th Spikes FAILS, `-fail`, draw-free); `landed` FALSE (no in-tryMoveHit Update). The SWITCH-IN DAMAGE is the gen-3 `runSwitch`'s `runEvent('EntryHazard')` (gen4-inherited; ORDER: EntryHazard → SwitchIn → `if (!pokemon.hp) return` → ability `Start` — so a Spikes-KO SKIPS the entrant's ability Start): GROUNDED-ONLY (a Flying-type / Levitate entrant takes ZERO), amount `max(floor([_,3,4,6][layers]·maxhp/24),1)` = `max(floor(maxhp/8),1)` / `max(floor(maxhp/6),1)` / `max(floor(maxhp/4),1)` for 1/2/3 layers (the resolved `spikes.onEntryHazard` → `damage()` → `clampIntRange(_,1)`), DRAW-FREE (the deterministic `this.damage`; the nested `runEvent('Damage')` has no drawing handler for the modeled abilities — VERIFIED vs the omniscient sim `harness/probe_spikes_rng.js`). A Spikes hit that zeroes HP faints the entrant (the runAction tail's `process_faints` sets fainted + `check_fainted` flags → forces ANOTHER forced replacement, which ALSO takes Spikes) — wired through the existing faint/replacement machinery, no Quick Claw / extra draw. The `spikes` layer is a SIDE condition → it PERSISTS across switches (cleared only at construction). DEFERRED (excluded / fail-loud): **Toxic Spikes** + **Stealth Rock** (NOT gen3), **Rapid Spin** (the hazard-CLEAR move — a damaging move the fuzz won't pick as a modeled status move; hazards persist). Spikes is the only gen-3 entry hazard. Validated by `tests/spikes_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+**SPIKES-LAYERS**)+SEED+winner differential to GAME-END over 5 scenarios × 80 seeds in gen3customgame: 400 runs, ~3475 decision rows, ~3475 seed + ~6950 HP + ~6950 spikes-layer assertions, 1440 switch-in-chip + 160 spikes-KO-on-entry rows, 400 wins — lay 1/stack 2/3 + the increasing damage, a Flying/Levitate IMMUNE entry, a Spikes-at-max FAIL, a spikes-KO-on-entry → forced replacement, and spikes-into-a-real-battle) + deterministic unit gates (`spikes_move_increments_foe_side_draw_free_and_caps_at_three`, `spikes_switch_in_damage_grounded_per_layer_draw_free`, `spikes_switch_in_flying_and_levitate_take_zero`, `spikes_ko_on_switch_in_zeroes_hp_draw_free`). The draw model was nailed by `harness/probe_spikes_rng.js`. **PHAZING — ROAR + WHIRLWIND now BUILT** (`run_status_move`'s phaze arm + `drag_in` at the `turn_loop` runAction tail; new `MoveResolution::force_switch_foe`): the gen-3 `forceSwitch` moves force the FOE to switch to a RANDOM eligible team member. The draw model, VERIFIED vs the omniscient sim PRNG probe `harness/probe_phaze_rng.js`: (1) PRIORITY −6 → the phazer moves LAST; (2) ACCURACY — gen-3 Roar/Whirlwind resolve to **`accuracy: 100`** (NOT `true`! — the SURPRISE this layer surfaced: the base Showdown data lists `accuracy:true`, but our `gen3_moves.json` already carries the resolved gen-3 value 100), so a phaze is NOT never-miss — it DRAWS `randomChance(100,100)` (always passes but CONSUMES a draw), drawn in the phaze arm BEFORE routing to the drag; (3) THE RANDOM TARGET DRAW — `forceSwitch` (battle-actions.ts:1167) sets the foe's `forceSwitchFlag` IFF `canSwitch(foe.side)` (≥1 eligible non-active, non-fainted bench mon); the ACTUAL drag happens at the runAction tail (battle.ts:2350, AFTER the move body, BEFORE faintMessages) via `drag_in` → `eligible_switch_ins` (= `possibleSwitches`, the array-order bench list) → `random(n)` (the `sample` — ONE draw, EVEN for n==1, since `random(1)` returns 0 but still calls `rng.next()` — the n=1 draw gotcha) → `execute_switch` (the array swap + the entrant's `updateSpeed` + the `insert_runswitch` enqueue). So the dragged mon takes Spikes via the EXISTING `runSwitch` EntryHazard, fires its ability `Start`, and a Spikes-KO on entry chains a NORMAL replacement — all through the existing switch machinery. A phaze with NO eligible foe (its last mon) FAILS draw-free (only the accuracy roll). **(4) THE PROTECT BLOCK — gen-3 Roar AND Whirlwind carry the `protect: 1` flag, so a Protect / Detect on the TARGET BLOCKS the phaze at `runEvent('TryHit')` (AFTER the accuracy roll) → NO `forceSwitchFlag` → NO drag → NO `sample` draw (`-activate Protect`, the target STAYS active). The phaze arm checks `protect_blocks(foe, foe_slot, false)` right after its accuracy roll (mirroring the leechseed / standalone-status arms); Substitute does NOT block a phaze (Roar/Whirlwind carry `bypasssub: 1` — no substitute check). MISSING this was the multi-phaze `sample` draw-POSITION desync (see EDGE_CASES.md ✅ FIXED): the port dragged an EXTRA `sample` into a protected foe the sim left in place, shifting every LATER phaze's `sample` PRNG position (same total draw COUNT, wrong `sample` INDEX, compensated elsewhere → the boundary seed matched while the dragged mon differed).** The phazed-OUT mon's boosts/volatiles are cleared (`execute_switch`); the dragged mon does NOT act this turn. A phaze DRAG that fires sets `pending_phaze_drag` → `DecisionRecord.phaze_drag` (a coverage/diagnostic signal only, no effect on any draw/state; the e2e capstone reads it to count phaze-drag decisions — a Protect-blocked / no-bench phaze does NOT set it). DEFERRED (fail-loud — `modeled_phaze_move` lists ONLY `roar`/`whirlwind`, everything else falls to the status-move fail-loud guard): **Haze** (resets boosts — a DIFFERENT mechanic, NOT `forceSwitch`), Perish Song, Roar of Time (not gen3). Roar + Whirlwind are the ONLY gen-3 phaze moves (`isPhaze` == `forceSwitch`). Validated by `tests/phaze_test.rs` (a per-seed PER-DECISION STATE(+HP+SPIKES-LAYERS+DRAG-SPECIES)+SEED+winner differential to GAME-END over 7 scenarios × 80 seeds in gen3customgame: 560 runs, 10388 seed + 20776 HP + 20776 spikes-layer assertions, 2795 drag decisions + 1769 phaze-into-spikes-damage rows, 560 wins — the seed sweep makes DIFFERENT mons get dragged [≥2 distinct per multi-bench scenario, the random-target proof], Roar/Whirlwind random drag, the n=1 sample, a Roar that FAILS, Roar INTO Spikes, repeated Roar into a stochastic spikes-KO, phaze-into-a-real-battle) + 4 DETERMINISTIC `tests/regression_test.rs` pins (`phaze_draws_accuracy_then_n1_sample_seed`, `phaze_fail_draws_only_accuracy_no_sample_seed`, `phaze_drag_into_a_spikes_ko_chains_a_replacement`, and **`phaze_blocked_by_protect_draws_no_sample_and_leaves_the_target`** [P4 — a Protect BLOCKS a Roar: accuracy drawn, NO `sample`, the protector stays active], ground-truth seeds from `harness/probe_phaze_regression_rng.js` [incl. its PHAZE-PROTECT case]). Phaze is **INCLUDED in the e2e capstone** (`PHAZE_E2E_EXCLUDED = false`, bit-for-bit — **1035 phaze-DRAG decisions across the 220-battle strict gate**, `filtered_diverged == 0`, `phaze_decisions >= 50` coverage floor) after fixing the Protect-blocks-phaze desync (above). Admitting phaze also surfaced an unmodeled `destinybond` (a reactive `volatileStatus` move Gengar can carry): it is now in `gen_e2e_fuzz.js`'s `MOVE_ID_BLOCKLIST` (belt-and-braces — `isModeledMove` already excludes it, so `pickMove` never picks it), and the port FAIL-LOUDS on it (pinned by `destinybond_status_move_panics_fail_loud`). **LEECH SEED now BUILT** (`run_status_move`'s leechseed arm + the `LeechSeed` `ResidualAction` + `apply_leech_seed`; new `MonState::leech_seed: Option<usize>`): a foe-targeting `volatileStatus:'leechseed'` Status MOVE (type Grass, **accuracy 90**) that plants the `leechseed` volatile on the FOE; each end-of-turn the seeded mon loses `⌊maxhp/8⌋` and the SEEDER's CURRENT active heals it. The draw model, VERIFIED bit-for-bit vs the omniscient sim PRNG probe `harness/probe_leechseed_rng.js`: (1) the MOVE DRAWS `randomChance(90,100)` — it CAN miss, drawn UNCONDITIONALLY (even into a Grass-immune or already-seeded target — VERIFIED: a splash/splash turn draws 1 [Quick Claw]; a Leech-Seed turn — land / Grass-immune / already-seeded-fail — ALL draw 2 [accuracy + Quick Claw]); (2) a **GRASS** target is IMMUNE (`onTryImmunity` → `!hasType('Grass')`; accuracy still drawn, then `-immune`, NO volatile); (3) an **ALREADY-SEEDED** target's re-seed FAILS (`addVolatile` false; accuracy drawn, "did nothing", the existing volatile unchanged); (4) on a landed non-immune non-already-seeded hit it PLANTS the volatile (DRAW-FREE); `landed` FALSE (a status `moveHit` returns `undefined` → the in-tryMoveHit Update is skipped). THE LEECH RESIDUAL (the risk area — DRAW-FREE but ORDER-SENSITIVE): the gen4-inherited override is **`onResidualOrder: 10, onResidualSubOrder: 5`** (NOT the base-data order-8) — so at order 10 the residual ladder is **Leftovers sub 4 → LEECH sub 5 → status DoT sub 6** (with sand/hail field-residual order 8 BEFORE all three). VERIFIED order `sandstorm[o=8] → leftovers[o=10,s=4] → leechseed[o=10,s=5] → brn[o=10,s=6]`. The drain is `⌊maxhp/8⌋` clamped to the seeded mon's HP (the sim's `damage()` return), the seeder's active HEALS that dealt amount (clamped to its maxhp) — applied even when the drain KOs the seeded mon (the heal is inside the same `onResidual`, before `faintMessages`). The **SEEDER-FAINTED gate**: if the seeder's active is fainted/0-HP the WHOLE `onResidual` returns early (`if (!target || target.fainted || target.hp <= 0) return`) — no drain, no heal. The leech is gathered with the VOLATILES (after the status DoT, before Leftovers, mirroring `findPokemonEventHandlers`'s status→volatiles→item order) so its handler participates in the residual speed-sort tie-shuffle (two seeded mons at equal speed TIE at order 10 sub 5 → one `random(0,2)`). The volatile clears on switch-out (`execute_switch`) and on faint (`process_faints`). DEFERRED (fail-loud in `apply_leech_seed`): a **Liquid Ooze** target reverses the drain (the seeder takes damage) — rare in gen-3 OU, excluded from the e2e filter. Validated by `tests/leechseed_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+SPIKES-LAYERS+**LEECH-SEEDED**)+SEED+winner differential to GAME-END over 7 scenarios × 80 seeds in gen3customgame: 560 runs, 5001 seed + 10002 HP + 10002 leech-state assertions, 3838 leech-seeded rows, 560 wins — seed lands→drain+heal, Grass-immune, already-seeded fail, the leech-drain KO, the leech+Leftovers+sand+burn 4-way residual ORDER, the seeder-replaced heal-follows, leech-into-a-real-battle) + 3 DETERMINISTIC `tests/regression_test.rs` pins (`leech_residual_order_leftovers_sand_burn`, `leech_handler_tie_at_equal_speed_draws_one_shuffle`, `leech_seeder_fainted_skips_the_drain`, ground-truth seeds from `harness/probe_leechseed_regression_rng.js`). **SUBSTITUTE now BUILT** (`run_status_move`'s substitute arm + `absorb_into_sub` in `run_move` + the secondary-suppression in `apply_secondaries`/`apply_triattack_secondary` + the sub-block in the status/leech arms; new `MonState::substitute: Option<u16>`): the SUBSTITUTE MOVE (`volatileStatus:'substitute'`, `target:'self'`, never-miss) spends `floor(maxhp/4)` HP to create a decoy with that much HP that ABSORBS incoming foe hits. The draw model, VERIFIED bit-for-bit vs the omniscient sim's PRNG probes (`harness/probe_substitute_*.js`): (1) the MOVE is NEVER-MISS (no accuracy draw); it FAILS draw-free if a `substitute` is ALREADY present OR `hp <= floor(maxhp/4)` (can't afford — VERIFIED: hp == floor(maxhp/4) FAILS, +1 SUCCEEDS); on success it pays `floor(maxhp/4)` HP + creates the volatile with that HP (DRAW-FREE; `landed` FALSE). (2) THE ABSORB/BLOCK CRUX (the SURPRISE that **CONTRADICTED the task's stated "one fewer random(100)" assumption** — settled by the probe, the project's source of truth): a DAMAGING foe move into a substituted mon draws acc+crit+damage as normal (UNCHANGED count) and the damage hits the SUB's HP (the sub BREAKS at 0; the excess does NOT carry to the mon in gen-3) — **and the per-move SECONDARY `random(100)` is STILL DRAWN** (gen-3 `secondaries()` iterates the now-`null` target list, so the draw fires — the SAME count as a bare hit), but its FOE-TARGETING EFFECT does NOT apply (no status / no foe stat-drop / no flinch / no confusion, AND **no confusion `random(2,6)` / Tri-Attack `random(3)` follow-on draw** — those are SUPPRESSED). A **SELF-boost** secondary (Meteor Mash +1 Atk to the USER) is EXEMPT — it STILL APPLIES through a sub (`secondary.self.boosts` targets the SOURCE, not the null sub target; VERIFIED vs the sim), so the suppression is FOE-targeting-only (`effect != "self_boost"`). So `absorb_into_sub` routes the damage; `apply_secondaries`/`apply_triattack_secondary` take an `absorbed_by_sub` flag that DRAWS the `random(100)` then SKIPS the FOE-targeting apply+follow-on (but keeps the self-boost). This is draw-COUNT-NEUTRAL vs a bare hit (unlike the phaze `sample` that ADDS a draw). (3) a STATUS / stat-DROP move (Thunder Wave / Toxic / Leech Seed / a -stat secondary) is BLOCKED by the sub (accuracy still drawn, then `-fail`/no effect — DRAW-FREE past accuracy; the status arm + the leech arm both check `substitute.is_some()` after the protect block). (4) a CONFUSION self-hit hits the MON, NOT the sub (the self-hit's `this.damage` bypasses the `onTryPrimaryHit` sub-intercept — `on_before_move`'s confusion arm is unchanged: `randomChance(1,2)` then `random(16)` on the mon). (5) PHAZE (Roar / Whirlwind) BYPASSES the sub — the user is dragged anyway (forceSwitch is a runAction-tail effect, not a moveHit target). The substitute clears on switch-out (`execute_switch`) and on faint (`process_faints`). Validated by `tests/substitute_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+BOOSTS+CONFUSION+SPIKES-LAYERS+**SUB-HP**)+SEED+winner differential to GAME-END over 9 scenarios × 80 seeds in gen3customgame: 720 runs, 4320 decision rows — create + already-subbed FAIL, the low-HP create FAIL boundary, a HELD-sub absorb (sub HP drops, secondary suppressed), the BREAK no-carry, a blocked STATUS move, a blocked STAT-DROP secondary, the CONFUSION self-hit hitting the mon, the phaze drag-through, and sub-into-a-real-battle to a win) + 5 DETERMINISTIC `tests/regression_test.rs` pins (`substitute_absorbs_a_hit_but_the_secondary_random_100_still_draws` [the draw-COUNT crux], `substitute_break_does_not_carry_excess_to_the_mon`, `confusion_self_hit_behind_a_sub_hits_the_mon_not_the_sub`, `tri_attack_into_a_sub_draws_random_100_but_not_the_sample_random_3` [the random(3)-suppression draw-COUNT], `self_boost_secondary_still_applies_through_a_sub` [a SELF-boost is exempt from the sub-block], ground-truth seeds from `harness/probe_substitute_regression_rng.js`). The draw model was nailed by `harness/probe_substitute_rng.js` + `probe_substitute_secondary.js` + `probe_substitute_confusion.js` + `probe_substitute_status.js`. Substitute is **INCLUDED in the e2e capstone** (`SUBSTITUTE_E2E_EXCLUDED = false`, 284 substitute-MOVE / 320 sub-up decisions across the 220-battle strict gate, bit-for-bit) after FIXING the SWITCHING/weather bug it surfaced — the **`eachEvent('WeatherChange')` switch-in tie-shuffle**: `Field.setWeather` (field.ts:87) ends with `this.battle.eachEvent('WeatherChange', sourceEffect)`, a 2-active `speedSort` that draws ONE `random(0,2)` tie-shuffle iff the actives TIE on cached speed (gen-3 `>=7` Update-nest NOT reached). When a Sand Stream / Drizzle / Drought entrant CHANGES the weather on a MID-TURN switch-in, that shuffle fires INSIDE the `runSwitch` runAction before its trailing Update; the port set the weather draw-FREE and MISSED it (e2e_84 dec4: a 213-Tyranitar switches in under Sand Stream while a 213-Suicune subs → the sim drew 8, the port 7 — a draw-COUNT desync, the SAME class as `forced_replacement_recaches_speed_seed`, in the SWITCHING/weather layer, NOT the substitute arm). The fix: `run_switch` snapshots `(weather, weather_turns)` across the ability `Start` and returns whether the weather changed; `turn_loop`'s `RunSwitch` handler then fires one `each_event_shuffle()` (for EVERY mid-turn switch-in — voluntary / forced-replacement / phaze-drag — since all route through that runAction). The `>start` path (`run_start_switchins` → `single_event_ability_start` directly) is UNAFFECTED (stays draw-free; `switchin_test`'s zero-draw assertion holds). Pinned by `regression_test.rs::switch_into_a_tie_under_sand_draws_the_weather_change_shuffle_seed` (a constructed 213-vs-213 switch-into-tie-under-Sand-Stream, real-Showdown ground-truth seed from `harness/probe_switch_tie_weather_regression_rng.js`; verified a TRUE pin by reverting the fix). See EDGE_CASES.md "switch-in-into-a-speed-TIE" (now FIXED) + the `probe_switch_sand.js` control (the +1 shuffle: Sand-Stream switch-into-a-TIE draws 9 vs no-weather 7 vs distinct-speed 1). DEFERRED: Baton Pass passing a sub, Shed Tail (gen-9), the `bypasssub`/`infiltrates` move flags (none on a modeled gen-3 move). **FIXED-DAMAGE / FIXED-FORMULA MOVES now BUILT** (`run_fixed_damage_move`, routed by `is_fixed_damage_move` in `run_move` BEFORE the `category == Status` branch — these carry `basePower:0` so `derive_category` classifies them Status; new id-gated helper `fixed_damage_amount`): the `damage:` / `damageCallback` moves that BYPASS `getDamage` (NO crit roll, NO 16-way damage roll). MODELED — **Seismic Toss / Night Shade** (`damage:'level'` → the USER's level, e.g. 100), **Sonic Boom** (fixed 20), **Dragon Rage** (fixed 40), **Super Fang** (`damageCallback` = `max(floor(target.hp/2),1)`). The draw model, VERIFIED bit-for-bit vs the omniscient sim's PRNG probe (`harness/probe_fixeddamage_rng.js`): (1) ACCURACY — `randomChance(acc,100)`, drawn UNLESS never_miss — **Seismic Toss / Night Shade / Dragon Rage are acc-100 but NOT never-miss so they STILL draw ONE accuracy roll** (the phaze acc-100 precedent), **Sonic Boom / Super Fang are acc-90 and CAN genuinely miss**; this is the ONLY per-move draw (NO crit, NO damage roll, NO secondary); (2) TYPE IMMUNITY — accuracy-drawn-THEN-`-immune` (the SAME short-circuit + draw count as a normal damaging move, via `move_is_immune`): **Seismic Toss (Fighting)→a GHOST**, **Night Shade (Ghost)→a NORMAL**, **Sonic Boom / Super Fang (Normal)→a GHOST** all report `-immune` (NOT `-miss`) with ZERO damage; (3) DAMAGE — the exact fixed amount applied through the EXISTING `absorb_into_sub` / `apply_damage` / deferred-faint machinery, so a fixed-damage KO goes through the normal faint/win/Quick-Claw protocol (no Quick Claw on a deciding faint). **The SUBSTITUTE interaction (VERIFIED, and it CONTRADICTED a naive assumption — settled by the probe):** a fixed-damage move into a sub hits the SUB (the fixed NUMBER hits the sub HP, breaks with no carry, `-activate Substitute [damage]` / `-end`) — and **Super Fang still halves the MON's current hp behind a sub** (the `damageCallback` reads `target.hp` BEFORE the sub-intercept redirects the resulting number; VERIFIED: SF into a full-HP-536 Blissey behind a 178-HP sub deals floor(536/2)=268 → the sub BREAKS, NOT floor(178/2)=89). `landed` is TRUE on a hit (a `damage:` move returns a truthy number → the in-tryMoveHit Update fires), FALSE on miss/immune/block. FAIL-LOUD (`run_fixed_damage_move` PANICS): the DEFERRED fixed-damage family — **Psywave** (variable, draws RNG), the OHKO moves **Fissure / Horn Drill / Guillotine** (accuracy-gated instakill + level gate), **Counter / Mirror Coat / Bide** (reactive), **Endeavor** (sets hp to the user's) — is routed here by `is_fixed_damage_move` (so it can NEVER silently no-op / desync) but has no `fixed_damage_amount` entry. Validated by `tests/fixeddamage_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS)+SEED+winner differential to game-end over 9 scenarios × 80 seeds in gen3customgame: 720 runs, 4144 seed + 8288 HP assertions, 2469 fixed-damage-hit decisions, 720 wins — Seismic Toss chip / into-a-Ghost-immune / Night Shade into-a-Normal-immune + lands / a fixed-damage KO-to-win / into-a-Substitute / Sonic Boom+Dragon Rage incl. the acc-90 miss / Super Fang halving incl. the miss / fixed-damage-into-a-real-battle) + 4 DETERMINISTIC `tests/regression_test.rs` pins (`seismic_toss_deals_user_level_damage`, `seismic_toss_into_a_ghost_is_immune_accuracy_only_seed`, `night_shade_into_a_normal_is_immune`, `fixed_damage_into_a_substitute`; ground-truth seeds from `harness/probe_fixeddamage_regression_rng.js`). e2e: the `MODELED_FIXED_DAMAGE_MOVES` set is in `gen_e2e_fuzz.js`'s `isModeledMove` (admitted early, before the `basePower>0`/`m.damage`/`m.damageCallback` rejects), but the regenerated 220-battle golden has **0 fixed-damage-MOVE decisions** because NONE of the 22 filter-clean teams happens to carry one (the leech-seed situation) — so the layer is proven by its DEDICATED golden + the 4 pins, not the e2e. **TAUNT + DISABLE (the move-SELECTION-restriction layer, `gen3_taunt_disable_v1`) now BUILT** (`run_status_move`'s taunt/disable arms + the `move_usable`/`must_struggle` restriction + the `on_before_move` execution-time cants + the residual duration ticks + the endTurn `runEvent('DisableMove')` handler-sort shuffle): **Taunt** (Dark, acc 100 — DRAWS `randomChance(100,100)`) applies a FIXED-duration-2 volatile (NO duration draw; the base onStart's `duration++` is SHADOWED by the gen4 mod's replaced onStart — probe-proven constant 2 in ALL branches) that makes every Status-category slot un-selectable (the derived-Status set MINUS the fixed-damage family) + cants a QUEUED status move at execution (`onBeforeMove` priority 0, AFTER the paralysis roll), residual tick at order 10/subOrder 15 (gen4-inherited, NOT the base's order 15); **Disable** (Normal, acc 55 — CAN miss) disables the target's `last_move` slot for **stored = disabler-faster ? random(2,6) : random(2,6)+1** (the gen4-inherited onStart `!willMove → duration++`; PROBE-SETTLED — a base-source reading mis-predicts by a constant +1), onTryHit FAILS draw-free with no lastMove, + cants a QUEUED now-disabled move at execution (`onBeforeMove` priority 7, BEFORE confusion/paralysis — a paralyzed+disabled mon draws NO para roll), residual tick at NO_ORDER/subOrder 2 (the Condition default; gen3 DELETES gen4's 10/13); both `protect: 1` + `bypasssub: 1`, re-application fails draw-free, cleared on switch-out + faint (with `last_move` reset). All slots restricted (taunt × disable × Choice lock × 0 PP) → forced Struggle. See "Taunt + Disable" below. Validated by `tests/taunt_disable_test.rs` (9 scenarios × 80 seeds: 720 runs, 4723 seed + 8595 taunt + 8595 disabled-slot assertions, free-up boundaries on BOTH disable branches — the golden gate FAILS on a ±1 duration perturbation, proven both ways) + 4 revert-verified `regression_test.rs` pins (TD1–TD4). |
| `protocol.rs` | **types + Phase-1+2 emit API, validated** | `Player`, `Choice` (→ wire grammar `move 1`/`switch 3`/…), `ProtocolLine` (raw bytes = source of truth), AND the **`ProtocolBuilder`** — an append-only, **PRNG-free** line buffer the engine writes at each observable event (one sim-mirroring retro-edit excepted: `attr_last_move_still` = `attrLastMove('[still]')`, used by the Disable 0-PP-guard fail). Centralizes the fiddly formatting in ONE place: `MonRef` (`p<N>a: <Name>`) / `HpStatus` (the three variants `x/y` / `x/y <status>` / `0 fnt`) / `Cause` (`[from] item: …` / `[from] ability: …` / `[from] move: …` / `[from] <bare>`) / `STAT_TOKENS` (the `-boost`/`-unboost` stat names). Typed constructors for the Phase-1 line types (framing / `turn` / `upkeep` / separator / `move`+`[miss]`/`[still]` / `switch` / `drag` / `-damage` (+ the `damage_of` `[from] Recoil|[of] <target>` form, `gen3_pp_tracking_v1`) / `-heal` / `faint` / `-crit` / `-supereffective` / `-resisted` / `-immune` / `-miss` / `win` / `tie`) **PLUS the Phase-2 types** (`status`+`[from] move: Rest` / `curestatus`+`[msg]` / `cant` / `boost`/`unboost` by sign / `weather` SET+`[upkeep]` / `ability`+detail / `fail`+`[weak]` / `sidestart` / `volatile_start`/`volatile_end`/`activate`+`[damage]` / `singleturn`). Disabled by default (the seed suite pays zero cost + draws nothing); `run_full_battle_logged` enables it. See "Protocol emission" below. |
| `battle.rs` | **write_line DONE, validated** | `BattleOptions`/`PlayerOptions`/`PackedTeam` + `Battle::start`/`start_with_switchins` (construction ± switch-in events over `state::BattleState`) + **`BattleStream::write_line`** (`gen3_writeline_stream_v1`) — the streaming drop-in for the bridge's `local_sim_bridge.js` pattern: `>start {formatid,seed}` / `>player pN {name,team}` / `>pN move K\|switch N` per write, returning EXACTLY the omniscient `\|...\|` chunk the real Node `BattleStream` flushes for that write (per-write byte-gated by `tests/writeline_test.rs` vs `harness/gen_writeline_capture.js` — 44 battles / 2377 writes / 7276 filtered lines, all 22 scenarios). INTERNALS (honest scope): replay-from-genesis — every choice write re-runs `run_full_battle_logged` over the accumulated ONE-SIDED `ScriptDecision` stream (the per-side pending acceptance pairs them; deterministic bit-for-bit ⇒ each replay reproduces every prior byte) and returns the new suffix; an illegal choice is rejected like `side.choose` (no lines, boundary open). The `>start` seed must be the PRE-first-decision seed (the sim's turn-0 construction window is unmodeled — the protocol-replay convention). CHOICE REVISION is FIRST-accepted-wins, NOT the sim's last-write-wins (`gen3_writeline_choice_revision_v1`, review finding F4): a repeat pre-commit `>pN` APPENDS a fresh one-sided `ScriptDecision` instead of REPLACING the pending one, whereas the sim's `side.choose` clears + re-parses on each write (probe `harness/probe_f4_choice_revision.js`, resolved Dex.mod('gen3'): `>p1 move 1` then `>p1 move 2` executes Earthquake, seed-identical to a single `>p1 move 2`; a 1→2→3 chain executes Rest). DOCUMENTED-not-fixed by design — the replay-from-genesis accumulator carries no open-boundary marker, so a same-side overwrite can't be told apart from a same side in an already-COMMITTED prior decision, and an overwrite rule fired on the latter DESTABILIZES the writeline gate (a forced-replacement `>p2 switch N` after a completed turn dropped the replacement chunk — verified); the real bridge sends exactly one choice per request, so a revised `>pN` is unreachable in production. The `\|request\|`/sideupdate frames + the per-side privacy fold (downstream of the omniscient stream) are now DONE in **`bridge.rs`** (the row below). Still OUT of scope on `write_line`: `>forcewin`/`>reseed`. Snapshot/reseed/choose on `Battle` stay `todo!()` (the search layer's JS clone path covers them today). |
| `bridge.rs` | **DONE, validated (in-scope corpus)** | The PER-SIDE (`p1`/`p2`) protocol streams a drop-in for `src/utils/bridge/local_sim_bridge.js` emits — Phase-1 gaps **G1 (`\|request\|` JSON)** + **G2 (`getPlayerStreams` split + HP-privacy fold)** ON TOP of the unchanged omniscient stream (additive, like `write_line`). `run_full_battle_bridge(opts, cmds, dex) → BridgeStreams{p1,p2}` replays-from-genesis at each boundary to snapshot the pending request, folds the flushed omniscient batch to each side, and injects each side's `\|request\|`/`\|error\|`. **G2 fold** (from `battle.js::extractChannelMessages` + `pokemon.js::getHealth` + `mods/gen3/abilities.js`): HP-bearing lines (`\|switch\|`/`\|drag\|`/`\|-damage\|`/`\|-heal\|`/`\|-sethp\|`) show the OWNER exact HP, the OTHER side `ceil(hp*100/maxhp)/100` (clamp 100→99 when `hp<maxhp`; `0 fnt` unchanged) — but ONLY in a non-debug format; **gen3customgame sets `reportExactHP`** (its `[Gen 3] Custom Game` def has `debug:true`) so both sides see exact HP. Owner-only (empty-`shared`, dropped for the other side): gen3 Pressure's `\|-ability\|pNa: X\|Pressure\|[silent]` (`addSplit`) + the Intimidate no-activate `\|-hint\|` (owner tagged by the driver since a hint has no `pNa:` prefix). **G1 request** serialized from the crate's EXISTING legality (`is_trapped`/`move_usable`/`switch_flag`), Showdown field+key order + compact `JSON.stringify` (no spaces after `:`/`,`): per-move `{move,id,pp,maxpp,target,disabled}` (typed HP → bare `id:"hiddenpower"` + `move:"Hidden Power <Type> <BP>"`), `active[0].{maybeTrapped\|trapped}` — the `'hidden'` traps (Arena Trap / Magnet Pull) show `maybeTrapped` on the first request (firmed to `trapped` only after a rejected switch → `[Unavailable choice]` + re-request), but **Shadow Tag** (the gen3 mod sets `trapped = true` directly) shows `trapped:true` from the FIRST request + a rejected switch draws `[Invalid choice]` with NO re-request (`state::trap_is_firm`, `gen3_shadowtag_firm_trap_v1`); a forced-Struggle mon also gets a per-side OWNER-ONLY `\|-activate\|<mon>\|move: Struggle` sideupdate line before the broadcast `\|move\|` (`gen3_struggle_activate_sideupdate_v1`) — both found by the bridge/request A/B fuzzer. `side.pokemon[]` in the crate's Showdown-faithful swap-reordered array order; `forceSwitch:[true]`+conditional `noCancel` (iff <2 non-wait requests, mirroring `multipleRequestsExist`); the non-choosing side's `{wait:true}`; the trapped-reject `\|error\|` + `trapped:true` re-request with trailing `"update":true`. gen3 has NO team preview. gen3ou framing (`\|tier\|[Gen 3] OU` + the 12 `\|rule\|` list) spliced in via `reframe` (the crate's `emit_framing` hard-codes the gen3customgame tier). **BYTE-GATE**: `tests/bridge_test.rs` — the trapping golden (`bridge_trapping_golden.txt`: 3 battles — arena_trap_reject / magnet_pull_reject / **shadow_tag_firm_trap** — 226 CHUNK lines, `\|request\|` / `trapped:true` / `\|error\|[Invalid\|Unavailable]` frames) is byte-for-byte, the definitive Phase-1 pass (the constructed IN-SCOPE corpus: explicit genders, MODELED moves only). The gen3ou `bridge_capture_golden.txt` (30 battles) is a scope-AUDIT: 28 fail-loud on unmodeled moves (Counter/Wish/Encore/Curse/Baton Pass/Endeavor/Refresh/Sunny Day — out of the ENGINE's faithful-replay scope, not a bridge defect) + ALL 30 carry a gendered species with an UNSPECIFIED gender → an unmodeled construction-time gender-ratio `sample` PRNG draw that desyncs the battle + adds the drawn gender to `details`; the audit asserts the bridge layer is byte-correct (gender-tolerant) up to that first ENGINE-scope divergence (framing + full first request proven on the 2 replayable battles). When the engine models the full move set + the gender draw, these become full byte-equal with NO bridge change. Entry point for the request A/B fuzzer: **`src/bin/bridge_replay.rs`** (`bridge_replay <golden> [id] [--print] [--ab]`; `--ab` = one JSON verdict/battle, panic-caught — the `harness/bridge_ab_fuzz.js` driver; also replays a repro DIR). See "## Bridge / request A/B fuzzer". Observation-only: the omniscient path is UNCHANGED (protocol/writeline/e2e_fuzz green, e2e md5 `a23d77ac60d4af168b8a4428f0b465c9`; the two fuzzer-found fixes [`gen3_shadowtag_firm_trap_v1` firm `trapped`, `gen3_struggle_activate_sideupdate_v1` per-side Struggle line] are bridge-path only + the `switch_details` gender addition is byte-neutral for the genderless goldens). **PHASE 2 (`gen3_sim_bridge_dropin_v1`): the CHUNK-aware `run_full_battle_bridge_chunked(opts,cmds,dex) → BridgeChunks{chunks:Vec<SideChunk>}`** preserves the `getPlayerStreams` FLUSH boundaries (framing → 3 chunks/side split at the two `\|player\|` lines; a resolved turn → 1 log chunk/side + 1 `\|request\|` chunk/side; a trapped reject → a `\|error\|` chunk + [hidden trap] a `trapped:true` re-request chunk; a forced-Struggle commit → a `\|-activate\|move: Struggle` chunk) — the flush unit the Node bridge base64-frames as ONE `pN <b64>` stdout line. `run_full_battle_bridge` now delegates to it via `.flatten()` (so the line-level `bridge_test` golden still validates the chunk logic — pinned by `chunked_flatten_equals_flat_streams`). `advance_seed_for_construction(raw)` advances a raw `>start` seed by the turn-0 Quick Claw `random(1,5)` (the pre-first-decision seed convention; pinned by `construction_seed_advance_matches_the_sim` vs the sim). These feed **`src/bin/sim_bridge.rs`** (below). |
| `bin/sim_bridge.rs` | **DONE, validated** | The drop-in **`node local_sim_bridge.js` REPLACEMENT** — a std-only Rust binary speaking the EXACT stdin/stdout protocol (`START <json>` / `CHOOSE <side> <choice>` / `FORCELOSE <side>` / `END`; `pN <base64(chunk)>` / `__END__` / `__ERR__ <b64>` frames, base64-per-chunk), so it swaps behind the Python bridge (`local_battle_runner.py`) with ZERO protocol change. Replay-from-genesis like `write_line`: accumulates the `CHOOSE` stream, re-runs `run_full_battle_bridge_chunked_ended` each write, emits the NEW chunk suffix; `__END__` on battle-end (+ persistent-mode reset on `persistent:true`). BYTE-VALIDATED vs the real Node bridge by **`harness/gen_sim_bridge_diff.js`** (spawns BOTH subprocesses, drives choices off the Node bridge's `\|request\|` frames, diffs PER-SIDE chunk sequences — `\|t:\|` normalized, `\|debug\|` dropped): trapping (Arena Trap) + persistent multi-battle-per-child are byte-identical over ~120+ battles. DEFERRED (honest, documented in the binary): **`__RECON__`** (the sim's internal `inputLog` — the port has no byte-identical `input_log`; EXCLUDED from the diff, the Python `_offer_recon` degrades gracefully) + **`resumeReseed`** (needs `Battle::reseed`, still `todo!()`; ignored with a stderr note — both serve the search layer, not core training/eval). SEED-CONVENTION GAP (not a bridge defect): the port's `run_full_battle` omits the sim's turn-0 CONSTRUCTION endTurn, so a **speed-TIED lead matchup** (or Magnet Pull's `onAny` trap shuffles at a tie) draws extra construction shuffles `advance_seed_for_construction` doesn't model → the turn-1 first-mover can flip on a raw seed (the diff harness skips speed-tied leads; Magnet Pull / Shadow Tag are behind `--extra-trappers`). With `seed=None` (the training default) there's no reference so it's a no-op. |

## The callable surface (battle.rs) maps to the existing bridge

`battle.rs` deliberately mirrors the FIVE ways `src/utils/bridge/` already drives
Showdown, so a finished core is a drop-in:

| Bridge today (Node)                     | Rust surface |
|---|---|
| streaming battle (`local_sim_bridge.js`) | `BattleStream::write_line` |
| mid-battle RNG swap (counterfactual/search) | `Battle::reseed` |
| clone-and-branch (`State.serialize…`)    | `Battle::serialize` / `deserialize` (must preserve PRNG continuity) |
| damage oracle (`damage_probe.js`)        | `Battle::new` + state accessors (TODO) |
| team pack / validate                     | out of core scope — keep as a thin shim |

When you implement the engine, keep these signatures stable; the bridge contract
is the spec.

## PRNG: the bit-for-bit gate (level-1 differential test)

- **Port** lives in `prng/`. `SodiumRng::next` = ChaCha20-encrypt a 36-byte zero
  buffer (key = 32-byte seed, nonce = `"LibsodiumDRG"`, counter 0); next seed =
  bytes `[0,32)`, output = big-endian u32 of bytes `[32,36)`. `Gen5Rng` = the
  64-bit LCG (`a=0x5D588B656C078965`, `c=0x00269EC3`) over four 16-bit words.
  ChaCha20 is hand-rolled (no crate) so the whole thing is std-only and auditable.
- **Reference + harness** in `harness/`. `prng_reference.js` is a dependency-free
  JS re-derivation (the executable spec the Rust mirrors). `gen_prng_vectors.js`
  cross-checks that reference against the **real** `prng.js` value-by-value and
  **aborts on any mismatch**, then emits `tests/vectors/prng_golden.txt` —
  ~2900 self-contained assertions (each keyed on a pre-state seed string, so
  `Prng::new(seed)` reconstructs and checks one call; no JSON parser needed).
- **Rust gate**: `tests/prng_golden.rs` replays every assertion. `cargo test`
  must stay green. Regenerate vectors after any PRNG change (see README).

This proves the determinism foundation in actual Rust: same seed ⇒ same draws ⇒
the rest of the engine can be trusted to replay.

## Dex: the source-of-truth gate (data differential)

- **Loader** in `dex/`. `Dex::for_gen(3)` reads `data/pokemon/*.json` via the
  std-only [`json`] reader and exposes `species`/`moves`/`item`/`ability`/
  `nature`/`type_chart`/`learnset`, plus `Type`/`MoveCategory`/`BaseStats` and a
  `to_id` normalizer (poke-env's `to_id_str`). It reads the SAME files the Python
  runtime reads, so the only real *logic* is `moves::derive_category` (the gen ≤ 3
  type-based physical/special split) — kept behind the `gen` parameter.
- **Parity harness** `harness/gen_dex_golden.py` dumps the `agents.gen3_data`
  facade's view (every species, every move incl. the derived category + resolved
  type, the full type chart, natures, learnsets) to
  `tests/vectors/dex_golden.txt`. `tests/dex_test.rs` asserts the Rust dex
  reproduces all ~1500 lines, so Rust and the Python runtime agree by
  construction (regenerate after any data/derivation change).

## Team: the packed-string gate (Showdown differential)

- **Codec** in `team.rs`. `unpack`/`pack` mirror Showdown's `Teams.unpack`/`pack`
  (`teams.ts`) for one team. `unpack` is a **sequential field walk** (NOT a
  `split(']')` — a `]` is legal inside a nickname) and is case-insensitive, so it
  ingests both Showdown's case-preserving form AND poke-env's lowercase-id form
  (our real bridge producer); `pack` re-emits Showdown-canonical bytes for the
  `>player` consumer. Bounded, documented deviations on *malformed* input only
  (numeric width, multi-char gender) — never reachable from a validator-clean team.
- **Differential harness** `harness/gen_team_golden.js` captures, from the REAL
  Showdown `Teams`, `(IN, UNPACK, PACK)` triples for ~10 constructed sets, each
  ALSO in poke-env lowercase form, PLUS hand-crafted raw fixtures pinning the
  fiddly decodes (`]`-in-nickname, short IV field `30,`, trailing-comma moves,
  empty species). `tests/team_test.rs` asserts `unpack(IN)` == UNPACK and
  `pack(unpack(IN))` == PACK for all 24. **These edge fixtures exist because an
  adversarial review caught four real bit-parity bugs the happy-path golden
  missed — a regression here must stay caught.**

## Stats: the in-battle-stat gate (sim-truth differential)

- **Calc** in `stats.rs`. `compute_stats` mirrors Showdown's `statModify`
  (`battle.js`) exactly: `EV/4` floored, `*level/100` floored, nature applied
  AFTER `+5` as **integer** math (`floor(stat*110/100)`, never f64), HP never
  natured, and the **Shedinja `maxHP` hook** (`setSpecies` overrides HP with
  `species.maxHP`). `overflowstatmod` clamps are gen3ou-absent and omitted.
- **`maxHP` is real Showdown data**, carried end-to-end: the extractor
  (`tools/pokemon_data_extractor/sync.py` `build_species`) now passes `maxHP`
  through from the pokedex → `gen3_species.json` (only Shedinja has it) →
  `dex::SpeciesData::max_hp`. So it's data-driven and **resync-safe**
  (`extractor_parity_test` passes); the Python obs facade ignores the key.
- **Differential harness** `harness/gen_stats_golden.js` drives an in-process
  omniscient `BattleStream` (the `damage_probe.js` pattern, no server) and reads
  each mon's `storedStats` + `maxhp` — the sim's OWN stats — over 18 cases (all-0
  and max EV/IV, every nature direction, levels 100/78/5/1, Shedinja, min-IV,
  extreme bases, non-÷4 EVs). `tests/stats_test.rs` reconstructs each input via
  `team::unpack` and asserts `compute_stats` matches.

## State: the construction-time gate (sim-state differential)

- **`BattleState::start`** (`state.rs`) composes the lower layers: `team::unpack`
  both teams → `stats::compute_stats` per mon → set `hp=maxhp=stats[0]`, `status=None`,
  `boosts=[0;7]`, lead = slot 0 (gen-3 singles, no team preview), `turn=0`,
  `field.weather=None`. It runs **NO switch-in events** — `event.rs` does (below).
- **The load-bearing split:** `start` builds/asserts only *construction-time*
  fields (stats, maxhp, hp, species, level, lead). Switch-in **event** effects —
  boosts (Intimidate), `field.weather` (Sand Stream), ability `Start` — are NOT in
  `start`, even though the golden's teams (Tyranitar/Salamence/Gyarados leads) make
  them fire in the sim; they live in `start_with_switchins` (`event.rs`) and have
  their own golden (next section).
- **Differential harness** `harness/gen_state_golden.js` starts a real gen3 battle
  (omniscient `BattleStream`), reaches the first request, and dumps each of the 12
  mons' `speciesid/level/maxhp/live-hp/storedStats/lead`. `tests/state_test.rs`
  feeds the identical packed teams + seed to `Battle::start` and asserts a match.

## Switch-in events: the `>start` event gate (post-switch-in sim differential)

- **`event.rs`** ports the two `sim/battle.ts` dispatch primitives in a reusable
  shape (moves/residuals reuse them later): `single_event_ability_start`
  (`singleEvent` — fire ONE effect's callback, no gather/sort/RNG/`modify`) and
  the generic `speed_sort` over `EventHandler<H>` — the `order→priority→speed→
  subOrder→effectOrder` (descending) selection sort whose **per-tie-group
  Fisher-Yates shuffle is the only PRNG draw inside a `runEvent`** (the
  RNG-consumption crux). Only the switch-in handlers are wired this step.
- **`BattleState::run_start_switchins`** runs the deferred `>start` half: both
  leads' switch-in ability events fire in **raw-Speed order (faster first)** — the
  order the two `runSwitch` actions dequeue. Wired abilities: **Intimidate** (gen3
  override) drops each foe's Atk stage by 1, clamped ≥ −6; **Sand Stream / Drizzle
  / Drought** `setWeather` to Sand/Rain/Sun with the gen-3 **permanent** duration
  (`weather_turns == 0`, the ability-source `onFieldStart` quirk). When both leads
  set weather the **slower fires last so its weather wins** (verified).
- **The gen-3 dispatch path** is gen4's `runSwitch` (`data/mods/gen4/scripts.ts`,
  inherited): per mon `runEvent('EntryHazard')` → `runEvent('SwitchIn')` (no gen-3
  ability has an `onSwitchIn` that touches the validated boosts/weather — Truant's
  only sets `truantTurn`, Trace acts via `onStart`; the `getCallback`
  `onStart`-fallback is `gen>=5`) → `singleEvent('Start', ability)`. So
  Intimidate/Sand Stream fire from the explicit ability `Start`, NOT the `SwitchIn`
  `runEvent`.
- **Differential harness** `harness/gen_switchin_golden.js` starts 5 real gen3
  scenarios (omniscient `BattleStream`) — Sand+Intimidate (both directions), the
  single-setter Drizzle/Drought, and the ORDER-dependent double-weather
  (Drizzle-faster vs Sand-slower ⇒ sand) — and dumps each lead's Atk boost +
  `field.weather`. `tests/switchin_test.rs` feeds the identical teams + seed to
  `Battle::start_with_switchins` and asserts a match; `event.rs`'s unit tests pin
  `speed_sort`'s tie-shuffle draw count (size-`k` group ⇒ `k−1` draws) bit-for-bit.
- **PRNG scope / carry-forward (honest):** OUR switch-in dispatch is **draw-free**.
  The real sim's `>start` window is NOT — even at distinct speeds it draws the
  per-mon **gender `sample`** (a *construction*/`addPokemon` draw), the gen-3
  **Quick Claw `randomChance(1,5)`** (a *turn-loop* `endTurn` draw), and, on a
  raw-Speed tie, the `BattleQueue.insertChoice` random splice + per-action
  `eachEvent('Update')` shuffles — but **all in phases this bounded step does not
  build**, so they are DELIBERATELY OMITTED (not prng-state parity). So `state.prng` after `run_start_switchins` is NOT yet a
  full `>start` RNG-state oracle (the validated output is the boosts/weather); a
  later queue/turn-loop step owns those draws. The `None`-seed gen5 `0,0,0,0`
  fallback (`state.rs`) is still harmless here only because the switch-in dispatch
  draws nothing — keep determinism tests on an explicit seed (the harness pins
  `[1,2,3,4]`), and when those bracketing draws are built, make `None` mirror
  Showdown's **sodium** default.

## Damage: the single-hit-physics gate (omniscient-oracle differential)

- **`calc_damage`** (`damage.rs`) is a self-contained pure function over EXPLICIT
  inputs (`DamageContext`: attacker/defender `Combatant` stats+types+boosts+status,
  the `MoveInput`, `weather`/`reflect`/`light_screen`, a resolved `crit`, and the
  caller-resolved `atk_stat_mods`/`defender_thick_fat`/`immune`) — it needs **no
  `BattleState`**, so the harness can construct any scenario. It returns
  `DamageResult { base: u16, rolls: [u16;16] }`: `base` is the deterministic max-roll
  (`random(16)==0`) damage; `rolls[r] = randomizer at r`, `rolls[15]` the 85% min.
- **Two stages, mirroring the source split** (the classic gen-3 trap): (1)
  `get_base_damage` reads atk/def via the boost-table floor, folds the ATTACKER
  STAT-event chain (CB ×1.5 / type-item ×1.1 / Sea Incense ×1.05 / the species stat
  items + the ability ModifyAtk folds Huge/Pure Power ×2 / Guts ×1.5-statused), the
  DEFENDER STAT-event chain (`def_stat_mods` — DeepSeaScale / Metal Powder / Soul Dew
  SpD + Marvel Scale ×1.5-Def-statused, before the Explosion Def-halve) + the BASE-POWER
  phase (**gen3 Thick Fat = `onSourceBasePower` ×0.5**, inherited from gen4 — NOT a stat
  mod; the modern data's `onSourceModifyAtk/SpA` is wrong for gen3 — joined by the
  item `bp_mods`: the incenses' chain + the bows' DIRECT float + the **pinch family**
  Torrent/Blaze/Overgrow/Swarm ×1.5 at `3*hp<=maxhp`, see "## Data-driven
  mechanics" — + **Facade's own `onBasePower` chainModify(2)** when the user carries a
  non-`slp` major status, `gen3_facade_v1` (id-gated in `run_move`; the gen3 burn-halve
  still applies to a burned Facade, and Guts composes). The runEvent TAIL is mirrored
  EXACTLY (battle.js:709): the accumulated chain applies iff relayVar is still
  integer-valued — `70 × 1.1 == 77` exactly in f64, so a Pink-Bow Facade RE-APPLIES the
  ×2 chain on the bow's float → BP 154 (probe `probe_facade_gen3.js`, pin FA-d)), then
  the four nested `trunc`s; (2) `modify_damage` runs gen3's `data/mods/gen3/scripts.ts`
  chain in EXACT order — burn ×0.5 FIRST (SUPPRESSED for a Guts holder via `has_guts`),
  **ModifyDamagePhase1** (screens ×0.5 crit-bypassed + **Flash Fire ×1.5 for a Fire move**
  `gen3_flashfire_boost_v1`, NOT crit-bypassed — the two ACCUMULATED into ONE `chain_modify`
  modifier, since sequential per-mod rounds diverge for ~¼ of bd), weather,
  physical-min-1, `+2`, crit ×2 (on `bd+2`), Phase2 floor, STAB ×1.5, type (raw
  `*2`/`floor(/2)`), then the randomizer SECOND-TO-LAST. All multiplies are the
  integer 4096-chain `modify` (`+2047` round); type uses raw `*2`/`floor(/2)`.
- **Differential harness** `harness/gen_damage_golden.js` drives the omniscient
  `BattleStream` (the `damage_probe.js` pattern, no server) over **63 scenarios**,
  each isolating ONE mechanic (neutral / STAB / SE / resist / 4× / type-immune /
  ability-immune [Levitate/Flash Fire/Water&Volt Absorb] / Thick Fat / Choice Band /
  type-item / Sea Incense / burn / Reflect / Light Screen / rain / sun / +Atk / +Def
  / defender +Def UNDER crit / crit-through-Light-Screen / Explosion def-halve /
  min-vs-max stats / low level — plus the 17 `gen3_item_mechanics_v1` item probes:
  the gen2 bows' DIRECT ×1.1 BP float, the 4 incenses' ×4915/4096 BP chain, the
  species stat items atk- AND def-side [3 appended columns —
  atk_species/def_species/def_item] — plus the 15 ability DMG_MOD probes: the pinch
  family at ≤⅓ HP + full-HP + wrong-type controls, Huge/Pure Power ×2 + a special
  control, Guts ×1.5 statused incl. a BURNED attacker (the burn-suppression) + a
  non-Guts-burned control, Marvel Scale ×1.5-Def vs a burned defender + an unstatused
  control [a constructed pinch-HP/status hook + 4 more appended columns —
  atk_hp/atk_maxhp/def_status/def_ability; all pre-existing lines prefix-identical]).
  It reads the sim's OWN stats + the exact HP delta,
  and **forces the MAX damage roll** (sweeps seeds, takes the max realized damage =
  the `random(16)==0` value, verified by the r==1 neighbour) so the realized damage
  == the deterministic pre-roll baseDamage — an **EXACT** equality, not a
  tolerance band. `tests/damage_test.rs` reconstructs each `DamageContext` and
  asserts `calc_damage().base == base` AND `rolls[15] == the sim's min roll`.
  (Crit scenarios sweep a large seed pool since a crit is ~1/16 per seed.)
- **Honest scope:** these 63 cover the gen3-OU-relevant single-hit modifiers; NOT
  modeled (deferred to the event engine / later) — multi-hit, fixed-damage
  (Seismic Toss / Night Shade / OHKO), Wonder Guard, spread/doubles, and the
  accuracy/secondary RNG (the engine's job — `calc_damage` takes a resolved `crit`).

## Turn: the single-turn RNG-consumption gate (per-seed STATE differential)

This is the layer where the **RNG-consumption-order crux** finally lands on a
production path — both moves resolved, draws consumed in Showdown's exact order +
count, validated end-to-end.

- **`BattleState::run_turn(p1_slot, p2_slot, &dex) → TurnResult`** (`turn.rs`)
  executes ONE turn where both sides use a DAMAGING move:
  1. **Order** the two move actions (priority → effective speed) by **wiring
     `event::speed_sort`** onto the action queue — the FIRST production path for its
     Fisher-Yates speed-tie shuffle DRAW (one `random(0,2)` on a priority+speed tie,
     zero on distinct speed). Effective speed is the gen-3 `getActionSpeed` OVERRIDE:
     the RAW boosted + ModifySpe stat (boost-table floor; paralysis ×0.5 =
     `floor(spe*50/100)`, the gen-3 value — NOT ×0.25), **NO `trunc(spe,13)`** (the
     base-sim path gen3 replaces), capped at 10000.
  2. Per move in resolved order: **accuracy** `randomChance(acc,100)` (SKIPPED iff
     `never_miss`) → **crit** `randomChance(1, critMult[critRatio])` (UNCONDITIONAL
     for a damaging, non-immune move — every gen-3 damaging move has `critRatio ≥ 1`;
     normal = `1/16`, the high-crit set = `1/8`) → **damage** `random(16)` selecting
     `calc_damage().rolls[r]` → **apply HP** (saturate at 0) → **faint at 0**.
  3. The gen-3 end-of-turn **Quick Claw `randomChance(1,5)`** — drawn ALWAYS
     (UNCONDITIONAL of Quick Claw possession), but ONLY if `endTurn()` completes,
     i.e. **no mon fainted this turn** (a faint defers it behind a switch request).
- **Three draw-COUNT subtleties** (each a desync if wrong, each pinned):
  - **IMMUNE move** (type-chart 0× or ability/Levitate immunity) — gen-3
    `tryMoveHit` resolves immunity AFTER the accuracy roll but BEFORE `getDamage`, so
    an immune move draws **only accuracy** (NO crit, NO damage). `run_move`
    short-circuits via `move_is_immune` before the crit roll. **Water/Volt Absorb**
    additionally HEAL the defender `floor(maxhp/4)` on the absorbed Water/Electric move
    (draw-free `onTryHit`, capped, no-op at full HP) via `apply_absorb_heal` at the
    short-circuit — an e2e-capstone fix (the bare-immune path missed the heal); Flash
    Fire's Fire-boost flag stays a deferred lesser gap.
  - **FAINT-SKIP** — if the first mover KOs the target, the second mover's queued
    move is cancelled (gen3 singles `cancelAction`-all) → it draws NOTHING.
  - **No Quick Claw on a faint** — a faint pauses for a switch before `endTurn`, so
    the trailing `randomChance(1,5)` is not drawn that turn.
- **Differential harness** `harness/gen_turn_golden.js` drives the omniscient
  `BattleStream` (no server) over **15 scenarios × 60 seeds**, submitting ONE
  damaging move per side. It captures the sim's PRNG state **right before** the turn
  (`SEED_BEFORE`) and **right after** (`SEED_AFTER`), plus per-mon post-turn
  hp/fainted and per-attacker crit/miss/moved + the first mover. `tests/turn_test.rs`
  SEEDS its `BattleState` prng with `SEED_BEFORE` (sidestepping the `>start` setup
  draws — gender `sample`, turn-1 Quick Claw — this bounded step omits), runs
  `run_turn`, and asserts, for the 13 DISTINCT-speed scenarios: (a) post-turn
  hp/fainted/crit/miss/moved match AND (b) the **post-turn PRNG seed equals
  `SEED_AFTER`** — an EXACT match across **780 (scenario,seed) rows** is the
  draw-ORDER+COUNT proof (a single extra/missing/mis-ordered draw shifts the LCG and
  the seed diverges on some seed). The 2 SPEED-TIE scenarios in `turn_test.rs` assert
  (a) + **who moved first** (the action-order shuffle's decision, exercised in BOTH
  directions across the sweep); FULL tie-cycle seed parity is now closed in
  `battle_test.rs` (below), which models the per-action `eachEvent` shuffles. The
  harness GUARDS the class invariant: a "distinct-speed" scenario whose actives
  silently TIE on action speed (or vice versa) fails loudly at generation.
- **Honest scope / deferred at the SINGLE-turn layer** (the `turn_test.rs` golden
  uses no-residual moves so its post-turn HP is a clean function of only the modeled
  move draws): the multi-turn layer below adds the per-action `eachEvent` shuffles +
  residuals; still deferred everywhere — secondary effects (the per-move
  `random(100)`), status MOVES, switching, recoil/drain HP, status `onBeforeMove`
  draws (para/sleep/freeze), Leech Seed / Wish / non-Leftovers items, Thick Club /
  non-folded stat events, and protocol-string emission. The golden uses Earthquake /
  Surf / Tackle / Hydro Pump / Megahorn / Crabhammer / Swift (never_miss) and bulky
  defenders.

## Multi-turn: the cross-turn RNG-consumption gate (per-seed STATE+SEED differential)

This is the closure of the RNG-order crux ACROSS turns — the full per-turn cycle
(incl. the per-action `eachEvent` shuffles + the end-of-turn residuals) looped, with
the running PRNG carried turn-to-turn, validated so the seed matches Showdown's after
EVERY turn (a single mis-ordered/missing/extra draw on turn k desyncs every turn ≥ k).

- **`BattleState::run_turn`** (extended) runs the FULL cycle — see the module table
  row + `src/turn.rs` docs for the exact 16-draw (tie) / ≤7-draw (distinct) order.
  The new pieces over the single-turn step:
  - **`each_event_shuffle`** — the `eachEvent('BeforeTurn'/'Update'/'Weather')`
    2-active `speed_sort` shuffle: draws one `random(0,2)` iff the actives tie on
    speed, zero otherwise. Wired at the exact 6 (tie) sites: BeforeTurn, end-of-each
    runAction (`gen<5` tail) ×3 (beforeTurn, residual, and each move), the in-gen3-
    `tryMoveHit` Update (only on a LANDED move), and the nested weather one.
  - **`run_residuals`** — first `update_speed()` (the `residual` action's `updateSpeed`),
    then builds the residual handler list, `speed_sort`s it on the CACHED speed (the
    handler-sort tie-shuffle, e.g. dual equal-speed Leftovers → one draw; sand present
    shifts the shuffle range to `[1,3)`), then applies each draw-free HP effect in
    the gen-3 residualOrder (sand chip order 8 → Leftovers order 10 sub 4 → status
    DoT order 10 sub 6), running **`faintMessages` (process_faints) + checkWin PER
    HANDLER** (mirroring `fieldEvent('Residual')`'s `while`-loop): a holder fainted by an
    earlier handler skips its later ones, a game-ending KO aborts the rest, a non-ending
    faint keeps ticking the other active. Values from the gen4-mod overrides gen3 INHERITS:
    burn **/8** (gen6 override), poison /8, Toxic `max(1,⌊maxhp/16⌋)·stage`, Leftovers
    ⌊maxhp/16⌋, sand `max(1,⌊maxhp/16⌋)` to non-Rock/Ground/Steel (immunity via the dex
    species types). The Toxic stage ramps on `Status::Toxic(stage)`.
  - **the deferred-faint protocol** — `apply_damage` zeroes HP only; `process_faints`
    (= `faintMessages`) sets `fainted` AFTER the in-`tryMoveHit` shuffle. So a KO turn
    fires that shuffle (the 0-HP mon is still in `getAllActive`) but NOT the trailing
    Update / second move / residual / Quick Claw — the faint-turn draw-COUNT crux
    (a faint turn = 3 pre-move shuffles + acc/crit/dmg + 1 in-tryMoveHit shuffle = 7
    in the tie case; verified).
- **`BattleState::run_battle(scripted, &dex) → Vec<TurnRecord>`** loops `run_turn`
  over the scripted `(p1_slot, p2_slot)` moves, STOPPING at the first faint (no
  switching this step). Each `TurnRecord` carries the post-turn per-side
  `MonSnapshot` (hp/maxhp/fainted/status) + the `TurnResult` + an `ended_on_faint`.
- **Differential harness** `harness/gen_battle_golden.js` drives the omniscient
  `BattleStream` (no server) over **12 scenarios × 40 seeds × several turns**:
  leftovers (no weather), sandstorm (Tyranitar Sand Stream — chip to non-immune mons),
  burn / poison / Toxic-ramp (status applied by a status MOVE on turn 1 — DEFERRED
  from the port — so recording starts at turn 2 with the status injected; the DoT is
  draw-free so cross-turn parity still holds), rain (no chip), and SPEED-TIE (snorlax,
  sand-tie, tauros-no-Leftovers). Per turn it captures `SEED_BEFORE`/`SEED_AFTER` +
  both actives' hp/maxhp/fainted/status (+ Toxic stage) + first-mover. It captures the
  pre-turn seed at the FIRST recorded turn so the Rust seeds once and carries.
- **`tests/battle_test.rs`** seeds a `BattleState` at the init seed, INJECTS the init
  status (so the turn-1 status move's effect is present), and runs `run_battle`
  WITHOUT re-seeding, asserting per turn (a) hp/maxhp/fainted/status-variant +
  first-mover AND (b) the post-turn PRNG seed == `SEED_AFTER`. It runs BOTH a
  single-seed cross-turn carry (the final-seed assertion) AND a **per-turn re-seed
  pass** that pinpoints the FIRST diverging turn and asserts EVERY turn boundary —
  ~**2034 (scenario,seed,turn) EXACT post-turn-seed assertions**, incl. the TIE class
  (full prng-state parity, closing the single-turn step's tie deferral), ~448
  status-DoT residual rows, 255 ≥4-turn chained runs, 341 faint-ending turns.
- **Residual-faint draw-order (the two BLOCKERS the review caught) —
  `tests/residual_faint_test.rs`.** The seed-sweep golden's 341 faints are all MOVE
  faints, so two residual-faint paths it can't reach are pinned by DETERMINISTIC
  regression tests instead: (1) a residual (sand chip) KO under a SPEED TIE must SKIP
  the trailing `[15]` Update shuffle + Quick Claw (else the seed desyncs every later
  turn) — `run_turn` returns on the residual faint BEFORE those draws; (2) a mon the
  weather chip KO'd is NOT revived by its own later Leftovers (`apply_leftovers`
  early-returns on `hp == 0`, like the sim's `heal`). The full seed-level parity on a
  residual-faint tie is proven by the **capstone e2e fuzz** (it occurs naturally in real
  battles) — now bit-for-bit after `run_residuals`'s per-handler `faintMessages` +
  cached-speed fix. One benign un-mirrored case: a status-DoT stage-ramp on a mon an
  earlier residual already KO'd — that mon faints + switches out (stage resets), so
  hp/seed are unaffected (and the per-handler `fainted`-skip now no-ops it explicitly).
- **Honest scope / deferred** (same as the single-turn list, minus what's now built):
  the residual MECHANICS for Leftovers / weather chip / burn / poison / Toxic + the
  per-action `eachEvent` + residual-handler tie-shuffles are MODELED; still deferred:
  secondaries, status MOVES (the harness applies status via the real sim then injects
  it), switching (a faint ends the step), Leech Seed / Wish / non-Leftovers items,
  recoil/drain, protocol bytes.

## Full battle: the to-WIN/LOSS RNG-consumption gate (per-seed PER-DECISION STATE+SEED+winner differential)

This closes the RNG-order crux across SWITCHES + post-faint replacements all the way
to game-end — a battle no longer stops at the first faint; it plays to a WIN/LOSS (or a
gen-3 tie) with the running PRNG matching Showdown's after EVERY decision boundary.

- **`BattleState::run_full_battle(&[ScriptDecision], &dex) → BattleOutcome`** (`turn.rs`)
  drives an explicit action-queue turn loop (mirroring `commitChoices`/`turnLoop`) that
  can PAUSE for a forced replacement and RESUME the saved tail. A per-side
  [`Choice`] is `Move(slot)` or `Switch(team_slot)`; a [`ScriptDecision`] is one REQUEST
  boundary's choices (a `move` request = both sides; a forced `switch` = the flagged
  side(s)). The [`BattleOutcome`] is `{ winner: Option<side>, ended, decisions }`.
- **The switch-phase draw model** (all the existing `event::speed_sort` / `random_range`,
  verified by `harness/trace_switch_rng.js` + the adversarial corrections):
  - **action-order shuffle** — `sort_actions` (`speed_sort` in `commitChoices`): a
    switch (order 103) sorts before a move (200) by `order`; two same-kind switches at
    equal outgoing-mon speed tie → the speed-tie shuffle draws. A forced replacement's
    instaswitch (order 3) sorts first; a DOUBLE replacement's two instaswitches tie →
    that shuffle draws.
  - **gen-3 runSwitch is DRAW-FREE** — the gen-4 `runSwitch` override (inherited) does
    NOT `speedSort(getAllActive())` or `fieldEvent('SwitchIn')`; only the entrant's
    ability `Start` fires (draw-free for our abilities, via `single_event_ability_start`).
    **Do NOT add a SwitchIn tie-shuffle** (the adversarial correction's #1 desync trap).
  - **double-replacement `insertChoice` splice** — `insert_runswitch` mirrors
    `BattleQueue.insertChoice`: the 2nd instaswitch's `switchIn` enqueues its order-101
    `runSwitch` into a queue already holding the 1st's (same order 101) → a `random(fi,
    li+1)` splice draw. A SINGLE replacement's runSwitch inserts with no tie window → no
    draw.
  - **per-action trailing `eachEvent('Update')`** at the END of each runAction — SKIPPED
    when the runAction paused (`makeRequest('switch'); return`) OR its next queued action
    is an `instaswitch` (`battle.ts:2372`'s `else if … return false`, BEFORE the gate).
    A fainted/inactive actor's no-op move `return false`s at the `case 'move'` guard →
    NO tail, NO trailing Update (the double-replacement's surviving no-op move draws
    nothing).
  - **Quick Claw** `randomChance(1,5)` at `endTurn` — UNLESS the turn ended on a faint
    pause or a game-ending faint (no trailing Quick Claw on the deciding turn).
- **The `switchIn` POSITION SWAP** (`execute_switch`): the entrant moves to the active
  index and the outgoing mon takes the entrant's old bench slot (battle-actions.ts:131-133),
  so `Switch(N)` keeps referring to the CURRENT array slot (like the sim's `switch N`).
  Queued move actions are keyed by a stable **`MonState::uid`** (the construction-time
  index, immutable across swaps), resolved to the current slot at execution — the gen3-
  singles equivalent of dereferencing `action.pokemon`.
- **Explosion/Selfdestruct self-KO** — `useMoveInner` (`gen ≠ 4`) faints the user BEFORE
  the hit (draw-free HP→0); `process_faints` then sets the flag for BOTH user + target,
  so a mutual Explosion is a true double-faint.
- **Win/loss** — `check_win` runs inside the faint protocol: a side with `pokemon_left ==
  0` loses, its foe wins; both 0 → a gen-3 TIE (`winner == None`, `ended == true`).
- **Differential harness** `harness/gen_fullbattle_golden.js` drives the omniscient
  `BattleStream` (no server) over **8 scenarios × 50 seeds** to game-end, reading
  `requestState` (move vs forceSwitch) + the `forceSwitch` table per decision and
  submitting the scripted `move`/`switch` choice (recorded as a compact token so the
  Rust replays the EXACT sequence — duplicate-species safe). It captures `initSeed`
  (the pre-first-decision seed the Rust seeds once) + per decision the `seedAfter`, both
  actives' species/hp/maxhp/fainted/status + side `pokemonLeft`, the first mover, and the
  final `|win|`/winner. **`tests/fullbattle_test.rs`** seeds at `initSeed`, runs
  `run_full_battle` WITHOUT re-seeding, and asserts per DECISION BOUNDARY (each `move`
  turn AND each forced-`switch` sub-step): (a) active species / hp / maxhp / fainted /
  status + pokemon_left + request kind + first mover, AND (b) the post-decision PRNG seed
  == the sim's `seedAfter` — ~**2053 per-decision EXACT seed assertions** to game-end —
  PLUS the final winner (or tie). The single-seed carry IS the per-decision pass (each
  intermediate boundary is asserted, so the first diverging decision panics with its
  index). Scenarios: both-switch distinct/tie, switch-vs-move, post-faint single + double
  replace, KO-to-win, last-mon double-KO TIE (350 wins + 50 ties; 50 double-replacements;
  350 past-faint runs). **All moves are secondary-free** (a secondary draws an unmodeled
  `random(100)` that would desync) per the adversarial correction.
- **Honest scope / deferred**: the switch ordering + switch-in ability `Start` + post-faint
  pause/resume + double-replacement draws + win/loss are MODELED; still deferred: secondary
  effects, status MOVES, entry hazards (Spikes — so `EntryHazard`/`SwitchIn` stay draw-free),
  Pursuit (the switch-trap `pursuitfaint`), Baton Pass (`copyVolatileFrom`), the
  `BeforeSwitchOut`-gate draws (Eject Button etc.), and protocol bytes. (**NaturalCure is now
  MODELED** — `gen3_natural_cure_v1`; the deferred "NaturalCure `CheckShow`" draw question is
  RESOLVED: `naturalcure` has `onCheckShow: undefined`, the cure fires `onSwitchOut` and is
  **draw-free** — see the "## E2E capstone" modeled-ability list + EDGE_CASES.)
- **Coverage honesty (review nit, left to the capstone):** the per-decision record asserts
  active species/hp/maxhp/fainted/status + pokemon_left + seed + winner, but NOT `field.weather`
  or `boosts`. So a mid-battle switch-in ability's **draw behaviour** (run_switch is draw-free)
  IS seed-verified, but its observable **effect** (a replacement Sand-Stream setting weather, an
  Intimidate dropping the foe's Atk) is not differentially asserted here, and a double-replacement's
  `insertChoice`-splice **order** is unobservable (identical entrants). Both reuse the
  switchin-test-validated `single_event_ability_start`; their full state-level coverage comes from
  the **e2e fuzz capstone** (real teams compare the WHOLE battle, weather/boosts included).

## Secondary effects + onBeforeMove status: the per-move-draw-bracket gate (per-seed PER-DECISION STATE+STATUS+SEED differential)

This closes the two NEW per-move draw sites — the leading `onBeforeMove` status draw and the
trailing secondary `random(100)` — so a battle where status is inflicted IN-ENGINE by a real
secondary move (and then fires its onBeforeMove draws on later turns) stays bit-for-bit seed-faithful
to game-end, with the per-decision STATUS asserted too.

- **`run_move` brackets the existing acc→crit→dmg with two new draw sites** (`turn.rs`):
  - **`on_before_move`** — the NEW LEADING draw, BEFORE accuracy (mirroring `runEvent('BeforeMove')`
    at `battle-actions.ts:255`, which precedes useMove/PP/accuracy). Handlers in
    `onBeforeMovePriority`-DESC order with **break-on-first-abort** (`battle.ts:912-920`): **sleep**
    (10, DRAW-FREE counter decrement; wake at 0 → cure + proceed, else abort), **freeze** (10,
    `randomChance(1,5)` thaw → cure+proceed, else abort — UNLESS the move is a gen3
    `flags.defrost` carrier (Sacred Fire / Flame Wheel, id-gated `is_defrost_move`): the roll
    still DRAWS, but on a failed roll the move PROCEEDS and the user thaws draw-free via
    `frz.onModifyMove` (`|-curestatus|…|frz|[from] move: <Move>` BEFORE the `|move|` line) —
    `gen3_defrost_v1`, probe `harness/probe_sacredfire_defrost.js`, pin
    `regression_test.rs::frozen_defrost_move_bypasses_the_cant_and_thaws`), **flinch** (8, DRAW-FREE; present → abort),
    **confusion** (3, decrement DRAW-FREE; at 0 remove+proceed, else `randomChance(1,2)` — pass →
    proceed, else a typeless-40-BP self-hit one `random(16)` via `calc_damage` with `crit:false`,
    then abort), **paralysis** (1, `randomChance(1,4)` full-para → abort). An abort returns with NO
    acc/crit/dmg/secondary (like a miss/immune). At most one major status (slp/par/frz exclusive)
    can be present, so the priority-10 slp/frz tie never fires concurrently.
  - **`apply_secondaries`** — the NEW TRAILING `random(100)`, one per surviving secondary AFTER the
    hit lands + HP applied (mirroring `secondaries()` at `battle-actions.ts:1357-1373`, step 5 of
    `spreadMoveHit`). Applied if `roll < chance` via `try_set_status` / the flinch volatile / the
    **CONFUSION** + **stat-boost** arms (below). The step moves (Body Slam par30 / Ice Beam frz10 /
    Thunderbolt par10 / Rock Slide flinch30 / Sludge Bomb psn30) each = ONE `random(100)`. **Serene
    Grace** ×2 the THRESHOLD (NOT the draw — `onModifyMove` pre-doubles before the hit); **Shield
    Dust** on the DEFENDER FILTERS the foe-targeting secondary out → ZERO `random(100)` (a draw-COUNT
    effect). A DAMAGE-immune target short-circuits BEFORE the secondary (no draw —
    `tbolt_ground_immune_no_secondary`); a status-immune-but-DAMAGED target STILL draws it but no-ops
    the apply (`sludgebomb_psn_immune_damaged`).
- **The CONFUSION secondary** (`add_confusion`, the secondary-completion step) — a landed confusion
  secondary (Water Pulse 20% etc.) draws the secondary `random(100)` (the caller) THEN, on a
  SUCCESSFUL `addVolatile`, ONE EXTRA `random(2,6)` duration (the `onStart`, min=2 → 2..5 turns) into
  the `confusion: Option<u8>` counter — the DRAW that was MISSING (a SEED desync on every landed
  confusion). The gates (each draws the secondary `random(100)` but NOT the `random(2,6)`):
  **ALREADY-CONFUSED** (`addVolatile` returns false before onStart — confusion has no onRestart) and
  **OWN TEMPO** (`owntempo.onTryAddVolatile` returns null). The downstream onBeforeMove self-hit
  (`randomChance(1,2)` + a `random(16)`) was already modeled. **Substitute** (a moveHit-level block)
  is N/A — not modeled, and only reached on a landed damaging hit.
- **The STRUCTURED stat-boost secondary** (`apply_secondary_boost`, DRAW-FREE) — the foe stat-DROP /
  self stat-RAISE the flat `secondaryEffects` `{col:percent}` loses (Crunch −1 SpD, Psychic/Shadow
  Ball −1 SpD, Iron Tail −1 Def, Rock Tomb/Icy Wind −1 Spe, Muddy Water −1 acc; Meteor Mash +1 Atk
  SELF, Ancient Power/Silver Wind +1 ALL, Metal Claw/Steel Wing +Atk/+Def). The lost `(stat, stages,
  target=foe|self)` is carried by the additive **`secondaryBoosts`** dex field (`[{chance, target,
  boosts}]`, only-when-present like `critRatio` — the extractor builds it from `secondary.boosts` /
  `secondary.self.boosts`; the obs facade ignores it). The apply routes to the foe (drop) or the user
  (raise), clamped to ±6 (`boost()` consumes NO PRNG → STATE-only, never a seed desync), gated by the
  foe boost-immunity abilities **Clear Body / White Smoke** (all), **Hyper Cutter** (atk), **Keen
  Eye** (accuracy).
- **The fail-loud multi-secondary GUARD + Tri Attack special-case** (`apply_secondaries` /
  `apply_triattack_secondary`) — a move with >1 secondary col in the flat data is GIGO for draw-order
  (the 3-col flatten would mis-draw 3 `random(100)`s). The ONLY such gen-3 move is **Tri Attack**
  (`triattack`, flattened to `{par:7,brn:7,frz:6}`), SPECIAL-CASED to its true model: ONE `random(100)`
  (the 20% gate) then ON LAND ONE `random(3)` `sample(['brn','par','frz'])` → `try_set_status`. Any
  OTHER >1-col move PANICS (a future shape can never silently desync).
- **The onTrySetStatus gates** (`try_set_status`, all DRAW-FREE): already-statused → fail; the GEN-3
  type immunity (`status_type_immune`) — **frz→Ice, brn→Fire, psn/tox→Poison&Steel; NO Electric→para
  immunity (that was added in Gen 6 — Electric/Steel CAN be paralyzed in gen3, VERIFIED vs the live
  sim)**, slp has none; a KO'd (hp==0) target no-ops. No in-scope secondary inflicts slp/tox, so the
  counter-set draws (`random(2,6)`) are unreachable here (Toxic begins at stage 1).
- **Two new `MonState` volatiles** (`state.rs`): `confusion: Option<u8>` (counter, set when added by a
  confusion secondary's `random(2,6)`, decremented draw-free, removed at 0) and `flinch: bool`
  (`duration:1` — cleared at the TOP of each turn via `clear_flinch`, and on switch-out via
  `execute_switch`'s clearVolatile). Both — plus the 7-stage `boosts` array — are now carried on the
  per-decision `MonSnapshot` so the differential asserts the boost STAGE and confusion STATE, not just
  the seed.
- **The gen-3 paralysis-speed FIX** (`effective_speed`): gen-3 inherits gen4's `par.onModifySpe`
  `chainModify(0.25)`, applied via `runEvent('ModifySpe')` as `modify(spe, 1, 4)` =
  `floor((spe·1024 + 2047)/4096)` — the `+2047`-rounded form (NOT a plain `floor(spe/4)`: raw 359 →
  90 via modify vs 89 via floor), and **NOT ×0.5** (the base-sim value gen3 replaces; the prior code
  was wrong). VERIFIED vs the sim's `getStat('spe')` (350→87, 359→90, 206→51, 14→3). This re-orders a
  paralyzed mon's turn (the surfaced bug: a paralyzed fast mon must drop below a slower foe).
- **The fire-move thaw** (`run_move` tail): a Fire damaging move cures the DEFENDER's freeze
  (gen3 `frz.onDamagingHit`, draw-free, after secondaries).
- **Differential harness** `harness/gen_secondary_golden.js` drives the omniscient `BattleStream`
  (no server) over **12 scenarios × 80 seeds** to game-end with REAL secondary moves so status is
  inflicted IN-ENGINE and the onBeforeMove draws fire on later turns: `bodyslam_para_loop`
  (self-contained para-then-full-para), `icebeam_freeze_thaw`, `tbolt_para_lands` (gen3 Electric IS
  para-able), `sludgebomb_psn` + `sludgebomb_psn_immune_damaged` (psn no-ops on a Poison target but
  the random(100) STILL draws), `tbolt_ground_immune_no_secondary` (immune → NO secondary draw),
  `rockslide_flinch`, `mixed_secondary_switch_grind` (secondaries across a switch + replacement), and
  the secondary-completion scenarios — `crunch_spd_drop` (foe −1 SpD, gen3 override), `psychic_
  shadowball_spd_drop` (foe −1 SpD at 10/20%), `meteormash_self_atk` (+1 Atk SELF), `waterpulse_
  confusion` (the confusion `random(2,6)` + the onBeforeMove confusion loop). It captures `initSeed`
  + per decision the `seedAfter`, both actives' species/hp/maxhp/fainted/STATUS + **the 5 stat-stage
  boosts + the confusion counter** + pokemon_left + first mover + the RNG OUTCOMES, and the winner —
  with floors (≥30 boosted, ≥10 self-boost, ≥20 confused rows) so every new branch realizes. The
  `firstMoverSince` counts `|cant|` + the confusion `-activate` (a cancelled mon still RAN its action
  first), matching the Rust action-order `first_mover`. **`tests/secondary_test.rs`** seeds at
  `initSeed`, runs `run_full_battle` WITHOUT re-seeding, and asserts per DECISION BOUNDARY: (a)
  species/hp/maxhp/fainted/**STATUS** + **BOOST stages** + **CONFUSION counter** + pokemon_left +
  request kind + first mover; AND (b) the post-decision seed == the sim's `seedAfter` — **~4328
  per-decision EXACT seed assertions + ~7457 status-variant + ~7457 boost-stage + ~7457 confusion-
  counter assertions** to game-end — PLUS the winner. A STATUS/BOOST/CONFUSION mismatch catches a
  secondary that wrongly applied/skipped (boost() is draw-free, so a wrong stat/stage/target diverges
  the STATE, not the seed); a SEED mismatch catches a mis-ordered/missing/extra draw (e.g. the
  confusion `random(2,6)` or a mis-flattened Tri Attack 3-draw).
- **Deterministic unit gates** (`turn.rs` tests): `paralysis_full_para_draws_only_the_para_roll`
  (1-draw abort), `paralysis_pass_then_full_move_plus_secondary` (5-draw order), `landed_secondary_
  draws_one_random_100_after_damage`, `gen3_status_type_immunity_rules` (incl. NO para immunity),
  `try_set_status_gates_already_statused_and_type_immune`, `shield_dust_defender_suppresses_the_
  secondary_draw` (3-draw, no secondary), `flinch_aborts_draw_free`, `sleep_counter_is_draw_free_and_
  wakes`, `confusion_self_hit_draws_chance_plus_one_random16` (2-draw, no crit), `freeze_draws_one_
  thaw_roll`, and the secondary-completion gates: `confusion_secondary_draws_random_100_then_random_
  2_6` (the LANDED-confusion 5-draw sequence), `confusion_secondary_already_confused_skips_the_
  duration_draw` + `confusion_secondary_own_tempo_skips_the_duration_draw` (the gate draws the
  `random(100)` but NOT `random(2,6)`), `tri_attack_draws_random_100_then_sample_random_3` (the
  `random(100)`+`random(3)` sequence + the sampled status), `unmodeled_multi_secondary_panics` (the
  fail-loud >1-col guard), `stat_drop_and_self_boost_apply_the_structured_spec`, and
  `clear_body_blocks_the_foe_stat_drop`.
- **Honest scope / deferred** (this section): the **top-level `move.self.boosts`
  `selfDrops` draw** (Overheat/Superpower/Psycho Boost draw a SEPARATE `random(100)` via `selfDrops`,
  NOT carried by `secondaryBoosts` which walks only `secondary`/`secondaries[i]` — a missing-draw
  desync if used, deferred), Substitute (its confusion-block is a moveHit-level gate, not modeled),
  and protocol bytes. **Now BUILT (this step's additions):** the CONFUSION secondary's `random(2,6)`
  duration draw, the stat-drop / self-boost apply (via the structured `secondaryBoosts` dex field),
  and the Tri Attack `random(100)`+`sample(3)` special-case + the fail-loud >1-col guard. (Standalone
  status MOVES are now their OWN layer — next section.)

## Status moves: the standalone-status-MOVE draw gate (per-seed PER-DECISION STATE+STATUS+SEED differential)

This closes the standalone STATUS-INFLICTING moves (category Status, bp 0): par (Thunder Wave / Stun
Spore / Glare), psn (Poison Powder / Poison Gas), tox (Toxic), brn (Will-O-Wisp), slp (Spore / Sleep
Powder / Hypnosis / Sing / Lovely Kiss / Grass Whistle). Their execution path + draw model are
bit-for-bit faithful to game-end, sustained through battles where status is inflicted IN-ENGINE by a
real status move (and then the onBeforeMove sleep-wake / para-full / brn-tox-DoT fire on later turns).

- **`run_move` routes a category-Status move to `run_status_move`** (`turn.rs`), the gen-3
  `data/mods/gen3/scripts.ts::tryMoveHit` status path — VERIFIED bit-for-bit vs the omniscient sim:
  1. **MOVE-TYPE IMMUNITY** (`runImmunity`, DRAW-FREE) — a Status move defaults `ignoreImmunity =
     true` (type immunity IGNORED); the ONLY two gen-3 status moves that set `ignoreImmunity: false`
     are **Thunder Wave** (Electric → a GROUND target immune) + **Glare** (Normal → a GHOST target
     immune). Every other status move ignores type immunity (its status-type / ability immunity lives
     in `try_set_status`).
  2. **ACCURACY** `random_chance(accuracy, 100)` — ALWAYS drawn (unless `never_miss`), EVEN on a
     type-immune target (gen3 draws accuracy THEN reports `-immune`; the gen-3 dex accuracies — TWave
     100 / Spore 100 / Toxic 85 / Will-O-Wisp 75 / Stun Spore 75 / Glare 75 / Sleep Powder 75 / Poison
     Powder 75 / Hypnosis 60 / Sing 55 / Grass Whistle 55 / Poison Gas 55 / Lovely Kiss 75).
  3. **APPLY** via `try_set_status` (the onTrySetStatus gates — already-statused / gen-3 status-type
     immunity [brn→Fire, frz→Ice, psn/tox→Poison&Steel] / ability immunity [Insomnia/Vital Spirit slp,
     Limber par, Immunity psn/tox, Water Veil brn, Magma Armor frz] / the **Sleep Clause Mod**), then
     the status onStart: **SLEEP** draws ONE `random(2,6)` duration (1-4 turns, stored in
     `Status::Sleep(n)`, decremented per move in `on_before_move` — **Early Bird** double-decrements);
     **TOXIC** starts at **stage 0** (the residual ramps it; `Status::Toxic(stage)` now mirrors the
     sim's `statusState.stage` EXACTLY — a one-off representation fix from the old `Toxic(1)`). NO
     crit, NO damage, NO secondary; `landed` is ALWAYS FALSE (a landed status `moveHit` returns
     `undefined` → the in-`tryMoveHit` `eachEvent('Update')` shuffle, `scripts.ts:470`, is skipped).
- **The gen3ou `runEvent('SetStatus')` HANDLER-SORT SHUFFLE** (`set_status_event_shuffle`, the
  gen3ou-only draw) — `setStatus` calls `runEvent('SetStatus')`, which gathers the 2 `Standard`
  format-clause handlers (**Sleep Clause Mod + Freeze Clause Mod**) at equal order/priority/speed → a
  TIE → a size-2 Fisher-Yates speed-sort shuffle draws EXACTLY one `random(0,2)` EVERY time the event
  is reached (a status that passed hp / already-statused / type-immunity — INCLUDING one the clause or
  an ability then BLOCKS, so a Sleep-Clause-blocked sleep draws the shuffle but NO `random(2,6)`). In
  **gen3customgame** (no clauses → the ONLY handler is a STATUS_IMMUNE ability's own, size-1 → NO tie) NO
  shuffle is drawn. Gated by `BattleState::sleep_clause`, derived from the format id
  (`state::format_has_sleep_clause` — true for non-`*customgame`). **A STATUS_IMMUNE target ABILITY with its
  own `onSetStatus` (Insomnia/Limber/Immunity/Water Veil/Vital Spirit — `gen3_status_immune_v1`) is now
  MODELED, NOT a fail-loud** (the old "size-3 shuffle" panic was WRONG): probe-settled
  (`harness/probe_statusimmune_shuffle_size.js`) the ability handler carries a DEFINED `speed` while the 2
  clause handlers have `speed=undefined`, so `speedSort` puts the ability in its OWN group (index 0) and the
  2 clauses STAY a SIZE-2 tie → `shuffle(list,1,3)` draws EXACTLY ONE `random` (identical to the control's
  `shuffle(list,0,2)`), so `set_status_event_shuffle`'s one draw is UNCHANGED and the ability's block is a
  DRAW-FREE handler return AFTER the shuffle. **Magma Armor** (frz) blocks via `onImmunity` at
  `runStatusImmunity` BEFORE the SetStatus event, so its clause shuffle NEVER fires (the sun-freeze gate's
  position). The fail-loud (`ability_unmodeled_on_set_status`) now fires ONLY on a genuinely UNMODELED
  `onSetStatus` ability (none exists in gen-3 beyond the 5 STATUS_IMMUNE members).
- **The SLEEP CLAUSE MOD** (`side_has_sleeper`, gen3ou-only) — a sleep move FAILS if any LIVING mon on
  the TARGET's side already has a (non-self-inflicted) sleep. All sleep this engine inflicts is
  foe-sourced (Rest is out of scope), so an asleep mon always counts. The block is at the SetStatus
  event (the shuffle drew) → NO `random(2,6)`.
- **FAIL-LOUD**: `run_status_move` PANICS on any status move NOT in the modeled set
  (`modeled_status_move`) — mirroring the >1-secondary guard, so a future recovery/boost/phaze/hazard/
  Substitute/field move can never silently desync.
- **Differential harness** `harness/gen_status_move_golden.js` drives the omniscient `BattleStream`
  (no server) in **gen3ou** (so the Sleep Clause Mod + the SetStatus shuffle are ACTIVE) over **10
  scenarios × 80 seeds** to game-end with CONSTRUCTED scenarios that each ISOLATE a draw/branch:
  Thunder Wave lands (par) + a full-para loop, Thunder Wave → GROUND immune (accuracy-only, no par),
  Toxic ramp (the stage climbs past 2), Toxic → Steel/Poison immune, Will-O-Wisp lands + a 75-acc
  MISS, Spore → sleep + the `random(2,6)` counter + the onBeforeMove WAKE, Sleep Powder miss, Stun
  Spore para, a **SLEEP CLAUSE block** (2nd sleep fails), and status-move-into-a-real-battle (a status
  open + a voluntary pivot + a grind to a win). It captures `initSeed` + per decision the `seedAfter`,
  both actives' species/hp/maxhp/fainted/STATUS(+the Sleep/Toxic inner counter via `statusState.time`/
  `.stage`)/boosts/confusion + pokemon_left + first mover + winner. The harness FAILS LOUD at
  generation if a scenario does not realize its declared branch (`require`/`forbid`), and a STALL
  guard (a rejected choice that doesn't advance the seed) aborts rather than emit a poisoned golden.
- **`tests/status_move_test.rs`** seeds a `BattleState` at `initSeed` (format `gen3ou` → `sleep_clause`
  ON), runs `run_full_battle` WITHOUT re-seeding, and asserts per DECISION BOUNDARY: (a) species / hp /
  maxhp / fainted / **STATUS** + the **sleep/Toxic inner counter** + boosts + confusion + pokemon_left
  + request kind + first mover; AND (b) the post-decision seed == the sim's `seedAfter` — the EXACT
  cross-decision draw-order+count proof to game-end (the accuracy draw + the sleep `random(2,6)` + the
  gen3ou SetStatus shuffle must each be in the exact place/count) — PLUS the winner. Coverage floors
  pin every branch (par/tox/brn/slp landed, immune, miss, Sleep-Clause, full-para, wake, the Toxic
  stage≥2 ramp). Deterministic unit gates in `turn.rs` pin the bare draw model in gen3customgame
  (TWave = 1 accuracy draw; TWave→Ground = accuracy-only immune; Spore = accuracy + `random(2,6)`),
  the SetStatus shuffle being gen3ou-ONLY, the Sleep-Clause block (shuffle + no `random(2,6)`), the
  Early-Bird double-decrement, and the fail-loud panics (unmodeled status move; Insomnia target under
  gen3ou).
- **Honest scope / deferred**: recovery/boost/phaze/hazard/Substitute/field status moves, entry
  hazards, Pursuit/Baton Pass/Leech Seed/Wish, and the `selfDrops` draw (same as the prior section's
  deferred). Self-target status (Rest) is out of scope — so the Sleep-Clause self-exemption never
  matters. (SELF-BOOST setup moves are now their OWN layer — next section.)

## Setup moves: the self-targeting STAT-BOOST draw gate (per-seed PER-DECISION STATE+BOOST-STAGE+SEED+first-mover differential)

This closes the SELF-TARGETING SETUP / STAT-BOOST moves (category Status, bp 0, target self) whose
ENTIRE effect is raising the USER's stat stages — proven bit-for-bit to game-end, with the per-decision
BOOST STAGES + the +Speed first-mover FLIP asserted.

- **The modeled set** (the 17 PURE self-boost moves): Calm Mind (+1 SpA/+1 SpD), Dragon Dance (+1 Atk/
  +1 Spe), Swords Dance (+2 Atk), Agility (+2 Spe), Bulk Up (+1 Atk/+1 Def), Amnesia (+2 SpD), Barrier/
  Acid Armor/Iron Defense (+2 Def), Cosmic Power (+1 Def/+1 SpD), Tail Glow (+2 SpA), Meditate/Sharpen/
  Howl (+1 Atk), Harden/Withdraw (+1 Def), Growth (+1 SpA). They are DATA-DRIVEN: the extractor
  (`tools/pokemon_data_extractor/sync.py` `_self_boosts`) emits a `selfBoosts` `{stat:stages}` map
  (only-when-present, obs-neutral — the facade ignores it, like `secondaryBoosts`/`critRatio`) for a
  `target:self` Status move whose declarative top-level `boosts` are all POSITIVE battle stats (no
  accuracy/evasion) AND that carries NO other effect; the Rust dex parses it into `MoveData::self_boosts`,
  and `run_status_move`'s self-boost branch (via `self_boost_spec`) applies it. **EXCLUDED, fail-loud**:
  Defense Curl / Minimize (a `volatileStatus`), Double Team / Minimize (+evasion — the engine's accuracy
  roll ignores the evasion table, so a +evasion state would silently desync the next move's accuracy
  against it), Belly Drum (HP-cost `onHit`, no declarative `boosts`), Curse (type-conditional onHit).
- **The draw model** (verified bit-for-bit vs `data/mods/gen3/scripts.ts::tryMoveHit` + `this.boost`):
  1. **ACCURACY** — every modeled setup move is NEVER-MISS (`accuracy:true`) → NO accuracy draw
     (handled generally: `random_chance(acc,100)` iff NOT `never_miss`; the set is all never-miss).
  2. **APPLY** `boost()` on the USER, each (stat, stages) clamped to ±6. **DRAW-FREE** — `boost()`
     consumes no PRNG (like `apply_secondary_boost`). Our OWN Clear Body / White Smoke etc. do NOT
     block our OWN self-boost (the `onTryBoost` immunity is FOE-drop-only). A boost into the +6 cap is
     a no-op-but-success that STILL draws nothing.
  3. **landed = FALSE** — a status `moveHit` returns `undefined` → the in-`tryMoveHit` `eachEvent('Update')`
     shuffle is SKIPPED. So a pure self-boost move is DRAW-FREE beyond the existing action-order/eachEvent
     shuffles — the seed is unchanged by the boost itself; only the user's boost STATE changes.
- **The +SPEED cached-speed crux** (the real validation target) — a Dragon Dance / Agility raises
  `boosts[4]` IMMEDIATELY but does NOT touch `MonState::cached_speed`. Showdown re-establishes the cached
  `pokemon.speed` only at the next re-cache site (turn-start `commitChoices`, residual `updateSpeed`,
  switch-in), so THIS turn's eachEvent tie-shuffles read the PRE-boost cached speed, and the NEXT turn's
  action order picks up the boosted speed at turn-start. The self-boost path deliberately does NOT call
  `update_speed()` (matching the stale-between-sites model the residual fix established) — so a +Spe
  setup FLIPS the first-mover on a FOLLOWING turn, bit-exact in the seed (the cached-speed timing + the
  Fisher-Yates tie-shuffle draw COUNT). A wrong model → a divergent first-mover AND/OR a seed desync.
- **Differential harness** `harness/gen_setup_move_golden.js` drives the omniscient `BattleStream`
  (no server, **gen3customgame**) over **6 scenarios × 80 seeds** to game-end with CONSTRUCTED scenarios
  that each ISOLATE a branch: Calm Mind climb to the +6 cap (multi-turn), Swords Dance +2 Atk (Flygon/
  Levitate sweeps a Ground foe it's immune to), **Dragon Dance → a first-mover FLIP** (a slower Salamence
  overtakes a faster Starmie at +2, the cached-speed timing bit-exact), **Agility → a first-mover FLIP**
  (Metagross overtakes Heracross at +2 the NEXT turn), a Swords Dance into the +6 CAP (no-op success), and
  setup-into-a-real-battle-to-game-end (Calm Mind + a voluntary pivot + Bulk Up + a grind to a win). It
  captures `initSeed` + per decision the `seedAfter`, both actives' species/hp/maxhp/fainted/status + **THE
  5 BOOST STAGES** + confusion + pokemon_left + first mover + a `boosted`/`capped`/`firstMoverFlip` flag +
  winner, with fail-loud `require` floors (a boost applied / the +6 cap / a first-mover flip) so every
  branch realizes, and a STALL guard.
- **`tests/setup_move_test.rs`** seeds a `BattleState` at `initSeed` (gen3customgame), runs
  `run_full_battle` WITHOUT re-seeding, and asserts per DECISION BOUNDARY: (a) species/hp/maxhp/fainted/
  status + **the 5 BOOST STAGES** + confusion + pokemon_left + request kind + **first mover**; AND (b) the
  post-decision seed == the sim's `seedAfter` — the EXACT cross-decision draw-order+count + CACHED-SPEED
  proof to game-end — PLUS the winner. **Result: 480 runs, 2667 per-decision seed assertions + 4736
  boost-array assertions (2637 boosted, 480 at the +6 cap) + 2549 first-mover assertions, 480 wins.** A
  BOOST mismatch catches a self-boost that mis-applied (boost() is draw-free → a wrong stat/stage/cap
  diverges the STATE, not the seed); a FIRST-MOVER or SEED mismatch on a post-Dragon-Dance turn catches a
  wrong cached-speed model. Deterministic unit gates in `turn.rs` pin the bare model (Calm Mind +1/+1
  draw-free; the +6 cap no-op; Dragon Dance boosts `boosts[4]` but leaves `cached_speed` stale until the
  next `update_speed`) + the fail-loud unmodeled-status-move panic (now keyed on Recover, since Calm Mind
  is modeled).
- **Honest scope / deferred**: the EXCLUDED setup moves above (Defense Curl/Minimize/Double Team/Belly
  Drum/Curse), plus everything the status-move section defers. The setup-move e2e expansion ALSO surfaced
  + FIXED one engine bug — **Water/Volt Absorb is now accuracy-gated** (the absorb heal is an `onTryHit`
  that fires only on a HIT, so a MISSED Water/Electric move no longer heals the holder; see the `turn.rs`
  row). (SELF-HEAL recovery moves are now their OWN layer — next section.)

## Recovery moves: the self-heal / Rest draw gate (per-seed PER-DECISION STATE+HP+STATUS+SEED differential)

This closes the SELF-TARGETING HP-RECOVERY moves (category Status, bp 0, target self, `isHeal`) — proven
bit-for-bit to game-end, with the per-decision HP (the heal amount) + the Rest sleep/cure asserted.

- **The modeled set** (routed in `run_status_move`'s recovery branch + `run_rest`, BEFORE the fail-loud):
  the flat-half recovers **Recover / Soft-Boiled / Slack Off / Milk Drink** (heal `floor(maxhp/2)`); the
  WEATHER-conditional **Moonlight / Synthesis / Morning Sun**; and **Rest** (full heal + self-sleep + cure).
  `splash` is also modeled (a true draw-free no-op — the golden's "do nothing" filler). The set is an
  explicit id list (`recovery_heal_amount` + the `rest`/`splash` arms) kept in lockstep with the e2e
  `MODELED_RECOVERY_MOVES`.
- **The heal amounts** (gen3 `maxhp == baseMaxhp`, all INTEGER truncation — verified bit-for-bit vs the
  omniscient sim): Recover & friends `floor(maxhp/2)` (the `move.heal:[1,2]` path, `Math.floor(baseMaxhp*1/2)`);
  Moonlight/Synthesis/Morning Sun the **gen4-inherited PLAIN-integer `onHit`** (NOT the 4096-`modify`):
  NONE → `floor(maxhp/2)`, SUN → `floor(maxhp*2/3)`, SAND/RAIN/HAIL → `floor(maxhp/4)`. (VERIFIED the
  PLAIN-integer form vs `modify`: Espeon maxhp 271 in sun heals `floor(271*2/3)=180`, not `modify(271,0.667)=181`.)
- **The draw model**: (1) ACCURACY — every recovery move is NEVER-MISS → NO accuracy draw; (2) HEAL
  (`apply_heal`, mirroring `apply_leftovers`) is DRAW-FREE — `this.heal` consumes no PRNG; a heal at FULL HP
  / heal-0 FAILS (`heal` returns false → `-fail`), draw-free either way; (3) `landed` ALWAYS FALSE (the
  in-`tryMoveHit` Update is skipped).
- **REST — the draw-COUNT crux (`run_rest`), got bit-for-bit RIGHT against the sim's PRNG probe:** Rest's
  `onHit` calls `target.setStatus('slp')`, which runs the gen-3 `slp.onStart` — and `slp.onStart` ALWAYS
  draws `random(2,6)` for the duration. So **Rest DOES consume one `random(2,6)`** (the value is then
  DISCARDED), and Rest OVERWRITES the time to a FIXED `Sleep(3)` (the stored counter is 3, not the rolled
  1-4). This is the opposite of "draw-free" — a per-PRNG probe shows the Rest turn draws exactly one MORE
  `random(2,6)` than a no-sleep turn. The user wakes via the EXISTING `on_before_move` sleep counter
  (3→cant→cant→wake+move). Rest CURES any prior major status (the `setStatus` override). In gen3ou Rest's
  `setStatus('slp')` ALSO draws the **SetStatus handler-sort shuffle** BEFORE the `random(2,6)` (`run_rest`
  does shuffle→`random(2,6)`, gated by `sleep_clause`); in gen3customgame (the recovery golden's format) it
  draws only the `random(2,6)`. A self-Rest sleep is EXEMPT from the Sleep Clause CAP (it never blocks).
  The FULL-HP onTry guard fails Rest (no sleep, no heal, draw-free) if the user is at max HP.
- **Differential harness** `harness/gen_recovery_move_golden.js` drives the omniscient `BattleStream`
  (no server, **gen3customgame**) over **8 scenarios × 80 seeds** to game-end with CONSTRUCTED scenarios
  that each ISOLATE a branch: Recover from low HP, Recover at FULL HP (the fail/no-op), Rest from low HP
  (full heal + sleep + the wake N turns later + the `random(2,6)`), Rest CURING a prior para (para→slp
  flip), Moonlight in NO weather / SUN (Groudon-Drought permanent sun — Sunny Day the MOVE is not modeled) /
  Synthesis in SAND (Tyranitar Sand Stream), and recovery-into-a-real-battle-to-game-end. It captures
  `initSeed` + per decision the `seedAfter`, both actives' species/hp/maxhp/fainted/status(+inner counter) +
  boosts + confusion + pokemon_left + first mover + winner, with fail-loud `require` floors (a heal applied,
  the full-HP fail, a Rest sleep, a wake) so every branch realizes, and a STALL guard. **Result: 8 scenarios,
  4468 decision rows, 640 wins; healed=525, healFail=80, restSlp=160, wake=160.** The harness shares the
  setup_move_golden TAB format so the Rust gate reuses the parser.
- **`tests/recovery_move_test.rs`** seeds a `BattleState` at `initSeed` (gen3customgame), runs
  `run_full_battle` WITHOUT re-seeding, and asserts per DECISION BOUNDARY: (a) species / **HP** (the heal
  amount — a 1-HP error desyncs the STATE) / maxhp / fainted / **STATUS** + the sleep/Toxic inner counter +
  boosts + confusion + pokemon_left + request kind + first mover; AND (b) the post-decision seed == the
  sim's `seedAfter` — the EXACT cross-decision draw-order+count proof to game-end (the heal amounts + Rest's
  `random(2,6)` must each be in the exact place/count) — PLUS the winner. Coverage floors pin every branch
  (heal-applied, the heal-fail, Rest-sleep). Deterministic `turn.rs` unit gates pin the bare model (the
  half-HP / weather-fraction amounts, the full-HP fails, Rest's one `random(2,6)` in gen3customgame, and the
  gen3ou shuffle-then-`random(2,6)` order).
- **Honest scope / deferred**: **Wish** (a DELAYED slot-keyed end-of-next-turn heal — a pending-heal model),
  **Heal Bell / Aromatherapy / Refresh** (team/self STATUS cure, not HP — a separate small layer), **Pain
  Split / Leech Seed / drain / Ingrain / Aqua Ring**, plus everything the setup-move section defers, and the
  weather-SETTING moves (Sunny Day / Rain Dance — not modeled, so the sun scenario uses Groudon's Drought).
  The recovery e2e expansion uses the in-engine weather from abilities only.

## Protect / Detect: the stall-draw + move-block gate (per-seed PER-DECISION STATE+HP+STATUS+STALL-COUNTER+SEED differential)

This closes PROTECT and DETECT (identical full-turn protection) — proven bit-for-bit to game-end, with
the per-decision STALL COUNTER + the block (no damage) asserted. The two cruxes (verified bit-for-bit vs
the omniscient sim's PRNG probe `harness/probe_protect_rng.js` + the resolved gen3 `stall` condition):

- **THE STALL / CONSECUTIVE-SUCCESS DRAW** (`run_protect`): Protect/Detect are NEVER-MISS (no accuracy
  draw) + **priority 3** (gen3) so they resolve BEFORE the foe's attack. `onPrepareHit` runs
  `!!this.queue.willAct() && runEvent('StallMove')`:
  - **The FIRST protect** (counter 0, no `stall` volatile) SHORT-CIRCUITS with **NO draw** — `runEvent`
    has no handler → true → always succeeds → `onHit` adds the `stall` volatile (`onStart` counter 2).
  - **A CONSECUTIVE protect** (the volatile present) draws ONE `randomChance(1, counter)` at the floored
    denominator **2 → 4 → 8 → 8** (the gen4-inherited `stall` `counterMax: 8` caps the doubling). A SUCCESS
    (re)adds the volatile (`onStart` 2 / `onRestart *= 2` capped at 8) + refreshes `duration` to 2.
  - **A FAILED stall roll** draws nothing more and — for the gen3 **resolved gen5-base** `onStallMove`
    (which, unlike the gen8+ base condition, has NO `delete pokemon.volatiles['stall']`) — does **NOT
    delete** the volatile: the counter + duration PERSIST UNCHANGED, so consecutive fails re-roll at the
    SAME denominator (`2(F),2(F),2(F)`) AND a `stall` residual handler still fires that turn. (Getting
    this no-delete-on-fail RIGHT was the e2e-capstone fix — the gen8+-base delete model wrongly dropped
    the stall residual handler.)
  - **The `willAct()` gate**: a Protect that resolves with NO move/switch action still pending — the foe
    **SWITCHED** (a switch is order 103 < the protect's move order 200, so it already ran) — FAILS
    draw-free, no volatile (the `&&`-short-circuit skips the stall roll AND `onHit`). VERIFIED: Protect
    into a foe switch leaves the protector with NO volatiles (`|move|…Protect||[still]`).
  - The stall counter **resets** after one non-protect/switch turn — the volatile's `duration: 2` expiry,
    modeled at the RESIDUAL via the `MonState::stall_duration` countdown (`onRestart` refreshes it to 2 on
    every SUCCESS, so consecutive successes never expire); switch-out (clearVolatile) clears all three.
- **THE MOVE-BLOCK DRAW** (the block check in `run_move`): in gen-3 `data/mods/gen3/scripts.ts::tryMoveHit`
  the **accuracy roll is drawn FIRST** (`randomChance(accuracy,100)` at line 364), then `if (accPass)
  runEvent('TryHit')` (line 369) where the protect `onTryHit` blocks. So a BLOCKED foe move TARGETING the
  protected mon **DRAWS its accuracy roll** (skip iff never_miss) then is blocked — drawing NO crit /
  damage / secondary / status — and the block precedes the immunity report (an EQ into a Flying/Levitate
  protector shows `-activate Protect`, NOT `-immune` — VERIFIED). A miss never reaches the block
  (`-miss`). Protect only blocks a move TARGETING the protected mon — a self-target move (the foe's own
  Protect / setup / recovery) is never blocked. DRAW-FREE block (only the accuracy roll already happened).
- **The RESIDUAL duration-handler interaction** (`run_residuals`): the `protect` (`duration: 1`), `stall`
  (`duration: 2`), and `flinch` (`duration: 1`) volatiles each register a residual duration handler
  (`findPokemonEventHandlers(..., 'duration')` gathers every duration-bearing volatile to count it down —
  no `onResidual` callback, draw-free apply). They sort at order **NO_ORDER** / subOrder **2** (the gen3
  effectTypeOrder for a Condition) and so PARTICIPATE in the residual speed-sort's tie-group Fisher-Yates
  shuffle: a protecting mon adds 2 tied handlers (protect+stall), and a FAILED-protect-into-a-RockSlide-
  flinch adds stall+flinch tied — a shuffle COUNT the model must match. (Flinch was draw-free before
  protect — alone at NO_ORDER it never tied — but now ties with a surviving stall; confusion has NO
  `duration` so registers NO handler.) The apply is a no-op except the stall handler decrements
  `stall_duration` → zeroes `protect_counter` on expiry.
- **FAIL-LOUD** (`run_protect`): only Protect / Detect are modeled. **Endure** (`isProtect` but
  `volatileStatus:'endure'` — a survive-at-1-HP `onDamage`, a different mechanic) + the gen4+ Quick Guard
  / Wide Guard / King's Shield / Spiky Shield (none in gen3) PANIC so a future protection variant can
  never silently desync.
- **Differential harness** `harness/gen_protect_move_golden.js` drives the omniscient `BattleStream`
  (no server, **gen3customgame**) over **6 scenarios × 80 seeds** to game-end with CONSTRUCTED scenarios
  that each ISOLATE a branch: a SINGLE protect blocking an attack, CONSECUTIVE protects (the stall
  denominator BOTH ways — the sweep realizes a consecutive-protect SUCCESS AND a FAILURE, proving the
  2/4/8 sequence), a protect → a NON-protect move → a protect (the counter reset), protect vs a STATUS
  move (Thunder Wave blocked, no para), Detect (the identical-protection variant), and protect-into-a-
  real-battle-to-game-end. It captures `initSeed` + per decision the `seedAfter`, both actives' species/
  hp/maxhp/fainted/status + boosts + confusion + pokemon_left + **the STALL COUNTER** (`volatiles.stall.
  counter`) + first mover + winner, with fail-loud `require` floors (a block, a protect SUCCESS, a
  consecutive-protect FAILURE) so every branch realizes, and a STALL guard. **Result: 6 scenarios, 2772
  decision rows, 480 wins; block=1102, protectUp=1133, stallFail=369.**
- **`tests/protect_move_test.rs`** seeds a `BattleState` at `initSeed` (gen3customgame), runs
  `run_full_battle` WITHOUT re-seeding, and asserts per DECISION BOUNDARY: (a) species / **HP** (a blocked
  move's damage must NOT land) / maxhp / fainted / **STATUS** + the sleep/Toxic inner counter + boosts +
  confusion + **the STALL COUNTER** (the consecutive-use escalation 0→2→4→8 + the reset) + pokemon_left +
  request kind + first mover; AND (b) the post-decision seed == the sim's `seedAfter` — the EXACT
  cross-decision draw-order+count proof to game-end (a wrong stall denominator, a first-protect that
  wrongly drew, or a blocked move that wrongly skipped/added its accuracy roll desyncs the SEED) — PLUS
  the winner. **Result: 480 runs, 2772 seed + 5544 HP + 4984 stall-counter assertions (1102 block
  decisions, 272 escalated counter≥4).** Deterministic `turn.rs` unit gates pin the bare model
  (`first_protect_draws_nothing_and_sets_counter_two`, `consecutive_protect_draws_one_stall_roll_and_
  escalates` incl. the no-delete-on-fail persist, `protect_blocks_foe_move_after_its_accuracy_draw`,
  `protect_does_not_block_the_protectors_own_move`, `endure_panics_fail_loud`).
- **Honest scope / deferred**: Endure + the gen4+ guard moves (fail-loud), plus everything the
  recovery-move section defers. The protect e2e expansion surfaced + FIXED the residual duration-handler
  gap (protect/stall/flinch register residual handlers) + the no-delete-on-fail stall model + the
  `willAct()` switch gate.

## Spikes: the entry-hazard + side-condition gate (per-seed PER-DECISION STATE+HP+SPIKES-LAYERS+SEED differential)

This closes **Spikes** — the gen-3 ENTRY HAZARD — which introduces the FIRST SIDE CONDITION (a per-side
persistent state, reusable by future hazards/phazing). Proven bit-for-bit to game-end, with the
per-decision SPIKES LAYERS (per side) + the grounded switch-in chip asserted. The two cruxes (verified
bit-for-bit vs the omniscient sim's PRNG probe `harness/probe_spikes_rng.js` + the resolved gen3 `spikes`
condition):

- **THE SIDE STATE + THE SPIKES MOVE** (`SideState::spikes: u8` 0..=3 + `run_status_move`'s spikes arm):
  Spikes (`sideCondition:'spikes'`, `target:'foeSide'`, `accuracy:true`) is the FOE-side-targeting
  hazard. The move is NEVER-MISS → NO accuracy draw; it increments the CASTER's FOE side's `spikes` layer
  by 1, CAPPED at 3 (`onSideRestart`: `if (layers >= 3) return false` → a 4th Spikes FAILS, `-fail`).
  DRAW-FREE both ways (the `onSideStart`/`onSideRestart` consume NO PRNG — the only effect is bumping the
  layer count); `landed` is FALSE (a status `moveHit` returns `undefined` → the in-tryMoveHit Update is
  skipped). So a Spikes-vs-move turn draws ONLY the existing action-order/eachEvent shuffles. The `spikes`
  count is a SIDE condition (not a mon volatile), so it PERSISTS across switches and is 0 only at
  construction.
- **THE SWITCH-IN DAMAGE** (`apply_entry_hazards` in `run_switch` — the gen-3 `runSwitch`'s
  `runEvent('EntryHazard')`, gen4-inherited). The gen4 `runSwitch` ORDER (verified vs the source):
  `runEvent('EntryHazard')` → `runEvent('SwitchIn')` → `if (!pokemon.hp) return false;` → `singleEvent(
  'Start', ability)`. So the **Spikes damage fires BEFORE the ability `Start`** — a Spikes-KO on entry
  SKIPS the entrant's Intimidate/weather Start. GROUNDED-ONLY: a Flying-type or **Levitate** entrant takes
  ZERO (`isGrounded()` is false/null; Iron Ball / Air Balloon / Magnet Rise don't exist in gen-3 OU →
  grounded == not-Flying && not-Levitate). The amount (the resolved `spikes.onEntryHazard`,
  `[_,3,4,6][layers]·maxhp/24` → `damage()` → `clampIntRange(_,1)` = floor-then-min-1): **1 layer
  `max(floor(maxhp/8),1)`, 2 layers `max(floor(maxhp/6),1)`, 3 layers `max(floor(maxhp/4),1)`** (e.g.
  Snorlax maxhp 524 → 65 / 87 / 131). DRAW-FREE (the deterministic `this.damage`; the nested
  `runEvent('Damage')` has no drawing handler for the modeled abilities — the probe shows a grounded
  switch-in turn draws the SAME count as an immune one). A Spikes hit that zeroes HP **faints the
  entrant** (the runAction tail's `process_faints` sets `fainted` + decrements `pokemon_left`, `check_win`
  runs, `check_fainted` flags → the switch gate forces ANOTHER replacement, which ALSO takes Spikes on its
  entry) — wired entirely through the EXISTING faint/replacement machinery, with **no Quick Claw / extra
  draw** in the hazard chain. The whole chip ladder + the spikes-KO are seed-independent (Spikes is
  draw-free + deterministic).
- **Differential harness** `harness/gen_spikes_golden.js` drives the omniscient `BattleStream` (no server,
  **gen3customgame**) over **5 scenarios × 80 seeds** to game-end with CONSTRUCTED scenarios that each
  ISOLATE a branch: lay 1 + a grounded switch-in (maxhp/8), stack 2/3 layers + a Spikes-at-max FAIL + the
  3-layer switch-in (maxhp/4), a FLYING + LEVITATE entry (ZERO), a spikes-KO-on-entry → forced replacement
  (which ALSO takes spikes), and spikes-into-a-real-battle-to-game-end. It captures `initSeed` + per
  decision the `seedAfter`, both actives' species/hp/maxhp/fainted/status + boosts + confusion + the
  **per-side SPIKES LAYERS** + pokemon_left + first mover + winner, with fail-loud `require` floors (a
  lay, a switch-in chip, a Spikes-at-max FAIL, an immune entry, a spikes-KO-on-entry) so every branch
  realizes, and a STALL guard. **Result: 5 scenarios, 3475 decision rows, 400 wins; sideStart=880,
  spikesDamage=1440, sideFail=80, immuneEntry=400, koEntry=160.** The golden EXTENDS the protect TAB
  format with a 2-col spikes-layers tail (p1Spikes, p2Spikes) → DEC has 49 fields.
- **`tests/spikes_test.rs`** seeds a `BattleState` at `initSeed` (gen3customgame), runs `run_full_battle`
  WITHOUT re-seeding, and asserts per DECISION BOUNDARY: (a) species / **HP** (the spikes chip — a
  grounded entry takes the EXACT floor, a Flying/Levitate entry takes ZERO) / maxhp / fainted / STATUS +
  the sleep/Toxic inner counter + boosts + confusion + **the per-side SPIKES LAYERS** + pokemon_left +
  request kind + first mover; AND (b) the post-decision seed == the sim's `seedAfter` — the EXACT
  cross-decision draw-order+count proof to game-end (a wrong switch-in chip desyncs the HP STATE; a wrong
  lay/cap desyncs the SPIKES-LAYERS; a wrong draw model — the Spikes move or the hazard wrongly drawing —
  desyncs the SEED) — PLUS the winner. **Result: 400 runs, ~3475 seed + ~6950 HP + ~6950 spikes-layer
  assertions (1440 switch-in-chip rows, 160 spikes-KO-on-entry rows), 400 wins.** Deterministic `turn.rs`
  unit gates pin the bare model (`spikes_move_increments_foe_side_draw_free_and_caps_at_three`,
  `spikes_switch_in_damage_grounded_per_layer_draw_free`, `spikes_switch_in_flying_and_levitate_take_zero`,
  `spikes_ko_on_switch_in_zeroes_hp_draw_free`).
- **Honest scope / deferred**: **Toxic Spikes** + **Stealth Rock** (NOT gen3), **Rapid Spin** (the
  hazard-CLEAR move — a damaging move the fuzz won't pick as a modeled status move, so hazards persist);
  Spikes is the only gen-3 entry hazard. Plus everything the protect-move section defers. The spikes e2e
  expansion adds the `MODELED_HAZARD_MOVES` set to `isModeledMove` (Spikes is special-cased in the
  category-Status branch + a belt-and-braces guard before the `sideCondition` reject) — real
  Skarmory/Forretress/Cloyster spikers now lay Spikes + grounded switch-ins take the hazard chip on the
  filtered gate, with the per-side spikes layers asserted bit-for-bit.

## Phazing: the Roar / Whirlwind forced-random-switch gate (per-seed PER-DECISION STATE+HP+SPIKES-LAYERS+DRAG-SPECIES+SEED differential)

This closes **PHAZING** — the gen-3 `forceSwitch` moves **Roar** + **Whirlwind**, which force the FOE
to switch to a RANDOM eligible team member. Proven bit-for-bit to game-end, with the per-decision
DRAGGED-IN species (which random mon was pulled) + the phaze-into-Spikes chip asserted. The cruxes
(verified bit-for-bit vs the omniscient sim's PRNG probe `harness/probe_phaze_rng.js` + the resolved
gen3 `forceSwitch` / `dragIn` source):

- **THE DRAW MODEL** (`run_status_move`'s phaze arm + `drag_in` at the `turn_loop` runAction tail):
  - **PRIORITY −6** → the phazer almost always moves LAST (the dex priority feeds `sort_actions`).
  - **ACCURACY** — gen-3 Roar/Whirlwind resolve to **`accuracy: 100`** (NOT `true`! — the SURPRISE this
    layer surfaced). The base Showdown `data/moves.ts` lists `accuracy: true`, but the resolved gen-3 dex
    value is **100** (our `data/pokemon/gen3_moves.json` already carries it: `accuracy:100`,
    `never_miss:false`, `isPhaze:true`). So a phaze is **NOT never-miss** — it DRAWS `randomChance(100,100)`
    (the accuracy roll always passes but CONSUMES a draw). A category-Status move routes to
    `run_status_move` BEFORE `run_move`'s damaging-path accuracy draw, so the phaze arm draws it itself.
  - **THE RANDOM TARGET DRAW** — `forceSwitch` (battle-actions.ts:1167) sets the foe's `forceSwitchFlag`
    IFF `canSwitch(foe.side)` (the foe has ≥1 eligible non-active, non-fainted bench mon). The ACTUAL drag
    happens LATER, at the runAction tail (battle.ts:2348-2353, AFTER the whole move body / any in-tryMoveHit
    Update, BEFORE `faintMessages`), via `dragIn` → `getRandomSwitchable` → `sample(possibleSwitches)` →
    `this.random(n)` — THE RANDOM TARGET DRAW. Drawn EVEN when **n == 1** (`random(1)` returns 0 but STILL
    calls `rng.next()` — the n=1 draw-count gotcha). The port mirrors this exactly: `drag_in` draws
    `random_below(eligible.len())` (the eligible list is `eligible_switch_ins` = `possibleSwitches`'s
    array-order bench walk, so the sampled index matches the sim's), then `execute_switch` (the array swap
    + the entrant's `updateSpeed` + the `insert_runswitch` enqueue). A phaze with NO eligible foe (its last
    mon alive) FAILS draw-free — only the accuracy roll draws (the `sample` is NOT reached).
- **THE DRAG** (the forced switch-in, gen-3 `dragIn` → `switchIn(isDrag=true)`): reuses the ENTIRE
  voluntary-switch machinery — the dragged mon takes Spikes via the existing `runSwitch`
  `runEvent('EntryHazard')` (drag → EntryHazard/Spikes → SwitchIn → ability `Start`, IDENTICAL ordering to
  a normal switch-in), fires its switch-in ability `Start` (Intimidate / Sand Stream), and a Spikes-KO on
  the dragged mon faints it → forces a NORMAL replacement (the owner chooses). The phazed-OUT mon's
  boosts/volatiles are cleared (`execute_switch`'s clearVolatile). The dragged mon does NOT get an action
  this turn (the phaze is end-of-turn, priority −6). gen-3 `dragIn` also runs a 2nd `runEvent('DragOut')` —
  DRAW-FREE for the modeled abilities; Suction Cups (`onDragOut`) is NOT on any modeled team.
- **FAIL-LOUD**: `modeled_phaze_move` lists ONLY `roar` / `whirlwind`; every other `forceSwitch` / status
  move falls through to `run_status_move`'s fail-loud guard. DEFERRED: **Haze** (resets boosts — a DIFFERENT
  mechanic, NOT `forceSwitch`), Perish Song, Roar of Time (not gen3). Roar + Whirlwind are the ONLY gen-3
  phaze moves (`isPhaze` == `forceSwitch`).
- **Differential harness** `harness/gen_phaze_golden.js` drives the omniscient `BattleStream` (no server,
  **gen3customgame**) over **7 scenarios × 80 seeds** to game-end with CONSTRUCTED scenarios that each
  ISOLATE a branch: Roar drags a RANDOM bench mon (the seed sweep makes DIFFERENT mons get dragged — the
  random-target proof), a Roar with ONE eligible bench (the n=1 sample draw), a Roar that FAILS (foe's last
  mon — no drag, accuracy-only draw), Whirlwind, Roar INTO Spikes (the dragged mon takes the hazard chip),
  repeated Roar into a stochastic spikes-KO, and phaze-into-a-real-battle-to-game-end. It captures
  `initSeed` + per decision the `seedAfter`, both actives' species/hp/maxhp/fainted/status + boosts +
  confusion + the per-side spikes layers + the DRAGGED-IN species + first mover + winner, with fail-loud
  `require` floors (a random drag with ≥2 DISTINCT mons / the n=1 drag / a phaze FAIL / a phaze-into-Spikes
  chip) so every branch realizes, and a STALL guard. **Result: 7 scenarios, ~10388 decision rows, 560 wins;
  phazeDrag≈2795, phazeFail≈2490, spikesDamage≈1769, koEntry≈273.** The golden EXTENDS the spikes TAB format
  with a 1-col dragged-species tail → DEC has 50 fields.
- **`tests/phaze_test.rs`** seeds a `BattleState` at `initSeed` (gen3customgame), runs `run_full_battle`
  WITHOUT re-seeding, and asserts per DECISION BOUNDARY: (a) the post-decision active SPECIES (which mon was
  dragged — a wrong sampled mon makes the WRONG mon active, a STATE desync the seed match alone wouldn't
  catch) / HP (the phaze-into-Spikes chip) / maxhp / fainted / STATUS + the inner counter + boosts +
  confusion + the per-side SPIKES LAYERS + pokemon_left + request kind + first mover; AND (b) the
  post-decision seed == the sim's `seedAfter` — the EXACT cross-decision draw-order+count proof (the
  accuracy roll + the n≥1 sample + the no-draw-on-FAIL must each be exact) — PLUS the winner. **Result: 560
  runs, 10388 seed + 20776 HP + 20776 spikes-layer assertions (2795 drag decisions, 1769
  phaze-into-spikes-damage rows), 560 wins.** A RANDOM-TARGET assertion checks ≥1 multi-bench scenario
  dragged ≥2 DISTINCT species across the seed sweep (else the "random" isn't proven). The exact single-drag
  spikes-KO is ALSO pinned deterministically in `regression_test.rs` (`phaze_drag_into_a_spikes_ko_*`).
- **Honest scope / deferred**: Haze / Perish Song (fail-loud), plus everything the spikes-move section
  defers. **Phaze is currently EXCLUDED from the e2e capstone** (`PHAZE_E2E_EXCLUDED = true` in
  `gen_e2e_fuzz.js`, gating `MODELED_PHAZE_MOVES` out of `isModeledMove`). It is bit-for-bit in its
  DEDICATED golden (`phaze_test.rs`, 560 runs), but the e2e's real-team battles with BOTH sides phazing
  across long switch histories surface a phaze-in-a-multi-draw-turn `sample` desync that the simpler
  dedicated scenarios don't reach: e.g. an e2e battle where p2 Roars p1 on a turn p1 also Ice-Beams (the
  freeze secondary) — the post-turn SEED matches but the Rust's `sample` index diverges from the sim's
  (a STATEFUL interaction: the FIRST phaze of the battle drags correctly, a LATER one in the same battle
  picks a different bench mon). Keeping phaze OUT of the STRICT e2e gate (rather than letting a silent
  desync in) honors the bit-for-bit law; the dedicated golden + the regression pins remain the phaze
  proof. Re-enable (`PHAZE_E2E_EXCLUDED = false`) once the multi-draw-turn `sample` ordering is fixed.
  **The switch-tie-weather `eachEvent('WeatherChange')` fix (2026-06-30, which re-enabled Substitute)
  did NOT resolve the phaze desync** — and is not expected to: the phaze bug is a draw-POSITION
  (`sample`-index-within-a-multi-draw-turn) desync at IDENTICAL total draw COUNT, a DIFFERENT class from
  the switch-tie-weather draw-COUNT bug (a missing draw). Tested empirically: flipping
  `PHAZE_E2E_EXCLUDED = false` + regenerating does NOT yield a clean strict pass (the phaze-changed
  filter-clean corpus also surfaces an unmodeled-move/move-slot leak — a Gengar's `DestinyBond` executes
  in the port — on top of the documented `sample` desync), so phaze stays excluded; the dedicated golden
  + the P1–P3 regression pins remain the phaze proof.

## Fixed-damage moves: the `damage:` / `damageCallback` gate (per-seed PER-DECISION STATE+HP+STATUS+SEED differential)

This closes **FIXED-DAMAGE / FIXED-FORMULA moves** — a `damage:` / `damageCallback` move in Showdown
that BYPASSES `getDamage`, so it deals a fixed/derived number with **NO crit roll + NO 16-way damage
roll**. Proven bit-for-bit to game-end, with the per-decision HP (the exact fixed amount) asserted.
The cruxes (verified bit-for-bit vs the omniscient sim's PRNG probe `harness/probe_fixeddamage_rng.js`):

- **The modeled set** (`fixed_damage_amount`, an id-gated helper mirroring `recovery_heal_amount`,
  routed by `is_fixed_damage_move` in `run_move` → `run_fixed_damage_move`): **Seismic Toss / Night
  Shade** (`damage:'level'` → the USER's level, e.g. level 100 → 100), **Sonic Boom** (fixed 20),
  **Dragon Rage** (fixed 40), **Super Fang** (`damageCallback` = `clampIntRange(target.hp/2,1)` =
  `max(floor(target.hp/2),1)`). The data (`gen3_moves.json`) carries NO `damage` field — a fixed-damage
  move is recognized by its move **id**, not a data flag (the same convention every other move layer
  uses; Showdown's `damage`/`damageCallback` is gated on the id set here).
- **The ROUTING crux**: a fixed-damage move carries `basePower:0`, so `derive_category` classifies it
  **Status** — so `is_fixed_damage_move` is checked in `run_move` **BEFORE the `category == Status`
  branch** (else it would fall into `run_status_move`'s fail-loud guard). It also precedes the
  `base_power == 0` no-op.
- **The draw model**: (1) ACCURACY — `randomChance(acc,100)`, drawn UNLESS never_miss. **Seismic Toss /
  Night Shade / Dragon Rage are acc-100 but NOT never-miss, so they STILL draw ONE accuracy roll** (the
  phaze acc-100 precedent — always passes but CONSUMES a draw); **Sonic Boom / Super Fang are acc-90 and
  CAN genuinely miss**. This is the ONLY per-move draw — **NO crit, NO damage roll, NO secondary**. (2)
  TYPE IMMUNITY — accuracy-drawn-**THEN-`-immune`** (the SAME short-circuit + draw count as a normal
  damaging move, via `move_is_immune` over a `DamageContext`): **Seismic Toss (Fighting)→a GHOST**,
  **Night Shade (Ghost)→a NORMAL**, **Sonic Boom / Super Fang (Normal)→a GHOST** all report `-immune`
  (NOT `-miss`) with ZERO damage. (3) DAMAGE — the exact fixed amount applied through the EXISTING
  `absorb_into_sub` / `apply_damage` / deferred-faint machinery. `landed` is TRUE on a hit (a `damage:`
  move returns a truthy number → the in-tryMoveHit Update fires), FALSE on miss/immune/block.
- **The SUBSTITUTE interaction (VERIFIED — and it CONTRADICTED a naive assumption, settled by the probe,
  the project's source of truth):** a fixed-damage move into a sub hits the SUB (the fixed NUMBER hits
  the sub HP, breaks with no carry — `-activate Substitute [damage]` on a survive / `-end` on a break),
  and **Super Fang STILL halves the MON's current hp behind a sub** (its `damageCallback` reads
  `target.hp` BEFORE the sub-intercept redirects the resulting number; VERIFIED: SF into a full-HP-536
  Blissey behind a 178-HP sub deals `floor(536/2)=268` → the sub BREAKS, NOT `floor(178/2)=89`). The draw
  model is UNCHANGED by the sub (accuracy-only).
- **Differential harness** `harness/gen_fixeddamage_golden.js` drives the omniscient `BattleStream`
  (no server, **gen3customgame**) over **9 scenarios × 80 seeds** to game-end with CONSTRUCTED scenarios
  that each ISOLATE a branch: Seismic Toss chip, Seismic Toss into a Ghost (immune), Night Shade into a
  Normal (immune), Night Shade lands, a fixed-damage KO-to-win, a fixed-damage move into a Substitute,
  Sonic Boom + Dragon Rage (incl. the acc-90 miss), Super Fang halving (incl. the miss), and
  fixed-damage-into-a-real-battle. It captures `initSeed` + per decision the `seedAfter`, both actives'
  species/hp/maxhp/fainted/status(+counter) + boosts + confusion + pokemon_left + first mover + winner,
  with fail-loud `require` floors (a fixed-damage hit, an immune, a miss, a sub-absorb) so every branch
  realizes, and a STALL guard. **Result: 9 scenarios, 4144 decision rows, 720 wins; fixedHit=479,
  fixedImmune=240, fixedMiss=60, fixedSub=80.** The golden SHARES the recovery_move_golden TAB format
  (the trailing branch flag re-meaninged to `fixedHit`) so the Rust gate reuses the parser.
- **`tests/fixeddamage_test.rs`** seeds a `BattleState` at `initSeed` (gen3customgame), runs
  `run_full_battle` WITHOUT re-seeding, and asserts per DECISION BOUNDARY: (a) species / **HP** (the fixed
  amount — a 1-HP error / a spurious crit-or-damage roll desyncs the STATE) / maxhp / fainted / STATUS +
  counter + boosts + confusion + pokemon_left + request kind + first mover; AND (b) the post-decision seed
  == the sim's `seedAfter` — the EXACT cross-decision draw-order+count proof (a spurious crit/damage roll,
  a skipped accuracy roll, or a wrong immune/miss branch desyncs the SEED) — PLUS the winner. **Result:
  720 runs, 4144 seed + 8288 HP assertions, 2469 fixed-damage-hit decisions, 720 wins.**
- **Regression pins** (4, DETERMINISTIC, revert-verified — ground truth from
  `harness/probe_fixeddamage_regression_rng.js`): `seismic_toss_deals_user_level_damage` (STATE 100 +
  SEED acc-only), `seismic_toss_into_a_ghost_is_immune_accuracy_only_seed` (STATE zero + SEED == the
  landed-hit seed), `night_shade_into_a_normal_is_immune` (STATE zero + SEED),
  `fixed_damage_into_a_substitute` (STATE sub 131→31, mon unchanged + SEED).
- **FAIL-LOUD / DEFERRED**: the DEFERRED fixed-damage family — **Psywave** (variable, draws RNG), the OHKO
  moves **Fissure / Horn Drill / Guillotine** (accuracy-gated instakill + level gate), **Counter / Mirror
  Coat / Bide** (reactive), **Endeavor** — is routed to `run_fixed_damage_move` by `is_fixed_damage_move`
  (so it can NEVER silently no-op / desync) but has no `fixed_damage_amount` entry → PANIC. e2e: the
  `MODELED_FIXED_DAMAGE_MOVES` set is added to `gen_e2e_fuzz.js`'s `isModeledMove` (admitted early, before
  the `basePower>0`/`m.damage`/`m.damageCallback` rejects), but the regenerated 220-battle golden has **0
  fixed-damage-MOVE decisions** because NONE of the 22 filter-clean teams happens to carry one (the
  leech-seed situation — the gen3ou fixed-damage users pair them with unmodeled abilities/items) — so the
  e2e doesn't EXERCISE fixed-damage; it stays fully proven by its DEDICATED golden + the 4 pins.

## PP tracking + Struggle (the first brick of `LegalActions`): per-decision STATE+HP+STATUS+PP+SEED differential

`gen3_pp_tracking_v1` — per-move PP counters + the forced-**Struggle** substitution + the Choice-Band
lock, the foundational piece of the move-legality / request-validation layer (a mon with no usable move is
FORCED to Struggle). It **un-skipped the 3 `recover_and_rest` Struggle protocol battles** (protocol
byte-diff 63/7223 → **66/8721**, 0 skipped). The draw model was nailed by `harness/probe_pp_struggle_rng.js`
(the hints were WRONG twice — corrected by the probe, the project's source of truth):

- **PP init.** A moveslot's in-battle MAX PP is the `Pokemon` ctor's `calculatePP(move, ppUps)` with the
  ctor's **hardcoded default 3 PP-ups** (NOT read from the set) → `pp * (5+3)/5 = pp * 8/5` (integer,
  gen3) for a normal move, or the raw `pp` for a `noPPBoosts` move (Struggle = 1). VERIFIED vs the sim's
  `moveSlots[k].maxpp` (Surf 15→24, Earthquake 10→16, Splash 40→64, Extreme Speed 5→8). Added `pp` +
  `noPPBoosts` to `gen3_moves.json` (via `tools/pokemon_data_extractor`; **obs-neutral** — the
  `agents.gen3_data` facade ignores it, like `critRatio`/`secondaryBoosts`, so the RL obs golden is
  unchanged) → `MoveData::max_pp()` → `MonState::move_pp`/`move_maxpp` init in `from_set`.
- **PP decrement.** −1 per USE (`Pokemon.deductPP`), **DRAW-FREE**, deducted right AFTER
  `runEvent('BeforeMove')` PASSES (battle-actions.ts:282) — so a full-para / still-asleep / flinched /
  frozen / confusion-self-hit turn deducts NOTHING (the port's `run_move` deducts only after
  `on_before_move` passes). A MISS / an IMMUNE hit STILL decrement. PP does NOT reset on switch-out
  (gen-3 — it PERSISTS; cleared only per battle).
- **Pressure −2.** A move TARGETING a **Pressure** holder deducts **2** PP (the `runEvent('DeductPP')`
  extra, battle-actions.ts:472-483), DRAW-FREE, and only for a move whose **`pressureTargets`** include
  the Pressure foe — i.e. a FOE-directed target (`turn.rs::pressure_targets_foe`, `gen3_pressure_allyteam_v1`).
  This is NOT `!targets_self`: a `self` heal/setup/protect deducts 1, and so does an **`allyTeam`** move
  (Aromatherapy / Heal Bell) / `allySide` / `allies` / `foeSide` (Spikes) — none put the foe in
  `pressureTargets` (`getMoveTargets`, pokemon.ts:854-861; the e2e_182 root cause — the old `!targets_self`
  wrongly gave Aromatherapy under a Pressure foe −2, draining its PP so the port rejected a legitimate
  late use). Only `normal` / `any` / `randomNormal` / `adjacentFoe` / `allAdjacentFoes` / the spread
  `all`+`allAdjacent` get the −2. (CLAUDE.md previously called Pressure a "provable no-op in a
  damaging-move-only fuzz" — PP-wise it is NOT.)
- **Choice-Band lock.** A Choice-item mon (gen-3: only **Choice Band**) LOCKS to the FIRST slot it uses
  (`choiceband.onModifyMove` → the `choicelock` volatile records `activeMove.id`; `choicelock.onDisableMove`
  disables the rest). `MonState::choice_locked_move = Some(k)` is set in `run_move` when the CB mon uses
  slot `k`, cleared on switch-out (`execute_switch`) + faint (`process_faints`). This is what forces
  Struggle when the LOCKED move hits 0 PP while other slots still have PP — the CB-Tyranitar exhausting
  Crunch → Struggle in the protocol battles.
- **Forced Struggle.** `MonState::must_struggle()` = every USABLE slot (respecting the Choice lock) is at
  0 PP → `side.choose` substitutes `moveid:'struggle'` for the scripted `move K`. The queue-build sets a
  `struggle` flag from `must_struggle()` (at the TOP of the turn, from the CURRENT PP — PP is exhausted on
  a PRIOR turn). A `move K` on a 0-PP slot while ANOTHER move is usable is REJECTED **draw-free** (the
  `move_decision_is_legal` PP gate, extending the out-of-range forced-replacement-resume gate — mirrors
  the sim's "doesn't have PP" reject; the boundary stays OPEN, verified vs the sim).
- **Struggle mechanics (probe-settled — the recoil hint was wrong).** Modeled as a synthetic move
  (`MoveAction::struggle`, resolving the `struggle` dex entry): type **typeless '???'** (`move.type='???'`
  in `onModifyMove` — no STAB, HITS everything INCLUDING Ghosts, a typeless move has no type-chart row →
  1×), **50 BP PHYSICAL**, **accuracy 100 — NOT never-miss** (gen-3 `data/mods/gen3/moves.ts` sets
  `accuracy:100` so it DRAWS an accuracy roll — the phaze/fixed-damage acc-100 precedent), then crit +
  damage like a normal move. **RECOIL = `max(floor(damageDealt / 4), 1)`** — the gen-3 mod's
  `{recoil:[1,4], struggleRecoil:false}` + `scripts.ts::calcRecoilDamage` **`Math.floor`** (NOT the
  base-sim `Math.round`, NOT the gen4+ `struggleRecoil = clampIntRange(trunc(maxhp/4),1)`): floor(130/4)=32,
  not round=33, not maxhp/4. Applied DRAW-FREE via `apply_damage` to the user, emitting `|-damage|<user>|
  <HP>|[from] Recoil|[of] <target>` (`ProtocolBuilder::damage_of`). Struggle consumes NO PP, does not set
  the Choice lock, and skips `apply_secondaries` (the stale scripted slot's secondary must NOT fire). A
  Struggle turn draws acc + crit + damage + Quick Claw (when not the deciding faint) — the SAME count as a
  normal damaging move (so the seed suites stay byte-identical; Struggle's OWN draws only occur in the new
  golden + the un-skipped protocol battles).
- **Truncation turn-marker fix.** The `|turn|N` marker is now emitted at the REQUEST (before choice
  validation, via a `framing_open` latch in `run_full_battle`), so a rejected fresh-turn `move`
  (out-of-range OR out-of-PP) still shows its marker — previously it emitted only on the first VALID
  submission, so a TRUNCATED capture whose final recorded decision is a rejected out-of-PP `move`
  (`spikes_and_phaze/2`: the 16th Earthquake exhausts EQ's 16 PP → the phantom `m0` is rejected) lost the
  trailing `|turn|21`. Observation-only (no seed change — the forced-replacement resume case emits the
  marker at the same filtered position).

**Validated** by `tests/pp_struggle_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+**PP**)+SEED+winner
differential to GAME-END over 5 scenarios × 80 seeds in gen3customgame: 400 runs, 4424 decision rows,
4424 seed + 8368 PP assertions, 1035 forced-Struggle / 1035 recoil / 378 Pressure−2 / 3292
immune-decrement / 1355 0-PP-slot rows, 400 wins — CB-lock forced Struggle into a Levitate Ghost [immune
decrement + Struggle-into-a-Ghost + recoil], single-move all-slots Struggle, Pressure −2, PP-persists-
across-switch + miss-decrement, PP-into-a-real-battle to a win) + the un-skipped protocol battles + a
`src/dex/moves.rs` `max_pp` unit gate + 4 DETERMINISTIC revert-verified `tests/regression_test.rs` pins
(`pp_decrements_on_use_draw_free`, `pressure_decrements_two_pp`,
`no_usable_move_forces_struggle_and_struggle_recoil_is_gen3_quarter_damage_dealt` [covers the forced
Struggle AND the gen-3 floor(dmg/4) recoil]; ground-truth seeds from
`harness/probe_pp_struggle_regression_rng.js`). OBSERVATION-ONLY: the seed suites (e2e 13367, battle 2034,
fullbattle 2053, secondary 4328) stay **BYTE-IDENTICAL** — PP + the Choice lock + the substitution are ALL
draw-free.

## Taunt + Disable: the move-SELECTION-restriction gate (per-decision STATE+TAUNT+DISABLED-SLOT+SEED differential)

`gen3_taunt_disable_v1` — the first FOE-imposed move-legality restrictions (extending the PP/Choice-lock
brick of `LegalActions`): **Taunt** blocks every Status-category move, **Disable** blocks the target's
last-used slot, both composing with PP + the Choice lock into forced Struggle. The draw/duration model
was nailed by the probe family `harness/probe_taunt_disable_rng.js` + `probe_disable_{onstart,
duration_branch,duration_direct,full_lifecycle,willmove_determinant,reviewer_scenario}.js` +
`probe_disable_zero_pp_rng.js` (the onStart 0-PP guard) + `probe_taunt_duration_branch.js` +
`probe_taunt_disable_onbeforemove_rng.js`:

- **THE MOD-CHAIN CAUTIONARY TALE (read before "fixing" anything here).** A 3-lens review once derived
  the durations from the BASE `data/moves.ts` conditions and confidently declared the port +1-wrong.
  Direct probes of the sim's gen3 behavior REFUTED the review — because gen3's conditions
  `inherit: true` from the **gen4 mod**, which REPLACES both `onStart`s: base taunt's
  `activeTurns && !willMove → duration++` is GONE (gen4's onStart is a bare `-start`), and base
  disable's `willMove → duration--` is REPLACED by gen4's `!willMove → duration++`. Read the WHOLE mod
  chain (gen3 → gen4 → base) to form hypotheses; let the probe decide. The orchestrator independently
  re-ran the lifecycle probe and confirmed the port.
- **TAUNT** (`taunt`): Dark, Status, **accuracy 100** (NOT never-miss → DRAWS `randomChance(100,100)`),
  `protect: 1` (a Protect/Detect BLOCKS it after the accuracy roll), `bypasssub: 1` (a Substitute does
  NOT). Applies the `taunt` volatile at a **FIXED duration 2 — NO duration draw, NO onStart increment
  in ANY branch** (probe: a taunter-SECOND-on-turn>=2 stores the same 2; the golden's
  `taunt_second_turn2_minor_a` scenario exercises that branch bit-for-bit). A re-Taunt into an
  already-taunted foe FAILS draw-free. While taunted: every **Status-category** slot is un-selectable
  (`move_usable` uses `MoveData::blocked_by_taunt()` = derived-Status MINUS the fixed-damage family —
  Seismic Toss is bp-0 but Showdown-Physical, so it stays usable, probe-verified) and a QUEUED status
  move is cant'd at EXECUTION (`|cant|<mon>|move: Taunt|<Move>`, `onBeforeMove` at priority 0 — gen3
  DELETES gen4's `onBeforeMovePriority: 5`, so it sorts AFTER paralysis(1): a taunted+paralyzed mon
  still draws the para roll first — probed +1 draw), draw-free, NO PP. The residual duration tick sits
  at **order 10 / subOrder 15** (gen4-inherited; the base's `onResidualOrder: 15` is SHADOWED — probed:
  a FAST taunted mon's `-end ... [silent]` precedes a SLOW foe's brn `-damage` in the same residual,
  which order-15 would reverse).
- **DISABLE** (`disable`): Normal, Status, **accuracy 55** (CAN miss — drawn unconditionally),
  `protect: 1` + `bypasssub: 1`, noCopy. `onTryHit` FAILS **draw-free** (BEFORE the duration draw) if
  the target has no `last_move` (never moved / just switched in / lastMove was Struggle). On a landed
  hit: ONE **`random(2,6)`** (the gen3 `durationCallback`, rolled ∈ {2..5}), then the gen4-inherited
  onStart does `duration++` iff the target has ALREADY moved this turn — **`turn.rs`:
  `stored = if foe_will_move { rolled } else { rolled + 1 }`** (the SETTLED, probe-verified model; the
  faster-disabler case frees one selection-turn EARLIER). **The onStart 0-PP guard**
  (`gen3_disable_zero_pp_v1`): if the lastMove's slot has **0 PP** left (last PP spent, lastMove not
  yet overwritten — e.g. a mono-move mon now forced to Struggle), the volatile is **REJECTED AFTER
  the draws** — accuracy + `random(2,6)` both consumed, then `-fail` with the announce retro-edited
  to `|move|<user>|Disable||[still]` (`attrLastMove('[still]')`), NO `-start`, NO volatile, NO
  residual duration handler (`probe_disable_zero_pp_rng.js`; pin TD5 — pre-fix the port recorded a
  PHANTOM volatile whose residual handler could tie a taunt/stall/flinch handler → an extra
  tie-shuffle draw → a latent seed desync). A re-Disable into an already-disabled foe
  FAILS draw-free (NO `random(2,6)` — `addVolatile` returns false first). While disabled: that ONE slot
  is un-selectable and a QUEUED now-disabled move is cant'd at EXECUTION (`|cant|<mon>|Disable|<Move>`,
  `onBeforeMove` at priority 7 — BEFORE confusion(3) + paralysis(1): a paralyzed+disabled mon draws NO
  para roll, the OPPOSITE of taunt — probed identical draw counts with/without the paralysis),
  draw-free, NO PP. The residual duration tick sits at **NO_ORDER / subOrder 2** (gen3 explicitly
  DELETES gen4's `onResidualOrder: 10, onResidualSubOrder: 13`; an order-less volatile Condition gets
  the `resolvePriority` effectType default subOrder 2 — tying with protect/stall/flinch duration
  handlers on the same mon).
- **The endTurn `runEvent('DisableMove')` handler-sort shuffle.** `endTurn` runs, per active mon in
  ARRAY order, `runEvent('DisableMove', pokemon)` — a SORTED dispatch over the mon's move-disabling
  volatiles (`taunt` / `disable` / `choicelock`), all order-less and same-speed → a mon with **>= 2**
  of them draws ONE size-2/3 tie-shuffle `random` per endTurn (a taunt-only / disable-only mon draws
  NONE). Modeled by `disable_move_event_shuffle`, drawn between the residuals and the Quick Claw.
- **Forced Struggle.** `must_struggle` composes ALL restriction sources (0 PP × Choice lock × Disable ×
  Taunt); the sim's request then offers ONLY Struggle (pinned end-to-end by TD3). Both volatiles clear
  on switch-out + faint, and `last_move` resets (a Disable into a fresh switch-in fails draw-free).

**Validated** by `tests/taunt_disable_test.rs` (a per-seed PER-DECISION
STATE(+STATUS+**TAUNT**+**DISABLED-SLOT**)+SEED+winner differential to GAME-END over **9 scenarios × 80
seeds** in gen3customgame: 720 runs, 4723 seed assertions, 8595 taunt + 8595 disabled-slot assertions,
1188 taunted / 803 disabled rows, free-ups taunt 930 / disable 289, 415 miss / 128 no-lastMove-fail /
273 forced-Struggle rows, 720 wins — incl. `taunt_second_turn2_minor_a` [the taunter-second branch] and
`disable_faster_disabler_free_up` / `disable_slower_disabler_free_up`, which pin BOTH duration branches
AT their exact free-up boundaries: **the golden gate was PROVEN by perturbation** — `rolled/rolled+2`
fails in `disable_slower_disabler_free_up`, `rolled+1/rolled+1` fails in
`disable_faster_disabler_free_up`, restore passes) + 5 revert-verified `tests/regression_test.rs` pins
(TD1 `taunt_blocks_status_move_selection_for_the_sim_window_draw_free`, TD2
`disable_duration_stored_per_branch_matches_sim` [trips on +1/−1 off-by-ones AND on dropping the
branch conditional], TD3 `taunt_plus_disable_forces_struggle`, TD4
`taunt_and_disable_onbeforemove_priority_vs_paralysis` [the cant-vs-para draw ORDER, both
directions — the golden doesn't cover a paralyzed queued move, perturb-checked], TD5
`disable_into_a_zero_pp_lastmove_fails_draws_but_no_volatile` [the onStart 0-PP guard: draws
consumed, volatile REJECTED — unreached by every other gate]; ground truth
`harness/probe_taunt_disable_regression_rng.js` + `probe_disable_zero_pp_rng.js` for TD5). **e2e**: `taunt`/`disable` are in the capstone's
modeled set (`MODELED_RESTRICTION_MOVES`); several gen3ou sample teams carry TAUNT (real filtered-gate
coverage), NO team carries DISABLE → 0 e2e disable decisions (the leech-seed situation — disable is
proven by the dedicated golden + pins). OBSERVATION-ONLY for the pre-existing suites: the engine is
untouched by the e2e admission — the pre-regen e2e golden (13367) replayed byte-identically against
this engine, and battle 2034 / fullbattle 2053 / secondary 4328 / protocol 66/8721 stay byte-identical;
the e2e corpus then legitimately shifts (13561, STRICT 220/220) because the GENERATOR's allow-list
changed which moves are pickable — byte-reproducible at the committed knobs.
**Protocol honesty note — CLOSED by Phase 3** (`gen3_protocol_phase3_v1`): the residual-expiry
**`-end` lines** (`|-end|<mon>|move: Taunt|[silent]` / `|-end|<mon>|Disable`) and the Disable
retro-edit nits are now **EMITTED + byte-verified** vs the dedicated `taunt_lifecycle` /
`disable_lifecycle` capture scenarios: a missed Disable's announce gains the `[miss]` attr
(`attr_last_move_miss`), the no-lastMove & already-disabled fails retro-edit to
`|move|…|Disable||[still]` + `|-fail|<user>`, and the re-Taunt fail is likewise `[still]` +
`|-fail|<user>` (the pre-Phase-3 fail-on-target form was wrong — never byte-gated until now).

## Trapping: the SWITCH-legality gate (per-decision STATE+per-side-TRAPPED+SEED differential)

`gen3_trapping_v1` — **Arena Trap** + **Magnet Pull**, the request-time SWITCH-legality layer (the
switch mirror of the PP/Taunt/Disable move-legality gate): a trapped mon cannot VOLUNTARILY switch
out — the sim's `chooseSwitch` at a `move` request rejects it draw-free ("Can't switch: The active
Pokémon is trapped"). The semantics + draw model were nailed by `harness/probe_trapping_rng.js`
(the sim is the oracle — a source-read hypothesis died here, again):

- **TRAPPED = the sim's `pokemon.trapped` truthiness**, computed at endTurn per active mon
  (battle.ts:1723-1726, right after that mon's `DisableMove` event): both abilities call
  `tryTrap(true)` → `trapped = 'hidden'`. The port computes it LIVE (`turn.rs::is_trapped`) —
  identical at every `move`-request boundary since nothing changes between endTurn and the request.
  Recorded per boundary in `DecisionRecord.trapped` (meaningful at Move boundaries only; a mid-turn
  forced-switch pause has a stale sim flag).
- **Arena Trap** (base-data `onFoeTrapPokemon`): traps GROUNDED foes — Flying-type / Levitate
  escape (gen-3 grounded == not-Flying && not-Levitate, the spikes rule). **A grounded GHOST IS
  trapped** — Showdown-gen3 resolves NO `trapped` type-immunity (the gen3 dex's Ghost
  `damageTaken.trapped` = undefined; the cartridge gen6+ escape doesn't exist here — probed:
  Sableye's switch is rejected). The DUGTRIO MIRROR mutually traps, and (onFoe = 1 handler per
  event) adds **ZERO draws** — probed byte-identical boundary seeds vs a Sand Veil control.
- **Magnet Pull** (gen3-mod `onAnyTrapPokemon`/`onAnyMaybeTrapPokemon` — the handlers are REPLACED,
  the mod-chain caution again): traps STEEL-type foes — groundedness IRRELEVANT (Skarmory,
  Steel/Flying, is trapped). NO self-trap (`isAdjacent(self, self)` is false in singles — the own
  handler registers but no-ops).
- **THE DRAW MODEL (the bit-for-bit crux):** endTurn runs `runEvent('TrapPokemon')` then (gated
  `!knownType || getImmunity('trapped')` — in gen3 ALWAYS true) `runEvent('MaybeTrapPokemon')`
  per active mon, INSIDE the same per-mon loop as `DisableMove` and BEFORE the gen3 quickClawRoll
  (battle.ts:1795). Because gen3 magnetpull is `onAny*`, BOTH actives' Magnet Pulls register on
  EVERY trap event → with ≥2 handlers the sort ties iff the holders' cached `pokemon.speed` are
  equal (abilities share order/priority/subOrder; `effectOrder` is only resolved for
  SwitchIn/RedirectTarget callbacks) → ONE Fisher-Yates draw per tied event. The speed-tied
  **MAGNETON MIRROR draws 4 per endTurn** (2 events × 2 mons — probed 11 draws/turn vs the Sturdy
  control's 7); an Arena-Trap-vs-Magnet-Pull cross at equal speed draws 2 (both events on the MP
  holder: own onAny + foe onFoe); a para that breaks the speed tie silences them (the cached-speed
  model). Modeled by `trap_event_shuffles`, interleaved per mon inside `disable_move_event_shuffle`
  (sim order: DisableMove → TrapPokemon → MaybeTrapPokemon per mon, then Quick Claw). The trapped
  COMPUTATION and the REJECTION are draw-free; the pre-turn-1 endTurn's mirror draws are absorbed
  into the goldens' recorded initSeed.
- **THE GATE** (`move_decision_is_legal`): a scripted `Choice::Switch` by a trapped side is
  REJECTED — the decision is SKIPPED draw-free and the boundary stays open (the same
  reject-and-re-request pattern as the phantom-move/0-PP rejects). Trapping gates ONLY the
  voluntary switch: a PHAZE (Roar) still drags a trapped mon (`drag_in` never consults `trapped` —
  probed), a fainted mon's forced replacement is accepted (the sim's check is
  `requestState === 'move'`-only), and the trapping mon itself switches freely.
- **Request-DISPLAY nuance** (informational; no legality/draw impact, not modeled): `'hidden'`
  trapping shows as `maybeTrapped` in the request JSON until a rejected attempt patches it to
  `trapped: true`, and BOTH flags are omitted when no bench is live (`getMoveRequestData`'s
  `canSwitchIn`/`isLastActive` gates).
- DEFERRED: **Mean Look / Spider Web / Block** (the trapping MOVES — `volatileStatus` status moves,
  fail-loud via the unmodeled-status-move guard) and **Shadow Tag** (unmodeled ability, off the e2e
  filter). See EDGE_CASES.md.

**Validated** by `tests/trapping_test.rs` (a per-seed PER-DECISION
STATE(+STATUS+**per-side TRAPPED**)+SEED+winner differential to GAME-END over **8 scenarios × 80
seeds** in gen3customgame: 640 runs, 5771 seed assertions, 8346 trapped assertions (2631
trapped-side rows, 508 mutual-trap rows), 160 phaze-drags-a-trapped-mon rows, 822 accepted FREE
voluntary switches, 640 wins — the mirrors pin the 4-vs-0 draw models bit-for-bit, incl. Thunderbolt
para breaking the mirror's speed tie mid-run) + 5 revert-verified `tests/regression_test.rs` pins
(T1 `arena_trap_rejects_a_grounded_foes_switch_draw_free`, T2
`arena_trap_does_not_trap_flying_or_levitate`, T3 `magnet_pull_traps_steel_only` [the mirror's
4-draw seeds + the non-Steel control], T4 `roar_drags_a_trapped_mon_out`, T5
`grounded_ghost_is_trapped_by_arena_trap_in_showdown_gen3`; ground truth
`harness/probe_trapping_regression_rng.js`). **e2e**: `arenatrap`/`magnetpull` are in the capstone's
`MODELED_ABILITIES` (the #3/#4 team-carry gaps — filter-clean teams 18 → 22 of 719) and the
generator's voluntary-switch picker respects the sim's `pokemon.trapped`, so trapped mons fight —
see the capstone section for the trapped-decision count. OBSERVATION-ONLY for the pre-existing
suites: no trap ability appears in them → `is_trapped` false + 0/1 trap handlers → byte-identical
(battle 2034 / fullbattle 2053 / secondary 4328 / taunt_disable 4723 / protocol 66/8721 all
unchanged); the e2e corpus legitimately shifts (the generator's allow-list grew) and is
byte-reproducible at the committed knobs.

## Move-coverage BATCH 1: the DRAW-FREE mismodeled post-hit effects (per-decision STATE+HP+BOOSTS+SPIKES+LEECH+ITEM+SEED differential)

`gen3_move_coverage_batch1_v1` — the FIVE highest-frequency SILENT-DESYNC classes from the
move-coverage roadmap (`harness/MOVE_COVERAGE_PLAN.md`): a damaging move that RUNS but drops a
post-hit side-effect. Each is now MODELED bit-for-bit in `run_move`'s landed-hit tail (probes
`harness/probe_batch1_movecoverage.js` / `probe_batch1_order.js` / `probe_batch1_selfdrops_rng.js`
— the mod chain is the only oracle, and it OVERTURNED the naive "self-drop is draw-free"
hypothesis). The tail order mirrors the sim's `moveHit`/`tryMoveHit`: **damage → drain (in the
same `damage()`) → self-drop (`selfDrops`) → secondaries → onAfterHit (item-removal / rapid-spin)
→ recoil (gen3 `tryMoveHit` line 460, LAST)**:

- **RECOIL** (`turn.rs::apply_recoil`) — Double-Edge `recoil:[1,3]` / Take Down / Submission
  `[1,4]`: the USER takes `max(floor(dmgDealt·num/den),1)` HP (the gen3 `calcRecoilDamage`
  `clampIntRange(floor(dmg·num/den),1)`). **Rock Head negates** (its `onDamage` returns null for a
  `recoil` effect — a no-op). Fires whether the mon or a SUBSTITUTE took the hit (`dealt` = the
  actual damage dealt; gen3 `substitute.onTryPrimaryHit` runs the SAME `calcRecoilDamage`).
  DRAW-FREE. Struggle's recoil rides its OWN dedicated `[1,4]` path (so `recoil_fraction` is 0 for
  a Struggle). Emitted `|-damage|<user>|<HP>|[from] Recoil|[of] <target>`.
- **DRAIN** (`turn.rs::apply_drain`) — Absorb / Mega Drain / Giga Drain / Leech Life `drain:[1,2]`:
  the USER heals the fraction of the damage dealt. The **gen<5 floor/ceil split**: non-sub =
  `floor(dmg·num/den)` clamped `>=1` (`battle.ts::damage`); behind a sub = `ceil(dmg·num/den)`
  (`substitute.onTryPrimaryHit`) — so `absorbed` selects the rounding (equal for `[1,2]` + even
  `dealt`). heal-at-full FAILS draw-free. DRAW-FREE. **Liquid Ooze** reverses (the drainer takes
  damage) → FAIL-LOUD (unreachable on the filtered path — a Liquid Ooze target is off the
  allow-list, the Leech-Seed liquidooze deferral). **Dream Eater** is EXCLUDED (its `onTryImmunity`
  sleep-only gate is unmodeled). Emitted `|-heal|<user>|<HP>|[from] drain|[of] <target>`.
- **SELF STAT-DROP** (`turn.rs::apply_self_drops`) — Overheat (self −2 SpA) / Superpower (self −1
  Atk/−1 Def), the top-level `move.self.boosts` (extractor `selfDrops` field, DATA-DRIVEN like
  `selfBoosts`). **THE DRAW (probe-settled — NOT draw-free):** gen3 `selfDrops`
  (battle-actions.ts:1338) draws ONE `random(100)` (the `secondaryRoll`), THEN applies the drop if
  `secondaryRoll < self.chance` OR — Overheat/Superpower have `self.chance === undefined` —
  UNCONDITIONALLY. So the drop ALWAYS lands but the roll is ALWAYS DRAWN (per-call-site PRNG trace
  `probe_batch1_selfdrops_rng.js`: the `random(100)` fires at the `selfDrops` position, after the
  move's own damage + before the foe's move). This is the ONE draw batch-1 adds — and the reason
  the port's Overheat/Superpower were NEVER seed-verified (MISMODELED, skipping it). The port
  draws-then-DISCARDS `random_below(100)` then `boost()`s the user (±6 clamp, our own Clear
  Body/Hyper Cutter never blocks our own drop; fires behind a sub — targets the USER). Emitted
  `|-boost|`/`|-unboost|` by the CLAMPED-applied delta's sign (into-floor delta 0 emits nothing).
- **ITEM REMOVAL** (`turn.rs::apply_item_removal`) — **Knock Off** removes the target's item
  (gen3: NO damage boost — that's gen4+; `onAfterHit`, `item = ""`); **Thief / Covet** STEAL it iff
  the ATTACKER holds NONE (the attacker GAINS the raw item form; Thief also emits the silent
  `-enditem`). **Sticky Hold** BLOCKS all three (`-activate|ability: Sticky Hold`, item unchanged).
  **Mail does NOT block** these three (its `onTakeItem` returns false only for OTHER take-item
  moves). Runs ONLY when the MON was damaged (`!absorbed` — the sim's `onAfterHit` iterates the
  `damagedTargets`, which a sub-absorbed hit leaves empty; the target keeps its item behind a sub).
  DRAW-FREE (the `TakeItem` event / `takeItem` consume no PRNG).
- **RAPID SPIN** (`turn.rs::apply_rapid_spin`) — after a LANDED hit CLEARS the USER's OWN side
  Spikes (`SideState::spikes = 0`) + the USER's Leech Seed (`leech_seed = None`) + partial-trap
  (N/A — no partial-trap move in scope). Carries BOTH `onAfterHit` AND `onAfterSubDamage`, so it
  clears behind a SUBSTITUTE too (`dealt > 0`, mon OR sub — UNLIKE Knock Off's `!absorbed` gate).
  DRAW-FREE. gen3 has only Spikes among the hazards. Emitted `|-end|<user>|Leech Seed|[from] move:
  Rapid Spin|[of] <user>` then `|-sideend|<user-side>|Spikes|[from] move: Rapid Spin|[of] <user>`.

**Validated** by `tests/movecoverage_batch1_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+
BOOSTS+SPIKES-LAYERS+LEECH-SEEDED+ITEM)+SEED+winner differential to GAME-END over 13 scenarios × 80
seeds in gen3customgame: **1040 runs, 10428 decision rows, 10428 seed + 20856 HP + 20856 item
assertions, 4894 item-held + 160 spikes-cleared rows, 1035 wins** — recoil / Rock-Head negation /
drain / self-drop-into-the-floor / Knock Off / Sticky-Hold block / Thief steal / Covet / Rapid Spin
[incl. through a sub] / batch-1-into-a-real-battle) + **8 revert-verified `tests/regression_test.rs`
pins** (MC1 recoil / MC1b Rock-Head / MC2 drain / **MC3 the selfDrops `random(100)`** / MC4
knock-off / MC4b Sticky-Hold / MC5 Thief / MC6 Rapid-Spin — ground truth
`harness/probe_batch1_regression_rng.js`). **THE DRAW-FREE PROOF in the pins:** MC1b/MC2/MC4/MC4b/MC6
ALL share the post-turn seed `4448,587,55846,30246` (recoil/drain/item/rapid-spin add NO PRNG
draw); MC3's seed DIFFERS (`43673,...`) — proving the self-drop `random(100)`. **DATA:** the
extractor emits a `selfDrops` field (only-when-present, obs-neutral, like `selfBoosts`) into
`gen3_moves.json` (Overheat/Superpower/Psycho Boost); `MoveData::self_drops` parses it. **e2e
INCLUDED** — the `MODELED_{RECOIL,DRAIN,SELFDROP,ITEM_REMOVAL,RAPIDSPIN}_MOVES` sets in
`gen_e2e_fuzz.js` admit the moves (the `m.recoil`/`m.drain` rejects become "reject UNLESS in the
modeled set"; a damaging `onAfterHit` move is admitted ONLY if in the item-removal/rapid-spin set;
`m.self.boosts` admitted via the self-drop set; `m.self.volatileStatus` [lockedmove] still
rejected; `knockoff`/`thief`/`covet`/`rapidspin` removed from `MOVE_ID_BLOCKLIST`). OBSERVATION-
NEUTRALITY PROVEN: the pre-regen e2e golden replayed BYTE-IDENTICAL (md5
`a23d77ac60d4af168b8a4428f0b465c9`) against the new engine (the OLD golden carries none of these
moves → the batch-1 tail never fires); the deliberate regen unlocks **719 → 722 filter-clean
teams** (all remaining teams) + shifts the golden md5 to **`dac97afb25317cc9def204ccc9af0e8d`**
(11049 decisions). The other seed suites (battle 2034 / fullbattle 2053 / secondary 4328 /
protocol 66/8721 / writeline / bridge) stay BYTE-IDENTICAL — the batch-1 code is a no-op on any
move not carrying these effects. **The e2e regen surfaced + FIXED TWO real-team-only engine bugs**
the dedicated golden couldn't reach: (1) **the DRAIN CEIL rounding** — the gen<5 sub drain uses
`ceil` but an EXACT even product (68·0.5==34.0) with a float `+1e-9` epsilon wrongly ceil'd to 35
(e2e_33 Giga-Drain-into-a-sub); FIXED by computing recoil/drain via EXACT INTEGER rational math
(`fraction_to_ratio`, no float floor/ceil). (2) **the gen3 `itemKnockedOff` GATE** — a mon whose
item was KNOCKED OFF can neither have its item taken NOR gain one (`takeItem` returns false in
gen≤4 if the target OR source is knocked-off), so a Thief by a Knocked-Off attacker does NOTHING;
the port wrongly stole + healed with the stolen item (e2e_83: a Knocked-Off Skarmory Thief'd a
Leftovers Salamence); FIXED via `MonState::item_knocked_off` (set on Knock Off in
`apply_item_removal`, gated for Thief/Covet), pinned MC7 `knocked_off_attacker_thief_takes_nothing`.
Admitting the batch-1 moves also pulled their handlers (recoil/drain/self/onAfterHit/
onAfterSubDamage) into the handler-audit surface (`gen3_handler_audit_v1`) — the manifest gained 15
rows via `handler_audit_dispositions.js`'s new recoil/drain/self/onAfterHit rules (679 rows).

## Move-coverage BATCH 2: the DRAW-friendly status-move classes (per-decision STATE+HP+STATUS+BOOSTS+WEATHER+SCREENS+SEED differential)

`gen3_move_coverage_batch2_v1` — the FOUR DRAW-friendly category-Status move classes from the
roadmap (`harness/MOVE_COVERAGE_PLAN.md`): STATUS-CURE / WEATHER-SET / STAT-DROP / SCREENS. Each
was probe-settled bit-for-bit vs the omniscient sim (`harness/probe_batch2_movecoverage.js`) and
wired into `run_status_move` (all four are category-Status):

- **STATUS-CURE** (`turn.rs::run_status_move` cure arms — id-gated on `MoveData::cures_self_status`
  / `cures_team_status`): **Refresh** (self-cure — clears **ANY major status EXCEPT sleep / freeze /
  none**, i.e. par / psn / **tox** / brn; the gen3 `onHit` is `if (["","slp","frz"].includes(status))
  return false; cureStatus()` — Toxic IS cured, the case an initial impl missed and the e2e Refresh
  teams surfaced); **Heal Bell** (whole-team major-status cure — active + bench; emits `|-activate|
  <user>|move: Heal Bell` then iterates the team SKIPPING a Soundproof ally [`|-immune|<ally>|[from]
  ability: Soundproof` if active], curing each other ally `|-curestatus|<ident>|<status>|[silent]`,
  bench as a SIDE ref); **Aromatherapy** (whole-team `clearStatus` — a single `|-cureteam|<user>|
  [from] move: Aromatherapy` banner, NO Soundproof gate [Aromatherapy is not a sound move], NO per-mon
  `-curestatus`). All NEVER-MISS → NO accuracy draw; DRAW-FREE (VERIFIED: a cure turn draws only the
  existing action-order / Quick Claw draws). `landed` FALSE.
- **WEATHER-SET** (`turn.rs::modeled_weather_set_move` — Rain Dance → Rain, Sunny Day → Sun): a
  never-miss `target:all` Status move that sets a **5-turn TIMED weather** (`weather_turns = 5`;
  gen3 has no Damp/Heat Rock → always 5), DISTINCT from the PERMANENT ability weather
  (`weather_turns = 0`). VERIFIED bit-for-bit: DRAW-FREE at the move (a distinct-speed set turn draws
  only Quick Claw); the `eachEvent('WeatherChange')` tie-shuffle DOES draw when the two actives TIE on
  cached speed (the shared model with the ability switch-in weather). `field.setWeather` FAILS (returns
  false for a MOVE source when `this.weather === status.id`) into the SAME weather — emits `|-weather|
  <W>` then `|-fail|<caster>`, the weather (incl. its duration) UNCHANGED (a permanent Rain stays
  permanent); a DIFFERENT weather OVERWRITES with the 5-turn timer. The 5-turn UPKEEP (`|-weather|<W>|
  [upkeep]`) + expiry (`|-weather|none`) are the end-of-turn FIELD residual (`apply_weather_chip`): at
  `weather_turns == 1` the weather EXPIRES this turn (emit `none` INSTEAD of the upkeep line) but STILL
  fires the eachEvent('Weather') shuffle (VERIFIED: the expiry turn draws the same count as an upkeep
  turn); `weather_turns > 1` decrements + upkeeps; `weather_turns == 0` (permanent) never decrements.
- **STAT-DROP** (`turn.rs::run_status_move` stat-drop arm — data-driven on `MoveData::stat_drop_boosts`,
  the extractor's `statDropBoosts`): **Screech** (−2 Def) / **Charm** (−2 Atk) / **Metal Sound** (−2
  SpD) / **Feather Dance** (−2 Atk) / **Tickle** (−1 Atk/−1 Def) / **Fake Tears** (−2 SpD) / **Cotton
  Spore** / **Scary Face** (−2 Spe). Draw model: (1) ACCURACY — `randomChance(acc,100)` drawn unless
  never-miss (Screech / Metal Sound / Cotton Spore acc-85 CAN miss; Charm / Feather Dance / Tickle /
  Fake Tears acc-100 but NOT never-miss so they STILL draw ONE roll) — the ONLY per-move draw; (2)
  SOUNDPROOF — Screech / Metal Sound carry `flags.sound` → immune vs a Soundproof holder (accuracy
  drawn, `-immune|[from] ability: Soundproof`); (3) PROTECT + SUBSTITUTE block (non-`bypasssub`; Tickle
  is `bypasssub` so it drops a subbed foe); (4) APPLY `boost()` on the FOE via the shared
  `apply_secondary_boost` (±6 clamp, DRAW-FREE, Clear Body / White Smoke / Hyper Cutter / Keen Eye
  `onTryBoost` gated). Memento is EXCLUDED (its `selfdestruct` faints the user — not a pure stat-drop).
- **SCREENS** (`turn.rs::modeled_screen_move` + `SideState::light_screen`/`reflect` + the
  `build_damage_context` fold): **Light Screen** (halves SPECIAL) / **Reflect** (halves PHYSICAL) — a
  never-miss `target:allySide` Status move setting a **5-turn SIDE condition** (`|-sidestart|<side>|
  move: Light Screen` / `|-sidestart|<side>|Reflect`; gen3 has no Light Clay → always 5). DRAW-FREE
  set; a re-use while ALREADY up FAILS (`|-fail|<caster>`, the timer unchanged). The damage calc
  reads `sides[foe].reflect/light_screen > 0` (`damage.rs::modify_damage` ModifyDamagePhase1 ×0.5,
  crit-bypassed). The 5-turn countdown + expiry (`|-sideend|`) are the end-of-turn SIDE residual
  (`run_residuals`, reflect subOrder 1 / lightscreen 2, DRAW-FREE). **THE DRAW CRUX (the finding that
  cost the effort):** a damaging hit into a side with BOTH Reflect AND Light Screen up draws ONE EXTRA
  `random(0,2)` — the gen3 `modifyDamage`'s `runEvent('ModifyDamagePhase1')` gathers the two screens'
  `onAnyModifyDamagePhase1` handlers, which TIE (both side-condition handlers, speed 0) → a size-2
  Fisher-Yates shuffle (`turn.rs::two_tied_handler_shuffle`, the shared helper with the SetStatus
  clause pair). Drawn in `run_move` AFTER the crit roll, BEFORE the `random(16)` damage roll; ONE
  screen (or none) draws nothing, Flash Fire is in a different tie group (probe-settled). A wrong
  model here desyncs the LCG on every both-screens hit.

**DATA:** the extractor (`tools/pokemon_data_extractor/sync.py::_stat_drop_boosts`) emits a
`statDropBoosts` `{stat:stages}` (negative) map for a pure foe-targeting stat-drop STATUS move
(only-when-present, obs-neutral, like `selfDrops`); `dex/moves.rs::MoveData::stat_drop_boosts` parses
it. Refresh / Heal Bell / Aromatherapy reuse the pre-existing `cures_self_status`/`cures_team_status`
flags; the weather/screen ids are pinned in `turn.rs` (`modeled_weather_set_move`/`modeled_screen_move`).

**Validated** by `tests/movecoverage_batch2_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+BOOSTS+
WEATHER[id+turns]+per-side SCREENS)+SEED+winner differential to GAME-END over 17 scenarios × 80 seeds
in gen3customgame: **1360 runs, 16178 decision rows, 16178 seed + 32356 HP + 16178 weather + 64712
screen assertions**, byte-reproducible) + **9 revert-verified `tests/regression_test.rs` pins**
(MC9-MC17): MC9 `refresh_cures_self_paralysis_draw_free` / MC10
`heal_bell_cures_team_but_skips_a_soundproof_ally` / MC11 `aromatherapy_cures_the_whole_team_draw_free`
/ MC12 `rain_dance_sets_a_five_turn_timed_rain_draw_free` / MC13
`rain_dance_into_an_already_active_rain_fails_draw_free` / MC14
`screech_drops_the_foe_defense_by_two_after_its_accuracy_roll` / MC15
`screech_blocked_by_clear_body_draws_accuracy_but_no_drop` / MC16
`light_screen_sets_a_five_turn_side_condition_draw_free` / **MC17
`double_screen_physical_hit_draws_the_modify_damage_phase1_shuffle`** (the CRUX — captures the
two-screen seed AND asserts it DIFFERS from a one-screen control). Ground truth
`harness/probe_batch2_regression_rng.js`; the draw model settled by
`harness/probe_batch2_movecoverage.js`.

**e2e — ADMITTED (`BATCH2_E2E_EXCLUDED = false` in `gen_e2e_fuzz.js`, 2026-07-12), STRICT clean.** The
engine models all four classes bit-for-bit (the dedicated golden + MC9-MC17 pins), and admitting them to
the e2e capstone surfaced ONE real-team-only divergence — **e2e_182**. It was FIRST described as "a 5-HP
Blissey residual-HEAL-ORDERING gap" (the port appeared to reach full HP one residual tick early), but
root-causing it (`harness/probe_e2e182_simtrace.js` — reseed the sim to the golden init_seed + dump the
per-turn `|-heal|`/`|move|` order — plus a real-battle PP probe) showed it is **NOT a residual-order bug
at all**: it is a **`Pressure` × `allyTeam` PP-DEDUCTION bug** (`gen3_pressure_allyteam_v1`). On the
diverging turn p2 Blissey's Aromatherapy (`target: allyTeam`) was REJECTED by the port as out-of-PP,
so the port's `run_full_battle` skipped that script token, pulled the NEXT one (Blissey's Soft-Boiled),
and ran the WRONG move (SoftBoiled → full HP vs the sim's Aromatherapy → Leftovers 709) — a
decision-count + state desync (rust 161 vs golden 179 decisions). ROOT CAUSE: the port keyed the
Pressure +1 extra PP drop on `!targets_self`, so an `allyTeam` move (Aromatherapy / Heal Bell) under a
Pressure foe wrongly deducted 2 PP → its 8 PP drained early → the port rejected a legitimate late use.
Showdown's rule (`Pokemon.getMoveTargets`, pokemon.ts:854-861): the Pressure foe fires its `onDeductPP`
only when it is in the move's **`pressureTargets`** — a FOE-directed target; `allyTeam` / `self` /
`allySide` / `allies` / `foeSide` never put the foe there (VERIFIED vs a real battle: Aromatherapy under
a Pressure Zapdos = −1, ThunderWave = −2). FIX: `turn.rs::pressure_targets_foe` replaces `!targets_self`,
pinned by revert-verified `regression_test.rs::pressure_does_not_add_pp_for_an_allyteam_move`
(ground truth `harness/probe_pressure_allyteam_rng.js`; the pre-fix predicate gives Aromatherapy 8→6).
With the fix, batch 2 is IN the e2e allow-list: STRICT `filtered_diverged == 0` over 220 battles / 11176
decisions, the regenerated golden **md5 `738da13e9ab666ae50ead17bc6329a08`** (722/722 filter-clean teams;
the handler-audit manifest gained the batch-2 move-class handler rows — 664 → 728, all implemented). The
OTHER seed suites (battle / fullbattle / secondary / protocol / writeline / bridge / leechseed /
substitute / explosion / movecoverage_batch1 / movecoverage_batch2) stay **BYTE-IDENTICAL** — the
Pressure×allyTeam fix is a no-op on any board without an `allyTeam` move used under a Pressure foe
(no constructed golden pairs them), and the batch-2 code is a no-op on any move not carrying these
effects. Probes kept: `harness/probe_batch2_movecoverage.js`, `probe_batch2_regression_rng.js`,
`probe_e2e182_simtrace.js`, `probe_pressure_allyteam_rng.js`.

## Move-coverage BATCH 3: the STATEFUL DRAW-FREE move classes (per-decision STATE+HP+BOOSTS+CURSE+WISH+SUB-HP+SEED differential)

`gen3_move_coverage_batch3_v1` — the THREE STATEFUL move classes CURSE / WISH / BATON PASS, each
probe-settled bit-for-bit vs the omniscient sim (`harness/probe_batch3_{curse,wish,batonpass}.js`) and
wired into `run_status_move` / the residual system / `execute_switch`:

- **CURSE** (`curse`, `curse.onModifyMove` + `onHit`, `MonState::curse: Option<usize>`) — a
  type-conditional move re-targeted at runtime by the USER's type:
  - **NON-GHOST** → `move.self = {boosts:{atk:1,def:1,spe:-1}}` (target SELF). The port applies the
    mixed self-boost {+Atk, +Def, −Spe} — line order `-unboost spe|1`, `-boost atk|1`, `-boost def|1`
    (the −Spe FIRST). **It is NOT draw-free** (the surface's headline probe finding): `move.self`
    rides the gen3 `selfDrops` path (battle-actions.ts:1338, `gen3_move_coverage_batch1_v1`), which
    DRAWS ONE `random(100)` (the `secondaryRoll`) then applies the boost unconditionally (`self.chance
    === undefined`) — exactly like Overheat/Superpower. The −Spe updates `boosts[4]` but NOT
    `cached_speed` (like Dragon Dance) → it can flip the first mover NEXT turn.
  - **GHOST** → lay the `curse` volatile on the FOE (`curse.onStart` → `|-start|<foe>|Curse|[of]
    <user>`) THEN pay `floor(maxhp/2)` HP (a bare `|-damage|<user>|<hp>`, no `[from]`; the volatile is
    laid BEFORE the cost, so a ghost at ≤maxhp/2 lays the curse then faints, foe still cursed). DRAW-FREE.
    A GHOST target is NOT immune (the curse volatile has no type gate). Re-curse into an already-cursed
    foe FAILS ([still]+-fail, no HP cost); curse into a SUBSTITUTE does NOTHING ([still]+-fail — the
    `onModifyMove` deletes the volatileStatus+onHit). The **CURSE RESIDUAL** (`apply_curse`,
    `ResidualAction::Curse`, order 10 subOrder **8** — AFTER Leftovers sub 4 / Leech sub 5 / status DoT
    sub 6, BEFORE Taunt sub 15) chips the cursed foe `floor(maxhp/4)/turn` (`|-damage|<foe>|<hp>|[from]
    Curse`), DRAW-FREE. Clears on switch-out (`execute_switch`) + faint (`process_faints`), like leech.
- **WISH** (`wish`, `slotCondition:'Wish'`, duration 2, `SideState::wish_pending: Option<(u8, String)>`)
  — a slot-keyed DELAYED heal. Cast is never-miss + DRAW-FREE; a 2nd Wish while one is pending FAILS
  ([still], no fail line, existing Wish untouched). The heal fires at `wish.onEnd` — the residual **order
  7** (BEFORE the sand chip order 8 and ALL order-10 handlers — VERIFIED: on the resolve turn `-heal
  Wish` precedes `-heal Leftovers` precedes `-damage brn`; pinned by a LIFE/DEATH order test where the
  order-7 heal saves a low-HP mon from the order-8 sand chip). The Wish handler participates in the
  residual speed-sort with the slot's active mon's cached speed, so TWO Wishes resolving at EQUAL speed
  draw ONE tie-shuffle `random(0,2)` (probe: a Blissey-mirror both-Wish resolve turn draws +1;
  distinct-speed draws none). Heal = `floor(maxhp/2)` of the slot occupant (slot-keyed: survives the
  wisher switching/fainting/being phazed — heals whoever is in the slot; a heal-at-full is SILENT — the
  `if(damage)` guard skips the line). The wisher fainting on the cast turn leaves duration at 2 (the
  residual is skipped), resolving the following turn. NOT cleared on switch (a side/slot condition).
- **BATON PASS** (`batonpass`, `selfSwitch:'copyvolatile'`, `SideState::baton_pass_pending: bool`) — a
  self-switch that PASSES the outgoing mon's boosts + copyable volatiles to the entrant. With NO eligible
  bench it FAILS ([still]+-fail, draw-free, NOT_FAIL). Else it sets the side's `switch_flag` (the runAction
  tail's `switch_request_gate` catches it → the forced switch-in) + the `baton_pass_pending` marker. When
  the forced replacement commits (an instaswitch → `execute_switch`), the port SNAPSHOTS the OUTGOING mon's
  PASS-SET (the 7 boosts + the copyable `noCopy==false` volatiles the port models: **substitute** HP /
  **leech_seed** seeder / **confusion** counter / **curse** source) BEFORE the clearVolatile block, then
  APPLIES it to the entrant AFTER the array swap (major STATUS is NOT a volatile → stays with the outgoing
  mon), and tags the switch line `|switch|…|[from] Baton Pass`. DRAW-FREE (the copy consumes no PRNG; the
  forced switch-in draws exactly like a normal switch). The leech/curse fields keep the seeder/source SIDE,
  so the residual keeps chipping the new mon.

**Validated** by `tests/movecoverage_batch3_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+BOOSTS+
**CURSE**+**WISH-PENDING**+**SUB-HP**)+SEED+winner differential to GAME-END over 16 scenarios × 80 seeds
in gen3customgame: **1280 runs, 4980 decision rows, 1280 wins**, byte-reproducible) + **12 revert-verified
`tests/regression_test.rs` pins** (MC18-MC29): MC18 `curse_non_ghost_self_boosts_atk_def_and_drops_spe`
(the mixed boost + the selfDrops `random(100)` — reverting the draw desyncs the seed), MC19
`curse_ghost_pays_half_hp_and_lays_the_curse_on_the_foe`, MC20 `curse_residual_chips_the_cursed_foe_a_
quarter_maxhp`, MC21 `curse_recurse_into_an_already_cursed_foe_fails_draw_free`, MC22 `wish_heals_half_
maxhp_the_turn_after_cast`, **MC23 `wish_residual_fires_at_order_7_saving_a_low_hp_mon_from_the_sand_chip`
(the CRITICAL order pin — LIFE/DEATH: at order 7 the Wish heals a low-HP mon BEFORE the order-8 sand chip;
at order 11 the sand chip KOs it first → the `!target.fainted` guard skips the heal → the mon dies)**,
MC24 `wish_double_cast_fails_and_the_pending_wish_resolves`, MC25 `wish_is_slot_keyed_and_survives_a_
switch`, MC26 `baton_pass_transfers_the_boosts_to_the_entrant`, MC27 `baton_pass_transfers_the_
substitute_to_the_entrant`, MC28 `baton_pass_transfers_the_leech_seed_to_the_entrant`, MC29
`baton_pass_with_no_bench_fails_draw_free`. Ground truth `harness/probe_batch3_regression_rng.js`; the
draw model settled by `harness/probe_batch3_{curse,wish,batonpass}.js`.

**e2e — ADMITTED (`BATCH3_E2E_EXCLUDED = false` in `gen_e2e_fuzz.js`, 2026-07-12), STRICT clean — a CLEAN
first-try pass, NO new engine bug.** The pre-regen golden replayed BYTE-IDENTICAL (md5
`738da13e9ab666ae50ead17bc6329a08` UNCHANGED — the old golden carries no curse/wish/batonpass, so the
batch-3 code is a no-op) then the deliberate regen shifted it to **md5 `529ab3f0940f8f9cbab383fb26d2a696`**
(722/722 filter-clean teams, STRICT `filtered_diverged == 0` over 220 battles / 11163 decisions). The
OTHER seed suites (battle / fullbattle / secondary / protocol / writeline / bridge / leechseed / substitute
/ explosion / movecoverage_batch1 / movecoverage_batch2) stay **BYTE-IDENTICAL** — the batch-3 code is a
no-op on any move not carrying these effects. The handler-audit manifest grew (728 → **750** rows — the
curse/wish/batonpass move + condition handlers, all implemented). One documented DEFERRAL (a
bridge-request-DISPLAY nuance, not an engine gap): the `|request|` JSON shows a non-ghost holder's Curse
with `target:"self"` (Curse's `nonGhostTarget`) but the port's request serializer renders the base dex
`target:"normal"` — the ENGINE runs Curse bit-for-bit; only the request-JSON `target` field differs (the
bridge_capture scope-audit strips it, like the drawn gender + the `return102` alias). Probes kept:
`harness/probe_batch3_{curse,wish,batonpass}.js`, `probe_batch3_regression_rng.js`.

## Move-coverage BATCH 4: the beforeTurnCallback move classes (FOCUS PUNCH / PURSUIT)

`gen3_move_coverage_batch4_v1` — the TWO gen-3 DAMAGING moves carrying a `beforeTurnCallback` (the
NEW queue-machinery piece the port didn't model): **FOCUS PUNCH** + **PURSUIT**. Both probe-settled
bit-for-bit vs the omniscient sim (`harness/probe_batch4_{focuspunch,pursuit}.js`) and wired into the
turn loop / `run_move` / `execute_switch`. New fields: `MonState::focus_punch: Option<bool>`
(the `focuspunch` volatile, `Some(lost_focus)`), `MonState::pursuit: Option<usize>` (the `pursuit`
volatile on the target, `Some(pursuer_uid)`), `BattleState::pursuit_strike: bool` (the transient
interrupt-strike flag). The NEW **`QAction::BeforeTurnMove { side, uid, move_index }`** (order 5,
between beforeTurn=4 and switch=103) is unshifted by the queue-builder for any move whose id is in
`move_has_before_turn_callback` (focuspunch/pursuit); it runs the `beforeTurnCallback` (draw-free) +
the standard gen<5 trailing `eachEvent('Update')` tail, and participates in the action-order
`speed_sort` at order 5 (so a MIRROR ties → the mirror shuffle).

- **FOCUS PUNCH** (`focuspunch`, 150-BP Fighting, priority -3) — the beforeTurnMove adds the
  `focuspunch` volatile to the USER (`|-singleturn|<user>|move: Focus Punch`, draw-free). A NON-Status
  move that HITS the user DIRECTLY (the `!absorbed` damage-apply block; chip absorbed by the user's OWN
  Substitute does NOT count — the sub intercept precedes the focuspunch onHit) sets `lost_focus`. At the
  FP move's execution the onTry CANCELS it draw-free BEFORE accuracy iff `lost_focus`
  (`|move|…Focus Punch||[still]` + `|cant|…Focus Punch|Focus Punch`, placed after PP/lastMove — the sim
  deducts PP + moveUsed before onTry). The volatile BLOCKS a flinch (`focuspunch.onTryAddVolatile`) — a
  DRAW-RELEVANT gate (a mon with BOTH `focus_punch` + `flinch` would register TWO tied NO_ORDER residual
  duration handlers → a phantom intra-mon tie-shuffle; the flinch-secondary `random(100)` STILL draws,
  draw-then-block). A STATUS hit (Thunder Wave) does NOT set lostFocus (FP lands even while newly
  paralyzed; the para roll still draws on the FP turn). The `duration:1` volatile registers a
  NO_ORDER/subOrder-2 residual duration handler → a FP MIRROR at equal speed adds one residual
  tie-shuffle (the bulky both-FP mirror's +1). Cleared at turn-top (`clear_flinch`) + switch-out + faint.
- **PURSUIT** (`pursuit`, 40-BP Dark SPECIAL, acc 100) — the beforeTurnMove lays the `pursuit` volatile
  on the FOE (skipped if the pursuer is frz/slp). THE INTERRUPT (`condition.onBeforeSwitchOut`, fired by
  `execute_switch`'s `!is_drag` top): a VOLUNTARY switch-out of the pursued mon lets the pursuer cancel
  its queued Pursuit (`queue.retain`, draw-free), deduct its Pursuit PP + set lastMove (draw-free), then
  run Pursuit against the SWITCHING mon at ×2 BP + NEVER-MISS (crit + damage, NO accuracy — the
  `pursuit_strike` flag makes `run_move` double the BP + skip acc/on_before_move/PP/lastMove) BEFORE the
  switch resolves (`|-activate|<switcher>|move: Pursuit`), Choice-locks the pursuer if it holds a Choice
  item, and fires the strike's in-tryMoveHit Update. A KO'd switcher still brings in the replacement (the
  gen 2-4 `-hint`, `process_faints` before the swap). A NORMAL Pursuit (foe stays) is a plain bp-40 hit.
  The interrupt is gated on **`is_voluntary`** (a menu switch, `QAction::Switch`) — matching the sim, which
  runs `runEvent('BeforeSwitchOut')` (hence pursuit's `onBeforeSwitchOut`) ONLY for a menu switch: a phaze
  DRAG (`!isDrag`), a **BATON-PASS selfSwitch** (`batonpass.self.onHit` sets `skipBeforeSwitchOutEventFlag`,
  moves.ts:1109 — the passer is NOT struck, Pursuit runs normally against the ENTRANT next decision), and a
  FAINT replacement (`QAction::InstaSwitch`, its corpse's `pursuit` already cleared) all pass
  `is_voluntary=false` → no interrupt (`gen3_move_coverage_batch4_v1`, the bench-order-desync fix; probe
  `harness/probe_batch4_pursuit_bench_regression_rng.js`, pins MC36/MC36b). On an interrupt turn the pursuer's
  `|move|Pursuit` is emitted BEFORE the switcher's `|switch|`, so the sim reads the pursuer as the first mover
  — the port records `pursuit_first_mover = pside` and `boundary_record` prefers it over the sorted-queue
  first_mover. Cleared at turn-top / consumed by the interrupt / switch-out / faint.

**Validated** by `tests/movecoverage_batch4_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+BOOSTS+
SUB-HP)+SEED+winner differential to GAME-END over **13 scenarios × 80 seeds = 1040 game-end battles,
8866 decision rows, 8866 seed + 17732 HP assertions, 2127 sub-up rows, 1040 wins** — FP land/cancel/
behind-sub/mirror-tie/flinch-hit + Pursuit interrupt/normal/KO-switcher/ghost/into-sub/target-faster/
mirror-tie; the golden REUSES the batch-3 42-field DEC format, CURSE/WISH columns asserted 0) + **6
revert-verified `tests/regression_test.rs` pins** (MC30-MC35): MC30 `focus_punch_cancelled_by_a_prior_
hit_draws_nothing`, MC31 `focus_punch_lands_when_the_user_keeps_focus`, MC32 `pursuit_interrupt_strikes_
the_switcher_at_double_bp_never_miss`, MC33 `pursuit_normal_when_the_foe_stays_is_a_plain_bp40_hit`, MC34
`pursuit_that_kos_the_switcher_still_brings_in_the_replacement`, MC35 `focus_punch_mirror_speed_tie_draws_
the_beforeturnmove_and_residual_ties`. Ground truth `harness/probe_batch4_movecoverage_regression_rng.js`.

Plus **4 revert-verified `regression_test.rs` pins** for the bench-order fix (`gen3_move_coverage_batch4_v1`,
ground truth `harness/probe_batch4_pursuit_bench_regression_rng.js`): **MC36** `pursuit_does_not_intercept_a_
baton_pass_selfswitch` (the passer survives + passes boosts, Pursuit hits the entrant normally), **MC36b**
`pursuit_does_not_faint_a_low_hp_baton_pass_passer` (the fainted-in-a-slot symptom — a low-HP passer stays
alive on the bench), **MC37** `pursuit_interrupt_into_entry_hazards` (VOLUNTARY switch → strike → the
replacement takes Spikes on runSwitch), **MC38** `pursuit_speed_tie_interrupt_draws_the_post_strike_each_event`
(the landed strike's in-tryMoveHit `eachEvent('Update')` draws the pursuer↔switcher tie-shuffle).

**e2e — ADMITTED (`BATCH4_E2E_EXCLUDED = false`, `gen3_move_coverage_batch4_v1`).** STRICT
`filtered_diverged == 0` over the 220-battle gate / **11481 decisions** (242 Focus Punch across 82 battles +
184 Pursuit across 64 battles); the regenerated golden **md5 `fe1529609264be655f36032e0261868d`** (from the
pre-batch-4 `529ab3f0940f8f9cbab383fb26d2a696`). The former deferral — a real-team-only BENCH-ORDER divergence
(e2e_11 rust 34 vs golden 55) — was ROOT-CAUSED as the PURSUIT interrupt firing for a BATON-PASS selfSwitch
(an `InstaSwitch`), striking the still-active passer where the sim leaves it alone (`skipBeforeSwitchOutEventFlag`).
Gating the interrupt on `is_voluntary` (menu switch only) fixed it. Admitting then surfaced + fixed TWO more
real-team-only issues, both bit-for-bit now: (1) a **first-mover attribution** nuance on interrupt turns — the
pursuer's `|move|` emits before the switcher's `|switch|`, so `first_mover` is overridden to the pursuer
(`pursuit_first_mover`, ~15 battles); (2) a **Choice-lock-not-released-on-item-removal** bug (e2e_126) — a
Thief/Knock-Off that removes a mon's Choice Band must clear its `choice_locked_move` (the lock is enforced by
the item's `onDisableMove`), else the port keeps rejecting a now-legal move → a decision-stream desync. All
OTHER seed suites stay BYTE-IDENTICAL (the fixes are no-ops on any board without a pursuit-into-selfswitch or a
Thief'd Choice item). The handler-audit manifest grew 750 → **759 rows** (the FP/Pursuit move + condition
handlers, all implemented). Probes kept:
`harness/probe_batch4_{focuspunch,pursuit,movecoverage_regression_rng,pursuit_bench_regression_rng}.js`.

**BRIDGE-SAFETY (`--use-bridge=rust`): RESOLVED.** The former caveat — the Pursuit-interrupt bench-order bug
sat in an UNCONDITIONAL engine path (`execute_switch`), Pursuit is common in gen3ou (Tyranitar/Metagross),
and `--use-bridge=rust` has no live vs-Showdown check — is CLOSED: the `is_voluntary` gate + the Choice-lock
release make `execute_switch` bit-for-bit on the pursuit-interrupt + Baton-Pass + item-removal compositions, so
a serverless-rust run reaching them now produces correct obs (validated by the STRICT 220/220 e2e).

## Move-coverage BATCH 4b: the last MISMODELED single-turn damaging moves (BEAT UP / THUNDER / WATER SPOUT)

`gen3_move_coverage_batch4b_v1` — the THREE remaining MISMODELED single-turn damaging moves (the
move-coverage roadmap's last silent-desync class), each probe-settled bit-for-bit vs the omniscient
sim (`harness/probe_batch4b_{beatup,thunder,waterspout}.js`) and wired into `run_move` / `run_beat_up`:

- **BEAT UP** (`beatup`, a `basePowerCallback` + `onModifyMove` multi-strike; `run_beat_up`) — ONE
  strike PER healthy (non-fainted, NON-STATUSED) party member of the USER's side, in PARTY ORDER (the
  ACTIVE user itself strikes when healthy; a statused ACTIVE user skips its own strike). Each strike is
  a TYPELESS '???' flat-BP-10 **Special** hit with the STAT SWAP — the attacker's SpA REPLACED by the
  current ally's dex `baseStats.atk` and the defender's SpD by the target's dex `baseStats.def`, both at
  `event.modifier=1` (NO boosts / items / CB / burn / abilities touch the stat), level = the ACTIVE
  user's level (verified: Slaking base-atk 160 vs Skarmory base-def 140 → base 11; Blissey base-atk 10 →
  base 2; vs Gengar base-def 60 → base 24). Typeless → 1× / hits Ghost / no STAB / no weather boost;
  Light Screen (Special) applies + crit ×2. **DRAW MODEL:** ONE whole-move accuracy roll
  `randomChance(100,100)` (acc 100 — drawn, always passes) BEFORE the multi-strike loop, then PER STRIKE
  [crit `randomChance(1,16)`] + [damage `random(16)`] + the gen3 multihit loop's **per-strike
  `eachEvent('Update')`** (scripts.js — drawn on a speed TIE, zero at distinct speed), then the trailing
  in-tryMoveHit Update + the runAction Update (caller, via `landed=true`). The multihit STOPS at the
  target's faint (`i < hits && target.hp` — later strikes + the Quick Claw skip). Each strike routes the
  sub-intercept (a break lets later strikes hit the mon); a direct strike sets the target's Focus-Punch
  **lostFocus** (cancelling a queued FP). The `beatup` `duration: 1` volatile (`MonState::beat_up`)
  registers a NO_ORDER/subOrder-2 residual DURATION handler (the protect/stall/flinch/focus-punch tie
  group) → a BEAT UP MIRROR at equal speed adds ONE residual tie-shuffle draw. A degenerate all-fainted/
  statused party fizzles draw-free (basePowerCallback null).
- **THUNDER** (`thunder`, a 120-BP Special Electric, 30% para, base acc 70; the `onModifyMove` id-gate in
  `run_move`) — the ONLY new draw-relevant piece is the weather-accuracy mutation, folded into the
  effAcc pipeline BEFORE the stage/accMod chain: effective RAIN → `never_miss=true` (the whole accuracy
  block AND its `randomChance` are SKIPPED — ONE FEWER draw), effective SUN → base 50, else (none / sand
  / hail / Cloud-Nine-or-Air-Lock-SUPPRESSED via `effective_weather()`) → base 70. Everything downstream
  is the ordinary 120-BP-special-with-30%-para draw model (crit + damage + secondary `random(100)`). The
  CRUX: rain removes EXACTLY the accuracy draw (a wrong effAcc flips hit/miss AND the draw count).
- **WATER SPOUT** (`waterspout`, a variable-BP Special Water move; the id-gate in `run_move`, Pursuit-`×2`
  precedent) — `bp = max(floor(150·hp/maxhp), 1)` (u32-widened — `150·714` overflows u16), a deterministic
  STATE read of the USER's CURRENT hp computed BEFORE the crit/damage draws → **DRAW-NEUTRAL** (probe:
  full-HP bp 150 and low-HP bp 74 end at the SAME seed; only the damage magnitude differs). At 1 HP the raw
  0.44 floors to min BP 1 (a min-damage HIT, does NOT fail). No secondary; identical draw count/order to Surf.

**Validated** by `tests/movecoverage_batch4b_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+BOOSTS+
SUB-HP)+SEED+winner differential to GAME-END over **14 scenarios × 80 seeds = 1120 game-end battles, 7527
decision rows, 7527 seed + 15054 HP assertions, 387 sub-up rows, 1120 wins** — Beat Up 6-strike / statused-
skip / KO-mid-sequence / into-a-sub / into-a-Ghost / **a speed-tie + a MIRROR at a speed tie** [the
per-strike Update + the residual duration tie]; Thunder rain/sun/none/sand accuracy; Water Spout full/low HP
+ into-a-sub; the golden REUSES the batch-3/4 42-field DEC format, CURSE/WISH columns asserted 0) + **7
revert-verified `tests/regression_test.rs` pins** (MC39-MC45): MC39
`beat_up_full_side_strikes_once_per_healthy_teammate`, MC40 `beat_up_ko_mid_sequence_stops_the_multihit_no_
quick_claw`, MC41 `thunder_rain_never_miss_skips_the_accuracy_draw` (+ a base control, seed-distinct), MC42
`thunder_sun_base_accuracy_fifty_lowers_the_hit_threshold` (a mid-roll seed where sun-50 MISSES while
base-70 HITS), MC43 `water_spout_variable_bp_is_draw_neutral` (full vs low HP, same seed, different damage),
MC44 `beat_up_mirror_speed_tie_draws_the_per_strike_and_residual_shuffles`, MC45
`beat_up_hit_cancels_the_targets_focus_punch`. Ground truth `harness/probe_batch4b_regression_rng.js`.

**e2e — ADMITTED (`BATCH4B_E2E_EXCLUDED = false`, `MODELED_BATCH4B_MOVES = {beatup, thunder, waterspout}`
in `gen_e2e_fuzz.js`; waterspout + beatup removed from `MOVE_ID_BLOCKLIST`).** STRICT
`filtered_diverged == 0` over the 220-battle gate / **11407 decisions** (217 wins + 3 ties); the
regenerated golden **md5 `64edcdcd5c6a63b1256fc23d3887d8c7`** (from the pre-batch-4b
`fe1529609264be655f36032e0261868d`). Admitting these surfaced + FIXED **THREE real-team-only engine bugs**
the distinct-speed dedicated golden couldn't reach, all now bit-for-bit + revert-pinned: (1) the gen3
multihit loop's **PER-STRIKE `eachEvent('Update')`** (e2e_52, Charizard↔Salamence tie — `run_beat_up` fires
`each_event_shuffle` after each strike, not just the caller's single trailing Update via `landed`); (2) the
**`beatup` `duration: 1` volatile's residual DURATION handler** (e2e_217, a Beat Up MIRROR — two beatup
volatiles tie → one residual shuffle; `MonState::beat_up` + the run_residuals gather); (3) **Beat Up sets
the target's Focus-Punch lostFocus** (e2e_196 — a Beat Up into a Focus-Punch user must cancel the FP; the
`!absorbed` direct-strike branch now sets `focus_punch = Some(true)`). All OTHER seed suites (battle /
fullbattle / secondary / protocol / writeline / bridge / every move-coverage batch) stay BYTE-IDENTICAL —
the batch-4b engine changes are id-gated no-ops on any board without beatup/thunder/waterspout. The
handler-audit manifest grew 759 → **767 rows** (the Beat Up multi-strike + stat-swap + duration handlers,
Thunder's weather onModifyMove, all implemented). Probes kept:
`harness/probe_batch4b_{beatup,thunder,waterspout,regression_rng}.js`. **BRIDGE-SAFETY
(`--use-bridge=rust`):** these were the last MISMODELED single-turn damaging moves — a serverless-rust run
reaching Beat Up / Thunder / Water Spout now produces correct obs (validated by the STRICT 220/220 e2e).

## Move-coverage BATCH 4c: the TURN-SPANNING move classes (HYPER BEAM / SOLAR BEAM / DOOM DESIRE + FUTURE SIGHT)

`gen3_move_coverage_batch4c_v1` — the LAST MISMODELED cluster: the moves whose effect SPANS turns
(new cross-turn move state). Probe-settled bit-for-bit vs the omniscient sim
(`harness/probe_batch4c_{hyperbeam,solarbeam,doomdesire}.js`); after this batch the MISMODELED set
is EMPTY (every remaining un-modeled move FAIL-LOUDs).

- **HYPER BEAM** (`hyperbeam`, 150-BP gen3-Physical acc 90; `MonState::must_recharge`) — a
  SUCCESSFUL damaging hit (plain / sub-absorb / sub-BREAK / target-KO; NOT a miss / immune /
  Protect-block — PP consumed either way) applies `mustrecharge` DRAW-FREE (`|-mustrecharge|`,
  before a KO's `|faint|`; the lock PERSISTS across the foe's force-switch). The LOCKED turn: the
  request offers ONLY `{move:"Recharge",id:"recharge"}` (no pp/maxpp/target/disabled) + FIRM
  `trapped:true` (`move 1`/`move recharge` accepted; `move 2` + any switch rejected); the turn is
  spent as `|cant|<user>|recharge` at the user's NORMAL speed-order position (the recharge action
  sorts at priority 0) — ZERO draws + NO PP: the gen3-resolved `mustrecharge.onBeforeMove` at
  priority **11** precedes EVERY status handler, so a par'd locked user draws NO para roll and a
  slp'd one's counter does NOT decrement (`|cant|recharge`, not `|cant|slp`); then the lock fully
  clears. The `duration: 2` volatile registers a NO_ORDER/subOrder-2 residual DURATION handler on
  the CAST turn's residual (the HB-mirror tie draw). TRUANT composes with NO special case in the
  port's `truant_turn` toggle model (the cant precedes the truant gate; the order-27 residual
  toggle consumes the loaf → the probed HB/recharge/HB cadence with no truant cant on the landed
  path; a MISSED HB legitimately loafs the next turn). FAIL-LOUD siblings: blastburn / frenzyplant
  / hydrocannon (gen3-legal, identical machinery, unprobed).
- **SOLAR BEAM** (`solarbeam`, 120-BP Special Grass acc 100; `MonState::two_turn` =
  `TwoTurnMove {move_index, duration, charging}`) — CHARGE turn: onBeforeMove first (a para roll
  IS drawn on the charge turn; a full-para cant = NO charge + NO PP), then PP (−1; **−2 under a
  Pressure foe — Pressure applies at the CHARGE**), then `|move|…||[still]` + `|-prepare|` and the
  `twoturnmove` volatile — ZERO move draws, `landed` false. The volatile registers a
  NO_ORDER/subOrder-2 residual duration handler on BOTH the charge- and fire-turn residuals
  (duration 2 → 1 → 0 removes it; after a fire-turn KO it LINGERS through the faint pause and the
  RESUMED tail's residual cleans it — the next request is fully normal). FIRE turn: the locked
  single-move request (`trapped:true`); NO PP; `removeVolatile('solarbeam')` (`charging` = false)
  → NORMAL execution — accuracy 100 DRAWN → crit → damage, `|move|…|[from]lockedmove`, NO
  secondary. An ABORT on the fire turn (slp/par/frz/flinch cant) fires `onMoveAborted` → the WHOLE
  charge is LOST (a fresh charge re-pays PP); a Protect-blocked fire consumes the charge (accuracy
  drawn, no crit/dmg). SUN (`effective_weather` — a Cloud Nine / Air Lock foe forces the charge
  back even under Drought) SKIPS the charge: `[still]` + `-prepare` + `|-anim|` then an IMMEDIATE
  normal 3-draw execution, PP −1, NO volatile. RAIN / SAND / HAIL **HALVE the BP** (the
  gen3-resolved `onBasePower chainModify(0.5)` — gen3 DOES have the modern halving, probed rain 54
  vs no-weather 105; suppression-aware; read at DAMAGE time; draw-free BP-chain fold). FAIL-LOUD
  siblings: razorwind / skyattack / skullbash + the semi-invulnerable fly / dig / dive / bounce
  (their conditions DIFFER — onInvulnerability etc.).
- **DOOM DESIRE + FUTURE SIGHT** (`doomdesire` bp 120 Physical acc 85 / `futuresight` bp 80
  Special acc 90 — probe-settled the SAME mechanic; `SideState::future_move` = `FutureMove
  {duration, damage, move_id, accuracy, source_side, source_uid}`, the Wish-precedent slot
  condition) — the CAST (`onTry` in `run_move`, BEFORE the protect block — a cast-turn Protect
  does NOT block it): exactly ONE `random(16)` computes the CAST-TIME DAMAGE SNAPSHOT — typeless
  '???' (no STAB / no chart row → NEVER immune: DD is neutral into Fire, FS hits Dark), cast-time
  stats/boosts/items/burn/screens (a both-screens defender side draws the ModifyDamagePhase1
  tie-shuffle like any hit), willCrit false (NO crit roll), NO accuracy roll — then
  `|-start|<caster>|<Name>` and return null (`landed` false). A DOUBLE-CAST (one `futuremove`
  condition per slot, DD-after-FS included) FAILS with a bare `|move|` line, ZERO draws — PP still
  deducted. The pending condition registers an **order-11** residual handler EVERY end-of-turn
  (speed = the slot occupant's cached speed — an equal-speed FS MIRROR draws one tie-shuffle per
  residual; the probed order: Wish 7 → sand 8 → Leftovers 10.4 → **futuremove 11** LAST),
  decrementing 3 → 2 → 1; the 1 → 0 tick RESOLVES (end of turn N+2): skip (draw-free) iff the slot
  occupant is FAINTED (the sim's `-hint` text is uncaptured → deliberately not emitted); else
  `|-end|<target>|move: <Name>`, remove the target's Protect (the strike lands through a
  resolve-turn Protect), ONE accuracy roll (the standard `hitStepAccuracy` fold; the resolve-time
  acc/eva-STAGE fold is probe-UNREACHED and modeled as the standard fold — the honest disposition),
  then the STORED number lands FIXED-DAMAGE-style on WHOEVER occupies the slot (a Substitute
  absorbs with no carry; Focus Band can roll; NO crit / damage roll) + the two `hitStepMoveHitLoop`
  `eachEvent('Update')`s with the in-loop `faintMessages` BETWEEN them — so a resolve KO draws only
  ONE tie-Update (the corpse leaves `getAllActive` before the second; probe-verified on the
  fs_mirror board) and defers the Quick Claw past the forced replacement. A MISS emits
  `|-miss|<caster>|<target>` + the sim's `attrLastMove('[miss]')` retro-append onto the LAST
  `|move|` line of the turn (the observed protocol quirk). The strike resolves even when the CASTER
  switched out or FAINTED (slot semantics — the entrant takes the OLD stored damage). The bridge
  needs NO new request shape for future moves (the class never locks the user).

REQUEST/LEGALITY: `MonState::move_locked()` (mustrecharge ∨ charging) threads through
`choice_is_legal` (only `Move(0)` accepted; a switch rejected — the FIRM-trap shape), the queue
build (the locked `Move(0)` maps to the recharge pseudo-move / the locked Solar Beam slot; no
Struggle substitution; no beforeTurnMove unshift), `sort_actions` (the recharge action sorts at
priority 0), AND `bridge.rs` (`serialize_active` emits the probed single-`{move,id}` +
`trapped:true` request; `resolve_choice` maps the wire's `move recharge`/`move solarbeam` to
`Move(0)`; a locked switch rejects with the `[Invalid choice]` firm form, no re-request — the
locked shapes are probe-derived, not yet byte-gated by a bridge capture scenario, disclosed at the
site).

**Validated** by `tests/movecoverage_batch4c_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+
BOOSTS+SUB-HP+WISH+**FUTURE-PENDING**)+SEED+winner differential to GAME-END over **23 scenarios ×
80 seeds = 1840 game-end battles, 16621 seed + 33242 HP assertions, 3260 future-pending + 197
wish-pending + 579 sub-up rows, 1840 wins** — the DEC format gains 2 per-side FUTURE-PENDING
columns → 44 fields; recharge cycles / KO-locks / statused locked turns / HB+SB mirrors at speed
ties / charge-fire cycles / the sun skip / rain halving / para aborts / fire-into-Protect /
cast-idle-resolve / double-casts / slot switches / resolve KOs / the FS mirror / the
Wish+sand+Leftovers+DD residual-order composition) + **10 revert-verified `regression_test.rs`
pins MC49-MC60** (ground truth `harness/probe_batch4c_regression_rng.js`): MC49
`hyper_beam_hit_locks_then_recharges_draw_free_then_clears`, MC50 `hyper_beam_miss_does_not_lock`,
MC51 `hyper_beam_ko_still_locks_across_the_force_switch`, MC52
`paralyzed_user_on_the_recharge_turn_draws_no_para_roll` (the priority-11 crux: the locked-turn
seed advance is IDENTICAL with and without par), MC53
`solar_beam_charges_then_fires_then_recharges_fresh` (the zero-draw charge / 3-draw fire / PP-once
split), MC54 `solar_beam_sun_skip_fires_immediately`, MC55
`solar_beam_rain_halves_the_bp_state_only` (state-only: rain 54 vs control 105 at BYTE-IDENTICAL
seeds), MC56 `solar_beam_full_para_on_the_charge_turn_no_charge_no_pp`, MC57
`doom_desire_and_future_sight_cast_idle_resolve_snapshot` (identical seed trajectories, damages
366/45 — the snapshot proof), MC58 `doom_desire_double_cast_fails_draw_free_but_deducts_pp`, MC59
`doom_desire_resolve_ko_defers_the_quick_claw`, MC60
`doom_desire_resolves_last_in_the_residual_order` (the LIFE/DEATH-class order pin: Wish +170 →
sand −21 → Leftovers +21 → the strike LAST). The resolve-KO single-Update crux (unreachable at the
pins' distinct speeds) is pinned by the golden's `fs_mirror_tie` scenario (revert-verified: 9 of
the 10 targeted reverts fail their named pin; the inner-faintMessages revert fails the golden).

**e2e — ADMITTED (`BATCH4C_E2E_EXCLUDED = false`, `MODELED_BATCH4C_MOVES = {hyperbeam, solarbeam,
doomdesire, futuresight, recharge}`; `futuresight`/`doomdesire` removed from `MOVE_ID_BLOCKLIST`;
a belt-and-braces `flags.futuremove` reject; the picker treats a locked `trapped:true` REQUEST as
trapped so it never submits a doomed switch).** The pre-regen golden replayed BYTE-IDENTICAL (md5
`64edcdcd5c6a63b1256fc23d3887d8c7` unchanged — the batch-4c code is a no-op on any board without
these moves); the deliberate regen's result is recorded below in the E2E capstone section. Probes
kept: `harness/probe_batch4c_{hyperbeam,solarbeam,doomdesire,regression_rng,fsmirror_debug}.js`.

## Move-coverage BATCH 5: the REACTIVE fixed-damage family + the VARIABLE-BP family + SLEEP TALK

`gen3_move_coverage_batch5_v1` — NINE moves (the top of the greedy team-unlock list), probe-settled
bit-for-bit (`harness/probe_batch5_{reactive,varbp,sleeptalk,reactive_edges}.js` — the edges probe
settled Beat-Up→Mirror-Coat + Struggle→Counter behaviorally):

- **COUNTER / MIRROR COAT** (`MonState::reactive` + `record_reactive_hit` + the
  `run_fixed_damage_move` arms) — the order-5 `beforeTurnMove` volatile (its onStart RESETS
  `{slot:null, damage:0}` EVERY selection turn — prev-turn damage never counts) + the priority-−101
  onDamage RECORDER: 2× each qualifying DIRECT foe **Move** hit — counter `Physical || bare
  hiddenpower`, mirrorcoat `Special && !hiddenpower` (the gen3 TYPE-derived category; the typed-HP
  ids fold via `starts_with("hiddenpower")`), OVERWRITING per hit (a MULTIHIT returns 2× the LAST
  strike — Beat Up's Special strikes arm MIRROR COAT, probed); Seismic-Toss-class fixed damage IS
  Physical → countered; Struggle IS countered; a SUB-absorbed hit never records; recoil/residual/
  confusion self-hits never call the recorder; `dealt` is the post-Focus-Band applied amount (the
  −101 = LAST priority) — and a **ZERO-damage hit NEVER records** (`dealt > 0` gate, the Lens-1
  review fix): a Focus-Band proc on a 1-HP holder reduces the hit to 0 and the sim's
  `runEvent('Damage')` BREAKS its chain on the falsy relayVar BEFORE the −101 recorder, so the
  sim leaves Counter un-armed where an armed-with-`Some(0)` port would draw an extra accuracy
  roll (probe `probe_lens1_batch5_review.js` R3; pin MC77
  `focus_band_zero_damage_hit_does_not_arm_counter`). EXECUTION: un-armed → a **ZERO-DRAW** bare-`|move|` fail (no `-fail`, no
  `[still]`, PP −1); armed → ONE accuracy draw (acc 100, NOT never-miss) then type immunity
  (Fighting→Ghost / Psychic→Dark → `-immune` AFTER the draw), **NO crit / NO damage roll**, the
  stored 2× applied through the normal sub/faint machinery, `landed` true (the in-tryMoveHit Update
  at a tie). `duration:1` → a NO_ORDER/subOrder-2 residual duration handler (the counter-mirror at
  an equal speed = the probed +4 delta: the order-5 pair sort tie + 2 trailing Updates + the
  residual duration tie). Cleared at turn-top (`clear_flinch`) + switch-out + faint.
- **ENDEAVOR** — onTry fails at `hp >= target.hp` (**EQUALITY INCLUDED** — probed 50v50 fails)
  with `|-fail|<user>` + ZERO draws (before accuracy; PP −1); else ONE accuracy draw, Normal→Ghost
  `-immune` after it, then `target.hp − user.hp` lands fixed-damage-style (the delta reads the
  MON's hp behind a sub; the number lands on the SUB, break, NO carry — probed E4).
- **RETURN / FRUSTRATION / FLAIL / REVERSAL / LOW KICK** (`turn.rs::variable_bp`, the BP+category
  override in `run_move`) — engine-computed BP over a bp-0 data row, probe-proven DRAW-NEUTRAL
  (byte-identical seeds across happiness/HP/weight extremes): Return `floor(h·10/25) || 1` (h ≤ 2 →
  the `||1` clamp → BP 1, a normal 4-draw HIT, not a fail); Frustration the 255-mirror; Flail /
  Reversal `ratio = max(floor(48·hp/maxhp),1)` → bands `<2:200, <5:150, <10:100, <17:80, <33:40,
  else 20` (gen3 is 48, NOT gen4's 64; they CAN crit — gen2's willCrit=false is NOT inherited);
  Low Kick the TARGET-weight ladder `≥2000:120, ≥1000:100, ≥500:80, ≥250:60, ≥100:40, else 20` over
  the NEW `SpeciesData::weighthg` (extractor: `gen3_species.json::weighthg` = round(weightkg·10),
  obs-neutral like maxHP; gen3 has NO ModifyWeight). The bp-0 row mis-derived category Status → the
  override re-derives Physical; `MoveData::is_variable_bp` carves the family out of
  `blocked_by_taunt` (probed: a taunted mon keeps Return/Flail/Counter selectable, Sleep Talk
  flips disabled).
- **SLEEP TALK** (the `run_status_move` arm + the `sleep_talk_call` transient +
  `MonState::sleep_skipped`) — the slp onBeforeMove prints `|cant|slp` and **PROCEEDS**
  (`sleepUsable`, id-gated to sleeptalk — Snore stays unmodeled AND the ENGINE fail-louds on it:
  a Snore selection PANICS at `run_move`'s top (`snore_panics_fail_loud`), since running it as a
  plain bp-40 damaging move would silently mismodel both branches [awake: the sim's silent onTry
  fail; asleep: the sim cants-then-PROCEEDS] — the picker blocklist alone was not fail-loud; the
  counter still decrements; `sleep_skipped`++ per proceed, reset to 0 on a normal blocked cant,
  RESTORED `time += skippedTime` at the runSwitch SwitchIn beside the tox reset — live-probed
  3→talk,talk→1,sk2→switch→3; same cancellation law as tox). The arm: onTry = asleep-only (an
  awake/wake-turn use fails SILENTLY — normal self-target announce, nothing else, zero draws);
  onTryHit = the choicelock gate (a PRIOR-turn lock → the `[still]` retro-edit + `|-fail|` BEFORE
  the sample; CB + Sleep Talk works exactly ONCE — the lock records Sleep Talk itself, and the
  lock THIS use just set does NOT count → the `was_choice_locked` pre-move snapshot threaded
  through run_move); onHit = the pool (moveSlots in SLOT ORDER keeping `!no_sleep_talk &&
  !is_charge` — the NEW data-enumerated `noSleepTalk`/`isCharge` move flags from `flags.nosleeptalk`
  / `flags.charge`; NO pp filter, NO disabled/Taunt filter) → **ONE `sample` = `random(n)`, drawn
  EVEN at n = 1** → a 0-PP pick wastes the turn (`|cant|…|nopp|<raw id>`, no further draws) → else
  the picked move runs via a bare `useMove` (the recursive `run_move` under `sleep_talk_call`:
  SKIPS on_before_move / PP — the picked move's PP is NEVER consumed — / lastMove; the FULL normal
  draw chain otherwise; the announce carries the byte-exact `|[from] Sleep Talk` via the
  ProtocolBuilder's one-shot `set_next_move_from`; the resolution PROPAGATES — a landed pick fires
  the caller's in-tryMoveHit Update, a called Roar's drag rides `force_switch_foe`). An
  asleep-called REST silently no-ops (`run_rest`'s asleep guard, BEFORE the full-HP guard — no
  heal, no `random(2,6)`, no counter reset). Empty pool → `[still]` + `-fail`, zero draws.

**Validated** by `tests/movecoverage_batch5_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+
**SLP-COUNTER**+BOOSTS+SUB-HP)+SEED+winner differential to GAME-END over **23 scenarios × 80 seeds
= 1840 game-end battles, 18548 seed + 37096 HP assertions, 3090 asleep rows, 1840 wins — a CLEAN
FIRST-TRY pass**; reuses the batch-4c 44-field DEC format with CURSE/WISH/FUTURE columns asserted
0; INJECT gains a per-slot `pp` set) + **19 revert-verified pins**: MC61-MC75 in
`regression_test.rs` (counter 2× + per-turn reset / wrong-category fails / return-fire immunities
/ sub-not-recorded / Seismic-Toss-countered + Beat-Up-last-strike / the counter-mirror tie draws /
the endeavor delta + equality + ghost + sub / the return-frustration `||1` clamp + draw-neutrality
/ the flail band boundary / the low-kick weight ladder / the n=1 sample + empty pool + the
damaged-sleeper called-Rest no-op / the CB one-use lock / the 0-PP pick / the skippedTime restore),
**MC76** `fixed_damage_hit_cancels_a_queued_focus_punch` (the e2e-surfaced fix below), **MC77**
`focus_band_zero_damage_hit_does_not_arm_counter` (the Lens-1 review fix — the `dealt > 0`
recorder gate; ground truth `probe_lens1_batch5_review.js` R3), **MC78**
`sleep_talk_called_roar_drags_the_foe` (the called-Roar drag composition — the review's coverage
gap; ground truth `probe_batch5_st_calls_roar_rng.js`), and the dex
`batch5_tests` data pin (taunt carve-outs + pool flags + weighthg anchors — probe-verified);
plus the `snore_panics_fail_loud` engine fail-loud unit gate (turn.rs) and a FAIL-LOUD Low Kick
weight read (a missing species / `weighthg == 0` PANICS instead of silently pricing BP 20 — the
resync-regression guard). Ground truth `harness/probe_batch5_regression_rng.js`.

**e2e — ADMITTED (`BATCH5_E2E_EXCLUDED = false`, `MODELED_BATCH5_{REACTIVE,VARBP}_MOVES`; the
batch-5 nine removed from `MOVE_ID_BLOCKLIST` — which ALSO un-shadowed the modeled fixed-damage
five [seismictoss/nightshade/sonicboom/dragonrage/superfang], whose blocklist rows had been
OVERRIDING their documented `MODELED_FIXED_DAMAGE_MOVES` early-admit since the fixed-damage layer
landed; Sleep Talk's pickability is CARRIER-conditional via `sleepTalkPoolModeled` — the CALLED
move bypasses the picker, so the sampled pool must be all-`isModeledMove`. NOTE phaze is
e2e-INCLUDED (`PHAZE_E2E_EXCLUDED = false`), so a Roar-carrying RestTalker PASSES the pool gate —
the called-Roar drag composition is pinned by MC78, not vetoed).** The pre-regen golden replayed
BYTE-IDENTICAL first (md5 `77c9205fef0cc0033a718fe549b4d5ca` unchanged); the deliberate regen +
STRICT result is in the E2E capstone section below. Admitting batch 5 surfaced + FIXED **ONE
real-team-only engine bug** (e2e_202): a FIXED-DAMAGE hit must set the Focus-Punch user's
`lostFocus` (the sim cants the punch after a Seismic Toss; the port's `run_fixed_damage_move`
never set it — LATENT while the fixed-damage family was blocklist-shadowed out of the e2e) — pin
MC76, revert-verified. Probes kept:
`harness/probe_batch5_{reactive,varbp,sleeptalk,reactive_edges,regression_rng}.js`.

## Move-coverage BATCH 6: the FINAL UNMODELED tail (13 status moves)

`gen3_move_coverage_batch6_v1` — ENCORE / DESTINY BOND / ENDURE / PERISH SONG / MEAN LOOK /
SPIDER WEB / BLOCK / BELLY DRUM / CHARGE / MEMENTO / MIMIC / PAIN SPLIT / PSYCH UP, all
probe-settled bit-for-bit (`harness/probe_batch6_{locks,field_trap,utility,dexfacts}.js`).
The full mechanic/draw record is in `harness/MOVE_COVERAGE_PLAN.md` → the BATCH 6 section;
the engine surfaces:

- **State** (`state.rs`): `MonState::{encore: Option<(slot, turns)>, destiny_bond,
  destiny_bond_ko_by, endure, perish: Option<u8>, trapped_by: Option<uid>, charge,
  mimic_overlay: Option<MimicOverlay>}` + `restore_mimic_overlay()`; `move_usable` folds the
  encore lock (every non-encored slot disabled — the request shape rides it for free).
- **Arms** (`turn.rs::run_status_move`): the encore arm (acc draw → protect → the
  already-encored ACC-ONLY fail → `durationCallback random(3,7)` → the onStart rejects
  [no-lastMove / `MoveData::fail_encore` / 0-PP lastMove, draws consumed] → `stored =
  willMove ? rolled : rolled+1`); the ZERO-draw DB / perish / trap / Group-C arms. ENDURE
  rides `run_protect` (the SHARED stall counter + the priority-4 dex row); `endure_clamp`
  fires at every MOVE-damage site (plain / fixed / per-multihit-strike / futuremove).
- **Turn loop**: the encore `onOverrideAction` (a queued different move executes AS the
  encored move — the ENCORED slot's PP deducts) + the CHARGE consumption (onAfterMove /
  onMoveAborted — any executed/aborted move != charge removes it, keyed on the OUTER queued
  move; a pursuit-interrupt bare useMove does NOT consume, faithful to no-runMove-no-AfterMove;
  a Baton Pass consumes it BEFORE the switch, so charge never survives a pass in practice).
- **Residuals** (`run_residuals`): `EncoreDuration` (order 10/subOrder 14 — the tick + the
  0-PP EARLY `-end`) + `Perish` (order 12, LAST) with the sim-faithful **DURATION-END
  `continue`** — the perish onEnd faint is ENQUEUED but the per-handler `faintMessages` is
  SKIPPED (the fieldEvent duration-end branch), so a speed-tied mirror's mutual perish-out is
  a same-residual double faint → the gen-3 TIE (the batch's one first-try pin failure,
  root-caused + revert-verified). The endure volatile registers the NO_ORDER/subOrder-2
  duration handler (the endure+stall intra-mon tie — ONE shuffle on every SUCCESS turn).
- **Faints** (`process_faints`): now a WORKLIST — the DB mutual-faint chain drains the
  killer in the SAME faintMessages pass (|faint| victim → `-activate` → |faint| killer; the
  record `destiny_bond_ko_by` is set ONLY at the Move damage sites, so residual /
  sub-absorbed / futuremove KOs never trigger). Corpse clears extend to the batch-6
  volatiles + the mimic overlay revert + the trap-link end (the latter observationally
  redundant with the replacement-switch clear — documented at the site).
- **Switching** (`execute_switch`): outgoing clears for all batch-6 volatiles + the
  TRAP-LINK source-left clear (the trapper leaving ANY way frees the foe) + the Baton-Pass
  snapshot now passes **perish + trapped_by** (noCopy false — the trap link re-points, the
  entrant is still firm-trapped) alongside the batch-3 set.
- **Legality/bridge**: `is_trapped`/`trap_is_firm` fold `trapped_by` (the FIRM Shadow-Tag
  request shape: `trapped:true` first request, `[Invalid choice]` reject, no re-request) —
  the bridge + choice gates ride the existing predicates unchanged; a trapped mon's Baton
  Pass stays legal (selfSwitch bypasses the trap gate).

**Validated** by `gen_movecoverage_batch6_golden.js` → `movecoverage_batch6_test.rs`
(24 scenarios × 80 seeds = **1920 game-end battles, 22074 per-decision seed assertions,
44148 HP assertions, 1711 encore / 1200 perish / 6479 trapped rows, 1707 wins + 213 ties
— a CLEAN FIRST-TRY pass**; the 44-field DEC format EXTENDED with six columns — per-side
ENCORE duration / PERISH counter / TRAPPED [the live volatile, NOT the sim's endTurn-stale
`pokemon.trapped` flag] → 50 fields; `DecisionRecord` gains `encore`/`perish`) + **21
revert-verified `regression_test.rs` pins MC79-MC99** (ground truth
`harness/probe_batch6_regression_rng.js` — incl. the MC79/MC80 same-seed duration-branch
perturbation pair, the MC83/MC84/MC85 DB trigger split, the MC92 belly-drum 262/263
boundary, and MC98's charge-consumed-by-Baton-Pass proof). **e2e ADMITTED**
(`BATCH6_E2E_EXCLUDED = false`; `destinybond` removed from `MOVE_ID_BLOCKLIST`; the per-DEC
`batch6Move` flag → DEC 39 fields + a GATED `batch6_decisions >= 50` floor; `ab_replay`
accepts 36/38/39): the pre-regen golden replayed BYTE-IDENTICAL first (md5
`614d47b9a5227dc7ad4e444b2e28313c` unchanged, `ab_replay` ok:220/diverged:0), then the
deliberate regen shifted it to **md5 `02fe5d9a59955eaf0360e9d881f46a83`** — STRICT
`filtered_diverged == 0` over **220 battles / 11584 decisions** (218 wins + 2 ties, **58
BATCH6-MOVE decisions** ≥ the gated 50 floor). Admitting batch 6 surfaced + FIXED **ONE
real-team-only engine bug — e2e_7: a CONTACT fixed-damage hit (Seismic Toss into an
EFFECT SPORE Breloom) must fire the defender's contact-proc `onDamagingHit`
(`random(10)` + the sample)** — a LATENT batch-5-era gap (the fixed-damage family was
only e2e-admitted in batch 5 and no such board was sampled until the batch-6 corpus
reshuffle); `run_fixed_damage_move` now calls `apply_contact_proc` for the contact
members (Seismic Toss / Super Fang / Counter / Endeavor), pinned by the revert-verified
**MC99** `fixed_damage_contact_hit_fires_the_contact_proc`. The handler-audit manifest
grew 815 → **889 rows** (the batch-6 move + condition handlers + the `trapped`/`trapper`
conditions, all implemented). The coverage scan:
**718 / 722 teams fully engine-playable** at the batch-6 landing — the residual was **SNATCH**
(4 teams), since MODELED (see the SNATCH section below → **722 / 722**, every `data/teams/` team
fully engine-playable).

## SNATCH: the LAST unmodeled gen-3 status move (→ 722/722)

`gen3_snatch_v1` — **SNATCH** (`snatch`), the SOLE remaining unmodeled gen3ou move (4 teams), now
MODELED bit-for-bit, closing **722/722** — every `data/teams/` team fully engine-playable, the
`--use-bridge=rust` endgame. A Dark, category-Status, **priority +4**, never-miss (`accuracy:true`),
`target:self` move that sets the `snatch` singleturn volatile (`duration:1`); while up, the NEXT
self-targeted `flags.snatch` status move used by the FOE (the only eligible victim in gen-3 singles)
is STOLEN — the snatcher executes it in ITS OWN context and the foe's move does nothing. Probe-settled
bit-for-bit vs the omniscient sim (`harness/probe_snatch.js`; the resolved `snatch.condition` is
`onAnyPrepareHit`, `onAnyPrepareHitPriority = -1`):

- **THE CAST** (`run_status_move`'s snatch-cast arm) — sets `MonState::snatch = true` DRAW-FREE +
  emits `|move|U|Snatch|U` + `|-singleturn|U|Snatch`. `landed` FALSE. Snatch itself is NOT snatchable
  (its flags = `{bypasssub,noassist,failcopycat}`, no `snatch`) → a mirror steals nothing.
- **THE INTERCEPTION** (`run_status_move`, right after the move-announce, gated on the move's
  data-derived `is_snatchable` flag + the FOE's `snatch` volatile) — fires INSIDE the foe's
  `tryMoveHit`, AFTER the foe's `|move|` line + AFTER its PP was deducted (in `run_move`). The exact
  sim ordering: (1) `removeVolatile('snatch')` on the snatcher FIRST (so its own nested `useMove`
  can't re-intercept), (2) `|-activate|SNATCHER|move: Snatch|[of] FOE`, (3) `DeductPP` (draw-free
  no-op — returns true → 0 extra snatch PP in gen3), (4) the snatcher executes the stolen move via a
  recursive `run_status_move` in the snatcher's context (a `set_next_move_from("Snatch")` fold →
  `|move|SNATCHER|Name|SNATCHER|[from] Snatch` + the effect), (5) return null → the foe's move does
  nothing. The stolen self-target moves are all category-Status, so they route through the SAME
  `run_status_move` arms (setup / recovery / Rest / Substitute / team-cure / screens / Belly Drum /
  Charge / Psych Up) — the snatcher gets the boost/heal/sub/status/etc. The VICTIM spends the stolen
  move's PP (deducted before the interception); the SNATCHER spends ONLY its Snatch PP.
- **THE DRAW MODEL** — SNATCH INTRODUCES **ZERO DRAWS OF ITS OWN** (cast + steal are entirely
  draw-free). The stolen move's OWN native draws fire in the snatcher's context (SwordsDance / Recover
  / Substitute = 0 extra; **Rest draws its sleep `random(2,6)`** — the draw-count teeth). Priority +4
  guarantees the volatile is up before ANY foe move even for a SLOW snatcher (SN3 == SN2,
  seed-identical). **THE ONE snatch-attributable draw is the residual duration-handler tie-shuffle a
  MIRROR draws:** the `snatch` `duration:1` volatile registers a NO_ORDER/subOrder-2 residual DURATION
  handler (like protect/flinch/focus-punch/beat-up/endure), so two EQUAL-speed snatchers' volatiles
  TIE → ONE `random(0,2)` tie-shuffle at the residual (**PROBE-VERIFIED 8 draws vs the both-Splash
  control's 7**). The volatile clears at the next turn-top (`clear_flinch`) + switch-out + faint.
- **NON-members** (the sim overturned the task's hypotheses — the sim is the oracle): **Wish** /
  **Spikes** / **Thunder Wave** carry NO `flags.snatch` → they pass through un-stolen; **Snatch itself**
  is not snatchable.
- **DATA**: the extractor emits a data-derived **`isSnatchable`** flag into `gen3_moves.json` from
  `flags.snatch` (only-when-present, obs-neutral — the facade ignores it, like `failEncore`); the 44
  gen3-legal carriers are data-enumerated, never hand-listed. `dex/moves.rs::MoveData::is_snatchable`
  parses it; the engine gates the interception on it, NOT a hard-coded id-list.

**Validated** by `gen_movecoverage_snatch_golden.js` → `movecoverage_snatch_test.rs` (a per-seed
PER-DECISION STATE(+HP+STATUS+BOOSTS+SUB-HP)+SEED+winner differential to GAME-END over 7 scenarios × 80
seeds — steal a self-boost / steal a Recover / steal a Rest [the snatcher sleeps + heals] / steal a
Substitute / a NON-snatchable Thunder Wave NOT stolen + cast-into-nothing / the SNATCH MIRROR residual
tie / snatch-into-a-real-battle; REUSES the batch-6 50-field DEC format so the parser is shared) +
**5 revert-verified `regression_test.rs` pins MC100-MC104** (ground truth
`harness/probe_snatch_regression_rng.js`): MC100 fast-steal-SwordsDance, MC101 slow-steal (the +4
interception proof — SAME post-seed as MC100), MC102 steal-Rest (the snatcher sleeps + full-heals + the
stolen sleep `random(2,6)` draw teeth), MC103 Thunder-Wave-NOT-stolen (passes through, snatcher
paralyzed), **MC104 the MIRROR residual-duration tie** (the CRUX — asserts the 8-draw mirror seed AND
that it DIFFERS from the 7-draw both-Splash control). The former `snatch_status_move_panics_fail_loud`
fail-loud test is re-keyed to `snatch_cast_sets_the_volatile_draw_free`. **e2e ADMITTED**
(`SNATCH_E2E_EXCLUDED = false`; `snatch` removed from the blocklist commentary; the interception steals
only moves the picker also picks — all `isModeledMove` — so the recursion always hits a modeled arm,
never a fail-loud) — a **CLEAN STRICT pass first-try, NO new engine bug**: the pre-regen golden replayed
BYTE-IDENTICAL (md5 `02fe5d9a59955eaf0360e9d881f46a83` unchanged — the snatch code is a no-op on the old
golden), then the deliberate regen shifted it to **md5 `3155eb796cb4bf453c6053d769ba98e5`** — STRICT
`filtered_diverged == 0` over **220 battles / 11575 decisions** (218 wins + 2 ties), **722 / 722
filter-clean teams** (the LAST team-blocking move — the taxonomy's 300-battle UNFILTERED sweep is now
300/300 clean with EMPTY ability+item gap lists). The handler-audit manifest grew 889 → **897 rows** (the
snatch move + `snatch` condition handlers, all implemented). The 220-battle sampled corpus exercises
**ZERO snatch decisions** (like DISABLE — only 4 of 722 teams carry it, and none was drawn active at a
snatch cast), so there is NO snatch e2e coverage floor and the mechanic is proven ENTIRELY by the dedicated
golden (`movecoverage_snatch_test`, 560 runs) + the MC100-MC104 pins — an honest disclosure, the leech-seed
situation. OBSERVATION-ONLY for the OTHER seed suites (battle / fullbattle / secondary / protocol
/ every move-coverage batch stay BYTE-IDENTICAL — the snatch code is an id-gated no-op on any board with
no snatch cast).

## E2E capstone: real teams, full battles, bit-for-bit (per-decision STATE+SEED+winner differential)

This is the closure: instead of constructed scenarios with hand-picked mons + scripted moves, the
capstone drives BOTH engines over **REAL Showdown-export teams** for **complete random battles to
game-end**, asserting per-decision state + status + boosts + confusion + running PRNG seed + winner
**bit-for-bit**. It is the union of every prior layer exercised on production data.

- **The generator** `harness/gen_e2e_fuzz.js` loads all 770 `data/teams/*.txt` (the sample/ +
  others/ pools), imports each with the real `Teams.import`, **validates it under gen3ou** (skips
  rejects/import-fails → 719 valid), and packs it (the EXACT bytes `team::unpack` ingests). From a
  fixed **`MASTER_SEED`** it pairs distinct teams + a battle seed, and at EACH decision reads the sim
  request and picks a RANDOM legal choice from a SEPARATE seeded **choice-RNG** (mulberry32, recorded
  via `MASTER_SEED` so a failing battle re-runs deterministically) — RESTRICTED to mechanics the port
  models (a damaging fixed-BP move with a modeled secondary shape, **OR a modeled standalone status
  move, OR a modeled SELF-BOOST SETUP move**; else a SWITCH; else the battle is forced-unmodeled →
  dropped). It records the SAME per-decision golden as `gen_fullbattle_golden.js` + the boosts/confusion
  of `gen_secondary_golden.js` + a `statusMove` AND a `setupMove` flag per decision (so the run reports
  how many decisions exercise the status-move / setup-move layer), plus the two packed teams + the
  compact choice tokens, to `tests/vectors/e2e_fuzz_golden.txt`.
  **The battle FORMAT is `gen3customgame`** (no clauses → no SetStatus handler-sort shuffle), so the
  Rust gate uses `format_id: "gen3customgame"` (`sleep_clause` OFF) — a status move here draws accuracy
  + apply (+ sleep `random(2,6)`) but NOT the gen3ou SetStatus shuffle.
- **The modeled allow/blocklist** (the FILTERED gate): a move is modeled iff (damaging + fixed BP +
  a modeled secondary shape — none / single status / flinch / confusion / structured stat-boost, or
  Tri Attack) **OR a MODELED STANDALONE STATUS MOVE** (`MODELED_STATUS_MOVES`: Thunder Wave / Stun
  Spore / Glare / Poison Powder / Poison Gas / Toxic / Will-O-Wisp / Spore / Sleep Powder / Hypnosis /
  Sing / Lovely Kiss / Grass Whistle) **OR a MODELED SELF-BOOST SETUP move** (`MODELED_SETUP_MOVES` —
  DERIVED from the Rust data's `selfBoosts` so it's GIGO-proof in lockstep with the engine's
  `self_boost_spec`: Calm Mind / Dragon Dance / Swords Dance / Agility / Bulk Up / Amnesia / Barrier /
  Acid Armor / Iron Defense / Cosmic Power / Tail Glow / Meditate / Sharpen / Howl / Harden / Withdraw /
  Growth) **OR a MODELED SELF-HEAL RECOVERY move** (`MODELED_RECOVERY_MOVES`: Recover / Soft-Boiled /
  Slack Off / Milk Drink / Moonlight / Synthesis / Morning Sun / Rest / Splash — kept in lockstep with the
  port's `recovery_heal_amount` + `run_rest` arms) **OR a MODELED PROTECT move** (`MODELED_PROTECT_MOVES`:
  Protect / Detect — kept in lockstep with `run_protect`; Endure + the gen4+ Quick/Wide Guard / King's
  Shield stay excluded) — `isModeledMove` returns true for a category-Status move iff it is in ANY of the
  four sets; every other Status move — incl. Defense Curl/Minimize/Double Team [volatile/evasion], Belly
  Drum, Curse, **Wish / Heal Bell / Aromatherapy / Refresh / Leech Seed / Endure** — stays excluded), AND
  not `basePowerCallback` /
  `beforeTurnCallback` (Focus Punch) / `beforeMoveCallback` / `priorityChargeCallback` / `damageCallback`
  / `onModifyMove` (Thunder-in-rain) / ohko / multihit / recoil / drain / selfdestruct / forceSwitch /
  a top-level `self.boosts|volatileStatus` (Overheat/Outrage selfDrops) / Hidden Power (variable) / an
  explicit id blocklist; plain `priority` is fine (the port reads it). EVERY mon on BOTH teams must hold a MODELED ability (Intimidate / Sand Stream /
  Drizzle / Drought / Levitate / Flash Fire / Water&Volt Absorb / Thick Fat / Clear Body / White Smoke
  / Hyper Cutter / Keen Eye / Serene Grace / Shield Dust / Own Tempo / **Natural Cure** / **the STATUS_IMMUNE class** (Limber / Insomnia / Vital Spirit / Immunity / Water Veil / Magma Armor — `gen3_status_immune_v1`) / **the BATCH-1 classes** (Shell Armor / Battle Armor [CRIT_IMMUNE] / Chlorophyll / Swift Swim [WEATHER_SPEED] / Cloud Nine / Air Lock [WEATHER_NEGATE] / Speed Boost / Rain Dish [RESIDUAL] — `gen3_ability_batch1_v1`)) OR a provable
  no-op (Pressure / Sturdy / Rock Head / Early Bird / **Lightning Rod / Sticky Hold** [batch-1 class-(a)] / … in a damaging-move-only fuzz — **Plus / Minus moved OUT of the no-op set 2026-07-10**: the gen3 `onModifySpA` scans `getAllActive()` FOES INCLUDED, so a cross-field Plus↔Minus pair is SpA ×1.5; now MODELED, `gen3_plus_minus_v1`), and a MODELED item
  (none / Leftovers / Choice Band / type-boost ×1.1 / Sea Incense / Quick Claw). Real gen3OU teams are
  saturated with Natural Cure / Immunity / berries, so — with **`gen3_natural_cure_v1`** now admitting
  Natural Cure (naturalcure=254, the #1 team-carry gap) — **449 of 719** teams are fully filter-clean (was
  151 after `gen3_sun_freeze_immunity_v1` admitted the 8 DMG_MOD abilities; 22 before that; 18 before
  `gen3_trapping_v1` admitted arenatrap/magnetpull). **Admitting Natural Cure was the BIGGEST single
  admission lever yet (151 → 449)** — the gate pairs those with varied seeds/choices. The MODELED ABILITY
  set ALSO includes the DMG_MOD family (Torrent / Blaze / Overgrow / Swarm / Huge Power / Pure Power / Guts
  / Marvel Scale — `gen3_item_mechanics_v1` ability side, `gen3_sun_freeze_immunity_v1` admission) + the
  ACCURACY family (Compound Eyes / Sand Veil / Hustle — `gen3_accuracy_pipeline_v1`). **Natural Cure** (the
  sole gen-3-REACHABLE SWITCH_OUT-cure ability — Regenerator/Zero-to-Hero also carry `onSwitchOut` but no
  gen≤3 species can have them, so the `naturalcure`-only gate is complete for gen3) is MODELED in
  `turn.rs::execute_switch` — an alive outgoing
  `naturalcure` holder's major `status` is cleared to `None` (voluntary pivot OR phaze-drag; the tox stage +
  sleep counter reset), DRAW-FREE (`onSwitchOut`, `onCheckShow` undefined — the cure + its `[silent]`
  `-curestatus` reveal consume ZERO PRNG, so it is seed-neutral; probe-settled by
  `harness/probe_naturalcure_rng.js`), validated by `tests/naturalcure_test.rs` (280 game-end battles) +
  the NC1-NC3 pins.
- **The gate** `tests/e2e_fuzz_test.rs::e2e_fuzz_golden_matches_showdown` seeds a `BattleState` ONCE
  at the sim's pre-first-decision PRNG state and replays the recorded choices via `run_full_battle`
  WITHOUT re-seeding, asserting per DECISION BOUNDARY to game-end: active species/hp/maxhp/fainted/
  **status** + 5 **boost** stages + **confusion** + pokemon_left + request kind + first mover, AND the
  post-decision PRNG **seed** == the sim's, PLUS the final **winner**. **Result (ALL move layers now
  INCLUDED — `SPIKES`/`SUBSTITUTE`/`EXPLOSION`/`PHAZE` all `*_E2E_EXCLUDED = false` + the
  `gen3_taunt_disable_v1` `MODELED_RESTRICTION_MOVES` admission + the `gen3_trapping_v1`
  `arenatrap`/`magnetpull` ability admission + the `gen3_sun_freeze_immunity_v1` 8-ability DMG_MOD
  admission + the `gen3_natural_cure_v1` **Natural Cure** admission + the `gen3_status_immune_v1`
  6-ability STATUS_IMMUNE admission + the **`gen3_ability_batch1_v1`** batch-1 admission
  (CRIT_IMMUNE `shellarmor`/`battlearmor`, WEATHER_SPEED `chlorophyll`/`swiftswim`, WEATHER_NEGATE
  `cloudnine`/`airlock`, RESIDUAL `speedboost`/`raindish` + the class-(a) no-ops
  `plus`/`minus`/`lightningrod`/`stickyhold`) + the **`gen3_ability_batch2_v1`** batch-2 admission
  (CONTACT_PROC `static`/`poisonpoint`/`flamebody`/`effectspore` + CONTACT-recoil `roughskin` +
  BLOCK `soundproof`/`damp`/`suctioncups` + `synchronize`) + the **`gen3_berry_trace_shedskin_v1`**
  batch-3 admission (the 22 `berryEffect` berries → MODELED_ITEMS + `trace`/`shedskin` →
  MODELED_ABILITIES, `TRACE_COPYABLE` in lockstep) + the **`gen3_ability_batch4_v1`** batch-4
  admission (`truant`/`innerfocus`/`shadowtag`/`cutecharm`/`colorchange` → MODELED_ABILITIES +
  `kingsrock`/`focusband` → MODELED_ITEMS, `TRACE_COPYABLE` in lockstep) + the **MOVE-COVERAGE
  BATCH 1** admission (`gen3_move_coverage_batch1_v1`, 2026-07-12 — the DRAW-FREE post-hit effects
  RECOIL/DRAIN/SELF-DROP/ITEM-REMOVAL/RAPID-SPIN via `MODELED_{RECOIL,DRAIN,SELFDROP,ITEM_REMOVAL,
  RAPIDSPIN}_MOVES`, growing the filter-clean pool **719 → 722**, a CLEAN STRICT pass first-try, NO
  new engine bug; the pre-regen golden replayed BYTE-IDENTICAL [md5 `a23d77ac60d4af168b8a4428f0b465c9`
  UNCHANGED] then the deliberate regen shifted it to **`dac97afb25317cc9def204ccc9af0e8d`**),
  ZERO exclusions remaining): 220
  battles, ALL 220 bit-for-bit clean (`filtered_diverged == 0` over EVERY battle — STRICT, no escape
  hatch), **11049 decisions** to game-end, of which 4210 USE SPIKES, 353 USE PHAZE, 612 USE EXPLOSION,
  343 USE SUBSTITUTE, **114 USE TAUNT**, and **201 involve
  a TRAPPED mon** (an Arena-Trap/Magnet-Pull
  trapped active at a move boundary — the switch-legality fact + the mirror tie-shuffle draws replay
  bit-for-bit on real teams; **0 USE DISABLE** — no sample team carries Disable, the
  honest disclosure; the newly-admitted TRUANT teams are IN the corpus [Slaking's loaf/toggle rows
  replay bit-for-bit], while the 2 Inner-Focus teams are filter-clean but weren't DRAWN into the 220
  sample — Inner Focus is exercised by the dedicated batch-4 golden instead) (plus the
  status/setup/recovery/protect-move coverage + the
  batch-2 CONTACT_PROC / BLOCK / SYNCHRONIZE decisions + the batch-3 berry eats / trace copies /
  shed-skin residual rolls; the
  per-side spikes layers + the sub absorb + the phaze DRAG species + the explosion self-KO + the
  taunted + trapped flags all asserted bit-for-bit via the active species/HP + seed), 218 wins + 2
  ties, 1705
  forced-switch. The decision totals shifted again because admitting the batch-1
  / batch-2 / batch-3 / batch-4 classes (+ move-coverage batch 1) grew the filter-clean team pool
  **525 → 571 → 585 → 712 → 719 → 722 / 722** (shellarmor
  the big batch-1 lever; batch-2 +14, `synchronize` [the #1 taxonomy gap] + `effectspore` the levers;
  **batch-3 +127, the biggest admission since Natural Cure** — `lumberry`=64 + `salacberry`=46 +
  `trace`=9 the levers; **batch-4 +7 — `truant`=4 + `innerfocus`=2 were the LAST team-carry gaps: the
  ENTIRE 719-team real-gen3ou pool is now filter-clean, and the honest taxonomy's 300-battle
  UNFILTERED sweep is 300/300 clean with an EMPTY ability+item gap list for the first time**)
  hence the battle sampling — a CLEAN STRICT pass first-try for ALL FOUR batches, NO new engine bug (the
  STEP-1 weather-eachEvent fix + the DRAW-FREE / draw-bearing class models composed cleanly; the
  batch-2 CONTACT_PROC `randomChance` draws AFTER the move secondary, Damp/Soundproof/Suction Cups draw
  fewer, Synchronize's reflect is draw-free in customgame; the batch-3 eat is DRAW-FREE at the
  Leftovers residual slot, Starf's `sample` + the Figy confusion + Trace's n=1 `randomFoe` + Shed
  Skin's `randomChance(33,100)` the only new draws; the batch-4 Truant loaf/toggle + Shadow Tag +
  Color Change are draw-free-or-fewer while the KR/FB/CC procs ride probed positions — all
  seed-faithful). The golden
  is byte-reproducible at the committed knobs (MASTER_SEED 0x1234abcd, FILTERED_TARGET 220).
  **UPDATE (2026-07-14) — MOVE-COVERAGE BATCH 5 ADMITTED (`BATCH5_E2E_EXCLUDED = false`).**
  Counter / Mirror Coat / Endeavor + Return / Frustration / Flail / Reversal / Low Kick are IN
  the allow-list (`MODELED_BATCH5_{REACTIVE,VARBP}_MOVES`), Sleep Talk is CARRIER-conditionally
  pickable (`sleepTalkPoolModeled`), and the blocklist un-shadowing finally lets the modeled
  FIXED-DAMAGE five be PICKED. After the ONE e2e-surfaced fix (MC76 — a fixed-damage hit sets
  the Focus-Punch `lostFocus`; see the batch-5 section) the regenerated golden is **md5
  `614d47b9a5227dc7ad4e444b2e28313c`** (722/722 filter-clean teams, STRICT
  `filtered_diverged == 0` over **220 battles / 11825 decisions**, 219 wins + 1 tie; **240
  BATCH5-MOVE decisions + 271 FIXED-DAMAGE-MOVE decisions** — the fixed-damage family is
  finally EXERCISED on real teams, closing the old "0 fixed-damage decisions" disclosure.
  The review pass appended two per-DEC columns — `fixedMove[36]` / `batch5Move[37]` (DEC is
  now 38 fields) — so `e2e_fuzz_test.rs` gates **GATED coverage FLOORS**
  `fixed_damage_decisions >= 50` + `batch5_decisions >= 50` (teeth-verified: zeroing the
  flags fails the gate) instead of a generator statistic; the regen was verified
  CONTENT-IDENTICAL to the batch-5 golden (md5 `dd0664015014d7e473e400a29a47eed2`) modulo the
  appended columns + header comments (`ab_replay` accepts both 36- and 38-field DEC rows, so
  saved repro dirs stay replayable) —
  5876 STATUS-MOVE, 1020 SETUP, 916 RECOVERY, 654 PROTECT, 629 PHAZE-move [629 selected; 463
  drag decisions], 240 LEECH, 367 SUBSTITUTE, 140 TAUNT/DISABLE, 366 EXPLOSION; taxonomy
  300/300 clean, EMPTY gap lists). The handler-audit manifest grew 787 → **815 rows** (the
  counter/mirrorcoat conditions + the endeavor/sleeptalk hooks + the un-shadowed `damage`
  declaratives; `sleeptalk` is added to the audit surface EXPLICITLY since `isModeledMove`
  deliberately rejects it). The coverage scan (`scan_move_coverage.js`, classifier refreshed
  for batches 4/4b/4c/5): **662 / 722 teams fully engine-playable**, 0 MISMODELED moves left.
  **UPDATE (2026-07-14) — MOVE-COVERAGE BATCH 4c ADMITTED (`BATCH4C_E2E_EXCLUDED = false`).**
  Hyper Beam / Solar Beam / Doom Desire / Future Sight (+ the locked-turn `recharge` pseudo-move)
  are IN the allow-list (`MODELED_BATCH4C_MOVES`; `futuresight`/`doomdesire` OUT of
  `MOVE_ID_BLOCKLIST`; a belt-and-braces `flags.futuremove` reject keeps any future futuremove out;
  `pickMove` treats a locked `trapped:true` REQUEST as trapped so the capture never stalls on a
  doomed switch). A CLEAN STRICT pass FIRST-TRY: the regenerated golden is **md5
  `77c9205fef0cc0033a718fe549b4d5ca`** (722/722 filter-clean teams, STRICT `filtered_diverged == 0`
  over **220 battles / 11459 decisions**, 217 wins + 3 ties; taxonomy 300/300 clean, EMPTY gap
  lists). The pre-regen golden replayed BYTE-IDENTICAL first (md5 `64edcdcd5c6a63b1256fc23d3887d8c7`
  unchanged). HONEST COVERAGE NOTE: the sampled corpus EXERCISES **3 Solar Beam casts** (e2e_213 —
  a sun/solarbeam team) but **0 Hyper Beam / 0 Doom Desire decisions** — the 1 HB-carrier + 3
  DD-carrier battles never had the carrier mon active at a move boundary (the fixed-damage /
  leech-seed situation) — so HB/DD stay proven by the DEDICATED batch-4c golden (2927 HB / 1069
  future-resolve decisions) + the MC49-MC60 pins. The handler-audit manifest grew 767 → **787**
  rows (the mustrecharge/twoturnmove/futuremove conditions are enumerated explicitly in
  `ENGINE_CONDITIONS`, all implemented).
  **UPDATE (2026-07-12) — MOVE-COVERAGE BATCH 2 ADMITTED (`BATCH2_E2E_EXCLUDED = false`).** The
  cure / weather-set / stat-drop / screen classes are now IN the e2e allow-list too (Refresh / Heal
  Bell / Aromatherapy / Rain Dance / Sunny Day / Screech&c. / Light Screen / Reflect). The result
  block's counts above are the pre-batch-2 golden; with batch 2 admitted the committed golden is
  **md5 `738da13e9ab666ae50ead17bc6329a08`**, **722/722 filter-clean teams**, STRICT
  `filtered_diverged == 0` over **220 battles / 11176 decisions** (5295 STATUS-MOVE, 1077 SETUP,
  1046 RECOVERY, 583 PROTECT, 575 PHAZE, 232 LEECH, 435 SUBSTITUTE, 126 TAUNT/DISABLE, 371 EXPLOSION,
  3725 SPIKES-up, 197 TRAPPED). Admitting batch 2 surfaced ONE real-team-only bug — the
  `gen3_pressure_allyteam_v1` **Pressure × allyTeam PP-deduction** desync (e2e_182, root-caused NOT to
  residual ordering but to the port applying Pressure's −1 to Aromatherapy/Heal Bell — see the batch-2
  move-class section above for the full story + the `pressure_does_not_add_pp_for_an_allyteam_move`
  pin). It was a CLEAN STRICT pass after that one fix; the handler-audit manifest grew to 728 rows
  (the batch-2 move handlers, all implemented). The pre-batch-2 seed suites stay BYTE-IDENTICAL.
  **Admitting Natural Cure surfaced NO new engine bug** — the enlarged 449-clean-team corpus is a CLEAN
  STRICT pass first-try (the cure is a draw-free, well-localized status clear at the switch-out site, so it
  composed cleanly with every existing mechanic; unlike the trapping admission, which surfaced the
  Intimidate-vs-Substitute STATE gap).** A
  `status_present_rows >= 500` floor pins the status-move exercise, a `spikes_decisions >= 50` floor pins
  the Spikes exercise, a `substitute_decisions >= 50` floor pins the Substitute exercise, a
  `taunt_decisions >= 50` floor pins the Taunt exercise (NO disable floor — expected 0), and a
  `trapped_decisions >= 50` floor pins the trapping exercise. **The trapping admission surfaced +
  FIXED one more real-team-only engine bug — gen-3 INTIMIDATE vs SUBSTITUTE** (e2e_171/e2e_204: a
  Jynx that Substituted the turn before a Salamence switch-in kept Atk 0 in the sim while the port
  dropped −1; the gen3 mod's Intimidate SKIPS a subbed foe — `event::intimidate_on_start` now gates
  on the target's `substitute`, probe-verified SEED-NEUTRAL by
  `harness/probe_intimidate_substitute_rng.js`, pinned by the revert-verified
  `regression_test.rs::intimidate_into_a_substitute_is_a_noop`). **LEECH SEED is in the e2e allow-list** (`MODELED_LEECH_MOVES`,
  `LEECHSEED_E2E_EXCLUDED = false`) — its residual is DRAW-FREE so it can't shift the LCG the way the
  phaze `sample` does — and, since the batch-3 admission, the golden finally **EXERCISES it: 354
  LEECH-MOVE decisions** (the old honest "0 leech decisions" disclosure is CLOSED — the gen3ou leech
  users like Celebi/Venusaur pair Leech Seed with berries/Lum, which used to keep them off the
  filtered path). It also stays proven by its DEDICATED golden (`leechseed_test.rs`, 560 runs) + the 3
  regression pins. (`liquidooze` was REMOVED from the harness's `NOOP_ABILITIES` since it is no longer a
  no-op once leech is modeled — it reverses the drain, which the port fail-louds on.) The headline assertion tallies are CLEAN-ONLY: the per-decision loop
  breaks at the first divergence so post-desync rows are never counted (today nothing diverges, so it
  never trips). **The RECOVERY-move expansion surfaced + FIXED one more real-team-only engine bug — the
  RESIDUAL HANDLER GATHER ORDER** (a Gengar-vs-Gengar burn+Leftovers+sand turn where the residual KO'd
  the foe: the speed-sort's tie-group Fisher-Yates shuffle PERMUTES the tied handlers IN THEIR PRE-SORT
  ORDER, so the gather order is load-bearing for the same draw. The port gathered Leftovers BEFORE the
  status DoT per mon, but `findPokemonEventHandlers` gathers STATUS first then ITEM — so a 2-mon DoT tie
  shuffled to the WRONG side-order, running OUR burn DoT before the foe's game-ending DoT aborted the
  residual [we should have stayed un-chipped]. Now the port pushes the StatusDot handler BEFORE
  Leftovers per mon — the subOrder still sorts Leftovers ahead, but the shuffle reads the right order. A
  STATE-only desync the seed match masked). **The prior SETUP-move expansion also surfaced + FIXED — WATER/
  VOLT ABSORB heal is now ACCURACY-GATED** (a missed Hydro Pump into a Water Absorb Politoed wrongly
  healed it `maxhp/4`; the absorb is an `onTryHit` that fires only on a HIT — `run_move` gates
  `apply_absorb_heal` on `acc_hit`; the draw count is accuracy-only either way, so it was a STATE-only
  HP desync the seed match masked). **The prior status-move expansion surfaced + FIXED the TOXIC STAGE
  RESET on switch-in (`tox.onSwitchIn`:
  `effectState.stage = 0`):** a badly-poisoned mon that switched OUT and back was resuming at its prior
  high stage (over-chipping by `~maxhp/16` per missed reset). Caught ONLY because Toxic is now a pickable
  move and mons pivot mid-battle. (Sleep PERSISTS across a switch; only the tox stage resets.)
  **RE-MEANED by fix-queue #4 (`gen3_tox_stage_persists_v1`):** the reset lives in `run_switch`
  (the runSwitch-time `runEvent('SwitchIn')`), NOT `execute_switch`'s array swap — so a replacement
  whose queued runSwitch is CANCELLED by gen3 faint-cancels-all keeps its prior stage (pins TX1/TX2).
- **Engine bugs THIS capstone surfaced + FIXED** (real-team-only, the prior constructed goldens
  couldn't reach): (1) **Water/Volt Absorb HEAL** — the port treated them as bare damage-immunity;
  gen3 heals the defender `floor(maxhp/4)` on the absorbed type (draw-free `onTryHit`), applied at the
  immunity short-circuit (`apply_absorb_heal`) — and, per the SETUP-move expansion, now **accuracy-
  gated** (the heal is an `onTryHit` that fires only on a HIT, so a MISSED Water/Electric move no longer
  heals). (2) **Intimidate vs Clear Body / White Smoke /
  Hyper Cutter** — the lead/switch-in Intimidate ignored the foe's `onTryBoost` Atk-drop immunity (a
  no-op into Metagross etc.); now gated in `event::intimidate_on_start`. (3) **The end-of-turn
  RESIDUAL-vs-FAINT ordering + the cached `pokemon.speed` model** — the prior gap (17/220 weather
  battles, FIXED). (4) **The RESIDUAL HANDLER GATHER ORDER** (the RECOVERY-move expansion, see the
  result bullet above) — the status DoT must be gathered BEFORE Leftovers per mon (mirroring
  `findPokemonEventHandlers`'s status-before-item order) so the tie-group shuffle permutes correctly.
  (5) **The PROTECT residual duration-handlers + the no-delete-on-fail stall + the `willAct()` switch
  gate** (the PROTECT-move expansion) — three interlocking fixes the dedicated protect golden's
  no-weather scenarios couldn't reach: (a) the `protect`/`stall`/**`flinch`** volatiles each register a
  residual duration handler (`findPokemonEventHandlers(..., 'duration')`) that participates in the residual
  speed-sort tie-shuffle — a failed-protect-into-a-RockSlide-flinch ties stall+flinch (flinch was draw-free
  before protect; confusion has NO `duration` so no handler); (b) a FAILED stall roll does NOT delete the
  gen3 (resolved gen5-base) `stall` volatile — the counter PERSISTS (the gen8+-base delete model wrongly
  dropped the stall residual handler); (c) the `willAct()` gate fails a Protect vs a foe switch draw-free.
  All validated by the full suite (no regression).
  (6) **The FORCED-REPLACEMENT `updateSpeed` on commit** (the SPIKES expansion) — `commitChoices()` runs
  `this.updateSpeed()` at its TOP (battle.ts:3020) on EVERY choice commit, **including a mid-turn
  forced-replacement submit**. The port refreshed the cached `pokemon.speed` only at turn-start / residual /
  switch-in, NOT on a forced-replacement commit — so a foe paralyzed MID-TURN kept its STALE turn-start speed
  through the resumed-turn `eachEvent('Update')` tie-shuffles, **spuriously TYING with the fresh entrant**
  (a Jirachi-vs-Jirachi para mirror: the para'd foe must read its para speed 53, not its stale 212, during
  the post-replacement Updates — else 2 phantom tie-shuffle draws desync the seed). The port now calls
  `update_speed()` when a forced replacement commits (mirroring `commitChoices`), before the instaswitch sort
  + the resumed tail's Updates. A SEED-only desync the spikes-modified battle selection exposed (NOT a spikes
  bug — spikes layers were 0 on the switching side; the EntryHazard step was a verified no-op there). All
  validated by the full suite (no regression — fullbattle/secondary/protect forced-replacement parity holds).
- **The end-of-turn residual + cached-speed fix (was the last gap, 17/220; now bit-for-bit).** Two
  intertwined `turn.rs` corrections, both faithful ports of Showdown's control flow (verified against
  `battle.js`'s `fieldEvent('Residual')` + the eachEvent speed semantics, and a per-eachEvent SIM
  probe):
  - **Per-handler `faintMessages` in `run_residuals`.** `fieldEvent('Residual')` runs `faintMessages()`
    after EACH handler and `if (this.ended) return`s — so `fainted` is set BETWEEN handlers (a holder
    fainted by an earlier handler `continue`s its later ones) and a GAME-ENDING residual KO aborts the
    rest, while a NON-game-ending residual faint does NOT abort (the other active still ticks). The port
    applied all handlers in one loop then processed faints once — so a fast burned mon's status-DoT
    self-KO did not stop a slower foe's Leftovers (heal `~maxhp/16` off). (Showdown sorts residual
    handlers by `order→priority→SPEED→subOrder`, so SPEED outranks subOrder — a faster mon's DoT runs
    before a slower mon's Leftovers.) Now mirrors the `while`-loop exactly. `run_turn`'s post-residual
    faint gate was switched from `process_faints()`'s newly-fainted return (now always false — the
    residual already consumed it) to the STATE `any_active_fainted()`, so the trailing Update + Quick
    Claw are still correctly skipped on a residual-faint turn.
  - **The cached `pokemon.speed` model (`MonState::cached_speed`).** The `eachEvent` speed-tie shuffles
    AND the residual handler-sort read `pokemon.speed` — a CACHED value, NOT recomputed live on every
    read. Showdown (re)establishes it to the para/boost-aware `getActionSpeed()` at exactly three sites:
    `commitChoices` (turn start), the `residual` action start, and SWITCH-IN. Between them it goes STALE
    — so a mon paralyzed WHILE already active keeps its turn-start (full) speed through the move-phase
    shuffles and only drops at the residual, while a mon that SWITCHES IN paralyzed ties on its PARA
    speed immediately. The port read the live `effective_speed` everywhere, so its mid-turn shuffle tie
    decisions (hence the Fisher-Yates draw COUNT) desynced from the sim on para/switch turns. Now the
    port caches `cached_speed`, refreshes it at those three sites (`update_speed()` + `execute_switch`),
    and the shuffles/residual-sort read it — bit-for-bit. (Verified vs a per-`eachEvent` SIM probe: a
    Jirachi paralyzed while active shows spd 212 until the residual then 53; a Jirachi that switches in
    paralyzed shows spd 53 immediately.)
- **The coverage taxonomy** `tests/vectors/e2e_fuzz_taxonomy.txt` — a separate UNFILTERED sweep
  (random real-team pairs, NO ability/item pre-filter) that ranks coverage gaps by **STATIC TEAM
  COMPOSITION** — which unmodeled ability/item the PAIRED TEAMS CARRY (`classifyTeamsGaps`), NOT by the
  observed first-divergence cause, and **MOVE-LEVEL-BLIND** (the sweep only ever picks damaging-or-
  switch choices, so status moves / Spikes / Calm Mind are never chosen → never counted). It still runs
  each pair through the sim, but only to drop empty/errored battles — the ranked counts come from the
  static team scan, not the run. **Every random pair is gappy** (real teams always carry an unmodeled
  mechanic), so the ranked counts are the prioritized "which unmodeled mechanic do real teams carry
  most" remaining-work list — except that, post-batch-3, **most random pairs are now CLEAN**: with the
  **`gen3_berry_trace_shedskin_v1`** batch-3 admission (the 22 berries + `trace`/`shedskin`) the sweep
  reads **294 of 300 unfiltered battles filter-CLEAN**, the only remaining ABILITY gaps are
  `truant` (=4) > `innerfocus` (=2), and the ITEM gap list is **EMPTY** — `trace` (was the #1 gap),
  `lumberry` (=64), `salacberry` (=46), `petayaberry` and the whole berry family have ALL DROPPED OFF
  (modeled). Batch-2 already cleared `synchronize`/`effectspore`/`static`/`poisonpoint`/`soundproof`/
  `suctioncups`/`damp`/`roughskin`; earlier levers off: `immunity`/`shellarmor` (batch-1 +
  STATUS_IMMUNE), the DMG_MOD family (`torrent`/`blaze`/`guts`/…, `gen3_sun_freeze_immunity_v1`),
  `magnetpull`/`arenatrap` (`gen3_trapping_v1`), `naturalcure` (`gen3_natural_cure_v1`). The
  `ENGINE_GAP` line reads `none` (the engine is bit-for-bit; the prior weather
  residual-vs-faint gap is FIXED). This does NOT gate `cargo test` (it's the measured coverage map, not a
  pass/fail).
- **Run it:** `node src/rust_sim/harness/gen_e2e_fuzz.js` (env knobs `E2E_FILTERED_TARGET` [default
  **220**, the committed golden's size, so a plain regen reproduces it byte-for-byte], `E2E_UNFILTERED`,
  `E2E_MAX_TRIES`, `E2E_MASTER_SEED`) regenerates both vectors; then `cargo test` re-pins the Rust
  against them. The ignored helpers `e2e_diag` (categorize divergences SEED/STATE/FIRSTMOVER) +
  `e2e_trace_one` (per-decision HP/seed trace, `E2E_TRACE`/`E2E_LO`/`E2E_HI`) are the triage tools used
  to build the allow/blocklist + localize the engine bugs.

## Regression tests (edge cases the e2e fuzz found — pinned deterministically)

The e2e capstone is a SEED-SWEEP over 220 random battles: it FINDS real-team-only engine bugs
bit-for-bit, but each repro is BURIED in the golden (regenerated every layer) — not a STABLE,
NAMED pin. **`tests/regression_test.rs` backfills a DEDICATED, NAMED, SELF-DOCUMENTING regression
test per such bug** — each a CONSTRUCTED scenario (explicit hacked `gen3customgame` teams + an
explicit seed + scripted choices via the public `Battle::start_with_switchins` / `run_turn` /
`run_full_battle` harness, in the style of `tests/residual_faint_test.rs`), so a future change can
never SILENTLY reintroduce the bug. The PRACTICE: **every edge case / engine bug the fuzz surfaces
becomes a dedicated deterministic test here** (or, if the minimal repro needs an irreducibly complex
board, a `# regression:`-named scenario in the relevant golden harness). Each test's NAME + doc
comment states WHICH bug it pins + the WRONG (pre-fix) behaviour; every test was verified a TRUE
PIN by REVERTING its fix and confirming the assertion fails.

Two assertion styles: **STATE pins** (hp/status/boost — no PRNG fragility) and **DRAW-COUNT (seed)
pins** (the post-decision PRNG seed vs the REAL-Showdown ground truth captured by
`harness/probe_regression_rng.js` + `harness/probe_residual_order_rng.js`, whose printed
`seedAfter`s are copied verbatim into the test as constants). The bug → pin map:

| Bug (e2e-found) | Pin (`tests/regression_test.rs`) | Style |
|---|---|---|
| #2 Intimidate respects the foe `onTryBoost` immunity (Clear Body / White Smoke / Hyper Cutter → no-op) | `intimidate_into_clear_body_is_a_noop` | STATE (foe Atk stays 0; non-immune control → −1) |
| #3a residual `faintMessages` PER handler + `if (ended) return` (a fast burned mon's DoT self-KO ends the game before a slower foe's Leftovers heal) | `burn_dot_self_ko_ends_before_foe_leftovers` | STATE (foe HP un-healed) + seed |
| #3b cached `pokemon.speed` for PARALYSIS (para-WHILE-active keeps the full turn-start speed through the move-phase shuffles) | `para_while_active_keeps_full_cached_speed_seed` | SEED (turn-1 para-mid-turn shuffle count) |
| #4 Toxic STAGE RESET on switch-in (`tox.onSwitchIn` → stage 0) | `toxic_stage_resets_on_switch_in` | STATE (status `Toxic(1)` + the maxhp/16 chip, not 6×) |
| #6 residual handler GATHER order (status DoT gathered BEFORE Leftovers per mon, so the tie-shuffle permutes correctly) | `residual_gather_order_status_before_leftovers` | STATE (our HP 284) + seed |
| #7 forced-replacement `updateSpeed`-on-commit (a foe paralyzed mid-turn drops to para-speed before the resumed tail's shuffles) | `forced_replacement_recaches_speed_seed` | SEED (forced-switch boundary draw count) |
| P1 gen-3 PHAZE draws its ACCURACY roll (acc 100, NOT never-miss) THEN the n=1 `sample` (a single eligible mon STILL draws) | `phaze_draws_accuracy_then_n1_sample_seed` | SEED + STATE (the lone bench mon dragged active) |
| P2 a PHAZE with NO eligible foe (its last mon) draws ONLY the accuracy roll (no `sample`) — ONE fewer draw than P1, same init seed | `phaze_fail_draws_only_accuracy_no_sample_seed` | SEED + STATE (no drag, foe stays active) |
| P3 a PHAZE that drags into a 3-layer-Spikes KO faints the dragged mon on entry → forces a NORMAL replacement (the composition) | `phaze_drag_into_a_spikes_ko_chains_a_replacement` | SEED + STATE (a p2 ForceSwitch boundary, pokemon_left 3→2) |
| W the `eachEvent('WeatherChange')` SWITCH-IN tie-shuffle — a mid-turn Sand-Stream/Drizzle/Drought entrant that TIES the opposing active draws ONE `random(0,2)` from `Field.setWeather` (the e2e_84 substitute-surfaced desync; the SAME class as #7) | `switch_into_a_tie_under_sand_draws_the_weather_change_shuffle_seed` | SEED + STATE (Tyranitar active, sandstorm up, p2 took the chip; ground truth `harness/probe_switch_tie_weather_regression_rng.js`) |
| FD1 SEISMIC TOSS deals the USER's LEVEL (100) — accuracy-only draw (NO crit / damage roll) | `seismic_toss_deals_user_level_damage` | STATE (Snorlax −100) + SEED (acc + Quick Claw only) |
| FD2 SEISMIC TOSS (Fighting) into a GHOST is IMMUNE — accuracy-drawn-THEN-`-immune`, SAME draw count as a landed hit | `seismic_toss_into_a_ghost_is_immune_accuracy_only_seed` | SEED (== the landed-hit seed) + STATE (Ghost takes ZERO) |
| FD3 NIGHT SHADE (Ghost) into a NORMAL is IMMUNE (0×) — zero damage, `-immune`, accuracy drawn | `night_shade_into_a_normal_is_immune` | STATE (Normal takes ZERO, not 100) + SEED |
| FD4 a FIXED-DAMAGE move into a SUBSTITUTE hits the SUB (the number → sub HP, breaks no carry), Super Fang still halves the MON's hp | `fixed_damage_into_a_substitute` | STATE (sub 131→31, mon HP unchanged) + SEED |
| TD1 a landed TAUNT restricts the target's Status moves for EXACTLY the sim's window (queued move cant'd draw-free + no PP; ONE restricted selection; the freed Thunder Wave then paralyzes = the free-up proof) | `taunt_blocks_status_move_selection_for_the_sim_window_draw_free` | STATE (`move_usable`/PP/par) + SEED |
| TD2 the DISABLE stored duration per branch — FASTER disabler (willMove TRUE) = `random(2,6)`, SLOWER = `random(2,6)+1` — pinned AT the exact free-up boundary (trips +1/−1 off-by-ones AND the dropped-branch model) | `disable_duration_stored_per_branch_matches_sim` | STATE (the per-boundary `disabled_slot` timeline) + SEED (all 5 boundaries, both branches) |
| TD3 DISABLE (the only attack) + TAUNT (the Status moves) → FORCED Struggle (the sim's request offers ONLY Struggle) | `taunt_plus_disable_forces_struggle` | STATE (`must_struggle` + the Struggle HP/recoil floor(15/4)=3) + SEED |
| TD4 the `onBeforeMove` PRIORITY ordering vs PARALYSIS — a taunted+paralyzed queued status move DRAWS the para roll BEFORE the taunt cant (taunt priority 0 < par 1); a paralyzed+disabled queued move is cant'd with NO para roll (disable 7 > par 1). The 720-run golden does NOT cover the paralyzed case (perturb-checked), so this pin is the ONLY ordering gate | `taunt_and_disable_onbeforemove_priority_vs_paralysis` | SEED (both directions; in-engine Thunder Wave para, no injection) |
| TD5 Disable into a **0-PP lastMove** — the gen4-inherited onStart 0-PP guard REJECTS the volatile AFTER the accuracy + `random(2,6)` draws (draws consumed, `-fail` + `[still]` retro-edit, NO volatile/`-start`/residual handler). Reviewer-probed, unreached by every other gate (organically reachable: last PP spent → Struggle-forced → Disabled); pre-fix the port recorded a PHANTOM volatile (state wrong + the phantom's duration handler TIES the live `stall` handler at dec8's residual → an extra tie-shuffle draw → the boundary seed desyncs — reviewer-verified, so BOTH assert families carry teeth) | `disable_into_a_zero_pp_lastmove_fails_draws_but_no_volatile` | STATE (`disabled_slot` −1 / `disable` None at every boundary + the Struggle HP/recoil) + SEED (all 10 boundaries; only the disable-ARM draw-count [acc + `random(2,6)`] is bug-invariant) |
| T1 TRAPPING (`gen3_trapping_v1`) — a trapped mon's voluntary switch is REJECTED draw-free (the switch mirror of the reject-and-re-request gate: the scripted `Switch` decision is SKIPPED, boundary open, seed untouched; Arena Trap adds ZERO endTurn draws) | `arena_trap_rejects_a_grounded_foes_switch_draw_free` | STATE (`is_trapped` / species / `trapped` columns) + SEED |
| T2 Arena Trap does NOT trap Flying (Zapdos) / Levitate (Gengar) — their voluntary switches are ACCEPTED (gen-3 grounded == not-Flying && not-Levitate) | `arena_trap_does_not_trap_flying_or_levitate` | STATE (species churn + trapped false) + SEED |
| T3 Magnet Pull traps STEEL only — the MAGNETON MIRROR mutual-traps AND draws the endTurn TrapPokemon+MaybeTrapPokemon tie-shuffles (gen3 magnetpull is `onAny` → 2 tied handlers per event → **4 draws/endTurn** in the speed-tied mirror); the non-Steel control switches out freely | `magnet_pull_traps_steel_only` | STATE (mutual `trapped` + the control's free switch) + SEED (the 4-draw splash-turn seeds) |
| T4 Roar DRAGS a trapped mon out — phaze BYPASSES trapping (only the VOLUNTARY switch is gated; `drag_in` never consults `trapped`) | `roar_drags_a_trapped_mon_out` | STATE (`phaze_drag` + the dragged species + the entrant trapped in turn) + SEED |
| T5 a grounded GHOST (Sableye) IS trapped in Showdown-gen3 — NO `trapped` type-immunity in the gen3 dex (the cartridge gen6+ escape does not exist here; a modern-gen Ghost escape is WRONG for this sim) | `grounded_ghost_is_trapped_by_arena_trap_in_showdown_gen3` | STATE (trapped + Sableye held) + SEED |
| FZ1 SUN → freeze immunity (`gen3_sun_freeze_immunity_v1`) — the base `sunnyday` weather's `onImmunity('frz')` blocks a freeze while the field is Sun (Drought / Sunny Day), at `runStatusImmunity` (BEFORE `runEvent('SetStatus')`), DRAW-FREE. WRONG (pre-fix): the port froze the mon (the A/B "ice-freeze cluster", 196 repros, seed matching). The freeze secondary's `random(100)` STILL draws (an already-frozen mon PERSISTS under sun — application-only gate) | `sun_blocks_freeze_secondary_draw_free` | STATE (Groudon un-frozen under Drought) + SEED (== the freeze-lands control, draw-free) |
| FZ2 MOVE-ID ALIAS (`gen3_move_alias_resolution_v1`) — a packed team CAN carry a move alias (`wisp` for Will-O-Wisp; the sample pool does), which Showdown resolves at `dex.moves.get()` and RUNS. WRONG (pre-fix): the port's `move_at → dex.moves("wisp")` returned `None` → `run_move` NO-OP'd the move drawing NOTHING while the sim ran it (a draw-count desync that cascaded the e2e_86 decision boundaries). Fix: the Rust dex resolves aliases via `gen3_move_aliases.json` | `move_alias_wisp_resolves_and_runs_will_o_wisp` | STATE (foe BURNED = the aliased move ran) + SEED (the accuracy roll drew) |
| MC1 RECOIL (`gen3_move_coverage_batch1_v1`) — Double-Edge recoils `floor(dmgDealt/3)` to the USER (`recoil:[1,3]`), DRAW-FREE. WRONG (pre-fix): the recoil was never applied | `double_edge_recoils_a_third_of_the_damage_dealt` | STATE (Tauros HP incl. the recoil) + SEED |
| MC1b ROCK HEAD negates Double-Edge recoil — the USER takes NO recoil. WRONG (a model that applied recoil regardless of ability): Aggron's HP lower | `rock_head_negates_double_edge_recoil` | STATE (Aggron near-full) + SEED (== the SHARED draw-free seed `4448,...`) |
| MC2 DRAIN — Giga Drain heals the USER `floor(dmgDealt/2)` (`drain:[1,2]`), DRAW-FREE. WRONG (pre-fix): the drain heal was never applied | `giga_drain_heals_half_the_damage_dealt` | STATE (the injured Sceptile heals) + SEED (== the SHARED draw-free seed) |
| MC3 SELF-DROP + the `selfDrops` DRAW — Overheat drops SpA −2 AND gen3 `selfDrops` DRAWS ONE `random(100)` (applied unconditionally, `self.chance === undefined`). WRONG (pre-fix): skipping the draw gives the shared draw-free seed → desync | `overheat_self_drops_spa_and_draws_the_selfdrops_random_100` | STATE (Charizard spa −2) + SEED (a DIFFERENT seed `43673,...` from the draw-free scenarios) |
| MC4 ITEM REMOVAL — Knock Off removes the TARGET's item (`onAfterHit`, gen3 no dmg boost), DRAW-FREE. WRONG (pre-fix): the item was never removed | `knock_off_removes_the_targets_item_draw_free` | STATE (Snorlax item gone) + SEED (== the SHARED draw-free seed) |
| MC4b STICKY HOLD blocks Knock Off — the target KEEPS its item. WRONG (a model ignoring Sticky Hold): the item removed | `knock_off_blocked_by_sticky_hold` | STATE (Muk keeps Leftovers) + SEED (== the SHARED draw-free seed) |
| MC5 THIEF steals (attacker itemless → the attacker GAINS the item, the target LOSES it), DRAW-FREE. WRONG (pre-fix): no steal | `thief_steals_the_targets_item_when_attacker_is_itemless` | STATE (Gengar gains / Snorlax loses Leftovers) + SEED |
| MC6 RAPID SPIN clears the USER's own Spikes + Leech Seed (`onAfterHit`+`onAfterSubDamage`), DRAW-FREE. WRONG (pre-fix): the hazards/leech persisted | `rapid_spin_clears_the_users_own_spikes_and_leech_seed` | STATE (p1 spikes 3→0 + leech cleared) + SEED (== the SHARED draw-free seed) |
| MC7 gen3 `itemKnockedOff` — a Knocked-Off mon can neither have its item taken NOR gain one; a Thief by a Knocked-Off attacker does NOTHING. WRONG (pre-fix): the port stole + healed with the stolen item (the e2e_83 real-team bug) | `knocked_off_attacker_thief_takes_nothing` | STATE (Skarmory stays itemless / Snorlax keeps its item) + SEED |
| MC8 RECOIL is computed on the POST-Focus-Band damage (`move.totalDamage`) — a FB save on a recoil KO recoils `floor((hp-1)/den)`, not `floor(hp/den)`. WRONG (pre-fix, code-review-found): `dealt` was captured before the FB survive-at-1 reduction → over-recoil (also fixes Struggle's recoil) | `recoil_is_computed_on_the_post_focus_band_damage` | STATE (Tauros 272, not 271) + SEED |
| MC9 REFRESH self-cures ANY major status EXCEPT slp/frz/none (`gen3_move_coverage_batch2_v1` — par/psn/**tox**/brn; Toxic IS cured), DRAW-FREE. WRONG (pre-fix): the cure arm missing → Vaporeon stays paralyzed | `refresh_cures_self_paralysis_draw_free` | STATE (Vaporeon un-par'd) + SEED (draw-free) |
| MC10 HEAL BELL cures the WHOLE team (active + bench) but SKIPS a Soundproof ally, DRAW-FREE. WRONG (pre-fix): the team-cure missing → active tox persists; WRONG (no Soundproof gate): the bench Electrode's par is cured too | `heal_bell_cures_team_but_skips_a_soundproof_ally` | STATE (Miltank un-tox'd + the Soundproof Electrode KEEPS its par) + SEED |
| MC11 AROMATHERAPY cures the whole team via `clearStatus` (no Soundproof gate — not a sound move), DRAW-FREE | `aromatherapy_cures_the_whole_team_draw_free` | STATE (active + bench both cleared) + SEED |
| MC12 RAIN DANCE sets a 5-turn TIMED Rain (distinct speed → DRAW-FREE); the 5-turn timer ticks once at the field residual → 4 remaining | `rain_dance_sets_a_five_turn_timed_rain_draw_free` | STATE (weather Rain, turns 4) + SEED (no WeatherChange shuffle at distinct speed) |
| MC13 RAIN DANCE into an ALREADY-active Rain FAILS (`setWeather` false for a MOVE source into the same weather), DRAW-FREE, the (permanent) weather UNCHANGED | `rain_dance_into_an_already_active_rain_fails_draw_free` | STATE (weather still permanent Rain, turns 0) + SEED |
| MC14 SCREECH drops the foe's Def by 2 (`statDropBoosts {def:-2}`) after its accuracy roll | `screech_drops_the_foe_defense_by_two_after_its_accuracy_roll` | STATE (Snorlax Def −2) + SEED (the accuracy roll drew) |
| MC15 SCREECH BLOCKED by Clear Body — the accuracy roll is STILL drawn, no drop lands. WRONG (a model ignoring Clear Body): Metagross Def −2 | `screech_blocked_by_clear_body_draws_accuracy_but_no_drop` | STATE (Metagross Def stays 0) + SEED (== a landed drop's seed) |
| MC16 LIGHT SCREEN sets a 5-turn SIDE condition, DRAW-FREE; the side residual ticks it once → 4 remaining | `light_screen_sets_a_five_turn_side_condition_draw_free` | STATE (p1 light_screen 4) + SEED |
| MC17 **the DOUBLE-SCREEN ModifyDamagePhase1 SHUFFLE (the CRUX)** — a physical hit into a side with BOTH Reflect AND Light Screen up draws ONE extra `random(0,2)` (the 2 screen `onAnyModifyDamagePhase1` handlers TIE → a size-2 shuffle). WRONG (pre-fix): NOT drawn → the seed matched the ONE-screen control. Pins the two-screen seed AND asserts it DIFFERS from the one-screen control | `double_screen_physical_hit_draws_the_modify_damage_phase1_shuffle` | SEED (both-screen ≠ one-screen — the extra draw) |
| PA1 **Pressure × `allyTeam` PP-deduction** (`gen3_pressure_allyteam_v1`, the **batch-2 e2e admission** bug e2e_182) — an `allyTeam` move (Aromatherapy / Heal Bell) under a **Pressure** foe deducts ONE PP, not two (the Pressure extra fires only when the foe is in the move's `pressureTargets` — a FOE-directed target; `allyTeam`/`self`/`allySide`/`allies`/`foeSide` never put the foe there). WRONG (pre-fix `!targets_self`): Aromatherapy 8→6 under Pressure → drained early → the port rejects a legit late Aromatherapy as out-of-PP → the script shifts (e2e_182 decision-count + state desync). Pins Aromatherapy −1 (8→7) + a ThunderWave −2 (32→30) control, both DRAW-FREE seeds | `pressure_does_not_add_pp_for_an_allyteam_move` | STATE (PP 8→7 vs 8→6) + SEED (both draw-free) |
| MC18 CURSE non-ghost self-boost {atk:+1, def:+1, spe:-1} (`gen3_move_coverage_batch3_v1`) — the `move.self` rides the gen3 `selfDrops` path → DRAWS ONE `random(100)` (NOT draw-free, like Overheat). WRONG (removing the curse arm): Snorlax un-boosted / the fail-loud panic; WRONG (skipping the selfDrops draw): the seed desyncs | `curse_non_ghost_self_boosts_atk_def_and_drops_spe` | STATE (the mixed +/- boosts) + SEED (the selfDrops draw) |
| MC19 CURSE ghost pays floor(maxhp/2) HP + lays the `curse` volatile on the FOE, DRAW-FREE | `curse_ghost_pays_half_hp_and_lays_the_curse_on_the_foe` | STATE (Gengar −maxhp/2; foe cursed) + SEED |
| MC20 CURSE residual chips the cursed foe floor(maxhp/4)/turn (order 10 subOrder 8), DRAW-FREE | `curse_residual_chips_the_cursed_foe_a_quarter_maxhp` | STATE (the maxhp/4 chip per turn) + SEED |
| MC21 CURSE re-curse into an ALREADY-CURSED foe FAILS ([still]+-fail, no HP cost), DRAW-FREE | `curse_recurse_into_an_already_cursed_foe_fails_draw_free` | STATE (no 2nd HP cost; foe still cursed) + SEED |
| MC22 WISH heals floor(maxhp/2) at the END of the turn AFTER cast (N+1), DRAW-FREE | `wish_heals_half_maxhp_the_turn_after_cast` | STATE (the +maxhp/2 heal at N+1) + SEED |
| MC23 **the WISH RESIDUAL-ORDER pin (CRITICAL, LIFE/DEATH)** — the Wish heal (order 7) fires BEFORE the sand chip (order 8); a low-HP mon under sand SURVIVES only because the order-7 heal beats the chip. WRONG (Wish at order 11): the sand chip KOs the mon FIRST → `wish.onEnd`'s `!target.fainted` guard skips the heal → the mon DIES | `wish_residual_fires_at_order_7_saving_a_low_hp_mon_from_the_sand_chip` | STATE (survives with the exact post-order HP; reverting to order 11 → faint) + SEED |
| MC24 WISH double-cast FAILS ([still], DRAW-FREE); the pending (1st) Wish resolves on schedule | `wish_double_cast_fails_and_the_pending_wish_resolves` | STATE (the 1st Wish resolved; no fresh pending) + SEED |
| MC25 WISH is SLOT-KEYED — survives the wisher switching out (a side/slot condition, not a mon volatile) | `wish_is_slot_keyed_and_survives_a_switch` | STATE (the entrant active + the slot-keyed Wish resolved) + SEED |
| MC26 BATON PASS boost transfer — the +2 Atk passes to the entrant (`copyVolatileFrom.boosts`), DRAW-FREE. WRONG (boosts not applied): the entrant enters +0 | `baton_pass_transfers_the_boosts_to_the_entrant` | STATE (Snorlax +2 Atk after the pass) + SEED |
| MC27 BATON PASS substitute transfer — the SUB HP passes to the entrant | `baton_pass_transfers_the_substitute_to_the_entrant` | STATE (Snorlax sub HP 83 after the pass) + SEED |
| MC28 BATON PASS leech-seed transfer — the seed passes (the seeder keeps draining the new mon) | `baton_pass_transfers_the_leech_seed_to_the_entrant` | STATE (Snorlax leech-drained + the +2 Spe passed) + SEED |
| MC29 BATON PASS with NO eligible bench FAILS ([still]+-fail, draw-free — NOT_FAIL, no switch request) | `baton_pass_with_no_bench_fails_draw_free` | STATE (the mon stays active, no forced switch) + SEED |

(The FZ1/FZ2 pins capture the two bugs the DMG_MOD e2e admission surfaced — FZ1 the sun-freeze
immunity gate [ground truth `harness/probe_sun_freeze_regression_rng.js`, semantics probe-settled by
`harness/probe_sun_freeze_immunity.js`], FZ2 the packed-team move-alias resolution [ground truth
`harness/probe_wisp_alias_regression_rng.js`, plus the dex unit test
`dex::alias_tests::move_aliases_resolve_to_the_canonical_move`].)
(The TD1–TD5 pins capture the taunt/disable selection-restriction layer — ground-truth seeds/state
from `harness/probe_taunt_disable_regression_rng.js` [TD1–TD4] + `probe_disable_zero_pp_rng.js`
[TD5]; the duration/onBeforeMove semantics were
probe-settled by `probe_disable_full_lifecycle.js` / `probe_taunt_duration_branch.js` /
`probe_taunt_disable_onbeforemove_rng.js`.)
(The T1–T5 pins capture the trapping switch-legality layer — ground-truth seeds/state from
`harness/probe_trapping_regression_rng.js`; the semantics/draw model were probe-settled by
`harness/probe_trapping_rng.js`.)
(The FD1–FD4 pins capture the gen-3 fixed-damage draw-model gotchas this layer surfaced — the
acc-100-not-never-miss roll, the accuracy-drawn-then-`-immune` short-circuit, the sub-absorb of the fixed
number — with ground-truth seeds from `harness/probe_fixeddamage_regression_rng.js`.)
(#1 Water/Volt Absorb HEAL-on-hit + #5 the absorb's accuracy-gating are already pinned by
`water_absorb_heals_on_hit_but_not_on_a_miss` in `src/turn.rs`.) The PHAZE pins (P1–P3) capture the
gen-3 phaze draw-model gotchas this layer surfaced — the acc-100-not-never-miss roll, the n=1 sample
draw, the fail-case no-draw — with ground-truth seeds from `harness/probe_phaze_regression_rng.js`.
Regenerate the ground-truth seeds with `node src/rust_sim/harness/probe_regression_rng.js` /
`node src/rust_sim/harness/probe_residual_order_rng.js` / `node src/rust_sim/harness/probe_phaze_regression_rng.js`
after any PRNG/draw-order change, then update the constants.

Besides the e2e-FOUND bugs above, the table's practice also carries **FEATURE pins** for newly-modeled
mechanics (the same revert-verified constructed-scenario style): the FLASH FIRE `flash_fire_*` pins
(`gen3_flashfire_boost_v1`), and the **NATURAL CURE `natural_cure_*` pins** (`gen3_natural_cure_v1`) —
**NC1** `natural_cure_cures_status_on_voluntary_switch_out` (a badly-poisoned NC Starmie pivots out then
back → RETURNS UNSTATUSED; STATE + the DRAW-FREE seed, plus a non-NC control that RETURNS still-toxic with
BYTE-IDENTICAL seeds — the seed-neutrality proof), **NC2** `natural_cure_is_a_no_op_on_a_faint` (an
EQ-KO'd burned NC Blissey KEEPS its burn — the `!fainted` gate; reverting the gate wrongly cures it),
**NC3** `natural_cure_phaze_drag_cures_the_dragged_out_mon` (a Roar drags the toxic'd NC Starmie OUT → it's
cured on the bench; STATE + the DRAW-FREE seed) — ground truth `harness/probe_naturalcure_regression_rng.js`,
semantics probe-settled by `probe_naturalcure_{dump,rng}.js`.

The **BATCH-1 ability-class pins** (`gen3_ability_batch1_v1`, ground truth
`harness/probe_ability_batch1_regression_rng.js`) — each a CONSTRUCTED scenario reseeded to the RAW seed
(so the port's draw-free `start_with_switchins` aligns), revert-verified (each FAILS when its class's engine
wiring is disabled): **B1** `battle_armor_prevents_the_crit_but_draws_the_roll` (a seed where the crit roll
COMES UP: an Insomnia control CRITS a Snorlax → 123 while Battle Armor prevents it → 324, at the IDENTICAL
post-turn seed — the draw-free override); **B2** `chlorophyll_speed_doubles_and_flips_the_first_mover_in_sun`
(a slow Bellossom's Chlorophyll ×2 = 272 OUTSPEEDS Groudon 216 in Drought-set sun → p1 first; a no-op control
is p2-first); **B3** `cloud_nine_suppresses_the_sandstorm_chip` (a Cloud Nine Psyduck takes NO sand chip
[304/304] while a Damp control takes maxhp/16 [285], sand still up); **B4**
`speed_boost_raises_the_spe_stage_by_one_each_active_turn` (Ninjask +1 spe stage at the end-of-turn residual,
DRAW-FREE); **B4b** `rain_dish_heals_each_end_of_turn_in_rain` (Ludicolo's exact post-turn HP includes the
+maxhp/16 heal, DRAW-FREE). Plus the STEP-1 weather-eachEvent pin
`sun_rain_weather_turn_tie_draws_the_eachevent_weather_shuffle_seed` (a Kyogre-Drizzle mirror rain-turn tie
draws the end-of-turn `eachEvent('Weather')` shuffle; ground truth
`harness/probe_weather_eachevent_tie_regression_rng.js`, semantics re-confirmed by
`probe_weather_eachevent_sunrain.js`). The class-sweep proof is `harness/gen_ability_batch1_golden.js` →
`tests/ability_batch1_test.rs` (300 game-end battles, byte-for-byte).

The **BATCH-2 ability-class pins** (`gen3_ability_batch2_v1`, ground truth
`harness/probe_ability_batch2_regression_rng.js`) — the DRAW-BEARING "reactive" classes + block tail, each a
CONSTRUCTED `gen3customgame` board reseeded to a RAW seed (aligning the port's draw-free
`start_with_switchins`), revert-verified (each FAILS when its class's engine wiring is disabled): **B2-1**
`static_contact_proc_paralyzes_the_attacker` (Body Slam into Static paras the ATTACKER; the proc's
`randomChance` draws AFTER the move secondary → the post-turn seed differs from a no-op control's);
**B2-2** `effect_spore_samples_a_status_onto_the_attacker` (the NESTED `random(10)`+`sample(3)` draw — pins
the sample vs a flat 3-status split); **B2-3** `rough_skin_recoils_the_attacker_draw_free` (maxhp/16 recoil +
the IDENTICAL-to-control seed = the draw-free proof); **B2-4** `damp_cancels_explosion_no_self_ko` (Damp
cancels Explosion at TryMove — the user does NOT self-KO, the move draws nothing → the seed differs from the
self-KO control); **B2-5** `soundproof_immune_to_sing` (a sound Sing is immune — no sleep — even at a
Sing-lands seed, proving the ABILITY blocks it); **B2-6** `suction_cups_blocks_the_roar_drag_no_sample` (a
Roar into Suction Cups → NO `sample`, the holder STAYS; the dragged control makes a different mon active +
a different seed); **B2-7** `synchronize_reflects_paralysis_to_the_caster` (Thunder Wave into Synchronize
paras the caster TOO — DRAW-FREE in customgame, so the seed is IDENTICAL to the no-op control's). The
class-sweep proof is `harness/gen_ability_batch2_golden.js` → `tests/ability_batch2_test.rs` (960 game-end
battles, per-decision STATE+HP+STATUS+SEED, byte-for-byte). See "## Data-driven mechanics" → the BATCH-2
subsection for the full draw model + the DEFERRED (Cute Charm / Color Change) list (Trace / Shed Skin /
the berries have since shipped in batch 3, `gen3_berry_trace_shedskin_v1`, pins BR1-BR6).

**The BATCH-4 pins** (`gen3_ability_batch4_v1` — ground truth `harness/probe_batch4_regression_rng.js`,
constructed customgame boards, revert-verified — each FAILS when its member's engine wiring is
disabled): **B4-1** `truant_loafs_every_other_turn_draw_free` (the loaf turn leaves the foe untouched,
deducts NO PP, and its seed is the DRAW-FREE ground truth — no acc/crit/dmg, no para roll; the t3
toggle parity); **B4-2** `inner_focus_blocks_the_flinch_at_the_apply_but_draws_the_roll` (on a
flinch-PASS seed the Inner-Focus Snorlax MOVES with the sim's exact seed — the roll IS drawn — while
the Thick-Fat control is cant'd on the SAME seed); **B4-3** `shadow_tag_traps_a_flying_foe_draw_free`
(a Flying Skarmory is trapped, the holder isn't, and the post-turn seed is IDENTICAL to a Keen-Eye
control's — the 0-draw proof); **B4-4** `cute_charm_attracts_and_the_attract_cants_gender_gated_after_the_roll`
(the M attacker is attracted then attract-cant'd; the F-into-F control's dec-0 seed is IDENTICAL —
the 1/3 roll draws for a same-gender pair too — and its dec-1 diverges); **B4-5**
`color_change_overrides_the_types_for_later_chart_reads` (TBolt→Electric, then the EQ is
SUPER-EFFECTIVE through the override 188→57 and re-overrides to Ground — `types_override` asserted
directly); **B4-6** `kings_rock_appends_the_trailing_flinch_secondary` (the KR flinch cants the
slower Snorlax at the probed seed; the no-item control's stream diverges); **B4-7**
`focus_band_survives_a_lethal_move_hit_at_one_hp` (survive at exactly 1 HP on the 1/10-pass seed;
the no-item control faints on a different stream).

## A/B fuzzer (the continuous differential parity hunter)

The e2e capstone is a FIXED 220-battle committed golden; the **A/B fuzzer** is its UNBOUNDED
sibling — a harness that runs for **hours unattended**, generating fresh team pairs + seeds +
random legal choices, driving the REAL Showdown sim and the port side-by-side, and saving a
**self-contained, standalone-replayable repro** for every divergence. Zero API quota while
running; every future mechanic layer becomes automatically-stress-tested. It found real gaps in
its first bounded smoke (see the fix queue below).

- **The driver** `harness/ab_fuzz.js` — per chunk (default 25 battles): generate/pick team
  pairs, drive the omniscient BattleStream to game-end via the e2e recorder (**REUSED, not
  copied** — `runBattle`/`emitBattle`/`isModeledMove`/ability/item predicates are
  `module.exports`ed from `gen_e2e_fuzz.js`, one source of truth; a direct
  `node gen_e2e_fuzz.js` still regenerates the e2e goldens byte-identically under a
  `require.main` guard), write the chunk in the SAME TAB golden format (SCEN/TEAM/INIT/DEC/END),
  replay it through `ab_replay`, tally per-battle verdicts, and save repro dirs. One stats line
  per chunk → `<out>/ab_fuzz.log` (ok/diverged/cumulative/kinds/battles-per-hour/coverage/
  adjustment rate). A chunk that errors logs + continues; SIGINT finishes the chunk then
  summarizes (second SIGINT aborts). Flags: `--mode randbats|random|pool` (default randbats),
  `--battles N` / `--hours H` (default: until killed), `--master-seed S` (default from time,
  ALWAYS printed → reproducible), `--chunk N`, `--out DIR` (default `harness/ab_fuzz_out/`),
  `--keep-chunks`. Battles that hit a forced-UNMODELED state are kept as comparable PREFIXES
  (ended=0/none — `run_full_battle` reproduces that exactly) instead of being dropped.
- **The modes.** `randbats` (default): Showdown's OWN gen3 random-battle generator
  (`Teams.generate('gen3randombattle', {seed})` — probe-verified deterministic under a gen5-style
  seed; backed by `dist/data/random-battles/gen3/teams`). Sets are adapted at the SET level to be
  port-replayable: unmodeled item → Leftovers; unmodeled ability → the species' modeled/no-op
  ability, else the TEAM is rejection-sampled (real gen3 species are ability-saturated → high
  disclosed rejection rate; rejection reasons tallied); missing nature → Hardy (neutral,
  stat-identical both engines — counted apart from adjustments); movesets/levels/EVs UNTOUCHED
  (the picker never PICKS an unmodeled move). Adjustment rate logged per chunk — adjusted teams
  are "randbats-derived". `random`: the MODELED-UNIVERSE generator — 6 species per team sampled
  from every gen3-dex species whose learnset ∩ modeled-move-universe ≥4 (≥1 modeled DAMAGING move
  forced per mon), modeled/no-op ability, modeled item, random nature/EVs/IVs, level 100, packed
  with the real `Teams.pack` (round-trip-verified). This is the coverage multiplier (the first
  smoke: **195 species / 146 distinct moves exercised** — every universe species became active —
  vs the pool mode's 21 species / 37 moves) and the mode that flushes out modeled-predicate ↔
  engine drift. `pool`: the e2e's 22 filter-clean `data/teams/` teams with fresh seeds/choices.
- **The replayer** `src/bin/ab_replay.rs` (ADDITIVE — links the existing public lib APIs;
  zero engine change): parses the TAB chunk (non-panicking parser), replays each battle via
  `Battle::start_with_switchins` → `run_full_battle` at the recorded init seed, and emits ONE
  JSON verdict line per battle — `ok` or the FIRST divergence with a kind taxonomy
  (`seed` [draw bug] > `request` > `species` > `state` [hp/maxhp/fainted/left] > `status` >
  `boost` > `confusion` > `spikes` > `firstmover`, plus per-battle `start_error`/`init_seed`/
  `decision_count`/`ended`/`winner`), decision index, and expected-vs-got. Engine PANICS are
  **caught** (`catch_unwind` + a message-capturing hook) and reported as `"verdict":"panic"` —
  the loop never dies. It also replays a saved repro DIR directly
  (`ab_replay <dir>` reads `<dir>/battle.txt`) — the repro→pin path. Sanity: the committed e2e
  golden replays `ok:220 diverged:0` through it.
- **Repros** `<out>/divergences/<runid>_<battleid>/`: `battle.txt` (a single-battle chunk —
  standalone forever, independent of generator drift) + `summary.json` (mode, master seed,
  battle/init/choose seeds, packed teams, choice tokens, first-divergence
  kind/decision/expected/got/detail, replay_cmd). After an engine fix the same
  `ab_replay <dir>` must flip to `ok` — then pin it as a NAMED deterministic
  `tests/regression_test.rs` test per the edge-case→pin law.
- **Fault-injection proof (the tool catches bugs — verified 2026-07-03):** three one-at-a-time
  engine perturbations, each caught by a 6-battle pool run at a fixed master seed, each repro
  verified standalone-replayable, each restore verified byte-identical (`diff` vs a pristine
  copy) + the full suite green: (1) DROPPED DRAW (the unconditional end-of-turn Quick Claw
  `random_chance(1,5)` skipped) → 6/6 flagged **kind=seed** at dec 0; (2) STATE error (+1 damage
  on the non-absorbed apply path) → 6/6 **kind=state** (`hp=390` vs `hp=389`, seed matching);
  (3) WINNER flip on the game-end path → 5/6 **kind=winner** (`expected P2 got P1`; the 6th is a
  never-ended prefix battle — correctly still ok).
- **The first bounded smoke (2026-07-03, master seed 20260703 — REAL findings, the fix queue;
  NOT fixed inline; repros under `harness/ab_fuzz_out/smoke_*/divergences/`):**
  `pool` 100 battles → **100 ok / 0 diverged** (the port stays bit-for-bit on the e2e corpus
  under fresh seeds/choices). `randbats` 300 battles (21135 decisions, ~2500 battles/h) →
  267 ok / **27 diverged + 1 panic** (kinds seed=21 status=4 state=2 panic=1). `random`
  200 battles (11412 decisions, ~3300 battles/h) → 49 ok / **151 diverged** (kinds state=89
  seed=60 firstmover=2) — the modeled-predicate surface beyond the 22 real teams' movesets had
  simply never been exercised. Triaged clusters (suspected mechanism per repro decode):
  1. ✅ **FIXED (2026-07-09, `gen3_facade_v1`) — Facade ×2-when-statused + the runEvent-tail
     INTEGER-GUARD.** Facade carries the dist `onBasePower` (`chainModify(2)` when the user
     has a non-`slp` major status); `isModeledMove` never rejected `onBasePower`, so it was
     admitted but priced flat BP 70. **Probe-settled** (`harness/probe_facade_gen3.js`):
     psn/tox/par all ×2 (BP 140); brn ×2 AND the gen3 burn damage-halve STILL applies (gen3
     Facade does NOT ignore burn — max-roll 108 == the unstatused 108); a burned GUTS user
     composes Atk ×1.5 + halve-suppressed + BP ×2 (318); DRAW-FREE (4 draws both arms). The
     fix is a BP-CHAIN member pushed in `run_move` (id-gated per the fixed-damage precedent —
     `gen3_moves.json` has no onBasePower field). The probe ALSO overturned the old "a Direct
     item discards the BP chain" shortcut in `calc_damage`: Pink Bow (Normal ×1.1 DIRECT
     float) + poisoned Facade (Normal chain ×2) CO-FIRE, and `70 * 1.1 == 77` EXACTLY in f64
     → the sim's runEvent-tail guard (`relayVar === Math.abs(Math.floor(relayVar))`,
     battle.js:709) PASSES and the accumulated chain RE-APPLIES → BP 154, NOT 77. `damage.rs`
     now implements the EXACT integer-guard (a non-integer float like 75×1.1=82.5 still skips
     the chain). **Pinned** by the revert-verified
     `regression_test.rs::facade_status_doubles_bp_and_composes` (FA-a..FA-e incl. the FA-d
     bow-composition; ground truth `harness/probe_facade_defrost_regression_rng.js`); each of
     the two components revert-fails its pin. **Parity: 143/145 facade-team repros in
     auto_0709_0805 + 333/344 in auto_0708_0304 flip to `ok`.** NO admission change was
     needed — `isModeledMove` already admitted facade (and the committed e2e golden replays
     220/220 byte-identical, md5 unchanged).
  2. **Pink Bow / Polkadot Bow (+ the gen4-named incenses)** — `MODELED_ITEMS` lists `pinkbow`,
     `polkadotbow`, `oddincense`, `rockincense`, `roseincense`, `waveincense` as modeled ×1.1
     type-boosters, but the port's `resolve_atk_stat_mods` table implements NONE of them (repro:
     Polkadot-Bow Body Slam dealt ×1.1 in the sim, flat in the port — kind=state). FIX: add the
     bows (real gen2/3 Normal ×1.1); DECIDE the incenses (gen4 items the sim still applies under
     gen3customgame) — implement or drop from `MODELED_ITEMS`.
  3. **Accuracy/evasion STAGES are not folded into the to-hit roll** (~8 random-mode battles) —
     the predicate admits accuracy-drop secondaries (Mud-Slap's `boosts:{accuracy:-1}` is a
     "structured stat-boost" shape); the port tracks the stage in `boosts[5]` but `run_move`
     rolls `random_chance(accuracy, 100)` on the RAW move accuracy (repro: a double-Mud-Slapped
     Entei's Bite hit in the port but missed in the sim → kind=seed via the hit-path draws).
     FIX: apply the gen-3 acc/eva stage table in the to-hit computation (or reject acc/eva-boost
     secondaries from the predicate).
  4. **Substitute-turn seed cluster** (7 of the randbats seed divergences have a Substitute at
     the diverging decision — new interleavings on varied-level randbats teams the constructed
     sub golden + the 284 e2e sub decisions never hit).
  5. **Ice-move FREEZE status cluster** (4 randbats battles: the port freezes a mon the sim does
     NOT, with the seed still matching — an equal-count mis-ORDERED draw or a freeze-gating rule;
     e.g. Delibird Ice Beam into Chimecho on a Toxic turn).
  6. **Switch-boundary seed cluster** (5 randbats + 6 random battles diverge at a switch-only
     decision — a switching/entry draw-order case beyond the e2e's team pool).
  7. **The 1 panic** (randbats, Whirlwind battle): the port's phaze DRAG diverged from the sim's
     upstream, so a later recorded slot choice landed on the port-active's unmodeled **Wish** →
     the fail-loud panic (caught by the replayer's `catch_unwind`, reported as
     `verdict=panic` with the message — the loop survived). The panic is the SYMPTOM; the drag
     divergence is the bug.
  Plus ~2 `firstmover` divergences (wrong first mover, seed + state matching) — an ordering-
  without-draw case. The `random` mode's high rate is dominated by clusters 1–3; dedupe by
  mechanism, not by repro count.
- ✅ **THE RESIDUAL TAIL CLEARED (2026-07-10)** — clusters 4–7 (and everything else left in the
  gender-pinned corpus) are FIXED: re-triaging the complete `auto_0709_0805` (307 repros) with
  the current binary and root-causing every survivor found **SEVEN engine bugs**, after which
  the corpus replays **307/307 ok** (incl. the 4 fail-loud panics — their upstream drag
  divergences sat inside the fixed clusters). The bugs (all probe-settled, each with a
  revert-verified pin — full record in EDGE_CASES.md "✅ CLEARED — the A/B residual tail"):
  (1) `gen3_plus_minus_v1` Plus/Minus cross-field SpA ×1.5 (the gen3 `onModifySpA` scans
  `getAllActive()` — FOES included; the old NOOP classification's "partner-less in singles"
  never faced Plus against Minus; `plus`/`minus` are now MODELED, not no-op — the admission
  union is unchanged so the e2e golden is untouched); (2) `gen3_ff_wisp_absorb_v1` Will-O-Wisp
  into a non-Fire, status-free, un-subbed Flash Fire holder is ABSORBED (arms, no burn — incl.
  a TRACED FF); (3) `gen3_cloudnine_end_v1` Cloud Nine / Air Lock `onEnd` fires
  `eachEvent('WeatherChange')` at BOTH End sites — switchIn's alive-outgoing ability End
  (pre-swap) AND faintMessages' pre-`fainted=true` ability End — one tie-shuffle on a
  cached-speed tie (the dominant "icebeam tail" was really randbats Golduck-mirror boards);
  (4) `gen3_ff_frozen_no_absorb_v1` a FROZEN Flash Fire holder is NOT fire-immune (full draws,
  then the fire-move thaw); (5+6) `gen3_fnt_clears_status_v1` `checkFainted` sets
  `status="fnt"` AND `clearVolatile` zeroes the corpse's boosts — so the replacement
  instaswitch sort reads the PLAIN corpse speed (para/+6 erased → the mirror ties draw);
  (7) `gen3_statusimmune_onupdate_cure_v1` the 6 STATUS_IMMUNE members' `onUpdate` CURES the
  holder's matching status (reachable only via TRACE — a slept Porygon2 tracing Insomnia).
  Pins PM/FFW/CN1/CN2/FZ3/FN1/TC1 in `tests/regression_test.rs`; ground truth
  `harness/probe_{plus_minus_gen3,plusminus_ffwisp_regression_rng,residual_tail_regression_rng}.js`.
  Full suite 273/0 green; e2e md5 `a23d77ac60d4af168b8a4428f0b465c9` UNCHANGED.
- ✅ **THE STEADY-STATE TAIL CLEARED (fix-queue #4, 2026-07-10)** — the first all-fixes 12h run
  (`auto_0709_2205`) produced **9 divergences / 35,018 battles (0.026%, 0 panics)**; re-triage on
  the residual-tail binary: 4 already-fixed noise, 5 true survivors → **THREE engine bugs**, after
  which the corpus replays **9/9 ok** (auto_0709_0805 stays 307/307; auto_0708_1705 replays
  489 ok / 0 diverged — its 1130 panics stay the pre-gender-pinning attract fail-loud noise):
  (1) `gen3_faint_queue_order_v1` — `faintMessages` drains `faintQueue` in ENQUEUE order (each
  corpse fully processed before the next corpse's ability-End), so a mutual Explosion's
  self-KO'd USER is already inactive when the Cloud Nine target's End WeatherChange fires → the
  dying holder gathers alone, NO tie draw (the port's side-order walk drew a phantom shuffle);
  (2) `gen3_fainted_no_ability_speed_v1` — a corpse's ability handlers no longer gather: a Swift
  Swim corpse under rain sorts the replacement instaswitch at PLAIN speed (alive 368 → fainted
  184, tying the mirror corpse → the shuffle the port missed);
  (3) `gen3_tox_stage_persists_v1` — the gen3 tox stage-0 reset fires via the runSwitch-time
  `runEvent('SwitchIn')`, NOT the raw switch swap: it RESETS on any switch-in whose runSwitch
  RUNS, but PERSISTS when the queued runSwitch is CANCELLED by gen3 faint-cancels-all (a
  co-replacement's Spikes-faint — the ab_1166_22 Mew's lethal stage-2 chip). Bug 3 was ALSO the
  fix-queue-#3 Lens-2 lead (auto_0708_1705 rmrcqwc2c_ab_793_13, state@38 hp 81-vs-97 —
  revert-reproduced, fix-flipped: a real bug, not gender noise). Pins FQ1/FS1/TX1/TX2 in
  `tests/regression_test.rs` (all revert-verified; TX2 pins the reset's PLACEMENT); ground truth
  `harness/probe_fixqueue4_regression_rng.js` + `probe_tox_stage_switch.js`. NEW REUSABLE TOOL:
  `harness/probe_repro_simtrace.js` — replay ANY saved repro dir through the REAL sim with
  per-draw PRNG call-site instrumentation (`node harness/probe_repro_simtrace.js <repro-dir>
  [decFrom] [decTo]`) — the root-causing workhorse for this queue. Full suite **277/0** green;
  e2e md5 `a23d77ac60d4af168b8a4428f0b465c9` UNCHANGED. Full record: EDGE_CASES.md
  "✅ CLEARED — A/B fix-queue #4".
- **Run it:** see the README runbook ("A/B differential fuzzer"). Quick start:
  `node src/rust_sim/harness/ab_fuzz.js --mode randbats --hours 12` (overnight),
  `--mode random --battles 200 --master-seed S` (reproducible bounded hunt);
  replay any repro with `target/release/ab_replay <repro-dir>`.

### The OMNISCIENT BYTE differential (`--protocol`, `gen3_omniscient_byte_fuzz_v1`)

The A/B fuzzer above checks a RECONSTRUCTED per-decision STATE tuple + seed + winner. `--protocol`
turns it into a **literal `|...|` protocol byte differential**: per battle it TEES the REAL omniscient
filtered log (`gen_e2e_fuzz.js::runBattle` now attaches `rec.lines`, `|t:|`-normalized) into the chunk
golden (`emitBattle` appends `FMT`/`L` rows in protocol mode only — the state golden is untouched), and
`ab_replay --protocol` replays via `run_full_battle_logged`, filters BOTH sides through a shared DENYLIST
(drop `debug`/`error`, normalize `|t:|` — a superset of `protocol_test.rs`'s allowlist, so newly-emitted
batch-4c/5/6/snatch line types are diffed automatically), and first-divergence-diffs to a NEW
**`kind:"protocol"`** verdict (reported ONLY after state/seed/winner match, so a draw bug still surfaces
as `seed`). `--format {gen3customgame,gen3ou}` threads the run format (gen3ou = the clause-shuffle draw
path + the OU framing, reframed via the now-`pub bridge::reframe` before the diff). The picker is widened
to admit typed **Hidden Power** in `pool` mode (`isModeledMove(id, allowHiddenPower)` — engine models
typed HP at fixed BP 70, byte-safe for gen3ou-validated 70-BP teams; `random` mode's random IVs keep it
excluded). Genders are pinned (`pinGenders`) so the sim never draws one at construction (the switch-details
construction-window gap). Isolated build + run:
`CARGO_TARGET_DIR=/tmp/pokesim_target_bytefuzz cargo build --release --bin ab_replay` then
`POKESIM_AB_REPLAY_BIN=/tmp/pokesim_target_bytefuzz/release/ab_replay node harness/ab_fuzz.js --mode pool
--protocol --format {gen3customgame|gen3ou} --battles N`. FAULT-INJECTION PROVEN (a mangled
`[from] item: Leftovers` tag → 6/6 flagged `kind=protocol` at the exact `-heal` line; restored
byte-identical via cp-aside).

**Byte bugs FIXED (revert-pinned in `tests/protocol_byte_fuzz_test.rs`, all observation-only — the full
seed suite stays BYTE-IDENTICAL, e2e md5 unchanged):**
- **BF1 typed-HP move-name leak** — the port rendered the typed dex name `Hidden Power Ice`; gen-3 HIDES
  the HP type → `run_move` canonicalizes any `hiddenpower*` id to the bare `Hidden Power` for the announce.
- **BF2 Toxic residual `[from]` cause** — the DoT chip was `[from] tox`; Showdown reports `[from] psn`
  (the HP-field status token stays `tox`). A LATENT gap (the constructed protocol golden never realized a
  Toxic residual).
- **BF3 self/side-move announce target** — a NON-foe-directed move (`allySide`/`all`/`allyTeam` — Light
  Screen / Sunny Day / Rain Dance / Perish Song / Heal Bell) renders the USER as the `|move|` target, not
  the foe (`status_move_announce_renders_user`).
- **BF4 Pursuit interrupt `[from] Pursuit`** — the interrupt strike's `|move|` announce now folds
  `|[from] Pursuit` (via `set_next_move_from`).

**The STATUS-MOVE EMISSION-FORM SWEEP (2026-07-17) — pool byte-clean rate 27% → ~95%.** A pool-mode
`--protocol` fuzz over both formats surfaced ~8 general status-move / did-nothing emission forms that hit
nearly every real-team battle. All FIXED + revert-pinned in `protocol_byte_fuzz_test.rs` (13 pins;
`sleep_move_status_carries_from_move` / `toxic_into_steel_reports_immune` / `natural_cure_emits_curestatus_on_switch_out`
/ `recover_at_full_hp_emits_still_and_fail_heal` / `beat_up_emits_per_strike_activate_in_customgame` /
`protect_blocks_status_move_before_the_immune_report` / `shiny_mon_shows_the_shiny_details_flag` /
`knock_off_hint_fires_once_per_battle` / `stat_drop_blocked_by_substitute_emits_still_and_fail`), **all
observation-only — the full seed suite stays BYTE-IDENTICAL (cargo test green, e2e md5 unchanged):**
- **`||[still]` did-nothing FAIL framing + the `-fail` line** (the sim `attrLastMove('[still]')`s the
  announce + `runMoveEffects` `-fail`): Recover/Soft-Boiled@full → `[still]` + `-fail|<user>|heal` (the
  `heal` sub-tag); Toxic-vs-Substitute / **STAT-DROP-vs-Substitute** (Screech/Charm/Metal Sound/&c. into a
  non-`bypasssub` sub — the FORM-1 residual the byte fuzzer surfaced last, `BF-F15`, captured
  `harness/probe_statdrop_substitute.js`; the stat-drop arm's sub-block used to return emitting NOTHING) /
  repeat-Protect-or-willAct-fail / double-Wish / no-bench Roar /
  Refresh-no-status / weather-set-into-same (Rain Dance) / Light-Screen-already-up / Beat-Up-fizzle →
  `[still]` + BARE `-fail|<user>`. (The Protect willAct-fail — a Protect after the foe SWITCHED — also
  emits `-fail`; the double-Wish `-fail` corrected a batch-3 comment that wrongly said "no -fail".)
- **status-TYPE-immunity `-immune`** (Toxic→Steel/Poison, Will-O-Wisp→Fire) — the `try_set_status`
  `status_type_immune` gate emits `|-immune|<target>` when the source is a status MOVE (`announce_immune_block`
  == the sim's `sourceEffect?.status`); a secondary-inflicted type-immune status stays silent.
- **Protect-before-immunity ORDER** — `run_status_move` reordered to match gen3 `tryMoveHit`: on a HIT the
  `runEvent('TryHit')` handlers (Protect → `-activate Protect`; Soundproof) win BEFORE the naturalImmunity
  `-immune`; on a MISS naturalImmunity still wins (`-immune`, no `-miss`). So Thunder Wave into a
  Ground-typed PROTECTING foe shows `-activate Protect`, not `-immune`.
- **sleep `-status …|[from] move: <Move>`** — threaded the source move name into `try_set_status_impl`; only
  SLEEP carries it (par/psn/brn/frz from a move stay bare, per `conditions.js` per-status `onStart`).
- **Natural Cure `-curestatus …|[from] ability: Natural Cure|[silent]`** — emitted in `execute_switch`
  BEFORE the replacement `|switch|`/`|drag|` line (`curestatus_from_ability_silent`).
- **confusion `-start|X|confusion` / `-activate|X|confusion` / `-end|X|confusion` + the self-hit
  `-damage|…|[from] confusion`** — the `add_confusion` onStart, the `on_before_move` reveal + the counter-0
  `onEnd`, and the self-hit damage line.
- **fire-move thaw `-curestatus|<t>|frz|[msg]`** (`frz.onDamagingHit` `cureStatus()`).
- **the delta-0 `-boost`/`-unboost` at the ±6 cap** — a PRIMARY self-boost MOVE emits `|-boost|…|spe|0`
  (Agility@+6) via `boost_applied` (the sim's `boost()` `!isSecondary && !isSelf` branch); the Clear
  Body / White Smoke / Hyper Cutter `-fail|unboost|[from] ability|[of]` for a PRIMARY foe-drop (Screech).
- **CONTACT-PROC status is BARE** (`|-status|<atk>|slp`, not `[from] ability: Effect Spore`) — the sim's
  `source.trySetStatus(status, target)` passes NO sourceEffect; the port let `try_set_status` emit the bare
  form (CONTRAST Synchronize, which pre-emits `-activate ability: Synchronize`). Draws unchanged.
- **Volt/Water Absorb `-heal` (not `-immune`) when the heal LANDS** — `apply_absorb_heal` now reports
  whether it healed; a below-full holder shows `-heal|…|[from] ability: Volt Absorb|[of] <user>`, only a
  full-HP holder shows `-immune|…|[from] ability` (the F3 capture had only exercised the full-HP case).
- **Beat Up per-strike `-activate|<user>|move: Beat Up|[of] <ally>`** — gen3customgame EMITS it (the mod's
  `condition.onModifySpA`, gated on `!beatupnicknamesmod` — a gen3 Standard rule present in gen3ou, absent
  in customgame → aligned with `sleep_clause`). The task's "gen3 does not emit it" claim was WRONG; the
  omniscient stream + resolved dist confirm the customgame emit.
- **shiny details flag** (`|switch|…|Quagsire, M, shiny|…`) — `switch_details` appends `, shiny` from
  `set.shiny`.
- **Knock Off `-hint` once per battle** — `ProtocolBuilder::hint` now dedups per battle (a `HashSet`,
  mirroring the sim's `this.hints`), so the Gens-3-4 Knock Off note fires once, not per Knock Off.
- **gen3ou clause `-message`** — a Sleep-Clause-blocked 2nd sleep emits `|-message|Sleep Clause Mod
  activated.` + a deduped `|-hint|`; a Freeze-Clause-blocked 2nd freeze emits `|-message|Freeze Clause
  activated.` (rulesets.js). gen3customgame has no clauses → no message (so the e2e/seed suites, all
  gen3customgame, are untouched).

**The FROZEN regression CORPUS (`tests/vectors/byte_fuzz_corpus/`, a `cargo test` gate).** The byte
fuzzer FINDS emission divergences but isn't itself a test gate, so the fixed forms are frozen as a
corpus of ≥15 self-contained full-battle fuzzer repros (each a `SCEN`/`TEAM`/`FMT`/`INIT`/`DEC`/`END`/`L`
chunk — a repro dir's `battle.txt`), named by the form each guards
(`01_recover_at_full_still_fail.txt`, `02_toxic_into_steel_immune.txt`,
`03_natural_cure_switchout_curestatus.txt`, `04_sleep_from_move.txt`, `05_confusion_start.txt`,
`06_beatup_activate.txt`, `07_protect_blocks_status_activate.txt`, `08_shiny_details.txt`,
`09_toxic_residual_from_psn.txt`, … through the gen3ou `18_freeze_clause_message_ou.txt` /
`19_natural_cure_ou.txt` / `20_protect_activate_ou.txt`; 20 fixtures, both formats). **`tests/byte_fuzz_corpus_test.rs`**
auto-discovers every `*.txt`, invokes the built `ab_replay` binary (`env!("CARGO_BIN_EXE_ab_replay")`)
in byte mode on each, and asserts NO `kind=protocol` divergence (a floor of `len >= 15` keeps the corpus
from silently shrinking; a per-file panic names the diverging file + line). FAULT-INJECTION PROVEN
(a Toxic-residual `[from] psn`→`tox` emit perturbation → the gate FAILS naming
`02_toxic_into_steel_immune.txt` at the exact line; restored byte-identical). To add a fixture, drop a
clean fuzzer repro `battle.txt` in the folder (see its `README.md`) — the test auto-discovers it. The
`ab_fuzz_out*` run dirs are gitignored — never commit run output.

**Residual byte divergences (HONESTLY UN-fixed — genuinely-deep / rare):** the switch-in `-weather` /
Pressure `-ability [silent]` **`[of]` p1-vs-p2 ATTRIBUTION** on a same-species speed-TIED lead (Tyranitar
mirror / Zapdos mirror) — the KNOWN turn-0 construction-shuffle speed-tie gap (the port doesn't model the
turn-0 construction window; a seedless deferral, ~2-3% of pool battles, masks the whole battle when it
hits at line ~14); the end-of-turn **Leftovers `-heal` emit ORDER on a residual speed TIE** (two
same-species same-para mons — the residual handler tie-shuffle permutation the emit loop doesn't follow;
~1 in 300, seed matches). SEPARATELY, in `pool` mode ~1.5% of battles hit a GENUINE state/seed/status
divergence (an hp off-by-1 cascading to a freeze/seed) — the **HP fixed-BP-70 caveat materializing**: a few
gen3ou pool teams carry Hidden Power with non-70-BP IVs, which the engine (fixed BP 70) mis-damages; keep
HP out of a byte run that must be state-clean, or restrict to teams verified 70-BP.

## Bridge / request A/B fuzzer (the per-side + `|request|` parity hunter)

The **PER-SIDE sibling** of `ab_fuzz.js` (which A/Bs the OMNISCIENT stream): it verifies, over
RANDOM teams, that the Rust crate's PER-SIDE (`p1`/`p2`) protocol streams + the `|request|` JSON
(the poke-env legal-action requests, incl. the maybeTrapped/trapped switch-legality state machine)
are BYTE-IDENTICAL to the real Node `getPlayerStreams`. It is the validation harness for Phase 1
(`bridge.rs`).

- **The driver** `harness/bridge_ab_fuzz.js` — per chunk: (1) drives a real in-process
  `BattleStream` + `getPlayerStreams` (the `local_sim_bridge.js` / `gen_bridge_capture.js` pattern)
  to game-end, picking random LEGAL + MODELED choices from a seeded choice-RNG, capturing BOTH
  per-side chunk streams + the ordered CMD stream; (2) **TRAPPING PROBES** — when a side's active is
  TRAPPED (Arena Trap / Magnet Pull / Shadow Tag, via the sim's `pokemon.trapped`), with `--trap-prob`
  (default 0.5) issues a REJECTED `switch` first (→ `|error|` + the `trapped:true`/`[Invalid]` re-request
  round) before the legal move; (3) drives `bridge_replay --ab` over the identical teams+cmds and diffs
  the Rust per-side chunks BYTE-FOR-BYTE with a first-divergence taxonomy (`preamble` / `perside` /
  `privacy`[HP-fold] / `request`[JSON] / `error`[trapped] / `chunk_count` / `panic`); (4) writes one
  stats line/chunk to `harness/bridge_ab_fuzz_out/bridge_ab_fuzz.log` + a self-contained
  standalone-replayable repro dir per divergence. `--debug`-free, no server.
- **Modes** (`--mode`, default `trapping`): **trapping** (a coordinated Arena-Trap/Magnet-Pull/Shadow-Tag
  matchup vs varied grounded/Flying/Levitate/Steel/Ghost foes + varied bench sizes incl. last-mon —
  the one mode where the port is bit-for-bit on the omniscient stream, so ALL divergences are genuine
  request/per-side-layer issues), **randbats**/**random**/**pool** (reuse `ab_fuzz.js`'s exported
  providers; genders pinned via `pinGenders`). `--format gen3customgame` (default, exact HP) or
  `gen3ou` (the OU framing + HP-privacy fold). One TAB golden format = `bridge_trapping_golden.txt`
  (SCEN/TEAM/INIT/CMD/CHUNK/END).
- **The replayer** `src/bin/bridge_replay.rs --ab` (ADDITIVE) — one JSON verdict per battle, panic-caught
  (a panic → `{"verdict":"panic"}`, never dies); it also replays a saved repro DIR directly
  (`bridge_replay <dir>` reads `<dir>/battle.txt`). Isolated build:
  `CARGO_TARGET_DIR=/tmp/pokesim_target_bridge` (NEVER the shared `target/` — the live `ab_replay`).
- **Three real Phase-1 bugs it found + FIXED** (all probe-settled vs the sim, revert-verified):
  1. **`gen3_shadowtag_firm_trap_v1`** — the gen3 mod's **Shadow Tag** sets `pokemon.trapped = true`
     DIRECTLY (`onFoeTrapPokemon`), so its FIRST `move` request already carries `trapped:true` (NO
     `maybeTrapped` phase), and a rejected switch draws `|error|[Invalid choice]…` with NO re-request
     (the `emitChoiceError` update no-ops — nothing to firm). Arena Trap / Magnet Pull call
     `tryTrap(true)` → `trapped = 'hidden'` → the `maybeTrapped`→`[Unavailable choice]`+re-request
     machine. `state::trap_is_firm` distinguishes them; `bridge.rs::serialize_active` + the rejection
     round read it. Pinned by the `shadow_tag_firm_trap` scenario in `bridge_trapping_golden.txt`
     (`gen_bridge_trapping_capture.js`, 3 scenarios now) → `bridge_test.rs`.
  2. **`gen3_struggle_activate_sideupdate_v1`** — a forced-Struggle mon emits `|-activate|<mon>|move:
     Struggle` (Struggle's `onModifyMove`) as a PER-SIDE `sideupdate` line, OWNER-ONLY, before the
     broadcast `|move|` batch (verified: the raw omniscient log emits it prefixed `pN\n…sideupdate`, so
     `getPlayerStreams` shows it only to the owner — like a `|request|`). `run_full_battle_bridge`
     injects it per-side after a Move commit whose active `must_struggle`. (The constructed protocol
     golden never runs a fully-out-of-PP mon.)
- **Smoke (trapping, bounded):** ~100–200 battles → 0 request/trapping/error divergences, rich coverage
  (per 100 battles ≈ 4700 requests, ≈740 trapped:true, ≈220 maybeTrapped, ≈420 `|error|` frames, ≈86
  forceSwitch), ~6000 battles/hr. Fault-injection PROVEN (drop firm-trapped → `kind=request`; wrong
  `|error|` text → `kind=error`; wrong HP-fold % [gen3ou] → `kind=privacy`; each caught + standalone-
  replayable + restored byte-identical). OBSERVATION-ONLY: the fixes are bridge-path only (turn.rs
  untouched by the Struggle line — it rides `run_full_battle_bridge`), so e2e md5
  `a23d77ac60d4af168b8a4428f0b465c9` UNCHANGED + protocol/writeline/e2e/bridge green.
- **Honest scope (next phase):** `randbats`/`random` modes surface PRE-EXISTING **omniscient-stream**
  gaps orthogonal to the request/per-side layer — the non-L100 `details` LEVEL display
  (`switch_details`/request `details` omit `, L84`; the port targets L100 gen3ou), a **mid-battle
  Intimidate `|-unboost|…|atk|0`** at the −6 Atk FLOOR (`turn.rs:6667` hardcodes the delta `-1` →
  always `atk|1`; the sim emits the CLAMPED-applied 0 — repro saved, probe-confirmed
  `harness` Intimidate-clamp), and the same Toxic-`[from]`/status-move/Water-Absorb clusters
  `ab_fuzz.js` already tracks. These belong to the omniscient fuzzer's fix-queue (they'd desync the
  raw stream too), not the bridge layer. `trapped:true` coverage is dense; a `gen3ou`-format trapping
  run additionally exercises the OU reframe + HP-privacy fold.
- **Run it:** `node src/rust_sim/harness/bridge_ab_fuzz.js --mode trapping --battles 200
  --master-seed S` (bounded), `--format gen3ou` (OU framing + HP fold), `--hours H` (overnight);
  replay any repro with `/tmp/pokesim_target_bridge/release/bridge_replay <repro-dir> --ab`.

## Data-driven mechanics (the class framework)

**The strategic shift (Phase 1 landed 2026-07-03, `gen3_item_mechanics_v1`):** stop hand-modeling
items/abilities one id at a time. The A/B fuzzer's motivating find: Pink Bow / Polkadot Bow + the
4 gen4-named incenses sat in the e2e's `MODELED_ITEMS` while the port's hardcoded
`resolve_atk_stat_mods` match-arm priced NONE of them — a drift class that recurs whenever an
allow-list and an engine table are maintained by hand in two places. The framework kills the
class: extract the gen3-RESOLVED item/ability tables ONCE (like the dex), classify EVERY entry
into mechanic CLASSES with machine-readable parameters, and implement ONE generic engine path per
class, validated by one class-sweep golden.

- **The extraction + the CLASS MAP.** `harness/dump_gen3_mechanics.js` reads the RESOLVED
  `Dex.mod('gen3')` (the WHOLE mod chain applied — the mod-chain law below), dumps every gen3
  item (132 = 128 gen≤3 + the 4 incense exceptions) + every gen3 ability (76, nums 1-76) with its
  resolved handler inventory + extracted parameters, classifies each (UNCLASSIFIED fails the
  dump), and writes **`tests/vectors/gen3_mechanics_inventory.md`** — the class map every future
  phase executes against (per-entry class / params / modeled-status / DRAW-vs-free tag).
  `--check` is the DRIFT GATE: it verifies the committed `data/pokemon/gen3_items.json` /
  `gen3_abilities.json` mechanics fields EXACTLY match the resolved dist (run it whenever either
  regenerates); `--json` emits the machine-readable extraction.
- **THE MOD-CHAIN LAW (the Light Ball cautionary tale).** gen3 resolves through gen4 → … → base,
  and later mods REPLACE and DELETE handlers: base Light Ball doubles Atk+SpA, the gen4 mod
  REWRITES it to an `onBasePower` double, the gen3 mod REWRITES it again to **SpA-ONLY ×2**.
  NEVER regex a single data file — extract from the resolved dist; the probe/golden against the
  real sim is the only oracle. (Same law as the taunt/disable durations.)
- **The data (the dex-style source of truth).** `tools/pokemon_data_extractor/sync.py` emits
  ADDITIVE, obs-neutral fields into `data/pokemon/gen3_items.json` (`typeBoost {type, mod:[num,
  den], fold: stat|basePower|basePowerDirect}`, `statMods {stat:[num,den]}` + `onlySpecies` +
  `untransformedOnly`, `choice`, `isBerry`) and `gen3_abilities.json` (`dmgMod {mod, fold, type/
  types, pinch, whenStatused, direct}` — DATA-ONLY until the ability class is wired) from a
  curated table derived via the dump (the callbacks are JS — invisible declaratively — so the
  table is curated like `_CURES_SELF_STATUS`, and the `--check` gate pins it to the dist). The
  4 gen4-named incenses are an explicit, documented exception to the extractor's gen filter (the
  sim applies them under gen3 formats; adding ENTRIES is obs-neutral — the obs encodes items by
  per-id `num` lookup, no enumeration index). `dex/items.rs` parses it all into `ItemData`.
- **The wired classes (Phase 1) — the STAT/BP-MODIFIER item family, all folds probe-settled:**
  - **TYPE_BOOST (24)** — 18 stat-fold members (`onModifyAtk/SpA chainModify` ×1.1 + Sea Incense
    ×1.05 → the offensive-stat chain), the 2 gen2 bows (`return basePower * 1.1` — a DIRECT
    float that REPLACES the event relayVar; the non-integer product SKIPS runEvent's final chain
    modifier and `clampIntRange` floors — mirrored exactly in `get_base_damage`), and the 4
    incenses (`chainModify([4915,4096])` ≈ **×1.2, NOT the assumed ×1.1** — the probe's headline
    surprise) at the BASE-POWER chain, ONE accumulated 4096 modifier shared with Thick Fat.
  - **SPECIES_STAT (6)** — Thick Club (Atk ×2 Cubone/Marowak), gen3 Light Ball (SpA-ONLY ×2
    Pikachu), DeepSeaTooth (SpA ×2 Clamperl), DeepSeaScale (SpD ×2 Clamperl — the first
    DEFENDER-side stat fold: `resolve_def_stat_mods` → the `ModifyDef/ModifySpD` chain, after
    the boost table, BEFORE the gen≤4 Explosion Def-halve), Metal Powder (Def ×2 untransformed
    Ditto), Soul Dew (SpA+SpD ×1.5 Lati@s — both directions).
  - **CHOICE** — Choice Band (`choice: true` + `statMods.atk [3,2]`; the lock was already
    modeled via `choice_locked_move`).
  `turn.rs::resolve_atk_stat_mods` / `resolve_def_stat_mods` / `resolve_bp_mods` are pure
  dex-data lookups — the hardcoded item match-arm is GONE, so the e2e `MODELED_ITEMS` (which
  gained the 6 species items) can never drift from the engine again for these classes. The
  confusion self-hit resolves the SAME helpers (a Thick Club Marowak's self-hit uses ×2 Atk, a
  Metal Powder Ditto's is halved by its own Def — the full-`getDamage` semantics the CB e2e_194
  fix established). All folds are DRAW-FREE.
- **Validation:** `harness/gen_damage_golden.js` grew 17 EXACT max-roll probes (48 scenarios
  total; 3 columns appended — atk_species/def_species/def_item — pre-existing lines
  prefix-identical) → `tests/damage_test.rs` (whose item mirrors are now dex-data-driven too);
  the CLASS-SWEEP golden `harness/gen_item_mods_golden.js` → **`tests/item_mods_test.rs`** (33
  scenarios × 30 seeds = 990 battles to game-end, 2664 per-decision STATE+HP+SEED assertions,
  1398 boosted-hit rows, ≥10 boosted hits enforced per member, matching + wrong-type/
  wrong-species controls; byte-reproducible); PERTURBATION-PROVEN (incense→×1.1, a leaked gen4
  Light-Ball Atk half, a removed species gate, a def-mult drift — each fails the gate; restores
  byte-identical); 6 revert-verified pins `IM1`-`IM6` in `tests/regression_test.rs`. The e2e
  regen after the `MODELED_ITEMS` growth left the golden BYTE-UNCHANGED (no pool team became
  newly filter-clean) — STRICT 220/220 stands.
- **The wired classes (Phase 2) — the ABILITY DMG_MOD family, all folds probe-settled against
  the RESOLVED gen3 dist (`gen3_item_mechanics_v1` ability side, `dex/abilities.rs::AbilityData.dmg_mod`):**
  - **PINCH (4)** — Torrent/Blaze/Overgrow/Swarm: an `onBasePower chainModify(1.5)` for the
    ability's type (Water/Fire/Grass/Bug) when the user is at `hp <= maxhp/3` — bit-exactly the
    integer `3*hp <= maxhp` (probe-verified at the maxhp=341 float boundary). A BP-chain member
    (`resolve_bp_mods`, joins the incense/Thick-Fat accumulate-once).
  - **Atk uncond (2)** — Huge/Pure Power: `onModifyAtk chainModify(2)`, PHYSICAL only (ModifyAtk
    touches only the Atk stat; a special move is un-boosted). An `AtkStatMod` in the ModifyAtk
    chain (`resolve_atk_stat_mods`), so Guts+CB stacks to ONE ×2.25 chain (not two rounds).
  - **Guts (Atk ×1.5 whenStatused)** — `onModifyAtk chainModify(1.5)` when the user has ANY major
    status, PLUS the physical burn-halve is SUPPRESSED (the port's existing `Combatant::has_guts`
    skip in `modify_damage`). PROBED: a burned Guts mon hits at full ×1.5 (=×1.497 realized), NOT
    ×0.75 — the ×1.5 stat fold composes with the burn-skip.
  - **Marvel Scale (Def ×1.5 whenStatused)** — `onModifyDef chainModify(1.5)` while the DEFENDER
    has a major status (the physical Def only), a `def_stat_mods` member (`resolve_def_stat_mods`).
  Hustle stays DATA-ONLY (`direct`, `this.modify(atk,1.5)` — its Atk half ships WITH the accuracy
  pipeline's ×0.8 acc side, never alone). Thick Fat keeps its dedicated `defender_thick_fat` hook
  (its `sourceBasePower` fold is a DEFENDER handler on the ATTACKER's BP). The confusion self-hit
  resolves the SAME helpers with the mon's own status on both sides (a burned Guts mon's self-hit
  is ×1.5 + burn-suppressed; a statused Marvel mon's self-hit is reduced — probe-verified). All
  folds are DRAW-FREE.
- **Validation (Phase 2):** `dex/abilities.rs` grew an `AbilityData` (replacing the abilities'
  `NamedNum`) that parses the `dmgMod` fields (`ability_dmg_mod_fields_parse`). The damage golden
  `harness/gen_damage_golden.js` grew 15 EXACT max-roll probes (63 scenarios total; a constructed
  pinch-HP/status hook + 4 columns appended — `atk_hp`/`atk_maxhp`/`def_status`/`def_ability` —
  pre-existing 28 indices unchanged) → `tests/damage_test.rs`; the CLASS-SWEEP golden
  `harness/gen_ability_dmgmod_golden.js` → **`tests/ability_dmgmod_test.rs`** (11 scenarios × 30
  seeds ≈ 330 decisive battles to game-end, per-decision STATE+HP+SEED, ≥10 boosted hits enforced
  per member, incl. wrong-type-pinch + unstatused-Guts controls; byte-reproducible). REVERT-PROVEN
  (the golden test catches every engine-fold revert — pinch/Huge/Pure/Guts/Marvel each diverges a
  battle); 5 revert-verified pins `AB1`-`AB5` in `tests/regression_test.rs` (AB1 Guts+burn, AB2 the
  pinch threshold boundary, AB3 Marvel def-side, AB4 Huge ×2, AB5 the status gate). **e2e admission
  DONE (`gen3_sun_freeze_immunity_v1`):** `torrent/blaze/overgrow/swarm/hugepower/purepower/guts/
  marvelscale` are now in `MODELED_ABILITIES` — the filter-clean pool grew **22 → 151 / 719** (the
  biggest single admission lever; the DMG_MOD gaps VANISHED from the taxonomy's top-gaps list), STRICT
  `filtered_diverged == 0` over 220 battles / 9963 decisions, byte-reproducible at the committed knobs.
  The admission was previously gated on ONE newly-admitted battle (e2e_86) that diverged on a
  PRE-EXISTING, non-DMG_MOD gap — the **SUN → freeze immunity** (`sunnyday.onImmunity('frz')` blocks a
  freeze while the field is Sun; the port used to freeze anyway — the A/B "ice-freeze cluster"). That
  is now FIXED (`turn.rs::try_set_status` sun-freeze gate, pinned FZ1). Admitting the abilities ALSO
  surfaced ONE cascade in an already-modeled mechanic — a Gengar packed with the move ALIAS `wisp`
  (Will-O-Wisp), which the port used to NO-OP (the dex read only canonical ids) — FIXED via
  `gen3_move_aliases.json` + Rust-dex alias resolution (pinned FZ2). No 219/220 or per-battle exclude
  was needed — the enlarged corpus is a CLEAN strict pass.
- **The wired classes (Phase 3) — the ACCURACY pipeline, all folds probe-settled against the
  RESOLVED gen3 dist (`gen3_accuracy_pipeline_v1`; `dex/accmod.rs::AccMod`, consumed by
  `turn.rs::effective_accuracy`/`roll_accuracy`):** the to-hit roll is now
  `effAcc = move.accuracy × the acc/eva STAGE TABLE × the accMod item/ability handlers`, then the
  ONE `random(100) < effAcc` draw. **DRAW-RELEVANT** (unlike the P1/P2 stat/BP folds): a wrong effAcc
  flips a hit↔miss → the crit/damage draws follow only on a hit → the seed desyncs, so the math AND
  draw-order are bit-for-bit. Members (all from the RESOLVED dist — the mod-chain law; the base `.ts`
  shapes differ, e.g. base Bright Powder is `chainModify([3686,4096])` but the gen3 mod REWRITES it to
  a DIRECT `accuracy * 0.9`):
  - **acc/eva STAGE TABLE** — gen3 `[3/3,4/3,5/3,6/3,7/3,8/3,9/3]`, applied inline BEFORE ModifyAccuracy:
    the attacker's accuracy stage (`boosts[5]`: `acc *= table[+s]` / `/= table[-s]`), the defender's
    evasion stage (`boosts[6]`: `acc /= table[+s]` for +, `*= table[-s]` for −). The result stays a
    RAW f64 into the comparison (`random(100)` is an integer 0..99 → `int < effAcc_f64` matches JS).
    (A foe accuracy-drop reaches `boosts[5]` via a modeled move's secondary — e.g. Mud-Slap/Muddy
    Water, the fuzzer's cluster; the extractor already emits accuracy in `secondaryBoosts`.)
  - **ACCURACY_ITEM (2)** — Bright Powder ×0.9 / Lax Incense ×0.95 (`AccOp::Multiply`, DEFENDER-side,
    a DIRECT float that mutates relayVar unconditionally).
  - **ACCURACY ability (3)** — Compound Eyes ×1.3 (attacker), Sand Veil ×0.8-in-sand (defender) +
    its `onImmunity('sandstorm')` sand-chip immunity (the ONLY gen3 weather-chip onImmunity, folded
    into `weather_immune`), Hustle ×3277/4096 for a physical-TYPE move (attacker — the gen3-mod gates
    on `move.type ∈ physicalTypes`, NOT category). All three are `AccOp::Chain` — accumulated into ONE
    4096 modifier applied at the END of runEvent via `modify(acc, modifier)` BUT ONLY when `acc` is a
    NON-NEGATIVE INTEGER (`relayVar === Math.abs(Math.floor(relayVar))`) — so a stage or a direct
    multiply having made `acc` a float SKIPS every chain member (the integer-guard, probe
    `harness/probe_accuracy_intguard.js`, mirrored in `accuracy_chain_modify` + its caller).
  **HUSTLE ships FULLY this phase** — its Atk ×1.5 (`dmgMod`, a DIRECT `this.modify(atk,1.5)` applied
  as a separate pre-chain `modify` in `build_damage_context.atk_direct_modify`, NOT a chainModify
  accumulation) pairs with its acc ×0.8; Hustle is OFF the DATA-ONLY/deferred list. All stage/mod math
  is DRAW-NEUTRAL (still exactly one accuracy draw); the empty/no-stage/no-mod path is BYTE-IDENTICAL
  (`acc` stays the integer `move.accuracy` → `random(100) < accuracy` == the pre-pipeline
  `random_chance(accuracy,100)`).
- **Validation (Phase 3):** `dump_gen3_mechanics.js` grew an `accMod` extraction (the RESOLVED
  onModifyAccuracy/onSourceModifyAccuracy handler → `{op, mod, side, weather?, physicalTypesOnly?}`);
  the `--check` drift gate now pins `accMod` on both items + abilities (132 items + 76 abilities match
  the dist). `dex/accmod.rs` parses it into `AccMod` (`item_acc_mod_fields_parse` /
  `ability_acc_mod_fields_parse`); `turn.rs::effective_accuracy_matches_sim_probe` pins the exact effAcc
  math bit-for-bit vs the sim-captured probe (stage table, each accMod member, the integer-guard). The
  CLASS-SWEEP golden `harness/gen_accuracy_golden.js` → **`tests/accuracy_test.rs`** (7 scenarios × 30
  seeds — the acc-stage fold via Muddy Water, each accMod member, a no-mod control; per-decision
  STATE+HP+**SEED** to game-end, ≥10 miss/accMod rows enforced per member, the control fires 0 accMod;
  byte-reproducible). REVERT-PROVEN (each fold's revert flips a roll → the golden test diverges); 5
  revert-verified pins `AC1`-`AC5` in `tests/regression_test.rs` (AC1 the acc-stage flip — the
  fuzzer's Sand-Attack/Mud-Slap cluster, AC2 Compound Eyes, AC3 Bright Powder, AC4 Hustle Atk×1.5+acc,
  AC5 Sand Veil's sandstorm-chip immunity).
  The e2e regen after adding `brightpowder`/`laxincense` (items) + `compoundeyes`/`sandveil`/`hustle`
  (abilities) to the `MODELED_*` sets grew the filter-clean pool (see the e2e note); STRICT
  `filtered_diverged == 0` over 220 battles, byte-reproducible at the committed knobs.
- **The per-class roadmap (execute against the inventory's class map):**
  1. **ACCURACY pipeline** — **WIRED (Phase 3, see above).** acc/eva stages + the ACCURACY_ITEM class
     (Bright Powder / Lax Incense) + Compound Eyes / Sand Veil + Hustle's ×0.8 physical-acc side (+ its
     Atk ×1.5); the fuzzer's acc-stage cluster is fixed.
  2. **Ability DMG_MOD** — **WIRED (Phase 2, see below).** The pinch family
     Torrent/Blaze/Overgrow/Swarm (BP chain ×1.5 at hp≤⅓, `3·hp<=maxhp` exact), Huge/Pure
     Power (Atk ×2), Guts (Atk ×1.5 statused + the burn-halve suppressed), Marvel Scale
     (Def ×1.5 while the DEFENDER is statused); Hustle now ships fully with the accuracy phase.
  3. **Berries — WIRED + E2E-ADMITTED (BATCH 3, `gen3_berry_trace_shedskin_v1`, 2026-07-07;
     admission + regen 2026-07-08 — see EDGE_CASES.md's batch-3 section for the
     full record).** ONE eatItem consumption mechanism (`MonState::item` — the CURRENT item;
     eaten → NONE permanently) + 22 data-driven `berryEffect` rows (`gen3_items.json`, extractor
     + `--check` gate, obs-neutral): CURE (7 — the Update-site eat, BEFORE the holder's move;
     LUM immediate-in-setStatus after Synchronize, incl. LumRest), HEAL (7 — residual order 10
     subOrder 4 = the LEFTOVERS slot, `2*hp<=maxhp` exact; oran 10 / sitrus 30 / the Figy family
     floor(maxhp/8) + nature-gated confusion random(2,6)), PINCH (7 — `4*hp<=maxhp`; +1 boosts;
     Starf's n≥1 `sample` +2; Lansat's focusenergy crit+2), PP (leppa +10 capped). Probe-settled
     (`probe_berry_rng.js`/`probe_berry_sub_tie_rng.js`: the eat is DRAW-FREE; a berry-vs-Leftovers
     equal-speed mirror draws IDENTICALLY; subs don't trigger it; a KO'd holder never eats;
     **`probe_berry_threshold_boundary.js`: the sim eats AT exact equality** — an EVEN-maxhp board
     [Vaporeon 400] landing on hp == maxhp/2 / maxhp/4 exactly EATS, one-HP-above does not, so
     `<=` is the probe-settled boundary, pinned by BR6 [the prior odd-maxhp boards left `<=`-vs-`<`
     unfalsifiable — the closed reviewer finding]);
     validated by `gen_berry_batch3_golden.js` → `tests/berry_batch3_test.rs` (1280 battles,
     per-decision STATE+STATUS+**ITEM**+BOOSTS+SEED, byte-reproducible) + pins BR1-BR3 + BR6.
     **TRACE + SHED SKIN ship in the same batch** (pins BR4/BR5): trace = the gen3-resolved
     onStart n=1 `randomFoe` sample + a LIVE current-ability copy (`MonState::ability`; no
     copied-onStart in gen3; switch-out reverts; lead-trace draw-free at the seeded start;
     fail-loud `TRACE_COPYABLE` guard — kept in LOCKSTEP with the e2e MODELED∪NOOP allow-lists),
     shed skin = residual order 10 subOrder 3
     `randomChance(33,100)` per STATUSED residual (cure before the DoT; handler gathered
     unconditionally for the tie-shuffle). **The e2e admission grew the filter-clean pool
     585 → 712 / 719** (the biggest admission since Natural Cure) — a CLEAN STRICT pass
     first-try, NO new engine bug, leech finally exercised (354 decisions, was 0).
  4. **PROC_ITEM — WIRED (BATCH 4, see the batch-4 note).** King's Rock (the appended trailing
     `{chance:10, flinch}` secondary for the LISTED moves — `ItemData::flinch_secondary`, the
     execution-derived 130-id list in `gen3_items.json`; [own secondary]→[KR]→[contact proc] order,
     Serene Grace ×2, Shield Dust filters, drawn-not-applied behind a sub, Seismic Toss/Struggle
     proc too — probes `probe_kingsrock_rng.js` + `probe_kingsrock_order_rng.js`) + Focus Band
     (`ItemData::survive_lethal` — `focus_band_damage` at EVERY Damage-event site: move hits,
     burn/sand chips, the leech drain, Spikes, Struggle/Rough-Skin recoil, confusion self-hits
     [effectType Move → CAN survive]; NOT sub-absorbed hits — probes `probe_focusband_rng.js` +
     `probe_focusband_confusion_rng.js`); Quick Claw was already modeled.
  5. **CRIT_ITEM** (Scope Lens / Lucky Punch / Stick → the existing `critRatio` fold),
     DRAIN_ITEM (Shell Bell), BOOST_RESTORE (White Herb), CURE_ITEM (Mental Herb), SPEED_MOD
     (Macho Brace), TAKE_ITEM_GUARD (Mail — with Thief/Knock Off).
  6. **Ability classes beyond DMG_MOD** — **SWITCH_OUT (Natural Cure) ✅ DONE** (`gen3_natural_cure_v1`,
     2026-07-06 — the BIGGEST admission lever, 151 → 449) + **STATUS_IMMUNE ✅ DONE** (`gen3_status_immune_v1`,
     2026-07-06, below — the #2 gap, immunity=97) + **BATCH-1 ✅ DONE** (`gen3_ability_batch1_v1`, 2026-07-07,
     below — **CRIT_IMMUNE** [shellarmor/battlearmor], **WEATHER_SPEED** [chlorophyll/swiftswim],
     **WEATHER_NEGATE** [cloudnine/airlock], **RESIDUAL** [speedboost/raindish]; 525 → **571**, shellarmor the
     lever) + **BATCH-2 ✅ DONE** (`gen3_ability_batch2_v1`, 2026-07-07, below — the DRAW-BEARING
     "reactive" classes: **CONTACT_PROC** [static/poisonpoint/flamebody/effectspore, `randomChance(1,3)` or
     Effect Spore's `random(10)`+`sample(3)` → status the ATTACKER, AFTER the move secondary] +
     **CONTACT-recoil** [roughskin, maxhp/16 draw-free] + **BLOCK** [soundproof / damp / suctioncups] +
     **SYNCHRONIZE** [reflect a foe status to the source]; 571 → **585**, synchronize [the #1 taxonomy gap]
     + effectspore the levers) + **BATCH-3 ✅ DONE** (`gen3_berry_trace_shedskin_v1`, 2026-07-07/08,
     roadmap item 3 above — **TRACE** [was the #1 gap] + **SHED_SKIN** + the 22 berries, all wired,
     goldened, pinned BR1-BR6, AND e2e-admitted: 585 → **712 / 719**, the biggest admission since
     Natural Cure) + **BATCH-4 ✅ DONE** (`gen3_ability_batch4_v1`, 2026-07-08, the FINAL mechanics
     tail — **TRUANT** [was the last team-carry gap, =4], **INNER FOCUS** [=2], **SHADOW_TAG**,
     **CUTE CHARM + the ATTRACT volatile**, **COLOR CHANGE** [the `MonState::types_override` thread
     through the ONE `mon_types` choke point], and the PROC_ITEM pair **KING'S ROCK** [the appended
     trailing 10% flinch secondary over the execution-derived 130-move list] + **FOCUS BAND** [the
     onDamage `randomChance(1,10)` on EVERY Damage event into the holder; survive-at-1 on a lethal
     MOVE hit] — see the batch-4 note below). **STILL DEFERRED (the ONE remaining member):
     FORECAST** (a Castform forme+TYPE change under rain/sun/hail — 0 sample teams; the probes settled
     the weather→forme map / revert-on-end / bench-revert / switch-in re-forme / draw-freeness
     [`probe_forecast_rng.js`], but the forme-change REPORTING surface + the Cloud-Nine
     effective-weather composition stay unprobed, so it is deferred honestly — the e2e filter keeps
     every Castform-Forecast team off the modeled path).
  - **STATUS_IMMUNE ability — DONE as a DATA-DRIVEN class** (`gen3_status_immune_v1`, 2026-07-06). The gen-3
    abilities that grant immunity to a specific MAJOR status: **Limber** (par) / **Insomnia** + **Vital
    Spirit** (slp) / **Immunity** (psn,tox) / **Water Veil** (brn) block via `onSetStatus`; **Magma Armor**
    (frz) blocks via `onImmunity` (BEFORE the SetStatus event). Own Tempo (confusion) + Oblivious (attract)
    block a VOLATILE via `onTryAddVolatile`, NOT a major status → NOT members (Leaf Guard is num 102 = NOT
    gen-3). DATA-DRIVEN: the extractor emits `statusImmune {statuses, phase}` into `gen3_abilities.json`
    (`_GEN3_ABILITY_MECHANICS`, drift-gated by `dump_gen3_mechanics.js --check` which DERIVES it from the
    resolved `onSetStatus`/`onImmunity` handlers; obs-neutral) → `dex/abilities.rs::AbilityData.status_immune`
    (`StatusImmune {statuses, phase: SetStatus|Immunity}`) → `turn.rs::try_set_status` reads it (the
    `Immunity` phase gates BEFORE `set_status_event_shuffle`, the `SetStatus` phase AFTER). **THE
    PROBE-SETTLED DRAW MODEL** (`harness/probe_statusimmune_{rng,setstatus_event,shuffle_size,
    magmaarmor,enumerate}.js`): DRAW-FREE in gen3customgame (the ability is the SetStatus event's only handler
    → size-1 → NO shuffle; Magma Armor blocks before the event) — so admission is SEED-CLEAN. In gen3ou an
    `onSetStatus`-phase ability adds a 3rd SetStatus handler, but it sorts into its OWN speed group (defined
    `speed` beats the clauses' `undefined`), leaving the 2 clauses a SIZE-2 tie → `shuffle(list,1,3)` draws
    EXACTLY ONE `random`, IDENTICAL to the control's `shuffle(list,0,2)` — so the draw COUNT is UNCHANGED
    (this REFUTED + REMOVED the old "size-3 shuffle" fail-loud panic). It was the **#2 e2e team-carry gap**
    (immunity=97); admitting the 6 members (+ moving `insomnia`/`vitalspirit` out of `NOOP_ABILITIES` —
    they genuinely block sleep) grew the filter-clean pool **449 → 525 / 719** (immunity=97, the #2 gap). The
    enlarged corpus surfaced + FIXED ONE real engine bug (NOT the STATUS_IMMUNE class) — the **EMPTY NATURE**
    (e2e_8/e2e_73 carry a Suicune with an OMITTED nature field, which the sim treats as NEUTRAL/Serious but the
    port PANICKED on; `stats.rs::compute_stats` now computes the neutral all-1.0 multipliers for an empty
    nature, VERIFIED vs the sim — pinned `empty_nature_computes_the_neutral_stats`). STRICT 220/220 clean
    (`filtered_diverged == 0`, 11651 decisions), byte-reproducible; `immunity` DROPPED OFF the taxonomy. Golden
    `gen_statusimmune_golden.js` → `statusimmune_test.rs` (480 game-end battles, the block observable on the
    active-status timeline + a stable-md5 byte-reproducibility gate); pins `limber_blocks_paralysis_draw_free`
    / `insomnia_blocks_sleep_draw_free` / `magma_armor_blocks_freeze` / `immunity_blocks_tox_but_not_burn`
    (SI1-SI4) in `regression_test.rs`; probes `probe_statusimmune_*.js`.
  - **BATCH-1 DRAW-FREE / STRUCTURAL classes — DONE** (`gen3_ability_batch1_v1`, 2026-07-07). FOUR ability
    classes, each DRAW-FREE (or draw-neutral) + validated by the class-sweep golden
    `harness/gen_ability_batch1_golden.js` → `tests/ability_batch1_test.rs` (**300 game-end battles, 941
    per-decision STATE+HP+SEED rows + 1582 spe-boost assertions, byte-for-byte**) + the **B1-B4b** revert-verified
    pins in `regression_test.rs` (ground truth `harness/probe_ability_batch1_regression_rng.js`):
    - **CRIT_IMMUNE** (Shell Armor / Battle Armor) — a hit into the holder NEVER crits: the crit
      `randomChance` is DRAWN normally (draw-count unchanged) then `runEvent('CriticalHit')` reads the
      defender's `onCriticalHit=false` and OVERRIDES the crit to false (`turn.rs`, after the crit roll,
      re-resolve damage with `crit=false`). DRAW-FREE — `probe_critimmune_rng.js` (Slash into Battle Armor:
      IDENTICAL draw count vs a Sturdy control, 0 crits vs the control's ~1/8). B1
      `battle_armor_prevents_the_crit_but_draws_the_roll`.
    - **WEATHER_SPEED** (Chlorophyll / Swift Swim) — `onModifySpe chainModify(2)` in EFFECTIVE sun / rain,
      folded into `effective_speed`'s ModifySpe chain (accumulated with paralysis ×0.25 into ONE 4096
      modifier), so the CACHED speed the eachEvent tie-shuffles + the action-order sort read includes the ×2.
      A slow Chlorophyll mon that ties/overtakes the foe at ×2 flips the first-mover. B2
      `chlorophyll_speed_doubles_and_flips_the_first_mover_in_sun`.
    - **WEATHER_NEGATE** (Cloud Nine / Air Lock) — `effective_weather()` returns None while a negater is
      active (the sim's `suppressingWeather()`), so the sand/hail chip is not scheduled AND the weather-speed
      ×2 doesn't apply; the RAW `field.weather` persists (upkeep/counter). B3 `cloud_nine_suppresses_the_sandstorm_chip`.
    - **RESIDUAL** (Speed Boost / Rain Dish) — `ResidualAction::SpeedBoost`/`RainDish` at residualOrder 10
      subOrder 3 (BEFORE Leftovers sub 4), DRAW-FREE: Speed Boost `+1 spe` stage per active turn
      (`if (pokemon.activeTurns)` — a switch-in skips its entry turn; the boost updates the stage but NOT
      `cached_speed`, so it takes effect NEXT turn), Rain Dish `+maxhp/16` heal in EFFECTIVE rain. B4
      `speed_boost_raises_the_spe_stage_by_one_each_active_turn` + B4b `rain_dish_heals_each_end_of_turn_in_rain`.
    **The STEP-1 sun/rain `eachEvent('Weather')` fix shipped with this batch** (see EDGE_CASES + the
    `run_residuals` note): gen3 sun/rain fire the end-of-turn `eachEvent('Weather')` tie-shuffle
    UNCONDITIONALLY (the port used to gate it on Sand|Hail), so a WEATHER-TURN speed tie under sun/rain drew
    one fewer call → a 1-draw desync. FIXED (schedule the field weather-residual off RAW weather for sun/rain,
    off `effective_weather()` for sand/hail — a negater silences sand/hail but NOT sun/rain, probe-verified);
    pinned `sun_rain_weather_turn_tie_draws_the_eachevent_weather_shuffle_seed`. **e2e admission** grew the
    filter-clean pool **525 → 571 / 719** (shellarmor the big lever; STRICT `filtered_diverged == 0` over 220
    battles / 11630 decisions, a CLEAN pass first-try — NO new engine bug). The class-(a) NO-OPS
    `plus`/`minus`/`lightningrod`/`stickyhold` are admitted (the 2026-07 no-op verification tested
    them PARTNER-LESS vs an Insomnia control, `probe_ability_batch1_noop_verify.js` — which missed
    the CROSS-FIELD Plus↔Minus pairing; **`plus`/`minus` are now MODELED**, `gen3_plus_minus_v1`,
    probe `probe_plus_minus_gen3.js`; `lightningrod`/`stickyhold` remain true no-ops); **FORECAST is DEFERRED** (a Castform forme+TYPE change under
    rain/sun/hail — the probe diverges — not a no-op). DEFERRED to batch 2: the DRAW-BEARING procs
    (static/poisonpoint/flamebody/cutecharm/effectspore/synchronize/shedskin/trace/shadowtag/roughskin/colorchange).
  - **BATCH-2 DRAW-BEARING "reactive" classes + block tail — DONE** (`gen3_ability_batch2_v1`, 2026-07-07).
    The draw-bearing ability procs the batch-1 note deferred, each PROBE-settled (the sim is the only oracle;
    `harness/probe_contact_proc_{rng,lands}.js` + `probe_effectspore_sample.js` + `probe_block_abilities_rng.js`
    + `probe_synchronize_rng.js`) + validated by the class-sweep golden `gen_ability_batch2_golden.js` →
    `tests/ability_batch2_test.rs` (**960 game-end battles, per-decision STATE+HP+STATUS+SEED, 3250 seed +
    5540 status assertions, byte-for-byte**) + the **B2-1..B2-7** revert-verified pins in `regression_test.rs`
    (ground truth `harness/probe_ability_batch2_regression_rng.js`):
    - **CONTACT_PROC** (Static par / Poison Point psn / Flame Body brn / Effect Spore slp|par|psn) — a
      DATA-DRIVEN `onDamagingHit` (`AbilityData.contact_proc`, `{statuses, chance, sample}`, extracted from the
      resolved dist): when the HOLDER is hit by a **CONTACT** move (`MoveData::contact`, the new `flags.contact`
      field) that dealt damage, it draws `randomChance(chance)` and (on a pass) inflicts a status ON THE ATTACKER.
      **THE DRAW-MODEL CRUX** (probe-settled): the proc's `randomChance` draws INSIDE `runEvent('DamagingHit')`
      (gen<5, battle-actions.ts:982) which the sim fires **AFTER** the move's OWN `secondaries()` (line 957) — so
      the ORDER in the draw stream is `[move secondary random(100)]` THEN `[contact-proc randomChance]`. Static/PP/
      FB roll ONE `randomChance(1,3)` → the single status; **Effect Spore** rolls `randomChance(1,10)` then, on a
      pass, ONE `sample(["slp","par","psn"])` (a `random(3)`) → the sampled status (the NESTED draw). The proc
      draws even **behind a Substitute** (the sub-absorbed target is still a DamagingHit target) and **on a KO**;
      the status lands on the ATTACKER with the gen-3 type/ability/already-statused gates (in gen3ou the reflected
      `trySetStatus` draws the SetStatus 2-clause shuffle — draw-free in the e2e customgame). Wired in
      `turn.rs::apply_contact_proc`, called from `run_move`'s landed-hit tail (the DamagingHit position, AFTER
      `apply_secondaries`). B2-1 `static_contact_proc_paralyzes_the_attacker` + B2-2
      `effect_spore_samples_a_status_onto_the_attacker` (the nested-sample draw pin).
    - **CONTACT recoil** (Rough Skin, `AbilityData.contact_recoil`) — DRAW-FREE `baseMaxhp/16` recoil to the
      attacker on a contact hit. B2-3 `rough_skin_recoils_the_attacker_draw_free` (STATE + the IDENTICAL-to-control
      seed proving draw-freeness).
    - **BLOCK** (`AbilityData.blocks_explosion`/`blocks_sound`/`blocks_phaze_drag`): **Damp** cancels Explosion /
      Self-Destruct at `runEvent('TryMove')` (battle-actions.ts:412, BEFORE the self-KO faint at 422 AND the
      accuracy roll) — the user does NOT self-KO, the move draws NOTHING (a big draw-count drop; `|cant|<damp
      holder>|ability: Damp|<Move>|[of] <user>`); fires for EITHER side's Explosion (`onAnyTryMove`), via
      `turn.rs::damp_holder`. **Soundproof** is IMMUNE to a SOUND move (`MoveData::is_sound`, the new `flags.sound`
      field — Sing / Grass Whistle in the status arm + Roar in the phaze arm): accuracy drawn, then `-immune|
      [from] ability: Soundproof`, no status / no drag / no sample (the same draw model as a type-immune move).
      **Suction Cups** blocks a phaze DRAG: the sim's `forceSwitch` runs `runEvent('DragOut')` in the MOVE BODY
      (battle-actions.ts:1166) → Suction Cups' `onDragOut` returns `null` → `forceSwitchFlag` NOT set → the
      runAction-tail `dragIn` never fires → **NO `sample` draw** (`-activate Suction Cups`, the holder STAYS), via
      the phaze arm's Suction-Cups gate. B2-4 `damp_cancels_explosion_no_self_ko` + B2-5 `soundproof_immune_to_sing`
      + B2-6 `suction_cups_blocks_the_roar_drag_no_sample`.
    - **SYNCHRONIZE** (`AbilityData.synchronize`) — when the holder is inflicted a MAJOR status by a FOE SOURCE (a
      status MOVE or a damaging move's SECONDARY), REFLECTS it back to that source (slp/frz EXEMPT; tox→psn). Wired
      at the SINGLE status choke point `turn.rs::try_set_status` (now `source: Option<(usize,usize)>`-threaded
      through its 3 real callers): after a foe-sourced major status applies to a Synchronize holder, it recurses
      `try_set_status(source, refl, None, …)` (source-less → no ping-pong). **DRAW-FREE in gen3customgame** (the
      reflected status draws no clause shuffle — probe-verified identical draws to a no-op control); in gen3ou it
      draws the reflected status's own 2-clause SetStatus shuffle. B2-7 `synchronize_reflects_paralysis_to_the_caster`.
    **DATA + move flags**: the CONTACT_PROC params + `contactRecoil`/`blocksSound`/`blocksExplosion`/
    `blocksPhazeDrag`/`synchronize` are extracted into `gen3_abilities.json` (`_GEN3_ABILITY_MECHANICS`, drift-gated
    by `dump_gen3_mechanics.js --check` which DERIVES the same from the resolved `onDamagingHit`/`onTryHit`/
    `onAnyTryMove`/`onDragOut`/`onAfterSetStatus` handlers), and the `contact` + `sound` move flags into
    `gen3_moves.json` (`flags.contact`/`flags.sound`). All obs-neutral (the Python `agents.gen3_data` facade ignores
    them; extractor-parity green). **e2e admission** grew the filter-clean pool **571 → 585 / 719** (`synchronize`
    [the #1 taxonomy gap] + `effectspore` the levers; STRICT `filtered_diverged == 0` over 220 battles / 11790
    decisions, a CLEAN pass first-try — **NO new engine bug**; the CONTACT_PROC draw-after-secondary + the BLOCK
    draw-count drops + the draw-free Synchronize reflect composed cleanly). Trace / Shed Skin / the berries
    have since shipped + been e2e-admitted in **batch 3** (`gen3_berry_trace_shedskin_v1`, 585 → 712 / 719 —
    see the roadmap item 3); Cute Charm + Color Change have since shipped in **batch 4** (below).
  - **BATCH-4 — the FINAL mechanics tail, DONE** (`gen3_ability_batch4_v1`, 2026-07-08). The last seven
    members, every draw model PROBE-settled (`harness/probe_{truant,truant_edges,innerfocus,shadowtag,
    cutecharm_attract,colorchange,kingsrock,kingsrock_order,focusband,focusband_confusion}_rng.js` +
    the shared `probe_batch4_lib.js`), validated by the class-sweep golden `gen_ability_batch4_golden.js`
    → `tests/ability_batch4_test.rs` (**21 scenarios × 60 seeds = 1260 game-end battles, 4220
    per-decision STATE+HP+STATUS+TRAPPED+SEED rows, byte-for-byte, md5-pinned**) + the **B4-1..B4-7**
    revert-verified pins in `regression_test.rs` (ground truth `harness/probe_batch4_regression_rng.js`):
    - **TRUANT** (`MonState::truant_turn`) — onBeforeMove priority **9** (slp/frz 10 > truant 9 >
      flinch 8): `|cant|…|ability: Truant` iff the flag, DRAW-FREE (a loaf turn draws NOTHING — no
      para roll [Q2b], no PP); `onSwitchIn` arms `turn !== 0`; the order-**27** residual TOGGLES it —
      so a mid-turn entrant (pivot/drag/action-faint replacement) is toggled back and MOVES its first
      full turn, while a POST-residual DoT-KO replacement keeps `true` and LOAFS (edge E1); a
      speed-tied Truant MIRROR adds ONE order-27 tie-shuffle draw (Q4's 9-vs-8).
    - **INNER FOCUS** — blocks the flinch volatile at the APPLY (the flinch-secondary `random(100)`
      STILL draws — draw-count-IDENTICAL to a landed flinch, probe-pinned vs a Thick Fat control;
      CONTRAST Shield Dust's filter-the-draw). One gate in `apply_one_secondary`'s flinch arm — it
      covers a move's own flinch AND the King's Rock appended one.
    - **SHADOW TAG** — `is_trapped` traps the foe UNCONDITIONALLY (no grounded/type gate — a Flying
      Skarmory is trapped; a MIRROR is MUTUALLY trapped — `onFoeTrapPokemon` has no fellow-holder
      exemption, only the display-only Maybe does), DRAW-FREE (a Wobbuffet mirror's draw count ==
      a no-trap control's; vs Magnet Pull's onAny* draws).
    - **CUTE CHARM + the ATTRACT volatile** (`AbilityData::contact_attract` +
      `MonState::{gender, attract}`) — the CC roll `randomChance(1,3)` draws UNCONDITIONALLY on a
      damaging contact hit (the GENDER gate lives INSIDE `attract.onStart` — F-into-F / genderless
      draw the roll, the volatile fails draw-free); attract: onBeforeMove priority **2** (confusion
      3 > attract 2 > par 1), `-activate` ALWAYS then `randomChance(1,2)` (cant on pass), NO
      duration, cleared when the SOURCE leaves the field (onUpdate) or the HOLDER switches out.
      Gender is parsed from the packed set; an UNSPECIFIED gender on a ratio species makes the SIM
      draw `battle.sample(['M','F'])` at construction (an unmodeled init draw) → the attract compare
      PANICS fail-loud on an unknown gender, and every golden pins genders explicitly.
    - **COLOR CHANGE** (`MonState::types_override` + the ONE **`mon_types`** choke point, which now
      serves EVERY live type read: STAB, chart effectiveness, status type-immunity, sand-chip
      immunity, Magnet Pull's Steel gate, Leech Seed's Grass gate) — onDamagingHit sets
      `[move.type]`: DRAW-FREE; NOT behind a sub (the mon's DamagingHit never fires — the batch-2
      lesson, probe-verified); not on the KO hit; never for typeless `???`; no-op on an
      already-matching type; switch-out reverts.
    - **KING'S ROCK + FOCUS BAND** — see roadmap item 4 (PROC_ITEM, WIRED).
    The batch surfaced NO new engine bug — the golden passed bit-for-bit first-try (two HARNESS
    fixes only: the first-mover scan now counts a voluntary `|switch|` as the first actor [matching
    the port's action queue], and one scenario was redesigned off the sim's accept-then-`|cant|nopp`
    0-PP path, which the port's strict request-legality gate deliberately rejects — scripted goldens
    stay within request-legal choices).
  - **SWITCH_OUT ability — DONE as a class** (`gen3_natural_cure_v1`, 2026-07-06). NATURAL CURE (the sole
    gen-3 member) cures an ALIVE outgoing holder's major status on switch-OUT (voluntary pivot OR
    phaze-DRAG-out; the tox stage + sleep counter reset), **DRAW-FREE** (`onSwitchOut`, `onCheckShow`
    undefined — resolving the long-deferred "NaturalCure CheckShow" draw question: NO CheckShow gate; the
    cure + its `[silent]` `-curestatus` reveal consume ZERO PRNG, so it is SEED-NEUTRAL; probe-settled by
    `harness/probe_naturalcure_rng.js` + `probe_naturalcure_dump.js`). It is an ENGINE FLAG (a `status =
    None` clear in `turn.rs::execute_switch`, gated on `naturalcure` + `!fainted`), NOT a `dmgMod` data row
    — `gen3_abilities.json` unchanged (obs-neutral). It was the **#1 e2e team-carry gap** (naturalcure=254);
    admitting it grew the filter-clean pool **151 → 449** — the biggest single lever yet — a CLEAN STRICT
    e2e pass (no new engine bug). Golden `gen_naturalcure_golden.js` → `naturalcure_test.rs` (280 game-end
    battles, the cure observable on the active-status timeline); pins `natural_cure_*` (NC1-NC3) in
    `regression_test.rs`; probes `probe_naturalcure_{dump,rng,scenario,regression_rng}.js`.
  - **TYPE-INTERACTION abilities — COMPLETE as a class** (Levitate immunity, Water/Volt Absorb
    heal+immunity, and now **Flash Fire's ×1.5 boost** — `gen3_flashfire_boost_v1`, 2026-07-06,
    the last gap). FF is NOT a `dmgMod` fold (its boost is a `flash_fire` activation VOLATILE +
    a ModifyDamagePhase1 damage fold, structurally like screens — an engine flag, not a data
    row), so it lives outside the DMG_MOD framework. The A/B fuzzer's evidence-based **#1 STATE
    cluster** (fireblast+flamethrower dominate the STATE repros, 397/402 with an FF mon); of 200
    replayed FF-team STATE repros, **185 (92.5%) flip to `ok`**. Golden `gen_flashfire_golden.js`
    → `flashfire_test.rs`; pins `flash_fire_*` in `regression_test.rs`; probe `probe_flashfire_rng.js`.

### Handler-completeness audit (`gen3_handler_audit_v1`, 2026-07-10) — the dispatch-bus guarantee as a STATIC gate

The port implements effects AT-SITE (no generic runEvent bus). The recurring bug class that
allowed: an effect carries a handler at a hook we never enumerated or hand-placed at the wrong
site — Immunity's onUpdate cure, Cloud Nine's onEnd WeatherChange, Plus/Minus's cross-field
onModifySpA, the tox onSwitchIn reset, sun/rain's unguarded onFieldResidual, facade's
onBasePower. The audit closes the class STATICALLY:

- **The enumerator** — `harness/dump_gen3_handlers.js` reads the RESOLVED `Dex.mod('gen3')`
  (the mod-chain law) and enumerates EVERY handler-bearing key (`on*` functions AND the numeric
  priority/order/subOrder metadata AND draw-relevant declaratives like `durationCallback` /
  `duration` / a move's `secondaries`/`selfdestruct`/`neverMiss`/…) on EVERY effect in the
  port's REACHABLE surface: the MODELED∪NOOP abilities + MODELED items (gen_e2e_fuzz.js — the
  one source of truth) + every condition the engine can enter (the 6 statuses, the modeled
  volatiles, the 4 weathers, spikes, the Sleep/Freeze Clause rules; `condition` sub-objects on
  surface effects auto-join) + every `isModeledMove` move + `struggle`. Each (effect, hook) row
  carries an FNV-1a **body fingerprint** of the resolved source, so a semantic change in the
  dist is DETECTED. **664 rows** (74 abilities / 59 items / 27 conditions / 168 moves) as of
  the first run. Deterministic (byte-stable regen).
- **The manifest** — `tests/vectors/gen3_handler_audit.json` (+ the human census
  `gen3_handler_audit.md`): one row per (effect, hook) with an explicit
  `disposition: implemented | noop_justified | unreachable_justified | failloud_guarded`, the
  fingerprint, and (for `implemented`) an **anchor** `file.rs::symbol` that must grep in
  `src/`. The dispositions are CURATED CODE in
  `harness/handler_audit_dispositions.js` (per-row entries + tight class rules — e.g. every
  `berryEffect` hook → the berry engine anchors; order/priority metadata inherits its sibling
  handler's disposition). First-run census: **595 implemented / 39 noop_justified /
  30 unreachable_justified**; the deferred fail-loud universe (Forecast, Liquid Ooze, the
  OHKO/Psywave/Counter family, the protocol-line gaps) is OUTSIDE the surface by construction
  and documented in the manifest's `_meta.excluded_deferred` — admitting one to a MODELED set
  pulls its handlers INTO the surface and the gate then demands rows.
- **The gate** — `node harness/dump_gen3_handlers.js --audit` FAILS on: (a) a resolved key
  with NO manifest row (a NEW/unnoticed handler), (b) a stale manifest row, (c) a body
  FINGERPRINT drift (re-probe before re-accepting), (d) a dead `implemented` anchor. Wired
  into `cargo test` as **`tests/handler_audit_test.rs`** (fails loudly if node/dist are
  unavailable — a silently-skipped completeness gate is no gate). All four failure modes
  perturbation-demonstrated. Regenerate after a triage:
  `node src/rust_sim/harness/dump_gen3_handlers.js`.
- **The first run surfaced TWO REAL MISSES, both fixed bit-for-bit + pinned:**
  1. **JUMP KICK / HIGH JUMP KICK crash** (`gen3_jump_kick_crash_v1`) — both passed
     `isModeledMove` (plain damaging moves) but carry an `onMoveFail`: a FAILED JK (miss or
     Protect block) crashes the USER for `clampIntRange(getDamage/2, 1, floor(TARGET.maxhp/2))`,
     and the crash's `getDamage` DRAWS crit + the 16-way roll (+2 draws vs a missed control).
     Probe-settled (`probe_jumpkick_crash_rng.js`): fires through Protect; NOT vs a
     Fighting-immune (Ghost) target; the crash can faint the user; Focus Band can survive it
     (a MOVE-effect Damage event). Fixed in `turn.rs::apply_jump_kick_crash` (the miss +
     protect-block returns); pins HA1 `jump_kick_miss_crashes_the_user_with_crit_and_roll_draws`
     + HA1b `jump_kick_crashes_through_a_protect_block` (revert-verified, sim ground truth
     `probe_handler_audit_regression_rng.js`). Zero e2e/A-B corpus exposure (no team carries it)
     — a pure latent bug.
  2. **FREEZE CLAUSE MOD** (`gen3_freeze_clause_v1`) — the engine modeled Sleep Clause but not
     Freeze Clause: under gen3ou a SECOND foe-inflicted freeze on a side must FAIL (the rule's
     `onSetStatus` returns false INSIDE the already-drawn SetStatus event — DRAW-FREE block;
     a fainted mon's status is `'fnt'`, so only LIVING frozen mons count). Probe-settled
     (`probe_freeze_clause_rng.js`); fixed in `turn.rs::try_set_status` + `side_has_frozen`
     (mirrors the sleep path, same `sleep_clause` format flag); pin HA2
     `freeze_clause_blocks_the_second_freeze_in_gen3ou` (revert-verified). Unreachable in the
     e2e/A-B (both run gen3customgame) — a latent bug for every clause format.
  Notable already-modeled rows the audit CONFIRMED (previously bug-class members): the
  fire-hit thaw (`frz.onDamagingHit`), Early Bird's double sleep decrement, Pressure's +1 PP,
  the Trace-route STATUS_IMMUNE onUpdate cures, Oblivious's attract gate, Rock Head's
  Struggle exemption (a true no-op — Struggle is the only modeled recoil).

## Protocol emission (level-2, Phase 1 + Phase 2): the byte-identical `|...|` stream

This is the **level-2** goal — emit the byte-identical OMNISCIENT `|...|` protocol stream our
poke-env fork parses, so the port is a drop-in behind the bridge. The engine is already
bit-for-bit RNG+state faithful; this layer is a **side output** of events that ALREADY happened.

- **The emit API** (`protocol.rs`): `ProtocolBuilder` is an **append-only, PRNG-free** line
  buffer on `BattleState` (the `log` field) — with ONE sim-mirroring exception:
  `attr_last_move_still()`, the port of `Battle.attrLastMove('[still]')` (blank the last `|move|`
  line's target + append `|[still]`), for fail forms the sim itself decides RETROACTIVELY, after
  draws the announce preceded (today only Disable's onStart 0-PP-guard rejection,
  `gen3_disable_zero_pp_v1`). The engine pushes lines at hook points in `turn.rs`;
  the fiddly formatting lives in ONE place — `MonRef` (`p<N>a: <Name>`; a `SideRef` is
  `p<N>: <PlayerName>`), `HpStatus` (the three variants `x/y` / `x/y <status>` / `0 fnt`, the #1
  correctness point), `Cause` (`[from] item: <Item>` / `[from] ability: <A>` / `[from] move: <M>` /
  `[from] <bare>`), `STAT_TOKENS` (the `-boost`/`-unboost` stat names). Move display names come
  from the dex (`MoveData.name` — Title-Case spaced). A `MonRef`'s IDENT name is the mon's ON-FIELD
  NICKNAME (`turn.rs::display_name` = the packed set's `set.name`, ← `SpeciesData.name` only when the
  set has no nickname — mirroring Showdown's `Pokemon.name = set.name || species.name`), NOT the
  species: poke-env keys each mon by this `p<N>a: <nick>` token, so rendering the species there
  (e.g. `p1a: Zapdos` for a Zapdos nicknamed `Electhor`) makes poke-env fail to match the mon it
  already tracks and try to ADD a 7th — the localized/nicknamed-team overflow crash
  (`gen3_nickname_ident_v1`, pinned by `regression_test::nicknamed_mon_renders_nickname_in_every_ident_not_species`).
  The SPECIES name (`turn.rs::species_name`) lives ONLY in the `|switch|`/`|drag|` DETAILS field
  (`|switch|p1a: Electhor|Zapdos|<hp>`). Disabled by
  default (`ProtocolBuilder::new()` → off): `run_full_battle` never enables it, so the seed suite
  keeps an empty, cost-free buffer AND every emit hook is a no-op that touches nothing.
  `run_full_battle_logged` enables it, emits the framing, runs the SAME `run_full_battle`, and
  returns `(BattleOutcome, Vec<ProtocolLine>)`.

- **Phase 1 emits** (the high-frequency core): the battle-init framing (`|t:|` normalized /
  `|gametype|singles` / `|player|` / `|gen|3` / `|tier|` / `|rule|` / the blank `|` separator /
  `|teamsize|` / `|start` / the leads' `|switch|`), the turn/phase markers (`|turn|N`, `|upkeep`),
  `|move|` (+ `[miss]`/`[still]`), `|switch|` / `|drag|`, `|-damage|` (all HP variants + the
  residual `[from]` tags — `Sandstorm` / `brn`/`psn`/`tox` / `Spikes`), `|-heal|` (+ `[from] item:
  Leftovers`), `|faint|`, `|-crit|`, `|-supereffective|`, `|-resisted|`, `|-immune|`, `|-miss|`,
  and `|win|` / `|tie|`. **Line ORDER inside a move** = `|move|` → `|-supereffective|`/`|-resisted|`
  → `|-crit|` → `|-damage|` (verified vs the golden). The `|faint|` ORDER follows Showdown's
  `faintQueue` (the self-KO'd Explosion USER before its KO'd target) via `faint_emit_queue` —
  since fix-queue #4 that queue is DRAW-BEARING, not emission-only (`gen3_faint_queue_order_v1`:
  `process_faints` drains it unconditionally so corpse ability-`End` events fire in enqueue order). The `|turn|N+1` marker is emitted at the TOP of the NEXT turn's outer-loop
  iteration (AFTER the previous turn's `|upkeep` + any forced replacement it triggered — a
  residual-faint's replacement `|switch|` precedes `|turn|N+1`), mirroring `makeRequest('move')`.

- **Phase 2 emits** (weather / boost / status / volatile / side-condition): the switch-in
  framing ability lines — `|-ability|<lead>|Intimidate|boost` + `|-unboost|<foe>|atk|1`
  (immunity-gated) + `|-weather|<W>|[from] ability: <A>|[of] <lead>` (RECONSTRUCTED in
  `emit_framing`'s `emit_switchin_ability_lines`, faster-lead first, from the post-switch-in state
  the construction already resolved); the STATUS-move `|move|` ANNOUNCE (`run_status_move`'s top —
  a self-target move renders the USER, a foe/foeSide move the FOE ACTIVE; the lone top-level
  `[still]` did-nothing form is **Spikes at the 3-layer cap**); `|-status|` (+ `[from] move: Rest`
  from `run_rest`) / `|-curestatus|`+`[msg]` (`on_before_move`'s slp/frz wake) / `|cant|`
  (par/slp/frz/flinch, `on_before_move`); `|-boost|`/`|-unboost|` by the CLAMPED delta's sign
  (setup `self_boost_spec` + the secondary `apply_secondary_boost` — a into-cap delta 0 emits
  nothing); `|-weather|<W>|[upkeep]` (the end-of-turn tick, at the TOP of `apply_weather_chip`);
  `|-fail|` (Recover-at-full / Rest-at-full / Spikes-at-cap / a phaze-with-no-bench / a failed
  Protect stall roll; + `move: Substitute`+`[weak]` for the Substitute can't-afford fail vs the
  bare `move: Substitute` already-up fail); `|-sidestart|<side>|Spikes`; the Substitute
  `|-start|`/`|-end|` (up / break) + `|-activate|…Substitute|[damage]` (a SURVIVED absorb, INSTEAD
  of `|-damage|`) + the sub-cost `|-damage|`; the Protect `|-singleturn|` (success) +
  `|-activate|…Protect` (block); and Rest's `|-heal|<user>|<HP> slp|[silent]`. The weather chip
  `|-damage|` ORDER now reads the `each_event_shuffle` RETURN (the shuffled side order — a
  same-species speed-TIE Snorlax-vs-Snorlax mirror chips in the shuffle's permutation, revealed by
  the golden's `-damage` order), a state-/seed-INVARIANT read that only fixes the emitted order.

- **NO scenario-level deferrals remain** (`DEFERRED_SCENARIOS` is EMPTY): `debug` (poke-env-ignored
  free-form sim text — a deliberate non-emit) stays filtered from BOTH sides. The two
  `status_para_and_boost_drop` / `secondary_status_flinch` scenarios (formerly deferred for a
  forced-replacement REQUEST-BOUNDARY resume "phantom") are now ASSERTED byte-exact — the blocker was
  NOT Seismic Toss (modeled bit-for-bit; `fixeddamage_test.rs` + FD1–FD4) and NOT a protocol gap but two
  fixes (`gen3_forced_replacement_resume_v1`): (1) the `run_full_battle` **reject-and-re-request gate**
  (`move_decision_is_legal`) — the "phantom" was really an INVALID scripted move slot after a
  replacement changed the active mon to one with FEWER moves (a 3-move Tyranitar → a 2-move Snorlax, then
  a scripted `move 3`), which the sim's `side.choose` REJECTS drawing 0 (probe
  `harness/probe_forced_replacement_queue.js`: the rejected choice → no `commitChoices`, no `turnLoop`);
  the port used to RUN a full turn for it (its own move no-op'd but the FOE's move + residual + Quick
  Claw drew) → the seed + line stream diverged. Now the invalid decision is SKIPPED (draw-free, re-pull
  the next) — VERIFIED zero-draw so the seed suite is byte-identical. (2) the standalone status-move
  **already-statused `|-fail|` emission** (`foe_status_move_fail`, see the turn.rs STANDALONE STATUS
  MOVES note) — a `|-fail|<target>|par` line the port omitted. The three `recover_and_rest` Struggle
  battles that were formerly per-battle-skipped now REPLAY byte-exact (`gen3_pp_tracking_v1` — PP
  tracking + the forced-Struggle substitution + the Choice-Band lock + the Struggle `|move|` /
  `|-damage|…|[from] Recoil|[of]` lines via `ProtocolBuilder::damage_of`); `unreplayable_move` catches
  NOTHING now. `-sethp`/`-cureteam`/`-setboost`/the Haze family/`-item`/`-prepare`/… land with their
  mechanic (still unbuilt).

- **OBSERVATION-ONLY (the load-bearing guarantee).** Emission draws NO PRNG and mutates no
  asserted state, so wiring it changes NO seed assertion. THE PROOF: the ENTIRE existing seed suite
  (`battle_test`'s 2034 cross-turn seed assertions, `fullbattle` 2053, `secondary`, the `e2e_fuzz`
  STRICT gate, every move layer, every regression pin — at the Phase-2 landing that was e2e 14228 +
  22 pins; the CURRENT tree is e2e 11673 + 44 pins, the corpus/pin growth from later layers) stays
  green with BYTE-IDENTICAL seed counts after Phase 2 — run the full suite before/after and diff (it does). The only
  engine-behaviour changes are the two emission-line REORDERS (the `|turn|N+1` marker moved to the
  next-turn top; the weather chip reads the shuffle permutation) — both provably state-/seed-
  invariant (the shuffle already drew; distinct/saturating mons) — which the seed suite re-confirms.

- **Phase 3 emits** (`gen3_protocol_phase3_v1` — the formerly-deferred long tail, each byte-verified
  by a NEW capture scenario): the **taunt/disable residual `-end`s** (`|-end|<mon>|move: Taunt|[silent]`
  / `|-end|<mon>|Disable`) + the **Disable retro-edit forms** (a missed Disable gains the `[miss]` attr
  via the new `attr_last_move_miss`; the no-lastMove & already-disabled fails retro-edit to
  `|move|…|Disable||[still]` + `|-fail|<user>` — the re-Taunt fail likewise, fixing the old
  fail-on-target form); the **status-move `[miss]` retro-edit generally** (Hypnosis/WoW/Toxic — the
  pre-Phase-3 "status announces never carry [miss]" claim was a corpus artifact); the **Trace reveal**
  (`|-ability|<mon>|<Copied>|Trace|[from] ability: Trace|[of] <foe>`, lead + mid-battle re-trace); the
  **Flash Fire cycle** (`|-start|…|ability: Flash Fire` on the arm INSTEAD of `-immune`;
  `|-immune|…|[from] ability: Flash Fire` when already armed; `|-end|…|ability: Flash Fire|[silent]`
  on an ALIVE switch-out — a FAINT emits nothing, capture-proven); the **STATUS_IMMUNE block lines**
  (`|-immune|<target>|[from] ability: Limber/Water Veil/Immunity/Insomnia` — ONLY for a status-MOVE
  source, `try_set_status_impl(announce_immune_block)`; a blocked SECONDARY is silent, and the
  Immunity-phase Magma Armor block stays silent — no direct gen-3 freeze move exists to capture);
  the **Synchronize→Lum interleave** (`-status holder → -status source [from] ability: Synchronize
  [of] holder → -enditem [eat] → -curestatus [msg]` — the reflect's `-status` form now emitted by the
  recursive apply itself via `sync_reveal`, so the source's Lum tail lands after it; + the LumRest
  chain byte-verified); the **MID-BATTLE switch-in ability lines** (`emit_ability_start_lines`, shared
  with the framing): weather SET on a REAL change, `|-ability|<mon>|Pressure|[silent]`, and Intimidate
  three ways — `-unboost`, the Clear-Body blocked form `|-fail|<foe>|unboost|[from] ability: Clear
  Body|[of] <foe>`, and the Substitute case (NO `-ability`, just the gen3
  `|-hint|In Gen 3, Intimidate does not activate if every target has a Substitute.`); **Leech Seed**
  (`|-start|…|move: Leech Seed`, the residual `|-damage|…|[from] Leech Seed|[of] <seeder-active>` +
  `|-heal|<seeder-active>|…|[silent]`, the Grass `-immune`, the re-seed `[still]`+`-fail`, the
  `[miss]` form); **Splash** `|-nothing`; **Pay Day** `|-fieldactivate|move: Pay Day` (after the
  `-damage`, direct hits only); and the **Rest-at-full-HP `|-fail|<user>|heal`** detail token (a real
  byte bug the per-write gate caught — the Phase-2 corpus never realized a full-HP Rest). **Protocol
  review FINDINGS F1-F3 now CLOSED** (`gen3_protocol_phase3_review_v1`, byte-verified by 3 NEW capture
  scenarios — probes `harness/probe_f1_f2_f3_lines.js` / `probe_f2_ff_armed_miss.js` /
  `probe_levitate_miss.js`): **F1** a sub-blocked Leech Seed now emits `|move|<user>|Leech Seed||[still]`
  + `|-fail|<user>` (IDENTICAL to the already-seeded form — scenario `leechseed_into_substitute`);
  **F2** a MISSED `onTryHit`-class ABILITY immunity (Flash Fire / Water&Volt Absorb — POST-accuracy)
  now emits `[miss]`+`-miss`, NOT `-immune` (gen3 rolls accuracy BEFORE TryHit; Levitate + type-chart
  0× stay pre-accuracy `-immune` even on a would-be miss, probe-confirmed 40/40 — scenarios
  `flashfire_tryhit_miss` / `waterabsorb_tryhit_miss`); **F3** a LANDED Water/Volt Absorb now emits
  `|-immune|<t>|[from] ability: Water Absorb` (resp. Volt Absorb), not a plain `-immune`. STILL
  UN-EMITTED (uncapturable, documented at their sites): a sub-absorbed Pay Day's form, the
  WoW-arms-a-non-Fire-FF status-path `-start` (impossible in gen-3 OU), Own Tempo's confusion-move
  `-immune` (no confusion-inflicting volatile move modeled). Phase 3 also fixed two REQUEST-BOUNDARY
  gaps the blind-plan scenarios exposed (both pinned + revert-verified): **per-side choice acceptance**
  (`side.choose` holds one side's valid choice while the other's is rejected — the old whole-decision
  skip mis-mapped split-accept boundaries; pin
  `per_side_choice_acceptance_maps_split_accept_boundaries_to_the_sims_seeds` replays the golden's DEC
  rows at SEED level) and the **switch-to-fainted reject** (pin
  `rejected_switch_to_a_fainted_slot_is_skipped_draw_free`); the forced-replacement pull got the same
  per-side accumulation (a double replacement may arrive as two one-sided writes — the `write_line`
  pattern).

- **The byte-differential gate** (`tests/protocol_test.rs`): replays the capture golden through
  `run_full_battle_logged`, FILTERS both the golden's lines and the engine's output to the gated
  types (only `debug` + `error` + still-deferred-mechanic lines dropped from BOTH; `|t:|`
  normalized), and asserts BYTE-EQUALITY per line, in order, with a first-divergence panic. A
  TRUNCATED golden (no terminal `|win|`/`|tie|` — the capture hit a decision/turn cap mid-stall,
  e.g. `spikes_and_phaze/2`'s infinite Spikes-at-cap↔immune-EQ loop) is asserted as a byte-exact
  PREFIX of the longer engine output. **Result: 132 battles asserted, 19348 lines byte-equal**
  (up from the F1-F3 review's 114 / 16115; from Phase-2's 66 / 8721; 63 / 7223, 51 / 5630, and
  Phase-1's 30 / 1512), across ALL 22 scenarios — the 11 Phase-1/2 + the 8 Phase-3 (`taunt_lifecycle` /
  `disable_lifecycle` / `trace_switchin` / `flashfire_cycle` / `status_immune_lines` /
  `synchronize_lum_rest` / `midswitch_ability_lines` / `leechseed_splash_payday`) + the 3
  F1-F3-review scenarios (`leechseed_into_substitute` / `flashfire_tryhit_miss` /
  `waterabsorb_tryhit_miss`).
  **0 battles DEFERRED** (`DEFERRED_SCENARIOS` empty) + **0 battles skipped** (`unreplayable_move`
  catches nothing now that Struggle is modeled — `>= 3` battles now REPLAY a forced Struggle
  byte-exact). The
  formatters are ALSO pinned by deterministic unit gates in `protocol.rs` (`HpStatus` three
  variants, the `p1a:`/`p1:` split, `Cause` item/ability/move/bare, the `[still]`+`[miss]` `|move|`
  forms, the `-status` `[from] move: Rest` variant, `cant`, `-curestatus` `[msg]`, `-boost`/
  `-unboost` by sign, `-weather` set-vs-`[upkeep]`, the Substitute `-start`/`-end`/`-fail [weak]`
  forms, the Rest `[silent]` heal, the disabled-builder-emits-nothing invariant, and the Phase-3
  forms — `attr_last_move_miss`, `ability_silent`/`ability_traced`, `fail_unboost_from_ability`,
  `hint`/`nothing`/`fieldactivate_move`, the taunt/disable `-end`s, the `-fail|heal` detail).

- **The drop-in endgame — BUILT** (`gen3_writeline_stream_v1`): `battle.rs`'s
  **`BattleStream::write_line`** accepts the bridge's command stream (`>start` / `>player pN` /
  `>pN move K|switch N`) and returns, PER WRITE, exactly the omniscient chunk the real Node
  `BattleStream` flushes for that write — gated by **`tests/writeline_test.rs`** against the
  per-write capture `harness/gen_writeline_capture.js` (the SAME 19-scenario corpus at 2 fresh
  seeds: **44 battles / 2377 writes / 7276 filtered lines, all chunks byte-equal**). Chunk
  attribution (probe-verified): `>start` → `|t:|`+`|gametype`; each `>player` → its `|player|`
  line (the second also the whole framing through `|turn|1`); a choice write → nothing until the
  boundary completes, then the whole turn chunk ENDING with the eager `|turn|N+1` (the sim's
  `makeRequest` flush — the port now emits the marker at turn END and the batch separator+`|t:|`
  at the COMMIT, concatenation-identical, chunk-correct). Internals + honest scope (replay-from-
  genesis; the pre-first-decision seed convention; request frames/privacy fold out of scope) are
  on the `battle.rs` module row above. Design: `PROTOCOL_EMISSION_DESIGN.md`; line grammar:
  `tests/vectors/protocol_inventory.md`.

## Conventions

- **std-only, zero dependencies.** Determinism + a no-network `cargo test` are
  the point. Add a dep only with a clear reason, and never one in the
  deterministic battle path.
- **Generation-generic.** Gen 3 OU is the only target now, but don't hard-code
  Gen-3 constants into the engine — put them behind the generation parameter
  (e.g. `Dex::for_gen(gen)`, `moves::derive_category(gen, …)`), mirroring
  Showdown's gen9→gen3 mod-delta layering, so other gens are a data layer + a
  branch, not an engine rewrite. Do not add anything that would break future gens.
- **Data source of truth** is this repo's `data/pokemon/*.json`, read the same
  way `agents.gen3_data` reads it — so Python and Rust agree by construction.

## Next steps (engine, not yet built)

1. ~~Dex loader over `data/pokemon/*.json` (+ a parity check vs `agents.gen3_data`).~~ **Done** (`dex/`).
2. ~~Team codec (`Teams.unpack`/`pack`, the packed string from `>player`).~~ **Done** (`team.rs`).
3. ~~Stat computation: `PokemonSet` + dex → in-sim stats (gen-3 formula).~~ **Done** (`stats.rs`).
4. ~~Core state (`Battle`, sides, field) + `>start`/team-lead construction.~~ **Done** (`state.rs`; switch-in events deferred).
5. ~~Damage calc with Gen 3's two-phase modifier ordering (validate vs the omniscient oracle; a self-contained pure function).~~ **Done** (`damage.rs`; `harness/gen_damage_golden.js` + `tests/damage_test.rs`, 48 EXACT max-roll scenarios (31 core + the 17 `gen3_item_mechanics_v1` item probes)).
6. The event-dispatch engine (`runEvent`/`singleEvent` ordering) — the crux.
   - 6a. ~~The dispatch core (`singleEvent` + the `speed_sort` order/priority/speed
     sort with the Fisher-Yates speed-tie shuffle) + the deferred `>start`
     switch-in events (Intimidate/weather abilities).~~ **Done** (`event.rs`;
     `harness/gen_switchin_golden.js` + `tests/switchin_test.rs`, 5 scenarios +
     `speed_sort` draw-count unit tests). NOTE the switch-in dispatch is currently
     draw-free — the bracketing gender/Quick-Claw/queue-tie draws belong to 6b/6c.
   - 6b. The full `runEvent` handler gather (`findEventHandlers` across target/
     allies/foes/source/side/field/battle) + the relayVar return protocol +
     `this.modify` — only the switch-in ability `Start` is wired so far.
   - 6c. **Single-turn move execution** (`turn.rs`; `harness/gen_turn_golden.js` +
     `tests/turn_test.rs`, 780 EXACT post-turn-seed-parity assertions + speed-tie
     first-mover). **Done:** both-damaging action ordering (the action-order
     speed-tie shuffle DRAW on a production path), accuracy/crit/damage rolls in the
     exact order, immune short-circuit (accuracy-only), faint-at-0 + faint-skips-rest,
     and the `endTurn` Quick Claw roll (drawn iff no faint).
   - 6d. **Multi-turn loop + residuals + tie-draw closure — DONE** (`turn.rs`;
     `harness/gen_battle_golden.js` + `tests/battle_test.rs`, a per-seed CROSS-TURN
     STATE+SEED differential, ~2034 EXACT post-turn-seed assertions). **Done:** the
     full per-turn cycle — the per-action `eachEvent('BeforeTurn'/'Update'/'Weather')`
     speed-tie shuffles (so a TIE turn is now FULL prng-state-faithful), the
     end-of-turn RESIDUALS (weather chip / Leftovers / burn /8 / poison /8 / Toxic
     ramp, in gen-3 residualOrder, draw-free except the handler-sort + nested-Weather
     tie-shuffles), the deferred-faint protocol (`apply_damage` zeroes HP /
     `process_faints` = `faintMessages` after the in-tryMoveHit shuffle), and the
     `run_battle` multi-turn driver (stops at the first faint).
   - 6e. **SWITCHING + post-faint replacement + win/loss → FULL battle — DONE**
     (`turn.rs`; `harness/gen_fullbattle_golden.js` + `tests/fullbattle_test.rs`, a
     per-seed PER-DECISION STATE+SEED+winner differential to game-end, ~2053 EXACT
     per-decision seed assertions over 8 scenarios × 50 seeds). **Done:**
     `run_full_battle` (a `Choice::Move`/`Choice::Switch` script to WIN/LOSS) — voluntary
     switches (order 103 before move 200; the two-switch action-order tie-shuffle), the
     `switchIn` position swap (stable `MonState::uid` keying), the gen-3 draw-FREE
     switch-in ability `Start`, post-faint replacement (single + DOUBLE, the
     `insertChoice` order-101 splice + the no-op fainted-mon move's tail-skip), the
     pause/resume of the saved turn tail, Explosion self-KO, and win/loss (`pokemon_left
     == 0` loses; both → a gen-3 tie; no Quick Claw on the deciding faint).
   - 6f. **SECONDARY effects + onBeforeMove STATUS draws — DONE** (`turn.rs` +
     `state.rs`; `harness/gen_secondary_golden.js` + `tests/secondary_test.rs`, a
     per-seed PER-DECISION STATE(+STATUS+BOOSTS+CONFUSION)+SEED+winner differential to
     game-end, ~4328 EXACT per-decision seed assertions + ~7457 status + ~7457
     boost-stage + ~7457 confusion-counter assertions over 12 scenarios × 80 seeds,
     with status/boosts/confusion inflicted IN-ENGINE by real secondary moves).
     **Done:** the per-move SECONDARY `random(100)` after a landed hit (par/frz/flinch/
     psn; Serene Grace ×2 / Shield Dust ×0; the onTrySetStatus already-statused +
     gen-3 type-immunity gates — **no Electric→para immunity in gen3**); the
     onBeforeMove status draws (the NEW LEADING draw before accuracy: sleep DRAW-FREE
     counter/wake, freeze `randomChance(1,5)` thaw, flinch DRAW-FREE, confusion
     `randomChance(1,2)` + a typeless-40-BP self-hit `random(16)`, paralysis
     `randomChance(1,4)` full-para — priority-DESC with break-on-first-abort); the
     `confusion`/`flinch` volatiles on `MonState`; the fire-move thaw of a frozen
     defender; the **gen-3 paralysis speed ×0.25** fix (`modify(spe,1,4)`, was
     wrongly ×0.5); the **CONFUSION secondary's `random(2,6)` duration draw** (gated by
     already-confused / Own Tempo — the draw-COUNT fix); the **structured stat-drop /
     self-boost apply** (the additive `secondaryBoosts` dex field → DRAW-FREE boost
     with the Clear Body / Hyper Cutter / Keen Eye immunity gates); and the **Tri Attack
     `random(100)`+`sample(3)` special-case** + the **fail-loud >1-col guard**.
   - 6g. **STANDALONE STATUS MOVES — DONE** (`turn.rs` + `state.rs`;
     `harness/gen_status_move_golden.js` + `tests/status_move_test.rs`, a per-seed
     PER-DECISION STATE(+STATUS+sleep/Toxic counter)+SEED+winner differential to
     game-end over 10 scenarios × 80 seeds in **gen3ou**). **Done:** `run_status_move`
     (the foe-targeting major-status moves par/psn/tox/brn/slp — accuracy-only draw +
     `try_set_status`); the gen-3 MOVE-TYPE immunity (Thunder Wave→Ground, Glare→Ghost —
     the two `ignoreImmunity:false` moves; accuracy still drawn → `-immune`); the sleep
     `random(2,6)` onStart duration + Early-Bird double-decrement wake; Toxic at stage 0
     (the residual ramps it — `Status::Toxic` now mirrors the sim's `statusState.stage`);
     the gen3ou **Sleep Clause Mod** + status-immunity ABILITIES; the gen3ou-only
     `runEvent('SetStatus')` 2-clause handler-sort shuffle (`set_status_event_shuffle`,
     gated by `BattleState::sleep_clause`); and the fail-loud guards (unmodeled status
     move; an `onSetStatus` ability under a clause format).
   - 6h. **SELF-TARGETING SETUP / STAT-BOOST MOVES — DONE** (`turn.rs` self-boost branch
     in `run_status_move` + the data-driven `selfBoosts` dex field;
     `harness/gen_setup_move_golden.js` + `tests/setup_move_test.rs`, a per-seed
     PER-DECISION STATE+BOOST-STAGE+SEED+first-mover differential to game-end over 6
     scenarios × 80 seeds in **gen3customgame**: 2667 seed + 4736 boost-array + 2549
     first-mover assertions). **Done:** the 17 pure self-boost moves (Calm Mind / Dragon
     Dance / Swords Dance / Agility / Bulk Up / Amnesia / Barrier / Acid Armor / Iron
     Defense / Cosmic Power / Tail Glow / Meditate / Sharpen / Howl / Harden / Withdraw /
     Growth) — never-miss → no accuracy draw, draw-free `boost()` apply (±6 clamp, own
     Clear Body never blocks self), `landed` FALSE; the +Speed CACHED-SPEED interaction
     (Dragon Dance / Agility flip the first-mover on the FOLLOWING turn, bit-exact); the
     e2e expansion (`MODELED_SETUP_MOVES` derived from `selfBoosts`, 717 setup-move
     decisions on real teams) + the WATER/VOLT ABSORB accuracy-gating fix it surfaced; and
     the fail-loud exclusion of Defense Curl / Minimize / Double Team / Belly Drum / Curse.
   - 6i. **SELF-HEAL / RECOVERY MOVES — DONE** (`turn.rs`'s recovery branch in
     `run_status_move` + `run_rest` + the `apply_heal` helper;
     `harness/gen_recovery_move_golden.js` + `tests/recovery_move_test.rs`, a per-seed
     PER-DECISION STATE+HP+STATUS+SEED+winner differential to game-end over 8 scenarios × 80
     seeds in **gen3customgame**: ~4468 decision rows). **Done:** Recover / Soft-Boiled /
     Slack Off / Milk Drink (`floor(maxhp/2)`); Moonlight / Synthesis / Morning Sun (the
     gen4-inherited PLAIN-integer weather heal — none `floor(maxhp/2)` / sun `floor(maxhp*2/3)`
     / sand+rain+hail `floor(maxhp/4)`, NOT the 4096-`modify`); Rest (full heal + a prior-
     status cure + a FIXED `Sleep(3)` whose `slp.onStart` STILL DRAWS-then-DISCARDS one
     `random(2,6)` — the draw-COUNT crux, verified vs the sim's PRNG probe — + the gen3ou
     SetStatus shuffle ordered shuffle→`random(2,6)`); `splash` as a draw-free no-op; the
     full-HP / heal-0 FAIL path; the e2e expansion (`MODELED_RECOVERY_MOVES`). **Still TODO:**
     Wish (a delayed slot-keyed heal), Heal Bell / Aromatherapy / Refresh (status cure),
     Pain Split / Leech Seed / drain / Ingrain / Aqua Ring, weather-SETTING moves (Sunny Day /
     Rain Dance), phaze/hazard/Substitute/field status moves, entry hazards (Spikes), Pursuit,
     Baton Pass, non-Leftovers items, Thick Club, the top-level `move.self.boosts` `selfDrops`
     draw (Overheat/Superpower/Psycho Boost), and the fixed-damage / multi-hit / Wonder Guard
     cases the single-hit calc defers.
   - 6j. **PROTECT / DETECT — DONE** (`turn.rs`'s `run_protect` + the foe-move BLOCK in
     `run_move` + the `protect`/`stall`/`flinch` residual duration handlers; new `MonState`
     fields `protected`/`protect_counter`/`stall_duration`; `harness/gen_protect_move_golden.js`
     + `harness/probe_protect_rng.js` + `tests/protect_move_test.rs`, a per-seed PER-DECISION
     STATE+HP+STATUS+**STALL-COUNTER**+SEED+winner differential to game-end over 6 scenarios × 80
     seeds in **gen3customgame**: 480 runs, 2772 seed + 5544 HP + 4984 stall-counter assertions).
     **Done:** the gen-3 consecutive-use STALL draw (FIRST protect NO draw / a consecutive
     `randomChance(1, counter)` at the floored 2/4/8 denominator via the gen4-inherited `stall`
     `counterMax: 8`; SUCCESS `onStart 2`/`onRestart *2`; a FAILED roll does NOT delete the
     volatile — the gen3 resolved gen5-base `onStallMove` persists the counter, so consecutive
     fails re-roll at the SAME denominator); the **`willAct()` gate** (a Protect vs a foe SWITCH
     fails draw-free, no volatile); the move-BLOCK (a foe move TARGETING the protected mon draws
     its accuracy roll then is blocked BEFORE crit/damage/secondary/status, gen-3-`tryMoveHit`-
     ordered, before the immunity report; a self-target move is never blocked); the counter reset
     after one non-protect/switch turn (the `duration: 2` expiry via `stall_duration` at the
     residual; switch-out clears all three); the `protect`/`stall`/`flinch` RESIDUAL duration
     handlers (order NO_ORDER/sub 2 — they participate in the residual tie-shuffle; flinch ties
     with a surviving stall now; confusion has NO duration); the e2e expansion
     (`MODELED_PROTECT_MOVES`, 2811 protect decisions on real teams — incl. consecutive chains);
     and the fail-loud exclusion of Endure + the gen4+ Quick/Wide Guard / King's Shield. **Still
     TODO:** Endure (survive-at-1-HP, a different `onDamage`), Feint (gen4+), + everything 6i defers.
   - 6k. **SPIKES (the entry hazard + the first SIDE CONDITION) — DONE** (`turn.rs`'s
     `run_status_move` spikes arm + `apply_entry_hazards` in `run_switch`; new `SideState::spikes:
     u8`; `harness/gen_spikes_golden.js` + `harness/probe_spikes_rng.js` + `tests/spikes_test.rs`, a
     per-seed PER-DECISION STATE+HP+**SPIKES-LAYERS**+SEED+winner differential to game-end over 5
     scenarios × 80 seeds in **gen3customgame**: 400 runs, ~3475 seed + ~6950 HP + ~6950 spikes-layer
     assertions). **Done:** the `SideState::spikes` layer count (0..=3, a per-side persistent SIDE
     condition that survives switches); the Spikes MOVE (`sideCondition:'spikes'`, `target:'foeSide'`,
     never-miss → no accuracy draw, increments the FOE side's layer by 1 capped at 3, a 4th FAILS —
     all DRAW-FREE, `landed` FALSE); the GROUNDED switch-in DAMAGE on the gen-3 `runSwitch`'s
     `runEvent('EntryHazard')` (BEFORE the ability Start; `max(floor([_,3,4,6][layers]·maxhp/24),1)` =
     maxhp/8 ÷6 ÷4, DRAW-FREE; Flying/Levitate take ZERO); the spikes-KO-on-entry → a forced
     replacement (which ALSO takes spikes), wired through the existing faint/replacement machinery, no
     Quick Claw; the e2e expansion (`MODELED_HAZARD_MOVES`, real Skarmory/Forretress/Cloyster spikers
     on the filtered gate, the per-side spikes layers asserted). **Still TODO:** Toxic Spikes / Stealth
     Rock (NOT gen3), Rapid Spin (the hazard-CLEAR move) + everything 6j defers.
     (The PHAZING / LEECH SEED / SUBSTITUTE / EXPLOSION move layers that followed 6k landed as their
     own dedicated sections above rather than numbered 6-series entries.)
   - 6l. **FIXED-DAMAGE / FIXED-FORMULA MOVES — DONE** (`turn.rs`'s `run_fixed_damage_move` + the
     id-gated `fixed_damage_amount` / `is_fixed_damage_move`, routed in `run_move` BEFORE the
     `category == Status` branch; `harness/gen_fixeddamage_golden.js` + `harness/probe_fixeddamage_rng.js`
     + `tests/fixeddamage_test.rs`, a per-seed PER-DECISION STATE+HP+STATUS+SEED+winner differential to
     game-end over 9 scenarios × 80 seeds in **gen3customgame**: 720 runs, 4144 seed + 8288 HP
     assertions, 2469 fixed-damage-hit decisions, 720 wins). **Done:** the `damage:` / `damageCallback`
     moves that BYPASS `getDamage` (NO crit / damage roll) — **Seismic Toss / Night Shade** (`damage:
     'level'` → the USER's level), **Sonic Boom** (20), **Dragon Rage** (40), **Super Fang**
     (`max(floor(target.hp/2),1)`); their draw model (accuracy-only — acc-100-but-NOT-never-miss STILL
     draws for ST/NS/DR, acc-90 CAN miss for Sonic Boom/Super Fang; NO crit / damage roll / secondary),
     the accuracy-drawn-THEN-`-immune` type-immunity short-circuit (Fighting→Ghost, Ghost→Normal,
     Normal→Ghost), the sub-absorb of the fixed NUMBER (Super Fang still halves the MON's hp behind a
     sub), the deferred-faint KO-to-win; the e2e allow-list (`MODELED_FIXED_DAMAGE_MOVES` — but 0
     filter-clean teams carry one, the leech-seed situation, so proven by the dedicated golden + the
     FD1–FD4 pins); the protocol un-block (the two Seismic-Toss scenarios' fixed-damage lines now replay
     byte-exact, though they stay deferred for a forced-replacement resume boundary in the switching
     layer). **Still TODO (fail-loud):** Psywave (variable RNG), the OHKO moves (Fissure/Horn Drill/
     Guillotine), Counter / Mirror Coat / Bide (reactive), Endeavor.
   - 6m. **TAUNT + DISABLE (the move-SELECTION-restriction layer) — DONE** (`turn.rs`'s
     taunt/disable arms in `run_status_move` + the `on_before_move` cants + `move_usable`/
     `must_struggle` in `state.rs` + `disable_move_event_shuffle`;
     `harness/gen_taunt_disable_golden.js` + `tests/taunt_disable_test.rs`, a per-seed
     PER-DECISION STATE(+STATUS+TAUNT+DISABLED-SLOT)+SEED+winner differential to game-end
     over 9 scenarios × 80 seeds in **gen3customgame**: 720 runs, 4723 seed + 8595 taunt +
     8595 disabled-slot assertions, both disable-duration branches pinned at their free-up
     boundaries). **Done:** Taunt (acc-100 draw; FIXED duration 2, no draw — the gen4 mod
     SHADOWS the base onStart's duration++; Status-slot selection restriction + the
     execution-time cant at priority 0; residual tick at order 10/subOrder 15) and Disable
     (acc-55 draw; ONE `random(2,6)` + the willMove-branch `stored = rolled` /
     `rolled + 1`; the lastMove slot restriction + the execution-time cant at priority 7;
     draw-free no-lastMove onTryHit fail; residual tick at NO_ORDER/subOrder 2), the
     endTurn `runEvent('DisableMove')` tie-shuffle for a multi-restriction mon, forced
     Struggle composition, and the e2e admission (`MODELED_RESTRICTION_MOVES`). See
     "Taunt + Disable" above. **Still TODO (fail-loud):** Torment, Imprison, Encore.
   - 6n. **TRAPPING (Arena Trap + Magnet Pull — the SWITCH-legality gate) — DONE**
     (`turn.rs::is_trapped` + the `move_decision_is_legal` Switch reject + the
     `trap_event_shuffles` endTurn draws + `DecisionRecord.trapped`;
     `harness/gen_trapping_golden.js` + `tests/trapping_test.rs`, a per-seed PER-DECISION
     STATE(+per-side TRAPPED)+SEED+winner differential to game-end over 8 scenarios × 80
     seeds in **gen3customgame**: 640 runs, 5771 seed + 8346 trapped assertions, 508
     mutual-trap rows, 160 phaze-drag rows, 640 wins). **Done:** Arena Trap (grounded
     foes; Flying/Levitate escape; a grounded GHOST **IS** trapped in Showdown-gen3 — no
     `trapped` type-immunity in the gen3 dex), Magnet Pull (Steel foes, groundedness
     irrelevant; the gen3 `onAny*` override → the speed-tied MAGNETON MIRROR draws 4 per
     endTurn, the AT-vs-MP cross 2, the Dugtrio mirror 0 — `trap_event_shuffles`
     interleaved per mon with the DisableMove event, before the Quick Claw), the
     draw-free `chooseSwitch` rejection (a scripted trapped `Switch` is SKIPPED — the
     reject-and-re-request pattern), phaze-drags-bypass-trapping, forced replacements
     un-gated, and the e2e admission (`arenatrap`/`magnetpull` in `MODELED_ABILITIES`;
     the generator's voluntary-switch picker respects `pokemon.trapped`). See "Trapping"
     above. **Still TODO (fail-loud / excluded):** Mean Look / Spider Web / Block (the
     trapping MOVES), Shadow Tag.
7. Protocol emission — byte-identical `|...|` stream; gated by a **level-2 differential
   harness** (`harness/gen_protocol_capture.js` → `tests/vectors/protocol_capture_golden.txt`)
   replayed through the emitting engine and diffed byte-for-byte.
   - 7a. **Phase 1 (framing + core move/damage/switch/faint/turn/win) — DONE**
     (`protocol.rs` `ProtocolBuilder` + the emit hooks in `turn.rs`;
     `tests/protocol_test.rs`, **30 battles / 1512 Phase-1 lines byte-equal** across the 4
     design-core scenarios + `sand_intimidate_effectiveness`, 36 battles deferred). **Done:**
     the append-only PRNG-free `ProtocolBuilder` (centralized `MonRef`/`HpStatus`/`Cause`
     formatting) + `run_full_battle_logged`; the framing / `turn` / `upkeep` / separator /
     `move`(+`[miss]`/`[still]`) / `switch` / `drag` / `-damage`(all HP variants + residual
     `[from]`) / `-heal`(Leftovers) / `faint`(faintQueue order) / `-crit` / `-supereffective` /
     `-resisted` / `-immune` / `-miss` / `win` / `tie` lines. **OBSERVATION-ONLY** — the full
     seed suite stays green with unchanged assertions (the only engine change is the
     state-equivalent weather-chip speed-order reorder). See "Protocol emission" above.
   - 7b. **Phase 2 (weather/boost/ability + status/cant/fail + volatiles/side-conditions) — DONE**
     (`protocol.rs` new constructors + the emit hooks in `turn.rs`; `tests/protocol_test.rs`,
     **51 battles / 5630 lines byte-equal** across 9 scenarios, up from 30 / 1512). **Done:** the
     switch-in `-ability`/`-unboost`/`-weather` framing (reconstructed faster-lead-first from the
     post-switch-in state); the STATUS-move `|move|` announce (self-target vs foe-target + the
     Spikes-at-cap `[still]` did-nothing form); `-status`(+`[from] move: Rest`) / `-curestatus`
     (+`[msg]`) / `cant`(par/slp/frz/flinch); `-boost`/`-unboost` (by the clamped-delta sign);
     `-weather`(SET `[from] ability:`+`[of]` + `[upkeep]` tick, speed-ordered chip); `-fail`
     (+`move: Substitute`+`[weak]`); `-sidestart`(Spikes); the Substitute `-start`/`-end`/
     `-activate [damage]`/cost-`-damage`; the Protect `-singleturn`/`-activate` block. This
     UN-DEFERS `substitute_absorb` / `protect_block` / `spikes_and_phaze` / `recover_and_rest`.
     **OBSERVATION-ONLY** — the full seed suite stays green with IDENTICAL counts (e2e 14228 /
     battle 2034 / fullbattle 2053); the only changes are two state-/seed-invariant emission
     REORDERS (the `|turn|N+1` marker → next-turn top; the weather chip → shuffle permutation).
     **Then deferred (now un-deferred in 7d):** `status_para_and_boost_drop` +
     `secondary_status_flinch` (a forced-replacement REQUEST-BOUNDARY resume phantom) + `debug`
     (poke-env-ignored, deliberate non-emit). 3 `recover_and_rest` battles were per-battle-skipped for
     **Struggle** — now un-skipped by 7e. See "Protocol emission" above.
   - 7c. **Phase 3 (FIXED-DAMAGE modeled) — Seismic Toss / Night Shade / Sonic Boom / Dragon Rage /
     Super Fang.** Modeling Seismic Toss (§ Fixed-damage moves) UN-BLOCKED the fixed-damage half of
     `status_para_and_boost_drop` / `secondary_status_flinch` (the ST `|move|`/`|-damage|` lines now
     match); their remaining blocker was the forced-replacement request-boundary resume (7d). PP
     tracking (7e) un-skips the Struggle battles.
   - 7d. **The forced-replacement resume + the already-statused `-fail` (un-defers the LAST 2
     scenarios) — DONE** (`gen3_forced_replacement_resume_v1`). Two fixes closed
     `status_para_and_boost_drop` / `secondary_status_flinch`: (1) `run_full_battle`'s
     **reject-and-re-request gate** (`move_decision_is_legal`) — the "phantom" was an INVALID scripted
     move slot after a replacement swapped in a mon with FEWER moves, which the sim rejects drawing 0
     (the port used to run a full turn for it); now SKIPPED draw-free. (2) the standalone status-move
     **already-statused `|-fail|` emission** (`foe_status_move_fail`: SAME status → `|-fail|<target>|par`,
     DIFFERENT status → `[still]` move form + `|-fail|<user>`, keyed on `move.status` so a secondary
     status emits nothing). Both OBSERVATION-ONLY (fix (1) is zero-draw, fix (2) draws only the accuracy
     roll the port already drew) — the ENTIRE seed suite (e2e 13367 diverged 0 / battle 2034 / fullbattle
     2053 / secondary 4328) stays BYTE-IDENTICAL. **Result: `protocol_test.rs` = 63 battles / 7223 lines
     byte-equal, 0 scenario deferrals** (`DEFERRED_SCENARIOS` empty), 3 Struggle battles per-battle-skipped
     (until 7e). Pinned by `forced_replacement_resume_runs_the_post_replacement_move_decision`.
   - 7e. **PP tracking + Struggle (un-skips the LAST 3 battles) — DONE** (`gen3_pp_tracking_v1`, see
     "PP tracking + Struggle" above). Per-move PP counters + the forced-Struggle substitution + the
     Choice-Band lock + the gen-3 Struggle move (typeless '???' 50 BP, accuracy 100 → draws accuracy,
     `max(floor(dmg/4),1)` recoil via the `damage_of` `[from] Recoil|[of]` line) + the truncation
     turn-marker fix (the `|turn|N` marker moved to the REQUEST so a rejected out-of-PP fresh-turn `move`
     still shows it). The 3 `recover_and_rest` CB-Tyranitar-out-of-Crunch battles REPLAY byte-exact →
     **`protocol_test.rs` = 66 battles / 8721 lines byte-equal, 0 skipped**. OBSERVATION-ONLY (PP + lock +
     substitution ALL draw-free) — the seed suite (e2e 13367 / battle 2034 / fullbattle 2053 / secondary
     4328) stays BYTE-IDENTICAL.
   - 7f. **Protocol PHASE 3 + the write_line drop-in — DONE** (`gen3_protocol_phase3_v1` +
     `gen3_writeline_stream_v1`; see "Phase 3 emits" + "The drop-in endgame — BUILT" above). The
     formerly-deferred lines (Trace / taunt+disable `-end`s + retro-edits / Flash Fire cycle /
     STATUS_IMMUNE `-immune` / Synchronize→Lum / mid-battle switch-in ability lines / Leech Seed /
     Splash / Pay Day) all EMITTED via 8 new capture scenarios → **`protocol_test.rs` = 114 battles
     / 16115 lines byte-equal** (later **132 / 19348** — see 7g); `BattleStream::write_line` per-write
     byte-gated by **`writeline_test.rs` = 44 battles / 2377 writes** vs the real Node `BattleStream`
     (`gen_writeline_capture.js`). OBSERVATION-ONLY: emission draws nothing; the two request-boundary
     fixes (per-side choice acceptance + the switch-to-fainted reject) are zero-draw boundary-MAPPING
     changes no pre-Phase-3 script reaches — the ENTIRE seed suite stays byte-identical and the e2e
     golden md5 (`a23d77ac60d4af168b8a4428f0b465c9`) is unchanged.
   - 7g. **Protocol review FINDINGS F1-F5 — CLOSED** (`gen3_protocol_phase3_review_v1`). Five
     Lens-1-diagnosed EMISSION/BOUNDARY-layer gaps (no engine state/draw/seed impact): **F1** a
     sub-blocked Leech Seed now emits `[still]`+`-fail` (was a bare `|move|`); **F2** a MISSED
     `onTryHit`-class ability immunity (Flash Fire / Water&Volt Absorb, POST-accuracy) now emits
     `[miss]`+`-miss` (was `-immune`; Levitate + type-chart 0× stay pre-accuracy `-immune`, probe
     40/40); **F3** a LANDED Water/Volt Absorb now emits `|-immune|…|[from] ability: <Name>` (was a
     plain `-immune`). Each byte-verified by a NEW capture scenario (`leechseed_into_substitute` /
     `flashfire_tryhit_miss` / `waterabsorb_tryhit_miss`, probes `probe_f1_f2_f3_lines.js` /
     `probe_f2_ff_armed_miss.js` / `probe_levitate_miss.js`) → **`protocol_test.rs` = 132 battles /
     19348 lines byte-equal** (the pre-review 114 scenarios stay a byte-identical golden PREFIX).
     **F4** the `write_line` choice-revision gap (sim `side.choose` = LAST-write-wins; the port's
     replay-from-genesis accumulator = FIRST-accepted-wins) is DOCUMENTED-not-fixed by design (probe
     `probe_f4_choice_revision.js`; an overwrite rule destabilizes the writeline gate, and a revised
     `>pN` is unreachable in production — see the battle.rs write_line scope-limit). **F5** the stale
     event.rs Trace `-ability` "un-emitted" comment corrected (it IS emitted via
     `emit_ability_start_lines`). OBSERVATION-ONLY: the ENTIRE seed suite stays BYTE-IDENTICAL and the
     e2e golden md5 (`a23d77ac60d4af168b8a4428f0b465c9`) is UNCHANGED.
