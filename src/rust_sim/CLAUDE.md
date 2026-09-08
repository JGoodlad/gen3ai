# CLAUDE.md — `src/rust_sim/` (pokesim)

A from-scratch Rust reimplementation of the Pokémon Showdown battle simulator,
scoped first to **Gen 3 OU singles**, whose hard requirement is **bit-for-bit
identical** output to upstream Showdown given the same seed + teams + choices.

**`sim_bridge` is now the DEFAULT training/eval transport** (`--use-bridge` defaults to `rust`,
2026-08-14): every run is serverless unless it passes `--use-bridge node` (the A/B arm + the parity
harness) or `--use-bridge off` (websocket). See the root `CLAUDE.md` → In-process bridge transport.

The engine is **live and bit-for-bit through full battles**: every layer in the
module map below is differentially validated against the real Showdown (PRNG →
dex → team → stats → state → events → damage → turns → full battles with
switching/secondaries/status/setup/recovery/protect/spikes/phazing/leech/
substitute/explosion/fixed-damage/PP+Struggle/taunt+disable/trapping), capped by
the 220-battle real-team e2e capstone (STRICT `filtered_diverged == 0`) and a
byte-identical protocol-emission Phase 1+2+3 (132 battles / 19348 lines), and the
bridge-facing `BattleStream::write_line` streaming surface — per-write byte-gated
against the real Node `BattleStream` (44 battles / 2377 writes, `writeline_test.rs`).

### 🚨 The move census is RECOUNTED, never quoted

**Every ROUND entry below states the census as of THAT round, and each says "current" because it
was.** They are round-scoped history — do not read one as today's number. The **only** current
answer is the tool:

```bash
cd src/rust_sim/harness && SCAN_UNIVERSE=1 node scan_move_coverage.js   # exits non-zero if any MISMODELED
cd src/rust_sim/harness && node scan_move_coverage.js                   # the 722-team pool report
```

**Measured 2026-08-23: 369 gen3-legal moves → 309 MODELED · 60 FAIL-LOUD · 0 MISMODELED**, pool
722/722 fully engine-playable. For scale, the ROUND-40 entry says 281/88 and ROUND 44 says 286/83 —
both were true when written. **The invariant is the load-bearing claim, not the split:** 0
MISMODELED is what makes an unmodeled move a loud construction failure rather than a silent
desync, and it has held under every round separately and combined. Re-run after admitting any move
class, and fix the two always-current docs that carry a copy of this number (root `CLAUDE.md`,
`src/utils/bridge/README.md`) in the same pass — both had gone stale by 28 moves before the
2026-08-23 doc audit.

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
| `state.rs` | **construction done, validated** | In-battle state: `BattleState`/`SideState`/`MonState`/`Field` + `Status`/`Weather` enums. `MonState` carries the major `status`, the `confusion: Option<u8>` counter, the `flinch: bool` volatile (the two volatiles the secondary/onBeforeMove step needs), and the **`cached_speed: u32`** (`pokemon.speed`) the `eachEvent` tie-shuffles + residual handler-sort read — refreshed para/boost-aware at turn-start / residual-start / switch-in, STALE between (the e2e-capstone draw-count fix). `SideState` carries the **`spikes: u8`** layer count (0..=3) — the **first SIDE CONDITION** (`gen3_entry_hazard_spikes_v1`), a per-side persistent state (NOT a mon volatile, so it PERSISTS across switches), reusable by future hazards/phazing; 0 at construction. `MonState` also carries the **`leech_seed: Option<usize>`** volatile (`gen3_leech_seed_v1`) — `Some(seeder_side)` when this mon is LEECH-SEEDED (the seeder side that owns the drain; gen-3 singles always heals the seeder's CURRENT active, so only the side is stored). It is set by the Leech Seed MOVE, drained each end-of-turn RESIDUAL (the seeded mon loses `⌊maxhp/8⌋`, the seeder's active heals it), and CLEARED on switch-out (`execute_switch`) AND on faint (`process_faints` → `clearVolatile`); `None` at construction. `MonState` also carries the **`substitute: Option<u16>`** volatile (`gen3_substitute_v1`) — `Some(hp)` is the SUBSTITUTE decoy's remaining HP (created at `floor(maxhp/4)`); it ABSORBS incoming foe damage (the sub HP drops, breaking at 0 with NO carry to the mon), is created by the Substitute MOVE (paying `floor(maxhp/4)` HP), and CLEARED on switch-out + faint; `None` at construction. `MonState` also carries the **`move_pp: Vec<u16>`** + `move_maxpp` per-move PP counters + the **`choice_locked_move: Option<usize>`** (`gen3_pp_tracking_v1`) — `move_pp[k]` is slot `k`'s current PP, INIT to `MoveData::max_pp()` = `pp*8/5` (the ctor's default 3 PP-ups) / raw `pp` for a `noPPBoosts` move; decremented −1 per USE (2 into a Pressure holder), DRAW-FREE, ONLY when the mon MOVES; PERSISTS across switch-out (gen-3, no reset). `choice_locked_move` is `Some(k)` when a Choice-Band mon has locked to slot `k` (set on its first move, cleared on switch-out/faint) — so `must_struggle()` (all usable slots at 0 PP, respecting the lock) forces Struggle. `MonState::pp_array()` → the fixed `[i16;4]` the per-decision differential asserts. `MonState` also carries the **`taunt: Option<u8>`** + **`disable: Option<(usize, u8)>`** selection-restriction volatiles + the **`last_move: Option<usize>`** slot record (`gen3_taunt_disable_v1`) — `taunt` is the remaining-turn counter (a FIXED 2 at apply), `disable` is `(disabled_slot, remaining_turns)`, `last_move` is the slot this mon last USED (set after BeforeMove passes; a Struggle stores `None`); `move_usable(k, dex)` folds the Choice lock + Disable + Taunt (per-slot derived-Status minus the fixed-damage family) + PP, and `must_struggle(dex)` forces Struggle when nothing is usable; all three clear on switch-out + faint. `MonState` also carries the **`flash_fire: bool`** activation volatile (`gen3_flashfire_boost_v1`) — `true` once this mon's Flash Fire ability has ABSORBED a Fire move (armed at the `acc_hit`-gated Fire-absorb site in `run_move`, DRAW-FREE — a MISSED Fire move does NOT arm it; skips a `frz`-status holder; cleared on switch-out + faint), so thereafter its OWN Fire moves get **×1.5** (the volatile's `onModifyDamagePhase1 chainModify(1.5)` — a DAMAGE-PHASE fold folded in `damage.rs::modify_damage`, NOT a stat mod; category-agnostic, NOT crit-bypassed; accumulated with any screen into ONE Phase1 chain modifier); `false` at construction. `MonState` also carries the **`curse: Option<usize>`** volatile (`gen3_move_coverage_batch3_v1`) — `Some(source_side)` when this mon is cursed by a GHOST Curse (the source side for the `[of]` clause), chipped `floor(maxhp/4)/turn` at the order-10 subOrder-8 residual, cleared on switch-out + faint (like `leech_seed`); `None` at construction. `SideState` also carries the **`wish_pending: Option<(u8, String)>`** slot condition (`gen3_move_coverage_batch3_v1`) — `(duration, wisher_name)` for a pending Wish (the slot-keyed order-7 delayed heal `floor(maxhp/2)` at N+1; survives a switch — a side/slot condition, NOT a mon volatile) — and the **`baton_pass_pending: bool`** marker — set when a Baton Pass resolves so `execute_switch` runs `copyVolatileFrom` (snapshot boosts + the copyable `noCopy==false` fields the port models — sub/leech/confusion/curse/perish/trapped_by/charge + the `stall` counter [`protect_counter`/`stall_duration`] + `pursuit`, `gen3_batonpass_stall_pursuit_copy_v1` — → the entrant, `[from] Baton Pass`). `BattleState::start` constructs from `>start`+teams (unpack → compute_stats → leads), runs NO events; `start_with_switchins` adds the `event.rs` switch-in sequence (post-event boosts + weather). Validated vs the sim's construction-time AND post-switch-in state. |
| `event.rs` | **switch-in done, validated** | The generic event-dispatch core: `single_event_ability_start` (`singleEvent` — no sort/RNG/modify) + the reusable `speed_sort` (`order→priority→speed→subOrder→effectOrder` selection sort with the **Fisher-Yates speed-tie shuffle** drawing from the `Prng` — the RNG-consumption crux) over `EventHandler<H>`. Wired for the `>start` switch-in abilities (Intimidate foe-Atk −1 clamp, **gated by the foe's `onTryBoost` immunity — Clear Body / White Smoke / Hyper Cutter — so Intimidate-into-Metagross is a no-op**, an e2e-capstone fix, **AND by the foe's SUBSTITUTE — the gen3 mod skips a subbed foe** (a mid-battle Intimidate switch-in must not drop a subbed foe's Atk; seed-neutral; the `gen3_trapping_v1` e2e regen surfaced it, pin `intimidate_into_a_substitute_is_a_noop`); Sand Stream/Drizzle/Drought permanent weather); `run_start_switchins` fires both leads in raw-Speed order (slower-last so weather overwrites). Validated by `tests/switchin_test.rs` (5 differential scenarios) + `speed_sort` unit tests. Move/turn-loop/residual dispatch NOT built. |
| `damage.rs` | **done, validated** | Gen-3 single-hit damage calc: `calc_damage(&DamageContext, &Dex) → DamageResult{base, rolls:[u16;16]}`. Self-contained (EXPLICIT inputs — stats/types/boosts/status/move/field/`crit` — no `BattleState`). Bit-faithful port of the shared `getDamage` base formula + gen3's OWN two-phase `modifyDamage` (burn-first / randomize-2nd-to-last); integer 4096-chain `modify`, crit ×2 + ignore-boosts/screens, STAB, type chart, weather, Choice Band / type-item / Sea Incense (stat modifiers) + the **ability DMG_MOD folds** (Huge/Pure Power ×2 Atk, Guts ×1.5 Atk statused, Marvel Scale ×1.5 Def statused, the pinch family ×1.5 BP — all data-driven from `AbilityData.dmg_mod`), **Thick Fat (a gen3 `onSourceBasePower` ×0.5, NOT a stat mod)**, the Guts burn-halve suppression, Explosion def-halve, immunity → 0. Validated vs the omniscient oracle with the MAX roll forced ⇒ EXACT (not banded). |
| `turn.rs` | **multi-turn + SWITCHING + post-faint + win/loss + secondaries + status moves + SETUP moves + RECOVERY moves + PROTECT + SPIKES + PHAZING + LEECH SEED + SUBSTITUTE + FIXED-DAMAGE + PP-TRACKING + STRUGGLE + TAUNT + DISABLE + TRAPPING + CONTACT_PROC/BLOCK/SYNCHRONIZE abilities + move-coverage BATCHES 1-5 (through Focus Punch/Pursuit, Beat Up/Thunder/Water Spout, Hyper Beam/Solar Beam/Doom Desire/Future Sight, and Counter/Mirror Coat/Endeavor + Return/Frustration/Flail/Reversal/Low Kick + Sleep Talk) done, validated** | **MODULE LAYOUT** (`gen3_turn_submodule_split_v1`, pure file-org, ZERO behavior change): `turn.rs` (~2.8k) is now the module ROOT (top-of-file consts + shared structs/enums + the `#[cfg(test)]` suite) declaring `mod {driver, moves, status_moves, secondaries, residuals, items, status, switch, speed, helpers}` under `src/turn/`, each an `impl crate::state::BattleState` (+ `impl FullBattleDriver` in `driver`) block of the moved methods — Rust's multi-file impls, with only cross-module methods/free-fns bumped to `pub(crate)`, no logic touched. `BattleState::run_turn(p1_slot, p2_slot, &dex) → TurnResult` runs ONE FULL turn cycle (both sides damaging); `BattleState::run_battle(scripted, &dex) → Vec<TurnRecord>` loops it, stopping at the first faint. The cycle, in Showdown's EXACT draw order: the **action-order** speed-tie shuffle (`event::speed_sort` on the queue), the per-action **`eachEvent('BeforeTurn'/'Update'/'Weather')`** speed-tie shuffles (the draws the single-turn step deferred — incl. the one INSIDE gen3 `tryMoveHit` that fires only on a LANDED move), each move accuracy `random(100) < effAcc` (skip iff `never_miss`; `gen3_accuracy_pipeline_v1` — `effAcc = move.accuracy × the acc/eva stage table × the accMod item/ability handlers`, via `effective_accuracy`/`roll_accuracy`; the empty path is byte-identical to `randomChance(acc,100)`) → crit `randomChance(1,critMult[critRatio])` → damage `random(16)`, the **end-of-turn RESIDUALS** in gen-3 residualOrder (weather chip Sandstorm/Hail `max(1,⌊maxhp/16⌋)` to non-Rock/Ground/Steel; Leftovers `+⌊maxhp/16⌋`; the major-status DoT — burn `max(1,⌊maxhp/8⌋)`, poison `max(1,⌊maxhp/8⌋)`, Toxic `max(1,⌊maxhp/16⌋)·stage` with the per-mon stage ramp on `Status::Toxic`), all DRAW-FREE except the handler-sort + nested-Weather tie-shuffles, then the **Quick Claw** `randomChance(1,5)`. The **deferred-faint protocol** (`apply_damage` zeroes HP; `process_faints` sets `fainted` AFTER the in-`tryMoveHit` shuffle, mirroring `faintMessages`) is the faint-turn draw-COUNT crux: a KO turn fires the in-tryMoveHit shuffle but NOT the trailing Update / second move / residual / Quick Claw. Residual order/values are the gen4-mod overrides gen3 INHERITS (burn **/8**, Leftovers order 10 sub 4, status DoT order 10 sub 6, sand field-residual order 8 — NOT the base-data values). `run_residuals` runs **`faintMessages` PER HANDLER** (sets `fainted` between handlers + `if (ended) return`s, mirroring `fieldEvent('Residual')`'s `while`-loop): a holder fainted by an earlier handler skips its later ones, a GAME-ENDING residual KO aborts the rest, but a non-ending faint does NOT abort (the other active still ticks) — and since `order→priority→SPEED→subOrder` puts SPEED above subOrder, a fast burned mon's DoT self-KO can end the battle before a slower foe's Leftovers (`run_turn`'s post-residual faint gate reads the STATE `any_active_fainted`, not the now-always-false newly-fainted return). The `eachEvent` tie-shuffles + the residual handler-sort read the **CACHED `pokemon.speed`** (`MonState::cached_speed`), refreshed para/boost-aware at turn-start, residual-start, and switch-in (`update_speed()` / `execute_switch`) and STALE between — so a mon paralyzed mid-turn ties on its FULL speed until the residual, while one that switches in paralyzed ties on its PARA speed (the e2e-capstone bit-for-bit fix). **`BattleState::run_full_battle(&[ScriptDecision], &dex) → BattleOutcome` plays a FULL battle to WIN/LOSS** with `Choice::Move`/`Choice::Switch`: voluntary switches (order 103 < move 200, so they resolve FIRST; the two-switch action-order tie-shuffle), the gen-3 draw-FREE switch-in (the entrant's ability `Start` via `single_event_ability_start`; the gen-4 `runSwitch` override has NO `speedSort(allActive)` — do NOT add a SwitchIn tie-shuffle), the `switchIn` POSITION SWAP (the entrant → active index, outgoing → its old bench slot; mon actions are keyed by a stable `MonState::uid`, not array slot), POST-FAINT replacement (single + DOUBLE; the double's `insertChoice` order-101 splice draw + the no-op fainted-mon move that `return false`s before its tail → NO trailing Update), the pause/resume of the saved turn tail (`makeRequest('switch')` returns the queue intact; the `peek===instaswitch` early-return SKIPs the first instaswitch's tail), the **reject-and-re-request boundary gate** (`move_decision_is_legal`, `gen3_forced_replacement_resume_v1`: a top-of-turn `move` decision whose `Move(K)` slot exceeds the CURRENT active mon's movepool is SKIPPED draw-free — run no turn, emit nothing, record nothing, re-pull the next decision — mirroring the sim's `side.choose` REJECTING an out-of-range slot after a replacement swapped in a mon with fewer moves; this is the "phantom zero-draw `move` decision" the omniscient capture records from a stale per-turn plan, and it un-deferred the last 2 protocol scenarios — pinned by `forced_replacement_resume_runs_the_post_replacement_move_decision`, ground truth `harness/probe_forced_replacement_resume_regression_rng.js`; VERIFIED zero-draw so the e2e/seed suites are byte-identical), and win/loss (`pokemon_left == 0` loses, foe wins; both 0 → a gen-3 TIE `win(None)`; the deciding faint draws NO Quick Claw). **EXPLOSION / SELF-DESTRUCT self-KO** (`useMoveInner` battle-actions.ts:501-503, `gen != 4 && selfdestruct == 'always'`: `this.battle.faint(pokemon)` BEFORE `trySpreadMoveHit`) is modeled bit-for-bit: the self-KO zeroes the user's HP + is DRAW-FREE + UNCONDITIONAL + PRECEDES the hit — sitting AFTER `on_before_move` (a fully-para/asleep/flinched user never reaches `useMoveInner`) but BEFORE the accuracy/protect-block/immunity/miss checks, so the USER FAINTS THROUGH a Protect (blocked, no foe damage), a Ghost (Normal-immune), a Substitute (the damage breaks the sub, no carry), or a miss (gen-3 Explosion accuracy is 100 → no self-accuracy miss, but a hypothetical miss would still faint the user). Explosion draws the SAME count as any damaging move (acc `randomChance(100,100)` + crit + damage; NO secondary); the resulting faint changes `pokemon_left` / cancels the foe's queued move (gen-3 singles) / draws NO trailing Quick Claw on a deciding faint. A mutual Explosion (both last mons) is a true double-faint gen-3 TIE. VALIDATED by `tests/explosion_test.rs` (the differential `harness/gen_explosion_golden.js`: 7 scenarios × 80 seeds, 3688 decision rows, 7376 FAINTED assertions, 544 self-KO rows, 294 sub-break boundaries, 341 wins + 59 ties — plain / into-a-sub / into-a-Protect / into-a-Ghost / mutual-TIE / double-replacement / into-a-real-battle) + 4 DETERMINISTIC `tests/regression_test.rs` pins E1-E4 (user-faints-through-Protect / -immunity / -a-sub-break-with-no-carry / the mutual double-faint TIE; ground-truth seeds from `harness/probe_explosion_regression_rng.js`, the probe `harness/probe_explosion_rng.js` nailed the draw model). The engine also carries a `pending_explosion_self_ko` flag → `DecisionRecord.explosion_self_ko` (a coverage/diagnostic signal only, no effect on any draw/state; the e2e capstone reads it to count explosion decisions). **e2e INCLUDED** (`EXPLOSION_E2E_EXCLUDED = false`, bit-for-bit — **544 explosion-move / self-KO decisions across the 220-battle strict gate**, `filtered_diverged == 0`, `explosion_decisions >= 50` coverage floor): admitting Explosion surfaced TWO STATEFUL desyncs in DIFFERENT layers (NOT the self-KO), both now FIXED. (1) A **double-faint → double-replacement → cascade `runSwitch` cancellation** (e2e_9): when a mutual double faint replaces BOTH sides and the FIRST runSwitch to run FAINTS its own entrant on its side's Spikes (the cascade), gen-3-singles `faintMessages` → `cancelAction(getAllActive())` (battle.ts:2606-2616 — "in gen 3, fainting skips all moves AND SWITCHES") REMOVES the OTHER side's still-pending `runSwitch` (a runSwitch's `action.pokemon` is the entrant, a getAllActive member) — so the foe entrant is NEVER chipped (stays FULL HP). The port's `cancel_active_actions` cancelled `Move`/`Switch` but NOT a pending `RunSwitch` → the stale foe runSwitch survived the cascade + re-chipped its already-settled entrant (e2e_9 dec43: 403 → 353). FIX: `cancel_active_actions` now also drops a `RunSwitch { side }` when `sides[side].active` is not fainted — DRAW-FREE (a queue splice, SEED untouched). VERIFIED vs the sim (`harness/probe_cascade_hazard_order.js` + `probe_double_replacement_spikes_rng.js`): with the FAINTING side's runSwitch FIRST the foe is UNCHIPPED; with the SURVIVING side's first it is chipped ONCE (its runSwitch already ran → nothing to cancel). (2) The **confusion self-hit dropped Choice Band** (e2e_194): gen-4 confusion (gen-3-inherited, conditions.ts:74-83) runs the FULL `getDamage(self,self,40)`, so Choice Band ×1.5 (physical) folds in; `apply_confusion_self_hit` passed no atk stat mods → it used the stored Atk not the CB-boosted Atk → the self-hit under-dealt. FIX: it now resolves `resolve_atk_stat_mods(item, None, Physical)` (typeless '???' → CB only, no type-item / Sea Incense) — DRAW-FREE (the self-hit still draws `random(1,2)` + `random(16)`). Both pinned by revert-verified `regression_test.rs` pins `double_replacement_cascade_does_not_rechip_the_other_sides_entrant` + `confusion_self_hit_applies_choice_band` (ground truth from `harness/probe_double_replacement_cascade_regression_rng.js` + `probe_confusion_choiceband_regression_rng.js`). See EDGE_CASES.md. **SECONDARY effects + onBeforeMove STATUS draws** (this step) bracket each move: (a) `on_before_move` fires the NEW LEADING draw BEFORE accuracy, priority-DESC with break-on-first-abort — sleep (DRAW-FREE counter decrement / wake), freeze `randomChance(1,5)` thaw, flinch (DRAW-FREE volatile), confusion (decrement → `randomChance(1,2)` → a typeless-40-BP self-hit one `random(16)`, NO crit, via `calc_damage`), paralysis `randomChance(1,4)` full-para; an abort draws NOTHING further; (b) `apply_secondaries` fires the NEW TRAILING `random(100)` per surviving secondary AFTER a landed hit (Body Slam par30 / Ice Beam frz10 / Thunderbolt par10 / Rock Slide flinch30 / Sludge Bomb psn30), applied if `roll<chance` via `try_set_status` (the onTrySetStatus gates: already-statused → no-op, gen-3 type immunity — **frz→Ice, brn→Fire, psn/tox→Poison&Steel; gen-3 has NO Electric→para immunity; **SUN → NO freeze** — `gen3_sun_freeze_immunity_v1`, the base `sunnyday` weather's `onImmunity('frz')` blocks a freeze while the field is Sun [Drought/Sunny Day], at `runStatusImmunity` before the SetStatus shuffle, DRAW-FREE; an already-frozen mon PERSISTS under sun — pinned FZ1**, verified vs sim) / the flinch volatile / **the CONFUSION arm** (`add_confusion`: a landed confusion secondary draws ONE EXTRA `random(2,6)` duration inside `addVolatile`'s onStart UNLESS gated — ALREADY-CONFUSED or OWN TEMPO draw the secondary `random(100)` but NOT the `random(2,6)`, the draw-COUNT gate) / **the STRUCTURED stat-boost arm** (`apply_secondary_boost`: the foe stat-DROP / self stat-RAISE the flat `secondaryEffects` `{col:percent}` loses — Crunch −1 SpD, Psychic/Shadow Ball −1 SpD, Meteor Mash +1 Atk SELF, Ancient Power +1 ALL — read from the additive **`secondaryBoosts`** dex field [`{chance,target:foe|self,boosts}`, only-when-present like `critRatio`], DRAW-FREE apply [`boost()` consumes no PRNG] clamped to ±6, with the Clear Body / White Smoke / Hyper Cutter / Keen Eye `onTryBoost` immunity gates); a **fail-loud guard** PANICS on any move with >1 secondary col except **Tri Attack** (`triattack`), which is SPECIAL-CASED to its true draw model — ONE `random(100)` (the 20% gate) then ON LAND ONE `random(3)` `sample(['brn','par','frz'])` → `try_set_status` (NOT three `random(100)`s the 3-col flatten would mis-draw). **Serene Grace ×2** the threshold (NOT the draw) and **Shield Dust** on the DEFENDER FILTERS foe-targeting secondaries out (a draw-COUNT effect — zero `random(100)`) — **but NOT behind a SUBSTITUTE** (`gen3_shielddust_sub_v1`: the filter is a TARGET-gathered ModifySecondaries handler and a sub-absorbed hit's target list is `null`, so the filter never gathers and the secondary / Tri-Attack-gate / King's-Rock `random(100)` STILL DRAWS, held AND breaking sub, while the effect stays sub-suppressed — the A/B fuzzer's #1 sub×secondary SEED cluster, probe `probe_sub_break_secondary_rng.js`, pin `shield_dust_behind_a_substitute_still_draws_the_secondary`); a DAMAGE-immune target short-circuits BEFORE the secondary (no draw), a status-immune-but-damaged target STILL draws it. The fire-move thaw cures the DEFENDER's freeze (draw-free, post-secondary). **gen-3 paralysis speed is ×0.25** (`modify(spe,1,4)` = `floor((spe·1024+2047)/4096)`, gen4-inherited `chainModify(0.25)` — NOT ×0.5; verified vs the sim's `getStat`). FLINCH (`duration:1`) clears at the top of each turn; switch-out clears confusion+flinch. Validated by `tests/turn_test.rs` (the single-turn 780-row gate), `tests/battle_test.rs` (a per-seed CROSS-TURN STATE+SEED differential over 12 scenarios × 40 seeds × several turns, no switching), `tests/fullbattle_test.rs` (a per-seed PER-DECISION STATE+SEED differential to GAME-END: ~2053 per-decision EXACT seed assertions over 8 scenarios × 50 seeds — both-switch distinct/tie, switch-vs-move, post-faint single + double replace, KO-to-win, last-mon double-KO TIE), AND `tests/secondary_test.rs` (a per-seed PER-DECISION STATE+**STATUS**+**BOOSTS**+**CONFUSION**+SEED+winner differential to GAME-END with REAL secondary moves: **~4328 per-decision EXACT seed assertions + ~7457 status-variant + ~7457 boost-stage + ~7457 confusion-counter assertions over 12 scenarios × 80 seeds** — full-para, freeze-thaw, flinch, psn-immune-but-damaged, Ground-damage-immune-no-secondary, **Crunch/Psychic/Shadow Ball −1 SpD foe, Meteor Mash +1 Atk self, Intimidate −1 Atk on entry, Water Pulse confusion + the random(2,6) duration**) + deterministic unit gates for the confusion-secondary draw (`random(100)`+`random(2,6)`), its already-confused / Own-Tempo no-`random(2,6)` gates, the Tri-Attack `random(100)`+`sample(3)` sequence, the >1-col fail-loud panic, the stat-drop/self-boost apply, and the Clear-Body block. **STANDALONE STATUS MOVES now BUILT** (`run_status_move`): the foe-targeting major-status moves — Thunder Wave/Stun Spore/Glare [par], Toxic [tox], Poison Powder/Poison Gas [psn], Will-O-Wisp [brn], Spore/Sleep Powder/Hypnosis/Sing/Lovely Kiss/Grass Whistle [slp] — drawing ONLY accuracy (`randomChance(acc,100)`, skip iff never_miss) then `try_set_status`, with the gen-3 MOVE-TYPE immunity (Thunder Wave→Ground, Glare→Ghost — the two `ignoreImmunity:false` moves; all others ignore type immunity, accuracy still drawn → `-immune`), the sleep `random(2,6)` onStart duration (Early Bird double-decrements the wake counter), Toxic at stage 0 (the residual ramps it), the **gen3ou SLEEP CLAUSE MOD** (a 2nd foe-sleep fails at the SetStatus event, draw-free) + the DATA-DRIVEN status-immunity ABILITIES (`gen3_status_immune_v1` — Limber par / Insomnia+Vital Spirit slp / Immunity psn,tox / Water Veil brn via `onSetStatus`; Magma Armor frz via `onImmunity` BEFORE the event — read from `AbilityData.status_immune`), and the **gen3ou-only `runEvent('SetStatus')` handler-sort shuffle** (the 2 `Standard` clauses tie → ONE `random(0,2)` on EVERY status APPLICATION reaching the SetStatus event — incl. a clause/ability block; gen3customgame has 0 handlers → no shuffle, gated by `BattleState::sleep_clause` from the format). NO crit/damage/secondary; `landed` always FALSE (a status `moveHit` returns `undefined` → the in-tryMoveHit Update is skipped). A fail-loud guard PANICS on any UNMODELED status move (and on a genuinely UNMODELED `onSetStatus` ability under a clause format via `ability_unmodeled_on_set_status` — the STATUS_IMMUNE members are MODELED, `gen3_status_immune_v1`, and sort into their own speed group so the 2-clause tie stays size-2, unchanged draw count). **The ALREADY-STATUSED `-fail` emission** (`gen3_forced_replacement_resume_v1`, `foe_status_move_fail` + `StatusMoveFail`): a foe-status MOVE into an already-statused foe emits a `|-fail|` line — draw-free past the accuracy roll (the fail is emitted at `setStatus` BEFORE `runEvent('SetStatus')`, so no clause shuffle). gen-3 `trySetStatus` re-passes the foe's OWN status to `setStatus` (`setStatus(this.status || status)`), so `status.id === this.status` (pokemon.ts:1699): **SAME status** as the move inflicts (Thunder Wave→par into par) → `|-fail|<target>|<status>` (fail on the TARGET, status token); **DIFFERENT status** (Thunder Wave→par into brn) → the move announce's `[still]` empty-target form + `|-fail|<user>` (fail on the USER, no token). Keyed on the move having a top-level `move.status` field so a SECONDARY status (Body Slam's par into a statused foe) correctly emits NOTHING (verified `harness/probe_status_move_fail_lines.js` + `probe_status_fail_accuracy.js` — the only per-move draw is the accuracy roll). **SELF-TARGETING SETUP / STAT-BOOST MOVES now BUILT** (`run_status_move` self-boost branch BEFORE the fail-loud): the PURE self-boost moves (category Status, bp 0, target self) — Calm Mind (+1 SpA/+1 SpD), Dragon Dance (+1 Atk/+1 Spe), Swords Dance (+2 Atk), Agility (+2 Spe), Bulk Up (+1 Atk/+1 Def), Amnesia (+2 SpD), Barrier/Acid Armor/Iron Defense (+2 Def), Cosmic Power (+1 Def/+1 SpD), Tail Glow (+2 SpA), Meditate/Sharpen/Howl (+1 Atk), Harden/Withdraw (+1 Def), Growth (+1 SpA) — resolved from the data-driven `selfBoosts` dex field (`self_boost_spec`, the GIGO-proof source). Their draw model: (1) ACCURACY — every modeled setup move is NEVER-MISS (`accuracy:true`) so NO accuracy draw (a non-never-miss setup move would draw it; none exist); (2) APPLY `boost()` on the USER, ±6 clamp, **DRAW-FREE** (`boost()` consumes no PRNG — like `apply_secondary_boost`); our OWN Clear Body / White Smoke etc. do NOT block our own self-boost (the `onTryBoost` immunity is for FOE drops); a boost into the +6 cap is a no-op-but-success that STILL draws nothing; (3) `landed` ALWAYS FALSE (the in-tryMoveHit Update is skipped). **The +SPEED cached-speed crux:** a Dragon Dance / Agility raises `boosts[4]` IMMEDIATELY but does NOT touch `cached_speed` — Showdown re-establishes `pokemon.speed` only at the next re-cache site (turn-start `commitChoices`, residual `updateSpeed`, switch-in), so THIS turn's eachEvent tie-shuffles read the PRE-boost cached speed and the NEXT turn's action order picks up the boosted speed at turn-start (the e2e + setup-golden first-mover FLIP proof). The EXCLUDED setup moves stay fail-loud: Defense Curl / Minimize (a `volatileStatus`), Double Team / Minimize (+evasion — the engine's accuracy roll ignores evasion → a silent desync), Belly Drum (HP-cost `onHit`, no declarative `boosts`), Curse (type-conditional). **WATER/VOLT ABSORB heal is now accuracy-gated** (the e2e-capstone fix this step surfaced): the absorb heal is an `onTryHit` that fires only when the move HITS, so a MISSED Water/Electric move (e.g. Hydro Pump's 80% accuracy fails) does NOT heal the Absorb holder — `run_move` only calls `apply_absorb_heal` when `acc_hit` (the draw count is accuracy-only either way, so a wrongly-applied heal desyncs the post-hit HP STATE but not the seed). **SELF-HEAL / RECOVERY MOVES now BUILT** (`run_status_move`'s recovery branch + `run_rest`, BEFORE the fail-loud): the self-targeting HP-recovery moves (category Status, bp 0, target self, isHeal). The flat-half recovers **Recover / Soft-Boiled / Slack Off / Milk Drink** heal `floor(maxhp/2)` (the `move.heal:[1,2]` path); the WEATHER-conditional **Moonlight / Synthesis / Morning Sun** heal — gen4-inherited PLAIN integer (NOT the 4096-`modify`) — `floor(maxhp/2)` (none) / `floor(maxhp*2/3)` (SUN) / `floor(maxhp/4)` (SAND/RAIN/HAIL); **Rest** FULL-heals + self-sleeps + cures the prior status. All NEVER-MISS (no accuracy draw) and the heal itself is DRAW-FREE (`apply_heal` mirrors `apply_leftovers`; a heal at FULL HP / heal-0 FAILS via the `heal`-returns-false path, draw-free); `landed` is FALSE (no in-tryMoveHit Update). **REST's draw subtlety (the verified-against-the-sim crux):** `setStatus('slp')` runs the gen-3 `slp.onStart`, which ALWAYS draws `random(2,6)` for the duration — so Rest **DOES consume one `random(2,6)`** (`run_rest` draws-then-DISCARDS it) — and Rest's `onHit` then OVERWRITES the time to a FIXED `Sleep(3)` (the stored counter is 3, NOT the rolled 1-4). The user wakes via the EXISTING `on_before_move` sleep counter (3→cant→cant→wake+move). A self-Rest sleep is EXEMPT from the Sleep Clause CAP, but its `setStatus('slp')` STILL draws the gen3ou **SetStatus handler-sort shuffle** in a clause format (`run_rest` does the shuffle THEN the `random(2,6)`, gated by `sleep_clause`; gen3customgame draws neither shuffle — only the `random(2,6)`). **`splash`** is modeled as a true DRAW-FREE no-op (the recovery golden's "do-nothing" filler). DEFERRED (fail-loud): **Wish** (a DELAYED slot-keyed end-of-next-turn heal — a pending-heal model), **Heal Bell / Aromatherapy / Refresh** (team/self STATUS cure, not HP), **Pain Split / Leech Seed / drain / Ingrain / Aqua Ring**, plus phaze/hazard/Substitute/field status moves, entry hazards (Spikes), Pursuit, Baton Pass, non-Leftovers items, the **top-level `move.self.boosts` `selfDrops` draw** (Overheat/Superpower/Psycho Boost), Thick Club, protocol emission. Validated by `tests/recovery_move_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS)+SEED+winner differential to GAME-END over 8 scenarios × 80 seeds: ~4468 decision rows, ~3500 seed + HP assertions, the heal floors + Rest sleep/wake + the weather fractions) + deterministic unit gates (`recover_heals_half_maxhp_draw_free`, `recover_at_full_hp_fails_draw_free`, `moonlight_weather_conditional_heal_amounts`, `rest_full_heal_fixed_sleep_three_and_cure_draws_one_random_2_6_customgame`, `rest_at_full_hp_fails_without_sleeping`, `rest_in_gen3ou_draws_the_setstatus_shuffle_then_random_2_6`). **PROTECT / DETECT now BUILT** (`run_protect` + the foe-move BLOCK in `run_move`; new `MonState` fields `protected`/`protect_counter`/`stall_duration`): NEVER-MISS + **priority 3** (resolves BEFORE the foe's attack so the volatile is up). The gen-3 consecutive-use STALL draw — the FIRST protect (counter 0, no `stall` volatile) SHORT-CIRCUITS with **NO draw** (always succeeds); a CONSECUTIVE one draws `onPrepareHit`'s `runEvent('StallMove')` = ONE `randomChance(1, counter)` at the floored denominator **2 → 4 → 8 → 8** (the gen4-inherited `stall` `counterMax: 8`); a SUCCESS (re)adds the volatile — `onStart` to 2, else `onRestart *= 2` capped at 8, refreshing `duration` to 2 — a FAILED roll draws nothing more and (the gen3 resolved **gen5-base** `onStallMove`, unlike gen8+) does **NOT delete** the volatile: the counter + duration PERSIST (so consecutive fails re-roll at the SAME denominator AND a `stall` residual handler still fires). The **`willAct()` gate** (`onPrepareHit`'s `!!this.queue.willAct() && …`): a Protect that resolves with NO foe action still pending — the foe **SWITCHED** (order 103 < the protect's move order 200, so the switch already ran) — FAILS draw-free, no volatile (the `&&`-short-circuit also skips the stall roll). The move-BLOCK: in gen-3 `tryMoveHit` accuracy (line 364) is drawn FIRST, then `if (accPass) runEvent('TryHit')` (line 369) where protect blocks — so a BLOCKED foe move TARGETING the protected mon DRAWS its accuracy roll (skip iff never_miss) then is blocked, drawing NO crit/damage/secondary/status (and BEFORE the immunity report — EQ into a Flying/Levitate protector shows `-activate Protect`, not `-immune`); a self-target move is never blocked. The stall counter resets after one non-protect/switch turn — the volatile's `duration: 2` expiry, modeled at the RESIDUAL via `stall_duration` countdown; switch-out clears all three. **The `protect`/`stall`/`flinch` DURATION-only volatiles register RESIDUAL duration handlers** (`findPokemonEventHandlers(..., 'duration')` gathers every duration-bearing volatile to count it down) at order NO_ORDER/subOrder 2 — they participate in the residual speed-sort (a protecting mon adds 2 tied handlers; a failed-protect-into-a-flinch adds stall+flinch tied → a tie-group shuffle COUNT the model must match; flinch was draw-free before protect but ties with stall now — confusion has NO `duration` so no handler). DEFERRED (fail-loud in `run_protect`): **Endure** (`volatileStatus:'endure'`, a survive-at-1-HP `onDamage`) + the gen4+ Quick/Wide Guard / King's Shield / Spiky Shield (none in gen3). Validated by `tests/protect_move_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+**STALL-COUNTER**)+SEED+winner differential to GAME-END over 6 scenarios × 80 seeds in gen3customgame: 480 runs, 2772 seed + 5544 HP + 4984 stall-counter assertions, 1102 blocks, 272 escalated counter≥4 — single protect block, CONSECUTIVE protects [the denominator BOTH ways], the counter reset, protect vs a STATUS move, Detect, protect-into-a-real-battle) + deterministic unit gates (`first_protect_draws_nothing_and_sets_counter_two`, `consecutive_protect_draws_one_stall_roll_and_escalates` [incl. the no-delete-on-fail persist], `protect_blocks_foe_move_after_its_accuracy_draw`, `protect_does_not_block_the_protectors_own_move`, `endure_panics_fail_loud`). The PRNG-draw crux was nailed by `harness/probe_protect_rng.js` (the instrumented probe). **SPIKES (the gen-3 ENTRY HAZARD — the first SIDE CONDITION) now BUILT** (`run_status_move`'s spikes arm + `apply_entry_hazards` in `run_switch`; new `SideState::spikes: u8`): the Spikes MOVE (`sideCondition:'spikes'`, `target:'foeSide'`) is NEVER-MISS + DRAW-FREE — it increments the CASTER's FOE side's `spikes` layer by 1, CAPPED at 3 (a 4th Spikes FAILS, `-fail`, draw-free); `landed` FALSE (no in-tryMoveHit Update). The SWITCH-IN DAMAGE is the gen-3 `runSwitch`'s `runEvent('EntryHazard')` (gen4-inherited; ORDER: EntryHazard → SwitchIn → `if (!pokemon.hp) return` → ability `Start` — so a Spikes-KO SKIPS the entrant's ability Start): GROUNDED-ONLY (a Flying-type / Levitate entrant takes ZERO), amount `max(floor([_,3,4,6][layers]·maxhp/24),1)` = `max(floor(maxhp/8),1)` / `max(floor(maxhp/6),1)` / `max(floor(maxhp/4),1)` for 1/2/3 layers (the resolved `spikes.onEntryHazard` → `damage()` → `clampIntRange(_,1)`), DRAW-FREE (the deterministic `this.damage`; the nested `runEvent('Damage')` has no drawing handler for the modeled abilities — VERIFIED vs the omniscient sim `harness/probe_spikes_rng.js`). A Spikes hit that zeroes HP faints the entrant (the runAction tail's `process_faints` sets fainted + `check_fainted` flags → forces ANOTHER forced replacement, which ALSO takes Spikes) — wired through the existing faint/replacement machinery, no Quick Claw / extra draw. The `spikes` layer is a SIDE condition → it PERSISTS across switches (cleared only at construction). DEFERRED (excluded / fail-loud): **Toxic Spikes** + **Stealth Rock** (NOT gen3), **Rapid Spin** (the hazard-CLEAR move — a damaging move the fuzz won't pick as a modeled status move; hazards persist). Spikes is the only gen-3 entry hazard. Validated by `tests/spikes_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+**SPIKES-LAYERS**)+SEED+winner differential to GAME-END over 5 scenarios × 80 seeds in gen3customgame: 400 runs, ~3475 decision rows, ~3475 seed + ~6950 HP + ~6950 spikes-layer assertions, 1440 switch-in-chip + 160 spikes-KO-on-entry rows, 400 wins — lay 1/stack 2/3 + the increasing damage, a Flying/Levitate IMMUNE entry, a Spikes-at-max FAIL, a spikes-KO-on-entry → forced replacement, and spikes-into-a-real-battle) + deterministic unit gates (`spikes_move_increments_foe_side_draw_free_and_caps_at_three`, `spikes_switch_in_damage_grounded_per_layer_draw_free`, `spikes_switch_in_flying_and_levitate_take_zero`, `spikes_ko_on_switch_in_zeroes_hp_draw_free`). The draw model was nailed by `harness/probe_spikes_rng.js`. **PHAZING — ROAR + WHIRLWIND now BUILT** (`run_status_move`'s phaze arm + `drag_in` at the `turn_loop` runAction tail; new `MoveResolution::force_switch_foe`): the gen-3 `forceSwitch` moves force the FOE to switch to a RANDOM eligible team member. The draw model, VERIFIED vs the omniscient sim PRNG probe `harness/probe_phaze_rng.js`: (1) PRIORITY −6 → the phazer moves LAST; (2) ACCURACY — gen-3 Roar/Whirlwind resolve to **`accuracy: 100`** (NOT `true`! — the SURPRISE this layer surfaced: the base Showdown data lists `accuracy:true`, but our `gen3_moves.json` already carries the resolved gen-3 value 100), so a phaze is NOT never-miss — it DRAWS `randomChance(100,100)` (always passes but CONSUMES a draw), drawn in the phaze arm BEFORE routing to the drag; (3) THE RANDOM TARGET DRAW — `forceSwitch` (battle-actions.ts:1167) sets the foe's `forceSwitchFlag` IFF `canSwitch(foe.side)` (≥1 eligible non-active, non-fainted bench mon); the ACTUAL drag happens at the runAction tail (battle.ts:2350, AFTER the move body, BEFORE faintMessages) via `drag_in` → `eligible_switch_ins` (= `possibleSwitches`, the array-order bench list) → `random(n)` (the `sample` — ONE draw, EVEN for n==1, since `random(1)` returns 0 but still calls `rng.next()` — the n=1 draw gotcha) → `execute_switch` (the array swap + the entrant's `updateSpeed` + the `insert_runswitch` enqueue). So the dragged mon takes Spikes via the EXISTING `runSwitch` EntryHazard, fires its ability `Start`, and a Spikes-KO on entry chains a NORMAL replacement — all through the existing switch machinery. A phaze with NO eligible foe (its last mon) FAILS draw-free (only the accuracy roll). **(4) THE PROTECT BLOCK — gen-3 Roar AND Whirlwind carry the `protect: 1` flag, so a Protect / Detect on the TARGET BLOCKS the phaze at `runEvent('TryHit')` (AFTER the accuracy roll) → NO `forceSwitchFlag` → NO drag → NO `sample` draw (`-activate Protect`, the target STAYS active). The phaze arm checks `protect_blocks(foe, foe_slot, false)` right after its accuracy roll (mirroring the leechseed / standalone-status arms); Substitute does NOT block a phaze (Roar/Whirlwind carry `bypasssub: 1` — no substitute check). MISSING this was the multi-phaze `sample` draw-POSITION desync (see EDGE_CASES.md ✅ FIXED): the port dragged an EXTRA `sample` into a protected foe the sim left in place, shifting every LATER phaze's `sample` PRNG position (same total draw COUNT, wrong `sample` INDEX, compensated elsewhere → the boundary seed matched while the dragged mon differed).** The phazed-OUT mon's boosts/volatiles are cleared (`execute_switch`); the dragged mon does NOT act this turn. A phaze DRAG that fires sets `pending_phaze_drag` → `DecisionRecord.phaze_drag` (a coverage/diagnostic signal only, no effect on any draw/state; the e2e capstone reads it to count phaze-drag decisions — a Protect-blocked / no-bench phaze does NOT set it). DEFERRED (fail-loud — `modeled_phaze_move` lists ONLY `roar`/`whirlwind`, everything else falls to the status-move fail-loud guard): **Haze** (resets boosts — a DIFFERENT mechanic, NOT `forceSwitch`), Perish Song, Roar of Time (not gen3). Roar + Whirlwind are the ONLY gen-3 phaze moves (`isPhaze` == `forceSwitch`). Validated by `tests/phaze_test.rs` (a per-seed PER-DECISION STATE(+HP+SPIKES-LAYERS+DRAG-SPECIES)+SEED+winner differential to GAME-END over 7 scenarios × 80 seeds in gen3customgame: 560 runs, 10388 seed + 20776 HP + 20776 spikes-layer assertions, 2795 drag decisions + 1769 phaze-into-spikes-damage rows, 560 wins — the seed sweep makes DIFFERENT mons get dragged [≥2 distinct per multi-bench scenario, the random-target proof], Roar/Whirlwind random drag, the n=1 sample, a Roar that FAILS, Roar INTO Spikes, repeated Roar into a stochastic spikes-KO, phaze-into-a-real-battle) + 4 DETERMINISTIC `tests/regression_test.rs` pins (`phaze_draws_accuracy_then_n1_sample_seed`, `phaze_fail_draws_only_accuracy_no_sample_seed`, `phaze_drag_into_a_spikes_ko_chains_a_replacement`, and **`phaze_blocked_by_protect_draws_no_sample_and_leaves_the_target`** [P4 — a Protect BLOCKS a Roar: accuracy drawn, NO `sample`, the protector stays active], ground-truth seeds from `harness/probe_phaze_regression_rng.js` [incl. its PHAZE-PROTECT case]). Phaze is **INCLUDED in the e2e capstone** (`PHAZE_E2E_EXCLUDED = false`, bit-for-bit — **1035 phaze-DRAG decisions across the 220-battle strict gate**, `filtered_diverged == 0`, `phaze_decisions >= 50` coverage floor) after fixing the Protect-blocks-phaze desync (above). Admitting phaze also surfaced an unmodeled `destinybond` (a reactive `volatileStatus` move Gengar can carry): it is now in `gen_e2e_fuzz.js`'s `MOVE_ID_BLOCKLIST` (belt-and-braces — `isModeledMove` already excludes it, so `pickMove` never picks it), and the port FAIL-LOUDS on it (pinned by `destinybond_status_move_panics_fail_loud`). **LEECH SEED now BUILT** (`run_status_move`'s leechseed arm + the `LeechSeed` `ResidualAction` + `apply_leech_seed`; new `MonState::leech_seed: Option<usize>`): a foe-targeting `volatileStatus:'leechseed'` Status MOVE (type Grass, **accuracy 90**) that plants the `leechseed` volatile on the FOE; each end-of-turn the seeded mon loses `⌊maxhp/8⌋` and the SEEDER's CURRENT active heals it. The draw model, VERIFIED bit-for-bit vs the omniscient sim PRNG probe `harness/probe_leechseed_rng.js`: (1) the MOVE DRAWS `randomChance(90,100)` — it CAN miss, drawn UNCONDITIONALLY (even into a Grass-immune or already-seeded target — VERIFIED: a splash/splash turn draws 1 [Quick Claw]; a Leech-Seed turn — land / Grass-immune / already-seeded-fail — ALL draw 2 [accuracy + Quick Claw]); (2) a **GRASS** target is IMMUNE (`onTryImmunity` → `!hasType('Grass')`; accuracy still drawn, then `-immune`, NO volatile); (3) an **ALREADY-SEEDED** target's re-seed FAILS (`addVolatile` false; accuracy drawn, "did nothing", the existing volatile unchanged); (4) on a landed non-immune non-already-seeded hit it PLANTS the volatile (DRAW-FREE); `landed` FALSE (a status `moveHit` returns `undefined` → the in-tryMoveHit Update is skipped). THE LEECH RESIDUAL (the risk area — DRAW-FREE but ORDER-SENSITIVE): the gen4-inherited override is **`onResidualOrder: 10, onResidualSubOrder: 5`** (NOT the base-data order-8) — so at order 10 the residual ladder is **Leftovers sub 4 → LEECH sub 5 → status DoT sub 6** (with sand/hail field-residual order 8 BEFORE all three). VERIFIED order `sandstorm[o=8] → leftovers[o=10,s=4] → leechseed[o=10,s=5] → brn[o=10,s=6]`. The drain is `⌊maxhp/8⌋` clamped to the seeded mon's HP (the sim's `damage()` return), the seeder's active HEALS that dealt amount (clamped to its maxhp) — applied even when the drain KOs the seeded mon (the heal is inside the same `onResidual`, before `faintMessages`). The **SEEDER-FAINTED gate**: if the seeder's active is fainted/0-HP the WHOLE `onResidual` returns early (`if (!target || target.fainted || target.hp <= 0) return`) — no drain, no heal. The leech is gathered with the VOLATILES (after the status DoT, before Leftovers, mirroring `findPokemonEventHandlers`'s status→volatiles→item order) so its handler participates in the residual speed-sort tie-shuffle (two seeded mons at equal speed TIE at order 10 sub 5 → one `random(0,2)`). The volatile clears on switch-out (`execute_switch`) and on faint (`process_faints`). DEFERRED (fail-loud in `apply_leech_seed`): a **Liquid Ooze** target reverses the drain (the seeder takes damage) — rare in gen-3 OU, excluded from the e2e filter. Validated by `tests/leechseed_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+SPIKES-LAYERS+**LEECH-SEEDED**)+SEED+winner differential to GAME-END over 7 scenarios × 80 seeds in gen3customgame: 560 runs, 5001 seed + 10002 HP + 10002 leech-state assertions, 3838 leech-seeded rows, 560 wins — seed lands→drain+heal, Grass-immune, already-seeded fail, the leech-drain KO, the leech+Leftovers+sand+burn 4-way residual ORDER, the seeder-replaced heal-follows, leech-into-a-real-battle) + 3 DETERMINISTIC `tests/regression_test.rs` pins (`leech_residual_order_leftovers_sand_burn`, `leech_handler_tie_at_equal_speed_draws_one_shuffle`, `leech_seeder_fainted_skips_the_drain`, ground-truth seeds from `harness/probe_leechseed_regression_rng.js`). **SUBSTITUTE now BUILT** (`run_status_move`'s substitute arm + `absorb_into_sub` in `run_move` + the secondary-suppression in `apply_secondaries`/`apply_triattack_secondary` + the sub-block in the status/leech arms; new `MonState::substitute: Option<u16>`): the SUBSTITUTE MOVE (`volatileStatus:'substitute'`, `target:'self'`, never-miss) spends `floor(maxhp/4)` HP to create a decoy with that much HP that ABSORBS incoming foe hits. The draw model, VERIFIED bit-for-bit vs the omniscient sim's PRNG probes (`harness/probe_substitute_*.js`): (1) the MOVE is NEVER-MISS (no accuracy draw); it FAILS draw-free if a `substitute` is ALREADY present OR `hp <= floor(maxhp/4)` (can't afford — VERIFIED: hp == floor(maxhp/4) FAILS, +1 SUCCEEDS); on success it pays `floor(maxhp/4)` HP + creates the volatile with that HP (DRAW-FREE; `landed` FALSE). (2) THE ABSORB/BLOCK CRUX (the SURPRISE that **CONTRADICTED the task's stated "one fewer random(100)" assumption** — settled by the probe, the project's source of truth): a DAMAGING foe move into a substituted mon draws acc+crit+damage as normal (UNCHANGED count) and the damage hits the SUB's HP (the sub BREAKS at 0; the excess does NOT carry to the mon in gen-3) — **and the per-move SECONDARY `random(100)` is STILL DRAWN** (gen-3 `secondaries()` iterates the now-`null` target list, so the draw fires — the SAME count as a bare hit), but its FOE-TARGETING EFFECT does NOT apply (no status / no foe stat-drop / no flinch / no confusion, AND **no confusion `random(2,6)` / Tri-Attack `random(3)` follow-on draw** — those are SUPPRESSED). A **SELF-boost** secondary (Meteor Mash +1 Atk to the USER) is EXEMPT — it STILL APPLIES through a sub (`secondary.self.boosts` targets the SOURCE, not the null sub target; VERIFIED vs the sim), so the suppression is FOE-targeting-only (`effect != "self_boost"`). So `absorb_into_sub` routes the damage; `apply_secondaries`/`apply_triattack_secondary` take an `absorbed_by_sub` flag that DRAWS the `random(100)` then SKIPS the FOE-targeting apply+follow-on (but keeps the self-boost). This is draw-COUNT-NEUTRAL vs a bare hit (unlike the phaze `sample` that ADDS a draw). (3) a STATUS / stat-DROP move (Thunder Wave / Toxic / Leech Seed / a -stat secondary) is BLOCKED by the sub (accuracy still drawn, then `-fail`/no effect — DRAW-FREE past accuracy; the status arm + the leech arm both check `substitute.is_some()` after the protect block). (4) a CONFUSION self-hit hits the MON, NOT the sub (the self-hit's `this.damage` bypasses the `onTryPrimaryHit` sub-intercept — `on_before_move`'s confusion arm is unchanged: `randomChance(1,2)` then `random(16)` on the mon). (5) PHAZE (Roar / Whirlwind) BYPASSES the sub — the user is dragged anyway (forceSwitch is a runAction-tail effect, not a moveHit target). The substitute clears on switch-out (`execute_switch`) and on faint (`process_faints`). Validated by `tests/substitute_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS+BOOSTS+CONFUSION+SPIKES-LAYERS+**SUB-HP**)+SEED+winner differential to GAME-END over 9 scenarios × 80 seeds in gen3customgame: 720 runs, 4320 decision rows — create + already-subbed FAIL, the low-HP create FAIL boundary, a HELD-sub absorb (sub HP drops, secondary suppressed), the BREAK no-carry, a blocked STATUS move, a blocked STAT-DROP secondary, the CONFUSION self-hit hitting the mon, the phaze drag-through, and sub-into-a-real-battle to a win) + 5 DETERMINISTIC `tests/regression_test.rs` pins (`substitute_absorbs_a_hit_but_the_secondary_random_100_still_draws` [the draw-COUNT crux], `substitute_break_does_not_carry_excess_to_the_mon`, `confusion_self_hit_behind_a_sub_hits_the_mon_not_the_sub`, `tri_attack_into_a_sub_draws_random_100_but_not_the_sample_random_3` [the random(3)-suppression draw-COUNT], `self_boost_secondary_still_applies_through_a_sub` [a SELF-boost is exempt from the sub-block], ground-truth seeds from `harness/probe_substitute_regression_rng.js`). The draw model was nailed by `harness/probe_substitute_rng.js` + `probe_substitute_secondary.js` + `probe_substitute_confusion.js` + `probe_substitute_status.js`. Substitute is **INCLUDED in the e2e capstone** (`SUBSTITUTE_E2E_EXCLUDED = false`, 284 substitute-MOVE / 320 sub-up decisions across the 220-battle strict gate, bit-for-bit) after FIXING the SWITCHING/weather bug it surfaced — the **`eachEvent('WeatherChange')` switch-in tie-shuffle**: `Field.setWeather` (field.ts:87) ends with `this.battle.eachEvent('WeatherChange', sourceEffect)`, a 2-active `speedSort` that draws ONE `random(0,2)` tie-shuffle iff the actives TIE on cached speed (gen-3 `>=7` Update-nest NOT reached). When a Sand Stream / Drizzle / Drought entrant CHANGES the weather on a MID-TURN switch-in, that shuffle fires INSIDE the `runSwitch` runAction before its trailing Update; the port set the weather draw-FREE and MISSED it (e2e_84 dec4: a 213-Tyranitar switches in under Sand Stream while a 213-Suicune subs → the sim drew 8, the port 7 — a draw-COUNT desync, the SAME class as `forced_replacement_recaches_speed_seed`, in the SWITCHING/weather layer, NOT the substitute arm). The fix: `run_switch` snapshots `(weather, weather_turns)` across the ability `Start` and returns whether the weather changed; `turn_loop`'s `RunSwitch` handler then fires one `each_event_shuffle()` (for EVERY mid-turn switch-in — voluntary / forced-replacement / phaze-drag — since all route through that runAction). The `>start` path (`run_start_switchins` → `single_event_ability_start` directly) is UNAFFECTED (stays draw-free; `switchin_test`'s zero-draw assertion holds). Pinned by `regression_test.rs::switch_into_a_tie_under_sand_draws_the_weather_change_shuffle_seed` (a constructed 213-vs-213 switch-into-tie-under-Sand-Stream, real-Showdown ground-truth seed from `harness/probe_switch_tie_weather_regression_rng.js`; verified a TRUE pin by reverting the fix). See EDGE_CASES.md "switch-in-into-a-speed-TIE" (now FIXED) + the `probe_switch_sand.js` control (the +1 shuffle: Sand-Stream switch-into-a-TIE draws 9 vs no-weather 7 vs distinct-speed 1). DEFERRED: Baton Pass passing a sub, Shed Tail (gen-9), the `bypasssub`/`infiltrates` move flags (none on a modeled gen-3 move). **FIXED-DAMAGE / FIXED-FORMULA MOVES now BUILT** (`run_fixed_damage_move`, routed by `is_fixed_damage_move` in `run_move` BEFORE the `category == Status` branch — these carry `basePower:0` so `derive_category` classifies them Status; new id-gated helper `fixed_damage_amount`): the `damage:` / `damageCallback` moves that BYPASS `getDamage` (NO crit roll, NO 16-way damage roll). MODELED — **Seismic Toss / Night Shade** (`damage:'level'` → the USER's level, e.g. 100), **Sonic Boom** (fixed 20), **Dragon Rage** (fixed 40), **Super Fang** (`damageCallback` = `max(floor(target.hp/2),1)`). The draw model, VERIFIED bit-for-bit vs the omniscient sim's PRNG probe (`harness/probe_fixeddamage_rng.js`): (1) ACCURACY — `randomChance(acc,100)`, drawn UNLESS never_miss — **Seismic Toss / Night Shade / Dragon Rage are acc-100 but NOT never-miss so they STILL draw ONE accuracy roll** (the phaze acc-100 precedent), **Sonic Boom / Super Fang are acc-90 and CAN genuinely miss**; this is the ONLY per-move draw (NO crit, NO damage roll, NO secondary); (2) TYPE IMMUNITY — accuracy-drawn-THEN-`-immune` (the SAME short-circuit + draw count as a normal damaging move, via `move_is_immune`): **Seismic Toss (Fighting)→a GHOST**, **Night Shade (Ghost)→a NORMAL**, **Sonic Boom / Super Fang (Normal)→a GHOST** all report `-immune` (NOT `-miss`) with ZERO damage; (3) DAMAGE — the exact fixed amount applied through the EXISTING `absorb_into_sub` / `apply_damage` / deferred-faint machinery, so a fixed-damage KO goes through the normal faint/win/Quick-Claw protocol (no Quick Claw on a deciding faint). **The SUBSTITUTE interaction (VERIFIED, and it CONTRADICTED a naive assumption — settled by the probe):** a fixed-damage move into a sub hits the SUB (the fixed NUMBER hits the sub HP, breaks with no carry, `-activate Substitute [damage]` / `-end`) — and **Super Fang still halves the MON's current hp behind a sub** (the `damageCallback` reads `target.hp` BEFORE the sub-intercept redirects the resulting number; VERIFIED: SF into a full-HP-536 Blissey behind a 178-HP sub deals floor(536/2)=268 → the sub BREAKS, NOT floor(178/2)=89). `landed` is TRUE on a hit (a `damage:` move returns a truthy number → the in-tryMoveHit Update fires), FALSE on miss/immune/block. FAIL-LOUD (`run_fixed_damage_move` PANICS): the DEFERRED fixed-damage family — **Psywave** (variable, draws RNG), the OHKO moves **Fissure / Horn Drill / Guillotine** (accuracy-gated instakill + level gate), **Counter / Mirror Coat / Bide** (reactive), **Endeavor** (sets hp to the user's) — is routed here by `is_fixed_damage_move` (so it can NEVER silently no-op / desync) but has no `fixed_damage_amount` entry. Validated by `tests/fixeddamage_test.rs` (a per-seed PER-DECISION STATE(+HP+STATUS)+SEED+winner differential to game-end over 9 scenarios × 80 seeds in gen3customgame: 720 runs, 4144 seed + 8288 HP assertions, 2469 fixed-damage-hit decisions, 720 wins — Seismic Toss chip / into-a-Ghost-immune / Night Shade into-a-Normal-immune + lands / a fixed-damage KO-to-win / into-a-Substitute / Sonic Boom+Dragon Rage incl. the acc-90 miss / Super Fang halving incl. the miss / fixed-damage-into-a-real-battle) + 4 DETERMINISTIC `tests/regression_test.rs` pins (`seismic_toss_deals_user_level_damage`, `seismic_toss_into_a_ghost_is_immune_accuracy_only_seed`, `night_shade_into_a_normal_is_immune`, `fixed_damage_into_a_substitute`; ground-truth seeds from `harness/probe_fixeddamage_regression_rng.js`). e2e: the `MODELED_FIXED_DAMAGE_MOVES` set is in `gen_e2e_fuzz.js`'s `isModeledMove` (admitted early, before the `basePower>0`/`m.damage`/`m.damageCallback` rejects), but the regenerated 220-battle golden has **0 fixed-damage-MOVE decisions** because NONE of the 22 filter-clean teams happens to carry one (the leech-seed situation) — so the layer is proven by its DEDICATED golden + the 4 pins, not the e2e. **TAUNT + DISABLE (the move-SELECTION-restriction layer, `gen3_taunt_disable_v1`) now BUILT** (`run_status_move`'s taunt/disable arms + the `move_usable`/`must_struggle` restriction + the `on_before_move` execution-time cants + the residual duration ticks + the endTurn `runEvent('DisableMove')` handler-sort shuffle): **Taunt** (Dark, acc 100 — DRAWS `randomChance(100,100)`) applies a FIXED-duration-2 volatile (NO duration draw; the base onStart's `duration++` is SHADOWED by the gen4 mod's replaced onStart — probe-proven constant 2 in ALL branches) that makes every Status-category slot un-selectable (the derived-Status set MINUS the fixed-damage family) + cants a QUEUED status move at execution (`onBeforeMove` priority 0, AFTER the paralysis roll), residual tick at order 10/subOrder 15 (gen4-inherited, NOT the base's order 15); **Disable** (Normal, acc 55 — CAN miss) disables the target's `last_move` slot for **stored = disabler-faster ? random(2,6) : random(2,6)+1** (the gen4-inherited onStart `!willMove → duration++`; PROBE-SETTLED — a base-source reading mis-predicts by a constant +1), onTryHit FAILS draw-free with no lastMove, + cants a QUEUED now-disabled move at execution (`onBeforeMove` priority 7, BEFORE confusion/paralysis — a paralyzed+disabled mon draws NO para roll), residual tick at NO_ORDER/subOrder 2 (the Condition default; gen3 DELETES gen4's 10/13); both `protect: 1` + `bypasssub: 1`, re-application fails draw-free, cleared on switch-out + faint (with `last_move` reset). All slots restricted (taunt × disable × Choice lock × 0 PP) → forced Struggle. See "Taunt + Disable" below. Validated by `tests/taunt_disable_test.rs` (9 scenarios × 80 seeds: 720 runs, 4723 seed + 8595 taunt + 8595 disabled-slot assertions, free-up boundaries on BOTH disable branches — the golden gate FAILS on a ±1 duration perturbation, proven both ways) + 4 revert-verified `regression_test.rs` pins (TD1–TD4). |
| `protocol.rs` | **types + Phase-1+2 emit API, validated** | `Player`, `Choice` (→ wire grammar `move 1`/`switch 3`/…), `ProtocolLine` (raw bytes = source of truth), AND the **`ProtocolBuilder`** — an append-only, **PRNG-free** line buffer the engine writes at each observable event (one sim-mirroring retro-edit excepted: `attr_last_move_still` = `attrLastMove('[still]')`, used by the Disable 0-PP-guard fail). Centralizes the fiddly formatting in ONE place: `MonRef` (`p<N>a: <Name>`) / `HpStatus` (the three variants `x/y` / `x/y <status>` / `0 fnt`) / `Cause` (`[from] item: …` / `[from] ability: …` / `[from] move: …` / `[from] <bare>`) / `STAT_TOKENS` (the `-boost`/`-unboost` stat names). Typed constructors for the Phase-1 line types (framing / `turn` / `upkeep` / separator / `move`+`[miss]`/`[still]` / `switch` / `drag` / `-damage` (+ the `damage_of` `[from] Recoil|[of] <target>` form, `gen3_pp_tracking_v1`) / `-heal` / `faint` / `-crit` / `-supereffective` / `-resisted` / `-immune` / `-miss` / `win` / `tie`) **PLUS the Phase-2 types** (`status`+`[from] move: Rest` / `curestatus`+`[msg]` / `cant` / `boost`/`unboost` by sign / `weather` SET+`[upkeep]` / `ability`+detail / `fail`+`[weak]` / `sidestart` / `volatile_start`/`volatile_end`/`activate`+`[damage]` / `singleturn`). Disabled by default (the seed suite pays zero cost + draws nothing); `run_full_battle_logged` enables it. See "Protocol emission" below. |
| `battle.rs` | **write_line DONE, validated** | `BattleOptions`/`PlayerOptions`/`PackedTeam` + `Battle::start`/`start_with_switchins` (construction ± switch-in events over `state::BattleState`) + **`BattleStream::write_line`** (`gen3_writeline_stream_v1`) — the streaming drop-in for the bridge's `local_sim_bridge.js` pattern: `>start {formatid,seed}` / `>player pN {name,team}` / `>pN move K\|switch N` per write, returning EXACTLY the omniscient `\|...\|` chunk the real Node `BattleStream` flushes for that write (per-write byte-gated by `tests/writeline_test.rs` vs `harness/gen_writeline_capture.js` — 44 battles / 2377 writes / 7276 filtered lines, all 22 scenarios). INTERNALS (`gen3_bridge_incremental_replay_v1`): **TRULY INCREMENTAL** — `BattleStream` OWNS one live `Battle` + the shared **`turn::FullBattleDriver`** stepping primitive (the SAME one `run_full_battle` drives — the ONE turn-loop in the crate). Once both `>player` writes are in it builds the battle + emits the framing ONCE; each `>pN` write feeds ONE one-sided `ScriptDecision` to the driver, which advances the LIVE battle to its next request boundary and returns ONLY the new omniscient-log suffix — **O(1)/input, O(N)/battle, NEVER re-simulating a prior turn** (was replay-from-genesis, O(turns²)/battle — the wedge). The per-side pending acceptance the old accumulator did now lives INSIDE the driver (`FullBattleDriver::pending`), so the write-order semantics are preserved bit-for-bit (byte-gated by `writeline_test`, which now exercises the incremental primitive). An illegal choice is rejected like `side.choose` (no lines, boundary open). The `>start` seed must be the PRE-first-decision seed (the sim's turn-0 construction window is unmodeled — the protocol-replay convention). CHOICE REVISION is FIRST-accepted-wins, NOT the sim's last-write-wins (`gen3_writeline_choice_revision_v1`, review finding F4): a repeat pre-commit `>pN` is DISCARDED by the driver's per-side accumulator (`pending` sets a side only `if for_side(side).is_none()`) instead of REPLACING it, whereas the sim's `side.choose` clears + re-parses on each write (probe `harness/probe_f4_choice_revision.js`, resolved Dex.mod('gen3'): `>p1 move 1` then `>p1 move 2` executes Earthquake, seed-identical to a single `>p1 move 2`; a 1→2→3 chain executes Rest). Unchanged from the old replay surface (which relied on the SAME `run_full_battle` accumulator); the real bridge sends exactly one choice per request, so a revised `>pN` is unreachable in production. The `\|request\|`/sideupdate frames + the per-side privacy fold (downstream of the omniscient stream) are now DONE in **`bridge.rs`** (the row below). Still OUT of scope on `write_line`: `>forcewin`/`>reseed`. Snapshot/reseed/choose on `Battle` stay `todo!()` (the search layer's JS clone path covers them today). |
| `bridge.rs` | **DONE, validated (in-scope corpus)** | The PER-SIDE (`p1`/`p2`) protocol streams a drop-in for `src/utils/bridge/local_sim_bridge.js` emits — Phase-1 gaps **G1 (`\|request\|` JSON)** + **G2 (`getPlayerStreams` split + HP-privacy fold)** ON TOP of the unchanged omniscient stream (additive, like `write_line`). `run_full_battle_bridge(opts, cmds, dex) → BridgeStreams{p1,p2}` (and the chunked `run_full_battle_bridge_core`) build the per-side streams by snapshotting the pending request at each boundary, folding the flushed omniscient batch to each side, and injecting each side's `\|request\|`/`\|error\|` — these stay **replay-from-genesis** (O(turns²)) as the **REFERENCE ORACLE** the parity test checks the incremental path against. The **PRODUCTION path is the incremental `BridgeSession`** (`gen3_bridge_incremental_replay_v1`): a PERSISTENT session owning ONE live `Battle` + the shared `turn::FullBattleDriver` (like `BattleStream`), advancing ONE request boundary per fed CMD (`feed_cmd`) and appending ONLY the new chunk — **O(1)/CMD, O(N)/battle, NEVER re-simulating a prior turn** (`run_full_battle_bridge_incremental` = feed the whole stream to one session; `sim_bridge` holds one across CHOOSE lines — the wedge fix). Byte-parity (chunks + `ended` + `script` + the A2 seed anchor) vs the genesis core is gated by `bridge_test::bridge_incremental_matches_genesis_replay` over the trapping golden + constructed LONG STALL battles (single-move Splasher teams, hundreds of decisions incl. forced replacements) + the persistent one-CMD-at-a-time path; the timing gate `timing_bridge_incremental_linear_vs_genesis_quadratic` (ignored) shows a flat ~12 µs/CMD incremental vs the genesis per-CHOOSE cost climbing to ~0.6 s at decision 884. **G2 fold** (from `battle.js::extractChannelMessages` + `pokemon.js::getHealth` + `mods/gen3/abilities.js`): HP-bearing lines (`\|switch\|`/`\|drag\|`/`\|-damage\|`/`\|-heal\|`/`\|-sethp\|`) show the OWNER exact HP, the OTHER side `ceil(hp*100/maxhp)/100` (clamp 100→99 when `hp<maxhp`; `0 fnt` unchanged) — but ONLY in a non-debug format; **gen3customgame sets `reportExactHP`** (its `[Gen 3] Custom Game` def has `debug:true`) so both sides see exact HP. Owner-only (empty-`shared`, dropped for the other side): gen3 Pressure's `\|-ability\|pNa: X\|Pressure\|[silent]` (`addSplit`) + the Intimidate no-activate `\|-hint\|` (owner tagged by the driver since a hint has no `pNa:` prefix). **G1 request** serialized from the crate's EXISTING legality (`is_trapped`/`move_usable`/`switch_flag`), Showdown field+key order + compact `JSON.stringify` (no spaces after `:`/`,`): per-move `{move,id,pp,maxpp,target,disabled}` (typed HP → bare `id:"hiddenpower"` + `move:"Hidden Power <Type> <BP>"`), `active[0].{maybeTrapped\|trapped}` — the `'hidden'` traps (Arena Trap / Magnet Pull) show `maybeTrapped` on the first request (firmed to `trapped` only after a rejected switch → `[Unavailable choice]` + re-request), but **Shadow Tag** (the gen3 mod sets `trapped = true` directly) shows `trapped:true` from the FIRST request + a rejected switch draws `[Invalid choice]` with NO re-request (`state::trap_is_firm`, `gen3_shadowtag_firm_trap_v1`); a forced-Struggle mon also gets a per-side OWNER-ONLY `\|-activate\|<mon>\|move: Struggle` sideupdate line before the broadcast `\|move\|` (`gen3_struggle_activate_sideupdate_v1`) — both found by the bridge/request A/B fuzzer. `side.pokemon[]` in the crate's Showdown-faithful swap-reordered array order; `forceSwitch:[true]`+conditional `noCancel` (iff <2 non-wait requests, mirroring `multipleRequestsExist`); the non-choosing side's `{wait:true}`; the trapped-reject `\|error\|` + `trapped:true` re-request with trailing `"update":true`. gen3 has NO team preview. gen3ou framing (`\|tier\|[Gen 3] OU` + the 12 `\|rule\|` list) spliced in via `reframe` (the crate's `emit_framing` hard-codes the gen3customgame tier). **BYTE-GATE**: `tests/bridge_test.rs` — the trapping golden (`bridge_trapping_golden.txt`: 5 battles — arena_trap_reject / magnet_pull_reject / **shadow_tag_firm_trap** + **taunt_struggle / pp_stall_struggle** (the forced-Struggle-resolve class, `gen3_bridge_struggle_resolve_v1`) — 581 CHUNK lines, `\|request\|` / `trapped:true` / `\|error\|[Invalid\|Unavailable]` frames + the `\|move\|…\|Struggle` + `\|-damage\|…\|[from] Recoil` lines) is byte-for-byte, the definitive Phase-1 pass (the constructed IN-SCOPE corpus: explicit genders, MODELED moves only). The two Struggle scenarios close the hole that let the Struggle wedge escape — NO committed node-vs-rust byte test drove a battle to Struggle before, so `bridge::resolve_choice`'s `move struggle` NAME→`Move(0)` branch was ungated (it returned None → the boundary never committed → an infinite re-request under `--use-bridge=rust`); `taunt_struggle` (a Taunt-stranded status-only Skarmory, non-ended prefix — gen3 Taunt strands for exactly one forced-Struggle turn) + `pp_stall_struggle` (a single-Mean-Look mon drained to 0 PP, plays to a `\|win\|`) both answer the struggling side with the wire NAME `move struggle` and revert-verify the gate FAILS with the fix disabled. `\|debug\|` free-form sim text (gen3customgame is `debug:true`) is filtered from both sides by the capture harness, matching the rust bridge's non-emit. The gen3ou `bridge_capture_golden.txt` (30 battles) is a scope-AUDIT: 28 fail-loud on unmodeled moves (Counter/Wish/Encore/Curse/Baton Pass/Endeavor/Refresh/Sunny Day — out of the ENGINE's faithful-replay scope, not a bridge defect) + ALL 30 carry a gendered species with an UNSPECIFIED gender → an unmodeled construction-time gender-ratio `sample` PRNG draw that desyncs the battle + adds the drawn gender to `details`; the audit asserts the bridge layer is byte-correct (gender-tolerant) up to that first ENGINE-scope divergence (framing + full first request proven on the 2 replayable battles). When the engine models the full move set + the gender draw, these become full byte-equal with NO bridge change. Entry point for the request A/B fuzzer: **`src/bin/bridge_replay.rs`** (`bridge_replay <golden> [id] [--print] [--ab]`; `--ab` = one JSON verdict/battle, panic-caught — the `harness/bridge_ab_fuzz.js` driver; also replays a repro DIR). See "## Bridge / request A/B fuzzer". Observation-only: the omniscient path is UNCHANGED (protocol/writeline/e2e_fuzz green, e2e md5 `a23d77ac60d4af168b8a4428f0b465c9`; the two fuzzer-found fixes [`gen3_shadowtag_firm_trap_v1` firm `trapped`, `gen3_struggle_activate_sideupdate_v1` per-side Struggle line] are bridge-path only + the `switch_details` gender addition is byte-neutral for the genderless goldens). **PHASE 2 (`gen3_sim_bridge_dropin_v1`): the CHUNK-aware `run_full_battle_bridge_chunked(opts,cmds,dex) → BridgeChunks{chunks:Vec<SideChunk>}`** preserves the `getPlayerStreams` FLUSH boundaries (framing → 3 chunks/side split at the two `\|player\|` lines; a resolved turn → 1 log chunk/side + 1 `\|request\|` chunk/side; a trapped reject → a `\|error\|` chunk + [hidden trap] a `trapped:true` re-request chunk; a forced-Struggle commit → a `\|-activate\|move: Struggle` chunk) — the flush unit the Node bridge base64-frames as ONE `pN <b64>` stdout line. `run_full_battle_bridge` now delegates to it via `.flatten()` (so the line-level `bridge_test` golden still validates the chunk logic — pinned by `chunked_flatten_equals_flat_streams`). The turn-0 CONSTRUCTION WINDOW is now MODELED (`gen3_turn0_construction_v1`): `Battle::start_with_turn0_construction` / `BridgeSession::new_construct_turn0` run the REAL turn-0 from the RAW `>start` seed — the per-mon gender `sample(['M','F'])`, the `start` action's insertChoice tie-break + `eachEvent('Update'/'WeatherChange')` shuffles (the `[start, runSwitch, runSwitch]` runActions through the shared `turn_loop`/`insert_runswitch` machinery), then the `endTurn` DisableMove/trap shuffles + Quick Claw — so the post-construction PRNG state + the sampled genders match the real sim bit-for-bit even on a **speed-TIED lead** or an **unspecified-gender** mon. The window runs with logging OFF and the framing is re-emitted afterwards, so it also RECORDS the resolved `runSwitch` order (`BattleState::turn0_switchin_order`, `gen3_turn0_construction_mirror_order_v1`, ROUND 34) — at a raw-Speed TIE that order is the `insertChoice` PRNG draw, and re-DERIVING it as "faster-first, tie = side order" permuted the emitted Intimidate / weather-setter block on a tied lead (seed + board correct, bytes wrong — the `sbd_msdd8698_b293` Masquerain-mirror repro). Gated by `tests/turn0_construction_test.rs` (distinct / speed-tie / weather-setter-tie / gender-sample vs the omniscient sim, + the mirror-order & emission↔state pins). The old pure `advance_seed_for_construction` seed hack (modeled ONLY the Quick Claw → desynced ties + genders) is REMOVED. These feed **`src/bin/sim_bridge.rs`** (below). |
| `bin/sim_bridge.rs` | **DONE, validated** | The drop-in **`node local_sim_bridge.js` REPLACEMENT** — a std-only Rust binary speaking the EXACT stdin/stdout protocol (`START <json>` / `CHOOSE <side> <choice>` / `FORCELOSE <side>` / `END`; `pN <base64(chunk)>` / `__END__` / `__ERR__ <b64>` frames, base64-per-chunk), so it swaps behind the Python bridge (`local_battle_runner.py`) with ZERO protocol change. **INCREMENTAL** (`gen3_bridge_incremental_replay_v1`): the `Session` holds ONE PERSISTENT `bridge::BridgeSession` across CHOOSE lines — START builds it (framing + first request emitted into its chunks), each CHOOSE `feed_cmd`s ONE command that advances the live battle to its next boundary, and the flush writes the NEW chunk suffix; `__END__` on battle-end (+ persistent-mode reset on `persistent:true`). **O(1)/CHOOSE, O(N)/battle** — this REPLACES the old replay-from-genesis (`re-run over the whole accumulated CHOOSE stream per write`, O(turns³)/battle) which **WEDGED long staller battles under `--use-bridge=rust`** (a late CHOOSE took ~0.6 s+ in-crate → minutes with the full obs pipeline → the `SubprocVecEnv` step-barrier deadlocked). Byte-identical to the genesis path (parity-gated, see the `bridge.rs` row). BYTE-VALIDATED vs the real Node bridge by **`harness/gen_sim_bridge_diff.js`** (spawns BOTH subprocesses, drives choices off the Node bridge's `\|request\|` frames, diffs PER-SIDE chunk sequences — `\|t:\|` normalized, `\|debug\|` dropped): trapping (Arena Trap) + persistent multi-battle-per-child are byte-identical over ~120+ battles. DEFERRED (honest, documented in the binary): **`__RECON__`** (the sim's internal `inputLog` — the port has no byte-identical `input_log`; EXCLUDED from the diff, the Python `_offer_recon` degrades gracefully) + **`resumeReseed`** — now **SUPPORTED** (`gen3_bridge_resume_reseed_v1`) via `BridgeSession::{turn, reseed}`, applied at the START of the divergence turn before its choices commit, ONCE, mirroring the node bridge; a malformed spec is a hard error rather than a silent fall-back to the base seed. The clone-and-branch SNAPSHOT surface is now `BridgeSession::snapshot` (`gen3_bridge_clone_branch_v1` — a derived `Clone`, see the row below); the never-built `Battle::{new,choose,ended,winner,seed,reseed}` convenience methods stay `todo!()` and no production path calls them. So only `__RECON__` still forces `--use-bridge=node`. **FORFEIT PARITY — `gen3_bridge_forfeit_win_v1` (2026-07-31), the bug that WEDGED `--use-bridge=rust` TRAINING.** `FORCELOSE <side>` used to emit a bare `__END__`: the binary had no `>forcelose` engine hook, so it just flushed pending chunks and terminated ("the forfeiting side simply gets no further protocol"). But Node writes `>forcelose` INTO the sim, which runs a real `win(otherSide)` and sends BOTH players `\|` + `\|win\|<name>` before the streams close — so under rust poke-env's `Battle.finished` stayed False FOREVER and the env's next `reset()` waited on a result that could never arrive. Since the training seam forfeits whenever `reset()` lands MID-BATTLE, every episode boundary could wedge: the multi-env soaks logged `🏁 Episode Finished` lines yet completed **zero** PPO iterations (318-byte tfevents, no checkpoints) — read for months as a "multi-env stall". FIX: **`BridgeSession::forfeit(side)`** mirrors the natural-end emission exactly (`bs.log.separator()` + `bs.log.win(&name)` — the same pair `turn::driver`'s `TurnLoopStop::Ended` arm writes — then `emit_log_batch_chunk` flushes the delta as ONE chunk per side); `handle_forcelose` calls it and lets `flush_new_chunks` emit the single `__END__` (calling `end_battle` again would DOUBLE it, since a persistent `end_battle` resets the `ended` guard). Byte-verified vs the Node oracle (`tmp/forcelose_probe.py`: both emit `\|` + `\|win\|P2` to p1 AND p2). Pinned by `bridge.rs::a_forfeit_emits_the_win_line_to_both_sides_not_a_bare_end` (asserts the OBSERVABLE win bytes reach BOTH sides, not merely `ended`); gated end-to-end by `src/utils/bridge/bridge_session_fuzz_test.py --impl rust`, whose every-9th-episode `early_reset_at` forfeit-reset is the reproducer (pre-fix: wedged on the first one; post-fix: 300 episodes / 59 s / 33 forfeits clean). OBSERVATION-NEUTRAL for every committed golden — no golden sends `FORCELOSE`, so the full suite (563 tests) and the e2e md5 are unchanged. SEED-CONVENTION GAP — **RESOLVED** (`gen3_turn0_construction_v1`, 2026-07-22): `sim_bridge` builds via `BridgeSession::new_construct_turn0`, running the REAL turn-0 CONSTRUCTION window from the RAW `>start` seed (the per-mon gender `sample`s + the speed-tie insertChoice/eachEvent shuffles + **Magnet Pull's `onAny` trap shuffles** [via the endTurn `disable_move_event_shuffle`, which interleaves the trap events] + Quick Claw) through the driver machinery — so a **speed-TIED lead** OR an **unspecified-gender** mon is byte-for-bit vs the real Node bridge (the diff harness `gen_sim_bridge_diff.js` no longer skips speed-tied leads; MP/ST ties under `--extra-trappers` are byte-correct). `seed=None` (the training default) has no reference — the construction still runs, harmlessly. The committed goldens' `write_line`/genesis paths keep the POST-construction convention (they seed at the sim's `initSeed` — untouched, byte-identical). |
| `search.rs` | **DONE, validated** | The search KERNELS (`gen3_rust_search_driver_v1`) — the port of `src/utils/bridge/replay_kernels.js`'s search half: the aux PRNG (`aux_rng_from_seed`, mulberry32 over an FNV-1a hash, bit-exact vs JS), `random_choice`/`followup_choice`, `Record::parse` + `session_from_record` + `at_turn_start`/`build_to_turn`/`recorded_turn_choices`, `resolve_turn_exact`/`resolve_turn`, and the omniscient `outcome_of`/`pre_state` renderers. Pure helpers — no stdin/stdout, no node map. See "## The SEARCH SERVER" below for the honest scope of `pre_state` and the one allowlisted divergence. |
| `bin/search_driver.rs` | **DONE, validated** | The drop-in replacement for **BOTH** node offline drivers. (a) `search_driver.js` — the WARM, PERSISTENT clone-and-branch SEARCH server (`open_root` / `expand_many` / `close` over newline-delimited JSON), holding a node-snapshot map so a multi-ply BEAM branches a TREE without re-replaying from turn 1; built on `BridgeSession::snapshot` + `clear_chunks` instead of Node's `State.serializeBattle` + `sendUpdates()` + baseline dance. (b) `replay_driver.js` — the ONE-SHOT REPLAY family `replay` / `reroll` / `reroll_many` (`gen3_rust_replay_driver_v1`), which `utils/bridge/reconstruction.py` drives and on which `better_line` / `lookahead` / `falsify` depend. **DISPATCH is on the KEY**: a request carrying `mode` is answered with a BARE JSON object (no `id`/`ok`, NO trailing newline) and the process EXITS 0 / `{"error"}`+1 exactly as Node does; anything else runs the persistent `{id, cmd}` loop. Gated by `tests/search_driver_test.rs` (5 tests) + `tests/replay_driver_test.rs` (12 tests — four of them `gen3_search_turn1_open_v1`: the FIRST decision of a battle opens on BOTH verb families, which it did not until 2026-08-23) — both node-free — plus the two golden/live differentials `src/rust_sim/harness/search_impl_parity.py` (18873 leaf fields, PASS) and `src/rust_sim/harness/replay_impl_parity.py` (30689 leaf fields, PASS). |

## The callable surface (battle.rs) maps to the existing bridge

`battle.rs` deliberately mirrors the FIVE ways `src/utils/bridge/` already drives
Showdown, so a finished core is a drop-in:

| Bridge today (Node)                     | Rust surface |
|---|---|
| streaming battle (`local_sim_bridge.js`) | `BattleStream::write_line` |
| mid-battle RNG swap (counterfactual/search) | `Battle::reseed` |
| clone-and-branch (`State.serialize…`)    | `BridgeSession::snapshot` (a derived `Clone`; PRNG continuity is automatic) |
| search server (`search_driver.js`)       | `src/bin/search_driver.rs` over `src/search.rs` (`gen3_rust_search_driver_v1`) |
| offline replay / re-roll (`replay_driver.js`) | the SAME `src/bin/search_driver.rs`, one-shot `mode` verbs (`gen3_rust_replay_driver_v1`) — node splits the two families across two scripts, the port serves both from one binary, which is what `sim_bridge_bin.py::search_driver_spawn_argv` assumes |
| damage oracle (`damage_probe.js`)        | `Battle::new` + state accessors (TODO) |
| team pack / validate                     | out of core scope — keep as a thin shim |

When you implement the engine, keep these signatures stable; the bridge contract
is the spec.

## Clone-and-branch: the SNAPSHOT primitive (`gen3_bridge_clone_branch_v1`)

The last surface `--use-bridge=rust` was missing for the prober's SEARCH path
(`better-line`'s CRN-anchored beam, which branches a paused mid-battle state by CLONING it).
The Node search server does this with `State.serializeBattle`/`deserializeBattle`; the port's
answer is **`BridgeSession::snapshot()` — a derived `Clone`**, and the reason a byte format is
NOT needed is structural: Showdown needs one because its battle graph is full of cyclic object
references, whereas everything under `BridgeSession` (battle state, PRNG, driver phase, chunk
stream, boundary progress) is **plain owned data** — no `Rc`/`RefCell`/`Box<dyn>`/closures/raw
pointers/lifetimes — so `#[derive(Clone)]` already gives a DEEP, independent battle. The
snapshot never crosses a process boundary (neither does Node's), so there is nothing to encode,
nothing to keep in sync, and no re-parse cost. The old `Battle::serialize`/`deserialize` +
`BattleSnapshot` stubs are DELETED, not implemented.

PRNG continuity is automatic (the whole dice state is `Prng`'s backend seed), and the `&Dex` is
threaded per method rather than owned, so a snapshot copies a couple of teams' worth of state
rather than ~16 MB. The one thing a clone shares with its parent is the process-global
`POKESIM_PRNG_TRACE` draw counter — diagnostics, not battle state, so branches interleave their
trace INDICES and nothing else.

The public search API on `BridgeSession` (all reads or copies — nothing here advances the
battle or draws a number, so it cannot perturb the production bridge):

| method | contract |
|---|---|
| `snapshot() -> BridgeSession` | the deep clone; shares NOTHING with the parent |
| `clear_chunks()` | reset the chunk stream so a branch's `chunks()` is only ITS suffix. Deliberately does NOT touch `prev_log_len` — that is a cursor into the BATTLE LOG, and resetting it would re-emit the whole battle into the next chunk |
| `request_kind(side) -> Option<RequestState>` | Showdown's `side.requestState`. `Move` / `Switch` (a forced replacement) / `Wait`; `None` when no boundary is open |
| `is_choice_done(side) -> bool` | `side.isChoiceDone()`. True also for a `Wait` side and when no boundary is open; FALSE across a REJECT (a reject re-issues the request and holds the boundary open) |
| `active_request_json(side) -> Option<&str>` | `side.activeRequest` — the EXACT bytes that went on the wire, stored (not rebuilt) at every emit site, INCLUDING the `trapped:true`+`update:true` re-request a hidden-trap reject re-issues. Cleared when the boundary closes / the battle ends |
| `battle_state() -> Option<&BattleState>` | the omniscient referee readout (HP/status/boosts/field/seed) |
| `winner() -> Option<usize>` | the winner once ended, from BOTH end paths (the driver's win check and `forfeit`, which ends via the protocol so the driver's phase never becomes `Ended`). `None` = still playing OR a gen-3 TIE — pair with `is_ended()` |

**Gate: `tests/bridge_clone_branch_test.rs`** (7 tests) on a real seeded gen3 battle — two
snapshots advanced with the SAME choices emit byte-identical chunks (and so does the original);
two advanced with DIFFERENT choices leave the parent's turn / chunk stream / outstanding request
/ PRNG seed bit-for-bit unchanged; a `reseed` on a clone leaves the parent matching an
independently-built control; `clear_chunks` yields exactly the tail the un-cleared branch
appends; and the boundary accessors track a partially-answered request, a REJECTED switch, and
game-end. Every test carries a non-vacuity guard (the fixture is a real mid-battle `move`
boundary, the two branch choices genuinely diverge, the reseed genuinely changed the dice) —
without those a snapshot of a finished battle would pass while proving nothing. Its lead pair is
a bulky Blissey MIRROR on purpose: a lead fainting inside the prefix would turn the pause into a
forced replacement and silently stop testing the branch point. FAULT-INJECTION PROVEN (a
`clear_chunks` that also reset the log cursor, and a reject that failed to record its re-issued
request, each fail exactly one named test).

**Not built here:** `Battle::{new,choose,ended,winner,seed,reseed}`, which stay `todo!()` because
every production path drives the engine through `FullBattleDriver` instead. The SECOND half — the
driver that consumes this API — is below.

## The SEARCH SERVER: `src/search.rs` + `src/bin/search_driver.rs` (`gen3_rust_search_driver_v1`)

The Rust half of the clone-and-branch search layer: a byte-compatible drop-in for
`src/utils/bridge/search_driver.js`, so `utils/bridge/search_session.py` (the prober's
`better_line` beam, and the search teacher) can swap `node search_driver.js` for this binary
with ZERO protocol change.

**The same binary ALSO serves the offline REPLAY family** — see
"## The REPLAY family" below (`gen3_rust_replay_driver_v1`). The two protocols share this
binary and every kernel; they differ only in dispatch (`mode` = one-shot, `cmd` = persistent).

- **`src/search.rs`** — the port of `src/utils/bridge/replay_kernels.js`'s search-relevant
  kernels. Pure helpers, no stdin/stdout, no node map: `aux_rng_from_seed` / `pick_uniform` /
  `random_choice` / `followup_choice`, `Record::parse` + `session_from_record` + `at_turn_start`
  + `build_to_turn` + `recorded_turn_choices`, `resolve_turn_exact` / `resolve_turn`, and the
  omniscient `outcome_of` / `pre_state` renderers. (Plus the replay half's `recorded_queues` /
  `TurnSource` / `resolve_turn_sourced` / `turn_log` / `log_len` / `write_cmd`.)
- **`src/bin/search_driver.rs`** — the persistent stdin/stdout JSON server (`open_root` /
  `expand_many` / `close`), the node-snapshot map, and the monotonic `n0, n1, …` ids. A panic is
  caught and returned as `{id, ok:false, error}` (the `sim_bridge` pattern) — a search server
  that dies mid-beam would strand the caller.

**WHY THE PORT IS STRUCTURALLY SIMPLER.** Node has to clone with `State.serializeBattle`, then
per arm `deserializeBattle` → `restart()` a fresh `BattleStream` with the same `send` wiring →
`sendUpdates()` to flush the whole re-emitted historical log → mark a BASELINE index so only this
ply's suffix is returned (the re-emitted prefix is useless: requests are sent out of band and
never stored in `battle.log`, so a materializer parsing it would never see the team). The port
does `snapshot()` + `clear_chunks()` and the branch's chunks ARE its suffix by construction. Same
wire contract, no byte format, no baseline.

**THE ONE STRUCTURAL DIFFERENCE, and why it is unobservable.** Node writes BOTH sides' choices
then ticks; `feed_cmd` advances synchronously per call. Showdown's `BattleStream._write` is itself
synchronous (`_writeLine` then `battle.sendUpdates()` inline) and the `await tick()` only drains
the async chunk collectors, so both engines resolve at the moment the LAST needed side answers; a
boundary needing both sides does not resolve on the first feed; and a REJECT re-issues the request
and leaves the boundary open in both (`is_choice_done` stays false). Verified end-to-end by the
golden below.

**TWO port-specific decisions worth knowing:**
- **`"default"`** is resolved by the ONE `Side.autoChoose()` port, `bridge::resolve_auto_choice`
  (a forced switch → the first non-fainted bench slot; a move request → the first non-disabled
  request slot; `Move(0)` under Struggle **or a move-lock**). `search.rs::resolve_default` only
  renders that back to a wire token and records the literal `"default"` in `choices_used`,
  matching Node. Exercised by the golden (2 arms).
  **It used to hold its own copy of the logic, and that copy was wrong for a MOVE-LOCKED mon** —
  it scanned the four real moveslots and could answer `move 2`, where the sim's single-entry
  locked request means slot 1 and `choice_is_legal` accepts only index 0. `default` is no longer
  search-only either: `parse_choice` accepts it (and `auto`/`pass`/`skip`) on the production
  wire, because poke-env really sends those tokens — see the bridge row's CHOOSE-path note.
- **`RESOLVE_GUARD`** mirrors Node's `if (guard++ > 40)` — the value BEFORE the increment, so 41
  iterations run. A naive `guard += 1; if guard > 40` cuts one short and can flip a `stuck`
  verdict; the count is pinned exactly (`used[1].len() == 41`).

**GATE 1 — `tests/search_driver_test.rs`** (5 tests, no node needed): the aux-RNG stream against
four draw tables produced by `replay_kernels.js`'s OWN `auxRngFromSeed` under node (EXACT f64
equality — `u32 / 2^32` is representable, so a tolerance would only hide a divergence) plus
`pickUniform`'s index and its one-draw cost; `random_choice` determinism per seed, non-determinism
across seeds, and legality of every pick against an independently-derived option set; CLONE
INDEPENDENCE (two arms diverge, the parent's turn / chunk stream / outstanding request / PRNG seed
are bit-for-bit unchanged, and re-expanding reproduces an arm byte-for-byte); and the `stuck`
guard's exact iteration count. Every test carries a non-vacuity guard — the fixture must really be
a mid-battle `move` boundary with a live bench, the two arms must really diverge, the wedge must
really have wedged.

**GATE 2 — `src/rust_sim/harness/search_impl_parity.py`** (scratch, needs node + a captured record): replays
`tmp/search_golden_node.json` — the NODE `search_driver.js` wire output over 6 decision points
across 2 real gen3ou battles, 54 arms (12 exercising multi-round forced-switch follow-ups) plus
depth-2 expansions — through the built binary over raw stdin/stdout JSON, and diffs every field.
**RESULT: PASS, 18873 leaf fields matched.** It normalizes ONLY `|t:|<epoch>` (the port's one
documented emission exception) and prints its allowlist + coverage notes on every run.
(**Re-measured 2026-08-23 on SEVEN freshly generated goldens: PASS on every one**, 12 cases /
120 arms / ~37.6k leaf fields each — roughly 2× the recorded golden, because the generator now
samples turn 1 as well. The live allowlist takes **0 hits** on all seven.)
FAULT-INJECTION PROVEN: a one-line off-by-one in `pick_uniform` produces 2716 divergences (and the
illegal-move allowlist below correctly REFUSES to reconcile, hits 0 → 1 → 0).

⚠️ **STATUS 2026-08-23 — three corrections, and that PASS is not reproducible as written.**
1. **Both parity harnesses were UN-RUNNABLE.** `ROOT = parents[1]` was correct while they lived in
   `<root>/tmp/`; the move to `src/rust_sim/harness/` (`ede4c79`) left the index behind, so every
   default path resolved under `src/rust_sim/` and did not exist. Same defect in
   `search_golden.py`'s `DRIVER`. All three now use `parents[3]`. A gate nobody can start is
   indistinguishable from a gate that passes.
2. **The golden now samples TURN 1** (`search_golden.py`, `turns = {1, 2, 5, 9}`). It sampled only
   2/5/9, which is exactly why the cross-impl gate never saw `gen3_search_turn1_open_v1`. On a
   turn-1 golden the pre-fix binary reports **157** divergences and the fixed one **1**.
3. ~~On a FRESHLY generated golden both gates currently report pre-existing divergences.~~
   **CLOSED 2026-08-23 (`gen3_fresh_golden_parity_triage_v1`) — BOTH gates are green on freshly
   generated goldens, and the count that reaches this table is now 0.** Do not re-derive a plan
   from the old wording, which guessed at TWO candidates and named neither correctly. The
   divergences were real, pre-existing, and reproduced on the pre-fix binary; SEVEN fresh goldens
   (21 real gen3ou battles, `search_golden.py 3` each) resolved them into **four classes, all of
   them rust BUGS against the node oracle, all now fixed**:

   | class | instances | who was wrong | reach | fix |
   |---|---|---|---|---|
   | Return / Frustration numeric-BP alias missing from the `\|request\|` | 8 fields / golden E | RUST | **the BRIDGE too** — node and the live server both emit it, so `--use-bridge=rust` was the odd transport out (poke-env's `Move.retrieve_id` collapses it, which is why nothing downstream ever noticed) | `gen3_happiness_bp_request_alias_v1` |
   | `pre_state.*.volatiles[len]` — the `substitutebroken` volatile unmodeled | 6 / golden D | RUST | offline drivers only (`pre_state` has no consumer) | `gen3_substitute_broken_volatile_v1` |
   | a SINGLE-ENTRY request (forced Struggle / move lock) silently ACCEPTED `move 2` | 6 / golden G | RUST | **the BRIDGE too** — an `\|error\|` frame and a different committed choice | `gen3_single_entry_request_slot_reject_v1` |
   | `outcome.pN.active_status` — a fire KO thawed the corpse | 1 / golden A | RUST | a STATE divergence in the bridge too, though `0 fnt` hides it on the wire | `gen3_fire_thaw_ko_keeps_status_v1` |

   The **typed-Hidden-Power `|error|` display name** the old text guessed at is a REAL fifth bug
   (`Side.chooseMove` names `dex.moves.get(moveid).name`, i.e. the BARE id — never the request's
   suffixed display), but it did **not** appear in any of the seven goldens: it needs a *disabled*
   typed HP fed as an explicit arm. It is fixed and unit-gated anyway
   (`gen3_reject_message_bare_move_name_v1`), on the training-path argument — poke-env sees
   `|error|` frames. **The "29" and the "1" were never stable counts**; a golden is three random
   battles, and the per-golden divergence count ran **1 / 0 / 0 / 6 / 8 / 0 / 6** across A-G — three
   of the seven would have read as a green gate, and the Struggle class turned up only on the
   SEVENTH. **Run each gate on at least two fresh seeds before calling it green.**

**THE CHOICE-REJECT DIVERGENCE IS CLOSED — the only live allowlist entry is `.error` TEXT**
(Node returns a JS `e.stack`, the port a plain message; the ok/`ok:false` VERDICT is still compared
strictly). When an arm feeds an explicit move the request marks `disabled` (the golden hits a
Choice-locked Aerodactyl), Node emits `|error|[Unavailable choice] Can't move: X's Y is disabled` +
a re-request **to that side only**, whose disabled slot gains `"disabledSource":""`
(`Side.chooseMove`'s `updateRequestForPokemon`, which is also what makes the error `[Unavailable]`
rather than `[Invalid]`) — and `bridge.rs` now emits exactly that
(`gen3_choice_reject_framing_v1`: `serialize_active_with_disabled_source`, the
`[Invalid]`/`[Unavailable]` split on whether there is a request to re-issue, gated by
`tests/bridge_choice_reject_test.rs`).

⚠️ **The old entry here claimed the port "emits no `|error|` … on a path poke-env never takes".
BOTH halves were false, and it is retracted — do not re-derive a plan from it.** The framing half
was closed by `gen3_choice_reject_framing_v1` and *this note survived its own fix*. The "never
takes" half was falsified by poke-env taking it and killing **two production launches at ~8
minutes** — root-caused as `gen3_locked_choice_never_rejected_v1` (a move-LOCKED mon gets a
single-entry `trapped:true` request, and `classify_reject` never consulted `move_locked()`, so the
port could REFUSE the only move offered — it contradicted ITSELF; fixed with a `move_locked()`
early-out beside `must_struggle`, gated by
`bridge_choice_reject_test::a_move_locked_mon_is_never_rejected_for_its_only_offered_move`). The
existing fuzz could not catch it by construction: it drives masked-LEGAL tokens, and there the
token IS masked-legal because the mask is built FROM the request. **The durable lesson is that an
allowlist entry can outlive its own fix and then mislead every reader after** — see the root
`CLAUDE.md`. Verify an allowlist claim against the harness (`src/rust_sim/harness/search_impl_parity.py::ALLOWLIST`)
and the code, never against this prose alone.

**HONEST SCOPE — `pre_state`.** It mirrors Node's `preState` and has NO consumer today. `turn`,
`weather`, the per-mon species/hp/maxhp/status/fainted/position, boosts, item, ability, the
bare-`hiddenpower` moveSlot ids + PP, and the three gen-3 side conditions are faithful reads.
Two fields are RECONSTRUCTIONS: `pseudoWeather` is derived from `BattleState::sleep_clause` (the
port has no pseudo-weather map; gen-3 has no pseudo-weather MOVE, so the clause rules are the only
real entries — VERIFIED, 6/6 golden cases), and `volatiles` is rebuilt from the port's typed
volatile fields. The golden gives that mapping exactly ONE piece of coverage, and it was a real
find: the port's **`duration: 1` single-turn fields** (`focuspunch`, `pursuit`, `protect`,
`flinch`, `beatup`, the Counter/Mirror-Coat record, `endure`, `snatch`) are cleared at the NEXT
turn's TOP (`turn::helpers::clear_flinch`) where Showdown removes them at the residual, so they
are STALE-SET at a move-request boundary — reporting `focuspunch` diverged 2 of the golden's 12
`pre_state`s, and the group is excluded.

**Exactly ONE volatile NAME is positively verified: `substitutebroken`** — `gen3_substitute_broken_volatile_v1`,
and it took a FRESHLY generated `replay_impl_parity` golden to find it, because the recorded
golden's twelve `pre_state`s are all EMPTY. gen3 inherits gen4's Substitute, which pairs the
removal with `addVolatile('substitutebroken')`; the condition has no duration and no gen3 reader,
so it sits on the mon until `clearVolatile`. It is mechanically INERT in gen 3 and is modeled
purely so this readout can name it. **Every other name (`choicelock`, `perishsong`, `twoturnmove`,
…) is still UNVERIFIED** — do not read a green parity run as evidence that one is spelled or timed
right; `replay_impl_parity` prints a `pre_state:nonempty-volatiles` count on every run precisely so
an all-empty record set cannot be mistaken for coverage.

**STILL NEEDED before this can replace node in `better_line`** (all Python-side, owned elsewhere):
`utils/bridge/search_session.py` must gain an impl switch that spawns this binary, and the
`search_clone_parity_fuzz_test` must be run with it. Note the search teacher's OTHER blocker is
unchanged and unrelated: `--search-teacher` needs the sim's OWN byte-identical `input_log`, which
the rust record renders replay-EQUIVALENT rather than byte-identical.

## The REPLAY family: the one-shot `replay` / `reroll` / `reroll_many` verbs (`gen3_rust_replay_driver_v1`)

The SECOND half of the offline layer, served by the SAME `src/bin/search_driver.rs` binary. It
closes the gap the search half opened: `utils/bridge/reconstruction.py::_run_driver` routes BOTH
verb families through `sim_bridge_bin.search_driver_spawn_argv`, so under `impl="rust"` a
`replay_battle` / `reroll_turn` / `reroll_many` call reached a binary that answered
`unknown cmd` — which broke `better_line` (it calls `replay_battle`), `lookahead` and `falsify`
even though the search half worked.

**PROTOCOL.** `{mode: "replay"|"reroll"|"reroll_many", …}` on stdin → ONE bare JSON object on
stdout with **no trailing newline** → exit **0**; on failure `{"error": …}` → exit **1**. That is
`replay_driver.js`'s whole lifecycle, and the EXIT CODE is the load-bearing half: `_run_driver`
branches on it BEFORE it parses stdout, so the message is informational. Dispatch is on the `mode`
KEY; a `cmd` request still runs the persistent search loop, in the same process, unchanged.

**KERNEL REUSE, not a second copy.** The only kernel the search half never needed is
`recorded_queues` — the replay family's `"recorded"` action source is a QUEUE, not the search
path's single-shot latch, because a live turn's command stream can contain a REFUSED choice
followed by its correction (the maybe-trapped probe). Replaying that faithfully under a fresh seed
requires pulling the NEXT recorded command on a refusal; a single-shot source would fall to the
follow-up POLICY and invent a pick the original battle never made. `resolve_turn` was refactored
onto a shared `resolve_turn_sourced` over a `TurnSource` enum (`Silent` / `Random` / `Explicit` /
`Queue`) so both families run ONE resolution loop — the guard arithmetic and the routing invariant
exist once. `resolve_turn`'s own behaviour is byte-identical (`ActionSpec::Recorded` lowers to
`Silent`).

**`turn_log` — the one field the search half does not produce.** It is defined as Node's
`b.log.slice(logStart)`, and Showdown's `battle.log` is NOT the port's line list: an `addSplit`
effect occupies THREE entries there (`|split|pN`, the SECRET line, the SHARED line), and EVERY
HP-bearing line is one, because `Pokemon.getHealth()` returns a `{side, secret, shared}` object
for the `Battle.add` split path — including `0 fnt` and including a `reportExactHP` format, where
`shared` merely equals `secret`. `bridge.rs::split_log_lines` reconstructs the triples from the
port's single omniscient line using EXACTLY the predicate set the production per-side fold
(`derive_side`) uses, so the two can never disagree about which lines are split. The owner-only
pair (`|-ability|…|[silent]`, the Intimidate `|-hint|`) gets an EMPTY shared entry, matching
`addSplit(side, secret)` with no `shared`. VERIFIED byte-for-byte on 133 arms.

**ONE REAL PORT BUG the widened coverage caught: the `fnt` status token.** `Battle.checkFainted`
writes `status = 'fnt'` for a FAINTED ACTIVE mon; `BattleActions.switchIn` clears it again when
the corpse is swapped off (`if (oldActive.fainted) oldActive.status = ''`); and `runAction` runs
`faintMessages(); if (this.ended) return true;` BEFORE `checkFainted()`, so the DECIDING faint
never gets it. The port models the state effect as `status = None` (that clear is what stops
paralysis ×0.25 applying to the replacement sort — `gen3_fnt_clears_status_v1`), so the token has
to be re-derived at render time: `search.rs::slot_status_token` shows `fnt` only for a fainted mon
in the ACTIVE slot of a battle that has NOT ended. The search golden never reaches this (its arms
all end at a live move boundary and it never reports a finished battle), which is why it took
`replay` + re-rolls of LATE turns to surface. All three branches are node-verified.

**GATE 1 — `tests/replay_driver_test.rs`** (12 tests, no node): the one-shot dispatch (bare object,
no trailing newline, exit 0 / exit 1 with `{"error"}`, unknown mode, invalid turn); that the
PERSISTENT protocol is untouched and the process stays alive after a `cmd` request (the regression
the dispatch could plausibly cause); the `recorded_queues` REFUSAL-PULL together with its contrast
case (a single-shot source must NOT re-send); and `recorded_queues`' own per-side split /
`forcelose` stop / cap. Its record is built in-test by playing the fixture board to game-end
through `BridgeSession::new_construct_turn0` — the same constructor `session_from_record` uses, so
the recorded stream replays bit-for-bit.

**Four of the twelve are `gen3_search_turn1_open_v1`** — the FIRST decision of a battle. They pin
the predicate (`at_turn_start` true for `t == 1` at a pre-commit boundary, false for every other
`t`, and unchanged at `t >= 2`), that `build_to_turn(…, 1)` applies NO commands, and that turn 1
opens through the real binary on BOTH verb families (one-shot `reroll` and persistent `open_root`).
Before the fix, `at_turn_start` compared `BattleState::turn` — still `0` at that boundary, because
the driver increments at `commitChoices` for turn 1 and EAGERLY at every later turn end — so
`build_to_turn` walked the whole command log and reported "battle never reached the start of turn
1". Node opened it fine, making this a silent impl-specific hole over ~3.35% of move decisions.

**GATE 2 — `src/rust_sim/harness/replay_impl_parity.py`** (scratch, needs node + `tmp/search_golden_node.json`'s
records): drives `node replay_driver.js` and the rust binary LIVE on the identical request and
diffs every field. Live rather than golden-captured on purpose — the requests are cheap, so a
stored golden would only add staleness. **RESULT: PASS — 76 cases (54 success + 22 error-path),
136 arms, 30689 leaf fields.** It normalizes ONLY `|t:|<epoch>` and prints its allowlist, its hit
counts and a COVERAGE table on every run.
(**Re-measured 2026-08-23 on SEVEN freshly generated record sets: PASS on every one**, 111 cases /
~204 arms / ~45-47k leaf fields each. The only allowlist arm — error message TEXT — takes 30 hits
per run and the verdict, exit code and error CLASS stay strict. Coverage varies by record set and
is REPORTED, never assumed: `pre_state:nonempty-volatiles` was 0 on three of the seven and 26 on
another, and `arm:stuck` / the Intimidate `|-hint|` still miss on most.)

**COVERAGE the search golden was missing, now exercised** (reported by the harness itself):
2 arms that END the battle, 1 `stuck` arm, 133 arms whose `turn_log` carries `|split|` triples,
17 with the owner-only empty-shared form, and 9 distinct error classes across BOTH families
(`invalid turn` ×8, `battle never reached the start of turn N` ×3, `replay`'s not-ended ×4,
`unknown mode` ×2, a seedless arm ×2, `unknown cmd`, `unknown node`, `recorded_exact on a non-root
node`). The `stuck` arm needs a deliberately TRUNCATED record — a well-formed record's turns are
always complete, so `resolveTurnExact` can never run out mid-turn on one; the harness truncates a
copy, which also drives `replay`'s not-ended path. STILL NOT EXERCISED and printed as such: a
NON-EMPTY `pre_state` volatile list, and an Intimidate `|-hint|` inside a re-rolled turn (the one
`split_log_lines` case whose owner attribution is a heuristic).

**THE CHOICE-REJECT DIVERGENCE IS CLOSED here too — the only live allowlist entry is `.error`
TEXT** (the VERDICT, the process EXIT CODE that `reconstruction.py` actually branches on, and the
error CLASS are all compared STRICTLY). An arm that feeds a choice the sim refuses — here
`switch N` into a FAINTED slot — used to diverge in the framing (`|error|[Invalid choice]` to the
offending side only vs no `|error|` and a re-opened boundary to BOTH sides) and, downstream of
that, in `choices_used`. `bridge.rs` now models every reject class, not just the trapped-SWITCH one
(`gen3_choice_reject_framing_v1`), so neither entry remains in
`src/rust_sim/harness/replay_impl_parity.py::ALLOWLIST`.

⚠️ **Retracted, same as the search half's note — do not re-derive a plan from the old text.** It
described the port as "modelling only the trapped-SWITCH reject" on a path poke-env never takes;
that path killed two production launches at ~8 minutes
(`gen3_locked_choice_never_rejected_v1`). Read the search half's note above for the root cause and
the lesson, and verify against the harness + the code rather than this prose.

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
  regression pins. (`liquidooze` is now in **`MODELED_ABILITIES`** — `gen3_liquid_ooze_v1`: the
  drain/leech-seed heal REVERSAL is MODELED bit-for-bit [`apply_drain` / `apply_leech_seed` turn the
  would-be heal into damage on the healer, DRAW-FREE], no longer a `NOOP_ABILITIES`/fail-loud
  exclusion; proven by the DEDICATED `liquidooze_test.rs` golden + the LO1-LO3 pins.) The headline assertion tallies are CLEAN-ONLY: the per-decision loop
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
| PA1 **Pressure × `allyTeam` PP-deduction** (`gen3_pressure_allyteam_v1`, the **batch-2 e2e admission** bug e2e_182) — an `allyTeam` move (Aromatherapy / Heal Bell) under a **Pressure** foe deducts ONE PP, not two (the Pressure extra fires only when the foe is in the move's `pressureTargets` — a FOE-directed target; `allyTeam`/`self`/`allySide`/`allies` never put the foe there — but `foeSide`/Spikes DOES). WRONG (pre-fix `!targets_self`): Aromatherapy 8→6 under Pressure → drained early → the port rejects a legit late Aromatherapy as out-of-PP → the script shifts (e2e_182 decision-count + state desync). Pins Aromatherapy −1 (8→7) + a ThunderWave −2 (32→30) control, both DRAW-FREE seeds | `pressure_does_not_add_pp_for_an_allyteam_move` | STATE (PP 8→7 vs 8→6) + SEED (both draw-free) |
| PA2 **Pressure × `foeSide` PP-deduction** (`gen3_pressure_foeside_v1`, the per-side/request byte-fuzz round-4 find) — a `foeSide` move (Spikes, the ONLY gen-3 `foeSide` move) under a **Pressure** foe deducts **TWO** PP (`foeSide` DOES put the foe in `pressureTargets`). SIM-PROBE-CONFIRMED: Skarmory Spikes vs a Pressure Suicune = pp 30/32 (−2), vs non-Pressure = 31/32 (−1); DRAW-FREE so the post-turn seed matches. WRONG (the pre-fix predicate excluded `foeSide` alongside `allyTeam`): 32→31 — INVISIBLE to the omniscient byte fuzzer (no PP in the `\|...\|` stream), only the request-JSON `pp` field diverges. Does NOT touch PA1's `allyTeam` case | `pressure_adds_pp_for_a_foeside_spikes_move` | STATE (Spikes pp 32→30) + SEED (draw-free) |
| PA3 **Pressure × a NON-GHOST CURSE PP-deduction** (`gen3_pressure_allyteam_v1`, the omniscient byte-fuzz round-4 DEEP-SEED find — the ROOT of 5_6 + 3_24 + the cg dec-30 cascade) — gen-3 `curse.onModifyMove` RE-TARGETS a non-Ghost user's Curse to `self` (`nonGhostTarget`) but the STATIC dex `target` is `"normal"`, so the Pressure foe wrongly landed in `pressureTargets` → **−2** PP instead of **−1**. WRONG (the pre-fix tuple reading `m.target`): a Swampert's 16 Curse PP drained 2/turn → exhausted ~1 cycle early → forced Struggle turns the sim still Curses (protocol `\|move\|…\|Struggle` vs `\|move\|…\|Curse`; Curse count SIM 16× / PORT 10×) → the deep seed desync + cascade. FIX: the move-resolution tuple feeds the RUNTIME-effective Curse target (`is_nonghost_curse → "self"`) to BOTH `targets_self` and `pressure_targets_foe`. DRAW-FREE (PP deduct consumes no PRNG) | `pressure_does_not_add_pp_for_a_nonghost_curse` | STATE (Curse pp 16→15 not →14; dec7 pp 8) + SEED (the dec7 post-decision seed) |
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
| RS1 REST into a SLEEP-IMMUNE ability (`gen3_rest_sleep_immune_v1`, the A/B repro rmry3vbgm_ab_1_1/ab_3_21 seed@15/62) — a damaged INSOMNIA / VITAL SPIRIT mon's Rest is BLOCKED by its own `onSetStatus` BEFORE `slp.onStart`, so the sleep `random(2,6)` NEVER draws (no sleep, no heal); WRONG (pre-fix) `run_rest` drew it + slept + healed → +1 draw. The Insomnia-Rest seed == a draw-free Amnesia control's, DIFFERS from a sleep-able Rest | `rest_into_a_sleep_immune_ability_fails_draw_free` | STATE (awake, un-healed) + SEED (draw-free == Amnesia, ≠ sleep-able Rest) |
| MD1 DISABLE of a MIMIC-overwritten slot (`gen3_mimic_disable_self_overwrite_v1`, rmry3vbgm_ab_6_16 seed@46) — Showdown's Disable stores the used move ID (`"mimic"`), gone from any slot after Mimic overwrote its OWN slot → `onStart` `!hasMove` returns false → Disable FAILS (draw-CONSUMED — the `random(2,6)` still drew). The port stored Disable by SLOT index → wrongly disabled the COPIED move. New `MonState::last_move_was_self_overwrite` flag (set by the Mimic success block, reset every `run_move`) drives the disable-arm guard | `disable_of_a_mimic_overwritten_slot_fails_leaving_the_copied_move_usable` | STATE (the copied move deals damage next turn) + SEED |
| YW1 YAWN RE-CAST into a still-pending yawn (`gen3_yawn_recast_v1`, rmry3ytkn_ab_6_22 seed@44) — Showdown's `addVolatile('yawn')` returns false (no `onRestart`) → the re-cast FAILS ([still]+`-fail`) and the EXISTING yawn is UNCHANGED (resolves on its ORIGINAL schedule); WRONG (pre-fix) the port RE-SET the duration to 2 → the sleep `random(2,6)` slipped ONE turn late (a draw-ORDER desync). The already-yawned guard sits in the yawn cast arm | `yawn_recast_into_a_pending_yawn_does_not_reset_the_duration` | STATE (foe asleep at the resolve turn) + SEED |

(RS1/MD1/YW1 capture the draw-count/first-mover tail fixes the `rmry3vbgm`/`rmry3ytkn` A/B round
surfaced — ground truth `harness/probe_dc_batch_regression_rng.js`. The one un-fixed repro of that
round, **rmry3vbgm_ab_4_6** [firstmover@0], is the project-wide turn-0 CONSTRUCTION-window deferral:
a QuickClaw holder [Shuppet] whose turn-0 `endTurn` `randomChance(1,5)` HIT gets the gen3
`getActionSpeed` speed=65535 override on turn 1, moving first despite a slower raw Speed — but the
offline `ab_replay` path [`start_with_switchins` from the POST-construction `initSeed`] does not
model the construction endTurn QC roll, so it cannot know it. Seed-identical [Pikachu's Agility is a
draw-free setup move, so p1-first vs p2-first draw the same stream]; the bridge path
[`BridgeSession::new_construct_turn0`] DOES model it. Modeling it offline would ripple the
pre-first-decision seed convention every committed golden depends on — the documented deferral.)

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

**The gen3ou-POOL byte-fuzz Hidden-Power / Intimidate pins (the 14h `sweep_fuzz14_omni_ou` finds, all
revert-verified STATE pins).** (1) **`taunt_does_not_block_a_bare_hidden_power`**
(`gen3_iv_derived_hidden_power_bp_v1`, the LEGALITY sibling of the round-12 IV-derived-HP fix; repro
ab_233_8): the BARE `hiddenpower` (num 237, data BP 0 → derived Status) is a REAL damaging move, so
`MoveData::blocked_by_taunt()` gates on `!self.id.starts_with("hiddenpower")` — a Taunted mon keeps its
HP selectable (else `choice_is_legal` rejects the legal HP → the choice accumulator runs the WRONG move,
the ab_233_8 over-KO). The dex lockstep test (`dex/mod.rs`) already pins the classifier. (2)
**`mimic_of_hidden_power_uses_the_mimickers_own_type`** (`gen3_mimic_hidden_power_type_v1`; repro
ab_777_3): gen-3 Hidden Power's TYPE is a property of the USER (Showdown stores the moveslot id BARE),
so `turn.rs`'s Mimic success block canonicalizes a copied `hiddenpower*` id to the BARE `hiddenpower` —
the mimicker re-derives the type from its OWN IVs (else it keeps the copied mon's TYPED HP, e.g. Grass
0.25x where the sim's neutral Dark lands). (3) **`intimidate_entrant_does_not_drop_a_forced_replacement_foe`**
(`gen3_intimidate_forced_replacement_v1`; repro ab_1381_0, DEC 51): a deferred switch-in Intimidate must
NOT drop the Atk of a foe that force-replaced its intended target between the switch action and the
deferred `RunSwitch` (the Pursuit→Destiny-Bond→replacement composition). `execute_switch` captures the
foe uid at switch-in **only for a VOLUNTARY switch** (`switchin_foe_uid`), and `event::intimidate_on_start`
suppresses the drop when that captured original is now FAINTED — matching Showdown, whose entrant
Intimidate resolves INLINE against the fainted original (`adjacentFoes()` empty → the "not activated"
hint). The `is_voluntary` gate + the original-FAINTED narrowing keep a DOUBLE FORCED replacement
(mutual Self-Destruct) and a live-foe double switch on the pre-fix path (both DO drop the co-/new foe —
e2e + byte-fuzz-corpus-verified). STATE-only fix; the residual PROTOCOL byte-ORDERING of the deferred
hint (the port runs the entrant RunSwitch past the turn boundary vs the sim inline) is a pre-existing
deferred-RunSwitch limitation, unmasked once the STATE matches — the spec's rejected higher-risk reorder,
out of scope. All three are OBSERVATION-NEUTRAL for the committed suites (e2e md5
`3155eb796cb4bf453c6053d769ba98e5` UNCHANGED, all 508 tests green).

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
- **CONTACT-PROC status carries `[from] ability: <A>|[of] <holder>`** (Static/Poison Point/Flame Body/Effect
  Spore) — a CORRECTED earlier claim (this note used to say the form was BARE; the live sim capture refuted
  it, `gen3_omniscient_byte_fuzz_v1`, golden `|-status|p1a: Metagross|par|[from] ability: Effect Spore|[of]
  p2a: Breloom`). The sim's `source.trySetStatus(status, target)` runs INSIDE the ability's `onDamagingHit`,
  so `setStatus`'s null `sourceEffect` falls back to `this.battle.effect` = the ABILITY → the status
  `onStart`'s `sourceEffect.effectType === 'Ability'` gate is TRUE → the `[from] ability` form. `apply_contact_proc`
  now threads the ability name through `try_set_status_impl`'s `ability_reveal` (the generalized Synchronize
  path). Draws unchanged (emission-only); all seed goldens byte-identical.
- **Volt/Water Absorb `-heal` (not `-immune`) when the heal LANDS** — `apply_absorb_heal` now reports
  whether it healed; a below-full holder shows `-heal|…|[from] ability: Volt Absorb|[of] <user>`, only a
  full-HP holder shows `-immune|…|[from] ability` (the F3 capture had only exercised the full-HP case).
- **Beat Up per-strike `-activate|<user>|move: Beat Up|[of] <ally>`** — gen3customgame EMITS it (the mod's
  `condition.onModifySpA`, gated on `!beatupnicknamesmod` — a gen3 Standard rule present in gen3ou, absent
  in customgame → aligned with `sleep_clause`). The task's "gen3 does not emit it" claim was WRONG; the
  omniscient stream + resolved dist confirm the customgame emit.
- **shiny details flag** (`|switch|…|Quagsire, M, shiny|…`) — `switch_details` appends `, shiny` from
  `set.shiny`.
- **`-hint` dedup mirrors the sim's `once` param (`ProtocolBuilder::hint(text, once)`)** — the sim's
  `Battle.hint(hint, once, side)` (battle.ts:3092) returns early if the text is already in `this.hints`,
  but only ADDS the text `if (once)`. So a `once: true` hint fires ONCE per battle (**Knock Off**,
  gen4/moves.ts:703), while a `once: false` hint fires on EVERY occurrence (**Sleep Clause Mod**,
  rulesets.ts:1395; the gen3 **Intimidate-vs-Substitute** + **Pursuit** notes, no `once` arg). The port
  used to dedup ALL hints (a `HashSet` insert unconditionally) → a gen3ou **double sleep-clause block**
  emitted the hint ONCE where the sim emits it twice (the byte-fuzzer-surfaced 4th residual). FIXED —
  `hint` now takes `once: bool` (dedup only when `once`); the four call sites pass the sim's value.
  Observation-only → all seed goldens byte-identical.
- **gen3ou clause `-message`** — a Sleep-Clause-blocked 2nd sleep emits `|-message|Sleep Clause Mod
  activated.` + the `once:false` `|-hint|` (once PER block, not deduped); a Freeze-Clause-blocked 2nd
  freeze emits `|-message|Freeze Clause activated.` (rulesets.js). gen3customgame has no clauses → no
  message (so the e2e/seed suites, all gen3customgame, are untouched).

**The WIDE-NET SWEEP round (2026-07-17, round 8) — 7 more emission forms + a bridge-request allowlist gap.**
A broader fuzz (`--mode random` = every gen3 species/ability + `pool` both formats + new master-seeds)
flushed rare forms the bounded pool missed. All SIM-PROBED (the sim is the oracle), FIXED, and pinned by
revert-verified CONSTRUCTED-scenario tests in `protocol_byte_fuzz_test.rs` (BF-F16..BF-F20 + the shared
builder pins); ALL observation-only — the full seed suite stays BYTE-IDENTICAL (cargo test 61/61 green,
e2e md5 unchanged):
- **BF-F16 the two-turn charge `[from] lockedmove` SPACE** — a locked Solar Beam fire-turn announce carries
  `|move|…|[from] lockedmove` WITH a space (the sim's `attrLastMove('[from] lockedmove')`); the port emitted
  the no-space `|[from]lockedmove`. Hits the real gen3ou POOL (Houndoom/Shiftry SolarBeam) at BOTH formats.
  ALSO the fire-turn-MISS ORDER: the sim appends `[from] lockedmove` (useMove) BEFORE `[miss]` (accuracy) →
  `|…|[from] lockedmove|[miss]`; the miss branch now emits the bare move line → lockedmove attr → miss attr.
- **BF-F17 SPEED BOOST `-ability` announce** — the end-of-turn Speed Boost residual emits
  `|-ability|<mon>|Speed Boost|boost` (the sim's ability-source `boost()` announce) BEFORE the
  `|-boost|<mon>|spe|1`; the port dropped it. Emitted only when the stage actually rose (a +6-cap draws
  nothing, both lines).
- **BF-F18 HYPER CUTTER / KEEN EYE `-fail` STAT token** — a SINGLE-STAT boost-blocker carries its stat
  token: Hyper Cutter → `unboost|Attack`, Keen Eye → `unboost|accuracy` (`unboost_fail_stat_token`); the
  whole-table blockers (Clear Body / White Smoke) carry none. The port dropped Hyper Cutter's `Attack` →
  a divergence on any Intimidate/Charm/Feather-Dance-into-Hyper-Cutter.
- **BF-F19 COLOR CHANGE typechange CASE** — `|-start|<mon>|typechange|<Type>|…` renders the DISPLAY-cased
  type (`Type::display_name`, e.g. `Psychic`), NOT the internal UPPERCASE key (`PSYCHIC`).
- **BF-F20 the confusion-berry DOUBLE `-start|confusion`** — a Figy/Mago/Iapapa/Aguav/Wiki confusion berry
  emitted its `-start confusion` TWICE (once inside `add_confusion`, once at the berry site); the redundant
  berry-site emission is removed → EXACTLY one.
- **the EFFECT-SPORE SLEEP over-attribution** — the gen3 `slp.onStart` (`mods/gen3/conditions.js`) DROPS
  the base `[from] ability` branch (only a MOVE source carries `[from] move: <Name>`), so Effect Spore
  inflicting sleep emits a BARE `|-status|<mon>|slp`; the port over-attributed it `[from] ability: Effect
  Spore|[of]`. (par/psn/brn/frz from an ability STILL carry `[from] ability` — the port already matched.)
  **PINNED (round-8 FIX)** by the revert-verified
  `protocol_byte_fuzz_test.rs::effect_spore_sleep_status_is_bare_not_ability_attributed` (Muk Tackles an
  Effect-Spore Breloom, seed `40,96,171,230` forces a proc that samples slp).
- **FUTURE SIGHT fainted-target `-hint`** — a future move resolving against a FAINTED slot occupant emits
  `|-hint|<Move> did not hit because the target is fainted.` (`futuremove.onEnd`, `once` falsy → no dedup);
  the port skipped it (formerly "uncaptured"). **PINNED (round-8 FIX)** by the revert-verified
  `protocol_byte_fuzz_test.rs::future_sight_resolving_against_a_fainted_slot_emits_the_hint` (Tyranitar
  Sand Stream casts Future Sight, chips Growlithe, the RESOLVE-turn sand chip [order 8] KOs it before the
  order-11 resolve → fainted slot → the hint; seed `19,47,80,111`; sim text confirmed vs `conditions.ts:399`).
- **the HEAL BELL / AROMATHERAPY bench-cure IDENT (bonus, random-mode find)** — a Heal Bell curing a
  STATUSED BENCH ally emits `|-curestatus|pN: <mon-name>|<status>|[silent]` (the mon's nickname/species,
  position-less), NOT the player-name `pN: <PlayerName>` the port rendered via `side_ref`.
- **[8]/[9] the bridge `|request|` allowlist gap** — the `return102`/`frustration102` numeric-BP alias and
  the non-Ghost Curse `target:self` ESCAPED the per-side gate on a CO-OCCURRING Curse+Return team: the sim
  renders the alias INCONSISTENTLY (roster `return102`, active id BARE `return`, active display `Return 102`
  with a SPACE — SIM-PROBED), so the old single-direction `replace("return","return102")` OVER-corrected the
  active id AND missed the display, leaving a residual that made the WHOLE reconcile (incl. the correct
  Curse-target fix) return None → both escaped. FIX (`bridge_replay.rs::classify_known_perside_residual`,
  gate-only): NORMALIZE BOTH sides by collapsing every alias form to the bare token before comparing (poke-env
  resolves both). Pinned by `perside_request_residual_tests` (co-occurrence allowlisted + a genuine-diff
  NOT swallowed). Verified GREEN: omniscient pool both formats (0 non-allowlisted) + `random` mode confirms
  the 7 emission forms GONE + bridge pool 0 diverged (curse-target 73 / return102 10 correctly allowlisted).
  ~~HONESTLY-OPEN random-mode-only residuals (NOT gen3ou POOL, deferred): a phaze `[miss]` on an
  evasion-miss, and the Damp-cancels-Self-Destruct `[still]` form.~~ (The Protect-vs-Soundproof TryHit
  order was FIXED in round-8 FIX below.) ⚠️ **BOTH of those are now CLOSED and this line is HISTORICAL** —
  the Damp `[still]` announce in **ROUND 22** (fixture `42_damp_blocks_explosion_move_still.txt`), the
  phaze/evasion + Attract `-end` forms across **ROUNDS 25/26**. Round 30 re-measured the benchmark and
  found **0 protocol divergences**; do NOT re-open these from this paragraph without re-measuring first.

### The KNOWN-RESIDUAL ALLOWLIST + the GREEN GATE (`gen3_omniscient_byte_fuzz_v1`)

The byte fuzzer is a **GREEN GATE**: a NEW divergence fails loudly, while a DOCUMENTED, non-gen3ou-impacting
artifact is EXPLICITLY allowlisted (not silently ignored). Mechanism:
- **`ab_replay --protocol`** classifies a first-divergence protocol byte diff via
  `classify_known_residual(golden_framing, engine_framing, leads_speed_tie)` and adds an `"allowlisted":<reason>`
  field to the per-battle JSON verdict ONLY when the divergence FORM matches a documented residual; otherwise
  `allowlisted` is absent → the divergence counts as a hard failure. Entries are NARROW + STRUCTURAL (never a
  blanket "ignore protocol"):
  - **E1 `turn0-construction-speed-tie-attribution`** (R1) — the R1-SPECIFIC signature is a PURE PERMUTATION
    of identical-CONTENT framing lines at a construction speed-tie: the classifier takes the two FULL framing
    WINDOWS (all lines BEFORE the first `|turn|1`), sorts both, and allowlists ONLY IF `leads_speed_tie` AND
    the two windows are an IDENTICAL MULTISET (a pure reorder). The canonical R1 is a Zapdos-mirror both-Pressure
    lead where the port emits the two `-ability|…|Pressure|[silent]` lines in the OPPOSITE order (same multiset,
    different order — the unmodeled turn-0 construction speed-tie Fisher-Yates shuffle, the project-wide
    seed-convention deferral EVERY seed golden depends on). Both port paths (offline replay
    `run_full_battle_logged` AND the production bridge `run_full_battle_bridge` → `start_with_switchins`) share
    `event::run_start_switchins`, which falls back to a DETERMINISTIC side-order at a raw-Speed tie and draws
    nothing → ZERO production impact under `--use-bridge=rust` (seed=None: the port is the sole oracle). **This
    signature was NARROWED (2026-07-16, the Lens-1 green-gate-integrity fix)** from the prior coarse
    per-line-TYPE key (`line_type ∈ {-ability,-weather,-unboost}`), which SWALLOWED a content-DIFFERENT framing
    divergence at a mirror lead (a genuinely-wrong `[of]` target / wrong weather / a missing-or-extra line — a
    potential real bug). A CONTENT change now makes the multisets DIFFER → returns None → the gate FAILS
    (fault-injection-proven: injecting `-weather|Sandstorm|…|[of] p2a: Zapdos` in place of one `-ability` line of
    the R1 fixture no longer allowlists). The single-line weather-`[of]`-flip consequence is NOW covered by the
    NARROW **A1** key below (it is the SAME construction-window root, but a distinct structural form E1's
    permutation check deliberately does not match).
  - **A1 `turn0-construction-speed-tie-mirror-of-flip`** (2026-07-17, `classify_construction_mirror_of_flip`) —
    the single-line weather-`[of]`-FLIP on a same-species MIRROR lead (repros ab_4_8 / ab_11_20, BOTH formats):
    on a same-species mirror lead at a construction speed-tie, ONE framing `-weather`/`-ability` line's
    `[of] pNa: <name>` attribution flips between the two mirror active slots (golden `[of] p2a`, engine `[of]
    p1a`) because the unmodeled turn-0 construction speed-tie shuffle decides which same-species mon's Sand
    Stream fires last. NOT a pure permutation (the multiset DIFFERS — one line's `[of]` CONTENT changed), so E1
    returns None. Allowlisted ONLY when ALL SIX STRUCTURAL clauses hold (else None → the gate FAILS): (1) the
    construction speed-tie (`leads_speed_tie`); (2) the two equal-length framing windows differ in EXACTLY ONE
    line; (3) that line, in BOTH golden + engine, is a `-weather`/`-ability` framing line; (4) the two are
    byte-identical after stripping the trailing `|[of] pNa: <name>` clause (same weather/ability + `[from]`); (5)
    the two `[of]` targets are the two DIFFERENT active slots (one `p1a:`, one `p2a:`); (6) both `[of]` idents
    map — via the framing `|switch|pNa: <ident>|<Species>,…` details' species field — to the SAME species (the
    sibling mirror). A `[of]` to a NON-sibling / different-species mon, a different weather/ability prefix, a
    missing/extra framing line, or a non-mirror pair breaks a clause → None. seed=None-INVISIBLE (a cosmetic
    `[of]` tag on identical-species mirror mons → ZERO production impact under `--use-bridge=rust`; the port is
    the sole oracle at `seed=None`). **GATE-INTEGRITY PROVEN:** two cp-aside/mangled-golden injections (weather
    `Sandstorm`→`RainDance` breaks clause (4); a non-mirror `Tyranitar`→`Blissey` lead breaks clause (2)/(6))
    both replay to `diverged` with NO `allowlisted` field → the gate FAILS; plus 7 `#[cfg(test)]`
    `a1_allowlist_tests` in `ab_replay.rs` (incl. `clause6_wrong_of_to_a_real_different_species_mon_fails` — the
    load-bearing "wrong-[of]-to-a-real-different-mon → must FAIL"). Corpus fixture
    `27_construction_mirror_weather_of.txt` (tagged). **GREEN both formats** (master-seed 424242, 300 battles
    each): 0 non-allowlisted, 4 allowlisted (2 R1 `turn0-construction-speed-tie-attribution` + 2 A1
    `turn0-construction-speed-tie-mirror-of-flip` = ab_4_8 + ab_11_20).
- **`ab_fuzz.js --protocol`** counts `allowlisted` SEPARATELY from `diverged`, reports allowlisted-by-reason,
  and **exits non-zero ONLY on a non-allowlisted `diverged`/`panic`/`parse_error`** (a non-protocol run stays
  a hunter, exit 0). Allowlisted repros are saved under `<out>/allowlisted/` (auditable + a fixture source);
  real divergences under `<out>/divergences/`.
- **`tests/byte_fuzz_corpus_test.rs`** (the `cargo test` gate) enforces "no NEW kinds": each fixture resolves
  to either `ok` (the 20 emission-form fixtures stay byte-clean) OR a `diverged` verdict whose `allowlisted`
  reason EXACTLY equals a `# ALLOWLIST <reason>` header the fixture is tagged with — the R1 fixture
  `21_construction_mirror_ability_of.txt` (`turn0-construction-speed-tie-attribution`) + the A1 fixture
  `27_construction_mirror_weather_of.txt` (`turn0-construction-speed-tie-mirror-of-flip`). So (i) a residual
  fixture that stops matching its reason FAILS,
  and (ii) nobody can add a silently-ignored divergence — every escape is a named allowlist entry backed by a
  tagged repro. FAULT-INJECTION PROVEN: stripping the R1 fixture's tag (→ an un-cataloged divergence) makes
  the corpus test FAIL (`REGRESSION … kind=protocol`), restored byte-identical.

**R3 (IV-derived Hidden Power BP) — FIXED in the engine (`gen3_iv_derived_hidden_power_bp_v1`), quarantine
REMOVED.** gen-3 computes Hidden Power's base power from the ATTACKER's IVs (`Dex.getHiddenPower`,
`⌊hpPowerX·40/63⌋+30`, range 30..=70), NOT the flat 70 the data ships (all 16 typed HP rows are BP 70 in
`gen3_moves.json`). The port used to read the data's 70, over-damaging any real gen3ou HP mon whose IVs give
BP != 70 (a -1 Atk-IV spread → BP 68 → ~2/68 ≈ 3% over-damage that cascaded KO thresholds — the ~1.5%-of-pool
byte-fuzz STATE divergence). FIX: `state.rs::hidden_power_bp(ivs)` precomputes each mon's `MonState.hidden_power_bp`
in `from_set` (the WEIGHT-ORDER CRUX: the sim iterates `{hp,atk,def,SPE,SPA,spd}` so Speed carries weight 8 —
BEFORE SpA 16 / SpD 32 — while the port's IV array is `[hp,atk,def,spa,spd,spe]`; a naive array-order mapping is
WRONG; GIGO-guarded to 30..=70). `run_move` overrides the data BP with it for any `hiddenpower*` move (the TYPE
stays from the typed id); the bridge request-JSON move name reads it too (the sim renders `Hidden Power <Type>
<IV-power>`). It is a bridge-correctness fix only — the Python RL obs / DamageOperator keep their own
internally-consistent BP-70 assumption (untouched). Confirmed bit-for-bit vs `getHiddenPower` (source read) AND a
live damage probe. The `ab_fuzz.js` pool `teamHpBp70Clean` picker quarantine is REMOVED — every real gen3ou HP
team is now byte-safe. STATE-fix but OBSERVATION-NEUTRAL for the committed goldens (the e2e never PICKS HP —
`isModeledMove` defaults `allowHiddenPower=false` — and the constructed goldens use IV-31 → BP 70 → unchanged; the
e2e golden md5 `3155eb796cb4bf453c6053d769ba98e5` is UNCHANGED). Pins: `hidden_power_bp_is_iv_derived_not_flat_seventy`
(a constructed BP-68 HP Ice → the sim's 53 dmg, revert-verified) + `hidden_power_bp_weight_order_and_boundaries`
(the SPE-8-not-32 crux + 30..=70 boundaries); ground truth `harness/probe_hidden_power_bp_regression_rng.js`.

**R2 (the Leftovers `-heal` emit ORDER on a residual speed TIE) — FIXED (`gen3_leftovers_slotcond_gather_order_v1`).**
Root-caused via `harness/probe_r2_repro_trace.js` on a byte-fuzz repro (a Jolteon mirror, both Leftovers,
sandstorm, a pending p2 Wish): `speed_sort` is a NON-STABLE selection sort whose swaps DISTURB the relative order
of the tied handlers, so the tie-group Fisher-Yates shuffle reads whatever pre-sort order the swaps LEFT the tied
pair in. Showdown gathers a side's SLOT CONDITIONS (Wish order 7 / Future Move order 11) via
`findSideEventHandlers(…, active)` — AFTER that active's pokemon handlers — so Wish sits AFTER Leftovers in the
pre-sort array. The port gathered Wish/FutureMove FIRST (a pre-loop at the array front), so the selection-sort's
Wish/weather swaps REVERSED the tied Leftovers pair vs the sim → the two `-heal` lines emitted in the OPPOSITE
order at the SAME shuffle value (the seed matched — the Sodium seed depends on the draw COUNT, not the
permutation). FIX: `run_residuals` now gathers the Wish + FutureMove handlers PER-ACTIVE at the end of the
per-active loop (after the item), mirroring the sim. DRAW-NEUTRAL (same handlers / sort keys / tie-group COUNT →
the seed is unchanged; only the emit permutation moves to match Showdown) → OBSERVATION-ONLY, every seed +
protocol + writeline golden stays BYTE-IDENTICAL. Pin: `leftovers_heal_order_follows_the_slot_condition_gather`
(a Jolteon-mirror + pending-Wish scenario whose `-heal` marker sequence == Showdown's, revert-verified) + the
byte-fuzz corpus fixture `23_leftovers_wish_heal_order.txt` (the repro replays byte-clean; reverting the fix →
`kind=protocol` at the exact `-heal` line, fault-injection proven); ground truth
`harness/probe_r2_wish_leftovers_regression_rng.js`.

**R13 (the ENCORE × SLEEP TALK draw-count desync) — FIXED (`gen3_encore_sleeptalk_trylhit_v1`, the pool
byte-fuzz `--protocol` gen3ou find ab_15_15 @ master-seed 222333, dec 47 — `kind=seed`).** Root-caused via
the FIXED `harness/probe_repro_simtrace.js` (in `--format gen3ou` — the default is customgame, so a
gen3ou repro MUST be replayed with `{format:'gen3ou', allowHiddenPower:true}` or the sim diverges from the
golden) + an env-gated per-draw PRNG backtrace in the port: at dec 47 the port drew **7** where the sim
drew **3** (+4). Suicune (asleep, lastMove Sleep Talk) is Encored by a FASTER Jumpluff THIS turn (Encore
locks Sleep Talk); the port's Encore `onOverrideAction` redirects Suicune's queued Rest → the encored Sleep
Talk slot, then RAN Sleep Talk (sampled a move + ran it, +4 draws). The SIM instead FAILS Sleep Talk
DRAW-FREE: Sleep Talk's resolved **`onTryHit(pokemon){ return !volatiles['choicelock'] && !volatiles['encore']; }`**
returns false when the user carries the `encore` volatile → `|move|…Sleep Talk||[still]` + `|-fail|`, NO
sample. (probe-confirmed: the sim's `singleEvent('TryHit', sleeptalk)` → false at turn 43, activeMove
sleeptalk, 0 draws; the golden protocol shows `cant slp` + `Sleep Talk [still]` + `-fail`.) So an
Encored-into-Sleep-Talk mon can NEVER resolve Sleep Talk while the encore holds. The port's sleeptalk arm
gated ONLY on `choicelock` (via `was_choice_locked`) — batch 5 wrote "Encore is unmodeled gen-3-wide" as
the reason, but **batch 6 later MODELED Encore and the gate was never updated** — the latent bug the pool
fuzzer surfaced. FIX (`turn.rs`, the sleeptalk arm): `if was_choice_locked || encore.is_some()` fails Sleep
Talk draw-free (the encore is read at its CURRENT value, NOT a pre-move snapshot, because the foe's Encore
can land THIS turn before the sleeper's Sleep Talk — matching the sim's live `volatiles['encore']` read).
NOT a clause-path fix, but OBSERVATION-NEUTRAL for every committed golden (no gen3customgame e2e/seed board
pairs an Encored mon USING Sleep Talk → full suite byte-identical, **e2e md5 `3155eb796cb4bf453c6053d769ba98e5`
UNCHANGED**, every seed golden byte-identical). Pinned by the revert-verified
`regression_test.rs::encored_mon_sleep_talk_fails_draw_free_via_ontryhit` (a fast Electrode Encores an
asleep Snorlax RestTalker → the encored Sleep Talk fails draw-free; ground truth
`harness/probe_r13_encore_sleeptalk_rng.js`, raw seed [7,11,13,17]). RE-VERIFIED: the 222333 pool
`--protocol` gate is now GREEN both formats (gen3ou 0 non-allowlisted, ab_15_15 → `ok`; gen3customgame
unchanged).

**R15 (the SLEEP-CLAUSE × self-REST-sleeper draw-count desync) — FIXED
(`gen3_sleep_clause_self_rest_exempt_v1`, the pool byte-fuzz `--protocol` gen3ou find ab_3_15 @
master-seed 333444, dec 24 — `kind=seed`, expected `56120,…` got `20330,…`; DISTINCT from R13).**
Root-caused via the R9-fixed `probe_repro_simtrace.js` + a temporary env-gated per-draw backtrace in
`prng::next` (removed): the port did **one EXTRA draw** at dec 24 (a second SetStatus clause shuffle).
The dec-24 turn is p1 Breloom **Spore → p2 Suicune** while p2's benched Zapdos is asleep FROM ITS OWN
turn-1 **Rest**. gen3 **Sleep Clause Mod** (`rulesets.ts` `sleepclausemod.onSetStatus`) counts an
existing sleeper toward the one-foe-asleep cap ONLY when `!pokemon.statusState.source?.isAlly(pokemon)`
— a mon that put ITSELF to sleep via Rest is EXEMPT (probe-verified `harness/probe_r15_sleepclause.js`:
a self-Rested Zapdos does NOT block a foe's Spore on Suicune → Suicune gets `slp`). WRONG (pre-fix):
`turn.rs::side_has_sleeper` counted ANY asleep mon (a stale "Rest is out of scope" assumption from
before Rest was modeled), so the port clause-BLOCKED the Spore; Suicune's own Rest then ran, drawing an
EXTRA clause shuffle (+1 draw → the seed divergence + a silent Suicune-state divergence ab_replay's
active-only state check missed). The fix: a new `MonState::sleep_from_rest` (appended LAST; set `true`
in `run_rest`, `false` in `try_set_status_impl` — mirrors `statusState.source.isAlly`), and
`side_has_sleeper` skips self-Rest sleepers. Revert-pinned:
`regression_test.rs::sleep_clause_exempts_a_self_rest_sleeper_from_blocking_a_foe_sleep_move`
(gen3ou, the sim's first-decision seed 53118,…; ground truth `harness/probe_r15_pin_truth.js` — post-T2
seed `21514,3448,20660,22314`, FAILS on revert). SEED-NEUTRAL for every committed golden: e2e md5
UNCHANGED `3155eb796cb4bf453c6053d769ba98e5`, all seed suites byte-identical (id-gated: only a gen3ou
board with a benched self-Rest sleeper AND a foe sleep move on a different mon hits it). The 333444 pool
`--protocol` gate is GREEN both formats (gen3ou ab_3_15 → `ok`, diverged 0; gen3customgame diverged 0).

**T1 (the FREEZE-PERSISTENCE deep-STATE bug) — FIXED (`gen3_omniscient_byte_fuzz_v1`, byte-fuzz repro
rmroh04is_ab_4_18 / _4_8 + cg rmrohcsti_ab_4_18/_11_23).** The round-1 "deep state/HP divergence" (the port at
FULL HP where the sim was not) root-caused to the port OVER-THAWING a frozen mon hit by **Hidden Power Fire**
(or Weather Ball). gen3 `frz.onDamagingHit` (conditions.ts:45-50) thaws ONLY when
`this.dex.moves.get(move.id).type === 'Fire'` — the BASE-dex move type — with the explicit "don't count Hidden
Power or Weather Ball as Fire-type" comment (`dex.moves.get('hiddenpower').type === 'Normal'`, `'weatherball'
=== 'Normal'`). The port computed `is_fire` from the RESOLVED runtime type (Fire for the typed-HP nums
355-370), so HP Fire WRONGLY thawed. It surfaced as a kind=status "sim=Freeze / port=None" (ab_replay compares
only the ACTIVE mon's status per decision, so the lost freeze is caught only when the frozen mon becomes active
again). FIX (turn.rs, the `is_fire` decl): `move_type == Some(Type::Fire) && category != Status &&
!to_id(&m.id).starts_with("hiddenpower") && to_id(&m.id) != "weatherball"`. `is_fire` is used ONLY at the thaw
(grep-confirmed decl + the thaw site) — Flash Fire absorb / fm_frozen read `move_type == Some(Type::Fire)`
DIRECTLY (the resolved type), so a Flash Fire holder STILL correctly absorbs HP Fire. STATE fix but
OBSERVATION-NEUTRAL for every committed golden (no committed golden pairs a frozen mon with an HP-Fire/Weather-
Ball hit → full suite byte-identical, e2e md5 `3155eb796cb4bf453c6053d769ba98e5` UNCHANGED). Pins (revert-verified):
`hidden_power_fire_does_not_thaw_a_frozen_defender` (STATE: still frozen — revert → thaws → None) +
`flamethrower_does_thaw_a_frozen_defender` (control: a base-type-Fire move DOES thaw) +
`flash_fire_still_absorbs_hidden_power_fire` (control: the narrowing didn't break the FF absorb); corpus fixture
`26_freeze_persists_vs_hp_fire.txt`. The 4 freeze repros (4_18/4_8/11_23 + the freeze-adjacent seed 10_18) flip
to ok.

**Two EMISSION-FORM byte fixes (observation-only, seed suite BYTE-IDENTICAL) — the actual round-2 protocol
tail (the round-1 T2-T6 list did NOT reproduce in the 600-battle pool run; do not trust it):**
- **ENDURE survive-at-1 `-damage` at exactly 1 HP** (repro rmroh04is_ab_8_4 / cg): when an endurer is ALREADY at
  1 HP, the endure clamp nets 0 damage (the mon stays at 1), so the caller's `realized > 0` emission gate SKIPPED
  the `|-damage|<mon>|1/<max>` line the sim STILL emits after `|-activate|move: Endure`. FIX: `endure_clamp`
  emits the `-damage` itself when it clamps to 0 (apply_damage(0) is a no-op → the current HP == post-apply HP),
  covering all 4 endure sites; the `hp > 1` path stays caller-emitted (no double). Corpus fixture
  `24_endure_survive_at_one_hp.txt`.
- **NATURAL CURE `-curestatus` on a Pursuit-KO'd switcher** (repro rmroh04is_ab_10_1): the gen 2-4 Pursuit
  interrupt runs the switcher's SwitchOut event (hence `naturalcure.onSwitchOut`) on the 0-HP-but-NOT-YET-fainted
  mon, so its tox/etc is CURED with a `|-curestatus|<mon>|<tok>|[from] ability: Natural Cure|[silent]` line BEFORE
  the `|-hint|`/`|faint|`. The port's Pursuit path called `process_faints` (setting `fainted`) BEFORE the switch,
  so the later clearVolatile Natural-Cure block (`!fainted`-gated) skipped it. FIX: run the NC cure on the
  hp==0-not-fainted switcher in the Pursuit-interrupt block, BEFORE the hint + process_faints (state-neutral —
  the mon faints anyway). Corpus fixture `25_natural_cure_pursuit_ko_curestatus.txt`.

**HONESTLY-OPEN residual TAIL (round-2, POST the T1 + emission + allowlist-narrowing fixes; master-seed 424242,
ISOLATED build).** gen3ou 300 battles: **7 → 4** non-allowlisted; gen3cg 300: **5 → 3**. The remaining are TWO
classes:
- **2 REAL seed (draw-count) bugs — both ✅ FIXED (round-4, `gen3_pressure_allyteam_v1` PA3):** the ROOT of
  BOTH **5_6** (both formats, ou dec159 + cg dec143) AND **3_24** (gen3ou, ou dec147) was the SAME
  **Pressure-over-deducts-PP-on-a-non-Ghost-Curse** bug — NOT the "Roar-loop PHAZE" / "Magnet-Pull switch"
  labels (those were wrong guesses). gen-3 `curse.onModifyMove` RE-TARGETS a NON-Ghost user's Curse to `self`
  (`nonGhostTarget`), but the STATIC dex `target` is `"normal"`, so `pressure_targets_foe` read `"normal"` and
  deducted 2 PP under a Pressure foe instead of 1 → the Curse-slot PP drained ~1 cycle early → forced Struggle
  turns the sim still Curses → the deep seed desync + cascade (the recorded protocol-diff was `|move|…|Struggle`
  vs `|move|…|Curse`; Curse counts SIM 16× / PORT 10× + 6 Struggles). FIX: `turn.rs`'s move-resolution tuple
  now computes the RUNTIME-effective Curse target (`is_nonghost_curse` → `"self"`) and feeds it to BOTH
  `targets_self` and `pressure_targets_foe`, so a non-Ghost Curse deducts 1 PP under Pressure. A NEW third case
  of the `gen3_pressure_allyteam_v1`/`gen3_pressure_foeside_v1` class. Pin: revert-verified
  `regression_test.rs::pressure_does_not_add_pp_for_a_nonghost_curse` (Zapdos(Pressure)-vs-Swampert(Curse):
  Curse PP 16→15 not →14 + the dec7 post-decision seed; ground truth
  `harness/probe_pressure_curse_regression_rng.js`). Verified: the 424242 pool run (both formats) now leaves
  ONLY the 2 harmless construction `[of]`-flip artifacts (5_6 / 3_24 / the cg dec-30 Tyranitar-sandstorm
  cascade all GONE — the cascade shared this same Pressure-Curse root).
- **The construction weather-`[of]`-flip protocol artifacts (2/format, harmless) — ✅ now NARROWLY ALLOWLISTED
  (round-6, A1 `turn0-construction-speed-tie-mirror-of-flip`, see the A1 entry above).** 4_8 / 11_20 (both
  formats) are single-line `-weather|…|[of] p1a` vs `[of] p2a` mirror flips (seed=None-invisible, zero obs
  impact) — the follow-on narrow signature the round-2 note anticipated. The 6-clause key allowlists ONLY this
  exact same-species-mirror `[of]`-flip form; injection-proven not to swallow a wrong-`[of]`-to-a-real-different-
  mon. So the OMNISCIENT gate is now GREEN both formats (0 non-allowlisted).

The round-1 T2-T6 list (`-miss` p2:/p2a:, Light Screen `-sideend`/`-resisted` order, faint-vs-`|upkeep|` order,
Solar Beam `|[from] lockedmove` gap, Morning Sun `||[still]`) did NOT reproduce as a non-allowlisted divergence
in this 600-battle pool run — either already resolved by the intervening emission sweep or lower-frequency;
NOT blind-fixed.

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
     (`gen_bridge_trapping_capture.js`, 5 scenarios — the 3 trap scenarios + `taunt_struggle` /
     `pp_stall_struggle`, `gen3_bridge_struggle_resolve_v1`) → `bridge_test.rs`.
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

### THE BANKED SPEC QUEUE — 8 probe-settled specs, ALL SHIPPED (rounds 45-52)

Each is a re-runnable oracle in `harness/` whose header carries a SETTLED block: the draw model, the
exact emission forms, the edges, and the named way a naive implementation desyncs. **Read the probe
before implementing; do not re-derive from source.** They are listed with the trap that makes each
one non-obvious, because that trap is the reason the spec is worth more than the move.

| move(s) | probe | the trap |
|---|---|---|
| ~~Safeguard~~ **SHIPPED** | `probe_safeguard.js` | two Safeguards **TIE at residual order 4** → an extra shuffle, invisible to any single-side test; and a blocked SECONDARY still rolls its `random(100)` |
| ~~Recycle~~ **SHIPPED** | `probe_recycle.js` | the discriminator is WHICH primitive removed the item — `eatItem`/`useItem` set `lastItem`, `takeItem` does NOT, so Knock Off is **not** recyclable |
| ~~Fake Out~~ **SHIPPED** | `probe_fakeout.js` | gen-3 priority is **+1, not +3**; a CANCELLED action does not burn the first-turn gate but a CANT turn does, so `active_turns` is silently wrong |
| ~~Conversion / Conversion 2~~ **SHIPPED** | `probe_conversion.js` | `n == 1` **still draws**; the candidate list uses `types.names()` order, which matches neither our type chart nor the port's enum |
| ~~Torment~~ **SHIPPED** | `probe_torment.js` | joins the endTurn **DisableMove tie group** (n−1 draws); permanent, so a duration would add a phantom handler |
| ~~Imprison~~ **SHIPPED** | `probe_imprison.js` | the disable is **HIDDEN** — the request keeps `disabled:false` and gains `maybeDisabled`/`maybeLocked`; all-imprisoned SUBSTITUTES Struggle rather than rejecting |
| ~~Weather Ball~~ **SHIPPED** | `probe_weatherball.js` | the CATEGORY flips with the type (sandstorm is PHYSICAL) — **a test board with Atk==SpA and Def==SpD cannot see it**; reads `effective_weather` so Air Lock reverts it fully |
| ~~Skill Swap~~ **SHIPPED** | `probe_skillswap.js` | swapped abilities do **NOT** re-fire `onStart` (so re-running switch-in makes Trace draw where the sim does not) — but `onEnd` DOES fire, and that is the only draw it creates |

### The `ab_replay` SUBSEQUENCE SEED ANCHOR — shipped, and it did NOT close the repro that motivated it

`gen3_ab_replay_seed_anchor_subsequence_v1`. Round 26 left this as the scoped next step: port
`bridge_replay.rs::anchor_seed_divergence` (round 20's A2 fix) to `ab_replay`, so a decision-boundary
CHECKPOINT offset stops reading as `kind:"seed"` and costing a full triage.

**What landed.** `align_seed_subsequence` aligns the sim's per-decision seeds as a SUBSEQUENCE of the
port's checkpoints and the per-decision STATE/species/status/boost checks then run at the ALIGNED
pairs (ab_replay checks more per boundary than its bridge sibling, so aligning only the seed would
have left the state checks comparing unrelated boundaries). Verdicts still report the GOLDEN's
decision index, which is what a reader greps for in `battle.txt`.

**Held to round 20's bar, because a sloppy anchor makes the omniscient gate VACUOUS — far worse than
the artifact it removes.** `anchor_tests` is 10 cases and the NEGATIVES are the load-bearing half: an
**INJECTED EXTRA DRAW** still fails (a real desync permanently shifts every later seed VALUE, so the
sim's post-divergence seeds never reappear and the subsequence necessarily breaks), as do a missing
draw, a REORDERED pair (same values, wrong order), a port stream that ends early, and an empty port
stream. Gates: committed e2e golden still replays **220 ok / 0 diverged** through `ab_replay`; full
suite **75 binaries / 666 passed / 0 failed**; e2e md5 `3155eb…` UNCHANGED.

**NEW DIAGNOSTIC `POKESIM_DUMP_SEEDS=1`** prints BOTH per-decision seed lists. The anchor is only
sound if the port's checkpoint list really is a SUPERSET of the sim's, and that is now CHECKABLE on
any repro rather than assumed.

⚠️ **THE FINDING: `ab_41_9` is a CHECKPOINT-PLACEMENT artifact — the port's boundary lands exactly
ONE DRAW EARLY — and the anchor correctly does not reconcile it.** The port's RNG consumption is
CORRECT; there is no draw bug and no missing request.

**THE PROOF, from draw POSITIONS inside one decision** (`POKESIM_PRNG_TRACE=1` for the port,
`probe_repro_simtrace.js` for the sim). Decision 5 opens at the port's draw line 20 (the dec-4 seed):

| | |
|---|---|
| port's dec-5 checkpoint | line **29** → **9** draws consumed |
| the SIM's dec-5 seed | line **30** → the port's VERY NEXT draw |

and the sim's own trace carries the two values ADJACENT, in the same order:

```
[dec 6] draw#5  seed_before=12362,51825,42696,29605   <- the PORT's dec-5 checkpoint
[dec 6] draw#6  seed_before=58601,2165,55458,35804    <- the SIM's dec-5 checkpoint
```

Both engines therefore traverse the SAME seed values in the SAME order; only the pause point differs,
by one draw. The board is a forced-switch (dec 4, Swampert faints → Charizard) followed by a move
turn — the boundary bookkeeping shifts by one across that transition.

**WHY THE ANCHOR CANNOT ABSORB IT, and why that is correct.** The anchor matches sim seeds against
the port's **CHECKPOINT LIST**, and `58601` never appears there — it exists only in the port's DRAW
stream. Closing this class properly means anchoring against the DRAW stream instead, which is a
materially larger change carrying its own vacuity risk (the draw stream is dense, so a sloppy
draw-anchor would match almost anything). NOT attempted; recorded here as the next step for whoever
takes it, together with the warning.

**THREE READINGS, AND ONLY THE THIRD IS RIGHT — the durable methodological lesson.** Within one
session this repro was called: (1) "the known segmentation artifact", asserted from a byte-clean
`POKESIM_PROTOCOL_ONLY` replay — under-evidenced; (2) "the port surfaces FEWER requests, possibly
round 26's hypothesis (b), a real draw-free legality bug" — WRONG, and the alarming one; (3) the
one-draw checkpoint offset above. What settled it was comparing draw POSITIONS **within a single
decision**. The earlier reads compared draw COUNTS across differently-scoped windows — `ab_replay`
plays the WHOLE scripted battle before it compares, so its 171-draw trace covers all 33 port
decisions, not the 5 the verdict names. **A count comparison whose two sides cover different windows
is not evidence, and it reads exactly like evidence.**

### `--mode ourandom` — "gen3ou-randbats", the fuzz surface that is actually the one we care about

`gen3_ou_random_teams_v1` (`harness/ou_random_teams.js`). The fuzzers had two team sources and
NEITHER is the training/ladder surface:

| mode | on-surface? | diverse? |
|---|---|---|
| `pool` | **yes** — the 722 real gen3ou teams | **no**: a FIXED human-built set from a narrow meta, and the committed capstone samples only 220 battles of it |
| `randbats` | **no** — non-L100 levels, curated movesets, near-uniform items | yes |
| **`ourandom`** | **yes** | **yes** |

**THE MOTIVATING MEASUREMENT.** Both bugs found on 2026-08-17 (ROUND 42's Trace/forecast, ROUND
43's Substitute/wrap) have **ZERO gen3ou-pool exposure** — 0 of 773 pool files carry Castform, 0
carry a wrap-family move. Training plays pool-vs-pool, so neither could ever have fired there.
That is the round-24 lesson ("fix bugs found on the SURFACE YOU CARE ABOUT") restated as a number,
and it is why a gen3ou-native random generator is worth having.

**EVERY INPUT IS SMOGON-DERIVED AND ALREADY COMMITTED** — no new data, no scraping:
`gen3_smogon_stats.json`'s per-species `usage` (which species appear) · `gen3_teammate_priors.json`
(the species×species joint, so teams are recognisable CORES not six unrelated mons) ·
`gen3_move_priors.json` / `gen3_item_priors.json` / `gen3_ability_priors.json` /
`gen3_spread_priors.json`. Deliberately NOT `data/teams/gen3_species_priors.json` — that is
POOL-derived, and the whole point is independence from the 722.

**LEGALITY IS SHOWDOWN'S VERDICT, NOT A REIMPLEMENTATION.** Every generated team goes through the
real `TeamValidator('gen3ou')`, so the banlist and every team-building clause are enforced by the
sim. Rejections are genuine gen-3 breeding facts (`Tyranitar can't get its egg move combination
(dragondance, pursuit)`), reject-sampled away.

**COVERAGE IS DISCLOSED, NOT ASSUMED.** Move sampling is renormalized over ENGINE-MODELED moves, so
a generated battle ALWAYS plays to completion — no fail-loud, no truncated prefix. The cost is
exactly the renormalized mass: **1.33% of gen3ou move-slot prior mass** sits on the 88 unmodeled
moves, printed in the run banner by `describeCoverage()` every run. Closing it is a NAMED, finite
queue: `sandattack recycle confuseray safeguard conversion weatherball fakeout imprison present
skillswap torment eruption`.

**HIDDEN POWER — the one non-obvious mechanic, and it is solved in closed form.** HP is ~12% of
gen3ou move slots (47.9% of mons carry one), so dropping it would gut the generator, and gen-3 HP
derives BOTH type and base power from IVs. BP 70 requires the bit-1 sum to be 63 — i.e. bit 1 set
in EVERY IV — which is true of both 30 (`11110`) and 31 (`11111`). So restricting IVs to {30,31}
PINS BP at 70 automatically and the bit-0 pattern alone selects the type; `hpIvsForType`
brute-forces the 64 patterns once. **Verified 16/16 by RECOMPUTING type and BP from the emitted
IVs**, not by trusting the table. Because BP is pinned, `allowHiddenPower` is enabled for this mode
exactly as it is for `pool`.

**TWO BUGS THE REAL DATA SHAPES CAUGHT** (both would have been invisible to a source read):
`gen3_spread_priors.json` stores a **LIST** of `[nature, [hp,atk,def,spa,spd,spe], weight]` triples,
not a dict — reading it as a dict silently passes the ARRAY INDEX as the nature, which Showdown
rejects with `"24" is an invalid nature`; and all typed Hidden Powers validate as ONE move, so a set
may carry at most one (`<species> has multiple copies of Hidden Power <type>`).

**First results:** a 50-battle smoke is `GREEN-GATE PASS`, 0 divergences, **69 distinct species and
95 distinct moves** — against 34 species in 25 `pool` battles. Same `{packed, genSeed}` interface as
`makeRandbatsProvider`, so it drops into `gen_sim_bridge_diff.js` unchanged too.

### The picker's own blind spots (found while measuring the above)

- **STRUGGLE is now pickable.** It is ENGINE-MODELED (`pp_struggle_test.rs` is a full
  STATE+PP+SEED+winner differential) but `isModeledMove` returns false for it, so a mon with every
  slot spent AND no switch had no pickable choice and the whole battle was DROPPED to a prefix.
  That truncated precisely the PP-exhaustion endgames — the deepest, most state-laden turns, and the
  ones gen3ou STALL teams produce most. The live per-side gate already accepted Struggle
  (`id === 'struggle' || isModeledMove(id)`); the two harnesses simply disagreed and the offline one
  was weaker. **A picker predicate that gates a test silently SHRINKS that test** — the same shape
  as ROUND 42's L100 pin.
- **The drop LABEL named an innocent bystander.** `forced-unmodeled-move:<moves[0]>` reported the
  FIRST slot in the request regardless of why the pick failed, so a drop on a mon whose first slot
  happened to be Substitute read as `forced-unmodeled-move:substitute` — and Substitute is modeled,
  so the label sent a reader hunting a bug that does not exist (it did, on 2026-08-17). It now names
  the moves that actually blocked, with a distinct `all-disabled(...)` reason. **A diagnostic that
  names an innocent bystander is worse than one that names nothing.**

### THE EXTERNAL-CONSISTENCY GATE (`gen_sim_bridge_diff.js`) — promoted to a green-gated fuzzer

`gen3_simbridge_diff_allowlist_v1`. **This is the strongest correctness gate in the project**, because
it is the only one that compares what poke-env ACTUALLY consumes, at the boundary poke-env sits on.

**Why it outranks the byte/seed gates.** `ab_fuzz`/`ab_replay` diff the OMNISCIENT log — which poke-env
never sees — and replay a FIXED recorded decision list, so they must ASSUME both engines segment
decisions identically. When they don't, you get a `kind=seed` artifact that cannot be distinguished from
a real legality bug (the round-26 finding: 12 of 13 open repros have ZERO wrong draws). This harness
instead spawns BOTH real bridges, feeds identical stdin, and **discovers boundaries live** (read a
request → choose → compare). The segmentation artifact CANNOT occur by construction, and a genuine extra
request is an unambiguous request-frame mismatch. It also covers the `|request|` JSON — a genuinely
separate observable with its own bug history (PA2's Spikes-under-Pressure PP was INVISIBLE to the
omniscient fuzzer and only diverged in the request's `pp` field).

**THE LAYERING PRINCIPLE.** per-side + request = the CONTRACT (the correctness requirement); the
omniscient log = a LOCALIZER (engine bug vs fold/serializer bug); the PRNG seed = a LEADING INDICATOR
(catches divergence before it is observable). **An outer-layer mismatch is ALWAYS a bug; an inner-layer
mismatch with a clean outer layer is NOT necessarily one** (the turn-0 construction residuals are exactly
that). Inner layers buy detection SPEED and LOCALIZATION, not correctness.

**What landed:**
- **The GREEN GATE + allowlist.** Previously any known-benign residual failed the run — a 10-battle
  randbats smoke reported **8/10 "diverged", ALL the same `return102` alias**, making the gate unusable.
  Now a documented request-DISPLAY residual is counted separately, filed under `<out>/allowlisted/`, and
  does not fail; exit is non-zero ONLY on a non-allowlisted divergence (the `ab_fuzz` convention). Keys
  mirror `bridge_replay.rs::classify_known_perside_residual`.
- **SAFETY: reconcile-then-compare.** The allowlist applies the documented transforms and requires the
  FULL lines to be byte-EQUAL after; ANY residual difference → not allowlisted → the gate FAILS. A
  co-occurring pair is allowed as a UNION.
- **⚠️ THE `gender-level-details` KEY IS DELIBERATELY NOT PORTED.** A blanket symmetric strip of
  `, L<n>` / `, <gender>` reconciles a genuine VALUE divergence (node `L84` vs rust `L83`, or M vs F)
  into a FALSE PASS — the exact way an allowlist turns a gate VACUOUS. It was written, caught by its own
  self-test, and REMOVED; the run then still went green under `return102-numeric-alias` alone, proving it
  was never needed. If a real presence-vs-absence residual appears, implement it PAIRWISE (strip only
  from the side that HAS the suffix) — never symmetrically. The other two keys are safe because they map
  an ALIAS to a CANONICAL form, so a real value difference still fails to reconcile.
- **`--selftest`** — 14 gate-integrity assertions: 10 allowlist ones whose negatives are load-bearing (a
  different move; an alias PLUS a residual pp difference; a LEVEL value difference; a GENDER value
  difference; a missing move; a non-request line; identical lines), plus 4 that pin the **switch-probe
  content discriminator** (ROUND 30). Run after ANY change to `ALLOWLIST_TRANSFORMS` or the probe path.
- **THE DRAIN / PROBE CONTRACT (ROUND 30).** Every wait on a child is bounded and every bound is
  CHECKED: `assertDrained` on START + per-decision (round 27), content-based acceptance on the trapped
  switch probe (`probeWasAccepted`, `PROBE_MAX_MS` 750 ms), and a per-battle wall-clock budget
  (`BATTLE_BUDGET_MS` 300 s) over the outer loop. The summary reports `switch_probes_accepted` /
  `drain_timeouts` / `drain_timeouts_by_tag` — **`drain_timeouts` must stay 0**; any non-zero value means
  a child went quiet somewhere, even on a path that recovers. Env overrides for investigation:
  `SBD_DRAIN_MAX_MS`, `SBD_PROBE_MAX_MS`, `SBD_BATTLE_BUDGET_MS`, `SBD_TRACE_DRAINS=1` (log every timed-out
  drain with its call-site tag), and `POKESIM_SIMBRIDGE_TARGET` (so two concurrent investigations never
  share one cargo target dir). Gate: `probe_switch_probe_accepted.js` (deterministic, hermetic,
  revert-verified 7.0 s → 181 s).
- **Throughput measured: ~600 battles/hr with `--persistent`, ~80/hr without** (each battle otherwise
  respawns a Node child that reloads the whole Showdown dist). **~96% of wall time is the per-write
  quiescence settle, not CPU** (`user` ≈ 2.3s per 59s wall) — so `--persistent` is mandatory for a soak,
  and the 40 ms settle is the remaining lever (still ~5x slower than `ab_fuzz`'s ~3000/hr).

**HONEST SCOPE (unchanged):** cross-side p1/p2 interleaving is not asserted (a Node scheduler artifact
the Python demux does not depend on); `__RECON__` is excluded (a real rust deferral; `resumeReseed` now works, `gen3_bridge_resume_reseed_v1`, so
reconstruction + search paths still require node); unmodeled moves fail loud, so "clean" is always
relative to the modeled universe.

**THE NEXT STEP THIS UNBLOCKS.** Running it at scale settles whether the round-26 `kind=seed` repros are
(a) offline-harness segmentation or (b) a draw-free legality bug — the question gating the `ab_replay`
subsequence anchor. A clean soak supports (a) and makes the anchor safe; request mismatches mean (b), and
the anchor would have HIDDEN them. **A further strengthening worth considering: parse both streams with
the real `src/poke_env` fork and diff the resulting battle STATE.** That is external consistency at the
OBSERVER's abstraction rather than the byte level — it would dissolve the `return102` class entirely
(poke-env resolves both forms) instead of allowlisting it on an assumption that is currently untested.

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
  5. **CRIT_ITEM — WIRED (`gen3_crit_item_v1`).** Scope Lens (+1 unconditional) / Lucky Punch
     (+2 Chansey) / Stick (+2 Farfetch'd) fold `onModifyCritRatio critRatio + N` into
     `turn.rs::effective_crit_ratio` (the Focus Energy precedent — DRAW-FREE, only the `CRIT_MULT`
     denominator index shifts; the crit `randomChance(1, denom)` draw COUNT is unchanged). Data:
     the `critBoost {boost, onlySpecies}` field in `gen3_items.json` (`ItemData.crit_boost`,
     extractor + `dump_gen3_mechanics.js --check` drift gate). Species-gated via `user.species.id`.
     `leek` is the gen8 rename — NOT gen3-legal. 0 team-carry (Farfetch'd/Chansey aren't gen3 OU),
     so admitting `scopelens`/`luckypunch`/`stick` to `MODELED_ITEMS` is byte-neutral on the e2e
     sample; proven by the CI1/CI2 pins (Stick crits vs a no-item control at the same seed → same
     seedAfter, the draw-free proof). Still UNMAPPED: DRAIN_ITEM (Shell Bell), BOOST_RESTORE (White
     Herb), CURE_ITEM (Mental Herb), SPEED_MOD (Macho Brace), TAKE_ITEM_GUARD (Mail).
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
     MOVE hit] — see the batch-4 note below). **FORECAST — the LAST member — is DONE**
     (`gen3_forecast_v1`, ROUND 35 above): the forme+TYPE swap at every WeatherChange site
     (incl. the previously-missing UNCONDITIONAL expiry draw — the T1 8-vs-7 fix), the entrant
     `onStart`, the start window, the silent `clearVolatile` revert, and the
     reporting surfaces reading the construction `base_species_id` (the forme name appears on
     the wire ONLY in the `|-formechange|` line). The old construction FAIL-LOUD
     (`gen3_forecast_failloud_v1`) is RETIRED — pinned by its negative control
     `forecast_castform_builds_now_that_forecast_is_modeled` — and
     `REJECT_ABILITIES`/`REJECT_SPECIES` are EMPTY (kept as the seam for the next ability
     deferral): gen3-randbats Castform teams now reach the port. The ability class census is
     therefore COMPLETE: every gen-3 ability is modeled or verified-no-op, none fail-loud.
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
    rain/sun/hail — the probe diverges — not a no-op; now **FAIL-LOUD** in `state::from_set` + fuzz-rejected —
    see the deferred fail-loud section below). DEFERRED to batch 2: the DRAW-BEARING procs
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

## Per-mechanic coverage records — where the detail lives

Each gen-3 mechanic below was modelled in its own coverage round, with a differential gate and
revert-verified regression pins. Those records are **CLOSED** and live verbatim in
[`designs/rust_sim/port_build_log.md`](../../designs/rust_sim/port_build_log.md) — read the one
you are debugging rather than loading all of them here.

| mechanic | its round |
|---|---|
| Damage | Damage: the single-hit-physics gate (omniscient-oracle differential) |
| Fixed-damage moves | Fixed-damage moves: the `damage:` / `damageCallback` gate (per-seed PER-DECISION STATE+HP+STATUS+SEED differential) |
| Full battle | Full battle: the to-WIN/LOSS RNG-consumption gate (per-seed PER-DECISION STATE+SEED+winner differential) |
| Multi-turn | Multi-turn: the cross-turn RNG-consumption gate (per-seed STATE+SEED differential) |
| PP tracking + Struggle | PP tracking + Struggle (the first brick of `LegalActions`): per-decision STATE+HP+STATUS+PP+SEED differential |
| Phazing | Phazing: the Roar / Whirlwind forced-random-switch gate (per-seed PER-DECISION STATE+HP+SPIKES-LAYERS+DRAG-SPECIES+SEED differential) |
| Protect / Detect | Protect / Detect: the stall-draw + move-block gate (per-seed PER-DECISION STATE+HP+STATUS+STALL-COUNTER+SEED differential) |
| Recovery moves | Recovery moves: the self-heal / Rest draw gate (per-seed PER-DECISION STATE+HP+STATUS+SEED differential) |
| SNATCH | SNATCH: the LAST unmodeled gen-3 status move (→ 722/722) |
| Secondary effects + onBeforeMove status | Secondary effects + onBeforeMove status: the per-move-draw-bracket gate (per-seed PER-DECISION STATE+STATUS+SEED differential) |
| Setup moves | Setup moves: the self-targeting STAT-BOOST draw gate (per-seed PER-DECISION STATE+BOOST-STAGE+SEED+first-mover differential) |
| Spikes | Spikes: the entry-hazard + side-condition gate (per-seed PER-DECISION STATE+HP+SPIKES-LAYERS+SEED differential) |
| Status moves | Status moves: the standalone-status-MOVE draw gate (per-seed PER-DECISION STATE+STATUS+SEED differential) |
| Switch-in events | Switch-in events: the `>start` event gate (post-switch-in sim differential) |
| TRICK | TRICK: the item-swap move (`gen3_trick_v1`) |
| Taunt + Disable | Taunt + Disable: the move-SELECTION-restriction gate (per-decision STATE+TAUNT+DISABLED-SLOT+SEED differential) |
| Trapping | Trapping: the SWITCH-legality gate (per-decision STATE+per-side-TRAPPED+SEED differential) |
| YAWN | YAWN: the delayed-sleep move (`gen3_yawn_v1`) |
