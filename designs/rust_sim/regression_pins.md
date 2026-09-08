# The regression-pin register — bug -> pin, with its ground-truth probe

<!-- Lifted verbatim out of `src/rust_sim/CLAUDE.md` on 2026-09-08 (the leaf split: 2,659 -> a command card). This tree is ALWAYS-CURRENT: state the truth, never narrate a change. The leaf keeps the rule, the command and the hazard; this file keeps the detail. -->

`tests/regression_test.rs` carries one NAMED, revert-verified pin per engine bug the differential
gates found. This document is the full bug -> pin map plus the per-family notes naming each pin's
ground-truth probe. The leaf keeps the PRACTICE (every surfaced bug becomes a pin; every pin is
revert-verified) and the two assertion styles.

---


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

