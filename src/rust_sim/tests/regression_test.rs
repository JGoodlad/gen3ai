//! Regression tests — the EDGE CASES the e2e fuzz capstone found, PINNED
//! DETERMINISTICALLY.
//!
//! Several real-team engine bugs were, until now, covered ONLY by the e2e_fuzz
//! golden (`tests/e2e_fuzz_test.rs`) — regenerated each layer, with the specific
//! repro buried in 220 random battles, so it was not a STABLE, NAMED pin. This file
//! backfills a DEDICATED, NAMED, SELF-DOCUMENTING regression test per bug, each a
//! CONSTRUCTED scenario (explicit teams + seed + scripted choices via the public
//! `Battle::start_with_switchins` / `run_turn` / `run_full_battle` harness), so a
//! future change can never SILENTLY reintroduce the bug. Mirrors the style of
//! `tests/residual_faint_test.rs` (deterministic blocker regressions).
//!
//! Each test's NAME + doc comment states WHICH bug it pins + what the WRONG
//! (pre-fix) behaviour was. STATE pins assert the FIXED hp/status/boost directly.
//! DRAW-COUNT pins assert the post-decision PRNG SEED against the REAL-Showdown
//! ground truth captured by `harness/probe_regression_rng.js` /
//! `harness/probe_residual_order_rng.js` (the seeds below are copied verbatim from
//! those probes' output) — so a re-introduced desync (a wrong tie-shuffle draw
//! count) trips the seed assertion. The hacked sets (arbitrary abilities on any mon)
//! use `gen3customgame`, the same trick the harness + `residual_faint_test.rs` use.
//!
//! Bug → pin map:
//!   #2  Intimidate respects onTryBoost immunity  → `intimidate_into_clear_body_is_a_noop`
//!   #3a residual faintMessages + if(ended)return  → `burn_dot_self_ko_ends_before_foe_leftovers`
//!   #3b cached-speed paralysis (para-while-active) → `para_while_active_keeps_full_cached_speed_seed`
//!   #4  Toxic stage reset on switch-in            → `toxic_stage_resets_on_switch_in`
//!   #6  residual handler GATHER order             → `residual_gather_order_status_before_leftovers`
//!   #7  forced-replacement updateSpeed-on-commit  → `forced_replacement_recaches_speed_seed`
//! (#1 Water/Volt Absorb heal-on-hit + #5 absorb accuracy-gating are already pinned
//!  by `water_absorb_heals_on_hit_but_not_on_a_miss` in `src/turn.rs`.)
//!
//! PHAZE layer (Roar / Whirlwind) edge cases this layer surfaced — each a draw-COUNT or
//! composition gotcha that would SILENTLY desync if modeled wrong (ground-truth seeds from
//! `harness/probe_phaze_regression_rng.js`):
//!   P1  phaze accuracy IS drawn (gen-3 acc 100, NOT never-miss) + the n=1 `sample` draws
//!                                                  → `phaze_draws_accuracy_then_n1_sample_seed`
//!   P2  a phaze with NO eligible foe draws ONLY the accuracy roll (no `sample`)
//!                                                  → `phaze_fail_draws_only_accuracy_no_sample_seed`
//!   P3  a phaze that drags into a 3-layer-Spikes KO faints the dragged mon on entry →
//!       forces a NORMAL replacement (the composition) → `phaze_drag_into_a_spikes_ko_chains_a_replacement`
//!
//! LEECH SEED layer edge cases — the RESIDUAL-ORDER subtleties (the risk area: a wrong leech
//! subOrder / gather / tie position desyncs the residual tie-shuffle + the HP composition).
//! Ground-truth seeds + HP from `harness/probe_leechseed_regression_rng.js`:
//!   L1  the 4-WAY residual ORDER (sand[o=8] → Leftovers[s=4] → LEECH[s=5] → burn[s=6]) on the
//!       SAME seeded mon — a wrong leech subOrder re-orders heal/drain → divergent HP
//!                                                  → `leech_residual_order_leftovers_sand_burn`
//!   L2  BOTH actives seeded at EQUAL speed → the two leech handlers TIE (order 10 sub 5) →
//!       one residual tie-shuffle draw; a wrong subOrder / missing handler desyncs the SEED
//!                                                  → `leech_handler_tie_at_equal_speed_draws_one_shuffle`
//!   L3  the SEEDER-FAINTED gate — a leech whose seeder's active is fainted does NOTHING (no
//!       drain, no heal) → the seeded mon's HP is unchanged that turn
//!                                                  → `leech_seeder_fainted_skips_the_drain`
//!
//! EXPLOSION / SELF-DESTRUCT layer — the gen-3 self-KO is UNCONDITIONAL and precedes the hit
//! (`useMoveInner` faints the user BEFORE `trySpreadMoveHit`), so the user faints THROUGH a
//! Protect / a Ghost immunity / a sub / a miss — and the self-KO is DRAW-FREE (only the acc/
//! crit/dmg draws fire; the resulting faint changes pokemon_left / who-acts). Ground-truth
//! seeds from `harness/probe_explosion_regression_rng.js`:
//!   E1  Explosion into a PROTECT — the move is BLOCKED (no foe damage) but the USER STILL
//!       FAINTS (foe at full HP + Protect up) → `explosion_into_a_protect_the_user_still_faints`
//!   E2  Explosion into a GHOST (Normal-immune) — no foe damage but the USER STILL FAINTS
//!                                                  → `explosion_into_a_ghost_the_user_still_faints`
//!   E3  Explosion BREAKS a SUBSTITUTE + the USER STILL FAINTS — the sub is gone, the subber's
//!       mon HP is UNCHANGED (no carry) → `explosion_breaks_a_substitute_and_the_user_still_faints`
//!   E4  a MUTUAL Explosion (both last mons) is a gen-3 double-faint TIE (both faint, win(None))
//!                                                  → `mutual_explosion_is_a_double_faint_tie`
//!
//! DOUBLE-REPLACEMENT CASCADE layer (found when Explosion was re-admitted to the e2e capstone,
//! e2e_9 / e2e_194 — the double-faint mechanic Explosion is the common trigger for). Ground truth
//! from `harness/probe_double_replacement_cascade_regression_rng.js` /
//! `harness/probe_confusion_choiceband_regression_rng.js`:
//!   DR1 a double-faint → double-replacement where p1's fresh entrant faints on its OWN Spikes
//!       (its runSwitch FIRST) drops the FOE's still-pending runSwitch (gen-3 `cancelAction` over
//!       getAllActive) → the foe entrant is NOT re-chipped (stays FULL HP) — STATE + SEED pin
//!                                                  → `double_replacement_cascade_does_not_rechip_the_other_sides_entrant`
//!   DR2 a confused Choice-Band mon's self-hit folds Choice Band (gen-4 confusion runs the FULL
//!       getDamage) → the CB-boosted self-hit HP — STATE + SEED pin
//!                                                  → `confusion_self_hit_applies_choice_band`
//!
//! TAUNT + DISABLE layer (`gen3_taunt_disable_v1`) — the move-SELECTION restriction. Ground
//! truth from `harness/probe_taunt_disable_regression_rng.js` (+ the duration/onBeforeMove
//! semantics from `probe_disable_full_lifecycle.js` / `probe_taunt_duration_branch.js` /
//! `probe_taunt_disable_onbeforemove_rng.js`):
//!   TD1 a landed Taunt restricts the target's Status moves for EXACTLY the sim's window
//!       (queued move cant'd draw-free; ONE restricted selection; then FREE) — STATE
//!       (move_usable) + SEED pin  → `taunt_blocks_status_move_selection_for_the_sim_window_draw_free`
//!   TD2 the Disable STORED duration per branch — FASTER disabler (willMove TRUE) stores
//!       random(2,6), SLOWER stores random(2,6)+1 — pinned at the exact FREE-UP boundary +
//!       per-decision SEEDs (trips on +1/-1 off-by-ones AND on dropping the branch)
//!                                                  → `disable_duration_stored_per_branch_matches_sim`
//!   TD3 Disable (the only attack) + Taunt (the Status moves) → FORCED Struggle — STATE
//!       (must_struggle + the Struggle's HP/recoil) + SEED pin
//!                                                  → `taunt_plus_disable_forces_struggle`
//!   TD4 the onBeforeMove PRIORITY ordering vs paralysis — taunt (priority 0) cants AFTER
//!       the para roll (a taunted+paralyzed queued status move DRAWS the randomChance(1,4)
//!       first); disable (priority 7) cants BEFORE it (NO para roll) — SEED pins both ways;
//!       the 720-run golden does NOT cover the paralyzed case, so this is the only gate
//!                                                  → `taunt_and_disable_onbeforemove_priority_vs_paralysis`
//!   TD5 Disable into a 0-PP lastMove — the gen4-inherited onStart 0-PP guard REJECTS the
//!       volatile AFTER the accuracy + random(2,6) draws (draws consumed, `-fail`, NO
//!       volatile/`-start`/residual handler) — STATE (disabled_slot / disable None) + SEED
//!       pin; ground truth `harness/probe_disable_zero_pp_rng.js`
//!                                                  → `disable_into_a_zero_pp_lastmove_fails_draws_but_no_volatile`
//!   T1  TRAPPING (`gen3_trapping_v1`) — Arena Trap REJECTS a grounded foe's voluntary
//!       switch DRAW-FREE (the switch mirror of the reject-and-re-request gate: the
//!       scripted Switch decision is SKIPPED, boundary open, seed untouched) — STATE
//!       (is_trapped / species / trapped columns) + SEED pins; ground truth
//!       `harness/probe_trapping_regression_rng.js`
//!                                                  → `arena_trap_rejects_a_grounded_foes_switch_draw_free`
//!   T2  Arena Trap does NOT trap Flying (Zapdos) / Levitate (Gengar) — their voluntary
//!       switches are ACCEPTED (gen-3 grounded == not-Flying && not-Levitate) — STATE +
//!       SEED pins                                   → `arena_trap_does_not_trap_flying_or_levitate`
//!   T3  Magnet Pull traps STEEL only — the MAGNETON MIRROR mutual-traps AND draws the
//!       endTurn TrapPokemon+MaybeTrapPokemon tie-shuffles (gen3 magnetpull is onAny → 2
//!       tied handlers per event → **4 draws per endTurn** in the speed-tied mirror,
//!       pinned by the splash-turn seeds); the non-Steel control switches out freely —
//!       STATE + SEED pins                           → `magnet_pull_traps_steel_only`
//!   T4  Roar DRAGS a trapped mon out (phaze bypasses trapping — only the VOLUNTARY
//!       switch is gated) — STATE (drag species/flag) + SEED pins
//!                                                  → `roar_drags_a_trapped_mon_out`
//!   T5  a grounded GHOST (Sableye) IS trapped in Showdown-gen3 (NO `trapped` type
//!       immunity in the gen3 dex — the cartridge gen6+ escape doesn't exist here) —
//!       STATE + SEED pins        → `grounded_ghost_is_trapped_by_arena_trap_in_showdown_gen3`
//!   I1  gen-3 INTIMIDATE vs SUBSTITUTE — a mid-battle Intimidate switch-in does NOT drop
//!       a subbed foe's Atk (the gen3 mod's substitute skip; seed-neutral). Surfaced by
//!       `gen3_trapping_v1`'s e2e regen (e2e_171/e2e_204); ground truth
//!       `harness/probe_intimidate_substitute_rng.js`
//!                                                  → `intimidate_into_a_substitute_is_a_noop`
//!   I2  gen-3 INTIMIDATE vs a FORCED REPLACEMENT (`gen3_intimidate_forced_replacement_v1`) —
//!       a deferred switch-in Intimidate (a VOLUNTARY switch whose intended foe FAINTED +
//!       force-replaced mid-turn, the Pursuit→Destiny-Bond→replacement composition) must NOT
//!       drop the replacement's Atk (Showdown resolves the entrant Intimidate INLINE against
//!       the fainted original). Repro ab_1381_0. STATE pin
//!                                                  → `intimidate_entrant_does_not_drop_a_forced_replacement_foe`
//!   HP-TAUNT  a Taunted mon keeps its BARE `hiddenpower` selectable
//!       (`gen3_iv_derived_hidden_power_bp_v1` legality sibling; repro ab_233_8) — STATE pin
//!                                                  → `taunt_does_not_block_a_bare_hidden_power`
//!   HP-MIMIC  a Mimic of Hidden Power re-derives the type from the MIMICKER's OWN IVs
//!       (`gen3_mimic_hidden_power_type_v1`; repro ab_777_3) — STATE pin
//!                                                  → `mimic_of_hidden_power_uses_the_mimickers_own_type`

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::{Status, Weather};
use pokesim::turn::{Choice, RequestKind, ScriptDecision};

fn dex() -> Dex {
    Dex::for_gen(3)
}

/// `gen3customgame` options (no clauses) with an explicit seed — the format the
/// harness probes use, so the captured ground-truth seeds line up bit-for-bit.
fn opts_cg(p1: &str, p2: &str, seed: &str) -> BattleOptions {
    BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some(seed.to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.to_string()) },
    }
}

/// Parse a `m,n,o,p` seed string into the comparable 4-word seed the differential
/// asserts (the engine stores it; `prng_seed()` re-emits the same comma string).
fn seed_str(s: &pokesim::prng::PrngSeed) -> String {
    s.clone()
}

// ============================================================================
// #2 — Intimidate respects the foe's onTryBoost immunity (Clear Body / White
//      Smoke / Hyper Cutter). The WRONG behaviour: the lead/switch-in Intimidate
//      dropped the foe's Atk stage even into a Clear Body mon (Intimidate-into-
//      Metagross should be a NO-OP). Pure STATE pin (no PRNG): the foe's Atk boost
//      must stay 0; a non-immune control drops to -1.
// ============================================================================

/// #2: Intimidate into a Clear Body / White Smoke / Hyper Cutter foe is a no-op.
/// WRONG (pre-fix): the foe's Atk dropped to -1 regardless of its ability. The fix
/// (`event::intimidate_on_start`) gates the drop on the foe `onTryBoost` immunity.
#[test]
fn intimidate_into_clear_body_is_a_noop() {
    let d = dex();
    // p1 Salamence has Intimidate (gen3customgame accepts the hacked ability).
    let salamence = "Salamence|||intimidate|dragonclaw|Adamant|,252,,,,252|||||";

    // (a) IMMUNE foes — Clear Body / White Smoke (all drops) + Hyper Cutter (Atk):
    for foe_ability in ["clearbody", "whitesmoke", "hypercutter"] {
        let foe = format!("Metagross|||{foe_ability}|meteormash|Adamant|,252,,,,|||||");
        let battle =
            Battle::start_with_switchins(&opts_cg(salamence, &foe, "1,2,3,4"), &d).expect("start");
        let st = battle.state().expect("state");
        // boosts index 0 == Atk. The Intimidate must NOT have dropped it.
        assert_eq!(
            st.sides[1].pokemon[0].boosts[0], 0,
            "Intimidate into {foe_ability} must be a NO-OP (the foe's Atk stays 0); \
             under the bug it would be -1"
        );
        // And our own Salamence is untouched (Intimidate drops the FOE only).
        assert_eq!(st.sides[0].pokemon[0].boosts[0], 0, "the Intimidate user's Atk is unchanged");
    }

    // (b) CONTROL — a non-immune foe DOES drop to -1 (proves the test would catch a
    //     fix that wrongly blocks ALL Intimidates).
    let plain_foe = "Snorlax|||immunity|bodyslam|Adamant|252,252,,,,|||||";
    let battle =
        Battle::start_with_switchins(&opts_cg(salamence, plain_foe, "1,2,3,4"), &d).expect("start");
    let st = battle.state().expect("state");
    assert_eq!(
        st.sides[1].pokemon[0].boosts[0], -1,
        "Intimidate into a non-immune foe DROPS its Atk to -1 (the control)"
    );
}

// ============================================================================
// #4 — Toxic STAGE RESET on switch-in (`tox.onSwitchIn`: effectState.stage = 0). A
//      badly-poisoned mon that pivots OUT and back must RESUME at stage 0 (the next
//      residual ramps 0→1), NOT its prior high stage. WRONG (pre-fix): the stage did
//      NOT reset, so the returned mon over-chipped at its old high stage. STATE pin:
//      after a switch-out + switch-back, the returned mon's Toxic stage is 0 (then
//      the residual ramps it to 1, chipping a SINGLE maxhp/16, not stage×maxhp/16).
// ============================================================================

/// #4: a badly-poisoned mon that switches OUT and back resets its Toxic stage to 0.
/// WRONG (pre-fix): the stage persisted, so the returned mon resumed chipping at its
/// old high stage (e.g. stage 6 → 6·maxhp/16 instead of 1·maxhp/16). The fix resets
/// `Status::Toxic(_)` → `Toxic(0)` in `execute_switch`.
#[test]
fn toxic_stage_resets_on_switch_in() {
    let d = dex();
    // p1: a 2-mon team so the active can pivot out and back. Salamence (Toxic'd, a
    // Dragon/Flying that is GROUND-IMMUNE via Flying — and NOT Poison/Steel so it CAN
    // be badly poisoned) + a bench Gengar. p2: an Earthquake user (Ground) — immune to
    // the pivoting Salamence both before AND after the pivot, so the ONLY HP change to
    // Salamence is its own residual Toxic chip. (We pivot via the bench Gengar.)
    let p1 = "Salamence|||intimidate|dragonclaw,refresh|Adamant|,252,,,,252|||||\
              ]Gengar|||levitate|tackle|Timid|,,,252,,252|||||";
    let p2 = "Flygon|||levitate|earthquake|Adamant|,252,,,,252|||||";

    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "1,2,3,4"), &d).expect("start");
    let st = battle.state_mut().expect("state");

    // INJECT: Salamence badly poisoned, already at a HIGH toxic stage (5) — the state a
    // few residual ticks would have produced. (We set it directly so the scenario is a
    // clean, deterministic single repro, not a multi-turn ramp.)
    st.sides[0].pokemon[0].status = Some(Status::Toxic(5));
    let maxhp = st.sides[0].pokemon[0].maxhp;

    // Turn 1: p1 SWITCHES Salamence OUT (to bench Gengar). p2 Earthquake (immune into
    // Gengar/Levitate). The benched Salamence takes no residual chip (only the active
    // does). Gengar (Levitate) is also Ground-immune so no confounding damage.
    let dec1 = ScriptDecision::both(Choice::Switch(1), Choice::Move(0));
    // Turn 2: p1 SWITCHES Salamence BACK IN (team slot 1 after the swap). On switch-in
    // its Toxic stage resets to 0. The end-of-turn residual then ramps 0→1 and chips
    // exactly ONE maxhp/16. p2 Earthquake is Flying-immune → no confounding damage.
    let dec2 = ScriptDecision::both(Choice::Switch(1), Choice::Move(0));

    let _ = st.run_full_battle(&[dec1, dec2], &d);

    // After turn 2, Salamence is active again. Its Toxic stage must be 1 (reset to 0 on
    // switch-in, then ramped to 1 by the post-switch residual). Under the BUG it would
    // be 6 (the persisted 5 ramped to 6).
    let mence = &st.sides[0].pokemon[st.sides[0].active];
    assert_eq!(mence.species_id, "salamence", "Salamence is back in the active slot");
    assert_eq!(
        mence.status,
        Some(Status::Toxic(1)),
        "Toxic stage RESET to 0 on switch-in, then ramped to 1 by the residual \
         (under the bug it would be Toxic(6))"
    );
    // And the residual chipped exactly maxhp/16 (stage 1), NOT 6·maxhp/16.
    let chip = (maxhp / 16).max(1);
    assert_eq!(
        mence.hp,
        maxhp - chip,
        "the post-reset residual chips ONE maxhp/16 (stage 1); under the bug it would \
         chip 6·maxhp/16 = {}",
        6 * chip
    );
}

// ============================================================================
// #3a — residual faintMessages PER HANDLER + `if (this.ended) return`. A FAST burned
//       mon's DoT self-KO (status DoT subOrder 6, but SPEED outranks subOrder) ENDS
//       the game BEFORE a SLOWER foe's Leftovers (subOrder 4) heal — so the foe does
//       NOT get its Leftovers tick that game-ending turn. WRONG (pre-fix): all
//       residual handlers applied then faints processed ONCE, so the foe's Leftovers
//       wrongly healed before the game ended. STATE pin: the foe's HP is UNCHANGED.
//       (Ground-truth seed from probe_residual_order_rng.js also asserted.)
// ============================================================================

/// #3a: a fast burned mon's residual self-KO ends the game before the slower foe's
/// Leftovers ticks. WRONG (pre-fix): the foe healed `maxhp/16` first (faints were
/// processed once, after every handler). The fix runs `faintMessages` + `if (ended)
/// return` PER handler, mirroring `fieldEvent('Residual')`'s while-loop.
#[test]
fn burn_dot_self_ko_ends_before_foe_leftovers() {
    let d = dex();
    // p1 Gengar (FAST, Timid, Ghost — its side's ONLY mon) burned at 1 HP: its burn
    // DoT self-KOs and, being the last mon, ENDS the game. p2 Gengar (SLOW, Brave,
    // Ghost) holds Leftovers, pre-chipped, NOT burned. Tackle is Ghost-immune BOTH
    // ways, so the residual is the SOLE HP change. (Probe seed [2,3,5,7] → initSeed.)
    let fast = "Gengar||leftovers|levitate|tackle|Timid|,,,252,,252|||||";
    let slow = "Gengar||leftovers|levitate|tackle|Brave|252,252,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(fast, slow, "40263,34842,41812,24710"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");

    // INJECT the post-move board the probe used: p1 burned at 1 HP, p2 at 200 HP.
    st.sides[0].pokemon[0].status = Some(Status::Burn);
    st.sides[0].pokemon[0].hp = 1;
    st.sides[1].pokemon[0].hp = 200;
    let foe_hp_before = st.sides[1].pokemon[0].hp;

    // Both use Tackle (slot 0) — Ghost-immune both ways → the residual is the only
    // effect. p1's burn DoT (max(maxhp/8,1) ≥ 1) self-KOs the 1-HP Gengar.
    let result = st.run_turn(0, 0, &d);

    // The burned Gengar self-KO'd; the game ended (its side is out of mons).
    assert!(st.sides[0].pokemon[0].fainted, "the 1-HP burned Gengar self-KO'd via burn DoT");
    assert_eq!(st.sides[0].pokemon[0].hp, 0, "at 0 HP");
    // THE FIX: the game-ending residual KO returned BEFORE the slower foe's Leftovers
    // ran — so the foe's HP is UNCHANGED (no maxhp/16 heal). Under the bug it would be
    // foe_hp_before + maxhp/16.
    assert_eq!(
        st.sides[1].pokemon[0].hp, foe_hp_before,
        "the foe's Leftovers must NOT tick once the burn self-KO ends the game; \
         under the bug it would heal maxhp/16"
    );
    // The game-ending faint defers the trailing Update + Quick Claw (no draw).
    assert!(!result.quick_claw_drawn, "a game-ending residual faint skips the Quick Claw");

    // GROUND-TRUTH SEED (probe_residual_order_rng.js): the post-turn PRNG seed matches
    // the real sim exactly — a re-introduced draw-count desync trips this.
    assert_eq!(
        seed_str(&st.prng_seed()),
        "2680,64033,5576,29768",
        "post-turn seed == the real Showdown seed for this constructed residual-faint turn"
    );
}

// ============================================================================
// #6 — residual handler GATHER order. The residual handler-sort's tie-group
//      Fisher-Yates shuffle PERMUTES handlers in their PRE-SORT (gather) order, so
//      the status-DoT handler must be GATHERED before Leftovers per mon (Showdown's
//      findPokemonEventHandlers gathers STATUS → volatile → item). WRONG (pre-fix):
//      the port gathered Leftovers BEFORE the status DoT, so a 2-mon DoT+Leftovers
//      speed-tie's shuffle permuted to the WRONG side-order → a divergent post-turn
//      STATE + SEED. Constructed: a Gengar mirror under SAND, both burned + Leftovers
//      at IDENTICAL speed (tie); the foe (its side's last mon) at a burn-DoT-lethal
//      HP so its DoT ends the game. STATE + SEED pin (ground-truth from the probe).
// ============================================================================

/// #6: the residual handler GATHER order (status DoT before Leftovers, per mon) drives
/// the tie-group shuffle's permutation. WRONG (pre-fix): Leftovers was gathered first,
/// so a 2-mon DoT+Leftovers speed tie permuted to the wrong side-order, desyncing the
/// post-turn HP STATE (and the seed). The fix pushes the StatusDot handler before
/// Leftovers, mirroring `findPokemonEventHandlers`'s status-before-item order.
#[test]
fn residual_gather_order_status_before_leftovers() {
    let d = dex();
    // p1 Gengar has Sand Stream (sets permanent sand at switch-in) + Leftovers; p2
    // Gengar (the foe, its side's ONLY mon) has Leftovers. Both Timid 252 HP/Spe →
    // IDENTICAL speed (the residual handler tie). Tackle is Ghost-immune both ways.
    let p1 = "Gengar||leftovers|sandstream|tackle|Timid|252,,,252,,252|||||";
    let p2 = "Gengar||leftovers|levitate|tackle|Timid|252,,,252,,252|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "35506,34554,10206,39717"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    assert_eq!(st.field.weather, Some(Weather::Sand), "Sand Stream set permanent sand");

    // INJECT the probe's post-move board: p1 (us) burned at full HP (324), p2 (foe)
    // burned at 40 HP — its burn DoT + sand chip ends the game.
    st.sides[0].pokemon[0].status = Some(Status::Burn);
    st.sides[0].pokemon[0].hp = 324;
    st.sides[1].pokemon[0].status = Some(Status::Burn);
    st.sides[1].pokemon[0].hp = 40;

    let _ = st.run_turn(0, 0, &d);

    // GROUND TRUTH (probe_residual_order_rng.js, real Showdown): the foe's DoT/sand KO'd
    // it (game over) and OUR Gengar ends at EXACTLY 284. A wrong gather order permutes
    // the tie-shuffle differently → a different surviving-mon HP and/or seed.
    assert!(st.sides[1].pokemon[0].fainted, "the foe Gengar fainted to its residual");
    assert_eq!(
        st.sides[0].pokemon[0].hp, 284,
        "our Gengar's post-residual HP must be 284 (the real-sim value); a wrong gather \
         order permutes the tie-shuffle and desyncs this STATE"
    );
    assert_eq!(
        seed_str(&st.prng_seed()),
        "20457,61479,13178,12055",
        "post-turn seed == the real Showdown seed; a wrong gather order desyncs it too"
    );
}

// ============================================================================
// #3b — the CACHED `pokemon.speed` model for PARALYSIS. A mon paralyzed WHILE active
//       keeps its FULL turn-start cached speed through the move-phase
//       `eachEvent('Update')` tie-shuffles (it only drops to para-speed at the
//       residual's `updateSpeed`), while one that SWITCHES IN paralyzed ties on its
//       para-speed AT ONCE. WRONG (pre-fix): the shuffles read the LIVE speed, so a
//       mon paralyzed mid-turn dropped to para-speed immediately → a different
//       tie-shuffle draw count → a divergent seed. Constructed: a Jirachi mirror
//       (equal full speed → tie). Turn 1 p1 Thunder-Waves p2 (para'd mid-turn); the
//       post-para shuffles must still read p2's STALE full speed (still tying p1).
//       SEED pin (ground truth from probe_regression_rng.js).
// ============================================================================

/// #3b: a mon paralyzed WHILE active keeps its full turn-start cached speed through
/// the rest of the turn's `eachEvent` tie-shuffles. WRONG (pre-fix): the shuffles
/// read the live para-speed, flipping a speed tie and changing the draw count → a
/// divergent post-turn seed. The fix reads `MonState::cached_speed` (refreshed only at
/// turn-start / residual / switch-in), not the live `effective_speed`.
#[test]
fn para_while_active_keeps_full_cached_speed_seed() {
    let d = dex();
    // A Jirachi MIRROR: both Timid 252 HP/252 Spe → IDENTICAL full speed (the tie that
    // makes every `eachEvent('Update')` shuffle DRAW). On turn 1 p1 Thunder-Waves p2,
    // paralyzing it MID-TURN; the remaining shuffles read p2's STALE full speed (still
    // tying p1) — so turn 1's draw count + post-turn seed match the real sim. Under the
    // bug (live read), p2 would drop to its para speed mid-turn → no tie → fewer draws.
    let p1 = "Jirachi|||serenegrace|thunderwave,swift|Timid|252,,,,,252|||||";
    let p2 = "Jirachi|||serenegrace|swift,thunderwave|Timid|252,,,,,252|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "52941,53060,64922,35342"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");

    // Turn 1: p1 Thunder Wave (slot 0) ; p2 Swift (slot 0).
    let dec1 = ScriptDecision::both(Choice::Move(0), Choice::Move(0));
    // Turns 2-3: both Swift — the cross-turn carry (p2 now para'd at turn start, so its
    // cached speed IS para-speed there; this confirms the residual-refresh handoff).
    let dec2 = ScriptDecision::both(Choice::Move(1), Choice::Move(0));
    let dec3 = ScriptDecision::both(Choice::Move(1), Choice::Move(0));

    let out = st.run_full_battle(&[dec1, dec2, dec3], &d);

    // p2 was paralyzed by the turn-1 Thunder Wave.
    assert_eq!(
        st.sides[1].pokemon[0].status,
        Some(Status::Paralysis),
        "p2 Jirachi was paralyzed by the turn-1 Thunder Wave"
    );

    // GROUND TRUTH (probe_regression_rng.js, real Showdown), per decision boundary:
    //   turn 1 seedAfter = 27654,2204,30943,58907  (the para-mid-turn stale-speed turn)
    //   turn 2 seedAfter = 34055,58601,16184,3801
    //   turn 3 seedAfter = 40070,2014,43377,12551
    let expected = [
        "27654,2204,30943,58907",
        "34055,58601,16184,3801",
        "40070,2014,43377,12551",
    ];
    assert_eq!(out.decisions.len(), 3, "three move decisions recorded");
    for (i, exp) in expected.iter().enumerate() {
        assert_eq!(
            seed_str(&out.decisions[i].seed_after),
            *exp,
            "decision {i} post-turn seed == the real Showdown seed; under the bug the \
             para-mid-turn turn-1 shuffle count differs and this desyncs"
        );
    }
}

// ============================================================================
// #7 — the FORCED-REPLACEMENT updateSpeed-on-commit. `commitChoices()` runs
//      `updateSpeed()` at its top on EVERY choice submit INCLUDING a mid-turn forced
//      replacement, so a foe paralyzed MID-TURN drops to its para-speed before the
//      resumed turn-tail's `eachEvent('Update')` tie-shuffles read it. WRONG
//      (pre-fix): the port did NOT re-cache on the forced-replacement commit, so the
//      para'd foe kept its STALE full speed → it spuriously TIED with the fresh
//      entrant → a phantom extra shuffle draw → a divergent seed. Constructed: p1's
//      low-HP lead Thunder-Waves p2 (para mid-turn) and is KO'd by p2's Swift → p1
//      forced to replace; the para'd p2's STALE full speed equals the entrant's, so
//      the re-cache (→ para-speed) is what avoids a phantom tie. SEED pin.
// ============================================================================

/// #7: a forced-replacement commit re-caches both actives' speed, so a foe paralyzed
/// mid-turn reads its para-speed (not its stale full speed) in the resumed turn-tail's
/// tie-shuffles. WRONG (pre-fix): the stale full speed spuriously tied the fresh
/// entrant → a phantom shuffle draw → a divergent seed. The fix calls `update_speed()`
/// on the forced-replacement commit (mirroring `commitChoices`).
#[test]
fn forced_replacement_recaches_speed_seed() {
    let d = dex();
    // p1: a 2-mon Jirachi team. Lead (Timid, full 328) Thunder-Waves p2 THIS turn, then
    // p2's Swift KOs the low-HP (injected) lead → p1 forced to replace with Jirachi-B
    // (Timid, full 328). p2: a single Timid Jirachi (full 328 == B; para-speed 82). The
    // para'd p2's STALE full speed (328) equals B's, so without the re-cache the resumed
    // tail's Update shuffles spuriously tie p2 with B → a phantom draw.
    let p1 = "Jirachi|||serenegrace|thunderwave,swift|Timid|252,,,,,252|||||\
              ]Jirachi|||serenegrace|swift|Timid|252,,,,,252|||||";
    let p2 = "Jirachi|||serenegrace|swift,thunderwave|Timid|252,,,,,252|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "42782,54377,52057,58231"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");

    // INJECT: p1's lead at 8 HP (the probe's board) so p2's Swift KOs it this turn.
    st.sides[0].pokemon[0].hp = 8;

    // Turn 1 (move): p1 Thunder Wave (slot 0) ; p2 Swift (slot 0). p1's 8-HP lead is
    // KO'd; p2 is paralyzed MID-TURN. → p1 forced replacement.
    let t1 = ScriptDecision::both(Choice::Move(0), Choice::Move(0));
    // Forced switch: p1 brings in Jirachi-B (team slot 1). On commit, updateSpeed
    // re-caches p2 to its para-speed (82), so it no longer ties B (328).
    let sw = ScriptDecision::one(0, Choice::Switch(1));
    // Turns 2-3 (move): both Swift — the tie-shuffle count on B(328) vs p2(para 82).
    let t2 = ScriptDecision::both(Choice::Move(0), Choice::Move(0));
    let t3 = ScriptDecision::both(Choice::Move(0), Choice::Move(0));

    let out = st.run_full_battle(&[t1, sw, t2, t3], &d);

    // p2 paralyzed; p1 replaced its KO'd lead with Jirachi-B.
    assert_eq!(
        st.sides[1].pokemon[0].status,
        Some(Status::Paralysis),
        "p2 Jirachi was paralyzed mid-turn by the turn-1 Thunder Wave"
    );

    // GROUND TRUTH (probe_residual_order_rng.js, hp=8 sweep, real Showdown):
    //   decision 0 (the turn-1 move boundary, recorded at the faint pause) seedAfter =
    //       60657,49635,61056,62686
    //   decision 1 (the forced-switch boundary)               seedAfter = 16931,35834,30244,2137
    //   decision 2 (next move turn)                           seedAfter = 40073,17316,4488,22151
    //   decision 3 (next move turn)                           seedAfter = 12137,14564,61669,17829
    let expected = [
        "60657,49635,61056,62686",
        "16931,35834,30244,2137",
        "40073,17316,4488,22151",
        "12137,14564,61669,17829",
    ];
    assert_eq!(out.decisions.len(), 4, "four decision boundaries (move, forced-switch, move, move)");
    for (i, exp) in expected.iter().enumerate() {
        assert_eq!(
            seed_str(&out.decisions[i].seed_after),
            *exp,
            "decision {i} seed == the real Showdown seed; WITHOUT the forced-replacement \
             re-cache the para'd foe stays at its stale full speed → a phantom tie-shuffle \
             draw → this desyncs"
        );
    }
}

// ============================================================================
// P1 — gen-3 PHAZE (Roar / Whirlwind) draws its ACCURACY roll, then the n=1 `sample`.
//      The SURPRISE this layer surfaced: gen-3 Roar/Whirlwind resolve to `accuracy: 100`
//      (NOT `true`), so a phaze is NOT never-miss — it draws `randomChance(100,100)` (always
//      passes, but CONSUMES a draw). THEN, on a successful phaze, `dragIn` →
//      `getRandomSwitchable` → `sample` → `random(n)` draws ONE more — EVEN when n == 1
//      (`random(1)` returns 0 but still calls `rng.next()`). WRONG (a naive "phaze is
//      never-miss + n=1 is draw-free" model): the turn would draw 2 FEWER PRNG calls
//      → a divergent post-turn seed. Ground-truth seed from probe_phaze_regression_rng.js.
// ============================================================================

/// P1: a Roar with EXACTLY ONE eligible foe bench mon draws the ACCURACY roll
/// (`randomChance(100,100)`, gen-3 phaze acc 100) THEN the n=1 `sample` (`random(1)`, which
/// STILL draws) — then the end-of-turn Quick Claw. WRONG (pre-fix model): a phaze treated as
/// never-miss + n=1-sample-draw-free would draw 2 fewer calls → the post-turn seed diverges.
#[test]
fn phaze_draws_accuracy_then_n1_sample_seed() {
    let d = dex();
    // p1 SLOW Suicune (Relaxed, moves last at priority −6) Roars. p2 Blissey + ONE bench
    // Snorlax → exactly one eligible drag target (the n=1 case). Blissey uses Soft-Boiled at
    // full HP (a draw-free no-op fail), so the ONLY draws this turn are: the phaze accuracy
    // roll + the n=1 `sample` + the end-of-turn Quick Claw = 3. (Probe seedBefore matches
    // the init_seed here; we run ONE move turn and assert the post-turn seed.)
    let suicune = "Suicune|||pressure|roar,surf|Relaxed|252,,252,,,|||||";
    let p2 = "Blissey|||pressure|softboiled|Bold|252,,,,,|||||\
              ]Snorlax|||pressure|bodyslam|Careful|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(suicune, p2, "53303,35262,36397,29520"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");

    // Turn 1: p1 Roar (drags the lone Snorlax) ; p2 Soft-Boiled (full HP → fail, draw-free).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // The dragged-in mon is now the p2 active — the n=1 `sample` pulled the lone Snorlax.
    assert_eq!(
        st.sides[1].pokemon[st.sides[1].active].species_id, "snorlax",
        "the n=1 phaze dragged the lone eligible bench mon (Snorlax) into the active slot"
    );
    // Blissey (the phazed-OUT mon) is now on the bench (the array swap).
    assert!(
        st.sides[1].pokemon.iter().any(|m| m.species_id == "blissey"),
        "the phazed-out Blissey moved to the bench"
    );

    // GROUND TRUTH (probe_phaze_regression_rng.js, PHAZE-ACC+N1): the post-turn seed == the
    // real Showdown seed (draws = accuracy + n=1 sample + Quick Claw). A model that skipped
    // the accuracy roll OR treated the n=1 sample as draw-free trips this.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "8759,60273,17756,56077",
        "post-turn seed == the real Showdown seed for a Roar with one eligible bench \
         (the gen-3 acc-100 roll + the n=1 sample must BOTH draw)"
    );
}

// ============================================================================
// P2 — a PHAZE with NO eligible foe (the foe's last mon) draws ONLY the accuracy roll.
//      `forceSwitch` checks `canSwitch(target.side)`; with the foe's last mon alive it is
//      FALSE → NO `forceSwitchFlag`, NO `dragIn`, NO `sample` draw — the phaze just `-fail`s.
//      So the turn draws the accuracy roll + the Quick Claw = 2 (ONE fewer than the n=1 case
//      above — the same init seed). WRONG (a model that drew the sample on a fail): an extra
//      draw → the post-turn seed diverges. Ground-truth seed from probe_phaze_regression_rng.js.
// ============================================================================

/// P2: a Roar with NO eligible foe bench (the foe's last mon) draws ONLY its accuracy roll
/// (no `sample`). WRONG (pre-fix model): drawing the `sample` regardless of `canSwitch` would
/// add a draw → the post-turn seed diverges. The SAME init seed as P1 — but ONE fewer draw.
#[test]
fn phaze_fail_draws_only_accuracy_no_sample_seed() {
    let d = dex();
    // p1 SLOW Suicune Roars. p2 has ONLY Blissey (its last mon) → no eligible drag target →
    // the phaze FAILS draw-free (only the accuracy roll). Same init seed as P1: the n=1 case
    // drew 3, this fail draws 2 (no sample) → a DIFFERENT (one-fewer-draw) post-turn seed.
    let suicune = "Suicune|||pressure|roar,surf|Relaxed|252,,252,,,|||||";
    let blissey = "Blissey|||pressure|softboiled|Bold|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(suicune, blissey, "53303,35262,36397,29520"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");

    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // The phaze FAILED → Blissey stays active (no drag); both sides still have their mon.
    assert_eq!(
        st.sides[1].pokemon[st.sides[1].active].species_id, "blissey",
        "the failed phaze leaves the foe's last mon (Blissey) active — no drag"
    );
    assert_eq!(st.sides[1].pokemon_left, 1, "the foe still has its last mon");

    // GROUND TRUTH (probe_phaze_regression_rng.js, PHAZE-FAIL): the post-turn seed == the real
    // Showdown seed (draws = accuracy + Quick Claw, NO sample). It DIFFERS from P1's seed
    // (which drew the extra sample) — proving the fail-case no-draw is exact.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "41762,18770,8812,43906",
        "post-turn seed == the real Showdown seed for a FAILED Roar (foe's last mon): only \
         the accuracy roll draws, NOT the sample"
    );
}

// ============================================================================
// P3 — a PHAZE that drags a mon into a 3-layer-Spikes KO faints it ON ENTRY → forces a
//      NORMAL replacement (the composition: drag → runSwitch EntryHazard/Spikes → KO →
//      forced replacement). WRONG (a model that skipped the EntryHazard on a forced drag, or
//      didn't chain the forced replacement): the dragged mon survives / the battle doesn't
//      pause for a replacement. STATE + draw-COUNT + request-kind pin.
// ============================================================================

/// P3: a Roar that drags a pre-chipped grounded mon into 3 layers of Spikes — the spikes KO
/// it on entry → forces a NORMAL replacement. WRONG (pre-fix / a broken composition): the
/// dragged mon would not take spikes (drag skipped the runSwitch EntryHazard) or the battle
/// would not pause for a forced replacement. The drag's runSwitch is draw-free, so the turn
/// draws only the phaze accuracy + sample. Ground-truth seed from probe_phaze_regression_rng.js.
#[test]
fn phaze_drag_into_a_spikes_ko_chains_a_replacement() {
    let d = dex();
    // p1 Skarmory Roars. p2 Blissey + two lvl-1 grounded bench (Diglett/Sandshrew). We INJECT
    // 3 spikes layers on the p2 side + pre-chip the lvl-1 mons to 1 HP, so whichever lvl-1 mon
    // the random `sample` drags in is KO'd by the 3-layer spikes (floor(maxhp/4) >= 1) ON
    // ENTRY → p2 is forced to replace. (Same init seed [1,2,3,4] as the probe → the sample
    // drags Sandshrew.)
    let skarmory = "Skarmory|||keeneye|roar,drillpeck|Adamant|252,252,,,,|||||";
    let p2 = "Blissey|||pressure|softboiled|Bold|252,,,,,|||||\
              ]Diglett|||sandveil|scratch|Serious|||||1|\
              ]Sandshrew|||sandveil|scratch|Serious|||||1|";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(skarmory, p2, "30982,33910,19571,50263"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");

    // INJECT: 3 spikes layers on the p2 side + pre-chip the lvl-1 bench mons to 1 HP.
    st.sides[1].spikes = 3;
    for m in st.sides[1].pokemon.iter_mut() {
        if m.level == 1 {
            m.hp = 1;
        }
    }

    // Turn 1: p1 Roar drags a lvl-1 mon → spikes KO on entry → p2 forced replace. With this
    // init seed [1,2,3,4] the `sample` over the eligible bench [Diglett@1, Sandshrew@2] picks
    // Sandshrew (index 1) → target slot 2. The drag swaps active(0)↔2: Sandshrew→slot 0
    // (active, then KO'd by the spikes), Blissey→slot 2, Diglett stays@1. So the live bench
    // after the KO is Diglett@1 (and Blissey@2). The forced replacement switches to Diglett
    // (slot 1) — a known-live, non-active slot. Decision 0 = the move turn (recorded at the
    // pause); decision 1 = the p2 ForceSwitch boundary.
    let t1 = ScriptDecision::both(Choice::Move(0), Choice::Move(0));
    let rep = ScriptDecision::one(1, Choice::Switch(1)); // p2 replaces the KO'd drag with Diglett@1

    let out = st.run_full_battle(&[t1, rep], &d);

    // Decision 0 is the move turn (recorded at the forced-switch pause). The dragged lvl-1 mon
    // was KO'd ON ENTRY by the 3-layer spikes → p2 forced to replace → decision 0's request is
    // the move, and a ForceSwitch boundary follows.
    assert!(out.decisions.len() >= 2, "the phaze drag → spikes KO → forced replacement chains ≥2 boundaries");

    // The drag pulled a lvl-1 mon that the spikes KO'd → p2's active right after the drag was
    // a FAINTED lvl-1 mon → a forced replacement (the second boundary is a ForceSwitch for p2).
    assert!(
        matches!(
            out.decisions[1].request,
            RequestKind::ForceSwitch { force: [false, true] }
        ),
        "the phaze-into-spikes KO forces a NORMAL p2 replacement (boundary 1 is a p2 ForceSwitch); \
         got {:?}",
        out.decisions[1].request
    );

    // The dragged lvl-1 mon was KO'd on entry (a spikes-KO-on-phaze-drag) → p2 lost a mon.
    assert_eq!(
        out.decisions[0].pokemon_left[1], 2,
        "p2 lost the phaze-dragged lvl-1 mon to the 3-layer spikes on entry (3 → 2 mons)"
    );

    // GROUND TRUTH (probe_phaze_regression_rng.js, PHAZE-KO): the post-MOVE-turn seed (decision
    // 0, recorded at the pause) == the real Showdown seed. The phaze drew its accuracy + the
    // sample; the drag's runSwitch (EntryHazard → KO) is DRAW-FREE. A wrong drag-draw model
    // (e.g. the EntryHazard drawing, or a missing sample) trips this.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "43514,9542,40559,8561",
        "post-move-turn seed == the real Showdown seed for a Roar that drags into a 3-layer \
         spikes KO (accuracy + sample drawn; the drag's EntryHazard KO is draw-free)"
    );
}

// ============================================================================
// P4 — a PHAZE (Roar / Whirlwind) BLOCKED BY PROTECT draws its accuracy roll then STOPS: NO
//      forceSwitchFlag, NO drag, NO `sample`. gen-3 Roar / Whirlwind carry the `protect: 1`
//      flag, so a Protect / Detect on the target BLOCKS the phaze at `runEvent('TryHit')` —
//      AFTER the accuracy roll (scripts.ts line 364 accuracy → line 369 TryHit), leaving the
//      target ACTIVE. This is the MULTI-PHAZE `sample` draw-POSITION desync (found by the
//      phaze-diff / e2e fuzz, 2026-07-01): the port's phaze arm did NOT check the Protect
//      block, so it dragged a mon (an EXTRA `sample`) into a protected foe the sim left in
//      place — shifting every LATER phaze's `sample` PRNG position ("same total draw COUNT,
//      wrong `sample` INDEX", compensated elsewhere → the post-turn seed still matched while
//      the dragged mon differed). WRONG (pre-fix): the port dragged a random bench mon + drew
//      the `sample`; the target's active would change. Ground-truth seed from
//      probe_phaze_regression_rng.js (PHAZE-PROTECT). STATE (target stays active, no drag) +
//      draw-COUNT (no sample) pin.
// ============================================================================

/// P4: a Protect / Detect BLOCKS a Roar / Whirlwind (gen-3 `protect: 1` flag). The phaze draws
/// its accuracy roll then is blocked at TryHit → NO drag, NO `sample`, the protector STAYS
/// active. WRONG (pre-fix multi-phaze desync): the port dragged a random bench mon (an extra
/// `sample`) into the protected foe, shifting later phazes' sample positions.
#[test]
fn phaze_blocked_by_protect_draws_no_sample_and_leaves_the_target() {
    let d = dex();
    // p1 FAST Skarmory (Keen Eye, 252 Spe) Protects (priority 3, never-miss → NO accuracy draw,
    // first-protect → NO stall roll) + ONE bench (Blissey). p2 SLOW Suicune (Relaxed) Roars
    // (priority −6 → resolves LAST, into the up Protect) + one bench (Snorlax → a drag WOULD be
    // possible if the Roar weren't blocked, so this proves the block, not a no-eligible-bench
    // fail). The Roar draws its accuracy `randomChance(100,100)` then is BLOCKED by Protect →
    // NO forceSwitchFlag → NO drag → NO `sample`. Draws this turn: Protect (first, no stall) +
    // Roar accuracy + the end-of-turn Quick Claw = 3, NO sample. Same init seed as the probe.
    let skarmory = "Skarmory|||keeneye|protect,steelwing|Serious|252,,,,,252|||||\
                    ]Blissey|||pressure|softboiled|Bold|252,,,,,|||||";
    let p2 = "Suicune|||pressure|roar,surf|Relaxed|252,,252,,,|||||\
              ]Snorlax|||pressure|bodyslam|Careful|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(skarmory, p2, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");

    // Turn 1: p1 Skarmory Protect ; p2 Suicune Roar → BLOCKED by Protect (no drag).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary (the Roar is blocked, no forced switch)");

    // STATE: the Protect BLOCKED the Roar → p1's Skarmory STAYS active (NOT dragged out). A
    // pre-fix port would have dragged Blissey in (Skarmory → bench).
    assert_eq!(
        st.sides[0].pokemon[st.sides[0].active].species_id, "skarmory",
        "a Protect-blocked Roar leaves the protector (Skarmory) ACTIVE — NO drag (the phaze arm \
         must check protect_blocks after the accuracy roll, like the leechseed / status arms)"
    );
    // The phaze drag diagnostic flag must be FALSE (no `sample` fired) — the coverage signal the
    // e2e phaze floor keys on. A pre-fix drag would set it.
    assert!(
        !out.decisions[0].phaze_drag,
        "a Protect-blocked Roar fires NO drag → phaze_drag stays false (no `sample` consumed)"
    );
    // p2's Suicune stays active + p2 keeps both mons (the phaze did nothing but draw accuracy).
    assert_eq!(st.sides[1].pokemon_left, 2, "p2 still has both mons (the Roar was blocked)");

    // GROUND TRUTH (probe_phaze_regression_rng.js, PHAZE-PROTECT): the post-turn seed == the real
    // Showdown seed. Draws = Protect (first, no stall roll) + Roar accuracy + Quick Claw, NO
    // sample. A pre-fix model that drew the `sample` (dragging a mon) adds ONE draw → a divergent
    // post-turn seed AND the wrong active mon. Reverting the phaze arm's protect_blocks check
    // trips BOTH the species assertion above AND this seed.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "3932,55062,24613,55040",
        "post-turn seed == the real Showdown seed for a Protect-blocked Roar (accuracy drawn, \
         NO `sample` — the phaze is blocked at TryHit before the forceSwitchFlag)"
    );
}

// ============================================================================
// L1 — THE 4-WAY RESIDUAL ORDER (the risk area). Leech Seed adds a residual handler at
//      order 10, subOrder 5 — BETWEEN Leftovers (sub 4) and the status DoT (sub 6). On a
//      mon that is leech-seeded + holds Leftovers + is burned + is under sand, the verified
//      residual order is sandstorm(o=8) → Leftovers(s=4) → LEECH(s=5) → burn(s=6). This pins
//      the leech DRAIN→SEEDER-heal ran + the post-turn SEED (the handler-sort order/COUNT);
//      it discriminates leech-vs-Leftovers, NOT leech-vs-burn (no KO here → that pair's order
//      is HP-irrelevant — the full ladder is pinned by the dedicated leechseed_golden). Constructed:
//      p1 Meganium (Leftovers, the SEEDER) Synthesis; p2 Gengar (Leftovers, burned, under sand
//      from an injected board, SEEDED) Splash — distinct speeds (Gengar faster) so NO residual
//      handler tie. STATE + SEED pin (ground truth from probe_leechseed_regression_rng.js).
// ============================================================================

/// L1: the Leech Seed residual sits at subOrder 5 — between Leftovers (4) and the status DoT
/// (6). On a seeded + Leftovers + burned + sand mon this pins (a) the leech DRAIN → SEEDER-heal
/// actually ran (the seeder's HP rises by the drained amount) and (b) the post-turn SEED — i.e.
/// the residual handler-sort ran leech at the correct ORDER/COUNT in the 4-handler ladder.
/// HONEST scope: because no residual KOs here, this scenario's final HP is INSENSITIVE to the
/// leech-vs-burn sub-order specifically (both subtract; the net is order-independent without a
/// KO) — it discriminates leech-vs-Leftovers + the seed, NOT leech-vs-burn. The FULL subOrder
/// ladder (sand 8 → Leftovers s4 → LEECH s5 → burn s6) is verified by the dedicated
/// `leechseed_golden` (the `probe_leechseed_rng.js` residual-dump). Ground truth seed:
/// probe_leechseed_regression_rng.js (real Showdown).
#[test]
fn leech_residual_order_leftovers_sand_burn() {
    let d = dex();
    // p1 Meganium (Leftovers) = the SEEDER (move 0 = Synthesis); p2 Gengar (Leftovers,
    // Levitate, Timid 252 HP/Spe → FASTER than Meganium so no residual handler tie) Splash.
    let meganium = "Meganium||leftovers|overgrow|synthesis,leechseed|Serious|252,,,,,|||||";
    let gengar = "Gengar||leftovers|levitate|splash|Timid|252,,,252,,252|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(meganium, gengar, "30982,33910,19571,50263"), &d).expect("start");
    let st = battle.state_mut().expect("state");

    // INJECT the probe's post-move board: permanent sand, Gengar burned + seeded by Meganium
    // (side 0) at 200 HP, Meganium chipped to 100 HP (so its Synthesis + leech heal are visible).
    st.field.weather = Some(Weather::Sand);
    st.field.weather_turns = 0;
    st.sides[1].pokemon[0].status = Some(Status::Burn);
    st.sides[1].pokemon[0].hp = 200;
    st.sides[1].pokemon[0].leech_seed = Some(0); // seeded BY side 0 (Meganium)
    st.sides[0].pokemon[0].hp = 100;

    // The turn: p1 Synthesis (move 0), p2 Splash (move 0).
    let _ = st.run_turn(0, 0, &d);

    // GROUND TRUTH (probe_leechseed_regression_rng.js): the 4-way residual on Gengar is
    // sand(-20) → Leftovers(+20) → LEECH(-40) → burn(-40) = 200 → 120; and the seeder
    // Meganium ends at 231 (Synthesis under sand + its own sand chip + Leftovers + the leech
    // heal). A wrong leech subOrder permutes the heal/drain and desyncs these.
    assert_eq!(
        st.sides[1].pokemon[0].hp, 120,
        "the seeded Gengar's post-residual HP must be 120 (sand → Leftovers → LEECH[s=5] → \
         burn); a wrong leech subOrder re-orders the heal/drain and diverges this"
    );
    assert_eq!(
        st.sides[0].pokemon[0].hp, 231,
        "the seeder Meganium's post-residual HP must be 231 (its Synthesis + sand chip + \
         Leftovers + the leech heal of 40); a wrong leech model diverges the heal"
    );
    assert!(
        st.sides[1].pokemon[0].leech_seed.is_some(),
        "Gengar stays seeded after the drain (the volatile persists turn-to-turn)"
    );
    assert_eq!(
        seed_str(&st.prng_seed()),
        "57890,13032,12358,42006",
        "post-turn seed == the real Showdown seed (the leech residual is DRAW-FREE; only the \
         move + Quick Claw draw); a stray leech draw or a wrong handler-sort count desyncs it"
    );
}

// ============================================================================
// L2 — THE LEECH HANDLER TIE (the residual tie-shuffle position). When BOTH actives are
//      leech-seeded at EQUAL cached speed, their two leech handlers (order 10, sub 5) TIE in
//      the residual speed-sort → the Fisher-Yates tie-group shuffle draws ONE random(0,2). A
//      WRONG leech subOrder (so it doesn't tie the other leech), or a MISSING leech handler in
//      the gathered set, changes the tie-group COUNT → a divergent post-turn SEED. Constructed:
//      a Snorlax MIRROR (equal speed), each seeded by the OTHER (the drains/heals cross). SEED
//      pin (ground truth from probe_leechseed_regression_rng.js).
// ============================================================================

/// L2: two leech handlers at equal speed TIE in the residual speed-sort → one tie-shuffle
/// draw. A wrong leech subOrder (no tie) or a missing leech handler changes the shuffle count
/// → a divergent post-turn seed. Ground truth: probe_leechseed_regression_rng.js (real Showdown).
#[test]
fn leech_handler_tie_at_equal_speed_draws_one_shuffle() {
    let d = dex();
    // A Snorlax MIRROR (Serious 252 HP → IDENTICAL speed → the residual leech handlers TIE).
    // Both Splash (draw-free move) so the residual tie-shuffle is the only non-QuickClaw draw.
    let snorlax = "Snorlax||none|immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax, snorlax, "3090,29639,52671,47823"), &d).expect("start");
    let st = battle.state_mut().expect("state");

    // INJECT: cross-seed — p1 seeded by side 1, p2 seeded by side 0 (the drains/heals cross),
    // both at 300 HP so neither faints (the tie persists). Their leech handlers tie at sub 5.
    st.sides[0].pokemon[0].leech_seed = Some(1);
    st.sides[1].pokemon[0].leech_seed = Some(0);
    st.sides[0].pokemon[0].hp = 300;
    st.sides[1].pokemon[0].hp = 300;

    // The turn: both Splash (move 0). The residual draws ONE tie-shuffle (the 2-leech tie) +
    // Quick Claw; the leech drain/heal is draw-free.
    let _ = st.run_turn(0, 0, &d);

    // Each Snorlax loses floor(524/8)=65 (drain) but is healed 65 by the OTHER's leech (the
    // cross-seed) → net 0... but the heal is to the SEEDER's active (the other Snorlax), so each
    // takes 65 drain AND heals 65 from being the other's seeder → net 0. Both end at 300.
    assert_eq!(
        st.sides[0].pokemon[0].hp, 300,
        "p1 Snorlax: -65 leech drain (seeded by p2) +65 leech heal (it seeds p2) = net 0"
    );
    assert_eq!(
        st.sides[1].pokemon[0].hp, 300,
        "p2 Snorlax: symmetric — net 0"
    );
    assert_eq!(
        seed_str(&st.prng_seed()),
        "7332,3983,10909,19927",
        "post-turn seed == the real Showdown seed: the 2-leech EQUAL-SPEED tie draws ONE \
         residual handler-sort shuffle. A wrong leech subOrder (no tie) or a missing leech \
         handler in the sort changes the tie-group count and desyncs this seed"
    );
}

// ============================================================================
// L3 — THE SEEDER-FAINTED GATE. The leech residual reads the seeder's CURRENT active
//      (getAtSlot(sourceSlot)); if it is FAINTED the whole onResidual returns early —
//      `if (!target || target.fainted || target.hp <= 0) return` — so the seeded mon takes
//      NO drain that turn (and nobody heals). WRONG model: draining anyway (the seeded mon
//      loses HP with no recipient). Constructed: p1 Meganium seeds p2 Gengar, then Meganium
//      is pre-FAINTED; the residual must SKIP the leech (Gengar unchanged). STATE + SEED pin
//      (ground truth from probe_leechseed_regression_rng.js).
// ============================================================================

/// L3: a leech whose seeder's active is fainted does NOTHING (no drain, no heal) — the gen-3
/// `if (!target || target.fainted) return`. WRONG model: the seeded mon loses HP anyway. The
/// fix's `apply_leech_seed` returns early on a fainted seeder. Ground truth:
/// probe_leechseed_regression_rng.js (real Showdown).
#[test]
fn leech_seeder_fainted_skips_the_drain() {
    let d = dex();
    // p1 Meganium (the seeder, move 0 = Leech Seed, move 1 = Splash) + a Blissey bench; p2
    // Gengar (Levitate, Timid 252 HP/Spe) Splash. We seed Gengar then pre-FAINT Meganium.
    let meganium = "Meganium||none|overgrow|leechseed,splash|Serious|252,,,,,|||||\
                    ]Blissey||none|naturalcure|softboiled|Bold|252,,,,,|||||";
    let gengar = "Gengar||none|levitate|splash|Timid|252,,,252,,252|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(meganium, gengar, "27226,35589,63448,3967"), &d).expect("start");
    let st = battle.state_mut().expect("state");

    // INJECT: Gengar seeded by Meganium (side 0) at 200 HP; Meganium pre-FAINTED (the seeder's
    // active is dead → the leech residual must skip the drain). A fainted active means the turn
    // is a forced-switch state for p1 — no Quick Claw — so the seed is unchanged (zero draws).
    st.sides[1].pokemon[0].leech_seed = Some(0);
    st.sides[1].pokemon[0].hp = 200;
    st.sides[0].pokemon[0].hp = 0;
    st.sides[0].pokemon[0].fainted = true;
    st.sides[0].pokemon_left = 1; // Meganium fainted → 1 mon left

    // The turn: p1's active is fainted (its move is skipped); p2 Splash (move 0). The leech
    // residual sees the seeder (p1 active) fainted → SKIPS the drain.
    let _ = st.run_turn(1, 0, &d); // p1 move slot is irrelevant (fainted → skipped)

    // GROUND TRUTH: Gengar's HP is UNCHANGED (200) — the leech did NOTHING (seeder fainted).
    // The seed is unchanged from the pre-turn seed (no Quick Claw — a faint pauses the turn).
    assert_eq!(
        st.sides[1].pokemon[0].hp, 200,
        "the seeded Gengar takes NO leech drain when the seeder's active is fainted (the \
         onResidual returns early); a wrong model would drain it anyway"
    );
    assert!(
        st.sides[1].pokemon[0].leech_seed.is_some(),
        "Gengar stays seeded (the volatile is not cleared just because the seeder fainted)"
    );
    assert_eq!(
        seed_str(&st.prng_seed()),
        "27226,35589,63448,3967",
        "post-turn seed == the real Showdown seed (the fainted seeder pauses the turn → no \
         Quick Claw → zero draws; the leech residual is draw-free and skipped)"
    );
}

// ============================================================================
// SUB1 — gen-3 SUBSTITUTE: a DAMAGING move into a sub STILL DRAWS the per-move SECONDARY
//        `random(100)` (the gen-3 quirk — the same draw count as a bare hit), but its EFFECT
//        does NOT apply (the sub blocked it). The SURPRISE this layer surfaced (CONTRADICTING
//        the task's stated "one fewer random(100)" assumption): gen-3 `secondaries()` iterates
//        the now-`null` target list, so the `random(100)` fires regardless of the sub. A model
//        that SKIPPED the secondary random(100) behind a sub would draw ONE FEWER → a divergent
//        post-turn seed. Ground-truth seed from probe_substitute_regression_rng.js (S1).
// ============================================================================

/// SUB1: a Body Slam (par 30 secondary) into a SUBBED Blissey draws acc+crit+dmg+SECONDARY(100)
/// +QuickClaw — the SAME as a bare hit. The sub absorbs the damage (Blissey's HP unchanged) and
/// NO paralysis applies (status `-`). WRONG (the task's stated model): skipping the secondary
/// random(100) behind a sub draws one fewer → the post-turn seed diverges. The Snorlax Body Slam
/// here BREAKS Blissey's sub (a strong hit), but the secondary is STILL suppressed (no par).
#[test]
fn substitute_absorbs_a_hit_but_the_secondary_random_100_still_draws() {
    let d = dex();
    // p1 fast Snorlax Body Slams (par 30). p2 Blissey has a sub up (injected). Seed = the
    // probe's seedBefore (post-construction), so the Rust draw-free construction lines up.
    let snorlax = "Snorlax|||immunity|bodyslam,splash|Adamant|,252,,,,252|||||";
    let blissey = "Blissey|||naturalcure|softboiled,splash|Bold|252,,,252,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax, blissey, "30982,33910,19571,50263"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // Inject a sub on the p2 Blissey (floor(maxhp/4)), exactly like the probe's addVolatile.
    let bliss = &mut st.sides[1].pokemon[0];
    let sub_hp = bliss.maxhp / 4;
    bliss.substitute = Some(sub_hp);
    let bliss_hp_before = bliss.hp;

    // Turn 1: p1 Body Slam INTO the sub ; p2 Splash (move 2 = splash, draw-free).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(1))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // The sub ABSORBED the hit: Blissey's OWN HP is UNCHANGED (the damage went to the sub),
    // and NO paralysis applies (the secondary was suppressed behind the sub).
    assert_eq!(
        st.sides[1].pokemon[0].hp, bliss_hp_before,
        "the sub absorbed the Body Slam — Blissey's OWN HP is unchanged"
    );
    assert_eq!(
        st.sides[1].pokemon[0].status, None,
        "the Body Slam's paralysis secondary is SUPPRESSED behind the sub (no par)"
    );

    // GROUND TRUTH (probe_substitute_regression_rng.js, S1): the post-turn seed == the real
    // Showdown seed (draws = acc + crit + dmg + the SECONDARY random(100) + Quick Claw). A model
    // that skipped the secondary random(100) behind the sub would draw ONE FEWER → this diverges.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37673,46633,62039,8266",
        "post-turn seed == the real Showdown seed: the per-move secondary random(100) MUST still \
         be drawn against a sub (gen-3), even though its effect is suppressed"
    );
}

// ============================================================================
// SUB2 — the sub BREAKS on a hit >= its HP, and the EXCESS does NOT carry to the mon (gen-3).
//        A 1-HP sub is broken by any hit; the mon's HP is UNCHANGED (no carry-over). The draws
//        are the same acc+crit+dmg+secondary(100)+QC. Ground-truth seed from the probe (S2).
// ============================================================================

/// SUB2: a Body Slam BREAKS a 1-HP sub on the p2 Snorlax — the mon's HP is UNCHANGED (the excess
/// damage does NOT carry to the mon in gen-3), the sub is gone (→ None). WRONG (a model that
/// carried the excess to the mon): Snorlax's HP would drop → a STATE divergence.
#[test]
fn substitute_break_does_not_carry_excess_to_the_mon() {
    let d = dex();
    let snorlax = "Snorlax|||immunity|bodyslam,splash|Adamant|,252,,,,252|||||";
    let p2lax = "Snorlax|||immunity|splash|Careful|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax, p2lax, "30982,33910,19571,50263"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // A 1-HP sub on the p2 Snorlax — ANY hit breaks it; the mon takes NO carry-over.
    let p2mon = &mut st.sides[1].pokemon[0];
    p2mon.substitute = Some(1);
    let p2_hp_before = p2mon.hp;

    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // The sub BROKE (→ None) and the mon's HP is UNCHANGED (no carry-over).
    assert_eq!(
        st.sides[1].pokemon[0].substitute, None,
        "the 1-HP sub broke on the hit (→ None)"
    );
    assert_eq!(
        st.sides[1].pokemon[0].hp, p2_hp_before,
        "the excess damage does NOT carry to the mon (gen-3) — its HP is unchanged"
    );

    // GROUND TRUTH (probe_substitute_regression_rng.js, S2): the post-turn seed == the real
    // Showdown seed (same acc+crit+dmg+secondary(100)+QC as a bare hit — the break is draw-free).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37673,46633,62039,8266",
        "post-turn seed == the real Showdown seed (the sub-break is a STATE change, draw-free)"
    );
}

// ============================================================================
// SUB3 — a CONFUSION self-hit (behind a sub) hits the MON, NOT the sub. The self-hit's
//        `this.damage` bypasses the `onTryPrimaryHit` sub-intercept — so the mon's OWN HP drops
//        while the sub HP stays put. The draw model is unchanged (randomChance(1,2) then
//        random(16)). Ground-truth seed from the probe (S3, seed 2,2,2,2 → a self-hit).
// ============================================================================

/// SUB3: a subbed + confused Snorlax that fails its confusion check self-hits — the MON's HP
/// DROPS while the sub HP (131) is UNCHANGED. WRONG (a model that routed the confusion self-hit
/// into the sub): the sub HP would drop and the mon's HP would be untouched → a STATE divergence.
#[test]
fn confusion_self_hit_behind_a_sub_hits_the_mon_not_the_sub() {
    let d = dex();
    let snorlax = "Snorlax|||immunity|splash,bodyslam|Adamant|252,252,,,,|||||";
    let blissey = "Blissey|||naturalcure|splash|Bold|252,,,,,|||||";
    // Seed = the probe's seedBefore for seed 2,2,2,2 (the self-hit case).
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax, blissey, "52170,17908,58343,8292"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // Sub up (floor(maxhp/4)) + confused (counter 4, matching the probe's injected time=4). Both
    // are draw-free direct sets (the probe's addVolatile random(2,6) was consumed BEFORE its
    // seedBefore, so seeding here at seedBefore + setting the counter directly lines up).
    let lax = &mut st.sides[0].pokemon[0];
    let sub_hp = lax.maxhp / 4;
    lax.substitute = Some(sub_hp);
    lax.confusion = Some(4);
    let lax_hp_before = lax.hp;

    // Turn 1: p1 Splash (the confusion check fires; on a fail it self-hits) ; p2 Splash.
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // The confusion self-hit damaged the MON (HP dropped) while the SUB HP is UNCHANGED.
    assert!(
        st.sides[0].pokemon[0].hp < lax_hp_before,
        "the confusion self-hit damaged the MON's OWN HP (got {} < {})",
        st.sides[0].pokemon[0].hp, lax_hp_before
    );
    assert_eq!(
        st.sides[0].pokemon[0].substitute, Some(sub_hp),
        "the confusion self-hit does NOT touch the sub — its HP is unchanged ({sub_hp})"
    );

    // GROUND TRUTH (probe_substitute_regression_rng.js, S3 seed 2,2,2,2): the post-turn seed ==
    // the real Showdown seed (the confusion check randomChance(1,2) + the self-hit random(16) +
    // Quick Claw — unchanged by the sub).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "57951,31032,5281,7249",
        "post-turn seed == the real Showdown seed (the confusion self-hit draws are unchanged \
         behind a sub — it hits the mon, drawing randomChance(1,2)+random(16))"
    );
}

// ============================================================================
// SUB4 — a TRI ATTACK into a sub draws its `random(100)` (the 20% gate) but NOT the `random(3)`
//        `sample` (the secondary's onHit runs on the now-`null` target) — so behind a sub it
//        draws ONE FEWER than a Tri Attack that LANDS on a bare mon. A model that drew the
//        random(3) behind the sub → a divergent seed. Ground-truth seed from the probe (S4).
// ============================================================================

/// SUB4: a Tri Attack into a SUBBED Blissey draws acc+crit+dmg+random(100)+QuickClaw but NOT the
/// `random(3)` sample (the secondary's status-pick runs on a null target) — so NO status applies
/// and the sub holds. WRONG (a model that drew the random(3) behind the sub): an extra draw →
/// the post-turn seed diverges. The seed (1,1,1,1) is one where the random(100) PASSES (so a
/// bare hit WOULD draw the random(3)) — isolating the suppression.
#[test]
fn tri_attack_into_a_sub_draws_random_100_but_not_the_sample_random_3() {
    let d = dex();
    let porygon2 = "Porygon2|||trace|triattack,splash|Modest|,,,252,,252|||||";
    let blissey = "Blissey|||naturalcure|softboiled,splash|Bold|252,,,252,,|||||";
    // Seed = the probe's seedBefore for seed 1,1,1,1.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(porygon2, blissey, "40617,35584,40138,56971"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let bliss = &mut st.sides[1].pokemon[0];
    let sub_hp = bliss.maxhp / 4;
    bliss.substitute = Some(sub_hp);

    // Turn 1: p1 Tri Attack INTO the sub ; p2 Splash (move 2, draw-free).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(1))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // NO status applies — whether the sub HELD or BROKE, Tri Attack's status secondary is
    // SUPPRESSED behind the sub (the `random(3)` sample never ran on the null target). The
    // status-suppression + the SEED (below) are the load-bearing proof of the random(3)
    // draw-COUNT suppression; the exact sub HP is a damage-AMOUNT question orthogonal to the
    // substitute draw model (the golden's absorb scenario pins the sub HP directly).
    assert_eq!(
        st.sides[1].pokemon[0].status, None,
        "Tri Attack's status secondary is SUPPRESSED behind the sub (the random(3) sample is \
         not drawn → no status)"
    );

    // GROUND TRUTH (probe_substitute_regression_rng.js, S4 seed 1,1,1,1): the post-turn seed ==
    // the real Showdown seed (draws = acc + crit + dmg + the random(100) 20% gate + Quick Claw,
    // but NOT the random(3) sample). A model that drew the random(3) behind the sub diverges.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "62471,53173,63396,19086",
        "post-turn seed == the real Showdown seed: Tri Attack into a sub draws the random(100) \
         but NOT the random(3) sample (the secondary's onHit runs on the null sub target)"
    );
}

// ============================================================================
// SUB5 — a SELF-boost secondary (Meteor Mash +1 Atk to the USER) STILL APPLIES through a sub.
//        The sub blocks FOE-targeting secondaries (status/stat-drop/flinch/confusion), but a
//        `secondary.self.boosts` targets the SOURCE (not the null sub target), so it is NOT
//        blocked. WRONG (a model that suppressed ALL secondaries behind a sub): the attacker
//        would NOT get +1 Atk → a STATE divergence. The self-boost apply is draw-free, so the
//        SEED is unchanged either way (it isolates the STATE). Ground-truth from the probe.
// ============================================================================

/// SUB5: a Metagross Meteor Mash (+1 Atk self, 20%) INTO a substituted Skarmory still gives
/// Metagross +1 Atk on a proc (the self-boost targets the user, not the sub) — while the sub
/// absorbs the damage. WRONG (suppress-all-behind-a-sub): the +1 Atk is dropped. Seed chosen so
/// the 20% self-boost PROCS (the boost lands) and the sub HOLDS.
#[test]
fn self_boost_secondary_still_applies_through_a_sub() {
    let d = dex();
    let metagross = "Metagross|||clearbody|meteormash,splash|Adamant|,252,,,,|||||";
    let skarmory = "Skarmory|||keeneye|substitute,splash|Impish|252,,252,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(metagross, skarmory, "10245,39561,58851,21846"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // Inject a sub on the p2 Skarmory (floor(maxhp/4)), exactly like the probe's addVolatile.
    let skarm = &mut st.sides[1].pokemon[0];
    let sub_hp = skarm.maxhp / 4;
    skarm.substitute = Some(sub_hp);

    // Turn 1: p1 Meteor Mash INTO the sub (the self-boost procs at this seed) ; p2 Splash.
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(1))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // The SELF-boost APPLIED through the sub: Metagross has +1 Atk (boosts[0] == 1), and the
    // sub absorbed the damage (Skarmory's own HP unchanged, the sub HP dropped).
    assert_eq!(
        st.sides[0].pokemon[0].boosts[0], 1,
        "Meteor Mash's +1 Atk SELF-boost applies through the sub (the self-boost targets the \
         USER, not the null sub target) — got {}",
        st.sides[0].pokemon[0].boosts[0]
    );
    assert!(
        st.sides[1].pokemon[0].substitute.is_some(),
        "the sub absorbed the Meteor Mash (it held)"
    );

    // GROUND TRUTH (probe_substitute_regression_rng.js / the self-boost probe): the post-turn
    // seed == the real Showdown seed. The self-boost apply is DRAW-FREE (`boost()` consumes no
    // PRNG), so the seed isolates the draw model (acc+crit+dmg+secondary random(100)+QC) while
    // the +1 Atk above isolates the STATE.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "59855,64989,1335,20530",
        "post-turn seed == the real Showdown seed (the self-boost apply through a sub is \
         draw-free; only the foe-targeting effect is suppressed)"
    );
}

// ============================================================================
// SWITCH-TIE-WEATHER — the `eachEvent('WeatherChange')` switch-in shuffle. A MID-TURN
//      switch-in whose entrant TIES the opposing active on cached speed AND CHANGES the
//      weather (Sand Stream / Drizzle / Drought) draws ONE `eachEvent('WeatherChange')`
//      speed-tie shuffle (`Field.setWeather` → field.ts:87), fired INSIDE the runSwitch
//      runAction (before its trailing `eachEvent('Update')`). WRONG (pre-fix): the port set
//      the weather but MISSED that shuffle, so a switch-into-a-tie-under-freshly-set-sand
//      turn drew ONE FEWER PRNG call than the sim → the post-turn seed diverged on every
//      later turn. This was the e2e_84 dec4 desync that kept Substitute out of the e2e
//      capstone (the SAME hard class as `forced_replacement_recaches_speed_seed` /
//      `para_while_active_keeps_full_cached_speed_seed` — a switching/weather draw-count bug,
//      NOT a substitute bug). SEED pin (a switch-into-tie-under-Sand-Stream turn).
// ============================================================================

/// SWITCH-TIE-WEATHER: a Sand-Stream entrant that TIES the opposing active draws the
/// `eachEvent('WeatherChange')` tie-shuffle on the switch-in. WRONG (pre-fix): the port set
/// the weather draw-free and MISSED the shuffle → the switch-into-tie-under-sand turn drew one
/// fewer call → the post-turn seed diverged (the e2e_84 dec4 substitute-surfaced desync). The
/// fix fires `each_event_shuffle()` when the runSwitch ability Start CHANGES the weather.
///
/// Constructed 213-vs-213-class mirror (exact tie): p1 [Suicune lead, Tyranitar (Sand Stream,
/// spe 221)] ; p2 Suicune (spe 221). Turn 1 (move): p1 SWITCHES Tyranitar in (slot 2 → the
/// runSwitch sets sandstorm while the actives TIE 221 == 221 → the WeatherChange shuffle) while
/// p2 Suicune Splashes. A model that misses the shuffle draws one fewer → the seed diverges.
#[test]
fn switch_into_a_tie_under_sand_draws_the_weather_change_shuffle_seed() {
    let d = dex();
    // Suicune 60-spe-EV serious = 221 spe; Tyranitar 252-spe-EV serious = 221 spe → exact TIE.
    let p1 = "Suicune|||pressure|surf,splash|Serious|252,,,,,60|||||\
              ]Tyranitar|||sandstream|crunch,rockslide|Serious|252,,,,,252|||||";
    let p2 = "Suicune|||pressure|surf,splash|Serious|252,,,,,60|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "52903,53571,56373,31187"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");

    // Turn 1 (move): p1 SWITCHES Tyranitar in (team slot 1 = Choice::Switch(1)); p2 Splash.
    // The switch (order 103) runs first → the runSwitch sets sandstorm → the actives TIE →
    // the `eachEvent('WeatherChange')` tie-shuffle draws.
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Switch(1), Choice::Move(1))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // STATE: Tyranitar is now the p1 active, sandstorm is up, and p2's Suicune took the
    // sandstorm chip (a Water type — not Rock/Ground/Steel → it chips maxhp/16).
    assert_eq!(
        st.sides[0].pokemon[st.sides[0].active].species_id, "tyranitar",
        "the switched-in Tyranitar is the p1 active"
    );
    assert_eq!(st.field.weather, Some(Weather::Sand), "Sand Stream set sandstorm on the switch-in");
    assert!(
        st.sides[1].pokemon[st.sides[1].active].hp < st.sides[1].pokemon[st.sides[1].active].maxhp,
        "p2's non-Rock/Ground/Steel Suicune took the sandstorm residual chip"
    );

    // GROUND TRUTH (probe_switch_tie_weather_regression_rng.js, real Showdown, reseeded to the
    // RAW seed at the decision so it lines up with the Rust's draw-free `start_with_switchins`):
    // the post-turn seed == the real Showdown seed. The switch-into-tie-under-sand turn draws 9
    // (incl. the `eachEvent('WeatherChange')` switch-in tie-shuffle AND the residual nested
    // weather shuffle). WITHOUT the WeatherChange shuffle the port draws 8 → this seed diverges
    // (the e2e_84 dec4 desync that kept Substitute out of the e2e capstone).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "54657,11218,22550,62890",
        "post-turn seed == the real Showdown seed for a switch-into-a-tie under freshly-set \
         Sand Stream — the `eachEvent('WeatherChange')` switch-in tie-shuffle MUST draw; \
         without it the port draws one fewer call and this desyncs (the e2e_84 dec4 bug)"
    );
}

// ============================================================================
// EXPLOSION / SELF-DESTRUCT — the gen-3 self-KO is UNCONDITIONAL + precedes the hit
//   (`useMoveInner` battle-actions.ts:501-503, `gen != 4 && selfdestruct == 'always'`:
//   `this.battle.faint(pokemon)` BEFORE `trySpreadMoveHit`). So the USER faints THROUGH a
//   Protect / a Ghost immunity / a sub / a miss — and it is DRAW-FREE (only the normal
//   acc/crit/dmg draws fire; the resulting faint changes pokemon_left / who-acts, e.g. NO
//   trailing Quick Claw on a deciding faint). Ground-truth seeds + STATE from
//   `harness/probe_explosion_regression_rng.js`. These are the CRUX edges the model must
//   never silently break: the user MUST faint even when the hit does nothing.
// ============================================================================

/// E1: Explosion into a PROTECT — the move is BLOCKED (`-activate Protect`, no foe damage) but
/// the USER STILL FAINTS (the self-KO precedes the hit). WRONG (a model that only faints the
/// user when the hit lands): Electrode would survive the blocked Explosion. The foe is at FULL
/// HP + Protect up; the user is fainted. Draws = the foe's first-Protect (draw-free) + the
/// Explosion accuracy (randomChance(100,100)), then the faint pauses for a replacement (no Quick
/// Claw). Ground truth: probe_explosion_regression_rng.js (E1).
#[test]
fn explosion_into_a_protect_the_user_still_faints() {
    let d = dex();
    let electrode =
        "Electrode|||soundproof|explosion,thunderbolt|Hasty|,252,,,,252|||||]Jolteon|||voltabsorb|thunderbolt|Timid|,,,,,252|||||";
    let blissey = "Blissey|||naturalcure|protect,softboiled|Bold|252,,252,,,|||||";
    // Seed at the probe's POST-CONSTRUCTION seedBefore (start_with_switchins is draw-free, so
    // the Rust prng at construction equals this string).
    let mut battle =
        Battle::start_with_switchins(&opts_cg(electrode, blissey, "30982,33910,19571,50263"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // dec 0: p1 Explode (move 1) ; p2 Protect (move 1) → blocked, user faints.
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);

    // STATE: the USER (p1 Electrode) is FAINTED even though the Explosion was BLOCKED; the foe
    // (p2 Blissey) took NO damage (full HP).
    let user = &st.sides[0].pokemon[0];
    assert!(user.fainted && user.hp == 0, "the Explosion user faints THROUGH the Protect block");
    let foe = &st.sides[1].pokemon[st.sides[1].active];
    assert_eq!(foe.hp, foe.maxhp, "the Protect blocked the Explosion — the foe took NO damage");
    assert_eq!(st.sides[0].pokemon_left, 1, "p1 lost the Explode user (2 → 1)");
    assert_eq!(st.sides[1].pokemon_left, 1, "p2 is intact (the block cost nothing)");

    // GROUND TRUTH (probe_explosion_regression_rng.js, E1): the post-turn seed == the real
    // Showdown seed. A wrong self-KO placement (adding/removing a draw) or wrongly NOT fainting
    // through the block would diverge here.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "57890,13032,12358,42006",
        "post-turn seed == the real Showdown seed for an Explosion blocked by Protect — the user \
         faints (draw-free) THROUGH the block; the deciding faint draws NO Quick Claw"
    );
}

/// E2: Explosion into a GHOST (Normal-immune) — no foe damage (`-immune`) but the USER STILL
/// FAINTS. WRONG (a model that only faints the user on a landed hit): Electrode would survive
/// exploding into a Ghost. The foe (Gengar) is at full HP; the user is fainted. Draws = the
/// Explosion accuracy only (immune short-circuits before crit/dmg), then the faint (no Quick
/// Claw). Ground truth: probe_explosion_regression_rng.js (E2).
#[test]
fn explosion_into_a_ghost_the_user_still_faints() {
    let d = dex();
    let electrode =
        "Electrode|||soundproof|explosion,thunderbolt|Hasty|,252,,,,252|||||]Jolteon|||voltabsorb|thunderbolt|Timid|,,,,,252|||||";
    let gengar = "Gengar|||levitate|splash,shadowball|Timid|252,,,,,252|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(electrode, gengar, "30982,33910,19571,50263"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // dec 0: p1 Explode (move 1) into the Ghost ; p2 Splash (move 1, draw-free).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);

    // STATE: the USER faints THROUGH the Normal immunity; the Ghost took NO damage.
    let user = &st.sides[0].pokemon[0];
    assert!(user.fainted && user.hp == 0, "the Explosion user faints THROUGH the Ghost immunity");
    let foe = &st.sides[1].pokemon[st.sides[1].active];
    assert_eq!(foe.hp, foe.maxhp, "the Ghost is immune to the Normal Explosion — NO damage");
    assert_eq!(st.sides[0].pokemon_left, 1, "p1 lost the Explode user (2 → 1)");
    assert_eq!(st.sides[1].pokemon_left, 1, "p2 is intact (immune)");

    // GROUND TRUTH (probe_explosion_regression_rng.js, E2): the post-turn seed == the real
    // Showdown seed (accuracy-only draw, then the user faint — no Quick Claw).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "57890,13032,12358,42006",
        "post-turn seed == the real Showdown seed for an Explosion into a Ghost — the user faints \
         (draw-free) THROUGH the immunity; only the accuracy roll is drawn before the faint"
    );
}

/// E3: Explosion BREAKS a SUBSTITUTE + the USER STILL FAINTS. The foe (Blissey) has a 1-HP sub;
/// the Explosion damage hits the sub (breaks it, no carry to the mon) AND the user faints. WRONG
/// (a model that skipped the self-KO when the sub absorbed, or carried the excess to the mon):
/// the user would survive / Blissey's HP would drop. The sub is gone, Blissey's OWN HP is
/// UNCHANGED, the user is fainted. Draws = acc+crit+dmg (Explosion has no secondary), then the
/// faint (no Quick Claw). Ground truth: probe_explosion_regression_rng.js (E3).
#[test]
fn explosion_breaks_a_substitute_and_the_user_still_faints() {
    let d = dex();
    let electrode =
        "Electrode|||soundproof|explosion,thunderbolt|Hasty|,252,,,,252|||||]Jolteon|||voltabsorb|thunderbolt|Timid|,,,,,252|||||";
    let blissey = "Blissey|||naturalcure|softboiled,splash|Bold|252,,252,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(electrode, blissey, "30982,33910,19571,50263"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // A 1-HP sub on the p2 Blissey — ANY Explosion breaks it; the mon takes NO carry-over.
    let bliss = &mut st.sides[1].pokemon[0];
    bliss.substitute = Some(1);
    let bliss_hp_before = bliss.hp;

    // dec 0: p1 Explode (move 1) INTO the sub ; p2 Splash (move 2, draw-free).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(1))], &d);

    // STATE: the USER faints; the sub BROKE (→ None); Blissey's OWN HP is UNCHANGED (no carry).
    let user = &st.sides[0].pokemon[0];
    assert!(user.fainted && user.hp == 0, "the Explosion user faints even though the sub absorbed the hit");
    assert_eq!(st.sides[1].pokemon[0].substitute, None, "the Explosion broke the sub (→ None)");
    assert_eq!(
        st.sides[1].pokemon[0].hp, bliss_hp_before,
        "the excess Explosion damage does NOT carry to the mon (gen-3) — Blissey's HP is unchanged"
    );
    assert_eq!(st.sides[0].pokemon_left, 1, "p1 lost the Explode user (2 → 1)");

    // GROUND TRUTH (probe_explosion_regression_rng.js, E3): the post-turn seed == the real
    // Showdown seed (acc+crit+dmg — the sub-break + the self-KO are both draw-free; no Quick Claw).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "8769,17248,37115,18776",
        "post-turn seed == the real Showdown seed for an Explosion into a sub — the sub absorbs \
         the damage (breaks, no carry) AND the user still faints, both DRAW-FREE"
    );
}

/// E4: a MUTUAL Explosion (both last mons) is a gen-3 double-faint TIE. Both Electrodes Explode
/// the SAME turn (equal speed → an action-order tie-shuffle); both faint, both pokemon_left → 0,
/// win(None) TIE. WRONG (a model that gave the FIRST mover the win, or that didn't self-KO the
/// second mover after the first already ended): a false winner instead of a tie. Ground truth:
/// probe_explosion_regression_rng.js (E4).
#[test]
fn mutual_explosion_is_a_double_faint_tie() {
    let d = dex();
    let electrode = "Electrode|||soundproof|explosion,thunderbolt|Timid|,,,,,252|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(electrode, electrode, "55250,62519,52978,42619"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);

    // STATE: both fainted, both sides out of mons → a gen-3 TIE (winner None, ended).
    assert!(st.sides[0].pokemon[0].fainted, "p1 Electrode fainted (its own Explosion self-KO)");
    assert!(st.sides[1].pokemon[0].fainted, "p2 Electrode fainted (its own Explosion self-KO)");
    assert_eq!(st.sides[0].pokemon_left, 0, "p1 out of mons");
    assert_eq!(st.sides[1].pokemon_left, 0, "p2 out of mons");
    assert!(out.ended, "the mutual Explosion ended the battle");
    assert_eq!(out.winner, None, "both sides out ⇒ a gen-3 TIE (win(None))");

    // GROUND TRUTH (probe_explosion_regression_rng.js, E4): the post-turn seed == the real
    // Showdown seed (the action-order tie-shuffle + acc/crit/dmg for the first Exploder; the
    // deciding double-faint draws NO Quick Claw).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "14735,5508,9502,51264",
        "post-turn seed == the real Showdown seed for a mutual Explosion double-faint TIE"
    );
}

// ============================================================================
// FD1 — SEISMIC TOSS deals the USER's LEVEL exactly (level 100 → 100), drawing ONLY its
//       accuracy roll (acc 100 but NOT never-miss → still draws) — NO crit roll, NO 16-way
//       damage roll, NO secondary. WRONG (the pre-fix engine): a fixed-damage move carries
//       `basePower: 0`, so it was classified Status → run_status_move's fail-loud guard (or,
//       for a naive fix, it would either no-op or wrongly run the standard damage calc with
//       a crit + damage roll). STATE pin (HP drops by EXACTLY 100) + SEED pin (draws == acc +
//       Quick Claw = 2; a spurious crit/damage roll desyncs). Ground truth from
//       probe_fixeddamage_regression_rng.js (FD-LEVEL).
// ============================================================================

/// FD1: Seismic Toss deals the user's LEVEL (100) as flat, typeless, non-crit, non-rolled
/// damage. WRONG (pre-fix): a bp-0 fixed-damage move no-op'd / fail-loud'd / drew a spurious
/// crit+damage roll. The Snorlax target drops by EXACTLY 100 (524 → 424) and the post-turn
/// seed matches a draw model of acc + Quick Claw only.
#[test]
fn seismic_toss_deals_user_level_damage() {
    let d = dex();
    // p1 Machamp (level 100) Seismic Tosses the slower p2 Snorlax; Snorlax Splashes (draw-free
    // no-op). Distinct speeds → no action-order tie-shuffle, so the only draws are the ST
    // accuracy roll + the end-of-turn Quick Claw = 2.
    let machamp = "Machamp|||guts|seismictoss|Adamant|252,252,,,,|||||";
    let snorlax = "Snorlax|||immunity|splash|Careful|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(machamp, snorlax, "53303,35262,36397,29520"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let hp_before = st.sides[1].pokemon[0].hp;

    // Turn 1: p1 Seismic Toss (100) ; p2 Splash.
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // STATE: the Snorlax lost EXACTLY 100 (the user's level), typeless, no crit/roll variance.
    assert_eq!(
        st.sides[1].pokemon[0].hp,
        hp_before - 100,
        "Seismic Toss deals the user's level (100) exactly — a flat, non-rolled amount"
    );
    assert_eq!(hp_before, 524, "Snorlax max/starting HP is 524 (sanity)");

    // GROUND TRUTH (probe_fixeddamage_regression_rng.js, FD-LEVEL): the post-turn seed == the
    // real Showdown seed (draws = the ST accuracy roll + the Quick Claw = 2; NO crit, NO damage
    // roll). A model that drew a crit and/or a damage roll would draw MORE → this diverges.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "41762,18770,8812,43906",
        "post-turn seed == the real Showdown seed for a Seismic Toss (accuracy-only draw; \
         a spurious crit/damage roll desyncs)"
    );
}

// ============================================================================
// FD2 — SEISMIC TOSS (Fighting) into a GHOST is IMMUNE (0×): the accuracy roll is drawn
//       THEN `-immune` (the SAME draw count as a landed hit — NO crit, NO damage roll). WRONG
//       (a naive model): a "never-miss on immune" short-circuit (skip the accuracy roll) OR a
//       crit/damage roll would desync. SEED pin (== the landed-hit seed: acc + Quick Claw) +
//       STATE pin (the Ghost takes ZERO). Ground truth from the probe (FD-GHOST).
// ============================================================================

/// FD2: Seismic Toss into a Ghost (Gengar) is IMMUNE — the accuracy roll is drawn, then
/// `-immune`, with the SAME draw count as a landed hit (Gengar takes ZERO). WRONG (pre-fix):
/// a fixed-damage move that skipped the accuracy roll on an immune target, or drew a
/// crit/damage roll, would desync. The post-turn seed is IDENTICAL to FD1's (acc + Quick Claw).
#[test]
fn seismic_toss_into_a_ghost_is_immune_accuracy_only_seed() {
    let d = dex();
    // p1 Machamp Seismic Tosses a Gengar (Ghost — Fighting 0× → IMMUNE); Gengar Splashes. The
    // draw model is the SAME as a landed hit (accuracy drawn THEN -immune), so the post-turn
    // seed equals FD1's exactly (same init seed, same draw count).
    let machamp = "Machamp|||guts|seismictoss|Adamant|252,252,,,,|||||";
    let gengar = "Gengar|||levitate|splash|Timid|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(machamp, gengar, "53303,35262,36397,29520"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let ghost_hp = st.sides[1].pokemon[0].hp;

    // Turn 1: p1 Seismic Toss into the Ghost (IMMUNE) ; p2 Splash.
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // STATE: the Ghost took ZERO (Fighting 0× vs Ghost — the immunity short-circuit).
    assert_eq!(
        st.sides[1].pokemon[0].hp, ghost_hp,
        "a Ghost is IMMUNE to Seismic Toss (Fighting 0×) — it takes ZERO damage"
    );
    assert_eq!(st.sides[1].pokemon[0].status, None, "no status — Seismic Toss has no secondary");

    // GROUND TRUTH (probe_fixeddamage_regression_rng.js, FD-GHOST): the post-turn seed == the
    // real Showdown seed — IDENTICAL to the landed FD1 case (the accuracy roll is drawn THEN
    // -immune; SAME draw count). A model that skipped the accuracy roll on immunity, or drew a
    // crit/damage roll, would produce a DIFFERENT seed.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "41762,18770,8812,43906",
        "post-turn seed == the real Showdown seed for an IMMUNE Seismic Toss (accuracy drawn \
         THEN -immune — the SAME draw count as a landed hit)"
    );
}

// ============================================================================
// FD3 — NIGHT SHADE (Ghost) into a NORMAL is IMMUNE (0×): zero damage, `-immune`, accuracy
//       drawn. The symmetric immunity to FD2 (the OTHER 0× fixed-damage pairing). STATE pin
//       (the Normal takes ZERO) + SEED pin (== the accuracy-only draw). Ground truth (FD-NORMAL).
// ============================================================================

/// FD3: Night Shade into a Normal (Snorlax) is IMMUNE (Ghost 0× vs Normal) — zero damage,
/// `-immune`, accuracy drawn. WRONG (a naive model): applying the level damage regardless of
/// the type immunity would chip the Snorlax by 100; skipping the accuracy roll on immunity
/// would desync the seed. STATE + SEED pin.
#[test]
fn night_shade_into_a_normal_is_immune() {
    let d = dex();
    // p1 Gengar Night Shades a Snorlax (Normal — Ghost 0× → IMMUNE); Snorlax Splashes. Accuracy
    // drawn THEN -immune (same draw count as a landed hit), so the seed == FD1/FD2's.
    let gengar = "Gengar|||levitate|nightshade|Timid|252,,,,,|||||";
    let snorlax = "Snorlax|||immunity|splash|Careful|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(gengar, snorlax, "53303,35262,36397,29520"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let normal_hp = st.sides[1].pokemon[0].hp;

    // Turn 1: p1 Night Shade into the Normal (IMMUNE) ; p2 Splash.
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // STATE: the Normal took ZERO (Ghost 0× vs Normal).
    assert_eq!(
        st.sides[1].pokemon[0].hp, normal_hp,
        "a Normal is IMMUNE to Night Shade (Ghost 0×) — it takes ZERO damage (NOT 100)"
    );

    // GROUND TRUTH (probe_fixeddamage_regression_rng.js, FD-NORMAL): the post-turn seed == the
    // real Showdown seed (accuracy drawn THEN -immune). A model that dealt the 100 anyway (STATE)
    // or skipped the accuracy roll (SEED) trips this.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "41762,18770,8812,43906",
        "post-turn seed == the real Showdown seed for an IMMUNE Night Shade (accuracy drawn \
         THEN -immune)"
    );
}

// ============================================================================
// FD4 — A FIXED-DAMAGE MOVE INTO A SUBSTITUTE hits the SUB (the fixed NUMBER hits the sub's
//       HP, breaks with no carry), NOT the mon — using the EXISTING sub-absorb machinery. The
//       draw model is unchanged (accuracy-only). WRONG (a naive model): a fixed-damage move
//       that hit the mon's HP directly (ignoring the sub) would drop the mon's HP. STATE pin
//       (the sub HP drops 131→31, the mon's HP unchanged) + SEED pin. Ground truth (FD-SUB).
// ============================================================================

/// FD4: Seismic Toss (100) into a held Substitute hits the SUB (131 → 31, survives), NOT the
/// mon — the fixed NUMBER routes through the existing `absorb_into_sub`. WRONG (a naive model):
/// applying the fixed damage to the mon's HP directly (bypassing the sub) would drop the mon by
/// 100. The mon's OWN HP is unchanged; the sub HP drops by exactly 100.
#[test]
fn fixed_damage_into_a_substitute() {
    let d = dex();
    // p1 Machamp Seismic Tosses (100) into a p2 Snorlax that has a sub up (injected at
    // floor(524/4)=131, with the mon's HP docked by the cost — mirroring a real Substitute
    // cast). The 100 hits the SUB (131 → 31, survives); Snorlax Splashes.
    let machamp = "Machamp|||guts|seismictoss|Adamant|252,252,,,,|||||";
    let snorlax = "Snorlax|||immunity|splash|Careful|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(machamp, snorlax, "53303,35262,36397,29520"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // Inject the sub exactly like a real Substitute cast: pay floor(maxhp/4) HP, sub at that HP.
    let lax = &mut st.sides[1].pokemon[0];
    let sub_hp = lax.maxhp / 4; // 131
    lax.hp -= sub_hp;
    lax.substitute = Some(sub_hp);
    let mon_hp_after_sub = lax.hp; // 393 — must NOT change (the sub takes the ST)

    // Turn 1: p1 Seismic Toss (100) INTO the sub ; p2 Splash.
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // STATE: the SUB absorbed the 100 (131 → 31, survives); the mon's OWN HP is UNCHANGED.
    assert_eq!(
        st.sides[1].pokemon[0].substitute,
        Some(sub_hp - 100),
        "the sub absorbed the Seismic Toss (131 → 31) — the fixed NUMBER hits the sub's HP"
    );
    assert_eq!(
        st.sides[1].pokemon[0].hp, mon_hp_after_sub,
        "the mon's OWN HP is unchanged — the sub took the fixed damage, not the mon"
    );

    // GROUND TRUTH (probe_fixeddamage_regression_rng.js, FD-SUB): the post-turn seed == the real
    // Showdown seed (the draw model is UNCHANGED by the sub — accuracy + Quick Claw).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "41762,18770,8812,43906",
        "post-turn seed == the real Showdown seed for a Seismic Toss into a sub (the sub-absorb \
         does not change the fixed-damage draw model)"
    );
}

// ============================================================================
// DR1 — the DOUBLE-FAINT → DOUBLE-REPLACEMENT → SPIKES-CASCADE state bug (found by the
//       e2e fuzz when Explosion was re-admitted, e2e_9 / e2e_194). When a mutual double
//       faint forces BOTH sides to replace, both fresh entrants enqueue an order-101
//       `runSwitch`. If the FIRST runSwitch to run FAINTS its own entrant (its own side's
//       Spikes KO on entry — the cascade), gen-3 singles `faintMessages` fires
//       `cancelAction` over `getAllActive()` (battle.ts:2606-2616), which REMOVES the
//       OTHER side's still-pending `runSwitch`. So the OTHER entrant's Spikes chip is NEVER
//       applied — it stays at FULL HP. The WRONG (pre-fix) behaviour: the port's
//       `cancel_active_actions` did NOT cancel a pending `RunSwitch`, so the stale foe
//       runSwitch survived the cascade + re-applied the foe's Spikes to its already-settled
//       entrant (e2e_9: the foe's fresh Jirachi was wrongly chipped, e.g. 403 → 353). SEED
//       bit-for-bit (a queue splice is draw-free) — a pure STATE (HP) mis-application.
//       CONSTRUCTED + INJECTED (mirrors the sim probe): raw seed 11,22,33,44, 3 Spikes on
//       p1 side + 1 Spike on p2 side, p1's Jolteon (FAST → its runSwitch first) pre-damaged
//       to 1 HP so its own 3-layer Spikes KO it on entry. Ground truth (per-decision seeds +
//       the UNCHIPPED p2 entrant HP) from
//       `harness/probe_double_replacement_cascade_regression_rng.js`.
// ============================================================================

/// DR1: a double-faint → double-replacement where p1's fresh entrant faints on its OWN side's
/// Spikes (its runSwitch runs FIRST) must DROP p2's still-pending runSwitch → p2's fresh entrant
/// is NEVER chipped by p2's Spikes (stays FULL HP). WRONG (pre-fix): the stale foe runSwitch
/// survived + re-applied p2's Spikes chip to its already-settled entrant. STATE pin (the
/// UNCHIPPED p2 entrant HP is the direct fix witness) + the per-decision SEED (a queue splice is
/// draw-free → the seed must be untouched). Reverting the `RunSwitch` arm of `cancel_active_actions`
/// trips the HP assertion (p2 entrant re-chipped).
#[test]
fn double_replacement_cascade_does_not_rechip_the_other_sides_entrant() {
    let d = dex();
    // p1: Electrode (mutual Explosion), Jolteon (the FAST grounded cascade entrant → its
    //   runSwitch sorts FIRST; pre-damaged to 1 HP so its own 3-layer Spikes KO it on entry),
    //   Sandshrew (the p1 cascade replacement). p2: Electrode (mutual Explosion) + Snorlax
    //   (the SLOW grounded entrant on the 1-Spike side — its runSwitch MUST be cancelled).
    let p1 = "Electrode|||NoAbility|explosion,splash|Serious|,252,,,,252|N||||\
              ]Jolteon|||VoltAbsorb|thunderbolt|Serious|,,,,,252|N||||\
              ]Sandshrew|||SandVeil|scratch|Serious||N|||20|";
    let p2 = "Electrode|||NoAbility|explosion,splash|Serious|,252,,,,252|N||||\
              ]Snorlax|||Immunity|bodyslam|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "11,22,33,44"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    // INJECT the board the probe injects: 3 Spikes on p1 side, 1 Spike on p2 side; pre-damage
    // p1's Jolteon (slot 1) to 1 HP so its own 3-layer Spikes KO it on entry.
    st.sides[0].spikes = 3;
    st.sides[1].spikes = 1;
    st.sides[0].pokemon[1].hp = 1;

    let script = vec![
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),      // mutual Explosion → DOUBLE faint
        ScriptDecision::both(Choice::Switch(1), Choice::Switch(1)), // p1 Jolteon (KO cascade) / p2 Snorlax (must be UNCHIPPED)
        ScriptDecision::one(0, Choice::Switch(2)),                  // p1 cascade Sandshrew
    ];
    let out = st.run_full_battle(&script, &d);
    assert_eq!(out.decisions.len(), 3, "double faint → double replace → p1 cascade = 3 boundaries");

    // dec0 — the mutual Explosion MOVE turn (both Electrodes fainted → the turn paused for the
    // DOUBLE forced replacement, recorded at the pause as a Move-request boundary).
    assert!(matches!(out.decisions[0].request, RequestKind::Move),
        "dec0 is the mutual-Explosion MOVE turn; got {:?}", out.decisions[0].request);
    assert!(out.decisions[0].active[0].fainted && out.decisions[0].active[1].fainted,
        "dec0: both Electrodes faint (mutual Explosion double faint)");

    // dec1 — the forced DOUBLE replacement (its request answered here). p1 Jolteon fainted on its
    // own 3-layer Spikes (its runSwitch ran FIRST) → the cascade. THE FIX WITNESS: p2 Snorlax is
    // at FULL HP (524) — its runSwitch was CANCELLED by the faint's `cancelAction(getAllActive())`,
    // so p2's 1-layer Spikes was NEVER applied. WRONG (pre-fix): Snorlax chipped floor(524/8)=65 → 459.
    assert!(matches!(out.decisions[1].request, RequestKind::ForceSwitch { force: [true, true] }),
        "dec1's request is the DOUBLE forced replacement; got {:?}", out.decisions[1].request);
    assert_eq!(out.decisions[1].active_species[1], "snorlax", "dec1: p2's fresh entrant is Snorlax");
    assert_eq!(
        out.decisions[1].active[1].hp, out.decisions[1].active[1].maxhp,
        "THE FIX: p2 Snorlax stays at FULL HP {} (its runSwitch was cancelled by the cascade faint \
         → p2's Spikes was NOT re-applied); a re-chip would drop it to {}",
        out.decisions[1].active[1].maxhp,
        out.decisions[1].active[1].maxhp - out.decisions[1].active[1].maxhp / 8
    );
    assert!(out.decisions[1].active[0].fainted, "dec1: p1's Jolteon entrant fainted on its own 3-layer Spikes (cascade)");

    // dec2 — the p1 CASCADE replacement (Sandshrew) takes p1's 3-layer Spikes (floor(56/4)=14);
    // p2 Snorlax is STILL at full HP (never touched by the cascade).
    assert!(matches!(out.decisions[2].request, RequestKind::ForceSwitch { force: [true, false] }),
        "dec2's request is the p1-only cascade replacement; got {:?}", out.decisions[2].request);
    assert_eq!(out.decisions[2].active_species[0], "sandshrew", "dec2: p1's cascade entrant is Sandshrew");
    assert_eq!(
        out.decisions[2].active[1].hp, out.decisions[2].active[1].maxhp,
        "p2 Snorlax remains at FULL HP through the p1 cascade (never re-chipped)"
    );

    // GROUND TRUTH (probe_double_replacement_cascade_regression_rng.js): the per-decision SEEDS ==
    // the real Showdown seeds. The cascade fix is a DRAW-FREE queue splice (cancelAction removes an
    // action, no PRNG) — so re-attributing the hazard must NOT perturb any seed. A wrong draw model
    // (e.g. the cancelled runSwitch's EntryHazard/ability-Start drawing) would trip these.
    assert_eq!(seed_str(&out.decisions[0].seed_after), "38085,56695,39077,36349",
        "dec0 (mutual-Explosion double faint) seed == real Showdown");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "60833,51486,28767,2196",
        "dec1 (forced double replace, p2 entrant UNCHIPPED) seed == real Showdown");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "21177,35776,56648,13607",
        "dec2 (p1 cascade) seed == real Showdown");
}

// ============================================================================
// DR2 — the CONFUSION self-hit CHOICE-BAND fold (found by the same e2e re-admission, e2e_194
//       dec15). gen-4 confusion (which gen-3 inherits, data/mods/gen4/conditions.ts:74-83) runs
//       `this.actions.getDamage(pokemon, pokemon, 40)` — the FULL `getDamage`, so the attacker's
//       `onModifyAtk` item **Choice Band ×1.5 (physical)** folds into the typeless self-hit. The
//       WRONG (pre-fix) behaviour: `apply_confusion_self_hit` passed NO atk stat mods → it used
//       the stored Atk (not the CB-boosted Atk) → the self-hit UNDER-dealt (a Choice-Band
//       Aerodactyl's self-hit used Atk 339, not the CB 508 → the mon kept too much HP). SEED
//       bit-for-bit (the self-hit draws the SAME random(1,2) + random(16) either way) — a pure
//       STATE (HP) mis-application. CONSTRUCTED + INJECTED: raw seed 7,11,13,17, a confused
//       Choice-Band Aerodactyl vs a passive Clear-Body Regirock (Splash); the seed lands on the
//       self-hit. Ground truth (the post-turn HP + seed) from
//       `harness/probe_confusion_choiceband_regression_rng.js`.
// ============================================================================

/// DR2: a confused Choice-Band mon's self-hit folds Choice Band (the gen-4 confusion runs the FULL
/// `getDamage`, so the CB ×1.5 physical applies). WRONG (pre-fix): the self-hit used the stored Atk
/// (no CB) → it under-dealt (the mon kept too much HP). STATE pin (the CB-boosted self-hit HP is the
/// direct witness) + the post-turn SEED (the draw model — random(1,2) + random(16) — is unchanged by
/// the CB fold). Reverting to `atk_stat_mods: Vec::new()` leaves the mon at MORE HP → trips the HP
/// assertion.
#[test]
fn confusion_self_hit_applies_choice_band() {
    let d = dex();
    // Aerodactyl (Choice Band), confused, vs Regirock (Clear Body) Splashing. Aerodactyl's Atk is
    // 339 stored / 508 with Choice Band. On its turn it hits ITSELF (the 50% confusion roll) — the
    // self-hit MUST use the CB-boosted Atk. Regirock only Splashes, so the only HP Aerodactyl loses
    // is its own CB-boosted self-hit.
    let aero = "Aerodactyl||choiceband|Pressure|rockslide|Adamant|,252,,,,252|N||||";
    let regi = "Regirock|||ClearBody|splash|Serious|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(aero, regi, "7,11,13,17"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    // INJECT confusion: counter 3 → decrements to 2 → the 50% self-hit check fires this turn.
    st.sides[0].pokemon[0].confusion = Some(3);
    let maxhp = st.sides[0].pokemon[0].maxhp; // 301

    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // STATE — the CB-boosted self-hit: Aerodactyl ends at 212/301 (took 89 = the CB-boosted
    // self-hit roll). WRONG (pre-fix, no CB): the self-hit would use Atk 339 not 508 → LESS damage
    // → Aerodactyl at a HIGHER HP. The exact 212 is the fix witness (ground truth from the probe).
    assert_eq!(
        out.decisions[0].active[0].hp, 212,
        "THE FIX: Aerodactyl's confusion self-hit folds Choice Band (Atk 508, not the stored 339) → \
         212/{maxhp} (took 89); a no-CB self-hit under-deals and leaves MORE HP"
    );
    assert!(!out.decisions[0].active[0].fainted, "Aerodactyl survives the self-hit");

    // GROUND TRUTH (probe_confusion_choiceband_regression_rng.js): the post-turn seed == real
    // Showdown (the self-hit draws random(1,2) + random(16); the CB fold is draw-free — a stat mod,
    // not a roll — so the seed is UNCHANGED by the fix).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "60880,31090,7619,34922",
        "post-turn seed == the real Showdown seed for a confused Choice-Band self-hit (CB is a \
         stat mod, not a draw → the seed is unchanged)"
    );
}

// ============================================================================
// FR-RESUME — the FORCED-REPLACEMENT REQUEST-BOUNDARY RESUME (the last 2 protocol
//      scenarios' blocker). After a mid-turn forced replacement changes the active mon
//      to one with FEWER moves, a scripted `move K` the NEW mon doesn't have is REJECTED
//      by the sim's `side.choose` ("Your <mon> doesn't have a move K") — it draws
//      NOTHING and leaves `requestState === 'move'` OPEN, so the real turn runs on the
//      NEXT valid submission. The omniscient capture (submitting from a stale per-turn
//      plan) records that rejection as a PHANTOM zero-draw `move` decision whose
//      `seedAfter` equals the prior boundary's. WRONG (pre-fix): the port's
//      `run_full_battle` RAN a full turn for that invalid decision (its own move no-op'd,
//      but the FOE's move + residual + Quick Claw drew) → the seed + boundary mapping
//      diverged from the phantom onward. FIX: `run_full_battle` validates a top-of-turn
//      `move` decision (`move_decision_is_legal`) and SKIPS an out-of-range slot (run no
//      turn, draw nothing, record nothing) — mirroring the sim's reject-and-re-request.
//      DRAW-free (observation-only). Constructed: a 3-move Aerodactyl is KO'd by Zapdos'
//      Thunderbolt → p1 replaces with a 2-move Snorlax → a scripted `move 3` (invalid for
//      Snorlax) must be SKIPPED, and the real `move 1` turn runs at the sim's seed.
//      Ground truth from `harness/probe_forced_replacement_resume_regression_rng.js`.
// ============================================================================

/// FR-RESUME: an invalid `move` slot submitted right after a forced replacement (the
/// new mon has fewer moves) is SKIPPED — the port re-pulls the next decision and runs the
/// REAL post-replacement turn, with byte-identical seeds. WRONG (pre-fix): the port ran a
/// full turn for the invalid decision → the seed + the whole boundary mapping diverged.
/// This is the fix that un-deferred `status_para_and_boost_drop` / `secondary_status_flinch`
/// in `protocol_test.rs`. Asserts BOTH the decision STRUCTURE (3 boundaries, phantom
/// dropped) and the per-boundary SEED (draw-free — the phantom consumes nothing).
#[test]
fn forced_replacement_resume_runs_the_post_replacement_move_decision() {
    let d = dex();
    // p1: a 3-move Aerodactyl (KO'd by Zapdos' Thunderbolt — Electric SE on Aero/Flying)
    // + a 2-move Snorlax replacement. p2: Zapdos (Thunderbolt) + Blissey. Switch-in is
    // DRAW-FREE (no weather ability / Intimidate), so the port's init seed == the sim's
    // pre-first-decision seed (fed here as the >start seed — the same convention the
    // protocol replay uses).
    let p1 = "Aerodactyl||NoItem|RockHead|earthquake,rockslide,ancientpower|Adamant|,252,,,,252|N||||\
              ]Snorlax||NoItem|Immunity|bodyslam,earthquake|Adamant|252,252,,,,|N||||";
    let p2 = "Zapdos||NoItem|Pressure|thunderbolt,roost|Modest|,,,252,,252|N||||\
              ]Blissey||NoItem|NaturalCure|seismictoss,icebeam|Bold|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "13127,45333,18295,15391"), &d).expect("start");
    let st = battle.state_mut().expect("state");

    // The submitted decisions — INCLUDING the phantom `Move(2)` (a `move 3` for the 2-move
    // Snorlax, invalid → must be SKIPPED):
    //   dec0: Move/Move — Aerodactyl EQ (immune vs Zapdos) / Zapdos Thunderbolt → KO Aero
    //   dec1: Switch (p1) — p1 replaces the KO'd Aerodactyl with Snorlax
    //   dec2: Move(2)/Move(0) — the PHANTOM (Snorlax has no move slot 2) → SKIPPED, draws 0
    //   dec3: Move(0)/Move(0) — the REAL post-replacement turn runs (Snorlax Body Slam)
    let choices = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::one(0, Choice::Switch(1)),
        ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // phantom (invalid slot)
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];
    let out = st.run_full_battle(&choices, &d);

    // STRUCTURE — exactly THREE real boundaries (the phantom is DROPPED, not run). WRONG
    // (pre-fix): the port ran a 4th turn for the phantom → 4 boundaries with wrong seeds.
    assert_eq!(
        out.decisions.len(),
        3,
        "the invalid `move 3` decision is SKIPPED (not run as a turn) → 3 real boundaries; \
         under the bug the port would run it as a full turn → 4 boundaries"
    );
    assert!(matches!(out.decisions[0].request, RequestKind::Move));
    assert!(matches!(out.decisions[1].request, RequestKind::ForceSwitch { force: [true, false] }));
    assert!(matches!(out.decisions[2].request, RequestKind::Move));
    // The replacement brought Snorlax to the active slot; the real turn ran on Snorlax.
    assert_eq!(out.decisions[1].active_species[0], "snorlax", "p1 replaced Aerodactyl with Snorlax");
    assert_eq!(out.decisions[2].active_species[0], "snorlax", "the real post-replacement turn ran on Snorlax");

    // SEED — each boundary's post-decision seed == the REAL Showdown seed (ground truth
    // from probe_forced_replacement_resume_regression_rng.js). Because the phantom is
    // zero-draw, dropping it leaves EVERY real boundary's seed byte-identical to the sim.
    let expected = [
        "55318,8071,46680,56242",  // dec0 (move) — Aerodactyl KO'd by Thunderbolt
        "38085,56695,39077,36349", // dec1 (forced switch) — Snorlax in
        "18621,25793,18448,35836", // dec2 (real move turn — phantom skipped)
    ];
    for (i, exp) in expected.iter().enumerate() {
        assert_eq!(
            seed_str(&out.decisions[i].seed_after),
            *exp,
            "boundary {i} seed == the real Showdown seed; under the bug the port would have \
             RUN the invalid `move 3` decision as a turn (drawing the foe move + residual + \
             Quick Claw) → this seed desyncs from decision 2 onward"
        );
    }
}

// ============================================================================
// PP1-PP4 — PP TRACKING + STRUGGLE (`gen3_pp_tracking_v1`). Deterministic, revert-verified
//           pins over CONSTRUCTED gen3customgame scenarios (fixed seed [1,2,3,4], scripted
//           choices via the public `run_full_battle` harness). Ground truth (post-decision
//           SEED + HP + PP) copied VERBATIM from `harness/probe_pp_struggle_regression_rng.js`.
// ============================================================================

/// PP1: a normal move decrements the USED slot's PP by 1, DRAW-FREE. Suicune Surf (m0, 24 PP)
/// into a bulky Snorlax → Surf pp 24→23 after one turn, and the post-turn SEED matches the real
/// Showdown seed (the decrement adds NO draw — the seed is a pure function of the move's own
/// acc/crit/damage + residual + Quick Claw). WRONG (a decrement that drew a PRNG value) would
/// desync the seed; a wrong decrement count would diverge the PP STATE.
#[test]
fn pp_decrements_on_use_draw_free() {
    let d = dex();
    // Teams packed EXACTLY as probe_pp_struggle_regression_rng.js (Serious, the same EVs) so
    // the ground-truth seed lines up bit-for-bit.
    let suicune = "Suicune|||NoAbility|surf,icebeam|Serious|252,,,252,,|N||||";
    let snorlax = "Snorlax|||Immunity|splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(suicune, snorlax, "30982,33910,19571,50263"), &d).expect("start");
    let out = battle
        .state_mut()
        .expect("state")
        .run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);

    // PP STATE: Surf (slot 0) decrements 24→23; Ice Beam (slot 1) UNCHANGED at 16.
    assert_eq!(
        out.decisions[0].active[0].move_pp, [23, 16, -1, -1],
        "Surf's PP drops 24→23 (−1 per use); Ice Beam is untouched"
    );
    // SEED: the decrement is DRAW-FREE — the post-turn seed == the real Showdown seed.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "55250,62519,52978,42619",
        "PP decrement is DRAW-FREE — the post-turn seed matches the real sim; a decrement that \
         consumed a PRNG value (or a wrong draw model) would desync here"
    );
}

/// PP2: a move TARGETING a Pressure holder decrements 2 PP (not 1), DRAW-FREE. Snorlax Body Slam
/// (m0, 24 PP) into a Pressure Suicune → bodyslam pp 24→22, and the post-turn seed matches the
/// real Showdown seed (Pressure's `DeductPP` is a deterministic modifier — no RNG). WRONG (a −1
/// decrement, i.e. ignoring Pressure) would diverge the PP STATE.
#[test]
fn pressure_decrements_two_pp() {
    let d = dex();
    // Teams packed EXACTLY as the probe (Serious; the same EVs).
    let snorlax = "Snorlax|||Immunity|bodyslam,splash|Serious|252,252,,,,|N||||";
    let suicune = "Suicune|||Pressure|splash|Serious|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax, suicune, "30982,33910,19571,50263"), &d).expect("start");
    let out = battle
        .state_mut()
        .expect("state")
        .run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);

    // PP STATE: Body Slam (slot 0) drops 24→22 (−2 into a Pressure holder); Splash untouched.
    assert_eq!(
        out.decisions[0].active[0].move_pp, [22, 64, -1, -1],
        "Body Slam into a Pressure Suicune drops 24→22 (−2, the Pressure extra); WRONG = −1"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37673,46633,62039,8266",
        "the Pressure −2 is DRAW-FREE (DeductPP is a deterministic modifier) — the post-turn seed \
         matches the real sim"
    );
}

/// PA1 (`gen3_pressure_allyteam_v1`, the e2e_182 root cause): an **`allyTeam`** move
/// (Aromatherapy / Heal Bell) under a **Pressure** foe deducts ONE PP, NOT two — the
/// Pressure extra fires ONLY when the Pressure foe is in the move's `pressureTargets`
/// (a FOE-directed target), and an ally/self/foeSide move never puts the foe there
/// (`Pokemon.getMoveTargets`, pokemon.ts:854-861). Blissey's Aromatherapy (slot 0,
/// target=allyTeam, 8 PP) into a Pressure Zapdos → PP 8→**7** (−1); a control ThunderWave
/// (slot 1, target=normal, 32 PP) into the same Pressure foe → 32→**30** (−2). Both are
/// DRAW-FREE so both post-turn seeds match the real sim. WRONG (the pre-fix `!targets_self`
/// predicate) deducts 2 for Aromatherapy (8→6), which — over a stall battle — drains its PP
/// early and makes the port REJECT a legitimate late Aromatherapy as out-of-PP, shifting the
/// whole script (the e2e_182 decision-count + state desync). Ground truth from
/// `harness/probe_pressure_allyteam_rng.js`.
#[test]
fn pressure_does_not_add_pp_for_an_allyteam_move() {
    let d = dex();
    // Packed EXACTLY as the probe so the ground-truth seeds line up bit-for-bit.
    let blissey = "Blissey||Leftovers|NaturalCure|Aromatherapy,ThunderWave,SeismicToss,SoftBoiled|Bold|252,,252,,4,|F||||";
    let zapdos = "Zapdos||Leftovers|Pressure|Thunderbolt,Roost,Rest,ThunderWave|Modest|252,,,252,,|N||||";
    // The seed is the POST-construction seed (the probe's SEED_BEFORE) — the port's
    // `start_with_switchins` skips the sim's turn-0 construction draws, so it must start where
    // the sim is right before the first decision.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(blissey, zapdos, "57890,13032,12358,42006"), &d).expect("start");
    // dec0: p1 Aromatherapy (allyTeam) into Pressure Zapdos; p2 Rest (self, so it draws nothing extra).
    // dec1: p1 ThunderWave (foe move) into the same Pressure foe; p2 Rest.
    let out = battle.state_mut().expect("state").run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(2)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(2)),
        ],
        &d,
    );

    // dec0: Aromatherapy (allyTeam, slot 0) drops 8→7 (−1, NOT the Pressure −2). The FOE-target
    // slots are untouched (twave 32, stoss 32, sboiled 16). WRONG (pre-fix) = 8→6.
    assert_eq!(
        out.decisions[0].active[0].move_pp, [7, 32, 32, 16],
        "Aromatherapy (allyTeam) under a Pressure foe deducts 1 PP (8→7), NOT 2 — the foe is not \
         in an allyTeam move's pressureTargets; the pre-fix !targets_self predicate wrongly gave 8→6"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "43514,9542,40559,8561",
        "the allyTeam PP deduct is DRAW-FREE — the post-turn seed matches the real sim"
    );
    // dec1 (control): ThunderWave (target=normal, foe-directed) DOES get the Pressure −2 → 32→30.
    assert_eq!(
        out.decisions[1].active[0].move_pp, [7, 30, 32, 16],
        "ThunderWave (a FOE-directed move) into a Pressure foe correctly deducts 2 PP (32→30) — \
         the fix keeps the real Pressure extra for foe-targeting moves"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "55250,62519,52978,42619",
        "the foe-move Pressure −2 is DRAW-FREE — the post-turn seed matches the real sim"
    );
}

/// PA2 (`gen3_pressure_foeside_v1`, the per-side/request byte-fuzz round-4 find): a **`foeSide`**
/// move (Spikes — the only gen-3 `foeSide` move) under a **Pressure** foe deducts **TWO** PP (the
/// Pressure extra fires because `foeSide` DOES put the Pressure foe in the move's `pressureTargets`).
/// SIM-PROBE-CONFIRMED (`/tmp/probe_spikes_pressure.js`): Skarmory Spikes (slot 0, target foeSide,
/// 32 PP) into a Pressure Suicune → PP 32→**30** (−2); vs a non-Pressure foe → 32→**31** (−1). It is
/// DRAW-FREE (DeductPP is a deterministic modifier) so the post-turn seed matches the real sim.
/// WRONG (the pre-fix predicate excluded `foeSide` alongside `allyTeam`): 32→31 — invisible to the
/// OMNISCIENT byte fuzzer (no PP in the `|...|` stream), but the request-JSON `pp` field diverges.
/// This does NOT touch the e2e_182 `allyTeam` case (PA1) — `allyTeam` stays −1. Ground truth from
/// `/tmp/probe_spikes_pressure_seed.js`.
#[test]
fn pressure_adds_pp_for_a_foeside_spikes_move() {
    let d = dex();
    let skarmory = "Skarmory||Leftovers|KeenEye|Spikes,DrillPeck|Serious||M||||";
    let suicune = "Suicune||Leftovers|Pressure|Surf,Splash|Serious||N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(skarmory, suicune, "30982,33910,19571,50263"), &d)
            .expect("start");
    // dec0: p1 Spikes (foeSide, slot 0) into a Pressure Suicune; p2 Splash (slot 1, draw-free)
    // so the ONLY turn draws are the shared action-order/Quick-Claw ones — matching the probe.
    let out = battle
        .state_mut()
        .expect("state")
        .run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(1))], &d);
    // Spikes (foeSide, slot 0) drops 32→30 (−2, the Pressure extra); DrillPeck untouched.
    assert_eq!(
        out.decisions[0].active[0].move_pp[0], 30,
        "Spikes (foeSide) under a Pressure foe deducts 2 PP (32→30) — foeSide IS in the Pressure \
         foe's pressureTargets; the pre-fix predicate wrongly gave 32→31"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "57890,13032,12358,42006",
        "the foeSide Pressure −2 is DRAW-FREE — the post-turn seed matches the real sim"
    );
}

/// PA3 (`gen3_pressure_allyteam_v1`, the byte-fuzz 5_6 / D1 deep-seed find): a **NON-GHOST**
/// user's **Curse** is RE-TARGETED to `self` at runtime (`curse.onModifyMove` → `nonGhostTarget`),
/// so under a **Pressure** foe it deducts **ONE** PP, not two — the STATIC dex `target` is
/// `"normal"` (which would put the foe in the move's `pressureTargets` → −2), but the RUNTIME
/// target is `self` (foe NOT in `pressureTargets` → −1). WRONG (the pre-fix `pressure_targets_foe`
/// reading the static `"normal"`): 2 PP/turn → a Swampert's 16 Curse PP drains ~1 cycle early →
/// forced Struggle turns the sim still Curses → the deep-seed desync (ou dec159 / cg dec143). The
/// PP deduction is DRAW-FREE, so the direct revert-catch is the PP STATE (dec1: fixed 14 vs broken
/// 12; by dec7 the broken port has 0 Curse PP → Struggle → wrong atkBoost + a diverged seed). Ground
/// truth `harness/probe_pressure_curse_regression_rng.js` (Swampert Curse vs a Pressure Zapdos that
/// Agilities each turn — draw-free, stays faster than the Curse-slowed Swampert so no ties).
#[test]
fn pressure_does_not_add_pp_for_a_nonghost_curse() {
    let d = dex();
    let swampert = "Swampert||Leftovers|Torrent|Curse,Surf,Earthquake,IcePunch|Relaxed|252,,252,,4,|N||||";
    let zapdos = "Zapdos||Leftovers|Pressure|Agility,Thunderbolt,Rest,ThunderWave|Modest|252,,,252,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(swampert, zapdos, "57890,13032,12358,42006"), &d)
            .expect("start");
    // 9 decisions: p1 Swampert Curse (slot 0), p2 Zapdos Agility (slot 0, self, draw-free).
    let script: Vec<ScriptDecision> =
        (0..9).map(|_| ScriptDecision::both(Choice::Move(0), Choice::Move(0))).collect();
    let out = battle.state_mut().expect("state").run_full_battle(&script, &d);

    // dec0: the first Curse under Pressure deducts 1 PP (16→15), NOT 2 (→14). +1 Atk applied.
    assert_eq!(
        out.decisions[0].active[0].move_pp[0], 15,
        "a non-Ghost Curse (runtime target=self) under a Pressure foe deducts 1 PP (16→15), NOT 2 — \
         the pre-fix predicate read the static target=\"normal\" and gave 16→14"
    );
    assert_eq!(out.decisions[0].active[0].boosts[0], 1, "Curse raises the user's Atk +1");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "8769,17248,37115,18776",
        "the non-Ghost Curse selfDrops random(100) is the only per-move draw — the PP deduct is \
         DRAW-FREE, so the post-turn seed matches the real sim"
    );
    // dec1: still −1/turn (16→14, NOT 16→12).
    assert_eq!(
        out.decisions[1].active[0].move_pp[0], 14,
        "the second Curse deducts 1 more PP (→14), NOT 2 (→12)"
    );
    // dec7: Curse PP is 8 (16 − 8). With the pre-fix −2, PP would be 0 by dec7 → the port would be
    // FORCED to Struggle (Atk stuck at +6, no more Curse) and the seed would diverge here.
    assert_eq!(
        out.decisions[7].active[0].move_pp[0], 8,
        "after 8 Curses at −1 PP each, Curse PP is 8 (NOT 0 — the pre-fix −2 exhausts it, forcing a \
         Struggle turn the sim still Curses)"
    );
    assert_eq!(
        seed_str(&out.decisions[7].seed_after),
        "61380,11535,34528,49510",
        "dec7 still Curses (selfDrops draw), matching the sim — the pre-fix Struggle-at-dec7 would \
         diverge the seed from here on"
    );
}

/// PP3 + PP4: a mon with NO usable move is FORCED to Struggle, and gen-3 Struggle recoil is
/// `max(floor(damageDealt / 4), 1)` (the `recoil:[1,4]` path, NOT struggleRecoil = maxhp/4).
/// A CHOICE-BAND Snorlax with Extreme Speed (m0, 8 PP) LOCKS to it and spams it 8× into a
/// LEVITATE Gengar (Extreme Speed [Normal] → `-immune`, 0 damage, but PP still −1). After 8
/// uses ES is at 0 PP and the other slots are Choice-disabled → decision 8 (0-based) FORCES
/// Struggle. Struggle (typeless '???') HITS the Ghost; Gengar Splashes (no damage to Snorlax),
/// so Snorlax's HP loss on the Struggle turn is EXACTLY the recoil. Ground truth: Struggle deals
/// Gengar 324→194 (= 130), recoil = floor(130/4) = 32 → Snorlax 524→492; post-turn seed matches.
/// WRONG models: not modeling the Choice lock (never forces Struggle — the engine keeps rejecting
/// the 0-PP ES with other slots "available"); using the gen4+ maxhp/4 recoil (would deal 131, not
/// 32); using `round` not `floor` (would deal 33, not 32).
#[test]
fn no_usable_move_forces_struggle_and_struggle_recoil_is_gen3_quarter_damage_dealt() {
    let d = dex();
    // Choice-Band Snorlax: Extreme Speed (slot 0, 8 PP) + 3 fillers it never gets to use (the
    // Choice lock disables them once ES is used). Packed EXACTLY as the probe (Serious).
    let snorlax = "Snorlax||ChoiceBand|Immunity|extremespeed,bodyslam,crunch,shadowball|Serious|252,252,,,,|N||||";
    let gengar = "Gengar|||Levitate|splash|Serious|252,,,,,252|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax, gengar, "30982,33910,19571,50263"), &d).expect("start");
    // 9 decisions: 8× Extreme Speed (immune, pp 8→0), then the forced Struggle.
    let script: Vec<ScriptDecision> =
        (0..9).map(|_| ScriptDecision::both(Choice::Move(0), Choice::Move(0))).collect();
    let out = battle.state_mut().expect("state").run_full_battle(&script, &d);

    assert_eq!(out.decisions.len(), 9, "all 9 decisions run (no decision was wrongly rejected)");

    // dec 7 (the LAST immune Extreme Speed): ES pp hit 0, no damage to Gengar (immune), Snorlax
    // full HP (ES is Normal into a Ghost → `-immune`, PP still −1 each turn).
    assert_eq!(
        out.decisions[7].active[0].move_pp, [0, 24, 24, 24],
        "after 8 immune Extreme Speeds, ES (slot 0) is at 0 PP (each immune hit STILL decrements); \
         the CB-disabled fillers are untouched"
    );
    assert_eq!(out.decisions[7].active[0].hp, 524, "Snorlax unhurt through 8 immune Extreme Speeds");
    assert_eq!(out.decisions[7].active[1].hp, 324, "Gengar unhurt (Normal ES is immune to the Ghost)");

    // dec 8 (the FORCED Struggle turn): Struggle HITS the Ghost (typeless) for 130 (324→194);
    // the recoil is floor(130/4) = 32 → Snorlax 524→492. ES PP stays 0 (Struggle is not a slot).
    assert_eq!(
        out.decisions[8].active[1].hp, 194,
        "Struggle (typeless '???') HITS the Levitate/Ghost Gengar (a typeless move has no type-chart \
         row → 1×) for 130 — NOT 0/immune"
    );
    assert_eq!(
        out.decisions[8].active[0].hp, 492,
        "gen-3 Struggle recoil = max(floor(damageDealt/4),1) = floor(130/4) = 32 → Snorlax 524→492 \
         (NOT the gen4+ maxhp/4 = 131, NOT round(130/4) = 33)"
    );
    assert_eq!(
        out.decisions[8].active[0].move_pp, [0, 24, 24, 24],
        "Struggle consumes NO PP (it is not a move slot) — ES stays at 0"
    );
    assert_eq!(
        seed_str(&out.decisions[8].seed_after),
        "30954,58228,1566,4235",
        "the Struggle turn draws acc + crit + damage (like a normal move) + Quick Claw; the recoil \
         is DRAW-FREE — the post-turn seed matches the real sim. A Struggle that mis-drew its \
         acc/crit/damage, or a mis-applied recoil, would desync here"
    );
}

// ============================================================================
// TAUNT + DISABLE layer (`gen3_taunt_disable_v1`) — the move-SELECTION-restriction
// pins. Ground-truth seeds + per-boundary state from
// `harness/probe_taunt_disable_regression_rng.js` (copied verbatim); the draw/
// duration semantics were settled by `harness/probe_disable_full_lifecycle.js` +
// `probe_taunt_duration_branch.js` + `probe_taunt_disable_onbeforemove_rng.js`.
//   TD1 a landed Taunt restricts the target's Status moves for EXACTLY the sim's
//       window (queued move cant'd; next selection restricted; then FREE) — and
//       the whole lifecycle is draw-free past the taunt's own acc-100 roll
//                    → `taunt_blocks_status_move_selection_for_the_sim_window_draw_free`
//   TD2 the Disable STORED duration per branch: FASTER disabler (target still to
//       move, willMove TRUE) stores random(2,6); SLOWER disabler (target already
//       moved) stores random(2,6)+1 — pinned at the exact FREE-UP boundary + seeds
//                    → `disable_duration_stored_per_branch_matches_sim`
//   TD3 Disable takes the only damaging move + Taunt takes the Status moves →
//       the target is FORCED to Struggle → `taunt_plus_disable_forces_struggle`
//   TD4 the onBeforeMove priority vs the para roll (taunt AFTER, disable BEFORE)
//                    → `taunt_and_disable_onbeforemove_priority_vs_paralysis`
//   TD5 Disable into a 0-PP lastMove: the onStart 0-PP guard rejects the volatile
//       AFTER the accuracy + random(2,6) draws (`harness/probe_disable_zero_pp_rng.js`)
//                    → `disable_into_a_zero_pp_lastmove_fails_draws_but_no_volatile`
// ============================================================================

/// TD1: a landed Taunt makes the target's Status moves un-selectable for EXACTLY the
/// sim's window, draw-free beyond the taunt's own accuracy roll.
/// Timeline (probe, seed "1,2,3,4"): dec0 Taunt lands + Blissey's QUEUED Thunder Wave
/// is cant'd at execution (`|cant|...|move: Taunt|Thunder Wave`) — the blocked action
/// draws NOTHING and deducts NO PP; dec1 is the RESTRICTED selection boundary (Thunder
/// Wave un-selectable — `move_usable` false); the taunt expires at dec1's residual
/// (`duration: 2` FIXED, no duration draw); dec2 Thunder Wave is selectable again and
/// RUNS — Aerodactyl ends up paralyzed (the free-up proof).
/// WRONG (a broken restriction): `move_usable` ignoring taunt lets Thunder Wave be
/// selected at dec1; a wrong duration (e.g. 3) keeps dec2's Thunder Wave rejected →
/// the decision is SKIPPED → count/seed mismatch.
#[test]
fn taunt_blocks_status_move_selection_for_the_sim_window_draw_free() {
    let d = dex();
    let aero = "Aerodactyl|||NoAbility|taunt,earthquake|Serious|,252,,,,252|N||||";
    let blissey = "Blissey|||NoAbility|thunderwave,icebeam|Serious|252,,252,,,|N||||";

    // (a) The mid-window restriction, asserted DIRECTLY on the live state: run dec0 only
    //     (Taunt lands; the queued Thunder Wave is cant'd), then check `move_usable`.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(aero, blissey, "30982,33910,19571,50263"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "dec0 ran");
    assert!(out.decisions[0].active[1].taunted, "Blissey is TAUNTED at the dec0 boundary");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "43514,9542,40559,8561",
        "dec0 (Taunt lands + the QUEUED Thunder Wave is cant'd at execution) draws ONLY the \
         taunt acc-100 roll + Quick Claw — the blocked action and the FIXED duration:2 are \
         draw-free (probe: draws=2)"
    );
    let lax = &st.sides[1].pokemon[st.sides[1].active];
    assert!(
        !lax.move_usable(0, &d),
        "mid-window: Thunder Wave (Status) is UN-selectable while taunted — reverting the \
         taunt arm of move_usable trips this"
    );
    assert!(lax.move_usable(1, &d), "mid-window: Ice Beam (Special) stays selectable");
    assert!(!lax.must_struggle(&d), "a usable attack remains → NOT forced to Struggle");
    assert_eq!(
        lax.move_pp[0],
        lax.move_maxpp[0],
        "the cant'd Thunder Wave deducted NO PP (deductPP runs only after BeforeMove passes)"
    );

    // (b) The FULL window + free-up, per-boundary (a fresh battle, same seed):
    //     dec1 = the restricted boundary (Blissey must Ice Beam; taunt expires at its
    //     residual); dec2 = Thunder Wave RUNS again → Aerodactyl is PARALYZED.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(aero, blissey, "30982,33910,19571,50263"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // taunt / (queued twave cant'd)
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // EQ / Ice Beam (restricted window)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // EQ / Thunder Wave (FREED)
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 3, "all three boundaries ran (a wrong taunt duration \
         would get dec2's Thunder Wave REJECTED as still-taunted → 2 decisions)");
    assert!(out.decisions[0].active[1].taunted, "dec0: taunted");
    assert!(
        !out.decisions[1].active[1].taunted,
        "dec1 boundary: the taunt EXPIRED at dec1's residual (duration 2: 2→1 at dec0's \
         residual, 1→0 at dec1's) — the sim window is exactly ONE restricted selection"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "7104,42574,9152,10809",
        "dec1 (EQ + Ice Beam under the restriction + the draw-free taunt expiry) seed"
    );
    assert_eq!(
        out.decisions[2].active[0].status,
        Some(Status::Paralysis),
        "dec2: the FREED Thunder Wave ran and paralyzed Aerodactyl — the free-up proof"
    );
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "56224,59251,18573,26068",
        "dec2 (EQ + the freed Thunder Wave) seed"
    );
    assert_eq!(out.decisions[2].active[1].hp, 323, "Blissey after two EQs (probe: 323/714)");
    assert_eq!(out.decisions[2].active[0].hp, 159, "Aerodactyl after the dec1 Ice Beam (159/301)");
}

/// TD2: the Disable STORED duration matches the sim on BOTH branches, pinned at the
/// exact FREE-UP boundary (+ per-decision seeds).
///   Branch A — FASTER disabler (probe, seed "3,4,5,6"): Suicune disables BEFORE
///   Snorlax moves (`willMove(target)` TRUE) → stored = random(2,6) = 3 here. The
///   disabled slot (EQ, slot 0) reads 0 at dec1/dec2 and FREES at dec3 (3 residual
///   ticks: dec1's own, dec2's, dec3's → 0).
///   Branch B — SLOWER disabler (probe, seed "1,8,14,22"): Suicune surfs FIRST, then
///   Snorlax disables (`willMove` FALSE) → stored = random(2,6)+1 = 2+1 = 3 here. The
///   disabled slot (Surf, slot 0) reads 0 at dec1/dec2 and FREES at dec3.
/// WRONG (the off-by-one): storing `rolled+1`/`rolled+2` (or `rolled-1`/`rolled`)
/// shifts the free-up boundary → the `disabled_slot` timeline diverges (and the
/// later scripted choices get rejected → count/seed mismatch). The two branches use
/// DIFFERENT rolled values behind the same stored 3, so a model that dropped the
/// branch conditional entirely also desyncs the seeds (the random(2,6) position).
#[test]
fn disable_duration_stored_per_branch_matches_sim() {
    let d = dex();

    // --- Branch A: FASTER disabler (stored = rolled). ---
    let suicune = "Suicune|||NoAbility|disable,surf|Serious|252,,252,,,|N||||";
    let snorlax = "Snorlax|||Immunity|earthquake,bodyslam|Serious|252,252,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(suicune, snorlax, "13659,34330,14156,55073"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // surf / EQ (lastMove = EQ)
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // DISABLE (faster) / queued EQ cant'd
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // surf / bodyslam
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // surf / bodyslam (FREE-UP boundary)
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // surf / bodyslam
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 5, "branch A: all five boundaries ran");
    let dis: Vec<i8> = out.decisions.iter().map(|r| r.active[1].disabled_slot).collect();
    assert_eq!(
        dis,
        vec![-1, 0, 0, -1, -1],
        "branch A (FASTER disabler, willMove TRUE): stored = random(2,6) = 3 → EQ (slot 0) is \
         disabled at dec1/dec2 and FREES at the dec3 boundary (ticks at dec1's own residual, \
         dec2's, dec3's). A +1 off-by-one keeps it disabled at dec3; a -1 frees it at dec2"
    );
    let seeds: Vec<String> = out.decisions.iter().map(|r| seed_str(&r.seed_after)).collect();
    assert_eq!(
        seeds,
        vec![
            "11268,27850,19520,61230".to_string(), // dec0: surf + EQ
            "51897,22645,25942,15411".to_string(), // dec1: disable acc + random(2,6); queued EQ cant'd DRAW-FREE
            "12048,49243,42554,54971".to_string(), // dec2
            "48057,27501,30477,18499".to_string(), // dec3: the free-up (draw-free -end)
            "35811,50068,17973,45259".to_string(), // dec4
        ],
        "branch A per-decision seeds == the real sim's (the disable turn draws accuracy(55) + \
         ONE random(2,6); the cant'd queued move + every residual tick draw NOTHING)"
    );

    // --- Branch B: SLOWER disabler (stored = rolled + 1). ---
    let snorlax_d = "Snorlax|||Immunity|disable,bodyslam|Serious|252,252,,,,|N||||";
    let suicune_t = "Suicune|||NoAbility|surf,icebeam|Serious|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax_d, suicune_t, "36553,11848,52306,28017"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // bodyslam / surf (lastMove = surf)
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // surf runs FIRST, then DISABLE (slower)
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // bodyslam / icebeam
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // bodyslam / icebeam (FREE-UP boundary)
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // bodyslam / icebeam
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 5, "branch B: all five boundaries ran");
    let dis: Vec<i8> = out.decisions.iter().map(|r| r.active[1].disabled_slot).collect();
    assert_eq!(
        dis,
        vec![-1, 0, 0, -1, -1],
        "branch B (SLOWER disabler, willMove FALSE): stored = random(2,6)+1 = 2+1 = 3 → Surf \
         (slot 0) is disabled at dec1/dec2 and FREES at the dec3 boundary. WITHOUT the +1 \
         (stored = 2) it would free at dec2 — this boundary is the branch proof"
    );
    let seeds: Vec<String> = out.decisions.iter().map(|r| seed_str(&r.seed_after)).collect();
    assert_eq!(
        seeds,
        vec![
            "61608,1996,47221,63033".to_string(),  // dec0: surf + bodyslam
            "11019,60253,13945,27495".to_string(), // dec1: surf FIRST, then disable acc + random(2,6); NO cant
            "40527,19422,36842,16526".to_string(), // dec2
            "37102,3788,55300,60049".to_string(),  // dec3: the free-up
            "61721,21360,23668,11744".to_string(), // dec4 (Body Slam's par30 lands on Suicune)
        ],
        "branch B per-decision seeds == the real sim's"
    );
    assert_eq!(
        out.decisions[4].active[1].status,
        Some(Status::Paralysis),
        "dec4: Body Slam's par30 secondary landed on Suicune (probe: 22/404 par)"
    );
}

/// TD3: Disable takes the only damaging move + Taunt takes the Status moves → the
/// target is FORCED to Struggle (the sim's request offers ONLY Struggle).
/// Timeline (probe, seed "15,16,26,34"): dec0 Blissey Ice Beams (lastMove = slot 2;
/// Gengar's Shadow Ball is Ghost→Normal IMMUNE all along — accuracy-only); dec1
/// Disable lands → Ice Beam disabled (+ the queued Ice Beam cant'd); dec2 Taunt lands
/// → Soft-Boiled/Toxic un-selectable too (+ the queued Soft-Boiled cant'd) — NOTHING
/// usable remains; dec3 Blissey STRUGGLES: it hits Gengar 184→169 (15) and takes
/// floor(15/4) = 3 recoil (714→711). Both volatiles expire at dec3's residual.
/// WRONG (a broken restriction): `must_struggle` ignoring taunt or disable lets
/// Blissey use a real move at dec3 → HP/seed diverge.
#[test]
fn taunt_plus_disable_forces_struggle() {
    let d = dex();
    let gengar = "Gengar|||NoAbility|shadowball,disable,taunt|Serious|,,,252,,252|N||||";
    let blissey = "Blissey|||NoAbility|softboiled,toxic,icebeam|Serious|252,,252,,,|N||||";

    // (a) The forced-Struggle state, asserted DIRECTLY: run dec0..dec2, then check
    //     `must_struggle` + per-slot `move_usable`.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(gengar, blissey, "23411,4748,19816,56877"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(2)), // shadowball(immune) / icebeam
            ScriptDecision::both(Choice::Move(1), Choice::Move(2)), // DISABLE icebeam / queued icebeam cant'd
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // TAUNT / queued softboiled cant'd
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 3, "dec0..dec2 ran");
    assert!(out.decisions[2].active[1].taunted, "dec2: Blissey is taunted");
    assert_eq!(out.decisions[2].active[1].disabled_slot, 2, "dec2: Ice Beam (slot 2) is disabled");
    let bliss = &st.sides[1].pokemon[st.sides[1].active];
    assert!(!bliss.move_usable(0, &d), "Soft-Boiled (Status) un-selectable: TAUNTED");
    assert!(!bliss.move_usable(1, &d), "Toxic (Status) un-selectable: TAUNTED");
    assert!(!bliss.move_usable(2, &d), "Ice Beam un-selectable: DISABLED");
    assert!(
        bliss.must_struggle(&d),
        "taunt (all Status moves) + disable (the only attack) leave NOTHING usable → \
         FORCED Struggle — reverting either arm of move_usable/must_struggle trips this"
    );
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "47343,31398,57866,38841",
        "dec2 (Taunt lands + the queued Soft-Boiled cant'd draw-free) seed"
    );

    // (b) The Struggle EXECUTES (a fresh battle, same seed, + dec3): typeless Struggle
    //     hits the Ghost for 15 (184→169) and Blissey takes floor(15/4) = 3 recoil.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(gengar, blissey, "23411,4748,19816,56877"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(2)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(2)),
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // shadowball(immune) / FORCED Struggle
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 4, "dec3 (the Struggle turn) ran");
    assert_eq!(
        out.decisions[3].active[0].hp, 169,
        "dec3: the forced Struggle (typeless) HIT Gengar 184→169 — a model that let Blissey \
         use a real move (or skipped the Struggle) diverges here"
    );
    assert_eq!(
        out.decisions[3].active[1].hp, 711,
        "dec3: Blissey took the gen-3 Struggle recoil floor(15/4) = 3 (714→711)"
    );
    assert!(
        !out.decisions[3].active[1].taunted && out.decisions[3].active[1].disabled_slot == -1,
        "both volatiles expired at dec3's residual (probe: -end Taunt [silent] + -end Disable)"
    );
    assert_eq!(
        seed_str(&out.decisions[3].seed_after),
        "39680,19331,35115,57684",
        "dec3 (Shadow Ball immune acc-only + Struggle acc/crit/dmg + Quick Claw) seed"
    );
}

/// TD4: the taunt/disable `onBeforeMove` PRIORITY ordering vs the paralysis roll — the
/// draw-ORDER pin the OBM probe settled (`probe_taunt_disable_onbeforemove_rng.js`):
///   (a) TAUNT sorts at priority 0 (gen3 DELETES gen4's `onBeforeMovePriority: 5`), AFTER
///       paralysis (1) — a taunted+PARALYZED mon with a queued status move DRAWS the para
///       `randomChance(1,4)` FIRST, then (not full-para'd) the taunt cant.
///   (b) DISABLE keeps its base `onBeforeMovePriority: 7`, BEFORE confusion (3) + paralysis
///       (1) — a paralyzed+DISABLED mon attempting the disabled slot is cant'd with NO para
///       roll at all.
/// WRONG (a swapped ordering): moving the taunt cant BEFORE the para check drops a draw in
/// (a); moving the disable cant AFTER it adds one in (b) — either desyncs the pinned seeds.
/// (The 720-run taunt_disable golden does NOT cover a paralyzed+taunted/disabled queued
/// move, so this pin is the only gate on the ordering — verified by perturbation.)
#[test]
fn taunt_and_disable_onbeforemove_priority_vs_paralysis() {
    let d = dex();

    // --- (a) TAUNT: para roll BEFORE the taunt cant. Probe seed "36705,59386,1163,10581":
    //     dec0 Thunder Wave paralyzes Blissey in-engine; dec1 Aerodactyl Taunts while the
    //     paralyzed Blissey QUEUED Thunder Wave → the sim drew taunt-acc + PARA + Quick Claw.
    let aero = "Aerodactyl|||NoAbility|thunderwave,taunt,earthquake|Serious|,252,,,,252|N||||";
    let blissey = "Blissey|||NoAbility|thunderwave,icebeam|Serious|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(aero, blissey, "36705,59386,1163,10581"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // thunderwave (par) / icebeam
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // TAUNT / queued twave: PARA ROLL then cant
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 2, "(a) both boundaries ran");
    assert_eq!(
        out.decisions[0].active[1].status,
        Some(Status::Paralysis),
        "(a) dec0: Blissey paralyzed in-engine"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "29999,2312,35328,57042",
        "(a) dec0 (Thunder Wave + Ice Beam) seed"
    );
    assert!(out.decisions[1].active[1].taunted, "(a) dec1: Blissey taunted");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "13442,34794,55622,32711",
        "(a) dec1: the taunted+paralyzed Blissey's queued Thunder Wave draws the PARA \
         randomChance(1,4) BEFORE the taunt cant (taunt priority 0 < par 1). A model that \
         cants BEFORE the para check drops that draw and desyncs here"
    );

    // --- (b) DISABLE: cant with NO para roll. Probe seed "11800,57299,34289,34330":
    //     dec0 Thunder Wave paralyzes Snorlax; dec1 both attack (lastMove = EQ, para roll
    //     passes both turns); dec2 Suicune Disables (lands) while the paralyzed Snorlax
    //     QUEUED EQ → the sim drew disable-acc + random(2,6) + Quick Claw — NO para roll.
    let suicune = "Suicune|||NoAbility|thunderwave,disable,surf|Serious|252,,252,,,|N||||";
    let snorlax = "Snorlax|||Immunity|earthquake,bodyslam|Serious|252,252,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(suicune, snorlax, "11800,57299,34289,34330"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // thunderwave (par) / EQ
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // surf / EQ (lastMove = EQ)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // DISABLE / queued EQ: cant, NO para roll
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 3, "(b) all three boundaries ran");
    assert_eq!(
        out.decisions[0].active[1].status,
        Some(Status::Paralysis),
        "(b) dec0: Snorlax paralyzed in-engine"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "30047,43372,41932,24696",
        "(b) dec1 (surf + EQ, para roll passes) seed"
    );
    assert_eq!(
        out.decisions[2].active[1].disabled_slot, 0,
        "(b) dec2: EQ (slot 0) disabled"
    );
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "42590,52732,34847,33941",
        "(b) dec2: the paralyzed+disabled Snorlax's queued EQ is cant'd with NO para roll \
         (disable priority 7 > par 1) — the turn draws disable acc + random(2,6) + Quick \
         Claw only. A model that ran the para check first adds a draw and desyncs here"
    );
    assert_eq!(
        out.decisions[2].active[0].hp, 256,
        "(b) Suicune HP unchanged on the cant turn (the blocked EQ dealt nothing; probe: 256/404)"
    );
}

/// TD5: Disable into a target whose LAST-USED move has 0 PP remaining — the
/// gen4-inherited onStart 0-PP guard (`!moveSlot.pp → return false`, which gen3's
/// condition INHERITS) REJECTS the volatile AFTER the draws: the accuracy roll AND the
/// `random(2,6)` durationCallback are both consumed, then `-fail` — NO `-start`, NO
/// volatile, NO residual duration handler. Ground truth (copied verbatim from
/// `harness/probe_disable_zero_pp_rng.js`, seed "1,8,14,22"):
///
/// Suicune [disable, calmmind] (faster) vs mono-move Blissey [detect] (slower).
/// dec0..dec7: Calm Mind / Detect — Blissey's Detect drains 8 → 0 PP (successes,
/// stall-roll failures, and the stall/protect residual tie-shuffles all play out
/// in-engine); dec8: Blissey (0-PP lastMove = detect, FORCED to Struggle) is Disabled —
/// the sim drew disable-acc (`random(100)=11`, HIT vs 55) + `random(2,6)=5`, then the
/// 0-PP guard rejected the volatile (`|move|p1a: Suicune|Disable||[still]` +
/// `|-fail|p1a: Suicune`, target volatiles EMPTY) and Blissey's Struggle ran
/// (Suicune 404→396, recoil 714→712); dec9: Calm Mind (+6-capped no-op) / Struggle again.
/// (The probe seeds the sim with `[1,8,14,22]`; the sim's init consumes ONE draw, so the
/// pin seeds the draw-free `start_with_switchins` with the POST-init seed — the same
/// convention as every TD pin.)
///
/// WRONG (pre-fix): the port consumed the same draws but RECORDED the volatile (+
/// emitted `-start`) — `disabled_slot` reads 0 at dec8/dec9 instead of -1, the phantom
/// registers a residual duration handler, and a later re-Disable would wrongly fail as
/// "already disabled". BOTH assert families carry teeth (reviewer-verified by
/// neutralizing the guard): the `disabled_slot`/`disable` STATE asserts trip on the
/// phantom directly, AND the SEED asserts trip at dec8 — the phantom's duration handler
/// TIES Blissey's live `stall` duration handler (same mon, NO_ORDER/subOrder 2) → one
/// extra residual tie-shuffle draw desyncs the boundary seed. Only the draw-count inside
/// the disable ARM itself (accuracy + `random(2,6)`) is bug-invariant.
#[test]
fn disable_into_a_zero_pp_lastmove_fails_draws_but_no_volatile() {
    let d = dex();
    let suicune = "Suicune|||NoAbility|disable,calmmind|Serious|252,,252,,,|N||||";
    let blissey = "Blissey|||NoAbility|detect|Serious|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(suicune, blissey, "36553,11848,52306,28017"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let mut script = vec![ScriptDecision::both(Choice::Move(1), Choice::Move(0)); 8];
    script.push(ScriptDecision::both(Choice::Move(0), Choice::Move(0))); // dec8: DISABLE / forced Struggle
    script.push(ScriptDecision::both(Choice::Move(1), Choice::Move(0))); // dec9: calmmind / Struggle
    let out = st.run_full_battle(&script, &d);
    assert_eq!(out.decisions.len(), 10, "all ten boundaries ran");

    // THE pin: no phantom volatile — Blissey is NOT disabled after the rejected Disable.
    let dis: Vec<i8> = out.decisions.iter().map(|r| r.active[1].disabled_slot).collect();
    assert_eq!(
        dis,
        vec![-1; 10],
        "the 0-PP-guard rejection leaves the target UN-disabled at every boundary — \
         under the pre-fix bug dec8/dec9 read 0 (the phantom volatile on the exhausted \
         detect slot)"
    );
    let bliss = &st.sides[1].pokemon[st.sides[1].active];
    assert!(
        bliss.disable.is_none(),
        "the disable volatile was REJECTED by the onStart 0-PP guard (probe: target \
         volatiles EMPTY, `-fail` not `-start`)"
    );
    assert_eq!(bliss.move_pp[0], 0, "Detect's slot is at 0 PP (the guard's trigger)");

    // The DRAW pin: per-decision post-seeds == the real sim's (dec8 consumed the
    // accuracy roll AND the random(2,6) BEFORE the rejection — skipping either desyncs
    // here; AND the phantom volatile itself desyncs dec8's seed via the extra residual
    // duration-handler tie-shuffle, so these carry teeth against the bug too).
    let seeds: Vec<String> = out.decisions.iter().map(|r| seed_str(&r.seed_after)).collect();
    assert_eq!(
        seeds,
        vec![
            "42494,47024,20202,4731".to_string(),  // dec0: calmmind / detect (succeeds)
            "23502,34830,46395,364".to_string(),   // dec1: stall roll passes
            "23430,11552,27475,15678".to_string(), // dec2: stall roll FAILS ([still] + -fail)
            "40629,35090,44804,17472".to_string(), // dec3
            "259,40325,10964,20029".to_string(),   // dec4
            "57552,49457,56409,7526".to_string(),  // dec5
            "63954,17720,28001,28200".to_string(), // dec6: stall roll fails
            "40401,16909,26293,9370".to_string(),  // dec7: detect PP hits 0 → struggle-only request
            "63009,8608,30925,25360".to_string(),  // dec8: disable acc + random(2,6) + Struggle acc/crit/dmg + QC
            "59958,21140,2276,29604".to_string(),  // dec9: Struggle acc/crit/dmg + QC
        ],
        "per-decision seeds == the real sim's (the rejected Disable still consumed \
         accuracy + random(2,6))"
    );

    // The Struggle proof (the 0-PP clock was real): dec8/dec9 HP from the probe.
    assert_eq!(out.decisions[8].active[0].hp, 396, "dec8: Blissey's forced Struggle hit 404→396");
    assert_eq!(
        out.decisions[8].active[1].hp, 712,
        "dec8: Blissey took the Struggle recoil 714→712 (floor(8/4) = 2)"
    );
    assert_eq!(out.decisions[9].active[0].hp, 388, "dec9: the second Struggle, 396→388");
    assert_eq!(out.decisions[9].active[1].hp, 710, "dec9: recoil 712→710");
}

// ============================================================================
// T1 — TRAPPING (`gen3_trapping_v1`): Arena Trap REJECTS a grounded foe's
//      voluntary switch DRAW-FREE (the switch mirror of the move reject-and-
//      re-request gate). The WRONG behaviour: the port runs the scripted
//      `Switch` decision (the mon leaves / a turn runs / draws fire) instead of
//      SKIPPING it with the boundary open, like the sim's `chooseSwitch` reject
//      ("Can't switch: The active Pokémon is trapped" — seed byte-identical).
//      Ground truth: harness/probe_trapping_regression_rng.js (PIN T1).
// ============================================================================

#[test]
fn arena_trap_rejects_a_grounded_foes_switch_draw_free() {
    let d = dex();
    let dugtrio = "Dugtrio|||ArenaTrap|earthquake,splash|Adamant|,252,,,,252|N||||";
    let team2 = "Snorlax|||NoAbility|bodyslam,splash|Brave|252,,,,,|N|,,,,,0|||]Regice|||NoAbility|icebeam,splash|Modest|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(dugtrio, team2, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");

    // The live trapped computation: Snorlax (grounded) is trapped by the foe's Arena
    // Trap; the TRAPPER itself is free.
    assert!(st.is_trapped(1, &d), "Snorlax (grounded) is TRAPPED by Arena Trap");
    assert!(!st.is_trapped(0, &d), "the trapping Dugtrio is NOT itself trapped");

    let out = st.run_full_battle(
        &[
            // The trapped Snorlax tries to flee — the sim REJECTS it draw-free and the
            // boundary stays OPEN (the port must SKIP this decision entirely).
            ScriptDecision::one(1, Choice::Switch(1)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // splash/splash commits
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // splash/splash again
        ],
        &d,
    );
    assert_eq!(
        out.decisions.len(),
        2,
        "the rejected trapped-switch decision is SKIPPED (no boundary recorded) — running \
         it as a turn (3 decisions) or ending the run (1) means the reject gate is gone"
    );
    // The rejected attempt drew NOTHING and left Snorlax in: the first RECORDED boundary
    // is the splash/splash turn, at the sim's exact seed.
    assert_eq!(
        out.decisions[0].active_species[1], "snorlax",
        "Snorlax never left (the switch was rejected)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "18464,3966,47670,60926",
        "dec0 (the splash/splash turn AFTER the draw-free reject): a port that ran or \
         drew on the rejected switch desyncs here — and Arena Trap's endTurn trap events \
         add ZERO draws (onFoe -> 1 handler per event, no tie possible)"
    );
    assert_eq!(out.decisions[0].trapped, [false, true], "dec0 boundary: p2 still trapped");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "57388,452,34593,29177",
        "dec1: the draw-free trapped state persists (seed rhythm unchanged)"
    );
    assert_eq!(out.decisions[1].trapped, [false, true], "dec1 boundary: p2 still trapped");
}

// ============================================================================
// T2 — TRAPPING: Arena Trap does NOT trap a Flying-type (Zapdos) or a Levitate
//      holder (Gengar) — gen-3 grounded == not-Flying && not-Levitate, so their
//      voluntary switches are ACCEPTED. The WRONG behaviour: the port rejects
//      (skips) their scripted switches -> decision count / species / seed all
//      desync. Ground truth: probe_trapping_regression_rng.js (PIN T2).
// ============================================================================

#[test]
fn arena_trap_does_not_trap_flying_or_levitate() {
    let d = dex();
    let dugtrio = "Dugtrio|||ArenaTrap|rockslide,splash|Adamant|,252,,,,252|N||||";
    let team2 = "Zapdos|||Pressure|drillpeck,splash|Modest|252,,,,,|N||||]Gengar|||Levitate|sludgebomb,splash|Modest|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(dugtrio, team2, "61872,34750,8741,59883"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    assert!(!st.is_trapped(1, &d), "Zapdos (FLYING) is NOT grounded -> NOT trapped");

    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // Zapdos -> Gengar (free)
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // Gengar -> Zapdos (free)
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)),
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 3, "ALL three decisions ran (no spurious reject)");
    assert_eq!(
        out.decisions[0].active_species[1], "gengar",
        "the Flying Zapdos switched out FREELY (its switch must be ACCEPTED)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "38565,7865,19639,43898",
        "dec0 seed (the accepted switch turn)"
    );
    assert_eq!(
        out.decisions[1].active_species[1], "zapdos",
        "the Levitate Gengar switched out FREELY too"
    );
    assert_eq!(seed_str(&out.decisions[1].seed_after), "4742,50088,47031,36837", "dec1 seed");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "49988,3828,17110,61724", "dec2 seed");
    for dec in &out.decisions {
        assert_eq!(dec.trapped, [false, false], "nobody is ever trapped in this pin");
    }
}

// ============================================================================
// T3 — TRAPPING: Magnet Pull traps STEEL only — the MAGNETON MIRROR is a MUTUAL
//      trap (Steel <-> Steel) whose speed-tied endTurn TrapPokemon +
//      MaybeTrapPokemon runEvents each carry TWO handlers (gen3 magnetpull is
//      `onAny*` — data/mods/gen3/abilities.ts — so BOTH actives' abilities
//      register on BOTH mons' events) -> ONE Fisher-Yates tie-shuffle draw per
//      event per mon = **4 draws per endTurn** (probe: 11/turn vs the Sturdy
//      control's 7). The WRONG behaviours: (a) missing trap-event shuffles ->
//      the splash/splash boundary seeds desync; (b) trapping a NON-Steel foe ->
//      the control's accepted switch is spuriously rejected. Ground truth:
//      probe_trapping_regression_rng.js (PINs T3a + T3b).
// ============================================================================

#[test]
fn magnet_pull_traps_steel_only() {
    let d = dex();
    // (a) The MIRROR: mutual trap + the 4-per-endTurn tie-shuffle draws.
    let t1 = "Magneton|||MagnetPull|thunderbolt,splash|Modest|,,,252,,|N||||]Snorlax|||NoAbility|bodyslam,splash|Adamant|252,,,,,|N||||";
    let t2 = "Magneton|||MagnetPull|thunderbolt,splash|Modest|,,,252,,|N||||]Regice|||NoAbility|icebeam,splash|Modest|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(t1, t2, "26557,11031,5008,39401"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    assert!(st.is_trapped(0, &d), "mirror: p1's Magneton (Steel) is trapped by the foe's Magnet Pull");
    assert!(st.is_trapped(1, &d), "mirror: p2's Magneton is trapped right back (MUTUAL)");

    let out = st.run_full_battle(
        &[
            ScriptDecision::one(1, Choice::Switch(1)), // trapped switch -> REJECTED draw-free
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // splash/splash
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // splash/splash
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 2, "the trapped switch was SKIPPED draw-free");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "7430,1246,52422,27018",
        "dec0 (splash/splash): the endTurn draws its 4 trap-event tie-shuffles (2 events x \
         2 mons at equal cached speed) — DROPPING the trap_event_shuffles (or drawing a \
         wrong count/order) desyncs THIS seed"
    );
    assert_eq!(out.decisions[0].trapped, [true, true], "dec0: MUTUALLY trapped");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "62863,24855,36863,41895",
        "dec1: the 4-draw endTurn rhythm continues"
    );
    assert_eq!(out.decisions[1].trapped, [true, true], "dec1: still mutually trapped");

    // (b) The non-Steel CONTROL: Snorlax walks out of a Magnet Pull freely.
    let t1b = "Magneton|||MagnetPull|thunderbolt,splash|Modest|,,,252,,|N||||";
    let t2b = "Snorlax|||NoAbility|bodyslam,splash|Brave|252,,,,,|N|,,,,,0|||]Regice|||NoAbility|icebeam,splash|Modest|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(t1b, t2b, "61965,2284,39104,59883"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    assert!(!st.is_trapped(1, &d), "Snorlax is NOT Steel -> NOT trapped by Magnet Pull");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // ACCEPTED (free)
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)),
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 2, "the free switch RAN (no spurious trap reject)");
    assert_eq!(out.decisions[0].active_species[1], "regice", "Snorlax left freely");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "21179,38340,2782,43898", "dec0 seed");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "18687,8633,39962,36837",
        "dec1 seed: ONE-SIDED Magnet Pull vs a non-trap foe = 1 handler per event -> the \
         endTurn trap events add ZERO draws here"
    );
    assert_eq!(out.decisions[1].trapped, [false, false], "Regice (Ice) is not Steel either");
}

// ============================================================================
// T4 — TRAPPING vs PHAZING: Roar still DRAGS a trapped mon out (trapping blocks
//      only the VOLUNTARY switch; `drag_in` / forceSwitch never consult
//      `trapped`). The WRONG behaviour: the port lets the trap block the phaze
//      (no drag -> no `sample` draw -> species + seed desync). The dragged-in
//      Regice (grounded) is trapped in turn. Ground truth:
//      probe_trapping_regression_rng.js (PIN T4).
// ============================================================================

#[test]
fn roar_drags_a_trapped_mon_out() {
    let d = dex();
    let dugtrio = "Dugtrio|||ArenaTrap|roar,splash|Adamant|,252,,,,252|N||||";
    let team2 = "Snorlax|||NoAbility|bodyslam,splash|Brave|252,,,,,|N|,,,,,0|||]Regice|||NoAbility|icebeam,splash|Modest|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(dugtrio, team2, "58116,36429,52618,13587"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    assert!(st.is_trapped(1, &d), "Snorlax is trapped (cannot leave VOLUNTARILY)");

    let out = st.run_full_battle(
        &[
            ScriptDecision::one(1, Choice::Switch(1)), // the voluntary flee -> REJECTED
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Roar / splash
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // splash/splash
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 2, "the rejected switch was skipped; Roar + splash ran");
    assert!(
        out.decisions[0].phaze_drag,
        "the Roar DRAG FIRED through the trap (phaze bypasses trapping)"
    );
    assert_eq!(
        out.decisions[0].active_species[1], "regice",
        "the trapped Snorlax was DRAGGED out anyway (Regice in)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "1353,42523,42638,16548",
        "dec0 (Roar): accuracy(100) + the n=1 `sample` drew as normal — a trap that \
         blocked the phaze would skip the sample and desync here"
    );
    assert_eq!(
        out.decisions[0].trapped,
        [false, true],
        "the dragged-in Regice (grounded) is trapped in turn"
    );
    assert_eq!(seed_str(&out.decisions[1].seed_after), "29841,29415,52057,58231", "dec1 seed");
}

// ============================================================================
// T5 — TRAPPING (the probe-settled Showdown-gen3 SURPRISE): a grounded GHOST
//      (Sableye) **IS** trapped by Arena Trap — the gen3 dex resolves NO
//      `trapped` type-immunity (Ghost damageTaken.trapped = undefined; the
//      cartridge gen6+ Ghost escape does not exist in Showdown-gen3, and the
//      port models the SIM). The WRONG behaviour: hardcoding the modern Ghost
//      escape -> Sableye's scripted switch is spuriously ACCEPTED. Ground
//      truth: probe_trapping_regression_rng.js (PIN T5).
// ============================================================================

#[test]
fn grounded_ghost_is_trapped_by_arena_trap_in_showdown_gen3() {
    let d = dex();
    let dugtrio = "Dugtrio|||ArenaTrap|earthquake,splash|Adamant|,252,,,,252|N||||";
    let team2 = "Sableye|||KeenEye|shadowball,splash|Bold|252,,,,,|N||||]Snorlax|||NoAbility|bodyslam,splash|Brave|252,,,,,|N|,,,,,0|||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(dugtrio, team2, "54360,38109,30958,32827"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    assert!(
        st.is_trapped(1, &d),
        "Sableye (Ghost/Dark, grounded) IS trapped in Showdown-gen3 — a modern-gen Ghost \
         trap-escape here is WRONG for this sim"
    );

    let out = st.run_full_battle(
        &[
            ScriptDecision::one(1, Choice::Switch(1)), // Sableye tries to flee -> REJECTED
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // splash/splash
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 1, "the Ghost's switch was rejected draw-free (skipped)");
    assert_eq!(out.decisions[0].active_species[1], "sableye", "Sableye never left");
    assert_eq!(out.decisions[0].trapped, [false, true], "Sableye trapped at the boundary");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "26800,52733,48763,51466",
        "dec0 seed (the reject drew nothing; the splash turn's draws only)"
    );
}

// ============================================================================
// I1 — gen-3 INTIMIDATE vs SUBSTITUTE: a MID-BATTLE Intimidate switch-in does
//      NOT drop the Atk of a foe behind a SUBSTITUTE (the gen3 mod's per-foe
//      substitute skip; probe: sub up -> NO |-unboost|, and the block is
//      SEED-NEUTRAL — identical draws/seeds with or without the sub). The WRONG
//      (pre-fix) behaviour: `event::intimidate_on_start` ignored the target's
//      sub and dropped Atk -1 (surfaced by `gen3_trapping_v1`'s e2e regen on
//      real teams — e2e_171/e2e_204: a Jynx that Substituted the turn before a
//      Salamence switch-in). Ground truth:
//      harness/probe_intimidate_substitute_rng.js.
// ============================================================================

#[test]
fn intimidate_into_a_substitute_is_a_noop() {
    let d = dex();
    let lax = "Snorlax|||NoAbility|substitute,splash|Serious|252,,,,,|N||||";
    let team2 = "Regice|||NoAbility|icebeam,splash|Serious|252,,,,,|N||||]Salamence|||Intimidate|dragonclaw,splash|Serious||N||||";

    // (a) SUB UP: Snorlax Substitutes on dec0; the Intimidate Salamence switches in on
    //     dec1 — Snorlax's Atk stays 0 (and the sub is intact).
    let mut battle =
        Battle::start_with_switchins(&opts_cg(lax, team2, "53303,35262,36397,29520"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Substitute / splash
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // splash / Salamence IN
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // splash/splash
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 3, "all three decisions ran");
    assert!(out.decisions[0].active[0].substitute.is_some(), "dec0: the sub is up");
    assert_eq!(
        out.decisions[1].active[0].boosts[0], 0,
        "dec1 (the Intimidate switch-in): Snorlax's Atk stays 0 BEHIND THE SUB — the gen-3 \
         Intimidate substitute skip (reverting the sub gate in intimidate_on_start trips this)"
    );
    assert!(
        out.decisions[1].active[0].substitute.is_some(),
        "dec1: the sub itself is untouched by the switch-in"
    );
    // The block is SEED-NEUTRAL (probed identical seeds both arms) — pin the boundary seed
    // so a 'fix' that turned the block into a draw would also trip.
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "41762,18770,8812,43906",
        "dec1 seed: the sub block draws NOTHING (identical to the no-sub control's seed)"
    );

    // (b) CONTROL (no sub): the same switch-in DOES drop Atk to -1.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(lax, team2, "53303,35262,36397,29520"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // splash / splash (NO sub)
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // splash / Salamence IN
        ],
        &d,
    );
    assert_eq!(
        out.decisions[1].active[0].boosts[0], -1,
        "control: without a sub the Intimidate switch-in drops Snorlax to -1 (the gate must \
         not over-block)"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "41762,18770,8812,43906",
        "control dec1 seed: same seed as the sub arm — the drop is draw-free either way"
    );
}

// ============================================================================
// IM — the gen3_item_mechanics_v1 ITEM-MODIFIER class pins (the DATA-DRIVEN
//      framework's motivating drift bug + the probe-settled fold gotchas). The
//      A/B fuzzer caught Pink Bow / Polkadot Bow / the 4 gen4-named incenses
//      sitting "modeled" in the e2e's MODELED_ITEMS while the port's hardcoded
//      `resolve_atk_stat_mods` match-arm priced NONE of them (repro:
//      harness/ab_fuzz_out/smoke_random/divergences/ — a Polkadot-Bow Body Slam
//      dealt ×1.1 in the sim, flat in the port). The fix replaced the match-arm
//      with a dex-data lookup (`ItemData.type_boost` / `stat_mods`), and these
//      pins hold the PROBE-SETTLED math (exact per-hit HP from the real sim via
//      the item_mods golden's constructed scenarios) so a revert — of the data
//      fields OR the resolvers — trips deterministically:
//        IM1 the bows' DIRECT ×1.1 base-power float → `polkadot_and_pink_bow_price_the_direct_bp_boost`
//        IM2 the incenses are ×4915/4096 (~1.2), NOT ×1.1 → `gen4_named_incense_is_x1_2_bp_chain_not_x1_1`
//        IM3 gen3 Light Ball is SpA-ONLY ×2 (the mod-chain law: the gen4 Atk half
//            must NOT exist)                    → `gen3_lightball_is_spa_only_x2`
//        IM4 the DEFENDER-side stat items fold into the ModifyDef/ModifySpD chain
//            (DeepSeaScale SpD ×2, Metal Powder Def ×2) → `def_side_stat_items_fold_into_the_stat_chain`
//        IM5 a WRONG-SPECIES holder gets NO boost (the species gate)
//                                                → `wrong_species_holder_gets_no_boost`
//        IM6 Soul Dew boosts BOTH SpA (outgoing) and SpD (incoming) ×1.5
//                                                → `souldew_boosts_both_spa_and_spd`
//      All values are the REAL sim's (gen3customgame, the golden's seeds[0]);
//      every fold is DRAW-FREE, so each pin also asserts a post-decision seed.
// ============================================================================

/// IM1: Pink Bow (and by the same handler Polkadot Bow) prices Normal moves at the
/// DIRECT ×1.1 base-power float (`return basePower * 1.1` — replaces the event
/// relayVar; clampIntRange floors). WRONG (pre-fix): the bow resolved to NO
/// modifier and Body Slam hit flat (Suicune took less damage than the sim).
#[test]
fn polkadot_and_pink_bow_price_the_direct_bp_boost() {
    let d = dex();
    let bow = "Snorlax||pinkbow|NoAbility|bodyslam,earthquake|Serious|,252,,252,,|N||||";
    let cune = "Suicune|||NoAbility|icebeam|Serious||N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(bow, cune, "7286,53657,17285,64642"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Sim truth (item_mods golden, seeds[0]): the bow-boosted Body Slam leaves
    // Suicune at 218/341. An unpriced bow (flat BP 85) leaves MORE HP.
    assert_eq!(out.decisions[0].active[1].hp, 218, "bow ×1.1 Body Slam: Suicune at 218/341");
    assert_eq!(out.decisions[0].active[0].hp, 343, "Ice Beam back: Snorlax at 343/461");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "38288,32135,64422,53589",
        "the bow fold is DRAW-FREE (same draw stream as the sim)"
    );
}

/// IM2: the 4 gen4-named incenses (odd/rock/rose/wave — items the sim still
/// applies under gen3 formats) are `chainModify([4915,4096])` ≈ ×1.2 at the
/// BASE-POWER chain — NOT the ×1.1 the old MODELED_ITEMS comment assumed. WRONG
/// (the perturbation this pin holds against): mod=[11,10] leaves Snorlax at 281
/// instead of the sim's 264.
#[test]
fn gen4_named_incense_is_x1_2_bp_chain_not_x1_1() {
    let d = dex();
    let mence = "Salamence||oddincense|NoAbility|psychic,strength|Serious|,252,,252,,|N||||";
    let lax = "Snorlax|||NoAbility|bodyslam|Serious||N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(mence, lax, "7286,53657,17285,64642"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        out.decisions[0].active[1].hp, 264,
        "Odd Incense Psychic is ×4915/4096 at the BP chain: Snorlax at 264/461 (a ×1.1 \
         mispricing gives 281; an unpriced incense gives more still)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "38288,32135,64422,53589",
        "the incense chain fold is DRAW-FREE"
    );
}

/// IM3: gen3 Light Ball is SpA-ONLY ×2 for Pikachu — THE MOD-CHAIN LAW pin. Base
/// data doubles Atk+SpA, the gen4 mod rewrites it to an onBasePower double, and
/// the gen3 mod rewrites it AGAIN to onModifySpA only. WRONG (the gen4-leak
/// perturbation): statMods.atk=[2,1] makes the PHYSICAL control hit leave Snorlax
/// at 20 instead of the sim's 77.
#[test]
fn gen3_lightball_is_spa_only_x2() {
    let d = dex();
    let pika = "Pikachu||lightball|NoAbility|thunderbolt,strength|Serious|252,,,252,,|N||||";
    let lax = "Snorlax|||NoAbility|bodyslam|Serious||N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(pika, lax, "7286,53657,17285,64642"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Thunderbolt (SpA ×2 fires)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Strength (NO Atk boost!)
        ],
        &d,
    );
    assert_eq!(
        out.decisions[0].active[1].hp, 136,
        "Light Ball Thunderbolt (SpA ×2): Snorlax at 136/461"
    );
    assert_eq!(
        out.decisions[1].active[1].hp, 77,
        "the PHYSICAL control (Strength) is NOT boosted — gen3 Light Ball has NO Atk half \
         (Snorlax at 77; a leaked gen4 Atk ×2 gives 20)"
    );
}

/// IM4: the DEFENDER-side stat items fold into the ModifyDef/ModifySpD stat chain
/// (after the boost table, before the Explosion def-halve): DeepSeaScale = SpD ×2
/// for Clamperl (it survives Alakazam's Psychic at EXACTLY 1 HP — an unpriced
/// scale is an OHKO), Metal Powder = Def ×2 for untransformed Ditto.
#[test]
fn def_side_stat_items_fold_into_the_stat_chain() {
    let d = dex();
    // (a) DeepSeaScale (SpD ×2, Clamperl).
    let zam = "Alakazam|||NoAbility|psychic|Modest|,,,252,,|N||||";
    let pearl = "Clamperl||deepseascale|NoAbility|surf|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(zam, pearl, "7286,53657,17285,64642"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        out.decisions[0].active[1].hp, 1,
        "DeepSeaScale SpD ×2: Clamperl survives the Psychic at exactly 1/274 (unpriced = OHKO)"
    );
    assert!(!out.decisions[0].active[1].fainted, "the scale save is a LIVE mon");

    // (b) Metal Powder (Def ×2, untransformed Ditto).
    let lax = "Snorlax|||NoAbility|bodyslam|Adamant|,252,,,,|N||||";
    let ditto = "Ditto||metalpowder|NoAbility|strength|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(lax, ditto, "7286,53657,17285,64642"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        out.decisions[0].active[1].hp, 15,
        "Metal Powder Def ×2: Ditto at 15/300 after the Body Slam (unpriced = KO'd)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "49617,7054,51918,34858",
        "the def-side fold is DRAW-FREE"
    );
}

/// IM5: the SPECIES GATE — a wrong-species holder gets NO boost. Thick Club on a
/// Snorlax (not Cubone/Marowak) leaves its Earthquake un-doubled. WRONG (the
/// gate-removal perturbation): Suicune takes the ×2 hit and the HP diverges.
#[test]
fn wrong_species_holder_gets_no_boost() {
    let d = dex();
    let lax = "Snorlax||thickclub|NoAbility|earthquake,bodyslam|Adamant|,252,,,,|N||||";
    let cune = "Suicune|||NoAbility|surf|Serious||N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(lax, cune, "7286,53657,17285,64642"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        out.decisions[0].active[1].hp, 120,
        "Thick Club on a NON-Marowak does nothing: Suicune at 120/341 (an ungated ×2 \
         roughly doubles the hit)"
    );
}

/// IM6: Soul Dew boosts BOTH directions for Lati@s — SpA ×1.5 on the outgoing
/// special hit AND SpD ×1.5 on the incoming one — in the same battle.
#[test]
fn souldew_boosts_both_spa_and_spd() {
    let d = dex();
    let latios = "Latios||souldew|NoAbility|psychic,thunderbolt|Modest|,,,252,,|N||||";
    let cune = "Suicune|||NoAbility|icebeam|Serious||N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(latios, cune, "16845,55639,3726,17102"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Sim truth (item_mods golden, the seeds[1] run): Latios' ×1.5-SpA Psychic
    // leaves Suicune at 109/341; Suicune's SE Ice Beam into the ×1.5-SpD Latios
    // leaves it at 222/301.
    assert_eq!(out.decisions[0].active[1].hp, 109, "Soul Dew SpA ×1.5 out: Suicune at 109/341");
    assert_eq!(out.decisions[0].active[0].hp, 222, "Soul Dew SpD ×1.5 in: Latios at 222/301");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "8237,29562,653,58833",
        "both folds are DRAW-FREE"
    );
}

// ============================================================================
// AB — the gen3_item_mechanics_v1 ABILITY DMG_MOD class pins (Phase 2). The
//      generic ability damage-modifier fold (`AbilityData.dmg_mod` →
//      `resolve_atk_stat_mods`/`resolve_def_stat_mods`/`resolve_bp_mods`), with
//      the probe-settled math + the interaction gotchas the resolved gen3 dist
//      pins (`harness/dump_gen3_mechanics.js`; damage verified vs the sim via the
//      ability_dmgmod / damage goldens). A revert of the data field OR the resolver
//      trips deterministically:
//        AB1 Guts — Atk ×1.5 while statused AND the physical burn-halve SUPPRESSED
//            (the KEY interaction: a burned Guts mon hits at full ×1.5, not ×0.75)
//                                       → `guts_is_x1_5_and_suppresses_the_burn_halve`
//        AB2 the PINCH THRESHOLD BOUNDARY — Torrent/Blaze/Overgrow/Swarm fire at
//            `hp <= maxhp/3` (bit-exactly `3*hp <= maxhp`): hp==floor(maxhp/3) boosts,
//            one above does NOT → `pinch_bp_boost_fires_at_exactly_one_third_hp`
//        AB3 MARVEL SCALE is DEFENDER-side — Def ×1.5 while the defender is statused
//                                       → `marvel_scale_is_a_defender_side_def_boost`
//        AB4 Huge/Pure Power — Atk ×2 unconditional (physical); a special move is
//            UNboosted (ModifyAtk touches only Atk) → `huge_power_is_x2_atk_physical_only`
//        AB5 the STATUS GATE — an UNstatused Guts mon gets NO boost (the whenStatused
//            gate; dropping it would boost every hit) → `guts_unstatused_gets_no_boost`
//      Every fold is DRAW-FREE stat/BP math, so the battle-level pins also assert a
//      post-decision seed. Values are the REAL sim's (the ability_dmgmod golden's
//      first-seed runs); AB2 is a self-contained `calc_damage` boundary check.
// ============================================================================

/// AB1: Guts is Atk ×1.5 while statused AND it SUPPRESSES the physical burn-halve —
/// the interaction the port must get exactly right (the ×1.5 stat fold in
/// `resolve_atk_stat_mods` composes with `Combatant::has_guts` skipping the ×0.5 in
/// `modify_damage`). A burned Machamp's Body Slam into Weezing lands the FULL
/// ×1.5-Atk, non-burn-halved hit. WRONG: without the Guts ×1.5 (or with the burn
/// halve applied) Weezing takes far less and its HP diverges here.
#[test]
fn guts_is_x1_5_and_suppresses_the_burn_halve() {
    let d = dex();
    // The ability_dmgmod golden's guts_burned_machamp, first-seed run. Machamp is
    // pre-burned by Weezing's Will-O-Wisp on the way in; by decision 0 it is already
    // burned and its Body Slam is the ×1.5 (burn-suppressed) hit.
    let machamp = "Machamp|||Guts|bodyslam,rest|Adamant|252,252,,,,|N||||";
    let weezing = "Weezing|||NoAbility|willowisp,rest|Bold|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(machamp, weezing, "23034,29935,33163,37924"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // move 0 = Body Slam every turn; Weezing move 0 = Will-O-Wisp then it Rests.
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
        ],
        &d,
    );
    // Sim truth (golden): the burned Machamp is at 336/384 (burn chip), and its ×1.5,
    // burn-SUPPRESSED Body Slam leaves Weezing at 219/334. A non-suppressed ×0.75 (or a
    // Guts-less ×1) hit leaves Weezing much healthier.
    assert!(
        matches!(out.decisions[0].active[0].status, Some(Status::Burn)),
        "Machamp is burned entering decision 0"
    );
    assert_eq!(
        out.decisions[0].active[1].hp, 219,
        "burned Guts Body Slam is ×1.5 AND burn-suppressed: Weezing at 219/334 \
         (drop the Guts ×1.5 → higher; apply the burn halve → higher still)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "10951,62401,10929,7530",
        "the Guts fold is DRAW-FREE"
    );
}

/// AB2: the PINCH threshold is `hp <= maxhp/3`, which for integer hp/maxhp is
/// bit-exactly `3*hp <= maxhp` (probe-verified at the maxhp=341 float boundary). A
/// self-contained `calc_damage` check: a STAB Water Surf gets the BP ×1.5 chain at
/// hp == floor(maxhp/3) but NOT one point above. WRONG (an off-by-one, or `<` vs
/// `<=`, or `maxhp/3` rounding): the boundary shifts and one of these two bases moves.
#[test]
fn pinch_bp_boost_fires_at_exactly_one_third_hp() {
    use pokesim::damage::{calc_damage, BpMod, Combatant, DamageContext, MoveInput};
    use pokesim::dex::{MoveCategory, Type};
    let d = dex();
    // The pinch condition is resolved by the ENGINE (`3*hp <= maxhp`); here we pin the
    // downstream calc: WITH the fold (BP ×1.5) vs WITHOUT, on a fixed STAB Water Surf.
    let base = |pinch: bool| {
        let att = Combatant { level: 100, spa_stat: 300, types: vec![Type::Water], ..Default::default() };
        let def = Combatant { level: 100, spd_stat: 200, types: vec![Type::Normal], ..Default::default() };
        let ctx = DamageContext {
            attacker: att,
            defender: def,
            mv: MoveInput { base_power: 95, move_type: Some(Type::Water), category: MoveCategory::Special, halves_defense: false },
            crit: false, weather: None, reflect: false, light_screen: false,
            atk_stat_mods: vec![], atk_direct_modify: None, def_stat_mods: vec![],
            bp_mods: if pinch { vec![BpMod::Chain(3, 2)] } else { vec![] },
            defender_thick_fat: false, immune: false, flash_fire: false,
        };
        calc_damage(&ctx, &d).base
    };
    assert_eq!(base(false), 181, "un-pinched STAB Surf base");
    assert_eq!(base(true), 270, "pinched STAB Surf base = BP ×1.5 (×1.49 realized after re-floor)");
    // The integer boundary the ENGINE applies is `3*hp <= maxhp` — pin its exactness so
    // an off-by-one in the resolve site is caught (33/100 triggers, 34/100 does not; the
    // non-divisible 113/341 triggers, 114/341 does not).
    let in_pinch = |hp: u32, maxhp: u32| 3 * hp <= maxhp;
    assert!(in_pinch(33, 100) && !in_pinch(34, 100), "1/3 of 100: hp≤33 pinches");
    assert!(in_pinch(113, 341) && !in_pinch(114, 341), "1/3 of 341 (non-divisible): hp≤113 pinches");
}

/// AB3: Marvel Scale is a DEFENDER-side Def ×1.5 while the DEFENDER is statused
/// (`resolve_def_stat_mods` → the ModifyDef fold, a `def_stat_mods` member — the
/// mechanism the def-side items also use). Two parts: (a) a self-contained
/// `calc_damage` check that the Def ×1.5 exactly reduces a physical hit to ×2/3; (b) the
/// ability_dmgmod golden's marvel battle seed (draw-free). WRONG (drop the def fold): a
/// physical hit into a statused Marvel mon does full damage.
#[test]
fn marvel_scale_is_a_defender_side_def_boost() {
    use pokesim::damage::{calc_damage, AtkStatMod, Combatant, DamageContext, MoveInput};
    use pokesim::dex::{MoveCategory, Type};
    let d = dex();
    // (a) The calc: a fixed physical hit WITH the defender Def ×1.5 (Marvel Scale) vs
    // WITHOUT. Def ×1.5 → damage ×2/3.
    let base = |marvel: bool| {
        let att = Combatant { level: 100, atk_stat: 300, types: vec![Type::Normal], ..Default::default() };
        let def = Combatant { level: 100, def_stat: 250, types: vec![Type::Normal], ..Default::default() };
        let ctx = DamageContext {
            attacker: att,
            defender: def,
            mv: MoveInput { base_power: 100, move_type: None, category: MoveCategory::Physical, halves_defense: false },
            crit: false, weather: None, reflect: false, light_screen: false,
            atk_stat_mods: vec![],
            atk_direct_modify: None,
            def_stat_mods: if marvel { vec![AtkStatMod::Item { num: 3, den: 2 }] } else { vec![] },
            bp_mods: vec![], defender_thick_fat: false, immune: false, flash_fire: false,
        };
        calc_damage(&ctx, &d).base
    };
    assert_eq!(base(false), 102, "no-Marvel physical base");
    assert_eq!(base(true), 69, "Marvel Scale Def ×1.5 reduces the hit to ×2/3 (102→69)");

    // (b) The golden marvel battle's seed — the Def fold is DRAW-FREE. (Milotic is burned
    // by Will-O-Wisp on T1; the Def×1.5-reduced Shadow Punch is out-healed by Recover, a
    // stable fixed point — the exact-HP fold is pinned by (a) + the ability_dmgmod golden,
    // which replays this battle to game-end.)
    let gengar = "Gengar|||NoAbility|willowisp,shadowpunch|Adamant|252,252,,,,|N||||";
    let milotic = "Milotic|||MarvelScale|recover|Bold|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(gengar, milotic, "23034,29935,33163,37924"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    assert!(
        matches!(out.decisions[0].active[1].status, Some(Status::Burn)),
        "Milotic is burned by Will-O-Wisp on T1"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "32024,33671,60115,27931",
        "the Marvel Def fold is DRAW-FREE (the golden battle's seed)"
    );
}

/// AB4: Huge/Pure Power is Atk ×2, UNCONDITIONAL but PHYSICAL only (ModifyAtk touches
/// only the Atk stat). Two parts: (a) a self-contained `calc_damage` check that the ×2
/// AtkStatMod exactly doubles a physical hit while a SPECIAL hit is unchanged (ModifyAtk
/// never touches SpA); (b) the ability_dmgmod golden's huge_power_azumarill first-seed
/// battle seed, so the ×2 fold stays DRAW-FREE end to end. WRONG (drop the ×2): the
/// physical base halves; WRONG (apply it to special too): the special base doubles.
#[test]
fn huge_power_is_x2_atk_physical_only() {
    use pokesim::damage::{calc_damage, AtkStatMod, Combatant, DamageContext, MoveInput};
    use pokesim::dex::{MoveCategory, Type};
    let d = dex();
    // (a) The calc: a fixed physical hit WITH the ×2 Atk mod (Huge/Pure Power) vs
    // WITHOUT, and a special hit (ModifyAtk must NOT touch SpA).
    let base = |cat: MoveCategory, huge: bool| {
        let att = Combatant { level: 100, atk_stat: 300, spa_stat: 300, types: vec![Type::Normal], ..Default::default() };
        let def = Combatant { level: 100, def_stat: 250, spd_stat: 250, types: vec![Type::Normal], ..Default::default() };
        let ctx = DamageContext {
            attacker: att,
            defender: def,
            // A typeless move (no STAB) so the ×2 shows undiluted.
            mv: MoveInput { base_power: 100, move_type: None, category: cat, halves_defense: false },
            crit: false, weather: None, reflect: false, light_screen: false,
            atk_stat_mods: if huge { vec![AtkStatMod::Item { num: 2, den: 1 }] } else { vec![] },
            atk_direct_modify: None,
            def_stat_mods: vec![], bp_mods: vec![], defender_thick_fat: false, immune: false,
            flash_fire: false,
        };
        calc_damage(&ctx, &d).base
    };
    let phys_off = base(MoveCategory::Physical, false);
    let phys_on = base(MoveCategory::Physical, true);
    assert_eq!(phys_off, 102, "un-boosted physical base");
    assert_eq!(phys_on, 203, "Huge Power ×2 Atk exactly doubles the physical base (102→203)");
    // (`calc_damage` applies `atk_stat_mods` to whatever the move's offensive stat is; the
    // PHYSICAL-only gate lives in the engine's `resolve_atk_stat_mods` [`category ==
    // Physical`], validated by the damage golden's `huge_power_special_control` — a special
    // move gets NO ×2.)

    // (b) The golden battle's seed — the ×2 fold adds NO draws.
    let azu = "Azumarill|||HugePower|waterfall,rest|Adamant|252,252,,,,|N||||";
    let cune = "Suicune|||NoAbility|surf|Bold|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(azu, cune, "23034,29935,33163,37924"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "44009,38689,9898,62357",
        "the Huge Power fold is DRAW-FREE (the golden battle's seed)"
    );
}

/// AB5: the STATUS GATE — an UNstatused Guts mon gets NO Atk boost (the `whenStatused`
/// condition). An unstatused Machamp's Body Slam is un-boosted. WRONG (drop the gate,
/// making Guts fire unconditionally): the hit becomes ×1.5, Suicune takes more, and its
/// HP diverges. This is the mirror of AB1 — the gate must NOT fire without a status.
#[test]
fn guts_unstatused_gets_no_boost() {
    let d = dex();
    // The ability_dmgmod golden's guts_unstatused_control, first-seed run: the Guts
    // Machamp never gets statused (Suicune only Surfs), so its Body Slam is flat.
    let machamp = "Machamp|||Guts|bodyslam,rest|Adamant|252,252,,,,|N||||";
    let cune = "Suicune|||NoAbility|surf|Bold|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(machamp, cune, "23034,29935,33163,37924"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert!(
        out.decisions[0].active[0].status.is_none(),
        "Machamp is UN-statused (the Guts gate must not fire)"
    );
    // Sim truth (golden): the un-boosted Body Slam leaves Suicune at 327/404. Dropping
    // the whenStatused gate (Guts ×1.5 always) would leave it LOWER.
    assert_eq!(
        out.decisions[0].active[1].hp, 327,
        "unstatused Guts Body Slam is NOT boosted: Suicune at 327/404 (an ungated ×1.5 → lower)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "58157,35838,17158,30092",
        "no fold fired → the draw stream is the un-boosted one"
    );
}

// ============================================================================
// FZ1 — SUN → freeze immunity (`gen3_sun_freeze_immunity_v1`). The base `sunnyday`
//       weather registers `onImmunity(type) { if effectiveWeather()!=='sunnyday'
//       return; if type==='frz' return false; }` — so while the field weather is Sun
//       (Drought / Sunny Day) a mon CANNOT be frozen. `setStatus` → `runStatusImmunity
//       ('frz')` → `runEvent('Immunity','frz')` returns FALSE, BEFORE `runEvent
//       ('SetStatus')`. The immunity is DRAW-FREE, so the freeze secondary's random(100)
//       still fires (the seed matches) — only the freeze APPLICATION is suppressed.
//       WRONG (pre-fix): the port froze the mon → the A/B-fuzz "ice-freeze cluster"
//       (196 repros, expected=None got=Some(Freeze), seed matching). This constructs
//       the exact repro (p1 Regice Ice-Beams p2 Groudon(Drought, sun up) on a seed whose
//       freeze secondary WOULD land) and asserts BOTH the STATE (Groudon stays un-frozen)
//       AND the post-decision SEED (identical to the freeze-LANDS control — the gate is
//       draw-free). Ground truth from `harness/probe_sun_freeze_regression_rng.js`.
//       A revert of the sun-freeze gate freezes Groudon → the status assertion fails.
// ============================================================================

/// FZ1: under Sun (Drought) a mon can't be frozen — the freeze secondary lands (its
/// random(100) draws) but the freeze does NOT apply (`sunnyday.onImmunity('frz')`).
/// WRONG (pre-fix): the port froze it. STATE (un-frozen) + SEED (draw-free) pin.
#[test]
fn sun_blocks_freeze_secondary_draw_free() {
    let d = dex();
    // p1 Regice (special Ice attacker) Ice-Beams p2 Groudon (Drought → permanent sun at
    // switch-in). Groudon is Ground-type (Ice Beam NEUTRAL, no frz type-immunity) with big
    // HP so it survives the hit. p2 just Swords Dances (a draw-free self-boost) so the ONLY
    // status source is Ice Beam's freeze secondary — which, at this seed, WOULD land (proven
    // by the no-sun control in the probe). Seeded at the probe's SEED-BEFORE-the-turn (the
    // port's start_with_switchins is draw-free, so this seed carries straight into the turn).
    let regice = "Regice|||NoAbility|icebeam,thunderbolt,psychic,explosion|Hardy|,,,252,,|N||||";
    let groudon = "Groudon|||Drought|swordsdance,earthquake,rockslide,thunderwave|Hardy||N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(regice, groudon, "37096,12962,35091,31037"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    assert_eq!(st.field.weather, Some(Weather::Sun), "Drought set permanent sun at switch-in");

    // Turn 1: p1 Ice Beam (slot 0) into Groudon; p2 Swords Dance (slot 0, draw-free).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);

    // THE FIX: Groudon is UN-frozen (sun blocks freeze). Under the bug it would be Frozen —
    // the direct STATE assertion that fails on a revert of the sun-freeze gate. (The
    // per-decision MonSnapshot carries no species, so assert species on the live state.)
    assert_eq!(
        st.sides[1].pokemon[st.sides[1].active].species_id, "groudon",
        "Groudon is p2's active"
    );
    assert!(
        out.decisions[0].active[1].status.is_none(),
        "sun BLOCKS the freeze — Groudon stays un-frozen (under the bug it would be Frozen)"
    );
    // (And Groudon did take the Ice Beam hit — the freeze secondary genuinely rolled a
    // land; it is only the APPLICATION that sun suppressed, not the damage/roll.)
    assert!(
        out.decisions[0].active[1].hp < out.decisions[0].active[1].maxhp,
        "Groudon took the Ice Beam damage (the hit landed; only the freeze was blocked)"
    );
    // GROUND-TRUTH SEED (probe_sun_freeze_regression_rng.js): the freeze secondary's
    // random(100) STILL drew (the immunity is draw-free), so the post-decision seed is
    // IDENTICAL to the freeze-LANDS no-sun control — a re-introduced extra/missing draw
    // (e.g. wrongly running the SetStatus shuffle under sun) trips this.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "60629,35650,47398,62760",
        "post-decision seed == the real Showdown seed (draw-free sun-freeze gate)"
    );
}

// ============================================================================
// FZ2 — MOVE-ID ALIAS resolution (`gen3_move_alias_resolution_v1`). A packed team CAN
//       carry a move ALIAS (`wisp` for Will-O-Wisp — the sample pool's Gengar does), which
//       Showdown resolves at `dex.moves.get()` and RUNS as the canonical move (drawing its
//       accuracy roll). WRONG (pre-fix): the port's `move_at -> dex.moves("wisp")` returned
//       `None`, so `run_move` NO-OP'd the move drawing NOTHING while the sim ran it — a
//       bit-for-bit draw-COUNT desync that cascaded the decision boundaries (the e2e_86
//       divergence surfaced once the DMG_MOD abilities admitted a `wisp`-carrying team). The
//       fix: `dex.moves()` resolves the id through `gen3_move_aliases.json` (extractor-emitted
//       from Showdown's `aliases.ts`), mirroring the sim. This constructs the minimal repro —
//       a Gengar whose slot-0 move is spelled `wisp` uses it, BURNING the foe — and asserts
//       BOTH the STATE (foe burned = the move actually RAN) AND the post-turn SEED (the
//       accuracy roll drew). Ground truth from `harness/probe_wisp_alias_regression_rng.js`.
//       Reverting the alias resolution → `wisp` NO-OPs → the foe is un-burned + the seed desyncs.
// ============================================================================

/// FZ2: a packed-team move ALIAS (`wisp`) resolves + runs as the canonical move (Will-O-Wisp),
/// drawing its accuracy roll and burning the foe. WRONG (pre-fix): the port NO-OP'd the aliased
/// move (drawing nothing) → a draw-count desync (the e2e_86 cascade). STATE + SEED pin.
#[test]
fn move_alias_wisp_resolves_and_runs_will_o_wisp() {
    let d = dex();
    // p1 Gengar's slot-0 move is spelled `wisp` (the alias, exactly as the sample-pool team
    // carries it) — the port must resolve it to Will-O-Wisp. p2 Snorlax uses Amnesia (slot 0),
    // a modeled draw-free self-boost that can't miss/confound. Gengar (Timid, fast) moves first
    // + burns. Seeded at the probe's SEED-BEFORE-the-turn (start_with_switchins is draw-free).
    let gengar = "Gengar|||NoAbility|wisp,icepunch,firepunch,explosion|Timid|,,,252,,252|N||||";
    let snorlax = "Snorlax|||NoAbility|amnesia,bodyslam,rest,earthquake|Hardy|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(gengar, snorlax, "26072,20393,8931,7575"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");

    // Turn 1: p1 `wisp` (slot 0 → Will-O-Wisp) into Snorlax; p2 Amnesia (slot 0, draw-free).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);

    // THE FIX: `wisp` resolved to Will-O-Wisp and RAN — Snorlax is BURNED. Under the bug the
    // move NO-OP'd (the alias was unknown) → Snorlax would be un-statused. This is the STATE
    // assertion that fails on a revert of the alias resolution.
    assert_eq!(
        out.decisions[0].active[1].status,
        Some(Status::Burn),
        "the `wisp` alias resolved + ran Will-O-Wisp — Snorlax is BURNED (under the bug it no-ops)"
    );
    // GROUND-TRUTH SEED (probe_wisp_alias_regression_rng.js): Will-O-Wisp drew its accuracy
    // roll, so the post-turn seed matches the real sim. A NO-OP (the bug) would draw one FEWER
    // → this seed desyncs.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "41712,42419,17467,2225",
        "post-turn seed == the real Showdown seed (the aliased move RAN + drew its accuracy)"
    );
}

// ============================================================================
// AC — the gen3_accuracy_pipeline_v1 ACCURACY pins. The to-hit fold
//      (`effective_accuracy`: acc/eva STAGE TABLE × the accMod item/ability handlers,
//      then `random(100) < effAcc`) + Hustle's Atk ×1.5. This phase is DRAW-RELEVANT:
//      a wrong effAcc flips a hit↔miss → the crit/damage draws follow only on a hit →
//      the SEED desyncs. Each pin replays a CONSTRUCTED battle at a fixed seed where the
//      mod is DECISIVE and asserts the post-decision seed + HP (the sim's ground truth
//      from `harness/probe_accuracy_pins.js`); reverting the specific fold flips the roll
//      and trips it deterministically:
//        AC1 the ACC-STAGE fold (the fuzzer's Sand-Attack/Mud-Slap cluster) — a Mud-Slap
//            −1 accuracy drop makes the foe's Cross Chop MISS → SEED flip
//                                        → `acc_stage_drop_flips_hit_to_miss`
//        AC2 COMPOUND EYES ×1.3 — lifts Thunder(70→91); reverting drops the effAcc and
//            the roll flips at this seed → `compound_eyes_lifts_shaky_move_accuracy`
//        AC3 BRIGHT POWDER ×0.9 — drops the attacker's Cross Chop(80→72) → an extra MISS
//                                        → `bright_powder_drops_attacker_accuracy`
//        AC4 HUSTLE — the Atk ×1.5 (dmgMod, DIRECT) + acc ×0.8 (accMod) SHIP TOGETHER: a
//            heavy ×1.5 Cross Chop hit; reverting the Atk fold under-damages
//                                        → `hustle_atk_x1_5_and_accuracy_x0_8`
//        AC5 SAND VEIL's SANDSTORM-CHIP IMMUNITY — the only gen3 weather-chip onImmunity;
//            a Sand-Veil mon takes NO sand chip (ships with its ×0.8 evasion)
//                                        → `sand_veil_grants_sandstorm_chip_immunity`
//      The stage/mod math is draw-NEUTRAL (still exactly one accuracy draw), so any seed
//      drift is a real to-hit bug. Values are the REAL sim's (probe_accuracy_pins.js).
// ============================================================================

/// AC1: the ACC-STAGE fold — the fuzzer's exact cluster (a Sand-Attack/Mud-Slap −accuracy
/// drop → a hit/miss FLIP → SEED desync). A DETERMINISTIC constructed scenario
/// (`harness/probe_accuracy_stage_pin.js`): p2 Snorlax's accuracy stage is pre-set to −6, so
/// its acc-100 Body Slam rolls at 100×(3/9) = 33.3% and MISSES at this seed — p1 Miltank
/// (which just Recovers to full) stays at FULL HP. WRONG (a revert of the accuracy-stage
/// fold in `effective_accuracy`): the raw-100 roll HITS → Miltank drops to 299/394 AND the
/// crit/damage draws fire → the SEED desyncs. The hit/miss flip is the harshest mode.
#[test]
fn acc_stage_drop_flips_hit_to_miss() {
    let d = dex();
    let miltank = "Miltank|||NoAbility|recover,bodyslam|Bold|252,,252,,,|N||||";
    let snorlax = "Snorlax|||NoAbility|bodyslam,rest|Adamant|252,252,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(miltank, snorlax, "41314,9510,64531,54776"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // Pre-drop p2 Snorlax's accuracy stage to −6 (boosts index 5 = accuracy): its Body Slam
    // now rolls at 100×3/9 = 33.3%. (Models a Sand-Attack/Mud-Slap chain having landed.)
    st.sides[1].pokemon[0].boosts[5] = -6;
    // p1 Miltank Recover (slot 0, draw-free heal); p2 Snorlax Body Slam (slot 0).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Body Slam MISSED (the −6 accuracy stage) — Miltank at FULL HP. A revert of the stage
    // fold rolls at 100% → Body Slam HITS → Miltank drops to 299/394.
    assert_eq!(
        out.decisions[0].active[0].hp, 394,
        "Miltank at FULL HP — the −6 accuracy stage made Snorlax's Body Slam MISS (revert → it hits → 299)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "30148,33207,58298,62442",
        "the to-hit fold is draw-neutral (one accuracy draw); a hit/miss flip desyncs THIS seed"
    );
}

/// AC2: Compound Eyes ×1.3 lifts a shaky move — Butterfree's Thunder (acc 70) rolls at
/// 70×1.3=91, landing far more often. The `compoundeyes_attacker` golden's first-seed run:
/// Thunder repeatedly HITS Blissey (which Soft-Boils back). Replays the full run and asserts
/// a mid-run hit + the game-ending seed. WRONG (a revert of the accMod chain member): the
/// effAcc drops to 70 → a Thunder roll FLIPS to a miss → the crit/damage draws vanish → the
/// run desyncs.
#[test]
fn compound_eyes_lifts_shaky_move_accuracy() {
    let d = dex();
    let butterfree = "Butterfree|||CompoundEyes|thunder,rest|Modest|252,,,252,,|N||||";
    let blissey = "Blissey|||NoAbility|softboiled,seismictoss|Calm|252,,,,252,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(butterfree, blissey, "50192,7304,11753,24806"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // The golden's plan: Butterfree Thunder (m0); Blissey Seismic Toss (m1) then Soft-Boiled (m0).
    let tb = |p2| ScriptDecision::both(Choice::Move(0), Choice::Move(p2));
    let out = st.run_full_battle(&[tb(1), tb(0), tb(1), tb(0), tb(1), tb(0), tb(1)], &d);
    // Thunder LANDED at dec 0 (Compound Eyes' 91% roll) — Blissey took ~70 (642/714).
    assert_eq!(
        out.decisions[0].active[1].hp, 642,
        "dec 0: Compound Eyes' 91% Thunder HIT Blissey (revert to 70% → miss at this seed)"
    );
    assert_eq!(out.decisions[6].active[0].hp, 0, "dec 6: Butterfree fainted (p2 wins the stall)");
    assert_eq!(
        seed_str(&out.decisions[6].seed_after),
        "60042,45818,6182,55055",
        "Compound Eyes changes the effAcc, not the draw count; a flipped Thunder roll desyncs the run"
    );
}

/// AC3: Bright Powder ×0.9 drops the ATTACKER's accuracy — a DETERMINISTIC flip-window
/// scenario (`harness/probe_brightpowder_pin.js`): Tauros's Cross Chop (acc 80) into the
/// Bright Powder Miltank rolls at 80×0.9=72, and at this seed the roll is in [72,80) → it
/// MISSES (Miltank, which just Recovers, stays FULL). WRONG (a revert of the accMod
/// direct-multiply): the raw-80 roll HITS → the crit/damage draws fire → the SEED desyncs
/// (49228… → 26464…). Bright Powder is the DEFENDER-side onModifyAccuracy the port folds.
#[test]
fn bright_powder_drops_attacker_accuracy() {
    let d = dex();
    let tauros = "Tauros|||NoAbility|crosschop,rest|Adamant|252,252,,,,|N||||";
    let miltank = "Miltank||brightpowder|NoAbility|recover,bodyslam|Bold|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(tauros, miltank, "13471,41464,23699,36146"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // Tauros Cross Chop (slot 0) into the Bright Powder Miltank; Miltank Recover (slot 0).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Cross Chop MISSED (Bright Powder dropped 80→72; the roll was in the [72,80) flip window).
    // Miltank stays full (Recover). The SEED discriminates: a revert (raw 80) HITS → different seed.
    assert_eq!(out.decisions[0].active[1].hp, 394, "Miltank at FULL HP (Recover; Cross Chop missed)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "49228,57252,13232,65044",
        "Bright Powder's ×0.9 flipped the Cross Chop to a MISS — the crit/damage draws vanish → \
         THIS seed. A revert (raw 80% → HIT) diverges to 26464,27780,64117,4006."
    );
}

/// AC4: Hustle ships BOTH sides together — the Atk ×1.5 (a DIRECT `modify`, dmgMod) AND the
/// physical-move accuracy ×0.8 (accMod). The `hustle_physical` golden's SECOND-seed run,
/// where Delibird's Cross Chop actually LANDS: its ×1.5-Atk hit drops Snorlax to 223/524 at
/// dec 0, then KO's it at dec 1 (p1 wins). The HP is the Atk-fold gate (a revert of the
/// Hustle Atk ×1.5 under-damages → Snorlax higher); the seed pins the acc ×0.8's one draw +
/// the fold's draw-freeness. WRONG (a revert of either fold): the damage or a hit/miss shifts.
#[test]
fn hustle_atk_x1_5_and_accuracy_x0_8() {
    let d = dex();
    let delibird = "Delibird|||Hustle|crosschop,rest|Adamant|252,252,,,,|N||||";
    let snorlax = "Snorlax|||NoAbility|bodyslam,rest|Adamant|252,252,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(delibird, snorlax, "42900,38714,32434,35186"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let both = ScriptDecision::both(Choice::Move(0), Choice::Move(0));
    let out = st.run_full_battle(&[both, both], &d);
    // dec 0: Hustle's ×1.5-Atk Cross Chop LANDED, dropping Snorlax to 223/524 (a revert of the
    // Atk fold under-damages → Snorlax higher). dec 1: Snorlax KO'd, p1 wins.
    assert_eq!(
        out.decisions[0].active[1].hp, 223,
        "dec 0: Hustle's Atk ×1.5 Cross Chop dropped Snorlax to 223/524 (revert the Atk fold → higher)"
    );
    assert_eq!(out.decisions[1].active[1].hp, 0, "dec 1: Snorlax KO'd");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "62118,7672,13649,10010",
        "Hustle's ×0.8 accuracy + ×1.5 Atk both leave the draw stream bit-exact (revert → desync)"
    );
    assert_eq!(out.winner, Some(0), "Delibird (p1) wins with the boosted Cross Chop");
}

/// AC5: Sand Veil grants SANDSTORM-CHIP IMMUNITY — the ONLY gen3 weather-chip `onImmunity`.
/// Sand Veil ships alongside its ×0.8 evasion (both this phase), and a Sand-Veil mon in sand
/// must take NO chip (else the port would tick a mon the sim leaves full). Tyranitar (Sand
/// Stream) sets sand; Cacturne (Grass/Dark — NOT type-immune) has Sand Veil, so after a turn
/// it stays at FULL HP. WRONG (a revert of the Sand-Veil branch in `weather_immune`): Cacturne
/// takes maxhp/16 = 21 sand chip (→ 323/344). Pure STATE pin (the residual chip is the signal).
#[test]
fn sand_veil_grants_sandstorm_chip_immunity() {
    let d = dex();
    let tyranitar = "Tyranitar|||SandStream|rest,crunch|Serious|252,,,,,|N||||";
    let cacturne = "Cacturne|||SandVeil|rest,spikes|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(tyranitar, cacturne, "9,9,9,9"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    // Both Rest (a full-HP Rest is a no-op fail; the sand-chip residual is what we assert).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        out.decisions[0].active[1].hp, 344,
        "Cacturne (Sand Veil) at FULL HP — Sand Veil immunes the sand chip (revert → 323, i.e. −21)"
    );
    // Tyranitar (Rock) is sand-immune by TYPE (the pre-existing type gate) — also full.
    assert_eq!(out.decisions[0].active[0].hp, 404, "Tyranitar (Rock) takes no sand chip (type-immune)");
}

// ============================================================================
// FF1-FF3 — FLASH FIRE ×1.5 fire-boost (`gen3_flashfire_boost_v1`). The deferred FF
//   gap: the port modeled FF IMMUNITY (a Fire move deals 0 to the holder) but NOT the
//   post-activation ×1.5 boost on the holder's OWN Fire moves. Probe-settled
//   (`harness/probe_flashfire_rng.js` + `..._regression_rng.js`): activation is the
//   `flashfire.onTryHit` (armed on a LANDED Fire hit, DRAW-FREE, cleared on switch/
//   faint); the boost is the volatile's `onModifyDamagePhase1 chainModify(1.5)` — a
//   DAMAGE-phase fold (the SAME phase as screens, category-agnostic, NOT crit-bypassed).
// ============================================================================

/// FF1: the ×1.5 BOOST. p1 Ninetales (Flash Fire) is out-sped by a Timid Charizard, so
/// Charizard's Fire Blast lands FIRST → FF ARMS → Ninetales' SAME-TURN Flamethrower is
/// ALREADY ×1.5, dropping Charizard to 126/360 on T1. WRONG (revert the ModifyDamagePhase1
/// FF fold): the un-boosted Flamethrower under-damages → Charizard keeps MORE HP (≈171, the
/// unboosted number). Ground truth: `harness/probe_flashfire_regression_rng.js` FF1 (init
/// seed `13127,45333,18295,15391`, T1 seedAfter `38085,56695,39077,36349`). The seed is
/// bit-identical either way (activation + boost are DRAW-FREE) — the HP is the signal.
#[test]
fn flash_fire_boosts_the_holders_own_fire_move() {
    let d = dex();
    let ninetales = "Ninetales|||FlashFire|flamethrower,rest|Modest|,,,252,,|N||||";
    let charizard = "Charizard|||Blaze|fireblast,rest|Timid|252,,,,,252|N||||";
    let mut battle = Battle::start_with_switchins(
        &opts_cg(ninetales, charizard, "13127,45333,18295,15391"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    // T1: both Fire moves. Charizard (faster) Fire Blasts → 0 to Ninetales + ARMS FF; Ninetales
    // then Flamethrowers BOOSTED. T2: Charizard Rests while taking another boosted hit. (Roost is
    // a gen-4 move the port does not model, so Charizard idles with Rest.)
    let t1 = ScriptDecision::both(Choice::Move(0), Choice::Move(0));
    let t2 = ScriptDecision::both(Choice::Move(0), Choice::Move(1));
    let out = st.run_full_battle(&[t1, t2], &d);
    // Ninetales absorbed the Fire Blast (0 damage) and is ARMED.
    assert!(
        out.decisions[0].active[0].hp == 287,
        "Ninetales absorbs the Fire Blast (0 damage, full HP)"
    );
    assert!(
        battle.state().unwrap().sides[0].pokemon[0].flash_fire,
        "Ninetales' Flash Fire is ARMED after absorbing the Fire Blast"
    );
    // THE BOOST: the armed Ninetales' Flamethrower dropped Charizard to 126/360 on T1.
    assert_eq!(
        out.decisions[0].active[1].hp, 126,
        "FF ×1.5 Flamethrower dropped Charizard to 126/360 on T1 — REVERT the ModifyDamagePhase1 \
         FF fold and the un-boosted Flamethrower under-damages (Charizard keeps more HP)"
    );
    // Draw-free: the post-T1 seed is bit-exact (a revert of the fold does NOT change it — the
    // HP above is the true signal, but the seed pins the draw-freeness).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "38085,56695,39077,36349",
        "FF activation + boost are DRAW-FREE (the post-turn seed is bit-exact)"
    );
}

/// FF2: the ACTIVATION GATE — a MISSED Fire move does NOT arm Flash Fire (the `onTryHit`
/// runs AFTER the accuracy roll). Two arms from `harness/probe_flashfire_regression_rng.js`
/// FF2 (the POST-switch-in seed the port's draw-free `start_with_switchins` seeds AT): a HIT
/// seed `34864,15597,16259,24710` ARMS (flash_fire true), a MISS seed `42627,44507,9634,39140`
/// does NOT (flash_fire false). WRONG (arm before the accuracy check / unconditionally): the
/// MISS arm would show flash_fire true. Pure STATE pin (the arm flag is the signal; both draw-free).
#[test]
fn flash_fire_arms_on_a_landed_fire_hit_but_not_on_a_miss() {
    let d = dex();
    let ninetales = "Ninetales|||FlashFire|flamethrower,rest|Modest|,,,252,,|N||||";
    let charizard = "Charizard|||Blaze|fireblast|Timid|252,,,,,252|N||||";
    // p1 Ninetales idles (Rest, no attack); p2 Charizard Fire Blasts (85% acc). One turn.
    let idle_vs_fireblast = ScriptDecision::both(Choice::Move(1), Choice::Move(0));

    // HIT seed → FF arms.
    {
        let mut b = Battle::start_with_switchins(&opts_cg(ninetales, charizard, "34864,15597,16259,24710"), &d).expect("start");
        b.state_mut().unwrap().run_full_battle(&[idle_vs_fireblast], &d);
        assert!(
            b.state().unwrap().sides[0].pokemon[0].flash_fire,
            "a LANDED Fire Blast ARMS Flash Fire (this seed hits)"
        );
    }
    // MISS seed → FF does NOT arm (the onTryHit runs after the accuracy roll).
    {
        let mut b = Battle::start_with_switchins(&opts_cg(ninetales, charizard, "42627,44507,9634,39140"), &d).expect("start");
        b.state_mut().unwrap().run_full_battle(&[idle_vs_fireblast], &d);
        assert!(
            !b.state().unwrap().sides[0].pokemon[0].flash_fire,
            "a MISSED Fire Blast does NOT arm Flash Fire (this seed misses) — REVERT the \
             acc_hit gate on the activation and this arm would wrongly show flash_fire = true"
        );
    }
}

/// FF3: SWITCH-CLEAR — Flash Fire's armed state clears on switch-out (`clearVolatile`).
/// Ground truth `harness/probe_flashfire_regression_rng.js` FF3 (init seed
/// `13127,45333,18295,15391`): T1 Ninetales arms; T2 it pivots OUT to Umbreon (Charizard
/// Rests to full); T3 it pivots BACK (FF now CLEARED); T4 its Flamethrower is UNBOOSTED →
/// Charizard 287/360 (73 damage, vs the 234 a boosted hit into full HP would deal). WRONG
/// (never clear FF on switch-out): T4 would be boosted (Charizard lower). The T4 HP + the
/// cleared flag are the signals (all draw-free). (Charizard idles with Rest — Roost is a
/// gen-4 move the port does not model.)
#[test]
fn flash_fire_clears_on_switch_out() {
    let d = dex();
    let p1 = "Ninetales|||FlashFire|flamethrower,rest|Modest|,,,252,,|N||||\
]Umbreon|||Synchronize|rest,protect|Calm|252,,,,252,|N||||";
    let charizard = "Charizard|||Blaze|fireblast,rest|Timid|252,,,,,252|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, charizard, "13127,45333,18295,15391"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),  // T1: Fire Blast arms Ninetales
            ScriptDecision::both(Choice::Switch(1), Choice::Move(1)), // T2: Ninetales OUT → Umbreon (Charizard Rests)
            ScriptDecision::both(Choice::Switch(1), Choice::Move(1)), // T3: Umbreon OUT → Ninetales back (FF cleared)
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),   // T4: Flamethrower — UNBOOSTED
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 4, "the switch scenario runs all 4 turns");
    // T1: Ninetales absorbed the Fire Blast and is ARMED (its boosted Flamethrower drops Charizard).
    assert!(
        battle_arm_after_t1(&out),
        "T1: Ninetales is ARMED after absorbing the Fire Blast (its Flamethrower dropped Charizard to 126)"
    );
    // After the pivot-out-and-back, Ninetales is the active on p1 again and FF is CLEARED.
    assert!(
        !battle.state().unwrap().sides[0].pokemon[0].flash_fire,
        "Flash Fire cleared on switch-out — a switched-back Ninetales is UNARMED (revert the \
         clearVolatile flash_fire reset → it stays armed)"
    );
    // T4 Flamethrower is UNBOOSTED: Charizard at 287/360 (73 damage). A boosted hit would deal
    // ~110 (→ lower HP). The unboosted number is the signal the clear worked.
    assert_eq!(
        out.decisions[3].active[1].hp, 287,
        "the re-entered Ninetales' Flamethrower is UNBOOSTED (Charizard 287/360) — REVERT the \
         switch-clear and the still-armed Ninetales boosts it (Charizard lower)"
    );
    assert_eq!(
        seed_str(&out.decisions[3].seed_after),
        "45576,22569,57262,33093",
        "the switch-clear path is DRAW-FREE (the post-T4 seed is bit-exact)"
    );
}

/// FF3 helper: after T1 the armed Ninetales' Flamethrower dropped Charizard to 126/360 (the
/// same boosted number FF1 pins), confirming the arm-and-boost happened before the pivot.
fn battle_arm_after_t1(out: &pokesim::turn::BattleOutcome) -> bool {
    out.decisions[0].active[1].hp == 126
}

// ============================================================================
// NATURAL CURE (`gen3_natural_cure_v1`, the sole gen-3 SWITCH_OUT-cure ability) —
// the holder's MAJOR STATUS is CURED when it SWITCHES OUT (voluntary pivot OR
// phaze-DRAG-out), DRAW-FREE. Probe-settled by `harness/probe_naturalcure_rng.js`
// (trigger `onSwitchOut`, `onCheckShow` undefined; the cure + its `[silent]`
// `-curestatus` reveal consume ZERO PRNG); ground-truth seeds/state for these pins
// from `harness/probe_naturalcure_regression_rng.js`.
// ============================================================================

/// NC1: a Natural Cure holder's major status is CURED when it VOLUNTARILY switches
/// out — it RETURNS UNSTATUSED. WRONG (pre-fix / cure removed): the mon keeps its
/// status across the pivot (a non-NC mon DOES — the control below). The fix clears
/// `status` on an alive outgoing `naturalcure` holder in `execute_switch`. The cure
/// is DRAW-FREE, so the post-decision seed == a non-NC pivot's (asserted).
/// Ground truth `harness/probe_naturalcure_regression_rng.js` NC1 (aligned init seed
/// `57388,452,34593,29177`, from the sim's pre-first-decision PRNG state): a badly-poisoned (stage 5) NC Starmie pivots to Snorlax then BACK;
/// it returns `status=None`, seeds `38085,56695,39077,36349` | `43673,61326,59799,37313`
/// (BYTE-IDENTICAL to the non-NC control — the cure adds no draw). The foe is Flygon
/// (EQ; Ground can't touch Starmie/Snorlax) so the cure/residual is the only state change.
#[test]
fn natural_cure_cures_status_on_voluntary_switch_out() {
    let d = dex();
    // p1 Starmie (NC) badly-poisoned + a bench Snorlax; p2 Flygon (Levitate, Earthquake).
    let starmie_nc = "Starmie|||naturalcure|surf,recover|Timid|4,,,252,,252|||||\
]Snorlax|||owntempo|bodyslam,rest|Careful|252,,,,252,|||||";
    let starmie_nonnc = "Starmie|||illuminate|surf,recover|Timid|4,,,252,,252|||||\
]Snorlax|||owntempo|bodyslam,rest|Careful|252,,,,252,|||||";
    let flygon = "Flygon|||levitate|earthquake|Adamant|,252,,,,252|||||";

    // (a) THE NC CASE — pivot out (cured) then back; the returned Starmie is UNSTATUSED.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(starmie_nc, flygon, "57388,452,34593,29177"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    // Inject a HIGH toxic stage (5) directly — the deterministic single repro.
    st.sides[0].pokemon[0].status = Some(Status::Toxic(5));
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)), // Starmie OUT (cured) → Snorlax
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)), // Snorlax OUT → Starmie BACK
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 2, "the pivot-out-and-back scenario runs both turns");
    // After the return, Starmie is the active on p1 again and is CURED (status None).
    let star = &battle.state().unwrap().sides[0].pokemon[0];
    assert_eq!(star.species_id, "starmie", "Starmie is back in the active slot");
    assert_eq!(
        star.status, None,
        "Natural Cure CURED Starmie's Toxic on switch-OUT → it RETURNS UNSTATUSED (revert the \
         naturalcure gate in execute_switch and it stays Toxic)"
    );
    // DRAW-FREE: the per-decision seeds are bit-exact (the cure adds no PRNG draw).
    assert_eq!(seed_str(&out.decisions[0].seed_after), "38085,56695,39077,36349", "NC pivot-out seed");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "43673,61326,59799,37313", "NC pivot-back seed");

    // (b) THE NON-NC CONTROL — the SAME plan/teams but Illuminate: Starmie RETURNS STILL
    //     TOXIC (proves the pin would catch a fix that cures EVERY switch-out), and the
    //     SEEDS ARE IDENTICAL to the NC case (the cure is draw-free).
    let mut cb =
        Battle::start_with_switchins(&opts_cg(starmie_nonnc, flygon, "57388,452,34593,29177"), &d).expect("start");
    let cst = cb.state_mut().expect("state");
    cst.sides[0].pokemon[0].status = Some(Status::Toxic(5));
    let cout = cst.run_full_battle(
        &[
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)),
        ],
        &d,
    );
    let cstar = &cb.state().unwrap().sides[0].pokemon[0];
    assert_eq!(cstar.species_id, "starmie", "the control Starmie is back active");
    assert!(
        matches!(cstar.status, Some(Status::Toxic(_))),
        "a NON-NC Starmie RETAINS its Toxic across the pivot (the wrong-target discriminator)"
    );
    // Seed-neutral: the non-NC control's seeds match the NC case's exactly.
    assert_eq!(
        seed_str(&cout.decisions[0].seed_after),
        seed_str(&out.decisions[0].seed_after),
        "the cure is SEED-NEUTRAL: NC and non-NC pivot-out seeds are byte-identical"
    );
    assert_eq!(
        seed_str(&cout.decisions[1].seed_after),
        seed_str(&out.decisions[1].seed_after),
        "the cure is SEED-NEUTRAL: NC and non-NC pivot-back seeds are byte-identical"
    );
}

/// NC2: Natural Cure is a NO-OP on a FAINT — a fainted NC holder is NOT "cured" as a
/// switch-out (a fainted mon never routes through the `runEvent('SwitchOut')` cure; the
/// `onSwitchOut` guard `if (!pokemon.status || pokemon.status==='fnt') return`). So the
/// port's cure gate (`!m.fainted`) leaves the fainted mon's `status` untouched, and the
/// forced replacement is a normal switch. WRONG (if the cure fired regardless of faint):
/// the fainted Blissey's burn would be cleared. STATE pin (a fainted mon has nothing to
/// cure). Ground truth `harness/probe_naturalcure_regression_rng.js` NC2 (aligned init seed
/// `31507,25098,53503,65113`): a 1-HP burned NC Blissey is EQ-KO'd by Tyranitar; Skarmory replaces it.
#[test]
fn natural_cure_is_a_no_op_on_a_faint() {
    let d = dex();
    // p1 Blissey (NC), burned + 1 HP + a bench Skarmory; p2 Tyranitar (Earthquake KOs the 1-HP
    // Blissey — Blissey is Normal, hit for neutral; Ground can't be dodged here).
    let blissey_nc = "Blissey|||naturalcure|softboiled|Calm|252,,252,,,|||||\
]Skarmory|||keeneye|spikes|Impish|252,,252,,,|||||";
    let ttar = "Tyranitar|||sandstream|earthquake|Adamant|,252,,,,252|||||";

    let mut battle =
        Battle::start_with_switchins(&opts_cg(blissey_nc, ttar, "31507,25098,53503,65113"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[0].pokemon[0].status = Some(Status::Burn);
    st.sides[0].pokemon[0].hp = 1;
    let out = st.run_full_battle(
        &[
            // T1: Tyranitar EQ KOs the 1-HP burned Blissey → a forced replacement.
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            // The forced replacement sends Skarmory (p1 only).
            ScriptDecision { p1: Some(Choice::Switch(1)), p2: None },
        ],
        &d,
    );
    assert!(out.decisions.len() >= 2, "the KO + forced replacement produce >=2 boundaries");
    // The Blissey FAINTED — its cure did NOT fire (it never reached the SwitchOut cure). The
    // port keeps its `Some(Burn)` on the fainted mon (a fainted mon is excluded from the
    // active-status differential, so this is a direct assertion the cure gate skipped it).
    let bliss = battle.state().unwrap().sides[0].pokemon.iter()
        .find(|p| p.species_id == "blissey").expect("Blissey on the team");
    assert!(bliss.fainted, "Blissey fainted to the Earthquake");
    // RE-MEANED under `gen3_fnt_clears_status_v1` (2026-07-10): the sim's `checkFainted`
    // sets EVERY fainted active's status to `fnt` (battle.js:2082) — the port now mirrors
    // that with `None` — so the old `Some(Burn)`-persists assertion pinned a port-internal
    // representation the sim never held. The Natural-Cure-on-faint no-op is no longer
    // STATE-observable (both the fnt clear and a hypothetical faint-cure yield None); the
    // `!fainted` cure gate in `execute_switch` stays order-faithful, and NC1/NC3 keep the
    // live-cure teeth.
    assert_eq!(
        bliss.status, None,
        "a FAINTED mon's status is cleared by checkFainted (the sim's `fnt`)"
    );
    // And the replacement (Skarmory) is now active + unstatused.
    let active = &battle.state().unwrap().sides[0].pokemon[battle.state().unwrap().sides[0].active];
    assert_eq!(active.species_id, "skarmory", "Skarmory is the forced replacement");
}

/// NC3: a PHAZE (Roar/Whirlwind) DRAGGING a Natural Cure holder OUT CURES it — the
/// `runEvent('SwitchOut')` fires on `isDrag` too (only `BeforeSwitchOut` is `!isDrag`-gated).
/// WRONG (if the cure only fired on a VOLUNTARY switch): the dragged-out mon keeps its status.
/// The port routes the phaze drag through the SAME `execute_switch`, so the same gate cures it.
/// STATE (the dragged-out Starmie is unstatused on the bench) + SEED (the drag's accuracy +
/// n=1 sample draw as normal; the cure adds nothing — draw-free). Ground truth
/// `harness/probe_naturalcure_regression_rng.js` NC3 (aligned init seed `38565,7865,19639,43898`): Starmie is Toxic'd
/// T1, then Suicune ROARS it OUT T2 → Starmie cured on the bench, seed `7332,3983,10909,19927`.
#[test]
fn natural_cure_phaze_drag_cures_the_dragged_out_mon() {
    let d = dex();
    // p1 Starmie (NC) + a bench Snorlax; p2 Suicune (0-Spe Bold — slower, so its Roar [priority
    // -6] drags AFTER Starmie moves) with Toxic + Roar.
    let starmie_nc = "Starmie|||naturalcure|surf,recover|Timid|4,,,252,,252|||||\
]Snorlax|||owntempo|bodyslam,rest|Careful|252,,,,252,|||||";
    let suicune = "Suicune|||pressure|toxic,roar|Bold|252,,252,,,|||||";

    let mut battle =
        Battle::start_with_switchins(&opts_cg(starmie_nc, suicune, "38565,7865,19639,43898"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            // T1: Starmie Surf; Suicune Toxic → Starmie badly poisoned.
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            // T2: Starmie Surf; Suicune ROAR → Starmie DRAGGED OUT (cured); Snorlax dragged in.
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 2, "the toxic + phaze-drag scenario runs both turns");
    // T1: Starmie was badly poisoned (the active-status after T1 shows Toxic).
    assert!(
        matches!(out.decisions[0].active[0].status, Some(Status::Toxic(_))),
        "T1: Suicune's Toxic badly-poisoned Starmie"
    );
    // T2: the Roar DRAGGED Starmie out — Snorlax is now active; Starmie is on the bench, CURED.
    assert!(out.decisions[1].phaze_drag, "T2: the Roar fired its drag (the sample ran)");
    let active = &battle.state().unwrap().sides[0].pokemon[battle.state().unwrap().sides[0].active];
    assert_eq!(active.species_id, "snorlax", "the Roar dragged Snorlax in");
    let star = battle.state().unwrap().sides[0].pokemon.iter()
        .find(|p| p.species_id == "starmie").expect("Starmie on the team");
    assert!(!star.fainted, "the dragged-out Starmie is alive (not fainted)");
    assert_eq!(
        star.status, None,
        "the phaze DRAG cured the dragged-out Starmie's Toxic (the cure fires on isDrag too) — \
         a voluntary-only cure would leave it Toxic on the bench"
    );
    // DRAW-FREE: the post-T2 seed is bit-exact (the drag draws as normal; the cure adds nothing).
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "7332,3983,10909,19927",
        "the phaze-drag cure is DRAW-FREE (the post-T2 seed is bit-exact vs the sim)"
    );
}

// ============================================================================
// STATUS_IMMUNE (`gen3_status_immune_v1`, the DATA-DRIVEN status-immunity ability class:
// Limber par / Insomnia+Vital Spirit slp / Immunity psn,tox / Water Veil brn / Magma Armor
// frz). Read from `AbilityData.status_immune` in `turn.rs::try_set_status`. PROBE-settled
// draw model (`harness/probe_statusimmune_*.js`): a `setStatus`-phase member blocks INSIDE
// runEvent('SetStatus') (its handler sorts into its own speed group, so the gen3ou 2-clause
// tie stays SIZE-2 — 1 draw, unchanged; DRAW-FREE in customgame); Magma Armor blocks via
// `onImmunity` BEFORE the SetStatus event. Ground-truth seeds/state for these pins from
// `harness/probe_statusimmune_regression_rng.js` (aligned init seeds, post-switch-in).
// ============================================================================

/// A Snorlax holder (Normal — no status TYPE-immunity, so the ability is the only block) with
/// the ability under test: Body Slam (slot 0) + Seismic Toss (slot 1). MATCHES the probe's
/// `holder()` byte-for-byte so the captured seeds line up.
fn si_holder(ability_id: &str) -> String {
    format!("Snorlax|||{ability_id}|bodyslam,seismictoss|Careful|252,,128,,128,|||||")
}
/// A Blissey foe that RE-FIRES the status move (slot 0) + Seismic Toss (slot 1). MATCHES the
/// probe's `foe()`.
fn si_foe(status_move: &str) -> String {
    format!("Blissey|||serenegrace|{status_move},seismictoss|Bold|4,,252,,252,|||||")
}

/// SI1: LIMBER blocks paralysis. Thunder Wave into a Limber Snorlax leaves it UNPARALYZED
/// (the `onSetStatus`-phase block, DRAW-FREE in customgame — the ability is the ONLY SetStatus
/// handler → no shuffle). WRONG (block reverted): the mon is PARALYZED → its speed drops to ¼ →
/// later turns' action order + the para roll diverge → the multi-turn SEED desyncs (asserted).
/// Ground truth `harness/probe_statusimmune_regression_rng.js` SI1 (aligned init seed
/// `61255,39458,1834,64539`).
#[test]
fn limber_blocks_paralysis_draw_free() {
    let d = dex();
    let mut battle = Battle::start_with_switchins(
        &opts_cg(&si_holder("limber"), &si_foe("thunderwave"), "61255,39458,1834,64539"),
        &d,
    )
    .expect("start");
    let out = battle.state_mut().expect("state").run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // TWave into Limber → blocked
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // holder still full-speed
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // TWave again → still blocked
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 3, "the 3-turn scenario runs to completion");
    // STATE: the Limber Snorlax is NEVER paralyzed.
    let lax = &battle.state().unwrap().sides[0].pokemon[0];
    assert_eq!(lax.species_id, "snorlax");
    assert_eq!(
        lax.status, None,
        "Limber BLOCKS paralysis — the Snorlax stays unstatused (revert the status_immune gate \
         and Thunder Wave paralyzes it)"
    );
    // SEED (draw-free block; a reverted block paralyzes → changes later action order + para draws).
    assert_eq!(seed_str(&out.decisions[0].seed_after), "37112,13693,28533,21721", "SI1 T1 seed");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "54523,22811,31582,9991", "SI1 T2 seed");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "4112,59617,252,56412", "SI1 T3 seed");
}

/// SI2: INSOMNIA blocks sleep. Spore into an Insomnia Snorlax leaves it AWAKE. The block is
/// DRAW-FREE, but a LANDED sleep draws the gen-3 `slp.onStart` `random(2,6)` duration — so
/// reverting the block desyncs the SAME turn's SEED (a draw-COUNT pin, not just state). WRONG
/// (block reverted): the mon SLEEPS + the `random(2,6)` fires → T1 seed diverges. Ground truth
/// SI2 (aligned init seed `25653,14440,23681,28933`).
#[test]
fn insomnia_blocks_sleep_draw_free() {
    let d = dex();
    let mut battle = Battle::start_with_switchins(
        &opts_cg(&si_holder("insomnia"), &si_foe("spore"), "25653,14440,23681,28933"),
        &d,
    )
    .expect("start");
    let out = battle.state_mut().expect("state").run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Spore into Insomnia → blocked
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // holder acts normally (awake)
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 2);
    let lax = &battle.state().unwrap().sides[0].pokemon[0];
    assert_eq!(
        lax.status, None,
        "Insomnia BLOCKS sleep — the Snorlax stays awake (revert the gate and Spore sleeps it)"
    );
    // SEED: T1 is DRAW-FREE here; a reverted block draws the slp random(2,6) → this seed changes.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "24947,26147,56575,55187",
        "SI2 T1 seed — the block draws NOTHING; a landed sleep would draw random(2,6) and desync"
    );
    assert_eq!(seed_str(&out.decisions[1].seed_after), "31057,12904,42828,64017", "SI2 T2 seed");
}

/// SI3: MAGMA ARMOR blocks freeze. Ice Beam (10% frz secondary) into a Magma Armor Snorlax
/// NEVER freezes — the `onImmunity`-phase block fires at `runStatusImmunity`, BEFORE the
/// SetStatus event (like the sun-freeze gate). On the chosen seed the freeze secondary FIRES
/// (a NO-ability control FROZE, verified by the probe), so this is a genuine block. The block is
/// DRAW-FREE (the secondary `random(100)` drew either way → the seed is UNCHANGED from the
/// control). WRONG (block reverted): the Snorlax FREEZES (a STATE change). STATE + SEED. Ground
/// truth SI3 (aligned init seed `9211,13032,12358,42006`; a NO-ability Snorlax on this seed FROZE
/// while Magma Armor stays `ok`; both seedAfter `63608,32001,50713,50540`).
#[test]
fn magma_armor_blocks_freeze() {
    let d = dex();
    let ice_foe = "Regice|||clearbody|icebeam,seismictoss|Modest|4,252,,252,,|||||";
    // (a) MAGMA ARMOR: Ice Beam (frz secondary fires on this seed) → NEVER frozen.
    let mut battle = Battle::start_with_switchins(
        &opts_cg(&si_holder("magmaarmor"), ice_foe, "9211,13032,12358,42006"),
        &d,
    )
    .expect("start");
    let out = battle.state_mut().expect("state").run_full_battle(
        &[ScriptDecision::both(Choice::Move(1), Choice::Move(0))], // holder Seismic Toss; foe Ice Beam
        &d,
    );
    let lax = &battle.state().unwrap().sides[0].pokemon[0];
    assert_eq!(
        lax.status, None,
        "Magma Armor BLOCKS freeze — the Snorlax is NEVER frozen (revert the immunity-phase gate \
         and the Ice Beam freeze secondary freezes it)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "63608,32001,50713,50540",
        "the freeze block is DRAW-FREE (the secondary random(100) drew either way)"
    );
    // (b) THE CONTROL: a NO-ability Snorlax on the SAME seed DOES freeze (the discriminator —
    //     proving Ice Beam CAN freeze this target, so Magma Armor's `ok` above is the block).
    let mut cb = Battle::start_with_switchins(
        &opts_cg(&si_holder("No Ability"), ice_foe, "9211,13032,12358,42006"),
        &d,
    )
    .expect("start");
    let cout = cb.state_mut().expect("state").run_full_battle(
        &[ScriptDecision::both(Choice::Move(1), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        cb.state().unwrap().sides[0].pokemon[0].status,
        Some(Status::Freeze),
        "the NO-ability control FREEZES on this seed (the block discriminator)"
    );
    // Draw-free: the block and the freeze land at the SAME seed (the secondary drew either way).
    assert_eq!(
        seed_str(&cout.decisions[0].seed_after),
        seed_str(&out.decisions[0].seed_after),
        "Magma Armor's block is SEED-NEUTRAL vs the freezing control (draw-free)"
    );
}

/// SI4: IMMUNITY is STATUS-SPECIFIC — it blocks psn/tox but NOT brn. Toxic into an Immunity
/// Snorlax is BLOCKED (stays clean); Will-O-Wisp into the SAME Immunity Snorlax BURNS. WRONG
/// (an over-broad block): the burn would also be blocked. STATE (two arms) + SEED (both draw-
/// free). Ground truth SI4 (aligned init seed `26800,52733,48763,51466`).
#[test]
fn immunity_blocks_tox_but_not_burn() {
    let d = dex();
    // (a) TOX arm — Toxic into Immunity → BLOCKED.
    let mut btox = Battle::start_with_switchins(
        &opts_cg(&si_holder("immunity"), &si_foe("toxic"), "26800,52733,48763,51466"),
        &d,
    )
    .expect("start");
    let otox = btox.state_mut().expect("state").run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        btox.state().unwrap().sides[0].pokemon[0].status,
        None,
        "Immunity BLOCKS Toxic — the Snorlax stays clean (revert the gate and it gets badly poisoned)"
    );
    assert_eq!(seed_str(&otox.decisions[0].seed_after), "14296,56023,929,26368", "SI4 tox-arm seed");
    // (b) BRN arm — Will-O-Wisp into the SAME Immunity Snorlax → BURNS (Immunity blocks psn/tox only).
    let mut bbrn = Battle::start_with_switchins(
        &opts_cg(&si_holder("immunity"), &si_foe("willowisp"), "26800,52733,48763,51466"),
        &d,
    )
    .expect("start");
    let obrn = bbrn.state_mut().expect("state").run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        bbrn.state().unwrap().sides[0].pokemon[0].status,
        Some(Status::Burn),
        "Immunity does NOT block a burn — Will-O-Wisp BURNS the Immunity Snorlax (the block is \
         status-specific; an over-broad block would wrongly block this)"
    );
    assert_eq!(seed_str(&obrn.decisions[0].seed_after), "14296,56023,929,26368", "SI4 brn-arm seed");
}

// ============================================================================
// WEATHER_EACHEVENT (`gen3_ability_batch1_v1`, the STEP-1 fix) — gen3 SUN + RAIN fire
//   `this.eachEvent('Weather')` at EVERY end-of-turn UNCONDITIONALLY (the resolved
//   `onFieldResidual` body is a bare `this.add('-weather',…,'[upkeep]'); this.eachEvent('Weather');`
//   — NO `isWeather` guard, unlike sand/hail). That `eachEvent('Weather')` speed-sorts the actives
//   → on a speed TIE it draws ONE `random(0,2)` Fisher-Yates shuffle. WRONG (pre-fix): the port
//   gated the END-OF-TURN weather tie-shuffle on `Sand | Hail` ONLY → a WEATHER-TURN speed TIE under
//   sun/rain MISSED that draw (a 1-draw desync on every later turn). The fix schedules the field
//   weather-residual (which fires the shuffle) off the RAW `field.weather` for sun/rain (so it fires
//   even under a Cloud Nine / Air Lock negater) and off `effectiveWeather()` for sand/hail (a negater
//   suppresses those). Ground truth: `harness/probe_weather_eachevent_tie_regression_rng.js`
//   (semantics re-confirmed vs the resolved dist by `probe_weather_eachevent_sunrain.js`).
// ============================================================================

/// WEATHER_EACHEVENT: a weather-turn SPEED TIE under RAIN draws the end-of-turn
/// `eachEvent('Weather')` shuffle. WRONG (pre-fix): the port fired the end-of-turn weather
/// tie-shuffle ONLY under Sand|Hail, so a rain-turn tie drew ONE FEWER PRNG call → the post-turn
/// seed diverged on every later turn. The fix (`run_residuals` schedules the weather field-residual
/// off the RAW `field.weather` for sun/rain) makes `apply_weather_chip` fire the shuffle under rain.
///
/// Constructed Kyogre-vs-Kyogre MIRROR (both Drizzle → RAIN on switch-in, exact speed TIE spe 216):
/// turn 1 (move) BOTH Splash. The tie makes the action-order + per-action eachEvent shuffles draw,
/// then the END-OF-TURN `eachEvent('Weather')` under RAIN draws ONE MORE `random(0,2)` (the FIX —
/// rain has no chip, so this shuffle IS the whole field-residual), then the Quick Claw. A model that
/// gates the end-of-turn weather shuffle on Sand|Hail draws one fewer → this seed diverges.
#[test]
fn sun_rain_weather_turn_tie_draws_the_eachevent_weather_shuffle_seed() {
    let d = dex();
    // Kyogre (Drizzle) mirror, 0 spe EV Serious → both spe 216 (exact TIE); both Splash.
    let p1 = "Kyogre|||drizzle|splash,surf|Serious|252,,,,,|||||\
              ]Snorlax|||immunity|splash,bodyslam|Serious|252,,,,,|||||";
    let p2 = "Kyogre|||drizzle|splash,surf|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "41231,8877,60013,25519"), &d).expect("start");
    let st = battle.state_mut().expect("state");

    // Both leads' Drizzle set rain on switch-in (draw-free in the port; permanent gen-3 weather).
    assert_eq!(
        st.field.weather,
        Some(Weather::Rain),
        "the Kyogre-mirror Drizzle set rain on the switch-in"
    );

    // Turn 1 (move): BOTH Splash — the actives TIE under rain, so the end-of-turn
    // `eachEvent('Weather')` shuffle draws (the FIX).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // STATE: rain persists, both Kyogre at full HP (rain does NO chip — Splash does nothing).
    assert_eq!(st.field.weather, Some(Weather::Rain), "rain persists (permanent)");
    for side in 0..2 {
        let a = st.sides[side].active;
        assert_eq!(
            st.sides[side].pokemon[a].hp, st.sides[side].pokemon[a].maxhp,
            "rain does NO chip — both actives stay at full HP after a double Splash"
        );
    }

    // GROUND TRUTH (probe_weather_eachevent_tie_regression_rng.js, real Showdown, reseeded to the
    // RAW seed at the decision so it lines up with the Rust's draw-free `start_with_switchins`):
    // the post-turn seed == the real Showdown seed. The rain-turn tie draws 8 (incl. the end-of-turn
    // `eachEvent('Weather')` tie-shuffle). WITHOUT that shuffle the port draws 7 → this seed diverges
    // (the STEP-1 sun/rain weather-eachEvent desync).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "57197,10518,160,15031",
        "post-turn seed == the real Showdown seed for a RAIN-turn speed tie — the end-of-turn \
         `eachEvent('Weather')` shuffle MUST draw under sun/rain (not just sand/hail); without it \
         the port draws one fewer call and this desyncs (the STEP-1 `gen3_ability_batch1_v1` fix)"
    );
}

// ============================================================================
// ABILITY BATCH-1 CLASSES (`gen3_ability_batch1_v1`) — the four DRAW-FREE / STRUCTURAL ability
//   classes wired in this batch, each pinned by a CONSTRUCTED scenario reseeded to the RAW seed
//   (so the port's draw-free `start_with_switchins` aligns) with a real-Showdown ground-truth
//   seed + STATE from `harness/probe_ability_batch1_regression_rng.js`. Each pin FAILS if its
//   class's engine wiring is reverted:
//     B1 CRIT_IMMUNE  — a hit into a Battle/Shell Armor holder NEVER crits (the crit roll is
//        DRAWN then overridden false → draw-free); a no-op control DOES crit at the SAME seed.
//     B2 WEATHER_SPEED — a Chlorophyll / Swift Swim mon's ×2 effective speed (in ability-set
//        sun / rain) FLIPS the first mover; a no-op control does not.
//     B3 WEATHER_NEGATE — a Cloud Nine / Air Lock mon takes NO sand chip (the weather's EFFECTS
//        are suppressed); a no-op control TAKES the chip.
//     B4 RESIDUAL — Speed Boost +1 spe stage per active turn (activeTurns-gated), DRAW-FREE.
// ============================================================================

/// B1 CRIT_IMMUNE: a high-crit foe move (Slash, 1/8) into a BATTLE ARMOR Snorlax NEVER crits —
/// the crit `randomChance` is DRAWN (draw-count unchanged) then `runEvent('CriticalHit')`
/// overrides it to false. WRONG (a model that skips the roll, or lets the crit land): the seed
/// desyncs (skipped roll) or the armor mon takes 2× (unprevented crit). On this seed the crit
/// roll COMES UP: a no-op-ability (Insomnia) control CRITS (Snorlax → 123) while Battle Armor
/// prevents it (Snorlax → 324) at the IDENTICAL post-turn seed (the draw-free override).
#[test]
fn battle_armor_prevents_the_crit_but_draws_the_roll() {
    let d = dex();
    let ursaring = "Ursaring|||Guts|slash,rest|Adamant|252,252,,,,|N||||";
    let seed = "21041,42460,1931,46958";

    // ARMOR: Battle Armor Snorlax — the crit is prevented (takes the NON-crit hit).
    let armor = "Snorlax|||BattleArmor|bodyslam,rest|Careful|252,,,,252,|N||||";
    let mut ba = Battle::start_with_switchins(&opts_cg(armor, ursaring, seed), &d).expect("start");
    let oa = ba.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let a_hp = ba.state().unwrap().sides[0].pokemon[0].hp;
    assert_eq!(a_hp, 324, "Battle Armor Snorlax took the NON-crit Slash (324), the crit prevented");
    assert_eq!(
        seed_str(&oa.decisions[0].seed_after),
        "45626,16763,19968,16790",
        "B1: the crit roll is STILL DRAWN under Battle Armor (draw-free override) — post-turn seed"
    );

    // CONTROL: no-op ability (Insomnia) — the crit LANDS at the SAME seed (much more damage).
    let ctl = "Snorlax|||Insomnia|bodyslam,rest|Careful|252,,,,252,|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(ctl, ursaring, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let c_hp = bc.state().unwrap().sides[0].pokemon[0].hp;
    assert_eq!(c_hp, 123, "the Insomnia control TOOK the crit (123) — proving the crit roll came up");
    assert!(a_hp > c_hp, "Battle Armor prevented the crit → LESS damage than the control");
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        "45626,16763,19968,16790",
        "B1: the control's post-turn seed is IDENTICAL to Battle Armor's (the override is draw-free)"
    );
}

/// CI1/CI2 CRIT_ITEM (`gen3_crit_item_v1`, STICK) — a Farfetch'd holding **Stick** folds
/// `onModifyCritRatio critRatio + 2` (species-gated to Farfetch'd), so its base critRatio 1 → 3
/// (`randomChance(1, CRIT_MULT[3]=4)` = 1/4). On the chosen raw seed the crit COMES UP (Cut into
/// Snorlax crits → 347/524), while the SAME board WITHOUT the item stays critRatio 1
/// (`randomChance(1,16)`) → NO crit → 436/524. WRONG (a model that ignores `crit_boost`): the
/// Stick Farfetch'd would NOT crit at this seed (same as the control). The crit roll draws ONE
/// `random(denom)` either way (same PRNG consumption), so both boards share the IDENTICAL
/// post-turn seed — the DRAW-FREE-denominator-shift proof (ground truth
/// `harness/probe_batch89_stick_regression_rng.js`).
#[test]
fn stick_boosts_farfetchd_crit_ratio_by_two() {
    let d = dex();
    let snorlax = "Snorlax|||Immunity|bodyslam,rest|Careful|252,,,,252,|N||||";
    let seed = "8,15,9,14";

    // CI1 STICK: the crit LANDS (critRatio 3 → 1/4).
    let stick = "Farfetch'd||stick|KeenEye|cut,rest|Adamant|,252,,,,252|N||||";
    let mut bs = Battle::start_with_switchins(&opts_cg(stick, snorlax, seed), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let s_hp = bs.state().unwrap().sides[1].pokemon[0].hp;
    assert_eq!(s_hp, 347, "CI1: the Stick Farfetch'd CRITS Cut into Snorlax (347/524)");
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "50794,12741,4441,38454",
        "CI1: the crit roll is DRAWN either way (draw-free denominator shift) — post-turn seed"
    );

    // CI2 CONTROL: no item → critRatio 1 (1/16) → NO crit at the SAME seed (more HP left).
    let noitem = "Farfetch'd|||KeenEye|cut,rest|Adamant|,252,,,,252|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(noitem, snorlax, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let c_hp = bc.state().unwrap().sides[1].pokemon[0].hp;
    assert_eq!(c_hp, 436, "CI2: the no-item control does NOT crit (436/524) — proving the +2 fold");
    assert!(s_hp < c_hp, "Stick's +2 crit ratio → the crit landed → MORE damage than the control");
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        "50794,12741,4441,38454",
        "CI2: the control's post-turn seed is IDENTICAL to Stick's (the crit-ratio fold is draw-free)"
    );
}

/// HZ1 HAZE (`gen3_haze_v1`) — Snorlax climbs to +4 Atk (Swords Dance) while Alakazam climbs to
/// +2 SpA/SpD (Calm Mind), then Snorlax HAZES → `getAllActive().clearBoosts()` zeroes BOTH actives'
/// boost stages INCLUDING the user's OWN +4 Atk. DRAW-FREE (the Haze turn's post-turn seed is
/// exact). WRONG (a model that only clears the foe, or nothing): the boosts stand at the Haze turn.
/// Ground truth `harness/probe_batch89_haze_regression_rng.js`.
#[test]
fn haze_clears_both_actives_boosts_including_the_users_own() {
    let d = dex();
    let snorlax = "Snorlax|||Immunity|swordsdance,haze,bodyslam,splash|Careful|252,4,,,252,|N||||";
    let alakazam = "Alakazam|||Synchronize|calmmind,psychic|Modest|4,,252,,252,|N||||";
    let seed = "7,19,23,31";
    let mut b = Battle::start_with_switchins(&opts_cg(snorlax, alakazam, seed), &d).expect("start");
    let o = b.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // dec0: SD +2 / CM +1
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // dec1: SD +4 / CM +2
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // dec2: HAZE (Alakazam CMs first at +3, then Haze wipes)
        ],
        &d,
    );
    // dec1: the boosts are UP (Snorlax +4 Atk, Alakazam +2 SpA/+2 SpD) — the pre-Haze peak.
    assert_eq!(o.decisions[1].active[0].boosts[0], 4, "HZ1: Snorlax reached +4 Atk before Haze");
    assert_eq!(
        (o.decisions[1].active[1].boosts[2], o.decisions[1].active[1].boosts[3]),
        (2, 2),
        "HZ1: Alakazam reached +2 SpA/+2 SpD before Haze"
    );
    // dec2: HAZE zeroed BOTH actives' 7 boost stages (the user's own +4 Atk INCLUDED).
    assert_eq!(
        o.decisions[2].active[0].boosts, [0i8; 7],
        "HZ1: Haze clears the USER's OWN boosts too (getAllActive)"
    );
    assert_eq!(
        o.decisions[2].active[1].boosts, [0i8; 7],
        "HZ1: Haze clears the FOE's boosts"
    );
    assert_eq!(
        seed_str(&o.decisions[2].seed_after),
        "21454,21319,1997,2816",
        "HZ1: the Haze turn is DRAW-FREE — exact post-turn seed"
    );
}

/// HZ2 HAZE draw-freeness — Haze cast with NO boosts up is a pure no-op: the post-turn seed is
/// IDENTICAL to a Splash control on the SAME board/seed (only the endTurn Quick Claw draws either
/// way). WRONG (a Haze that drew anything): the seeds would differ.
#[test]
fn haze_with_no_boosts_is_draw_free_like_splash() {
    let d = dex();
    let snorlax = "Snorlax|||Immunity|swordsdance,haze,bodyslam,splash|Careful|252,4,,,252,|N||||";
    let alakazam = "Alakazam|||Synchronize|calmmind,psychic|Modest|4,,252,,252,|N||||";
    let seed = "7,19,23,31";

    // Haze (move 2 = idx 1) with nothing to clear vs the foe's Psychic.
    let mut bh = Battle::start_with_switchins(&opts_cg(snorlax, alakazam, seed), &d).expect("start");
    let oh = bh.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(1), Choice::Move(1))],
        &d,
    );
    assert_eq!(oh.decisions[0].active[0].boosts, [0i8; 7], "HZ2: nothing to clear (stays all-0)");
    assert_eq!(
        seed_str(&oh.decisions[0].seed_after),
        "41538,50205,58183,61362",
        "HZ2: the Haze no-op post-turn seed"
    );

    // Splash (move 4 = idx 3) control — the SAME draws (Haze's clear is draw-free) → SAME seed.
    let mut bs = Battle::start_with_switchins(&opts_cg(snorlax, alakazam, seed), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(3), Choice::Move(1))],
        &d,
    );
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        seed_str(&oh.decisions[0].seed_after),
        "HZ2: the Splash control's post-turn seed is IDENTICAL to Haze's (Haze is draw-free)"
    );
}

/// LO1 LIQUID OOZE drain reversal (`gen3_liquid_ooze_v1`) — a Giga Drain into a Liquid Ooze mon
/// turns the would-be heal into DAMAGE on the DRAINER (`onSourceTryHeal`). Venusaur takes the
/// reversal (364 → 328) instead of healing; a Clear-Body control on the SAME board/seed does NOT
/// (Venusaur stays 364), and the two share the IDENTICAL post-turn seed (the reversal is DRAW-FREE).
/// WRONG (a model that heals, or the old fail-loud): the drainer would not lose HP. Ground truth
/// `harness/probe_batch89_liquidooze_regression_rng.js`.
#[test]
fn liquid_ooze_reverses_a_drain_heal_into_damage_on_the_drainer() {
    let d = dex();
    let venu = "Venusaur|||Overgrow|gigadrain,sludgebomb|Modest|252,,4,252,,|N||||";
    let seed = "11,23,37,41";

    // Liquid Ooze Tentacruel — the reversal DAMAGES Venusaur (364 → 328).
    let tenta_lo = "Tentacruel|||Liquid Ooze|barrier,splash|Bold|252,,,,252,|N||||";
    let mut bo = Battle::start_with_switchins(&opts_cg(venu, tenta_lo, seed), &d).expect("start");
    let oo = bo.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(1))],
        &d,
    );
    assert_eq!(bo.state().unwrap().sides[0].pokemon[0].hp, 328, "LO1: the drainer TAKES the reversal (328)");
    assert_eq!(
        seed_str(&oo.decisions[0].seed_after),
        "35719,30677,88,6445",
        "LO1: the reversal is DRAW-FREE — the drain move's normal draws only"
    );

    // Clear Body control — Venusaur does NOT take the reversal (stays at full 364).
    let tenta_cb = "Tentacruel|||Clear Body|barrier,splash|Bold|252,,,,252,|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(venu, tenta_cb, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(1))],
        &d,
    );
    assert_eq!(bc.state().unwrap().sides[0].pokemon[0].hp, 364, "LO1: the Clear-Body control does NOT reverse (364)");
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        "35719,30677,88,6445",
        "LO1: the control's post-turn seed is IDENTICAL to Liquid Ooze's (the reversal is draw-free)"
    );
}

/// LO2 LIQUID OOZE leech-seed reversal (`gen3_liquid_ooze_v1`) — the Leech Seed residual into a
/// Liquid Ooze mon drains the seeded mon (Tentacruel `-damage [from] Leech Seed`, 364 → 319 =
/// maxhp/8) THEN reverses the seeder's would-be heal into DAMAGE (Venusaur `-damage [from] ability:
/// Liquid Ooze`, 364 → 319). DRAW-FREE. WRONG (a model that heals the seeder, or fail-louds): the
/// seeder gains HP instead of losing it. Ground truth `harness/probe_batch89_liquidooze_regression_rng.js`.
#[test]
fn liquid_ooze_reverses_the_leech_seed_drain_onto_the_seeder() {
    let d = dex();
    let venu = "Venusaur|||Overgrow|leechseed,sludgebomb|Modest|252,,4,252,,|N||||";
    let tenta_lo = "Tentacruel|||Liquid Ooze|barrier,splash|Bold|252,,,,252,|N||||";
    let seed = "11,23,37,41";
    let mut b = Battle::start_with_switchins(&opts_cg(venu, tenta_lo, seed), &d).expect("start");
    let o = b.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(1))], // Leech Seed + first residual
        &d,
    );
    assert_eq!(o.decisions[0].active[0].hp, 319, "LO2: the SEEDER takes the reversal (Venusaur 364 → 319)");
    assert_eq!(o.decisions[0].active[1].hp, 319, "LO2: the seeded Liquid Ooze mon still takes the leech drain (Tentacruel 364 → 319)");
    assert_eq!(
        seed_str(&o.decisions[0].seed_after),
        "50750,2780,45521,10355",
        "LO2: the leech residual + reversal are DRAW-FREE"
    );
}

/// LO3 LIQUID OOZE drain reversal KO (`gen3_liquid_ooze_v1`) — the reversal DAMAGE can KO the
/// drainer through the normal deferred-faint machinery (like a recoil KO). A frail Venusaur
/// draining a Liquid Ooze Tentacruel (that Surfs it low first) FAINTS to its OWN Giga Drain's
/// reversal on dec2 → a mono Venusaur loses. WRONG (a model that heals): the drainer never dies.
/// Ground truth `harness/probe_batch89_liquidooze_regression_rng.js`.
#[test]
fn liquid_ooze_drain_reversal_can_ko_the_drainer() {
    let d = dex();
    let venu = "Venusaur|||Overgrow|gigadrain,synthesis|Modest|,,,252,,252|N||||";
    let tenta = "Tentacruel|||Liquid Ooze|surf,splash|Modest|252,,,252,,|N||||";
    let seed = "11,23,37,41";
    let mut b = Battle::start_with_switchins(&opts_cg(venu, tenta, seed), &d).expect("start");
    let o = b.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );
    assert_eq!(o.decisions.len(), 3, "LO3: the reversal KO ends the battle at dec2 (mono Venusaur)");
    assert!(o.decisions[2].active[0].fainted, "LO3: Venusaur FAINTED to its OWN Giga Drain reversal");
    assert_eq!(o.decisions[2].active[0].hp, 0, "LO3: Venusaur is at 0 HP");
    assert_eq!(o.winner, Some(1), "LO3: mono Venusaur fainting to the reversal → P2 wins");
    assert_eq!(
        seed_str(&o.decisions[2].seed_after),
        "23735,6545,34360,21181",
        "LO3: the reversal KO turn's post-turn seed"
    );
}

/// WH1 WHITE HERB self-drop restore + single-use (`gen3_white_herb_v1`) — a White Herb Snorlax
/// SUPERPOWERS: the 1st self-drop (Atk/Def −1) is RESTORED to 0 + the item consumed (dec0); the 2nd
/// Superpower drops −1/−1 with NO restore (item gone — SINGLE-USE). The restore is DRAW-FREE, so
/// dec0's post-turn seed is IDENTICAL to a no-White-Herb control's (which shows the unrestored
/// −1/−1). WRONG (a model that skips the restore): dec0 shows −1/−1 + the item retained.
/// Ground truth `harness/probe_batch89_whiteherb_regression_rng.js`.
#[test]
fn white_herb_self_drop_restores_then_single_use() {
    let d = dex();
    let wh = "Snorlax||whiteherb|Own Tempo|superpower,splash|Adamant|252,252,4,,,|N||||";
    let noitem = "Snorlax|||Own Tempo|superpower,splash|Adamant|252,252,4,,,|N||||";
    let foe = "Snorlax|||Own Tempo|splash|Careful|252,,252,,4,|N||||";
    let seed = "11,29,37,53";
    // White Herb: dec0 restores Atk/Def to 0 + consumes; dec1 drops −1/−1 (single-use).
    let mut bo = Battle::start_with_switchins(&opts_cg(wh, foe, seed), &d).expect("start");
    let o = bo.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );
    assert_eq!(o.decisions[0].active[0].boosts, [0i8; 7], "WH1: the self-drop Atk/Def −1 is RESTORED to 0");
    assert!(!o.decisions[0].active[0].item_held, "WH1: White Herb was CONSUMED (single-use)");
    assert_eq!(
        (o.decisions[1].active[0].boosts[0], o.decisions[1].active[0].boosts[1]),
        (-1, -1),
        "WH1: the 2nd Superpower drops −1/−1 with NO restore (item gone — single-use)"
    );
    assert_eq!(
        seed_str(&o.decisions[0].seed_after),
        "24105,41226,41889,57409",
        "WH1: the restore is DRAW-FREE — dec0 post-turn seed"
    );
    // The no-item control: dec0 shows the UNrestored −1/−1 at the SAME seed (draw-free proof).
    let mut bc = Battle::start_with_switchins(&opts_cg(noitem, foe, seed), &d).expect("start");
    let c = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        (c.decisions[0].active[0].boosts[0], c.decisions[0].active[0].boosts[1]),
        (-1, -1),
        "WH1: the no-White-Herb control does NOT restore (−1/−1)"
    );
    assert_eq!(
        seed_str(&c.decisions[0].seed_after),
        "24105,41226,41889,57409",
        "WH1: the control's post-turn seed is IDENTICAL to White Herb's (the restore is draw-free)"
    );
}

/// WH2 WHITE HERB foe stat-drop-MOVE restore (`gen3_white_herb_v1`) — a foe Charm (−2 Atk) into a
/// White Herb Snorlax is RESTORED to 0 + the item consumed (the `apply_secondary_boost` trigger
/// path). DRAW-FREE. WRONG (a model that skips the restore): Atk stays −2 + the item retained.
/// Ground truth `harness/probe_batch89_whiteherb_regression_rng.js`.
#[test]
fn white_herb_restores_a_foe_charm_stat_drop() {
    let d = dex();
    let wh = "Snorlax||whiteherb|Own Tempo|bodyslam,splash|Adamant|252,252,4,,,|N||||";
    let foe = "Alakazam|||Own Tempo|charm,splash|Timid|4,,,,,252|N||||";
    let seed = "11,29,37,53";
    let mut bo = Battle::start_with_switchins(&opts_cg(wh, foe, seed), &d).expect("start");
    let o = bo.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(1), Choice::Move(0))], // p1 splash, p2 Charm
        &d,
    );
    assert_eq!(o.decisions[0].active[0].boosts, [0i8; 7], "WH2: the Charm −2 Atk is RESTORED to 0");
    assert!(!o.decisions[0].active[0].item_held, "WH2: White Herb was CONSUMED");
    assert_eq!(
        seed_str(&o.decisions[0].seed_after),
        "16359,1042,61688,16031",
        "WH2: the restore is DRAW-FREE — dec0 post-turn seed"
    );
}

/// WH3 WHITE HERB net-non-negative NO-trigger (`gen3_white_herb_v1`) — a White Herb Snorlax Swords
/// Dances (+2 Atk, a self-RAISE → no trigger), then a foe Charm (−2 Atk) brings it to NET 0: with
/// NO negative stage, White Herb does NOT trigger, the item is RETAINED. WRONG (a model that
/// triggers on any drop / on `item present`): the item is consumed at dec0/dec1. Ground truth
/// `harness/probe_batch89_whiteherb_regression_rng.js`.
#[test]
fn white_herb_does_not_trigger_when_the_net_leaves_no_negative_stage() {
    let d = dex();
    let wh = "Snorlax||whiteherb|Own Tempo|swordsdance,bodyslam,splash|Adamant|252,252,4,,,|N||||";
    let foe = "Alakazam|||Own Tempo|charm,splash|Timid|4,,,,,252|N||||";
    let seed = "11,29,37,53";
    let mut bo = Battle::start_with_switchins(&opts_cg(wh, foe, seed), &d).expect("start");
    let o = bo.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // p1 SD +2, p2 splash
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // p1 splash, p2 Charm −2 → net 0
        ],
        &d,
    );
    // dec0: SD +2 (a self-RAISE — no negative → no trigger), item RETAINED.
    assert_eq!(o.decisions[0].active[0].boosts[0], 2, "WH3: Swords Dance +2 Atk");
    assert!(o.decisions[0].active[0].item_held, "WH3: a self-RAISE does NOT trigger White Herb (retained)");
    // dec1: Charm −2 → net 0 (NO negative stage) → NO trigger, item STILL RETAINED.
    assert_eq!(o.decisions[1].active[0].boosts[0], 0, "WH3: +2 then −2 = net 0");
    assert!(
        o.decisions[1].active[0].item_held,
        "WH3: net 0 leaves NO negative stage → White Herb must NOT trigger (item retained)"
    );
}

/// WH4 WHITE HERB LEAD-Intimidate switch-in restore (`gen3_white_herb_v1`, onAnySwitchIn) — a
/// White Herb Snorlax whose Atk was dropped −1 by the opposing lead's Intimidate restores it to 0
/// + consumes the item DURING CONSTRUCTION (`start_with_switchins`), so dec0 already reads Atk 0 +
/// item gone. WRONG (a model missing the construction restore): dec0 shows Atk −1 + item retained.
/// Ground truth `harness/probe_batch89_whiteherb_regression_rng.js`.
#[test]
fn white_herb_restores_a_lead_intimidate_drop_during_construction() {
    let d = dex();
    let mence = "Salamence|||Intimidate|earthquake,splash|Adamant|4,252,,,,252|N||||";
    let wh = "Snorlax||whiteherb|Own Tempo|splash|Careful|252,,252,,4,|N||||";
    let seed = "11,29,37,53";
    let mut bo = Battle::start_with_switchins(&opts_cg(mence, wh, seed), &d).expect("start");
    // The construction restore already fired: the WH holder (p2) reads Atk 0 + no item.
    let st = bo.state().unwrap();
    let p2 = &st.sides[1].pokemon[st.sides[1].active];
    assert_eq!(p2.boosts[0], 0, "WH4: the lead Intimidate −1 Atk was RESTORED to 0 (construction)");
    assert!(pokesim::dex::to_id(&p2.item).is_empty(), "WH4: White Herb was CONSUMED (construction)");
    let o = bo.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(o.decisions[0].active[1].boosts[0], 0, "WH4: dec0 p2 Atk stays 0 (restored)");
    assert!(!o.decisions[0].active[1].item_held, "WH4: dec0 p2 no longer holds White Herb");
    assert_eq!(
        seed_str(&o.decisions[0].seed_after),
        "6904,16547,7839,4217",
        "WH4: the construction restore is DRAW-FREE — dec0 post-turn seed"
    );
}

/// WH5 WHITE HERB MID-BATTLE Intimidate switch-in restore (`gen3_white_herb_v1`, onAnySwitchIn) —
/// p1 switches to a Salamence whose Intimidate drops the White Herb Snorlax's Atk −1 → restored to
/// 0 + consumed via `run_switch`. WRONG (a model missing the run_switch check): p2 shows Atk −1 +
/// item retained after the switch. Ground truth `harness/probe_batch89_whiteherb_regression_rng.js`.
#[test]
fn white_herb_restores_a_mid_battle_intimidate_drop_via_run_switch() {
    let d = dex();
    let p1 = "Snorlax|||Own Tempo|splash|Adamant|252,,252,,4,|N||||\
              ]Salamence|||Intimidate|earthquake,splash|Adamant|4,252,,,,252|N||||";
    let wh = "Snorlax||whiteherb|Own Tempo|splash|Careful|252,,252,,4,|N||||";
    let seed = "11,29,37,53";
    let mut bo = Battle::start_with_switchins(&opts_cg(p1, wh, seed), &d).expect("start");
    let o = bo.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Switch(1), Choice::Move(0))], // p1 -> Salamence (Intimidate), p2 splash
        &d,
    );
    assert_eq!(o.decisions[0].active[1].boosts[0], 0, "WH5: the mid-battle Intimidate −1 Atk was RESTORED to 0");
    assert!(!o.decisions[0].active[1].item_held, "WH5: White Herb was CONSUMED (run_switch path)");
    assert_eq!(
        seed_str(&o.decisions[0].seed_after),
        "5913,6493,46579,27262",
        "WH5: the run_switch restore is DRAW-FREE — dec0 post-turn seed"
    );
}

/// TRACE-LO the batch-8/9 `TRACE_COPYABLE` lockstep gap (`gen3_liquid_ooze_v1`) — Trace copying a
/// NEWLY-admitted MODELED ability must NOT fail-loud panic. Batch 8/9 admitted **Liquid Ooze** (+
/// Wonder Guard) to `MODELED_ABILITIES` but did NOT sync the `event.rs::TRACE_COPYABLE` set, so a
/// Porygon2 that Traced a Liquid Ooze mon PANICKED ("Trace copied the UNMODELED ability liquidooze"
/// — a `--use-bridge=rust` crash, repro rmry3vbgm_ab_1_15). With Liquid Ooze in the set the copy
/// succeeds AND — species-agnostically (Liquid Ooze keys on the mon's CURRENT ability) — REVERSES a
/// drain: p1 Tentacruel Giga Drains the Porygon2 it just gave Liquid Ooze to, so the DRAINER
/// (Tentacruel) TAKES the would-be heal as damage instead of healing. REVERT-VERIFIED: without the
/// `TRACE_COPYABLE` sync the test PANICS at `Battle::start_with_switchins` (the Trace copy).
#[test]
fn trace_of_liquid_ooze_does_not_panic_and_reverses_the_drain() {
    let d = dex();
    // p1 lead = a Liquid Ooze drainer (Tentacruel); p2 = Porygon2 (Trace) copies its Liquid Ooze.
    let tenta = "Tentacruel|||Liquid Ooze|gigadrain,splash|Bold|252,,,,252,|N||||";
    let p2gon = "Porygon2|||Trace|recover,splash|Bold|252,252,4,,,|N||||";
    let seed = "11,23,37,41";
    let mut bo = Battle::start_with_switchins(&opts_cg(tenta, p2gon, seed), &d).expect("start");
    // The Trace copy ALREADY fired at construction (reaching here at all is the no-panic proof).
    {
        let st = bo.state().unwrap();
        assert_eq!(
            pokesim::dex::to_id(&st.sides[1].pokemon[st.sides[1].active].ability),
            "liquidooze",
            "TRACE-LO: Porygon2 Traced the foe's Liquid Ooze (no fail-loud panic)"
        );
    }
    // p1 Tentacruel Giga Drains (move 0) the now-Liquid-Ooze Porygon2 (move 1 = splash) → the drain
    // heal REVERSES onto the drainer: Tentacruel TAKES damage (HP < maxhp) instead of healing.
    let o = bo.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(1))],
        &d,
    );
    let (thp, tmax) = (o.decisions[0].active[0].hp, o.decisions[0].active[0].maxhp);
    assert!(thp < tmax, "TRACE-LO: the drainer took the Liquid Ooze reversal (HP {thp} < {tmax})");
    assert!(
        o.decisions[0].active[1].hp < o.decisions[0].active[1].maxhp,
        "TRACE-LO: Porygon2 took the Giga Drain damage"
    );
}

/// TRACE-WG the batch-8/9 `TRACE_COPYABLE` lockstep gap (`gen3_wonder_guard_v1`) — the Wonder Guard
/// half of the same sync: a Porygon2 that Traces a Wonder Guard mon must NOT panic, and the copied
/// gate is SPECIES-AGNOSTIC (it keys on the holder's CURRENT ability, not Shedinja). p1 Shedinja
/// (Wonder Guard) Tackles the Porygon2 that Traced its Wonder Guard: Normal-vs-Normal is NEUTRAL, so
/// the traced Wonder Guard BLOCKS it → Porygon2 stays at full HP. REVERT-VERIFIED: without the
/// `TRACE_COPYABLE` sync the test PANICS at construction (the Trace copy of `wonderguard`).
#[test]
fn trace_of_wonder_guard_does_not_panic_and_blocks_a_neutral_move() {
    let d = dex();
    let shed = "Shedinja|||Wonder Guard|tackle,splash|Serious|,,,,,|N||||";
    let p2gon = "Porygon2|||Trace|splash,recover|Bold|252,252,4,,,|N||||";
    let seed = "13,29,41,53";
    let mut bo = Battle::start_with_switchins(&opts_cg(shed, p2gon, seed), &d).expect("start");
    {
        let st = bo.state().unwrap();
        assert_eq!(
            pokesim::dex::to_id(&st.sides[1].pokemon[st.sides[1].active].ability),
            "wonderguard",
            "TRACE-WG: Porygon2 Traced the foe's Wonder Guard (no fail-loud panic)"
        );
    }
    // p1 Shedinja Tackles (move 0, Normal → NEUTRAL vs Porygon2/Normal) the Wonder Guard Porygon2
    // (move 1 = recover) → the traced Wonder Guard BLOCKS the neutral move → Porygon2 stays full.
    let o = bo.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(1))],
        &d,
    );
    assert_eq!(
        o.decisions[0].active[1].hp, o.decisions[0].active[1].maxhp,
        "TRACE-WG: the traced Wonder Guard blocked the neutral Tackle (Porygon2 at full HP)"
    );
}

/// WG1 WONDER GUARD neutral block (`gen3_wonder_guard_v1`) — a NEUTRAL damaging move (Water Gun,
/// Water 1× vs Bug/Ghost) into a Wonder Guard Shedinja is BLOCKED (`runEffectiveness <= 0`): Shedinja
/// stays 1/1 and the block draws ONLY its accuracy roll (`-immune [from] ability: Wonder Guard`). A
/// COMPOUND-EYES (no-WG) Shedinja on the SAME board HITS + KOs (a DIFFERENT seed — the hit's
/// crit+damage draws). WRONG (a model without the gate): Water Gun connects → the WG Shedinja dies
/// (HP 0) at a control-matching seed. Ground truth `harness/probe_batch89_wonderguard_regression_rng.js`.
#[test]
fn wonder_guard_blocks_a_neutral_move_drawing_only_accuracy() {
    let d = dex();
    let chari = "Charizard|||Blaze|watergun,magicalleaf,ember,bodyslam|Serious|,,,,,|N||||";
    let shed_wg = "Shedinja|||Wonder Guard|splash|Serious|,,,,,|N||||";
    let shed_no = "Shedinja|||Compound Eyes|splash|Serious|,,,,,|N||||";
    let seed = "13,29,41,53";

    // Wonder Guard: Water Gun (move 1 = idx 0) is BLOCKED — Shedinja survives at 1 HP, draws only acc.
    let mut bw = Battle::start_with_switchins(&opts_cg(chari, shed_wg, seed), &d).expect("start");
    let ow = bw.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(ow.decisions[0].active[1].hp, 1, "WG1: Wonder Guard BLOCKS the neutral Water Gun (Shedinja stays 1/1)");
    assert!(!ow.decisions[0].active[1].fainted, "WG1: the blocked-neutral Shedinja survives");
    assert_eq!(
        seed_str(&ow.decisions[0].seed_after),
        "37625,6431,63580,16031",
        "WG1: the block draws ONLY the accuracy roll (no crit/damage/secondary)"
    );

    // Control (no WG): Water Gun HITS + KOs the 1-HP Shedinja → P1 wins, a DIFFERENT seed.
    let mut bc = Battle::start_with_switchins(&opts_cg(chari, shed_no, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert!(oc.decisions[0].active[1].fainted, "WG1: without Wonder Guard the neutral Water Gun KOs the 1-HP Shedinja");
    assert_eq!(oc.winner, Some(0), "WG1: the no-WG Shedinja fainting → P1 wins");
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        "54124,50226,9319,27262",
        "WG1: the hitting control's seed DIFFERS from the block (it drew crit+damage)"
    );
}

/// WG2 WONDER GUARD resisted NEVER-MISS block (`gen3_wonder_guard_v1`) — a RESISTED never-miss move
/// (Magical Leaf, Grass 0.5× vs Bug/Ghost, `accuracy: true`) into a Wonder Guard Shedinja is BLOCKED
/// drawing NOTHING (no accuracy roll): Shedinja stays 1/1 and the turn draws only the endTurn Quick
/// Claw. A no-WG control HITS + KOs. WRONG (a model without the gate): Magical Leaf connects → the WG
/// Shedinja dies + the seed picks up crit+damage draws. Ground truth
/// `harness/probe_batch89_wonderguard_regression_rng.js`.
#[test]
fn wonder_guard_blocks_a_resisted_never_miss_move_drawing_nothing() {
    let d = dex();
    let chari = "Charizard|||Blaze|watergun,magicalleaf,ember,bodyslam|Serious|,,,,,|N||||";
    let shed_wg = "Shedinja|||Wonder Guard|splash|Serious|,,,,,|N||||";
    let shed_no = "Shedinja|||Compound Eyes|splash|Serious|,,,,,|N||||";
    let seed = "13,29,41,53";

    // Wonder Guard: Magical Leaf (move 2 = idx 1) BLOCKED — Shedinja survives, the block draws NOTHING.
    let mut bw = Battle::start_with_switchins(&opts_cg(chari, shed_wg, seed), &d).expect("start");
    let ow = bw.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(1), Choice::Move(0))],
        &d,
    );
    assert_eq!(ow.decisions[0].active[1].hp, 1, "WG2: Wonder Guard BLOCKS the resisted never-miss Magical Leaf");
    assert!(!ow.decisions[0].active[1].fainted, "WG2: the blocked-resisted Shedinja survives");
    assert_eq!(
        seed_str(&ow.decisions[0].seed_after),
        "56718,47525,24291,4268",
        "WG2: a NEVER-MISS blocked move draws NOTHING (only the endTurn Quick Claw)"
    );

    // Control (no WG): Magical Leaf HITS + KOs.
    let mut bc = Battle::start_with_switchins(&opts_cg(chari, shed_no, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(1), Choice::Move(0))],
        &d,
    );
    assert!(oc.decisions[0].active[1].fainted, "WG2: without Wonder Guard the never-miss Magical Leaf KOs Shedinja");
    assert_eq!(oc.winner, Some(0), "WG2: the no-WG Shedinja fainting → P1 wins");
}

/// WG3 WONDER GUARD leech-residual bypass KO (`gen3_wonder_guard_v1`) — Leech Seed is a STATUS move
/// so it BYPASSES Wonder Guard and plants on the 1-HP Shedinja; the end-of-turn Leech Seed RESIDUAL
/// (a residual is NOT a MOVE hook → also bypasses WG; `this.damage(maxhp/8)` clamps to min 1) drains
/// Shedinja to 0 → it FAINTS on the cast turn → P1 wins. WRONG (the pre-fix `maxhp/8` early-return on
/// a sub-8-maxhp mon): the drain is 0 → Shedinja survives → the battle does not end (fainted false).
/// Ground truth `harness/probe_batch89_wonderguard_regression_rng.js`.
#[test]
fn wonder_guard_leech_seed_residual_kos_the_one_hp_shedinja() {
    let d = dex();
    let venu = "Venusaur|||Overgrow|leechseed,ember|Modest|252,,,252,,|N||||";
    let shed_wg = "Shedinja|||Wonder Guard|splash|Serious|,,,,,|N||||";
    let seed = "13,29,41,53";
    let mut b = Battle::start_with_switchins(&opts_cg(venu, shed_wg, seed), &d).expect("start");
    let o = b.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], // Leech Seed (bypass) + the residual KO
        &d,
    );
    assert_eq!(o.decisions.len(), 1, "WG3: the leech residual KOs Shedinja on the cast turn → the battle ends");
    assert!(o.decisions[0].active[1].fainted, "WG3: Shedinja FAINTED to the Leech Seed residual (a residual bypasses WG)");
    assert_eq!(o.decisions[0].active[1].hp, 0, "WG3: Shedinja is at 0 HP");
    assert_eq!(o.winner, Some(0), "WG3: mono Shedinja fainting to the residual → P1 wins");
    assert_eq!(
        seed_str(&o.decisions[0].seed_after),
        "56718,47525,24291,4268",
        "WG3: the leech move accuracy is the only draw (the KO defers the Quick Claw)"
    );
}

/// WG4 WONDER GUARD status bypass + SE connect (`gen3_wonder_guard_v1`) — Thunder Wave is a STATUS
/// move so it BYPASSES Wonder Guard and PARALYZES Shedinja (dec0 status → par); then Shadow Ball
/// (Ghost 2× vs Ghost — STRICTLY super-effective) CONNECTS and KOs it (WG lets an SE move through).
/// Ground truth `harness/probe_batch89_wonderguard_regression_rng.js`.
#[test]
fn wonder_guard_status_move_bypasses_then_super_effective_connects() {
    let d = dex();
    let gengar = "Gengar|||Levitate|thunderwave,shadowball|Timid|,,,252,,252|N||||";
    let shed_wg = "Shedinja|||Wonder Guard|splash|Serious|,,,,,|N||||";
    let seed = "13,29,41,53";
    let mut b = Battle::start_with_switchins(&opts_cg(gengar, shed_wg, seed), &d).expect("start");
    let o = b.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Thunder Wave → par (status bypass)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Shadow Ball SE → KO
        ],
        &d,
    );
    assert_eq!(
        o.decisions[0].active[1].status,
        Some(Status::Paralysis),
        "WG4: Thunder Wave (a Status move) BYPASSES Wonder Guard and paralyzes Shedinja"
    );
    assert!(!o.decisions[0].active[1].fainted, "WG4: Shedinja survives the status-move turn (WG doesn't block a KO — TWave deals 0)");
    assert!(o.decisions[1].active[1].fainted, "WG4: the super-effective Shadow Ball CONNECTS and KOs Shedinja");
    assert_eq!(o.winner, Some(0), "WG4: Shedinja fainting to the SE move → P1 wins");
    assert_eq!(
        seed_str(&o.decisions[1].seed_after),
        "63086,37694,32206,36914",
        "WG4: the SE-connect turn's post-decision seed"
    );
}

/// SP DAMAGING Soundproof block (`gen3_ability_batch2_v1`) — a DAMAGING `flags.sound` move
/// (Hyper Voice, bp 90) into a Soundproof holder (Mr. Mime) is IMMUNE: the target takes NO damage
/// and the block draws ONLY its accuracy roll then `-immune|<mon>|[from] ability: Soundproof`
/// (SIM-PROBED bit-for-bit, `harness/probe_soundproof_regression_rng.js`: Hyper Voice into a
/// Soundproof Mr. Mime draws `random(100)`(accuracy) then `-immune`, NO crit / damage roll — 4
/// draws vs the 7-draw non-Soundproof control). This is the DAMAGING-path MIRROR of the STATUS-move
/// Soundproof gate (Sing / Grass Whistle) in `run_status_move`; the `move_is_sound` helper existed
/// but was UNUSED on `run_move`'s damaging path, so the port emitted `-damage`. A no-Soundproof
/// (Own Tempo) Mr. Mime on the SAME board TAKES damage (68/221) at a DIFFERENT seed (the hit's
/// crit+damage draws). WRONG (a model without the gate): Hyper Voice connects → damage.
#[test]
fn soundproof_blocks_a_damaging_sound_move() {
    let d = dex();
    let snorlax = "Snorlax|||Immunity|hypervoice,bodyslam|Serious|,,,,,|N||||";
    let mime_sp = "Mr. Mime|||Soundproof|calmmind,tackle|Serious|,,,,,|N||||";
    let mime_no = "Mr. Mime|||Own Tempo|calmmind,tackle|Serious|,,,,,|N||||";
    let seed = "13,29,41,53";

    // Soundproof: Hyper Voice (move 1 = idx 0) is BLOCKED — Mr. Mime takes NO damage, draws only acc.
    let mut bw = Battle::start_with_switchins(&opts_cg(snorlax, mime_sp, seed), &d).expect("start");
    let ow = bw.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        ow.decisions[0].active[1].hp, 221,
        "SP: Soundproof BLOCKS the damaging Hyper Voice (Mr. Mime stays 221/221)"
    );
    assert!(!ow.decisions[0].active[1].fainted, "SP: the blocked Mr. Mime is untouched");
    assert_eq!(
        seed_str(&ow.decisions[0].seed_after),
        "37625,6431,63580,16031",
        "SP: the block draws ONLY the accuracy roll (no crit/damage/secondary)"
    );

    // Control (no Soundproof): Hyper Voice HITS + damages → 68/221, a DIFFERENT seed (crit+damage drew).
    let mut bc = Battle::start_with_switchins(&opts_cg(snorlax, mime_no, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        oc.decisions[0].active[1].hp, 68,
        "SP: without Soundproof the Hyper Voice HITS (Mr. Mime 68/221)"
    );
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        "2194,19130,50787,4217",
        "SP: the hitting control's seed DIFFERS from the block (it drew crit+damage)"
    );
}

/// B2 WEATHER_SPEED: a slow Chlorophyll Bellossom (spe 136) FLIPS the first mover under sun (its
/// ×2 → 272 > Groudon's 216). WRONG (a model that ignores the ×2): Groudon (216) moves first, and
/// the tie-shuffle / action-order draws desync the seed. Groudon's own Drought sets the sun.
/// Control: a no-op ability (Insomnia) keeps Bellossom at 136 → Groudon first (no flip).
#[test]
fn chlorophyll_speed_doubles_and_flips_the_first_mover_in_sun() {
    let d = dex();
    let groudon = "Groudon|||Drought|earthquake,rest|Serious||||||";
    let seed = "23145,51002,8890,44120";

    // CHLOROPHYLL: Bellossom moves FIRST (×2 = 272 > 216).
    let chloro = "Bellossom|||Chlorophyll|razorleaf,rest|Serious||||||";
    let mut bch = Battle::start_with_switchins(&opts_cg(chloro, groudon, seed), &d).expect("start");
    let och = bch.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(bch.state().unwrap().field.weather, Some(Weather::Sun), "Drought set sun");
    assert_eq!(
        och.decisions[0].first_mover,
        Some(0),
        "B2: the Chlorophyll ×2 (272) makes Bellossom OUTSPEED Groudon (216) in sun → p1 first"
    );
    assert_eq!(
        seed_str(&och.decisions[0].seed_after),
        "31591,44253,33227,28985",
        "B2: post-turn seed with the ×2 speed order"
    );

    // CONTROL: Insomnia Bellossom stays 136 < 216 → Groudon first (no flip).
    let ctl = "Bellossom|||Insomnia|razorleaf,rest|Serious||||||";
    let mut bct = Battle::start_with_switchins(&opts_cg(ctl, groudon, seed), &d).expect("start");
    let oct = bct.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        oct.decisions[0].first_mover,
        Some(1),
        "B2 control: WITHOUT Chlorophyll, Bellossom (136) is slower than Groudon (216) → p2 first"
    );
}

/// B3 WEATHER_NEGATE: a Cloud Nine Psyduck (Water — normally not sand-immune, but non-Rock/
/// Ground/Steel) takes NO sand chip while a Sand Stream Tyranitar keeps sand up — the negater
/// SUPPRESSES the weather's effects (`effectiveWeather()` = ''). WRONG (a model that still chips):
/// Psyduck loses maxhp/16 to sand. Control: a no-op ability (Damp) Psyduck TAKES the sand chip.
#[test]
fn cloud_nine_suppresses_the_sandstorm_chip() {
    let d = dex();
    let ttar = "Tyranitar|||SandStream|rockslide,rest|Careful|252,,,,252,|N||||";
    let seed = "40012,7781,55230,19004";

    // CLOUD NINE: Psyduck takes NO sand chip (stays full HP after only the foe's move — but here
    // Tyranitar's Rock Slide misses/does little; the KEY is Psyduck's HP is NOT further reduced by
    // sand). We assert Psyduck's HP == the sim's (which reflects NO sand chip).
    let cn = "Psyduck|||CloudNine|surf,rest|Bold|252,,252,,,|N||||";
    let mut bcn = Battle::start_with_switchins(&opts_cg(cn, ttar, seed), &d).expect("start");
    let ocn = bcn.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let cn_hp = bcn.state().unwrap().sides[0].pokemon[0].hp;
    assert_eq!(cn_hp, 304, "Cloud Nine Psyduck took NO sand chip (full 304 — the negater suppressed it)");
    // The RAW weather still persists (only its effects die).
    assert_eq!(bcn.state().unwrap().field.weather, Some(Weather::Sand), "raw sand persists under a negater");
    assert_eq!(
        seed_str(&ocn.decisions[0].seed_after),
        "55553,32658,31456,54547",
        "B3: post-turn seed with the sand chip suppressed"
    );

    // CONTROL: Damp Psyduck (a no-op weather-wise) TAKES the sand chip (304 → 285, maxhp/16 = 19).
    let ctl = "Psyduck|||Damp|surf,rest|Bold|252,,252,,,|N||||";
    let mut bct = Battle::start_with_switchins(&opts_cg(ctl, ttar, seed), &d).expect("start");
    let _ = bct.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let ctl_hp = bct.state().unwrap().sides[0].pokemon[0].hp;
    assert_eq!(ctl_hp, 285, "the Damp control TOOK the sand chip (304 → 285 = maxhp/16) — proving the chip fires without a negater");
    assert!(cn_hp > ctl_hp, "Cloud Nine suppressed the chip → MORE HP than the control");
}

/// B4 RESIDUAL: Speed Boost's +1 spe residual (residualOrder 10, subOrder 3, DRAW-FREE) raises
/// Ninjask's spe stage by 1 at the FIRST end-of-turn (its entry turn is activeTurns-gated but the
/// leads are placed at construction, so turn 1 IS an active turn → +1). WRONG (a wrong gate /
/// timing / a draw): the spe stage is off, or the seed desyncs. Ninjask (already the fastest) vs a
/// bulky Snorlax.
#[test]
fn speed_boost_raises_the_spe_stage_by_one_each_active_turn() {
    let d = dex();
    let ninjask = "Ninjask|||SpeedBoost|aerialace,rest|Jolly|252,252,,,,|N||||";
    let snorlax = "Snorlax|||Immunity|bodyslam,rest|Impish|252,,252,,,|N||||";
    let mut b =
        Battle::start_with_switchins(&opts_cg(ninjask, snorlax, "11002,62210,3345,28890"), &d).expect("start");
    let out = b.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    // After the first end-of-turn, Ninjask's spe stage is +1 (Speed Boost fired).
    assert_eq!(
        b.state().unwrap().sides[0].pokemon[0].boosts[4],
        1,
        "B4: Speed Boost raised Ninjask's spe stage to +1 at the end-of-turn residual"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "7159,28120,61544,4947",
        "B4: Speed Boost is DRAW-FREE — post-turn seed unchanged by the boost"
    );
}

/// B4b RESIDUAL: Rain Dish's +maxhp/16 heal (residualOrder 10, subOrder 3, DRAW-FREE) heals
/// Ludicolo each end-of-turn in rain (set by the foe Kyogre's Drizzle). WRONG (no heal / a draw):
/// Ludicolo's HP is off, or the seed desyncs. The exact post-turn HP (which INCLUDES the heal) is
/// the pin.
#[test]
fn rain_dish_heals_each_end_of_turn_in_rain() {
    let d = dex();
    let ludicolo = "Ludicolo|||RainDish|surf,rest|Calm|252,,,,252,|N||||";
    let kyogre = "Kyogre|||Drizzle|icebeam,rest|Modest|,,,4,,|N||||";
    let mut b =
        Battle::start_with_switchins(&opts_cg(ludicolo, kyogre, "50501,9987,44012,60123"), &d).expect("start");
    assert_eq!(b.state().unwrap().field.weather, Some(Weather::Rain), "Drizzle set rain");
    let out = b.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    // Ludicolo took Kyogre's Ice Beam (SE on Grass) then Rain Dish healed maxhp/16 — the net HP is
    // the sim's exact value (proving the heal composed with the damage in the right order).
    assert_eq!(
        b.state().unwrap().sides[0].pokemon[0].hp,
        306,
        "B4b: Ludicolo's post-turn HP == the sim's (Ice Beam damage then the Rain Dish +maxhp/16 heal)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "2775,51419,6771,30819",
        "B4b: Rain Dish is DRAW-FREE — post-turn seed unchanged by the heal"
    );
}

// ─── BATCH-2 ability-class pins (`gen3_ability_batch2_v1`) ──────────────────────────────────
// The DRAW-BEARING "reactive" classes + the block tail. Each a CONSTRUCTED gen3customgame board
// reseeded to a RAW seed (so the port's draw-free `start_with_switchins` aligns), revert-verified
// (each FAILS when its class's engine wiring is disabled). Ground truth:
// `harness/probe_ability_batch2_regression_rng.js`.

/// B2-1 CONTACT_PROC Static: Body Slam (contact) into a Static holder PARALYZES THE ATTACKER. The
/// proc's `randomChance(1,3)` draws INSIDE runEvent('DamagingHit') (AFTER the move's own secondary
/// random(100)). On the chosen seed it PASSES → Snorlax is para'd. WRONG (no proc): Snorlax is
/// un-statused AND the contact-proc draw is absent → the seed diverges. Control: a no-op ability
/// (Insomnia) → no proc, un-statused, a DIFFERENT seed (the proc draw shifts the stream).
#[test]
fn static_contact_proc_paralyzes_the_attacker() {
    let d = dex();
    let attacker = "Snorlax|||NoAbility|bodyslam,earthquake|Adamant|252,252,,,,|N||||";
    let seed = "49115,62334,33989,12128";

    // STATIC: Body Slam into the Static holder paras the ATTACKER (Snorlax).
    let holder = "Electabuzz|||Static|thunderbolt,thunderbolt|Modest|,,,,252,252|N||||";
    let mut bs = Battle::start_with_switchins(&opts_cg(attacker, holder, seed), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        bs.state().unwrap().sides[0].pokemon[0].status,
        Some(Status::Paralysis),
        "B2-1: Static paralyzes the ATTACKER (Snorlax) on the contact Body Slam"
    );
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "59434,45884,11610,28346",
        "B2-1: the contact-proc randomChance draws AFTER the move secondary — post-turn seed"
    );

    // CONTROL: Insomnia holder — NO proc, the attacker stays un-statused, the seed DIFFERS (the
    // absent contact-proc draw shifts the stream) — proving the proc is what statuses + draws.
    let ctl = "Electabuzz|||Insomnia|thunderbolt,thunderbolt|Modest|,,,,252,252|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(attacker, ctl, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        bc.state().unwrap().sides[0].pokemon[0].status, None,
        "B2-1 control: no contact-proc ability → the attacker stays un-statused"
    );
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        "18037,43851,24339,3115",
        "B2-1 control: the seed DIFFERS from Static's (the proc's randomChance is a real draw)"
    );
    assert_ne!(
        seed_str(&os.decisions[0].seed_after),
        seed_str(&oc.decisions[0].seed_after),
        "B2-1: the proc draw makes Static's post-turn seed differ from the no-op control's"
    );
}

/// B2-2 Effect Spore: Body Slam into an Effect Spore holder draws `randomChance(1,10)` then, on a
/// pass, a `sample(["slp","par","psn"])` (one `random(3)`) → statuses the ATTACKER. On the chosen
/// seed the gate passes + the sample picks par. WRONG (no nested sample / wrong draw count): the
/// seed diverges. This pins the NESTED draw (the sample is the crux vs a flat 3-status split).
#[test]
fn effect_spore_samples_a_status_onto_the_attacker() {
    let d = dex();
    let attacker = "Snorlax|||NoAbility|bodyslam,earthquake|Adamant|252,252,,,,|N||||";
    let holder = "Vileplume|||EffectSpore|sludgebomb,sludgebomb|Modest|,,,,252,252|N||||";
    let seed = "6767,34050,32889,44932";
    let mut b = Battle::start_with_switchins(&opts_cg(attacker, holder, seed), &d).expect("start");
    let out = b.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        b.state().unwrap().sides[0].pokemon[0].status,
        Some(Status::Paralysis),
        "B2-2: Effect Spore's random(10)+sample(3) picks par onto the ATTACKER on this seed"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "31439,46507,40086,46270",
        "B2-2: the random(10) gate + the random(3) sample draw in the exact place/count — post-turn seed"
    );
}

/// B2-3 Rough Skin: Body Slam into a Rough Skin holder deals `baseMaxhp/16` recoil to the ATTACKER
/// — DRAW-FREE. WRONG (no recoil): the attacker keeps its HP. The CONTROL (a no-op ability, SAME
/// seed) takes NO recoil AND the seed is IDENTICAL (the recoil consumes no PRNG).
#[test]
fn rough_skin_recoils_the_attacker_draw_free() {
    let d = dex();
    let attacker = "Snorlax|||NoAbility|bodyslam,earthquake|Adamant|252,252,,,,|N||||";
    let seed = "40012,7781,55230,19004";

    // ROUGH SKIN: the attacker loses maxhp/16 (524/16 = 32) → 424 - 32 = 392.
    let holder = "Sharpedo|||RoughSkin|surf,surf|Modest|,,,,252,252|N||||";
    let mut bs = Battle::start_with_switchins(&opts_cg(attacker, holder, seed), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let rs_hp = bs.state().unwrap().sides[0].pokemon[0].hp;
    assert_eq!(rs_hp, 392, "B2-3: Rough Skin dealt maxhp/16 = 32 recoil to the attacker (424 → 392)");

    // CONTROL: Insomnia holder — NO recoil (424), the seed IDENTICAL (Rough Skin is draw-free).
    let ctl = "Sharpedo|||Insomnia|surf,surf|Modest|,,,,252,252|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(attacker, ctl, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let c_hp = bc.state().unwrap().sides[0].pokemon[0].hp;
    assert_eq!(c_hp, 424, "B2-3 control: no Rough Skin → the attacker keeps its HP (424)");
    assert!(rs_hp < c_hp, "B2-3: Rough Skin's recoil made the attacker LOSE HP vs the control");
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        seed_str(&oc.decisions[0].seed_after),
        "B2-3: Rough Skin is DRAW-FREE — its post-turn seed is IDENTICAL to the no-op control's"
    );
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "51102,626,46934,3277",
        "B2-3: the post-turn seed (Rough Skin draw-free)"
    );
}

/// B2-4 Damp: p1 Snorlax uses Explosion while p2 Golduck (Damp) is active → CANCELLED at TryMove.
/// The user does NOT self-KO (Snorlax survives), and the move draws NOTHING (a big draw-count drop
/// vs a normal Explosion's acc+crit+dmg). WRONG (no Damp): Snorlax self-KOs (both faint). Control:
/// a no-op Golduck → Explosion self-KOs Snorlax AND KOs Golduck (a different seed — Explosion drew).
#[test]
fn damp_cancels_explosion_no_self_ko() {
    let d = dex();
    let boomer = "Snorlax|||NoAbility|explosion,bodyslam|Adamant|,252,,,,|N||||";
    let seed = "12345,54321,11111,22222";

    // DAMP: Explosion cancelled — Snorlax does NOT self-KO (still has HP), Golduck untouched.
    let holder = "Golduck|||Damp|surf,surf|Modest|,,,,252,252|N||||";
    let mut bs = Battle::start_with_switchins(&opts_cg(boomer, holder, seed), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert!(
        !bs.state().unwrap().sides[0].pokemon[0].fainted && bs.state().unwrap().sides[0].pokemon[0].hp > 0,
        "B2-4: Damp CANCELS the Explosion — Snorlax does NOT self-KO"
    );
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "51139,52449,26720,55170",
        "B2-4: a Damp-blocked Explosion draws NOTHING (only Quick Claw) — post-turn seed"
    );

    // CONTROL: a no-op Golduck → the Explosion self-KOs Snorlax (fainted). The seed DIFFERS
    // (Explosion drew acc+crit+dmg), proving Damp is what blocks + drops the draws.
    let ctl = "Golduck|||Insomnia|surf,surf|Modest|,,,,252,252|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(boomer, ctl, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert!(
        bc.state().unwrap().sides[0].pokemon[0].fainted,
        "B2-4 control: a no-op foe → the Explosion self-KOs Snorlax"
    );
    assert_ne!(
        seed_str(&os.decisions[0].seed_after),
        seed_str(&oc.decisions[0].seed_after),
        "B2-4: the Damp block drops the Explosion's acc/crit/dmg draws → its seed differs from the control's"
    );
}

/// B2-5 Soundproof: p1 Jynx Sings (a SOUND move) into p2 Electrode (Soundproof) → IMMUNE (accuracy
/// drawn, then -immune, NO sleep). WRONG (no Soundproof): Electrode is put to sleep. Control: a
/// SLEEP-allowing no-op (Static) at a Sing-lands seed → Electrode sleeps (proving Soundproof blocks).
#[test]
fn soundproof_immune_to_sing() {
    let d = dex();
    let singer = "Jynx|||NoAbility|sing,icebeam|Modest|,,252,,,|N||||";

    // SOUNDPROOF: Sing is immune → Electrode NOT asleep.
    let holder = "Electrode|||Soundproof|thunderbolt,thunderbolt|Timid|,,,,252,252|N||||";
    let mut bs = Battle::start_with_switchins(&opts_cg(singer, holder, "30982,33910,19571,50263"), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        bs.state().unwrap().sides[1].pokemon[0].status, None,
        "B2-5: Soundproof is IMMUNE to Sing (a sound move) → Electrode stays un-statused"
    );
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "62891,40560,22227,62965",
        "B2-5: Sing into Soundproof draws its accuracy then -immune (no sleep random(2,6)) — post-turn seed"
    );

    // CONTROL: a Static Electrode (sleep-allowing) at a Sing-LANDS seed → Electrode sleeps, proving
    // Soundproof is the thing that blocks it (Static is a sleep-allowing no-op vs Sing).
    let ctl = "Electrode|||Static|thunderbolt,thunderbolt|Timid|,,,,252,252|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(singer, ctl, "50363,29406,15525,50624"), &d).expect("start");
    let _ = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert!(
        matches!(bc.state().unwrap().sides[1].pokemon[0].status, Some(Status::Sleep(_))),
        "B2-5 control: a sleep-allowing (Static) Electrode is put to SLEEP by Sing at the land seed"
    );
    // And Soundproof at the SAME land seed is STILL immune (no sleep).
    let mut bsl = Battle::start_with_switchins(&opts_cg(singer, holder, "50363,29406,15525,50624"), &d).expect("start");
    let _ = bsl.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        bsl.state().unwrap().sides[1].pokemon[0].status, None,
        "B2-5: Soundproof at the Sing-LANDS seed is STILL immune (the ability, not luck, blocks it)"
    );
}

/// B2-6 Suction Cups: p2 Suicune Roars (priority −6) into p1 Cradily (Suction Cups) with a bench →
/// the drag is BLOCKED (Cradily STAYS active), drawing NO `sample` (only the phaze's accuracy).
/// WRONG (no Suction Cups): Cradily is dragged out (the bench Snorlax becomes active) + a sample
/// draws. Control: a no-op Cradily is DRAGGED (Snorlax active, a DIFFERENT seed — the sample drew).
#[test]
fn suction_cups_blocks_the_roar_drag_no_sample() {
    let d = dex();
    let bench = "Snorlax|||NoAbility|bodyslam,rest|Impish|252,,252,,,|N||||";
    let roarer = "Suicune|||NoAbility|roar,surf|Bold|252,,252,,,|N||||";
    let seed = "13127,45333,18295,15391";

    // SUCTION CUPS: Cradily STAYS active (not dragged); no sample drawn.
    let holder = format!("Cradily|||SuctionCups|surf,rest|Bold|252,,252,,,|N||||]{bench}");
    let mut bs = Battle::start_with_switchins(&opts_cg(&holder, roarer, seed), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        os.decisions[0].active_species[0].to_lowercase().replace(' ', ""),
        "cradily",
        "B2-6: Suction Cups BLOCKS the Roar drag — Cradily STAYS active"
    );
    assert!(
        !os.decisions[0].phaze_drag,
        "B2-6: a Suction-Cups-blocked phaze sets NO phaze_drag (no drag happened)"
    );
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "55318,8071,46680,56242",
        "B2-6: the blocked phaze draws its accuracy then -activate — NO `sample` — post-turn seed"
    );

    // CONTROL: an Insomnia Cradily is DRAGGED (the bench Snorlax becomes active), a DIFFERENT seed
    // (the `sample` drew) — proving Suction Cups is what blocks the drag + suppresses the sample.
    let ctl = format!("Cradily|||Insomnia|surf,rest|Bold|252,,252,,,|N||||]{bench}");
    let mut bc = Battle::start_with_switchins(&opts_cg(&ctl, roarer, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        oc.decisions[0].active_species[0].to_lowercase().replace(' ', ""),
        "snorlax",
        "B2-6 control: a no-op Cradily is DRAGGED → the bench Snorlax is now active"
    );
    assert_ne!(
        seed_str(&os.decisions[0].seed_after),
        seed_str(&oc.decisions[0].seed_after),
        "B2-6: Suction Cups suppresses the drag `sample` → its seed differs from the dragged control's"
    );
}

/// B2-7 Synchronize: p1 Jolteon Thunder Waves p2 Alakazam (Synchronize) → Alakazam is para'd AND
/// Jolteon (the SOURCE) is para'd too (reflected). DRAW-FREE in gen3customgame. WRONG (no
/// Synchronize): only Alakazam is para'd. Control: a no-op Alakazam → Jolteon un-statused, SAME
/// seed (the reflect is draw-free in customgame).
#[test]
fn synchronize_reflects_paralysis_to_the_caster() {
    let d = dex();
    let caster = "Jolteon|||NoAbility|thunderwave,thunderbolt|Timid|,,,,252,252|N||||";
    let seed = "42782,54377,52057,58231";

    // SYNCHRONIZE: both Alakazam AND Jolteon (the caster) are paralyzed.
    let holder = "Alakazam|||Synchronize|psychic,recover|Timid|,,252,,252,|N||||";
    let mut bs = Battle::start_with_switchins(&opts_cg(caster, holder, seed), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        bs.state().unwrap().sides[1].pokemon[0].status, Some(Status::Paralysis),
        "B2-7: Thunder Wave paralyzes the Synchronize holder (Alakazam)"
    );
    assert_eq!(
        bs.state().unwrap().sides[0].pokemon[0].status, Some(Status::Paralysis),
        "B2-7: Synchronize REFLECTS the paralysis back to the caster (Jolteon)"
    );
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "29377,55311,1818,4620",
        "B2-7: the reflect is DRAW-FREE in gen3customgame — post-turn seed"
    );

    // CONTROL: a no-op Alakazam → only Alakazam is para'd; Jolteon un-statused. The seed is
    // IDENTICAL (Synchronize's reflect is draw-free in customgame — a pure STATE effect).
    let ctl = "Alakazam|||Insomnia|psychic,recover|Timid|,,252,,252,|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(caster, ctl, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        bc.state().unwrap().sides[0].pokemon[0].status, None,
        "B2-7 control: no Synchronize → the caster (Jolteon) stays un-statused"
    );
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        seed_str(&oc.decisions[0].seed_after),
        "B2-7: Synchronize's reflect is DRAW-FREE in customgame — the seed is IDENTICAL to the control's"
    );
}

/// B2-8 CONTACT_PROC behind a SUBSTITUTE (the adversarial-review-caught bug): a Static holder subs,
/// then a WEAK contact hit (Chansey Tackle) is ABSORBED by the SURVIVING sub. The proc's
/// `onDamagingHit` is on the MON, not the sub, so it does NOT fire — the ATTACKER (Chansey) stays
/// UN-statused and the turn is DRAW-FREE (no contact-proc `randomChance`). WRONG (pre-fix, gate
/// `is_contact && dealt > 0` with no `!absorbed`): the port fired the proc behind the sub, drawing a
/// phantom `randomChance(1,3)` (shifting the seed, and paralyzing Chansey when it passed). The
/// `!absorbed` gate — the SAME one the fire-thaw uses — fixes it. Ground truth
/// `harness/probe_ability_batch2_regression_rng.js` (B2-8).
#[test]
fn contact_proc_does_not_fire_behind_a_surviving_substitute() {
    let d = dex();
    // p1 = a very weak contact attacker (Chansey Tackle won't break the sub); p2 = a fast Static
    // holder that Substitutes first (so the Tackle is absorbed by the SURVIVING sub).
    let attacker = "Chansey|||NoAbility|tackle,softboiled|Bold|252,,252,,,|N||||";
    let holder = "Electabuzz|||Static|substitute,thunderbolt|Timid|,,,,252,252|N||||";
    let seed = "9137,21044,5510,43902";
    let mut b = Battle::start_with_switchins(&opts_cg(attacker, holder, seed), &d).expect("start");
    let os = b.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        b.state().unwrap().sides[0].pokemon[0].status, None,
        "B2-8: Static does NOT proc behind a surviving sub — the attacker (Chansey) stays un-statused"
    );
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "39376,31046,45923,49458",
        "B2-8: no contact-proc randomChance draws behind a sub — the turn is DRAW-FREE (revert of the \
         `!absorbed` gate fires a phantom randomChance → this seed diverges)"
    );
}

// ============================================================================
// BATCH-3 pins (`gen3_berry_trace_shedskin_v1`) — BR1-BR6. Ground truth:
// `harness/probe_berry_batch3_regression_rng.js` (BR1-BR5) +
// `harness/probe_berry_threshold_boundary.js` (BR6 — the exact-equality
// boundary oracle): the REAL sim's printed per-decision seedAfter/state
// values, copied verbatim. All scenarios start from the shared
// pre-first-decision seed the probes printed.
// ============================================================================

const BR_SEED: &str = "43932,6299,55466,29623";

/// BR1 — a SITRUS berry eats at the residual order-10-subOrder-4 slot when the toss
/// grind crosses `2*hp <= maxhp` EXACTLY (+30, item → NONE, permanently), and its
/// residual handler occupies the SAME sort slot as Leftovers: the LEFTOVERS TWIN runs
/// the IDENTICAL seed stream (both draw-free — only the STATE differs). The WRONG
/// (pre-batch-3) behaviour: the berry never eats (hp 224 at dec2, item kept).
#[test]
fn sitrus_eats_at_the_half_threshold_in_the_leftovers_slot() {
    let d = dex();
    let blissey = "Blissey|||NoAbility|seismictoss,softboiled||,,,,,252|N||||";
    let script = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];

    let holder = "Snorlax||sitrusberry|NoAbility|splash,bodyslam|Adamant|252,252,,,,|N||||";
    let mut bs = Battle::start_with_switchins(&opts_cg(blissey, holder, BR_SEED), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(&script, &d);
    // dec2: 324 − 100 = 224 <= 262 (maxhp 524 half) → eat + 30 → 254, item gone.
    let lax = &bs.state().unwrap().sides[1].pokemon[0];
    assert_eq!(lax.hp, 254, "BR1: the sitrus eats at 2*hp<=maxhp and heals +30 (224 → 254)");
    assert!(lax.item.is_empty(), "BR1: the eaten berry is NONE for the battle");
    assert_eq!(
        seed_str(&os.decisions[2].seed_after),
        "56087,52753,40765,14421",
        "BR1: the eat + heal are DRAW-FREE (the probe's dec2 seed)"
    );

    // The LEFTOVERS twin: the same slot, per-turn heal, IDENTICAL seeds at every decision.
    let twin = "Snorlax||leftovers|NoAbility|splash,bodyslam|Adamant|252,252,,,,|N||||";
    let mut bt = Battle::start_with_switchins(&opts_cg(blissey, twin, BR_SEED), &d).expect("start");
    let ot = bt.state_mut().unwrap().run_full_battle(&script, &d);
    for (i, (a, b)) in os.decisions.iter().zip(ot.decisions.iter()).enumerate() {
        assert_eq!(
            seed_str(&a.seed_after),
            seed_str(&b.seed_after),
            "BR1 twin dec {i}: a berry residual handler must be draw-identical to Leftovers'"
        );
    }
    assert_eq!(
        bt.state().unwrap().sides[1].pokemon[0].hp,
        320,
        "BR1 twin: Leftovers healed per-turn instead (the STATE is the only difference)"
    );
}

/// BR2 — a LUM berry eats IMMEDIATELY inside setStatus (onAfterSetStatus −1): the
/// TWave'd holder is CURED BEFORE ITS OWN MOVE, so it never rolls full-para that turn
/// — the no-item CONTROL's par sticks and its onBeforeMove full-para roll SHIFTS the
/// whole seed stream. The WRONG behaviour: the cure waits (a para'd boundary shows) or
/// the eat draws.
#[test]
fn lum_eats_immediately_inside_set_status_before_the_holders_move() {
    let d = dex();
    let jolt = "Jolteon|||NoAbility|thunderwave,thunderbolt|Timid|,,,,252,252|N||||";
    let script = [ScriptDecision::both(Choice::Move(0), Choice::Move(0))];

    let holder = "Snorlax||lumberry|NoAbility|bodyslam,earthquake|Adamant|252,252,,,,|N||||";
    let mut bs = Battle::start_with_switchins(&opts_cg(jolt, holder, BR_SEED), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(&script, &d);
    let lax = &bs.state().unwrap().sides[1].pokemon[0];
    assert_eq!(lax.status, None, "BR2: the lum cured the TWave para inside setStatus");
    assert!(lax.item.is_empty(), "BR2: the lum is eaten");
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "56087,52753,40765,14421",
        "BR2: the cured holder moves WITHOUT a full-para roll (the probe's dec0 seed)"
    );

    let ctl = "Snorlax|||NoAbility|bodyslam,earthquake|Adamant|252,252,,,,|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(jolt, ctl, BR_SEED), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(&script, &d);
    assert_eq!(
        bc.state().unwrap().sides[1].pokemon[0].status,
        Some(Status::Paralysis),
        "BR2 control: without the lum the para sticks"
    );
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        "36426,18980,64710,21836",
        "BR2 control: the stuck para adds the onBeforeMove full-para roll (the probe's control seed)"
    );
}

/// BR3 — a STARF berry's pinch eat (`4*hp <= maxhp`) draws ONE `sample` over the
/// non-capped [atk,def,spa,spd,spe] and boosts the drawn stat +2 (here spe). The
/// WRONG behaviour: no sample draw (the seed matches the pre-eat stream) or a wrong
/// pool/stat.
#[test]
fn starf_pinch_draws_the_sample_and_boosts_plus_two() {
    let d = dex();
    let blissey = "Blissey|||NoAbility|seismictoss,softboiled||,,,,,252|N||||";
    let holder = "Snorlax||starfberry|NoAbility|splash,bodyslam|Adamant|252,252,,,,|N||||";
    let script = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];
    let mut bs = Battle::start_with_switchins(&opts_cg(blissey, holder, BR_SEED), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(&script, &d);
    let lax = &bs.state().unwrap().sides[1].pokemon[0];
    // dec3: 224 − 100 = 124 <= 131 (maxhp 524 quarter) → eat + sample → +2 spe.
    assert!(lax.item.is_empty(), "BR3: the starf is eaten at the pinch threshold");
    assert_eq!(lax.boosts[4], 2, "BR3: the sampled stat (spe on this seed) is boosted +2");
    assert_eq!(
        seed_str(&os.decisions[3].seed_after),
        "60588,12615,50265,13854",
        "BR3: the onEat sample is ONE real draw (the probe's dec3 seed — the sitrus/leftovers \
         stream reads 57171,35906,17183,61887 here, one draw earlier)"
    );
}

/// BR4 — SHED SKIN draws ONE `randomChance(33,100)` per STATUSED residual (order 10
/// subOrder 3); an unstatused holder / no-op control draws NOTHING, so the streams
/// diverge from the first statused residual on. The WRONG behaviour: no roll (the
/// control seed shows on the Shed Skin run) or a roll while unstatused.
#[test]
fn shed_skin_rolls_once_per_statused_residual() {
    let d = dex();
    let blissey = "Blissey|||NoAbility|thunderwave,softboiled||,,,,,252|N||||";
    let script = [ScriptDecision::both(Choice::Move(0), Choice::Move(0))];

    let holder = "Arbok|||ShedSkin|sludgebomb,earthquake|Adamant|252,252,,,,|N||||";
    let mut bs = Battle::start_with_switchins(&opts_cg(blissey, holder, BR_SEED), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(&script, &d);
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "57171,35906,17183,61887",
        "BR4: the par'd Shed Skin holder rolls randomChance(33,100) at its residual (the probe's dec0 seed)"
    );

    let ctl = "Arbok|||RunAway|sludgebomb,earthquake|Adamant|252,252,,,,|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(blissey, ctl, BR_SEED), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(&script, &d);
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        "36426,18980,64710,21836",
        "BR4 control: a no-op ability draws NO residual roll (the probe's control seed)"
    );
}

/// BR5 — a MID-BATTLE TRACE switch-in draws the n=1 `randomFoe` sample (`random(1)`
/// — it draws even for a single foe, the phaze-n=1 gotcha) and copies the foe's
/// CURRENT ability LIVE. The Limber control draws nothing — the streams diverge from
/// the switch decision on. The WRONG behaviour: no draw / no copy.
#[test]
fn trace_draws_the_n1_sample_and_copies_the_foes_ability() {
    let d = dex();
    let p1 = "Machamp|||Guts|crosschop,crosschop|Adamant|252,252,,,,|N||||]Gardevoir|||Trace|psychic,thunderbolt|Modest|,,,,252,252|N||||";
    let foe = "Snorlax|||Immunity|bodyslam,earthquake|Adamant|252,252,,,,|N||||";
    let script = [ScriptDecision { p1: Some(Choice::Switch(1)), p2: Some(Choice::Move(0)) }];

    let mut bs = Battle::start_with_switchins(&opts_cg(p1, foe, BR_SEED), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(&script, &d);
    let st = bs.state().unwrap();
    let active = st.sides[0].active;
    let traced = st.sides[0].pokemon[active].ability.to_lowercase().replace(|c: char| !c.is_ascii_alphanumeric(), "");
    assert_eq!(traced, "immunity", "BR5: Trace copied the foe's CURRENT ability (Immunity)");
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "56087,52753,40765,14421",
        "BR5: the n=1 randomFoe sample is ONE real draw (the probe's dec0 seed)"
    );

    let ctl = "Machamp|||Guts|crosschop,crosschop|Adamant|252,252,,,,|N||||]Gardevoir|||Limber|psychic,thunderbolt|Modest|,,,,252,252|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(ctl, foe, BR_SEED), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(&script, &d);
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        "14910,9013,11608,25386",
        "BR5 control: a non-Trace switch-in draws nothing extra (the probe's control seed)"
    );
}

/// BR6 — the berry thresholds eat at EXACT equality (`hp <= maxhp/2` == `2*hp <= maxhp`,
/// pinch `4*hp <= maxhp`) — the `<=`-vs-`<` BOUNDARY pin the prior probes could not
/// reach (their Snorlax boards had ODD maxhp 461/524, so exact equality was unreachable
/// and a `<=` → `<` engine mutation passed every golden + pin). Ground truth:
/// `harness/probe_berry_threshold_boundary.js` — an EVEN-maxhp Vaporeon (base HP 130,
/// IV hp 30 → 2*130+30+0+110 = 400) ground by Blissey's Seismic Toss (fixed 100):
/// 400 → 300 → **200 == maxhp/2 EXACTLY** (sitrus ATE, per the sim) and
/// → **100 == maxhp/4 EXACTLY** (salac ATE, +1 spe). The sim eats AT equality; the
/// WRONG (`<`) behaviour: no eat at either boundary (hp 200 item kept / boosts 0 item
/// kept) — this pin FAILS under that mutation (verified by mutating, then restoring).
#[test]
fn berry_thresholds_eat_at_exact_equality() {
    let d = dex();
    let blissey = "Blissey|||NoAbility|seismictoss,softboiled||,,,,,252|N||||";

    // BR6a — HEAL class at hp == maxhp/2 exactly: 2 tosses → 200/400, the sitrus eats
    // (+30 → 230, item NONE). The eat is DRAW-FREE, so the seed does NOT discriminate
    // `<=` vs `<` here — the STATE (hp/item) is the pin.
    let sitrus = "Vaporeon||sitrusberry|NoAbility|splash,watergun||,,,,,|N|30,,,,,|||";
    let script2 = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];
    let mut bs = Battle::start_with_switchins(&opts_cg(blissey, sitrus, BR_SEED), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(&script2, &d);
    let vap = &bs.state().unwrap().sides[1].pokemon[0];
    assert_eq!(vap.maxhp, 400, "BR6a: the constructed EVEN maxhp (2*130+30+0+110)");
    assert_eq!(
        vap.hp, 230,
        "BR6a: at hp == maxhp/2 EXACTLY (200/400) the sitrus EATS (the sim's `<=` boundary) and heals +30"
    );
    assert!(vap.item.is_empty(), "BR6a: the boundary eat consumed the berry");
    assert_eq!(
        seed_str(&os.decisions[1].seed_after),
        "40791,31030,28908,8155",
        "BR6a: the boundary eat is DRAW-FREE (the probe's dec1 seed)"
    );

    // BR6b — PINCH class at hp == maxhp/4 exactly: 3 tosses → 100/400, the salac eats
    // (+1 spe, item NONE).
    let salac = "Vaporeon||salacberry|NoAbility|splash,watergun||,,,,,|N|30,,,,,|||";
    let script3 = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];
    let mut bp = Battle::start_with_switchins(&opts_cg(blissey, salac, BR_SEED), &d).expect("start");
    let op = bp.state_mut().unwrap().run_full_battle(&script3, &d);
    let vap = &bp.state().unwrap().sides[1].pokemon[0];
    assert_eq!(
        vap.hp, 100,
        "BR6b: 3 tosses land EXACTLY on maxhp/4 (100/400) — no heal, the pinch class"
    );
    assert_eq!(
        vap.boosts[4], 1,
        "BR6b: at hp == maxhp/4 EXACTLY the salac EATS (the sim's `<=` boundary) and boosts +1 spe"
    );
    assert!(vap.item.is_empty(), "BR6b: the boundary eat consumed the berry");
    assert_eq!(
        seed_str(&op.decisions[2].seed_after),
        "56087,52753,40765,14421",
        "BR6b: the boundary eat is DRAW-FREE (the probe's dec2 seed)"
    );
}

// ============================================================================
// BATCH-4 pins (`gen3_ability_batch4_v1`) — Truant / Inner Focus / Shadow Tag /
// Cute Charm+attract / Color Change / King's Rock / Focus Band. Ground truth:
// `harness/probe_batch4_regression_rng.js` (constructed gen3customgame boards,
// per-decision seedAfter + observables copied verbatim from the real sim).
// ============================================================================

/// B4-1 TRUANT: Slaking moves turn 1 and LOAFS turn 2 (`|cant|…|ability: Truant` — DRAW-FREE:
/// the loaf turn draws NOTHING for the loafer, the foe's HP is untouched, and NO PP is
/// deducted), then moves turn 3 (the order-27 residual toggle parity). REVERT (no truant gate /
/// no toggle): Slaking attacks turn 2 → the foe's HP + the seed diverge.
#[test]
fn truant_loafs_every_other_turn_draw_free() {
    let d = dex();
    let slaking = "Slaking|||Truant|bodyslam,earthquake|Adamant|,252,,,,|N||||";
    let swampert = "Swampert|||Torrent|surf,surf|Modest|252,,,252,,|N||||";
    let seed = "21001,4383,9902,61177";
    let mut b = Battle::start_with_switchins(&opts_cg(slaking, swampert, seed), &d).expect("start");
    let out = b.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );
    // t1: Slaking Body Slams (Swampert 404→185, paralyzed by the 30% secondary on this seed).
    assert_eq!(out.decisions[0].active[1].hp, 185, "B4-1 t1: Slaking attacked (Swampert damaged)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "32363,15782,37164,39400",
        "B4-1 t1 seed (the move turn)"
    );
    // t2: the LOAF — Swampert's HP is UNCHANGED by Slaking (only Surf hits Slaking) and the
    // loaf turn deducts NO PP (Body Slam still at t1's count).
    assert_eq!(out.decisions[1].active[1].hp, 185, "B4-1 t2: the loaf turn leaves Swampert untouched");
    assert_eq!(
        out.decisions[1].active[0].move_pp[0],
        out.decisions[0].active[0].move_pp[0],
        "B4-1 t2: the loaf deducts NO PP"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "497,35348,37006,47663",
        "B4-1 t2 seed: the loaf turn is DRAW-FREE for the loafer (no acc/crit/dmg, no para roll)"
    );
    // t3: Slaking moves again (the toggle) and KOs Swampert.
    assert_eq!(out.decisions[2].active[1].hp, 0, "B4-1 t3: Slaking moves again (the toggle parity)");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "1652,38127,57165,44755", "B4-1 t3 seed");
}

/// B4-2 INNER FOCUS: on a seed where Bite's 30% flinch PASSES, the Inner-Focus Snorlax still
/// MOVES (the block is at the volatile APPLY — the secondary random(100) IS drawn), while the
/// Thick-Fat control is `|cant|…|flinch`'d on the SAME seed. REVERT (filter-the-draw instead of
/// block-at-apply): the Inner-Focus seed diverges; (no block): Snorlax is wrongly cant'd (HP).
#[test]
fn inner_focus_blocks_the_flinch_at_the_apply_but_draws_the_roll() {
    let d = dex();
    let jolt = "Jolteon|||Static|bite,thunderbolt|Timid|,,,252,,252|N||||";
    let seed = "4530,50537,53172,14211";
    // INNER FOCUS: the flinch is blocked → Snorlax Body Slams Jolteon (271→28).
    let lax_if = "Snorlax|||InnerFocus|bodyslam,bodyslam|Adamant|252,252,,,,|N||||";
    let mut bi = Battle::start_with_switchins(&opts_cg(jolt, lax_if, seed), &d).expect("start");
    let oi = bi.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(oi.decisions[0].active[0].hp, 28, "B4-2: Inner Focus holder MOVED (Jolteon hit)");
    assert_eq!(
        seed_str(&oi.decisions[0].seed_after),
        "55682,61108,31594,43269",
        "B4-2: the flinch roll IS drawn (block at the apply, not the draw)"
    );
    // CONTROL (Thick Fat): the same seed's flinch LANDS → Snorlax cant, Jolteon untouched.
    let lax_tf = "Snorlax|||ThickFat|bodyslam,bodyslam|Adamant|252,252,,,,|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(jolt, lax_tf, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(oc.decisions[0].active[0].hp, 271, "B4-2 control: the flinch cants Snorlax");
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        "40884,39481,8915,28390",
        "B4-2 control seed (the cant'd Body Slam draws nothing)"
    );
}

/// B4-3 SHADOW TAG: a FLYING foe (Skarmory) is trapped UNCONDITIONALLY (no grounded gate) and
/// the trap adds ZERO draws — the Keen-Eye control's post-turn seed is IDENTICAL. REVERT (a
/// grounded gate like Arena Trap's): Skarmory untrapped; (a draw): the seed diverges.
#[test]
fn shadow_tag_traps_a_flying_foe_draw_free() {
    let d = dex();
    let skarm = "Skarmory|||KeenEye|drillpeck,drillpeck|Adamant|252,252,,,,|N||||]Snorlax|||ThickFat|bodyslam,bodyslam|Serious||N||||";
    let seed = "30303,11111,47474,5252";
    let st = "Golduck|||ShadowTag|surf,surf|Modest|252,,,252,,|N||||";
    let mut bs = Battle::start_with_switchins(&opts_cg(st, skarm, seed), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert!(os.decisions[0].trapped[1], "B4-3: the FLYING Skarmory is trapped (no grounded gate)");
    assert!(!os.decisions[0].trapped[0], "B4-3: the holder itself is not trapped");
    assert_eq!(
        seed_str(&os.decisions[0].seed_after),
        "10731,36808,49268,57461",
        "B4-3: Shadow Tag adds ZERO draws"
    );
    let ke = "Golduck|||KeenEye|surf,surf|Modest|252,,,252,,|N||||";
    let mut bc = Battle::start_with_switchins(&opts_cg(ke, skarm, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert!(!oc.decisions[0].trapped[1], "B4-3 control: no trap ability → untrapped");
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        "10731,36808,49268,57461",
        "B4-3 control: the IDENTICAL seed (the trap computation is draw-free)"
    );
}

/// B4-4 CUTE CHARM + ATTRACT: the 1/3 DamagingHit roll PASSES → the M attacker is attracted
/// (`-start … Attract`), and its NEXT turn's attract 1/2 CANTS it (Miltank untouched turn 2).
/// The F-into-F CONTROL on the SAME seed draws the SAME 1/3 roll (dec-0 seed IDENTICAL — the
/// gender gate lives INSIDE the volatile's onStart) but never attracts (its dec-1 stream
/// diverges: Zangoose moves). REVERT (gender-gate-before-the-roll): the control's dec-0 seed
/// diverges; (no attract): the M's dec-1 diverges.
#[test]
fn cute_charm_attracts_and_the_attract_cants_gender_gated_after_the_roll() {
    let d = dex();
    let miltank = "Miltank|||CuteCharm|bodyslam,bodyslam|Adamant|252,252,,,,|F||||";
    let seed = "64109,6376,42791,64090";
    let script = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];
    // The M attacker: attracted on dec 0; attract-cant'd (then KO'd) on dec 1.
    let zang_m = "Zangoose|||Immunity|scratch,scratch|Adamant|,252,,,,252|M||||";
    let mut bm = Battle::start_with_switchins(&opts_cg(zang_m, miltank, seed), &d).expect("start");
    let om = bm.state_mut().unwrap().run_full_battle(&script, &d);
    assert!(
        bm.state().unwrap().sides[0].pokemon[0].attract.is_none()
            || bm.state().unwrap().sides[0].pokemon[0].fainted,
        "B4-4: (sanity) the attracted mon fainted at dec 1 on this seed"
    );
    assert_eq!(
        seed_str(&om.decisions[0].seed_after),
        "30033,64362,63935,24717",
        "B4-4 dec 0: the 1/3 roll + the attract add"
    );
    assert_eq!(om.decisions[1].active[1].hp, 324, "B4-4 dec 1: the attract-cant'd Zangoose never hit");
    assert_eq!(
        seed_str(&om.decisions[1].seed_after),
        "12100,55611,12281,32440",
        "B4-4 dec 1: the -activate + attract 1/2 cant"
    );
    // The F-into-F control: the SAME dec-0 seed (the roll STILL draws), no attract, dec 1 differs.
    let zang_f = "Zangoose|||Immunity|scratch,scratch|Adamant|,252,,,,252|F||||";
    let mut bf = Battle::start_with_switchins(&opts_cg(zang_f, miltank, seed), &d).expect("start");
    let of = bf.state_mut().unwrap().run_full_battle(&script, &d);
    assert_eq!(
        seed_str(&of.decisions[0].seed_after),
        "30033,64362,63935,24717",
        "B4-4 control dec 0: the 1/3 roll DRAWS for a same-gender pair too (IDENTICAL seed)"
    );
    assert_eq!(of.decisions[1].active[1].hp, 259, "B4-4 control dec 1: the un-attracted Zangoose hits");
    assert_eq!(
        seed_str(&of.decisions[1].seed_after),
        "43496,14430,9531,53717",
        "B4-4 control dec 1 seed"
    );
}

/// B4-5 COLOR CHANGE: TBolt overrides Kecleon to Electric (`typechange`), and the NEXT turn's
/// Earthquake reads the chart THROUGH the override (super-effective into the Electric-Kecleon:
/// 188→57) then re-overrides to Ground. REVERT (no override / a stale species-types read):
/// the EQ damage + the seed diverge.
#[test]
fn color_change_overrides_the_types_for_later_chart_reads() {
    let d = dex();
    let jolt = "Jolteon|||Static|thunderbolt,earthquake|Timid|,,,252,,252|N||||";
    let kec = "Kecleon|||ColorChange|surf,surf|Modest|252,,,252,,|N||||";
    let seed = "42424,3141,59265,35897";
    let mut b = Battle::start_with_switchins(&opts_cg(jolt, kec, seed), &d).expect("start");
    let out = b.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[1].hp, 188, "B4-5 t1: TBolt lands (Kecleon → Electric)");
    assert_eq!(
        b.state().unwrap().sides[1].pokemon[0].types_override,
        Some(vec![pokesim::dex::Type::Ground]),
        "B4-5: the override is LIVE and re-overridden to Ground by the EQ"
    );
    assert_eq!(
        out.decisions[1].active[1].hp, 57,
        "B4-5 t2: the EQ is SUPER-EFFECTIVE through the Electric override (188→57)"
    );
    assert_eq!(seed_str(&out.decisions[0].seed_after), "31090,28568,52667,41985", "B4-5 t1 seed");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "3721,7167,39301,40590", "B4-5 t2 seed (the override is draw-free)");
}

/// B4-6 KING'S ROCK: the appended trailing 10% flinch secondary PASSES on Slash → the slower
/// Snorlax is `|cant|…|flinch`'d (Zangoose untouched). The no-item CONTROL on the SAME seed
/// never draws the extra roll — its stream diverges and Snorlax hits (287→52). REVERT (no
/// appended secondary): the KR run matches the control instead.
#[test]
fn kings_rock_appends_the_trailing_flinch_secondary() {
    let d = dex();
    let lax = "Snorlax|||ThickFat|bodyslam,bodyslam|Adamant|252,252,,,,|N||||";
    let seed = "45753,48324,41299,13974";
    let kr = "Zangoose||kingsrock|Immunity|slash,slash|Adamant|,252,,,,252|N||||";
    let mut bk = Battle::start_with_switchins(&opts_cg(kr, lax, seed), &d).expect("start");
    let ok = bk.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(ok.decisions[0].active[0].hp, 287, "B4-6: the KR flinch cants Snorlax (Zangoose untouched)");
    assert_eq!(
        seed_str(&ok.decisions[0].seed_after),
        "30846,47153,27480,31349",
        "B4-6: the appended random(100) is a real trailing draw"
    );
    let no = "Zangoose|||Immunity|slash,slash|Adamant|,252,,,,252|N||||";
    let mut bn = Battle::start_with_switchins(&opts_cg(no, lax, seed), &d).expect("start");
    let on = bn.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(on.decisions[0].active[0].hp, 52, "B4-6 control: no item → no flinch, Snorlax hits");
    assert_eq!(
        seed_str(&on.decisions[0].seed_after),
        "56259,61109,53674,9150",
        "B4-6 control seed (no extra roll)"
    );
}

/// B4-7 FOCUS BAND: a LETHAL Cross Chop into the lv-5 FB Rattata passes the 1/10 onDamage roll
/// → survive at exactly 1 HP. The no-item CONTROL on the SAME seed never draws the roll — its
/// stream diverges and Rattata faints. REVERT (no onDamage roll / no survive cap): the HP + the
/// seed diverge.
#[test]
fn focus_band_survives_a_lethal_move_hit_at_one_hp() {
    let d = dex();
    let machamp = "Machamp|||Guts|crosschop,crosschop|Adamant|,252,,,,252|N||||";
    let seed = "17127,25370,22449,60764";
    let fb = "Rattata||focusband|Guts|scratch,scratch|Serious||N|||5|";
    let mut bf = Battle::start_with_switchins(&opts_cg(machamp, fb, seed), &d).expect("start");
    let of = bf.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(of.decisions[0].active[1].hp, 1, "B4-7: the lethal hit is survived at 1 HP");
    assert!(!of.decisions[0].active[1].fainted, "B4-7: the holder did NOT faint");
    assert_eq!(
        seed_str(&of.decisions[0].seed_after),
        "13375,25876,9023,16324",
        "B4-7: the onDamage 1/10 is a real draw"
    );
    let no = "Rattata|||Guts|scratch,scratch|Serious||N|||5|";
    let mut bn = Battle::start_with_switchins(&opts_cg(machamp, no, seed), &d).expect("start");
    let on = bn.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(on.decisions[0].active[1].hp, 0, "B4-7 control: no item → the hit KOs");
    assert_eq!(
        seed_str(&on.decisions[0].seed_after),
        "16582,10104,5878,63593",
        "B4-7 control seed (the onDamage roll absent)"
    );
}

/// `gen3_shielddust_sub_v1` — SHIELD DUST does NOT filter a secondary when a SUBSTITUTE
/// absorbs the hit (the A/B fuzzer's #1 sub×secondary SEED cluster, auto_0708_0304 corpus:
/// 347/365 ShieldDust-team repros flip ok on this fix). Shield Dust's filter is a
/// TARGET-gathered ModifySecondaries handler; a sub-absorbed hit's target list is `null`,
/// so the filter never gathers and the secondary `random(100)` STILL DRAWS (held AND
/// breaking sub) — while the effect stays sub-suppressed. The same holds for the Tri
/// Attack 20% gate and the King's Rock appended secondary. A BARE Shield Dust defender
/// still filters (no draw). WRONG (pre-fix): the port filtered unconditionally → one
/// missing draw per secondary-into-a-Shield-Dust-sub → every later draw desynced.
/// Probes: `harness/probe_sub_break_secondary_rng.js` (semantics) +
/// `harness/probe_shielddust_sub_regression_rng.js` (this pin's ground-truth seeds).
#[test]
fn shield_dust_behind_a_substitute_still_draws_the_secondary() {
    let d = dex();
    let venomoth_sub = "Venomoth|||ShieldDust|substitute,splash|Serious|252,,,,,|N||||";
    let seed = "21,32,43,54";
    // The shared plan: dec0 p2 subs (p1 splashes), dec1 the attacker hits the sub.
    let sub_plan = [
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
    ];

    // SD-a: the move's OWN secondary — Flamethrower (brn10) into the subbed Shield Dust
    // Venomoth: acc + crit + dmg + the SECONDARY random(100) + Quick Claw (5 draws).
    let magcargo = "Magcargo|||FlameBody|flamethrower,splash|Serious||N||||";
    let mut ba = Battle::start_with_switchins(&opts_cg(magcargo, venomoth_sub, seed), &d).expect("start");
    let oa = ba.state_mut().unwrap().run_full_battle(&sub_plan, &d);
    assert_eq!(
        seed_str(&oa.decisions[0].seed_after),
        "57584,47432,56756,39441",
        "SD-a dec0: sub-up turn (Quick Claw only)"
    );
    assert_eq!(
        seed_str(&oa.decisions[1].seed_after),
        "24708,24771,25359,51852",
        "SD-a dec1: the secondary random(100) DRAWS behind the Shield Dust sub"
    );
    assert_eq!(
        ba.state().unwrap().sides[1].pokemon[0].status,
        None,
        "SD-a: the sub still SUPPRESSES the effect (Venomoth never burned)"
    );

    // SD-a2 BARE control: same board, no sub — the filter holds (NO random(100); the
    // post-hit seed DIFFERS from a would-be drawn stream).
    let bare_plan = [
        ScriptDecision::both(Choice::Move(1), Choice::Move(1)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
    ];
    let mut bc = Battle::start_with_switchins(&opts_cg(magcargo, venomoth_sub, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(&bare_plan, &d);
    assert_eq!(
        seed_str(&oc.decisions[1].seed_after),
        "63025,65261,3364,9365",
        "SD-a2 control: a BARE Shield Dust defender still FILTERS the secondary (4 draws)"
    );

    // SD-b: the Tri Attack 20% gate random(100) draws behind the sub (the random(3)
    // sample stays sub-suppressed) — same draw count as SD-a from the same seed.
    let dodrio = "Dodrio|||EarlyBird|triattack,splash|Serious||N||||";
    let mut bb = Battle::start_with_switchins(&opts_cg(dodrio, venomoth_sub, seed), &d).expect("start");
    let ob = bb.state_mut().unwrap().run_full_battle(&sub_plan, &d);
    assert_eq!(
        seed_str(&ob.decisions[1].seed_after),
        "24708,24771,25359,51852",
        "SD-b: the Tri Attack gate random(100) DRAWS behind the Shield Dust sub"
    );
    assert_eq!(bb.state().unwrap().sides[1].pokemon[0].status, None, "SD-b: no status through the sub");

    // SD-c: the King's Rock APPENDED secondary draws behind the sub (flinch suppressed);
    // the weak typed-HP hit leaves the sub HELD at 9.
    let sceptile = "Sceptile||kingsrock|Overgrow|hiddenpowerdark,splash|Serious||N||||";
    let mut bk = Battle::start_with_switchins(&opts_cg(sceptile, venomoth_sub, seed), &d).expect("start");
    let ok = bk.state_mut().unwrap().run_full_battle(&sub_plan, &d);
    assert_eq!(
        seed_str(&ok.decisions[1].seed_after),
        "24708,24771,25359,51852",
        "SD-c: the King's Rock random(100) DRAWS behind the Shield Dust sub"
    );
    assert_eq!(
        bk.state().unwrap().sides[1].pokemon[0].substitute,
        Some(9),
        "SD-c: the sub is HELD at 9 HP (the KR flinch never applied)"
    );
}

// ============================================================================
// FA — FACADE ×2-when-statused (`gen3_facade_v1`) + the runEvent-tail
//      INTEGER-GUARD (the A/B fuzzer's post-ShieldDust #1 cluster: 143/145
//      facade-team repros in auto_0709_0805 flip ok on the fix). The WRONG
//      (pre-fix) behaviour: the port priced Facade FLAT BP 70 (the dist
//      `onBasePower` — `chainModify(2)` when the user has a non-slp major
//      status — was unmodeled), and a co-firing Direct item (Pink Bow ×1.1)
//      DISCARDED the accumulated BP chain (the sim re-applies it when the
//      float is integer-valued: `70 * 1.1 == 77` exactly in f64 →
//      `modify(77, ×2)` = BP 154). Semantics: `harness/probe_facade_gen3.js`;
//      pin ground truth: `harness/probe_facade_defrost_regression_rng.js`.
// ============================================================================

/// FA: a statused Facade doubles its BP as a BP-CHAIN member — poisoned ×2 (140),
/// burned ×2 with the gen3 burn-halve STILL applied, burned GUTS composing
/// (Atk ×1.5 + halve-suppressed + BP ×2 → the KO), and Pink Bow + poisoned
/// composing through the integer-guard (BP 154) — each bit-for-bit vs the sim
/// at the same raw seed (same draw stream: the boost is DRAW-FREE).
#[test]
fn facade_status_doubles_bp_and_composes() {
    let d = dex();
    let snorlax = "Snorlax||||splash,splash|Serious||N||||";
    let seed = "21,32,43,54";
    let one_turn = [ScriptDecision::both(Choice::Move(0), Choice::Move(0))];

    // FA-a: POISONED Facade = BP 140 (Snorlax 461 → 61 at this seed's roll).
    let mut ba = Battle::start_with_switchins(
        &opts_cg("Raticate||||facade,splash|Serious||N||||", snorlax, seed), &d,
    ).expect("start");
    let sta = ba.state_mut().unwrap();
    sta.sides[0].pokemon[0].status = Some(Status::Poison);
    let oa = sta.run_full_battle(&one_turn, &d);
    assert_eq!(sta.sides[1].pokemon[0].hp, 61, "FA-a: poisoned Facade hits at BP 140");
    assert_eq!(
        seed_str(&oa.decisions[0].seed_after),
        "8675,54279,40138,64106",
        "FA-a: the boost is DRAW-FREE (seed matches the sim)"
    );

    // FA-b: BURNED Facade = BP 140 AND the burn damage-halve still applies —
    // at this seed the damage equals the UNSTATUSED control (203: 461 → 258).
    let mut bb = Battle::start_with_switchins(
        &opts_cg("Raticate||||facade,splash|Serious||N||||", snorlax, seed), &d,
    ).expect("start");
    let stb = bb.state_mut().unwrap();
    stb.sides[0].pokemon[0].status = Some(Status::Burn);
    let ob = stb.run_full_battle(&one_turn, &d);
    assert_eq!(stb.sides[1].pokemon[0].hp, 258, "FA-b: burned Facade = ×2 BP × the burn-halve");
    assert_eq!(seed_str(&ob.decisions[0].seed_after), "8675,54279,40138,64106", "FA-b seed");

    // FA-c: BURNED GUTS Facade — Atk ×1.5 + halve SUPPRESSED + BP ×2 → the KO
    // (461 → 0; the faint turn skips Quick Claw → a different seed).
    let mut bc = Battle::start_with_switchins(
        &opts_cg("Raticate|||Guts|facade,splash|Serious||N||||", snorlax, seed), &d,
    ).expect("start");
    let stc = bc.state_mut().unwrap();
    stc.sides[0].pokemon[0].status = Some(Status::Burn);
    let oc = stc.run_full_battle(&one_turn, &d);
    assert_eq!(stc.sides[1].pokemon[0].hp, 0, "FA-c: burned Guts Facade KOs (×1.5 Atk, no halve)");
    assert_eq!(seed_str(&oc.decisions[0].seed_after), "25995,36281,50774,34331", "FA-c seed");

    // FA-d: PINK BOW + POISONED Facade — the Direct ×1.1 float lands EXACTLY on 77
    // in f64, the runEvent-tail integer-guard PASSES, and the ×2 chain RE-APPLIES:
    // BP 154 (461 → 22). Under the old "Direct discards the chain" shortcut this
    // would price BP 77.
    let mut bd = Battle::start_with_switchins(
        &opts_cg("Raticate||pinkbow||facade,splash|Serious||N||||", snorlax, seed), &d,
    ).expect("start");
    let std_ = bd.state_mut().unwrap();
    std_.sides[0].pokemon[0].status = Some(Status::Poison);
    let od = std_.run_full_battle(&one_turn, &d);
    assert_eq!(std_.sides[1].pokemon[0].hp, 22, "FA-d: bow ×1.1 THEN the ×2 chain re-applies (BP 154)");
    assert_eq!(seed_str(&od.decisions[0].seed_after), "8675,54279,40138,64106", "FA-d seed");

    // FA-e control: UNSTATUSED Facade stays BP 70 (461 → 258, same as FA-b's total).
    let mut be = Battle::start_with_switchins(
        &opts_cg("Raticate||||facade,splash|Serious||N||||", snorlax, seed), &d,
    ).expect("start");
    let ste = be.state_mut().unwrap();
    let oe = ste.run_full_battle(&one_turn, &d);
    assert_eq!(ste.sides[1].pokemon[0].hp, 258, "FA-e: unstatused Facade stays BP 70");
    assert_eq!(seed_str(&oe.decisions[0].seed_after), "8675,54279,40138,64106", "FA-e seed");
}

// ============================================================================
// DF — FROZEN user × `flags.defrost` (`gen3_defrost_v1`, the A/B fuzzer's
//      sacredfire tail: 8/10 sacredfire/flamewheel-team repros in
//      auto_0709_0805 flip ok on the fix). The WRONG (pre-fix) behaviour: the
//      port cant'd EVERY failed 1/5 thaw roll. The resolved gen3
//      `frz.onBeforeMove` puts the roll FIRST (it always DRAWS), but on a
//      failed roll a defrost move (Sacred Fire / Flame Wheel) PROCEEDS and is
//      thawed draw-free by `frz.onModifyMove` (`|-curestatus|…|[from] move:`
//      before the `|move|` line). Semantics: `harness/probe_sacredfire_defrost.js`
//      (frozen defrost user: moved 25/25, thawed 25/25, EXACTLY +1 draw vs
//      healthy); pin ground truth: `harness/probe_facade_defrost_regression_rng.js`.
// ============================================================================

/// DF: a frozen Sacred Fire / Flame Wheel user draws the thaw roll, PROCEEDS on a
/// failed roll, and ends the turn un-frozen — while a frozen NON-defrost Fire move
/// (Flamethrower) at the same seed is cant'd (stays frozen, target untouched).
#[test]
fn frozen_defrost_move_bypasses_the_cant_and_thaws() {
    let d = dex();
    let blissey = "Blissey||||splash,splash|Serious||N||||";
    let seed = "21,32,43,54";

    // DF-a: frozen Ho-Oh Sacred Fire — the 1/5 thaw roll FAILS at this seed, the move
    // proceeds anyway (Blissey 651 → 546), Ho-Oh ends un-frozen.
    let hooh = "Ho-Oh||||sacredfire,flamethrower|Serious||N||||";
    let mut ba = Battle::start_with_switchins(&opts_cg(hooh, blissey, seed), &d).expect("start");
    let sta = ba.state_mut().unwrap();
    sta.sides[0].pokemon[0].status = Some(Status::Freeze);
    let oa = sta.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(sta.sides[1].pokemon[0].hp, 546, "DF-a: the frozen Sacred Fire STILL fires");
    assert_eq!(sta.sides[0].pokemon[0].status, None, "DF-a: the user THAWS on use");
    assert_eq!(
        seed_str(&oa.decisions[0].seed_after),
        "24708,24771,25359,51852",
        "DF-a: the thaw roll IS drawn + the full move draws follow (sim seed)"
    );

    // DF-b: frozen Entei Flame Wheel — same model (Blissey 651 → 598).
    let entei = "Entei||||flamewheel,splash|Serious||N||||";
    let mut bb = Battle::start_with_switchins(&opts_cg(entei, blissey, seed), &d).expect("start");
    let stb = bb.state_mut().unwrap();
    stb.sides[0].pokemon[0].status = Some(Status::Freeze);
    let ob = stb.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(stb.sides[1].pokemon[0].hp, 598, "DF-b: the frozen Flame Wheel STILL fires");
    assert_eq!(stb.sides[0].pokemon[0].status, None, "DF-b: the user THAWS on use");
    assert_eq!(seed_str(&ob.decisions[0].seed_after), "24708,24771,25359,51852", "DF-b seed");

    // DF-c control: frozen Ho-Oh FLAMETHROWER (non-defrost) at the same seed — the
    // failed thaw roll CANTS the move (Blissey untouched, Ho-Oh stays frozen; the
    // 2-draw turn's seed matches the sim).
    let mut bc = Battle::start_with_switchins(&opts_cg(hooh, blissey, seed), &d).expect("start");
    let stc = bc.state_mut().unwrap();
    stc.sides[0].pokemon[0].status = Some(Status::Freeze);
    let oc = stc.run_full_battle(&[ScriptDecision::both(Choice::Move(1), Choice::Move(0))], &d);
    assert_eq!(stc.sides[1].pokemon[0].hp, stc.sides[1].pokemon[0].maxhp, "DF-c: cant — no hit");
    assert_eq!(stc.sides[0].pokemon[0].status, Some(Status::Freeze), "DF-c: stays frozen");
    assert_eq!(seed_str(&oc.decisions[0].seed_after), "2919,23817,33105,32888", "DF-c seed");
}

// ============================================================================
// PM — PLUS / MINUS cross-field SpA ×1.5 (`gen3_plus_minus_v1`, the A/B
//      fuzzer's thunderbolt-vs-Plusle/Minun STATE cluster — 18 recurring
//      repros in the auto_0709_0805 re-triage, 2026-07-10). The WRONG
//      (pre-fix) behaviour: the port priced a Minus attacker's special move
//      FLAT when the OPPOSING active carried Plus — the gen3 RESOLVED
//      `onModifySpA` scans `getAllActive()` (FOES INCLUDED; gen5+ narrowed it
//      to allies), so the old NOOP classification's "partner-less in singles
//      → no-op" was wrong for the cross-field pairing. Semantics:
//      `harness/probe_plus_minus_gen3.js` (maxRoll 90 vs 60 control = ×1.5
//      both directions; plus-vs-plus / minus-vs-minus = NO boost; physical
//      unchanged; post-turn seed identical to the control's). Pin ground
//      truth: `harness/probe_plusminus_ffwisp_regression_rng.js`.
// ============================================================================

/// PM: Minun's (Minus) Thunderbolt into an ACTIVE Plus Plusle is SpA ×1.5
/// (155 vs the Sturdy control's 104 at the same raw seed) — a DRAW-FREE
/// ModifySpA chain member (both boards land on the SAME post-decision seeds).
#[test]
fn minus_boosts_spa_when_the_foe_active_has_plus() {
    let d = dex();
    let minun = "Minun||Leftovers|Minus|thunderbolt,splash|Hardy|85,85,85,85,85,85|M|||100|";
    let plusle = "Plusle||Leftovers|Plus|splash,thunderbolt|Hardy|85,85,85,85,85,85|M|||100|";
    let plusle_ctl = "Plusle||Leftovers|Sturdy|splash,thunderbolt|Hardy|85,85,85,85,85,85|M|||100|";
    let seed = "0,0,0,21";
    let plan = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // tbolt vs splash
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // splash vs splash (leftovers tick)
    ];

    // PM-a BOOSTED: the paired ability is on the FOE active → ×1.5 (282 → 144, then
    // +17 Leftovers → 161). The tbolt's par secondary lands at this seed (sim-verified).
    let mut ba = Battle::start_with_switchins(&opts_cg(minun, plusle, seed), &d).expect("start");
    let sta = ba.state_mut().unwrap();
    let oa = sta.run_full_battle(&plan, &d);
    assert_eq!(sta.sides[1].pokemon[0].hp, 161, "PM-a: Minus vs Plus foe = SpA x1.5 (dmg 155)");
    assert_eq!(
        seed_str(&oa.decisions[0].seed_after),
        "8002,2492,7543,49926",
        "PM-a dec0: the boost is DRAW-FREE (sim seed)"
    );
    assert_eq!(seed_str(&oa.decisions[1].seed_after), "51323,8243,57673,29384", "PM-a dec1 seed");

    // PM-b CONTROL: a Sturdy foe → NO boost (282 → 195, then +17 → 212) at the
    // IDENTICAL seeds (the draw-free proof: only the damage differs).
    let mut bb = Battle::start_with_switchins(&opts_cg(minun, plusle_ctl, seed), &d).expect("start");
    let stb = bb.state_mut().unwrap();
    let ob = stb.run_full_battle(&plan, &d);
    assert_eq!(stb.sides[1].pokemon[0].hp, 212, "PM-b control: no pair on the field = flat (dmg 104)");
    assert_eq!(seed_str(&ob.decisions[0].seed_after), "8002,2492,7543,49926", "PM-b dec0 seed");
    assert_eq!(seed_str(&ob.decisions[1].seed_after), "51323,8243,57673,29384", "PM-b dec1 seed");
}

// ============================================================================
// FFW — WILL-O-WISP into FLASH FIRE is ABSORBED (`gen3_ff_wisp_absorb_v1`, the
//      A/B fuzzer's willowisp STATE cluster — incl. a TRACED Flash Fire on
//      Porygon2). The WRONG (pre-fix) behaviour: the port BURNED a NON-Fire
//      Flash Fire holder (a maxhp/8 DoT desync per residual) — the resolved
//      gen3 `flashfire.onTryHit` absorbs a landed WoW on a non-Fire,
//      status-free, un-subbed holder (the volatile ARMS, no burn; a
//      Fire-type / statused / subbed target falls through to the normal
//      gates). Semantics: `harness/probe_flashfire_rng.js` A3. Pin ground
//      truth: `harness/probe_plusminus_ffwisp_regression_rng.js`.
// ============================================================================

/// FFW: a LANDED Will-O-Wisp into a NON-Fire Flash Fire Snorlax is absorbed —
/// no burn, the `flash_fire` volatile ARMS (its own Flamethrower next turn is
/// ×1.5: Gengar 282 → 191), all DRAW-FREE past the accuracy roll (sim seeds).
#[test]
fn will_o_wisp_into_flash_fire_is_absorbed() {
    let d = dex();
    let gengar = "Gengar||Leftovers|Levitate|willowisp,splash|Hardy|85,85,85,85,85,85|M|||100|";
    let snorlax_ff = "Snorlax||Leftovers|FlashFire|splash,flamethrower|Hardy|85,85,85,85,85,85|M|||100|";
    let seed = "0,0,0,33";
    let plan = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // WoW (lands) vs splash
        ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // splash vs the ARMED flamethrower
    ];

    let mut ba = Battle::start_with_switchins(&opts_cg(gengar, snorlax_ff, seed), &d).expect("start");
    let sta = ba.state_mut().unwrap();
    let oa = sta.run_full_battle(&plan, &d);
    assert_eq!(sta.sides[1].pokemon[0].status, None, "FFW: the WoW is ABSORBED — no burn");
    assert_eq!(
        sta.sides[1].pokemon[0].hp, sta.sides[1].pokemon[0].maxhp,
        "FFW: no burn DoT chip (the pre-fix port chipped maxhp/8 per residual)"
    );
    assert!(sta.sides[1].pokemon[0].flash_fire, "FFW: the absorb ARMS the flash_fire volatile");
    assert_eq!(
        seed_str(&oa.decisions[0].seed_after),
        "10563,48869,858,6571",
        "FFW dec0: accuracy drawn, the absorb is DRAW-FREE (sim seed)"
    );
    // The armed boost is REAL: Snorlax's own Flamethrower is x1.5 (282 -> 191 incl.
    // Gengar's Leftovers), at the sim's exact seed.
    assert_eq!(sta.sides[0].pokemon[0].hp, 191, "FFW dec1: the armed FF x1.5 Flamethrower");
    assert_eq!(seed_str(&oa.decisions[1].seed_after), "63773,5234,47635,50222", "FFW dec1 seed");
}

// ============================================================================
// The 2026-07-10 A/B RESIDUAL-TAIL pins (the auto_0709_0805 re-triage — after
// these five fixes + PM/FFW the corpus replays 307/307 ok). Ground truth:
// `harness/probe_residual_tail_regression_rng.js`.
// ============================================================================

/// CN1 (`gen3_cloudnine_end_v1`, switch-out site): the resolved gen3 Cloud Nine /
/// Air Lock `onEnd` fires `eachEvent('WeatherChange')` when the holder LEAVES the
/// field (switchIn's alive-outgoing ability End, PRE-swap) — one tie-shuffle draw
/// in a Golduck-L81 mirror. The Damp control's switch-out draws nothing (its dec-0
/// seed differs by exactly that draw). WRONG (pre-fix): the port never drew it.
#[test]
fn cloud_nine_switch_out_fires_the_weatherchange_shuffle() {
    let d = dex();
    let golduck_cn = "Golduck||Leftovers|CloudNine|surf,splash|Hardy|85,85,85,85,85,85|M|||81|";
    let golduck_damp = "Golduck||Leftovers|Damp|surf,splash|Hardy|85,85,85,85,85,85|M|||81|";
    let snorlax = "Snorlax||Leftovers|ThickFat|splash,bodyslam|Hardy|85,85,85,85,85,85|M|||100|";
    let seed = "0,0,0,11";
    let plan = [
        ScriptDecision::both(Choice::Switch(1), Choice::Move(1)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];

    let p1_cn = format!("{golduck_cn}]{snorlax}");
    let mut ba = Battle::start_with_switchins(&opts_cg(&p1_cn, golduck_cn, seed), &d).expect("start");
    let oa = ba.state_mut().unwrap().run_full_battle(&plan, &d);
    assert_eq!(
        seed_str(&oa.decisions[0].seed_after),
        "21884,54510,58577,44783",
        "CN1: the Cloud Nine switch-out draws the onEnd WeatherChange tie-shuffle"
    );
    assert_eq!(seed_str(&oa.decisions[1].seed_after), "51742,47708,51928,55187", "CN1 dec1 seed");

    let p1_damp = format!("{golduck_damp}]{snorlax}");
    let mut bb = Battle::start_with_switchins(&opts_cg(&p1_damp, golduck_cn, seed), &d).expect("start");
    let ob = bb.state_mut().unwrap().run_full_battle(&plan, &d);
    assert_eq!(
        seed_str(&ob.decisions[0].seed_after),
        "14058,5850,13963,58044",
        "CN1 control: a Damp switch-out has NO onEnd WeatherChange (one fewer draw)"
    );
}

/// CN2 (`gen3_cloudnine_end_v1`, FAINT site): `faintMessages` fires the faintee's
/// ability End BEFORE `fainted = true` (battle.js:2109) — so a KO'd Cloud Nine
/// Golduck's `onEnd` WeatherChange still gathers BOTH actives. In a Golduck-L81
/// MIRROR (cached speeds tie) the KO turn draws that ONE extra tie-shuffle.
/// WRONG (pre-fix): the port drew nothing at the faint site (the ab_916_16
/// fingerprint — an Air Lock Rayquaza KO'd under a Seaking tie).
#[test]
fn cloud_nine_faint_fires_the_weatherchange_shuffle() {
    let d = dex();
    let golduck_cn = "Golduck||Leftovers|CloudNine|surf,splash|Hardy|85,85,85,85,85,85|M|||81|";
    let golduck_cn_weak = "Golduck||Leftovers|CloudNine|splash,surf|Hardy|85,85,85,85,85,85|M|||81|";
    let snorlax = "Snorlax||Leftovers|ThickFat|splash,bodyslam|Hardy|85,85,85,85,85,85|M|||100|";
    let seed = "0,0,0,5";
    let p2 = format!("{golduck_cn_weak}]{snorlax}");
    let plan = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Surf KOs the 1-HP CN mirror
        ScriptDecision::one(1, Choice::Switch(1)),              // p2 replaces
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
    ];
    let mut ba = Battle::start_with_switchins(&opts_cg(golduck_cn, &p2, seed), &d).expect("start");
    let sta = ba.state_mut().unwrap();
    sta.sides[1].pokemon[0].hp = 1;
    let oa = sta.run_full_battle(&plan, &d);
    assert_eq!(
        // The replacement position-swap moved the corpse to bench slot 1.
        sta.sides[1].pokemon[1].hp, 0,
        "CN2: the Cloud Nine Golduck is KO'd"
    );
    assert_eq!(
        seed_str(&oa.decisions[0].seed_after),
        "52716,62194,16741,29732",
        "CN2: the KO'd Cloud Nine's faint-site onEnd WeatherChange draws (mirror tie)"
    );
    assert_eq!(seed_str(&oa.decisions[1].seed_after), "51244,29580,30032,46327", "CN2 dec1 seed");
    assert_eq!(seed_str(&oa.decisions[2].seed_after), "14518,34098,41111,13110", "CN2 dec2 seed");
}

/// FZ3 (`gen3_ff_frozen_no_absorb_v1`): a FROZEN Flash Fire holder is NOT fire-immune
/// (the resolved `flashfire.onTryHit` returns early for `frz`) — the Flamethrower
/// lands with FULL draws (248 → 192 at this seed), its fire-move thaw CURES the
/// freeze, and the thawed Houndoom's own move runs with NO thaw roll. WRONG
/// (pre-fix): the port kept the frozen holder immune (accuracy-only) then rolled a
/// phantom thaw on its own move — a 3-vs-9-draw desync.
#[test]
fn frozen_flash_fire_holder_is_not_fire_immune() {
    let d = dex();
    let mewtwo = "Mewtwo||Leftovers|Pressure|flamethrower,splash|Hardy|85,85,85,85,85,85|M|||66|";
    let houndoom = "Houndoom||Leftovers|FlashFire|crunch,splash|Hardy|85,85,85,85,85,85|M|||79|";
    let seed = "0,0,0,17";
    let mut ba = Battle::start_with_switchins(&opts_cg(mewtwo, houndoom, seed), &d).expect("start");
    let sta = ba.state_mut().unwrap();
    sta.sides[1].pokemon[0].status = Some(Status::Freeze);
    let oa = sta.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(sta.sides[1].pokemon[0].hp, 192, "FZ3: the Flamethrower HITS the frozen FF holder");
    assert_eq!(sta.sides[1].pokemon[0].status, None, "FZ3: the fire hit THAWS it");
    assert!(!sta.sides[1].pokemon[0].flash_fire, "FZ3: a frozen holder does NOT arm the absorb");
    assert_eq!(
        seed_str(&oa.decisions[0].seed_after),
        "39495,13599,46791,14688",
        "FZ3: full move draws + no phantom thaw roll (sim seed)"
    );
}

/// FN1 (`gen3_fnt_clears_status_v1`): `checkFainted` sets a fainted active's status
/// to `fnt` (para erased) and `faintMessages → clearVolatile` ZEROES its boosts — so
/// a +6-Agility PARALYZED Muk corpse sorts the replacement instaswitch at its PLAIN
/// speed, TYING the foe's plain Muk corpse → the shuffle draw + the resumed tail's
/// Quick Claw. WRONG (pre-fix): the port kept par (×0.25) and the +6 (×4) on the
/// corpse — no tie, a missing draw.
#[test]
fn fainted_replacement_sort_clears_status_and_boosts() {
    let d = dex();
    let muk_a = "Muk||Leftovers|StickyHold|explosion,agility,splash|Hardy|85,85,85,85,85,85|M|||84|";
    let muk_b = "Muk||Leftovers|StickyHold|splash,explosion|Hardy|85,85,85,85,85,85|M|||84|";
    let snorlax = "Snorlax||Leftovers|ThickFat|splash,bodyslam|Hardy|85,85,85,85,85,85|M|||100|";
    let blissey = "Blissey||Leftovers|NaturalCure|splash,icebeam|Hardy|85,85,85,85,85,85|M|||100|";
    let seed = "0,0,0,23";
    let p1 = format!("{muk_a}]{snorlax}");
    let p2 = format!("{muk_b}]{blissey}");
    let plan = [
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Agility +2
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // +4
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // +6
        ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // mutual Explosions → double faint
        ScriptDecision::both(Choice::Switch(1), Choice::Switch(1)), // the double replacement
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];
    let mut ba = Battle::start_with_switchins(&opts_cg(&p1, &p2, seed), &d).expect("start");
    let sta = ba.state_mut().unwrap();
    sta.sides[0].pokemon[0].status = Some(Status::Paralysis);
    let oa = sta.run_full_battle(&plan, &d);
    assert_eq!(
        seed_str(&oa.decisions[3].seed_after),
        "33848,34073,3737,9876",
        "FN1: the mutual-Explosion double-faint boundary (sim seed)"
    );
    assert_eq!(
        seed_str(&oa.decisions[4].seed_after),
        "55569,55220,38151,58406",
        "FN1: the corpses TIE the replacement sort (par + boosts cleared) — the shuffle \
         draw + the resumed Quick Claw"
    );
}

/// TC1 (`gen3_statusimmune_onupdate_cure_v1`): a SLEPT Trace Porygon2 re-enters
/// against an Insomnia Hypno — the TRACED Insomnia's `onUpdate` CURES the sleep at
/// the first Update after the copy (draw-free; the boundary seeds match the sim).
/// WRONG (pre-fix): the port kept the sleep (the A/B status cluster, 9 repros —
/// every STATUS_IMMUNE member carries the onUpdate cure).
#[test]
fn traced_status_immune_ability_cures_the_status_on_update() {
    let d = dex();
    let porygon2 = "Porygon2||Leftovers|Trace|recover,splash|Hardy|85,85,85,85,85,85|N|||80|";
    let snorlax = "Snorlax||Leftovers|ThickFat|splash,bodyslam|Hardy|85,85,85,85,85,85|M|||100|";
    let hypno = "Hypno||Leftovers|Insomnia|hypnosis,splash|Hardy|85,85,85,85,85,85|M|||85|";
    let seed = "0,0,0,29";
    let p1 = format!("{porygon2}]{snorlax}");
    let plan = [
        ScriptDecision::both(Choice::Switch(1), Choice::Move(1)), // slept P2 out
        ScriptDecision::both(Choice::Switch(1), Choice::Move(1)), // P2 back IN → traces Insomnia
        ScriptDecision::both(Choice::Move(1), Choice::Move(1)),
    ];
    let mut ba = Battle::start_with_switchins(&opts_cg(&p1, hypno, seed), &d).expect("start");
    let sta = ba.state_mut().unwrap();
    sta.sides[0].pokemon[0].status = Some(Status::Sleep(3));
    let oa = sta.run_full_battle(&plan, &d);
    assert_eq!(
        sta.sides[0].pokemon[0].status, None,
        "TC1: the traced Insomnia's onUpdate cures the sleep"
    );
    assert_eq!(seed_str(&oa.decisions[1].seed_after), "50345,28109,46919,54214", "TC1 dec1 seed");
    assert_eq!(seed_str(&oa.decisions[2].seed_after), "5616,51371,4668,8673", "TC1 dec2 seed");
}

// ============================================================================
// A/B FIX-QUEUE #4 (2026-07-10, the auto_0709_2205 steady-state 9-repro corpus).
// Ground truth: harness/probe_fixqueue4_regression_rng.js (the resolved sim).
// ============================================================================

/// FQ1 (`gen3_faint_queue_order_v1`): `faintMessages` drains `faintQueue` in
/// ENQUEUE order, fully processing each corpse (`fainted = true`) BEFORE the next
/// corpse's ability-End — so on a mutual Explosion the USER (self-KO'd first in
/// `useMove`) is already a processed corpse when the Cloud Nine TARGET's `onEnd`
/// fires `eachEvent('WeatherChange')`: only the dying holder gathers → NO
/// tie-shuffle even though the two corpses TIE on cached speed (Golduck-L81 184 ==
/// Smeargle-L89 184). WRONG (pre-fix): the port walked SIDE order (the enqueue
/// order was recorded only under logging) → processed the side-0 Cloud Nine corpse
/// FIRST, while the side-1 Smeargle still gathered → a phantom tie draw (the
/// ab_723_13 / ab_464_16 divergences at the explosion turn).
#[test]
fn double_faint_processes_corpses_in_enqueue_order() {
    let d = dex();
    let golduck_cn = "Golduck||Leftovers|CloudNine|splash,surf|Hardy|85,85,85,85,85,85|M|||81|";
    let smeargle_boom = "Smeargle||Leftovers|OwnTempo|explosion,splash|Hardy|85,85,85,85,85,85|M|||89|";
    let snorlax = "Snorlax||Leftovers|ThickFat|splash,bodyslam|Hardy|85,85,85,85,85,85|M|||100|";
    let seed = "0,0,0,41";
    let p1 = format!("{golduck_cn}]{snorlax}");
    let p2 = format!("{smeargle_boom}]{snorlax}");
    let plan = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // splash vs Explosion → double faint
        ScriptDecision::both(Choice::Switch(1), Choice::Switch(1)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];
    let mut ba = Battle::start_with_switchins(&opts_cg(&p1, &p2, seed), &d).expect("start");
    let sta = ba.state_mut().unwrap();
    sta.sides[0].pokemon[0].hp = 1; // the Explosion KO is guaranteed
    let oa = sta.run_full_battle(&plan, &d);
    assert_eq!(
        seed_str(&oa.decisions[0].seed_after),
        "44717,38369,41890,58006",
        "FQ1: the CN corpse's End WeatherChange gathers ALONE (user processed first) — no tie draw"
    );
    assert_eq!(seed_str(&oa.decisions[1].seed_after), "36226,57602,33056,20414", "FQ1 dec1 seed");
    assert_eq!(seed_str(&oa.decisions[2].seed_after), "68,15394,12616,24550", "FQ1 dec2 seed");
}

/// FS1 (`gen3_fainted_no_ability_speed_v1`): a FAINTED mon's ability handlers no
/// longer gather (`faintMessages` sets `isActive = false`), so a Swift Swim CORPSE
/// under rain sorts the replacement instaswitch at its PLAIN `getActionSpeed` —
/// Kingdra-L81: alive-in-rain 368 → fainted 184, TYING the plain 184 Smeargle-L89
/// corpse → the instaswitch shuffle draw + the resumed Quick Claw. WRONG (pre-fix):
/// the port applied the `weather_speed` ×2 to the corpse (368, no tie, a missing
/// draw — the ab_894_12 divergence at the double-replacement decision).
#[test]
fn fainted_swift_swim_corpse_sorts_at_plain_speed() {
    let d = dex();
    let smeargle_boom = "Smeargle||Leftovers|OwnTempo|explosion,splash|Hardy|85,85,85,85,85,85|M|||89|";
    // Rain source = Kyogre's Drizzle (permanent; the ab_894_12 rain source — Rain
    // Dance the MOVE is unmodeled/fail-loud).
    let kyogre = "Kyogre||Leftovers|Drizzle|splash,surf|Hardy|85,85,85,85,85,85|N|||67|";
    let kingdra_ss = "Kingdra||Leftovers|SwiftSwim|splash,icebeam|Hardy|85,85,85,85,85,85|M|||81|";
    let snorlax = "Snorlax||Leftovers|ThickFat|splash,bodyslam|Hardy|85,85,85,85,85,85|M|||100|";
    let blissey = "Blissey||Leftovers|NaturalCure|splash,icebeam|Hardy|85,85,85,85,85,85|M|||100|";
    let seed = "0,0,0,43";
    let p1 = format!("{smeargle_boom}]{snorlax}");
    let p2 = format!("{kyogre}]{kingdra_ss}]{blissey}");
    let plan = [
        ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // Kingdra in (rain permanent)
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Explosion → mutual double faint
        ScriptDecision::both(Choice::Switch(1), Choice::Switch(1)), // the tie shuffle + resumed QC
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];
    let mut ba = Battle::start_with_switchins(&opts_cg(&p1, &p2, seed), &d).expect("start");
    let sta = ba.state_mut().unwrap();
    sta.sides[1].pokemon[1].hp = 1; // bench Kingdra: the Explosion KO is guaranteed
    let oa = sta.run_full_battle(&plan, &d);
    assert_eq!(seed_str(&oa.decisions[1].seed_after), "245,41647,13710,60687", "FS1 KO-turn seed");
    assert_eq!(
        seed_str(&oa.decisions[2].seed_after),
        "29086,50139,23539,23913",
        "FS1: the Swift Swim corpse sorts PLAIN under rain — the 184==184 tie draws the shuffle"
    );
    assert_eq!(seed_str(&oa.decisions[3].seed_after), "23569,46653,24070,43568", "FS1 dec3 seed");
}

/// TX1 (`gen3_tox_stage_persists_v1`, the RESET direction): the gen3 tox stage
/// reset (`tox.onSwitchIn`) fires via the gen4-override runSwitch's
/// `runEvent('SwitchIn')` — so a badly-poisoned Swampert that pivots out and back
/// (its runSwitch RUNS) resumes at stage 1: residuals 22 / [out] / 22 / 44 on a
/// 362-maxhp Swampert (heal-at-full folds to 0). Probe:
/// `harness/probe_tox_stage_switch.js` + the TX1 scenario. WRONG (either way):
/// never resetting (stage 2 → 66 on re-entry) — or resetting at the raw
/// `execute_switch` swap (see TX2, which pins the placement).
#[test]
fn tox_stage_resets_when_the_runswitch_runs() {
    let d = dex();
    let smeargle_tox = "Smeargle||Leftovers|OwnTempo|toxic,splash|Hardy|85,85,85,85,85,85|M|||89|";
    let swampert = "Swampert||Leftovers|Torrent|splash,surf|Hardy|85,85,85,85,85,85|M|||100|";
    let snorlax = "Snorlax||Leftovers|ThickFat|splash,bodyslam|Hardy|85,85,85,85,85,85|M|||100|";
    let seed = "0,0,0,47";
    let p2 = format!("{swampert}]{snorlax}");
    let plan = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Toxic lands → residual stage 1
        ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // Swampert out
        ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // back IN — runSwitch runs → RESET
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
    ];
    let mut ba = Battle::start_with_switchins(&opts_cg(smeargle_tox, &p2, seed), &d).expect("start");
    let sta = ba.state_mut().unwrap();
    let oa = sta.run_full_battle(&plan, &d);
    assert_eq!(sta.sides[1].pokemon[0].status, Some(Status::Toxic(2)), "TX1: stage 2 after the reset+2 residuals");
    assert_eq!(sta.sides[1].pokemon[0].hp, 318, "TX1: 362 -22 [out/in resets] -22 -44 (+heals at full)");
    assert_eq!(seed_str(&oa.decisions[0].seed_after), "26546,5301,19719,45961", "TX1 dec0 seed (Toxic lands)");
    assert_eq!(seed_str(&oa.decisions[3].seed_after), "37709,57954,22601,16642", "TX1 dec3 seed");
}

/// TX2 (`gen3_tox_stage_persists_v1`, the PERSIST direction — pins the reset's
/// PLACEMENT): a replacement whose queued runSwitch is CANCELLED by the gen3
/// faint-cancels-all rule KEEPS its prior tox stage. p2's pre-ramped (stage-2)
/// tox Swampert replaces into a double-faint board alongside p1's 1-HP Electrode;
/// Electrode's runSwitch runs FIRST (faster entrant), its Spikes chip KOs it, and
/// the faint CANCELS Swampert's pending runSwitch → NO SwitchIn event → NO reset:
/// the resumed residual ramps 2→3 (66 chip, 362→296), then 3→4 (88, →230 net of
/// the +22 heal). This is the ab_1166_22 Mew (re-entered tox at 13/263 beside a
/// Spikes-KO'd Entei; its unreset stage-2 chip killed it where the port's stage-1
/// chip left it alive → a phantom endTurn Quick Claw). WRONG (pre-fix): the port
/// reset the stage at `execute_switch`'s array swap, which the cancellation never
/// guards.
#[test]
fn tox_stage_persists_when_the_runswitch_is_cancelled() {
    let d = dex();
    let smeargle_boom = "Smeargle||Leftovers|OwnTempo|explosion,splash|Hardy|85,85,85,85,85,85|M|||89|";
    let smeargle_spk = "Smeargle||Leftovers|OwnTempo|spikes,splash|Hardy|85,85,85,85,85,85|M|||89|";
    let electrode = "Electrode||Leftovers|Static|splash,thunderbolt|Hardy|85,85,85,85,85,85|M|||100|";
    let swampert = "Swampert||Leftovers|Torrent|splash,surf|Hardy|85,85,85,85,85,85|M|||100|";
    let snorlax = "Snorlax||Leftovers|ThickFat|splash,bodyslam|Hardy|85,85,85,85,85,85|M|||100|";
    let blissey = "Blissey||Leftovers|NaturalCure|splash,icebeam|Hardy|85,85,85,85,85,85|M|||100|";
    let seed = "0,0,0,53";
    let p1 = format!("{smeargle_boom}]{electrode}]{snorlax}");
    let p2 = format!("{smeargle_spk}]{swampert}]{blissey}");
    let plan = [
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // splash vs Spikes (p1 side)
        ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Explosion → mutual double faint
        ScriptDecision::both(Choice::Switch(1), Choice::Switch(1)), // Electrode Spikes-KO cancels Swampert's runSwitch
        ScriptDecision::one(0, Choice::Switch(2)),              // Snorlax in; resumed residual ramps 2→3
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];
    let mut ba = Battle::start_with_switchins(&opts_cg(&p1, &p2, seed), &d).expect("start");
    let sta = ba.state_mut().unwrap();
    sta.sides[0].pokemon[1].hp = 1; // Electrode dies to the Spikes chip on entry
    sta.sides[1].pokemon[1].status = Some(Status::Toxic(2)); // pre-ramped badly-poisoned Swampert
    let oa = sta.run_full_battle(&plan, &d);
    // Swampert is p2's active after the replacement (array-swapped to slot 0).
    assert_eq!(sta.sides[1].pokemon[0].status, Some(Status::Toxic(4)), "TX2: the stage kept ramping (no reset)");
    assert_eq!(sta.sides[1].pokemon[0].hp, 230, "TX2: 362 -66 (stage 3) -88+22 (stage 4 + heal)");
    assert_eq!(seed_str(&oa.decisions[2].seed_after), "44414,33024,37304,31868", "TX2 dec2 seed (Electrode Spikes-KO)");
    assert_eq!(seed_str(&oa.decisions[3].seed_after), "27248,29872,61426,6063", "TX2 dec3 seed (persisted stage-3 chip)");
    assert_eq!(seed_str(&oa.decisions[4].seed_after), "55144,51422,45855,40398", "TX2 dec4 seed");
}

// ============================================================================
// HA1/HA2 — the HANDLER-COMPLETENESS AUDIT's two real misses (`gen3_handler_audit_v1`,
//       ground truth `harness/probe_handler_audit_regression_rng.js`; draw models
//       settled by `probe_jumpkick_crash_rng.js` / `probe_freeze_clause_rng.js`).
// ============================================================================

/// HA1: JUMP KICK / HIGH JUMP KICK crash damage (`gen3_jump_kick_crash_v1`). The
/// resolved gen3 `onMoveFail`: a FAILED Jump Kick (an accuracy miss — or a Protect
/// block, HA1b) crashes the USER for `clampIntRange(getDamage/2, 1, floor(target.maxhp/2))`,
/// and the crash's `getDamage` DRAWS the crit + the 16-way damage roll (a missed JK is
/// +2 draws vs a missed control move). WRONG (pre-fix): the port admitted jumpkick as a
/// plain damaging move and did NOTHING on the miss — a silent HP desync AND a 2-draw
/// seed desync. The fix: `turn.rs::apply_jump_kick_crash` at the miss + protect-block
/// returns. GROUND TRUTH: probe_handler_audit_regression_rng.js (sim init seed 164,21,56,26, port seeded at the sim POST-LEAD seed 55936,3900,1794,37637 —
/// Jump Kick misses turn 1; the sim's Hitmonlee ends 116/241 [crash 125] and the
/// post-turn seed is 63948,64443,9701,38473).
#[test]
fn jump_kick_miss_crashes_the_user_with_crit_and_roll_draws() {
    let d = dex();
    let lee = "Hitmonlee|||noability|jumpkick,doublekick|Hardy||N||||";
    let lax = "Snorlax|||noability|splash,tackle|Hardy|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(lee, lax, "55936,3900,1794,37637"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // The miss crashed the USER: Hitmonlee 241 → 116 (crash 125 = floor(rolled/2),
    // within [1, floor(524/2)] — the TARGET's maxhp ceiling); Snorlax untouched.
    assert_eq!(out.decisions[0].active[0].hp, 116, "the missed Jump Kick CRASHES the user 241→116");
    assert_eq!(out.decisions[0].active[1].hp, out.decisions[0].active[1].maxhp, "Snorlax untouched (the move missed)");
    // GROUND-TRUTH SEED: the crash's getDamage drew crit + the damage roll (without
    // them the post-turn seed desyncs by 2 draws — the revert trips this).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "63948,64443,9701,38473",
        "post-turn seed == the real Showdown seed (crash draws crit + roll)"
    );
}

/// HA1b: the crash fires THROUGH A PROTECT BLOCK too (probe D — a JK into a protecting
/// Snorlax is blocked at TryHit yet still crashes the user, +2 draws). Sim init seed 65,7,11,53 (port at post-lead 10122,16917,17677,4268):
/// the sim's Hitmonlee ends 95/241 (crash 146), post-turn seed 61856,40248,34547,19523.
#[test]
fn jump_kick_crashes_through_a_protect_block() {
    let d = dex();
    let lee = "Hitmonlee|||noability|jumpkick,doublekick|Hardy||N||||";
    let lax = "Snorlax|||noability|protect,splash|Hardy|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(lee, lax, "10122,16917,17677,4268"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].hp, 95, "the Protect-blocked Jump Kick CRASHES the user 241→95");
    assert_eq!(out.decisions[0].active[1].hp, out.decisions[0].active[1].maxhp, "Snorlax untouched (Protect blocked)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "61856,40248,34547,19523",
        "post-turn seed == the real Showdown seed (crash through Protect draws crit + roll)"
    );
}

/// HA2: FREEZE CLAUSE MOD (`gen3_freeze_clause_v1`). Under gen3ou (a clause format) a
/// SECOND foe-inflicted freeze on the same side FAILS — the resolved rule's
/// `onSetStatus` returns false INSIDE the SetStatus event (the 2-clause handler-sort
/// shuffle already drew → the block is DRAW-FREE; a fainted mon's status is 'fnt' so
/// only LIVING frozen mons count). WRONG (pre-fix): the port modeled only Sleep Clause
/// — the second freeze landed, a silent STATUS desync under every clause format. The
/// fix: `turn.rs::try_set_status` frz gate + `side_has_frozen`. GROUND TRUTH:
/// probe_handler_audit_regression_rng.js (gen3ou sim init seed 196,28,12,11, port at post-lead 36278,5314,5429,34330 — T1 Ice Beam
/// freezes Snorlax; T2 p2 switches to Blissey; T3's freeze secondary WOULD land but
/// Freeze Clause blocks it; final seed 40823,3352,56752,24696).
#[test]
fn freeze_clause_blocks_the_second_freeze_in_gen3ou() {
    let d = dex();
    let suicune = "Suicune|||noability|icebeam,splash|Hardy||N||||";
    let lax = "Snorlax|||noability|splash,tackle|Hardy|252,,,,,|N||||";
    let bliss = "Blissey|||noability|splash,tackle|Hardy|252,,,,,|N||||";
    let p2 = format!("{lax}]{bliss}");
    let opts = BattleOptions {
        format_id: "gen3ou".to_string(),
        seed: Some("36278,5314,5429,34330".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(suicune.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2) },
    };
    let mut battle = Battle::start_with_switchins(&opts, &d).expect("start");
    let st = battle.state_mut().expect("state");
    assert!(st.sleep_clause, "gen3ou carries the clause formats flag");
    let plan = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),   // T1: Ice Beam freezes Snorlax
        ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // T2: splash / switch Blissey
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),   // T3: freeze secondary → CLAUSE BLOCK
    ];
    let out = st.run_full_battle(&plan, &d);
    assert_eq!(out.decisions[0].active[1].status, Some(Status::Freeze), "T1: Snorlax frozen");
    // T3: the freeze secondary rolled a land (probe) but Freeze Clause BLOCKS it —
    // Blissey stays status-free. Under the revert it would be Frozen.
    assert!(
        out.decisions[2].active[1].status.is_none(),
        "Freeze Clause blocks the SECOND freeze — Blissey stays un-frozen"
    );
    // The benched Snorlax is still the (one) frozen mon.
    let snorlax = st.sides[1].pokemon.iter().find(|m| m.species_id == "snorlax").expect("snorlax");
    assert_eq!(snorlax.status, Some(Status::Freeze), "the first freeze persists on the bench");
    // GROUND-TRUTH SEED: the clause block is DRAW-FREE (the 2-clause shuffle already
    // drew) — a wrongly-applied freeze or a wrong draw count trips this.
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "40823,3352,56752,24696",
        "post-T3 seed == the real Showdown seed (the clause block is draw-free)"
    );
}

/// R15-P1: SLEEP CLAUSE MOD exempts a SELF-inflicted (Rest) sleeper
/// (`gen3_sleep_clause_self_rest_exempt_v1`). The gen-3 `rulesets.ts`
/// `sleepclausemod.onSetStatus` counts an existing sleeper toward the one-foe-asleep cap
/// ONLY when `!pokemon.statusState.source?.isAlly(pokemon)` — a mon that put ITSELF to
/// sleep via **Rest** does NOT count, so a foe sleep move on a DIFFERENT mon still LANDS.
/// WRONG (pre-fix): `side_has_sleeper` counted ANY asleep mon (a stale "Rest is out of
/// scope" assumption from before Rest was modeled), so a benched self-Rested mon blocked
/// the foe's Spore — the Spore fizzled and the target's own move ran instead, drawing ONE
/// EXTRA SetStatus clause shuffle (a real gen3ou pool desync — ab_3_15 @ master-seed
/// 333444 dec 24: p1 Breloom Spores p2 Suicune while p2's benched Zapdos is asleep from
/// its own turn-1 Rest → the port clause-blocked the Spore, Suicune Rested instead, +1
/// draw → seed divergence). The fix: track `MonState::sleep_from_rest` (set in
/// `run_rest`, cleared in `try_set_status_impl`) and skip self-Rest sleepers in
/// `side_has_sleeper`. GROUND TRUTH: `harness/probe_r15_sleepclause.js` /
/// `probe_r15_pin_truth.js` (gen3ou, the sim's first-decision seed 53118,34657,41207,29520
/// — the port constructs to the SAME seed): T1 p1 HP-Rocks + p2 Zapdos self-Rests (→ slp);
/// T2 p2 switches to Suicune while Zapdos sits asleep on the bench + p1 Spores → the Spore
/// LANDS (Suicune slp, foe-inflicted counter) with post-T2 seed 21514,3448,20660,22314.
#[test]
fn sleep_clause_exempts_a_self_rest_sleeper_from_blocking_a_foe_sleep_move() {
    let d = dex();
    let p1 = "Breloom||Leftovers|EffectSpore|Spore,BrickBreak,HiddenPowerRock,FocusPunch|Jolly|,252,,,4,252|M|,,30,,30,30|||,Rock,,,,]Snorlax||Leftovers|Immunity|BodySlam,Earthquake,FocusPunch,SelfDestruct|Adamant|80,176,124,,108,20|M||||";
    let p2 = "Zapdos||Leftovers|Pressure|Rest,Roar,Thunderbolt,Toxic|Calm|200,,,,252,56|N|,0,,,,|||]Suicune||Leftovers|Pressure|Rest,Surf,CalmMind,Roar|Bold|252,,228,,,28|N|,0,,,,|||";
    let opts = BattleOptions {
        format_id: "gen3ou".to_string(),
        seed: Some("53118,34657,41207,29520".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.to_string()) },
    };
    let mut battle = Battle::start_with_switchins(&opts, &d).expect("start");
    let st = battle.state_mut().expect("state");
    assert!(st.sleep_clause, "gen3ou carries the clause formats flag");
    let plan = [
        // T1: p1 Hidden Power Rock chips Zapdos; p2 Zapdos REST (self-sleep + heal).
        ScriptDecision::both(Choice::Move(2), Choice::Move(0)),
        // T2: p1 SPORE; p2 switches to Suicune (Spore hits the switched-in Suicune while
        //     the self-Rested Zapdos sits asleep on the bench).
        ScriptDecision::both(Choice::Move(0), Choice::Switch(1)),
    ];
    let out = st.run_full_battle(&plan, &d);
    // The benched Zapdos is asleep from its OWN Rest (Sleep(3), self-inflicted).
    let zapdos = st.sides[1].pokemon.iter().find(|m| m.species_id == "zapdos").expect("zapdos");
    assert_eq!(zapdos.status, Some(Status::Sleep(3)), "Zapdos self-Rested to sleep");
    assert!(zapdos.sleep_from_rest, "Zapdos's sleep is self-inflicted (Rest)");
    // THE FIX: the self-Rest sleeper does NOT block the foe's Spore — Suicune is asleep
    // (foe-inflicted, so NOT `sleep_from_rest`). Under the revert the clause blocks the
    // Spore and Suicune Rests instead (its own move runs → an extra clause shuffle).
    let suicune = st.sides[1].pokemon.iter().find(|m| m.species_id == "suicune").expect("suicune");
    assert!(
        matches!(suicune.status, Some(Status::Sleep(_))),
        "Spore LANDS on Suicune despite the benched self-Rest sleeper (Sleep Clause exempts Rest)"
    );
    assert!(!suicune.sleep_from_rest, "Suicune's sleep is foe-inflicted (Spore), not Rest");
    // GROUND-TRUTH SEED: the extra clause shuffle a wrongly-blocked Spore would trigger
    // (target's own move runs) shifts this — a wrong draw count trips it.
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "21514,3448,20660,22314",
        "post-T2 seed == the real Showdown seed (the exempt Spore lands with its own counter draw)"
    );
}

// ============================================================================
// P3 — PROTOCOL PHASE 3's two request-boundary fixes (`gen3_protocol_phase3_v1`):
//   P3-1: a scripted VOLUNTARY `switch` to a FAINTED / already-active / out-of-range
//         team slot is REJECTED draw-free (the sim's `chooseSwitch` "can't switch to a
//         fainted Pokémon"), leaving the boundary open — pre-fix the port EXECUTED it
//         (a `|switch|…|0 fnt` phantom entrant).
//   P3-2: choice acceptance is PER-SIDE (the sim's `side.choose`): one side's valid
//         choice is ACCEPTED-and-HELD while the other side's invalid choice is
//         rejected, and the turn later commits with choices accepted at DIFFERENT
//         capture decisions; a re-submission by the already-chosen side is DISCARDED
//         ("You already made choices"). Pre-fix the port skipped the WHOLE decision,
//         mis-mapping the split-accept (midswitch_ability_lines/2's turn 5 ran the
//         held `move 2` + the NEXT decision's p1 move in the sim; the port ran the
//         next decision's BOTH choices — a boundary-mapping divergence).
// Ground truth: the Phase-3 capture golden's DEC rows (the sim's own recorded
// `seedAfter` per submission — rejected rows carry an UNCHANGED seed), scenario
// `midswitch_ability_lines` battle 2, replayed here at the SEED level (the protocol
// gate asserts only the LINE stream; this pin asserts the boundary/seed mapping).
// ============================================================================

/// P3-1: a voluntary switch to a FAINTED slot is skipped draw-free (no turn runs, no
/// boundary records) and the next valid submission runs the real turn. Constructed:
/// Electrode self-KOs with Explosion → Snorlax replaces it (the fainted Electrode now
/// sits at team slot 1) → a scripted `Switch(1)` names the corpse → REJECTED; the
/// following real move turn runs. Under the revert the port EXECUTES the corpse switch
/// (a 4th boundary with a fainted active) — the structure assertions trip.
#[test]
fn rejected_switch_to_a_fainted_slot_is_skipped_draw_free() {
    let d = dex();
    let p1 = "Electrode||NoItem|Static|explosion,thunderbolt|Hasty|,252,,,,252|N||||\
              ]Snorlax||Leftovers|Thick Fat|bodyslam,earthquake|Adamant|252,252,,,,|N||||";
    let p2 = "Snorlax||Leftovers|Thick Fat|bodyslam,earthquake|Adamant|252,252,,,,|N||||\
              ]Blissey||Leftovers|Natural Cure|seismictoss,icebeam|Bold|252,,252,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "11111,22222,33333,44444"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let choices = [
        // dec0: Explosion — Electrode self-KOs (Snorlax survives).
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        // dec1: forced replacement — Snorlax in (the corpse now at slot 1).
        ScriptDecision::one(0, Choice::Switch(1)),
        // dec2: the PHANTOM — p1 switches to the FAINTED Electrode (slot 1). The sim
        // REJECTS it ("can't switch to a fainted Pokémon"), draw-free; p2's move is
        // ACCEPTED-and-HELD (the per-side split, P3-2's mechanism).
        ScriptDecision::both(Choice::Switch(1), Choice::Move(0)),
        // dec3: p1 re-submits a valid move; the turn commits with p2's HELD choice.
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];
    let out = st.run_full_battle(&choices, &d);
    // STRUCTURE: 3 real boundaries (move / forced-switch / move) — the corpse switch is
    // DROPPED, not executed. Under the revert: 4 boundaries, and boundary 2's p1 active
    // would be the FAINTED Electrode.
    assert_eq!(
        out.decisions.len(),
        3,
        "the switch-to-a-fainted-slot decision must be SKIPPED (sim: chooseSwitch reject), \
         not executed as a turn"
    );
    assert_eq!(out.decisions[1].active_species[0], "snorlax", "the forced replacement stands");
    assert_eq!(
        out.decisions[2].active_species[0], "snorlax",
        "the real post-reject turn runs on the LIVING active — a fainted-corpse switch-in \
         here means the reject gate was reverted"
    );
    assert!(
        !st.sides[0].pokemon[st.sides[0].active].fainted,
        "the active after the rejected switch must be alive"
    );
}

/// P3-2: PER-SIDE choice acceptance — the split-accept boundary mapping, at SEED level,
/// against the sim's own recorded DEC rows (capture golden, `midswitch_ability_lines`
/// battle 2: dec5 `{p1: switch→REJECTED, p2: move 2→ACCEPTED-HELD}` with an UNCHANGED
/// seedAfter, then dec6 `{p1: move 1→ACCEPTED, p2: switch→DISCARDED (already chose)}`
/// commits the turn at seed 12356,10054,4144,9146 — the turn ran p2's dec5 choice with
/// p1's dec6 choice). The port replays the FULL submission stream (rejects included)
/// and must land every REAL boundary on the sim's recorded seed. Under the pre-fix
/// whole-decision skip, the turn runs dec6's BOTH choices → the seed stream diverges.
#[test]
fn per_side_choice_acceptance_maps_split_accept_boundaries_to_the_sims_seeds() {
    let d = dex();
    // The scenario's teams + battle 2's init seed + submission stream, from the capture
    // golden's TEAM/INIT/DEC rows (regenerate with gen_protocol_capture.js; these values
    // are the sim's own records for MASTER_SEED 0x50524f54).
    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/vectors/protocol_capture_golden.txt"
    ))
    .expect("protocol capture golden");
    let mut teams: [Option<String>; 2] = [None, None];
    let mut init_seed: Option<String> = None;
    let mut subs: Vec<(String, String, String)> = Vec::new(); // (cp1, cp2, seed_after)
    for line in golden.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f.first().copied() {
            Some("TEAM") if f[1] == "midswitch_ability_lines" => {
                teams[if f[2] == "p1" { 0 } else { 1 }] = Some(f[3].to_string());
            }
            Some("INIT") if f[1] == "midswitch_ability_lines" && f[2] == "2" => {
                init_seed = Some(f[3].to_string());
            }
            Some("DEC") if f[1] == "midswitch_ability_lines" && f[2] == "2" => {
                subs.push((f[7].to_string(), f[8].to_string(), f[9].to_string()));
            }
            _ => {}
        }
    }
    let init_seed = init_seed.expect("battle 2 INIT row");
    assert!(subs.len() >= 15, "battle 2 has a long submission stream, got {}", subs.len());
    // The battle must actually contain a SPLIT-ACCEPT: a rejected row (seed unchanged
    // from its predecessor) whose OTHER side's token is a real choice — else this pin
    // pins nothing (fail loud on a regenerated-away scenario).
    let mut has_split = false;
    for i in 1..subs.len() {
        if subs[i].2 == subs[i - 1].2 && subs[i].0 != "-" && subs[i].1 != "-" {
            has_split = true;
        }
    }
    assert!(has_split, "battle 2 must contain a rejected-while-other-side-chose submission");

    let parse = |tok: &str| -> Option<Choice> {
        if tok == "-" {
            return None;
        }
        let (k, n) = tok.split_at(1);
        let n: usize = n.parse().unwrap();
        Some(match k {
            "m" => Choice::Move(n),
            "s" => Choice::Switch(n),
            _ => panic!("bad token {tok}"),
        })
    };
    let script: Vec<ScriptDecision> = subs
        .iter()
        .map(|(a, b, _)| ScriptDecision { p1: parse(a), p2: parse(b) })
        .collect();
    let opts = opts_cg(teams[0].as_deref().unwrap(), teams[1].as_deref().unwrap(), &init_seed);
    let mut battle = Battle::start_with_switchins(&opts, &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&script, &d);

    // The sim's REAL boundary seeds = the DEC seedAfter stream with consecutive
    // duplicates collapsed (a rejected submission leaves the seed unchanged; every real
    // turn in this battle draws ≥1). The port's recorded boundaries must match 1:1.
    let mut expected: Vec<String> = Vec::new();
    let mut prev = init_seed.clone();
    for (_, _, seed) in &subs {
        if *seed != prev {
            expected.push(seed.clone());
            prev = seed.clone();
        }
    }
    assert_eq!(
        out.decisions.len(),
        expected.len(),
        "port boundary count == the sim's real (seed-advancing) submissions"
    );
    for (i, exp) in expected.iter().enumerate() {
        assert_eq!(
            &seed_str(&out.decisions[i].seed_after),
            exp,
            "boundary {i}: the port's post-decision seed must equal the sim's DEC record \
             (the split-accept mapping — pre-fix the whole-decision skip diverges here)"
        );
    }
}

// ============================================================================
// NICKNAME — every emitted mon-reference token (`p<N>a: <name>`) renders the packed
//   set's ON-FIELD NICKNAME, never the English species name.
//
//   Showdown's `Pokemon.name` = `set.name || species.name`, and EVERY protocol
//   line references a mon by that ident (`|move|<user>|<Move>|<target>`, `|switch|`
//   IDENT, `|-damage|`, `|-ability|`, `|-status|`, `[of]`, …). poke-env tracks each
//   mon by THIS token; if the bridge emits the SPECIES where the ident should be the
//   nickname, poke-env can't match the mon it already knows and tries to ADD it —
//   overflowing to a 7th mon (`p1's team already has 6 pokemons: cannot add p1:
//   Zapdos`), the localized/nicknamed-team crash (a Zapdos nicknamed `Electhor` etc).
//
//   WRONG (pre-fix): `display_name` returned the SPECIES name for the ident, so
//   `|move|p1a: Zapdos|…` was emitted instead of `|move|p1a: Electhor|…`.
//   FIX: `display_name` renders the nickname (`set.name`, ← species only when empty);
//   the SPECIES lives ONLY in the `|switch|` DETAILS field (`species_name`).
// ============================================================================

/// A nicknamed mon (`Electhor` = Zapdos) has EVERY emitted mon-reference token
/// render the NICKNAME `p<N>a: Electhor`, while the `|switch|` DETAILS field keeps
/// the SPECIES `Zapdos`. The species name must NEVER appear as an ident token
/// (`p<N>a: Zapdos`) — that's the exact byte that overflows poke-env's team to a
/// 7th mon. WRONG (pre-fix): the ident rendered the species → the crash.
#[test]
fn nicknamed_mon_renders_nickname_in_every_ident_not_species() {
    let d = dex();
    // p1: `Electhor` (nickname) = Zapdos, Pressure — Thunderbolt hits the foe (so a
    //     `|move|<Electhor>|Thunderbolt|<foe>` + `|-damage|` + a switch-in `|-ability|`
    //     all reference our mon by ident). p2: a plain Snorlax (species == name).
    let electhor = "Electhor|zapdos|leftovers|pressure|thunderbolt|Timid|,,,,,252|,0,,,,|||100|";
    let snorlax = "Snorlax|||immunity|bodyslam|Adamant|252,252,,,,|||||";

    // Both attack each turn; a short scripted battle so the Zapdos is on-field and
    // referenced as user, target, and switch-in.
    let script = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
    ];
    let mut battle =
        Battle::start_with_switchins(&opts_cg(electhor, snorlax, "40263,34842,41812,24710"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let (_out, lines) = st.run_full_battle_logged(&script, &d);
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();

    // (a) our mon IS referenced by ident somewhere (guard against a vacuous pass).
    assert!(
        raw.iter().any(|l| l.contains("p1a: Electhor")),
        "the nicknamed Zapdos must be referenced by its nickname ident `p1a: Electhor` \
         at least once; emitted lines:\n{}",
        raw.join("\n")
    );

    // (b) THE PIN — the SPECIES name must NEVER appear as an IDENT token. The ident is
    //     always `p<N>a: <name>`; a `p1a: Zapdos` / `p2a: Zapdos` token is the exact
    //     byte that overflows poke-env's team. (The species IS allowed in the `|switch|`
    //     DETAILS field, checked in (c) — that's a different, non-`pNa:` position.)
    for l in &raw {
        assert!(
            !l.contains("p1a: Zapdos") && !l.contains("p2a: Zapdos"),
            "no emitted line may render the SPECIES `Zapdos` as an ident token \
             (`p<N>a: Zapdos`) — poke-env keys the mon by its nickname `Electhor`, so \
             the species ident overflows its team to a 7th mon. Offending line: {l:?}"
        );
    }

    // (c) the `|switch|` line proves the split: IDENT = nickname, DETAILS = species.
    let switch_line = raw
        .iter()
        .find(|l| l.starts_with("|switch|p1a:"))
        .expect("p1's lead switch-in line is emitted");
    assert!(
        switch_line.starts_with("|switch|p1a: Electhor|Zapdos"),
        "the `|switch|` line must be `|switch|p1a: Electhor|Zapdos|<hp>` — the IDENT is \
         the nickname `Electhor`, the DETAILS the species `Zapdos`. Got: {switch_line:?}"
    );
}

// ============================================================================
// MOVE-COVERAGE BATCH 1 (`gen3_move_coverage_batch1_v1`) — the DRAW-FREE post-hit effects
// (RECOIL / DRAIN / ITEM-REMOVAL / RAPID-SPIN) + the SELF-DROP that draws ONE random(100).
// Ground truth from `harness/probe_batch1_regression_rng.js` (raw seed [11,22,33,44]; the
// genderless leads construct to the init seed "57388,452,34593,29177" — Tauros has a gender
// ratio so MC1 constructs to a different init seed "18464,3966,47670,60926").
//
// **THE DRAW-FREE PROOF**: MC1b (Rock-Head recoil) / MC2 (drain) / MC4 (knock-off) / MC4b
// (sticky-hold) / MC6 (rapid-spin) ALL share seedAfter "4448,587,55846,30246" — recoil/drain/
// item/rapid-spin add NO PRNG draw, so they produce the IDENTICAL post-turn seed as each other
// on the shared init seed. The SELF-DROP scenario (MC3) has a DIFFERENT seedAfter — it draws
// the selfDrops random(100). Each pin is revert-verified (removing its effect's wiring flips
// the state; skipping the self-drop draw flips MC3's seed off the shared draw-free value).
// ============================================================================

/// MC1: Double-Edge recoils `floor(dmgDealt/3)` to the USER (`recoil:[1,3]`), DRAW-FREE.
/// WRONG (pre-fix): the recoil was never applied (the USER took no recoil). STATE (Tauros HP
/// includes the recoil) + SEED (the post-turn seed == the real Showdown seed).
#[test]
fn double_edge_recoils_a_third_of_the_damage_dealt() {
    let d = dex();
    let tauros = "Tauros|||sturdy|doubleedge|Adamant|,252,,,,252|||||";
    let snorlax = "Snorlax|||immunity|pound|Careful|252,,,252,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(tauros, snorlax, "18464,3966,47670,60926"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1);
    // Tauros (maxhp 291) took the Double-Edge recoil (Snorlax was KO'd this turn). Ground
    // truth: 117/291 — the recoil chipped it. A no-recoil model leaves Tauros near-full.
    assert_eq!(out.decisions[0].active[0].hp, 117, "Tauros HP includes the DE recoil (dealt/3)");
    assert!(!out.decisions[0].active[0].fainted, "Tauros survives the recoil");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "29587,16389,4131,29123",
        "recoil is DRAW-FREE → the post-turn seed matches the real Showdown seed"
    );
}

/// MC1b: Rock Head NEGATES Double-Edge recoil — the USER takes NO recoil (only the foe's
/// chip). WRONG (a model that applied recoil regardless of ability): Aggron's HP would be
/// lower. STATE (Aggron near-full HP) + SEED (== the SHARED draw-free seed — recoil is
/// draw-free either way).
#[test]
fn rock_head_negates_double_edge_recoil() {
    let d = dex();
    let aggron = "Aggron|||rockhead|doubleedge|Adamant|,252,,,,252|||||";
    let snorlax = "Snorlax|||immunity|pound|Careful|252,,,252,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(aggron, snorlax, "57388,452,34593,29177"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Aggron (maxhp 281) took ONLY Snorlax's Pound — NO recoil. Ground truth 274/281.
    assert_eq!(out.decisions[0].active[0].hp, 274, "Rock Head → Aggron takes NO Double-Edge recoil");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "4448,587,55846,30246",
        "recoil (or its Rock-Head negation) is DRAW-FREE → the SHARED draw-free seed (== MC2/MC4/MC4b/MC6)"
    );
}

/// MC2: Giga Drain heals the USER `floor(dmgDealt/2)` (`drain:[1,2]`), DRAW-FREE. WRONG
/// (pre-fix): the drain heal was never applied. STATE (the injured Sceptile heals) + SEED
/// (== the SHARED draw-free value — drain draws nothing).
#[test]
fn giga_drain_heals_half_the_damage_dealt() {
    let d = dex();
    let sceptile = "Sceptile|||overgrow|gigadrain|Modest|,,,252,,252|||||";
    let snorlax = "Snorlax|||immunity|pound|Careful|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(sceptile, snorlax, "57388,452,34593,29177"), &d)
            .expect("start");
    // Injure Sceptile to 80 so the heal is visible (mirrors the golden/probe inject).
    st_set_hp_b1(battle.state_mut().unwrap(), 0, 80);
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Sceptile healed the drain (dealt/2), then took Snorlax's Pound. Ground truth 65/281.
    assert_eq!(out.decisions[0].active[0].hp, 65, "Sceptile HP includes the Giga Drain heal (dealt/2)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "4448,587,55846,30246",
        "drain is DRAW-FREE → the SHARED draw-free seed (== MC1b/MC4/MC4b/MC6)"
    );
}

/// MC3: Overheat drops the USER's SpA by 2 (`move.self.boosts {spa:-2}`) AND gen3 `selfDrops`
/// DRAWS ONE `random(100)` (the `secondaryRoll`) — the drop applies unconditionally
/// (`self.chance === undefined`) but the roll fires. STATE (Charizard spa -2) + SEED (a
/// DIFFERENT post-turn seed from the physical scenarios — proving the extra `random(100)`;
/// WRONG (pre-fix): skipping the draw would give the shared draw-free "4448,...", desyncing).
#[test]
fn overheat_self_drops_spa_and_draws_the_selfdrops_random_100() {
    let d = dex();
    let charizard = "Charizard|||blaze|overheat|Modest|,,,252,,252|||||";
    let snorlax = "Snorlax|||immunity|pound|Careful|252,,,,252,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(charizard, snorlax, "57388,452,34593,29177"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Charizard's SpA dropped -2 (boosts index 2 == spa).
    assert_eq!(out.decisions[0].active[0].boosts[2], -2, "Overheat self-drops SpA by 2");
    // Ground truth: the selfDrops random(100) makes THIS seed differ from the draw-free ones.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "43673,61326,59799,37313",
        "the selfDrops random(100) is DRAWN → a DIFFERENT seed than the draw-free scenarios; \
         skipping it would give the shared draw-free \"4448,...\" (the pre-fix bug)"
    );
}

/// MC4: Knock Off removes the TARGET's item (`onAfterHit`, gen3 no dmg boost), DRAW-FREE.
/// WRONG (pre-fix): the item was never removed (the target kept it). STATE (Snorlax item gone)
/// + SEED (== the SHARED draw-free value).
#[test]
fn knock_off_removes_the_targets_item_draw_free() {
    let d = dex();
    let ttar = "Tyranitar|||sandstream|knockoff|Adamant|,252,,,,252|||||";
    let snorlax = "Snorlax||leftovers|immunity|pound|Careful|252,,4,252,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(ttar, snorlax, "57388,452,34593,29177"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // Pre-condition: Snorlax holds Leftovers.
    assert_eq!(st.sides[1].pokemon[0].item, "Leftovers", "Snorlax starts with Leftovers");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(st.sides[1].pokemon[st.sides[1].active].item, "", "Knock Off REMOVED Snorlax's item");
    assert!(!out.decisions[0].active[1].item_held, "Snorlax no longer holds an item");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "4448,587,55846,30246",
        "Knock Off's onAfterHit item removal is DRAW-FREE → the SHARED draw-free seed"
    );
}

/// MC4b: Knock Off is BLOCKED by Sticky Hold — the target KEEPS its item. WRONG (a model that
/// ignored Sticky Hold): the item would be removed. STATE (Muk keeps Leftovers) + SEED (==
/// the SHARED draw-free value — the block is draw-free too).
#[test]
fn knock_off_blocked_by_sticky_hold() {
    let d = dex();
    let ttar = "Tyranitar|||sandstream|knockoff|Adamant|,252,,,,252|||||";
    let muk = "Muk||leftovers|stickyhold|pound|Careful|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(ttar, muk, "57388,452,34593,29177"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        st.sides[1].pokemon[st.sides[1].active].item, "Leftovers",
        "Sticky Hold BLOCKS Knock Off — Muk KEEPS its Leftovers"
    );
    assert!(out.decisions[0].active[1].item_held, "Muk still holds an item");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "4448,587,55846,30246",
        "the Sticky-Hold-blocked Knock Off is DRAW-FREE → the SHARED draw-free seed"
    );
}

/// MC5: Thief STEALS the target's item when the attacker holds NONE — the attacker GAINS it,
/// the target LOSES it. DRAW-FREE. WRONG (pre-fix): no steal (both items unchanged). STATE
/// (Gengar gains Leftovers, Snorlax loses it) + SEED.
#[test]
fn thief_steals_the_targets_item_when_attacker_is_itemless() {
    let d = dex();
    let gengar = "Gengar|||levitate|thief|Timid|,,,252,,252|||||";
    let snorlax = "Snorlax||leftovers|immunity|pound|Careful|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(gengar, snorlax, "57388,452,34593,29177"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    assert_eq!(st.sides[0].pokemon[0].item, "", "Gengar starts itemless");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        st.sides[0].pokemon[st.sides[0].active].item, "Leftovers",
        "Thief → the itemless Gengar STOLE Snorlax's Leftovers"
    );
    assert_eq!(st.sides[1].pokemon[st.sides[1].active].item, "", "Snorlax LOST its Leftovers");
    assert!(out.decisions[0].active[0].item_held, "Gengar now holds the stolen item");
    assert!(!out.decisions[0].active[1].item_held, "Snorlax no longer holds an item");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "60833,51486,28767,2196",
        "Thief's steal is DRAW-FREE (Gengar's construction seed differs from the physical leads)"
    );
}

/// MC6: Rapid Spin clears the USER's OWN Spikes + Leech Seed (`onAfterHit` +
/// `onAfterSubDamage`), DRAW-FREE. WRONG (pre-fix): the hazards/leech persisted. STATE (p1's
/// spikes 3→0 + its leech cleared) + SEED (== the SHARED draw-free value).
#[test]
fn rapid_spin_clears_the_users_own_spikes_and_leech_seed() {
    let d = dex();
    let forretress = "Forretress|||sturdy|rapidspin|Relaxed|252,,252,,,|||||";
    let snorlax = "Snorlax|||immunity|pound|Careful|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(forretress, snorlax, "57388,452,34593,29177"), &d)
            .expect("start");
    {
        // Inject 3 Spikes on p1's side + a Leech Seed on p1's Forretress (seeded by p2) —
        // the board Rapid Spin must clear (STATE-only, no PRNG).
        let st = battle.state_mut().expect("state");
        st.sides[0].spikes = 3;
        let active = st.sides[0].active;
        st.sides[0].pokemon[active].leech_seed = Some(1);
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(st.sides[0].spikes, 0, "Rapid Spin CLEARED the user's own Spikes (3→0)");
    assert!(
        st.sides[0].pokemon[st.sides[0].active].leech_seed.is_none(),
        "Rapid Spin CLEARED the user's own Leech Seed"
    );
    assert!(!out.decisions[0].active[0].leech_seeded, "Forretress is no longer leech-seeded");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "4448,587,55846,30246",
        "Rapid Spin's clear is DRAW-FREE → the SHARED draw-free seed"
    );
}

/// MC7: gen3 `itemKnockedOff` — a mon whose item was KNOCKED OFF can neither have its item
/// TAKEN nor GAIN one (the sim's `takeItem` returns false in gen≤4 if the target OR the source
/// has `itemKnockedOff`). So a Thief by a Knocked-Off attacker does NOTHING (no removal, no
/// gain). WRONG (pre-fix): the port's Thief stole the item, giving the Knocked-Off attacker a
/// Leftovers it then healed with (the e2e_83 real-team bug — Skarmory Thief'd + wrongly healed).
/// STATE (Skarmory stays itemless / Snorlax keeps its item) + SEED. Ground truth
/// `harness/probe_batch1_regression_rng.js` (the itemKnockedOff scenario).
#[test]
fn knocked_off_attacker_thief_takes_nothing() {
    let d = dex();
    // Skarmory holds Leftovers + has Thief; Snorlax has Knock Off (removes Skarmory's item).
    // Move order: Skarmory [protect, thief] → Protect=Move(0)/Thief=Move(1); Snorlax [knockoff,
    // pound] → Knock Off=Move(0)/Pound=Move(1). (Protect is a modeled gen-3 filler; on dec0 it
    // resolves BEFORE the Knock Off — but Knock Off has `protect:1` so Protect BLOCKS it!). Use a
    // non-protecting filler instead: Skarmory Spikes (draw-free, modeled) so the Knock Off lands.
    let skarmory = "Skarmory||leftovers|keeneye|spikes,thief|Impish|252,,252,,,|||||";
    let snorlax = "Snorlax||leftovers|immunity|knockoff,pound|Adamant|252,252,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(skarmory, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // Skarmory moves [roost, thief] → Roost=Move(0), Thief=Move(1); Snorlax moves [knockoff,
    // pound] → Knock Off=Move(0), Pound=Move(1). dec0: Skarmory Roost, Snorlax Knock Off →
    // removes Skarmory's Leftovers + sets `item_knocked_off`. dec1: Skarmory Thief, Snorlax
    // Pound → the Thief does NOTHING (Skarmory is Knocked-Off; Snorlax keeps its Leftovers).
    let out = st.run_full_battle(&[
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(1), Choice::Move(1)),
    ], &d);
    assert_eq!(out.decisions.len(), 2);
    // After the Knock Off (dec0): Skarmory has no item AND is flagged Knocked-Off.
    assert_eq!(st.sides[0].pokemon[0].item, "", "Knock Off removed Skarmory's Leftovers");
    assert!(st.sides[0].pokemon[0].item_knocked_off, "Skarmory's slot is flagged itemKnockedOff");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "29587,16389,4131,29123",
        "the Knock Off dec0 seed"
    );
    // After the Thief (dec1): the Knocked-Off Skarmory could NOT gain the item; Snorlax KEEPS it.
    assert_eq!(st.sides[0].pokemon[st.sides[0].active].item, "", "the Knocked-Off Skarmory gained NOTHING");
    assert_eq!(
        st.sides[1].pokemon[st.sides[1].active].item, "Leftovers",
        "Snorlax KEEPS its Leftovers — the Thief by a Knocked-Off mon does nothing"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "13227,40747,44602,26856",
        "the Thief-does-nothing dec1 seed (the item op is DRAW-FREE)"
    );
}

/// MC8: RECOIL is computed on the POST-Focus-Band damage (`move.totalDamage`). When a Focus
/// Band SAVES the target from a KO by a recoil move, the recoil is `floor((hp-1)/den)`, NOT
/// `floor(hp/den)` off the full lethal roll. WRONG (pre-fix): `dealt` was captured BEFORE the
/// Focus Band survive-at-1 reduction, so the attacker over-recoiled by the FB-clipped amount
/// (a code-review-found latent bug — reachable on real teams since both recoil [batch-1] and
/// Focus Band [batch-4] are e2e-admitted). STATE (Tauros HP) + SEED. The FB roll draws at a
/// probed seed; the recoil itself is draw-free.
#[test]
fn recoil_is_computed_on_the_post_focus_band_damage() {
    let d = dex();
    let tauros = "Tauros||silkscarf|sturdy|doubleedge|Adamant|,252,,,,252|||||";
    // Snorlax holds Focus Band; injected to 60 HP so Double-Edge is LETHAL and the 1/10 FB
    // roll (which passes at this seed) saves it at 1 HP.
    let snorlax = "Snorlax||focusband|immunity|splash,pound|Careful|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(tauros, snorlax, "11645,6730,59128,17195"), &d)
            .expect("start");
    {
        let st = battle.state_mut().expect("state");
        let a = st.sides[1].active;
        st.sides[1].pokemon[a].hp = 60;
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Snorlax survives at 1 HP (Focus Band); the damage DEALT is 59 → recoil floor(59/3)=19,
    // so Tauros (maxhp 291) takes 19 → 272. A pre-fix model (recoil off the full lethal roll,
    // clamped to 60) would recoil floor(60/3)=20 → Tauros 271.
    assert_eq!(out.decisions[0].active[1].hp, 1, "Focus Band saved Snorlax at 1 HP");
    assert!(!out.decisions[0].active[1].fainted, "Snorlax did not faint");
    assert_eq!(
        out.decisions[0].active[0].hp, 272,
        "recoil is floor((hp-1)/3)=floor(59/3)=19 off the POST-Focus-Band damage (272), NOT \
         floor(60/3)=20 off the full lethal roll (271)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "49711,14316,42044,43950",
        "the FB roll drew; the recoil itself is draw-free → the real Showdown seed"
    );
}

/// Helper: set side `s`'s active mon HP (a STATE-only inject, no PRNG) for the drain pin.
fn st_set_hp_b1(st: &mut pokesim::state::BattleState, s: usize, hp: u16) {
    let active = st.sides[s].active;
    st.sides[s].pokemon[active].hp = hp;
}

// ============================================================================
// MOVE-COVERAGE BATCH 2 (`gen3_move_coverage_batch2_v1`) — the DRAW-FRIENDLY status-move
// classes: STATUS-CURE / WEATHER-SET / STAT-DROP / SCREENS. Ground truth from
// `harness/probe_batch2_regression_rng.js` (raw seed [11,22,33,44]; the genderless leads
// construct to init "13127,45333,18295,15391" for the Vaporeon/Electrode/Persian/Blissey
// scenarios, and "18464,3966,47670,60926" for the Vileplume/Snorlax Aromatherapy team).
//
// **THE DRAW MODEL**: the cures / distinct-speed weather-set / screens are DRAW-FREE (the
// SET turn produces the shared draw-free seed "57388,452,34593,29177" or the scenario's own
// draw-free value); the stat-drops draw ONE accuracy roll; and — the CRUX — a physical hit
// into a side with BOTH Reflect AND Light Screen up draws ONE EXTRA `random(0,2)` (the
// `runEvent('ModifyDamagePhase1')` handler-sort shuffle: the 2 screen `onAnyModifyDamage
// Phase1` handlers tie), so MC17 (both screens) and its ONE-screen control produce DIFFERENT
// seeds. Each pin is revert-verified.
// ============================================================================

/// MC9: Refresh self-cures paralysis (par/psn/brn), DRAW-FREE. WRONG (pre-fix / if the cure
/// arm is removed): Vaporeon stays paralyzed (its `onHit` would fail-loud or no-op). STATE
/// (Vaporeon un-paralyzed) + SEED (the draw-free post-turn seed).
#[test]
fn refresh_cures_self_paralysis_draw_free() {
    let d = dex();
    let vaporeon = "Vaporeon|||waterabsorb|refresh,surf|Serious|,,,252,,|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(vaporeon, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    st_set_status_b2(battle.state_mut().unwrap(), 0, Status::Paralysis);
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        out.decisions[0].active[0].status, None,
        "Refresh cured Vaporeon's paralysis"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "57388,452,34593,29177",
        "Refresh's cure is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC10: Heal Bell cures the WHOLE team (active + bench) but SKIPS a Soundproof ally,
/// DRAW-FREE. WRONG (pre-fix / if the team-cure arm is missing): the active tox persists;
/// WRONG (if the Soundproof skip is dropped): the bench Electrode's par would be cured too.
/// STATE (Miltank un-tox'd; the Soundproof Electrode bench KEEPS its par) + SEED (draw-free).
#[test]
fn heal_bell_cures_team_but_skips_a_soundproof_ally() {
    let d = dex();
    let p1 = "Miltank|||thickfat|healbell,bodyslam|Serious|252,,,,,|||||]\
              Electrode|||soundproof|thunderbolt|Serious|252,,,,,|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    {
        let st = battle.state_mut().unwrap();
        st.sides[0].pokemon[0].status = Some(Status::Toxic(0)); // active Miltank
        st.sides[0].pokemon[1].status = Some(Status::Paralysis); // bench Soundproof Electrode
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].status, None, "Heal Bell cured the active Miltank's tox");
    // The Soundproof bench ally is SKIPPED — it keeps its paralysis (read the state directly,
    // it's not the active). A model without the Soundproof gate would cure it.
    assert_eq!(
        st.sides[0].pokemon[1].status,
        Some(Status::Paralysis),
        "Heal Bell SKIPS the Soundproof bench Electrode → it KEEPS its paralysis"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "18464,3966,47670,60926",
        "Heal Bell's cure is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC11: Aromatherapy cures the WHOLE team (active + bench) via `clearStatus` — NO Soundproof
/// gate (Aromatherapy is not a sound move). DRAW-FREE. STATE (active + bench both cleared) +
/// SEED. WRONG (pre-fix): the team-cure arm missing → the brn/slp persist.
#[test]
fn aromatherapy_cures_the_whole_team_draw_free() {
    let d = dex();
    let p1 = "Vileplume|||chlorophyll|aromatherapy,gigadrain|Serious|252,,,,,|||||]\
              Snorlax|||thickfat|bodyslam|Serious|252,,,,,|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, snorlax, "18464,3966,47670,60926"), &d)
            .expect("start");
    {
        let st = battle.state_mut().unwrap();
        st.sides[0].pokemon[0].status = Some(Status::Burn); // active Vileplume
        st.sides[0].pokemon[1].status = Some(Status::Sleep(2)); // bench Snorlax
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].status, None, "Aromatherapy cured the active Vileplume's burn");
    assert_eq!(st.sides[0].pokemon[1].status, None, "Aromatherapy cured the bench Snorlax's sleep too");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "57388,452,34593,29177",
        "Aromatherapy's cure is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC12: Rain Dance sets a 5-turn TIMED Rain (distinct speed → DRAW-FREE). WRONG (pre-fix):
/// the weather-set arm missing → weather stays clear (fail-loud). STATE (weather Rain, turns
/// 4 after the first upkeep) + SEED (draw-free at distinct speed).
#[test]
fn rain_dance_sets_a_five_turn_timed_rain_draw_free() {
    let d = dex();
    let electrode = "Electrode|||noability|raindance,thunderbolt|Serious|,,,,,252|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(electrode, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].weather, Some(Weather::Rain), "Rain Dance set Rain");
    assert_eq!(
        out.decisions[0].weather_turns, 4,
        "the 5-turn timer ticked once at the end-of-turn field residual → 4 remaining"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "18464,3966,47670,60926",
        "at distinct speed the weather-set draws NO eachEvent('WeatherChange') shuffle → draw-free seed"
    );
}

/// MC13: Rain Dance into an ALREADY-active Rain FAILS (`setWeather` returns false for a MOVE
/// source when `this.weather === status.id`), DRAW-FREE, the weather (incl. its permanent
/// duration) UNCHANGED. STATE (weather still permanent Rain, turns 0) + SEED. WRONG (a model
/// that re-set the timer): the weather would become a 5-turn timed Rain.
#[test]
fn rain_dance_into_an_already_active_rain_fails_draw_free() {
    let d = dex();
    let electrode = "Electrode|||noability|raindance,thunderbolt|Serious|,,,,,252|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(electrode, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    {
        // Inject a PERMANENT Rain (the Drizzle-style board — weather_turns 0).
        let st = battle.state_mut().unwrap();
        st.field.weather = Some(Weather::Rain);
        st.field.weather_turns = 0;
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].weather, Some(Weather::Rain), "the weather is still Rain");
    assert_eq!(
        out.decisions[0].weather_turns, 0,
        "the Rain Dance FAILED into the same weather → the PERMANENT (turns 0) Rain is unchanged"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "18464,3966,47670,60926",
        "the failed weather-set is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC14: Screech drops the foe's Def by 2 (`statDropBoosts {def:-2}`) after its accuracy
/// roll. STATE (Snorlax Def -2) + SEED (the accuracy roll drew). WRONG (pre-fix): the
/// stat-drop arm missing → Screech fail-louds.
#[test]
fn screech_drops_the_foe_defense_by_two_after_its_accuracy_roll() {
    let d = dex();
    let persian = "Persian|||limber|screech,slash|Serious|,,,,,252|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(persian, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[1].boosts[1], -2, "Screech dropped Snorlax's Def by 2 (boosts idx 1 == def)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "57388,452,34593,29177",
        "Screech draws its accuracy roll (then a draw-free boost) → the real Showdown seed"
    );
}

/// MC15: Screech BLOCKED by Clear Body — the accuracy roll is STILL drawn, but no drop lands.
/// STATE (Metagross Def stays 0) + SEED (the accuracy roll drew, same as a landed drop). WRONG
/// (a model ignoring Clear Body): Metagross Def would be -2.
#[test]
fn screech_blocked_by_clear_body_draws_accuracy_but_no_drop() {
    let d = dex();
    let persian = "Persian|||limber|screech,slash|Serious|,,,,,252|||||";
    let metagross = "Metagross|||clearbody|meteormash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(persian, metagross, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[1].boosts[1], 0, "Clear Body BLOCKED the Screech drop → Def stays 0");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "3932,55062,24613,55040",
        "the accuracy roll STILL draws (the block is at the boost apply) → the real Showdown seed"
    );
}

/// MC16: Light Screen sets a 5-turn SIDE condition, DRAW-FREE. STATE (p1 light_screen 4 after
/// one residual tick) + SEED (draw-free). WRONG (pre-fix): the screen arm missing → fail-loud.
#[test]
fn light_screen_sets_a_five_turn_side_condition_draw_free() {
    let d = dex();
    let blissey = "Blissey|||naturalcure|lightscreen,softboiled|Serious|252,,252,,,|||||";
    let snorlax = "Snorlax|||immunity|pound|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(blissey, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        out.decisions[0].light_screen[0], 4,
        "Light Screen set duration 5, ticked once at the side residual → 4 remaining on p1's side"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "29587,16389,4131,29123",
        "Light Screen is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC17: THE DOUBLE-SCREEN `ModifyDamagePhase1` SHUFFLE (the crux) — a physical hit into a
/// side with BOTH Reflect AND Light Screen up draws ONE extra `random(0,2)` (the 2 screen
/// `onAnyModifyDamagePhase1` handlers tie → a size-2 Fisher-Yates shuffle). WRONG (pre-fix):
/// the shuffle was NOT drawn → the seed matched the ONE-screen control. This pin captures the
/// TWO-screen seed AND asserts it DIFFERS from the ONE-screen control (the extra draw).
#[test]
fn double_screen_physical_hit_draws_the_modify_damage_phase1_shuffle() {
    let d = dex();
    let blissey = "Blissey|||naturalcure|softboiled|Serious|252,,252,,,|||||";
    let snorlax = "Snorlax|||immunity|pound|Serious|252,,,,,|||||";
    // TWO screens up on p1's side (Reflect + Light Screen).
    let mut both =
        Battle::start_with_switchins(&opts_cg(blissey, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    {
        let st = both.state_mut().unwrap();
        st.sides[0].reflect = 5;
        st.sides[0].light_screen = 5;
    }
    let out_both = both
        .state_mut()
        .unwrap()
        .run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        seed_str(&out_both.decisions[0].seed_after),
        "55318,8071,46680,56242",
        "with BOTH screens up, the physical Pound draws the ModifyDamagePhase1 shuffle → the \
         real Showdown seed"
    );

    // ONE screen up (Reflect only) — NO tie, NO shuffle. Its seed must DIFFER from the above.
    let mut one =
        Battle::start_with_switchins(&opts_cg(blissey, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    {
        let st = one.state_mut().unwrap();
        st.sides[0].reflect = 5;
    }
    let out_one = one
        .state_mut()
        .unwrap()
        .run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        seed_str(&out_one.decisions[0].seed_after),
        "29587,16389,4131,29123",
        "with ONE screen up, NO ModifyDamagePhase1 tie → NO shuffle draw → a DIFFERENT seed"
    );
    assert_ne!(
        seed_str(&out_both.decisions[0].seed_after),
        seed_str(&out_one.decisions[0].seed_after),
        "the double-screen shuffle draw MUST make the two seeds differ (the crux — reverting the \
         shuffle would make them equal)"
    );
}

/// The CONFUSION self-hit × ModifyDamagePhase1 SCREEN SHUFFLE
/// (`gen3_confusion_self_hit_screen_shuffle_v1`, the random-mode byte-fuzz find ab_3_17 @
/// master-seed 100125). gen-4 confusion (gen-3-inherited) runs the FULL `getDamage(self,self,40)`
/// → `modifyDamage` → `runEvent('ModifyDamagePhase1')`, which GATHERS the screens'
/// `onAnyModifyDamagePhase1` SIDE handlers exactly like a normal hit — once per side across BOTH
/// sides. So ≥2 screens (here Reflect on p1 + Light Screen on p2) TIE → a size-2 Fisher-Yates
/// speed-sort shuffle draws ONE `random(0,2)` DURING the self-hit, AFTER the confusion
/// `randomChance(1,2)` and BEFORE the `random(16)` randomizer. WRONG (pre-fix):
/// `apply_confusion_self_hit` never called `modify_damage_phase1_shuffle` → that draw was MISSING →
/// the post-turn seed diverged whenever ≥2 screens were up during a confusion self-hit. Ground
/// truth `harness/probe_confusion_screen_shuffle_rng.js` (C-2SCR / C-NONE, seed 2,2,2,2).
#[test]
fn confusion_self_hit_with_two_screens_draws_the_modify_damage_phase1_shuffle() {
    let d = dex();
    // A confused Snorlax (counter injected) Splashes (a draw-free move) but SELF-HITS on its
    // confusion randomChance(1,2). Reflect on p1 + Light Screen on p2 → 2 tied ModifyDamagePhase1
    // handlers gather at the self-hit → the shuffle. reflect(onSideResidualOrder 1)/lightscreen(2)
    // DIFFER → NO residual tie, so the seed isolates ONLY the confusion self-hit shuffle. Blissey
    // Splashes (draw-free). Seed = the probe's post-construction seedBefore for raw [2,2,2,2].
    let snorlax = "Snorlax|||immunity|splash|Adamant|252,252,,,,|||||";
    let blissey = "Blissey|||naturalcure|splash|Bold|252,,252,,,|||||";
    let mut two =
        Battle::start_with_switchins(&opts_cg(snorlax, blissey, "52170,17908,58343,8292"), &d)
            .expect("start");
    let st2 = two.state_mut().unwrap();
    st2.sides[0].pokemon[0].confusion = Some(4); // matches the probe's injected confusion time=4
    st2.sides[0].reflect = 5;
    st2.sides[1].light_screen = 5;
    let hp_before = st2.sides[0].pokemon[0].hp;
    let out_two =
        st2.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out_two.decisions.len(), 1, "one move boundary");
    assert!(
        st2.sides[0].pokemon[0].hp < hp_before,
        "the confusion self-hit fired (Snorlax HP dropped from {hp_before})"
    );
    // GROUND TRUTH (probe C-2SCR): draws = confusion randomChance(1,2) + the MDP1 shuffle + the
    // self-hit random(16) + Quick Claw. Reverting the shuffle drops a draw → this seed diverges.
    assert_eq!(
        seed_str(&out_two.decisions[0].seed_after),
        "10126,34169,19989,9144",
        "the confusion self-hit with 2 screens draws the ModifyDamagePhase1 shuffle → the real \
         Showdown seed"
    );

    // CONTROL: NO screens — the self-hit gathers 0 screen handlers → NO shuffle → one FEWER draw →
    // a DIFFERENT seed (== the SUB3 no-screen confusion self-hit, the C-NONE ground truth).
    let mut none =
        Battle::start_with_switchins(&opts_cg(snorlax, blissey, "52170,17908,58343,8292"), &d)
            .expect("start");
    let stn = none.state_mut().unwrap();
    stn.sides[0].pokemon[0].confusion = Some(4);
    let out_none =
        stn.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        seed_str(&out_none.decisions[0].seed_after),
        "57951,31032,5281,7249",
        "with NO screens, the confusion self-hit draws no ModifyDamagePhase1 shuffle → a different seed"
    );
    assert_ne!(
        seed_str(&out_two.decisions[0].seed_after),
        seed_str(&out_none.decisions[0].seed_after),
        "the confusion self-hit shuffle draw MUST make the two seeds differ (the crux — reverting \
         the shuffle would make them equal)"
    );
}

/// The BEAT UP strike × FOCUS BAND draw (`gen3_beatup_focus_band_v1`, random-mode byte-fuzz find
/// ab_7_7 @ master-seed 200724). gen-4's `focusband.onDamage` puts `randomChance(1,10)` FIRST in
/// its `&&`, so a Focus Band holder draws the roll on EVERY move-damage hit (JS short-circuit),
/// not only lethal ones. A Beat Up STRIKE runs the full `spreadMoveHit` → `spreadDamage` →
/// `runEvent('Damage')` → the Focus Band handler, exactly like the single-hit path (`run_move`)
/// and the generic `[2,5]` multihit (`run_multihit`). `run_beat_up` used to apply its strike
/// damage DIRECTLY (no `focus_band_damage`), so a Beat Up strike into a FB holder MISSED the draw
/// → a one-fewer-draw desync. Ground truth `harness/probe_beatup_focus_band_rng.js`.
#[test]
fn beat_up_strike_into_a_focus_band_holder_draws_the_focus_band_roll() {
    let d = dex();
    // p1 Houndour is a 1-MON team → Beat Up strikes exactly ONCE (only the active is healthy).
    // p2 Snorlax (bulky) survives the typeless small-BP strike (non-lethal), so the FB roll draws
    // but the survive branch is not taken. Both use move slot 0 (Beat Up / Splash — draw-free).
    // Seed = the probe's seedBefore.
    let houndour = "Houndour|||flashfire|beatup,splash|Adamant|,252,,,,|||||";
    let snorlax_fb = "Snorlax||FocusBand|immunity|splash|Careful|252,,,,252,|||||";
    let mut two =
        Battle::start_with_switchins(&opts_cg(houndour, snorlax_fb, "53303,35262,36397,29520"), &d)
            .expect("start");
    let out = two
        .state_mut()
        .unwrap()
        .run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // GROUND TRUTH (probe FB): draws = Beat Up whole-move acc + strike crit + strike random(16) +
    // the Focus Band randomChance(1,10) + Quick Claw. Reverting the fix drops the FB draw → the
    // seed collapses to the no-item control below.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "10162,51001,19049,38839",
        "a Beat Up strike into a Focus Band holder draws the randomChance(1,10) → the real \
         Showdown seed"
    );

    // CONTROL: no Focus Band → the strike's Damage event has no Focus Band handler → NO
    // randomChance(1,10) → one FEWER draw → a DIFFERENT seed.
    let snorlax_none = "Snorlax|||immunity|splash|Careful|252,,,,252,|||||";
    let mut none =
        Battle::start_with_switchins(&opts_cg(houndour, snorlax_none, "53303,35262,36397,29520"), &d)
            .expect("start");
    let out_n = none
        .state_mut()
        .unwrap()
        .run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        seed_str(&out_n.decisions[0].seed_after),
        "29814,3811,34909,65508",
        "no Focus Band → no FB roll → a different seed"
    );
    assert_ne!(
        seed_str(&out.decisions[0].seed_after),
        seed_str(&out_n.decisions[0].seed_after),
        "the Beat Up strike Focus Band draw MUST make the two seeds differ (the crux — reverting \
         the fix would make them equal)"
    );
}

/// ENDURE does NOT clamp a FUTURE-MOVE resolve (`gen3_futuremove_no_endure_v1`, the random-mode
/// byte-fuzz find ab_19_1 @ master-seed 200724). endure's `duration: 1` means it is only active
/// the turn it is used and EXPIRES at THAT turn's residual BEFORE the order-11 future-move resolve
/// (the LAST residual handler), so the enduring mon takes the FULL stored damage and FAINTS — the
/// sim emits NO `|-activate|move: Endure` (golden ab_19_1: an Enduring Vaporeon dies to a lethal
/// Future Sight resolve). The port used to (CLASS-INFERRED, "not probe-verified") clamp the resolve
/// with endure → the mon survived at 1 HP → a decision-stream divergence. STATE pin (endure_clamp is
/// draw-free, so the seed is unchanged either way — the fainted-vs-survive-at-1 STATE is the crux).
#[test]
fn endure_does_not_clamp_a_future_move_resolve() {
    let d = dex();
    // p1 is a 1-MON Snorlax (so the KO ends the battle cleanly). It USES Endure this turn (move
    // slot 0 — a first Endure always succeeds, draw-free) so the endure volatile is ACTIVE at the
    // residual (injecting it pre-turn wouldn't work — the turn-top clear_flinch drops it). A pending
    // Future Sight sits on p1's slot (targeting p1's active), resolving THIS turn's residual
    // (duration 1 → 0) with LETHAL stored damage. p2 Blissey Splashes (a foe action, so Endure's
    // willAct gate passes).
    let snorlax = "Snorlax|||immunity|endure,splash|Careful|252,,,252,,|||||";
    let blissey = "Blissey|||naturalcure|splash|Bold|252,,252,,,|||||";
    let mut b =
        Battle::start_with_switchins(&opts_cg(snorlax, blissey, "1,2,3,4"), &d).expect("start");
    let st = b.state_mut().unwrap();
    let src_uid = st.sides[1].pokemon[0].uid;
    st.sides[0].future_move = Some(pokesim::state::FutureMove {
        duration: 1,
        damage: 9999, // lethal; apply_damage saturates the mon to 0
        move_id: "futuresight".to_string(),
        accuracy: 100, // resolve always hits (still draws its accuracy roll)
        source_side: 1,
        source_uid: src_uid,
    });
    let _out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // The future-move resolve KO's the enduring Snorlax — endure does NOT clamp it. Reverting the
    // fix (re-adding `endure_clamp` to the future-move resolve) → endure clamps → Snorlax survives
    // at 1 HP (hp == 1, not fainted).
    assert_eq!(
        st.sides[0].pokemon[0].hp, 0,
        "the future-move resolve KO's the enduring mon — endure does NOT clamp a resolve"
    );
    assert!(
        st.sides[0].pokemon[0].fainted,
        "the enduring mon FAINTED from the future-move resolve"
    );
}

/// Helper: set side `s`'s active mon major status (a STATE-only inject, no PRNG).
fn st_set_status_b2(st: &mut pokesim::state::BattleState, s: usize, status: Status) {
    let active = st.sides[s].active;
    st.sides[s].pokemon[active].status = Some(status);
}

// ============================================================================
// MOVE-COVERAGE BATCH 3 pins (MC18…MC29, `gen3_move_coverage_batch3_v1`) — CURSE /
// WISH / BATON PASS, each a CONSTRUCTED gen3customgame board reseeded to the RAW seed
// "13127,45333,18295,15391" (the port's draw-free `start_with_switchins` aligns), each
// revert-verified (each FAILS when its class's engine wiring is disabled). Ground truth:
// `harness/probe_batch3_regression_rng.js`; the draw model was settled by
// `harness/probe_batch3_{curse,wish,batonpass}.js`.
//
// THE DRAW MODEL: the NON-GHOST curse draws ONE `random(100)` (the `selfDrops` roll, like
// Overheat); GHOST curse / Wish / Baton Pass are draw-free (a wish-mirror at equal speed
// draws ONE tie-shuffle). The residual ORDER is the highest-risk item: Wish fires at order
// 7 (BEFORE Leftovers order 10 + the burn DoT); the curse chip fires at order 10 subOrder 8.
// ============================================================================

/// MC18: CURSE non-ghost self-boost {atk:+1, def:+1, spe:-1}. WRONG (if the non-ghost curse
/// arm is removed): Snorlax stays un-boosted (the fail-loud guard would panic). STATE (the
/// mixed +/- boosts) + SEED (the ONE selfDrops `random(100)` — reverting the draw desyncs).
#[test]
fn curse_non_ghost_self_boosts_atk_def_and_drops_spe() {
    let d = dex();
    let snorlax = "Snorlax|||immunity|curse,bodyslam|Serious|252,,,,252,|||||";
    let blissey = "Blissey|||naturalcure|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax, blissey, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // atk+1, def+1, spa 0, spd 0, spe-1 — the mixed self-boost.
    assert_eq!(
        &out.decisions[0].active[0].boosts[0..5],
        &[1, 1, 0, 0, -1],
        "non-ghost Curse self-boosts +1 Atk / +1 Def / -1 Spe"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "57388,452,34593,29177",
        "non-ghost Curse draws ONE selfDrops random(100) → the real Showdown post-turn seed"
    );
}

/// MC19: CURSE ghost pays floor(maxhp/2) HP + lays the `curse` volatile on the FOE. WRONG (if
/// the ghost arm is missing): no HP cost, no curse. STATE (Gengar −maxhp/2; Snorlax cursed) +
/// SEED (the ghost curse is DRAW-FREE).
#[test]
fn curse_ghost_pays_half_hp_and_lays_the_curse_on_the_foe() {
    let d = dex();
    let gengar = "Gengar|||levitate|curse,shadowball|Serious|,,,252,,252|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(gengar, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Gengar (maxhp 261) pays floor(261/2)=130 → hp 131.
    assert_eq!(out.decisions[0].active[0].hp, 131, "ghost Curse pays floor(maxhp/2) HP");
    assert!(out.decisions[0].curse[1], "the FOE (Snorlax) is now cursed");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "18464,3966,47670,60926",
        "ghost Curse (lay + HP cost) is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC20: the CURSE residual chips the cursed foe floor(maxhp/4)/turn, DRAW-FREE. WRONG (if the
/// residual is missing): the foe never chips. STATE (Snorlax loses maxhp/4 on the resolve
/// turn) + SEED.
#[test]
fn curse_residual_chips_the_cursed_foe_a_quarter_maxhp() {
    let d = dex();
    let gengar = "Gengar|||levitate|curse,shadowball|Serious|,,,252,,252|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(gengar, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // curse (lay)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // shadowball; the curse chips
        ],
        &d,
    );
    // Snorlax maxhp 524 → cursed → loses floor(524/4)=131 at the turn-1 residual → 393;
    // at the turn-2 residual → 262.
    assert_eq!(out.decisions[0].active[1].hp, 393, "curse chip 1: 524 → 393 (−131 = maxhp/4)");
    assert_eq!(out.decisions[1].active[1].hp, 262, "curse chip 2: 393 → 262 (−131)");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "3932,55062,24613,55040",
        "the curse chip is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC21: a GHOST re-curse into an ALREADY-CURSED foe FAILS ([still]+-fail), DRAW-FREE, no HP
/// cost, the volatile unchanged. WRONG (if the re-curse guard is missing): Gengar pays HP
/// again. STATE (Gengar's HP unchanged on the 2nd curse; Snorlax still cursed) + SEED.
#[test]
fn curse_recurse_into_an_already_cursed_foe_fails_draw_free() {
    let d = dex();
    let gengar = "Gengar|||levitate|curse,shadowball|Serious|,,,252,,252|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(gengar, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // curse (lay) → Gengar 131
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // curse again → FAILS, no HP cost
        ],
        &d,
    );
    // dec 1 (the re-curse): Gengar's HP must be UNCHANGED from dec 0 (no 2nd HP cost) — the
    // only HP change is the curse chip on Snorlax; Gengar took no Splash damage.
    assert_eq!(out.decisions[0].active[0].hp, 131, "the FIRST curse paid maxhp/2 → Gengar 131");
    assert_eq!(
        out.decisions[1].active[0].hp, 131,
        "the RE-CURSE fails (no 2nd HP cost) → Gengar stays 131"
    );
    assert!(out.decisions[1].curse[1], "Snorlax is still cursed after the failed re-curse");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "57388,452,34593,29177",
        "the re-curse fail is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC22: WISH heals floor(maxhp/2) at the END of the turn AFTER cast (the N+1 heal). WRONG (if
/// the Wish residual is missing): the heal never fires. STATE (Blissey +floor(maxhp/2) on the
/// resolve turn) + SEED (draw-free).
#[test]
fn wish_heals_half_maxhp_the_turn_after_cast() {
    let d = dex();
    let blissey = "Blissey|||naturalcure|wish,splash|Serious|252,,,,,|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(blissey, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    battle.state_mut().unwrap().sides[0].pokemon[0].hp = 100; // low HP so the heal is visible
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Wish (cast) — no heal yet
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Splash — the Wish resolves
        ],
        &d,
    );
    // dec 0: cast, wish pending (duration 1 after the cast-turn residual decrement), no heal.
    assert_eq!(out.decisions[0].active[0].hp, 100, "no heal on the cast turn");
    assert_eq!(out.decisions[0].wish_pending[0], 1, "wish pending (2 → 1 at the cast-turn residual)");
    // dec 1: the Wish resolves → heal floor(714/2)=357 → 100 + 357 = 457.
    assert_eq!(out.decisions[1].active[0].hp, 457, "Wish heals floor(maxhp/2)=357 at N+1 → 457");
    assert_eq!(out.decisions[1].wish_pending[0], 0, "the Wish resolved → cleared");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "57388,452,34593,29177",
        "the Wish heal is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC23 (CRITICAL): the WISH RESIDUAL-ORDER pin — a LIFE/DEATH order test. A low-HP Snorlax
/// under SANDSTORM (chips maxhp/16 at order 8) with a Wish pending: the Wish heal at ORDER 7
/// fires BEFORE the sand chip, so on the resolve turn the mon HEALS first (survives), then
/// takes the chip. If the Wish were slotted at order 11 (after the sand chip), the low-HP mon
/// would be KO'd by the sand chip FIRST → `wish.onEnd`'s `!target.fainted` guard skips the
/// heal → the mon is DEAD. So the ORDER-7 slot is the difference between life and death here.
/// STATE (the mon SURVIVES with the exact post-order HP) + SEED. Reverting Wish to order 11
/// makes this mon faint (the assertion fails).
#[test]
fn wish_residual_fires_at_order_7_saving_a_low_hp_mon_from_the_sand_chip() {
    let d = dex();
    let snorlax = "Snorlax|||immunity|wish,splash|Serious|252,,,,,|||||";
    let blissey = "Blissey|||naturalcure|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax, blissey, "13127,45333,18295,15391"), &d)
            .expect("start");
    {
        let st = battle.state_mut().unwrap();
        // Permanent sandstorm + a low-HP Snorlax (survives one cast-turn chip, then relies on
        // the order-7 Wish to survive the resolve turn).
        st.field.weather = Some(pokesim::state::Weather::Sand);
        st.field.weather_turns = 0; // permanent (ability-style inject)
        st.sides[0].pokemon[0].hp = 40;
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Wish cast
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Splash — the Wish resolves
        ],
        &d,
    );
    // dec 0 (cast turn): sand chip maxhp/16 (524/16=32): 40 - 32 = 8, survives.
    assert_eq!(out.decisions[0].active[0].hp, 8, "cast turn: sand chip 40 → 8 (survives)");
    // dec 1 (resolve turn): Wish +262 (order 7, BEFORE the sand chip) → 8 + 262 = 270, THEN
    //   sand chip −32 → 238. The mon SURVIVES because the heal fired FIRST. With Wish at order
    //   11 the sand chip (order 8) would KO the 8-HP mon before the heal → faint (no heal).
    assert!(!out.decisions[1].active[0].fainted, "the order-7 Wish heal SAVED the low-HP mon");
    assert_eq!(
        out.decisions[1].active[0].hp, 238,
        "resolve turn: Wish(+262, order 7 FIRST) THEN sand chip(−32) → 238"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "57388,452,34593,29177",
        "the Wish/sand residual chain is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC24: a 2nd WISH while one is pending FAILS ([still], DRAW-FREE), the existing Wish
/// untouched (it resolves that turn normally). WRONG (if double-Wish is allowed): the 2nd Wish
/// resets the timer. STATE (the pending Wish resolves on schedule) + SEED.
#[test]
fn wish_double_cast_fails_and_the_pending_wish_resolves() {
    let d = dex();
    let blissey = "Blissey|||naturalcure|wish,splash|Serious|252,,,,,|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(blissey, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    battle.state_mut().unwrap().sides[0].pokemon[0].hp = 100;
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Wish (cast)
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Wish AGAIN — FAILS; the 1st resolves
        ],
        &d,
    );
    // dec 1: the 1st Wish resolves (+357 → 457); the 2nd Wish FAILED (no new pending set until
    // the residual? — the 2nd cast is rejected, and NO fresh Wish exists this turn's end).
    assert_eq!(out.decisions[1].active[0].hp, 457, "the pending (1st) Wish resolved → 457");
    assert_eq!(
        out.decisions[1].wish_pending[0], 0,
        "the 2nd Wish FAILED (no fresh pending set) → the slot is clear after the 1st resolved"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "57388,452,34593,29177",
        "the double-Wish fail is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC25: WISH is SLOT-KEYED — it survives the wisher switching out. p1 Wishes, switches to
/// Chansey; the Wish resolves this turn's end onto WHOEVER is in the slot. WRONG (if Wish is a
/// mon volatile): switching out would clear it. STATE (Chansey active, the slot survived) + SEED.
#[test]
fn wish_is_slot_keyed_and_survives_a_switch() {
    let d = dex();
    let p1 = "Blissey|||naturalcure|wish,splash|Serious|252,,,,,|||||]\
              Chansey|||naturalcure|splash|Serious|252,,,,,|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    battle.state_mut().unwrap().sides[0].pokemon[0].hp = 100;
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Wish
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)), // switch to Chansey; Wish resolves
        ],
        &d,
    );
    // dec 1: Chansey is now active AND the Wish resolved onto it (it entered full 704, so the
    // heal is silent — the slot-key survival is proven by the wish_pending clearing + Chansey
    // active). A mon-volatile Wish would have been cleared when Blissey switched out (never
    // resolving).
    assert_eq!(out.decisions[1].active_species[0], "chansey", "Chansey is active after the switch");
    assert_eq!(out.decisions[1].wish_pending[0], 0, "the slot-keyed Wish RESOLVED (survived the switch)");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "57388,452,34593,29177",
        "the slot-keyed Wish resolve is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC26: BATON PASS boost transfer. Jolteon Swords Dances (+2 Atk), Baton Passes to Snorlax
/// which inherits the +2 Atk. WRONG (if copyVolatileFrom's boosts aren't applied): Snorlax
/// enters at +0. STATE (Snorlax atk +2 after the pass) + SEED (the pass + forced switch are
/// draw-free).
#[test]
fn baton_pass_transfers_the_boosts_to_the_entrant() {
    let d = dex();
    let p1 = "Jolteon|||voltabsorb|swordsdance,batonpass|Serious|,252,,,,252|||||]\
              Snorlax|||immunity|bodyslam|Serious|252,252,,,,|||||";
    let blissey = "Blissey|||naturalcure|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, blissey, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Swords Dance (+2 Atk)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Baton Pass → forced switch
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)), // switch to Snorlax (the pass entrant)
        ],
        &d,
    );
    // dec 2 (the forced switch to Snorlax): Snorlax inherits the +2 Atk.
    assert_eq!(out.decisions[2].active_species[0], "snorlax", "Snorlax is the Baton-Pass entrant");
    assert_eq!(
        out.decisions[2].active[0].boosts[0], 2,
        "the +2 Atk passed to Snorlax (copyVolatileFrom.boosts)"
    );
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "57388,452,34593,29177",
        "the Baton-Pass switch-in is DRAW-FREE (the copy adds no PRNG) → the real Showdown seed"
    );
}

/// MC27: BATON PASS substitute transfer. Jolteon Subs (sub HP floor(maxhp/4)=83), Baton Passes;
/// the SUB HP transfers to Snorlax. WRONG (if the sub isn't copied): Snorlax enters with no
/// sub. STATE (Snorlax sub HP 83 after the pass) + SEED.
#[test]
fn baton_pass_transfers_the_substitute_to_the_entrant() {
    let d = dex();
    let p1 = "Jolteon|||voltabsorb|substitute,batonpass|Serious|252,,,,,252|||||]\
              Snorlax|||immunity|bodyslam|Serious|252,252,,,,|||||";
    let blissey = "Blissey|||naturalcure|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, blissey, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Substitute (sub HP 83)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Baton Pass → forced switch
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)), // switch to Snorlax
        ],
        &d,
    );
    // dec 2: Snorlax inherits the substitute (HP 83 = floor(Jolteon 334/4)).
    assert_eq!(out.decisions[2].active_species[0], "snorlax", "Snorlax is the entrant");
    assert_eq!(out.decisions[2].sub_hp[0], 83, "the SUB HP (83) passed to Snorlax");
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "57388,452,34593,29177",
        "the Baton-Pass sub transfer is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC28: BATON PASS leech-seed transfer. Meganium seeds Jolteon; Jolteon Agilities (+2 Spe)
/// then Baton Passes; the SEED (+ the +2 Spe) transfers to Snorlax (the seeder keeps draining
/// the new mon). WRONG (if leech_seed isn't copied): Snorlax enters unseeded. STATE (Snorlax
/// leech-seeded + drained after the pass) + SEED.
#[test]
fn baton_pass_transfers_the_leech_seed_to_the_entrant() {
    let d = dex();
    let p1 = "Jolteon|||voltabsorb|agility,batonpass|Serious|252,,,,,252|||||]\
              Snorlax|||immunity|bodyslam|Serious|252,252,,,,|||||";
    let meganium = "Meganium|||overgrow|leechseed,splash|Serious|252,,,252,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, meganium, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Agility; Meganium seeds Jolteon
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // Baton Pass → forced switch
            ScriptDecision::both(Choice::Switch(1), Choice::Move(1)), // switch to Snorlax
        ],
        &d,
    );
    // dec 2 (the forced switch): Snorlax is leech-seeded (the seed passed) AND drained
    // (the leech residual fires on it) — and carries the +2 Spe from the Agility.
    assert_eq!(out.decisions[2].active_species[0], "snorlax", "Snorlax is the entrant");
    assert_eq!(out.decisions[2].active[0].boosts[4], 2, "the +2 Spe (Agility) passed to Snorlax");
    // Snorlax leech-seeded → drained maxhp/8 at the switch turn's residual: 524 − 65 = 459.
    assert_eq!(out.decisions[2].active[0].hp, 459, "Snorlax leech-drained maxhp/8 (the seed passed)");
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "3932,55062,24613,55040",
        "the Baton-Pass leech transfer is DRAW-FREE → the real Showdown post-turn seed"
    );
}

/// MC29: BATON PASS with NO eligible bench FAILS ([still]+-fail, DRAW-FREE) — the move still
/// "counts" as used (NOT_FAIL). WRONG (if the no-bench guard is missing): a phantom switch
/// request. STATE (Jolteon stays active, unchanged) + SEED (the fail is draw-free — the Quick
/// Claw still draws since no switch pause).
#[test]
fn baton_pass_with_no_bench_fails_draw_free() {
    let d = dex();
    let jolteon = "Jolteon|||voltabsorb|batonpass,thunderbolt|Serious|,,,252,,252|||||";
    let blissey = "Blissey|||naturalcure|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(jolteon, blissey, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Baton Pass fails (no bench) → Jolteon stays active, no switch request.
    assert!(matches!(out.decisions[0].request, RequestKind::Move), "no forced switch — the BP FAILED");
    assert_eq!(out.decisions[0].active_species[0], "jolteon", "Jolteon stays active (the pass failed)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "18464,3966,47670,60926",
        "the no-bench Baton-Pass fail is DRAW-FREE → the real Showdown post-turn seed"
    );
}

// ============================================================================
// MOVE-COVERAGE BATCH 4 pins (MC30…MC35, `gen3_move_coverage_batch4_v1`) — FOCUS PUNCH +
// PURSUIT, each a CONSTRUCTED gen3customgame board reseeded to the RAW seed
// "44317,42357,9927,48760" (the port's draw-free `start_with_switchins` aligns), each
// revert-verified (each FAILS when its class's engine wiring is disabled). Ground truth:
// `harness/probe_batch4_movecoverage_regression_rng.js`; the draw model was settled by
// `harness/probe_batch4_{focuspunch,pursuit}.js`.
//
// THE DRAW MODEL: the `beforeTurnMove` (order 5) is draw-free but adds a per-action Update tail
// (draws on a speed TIE) + the focuspunch/pursuit `duration: 1` volatiles register a residual
// duration handler (a MIRROR at equal speed adds one residual tie-shuffle). A CANCELLED Focus
// Punch (lostFocus) draws NOTHING (the onTry cancel precedes accuracy). The PURSUIT INTERRUPT
// strikes the switching mon at ×2 BP + NEVER-MISS (crit + damage, NO accuracy) BEFORE the switch
// resolves; a NORMAL Pursuit (foe stays) is a plain bp-40 hit (acc + crit + dmg).
// ============================================================================

/// MC30: FOCUS PUNCH CANCELLED — the foe Tackles the FP user first (FP is priority -3 → moves
/// last) → lostFocus → the move's onTry CANCELS the punch draw-free BEFORE accuracy. WRONG (if
/// the onTry cancel is missing): Focus Punch would land + damage Snorlax. STATE (Snorlax
/// UNHARMED — FP dealt 0; Machamp took the Tackle) + SEED (the FP is draw-free; the turn draws
/// only the Tackle acc/crit/dmg + Quick Claw).
#[test]
fn focus_punch_cancelled_by_a_prior_hit_draws_nothing() {
    let d = dex();
    let machamp = "Machamp||||focuspunch,seismictoss|Serious|252,252,,,,|||||";
    let snorlax = "Snorlax|||immunity|tackle|Serious|,252,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(machamp, snorlax, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Snorlax is UNHARMED — Focus Punch was cancelled (lostFocus) and dealt no damage.
    assert_eq!(out.decisions[0].active[1].hp, 461, "Focus Punch cancelled → Snorlax takes NO damage");
    // Machamp took the Tackle (313/384).
    assert_eq!(out.decisions[0].active[0].hp, 313, "Machamp took the Tackle that cancelled its Focus Punch");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37635,3740,64462,10380",
        "the cancelled Focus Punch is DRAW-FREE (only Tackle acc/crit/dmg + Quick Claw) → the real Showdown post-turn seed"
    );
}

/// MC31: FOCUS PUNCH LANDS — the foe Splashes (non-damaging) → the user keeps focus → Focus
/// Punch executes (the beforeTurnMove laid the volatile; the onTry did NOT cancel). WRONG (if
/// the beforeTurnMove/onTry model is broken): the wrong draw count. STATE (the Blissey is KO'd
/// by the 150-BP Fighting hit) + SEED (the FP acc/crit/dmg + the beforeTurnMove tie-shuffles).
#[test]
fn focus_punch_lands_when_the_user_keeps_focus() {
    let d = dex();
    let machamp = "Machamp||||focuspunch,splash|Serious|252,252,,,,|||||";
    let blissey = "Blissey|||naturalcure|splash|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(machamp, blissey, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Focus Punch KOs the Blissey (150 BP Fighting super-effective vs Normal).
    assert_eq!(out.decisions[0].active[1].hp, 0, "Focus Punch lands → Blissey KO'd");
    assert!(out.decisions[0].active[1].fainted, "Blissey fainted to the landed Focus Punch");
    assert_eq!(out.decisions[0].pokemon_left[1], 0, "Blissey was the last mon → p1 wins");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "5621,5056,41416,14688",
        "the landed Focus Punch's acc/crit/dmg + the beforeTurnMove tie-shuffles → the real Showdown post-turn seed"
    );
}

/// MC32: PURSUIT INTERRUPT — the foe VOLUNTARILY switches → the pursuer STRIKES the switching
/// mon at ×2 BP + NEVER-MISS (crit + damage, NO accuracy) BEFORE the switch resolves, then the
/// replacement comes in. WRONG (if the interrupt is missing): Pursuit runs at its normal turn
/// (bp 40, no strike) and the switch is uninterrupted. STATE (the SWITCHER Jolteon takes the ×2
/// chip → 160/271; the replacement Snorlax is now active; p2 still has 2 mons) + SEED (the strike
/// is never-miss → no accuracy draw; the turn draws crit + damage + Quick Claw).
#[test]
fn pursuit_interrupt_strikes_the_switcher_at_double_bp_never_miss() {
    let d = dex();
    let ttar = "Tyranitar|||pressure|pursuit,crunch|Serious|,252,,252,,|||||";
    let p2 = "Jolteon|||voltabsorb|thunderbolt|Serious|,,,,,252|||||]Snorlax|||immunity|bodyslam|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(ttar, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[ScriptDecision { p1: Some(Choice::Move(0)), p2: Some(Choice::Switch(1)) }],
        &d,
    );
    // The replacement (Snorlax) is now active + full HP; p2 still has 2 mons.
    assert_eq!(out.decisions[0].active_species[1], "snorlax", "the replacement switched in after the interrupt");
    assert_eq!(out.decisions[0].active[1].hp, 524, "the replacement Snorlax entered at full HP");
    assert_eq!(out.decisions[0].pokemon_left[1], 2, "no faint — the interrupt only chipped the switcher (now on the bench)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "10897,43434,54578,10901",
        "the Pursuit interrupt is NEVER-MISS (no accuracy draw): crit + damage + Quick Claw → the real Showdown post-turn seed"
    );
}

/// MC33: NORMAL PURSUIT (the foe STAYS in) — a plain bp-40 Dark hit (acc + crit + dmg), NO ×2,
/// NO interrupt. WRONG (if the beforeTurnMove/normal path is broken): the wrong draw count. STATE
/// (Snorlax takes the small bp-40 chip → 468/524; no switch) + SEED (Pursuit acc/crit/dmg +
/// Body Slam acc/crit/dmg/secondary + Quick Claw). Contrast MC32's ×2 never-miss strike.
#[test]
fn pursuit_normal_when_the_foe_stays_is_a_plain_bp40_hit() {
    let d = dex();
    let ttar = "Tyranitar|||pressure|pursuit,crunch|Serious|,252,,252,,|||||";
    let p2 = "Snorlax|||immunity|bodyslam|Serious|252,252,,,,|||||]Blissey|||naturalcure|softboiled|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(ttar, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Snorlax stays in and takes the plain bp-40 Pursuit chip (56 dmg, NOT the ×2 interrupt).
    assert_eq!(out.decisions[0].active_species[1], "snorlax", "the foe stayed in (no interrupt)");
    assert_eq!(out.decisions[0].active[1].hp, 468, "a NORMAL Pursuit is a plain bp-40 hit → Snorlax 524 → 468");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "5621,5056,41416,14688",
        "a normal Pursuit draws acc + crit + dmg (like any bp-40 move) + the Body Slam + Quick Claw → the real Showdown post-turn seed"
    );
}

/// MC34: PURSUIT KOs the SWITCHER — a low-HP mon switches into a Pursuit that KOs it (Dark
/// super-effective vs Ghost); the ALREADY-CHOSEN switch STILL brings in the replacement (the
/// gen 2-4 `-hint`), and the turn completes (Quick Claw drawn). WRONG (if the pursuitfaint path
/// is wrong): the replacement is lost / pokemon_left desyncs. STATE (the Gengar fainted →
/// pokemon_left 1; the Snorlax replacement is active at full HP) + SEED.
#[test]
fn pursuit_that_kos_the_switcher_still_brings_in_the_replacement() {
    let d = dex();
    let ttar = "Tyranitar|||pressure|pursuit,crunch|Serious|,252,,252,,|||||";
    let p2 = "Gengar|||levitate|shadowball|Serious|,,,,,252|||||]Snorlax|||immunity|bodyslam|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(ttar, p2, "44317,42357,9927,48760"), &d).expect("start");
    // Inject Gengar to 40 HP (STATE-only, no PRNG — mirrors the probe's `b.sides[1].active[0].hp = 40`).
    {
        let st = battle.state_mut().expect("state");
        let active = st.sides[1].active;
        st.sides[1].pokemon[active].hp = 40;
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[ScriptDecision { p1: Some(Choice::Move(0)), p2: Some(Choice::Switch(1)) }],
        &d,
    );
    // Gengar was KO'd by the ×2 interrupt, but the replacement Snorlax STILL switched in.
    assert_eq!(out.decisions[0].active_species[1], "snorlax", "the replacement STILL comes in after a pursuitfaint");
    assert_eq!(out.decisions[0].active[1].hp, 524, "the Snorlax replacement entered at full HP");
    assert_eq!(out.decisions[0].pokemon_left[1], 1, "the KO'd Gengar dropped pokemon_left to 1 (the replacement is not counted a faint)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "10897,43434,54578,10901",
        "the pursuitfaint (KO + hint + the still-chosen switch) is bit-for-bit → the real Showdown post-turn seed"
    );
}

/// MC35: FOCUS PUNCH MIRROR at a SPEED TIE — both Focus Punch: the two `beforeTurnMove` order-5
/// actions tie (the mirror action-sort shuffle), BOTH mons carry the `focuspunch` volatile at
/// the residual (the +1 residual duration-handler tie-shuffle), the first FP lands and the
/// second is CANCELLED draw-free. WRONG (if the beforeTurnMove tie or the focuspunch residual
/// handler is missing): the draw COUNT desyncs. STATE (the loser chipped by the landed FP; both
/// still alive) + SEED (the crux: the beforeTurnMove ties + the residual mirror tie).
#[test]
fn focus_punch_mirror_speed_tie_draws_the_beforeturnmove_and_residual_ties() {
    let d = dex();
    let machamp = "Machamp||||focuspunch|Serious|,252,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(machamp, machamp, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // p2 moved first (the tie-shuffle) and its FP chipped p1 to 15; p1's FP was cancelled.
    assert_eq!(out.decisions[0].active[0].hp, 15, "the loser (p1) took the landed FP; both survive");
    assert_eq!(out.decisions[0].active[1].hp, 321, "the winner (p2) is unharmed (p1's FP was cancelled)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "56830,34298,10811,30881",
        "the two beforeTurnMove order-5 ties + the +1 residual focuspunch duration-handler tie + the landed FP's acc/crit/dmg → the real Showdown post-turn seed"
    );
}

/// MC36: PURSUIT does NOT intercept a BATON-PASS selfSwitch (`gen3_move_coverage_batch4_v1`, the
/// bench-order-desync fix). The sim SUPPRESSES `BeforeSwitchOut` for a Baton Pass
/// (`batonpass.self.onHit` sets `skipBeforeSwitchOutEventFlag = true`, moves.ts:1109), so the
/// pursued passer is NOT struck: it survives (boosts intact), the Baton Pass completes (the +2 Atk
/// passes to the entrant), and the pursuer's Pursuit runs NORMALLY (bp 40) against the ENTRANT on
/// the resumed turn. WRONG (pre-fix, the interrupt fired for ANY `!is_drag` switch incl. a
/// selfSwitch InstaSwitch): the port struck the still-active passer at ×2 never-miss + `queue.retain`d
/// away the pursuer's Pursuit → a spurious strike (and, for a low-HP passer, a spurious faint in a
/// bench slot → the downstream slot-desync of e2e_11). STATE (Jolteon ALIVE on the bench 271/271;
/// Vaporeon active with +2 Atk having taken the normal Pursuit → 400/464) + SEED (37635,… — the sim's
/// post-switch seed; the pre-fix strike gives a DIFFERENT seed). Ground truth
/// `harness/probe_batch4_pursuit_bench_regression_rng.js`.
#[test]
fn pursuit_does_not_intercept_a_baton_pass_selfswitch() {
    let d = dex();
    let ttar = "Tyranitar|||pressure|pursuit,crunch|Serious|,,,252,,|||||";
    let p2 = "Jolteon|||voltabsorb|batonpass,thunderbolt|Serious|,,,,,252|||||]\
              Vaporeon|||waterabsorb|surf|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(ttar, p2, "44317,42357,9927,48760"), &d).expect("start");
    // Inject +2 Atk on Jolteon (STATE-only, no PRNG — mirrors the probe injection) so the pass
    // is observable on the entrant. Atk does not affect speed, so the draw stream is unchanged.
    {
        let st = battle.state_mut().expect("state");
        let active = st.sides[1].active;
        st.sides[1].pokemon[active].boosts[0] = 2;
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // p1 Pursuit, p2 Baton Pass → forced switch
            ScriptDecision::one(1, Choice::Switch(1)),              // p2 switch to Vaporeon (the pass entrant)
        ],
        &d,
    );
    // dec 1 (the Baton-Pass forced switch): Vaporeon is the entrant, inherits +2 Atk, and takes the
    // NORMAL Pursuit (bp 40) on the resumed turn — the passer was NOT struck.
    assert_eq!(out.decisions[1].active_species[1], "vaporeon", "Vaporeon is the Baton-Pass entrant");
    assert_eq!(out.decisions[1].active[1].boosts[0], 2, "the +2 Atk passed to Vaporeon (copyVolatileFrom.boosts)");
    assert_eq!(out.decisions[1].active[1].hp, 400, "Vaporeon (the ENTRANT) took a NORMAL bp-40 Pursuit → 464 → 400");
    assert_eq!(out.decisions[1].pokemon_left[1], 2, "no faint — the passer was NOT struck");
    // The passer Jolteon is ALIVE on the bench (271/271 untouched), NOT struck.
    let bench_active = st.sides[1].active;
    let bench_idx = (0..st.sides[1].pokemon.len()).find(|&i| i != bench_active).expect("a bench slot");
    assert_eq!(st.sides[1].pokemon[bench_idx].set.species, "Jolteon", "the passer sits on the bench");
    assert_eq!(st.sides[1].pokemon[bench_idx].hp, 271, "the passer is UNTOUCHED (271/271) — no Pursuit strike");
    assert!(!st.sides[1].pokemon[bench_idx].fainted, "the passer is ALIVE (no spurious strike-faint)");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "37635,3740,64462,10380",
        "Baton Pass suppresses the interrupt → Pursuit runs normally against the entrant → the real Showdown seed (a pre-fix strike diverges it)"
    );
}

/// MC36b: the LOW-HP-passer variant of MC36 — pins the exact "fainted-in-a-bench-slot" symptom.
/// Jolteon is injected to 10 HP (≤ a ×2 Pursuit strike). WRONG (pre-fix): the port strikes the
/// passer, KOing it → a FAINTED mon sits in the bench slot where the sim has an ALIVE one → a later
/// "switch to a fainted slot" reject / decision-count desync (the e2e_11 root symptom). With the fix
/// the passer is left ALIVE at 10 HP on the bench. STATE (Jolteon alive 10/271 on the bench,
/// pokemon_left 2) + SEED.
#[test]
fn pursuit_does_not_faint_a_low_hp_baton_pass_passer() {
    let d = dex();
    let ttar = "Tyranitar|||pressure|pursuit,crunch|Serious|,,,252,,|||||";
    let p2 = "Jolteon|||voltabsorb|batonpass,thunderbolt|Serious|,,,,,252|||||]\
              Vaporeon|||waterabsorb|surf|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(ttar, p2, "44317,42357,9927,48760"), &d).expect("start");
    {
        let st = battle.state_mut().expect("state");
        let active = st.sides[1].active;
        st.sides[1].pokemon[active].hp = 10; // ≤ a ×2 Pursuit strike would KO
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::one(1, Choice::Switch(1)),
        ],
        &d,
    );
    assert_eq!(out.decisions[1].active_species[1], "vaporeon", "Vaporeon entered");
    assert_eq!(out.decisions[1].pokemon_left[1], 2, "the low-HP passer did NOT faint (no strike)");
    let bench_active = st.sides[1].active;
    let bench_idx = (0..st.sides[1].pokemon.len()).find(|&i| i != bench_active).expect("a bench slot");
    assert_eq!(st.sides[1].pokemon[bench_idx].set.species, "Jolteon", "the low-HP passer sits on the bench");
    assert_eq!(st.sides[1].pokemon[bench_idx].hp, 10, "the passer stays at 10 HP — NOT struck");
    assert!(!st.sides[1].pokemon[bench_idx].fainted, "the passer is ALIVE (no spurious strike-faint-in-a-slot)");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "37635,3740,64462,10380",
        "the low-HP passer variant is bit-for-bit identical to MC36 (the strike is suppressed either way)"
    );
}

/// R21 (`gen3_batonpass_stall_pursuit_copy_v1`): a Pursuit-into-a-Baton-Passing foe that had
/// Protected — the entrant inherits BOTH `stall` (the Protect counter) AND `pursuit` (the
/// beforeTurnMove-laid volatile) via `copyVolatileFrom`, so its TWO NO_ORDER/subOrder-2 residual
/// duration handlers TIE at the entrant's speed → the turn-2 residual draws ONE Fisher-Yates
/// `random(0,2)` the port previously MISSED (the bab_9_4 pool desync: a BYTE-INVISIBLE hidden draw
/// at the Pursuit-into-Baton-Pass forced-switch turn that shifted the seed and eventually flipped a
/// downstream KO). Turn 1: Zapdos Protects (→ `stall`), Tyranitar Roars (blocked). Turn 2: Tyranitar
/// Pursuit (lays `pursuit` on Zapdos), Zapdos Baton Passes → Metagross enters inheriting stall +
/// pursuit; the resumed-turn Pursuit strikes Metagross (350 → 288) and the end-of-turn residual draws
/// the stall↔pursuit tie-shuffle. WRONG (pre-fix, the pass-set omitted stall + pursuit): the entrant
/// had NEITHER handler → no tie → the shuffle was NOT drawn → the dec-2 seed is one draw short. STATE
/// (Metagross the entrant, 288 = post-Pursuit) + SEED (the residual tie-shuffle). Ground truth
/// `harness/probe_r21_regression_rng.js` (seed [11,22,33,44], gen3customgame).
#[test]
fn baton_pass_copies_stall_and_pursuit_so_the_entrant_residual_ties() {
    let d = dex();
    let p1 = "Tyranitar||Leftovers|SandStream|Crunch,Pursuit,Flamethrower,Roar|Modest|240,,,176,76,16|M|,0,,,,|||";
    let p2 = "Zapdos||Leftovers|Pressure|Thunderbolt,HiddenPowerGrass,Protect,BatonPass|Modest|,,,192,116,200|N|,2,,30,,|||,Grass,,,,]\
              Metagross||ChoiceBand|ClearBody|MeteorMash,Earthquake,HiddenPowerRock,Explosion|Adamant|196,252,,,,60|N|,,30,,30,30|||,Rock,,,,";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "11,22,33,44"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(3), Choice::Move(2)), // T1: Roar (blocked) + Protect → stall
            ScriptDecision::both(Choice::Move(1), Choice::Move(3)), // T2: Pursuit (lays pursuit) + Baton Pass
            ScriptDecision::one(1, Choice::Switch(1)),              // T2 forced switch: Metagross in (stall + pursuit)
        ],
        &d,
    );
    // dec 2 (the Baton-Pass forced switch resuming turn 2): Metagross is the entrant, took the
    // NORMAL Pursuit (350 → 288), and the end-of-turn residual drew the stall↔pursuit tie-shuffle.
    assert_eq!(out.decisions[2].active_species[1], "metagross", "Metagross is the Baton-Pass entrant");
    assert_eq!(out.decisions[2].active[1].hp, 288, "Metagross (the ENTRANT) took the normal Pursuit → 350 → 288");
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "60833,51486,28767,2196",
        "the entrant's inherited stall+pursuit tie their residual duration handlers → the shuffle draws (a pre-fix miss diverges this seed)"
    );
}

/// MC37 (batch-4 nit): PURSUIT INTERRUPT composed with ENTRY HAZARDS — a VOLUNTARY switch into a
/// Pursuit STRIKE, then the replacement enters through the runSwitch EntryHazard (Spikes chip). The
/// strike (×2, on the switcher Jolteon 271 → 153) precedes the swap; the entrant Snorlax then takes
/// the Spikes chip (524 → 459 = maxhp/8) on its runSwitch. Pins the strike→process_faints→swap→
/// runSwitch(spikes) composition the reviewer flagged. WRONG (if the strike or the spikes-on-
/// replacement ordering is broken): the switcher/entrant HP or the seed desyncs. STATE (switcher
/// Jolteon 153 on the bench; entrant Snorlax 459 = post-Spikes) + SEED. Ground truth
/// `harness/probe_batch4_pursuit_bench_regression_rng.js`.
#[test]
fn pursuit_interrupt_into_entry_hazards() {
    let d = dex();
    let ttar = "Tyranitar|||pressure|pursuit,spikes|Serious|,,,252,,|||||";
    let p2 = "Jolteon|||voltabsorb|thunderbolt|Serious|,,,,,252|||||]\
              Snorlax|||immunity|bodyslam|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(ttar, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // p1 Spikes (1 layer), p2 Thunderbolt
            ScriptDecision::both(Choice::Move(0), Choice::Switch(1)), // p1 Pursuit, p2 VOLUNTARY switch → strike + Spikes on entrant
        ],
        &d,
    );
    // dec 1: Pursuit STRUCK the switching Jolteon (×2 → 153 on the bench); Snorlax entered and took
    // the Spikes chip (524 → 459).
    assert_eq!(out.decisions[1].active_species[1], "snorlax", "the replacement Snorlax is active after the strike");
    assert_eq!(out.decisions[1].active[1].hp, 459, "Snorlax took the Spikes chip on entry (524 → 459 = maxhp/8)");
    assert_eq!(out.decisions[1].pokemon_left[1], 2, "no faint — the switcher was only chipped");
    let sw_active = st.sides[1].active;
    let sw_idx = (0..st.sides[1].pokemon.len()).find(|&i| i != sw_active).expect("a bench slot");
    assert_eq!(st.sides[1].pokemon[sw_idx].hp, 153, "the STRUCK switcher Jolteon (×2 Pursuit) sits on the bench at 153/271");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "5621,5056,41416,14688",
        "the strike (never-miss crit+dmg) → swap → runSwitch Spikes (draw-free) composition → the real Showdown seed"
    );
}

/// MC38 (batch-4 nit): PURSUIT INTERRUPT at a pursuer/switcher SPEED TIE — the strike lands, so its
/// in-tryMoveHit `eachEvent('Update')` fires, and because the pursuer (p1 Tyranitar) and the
/// SWITCHER (p2 Tyranitar, still hp>0 pre-swap) TIE on cached speed, that eachEvent draws ONE
/// tie-shuffle `random(0,2)`. Pins the post-strike each-event draw the reviewer flagged. WRONG (if
/// the interrupt's `each_event_shuffle` on a landed strike is dropped, or the tie is mis-evaluated):
/// the draw COUNT desyncs. STATE (the struck switcher Tyranitar 282 on the bench; Snorlax active) +
/// SEED (the crux — the tie-shuffle draw). Ground truth
/// `harness/probe_batch4_pursuit_bench_regression_rng.js`.
#[test]
fn pursuit_speed_tie_interrupt_draws_the_post_strike_each_event() {
    let d = dex();
    let ttar = "Tyranitar|||pressure|pursuit,crunch|Serious|,,,252,,|||||";
    let p2 = "Tyranitar|||pressure|crunch|Serious|,,,252,,|||||]\
              Snorlax|||immunity|bodyslam|Serious|252,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(ttar, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[ScriptDecision { p1: Some(Choice::Move(0)), p2: Some(Choice::Switch(1)) }],
        &d,
    );
    assert_eq!(out.decisions[0].active_species[1], "snorlax", "the replacement Snorlax is active after the strike");
    let sw_active = st.sides[1].active;
    let sw_idx = (0..st.sides[1].pokemon.len()).find(|&i| i != sw_active).expect("a bench slot");
    assert_eq!(st.sides[1].pokemon[sw_idx].hp, 282, "the struck switcher Tyranitar (×2 Pursuit) sits on the bench at 282/341");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37112,13693,28533,21721",
        "the landed strike's in-tryMoveHit eachEvent('Update') draws the pursuer↔switcher speed-tie shuffle → the real Showdown seed"
    );
}

// ── MOVE-COVERAGE BATCH 4b (`gen3_move_coverage_batch4b_v1`): BEAT UP / THUNDER / WATER SPOUT ──
// Ground truth: `harness/probe_batch4b_regression_rng.js`. Each is a CONSTRUCTED gen3customgame
// board reseeded to the raw seed (aligning the port's draw-free `start_with_switchins`),
// revert-verified (each FAILS when its move's engine wiring is reverted).

/// MC39: BEAT UP full 6-strike — Slaking + 5 healthy bench each strike the bulky Skarmory once
/// (typeless flat-BP-10 Special with the ally-base-atk → SpA / target-base-def → SpD stat swap).
/// ONE accuracy roll + 6*(crit+damage) + Quick Claw. WRONG (if the beatup arm is reverted): the
/// move runs as a plain bp-10 Dark move — ONE crit+damage pair (not six) → both the per-strike HP
/// AND the seed desync. STATE (Skarmory 271 → 237, the six-strike chip in ally base-atk order
/// −10/−8/−4/−7/−1/−4) + SEED (the 12 per-strike draws + Quick Claw).
#[test]
fn beat_up_full_side_strikes_once_per_healthy_teammate() {
    let d = dex();
    let p1 = "Slaking|||keeneye|beatup,seismictoss|Serious|252,252,,,,|||||\
]Machamp|||noguard|splash|Serious|,,,,,|||||\
]Alakazam|||synchronize|splash|Serious|,,,,,|||||\
]Snorlax|||immunity|splash|Serious|,,,,,|||||\
]Blissey|||naturalcure|splash|Serious|,,,,,|||||\
]Gengar|||keeneye|splash|Serious|,,,,,|||||";
    let p2 = "Skarmory|||keeneye|splash|Serious|,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    // Skarmory survived all 6 strikes: 271 → 237 (the six-strike stat-swap chip).
    assert_eq!(out.decisions[0].active[1].hp, 237, "Beat Up = 6 strikes (one per healthy teammate) → 271-34");
    assert!(!out.decisions[0].active[1].fainted, "Skarmory survived the 6-strike Beat Up");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "21092,10154,5215,50054",
        "1 accuracy roll + 6*(crit+damage) + Quick Claw → the real Showdown post-turn seed"
    );
}

/// MC40: BEAT UP KOs the target MID-SEQUENCE — a 30-HP Gengar SURVIVES the first strike (→ 7)
/// then the SECOND strike KOs it → the multihit STOPS (later strikes + the Quick Claw skip; the
/// deferred-faint protocol). WRONG (if the beatup arm is reverted → a plain bp-10 Dark hit, or if
/// the loop does not stop): a plain single hit leaves Gengar alive (30-HP survives a bp-10 hit) →
/// the pokemon_left + seed desync; a non-stopping multihit fires extra strikes / a Quick Claw on
/// the deciding faint → the seed desyncs. STATE (Gengar fainted after exactly 2 strikes, p1 wins)
/// + SEED (1 acc + 2*(crit+damage), NO Quick Claw).
#[test]
fn beat_up_ko_mid_sequence_stops_the_multihit_no_quick_claw() {
    let d = dex();
    let p1 = "Slaking|||keeneye|beatup,seismictoss|Serious|252,,,,,|||||\
]Snorlax|||immunity|splash|Serious|,,,,,|||||\
]Blissey|||naturalcure|splash|Serious|,,,,,|||||";
    let p2 = "Gengar|||keeneye|splash|Serious|,,,,,252|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    // Inject Gengar to 30 HP (STATE-only, no PRNG): strike 1 leaves it at 7, strike 2 KOs it.
    {
        let st = battle.state_mut().expect("state");
        let active = st.sides[1].active;
        st.sides[1].pokemon[active].hp = 30;
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert!(out.decisions[0].active[1].fainted, "the SECOND Beat Up strike KO'd the 30-HP Gengar");
    assert_eq!(out.decisions[0].pokemon_left[1], 0, "Gengar was p2's only mon → p1 wins");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "22534,42410,55299,35327",
        "the multihit STOPS at the faint after 2 strikes (1 acc + 2*(crit+damage), NO Quick Claw on the deciding faint) → the real Showdown seed"
    );
}

/// MC41: THUNDER in RAIN — the id-gated onModifyMove makes Thunder NEVER-MISS in rain, so the
/// accuracy `random(100)` is SKIPPED (ONE FEWER draw). WRONG (if the weather-accuracy mutation is
/// reverted): Thunder uses base acc 70 in rain → it DRAWS the accuracy roll (and, at this seed,
/// MISSES) → the post-turn seed matches the BASE control (60880,...) instead of the rain seed.
/// STATE (the never-miss Thunder HITS + paralyzes Blissey) + SEED (distinct from the base
/// control's) — the never-miss-skips-the-accuracy-draw proof.
#[test]
fn thunder_rain_never_miss_skips_the_accuracy_draw() {
    let d = dex();
    let zap = "Zapdos|||keeneye|thunder,seismictoss|Serious|,,252,,,252|||||";
    let rain = "Blissey|||drizzle|splash|Serious|252,,,,,|||||";
    let base = "Blissey|||naturalcure|splash|Serious|252,,,,,|||||";

    // RAIN: never-miss → 0 accuracy draw → HIT + para.
    let mut br =
        Battle::start_with_switchins(&opts_cg(zap, rain, "44317,42357,9927,48760"), &d).expect("start");
    let outr = br.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(outr.decisions[0].active[1].hp, 591, "rain Thunder is never-miss → it HITS (714 → 591)");
    assert_eq!(outr.decisions[0].active[1].status, Some(Status::Paralysis), "the 30% para landed");
    let rain_seed = seed_str(&outr.decisions[0].seed_after);
    assert_eq!(rain_seed, "22534,42410,55299,35327", "rain: crit+damage+para-secondary+full-para+Quick Claw (NO accuracy draw) → the real Showdown seed");

    // BASE control (no weather): base acc 70 IS drawn → at this seed it MISSES → a DIFFERENT seed.
    let mut bb =
        Battle::start_with_switchins(&opts_cg(zap, base, "44317,42357,9927,48760"), &d).expect("start");
    let outb = bb.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(outb.decisions[0].active[1].hp, 714, "base Thunder DRAWS the accuracy roll and MISSES at this seed");
    let base_seed = seed_str(&outb.decisions[0].seed_after);
    assert_eq!(base_seed, "60880,31090,7619,34922", "base: the accuracy random(100) draws + a miss → the real Showdown seed");

    assert_ne!(rain_seed, base_seed, "reverting the rain never-miss makes rain == base — the extra accuracy draw is the crux");
}

/// MC42: THUNDER in SUN — the id-gated onModifyMove sets base accuracy 50 (a LOWER threshold). At
/// a seed whose accuracy roll ∈ [50,70), SUN (thresh 50) MISSES while BASE (thresh 70) HITS. WRONG
/// (if the sun→50 mutation is reverted): sun uses base 70 → it HITS → Blissey takes damage instead
/// of staying at full HP. STATE (sun MISS = Blissey 714; base HIT = Blissey 589). Same accuracy
/// draw COUNT either way (only the threshold differs), so this is a STATE pin.
#[test]
fn thunder_sun_base_accuracy_fifty_lowers_the_hit_threshold() {
    let d = dex();
    let zap = "Zapdos|||keeneye|thunder,seismictoss|Serious|,,252,,,252|||||";
    let sun = "Blissey|||drought|splash|Serious|252,,,,,|||||";
    let base = "Blissey|||naturalcure|splash|Serious|252,,,,,|||||";
    let seed = "8,25,58,91"; // its first random(100) == 60 ∈ [50,70)

    // SUN (thresh 50): 60 >= 50 → MISS.
    let mut bs = Battle::start_with_switchins(&opts_cg(zap, sun, seed), &d).expect("start");
    let outs = bs.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(outs.decisions[0].active[1].hp, 714, "sun Thunder (thresh 50) MISSES at roll 60 → Blissey unharmed");

    // BASE (thresh 70): 60 < 70 → HIT.
    let mut bb = Battle::start_with_switchins(&opts_cg(zap, base, seed), &d).expect("start");
    let outb = bb.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(outb.decisions[0].active[1].hp, 589, "base Thunder (thresh 70) HITS at roll 60 → Blissey 714 → 589");
}

/// MC43: WATER SPOUT variable BP — `bp = max(floor(150·hp/maxhp), 1)`. At the SAME seed, full HP
/// (bp 150) and low HP (179/341 → bp 78) deal DIFFERENT damage but end at the IDENTICAL post-turn
/// seed (the variable BP is a deterministic STATE read, DRAW-NEUTRAL). WRONG (if the basePower
/// callback is reverted): both HP levels run at the flat placeholder bp → the low-HP damage is
/// wrong. STATE (full: Snorlax 524 → 279; low: Snorlax 524 → 397) + the SAME SEED (draw-neutral).
#[test]
fn water_spout_variable_bp_is_draw_neutral() {
    let d = dex();
    let kyogre = "Kyogre|||keeneye|waterspout,seismictoss|Serious|,,252,,,252|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";

    // FULL HP: bp 150.
    let mut bf =
        Battle::start_with_switchins(&opts_cg(kyogre, snorlax, "44317,42357,9927,48760"), &d).expect("start");
    let outf = bf.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(outf.decisions[0].active[1].hp, 279, "Water Spout at full HP is bp 150 → Snorlax 524 → 279");
    let full_seed = seed_str(&outf.decisions[0].seed_after);
    assert_eq!(full_seed, "37635,3740,64462,10380", "the full-HP Water Spout draws acc+crit+damage + Quick Claw → the real Showdown seed");

    // LOW HP (Kyogre injected to 179/341 → bp floor(150*179/341)=78): smaller damage, SAME seed.
    let mut bl =
        Battle::start_with_switchins(&opts_cg(kyogre, snorlax, "44317,42357,9927,48760"), &d).expect("start");
    {
        let st = bl.state_mut().expect("state");
        let active = st.sides[0].active;
        st.sides[0].pokemon[active].hp = 179;
    }
    let outl = bl.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(outl.decisions[0].active[1].hp, 397, "Water Spout at 179/341 HP is bp 78 → smaller damage → Snorlax 524 → 397");
    assert_eq!(
        seed_str(&outl.decisions[0].seed_after),
        full_seed,
        "the variable BP is a STATE read → DRAW-NEUTRAL: the low-HP turn draws the SAME PRNG sequence as the full-HP turn"
    );
}


/// MC44: BEAT UP MIRROR at a SPEED TIE — both Charizards Beat Up at equal speed. This pins TWO
/// tie-only draws the distinct-speed Beat Up scenarios hide: (a) the gen3 multihit loop's
/// PER-STRIKE `eachEvent('Update')` (scripts.js — drawn after each strike on a tie); (b) the
/// `beatup` `duration: 1` volatile's residual DURATION handler (two `beatup` volatiles tie → one
/// residual shuffle). WRONG (reverting EITHER): the post-turn seed desyncs. STATE (p1 Charizard
/// 285, p2 Charizard 258 — both survive) + SEED.
#[test]
fn beat_up_mirror_speed_tie_draws_the_per_strike_and_residual_shuffles() {
    let d = dex();
    let p1 = "Charizard|||Blaze|beatup,seismictoss|Modest|,,,252,,252|N||||\
]Slaking|||keeneye|splash|Serious|,,,,,|N||||\
]Machamp|||noguard|splash|Serious|,,,,,|N||||";
    let p2 = "Charizard|||Blaze|beatup,roost|Modest|,,,252,,252|N||||\
]Magikarp|||keeneye|splash|Serious|,,,,,|N|||5|";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let out = battle.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].hp, 285, "p1 Charizard survived p2's 2-strike Beat Up");
    assert_eq!(out.decisions[0].active[1].hp, 258, "p2 Charizard survived p1's 3-strike Beat Up");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37710,62820,53795,28781",
        "the per-strike eachEvent('Update') (each Beat Up) + the two beatup-volatile residual duration handlers (the mirror tie) → the real Showdown seed"
    );
}

/// MC45: BEAT UP CANCELS A FOCUS PUNCH — p2 Charizard Beat Ups (priority 0) into p1 Charizard,
/// which chose Focus Punch (priority -3 → moves LAST). The Beat Up direct strikes set lostFocus
/// → the Focus Punch is CANCELLED draw-free (0 damage to p2). WRONG (if `run_beat_up` does not
/// set lostFocus): the Focus Punch would land + damage p2 (the e2e_196 real-team desync). STATE
/// (p2 Charizard UNHARMED at 297 — the FP was cancelled; p1 Charizard took the Beat Up → 256) +
/// SEED.
#[test]
fn beat_up_hit_cancels_the_targets_focus_punch() {
    let d = dex();
    let p1 = "Charizard|||Blaze|focuspunch,seismictoss|Modest|,4,,,,252|N||||";
    let p2 = "Charizard|||Blaze|beatup,roost|Modest|,,,252,,252|N||||\
]Slaking|||keeneye|splash|Serious|,,,,,|N||||\
]Machamp|||noguard|splash|Serious|,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let out = battle.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[1].hp, 297, "the Beat Up set lostFocus → the Focus Punch was CANCELLED → p2 Charizard UNHARMED");
    assert_eq!(out.decisions[0].active[0].hp, 256, "p1 Charizard took the 3-strike Beat Up");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "39289,57899,39292,47653",
        "the Beat Up (3 strikes + per-strike Updates) + the cancelled Focus Punch (draw-free) + Quick Claw → the real Showdown seed"
    );
}

/// MC46: BEAT UP's `beatup` `duration: 1` volatile is dropped by clearVolatile on switch-out —
/// exactly like its sibling focus_punch/pursuit volatiles (`gen3_move_coverage_batch4b_v1`).
/// WRONG (pre-fix): `execute_switch`'s clearVolatile reset focus_punch/pursuit but OMITTED
/// beat_up, so a Beat Up user phazed out the SAME turn it Beat Up'd (Roar priority -6 resolves
/// AFTER the move) kept a STALE `beat_up = true` on the bench (the turn-top `clear_flinch` is
/// ACTIVE-mon-only), and on RE-ENTRY the active-only residual gather pushed a SPURIOUS
/// NO_ORDER/subOrder-2 VolatileDuration handler that TIED the foe's → one extra `random(0,2)`
/// tie-shuffle vs the sim (a silent draw-order desync).
///
/// Choreography (all Charizards → equal speed → residual ties): turn 1 p1 Charizard Beat Ups
/// (sets the beatup volatile) + p2 Charizard Roars (priority -6 → drags the n=1-eligible bench
/// Charizard in, benching the Beat Up user — the sim's clearVolatile drops its beatup volatile);
/// turn 2 p1 SWITCHES the Beat Up user back in + p2 Beat Ups (registers ITS OWN beatup residual
/// handler). At the turn-2 residual the returned mon is active BEFORE the next turn-top
/// clear_flinch, so a stale beat_up would tie p2's handler → an extra shuffle. STATE (the
/// dragged-out mon's beat_up == false on the bench + the returned/foe HP) + SEED (both decisions
/// vs the real Showdown ground truth). Ground truth
/// `harness/probe_batch4b_beatup_switchout_regression_rng.js`.
#[test]
fn beat_up_volatile_clears_on_switch_out() {
    let d = dex();
    let p1 = "Charizard|||Blaze|beatup,splash|Modest|,,,252,,252|N||||\
]Charizard|||Blaze|beatup,splash|Modest|,,,252,,252|N||||";
    let p2 = "Charizard|||Blaze|roar,beatup|Modest|,,,252,,252|N||||\
]Magikarp|||keeneye|splash|Serious|,,,,,|N|||5|";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let out = battle.state_mut().unwrap().run_full_battle(
        &[
            // turn 1: p1 Beat Up, p2 Roar (drags the p1 Beat Up user OUT).
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            // turn 2: p1 switches the Beat Up user (array slot 1) back in, p2 Beat Ups.
            ScriptDecision::both(Choice::Switch(1), Choice::Move(1)),
        ],
        &d,
    );

    // dec 0: the phaze dragged the n=1-eligible bench Charizard in; the Beat Up user is BENCHED
    // (its beatup volatile is cleared by clearVolatile — the fix; pre-fix it stayed true and
    // re-entered stale on turn 2). p2 Charizard took the 2-strike Beat Up (297 → 277).
    assert_eq!(out.decisions[0].active[1].hp, 277, "p2 Charizard took the turn-1 2-strike Beat Up");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "63651,14230,62171,48683",
        "turn 1: p1 Beat Up (2 strikes) + p2 Roar (acc + n=1 sample) + Quick Claw → the real Showdown seed"
    );

    // dec 1: the Beat Up user is back active; p2 Beat Ups it (2 strikes, 297 → 286). At the
    // residual only p2 has a beatup handler (the returned mon's is cleared), so NO tie shuffle —
    // a stale beat_up would tie it, drawing one extra shuffle and desyncing THIS seed.
    assert_eq!(out.decisions[1].active[0].hp, 286, "the returned Beat Up user took p2's 2-strike Beat Up");
    assert_eq!(out.decisions[1].active[1].hp, 277, "p2 Charizard unchanged (p1 switched, did not attack)");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "53016,60759,29878,11288",
        "turn 2: p1 switch + p2 Beat Up (2 strikes + per-strike Updates) with ONLY p2's beatup residual handler (NO stale-flag tie) → the real Showdown seed"
    );
}

/// MC47: WATER SPOUT's min-BP-1 clamp — `bp = max(floor(150·hp/maxhp), 1)`. At 1 HP,
/// `floor(150·1/341)=0` → the `.max(1)` clamps the BP to 1 (a MIN-damage HIT, NOT a fail).
/// The batch-4b golden (`waterspout_low_hp` at hp 40 → bp 17) never realizes the clamp; this
/// pins the hp=1 boundary. STATE (Snorlax 524 → 521, a 3-HP min hit) + SEED (identical to the
/// full-HP Water Spout MC43 — the variable BP is a deterministic STATE read → DRAW-NEUTRAL even
/// at the clamp). Ground truth `harness/probe_batch4b_edge_regression_rng.js`.
#[test]
fn water_spout_at_one_hp_clamps_the_base_power_to_one() {
    let d = dex();
    let kyogre = "Kyogre|||keeneye|waterspout,seismictoss|Serious|,,252,,,252|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|252,,,,,|||||";
    let mut b =
        Battle::start_with_switchins(&opts_cg(kyogre, snorlax, "44317,42357,9927,48760"), &d).expect("start");
    {
        let st = b.state_mut().expect("state");
        let active = st.sides[0].active;
        st.sides[0].pokemon[active].hp = 1; // floor(150*1/341)=0 → clamp to bp 1
    }
    let out = b.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[1].hp, 521, "Water Spout at 1 HP is bp 1 → a MIN-damage HIT (Snorlax 524 → 521), NOT a fizzle");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37635,3740,64462,10380",
        "the clamped-BP Water Spout draws the SAME acc+crit+damage+Quick-Claw sequence as the full-HP one (draw-neutral) → the real Showdown seed"
    );
}

/// MC48: BEAT UP with NO eligible party member fizzles — a STATUSED active user whose ONLY
/// party member is itself (statused → skipped) yields 0 strikes → the `basePowerCallback`
/// returns null → the move FIZZLES (`|move|Beat Up||[still]` + `|-fail|`), drawing only the
/// whole-move accuracy roll (drawn in `run_move` BEFORE the empty-strikes return). The batch-4b
/// golden never realizes the empty-strikes branch; this pins it. STATE (Gengar UNTOUCHED at 261;
/// the burned Slaking takes only its burn chip 504 → 441) + SEED (acc + Quick Claw only — no
/// strikes → no crit/damage draws). Ground truth `harness/probe_batch4b_edge_regression_rng.js`.
#[test]
fn beat_up_with_no_eligible_party_fizzles_drawing_only_accuracy() {
    let d = dex();
    let slaking = "Slaking|||keeneye|beatup,seismictoss|Serious|252,252,,,,|||||";
    let gengar = "Gengar|||levitate|splash|Serious|,,,,,252|||||";
    let mut b =
        Battle::start_with_switchins(&opts_cg(slaking, gengar, "44317,42357,9927,48760"), &d).expect("start");
    {
        let st = b.state_mut().expect("state");
        let active = st.sides[0].active;
        st.sides[0].pokemon[active].status = Some(Status::Burn); // the only party member is statused → 0 strikes
    }
    let out = b.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[1].hp, 261, "the Beat Up FIZZLED (0 eligible strikers) → Gengar UNTOUCHED");
    assert_eq!(out.decisions[0].active[0].hp, 441, "the burned Slaking took only its burn chip (504 → 441)");
    assert_eq!(out.decisions[0].active[0].status, Some(Status::Burn), "Slaking stays burned");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "60880,31090,7619,34922",
        "the fizzle draws the whole-move accuracy roll (no strikes → no crit/damage) + Quick Claw → the real Showdown seed"
    );
}

// ============================================================================
// MOVE-COVERAGE BATCH 4c (`gen3_move_coverage_batch4c_v1`) — HYPER BEAM
// (mustrecharge) / SOLAR BEAM (two-turn charge) / DOOM DESIRE + FUTURE SIGHT
// (the slot-keyed future strike). Ground-truth seeds/state from
// `harness/probe_batch4c_regression_rng.js` (the REAL Showdown, verbatim);
// the draw models were settled by `harness/probe_batch4c_{hyperbeam,solarbeam,
// doomdesire}.js`. Teams mirror the probe exactly (packed via Teams.pack).
// ============================================================================

const HB_P1: &str = "Snorlax|||NoAbility|hyperbeam,splash|Serious||N||||]Blissey|||NoAbility|softboiled|Serious||N||||";
const HB_P2: &str = "Skarmory|||NoAbility|spikes,protect,splash,roar|Serious|252,,,,,|N||||]Forretress|||NoAbility|spikes,splash|Serious|252,,,,,|N||||";

/// MC49: HYPER BEAM hit → RECHARGE → clear. The cast turn is an ordinary damaging move
/// (acc+crit+dmg + the foe's draw-free Spikes + Quick Claw) that applies `mustrecharge`
/// DRAW-FREE; the LOCKED turn's user action draws ZERO and costs NO PP (`|cant|…|recharge`
/// — the only draw is the endTurn Quick Claw); the lock then fully CLEARS (Hyper Beam
/// fires again the next turn). WRONG (pre-batch-4c): the port ran HB recharge-less — the
/// locked turn ran a SECOND Hyper Beam (acc+crit+dmg draws + damage + PP) → both the seed
/// AND the state desync. STATE (Skarmory 259 after ONE hit, unchanged through the locked
/// turn, 183 after the next HB; PP 7 → 7 → 6) + SEED (all three boundaries).
#[test]
fn hyper_beam_hit_locks_then_recharges_draw_free_then_clears() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(HB_P1, HB_P2, "53118,34657,41207,29520"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // HB hits (Skarm lays Spikes)
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // LOCKED: move 1 == Recharge
            ScriptDecision::both(Choice::Move(0), Choice::Move(2)), // free again: HB vs Splash
        ],
        &d,
    );
    // dec0 — the cast: one HB hit (334-75=259), PP 8→7, the lock applied.
    assert_eq!(out.decisions[0].active[1].hp, 259, "one Hyper Beam hit: Skarmory 334 → 259");
    assert_eq!(out.decisions[0].active[0].move_pp[0], 7, "the cast consumed 1 PP");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "48590,39028,4743,65508",
        "cast turn: HB acc+crit+dmg + draw-free Spikes + Quick Claw → the real Showdown seed"
    );
    // dec1 — the LOCKED turn: the user's action draws ZERO + costs NO PP; Skarmory
    // untouched (the recharge is spent doing nothing).
    assert_eq!(out.decisions[1].active[1].hp, 259, "the locked turn deals nothing");
    assert_eq!(out.decisions[1].active[0].move_pp[0], 7, "the recharge costs NO PP");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "34574,58966,18171,38839",
        "locked turn: |cant|recharge draws ZERO — only the endTurn Quick Claw → the real Showdown seed"
    );
    // dec2 — the lock CLEARED: Hyper Beam fires again (259-76=183), PP 7→6.
    assert_eq!(out.decisions[2].active[1].hp, 183, "the lock cleared — HB fires again (259 → 183)");
    assert_eq!(out.decisions[2].active[0].move_pp[0], 6, "the second cast consumed its PP");
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "65303,16015,55571,987",
        "free turn: a normal HB again → the real Showdown seed"
    );
}

/// MC50: a MISSED Hyper Beam does NOT lock (PP still consumed; the user acts FREELY the
/// next turn — at this seed the follow-up HB HITS and re-locks). WRONG (a model locking
/// on any use): the next turn would be a zero-draw recharge instead of a full HB → seed +
/// state desync. STATE (Skarmory untouched by the miss, 260 after the free-turn hit; PP
/// 7 → 6) + SEED (both boundaries).
#[test]
fn hyper_beam_miss_does_not_lock() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(HB_P1, HB_P2, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(2)), // HB MISSES (acc 90 fails)
            ScriptDecision::both(Choice::Move(0), Choice::Move(2)), // NOT locked: a fresh HB hits
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[1].hp, 334, "the missed HB deals nothing");
    assert_eq!(out.decisions[0].active[0].move_pp[0], 7, "a miss still consumes PP");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "60880,31090,7619,34922",
        "miss turn: the accuracy draw fails + Quick Claw → the real Showdown seed"
    );
    assert_eq!(out.decisions[1].active[1].hp, 260, "NOT locked — the next HB fires and hits");
    assert_eq!(out.decisions[1].active[0].move_pp[0], 6, "the free-turn HB consumed its PP");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "7184,5868,30814,34654",
        "free turn: a full HB (acc+crit+dmg) — NOT a zero-draw recharge → the real Showdown seed"
    );
}

/// MC51: a Hyper Beam that KOs the target STILL locks — the `|-mustrecharge|` precedes
/// the `|faint|`, the KO turn draws NO endTurn residual (the deferred-faint protocol —
/// the Quick Claw lands at the force-switch boundary), and the lock PERSISTS across the
/// opponent's force-switch (the next full turn is a zero-draw recharge vs the entrant).
/// WRONG (a lock consumed by the force-switch / no lock on a KO): the post-replacement
/// turn runs a full HB → seed + state desync. STATE + SEED (all three boundaries).
#[test]
fn hyper_beam_ko_still_locks_across_the_force_switch() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(HB_P1, HB_P2, "53118,34657,41207,29520"), &d)
            .expect("start");
    {
        let st = battle.state_mut().expect("state");
        let a = st.sides[1].active;
        st.sides[1].pokemon[a].hp = 20; // STATE-only inject: the HB KOs
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // HB KOs Skarmory
            ScriptDecision { p1: None, p2: Some(Choice::Switch(1)) }, // the forced replacement
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // p1 STILL locked
        ],
        &d,
    );
    assert!(out.decisions[0].active[1].fainted, "the HB KO'd the 20-HP Skarmory");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "13521,38977,6462,56077",
        "KO turn: acc+crit+dmg ONLY (the endTurn residual/Quick Claw is deferred) → the real Showdown seed"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "48590,39028,4743,65508",
        "force-switch boundary: the deferred endTurn Quick Claw draws → the real Showdown seed"
    );
    assert_eq!(out.decisions[2].active[1].hp, 354, "the locked turn deals nothing to the entrant");
    assert_eq!(out.decisions[2].active[0].move_pp[0], 7, "the recharge costs NO PP");
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "34574,58966,18171,38839",
        "the post-replacement turn is STILL the zero-draw recharge → the real Showdown seed"
    );
}

/// MC52: a PARALYZED user on the LOCKED turn — the recharge cant (onBeforeMove priority
/// 11) WINS over paralysis (1): NO para roll is drawn on the locked turn (the seed
/// advance is IDENTICAL to the un-paralyzed MC49 dec1: 48590… → 34574…), the par STAYS,
/// and the para roll resumes on the FOLLOWING turn (the probed |cant|par at this seed).
/// WRONG (para before recharge / a para roll on the locked turn): one extra random(4) →
/// the locked-turn seed desyncs. SEED (both boundaries) + STATE (par persists).
#[test]
fn paralyzed_user_on_the_recharge_turn_draws_no_para_roll() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(HB_P1, HB_P2, "53118,34657,41207,29520"), &d)
            .expect("start");
    // Phase 1 — the cast turn (identical to MC49 dec0).
    let out0 = battle.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(seed_str(&out0.decisions[0].seed_after), "48590,39028,4743,65508");
    // Inject par on the locked Snorlax (STATE-only, mirroring the probe's pre-act).
    {
        let st = battle.state_mut().expect("state");
        let a = st.sides[0].active;
        st.sides[0].pokemon[a].status = Some(Status::Paralysis);
    }
    // Phase 2 — the LOCKED turn (par'd) + the following turn (the para roll resumes).
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // locked + par'd
            ScriptDecision::both(Choice::Move(1), Choice::Move(2)), // Splash — the para roll NOW draws
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[0].status, Some(Status::Paralysis), "the par persists through the recharge");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "34574,58966,18171,38839",
        "the locked turn draws NO para roll — the seed advance is IDENTICAL to the un-par'd control"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "10368,44139,43612,44497",
        "the FOLLOWING turn resumes the para roll (the probed |cant|par) → the real Showdown seed"
    );
}

const SB_P1: &str = "Venusaur|||NoAbility|solarbeam,razorleaf|Serious||N||||]Snorlax|||NoAbility|bodyslam|Serious||N||||";
const SB_P2: &str = "Swampert|||NoAbility|surf|Serious|252,,,,,|N||||]Blissey|||NoAbility|softboiled|Serious|252,,,,,|N||||";

/// MC53: SOLAR BEAM charge → fire → a FRESH charge. The CHARGE turn draws ZERO move
/// draws (only the foe's Surf + Quick Claw; PP is paid HERE — 16 → 15); the FIRE turn
/// draws acc+crit+dmg with NO PP (it KOs the 4×-weak Swampert — no Quick Claw, the
/// deferred faint); the lingering twoturnmove is residual-cleaned across the
/// replacement boundary (1 draw — the resumed Quick Claw); the NEXT Solar Beam is a
/// FRESH charge that re-pays PP (15 → 14, the failed full-HP SoftBoiled draw-free).
/// WRONG (pre-batch-4c: SB collapsed to one turn): the charge turn draws acc+crit+dmg →
/// every boundary desyncs. STATE (HP/PP) + SEED (all four boundaries).
#[test]
fn solar_beam_charges_then_fires_then_recharges_fresh() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(SB_P1, SB_P2, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // CHARGE (Swampert Surfs)
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // FIRE (KOs Swampert)
            ScriptDecision { p1: None, p2: Some(Choice::Switch(1)) }, // the replacement
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // a FRESH charge vs Blissey
        ],
        &d,
    );
    // dec0 — the charge: zero move draws; Venusaur took the Surf; PP 16 → 15.
    assert_eq!(out.decisions[0].active[0].hp, 250, "Venusaur was hit DURING the charge (the charge survives)");
    assert_eq!(out.decisions[0].active[0].move_pp[0], 15, "the CHARGE turn pays the PP");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37635,3740,64462,10380",
        "charge turn: Surf acc+crit+dmg + Quick Claw ONLY (the charge draws nothing) → the real Showdown seed"
    );
    // dec1 — the fire: acc+crit+dmg, NO PP, KOs Swampert (no Quick Claw).
    assert!(out.decisions[1].active[1].fainted, "the fire KO'd the 4×-weak Swampert");
    assert_eq!(out.decisions[1].active[0].move_pp[0], 15, "the FIRE turn pays NO PP");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "37112,13693,28533,21721",
        "fire turn: SB acc+crit+dmg, no Quick Claw on the deciding-side faint → the real Showdown seed"
    );
    // dec2 — the replacement boundary: the resumed tail's Quick Claw (the lingering
    // twoturnmove is cleaned draw-free by the resumed residual).
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "5621,5056,41416,14688",
        "replacement boundary: exactly the resumed Quick Claw → the real Showdown seed"
    );
    // dec3 — a FRESH charge (the volatile did NOT linger into the request): PP 15 → 14.
    assert_eq!(out.decisions[3].active[0].move_pp[0], 14, "the fresh charge re-pays PP");
    assert_eq!(
        seed_str(&out.decisions[3].seed_after),
        "50157,35106,3193,41379",
        "fresh charge vs Blissey (its full-HP SoftBoiled fails draw-free) → the real Showdown seed"
    );
}

/// MC54: SOLAR BEAM in SUN (Drought) SKIPS the charge — the whole turn draws like a
/// normal move (Groudon's EQ 3 + SB 3 + Quick Claw = 7), PP −1, NO volatile (the next
/// request is NOT locked). WRONG (a sun charge / a wrong skip draw count): the seed
/// desyncs. STATE (both sides hit hard on ONE turn) + SEED.
#[test]
fn solar_beam_sun_skip_fires_immediately() {
    let d = dex();
    let p1 = "Venusaur|||NoAbility|solarbeam|Serious||N||||";
    let p2 = "Groudon|||Drought|earthquake|Serious|252,,,,,|N||||]Blissey|||NoAbility|softboiled|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(out.decisions[0].active[0].hp, 95, "Groudon's EQ landed the same turn");
    assert_eq!(out.decisions[0].active[1].hp, 72, "the sun-skipped Solar Beam fired IMMEDIATELY (404 → 72)");
    assert_eq!(out.decisions[0].active[0].move_pp[0], 15, "the skip pays the normal 1 PP");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37112,13693,28533,21721",
        "sun-skip turn: EQ 3 + SB 3 + Quick Claw = 7 draws → the real Showdown seed"
    );
}

/// MC55: SOLAR BEAM's BP is HALVED in RAIN (the gen3-resolved onBasePower chainModify(0.5)
/// — gen3 DOES have the modern halving, probe-confirmed) — a STATE-only fold: the rain
/// fire deals 54 (404 → 350) while the no-weather control deals 105 (404 → 299) at
/// BYTE-IDENTICAL boundary seeds (the halving is draw-free). WRONG (no halving): the rain
/// Kyogre's HP reads the control's. STATE (both damages) + the SHARED seeds.
#[test]
fn solar_beam_rain_halves_the_bp_state_only() {
    let d = dex();
    let p1 = "Venusaur|||NoAbility|solarbeam|Serious||N||||";
    let rain = "Kyogre|||Drizzle|calmmind|Serious|252,,,,,|N||||]Blissey|||NoAbility|softboiled|Serious|252,,,,,|N||||";
    let ctl = "Kyogre|||ShellArmor|calmmind|Serious|252,,,,,|N||||]Blissey|||NoAbility|softboiled|Serious|252,,,,,|N||||";
    let script = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // charge
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // fire
    ];

    let mut br = Battle::start_with_switchins(&opts_cg(p1, rain, "44317,42357,9927,48760"), &d).expect("start");
    let outr = br.state_mut().unwrap().run_full_battle(&script, &d);
    assert_eq!(outr.decisions[1].active[1].hp, 350, "RAIN: the fire's BP is halved (404 → 350, dmg 54)");
    let rain_seeds = (
        seed_str(&outr.decisions[0].seed_after),
        seed_str(&outr.decisions[1].seed_after),
    );
    assert_eq!(rain_seeds.0, "61255,39458,1834,64539");
    assert_eq!(rain_seeds.1, "22534,42410,55299,35327");

    let mut bc = Battle::start_with_switchins(&opts_cg(p1, ctl, "44317,42357,9927,48760"), &d).expect("start");
    let outc = bc.state_mut().unwrap().run_full_battle(&script, &d);
    assert_eq!(outc.decisions[1].active[1].hp, 299, "CONTROL: full BP (404 → 299, dmg 105)");
    assert_eq!(seed_str(&outc.decisions[0].seed_after), rain_seeds.0, "the halving is DRAW-FREE (identical charge seed)");
    assert_eq!(seed_str(&outc.decisions[1].seed_after), rain_seeds.1, "the halving is DRAW-FREE (identical fire seed)");
}

/// MC56: a FULL-PARA on the CHARGE turn — the para roll IS drawn on the charge turn
/// (onBeforeMove precedes onTryMove) and a full-para cant means NO charge and NO PP
/// (16/16 — the PP deduction sits after BeforeMove). WRONG (charge before the para roll /
/// PP paid anyway): the seed / the PP desyncs. STATE (PP untouched) + SEED.
#[test]
fn solar_beam_full_para_on_the_charge_turn_no_charge_no_pp() {
    let d = dex();
    let p1 = "Venusaur|||NoAbility|solarbeam|Serious||N||||";
    let p2 = "Swampert|||NoAbility|curse|Serious|252,,,,,|N||||]Blissey|||NoAbility|softboiled|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "61872,34750,8741,59883"), &d).expect("start");
    {
        let st = battle.state_mut().expect("state");
        let a = st.sides[0].active;
        st.sides[0].pokemon[a].status = Some(Status::Paralysis);
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(out.decisions[0].active[0].move_pp[0], 16, "a full-para'd charge pays NO PP");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "49988,3828,17110,61724",
        "the charge turn draws the para roll (full-para → cant, no charge) + Curse selfDrops + Quick Claw → the real Showdown seed"
    );
}

const DD_P1: &str = "Jirachi|||NoAbility|doomdesire,splash|Serious||N||||";
const FS_P1: &str = "Jirachi|||NoAbility|futuresight,splash|Serious||N||||";
const DD_P2: &str = "Blissey|||NoAbility|splash|Serious|252,,,,,|N||||";

/// MC57: DOOM DESIRE cast → idle → RESOLVE at the end of turn N+2, and the FUTURE SIGHT
/// twin — IDENTICAL boundary seeds, different STORED damage (DD 366 physical bp-120 /
/// FS 45 special bp-80 vs Blissey's huge SpD — the cast-time typeless snapshot). The
/// cast draws exactly ONE random(16) (no accuracy / no crit); the idle turn draws
/// nothing for the pending strike; the resolve draws ONE accuracy roll then applies the
/// STORED number. WRONG (pre-batch-4c: DD ran as an instant hit): every boundary
/// desyncs. STATE (714 → 348 / 714 → 669 at the N+2 boundary ONLY) + SEED (all three).
#[test]
fn doom_desire_and_future_sight_cast_idle_resolve_snapshot() {
    let d = dex();
    let script = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // cast
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // idle (Splash)
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // the resolve turn
    ];
    let expected_seeds = [
        "60880,31090,7619,34922",
        "10897,43434,54578,10901",
        "22534,42410,55299,35327",
    ];

    for (team, resolved_hp, label) in [(DD_P1, 348u16, "DD 366"), (FS_P1, 669u16, "FS 45")] {
        let mut battle =
            Battle::start_with_switchins(&opts_cg(team, DD_P2, "44317,42357,9927,48760"), &d)
                .expect("start");
        let out = battle.state_mut().unwrap().run_full_battle(&script, &d);
        assert_eq!(out.decisions[0].active[1].hp, 714, "{label}: the cast deals NOTHING now");
        assert_eq!(out.decisions[1].active[1].hp, 714, "{label}: the idle turn deals nothing");
        assert_eq!(
            out.decisions[2].active[1].hp, resolved_hp,
            "{label}: the STORED snapshot lands at the end of turn N+2"
        );
        for (i, exp) in expected_seeds.iter().enumerate() {
            assert_eq!(
                &seed_str(&out.decisions[i].seed_after), exp,
                "{label}: dec{i} — cast=1×random(16), idle=QC only, resolve=1×accuracy+QC → the real Showdown seeds"
            );
        }
    }
}

/// MC58: a DOUBLE-CAST while a strike is pending FAILS with ZERO move draws — but PP IS
/// still deducted (7 → 6; deductPP precedes the onTry fail) — and the ORIGINAL pending
/// strike resolves on schedule. The failed-cast turn's seed equals the plain idle turn's
/// (the fail is draw-free). WRONG (a second snapshot random(16) / no PP): seed / PP
/// desync. STATE (PP 6; the resolve still lands 366) + SEED.
#[test]
fn doom_desire_double_cast_fails_draw_free_but_deducts_pp() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(DD_P1, DD_P2, "44317,42357,9927,48760"), &d)
            .expect("start");
    let out = battle.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // cast
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // DOUBLE-CAST → fails
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // the resolve turn
        ],
        &d,
    );
    assert_eq!(out.decisions[1].active[0].move_pp[0], 6, "the failed double-cast STILL deducts PP (7 → 6)");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "10897,43434,54578,10901",
        "the double-cast turn is draw-free (== the plain idle turn's seed) → the real Showdown seed"
    );
    assert_eq!(out.decisions[2].active[1].hp, 348, "the ORIGINAL strike resolves on schedule (714 → 348)");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "22534,42410,55299,35327");
}

/// MC59: a RESOLVE KO defers the Quick Claw — the resolve turn draws the accuracy roll
/// ONLY (the strike faints the 60-HP Gengar at the order-11 residual → the endTurn Quick
/// Claw is deferred past the forced replacement, which draws it). Also pins DD's
/// TYPELESS-ness (a Steel move would be resisted by nothing relevant here; the point is
/// the LEVITATE GHOST takes the full typeless hit — no immunity ever). WRONG (a QC on
/// the resolve-KO turn / a typed chart read): the boundary seeds desync. STATE + SEED.
#[test]
fn doom_desire_resolve_ko_defers_the_quick_claw() {
    let d = dex();
    let p2 = "Gengar|||Levitate|splash|Serious|252,,,,,|N||||]Blissey|||NoAbility|splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(DD_P1, p2, "44317,42357,9927,48760"), &d)
            .expect("start");
    {
        let st = battle.state_mut().expect("state");
        let a = st.sides[1].active;
        st.sides[1].pokemon[a].hp = 60; // STATE-only inject: the resolve KOs
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // cast
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // idle
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // the resolve KOs
            ScriptDecision { p1: None, p2: Some(Choice::Switch(1)) }, // the forced replacement
        ],
        &d,
    );
    assert!(out.decisions[2].active[1].fainted, "the resolve KO'd the 60-HP (Levitate, Ghost) Gengar — typeless never misses the chart");
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "37635,3740,64462,10380",
        "the resolve-KO turn draws the accuracy roll ONLY (NO Quick Claw — deferred) → the real Showdown seed"
    );
    assert_eq!(
        seed_str(&out.decisions[3].seed_after),
        "22534,42410,55299,35327",
        "the forced-replacement boundary draws the deferred Quick Claw → the real Showdown seed"
    );
}

/// MC60: the RESIDUAL-ORDER composition — Wish (order 7) → the sand chip (8) → Leftovers
/// (10.4) → the FUTUREMOVE strike (11) LAST, all in ONE turn: the 150-HP Celebi ends at
/// 320 (Wish +170 → 320, sand −21 → 299, Leftovers +21 → 320) and the DD strike lands
/// on Tyranitar (404 → 324, the stored 80) AFTER every heal/chip. WRONG (the futuremove
/// at any earlier order): Celebi's HP arithmetic and/or the strike's position shifts →
/// the STATE (both HPs) and/or the SEED desyncs. STATE + SEED at every boundary.
#[test]
fn doom_desire_resolves_last_in_the_residual_order() {
    let d = dex();
    let p1 = "Celebi||Leftovers|NoAbility|doomdesire,wish,splash|Serious||N||||";
    let p2 = "Tyranitar||Leftovers|SandStream|splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    {
        let st = battle.state_mut().expect("state");
        let a = st.sides[0].active;
        st.sides[0].pokemon[a].hp = 150; // STATE-only inject: room for the Wish heal
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // DD cast (turn 1)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Wish (turn 2)
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // Splash (turn 3 — everything resolves)
        ],
        &d,
    );
    // Turn 1: sand −21 then Leftovers +21 → net 150. Turn 2: same. Turn 3: Wish +170
    // (→ 320) then sand −21 (→ 299) then Leftovers +21 (→ 320) then the DD strike.
    assert_eq!(out.decisions[0].active[0].hp, 150, "turn 1: sand −21 + Leftovers +21 = net 0");
    assert_eq!(out.decisions[1].active[0].hp, 150, "turn 2: net 0 again");
    assert_eq!(
        out.decisions[2].active[0].hp, 320,
        "turn 3: Wish(7) +170 → 320, sand(8) −21 → 299, Leftovers(10.4) +21 → 320 — the ORDER is load-bearing"
    );
    assert_eq!(
        out.decisions[2].active[1].hp, 324,
        "the DD strike (order 11, LAST) lands the stored 80 on Tyranitar (404 → 324)"
    );
    assert_eq!(seed_str(&out.decisions[0].seed_after), "60880,31090,7619,34922");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "10897,43434,54578,10901");
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "22534,42410,55299,35327",
        "the resolve turn: ONE accuracy roll + Quick Claw (the residual heals/chips are draw-free) → the real Showdown seed"
    );
}

// ============================================================================
// MC61-MC78 — MOVE-COVERAGE BATCH 5 (`gen3_move_coverage_batch5_v1`): the
// reactive fixed-damage family COUNTER / MIRROR COAT / ENDEAVOR, the
// variable-BP family RETURN / FRUSTRATION / FLAIL / REVERSAL / LOW KICK, and
// SLEEP TALK. Ground-truth seeds/state from the REAL Showdown probe
// `harness/probe_batch5_regression_rng.js` (re-run it after any PRNG /
// draw-order change and update the constants; MC77's ground truth is
// `probe_lens1_batch5_review.js` R3, MC78's is
// `probe_batch5_st_calls_roar_rng.js`). Draw/mechanic models settled by
// `harness/probe_batch5_{reactive,varbp,sleeptalk,reactive_edges}.js`.
// ============================================================================

const B5_CT_P1: &str = "Snorlax|||NoAbility|counter,splash|Serious|252,,,,,|N||||";
const B5_CT_P2: &str = "Skarmory|||NoAbility|drillpeck,splash|Serious|252,,,,,|N||||";

/// MC61: COUNTER returns 2× a landed foe PHYSICAL hit (Drill Peck 118 → Counter 236 —
/// the recorder's exact double), and the NEXT turn's Counter (the foe splashes) fails
/// with **ZERO draws** and a bare `|move|` line — the volatile's onStart RESETS the
/// record EVERY selection turn, so PREV-TURN damage never counts. WRONG (a model that
/// persists the record): the second Counter would return 236 again (STATE) and draw an
/// accuracy roll (SEED). STATE + SEED, both boundaries.
#[test]
fn counter_returns_double_the_physical_hit_and_resets_each_turn() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(B5_CT_P1, B5_CT_P2, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Drill Peck 118 → Counter 236
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Splash → Counter fails zero-draw
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[0].hp, 406, "Drill Peck dealt 118 (524 → 406)");
    assert_eq!(out.decisions[0].active[1].hp, 98, "Counter returned EXACTLY 2×118 = 236 (334 → 98)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "22534,42410,55299,35327",
        "the landed-counter turn: DP acc+crit+dmg + Counter's ONE accuracy roll (NO crit/damage) + QC"
    );
    assert_eq!(out.decisions[1].active[1].hp, 98, "the un-armed Counter deals NOTHING (prev-turn reset)");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "7184,5868,30814,34654",
        "the fail turn draws ZERO for Counter (only the Quick Claw) — the onTry zero-draw fail"
    );
}

/// MC62: the WRONG-CATEGORY recorder gates — a SPECIAL hit (Surf) does NOT arm Counter,
/// a PHYSICAL hit (Return) does NOT arm Mirror Coat; both executions fail ZERO-DRAW.
/// WRONG (a category-blind recorder): either would return 2× (STATE) + draw accuracy
/// (SEED). SEED pins on both boundaries.
#[test]
fn reactive_wrong_category_hits_do_not_arm() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|counter,mirrorcoat|Serious|252,,,,,|N||||";
    let p2 = "Skarmory|||NoAbility|surf,return|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Counter vs Surf (special) → fail
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // Mirror Coat vs Return (physical) → fail
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[1].hp, 334, "Counter vs a SPECIAL hit deals nothing");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "37635,3740,64462,10380");
    assert_eq!(out.decisions[1].active[1].hp, 334, "Mirror Coat vs a PHYSICAL hit deals nothing");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "5621,5056,41416,14688");
}

/// MC63: the RETURN-FIRE type immunity — an ARMED Counter (Shadow Ball, Ghost =
/// PHYSICAL in gen3, arms it) into a GHOST reports `-immune` AFTER its accuracy draw
/// (Fighting → Ghost 0×); an ARMED Mirror Coat (Crunch, Dark = SPECIAL, arms it) into
/// a DARK likewise (Psychic → Dark 0×). ZERO damage either way, the accuracy roll
/// consumed. STATE + SEED.
#[test]
fn armed_counter_and_mirror_coat_respect_the_return_fire_type_immunity() {
    let d = dex();
    // (a) Counter → Ghost.
    let p1 = "Machamp|||NoAbility|counter,splash|Serious|252,,,,,|N||||";
    let p2 = "Gengar|||Levitate|shadowball,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].hp, 299, "Shadow Ball dealt 85 (384 → 299) — Counter ARMED");
    assert_eq!(out.decisions[0].active[1].hp, 324, "the Fighting return fire is IMMUNE into the Ghost");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "7184,5868,30814,34654");

    // (b) Mirror Coat → Dark.
    let p1 = "Blissey|||NoAbility|mirrorcoat,splash|Serious|252,,,,,|N||||";
    let p2 = "Tyranitar|||NoAbility|crunch,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].hp, 640, "Crunch dealt 74 (714 → 640) — Mirror Coat ARMED");
    assert_eq!(out.decisions[0].active[1].hp, 404, "the Psychic return fire is IMMUNE into the Dark");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "7184,5868,30814,34654");
}

/// MC64: a foe physical hit ABSORBED by the counter user's own SUBSTITUTE is NOT
/// recorded (the mon's Damage event never fires behind a sub) → the Counter fails
/// ZERO-DRAW. WRONG (a recorder keyed on the raw hit): Counter returns 2×118 (STATE) +
/// draws accuracy (SEED). STATE (sub HP 131−118=13, foe untouched) + SEED.
#[test]
fn sub_absorbed_hit_does_not_arm_counter() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|counter,substitute,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, B5_CT_P2, "44317,42357,9927,48760"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // sub up (Skarm splashes)
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Drill Peck into the SUB → Counter fails
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[0].hp, 393, "the sub cost 131 (524 → 393)");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "61255,39458,1834,64539");
    assert_eq!(out.decisions[1].sub_hp[0], 21, "the sub absorbed Drill Peck's 110-roll (131 → 21)");
    assert_eq!(out.decisions[1].active[0].hp, 393, "the MON took nothing behind the sub");
    assert_eq!(out.decisions[1].active[1].hp, 334, "the un-armed Counter deals NOTHING");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "22534,42410,55299,35327",
        "Counter drew ZERO (the sub-absorbed hit never armed it)"
    );
}

/// MC65: (a) SEISMIC TOSS (fixed damage, Fighting → the gen3 type-derived category is
/// PHYSICAL) IS countered — 2×100 = 200; (b) BEAT UP's strikes (Special) arm MIRROR
/// COAT with 2× the **LAST STRIKE ONLY** (the per-hit OVERWRITE — strikes 17/89/9 →
/// the return is 18, NOT 230). Probe `probe_batch5_reactive_edges.js`. STATE + SEED.
#[test]
fn fixed_damage_is_countered_and_beat_up_arms_mirror_coat_with_the_last_strike() {
    let d = dex();
    // (a) Seismic Toss countered.
    let p2 = "Blissey|||NoAbility|seismictoss,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(B5_CT_P1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].hp, 424, "Seismic Toss dealt the level 100");
    assert_eq!(out.decisions[0].active[1].hp, 514, "Counter returned 2×100 = 200 (714 → 514)");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "10897,43434,54578,10901");

    // (b) Beat Up → Mirror Coat, 2× the LAST strike.
    let p1 = "Blissey|||NoAbility|mirrorcoat,splash|Serious|252,,,,,|N||||";
    let p2 = "Smeargle|||NoAbility|beatup,splash|Serious|252,,,,,|N||||]Snorlax|||NoAbility|splash|Serious|252,,,,,|N||||]Blissey|||NoAbility|splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].hp, 599, "three Beat Up strikes: 17 + 89 + 9 (714 → 599)");
    assert_eq!(
        out.decisions[0].active[1].hp, 296,
        "Mirror Coat returned 2× the LAST strike = 18 (314 → 296) — NOT 2× the total"
    );
    assert_eq!(seed_str(&out.decisions[0].seed_after), "50157,35106,3193,41379");
}

/// MC66: the COUNTER-MIRROR at an EQUAL SPEED — the both-counter turn draws +4 vs the
/// both-splash control (the order-5 beforeTurnMove pair tie-shuffle + the 2 trailing
/// runAction Updates + the residual `duration:1` handler tie). Both counters FAIL
/// (nothing dealt damage). SEED pins on both boundaries (the control anchors the
/// baseline; the delta is the counter machinery's draws).
#[test]
fn counter_mirror_speed_tie_draws_the_before_turn_and_residual_ties() {
    let d = dex();
    let p2 = "Snorlax|||NoAbility|counter,splash|Serious|252,,,,,|N||||";
    // The tied-lead construction shuffles pre-date the seeded start: the probe's
    // printed initSeed (37635,…) is the pre-first-decision state.
    let mut battle = Battle::start_with_switchins(&opts_cg(B5_CT_P1, p2, "37635,3740,64462,10380"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // both-splash CONTROL
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // both-counter (+4 draws)
        ],
        &d,
    );
    assert_eq!(seed_str(&out.decisions[0].seed_after), "60443,61849,18300,733", "the control turn");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "63246,52257,55308,49838",
        "the both-counter turn: the order-5 sort tie + 2 trailing Updates + the residual duration tie"
    );
    assert_eq!(out.decisions[1].active[0].hp, 524, "both counters fail — no damage either way");
    assert_eq!(out.decisions[1].active[1].hp, 524, "both counters fail — no damage either way");
}

/// MC67: ENDEAVOR sets the target's hp to EXACTLY the user's hp (the delta
/// `target.hp − user.hp`, accuracy-only draw), and the follow-up at EQUAL hp FAILS
/// (`hp >= target.hp` — EQUALITY INCLUDED) with **ZERO draws** + `|-fail|`. PP is
/// consumed on BOTH (the onTry fail sits after deductPP). STATE + SEED.
#[test]
fn endeavor_sets_target_hp_to_users_and_fails_at_equality_zero_draw() {
    let d = dex();
    let p1 = "Swellow|||NoAbility|endeavor,splash|Serious||N||||";
    let p2 = "Snorlax|||NoAbility|splash,drillpeck|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    battle.state_mut().unwrap().sides[0].pokemon[0].hp = 50; // the probe's inject
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Endeavor: 524 → 50
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // 50 vs 50 → EQUALITY fail
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[1].hp, 50, "Endeavor left the Snorlax at EXACTLY the user's 50");
    assert_eq!(out.decisions[0].active[0].move_pp[0], 7, "the landed Endeavor consumed 1 PP");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "60880,31090,7619,34922");
    assert_eq!(out.decisions[1].active[1].hp, 50, "the equality Endeavor deals nothing");
    assert_eq!(out.decisions[1].active[0].move_pp[0], 6, "the FAILED Endeavor still consumed 1 PP");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "10897,43434,54578,10901",
        "the equality fail draws ZERO (onTry precedes accuracy)"
    );
}

/// MC68: (a) ENDEAVOR into a GHOST is `-immune` AFTER its accuracy draw (Normal →
/// Ghost 0×, ignoreImmunity false); (b) ENDEAVOR into a SUBSTITUTE computes the delta
/// from the MON's hp (393 − 50 = 343) and the number lands on the SUB — it BREAKS with
/// NO carry (the mon's hp unchanged). STATE + SEED.
#[test]
fn endeavor_ghost_immunity_and_substitute_break_no_carry() {
    let d = dex();
    // (a) → Ghost.
    let p1 = "Swellow|||NoAbility|endeavor,splash|Serious||N||||";
    let p2 = "Gengar|||Levitate|splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    battle.state_mut().unwrap().sides[0].pokemon[0].hp = 50;
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[1].hp, 324, "the Ghost takes ZERO (immune)");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "60880,31090,7619,34922");

    // (b) → Substitute.
    let p2 = "Snorlax|||NoAbility|substitute,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    battle.state_mut().unwrap().sides[0].pokemon[0].hp = 50;
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Snorlax subs
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Endeavor into the sub
        ],
        &d,
    );
    assert_eq!(out.decisions[1].sub_hp[1], 0, "the 343 delta BROKE the 131-HP sub");
    assert_eq!(out.decisions[1].active[1].hp, 393, "NO carry — the mon's hp is untouched");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "10897,43434,54578,10901");
}

/// MC69: RETURN / FRUSTRATION happiness extremes — h255 Return = BP 102 (60 dmg into
/// this Skarmory), h3 Return = the `|| 1`-clamped BP **1** (a 1-damage HIT, not a
/// fail), h0 Frustration = BP 102 — and ALL THREE runs end at the SAME post-turn seed
/// (the BP is a draw-free state read: acc+crit+dmg+QC regardless). STATE + the
/// seed-EQUALITY draw-neutrality proof.
#[test]
fn return_frustration_happiness_extremes_are_draw_neutral() {
    let d = dex();
    let p2 = "Skarmory|||NoAbility|splash,drillpeck|Serious|252,,,,,|N||||";
    let run = |p1: &str| {
        let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
            .expect("start");
        let st = battle.state_mut().expect("state");
        let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
        (out.decisions[0].active[1].hp, seed_str(&out.decisions[0].seed_after))
    };
    let (hp_h255, seed_h255) = run("Tauros|||NoAbility|return,splash|Serious|,252,,,,|N||||");
    let (hp_h3, seed_h3) = run("Tauros|||NoAbility|return,splash|Serious|,252,,,,|N||||3");
    let (hp_h0, seed_h0) = run("Tauros|||NoAbility|return,splash|Serious|,252,,,,|N||||0");
    let (hp_f0, seed_f0) = run("Tauros|||NoAbility|frustration,splash|Serious|,252,,,,|N||||0");
    assert_eq!(hp_h255, 274, "h255 Return (BP 102) dealt 60 (334 → 274)");
    assert_eq!(hp_h3, 333, "h3 Return: floor(30/25) = BP 1 → EXACTLY 1 damage");
    assert_eq!(hp_h0, 333, "h0 Return: floor(0/25) = 0 → the `|| 1` CLAMP → BP 1 (a HIT, not a fail)");
    assert_eq!(hp_f0, 274, "h0 Frustration (BP 102) mirrors the h255 Return");
    assert_eq!(seed_h255, "37635,3740,64462,10380", "the real Showdown post-turn seed");
    assert_eq!(seed_h3, seed_h255, "BP 1 vs BP 102: BYTE-IDENTICAL seeds — the BP is draw-free");
    assert_eq!(seed_h0, seed_h255, "the clamped BP-1 hit shares the identical draw chain");
    assert_eq!(seed_f0, seed_h255, "Frustration shares the same draw chain");
}

/// MC70: the FLAIL band boundary at Snorlax maxhp 524 — hp 21 (ratio ⌊48·21/524⌋ = 1
/// → BP 200) vs hp 22 (ratio 2 → BP 150): different damage (125 vs 94) at the
/// BYTE-IDENTICAL post-turn seed. A ±1 band-threshold error flips the BP at exactly
/// this boundary. STATE + the seed-equality proof.
#[test]
fn flail_band_boundary_bp200_vs_bp150() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|flail,splash|Serious|252,252,,,,|N||||";
    let p2 = "Skarmory|||NoAbility|splash,drillpeck|Serious|252,,,,,|N||||";
    let run = |hp: u16| {
        let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
            .expect("start");
        battle.state_mut().unwrap().sides[0].pokemon[0].hp = hp;
        let st = battle.state_mut().expect("state");
        let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
        (out.decisions[0].active[1].hp, seed_str(&out.decisions[0].seed_after))
    };
    let (hp21, seed21) = run(21);
    let (hp22, seed22) = run(22);
    assert_eq!(hp21, 209, "hp 21 → ratio 1 → BP 200 → 125 damage (334 → 209)");
    assert_eq!(hp22, 240, "hp 22 → ratio 2 → BP 150 → 94 damage (334 → 240)");
    assert_eq!(seed21, "37635,3740,64462,10380", "the real Showdown post-turn seed");
    assert_eq!(seed22, seed21, "the band only changes the BP — the draws are identical");
}

/// MC71: the LOW KICK weight ladder — Pichu (20 hg → BP 20, 31 dmg), Wobbuffet
/// (285 hg → BP 60, 19 dmg), Snorlax (4600 hg → BP 120, 145 dmg ×2 SE), all at the
/// BYTE-IDENTICAL post-turn seed (the weight read is draw-free). A wrong `weighthg`
/// datum or ladder cutoff shows as the wrong rung's damage. STATE + seed equality.
#[test]
fn low_kick_weight_ladder_is_draw_neutral() {
    let d = dex();
    let p1 = "Blissey|||NoAbility|lowkick,splash|Serious|,252,,,,|N||||";
    let run = |p2: &str| {
        let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
            .expect("start");
        let st = battle.state_mut().expect("state");
        let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
        (out.decisions[0].active[1].hp, seed_str(&out.decisions[0].seed_after))
    };
    let (pichu, s1) = run("Pichu|||NoAbility|splash|Serious|252,,,,,|N||||");
    let (wobb, s2) = run("Wobbuffet|||NoAbility|splash|Serious|252,,,,,|N||||");
    let (lax, s3) = run("Snorlax|||NoAbility|splash|Serious|252,,,,,|N||||");
    assert_eq!(pichu, 213, "Pichu 20 hg → BP 20 (244 → 213)");
    assert_eq!(wobb, 565, "Wobbuffet 285 hg → BP 60 (584 → 565)");
    assert_eq!(lax, 379, "Snorlax 4600 hg → BP 120 (524 → 379)");
    assert_eq!(s1, "37635,3740,64462,10380", "the real Showdown post-turn seed");
    assert_eq!(s2, s1, "the weight rung is draw-free");
    assert_eq!(s3, s1, "the weight rung is draw-free");
}

/// MC72: SLEEP TALK — (a) the n=1 pool ([rest] — Sleep Talk excludes itself) STILL
/// draws the `sample` (`random(1)`) and the picked Rest-while-asleep SILENTLY no-ops
/// (no heal, no counter reset, no `-fail`); the picked move's PP is NEVER consumed
/// (rest stays 15 — only the direct dec0 use paid). (b) the EMPTY pool ([sleeptalk,
/// solarbeam] — solarbeam is `charge`-flagged) fails `[still]` + `-fail` with ZERO
/// Sleep-Talk draws, and the slp counter still ticks (a sleep-talking turn decrements).
/// STATE (slp counter + PP) + SEED.
#[test]
fn sleep_talk_n1_sample_and_empty_pool() {
    let d = dex();
    // (a) n=1 pool: a Rest-based sleep (fixed 3), then Sleep Talk picks Rest.
    let p1 = "Snorlax|||NoAbility|sleeptalk,rest|Serious|252,,,,,|N||||";
    let p2 = "Skarmory|||NoAbility|drillpeck,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    battle.state_mut().unwrap().sides[0].pokemon[0].hp = 300;
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Rest → Sleep(3), full heal
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Sleep Talk: sample(1) → Rest no-op
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[0].status, Some(Status::Sleep(3)), "Rest → the fixed Sleep(3)");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "22534,42410,55299,35327");
    assert_eq!(
        out.decisions[1].active[0].status,
        Some(Status::Sleep(2)),
        "the sleep-talking turn still decrements the counter (3 → 2)"
    );
    assert_eq!(out.decisions[1].active[0].move_pp[0], 15, "Sleep Talk's own PP −1");
    assert_eq!(
        out.decisions[1].active[0].move_pp[1], 15,
        "the PICKED Rest's PP is NEVER consumed (15 from the direct dec0 use only)"
    );
    assert_eq!(out.decisions[1].active[0].hp, 524, "the called Rest-while-asleep silently no-ops (already full)");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "37112,13693,28533,21721",
        "the n=1 pool STILL draws the sample (random(1)) — then the Rest no-op draws nothing"
    );

    // (a2) the called Rest on a DAMAGED sleeper — the silent no-op must NOT heal /
    // redraw random(2,6) / reset the counter (a full-HP board masks this behind the
    // full-HP guard; here Drill Peck lands the same turn BEFORE the talk).
    let p1 = "Snorlax|||NoAbility|sleeptalk,rest|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    battle.state_mut().unwrap().sides[0].pokemon[0].hp = 300;
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Rest → Sleep(3), full heal
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Drill Peck 119, then talk → Rest no-op
        ],
        &d,
    );
    assert_eq!(
        out.decisions[1].active[0].hp, 405,
        "the called Rest-while-asleep did NOT heal the damaged sleeper (524 − 119 = 405)"
    );
    assert_eq!(
        out.decisions[1].active[0].status,
        Some(Status::Sleep(2)),
        "…and did NOT reset the sleep counter (3 → 2 by the talk decrement only)"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "17940,16623,13080,40722",
        "…and drew NOTHING (no random(2,6) — the asleep guard precedes the Rest body)"
    );

    // (b) the EMPTY pool: a Spore-based sleep, [sleeptalk, solarbeam] → pool [].
    let p1 = "Snorlax|||NoAbility|sleeptalk,solarbeam|Serious|252,,,,,|N||||";
    let p2 = "Breloom|||NoAbility|spore,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Spore lands; the queued Sleep Talk PROCEEDS asleep → empty pool fail
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // still asleep → empty pool fail again
        ],
        &d,
    );
    assert_eq!(
        out.decisions[0].active[0].status,
        Some(Status::Sleep(4)),
        "the Spore rolled 5; the SAME-TURN queued Sleep Talk decremented it to 4 and PROCEEDED (sleepUsable)"
    );
    assert_eq!(seed_str(&out.decisions[0].seed_after), "10897,43434,54578,10901");
    assert_eq!(out.decisions[1].active[0].status, Some(Status::Sleep(3)));
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "37635,3740,64462,10380",
        "the empty-pool fail draws ZERO Sleep-Talk draws (only the Quick Claw)"
    );
}

/// MC73: the CHOICE-LOCKED Sleep Talk — a Choice Band sleeper's FIRST Sleep Talk of a
/// lock samples + executes (the lock records Sleep Talk ITSELF — the lock this very
/// use sets does NOT count); every LATER one fails `[still]` + `-fail` BEFORE the
/// sample (no sample draw). WRONG (gating on the post-set lock): the FIRST use would
/// fail too. STATE + SEED, all three boundaries.
#[test]
fn choice_band_sleep_talk_works_once_then_fails_on_the_lock() {
    let d = dex();
    let p1 = "Snorlax||ChoiceBand|NoAbility|sleeptalk,bodyslam|Serious|252,252,,,,|N||||";
    let p2 = "Breloom|||NoAbility|spore,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Spore; the queued Sleep Talk #1 locks + samples + Body Slams
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Sleep Talk #2 → the choicelock [still]+fail
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Sleep Talk #3 → still locked
        ],
        &d,
    );
    assert_eq!(
        out.decisions[0].active[1].hp, 63,
        "Sleep Talk #1 called Body Slam (324 → 63) — the first use of the lock WORKS"
    );
    assert_eq!(seed_str(&out.decisions[0].seed_after), "5621,5056,41416,14688");
    assert_eq!(out.decisions[1].active[1].hp, 63, "Sleep Talk #2 fails on the PRIOR-turn lock");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "50157,35106,3193,41379",
        "the choicelock fail draws NO sample"
    );
    assert_eq!(seed_str(&out.decisions[2].seed_after), "17940,16623,13080,40722");
}

/// MC74: the 0-PP PICK — the pool keeps a 0-PP member (NO pp filter at build), the
/// n=1 `sample` DRAWS, and the pick STOPS at `|cant|…|nopp|bodyslam` (the turn is
/// wasted, NO further draws, the picked move never runs). WRONG (a pp-filtered pool):
/// the pool would be empty → a `[still]`+`-fail` with NO sample → a seed desync.
/// STATE + SEED.
#[test]
fn sleep_talk_zero_pp_pick_wastes_the_turn_after_the_sample() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|sleeptalk,bodyslam|Serious|252,,,,,|N||||";
    let p2 = "Breloom|||NoAbility|spore,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    battle.state_mut().unwrap().sides[0].pokemon[0].move_pp[1] = 0; // Body Slam at 0 PP
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Spore; Sleep Talk samples → nopp
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // again
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[1].hp, 324, "the 0-PP pick never executes (Breloom untouched)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37635,3740,64462,10380",
        "the sample DREW (random(1)) then the nopp cant stopped — no further draws"
    );
    assert_eq!(seed_str(&out.decisions[1].seed_after), "7184,5868,30814,34654");
}

/// MC75: the slp `skippedTime` RESTORE — each sleep-talking turn decrements the
/// counter AND banks a skip (3 → 2 → 1, skipped 2); a switch out + back RESTORES
/// `time += skippedTime` (the counter reads 3 again on re-entry), so the later wake
/// timing shifts. WRONG (no skippedTime): the re-entered sleeper wakes 2 turns early.
/// STATE (the counter timeline) + SEED (every boundary).
#[test]
fn sleep_talk_skipped_time_restores_on_switch_in() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|sleeptalk,rest|Serious|252,,,,,|N||||]Blissey|||NoAbility|splash|Serious|252,,,,,|N||||";
    let p2 = "Skarmory|||NoAbility|drillpeck,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    battle.state_mut().unwrap().sides[0].pokemon[0].hp = 300;
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),   // Rest → Sleep(3)
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),   // talk (2, skipped 1)
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),   // talk (1, skipped 2)
            ScriptDecision::both(Choice::Switch(1), Choice::Move(1)), // pivot out
            ScriptDecision::both(Choice::Switch(1), Choice::Move(1)), // pivot back → RESTORED to 3
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),   // talk (2 again)
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[0].status, Some(Status::Sleep(3)));
    assert_eq!(out.decisions[1].active[0].status, Some(Status::Sleep(2)));
    assert_eq!(out.decisions[2].active[0].status, Some(Status::Sleep(1)));
    assert_eq!(
        out.decisions[4].active[0].status,
        Some(Status::Sleep(3)),
        "the switch-in RESTORE: time += skippedTime (1 + 2 = 3)"
    );
    assert_eq!(out.decisions[5].active[0].status, Some(Status::Sleep(2)));
    assert_eq!(seed_str(&out.decisions[0].seed_after), "22534,42410,55299,35327");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "37112,13693,28533,21721");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "50157,35106,3193,41379");
    assert_eq!(seed_str(&out.decisions[3].seed_after), "17940,16623,13080,40722");
    assert_eq!(seed_str(&out.decisions[4].seed_after), "60443,61849,18300,733");
    assert_eq!(seed_str(&out.decisions[5].seed_after), "54523,22811,31582,9991");
}

/// MC76: a FIXED-DAMAGE hit sets the Focus-Punch user's `lostFocus` — the batch-5 e2e
/// admission bug (e2e_202 dec44: the sim's Blissey Seismic Toss into a Focus-Punch
/// Dragonite cants the punch; the port's `run_fixed_damage_move` never set `lost_focus`,
/// so the punch LANDED and KO'd the Blissey — a LATENT gap unreachable while the
/// fixed-damage family was blocklist-shadowed out of the e2e picker). WRONG (pre-fix):
/// the punch runs (acc+crit+dmg draws + a huge hit) → STATE + SEED desync. Ground truth
/// `harness/probe_batch5_regression_rng.js` (MC76).
#[test]
fn fixed_damage_hit_cancels_a_queued_focus_punch() {
    let d = dex();
    let p1 = "Blissey|||NoAbility|seismictoss,splash|Serious|252,,,,,|N||||";
    let p2 = "Dragonite|||NoAbility|focuspunch,splash|Serious|252,252,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[1].hp, 286, "Seismic Toss dealt 100 (386 → 286)");
    assert_eq!(
        out.decisions[0].active[0].hp, 714,
        "the Focus Punch was CANT'd (lostFocus from the fixed-damage hit) — Blissey untouched"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "60880,31090,7619,34922",
        "the cant'd punch draws NOTHING (only the ST accuracy + Quick Claw)"
    );
}

/// MC78: a Sleep-Talk-CALLED ROAR drags the foe — the called move's resolution
/// PROPAGATES through the recursive `run_move` (`force_switch_foe` rides out of the
/// Sleep Talk arm to the runAction-tail `drag_in`), so an asleep RestTalker's called
/// Roar phazes exactly like a selected Roar (the n=1 `sample` pulls the lone bench
/// mon). The review's coverage gap: the batch-5 golden's 23 scenarios call only
/// Rest/attacks via Sleep Talk — the called-Roar drag composition (e2e-INCLUDED, since
/// phaze is admitted + `sleepTalkPoolModeled` passes a Roar carrier) was coded but
/// unpinned. Scenario: Suicune [Sleep Talk, Roar] (pool = [roar] — Sleep Talk itself is
/// flags.nosleeptalk → the n=1 sample) vs Parasect [Spore, Splash] + a bench Snorlax.
/// dec0 = ST awake (the silent zero-draw onTry fail) + Spore lands; dec1 = ST asleep →
/// |cant|slp + sleepUsable-proceeds → sample picks Roar → the CALLED Roar draws its
/// accuracy then the n=1 drag `sample` pulls Snorlax in. Ground truth
/// `harness/probe_batch5_st_calls_roar_rng.js` (raw seed [7,11,13,17]).
#[test]
fn sleep_talk_called_roar_drags_the_foe() {
    let d = dex();
    let p1 = "Suicune|||NoAbility|sleeptalk,roar|Serious|252,,,,,|N||||";
    let p2 = "Parasect|||NoAbility|spore,splash|Serious|252,,,,,|N||||\
              ]Snorlax|||NoAbility|splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // ST awake-fail ; Spore
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // ST asleep → called Roar ; Splash
        ],
        &d,
    );
    assert_eq!(
        out.decisions[0].active[0].status,
        Some(Status::Sleep(5)),
        "Spore slept Suicune (the awake Sleep Talk fails silently, zero draws)"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "10897,43434,54578,10901",
        "dec0: the awake Sleep Talk draws NOTHING; Spore acc + slp random(2,6) + Quick Claw"
    );
    assert_eq!(
        st.sides[1].pokemon[st.sides[1].active].species_id, "snorlax",
        "the CALLED Roar dragged the lone bench Snorlax in (n=1 sample)"
    );
    assert_eq!(
        out.decisions[1].active[0].status,
        Some(Status::Sleep(4)),
        "Suicune stayed asleep through the called Roar (the counter decremented once)"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "37112,13693,28533,21721",
        "dec1: [slp-cant draw-free, the n=1 Sleep Talk sample, the called Roar's accuracy, \
         the n=1 drag sample, Quick Claw] — the composition's exact draw chain"
    );
}

/// MC77: a FOCUS-BAND-proc'd ZERO-damage hit does NOT arm Counter/Mirror Coat — the
/// sim's `runEvent('Damage')` BREAKS its handler chain on a falsy relayVar
/// (battle.js:695) BEFORE the counter's priority-−101 recorder, so a 1-HP Focus-Band
/// holder whose lethal incoming hit is reduced to 0 keeps Counter UN-ARMED: the
/// Counter is a bare zero-draw `|move|` fail (NO accuracy roll, foe untouched).
/// WRONG (pre-fix): `record_reactive_hit` armed `Some(0)` unconditionally → the port's
/// Counter proceeded past the onTry gate and drew ONE extra accuracy roll (a latent
/// SEED desync; e2e-unreachable today — 0 Focus Band carriers in the team pool).
/// Ground truth: `harness/probe_lens1_batch5_review.js` R3, raw seed [6,6,6,6] →
/// pre-decision seed 13567,1259,49329,55073; the FB-proc turn draws EXACTLY
/// [DP acc, crit, dmg, FB roll, Quick Claw] → 24097,1527,37675,7388.
#[test]
fn focus_band_zero_damage_hit_does_not_arm_counter() {
    let d = dex();
    let p1 = "Snorlax||focusband|NoAbility|counter,splash|Serious|252,,,,,|N||||";
    let p2 = "Skarmory|||NoAbility|drillpeck,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, "13567,1259,49329,55073"), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    // The probe injects hp=1 on the FB holder post-construction (the sim's applyActs).
    st.sides[0].pokemon[0].hp = 1;
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        out.decisions[0].active[0].hp, 1,
        "the Focus Band proc reduced the lethal Drill Peck to 0 dealt (holder stays at 1 HP)"
    );
    assert_eq!(
        out.decisions[0].active[1].hp, 334,
        "Counter stayed UN-ARMED by the 0-damage hit — the return fire never happened"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "24097,1527,37675,7388",
        "the un-armed Counter is a ZERO-draw fail (no accuracy roll) — \
         only [DP acc, crit, dmg, FB roll, Quick Claw] drew"
    );
}

// ============================================================================
// MC79-MC98 — MOVE-COVERAGE BATCH 6 (`gen3_move_coverage_batch6_v1`): the FINAL
// UNMODELED tail — ENCORE / DESTINY BOND / ENDURE / PERISH SONG / MEAN LOOK
// family / BELLY DRUM / CHARGE / MEMENTO / MIMIC / PAIN SPLIT / PSYCH UP.
// Ground-truth seeds/state from the REAL Showdown probe
// `harness/probe_batch6_regression_rng.js` (re-run it after any PRNG /
// draw-order change and update the constants). Draw/mechanic models settled by
// `harness/probe_batch6_{locks,field_trap,utility,dexfacts}.js`.
// ============================================================================

const B6_EN_JOLTEON: &str = "Jolteon|||NoAbility|encore,thunderbolt|Serious|252,,,252,,252|N||||";
const B6_EN_SNORLAX: &str = "Snorlax|||NoAbility|splash,bodyslam|Serious|252,,,,,|N||||";

/// MC79: the FASTER encore user (the target has NOT moved this turn → the gen4-inherited
/// onStart's `!willMove → duration++` does NOT fire → stored = rolled) + the
/// `onOverrideAction` EXECUTION OVERRIDE — the Snorlax QUEUED Body Slam the very turn
/// the encore landed, and the queued move executed AS the encored Splash: the ENCORED
/// slot's PP deducts (splash 63→62), Body Slam's is UNTOUCHED (24). The ENCORE column
/// then ticks 4→3→2→gone at each residual. WRONG (no override): Body Slam runs (a huge
/// HP/state desync); WRONG (+1 duration branch): every ENCORE column is off by one
/// (MC80 is the SAME seeds with the OTHER branch — the perturbation pair).
#[test]
fn faster_encore_stores_rolled_and_overrides_the_queued_move() {
    let d = dex();
    let mut battle = Battle::start_with_switchins(
        &opts_cg(B6_EN_JOLTEON, B6_EN_SNORLAX, "44317,42357,9927,48760"),
        &d,
    )
    .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // tbolt / splash (lastMove)
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // encore; queued BODYSLAM → overridden to splash
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    // dec1 — the encore landed + the override executed splash (its PP deducted).
    assert_eq!(out.decisions[1].encore, [0, 4], "dec1: stored = ROLLED (5) − the landing residual tick = 4");
    assert_eq!(
        out.decisions[1].active[1].move_pp,
        [62, 24, -1, -1],
        "dec1: the OVERRIDE deducted the ENCORED splash (63→62); the queued Body Slam is untouched"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "5621,5056,41416,14688",
        "dec1: encore acc + durationCallback random(3,7) + QC (the landed-encore draw model)"
    );
    // The residual tick timeline (4 → 3 → 2 → expired).
    assert_eq!(out.decisions[2].encore, [0, 3]);
    assert_eq!(seed_str(&out.decisions[2].seed_after), "54523,22811,31582,9991");
    assert_eq!(out.decisions[3].encore, [0, 2]);
    assert_eq!(seed_str(&out.decisions[3].seed_after), "50037,28344,49354,43194");
    assert_eq!(out.decisions[4].encore, [0, 0], "dec4: the KO turn (the lock is moot)");
    assert_eq!(seed_str(&out.decisions[4].seed_after), "63246,52257,55308,49838");
    assert_eq!(out.winner, Some(0), "Jolteon sweeps");
}

/// MC80: the SLOWER encore user (the target ALREADY moved this turn → `!willMove` →
/// `duration++` → stored = rolled + 1). The SAME init seed + the SAME draw stream as
/// MC79's landing turn (byte-identical boundary seeds), but the ENCORE column reads
/// **5** where MC79 read 4 — the branch is STATE-only at an equal draw count (the
/// perfect perturbation pin pair). WRONG (either fixed model): one of MC79/MC80 fails.
#[test]
fn slower_encore_stores_rolled_plus_one() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|encore,bodyslam|Serious|252,,,,,|N||||";
    let p2 = "Jolteon|||NoAbility|splash,thunderbolt|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Jolteon splashes FIRST
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // the SLOWER encore
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    assert_eq!(
        out.decisions[1].encore,
        [0, 5],
        "dec1: stored = rolled + 1 (the !willMove branch) − the landing tick = 5 (MC79 read 4 \
         at the IDENTICAL boundary seed — the branch is state-only)"
    );
    assert_eq!(seed_str(&out.decisions[1].seed_after), "5621,5056,41416,14688");
    assert_eq!(out.decisions[2].encore, [0, 4]);
    assert_eq!(seed_str(&out.decisions[2].seed_after), "54523,22811,31582,9991");
}

/// MC81: the encore FAIL SPLIT is NOT uniform — the NO-LASTMOVE fail consumes the
/// accuracy roll AND the durationCallback `random(3,7)` (the callback fires before
/// onStart rejects), while the ALREADY-ENCORED fail consumes the accuracy roll ONLY
/// (`addVolatile` returns false BEFORE the durationCallback), the existing volatile
/// UNCHANGED. A uniform fail model desyncs the LCG on one of the two boundaries.
#[test]
fn encore_fail_split_draws_differ_by_the_duration_callback() {
    let d = dex();
    let mut battle = Battle::start_with_switchins(
        &opts_cg(B6_EN_JOLTEON, B6_EN_SNORLAX, "44317,42357,9927,48760"),
        &d,
    )
    .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // NO-LASTMOVE fail: acc + dur
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // lands
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // ALREADY-ENCORED fail: acc ONLY
        ],
        &d,
    );
    assert_eq!(out.decisions[0].encore, [0, 0], "dec0: the no-lastMove encore FAILED");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "10897,43434,54578,10901",
        "dec0: acc + durationCallback + QC (the 2-draw fail form)"
    );
    assert_eq!(out.decisions[1].encore, [0, 3], "dec1: the encore LANDS (stored 4 − the tick)");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "7184,5868,30814,34654");
    assert_eq!(out.decisions[2].encore, [0, 2], "dec2: the existing volatile just ticks (UNCHANGED by the fail)");
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "5621,5056,41416,14688",
        "dec2: the already-encored fail draws the accuracy roll ONLY (1-draw fail form)"
    );
}

/// MC82: the encored slot hitting 0 PP removes the volatile EARLY at THAT residual
/// (`encore.onResidual`'s pp check — the `-end` fires the same turn even at a high
/// remaining duration), and a later re-encore into the still-lastMove 0-PP slot FAILS
/// with BOTH draws (acc + durationCallback — the onStart 0-PP reject fires after them).
#[test]
fn encore_zero_pp_ends_early_and_rejects_the_zero_pp_lastmove() {
    let d = dex();
    let mut battle = Battle::start_with_switchins(
        &opts_cg(B6_EN_JOLTEON, B6_EN_SNORLAX, "44317,42357,9927,48760"),
        &d,
    )
    .expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[1].pokemon[0].move_pp[0] = 2; // splash injected to 2 PP
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // splash 2→1 (lastMove)
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // encore lands; splash 1→0 → EARLY -end
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // re-encore into the 0-PP lastMove: FAIL
        ],
        &d,
    );
    assert_eq!(
        out.decisions[1].encore,
        [0, 0],
        "dec1: the encore -end'd EARLY at the same residual (the encored slot hit 0 PP)"
    );
    assert_eq!(out.decisions[1].active[1].move_pp, [0, 24, -1, -1]);
    assert_eq!(seed_str(&out.decisions[1].seed_after), "5621,5056,41416,14688");
    assert_eq!(out.decisions[2].encore, [0, 0], "dec2: the 0-PP-lastMove re-encore FAILED");
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "56830,34298,10811,30881",
        "dec2: the 0-PP-lastMove fail consumes acc AND the durationCallback"
    );
}

const B6_DB_GENGAR: &str = "Gengar|||Levitate|destinybond,splash|Serious||N||||";
const B6_DB_SNORLAX: &str = "Snorlax|||NoAbility|shadowball,splash|Serious|252,252,,,,|N||||";

/// MC83: DESTINY BOND cast-and-KO'd the SAME turn → the MUTUAL FAINT: Gengar's |faint|
/// FIRST, then `-activate|move: Destiny Bond`, then the killer Snorlax's |faint| — a
/// both-last-mons mutual faint is the gen-3 TIE. The cast is ZERO draws; the KO turn
/// draws Shadow Ball's acc/crit/dmg/secondary and NO Quick Claw. WRONG (no onFaint
/// chain): Snorlax survives and WINS (the MC84 outcome at a different seed).
#[test]
fn destiny_bond_mutual_faint_is_a_tie() {
    let d = dex();
    let mut battle = Battle::start_with_switchins(
        &opts_cg(B6_DB_GENGAR, B6_DB_SNORLAX, "44317,42357,9927,48760"),
        &d,
    )
    .expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[0].pokemon[0].hp = 120; // in Shadow Ball KO range
    let out = st.run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert!(out.decisions[0].active[0].fainted, "Gengar fainted to the Shadow Ball");
    assert!(out.decisions[0].active[1].fainted, "the KILLER fainted too (Destiny Bond)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37635,3740,64462,10380",
        "the cast is draw-free; the KO turn draws SB acc/crit/dmg/secondary, no QC, and the \
         mutual-faint chain itself is DRAW-FREE"
    );
    assert!(out.ended, "both last mons fainted");
    assert_eq!(out.winner, None, "the gen-3 TIE (winner='')");
}

/// MC84: the DB WINDOW CLOSES at the user's NEXT MOVE ATTEMPT (`onBeforeMove` −1): the
/// bond, then a SPLASH (removes the volatile), then the KO → NO mutual faint — the
/// killer survives at full HP and WINS. WRONG (a persistent volatile): a phantom
/// mutual faint (MC83's tie instead of the P2 win).
#[test]
fn destiny_bond_window_closes_at_the_next_move_attempt() {
    let d = dex();
    let mut battle = Battle::start_with_switchins(
        &opts_cg(B6_DB_GENGAR, B6_DB_SNORLAX, "44317,42357,9927,48760"),
        &d,
    )
    .expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[0].pokemon[0].hp = 120;
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // DB cast; Snorlax splashes
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // Gengar SPLASHES → the window closes
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Shadow Ball KOs → NO mutual faint
        ],
        &d,
    );
    assert_eq!(seed_str(&out.decisions[0].seed_after), "61255,39458,1834,64539", "the cast turn: QC only");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "60880,31090,7619,34922");
    assert!(out.decisions[2].active[0].fainted, "Gengar fainted");
    assert!(!out.decisions[2].active[1].fainted, "the killer did NOT faint (the window closed)");
    assert_eq!(out.decisions[2].active[1].hp, 524, "Snorlax untouched");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "7184,5868,30814,34654");
    assert_eq!(out.winner, Some(1), "P2 wins — no mutual faint");
}

/// MC85: a RESIDUAL (sand chip) KO does NOT trigger Destiny Bond — and the WHOLE turn
/// (DB cast + foe splash + sand chip + faint) is ZERO draws at distinct speeds: the
/// post-turn seed is LITERALLY the init seed. WRONG (a residual-triggered bond):
/// Tyranitar faints too (a tie instead of the P2 win).
#[test]
fn destiny_bond_is_not_triggered_by_a_residual_ko() {
    let d = dex();
    let ttar = "Tyranitar|||SandStream|splash,crunch|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(B6_DB_GENGAR, ttar, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[0].pokemon[0].hp = 15; // within the sand chip
    let out = st.run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert!(out.decisions[0].active[0].fainted, "the sand chip KO'd the bond holder");
    assert!(!out.decisions[0].active[1].fainted, "NO mutual faint on a residual KO");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "44317,42357,9927,48760",
        "the whole cast+chip+faint turn is ZERO draws (the seed is unchanged)"
    );
    assert_eq!(out.winner, Some(1));
}

/// MC86: ENDURE survives any MOVE damage at 1 HP and rides the SHARED protect `stall`
/// ladder: the first endure is draw-free (counter → 2), consecutive uses roll
/// `randomChance(1, counter)` escalating 2 → 4 → 8, and a FAILED roll leaves the user
/// unprotected (the Double-Edge KOs). Every successful endure turn ALSO adds the
/// endure+stall intra-mon residual duration tie (ONE shuffle at ANY speed) — a wrong
/// tie model desyncs every boundary seed here.
#[test]
fn endure_survives_at_one_hp_and_rides_the_shared_stall_ladder() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|endure,splash|Serious|252,,,,,|N||||";
    let p2 = "Tauros|||NoAbility|doubleedge,splash|Serious|,252,,,,252|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[0].hp, 256, "dec0: DE landed (no clamp needed at full HP)");
    assert_eq!(out.decisions[0].active[0].protect_counter, 2, "the SHARED stall counter");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "22534,42410,55299,35327");
    assert_eq!(out.decisions[1].active[0].hp, 15);
    assert_eq!(out.decisions[1].active[0].protect_counter, 4);
    assert_eq!(seed_str(&out.decisions[1].seed_after), "60443,61849,18300,733");
    assert_eq!(
        out.decisions[2].active[0].hp, 1,
        "dec2: the endure onDamage CLAMP — the lethal Double-Edge leaves EXACTLY 1 HP"
    );
    assert_eq!(out.decisions[2].active[0].protect_counter, 8);
    assert_eq!(seed_str(&out.decisions[2].seed_after), "63651,14230,62171,48683");
    assert!(out.decisions[3].active[0].fainted, "dec3: the stall roll FAILED → the KO");
    assert_eq!(seed_str(&out.decisions[3].seed_after), "35046,57977,35930,51983");
    assert_eq!(out.winner, Some(1));
}

/// MC87: ENDURE guards MOVE damage ONLY — the burned endurer survives the (fixed-
/// damage) Seismic Toss at exactly 1 HP, then the SAME turn's burn residual kills it
/// (residual damage is not a Move effect → no clamp). One decision, game over.
#[test]
fn endure_clamps_fixed_damage_but_not_the_burn_residual() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|endure,splash|Serious|252,,,,,|N||||";
    let p2 = "Blissey|||NoAbility|seismictoss,splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[0].pokemon[0].status = Some(Status::Burn);
    st.sides[0].pokemon[0].hp = 80; // within Seismic Toss KO range (the clamp must fire)
    let out = st.run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert!(out.decisions[0].active[0].fainted, "the burn residual killed the 1-HP endurer");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "60880,31090,7619,34922");
    assert_eq!(out.winner, Some(1));
}

/// MC88: PERISH SONG counters tick 3 → 2 → 1 → the perish0 FAINT at the order-12
/// residual (LAST in the ladder), the CAST is draw-free, and a SWITCH-OUT at perish1
/// CLEARS the leaver's counter (the pivoted-out Celebi survives; the Snorlax that
/// stayed faints on schedule → a forced replacement).
#[test]
fn perish_song_ticks_to_the_faint_and_a_pivot_clears_the_counter() {
    let d = dex();
    let p1 = "Celebi|||NoAbility|perishsong,splash|Serious|252,,,,,|N||||]Blissey|||NoAbility|seismictoss,splash|Serious|252,,,,,|N||||";
    let p2 = "Snorlax|||NoAbility|splash,bodyslam|Serious|252,,,,,|N||||]Skarmory|||NoAbility|drillpeck,splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // the cast (draw-free)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)), // Celebi pivots at perish1
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),   // Snorlax faints at 0
            ScriptDecision::one(1, Choice::Switch(1)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
        ],
        &d,
    );
    assert_eq!(out.decisions[0].perish, [3, 3], "the cast turn's boundary: BOTH actives at perish3");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "61255,39458,1834,64539",
        "the cast is DRAW-FREE (QC only)"
    );
    assert_eq!(out.decisions[1].perish, [2, 2]);
    assert_eq!(seed_str(&out.decisions[1].seed_after), "60880,31090,7619,34922");
    assert_eq!(
        out.decisions[2].perish,
        [0, 1],
        "the pivoted-in Blissey has NO counter (the switch cleared Celebi's); Snorlax ticked to 1"
    );
    assert_eq!(seed_str(&out.decisions[2].seed_after), "10897,43434,54578,10901");
    assert!(out.decisions[3].active[1].fainted, "Snorlax fainted at perish0");
    assert_eq!(seed_str(&out.decisions[3].seed_after), "37635,3740,64462,10380");
    assert_eq!(out.decisions[4].active_species[1], "skarmory", "the forced replacement");
    assert_eq!(seed_str(&out.decisions[5].seed_after), "37112,13693,28533,21721");
}

/// MC89: the PERISH MIRROR at an EQUAL cached speed — the two order-12 perish handlers
/// tie at EVERY residual (ONE `random(0,2)` shuffle per residual, the P5 draw model),
/// and the simultaneous 1→0 tick is a same-residual DOUBLE faint: both LAST mons → the
/// gen-3 TIE. A wrong pair-tie model desyncs every boundary seed.
#[test]
fn perish_mirror_ties_each_residual_and_the_mutual_perish_out_is_a_tie() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|perishsong,splash|Serious|252,,,,,|N||||";
    let p2 = "Snorlax|||NoAbility|splash,bodyslam|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "37635,3740,64462,10380"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    assert_eq!(out.decisions[0].perish, [3, 3]);
    assert_eq!(seed_str(&out.decisions[0].seed_after), "12061,48772,57767,1268");
    assert_eq!(out.decisions[1].perish, [2, 2]);
    assert_eq!(seed_str(&out.decisions[1].seed_after), "4112,59617,252,56412");
    assert_eq!(out.decisions[2].perish, [1, 1]);
    assert_eq!(seed_str(&out.decisions[2].seed_after), "609,418,60999,20164");
    assert!(out.decisions[3].active[0].fainted && out.decisions[3].active[1].fainted);
    assert_eq!(seed_str(&out.decisions[3].seed_after), "47292,19098,34708,5386");
    assert!(out.ended);
    assert_eq!(out.winner, None, "the mutual perish-out of both LAST mons is the TIE");
}

/// MC90: MEAN LOOK firm-traps the grounded GHOST (no `trapped` type-immunity in
/// Showdown-gen3): the trapped Gengar's voluntary switch is REJECTED draw-free (the
/// decision is SKIPPED, the boundary stays open — the T1 pattern); the moment the
/// TRAPPER pivots out the link ends and the next switch is ACCEPTED.
#[test]
fn mean_look_firm_traps_the_ghost_until_the_trapper_leaves() {
    let d = dex();
    let p1 = "Umbreon|||NoAbility|meanlook,seismictoss|Serious|252,,,,,|N||||]Blissey|||NoAbility|seismictoss,splash|Serious|252,,,,,|N||||";
    let p2 = "Gengar|||Levitate|nightshade,splash|Serious|252,,,,,|N||||]Misdreavus|||NoAbility|nightshade,splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // mean look lands
            ScriptDecision::one(1, Choice::Switch(1)),              // REJECTED (trapped) — skipped
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // the re-choice commits
            ScriptDecision::both(Choice::Switch(1), Choice::Move(1)), // the TRAPPER pivots → freed
            ScriptDecision::both(Choice::Move(0), Choice::Switch(1)), // the switch is now ACCEPTED
        ],
        &d,
    );
    assert_eq!(
        out.decisions.len(),
        4,
        "the rejected trapped-switch decision is SKIPPED (no boundary recorded)"
    );
    assert_eq!(out.decisions[0].trapped, [false, true], "the Ghost IS trapped (no type immunity)");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "61255,39458,1834,64539");
    assert_eq!(out.decisions[1].trapped, [false, true], "still trapped after the draw-free reject");
    assert_eq!(out.decisions[1].active_species[1], "gengar", "Gengar never left");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "10897,43434,54578,10901");
    assert_eq!(out.decisions[2].trapped, [false, false], "the trapper LEFT → the link ended");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "37635,3740,64462,10380");
    assert_eq!(out.decisions[3].active_species[1], "misdreavus", "the freed switch was ACCEPTED");
    assert_eq!(seed_str(&out.decisions[3].seed_after), "7184,5868,30814,34654");
}

/// MC91: SPIDER WEB + BATON PASS — the trapped Celebi legally Baton-Passes (selfSwitch
/// bypasses the trap gate) and the ENTRANT INHERITS the firm trap (`trapped` noCopy
/// FALSE — the resolved gen3 fact): the inheriting Snorlax's switch is REJECTED. The
/// TRAPPER'S FAINT then frees it (the link dies with the corpse) — the next switch is
/// ACCEPTED.
#[test]
fn spider_web_survives_a_baton_pass_and_dies_with_the_trapper() {
    let d = dex();
    let p1 = "Ariados|||NoAbility|spiderweb,splash|Serious|252,,,,,|N||||]Skarmory|||NoAbility|drillpeck,splash|Serious|252,,,,,|N||||";
    let p2 = "Celebi|||NoAbility|batonpass,splash|Serious|252,,,,,|N||||]Snorlax|||NoAbility|bodyslam,splash|Serious|252,252,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[0].pokemon[0].hp = 230; // 2 Body Slams KO Ariados
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // spider web traps Celebi
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // the trapped Celebi Baton-Passes (LEGAL)
            ScriptDecision::one(1, Choice::Switch(1)),              // the BP replacement target
            ScriptDecision::one(1, Choice::Switch(1)),              // REJECTED — the ENTRANT inherited the trap
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // the re-choice commits
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Body Slam KOs the trapper
            ScriptDecision::one(0, Choice::Switch(1)),
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // the freed switch is ACCEPTED
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 7, "ONE rejected decision was skipped");
    assert_eq!(out.decisions[0].trapped, [false, true], "Celebi trapped");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "61255,39458,1834,64539");
    // dec1 = the BP pause; dec2 = the replacement committed — the ENTRANT is trapped.
    assert_eq!(out.decisions[2].active_species[1], "snorlax");
    assert_eq!(out.decisions[2].trapped, [false, true], "the BP ENTRANT INHERITED the firm trap");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "60880,31090,7619,34922");
    // dec3 = the splash/bodyslam turn after the draw-free reject.
    assert_eq!(out.decisions[3].trapped, [false, true]);
    assert_eq!(seed_str(&out.decisions[3].seed_after), "37112,13693,28533,21721");
    // dec4 = the trapper's KO — the link dies with the corpse.
    assert!(out.decisions[4].active[0].fainted, "Ariados fainted");
    assert_eq!(out.decisions[4].trapped, [false, false], "the trapper's faint FREED the target");
    assert_eq!(seed_str(&out.decisions[4].seed_after), "60443,61849,18300,733");
    // dec5 = p1's forced replacement; dec6 = the freed switch accepted (Celebi back).
    assert_eq!(out.decisions[6].active_species[1], "celebi");
    assert_eq!(seed_str(&out.decisions[6].seed_after), "54523,22811,31582,9991");
}

/// MC92: the BELLY DRUM hp gate is the FLOAT `hp <= maxhp/2` — integer-exact as
/// `2*hp <= maxhp`: at Snorlax's 524 maxhp, hp=262 FAILS ([still]+fail, atk 0, HP
/// untouched, draw-free) while hp=263 SUCCEEDS (pays 262 → 1 HP, atk SET to +6), and
/// the immediate re-drum fails (atk >= 6). Both boundary seeds are the draw-free QC-only
/// trajectory.
#[test]
fn belly_drum_hp_boundary_and_the_atk_cap_fail() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|bellydrum,return|Serious|252,252,,,,|N||||";
    let p2 = "Skarmory|||NoAbility|splash,seismictoss|Serious|252,,,,,|N||||";
    // (a) hp == 262 → FAIL.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[0].pokemon[0].hp = 262;
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].hp, 262, "hp UNTOUCHED (the drum failed)");
    assert_eq!(out.decisions[0].active[0].boosts[0], 0, "no boost");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "61255,39458,1834,64539");

    // (b) hp == 263 → SUCCESS (1 HP, atk +6); the re-drum FAILS at atk >= 6.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[0].pokemon[0].hp = 263;
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[0].hp, 1, "paid floor(524/2)=262, leaving 1");
    assert_eq!(out.decisions[0].active[0].boosts[0], 6, "atk SET to +6");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "61255,39458,1834,64539");
    assert_eq!(out.decisions[1].active[0].hp, 1, "the re-drum FAILED (atk >= 6): no second cost");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "60880,31090,7619,34922");
}

/// MC93: CHARGE ×2s the NEXT ELECTRIC move's BP only (charged Thunderbolt 215 vs the
/// uncharged control 105 at BYTE-IDENTICAL draw counts — the ×2 is a BP-chain fold,
/// never a draw), and the volatile is CONSUMED by the user's next move attempt OF ANY
/// KIND (the Surf consumed it with NO boost — the following Thunderbolt is back at ×1).
#[test]
fn charge_doubles_the_next_electric_move_and_is_consumed_by_any_move() {
    let d = dex();
    let p1 = "Lanturn|||NoAbility|charge,thunderbolt,surf,splash|Serious|252,,,252,,|N||||";
    let p2 = "Snorlax|||NoAbility|splash,bodyslam|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // charge
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // CHARGED tbolt (×2)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // uncharged control
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // charge
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // Surf CONSUMES it (no boost)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // tbolt back at ×1 → the KO
        ],
        &d,
    );
    assert_eq!(seed_str(&out.decisions[0].seed_after), "61255,39458,1834,64539", "the charge cast is draw-free");
    assert_eq!(out.decisions[1].active[1].hp, 309, "the CHARGED Thunderbolt dealt 215 (524→309)");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "7184,5868,30814,34654");
    assert_eq!(out.decisions[2].active[1].hp, 204, "the uncharged control dealt 105 (309→204)");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "60443,61849,18300,733");
    assert_eq!(out.decisions[4].active[1].hp, 100, "the Surf consumed the charge with NO ×2");
    assert_eq!(seed_str(&out.decisions[4].seed_after), "59155,64853,46482,24392");
    assert_eq!(out.decisions[5].active[1].hp, 0, "the post-consumption tbolt is back at ×1");
    assert_eq!(seed_str(&out.decisions[5].seed_after), "4112,59617,252,56412");
    assert_eq!(out.winner, Some(0));
}

/// MC94: MEMENTO — the LANDED turn is ZERO draws TOTAL (never-miss in the resolved
/// gen3, the user's self-faint CANCELS the foe's queued Body Slam via gen3
/// faint-cancels-all, and the faint pause skips the Quick Claw): the post-turn seed is
/// the INIT seed. The foe drops −2 Atk / −2 SpA; the user faints → a forced
/// replacement. A PROTECTED memento is BLOCKED and the user does NOT faint.
#[test]
fn memento_lands_zero_draw_with_the_foe_move_cancelled_and_protect_blocks_it() {
    let d = dex();
    let p1 = "Dugtrio|||NoAbility|memento,splash|Serious|,,,,,252|N||||]Blissey|||NoAbility|seismictoss,splash|Serious|252,,,,,|N||||";
    let p2 = "Snorlax|||NoAbility|bodyslam,protect|Serious|252,252,,,,|N||||";
    // (a) the LANDED memento.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // memento; Body Slam CANCELLED
            ScriptDecision::one(0, Choice::Switch(1)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );
    assert!(out.decisions[0].active[0].fainted, "the user fainted (selfdestruct ifHit)");
    assert_eq!(out.decisions[0].active[1].boosts[0], -2, "foe atk −2");
    assert_eq!(out.decisions[0].active[1].boosts[2], -2, "foe spa −2");
    assert_eq!(out.decisions[0].active[1].hp, 524, "the foe's queued Body Slam was CANCELLED");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "44317,42357,9927,48760",
        "the landed-memento turn consumed ZERO draws (never-miss + cancel + no QC)"
    );
    assert_eq!(out.decisions[1].active_species[0], "blissey");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "37112,13693,28533,21721");

    // (b) memento INTO A PROTECT: blocked, the user does NOT faint.
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // protect blocks the memento
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    assert!(!out.decisions[0].active[0].fainted, "BLOCKED (ifHit) — the user does NOT faint");
    assert_eq!(out.decisions[0].active[0].hp, 211);
    assert_eq!(out.decisions[0].active[1].boosts[0], 0, "no drops through the Protect");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "60880,31090,7619,34922");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "7184,5868,30814,34654");
}

/// MC95: MIMIC slot semantics — the copy OVERWRITES the Mimic slot with the target's
/// lastMove at `pp = min(5, base)` / `maxpp = calculatePP(copied, 3)` (Psychic → 5/16);
/// the copied slot's PP decrements INDEPENDENTLY (5→4 on use); the slot REVERTS on
/// switch-out with Mimic's OWN remaining PP (15/16); a re-mimic pointed at an
/// already-known lastMove (Splash) FAILS. All draw-free. Plus the no-lastMove fail.
#[test]
fn mimic_overlays_the_slot_and_reverts_on_switch_out() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|mimic,splash|Serious|252,,,,,|N||||]Blissey|||NoAbility|seismictoss,splash|Serious|252,,,,,|N||||";
    let p2 = "Alakazam|||NoAbility|psychic,splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Psychic first → Mimic COPIES it
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // the COPIED Psychic runs (5→4)
            ScriptDecision::both(Choice::Switch(1), Choice::Move(1)), // pivot → the overlay REVERTS
            ScriptDecision::both(Choice::Switch(1), Choice::Move(1)), // back in (slot = Mimic 15/16)
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // mimic at a SPLASH lastMove → already-known FAIL
        ],
        &d,
    );
    assert_eq!(
        out.decisions[0].active[0].move_pp,
        [5, 64, -1, -1],
        "the Mimic slot now holds the copied Psychic at pp 5"
    );
    assert_eq!(seed_str(&out.decisions[0].seed_after), "22534,42410,55299,35327");
    assert_eq!(out.decisions[1].active[0].move_pp, [4, 64, -1, -1], "the copied slot decrements independently");
    assert_eq!(out.decisions[1].active[1].hp, 284, "the copied Psychic DEALT damage (314→284)");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "17940,16623,13080,40722");
    assert_eq!(
        out.decisions[3].active[0].move_pp,
        [15, 64, -1, -1],
        "back in: the slot REVERTED to Mimic with Mimic's own remaining PP (15/16)"
    );
    assert_eq!(seed_str(&out.decisions[3].seed_after), "12061,48772,57767,1268");
    assert_eq!(
        out.decisions[4].active[0].move_pp,
        [14, 64, -1, -1],
        "the re-mimic FAILED (already-known Splash): Mimic's own PP paid, no overlay"
    );
    assert_eq!(seed_str(&out.decisions[4].seed_after), "54523,22811,31582,9991");

    // The NO-LASTMOVE fail (the faster mimicker moves before the foe ever moved).
    let p1b = "Jolteon|||NoAbility|mimic,thunderbolt|Serious|252,,,,,252|N||||";
    let mut battle = Battle::start_with_switchins(
        &opts_cg(p1b, B6_EN_SNORLAX, "44317,42357,9927,48760"),
        &d,
    )
    .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].move_pp, [15, 24, -1, -1], "no overlay (the fail)");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "61255,39458,1834,64539", "draw-free");
}

/// MC96: PAIN SPLIT floor-averages the two actives' HP with each side clamped at its
/// OWN maxhp — Gengar 41 + Blissey 714 → avg 377: Blissey takes the FULL loss to 377
/// while Gengar caps at its 261 maxhp (NOT conservative). A SUBSTITUTE blocks it
/// ([still]+fail). All draw-free.
#[test]
fn pain_split_averages_with_the_maxhp_clamp_and_a_sub_blocks_it() {
    let d = dex();
    let p1 = "Gengar|||Levitate|painsplit,splash|Serious||N||||";
    let p2 = "Blissey|||NoAbility|splash,substitute|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[0].pokemon[0].hp = 41;
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // the clamped split
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // Blissey subs
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // BLOCKED by the sub
        ],
        &d,
    );
    assert_eq!(out.decisions[0].active[0].hp, 261, "Gengar CLAMPED at its own maxhp");
    assert_eq!(out.decisions[0].active[1].hp, 377, "Blissey took the FULL loss to the raw average");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "61255,39458,1834,64539");
    assert_eq!(out.decisions[2].active[1].hp, 199, "the subbed split did NOTHING");
    assert_eq!(out.decisions[2].active[0].hp, 261);
    assert_eq!(seed_str(&out.decisions[2].seed_after), "10897,43434,54578,10901");
}

/// MC97: PSYCH UP copies ALL the target's boost stages VERBATIM — including the ZEROS,
/// so the user's own prior stages are fully OVERWRITTEN: the twice-Cursed Snorlax
/// (+2 atk/+2 def/−2 spe) becomes exactly {spa+2, spd+2} after psyching up the
/// twice-Calm-Minded Suicune. Draw-free.
#[test]
fn psych_up_copies_all_stages_verbatim_overwriting_the_users_own() {
    let d = dex();
    let p1 = "Snorlax|||NoAbility|psychup,curse|Serious|252,252,,,,|N||||";
    let p2 = "Suicune|||NoAbility|calmmind,splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
        ],
        &d,
    );
    assert_eq!(out.decisions[1].active[0].boosts[0..5], [2, 2, 0, 0, -2], "the cursed stages");
    assert_eq!(
        out.decisions[2].active[0].boosts[0..5],
        [0, 0, 2, 2, 0],
        "psych up OVERWROTE every stage with the target's (zeros included)"
    );
    assert_eq!(seed_str(&out.decisions[2].seed_after), "22534,42410,55299,35327");
}

/// MC98: CHARGE does NOT survive a Baton Pass in practice — the BP itself is the
/// user's next move attempt, so `charge.onAfterMove` CONSUMES the volatile BEFORE the
/// switch resolves (probed: the sim emits `-end|Charge|[silent]` on the BP turn and
/// the entrant's Thunderbolt deals the ×1 control damage, 524→381 in BOTH arms).
/// WRONG (charge surviving to the entrant): the tbolt deals ~×2 → the HP desyncs.
#[test]
fn charge_is_consumed_by_the_baton_pass_move_itself() {
    let d = dex();
    let p1 = "Lanturn|||NoAbility|charge,batonpass,splash|Serious|252,,,,,|N||||]Jolteon|||NoAbility|thunderbolt,splash|Serious|252,,,252,,|N||||";
    let p2 = "Snorlax|||NoAbility|splash,bodyslam|Serious|252,,,,,|N||||";
    for (first_move, label) in [(Choice::Move(0), "charge"), (Choice::Move(2), "splash control")] {
        let mut battle =
            Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d)
                .expect("start");
        let st = battle.state_mut().expect("state");
        let out = st.run_full_battle(
            &[
                ScriptDecision::both(first_move, Choice::Move(0)),
                ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Baton Pass (consumes the charge)
                ScriptDecision::one(0, Choice::Switch(1)),
                ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // the entrant's tbolt
            ],
            &d,
        );
        assert_eq!(
            out.decisions[3].active[1].hp,
            381,
            "[{label}] the entrant's Thunderbolt is ×1 (the BP consumed any charge)"
        );
        assert_eq!(seed_str(&out.decisions[3].seed_after), "37112,13693,28533,21721", "[{label}]");
    }
}

/// MC99: a CONTACT **fixed-damage** hit fires the DEFENDER's contact-proc
/// `onDamagingHit` — Seismic Toss (`flags.contact`) into an EFFECT SPORE Breloom rolls
/// `randomChance(1,10)` after the damage apply (+ the `sample(3)` on a pass, landing
/// psn on the attacker at dec2). THE e2e_7 FIX (`gen3_move_coverage_batch6_v1` regen):
/// a LATENT batch-5-era gap — the fixed-damage family was only e2e-admitted in batch 5
/// and no ST-into-a-contact-proc-holder board was sampled until the batch-6 corpus
/// reshuffle; the port's `run_fixed_damage_move` never called `apply_contact_proc`, so
/// the sim drew `random(10)` where the port drew nothing (a seed desync at e2e_7
/// dec37). WRONG (no proc): every boundary seed here shifts and Blissey never gets
/// poisoned. Ground truth: harness/probe_batch6_regression_rng.js (MC99).
#[test]
fn fixed_damage_contact_hit_fires_the_contact_proc() {
    let d = dex();
    let p1 = "Breloom|||EffectSpore|splash,machpunch|Serious|252,,,,,|N||||";
    let p2 = "Blissey|||NoAbility|seismictoss,splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "10897,43434,54578,10901",
        "dec0: ST acc + the Effect Spore randomChance(1,10) (a FAIL roll) + QC"
    );
    assert_eq!(seed_str(&out.decisions[1].seed_after), "7184,5868,30814,34654");
    assert!(
        matches!(out.decisions[2].active[1].status, Some(Status::Poison)),
        "dec2: the proc PASSED + sampled psn onto the ATTACKER (Blissey)"
    );
    assert_eq!(out.decisions[2].active[1].hp, 625, "Blissey took the psn chip (714 − 89)");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "17940,16623,13080,40722");
}

// ============================================================================
// MC100-MC104 — SNATCH (`gen3_snatch_v1`): the LAST unmodeled gen-3 status move,
// which closes 722/722. A Dark, category-Status, priority-+4, never-miss target-self
// move that sets the `snatch` singleturn volatile; while up it STEALS the next foe
// self-targeted `flags.snatch` status move (the snatcher executes it, the foe's move
// does nothing). Probe-settled bit-for-bit (harness/probe_snatch.js); ground-truth
// seeds/state from harness/probe_snatch_regression_rng.js.
// ============================================================================

/// MC100: a FAST snatcher (Jolteon 130 > Skarmory 70) STEALS Swords Dance — the SNATCHER
/// gets +2 Atk, the foe is unboosted, the foe's SD PP drops (48→47), the snatcher spends
/// ONLY its Snatch PP (16→15). SNATCH is draw-free (the only turn draw is the endTurn Quick
/// Claw). WRONG (no steal / a panic): Jolteon unboosted; WRONG (a drawing steal): the seed
/// desyncs.
#[test]
fn snatch_fast_steals_swords_dance() {
    let d = dex();
    let jolteon = "Jolteon|||NoAbility|snatch,splash|Serious||N||||";
    let skarmory = "Skarmory|||NoAbility|swordsdance,splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(jolteon, skarmory, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].boosts[0], 2, "the SNATCHER (Jolteon) got the stolen +2 Atk");
    assert_eq!(out.decisions[0].active[1].boosts[0], 0, "the foe (Skarmory) is UNBOOSTED — its SD was stolen");
    assert_eq!(out.decisions[0].active[0].move_pp[0], 15, "the snatcher spent ONLY its Snatch PP (16→15)");
    assert_eq!(out.decisions[0].active[1].move_pp[0], 47, "the VICTIM spent the stolen SD's PP (48→47)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "61255,39458,1834,64539",
        "SNATCH is draw-free — the only turn draw is the endTurn Quick Claw"
    );
}

/// MC101: a SLOW snatcher (Snorlax 30 < Skarmory 70) STEALS Swords Dance ANYWAY — priority
/// +4 guarantees the `snatch` volatile is up before the foe's move, so the post-turn seed is
/// IDENTICAL to MC100's (the +4 interception proof). WRONG (a naive "reactive after the foe"
/// speed-race model): the draw order differs → the seed diverges from MC100.
#[test]
fn snatch_slow_snatcher_still_steals_priority_plus_four() {
    let d = dex();
    let snorlax = "Snorlax|||NoAbility|snatch,splash|Serious|252,,,,,|N||||";
    let skarmory = "Skarmory|||NoAbility|swordsdance,splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax, skarmory, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[0].boosts[0], 2, "the SLOW snatcher (Snorlax) STILL got the stolen +2 Atk");
    assert_eq!(out.decisions[0].active[1].boosts[0], 0, "Skarmory unboosted");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "61255,39458,1834,64539",
        "IDENTICAL to MC100 — priority +4 means the snatch fires regardless of speed"
    );
}

/// MC102: STEAL A REST (the DRAW-COUNT teeth) — the snatcher steals Snorlax's Rest → the
/// SNATCHER goes to sleep + FULL-heals (100→394, its own state; the foe stays awake). The
/// stolen Rest's sleep `random(2,6)` fires in the snatcher's context (2 draws vs 1). WRONG
/// (a draw-free steal): the seed desyncs by exactly that `random(2,6)`.
#[test]
fn snatch_steals_rest_snatcher_sleeps_and_draws_the_sleep_roll() {
    let d = dex();
    let umbreon = "Umbreon|||NoAbility|snatch,splash|Serious|252,,,,,|N||||";
    let snorlax = "Snorlax|||NoAbility|rest,splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(umbreon, snorlax, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[0].pokemon[0].hp = 100; // injure the snatcher so the stolen Rest's full heal is observable
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert!(
        matches!(out.decisions[0].active[0].status, Some(Status::Sleep(_))),
        "the SNATCHER (Umbreon) is asleep from the stolen Rest"
    );
    assert_eq!(out.decisions[0].active[0].hp, 394, "the stolen Rest FULL-healed the SNATCHER (100→394)");
    assert!(
        out.decisions[0].active[1].status.is_none(),
        "the foe (Snorlax) is NOT asleep — the Rest was stolen"
    );
    assert_eq!(out.decisions[0].active[1].move_pp[0], 15, "the VICTIM spent the stolen Rest's PP (16→15)");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "60880,31090,7619,34922",
        "the stolen Rest draws its sleep random(2,6) in the snatcher's context (+ the endTurn QC)"
    );
}

/// MC103: THUNDER WAVE is NOT snatchable (it carries no `flags.snatch`) → it PASSES THROUGH:
/// the snatcher is paralyzed normally, NO steal, NO `-activate`. WRONG (a wrongly-stolen
/// TWave): the snatcher would be UN-paralyzed and the seed would differ.
#[test]
fn snatch_does_not_steal_thunder_wave() {
    let d = dex();
    let umbreon = "Umbreon|||NoAbility|snatch,splash|Serious|252,,,,,|N||||";
    let jolteon = "Jolteon|||NoAbility|thunderwave,splash|Serious||N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(umbreon, jolteon, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert!(
        matches!(out.decisions[0].active[0].status, Some(Status::Paralysis)),
        "the snatcher is PARALYZED — Thunder Wave passed through (not snatchable)"
    );
    assert_eq!(out.decisions[0].active[0].boosts[0], 0, "no self-boost — nothing was stolen");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "60880,31090,7619,34922",
        "TWave draws its accuracy roll (+ the endTurn QC); no steal machinery"
    );
}

/// MC104: the SNATCH MIRROR residual-duration tie (the CRUX). Two EQUAL-speed Umbreon both
/// cast Snatch → both `snatch` `duration:1` volatiles register the NO_ORDER/subOrder-2
/// residual duration handler → they TIE → ONE extra `random(0,2)` tie-shuffle at the residual
/// (8 total draws). Neither steals (Snatch itself is not snatchable). WRONG (no residual
/// handler): the seed matches the both-Splash control (MC104b, 7 draws) → a silent desync on
/// every mirror turn. This pin asserts the mirror seed AND that it DIFFERS from the control.
#[test]
fn snatch_mirror_draws_the_residual_duration_tie_shuffle() {
    let d = dex();
    let umbreon = "Umbreon|||NoAbility|snatch,splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(umbreon, umbreon, "37635,3740,64462,10380"), &d)
            .expect("start");
    let out = battle
        .state_mut()
        .expect("state")
        .run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "12061,48772,57767,1268",
        "the snatch mirror draws the residual duration-handler tie-shuffle (8 draws)"
    );
    // The CONTROL: both Splash (slot swapped) → NO snatch volatile → NO residual tie-shuffle
    // (7 draws). Its post-seed MUST DIFFER — the mirror's extra draw is snatch-attributable.
    let umbreon_splash = "Umbreon|||NoAbility|splash,snatch|Serious|252,,,,,|N||||";
    let mut ctrl =
        Battle::start_with_switchins(&opts_cg(umbreon_splash, umbreon_splash, "37635,3740,64462,10380"), &d)
            .expect("start");
    let ctrl_out = ctrl
        .state_mut()
        .expect("state")
        .run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        seed_str(&ctrl_out.decisions[0].seed_after),
        "60443,61849,18300,733",
        "the both-Splash control draws NO residual tie-shuffle (7 draws)"
    );
    assert_ne!(
        out.decisions[0].seed_after, ctrl_out.decisions[0].seed_after,
        "the mirror's extra residual tie-shuffle draw MUST make its seed differ from the control"
    );
}

/// MC105 — SNATCH steal of a PRESSURE victim deducts an EXTRA Snatch PP
/// (`gen3_snatch_pressure_pp_v1`, the `bab_4_16` per-side/request byte-fuzz find). The
/// snatch interception fires `runEvent("DeductPP", source=VICTIM, snatchUser=SNATCHER,
/// Snatch)`; if the VICTIM (the stolen move's user) has **Pressure**, its `onDeductPP`
/// returns 1 → the SNATCHER's Snatch loses an EXTRA 1 PP. So a Snatch steal costs the
/// snatcher 1 (cast) + 1 (Pressure victim) = 2 Snatch PP (16→14); a NON-Pressure victim
/// costs 1 (16→15). DRAW-FREE (a PP deduct consumes no PRNG) → invisible to the omniscient
/// stream, surfaced only in the request `pp` field. WRONG (pre-fix): the port modeled the
/// DeductPP step as a pure no-op → 16→15 under Pressure too (`pp:15` where the sim shows
/// `pp:14`). STATE pin (the snatcher's Snatch PP) + a non-Pressure control.
#[test]
fn snatch_steal_of_a_pressure_victim_deducts_an_extra_snatch_pp() {
    let d = dex();
    let umbreon = "Umbreon|||NoAbility|snatch,splash|Serious|252,,,,,|N||||";
    // The VICTIM has Pressure + Rest (a snatchable self-target move).
    let suicune_pressure = "Suicune|||Pressure|rest,splash|Serious|252,,,,,|N||||";
    let mut battle = Battle::start_with_switchins(
        &opts_cg(umbreon, suicune_pressure, "44317,42357,9927,48760"),
        &d,
    )
    .expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[0].pokemon[0].hp = 100; // so the stolen Rest's heal is observable (steal confirmed)
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert!(
        matches!(out.decisions[0].active[0].status, Some(Status::Sleep(_))),
        "the SNATCHER stole the Rest (asleep) — the interception fired"
    );
    assert_eq!(
        out.decisions[0].active[0].move_pp[0], 14,
        "Snatch PP: 16 -1 (cast) -1 (Pressure victim DeductPP) = 14"
    );

    // CONTROL: a NON-Pressure victim (NaturalCure) → NO Pressure extra → Snatch PP 16→15.
    let suicune_plain = "Suicune|||NaturalCure|rest,splash|Serious|252,,,,,|N||||";
    let mut ctrl = Battle::start_with_switchins(
        &opts_cg(umbreon, suicune_plain, "44317,42357,9927,48760"),
        &d,
    )
    .expect("start");
    let cst = ctrl.state_mut().expect("state");
    cst.sides[0].pokemon[0].hp = 100;
    let cout = cst.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        cout.decisions[0].active[0].move_pp[0], 15,
        "non-Pressure victim → Snatch PP 16→15 (cast only)"
    );
}

/// MC106 — PURSUIT INTERRUPT into a switching PRESSURE mon deducts an EXTRA Pursuit PP
/// (`gen3_pursuit_pressure_pp_v1`, the `bab_7_1` per-side/request byte-fuzz find — the
/// Snatch-Pressure sibling). The interrupt runs `runMove('pursuit', source, {target:
/// switcher})`, which deducts the pursuer's Pursuit PP with the SWITCHER as the target; a
/// **Pressure** switcher's `onDeductPP` returns 1 → −2 total. WRONG (pre-fix, the port's
/// stated "the sim passes NO target so no Pressure extra" comment): the interrupt deducted a
/// flat 1 (`pp:30` where the sim shows `pp:29`). DRAW-FREE → surfaced only in the request
/// `pp`. STATE pin (Pursuit PP after intercepting a switching Pressure Moltres) + control.
#[test]
fn pursuit_interrupt_into_a_pressure_switcher_deducts_an_extra_pp() {
    let d = dex();
    let umbreon = "Umbreon|||NoAbility|pursuit,splash|Serious|252,,,,,|N||||";
    // p2 LEADS a Pressure Moltres + a bench Snorlax to switch to. Umbreon Pursuits the
    // LEAVING Moltres (the Pressure target).
    let p2_pressure =
        "Moltres|||Pressure|flamethrower,splash|Serious|252,,,,,|N||||]Snorlax|||NoAbility|splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(umbreon, p2_pressure, "44317,42357,9927,48760"), &d)
            .expect("start");
    // p2 SWITCHES (Moltres → Snorlax); p1 Pursuits → the interrupt strikes the leaving Moltres.
    let out = battle.state_mut().expect("state").run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Switch(1))],
        &d,
    );
    assert_eq!(
        out.decisions[0].active[0].move_pp[0], 30,
        "Pursuit PP: 32 -1 (strike) -1 (Pressure switcher DeductPP) = 30"
    );

    // CONTROL: a NON-Pressure switcher (Insomnia) → NO extra → Pursuit PP 32→31.
    let p2_plain =
        "Moltres|||Insomnia|flamethrower,splash|Serious|252,,,,,|N||||]Snorlax|||NoAbility|splash|Serious|252,,,,,|N||||";
    let mut ctrl =
        Battle::start_with_switchins(&opts_cg(umbreon, p2_plain, "44317,42357,9927,48760"), &d)
            .expect("start");
    let cout = ctrl.state_mut().expect("state").run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Switch(1))],
        &d,
    );
    assert_eq!(
        cout.decisions[0].active[0].move_pp[0], 31,
        "non-Pressure switcher → Pursuit PP 32→31 (strike only)"
    );
}

/// MC107 — SLEEP TALK fails DRAW-FREE for an ENCORED mon (`gen3_encore_sleeptalk_trylhit_v1`,
/// the R13 pool byte-fuzz find ab_15_15 dec 47). Sleep Talk's resolved onTryHit is
/// `return !volatiles['choicelock'] && !volatiles['encore']` — so a mon carrying the `encore`
/// volatile fails Sleep Talk at TryHit (`|move|…Sleep Talk||[still]` + `|-fail|`), drawing
/// NOTHING (no sample, no called move). The port's sleeptalk arm gated ONLY on `choicelock`
/// (batch 5 was written when Encore was still unmodeled — batch 6 later modeled it but the
/// gate was never updated), so an Encored-into-Sleep-Talk asleep mon SAMPLED + ran the picked
/// move (+4 draws) where the sim fails it draw-free — the ab_15_15 SEED desync.
/// Scenario: a fast Electrode [Encore, Spore] vs a slow Snorlax [Sleep Talk, Splash]. dec0:
/// Spore sleeps Snorlax. dec1: Snorlax (asleep) Sleep Talk → sample picks Splash (lastMove →
/// sleeptalk). dec2: Electrode (faster) Encores Snorlax THIS turn (locks Sleep Talk); Snorlax's
/// queued Sleep Talk is `cant slp` then FAILS onTryHit (encore present) — DRAW-FREE (only the
/// Encore accuracy + its random(3,7) duration + Quick Claw draw). WRONG (pre-fix): the port
/// sampled from {Splash} + ran it → +draws → a different seed. Ground truth
/// `harness/probe_r13_encore_sleeptalk_rng.js` (raw seed [7,11,13,17] → initSeed
/// 44317,42357,9927,48760).
#[test]
fn encored_mon_sleep_talk_fails_draw_free_via_ontryhit() {
    let d = dex();
    let electrode = "Electrode|||NoAbility|encore,spore,splash|Serious|,,,,,252|N||||";
    let snorlax = "Snorlax|||NoAbility|sleeptalk,splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(electrode, snorlax, "44317,42357,9927,48760"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // Spore ; Splash → Snorlax asleep
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // Splash ; Sleep Talk (sample→Splash)
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Encore ; Sleep Talk → onTryHit encore FAIL
        ],
        &d,
    );
    // dec1: the asleep Sleep Talk sampled Splash (lastMove → sleeptalk); the pre-Encore seed.
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "22534,42410,55299,35327",
        "dec1: Snorlax's asleep Sleep Talk samples + runs Splash (lastMove → sleeptalk)"
    );
    // dec2 (the PIN): Electrode Encores Snorlax → the encored Sleep Talk fails DRAW-FREE at
    // onTryHit — the turn draws ONLY [Encore accuracy, Encore random(3,7) duration, Quick Claw].
    // Pre-fix the port would sample {Splash} + run it (extra draws) → a different seed.
    assert_eq!(
        seed_str(&out.decisions[2].seed_after),
        "5621,5056,41416,14688",
        "dec2: the ENCORED Sleep Talk fails draw-free (onTryHit !encore) — no sample, no called move"
    );
    assert!(
        matches!(out.decisions[2].active[1].status, Some(Status::Sleep(_))),
        "Snorlax stays asleep (the Sleep Talk cant'd + failed onTryHit; the counter decremented)"
    );
    assert!(
        out.decisions[2].encore[1] > 0,
        "Snorlax carries the encore volatile (locked to Sleep Talk) at the fail boundary"
    );
}

// ============================================================================
// MC108–MC112 — the MOVE-COVERAGE BATCH 7 GENERIC MULTI-STRIKE family
//       (`gen3_move_coverage_batch7_v1`, `turn.rs::run_multihit`) + the cross-side
//       ModifyDamagePhase1 shuffle fix. The pinned (hp, seed) constants are the PORT's
//       `opts_cg` output — byte-validated == the real Showdown sim by
//       `harness/probe_batch7_isolated.js` (80 constructed multihit battles / 1839 strikes,
//       node-vs-rust `--protocol` clean) + the omniscient draw model in
//       `harness/probe_batch7_multihit.js`. (A sim `>start` probe can't reproduce these exact
//       seeds: its turn-0 construction window — the gender `sample` + Quick Claw — differs from
//       the port's `start_with_switchins`, the project-wide seed-convention deferral.) Each pin
//       is revert-verified: reverting `run_multihit` / the cross-side shuffle flips its (hp,seed).
// ============================================================================

/// MC108: DOUBLE KICK is a FIXED-2 multihit — NO count draw, 2*(crit+damage) + Quick Claw.
/// Hitmonlee Double Kicks a bulky Snorlax (survives 2 strikes → 524-204 = 320). WRONG (if
/// `run_multihit` mis-draws a count for a fixed move, or does not loop): the hp / seed desync.
/// STATE (Snorlax 320, hitcount 2) + SEED (no count sample, 2*(crit+damage) + Quick Claw).
#[test]
fn double_kick_is_a_fixed_two_strike_multihit_no_count_draw() {
    let d = dex();
    let p1 = "Hitmonlee|||noguard|doublekick,splash|Serious|,252,,,,|||||";
    let p2 = "Snorlax|||immunity|splash|Serious|252,,252,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[1].hp, 301, "Double Kick = 2 strikes → 524-223");
    assert!(!out.decisions[0].active[1].fainted, "Snorlax survived the 2-strike Double Kick");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "7184,5868,30814,34654",
        "fixed-2: NO count draw — accuracy + 2*(crit+damage) + Quick Claw"
    );
}

/// MC109: ICICLE SPEAR is a VARIABLE [2,5] multihit — it draws ONE `sample([2,2,2,3,3,3,4,5])`
/// count draw (gen<5) AFTER accuracy, BEFORE the per-strike loop. At this seed the sample = 5.
/// Cloyster Icicle Spears a bulky Blissey (survives 5 strikes → 714-52 = 662, hitcount 5).
/// WRONG (if the count sample is MISSING / mis-drawn, or the loop count is wrong): the hp / seed
/// desync. STATE (Blissey 662, 5 strikes) + SEED (accuracy + ONE sample + 5*(crit+damage) + QC).
#[test]
fn icicle_spear_variable_multihit_draws_one_count_sample() {
    let d = dex();
    let p1 = "Cloyster|||noguard|iciclespear,splash|Serious|,,,252,,|||||";
    let p2 = "Blissey|||naturalcure|splash|Serious|252,,,,252,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[1].hp, 660, "Icicle Spear = a variable count of strikes → 714-54");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "54523,22811,31582,9991",
        "[2,5] draws ONE sample([2,2,2,3,3,3,4,5]) = 5, then 5*(crit+damage) + Quick Claw"
    );
}

/// MC110: a multihit that KOs the target on the FIRST strike STOPS the loop (no 2nd strike, no
/// Quick Claw on the deciding faint). Diglett injected to 5 HP; Hitmonlee (faster) Double Kicks —
/// strike 1 KOs it, strike 2 is skipped (`targets.every(!hp)`), the battle ends. WRONG (if the
/// loop does not stop): a 2nd strike fires / a Quick Claw on the deciding faint → the seed desync.
/// STATE (Diglett fainted, p2 left 0) + SEED (1 strike, NO Quick Claw).
#[test]
fn multihit_ko_on_the_first_strike_stops_the_loop_no_quick_claw() {
    let d = dex();
    let p1 = "Hitmonlee|||noguard|doublekick,splash|Serious|,252,,,,|||||";
    let p2 = "Diglett|||sandveil|splash|Serious|,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    {
        let st = battle.state_mut().expect("state");
        let a = st.sides[1].active;
        st.sides[1].pokemon[a].hp = 5; // STATE-only: strike 1 KOs it
    }
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert!(out.decisions[0].active[1].fainted, "Diglett fainted on strike 1");
    assert_eq!(out.decisions[0].pokemon_left[1], 0, "Diglett was p2's last mon → p1 wins");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "10897,43434,54578,10901",
        "the loop stops at the faint — ONE strike, NO Quick Claw on the deciding faint"
    );
}

/// MC111: TWINEEDLE draws its 20% psn SECONDARY `random(100)` PER STRIKE (like every multihit
/// move's secondary — the sim's `spreadMoveHit` runs `secondaries()` per hit). At this seed the
/// psn lands → Snorlax poisoned; the 2nd strike's secondary draws but no-ops (already-statused).
/// WRONG (if the secondary is drawn ONCE after all strikes, or per-move): the seed desync (2
/// draws vs 1). STATE (Snorlax 393 + psn) + SEED (per-strike secondary random(100)).
#[test]
fn twineedle_draws_its_secondary_per_strike_and_poisons() {
    let d = dex();
    let p1 = "Beedrill|||swarm|twineedle,splash|Serious|,252,,,,|||||";
    let p2 = "Snorlax|||thickfat|splash|Serious|252,,252,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "7,11,13,17"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions[0].active[1].hp, 393, "Twineedle = 2 strikes → 524-131");
    assert_eq!(
        out.decisions[0].active[1].status,
        Some(Status::Poison),
        "the 20% psn secondary landed (per-strike random(100))"
    );
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "37112,13693,28533,21721",
        "the SECONDARY random(100) draws PER STRIKE (2 draws) — not once"
    );
}

/// MC112: the CROSS-SIDE ModifyDamagePhase1 SHUFFLE — when BOTH sides carry a screen, a multi-
/// strike hit draws ONE `random(0,2)` PER STRIKE (the two `onAnyModifyDamagePhase1` handlers tie).
/// dec0 both set Light Screen; dec1 Cloyster Icicle Spears the both-Light-Screen Snorlax. WRONG
/// (pre-fix: the shuffle was gated on the FOE having BOTH Reflect AND Light Screen, so a cross-
/// side pair — here BOTH sides' Light Screen — drew NOTHING): the port under-draws the per-strike
/// shuffle → the dec1 seed desyncs. STATE (Snorlax 501, LS-halved) + SEED (per-strike shuffle).
/// NOTE (`gen3_screen_residual_tie_shuffle_v1`): dec0's END-OF-TURN RESIDUAL now ALSO draws the
/// both-sides-Light-Screen tie-shuffle (a SEPARATE event from this per-strike one — see
/// `both_sides_light_screen_residual_draws_the_handler_sort_shuffle`), so the constants were
/// updated from the pre-residual-fix (509 / `17940,…`) to include that dec0 residual draw. The
/// per-strike ModifyDamagePhase1 shuffle this pin guards is unaffected (reverting it still fails).
#[test]
fn multihit_into_both_sides_light_screen_draws_the_modify_damage_phase1_shuffle_per_strike() {
    let d = dex();
    let p1 = "Cloyster|||noguard|iciclespear,lightscreen,splash|Serious|,,,252,,|||||";
    let p2 = "Snorlax|||immunity|lightscreen,splash|Serious|252,,252,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    // dec0: both set Light Screen (p1 slot 1, p2 slot 0). dec1: p1 Icicle Spear (slot 0), p2 Splash.
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
        ],
        &d,
    );
    assert_eq!(out.decisions[1].active[1].hp, 501, "Icicle Spear (LS-halved) → 524-23");
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "56830,34298,10811,30881",
        "BOTH sides' Light Screen = 2 tied ModifyDamagePhase1 handlers → ONE random(0,2) PER STRIKE"
    );
}

/// LS-RESIDUAL (`gen3_screen_residual_tie_shuffle_v1`): the BOTH-SIDES-SAME-SCREEN RESIDUAL
/// tie-shuffle — the residual-side sibling of MC17's per-hit ModifyDamagePhase1 shuffle (a
/// SEPARATE event; do not conflate). `fieldEvent('Residual')` gathers each side's Reflect /
/// Light Screen as an `onSideResidual` DURATION handler (`findSideEventHandlers(..., 'duration')`,
/// order 1/2, speed 0, subOrder 4) and speed-sorts them with every other residual handler — so
/// when BOTH sides carry the SAME screen the two tie (order 2↔2, speed 0↔0) and the tie-group
/// Fisher-Yates shuffle draws ONE `random(0,2)`. WRONG (pre-fix): the port DECREMENTED the screens
/// directly at the top of `run_residuals`, before the handler list was built, so the screens never
/// became sorted duration handlers and never tie-shuffled — a both-sides-same-screen residual drew
/// ONE FEWER call than the sim → downstream RNG desync (the ab_1_6 draw-#191 divergence). The pin
/// injects Light Screen on BOTH sides (the tie) vs ONE side (the control), runs a draw-minimal
/// splash/splash turn, and asserts the both-sides post-turn seed == the REAL-Showdown ground truth
/// AND that it DIFFERS from the one-side control (reverting the fix — screens not gathered as
/// handlers — makes the two seeds EQUAL). Ground truth: harness/probe_screen_residual_regression_rng.js
/// (sim `>start` [11,22,33,44] → post-construction initSeed 13127,45333,18295,15391; the sim's
/// both-vs-one draw is confirmed separately by harness/probe_screen_residual_shuffle.js).
#[test]
fn both_sides_light_screen_residual_draws_the_handler_sort_shuffle() {
    let d = dex();
    // Genderless-treated Blissey (55 spe) + Snorlax (30 spe) — distinct speeds (no action-order /
    // eachEvent tie) both SPLASH (draw-free move phase), so the ONLY turn draws are the residual
    // (the screen shuffle in the both-sides case) + the end-of-turn Quick Claw.
    let blissey = "Blissey|||naturalcure|splash|Serious|,,,,,|||||";
    let snorlax = "Snorlax|||immunity|splash|Serious|,,,,,|||||";

    // BOTH sides Light Screen (duration 5) → the residual ties the two Light Screen handlers.
    let mut both =
        Battle::start_with_switchins(&opts_cg(blissey, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    {
        let st = both.state_mut().unwrap();
        st.sides[0].light_screen = 5;
        st.sides[1].light_screen = 5;
    }
    let out_both = both
        .state_mut()
        .unwrap()
        .run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        seed_str(&out_both.decisions[0].seed_after),
        "57388,452,34593,29177",
        "BOTH sides' Light Screen ties at the residual → ONE random(0,2) shuffle → the real Showdown seed"
    );

    // ONE side Light Screen (control) — no tie, no residual shuffle → a DIFFERENT seed.
    let mut one =
        Battle::start_with_switchins(&opts_cg(blissey, snorlax, "13127,45333,18295,15391"), &d)
            .expect("start");
    {
        let st = one.state_mut().unwrap();
        st.sides[0].light_screen = 5;
    }
    let out_one = one
        .state_mut()
        .unwrap()
        .run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        seed_str(&out_one.decisions[0].seed_after),
        "18464,3966,47670,60926",
        "ONE side's Light Screen is alone in its residual order group → NO tie → NO shuffle draw"
    );
    assert_ne!(
        seed_str(&out_both.decisions[0].seed_after),
        seed_str(&out_one.decisions[0].seed_after),
        "the residual tie-shuffle draw MUST make the two seeds differ (the crux — reverting the fix, \
         so screens are not gathered as handlers, makes them equal)"
    );
}

// ============================================================================
// R3 / HP1 — HIDDEN POWER's BASE POWER is IV-DERIVED, not the flat 70 the data ships
//       (`gen3_iv_derived_hidden_power_bp_v1`). gen-3 computes HP's BP from the ATTACKER's IVs
//       (Dex.getHiddenPower, `⌊hpPowerX·40/63⌋+30`, range 30..=70), so a real gen3ou HP mon
//       whose IVs give BP != 70 (e.g. a -1 Atk IV → BP 68) must damage at its IV-true BP. WRONG
//       (pre-fix): the engine read the data's hard-coded 70 for every HP → it over-damaged any
//       non-BP-70 HP mon (the ~1.5%-of-pool byte-fuzz divergence). STATE pin (a BP-68 HP Ice
//       deals the sim's 53, not the BP-70 55) + SEED pin (the BP change is DRAW-NEUTRAL — a
//       precomputed state read — so the seed is UNCHANGED both ways). Ground truth from
//       probe_hidden_power_bp_regression_rng.js (HP-ICE-BP68).
// ============================================================================

/// R3: a BP-68 Hidden Power Ice (IVs [hp31,atk28,def30,spa31,spd31,spe31] → hpType Ice,
/// hpPower 68) from a Starmie into a bulky Blissey deals the SIM's IV-true-BP damage (53),
/// NOT the flat-BP-70 damage (55). WRONG (pre-fix): the data's hard-coded BP 70 over-damaged.
/// The BP override is a precomputed STATE read (draw-neutral), so the post-turn seed is
/// UNCHANGED — the pin's teeth are the STATE (the exact HP), verified against the real sim.
#[test]
fn hidden_power_bp_is_iv_derived_not_flat_seventy() {
    let d = dex();
    // p1 Starmie HP Ice (BP 68) into a p2 Blissey (neutral to Ice, huge HP → never faints, a
    // clear HP delta); Blissey Splashes (draw-free). Distinct speeds (Starmie 115 > Blissey 55,
    // no action-order tie). The IV field `,28,30,,,` = [hp31,atk28,def30,spa31,spd31,spe31].
    let starmie = "Starmie|||NoAbility|hiddenpowerice|Serious|,,,252,,252|N|,28,30,,,|||";
    let blissey = "Blissey|||NoAbility|splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(starmie, blissey, "4905,34237,46622,24710"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let blissey_maxhp = st.sides[1].pokemon[0].maxhp;
    assert_eq!(blissey_maxhp, 714, "Blissey max HP is 714 (sanity)");

    // Turn 1: p1 HP Ice (BP 68) ; p2 Splash.
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // STATE (the pin's teeth): the Blissey is at EXACTLY 661/714 — the sim's BP-68 damage (53).
    // WRONG (pre-fix, BP 70): the sim deals 55 → Blissey 659/714. So a flat-BP-70 engine trips
    // this by 2 HP (the ~2/68 ≈ 3% over-damage that cascades KO thresholds on real teams).
    assert_eq!(
        st.sides[1].pokemon[0].hp, 661,
        "a BP-68 Hidden Power Ice deals the SIM's IV-true-BP damage (53 → Blissey 661/714), \
         NOT the flat-BP-70 damage (55 → 659/714)"
    );

    // GROUND TRUTH (probe_hidden_power_bp_regression_rng.js, HP-ICE-BP68): the post-turn seed ==
    // the real Showdown seed. The IV-derived BP is a DRAW-NEUTRAL precomputed state read, so the
    // seed is IDENTICAL whether the engine uses BP 68 or BP 70 (only the damage magnitude moves).
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "41902,55468,4897,30138",
        "post-turn seed == the real Showdown seed (the IV-derived BP is draw-neutral)"
    );
}

/// R3 unit gate: the `hidden_power_bp` weight-order + boundary values, pinned as a
/// data-independent regression (a companion to the engine pin above). These pin the
/// SPE-before-SPA/SPD weight crux — a naive [hp,atk,def,spa,spd,spe]→1,2,4,8,16,32
/// mapping (weighting Spe 32 instead of 8) mis-computes any spread with an asymmetric
/// Spe/SpD IV. IV order is [hp,atk,def,spa,spd,spe].
#[test]
fn hidden_power_bp_weight_order_and_boundaries() {
    use pokesim::state::hidden_power_bp;
    // IV 31 in every stat → all six 2nd-LSBs set → hpPowerX 63 → ⌊63·40/63⌋+30 = 70.
    assert_eq!(hidden_power_bp(&[31, 31, 31, 31, 31, 31]), 70, "IV-31 spread → BP 70");
    // A -1 Atk bit (atk28, 2nd-LSB 0) drops weight 2 → hpPowerX 61 → ⌊61·40/63⌋+30 = 68.
    assert_eq!(hidden_power_bp(&[31, 28, 30, 31, 31, 31]), 68, "the HP-Ice BP-68 spread");
    // A -1 Spe bit (spe idx 5 = 0, weight 8) drops 8 → hpPowerX 55 → ⌊55·40/63⌋+30 = 64. If Spe
    // were mis-weighted 32 (the naive array-order bug), dropping it would give hpPowerX 31 → BP 49.
    assert_eq!(hidden_power_bp(&[31, 31, 31, 31, 31, 0]), 64, "SPE weighted 8, not 32 (the crux)");
    // A -1 SpD bit (spd idx 4 = 0, weight 32) drops 32 → hpPowerX 31 → ⌊31·40/63⌋+30 = 49.
    assert_eq!(hidden_power_bp(&[31, 31, 31, 31, 0, 31]), 49, "SPD weighted 32");
    // The floor: all 2nd-LSBs clear (IV 0/1) → hpPowerX 0 → BP 30.
    assert_eq!(hidden_power_bp(&[0, 0, 0, 0, 0, 0]), 30, "the 30..=70 floor");
}

// ============================================================================
// R12 / HP-BARE — a BARE-STORED Hidden Power executes as the correct TYPED damaging move
//       (`gen3_typed_hidden_power_ids_v1` round-12 pool-crash P0). A packed gen3ou team can
//       store the move SLOT as the BARE `hiddenpower` (num 237, data type **Normal**, BP **0**),
//       with the real HP type carried ONLY by the IVs (Showdown's Pokemon constructor resolves it
//       to the typed variant at construction). The bp-0 data row derives category **Status**, so
//       WRONG (pre-fix): `run_move`'s BP override fixed BP but left `category`==Status → the move
//       ROUTED into `run_status_move`'s fail-loud guard → a PANIC ("status move \"hiddenpower\" is
//       not modeled …" at turn.rs). This is PRODUCTION-REACHABLE: `sim_bridge` (--use-bridge=rust)
//       shares the SAME team.rs unpack + run_move, so it CRASHED on any real gen3ou Hidden Power
//       team (the byte-fuzz pool gate found it on battle ab_7_10: a Charizard with a bare
//       `HiddenPower`/hpType Dark). FIX (turn.rs, after the BP override): for the bare `hiddenpower`
//       id, resolve the RUNTIME type from the attacker's IVs (`hidden_power_type`) and RE-DERIVE
//       the category from the overridden BP + that type — so the bare HP executes as the correct
//       TYPED damaging move (right type, right BP, right phys/spec split), mirroring the variable-BP
//       block. STATE pin (the move DEALS Dark-type SUPER-EFFECTIVE damage — Dark 2× vs Psychic —
//       NOT Normal 1×, and NOT a panic) + SEED pin (the resolution is a deterministic IV read →
//       DRAW-NEUTRAL, so the post-turn seed == the sim's). Ground truth from the cloned R3 probe
//       (/tmp/probe_bare_hp3.js: Starmie bare HP Dark into Slowbro, post-construction seed
//       4905,34237,46622,24710) + the byte-fuzz pool gate (ab_7_10 flips panic → ok).
// ============================================================================

/// R12: a Starmie whose move SLOT stores the BARE `hiddenpower` (all-31 IVs → hpType Dark, BP 70)
/// USES it into a bulky Slowbro (Water/Psychic). The move must execute as the correct TYPED damaging
/// move — Dark is SUPER-EFFECTIVE (2×) vs Psychic — dealing the sim's exact damage (Slowbro 223/394),
/// NOT panicking (pre-fix) and NOT dealing Normal-type (1×, the data-237 type) damage. The type +
/// category resolution is a deterministic IV read → DRAW-NEUTRAL, so the post-turn seed == the sim's.
#[test]
fn bare_hidden_power_executes_as_the_typed_damaging_move() {
    let d = dex();
    // p1 Starmie (base SpA 100), move slot = BARE `hiddenpower`, empty IV field → all-31 → hpType
    // Dark, BP 70. p2 Slowbro (Water/Psychic, bulky) Splashes (draw-free). Distinct speeds
    // (Starmie 115 > Slowbro 30, no action-order tie). Seed = the sim's POST-CONSTRUCTION seed
    // (the port does not model the turn-0 construction window; probe read 4905,… after construction).
    let starmie = "Starmie|||NoAbility|hiddenpower|Serious|,,,252,,252|N||||";
    let slowbro = "Slowbro|||NoAbility|splash|Serious|252,,252,,,|M||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(starmie, slowbro, "4905,34237,46622,24710"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let slowbro_maxhp = st.sides[1].pokemon[0].maxhp;
    assert_eq!(slowbro_maxhp, 394, "Slowbro max HP is 394 (sanity)");

    // Turn 1: p1 bare Hidden Power (Dark) ; p2 Splash. Pre-fix this PANICS (routes to
    // run_status_move's fail-loud guard) — so reaching the asserts at all is the crash-fix teeth.
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");

    // STATE (the pin's teeth): the bare HP resolved to Dark (super-effective 2× vs Psychic) — the
    // sim deals 171 → Slowbro 223/394. A NORMAL-type (data-237) resolution would be neutral (1×,
    // ~half) → a much higher HP; the pre-fix panic never reaches here at all.
    assert_eq!(
        st.sides[1].pokemon[0].hp, 223,
        "a BARE Hidden Power (hpType Dark) deals Dark super-effective damage (171 → Slowbro 223/394), \
         NOT Normal-type neutral damage and NOT a fail-loud panic"
    );

    // GROUND TRUTH (probe: Starmie bare HP Dark into Slowbro, post-construction seed 4905,…): the
    // post-turn seed == the real Showdown seed. The type/category resolution is a deterministic IV
    // read (DRAW-NEUTRAL), so the seed is IDENTICAL — the fix adds no draw, only corrects routing.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "41902,55468,4897,30138",
        "post-turn seed == the real Showdown seed (the bare-HP type/category resolution is draw-neutral)"
    );
}

// ============================================================================
// R2 — the LEFTOVERS residual-tie -heal EMIT ORDER on a same-speed mirror with a pending
//       Wish (`gen3_leftovers_slotcond_gather_order_v1`). The residual handler-sort
//       (`speed_sort`) is a NON-STABLE selection sort whose swaps DISTURB the relative order
//       of the tied handlers, so the tie-group Fisher-Yates shuffle reads whatever pre-sort
//       order the swaps LEFT the tied pair in. Showdown gathers a side's slot conditions
//       (Wish order 7 / Future Move order 11) via `findSideEventHandlers(…, active)` — AFTER
//       that active's pokemon handlers — so Wish sits AFTER Leftovers in the pre-sort array.
//       WRONG (pre-fix): the port gathered Wish FIRST (a pre-loop at the array front), so the
//       selection-sort's Wish/weather swaps REVERSED the tied Leftovers pair vs the sim → the
//       two `-heal` lines emitted in the OPPOSITE order at the SAME shuffle value (the R2
//       byte-fuzz divergence). FIX: gather Wish/FutureMove per-active, after the item. It is
//       OBSERVATION-ONLY (draw-neutral — same handlers/keys/tie-count) → the seed is
//       UNCHANGED; only the emit permutation moves to match Showdown. Ground truth from
//       harness/probe_r2_wish_leftovers_regression_rng.js (seed [2,2,2,2]).
// ============================================================================

/// R2: a Jolteon mirror (both Leftovers, equal speed) under sandstorm with a PENDING p2 Wish
/// emits its Leftovers `-heal` lines in Showdown's exact order — `[p2, p1]` on turn 1 (no Wish
/// yet) then `[p1, p2]` on turn 2 (p2 Wish pending, order-7 handler in the pre-sort array).
/// WRONG (pre-fix): the port gathered Wish at the array front, so turn 2 emitted `[p2, p1]` —
/// the reversed pair at the SAME shuffle value. The post-turn seed is UNCHANGED both ways (the
/// fix is emission-only), so the pin's teeth are the `-heal` MARKER SEQUENCE.
#[test]
fn leftovers_heal_order_follows_the_slot_condition_gather() {
    let d = dex();
    let p1 = "Tyranitar||Leftovers|SandStream|splash|Serious||N||||]Jolteon||Leftovers|VoltAbsorb|splash|Serious|4,,,,,|N||||";
    let p2 = "Tyranitar||Leftovers|SandStream|splash|Serious||N||||]Jolteon||Leftovers|VoltAbsorb|wish,splash|Serious|4,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "10126,34169,19989,9144"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    // Turn 1: both switch Tyranitar → Jolteon (slot 2). Turn 2: p1 Splash, p2 Wish (leaves a
    // pending p2 Wish → the order-7 slot-condition handler at turn 2's residual).
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Switch(1), Choice::Switch(1)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );

    // The Leftovers `-heal` side-marker sequence == Showdown's EXACTLY (2 pairs: turn 1's
    // residual then turn 2's Wish-pending residual). Pre-fix, turn 2's pair was reversed.
    let heals: Vec<&str> = lines
        .iter()
        .filter(|l| l.0.contains("-heal") && l.0.contains("Leftovers"))
        .map(|l| if l.0.contains("p1a") { "p1" } else { "p2" })
        .collect();
    assert_eq!(
        heals,
        vec!["p2", "p1", "p1", "p2"],
        "the Leftovers -heal order matches Showdown (turn 2's Wish-pending pair is p1-then-p2, \
         NOT the pre-fix reversed p2-then-p1)"
    );

    // GROUND TRUTH (probe_r2_wish_leftovers_regression_rng.js, seed [2,2,2,2]): the post-turn
    // seed == the real Showdown seed — the -heal ORDER fix is DRAW-NEUTRAL (emission-only), so
    // the seed is IDENTICAL whether the pair emits p1-first or p2-first.
    assert_eq!(
        seed_str(&out.decisions.last().unwrap().seed_after),
        "28797,58885,11195,27642",
        "post-turn seed == the real Showdown seed (the emit-order fix is draw-neutral)"
    );
}

// ============================================================================
// T1F — FREEZE PERSISTENCE vs Hidden Power Fire / Weather Ball
//       (`gen3_omniscient_byte_fuzz_v1`, the deep-state byte-fuzz repro
//       rmroh04is_ab_4_18: a frozen Tyranitar hit by Metagross Hidden Power Fire).
//       gen3 `frz.onDamagingHit` (deps/pokemon-showdown/data/mods/gen3/conditions.ts:45-50)
//       thaws a frozen DEFENDER only when `this.dex.moves.get(move.id).type === 'Fire'` — the
//       BASE-dex move type — with the explicit "don't count Hidden Power or Weather Ball as
//       Fire-type" comment (`dex.moves.get('hiddenpower').type === 'Normal'`, `'weatherball'
//       === 'Normal'`). The port computed `is_fire` from the RESOLVED runtime type (Fire for
//       the typed-HP move nums 355-370), so a Hidden Power Fire hit WRONGLY thawed the frozen
//       mon. It surfaced as a kind=status "sim=Freeze / port=None" divergence (ab_replay
//       compares the ACTIVE mon's status per decision, so the lost freeze is caught only when
//       the frozen mon becomes active again). The fix excludes `hiddenpower*`/`weatherball`
//       from `is_fire` (turn.rs, mirroring the base-type semantics). DRAW-NEUTRAL (the thaw is
//       a pure status clear — no PRNG), so the STATE (still frozen) is the diagnostic; the seed
//       is a stability anchor. Revert the `is_fire` narrowing → HP Fire thaws → the STATE fails.
#[test]
fn hidden_power_fire_does_not_thaw_a_frozen_defender() {
    let d = dex();
    // p1 Metagross Hidden-Power-Fires a FROZEN Regirock (pure Rock → Fire resisted 0.5x + big
    // SpD → survives; genderless → no construction gender draw). Regirock is slower, so it also
    // rolls its OWN on_before_move freeze-thaw (1/5) — this seed FAILS that roll, so a
    // still-frozen result isolates the onDamagingHit (non-)thaw the fix controls.
    let atk = "Metagross|||ClearBody|hiddenpowerfire,flamethrower,psychic,earthquake|Hardy|,,,252,,|N||||";
    let def = "Regirock|||ClearBody|rockslide,earthquake,rest,curse|Sassy|252,,,,252,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(atk, def, "8,15,20,27"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[1].pokemon[0].status = Some(Status::Freeze);

    // p1 Hidden Power Fire (slot 0); p2 "Rock Slide" (slot 0) — but frozen + the thaw roll fails
    // → p2 cants (draw-free past its freeze roll).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);

    assert_eq!(
        st.sides[1].pokemon[0].species_id, "regirock",
        "Regirock is p2's active"
    );
    // THE FIX: Regirock is STILL FROZEN after the HP Fire hit (its OWN thaw roll failed at this
    // seed, and HP Fire does NOT run the onDamagingHit thaw). Under the bug it would be None.
    assert_eq!(
        out.decisions[0].active[1].status, Some(Status::Freeze),
        "Hidden Power Fire does NOT thaw the frozen defender (REVERT the is_fire narrowing → \
         onDamagingHit thaws it → status None)"
    );
    // Regirock took the (resisted) HP Fire damage — the hit LANDED; only the thaw was suppressed.
    assert!(
        out.decisions[0].active[1].hp < out.decisions[0].active[1].maxhp,
        "Regirock took the Hidden Power Fire damage (the hit landed; only the thaw was suppressed)"
    );
    // Draw-neutral stability anchor (the thaw is a pure status clear — no PRNG; the STATE above
    // is the true signal). BAKE the port's post-turn seed.
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "42908,65324,10639,56158",
        "post-turn seed (draw-neutral: the freeze-persistence fix consumes no PRNG)"
    );
}

/// T1F control (a): a REAL Fire move (Flamethrower — base type Fire) DOES thaw the frozen
/// defender via `frz.onDamagingHit` — proving the fix is scoped to Hidden Power / Weather Ball
/// only (not a blanket "no thaw"). Revert the narrowing and BOTH this and the HP-Fire test
/// thaw; keep the narrowing and only this one thaws.
#[test]
fn flamethrower_does_thaw_a_frozen_defender() {
    let d = dex();
    let atk = "Metagross|||ClearBody|hiddenpowerfire,flamethrower,psychic,earthquake|Hardy|,,,252,,|N||||";
    let def = "Regirock|||ClearBody|rockslide,earthquake,rest,curse|Sassy|252,,,,252,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(atk, def, "8,15,20,27"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    st.sides[1].pokemon[0].status = Some(Status::Freeze);

    // p1 Flamethrower (slot 1); p2 cants (frozen, thaw roll fails at this seed).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(1), Choice::Move(0))], &d);

    assert_eq!(
        out.decisions[0].active[1].status, None,
        "Flamethrower (base-type Fire) THAWS the frozen defender via frz.onDamagingHit"
    );
}

/// T1F control (b): Flash Fire STILL ABSORBS a Hidden Power Fire hit (its `onTryHit` reads the
/// RESOLVED move type, not `is_fire`), so the `is_fire` narrowing did NOT break the absorb.
/// An un-frozen Ninetales absorbs the HP Fire (0 damage) and ARMS `flash_fire`.
#[test]
fn flash_fire_still_absorbs_hidden_power_fire() {
    let d = dex();
    let atk = "Metagross|||ClearBody|hiddenpowerfire,flamethrower,psychic,earthquake|Hardy|,,,252,,|N||||";
    let def = "Ninetales|||FlashFire|rest,protect,flamethrower,psychic|Modest|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(atk, def, "1,2,3,4"), &d).expect("start");
    let st = battle.state_mut().expect("state");

    // p1 Hidden Power Fire (slot 0) into the Flash Fire Ninetales; p2 Rest (slot 0, idle at full HP).
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);

    assert_eq!(
        out.decisions[0].active[1].hp, out.decisions[0].active[1].maxhp,
        "Flash Fire ABSORBS the Hidden Power Fire hit (0 damage — the absorb reads the resolved \
         Fire type, unaffected by the is_fire thaw narrowing)"
    );
    assert!(
        battle.state().unwrap().sides[1].pokemon[0].flash_fire,
        "Ninetales' Flash Fire is ARMED after absorbing the Hidden Power Fire"
    );
}



// ============================================================================
// D4b — MUSTRECHARGE's `duration: 2` residual handler RUNS faintMessages
//       (`gen3_perside_residual_faint_upkeep_order_v1`, the round-5 D4 regression).
//       The D4 fix added a blanket `else { continue; }` to the non-stall
//       VolatileDuration arm — correct for the duration:1 group
//       (protect/flinch/focuspunch/pursuit/reactive/beatup/endure/snatch), each of
//       which decrements 1 → 0 this turn so the sim's `fieldEvent('Residual')` takes
//       the `duration-- == 0` end/`continue` branch (SKIP faintMessages). But
//       `mustrecharge` is `duration: 2`: its ONLY residual tick (the Hyper Beam CAST
//       turn) decrements 2 → 1 (NON-zero), so the sim FALLS THROUGH to
//       `this.faintMessages()` (battle.ts:508-567; `handler.end` = removeVolatile is
//       truthy for every volatile, so the branch fires but does NOT end at 2→1) and
//       `|upkeep` is added AFTER `fieldEvent` (battle.ts:2837-2838). So a faint an
//       earlier order-≤12 handler (Perish, order 12) enqueued-but-deferred is DRAINED
//       at the NO_ORDER mustrecharge handler → `|faint|` BEFORE `|upkeep|`. WRONG
//       (post-D4, pre-D4b): the blanket `continue` treated mustrecharge as duration:1
//       → SKIPPED faintMessages → the Perish `|faint|` deferred PAST `|upkeep|` to the
//       runAction-tail `process_faints`. FIX: mustrecharge gets its own
//       `MustRechargeDuration` residual variant that falls through (no `continue`).
//       Emission-order fix (draw-neutral — process_faints consumes no PRNG here), so
//       the teeth are the `|faint|`-vs-`|upkeep|` SEQUENCE.
// ============================================================================

/// D4b: a Celebi perishing-out on the SAME turn the foe's Chansey has `mustrecharge`
/// (from a Hyper Beam that hit) emits the perish `|faint|` BEFORE `|upkeep|` — the sim
/// drains the deferred faint at Chansey's NO_ORDER mustrecharge handler (duration 2→1,
/// runs faintMessages), inside `fieldEvent('Residual')`, ahead of the trailing
/// `|upkeep|`. WRONG (pre-fix blanket `continue`): the faint deferred past `|upkeep|`.
/// The p2 Snorlax perishes with Celebi on the cast turn, then SWITCHES OUT (its perish
/// clears) so ONLY Celebi reaches perish0 while the un-perished Chansey casts Hyper Beam.
#[test]
fn mustrecharge_duration_two_runs_faintmessages_so_perish_faint_precedes_upkeep() {
    let d = dex();
    let p1 = "Celebi|||NoAbility|perishsong,splash|Serious|252,,,,,|N||||]Snorlax|||NoAbility|splash|Serious|252,,,,,|N||||";
    let p2 = "Snorlax|||NoAbility|splash|Serious|252,,,,,|N||||]Chansey|||NoAbility|hyperbeam,splash|Serious|252,,,,,|N||||";
    // seed 44317,42357,9927,48760 → the Hyper Beam HITS (so Chansey gets mustrecharge).
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44317,42357,9927,48760"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Celebi Perish Song (both →perish3)
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // Celebi Splash; p2 Snorlax→Chansey (clears p2 perish)
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // Celebi Splash; Chansey Splash
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Celebi Splash; Chansey Hyper Beam → mustrecharge; Celebi perish0
            ScriptDecision::one(0, Choice::Switch(1)),              // Celebi's forced replacement
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();

    // Sanity: the Hyper Beam landed (Chansey is under mustrecharge) and Celebi hit perish0.
    let mr = raw.iter().position(|l| l.contains("-mustrecharge") && l.contains("Chansey"));
    let mr = mr.expect("Chansey's Hyper Beam must land → |-mustrecharge| (else the scenario is vacuous)");
    let faint = raw[mr..]
        .iter()
        .position(|l| l.contains("faint") && l.contains("Celebi"))
        .map(|i| i + mr)
        .expect("Celebi must faint from perish0 after the mustrecharge line");
    let upkeep = raw[mr..]
        .iter()
        .position(|l| l.contains("upkeep"))
        .map(|i| i + mr)
        .expect("the |upkeep| marker of the perish turn");

    // THE PIN: the perish `|faint|` is DRAINED at the mustrecharge handler → BEFORE
    // `|upkeep|`. Pre-fix (blanket continue): `|upkeep|` came first (faint deferred to
    // the runAction tail).
    assert!(
        faint < upkeep,
        "the mustrecharge (duration:2) handler must RUN faintMessages so the deferred perish \
         |faint| (idx {faint}) precedes |upkeep| (idx {upkeep}); pre-fix the blanket `continue` \
         skipped it → |upkeep| first. Emitted tail:\n{}",
        raw[mr..=upkeep].join("\n")
    );

    // Draw-neutral: emission-order only (process_faints consumes no PRNG on this board),
    // so the final decision's post-turn seed is unchanged by the fix.
    assert!(out.decisions[3].active[0].fainted, "Celebi fainted at perish0 on the Hyper Beam turn");
}

// ============================================================================
// #QC — gen3 QUICK CLAW speed=65535 override (`gen3_quick_claw_speed_v1`, the P1/P2
//        byte-fuzz fix). `Battle.quickClawRoll` is drawn UNCONDITIONALLY at every
//        completed `endTurn` (`randomChance(1,5)`, battle.js:1485); next turn gen3
//        `getActionSpeed` (scripts.js:47-48) returns `speed = 65535` for a Quick-Claw
//        HOLDER whose roll hit TRUE, so the (raw-)SLOWER holder moves FIRST within its
//        priority bracket. The port previously drew-and-DISCARDED the endTurn roll and
//        ordered purely on raw speed, so a QC-proc turn mis-ordered (and, when a swapped
//        damage roll crossed a KO threshold, produced a wrongful faint / HP off-by-one —
//        invisible to the seed check because a move-order swap consumes the SAME draws).
//        Constructed: a FAST Electrode (no item) vs a SLOW Shuckle holding Quick Claw,
//        both spamming Swift (never-miss, no secondary). GROUND TRUTH from the real
//        Showdown sim (`harness/probe_quick_claw_rng.js`, seed [15,106,198,260] →
//        post-construction initSeed 26217,1191,64492,10583):
//          turn 1: firstMover=p1 (raw speed), qcRoll set TRUE at endTurn,
//                  seedAfter=1766,21561,8304,26954, p1=245/261 p2=169/181
//          turn 2: firstMover=p2  (QC 65535 override — the SLOW Shuckle moves FIRST),
//                  seedAfter=3406,14238,51840,10041, p1=230/261 p2=162/181
//          turn 3: firstMover=p2  (qcRoll still TRUE from turn 2),
//                  seedAfter=56758,26706,56190,9940, p1=215/261 p2=156/181
//        Pre-fix the port keeps p1 first on turns 2-3 (raw speed) → wrong first_mover +
//        divergent post-turn seeds (the swapped draws) — this test FAILS if the fix is
//        reverted. Both mons are No-Ability so construction/switch-ins draw nothing (the
//        Rust initSeed == the sim's post-construction getSeed).
// ============================================================================
#[test]
fn quick_claw_proc_makes_the_slow_holder_move_first_seed() {
    let d = dex();
    let p1 = "Electrode|||NoAbility|swift|Serious||N||||";
    let p2 = "Shuckle||QuickClaw|NoAbility|swift|Serious||N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "26217,1191,64492,10583"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");

    let dec = ScriptDecision::both(Choice::Move(0), Choice::Move(0));
    let out = st.run_full_battle(&[dec, dec, dec], &d);

    assert_eq!(out.decisions.len(), 3, "three move decisions recorded");

    // FIRST MOVER: p1 (raw-fast Electrode) on turn 1; then the Quick-Claw holder (p2
    // Shuckle) FIRST on turns 2-3 via the 65535 override. Pre-fix: p1 all three turns.
    let expect_first: [Option<usize>; 3] = [Some(0), Some(1), Some(1)];
    for (i, exp) in expect_first.iter().enumerate() {
        assert_eq!(
            out.decisions[i].first_mover, *exp,
            "decision {i} first mover: turn 1 = raw-fast p1, turns 2-3 = the Quick-Claw \
             holder p2 (65535 override). Pre-fix the port kept p1 first → wrong order"
        );
    }

    // POST-TURN SEEDS: the QC reorder swaps which mon's move draws run first, so a
    // mis-order desyncs these even though the draw COUNT is unchanged.
    let expect_seed = [
        "1766,21561,8304,26954",
        "3406,14238,51840,10041",
        "56758,26706,56190,9940",
    ];
    for (i, exp) in expect_seed.iter().enumerate() {
        assert_eq!(
            seed_str(&out.decisions[i].seed_after), *exp,
            "decision {i} post-turn seed == the real Showdown seed"
        );
    }

    // FINAL HP after turn 3 (the swapped Swift damage rolls under the QC order).
    assert_eq!(st.sides[0].pokemon[0].hp, 215, "p1 Electrode HP after turn 3");
    assert_eq!(st.sides[1].pokemon[0].hp, 156, "p2 Shuckle HP after turn 3");
}

// ============================================================================
// L1D — the `switch`/`drag` DETAILS LEVEL SUFFIX (`gen3_details_level_suffix_v1`,
//       the round-9 randbats-byte-arm unblock). Showdown's `Pokemon.details` is
//       `<Species>[, L<level>][, <gender>][, shiny]` — it emits `, L<n>` iff level
//       != 100 and OMITS it at L100 (probe-confirmed via /tmp/probe_level_details.js,
//       gen3randombattle). The port's `switch_details` (turn.rs, the omniscient
//       |switch|/|drag| + the per-side stream via fold_hp_line) and `bridge.rs::details`
//       (the request JSON) used to emit NO level suffix, so the randbats byte-differential
//       arm WALLED at the first non-L100 |switch|. gen3ou is always L100, which is why the
//       pool goldens never hit it (and stay byte-identical). WRONG (pre-fix): a non-L100
//       lead showed `|switch|p1a: Lunatone|Lunatone|255/255` where the sim shows
//       `|switch|p1a: Lunatone|Lunatone, L84|255/255`. Revert the `, L{level}` insertion
//       in switch_details → the L84 assertion fails; the L100 mon must still OMIT it.
#[test]
fn switch_details_emit_level_suffix_only_when_not_l100() {
    let d = dex();
    // p1 lead is an L84 Lunatone (genderless → no construction gender draw), p2 an L100
    // Snorlax. Both Splash a turn (draw-free besides Quick Claw) — the framing |switch|
    // lines are what we assert.
    let p1 = "Lunatone|||Levitate|splash|Serious|,,,,,|N|||84|";
    let p2 = "Snorlax|||Immunity|splash|Serious|,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "1,2,3,4"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (_out, lines) =
        st.run_full_battle_logged(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);

    // The non-L100 lead's framing |switch| DETAILS carries `, L84` after the species.
    let p1_switch = lines
        .iter()
        .map(|l| l.0.as_str())
        .find(|l| l.starts_with("|switch|p1a:"))
        .expect("a p1 lead |switch| line");
    assert!(
        p1_switch.contains("|Lunatone, L84|"),
        "the L84 lead's |switch| DETAILS must carry `, L84` after the species — got `{p1_switch}`"
    );

    // The L100 mon OMITS the level suffix entirely (bare species in DETAILS).
    let p2_switch = lines
        .iter()
        .map(|l| l.0.as_str())
        .find(|l| l.starts_with("|switch|p2a:"))
        .expect("a p2 lead |switch| line");
    assert!(
        p2_switch.contains("|Snorlax|") && !p2_switch.contains(", L"),
        "the L100 mon's |switch| DETAILS must be the bare species (no `, L<n>`) — got `{p2_switch}`"
    );
}

// ============================================================================
// ROUND 10 — the byte-fuzz sweep bugs (RB1 / RM1 / RM3; BR1 lives in bridge_test.rs).
// Ground truth: `harness/probe_round10_regression_rng.js`.
// ============================================================================

/// RB1 (`gen3_encore_disable_move_shuffle_v1`): a Choice-Band mon (choicelock) whose FOE
/// Encores it carries BOTH `choicelock` + `encore` at that turn's endTurn, so
/// `runEvent('DisableMove')` gathers >=2 move-disabling `onDisableMove` volatiles → a size-2
/// Fisher-Yates tie-shuffle draws ONE `random` BEFORE the Quick Claw. The port's
/// `disable_move_event_shuffle` used to OMIT `encore` from the handler count, so an
/// encore+choicelock mon drew ONE FEWER at endTurn → a draw-count desync one call before the
/// Quick Claw. WRONG (pre-fix): the port draws one fewer at endTurn → the post-turn seed
/// diverges from the ground truth (revert-verified: it lands on `60833,51486,28767,2196`).
/// With the fix the extra shuffle fires and the post-turn seed equals the real Showdown ground
/// truth. The `choice_lock_only_draws_no_disable_move_shuffle` control (no encore, n==1) draws
/// NO shuffle — its seed DIFFERS, proving the extra draw is encore-gated.
#[test]
fn encore_plus_choice_lock_draws_the_disable_move_shuffle() {
    let d = dex();
    // p1 Snorlax holds Choice Band → Body Slam LOCKS the slot (choicelock). p2 BULKY Blissey
    // (survives the hit so the turn reaches endTurn) Encores the Snorlax.
    let snorlax = "Snorlax||choiceband|thickfat|bodyslam,earthquake|Jolly|,252,,,4,252|||||";
    let blissey = "Blissey|||naturalcure|encore,splash|Bold|252,,252,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax, blissey, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // Turn 1: p1 Body Slam (locks + sets lastMove) ; p2 Encore the Snorlax. endTurn: the
    // encore+choicelock mon draws the size-2 DisableMove tie-shuffle.
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(out.decisions.len(), 1, "one move boundary");
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "21177,35776,56648,13607",
        "encore+choicelock mon draws the DisableMove tie-shuffle before the Quick Claw — post-turn \
         seed == the real Showdown ground truth (reverting the `+ encore` addend drops the shuffle \
         → the port draws one fewer → the seed diverges)"
    );
}

/// RB1 CONTROL: a choicelock-ONLY mon (the foe does NOT Encore) carries a SINGLE
/// move-disabling volatile at endTurn (n==1) → NO tie-shuffle → the post-turn seed DIFFERS
/// from the encore+choicelock case. This pins that the shuffle is EncorE-gated (n>=2), not
/// always drawn — so the fix adds the draw ONLY when a second disabling volatile co-occurs.
#[test]
fn choice_lock_only_draws_no_disable_move_shuffle() {
    let d = dex();
    let snorlax = "Snorlax||choiceband|thickfat|bodyslam,earthquake|Jolly|,252,,,4,252|||||";
    let blissey = "Blissey|||naturalcure|splash,softboiled|Bold|252,,252,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(snorlax, blissey, "13127,45333,18295,15391"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        seed_str(&out.decisions[0].seed_after),
        "55318,8071,46680,56242",
        "choicelock-only (no encore) → n==1 → NO DisableMove shuffle → the control seed \
         (DIFFERS from the encore+choicelock case, proving the extra draw is encore-gated)"
    );
}

/// RM1 (`gen3_brick_break_screens_v1`): Brick Break — the ONLY gen3 screen-breaking move —
/// removes BOTH the foe side's screens BEFORE the damage step (draw-free `onTryHit`), so it
/// deals FULL (non-halved) damage AND clears the screen. WRONG (pre-fix): the port modeled
/// Brick Break as a plain Fighting move, so `build_damage_context` read `sides[foe].reflect>0`
/// and `modify_damage` halved the damage (~2× under-deal), and the screen persisted. STATE
/// pin: after Brick Break into a Reflecting foe, `sides[1].reflect == 0` AND the defender's HP
/// == the NO-screen control's HP (full damage) — reverting leaves reflect==1 + ~half damage
/// (higher HP). SEED pin: draws are unchanged (acc+crit+dmg+QC; the screen removal is
/// draw-free and no both-screens tie-shuffle fires), so the post-turn seed == the real sim.
#[test]
fn brick_break_removes_reflect_and_deals_full_damage() {
    let d = dex();
    let machamp = "Machamp|||noability|brickbreak,splash|Adamant|252,252,,,,|||||";
    let snorlax = "Snorlax|||thickfat|reflect,splash|Careful|252,,,,252,|||||";
    // Turn 1: p1 Splash, p2 Reflect (screen up). Turn 2: p1 Brick Break (removes + full dmg).
    let mut battle =
        Battle::start_with_switchins(&opts_cg(machamp, snorlax, "59913,41696,27939,16894"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 2, "two move boundaries");
    // The screen is REMOVED (Brick Break's onTryHit).
    assert_eq!(
        st.sides[1].reflect, 0,
        "Brick Break removes the foe's Reflect (pre-fix it persisted → reflect stays 1)"
    );
    // FULL (non-halved) damage: the defender lands on the SAME HP as the no-screen control
    // (probe RM1: 122/524). Pre-fix, the Reflect halved the damage → higher HP.
    assert_eq!(
        st.sides[1].pokemon[0].hp, 122,
        "Brick Break deals FULL non-halved damage (== the no-screen control's 122); pre-fix the \
         Reflect ×0.5 left the defender at a HIGHER HP"
    );
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "13764,11416,53420,29181",
        "the screen removal is draw-free + no both-screens ModifyDamagePhase1 tie-shuffle fires \
         → post-turn seed == the real Showdown ground truth"
    );
}

/// RM1 negative pin (`gen3_brick_break_screens_v1`, round-11 hardening): the Brick Break
/// screen-clear sits AFTER the immunity short-circuit in `run_move` — a GHOST is Fighting-immune,
/// so the hit-but-immune return fires BEFORE the screen-removal site. So a Brick Break into a
/// Ghost must LEAVE the foe's Reflect + Light Screen UP (only the normal end-of-turn SIDE residual
/// ticks them down by one — reflect 3→2, light_screen 5→4). WRONG (a mis-placed clear that fired
/// on an immune hit — i.e. moving the RM1 clear ABOVE the immune return): both screens would drop
/// to 0. The LANDING case is covered by `brick_break_removes_reflect_and_deals_full_damage`; this
/// hardens the immune branch. STATE pin (screens NOT cleared to 0 — the residual tick only). No
/// seed pin: the screen-clear is draw-free and the immune move draws only its accuracy roll.
#[test]
fn brick_break_into_a_ghost_does_not_clear_screens() {
    let d = dex();
    let machamp = "Machamp|||noability|brickbreak,splash|Adamant|252,252,,,,|||||";
    // Gengar (Ghost/Poison) — Fighting is 0× vs Ghost, so Brick Break is IMMUNE.
    let gengar = "Gengar|||levitate|splash|Timid|,,,252,,252|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(machamp, gengar, "59913,41696,27939,16894"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // Screens already UP on the Ghost's side (as if cast on a prior turn).
    st.sides[1].reflect = 3;
    st.sides[1].light_screen = 5;
    let out = st.run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    assert_eq!(out.decisions.len(), 1, "one move boundary");
    // Brick Break is Fighting-IMMUNE vs the Ghost → the immune return fires BEFORE the screen-clear
    // site, so the screens are NOT removed; the end-of-turn SIDE residual just ticks them once
    // (reflect 3→2, light_screen 5→4). A clear-on-immune bug would leave BOTH at 0.
    assert_ne!(
        st.sides[1].reflect, 0,
        "Brick Break into a Ghost must NOT clear Reflect (the immune return precedes the clear)"
    );
    assert_ne!(
        st.sides[1].light_screen, 0,
        "Brick Break into a Ghost must NOT clear Light Screen (the immune return precedes the clear)"
    );
    assert_eq!(
        st.sides[1].reflect, 2,
        "the screens are untouched by the immune Brick Break — only the residual ticks Reflect 3→2"
    );
    assert_eq!(
        st.sides[1].light_screen, 4,
        "the screens are untouched by the immune Brick Break — only the residual ticks Light Screen 5→4"
    );
    // The Ghost took ZERO damage (Fighting-immune) — confirms the immune path really fired.
    assert_eq!(
        st.sides[1].pokemon[0].hp, st.sides[1].pokemon[0].maxhp,
        "the Ghost is Fighting-immune → takes no Brick Break damage"
    );
}

/// RM3 (`gen3_sand_upkeep_under_air_lock_v1`): under Air Lock / Cloud Nine the sand/hail field
/// residual STILL emits its `|-weather|Sandstorm|[upkeep]` line (order 8) BEFORE the leech
/// `|-damage|` (order 10.5) — the sim gates only the eachEvent shuffle + the chip on
/// `effectiveWeather()`, NOT the whole handler + its upkeep-line emission. WRONG (pre-fix): the
/// port gated the ENTIRE WeatherChip handler off `effective_weather()`, so under Cloud Nine the
/// upkeep line was OMITTED and the leech `-damage` led. This is emission-only + DRAW-NEUTRAL
/// (the negated residual draws nothing either way), so the pin asserts the EMIT ORDER (the
/// upkeep line exists + precedes the leech damage) — reverting drops the upkeep line, failing
/// the order. The post-turn seed is unchanged (a draw-neutrality confirmation).
#[test]
fn sand_upkeep_line_emitted_under_cloud_nine_before_leech_damage() {
    let d = dex();
    // p1 Tyranitar (Sand Stream) + Leech Seed. p2 Golduck (Cloud Nine, Water so leech lands,
    // grounded non-Rock/Ground/Steel). T1: p1 Leech Seed, p2 Splash. T2 (record): both Splash.
    let ttar = "Tyranitar|||sandstream|leechseed,splash|Adamant|252,252,,,,|||||";
    let golduck = "Golduck|||cloudnine|splash,surf|Bold|252,,252,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(ttar, golduck, "42281,21615,44080,6072"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    assert_eq!(out.decisions.len(), 2, "two move boundaries");
    let raw: Vec<&str> = lines.iter().map(|l| l.0.as_str()).collect();
    let upkeep_idx = raw
        .iter()
        .position(|l| *l == "|-weather|Sandstorm|[upkeep]")
        .expect(
            "the sand `|-weather|Sandstorm|[upkeep]` line MUST be emitted under Cloud Nine \
             (pre-fix the whole WeatherChip handler was gated off effective_weather → no upkeep line)",
        );
    let leech_idx = raw
        .iter()
        .position(|l| l.contains("[from] Leech Seed"))
        .expect("a leech `|-damage|…|[from] Leech Seed` line");
    assert!(
        upkeep_idx < leech_idx,
        "the order-8 sand `[upkeep]` line must precede the order-10.5 leech `-damage` — got \
         upkeep@{upkeep_idx} leech@{leech_idx}"
    );
    // DRAW-NEUTRAL: the post-turn seed is unchanged (the negated residual draws nothing).
    assert_eq!(
        seed_str(&out.decisions[1].seed_after),
        "44727,38044,16858,42709",
        "RM3 is draw-neutral — the post-turn seed == the real Showdown ground truth"
    );
}

// ============================================================================
// HP-TAUNT — a Taunted mon keeps its BARE `hiddenpower` selectable
//      (`gen3_iv_derived_hidden_power_bp_v1`, the LEGALITY sibling; repro ab_233_8).
// ============================================================================

/// A Taunted mon's BARE Hidden Power stays selectable (the ab_233_8 legality fix).
#[test]
fn taunt_does_not_block_a_bare_hidden_power() {
    let d = dex();
    let mon = "Charizard|||blaze|hiddenpower,swordsdance,flamethrower,toxic|Modest|,,,252,,252|N||||";
    let foe = "Snorlax|||immunity|splash|Serious|252,,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(mon, foe, "1,2,3,4"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    assert!(!d.moves("hiddenpower").unwrap().blocked_by_taunt());
    assert!(d.moves("swordsdance").unwrap().blocked_by_taunt());
    st.sides[0].pokemon[0].taunt = Some(2);
    let cha = &st.sides[0].pokemon[0];
    assert!(cha.move_usable(0, &d), "bare Hidden Power selectable under Taunt (ab_233_8)");
    assert!(!cha.move_usable(1, &d), "Swords Dance (Status) IS taunt-blocked");
    assert!(cha.move_usable(2, &d), "Flamethrower selectable under Taunt");
    assert!(!cha.move_usable(3, &d), "Toxic (Status) IS taunt-blocked");
    assert!(!cha.must_struggle(&d), "not forced to Struggle");
}

// ============================================================================
// HP-MIMIC — a Mimic of Hidden Power uses the MIMICKER's OWN HP type
//      (`gen3_mimic_hidden_power_type_v1`; repro rmrr03rmc_ab_777_3). gen-3 Hidden
//      Power's TYPE is a property of the USER (derived from the user's IVs at use
//      time); Showdown stores the moveslot id BARE (`hiddenpower`) and re-derives
//      per holder, so a Mimic-copied HP takes the MIMICKER's type. WRONG (pre-fix):
//      `turn.rs`'s Mimic success block stored the copied mon's TYPED HP id
//      (`hiddenpowergrass`, num 363), so the mimicker used the ORIGINAL user's Grass
//      type instead of re-deriving Dark from its own IVs -> a Grass 0.25x hit that
//      barely dented the target (ab_777_3: Claydol's Mimic-copied HP left Charizard
//      near full instead of dealing neutral Dark damage). The fix canonicalizes a
//      copied `hiddenpower*` id to the BARE `hiddenpower`. STATE pin (the isolated
//      T3 mimicked-HP damage is NEUTRAL Dark, not 0.25x Grass).
// ============================================================================

/// A Mimic of a foe's Hidden Power re-derives the type from the MIMICKER's own IVs
/// (the ab_777_3 fix). WRONG (pre-fix): the mimicker kept the copied mon's typed HP
/// (Grass 0.25x). Moltres (all-31 IVs -> HP Dark) Mimics a foe's HP Grass then uses
/// it vs a Fire/Flying target -> NEUTRAL Dark damage. Reverting the bare-HP
/// canonicalization stores `hiddenpowergrass` -> Grass 0.25x -> the T3 hit is ~1/4.
#[test]
fn mimic_of_hidden_power_uses_the_mimickers_own_type() {
    let d = dex();
    // p1 Moltres (Fire/Flying, all-31 IVs -> its OWN Hidden Power is Dark, BP 70) uses only
    // Mimic + a harmless SPLASH filler, so the ONLY damage Charizard takes is the mimicked HP.
    let p1 = "Moltres|||pressure|mimic,splash|Modest|,,,252,,252|N||||";
    // p2 Charizard (Fire/Flying) carries the TYPED `hiddenpowergrass` (num 363, type Grass);
    // its last move becomes HP Grass so Mimic copies it. Dark is NEUTRAL vs Fire/Flying (1x);
    // Grass is 0.25x -> the two differ ~4x, isolating the bug.
    let p2 = "Charizard|||blaze|hiddenpowergrass,splash|Modest|,,,252,,4|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "53303,35262,36397,29520"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let cha_max = st.sides[1].pokemon[0].maxhp;

    // T1: p1 SPLASH ; p2 HP Grass -> p2's last_move = HP Grass (Charizard untouched).
    // T2: p1 Mimic (faster) -> copies HP Grass into slot 0 (stored BARE `hiddenpower` by the
    //     fix, BEFORE p2's T2 move) ; p2 splash.
    let _ = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
        ],
        &d,
    );
    // Post-Mimic: the mimicked slot is the BARE `hiddenpower` (the fix; reverting stores
    // `hiddenpowergrass`). Pin the stored id directly (the mechanism).
    assert_eq!(
        st.sides[0].pokemon[0].set.moves[0], "hiddenpower",
        "Mimic canonicalizes the copied Hidden Power to the BARE id (reverting stores \
         `hiddenpowergrass` -> the mimicker keeps Grass instead of its own Dark)"
    );
    // T3: p1 uses the mimicked slot 0 -> Hidden Power re-derived from Moltres's OWN IVs (DARK,
    //     NEUTRAL vs Fire/Flying) ; p2 splash. Measure ONLY the T3 mimicked-HP damage.
    let cha_before = st.sides[1].pokemon[st.sides[1].active].hp;
    let _ = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(1))], &d);
    let cha_after = st.sides[1].pokemon[st.sides[1].active].hp;
    let dealt = cha_before - cha_after;
    // A NEUTRAL Dark HP (BP 70) deals a substantial chunk (>1/4 maxhp); the pre-fix
    // `hiddenpowergrass` storage would deal 0.25x -> ~1/4 as much (well under 1/4 maxhp).
    assert!(
        dealt > cha_max / 4,
        "the mimicked HP hit NEUTRALLY (Moltres's OWN Dark type), NOT 0.25x Grass — the T3 hit \
         dealt {dealt} of {cha_max}"
    );
}

// ============================================================================
// I2 — gen-3 INTIMIDATE vs a FORCED REPLACEMENT
//      (`gen3_intimidate_forced_replacement_v1`; repro rmrr03rmc_ab_1381_0, DEC 51).
//      A deferred switch-in Intimidate must NOT drop the Atk of a foe that REPLACED
//      the intended target between the switch action and the deferred `RunSwitch`.
//      Composition: p1 Tyranitar Pursuit-KOs a switching Destiny-Bond Gengar; the DB
//      faints Tyranitar; p2's Salamence (Intimidate) enters; p1 force-replaces the
//      DB-fainted Tyranitar with Snorlax. In Showdown the entrant's runSwitch/
//      Intimidate resolves as part of its own switch action (against the original,
//      now-fainted Tyranitar), NEVER against the later Snorlax. WRONG (pre-fix): the
//      port's deferred RunSwitch read the post-replacement foe active (Snorlax) ->
//      dropped its Atk to -1 (golden DEC 51 p1 Snorlax boosts [0,0,0,0,0]; port
//      [-1,0,0,0,0]). The fix captures the foe uid at switch-in (`switchin_foe_uid`)
//      and suppresses the drop on a mis-target. STATE pin (Snorlax's Atk stays 0).
//      (The residual PROTOCOL byte-ORDERING of the deferred hint is a pre-existing
//      deferred-RunSwitch limitation, unmasked once the STATE matches — the spec's
//      rejected higher-risk reorder; out of scope for this STATE pin.)
// ============================================================================

/// I2: a deferred switch-in Intimidate does not drop a foe that force-replaced its
/// intended target. WRONG (pre-fix): `intimidate_on_start` read the post-replacement
/// foe -> Atk -1. Reverting the `switchin_foe_uid` guard drops the replacement's Atk to -1.
#[test]
fn intimidate_entrant_does_not_drop_a_forced_replacement_foe() {
    let d = dex();
    // p1: Tyranitar (Pursuit) — the pursuer that KOs the switching Gengar then FAINTS to its
    // Destiny Bond — + Snorlax, the forced replacement whose Atk must stay 0.
    let p1 = "Tyranitar|||sandstream|pursuit,crunch|Adamant|252,252,,,,|N||||\
              ]Snorlax|||immunity|bodyslam,splash|Adamant|252,252,,,,|N||||";
    // p2: Gengar (Destiny Bond, injected low-HP + armed) + Salamence (Intimidate), the entrant.
    let p2 = "Gengar|||levitate|destinybond,shadowball|Timid|,,,252,,252|N||||\
              ]Salamence|||intimidate|dragonclaw,splash|Adamant|,252,,,,|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "53303,35262,36397,29520"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    // INJECT: Gengar at 1 HP (Pursuit x2 KOs it on the switch-out interrupt) + its Destiny Bond
    // ARMED (as if it had used Destiny Bond).
    st.sides[1].pokemon[0].hp = 1;
    st.sides[1].pokemon[0].destiny_bond = true;

    // dec0: p1 Tyranitar Pursuit ; p2 voluntarily switches Gengar -> Salamence. The Pursuit
    // INTERRUPT strikes the switching Gengar -> KO -> its Destiny Bond faints Tyranitar ->
    // Salamence enters -> p1 is forced to replace the DB-fainted Tyranitar.
    // dec1: p1 forced-switch -> Snorlax (slot 1).
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Switch(1)),
            ScriptDecision::one(0, Choice::Switch(1)),
        ],
        &d,
    );
    assert!(
        out.decisions.len() >= 2,
        "the Pursuit -> Destiny Bond -> forced-replacement composition chains >=2 boundaries"
    );
    // Snorlax (p1's forced replacement) is now active. Its Atk MUST stay 0 — Salamence's
    // deferred switch-in Intimidate targeted the (now-fainted/replaced) Tyranitar, NOT Snorlax.
    let lax = &st.sides[0].pokemon[st.sides[0].active];
    assert_eq!(lax.species_id, "snorlax", "p1's forced replacement is Snorlax");
    assert_eq!(
        lax.boosts[0], 0,
        "the forced-replacement Snorlax's Atk stays 0 — the deferred switch-in Intimidate must \
         NOT drop a foe that REPLACED its intended target (reverting the switchin_foe_uid guard \
         in intimidate_on_start drops it to -1)"
    );
}

/// Y1 YAWN — the resolve draws the sleep `random(2,6)` at the RIGHT turn (`gen3_yawn_v1`, the KEY
/// draw-model pin). Yawn's CAST is DRAW-FREE (the volatile add draws nothing), and the sleep
/// `random(2,6)` fires at the RESOLVE — the end of the turn AFTER cast, via the residual `Yawn`
/// handler's `onEnd` → `trySetStatus('slp')`. So: the CAST turn's post-turn seed is IDENTICAL to a
/// Splash control (draw-free), and the RESOLVE turn's seed carries the extra `random(2,6)` (differs
/// from the Splash control). WRONG (the draw at cast, or absent, or the wrong turn): the resolve
/// seed diverges. Ground truth `harness/probe_batch89_yawn_regression_rng.js`.
#[test]
fn yawn_resolve_draws_the_sleep_random_2_6_at_the_next_turn_end() {
    let d = dex();
    let lax = "Snorlax|||Immunity|yawn,splash,earthquake,thunderwave|Adamant|4,252,,,252,|N||||";
    let bliss = "Blissey|||NaturalCure|splash|Careful|252,,,252,,|N||||";
    let seed = "13,27,41,55";

    // Yawn (move 1 = idx 0) at dec0 (cast), then Splash (move 2 = idx 1) at dec1 (the RESOLVE).
    let mut by = Battle::start_with_switchins(&opts_cg(lax, bliss, seed), &d).expect("start");
    let oy = by.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // dec0: Yawn cast (DRAW-FREE)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // dec1: Splash — the yawn RESOLVES
        ],
        &d,
    );
    // dec0: the cast is draw-free — Blissey is NOT yet asleep.
    assert!(oy.decisions[0].active[1].status.is_none(), "Y1: the Yawn cast does NOT sleep the foe");
    // dec1: the yawn RESOLVED — Blissey is asleep with the random(2,6) counter (3 at this seed).
    assert_eq!(
        oy.decisions[1].active[1].status,
        Some(Status::Sleep(3)),
        "Y1: the resolve sleeps the foe with the random(2,6) counter (3)"
    );
    assert_eq!(
        seed_str(&oy.decisions[0].seed_after),
        "49200,48550,14066,9078",
        "Y1: the CAST turn is DRAW-FREE — exact post-turn seed"
    );
    assert_eq!(
        seed_str(&oy.decisions[1].seed_after),
        "25367,53124,63953,41400",
        "Y1: the RESOLVE turn draws the sleep random(2,6) — exact post-turn seed"
    );

    // Splash control — the CAST-turn seed is IDENTICAL (draw-free), the RESOLVE-turn seed DIFFERS
    // (no random(2,6) — the discriminator that the draw is at the RESOLVE, not the cast).
    let mut bc = Battle::start_with_switchins(&opts_cg(lax, bliss, seed), &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // dec0: Splash
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // dec1: Splash (no resolve)
        ],
        &d,
    );
    assert_eq!(
        seed_str(&oc.decisions[0].seed_after),
        seed_str(&oy.decisions[0].seed_after),
        "Y1: the Splash control's cast-turn seed == Yawn's (the cast is draw-free)"
    );
    assert_ne!(
        seed_str(&oc.decisions[1].seed_after),
        seed_str(&oy.decisions[1].seed_after),
        "Y1: the Splash control's dec1 seed DIFFERS from Yawn's resolve (the extra random(2,6))"
    );
}

/// Y2 YAWN MIRROR — the `yawn` residual DURATION handler registers at (order 10, subOrder 19) and so
/// participates in the residual `speed_sort` tie-shuffle (`gen3_yawn_v1`). Two Snorlax Yawn each
/// OTHER at EQUAL speed → BOTH carry a yawn handler at (10, 19, equal speed) → they TIE → ONE
/// Fisher-Yates `random(0,2)` per residual (both the cast-turn 2→1 tick AND the resolve-turn 1→0
/// tick). A single-yawn control (only one mon carries the volatile) has NO tie → NO shuffle. WRONG
/// (a yawn handler NOT registered in the residual sort): the mirror loses the tie draws → the seed
/// matches the single-yawn control. Ground truth `harness/probe_batch89_yawn_regression_rng.js`.
#[test]
fn yawn_mirror_residual_handler_ties_and_draws_the_shuffle() {
    let d = dex();
    let lax = "Snorlax|||Immunity|yawn,splash|Adamant|4,252,,,252,|N||||";
    let seed = "17,33,49,61";

    // BOTH cast Yawn (dec0), BOTH Splash (dec1 → both resolve). The tie-shuffle draws at BOTH ticks.
    let mut bm = Battle::start_with_switchins(&opts_cg(lax, lax, seed), &d).expect("start");
    let om = bm.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // dec0: BOTH Yawn (draw-free casts)
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // dec1: BOTH Splash — both resolve
        ],
        &d,
    );
    // dec1: BOTH slept (each its own random(2,6) counter).
    assert!(matches!(om.decisions[1].active[0].status, Some(Status::Sleep(_))), "Y2: p1 slept");
    assert!(matches!(om.decisions[1].active[1].status, Some(Status::Sleep(_))), "Y2: p2 slept");
    assert_eq!(
        seed_str(&om.decisions[0].seed_after),
        "26004,37838,20426,60293",
        "Y2: the cast-turn residual (both yawn 2→1) draws the (10,19) tie-shuffle — exact seed"
    );
    assert_eq!(
        seed_str(&om.decisions[1].seed_after),
        "33154,17033,24911,25719",
        "Y2: the resolve-turn residual (both yawn 1→0) draws the tie-shuffle + 2×random(2,6)"
    );

    // Single-yawn control — only p2's mon carries a yawn volatile (p1 Splashes at dec0) → NO
    // tie → the cast-turn seed DIFFERS from the mirror (the mirror's extra tie-shuffle).
    let mut bs = Battle::start_with_switchins(&opts_cg(lax, lax, seed), &d).expect("start");
    let os = bs.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // dec0: p1 Splash / p2 Yawn
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // dec1: BOTH Splash — one resolve
        ],
        &d,
    );
    assert_ne!(
        seed_str(&os.decisions[0].seed_after),
        seed_str(&om.decisions[0].seed_after),
        "Y2: the single-yawn control's dec0 seed DIFFERS from the mirror (no residual tie-shuffle)"
    );
}

/// Y3 YAWN × gen3ou SLEEP CLAUSE at RESOLVE — the resolve routes through the EXISTING
/// `try_set_status('slp')` path, so the gen3ou **Sleep Clause Mod** applies at the resolve
/// (`gen3_yawn_v1`). Smeargle Spores Blissey-A (asleep, foe-inflicted), p2 switches A→B (A benched,
/// still asleep — Serene Grace, no cure), p1 Yawns B; B's yawn RESOLVES but p2's side already has a
/// living foe-slept mon (benched A) → the clause BLOCKS the sleep at the SetStatus event: it DRAWS
/// the 2-clause shuffle (+ the `|-message|Sleep Clause Mod activated.`) but NO `random(2,6)` (B stays
/// AWAKE). A no-prior-sleeper control resolves normally (the shuffle + random(2,6) → B sleeps), so
/// its resolve seed DIFFERS (the extra random(2,6)). WRONG (yawn NOT routed through try_set_status):
/// B would sleep + draw a stray random(2,6). Ground truth
/// `harness/probe_batch89_yawn_regression_rng.js`.
#[test]
fn yawn_resolve_is_blocked_by_the_sleep_clause_in_gen3ou() {
    let d = dex();
    let smeargle = "Smeargle|||Own Tempo|spore,yawn,splash,earthquake|Jolly|4,,,,,252|N||||";
    let bliss_a = "A|Blissey||Serene Grace|splash|Careful|252,,,252,,|N||||";
    let bliss_b = "B|Blissey||Serene Grace|splash|Careful|252,,,252,,|N||||";
    let p2 = format!("{bliss_a}]{bliss_b}");
    let opts = BattleOptions {
        format_id: "gen3ou".to_string(),
        seed: Some("23,37,51,63".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(smeargle.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.clone()) },
    };
    let mut bb = Battle::start_with_switchins(&opts, &d).expect("start");
    assert!(bb.state().unwrap().sleep_clause, "gen3ou carries the Sleep Clause Mod flag");
    let ob = bb.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // dec0: Spore -> Blissey-A slp
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // dec1: p2 switch A->B; p1 Yawn B
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // dec2: Splash — B's yawn RESOLVES -> CLAUSE
        ],
        &d,
    );
    // dec2: the clause BLOCKED the resolve — Blissey-B stays AWAKE (no slp).
    assert!(
        ob.decisions[2].active[1].status.is_none(),
        "Y3: the Sleep Clause blocks the Yawn resolve — Blissey-B stays awake"
    );
    assert_eq!(
        seed_str(&ob.decisions[2].seed_after),
        "51902,63332,1987,30260",
        "Y3: the clause-blocked resolve draws the SetStatus shuffle but NO random(2,6) — exact seed"
    );

    // Control (no prior sleeper — Splash at dec0 instead of Spore) — B's yawn resolves NORMALLY
    // (the shuffle + random(2,6) → B sleeps), so the resolve seed DIFFERS (the extra random(2,6)).
    let opts_c = BattleOptions {
        format_id: "gen3ou".to_string(),
        seed: Some("23,37,51,63".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(smeargle.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2) },
    };
    let mut bc = Battle::start_with_switchins(&opts_c, &d).expect("start");
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // dec0: Splash (no Spore)
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // dec1: p2 switch A->B; p1 Yawn B
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // dec2: Splash — B's yawn RESOLVES normally
        ],
        &d,
    );
    assert!(
        matches!(oc.decisions[2].active[1].status, Some(Status::Sleep(_))),
        "Y3: the control (no prior sleeper) sleeps Blissey-B normally"
    );
    assert_ne!(
        seed_str(&oc.decisions[2].seed_after),
        seed_str(&ob.decisions[2].seed_after),
        "Y3: the control's resolve seed DIFFERS from the clause-blocked one (the extra random(2,6))"
    );
}

/// TR1/TR2/TR3 TRICK — the SWAP is DRAW-FREE past accuracy (`gen3_trick_v1`). Trick draws ONE
/// accuracy roll then a draw-free item swap, so a genuine two-item SWAP, a Sticky-Hold `-immune`
/// (onTryImmunity), and a both-itemless FAIL (onHit `return false`) ALL draw the SAME count
/// (accuracy + the endTurn Quick Claw) → their post-turn seeds COINCIDE at the same init seed. This
/// pins BOTH the swap STATE (which item each side holds) AND the draw-parity (the three seeds are
/// equal — a spurious/missing swap draw would break it). WRONG (a swap that draws, or a Sticky-Hold /
/// fail that SKIPS the accuracy roll): the seeds diverge. Ground truth
/// `harness/probe_batch89_trick_regression_rng.js`.
#[test]
fn trick_swap_immune_and_fail_all_draw_only_accuracy_and_quick_claw() {
    let d = dex();
    let zam = |item: &str| format!("Alakazam||{item}|Synchronize|trick,splash|Timid|4,,,252,,252|N||||");

    // TR1: a full two-item swap (Silk Scarf <-> Leftovers).
    let foe1 = "Snorlax||leftovers|Immunity|splash|Careful|252,,252,,,|N||||";
    let mut b1 = Battle::start_with_switchins(&opts_cg(&zam("silkscarf"), foe1, "23,44,61,82"), &d).expect("start");
    let o1 = b1.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    let s1 = b1.state().unwrap();
    assert_eq!(pokesim::dex::to_id(&s1.sides[0].pokemon[0].item), "leftovers", "TR1: the user GAINS the foe's Leftovers");
    assert_eq!(pokesim::dex::to_id(&s1.sides[1].pokemon[0].item), "silkscarf", "TR1: the foe GAINS the user's Silk Scarf");
    assert_eq!(
        seed_str(&o1.decisions[0].seed_after),
        "28749,32144,46574,46132",
        "TR1: the swap is draw-free past accuracy — exact post-turn seed (accuracy + Quick Claw only)"
    );

    // TR2: a Sticky-Hold target → PLAIN `-immune`, NO swap. SAME draw count as TR1.
    let foe2 = "Muk||leftovers|StickyHold|splash|Careful|252,,252,,,|N||||";
    let mut b2 = Battle::start_with_switchins(&opts_cg(&zam("silkscarf"), foe2, "23,44,61,82"), &d).expect("start");
    let o2 = b2.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    let s2 = b2.state().unwrap();
    assert_eq!(pokesim::dex::to_id(&s2.sides[0].pokemon[0].item), "silkscarf", "TR2: Sticky Hold blocks the swap — the user keeps its item");
    assert_eq!(pokesim::dex::to_id(&s2.sides[1].pokemon[0].item), "leftovers", "TR2: the Sticky-Hold target keeps its item");
    assert_eq!(
        seed_str(&o2.decisions[0].seed_after),
        seed_str(&o1.decisions[0].seed_after),
        "TR2: a Sticky-Hold `-immune` Trick draws the SAME as a swap (accuracy + Quick Claw, no swap draw)"
    );

    // TR3: both sides itemless → FAIL (`[still]`+`-fail`), NO swap. SAME draw count.
    let foe3 = "Snorlax|||Immunity|splash|Careful|252,,252,,,|N||||";
    let mut b3 = Battle::start_with_switchins(&opts_cg(&zam(""), foe3, "23,44,61,82"), &d).expect("start");
    let o3 = b3.state_mut().unwrap().run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    let s3 = b3.state().unwrap();
    assert!(s3.sides[0].pokemon[0].item.is_empty() && s3.sides[1].pokemon[0].item.is_empty(), "TR3: both stay itemless (the fail does not swap)");
    assert_eq!(
        seed_str(&o3.decisions[0].seed_after),
        seed_str(&o1.decisions[0].seed_after),
        "TR3: a both-itemless FAIL draws the SAME as a swap (accuracy + Quick Claw)"
    );
}

/// TR4 TRICK CHOICE-LOCK RELEASE — a Choice-Band mon that Tricks its Band AWAY is UNLOCKED
/// (`gen3_trick_v1`): the lock is enforced by the item's `choiceband.onDisableMove`, so with the Band
/// gone the mon may pick any move again. This scripts a CB Alakazam that Tricks (locking to the Trick
/// slot, then swapping the Band away) then uses a DIFFERENT slot (Psychic) the NEXT turn. If the port
/// FAILED to release the user's `choice_locked_move`, `run_full_battle` would REJECT the second
/// decision (a locked-out slot) and skip it → only ONE decision record. WRONG (kept lock):
/// `decisions.len() == 1`. Ground truth `harness/probe_batch89_trick_regression_rng.js`.
#[test]
fn trick_away_a_choice_band_releases_the_users_choice_lock() {
    let d = dex();
    let zam = "Alakazam||choiceband|Synchronize|trick,psychic,splash|Timid|4,,,252,,252|N||||";
    let foe = "Snorlax||leftovers|Immunity|splash|Careful|252,,252,,,|N||||";
    let mut b = Battle::start_with_switchins(&opts_cg(zam, foe, "23,44,61,82"), &d).expect("start");
    let o = b.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // dec0: Trick (CB away, lock then RELEASED)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // dec1: Psychic — a DIFFERENT slot
        ],
        &d,
    );
    // BOTH decisions ran — the port accepted the follow-up Psychic (the lock was released). A kept
    // lock rejects dec1 draw-free → only ONE decision record.
    assert_eq!(
        o.decisions.len(),
        2,
        "TR4: the CB user is UNLOCKED after Tricking its Band away (both decisions ran); a kept lock would reject the 2nd move"
    );
    let s = b.state().unwrap();
    assert_eq!(pokesim::dex::to_id(&s.sides[0].pokemon[0].item), "leftovers", "TR4: the user swapped its Band for Leftovers");
    assert_eq!(pokesim::dex::to_id(&s.sides[1].pokemon[0].item), "choiceband", "TR4: the foe received the Choice Band");
    assert_eq!(
        seed_str(&o.decisions[1].seed_after),
        "59422,2393,22649,24171",
        "TR4: dec1 (the released Psychic) runs — exact post-turn seed"
    );
}

/// TR5 TRICK INTO A SUBSTITUTE — a Substitute blocks Trick (no `bypasssub`, `gen3_trick_v1`): the
/// accuracy is STILL drawn, then the move FAILS (`[still]`+`-fail`) with NO swap. This subs the foe
/// (dec0) then Tricks into it (dec1); the items stay UNCHANGED and the fail is draw-free past
/// accuracy (dec1 seed literal). WRONG (a swap through the sub, or a skipped accuracy roll): the
/// items diverge / the seed diverges. Ground truth `harness/probe_batch89_trick_regression_rng.js`.
#[test]
fn trick_into_a_substitute_fails_and_does_not_swap() {
    let d = dex();
    let zam = "Alakazam||silkscarf|Synchronize|trick,splash|Timid|4,,,252,,252|N||||";
    let foe = "Snorlax||leftovers|Immunity|substitute,splash|Careful|252,,252,,,|N||||";
    let mut b = Battle::start_with_switchins(&opts_cg(zam, foe, "23,44,61,82"), &d).expect("start");
    let o = b.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // dec0: p1 Splash / p2 Substitute
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // dec1: p1 Trick into the sub (FAIL) / p2 Splash
        ],
        &d,
    );
    let s = b.state().unwrap();
    assert!(s.sides[1].pokemon[0].substitute.is_some(), "TR5: the foe's Substitute is up");
    assert_eq!(pokesim::dex::to_id(&s.sides[0].pokemon[0].item), "silkscarf", "TR5: the user keeps its item (sub blocked the swap)");
    assert_eq!(pokesim::dex::to_id(&s.sides[1].pokemon[0].item), "leftovers", "TR5: the foe keeps its item behind the sub");
    assert_eq!(
        seed_str(&o.decisions[1].seed_after),
        "31583,64901,25391,35655",
        "TR5: the sub-blocked Trick draws its accuracy then fails draw-free — exact post-turn seed"
    );
}

// ============================================================================
// FORECAST fail-loud GIGO guard (Castform is DEFERRED / UNMODELED).
//
// Forecast swaps Castform's forme + TYPE under rain/sun/hail. It is MODELED now
// (`gen3_forecast_v1`, ROUND 35) — the construction fail-loud that stood guard
// while it was deferred is RETIRED, and the pin below is the guard's NEGATIVE
// CONTROL (the transform-carriers pattern): a Castform team must CONSTRUCT and
// forme correctly, so the guard cannot silently return.
// ============================================================================

/// The RETIRED Forecast guard's NEGATIVE CONTROL (`gen3_forecast_v1`): a Castform
/// (Forecast) team constructs fine, and a Drought foe's start-window sun formes it
/// `castformsunny` with the Fire type at `|turn|1` (probe
/// `probe_r35_forecast_edges.js` E1 drought-foe: `|-weather|SunnyDay|[from] ability:
/// Drought` → `|-formechange|p1a: Castform|Castform-Sunny|[msg]|[from] ability:
/// Forecast`). WRONG (the old guard restored): construction panics.
#[test]
fn forecast_castform_builds_now_that_forecast_is_modeled() {
    let d = dex();
    let castform = "Castform|||forecast|tackle|Hardy|252,,,,,|||||";
    let foe = "Ninetales|||drought|flamethrower|Modest|252,,,,,|||||";
    let b = Battle::start_with_switchins(&opts_cg(castform, foe, "1,2,3,4"), &d)
        .expect("a Castform team must construct now that Forecast is modeled");
    let st = b.state().unwrap();
    let cf = &st.sides[0].pokemon[st.sides[0].active];
    assert_eq!(cf.species_id, "castformsunny", "the start-window sun formes the lead Castform");
    assert_eq!(cf.base_species_id, "castform", "baseSpecies stays the construction species");
}

// ── ROUND 35 (`gen3_forecast_v1`) — the hail/sandstorm weather-set moves + the expiry
//    WeatherChange draw + the Forecast forme cycle. Ground truth
//    `harness/probe_r35_pin_truth.js` (the real sim's post-construction initSeed +
//    per-decision seedAfter on each board — the round-29 methodology). ──────────────────

/// FC1 — the HAIL move on a speed-TIED Snorlax mirror (96 = 96): the 5-turn timed set
/// (`|-weather|Hail`), the per-turn chip `max(1, 461/16) = 28` with the `[from] Hail`
/// cause, the expiry `|-weather|none`, all at the sim's EXACT per-decision seeds — on a
/// tied board every eachEvent shuffle draws, so ONE missing/mis-ordered draw anywhere in
/// the weather-move machinery desyncs a seed. WRONG (pre-round-35): the `hail` move was a
/// runtime fail-loud (`run_status_move` panic).
#[test]
fn hail_move_sets_chips_and_expires_on_a_tied_mirror() {
    let d = dex();
    let caster = "Snorlax|||immunity|hail,splash|Serious|,,,,,|||||";
    let filler = "Snorlax|||immunity|splash,splash|Serious|,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(caster, filler, "38085,56695,39077,36349"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let script: Vec<ScriptDecision> = (0..6)
        .map(|i| {
            ScriptDecision::both(Choice::Move(if i == 0 { 0 } else { 1 }), Choice::Move(0))
        })
        .collect();
    let (out, lines) = st.run_full_battle_logged(&script, &d);
    assert_eq!(out.decisions.len(), 6, "six move boundaries");
    let raw: Vec<&str> = lines.iter().map(|l| l.0.as_str()).collect();
    assert!(raw.contains(&"|-weather|Hail"), "the hail SET line");
    assert!(
        raw.contains(&"|-damage|p2a: Snorlax|433/461|[from] Hail"),
        "the first hail chip: 461 - 28 = 433, cause `[from] Hail`"
    );
    assert!(raw.contains(&"|-weather|none"), "the expiry line");
    for (i, want) in [
        "18621,25793,18448,35836",
        "44192,15,54621,10852",
        "11699,27371,61848,33740",
        "54635,13711,14274,14388",
        "12422,29923,60165,59292",
        "9210,15857,42907,12973",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(
            seed_str(&out.decisions[i].seed_after),
            *want,
            "FC1 dec{} seed == the real Showdown ground truth",
            i + 1
        );
    }
}

/// FC2 — **the T1 8-vs-7 fix** (`probe_r35_forecast_ties.js` T1, re-derived on a
/// VERIFIED-tied board by `probe_r35_weather_moves.js`): a SUPPRESSED (Cloud Nine) hail's
/// UPKEEP turns skip the eachEvent('Weather') draw (dec2-4 — the RM3 gate, unchanged), but
/// its EXPIRY turn draws the `clearWeather` → `eachEvent('WeatherChange')` shuffle
/// **UNCONDITIONALLY** (dec5). WRONG (pre-fix): the expiry branch kept the
/// `Sun|Rain || effective` gate → drew NOTHING at a suppressed sand/hail expiry → the dec5
/// seed (and every later one) desyncs. Suicune 206 vs Psyduck(spe-EV 240) 206 — the tie is
/// what makes the shuffle draw at all. Revert-verified: restoring the old gate fails dec5.
#[test]
fn suppressed_hail_expiry_draws_the_unconditional_weatherchange_shuffle() {
    let d = dex();
    let suicune = "Suicune|||pressure|hail,splash|Serious|,,,,,|||||";
    let psyduck = "Psyduck|||cloudnine|splash,splash|Serious|,,,,,240,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(suicune, psyduck, "55318,8071,46680,56242"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let script: Vec<ScriptDecision> = (0..6)
        .map(|i| {
            ScriptDecision::both(Choice::Move(if i == 0 { 0 } else { 1 }), Choice::Move(0))
        })
        .collect();
    let (out, lines) = st.run_full_battle_logged(&script, &d);
    assert_eq!(out.decisions.len(), 6, "six move boundaries");
    let raw: Vec<&str> = lines.iter().map(|l| l.0.as_str()).collect();
    assert!(
        !raw.iter().any(|l| l.contains("[from] Hail")),
        "a suppressed hail chips NOTHING (Cloud Nine)"
    );
    assert!(raw.contains(&"|-weather|Hail|[upkeep]"), "the upkeep line is UNCONDITIONAL");
    for (i, want) in [
        "60043,19557,54924,29018",
        "13954,21730,61912,63955",
        "55163,59650,16206,36280",
        "10158,60209,34927,2841",
        "16432,1625,37777,42721",
        "64835,58268,46930,35822",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(
            seed_str(&out.decisions[i].seed_after),
            *want,
            "FC2 dec{} seed == the real Showdown ground truth (dec5 is the expiry \
             WeatherChange draw the old gate missed)",
            i + 1
        );
    }
}

/// FC3 — the FORECAST forme CYCLE on a Castform casting Rain Dance: formes
/// `castformrainy` (type Water) at the set with the exact `|-formechange|` bytes, stays
/// formed through the upkeep turns, REVERTS to base at the expiry (the second
/// `-formechange`), all seed-exact. WRONG (pre-round-35): Castform was a construction
/// fail-loud; and a wrong forme/type would desync any damage chart read while formed.
#[test]
fn forecast_formes_on_the_weather_set_and_reverts_at_the_expiry() {
    let d = dex();
    let castform = "Castform|||forecast|raindance,splash|Serious|,,,,,|||||";
    let filler = "Snorlax|||immunity|splash,splash|Serious|,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(castform, filler, "57388,452,34593,29177"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let script: Vec<ScriptDecision> = (0..6)
        .map(|i| {
            ScriptDecision::both(Choice::Move(if i == 0 { 0 } else { 1 }), Choice::Move(0))
        })
        .collect();
    let (out, lines) = st.run_full_battle_logged(&script, &d);
    let raw: Vec<&str> = lines.iter().map(|l| l.0.as_str()).collect();
    let forme_idx = raw
        .iter()
        .position(|l| {
            *l == "|-formechange|p1a: Castform|Castform-Rainy|[msg]|[from] ability: Forecast"
        })
        .expect("the SET-turn formechange line, exact bytes");
    let weather_idx = raw.iter().position(|l| *l == "|-weather|RainDance").unwrap();
    assert!(weather_idx < forme_idx, "the -weather SET line precedes the -formechange");
    assert!(
        raw.contains(&"|-formechange|p1a: Castform|Castform|[msg]|[from] ability: Forecast"),
        "the EXPIRY revert formechange line, exact bytes"
    );
    let cf = &st.sides[0].pokemon[0];
    assert_eq!(cf.species_id, "castform", "reverted to base after the expiry");
    assert_eq!(cf.base_species_id, "castform");
    for (i, want) in [
        "3932,55062,24613,55040",
        "29587,16389,4131,29123",
        "55318,8071,46680,56242",
        "38085,56695,39077,36349",
        "60833,51486,28767,2196",
        "21177,35776,56648,13607",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(
            seed_str(&out.decisions[i].seed_after),
            *want,
            "FC3 dec{} seed == the real Showdown ground truth (the forme change is \
             DRAW-FREE — a spurious draw here desyncs)",
            i + 1
        );
    }
}

/// FC4 — the START-window forme is DRAW-FREE and seed-exact: a Castform lead vs a Drought
/// lead formes `castformsunny` (type FIRE — the chart read while formed) at `|turn|1`, and
/// the first two decisions land on the sim's exact seeds. Extends the negative control
/// above with the seed + type assertions.
#[test]
fn forecast_start_window_forme_is_draw_free_and_typed() {
    let d = dex();
    let castform = "Castform|||forecast|splash,splash|Serious|,,,,,|||||";
    let nine = "Ninetales|||drought|splash,splash|Serious|,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(castform, nine, "57388,452,34593,29177"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    {
        let cf = &st.sides[0].pokemon[0];
        assert_eq!(cf.species_id, "castformsunny", "formed by the start window");
    }
    let out = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );
    assert_eq!(seed_str(&out.decisions[0].seed_after), "3932,55062,24613,55040", "FC4 dec1");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "29587,16389,4131,29123", "FC4 dec2");
}

/// FC5 — the WEATHER_NEGATE `onEnd` site with the ENDING-NEGATER EXCLUSION: a Cloud Nine
/// Psyduck VOLUNTARILY leaves while rain stands, and the Castform formes ON THAT
/// WeatherChange — with the `-formechange` line emitted BEFORE the incoming `|switch|`
/// (probe O1c t2; ground truth PT-E). The exclusion is the load-bearing bit: at the event
/// the suppressor is STILL in its slot, so a plain `effective_weather` read would say
/// "suppressed" and leave the Castform base — the sim's `abilityState.ending` semantics
/// (`effective_weather_excluding`) is what formes it.
#[test]
fn cloud_nine_leaving_formes_the_castform_via_the_ending_exclusion() {
    let d = dex();
    let castform = "Castform|||forecast|raindance,splash|Serious|,,,,,|||||";
    let p2 = "Psyduck|||cloudnine|splash,splash|Serious|,,,,,|||||]\
              Blissey|||naturalcure|splash|Serious|,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(castform, p2, "57388,452,34593,29177"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    let raw: Vec<&str> = lines.iter().map(|l| l.0.as_str()).collect();
    let forme_idx = raw
        .iter()
        .position(|l| {
            *l == "|-formechange|p1a: Castform|Castform-Rainy|[msg]|[from] ability: Forecast"
        })
        .expect("the End-site formechange (the exclusion made the rain effective)");
    let switch_idx = raw
        .iter()
        .position(|l| l.starts_with("|switch|p2a: Blissey"))
        .expect("the incoming switch line");
    assert!(
        forme_idx < switch_idx,
        "the formechange fires on the LEAVING suppressor's onEnd — BEFORE the |switch| \
         (got forme@{forme_idx} switch@{switch_idx})"
    );
    assert_eq!(st.sides[0].pokemon[0].species_id, "castformrainy", "formed + stays formed");
    for (i, want) in [
        "3932,55062,24613,55040",
        "29587,16389,4131,29123",
        "55318,8071,46680,56242",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(seed_str(&out.decisions[i].seed_after), *want, "FC5 dec{}", i + 1);
    }
}

/// FC6 — the SILENT `clearVolatile` revert + the re-entry re-forme: a FORMED Castform
/// pivots out (base `castform` on the bench, NO line), then re-enters under the standing
/// rain and re-formes via its `onStart` (a SECOND `-formechange`, probe S2 t3; ground
/// truth PT-F — whose turn-0 Quick Claw came up TRUE, so the pin restores that bit like
/// `ab_replay`'s INIT `quickClawRoll` field, the round-30 lesson).
#[test]
fn formed_castform_reverts_on_switch_out_and_reformes_on_reentry() {
    let d = dex();
    let p1 = "Castform|||forecast|raindance,splash|Serious|,,,,,|||||]\
              Rattata|||guts|splash|Serious|,,,,,|||||";
    let snorlax = "Snorlax|||immunity|splash,splash|Serious|,,,,,|||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, snorlax, "3932,55062,24613,55040"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    st.quick_claw_roll = true; // PT-F: the sim's turn-0 endTurn Quick Claw HIT
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    let raw: Vec<&str> = lines.iter().map(|l| l.0.as_str()).collect();
    let formes: Vec<usize> = raw
        .iter()
        .enumerate()
        .filter(|(_, l)| l.contains("-formechange"))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(formes.len(), 2, "exactly TWO formechanges: the set + the re-entry re-forme");
    // The benched revert is SILENT: between the two formechanges sits the |switch| to
    // Rattata but NO forme line, and the re-entry |switch| PRECEDES the second forme line.
    let reentry_switch = raw
        .iter()
        .rposition(|l| l.starts_with("|switch|p1a: Castform"))
        .expect("the re-entry switch line");
    assert!(
        reentry_switch < formes[1],
        "|switch| then |-formechange| on re-entry (probe S2 t3 order)"
    );
    assert_eq!(st.sides[0].pokemon[st.sides[0].active].species_id, "castformrainy");
    for (i, want) in [
        "29587,16389,4131,29123",
        "55318,8071,46680,56242",
        "38085,56695,39077,36349",
        "60833,51486,28767,2196",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(seed_str(&out.decisions[i].seed_after), *want, "FC6 dec{}", i + 1);
    }
}

/// FC8 — the DOUBLE-replacement Castform × Intimidate ordering (the round's disclosed
/// honest-scope hole, now probed CLOSED — `probe_r35_double_replacement.js`): a mutual
/// Explosion double-KO under standing rain replaces BOTH actives simultaneously. The sim's
/// observable order is: both `|switch|` lines in ENTRANT-SPEED order (Salamence 236 before
/// Castform 176 — in BOTH side orientations, so it is speed, not side), then each entrant's
/// Start in that same order — the Intimidate block, THEN the `-formechange`. Each runSwitch
/// fires its own singleEvent Start, so `forecast.onSwitchInPriority: -2` never reorders
/// anything in gen-3 singles. Seed-exact through the whole boundary.
#[test]
fn double_replacement_castform_and_intimidate_order_by_entrant_speed() {
    let d = dex();
    let p1 = "Snorlax|||immunity|explosion,raindance|Adamant|,252,,,,|M||||]\
              Castform|||forecast|icebeam,splash|Serious|,,,,,|M||||";
    let p2 = "Abra|||innerfocus|splash,splash|Serious|,,,,,|M||||]\
              Salamence|||intimidate|splash,splash|Serious|,,,,,|M||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "29814,3811,34909,65508"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Rain Dance / Splash
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Explosion double-KO
            ScriptDecision::both(Choice::Switch(1), Choice::Switch(1)), // double replacement
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    let raw: Vec<&str> = lines.iter().map(|l| l.0.as_str()).collect();
    let idx = |needle: &str| {
        raw.iter()
            .position(|l| l.starts_with(needle))
            .unwrap_or_else(|| panic!("missing line starting {needle:?} in {raw:#?}"))
    };
    let sw_mence = idx("|switch|p2a: Salamence|Salamence, M|331/331");
    let sw_cf = idx("|switch|p1a: Castform|Castform, M|281/281");
    let intim = idx("|-ability|p2a: Salamence|Intimidate|boost");
    let unboost = idx("|-unboost|p1a: Castform|atk|1");
    let forme = idx("|-formechange|p1a: Castform|Castform-Rainy|[msg]|[from] ability: Forecast");
    assert!(
        sw_mence < sw_cf && sw_cf < intim && intim < unboost && unboost < forme,
        "the sim's boundary shape: switches (speed order) then Starts (same order) — got \
         mence@{sw_mence} cf@{sw_cf} intim@{intim} unboost@{unboost} forme@{forme}"
    );
    for (i, want) in [
        "10162,51001,19049,38839",
        "6877,19027,28905,2872",
        "35400,13963,12705,987",
        "20050,47592,16058,22314",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(seed_str(&out.decisions[i].seed_after), *want, "FC8 dec{}", i + 1);
    }
}

/// FC7 — the Castform MIRROR at a speed tie: the two `-formechange` lines emit in the
/// WeatherChange SHUFFLE's permutation, not a fixed side walk — seed `[11,22,33,44]`
/// (init `38085,…`) is p1-first, seed `[7,7,7,7]` (init `58176,…`) is p2-first, each at
/// the sim's exact post-decision seed (ground truth PT-G + the flip mini-sweep).
#[test]
fn castform_mirror_forme_order_follows_the_weatherchange_shuffle() {
    let d = dex();
    let caster = "Castform|||forecast|raindance,splash|Serious|,,,,,|||||";
    let sitter = "Castform|||forecast|splash,splash|Serious|,,,,,|||||";
    for (init, want_first, want_seed) in [
        ("38085,56695,39077,36349", "p1a", "18621,25793,18448,35836"),
        ("58176,46786,32157,41308", "p2a", "41619,23581,3836,12055"),
    ] {
        let mut battle =
            Battle::start_with_switchins(&opts_cg(caster, sitter, init), &d).expect("start");
        let st = battle.state_mut().expect("state");
        let (out, lines) = st.run_full_battle_logged(
            &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
            &d,
        );
        let formes: Vec<&str> =
            lines.iter().map(|l| l.0.as_str()).filter(|l| l.contains("-formechange")).collect();
        assert_eq!(formes.len(), 2, "both mirror Castforms forme");
        assert!(
            formes[0].starts_with(&format!("|-formechange|{want_first}")),
            "init {init}: the shuffle's permutation decides the order — expected \
             {want_first} first, got {:?}",
            formes
        );
        assert_eq!(seed_str(&out.decisions[0].seed_after), want_seed, "FC7 seed @ init {init}");
    }
}

/// `gen3_transform_failloud_v1` — a mon CARRYING **Transform** fail-louds at construction, the
/// MOVE analogue of the Forecast guard above. WRONG (pre-guard): the port built the mon and
/// `run_status_move` silently did NOTHING for Transform, while the sim emits
/// `|-transform|<user>|<target>` and rewrites the user's species/types/stats/moves wholesale —
/// so a `--use-bridge=rust` run kept feeding the policy WRONG OBSERVATIONS for the rest of the
/// battle instead of crashing. FOUND BY the external-consistency gate `gen_sim_bridge_diff.js`
/// (repro `sim_bridge_diff_out/soak_randbats/divergences/sbd_msapcesj_b22`, a randbats Ditto),
/// which reaches the LIVE bridge path the offline fuzzers' pickers had filtered Transform out
/// of — the reason this survived every existing gate.
/// The WRAP-FAMILY half of `gen3_unmodeled_move_failloud_v1` was RETIRED in ROUND 32
/// (`gen3_partial_trap_v1`) — the family is now MODELED bit-for-bit, so a wrap-carrying team must
/// BUILD, not panic. (It used to be `#[should_panic(expected = "wrap is unmodeled")]`; the repro
/// that motivated the guard was `soak3/.../sbd_msb1zfxs_b237`, where node emitted
/// `|-activate|p1a: Snorlax|move: Wrap|[of] p2a: Shuckle` and the port emitted nothing.) The
/// LIVE behaviour is pinned by the PT1-PT5 pins below; this one only guards the un-gating.
#[test]
fn wrap_family_builds_now_that_the_partial_trap_is_modeled() {
    let d = dex();
    for mv in ["wrap", "bind", "firespin", "clamp", "whirlpool", "sandtomb"] {
        let user = format!("Snorlax|||immunity|{mv}|Hardy|252,,,,,|||||");
        let foe = "Shuckle|||sturdy|toxic|Adamant|252,252,,,,|||||";
        assert!(
            Battle::start_with_switchins(&opts_cg(&user, foe, "1,2,3,4"), &d).is_ok(),
            "{mv} must build — the partial-trap family is modeled"
        );
    }
}

/// Negative control for the (now Transform-only) fail-loud guard — a mon whose moves merely
/// RESEMBLE the family must build fine. Guards against a future over-broad substring match.
#[test]
fn a_non_trapping_move_still_builds_fine() {
    let d = dex();
    let snorlax = "Snorlax|||immunity|bodyslam|Hardy|252,,,,,|||||";
    let foe = "Shuckle|||sturdy|toxic|Adamant|252,252,,,,|||||";
    assert!(Battle::start_with_switchins(&opts_cg(snorlax, foe, "1,2,3,4"), &d).is_ok());
}

// ============================================================================
// PARTIAL TRAP pins (PT1…PT5, `gen3_partial_trap_v1`) — the gen-3 wrap family
// (Wrap / Bind / Fire Spin / Clamp / Whirlpool / Sand Tomb).
//
// Ground truth: the REAL sim on each exact board, captured by `harness/probe_ptrap_pin.js`
// (which prints the POST-CONSTRUCTION seed each pin is seeded at — the round-29 methodology
// note: `start_with_switchins` is draw-free while the sim spends its first draws on the turn-0
// construction window, so seeding the port with the RAW seed would silently replay those as
// move-phase draws). The MECHANICS were settled by `harness/probe_batch89_trap.js` (the
// duration draw + the lifecycle) and `harness/probe_ptrap_edges{,2}.js` (the edges).
//
// THE DRAW MODEL: the ordinary damaging chain PLUS exactly ONE `random(3, 7)` at cast (the
// gen4-mod `partiallytrapped.durationCallback` inside `addVolatile`). Everything after that is
// DRAW-FREE. Each pin asserts the exact post-turn SEED as well as the protocol bytes, so a
// mis-POSITIONED or MISSING duration draw fails even when the visible lines happen to match.
// ============================================================================

/// PT1 — the FULL LIFECYCLE. Wrap lands, the `random(3,7)` returns 5 → FOUR chip turns
/// (`duration − 1`) of `floor(482/16) = 30` each, then the duration hits 0 and the `onEnd`
/// emits the NON-silent `|-end|p2a: Snorlax|Wrap|[partiallytrapped]` with NO chip that turn.
/// Also pins the FIRM switch-block (`trap_is_firm` — the `trapped:true` request shape) while
/// the trap is live, and its release the moment the volatile ends.
///
/// WRONG (pre-fix): the port ran Wrap as a plain damaging move — it lost the duration DRAW (so
/// every later seed diverged), the chip, AND the switch-legality: a triple desync.
#[test]
fn partial_trap_wrap_lifecycle_matches_the_sim() {
    let d = dex();
    let p1 = "Dragonite||Leftovers|InnerFocus|wrap,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||]\
              Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Wrap cast (+ the random(3,7))
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // chip
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // chip
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // chip
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // duration 1 -> 0: the -end
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let marks: Vec<&str> = raw
        .iter()
        .filter(|l| l.contains("partiallytrapped") || l.contains("move: Wrap"))
        .map(|l| l.as_str())
        .collect();
    assert_eq!(
        marks,
        vec![
            "|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Dragonite",
            "|-damage|p2a: Snorlax|452/482|[from] move: Wrap|[partiallytrapped]",
            "|-damage|p2a: Snorlax|452/482|[from] move: Wrap|[partiallytrapped]",
            "|-damage|p2a: Snorlax|452/482|[from] move: Wrap|[partiallytrapped]",
            "|-damage|p2a: Snorlax|452/482|[from] move: Wrap|[partiallytrapped]",
            "|-end|p2a: Snorlax|Wrap|[partiallytrapped]",
        ],
        "PT1: the real sim's cast/-activate, FOUR chips (duration 5 - 1) and the NON-silent \
         natural expiry, in order. got:\n{}",
        raw.join("\n")
    );
    // The SEED gate — a missing / mis-positioned `random(3,7)` breaks these even if the lines
    // coincidentally match (the chips are draw-free, so only the cast turn moves the stream).
    for (i, want) in [
        "58424,44781,32445,22157",
        "8730,10456,11782,14692",
        "23935,41153,16141,50999",
        "60870,54810,59059,42614",
        "4841,46532,32342,29009",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(seed_str(&out.decisions[i].seed_after), *want, "PT1 dec{i} post-turn seed");
    }
}

/// PT1b — the SWITCH-BLOCK half, on the same board: while the `partiallytrapped` volatile is
/// live the victim is TRAPPED and the trap is FIRM (`trapped:true` on the very first request +
/// an `[Invalid choice]` reject with NO re-request — the Shadow-Tag shape, probe M). It frees
/// the instant the volatile ends. Pure STATE, no PRNG.
#[test]
fn partial_trap_firm_trap_blocks_the_victims_switch() {
    let d = dex();
    let p1 = "Dragonite||Leftovers|InnerFocus|wrap,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||]\
              Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    assert!(!st.is_trapped(1, &d), "PT1b: untrapped before the cast");
    let _ = st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert!(st.is_trapped(1, &d), "PT1b: the wrapped mon is TRAPPED");
    assert!(st.trap_is_firm(1, &d), "PT1b: partiallytrapped is a FIRM trap (bare tryTrap())");
    assert!(!st.is_trapped(0, &d), "PT1b: the TRAPPER is free to switch");
    // Run it out: 3 more chips + the expiry turn; the trap must then be gone.
    let _ = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    assert!(!st.is_trapped(1, &d), "PT1b: the expiry frees the victim");
}

/// PT2 — the TRAPPER LEAVES. The turn the wrapper switches out, the residual sees
/// `!source.isActive` and takes the `onResidual` release branch: the volatile is deleted, a
/// SILENT `|-end|…|[partiallytrapped]|[silent]` is emitted and there is NO chip that turn.
#[test]
fn partial_trap_releases_silently_when_the_trapper_leaves() {
    let d = dex();
    let p1 = "Dragonite||Leftovers|InnerFocus|wrap,splash|Hardy|85,85,85,85,85,85|M||||]\
              Gengar||Leftovers|Levitate|splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||]\
              Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),   // Wrap lands
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)), // the TRAPPER pivots out
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),   // nothing left to chip
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let marks: Vec<&str> = raw
        .iter()
        .filter(|l| l.contains("partiallytrapped") || l.contains("move: Wrap"))
        .map(|l| l.as_str())
        .collect();
    assert_eq!(
        marks,
        vec![
            "|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Dragonite",
            "|-damage|p2a: Snorlax|452/482|[from] move: Wrap|[partiallytrapped]",
            "|-end|p2a: Snorlax|Wrap|[partiallytrapped]|[silent]",
        ],
        "PT2: the trapper-gone branch is SILENT and takes NO chip. got:\n{}",
        raw.join("\n")
    );
    assert!(!st.is_trapped(1, &d), "PT2: the victim is free once the trapper left");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "8730,10456,11782,14692",
        "PT2: the release is DRAW-FREE (the real sim's post-turn seed)");
}

/// PT3 — a SUBSTITUTE eats the whole cast: the sub's `onTryPrimaryHit` returns before
/// `runMoveEffects`, so there is NO volatile AND — the bit-for-bit crux — NO `random(3,7)`.
/// The seed assertion is what makes this pin meaningful: a port that drew the duration and
/// merely skipped the volatile would still print the right lines but desync every later turn.
#[test]
fn partial_trap_substitute_blocks_the_volatile_and_the_duration_draw() {
    let d = dex();
    let p1 = "Dragonite||Leftovers|InnerFocus|wrap,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|substitute,splash|Hardy|85,85,85,85,85,85|M||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "42771,25880,27089,15090"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // p2 Substitute
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Wrap into the sub
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // ...and again
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    assert!(
        !raw.iter().any(|l| l.contains("partiallytrapped")),
        "PT3: a sub-absorbed Wrap applies NO partial trap. got:\n{}",
        raw.join("\n")
    );
    assert!(!st.is_trapped(1, &d), "PT3: the sub owner is not trapped");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "13438,47692,23971,41473",
        "PT3: the sub-absorbed cast draws NO random(3,7) — the real sim's post-turn seed");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "50887,33032,2836,3461", "PT3 dec2 seed");
}

/// PT4 — RAPID SPIN by the victim clears the trap. The gen4-mod `rapidspin.self.onHit` calls
/// `removeVolatile('partiallytrapped')` LAST (after the leech-seed / hazard clears), and because
/// it is a `removeVolatile` the condition's `onEnd` fires — so the NON-silent
/// `|-end|<user>|<Move>|[partiallytrapped]` is emitted, with NO `[from] move: Rapid Spin` tag
/// (unlike the leech-seed and Spikes clears on the same handler). DRAW-FREE.
#[test]
fn partial_trap_rapid_spin_clears_it_with_the_non_silent_end() {
    let d = dex();
    let p1 = "Dragonite||Leftovers|InnerFocus|wrap,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Starmie||Leftovers|NaturalCure|rapidspin,splash|Hardy|85,85,85,85,85,85|N||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Wrap lands on Starmie
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Starmie SPINS
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)),
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let marks: Vec<&str> = raw
        .iter()
        .filter(|l| l.contains("partiallytrapped") || l.contains("move: Wrap"))
        .map(|l| l.as_str())
        .collect();
    assert_eq!(
        marks,
        vec![
            "|-activate|p2a: Starmie|move: Wrap|[of] p1a: Dragonite",
            "|-damage|p2a: Starmie|263/282|[from] move: Wrap|[partiallytrapped]",
            "|-end|p2a: Starmie|Wrap|[partiallytrapped]",
        ],
        "PT4: Rapid Spin ends the trap with the NON-silent form and no [from] tag. got:\n{}",
        raw.join("\n")
    );
    assert!(!st.is_trapped(1, &d), "PT4: the spin frees the victim");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "4841,46532,32342,29009",
        "PT4: the spin's release is DRAW-FREE (the real sim's post-turn seed)");
}

/// RG1 — RAGE: the `singlemove` volatile raises the holder's Atk whenever a FOE lands a
/// damaging move while it is up, and the HOLDER'S OWN next move removes it.
///
/// ⚠️ THE WINDOW IS WHAT MAKES THIS TESTABLE. `onBeforeMove` runs at priority 100, so the
/// volatile is gone before the holder acts again — the boost is only observable when the foe
/// strikes BETWEEN the Rage and the holder's next action. The first probe of this move used a
/// board where the holder moved first every turn and read `atk=0` throughout, which reads
/// exactly like "Rage does nothing". Nidoking outspeeds Snorlax, so the foe's Tackle lands
/// inside the window on the SAME turn.
#[test]
fn rage_boosts_attack_when_the_foe_hits_inside_the_window() {
    let d = dex();
    let user = "Nidoking||none|PoisonPoint|rage,splash|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Snorlax||none|Sturdy|tackle,splash|Hardy|85,85,85,85,85,85|M||||";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "3,3,3,3"), &d).expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Rage, then Tackle lands
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // again
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let joined = raw.join("\n");
    assert!(
        raw.iter().any(|l| l.as_str() == "|-singlemove|p1a: Nidoking|Rage"),
        "RG1: casting Rage announces the singlemove volatile. got:\n{joined}"
    );
    // NON-VACUITY: the foe must actually have hit inside the window.
    assert!(
        raw.iter().any(|l| l.starts_with("|-damage|p1a: Nidoking|")),
        "RG1: the foe must land a damaging move for the boost to be testable. got:\n{joined}"
    );
    assert!(
        st.sides[0].pokemon[0].boosts[1] >= 1,
        "RG1: a foe hit inside the window raises the rager's Atk (got {})",
        st.sides[0].pokemon[0].boosts[1]
    );
}

/// RG2 — SECRET POWER is an ORDINARY 70-BP Normal move in gen 3.
///
/// Its `onModifyMove` swaps the secondary by TERRAIN, and gen-3 has no terrain at all — the
/// handler's first line is `if (this.field.isTerrain("")) return;`, so it returns early and
/// the base 30% paralysis secondary stands. Worth a pin precisely because "this move carries
/// an onModifyMove" reads as MISMODELED until you check what it does with no terrain.
#[test]
fn secret_power_keeps_its_base_paralysis_secondary_with_no_terrain() {
    let d = dex();
    let user = "Nidoking||none|PoisonPoint|secretpower,splash|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Snorlax||none|Sturdy|splash|Hardy|85,85,85,85,85,85|M||||";
    // Sweep seeds until the 30% secondary lands at least once — a single seed would leave the
    // "it still has its secondary" claim resting on luck.
    let mut ever_par = false;
    for seed in ["3,3,3,3", "11,11,11,11", "7,7,7,7", "1,2,3,4"] {
        let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        for _ in 0..4 {
            st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
        }
        assert!(
            st.sides[1].pokemon[0].hp < st.sides[1].pokemon[0].maxhp,
            "RG2: Secret Power must deal ordinary damage"
        );
        if st.sides[1].pokemon[0].status == Some(pokesim::state::Status::Paralysis) {
            ever_par = true;
            break;
        }
    }
    assert!(
        ever_par,
        "RG2: the base 30% paralysis secondary must still fire — gen-3 has no terrain, so the \
         onModifyMove returns early and must NOT clear the secondary"
    );
}

/// RO1 — ROLLOUT / ICE BALL: the base power doubles per EXECUTION over a 5-execution lock,
/// and DEFENSE CURL doubles every rung again. Ground truth
/// `harness/probe_rollout_defensecurl.js` (HP deltas 30/57/101/213, and 56/108/204 curled).
///
/// ⚠️ The count is EXECUTIONS, not turns — a MISS never reaches the callback, so it does not
/// advance the ladder. The pin therefore reads the counter out of STATE and compares damage
/// only between two runs that differ solely by Defense Curl.
#[test]
fn rollout_doubles_per_execution_and_defense_curl_doubles_it_again() {
    let d = dex();
    let user = "Marowak||none|RockHead|rollout,splash,defensecurl|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Snorlax||none|Sturdy|splash|Hardy|85,85,85,85,85,85|M||||";
    let seed = "3,3,3,3";

    // The execution counter advances 1,2,3,4 and the lock ENDS after the 5th.
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
    let st = b.state_mut().expect("state");
    let pp0 = st.sides[0].pokemon[0].pp_of(0);
    let mut counter = Vec::new();
    for _ in 0..5 {
        st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
        counter.push(st.sides[0].pokemon[0].rollout.map(|(h, _)| h));
    }
    assert!(
        counter.iter().any(|c| c.is_some()),
        "RO1: the lock must arm at all — got {counter:?}"
    );
    assert_eq!(
        counter.last().copied().flatten(),
        None,
        "RO1: the lock ENDS after the 5th execution — got {counter:?}"
    );
    assert_eq!(
        st.sides[0].pokemon[0].pp_of(0),
        pp0 - 1,
        "RO1: PP is spent ONCE for the whole 5-execution lock"
    );

    // DEFENSE CURL doubles the rung. Same seed, same script shape, one extra setup turn — and
    // the comparison is made on the FIRST Rollout of each run so the ladder position matches.
    // ⚠️ Rollout is accuracy 90, so a fixed seed can MISS and silently make the comparison
    // vacuous (the first draft asserted against a 0-damage control). Sweep seeds and use the
    // first where BOTH arms connect; the assertion below then guards non-vacuity anyway.
    let first_hit = |curl: bool, seed: &str| -> u16 {
        let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        if curl {
            st.run_full_battle(&[ScriptDecision::both(Choice::Move(2), Choice::Move(0))], &d);
            assert!(st.sides[0].pokemon[0].defense_curl, "RO1: the curl must land");
        } else {
            st.run_full_battle(&[ScriptDecision::both(Choice::Move(1), Choice::Move(0))], &d);
        }
        let before = st.sides[1].pokemon[0].hp;
        st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
        before - st.sides[1].pokemon[0].hp
    };
    let mut plain = 0u16;
    let mut curled = 0u16;
    for sd in ["3,3,3,3", "11,11,11,11", "7,7,7,7", "1,2,3,4", "31,29,23,19"] {
        let p = first_hit(false, sd);
        let c = first_hit(true, sd);
        if p > 0 && c > 0 {
            plain = p;
            curled = c;
            break;
        }
    }
    assert!(
        plain > 0 && curled > 0,
        "RO1: no seed had BOTH arms connect — the comparison would be vacuous"
    );
    assert!(
        curled > plain + plain / 2,
        "RO1: Defense Curl must roughly DOUBLE the rung ({curled} vs {plain})"
    );
}

/// RO2 — DEFENSE CURL's own effect: a never-miss, draw-free +1 Def, plus the volatile that
/// Rollout reads. Both halves matter — the boost is what the move does, the volatile is why it
/// had to be modeled alongside Rollout.
#[test]
fn defense_curl_raises_defense_and_sets_the_volatile() {
    let d = dex();
    let user = "Marowak||none|RockHead|defensecurl,splash|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Snorlax||none|Sturdy|splash|Hardy|85,85,85,85,85,85|M||||";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "3,3,3,3"), &d).expect("start");
    let st = b.state_mut().expect("state");
    st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        st.sides[0].pokemon[0].boosts[2], 1,
        "RO2: Defense Curl raises Def by one stage"
    );
    assert!(
        st.sides[0].pokemon[0].defense_curl,
        "RO2: and sets the volatile Rollout reads"
    );
}

/// LI1 — the LOCK-IN family (OUTRAGE / PETAL DANCE / THRASH): a `random(2,4)` duration drawn
/// ONCE on the cast, the user locked for exactly that many turns, PP spent once, then
/// CONFUSION when the lock runs out. Ground truth `harness/probe_lockin_family.js`.
///
/// All three share ONE `lockedmove` condition, so the pin walks all three ids rather than
/// trusting that a single representative generalises.
#[test]
fn the_lock_in_family_locks_for_its_rolled_duration_then_confuses() {
    let d = dex();
    for (mv, slot) in [("outrage", 0usize), ("petaldance", 0), ("thrash", 0)] {
        let user = format!(
            "Dragonite||none|InnerFocus|{mv},splash|Hardy|85,85,85,85,85,85|M||||"
        );
        let foe = "Snorlax||none|Sturdy|splash|Hardy|85,85,85,85,85,85|M||||";
        let mut b = Battle::start_with_switchins(&opts_cg(&user, foe, "3,3,3,3"), &d)
            .expect("start");
        let st = b.state_mut().expect("state");
        let pp0 = st.sides[0].pokemon[0].pp_of(slot);

        // Turn 1: the cast arms the lock with a rolled duration of 2 or 3.
        st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
        let rolled = st.sides[0].pokemon[0]
            .locked_move
            .map(|(n, _)| n)
            .unwrap_or_else(|| panic!("LI1 [{mv}]: the cast must arm the lock"));
        assert!(
            (1..=3).contains(&rolled),
            "LI1 [{mv}]: the duration is a random(2,4) → 2 or 3 (one already ticked), got {rolled}"
        );
        // The user is LOCKED: a switch must be refused.
        assert!(
            st.sides[0].pokemon[0].move_locked(),
            "LI1 [{mv}]: the user must be move-locked while the volatile is live"
        );

        // Run it out. The lock ends in CONFUSION — which is what separates this family from
        // Uproar, whose lock ends cleanly.
        for _ in 0..4 {
            st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
            if st.sides[0].pokemon[0].locked_move.is_none() {
                break;
            }
        }
        assert!(
            st.sides[0].pokemon[0].locked_move.is_none(),
            "LI1 [{mv}]: the lock must run out within its rolled duration"
        );
        assert!(
            st.sides[0].pokemon[0].confusion.is_some(),
            "LI1 [{mv}]: the user is CONFUSED when the lock ends (unlike Uproar)"
        );
        // PP is spent ONCE for the whole lock, not once per turn.
        assert_eq!(
            st.sides[0].pokemon[0].pp_of(slot),
            pp0 - 1,
            "LI1 [{mv}]: PP is deducted once, on the CAST — every continuing turn is a lockedmove"
        );
    }
}

/// LI2 — SLEEP breaks the lock cleanly, with NO end-of-lock confusion.
///
/// The sim's `onResidual` deletes the volatile when the user is asleep, BEFORE the duration
/// is consulted — so an asleep user never takes the confusion. A model that only counted the
/// duration down would confuse it anyway.
#[test]
fn sleep_breaks_the_lock_in_without_confusing_the_user() {
    let d = dex();
    let user = "Dragonite||none|InnerFocus|outrage,splash|Hardy|85,85,85,85,85,85|M||||";
    // The foe Spores the locked user, which should end the lock silently.
    let foe = "Butterfree||none|Sturdy|spore,splash|Hardy|85,85,85,85,85,85|M||||";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "3,3,3,3"), &d).expect("start");
    let st = b.state_mut().expect("state");
    st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(1))], &d);
    assert!(
        st.sides[0].pokemon[0].locked_move.is_some(),
        "LI2: the lock must be live before the Spore, else this proves nothing"
    );
    st.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);

    // NON-VACUITY: the user really is asleep.
    assert!(
        matches!(st.sides[0].pokemon[0].status, Some(pokesim::state::Status::Sleep(_))),
        "LI2: the Spore must land for this pin to test the sleep branch"
    );
    assert!(
        st.sides[0].pokemon[0].locked_move.is_none(),
        "LI2: sleep BREAKS the lock at the residual"
    );
    assert!(
        st.sides[0].pokemon[0].confusion.is_none(),
        "LI2: and it breaks it CLEANLY — no end-of-lock confusion"
    );
}

/// BP1 — REVENGE doubles when the user was DAMAGED BY THE TARGET this turn.
///
/// Priority −4 makes the foe's move land first, which is what makes the doubling the common
/// case. Ground truth `harness/probe_bp_cluster_clean.js` (50 vs 24 on a Mew mirror).
#[test]
fn revenge_doubles_after_being_damaged_by_the_target() {
    let d = dex();
    let user = "Mew||none|Synchronize|revenge,splash|Hardy|85,85,85,85,85,85|N||||";
    let foe = "Mew||none|Synchronize|tackle,splash|Hardy|85,85,85,85,85,85|N||||";
    let seed = "3,3,3,3";
    let dmg = |foe_move: usize| -> u16 {
        let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        st.run_full_battle(
            &[ScriptDecision::both(Choice::Move(0), Choice::Move(foe_move))],
            &d,
        );
        st.sides[1].pokemon[0].maxhp - st.sides[1].pokemon[0].hp
    };
    let hit = dmg(0); // foe Tackles first → Revenge doubles
    let calm = dmg(1); // foe Splashes → base power
    assert!(
        hit > calm + calm / 2,
        "BP1: Revenge must roughly DOUBLE after being hit ({hit} vs {calm})"
    );
}

/// BP2 — SMELLING SALTS doubles vs a PARALYZED target AND cures the paralysis.
///
/// ⚠️ The first attempt at this measurement read the SAME damage in both arms and nearly
/// concluded "no doubling". It was a coincidence: the control arm CRITTED (`random(16)->0`),
/// and a crit is itself a ×2. The board below uses a Sturdy foe (no Synchronize reflecting
/// paralysis back at the user) and asserts the CURE as well, which no crit can fake.
#[test]
fn smelling_salts_doubles_on_a_paralyzed_target_and_cures_it() {
    let d = dex();
    let user = "Mew||none|Synchronize|thunderwave,smellingsalts,splash\
|Hardy|85,85,85,85,85,85|N||||";
    let foe = "Snorlax||none|Sturdy|splash|Hardy|85,85,85,85,85,85|M||||";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "3,3,3,3"), &d).expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Thunder Wave
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Salts vs par → 2x + cure
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Salts vs cured → base
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let joined = raw.join("\n");
    assert!(
        raw.iter().any(|l| l.as_str() == "|-status|p2a: Snorlax|par"),
        "BP2: the foe must actually be paralyzed first. got:\n{joined}"
    );
    assert!(
        raw.iter().any(|l| l.as_str() == "|-curestatus|p2a: Snorlax|par|[msg]"),
        "BP2: a landed Smelling Salts CURES the paralysis. got:\n{joined}"
    );
    assert!(
        st.sides[1].pokemon[0].status.is_none(),
        "BP2: and the state agrees. got:\n{joined}"
    );
    // The two damage numbers, read off the -damage lines in order.
    let dmgs: Vec<u16> = raw
        .iter()
        .filter_map(|l| {
            let p = l.strip_prefix("|-damage|p2a: Snorlax|")?;
            let hp: u16 = p.split('/').next()?.parse().ok()?;
            Some(hp)
        })
        .collect();
    assert!(dmgs.len() >= 2, "BP2: two Salts hits expected. got:\n{joined}");
    let max = st.sides[1].pokemon[0].maxhp;
    let first = max - dmgs[0];
    let second = dmgs[0].saturating_sub(dmgs[1]);
    assert!(
        first > second + second / 2,
        "BP2: the vs-paralyzed hit must roughly DOUBLE the cured one ({first} vs {second})"
    );
}

/// BP3 — FURY CUTTER's consecutive-use ladder (10/20/40/80/160) and its `duration: 2` lapse.
///
/// ⚠️ Fury Cutter is accuracy 95, so a MISS silently shifts the whole ladder, and a CRIT is a
/// ×2 that mimics a rung. The assertion is therefore RATIO-based over consecutive NON-crit,
/// NON-miss hits rather than absolute numbers.
#[test]
fn fury_cutter_doubles_each_consecutive_use_and_lapses_after_one_gap() {
    let d = dex();
    let user = "Mew||none|Synchronize|furycutter,splash|Hardy|85,85,85,85,85,85|N||||";
    let foe = "Snorlax||none|Sturdy|splash|Hardy|85,85,85,85,85,85|M||||";

    // The multiplier is pure STATE, so read it directly — that dodges both confounds.
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "3,3,3,3"), &d).expect("start");
    let st = b.state_mut().expect("state");
    let mut ladder = Vec::new();
    for _ in 0..5 {
        st.run_full_battle(
            &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
            &d,
        );
        ladder.push(st.sides[0].pokemon[0].fury_cutter.map(|(m, _)| m));
    }
    assert_eq!(
        ladder,
        vec![Some(1), Some(2), Some(4), Some(8), Some(16)],
        "BP3: the multiplier doubles per consecutive use and caps at 16 (bp 160)"
    );

    // ONE non-Fury-Cutter turn lapses the duration-2 volatile → back to multiplier 1.
    let mut b2 = Battle::start_with_switchins(&opts_cg(user, foe, "3,3,3,3"), &d).expect("start");
    let st2 = b2.state_mut().expect("state");
    st2.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(st2.sides[0].pokemon[0].fury_cutter.map(|(m, _)| m), Some(1));
    st2.run_full_battle(&[ScriptDecision::both(Choice::Move(1), Choice::Move(0))], &d);
    st2.run_full_battle(&[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert_eq!(
        st2.sides[0].pokemon[0].fury_cutter.map(|(m, _)| m),
        Some(1),
        "BP3: one skipped turn lapses the duration-2 volatile, so the ladder RESTARTS at 1"
    );
}

/// BP4 — DREAM EATER's `onTryImmunity`: the target must be ASLEEP (and un-subbed).
#[test]
fn dream_eater_is_immune_against_an_awake_target() {
    let d = dex();
    let user = "Mew||none|Synchronize|spore,dreameater,splash|Hardy|85,85,85,85,85,85|N||||";
    let foe = "Snorlax||none|Sturdy|splash|Hardy|85,85,85,85,85,85|M||||";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "3,3,3,3"), &d).expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // AWAKE → immune
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Spore
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // ASLEEP → connects
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let joined = raw.join("\n");
    assert!(
        raw.iter().any(|l| l.as_str() == "|-immune|p2a: Snorlax"),
        "BP4: Dream Eater into an AWAKE target is immune. got:\n{joined}"
    );
    assert!(
        raw.iter().any(|l| l.contains("|-status|p2a: Snorlax|slp")),
        "BP4: the foe must then actually be asleep, else the second half is vacuous. \
         got:\n{joined}"
    );
    assert!(
        st.sides[1].pokemon[0].hp < st.sides[1].pokemon[0].maxhp,
        "BP4: and Dream Eater then CONNECTS against the sleeping target. got:\n{joined}"
    );
}

/// BP5 — FALSE SWIPE's `onDamage` clamp leaves the target on exactly 1 HP.
///
/// It is NOT a base-power change: the probe shows the move computing **839** damage into a
/// 17-HP target and the target ending at 1.
#[test]
fn false_swipe_never_kos_and_leaves_exactly_one_hp() {
    let d = dex();
    let user = "Mew||none|Synchronize|falseswipe,splash|Hardy|85,85,85,85,85,85|N||||";
    // A level-5 Magikarp: False Swipe out-damages its whole HP bar many times over.
    let foe = "Magikarp||none|Swift Swim|splash|Hardy|0,0,0,0,0,0|M|0,0,0,0,0,0||5|";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "3,3,3,3"), &d).expect("start");
    let st = b.state_mut().expect("state");
    st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );
    // NON-VACUITY: the move must genuinely out-damage the target, else the clamp is untested.
    assert!(
        st.sides[1].pokemon[0].maxhp < 30,
        "BP5: the target must be frail enough that False Swipe WOULD KO it"
    );
    assert_eq!(
        st.sides[1].pokemon[0].hp, 1,
        "BP5: False Swipe clamps to exactly 1 HP and never faints the target"
    );
    assert!(!st.sides[1].pokemon[0].fainted, "BP5: and it is not fainted");
}

/// UP1 — UPROAR's LOCK: the duration is drawn ONCE on the cast, PP is spent ONCE, the user is
/// locked to Uproar (a switch is rejected), and the residual emits `[upkeep]` each live turn
/// then `-end` on expiry — with NO end-of-lock confusion. Ground truth
/// `harness/probe_uproar.js`.
#[test]
fn uproar_locks_the_user_for_its_rolled_duration_and_spends_pp_once() {
    let d = dex();
    let user = "Exploud||Leftovers|Soundproof|uproar,splash|Hardy|85,85,85,85,85,85|M||||]Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "44446,15321,46848,55374"), &d)
        .expect("start");
    let st = b.state_mut().expect("state");
    let pp0 = st.sides[0].pokemon[0].pp_of(0);
    let (_o, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // CAST
            // While locked the request offers ONLY Uproar and is trapped, so a SWITCH is
            // rejected draw-free and a `move 1` maps back to the locked slot.
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let joined = raw.join("\n");

    assert!(
        raw.iter().any(|l| l.as_str() == "|-start|p1a: Exploud|Uproar"),
        "UP1: the cast emits -start. got:\n{joined}"
    );
    // The rejected switch: the uproarer is still active right after it.
    assert_eq!(
        st.sides[0].active, 0,
        "UP1: a locked uproarer must NOT be able to switch out"
    );
    // PP is spent ONCE PER CAST — every CONTINUING turn is a lockedmove and spends none.
    // ⚠️ The duration is a random(2,6), so this fixed script can OUTLIVE the lock and start a
    // SECOND uproar, which legitimately spends PP again. Assert the real invariant
    // (spent == casts), not a fixed −1, and guard that the lock actually made the user act
    // more often than it cast — otherwise the claim is vacuous.
    let casts = raw
        .iter()
        .filter(|l| l.as_str() == "|-start|p1a: Exploud|Uproar")
        .count();
    let uses = raw
        .iter()
        .filter(|l| l.starts_with("|move|p1a: Exploud|Uproar"))
        .count();
    assert!(
        uses > casts,
        "UP1: the lock must make the user act MORE often than it CAST (uses {uses}, casts {casts})"
    );
    assert_eq!(
        st.sides[0].pokemon[0].pp_of(0) as usize,
        pp0 as usize - casts,
        "UP1: PP is deducted once per CAST ({casts}), not once per USE ({uses}). got:\n{joined}"
    );
    // The residual ticks with `[upkeep]` while live, then `-end` exactly once.
    let upkeeps = raw
        .iter()
        .filter(|l| l.as_str() == "|-start|p1a: Exploud|Uproar|[upkeep]")
        .count();
    assert!(upkeeps >= 1, "UP1: at least one live [upkeep] tick. got:\n{joined}");
    assert!(
        raw.iter().any(|l| l.as_str() == "|-end|p1a: Exploud|Uproar"),
        "UP1: the lock expires with an -end on the tick that reaches 0. got:\n{joined}"
    );
    // NO end-of-lock confusion (unlike outrage/thrash).
    assert!(
        !raw.iter().any(|l| l.contains("|confusion")),
        "UP1: the lock ends WITHOUT confusion. got:\n{joined}"
    );
    // (the script deliberately outlives the first lock, so a SECOND may be live here — the
    // expiry assertions above already cover the -end)
}

/// UP2 — the FIELD-WIDE sleep block. Uproar's handler is `onAnySetStatus`, so ONE live uproar
/// blocks sleep for EVERY mon — and the `[msg]` attr appears only when the blocked mon IS the
/// uproarer. The sleep move still draws its accuracy roll; the sleep `random(2,6)` does not.
#[test]
fn uproar_blocks_sleep_field_wide_with_the_msg_attr_only_for_itself() {
    let d = dex();
    // p2 uproars; p1 then tries to Spore it (the UPROARER — `[msg]` form).
    let sleeper = "Butterfree||Leftovers|Synchronize|spore,splash|Hardy|85,85,85,85,85,85|M||||";
    let uproarer = "Exploud||Leftovers|Soundproof|uproar,splash|Hardy|85,85,85,85,85,85|M||||";
    let mut b =
        Battle::start_with_switchins(&opts_cg(sleeper, uproarer, "44446,15321,46848,55374"), &d)
            .expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // p2 casts Uproar
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // p1 Spores the uproarer
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let joined = raw.join("\n");

    // NON-VACUITY: the lock is genuinely live at the Spore.
    assert!(
        st.sides[1].pokemon[0].uproar.is_some(),
        "UP2: the uproar must still be live. got:\n{joined}"
    );
    assert!(
        raw.iter().any(|l| l.as_str() == "|-fail|p2a: Exploud|slp|[from] Uproar|[msg]"),
        "UP2: blocking the UPROARER itself carries the [msg] attr. got:\n{joined}"
    );
    assert!(
        !matches!(st.sides[1].pokemon[0].status, Some(pokesim::state::Status::Sleep(_))),
        "UP2: nothing may fall asleep. got:\n{joined}"
    );
}

/// UP3 — the WAKE. A LANDED uproar cures `slp` on BOTH ACTIVES (the move's own `onTryHit`),
/// DRAW-FREE, emitted as a bare `|-curestatus|<mon>|slp|[msg]`.
#[test]
fn a_landed_uproar_wakes_both_actives() {
    let d = dex();
    // ⚠️ The USER needs a CHIP move: Rest FAILS at full HP, so without damaging the foe
    // first the "foe is asleep" precondition never holds and the pin tests nothing.
    let user = "Exploud||Leftovers|Soundproof|uproar,seismictoss,splash\
|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Blissey||Leftovers|NaturalCure|rest,splash|Hardy|85,85,85,85,85,85|F||||";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "44446,15321,46848,55374"), &d)
        .expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[
            // Chip the foe so its Rest can heal, then let it Rest itself asleep.
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)),
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)),
            // Now uproar: the sleeping foe must WAKE.
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let joined = raw.join("\n");

    // NON-VACUITY: the foe really was asleep before the uproar.
    assert!(
        raw.iter().any(|l| l.contains("|-status|p2a: Blissey|slp")),
        "UP3: the foe must be asleep first, else this proves nothing. got:\n{joined}"
    );
    assert!(
        raw.iter().any(|l| l.as_str() == "|-curestatus|p2a: Blissey|slp|[msg]"),
        "UP3: a landed uproar wakes the sleeping foe. got:\n{joined}"
    );
    assert!(
        st.sides[1].pokemon[0].status.is_none(),
        "UP3: and the state agrees. got:\n{joined}"
    );
}

/// WB1 — WEATHER BALL: type + base power follow the EFFECTIVE weather, and a suppressor
/// reverts it fully. Ground truth `harness/probe_weatherball.js`.
#[test]
fn weather_ball_retypes_and_doubles_power_per_weather() {
    let d = dex();
    // Mew mirror: base 100 across the board, and Psychic is NEUTRAL to every candidate type
    // with no STAB — so realized damage is a pure function of base power.
    let user = "Mew||Leftovers|Synchronize|weatherball,splash,raindance,sunnyday,sandstorm,hail\
|Hardy|85,85,85,85,85,85|N||||";
    let foe = "Mew||Leftovers|Synchronize|splash|Hardy|85,85,85,85,85,85|N||||";
    let seed = "44446,15321,46848,55374";

    // (weather-setting slot, expected type). Slot 1 = splash (no weather).
    let cases: &[(Option<usize>, &str)] = &[
        (Some(2), "Water"),
        (Some(3), "Fire"),
        (Some(4), "Rock"),
        (Some(5), "Ice"),
    ];
    for (setter, want) in cases {
        let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        let (_o, lines) = st.run_full_battle_logged(
            &[
                ScriptDecision::both(Choice::Move(setter.unwrap()), Choice::Move(0)),
                ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Weather Ball
            ],
            &d,
        );
        let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
        let joined = raw.join("\n");
        // NON-VACUITY: the weather really is up.
        assert!(
            st.field.weather.is_some(),
            "WB1: the weather must be set for {want}. got:\n{joined}"
        );
        // Under weather the ball is bp 100 vs the dex row's 50, so it must out-damage the
        // no-weather control below; the TYPE is asserted through effectiveness lines being
        // absent (all four are neutral into Psychic) plus the damage step.
        assert!(
            raw.iter().any(|l| l.starts_with("|-damage|p2a: Mew|")),
            "WB1: the {want} Weather Ball must connect. got:\n{joined}"
        );
    }

    // The no-weather control: bp 50, so STRICTLY less damage than any weather case.
    let dmg_of = |setter: Option<usize>| -> u16 {
        let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        let first = setter.unwrap_or(1);
        st.run_full_battle(
            &[
                ScriptDecision::both(Choice::Move(first), Choice::Move(0)),
                ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ],
            &d,
        );
        st.sides[1].pokemon[0].maxhp - st.sides[1].pokemon[0].hp
    };
    let none = dmg_of(None);
    let rain = dmg_of(Some(2));
    assert!(
        rain > none,
        "WB1: bp 100 under rain must out-damage the bp-50 no-weather control ({rain} vs {none})"
    );
}

/// WB2 — the CATEGORY flips with the type, and this is the pin that can actually SEE it.
///
/// ⚠️ gen-3 splits physical/special BY TYPE, so sandstorm's Rock ball is PHYSICAL while
/// rain's Water ball is SPECIAL. On the natural measurement board — a Mew mirror, base 100
/// everywhere — Atk == SpA and Def == SpD, so the flip is INVISIBLE in the damage number and
/// a damage-only test proves nothing. COUNTER (which answers physical) and MIRROR COAT (which
/// answers special) are the discriminator that works regardless of stats: under sandstorm
/// Counter fires back and Mirror Coat does nothing; under rain the reverse. Probe-settled in
/// `harness/probe_weatherball2.js` section B.
#[test]
fn weather_ball_category_follows_the_type_seen_by_counter_and_mirror_coat() {
    let d = dex();
    let user = "Mew||Leftovers|Synchronize|weatherball,sandstorm,raindance\
|Hardy|85,85,85,85,85,85|N||||";
    // Blissey answers with Counter (physical) or Mirror Coat (special).
    let foe = "Blissey||Leftovers|NaturalCure|counter,mirrorcoat,splash\
|Hardy|85,85,85,85,85,85|F||||";
    let seed = "44446,15321,46848,55374";

    // (weather setter slot, answer slot) -> did the answer deal damage back?
    let answered = |setter: usize, answer: usize| -> bool {
        let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        st.run_full_battle(
            &[
                ScriptDecision::both(Choice::Move(setter), Choice::Move(2)),
                ScriptDecision::both(Choice::Move(0), Choice::Move(answer)),
            ],
            &d,
        );
        st.sides[0].pokemon[0].hp < st.sides[0].pokemon[0].maxhp
    };

    // SANDSTORM → Rock → PHYSICAL: Counter answers, Mirror Coat does not.
    assert!(
        answered(1, 0),
        "WB2: a sandstorm (Rock) Weather Ball is PHYSICAL — Counter must answer it"
    );
    assert!(
        !answered(1, 1),
        "WB2: a sandstorm (Rock) Weather Ball is PHYSICAL — Mirror Coat must NOT answer it"
    );
    // RAIN → Water → SPECIAL: the reverse.
    assert!(
        !answered(2, 0),
        "WB2: a rain (Water) Weather Ball is SPECIAL — Counter must NOT answer it"
    );
    assert!(
        answered(2, 1),
        "WB2: a rain (Water) Weather Ball is SPECIAL — Mirror Coat must answer it"
    );
}

/// WB3 — AIR LOCK / CLOUD NINE revert it FULLY, asserted CATEGORICALLY.
///
/// ⚠️ TWO EARLIER VERSIONS OF THIS PIN WERE VACUOUS, both mutation-proven, and the reason is
/// worth keeping: a DAMAGE-magnitude assertion cannot isolate this.
///   (1) The first compared a Mew control against a Psyduck suppressor — its inequality held
///       because Psyduck has different bulk, so it passed even reading the RAW weather.
///   (2) Same-species arms still passed, because rain's ×1.5 WATER damage boost is itself
///       suppressed by Cloud Nine: the damage fell for a reason that has nothing to do with
///       which type the ball took.
/// The discriminator that DOES isolate it is categorical. Under rain the ball is Water →
/// SPECIAL, so MIRROR COAT answers it; under rain PLUS a suppressor it reverts to Normal →
/// PHYSICAL, so COUNTER answers instead. No magnitude is involved, so no stat or weather
/// multiplier can fake it.
#[test]
fn weather_ball_is_reverted_by_a_weather_suppressor() {
    let d = dex();
    let user = "Mew||Leftovers|Synchronize|weatherball,raindance,splash\
|Hardy|85,85,85,85,85,85|N||||";
    // Same species + spread in both arms; ONLY the ability differs (Damp is an inert control).
    let plain = "Psyduck||Leftovers|Damp|counter,mirrorcoat,splash|Hardy|85,85,85,85,85,85|M||||";
    let cloud = "Psyduck||Leftovers|CloudNine|counter,mirrorcoat,splash\
|Hardy|85,85,85,85,85,85|M||||";
    let seed = "44446,15321,46848,55374";

    let answered = |foe: &str, answer: usize| -> bool {
        let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        st.run_full_battle(
            &[
                ScriptDecision::both(Choice::Move(1), Choice::Move(2)), // Rain Dance
                ScriptDecision::both(Choice::Move(0), Choice::Move(answer)),
            ],
            &d,
        );
        st.sides[0].pokemon[0].hp < st.sides[0].pokemon[0].maxhp
    };

    // UNSUPPRESSED rain: Water → SPECIAL → Mirror Coat answers, Counter does not.
    assert!(
        answered(plain, 1),
        "WB3: under rain the ball is Water/SPECIAL — Mirror Coat must answer it"
    );
    assert!(
        !answered(plain, 0),
        "WB3: under rain the ball is Water/SPECIAL — Counter must NOT answer it"
    );
    // SUPPRESSED: reverts to Normal → PHYSICAL → Counter answers, Mirror Coat does not.
    assert!(
        answered(cloud, 0),
        "WB3: under Cloud Nine the ball reverts to Normal/PHYSICAL — Counter must answer it"
    );
    assert!(
        !answered(cloud, 1),
        "WB3: under Cloud Nine the ball reverts to Normal/PHYSICAL — Mirror Coat must NOT"
    );
}

/// CV1 — CONVERSION: the list is the USER's LIVE slots minus its CURRENT types, duplicates
/// kept, and the single `random(n)` fires even at n == 1. Ground truth
/// `harness/probe_conversion.js`.
///
/// The pin's teeth are the CROSS-SEED spread: one seed cannot distinguish "samples the list"
/// from "always takes slot 0", so it asserts that different seeds reach DIFFERENT types and
/// that every reached type is a member of the expected set.
#[test]
fn conversion_samples_a_live_slot_type_the_user_does_not_have() {
    let d = dex();
    // Porygon2 is NORMAL; the three attacking slots are Ice / Electric / Ghost, so all three
    // are eligible and Conversion's own Normal slot is excluded (the user IS Normal).
    let user = "Porygon2||Leftovers|Trace|conversion,icebeam,thunderbolt,shadowball\
|Hardy|85,85,85,85,85,85|N||||";
    let foe = "Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||";

    let mut seen = std::collections::BTreeSet::new();
    for seed in ["9,9,9,9", "1,2,3,4", "42,42,42,42", "7,11,13,17", "31,29,23,19"] {
        let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        st.run_full_battle(
            &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
            &d,
        );
        let t = st.sides[0].pokemon[0]
            .types_override
            .clone()
            .unwrap_or_else(|| panic!("CV1: Conversion must set a type override (seed {seed})"));
        assert_eq!(t.len(), 1, "CV1: setType REPLACES the array with ONE type");
        let name = t[0].display_name().to_string();
        assert!(
            matches!(name.as_str(), "Ice" | "Electric" | "Ghost"),
            "CV1: picked {name}, which is not one of the three eligible slot types"
        );
        seen.insert(name);
    }
    assert!(
        seen.len() >= 2,
        "CV1: the pick must actually SAMPLE — every seed reached the same type {seen:?}, which \
         a hard-coded slot-0 implementation would also satisfy"
    );
}

/// CV2 — CONVERSION fails DRAW-FREE when no slot offers a new type, and the failure is the
/// `[still]` did-nothing form with a bare `-fail` on the USER.
///
/// This is also the CURSE-exclusion pin: Curse is excluded BY ID (not by type), so a Normal
/// user whose only non-Normal slot is Curse has an EMPTY list.
#[test]
fn conversion_with_no_eligible_slot_fails_draw_free() {
    let d = dex();
    // Every slot is Normal EXCEPT curse, which is excluded by id → the list is empty.
    let user =
        "Porygon2||Leftovers|Trace|conversion,curse,splash|Hardy|85,85,85,85,85,85|N||||";
    let foe = "Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||";
    let seed = "9,9,9,9";

    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let joined = raw.join("\n");
    let fail_seed = seed_str(&st.prng_seed());

    assert!(
        st.sides[0].pokemon[0].types_override.is_none(),
        "CV2: no type may change. got:\n{joined}"
    );
    assert!(
        raw.iter().any(|l| l.as_str() == "|move|p1a: Porygon2|Conversion||[still]"),
        "CV2: the failure takes the [still] did-nothing form. got:\n{joined}"
    );
    assert!(
        raw.iter().any(|l| l.as_str() == "|-fail|p1a: Porygon2"),
        "CV2: with a BARE -fail on the USER. got:\n{joined}"
    );

    // DRAW-FREE: a Splash control from the same start must land on the SAME seed.
    let mut b2 = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
    let st2 = b2.state_mut().expect("state");
    st2.run_full_battle(
        &[ScriptDecision::both(Choice::Move(2), Choice::Move(0))],
        &d,
    );
    assert_eq!(
        fail_seed,
        seed_str(&st2.prng_seed()),
        "CV2: a FAILED Conversion draws nothing — it must match a Splash control exactly"
    );
}

/// CV3 — CONVERSION 2 reads the TARGET's last-used move and picks a type that RESISTS it,
/// from the TYPEDEX FILE's key order.
///
/// ⚠️ THE ORDER IS THE WHOLE PIN. The single `random(n)` indexes a list built in
/// `Dex.mod('gen3').types.names()` order — neither alphabetical nor the port's `Type` enum
/// order — so a correct-looking implementation that iterates the enum picks a DIFFERENT type
/// from the identical roll. Against a Normal last-move the eligible set is {Ghost, Steel,
/// Rock} and the probe's captured pick at these seeds is Rock; an enum-ordered list would
/// yield Rock only by coincidence.
#[test]
fn conversion_2_picks_a_resisting_type_in_the_typedex_order() {
    let d = dex();
    let user = "Shuckle||Leftovers|Sturdy|conversion2,splash|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Snorlax||Leftovers|Immunity|tackle,splash|Hardy|85,85,85,85,85,85|M||||";
    let seed = "9,9,9,9";

    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // foe Tackles (Normal)
            // The foe Tackles AGAIN so its lastMoveUsed is still `tackle` when the assertion
            // reads it after the battle — with Splash here the guard would read turn 2's move
            // and fail while the mechanic was fine.
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Conversion 2
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let joined = raw.join("\n");

    // NON-VACUITY: the foe really did record a Normal last move.
    assert_eq!(
        st.sides[1].pokemon[0].last_move_used.as_deref(),
        Some("tackle"),
        "CV3: the target must have a lastMoveUsed for this to test anything"
    );
    let t = st.sides[0].pokemon[0]
        .types_override
        .clone()
        .unwrap_or_else(|| panic!("CV3: Conversion 2 must set a type. got:\n{joined}"));
    let name = t[0].display_name();
    assert!(
        matches!(name, "Ghost" | "Steel" | "Rock"),
        "CV3: {name} does not resist Normal — the eligible set is Ghost/Steel/Rock"
    );
    // NOTE: the exact PICK is deliberately NOT asserted here. The probe's seeds are RAW
    // `>start` seeds and the sim spends draws on its turn-0 construction window, which the
    // port's draw-free `start_with_switchins` does not run — so the two rolls are not
    // comparable and an equality here would be a methodology error, not a gate (round 29).
    // The ORDER, which is what the single `random(n)` actually indexes, is pinned exactly and
    // seed-free by `conversion2_candidate_order_is_the_typedex_order` below.
}

/// CV3b — the ORDER pin, and the reason it is a pure-function test.
///
/// Conversion 2's one `random(n)` indexes a list built in `Dex.mod('gen3').types.names()`
/// order. The eligible SET is order-independent, so no battle-level assertion can
/// discriminate a correctly-ordered list from an enum-ordered one without aligning seeds to
/// the sim's post-construction state (the round-29 trap). Asserting the list sidesteps that.
///
/// The expected table is the SIM'S OWN computation, dumped verbatim from `Dex.mod('gen3')`:
/// `types.names().filter(t => types.get(t).damageTaken[att] in {2, 3})`. All 17 attacking
/// types are pinned, not a sample — the probe's battles only ever exercised Normal and Water,
/// and a three-case version of this test was written with a WRONG Electric row that the port
/// correctly contradicted. Deriving the whole table from the oracle is both stronger and the
/// only way to avoid "fixing" the test to match the code.
#[test]
fn conversion2_candidate_order_is_the_typedex_order() {
    use pokesim::dex::Type;
    let d = dex();
    let expected: &[(&str, &[&str])] = &[
        ("Electric", &["Electric", "Grass", "Dragon", "Ground"]),
        ("Ghost", &["Steel", "Dark", "Normal"]),
        ("Grass", &["Grass", "Steel", "Bug", "Dragon", "Fire", "Flying", "Poison"]),
        ("Steel", &["Electric", "Steel", "Fire", "Water"]),
        ("Dark", &["Steel", "Dark", "Fighting"]),
        ("Bug", &["Ghost", "Steel", "Fighting", "Fire", "Flying", "Poison"]),
        ("Dragon", &["Steel"]),
        ("Fighting", &["Ghost", "Bug", "Flying", "Poison", "Psychic"]),
        ("Fire", &["Dragon", "Fire", "Rock", "Water"]),
        ("Flying", &["Electric", "Steel", "Rock"]),
        ("Ground", &["Grass", "Bug", "Flying"]),
        ("Ice", &["Steel", "Fire", "Ice", "Water"]),
        ("Normal", &["Ghost", "Steel", "Rock"]),
        ("Poison", &["Ghost", "Steel", "Ground", "Poison", "Rock"]),
        ("Psychic", &["Steel", "Dark", "Psychic"]),
        ("Rock", &["Steel", "Fighting", "Ground"]),
        ("Water", &["Grass", "Dragon", "Water"]),
    ];
    for (att, want) in expected {
        let t = Type::from_name(att).expect("known type");
        let got: Vec<&str> = pokesim::turn::conversion2_candidates(t, &d)
            .into_iter()
            .map(|x| x.display_name())
            .collect();
        assert_eq!(
            got, *want,
            "CV3b: the {att} candidate list must match the sim's typedex order exactly"
        );
    }
}

/// CV4 — CONVERSION 2 fails DRAW-FREE against a target that has not moved, and `lastMoveUsed`
/// RESETS on switch-out (probe C8) — so a freshly-entered target also fails.
#[test]
fn conversion_2_fails_draw_free_with_no_target_last_move() {
    let d = dex();
    let user = "Shuckle||Leftovers|Sturdy|conversion2,splash|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Snorlax||Leftovers|Immunity|tackle,splash|Hardy|85,85,85,85,85,85|M||||]Miltank||Leftovers|Scrappy|splash|Hardy|85,85,85,85,85,85|F||||";
    let seed = "9,9,9,9";

    // Turn 1 the foe Tackles (so Snorlax HAS a lastMoveUsed), turn 2 it switches OUT
    // (clearing Snorlax's), turn 3 it switches BACK IN and Conversion 2 must fail — proving
    // the RESET, not merely an unmoved target.
    //
    // ⚠️ The turn-3 action must be a SWITCH, not a move: switches resolve at order 103 and
    // moves at 200, so the entrant is in place before Conversion 2 runs AND has not acted. A
    // move there would let the much faster foe set a fresh lastMoveUsed first and the pin
    // would silently test nothing.
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)),
            ScriptDecision::both(Choice::Move(0), Choice::Switch(1)),
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let joined = raw.join("\n");

    assert!(
        st.sides[0].pokemon[0].types_override.is_none(),
        "CV4: nothing may change type. got:\n{joined}"
    );
    assert!(
        raw.iter().any(|l| l.as_str() == "|move|p1a: Shuckle|Conversion 2||[still]"),
        "CV4: the [still] did-nothing form. got:\n{joined}"
    );
}

/// SG1 — SAFEGUARD (`gen3_safeguard_v1`): the cast, the re-cast fail, the expiry, and the
/// status BLOCK. Ground truth `harness/probe_safeguard.js`.
///
/// Every branch here is DRAW-FREE — the move is never-miss and the FIXED duration 5 is a
/// deterministic `durationCallback` — so the pin asserts the exact emission forms, which is
/// what a draw-free mechanic can actually be wrong about. Note the `-sidestart`/`-sideend`
/// effect name is BARE `Safeguard` (Light Screen uses a `move: ` prefix; this does not), and
/// the re-cast fail lands on the USER, not the side.
#[test]
fn safeguard_cast_blocks_status_then_expires_on_the_fifth_turn() {
    let d = dex();
    let user = "Snorlax||Leftovers|Immunity|safeguard,splash|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Miltank||Leftovers|Scrappy|thunderwave,splash|Hardy|85,85,85,85,85,85|F||||";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "44446,15321,46848,55374"), &d)
        .expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // T1 cast
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // T2 RE-cast → fail
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // T3 foe Thunder Wave → blocked
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // T4 idle
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // T5 idle → EXPIRY
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // T6 Thunder Wave now LANDS
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let joined = raw.join("\n");

    assert!(
        raw.iter().any(|l| l.as_str() == "|-sidestart|p1: P1|Safeguard"),
        "SG1: the cast emits the BARE-named -sidestart. got:\n{joined}"
    );
    // The re-cast did NOTHING → the [still] retro-edit plus a BARE -fail on the USER.
    assert!(
        raw.iter().any(|l| l.as_str() == "|move|p1a: Snorlax|Safeguard||[still]"),
        "SG1: a re-cast while active takes the [still] did-nothing form. got:\n{joined}"
    );
    assert_eq!(
        raw.iter().filter(|l| l.as_str() == "|-fail|p1a: Snorlax").count(),
        1,
        "SG1: exactly one bare -fail on the USER for the re-cast. got:\n{joined}"
    );
    // The BLOCK: the attempt keeps its own accuracy roll, then reports on the TARGET MON.
    assert!(
        raw.iter().any(|l| l.as_str() == "|-activate|p1a: Snorlax|move: Safeguard"),
        "SG1: a blocked Thunder Wave activates Safeguard on the TARGET. got:\n{joined}"
    );
    // EXPIRY at the end of the FIFTH turn from the cast, and the block really does lift.
    let sideend = raw
        .iter()
        .position(|l| l.as_str() == "|-sideend|p1: P1|Safeguard")
        .unwrap_or_else(|| panic!("SG1: no -sideend emitted. got:\n{joined}"));
    let par = raw
        .iter()
        .position(|l| l.as_str() == "|-status|p1a: Snorlax|par")
        .unwrap_or_else(|| panic!("SG1: the post-expiry Thunder Wave must land. got:\n{joined}"));
    assert!(
        sideend < par,
        "SG1: the expiry must PRECEDE the landing Thunder Wave. got:\n{joined}"
    );
    assert_eq!(st.sides[0].safeguard, 0, "SG1: the counter is spent");
}

/// SG2 — the BOTH-SIDES residual TIE, the one fact a single-side test cannot see.
///
/// Safeguard sits at side-residual order 4 with the same subOrder/speed as its screen
/// siblings, so the SAME condition on the OTHER side is the only thing that ties it: two
/// live Safeguards make the residual handler-sort draw ONE extra `random(0,2)`. Measured on
/// a mirror in the probe as 8 draws/turn → 9. This is the `double_screen…` sibling, and the
/// assertion is the same shape: capture the two-safeguard seed and require it to DIFFER from
/// a one-safeguard control run from the identical start.
#[test]
fn both_sides_safeguard_ties_the_side_residual_and_draws_the_shuffle() {
    let d = dex();
    // A speed-TIED mirror: identical species + spread, so the residual handlers tie on speed
    // and the ONLY difference between the two arms is the second Safeguard.
    let a = "Snorlax||Leftovers|Immunity|safeguard,splash|Hardy|85,85,85,85,85,85|M||||";
    let seed = "44446,15321,46848,55374";

    let run = |p2_first: Choice| -> String {
        let mut b = Battle::start_with_switchins(&opts_cg(a, a, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        st.run_full_battle(
            &[
                ScriptDecision::both(Choice::Move(0), p2_first),        // p1 casts; p2 maybe
                ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // both idle
            ],
            &d,
        );
        seed_str(&st.prng_seed())
    };
    let one = run(Choice::Move(1)); // p2 Splashes → ONE safeguard live
    let two = run(Choice::Move(0)); // p2 casts too → BOTH live → the tie

    // NON-VACUITY: the control arm really is one-sided.
    {
        let mut b = Battle::start_with_switchins(&opts_cg(a, a, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        st.run_full_battle(
            &[ScriptDecision::both(Choice::Move(0), Choice::Move(1))],
            &d,
        );
        assert!(
            st.sides[0].safeguard > 0 && st.sides[1].safeguard == 0,
            "SG2: the control must have exactly ONE side warded"
        );
    }
    assert_ne!(
        one, two,
        "SG2: two live Safeguards must TIE at side-residual order 4 and draw one extra \
         shuffle — the seeds cannot match a one-sided control"
    );
}

/// SG3 — REST is not blocked by your OWN Safeguard (the sim's `onSetStatus` returns early
/// when `target === source`). A model that blocked every status on a warded side would make
/// Safeguard shut off its own team's Rest.
///
/// ⚠️ WHAT THIS PIN ACTUALLY GUARDS, stated precisely because the obvious reading is wrong:
/// `run_rest` does not call `try_set_status_impl` at all — it inlines its own status path —
/// so the `source != target` clause there is UNREACHABLE and deleting it fails NOTHING.
/// Mutation-established: what this test catches is a safeguard check being added INSIDE
/// `run_rest`, which is the realistic way someone breaks this.
#[test]
fn safeguard_does_not_block_your_own_rest() {
    let d = dex();
    let user = "Snorlax||Leftovers|Immunity|safeguard,rest,splash|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Miltank||Leftovers|Scrappy|seismictoss,splash|Hardy|85,85,85,85,85,85|F||||";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "44446,15321,46848,55374"), &d)
        .expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // chip us so Rest can heal
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // ward the side
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // REST under our own ward
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let joined = raw.join("\n");

    // NON-VACUITY: the ward is genuinely up when Rest resolves.
    assert!(st.sides[0].safeguard > 0, "SG3: our own Safeguard must be live");
    assert!(
        raw.iter().any(|l| l.as_str() == "|-status|p1a: Snorlax|slp|[from] move: Rest"),
        "SG3: Rest must still sleep its user through our own Safeguard. got:\n{joined}"
    );
    assert_eq!(
        st.sides[0].pokemon[0].hp, st.sides[0].pokemon[0].maxhp,
        "SG3: and it must still full-heal"
    );
    assert!(
        !raw.iter().any(|l| l.contains("move: Safeguard")),
        "SG3: a self-inflicted status emits NO Safeguard activate. got:\n{joined}"
    );
}

/// SG4 — YAWN vs SAFEGUARD: the CAST is blocked, the pending RESOLVE is EXEMPT.
///
/// gen-3 safeguard's `onSetStatus` opens `if (effect.id === 'yawn') return;` — the exemption
/// keys on the EFFECT, not the source — so these two are deliberately OPPOSITE. A model that
/// treated Safeguard as a blanket sleep ward would get the resolve wrong; one that forgot the
/// ward entirely would get the cast wrong. Ground truth `harness/probe_safeguard.js`.
#[test]
fn safeguard_blocks_a_yawn_cast_but_not_a_pending_yawn_resolve() {
    let d = dex();
    let foe = "Miltank||Leftovers|Scrappy|yawn,splash|Hardy|85,85,85,85,85,85|F||||";
    let user = "Snorlax||Leftovers|Immunity|safeguard,splash|Hardy|85,85,85,85,85,85|M||||";
    let seed = "44446,15321,46848,55374";

    // (a) CAST into a live ward → blocked, no volatile, no sleep.
    {
        let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        let (_o, lines) = st.run_full_battle_logged(
            &[
                ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // ward up
                ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Yawn into it
                ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // would-be resolve turn
            ],
            &d,
        );
        let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
        let joined = raw.join("\n");
        assert!(st.sides[0].safeguard > 0, "SG4a: the ward must be live");
        assert!(
            raw.iter().any(|l| l.as_str() == "|-activate|p1a: Snorlax|move: Safeguard"),
            "SG4a: a blocked Yawn CAST activates Safeguard. got:\n{joined}"
        );
        assert!(
            st.sides[0].pokemon[0].yawn.is_none(),
            "SG4a: no yawn volatile is laid"
        );
        assert!(
            !raw.iter().any(|l| l.contains("|slp")),
            "SG4a: and nothing ever falls asleep. got:\n{joined}"
        );
    }

    // (b) Yawn lands FIRST, the ward goes up after → the PENDING resolve sleeps through it.
    {
        let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        let (_o, lines) = st.run_full_battle_logged(
            &[
                ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Yawn lands (no ward yet)
                ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // NOW ward up; yawn resolves
            ],
            &d,
        );
        let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
        let joined = raw.join("\n");
        // NON-VACUITY: the ward really was live when the resolve fired.
        assert!(
            st.sides[0].safeguard > 0,
            "SG4b: the ward must be up at the resolve, else this proves nothing"
        );
        assert!(
            matches!(st.sides[0].pokemon[0].status, Some(pokesim::state::Status::Sleep(_))),
            "SG4b: a PENDING yawn resolves THROUGH the ward. got:\n{joined}"
        );
    }
}

/// FO1 — FAKE OUT (`gen3_fakeout_v1`). The first-turn gate counts MOVE ACTIONS, not turns.
///
/// Ground truth `harness/probe_fakeout.js`. The failure form is UNIQUE in this engine: the
/// announce plus a `|-hint|`, with NO `[still]` attr and NO `-fail`, and `once = false` so a
/// repeat emits one hint EVERY time. Zero draws on the block, but PP is already paid.
///
/// THE DISTINCTION THAT MATTERS: a turn spent CANT-ed still BURNS the gate (the sim increments
/// `activeMoveActions` at the top of `runMove`, before BeforeMove), while a turn whose action was
/// CANCELLED never reaches `runMove` and does not. An `active_turns` implementation passes the
/// obvious cases and is silently wrong on that one — which is why the counter is its own field.
#[test]
fn fake_out_works_on_the_first_action_and_is_gated_after() {
    let d = dex();
    let user = "Persian||Leftovers|Limber|fakeout,splash|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "44446,15321,46848,55374"), &d)
        .expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // T1 Fake Out — WORKS
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // T2 Fake Out — GATED
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let hint = "|-hint|Fake Out only works on your first turn out.";

    // T1 landed: it DAMAGED and flinched. Asserted on the LOG, not the post-battle HP —
    // Leftovers heals the target back to full by the end, so a residual HP read proves nothing.
    let t1: Vec<&String> = raw.iter().take_while(|l| l.as_str() != hint).collect();
    assert!(
        t1.iter().any(|l| l.starts_with("|-damage|p2a: Snorlax|")),
        "FO1: the first-action Fake Out must land and damage. got:\n{}",
        raw.join("\n")
    );
    assert!(
        t1.iter().any(|l| l.as_str() == "|cant|p2a: Snorlax|flinch"),
        "FO1: the first-action Fake Out flinches (a chance-100 secondary). got:\n{}",
        raw.join("\n")
    );
    // T2 blocked, in the unique hint form.
    assert_eq!(
        raw.iter().filter(|l| l.as_str() == hint).count(),
        1,
        "FO1: the SECOND use is gated and emits exactly one hint. got:\n{}",
        raw.join("\n")
    );
    // The gate form carries NEITHER `[still]` NOR a `-fail` — unlike every other failing move.
    let after: Vec<&String> = raw
        .iter()
        .skip_while(|l| l.as_str() != hint)
        .collect();
    assert!(
        !raw.iter().any(|l| l.contains("Fake Out") && l.contains("[still]")),
        "FO1: the gate does NOT use the `[still]` did-nothing form. got:\n{}",
        raw.join("\n")
    );
    assert!(
        !after.iter().any(|l| l.starts_with("|-fail|p1a: Persian")),
        "FO1: the gate emits NO -fail. got:\n{}",
        raw.join("\n")
    );
    assert_eq!(
        st.sides[0].pokemon[0].active_move_actions, 2,
        "FO1: the counter tracks MOVE ACTIONS — two runs, two increments"
    );
}

/// FO2 — the CANCELLED-vs-CANT distinction that makes `active_turns` silently WRONG.
///
/// gen-3 faint-cancels-all removes a still-queued move when a mon faints
/// (`cancel_active_actions`), and a cancelled action never reaches `runMove` — so it does
/// NOT bump `activeMoveActions`. A turn spent ACTIVE but cancelled therefore does NOT burn
/// the Fake Out gate, while an implementation keyed on `active_turns` would count it and
/// wrongly refuse the punch. This is the ONLY reachable board where the two models differ:
/// Fake Out is priority +1, so its user can never be pre-empted while USING it — the
/// cancelled turn has to be one where the user queued something ELSE.
///
/// T1: Snorlax queues Splash; the faster Electrode Self-Destructs, faints, and cancels it.
/// T2 (after p2's forced replacement): Fake Out is the user's FIRST move action, so it LANDS.
#[test]
fn fake_out_survives_a_turn_whose_action_was_cancelled_by_a_faint() {
    let d = dex();
    let user = "Snorlax||Leftovers|Immunity|fakeout,splash|Hardy|85,85,85,85,85,85|M||||";
    let foe = "Electrode||Leftovers|Static|selfdestruct,splash|Hardy|85,85,85,85,85,85|M||||]Pikachu||Leftovers|Static|splash|Hardy|85,85,85,85,85,85|M||||";
    let mut b = Battle::start_with_switchins(&opts_cg(user, foe, "44446,15321,46848,55374"), &d)
        .expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // T1 Splash vs Self-Destruct
            ScriptDecision::one(1, Choice::Switch(1)),              // p2 replaces with Pikachu
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // T2 Fake Out — must LAND
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();

    // NON-VACUITY: the cancel must actually have happened — Electrode fainted and Snorlax's
    // Splash never resolved. Without this the test could pass on a board where Splash ran.
    assert!(
        raw.iter().any(|l| l.as_str() == "|faint|p2a: Electrode"),
        "FO2: the exploder must faint on T1. got:\n{}",
        raw.join("\n")
    );
    assert!(
        !raw.iter().any(|l| l.contains("|Splash|")),
        "FO2: Snorlax's queued Splash must be CANCELLED, never executed. got:\n{}",
        raw.join("\n")
    );
    // And `active_turns` must have advanced past 1, so an `active_turns`-keyed gate WOULD
    // have fired here — that is what makes the assertion below discriminating.
    assert!(
        st.sides[0].pokemon[0].active_turns > 1,
        "FO2: the user must have been active for >1 turn, else the two models agree"
    );

    assert!(
        !raw.iter().any(|l| l.contains("Fake Out only works on your first turn out.")),
        "FO2: a CANCELLED turn does not burn the gate — Fake Out must land. got:\n{}",
        raw.join("\n")
    );
    assert!(
        raw.iter().any(|l| l.starts_with("|-damage|p2a: Pikachu|")),
        "FO2: the ungated Fake Out must damage the replacement. got:\n{}",
        raw.join("\n")
    );
    assert_eq!(
        st.sides[0].pokemon[0].active_move_actions, 1,
        "FO2: exactly ONE move ACTION ran (the cancelled Splash does not count)"
    );
}

/// SS1 — SKILL SWAP (`gen3_skill_swap_v1`). Exchanges the two actives' abilities.
///
/// THE TRAP the pin exists for: the swapped-in abilities do **NOT** re-fire their switch-in
/// `onStart` (the sim gates those calls on `gen > 3`). The obvious "abilities changed, so re-run
/// switch-in" reading emits a phantom Intimidate `-unboost`, a phantom `-weather`, and — worst —
/// makes Trace consume `random_below(1)` where the sim draws nothing. Ground truth
/// `harness/probe_skillswap.js`.
///
/// So the pin uses an INTIMIDATE holder: a re-fired onStart would drop the foe's Attack a second
/// time, which is directly observable in the boost array.
#[test]
fn skill_swap_exchanges_abilities_without_refiring_on_start() {
    let d = dex();
    // Gyarados leads with Intimidate (one -1 atk on the foe at switch-in). Swapping it AWAY must
    // not fire it again, and swapping it ONTO the foe must not fire it either.
    let p1 = "Alakazam||Leftovers|Synchronize|skillswap,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Gyarados||Leftovers|Intimidate|splash,tackle|Hardy|85,85,85,85,85,85|M||||";
    let mut b = Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d)
        .expect("start");
    let st = b.state_mut().expect("state");
    let atk_before = st.sides[0].pokemon[0].boosts[1];
    let (_o, lines) = st.run_full_battle_logged(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();

    assert!(
        raw.iter().any(|l| l == "|-activate|p1a: Alakazam|Skill Swap|||[of] p2a: Gyarados"),
        "SS1: the gen<=4 form has TWO EMPTY fields and is the ONLY line emitted. got:\n{}",
        raw.join("\n")
    );
    // Scope to AFTER the move line: the log opens with Intimidate's own switch-in announce
    // (`|-ability|p2a: Gyarados|Intimidate|boost` + one `-unboost`), which is correct and must
    // not be mistaken for a Skill-Swap emission.
    let after: Vec<&String> = raw
        .iter()
        .skip_while(|l| !l.contains("|move|p1a: Alakazam|Skill Swap"))
        .collect();
    assert!(
        !after.iter().any(|l| l.contains("-endability") || l.contains("|-ability|")),
        "SS1: gen 3 emits NO -endability / -ability lines for the swap itself. got:\n{}",
        raw.join("\n")
    );
    // ⚠️ WATCH BOTH SIDES. After the swap the USER holds Intimidate, so a re-fired onStart drops
    // the *foe's* Attack, not the user's — an earlier draft counted only `p1a: Alakazam` unboosts
    // and therefore PASSED with the re-fire injected, i.e. it was vacuous. The correct assertion
    // is that the swap emits NO boost line at all: after the move line there must be no
    // `-unboost`/`-boost`/`-weather` from any re-fired switch-in handler.
    assert!(
        !after.iter().any(|l| l.contains("|-unboost|") || l.contains("|-boost|")
            || l.contains("|-weather|")),
        "SS1: the swap must fire NO switch-in onStart — any boost/weather line after the move \
         means `single_event_ability_start` was wrongly called. got:\n{}",
        raw.join("\n")
    );
    // THE LOAD-BEARING ONE: no second Intimidate.
    assert_eq!(
        st.sides[0].pokemon[0].boosts[1], atk_before,
        "SS1: swapping Intimidate must NOT re-fire its switch-in onStart — a second -unboost here \
         means `single_event_ability_start` was wrongly called on the swap"
    );
    // The abilities really did exchange.
    assert_eq!(pokesim::dex::to_id(&st.sides[0].pokemon[0].ability), "intimidate");
    assert_eq!(pokesim::dex::to_id(&st.sides[1].pokemon[0].ability), "synchronize");
}

/// SS2 — SKILL SWAP FAILS on Wonder Guard or a same-ability pair (`gen3_skill_swap_v1`).
/// Both take the `[still]` + bare `-fail|<USER>` form, draw-free.
#[test]
fn skill_swap_fails_on_wonder_guard_and_on_identical_abilities() {
    let d = dex();
    let run = |p1: &str, p2: &str| {
        let mut b = Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d)
            .expect("start");
        let st = b.state_mut().expect("state");
        let (_o, lines) = st.run_full_battle_logged(
            &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
            &d,
        );
        let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
        let failed = raw.iter().any(|l| l == "|-fail|p1a: Alakazam")
            && !raw.iter().any(|l| l.contains("Skill Swap|||[of]"));
        (failed, raw.join("\n"))
    };
    let user = "Alakazam||Leftovers|Synchronize|skillswap,splash|Hardy|85,85,85,85,85,85|M||||";
    let (wg_failed, wg_log) =
        run(user, "Shedinja||Leftovers|WonderGuard|splash|Hardy|85,85,85,85,85,85|M||||");
    assert!(wg_failed, "SS2: Wonder Guard is gen-3's only failskillswap. got:\n{wg_log}");
    let (same_failed, same_log) =
        run(user, "Espeon||Leftovers|Synchronize|splash|Hardy|85,85,85,85,85,85|M||||");
    assert!(same_failed, "SS2: an identical ability pair fails (gen<=5 rule). got:\n{same_log}");
}

/// RC1 — RECYCLE (`gen3_recycle_v1`). Restores the item the mon CONSUMED ITSELF.
///
/// THE DISCRIMINATOR IS WHICH PRIMITIVE REMOVED THE ITEM, not that the slot went empty: the sim's
/// `eatItem`/`useItem` set `lastItem`, `takeItem` does NOT. So a berry eaten at its own threshold
/// is restorable and a KNOCKED-OFF item is not — the intuitive "the item left, so remember it"
/// implementation gets the second case backwards. Ground truth `harness/probe_recycle.js`.
///
/// The pin drives BOTH arms so the distinction is load-bearing: an eaten Lum restores, a
/// knocked-off Leftovers does not.
#[test]
fn recycle_restores_an_eaten_berry_but_not_a_knocked_off_item() {
    let d = dex();
    // ARM A: the holder EATS its Lum Berry curing a self-inflicted status, then Recycles it back.
    // Rest puts it to sleep -> Lum cures + is EATEN (an `eatItem`, so lastItem is set).
    // Rest FAILS at full HP, so the foe chips first — then Rest sleeps the mon, the Lum cures
    // that sleep and is EATEN (an `eatItem`, which is what sets `last_item`).
    let a_user = "Snorlax||LumBerry|Immunity|recycle,rest,splash|Hardy|85,85,85,85,85,85|M||||";
    // Seismic Toss (Fighting, fixed 100) — Night Shade is GHOST and a Normal-type Snorlax is
    // IMMUNE to it, which silently left the mon at full HP and made Rest fail.
    let foe = "Gengar||Leftovers|Levitate|seismictoss,knockoff|Hardy|85,85,85,85,85,85|M||||";
    let mut ba = Battle::start_with_switchins(&opts_cg(a_user, foe, "44446,15321,46848,55374"), &d)
        .expect("start");
    let sa = ba.state_mut().expect("state");
    let (_o, la) = sa.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // foe Seismic Toss -> chip
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Rest -> Lum cures + EATEN
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Recycle
        ],
        &d,
    );
    let raw_a: Vec<String> = la.into_iter().map(|l| l.0).collect();
    assert!(
        raw_a.iter().any(|l| l.contains("|-item|p1a: Snorlax|Lum Berry|[from] move: Recycle")),
        "RC1-A: an EATEN berry must be restored with the `[from] move: Recycle` tag. got:\n{}",
        raw_a.join("\n")
    );
    assert_eq!(
        pokesim::dex::to_id(&sa.sides[0].pokemon[0].item), "lumberry",
        "RC1-A: the mon is holding the restored berry"
    );

    // ARM B: the foe KNOCKS OFF the item (a `takeItem`) -> Recycle must FAIL.
    let b_user = "Snorlax||Leftovers|Immunity|recycle,splash|Hardy|85,85,85,85,85,85|M||||";
    let mut bb = Battle::start_with_switchins(&opts_cg(b_user, foe, "44446,15321,46848,55374"), &d)
        .expect("start");
    let sb = bb.state_mut().expect("state");
    let (_o2, lb) = sb.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // foe Knock Off
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Recycle
        ],
        &d,
    );
    let raw_b: Vec<String> = lb.into_iter().map(|l| l.0).collect();
    assert!(
        raw_b.iter().any(|l| l.contains("Knock Off")),
        "RC1-B: the board must actually knock the item off. got:\n{}",
        raw_b.join("\n")
    );
    assert!(
        !raw_b.iter().any(|l| l.contains("[from] move: Recycle")),
        "RC1-B: a KNOCKED-OFF item is NOT recyclable (takeItem never sets lastItem). got:\n{}",
        raw_b.join("\n")
    );
    assert!(
        sb.sides[0].pokemon[0].item.is_empty(),
        "RC1-B: the mon stays itemless"
    );
}

/// ER1 — ERUPTION (`gen3_eruption_v1`). The HP-scaled BP twin of Water Spout: `bp =
/// max(floor(150 * hp / maxhp), 1)`. PROBE-SETTLED side by side in
/// `harness/probe_varbp_cluster.js` — the SAME resolved callback and dataBP 150, so at identical
/// hp the two derive the IDENTICAL base power. It differs only in type (Fire vs Water).
///
/// WRONG (pre-fix): `eruption` was a CONSTRUCTION fail-loud — one of the 16 silent-desync moves —
/// because it was absent from the Water Spout id-gate and would otherwise run at FLAT BP 150
/// regardless of the user's HP.
///
/// ⚠️ IT MUST FIRE FROM REDUCED HP. The first draft fired at FULL HP, where the derivation yields
/// exactly the dex BP 150 — so it passed with Eruption deleted from the gate, i.e. it was VACUOUS.
/// Full HP is precisely the one value at which this mechanism is invisible. The user now spends
/// Substitute first, whose cost is a deterministic floor(maxhp/4), putting it at 3/4 HP where the
/// derived BP is 112 and a flat-150 implementation is plainly distinguishable.
#[test]
fn eruption_scales_with_hp_exactly_like_water_spout() {
    let d = dex();
    // Snorlax is neutral to BOTH Fire and Water, and the Normal-type users get no STAB on either,
    // so the ONLY thing that can differ between the arms is the BP derivation.
    let foe = "Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||";
    let seed = "44446,15321,46848,55374";
    // T1 Substitute (costs exactly floor(maxhp/4)) -> T2 fire from 3/4 HP.
    let run = |mv: &str, chip: bool| {
        let user = format!(
            "Kangaskhan||Leftovers|EarlyBird|{mv},substitute|Hardy|85,85,85,85,85,85|M||||"
        );
        let mut b = Battle::start_with_switchins(&opts_cg(&user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        let mut script = Vec::new();
        if chip {
            script.push(ScriptDecision::both(Choice::Move(1), Choice::Move(0))); // Substitute
        }
        script.push(ScriptDecision::both(Choice::Move(0), Choice::Move(0)));     // the move
        let _ = st.run_full_battle(&script, &d);
        st.sides[1].pokemon[0].maxhp - st.sides[1].pokemon[0].hp                  // damage dealt
    };

    let dmg_full = run("eruption", false);
    let dmg_chipped = run("eruption", true);
    let ws_chipped = run("waterspout", true);

    assert!(dmg_full > 0 && dmg_chipped > 0, "ER1: both arms must deal damage");
    // THE LOAD-BEARING ASSERTION: the derivation must actually SCALE. At 3/4 HP the BP is 112,
    // not 150 — a flat-BP implementation deals the same damage as the full-HP arm and fails here.
    assert!(
        dmg_chipped < dmg_full,
        "ER1: Eruption from 3/4 HP must deal LESS than from full HP (bp 112 vs 150). Equal damage \
         means the HP scaling never ran — i.e. Eruption is not on the Water Spout gate. \
         full={dmg_full} chipped={dmg_chipped}"
    );
    // ...and it must scale by the SAME rule as its twin.
    assert_eq!(
        dmg_chipped, ws_chipped,
        "ER1: Eruption and Water Spout share one callback, so at equal HP against a neutral \
         target they must deal identical damage"
    );
}

/// TM1 — TORMENT (`gen3_torment_v1`). The cheapest move-restriction: Taunt minus the duration and
/// minus the execution-time cant. Ground truth `harness/probe_torment.js`.
///
/// Pins the three things a naive implementation gets wrong:
///   * the `-start` detail is the BARE `Torment` (Taunt renders `move: Taunt`);
///   * the restriction blocks the move used LAST and is read LIVE (so it MOVES as `last_move` does);
///   * a re-cast into an already-tormented foe takes the `[still]` + `-fail|<USER>` form and STILL
///     draws its accuracy roll.
/// It is PERMANENT — no duration, no residual handler. A phantom duration handler would join the
/// NO_ORDER/subOrder-2 protect/stall/flinch tie group and draw a shuffle the sim never draws.
#[test]
fn torment_blocks_the_last_used_move_and_is_permanent() {
    let d = dex();
    let p1 = "Gengar||Leftovers|Levitate|torment,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|tackle,splash|Hardy|85,85,85,85,85,85|M||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    // T1: the foe uses Tackle (slot 0) so it has a `last_move`; we Torment it.
    let (_o, lines) = st.run_full_battle_logged(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    assert!(
        raw.iter().any(|l| l == "|-start|p2a: Snorlax|Torment"),
        "TM1: the -start detail is the BARE `Torment`, not `move: Torment`. got:\n{}",
        raw.join("\n")
    );
    assert!(st.sides[1].pokemon[0].torment, "TM1: the volatile is set on the TARGET");
    // The slot the foe used last is now unusable; its OTHER slot still is.
    assert!(
        !st.sides[1].pokemon[0].move_usable(0, &d),
        "TM1: the LAST-USED slot is blocked"
    );
    assert!(
        st.sides[1].pokemon[0].move_usable(1, &d),
        "TM1: every other slot stays usable — torment blocks ONE move, not all"
    );
    // PERMANENT: no duration to tick, so it is still live many turns later.
    let _ = st.run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)),
        ],
        &d,
    );
    assert!(
        st.sides[1].pokemon[0].torment,
        "TM1: torment is PERMANENT — it must not tick away like Taunt"
    );
}

/// TM2 — TORMENT JOINS THE endTurn `DisableMove` TIE GROUP (`gen3_torment_v1`).
///
/// THE SILENT ONE, and the reason torment is not observation-only. Its `onDisableMove` handler
/// ties with taunt / disable / choicelock / encore on the same mon, so n handlers draw n-1
/// `random` calls per endTurn — one call BEFORE the Quick Claw. Omitting it is exactly RB1
/// (`gen3_encore_disable_move_shuffle_v1`): invisible to every state/HP assertion, surfacing only
/// as a downstream seed divergence.
///
/// ⚠️ THE ARMS MUST DIFFER ONLY IN `n`. The first draft of this test cast Torment in one arm and
/// Splash in the other, so the seeds differed because of the ACCURACY ROLL — it passed with the
/// torment term deleted from `disable_move_event_shuffle`, i.e. it was VACUOUS. Both arms now cast
/// Torment identically; the ONLY difference is the foe's item, which decides whether a choicelock
/// handler exists (n=2 vs n=1). Neither foe holds Leftovers, so no residual-heal handler joins the
/// separate residual tie group and confounds the comparison.
#[test]
fn torment_adds_the_disable_move_tie_shuffle_draw() {
    let d = dex();
    let user = "Gengar||Leftovers|Levitate|torment,splash|Hardy|85,85,85,85,85,85|M||||";
    let seed = "44446,15321,46848,55374";
    // A: a CHOICE-BAND foe acquires a choicelock handler on its first move -> torment makes n=2.
    let foe_locked = "Snorlax||ChoiceBand|Immunity|tackle,splash|Hardy|85,85,85,85,85,85|M||||";
    // B: the SAME foe with NO item -> no choicelock -> torment alone leaves n=1 (no tie, no draw).
    let foe_free = "Snorlax|||Immunity|tackle,splash|Hardy|85,85,85,85,85,85|M||||";

    let run = |foe: &str| {
        let mut b = Battle::start_with_switchins(&opts_cg(user, foe, seed), &d).expect("start");
        let st = b.state_mut().expect("state");
        let out = st.run_full_battle(
            &[
                ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // foe Tackles
                ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // p1 TORMENTS (both arms)
            ],
            &d,
        );
        (seed_str(&out.decisions[1].seed_after), st.sides[1].pokemon[0].torment)
    };
    let (seed_locked, torment_a) = run(foe_locked);
    let (seed_free, torment_b) = run(foe_free);

    assert!(torment_a && torment_b, "TM2: BOTH arms must actually land the torment");
    assert_ne!(
        seed_locked, seed_free,
        "TM2: the n=2 arm (torment + choicelock) must consume ONE MORE endTurn DisableMove \
         tie-shuffle draw than the n=1 arm (torment alone), so their post-turn seeds cannot \
         match. Equal seeds mean torment was never counted in `disable_move_event_shuffle`."
    );
}

/// DT1 — DOUBLE TEAM (`gen3_double_team_v1`). The SELF-evasion sibling of SA1's foe accuracy-drop,
/// and the SECOND stale-justification unlock: `_self_boosts` excluded accuracy/evasion for the same
/// reason `_stat_drop_boosts` did, and that reason was equally false. Double Team is a PURE
/// declarative self-boost — never-miss, DRAW-FREE, no volatileStatus — structurally Swords Dance
/// with a different stat, so it needed no engine work at all.
///
/// Ground truth `harness/probe_doubleteam_minimize.js`. That probe also settled why MINIMIZE stays
/// fail-loud: its gen-3 volatile is LIVE (it doubles any move carrying `flags.minimize` — in gen 3
/// exactly stomp/astonish/extrasensory/needlearm), which is real engine work at the late
/// ModifyDamage fold, and the x2 SURVIVES Haze while the evasion stage does not.
#[test]
fn double_team_raises_evasion_and_folds_into_the_accuracy_roll() {
    let d = dex();
    let p1 = "Ditto||Leftovers|Limber|doubleteam,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (_out, lines) = st.run_full_battle_logged(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    assert!(
        raw.iter().any(|l| l == "|-boost|p1a: Ditto|evasion|1"),
        "DT1: the cast raises evasion by one stage. got:\n{}",
        raw.join("\n")
    );
    // boosts[6] is the EVASION stage — the index the old data guard was afraid of, and the one
    // `speed.rs::effective_accuracy` already folds through the boostTable in the DIVIDE direction.
    assert_eq!(st.sides[0].pokemon[0].boosts[6], 1, "DT1: evasion lands in boosts[6]");
}

/// SA1 — SAND ATTACK (`gen3_sand_attack_v1`). The accuracy/evasion STAT-DROP unlock. This was
/// NOT an engine gap: `speed.rs::effective_accuracy` (`gen3_accuracy_pipeline_v1`) has folded
/// boosts[5]/boosts[6] through the gen-3 boostTable for a long time. It was a DATA gap — the
/// extractor's `_stat_drop_boosts` guard excluded accuracy/evasion, justified by a claim ("the
/// engine's evasion is not folded into the accuracy roll") that was FALSE by the time it mattered.
/// Relaxing the guard admits exactly four moves: sandattack / smokescreen / kinesis / flash.
///
/// Ground truth `harness/probe_sandattack.js`: ONE `randomChance(100,100)` accuracy draw (acc 100,
/// NOT never-miss), then a DRAW-FREE apply emitting `|-unboost|<target>|accuracy|1`. Sand Attack is
/// NOT type-immune on a Ground target (Status ⇒ ignoreImmunity).
///
/// WRONG (pre-fix): `statDropBoosts` was absent, so the stat-drop arm was skipped and the move fell
/// through to the unmodeled-status-move fail-loud panic.
#[test]
fn sand_attack_drops_accuracy_and_is_not_ground_immune() {
    let d = dex();
    let p1 = "Sandslash||Leftovers|SandVeil|sandattack,splash|Hardy|85,85,85,85,85,85|M||||";
    // Marowak is GROUND — a Ground target must still take the drop (Status ignores type immunity).
    let p2 = "Marowak||Leftovers|RockHead|splash|Hardy|85,85,85,85,85,85|M||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (_out, lines) = st.run_full_battle_logged(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    assert!(
        raw.iter().any(|l| l == "|-unboost|p2a: Marowak|accuracy|1"),
        "SA1: a Ground target still takes the accuracy drop. got:\n{}",
        raw.join("\n")
    );
    // boosts[5] is the ACCURACY stage — the index the old data guard was afraid of.
    assert_eq!(
        st.sides[1].pokemon[0].boosts[5], -1,
        "SA1: the accuracy stage lands in boosts[5]"
    );
}

/// CR1 — CONFUSE RAY (`gen3_confuse_ray_v1`). A VOLATILE-inflicting status move, so it lives
/// outside `modeled_status_move` (which maps only MAJOR statuses) and used to hit that path's
/// FAIL-LOUD panic. Ground truth `harness/probe_confuseray.js` (re-runnable):
///   plain hit       -> `|-start|<target>|confusion`, and the random(2,6) duration draw
///   already confused-> `|move|…|[still]` + `|-fail|<USER>`  (the USER, not the target), NO draw
/// The duration draw is the SHARED `add_confusion` path (Water Pulse & co), so this pins the
/// MOVE-LEVEL forms — the two emissions a secondary never produces.
#[test]
fn confuse_ray_starts_confusion_then_fails_on_an_already_confused_target() {
    let d = dex();
    let p1 = "Gengar||Leftovers|Levitate|confuseray,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (_out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Confuse Ray lands
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // again -> already confused
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let marks: Vec<&str> = raw
        .iter()
        .filter(|l| l.contains("Confuse Ray") || l.contains("confusion") || l.starts_with("|-fail|"))
        .map(|l| l.as_str())
        .collect();
    assert!(
        marks.iter().any(|l| *l == "|-start|p2a: Snorlax|confusion"),
        "CR1: the landing cast must emit -start|confusion. got:\n{}",
        raw.join("\n")
    );
    assert!(
        marks.iter().any(|l| l.contains("Confuse Ray") && l.contains("[still]")),
        "CR1: the second cast must carry the [still] attr. got:\n{}",
        raw.join("\n")
    );
    assert!(
        marks.iter().any(|l| *l == "|-fail|p1a: Gengar"),
        "CR1: the already-confused fail names the USER, not the target. got:\n{}",
        raw.join("\n")
    );
    // Exactly ONE -start: the second cast hit the already-confused gate, so it neither
    // re-applied the volatile nor took the random(2,6) duration draw. (Do NOT assert the
    // volatile still EXISTS here — it rolled a 2-turn duration and legitimately expires,
    // which is what the sim's `|-end|…|confusion` on turn 2 shows.)
    assert_eq!(
        marks.iter().filter(|l| **l == "|-start|p2a: Snorlax|confusion").count(),
        1,
        "CR1: the already-confused cast must NOT re-add the volatile. got:\n{}",
        raw.join("\n")
    );
}

/// PT8 — SUBSTITUTE by the victim frees the trap, SILENTLY
/// (`gen3_substitute_frees_partial_trap_v1`). The sibling of PT4, and the FORM differs: the
/// substitute condition's own `onStart` (moves.js:18380-18382) does an explicit
/// `this.add('-end', …, '[partiallytrapped]', '[silent]')` + `delete`, rather than a
/// `removeVolatile` — so this emits the **SILENT** end where Rapid Spin emits the bare one.
/// It sits BETWEEN `|-start|<user>|Substitute` and the directDamage `-damage` (the cost),
/// because `onStart` runs inside `addVolatile`, before the move's `onHit`. DRAW-FREE.
///
/// Living in `onStart` also fixes the scope: a FAILED Substitute never starts the volatile, so
/// it must NOT free the trap. (The already-subbed pairing is unreachable anyway — a sub BLOCKS
/// an incoming partial trap, and starting one frees an existing trap.)
///
/// WRONG (pre-fix): the port kept the trap and went on chipping `maxhp/16` every residual. It
/// hid well because Leftovers heals the SAME `maxhp/16`, so the HP sat still turn after turn and
/// only the total diverged — three fuzz repros across two gates (`ab_9_21` dec 88, `ab_14_8`
/// dec 48, `sbd_msxkl91p_b62` dec 53), each the port LOW by exactly one tick with the seed
/// matching. Ground truth + the stack trace that located it: `harness/probe_ptrap_substitute.js`.
#[test]
fn partial_trap_substitute_frees_it_with_the_silent_end() {
    let d = dex();
    let p1 = "Dragonite||Leftovers|InnerFocus|wrap,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|substitute,splash|Hardy|85,85,85,85,85,85|M||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (_out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Wrap lands on Snorlax
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Snorlax SUBSTITUTES
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // no further chip
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)), // …nor here
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let marks: Vec<&str> = raw
        .iter()
        .filter(|l| {
            l.contains("partiallytrapped") || l.contains("move: Wrap") || l.contains("Substitute")
        })
        .map(|l| l.as_str())
        .collect();
    assert_eq!(
        marks,
        vec![
            "|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Dragonite",
            "|-damage|p2a: Snorlax|452/482|[from] move: Wrap|[partiallytrapped]",
            "|move|p2a: Snorlax|Substitute|p2a: Snorlax",
            "|-start|p2a: Snorlax|Substitute",
            "|-end|p2a: Snorlax|Wrap|[partiallytrapped]|[silent]",
        ],
        "PT8: ONE chip, then the sub's -start followed by the SILENT partial-trap end, and NO \
         chip on any later turn. got:\n{}",
        raw.join("\n")
    );
    // THE ORDERING CLAIM, pinned on its own: the release happens inside `addVolatile`, so it
    // precedes the move's `onHit` directDamage — the very next line must be the sub's HP cost
    // (a BARE `-damage`, no `[from]`). Reversing the two emits would still satisfy `marks`.
    let end_at = raw
        .iter()
        .position(|l| l.contains("[partiallytrapped]") && l.contains("[silent]"))
        .expect("PT8: the silent end must be emitted");
    let next = raw.get(end_at + 1).map(String::as_str).unwrap_or("<eof>");
    assert!(
        next.starts_with("|-damage|p2a: Snorlax|") && !next.contains("[from]"),
        "PT8: the trap release must PRECEDE the substitute's HP cost (onStart runs inside \
         addVolatile, before onHit). next line was {next:?}\n{}",
        raw.join("\n")
    );
    assert!(!st.is_trapped(1, &d), "PT8: the substitute frees the victim");
}

/// PT7 — the endTurn `TrapPokemon` HANDLER-SORT TIE (`gen3_partial_trap_v1`). The trap-MOVE
/// `trapped` volatile (Block / Mean Look / Spider Web) and `partiallytrapped` BOTH carry an
/// `onTrapPokemon`, and both are **Conditions** ⇒ `subOrder` 2 (`sim/battle.ts:957`). Held
/// together on the SAME mon they have the same speed AND the same subOrder ⇒ the event's handler
/// list TIES ⇒ **ONE extra `random(0,2)` per endTurn** for as long as both are live. Exactly one,
/// not two: neither condition carries an `onMaybeTrapPokemon`, so the `MaybeTrapPokemon` event
/// still sees only the ability handlers.
///
/// WRONG (pre-fix): `trap_event_shuffles` counted ONLY the ability handlers (Arena Trap / Magnet
/// Pull) — correct while at most one trapping CONDITION could exist, which is exactly what
/// modelling the wrap family changed. FOUND BY the round-32 byte fuzz (`ab_fuzz --mode random
/// --protocol --master-seed 803324991`, repro `rmsde6xp4_ab_10_9`: an Onix that Blocks and then
/// Sand Tombs the same Wartortle → `kind=seed` at dec 26, one draw short). Draw model
/// probe-measured by `harness/probe_ptrap_trapevent.js`: block-only +0, sandtomb-only +0, BOTH
/// +1, arena-trap-only +0. Also pinned end-to-end by byte-fuzz corpus fixture
/// `72_block_plus_partial_trap_ties_the_trapevent_sort.txt` (the original repro).
#[test]
fn partial_trap_plus_block_ties_the_endturn_trap_event_sort() {
    let d = dex();
    let p1 = "Onix||BlackBelt|Sturdy|block,sandtomb,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Wartortle||Leftovers|Torrent|splash|Hardy|85,85,85,85,85,85|M||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Sand Tomb -> partiallytrapped
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Block -> ALSO `trapped`
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // idle: both live -> the tie
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)), // idle: still both live
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    // GUARDS: this pin is vacuous unless BOTH volatiles actually landed and are still live.
    assert!(
        raw.iter().any(|l| l.contains("|-activate|p2a: Wartortle|move: Sand Tomb")),
        "PT7: the Sand Tomb must land. got:\n{}",
        raw.join("\n")
    );
    assert!(
        raw.iter().any(|l| l == "|-activate|p2a: Wartortle|trapped"),
        "PT7: the Block must land. got:\n{}",
        raw.join("\n")
    );
    assert!(
        st.sides[1].pokemon[st.sides[1].active].partial_trap.is_some()
            && st.sides[1].pokemon[st.sides[1].active].trapped_by.is_some(),
        "PT7: BOTH trapping volatiles must still be live at the end (else no tie is exercised)"
    );
    // The SEED is the whole pin: dec1 onward each carry the extra tie-shuffle draw.
    for (i, want) in [
        "58424,44781,32445,22157", // Sand Tomb cast (the random(3,7)); only ONE condition yet
        "23935,41153,16141,50999", // Block lands -> the endTurn TrapPokemon sort now TIES
        "4841,46532,32342,29009",
        "13170,34781,40173,15195",
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(seed_str(&out.decisions[i].seed_after), *want, "PT7 dec{i} post-turn seed");
    }
}

/// PT6 — PRECEDENCE. On the SAME residual the duration hits 0 AND the trapper switches out.
/// Showdown's `fieldEvent('Residual')` decrements `handler.state.duration` and calls the
/// `end()` (→ `onEnd`) BEFORE it ever invokes `onResidual`, so the DURATION branch wins and the
/// **NON-silent** `|-end|…|[partiallytrapped]` is emitted — not the trapper-gone `[silent]`
/// form. Probe O settled this; a port that tested "trapper gone" first would emit `[silent]`
/// here and match everywhere else, which is exactly why it needs its own pin.
#[test]
fn partial_trap_duration_expiry_beats_the_trapper_gone_release() {
    let d = dex();
    let p1 = "Dragonite||Leftovers|InnerFocus|wrap,splash|Hardy|85,85,85,85,85,85|M||||]\
              Gengar||Leftovers|Levitate|splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|splash|Hardy|85,85,85,85,85,85|M||||]\
              Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "46121,4763,1072,30122"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),   // Wrap (random(3,7) == 3)
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),   // the 2nd (last) chip
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)), // trapper leaves AND dur -> 0
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let marks: Vec<&str> = raw
        .iter()
        .filter(|l| l.contains("partiallytrapped") || l.contains("move: Wrap"))
        .map(|l| l.as_str())
        .collect();
    assert_eq!(
        marks,
        vec![
            "|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Dragonite",
            "|-damage|p2a: Snorlax|452/482|[from] move: Wrap|[partiallytrapped]",
            "|-damage|p2a: Snorlax|452/482|[from] move: Wrap|[partiallytrapped]",
            "|-end|p2a: Snorlax|Wrap|[partiallytrapped]",
        ],
        "PT6: the duration-END branch wins — the NON-silent form, NOT `|[silent]`. got:\n{}",
        raw.join("\n")
    );
    assert_eq!(seed_str(&out.decisions[2].seed_after), "36585,19230,53587,24035", "PT6 dec2 seed");
}

/// PT5 — BATON PASS carries the trap. `partiallytrapped` has NO `noCopy`, so
/// `copyVolatileFrom` transfers it: the entrant inherits the SAME duration and the SAME source
/// (NOT re-pointed, unlike the linked `trapped` volatile), is itself firm-trapped, and is
/// chipped off ITS OWN maxhp — 672/16 = 42 for the Blissey here vs the passer's 482/16 = 30.
/// A trapped mon CAN Baton Pass: `trapped` gates the switch CHOICE, not a self-switch MOVE.
#[test]
fn partial_trap_is_baton_passed_to_the_entrant() {
    let d = dex();
    let p1 = "Dragonite||Leftovers|InnerFocus|wrap,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|batonpass,splash|Hardy|85,85,85,85,85,85|M||||]\
              Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "44446,15321,46848,55374"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Wrap lands on Snorlax
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Snorlax Baton Passes
            ScriptDecision::one(1, Choice::Switch(1)),              // Blissey in — inherits it
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let marks: Vec<&str> = raw
        .iter()
        .filter(|l| l.contains("partiallytrapped") || l.contains("move: Wrap"))
        .map(|l| l.as_str())
        .collect();
    assert_eq!(
        marks,
        vec![
            "|-activate|p2a: Snorlax|move: Wrap|[of] p1a: Dragonite",
            "|-damage|p2a: Snorlax|452/482|[from] move: Wrap|[partiallytrapped]",
            "|-damage|p2a: Blissey|630/672|[from] move: Wrap|[partiallytrapped]",
            "|-damage|p2a: Blissey|630/672|[from] move: Wrap|[partiallytrapped]",
        ],
        "PT5: the passed trap keeps chipping — off the ENTRANT's own maxhp (672/16 = 42). \
         got:\n{}",
        raw.join("\n")
    );
    assert!(st.is_trapped(1, &d), "PT5: the entrant inherits the FIRM trap");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "8730,10456,11782,14692",
        "PT5: the pass + the entrant's first chip are DRAW-FREE (the real sim's post-turn seed)");
}

// ── TRANSFORM (`gen3_transform_v1`, ROUND 33) ──────────────────────────────────────────
//
// Ground truth: `harness/probe_transform_pin.js` — each board driven through the real
// omniscient gen3 sim on the SAME packed teams, printing the POST-CONSTRUCTION seed (the
// round-29 methodology note) plus the per-decision lines AND post-turn seed. Transform is
// DRAW-FREE, so the seeds are the load-bearing half of every pin: they are what fails if the
// HYBRID speed cache (see `run_status_move`'s transform arm) is modeled as the copied or the
// original speed, because `each_event_shuffle` ties on `cached_speed`.

const TF_DITTO_1: &str = "Ditto||MetalPowder|Limber|transform|Hardy|85,85,85,85,85,85|N||||";
const TF_DITTO_1_GENGAR: &str = "Ditto||MetalPowder|Limber|transform|Hardy|85,85,85,85,85,85|N||||]Gengar||Leftovers|Levitate|splash|Hardy|85,85,85,85,85,85|M||||";
const TF_DITTO_MIRROR: &str = "Ditto||MetalPowder|Limber|transform,splash|Hardy|85,85,85,85,85,85|N||||";
const TF_LAX_4: &str = "Snorlax||Leftovers|ThickFat|swordsdance,bodyslam,rest,splash|Adamant|252,252,,,,4|M|31,31,31,31,31,3|||";
const TF_LAX_SD: &str = "Snorlax||Leftovers|ThickFat|swordsdance,bodyslam,splash|Hardy|85,85,85,85,85,85|M||||";
const TF_LAX_SUB: &str = "Snorlax||Leftovers|ThickFat|substitute,splash|Hardy|85,85,85,85,85,85|M||||";
const TF_LAX_BOOM: &str = "Snorlax||Leftovers|ThickFat|explosion,splash|Hardy|85,85,85,85,85,85|M||||]Blissey||Leftovers|NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||";
/// A Mimic carrier PLUS a second mon whose moves the carrier does NOT have — so a transformed
/// Ditto (which copied the carrier's WHOLE moveset) does NOT already know the foe's `lastMove`.
/// That isolation is the whole point: with a same-mon board the `already knows the move` fail
/// masks the `source.transformed` fail and the pin passes even when the gate is deleted.
const TF_BLISS_MIMIC_LAX: &str = "Blissey||Leftovers|NaturalCure|mimic,splash,softboiled,toxic|Hardy|85,85,85,85,85,85|F||||]Snorlax||Leftovers|ThickFat|curse,splash|Hardy|85,85,85,85,85,85|M||||";
/// NICKNAME-LESS packed forms (`|Species|…` ⇒ `set.name` empty), so the protocol ident has to be
/// derived from the mon's BASE species rather than the copied one.
const TF_DITTO_NONICK: &str = "|Ditto|MetalPowder|Limber|transform|Hardy|85,85,85,85,85,85|N||||";
const TF_LAX_NONICK: &str = "|Snorlax|Leftovers|ThickFat|bodyslam,splash|Hardy|85,85,85,85,85,85|M||||";
/// An INTIMIDATE carrier — the TF9 board.
const TF_GYARA: &str = "Gyarados||Leftovers|Intimidate|splash,bodyslam|Hardy|85,85,85,85,85,85|M||||";
/// The post-construction seed shared by every board that leads Ditto vs a Snorlax/Blissey.
const TF_SEED: &str = "42771,25880,27089,15090";

/// Strip the battle FRAMING (banner / `|t:|` / `|switch|` / `|upkeep` / `|turn|` / blanks) so a
/// pin asserts only the EFFECT lines, in order.
fn tf_effect_lines(raw: &[String]) -> Vec<&str> {
    raw.iter()
        .map(|l| l.as_str())
        .filter(|l| {
            l.starts_with("|move|")
                || l.starts_with("|-")
                || l.starts_with("|faint|")
                || l.starts_with("|cant|")
        })
        .collect()
}

/// TF1 — the FULL COPY, and the `ppUps[i] || 0` maxpp ladder that the gen3-randbats Ditto
/// (ONE move: Transform) makes the COMMON case, not an edge. A 1-move Ditto copying
/// Snorlax's `swordsdance,bodyslam,rest,splash` gets `5/48, 5/15, 5/10, 5/40`: only slot 0
/// has a `ppUps` entry to read, so slots 1-3 get the RAW base PP as their max. (A 4-move
/// transformer would get `5/48, 5/24, 5/16, 5/64` — probe B.) Also pins the copied species /
/// stats / types / ability / the copied-Speed action order, and a copied move actually USED
/// (Body Slam 5 → 4 PP, damage computed off the COPIED Atk 350 vs Snorlax's own bulk).
#[test]
fn transform_copies_the_target_with_the_ppups_maxpp_ladder() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(TF_DITTO_1, TF_LAX_4, TF_SEED), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Transform vs Swords Dance
            ScriptDecision::both(Choice::Move(1), Choice::Move(3)), // the COPIED Body Slam
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    assert_eq!(
        tf_effect_lines(&raw),
        vec![
            "|move|p1a: Ditto|Transform|p2a: Snorlax",
            "|-transform|p1a: Ditto|p2a: Snorlax",
            "|move|p2a: Snorlax|Swords Dance|p2a: Snorlax",
            "|-boost|p2a: Snorlax|atk|2",
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|move|p1a: Ditto|Body Slam|p2a: Snorlax",
            "|-damage|p2a: Snorlax|324/524",
            "|-heal|p2a: Snorlax|356/524|[from] item: Leftovers",
        ],
        "TF1: got:\n{}",
        raw.join("\n")
    );
    let u = &st.sides[0].pokemon[0];
    assert_eq!(u.species_id, "snorlax", "TF1: the species is copied");
    assert!(u.transformed(), "TF1: the transform overlay is up");
    assert_eq!(u.ability, "Thick Fat", "TF1: the ability is copied");
    assert_eq!(
        (u.stats[1], u.stats[2], u.stats[3], u.stats[4], u.stats[5]),
        (350, 166, 149, 256, 69),
        "TF1: the five NON-HP stats are copied verbatim"
    );
    assert_eq!((u.hp, u.maxhp), (258, 258), "TF1: HP/maxHP are NOT copied (Ditto keeps its own)");
    assert_eq!(
        u.set.moves.iter().map(|m| pokesim::dex::to_id(m)).collect::<Vec<_>>(),
        vec!["swordsdance", "bodyslam", "rest", "splash"],
        "TF1: the moveset is copied"
    );
    assert_eq!(u.move_pp, vec![5, 4, 5, 5], "TF1: every copied slot starts at min(5, pp); Body Slam was used");
    assert_eq!(
        u.move_maxpp,
        vec![48, 15, 10, 40],
        "TF1: maxpp = calculatePP(move, ppUps[i] || 0) — the 1-move Ditto has ppUps=[3], so \
         ONLY slot 0 is PP-upped (30*8/5=48); slots 1-3 take the raw base PP (15/10/40)"
    );
    assert_eq!(seed_str(&out.decisions[0].seed_after), "6325,61109,20311,1492",
        "TF1: the Transform turn is DRAW-FREE and its eachEvent tie-shuffles ride the HYBRID speed");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "7509,52024,24474,34576",
        "TF1: the copied-move turn matches the real sim's post-turn seed");
}

/// TF2 — REVERT ON SWITCH-OUT (`clearVolatile`): the outgoing Ditto goes back to its own
/// species/stats/ability/moveslots, and Transform's OWN remaining PP persists (15/16 — the
/// sim's `moveSlots`/`baseMoveSlots` share slot objects, so the cast's decrement survives the
/// revert). The re-entry `|switch|` must render `Ditto|258/258`, NOT the copied species.
#[test]
fn transform_reverts_on_switch_out_keeping_its_own_used_pp() {
    let d = dex();
    let mut battle = Battle::start_with_switchins(&opts_cg(TF_DITTO_1_GENGAR, TF_LAX_SD, TF_SEED), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(2)), // Transform vs Splash
            ScriptDecision::both(Choice::Switch(1), Choice::Move(2)), // pivot out → revert
            ScriptDecision::both(Choice::Switch(1), Choice::Move(2)), // and back in
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    assert!(
        raw.iter().any(|l| l == "|switch|p1a: Ditto|Ditto|258/258"),
        "TF2: the re-entry renders the mon's OWN species. got:\n{}",
        raw.join("\n")
    );
    let u = &st.sides[0].pokemon[0];
    assert_eq!(u.species_id, "ditto", "TF2: species reverted");
    assert!(!u.transformed(), "TF2: the overlay is gone");
    assert_eq!(u.ability, "Limber", "TF2: ability reverted");
    assert_eq!(u.stats[5], 153, "TF2: the five stats reverted");
    assert_eq!(u.cached_speed, 153, "TF2: `setSpecies(baseSpecies)` re-seeds the speed cache");
    assert_eq!(
        u.set.moves.iter().map(|m| pokesim::dex::to_id(m)).collect::<Vec<_>>(),
        vec!["transform"],
        "TF2: the moveslots revert to baseMoveSlots"
    );
    assert_eq!((u.move_pp.clone(), u.move_maxpp.clone()), (vec![15], vec![16]),
        "TF2: Transform's OWN remaining PP persists across the revert (15/16)");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "64322,23189,17641,10394",
        "TF2: the revert is DRAW-FREE (the real sim's post-turn seed)");
}

/// TF3 — the DITTO MIRROR: `transformInto`'s `pokemon.transformed && gen >= 2` guard makes the
/// SECOND Transform fail draw-free (`|move|…|Transform||[still]` + `|-fail|`). The USER being
/// transformed does NOT block (that guard is gen5+), which the second decision pins: the
/// already-transformed p1 Ditto Transforms AGAIN into the (still untransformed) p2 Ditto and
/// its copied Transform slot resets to 5/16.
#[test]
fn transform_into_an_already_transformed_target_fails_draw_free() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(TF_DITTO_MIRROR, TF_DITTO_MIRROR, "2775,52901,22889,24934"), &d)
            .expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    assert_eq!(
        tf_effect_lines(&raw),
        vec![
            "|move|p1a: Ditto|Transform|p2a: Ditto",
            "|-transform|p1a: Ditto|p2a: Ditto",
            "|move|p2a: Ditto|Transform||[still]",
            "|-fail|p2a: Ditto",
            "|move|p1a: Ditto|Transform|p2a: Ditto",
            "|-transform|p1a: Ditto|p2a: Ditto",
            "|move|p2a: Ditto|Splash|p2a: Ditto",
            "|-nothing",
        ],
        "TF3: got:\n{}",
        raw.join("\n")
    );
    let u = &st.sides[0].pokemon[0];
    assert!(u.transformed(), "TF3: the re-transform holds");
    assert_eq!((u.move_pp.clone(), u.move_maxpp.clone()), (vec![5, 5], vec![16, 64]),
        "TF3: the second copy re-fills the copied Transform slot to 5/16");
    assert_eq!(seed_str(&out.decisions[0].seed_after), "28052,44136,62443,55151",
        "TF3: the mirror turn (one success + one fail) is DRAW-FREE");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "44763,13952,47559,22436", "TF3: turn 2 seed");
}

/// TF4 — a SUBSTITUTE does NOT block Transform. gen3 Transform carries `bypasssub` and the
/// `pokemon.volatiles['substitute']` fail-guard in `transformInto` is gen5-ONLY, so the copy
/// goes straight THROUGH the sub. (The same board also demonstrates the absent `protect` flag
/// class: Transform's gen3 flags are `{bypasssub, metronome, failencore}` only.)
#[test]
fn transform_copies_through_a_substitute() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(TF_DITTO_MIRROR, TF_LAX_SUB, TF_SEED), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Splash vs Substitute
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Transform INTO the sub
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    assert!(
        raw.iter().any(|l| l == "|-transform|p1a: Ditto|p2a: Snorlax"),
        "TF4: the copy lands through the Substitute. got:\n{}",
        raw.join("\n")
    );
    let u = &st.sides[0].pokemon[0];
    assert_eq!(u.species_id, "snorlax");
    assert_eq!(u.stats[5], 117, "TF4: the subbed target's stats are copied");
    assert_eq!((u.move_pp.clone(), u.move_maxpp.clone()), (vec![5, 5], vec![16, 64]),
        "TF4: a 2-move Ditto PP-ups BOTH copied slots (substitute 10*8/5=16, splash 40*8/5=64)");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "13438,47692,23971,41473",
        "TF4: the through-a-sub copy is DRAW-FREE");
}

/// TF5 — REVERT ON FAINT: `faintMessages`'s `clearVolatile` reverts the copy too, so the
/// corpse reads back as the original mon (species / `transformed=false` / its own moveslots).
/// Without it a re-encoded corpse would report the copied species forever.
#[test]
fn transform_reverts_on_faint() {
    let d = dex();
    let mut battle = Battle::start_with_switchins(&opts_cg(TF_DITTO_1_GENGAR, TF_LAX_BOOM, TF_SEED), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Transform vs Splash
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // the COPIED Explosion
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    assert_eq!(
        raw.iter().map(|l| l.as_str()).filter(|l| l.starts_with("|faint|") || l.starts_with("|move|p1a")).collect::<Vec<_>>(),
        vec![
            "|move|p1a: Ditto|Transform|p2a: Snorlax",
            "|move|p1a: Ditto|Explosion|p2a: Snorlax",
            "|faint|p1a: Ditto",
            "|faint|p2a: Snorlax",
        ],
        "TF5: the self-KO'd copy still emits its ORIGINAL ident. got:\n{}",
        raw.join("\n")
    );
    let u = &st.sides[0].pokemon[0];
    assert!(u.fainted, "TF5: the copy self-KO'd");
    assert!(!u.transformed(), "TF5: the corpse is reverted");
    assert_eq!(u.species_id, "ditto", "TF5: the corpse reads back as Ditto");
    assert_eq!(
        u.set.moves.iter().map(|m| pokesim::dex::to_id(m)).collect::<Vec<_>>(),
        vec!["transform"],
        "TF5: the corpse's moveslots reverted"
    );
    assert_eq!(seed_str(&out.decisions[1].seed_after), "28052,44136,62443,55151",
        "TF5: the real sim's post-Explosion seed");
}

/// TF6 — MIMIC USED BY a transformed mon FAILS (`mimic.onHit`'s FIRST clause,
/// `source.transformed`). A Ditto that copied a Mimic carrier holds a `virtual` Mimic slot;
/// using it emits `[still]` + `-fail` and STILL deducts the PP. Only reachable now that
/// Transform is modeled — the port's old comment called this gate unreachable.
///
/// The board deliberately makes the foe SWITCH to a mon whose Curse the copied moveset does
/// NOT contain: on a same-mon board the transformed user always knows the foe's lastMove (it
/// copied the whole set), so the `already knows the move` fail masks this one and the pin
/// passes with the gate deleted. Revert-verified against that.
#[test]
fn mimic_used_by_a_transformed_mon_fails() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(TF_DITTO_1, TF_BLISS_MIMIC_LAX, TF_SEED), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),  // Transform vs Splash
            ScriptDecision::both(Choice::Move(1), Choice::Switch(1)), // foe brings in Snorlax
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),  // Snorlax Curses (an unknown move)
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),  // the COPIED Mimic
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    assert_eq!(
        raw.iter().map(|l| l.as_str()).filter(|l| l.starts_with("|move|p1a") || l.starts_with("|-fail|") || l.starts_with("|-transform|")).collect::<Vec<_>>(),
        vec![
            "|move|p1a: Ditto|Transform|p2a: Blissey",
            "|-transform|p1a: Ditto|p2a: Blissey",
            "|move|p1a: Ditto|Splash|p1a: Ditto",
            "|move|p1a: Ditto|Splash|p1a: Ditto",
            "|move|p1a: Ditto|Mimic||[still]",
            "|-fail|p1a: Ditto",
        ],
        "TF6: got:\n{}",
        raw.join("\n")
    );
    let u = &st.sides[0].pokemon[0];
    assert!(u.mimic_overlay.is_none(), "TF6: the failed Mimic overlaid nothing");
    assert_eq!(
        u.set.moves.iter().map(|m| pokesim::dex::to_id(m)).collect::<Vec<_>>(),
        vec!["mimic", "splash", "softboiled", "toxic"],
        "TF6: the copied moveset is untouched — Curse was NOT mimicked"
    );
    assert_eq!(u.move_pp, vec![4, 3, 5, 5], "TF6: the failed Mimic still cost its PP");
    assert_eq!(seed_str(&out.decisions[3].seed_after), "50887,33032,2836,3461",
        "TF6: the failed Mimic turn matches the real sim's post-turn seed");
}

/// TF8 — the protocol IDENT of a NICKNAME-LESS transformed mon. `pokemon.name` is
/// `set.name || set.species`, computed at CONSTRUCTION and never touched by `transformInto`,
/// so EVERY line still reads `p1a: Ditto` after the copy. The port's `helpers::display_name`
/// falls back to the species when there is no nickname, so it must read through the transform
/// overlay — and only a packed team in the `|Species|…` form (empty `set.name`) exercises that
/// path, which is why TF1-TF7 (all `Nickname||…` teams) cannot cover it.
#[test]
fn transform_ident_stays_the_base_species_without_a_nickname() {
    let d = dex();
    let mut battle = Battle::start_with_switchins(&opts_cg(TF_DITTO_NONICK, TF_LAX_NONICK, TF_SEED), &d)
        .expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Transform vs Splash
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // the COPIED Body Slam
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    assert_eq!(
        tf_effect_lines(&raw),
        vec![
            "|move|p1a: Ditto|Transform|p2a: Snorlax",
            "|-transform|p1a: Ditto|p2a: Snorlax",
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|move|p1a: Ditto|Body Slam|p2a: Snorlax",
            "|-damage|p2a: Snorlax|330/482",
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|-heal|p2a: Snorlax|360/482|[from] item: Leftovers",
        ],
        "TF8: every ident stays `p1a: Ditto`. got:\n{}",
        raw.join("\n")
    );
    assert_eq!(st.sides[0].pokemon[0].move_maxpp, vec![24, 40],
        "TF8: the `ppUps[i] || 0` ladder again — a 1-move Ditto PP-ups slot 0 (Body Slam \
         15*8/5=24) but NOT slot 1 (Splash keeps its raw 40)");
    assert_eq!(seed_str(&out.decisions[1].seed_after), "52552,37552,17702,10562", "TF8: post-turn seed");
}

/// TF9 — the ABILITY (`gen3_transform_v1`). Three facts, all read straight off the resolved
/// gen3 sim and probe-confirmed on this board:
///   1. it IS copied — `transformInto` ends with `if (gen > 2) this.setAbility(pokemon.ability,
///      …, isFromFormeChange = true, isTransform = true)`, so a Ditto that copies Gyarados
///      HOLDS Intimidate;
///   2. its `onStart` does **NOT** re-fire in gen3 — `setAbility`'s Start `singleEvent` is
///      gated on `this.battle.gen > 3`, so the copy drops NO Atk (both sides' boosts stay 0);
///   3. the revert target is `this.baseAbility`, **not** the live ability the copy overwrote
///      — `clearVolatile` does `this.ability = this.baseAbility`, and `baseAbility` is written
///      ONLY at construction and by `formeChange` (`sim/pokemon.ts:421`/`:1490`; `setAbility`
///      never touches it, so not even a Trace moves it). That is why `TransformOverlay` carries
///      NO `base_ability` field and `restore_transform_overlay` restores `set.ability`:
///      capturing the live ability would fight the switch-out Trace revert
///      (`gen3_berry_trace_shedskin_v1`), which runs FIRST at the same site.
#[test]
fn transform_copies_the_ability_without_refiring_its_on_start() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(TF_DITTO_1_GENGAR, TF_GYARA, TF_SEED), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let (out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Transform vs Splash
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)), // pivot out → revert
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)), // and back in
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    // (2) NOTHING ability-shaped AFTER the copy. Scoped to the post-`-transform` slice on
    //     purpose: Gyarados's OWN lead Intimidate fires at construction (`|-ability|p2a:
    //     Gyarados|Intimidate|boost` + `|-unboost|p1a: Ditto|atk|1`) and is correct — an
    //     unscoped assertion would trip on it and prove nothing about the copy.
    let after = raw
        .iter()
        .position(|l| l.starts_with("|-transform|"))
        .expect("TF9: the copy must land");
    assert!(
        !raw[after + 1..].iter().any(|l| l.contains("Intimidate") || l.starts_with("|-unboost|")),
        "TF9: gen3 gates setAbility's Start singleEvent on `gen > 3` — the COPY must fire \
         NOTHING. got:\n{}",
        raw[after + 1..].join("\n")
    );
    assert_eq!(
        raw.iter().filter(|l| l.starts_with("|-unboost|")).count(),
        1,
        "TF9: exactly ONE Atk drop in the whole battle — Gyarados's own LEAD Intimidate. \
         (The mon's live boosts are 0 by the end because the switch-out cleared them, so the \
         COUNT has to be read off the log, not the state.) got:\n{}",
        raw.join("\n")
    );
    assert_eq!(st.sides[1].pokemon[0].boosts, [0; 7], "TF9: the foe took NO Intimidate drop");
    // (3) the revert target is the SET ability.
    let u = &st.sides[0].pokemon[0];
    assert!(!u.transformed(), "TF9: reverted");
    assert_eq!(u.ability, "Limber", "TF9: clearVolatile restores baseAbility (the SET ability)");
    assert_eq!(seed_str(&out.decisions[2].seed_after), "11285,63075,44232,12427",
        "TF9: the whole copy+revert sequence is DRAW-FREE (the real sim's post-turn seed)");
}

/// TF9b — (1) the ability IS copied, asserted MID-transform (TF9 can only see the reverted
/// value, so on its own it would pass even if the copy were skipped entirely).
#[test]
fn transform_copies_the_ability() {
    let d = dex();
    let mut battle =
        Battle::start_with_switchins(&opts_cg(TF_DITTO_1_GENGAR, TF_GYARA, TF_SEED), &d).expect("start");
    let st = battle.state_mut().expect("state");
    let _ = st.run_full_battle_logged(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))],
        &d,
    );
    let u = &st.sides[0].pokemon[0];
    assert!(u.transformed(), "TF9b: the copy landed");
    assert_eq!(
        pokesim::dex::to_id(&u.ability),
        "intimidate",
        "TF9b: `if (gen > 2) setAbility(target.ability)` — the ability IS copied"
    );
    assert_eq!(
        pokesim::dex::to_id(&u.set.ability),
        "limber",
        "TF9b: …but `baseAbility` (the SET ability) is UNTOUCHED — the request's `baseAbility` \
         field and the revert both read it"
    );
}

/// TF7 — the REQUEST surface, byte-for-byte (`gen3_transform_v1`). This is where a
/// filtering-hidden Transform bug would live: the offline pickers rejected Transform, so no
/// gate ever built a transformed mon's `|request|` JSON — the exact class of hole that let the
/// original silent no-op ship. The sim's `getSwitchRequestData` serializes `this.details` and
/// `baseStoredStats`, NEITHER of which `transformInto` refreshes, so the roster entry keeps the
/// OWNER's ident / details / stats while its `moves[]` (which read the live `moveSlots`) show
/// the COPY. The `active[]` entries carry `pp:5`, the `ppUps[i] || 0` maxpp ladder AND a
/// `target` key — unlike a MIMIC-acquired slot, whose gen4 literal omits `target`
/// (`gen3_mimic_request_no_target_v1`); `transformInto`'s literal sets `target: moveSlot.target`,
/// so blanket-applying the Mimic omission here would be wrong.
///
/// Ground truth: `harness/probe_transform_pin.js`'s TF7 block (the real `getPlayerStreams` p1
/// frames on this exact board/seed).
#[test]
fn transform_request_json_is_byte_identical_to_the_sim() {
    let d = dex();
    let cmds = vec![
        pokesim::bridge::Cmd { side: 0, choice: pokesim::bridge::parse_choice("move 1").unwrap() },
        pokesim::bridge::Cmd { side: 1, choice: pokesim::bridge::parse_choice("move 4").unwrap() },
    ];
    let streams = pokesim::bridge::run_full_battle_bridge(
        &opts_cg(TF_DITTO_1_GENGAR, TF_LAX_4, TF_SEED),
        &cmds,
        &d,
    )
    .expect("bridge run");
    let reqs: Vec<&String> =
        streams.p1.iter().filter(|l| l.starts_with("|request|")).collect();
    assert!(reqs.len() >= 2, "TF7: expected the pre- and post-transform requests, got {}", reqs.len());
    assert_eq!(
        *reqs[1],
        "|request|{\"active\":[{\"moves\":[\
         {\"move\":\"Swords Dance\",\"id\":\"swordsdance\",\"pp\":5,\"maxpp\":48,\"target\":\"self\",\"disabled\":false},\
         {\"move\":\"Body Slam\",\"id\":\"bodyslam\",\"pp\":5,\"maxpp\":15,\"target\":\"normal\",\"disabled\":false},\
         {\"move\":\"Rest\",\"id\":\"rest\",\"pp\":5,\"maxpp\":10,\"target\":\"self\",\"disabled\":false},\
         {\"move\":\"Splash\",\"id\":\"splash\",\"pp\":5,\"maxpp\":40,\"target\":\"self\",\"disabled\":false}]}],\
         \"side\":{\"name\":\"P1\",\"id\":\"p1\",\"pokemon\":[\
         {\"ident\":\"p1: Ditto\",\"details\":\"Ditto\",\"condition\":\"258/258\",\"active\":true,\
         \"stats\":{\"atk\":153,\"def\":153,\"spa\":153,\"spd\":153,\"spe\":153},\
         \"moves\":[\"swordsdance\",\"bodyslam\",\"rest\",\"splash\"],\"baseAbility\":\"limber\",\
         \"item\":\"metalpowder\",\"pokeball\":\"pokeball\"},\
         {\"ident\":\"p1: Gengar\",\"details\":\"Gengar, M\",\"condition\":\"282/282\",\"active\":false,\
         \"stats\":{\"atk\":187,\"def\":177,\"spa\":317,\"spd\":207,\"spe\":277},\
         \"moves\":[\"splash\"],\"baseAbility\":\"levitate\",\"item\":\"leftovers\",\"pokeball\":\"pokeball\"}]}}",
        "TF7: the transformed mon's request must be byte-identical to the sim's"
    );
}

/// The GUARD'S NEGATIVE CONTROL, half 2 (`gen3_transform_v1`) — Transform is MODELED now, so
/// a Transform CARRIER must construct fine. Paired with `a_ditto_without_transform_builds_fine`
/// below, this is what stops a future over-broad `UNMODELED_FAILLOUD_MOVES` entry (or a
/// re-added species guard) from silently rejecting legal teams again.
#[test]
fn transform_carriers_build_now_that_transform_is_modeled() {
    let d = dex();
    let foe = "Snorlax|||immunity|bodyslam|Adamant|252,252,,,,|||||";
    for carrier in [
        "Ditto|||limber|transform|Hardy|252,,,,,|||||",
        "Mew|||synchronize|transform,psychic,softboiled,explosion|Hardy|252,,,,,|||||",
    ] {
        let b = Battle::start_with_switchins(&opts_cg(carrier, foe, "1,2,3,4"), &d);
        assert!(b.is_ok(), "a Transform carrier must construct: {carrier}");
    }
}

/// The GUARD'S NEGATIVE CONTROL — a mon that merely COULD learn Transform but does not carry
/// it must build FINE. Without this, a future over-broad guard (e.g. keyed on the SPECIES
/// `ditto`) would pass `transform_fails_loud_at_construction` while wrongly rejecting legal
/// teams, and nothing would catch it.
#[test]
fn a_ditto_without_transform_builds_fine() {
    let d = dex();
    let ditto = "Ditto|||limber|tackle|Hardy|252,,,,,|||||";
    let foe = "Snorlax|||immunity|bodyslam|Adamant|252,252,,,,|||||";
    let b = Battle::start_with_switchins(&opts_cg(ditto, foe, "1,2,3,4"), &d);
    assert!(b.is_ok(), "a Ditto with no Transform must construct normally");
}

/// `gen3_unmodeled_item_failloud_v1` — each of the FIVE items with a genuine gen-3 battle
/// effect the engine models NOTHING for must FAIL LOUD at construction rather than silently
/// no-op. Verified per-item (not once for the set) so a partial revert cannot pass.
#[test]
fn unmodeled_items_fail_loud_at_construction() {
    let d = dex();
    let foe = "Snorlax|||immunity|bodyslam|Adamant|252,252,,,,|||||";
    for item in ["shellbell", "berryjuice", "mentalherb", "machobrace", "mail"] {
        let carrier = format!("Snorlax||{item}|immunity|bodyslam|Adamant|252,252,,,,|||||");
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Battle::start_with_switchins(&opts_cg(&carrier, foe, "1,2,3,4"), &d)
        }));
        assert!(
            caught.is_err(),
            "item {item} is UNMODELED and must fail loud at construction, not silently no-op"
        );
    }
}

/// The ITEM guard's NEGATIVE CONTROL — items the engine DOES model, and an item with no gen-3
/// battle effect at all, must still construct fine. Without this, an over-broad item guard
/// (e.g. one keyed on "unreferenced in the engine source") would pass the test above while
/// wrongly rejecting legal teams, and nothing would catch it. `pokeball` stands in for the
/// 27 inert ids the guard deliberately does NOT list.
#[test]
fn modeled_and_inert_items_still_build_fine() {
    let d = dex();
    let foe = "Snorlax|||immunity|bodyslam|Adamant|252,252,,,,|||||";
    for item in ["leftovers", "choiceband", "lumberry", "brightpowder", "pokeball", ""] {
        let carrier = format!("Snorlax||{item}|immunity|bodyslam|Adamant|252,252,,,,|||||");
        let b = Battle::start_with_switchins(&opts_cg(&carrier, foe, "1,2,3,4"), &d);
        assert!(b.is_ok(), "item {item:?} must construct normally");
    }
}

/// `gen3_unmodeled_move_failloud_v2` — the ROUND-40 move audit's SIXTEEN silent-desync moves
/// (unmodeled, yet the engine RAN them with no guard per the `scan_move_probe` sweep: flat BP
/// on a variable-BP move, the lock-in family with no lock, missing first-turn / asleep-only /
/// no-KO gates) must FAIL LOUD at construction rather than run silently wrong. Verified
/// per-move (not once for the set) so a partial revert cannot pass. Exposure is ZERO on both
/// fuzzed surfaces (0 pool carriers; 0 in the entire curated randbats movepool) — this is a
/// LATENT-HAZARD guard, the moves sibling of `unmodeled_items_fail_loud_at_construction`.
#[test]
fn the_construction_fail_loud_seam_is_empty_but_still_wired() {
    // ⚠️ `UNMODELED_FAILLOUD_MOVES` is now EMPTY — every gen-3 move that used to panic at
    // construction is MODELED. The old per-move loop would therefore assert NOTHING, which is
    // exactly the vacuous-test shape this suite keeps catching, so it is replaced rather than
    // left iterating an empty list.
    //
    // The SEAM is deliberately kept (as an empty array) for the next deferral, and this pin
    // guards the property that actually matters now: every id the seam used to carry must
    // still CONSTRUCT. If someone re-adds an entry, `unmodeled_moves_fail_loud_at_construction`
    // should come back with it.
    let d = dex();
    let foe = "Snorlax|||immunity|bodyslam|Adamant|252,252,,,,|||||";
    for mv in [
        "dreameater", "falseswipe", "furycutter", "iceball", "outrage", "petaldance", "rage",
        "revenge", "rollout", "secretpower", "smellingsalts", "thrash",
    ] {
        let user = format!("Marowak|||rockhead|{mv}|Adamant|252,252,,,,|||||");
        Battle::start_with_switchins(&opts_cg(&user, foe, "1,2,3,4"), &d)
            .unwrap_or_else(|e| panic!("{mv} must construct now that it is modeled: {e:?}"));
    }
}

/// The MOVE guard's NEGATIVE CONTROL — the MODELED near-neighbours of the guarded classes
/// must still construct fine: `waterspout` (the modeled hp-scaled twin of the guarded
/// `eruption`), `return`/`flail` (modeled variable-BP), `doubleedge` (modeled recoil),
/// `sleeptalk` (engine-MODELED though the e2e picker rejects it carrier-conditionally — it
/// must NOT be swept into the guard), and `hyperbeam` (modeled recharge). Without this, an
/// over-broad guard (e.g. keyed on "has a basePowerCallback") would pass the test above
/// while wrongly rejecting legal teams, and nothing would catch it.
#[test]
fn modeled_near_neighbour_moves_still_build_fine() {
    let d = dex();
    let foe = "Snorlax|||immunity|bodyslam|Adamant|252,252,,,,|||||";
    for mv in ["waterspout", "return", "flail", "doubleedge", "sleeptalk", "hyperbeam"] {
        let carrier = format!("Snorlax|||immunity|{mv},bodyslam|Adamant|252,252,,,,|||||");
        let b = Battle::start_with_switchins(&opts_cg(&carrier, foe, "1,2,3,4"), &d);
        assert!(b.is_ok(), "modeled move {mv} must construct normally");
    }
}

// ── Draw-count / first-mover tail fixes (rmry3vbgm / rmry3ytkn A/B round) ──────────────
//
// Ground truth: `harness/probe_dc_batch_regression_rng.js` (raw seed 13,27,41,55, aligned to
// the port's draw-free `start_with_switchins` via a post-construction PRNG reset).

/// RS1 `gen3_rest_sleep_immune_v1` — a damaged mon with a SLEEP-IMMUNE ability (INSOMNIA /
/// VITAL SPIRIT) that uses REST FAILS DRAW-FREE: gen3 `setStatus('slp')` is blocked by the
/// ability's `onSetStatus` BEFORE `slp.onStart`, so the sleep `random(2,6)` is NEVER drawn
/// (no sleep, no heal). The A/B repro rmry3vbgm_ab_1_1 (seed@15) = a damaged Insomnia Hypno
/// Resting. WRONG (pre-fix): `run_rest` drew the `random(2,6)` + slept + healed → +1 draw. The
/// Insomnia-Rest turn's seed == a draw-free Amnesia control's (proof it's draw-free) and DIFFERS
/// from a genuinely sleep-able Rest (which draws the `random(2,6)`).
#[test]
fn rest_into_a_sleep_immune_ability_fails_draw_free() {
    let d = dex();
    let hypno = "Hypno|||Insomnia|rest,amnesia,,|Careful|248,,,,,|N||||";
    let snor = "Snorlax|||Immunity|amnesia,rest,,|Careful|,,,,,|N||||";

    // RS1a: Insomnia Hypno (damaged to 100) uses Rest → FAILS draw-free.
    let mut ba = Battle::start_with_switchins(&opts_cg(hypno, snor, "13,27,41,55"), &d).expect("start");
    ba.state_mut().unwrap().sides[0].pokemon[0].hp = 100;
    let oa = ba.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))], &d);
    assert!(oa.decisions[0].active[0].status.is_none(), "RS1: Insomnia blocks Rest's self-sleep");
    assert_eq!(oa.decisions[0].active[0].hp, 100, "RS1: a failed Rest does NOT heal");
    assert_eq!(seed_str(&oa.decisions[0].seed_after), "49200,48550,14066,9078",
        "RS1: the Insomnia-Rest turn is DRAW-FREE — exact post-turn seed");

    // RS1b control: the SAME Hypno does a draw-free Amnesia instead → IDENTICAL seed.
    let mut bb = Battle::start_with_switchins(&opts_cg(hypno, snor, "13,27,41,55"), &d).expect("start");
    bb.state_mut().unwrap().sides[0].pokemon[0].hp = 100;
    let ob = bb.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(1), Choice::Move(0))], &d);
    assert_eq!(seed_str(&oa.decisions[0].seed_after), seed_str(&ob.decisions[0].seed_after),
        "RS1: Insomnia-Rest == draw-free Amnesia (both draw only the endTurn Quick Claw)");

    // RS1c control: a genuinely SLEEP-ABLE Rest (Snorlax@Immunity blocks psn, NOT slp) DRAWS the
    // random(2,6) → a DIFFERENT seed (the discriminator that Rest normally draws it).
    let mut bc = Battle::start_with_switchins(&opts_cg(snor, snor, "13,27,41,55"), &d).expect("start");
    bc.state_mut().unwrap().sides[0].pokemon[0].hp = 100;
    let oc = bc.state_mut().unwrap().run_full_battle(
        &[ScriptDecision::both(Choice::Move(1), Choice::Move(0))], &d); // Snorlax move 2 = Rest
    assert_eq!(oc.decisions[0].active[0].status, Some(Status::Sleep(3)), "RS1: a sleep-able Rest sleeps");
    assert_eq!(seed_str(&oc.decisions[0].seed_after), "52334,26639,37779,28223",
        "RS1: a sleep-able Rest draws the random(2,6) — exact post-turn seed");
    assert_ne!(seed_str(&oa.decisions[0].seed_after), seed_str(&oc.decisions[0].seed_after),
        "RS1: the Insomnia-Rest seed DIFFERS from a sleep-able Rest (the missing random(2,6))");
}

/// MD1 `gen3_mimic_disable_self_overwrite_v1` — a Disable of a mon whose LAST move was a MIMIC
/// that OVERWROTE its own slot FAILS: Showdown's Disable stores the used move ID (`"mimic"`),
/// which is no longer in any moveSlot (the slot now holds the COPIED move), so `onStart`'s
/// `!hasMove` returns false. The port stored Disable by SLOT index → it wrongly disabled the
/// copied move (the A/B repro rmry3vbgm_ab_6_16 seed@46). Observable: the Mimic-copied move
/// stays USABLE and deals damage at the next turn (a disabled move would be rejected → the foe
/// takes no damage + the seed drifts).
#[test]
fn disable_of_a_mimic_overwritten_slot_fails_leaving_the_copied_move_usable() {
    let d = dex();
    let noctowl = "Noctowl|||Insomnia|mimic,amnesia,recover,rest|Careful|248,,,,,|N||||";
    let grimer = "Grimer|||Stench|tackle,disable,amnesia,recover|Careful|248,,,,,|N||||";
    let mut b = Battle::start_with_switchins(&opts_cg(noctowl, grimer, "13,27,41,55"), &d).expect("start");
    let o = b.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // Noctowl Amnesia; Grimer Tackle
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // Noctowl Mimic(Tackle); Grimer Disable → FAILS
            ScriptDecision::both(Choice::Move(0), Choice::Move(2)), // Noctowl uses the copied Tackle → Grimer takes damage
        ], &d);
    assert_eq!(o.decisions.len(), 3, "MD1: all 3 decisions committed (the copied move was NOT rejected)");
    assert_eq!(o.decisions[2].active[1].hp, 317,
        "MD1: the Mimic-copied Tackle is USABLE (Grimer 363 → 317); a wrongly-disabled slot leaves 363");
    assert_eq!(seed_str(&o.decisions[2].seed_after), "58462,50702,46937,62624",
        "MD1: exact post-decision seed with the disable correctly FAILED");
}

/// YW1 `gen3_yawn_recast_v1` — a Yawn re-cast into an ALREADY-pending yawn FAILS without
/// resetting the duration (Showdown's `addVolatile('yawn')` returns false), so the sleep
/// resolves on the ORIGINAL schedule. WRONG (pre-fix): the port RE-SET the duration to 2 → the
/// resolve (the sleep `random(2,6)`) slipped ONE turn late (the A/B repro rmry3ytkn_ab_6_22
/// seed@44). Observable: the target is ALREADY ASLEEP at dec1 (the resolve turn), and the dec1
/// seed carries the resolve's `random(2,6)`.
#[test]
fn yawn_recast_into_a_pending_yawn_does_not_reset_the_duration() {
    let d = dex();
    let swalot = "Swalot|||LiquidOoze|yawn,amnesia,,|Careful|248,,,,,|N||||";
    let snor = "Snorlax|||Immunity|amnesia,rest,,|Careful|,,,,,|N||||";
    let mut b = Battle::start_with_switchins(&opts_cg(swalot, snor, "13,27,41,55"), &d).expect("start");
    let o = b.state_mut().unwrap().run_full_battle(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // dec0: Yawn cast (draw-free)
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // dec1: Yawn RE-CAST — the yawn RESOLVES here
        ], &d);
    assert!(o.decisions[0].active[1].status.is_none(), "YW1: the cast does not sleep the foe");
    assert!(matches!(o.decisions[1].active[1].status, Some(Status::Sleep(_))),
        "YW1: the re-cast does NOT delay the resolve — the foe sleeps at dec1 (original schedule)");
    assert_eq!(seed_str(&o.decisions[1].seed_after), "25367,53124,63953,41400",
        "YW1: the resolve's random(2,6) draws at dec1 — exact post-turn seed");
}

/// PV1 (`gen3_perish_volatile_insertion_turn_v1`) — a mon PERISHED on an EARLIER turn than it
/// PROTECTS gathers its `perish` residual handler BEFORE its NO_ORDER `stall` handler, because
/// Showdown walks `pokemon.volatiles` in INSERTION order. `speed_sort` is a NON-STABLE selection
/// sort, so that pre-sort order is handed to the tie-group Fisher-Yates shuffle and decides the
/// emitted `-start|<mon>|perish<n>` order (and the faint order that follows).
///
/// WRONG (pre-fix): the port pushed perish at a FIXED position AFTER the NO_ORDER duration block,
/// which reversed the tied perish pair and emitted `perish0` for p1 first where the sim emits p2
/// first — the live external-consistency finding `soak3/divergences/sbd_msb1zfxs_b134` (Misdreavus
/// Perish-Songs on turn 16, Protects on turn 18, perish0 lands turn 19 with the `stall` volatile
/// still live). A fixed position CANNOT serve both real boards: the SAME-TURN mirror
/// (`rmrr03rmc_ab_453_2`, byte-fuzz corpus fixture 32) has Protect at priority 3 inserting `stall`
/// FIRST, so there perish must gather LAST. Comparing INSERTION TURNS separates them.
///
/// Ground truth: the real sim on this exact board (raw seed [7,11,13,17]; the port is seeded at
/// the POST-CONSTRUCTION seed below, which is the convention every golden uses — seeding the port
/// with the RAW seed instead silently replays the sim's construction draws as move-phase draws).
/// DRAW-FREE: the tie-group size and draw count are unchanged; only the pre-shuffle permutation
/// moves — so the whole seed suite + the e2e golden md5 are untouched.
#[test]
fn perish_gathered_before_a_later_protect_matches_the_sim_tick_order() {
    let d = dex();
    // A speed-TIED Aipom mirror (both 227) so every residual tie group actually shuffles, both
    // holding Leftovers so a SECOND tie group exists — it is the Leftovers group's selection-sort
    // swaps that reorder the still-unsorted perish pair.
    let p1 = "Aipom||Leftovers|RunAway|perishsong,protect,splash|Hardy|85,85,85,85,85,85|M||||";
    let p2 = "Aipom||Leftovers|RunAway|splash,return|Hardy|85,85,85,85,85,85|M||||";
    let mut battle =
        Battle::start_with_switchins(&opts_cg(p1, p2, "37635,3740,64462,10380"), &d).expect("start");
    let st = battle.state_mut().expect("state");
    // t1 Perish Song (perish inserted turn 1) · t2 idle · t3 PROTECT (stall inserted turn 3, so
    // it is still live at t4's residual) · t4 idle → the perish0 tick with BOTH volatiles present.
    let (_out, lines) = st.run_full_battle_logged(
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(2), Choice::Move(0)),
        ],
        &d,
    );
    let raw: Vec<String> = lines.into_iter().map(|l| l.0).collect();

    // GUARD: the Protect must actually have gone up, or there is no `stall` volatile and this pin
    // proves nothing about gather order.
    assert!(
        raw.iter().any(|l| l.starts_with("|-singleturn|p1a: Aipom|Protect")),
        "the Protect must succeed (it is what inserts the `stall` volatile); lines:\n{}",
        raw.join("\n")
    );

    // The non-silent perish ticks + the faints, in emission order.
    let marks: Vec<&str> = raw
        .iter()
        .filter(|l| {
            (l.contains("|perish") && !l.ends_with("|[silent]")) || l.starts_with("|faint|")
        })
        .map(|l| l.as_str())
        .collect();
    let expected = vec![
        "|-start|p1a: Aipom|perish3",
        "|-start|p2a: Aipom|perish3",
        "|-start|p2a: Aipom|perish2",
        "|-start|p1a: Aipom|perish2",
        "|-start|p1a: Aipom|perish1",
        "|-start|p2a: Aipom|perish1",
        "|-start|p2a: Aipom|perish0",
        "|-start|p1a: Aipom|perish0",
        "|faint|p2a: Aipom",
        "|faint|p1a: Aipom",
    ];
    assert_eq!(
        marks, expected,
        "perish tick + faint order must match the real sim; reverting the insertion-turn gather \
         flips the perish0 pair (and the faints) to p1-first. got:\n{}",
        marks.join("\n")
    );
}
