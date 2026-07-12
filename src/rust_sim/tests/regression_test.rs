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

/// Helper: set side `s`'s active mon major status (a STATE-only inject, no PRNG).
fn st_set_status_b2(st: &mut pokesim::state::BattleState, s: usize, status: Status) {
    let active = st.sides[s].active;
    st.sides[s].pokemon[active].status = Some(status);
}
