//! Residual-faint regression tests — the two BLOCKER fixes the multi-turn review
//! found, which the seed-sweep `battle_golden` never triggers (its faints are all
//! MOVE faints, and its residual-faint scenarios are distinct-speed where the
//! trailing shuffle draws 0). These construct the exact paths DETERMINISTICALLY
//! and pin the fix BEHAVIOUR (the per-seed seed-parity is the harness/capstone's
//! job; the reviewers already proved these against the live sim with exact seeds).
//!
//! Both use `gen3customgame`-style hacked sets (Sand Stream on any mon) via the
//! team codec, which accepts arbitrary abilities — the same trick the harness uses.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Weather;

fn dex() -> Dex {
    Dex::for_gen(3)
}

fn opts(p1: &str, p2: &str) -> BattleOptions {
    BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some("[1,2,3,4]".to_string()),
        p1: PlayerOptions { name: "A".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "B".to_string(), team: PackedTeam(p2.to_string()) },
    }
}

/// BLOCKER 1: a residual (sand chip) KO under a SPEED TIE must SKIP the trailing
/// `eachEvent('Update')` `[15]` shuffle + the end-of-turn Quick Claw — the sim
/// faints the mon (faintMessages) and requests a switch BEFORE that trailing
/// Update, so it is never drawn. Drawing it desyncs the running PRNG on every
/// later turn. Repro: two identical Sand-Stream Gengar (Ghost/Poison — sand-
/// vulnerable, always speed-tied) using Tackle (Normal → the OTHER Gengar is
/// IMMUNE, 0 damage), both at 1 HP, so the sandstorm chip is the sole, simultaneous
/// KO on a tie turn.
#[test]
fn residual_ko_under_tie_skips_trailing_draws() {
    let d = dex();
    // name | species | item | ability | moves | nature | evs | gender | ivs | shiny | level | happiness
    let gengar_ss = "Gengar|||sandstream|tackle|Serious||||||";
    let mut battle = Battle::start_with_switchins(&opts(gengar_ss, gengar_ss), &d).expect("start");
    let st = battle.state_mut().expect("state");

    // Sand Stream (mirror) set permanent sand at switch-in.
    assert_eq!(st.field.weather, Some(Weather::Sand), "both Gengar set sand");

    // Drop both leads to 1 HP so the sand chip (>=1) is the lethal, simultaneous
    // residual under the (identical-mon) speed tie.
    for side in 0..2 {
        let slot = st.sides[side].active;
        st.sides[side].pokemon[slot].hp = 1;
    }

    let result = st.run_turn(0, 0, &d); // both use Tackle (slot 0) — immune, 0 damage

    // Both fainted to the sand chip.
    for side in 0..2 {
        let slot = st.sides[side].active;
        let mon = &st.sides[side].pokemon[slot];
        assert!(mon.fainted, "side {side} fainted to the sand chip");
        assert_eq!(mon.hp, 0, "side {side} at 0 HP");
    }
    // THE FIX: the residual faint returned BEFORE the trailing [15] Update shuffle
    // and the Quick Claw — so Quick Claw was NOT drawn (the blocker-1 desync).
    assert!(
        !result.quick_claw_drawn,
        "a residual faint must skip the trailing Update shuffle + Quick Claw"
    );
}

/// BLOCKER 2: a mon the weather chip KO'd (order 8) must NOT be revived by its OWN
/// later Leftovers handler (order 10) — Showdown's `heal` early-returns on a 0-HP
/// target. Repro: a Sand-Stream Tyranitar (Rock/Dark — sand-immune) vs a Leftovers
/// Gengar (sand-vulnerable) at 1 HP; the sand chip KOs the Gengar, then its
/// Leftovers must be skipped (no revive to `maxhp/16`).
#[test]
fn weather_chip_ko_is_not_revived_by_own_leftovers() {
    let d = dex();
    let ttar_ss = "Tyranitar|||sandstream|tackle|Adamant||||||";
    let gengar_lefto = "Gengar||leftovers||tackle|Serious||||||";
    let mut battle = Battle::start_with_switchins(&opts(ttar_ss, gengar_lefto), &d).expect("start");
    let st = battle.state_mut().expect("state");
    assert_eq!(st.field.weather, Some(Weather::Sand), "Tyranitar set sand");

    // Gengar (p2) to 1 HP; the sand chip will KO it before its Leftovers runs.
    let g_slot = st.sides[1].active;
    st.sides[1].pokemon[g_slot].hp = 1;

    let _ = st.run_turn(0, 0, &d); // Tackle into Gengar is immune (Ghost) — 0 damage

    let gengar = &st.sides[1].pokemon[g_slot];
    // THE FIX: Leftovers did NOT heal the 0-HP chip-KO'd Gengar back to maxhp/16.
    assert_eq!(gengar.hp, 0, "chip-KO'd Leftovers holder stays at 0 (no revive)");
    assert!(gengar.fainted, "and is fainted");
}
