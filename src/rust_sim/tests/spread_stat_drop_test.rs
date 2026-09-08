//! FEATURE PINS for the SPREAD STAT-DROP family (`gen3_spread_stat_drop_v1`, ROUND 58).
//!
//! leer / growl / tailwhip / stringshot / sweetscent — the five gen-3 stat-drop status moves whose
//! target is `allAdjacentFoes`. They reached the engine through a **data** change alone (the
//! extractor's `_stat_drop_boosts` target gate widened to admit `allAdjacentFoes`, which in
//! SINGLES resolves to the one foe), so there is no new engine code to pin — what these pin is
//! that the EXISTING arm serves them correctly on the axes a spread target could plausibly have
//! changed. `spread_stat_drop_golden_test.rs` is the 1032-row sweep; these are the named,
//! human-readable statements of what must stay true, in the style the regression file uses.
//!
//! Ground truth: `harness/probe_spread_stat_drop.js` (re-runnable — do not re-derive from source).
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision};

fn dex() -> Dex {
    Dex::for_gen(3)
}

fn play(p1: &str, p2: &str, casts: usize) -> Vec<String> {
    let d = dex();
    let opts = BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some("9,9,9,9".to_string()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let st = b.state_mut().expect("state");
    let script: Vec<ScriptDecision> = (0..casts)
        .map(|_| ScriptDecision::both(Choice::Move(0), Choice::Move(0)))
        .collect();
    let (_o, lines) = st.run_full_battle_logged(&script, &d);
    lines.into_iter().map(|l| l.0).collect()
}

fn gengar(m: &str) -> String {
    format!("|Gengar|Leftovers|Levitate|{m},splash,splash,splash|Hardy|85,85,85,85,85,85|M||||")
}
const INERT: &str =
    "|Snorlax|Leftovers|Immunity|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||";

fn has(lines: &[String], needle: &str) -> bool {
    lines.iter().any(|l| l == needle)
}

/// SD1 — each spread stat-drop lands its declared drop on the FOE, and the `|move|` announce
/// renders the FOE (not the user).
///
/// The announce half is the one a spread target could genuinely have broken:
/// `status_move_announce_renders_user` decides the `|move|` target field from `move.target`, and
/// an `allAdjacentFoes` entry added to that list would render the USER — which makes poke-env key
/// the line to the wrong mon (the same class as the `gen3_nickname_ident_v1` crash).
#[test]
fn every_spread_stat_drop_lands_on_the_foe_and_announces_the_foe() {
    // (move, display name, expected -unboost token)
    let cases = [
        ("leer", "Leer", "def"),
        ("growl", "Growl", "atk"),
        ("tailwhip", "Tail Whip", "def"),
        ("stringshot", "String Shot", "spe"),
        ("sweetscent", "Sweet Scent", "evasion"),
    ];
    for (id, name, stat) in cases {
        let lines = play(&gengar(id), INERT, 1);
        assert!(
            has(&lines, &format!("|move|p1a: Gengar|{name}|p2a: Snorlax")),
            "SD1/{id}: the announce must render the FOE. got:\n{}",
            lines.join("\n")
        );
        assert!(
            has(&lines, &format!("|-unboost|p2a: Snorlax|{stat}|1")),
            "SD1/{id}: expected a −1 {stat} drop on the foe. got:\n{}",
            lines.join("\n")
        );
    }
}

/// SD2 — at the −6 floor the drop is CAPPED to 0 and the delta-0 line still prints.
///
/// `Battle.boost` runs `getCappedBoost` BEFORE `runEvent('TryBoost')`
/// (`gen3_boost_cap_before_tryboost_v1`, ROUND 55), and a PRIMARY foe-drop emits its line even at
/// a zero delta. Seven Leers: the first six each drop a stage, the seventh reports `def|0`.
///
/// NON-VACUITY: the six real drops are asserted too, so a model that emitted nothing at all
/// cannot pass the "no `def|1` on the last cast" half by accident.
#[test]
fn a_spread_drop_at_the_floor_emits_the_delta_zero_line() {
    let lines = play(&gengar("leer"), INERT, 7);
    let ones = lines.iter().filter(|l| l.as_str() == "|-unboost|p2a: Snorlax|def|1").count();
    let zeros = lines.iter().filter(|l| l.as_str() == "|-unboost|p2a: Snorlax|def|0").count();
    assert_eq!(ones, 6, "SD2: expected six real −1 drops before the floor. got:\n{}", lines.join("\n"));
    assert_eq!(zeros, 1, "SD2: expected exactly one DELTA-0 line at the floor. got:\n{}", lines.join("\n"));
}

/// SD3 — SOUNDPROOF blocks GROWL (the family's only `sound` move) and NOT Leer.
///
/// The Leer control is load-bearing: without it the test would also pass on an engine that
/// blocked every spread stat-drop against a Soundproof holder.
#[test]
fn soundproof_blocks_growl_but_not_leer() {
    const VOLTORB: &str =
        "|Voltorb|Leftovers|Soundproof|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||";
    let growl = play(&gengar("growl"), VOLTORB, 1);
    assert!(
        has(&growl, "|-immune|p2a: Voltorb|[from] ability: Soundproof"),
        "SD3: Soundproof must block Growl. got:\n{}",
        growl.join("\n")
    );
    assert!(
        !growl.iter().any(|l| l.starts_with("|-unboost|")),
        "SD3: a blocked Growl must drop nothing. got:\n{}",
        growl.join("\n")
    );

    let leer = play(&gengar("leer"), VOLTORB, 1);
    assert!(
        has(&leer, "|-unboost|p2a: Voltorb|def|1"),
        "SD3 control: Leer is NOT a sound move and must land. got:\n{}",
        leer.join("\n")
    );
}

/// SD4 — the `onTryBoost` immunity abilities gate a spread drop exactly as they gate Screech.
///
/// Clear Body blocks everything; Hyper Cutter blocks ONLY Attack — so Growl is refused and Leer
/// (Defense) is not. The Hyper Cutter control is what stops this passing on a blanket
/// "any ability blocks any drop".
#[test]
fn the_boost_immunity_abilities_gate_a_spread_drop() {
    const REGIROCK: &str =
        "|Regirock|Leftovers|ClearBody|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||";
    const PINSIR: &str =
        "|Pinsir|Leftovers|HyperCutter|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||";

    let cb = play(&gengar("growl"), REGIROCK, 1);
    assert!(
        cb.iter().any(|l| l.contains("[from] ability: Clear Body")),
        "SD4: Clear Body must refuse Growl. got:\n{}",
        cb.join("\n")
    );

    let hc = play(&gengar("growl"), PINSIR, 1);
    assert!(
        hc.iter().any(|l| l.contains("[from] ability: Hyper Cutter")),
        "SD4: Hyper Cutter must refuse Growl (Attack). got:\n{}",
        hc.join("\n")
    );

    // CONTROL: Hyper Cutter guards ATTACK only, so Leer's Defense drop must land.
    let hc_leer = play(&gengar("leer"), PINSIR, 1);
    assert!(
        has(&hc_leer, "|-unboost|p2a: Pinsir|def|1"),
        "SD4 control: Hyper Cutter must NOT block a Defense drop. got:\n{}",
        hc_leer.join("\n")
    );
}
