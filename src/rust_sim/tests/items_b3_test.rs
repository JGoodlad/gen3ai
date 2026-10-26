//! The B3 item pins (`gen3_items_b3_v1`, gen3 full parity B3). Each asserts the sim's observable
//! for one golden board and seed (`tests/vectors/items_b3_golden.txt`, `harness/gen_items_b3_golden.js`;
//! spec in `harness/probe_items_b3.js`). Before B3 each of the four items PANICKED at construction.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Weather;
use pokesim::turn::{Choice, ScriptDecision};

const SEED: &str = "19975,46914,35677,52368";

fn set(sp: &str, item: &str, ab: &str, moves: &str, evs: &str, g: &str) -> String {
    format!("{sp}|{sp}|{item}|{ab}|{moves}|Hardy|{evs}|{g}||||")
}
const EV: &str = "85,85,85,85,85,85";

fn play(p1: &str, p2: &str, turns: usize) -> (Vec<String>, Battle) {
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".into(),
        seed: Some(SEED.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let script = vec![ScriptDecision::both(Choice::Move(0), Choice::Move(0)); turns];
    let (_o, lines) = b.state_mut().unwrap().run_full_battle_logged(&script, &d);
    (lines.into_iter().map(|l| l.0).collect(), b)
}

fn blissey() -> String {
    set("Blissey", "Leftovers", "NaturalCure", "seismictoss,splash", EV, "F")
}

/// IB1 — Shell Bell heals `max(1, floor(totalDamage / 8))` of a multi-hit's SUMMED strikes, after
/// the `-hitcount`, `[of]` the target (sim: 48 + 47 = 95 → no heal at full HP on turn 1, then
/// 53 + 47 = 100 → +12, then 52 + 107 + 49 = 208 → +26... as the golden row records).
#[test]
fn ib1_shell_bell_heals_off_a_multihits_summed_strikes() {
    let (lines, _) = play(&set("Cloyster", "Shell Bell", "ShellArmor", "spikecannon,splash", EV, "M"), &blissey(), 3);
    for want in [
        "|-heal|p1a: Cloyster|174/262|[from] item: Shell Bell|[of] p2a: Blissey",
        "|-heal|p1a: Cloyster|99/262|[from] item: Shell Bell|[of] p2a: Blissey",
    ] {
        assert!(lines.iter().any(|l| l == want), "missing {want:?} in {lines:#?}");
    }
    let heal = lines.iter().position(|l| l.contains("Shell Bell")).unwrap();
    assert!(lines[..heal].iter().any(|l| l.starts_with("|-hitcount|")), "the heal follows the hitcount");
}

/// IB2 — a hit on a Substitute heals nothing (gen-3 `moveHit` returns `undefined` for a
/// HIT_SUBSTITUTE): the turn-3 Body Slam breaks the sub with Snorlax below full HP, and no
/// Shell Bell line appears that turn; the turn-2 and turn-4 hits on the mon do heal.
#[test]
fn ib2_a_substitute_strike_heals_nothing() {
    let p2 = set("Jolteon", "Leftovers", "VoltAbsorb", "substitute,seismictoss", EV, "M");
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".into(),
        seed: Some(SEED.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(set("Snorlax", "Shell Bell", "ThickFat", "bodyslam,splash", EV, "M")) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let script: Vec<ScriptDecision> = [0, 1, 0, 1]
        .iter()
        .map(|&k| ScriptDecision::both(Choice::Move(0), Choice::Move(k)))
        .collect();
    let (_o, lines) = b.state_mut().unwrap().run_full_battle_logged(&script, &d);
    let lines: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let at = |t: &str| lines.iter().position(|l| l == t).unwrap();
    let turn3 = &lines[at("|turn|3")..at("|turn|4")];
    assert!(turn3.iter().any(|l| l == "|-end|p2a: Jolteon|Substitute"), "turn 3 breaks the sub");
    assert!(!turn3.iter().any(|l| l.contains("Shell Bell")), "a sub strike healed: {turn3:#?}");
    assert!(lines.iter().any(|l| l == "|-heal|p1a: Snorlax|403/482|[from] item: Shell Bell|[of] p2a: Jolteon"));
}

/// IB3 — Macho Brace halves Speed: the braced Jolteon, faster than the max-Speed Snorlax without
/// it, moves second every turn.
#[test]
fn ib3_macho_brace_halves_speed() {
    let foe = set("Snorlax", "Leftovers", "ThickFat", "tackle", "0,0,0,0,0,252", "M");
    let (braced, _) = play(&set("Jolteon", "Macho Brace", "VoltAbsorb", "tackle", EV, "M"), &foe, 1);
    let (plain, _) = play(&set("Jolteon", "Leftovers", "VoltAbsorb", "tackle", EV, "M"), &foe, 1);
    let first = |ls: &[String]| ls.iter().find(|l| l.starts_with("|move|")).cloned().unwrap();
    assert_eq!(first(&braced), "|move|p2a: Snorlax|Tackle|p1a: Jolteon");
    assert_eq!(first(&plain), "|move|p1a: Jolteon|Tackle|p2a: Snorlax");
}

/// IB4 — a Macho Brace lead's turn-0 action Speed is halved too: the raw-faster Drought Ninetales
/// fires AFTER the Drizzle Politoed, so its sun is the weather that stands.
#[test]
fn ib4_a_macho_brace_lead_fires_after_a_slower_lead() {
    let foe = set("Politoed", "Leftovers", "Drizzle", "splash", EV, "M");
    let (_, braced) = play(&set("Ninetales", "Macho Brace", "Drought", "splash", EV, "M"), &foe, 1);
    let (_, plain) = play(&set("Ninetales", "Leftovers", "Drought", "splash", EV, "M"), &foe, 1);
    assert_eq!(braced.state().unwrap().field.weather, Some(Weather::Sun));
    assert_eq!(plain.state().unwrap().field.weather, Some(Weather::Rain));
}

/// IB5 — Trick fails on a Mail holder, either side: `[still]` + `-fail` on the user, items kept.
#[test]
fn ib5_mail_blocks_trick_either_way() {
    for (mine, theirs) in [("Choice Band", "Mail"), ("Mail", "Leftovers")] {
        let (lines, b) = play(
            &set("Alakazam", mine, "Synchronize", "trick,splash", EV, "M"),
            &set("Snorlax", theirs, "ThickFat", "splash", EV, "M"),
            1,
        );
        assert!(lines.iter().any(|l| l == "|move|p1a: Alakazam|Trick||[still]"), "{mine}/{theirs}: {lines:#?}");
        assert!(lines.iter().any(|l| l == "|-fail|p1a: Alakazam"));
        let st = b.state().unwrap();
        assert_eq!((st.sides[0].pokemon[0].item.as_str(), st.sides[1].pokemon[0].item.as_str()), (mine, theirs));
    }
}

/// IB6 — Mental Herb cures a Cute Charm infatuation at the Update after the contact hit, and is
/// spent: the next infatuation sticks.
#[test]
fn ib6_mental_herb_cures_attraction_once() {
    let (lines, b) = play(
        &set("Snorlax", "Mental Herb", "ThickFat", "tackle,splash", EV, "M"),
        &set("Clefable", "Leftovers", "CuteCharm", "splash", EV, "F"),
        5,
    );
    let i = lines.iter().position(|l| l == "|-enditem|p1a: Snorlax|Mental Herb").expect("the herb is used");
    assert_eq!(
        &lines[i - 1..i + 3],
        &[
            "|-start|p1a: Snorlax|Attract|[from] ability: Cute Charm|[of] p2a: Clefable".to_string(),
            "|-enditem|p1a: Snorlax|Mental Herb".to_string(),
            "|-end|p1a: Snorlax|Attract|[silent]".to_string(),
            "|-end|p1a: Snorlax|move: Attract|[from] item: Mental Herb".to_string(),
        ]
    );
    assert!(lines.iter().any(|l| l == "|cant|p1a: Snorlax|Attract"), "the second infatuation sticks");
    assert_eq!(b.state().unwrap().sides[0].pokemon[0].item, "");
}
