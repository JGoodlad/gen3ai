//! FEATURE PINS for the FOCUS ENERGY MOVE (`gen3_focus_energy_move_v1`, ROUND 59).
//!
//! The volatile has been modelled since `gen3_ability_batch4_v1`; until this round the only way to
//! reach it in gen 3 was a **Lansat Berry** eat. These pin the MOVE entry.
//!
//! Ground truth: `harness/probe_focus_energy.js` (re-runnable — do not re-derive from source).
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision};

const USER: &str =
    "|Machamp|Leftovers|Guts|focusenergy,tackle,splash,splash|Hardy|85,85,85,85,85,85|M||||";
const INERT: &str =
    "|Snorlax|Leftovers|Immunity|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||";
const SNATCHER: &str =
    "|Snorlax|Leftovers|Immunity|snatch,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||";

fn play_seeded(p1: &str, p2: &str, script: &[(usize, usize)], seed: &str) -> Vec<String> {
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some(seed.to_string()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let st = b.state_mut().expect("state");
    let s: Vec<ScriptDecision> = script
        .iter()
        .map(|&(a, c)| ScriptDecision::both(Choice::Move(a), Choice::Move(c)))
        .collect();
    let (_o, lines) = st.run_full_battle_logged(&s, &d);
    lines.into_iter().map(|l| l.0).collect()
}

fn play(p1: &str, p2: &str, script: &[(usize, usize)]) -> Vec<String> {
    play_seeded(p1, p2, script, "9,9,9,9")
}

fn has(lines: &[String], needle: &str) -> bool {
    lines.iter().any(|l| l == needle)
}

/// FE1 — a plain cast starts the volatile with the `move: `-prefixed line, and is DRAW-FREE.
///
/// The prefix matters: the Lansat Berry path emits its own item framing, and a bare
/// `|-start|<user>|Focus Energy` would not match the sim's `|-start|<user>|move: Focus Energy`.
#[test]
fn a_plain_cast_starts_focus_energy() {
    let lines = play(USER, INERT, &[(0, 0)]);
    assert!(
        has(&lines, "|move|p1a: Machamp|Focus Energy|p1a: Machamp"),
        "FE1: the self-target announce must render the USER. got:\n{}",
        lines.join("\n")
    );
    assert!(
        has(&lines, "|-start|p1a: Machamp|move: Focus Energy"),
        "FE1: expected the `move: `-prefixed start line. got:\n{}",
        lines.join("\n")
    );
}

/// FE2 — a SECOND cast fails with the did-nothing form: the `|move|` target field is BLANKED and
/// `|[still]` appended, then a bare `-fail` on the USER.
///
/// NON-VACUITY: the first cast's `-start` is asserted too, so an engine that never applied the
/// volatile at all could not pass the "second cast fails" half by accident.
#[test]
fn a_second_cast_fails_with_the_still_form() {
    let lines = play(USER, INERT, &[(0, 0), (0, 0)]);
    assert!(
        has(&lines, "|-start|p1a: Machamp|move: Focus Energy"),
        "FE2: the FIRST cast must land, else this proves nothing. got:\n{}",
        lines.join("\n")
    );
    assert!(
        has(&lines, "|move|p1a: Machamp|Focus Energy||[still]"),
        "FE2: the re-cast announce must blank the target and carry [still]. got:\n{}",
        lines.join("\n")
    );
    assert!(
        has(&lines, "|-fail|p1a: Machamp"),
        "FE2: the re-cast must `-fail` on the USER. got:\n{}",
        lines.join("\n")
    );
    assert_eq!(
        lines.iter().filter(|l| l.as_str() == "|-start|p1a: Machamp|move: Focus Energy").count(),
        1,
        "FE2: the volatile must be started exactly ONCE. got:\n{}",
        lines.join("\n")
    );
}

/// FE3 — the volatile is **READ**, not merely set: with it up, the crit rate goes from gen-3
/// stage 0 (1/16) to stage +2 (1/4).
///
/// 🚨 **THIS IS THE PIN THAT MATTERS.** An engine that emitted the `-start` line and never touched
/// the crit ratio would satisfy FE1 and FE2 completely. That failure mode is not hypothetical:
/// ROUND 57 found a move whose named pin asserted its emissions while its draw model had been
/// wrong for months. The CONTROL arm is the same board and the same seeds with the cast replaced
/// by a Splash, so the difference is attributable to the volatile alone.
#[test]
fn focus_energy_actually_raises_the_crit_rate() {
    let mut with = 0usize;
    let mut without = 0usize;
    let n = 200;
    for k in 0..n {
        let seed = format!("{},{},{},{}", 100 + k * 3, 200 + k * 7, 300 + k * 11, 400 + k * 13);
        // slot 0 = Focus Energy, slot 2 = Splash (the control), slot 1 = Tackle.
        let a = play_seeded(USER, INERT, &[(0, 0), (1, 0)], &seed);
        let b = play_seeded(USER, INERT, &[(2, 0), (1, 0)], &seed);
        if a.iter().any(|l| l.starts_with("|-crit|")) {
            with += 1;
        }
        if b.iter().any(|l| l.starts_with("|-crit|")) {
            without += 1;
        }
    }
    assert!(
        with > without * 2,
        "FE3: crits WITH Focus Energy {with}/{n} vs control {without}/{n} — the volatile is not \
         being read by the crit path (gen3 expects roughly 1/4 vs 1/16)"
    );
    // NON-VACUITY on the control: if the control never crit at all the comparison would be
    // trivially satisfiable by any nonzero `with`.
    assert!(without > 0, "FE3: the control never crit in {n} seeds — the comparison is vacuous");
}

/// FE4 — SNATCH steals Focus Energy, and the steal comes for free because the interception gates
/// on the dex's own `flags.snatch` rather than an id list.
#[test]
fn snatch_steals_focus_energy() {
    let lines = play(USER, SNATCHER, &[(0, 0)]);
    assert!(
        lines.iter().any(|l| l.contains("|-activate|p2a: Snorlax|move: Snatch")),
        "FE4: the snatcher must announce the steal. got:\n{}",
        lines.join("\n")
    );
    assert!(
        has(&lines, "|-start|p2a: Snorlax|move: Focus Energy"),
        "FE4: the SNATCHER gets the volatile. got:\n{}",
        lines.join("\n")
    );
    assert!(
        !has(&lines, "|-start|p1a: Machamp|move: Focus Energy"),
        "FE4: the victim must NOT also get it. got:\n{}",
        lines.join("\n")
    );
}
