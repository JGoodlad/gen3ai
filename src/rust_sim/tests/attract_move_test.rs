//! The ATTRACT move pins (`gen3_attract_move_v1`, gen3 full parity B9). Each asserts the sim's
//! observable on one golden board and seed (`tests/vectors/attract_move_golden.txt`; spec
//! `harness/probe_attract_move.js`). Before B9 the move PANICKED as an unmodeled status move.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision};

fn set(sp: &str, ab: &str, moves: &str, g: &str) -> String {
    format!("{sp}|{sp}|Leftovers|{ab}|{moves}|Hardy|85,85,85,85,85,85|{g}||||")
}

fn play(seed: &str, p1: &str, p2: &str, script: &[(usize, usize)]) -> Vec<String> {
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".into(),
        seed: Some(seed.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let script: Vec<ScriptDecision> =
        script.iter().map(|&(a, c)| ScriptDecision::both(Choice::Move(a), Choice::Move(c))).collect();
    let (_o, lines) = b.state_mut().unwrap().run_full_battle_logged(&script, &d);
    lines.into_iter().map(|l| l.0).collect()
}

const SEED: &str = "33542,48174,19433,1262";

fn caster() -> String {
    set("Nidoking", "PoisonPoint", "attract,splash,tackle", "M")
}

/// AM1 — M into F lands: a BARE `-start` (no `[from]`, unlike Cute Charm's), then the slower
/// target's onBeforeMove `-activate`.
#[test]
fn am1_a_compatible_cast_lands_bare() {
    let lines = play(SEED, &caster(), &set("Nidoqueen", "PoisonPoint", "splash,tackle", "F"), &[(0, 1)]);
    let i = lines.iter().position(|l| l == "|move|p1a: Nidoking|Attract|p2a: Nidoqueen").expect("the cast");
    assert_eq!(lines[i + 1], "|-start|p2a: Nidoqueen|Attract");
    assert_eq!(lines[i + 2], "|-activate|p2a: Nidoqueen|move: Attract|[of] p1a: Nidoking");
}

/// AM2 — the move's own `onTryImmunity`: a same-gender target shows a bare `-immune`.
#[test]
fn am2_a_same_gender_target_is_immune() {
    let lines = play("33880,3390,49296,58274", &caster(), &set("Nidoking", "PoisonPoint", "splash", "M"), &[(0, 0)]);
    assert!(lines.iter().any(|l| l == "|-immune|p2a: Nidoking"), "{lines:#?}");
    assert!(!lines.iter().any(|l| l.contains("|-start|") && l.contains("Attract")));
}

/// AM3 — Oblivious refuses the volatile: the move fails `[still]` + `-fail|<user>`, not `-immune`.
#[test]
fn am3_oblivious_fails_the_move() {
    let lines = play(SEED, &caster(), &set("Slowbro", "Oblivious", "splash", "F"), &[(0, 0)]);
    assert!(lines.iter().any(|l| l == "|move|p1a: Nidoking|Attract||[still]"), "{lines:#?}");
    assert!(lines.iter().any(|l| l == "|-fail|p1a: Nidoking"));
}

/// AM4 — `bypasssub`: a Substitute does not block it (the probe's settled branch 6).
#[test]
fn am4_a_substitute_does_not_block_it() {
    let lines = play(SEED, &caster(), &set("Nidoqueen", "PoisonPoint", "substitute,splash", "F"), &[(1, 0), (0, 1)]);
    assert!(lines.iter().any(|l| l == "|-start|p2a: Nidoqueen|Substitute"));
    assert!(lines.iter().any(|l| l == "|-start|p2a: Nidoqueen|Attract"), "{lines:#?}");
}

/// AM5 — the post-construction-seed path cannot know a sampled gender: an UNSPECIFIED one panics
/// at the compare (fail-loud), it is never guessed.
#[test]
#[should_panic(expected = "UNSPECIFIED gender")]
fn am5_an_unspecified_gender_fails_loud() {
    play(SEED, &caster(), "Nidoqueen|Nidoqueen|Leftovers|PoisonPoint|splash|Hardy|85,85,85,85,85,85|||||", &[(0, 0)]);
}
