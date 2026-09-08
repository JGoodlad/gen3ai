//! The CONFUSION-MOVE FAMILY differential gate (`gen3_confusion_move_family_v1`, ROUND 57).
//!
//! The four gen-3 status moves whose whole effect is `volatileStatus: 'confusion'` —
//! **confuseray** (acc 100), **supersonic** (acc 55, SOUND), **sweetkiss** (acc 75) and
//! **teeterdance** (acc 100, `allAdjacent`).
//!
//! Two assertion styles, both against the REAL Showdown:
//!   * a **DRAW-COUNT (seed) gate** — the post-decision PRNG seed vs the sim's, seeded at the
//!     sim's own pre-first-decision state. The ground-truth constants below are copied verbatim
//!     from `harness/probe_confusion_family_rng.js`; regenerate with that script after any
//!     draw-order change.
//!   * **EMISSION gates** — the exact `|...|` lines for each of the six dispositions.
//!
//! 🚨 **THE SPLASH CONTROL IS LOAD-BEARING.** Without it a seed mismatch cannot be told apart
//! from a wrong seeding convention, and that is precisely the mistake that made the first read of
//! this bug ambiguous: the port and the sim were compared across DIFFERENT windows and the
//! difference looked like evidence. The control shares the board, the seed and the harness and
//! differs only in the move, so a control that matches while a family member does not localizes
//! the defect to the move.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision};

/// The sim's PRNG state at the pre-first-decision boundary for every board below (they share
/// the `[9,9,9,9]` start seed and a one-mon-per-side team, so construction consumes the same
/// draws). Printed as `BEFORE=` by `harness/probe_confusion_family_rng.js`.
const SEED_BEFORE: &str = "53118,34657,41207,29520";

fn dex() -> Dex {
    Dex::for_gen(3)
}

fn opts(p1: &str, p2: &str) -> BattleOptions {
    BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some(SEED_BEFORE.to_string()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    }
}

fn gengar(m: &str) -> String {
    format!("|Gengar|Leftovers|Levitate|{m},splash,splash,splash|Hardy|85,85,85,85,85,85|M||||")
}
const INERT: &str =
    "|Snorlax|Leftovers|Immunity|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||";
const SUBBER: &str =
    "|Snorlax|Leftovers|Immunity|substitute,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||";
const OWN_TEMPO: &str =
    "|Slowbro|Leftovers|OwnTempo|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||";
const SOUNDPROOF: &str =
    "|Voltorb|Leftovers|Soundproof|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||";
const SAFEGUARDER: &str =
    "|Snorlax|Leftovers|Immunity|safeguard,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||";

/// Run `script` and return `(seed_after, emitted lines)`.
fn play(p1: &str, p2: &str, script: &[ScriptDecision]) -> (String, Vec<String>) {
    let d = dex();
    let mut b = Battle::start_with_switchins(&opts(p1, p2), &d).expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(script, &d);
    (st.prng_seed(), lines.into_iter().map(|l| l.0).collect())
}

fn both(a: usize, b: usize) -> ScriptDecision {
    ScriptDecision::both(Choice::Move(a), Choice::Move(b))
}

// ─────────────────────────────────────────────────────────────────────────────
// CF1 — the DRAW-COUNT gate.
// ─────────────────────────────────────────────────────────────────────────────

/// CF1 — every disposition of the family consumes the SIM'S EXACT DRAW COUNT.
///
/// 🚨 **THE PRE-FIX PORT FAILED THE FIRST ROW OF THIS TEST.** The ROUND-44 Confuse Ray arm
/// never rolled its accuracy (its header claimed the roll happened "upstream"; `run_move` rolls
/// accuracy on the DAMAGING path only, and every status arm rolls its own), so a Confuse Ray
/// turn consumed one draw FEWER than the sim and desynced the stream from that point on.
#[test]
fn confusion_family_seed_matches_showdown() {
    // (label, p1 move, p2 team, script, sim seedAfter)
    let cases: Vec<(&str, &str, &str, Vec<ScriptDecision>, &str)> = vec![
        // THE CONTROL. Same board, same seed, a move with no draw of its own — it isolates the
        // seeding convention from the mechanic.
        ("CONTROL splash", "splash", INERT, vec![both(0, 0)], "1967,25250,54700,60755"),
        // A plain hit: accuracy + the random(2,6) duration. All four land on the same seed
        // because a randomChance is ONE draw whatever its numerator.
        ("confuseray plain", "confuseray", INERT, vec![both(0, 0)], "48590,39028,4743,65508"),
        ("supersonic plain", "supersonic", INERT, vec![both(0, 0)], "48590,39028,4743,65508"),
        ("sweetkiss plain", "sweetkiss", INERT, vec![both(0, 0)], "48590,39028,4743,65508"),
        ("teeterdance plain", "teeterdance", INERT, vec![both(0, 0)], "48590,39028,4743,65508"),
        // Already confused on the second cast: accuracy ONLY, no duration draw.
        (
            "confuseray x2 (already confused)",
            "confuseray",
            INERT,
            vec![both(0, 0), both(0, 0)],
            "13054,22811,65435,26870",
        ),
        // OWN TEMPO and SOUNDPROOF both short-circuit AFTER the accuracy roll, so they share a
        // seed — and that shared value is what proves the accuracy roll is still drawn on a
        // blocked cast (a "skip the whole move" implementation would land elsewhere).
        (
            "supersonic vs Own Tempo",
            "supersonic",
            OWN_TEMPO,
            vec![both(0, 0)],
            "13873,36144,22950,43906",
        ),
        (
            "supersonic vs Soundproof",
            "supersonic",
            SOUNDPROOF,
            vec![both(0, 0)],
            "13873,36144,22950,43906",
        ),
        // Two-decision boards: the blocker goes up on turn 1, the cast lands on turn 2.
        (
            "confuseray vs Substitute",
            "confuseray",
            SUBBER,
            vec![
                ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
                ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
            ],
            "13521,38977,6462,56077",
        ),
        (
            "supersonic vs Safeguard",
            "supersonic",
            SAFEGUARDER,
            vec![
                ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
                ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
            ],
            "13521,38977,6462,56077",
        ),
    ];

    let mut failures = Vec::new();
    for (label, mv, p2, script, expect) in &cases {
        let (seed, _lines) = play(&gengar(mv), p2, script);
        if &seed.as_str() != expect {
            failures.push(format!("  {label}: port {seed} != sim {expect}"));
        }
    }
    assert!(
        failures.is_empty(),
        "CF1: the confusion family's draw count differs from the sim's:\n{}",
        failures.join("\n")
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CF2 — the EMISSION gates, one per disposition.
// ─────────────────────────────────────────────────────────────────────────────

fn assert_has(lines: &[String], needle: &str, what: &str) {
    assert!(
        lines.iter().any(|l| l == needle),
        "{what}: expected the line `{needle}`, got:\n{}",
        lines.join("\n")
    );
}

fn assert_lacks(lines: &[String], needle: &str, what: &str) {
    assert!(
        !lines.iter().any(|l| l.contains(needle)),
        "{what}: did NOT expect `{needle}`, got:\n{}",
        lines.join("\n")
    );
}

/// CF2 — a plain hit starts the confusion, for every member.
#[test]
fn confusion_family_plain_hit_starts_confusion() {
    for mv in ["confuseray", "supersonic", "sweetkiss", "teeterdance"] {
        let (_s, lines) = play(&gengar(mv), INERT, &[both(0, 0)]);
        assert_has(&lines, "|-start|p2a: Snorlax|confusion", &format!("CF2/{mv}"));
    }
}

/// CF3 — a SUBSTITUTE blocks the whole family: `[still]` + `-fail` on the USER, NO confusion.
///
/// 🚨 **THE PRE-FIX PORT APPLIED THE CONFUSION STRAIGHT THROUGH THE SUB.** None of the family
/// carries `bypasssub`, so the sim blocks at `onTryPrimaryHit`.
#[test]
fn a_substitute_blocks_every_confusion_move() {
    for mv in ["confuseray", "supersonic", "sweetkiss", "teeterdance"] {
        let (_s, lines) = play(
            &gengar(mv),
            SUBBER,
            &[
                ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
                ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
            ],
        );
        let what = format!("CF3/{mv}");
        assert_has(&lines, "|-fail|p1a: Gengar", &what);
        assert_lacks(&lines, "|-start|p2a: Snorlax|confusion", &what);
        // NON-VACUITY: the substitute really did go up, else this proves nothing.
        assert_has(&lines, "|-start|p2a: Snorlax|Substitute", &what);
    }
}

/// CF4 — OWN TEMPO is immune, and the emission carries the `confusion` token.
#[test]
fn own_tempo_is_immune_with_the_confusion_token() {
    let (_s, lines) = play(&gengar("supersonic"), OWN_TEMPO, &[both(0, 0)]);
    assert_has(
        &lines,
        "|-immune|p2a: Slowbro|confusion|[from] ability: Own Tempo",
        "CF4",
    );
    assert_lacks(&lines, "|-start|p2a: Slowbro|confusion", "CF4");
}

/// CF5 — SOUNDPROOF blocks Supersonic (the family's only `sound` move) with a DIFFERENT
/// emission form from Own Tempo's: **no `confusion` token**. Sweet Kiss is the control that
/// keeps the test from passing on a blanket "Soundproof blocks everything".
#[test]
fn soundproof_blocks_supersonic_but_not_sweet_kiss() {
    let (_s, lines) = play(&gengar("supersonic"), SOUNDPROOF, &[both(0, 0)]);
    assert_has(&lines, "|-immune|p2a: Voltorb|[from] ability: Soundproof", "CF5");
    assert_lacks(&lines, "|-start|p2a: Voltorb|confusion", "CF5");

    // The CONTROL: Sweet Kiss is not a sound move, so the same board must be confused.
    let (_s2, control) = play(&gengar("sweetkiss"), SOUNDPROOF, &[both(0, 0)]);
    assert_has(&control, "|-start|p2a: Voltorb|confusion", "CF5 control");
}

/// CF6 — SAFEGUARD wards the family off with `-activate|<target>|move: Safeguard`, and (unlike
/// the Substitute case) does NOT produce a `-fail` on the user.
#[test]
fn safeguard_wards_the_family_without_a_user_fail() {
    let (_s, lines) = play(
        &gengar("supersonic"),
        SAFEGUARDER,
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)),
        ],
    );
    assert_has(&lines, "|-activate|p2a: Snorlax|move: Safeguard", "CF6");
    assert_lacks(&lines, "|-start|p2a: Snorlax|confusion", "CF6");
    assert_lacks(&lines, "|-fail|", "CF6");
}

/// CF7 — TEETER DANCE's `allAdjacent` target still renders the FOE in the `|move|` announce.
/// A `status_move_announce_renders_user` that swept `allAdjacent` in would render the USER and
/// poke-env would key the line to the wrong mon.
#[test]
fn teeter_dance_announce_renders_the_foe_not_the_user() {
    let (_s, lines) = play(&gengar("teeterdance"), INERT, &[both(0, 0)]);
    assert_has(&lines, "|move|p1a: Gengar|Teeter Dance|p2a: Snorlax", "CF7");
}
