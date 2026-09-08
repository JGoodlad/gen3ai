//! The FOCUS ENERGY **golden differential** (`gen3_focus_energy_move_v1`, ROUND 59).
//!
//! Focus Energy is NEVER-MISS and DRAW-FREE, so one seed would settle its own emission. The sweep
//! exists for the half that is not draw-free: the **crit consequence**. The move's entire point is
//! shifting the crit ratio from stage 0 (1/16) to stage +2 (1/4), and that only appears across
//! seeds — so each scenario casts, then attacks, and the vector records `|-crit|` lines.
//!
//! 🚨 **A FLAG THAT IS SET AND NEVER READ WOULD PASS AN EMISSION-ONLY TEST.** That is not a
//! hypothetical here: ROUND 57 found a move that had a named pin asserting its emissions while its
//! draw model was wrong for months. The CONTROL scenario is the same board with the cast replaced
//! by a Splash, and `focus_energy_golden_crit_rate_shows_the_volatile_is_read` compares the two
//! rates — so an engine that emitted the `-start` line and never touched the crit ratio FAILS.
//!
//! Per row it asserts, against the sim's own recording:
//!   * the post-decision **PRNG seed** (the draw-count proof), and
//!   * the filtered `|...|` **protocol lines**, byte-equal and in order.
//!
//! 🚨 **THE COVERAGE FLOORS BELOW ARE ENFORCED, NOT REPORTED.** A generator statistic would be a
//! coverage claim nobody could fail: if a future regeneration stopped producing miss rows, the
//! accuracy branch this vector exists to cover would silently vanish and every row would still
//! pass. The floors make that a test failure.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::turn::{Choice, ScriptDecision};

/// One recorded row.
struct Row {
    name: String,
    p1: String,
    p2: String,
    script: Vec<String>,
    seed_before: String,
    seed_after: String,
    lines: Vec<String>,
}

/// `PRNG.getSeed()` already returns the four 16-bit words as one comma STRING (the same shape
/// `BattleState::prng_seed()` re-emits), so this is a read, not a join. Asserted rather than
/// assumed: a future Showdown that returned an array here would otherwise silently compare a
/// debug-formatted array against a comma string and fail every row for the wrong reason.
fn seed_str(v: &Json) -> String {
    let s = v.as_str().expect("the golden's seed field is a comma STRING, not an array");
    assert_eq!(s.split(',').count(), 4, "a gen3 PRNG seed is four words, got {s:?}");
    s.to_string()
}

fn load() -> Vec<Row> {
    let raw = include_str!("vectors/focus_energy_golden.txt");
    let mut rows = Vec::new();
    for line in raw.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let v = Json::parse(line).expect("golden row is JSON");
        rows.push(Row {
            name: v.get("name").and_then(|x| x.as_str()).expect("name").to_string(),
            p1: v.get("p1").and_then(|x| x.as_str()).expect("p1").to_string(),
            p2: v.get("p2").and_then(|x| x.as_str()).expect("p2").to_string(),
            script: v
                .get("script")
                .and_then(|x| x.as_array())
                .expect("script")
                .iter()
                .map(|s| s.as_str().expect("script line").to_string())
                .collect(),
            seed_before: seed_str(v.get("seedBefore").expect("seedBefore")),
            seed_after: seed_str(v.get("seedAfter").expect("seedAfter")),
            lines: v
                .get("lines")
                .and_then(|x| x.as_array())
                .expect("lines")
                .iter()
                .map(|s| s.as_str().expect("line").to_string())
                .collect(),
        });
    }
    rows
}

/// `>p1 move K\n>p2 move J` → the port's `ScriptDecision`. The generator only ever emits that
/// one shape, so anything else is a generator change the test must NOT silently absorb.
fn decision(cmd: &str) -> ScriptDecision {
    let mut p1 = None;
    let mut p2 = None;
    for part in cmd.split('\n') {
        let toks: Vec<&str> = part.split_whitespace().collect();
        assert_eq!(
            toks.len(),
            3,
            "golden script line {part:?} is not the `>pN move K` shape this test understands"
        );
        assert_eq!(toks[1], "move", "golden script line {part:?} is not a move choice");
        let k: usize = toks[2].parse::<usize>().expect("slot number") - 1;
        match toks[0] {
            ">p1" => p1 = Some(Choice::Move(k)),
            ">p2" => p2 = Some(Choice::Move(k)),
            other => panic!("unknown side {other:?}"),
        }
    }
    ScriptDecision::both(p1.expect("p1 choice"), p2.expect("p2 choice"))
}

/// The same line filter the generator applied, so both sides are compared on the same alphabet.
fn keep(l: &str) -> bool {
    const PREFIXES: [&str; 12] = [
        "|move|", "|-start|", "|-end|", "|-fail|", "|-immune|", "|-activate|", "|-miss|",
        "|-crit|", "|-damage|", "|-heal|", "|turn|", "|-status|",
    ];
    PREFIXES.iter().any(|p| l.starts_with(p))
}

#[test]
fn focus_energy_golden_matches_showdown() {
    let rows = load();
    assert!(rows.len() >= 400, "golden shrank to {} rows", rows.len());
    let d = Dex::for_gen(3);

    let mut seed_fail = Vec::new();
    let mut line_fail = Vec::new();
    for r in &rows {
        let opts = BattleOptions {
            format_id: "gen3customgame".to_string(),
            seed: Some(r.seed_before.clone()),
            p1: PlayerOptions { name: "P1".into(), team: PackedTeam(r.p1.clone()) },
            p2: PlayerOptions { name: "P2".into(), team: PackedTeam(r.p2.clone()) },
        };
        let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
        let st = b.state_mut().expect("state");
        let script: Vec<ScriptDecision> = r.script.iter().map(|c| decision(c)).collect();
        let (_o, emitted) = st.run_full_battle_logged(&script, &d);
        // The generator slices the sim's log AFTER the first `|turn|1`, so the port must be
        // sliced identically — the pre-turn-1 FRAMING (teamsize / start / the switch-ins) is a
        // different window and comparing across windows is not evidence.
        let all: Vec<String> = emitted.into_iter().map(|l| l.0).collect();
        let start = all
            .iter()
            .position(|l| l == "|turn|1")
            .map(|i| i + 1)
            .expect("the port must emit a |turn|1 framing marker");
        let got: Vec<String> = all[start..].iter().filter(|l| keep(l)).cloned().collect();

        if st.prng_seed() != r.seed_after {
            seed_fail.push(format!(
                "  {} [{}]: port {} != sim {}",
                r.name,
                r.seed_before,
                st.prng_seed(),
                r.seed_after
            ));
        }
        if got != r.lines {
            let first = got
                .iter()
                .zip(r.lines.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(r.lines.len()));
            line_fail.push(format!(
                "  {} [{}] first divergence at line {first}:\n    port: {:?}\n    sim : {:?}",
                r.name,
                r.seed_before,
                got.get(first),
                r.lines.get(first)
            ));
        }
    }

    assert!(
        seed_fail.is_empty(),
        "FEG: {} of {} rows have the WRONG DRAW COUNT:\n{}",
        seed_fail.len(),
        rows.len(),
        seed_fail.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
    );
    assert!(
        line_fail.is_empty(),
        "FEG: {} of {} rows emit DIFFERENT BYTES:\n{}",
        line_fail.len(),
        rows.len(),
        line_fail.iter().take(6).cloned().collect::<Vec<_>>().join("\n")
    );
}

/// FEG-RATE — the CONTROL comparison that proves the volatile is READ, not merely SET.
///
/// Both scenario families share the board, the seeds and the turn-2 Tackle; they differ only in
/// whether turn 1 cast Focus Energy or Splashed. gen-3 crit stage 0 is 1/16 and stage +2 is 1/4,
/// so the two rates must be far apart. This is asserted on the GOLDEN (the sim's own recording),
/// which makes it a statement about the vector's fitness; `focus_energy_test.rs::FE3` asserts the
/// same shape against the PORT.
#[test]
fn focus_energy_golden_crit_rate_shows_the_volatile_is_read() {
    let rows = load();
    let rate = |name: &str| -> (usize, usize) {
        let all: Vec<&Row> = rows.iter().filter(|r| r.name == name).collect();
        let crits = all.iter().filter(|r| r.lines.iter().any(|l| l.starts_with("|-crit|"))).count();
        (crits, all.len())
    };
    let (fe, fe_n) = rate("focusenergy-then-attack");
    let (ctl, ctl_n) = rate("control-nocast-then-attack");
    assert!(fe_n >= 100 && ctl_n >= 100, "FEG-RATE: too few rows ({fe_n} / {ctl_n}) to compare rates");
    assert!(
        fe > ctl * 2,
        "FEG-RATE: Focus Energy crit {fe}/{fe_n} vs control {ctl}/{ctl_n} — the vector cannot \
         show the volatile is READ (expected roughly 1/4 vs 1/16)"
    );
    // The `-start` line must actually be in the vector, else the rate half is measuring a board
    // where the move never resolved.
    let starts = rows
        .iter()
        .filter(|r| r.lines.iter().any(|l| l.contains("|move: Focus Energy")))
        .count();
    assert!(starts >= 200, "FEG-RATE: only {starts} rows carry the `move: Focus Energy` start line");
    // And the already-up `[still]`+`-fail` branch must be present.
    let fails = rows
        .iter()
        .filter(|r| r.lines.iter().any(|l| l.starts_with("|-fail|p1a: Machamp")))
        .count();
    assert!(fails >= 100, "FEG-RATE: only {fails} rows reach the already-up `-fail` branch");
    // And the SNATCH steal.
    let snatched = rows
        .iter()
        .filter(|r| r.lines.iter().any(|l| l.contains("move: Snatch")))
        .count();
    assert!(snatched >= 100, "FEG-RATE: only {snatched} rows reach the Snatch steal");
}
