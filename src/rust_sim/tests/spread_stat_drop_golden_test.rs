//! The SPREAD STAT-DROP **golden differential** (`gen3_spread_stat_drop_v1`, ROUND 58).
//!
//! The five `allAdjacentFoes` stat-drop moves — leer / growl / tailwhip / stringshot /
//! sweetscent — reached the engine as a pure DATA change: the extractor's `_stat_drop_boosts`
//! target gate widened, and not one engine line was written for them. **That is precisely why
//! they need a differential.** "The existing arm should serve them" is a claim about behaviour;
//! the only thing that settles it is running both engines side by side over the whole
//! disposition matrix — including the two the arm handles specially (Soundproof for the sound
//! member, Growl; a Substitute's `[still]`+`-fail`) and the boost-immunity abilities.
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
    let raw = include_str!("vectors/spread_stat_drop_golden.txt");
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
    const PREFIXES: [&str; 13] = [
        "|move|", "|-start|", "|-end|", "|-fail|", "|-immune|", "|-activate|", "|-miss|",
        "|-boost|", "|-unboost|", "|-status|", "|-damage|", "|-heal|", "|turn|",
    ];
    PREFIXES.iter().any(|p| l.starts_with(p))
}

#[test]
fn spread_stat_drop_golden_matches_showdown() {
    let rows = load();
    assert!(rows.len() >= 900, "golden shrank to {} rows", rows.len());
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
        "SSD: {} of {} rows have the WRONG DRAW COUNT:\n{}",
        seed_fail.len(),
        rows.len(),
        seed_fail.iter().take(10).cloned().collect::<Vec<_>>().join("\n")
    );
    assert!(
        line_fail.is_empty(),
        "SSD: {} of {} rows emit DIFFERENT BYTES:\n{}",
        line_fail.len(),
        rows.len(),
        line_fail.iter().take(6).cloned().collect::<Vec<_>>().join("\n")
    );
}

/// SSD-FLOORS — the vector must actually contain the branches it exists to cover.
///
/// Each is ENFORCED, not reported. A generator statistic would be a coverage claim nobody could
/// fail: ROUND 57's first draft produced 0 Protect rows and every one of those rows still passed
/// the byte comparison, because the port and the sim agree perfectly about a scenario that tests
/// nothing. Only a floor tells a green run from a vacuous one.
#[test]
fn spread_stat_drop_golden_covers_its_branches() {
    let rows = load();
    let any = |needle: &str| {
        rows.iter().filter(|r| r.lines.iter().any(|l| l.contains(needle))).count()
    };

    // The DELTA-0 line at the −6 floor — the cap-before-TryBoost form ROUND 55 pinned.
    let floor = rows
        .iter()
        .filter(|r| r.lines.iter().any(|l| l.starts_with("|-unboost|") && l.ends_with("|0")))
        .count();
    let unboost = rows
        .iter()
        .filter(|r| r.lines.iter().any(|l| l.starts_with("|-unboost|")))
        .count();
    let misses = any("|-miss|");
    let clear_body = any("[from] ability: Clear Body");
    let hyper_cutter = any("[from] ability: Hyper Cutter");
    let soundproof = any("[from] ability: Soundproof");
    let protect = any("|Protect");
    let user_fail = any("|-fail|p1a: Gengar");
    let evasion = any("|evasion|");

    assert!(unboost >= 400, "floor: |-unboost| rows = {unboost} (< 400)");
    assert!(floor >= 50, "floor: DELTA-0 (−6 cap) rows = {floor} (< 50)");
    assert!(misses >= 10, "floor: miss rows = {misses} (< 10) — String Shot is accuracy 95");
    assert!(clear_body >= 20, "floor: Clear Body rows = {clear_body} (< 20)");
    assert!(hyper_cutter >= 10, "floor: Hyper Cutter rows = {hyper_cutter} (< 10)");
    assert!(soundproof >= 10, "floor: Soundproof rows = {soundproof} (< 10) — Growl is the sound member");
    assert!(protect >= 20, "floor: Protect rows = {protect} (< 20)");
    assert!(user_fail >= 20, "floor: substitute `-fail` on the USER rows = {user_fail} (< 20)");
    assert!(evasion >= 20, "floor: evasion-drop rows = {evasion} (< 20) — Sweet Scent is gen3's only foe-evasion drop");
}
