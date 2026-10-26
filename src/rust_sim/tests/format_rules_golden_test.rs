//! The gen-3 FORMAT **golden differential** (`gen3_format_rules_v1`, gen3 full parity B1).
//!
//! Every gen-3 singles format the engine runs, through six boards (`harness/gen_format_rules_golden.js`):
//! a status application, a second sleep, a second freeze, a self-Rest, Beat Up and a Sand Veil
//! Dugtrio. Per row, against the sim's own recording: the post-script PRNG SEED (the draw count:
//! only gen3ou / gen3uubl / gen3draft draw the 2-clause SetStatus shuffle), the post-`|turn|1`
//! lines byte-equal, the FRAMING window through `bridge::reframe` (the format's `|tier|` and
//! `|rule|` lines), and on the Dugtrio rows the endTurn `maybeTrapped`.
//!
//! Before this round the engine ran every id but gen3customgame as gen3ou, so every Ubers / UU /
//! RU / NU / PU / ZU / LC / Random Battle row carrying a status drew one call too many, the
//! Freeze Clause blocked a freeze those formats allow, and the framing read `[Gen 3] Custom Game`.
//! The floors below are ENFORCED: a regeneration that lost a branch fails here.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::turn::{Choice, ScriptDecision};

struct Row {
    name: String,
    format: String,
    p1: String,
    p2: String,
    script: Vec<(String, String)>,
    seed_before: String,
    seed_after: String,
    framing: Vec<String>,
    lines: Vec<String>,
    maybe_trapped: bool,
}

fn strs(v: &Json) -> Vec<String> {
    v.as_array().expect("array").iter().map(|x| x.as_str().expect("string").to_string()).collect()
}

fn load() -> Vec<Row> {
    let mut rows = Vec::new();
    for line in include_str!("vectors/format_rules_golden.txt").lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let v = Json::parse(line).expect("golden row is JSON");
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or_else(|| panic!("row lacks {k}")).to_string();
        let script = v
            .get("script")
            .and_then(|x| x.as_array())
            .expect("script")
            .iter()
            .map(|d| {
                let pair = strs(d);
                assert_eq!(pair.len(), 2, "a decision is [p1, p2]");
                (pair[0].clone(), pair[1].clone())
            })
            .collect();
        rows.push(Row {
            name: s("name"),
            format: s("format"),
            p1: s("p1"),
            p2: s("p2"),
            script,
            seed_before: s("seedBefore"),
            seed_after: s("seedAfter"),
            framing: strs(v.get("framing").expect("framing")),
            lines: strs(v.get("lines").expect("lines")),
            maybe_trapped: v.get("maybeTrapped").and_then(|x| x.as_bool()).expect("maybeTrapped"),
        });
    }
    rows
}

/// `move K` / `switch K` (1-based, the sim's choice syntax) → the port's 0-based `Choice`.
fn choice(c: &str) -> Choice {
    let (kind, n) = c.split_once(' ').unwrap_or_else(|| panic!("choice {c:?}"));
    let k = n.parse::<usize>().expect("slot") - 1;
    match kind {
        "move" => Choice::Move(k),
        "switch" => Choice::Switch(k),
        other => panic!("choice kind {other:?}"),
    }
}

/// The lines the generator drops on its side too: the wall clock, blanks, debug and error.
fn kept(l: &str) -> bool {
    !(l.is_empty() || l.starts_with("|t:|") || l.starts_with("|debug|") || l.starts_with("|error|"))
}

#[test]
fn every_format_matches_showdown() {
    let rows = load();
    assert!(rows.len() >= 250, "golden shrank to {} rows", rows.len());
    let d = Dex::for_gen(3);
    let mut fails = Vec::new();
    for r in &rows {
        let opts = BattleOptions {
            format_id: r.format.clone(),
            seed: Some(r.seed_before.clone()),
            p1: PlayerOptions { name: "P1".into(), team: PackedTeam(r.p1.clone()) },
            p2: PlayerOptions { name: "P2".into(), team: PackedTeam(r.p2.clone()) },
        };
        let mut b = Battle::start_with_switchins(&opts, &d).unwrap_or_else(|e| panic!("{} {}: {e}", r.format, r.name));
        let st = b.state_mut().expect("state");
        let script: Vec<ScriptDecision> =
            r.script.iter().map(|(a, c)| ScriptDecision::both(choice(a), choice(c))).collect();
        let (_o, emitted) = st.run_full_battle_logged(&script, &d);
        let framed = pokesim::bridge::reframe(&emitted, &r.format);
        let t1 = framed.iter().position(|l| l == "|turn|1").expect("the port emits |turn|1");
        let g = framed.iter().position(|l| l == "|gametype|singles").expect("the port emits |gametype|");
        let framing: Vec<String> = framed[g..=t1].iter().filter(|l| kept(l)).cloned().collect();
        let lines: Vec<String> = framed[t1 + 1..].iter().filter(|l| kept(l)).cloned().collect();
        let tag = format!("{} {} [{}]", r.format, r.name, r.seed_before);
        if st.prng_seed() != r.seed_after {
            fails.push(format!("{tag}: SEED port {} != sim {}", st.prng_seed(), r.seed_after));
        }
        for (what, got, want) in [("framing", &framing, &r.framing), ("lines", &lines, &r.lines)] {
            if got != want {
                let i = got.iter().zip(want.iter()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                fails.push(format!("{tag}: {what} differ at {i}: port {:?} / sim {:?}", got.get(i), want.get(i)));
            }
        }
        if r.name == "leak" && st.is_maybe_trapped(0, &d) != r.maybe_trapped {
            fails.push(format!("{tag}: maybeTrapped port {} != sim {}", !r.maybe_trapped, r.maybe_trapped));
        }
    }
    assert!(fails.is_empty(), "{} of {} rows diverge:\n{}", fails.len(), rows.len(), fails.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
}

/// The vector must carry the branches it exists to cover, per format where it matters.
#[test]
fn the_golden_covers_its_branches() {
    let rows = load();
    let n = |pred: &dyn Fn(&Row) -> bool| rows.iter().filter(|r| pred(r)).count();
    let has = |r: &Row, l: &str| r.lines.iter().any(|x| x == l);
    for f in ["gen3ou", "gen3uubl", "gen3draft"] {
        assert!(n(&|r| r.format == f && has(r, "|-message|Freeze Clause activated.")) >= 1, "{f}: no blocked second freeze");
    }
    for f in ["gen3ubers", "gen3uu", "gen3ru", "gen3nu", "gen3pu", "gen3zu", "gen3lc", "gen3randombattle", "gen3customgame"] {
        assert!(n(&|r| r.format == f && r.name == "freeze2" && has(r, "|-status|p2a: Blissey|frz")) >= 1, "{f}: no landed second freeze");
        assert!(n(&|r| r.format == f && has(r, "|-message|Freeze Clause activated.")) == 0, "{f}: a Freeze Clause block");
    }
    assert!(n(&|r| has(r, "|-message|Sleep Clause Mod activated.")) >= 40, "sleep clause blocks");
    assert!(
        n(&|r| r.format == "gen3customgame" && r.name == "sleep2" && r.lines.iter().any(|l| l.starts_with("|-status|p2a: Blissey|slp"))) >= 1,
        "customgame second sleep"
    );
    assert!(n(&|r| r.lines.iter().any(|l| l.starts_with("|-activate|") && l.contains("Beat Up"))) >= 1, "Beat Up -activate");
    assert!(n(&|r| r.name == "leak" && r.maybe_trapped) >= 30, "leak true");
    for f in ["gen3uu", "gen3ru", "gen3customgame"] {
        assert!(n(&|r| r.format == f && r.name == "leak" && !r.maybe_trapped) >= 1, "{f}: leak false");
    }
    assert!(n(&|r| r.format == "gen3lc" && r.framing.iter().any(|l| l.contains(", L5"))) >= 1, "LC level-5 details");
    // Every format's status rows: the draw count is the point.
    let formats: std::collections::BTreeSet<&str> = rows.iter().map(|r| r.format.as_str()).collect();
    assert_eq!(formats.len(), 12, "formats in the vector: {formats:?}");
}
