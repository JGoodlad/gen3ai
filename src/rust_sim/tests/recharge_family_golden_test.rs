//! The RECHARGE family **golden differential** (`gen3_recharge_family_v1`, gen3 full parity B2).
//!
//! Blast Burn, Frenzy Plant and Hydro Cannon are Hyper Beam with another type (gen3 resolves all
//! four to 150 BP / 90 accuracy / `self.volatileStatus: 'mustrecharge'` and no handler of their
//! own). Hyper Beam rides every board as the CONTROL. Per row, against the sim's recording
//! (`harness/gen_recharge_family_golden.js`): the post-script PRNG SEED and the post-`|turn|1`
//! lines byte-equal, over a plain cycle (hits, recharges, accuracy-90 misses), Protect, a
//! Substitute, a KO with the lock held across the forced switch, the type-specific blockers
//! (Flash Fire, Water Absorb, a Ghost), a paralysed recharge turn, and Truant.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::turn::{Choice, ScriptDecision};

struct Row {
    name: String,
    mv: String,
    p1: String,
    p2: String,
    script: Vec<ScriptDecision>,
    seed_before: String,
    seed_after: String,
    lines: Vec<String>,
}

/// `move K` / `switch K` (1-based) → the port's 0-based `Choice`; JSON `null` → no choice.
fn choice(v: &Json) -> Option<Choice> {
    let c = v.as_str()?;
    let (kind, n) = c.split_once(' ').unwrap_or_else(|| panic!("choice {c:?}"));
    let k = n.parse::<usize>().expect("slot") - 1;
    Some(match kind {
        "move" => Choice::Move(k),
        "switch" => Choice::Switch(k),
        other => panic!("choice kind {other:?}"),
    })
}

fn load() -> Vec<Row> {
    let mut rows = Vec::new();
    for line in include_str!("vectors/recharge_family_golden.txt").lines() {
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
                let pair = d.as_array().expect("a decision is [p1, p2]");
                ScriptDecision { p1: choice(&pair[0]), p2: choice(&pair[1]) }
            })
            .collect();
        rows.push(Row {
            name: s("name"),
            mv: s("move"),
            p1: s("p1"),
            p2: s("p2"),
            script,
            seed_before: s("seedBefore"),
            seed_after: s("seedAfter"),
            lines: v
                .get("lines")
                .and_then(|x| x.as_array())
                .expect("lines")
                .iter()
                .map(|x| x.as_str().expect("line").to_string())
                .collect(),
        });
    }
    rows
}

fn kept(l: &str) -> bool {
    !(l.is_empty() || l.starts_with("|t:|") || l.starts_with("|debug|") || l.starts_with("|error|"))
}

#[test]
fn the_recharge_family_matches_showdown() {
    let rows = load();
    assert!(rows.len() >= 200, "golden shrank to {} rows", rows.len());
    let d = Dex::for_gen(3);
    let mut fails = Vec::new();
    for r in &rows {
        let opts = BattleOptions {
            format_id: "gen3customgame".into(),
            seed: Some(r.seed_before.clone()),
            p1: PlayerOptions { name: "P1".into(), team: PackedTeam(r.p1.clone()) },
            p2: PlayerOptions { name: "P2".into(), team: PackedTeam(r.p2.clone()) },
        };
        let mut b = Battle::start_with_switchins(&opts, &d).unwrap_or_else(|e| panic!("{} {}: {e}", r.name, r.mv));
        let st = b.state_mut().expect("state");
        let (_o, emitted) = st.run_full_battle_logged(&r.script, &d);
        let all: Vec<String> = emitted.into_iter().map(|l| l.0).collect();
        let t1 = all.iter().position(|l| l == "|turn|1").expect("the port emits |turn|1");
        let got: Vec<String> = all[t1 + 1..].iter().filter(|l| kept(l)).cloned().collect();
        let tag = format!("{} {} [{}]", r.name, r.mv, r.seed_before);
        if st.prng_seed() != r.seed_after {
            fails.push(format!("{tag}: SEED port {} != sim {}", st.prng_seed(), r.seed_after));
        }
        if got != r.lines {
            let i = got.iter().zip(r.lines.iter()).position(|(a, b)| a != b).unwrap_or(got.len().min(r.lines.len()));
            fails.push(format!("{tag}: lines differ at {i}: port {:?} / sim {:?}", got.get(i), r.lines.get(i)));
        }
    }
    assert!(fails.is_empty(), "{} of {} rows diverge:\n{}", fails.len(), rows.len(), fails.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
}

/// The vector must carry each branch for each SIBLING (not only the Hyper Beam control).
#[test]
fn the_golden_covers_each_sibling() {
    let rows = load();
    for mv in ["blastburn", "frenzyplant", "hydrocannon"] {
        let n = |pred: &dyn Fn(&Row) -> bool| rows.iter().filter(|r| r.mv == mv && pred(r)).count();
        let has = |r: &Row, p: &str| r.lines.iter().any(|l| l.starts_with(p));
        assert!(n(&|r| has(r, "|-mustrecharge|")) >= 20, "{mv}: mustrecharge rows");
        assert!(n(&|r| r.lines.iter().any(|l| l.starts_with("|cant|") && l.ends_with("|recharge"))) >= 20, "{mv}: recharge cant rows");
        assert!(n(&|r| has(r, "|-miss|")) >= 3, "{mv}: miss rows (accuracy 90)");
        assert!(n(&|r| r.name == "ko" && r.script.iter().any(|d| d.p1.is_none())) >= 3, "{mv}: KO + forced switch rows");
        assert!(n(&|r| r.name == "sub" && r.lines.iter().any(|l| l.contains("Substitute"))) >= 3, "{mv}: Substitute rows");
    }
    let absorbed = |mv: &str, needle: &str| {
        rows.iter().filter(|r| r.mv == mv && r.lines.iter().any(|l| l.contains(needle))).count()
    };
    assert!(absorbed("blastburn", "Flash Fire") >= 3, "Flash Fire absorbs Blast Burn");
    assert!(absorbed("hydrocannon", "Water Absorb") >= 3, "Water Absorb absorbs Hydro Cannon");
    // An absorbed or immune hit locks nothing.
    for r in rows.iter().filter(|r| ["flashfire", "waterabsorb", "ghost"].contains(&r.name.as_str())) {
        let absorbed = r.lines.iter().any(|l| l.contains("Flash Fire") || l.contains("Water Absorb") || l.starts_with("|-immune|"));
        if absorbed && !r.lines.iter().any(|l| l.starts_with("|-damage|p2a")) {
            assert!(!r.lines.iter().any(|l| l.starts_with("|-mustrecharge|")), "{} {}: an absorbed hit locked the user", r.name, r.mv);
        }
    }
}
