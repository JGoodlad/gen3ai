//! The ATTRACT move **golden differential** (`gen3_attract_move_v1`, gen3 full parity B9). Per row,
//! against the sim's recording (`harness/gen_attract_move_golden.js`; spec
//! `harness/probe_attract_move.js`): the post-script PRNG SEED, the FRAMING window and the
//! post-`|turn|1` lines, byte-equal. Boards: a plain M-into-F cast and the immobilize rolls after it,
//! same gender, genderless, a second cast, Oblivious, a Substitute and Safeguard (neither blocks),
//! Protect on a compatible and on an immune target, evasion misses on both, a Mental Herb holder,
//! and the source and the holder switching out.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::turn::{Choice, ScriptDecision};

struct Row {
    name: String,
    p1: String,
    p2: String,
    script: Vec<ScriptDecision>,
    seed_before: String,
    seed_after: String,
    framing: Vec<String>,
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
    for line in include_str!("vectors/attract_move_golden.txt").lines() {
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
            p1: s("p1"),
            p2: s("p2"),
            script,
            seed_before: s("seedBefore"),
            seed_after: s("seedAfter"),
            framing: v
                .get("framing")
                .and_then(|x| x.as_array())
                .expect("framing")
                .iter()
                .map(|x| x.as_str().expect("line").to_string())
                .collect(),
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
fn the_attract_move_matches_showdown() {
    let rows = load();
    assert!(rows.len() >= 100, "golden shrank to {} rows", rows.len());
    let d = Dex::for_gen(3);
    let mut fails = Vec::new();
    for r in &rows {
        let opts = BattleOptions {
            format_id: "gen3customgame".into(),
            seed: Some(r.seed_before.clone()),
            p1: PlayerOptions { name: "P1".into(), team: PackedTeam(r.p1.clone()) },
            p2: PlayerOptions { name: "P2".into(), team: PackedTeam(r.p2.clone()) },
        };
        let mut b = Battle::start_with_switchins(&opts, &d).unwrap_or_else(|e| panic!("{}: {e}", r.name));
        let st = b.state_mut().expect("state");
        let (_o, emitted) = st.run_full_battle_logged(&r.script, &d);
        let all: Vec<String> = emitted.into_iter().map(|l| l.0).collect();
        let t1 = all.iter().position(|l| l == "|turn|1").expect("the port emits |turn|1");
        let g = all.iter().position(|l| l == "|gametype|singles").expect("the port emits |gametype|");
        let framing: Vec<String> = all[g..=t1].iter().filter(|l| kept(l)).cloned().collect();
        let got: Vec<String> = all[t1 + 1..].iter().filter(|l| kept(l)).cloned().collect();
        let tag = format!("{} [{}]", r.name, r.seed_before);
        if st.prng_seed() != r.seed_after {
            fails.push(format!("{tag}: SEED port {} != sim {}", st.prng_seed(), r.seed_after));
        }
        for (what, got, want) in [("framing", &framing, &r.framing), ("lines", &got, &r.lines)] {
            if got != want {
                let i = got.iter().zip(want.iter()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                fails.push(format!("{tag}: {what} differ at {i}: port {:?} / sim {:?}", got.get(i), want.get(i)));
            }
        }
    }
    assert!(fails.is_empty(), "{} of {} rows diverge:\n{}", fails.len(), rows.len(), fails.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
}

/// The vector must carry each branch it exists to cover (ENFORCED floors).
#[test]
fn the_golden_covers_each_branch() {
    let rows = load();
    let n = |pred: &dyn Fn(&Row) -> bool| rows.iter().filter(|r| pred(r)).count();
    let has = |r: &Row, s: &str| r.lines.iter().any(|l| l.contains(s));
    assert!(n(&|r| has(r, "|-start|p2a: Nidoqueen|Attract")) >= 40, "landed casts");
    assert!(n(&|r| r.name == "same_gender" && has(r, "|-immune|p2a: Nidoking")) >= 4, "same gender -immune");
    assert!(n(&|r| r.name == "genderless" && has(r, "|-immune|p2a: Magneton")) >= 4, "genderless -immune");
    assert!(n(&|r| r.name == "evasive_m" && has(r, "|-immune|p2a: Nidoking")) >= 4, "an immune target's miss shows -immune");
    assert!(n(&|r| r.name == "evasive_f" && has(r, "|-miss|p1a: Nidoking")) >= 4, "a compatible target's miss");
    assert!(n(&|r| r.name == "twice" && has(r, "|-fail|p1a: Nidoking")) >= 4, "a second cast fails");
    assert!(n(&|r| r.name == "oblivious" && has(r, "|-fail|p1a: Nidoking")) >= 4, "Oblivious fails the move");
    assert!(n(&|r| r.name == "protect_immune" && has(r, "|-activate|p2a: Magneton|Protect")) >= 2, "Protect before the immunity");
    assert!(n(&|r| r.name == "substitute" && has(r, "|-start|p2a: Nidoqueen|Attract")) >= 4, "a Substitute does not block it");
    assert!(n(&|r| r.name == "safeguard" && has(r, "|-start|p2a: Nidoqueen|Attract")) >= 4, "Safeguard does not block it");
    assert!(n(&|r| has(r, "|cant|p2a: Nidoqueen|Attract")) >= 20, "immobilized turns");
    assert!(n(&|r| has(r, "[from] item: Mental Herb")) >= 4, "Mental Herb cures the move's infatuation");
    assert!(n(&|r| r.name == "source_leaves" && has(r, "|-end|p2a: Nidoqueen|Attract|[silent]")) >= 4, "the source leaving ends it");
}
