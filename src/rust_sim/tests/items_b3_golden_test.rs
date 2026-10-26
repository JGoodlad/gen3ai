//! The B3 ITEMS **golden differential** (`gen3_items_b3_v1`, gen3 full parity B3): Shell Bell,
//! Macho Brace, Mail, Mental Herb. Per row, against the sim's recording
//! (`harness/gen_items_b3_golden.js`; the settled spec is `harness/probe_items_b3.js`): the
//! post-script PRNG SEED, the FRAMING window (a Macho Brace lead's turn-0 order decides which lead's
//! weather wins) and the post-`|turn|1` lines, byte-equal. The boards: Shell Bell after a plain hit,
//! a multi-hit (the strikes' sum), a Substitute (no heal), Double-Edge (after the recoil),
//! Seismic Toss, a KO, Beat Up, Pursuit on a switch, the move Sleep Talk calls; Macho Brace turn
//! order, a speed tie it creates, the lead weather; Mail under Trick, Thief, Knock Off, Covet; Mental
//! Herb against Cute Charm.
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
    for line in include_str!("vectors/items_b3_golden.txt").lines() {
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
fn the_b3_items_match_showdown() {
    let rows = load();
    assert!(rows.len() >= 150, "golden shrank to {} rows", rows.len());
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
fn the_golden_covers_each_item() {
    let rows = load();
    let n = |pred: &dyn Fn(&Row) -> bool| rows.iter().filter(|r| pred(r)).count();
    let has = |r: &Row, s: &str| r.lines.iter().any(|l| l.contains(s));
    let bell = "[from] item: Shell Bell";
    assert!(n(&|r| has(r, bell)) >= 50, "Shell Bell heals");
    for b in ["sb_multihit", "sb_beatup", "sb_pursuit", "sb_doubleedge", "sb_fixed"] {
        assert!(n(&|r| r.name == b && has(r, bell)) >= 4, "{b}: Shell Bell heals");
    }
    // A Sleep Talk's CALLED move heals off its own damage.
    assert!(
        n(&|r| r.lines.iter().enumerate().any(|(i, l)| l.ends_with("|[from] Sleep Talk")
            && r.lines[i + 1..(i + 4).min(r.lines.len())].iter().any(|x| x.contains(bell)))) >= 2,
        "the move Sleep Talk calls heals"
    );
    // A Substitute strike heals nothing: no heal on the turn a sub absorbs.
    assert!(n(&|r| r.name == "sb_sub" && has(r, "|Substitute|[damage]") || r.name == "sb_sub" && has(r, "|-end|p2a: Jolteon|Substitute")) >= 4, "Substitute rows");
    assert!(n(&|r| r.name == "mb_lead_weather" && r.framing.iter().any(|l| l.starts_with("|-weather|SunnyDay"))) >= 4, "Macho Brace lead sun");
    assert!(n(&|r| r.name == "mb_lead_weather_control" && r.framing.iter().any(|l| l.starts_with("|-weather|RainDance|[from] ability: Drizzle"))
        && r.framing.last().map_or(false, |l| l == "|turn|1")) >= 4, "control rain");
    assert!(n(&|r| r.name.starts_with("mail_trick") && has(r, "|-fail|p1a: Alakazam")) >= 8, "Mail Trick failures");
    assert!(n(&|r| r.name == "mail_thief" && has(r, "|-item|p1a: Sneasel|Mail")) >= 4, "Thief takes Mail");
    assert!(n(&|r| r.name == "mail_covet" && has(r, "|-item|p1a: Sneasel|Mail")) >= 4, "Covet takes Mail");
    assert!(n(&|r| has(r, "[from] item: Mental Herb")) >= 3, "Mental Herb cures");
}
