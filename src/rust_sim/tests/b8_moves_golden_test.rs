//! The B8 MOVES **golden differential** (`gen3_b8_moves_v1`, gen3 full parity B8): Bide, Magnitude,
//! Psywave, Present, Triple Kick and the OHKO moves (Fissure, Horn Drill, Guillotine, Sheer Cold). Per
//! row, against the sim's recording (`harness/gen_b8_moves_golden.js`, the lines as SENT; the settled
//! spec is `harness/probe_b8_moves.js`): the post-script PRNG SEED, the FRAMING window and the
//! post-`|turn|1` lines, byte-equal, in gen3customgame. The boards: Bide (cast, wait, release into a
//! foe / Ghost / Protect / Substitute / Wonder Guard / Rough Skin / a same-turn Fly, nothing stored,
//! multi-hit and Future Sight storage, its own sub, every cant that ends it, the faint-skipped release
//! ended by the residual, Roar, Pressure, Shell Bell, Encore, Choice Band, Counter, Endure, King's
//! Rock's self-flinch, a +evasion self-miss, a mirror), Magnitude (every magnitude, Flying / Levitate
//! immunity, Dig, Fly, Protect, a sub, a Bright Powder miss, Sleep Talk, King's Rock), Psywave (levels
//! 100 / 50 / 37 / 5 / 1, Dark, Wonder Guard, a sub, Protect, King's Rock, Focus Band), Present (every
//! branch, the heal's full-HP and sub fails, Ghost, Wonder Guard, Protect, Fly), Triple Kick (1 / 2 /
//! 3 strikes and a first-strike miss, Ghost, a sub, a KO with Static, King's Rock, Bright Powder,
//! Rough Skin, Protect) and the OHKO moves (the level gate, a hit, a miss, Ghost / Flying, Sturdy, a
//! sub broken and held, Endure, Focus Band, Protect, Wonder Guard, Dig, Bright Powder, Static).
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::turn::{Choice, ScriptDecision};

struct Row {
    name: String,
    format: String,
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
    for line in include_str!("vectors/b8_moves_golden.txt").lines() {
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
            format: s("format"),
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
fn the_b8_moves_match_showdown() {
    let rows = load();
    assert!(rows.len() >= 750, "golden shrank to {} rows", rows.len());
    let d = Dex::for_gen(3);
    let mut fails = Vec::new();
    for r in &rows {
        let opts = BattleOptions {
            format_id: r.format.clone(),
            seed: Some(r.seed_before.clone()),
            p1: PlayerOptions { name: "P1".into(), team: PackedTeam(r.p1.clone()) },
            p2: PlayerOptions { name: "P2".into(), team: PackedTeam(r.p2.clone()) },
        };
        let mut b = Battle::start_with_switchins(&opts, &d).unwrap_or_else(|e| panic!("{}: {e}", r.name));
        let st = b.state_mut().expect("state");
        let (_o, emitted) = st.run_full_battle_logged(&r.script, &d);
        // The format's own `|tier|` / `|rule|` frames (`bridge::reframe`, as the format-rules
        // golden does); a no-op for gen3customgame.
        let all: Vec<String> = pokesim::bridge::reframe(&emitted, &r.format);
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

/// The vector must carry each branch it exists to cover (ENFORCED floors, per move and per
/// branch; the generator refuses a zero, this test refuses a thinned vector).
#[test]
fn the_golden_covers_each_move_and_branch() {
    let rows = load();
    let n = |pred: &dyn Fn(&Row) -> bool| rows.iter().filter(|r| pred(r)).count();
    let has = |r: &Row, s: &str| r.lines.iter().any(|l| l.contains(s));
    let pair = |r: &Row, a: &dyn Fn(&str) -> bool, b: &dyn Fn(&str) -> bool| {
        r.lines.windows(2).any(|w| a(&w[0]) && b(&w[1]))
    };
    let release = |l: &str| l.starts_with("|-end|") && l.ends_with("|move: Bide");
    // BIDE
    assert!(n(&|r| has(r, "|-start|") && has(r, "|move: Bide")) >= 200, "Bide casts");
    assert!(n(&|r| r.lines.iter().any(|l| l.starts_with("|-activate|") && l.ends_with("|move: Bide"))) >= 150, "Bide waits");
    assert!(n(&|r| pair(r, &release, &|b| b.starts_with("|-damage|"))) >= 100, "Bide releases that hit");
    assert!(n(&|r| pair(r, &release, &|b| b.starts_with("|-fail|"))) >= 8, "Bide releases with nothing stored");
    assert!(n(&|r| pair(r, &release, &|b| b.starts_with("|-immune|") && !b.contains("[from]"))) >= 6, "a release into a Ghost");
    assert!(n(&|r| pair(r, &release, &|b| b.ends_with("|Protect"))) >= 4, "a release into Protect");
    assert!(n(&|r| pair(r, &release, &|b| b.ends_with("ability: Wonder Guard"))) >= 4, "a release into Wonder Guard");
    assert!(n(&|r| pair(r, &release, &|b| b.contains("|Substitute"))) >= 4, "a release into a Substitute");
    assert!(n(&|r| pair(r, &release, &|b| b.starts_with("|-miss|")) && has(r, "|Fly||[still]|[miss]")) >= 4, "a release into a same-turn Fly");
    assert!(n(&|r| r.lines.iter().any(|l| l.contains("|Bide|") && l.ends_with("|[miss]"))) >= 2, "a +evasion Bide misses its user");
    assert!(n(&|r| pair(r, &|a| a.starts_with("|cant|"), &|b| b.ends_with("|move: Bide|[silent]"))) >= 15, "a cant ends Bide");
    assert!(n(&|r| pair(r, &|a| a.ends_with("[from] confusion"), &|b| b.ends_with("|move: Bide|[silent]"))) >= 4, "a confusion self-hit ends Bide");
    assert!(n(&|r| r.name == "bd_faint_skip" && has(r, "|move: Bide|[silent]") && !r.lines.iter().any(|l| release(l))) >= 6, "the residual ends a faint-skipped Bide");
    assert!(n(&|r| r.name == "bd_futuresight" && pair(r, &release, &|b| b.starts_with("|-damage|"))) >= 6, "a stored Future Sight released");
    assert!(n(&|r| r.name == "bd_roughskin" && has(r, "ability: Rough Skin")) >= 6, "a release procs Rough Skin");
    assert!(n(&|r| r.name == "bd_encore" && has(r, "|-start|p1a: Raichu|Encore")) >= 6, "an Encored Bide");
    for b in ["bd_kingsrock", "bd_mirror", "bd_pressure", "bd_shellbell", "bd_choiceband", "bd_counter", "bd_endure", "bd_roar", "bd_imprison", "bd_disable", "bd_para", "bd_sleep"] {
        assert!(n(&|r| r.name == b) >= 8, "{b} rows");
    }
    // MAGNITUDE — every magnitude, the -activate after the announce in every branch.
    for (m, floor) in [(4, 4), (5, 10), (6, 20), (7, 30), (8, 20), (9, 4), (10, 4)] {
        assert!(n(&|r| has(r, &format!("|move: Magnitude|{m}"))) >= floor, "Magnitude {m}");
    }
    assert!(n(&|r| r.lines.windows(2).any(|w| w[0].contains("|Magnitude|") && w[0].ends_with("[miss]") && w[1].contains("move: Magnitude"))) >= 8, "a missed Magnitude");
    assert!(n(&|r| pair(r, &|a| a.contains("move: Magnitude"), &|b| b.starts_with("|-immune|"))) >= 12, "Magnitude into Flying / Levitate");
    assert!(n(&|r| pair(r, &|a| a.contains("move: Magnitude"), &|b| b.ends_with("|Protect"))) >= 6, "Magnitude into Protect");
    assert!(n(&|r| r.name == "mg_dig" && pair(r, &|a| a.contains("move: Magnitude"), &|b| b.starts_with("|-damage|"))) >= 6, "Magnitude into Dig");
    assert!(n(&|r| r.lines.iter().any(|l| l.contains("|Magnitude|") && l.contains("[from] Sleep Talk"))) >= 3, "Sleep Talk calls Magnitude");
    // PSYWAVE
    assert!(n(&|r| pair(r, &|a| a.contains("|Psywave|"), &|b| b.starts_with("|-damage|"))) >= 60, "Psywave hits");
    assert!(n(&|r| r.lines.iter().any(|l| l.contains("|Psywave|") && l.ends_with("[miss]"))) >= 15, "Psywave misses");
    for b in ["pw_l50", "pw_l37", "pw_l5", "pw_l1"] {
        assert!(n(&|r| r.name == b && has(r, "|-damage|")) >= 6, "{b}: a below-100 Psywave");
    }
    assert!(n(&|r| r.name == "pw_dark" && has(r, "|-immune|")) >= 6, "Psywave into Dark");
    assert!(n(&|r| r.name == "pw_sub" && pair(r, &|a| a.contains("|Psywave|"), &|b| b.contains("|Substitute"))) >= 6, "Psywave into a Substitute");
    // PRESENT — every branch.
    assert!(n(&|r| pair(r, &|a| a.contains("|Present|"), &|b| b.starts_with("|-heal|") && b.matches('|').count() == 3)) >= 8, "Present heals");
    assert!(n(&|r| pair(r, &|a| a.ends_with("|Present||[still]"), &|b| b.ends_with("|heal"))) >= 6, "Present's heal fails at full HP");
    assert!(n(&|r| pair(r, &|a| a.ends_with("|Present||[still]"), &|b| b.starts_with("|-fail|p1a: ") && b.matches('|').count() == 2)) >= 2, "Present's heal fails into a Substitute");
    assert!(n(&|r| pair(r, &|a| a.contains("|Present|"), &|b| b.starts_with("|-damage|") || b.starts_with("|-crit|"))) >= 40, "Present damage hits");
    assert!(n(&|r| r.name == "pr_ghost" && has(r, "|-immune|")) >= 6, "Present into a Ghost");
    // TRIPLE KICK — 1 / 2 / 3 strikes.
    for (k, floor) in [(1, 15), (2, 10), (3, 30)] {
        assert!(n(&|r| has(r, "|Triple Kick|") && r.lines.iter().any(|l| l.starts_with("|-hitcount|") && l.ends_with(&format!("|{k}")))) >= floor, "Triple Kick {k} strikes");
    }
    assert!(n(&|r| r.lines.iter().any(|l| l.contains("|Triple Kick|") && l.ends_with("[miss]"))) >= 10, "Triple Kick's first strike misses");
    // OHKO
    assert!(n(&|r| has(r, "|[ohko]")) >= 24, "the OHKO level gate");
    assert!(n(&|r| r.lines.windows(2).any(|w| w[0].ends_with(" fnt") && w[1] == "|-ohko")) >= 60, "an OHKO KO");
    assert!(n(&|r| r.lines.iter().any(|l| (l.contains("|Fissure|") || l.contains("|Horn Drill|") || l.contains("|Guillotine|") || l.contains("|Sheer Cold|")) && l.ends_with("[miss]"))) >= 30, "an OHKO miss");
    assert!(n(&|r| has(r, "ability: Sturdy")) >= 6, "Sturdy blocks an OHKO");
    assert!(n(&|r| pair(r, &|a| a.starts_with("|-end|") && a.ends_with("|Substitute"), &|b| b == "|-ohko")) >= 6, "an OHKO breaks a Substitute");
    assert!(n(&|r| pair(r, &|a| a.ends_with("|Substitute|[damage]"), &|b| b == "|-ohko")) >= 6, "an OHKO into a sub that outlives its mon");
    assert!(n(&|r| r.name == "oh_endure" && has(r, "move: Endure")) >= 6, "Endure survives an OHKO");
    assert!(n(&|r| r.name == "oh_focusband" && has(r, "item: Focus Band")) >= 2, "Focus Band survives an OHKO");
    assert!(n(&|r| ["oh_ghost", "oh_flying"].contains(&r.name.as_str()) && r.lines.iter().any(|l| l.starts_with("|-immune|") && l.matches('|').count() == 2)) >= 12, "an OHKO into a natural immunity");
    for (b, s) in [("oh_protect", "|Protect"), ("oh_wonderguard", "Wonder Guard"), ("oh_dig", "|-miss|"), ("oh_static", "ability: Static")] {
        assert!(n(&|r| r.name == b && has(r, s)) >= 1, "{b}");
    }
}
