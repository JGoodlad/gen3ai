//! The B7 MOVES **golden differential** (`gen3_b7_moves_v1`, gen3 full parity B7): Stockpile, Spit
//! Up, Swallow, Mud Sport, Water Sport, Mist, Lock-On, Mind Reader. Per row, against the sim's
//! recording (`harness/gen_b7_moves_golden.js`; the settled spec is `harness/probe_b7_moves.js`): the
//! post-script PRNG SEED, the FRAMING window and the post-`|turn|1` lines, byte-equal, in the row's
//! format (`gen3customgame`, and `gen3ou` for one board per family — Mist, Mud Sport and the
//! stockpile family run there under the clause shuffle). The boards: the stockpile ladder (the cap,
//! Spit Up at 1-3 layers, Swallow's 25 / 50 / 100 % and its full-HP fail, the 0-layer fails, a miss,
//! Protect, a Ghost, a Substitute, Wonder Guard, Counter, Taunt, switch-out, Baton Pass, faint,
//! Snatch, Sleep Talk), the sports (the halving, both on one mon, a speed-tie mirror, two holders,
//! Baton Pass, switch-out, Pressure, Hidden Power, the BasePower tie draw on a confusion self-hit /
//! Beat Up / a High Jump Kick crash / a Future Sight cast, Explosion, Weather Ball, Charge, Blaze +
//! Thick Fat, Sleep Talk), Mist (status and secondary drops, Intimidate, self drops, Haze, Memento,
//! Clear Body / Hyper Cutter / Keen Eye, the −6 floor, both sides and every screen, Snatch, a
//! Substitute, switch-out, Sleep Talk) and Lock-On / Mind Reader (the skipped accuracy draw, the
//! recast fail, the lapse, Protect / Substitute, Dig / Fly, a switched target, the holder switching,
//! Baton Pass, a mirror tie, +6 evasion, a status move, Future Sight, Perish Song, Pressure, Sleep
//! Talk, a multi-hit).
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
    for line in include_str!("vectors/b7_moves_golden.txt").lines() {
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
fn the_b7_moves_match_showdown() {
    let rows = load();
    assert!(rows.len() >= 650, "golden shrank to {} rows", rows.len());
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
/// interaction; the generator refuses a zero, this test refuses a thinned vector).
#[test]
fn the_golden_covers_each_move_and_interaction() {
    let rows = load();
    let n = |pred: &dyn Fn(&Row) -> bool| rows.iter().filter(|r| pred(r)).count();
    let has = |r: &Row, s: &str| r.lines.iter().any(|l| l.contains(s));
    let pair = |r: &Row, a: &dyn Fn(&str) -> bool, b: &dyn Fn(&str) -> bool| {
        r.lines.windows(2).any(|w| a(&w[0]) && b(&w[1]))
    };
    let is_move = |l: &str, name: &str| l.starts_with("|move|") && l.split('|').nth(3) == Some(name);
    // STOCKPILE / SPIT UP / SWALLOW
    assert!(n(&|r| has(r, "|stockpile1")) >= 150, "Stockpile starts");
    assert!(n(&|r| has(r, "|stockpile3")) >= 40, "three layers");
    assert!(n(&|r| pair(r, &|a| is_move(a, "Stockpile"), &|b| !b.starts_with("|-start|") && !b.starts_with("|-activate|"))) >= 16, "Stockpile at the cap (bare)");
    assert!(n(&|r| pair(r, &|a| is_move(a, "Spit Up"), &|b| b.starts_with("|-damage|") || b.starts_with("|-crit|") || b.starts_with("|-supereffective|") || b.starts_with("|-resisted|"))) >= 35, "Spit Up hits");
    assert!(n(&|r| r.lines.iter().any(|l| l.starts_with("|-end|") && l.ends_with("|Stockpile"))) >= 100, "the stockpile ends");
    assert!(n(&|r| r.lines.iter().any(|l| is_move(l, "Spit Up") && l.ends_with("[miss]"))) >= 3, "Spit Up misses");
    for (b, s) in [("st_ghost", "|-immune|p2a: Gengar"), ("st_protect", "|-activate|p2a: Gengar|Protect"), ("st_wonderguard", "ability: Wonder Guard")] {
        assert!(n(&|r| r.name == b && has(r, s)) >= 8, "{b}");
    }
    assert!(n(&|r| r.name == "st_counter" && r.lines.iter().any(|l| is_move(l, "Counter"))) >= 6, "Counter answers Spit Up");
    assert!(n(&|r| pair(r, &|a| is_move(a, "Swallow"), &|b| b.starts_with("|-heal|"))) >= 18, "Swallow heals");
    assert!(n(&|r| r.lines.iter().any(|l| l.starts_with("|-fail|") && l.ends_with("|heal"))) >= 12, "Swallow at full HP");
    assert!(n(&|r| has(r, "|Stockpile|p2a: Sableye|[from] Snatch")) >= 8, "Snatch steals Stockpile");
    assert!(n(&|r| has(r, "|Swallow|p2a: Sableye|[from] Snatch")) >= 8, "Snatch steals Swallow");
    assert!(n(&|r| r.lines.iter().any(|l| is_move(l, "Spit Up") && l.contains("[from] Sleep Talk"))) >= 4, "Sleep Talk calls Spit Up");
    assert!(n(&|r| r.lines.iter().any(|l| is_move(l, "Swallow") && l.contains("[from] Sleep Talk"))) >= 4, "Sleep Talk calls Swallow");
    for b in ["st_switch", "st_batonpass", "st_faint", "st_taunt", "st_zero", "st_evasion", "st_sub", "st_ou"] {
        assert!(n(&|r| r.name == b) >= 8, "{b} rows");
    }
    // MUD SPORT / WATER SPORT
    assert!(n(&|r| has(r, "|-start|") && has(r, "|Mud Sport")) >= 120, "Mud Sport starts");
    assert!(n(&|r| has(r, "|move: Water Sport")) >= 70, "Water Sport starts");
    assert!(n(&|r| r.lines.windows(2).any(|w| (w[0].ends_with("|Mud Sport||[still]") || w[0].ends_with("|Water Sport||[still]")) && w[1].starts_with("|-fail|"))) >= 24, "a sport recast fails");
    assert!(n(&|r| r.name == "sp_batonpass" && has(r, "[from] Baton Pass")) >= 8, "Baton Pass passes Mud Sport");
    assert!(n(&|r| r.name == "sp_confusion" && has(r, "[from] confusion")) >= 10, "a confusion self-hit under both sports");
    assert!(n(&|r| r.lines.iter().any(|l| (l.contains("|Mud Sport|") || l.contains("|Water Sport|")) && l.contains("[from] Sleep Talk"))) >= 16, "Sleep Talk calls a sport");
    for b in ["sp_both_same", "sp_mirror", "sp_quarter", "sp_beatup", "sp_hjk", "sp_explosion", "sp_weatherball", "sp_charge", "sp_blaze_thickfat", "sp_futuresight", "sp_hiddenpower", "sp_pressure", "sp_switch", "sp_ou"] {
        assert!(n(&|r| r.name == b) >= 8, "{b} rows");
    }
    // MIST
    assert!(n(&|r| r.lines.iter().any(|l| l.starts_with("|-sidestart|") && l.ends_with("|Mist"))) >= 150, "Mist starts");
    assert!(n(&|r| has(r, "|move: Mist")) >= 70, "Mist blocks with its line");
    assert!(n(&|r| r.lines.iter().any(|l| l.starts_with("|-sideend|") && l.ends_with("|Mist"))) >= 60, "Mist ends");
    assert!(n(&|r| pair(r, &|a| a.ends_with("|Mist||[still]"), &|b| b.starts_with("|-fail|"))) >= 12, "a Mist recast fails");
    assert!(n(&|r| pair(r, &|a| a.ends_with("|Intimidate|boost"), &|b| b.ends_with("|move: Mist"))) >= 8, "Mist blocks Intimidate");
    assert!(n(&|r| pair(r, &|a| a.ends_with("|move: Mist"), &|b| b.starts_with("|-unboost|") && b.ends_with("|0"))) >= 8, "Mist at the -6 floor");
    assert!(n(&|r| r.name == "mi_clearbody" && has(r, "ability: Clear Body")) >= 8, "Clear Body before Mist");
    assert!(n(&|r| pair(r, &|a| a.starts_with("|-sideend|") && a.ends_with("|Mist"), &|b| b.starts_with("|-sideend|") && b.ends_with("|Mist"))) >= 40, "both Mists end on one residual");
    assert!(n(&|r| has(r, "|Mist|p2a: Sableye|[from] Snatch")) >= 8, "Snatch steals Mist");
    for b in ["mi_icywind", "mi_selfdrop", "mi_haze", "mi_tickle", "mi_hypercutter", "mi_keeneye", "mi_screens", "mi_screens_b", "mi_sub", "mi_switch", "mi_sleeptalk", "mi_crunch", "mi_ou"] {
        assert!(n(&|r| r.name == b) >= 8, "{b} rows");
    }
    // LOCK-ON / MIND READER
    assert!(n(&|r| has(r, "|move: Lock-On|[of]")) >= 120, "Lock-On activations");
    assert!(n(&|r| has(r, "|move: Mind Reader|[of]")) >= 25, "Mind Reader activations");
    assert!(n(&|r| r.lines.windows(2).any(|w| (w[0].ends_with("|Lock-On||[still]") || w[0].ends_with("|Mind Reader||[still]")) && w[1].starts_with("|-fail|"))) >= 40, "Lock-On / Mind Reader fails");
    assert!(n(&|r| r.lines.iter().any(|l| (is_move(l, "Lock-On") || is_move(l, "Mind Reader")) && l.ends_with("[miss]"))) >= 12, "Lock-On misses");
    assert!(n(&|r| r.name == "lo_dig" && has(r, "|-damage|p2a: Golem")) >= 4, "the lock reaches a Dig-ing target");
    assert!(n(&|r| r.name == "lo_fly" && has(r, "|-damage|p2a: Skarmory")) >= 8, "the lock reaches a flying target");
    assert!(n(&|r| r.name == "lo_perish" && has(r, "|-start|p2a: Swellow|perish3|[silent]")) >= 8, "Perish Song reaches a locked flyer");
    assert!(n(&|r| r.name == "lo_batonpass" && has(r, "[from] Baton Pass")) >= 8, "Baton Pass passes the lock");
    assert!(n(&|r| r.lines.iter().any(|l| (is_move(l, "Lock-On") || is_move(l, "Mind Reader")) && l.contains("[from] Sleep Talk"))) >= 10, "Sleep Talk calls Lock-On / Mind Reader");
    for b in ["lo_zap", "lo_lapse", "lo_protect_sub", "lo_locked_protect", "lo_switch", "lo_switch_back", "lo_holder_switch", "lo_mirror", "lo_evasion", "lo_hypnosis", "lo_futuresight", "lo_pressure", "lo_rockblast", "lo_ou"] {
        assert!(n(&|r| r.name == b) >= 8, "{b} rows");
    }
    assert!(n(&|r| r.format == "gen3ou") >= 32, "gen3ou rows");
}
