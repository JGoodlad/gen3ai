//! The B10 MOVES **golden differential** (`gen3_b10_moves_v1`, `gen3_b10_endless_battle_v1`, gen3 full
//! parity B10): Teleport, Follow Me, Helping Hand, Role Play, Camouflage and the Endless Battle Clause.
//! Per row, against the sim's recording (`harness/gen_b10_moves_golden.js`; the settled spec is
//! `harness/probe_b10_moves.js`): the post-script PRNG SEED, the FRAMING window and the post-`|turn|1`
//! lines, byte-equal, in the row's format (`gen3customgame`, and `gen3ou` for a board per move and for
//! the clause). The boards: Teleport and Follow Me (the bare failing announce, speed ties, Encore,
//! Sleep Talk), Helping Hand (`null|[notarget]` + `-notarget`, Pressure, Encore / Disable / Spite /
//! Mimic of it, Sleep Talk), Role Play (the copy and its fails, Protect / Substitute / a semi-invulnerable
//! target, Truant, the status-immune cures, Flash Fire's end, Cloud Nine / Air Lock, Forecast, Speed
//! Boost, Shadow Tag, Natural Cure, Synchronize, Color Change, Shed Skin, the switch revert), Camouflage
//! (Color Change, Normal fails, Snatch both ways, the switch revert, Transform, Conversion, a Ghost
//! retyped Normal) and the ENDLESS BATTLE CLAUSE — a golden of long random battles is impractical, so its
//! rows are TARGETED boards played past turn 100 (a Recycle loss, a mutual tie, a Thief bare tie, a
//! Trick + Pickup loss, a Shadow-Tag-trapped side, and four controls that never trigger).
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
    for line in include_str!("vectors/b10_moves_golden.txt").lines() {
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
fn the_b10_moves_match_showdown() {
    let rows = load();
    assert!(rows.len() >= 450, "golden shrank to {} rows", rows.len());
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
        // The format's own `|tier|` / `|rule|` frames (`bridge::reframe`); a no-op for gen3customgame.
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
    // The bare failing self-target announce: `|move|<u>|<Move>|<u>`, followed by no fail line.
    let bare = |r: &Row, mv: &str| {
        r.lines.iter().enumerate().any(|(i, l)| {
            let f: Vec<&str> = l.split('|').collect();
            f.len() == 5 && f[1] == "move" && f[3] == mv && f[2] == f[4]
                && !r.lines.get(i + 1).is_some_and(|x| x.starts_with("|-fail|") || x.starts_with("|-singleturn|"))
        })
    };
    // TELEPORT
    assert!(n(&|r| bare(r, "Teleport")) >= 30, "bare Teleport");
    assert!(n(&|r| r.lines.iter().any(|l| l.contains("|Teleport|") && l.contains("[from] Sleep Talk"))) >= 4, "Sleep Talk calls Teleport");
    assert!(n(&|r| r.name == "tp_encore" && has(r, "|-start|p1a: Abra|Encore")) >= 8, "Encore repeats Teleport");
    assert!(n(&|r| r.format == "gen3ou" && bare(r, "Teleport")) >= 8, "Teleport in gen3ou");
    // FOLLOW ME
    assert!(n(&|r| bare(r, "Follow Me")) >= 30, "bare Follow Me");
    assert!(n(&|r| r.lines.iter().any(|l| l.contains("|Follow Me|") && l.contains("[from] Sleep Talk"))) >= 2, "Sleep Talk calls Follow Me");
    assert!(n(&|r| r.name == "fm_encore" && has(r, "|-start|p1a: Clefable|Encore")) >= 8, "Encore repeats Follow Me");
    assert!(n(&|r| has(r, "|-singleturn|") && has(r, "Follow Me") && r.lines.iter().any(|l| l.starts_with("|-singleturn|") && l.contains("Follow Me"))) == 0, "Follow Me never starts in singles");
    // HELPING HAND
    assert!(n(&|r| pair(r, &|a| a.ends_with("|Helping Hand|null|[notarget]"), &|b| b.starts_with("|-notarget|"))) >= 60, "Helping Hand finds no target");
    assert!(n(&|r| has(r, "|Helping Hand|null|[from] Sleep Talk|[notarget]")) >= 4, "Sleep Talk calls Helping Hand");
    assert!(n(&|r| has(r, "|move: Spite|helpinghand|")) >= 8, "Spite on a Helping Hand lastMove");
    assert!(n(&|r| has(r, "|move: Mimic|Helping Hand")) >= 8, "Mimic copies Helping Hand");
    assert!(n(&|r| r.name == "hh_encore" && has(r, "|-start|p1a: Plusle|Encore")) >= 8, "Encore repeats Helping Hand");
    assert!(n(&|r| has(r, "|Disable|Helping Hand")) >= 3, "Disable on Helping Hand");
    assert!(n(&|r| r.format == "gen3ou" && has(r, "|-notarget|")) >= 8, "Helping Hand in gen3ou");
    // ROLE PLAY
    assert!(n(&|r| has(r, "|[from] move: Role Play|[of] ")) >= 150, "Role Play copies");
    assert!(n(&|r| pair(r, &|a| a.ends_with("|Role Play||[still]"), &|b| b.starts_with("|-fail|"))) >= 50, "Role Play fails");
    for (b, s) in [
        ("rp_wonderguard", "|Role Play||[still]"),
        ("rp_flashfire_same", "|Role Play||[still]"),
        ("rp_protect", "[from] move: Role Play"),
        ("rp_sub", "[from] move: Role Play"),
        ("rp_bounce", "|Role Play|p2a: Jolteon|[miss]"),
        ("rp_wg_user", "|Intimidate|Wonder Guard|[from] move: Role Play"),
        ("rp_airlock", "|Air Lock|Intimidate|[from] move: Role Play"),
        ("rp_cloudnine", "|Intimidate|Cloud Nine|[from] move: Role Play"),
        ("rp_castform", "|Intimidate|Forecast|[from] move: Role Play"),
        ("rp_shadowtag", "|Shadow Tag|Synchronize|[from] move: Role Play"),
        ("rp_colorchange", "|-start|p1a: Alakazam|typechange|Electric|[from] ability: Color Change"),
        ("rp_speedboost", "|-ability|p1a: Alakazam|Speed Boost|boost"),
        ("rp_truant", "|cant|p1a: Alakazam|ability: Truant"),
        ("rp_truant_user", "|Intimidate|Truant|[from] move: Role Play"),
        ("rp_synchronize", "|[from] ability: Synchronize|[of] p1a: Snorlax"),
        ("rp_flashfire", "|-end|p1a: Houndoom|ability: Flash Fire|[silent]"),
    ] {
        assert!(n(&|r| r.name == b && has(r, s)) >= 6, "{b}: {s}");
    }
    assert!(n(&|r| r.lines.windows(2).any(|w| (w[0].ends_with("|ability: Insomnia") || w[0].ends_with("|ability: Water Veil")) && w[1].starts_with("|-curestatus|p1a"))) >= 10, "a copied status-immune ability cures at the Update");
    assert!(n(&|r| has(r, "|-curestatus|p1a: Alakazam|par|[from] ability: Natural Cure")) >= 4, "a copied Natural Cure cures at switch-out");
    assert!(n(&|r| r.name == "rp_shedskin" && has(r, "|-activate|p1a: Alakazam|ability: Shed Skin")) >= 2, "a copied Shed Skin rolls");
    assert!(n(&|r| r.name == "rp_switch" && has(r, "|switch|p1a: Alakazam")) >= 8, "the switch revert");
    assert!(n(&|r| r.lines.iter().any(|l| l.contains("|Role Play|") && l.contains("[from] Sleep Talk"))) >= 4, "Sleep Talk calls Role Play");
    assert!(n(&|r| r.format == "gen3ou" && has(r, "[from] move: Role Play")) >= 6, "Role Play in gen3ou");
    // CAMOUFLAGE
    assert!(n(&|r| r.lines.iter().any(|l| l.starts_with("|-start|") && l.ends_with("|typechange|Normal"))) >= 50, "Camouflage retypes");
    assert!(n(&|r| pair(r, &|a| a.contains("|Camouflage||") && a.ends_with("[still]"), &|b| b.starts_with("|-fail|"))) >= 40, "Camouflage fails");
    assert!(n(&|r| has(r, "|Camouflage|p2a: Sableye|[from] Snatch")) >= 8, "Snatch steals Camouflage");
    assert!(n(&|r| has(r, "|Camouflage||[from] Snatch|[still]")) >= 8, "a stolen Camouflage fails on a Normal snatcher");
    assert!(n(&|r| r.name == "cf_transform" && has(r, "|-start|p1a: Ditto|typechange|Normal")) >= 8, "a transformed user retypes");
    assert!(n(&|r| r.name == "cf_switch" && has(r, "|-immune|p1a: Gengar")) >= 4, "the switch revert restores Ghost");
    assert!(n(&|r| r.name == "cf_ghost_immunity" && has(r, "|Cross Chop|p1a: Gengar") && has(r, "|-supereffective|p1a: Gengar")) >= 4, "a Normal-retyped Ghost takes Fighting");
    assert!(n(&|r| r.lines.iter().any(|l| l.contains("|Camouflage|") && l.contains("[from] Sleep Talk"))) >= 4, "Sleep Talk calls Camouflage");
    assert!(n(&|r| r.format == "gen3ou" && r.lines.iter().any(|l| l.ends_with("|typechange|Normal"))) >= 8, "Camouflage in gen3ou");
    // ENDLESS BATTLE CLAUSE
    let last = |r: &Row| r.lines.last().cloned().unwrap_or_default();
    let lost = |r: &Row| has(r, "|-message|P1's team started with the rudimentary means to perform restorative berry-cycling and thus loses.") && last(r) == "|win|P2";
    for b in ["ebc_recycle_loses", "ebc_trick_pickup_loses", "ebc_shadowtag_trapped"] {
        assert!(n(&|r| r.name == b && lost(r) && has(r, "|turn|100") && !has(r, "|turn|101")) >= 2, "{b}: P1 loses at the turn-101 endTurn");
    }
    assert!(n(&|r| r.name == "ebc_both_cycle_tie" && has(r, "|-message|Each side's team started with the rudimentary means to perform restorative berry-cycling.") && last(r) == "|tie") >= 2, "the mutual tie");
    assert!(n(&|r| r.name == "ebc_thief_bare_tie" && !has(r, "|-message|") && last(r) == "|tie" && !has(r, "|turn|101")) >= 2, "the bare tie");
    for b in ["ebc_ctrl_both_can_switch", "ebc_ctrl_switch_clears", "ebc_ctrl_starting_berry", "ebc_ctrl_customgame"] {
        assert!(n(&|r| r.name == b && has(r, "|turn|104") && !has(r, "berry-cycling") && !has(r, "|tie") && !has(r, "|win|")) >= 2, "{b}: no trigger");
    }
}
