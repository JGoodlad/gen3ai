//! The B11 MOVES **golden differential** (`gen3_b11_moves_v1`, gen3 full parity B11): Metronome, Mirror
//! Move, Assist, Nature Power and Sketch, and the one calling path they share with Sleep Talk. Per row,
//! against the sim's recording (`harness/gen_b11_moves_golden.js`, real seeds, the lines as SENT; the
//! settled spec is `harness/probe_b11_moves.js`): the post-script PRNG SEED, the FRAMING window and the
//! post-`|turn|1` lines, byte-equal, in gen3customgame (and gen3ou). The boards: Metronome mirrors and
//! walls over natural samples (two-turn, recharge, multi-hit, fixed damage, status, Bide, the lock-ins,
//! Metronome calling Assist / Mirror Move / Nature Power), Pressure, a Choice Band lock, gen3ou, Encore
//! overriding a called lock, Disable / Spite / Encore / Mimic after a called lock's repeat; Mirror Move
//! (same-turn and previous-turn hits, status moves, the noMirror list, a switched / fainted source, a
//! Metronome or Sleep Talk callee, Counter, Protect, Fly, Future Sight, Bide, multi-hit, a mirror);
//! Assist (two party layouts covering every family, a typed Hidden Power, an empty pool, alone, a fainted
//! mate); Nature Power (Swift, a Ghost, Protect, a +6-evasion miss, Sleep Talk); Sketch (success, every
//! fail form, through Protect, a Metronome user, Struggle, a typed Hidden Power, transformed, across a
//! switch, Sleep Talk); and Conversion 2 reading a Sleep-Talk callee.
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
    for line in include_str!("vectors/b11_moves_golden.txt").lines() {
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
fn the_b11_moves_match_showdown() {
    let rows = load();
    assert!(rows.len() >= 700, "golden shrank to {} rows", rows.len());
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
    let called = |r: &Row, caller: &str, names: &[&str]| {
        r.lines.iter().any(|l| {
            l.starts_with("|move|") && l.contains(&format!("[from] {caller}")) && names.iter().any(|c| l.contains(&format!("|{c}|")))
        })
    };
    const TWO_TURN: [&str; 8] = ["Solar Beam", "Razor Wind", "Sky Attack", "Skull Bash", "Fly", "Bounce", "Dig", "Dive"];
    const RECHARGE: [&str; 4] = ["Hyper Beam", "Blast Burn", "Frenzy Plant", "Hydro Cannon"];
    const MULTI: [&str; 17] = [
        "Double Kick", "Twineedle", "Bonemerang", "Pin Missile", "Bullet Seed", "Icicle Spear", "Rock Blast", "Barrage",
        "Comet Punch", "Double Slap", "Spike Cannon", "Arm Thrust", "Fury Attack", "Fury Swipes", "Bone Rush", "Triple Kick", "Beat Up",
    ];
    const FIXED: [&str; 11] = [
        "Seismic Toss", "Night Shade", "Sonic Boom", "Dragon Rage", "Super Fang", "Psywave", "Endeavor", "Fissure", "Horn Drill",
        "Guillotine", "Sheer Cold",
    ];
    const STATUS: [&str; 19] = [
        "Thunder Wave", "Toxic", "Poison Powder", "Poison Gas", "Will-O-Wisp", "Spore", "Sleep Powder", "Hypnosis", "Sing",
        "Lovely Kiss", "Grass Whistle", "Stun Spore", "Glare", "Confuse Ray", "Supersonic", "Sweet Kiss", "Teeter Dance",
        "Leech Seed", "Yawn",
    ];
    const LOCKIN: [&str; 6] = ["Outrage", "Thrash", "Petal Dance", "Rollout", "Ice Ball", "Uproar"];
    // METRONOME — a broad spread of callee families over natural samples.
    let calls: usize = rows.iter().map(|r| r.lines.iter().filter(|l| l.contains("[from] Metronome")).count()).sum();
    assert!(calls >= 2000, "Metronome calls {calls}");
    assert!(n(&|r| called(r, "Metronome", &TWO_TURN)) >= 40, "Metronome -> a two-turn move");
    assert!(n(&|r| called(r, "Metronome", &RECHARGE) && has(r, "|-mustrecharge|")) >= 12, "Metronome -> a recharge move");
    assert!(n(&|r| called(r, "Metronome", &MULTI) && has(r, "|-hitcount|")) >= 50, "Metronome -> a multi-hit move");
    assert!(n(&|r| called(r, "Metronome", &FIXED)) >= 40, "Metronome -> a fixed-damage move");
    assert!(n(&|r| called(r, "Metronome", &STATUS)) >= 50, "Metronome -> a status move");
    assert!(n(&|r| called(r, "Metronome", &["Bide"])) >= 4, "Metronome -> Bide");
    assert!(n(&|r| called(r, "Metronome", &LOCKIN) && has(r, "|[from] lockedmove")) >= 25, "Metronome -> a lock-in, repeated");
    assert!(n(&|r| called(r, "Metronome", &["Teleport", "Camouflage", "Role Play"])) >= 10, "Metronome -> a B10 move");
    for c in ["Assist", "Mirror Move", "Nature Power"] {
        assert!(n(&|r| called(r, "Metronome", &[c])) >= 3, "Metronome -> {c}");
    }
    assert!(n(&|r| r.format == "gen3ou" && has(r, "[from] Metronome")) >= 20, "Metronome in gen3ou");
    assert!(n(&|r| r.name == "mt_choiceband" && has(r, "[from] Metronome")) >= 20, "a Choice Band Metronome");
    assert!(
        n(&|r| r.lines.iter().any(|l| l == "|move|p1a: Clefable|Metronome|p1a: Clefable|[from] lockedmove")) >= 8,
        "Encore overrides a called lock"
    );
    for x in ["Disable", "Spite", "Encore"] {
        let board = format!("mt_{}_repeat", x.to_lowercase());
        let line = format!("|move|p2a: Smeargle|{x}||[still]");
        assert!(n(&|r| r.name == board && r.lines.iter().any(|l| *l == line)) >= 3, "{x} fails on a called lock's repeat");
    }
    assert!(
        n(&|r| r.name == "mt_mimic_repeat"
            && r.lines.iter().any(|l| l.starts_with("|-activate|p2a: Smeargle|move: Mimic|") && !l.ends_with("|Metronome")))
            >= 3,
        "Mimic copies a called lock's callee"
    );
    // MIRROR MOVE
    assert!(n(&|r| has(r, "[from] Mirror Move")) >= 80, "Mirror Move calls");
    assert!(n(&|r| has(r, "|Mirror Move||[still]")) >= 40, "Mirror Move fails");
    for (fam, names, floor) in [
        ("Counter", &["Counter"][..], 6),
        ("status", &STATUS[..], 12),
        ("two-turn", &TWO_TURN[..], 6),
        ("Bide", &["Bide"][..], 6),
        ("multi-hit", &MULTI[..], 6),
    ] {
        assert!(n(&|r| called(r, "Mirror Move", names)) >= floor, "Mirror Move -> {fam}");
    }
    assert!(n(&|r| r.name == "mm_protect" && has(r, "[from] Mirror Move") && has(r, "|-activate|p2a: Snorlax|Protect")) >= 6, "a mirrored move into Protect");
    assert!(n(&|r| r.name == "mm_vs_metronome" && has(r, "|Mirror Move||[still]")) >= 12, "a Metronome callee the source lacks");
    assert!(n(&|r| r.name == "mm_vs_sleeptalk" && has(r, "[from] Mirror Move")) >= 6, "a Sleep Talk callee the source holds");
    for b in ["mm_switched", "mm_future", "mm_faint", "mm_mirror", "mm_nomirror"] {
        assert!(n(&|r| r.name == b) >= 8, "{b} rows");
    }
    // ASSIST — the same spread from the party.
    assert!(n(&|r| has(r, "[from] Assist")) >= 70, "Assist calls");
    for (fam, names, floor) in [
        ("two-turn", &TWO_TURN[..], 20),
        ("multi-hit", &MULTI[..], 10),
        ("fixed damage", &FIXED[..], 15),
        ("status", &STATUS[..], 12),
        ("Bide", &["Bide"][..], 8),
        ("Hidden Power", &["Hidden Power"][..], 12),
    ] {
        assert!(n(&|r| called(r, "Assist", names)) >= floor, "Assist -> {fam}");
    }
    assert!(n(&|r| called(r, "Assist", &RECHARGE) && has(r, "|-mustrecharge|")) >= 4, "Assist -> a recharge move");
    assert!(n(&|r| has(r, "|Assist||[still]")) >= 6, "Assist's empty pool");
    assert!(n(&|r| r.name == "as_fainted" && called(r, "Assist", &["Explosion"])) >= 6, "Assist draws a fainted mate's move");
    // NATURE POWER
    assert!(n(&|r| has(r, "|Swift|") && has(r, "[from] Nature Power")) >= 30, "Nature Power -> Swift");
    assert!(n(&|r| has(r, "|Nature Power|") && has(r, "|[miss]")) >= 8, "Nature Power misses its user");
    assert!(n(&|r| r.name == "np_ghost" && has(r, "|-immune|")) >= 4, "Swift into a Ghost");
    assert!(n(&|r| r.lines.iter().any(|l| l.contains("|Nature Power|") && l.contains("[from] Sleep Talk"))) >= 6, "Sleep Talk -> Nature Power");
    // SKETCH
    assert!(n(&|r| has(r, "|Sketch|") && r.lines.iter().any(|l| l.starts_with("|-activate|") && l.contains("|move: Mimic|"))) >= 24, "Sketch copies");
    assert!(n(&|r| has(r, "|Sketch||[still]")) >= 16, "Sketch fails");
    for (b, s) in [
        ("sk_protect", "move: Mimic|Protect"),
        ("sk_struggle", "|Struggle|"),
        ("sk_transformed", "|Sketch||[still]"),
        ("sk_hiddenpower", "move: Mimic|Hidden Power"),
        ("sk_metronome", "move: Mimic|Metronome"),
    ] {
        assert!(n(&|r| r.name == b && has(r, s)) >= 1, "{b}");
    }
    // THE SHARED PATH: every caller's callee is the user's lastMoveUsed (a later Conversion 2 reads it).
    for (c, b, floor) in [("Metronome", "cv_metronome", 8), ("Assist", "cv_assist", 6), ("Mirror Move", "cv_mirrormove", 6), ("Nature Power", "cv_naturepower", 3)] {
        assert!(
            n(&|r| r.name == b
                && r.lines.iter().enumerate().any(|(i, l)| l.contains(&format!("[from] {c}"))
                    && r.lines[i + 1..].iter().any(|m| m.starts_with("|-start|p2a: Porygon2|typechange|"))))
                >= floor,
            "Conversion 2 after {c}"
        );
    }
    // THE SHARED PATH: AfterMove keys on the callee (a Sleep-Talk Charge is kept).
    assert!(
        n(&|r| r.name == "st_charge" && r.lines.iter().any(|l| l == "|move|p1a: Jolteon|Charge|p1a: Jolteon|[from] Sleep Talk")) >= 6,
        "Sleep Talk -> Charge"
    );
    // THE SHARED PATH: Conversion 2 reads a Sleep-Talk callee.
    assert!(
        n(&|r| r.name == "st_conversion2"
            && has(r, "[from] Sleep Talk")
            && r.lines.iter().any(|l| l.ends_with("typechange|Bug") || l.ends_with("typechange|Grass") || l.ends_with("typechange|Flying")))
            >= 6,
        "Conversion 2 after a Sleep-Talk Earthquake"
    );
}
