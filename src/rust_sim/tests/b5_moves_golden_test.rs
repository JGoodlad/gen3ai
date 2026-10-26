//! The B5 MOVES **golden differential** (`gen3_b5_moves_v1`, gen3 full parity B5): Spite, Grudge,
//! Snore, Nightmare, Ingrain, and Rest under a live Uproar. Per row, against the sim's recording
//! (`harness/gen_b5_moves_golden.js`; the settled spec is `harness/probe_b5_moves.js`): the
//! post-script PRNG SEED, the FRAMING window and the post-`|turn|1` lines, byte-equal, in the row's
//! format (`gen3customgame`, and `gen3ou` for the Rest-under-Uproar clause-shuffle board). The
//! boards: Spite (no lastMove, Pressure, Substitute, Protect, drained to 0 PP and Struggle, Sleep
//! Talk, Choice Band, Encore, Mimic), Grudge (a KO, removal by the next move, by a sleep cant, a
//! recoil-KO'd killer, Sleep Talk, Explosion, Pursuit, Future Sight, Struggle), Snore (awake, asleep,
//! flinch, via Sleep Talk, Soundproof, Ghost, Substitute, Protect), Nightmare (the residual ladder
//! against Leftovers / Leech Seed / Curse, every sleep cure that ends it, Focus Band, a mirror tie),
//! Ingrain (heal, Roar / Whirlwind, Suction Cups, Baton Pass, Snatch, Sleep Talk, Wrap, Mean Look, a
//! mirror tie) and Rest under Uproar.
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
    for line in include_str!("vectors/b5_moves_golden.txt").lines() {
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
fn the_b5_moves_match_showdown() {
    let rows = load();
    assert!(rows.len() >= 500, "golden shrank to {} rows", rows.len());
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
    // SPITE
    assert!(n(&|r| has(r, "|move: Spite|")) >= 60, "Spite activations");
    assert!(n(&|r| pair(r, &|a| a.ends_with("|Spite||[still]"), &|b| b.starts_with("|-fail|"))) >= 20, "Spite fails");
    assert!(n(&|r| r.name == "sp_sub" && has(r, "|move: Spite|")) >= 8, "Spite through a Substitute");
    assert!(n(&|r| r.name == "sp_protect" && has(r, "|-activate|p2a: Jolteon|Protect")) >= 8, "Spite into Protect");
    assert!(n(&|r| has(r, "|move: Spite|sleeptalk|")) >= 8, "Spite on a Sleep Talk lastMove");
    assert!(n(&|r| r.name == "sp_drain_struggle" && has(r, "|Struggle|") && has(r, "|Spite||[still]")) >= 4, "Spite on Struggle");
    // A slot Spite emptied after the choice: `runMove`'s `|cant|<mon>|nopp|<Move>`.
    assert!(n(&|r| r.lines.iter().any(|l| l.starts_with("|cant|") && l.contains("|nopp|"))) >= 3, "the nopp cant");
    for b in ["sp_choice", "sp_encore", "sp_pressure", "sp_mimic"] {
        assert!(n(&|r| r.name == b) >= 8, "{b} rows");
    }
    // GRUDGE
    assert!(n(&|r| has(r, "|-singlemove|") && has(r, "|Grudge")) >= 50, "Grudge casts");
    assert!(n(&|r| has(r, "|move: Grudge|")) >= 25, "Grudge activations");
    assert!(n(&|r| r.name == "gr_recoil" && has(r, "|move: Grudge|Double-Edge")) >= 8, "Grudge on a recoil-KO'd killer");
    assert!(n(&|r| has(r, "|move: Grudge|Sleep Talk")) >= 4, "Grudge zeroes Sleep Talk");
    assert!(n(&|r| has(r, "|move: Grudge|Pursuit")) >= 8, "Grudge on Pursuit");
    for b in ["gr_removed", "gr_slept", "gr_explosion", "gr_futuresight"] {
        assert!(n(&|r| r.name == b && has(r, "|faint|p1a") && !has(r, "move: Grudge|")) >= 6, "{b}: a KO with no Grudge activation");
    }
    // SNORE
    assert!(n(&|r| r.lines.windows(3).any(|w| w[0].starts_with("|cant|") && w[1].contains("|Snore|") && w[2].starts_with("|-damage|"))) >= 15, "asleep Snore hits");
    assert!(n(&|r| r.name == "sn_awake") >= 8, "awake Snore rows");
    assert!(n(&|r| r.name == "sn_flinch" && has(r, "|cant|p2a: Snorlax|flinch")) >= 4, "Snore flinches");
    assert!(n(&|r| r.lines.iter().any(|l| l.contains("|Snore|") && l.contains("[from] Sleep Talk"))) >= 4, "Sleep Talk calls Snore");
    assert!(n(&|r| r.name == "sn_soundproof" && has(r, "ability: Soundproof")) >= 8, "Snore into Soundproof");
    // NIGHTMARE
    assert!(n(&|r| has(r, "|-start|") && has(r, "|Nightmare")) >= 90, "Nightmare casts");
    assert!(n(&|r| has(r, "[from] Nightmare")) >= 80, "Nightmare chips");
    assert!(n(&|r| pair(r, &|a| a.ends_with("|Nightmare||[still]"), &|b| b.starts_with("|-fail|"))) >= 30, "Nightmare fails");
    assert!(n(&|r| pair(r, &|a| a.starts_with("|-curestatus|"), &|b| b.ends_with("|Nightmare|[silent]"))) >= 80, "a sleep cure ends Nightmare");
    for b in ["nm_naturalcure", "nm_shedskin", "nm_healbell", "nm_aromatherapy", "nm_chesto_trick", "nm_skillswap"] {
        assert!(n(&|r| r.name == b && has(r, "|Nightmare|[silent]")) >= 3, "{b}: the cure ends Nightmare");
    }
    assert!(n(&|r| pair(r, &|a| a.ends_with("[from] Nightmare"), &|b| b.ends_with("[from] Nightmare"))) >= 2, "a Nightmare residual tie");
    // INGRAIN
    assert!(n(&|r| has(r, "|-start|") && has(r, "|move: Ingrain")) >= 90, "Ingrain casts");
    assert!(n(&|r| has(r, "[from] Ingrain")) >= 50, "Ingrain heals");
    assert!(n(&|r| has(r, "|-activate|p1a: Cradily|move: Ingrain")) >= 8, "Ingrain blocks a phaze");
    assert!(n(&|r| r.name == "ig_suctioncups" && has(r, "ability: Suction Cups")) >= 8, "Suction Cups outranks Ingrain");
    assert!(n(&|r| r.name == "ig_batonpass" && has(r, "[from] Baton Pass") && has(r, "|-heal|p1a: Blissey")) >= 4, "Baton Pass passes Ingrain");
    assert!(n(&|r| has(r, "|Ingrain|p2a: Sableye|[from] Snatch")) >= 8, "Snatch steals Ingrain");
    assert!(n(&|r| pair(r, &|a| a.ends_with("[from] Ingrain"), &|b| b.ends_with("[from] Ingrain"))) >= 4, "an Ingrain residual tie");
    for b in ["ig_wrap", "ig_meanlook", "ig_speedboost", "ig_sleeptalk"] {
        assert!(n(&|r| r.name == b && has(r, "|move: Ingrain")) >= 8, "{b}: Ingrain up");
    }
    // REST UNDER UPROAR
    assert!(n(&|r| has(r, "|slp|[from] Uproar")) >= 20, "Rest blocked by Uproar");
    assert!(n(&|r| r.format == "gen3ou" && has(r, "|slp|[from] Uproar")) >= 8, "Rest blocked by Uproar under gen3ou's clause shuffle");
}
