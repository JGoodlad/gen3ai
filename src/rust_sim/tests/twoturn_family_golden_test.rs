//! The TWO-TURN family **golden differential** (`gen3_twoturn_family_v1`, gen3 full parity B4).
//!
//! Razor Wind, Sky Attack, Skull Bash, Fly, Bounce, Dig and Dive (the spec: the header of
//! `harness/probe_twoturn_family.js`). Per row, against the sim's recording
//! (`harness/gen_twoturn_family_golden.js`): the post-script PRNG SEED and the post-`|turn|1`
//! lines byte-equal, over the charge / strike cycle (high crits, Sky Attack's flinch, Bounce's
//! paralysis, accuracy misses), the semi-invulnerable state against damaging, status, fixed-damage,
//! multi-hit, field and future moves, the bypassing moves and their doubling, weather, the strike
//! aborts (paralysis, confusion, Truant, Yawn), Protect, switches, a faint-cancelled strike, a
//! mirror, the request locks (Imprison, Disable, Encore, Torment, Choice Band) and Skull Bash's cap.
//! The coverage floors are enforced here on the vector the test reads.
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
    for line in include_str!("vectors/twoturn_family_golden.txt").lines() {
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

fn kept_line(l: &str) -> bool {
    kept(l)
}

#[test]
fn the_twoturn_family_matches_showdown() {
    let rows = load();
    assert!(rows.len() >= 450, "golden shrank to {} rows", rows.len());
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
        let got: Vec<String> = all[t1 + 1..].iter().filter(|l| kept_line(l)).cloned().collect();
        let tag = format!("{} {} [{}]", r.name, r.mv, r.seed_before);
        if st.prng_seed() != r.seed_after {
            fails.push(format!("{tag}: SEED port {} != sim {}", st.prng_seed(), r.seed_after));
        }
        if got != r.lines {
            let i = got.iter().zip(r.lines.iter()).position(|(a, b)| a != b).unwrap_or(got.len().min(r.lines.len()));
            fails.push(format!("{tag}: lines differ at {i}: port {:?} / sim {:?}", got.get(i), r.lines.get(i)));
        }
    }
    assert!(fails.is_empty(), "{} of {} rows diverge:\n{}", fails.len(), rows.len(), fails.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
}

/// A strike turn LOST to a BeforeMove cant: p1 charged (`-prepare`) and its next action is a
/// `|cant|` or a confusion self-hit instead of the `[from] lockedmove` strike.
fn strike_aborted(r: &Row) -> bool {
    let mut up = false;
    for l in &r.lines {
        if l.starts_with("|-prepare|p1a: ") {
            up = true;
        } else if l.starts_with("|move|p1a: ") && l.contains("[from] lockedmove") {
            up = false;
        } else if up && (l.starts_with("|cant|p1a: ") || (l.starts_with("|-damage|p1a: ") && l.ends_with("|[from] confusion"))) {
            return true;
        }
    }
    false
}

/// The ENFORCED coverage floors: every move's charge, strike and strike abort; the accuracy-miss,
/// crit, flinch, paralysis and Def-boost branches; and per semi-invulnerable move the
/// invulnerable miss, the bypassing hit, the Perish Song and Future Sight misses (and the weather
/// skip for Dig / Dive); plus the cancelled strike, the HJK crash, the Imprison / Disable cants,
/// Protect on the strike and the strike KO.
#[test]
fn the_golden_covers_each_move_and_interaction() {
    let rows = load();
    let n = |pred: &dyn Fn(&Row) -> bool| rows.iter().filter(|r| pred(r)).count();
    let has = |r: &Row, p: &dyn Fn(&str) -> bool| r.lines.iter().any(|l| p(l));
    let names = [
        ("razorwind", "Razor Wind"),
        ("skyattack", "Sky Attack"),
        ("skullbash", "Skull Bash"),
        ("fly", "Fly"),
        ("bounce", "Bounce"),
        ("dig", "Dig"),
        ("dive", "Dive"),
    ];
    for (mv, nm) in names {
        let prep = format!("|{nm}");
        let charge = n(&|r| r.mv == mv && has(r, &|l| l.starts_with("|-prepare|p1a: ") && l.ends_with(&prep)));
        assert!(charge >= 40, "{mv}: charge rows {charge}");
        let strike_tail = format!("|{nm}|");
        let strike = n(&|r| r.mv == mv && has(r, &|l| l.starts_with("|move|p1a: ") && l.contains(&strike_tail) && l.ends_with("|[from] lockedmove")));
        assert!(strike >= 30, "{mv}: strike rows {strike}");
        let aborted = n(&|r| r.mv == mv && strike_aborted(r));
        assert!(aborted >= 8, "{mv}: strike-abort rows {aborted}");
    }
    for (mv, nm) in [("skyattack", "Sky Attack"), ("fly", "Fly"), ("bounce", "Bounce")] {
        let tail = format!("|{nm}|");
        let k = n(&|r| r.mv == mv && has(r, &|l| l.starts_with("|move|p1a: ") && l.contains(&tail) && l.ends_with("|[from] lockedmove|[miss]")));
        assert!(k >= 3, "{mv}: strike accuracy-miss rows {k}");
    }
    for mv in ["razorwind", "skyattack"] {
        assert!(n(&|r| r.mv == mv && has(r, &|l| l.starts_with("|-crit|p2a: "))) >= 8, "{mv}: crit rows");
    }
    assert!(n(&|r| r.mv == "skyattack" && has(r, &|l| l.starts_with("|cant|p2a: ") && l.ends_with("|flinch"))) >= 5, "Sky Attack flinch rows");
    assert!(n(&|r| r.mv == "bounce" && has(r, &|l| l.starts_with("|-status|p2a: ") && l.ends_with("|par"))) >= 10, "Bounce paralysis rows");
    assert!(n(&|r| r.mv == "skullbash" && has(r, &|l| l.starts_with("|-boost|p1a: ") && l.ends_with("|def|1"))) >= 40, "Skull Bash +1 Def rows");
    assert!(n(&|r| r.name == "skullcap" && has(r, &|l| l.starts_with("|-boost|p1a: ") && l.ends_with("|def|0"))) >= 3, "Skull Bash capped rows");
    for (mv, bypass) in [("fly", "|Gust|"), ("bounce", "|Gust|"), ("dig", "|Earthquake|"), ("dive", "|Surf|")] {
        let miss = n(&|r| r.mv == mv && has(r, &|l| l.starts_with("|move|p2a: ") && l.contains("|p1a: ") && l.ends_with("|[miss]")));
        assert!(miss >= 20, "{mv}: invulnerable-miss rows {miss}");
        let hit = n(&|r| r.mv == mv && r.name == "bypass" && has(r, &|l| l.starts_with("|move|p2a: ") && l.contains(bypass) && !l.ends_with("[miss]")));
        assert!(hit >= 3, "{mv}: bypassing-hit rows {hit}");
        let perish = n(&|r| r.mv == mv && r.name == "field" && has(r, &|l| l.starts_with("|-miss|p2a: ") && l.contains("|p1a: ")));
        assert!(perish >= 3, "{mv}: Perish Song miss rows {perish}");
        let fs = n(&|r| {
            r.mv == mv
                && r.name == "futuresight"
                && r.lines.windows(2).any(|w| w[0].starts_with("|-end|p1a: ") && w[0].ends_with("|move: Future Sight") && w[1].starts_with("|-miss|"))
        });
        assert!(fs >= 3, "{mv}: Future Sight miss rows {fs}");
    }
    for (mv, w) in [("dig", "Sandstorm"), ("dive", "Hail")] {
        // A charge turn whose residual chips the foe but not the user.
        let foe_chip = format!("|[from] {w}");
        let skipped = n(&|r| {
            r.mv == mv && r.name == "weather" && {
                let mut found = false;
                for (i, l) in r.lines.iter().enumerate() {
                    if l.starts_with("|-prepare|p1a: ") {
                        let end = r.lines[i..].iter().position(|x| x == "|upkeep").map_or(r.lines.len(), |k| i + k);
                        let seg = &r.lines[i..end];
                        if seg.iter().any(|x| x.starts_with("|-damage|p2a: ") && x.ends_with(&foe_chip))
                            && !seg.iter().any(|x| x.starts_with("|-damage|p1a: ") && x.ends_with(&foe_chip))
                        {
                            found = true;
                        }
                    }
                }
                found
            }
        });
        assert!(skipped >= 3, "{mv}: weather-skip rows {skipped}");
    }
    assert!(n(&|r| r.name == "explode" && has(r, &|l| l == "|faint|p2a: Gengar") && r.script.iter().any(|d| d.p1.is_none())) >= 8, "faint-cancelled strike rows");
    assert!(n(&|r| r.name == "attacks" && has(r, &|l| l.starts_with("|move|p2a: ") && l.contains("|High Jump Kick|") && l.ends_with("|[miss]"))) >= 4, "HJK crash rows");
    assert!(n(&|r| r.name == "lock_imprison" && has(r, &|l| l.starts_with("|cant|p1a: ") && l.contains("|move: Imprison|"))) >= 6, "Imprison strike cant rows");
    assert!(n(&|r| r.name == "lock_disable" && has(r, &|l| l.starts_with("|cant|p1a: ") && l.contains("|Disable|"))) >= 6, "Disable strike cant rows");
    assert!(n(&|r| r.name == "protect" && has(r, &|l| l.starts_with("|-activate|p2a: ") && l.ends_with("|Protect"))) >= 14, "Protect-on-strike rows");
    assert!(n(&|r| r.name == "choice" && has(r, &|l| l.starts_with("|faint|p2a: "))) >= 14, "strike KO rows");
}
