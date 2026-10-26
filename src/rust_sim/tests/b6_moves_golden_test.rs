//! The gen3 full parity B6 **golden differential** (`gen3_b6_moves_v1`): SWAGGER, FLATTER,
//! FORESIGHT, ODOR SLEUTH and MAGIC COAT. Per row, against the sim's recording
//! (`harness/gen_b6_moves_golden.js`; spec `harness/probe_b6_moves.js`): the post-script PRNG SEED,
//! the FRAMING window and the post-`|turn|1` lines, byte-equal. Boards: each boost-confuser's hit,
//! cap, Own Tempo / Substitute / Safeguard / Protect / miss / berry; Foresight's evasion and Ghost
//! reads (Normal, Fighting, Glare, Counter, Seismic Toss, a Jump Kick crash, Wonder Guard), its
//! fail, clear and Baton Pass; and a Magic Coat board for EVERY reflectable gen-3 move plus the
//! bounce meeting the user's Substitute / immunity / ability / Safeguard / status, the holder's
//! Soundproof / Ground type / Substitute, Snatch, a mirror, a drag, Encore, Conversion 2, a Sleep
//! Talk call and Sleep Clause (`gen3ou`).
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
    for line in include_str!("vectors/b6_moves_golden.txt").lines() {
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
fn the_b6_moves_match_showdown() {
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
        // The format's own framing (`gen3_format_rules_v1`): the logged run frames as Custom Game.
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
/// interaction; the generator refuses a vector with any zero).
#[test]
fn the_golden_covers_each_branch() {
    let rows = load();
    let n = |pred: &dyn Fn(&Row) -> bool| rows.iter().filter(|r| pred(r)).count();
    let has = |r: &Row, s: &str| r.lines.iter().any(|l| l.contains(s));
    let line = |r: &Row, s: &str| r.lines.iter().any(|l| l == s);
    let at = |name: &'static str| move |r: &Row| r.name == name;
    // SWAGGER / FLATTER
    assert!(n(&|r| has(r, "|Swagger|p2a:") && line(r, "|-boost|p2a: Snorlax|atk|2")) >= 30, "Swagger's +2 Atk");
    assert!(n(&|r| has(r, "|Flatter|p2a:") && line(r, "|-boost|p2a: Snorlax|spa|1")) >= 12, "Flatter's +1 SpA");
    assert!(n(&|r| at("sw_cap")(r) && line(r, "|-boost|p2a: Snorlax|atk|0")) >= 6, "the delta-0 line at +6");
    assert!(n(&|r| at("fl_cap")(r) && line(r, "|-boost|p2a: Snorlax|spa|0")) >= 6, "Flatter's delta-0 line at +6");
    assert!(n(&|r| at("sw_plus5")(r) && line(r, "|-boost|p2a: Snorlax|atk|1")) >= 6, "+5 caps the +2 at +1");
    assert!(n(&|r| at("sw_cap")(r) && line(r, "|-fail|p1a: Nidoking")) >= 4, "capped + already confused fails");
    assert!(n(&|r| at("sw_twice")(r) && !has(r, "|-fail|")) >= 4, "into a confused target the boost lands, no -fail");
    assert!(n(&|r| at("sw_owntempo")(r) && line(r, "|-immune|p2a: Slowbro|confusion|[from] ability: Own Tempo")) >= 6, "Own Tempo");
    assert!(n(&|r| at("sw_safeguard")(r) && line(r, "|-activate|p2a: Snorlax|move: Safeguard")) >= 6, "Safeguard");
    assert!(n(&|r| at("sw_sub")(r) && line(r, "|-fail|p1a: Nidoking")) >= 6, "a Substitute blocks it");
    assert!(n(&|r| at("sw_protect")(r) && line(r, "|-activate|p2a: Snorlax|Protect")) >= 6, "Protect");
    assert!(n(&|r| at("sw_evasive")(r) && has(r, "|-miss|p1a: Nidoking")) >= 3, "a miss");
    assert!(n(&|r| (r.name == "sw_lum" || r.name == "sw_persim") && has(r, "|-enditem|p2a: Snorlax|")) >= 12, "a confusion berry");
    assert!(n(&|r| at("sw_clearbody")(r) && line(r, "|-boost|p2a: Metagross|atk|2")) >= 6, "Clear Body does not gate a raise");
    // FORESIGHT / ODOR SLEUTH
    assert!(n(&|r| line(r, "|-start|p2a: Gengar|Foresight")) >= 30, "identified Gengar");
    assert!(n(&|r| has(r, "|Odor Sleuth|p2a:") && has(r, "|Foresight")) >= 12, "Odor Sleuth's start line is Foresight");
    assert!(n(&|r| r.name.starts_with("fs_") && has(r, "|Foresight||[still]")) >= 6, "a second Foresight fails");
    assert!(n(&|r| r.name.starts_with("fs_") && has(r, "|Odor Sleuth||[still]")) >= 6, "Odor Sleuth into an identified target fails");
    assert!(n(&|r| at("fs_ghost")(r) && line(r, "|-resisted|p2a: Gengar")) >= 4, "Fighting into identified Ghost/Poison is resisted");
    assert!(n(&|r| at("fs_control")(r) && line(r, "|-immune|p2a: Gengar")) >= 6, "the unidentified control is immune");
    assert!(n(&|r| at("fs_glare")(r) && line(r, "|-status|p2a: Gengar|par")) >= 3, "Glare paralyses an identified Ghost");
    assert!(n(&|r| at("fs_switch")(r) && line(r, "|-immune|p2a: Gengar")) >= 6, "switching out clears it");
    assert!(n(&|r| at("fs_baton")(r) && line(r, "|-immune|p2a: Misdreavus")) >= 6, "Baton Pass does not pass it");
    assert!(n(&|r| at("fs_jumpkick")(r) && has(r, "|-damage|p1a: Hitmonlee")) >= 6, "the Jump Kick crash on an identified Ghost");
    assert!(n(&|r| at("fs_seismic")(r) && has(r, "|Counter|p2a: Gengar") && has(r, "|-damage|p2a: Gengar")) >= 4, "Counter / Seismic Toss hit");
    assert!(n(&|r| at("fs_wonderguard")(r) && has(r, "[from] ability: Wonder Guard")) >= 6, "Wonder Guard still blocks");
    assert!(n(&|r| at("fs_protect")(r) && line(r, "|-activate|p2a: Gengar|Protect")) >= 6, "Protect blocks it");
    assert!(n(&|r| at("fs_sub")(r) && line(r, "|-start|p2a: Gengar|Foresight")) >= 6, "bypasssub");
    assert!(n(&|r| r.name.starts_with("fs_evasion") && has(r, "|-miss|p1a: Machamp")) >= 3, "the evasion boards miss somewhere");
    // MAGIC COAT
    let coat = n(&|r| has(r, "|-singleturn|p1a: ") && has(r, "|move: Magic Coat"));
    assert!(coat >= 400, "coat casts {coat}");
    for mv in pokesim::turn::REFLECTABLE_MOVES {
        let name = format!("mc_{mv}");
        assert!(
            n(&|r| r.name == name && has(r, "|p2a: Snorlax|[from] Magic Coat")) >= 6,
            "{mv}: every reflectable move bounces on its own board"
        );
    }
    assert!(n(&|r| has(r, "|[from] Magic Coat|[still]")) >= 30, "a bounced move that did nothing");
    assert!(n(&|r| has(r, "|[from] Magic Coat|[miss]")) >= 20, "a bounced move that missed");
    assert!(n(&|r| r.name.starts_with("mc_user_") && has(r, "[from] Magic Coat") && has(r, "|-immune|p2a:")) >= 20, "the user's immunity");
    assert!(n(&|r| at("mc_user_ground")(r) && line(r, "|-immune|p2a: Marowak")) >= 6, "a bounced Thunder Wave into Ground");
    assert!(n(&|r| at("mc_holder_ground")(r) && line(r, "|-status|p2a: Snorlax|par")) >= 6, "a Ground holder still bounces");
    assert!(n(&|r| at("mc_holder_soundproof")(r) && has(r, "|Supersonic|p2a: Snorlax|[from] Magic Coat")) >= 6, "the coat before Soundproof");
    assert!(n(&|r| at("mc_holder_sub")(r) && has(r, "[from] Magic Coat")) >= 6, "the coat before the holder's own Substitute");
    assert!(n(&|r| at("mc_user_sub")(r) && has(r, "|[from] Magic Coat|[still]")) >= 6, "the bounce into the user's Substitute");
    assert!(n(&|r| at("mc_user_safeguard")(r) && line(r, "|-activate|p2a: Snorlax|move: Safeguard")) >= 6, "the user's Safeguard");
    assert!(n(&|r| at("mc_miss")(r) && has(r, "|-miss|p2a: Snorlax|p1a: Alakazam")) >= 6, "an original miss: no bounce");
    assert!(n(&|r| at("mc_mirror")(r) || at("mc_mirror_control")(r)) >= 12, "the residual tie board");
    assert!(n(&|r| at("mc_snatch")(r) && has(r, "[from] Snatch")) >= 6, "Snatch");
    assert!(n(&|r| at("mc_drag")(r) && has(r, "|drag|p1a: Blissey")) >= 6, "a dragged holder");
    assert!(n(&|r| at("mc_encore")(r) && line(r, "|-start|p1a: Snorlax|Encore")) >= 6, "Encore reads lastMove = Magic Coat");
    assert!(n(&|r| at("mc_conversion2")(r) && has(r, "|typechange|")) >= 10, "Conversion 2 reads the bounced move");
    assert!(n(&|r| at("mc_sleeptalk")(r) && has(r, "[from] Sleep Talk") && has(r, "[from] Magic Coat")) >= 4, "a Sleep-Talk-called bounce");
    assert!(n(&|r| at("mc_sleepclause")(r) && has(r, "Sleep Clause Mod activated") && r.format == "gen3ou") >= 6, "Sleep Clause on a bounce");
    assert!(n(&|r| at("mc_holder_statused")(r) && has(r, "|-status|p1a: Alakazam|brn") && has(r, "|Yawn|p2a: Snorlax|[from] Magic Coat")) >= 3, "a statused holder still bounces Yawn");
    assert!(n(&|r| at("mc_taunt")(r) && line(r, "|-start|p1a: Alakazam|move: Taunt")) >= 6, "Taunt is not reflectable");
    assert!(n(&|r| at("mc_foresight_user")(r) && line(r, "|-start|p1a: Alakazam|Foresight")) >= 4, "Foresight is not reflectable");
}
