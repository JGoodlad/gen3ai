//! The gen3 full parity B6 pins (`gen3_b6_moves_v1`): SWAGGER, FLATTER, FORESIGHT, ODOR SLEUTH and
//! MAGIC COAT. Each pin asserts the sim's observable (the protocol lines and the post-script PRNG
//! seed) on one board of the golden (`tests/vectors/b6_moves_golden.txt`, seed row k=0; spec
//! `harness/probe_b6_moves.js`). Before B6 every one of the five moves PANICKED as an unmodeled
//! status move. The seed and byte differential over every board is `b6_moves_golden_test.rs`.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision, REFLECTABLE_MOVES, TARGET_BOOST_MOVES};

/// The golden's k=0 post-construction seed.
const SEED: &str = "54207,53824,8603,10882";

fn set(sp: &str, ab: &str, moves: &str, g: &str) -> String {
    format!("{sp}|{sp}|Leftovers|{ab}|{moves}|Hardy|85,85,85,85,85,85|{g}||||")
}

/// Play `script` (1-based move slots) and return the lines and the final PRNG seed.
fn play(p1: &str, p2: &str, script: &[(usize, usize)]) -> (Vec<String>, String) {
    play_at(SEED, p1, p2, script)
}

fn play_at(seed: &str, p1: &str, p2: &str, script: &[(usize, usize)]) -> (Vec<String>, String) {
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".into(),
        seed: Some(seed.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let script: Vec<ScriptDecision> =
        script.iter().map(|&(a, c)| ScriptDecision::both(Choice::Move(a - 1), Choice::Move(c - 1))).collect();
    let st = b.state_mut().unwrap();
    let (_o, lines) = st.run_full_battle_logged(&script, &d);
    // The section-break `|`, `|upkeep` and `|t:|` lines carry nothing these pins read.
    let kept = |l: &String| !(l.is_empty() || l == "|" || l.starts_with("|upkeep") || l.starts_with("|t:|"));
    (lines.into_iter().map(|l| l.0).filter(kept).collect(), st.prng_seed())
}

fn after<'a>(lines: &'a [String], anchor: &str) -> &'a [String] {
    let i = lines.iter().position(|l| l == anchor).unwrap_or_else(|| panic!("no {anchor:?} in {lines:#?}"));
    &lines[i + 1..]
}

/// A node one-liner over the RESOLVED gen-3 dex (fails loudly without node, like the handler audit).
fn node(js: &str) -> String {
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(format!(
            "const {{Dex}}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.mod('gen3');\
             const all=d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3);{js}"
        ))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("node must be runnable (see root CLAUDE.md worktree setup)");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

const STATS: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

/// B6M1 — the hardcoded TARGET-boost table is every gen-3 Status move with a POSITIVE `boosts`
/// entry on a foe (`normal`) target, as the resolved dex states it (`data/` carries no such field;
/// the negative ones are `statDropBoosts`).
#[test]
fn b6m1_the_target_boost_table_matches_the_dex() {
    let want = node(
        "console.log(all.filter(m=>m.category==='Status'&&m.boosts&&m.target==='normal'&&Object.values(m.boosts).some(v=>v>0))\
         .map(m=>m.id+':'+Object.entries(m.boosts).map(([k,v])=>k+'='+v).join('+')).sort().join(','))",
    );
    let mut ours: Vec<String> =
        TARGET_BOOST_MOVES.iter().map(|(id, (s, n))| format!("{id}:{}={n}", STATS[*s])).collect();
    ours.sort();
    assert_eq!(want, ours.join(","));
}

/// B6M2 — the reflectable list is every gen-3 move with `flags.reflectable`, SORTED (the engine
/// binary-searches it).
#[test]
fn b6m2_the_reflectable_list_matches_the_dex() {
    let want = node("console.log(all.filter(m=>m.flags.reflectable).map(m=>m.id).sort().join(','))");
    assert_eq!(want, REFLECTABLE_MOVES.join(","));
}

fn nidoking() -> String {
    set("Nidoking", "PoisonPoint", "swagger,flatter,splash,tackle", "M")
}

/// B6P1 — the two halves are independent: a Swagger into an already-confused target still lands
/// its +2 Atk and prints NO `-fail` (the confusion half fails silently).
#[test]
fn b6p1_swagger_into_a_confused_target_still_boosts() {
    let (lines, seed) = play(&nidoking(), &set("Snorlax", "ThickFat", "splash", "M"), &[(1, 1), (1, 1), (1, 1)]);
    let t2 = after(&lines, "|turn|2");
    assert_eq!(t2[0], "|move|p1a: Nidoking|Swagger|p2a: Snorlax");
    assert_eq!(t2[1], "|-boost|p2a: Snorlax|atk|2");
    assert!(!lines.iter().any(|l| l.starts_with("|-fail|")), "{lines:#?}");
    assert_eq!(seed, "16958,43956,39898,6335");
}

/// B6P2 — at the +6 cap the delta-0 `-boost|…|0` prints; into an already-confused target that
/// combines to a `false` move result: `[still]` + `-fail|<user>` AFTER the delta-0 line.
#[test]
fn b6p2_a_capped_swagger_into_confusion_fails_after_the_delta0_line() {
    let (lines, seed) =
        play(&nidoking(), &set("Snorlax", "ThickFat", "bellydrum,splash", "M"), &[(3, 1), (1, 2), (1, 2), (1, 2)]);
    let t3 = after(&lines, "|turn|3");
    assert_eq!(t3[0], "|move|p1a: Nidoking|Swagger||[still]");
    assert_eq!(t3[1], "|-boost|p2a: Snorlax|atk|0");
    assert_eq!(t3[2], "|-fail|p1a: Nidoking");
    assert_eq!(seed, "16958,43956,39898,6335");
}

/// B6P3 — Own Tempo refuses only the confusion (`null`), so a capped Swagger into it does NOT fail.
#[test]
fn b6p3_own_tempo_refuses_only_the_confusion() {
    let (lines, seed) =
        play(&nidoking(), &set("Slowbro", "OwnTempo", "bellydrum,splash", "M"), &[(1, 2), (3, 1), (1, 2), (1, 2)]);
    let t3 = after(&lines, "|turn|3");
    assert_eq!(t3[0], "|move|p1a: Nidoking|Swagger|p2a: Slowbro");
    assert_eq!(t3[1], "|-boost|p2a: Slowbro|atk|0");
    assert_eq!(t3[2], "|-immune|p2a: Slowbro|confusion|[from] ability: Own Tempo");
    assert!(!lines.iter().any(|l| l.starts_with("|-fail|")));
    assert_eq!(seed, "58575,13552,4985,46811");
}

/// B6P4 — an identified Ghost takes Fighting (resisted through Poison) and Normal hits, and a
/// second Foresight fails `[still]` + `-fail`.
#[test]
fn b6p4_foresight_lets_normal_and_fighting_hit_a_ghost() {
    let (lines, seed) = play(
        &set("Machamp", "Guts", "foresight,odorsleuth,crosschop,bodyslam", "M"),
        &set("Gengar", "Levitate", "splash", "M"),
        &[(1, 1), (3, 1), (4, 1), (1, 1)],
    );
    assert!(lines.iter().any(|l| l == "|-start|p2a: Gengar|Foresight"));
    let t2 = after(&lines, "|turn|2");
    assert_eq!(t2[2], "|move|p1a: Machamp|Cross Chop|p2a: Gengar");
    assert_eq!(t2[3], "|-resisted|p2a: Gengar");
    assert_eq!(t2[4], "|-damage|p2a: Gengar|173/282");
    let t3 = after(&lines, "|turn|3");
    assert_eq!(t3[3], "|-damage|p2a: Gengar|73/282");
    let t4 = after(&lines, "|turn|4");
    assert_eq!(t4[0], "|move|p1a: Machamp|Foresight||[still]");
    assert_eq!(t4[1], "|-fail|p1a: Machamp");
    assert_eq!(seed, "58083,58803,52012,1312");
}

/// B6P5 — Foresight negates Glare's Ghost immunity too (Glare is `ignoreImmunity: false`).
#[test]
fn b6p5_glare_paralyses_an_identified_ghost() {
    let (lines, seed) = play(
        &set("Arbok", "Intimidate", "foresight,glare,splash", "M"),
        &set("Gengar", "Levitate", "splash", "M"),
        &[(2, 1), (1, 1), (2, 1)],
    );
    assert!(after(&lines, "|turn|1").iter().take(4).any(|l| l == "|-immune|p2a: Gengar"));
    assert!(after(&lines, "|turn|3").iter().any(|l| l == "|-status|p2a: Gengar|par"), "{lines:#?}");
    assert_eq!(seed, "56049,5757,7085,16952");
}

fn alakazam(moves: &str) -> String {
    set("Alakazam", "InnerFocus", moves, "M")
}

/// B6P6 — the bounce: `|move|<holder>|<Move>|<user>|[from] Magic Coat`, the bounced status on the
/// user; the coat is single-use and single-turn (a second bounce into the paralysed user shows the
/// same-status `-fail`).
#[test]
fn b6p6_magic_coat_bounces_thunder_wave() {
    let (lines, seed) =
        play(&alakazam("magiccoat,splash"), &set("Snorlax", "ThickFat", "thunderwave,splash", "F"), &[(1, 1), (2, 1), (1, 1)]);
    let t1 = after(&lines, "|turn|1");
    assert_eq!(t1[0], "|move|p1a: Alakazam|Magic Coat|p1a: Alakazam");
    assert_eq!(t1[1], "|-singleturn|p1a: Alakazam|move: Magic Coat");
    assert_eq!(t1[2], "|move|p2a: Snorlax|Thunder Wave|p1a: Alakazam");
    assert_eq!(t1[3], "|move|p1a: Alakazam|Thunder Wave|p2a: Snorlax|[from] Magic Coat");
    assert_eq!(t1[4], "|-status|p2a: Snorlax|par");
    assert_eq!(seed, "40014,51123,57753,14165");
}

/// B6P7 — a bounced move into the ORIGINAL user's Substitute: the retro `[still]` comes AFTER
/// the `[from]` fold (`|move|…|Thunder Wave||[from] Magic Coat|[still]`), then `-fail|<holder>`.
#[test]
fn b6p7_a_bounce_into_the_users_substitute() {
    let (lines, seed) = play(
        &alakazam("magiccoat,splash"),
        &set("Snorlax", "ThickFat", "substitute,thunderwave,swagger,leechseed", "F"),
        &[(2, 1), (1, 2), (1, 3), (1, 4)],
    );
    let t2 = after(&lines, "|turn|2");
    assert_eq!(t2[3], "|move|p1a: Alakazam|Thunder Wave||[from] Magic Coat|[still]");
    assert_eq!(t2[4], "|-fail|p1a: Alakazam");
    assert!(lines.iter().any(|l| l == "|move|p1a: Alakazam|Swagger||[from] Magic Coat|[still]"));
    assert_eq!(seed, "17857,55236,24456,61516");
}

/// B6P8 — the coat (TryHit) runs BEFORE the natural-immunity report: a Thunder Wave into a GROUND
/// coat holder is bounced, not `-immune`.
#[test]
fn b6p8_a_ground_holder_still_bounces_thunder_wave() {
    let (lines, seed) = play(
        &set("Marowak", "RockHead", "magiccoat,splash", "M"),
        &set("Snorlax", "ThickFat", "thunderwave,splash", "F"),
        &[(1, 1), (1, 1)],
    );
    let t1 = after(&lines, "|turn|1");
    assert_eq!(t1[3], "|move|p1a: Marowak|Thunder Wave|p2a: Snorlax|[from] Magic Coat");
    assert_eq!(t1[4], "|-status|p2a: Snorlax|par");
    assert_eq!(seed, "18118,29896,22950,6353");
}

/// B6P9 — the bounce sets the HOLDER's `lastMoveUsed` to the bounced move, so a later Conversion 2
/// into it reads Thunder Wave's Electric (the `sample` over Electric's resists picks Electric here).
#[test]
fn b6p9_conversion2_reads_the_bounced_move() {
    let p2 = "Porygon2|Porygon2|Leftovers|Limber|thunderwave,conversion2,splash|Hardy|0,85,85,85,0,252|N||||";
    let (lines, seed) = play(&set("Snorlax", "ThickFat", "magiccoat,splash", "M"), p2, &[(1, 1), (2, 2), (2, 3)]);
    assert!(lines.iter().any(|l| l == "|-start|p2a: Porygon2|typechange|Electric"), "{lines:#?}");
    assert_eq!(seed, "56049,5757,7085,16952");
}

/// B6P10 — a Yawn bounced into an already-statused user takes the up-front `[still]` form, with the
/// `[from]` fold FIRST: `|move|<holder>|Yawn||[from] Magic Coat|[still]` + `-fail|<holder>`.
#[test]
fn b6p10_a_bounced_yawn_into_a_statused_user() {
    let (lines, seed) = play(
        &alakazam("magiccoat,thunderwave,splash"),
        &set("Snorlax", "ThickFat", "thunderwave,toxic,yawn", "F"),
        &[(2, 3), (1, 1), (1, 2), (1, 3), (1, 3)],
    );
    let i = lines.iter().position(|l| l == "|move|p1a: Alakazam|Yawn||[from] Magic Coat|[still]").expect("the bounced Yawn");
    assert_eq!(lines[i - 1], "|move|p2a: Snorlax|Yawn|p1a: Alakazam");
    assert_eq!(lines[i + 1], "|-fail|p1a: Alakazam");
    assert_eq!(seed, "23921,9067,30121,39198");
}

/// B6P11 — an unused coat registers the residual duration handler: a Magic Coat MIRROR at equal
/// speed draws one more tie-shuffle than the both-Splash control. Asserted as the sim's seed.
#[test]
fn b6p11_a_magic_coat_mirror_draws_the_residual_tie() {
    let d = Dex::for_gen(3);
    let run = |a: usize| -> String {
        let opts = BattleOptions {
            format_id: "gen3customgame".into(),
            seed: Some("11732,5880,48989,63478".into()),
            p1: PlayerOptions { name: "P1".into(), team: PackedTeam(alakazam("magiccoat,splash")) },
            p2: PlayerOptions { name: "P2".into(), team: PackedTeam(set("Alakazam", "InnerFocus", "magiccoat,splash", "F")) },
        };
        let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
        let st = b.state_mut().unwrap();
        let script = vec![ScriptDecision::both(Choice::Move(a), Choice::Move(a)); 2];
        let _ = st.run_full_battle_logged(&script, &d);
        st.prng_seed()
    };
    assert_eq!(run(0), "34803,36341,5011,29448", "the coat mirror");
}

/// B6P12 — an identified target's POSITIVE evasion reads 0: into +3 evasion, this seed's Cross Chop
/// roll (64) hits at accuracy 80, where the unidentified 80 / 2 = 40 would miss (golden row
/// `fs_evasion_ghost` k=1; the Foresight itself rolled against the full +3).
#[test]
fn b6p12_foresight_ignores_positive_evasion() {
    let gengar = "Gengar|Gengar|Leftovers|Levitate|doubleteam,splash|Hardy|0,85,85,85,0,252|M||||";
    let (lines, seed) = play_at(
        "29957,52848,14220,35233",
        &set("Machamp", "Guts", "foresight,odorsleuth,crosschop,bodyslam", "M"),
        gengar,
        &[(4, 1), (4, 1), (4, 1), (1, 2), (3, 2), (3, 2), (3, 2)],
    );
    let t5 = after(&lines, "|turn|5");
    assert_eq!(t5[2], "|move|p1a: Machamp|Cross Chop|p2a: Gengar");
    assert_eq!(t5[3], "|-resisted|p2a: Gengar");
    assert_eq!(seed, "8369,40010,31953,16906");
}

/// B6P13 — the coat (TryHit) precedes Yawn's own already-statused `onTryHit`: a Yawn into a BURNED
/// coat holder keeps its normal announce and is bounced (probe MC34), where without the coat it
/// would fail `[still]`.
#[test]
fn b6p13_a_statused_holder_still_bounces_yawn() {
    let (lines, seed) =
        play(&alakazam("magiccoat,splash"), &set("Snorlax", "ThickFat", "willowisp,yawn", "F"), &[(2, 1), (1, 2), (1, 2)]);
    assert!(lines.iter().any(|l| l == "|-status|p1a: Alakazam|brn"));
    let t2 = after(&lines, "|turn|2");
    assert_eq!(t2[2], "|move|p2a: Snorlax|Yawn|p1a: Alakazam");
    assert_eq!(t2[3], "|move|p1a: Alakazam|Yawn|p2a: Snorlax|[from] Magic Coat");
    assert_eq!(t2[4], "|-start|p2a: Snorlax|move: Yawn|[of] p1a: Alakazam");
    assert_eq!(seed, "18118,29896,22950,6353");
}
