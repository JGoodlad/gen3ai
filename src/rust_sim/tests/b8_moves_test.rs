//! The B8 moves pins (`gen3_b8_moves_v1`, gen3 full parity B8): Bide, Magnitude, Psywave, Present,
//! Triple Kick and the OHKO moves. Each pin asserts the sim's observables — the protocol lines (as
//! SENT; the `|t:|`, empty, `|` and `|upkeep` separators dropped) and the post-script PRNG seed,
//! recorded from Showdown on the same teams and choices (`new Battle({seed})`, `setPlayer` both sides;
//! the port starts from that post-construction seed). The settled spec is `harness/probe_b8_moves.js`;
//! the seed and byte differential over every board is `b8_moves_golden_test.rs`. The family sets the
//! engine keys on are checked against the RESOLVED `Dex.mod('gen3')` (node required, like
//! `recharge_family_test.rs` RF1).
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision, MAGNITUDE_TABLE, OHKO_MOVES, PRESENT_TABLE, RANDOM_BP_MOVES};

/// The sim's seed after `setPlayer` (team construction) from `[17, 29, 43, 59]`.
const SEED: &str = "18944,54804,63962,18698";

/// Play the script in gen3customgame from the post-construction `seed`; returns the lines from the
/// first `|turn|1` on (separators dropped) and the final PRNG seed.
fn play(seed: &str, p1: &str, p2: &str, script: &[ScriptDecision]) -> (Vec<String>, String) {
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".into(),
        seed: Some(seed.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(script, &d);
    let all: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let t1 = all.iter().position(|l| l == "|turn|1").expect("|turn|1");
    let kept = all[t1 + 1..]
        .iter()
        .filter(|l| !(l.is_empty() || *l == "|" || *l == "|upkeep" || l.starts_with("|t:|")))
        .cloned()
        .collect();
    (kept, st.prng_seed())
}

fn mv(a: usize, b: usize) -> ScriptDecision {
    ScriptDecision::both(Choice::Move(a - 1), Choice::Move(b - 1))
}

fn set(species: &str, item: &str, ability: &str, moves: &str, lv: &str) -> String {
    format!("{species}|{species}|{item}|{ability}|{moves}|Hardy|85,85,85,85,85,85|M|||{lv}|")
}

/// Run a node one-liner against the RESOLVED gen-3 dex and return its trimmed stdout.
fn node(script: &str) -> String {
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(format!("const {{Dex}}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.forFormat('gen3customgame');{script}"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("node must be runnable (see root CLAUDE.md worktree setup)");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// B8-0a — the OHKO set is every gen-3 move whose resolved data carries `ohko`.
#[test]
fn b8_0a_the_ohko_set_is_showdowns() {
    let got = node(
        "console.log(d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&m.ohko).map(m=>m.id).sort().join(','))",
    );
    assert_eq!(got, OHKO_MOVES.join(","));
}

/// B8-0b — Triple Kick is the only gen-3 `multiaccuracy` move (the engine keys its BP callback on it),
/// and its resolved `basePowerCallback` is `10 * move.hit`.
#[test]
fn b8_0b_triple_kick_is_the_only_multiaccuracy_move() {
    let got = node(
        "const ids=d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&m.multiaccuracy).map(m=>m.id);\
         const tk=d.moves.get('triplekick');console.log(ids.join(',')+'|'+tk.multihit+'|'+[1,2,3].map(h=>tk.basePowerCallback.call(null,null,null,{hit:h})).join(','))",
    );
    assert_eq!(got, "triplekick|3|10,20,30");
}

/// B8-0c — MAGNITUDE_TABLE and PRESENT_TABLE reproduce the resolved `onModifyMove` for every value of
/// its single draw (the handler evaluated with a stubbed `this.random`), and the two are the only gen-3
/// bp-0 rows that draw their base power there (`RANDOM_BP_MOVES`).
#[test]
fn b8_0c_the_random_bp_tables_are_showdowns() {
    let got = node(
        "const mg=d.moves.get('magnitude'),pr=d.moves.get('present');const a=[],b=[];\
         for(let i=0;i<100;i++){const m={basePower:0};mg.onModifyMove.call({random:()=>i},m);a.push(m.magnitude+':'+m.basePower)}\
         for(let r=0;r<10;r++){const m={basePower:0};pr.onModifyMove.call({random:()=>r},m,null,null);b.push(m.heal?'heal':m.basePower)}\
         const ids=d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&!m.basePower&&m.category!=='Status'&&m.onModifyMove&&/basePower\\s*=/.test(m.onModifyMove.toString())).map(m=>m.id).sort();\
         console.log(a.join(',')+'|'+b.join(',')+'|'+ids.join(','))",
    );
    let mut a = Vec::new();
    for i in 0..100u32 {
        let (_, m, bp) = MAGNITUDE_TABLE.iter().find(|(bound, _, _)| i < *bound).unwrap();
        a.push(format!("{m}:{bp}"));
    }
    let mut b = Vec::new();
    for r in 0..10u32 {
        let (_, bp) = PRESENT_TABLE.iter().find(|(bound, _)| r < *bound).unwrap();
        b.push(if *bp == 0 { "heal".to_string() } else { bp.to_string() });
    }
    assert_eq!(got, format!("{}|{}|{}", a.join(","), b.join(","), RANDOM_BP_MOVES.join(",")));
}

/// B8-0d — Taunt does not block the bp-0 B8 moves: their resolved gen-3 category is Physical / Special
/// by type (Magnitude, Present, Psywave, Bide, the four OHKO moves).
#[test]
fn b8_0d_taunt_does_not_block_the_b8_moves() {
    let d = Dex::for_gen(3);
    for id in ["magnitude", "present", "psywave", "bide", "fissure", "horndrill", "guillotine", "sheercold", "triplekick"] {
        assert!(!d.moves(id).unwrap().blocked_by_taunt(), "{id} must stay selectable under Taunt");
    }
}

/// B8-1 — BIDE: the cast (`|move|<u>|Bide|<u>`, one accuracy roll, `-start`), the WAIT turn (`-activate`, no `|move|`
/// line, no PP), the RELEASE (`-end`, one accuracy roll, twice the 100 stored Seismic Toss damage with no `|move|`
/// line, then `-end …|[silent]`).
#[test]
fn b8_1_bide_casts_waits_and_releases_double_the_stored_damage() {
    let (lines, seed) = play("18944,54804,63962,18698", "Raichu|Raichu||Static|bide,splash|Hardy|85,85,85,85,85,85|M||||", "Snorlax|Snorlax||ThickFat|seismictoss,splash|Hardy|85,85,85,85,85,85|M|||50|", &[mv(1, 1), mv(1, 1), mv(1, 2)]);
    assert_eq!(
        lines,
        [
            "|move|p1a: Raichu|Bide|p1a: Raichu",
            "|-start|p1a: Raichu|move: Bide",
            "|move|p2a: Snorlax|Seismic Toss|p1a: Raichu",
            "|-damage|p1a: Raichu|232/282",
            "|turn|2",
            "|-activate|p1a: Raichu|move: Bide",
            "|move|p2a: Snorlax|Seismic Toss|p1a: Raichu",
            "|-damage|p1a: Raichu|182/282",
            "|turn|3",
            "|-end|p1a: Raichu|move: Bide",
            "|-damage|p2a: Snorlax|46/246",
            "|-end|p1a: Raichu|move: Bide|[silent]",
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|turn|4",
        ]
    );
    assert_eq!(seed, "28376,17824,3029,24061");
}

/// B8-3 — the Bide release is a Normal-type `tryMoveHit` with `ignoreImmunity` unset: a Ghost is `-immune`
/// (after the accuracy roll), not hit for the stored Thunderbolt.
#[test]
fn b8_3_the_bide_release_checks_the_normal_type() {
    let (lines, seed) = play("18944,54804,63962,18698", "Raichu|Raichu||Static|bide,splash|Hardy|85,85,85,85,85,85|M||||", "Gengar|Gengar||Levitate|thunderbolt,splash|Hardy|85,85,85,85,85,85|M||||", &[mv(1, 2), mv(1, 1), mv(1, 2)]);
    assert_eq!(
        lines,
        [
            "|move|p2a: Gengar|Splash|p2a: Gengar",
            "|-nothing",
            "|move|p1a: Raichu|Bide|p1a: Raichu",
            "|-start|p1a: Raichu|move: Bide",
            "|turn|2",
            "|move|p2a: Gengar|Thunderbolt|p1a: Raichu",
            "|-resisted|p1a: Raichu",
            "|-damage|p1a: Raichu|226/282",
            "|-activate|p1a: Raichu|move: Bide",
            "|turn|3",
            "|move|p2a: Gengar|Splash|p2a: Gengar",
            "|-nothing",
            "|-end|p1a: Raichu|move: Bide",
            "|-immune|p2a: Gengar",
            "|-end|p1a: Raichu|move: Bide|[silent]",
            "|turn|4",
        ]
    );
    assert_eq!(seed, "28376,17824,3029,24061");
}

/// B8-4 — a BeforeMove cant on a locked Bide turn (`|cant|<u>|slp`) ends it: `MoveAborted` →
/// `|-end|<u>|move: Bide|[silent]` right after the cant.
#[test]
fn b8_4_a_cant_ends_bide() {
    let (lines, seed) = play("18944,54804,63962,18698", "Snorlax|Snorlax||ThickFat|bide,splash|Hardy|85,85,85,85,85,85|M||||", "Breloom|Breloom||EffectSpore|spore,tackle|Hardy|85,85,85,85,85,85|M||||", &[mv(1, 2), mv(1, 1), mv(2, 2)]);
    assert_eq!(
        lines,
        [
            "|move|p2a: Breloom|Tackle|p1a: Snorlax",
            "|-damage|p1a: Snorlax|435/482",
            "|move|p1a: Snorlax|Bide|p1a: Snorlax",
            "|-start|p1a: Snorlax|move: Bide",
            "|turn|2",
            "|move|p2a: Breloom|Spore|p1a: Snorlax",
            "|-status|p1a: Snorlax|slp|[from] move: Spore",
            "|cant|p1a: Snorlax|slp",
            "|-end|p1a: Snorlax|move: Bide|[silent]",
            "|turn|3",
            "|move|p2a: Breloom|Tackle|p1a: Snorlax",
            "|-damage|p1a: Snorlax|387/482 slp",
            "|cant|p1a: Snorlax|slp",
            "|turn|4",
        ]
    );
    assert_eq!(seed, "54484,17454,39273,1574");
}

/// B8-5 — a release into a mid-Fly foe misses with NO draw, and `attrLastMove('[miss]')` tags the last
/// `|move|` line of the turn — the foe's own Fly charge: `|move|<foe>|Fly||[still]|[miss]`.
#[test]
fn b8_5_a_release_into_a_same_turn_fly_tags_the_fly_charge() {
    let (lines, seed) = play("18944,54804,63962,18698", "Shuckle|Shuckle||Sturdy|bide,splash|Hardy|85,85,85,85,85,85|M||||", "Pidgeot|Pidgeot||KeenEye|fly,quickattack|Hardy|85,85,85,85,85,85|M||||", &[mv(1, 2), mv(1, 2), mv(1, 1)]);
    assert_eq!(
        lines,
        [
            "|move|p2a: Pidgeot|Quick Attack|p1a: Shuckle",
            "|-resisted|p1a: Shuckle",
            "|-damage|p1a: Shuckle|191/202",
            "|move|p1a: Shuckle|Bide|p1a: Shuckle",
            "|-start|p1a: Shuckle|move: Bide",
            "|turn|2",
            "|move|p2a: Pidgeot|Quick Attack|p1a: Shuckle",
            "|-resisted|p1a: Shuckle",
            "|-damage|p1a: Shuckle|181/202",
            "|-activate|p1a: Shuckle|move: Bide",
            "|turn|3",
            "|move|p2a: Pidgeot|Fly||[still]|[miss]",
            "|-prepare|p2a: Pidgeot|Fly",
            "|-end|p1a: Shuckle|move: Bide",
            "|-miss|p1a: Shuckle|p2a: Pidgeot",
            "|-end|p1a: Shuckle|move: Bide|[silent]",
            "|turn|4",
        ]
    );
    assert_eq!(seed, "20579,45021,5279,34087");
}

/// B8-6 — gen 3 cancels every remaining action after a faint, so a release turn on which the foe
/// Explodes never runs the release; the residual ticks the duration to 0 and `onEnd` prints
/// `|-end|<u>|move: Bide|[silent]` after the replacement's switch-in.
#[test]
fn b8_6_a_faint_skipped_release_ends_at_the_residual() {
    let (lines, seed) = play("18944,54804,63962,18698", "Shuckle|Shuckle||Sturdy|bide,splash|Hardy|85,85,85,85,85,85|M||||", "Electrode|Electrode||Soundproof|splash,explosion|Hardy|85,85,85,85,85,85|M||||]Blissey|Blissey||NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||", &[mv(1, 1), mv(1, 1), mv(1, 2), ScriptDecision::one(1, Choice::Switch(1))]);
    assert_eq!(
        lines,
        [
            "|move|p2a: Electrode|Splash|p2a: Electrode",
            "|-nothing",
            "|move|p1a: Shuckle|Bide|p1a: Shuckle",
            "|-start|p1a: Shuckle|move: Bide",
            "|turn|2",
            "|move|p2a: Electrode|Splash|p2a: Electrode",
            "|-nothing",
            "|-activate|p1a: Shuckle|move: Bide",
            "|turn|3",
            "|move|p2a: Electrode|Explosion|p1a: Shuckle",
            "|-resisted|p1a: Shuckle",
            "|-damage|p1a: Shuckle|145/202",
            "|faint|p2a: Electrode",
            "|switch|p2a: Blissey|Blissey, F|672/672",
            "|-end|p1a: Shuckle|move: Bide|[silent]",
            "|turn|4",
        ]
    );
    assert_eq!(seed, "38721,49529,30239,49603");
}

/// B8-8 — MAGNITUDE: the `random(100)` precedes the announce, `|-activate|<u>|move: Magnitude|<n>`
/// follows it and precedes the `-miss`, and the miss's `[miss]` lands on the `|move|` line above it.
#[test]
fn b8_8_magnitude_draws_before_the_announce_and_activates_after_it() {
    let (lines, seed) = play("18944,54804,63962,18698", "Golem|Golem||Sturdy|magnitude,splash|Hardy|85,85,85,85,85,85|M||||", "Snorlax|Snorlax|Bright Powder|ThickFat|doubleteam|Hardy|85,85,85,85,85,85|M||||", &[mv(2, 1), mv(2, 1), mv(2, 1), mv(2, 1), mv(2, 1), mv(2, 1), mv(1, 1)]);
    assert_eq!(
        lines,
        [
            "|move|p1a: Golem|Splash|p1a: Golem",
            "|-nothing",
            "|move|p2a: Snorlax|Double Team|p2a: Snorlax",
            "|-boost|p2a: Snorlax|evasion|1",
            "|turn|2",
            "|move|p1a: Golem|Splash|p1a: Golem",
            "|-nothing",
            "|move|p2a: Snorlax|Double Team|p2a: Snorlax",
            "|-boost|p2a: Snorlax|evasion|1",
            "|turn|3",
            "|move|p1a: Golem|Splash|p1a: Golem",
            "|-nothing",
            "|move|p2a: Snorlax|Double Team|p2a: Snorlax",
            "|-boost|p2a: Snorlax|evasion|1",
            "|turn|4",
            "|move|p1a: Golem|Splash|p1a: Golem",
            "|-nothing",
            "|move|p2a: Snorlax|Double Team|p2a: Snorlax",
            "|-boost|p2a: Snorlax|evasion|1",
            "|turn|5",
            "|move|p1a: Golem|Splash|p1a: Golem",
            "|-nothing",
            "|move|p2a: Snorlax|Double Team|p2a: Snorlax",
            "|-boost|p2a: Snorlax|evasion|1",
            "|turn|6",
            "|move|p1a: Golem|Splash|p1a: Golem",
            "|-nothing",
            "|move|p2a: Snorlax|Double Team|p2a: Snorlax",
            "|-boost|p2a: Snorlax|evasion|1",
            "|turn|7",
            "|move|p1a: Golem|Magnitude|p2a: Snorlax|[miss]",
            "|-activate|p1a: Golem|move: Magnitude|7",
            "|-miss|p1a: Golem|p2a: Snorlax",
            "|move|p2a: Snorlax|Double Team|p2a: Snorlax",
            "|-boost|p2a: Snorlax|evasion|0",
            "|turn|8",
        ]
    );
    assert_eq!(seed, "28376,17824,3029,24061");
}

/// B8-9 — Magnitude into a Flying foe: the draw, the announce, the `-activate`, the accuracy roll,
/// then `-immune`.
#[test]
fn b8_9_magnitude_into_flying_activates_then_is_immune() {
    let (lines, seed) = play("18944,54804,63962,18698", "Golem|Golem||Sturdy|magnitude,splash|Hardy|85,85,85,85,85,85|M||||", "Skarmory|Skarmory||KeenEye|splash|Hardy|85,85,85,85,85,85|M||||", &[mv(1, 1)]);
    assert_eq!(
        lines,
        [
            "|move|p2a: Skarmory|Splash|p2a: Skarmory",
            "|-nothing",
            "|move|p1a: Golem|Magnitude|p2a: Skarmory",
            "|-activate|p1a: Golem|move: Magnitude|6",
            "|-immune|p2a: Skarmory",
            "|turn|2",
        ]
    );
    assert_eq!(seed, "32088,63066,43368,35871");
}

/// B8-10 — PSYWAVE at level 37: accuracy, then ONE `random(50, 151)`; `r * 37 / 100` is floored to the mon.
#[test]
fn b8_10_psywave_floors_level_scaled_damage() {
    let (lines, seed) = play("18944,54804,63962,18698", "Gengar|Gengar||Levitate|psywave,splash|Hardy|85,85,85,85,85,85|M|||37|", "Snorlax|Snorlax||ThickFat|splash|Hardy|85,85,85,85,85,85|M||||", &[mv(1, 1), mv(1, 1)]);
    assert_eq!(
        lines,
        [
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|move|p1a: Gengar|Psywave|p2a: Snorlax",
            "|-damage|p2a: Snorlax|438/482",
            "|turn|2",
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|move|p1a: Gengar|Psywave|p2a: Snorlax",
            "|-damage|p2a: Snorlax|408/482",
            "|turn|3",
        ]
    );
    assert_eq!(seed, "44586,44725,29843,59136");
}

/// B8-11 — PRESENT's heal branch into a full-HP target: `|-fail|<t>|heal` and the `[still]` retro-edit
/// (`|move|<u>|Present||[still]`); the accuracy roll drew, no crit roll.
#[test]
fn b8_11_presents_heal_fails_on_a_full_hp_target() {
    let (lines, seed) = play("18944,54804,63962,18698", "Delibird|Delibird||Hustle|present,splash|Hardy|85,85,85,85,85,85|M||||", "Snorlax|Snorlax||ThickFat|splash|Hardy|85,85,85,85,85,85|M||||", &[mv(1, 1)]);
    assert_eq!(
        lines,
        [
            "|move|p1a: Delibird|Present||[still]",
            "|-fail|p2a: Snorlax|heal",
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|turn|2",
        ]
    );
    assert_eq!(seed, "32088,63066,43368,35871");
}

/// B8-12 — TRIPLE KICK: 10 / 20 / 30 BP per strike (37 then 61 then 90 off a Snorlax), and against
/// +1 evasion the third strike's accuracy re-roll misses: `-hitcount|…|2`, no miss line.
#[test]
fn b8_12_triple_kick_escalates_and_rerolls_accuracy() {
    let (lines, seed) = play("30982,33910,19571,50263", "Hitmontop|Hitmontop||Intimidate|triplekick,splash|Hardy|85,85,85,85,85,85|M||||", "Snorlax|Snorlax||ThickFat|splash,doubleteam|Hardy|85,85,85,85,85,85|M||||", &[mv(1, 2), mv(1, 2)]);
    assert_eq!(
        lines,
        [
            "|move|p1a: Hitmontop|Triple Kick|p2a: Snorlax",
            "|-supereffective|p2a: Snorlax",
            "|-damage|p2a: Snorlax|445/482",
            "|-supereffective|p2a: Snorlax",
            "|-damage|p2a: Snorlax|384/482",
            "|-supereffective|p2a: Snorlax",
            "|-damage|p2a: Snorlax|294/482",
            "|-hitcount|p2a: Snorlax|3",
            "|move|p2a: Snorlax|Double Team|p2a: Snorlax",
            "|-boost|p2a: Snorlax|evasion|1",
            "|turn|2",
            "|move|p1a: Hitmontop|Triple Kick|p2a: Snorlax",
            "|-supereffective|p2a: Snorlax",
            "|-damage|p2a: Snorlax|261/482",
            "|-supereffective|p2a: Snorlax",
            "|-damage|p2a: Snorlax|191/482",
            "|-hitcount|p2a: Snorlax|2",
            "|move|p2a: Snorlax|Double Team|p2a: Snorlax",
            "|-boost|p2a: Snorlax|evasion|1",
            "|turn|3",
        ]
    );
    assert_eq!(seed, "9621,18248,53276,33281");
}

/// B8-13a — an OHKO move whose user is below the target's level: `|-immune|<t>|[ohko]`, NO draw.
#[test]
fn b8_13a_an_ohko_from_a_lower_level_is_immune_with_no_draw() {
    let (lines, seed) = play("18944,54804,63962,18698", "Dugtrio|Dugtrio||ArenaTrap|fissure,splash|Hardy|85,85,85,85,85,85|M|||50|", "Snorlax|Snorlax||ThickFat|splash|Hardy|85,85,85,85,85,85|M||||", &[mv(1, 1)]);
    assert_eq!(
        lines,
        [
            "|move|p1a: Dugtrio|Fissure|p2a: Snorlax",
            "|-immune|p2a: Snorlax|[ohko]",
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|turn|2",
        ]
    );
    assert_eq!(seed, "12055,55990,39841,51637");
}

/// B8-13b — an OHKO hit at equal level (accuracy 30): `-damage …|0 fnt`, then `|-ohko`, then `|faint|`.
#[test]
fn b8_13b_an_ohko_hit_prints_ohko_after_the_damage() {
    let (lines, seed) = play("18944,54804,63962,18698", "Dugtrio|Dugtrio||ArenaTrap|fissure,splash|Hardy|85,85,85,85,85,85|M||||", "Snorlax|Snorlax||ThickFat|splash|Hardy|85,85,85,85,85,85|M||||]Blissey|Blissey||NaturalCure|splash|Hardy|85,85,85,85,85,85|F||||", &[mv(1, 1)]);
    assert_eq!(
        lines,
        [
            "|move|p1a: Dugtrio|Fissure|p2a: Snorlax",
            "|-damage|p2a: Snorlax|0 fnt",
            "|-ohko",
            "|faint|p2a: Snorlax",
        ]
    );
    assert_eq!(seed, "12055,55990,39841,51637");
}

/// B8-13c — STURDY: `|-immune|<t>|[from] ability: Sturdy` at TryHit, after the (level-gap) accuracy roll.
#[test]
fn b8_13c_sturdy_blocks_an_ohko_after_the_accuracy_roll() {
    let (lines, seed) = play("18944,54804,63962,18698", "Lapras|Lapras||ShellArmor|sheercold,splash|Hardy|85,85,85,85,85,85|M||||", "Golem|Golem||Sturdy|splash|Hardy|85,85,85,85,85,85|M|||1|", &[mv(1, 1)]);
    assert_eq!(
        lines,
        [
            "|move|p1a: Lapras|Sheer Cold|p2a: Golem",
            "|-immune|p2a: Golem|[from] ability: Sturdy",
            "|move|p2a: Golem|Splash|p2a: Golem",
            "|-nothing",
            "|turn|2",
        ]
    );
    assert_eq!(seed, "45997,24898,56288,4140");
}

/// B8-13d — an OHKO deals the MON's current HP, also against a Substitute: a 1-HP mon behind a 3-HP sub
/// keeps the sub (`-activate …|Substitute|[damage]`), and `-ohko` prints anyway.
#[test]
fn b8_13d_an_ohko_into_a_sub_that_outlives_its_mon() {
    let (lines, seed) = play("18944,54804,63962,18698", "Kingler|Kingler||HyperCutter|guillotine,toxic,splash|Hardy|85,85,85,85,85,85|M||||", "Snorlax|Snorlax||ThickFat|substitute,bellydrum,splash|Hardy|85,85,85,85,85,85|M|||1|", &[mv(2, 1), mv(3, 2), mv(1, 3)]);
    assert_eq!(
        lines,
        [
            "|move|p1a: Kingler|Toxic|p2a: Snorlax",
            "|-status|p2a: Snorlax|tox",
            "|move|p2a: Snorlax|Substitute|p2a: Snorlax",
            "|-start|p2a: Snorlax|Substitute",
            "|-damage|p2a: Snorlax|11/14 tox",
            "|-damage|p2a: Snorlax|10/14 tox|[from] psn",
            "|turn|2",
            "|move|p1a: Kingler|Splash|p1a: Kingler",
            "|-nothing",
            "|move|p2a: Snorlax|Belly Drum|p2a: Snorlax",
            "|-damage|p2a: Snorlax|3/14 tox",
            "|-setboost|p2a: Snorlax|atk|6|[from] move: Belly Drum",
            "|-damage|p2a: Snorlax|1/14 tox|[from] psn",
            "|turn|3",
            "|move|p1a: Kingler|Guillotine|p2a: Snorlax",
            "|-activate|p2a: Snorlax|Substitute|[damage]",
            "|-ohko",
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|-damage|p2a: Snorlax|0 fnt|[from] psn",
            "|faint|p2a: Snorlax",
            "|win|P1",
        ]
    );
    assert_eq!(seed, "5086,37025,45627,32254");
}

/// B8-14 — PRESENT: a 40-BP hit, then the heal branch `|-heal|<t>|<hp>` (+120 = floor(482 / 4), no `[from]`),
/// then a miss.
#[test]
fn b8_14_present_heals_a_quarter_with_no_crit_roll() {
    let (lines, seed) = play("54360,38109,30958,32827", "Delibird|Delibird||Hustle|present,splash|Hardy|85,85,85,85,85,85|M||||", "Snorlax|Snorlax||ThickFat|splash,bellydrum|Hardy|85,85,85,85,85,85|M||||", &[mv(1, 2), mv(1, 1), mv(1, 1)]);
    assert_eq!(
        lines,
        [
            "|move|p1a: Delibird|Present|p2a: Snorlax",
            "|-damage|p2a: Snorlax|442/482",
            "|move|p2a: Snorlax|Belly Drum|p2a: Snorlax",
            "|-damage|p2a: Snorlax|201/482",
            "|-setboost|p2a: Snorlax|atk|6|[from] move: Belly Drum",
            "|turn|2",
            "|move|p1a: Delibird|Present|p2a: Snorlax",
            "|-heal|p2a: Snorlax|321/482",
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|turn|3",
            "|move|p1a: Delibird|Present|p2a: Snorlax|[miss]",
            "|-miss|p1a: Delibird|p2a: Snorlax",
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|turn|4",
        ]
    );
    assert_eq!(seed, "24223,27797,40385,39060");
}

/// B8-7 — the King's Rock flinch secondary Bide's `onModifyMove` appends rolls against the USER (the
/// move's only target): at seed [3,3,3,3] it flinches the user, whose silent `flinch` volatile then ties
/// the `bide` volatile at the residual — one more shuffle draw than the [4,4,4,4] control.
#[test]
fn b8_7_kings_rock_flinches_the_bide_user_and_ties_the_residual() {
    let p1 = set("Raichu", "Kings Rock", "Static", "bide,splash", "");
    let p2 = set("Snorlax", "", "ThickFat", "bodyslam,splash", "");
    let want = [
        "|move|p1a: Raichu|Bide|p1a: Raichu",
        "|-start|p1a: Raichu|move: Bide",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|2",
    ];
    let (lines, seed) = play("39551,33397,57452,15090", &p1, &p2, &[mv(1, 2)]);
    assert_eq!(lines, want);
    assert_eq!(seed, "28527,22812,5204,24934", "seed [3,3,3,3]: the self-flinch adds a residual tie draw");
    let (lines, seed) = play("30890,839,54744,50263", &p1, &p2, &[mv(1, 2)]);
    assert_eq!(lines, want);
    assert_eq!(seed, "43918,6600,31468,18776", "seed [4,4,4,4]: no flinch, no tie");
}

/// B8-10b — PSYWAVE's raw float comes off a Substitute's HP (`data/mods/gen4/moves.ts` substitute); a
/// level-37 user leaves a fractional sub the port cannot hold, so it FAILS LOUD (never an approximation).
#[test]
#[should_panic(expected = "Psywave dealt the fractional")]
fn b8_10b_psywave_into_a_sub_at_a_fractional_amount_fails_loud() {
    let p1 = set("Gengar", "", "Levitate", "psywave,splash", "37");
    let p2 = set("Jolteon", "", "VoltAbsorb", "substitute,splash", "");
    let _ = play(SEED, &p1, &p2, &[mv(2, 1), mv(1, 2)]);
}
