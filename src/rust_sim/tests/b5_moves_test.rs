//! The B5 moves pins (`gen3_b5_moves_v1`, gen3 full parity B5): Spite, Grudge, Snore, Nightmare,
//! Ingrain. Each pin asserts the sim's observables — the protocol lines and the post-script PRNG
//! seed, recorded from Showdown on the same teams and choices (`new Battle({seed: [17, 29, 43, 59]})`,
//! whose post-start seed — the one the port starts from — is 18944,54804,63962,18698; the settled spec
//! is `harness/probe_b5_moves.js`); the seed and byte differential over every board is
//! `b5_moves_golden_test.rs`. The Rest-under-Uproar pin is in `regression_test.rs`.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision, SLEEP_USABLE_MOVES};

/// The sim's seed after `setPlayer` (team construction) from `[17, 29, 43, 59]`.
const SEED: &str = "18944,54804,63962,18698";

fn set(species: &str, item: &str, ability: &str, moves: &str, evs: &str, gender: &str, lv: &str) -> String {
    format!("{species}|{species}|{item}|{ability}|{moves}|Hardy|{evs}|{gender}|||{lv}|")
}
fn mon(species: &str, ability: &str, moves: &str) -> String {
    set(species, "", ability, moves, "85,85,85,85,85,85", "M", "")
}
fn blissey(moves: &str) -> String {
    set("Blissey", "", "NaturalCure", moves, "85,85,85,85,85,85", "F", "")
}

/// Play the script in gen3customgame; returns the lines (the `|t:|`, empty, `|` and `|upkeep`
/// separators dropped) from the first `|turn|1` on, and the final PRNG seed.
fn play(p1: &str, p2: &str, script: &[ScriptDecision]) -> (Vec<String>, String) {
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".into(),
        seed: Some(SEED.into()),
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

/// Assert `want` appears in `lines` as a contiguous run.
fn has_run(lines: &[String], want: &[&str]) {
    assert!(
        lines.windows(want.len()).any(|w| w.iter().zip(want).all(|(a, b)| a == b)),
        "missing the run {want:#?} in {lines:#?}"
    );
}

/// B5-0 — the `sleepUsable` set is every gen-3 move whose RESOLVED data carries `sleepUsable`, read
/// off Showdown (fails loudly without node, like `recharge_family_test.rs` RF1).
#[test]
fn b5_0_the_sleep_usable_set_is_showdowns() {
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(
            "const {Dex}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.forFormat('gen3customgame');\
             console.log(d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&m.sleepUsable)\
             .map(m=>m.id).sort().join(','))",
        )
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("node must be runnable (see root CLAUDE.md worktree setup)");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let mut ours: Vec<&str> = SLEEP_USABLE_MOVES.to_vec();
    ours.sort_unstable();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), ours.join(","));
}

/// B5-1 — SPITE draws its `random(2, 6)` BEFORE it looks for a lastMove: a faster Spite into a
/// mon that has not moved fails (`[still]` + `-fail|<user>`) but still consumed the roll.
#[test]
fn b5_1_spite_draws_its_roll_before_failing_on_no_last_move() {
    let (lines, seed) = play(&mon("Jolteon", "VoltAbsorb", "spite,splash"), &mon("Snorlax", "ThickFat", "bodyslam,splash"), &[mv(1, 1)]);
    has_run(&lines, &["|move|p1a: Jolteon|Spite||[still]", "|-fail|p1a: Jolteon", "|move|p2a: Snorlax|Body Slam|p1a: Jolteon"]);
    assert_eq!(seed, "38721,49529,30239,49603");
}

/// B5-2 — SPITE prints the ROLL, not the PP it removed (a 1-PP Zap Cannon Spited for 5 prints 5
/// and goes to 0), and it reads the lastMove the target RAN (Splash, once Zap Cannon is empty).
#[test]
fn b5_2_spite_prints_the_roll_and_drains_to_zero() {
    let (lines, seed) = play(
        &mon("Gengar", "Levitate", "spite,splash"),
        &set("Snorlax", "Leftovers", "ThickFat", "zapcannon,splash", "252,0,252,0,4,0", "M", ""),
        &[mv(2, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 2)],
    );
    let spites: Vec<&String> = lines.iter().filter(|l| l.contains("|move: Spite|")).collect();
    assert_eq!(
        spites,
        [
            "|-activate|p2a: Snorlax|move: Spite|zapcannon|3",
            "|-activate|p2a: Snorlax|move: Spite|zapcannon|5",
            "|-activate|p2a: Snorlax|move: Spite|splash|5",
        ]
    );
    has_run(&lines, &["|move|p2a: Snorlax|Splash|p2a: Snorlax", "|-nothing"]);
    assert_eq!(seed, "51362,17969,35293,50061");
}

/// B5-3 — GRUDGE zeroes the KILLER's lastMove, which is SLEEP TALK when Sleep Talk called the KO
/// move (`moveUsed` records the move `runMove` ran, not the one `useMove` called).
#[test]
fn b5_3_grudge_zeroes_sleep_talk_when_it_called_the_ko() {
    let (lines, seed) = play(
        &[mon("Shedinja", "WonderGuard", "slash,grudge"), blissey("splash")].join("]"),
        &mon("Snorlax", "ThickFat", "rest,sleeptalk,shadowball"),
        &[mv(1, 1), mv(2, 2)],
    );
    has_run(&lines, &["|faint|p1a: Shedinja", "|-activate|p2a: Snorlax|move: Grudge|Sleep Talk"]);
    assert_eq!(seed, "41506,59591,10380,6292");
}

/// B5-4 — GRUDGE ends at the holder's next move ATTEMPT (`onBeforeMovePriority: 100`, before the
/// sleep cant at 10): a holder Spored after its Grudge loses the Grudge on its `|cant|…|slp` turn,
/// so the KO that follows wipes nothing.
#[test]
fn b5_4_grudge_is_lost_on_a_sleep_cant() {
    let (lines, seed) = play(
        &[mon("Shedinja", "WonderGuard", "grudge,splash"), blissey("splash")].join("]"),
        &mon("Snorlax", "ThickFat", "spore,shadowball"),
        &[mv(1, 1), mv(2, 2)],
    );
    has_run(&lines, &["|cant|p1a: Shedinja|slp", "|move|p2a: Snorlax|Shadow Ball|p1a: Shedinja"]);
    assert!(lines.iter().any(|l| l == "|faint|p1a: Shedinja"));
    assert!(!lines.iter().any(|l| l.contains("move: Grudge|")), "{lines:#?}");
    assert_eq!(seed, "38721,49529,30239,49603");
}

/// B5-5 — a killer KO'd by its OWN recoil still loses the move: the holder's corpse is drained
/// first and `source.fainted` is not yet set (`|faint|` holder → `-activate` → `|faint|` killer).
#[test]
fn b5_5_grudge_reaches_a_killer_its_own_recoil_koed() {
    let (lines, seed) = play(
        &[set("Pichu", "", "Static", "grudge,splash", "85,85,85,85,85,85", "M", "5"), blissey("splash")].join("]"),
        &[mon("Shedinja", "WonderGuard", "doubleedge,splash"), blissey("splash")].join("]"),
        &[mv(1, 2), mv(2, 1)],
    );
    has_run(
        &lines,
        &["|faint|p1a: Pichu", "|-activate|p2a: Shedinja|move: Grudge|Double-Edge", "|faint|p2a: Shedinja"],
    );
    assert_eq!(seed, "20996,12330,49662,16889");
}

/// B5-6 — an AWAKE SNORE fails at its `onTry` with NO line (no `[still]`, no `-fail`) and draws
/// nothing past the announce.
#[test]
fn b5_6_an_awake_snore_is_a_silent_drawless_fail() {
    let (lines, seed) = play(&mon("Snorlax", "ThickFat", "snore,rest"), &blissey("seismictoss,splash"), &[mv(1, 2)]);
    assert_eq!(
        lines,
        ["|move|p2a: Blissey|Splash|p2a: Blissey", "|-nothing", "|move|p1a: Snorlax|Snore|p2a: Blissey", "|turn|2"]
    );
    assert_eq!(seed, "12055,55990,39841,51637");
}

/// B5-7 — an ASLEEP SNORE is `sleepUsable`: the slp `onBeforeMove` prints the cant, then the move
/// PROCEEDS and hits.
#[test]
fn b5_7_an_asleep_snore_cants_then_hits() {
    let (lines, seed) = play(&mon("Snorlax", "ThickFat", "snore,rest"), &blissey("seismictoss,splash"), &[mv(2, 1), mv(1, 1)]);
    has_run(&lines, &["|cant|p1a: Snorlax|slp", "|move|p1a: Snorlax|Snore|p2a: Blissey", "|-damage|p2a: Blissey|506/672"]);
    assert_eq!(seed, "28376,17824,3029,24061");
}

/// B5-8 — NIGHTMARE chips `baseMaxhp/4` at residual order 10 subOrder 7 (after Leftovers), and the
/// wake's `cureStatus()` ends it with `|-end|…|Nightmare|[silent]` right after the cure line.
#[test]
fn b5_8_nightmare_chips_after_leftovers_and_ends_on_waking() {
    let (lines, seed) = play(
        &mon("Gengar", "Levitate", "spore,nightmare,splash"),
        &set("Snorlax", "Leftovers", "ThickFat", "splash", "85,85,85,85,85,85", "M", ""),
        &[mv(1, 1), mv(2, 1), mv(3, 1), mv(3, 1), mv(3, 1)],
    );
    has_run(&lines, &["|-heal|p2a: Snorlax|392/482 slp|[from] item: Leftovers", "|-damage|p2a: Snorlax|272/482 slp|[from] Nightmare"]);
    has_run(&lines, &["|-curestatus|p2a: Snorlax|slp|[msg]", "|-end|p2a: Snorlax|Nightmare|[silent]"]);
    assert_eq!(lines.iter().filter(|l| l.ends_with("[from] Nightmare")).count(), 2);
    assert_eq!(seed, "38721,49529,30239,49603");
}

/// B5-9 — Natural Cure's switch-out `clearStatus()` ends NIGHTMARE too: the `[silent]` cure, then
/// the `[silent]` end, then the entrant.
#[test]
fn b5_9_natural_cure_ends_nightmare_before_the_switch_line() {
    let (lines, seed) = play(
        &mon("Gengar", "Levitate", "spore,nightmare,splash"),
        &[blissey("splash"), mon("Snorlax", "ThickFat", "splash")].join("]"),
        &[mv(1, 1), mv(2, 1), ScriptDecision::both(Choice::Move(2), Choice::Switch(1))],
    );
    has_run(
        &lines,
        &[
            "|-curestatus|p2a: Blissey|slp|[from] ability: Natural Cure|[silent]",
            "|-end|p2a: Blissey|Nightmare|[silent]",
            "|switch|p2a: Snorlax|Snorlax, M|482/482",
        ],
    );
    assert_eq!(seed, "20996,12330,49662,16889");
}

/// B5-10 — the order-10 ladder on one mon: Leftovers (4) → Leech Seed (5) → Nightmare (7).
#[test]
fn b5_10_nightmare_sorts_after_leftovers_and_leech_seed() {
    let (lines, seed) = play(
        &mon("Venusaur", "Overgrow", "spore,nightmare,leechseed"),
        &set("Snorlax", "Leftovers", "ThickFat", "splash", "85,85,85,85,85,85", "M", ""),
        &[mv(1, 1), mv(2, 1), mv(3, 1)],
    );
    has_run(
        &lines,
        &[
            "|-heal|p2a: Snorlax|392/482 slp|[from] item: Leftovers",
            "|-damage|p2a: Snorlax|332/482 slp|[from] Leech Seed|[of] p1a: Venusaur",
            "|-damage|p2a: Snorlax|212/482 slp|[from] Nightmare",
        ],
    );
    assert_eq!(seed, "44586,44725,29843,59136");
}

/// B5-11 — INGRAIN's `onDragOut` blocks Roar and Whirlwind with `-activate|…|move: Ingrain`, and,
/// like Suction Cups, draws NO switch-in `sample` (only the accuracy roll).
#[test]
fn b5_11_ingrain_blocks_a_phaze_without_the_sample_draw() {
    let (lines, seed) = play(
        &[mon("Cradily", "Chlorophyll", "ingrain,splash"), blissey("splash")].join("]"),
        &mon("Snorlax", "ThickFat", "roar,whirlwind"),
        &[mv(1, 2), mv(2, 1)],
    );
    has_run(&lines, &["|move|p2a: Snorlax|Whirlwind|p1a: Cradily", "|-activate|p1a: Cradily|move: Ingrain"]);
    has_run(&lines, &["|move|p2a: Snorlax|Roar|p1a: Cradily", "|-activate|p1a: Cradily|move: Ingrain"]);
    assert!(!lines.iter().any(|l| l.starts_with("|drag|")));
    assert_eq!(seed, "5086,37025,45627,32254");
}

/// B5-12 — INGRAIN has no `noCopy`: Baton Pass hands it on (the entrant heals `[from] Ingrain` off
/// its own max HP), and it FIRM-traps both holders (`trapped` in the request from the cast on).
#[test]
fn b5_12_baton_pass_passes_ingrain_and_its_trap() {
    let p1 = [mon("Cradily", "Chlorophyll", "ingrain,batonpass"), blissey("splash")].join("]");
    let p2 = mon("Snorlax", "ThickFat", "bodyslam,splash");
    let script = [mv(1, 1), mv(2, 1), ScriptDecision::one(0, Choice::Switch(1)), mv(1, 1)];
    let (lines, seed) = play(&p1, &p2, &script);
    has_run(&lines, &["|switch|p1a: Blissey|Blissey, F|672/672|[from] Baton Pass"]);
    assert!(lines.iter().any(|l| l == "|-heal|p1a: Blissey|331/672 par|[from] Ingrain"), "{lines:#?}");
    assert_eq!(seed, "38202,30116,18758,346");
    // The request's trap flag, decision by decision (the engine's live `is_trapped`).
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".into(),
        seed: Some(SEED.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let out = b.state_mut().expect("state").run_full_battle(&script, &d);
    assert_eq!(out.decisions[0].trapped, [true, false], "rooted after the cast");
    assert_eq!(out.decisions.last().expect("decisions").trapped, [true, false], "the Baton Pass entrant is rooted too");
}

/// B5-13 — a slot SPITE empties AFTER the target chose it fails at the target's `runMove`
/// `deductPP`: `|cant|<mon>|nopp|<Move>`, draw-free (golden board `sp_drain`, the sim's
/// post-start seed 19975,46914,35677,52368).
#[test]
fn b5_13_a_move_spite_emptied_after_the_choice_cants_nopp() {
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".into(),
        seed: Some("19975,46914,35677,52368".into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(mon("Gengar", "Levitate", "spite,splash")) },
        p2: PlayerOptions {
            name: "P2".into(),
            team: PackedTeam(set("Snorlax", "Leftovers", "ThickFat", "zapcannon,splash", "252,0,252,0,4,0", "M", "")),
        },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let st = b.state_mut().expect("state");
    let script = [mv(2, 1), mv(1, 1), mv(1, 1), mv(1, 2), mv(1, 2), mv(1, 2), mv(1, 2)];
    let (_o, lines) = st.run_full_battle_logged(&script, &d);
    let lines: Vec<String> = lines
        .into_iter()
        .map(|l| l.0)
        .filter(|l| !(l.is_empty() || l == "|" || l == "|upkeep" || l.starts_with("|t:|")))
        .collect();
    has_run(&lines, &["|-activate|p2a: Snorlax|move: Spite|zapcannon|5", "|cant|p2a: Snorlax|nopp|Zap Cannon"]);
    assert_eq!(st.prng_seed(), "22760,60535,24918,64775");
}
