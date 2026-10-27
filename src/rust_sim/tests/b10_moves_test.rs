//! The B10 pins (`gen3_b10_moves_v1`, `gen3_b10_endless_battle_v1`, gen3 full parity B10): Teleport,
//! Follow Me, Helping Hand, Role Play, Camouflage and the Endless Battle Clause. Each pin asserts the
//! sim's observable (the protocol lines and the post-script PRNG seed, recorded from Showdown by
//! `harness/probe_b10_moves.js`'s boards); the seed and byte differential over every board is
//! `b10_moves_golden_test.rs`.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{BattleOutcome, Choice, ScriptDecision, BERRY_CYCLE_ABILITIES, RESTORATIVE_BERRIES, ROLE_PLAY_FAIL_ABILITIES};

/// The post-construction seed Showdown reaches from `[17, 29, 43, 59]` on every board below without a
/// construction speed tie (the golden's `seedBefore` convention: the port starts AFTER construction).
const SEED: &str = "18944,54804,63962,18698";
/// The same for the speed-tied Clefable mirror (its construction tie draws).
const SEED_MIRROR: &str = "5086,37025,45627,32254";

fn set(species: &str, item: &str, ability: &str, moves: &str) -> String {
    format!("{species}|{species}|{item}|{ability}|{moves}|Hardy|85,85,85,85,85,85|M||||")
}

/// The post-`|turn|1` lines (separators, `|upkeep`, `|t:|` dropped), the outcome and the final seed.
fn play(format: &str, p1: &str, p2: &str, script: &[ScriptDecision]) -> (Vec<String>, BattleOutcome, String) {
    play_from(SEED, format, p1, p2, script)
}

fn play_from(seed: &str, format: &str, p1: &str, p2: &str, script: &[ScriptDecision]) -> (Vec<String>, BattleOutcome, String) {
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: format.into(),
        seed: Some(seed.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let st = b.state_mut().unwrap();
    let (o, lines) = st.run_full_battle_logged(script, &d);
    let all: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let t1 = all.iter().position(|l| l == "|turn|1").expect("|turn|1");
    let kept = all[t1 + 1..]
        .iter()
        .filter(|l| !(l.is_empty() || *l == "|" || *l == "|upkeep" || l.starts_with("|t:|")))
        .cloned()
        .collect();
    (kept, o, st.prng_seed().to_string())
}

fn mv(a: usize, b: usize) -> ScriptDecision {
    ScriptDecision::both(Choice::Move(a - 1), Choice::Move(b - 1))
}

fn node(js: &str) -> String {
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(js)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("node must be runnable (see root CLAUDE.md worktree setup)");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// B10-0 — the hardcoded id sets equal Showdown's: Role Play's fail list is every gen-3 ability whose
/// RESOLVED flags carry `failroleplay` (and none carries `cantsuppress`); the restorative berries are
/// `sim/pokemon.ts`'s exported `RESTORATIVE_BERRIES`; the cycling abilities are the literal pair in
/// `sim/battle.ts`'s loser test.
#[test]
fn b10_0_the_id_sets_are_showdowns() {
    let fail = node(
        "const {Dex}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.mod('gen3');\
         const a=d.abilities.all().filter(x=>x.exists&&x.gen<=3&&!x.isNonstandard);\
         if(a.some(x=>x.flags.cantsuppress))throw new Error('a gen-3 cantsuppress ability');\
         console.log(a.filter(x=>x.flags.failroleplay).map(x=>x.id).sort().join(','))",
    );
    assert_eq!(fail, ROLE_PLAY_FAIL_ABILITIES.join(","));
    let berries = node(
        "console.log([...require('../../deps/pokemon-showdown/dist/sim/pokemon').RESTORATIVE_BERRIES].sort().join(','))",
    );
    let mut ours = RESTORATIVE_BERRIES.to_vec();
    ours.sort_unstable();
    assert_eq!(berries, ours.join(","));
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../deps/pokemon-showdown/sim/battle.ts"))
        .expect("sim/battle.ts");
    let lit = format!("[{}].includes(toID(pokemon.set.ability))", BERRY_CYCLE_ABILITIES.map(|a| format!("'{a}'")).join(", "));
    assert!(src.contains(&lit), "sim/battle.ts no longer reads {lit}");
}

/// B10-1 — TELEPORT fails in a trainer battle with NO line and NO draw: the bare self announce, and
/// the seed equals a Splash turn's (`12055,…`).
#[test]
fn b10_1_teleport_is_a_bare_announce_and_draws_nothing() {
    let (lines, _, seed) = play(
        "gen3customgame",
        &set("Abra", "", "Synchronize", "teleport,splash"),
        &set("Snorlax", "", "ThickFat", "bodyslam,splash"),
        &[mv(1, 2)],
    );
    assert_eq!(lines, ["|move|p1a: Abra|Teleport|p1a: Abra", "|move|p2a: Snorlax|Splash|p2a: Snorlax", "|-nothing", "|turn|2"]);
    assert_eq!(seed, "12055,55990,39841,51637");
}

/// B10-2 — FOLLOW ME fails in singles: no `-singleturn` and no slot condition, so a speed-tied mirror
/// draws only its turn-order and Update ties (the sim's seed after two turns).
#[test]
fn b10_2_follow_me_never_starts_in_singles() {
    let (lines, _, seed) = play_from(
        SEED_MIRROR,
        "gen3customgame",
        &set("Clefable", "", "CuteCharm", "followme"),
        &set("Clefable", "", "CuteCharm", "followme"),
        &[mv(1, 1), mv(1, 1)],
    );
    assert_eq!(
        lines,
        [
            "|move|p1a: Clefable|Follow Me|p1a: Clefable",
            "|move|p2a: Clefable|Follow Me|p2a: Clefable",
            "|turn|2",
            "|move|p2a: Clefable|Follow Me|p2a: Clefable",
            "|move|p1a: Clefable|Follow Me|p1a: Clefable",
            "|turn|3",
        ]
    );
    assert_eq!(seed, "23687,23602,21739,39932");
}

/// B10-3 — HELPING HAND finds no ally: `|move|<u>|Helping Hand|null|[notarget]` + `|-notarget|<u>`,
/// and it is still the user's lastMove (a later Spite drains it).
#[test]
fn b10_3_helping_hand_has_no_target_and_is_the_last_move() {
    let (lines, _, seed) = play(
        "gen3customgame",
        &set("Plusle", "", "Plus", "helpinghand,thunderbolt"),
        &set("Jolteon", "", "VoltAbsorb", "spite,splash"),
        &[mv(1, 2), mv(2, 1)],
    );
    assert_eq!(
        lines,
        [
            "|move|p1a: Plusle|Helping Hand|null|[notarget]",
            "|-notarget|p1a: Plusle",
            "|move|p2a: Jolteon|Splash|p2a: Jolteon",
            "|-nothing",
            "|turn|2",
            "|move|p2a: Jolteon|Spite|p1a: Plusle",
            "|-activate|p1a: Plusle|move: Spite|helpinghand|3",
            "|move|p1a: Plusle|Thunderbolt|p2a: Jolteon",
            "|-immune|p2a: Jolteon|[from] ability: Volt Absorb",
            "|turn|3",
        ]
    );
    assert_eq!(seed, "20996,12330,49662,16889");
}

/// B10-4 — ROLE PLAY copies the target's ability with the setAbility reveal, and a copied Intimidate
/// does NOT fire (no gen-3 `Start`).
#[test]
fn b10_4_role_play_copies_and_starts_nothing() {
    let (lines, _, seed) = play(
        "gen3customgame",
        &set("Alakazam", "", "Synchronize", "roleplay,splash"),
        &set("Gyarados", "", "Intimidate", "splash"),
        &[mv(1, 1)],
    );
    assert_eq!(
        lines,
        [
            "|move|p1a: Alakazam|Role Play|p2a: Gyarados",
            "|-ability|p1a: Alakazam|Intimidate|Synchronize|[from] move: Role Play|[of] p2a: Gyarados",
            "|move|p2a: Gyarados|Splash|p2a: Gyarados",
            "|-nothing",
            "|turn|2",
        ]
    );
    assert_eq!(seed, "12055,55990,39841,51637");
}

/// B10-5 — ROLE PLAY into Wonder Guard (gen-3 `failroleplay`) fails: `[still]` + `-fail` on the user.
#[test]
fn b10_5_role_play_fails_into_wonder_guard() {
    let (lines, _, _) = play(
        "gen3customgame",
        &set("Alakazam", "", "Synchronize", "roleplay,splash"),
        &set("Shedinja", "", "WonderGuard", "splash"),
        &[mv(1, 1)],
    );
    assert_eq!(&lines[..2], ["|move|p1a: Alakazam|Role Play||[still]", "|-fail|p1a: Alakazam"]);
}

/// B10-6 — the user's OLD ability ends BEFORE the reveal: an armed Flash Fire's silent `-end` precedes
/// the `-ability` line.
#[test]
fn b10_6_role_play_ends_the_old_ability_first() {
    let (lines, _, seed) = play(
        "gen3customgame",
        &set("Houndoom", "", "FlashFire", "roleplay,flamethrower"),
        &set("Gyarados", "", "Intimidate", "splash,ember"),
        &[mv(2, 2), mv(1, 1)],
    );
    let i = lines.iter().position(|l| l == "|move|p1a: Houndoom|Role Play|p2a: Gyarados").expect("the Role Play");
    assert_eq!(
        &lines[i + 1..i + 3],
        [
            "|-end|p1a: Houndoom|ability: Flash Fire|[silent]",
            "|-ability|p1a: Houndoom|Intimidate|Flash Fire|[from] move: Role Play|[of] p2a: Gyarados",
        ]
    );
    assert_eq!(seed, "38721,49529,30239,49603");
}

/// B10-7 — a copied TRUANT loafs on the user's own `truantTurn`: the copier moves the copy turn, its
/// residual arms the flag, and it loafs the next turn.
#[test]
fn b10_7_a_copied_truant_loafs_the_next_turn() {
    let (lines, _, seed) = play(
        "gen3customgame",
        &set("Alakazam", "", "Synchronize", "roleplay,psychic"),
        &set("Slaking", "", "Truant", "splash"),
        &[mv(2, 1), mv(1, 1), mv(2, 1)],
    );
    let t3 = lines.iter().position(|l| l == "|turn|3").expect("turn 3");
    assert_eq!(lines[t3 + 1], "|cant|p1a: Alakazam|ability: Truant");
    assert_eq!(seed, "38721,49529,30239,49603");
}

/// B10-8 (a copied Own Tempo on a confused user, a copied Oblivious on an infatuated user) moved to
/// `tests/regression_test.rs` bf12c / bf13b once the BF Update cure landed.

/// B10-9 — CAMOUFLAGE retypes the user Normal (after a Color Change), draw-free.
#[test]
fn b10_9_camouflage_retypes_normal() {
    let (lines, _, seed) = play(
        "gen3customgame",
        &set("Kecleon", "", "ColorChange", "camouflage,splash"),
        &set("Jolteon", "", "VoltAbsorb", "thunderbolt"),
        &[mv(1, 1), mv(1, 1)],
    );
    let want = [
        "|-start|p1a: Kecleon|typechange|Electric|[from] ability: Color Change",
        "|move|p1a: Kecleon|Camouflage|p1a: Kecleon",
        "|-start|p1a: Kecleon|typechange|Normal",
    ];
    assert_eq!(lines.windows(3).filter(|w| *w == want).count(), 2, "{lines:#?}");
    assert_eq!(seed, "54484,17454,39273,1574");
}

/// B10-10 — CAMOUFLAGE fails on a Normal type, here a Normal SNATCHER: the `[from] Snatch` fold comes
/// before the `[still]` retro-edit.
#[test]
fn b10_10_a_snatched_camouflage_fails_on_a_normal_snatcher() {
    let (lines, _, _) = play(
        "gen3customgame",
        &set("Kecleon", "", "ColorChange", "camouflage"),
        &set("Snorlax", "", "ThickFat", "snatch"),
        &[mv(1, 1)],
    );
    assert_eq!(
        lines,
        [
            "|move|p2a: Snorlax|Snatch|p2a: Snorlax",
            "|-singleturn|p2a: Snorlax|Snatch",
            "|move|p1a: Kecleon|Camouflage|p1a: Kecleon",
            "|-activate|p2a: Snorlax|move: Snatch|[of] p1a: Kecleon",
            "|move|p2a: Snorlax|Camouflage||[from] Snatch|[still]",
            "|-fail|p2a: Snorlax",
            "|turn|2",
        ]
    );
}

// ---------------------------------------------------------------- ENDLESS BATTLE CLAUSE
// Long scripted boards (the sim reaches the trigger only past turn 100). `Idle` plays the side's
// Splash until its PP runs out, then Harden.

struct Idle {
    splash: usize,
    harden: usize,
    used: u32,
}
impl Idle {
    fn next(&mut self) -> Choice {
        self.used += 1;
        Choice::Move(if self.used <= 64 { self.splash } else { self.harden })
    }
}

/// Builds a 104-turn script from `first` (the opening turns, 1-based move slots or `None` = idle;
/// `Some(0)` = switch to slot 2) and the two idlers.
fn long_script(first: &[(Option<usize>, Option<usize>)], mut a: Idle, mut b: Idle, splash_used: (u32, u32)) -> Vec<ScriptDecision> {
    a.used = splash_used.0;
    b.used = splash_used.1;
    let pick = |c: Option<usize>, idle: &mut Idle| match c {
        Some(0) => Choice::Switch(1),
        Some(k) => Choice::Move(k - 1),
        None => idle.next(),
    };
    (0..104)
        .map(|t| {
            let (x, y) = first.get(t).copied().unwrap_or((None, None));
            ScriptDecision::both(pick(x, &mut a), pick(y, &mut b))
        })
        .collect()
}

const AZ: (&str, &str, &str, &str) = ("Alakazam", "Sitrus Berry", "Synchronize", "skillswap,recycle,splash,harden");
const SX: (&str, &str, &str, &str) = ("Snorlax", "Leftovers", "ThickFat", "seismictoss,splash,harden");
const BENCH: &str = "Blissey|Blissey||NaturalCure|splash,harden|Hardy|85,85,85,85,85,85|F||||";

fn s4(x: (&str, &str, &str, &str)) -> String {
    set(x.0, x.1, x.2, x.3)
}

/// Toss Alakazam twice (it eats its STARTING Sitrus: no staleness), Recycle (pending), eat again
/// (INTERNAL staleness), Skill Swap the Snorlax (EXTERNAL), idle. Splash counts: P1 used 2, P2 0.
fn plan_a() -> Vec<ScriptDecision> {
    long_script(
        &[(Some(3), Some(1)), (Some(3), Some(1)), (Some(2), None), (Some(1), None)],
        Idle { splash: 2, harden: 3, used: 0 },
        Idle { splash: 1, harden: 2, used: 0 },
        (2, 0),
    )
}

fn closing(lines: &[String]) -> Vec<String> {
    lines[lines.len().saturating_sub(3)..].to_vec()
}

/// B10-11 — the clause fires at the turn-101 endTurn: both actives stale (Alakazam by its Recycled
/// Sitrus, Snorlax by the Skill Swap), neither side can switch, and only P1's SET carries a berry +
/// Recycle: P1 loses before the `|turn|101` line.
#[test]
fn b10_11_endless_battle_clause_p1_cycled_and_loses() {
    let (lines, o, _) = play("gen3ou", &s4(AZ), &s4(SX), &plan_a());
    assert!(lines.iter().any(|l| l == "|turn|100") && !lines.iter().any(|l| l == "|turn|101"), "ends at the turn-101 endTurn");
    assert_eq!(
        closing(&lines),
        [
            "|-boost|p2a: Snorlax|def|0",
            "|-message|P1's team started with the rudimentary means to perform restorative berry-cycling and thus loses.",
            "|win|P2",
        ]
    );
    assert_eq!((o.ended, o.winner), (true, Some(1)));
}

/// B10-12 — a THIEF-stolen Sitrus stales its eater (the steal's `setItem` sets the pending staleness);
/// neither SET can cycle a restorative berry, so the clause ties with no message.
#[test]
fn b10_12_endless_battle_clause_thief_bare_tie() {
    let script = long_script(
        &[(Some(2), Some(1)), (Some(3), Some(1)), (Some(1), None)],
        Idle { splash: 2, harden: 3, used: 0 },
        Idle { splash: 1, harden: 2, used: 0 },
        (1, 0),
    );
    let (lines, o, _) = play(
        "gen3ou",
        &set("Alakazam", "", "Synchronize", "skillswap,thief,splash,harden"),
        &set("Snorlax", "Sitrus Berry", "ThickFat", "seismictoss,splash,harden"),
        &script,
    );
    assert_eq!(closing(&lines)[1..], ["|-boost|p2a: Snorlax|def|0", "|tie"]);
    assert!(!lines.iter().any(|l| l.starts_with("|-message|")), "no message on a bare tie");
    assert_eq!((o.ended, o.winner), (true, None));
}

/// B10-13 — TRICK hands the Sitrus over (the receiver's pending staleness); the receiver eats it
/// (internal) and Skill Swaps the Tricker (external). The loser test reads SETS: the Tricker's set
/// held the Sitrus and its set ability is PICKUP — P1 loses though it never cycled anything.
#[test]
fn b10_13_endless_battle_clause_trick_and_pickup() {
    let script = long_script(
        &[(Some(1), None), (Some(2), None), (Some(2), None), (None, Some(1))],
        Idle { splash: 2, harden: 3, used: 0 },
        Idle { splash: 1, harden: 2, used: 0 },
        (0, 0),
    );
    let (lines, o, _) = play(
        "gen3ou",
        &set("Linoone", "Sitrus Berry", "Pickup", "trick,seismictoss,splash,harden"),
        &set("Alakazam", "Leftovers", "Synchronize", "skillswap,splash,harden"),
        &script,
    );
    assert_eq!(
        closing(&lines)[1..],
        ["|-message|P1's team started with the rudimentary means to perform restorative berry-cycling and thus loses.", "|win|P2"]
    );
    assert_eq!((o.ended, o.winner), (true, Some(1)));
}

/// B10-14 — CONTROL: a switch-out clears the Skill Swap staleness, so the clause never fires.
#[test]
fn b10_14_a_switch_clears_the_skill_swap_staleness() {
    let mut first = vec![(Some(3), Some(1)), (Some(3), Some(1)), (Some(2), None), (Some(1), None)];
    first.extend([(None, Some(0)), (None, Some(0))]);
    let script = long_script(&first, Idle { splash: 2, harden: 3, used: 0 }, Idle { splash: 1, harden: 2, used: 0 }, (2, 0));
    let (lines, o, _) = play("gen3ou", &s4(AZ), &format!("{}]{BENCH}", s4(SX)), &script);
    assert!(lines.iter().any(|l| l == "|turn|104"), "the battle runs past turn 101");
    assert!(!lines.iter().any(|l| l.contains("berry-cycling")));
    assert!(!o.ended);
}

/// B10-15 — CONTROL: Custom Game has no Endless Battle Clause; the same board plays on.
#[test]
fn b10_15_custom_game_has_no_clause() {
    let (lines, o, _) = play("gen3customgame", &s4(AZ), &s4(SX), &plan_a());
    assert!(lines.iter().any(|l| l == "|turn|104"));
    assert!(!o.ended);
}
