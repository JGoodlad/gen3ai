//! The B11 MOVES pins (`gen3_b11_moves_v1`, gen3 full parity B11): Metronome, Mirror Move, Assist,
//! Nature Power, Sketch and the one calling path they share with Sleep Talk. Every expected value is the
//! REAL sim's (`harness/probe_b11_moves.js`'s settled spec; each case's lines and seeds were measured on
//! Showdown with the case's own teams, seed and choices, the post-`|turn|1` omniscient lines with the `|`
//! separators and `|upkeep` dropped). Each pin asserts the sim's observable (the lines, the seed, the
//! request), and each was REVERT-VERIFIED: the named fix was reverted and the pin watched to fail.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{BridgeSession, Cmd, WireChoice};
use pokesim::dex::Dex;
use pokesim::turn::{
    metronome_pool, Choice, ScriptDecision, METRONOME_EXCLUDED, MIRROR_MOVE_EXCLUDED, NOASSIST_MOVES, NOSKETCH_MOVES,
};

struct Case {
    format: &'static str,
    p1: &'static str,
    p2: &'static str,
    raw_seed: &'static str,
    seed_before: &'static str,
    seed_after: &'static str,
    lines: &'static [&'static str],
}

fn d(p1: Option<Choice>, p2: Option<Choice>) -> ScriptDecision {
    ScriptDecision { p1, p2 }
}

/// The port's post-`|turn|1` lines (separators, `|upkeep`, `|t:|` dropped) and its final seed.
fn play(c: &Case, script: &[ScriptDecision]) -> (Vec<String>, String) {
    let dex = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: c.format.into(),
        seed: Some(c.seed_before.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(c.p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(c.p2.into()) },
    };
    let mut b = Battle::start_with_switchins(&opts, &dex).expect("start");
    let st = b.state_mut().expect("state");
    let (_o, emitted) = st.run_full_battle_logged(script, &dex);
    let all: Vec<String> = pokesim::bridge::reframe(&emitted, c.format);
    let t1 = all.iter().position(|l| l == "|turn|1").expect("|turn|1");
    let kept = all[t1 + 1..]
        .iter()
        .filter(|l| !(l.is_empty() || *l == "|" || *l == "|upkeep" || l.starts_with("|t:|") || l.starts_with("|debug|")))
        .cloned()
        .collect();
    (kept, st.prng_seed())
}

fn check(c: &Case, script: &[ScriptDecision]) {
    let (lines, seed) = play(c, script);
    let want: Vec<String> = c.lines.iter().map(|s| s.to_string()).collect();
    assert_eq!(lines, want, "lines");
    assert_eq!(seed, c.seed_after, "seed after");
}

/// The `active` request JSON of `side` after feeding `script` through the RAW-seed bridge path.
fn request_after(c: &Case, script: &[(usize, usize)], side: usize) -> String {
    let dex = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: c.format.into(),
        seed: Some(c.raw_seed.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(c.p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(c.p2.into()) },
    };
    let mut s = BridgeSession::new_construct_turn0(&opts, &dex).expect("session");
    for &(a, b) in script {
        s.feed_cmd(Cmd { side: 0, choice: WireChoice::Move(a) }, &dex);
        s.feed_cmd(Cmd { side: 1, choice: WireChoice::Move(b) }, &dex);
    }
    s.active_request_json(side).expect("an open request").to_string()
}

/// Node's answer to one expression over the RESOLVED `Dex.mod('gen3')` (fails loudly without node).
fn node(expr: &str) -> String {
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(format!("const {{Dex}}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.mod('gen3');console.log({expr})"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("node must be runnable (see root CLAUDE.md worktree setup)");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// B11-1 — the hardcoded id sets are the resolved dex's: Metronome's pool IN ITS num ORDER (the
/// `sample` index → callee map), Assist's `noassist`, Mirror Move's gen-3 `noMirror` list (read off the
/// resolved handler's source) and Sketch's `nosketch`.
#[test]
fn b11_1_the_id_sets_are_the_resolved_dex() {
    let dex = Dex::for_gen(3);
    let sim_pool = node(
        "d.moves.all().filter(m=>(!m.isNonstandard||m.isNonstandard==='Unobtainable')&&m.flags.metronome)\
         .sort((a,b)=>a.num-b.num).map(m=>m.id).join(',')",
    );
    assert_eq!(metronome_pool(&dex).join(","), sim_pool, "Metronome's num-sorted pool");
    let mut ours: Vec<&str> = METRONOME_EXCLUDED.to_vec();
    ours.sort_unstable();
    let sim_ex = node(
        "[...new Set(d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&!m.flags.metronome\
         &&!m.id.startsWith('hiddenpower')).map(m=>m.id))].sort().join(',')",
    );
    assert_eq!(ours.join(","), sim_ex, "METRONOME_EXCLUDED");
    let mut ours: Vec<&str> = NOASSIST_MOVES.to_vec();
    ours.sort_unstable();
    let sim = node("d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&m.flags.noassist).map(m=>m.id).sort().join(',')");
    assert_eq!(ours.join(","), sim, "NOASSIST_MOVES");
    let sim = node(
        "JSON.parse(d.moves.get('mirrormove').onHit.toString().match(/const noMirror = (\\[[^\\]]*\\])/)[1].replace(/'/g,'\"'))\
         .sort().join(',')",
    );
    let mut ours: Vec<&str> = MIRROR_MOVE_EXCLUDED.to_vec();
    ours.sort_unstable();
    assert_eq!(ours.join(","), sim, "MIRROR_MOVE_EXCLUDED");
    let sim = node("d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&m.flags.nosketch).map(m=>m.id).sort().join(',')");
    let mut ours: Vec<&str> = NOSKETCH_MOVES.to_vec();
    ours.sort_unstable();
    assert_eq!(ours.join(","), sim, "NOSKETCH_MOVES");
}

/// B11-2 — Metronome: the announce, ONE `random(336)` after it, the callee with `[from] Metronome`.
#[test]
fn b11_2_metronome_samples_once_then_runs_the_callee() {
    check(&MET_TBOLT, &met_tbolt_script());
}

/// B11-3 — Metronome's pool reaches the B10 moves (Teleport here: it fails in a trainer battle) through
/// the same call path.
#[test]
fn b11_3_metronome_calls_a_b10_move() {
    check(&MET_TELEPORT, &met_teleport_script());
}

/// B11-4 — a lock a CALLED move starts names the callee in the request even though the user does not
/// carry it: `{"moves":[{"move":"Outrage","id":"outrage"}],"trapped":true}` (the CALLED-MOVE slot).
#[test]
fn b11_4_a_called_lock_requests_the_callee() {
    let req = request_after(&MET_OUTRAGE_REQ, &[(0, 0)], 0);
    assert!(req.contains("\"active\":[{\"moves\":[{\"move\":\"Outrage\",\"id\":\"outrage\"}],\"trapped\":true}]"), "{req}");
    check(&MET_OUTRAGE_REQ, &met_outrage_req_script());
}

/// B11-5 — after a called lock's repeat `lastMove` is the callee, in no moveslot: Disable passes its
/// `onTryHit`, DRAWS its duration and fails at `onStart` (`|move|…|Disable||[still]` + `-fail`).
#[test]
fn b11_5_disable_after_a_called_repeat_draws_then_fails() {
    check(&MET_DISABLE, &met_disable_script());
}

/// B11-6 — Encore lands on the caller (the `lastMove`) and overrides the locked repeat: Metronome runs
/// with NO PP and `|[from] lockedmove`.
#[test]
fn b11_6_encore_overrides_a_called_lock() {
    check(&MET_ENCORE, &met_encore_script());
}

/// B11-7 — a called Bide's wait turn under that override prints `-activate|…|move: Bide`, then RUNS the
/// overriding Metronome (only Bide's own `beforeMoveCallback` ends `runMove`); the release then fails.
#[test]
fn b11_7_a_called_bides_wait_runs_the_overriding_move() {
    check(&MET_BIDE_ENCORE, &met_bide_encore_script());
}

/// B11-8 — a Choice Band Metronome user stays locked to METRONOME through the Outrage it called (the
/// repeat sets `lastMove = outrage`, but `addVolatile('choicelock')` is a no-op once held).
#[test]
fn b11_8_a_choice_lock_stays_on_the_caller() {
    check(&MET_CB, &met_cb_script());
}

/// B11-9 — Mirror Move fails on an attacker that switched out (`clearVolatile` nulled its `lastMove`).
#[test]
fn b11_9_mirror_move_fails_on_a_switched_source() {
    check(&MM_SWITCHED, &mm_switched_script());
}

/// B11-10 — Mirror Move fails on a Metronome callee the source does not hold (`hasMove`).
#[test]
fn b11_10_mirror_move_fails_on_a_callee_the_source_lacks() {
    check(&MM_METRONOME_CALLEE, &mm_metronome_callee_script());
}

/// B11-11 — Counter is NOT on gen 3's `noMirror` list: a mirrored Counter announces and fails silently
/// (no `counter` volatile — that is added only for a queued Counter).
#[test]
fn b11_11_a_mirrored_counter_announces_and_fails() {
    check(&MM_COUNTER, &mm_counter_script());
}

/// B11-12 — a MISSED move pushes no `attackedBy` entry: Mirror Move copies the earlier Tackle.
#[test]
fn b11_12_a_missed_move_leaves_the_earlier_entry() {
    check(&MM_MISS, &mm_miss_script());
}

/// B11-13 — Assist's pool keeps a FAINTED party member's moves.
#[test]
fn b11_13_assist_draws_a_fainted_mates_move() {
    check(&AS_FAINTED, &as_fainted_script());
}

/// B11-14 — Assist with only `noassist` moves in the party fails draw-free.
#[test]
fn b11_14_assist_fails_on_an_empty_pool() {
    check(&AS_EMPTY, &as_empty_script());
}

/// B11-15 — Nature Power's own 95% accuracy roll runs against its +6-evasion user (`[miss]` + `-miss|<u>|<u>`).
#[test]
fn b11_15_nature_power_misses_its_evasive_user() {
    check(&NP_MISS, &np_miss_script());
}

/// B11-16 — Sketch copies permanently (it survives a switch), with base PP and no `target` key.
#[test]
fn b11_16_sketch_copies_permanently() {
    check(&SK_SWITCH, &sk_switch_script());
    let req = request_after(&SK_REQ, &[(1, 0), (0, 1)], 0);
    assert!(req.contains("{\"move\":\"Tackle\",\"id\":\"tackle\",\"pp\":35,\"maxpp\":35,\"disabled\":false}"), "{req}");
}

/// B11-17 — THE SHARED PATH sets `lastMoveUsed` to the callee (`useMoveInner`): Conversion 2 after a
/// Sleep-Talk Earthquake picks a type that resists GROUND (Bug), not one resisting Sleep Talk's Normal.
/// A Sleep Talk behaviour change (the port kept `sleeptalk` before B11).
#[test]
fn b11_17_conversion2_reads_a_sleep_talk_callee() {
    check(&ST_CONV2, &st_conv2_script());
}

// ---------------------------------------------------------------- the measured cases

const MM_SWITCHED: Case = Case {
    format: "gen3customgame",
    p1: "Pidgeot|Pidgeot||KeenEye|mirrormove,splash|Hardy|85,85,85,85,85,85|M||||",
    p2: "Snorlax|Snorlax||ThickFat|tackle,splash|Hardy|85,85,85,85,85,85|M||||]Blissey|Blissey||NaturalCure|splash,softboiled|Hardy|85,85,85,85,85,85|F||||",
    raw_seed: "17,29,43,59",
    seed_before: "18944,54804,63962,18698",
    seed_after: "20996,12330,49662,16889",
    lines: &[
        "|move|p1a: Pidgeot|Splash|p1a: Pidgeot",
        "|-nothing",
        "|move|p2a: Snorlax|Tackle|p1a: Pidgeot",
        "|-damage|p1a: Pidgeot|272/328",
        "|turn|2",
        "|switch|p2a: Blissey|Blissey, F|672/672",
        "|move|p1a: Pidgeot|Mirror Move||[still]",
        "|-fail|p1a: Pidgeot",
        "|turn|3",
    ],
};
fn mm_switched_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Switch(1)))]
}

const MM_METRONOME_CALLEE: Case = Case {
    format: "gen3customgame",
    p1: "Pidgeot|Pidgeot||KeenEye|mirrormove,splash|Hardy|85,85,85,85,85,85|M||||",
    p2: "Clefable|Clefable||CuteCharm|metronome,splash|Hardy|85,85,85,85,85,85|F||||",
    raw_seed: "283,515,1105,1909",
    seed_before: "64479,11781,41295,11500",
    seed_after: "57558,46719,60685,41917",
    lines: &[
        "|move|p1a: Pidgeot|Splash|p1a: Pidgeot",
        "|-nothing",
        "|move|p2a: Clefable|Metronome|p2a: Clefable",
        "|move|p2a: Clefable|Thunderbolt|p1a: Pidgeot|[from] Metronome",
        "|-supereffective|p1a: Pidgeot",
        "|-damage|p1a: Pidgeot|159/328",
        "|turn|2",
        "|move|p1a: Pidgeot|Mirror Move||[still]",
        "|-fail|p1a: Pidgeot",
        "|move|p2a: Clefable|Splash|p2a: Clefable",
        "|-nothing",
        "|turn|3",
    ],
};
fn mm_metronome_callee_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(1)))]
}

const MM_COUNTER: Case = Case {
    format: "gen3customgame",
    p1: "Pidgeot|Pidgeot||KeenEye|mirrormove,tackle|Hardy|85,85,85,85,85,85|M||||",
    p2: "Machamp|Machamp||Guts|counter,splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "17,29,43,59",
    seed_before: "18944,54804,63962,18698",
    seed_after: "44586,44725,29843,59136",
    lines: &[
        "|move|p1a: Pidgeot|Tackle|p2a: Machamp",
        "|-damage|p2a: Machamp|300/342",
        "|move|p2a: Machamp|Counter|p1a: Pidgeot",
        "|-damage|p1a: Pidgeot|244/328",
        "|turn|2",
        "|move|p1a: Pidgeot|Mirror Move|p1a: Pidgeot",
        "|move|p1a: Pidgeot|Counter|p2a: Machamp|[from] Mirror Move",
        "|move|p2a: Machamp|Splash|p2a: Machamp",
        "|-nothing",
        "|turn|3",
    ],
};
fn mm_counter_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(1)))]
}

const MM_MISS: Case = Case {
    format: "gen3customgame",
    p1: "Pidgeot|Pidgeot||KeenEye|mirrormove,splash|Hardy|85,85,85,85,85,85|M||||",
    p2: "Jolteon|Jolteon||VoltAbsorb|tackle,hypnosis|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "108,190,380,634",
    seed_before: "15369,36179,31932,58085",
    seed_after: "5508,25963,32410,27286",
    lines: &[
        "|move|p2a: Jolteon|Tackle|p1a: Pidgeot",
        "|-damage|p1a: Pidgeot|302/328",
        "|move|p1a: Pidgeot|Splash|p1a: Pidgeot",
        "|-nothing",
        "|turn|2",
        "|move|p2a: Jolteon|Hypnosis|p1a: Pidgeot|[miss]",
        "|-miss|p2a: Jolteon|p1a: Pidgeot",
        "|move|p1a: Pidgeot|Splash|p1a: Pidgeot",
        "|-nothing",
        "|turn|3",
        "|move|p2a: Jolteon|Hypnosis|p1a: Pidgeot|[miss]",
        "|-miss|p2a: Jolteon|p1a: Pidgeot",
        "|move|p1a: Pidgeot|Mirror Move|p1a: Pidgeot",
        "|move|p1a: Pidgeot|Tackle|p2a: Jolteon|[from] Mirror Move",
        "|-damage|p2a: Jolteon|243/292",
        "|turn|4",
    ],
};
fn mm_miss_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(1)), Some(Choice::Move(1))), d(Some(Choice::Move(0)), Some(Choice::Move(1)))]
}

const AS_FAINTED: Case = Case {
    format: "gen3customgame",
    p1: "Electrode|Electrode||Soundproof|explosion|Hardy|85,85,85,85,85,85|M||||]Persian|Persian||Limber|assist,splash|Hardy|85,85,85,85,85,85|M||||",
    p2: "Snorlax|Snorlax||ThickFat|splash|Hardy|85,85,85,85,85,85|M||||]Blissey|Blissey||NaturalCure|splash,softboiled|Hardy|85,85,85,85,85,85|F||||",
    raw_seed: "17,29,43,59",
    seed_before: "18944,54804,63962,18698",
    seed_after: "45159,18606,10676,27570",
    lines: &[
        "|move|p1a: Electrode|Explosion|p2a: Snorlax",
        "|-damage|p2a: Snorlax|151/482",
        "|faint|p1a: Electrode",
        "|switch|p1a: Persian|Persian, M|292/292",
        "|turn|2",
        "|move|p1a: Persian|Assist|p1a: Persian",
        "|move|p1a: Persian|Explosion|p2a: Snorlax|[from] Assist",
        "|-damage|p2a: Snorlax|0 fnt",
        "|faint|p1a: Persian",
        "|faint|p2a: Snorlax",
        "|win|P2",
    ],
};
fn as_fainted_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(0)), Some(Choice::Move(0))), d(Some(Choice::Switch(1)), None), d(Some(Choice::Move(0)), Some(Choice::Move(0)))]
}

const AS_EMPTY: Case = Case {
    format: "gen3customgame",
    p1: "Persian|Persian||Limber|assist,splash|Hardy|85,85,85,85,85,85|M||||]Jolteon|Jolteon||VoltAbsorb|protect,metronome,sleeptalk,mimic|Hardy|85,85,85,85,85,85|M||||",
    p2: "Snorlax|Snorlax||ThickFat|splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "17,29,43,59",
    seed_before: "18944,54804,63962,18698",
    seed_after: "12055,55990,39841,51637",
    lines: &[
        "|move|p1a: Persian|Assist||[still]",
        "|-fail|p1a: Persian",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|2",
    ],
};
fn as_empty_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(0)), Some(Choice::Move(0)))]
}

const NP_MISS: Case = Case {
    format: "gen3customgame",
    p1: "Ludicolo|Ludicolo||SwiftSwim|naturepower,doubleteam|Hardy|85,85,85,85,85,85|M||||",
    p2: "Snorlax|Snorlax||ThickFat|splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "17,29,43,59",
    seed_before: "18944,54804,63962,18698",
    seed_after: "6367,58923,25654,30952",
    lines: &[
        "|move|p1a: Ludicolo|Double Team|p1a: Ludicolo",
        "|-boost|p1a: Ludicolo|evasion|1",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|2",
        "|move|p1a: Ludicolo|Double Team|p1a: Ludicolo",
        "|-boost|p1a: Ludicolo|evasion|1",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|3",
        "|move|p1a: Ludicolo|Double Team|p1a: Ludicolo",
        "|-boost|p1a: Ludicolo|evasion|1",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|4",
        "|move|p1a: Ludicolo|Double Team|p1a: Ludicolo",
        "|-boost|p1a: Ludicolo|evasion|1",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|5",
        "|move|p1a: Ludicolo|Double Team|p1a: Ludicolo",
        "|-boost|p1a: Ludicolo|evasion|1",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|6",
        "|move|p1a: Ludicolo|Double Team|p1a: Ludicolo",
        "|-boost|p1a: Ludicolo|evasion|1",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|7",
        "|move|p1a: Ludicolo|Nature Power|p1a: Ludicolo|[miss]",
        "|-miss|p1a: Ludicolo|p1a: Ludicolo",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|8",
        "|move|p1a: Ludicolo|Nature Power|p1a: Ludicolo|[miss]",
        "|-miss|p1a: Ludicolo|p1a: Ludicolo",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|9",
        "|move|p1a: Ludicolo|Nature Power|p1a: Ludicolo",
        "|move|p1a: Ludicolo|Swift|p2a: Snorlax|[from] Nature Power",
        "|-damage|p2a: Snorlax|432/482",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|10",
    ],
};
fn np_miss_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(0)))]
}

const SK_SWITCH: Case = Case {
    format: "gen3customgame",
    p1: "Smeargle|Smeargle||OwnTempo|sketch,splash|Hardy|0,0,0,0,0,252|M||||]Blissey|Blissey||NaturalCure|splash,softboiled|Hardy|85,85,85,85,85,85|F||||",
    p2: "Snorlax|Snorlax||ThickFat|tackle,splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "17,29,43,59",
    seed_before: "18944,54804,63962,18698",
    seed_after: "20579,45021,5279,34087",
    lines: &[
        "|move|p1a: Smeargle|Splash|p1a: Smeargle",
        "|-nothing",
        "|move|p2a: Snorlax|Tackle|p1a: Smeargle",
        "|-damage|p1a: Smeargle|143/251",
        "|turn|2",
        "|move|p1a: Smeargle|Sketch|p2a: Snorlax",
        "|-activate|p1a: Smeargle|move: Mimic|Tackle",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|3",
        "|switch|p1a: Blissey|Blissey, F|672/672",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|4",
        "|switch|p1a: Smeargle|Smeargle, M|143/251",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|5",
        "|move|p1a: Smeargle|Tackle|p2a: Snorlax",
        "|-damage|p2a: Snorlax|465/482",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|6",
    ],
};
fn sk_switch_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(1))), d(Some(Choice::Switch(1)), Some(Choice::Move(1))), d(Some(Choice::Switch(1)), Some(Choice::Move(1))), d(Some(Choice::Move(0)), Some(Choice::Move(1)))]
}

const ST_CONV2: Case = Case {
    format: "gen3customgame",
    p1: "Snorlax|Snorlax||ThickFat|sleeptalk,earthquake|Hardy|85,85,85,85,85,85|M||||",
    p2: "Porygon2|Porygon2||Trace|spore,conversion2|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "17,29,43,59",
    seed_before: "12055,55990,39841,51637",
    seed_after: "20579,45021,5279,34087",
    lines: &[
        "|move|p2a: Porygon2|Spore|p1a: Snorlax",
        "|-status|p1a: Snorlax|slp|[from] move: Spore",
        "|cant|p1a: Snorlax|slp",
        "|turn|2",
        "|move|p2a: Porygon2|Conversion 2||[still]",
        "|-fail|p2a: Porygon2",
        "|cant|p1a: Snorlax|slp",
        "|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
        "|move|p1a: Snorlax|Earthquake|p2a: Porygon2|[from] Sleep Talk",
        "|-damage|p2a: Porygon2|243/332",
        "|turn|3",
        "|move|p2a: Porygon2|Conversion 2|p1a: Snorlax",
        "|-start|p2a: Porygon2|typechange|Bug",
        "|-curestatus|p1a: Snorlax|slp|[msg]",
        "|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
        "|turn|4",
    ],
};
fn st_conv2_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(1))), d(Some(Choice::Move(0)), Some(Choice::Move(1)))]
}

const MET_TELEPORT: Case = Case {
    format: "gen3customgame",
    p1: "Clefable|Clefable||CuteCharm|metronome,splash|Hardy|85,85,85,85,85,85|F||||",
    p2: "Snorlax|Snorlax||ThickFat|splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "388,710,1540,2674",
    seed_before: "28408,62678,60019,49085",
    seed_after: "61142,14658,21746,8423",
    lines: &[
        "|move|p1a: Clefable|Metronome|p1a: Clefable",
        "|move|p1a: Clefable|Teleport|p1a: Clefable|[from] Metronome",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|2",
    ],
};
fn met_teleport_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(0)), Some(Choice::Move(0)))]
}

const MET_TBOLT: Case = Case {
    format: "gen3customgame",
    p1: "Clefable|Clefable||CuteCharm|metronome,splash|Hardy|85,85,85,85,85,85|F||||",
    p2: "Snorlax|Snorlax||ThickFat|splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "283,515,1105,1909",
    seed_before: "64479,11781,41295,11500",
    seed_after: "61167,44618,62346,7794",
    lines: &[
        "|move|p1a: Clefable|Metronome|p1a: Clefable",
        "|move|p1a: Clefable|Thunderbolt|p2a: Snorlax|[from] Metronome",
        "|-damage|p2a: Snorlax|422/482",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|2",
    ],
};
fn met_tbolt_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(0)), Some(Choice::Move(0)))]
}

const MET_DISABLE: Case = Case {
    format: "gen3customgame",
    p1: "Clefable|Clefable||CuteCharm|metronome,splash|Hardy|85,85,85,85,85,85|F||||",
    p2: "Smeargle|Smeargle||OwnTempo|disable,splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "248,450,960,1654",
    seed_before: "54657,16661,13208,20817",
    seed_after: "19359,24464,54331,44282",
    lines: &[
        "|move|p2a: Smeargle|Splash|p2a: Smeargle",
        "|-nothing",
        "|move|p1a: Clefable|Metronome|p1a: Clefable",
        "|move|p1a: Clefable|Outrage|p2a: Smeargle|[from] Metronome",
        "|-damage|p2a: Smeargle|159/272",
        "|turn|2",
        "|move|p2a: Smeargle|Splash|p2a: Smeargle",
        "|-nothing",
        "|move|p1a: Clefable|Outrage|p2a: Smeargle|[from] lockedmove",
        "|-damage|p2a: Smeargle|43/272",
        "|-start|p1a: Clefable|confusion|[fatigue]",
        "|turn|3",
        "|move|p2a: Smeargle|Disable||[still]",
        "|-fail|p2a: Smeargle",
        "|-activate|p1a: Clefable|confusion",
        "|move|p1a: Clefable|Metronome|p1a: Clefable",
        "|move|p1a: Clefable|Thunder Shock|p2a: Smeargle|[from] Metronome",
        "|-damage|p2a: Smeargle|0 fnt",
        "|faint|p2a: Smeargle",
        "|win|P1",
    ],
};
fn met_disable_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(0)), Some(Choice::Move(1))), d(Some(Choice::Move(0)), Some(Choice::Move(1))), d(Some(Choice::Move(0)), Some(Choice::Move(0)))]
}

const MET_ENCORE: Case = Case {
    format: "gen3customgame",
    p1: "Clefable|Clefable||CuteCharm|metronome,splash|Hardy|85,85,85,85,85,85|F||||",
    p2: "Smeargle|Smeargle||OwnTempo|encore,splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "248,450,960,1654",
    seed_before: "54657,16661,13208,20817",
    seed_after: "33304,19820,27685,22557",
    lines: &[
        "|move|p2a: Smeargle|Splash|p2a: Smeargle",
        "|-nothing",
        "|move|p1a: Clefable|Metronome|p1a: Clefable",
        "|move|p1a: Clefable|Outrage|p2a: Smeargle|[from] Metronome",
        "|-damage|p2a: Smeargle|159/272",
        "|turn|2",
        "|move|p2a: Smeargle|Encore|p1a: Clefable",
        "|-start|p1a: Clefable|Encore",
        "|move|p1a: Clefable|Metronome|p1a: Clefable|[from] lockedmove",
        "|move|p1a: Clefable|Sand Attack|p2a: Smeargle|[from] Metronome",
        "|-unboost|p2a: Smeargle|accuracy|1",
        "|-start|p1a: Clefable|confusion|[fatigue]",
        "|turn|3",
    ],
};
fn met_encore_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(0)), Some(Choice::Move(1))), d(Some(Choice::Move(0)), Some(Choice::Move(0)))]
}

const MET_BIDE_ENCORE: Case = Case {
    format: "gen3customgame",
    p1: "Clefable|Clefable||CuteCharm|metronome,splash|Hardy|85,85,85,85,85,85|F||||",
    p2: "Smeargle|Smeargle||OwnTempo|encore,splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "1151,2127,4701,8233",
    seed_before: "6597,21840,16953,16368",
    seed_after: "16526,18546,59716,46709",
    lines: &[
        "|move|p2a: Smeargle|Splash|p2a: Smeargle",
        "|-nothing",
        "|move|p1a: Clefable|Metronome|p1a: Clefable",
        "|move|p1a: Clefable|Bide|p1a: Clefable|[from] Metronome",
        "|-start|p1a: Clefable|move: Bide",
        "|turn|2",
        "|move|p2a: Smeargle|Encore|p1a: Clefable",
        "|-start|p1a: Clefable|Encore",
        "|-activate|p1a: Clefable|move: Bide",
        "|move|p1a: Clefable|Metronome|p1a: Clefable|[from] lockedmove",
        "|move|p1a: Clefable|Jump Kick|p2a: Smeargle|[from] Metronome",
        "|-supereffective|p2a: Smeargle",
        "|-damage|p2a: Smeargle|101/272",
        "|turn|3",
        "|move|p2a: Smeargle|Splash|p2a: Smeargle",
        "|-nothing",
        "|-end|p1a: Clefable|move: Bide",
        "|-fail|p1a: Clefable",
        "|-end|p1a: Clefable|move: Bide|[silent]",
        "|turn|4",
    ],
};
fn met_bide_encore_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(0)), Some(Choice::Move(1))), d(Some(Choice::Move(0)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(1)))]
}

const MET_OUTRAGE_REQ: Case = Case {
    format: "gen3customgame",
    p1: "Clefable|Clefable||CuteCharm|metronome,splash|Hardy|85,85,85,85,85,85|F||||",
    p2: "Snorlax|Snorlax||ThickFat|splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "248,450,960,1654",
    seed_before: "54657,16661,13208,20817",
    seed_after: "6885,28748,38843,13631",
    lines: &[
        "|move|p1a: Clefable|Metronome|p1a: Clefable",
        "|move|p1a: Clefable|Outrage|p2a: Snorlax|[from] Metronome",
        "|-damage|p2a: Snorlax|422/482",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|2",
    ],
};
fn met_outrage_req_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(0)), Some(Choice::Move(0)))]
}

const SK_REQ: Case = Case {
    format: "gen3customgame",
    p1: "Smeargle|Smeargle||OwnTempo|sketch,splash|Hardy|0,0,0,0,0,252|M||||",
    p2: "Snorlax|Snorlax||ThickFat|tackle,splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "17,29,43,59",
    seed_before: "18944,54804,63962,18698",
    seed_after: "20996,12330,49662,16889",
    lines: &[
        "|move|p1a: Smeargle|Splash|p1a: Smeargle",
        "|-nothing",
        "|move|p2a: Snorlax|Tackle|p1a: Smeargle",
        "|-damage|p1a: Smeargle|143/251",
        "|turn|2",
        "|move|p1a: Smeargle|Sketch|p2a: Snorlax",
        "|-activate|p1a: Smeargle|move: Mimic|Tackle",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|3",
    ],
};
fn sk_req_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(1)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(1)))]
}

const MET_CB: Case = Case {
    format: "gen3customgame",
    p1: "Clefable|Clefable|Choice Band|CuteCharm|metronome,splash|Hardy|85,85,85,85,85,85|F||||",
    p2: "Snorlax|Snorlax||ThickFat|splash|Hardy|85,85,85,85,85,85|M||||",
    raw_seed: "248,450,960,1654",
    seed_before: "54657,16661,13208,20817",
    seed_after: "16619,11508,56857,42853",
    lines: &[
        "|move|p1a: Clefable|Metronome|p1a: Clefable",
        "|move|p1a: Clefable|Outrage|p2a: Snorlax|[from] Metronome",
        "|-damage|p2a: Snorlax|422/482",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|2",
        "|move|p1a: Clefable|Outrage|p2a: Snorlax|[from] lockedmove",
        "|-damage|p2a: Snorlax|360/482",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|-start|p1a: Clefable|confusion|[fatigue]",
        "|turn|3",
        "|-activate|p1a: Clefable|confusion",
        "|-damage|p1a: Clefable|310/352|[from] confusion",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|4",
        "|-end|p1a: Clefable|confusion",
        "|move|p1a: Clefable|Metronome|p1a: Clefable",
        "|move|p1a: Clefable|Thunder Shock|p2a: Snorlax|[from] Metronome",
        "|-damage|p2a: Snorlax|332/482",
        "|move|p2a: Snorlax|Splash|p2a: Snorlax",
        "|-nothing",
        "|turn|5",
    ],
};
fn met_cb_script() -> Vec<ScriptDecision> {
    vec![d(Some(Choice::Move(0)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(0))), d(Some(Choice::Move(0)), Some(Choice::Move(0)))]
}
