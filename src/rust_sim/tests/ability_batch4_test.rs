//! BATCH-4 class-sweep tests (`gen3_ability_batch4_v1`) — the per-seed PER-DECISION
//! STATE+HP+STATUS+TRAPPED+SEED+winner differential proving the final mechanics tail matches
//! Showdown EXACTLY, to GAME-END:
//!
//!   TRUANT — the loaf cant (onBeforeMove priority 9, DRAW-FREE — a loaf turn draws NOTHING,
//!     no para roll either), the switch-in arming + order-27 residual toggle parity, and the
//!     speed-tied MIRROR's one extra residual tie-shuffle draw (the seed timeline pins all).
//!   INNER FOCUS — the flinch secondary's random(100) STILL DRAWS (block at the APPLY): the
//!     seed timeline must match the sim's WITH the roll; the holder always moves (first-mover
//!     + status rows), while the Thick-Fat CONTROL pair flinches on the same seeds.
//!   SHADOW TAG — the per-side TRAPPED columns (unconditional — a Flying Skarmory is trapped;
//!     a MIRROR is mutually trapped) + the 0-draw seed parity.
//!   CUTE CHARM + ATTRACT — the unconditional 1/3 DamagingHit roll (the F-into-F control draws
//!     it too — same seeds, no attract), the attract onBeforeMove 1/2 (priority 2: after
//!     confusion, before para), the source-leaves clear.
//!   COLOR CHANGE — the type override read through the chart (an EQ into an Electric-overridden
//!     Kecleon is super-effective — the HP rows), the status type-immunity (Toxic fails on a
//!     Poison-overridden Kecleon — the status rows), draw-free (seed parity).
//!   KING'S ROCK — the appended trailing 10% flinch secondary (one extra random(100) — the
//!     seed timeline), [own secondary][KR] order (Muddy Water), Serene Grace ×2, the
//!     fixed-damage proc (Seismic Toss), the no-item control.
//!   FOCUS BAND — the onDamage 1/10 roll on EVERY Damage event into the holder (move hits,
//!     burn chips, Spikes — the seed timeline) + the survive-at-1 on a lethal MOVE hit (HP rows).
//!
//! The golden (`harness/gen_ability_batch4_golden.js`) drives the OMNISCIENT BattleStream to
//! game-end; this test replays each (scenario, seed) from the sim's init seed WITHOUT
//! re-seeding and asserts, per decision boundary: both actives' species/hp/maxhp/fainted/status
//! + pokemon_left + per-side TRAPPED + the request kind + the first mover + the post-decision
//! PRNG seed + the final winner.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Status;
use pokesim::turn::{Choice, RequestKind, ScriptDecision};
use std::collections::BTreeMap;

fn dex() -> Dex {
    Dex::for_gen(3)
}

#[derive(Debug, Clone, Default)]
struct ScenMeta {
    teams: [Option<String>; 2],
}

#[derive(Debug, Clone)]
struct RunCase {
    scen: String,
    init_seed: String,
    decisions: Vec<DecExpect>,
    ended: bool,
    winner: WinTok,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WinTok {
    P1,
    P2,
    Tie,
    None,
}

#[derive(Debug, Clone)]
struct DecExpect {
    request: ReqTok,
    force: [bool; 2],
    choice: [Option<Choice>; 2],
    seed_after: String,
    p1: SideExpect,
    p2: SideExpect,
    first_mover: String,
    covered: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReqTok {
    Move,
    Switch,
}

#[derive(Debug, Clone)]
struct SideExpect {
    species: String,
    hp: u16,
    maxhp: u16,
    fainted: bool,
    status: Option<Status>,
    left: usize,
    trapped: bool,
}

fn parse_choice(tok: &str) -> Option<Choice> {
    if tok == "-" {
        return None;
    }
    let (kind, num) = tok.split_at(1);
    let n: usize = num.parse().unwrap_or_else(|e| panic!("bad choice token {tok:?}: {e}"));
    match kind {
        "m" => Some(Choice::Move(n)),
        "s" => Some(Choice::Switch(n)),
        other => panic!("bad choice kind {other:?} in {tok:?}"),
    }
}

fn parse_status(tok: &str) -> Option<Status> {
    match tok {
        "-" => None,
        "fnt" => None,
        "brn" => Some(Status::Burn),
        "par" => Some(Status::Paralysis),
        "slp" => Some(Status::Sleep(0)),
        "frz" => Some(Status::Freeze),
        "psn" => Some(Status::Poison),
        "tox" => Some(Status::Toxic(0)),
        other => panic!("unknown status token {other:?}"),
    }
}

fn status_variant_eq(a: Option<Status>, b: Option<Status>) -> bool {
    use Status::*;
    matches!(
        (a, b),
        (None, None)
            | (Some(Burn), Some(Burn))
            | (Some(Paralysis), Some(Paralysis))
            | (Some(Sleep(_)), Some(Sleep(_)))
            | (Some(Freeze), Some(Freeze))
            | (Some(Poison), Some(Poison))
            | (Some(Toxic(_)), Some(Toxic(_)))
    )
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/ability_batch4_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing batch4 golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_ability_batch4_golden.js")
    });
    let mut meta: BTreeMap<String, ScenMeta> = BTreeMap::new();
    let mut cases: Vec<RunCase> = Vec::new();
    let mut cur: Option<RunCase> = None;
    let mut last_scen = String::new();
    let flush = |cur: &mut Option<RunCase>, cases: &mut Vec<RunCase>| {
        if let Some(c) = cur.take() {
            cases.push(c);
        }
    };
    for (i, line) in data.lines().enumerate() {
        let ln = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "SCEN" => {
                last_scen = f[1].to_string();
                meta.entry(last_scen.clone()).or_default();
            }
            "TEAM" => {
                assert_eq!(f.len(), 3, "TEAM needs 3 fields (line {ln})");
                let side = if f[1] == "p1" { 0 } else { 1 };
                meta.entry(last_scen.clone()).or_default().teams[side] = Some(f[2].to_string());
            }
            "INIT" => {
                assert_eq!(f.len(), 2, "INIT needs 2 fields (line {ln})");
                flush(&mut cur, &mut cases);
                cur = Some(RunCase {
                    scen: last_scen.clone(),
                    init_seed: f[1].to_string(),
                    decisions: Vec::new(),
                    ended: false,
                    winner: WinTok::None,
                });
            }
            "DEC" => {
                assert_eq!(f.len(), 11, "DEC needs 11 fields (line {ln}), got {}", f.len());
                let req = match f[1] {
                    "move" => ReqTok::Move,
                    "switch" => ReqTok::Switch,
                    other => panic!("bad request {other:?} (line {ln})"),
                };
                let force = [f[2] == "1", f[3] == "1"];
                let choice = [parse_choice(f[4]), parse_choice(f[5])];
                let seed_after = f[6].to_string();
                let parse_side = |field: &str| -> SideExpect {
                    let g: Vec<&str> = field.split(',').collect();
                    assert_eq!(g.len(), 7, "snapshot needs 7 comma fields (line {ln}): {field:?}");
                    SideExpect {
                        species: g[0].to_string(),
                        hp: g[1].parse().unwrap_or_else(|e| panic!("bad hp (line {ln}): {e}")),
                        maxhp: g[2].parse().unwrap_or_else(|e| panic!("bad maxhp (line {ln}): {e}")),
                        fainted: g[3] == "1",
                        status: parse_status(g[4]),
                        left: g[5].parse::<u16>().unwrap_or_else(|e| panic!("bad left (line {ln}): {e}")) as usize,
                        trapped: g[6] == "1",
                    }
                };
                let p1 = parse_side(f[7]);
                let p2 = parse_side(f[8]);
                let first_mover = f[9].to_string();
                let covered = f[10] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect { request: req, force, choice, seed_after, p1, p2, first_mover, covered });
            }
            "END" => {
                assert_eq!(f.len(), 3, "END needs 3 fields (line {ln})");
                let c = cur.as_mut().unwrap_or_else(|| panic!("END before INIT (line {ln})"));
                c.winner = match f[1] {
                    "p1" | "P1" => WinTok::P1,
                    "p2" | "P2" => WinTok::P2,
                    "tie" => WinTok::Tie,
                    "none" => WinTok::None,
                    other => panic!("bad winner {other:?} (line {ln})"),
                };
                c.ended = f[2] == "1";
            }
            other => panic!("unknown record {other:?} (line {ln})"),
        }
    }
    flush(&mut cur, &mut cases);
    (meta, cases)
}

fn opts_for(meta: &ScenMeta, init_seed: &str) -> BattleOptions {
    let t = &meta.teams;
    BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some(init_seed.to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(t[0].clone().expect("p1 team")) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(t[1].clone().expect("p2 team")) },
    }
}

fn species_id(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

fn req_eq(rust: &RequestKind, golden: ReqTok, force: [bool; 2]) -> bool {
    match (rust, golden) {
        (RequestKind::Move, ReqTok::Move) => true,
        (RequestKind::ForceSwitch { force: rf }, ReqTok::Switch) => *rf == force,
        _ => false,
    }
}

#[test]
fn ability_batch4_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(
        meta.len() >= 21,
        "expected >=21 scenarios (truant + inner focus + shadow tag + cute charm + color change \
         + king's rock + focus band + controls), got {}",
        meta.len()
    );
    assert!(cases.len() >= 1200, "expected the per-seed corpus (>=1200 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut trapped_assertions = 0usize;
    let mut covered_rows = 0usize;
    let mut covered_per_scen: BTreeMap<String, usize> = BTreeMap::new();
    let mut win_runs = 0usize;

    for case in &cases {
        let m = meta.get(&case.scen).unwrap_or_else(|| panic!("no meta for {}", case.scen));
        assert!(!case.decisions.is_empty(), "[{}] empty run", case.scen);

        let opts = opts_for(m, &case.init_seed);
        let mut battle = Battle::start_with_switchins(&opts, &d)
            .unwrap_or_else(|e| panic!("[{}] start failed: {e}", case.scen));

        assert_eq!(
            battle.state().unwrap().prng_seed(),
            case.init_seed,
            "[{}] init prng seed must equal the sim's (switch-ins draw-free)",
            case.scen
        );

        let script: Vec<ScriptDecision> = case
            .decisions
            .iter()
            .map(|dec| ScriptDecision { p1: dec.choice[0], p2: dec.choice[1] })
            .collect();
        let outcome = battle.state_mut().unwrap().run_full_battle(&script, &d);

        // (The count assert runs AFTER the per-decision loop — zip stops at the shorter
        // side, so the FIRST diverging decision's seed/state assert fires with the precise
        // diagnosis before a bare count mismatch would.)
        for (di, (rec, exp)) in outcome.decisions.iter().zip(case.decisions.iter()).enumerate() {
            assert!(
                req_eq(&rec.request, exp.request, exp.force),
                "[{}] decision {} request mismatch (init_seed {}): got {:?} exp {:?} force {:?}",
                case.scen, di, case.init_seed, rec.request, exp.request, exp.force
            );

            for (idx, (snap, e, sp)) in [
                (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0])),
                (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1])),
            ] {
                assert_eq!(
                    species_id(sp),
                    species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {})",
                    case.scen, di, idx, case.init_seed
                );
                // --- THE HP GATE: a Focus-Band survive that mis-capped, a Color-Change
                //     override whose chart read is wrong (an EQ into Electric-Kecleon), or a
                //     wrongly-cant'd/wrongly-loafing move lands a wrong HP. ---
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     a batch-4 effect is wrong (a Focus-Band survive, a Color-Change chart \
                     read, a Truant loaf that attacked, an attract-cant'd move that hit).",
                    case.scen, di, idx, case.init_seed, snap.hp, e.hp
                );
                assert_eq!(snap.maxhp, e.maxhp, "[{}] dec {} side {} maxhp", case.scen, di, idx);
                assert_eq!(
                    snap.fainted, e.fainted,
                    "[{}] dec {} side {} fainted mismatch (init_seed {}): a Focus-Band survive \
                     that should/shouldn't have fired, or a wrong loaf/cant, KO'd the wrong mon",
                    case.scen, di, idx, case.init_seed
                );
                if !e.fainted {
                    // --- THE STATUS GATE: Color Change's status type-immunity (Toxic must FAIL
                    //     on a Poison-overridden Kecleon), the Cute-Charm gender gate. ---
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                }
                // --- THE TRAPPED GATE (Shadow Tag): unconditional (Flying Skarmory trapped),
                //     mutual in the mirror. Asserted at move-request boundaries where BOTH
                //     actives are ALIVE — the sim's `pokemon.trapped` is a stale endTurn-cached
                //     flag that persists on a mon KO'd this turn, while the port's live
                //     `is_trapped` returns false for a fainted pair (`isAdjacent` semantics;
                //     the forced replacement is always accepted either way, so the LEGALITY
                //     gate is identical — only the stale flag differs). ---
                if exp.request == ReqTok::Move && !exp.p1.fainted && !exp.p2.fainted {
                    assert_eq!(
                        rec.trapped[idx], e.trapped,
                        "[{}] dec {} side {} TRAPPED mismatch (init_seed {}): got {} exp {} — \
                         the Shadow Tag gate disagrees with the sim's pokemon.trapped",
                        case.scen, di, idx, case.init_seed, rec.trapped[idx], e.trapped
                    );
                    trapped_assertions += 1;
                }
            }
            assert_eq!(rec.pokemon_left[0], exp.p1.left, "[{}] dec {} p1 left", case.scen, di);
            assert_eq!(rec.pokemon_left[1], exp.p2.left, "[{}] dec {} p2 left", case.scen, di);

            if exp.request == ReqTok::Move {
                let sim_first: Option<usize> = match exp.first_mover.as_str() {
                    "p1" => Some(0),
                    "p2" => Some(1),
                    _ => None,
                };
                if sim_first.is_some() {
                    // --- THE FIRST-MOVER GATE: a Truant loaf / attract cant / flinch shows as a
                    //     `cant` line the sim counts as the "first mover" too — a wrong block
                    //     flips this. ---
                    assert_eq!(
                        rec.first_mover, sim_first,
                        "[{}] dec {} first-mover mismatch (init_seed {})",
                        case.scen, di, case.init_seed
                    );
                }
            }

            // --- PER-DECISION SEED PARITY (the draw-model gate): the KR appended random(100)
            //     [own-secondary-first order], the FB onDamage 1/10 (on every Damage event —
            //     chips + Spikes included), the CC 1/3 (drawn even for the F-into-F control),
            //     the attract 1/2, the Truant mirror's order-27 tie-shuffle — and the
            //     DRAW-FREENESS of the loaf turn (NO para roll), Inner Focus (the flinch roll
            //     still drawn), Shadow Tag (0 draws), Color Change (0 draws). ANY extra/
            //     missing/mis-ordered draw desyncs the LCG here. FIX THE DRAW ORDER, do not
            //     loosen the assert. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a batch-4 mechanic consumed/skipped/mis-ordered a PRNG draw.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
            if exp.covered {
                covered_rows += 1;
                *covered_per_scen.entry(case.scen.clone()).or_default() += 1;
            }
        }

        assert_eq!(
            outcome.decisions.len(),
            case.decisions.len(),
            "[{}] decision count mismatch (init_seed {}): rust {} vs golden {}",
            case.scen,
            case.init_seed,
            outcome.decisions.len(),
            case.decisions.len()
        );
        assert_eq!(
            outcome.ended, case.ended,
            "[{}] ended mismatch (init_seed {})",
            case.scen, case.init_seed
        );
        let rust_win = match outcome.winner {
            Some(0) => WinTok::P1,
            Some(1) => WinTok::P2,
            Some(other) => panic!("[{}] bad winner side {other}", case.scen),
            None if outcome.ended => WinTok::Tie,
            None => WinTok::None,
        };
        assert_eq!(
            rust_win, case.winner,
            "[{}] WINNER mismatch (init_seed {}): got {:?} exp {:?}",
            case.scen, case.init_seed, rust_win, case.winner
        );
        if matches!(case.winner, WinTok::P1 | WinTok::P2) {
            win_runs += 1;
        }
    }

    // Coverage floors: EVERY effect-bearing scenario must land its observable effect; the three
    // CONTROLS must have 0 (Inner Focus never flinch-cants; F-into-F never attracts; no-item
    // never flinches).
    let controls = ["innerfocus_blocks_flinch", "cutecharm_gender_control", "kingsrock_control_no_item"];
    let cover_exempt = ["shadowtag_traps_unconditionally", "shadowtag_mirror_mutual"];
    for (scen, _) in meta.iter() {
        let n = covered_per_scen.get(scen).copied().unwrap_or(0);
        if controls.contains(&scen.as_str()) {
            assert_eq!(n, 0, "[{scen}] a CONTROL must have 0 cover rows, got {n}");
        } else if cover_exempt.contains(&scen.as_str()) {
            // Shadow Tag has no protocol marker — the TRAPPED columns are its assert.
        } else {
            assert!(n >= 1, "[{scen}] {n} covered rows (<1) — the class effect never fired");
        }
    }
    // Shadow Tag floors: the trapped columns must have realized TRUE rows (both scenarios).
    assert!(trapped_assertions >= 2000, "expected trapped assertions (>=2000), got {trapped_assertions}");
    assert!(seed_assertions >= 4000, "expected the per-decision seed corpus (>=4000), got {seed_assertions}");
    assert!(covered_rows >= 800, "expected covered rows (>=800), got {covered_rows}");
    assert!(win_runs >= 500, "expected real game-end WIN runs (>=500), got {win_runs}");

    eprintln!(
        "batch4 golden: {} runs over {} scenarios, {dec_assertions} STATE rows, \
         {seed_assertions} seed assertions, {trapped_assertions} trapped assertions, \
         {covered_rows} covered rows, {win_runs} wins",
        cases.len(),
        meta.len()
    );
}

/// Byte-reproducibility gate (Mandate 3): the committed golden has a STABLE md5 — a regen at
/// the committed seeds/scenarios must reproduce it byte-for-byte. If this fails after a
/// DELIBERATE golden change, update the constant; an ACCIDENTAL change is a red flag.
#[test]
fn ability_batch4_golden_is_byte_reproducible() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/ability_batch4_golden.txt");
    let data = std::fs::read(path).expect("batch4 golden present");
    let digest = md5_hex(&data);
    assert_eq!(
        digest, BATCH4_GOLDEN_MD5,
        "ability_batch4_golden.txt md5 changed ({digest} != {BATCH4_GOLDEN_MD5}).\n  \
         If you DELIBERATELY changed the golden (scenarios/seeds), update the constant. Otherwise \
         a regen drifted — investigate before committing."
    );
}

const BATCH4_GOLDEN_MD5: &str = "519af5984cfae23f933e229f118b23fb";


// A tiny dependency-free md5 (RFC 1321) — the crate is std-only, so we can't pull `md-5`.
fn md5_hex(msg: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
        14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15,
        21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8,
        0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
        0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87,
        0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039,
        0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
        0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];
    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    let mut data = msg.to_vec();
    let bitlen = (msg.len() as u64).wrapping_mul(8);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bitlen.to_le_bytes());

    for chunk in data.chunks(64) {
        let mut mword = [0u32; 16];
        for (i, w) in mword.iter_mut().enumerate() {
            *w = u32::from_le_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]);
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | (!b & d), i),
                16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(mword[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = String::with_capacity(32);
    for v in [a0, b0, c0, d0] {
        for byte in v.to_le_bytes() {
            out.push_str(&format!("{byte:02x}"));
        }
    }
    out
}
