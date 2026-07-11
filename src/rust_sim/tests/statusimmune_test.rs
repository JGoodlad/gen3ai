//! STATUS_IMMUNE ability class-sweep tests (`gen3_status_immune_v1`) — the per-seed
//! PER-DECISION STATE+HP+STATUS+SEED+winner differential proving the status-immunity abilities
//! match Showdown EXACTLY, to GAME-END. The gen-3 members (probe-settled +
//! `AbilityData.status_immune`-driven): Limber (par) / Insomnia + Vital Spirit (slp) / Immunity
//! (psn,tox) / Water Veil (brn) block via `onSetStatus`; Magma Armor (frz) blocks via
//! `onImmunity` (BEFORE the SetStatus event). In gen3customgame every block is DRAW-FREE.
//!
//! The immunity is OBSERVABLE on the ACTIVE-mon STATUS timeline: a foe RE-FIRES the matching
//! status move at the immune Snorlax every turn; the immune Snorlax STAYS `-` (unstatused) all
//! battle and Seismic-Tosses the foe down (WIN). A NON-immune CONTROL on the IDENTICAL plan/teams
//! gets STATUSED → its trajectory DIVERGES (a paralyzed/asleep control acts differently; a
//! burned/toxic'd control takes DoT; a frozen control can't attack). WRONG-STATUS controls
//! (a Limber mon takes a BURN normally) prove the block is status-SPECIFIC.
//!
//! The golden (`harness/gen_statusimmune_golden.js`) drives the OMNISCIENT BattleStream to
//! game-end; this test replays each (scenario, seed) from the sim's init seed WITHOUT re-seeding
//! and asserts, per decision boundary: both actives' species/hp/maxhp/fainted/status +
//! pokemon_left + the request kind + the first mover + the post-decision PRNG seed + the final
//! winner. The block is DRAW-FREE, so the per-decision seed must match a status-lands battle
//! bit-for-bit at the application point — any spurious block draw desyncs it. The `status` column
//! assertion IS the immunity gate: the immune holder's `-` (vs the control's status) at every
//! decision proves the block.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Status;
use pokesim::turn::{Choice, RequestKind, ScriptDecision};
use std::collections::BTreeMap;

fn dex() -> Dex {
    Dex::for_gen(3)
}

const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/statusimmune_golden.txt");

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
    /// The sim-side marker: an immunity `-immune` fired this decision (coverage only). Note a
    /// SECONDARY-freeze block (Magma Armor vs Ice Beam) is SILENT (no `-immune`), so its coverage
    /// is proven by the STATE differential (the frz-control freezes, the immune holder never).
    blocked: bool,
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
    let data = std::fs::read_to_string(GOLDEN).unwrap_or_else(|e| {
        panic!("missing statusimmune golden ({GOLDEN}): {e}\nrun: node src/rust_sim/harness/gen_statusimmune_golden.js")
    });

    let mut meta: BTreeMap<String, ScenMeta> = BTreeMap::new();
    let mut cases: Vec<RunCase> = Vec::new();
    let mut cur: Option<RunCase> = None;

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
                meta.entry(f[1].to_string()).or_default();
            }
            "TEAM" => {
                assert_eq!(f.len(), 4, "TEAM needs 4 fields (line {ln})");
                let side = if f[2] == "p1" { 0 } else { 1 };
                meta.entry(f[1].to_string()).or_default().teams[side] = Some(f[3].to_string());
            }
            "INIT" => {
                assert_eq!(f.len(), 4, "INIT needs 4 fields (line {ln})");
                flush(&mut cur, &mut cases);
                cur = Some(RunCase {
                    scen: f[1].to_string(),
                    init_seed: f[2].to_string(),
                    decisions: Vec::new(),
                    ended: false,
                    winner: WinTok::None,
                });
            }
            "DEC" => {
                // DEC <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter>
                //   p1(species hp maxhp fnt status left)[9..15)
                //   p2(...)[15..21)  first[21]  blocked[22]
                assert_eq!(f.len(), 23, "DEC needs 23 fields (line {ln}), got {}", f.len());
                let req = match f[3] {
                    "move" => ReqTok::Move,
                    "switch" => ReqTok::Switch,
                    other => panic!("bad request {other:?} (line {ln})"),
                };
                let force = [f[4] == "1", f[5] == "1"];
                let choice = [parse_choice(f[6]), parse_choice(f[7])];
                let seed_after = f[8].to_string();
                let g = |i: usize| {
                    f[i].parse::<u16>().unwrap_or_else(|e| panic!("bad num f[{i}] (line {ln}): {e}"))
                };
                let p1 = SideExpect {
                    species: f[9].to_string(),
                    hp: g(10),
                    maxhp: g(11),
                    fainted: f[12] == "1",
                    status: parse_status(f[13]),
                    left: g(14) as usize,
                };
                let p2 = SideExpect {
                    species: f[15].to_string(),
                    hp: g(16),
                    maxhp: g(17),
                    fainted: f[18] == "1",
                    status: parse_status(f[19]),
                    left: g(20) as usize,
                };
                let first_mover = f[21].to_string();
                let blocked = f[22] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req,
                    force,
                    choice,
                    seed_after,
                    p1,
                    p2,
                    first_mover,
                    blocked,
                });
            }
            "END" => {
                assert_eq!(f.len(), 5, "END needs 5 fields (line {ln})");
                let c = cur.as_mut().unwrap_or_else(|| panic!("END before INIT (line {ln})"));
                c.ended = f[3] == "1";
                c.winner = match f[4] {
                    "p1" => WinTok::P1,
                    "p2" => WinTok::P2,
                    "tie" => WinTok::Tie,
                    "none" => WinTok::None,
                    other => panic!("bad winner {other:?} (line {ln})"),
                };
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
fn statusimmune_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(
        meta.len() >= 12,
        "expected >=12 scenarios (6 immune members + 4 status-controls + 2 wrong-status controls), got {}",
        meta.len()
    );
    assert!(cases.len() >= 400, "expected the per-seed corpus (>=400 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut status_assertions = 0usize;
    let mut block_rows = 0usize;
    let mut block_per_scen: BTreeMap<String, usize> = BTreeMap::new();
    let mut win_runs = 0usize;
    let mut tie_runs = 0usize;
    // The STATE proofs: the immune holders NEVER become statused; the Magma Armor frz-control DOES
    // freeze on >=1 seed (so the immune holder staying frz-free IS the block).
    let immune_scens = [
        "si_limber_par",
        "si_insomnia_slp",
        "si_vitalspirit_slp",
        "si_immunity_tox",
        "si_waterveil_brn",
        "si_magmaarmor_frz",
    ];
    let mut frz_control_froze = false;
    let mut immune_holder_clean_rows = 0usize;

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

        assert_eq!(
            outcome.decisions.len(),
            case.decisions.len(),
            "[{}] decision count mismatch (init_seed {}): rust {} vs golden {}",
            case.scen,
            case.init_seed,
            outcome.decisions.len(),
            case.decisions.len()
        );

        let is_immune_scen = immune_scens.contains(&case.scen.as_str());

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
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, snap.hp, e.hp
                );
                assert_eq!(snap.maxhp, e.maxhp, "[{}] dec {} side {} maxhp", case.scen, di, idx);
                assert_eq!(
                    snap.fainted, e.fainted,
                    "[{}] dec {} side {} fainted mismatch (init_seed {})",
                    case.scen, di, idx, case.init_seed
                );
                // --- THE STATUS_IMMUNE GATE: the immune holder (side 0 on an immune scenario)
                //     must stay UNSTATUSED (status column `-`) while the foe re-fires the matching
                //     status move; a non-immune control (or a wrong-status one) gets STATUSED, which
                //     the general status assertion also pins. A wrong/absent block would status the
                //     immune holder here (and desync the DoT/wake HP + the winner). ---
                if !e.fainted {
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         A STATUS_IMMUNE ability must BLOCK its immune status (the holder stays \
                         unstatused); a non-immune mon takes it. Check try_set_status's data-driven \
                         status_immune gate (the setStatus/immunity phase) in turn.rs.",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    status_assertions += 1;
                    if is_immune_scen && idx == 0 {
                        // The immune holder's status column MUST be None here (the STATE proof).
                        assert!(
                            snap.status.is_none(),
                            "[{}] dec {} the IMMUNE HOLDER is statused ({:?}) — the block failed",
                            case.scen, di, snap.status
                        );
                        immune_holder_clean_rows += 1;
                    }
                }
                // The frz-control STATE discriminator (Magma Armor's silent-block proof).
                if case.scen == "si_control_frz_none"
                    && idx == 0
                    && matches!(e.status, Some(Status::Freeze))
                {
                    frz_control_froze = true;
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
                    assert_eq!(
                        rec.first_mover, sim_first,
                        "[{}] dec {} first-mover mismatch (init_seed {})",
                        case.scen, di, case.init_seed
                    );
                }
            }

            // --- PER-DECISION SEED PARITY: the block is DRAW-FREE (in gen3customgame the
            //     onSetStatus-phase ability makes the ONLY SetStatus handler → no shuffle; Magma
            //     Armor blocks before the event). A spurious block draw desyncs the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 the STATUS_IMMUNE path consumed/skipped a PRNG draw it must not (the block is \
                 draw-free in customgame). FIX THE DRAW ORDER, do not loosen the assert.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
            if exp.blocked {
                block_rows += 1;
                *block_per_scen.entry(case.scen.clone()).or_default() += 1;
            }
        }

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
        match case.winner {
            WinTok::P1 | WinTok::P2 => win_runs += 1,
            WinTok::Tie => tie_runs += 1,
            WinTok::None => {}
        }
    }

    // Coverage floor: each `onSetStatus`-phase member fires its `-immune` block repeatedly; the
    // controls fire ZERO (their status LANDS). Magma Armor's frz block is SILENT so it has no
    // `-immune` floor — its coverage is the STATE proof (the frz-control froze, the immune never).
    for scen in ["si_limber_par", "si_insomnia_slp", "si_vitalspirit_slp", "si_immunity_tox", "si_waterveil_brn"] {
        let n = block_per_scen.get(scen).copied().unwrap_or(0);
        assert!(n >= 10, "[{scen}] only {n} immunity `-immune` block rows (<10) — the block never fired");
    }
    for scen in [
        "si_control_par_none",
        "si_control_slp_none",
        "si_control_tox_none",
        "si_control_frz_none",
        "si_limber_takes_brn",
        "si_immunity_takes_brn",
        "si_magmaarmor_frz",
    ] {
        let n = block_per_scen.get(scen).copied().unwrap_or(0);
        assert_eq!(n, 0, "[{scen}] expected 0 `-immune` block rows (a control / silent-block scenario), got {n}");
    }
    assert!(
        frz_control_froze,
        "the NO-ability frz control NEVER froze — Magma Armor's block is unproven (widen the golden seeds)"
    );
    assert!(immune_holder_clean_rows >= 200, "expected the immune-holder clean-status corpus (>=200), got {immune_holder_clean_rows}");
    assert!(seed_assertions >= 1500, "expected the per-decision seed corpus (>=1500), got {seed_assertions}");
    assert!(block_rows >= 60, "expected immunity `-immune` block rows (>=60), got {block_rows}");
    assert!(win_runs >= 400, "expected real game-end WIN runs (>=400), got {win_runs}");

    eprintln!(
        "statusimmune golden: {} runs over {} scenarios, {dec_assertions} STATE rows, \
         {status_assertions} status assertions, {seed_assertions} seed assertions, \
         {block_rows} block rows, {immune_holder_clean_rows} immune-holder clean rows, \
         {win_runs} wins, {tie_runs} ties",
        cases.len(),
        meta.len()
    );
}

/// Byte-reproducibility gate (Mandate 3): the committed golden has a STABLE md5. This pins the
/// golden as a versioned artifact — a regen at the committed seeds/scenarios must reproduce it
/// byte-for-byte (the harness's PRNG-seed generator + scenario list are fixed). If this fails
/// after a deliberate golden change, update the constant; an ACCIDENTAL change is a red flag.
#[test]
fn statusimmune_golden_is_byte_reproducible() {
    let data = std::fs::read(GOLDEN).expect("statusimmune golden present");
    let digest = md5_hex(&data);
    assert_eq!(
        digest, STATUSIMMUNE_GOLDEN_MD5,
        "statusimmune_golden.txt md5 changed ({digest} != {STATUSIMMUNE_GOLDEN_MD5}).\n  \
         If you DELIBERATELY changed the golden (scenarios/seeds), update the constant. Otherwise \
         a regen drifted — investigate before committing."
    );
}

const STATUSIMMUNE_GOLDEN_MD5: &str = "4f258040cebfb803f2e9359f220f6ecf";

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
