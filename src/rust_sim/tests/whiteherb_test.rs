//! WHITE HERB (BOOST_RESTORE) test (`gen3_white_herb_v1`) — the per-seed PER-DECISION
//! STATE+HP+STATUS+BOOSTS(7-stage/side)+ITEM(/side)+SEED+winner differential proving White Herb
//! matches Showdown EXACTLY, to GAME-END.
//!
//! White Herb (`ItemData.boost_restore`) restores ALL of the holder's NEGATIVE boost stages to 0
//! (positives untouched) then CONSUMES itself, DRAW-FREE. It fires from `onAnyAfterMove` /
//! `onAnySwitchIn` / `onResidual(order 29)`, so it triggers immediately after the causing stat-drop:
//! the holder's OWN self-drop move (Superpower), a foe's stat-drop MOVE (Charm), or a foe's
//! Intimidate-on-switch-in. Since it is draw-free, the per-decision SEED must match bit-for-bit; the
//! 7-stage BOOST columns + the ITEM presence (`item_held`, whiteherb → consumed) are the effect proof.
//!
//! The golden (`harness/gen_whiteherb_golden.js`) drives the OMNISCIENT BattleStream to game-end (the
//! foe never attacks → guaranteed P1 win); this test replays each (scenario, seed) from the sim's
//! init seed WITHOUT re-seeding and asserts, per decision boundary: both actives' species / hp /
//! maxhp / fainted / status + the 7 BOOST stages + the ITEM-held flag + pokemon_left + request kind
//! + first mover + the post-decision PRNG seed + the final winner.
//!   wh_self_drop    — Superpower self-drop restored then single-use (later drops unrestored).
//!   wh_foe_charm    — a foe Charm (−2 Atk) restored (`apply_secondary_boost` path).
//!   wh_intimidate   — a LEAD Intimidate restore during construction (`start_with_switchins` path).
//!   wh_net_positive — SD (+2) then Charm (−2) → net 0, NO negative → item RETAINED (no-trigger).
//! The revert-verified WH1-WH4 pins in `regression_test.rs` are the wrong-restore discriminator.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::{Status, BOOST_LEN};
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
    /// The sim-side marker: a White Herb `|-enditem|…|White Herb` fired this decision (coverage).
    wh: bool,
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
    /// The 7 boost stages `[atk, def, spa, spd, spe, acc, eva]` — the RESTORE proof.
    boosts: [i8; BOOST_LEN],
    /// Whether the active still HOLDS an item (whiteherb present) — the CONSUMPTION proof.
    item_held: bool,
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
        "-" | "fnt" => None,
        "brn" => Some(Status::Burn),
        "par" => Some(Status::Paralysis),
        "slp" => Some(Status::Sleep(0)),
        "frz" => Some(Status::Freeze),
        "psn" => Some(Status::Poison),
        "tox" => Some(Status::Toxic(0)),
        other => panic!("unknown status token {other:?}"),
    }
}

fn parse_boosts(tok: &str) -> [i8; BOOST_LEN] {
    let mut out = [0i8; BOOST_LEN];
    let parts: Vec<&str> = tok.split(',').collect();
    assert_eq!(parts.len(), BOOST_LEN, "boost column needs {BOOST_LEN} stages, got {tok:?}");
    for (i, p) in parts.iter().enumerate() {
        out[i] = p.parse().unwrap_or_else(|e| panic!("bad boost {p:?}: {e}"));
    }
    out
}

/// The golden's item column ("" = consumed, "-" = absent active, else the item id like
/// "whiteherb") → the port's `item_held` boolean (does the mon hold ANY item?).
fn parse_item_held(tok: &str) -> bool {
    !(tok.is_empty() || tok == "-")
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/whiteherb_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing whiteherb golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_whiteherb_golden.js")
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
                //   p1(species hp maxhp fnt status left)[9..15) p1boosts[15] p1item[16]
                //   p2(...)[17..23) p2boosts[23] p2item[24]  first[25]  wh[26]
                assert_eq!(f.len(), 27, "DEC needs 27 fields (line {ln}), got {}", f.len());
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
                    boosts: parse_boosts(f[15]),
                    item_held: parse_item_held(f[16]),
                };
                let p2 = SideExpect {
                    species: f[17].to_string(),
                    hp: g(18),
                    maxhp: g(19),
                    fainted: f[20] == "1",
                    status: parse_status(f[21]),
                    left: g(22) as usize,
                    boosts: parse_boosts(f[23]),
                    item_held: parse_item_held(f[24]),
                };
                let first_mover = f[25].to_string();
                let wh = f[26] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect { request: req, force, choice, seed_after, p1, p2, first_mover, wh });
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
fn whiteherb_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 4, "expected >=4 scenarios, got {}", meta.len());
    assert!(cases.len() >= 120, "expected the per-seed corpus (>=120 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut boost_assertions = 0usize;
    let mut item_assertions = 0usize;
    let mut wh_rows = 0usize;
    let mut wh_per_scen: BTreeMap<String, usize> = BTreeMap::new();
    // The ctor-restore proof: dec0's WH holder (p2) already consumed the item.
    let mut ctor_restores = 0usize;
    let mut win_runs = 0usize;
    let mut tie_runs = 0usize;

    for case in &cases {
        let m = meta.get(&case.scen).unwrap_or_else(|| panic!("no meta for {}", case.scen));
        assert!(!case.decisions.is_empty(), "[{}] empty run", case.scen);

        let opts = opts_for(m, &case.init_seed);
        let mut battle = Battle::start_with_switchins(&opts, &d)
            .unwrap_or_else(|e| panic!("[{}] start failed: {e}", case.scen));
        assert_eq!(
            battle.state().unwrap().prng_seed(),
            case.init_seed,
            "[{}] init prng seed must equal the sim's (switch-ins + White Herb restore are draw-free)",
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
            case.scen, case.init_seed, outcome.decisions.len(), case.decisions.len()
        );

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
                if !e.fainted {
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    // --- THE WHITE HERB GATE (1): the 7 boost stages. White Herb restores ONLY the
                    //     NEGATIVE stages to 0 (positives kept). A wrong/absent restore leaves the
                    //     negative stages standing → diverges here (and downstream). ---
                    assert_eq!(
                        &snap.boosts[..], &e.boosts[..],
                        "[{}] dec {} side {} BOOST mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         White Herb must restore ONLY the negative stages to 0 (positives kept). \
                         Check white_herb_restore at the after-move / switch-in stat-drop sites.",
                        case.scen, di, idx, case.init_seed, &snap.boosts[..], &e.boosts[..]
                    );
                    boost_assertions += 1;
                    // --- THE WHITE HERB GATE (2): item consumption. White Herb is SINGLE-USE — it
                    //     is consumed the first time it restores (whiteherb → item_held false) and
                    //     NEVER restores again; the net-positive case leaves it RETAINED. ---
                    assert_eq!(
                        snap.item_held, e.item_held,
                        "[{}] dec {} side {} ITEM mismatch (init_seed {}): got item_held {} exp {}\n  \
                         White Herb is single-use (consumed on restore) / RETAINED on a net-positive drop.",
                        case.scen, di, idx, case.init_seed, snap.item_held, e.item_held
                    );
                    item_assertions += 1;
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

            // --- PER-DECISION SEED PARITY: White Herb is DRAW-FREE — a spurious restore/emit draw
            //     would desync the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 the White Herb path consumed/skipped a PRNG draw it must not (White Herb is draw-free).",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
            if exp.wh {
                wh_rows += 1;
                *wh_per_scen.entry(case.scen.clone()).or_default() += 1;
            }
        }

        // The construction restore (`start_with_switchins`): a lead-Intimidate scenario's dec0 WH
        // holder must already read item_held false (consumed during construction).
        if case.scen == "wh_intimidate" {
            let d0 = &outcome.decisions[0];
            if !d0.active[1].item_held && d0.active[1].boosts[0] >= 0 {
                ctor_restores += 1;
            }
        }

        assert_eq!(outcome.ended, case.ended, "[{}] ended mismatch (init_seed {})", case.scen, case.init_seed);
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

    // Coverage floors: the in-decision restore fires repeatedly for the self-drop + foe-Charm
    // scenarios; the net-positive scenario NEVER restores (item retained); the lead-Intimidate
    // scenario restores during construction (dec0 item consumed).
    for scen in ["wh_self_drop", "wh_foe_charm"] {
        let n = wh_per_scen.get(scen).copied().unwrap_or(0);
        assert!(n >= 10, "[{scen}] only {n} White Herb rows (<10) — the restore never fired");
    }
    let np = wh_per_scen.get("wh_net_positive").copied().unwrap_or(0);
    assert_eq!(np, 0, "[wh_net_positive] expected 0 White Herb rows (net non-negative), got {np}");
    assert!(ctor_restores >= 10, "expected the lead-Intimidate construction restore (>=10), got {ctor_restores}");
    assert!(seed_assertions >= 300, "expected the per-decision seed corpus (>=300), got {seed_assertions}");
    assert!(win_runs >= 120, "expected real game-end WIN runs (>=120), got {win_runs}");

    eprintln!(
        "whiteherb golden: {} runs over {} scenarios, {dec_assertions} STATE rows, \
         {boost_assertions} boost assertions, {item_assertions} item assertions, \
         {seed_assertions} seed assertions, {wh_rows} in-decision restore rows, \
         {ctor_restores} construction restores, {win_runs} wins, {tie_runs} ties",
        cases.len(),
        meta.len()
    );
}
