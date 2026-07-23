//! TRICK (item swap) test (`gen3_trick_v1`) — the per-seed PER-DECISION
//! STATE+HP+STATUS+BOOSTS(7-stage/side)+ITEM-NUM(/side)+SEED+winner differential proving Trick
//! matches Showdown EXACTLY, to GAME-END.
//!
//! Trick (`trick`, num 271) is a category-Status ITEM-SWAP move (type Psychic, accuracy 100 →
//! ONE `randomChance(100,100)` draw, `target: normal`, NO `bypasssub`). The DRAW MODEL is ONE
//! accuracy draw then a DRAW-FREE swap, so the per-decision SEED must match bit-for-bit; the
//! per-side ITEM dex-NUM columns (0 = itemless) are the swap-identity proof — a two-item swap
//! keeps BOTH sides holding an item, so only the NUM (not a presence boolean) distinguishes it.
//!
//! The golden (`harness/gen_trick_golden.js`) drives the OMNISCIENT BattleStream to game-end (the
//! foe never attacks → guaranteed P1 win); this test replays each (scenario, seed) from the sim's
//! init seed WITHOUT re-seeding and asserts, per decision boundary: both actives' species / hp /
//! maxhp / fainted / status + the 7 boosts + the ITEM NUM + pokemon_left + request kind + first
//! mover + the post-decision seed + the final winner.
//!   trick_two_swap      — a full two-item swap (Silk Scarf <-> Leftovers).
//!   trick_one_sided     — p1 itemless → the foe loses its item (`-enditem [silent]`), p1 gains it.
//!   trick_sticky_hold   — Sticky Hold Muk → PLAIN `-immune`, NO swap (items UNCHANGED).
//!   trick_substitute    — Trick into a Substitute → `[still]`+`-fail`, NO swap (no bypasssub).
//!   trick_both_itemless — both sides itemless → FAILS (`[still]`+`-fail`, both num 0).
//!   trick_cb_release    — a Choice-Band user Tricks its Band AWAY → UNLOCKED (uses a different
//!                         slot next turn; a kept lock would diverge the DECISION COUNT here).
//!   trick_real_battle   — Trick composed with a voluntary switch + a forced replacement.
//! The revert-verified pins in `regression_test.rs` are the wrong-swap/wrong-fail discriminator.

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
    /// The sim-side markers: `used` = a `|move|…|Trick|` resolved this decision; `swap` = a
    /// `move: Trick` line (a successful swap) fired this decision (coverage).
    used: bool,
    swap: bool,
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
    boosts: [i8; BOOST_LEN],
    /// The active's held item's dex NUM (`0` = itemless) — the swap-identity proof.
    item_num: u16,
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/trick_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing trick golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_trick_golden.js")
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
                //   p1(species hp maxhp fnt status left)[9..15) p1boosts[15] p1itemnum[16]
                //   p2(...)[17..23) p2boosts[23] p2itemnum[24]  first[25]  used[26]  swap[27]
                assert_eq!(f.len(), 28, "DEC needs 28 fields (line {ln}), got {}", f.len());
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
                    item_num: g(16),
                };
                let p2 = SideExpect {
                    species: f[17].to_string(),
                    hp: g(18),
                    maxhp: g(19),
                    fainted: f[20] == "1",
                    status: parse_status(f[21]),
                    left: g(22) as usize,
                    boosts: parse_boosts(f[23]),
                    item_num: g(24),
                };
                let first_mover = f[25].to_string();
                let used = f[26] == "1";
                let swap = f[27] == "1";
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect { request: req, force, choice, seed_after, p1, p2, first_mover, used, swap });
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
fn trick_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 7, "expected >=7 scenarios, got {}", meta.len());
    assert!(cases.len() >= 200, "expected the per-seed corpus (>=200 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut boost_assertions = 0usize;
    let mut item_assertions = 0usize;
    let mut used_rows = 0usize;
    let mut swap_rows = 0usize;
    let mut swap_per_scen: BTreeMap<String, usize> = BTreeMap::new();
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
            "[{}] init prng seed must equal the sim's",
            case.scen
        );

        let script: Vec<ScriptDecision> = case
            .decisions
            .iter()
            .map(|dec| ScriptDecision { p1: dec.choice[0], p2: dec.choice[1] })
            .collect();
        let outcome = battle.state_mut().unwrap().run_full_battle(&script, &d);

        // The DECISION-COUNT match is the load-bearing choice-lock check: `trick_cb_release`
        // Tricks a Choice Band AWAY then uses a DIFFERENT slot the next turn. If the port failed
        // to RELEASE the Choice lock, `run_full_battle` would reject that decision (locked-out
        // slot) and skip it → a decision-count mismatch here.
        assert_eq!(
            outcome.decisions.len(),
            case.decisions.len(),
            "[{}] decision count mismatch (init_seed {}): rust {} vs golden {}\n  \
             (a kept Choice lock after Tricking the Band away would reject the next move here)",
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
                    assert_eq!(
                        &snap.boosts[..], &e.boosts[..],
                        "[{}] dec {} side {} BOOST mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, &snap.boosts[..], &e.boosts[..]
                    );
                    boost_assertions += 1;
                    // --- THE TRICK GATE: WHICH item each side holds. A two-item swap keeps both
                    //     sides item-holding, so only the dex NUM distinguishes a correct swap from
                    //     a wrong / absent / one-sided one. A Sticky-Hold `-immune`, a Substitute /
                    //     both-itemless / knocked-off FAIL, and a genuine swap all diverge here if
                    //     the item routing is wrong. ---
                    assert_eq!(
                        snap.item_num, e.item_num,
                        "[{}] dec {} side {} ITEM-NUM mismatch (init_seed {}): got {} exp {}\n  \
                         Trick must swap the two items (or leave them on a Sticky-Hold / Substitute \
                         / both-itemless / knocked-off fail). Check the trick arm in run_status_move.",
                        case.scen, di, idx, case.init_seed, snap.item_num, e.item_num
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

            // --- PER-DECISION SEED PARITY: Trick draws ONE accuracy roll then a DRAW-FREE swap. A
            //     spurious/missing swap draw would desync the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 the Trick path consumed/skipped a PRNG draw it must not (swap is draw-free past accuracy).",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
            if exp.used {
                used_rows += 1;
            }
            if exp.swap {
                swap_rows += 1;
                *swap_per_scen.entry(case.scen.clone()).or_default() += 1;
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

    // Coverage floors: the swap fires for the swap scenarios; the no-swap scenarios (Sticky Hold /
    // Substitute / both-itemless) NEVER swap (item nums unchanged / itemless).
    for scen in ["trick_two_swap", "trick_one_sided", "trick_cb_release", "trick_real_battle"] {
        let n = swap_per_scen.get(scen).copied().unwrap_or(0);
        assert!(n >= 30, "[{scen}] only {n} swap rows (<30) — the swap never fired");
    }
    for scen in ["trick_sticky_hold", "trick_substitute", "trick_both_itemless"] {
        let n = swap_per_scen.get(scen).copied().unwrap_or(0);
        assert_eq!(n, 0, "[{scen}] expected 0 swap rows (blocked/failed Trick), got {n}");
    }
    assert!(used_rows >= 200, "expected the Trick-used corpus (>=200), got {used_rows}");
    assert!(seed_assertions >= 800, "expected the per-decision seed corpus (>=800), got {seed_assertions}");
    assert!(win_runs >= 200, "expected real game-end WIN runs (>=200), got {win_runs}");

    eprintln!(
        "trick golden: {} runs over {} scenarios, {dec_assertions} STATE rows, \
         {boost_assertions} boost assertions, {item_assertions} item-num assertions, \
         {seed_assertions} seed assertions, {used_rows} Trick-used rows, {swap_rows} swap rows, \
         {win_runs} wins, {tie_runs} ties",
        cases.len(),
        meta.len()
    );
}
