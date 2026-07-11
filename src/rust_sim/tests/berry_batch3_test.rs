//! BATCH-3 class-sweep tests (`gen3_berry_trace_shedskin_v1`) — the per-seed PER-DECISION
//! STATE+HP+STATUS+**ITEM**+BOOSTS+SEED+winner differential proving the BERRY item classes
//! (ONE eatItem consumption mechanism + parameter rows), TRACE, and SHED SKIN match Showdown
//! EXACTLY, to game-end (or the capped rest-loop tail):
//!
//!   CURE_BERRY (cheri/chesto/pecha/rawst/aspear/persim/lum) — eats at the FIRST
//!     eachEvent('Update') after the condition (the holder never full-para-rolls that turn);
//!     LUM eats IMMEDIATELY inside setStatus (incl. Rest's self-sleep — LumRest). The ITEM
//!     timeline (held → eaten → stays gone) + the STATUS timeline are asserted per decision.
//!   HEAL_BERRY (oran 10 / sitrus 30 / the figy family floor(maxhp/8) + nature-gated
//!     confusion) — the residual order-10-subOrder-4 slot at 2*hp <= maxhp exactly. The figy
//!     confusion draws random(2,6) — a missing/extra draw desyncs the SEED.
//!   PINCH_BERRY (liechi/ganlon/salac/petaya/apicot +1; starf sample→+2; lansat focus
//!     energy) — at 4*hp <= maxhp. The BOOSTS CSV pins the +1/+2 stage; Starf's sample and
//!     the lansat crit-stage shift ride the seed.
//!   PP_BERRY (leppa) — eats at the Update when a slot hits 0 PP (+10 capped at maxpp).
//!   SUBSTITUTE — a sub-absorbed hit leaves the real hp untouched → no pinch trigger until
//!     the REAL hp crosses (the sub-cost path).
//!   TRACE — a mid-battle switch-in draws ONE sample (random(1) even for a single foe) and
//!     copies the foe's CURRENT ability (live passives — a traced Flash Fire absorbs); a
//!     LEAD trace's draw pre-dates the seeded start (the copy applies draw-free).
//!   SHED SKIN — ONE randomChance(33,100) per STATUSED residual (order 10 subOrder 3 — a
//!     cure turn takes NO DoT chip); an unstatused holder draws NOTHING.
//!
//! The golden (`harness/gen_berry_batch3_golden.js`, byte-reproducible) drives the OMNISCIENT
//! BattleStream; this test replays each (scenario, seed) from the sim's init seed WITHOUT
//! re-seeding and asserts, per decision boundary: both actives' species/hp/maxhp/fainted/
//! status/item-held/boosts + pokemon_left + the request kind + the first mover + the
//! post-decision PRNG seed + the final winner/ended.

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
    expect_zero_cover: bool,
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
    /// The sim's `pokemon.item` token (`-` = no item — either never held one, or the
    /// berry was EATEN). The scenario's held item is fixed, so this bool-izes cleanly.
    item_held: bool,
    /// `atk:def:spa:spd:spe` stage CSV — the pinch +1 / Starf +2 pin.
    boosts: [i8; 5],
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/berry_batch3_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing batch3 golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_berry_batch3_golden.js")
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
                assert_eq!(f.len(), 4, "SCEN needs 4 fields (line {ln})");
                last_scen = f[1].to_string();
                let m = meta.entry(last_scen.clone()).or_default();
                m.expect_zero_cover = f[3] == "1";
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
                    assert_eq!(g.len(), 8, "snapshot needs 8 comma fields (line {ln}): {field:?}");
                    let bs: Vec<i8> = g[7]
                        .split(':')
                        .map(|x| x.parse().unwrap_or_else(|e| panic!("bad boost (line {ln}): {e}")))
                        .collect();
                    assert_eq!(bs.len(), 5, "boosts CSV needs 5 stages (line {ln})");
                    SideExpect {
                        species: g[0].to_string(),
                        hp: g[1].parse().unwrap_or_else(|e| panic!("bad hp (line {ln}): {e}")),
                        maxhp: g[2].parse().unwrap_or_else(|e| panic!("bad maxhp (line {ln}): {e}")),
                        fainted: g[3] == "1",
                        status: parse_status(g[4]),
                        left: g[5].parse::<u16>().unwrap_or_else(|e| panic!("bad left (line {ln}): {e}")) as usize,
                        item_held: g[6] != "-",
                        boosts: [bs[0], bs[1], bs[2], bs[3], bs[4]],
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
fn berry_batch3_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(
        meta.len() >= 30,
        "expected >=30 scenarios (7 cure + 7 heal + 7 pinch + leppa + sub + shedskin + trace + controls), got {}",
        meta.len()
    );
    assert!(cases.len() >= 1200, "expected the per-seed corpus (>=1200 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut status_assertions = 0usize;
    let mut item_assertions = 0usize;
    let mut boost_assertions = 0usize;
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
            "[{}] init prng seed must equal the sim's (the lead-trace copy is draw-free here)",
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
                // --- THE HP GATE: a berry heal (oran 10 / sitrus 30 / figy maxhp/8), the
                //     LumRest full-heal-awake, or a traced Flash Fire absorb lands here. ---
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     a berry HEAL amount / threshold, or a traced-ability passive, is wrong.",
                    case.scen, di, idx, case.init_seed, snap.hp, e.hp
                );
                assert_eq!(snap.maxhp, e.maxhp, "[{}] dec {} side {} maxhp", case.scen, di, idx);
                assert_eq!(snap.fainted, e.fainted, "[{}] dec {} side {} fainted (init_seed {})",
                    case.scen, di, idx, case.init_seed);
                // --- THE ITEM GATE (the consumption crux): the berry must be EATEN at
                //     exactly the sim's decision (held → NONE), and STAY gone. ---
                assert_eq!(
                    snap.item_held, e.item_held,
                    "[{}] dec {} side {} ITEM mismatch (init_seed {}): held={} exp={}\n  \
                     the eatItem trigger (threshold / Update site / setStatus tail) fired at \
                     the wrong time, or a second eat happened.",
                    case.scen, di, idx, case.init_seed, snap.item_held, e.item_held
                );
                item_assertions += 1;
                if !e.fainted {
                    // --- THE STATUS GATE: a cure berry / Shed Skin cure (or its absence —
                    //     the wrong-status control keeps its status), a lum-immediate cure. ---
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         a CURE berry / Shed Skin cure fired wrongly (or failed to fire).",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    status_assertions += 1;
                    // --- THE BOOSTS GATE: the pinch +1 stage on the RIGHT stat / Starf's +2. ---
                    let got: [i8; 5] = [snap.boosts[0], snap.boosts[1], snap.boosts[2], snap.boosts[3], snap.boosts[4]];
                    assert_eq!(
                        got, e.boosts,
                        "[{}] dec {} side {} BOOSTS mismatch (init_seed {}): a PINCH berry's \
                         stat/stage (or Starf's sampled stat) is wrong.",
                        case.scen, di, idx, case.init_seed
                    );
                    boost_assertions += 1;
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

            // --- PER-DECISION SEED PARITY (the draw-model gate): the batch-3 draws are
            //     Starf's sample, the figy-family confusion random(2,6), Shed Skin's
            //     randomChance(33,100), and the mid-battle Trace sample — plus the
            //     DRAW-FREEness of every other eat/cure. An extra/missing/mis-positioned
            //     draw (a cure at the wrong Update site, a berry residual that drew, a
            //     Shed Skin roll on an unstatused holder) desyncs the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a batch-3 mechanic consumed/skipped/mis-ordered a PRNG draw.",
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

    // Coverage floors: every effect-bearing scenario fires its class effect; the declared
    // zero-cover controls (wrong-status cure / unstatused Shed Skin / the lead trace whose
    // -ability line pre-dates decision 0) MUST stay at 0.
    for (scen, m) in meta.iter() {
        let n = covered_per_scen.get(scen).copied().unwrap_or(0);
        if m.expect_zero_cover {
            assert_eq!(n, 0, "[{scen}] a zero-cover control must have 0 covered rows, got {n}");
        } else {
            assert!(n >= 5, "[{scen}] {n} covered rows (<5) — the class effect barely fired");
        }
    }
    assert!(seed_assertions >= 5000, "expected the per-decision seed corpus (>=5000), got {seed_assertions}");
    assert!(covered_rows >= 800, "expected covered rows (>=800), got {covered_rows}");
    assert!(win_runs >= 900, "expected real game-end WIN runs (>=900), got {win_runs}");

    eprintln!(
        "berry batch3 golden: {} runs over {} scenarios, {dec_assertions} STATE rows, \
         {seed_assertions} seed, {status_assertions} status, {item_assertions} item, \
         {boost_assertions} boost assertions, {covered_rows} covered rows, {win_runs} wins",
        cases.len(),
        meta.len()
    );
}
