//! Secondary-effects + onBeforeMove-status full-battle tests — the per-seed
//! PER-DECISION STATE(+STATUS)+SEED differential that proves the TWO new draw sites
//! this step adds match Showdown EXACTLY, sustained to GAME-END through real
//! secondary moves that status mons IN-ENGINE:
//!
//!   (A) the onBeforeMove STATUS draws (the NEW LEADING draw, BEFORE accuracy):
//!       para `randomChance(1,4)`, freeze `randomChance(1,5)` thaw, confusion
//!       `randomChance(1,2)` + a self-hit `random(16)`, sleep/flinch DRAW-FREE; and
//!   (B) the per-move SECONDARY `random(100)` (the NEW TRAILING draw, AFTER damage):
//!       Body Slam par30 / Ice Beam frz10 / Thunderbolt par10 / Rock Slide flinch30
//!       / Sludge Bomb psn30 — fired only on a LANDED hit, applied if `roll<chance`,
//!       with the onTrySetStatus gates (already-statused / type-immunity) DRAW-FREE,
//!       and NOT drawn on a DAMAGE-immune target (immune short-circuits before it).
//!
//!   `secondary_golden_matches_showdown` — the DIFFERENTIAL gate. For each
//!   (scenario, seed) in `harness/gen_secondary_golden.js`'s golden, seed a
//!   `BattleState` at the sim's PRNG state at the FIRST decision's pre-choice
//!   boundary (`init_seed`), run `run_full_battle(script)` WITHOUT re-seeding, and
//!   assert per DECISION BOUNDARY: (a) each side's post-decision active
//!   (species/hp/maxhp/fainted/STATUS) + pokemon_left + request kind + first mover
//!   match; AND (b) the post-decision PRNG seed equals the sim's `seed_after` — the
//!   EXACT cross-decision draw-order+count proof to game-end (one extra/missing/
//!   mis-ordered onBeforeMove or secondary draw desyncs the LCG). PLUS the final
//!   WINNER. A per-decision STATUS mismatch catches a secondary that wrongly
//!   applied/skipped; a SEED mismatch catches a mis-ordered/missing/extra draw.

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
    /// RNG outcomes the sim reported this decision (for the floor counters only):
    /// per side `[fullpara, wake, thaw, selfhit, flinch]`, then secondary-landed.
    out_p1: [bool; 5],
    out_p2: [bool; 5],
    secondary_landed: bool,
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
    /// The 5 stat stages `[atk, def, spa, spd, spe]` the sim reported (the boost
    /// STATE a stat-drop / self-boost secondary — or Intimidate on entry — changes).
    boosts: [i8; 5],
    /// The confusion counter the sim reported (0 = not confused).
    confusion: u8,
}

fn parse_status(tok: &str, stage: u16) -> Option<Status> {
    match tok {
        "-" => None,
        "fnt" => None,
        "brn" => Some(Status::Burn),
        "par" => Some(Status::Paralysis),
        "slp" => Some(Status::Sleep(stage as u8)),
        "frz" => Some(Status::Freeze),
        "psn" => Some(Status::Poison),
        "tox" => Some(Status::Toxic(stage as u8)),
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
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/secondary_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing secondary golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_secondary_golden.js")
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
                //   p1(species hp max fnt status stage left atk def spa spd spe confusion)[9..22)
                //   p2(...)[22..35)  first[35]
                //   p1 out(fullpara wake thaw selfhit flinch)[36..41)
                //   p2 out(...)[41..46)  secondaryLanded[46]
                assert_eq!(f.len(), 47, "DEC needs 47 fields (line {ln}), got {}", f.len());
                let req = match f[3] {
                    "move" => ReqTok::Move,
                    "switch" => ReqTok::Switch,
                    other => panic!("bad request {other:?} (line {ln})"),
                };
                let force = [f[4] == "1", f[5] == "1"];
                let choice = [parse_choice(f[6]), parse_choice(f[7])];
                let seed_after = f[8].to_string();
                let g = |i: usize| f[i].parse::<u16>().unwrap_or_else(|e| panic!("bad num f[{i}] (line {ln}): {e}"));
                let gi = |i: usize| f[i].parse::<i8>().unwrap_or_else(|e| panic!("bad int f[{i}] (line {ln}): {e}"));
                let p1 = SideExpect {
                    species: f[9].to_string(),
                    hp: g(10),
                    maxhp: g(11),
                    fainted: f[12] == "1",
                    status: parse_status(f[13], g(14)),
                    left: g(15) as usize,
                    boosts: [gi(16), gi(17), gi(18), gi(19), gi(20)],
                    confusion: g(21) as u8,
                };
                let p2 = SideExpect {
                    species: f[22].to_string(),
                    hp: g(23),
                    maxhp: g(24),
                    fainted: f[25] == "1",
                    status: parse_status(f[26], g(27)),
                    left: g(28) as usize,
                    boosts: [gi(29), gi(30), gi(31), gi(32), gi(33)],
                    confusion: g(34) as u8,
                };
                let first_mover = f[35].to_string();
                let b = |i: usize| f[i] == "1";
                let out_p1 = [b(36), b(37), b(38), b(39), b(40)];
                let out_p2 = [b(41), b(42), b(43), b(44), b(45)];
                let secondary_landed = b(46);
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect {
                    request: req, force, choice, seed_after, p1, p2, first_mover,
                    out_p1, out_p2, secondary_landed,
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

fn script_from_decisions(case: &RunCase) -> Vec<ScriptDecision> {
    case.decisions
        .iter()
        .map(|dec| ScriptDecision { p1: dec.choice[0], p2: dec.choice[1] })
        .collect()
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
fn secondary_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 6, "expected >=6 scenarios, got {}", meta.len());
    assert!(cases.len() >= 200, "expected the per-seed corpus (>=200 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut status_assertions = 0usize; // per-decision STATUS-variant assertions
    let mut statused_dec_rows = 0usize; // decisions where SOME active is statused
    let mut secondary_landed_rows = 0usize;
    let mut fullpara_rows = 0usize;
    let mut thaw_rows = 0usize;
    let mut flinch_rows = 0usize;
    let mut win_runs = 0usize;
    let mut tie_runs = 0usize;
    // NEW boost/confusion-STATE coverage (this step's additions):
    let mut boost_assertions = 0usize; // per-live-mon boost-stage assertions
    let mut boosted_dec_rows = 0usize; // a live mon has ANY non-zero stage
    let mut self_boost_rows = 0usize; // a live mon has a POSITIVE stage (self-raise)
    let mut confusion_assertions = 0usize; // per-live-mon confusion-counter assertions
    let mut confused_dec_rows = 0usize; // a live mon is confused

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

        let script = script_from_decisions(case);
        let outcome = battle.state_mut().unwrap().run_full_battle(&script, &d);

        assert_eq!(
            outcome.decisions.len(),
            case.decisions.len(),
            "[{}] decision count mismatch (init_seed {}): rust {} vs golden {}",
            case.scen, case.init_seed, outcome.decisions.len(), case.decisions.len()
        );

        for (di, (rec, exp)) in outcome.decisions.iter().zip(case.decisions.iter()).enumerate() {
            // --- request kind + force table ---
            assert!(
                req_eq(&rec.request, exp.request, exp.force),
                "[{}] decision {} request mismatch (init_seed {}): got {:?} exp {:?} force {:?}",
                case.scen, di, case.init_seed, rec.request, exp.request, exp.force
            );

            // --- (a) STATE: species / hp / maxhp / fainted / STATUS + pokemon_left ---
            for (idx, (snap, e, sp)) in [
                (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0])),
                (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1])),
            ] {
                assert_eq!(
                    species_id(sp), species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {}): got {:?} exp {:?}",
                    case.scen, di, idx, case.init_seed, sp, e.species
                );
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, snap.hp, e.hp
                );
                assert_eq!(
                    snap.maxhp, e.maxhp,
                    "[{}] dec {} side {} maxhp mismatch (init_seed {})",
                    case.scen, di, idx, case.init_seed
                );
                assert_eq!(
                    snap.fainted, e.fainted,
                    "[{}] dec {} side {} fainted mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, snap.fainted, e.fainted
                );
                // STATUS is the CRUX of this step: a secondary that wrongly applied/
                // skipped, or an onBeforeMove cure that misfired, diverges HERE. We
                // assert it on every LIVE mon (a fainted mon reports 'fnt' in the sim
                // while the Rust keeps its pre-faint status; both dead).
                if !e.fainted {
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         a secondary applied/skipped wrongly or an onBeforeMove cure misfired.",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    status_assertions += 1;
                    if e.status.is_some() {
                        statused_dec_rows += 1;
                    }

                    // --- BOOST STAGES (the stat-drop / self-boost secondary STATE).
                    //     A wrong stat / wrong stage / wrong target (foe vs self) /
                    //     missing-or-extra apply diverges HERE (not the seed — boost()
                    //     is draw-free). Snapshot boosts are [atk,def,spa,spd,spe,...];
                    //     the golden records the first 5 stat stages.
                    assert_eq!(
                        &snap.boosts[0..5], &e.boosts[..],
                        "[{}] dec {} side {} BOOST mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         a foe stat-drop / self stat-raise secondary applied the wrong \
                         stat/stage/target, or an immunity (Clear Body) gate misfired.",
                        case.scen, di, idx, case.init_seed, &snap.boosts[0..5], e.boosts
                    );
                    boost_assertions += 1;
                    if e.boosts.iter().any(|&v| v != 0) {
                        boosted_dec_rows += 1;
                    }
                    if e.boosts.iter().any(|&v| v > 0) {
                        self_boost_rows += 1;
                    }

                    // --- CONFUSION counter (the Water-Pulse secondary inflicted it AND
                    //     its random(2,6) duration matches). A missing duration draw
                    //     ALSO desyncs the seed; this pins the STATE too.
                    let rust_conf = snap.confusion.unwrap_or(0);
                    assert_eq!(
                        rust_conf, e.confusion,
                        "[{}] dec {} side {} CONFUSION mismatch (init_seed {}): got {} exp {}\n  \
                         the confusion secondary's random(2,6) duration is wrong/missing, \
                         or the onBeforeMove decrement diverged.",
                        case.scen, di, idx, case.init_seed, rust_conf, e.confusion
                    );
                    confusion_assertions += 1;
                    if e.confusion > 0 {
                        confused_dec_rows += 1;
                    }
                }
            }
            assert_eq!(rec.pokemon_left[0], exp.p1.left, "[{}] dec {} p1 left", case.scen, di);
            assert_eq!(rec.pokemon_left[1], exp.p2.left, "[{}] dec {} p2 left", case.scen, di);

            // first mover on a move turn.
            if exp.request == ReqTok::Move {
                let sim_first: Option<usize> = match exp.first_mover.as_str() {
                    "p1" => Some(0),
                    "p2" => Some(1),
                    _ => None,
                };
                if sim_first.is_some() {
                    assert_eq!(
                        rec.first_mover, sim_first,
                        "[{}] dec {} first-mover mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, case.init_seed, rec.first_mover, sim_first
                    );
                }
            }

            // --- (b) PER-DECISION SEED PARITY (the draw-order+count proof). The
            //     onBeforeMove status draw is the NEW LEADING draw (before accuracy);
            //     the secondary random(100) is the NEW TRAILING draw (after damage).
            //     One extra/missing/mis-ordered draw desyncs the LCG here. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a status onBeforeMove draw (para/freeze/confusion) or a secondary \
                 random(100) is mis-ordered/missing/extra (e.g. a secondary drawn on \
                 an immune target, or a draw-free flinch/sleep that wrongly drew). \
                 FIX THE DRAW ORDER, do not loosen the assert.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;

            // --- floor counters (corpus-coverage only; not asserted per-row) ---
            if exp.secondary_landed {
                secondary_landed_rows += 1;
            }
            if exp.out_p1[0] || exp.out_p2[0] {
                fullpara_rows += 1;
            }
            if exp.out_p1[2] || exp.out_p2[2] {
                thaw_rows += 1;
            }
            if exp.out_p1[4] || exp.out_p2[4] {
                flinch_rows += 1;
            }
        }

        // --- WINNER + ended ---
        assert_eq!(
            outcome.ended, case.ended,
            "[{}] ended mismatch (init_seed {}): got {} exp {}",
            case.scen, case.init_seed, outcome.ended, case.ended
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

    // Coverage floors: every NEW branch must actually realize across the corpus, so
    // a green test genuinely exercises the secondary + onBeforeMove draws.
    assert!(seed_assertions >= 2000, "expected the per-decision seed corpus (>=2000), got {seed_assertions}");
    assert!(status_assertions >= 2000, "expected per-decision status assertions (>=2000), got {status_assertions}");
    assert!(statused_dec_rows >= 100, "expected statused-active decision rows (>=100), got {statused_dec_rows}");
    assert!(secondary_landed_rows >= 100, "expected landed-secondary rows (>=100), got {secondary_landed_rows}");
    assert!(fullpara_rows >= 20, "expected full-para onBeforeMove rows (>=20), got {fullpara_rows}");
    assert!(thaw_rows >= 5, "expected freeze-thaw onBeforeMove rows (>=5), got {thaw_rows}");
    assert!(flinch_rows >= 10, "expected flinch rows (>=10), got {flinch_rows}");
    assert!(win_runs >= 50, "expected real game-end WIN runs, got {win_runs}");
    // NEW boost/confusion-STATE floors: every new branch must realize across the corpus.
    assert!(boost_assertions >= 2000, "expected per-decision boost assertions (>=2000), got {boost_assertions}");
    assert!(boosted_dec_rows >= 30, "expected boosted-active decision rows (>=30), got {boosted_dec_rows}");
    assert!(self_boost_rows >= 10, "expected self-boost (positive-stage) rows (>=10), got {self_boost_rows}");
    assert!(confusion_assertions >= 2000, "expected per-decision confusion assertions (>=2000), got {confusion_assertions}");
    assert!(confused_dec_rows >= 20, "expected confused-active decision rows (>=20), got {confused_dec_rows}");

    eprintln!(
        "secondary golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {status_assertions} status assertions ({statused_dec_rows} statused), \
         {secondary_landed_rows} secondary-landed, {fullpara_rows} full-para, {thaw_rows} thaw, \
         {flinch_rows} flinch, {win_runs} wins, {tie_runs} ties; \
         {boost_assertions} boost assertions ({boosted_dec_rows} boosted, {self_boost_rows} self-boost), \
         {confusion_assertions} confusion assertions ({confused_dec_rows} confused)",
        cases.len()
    );
}
