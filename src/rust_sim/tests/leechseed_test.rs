//! LEECH SEED full-battle tests — the per-seed PER-DECISION
//! STATE(+HP+STATUS+SPIKES-LAYERS+LEECH-SEEDED)+SEED+winner differential that proves the
//! NEW Leech Seed mechanic matches Showdown EXACTLY, sustained to GAME-END:
//!
//!   **Leech Seed** (`leechseed`) — a foe-targeting Status move (type Grass, accuracy 90)
//!   that plants the `leechseed` volatile on the FOE; each end-of-turn the seeded mon loses
//!   `floor(maxhp/8)` and the SEEDER's CURRENT active heals the drained amount. DEFERRED
//!   (fail-loud in the engine): a Liquid Ooze target reverses the drain (rare in gen-3 OU),
//!   plus everything already deferred. The model (verified bit-for-bit vs the omniscient
//!   sim's PRNG probe — `harness/probe_leechseed_rng.js`):
//!
//!   THE LEECH SEED MOVE (`volatileStatus:'leechseed'`, `target:'normal'`, type Grass,
//!   accuracy 90):
//!     * ACCURACY — gen-3 Leech Seed is `accuracy: 90` (NOT never-miss), so it DRAWS
//!       `randomChance(90,100)` — it CAN miss. The accuracy roll is drawn UNCONDITIONALLY,
//!       even into a Grass-immune OR already-seeded target (the immunity / fail is reported
//!       only AFTER the accuracy roll — VERIFIED: a splash/splash turn draws 1 [Quick Claw],
//!       a Leech-Seed turn — land / Grass-immune / already-seeded-fail — ALL draw 2).
//!     * GRASS IMMUNITY — a Grass target is IMMUNE (accuracy drawn, then `-immune`, NO
//!       volatile). ALREADY-SEEDED — a 2nd Leech Seed FAILS (accuracy drawn, "did nothing").
//!     * PLANT — on a landed non-immune non-already-seeded hit, the `leechseed` volatile is
//!       added to the foe (DRAW-FREE). `landed` is FALSE (a status moveHit returns undefined
//!       → the in-tryMoveHit Update is skipped).
//!
//!   THE LEECH RESIDUAL (the crux — DRAW-FREE but ORDER-SENSITIVE; gen4-inherited
//!   onResidualOrder 10, onResidualSubOrder 5 — BETWEEN Leftovers sub 4 and the status DoT
//!   sub 6): the seeded mon loses `floor(maxhp/8)` (clamped) and the SEEDER's CURRENT active
//!   HEALS the drained amount (clamped). VERIFIED order `sandstorm[o=8] → leftovers[o=10,s=4]
//!   → leechseed[o=10,s=5] → brn[o=10,s=6]`. A seeder whose active is FAINTED → no drain, no
//!   heal (the whole onResidual returns early).
//!
//!   `leechseed_golden_matches_showdown` — the DIFFERENTIAL gate. For each (scenario, seed)
//!   in `harness/gen_leechseed_golden.js`'s golden (FORMAT gen3customgame), seed a
//!   `BattleState` at the sim's PRNG state at the first decision (`init_seed`), apply the
//!   scenario's one-time STATE-only INJECT (sand/burn/HP, for the residual-order scenario),
//!   run `run_full_battle(script)` WITHOUT re-seeding, and assert per DECISION BOUNDARY: (a)
//!   each side's post-decision active (species/HP/maxhp/fainted/STATUS + counters) + boosts +
//!   confusion + pokemon_left + the SPIKES LAYERS + THE LEECH-SEEDED FLAG (per side) + request
//!   kind + first mover; AND (b) the post-decision PRNG seed == the sim's `seed_after`. PLUS
//!   the final WINNER. The leech drain/heal is proved by the HP column (a wrong subOrder /
//!   amount / seeder-faint gate diverges HP), the SEED proves the accuracy draw model, and
//!   the LEECH-SEEDED flag proves the volatile state (land / immune / already-seeded / clear
//!   on switch-out).
//!
//! The golden TAB format extends the phaze one: it replaces the dragSpecies tail with the two
//! per-side leechSeeded flags → DEC has 50 fields, plus an INJECT line per scenario.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::state::{Status, Weather};
use pokesim::turn::{Choice, RequestKind, ScriptDecision};
use std::collections::BTreeMap;

fn dex() -> Dex {
    Dex::for_gen(3)
}

#[derive(Debug, Clone, Default)]
struct ScenMeta {
    teams: [Option<String>; 2],
    /// The one-time post-start STATE-only injections (sand/burn/HP) for this scenario, in
    /// order. Empty for every scenario but the residual-order one.
    inject: Vec<Inject>,
}

/// One injected board mutation (a STATE-only set, no PRNG — so the seed parity is unaffected
/// by the injection itself). Mirrors the sim harness's `sc.inject` entries.
#[derive(Debug, Clone, Default)]
struct Inject {
    weather: Option<Weather>,
    side: Option<usize>,
    status: Option<Status>,
    hp: Option<u16>,
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
    stage: u8,
    left: usize,
    boosts: [i8; 5],
    confusion: u8,
    spikes: u8,
    /// Whether this active mon is LEECH-SEEDED (the sim's `leechseed` volatile present) at
    /// the decision boundary — the volatile-state proof.
    leech_seeded: bool,
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

fn status_stage(s: Option<Status>) -> u8 {
    match s {
        Some(Status::Sleep(n)) => n,
        Some(Status::Toxic(n)) => n,
        _ => 0,
    }
}

/// Parse the INJECT JSON array into the scenario's [`Inject`] list (weather/side/status/hp).
fn parse_inject(json: &str, ln: usize) -> Vec<Inject> {
    let parsed = Json::parse(json).unwrap_or_else(|e| panic!("bad INJECT json (line {ln}): {e}"));
    let arr = parsed.as_array().unwrap_or_else(|| panic!("INJECT not an array (line {ln})"));
    arr.iter()
        .map(|e| {
            let weather = e.str_at("weather").map(|w| match w {
                "sandstorm" => Weather::Sand,
                "raindance" => Weather::Rain,
                "sunnyday" => Weather::Sun,
                "hail" => Weather::Hail,
                other => panic!("unknown INJECT weather {other:?} (line {ln})"),
            });
            let side = e.get("side").and_then(|s| s.as_f64()).map(|f| f as usize);
            let status = e.str_at("status").map(|s| match s {
                "brn" => Status::Burn,
                "par" => Status::Paralysis,
                "psn" => Status::Poison,
                "tox" => Status::Toxic(0),
                "frz" => Status::Freeze,
                "slp" => Status::Sleep(0),
                other => panic!("unknown INJECT status {other:?} (line {ln})"),
            });
            let hp = e.get("hp").and_then(|h| h.as_f64()).map(|f| f as u16);
            Inject { weather, side, status, hp }
        })
        .collect()
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/leechseed_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing leechseed golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_leechseed_golden.js")
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
            "INJECT" => {
                assert_eq!(f.len(), 3, "INJECT needs 3 fields (line {ln})");
                meta.entry(f[1].to_string()).or_default().inject = parse_inject(f[2], ln);
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
                //   p2 out(...)[41..46)  p1Spikes[46] p2Spikes[47]  p1LeechSeeded[48] p2LeechSeeded[49]
                assert_eq!(f.len(), 50, "DEC needs 50 fields (line {ln}), got {}", f.len());
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
                    stage: g(14) as u8,
                    left: g(15) as usize,
                    boosts: [gi(16), gi(17), gi(18), gi(19), gi(20)],
                    confusion: g(21) as u8,
                    spikes: g(46) as u8,
                    leech_seeded: f[48] == "1",
                };
                let p2 = SideExpect {
                    species: f[22].to_string(),
                    hp: g(23),
                    maxhp: g(24),
                    fainted: f[25] == "1",
                    status: parse_status(f[26], g(27)),
                    stage: g(27) as u8,
                    left: g(28) as usize,
                    boosts: [gi(29), gi(30), gi(31), gi(32), gi(33)],
                    confusion: g(34) as u8,
                    spikes: g(47) as u8,
                    leech_seeded: f[49] == "1",
                };
                let first_mover = f[35].to_string();
                let c = cur.as_mut().unwrap_or_else(|| panic!("DEC before INIT (line {ln})"));
                c.decisions.push(DecExpect { request: req, force, choice, seed_after, p1, p2, first_mover });
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
        // gen3customgame → NO Sleep Clause / SetStatus shuffle (the golden's battle format).
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

/// Apply the scenario's one-time STATE-only INJECT to the freshly-constructed battle (after
/// `start_with_switchins`, before `run_full_battle`) — mirroring the sim harness's post-start
/// inject. PURELY a board set (weather / status / HP), NO PRNG, so the seed parity is intact.
fn apply_inject(battle: &mut Battle, inject: &[Inject]) {
    let state = battle.state_mut().expect("state");
    for inj in inject {
        if let Some(w) = inj.weather {
            state.field.weather = Some(w);
            // Sand Stream-style PERMANENT weather (the harness sets `weatherState.duration=0`).
            state.field.weather_turns = 0;
        }
        if let Some(side) = inj.side {
            let active = state.sides[side].active;
            let mon = &mut state.sides[side].pokemon[active];
            if let Some(st) = inj.status {
                mon.status = Some(st);
            }
            if let Some(hp) = inj.hp {
                mon.hp = hp;
            }
        }
    }
}

#[test]
fn leechseed_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 5, "expected >=5 scenarios, got {}", meta.len());
    assert!(cases.len() >= 350, "expected the per-seed corpus (>=350 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut hp_assertions = 0usize;
    let mut leech_state_assertions = 0usize;
    let mut leech_seeded_rows = 0usize; // a side was leech-seeded at the boundary
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
            "[{}] init prng seed must equal the sim's (switch-ins draw-free)",
            case.scen
        );

        // The one-time STATE-only inject (sand/burn/HP) for the residual-order scenario.
        apply_inject(&mut battle, &m.inject);

        let script = script_from_decisions(case);
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

            for (idx, (snap, e, sp, spikes)) in [
                (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0], rec.spikes[0])),
                (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1], rec.spikes[1])),
            ] {
                assert_eq!(
                    species_id(sp), species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {}): got {:?} exp {:?}",
                    case.scen, di, idx, case.init_seed, sp, e.species
                );
                // HP: the leech drain (floor(maxhp/8)) + the seeder heal + the 4-way residual
                // ORDER all land in HP. A wrong leech subOrder (re-ordering heal/drain), a wrong
                // drain amount, or a wrong seeder-faint gate diverges HERE.
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     the Leech Seed drain/heal (or its residual sub-order vs Leftovers/weather/burn) \
                     is wrong. FIX THE MODEL, do not loosen.",
                    case.scen, di, idx, case.init_seed, snap.hp, e.hp
                );
                hp_assertions += 1;
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
                assert_eq!(
                    spikes, e.spikes,
                    "[{}] dec {} side {} SPIKES-LAYERS mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, spikes, e.spikes
                );

                // THE LEECH-SEEDED VOLATILE STATE: a wrong land (seed didn't plant), a missing
                // Grass-immunity/already-seeded gate (a seed planted when it shouldn't), or a
                // missing switch-out clear all diverge this flag.
                assert_eq!(
                    snap.leech_seeded, e.leech_seeded,
                    "[{}] dec {} side {} LEECH-SEEDED mismatch (init_seed {}): got {} exp {}\n  \
                     the leechseed volatile state is wrong (land / Grass-immune / already-seeded / \
                     switch-out clear). FIX THE MODEL, do not loosen.",
                    case.scen, di, idx, case.init_seed, snap.leech_seeded, e.leech_seeded
                );
                leech_state_assertions += 1;
                if e.leech_seeded {
                    leech_seeded_rows += 1;
                }

                if !e.fainted {
                    assert!(
                        status_variant_eq(snap.status, e.status),
                        "[{}] dec {} side {} STATUS mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, snap.status, e.status
                    );
                    assert_eq!(
                        status_stage(snap.status), e.stage,
                        "[{}] dec {} side {} STATUS-COUNTER mismatch (init_seed {}): got {} exp {}",
                        case.scen, di, idx, case.init_seed, status_stage(snap.status), e.stage
                    );
                    assert_eq!(
                        &snap.boosts[0..5], &e.boosts[..],
                        "[{}] dec {} side {} BOOST mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, idx, case.init_seed, &snap.boosts[0..5], e.boosts
                    );
                    let rust_conf = snap.confusion.unwrap_or(0);
                    assert_eq!(
                        rust_conf, e.confusion,
                        "[{}] dec {} side {} CONFUSION mismatch (init_seed {}): got {} exp {}",
                        case.scen, di, idx, case.init_seed, rust_conf, e.confusion
                    );
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
                        "[{}] dec {} FIRST-MOVER mismatch (init_seed {}): got {:?} exp {:?}",
                        case.scen, di, case.init_seed, rec.first_mover, sim_first
                    );
                }
            }

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). A Leech Seed move
            //     draws its accuracy roll (`randomChance(90,100)` — gen-3 Leech Seed is acc 90,
            //     NOT never-miss) — UNCONDITIONALLY, even into a Grass-immune or already-seeded
            //     target. The leech RESIDUAL drain/heal is DRAW-FREE. A wrong draw model
            //     (skipping the accuracy roll, drawing it twice, or a stray residual draw)
            //     desyncs the LCG HERE. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 the Leech Seed draw model is wrong (the accuracy roll must be drawn once, \
                 unconditionally, and the leech residual must be draw-free). FIX THE DRAW MODEL.",
                case.scen, di, case.init_seed, rec.seed_after, exp.seed_after
            );
            seed_assertions += 1;
            dec_assertions += 1;
        }

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

    // Coverage floors: the leech mechanic must actually realize across the corpus.
    assert!(seed_assertions >= 1500, "expected the per-decision seed corpus (>=1500), got {seed_assertions}");
    assert!(hp_assertions >= 1500, "expected per-decision HP assertions (>=1500), got {hp_assertions}");
    assert!(leech_state_assertions >= 1500, "expected per-decision leech-state assertions (>=1500), got {leech_state_assertions}");
    assert!(leech_seeded_rows >= 100, "expected leech-seeded STATE rows (>=100), got {leech_seeded_rows}");
    assert!(win_runs >= 100, "expected real game-end WIN runs (>=100), got {win_runs}");

    eprintln!(
        "leechseed golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {hp_assertions} HP assertions, {leech_state_assertions} leech-state assertions \
         ({leech_seeded_rows} leech-seeded rows), {win_runs} wins, {tie_runs} ties",
        cases.len()
    );
}
