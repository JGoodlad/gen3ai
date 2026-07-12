//! MOVE-COVERAGE BATCH 1 full-battle test (`gen3_move_coverage_batch1_v1`) — the per-seed
//! PER-DECISION STATE(+HP+STATUS+BOOSTS+SPIKES+LEECH+ITEM)+SEED+winner differential that
//! proves the five DRAW-FREE post-hit effect classes match Showdown EXACTLY, to GAME-END:
//!
//!   * **RECOIL** — Double-Edge `recoil:[1,3]` (Take Down / Submission `[1,4]`): the USER
//!     takes `max(floor(dmgDealt·num/den),1)` HP after a landed hit; Rock Head negates;
//!     fires behind a substitute too. Proved by the USER's HP column.
//!   * **DRAIN** — Giga Drain `drain:[1,2]`: the USER heals a fraction of the damage dealt;
//!     heal-at-full fails; fires behind a sub. Proved by the USER's HP column.
//!   * **SELF-DROP** — Overheat (self −2 SpA) / Superpower (self −1 Atk/−1 Def): the
//!     `move.self.boosts` on the USER, ±6 clamp. gen3 `selfDrops` DRAWS ONE `random(100)`
//!     (the `secondaryRoll`) then applies UNCONDITIONALLY (`self.chance === undefined`), so it
//!     is NOT draw-free — the reason the port's Overheat/Superpower were never seed-verified.
//!     Proved by the BOOST columns + the SEED (the `random(100)` must be drawn once).
//!   * **ITEM REMOVAL** — Knock Off removes the target's item (gen3 no dmg boost); Thief /
//!     Covet STEAL iff the attacker is itemless; Sticky Hold BLOCKS. onAfterHit → only when
//!     the MON was damaged (behind a sub the item stays). Proved by the ITEM columns.
//!   * **RAPID SPIN** — clears the USER's own Spikes + Leech Seed (onAfterHit +
//!     onAfterSubDamage, so it clears behind a sub too). Proved by the SPIKES + LEECH columns.
//!
//! FOUR are draw-free (recoil/drain/item/rapid-spin) + SELF-DROP draws ONE `random(100)`
//! (probe `harness/probe_batch1_movecoverage.js` + a per-call-site PRNG trace), so the SEED
//! parity is the draw-model proof (a stray/missing draw — incl. the self-drop `random(100)` —
//! desyncs the LCG) and the STATE columns are the effect proof (a wrong recoil/drain/self-drop/
//! item/hazard-clear diverges HP / boosts / item / spikes / leech). The golden TAB format
//! extends the leechseed one: it drops the per-side leechSeeded tail into the STATE tail and
//! appends per-side ITEM columns → DEC has 42 fields, plus an INJECT line per scenario
//! (spikes/leech/hp).

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
    inject: Vec<Inject>,
}

/// One injected board mutation (a STATE-only set, no PRNG — so seed parity is unaffected).
#[derive(Debug, Clone, Default)]
struct Inject {
    weather: Option<Weather>,
    side: Option<usize>,
    status: Option<Status>,
    hp: Option<u16>,
    spikes: Option<u8>,
    leechseed: bool,
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
    leech_seeded: bool,
    /// Whether this active mon HOLDS an item at the boundary (`MonState::item` non-empty).
    /// The golden serializes the item id; we track presence (the exact item transfer is
    /// byte-verified by the protocol test) — Knock Off removes (true→false), Thief steals
    /// (attacker false→true / target true→false), Sticky Hold keeps it (true).
    item_held: bool,
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
            let spikes = e.get("spikes").and_then(|s| s.as_f64()).map(|f| f as u8);
            let leechseed = e.get("leechseed").and_then(|b| b.as_bool()).unwrap_or(false);
            Inject { weather, side, status, hp, spikes, leechseed }
        })
        .collect()
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<RunCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/movecoverage_batch1_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!(
            "missing batch1 golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_movecoverage_batch1_golden.js"
        )
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
                //   p1Spikes[36] p2Spikes[37]  p1Leech[38] p2Leech[39]  p1Item[40] p2Item[41]
                assert_eq!(f.len(), 42, "DEC needs 42 fields (line {ln}), got {}", f.len());
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
                    spikes: g(36) as u8,
                    leech_seeded: f[38] == "1",
                    item_held: f[40] != "-",
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
                    spikes: g(37) as u8,
                    leech_seeded: f[39] == "1",
                    item_held: f[41] != "-",
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

/// Apply the scenario's one-time STATE-only INJECT (weather/status/HP/spikes/leech) to the
/// freshly-constructed battle — mirroring the sim harness's post-start inject. NO PRNG.
fn apply_inject(battle: &mut Battle, inject: &[Inject]) {
    let state = battle.state_mut().expect("state");
    for inj in inject {
        if let Some(w) = inj.weather {
            state.field.weather = Some(w);
            state.field.weather_turns = 0;
        }
        if let Some(spikes) = inj.spikes {
            if let Some(side) = inj.side {
                state.sides[side].spikes = spikes;
            }
        }
        if let Some(side) = inj.side {
            let active = state.sides[side].active;
            if inj.leechseed {
                // Seed this side's active from the OTHER side (the seeder).
                state.sides[side].pokemon[active].leech_seed = Some(1 - side);
            }
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
fn movecoverage_batch1_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 10, "expected >=10 scenarios, got {}", meta.len());
    assert!(cases.len() >= 800, "expected the per-seed corpus (>=800 runs), got {}", cases.len());

    let mut dec_assertions = 0usize;
    let mut seed_assertions = 0usize;
    let mut hp_assertions = 0usize;
    let mut item_assertions = 0usize;
    let mut item_held_rows = 0usize; // a side HOLDS an item at the boundary (Sticky Hold / Thief-hold / Thief-gain)
    let mut spikes_cleared_rows = 0usize; // Rapid Spin took a side from >0 spikes to 0
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

        apply_inject(&mut battle, &m.inject);

        // Did this scenario inject Spikes on a side (the Rapid-Spin clear scenarios)? If so,
        // the FIRST decision must show 0 spikes on that side — proving Rapid Spin cleared the
        // injected layers (a non-clearing port would show the injected count → the per-decision
        // spikes assert would FAIL). Counted below as the clear coverage.
        let injected_spikes: [bool; 2] = {
            let mut out = [false; 2];
            for inj in &m.inject {
                if let (Some(side), Some(sp)) = (inj.side, inj.spikes) {
                    if sp > 0 {
                        out[side] = true;
                    }
                }
            }
            out
        };

        let script = script_from_decisions(case);
        let outcome = battle.state_mut().unwrap().run_full_battle(&script, &d);

        assert_eq!(
            outcome.decisions.len(),
            case.decisions.len(),
            "[{}] decision count mismatch (init_seed {}): rust {} vs golden {}",
            case.scen, case.init_seed, outcome.decisions.len(), case.decisions.len()
        );

        let mut prev_spikes = [0u8; 2];
        for (di, (rec, exp)) in outcome.decisions.iter().zip(case.decisions.iter()).enumerate() {
            assert!(
                req_eq(&rec.request, exp.request, exp.force),
                "[{}] decision {} request mismatch (init_seed {}): got {:?} exp {:?} force {:?}",
                case.scen, di, case.init_seed, rec.request, exp.request, exp.force
            );

            for (idx, (snap, e, sp, spikes, item_held)) in [
                (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0], rec.spikes[0], rec.active[0].item_held)),
                (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1], rec.spikes[1], rec.active[1].item_held)),
            ] {
                assert_eq!(
                    species_id(sp), species_id(&e.species),
                    "[{}] dec {} side {} active species mismatch (init_seed {}): got {:?} exp {:?}",
                    case.scen, di, idx, case.init_seed, sp, e.species
                );
                // HP: the RECOIL (user takes floor(dmg/den)) + the DRAIN heal land HERE. A
                // wrong recoil/drain amount or a wrong Rock-Head-negation diverges the HP.
                assert_eq!(
                    snap.hp, e.hp,
                    "[{}] dec {} side {} HP mismatch (init_seed {}): got {} exp {}\n  \
                     the recoil/drain amount is wrong. FIX THE MODEL, do not loosen.",
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
                // SPIKES: Rapid Spin clears the USER's own side (>0 → 0). A wrong gate (not
                // clearing behind a sub, or clearing the wrong side) diverges HERE.
                assert_eq!(
                    spikes, e.spikes,
                    "[{}] dec {} side {} SPIKES-LAYERS mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, spikes, e.spikes
                );
                // Rapid Spin cleared the injected Spikes: the scenario injected >0 on this side
                // and the FIRST decision shows 0 (the port must have cleared them). Also count a
                // later >0→0 transition (a mid-battle spin), if any.
                if di == 0 && injected_spikes[idx] && spikes == 0 {
                    spikes_cleared_rows += 1;
                }
                if prev_spikes[idx] > 0 && spikes == 0 {
                    spikes_cleared_rows += 1;
                }
                prev_spikes[idx] = spikes;

                // LEECH: Rapid Spin clears the USER's own Leech Seed.
                assert_eq!(
                    snap.leech_seeded, e.leech_seeded,
                    "[{}] dec {} side {} LEECH-SEEDED mismatch (init_seed {}): got {} exp {}",
                    case.scen, di, idx, case.init_seed, snap.leech_seeded, e.leech_seeded
                );

                // ITEM: Knock Off removes the target's item; Thief/Covet steal (attacker
                // gains); Sticky Hold keeps it. A wrong gate (removing behind a sub, stealing
                // when the attacker holds an item, or ignoring Sticky Hold) diverges HERE.
                assert_eq!(
                    item_held, e.item_held,
                    "[{}] dec {} side {} ITEM-HELD mismatch (init_seed {}): got {} exp {}\n  \
                     the item removal/steal/Sticky-Hold gate is wrong. FIX THE MODEL, do not loosen.",
                    case.scen, di, idx, case.init_seed, item_held, e.item_held
                );
                item_assertions += 1;
                if e.item_held {
                    item_held_rows += 1;
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
                    // BOOST: the SELF-DROP (Overheat −2 SpA / Superpower −1 Atk/−1 Def) lands
                    // HERE — clamped to the −6 floor. A missing/wrong self-drop diverges.
                    assert_eq!(
                        &snap.boosts[0..5], &e.boosts[..],
                        "[{}] dec {} side {} BOOST mismatch (init_seed {}): got {:?} exp {:?}\n  \
                         the self stat-drop is wrong. FIX THE MODEL, do not loosen.",
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

            // --- PER-DECISION SEED PARITY (the draw-order+count proof). FOUR of the batch-1
            //     effects are DRAW-FREE — recoil (`this.damage`), drain (`this.heal`), item
            //     removal (`TakeItem`/`takeItem`), rapid spin (side/volatile removes) consume
            //     NO PRNG — while SELF-DROP DRAWS ONE `random(100)` (the gen3 `selfDrops`
            //     `secondaryRoll`, applied unconditionally). A stray/missing draw — incl. a
            //     skipped self-drop `random(100)` — desyncs HERE. ---
            assert_eq!(
                rec.seed_after, exp.seed_after,
                "[{}] dec {} SEED mismatch (init_seed {}): got {} exp {}\n  \
                 a batch-1 post-hit effect drew (or skipped) a PRNG call — recoil/drain/item/\
                 rapid-spin are draw-free; the self-drop draws ONE random(100). FIX THE DRAW MODEL.",
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

    // Coverage floors: each mechanic must actually realize across the corpus.
    assert!(seed_assertions >= 800, "expected the per-decision seed corpus (>=800), got {seed_assertions}");
    assert!(hp_assertions >= 800, "expected per-decision HP assertions (>=800), got {hp_assertions}");
    assert!(item_assertions >= 800, "expected per-decision item assertions (>=800), got {item_assertions}");
    assert!(item_held_rows >= 40, "expected item-held rows (Sticky Hold / Thief hold+gain, >=40), got {item_held_rows}");
    assert!(spikes_cleared_rows >= 40, "expected Rapid-Spin spikes-cleared rows (>=40), got {spikes_cleared_rows}");
    assert!(win_runs >= 50, "expected real game-end WIN runs (>=50), got {win_runs}");

    eprintln!(
        "movecoverage batch1 golden: {} runs, {dec_assertions} STATE rows, {seed_assertions} seed assertions, \
         {hp_assertions} HP assertions, {item_assertions} item assertions ({item_held_rows} item-held rows, \
         {spikes_cleared_rows} spikes-cleared rows), {win_runs} wins, {tie_runs} ties",
        cases.len()
    );
}
