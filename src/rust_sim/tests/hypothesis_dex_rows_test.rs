//! hypothesis_dex_rows_test.rs — X5's REAL-STATE CROSS-CHECK of the hypothesis row
//! (`gen3_x5_dex_rows_v1`; `designs/endstate/design_x5_belief_tokens.md` §3.4, build unit U1).
//!
//! `encoder::hypothesis::hypothesis_slot(s)` builds a SYNTHETIC input ("species s present,
//! unrevealed set, full HP, no status") that the encoder never meets on its own, so this file holds it
//! to REALITY: over a fixed, seeded set of real Rust-core battles, at each opponent mon's FIRST
//! appearance (the first decision of the viewing side at which the mon is in its reading), the slot
//! the encoder writes for it — on the PARSE chain, the road `sim_bridge`'s `core_obs` ships to
//! training — must equal the synthetic row BYTE for byte on every cell except the DECLARED on-field
//! blocks of `encoder::hypothesis::CELLS` (HP, status, its counters and sleep belief, recency, last
//! action, the active flag), plus an item / ability / move block the FIELD revealed by the real row's
//! own flag. No tolerance (standing rule 8).
//!
//! Two battle sets:
//! * COVERAGE — every BASE-FORM species of `data/pokemon/gen3_species.json` (one per national-dex
//!   num, the table's rows), six to a team in num order, team 2k against 2k + 1: each mon carries its
//!   first pokedex ability and Leftovers, and Roar / Toxic / Seismic Toss / Protect, so mons are
//!   phazed in, damaged (Leftovers then reveals itself) and statused; a seeded random policy that
//!   prefers switching. Every species must appear.
//! * REALISM — every team pair of the bridge corpus (real pool teams), seeded random play.

use std::collections::{BTreeMap, BTreeSet};

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::encoder::hypothesis::{cross_check, hypothesis_slot};
use pokesim::encoder::layout::{OFFSET_OPP_TEAM, POKEMON_FULL_DIM};
use pokesim::encoder::OBS_DIM;
use pokesim::json::Json;
use pokesim::present;
use pokesim::trackers::clock::ClockConfig;
use pokesim::version::BattleVersion;

const NAMES: [&str; 2] = ["Alice", "Bob"];
const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/bridge_corpus");
const SPECIES_JSON: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/pokemon/gen3_species.json");
const COVERAGE_MOVES: &str = "roar,toxic,seismictoss,protect";

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n as u64) as usize
    }
}

/// Every first appearance one battle set produced, and what the cross-check did with it.
#[derive(Default)]
struct Census {
    battles: usize,
    decisions: usize,
    appearances: usize,
    species: BTreeSet<String>,
    compared_cells: usize,
    revealed: BTreeMap<&'static str, usize>,
    failures: Vec<String>,
}

impl Census {
    fn report(&self, set: &str) -> String {
        format!(
            "{set}: {} battles, {} decisions, {} first appearances of {} species, {} cells compared, revealed-on-field skips {:?}",
            self.battles, self.decisions, self.appearances, self.species.len(), self.compared_cells, self.revealed
        )
    }
}

/// Play one battle on the library road (a live session + one PARSE chain per side, trackers on,
/// `note_choice` before each CHOOSE) and cross-check every first appearance.
fn play(label: &str, teams: [String; 2], seed: String, rng_seed: u64, prefer_switch: bool, cap: usize, census: &mut Census) {
    let dex = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3ou".into(),
        seed: Some(seed),
        p1: PlayerOptions { name: NAMES[0].into(), team: PackedTeam(teams[0].clone()) },
        p2: PlayerOptions { name: NAMES[1].into(), team: PackedTeam(teams[1].clone()) },
    };
    let mut sess = BridgeSession::new_construct_turn0(&opts, &dex).unwrap_or_else(|e| panic!("{label}: session: {e:?}"));
    let cfg = ClockConfig::default();
    let mut chains =
        [0, 1].map(|s| Some(BattleVersion::parse_root_with(s, NAMES[s], Some(&teams[s]), Some(cfg)).expect("root")));
    let mut seen: [BTreeSet<String>; 2] = [BTreeSet::new(), BTreeSet::new()];
    let mut rng = Rng(rng_seed);
    let mut pending: [Option<Vec<String>>; 2] = [None, None];
    census.battles += 1;
    let mut cmds = 0usize;
    let mut sent: Option<usize> = None;
    loop {
        // advance both chains over what each side was newly shipped; check + collect its decision
        for side in 0..2 {
            let c = chains[side].take().unwrap();
            let from = c.stream(side).unwrap().lines;
            let shipped = &sess.side_lines(side)[from..];
            // a legal-looking token can still be REJECTED (a hidden trapper): the decision stays open
            let rejected = shipped.iter().any(|l| l.starts_with("|error|"));
            let next = c.parse_advance(shipped).expect("parse");
            if next.decision(side).is_some() {
                census.decisions += 1;
                let mut row = [0.0f32; OBS_DIM];
                next.encode(side, &mut row).expect("encode");
                let reading = &next.stream(side).unwrap().board_reading;
                for (i, (_, mon)) in reading.opp.iter().enumerate() {
                    if !seen[side].insert(mon.species.clone()) {
                        continue;
                    }
                    census.appearances += 1;
                    census.species.insert(mon.species.clone());
                    let lo = OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM;
                    let syn = hypothesis_slot(&mon.species).unwrap_or_else(|e| panic!("{label}: {}: {e:?}", mon.species));
                    match cross_check(&row[lo..lo + POKEMON_FULL_DIM], &syn) {
                        Ok(c) => {
                            census.compared_cells += c.cells;
                            for b in c.revealed {
                                *census.revealed.entry(b).or_default() += 1;
                            }
                        }
                        Err(e) => census.failures.push(format!("{label} p{} turn {} {}: {e:?}", side + 1, reading.turn, mon.species)),
                    }
                }
                let legal = next.legal(side).expect("legal");
                pending[side] = Some(present::choice_tokens(reading, &legal).expect("tokens").into_iter().map(|(_, t)| t).collect());
            } else if sent == Some(side) && !rejected {
                // the side's choice was taken: its decision closes until the next request
                pending[side] = None;
            }
            chains[side] = Some(next);
        }
        if sess.is_ended() {
            break;
        }
        let all_seen = seen.iter().all(|s| s.len() == 6);
        let Some(side) = (0..2).find(|&s| pending[s].is_some()) else { panic!("{label}: stuck with no decision open") };
        if cmds >= cap || (prefer_switch && all_seen) {
            sess.forfeit(side);
            break;
        }
        let toks = pending[side].clone().unwrap();
        let switches: Vec<&String> = toks.iter().filter(|t| t.starts_with("switch")).collect();
        let tok = if prefer_switch && !switches.is_empty() && rng.below(2) == 0 {
            switches[rng.below(switches.len())].clone()
        } else {
            toks[rng.below(toks.len())].clone()
        };
        chains[side].as_mut().unwrap().note_choice(side, &tok);
        sess.feed_cmd(Cmd { side, choice: parse_choice(&tok).expect("choice") }, &dex);
        assert!(sess.fatal().is_none(), "{label}: bridge fatal: {:?}", sess.fatal());
        cmds += 1;
        sent = Some(side);
    }
}

/// Every base-form species (no `baseSpecies`), `(num, display name)`, in num order.
fn base_forms() -> Vec<(i64, String, String)> {
    let text = std::fs::read_to_string(SPECIES_JSON).expect("species json");
    let root = Json::parse(&text).expect("parse");
    let mut out: Vec<(i64, String, String)> = root
        .as_object()
        .unwrap()
        .iter()
        .filter(|(_, v)| v.get("baseSpecies").is_none())
        .map(|(id, v)| (v.get("num").and_then(Json::as_f64).unwrap() as i64, id.clone(), v.str_at("name").unwrap().to_string()))
        .collect();
    out.sort();
    out
}

fn coverage_team(mons: &[(i64, String, String)]) -> String {
    mons.iter()
        .map(|(_, id, name)| {
            let ability = present::dex::species(id).expect("pokedex").abilities[0];
            format!("{name}||leftovers|{ability}|{COVERAGE_MOVES}|Serious||||||")
        })
        .collect::<Vec<_>>()
        .join("]")
}

/// Every (p1, p2) team pair of the bridge corpus, per scenario, in file order, deduplicated.
fn corpus_pairs() -> Vec<[String; 2]> {
    let mut files: Vec<_> = std::fs::read_dir(CORPUS).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
    files.sort();
    let mut out: Vec<[String; 2]> = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        let mut t: [Option<String>; 2] = [None, None];
        let flush = |t: &mut [Option<String>; 2], out: &mut Vec<[String; 2]>| {
            if let [Some(a), Some(b)] = t.clone() {
                if !out.contains(&[a.clone(), b.clone()]) {
                    out.push([a, b]);
                }
            }
            *t = [None, None];
        };
        for l in text.lines() {
            if l.starts_with("SCEN\t") {
                flush(&mut t, &mut out);
            } else if l.starts_with("TEAM\t") {
                let f: Vec<&str> = l.split('\t').collect();
                let s = if f[2] == "p1" { 0 } else { 1 };
                t[s].get_or_insert_with(|| f[3].to_string());
            }
        }
        flush(&mut t, &mut out);
    }
    out
}

#[test]
fn every_base_form_species_first_appears_exactly_as_its_hypothesis_row() {
    let forms = base_forms();
    assert_eq!(forms.len(), 386, "one base form per national-dex num");
    let teams: Vec<String> = forms.chunks(6).map(coverage_team).collect();
    let mut census = Census::default();
    for k in 0..teams.len().div_ceil(2) {
        let (a, b) = (2 * k, (2 * k + 1).min(teams.len() - 1));
        let seed = format!("{},{},{},{}", 31 + k, 7 * k + 5, 19, 23 + k);
        play(&format!("coverage{k}"), [teams[a].clone(), teams[b].clone()], seed, 0x5EED_0000 + k as u64, true, 1200, &mut census);
    }
    eprintln!("{}", census.report("COVERAGE"));
    assert!(census.failures.is_empty(), "{} first appearances differ from their hypothesis row:\n{}", census.failures.len(), census.failures.join("\n"));
    let missing: Vec<&String> = forms.iter().map(|(_, id, _)| id).filter(|id| !census.species.contains(*id)).collect();
    assert!(missing.is_empty(), "{} base-form species never appeared: {missing:?}", missing.len());
    // non-vacuity: the field revealed items and abilities, and the comparison skipped them by flag
    assert!(census.revealed.get("item").copied().unwrap_or(0) > 0, "no Leftovers reveal at a first appearance — vacuous");
    assert!(census.revealed.get("ability").copied().unwrap_or(0) > 0, "no ability announced on entry — vacuous");
}

#[test]
fn every_corpus_first_appearance_is_its_hypothesis_row() {
    let pairs = corpus_pairs();
    assert!(pairs.len() >= 20, "the corpus holds {} team pairs", pairs.len());
    let mut census = Census::default();
    for (i, p) in pairs.into_iter().enumerate() {
        let seed = format!("{},{},{},{}", 101 + i, 7 * i + 3, 13, 17 + i);
        play(&format!("corpus{i}"), p, seed, 0xC0FFEE + i as u64, false, 1500, &mut census);
    }
    eprintln!("{}", census.report("REALISM"));
    assert!(census.failures.is_empty(), "{} first appearances differ from their hypothesis row:\n{}", census.failures.len(), census.failures.join("\n"));
    assert!(census.appearances >= 200, "only {} first appearances — vacuous", census.appearances);
}
