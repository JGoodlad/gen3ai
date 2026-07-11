//! Battle-state construction tests:
//!   - `state_golden_matches_showdown` — differential vs the REAL Showdown sim's
//!     CONSTRUCTION-time per-mon state (golden from `harness/gen_state_golden.js`):
//!     `Battle::start` from the SAME two packed teams + seed reproduces every
//!     mon's species id, level, maxhp, the six stats (the sim's OWN
//!     `storedStats` + `maxhp`), and the gen-3-singles lead (active slot 0).
//!     We do NOT assert boosts or weather — those are switch-in-EVENT outputs
//!     (Sand Stream / Intimidate fire in the golden's started battle but the
//!     construct-only Rust state, correctly, has clean boosts + no weather).
//!   - `state_smoke` — authoritative hand-checks independent of the golden:
//!     hp == maxhp == stats[0], boosts all 0, status None, fainted false, lead
//!     == slot 0, turn 0, no weather, plus the crash-don't-drop Err paths.
//!
//! Mirrors the structure of `tests/stats_test.rs`.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;

fn dex() -> Dex {
    Dex::for_gen(3)
}

/// One parsed golden mon row.
struct GoldenMon {
    side: usize, // 0 = p1, 1 = p2
    slot: usize,
    species: String,
    level: u8,
    maxhp: u16,
    stats: [u16; 6],
    active: bool,
}

/// The parsed golden: the two packed teams, the seed, and every mon row.
struct Golden {
    teams: [String; 2],
    seed: String,
    mons: Vec<GoldenMon>,
}

fn side_index(tag: &str, ln: usize) -> usize {
    match tag {
        "p1" => 0,
        "p2" => 1,
        other => panic!("bad side tag {other:?} (line {ln})"),
    }
}

fn parse_golden() -> Golden {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/state_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing state golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_state_golden.js")
    });

    let mut teams: [Option<String>; 2] = [None, None];
    let mut seed: Option<String> = None;
    let mut mons = Vec::new();

    for (i, line) in data.lines().enumerate() {
        let ln = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "TEAM" => {
                assert_eq!(f.len(), 3, "TEAM needs <side> <packed> (line {ln})");
                teams[side_index(f[1], ln)] = Some(f[2].to_string());
            }
            "SEED" => {
                assert_eq!(f.len(), 2, "SEED needs <m,n,o,p> (line {ln})");
                seed = Some(f[1].to_string());
            }
            "MON" => {
                // MON <side> <slot> <speciesid> <level> <maxhp> <hp> <atk> <def> <spa> <spd> <spe> <active>
                assert_eq!(f.len(), 13, "MON needs 13 fields (line {ln})");
                let num = |k: usize| -> u32 {
                    f[k].parse().unwrap_or_else(|e| panic!("bad number {:?} (line {ln}): {e}", f[k]))
                };
                let stats = [
                    num(6) as u16,
                    num(7) as u16,
                    num(8) as u16,
                    num(9) as u16,
                    num(10) as u16,
                    num(11) as u16,
                ];
                mons.push(GoldenMon {
                    side: side_index(f[1], ln),
                    slot: num(2) as usize,
                    species: f[3].to_string(),
                    level: num(4) as u8,
                    maxhp: num(5) as u16,
                    stats,
                    active: match f[12] {
                        "0" => false,
                        "1" => true,
                        o => panic!("bad active flag {o:?} (line {ln})"),
                    },
                });
            }
            other => panic!("unknown record {other:?} (line {ln})"),
        }
    }

    Golden {
        teams: [
            teams[0].clone().expect("golden missing TEAM p1"),
            teams[1].clone().expect("golden missing TEAM p2"),
        ],
        seed: seed.expect("golden missing SEED"),
        mons,
    }
}

/// Build `Battle::start` from the golden's exact packed teams + seed.
fn start_from_golden(g: &Golden, d: &Dex) -> Battle {
    let opts = BattleOptions {
        format_id: "gen3ou".to_string(),
        seed: Some(format!("[{}]", g.seed)), // the `>start` bracketed-array form
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(g.teams[0].clone()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(g.teams[1].clone()) },
    };
    Battle::start(&opts, d).expect("Battle::start failed")
}

#[test]
fn state_golden_matches_showdown() {
    let d = dex();
    let g = parse_golden();
    let battle = start_from_golden(&g, &d);
    let state = battle.state().expect("battle constructed");

    assert_eq!(state.turn, 0, "construction turn is 0 (start action not run)");
    assert!(state.field.weather.is_none(), "no weather at construction");

    assert_eq!(g.mons.len(), 12, "expected 12 mons (2 teams of 6)");
    let mut checked = 0usize;
    for gm in &g.mons {
        let side = state.side(gm.side);
        let mon = &side.pokemon[gm.slot];

        assert_eq!(
            mon.species_id, gm.species,
            "species mismatch side {} slot {}: got {:?} exp {:?}",
            gm.side, gm.slot, mon.species_id, gm.species
        );
        assert_eq!(mon.level, gm.level, "level mismatch side {} slot {}", gm.side, gm.slot);
        assert_eq!(mon.maxhp, gm.maxhp, "maxhp mismatch side {} slot {}", gm.side, gm.slot);
        assert_eq!(
            mon.stats, gm.stats,
            "stats mismatch side {} slot {} ({}): got {:?} exp {:?}",
            gm.side, gm.slot, gm.species, mon.stats, gm.stats
        );
        // hp == maxhp == stats[0] at construction.
        assert_eq!(mon.hp, mon.maxhp, "hp != maxhp side {} slot {}", gm.side, gm.slot);
        assert_eq!(mon.maxhp, mon.stats[0], "maxhp != stats[0] side {} slot {}", gm.side, gm.slot);
        // position == slot.
        assert_eq!(mon.position, gm.slot, "position != slot side {} slot {}", gm.side, gm.slot);

        // The lead flag: the golden marks slot-0 (active[0] == pokemon[0]) as the
        // lead. We compare it to our STRUCTURAL lead (side.active index), not to a
        // switch-in we never run.
        let is_lead = side.active == gm.slot;
        assert_eq!(
            is_lead, gm.active,
            "active/lead mismatch side {} slot {}",
            gm.side, gm.slot
        );

        checked += 1;
    }
    // Both leads are slot 0 in gen3 singles.
    assert_eq!(state.side(0).active, 0, "p1 lead is slot 0");
    assert_eq!(state.side(1).active, 0, "p2 lead is slot 0");
    eprintln!("state golden: {checked} mons matched (construction-time)");
}

#[test]
fn state_smoke() {
    let d = dex();

    // A minimal two-mon-per-side battle, packed inline (lowercase ids — the
    // poke-env producer form the codec ingests). Tyranitar lead (Sand Stream is
    // event-driven, so we DON'T expect weather/boosts here).
    let p1 = "Tyranitar||leftovers|sandstream|rockslide,earthquake,crunch,dragondance|Adamant|252,252,,,4,|||||]Blissey||leftovers|naturalcure|softboiled,seismictoss,toxic,aromatherapy|Calm|252,,,,252,4|||||";
    let p2 = "Skarmory||leftovers|keeneye|spikes,roar,drillpeck,protect|Impish|252,,252,,4,|||||]Gengar||leftovers|levitate|thunderbolt,icepunch,hiddenpowerfire,willowisp|Timid|4,,,252,,252|||||";

    let opts = BattleOptions {
        format_id: "gen3ou".to_string(),
        seed: Some("[1,2,3,4]".to_string()),
        p1: PlayerOptions { name: "A".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "B".to_string(), team: PackedTeam(p2.to_string()) },
    };
    let battle = Battle::start(&opts, &d).expect("start");
    let state = battle.state().unwrap();

    // Construction invariants.
    assert_eq!(state.turn, 0);
    assert_eq!(battle.turn(), 0);
    assert!(state.field.weather.is_none(), "no weather built (Sand Stream is an event)");
    assert_eq!(state.gen, 3);

    for si in 0..2 {
        let side = state.side(si);
        assert_eq!(side.pokemon.len(), 2);
        assert_eq!(side.pokemon_left, 2, "pokemon_left == team size");
        assert_eq!(side.active, 0, "lead is slot 0 (gen3 singles)");
        for (slot, mon) in side.pokemon.iter().enumerate() {
            assert_eq!(mon.position, slot);
            assert_eq!(mon.hp, mon.maxhp, "hp == maxhp at start");
            assert_eq!(mon.maxhp, mon.stats[0], "maxhp == stats[0]");
            assert!(mon.status.is_none(), "no status at construction");
            assert_eq!(mon.boosts, [0i8; 7], "boosts all 0 (Intimidate is an event)");
            assert!(!mon.fainted, "not fainted at start");
            assert!(mon.maxhp > 0, "maxhp computed");
        }
    }

    // Specific value, hand-derived from the gen-3 formula (independent of any
    // golden) for THIS exact spread — Tyranitar base 100/134/110/95/100/61,
    // Adamant (+Atk/-SpA), 252 HP / 252 Atk / 4 SpD, 31 IVs, L100:
    //   HP  = floor((2*100 + 31 + 252/4 + 100) + 10)      = 404
    //   Atk = floor((floor(2*134+31+63) + 5) * 110/100)   = floor(367*1.1) = 403
    //   Def = (2*110 + 31) + 5                              = 256
    //   SpA = floor(((2*95+31) + 5) * 90/100)              = floor(226*0.9) = 203
    //   SpD = (2*100 + 31 + 4/4) + 5                        = 237   (the 4 SpD EVs add 1; cf. 236 at 0 EV)
    //   Spe = (2*61 + 31) + 5                               = 158
    let ttar = state.side(0).active();
    assert_eq!(ttar.species_id, "tyranitar");
    assert_eq!(ttar.stats, [404, 403, 256, 203, 237, 158], "Tyranitar Adamant 252/252/4-SpD stats");
    assert_eq!(ttar.hp, 404);

    // Crash-don't-drop: an unknown species in a team aborts start with Err.
    let bad = "NotAMon||leftovers||tackle|Hardy|||||||";
    let bad_opts = BattleOptions {
        format_id: "gen3ou".to_string(),
        seed: None,
        p1: PlayerOptions { name: "A".to_string(), team: PackedTeam(bad.to_string()) },
        p2: PlayerOptions { name: "B".to_string(), team: PackedTeam(p2.to_string()) },
    };
    assert!(Battle::start(&bad_opts, &d).is_err(), "unknown species team -> Err");

    // An empty team also Errs (no lead).
    let empty_opts = BattleOptions {
        format_id: "gen3ou".to_string(),
        seed: None,
        p1: PlayerOptions { name: "A".to_string(), team: PackedTeam(String::new()) },
        p2: PlayerOptions { name: "B".to_string(), team: PackedTeam(p2.to_string()) },
    };
    assert!(Battle::start(&empty_opts, &d).is_err(), "empty team -> Err");
}
