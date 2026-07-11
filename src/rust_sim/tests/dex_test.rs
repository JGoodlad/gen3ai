//! Dex tests:
//!   - `dex_parity_matches_facade` — differential vs `agents.gen3_data` (golden
//!     from `harness/gen_dex_golden.py`): every species/move/type/nature/learnset.
//!   - `dex_smoke` — a few authoritative hand-checks, independent of the golden.

use pokesim::dex::{Dex, Type};

fn dex() -> Dex {
    Dex::for_gen(3)
}

#[test]
fn dex_parity_matches_facade() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/dex_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing dex golden ({path}): {e}\nrun: PYTHONPATH=src python3 src/rust_sim/harness/gen_dex_golden.py")
    });
    let d = dex();
    let approx = |a: f64, b: f64| (a - b).abs() < 1e-9;

    let mut checked = 0usize;
    for (i, line) in data.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let ln = i + 1;
        match f[0] {
            "SPECIES" => {
                let s = d.species(f[1]).unwrap_or_else(|| panic!("missing species {} (line {ln})", f[1]));
                assert_eq!(s.num.to_string(), f[2], "species num {} (line {ln})", f[1]);
                let bs = &s.base_stats;
                let got = [bs.hp, bs.atk, bs.def, bs.spa, bs.spd, bs.spe];
                let exp: Vec<u16> = f[3..9].iter().map(|x| x.parse().unwrap()).collect();
                assert_eq!(got.to_vec(), exp, "species stats {} (line {ln})", f[1]);
                let types = if s.types.is_empty() {
                    "-".to_string()
                } else {
                    s.types.iter().map(|t| t.name()).collect::<Vec<_>>().join(",")
                };
                assert_eq!(types, f[9], "species types {} (line {ln})", f[1]);
            }
            "MOVE" => {
                let m = d.moves(f[1]).unwrap_or_else(|| panic!("missing move {} (line {ln})", f[1]));
                assert_eq!(m.num.to_string(), f[2], "move num {} (line {ln})", f[1]);
                assert_eq!(m.base_power.to_string(), f[3], "move bp {} (line {ln})", f[1]);
                let ty = m.move_type.map_or("THREE_QUESTION_MARKS", Type::name);
                assert_eq!(ty, f[4], "move type {} (line {ln})", f[1]);
                assert_eq!(m.category.name(), f[5], "move category {} (line {ln})", f[1]);
                assert_eq!(m.accuracy.to_string(), f[6], "move accuracy {} (line {ln})", f[1]);
            }
            "TYPE" => {
                let def_t = Type::from_name(f[1]).unwrap();
                let att_t = Type::from_name(f[2]).unwrap();
                let exp: f64 = f[3].parse().unwrap();
                let got = d.type_chart().multiplier(def_t, att_t);
                assert!(approx(got, exp), "type {} vs {}: {got} != {exp} (line {ln})", f[2], f[1]);
            }
            "NATURE" => {
                let n = d.nature(f[1]).unwrap_or_else(|| panic!("missing nature {} (line {ln})", f[1]));
                for (got, raw) in [n.atk, n.def, n.spa, n.spd, n.spe].iter().zip(&f[2..7]) {
                    let exp: f64 = raw.parse().unwrap();
                    assert!(approx(*got, exp), "nature {} mult {got} != {exp} (line {ln})", f[1]);
                }
            }
            "LEARNSET" => {
                let mut got: Vec<&str> = d.learnset(f[1]).iter().map(String::as_str).collect();
                got.sort();
                assert_eq!(got.join(","), f[2], "learnset {} (line {ln})", f[1]);
            }
            other => panic!("unknown record {other:?} (line {ln})"),
        }
        checked += 1;
    }
    assert!(checked > 1000, "expected the full dex golden corpus, got {checked}");
    eprintln!("dex parity: {checked} assertions passed");
}

#[test]
fn dex_smoke() {
    let d = dex();
    assert!(d.species_count() > 250, "gen3 should have 250+ species");
    assert!(d.move_count() > 300);

    // Tyranitar: Rock/Dark, known base stats.
    let ttar = d.species("Tyranitar").expect("tyranitar");
    assert_eq!(ttar.types, vec![Type::Rock, Type::Dark]);
    assert_eq!(ttar.base_stats.atk, 134);
    assert_eq!(ttar.base_stats.spe, 61);

    // Gen-3 category derivation: Thunderbolt is type-based Special; Earthquake Physical.
    assert_eq!(d.moves("Thunderbolt").unwrap().category, pokesim::dex::MoveCategory::Special);
    assert_eq!(d.moves("Earthquake").unwrap().category, pokesim::dex::MoveCategory::Physical);
    assert_eq!(d.moves("Recover").unwrap().category, pokesim::dex::MoveCategory::Status);

    // Type chart: Ice vs Dragon/Flying = 4×; Ground vs Flying = 0× (immune).
    let chart = d.type_chart();
    assert!((chart.effectiveness(Type::Ice, &[Type::Dragon, Type::Flying]) - 4.0).abs() < 1e-9);
    assert!(chart.effectiveness(Type::Ground, &[Type::Flying]).abs() < 1e-9);

    // Learnset legality + id normalization (display name -> id).
    assert!(d.can_learn("Tyranitar", "Rock Slide"));
    assert!(!d.can_learn("Tyranitar", "notarealmove"));
    assert!(d.moves("Hidden Power Ice").is_some(), "typed HP present as distinct id");
}
