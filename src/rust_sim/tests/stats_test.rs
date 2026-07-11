//! Stat-computation tests:
//!   - `stats_golden_matches_showdown` — differential vs the REAL Showdown sim's
//!     OWN computed stats (golden from `harness/gen_stats_golden.js`): for every
//!     case, Rust `compute_stats(unpack(PACKED)[0], dex)` == the 6 stats the sim
//!     read off the live Pokémon object (`a.maxhp` + `a.storedStats`).
//!   - `stats_smoke` — a few authoritative hand-checks independent of the golden
//!     (the spec's verified canonical values + the floor/nature edges).
//!
//! Mirrors the structure of `tests/dex_test.rs` / `tests/team_test.rs`.

use pokesim::dex::Dex;
use pokesim::stats::compute_stats;
use pokesim::team::{unpack, PokemonSet};

fn dex() -> Dex {
    Dex::for_gen(3)
}

#[test]
fn stats_golden_matches_showdown() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/stats_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing stats golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_stats_golden.js")
    });
    let d = dex();

    let mut checked = 0usize;
    for (i, line) in data.lines().enumerate() {
        let ln = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f[0], "STAT", "unknown record {:?} (line {ln})", f[0]);
        assert_eq!(f.len(), 8, "STAT needs packed + 6 stats (line {ln})");

        let packed = f[1];
        let exp: Vec<u16> = f[2..8]
            .iter()
            .map(|x| x.parse().unwrap_or_else(|e| panic!("bad stat {x:?} (line {ln}): {e}")))
            .collect();

        let sets = unpack(packed, &d).unwrap_or_else(|e| panic!("unpack failed (line {ln}): {e}"));
        assert_eq!(sets.len(), 1, "single-set golden case (line {ln})");

        let got = compute_stats(&sets[0], &d)
            .unwrap_or_else(|e| panic!("compute_stats failed (line {ln}): {e}"));
        assert_eq!(
            got.to_vec(),
            exp,
            "stat mismatch (line {ln})\n  packed: {packed}\n  species: {}\n  got: {got:?} exp: {exp:?}",
            sets[0].species
        );
        checked += 1;
    }
    assert!(checked >= 15, "expected the full stats golden corpus, got {checked}");
    eprintln!("stats golden: {checked} cases passed");
}

#[test]
fn stats_smoke() {
    let d = dex();

    // Canonical values read off the live sim (the golden's ground truth).
    // Adamant 252HP/252Atk Tyranitar -> [404,403,256,203,236,158] (0-EV SpD = 236;
    // the spec's "237" was an approximation — the sim is authoritative here).
    let ttar = PokemonSet {
        species: "Tyranitar".into(),
        nature: "Adamant".into(),
        evs: [252, 252, 0, 0, 0, 0],
        moves: vec!["Crunch".into()],
        ..Default::default()
    };
    assert_eq!(compute_stats(&ttar, &d).unwrap(), [404, 403, 256, 203, 236, 158]);

    // Bold 252HP/252Def Skarmory -> [334,176,416,116,176,176] (Atk hindered to 176,
    // Def boosted to 416) — confirms the nature multiply is integer + AFTER +5.
    let skarm = PokemonSet {
        species: "Skarmory".into(),
        nature: "Bold".into(),
        evs: [252, 0, 252, 0, 0, 0],
        moves: vec!["Spikes".into()],
        ..Default::default()
    };
    assert_eq!(compute_stats(&skarm, &d).unwrap(), [334, 176, 416, 116, 176, 176]);

    // Serious (neutral) 0-EV Blissey -> [651,56,56,186,306,146]; HP never natured.
    let bliss = PokemonSet {
        species: "Blissey".into(),
        nature: "Serious".into(),
        evs: [0, 0, 0, 0, 0, 0],
        moves: vec!["Soft-Boiled".into()],
        ..Default::default()
    };
    assert_eq!(compute_stats(&bliss, &d).unwrap(), [651, 56, 56, 186, 306, 146]);

    // Shedinja base HP 1 -> HP stat is 1 regardless of HP EVs/IVs.
    let shed = PokemonSet {
        species: "Shedinja".into(),
        nature: "Adamant".into(),
        evs: [252, 252, 0, 0, 0, 4],
        moves: vec!["Shadow Ball".into()],
        ..Default::default()
    };
    assert_eq!(compute_stats(&shed, &d).unwrap()[0], 1, "Shedinja HP == 1");

    // Crash-don't-drop: unknown species / nature each Err (never a silent value).
    let bad_species = PokemonSet { species: "NotAMon".into(), nature: "Hardy".into(), ..Default::default() };
    assert!(compute_stats(&bad_species, &d).is_err(), "unknown species -> Err");
    let bad_nature = PokemonSet { species: "Blissey".into(), nature: "Zany".into(), ..Default::default() };
    assert!(compute_stats(&bad_nature, &d).is_err(), "unknown nature -> Err");
}
