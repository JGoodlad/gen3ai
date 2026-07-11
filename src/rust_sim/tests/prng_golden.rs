//! Differential golden test: the Rust PRNG must reproduce the REAL Pokémon
//! Showdown `prng.js` bit-for-bit.
//!
//! Reads `tests/vectors/prng_golden.txt` (regenerate via
//! `node src/rust_sim/harness/gen_prng_vectors.js`). Each line is an independent
//! assertion keyed on a pre-state seed string, so every check reconstructs a
//! fresh [`Prng`] from that seed — no JSON parser, no shared call-order coupling.

use pokesim::prng::Prng;
use std::fs;

#[test]
fn prng_matches_showdown_golden() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/prng_golden.txt");
    let data = fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing golden vectors ({path}): {e}\nrun: node src/rust_sim/harness/gen_prng_vectors.js")
    });

    let mut checked = 0usize;
    for (i, line) in data.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let ln = i + 1;
        match f[0] {
            "NEXT" => {
                let mut p = Prng::new(f[1]);
                let val: u32 = f[2].parse().unwrap();
                assert_eq!(p.next(), val, "NEXT value (line {ln})");
                assert_eq!(p.get_seed(), f[3], "NEXT post-seed (line {ln})");
            }
            "RANDN" => {
                let mut p = Prng::new(f[1]);
                let n: u32 = f[2].parse().unwrap();
                let exp: u32 = f[3].parse().unwrap();
                assert_eq!(p.random_below(n), exp, "RANDN (line {ln})");
            }
            "RANDMN" => {
                let mut p = Prng::new(f[1]);
                let m: u32 = f[2].parse().unwrap();
                let n: u32 = f[3].parse().unwrap();
                let exp: u32 = f[4].parse().unwrap();
                assert_eq!(p.random_range(m, n), exp, "RANDMN (line {ln})");
            }
            "CHANCE" => {
                let mut p = Prng::new(f[1]);
                let num: u32 = f[2].parse().unwrap();
                let den: u32 = f[3].parse().unwrap();
                let exp = f[4] == "1";
                assert_eq!(p.random_chance(num, den), exp, "CHANCE (line {ln})");
            }
            "SHUFFLE" => {
                let mut p = Prng::new(f[1]);
                let len: usize = f[2].parse().unwrap();
                let mut arr: Vec<u32> = (0..len as u32).collect();
                p.shuffle(&mut arr);
                let got = arr
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                assert_eq!(got, f[3], "SHUFFLE (line {ln})");
            }
            other => panic!("unknown record {other:?} (line {ln})"),
        }
        checked += 1;
    }
    assert!(checked > 1000, "expected the full golden corpus, got {checked}");
    eprintln!("prng golden: {checked} assertions passed");
}
