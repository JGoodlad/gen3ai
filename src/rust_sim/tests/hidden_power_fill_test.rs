//! The engine's Hidden Power reverse formula (`state::hidden_power_type` / `hidden_power_bp`, the port's copy of
//! `Dex.getHiddenPower`) against the SAME source as the one Python table (`agents.gen3_data.hidden_power.HP_IVS`):
//! Showdown's validator fill, `fillStats(typechart.ts HPivs, 31)`, read from the pinned submodule. Every type's fill
//! must come back as that type at 70 power (`hidden_power_test.py` pins the Python side to the same rows, and the
//! engine's type list to the Python formula's).

use pokesim::state::{hidden_power_bp, hidden_power_type};

/// `typechart.ts` `HPivs` per type, filled with 31, as `[HP, Atk, Def, SpA, SpD, Spe]`.
fn showdown_fill() -> Vec<(String, [u8; 6])> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deps/pokemon-showdown/data/typechart.ts");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let stats = ["hp", "atk", "def", "spa", "spd", "spe"];
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    for line in src.lines() {
        // A type entry opens at one tab: `\tbug: {`.
        if let Some(rest) = line.strip_prefix('\t') {
            if !rest.starts_with('\t') && rest.ends_with(": {") {
                current = Some(rest.trim_end_matches(": {").to_string());
            }
        }
        let t = line.trim();
        if let (Some(name), Some(body)) = (current.as_ref(), t.strip_prefix("HPivs: {")) {
            if ["normal", "fairy", "stellar"].contains(&name.as_str()) {
                continue; // no gen-3 Hidden Power of these types
            }
            let body = body.trim_end_matches(&[',', '}'][..]);
            let mut ivs = [31u8; 6];
            for kv in body.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let (k, v) = kv.split_once(':').expect("HPivs entry `stat: n`");
                let i = stats.iter().position(|s| *s == k.trim()).unwrap_or_else(|| panic!("stat {k}"));
                ivs[i] = v.trim().parse().expect("an IV");
            }
            out.push((name.clone(), ivs));
        }
    }
    out
}

#[test]
fn every_showdown_hp_fill_reads_back_as_its_type_at_power_70() {
    let fill = showdown_fill();
    let names: Vec<&str> = fill.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(fill.len(), 16, "typechart.ts HPivs rows: {names:?}");
    for (name, ivs) in &fill {
        assert_eq!(hidden_power_type(ivs), name.as_str(), "{name} {ivs:?}");
        assert_eq!(hidden_power_bp(ivs), 70, "{name} {ivs:?}");
    }
}

#[test]
fn the_reverse_formula_on_a_hand_worked_spread() {
    // Atk 28 drops bit 1 (power 68), and the type bits read Ice (probe_hidden_power_bp_regression_rng.js).
    let ivs = [31, 28, 30, 31, 31, 31];
    assert_eq!((hidden_power_type(&ivs), hidden_power_bp(&ivs)), ("ice", 68));
    assert_eq!(hidden_power_type(&[31; 6]), "dark");
}
