//! The WHOLE-BATTLE guard for the per-side B1 and mirror construction-window allowlist keys
//! (`gen3_perside_window_keys_whole_battle_v1`), proven on the REAL repro battles.
//!
//! The `--ab` verdict is first-divergence: a battle whose first per-side divergence is a turn-0
//! construction speed-tie reorder is admitted by a key, and nothing later in it is read. B1 and the
//! mirror key used to judge only that framing window, so a REAL engine bug later in such a battle
//! (the class the Rollout perish-faint repro `bab_0_8` belonged to) would have passed the bridge
//! fuzzer as `allowlisted`. Each case here takes a committed allowlisted fixture (16 = B1, 17 = the
//! mirror `[of]` flip, 18 = the mirror Intimidate permutation), changes ONE `-damage` HP after
//! `|turn|1` on ONE side of the golden, and requires the replayer to FAIL the gate (a `diverge`
//! with no `allowlisted` reason). With the whole-battle clause reverted every case is
//! `allowlisted` again, so each fails on revert.

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/bridge_corpus").join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// The raw protocol line of a `CHUNK` row (`CHUNK\tid\tbatch\tside\tchunk\tidx\t<raw>`).
fn raw(row: &str) -> Option<&str> {
    let mut it = row.splitn(7, '\t');
    (it.next()? == "CHUNK").then_some(())?;
    it.nth(5)
}

fn side_of(row: &str) -> Option<&str> {
    let f: Vec<&str> = row.splitn(5, '\t').collect();
    (f.first() == Some(&"CHUNK")).then(|| f.get(3).copied()).flatten()
}

/// Lower the HP of `side`'s first `|-damage|` line after its first `|turn|` by one.
fn mangle_later_damage(text: &str, side: &str) -> String {
    let mut rows: Vec<String> = text.split('\n').map(str::to_string).collect();
    let mut past_turn = false;
    let mut hit = None;
    for (i, row) in rows.iter().enumerate() {
        if side_of(row) != Some(side) {
            continue;
        }
        let r = raw(row).unwrap_or("");
        if r.starts_with("|turn|") {
            past_turn = true;
        } else if past_turn && r.starts_with("|-damage|") {
            hit = Some(i);
            break;
        }
    }
    let i = hit.unwrap_or_else(|| panic!("precondition: no {side} -damage after |turn|"));
    let row = rows[i].clone();
    let line = raw(&row).unwrap();
    let mut fields: Vec<String> = line.split('|').map(str::to_string).collect();
    // `|-damage|<ident>|<hp>/<max>…` or `|-damage|<ident>|0 fnt`: fields[3] is the condition.
    let cond = fields[3].clone();
    let (hp, rest) = cond.split_at(cond.find(|c: char| !c.is_ascii_digit()).unwrap_or(cond.len()));
    let hp: u32 = hp.parse().unwrap_or_else(|_| panic!("precondition: numeric HP in {line}"));
    fields[3] = format!("{}{rest}", if hp == 0 { 1 } else { hp - 1 });
    let new_line = fields.join("|");
    rows[i] = format!("{}{}", &row[..row.len() - line.len()], new_line);
    rows.join("\n")
}

/// Replay one single-battle golden text and return `(verdict, allowlisted)`.
fn replay(text: &str, tag: &str) -> (String, Option<String>) {
    let dir = std::env::temp_dir().join(format!("bridge_window_keys_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("battle.txt");
    std::fs::write(&file, text).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_bridge_replay"))
        .arg(&file)
        .arg("--ab")
        .output()
        .expect("run bridge_replay");
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with('{') && !l.starts_with("{\"chunk_summary\""))
        .collect();
    assert_eq!(lines.len(), 1, "expected one battle verdict, got: {stdout}");
    let pluck = |key: &str| {
        let needle = format!("\"{key}\":\"");
        let s = lines[0].find(&needle)? + needle.len();
        let rest = &lines[0][s..];
        Some(rest[..rest.find('"')?].to_string())
    };
    (pluck("verdict").unwrap_or_default(), pluck("allowlisted"))
}

fn check(name: &str, reason: &str) {
    let text = fixture(name);
    // Precondition: the untouched repro is admitted by its key.
    let (v, a) = replay(&text, &format!("{}_pos", &name[..2]));
    assert_eq!((v.as_str(), a.as_deref()), ("diverge", Some(reason)), "{name}: precondition");
    for side in ["p1", "p2"] {
        let (v, a) = replay(&mangle_later_damage(&text, side), &format!("{}_{side}", &name[..2]));
        assert_eq!(
            (v.as_str(), a.as_deref()),
            ("diverge", None),
            "{name}: a later {side} -damage divergence must FAIL the gate, not hide behind {reason}"
        );
    }
}

#[test]
fn b1_order_flip_does_not_hide_a_later_divergence() {
    check("16_construction_order_flip_cg.txt", "turn0-construction-speed-tie-order-flip");
}

#[test]
fn mirror_of_flip_does_not_hide_a_later_divergence() {
    check("17_perside_construction_mirror_flip.txt", "perside-construction-speed-tie-mirror-of-flip");
}

#[test]
fn mirror_permutation_does_not_hide_a_later_divergence() {
    check(
        "18_perside_construction_mirror_permutation.txt",
        "perside-construction-speed-tie-mirror-of-flip",
    );
}
