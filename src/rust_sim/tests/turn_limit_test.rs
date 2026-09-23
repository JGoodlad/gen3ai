//! turn_limit_test.rs — the TURN-LIMIT TIE gate (`gen3_turn_limit_tie_v1`).
//!
//! The pinned Showdown TIES a battle past turn 1000 and warns with `|bigerror|` countdown lines
//! from turn 500 (`sim/battle.ts:1834-1849`); the port used to PANIC at 1,000 committed turns
//! (`BATTLE_TURN_CAP`, the Rust Core Program's Phase-0 finding F1 — 3 of 300 random-player
//! battles reached it). Two gates:
//!
//! 1. **The differential** (`turn_limit_battle_matches_showdown_byte_for_byte`): a 2-mon vs 2-mon
//!    gen3ou battle in which both players switch every turn, captured from the REAL Node
//!    `getPlayerStreams` by `harness/gen_turn_limit_capture.js`, replayed through the PRODUCTION
//!    bridge session (`BridgeSession`, what `sim_bridge` runs) — every per-side line
//!    byte-identical (FNV-1a digest over the whole stream + the line count), and every line inside
//!    the captured windows (the first turns, each `|bigerror|`, the tie tail) compared verbatim so a
//!    failure names the line. Run from BOTH seed conventions: the golden's post-construction INIT
//!    seed, and the RAW `>start` seed through the full turn-0 construction window (the live path).
//! 2. **The regression pin** (`turn_limit_ties_instead_of_panicking`): on revert (the panic, or a
//!    tie without its message / at the wrong turn) this fails.

use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;

const GOLDEN: &str = include_str!("vectors/turn_limit_golden.txt");

/// FNV-1a 64 — mirrors `harness/gen_turn_limit_capture.js::fnv1a64`.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct Golden {
    p1_team: String,
    p2_team: String,
    init_seed: String,
    format: String,
    cmds: Vec<Cmd>,
    /// (side, flat line index, expected line)
    window: Vec<(usize, usize, String)>,
    /// per side: (line count, digest)
    digest: [(usize, u64); 2],
}

fn side_idx(s: &str) -> usize {
    match s {
        "p1" => 0,
        "p2" => 1,
        o => panic!("bad side {o}"),
    }
}

fn parse_golden() -> Golden {
    let mut g = Golden {
        p1_team: String::new(),
        p2_team: String::new(),
        init_seed: String::new(),
        format: String::new(),
        cmds: Vec::new(),
        window: Vec::new(),
        digest: [(0, 0); 2],
    };
    for line in GOLDEN.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.splitn(6, '\t').collect();
        match f[0] {
            "TEAM" => {
                let team = line.splitn(4, '\t').nth(3).unwrap().to_string();
                if f[2] == "p1" {
                    g.p1_team = team;
                } else {
                    g.p2_team = team;
                }
            }
            "INIT" => {
                g.init_seed = f[3].to_string();
                g.format = f[4].to_string();
            }
            "CMD" => {
                let side = side_idx(f[4]);
                let choice = parse_choice(f[5]).expect("choice");
                g.cmds.push(Cmd { side, choice });
            }
            "CHUNK" => {
                let parts: Vec<&str> = line.splitn(5, '\t').collect();
                g.window.push((side_idx(parts[2]), parts[3].parse().unwrap(), parts[4].to_string()));
            }
            "DIGEST" => {
                let parts: Vec<&str> = line.split('\t').collect();
                let n: usize = parts[3].parse().unwrap();
                let d = u64::from_str_radix(parts[4], 16).unwrap();
                g.digest[side_idx(parts[2])] = (n, d);
            }
            _ => {}
        }
    }
    assert!(!g.cmds.is_empty() && !g.window.is_empty(), "golden parsed empty");
    g
}

fn flat_lines(sess: &BridgeSession, side: usize) -> Vec<String> {
    sess.chunks().side_chunks(side).flat_map(|c| c.lines.iter().cloned()).collect()
}

fn assert_matches_golden(label: &str, sess: &BridgeSession, g: &Golden) {
    assert!(sess.is_ended(), "[{label}] the battle must END (at the turn limit)");
    assert_eq!(sess.winner(), None, "[{label}] the turn limit is a TIE");
    for side in 0..2 {
        let got = flat_lines(sess, side);
        for (s, idx, want) in g.window.iter().filter(|w| w.0 == side) {
            let _ = s;
            let have = got.get(*idx).map(String::as_str).unwrap_or("<missing>");
            assert_eq!(have, want, "[{label}] p{} line {idx} diverges from Showdown", side + 1);
        }
        let mut buf = String::new();
        for l in &got {
            buf.push_str(l);
            buf.push('\n');
        }
        let (n, d) = g.digest[side];
        assert_eq!(got.len(), n, "[{label}] p{} line COUNT", side + 1);
        assert_eq!(fnv1a64(buf.as_bytes()), d, "[{label}] p{} stream DIGEST (a line outside the windows differs)", side + 1);
    }
}

#[test]
fn turn_limit_battle_matches_showdown_byte_for_byte() {
    let g = parse_golden();
    let dex = Dex::for_gen(3);
    let opts = bridge_opts(&g.format, g.init_seed.clone(), &g.p1_team, &g.p2_team);
    let mut sess = BridgeSession::new(&opts, &dex).expect("session");
    sess.feed_cmds(&g.cmds, &dex);
    assert_matches_golden("post-construction seed", &sess, &g);
}

#[test]
fn turn_limit_battle_matches_showdown_from_the_raw_start_seed() {
    // The LIVE path: `sim_bridge` is fed poke-env's raw `>start` seed and runs the turn-0
    // construction window itself (`new_construct_turn0`). The harness's `>start` seed is
    // [101, 103, 107, 109].
    let g = parse_golden();
    let dex = Dex::for_gen(3);
    let opts = bridge_opts(&g.format, "101,103,107,109".to_string(), &g.p1_team, &g.p2_team);
    let mut sess = BridgeSession::new_construct_turn0(&opts, &dex).expect("session");
    sess.feed_cmds(&g.cmds, &dex);
    assert_matches_golden("raw >start seed", &sess, &g);
}

/// REGRESSION PIN (`gen3_turn_limit_tie_v1`). WRONG pre-fix: the driver panicked with "run_full_battle
/// runaway: >1000 committed turns" on the 1001st commit, so a battle Showdown TIES crashed the
/// bridge child (a random-player corpus hits it on ~1% of battles). Asserts the tie, its exact
/// announcement, the countdown's first/last lines and that nothing follows the `|tie|`.
#[test]
fn turn_limit_ties_instead_of_panicking() {
    let g = parse_golden();
    let dex = Dex::for_gen(3);
    let opts = bridge_opts(&g.format, g.init_seed.clone(), &g.p1_team, &g.p2_team);
    let mut sess = BridgeSession::new(&opts, &dex).expect("session");
    sess.feed_cmds(&g.cmds, &dex);
    assert!(sess.is_ended() && sess.winner().is_none(), "a TIE at the limit");
    assert_eq!(sess.turn(), 1001, "Showdown's `this.turn` when it ties");
    let p1 = flat_lines(&sess, 0);
    let tail: Vec<&str> = p1.iter().rev().take(3).rev().map(String::as_str).collect();
    assert_eq!(tail, ["|message|It is turn 1000. You have hit the turn limit!", "|", "|tie"]);
    let bigerrors: Vec<&String> = p1.iter().filter(|l| l.starts_with("|bigerror|")).collect();
    assert_eq!(bigerrors.len(), 5 + 8 + 11, "500..=900 by 100, 910..=980 by 10, 990..=1000 every turn");
    assert_eq!(bigerrors[0], "|bigerror|You will auto-tie if the battle doesn't end in 500 turns (on turn 1000).");
    assert_eq!(bigerrors[bigerrors.len() - 2], "|bigerror|You will auto-tie if the battle doesn't end in 1 turn (on turn 1000).");
    assert_eq!(pokesim::turn::turn_limit_warning(499), None);
    assert_eq!(pokesim::turn::turn_limit_warning(1001), None);
}
