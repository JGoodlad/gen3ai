//! `bridge_replay` — replay a bridge-capture golden's (teams, seed, CMD stream)
//! through the crate's per-side bridge emitter and either PRINT the per-side chunk
//! streams or DIFF them against the golden's recorded per-side CHUNK rows.
//!
//! Additive (links only public lib APIs), like `ab_replay`. This is the entry point
//! the future random-team request A/B fuzzer drives: it feeds a (teams, seed, CMD)
//! record and compares the Rust per-side stream to the real sim's.
//!
//! Usage:
//!   bridge_replay <golden.txt>                 # diff EVERY battle, print PASS/FAIL summary
//!   bridge_replay <golden.txt> <battle_id>     # diff ONE battle, print first divergence
//!   bridge_replay <golden.txt> <battle_id> --print   # print the Rust per-side streams
//!   bridge_replay <golden.txt> --ab            # ONE JSON verdict per battle (the A/B fuzzer
//!                                                driver; never panics — a panic is caught +
//!                                                reported as {"verdict":"panic"})
//!   bridge_replay <repro-dir>                  # replay a saved repro dir's battle.txt
//!
//! Exit code 0 = all diffed battles byte-equal; 1 = a divergence (or an error).

use std::process::ExitCode;

use pokesim::bridge::{parse_bridge_golden, run_full_battle_bridge, bridge_opts, GoldenBattle};
use pokesim::dex::Dex;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: bridge_replay <golden.txt> [battle_id] [--print]");
        return ExitCode::from(2);
    }
    let arg1 = &args[1];
    let battle_id = args.get(2).filter(|a| !a.starts_with("--")).cloned();
    let do_print = args.iter().any(|a| a == "--print");
    let do_ab = args.iter().any(|a| a == "--ab");

    // A repro-dir argument resolves to `<dir>/battle.txt` (the repro→replay path).
    let path: String = {
        let p = std::path::Path::new(arg1);
        if p.is_dir() {
            p.join("battle.txt").to_string_lossy().into_owned()
        } else {
            arg1.clone()
        }
    };

    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let battles = match parse_bridge_golden(&text) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("parse error: {e}");
            return ExitCode::from(2);
        }
    };
    let dex = Dex::for_gen(3);

    // ── A/B mode: one JSON verdict per battle, panic-caught (never dies). ──
    if do_ab {
        let mut all_ok = true;
        let mut n_ok = 0usize;
        let mut n_div = 0usize;
        for b in &battles {
            let v = ab_verdict(b, &dex);
            if v.contains("\"verdict\":\"ok\"") {
                n_ok += 1;
            } else {
                n_div += 1;
                all_ok = false;
            }
            println!("{v}");
        }
        println!("{{\"chunk_summary\":true,\"battles\":{},\"ok\":{n_ok},\"diverged\":{n_div}}}", battles.len());
        return if all_ok { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }

    let selected: Vec<&GoldenBattle> = match &battle_id {
        Some(id) => battles.iter().filter(|b| &b.id == id).collect(),
        None => battles.iter().collect(),
    };
    if selected.is_empty() {
        eprintln!("no battle matched {battle_id:?}");
        return ExitCode::from(2);
    }

    let mut all_pass = true;
    let mut total_chunks = 0usize;
    let mut total_requests = 0usize;
    let mut total_trapped = 0usize;
    for b in &selected {
        let opts = bridge_opts(&b.format_id, b.seed.clone(), &b.p1_team, &b.p2_team);
        let streams = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_full_battle_bridge(&opts, &b.cmds, &dex)
        })) {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                println!("[{}] ERROR: {e}", b.id);
                all_pass = false;
                continue;
            }
            Err(_) => {
                println!("[{}] PANIC during replay", b.id);
                all_pass = false;
                continue;
            }
        };
        if do_print {
            println!("=== {} p1 ===", b.id);
            for l in &streams.p1 {
                println!("{l}");
            }
            println!("=== {} p2 ===", b.id);
            for l in &streams.p2 {
                println!("{l}");
            }
            continue;
        }
        let (ok, msg) = diff_side(&b.id, "p1", &streams.p1, &b.p1_expected);
        let (ok2, msg2) = diff_side(&b.id, "p2", &streams.p2, &b.p2_expected);
        for l in &streams.p1 {
            if l.starts_with("|request|") {
                total_requests += 1;
                if l.contains("\"trapped\":true") {
                    total_trapped += 1;
                }
            }
        }
        for l in &streams.p2 {
            if l.starts_with("|request|") {
                total_requests += 1;
                if l.contains("\"trapped\":true") {
                    total_trapped += 1;
                }
            }
        }
        total_chunks += streams.p1.len() + streams.p2.len();
        if ok && ok2 {
            if battle_id.is_some() {
                println!("[{}] PASS ({} p1 lines, {} p2 lines)", b.id, streams.p1.len(), streams.p2.len());
            }
        } else {
            all_pass = false;
            if !ok {
                println!("{msg}");
            }
            if !ok2 {
                println!("{msg2}");
            }
        }
    }
    if !do_print {
        println!(
            "SUMMARY: {} battles, {} per-side lines, {} |request| frames, {} trapped:true — {}",
            selected.len(),
            total_chunks,
            total_requests,
            total_trapped,
            if all_pass { "ALL BYTE-EQUAL" } else { "DIVERGED" }
        );
    }
    if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Compare a Rust per-side stream to the golden's expected lines; return (ok, msg)
/// with a first-divergence description.
fn diff_side(id: &str, side: &str, got: &[String], want: &[String]) -> (bool, String) {
    let n = got.len().min(want.len());
    for i in 0..n {
        if got[i] != want[i] {
            return (
                false,
                format!(
                    "[{id}] {side} DIVERGE at line {i}:\n  want: {}\n  got:  {}",
                    truncate(&want[i]),
                    truncate(&got[i])
                ),
            );
        }
    }
    if got.len() != want.len() {
        let extra_side = if got.len() > want.len() { "GOT extra" } else { "WANT extra" };
        let sample = if got.len() > want.len() {
            got.get(n).map(|s| truncate(s)).unwrap_or_default()
        } else {
            want.get(n).map(|s| truncate(s)).unwrap_or_default()
        };
        return (
            false,
            format!(
                "[{id}] {side} LENGTH mismatch: got {} lines, want {} ({extra_side} at {n}: {sample})",
                got.len(),
                want.len()
            ),
        );
    }
    (true, String::new())
}

fn truncate(s: &str) -> String {
    if s.len() > 200 {
        format!("{}…", &s[..200])
    } else {
        s.to_string()
    }
}

// ===========================================================================
// A/B mode — one JSON verdict per battle for the request/per-side fuzzer.
// ===========================================================================

/// Replay `b` through the bridge emitter (panic-caught) and produce ONE JSON
/// verdict line: `ok` or the FIRST per-side divergence with a taxonomy kind.
fn ab_verdict(b: &GoldenBattle, dex: &Dex) -> String {
    let opts = bridge_opts(&b.format_id, b.seed.clone(), &b.p1_team, &b.p2_team);
    let streams = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_full_battle_bridge(&opts, &b.cmds, dex)
    }));
    let streams = match streams {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            return format!(
                "{{\"battle\":{},\"verdict\":\"error\",\"kind\":\"error\",\"detail\":{}}}",
                json_str(&b.id),
                json_str(&e)
            );
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".to_string());
            return format!(
                "{{\"battle\":{},\"verdict\":\"panic\",\"kind\":\"panic\",\"detail\":{}}}",
                json_str(&b.id),
                json_str(&msg)
            );
        }
    };
    // Diff each side; report the FIRST (earliest-line) divergence across both.
    let p1 = ab_diff_side("p1", &streams.p1, &b.p1_expected);
    let p2 = ab_diff_side("p2", &streams.p2, &b.p2_expected);
    let first = match (p1, p2) {
        (None, None) => {
            return format!("{{\"battle\":{},\"verdict\":\"ok\"}}", json_str(&b.id));
        }
        (Some(a), None) => a,
        (None, Some(b_)) => b_,
        (Some(a), Some(b_)) => {
            if a.line <= b_.line {
                a
            } else {
                b_
            }
        }
    };
    format!(
        "{{\"battle\":{},\"verdict\":\"diverge\",\"kind\":\"{}\",\"side\":\"{}\",\"line\":{},\"expected\":{},\"got\":{}}}",
        json_str(&b.id),
        first.kind,
        first.side,
        first.line,
        json_str(&first.expected),
        json_str(&first.got),
    )
}

struct AbDiff {
    side: &'static str,
    line: usize,
    kind: &'static str,
    expected: String,
    got: String,
}

/// Diff one side; return the first divergence (a length mismatch is `chunk_count`,
/// else the kind is classified from the first-diverging expected/got line).
fn ab_diff_side(side: &'static str, got: &[String], want: &[String]) -> Option<AbDiff> {
    let n = got.len().min(want.len());
    for i in 0..n {
        if got[i] != want[i] {
            return Some(AbDiff {
                side,
                line: i,
                kind: classify_line(&want[i], &got[i]),
                expected: want[i].clone(),
                got: got[i].clone(),
            });
        }
    }
    if got.len() != want.len() {
        let (exp, g) = if got.len() > want.len() {
            (String::new(), got.get(n).cloned().unwrap_or_default())
        } else {
            (want.get(n).cloned().unwrap_or_default(), String::new())
        };
        return Some(AbDiff {
            side,
            line: n,
            kind: "chunk_count",
            expected: exp,
            got: g,
        });
    }
    None
}

/// Taxonomy of a first-diverging per-side line: `request` (a `|request|` JSON) /
/// `error` (a trapped `|error|`) / `privacy` (an HP-bearing line — the HP-fold) /
/// `preamble` (framing before any request) / `perside` (anything else).
fn classify_line(expected: &str, got: &str) -> &'static str {
    let is_req = |l: &str| l.starts_with("|request|");
    let is_err = |l: &str| l.starts_with("|error|");
    let is_hp = |l: &str| {
        l.starts_with("|switch|")
            || l.starts_with("|drag|")
            || l.starts_with("|-damage|")
            || l.starts_with("|-heal|")
            || l.starts_with("|-sethp|")
    };
    let is_framing = |l: &str| {
        l.starts_with("|player|")
            || l.starts_with("|gametype|")
            || l.starts_with("|gen|")
            || l.starts_with("|tier|")
            || l.starts_with("|rule|")
            || l.starts_with("|teamsize|")
            || l.starts_with("|start")
            || l.starts_with("|t:|")
    };
    if is_req(expected) || is_req(got) {
        "request"
    } else if is_err(expected) || is_err(got) {
        "error"
    } else if is_hp(expected) || is_hp(got) {
        "privacy"
    } else if is_framing(expected) || is_framing(got) {
        "preamble"
    } else {
        "perside"
    }
}

/// Minimal JSON string escaper for the verdict lines.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
