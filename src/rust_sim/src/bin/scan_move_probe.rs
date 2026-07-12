//! scan_move_probe — the AUTHORITATIVE engine move-coverage oracle.
//!
//! For each candidate gen3 move id (given on stdin, one per line), construct a minimal
//! gen3customgame battle where p1's lead USES that move (move slot 0) against an inert
//! p2, drive it through the UNCHANGED public engine (`Battle::start_with_switchins` →
//! `run_full_battle`), and report whether the ENGINE fail-louds — a `panic!` (caught via
//! `catch_unwind`, incl. the `is not modeled` / `not supported` guards). This is the
//! mod-chain-law-clean source of truth for "can the engine EXECUTE this move" — separate
//! from the e2e picker `isModeledMove` (which has false positives).
//!
//! Emits ONE JSON verdict line per move on stdout:
//!   {"move":"doubleedge","verdict":"ran"}        // engine executed it, no fail-loud
//!   {"move":"wish","verdict":"panic","detail":"status move \"wish\" is not modeled ..."}
//!   {"move":"...","verdict":"build_error","detail":"..."}
//!
//! ADDITIVE ONLY: links the existing lib APIs; the engine's battle path is untouched.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision};
use std::io::{self, BufRead, Write};
use std::panic::{self, AssertUnwindSafe};

// A packed gen3 set: name|species|item|ability|moves|nature|evs|gender|ivs|...|
// We build the packed string directly (the team codec ingests it). The attacker's
// slot-0 move is the candidate; slots 1-3 are Splash filler so the mon is legal.
// Both sides get full HP tanks so the move actually EXECUTES over several turns.
fn packed(species: &str, ability: &str, m0: &str) -> String {
    // name|species|item|ability|move1,move2,move3,move4|nature|evs|gender|ivs|shiny|level|happiness,...
    // gen3: 6 EV fields, IVs default 31. Use Snorlax-like bulk via the species chosen.
    format!(
        "|{species}||{ability}|{m0},splash,splash,splash|Serious|252,252,252,252,252,252|||||",
    )
}

fn probe_move(move_id: &str, dex: &Dex) -> (String, Option<String>) {
    // Attacker: a bulky normal (Snorlax) using the move. Defender: an inert Snorlax
    // (Immunity so no status noise) with Splash — it just sits there.
    let p1_team = packed("snorlax", "immunity", move_id);
    let p2_team = packed("snorlax", "immunity", "splash");

    let opts = BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some("1,2,3,4".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1_team) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2_team) },
    };

    let build = panic::catch_unwind(AssertUnwindSafe(|| Battle::start_with_switchins(&opts, dex)));
    let mut battle = match build {
        Ok(Ok(b)) => b,
        Ok(Err(e)) => return ("build_error".into(), Some(e)),
        Err(p) => return ("build_panic".into(), Some(panic_msg(p))),
    };

    // p1 uses the candidate move (slot 0) for several turns; p2 splashes.
    let script: Vec<ScriptDecision> = (0..6)
        .map(|_| ScriptDecision::both(Choice::Move(0), Choice::Move(0)))
        .collect();

    let run = panic::catch_unwind(AssertUnwindSafe(|| {
        battle.state_mut().unwrap().run_full_battle(&script, dex)
    }));
    match run {
        Ok(_) => ("ran".into(), None),
        Err(p) => ("panic".into(), Some(panic_msg(p))),
    }
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic>".to_string()
    }
}

fn main() {
    // Silence panic's default stderr backtrace spam (we catch + report ourselves).
    panic::set_hook(Box::new(|_| {}));
    let dex = Dex::for_gen(3);
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let mv = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => continue,
        };
        if mv.is_empty() {
            continue;
        }
        let (verdict, detail) = probe_move(&mv, &dex);
        let detail_json = detail
            .map(|d| format!(",\"detail\":{}", json_str(&d)))
            .unwrap_or_default();
        writeln!(out, "{{\"move\":{},\"verdict\":\"{}\"{}}}", json_str(&mv), verdict, detail_json)
            .ok();
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}
