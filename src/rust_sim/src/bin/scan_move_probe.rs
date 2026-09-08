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
//! `PROBE_KIND` extends the SAME oracle to the other three gen3 universes, so the
//! coverage census is one empirical instrument rather than four hand-maintained lists
//! (`gen3_coverage_census_v1`, 2026-09-08):
//!   * `PROBE_KIND=move`    (default) — the candidate is p1's slot-0 MOVE.
//!   * `PROBE_KIND=species` — the candidate is p1's lead SPECIES (Tackle/Splash filler),
//!     so a species whose dex row the port cannot build fails LOUD here rather than at
//!     the first team that carries it.
//!   * `PROBE_KIND=item`    — the candidate is p1's lead's HELD ITEM.
//!   * `PROBE_KIND=ability` — the candidate is p1's lead's ABILITY.
//! Every kind emits the same verdict line, keyed `"id"` in addition to the legacy
//! `"move"` key (kept so existing readers do not break).
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
    packed_with_item(species, "", ability, m0)
}

fn packed_with_item(species: &str, item: &str, ability: &str, m0: &str) -> String {
    // name|species|item|ability|move1,move2,move3,move4|nature|evs|gender|ivs|shiny|level|happiness,...
    // gen3: 6 EV fields, IVs default 31. Use Snorlax-like bulk via the species chosen.
    format!(
        "|{species}|{item}|{ability}|{m0},splash,splash,splash|Serious|252,252,252,252,252,252|||||",
    )
}

/// Which gen3 universe the stdin ids belong to. Read ONCE from `PROBE_KIND`.
#[derive(Clone, Copy, PartialEq)]
enum ProbeKind {
    Move,
    Species,
    Item,
    Ability,
}

/// Probe one id of `kind` through the unchanged public engine. Same construction as
/// [`probe_move`] — only the field the candidate lands in changes, so a `panic` verdict
/// always means "the ENGINE fail-louds on this id", never "the harness could not build a
/// legal set".
fn probe_id(kind: ProbeKind, id: &str, dex: &Dex) -> (String, Option<String>) {
    match kind {
        ProbeKind::Move => probe_move(id, dex),
        // Tackle (not Splash) in slot 0 so the probed mon actually ACTS: an item or ability
        // whose only handler is on the damage/contact path would otherwise never be reached
        // and would read as `ran` while being wholly unexercised.
        ProbeKind::Species => probe_pair(&packed(id, "", "tackle"), dex),
        ProbeKind::Item => probe_pair(&packed_with_item("snorlax", id, "immunity", "tackle"), dex),
        ProbeKind::Ability => probe_pair(&packed_with_item("snorlax", "", id, "tackle"), dex),
    }
}

/// The shared build-and-run: `p1_team` vs the same inert Snorlax [`probe_move`] uses.
fn probe_pair(p1_team: &str, dex: &Dex) -> (String, Option<String>) {
    let p2_team = packed("snorlax", "immunity", "splash");
    let opts = BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some("1,2,3,4".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1_team.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2_team) },
    };
    let build = panic::catch_unwind(AssertUnwindSafe(|| Battle::start_with_switchins(&opts, dex)));
    let mut battle = match build {
        Ok(Ok(b)) => b,
        Ok(Err(e)) => return ("build_error".into(), Some(e)),
        Err(p) => return ("build_panic".into(), Some(panic_msg(p))),
    };
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
    let kind = match std::env::var("PROBE_KIND").unwrap_or_else(|_| "move".into()).as_str() {
        "move" => ProbeKind::Move,
        "species" => ProbeKind::Species,
        "item" => ProbeKind::Item,
        "ability" => ProbeKind::Ability,
        other => {
            eprintln!("PROBE_KIND={other:?} is not one of move|species|item|ability");
            std::process::exit(2);
        }
    };
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
        let (verdict, detail) = probe_id(kind, &mv, &dex);
        let detail_json = detail
            .map(|d| format!(",\"detail\":{}", json_str(&d)))
            .unwrap_or_default();
        writeln!(
            out,
            "{{\"id\":{},\"move\":{},\"verdict\":\"{}\"{}}}",
            json_str(&mv),
            json_str(&mv),
            verdict,
            detail_json
        )
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
