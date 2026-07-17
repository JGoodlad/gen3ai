//! ab_replay — the A/B differential fuzzer's Rust-side chunk REPLAYER.
//!
//! Reads one or more chunk files in the e2e TAB golden format (SCEN/TEAM/INIT/
//! DEC/END — the exact format `harness/gen_e2e_fuzz.js`'s `emitBattle` writes and
//! `tests/e2e_fuzz_test.rs` parses), replays each battle through the UNCHANGED
//! public engine APIs (`Battle::start_with_switchins` → `run_full_battle`), and
//! emits ONE JSON verdict line per battle on stdout — **never panicking on a
//! divergence** (engine panics are caught and reported as `"verdict":"panic"`).
//!
//! Verdict lines (parsed by `harness/ab_fuzz.js`):
//!   {"battle":"ab_0_3","verdict":"ok","decisions":57}
//!   {"battle":"ab_0_3","verdict":"diverged","kind":"seed","decision":12,
//!    "expected":"...","got":"...","detail":"..."}
//!   {"battle":"ab_0_3","verdict":"panic","detail":"..."}
//! plus a trailing {"chunk_summary":{...}} line.
//!
//! First-divergence `kind` taxonomy (priority order within a decision):
//!   protocol  — (--protocol / L-rows present) an emitted `|...|` byte-stream divergence,
//!               reported ONLY after state/seed/winner all match — so a draw bug still
//!               reports `seed` (the root cause), never `protocol` (`decision` = line index)
//!   seed      — post-decision PRNG seed mismatch (a draw is mis-ordered/missing/extra)
//!   request   — move-vs-forceSwitch boundary mismatch
//!   species   — wrong active species (a wrong switch/drag target)
//!   state     — hp / maxhp / fainted / pokemon_left mismatch (seed matches)
//!   status    — major-status variant mismatch
//!   boost     — stat-stage mismatch
//!   confusion — confusion-counter mismatch
//!   spikes    — side spikes-layer mismatch
//!   firstmover— wrong first mover on a move turn
//! and per battle: start_error, init_seed, decision_count, ended, winner,
//! panic, parse_error.
//!
//! A REPRO DIR (as saved by ab_fuzz.js under <out>/divergences/) is accepted
//! directly: `ab_replay <repro-dir>` replays `<repro-dir>/battle.txt`.
//!
//! ADDITIVE ONLY: this binary links the existing lib APIs; the engine's battle
//! path is untouched.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::Status;
use pokesim::turn::{BattleOutcome, Choice, RequestKind, ScriptDecision};
use std::cell::RefCell;
use std::panic::{self, AssertUnwindSafe};

// ── Golden model (mirrors tests/e2e_fuzz_test.rs) ───────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WinTok {
    P1,
    P2,
    Tie,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReqTok {
    Move,
    Switch,
}

#[derive(Debug, Clone)]
struct SideExpect {
    species: String,
    hp: u16,
    maxhp: u16,
    fainted: bool,
    status: Option<Status>,
    boosts: [i8; 5],
    confusion: u8,
    left: usize,
    spikes: u8,
}

#[derive(Debug, Clone)]
struct DecExpect {
    request: ReqTok,
    force: [bool; 2],
    choice: [Option<Choice>; 2],
    seed_after: String,
    p1: SideExpect,
    p2: SideExpect,
    first_mover: String,
}

#[derive(Debug, Clone)]
struct CaseExpect {
    id: String,
    teams: [String; 2],
    init_seed: String,
    decisions: Vec<DecExpect>,
    ended: bool,
    winner: WinTok,
    /// The run format (`gen3customgame` default; `gen3ou` turns on the Sleep/Freeze-Clause
    /// SetStatus handler-sort shuffle + the OU framing). From the golden's `FMT` row.
    format: String,
    /// The captured REAL OMNISCIENT `|...|` lines (verbatim, `|t:|` normalized) — the
    /// byte-diff target (`gen3_omniscient_byte_fuzz_v1`). Empty on a state-only chunk.
    lines: Vec<String>,
}

// ── Shared line filter (mirrors tests/protocol_test.rs verbatim + a DENYLIST) ────
//
// A DENYLIST (keep everything except `debug`/`error`, normalize `|t:|`) rather than
// protocol_test.rs's batch-3-era ALLOWLIST — so the NEWLY-emitted batch-4c/5/6/snatch
// line types (`-mustrecharge`/`-prepare`/`-anim`/`-sethp`/`-copyboost`/`-hitcount`/
// `-singlemove`/`-setboost`/`-sideend`/`-cureteam`/…) are diffed AUTOMATICALLY instead of
// silently dropped on BOTH sides (which an allowlist would do → a blind pass). The
// `debug`/`error` drop + the `|t:|` normalization are the SAME two exclusions the
// protocol_test.rs helpers apply; BOTH sides run through the identical filter so they
// compare the same gated set.

/// The "type" of a raw protocol line (`|X|…` → `X`; the bare `|` → `""`).
fn line_type(raw: &str) -> &str {
    raw.strip_prefix('|').map_or(raw, |rest| rest.split('|').next().unwrap_or(""))
}

/// Normalize a `|t:|<unixtime>` line to the golden placeholder (positional-assert, not
/// value-assert — the timestamp is inherently non-reproducible by the engine).
fn normalize(raw: &str) -> String {
    if raw.starts_with("|t:|") {
        "|t:|<NORMALIZED>".to_string()
    } else {
        raw.to_string()
    }
}

/// Keep every line except the two poke-env-ignored free-form types (`debug`/`error`),
/// normalizing `|t:|`. Applied identically to BOTH the golden's captured lines and the
/// engine's emitted lines.
fn filter_bytes(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|l| {
            let t = line_type(l);
            t != "debug" && t != "error"
        })
        .map(|l| normalize(l))
        .collect()
}

// ── Non-panicking TAB parser ─────────────────────────────────────────────────

fn parse_choice(tok: &str) -> Result<Option<Choice>, String> {
    if tok == "-" {
        return Ok(None);
    }
    if tok.len() < 2 {
        return Err(format!("bad choice token {tok:?}"));
    }
    let (kind, num) = tok.split_at(1);
    let n: usize = num.parse().map_err(|e| format!("bad choice token {tok:?}: {e}"))?;
    match kind {
        "m" => Ok(Some(Choice::Move(n))),
        "s" => Ok(Some(Choice::Switch(n))),
        other => Err(format!("bad choice kind {other:?} in {tok:?}")),
    }
}

fn parse_status(tok: &str) -> Result<Option<Status>, String> {
    match tok {
        "-" | "fnt" => Ok(None),
        "brn" => Ok(Some(Status::Burn)),
        "par" => Ok(Some(Status::Paralysis)),
        "slp" => Ok(Some(Status::Sleep(0))),
        "frz" => Ok(Some(Status::Freeze)),
        "psn" => Ok(Some(Status::Poison)),
        "tox" => Ok(Some(Status::Toxic(0))),
        other => Err(format!("unknown status token {other:?}")),
    }
}

fn status_variant_eq(a: Option<Status>, b: Option<Status>) -> bool {
    use Status::*;
    matches!(
        (a, b),
        (None, None)
            | (Some(Burn), Some(Burn))
            | (Some(Paralysis), Some(Paralysis))
            | (Some(Sleep(_)), Some(Sleep(_)))
            | (Some(Freeze), Some(Freeze))
            | (Some(Poison), Some(Poison))
            | (Some(Toxic(_)), Some(Toxic(_)))
    )
}

/// Parse a chunk file into cases. Battles whose lines are malformed are returned
/// as `Err` entries (verdict `parse_error`) without sinking the whole chunk.
fn parse_chunk(data: &str) -> Vec<Result<CaseExpect, (String, String)>> {
    let mut out: Vec<Result<CaseExpect, (String, String)>> = Vec::new();
    // id -> teams, filled by TEAM lines (which precede INIT for each battle).
    let mut teams: Vec<(String, [Option<String>; 2])> = Vec::new();
    // id -> format, filled by the optional FMT row (protocol chunks only; before INIT).
    let mut formats: Vec<(String, String)> = Vec::new();
    let mut cur: Option<CaseExpect> = None;
    let mut cur_broken: Option<(String, String)> = None; // (id, why)

    let team_for = |teams: &[(String, [Option<String>; 2])], id: &str| -> Option<[String; 2]> {
        teams.iter().find(|(tid, _)| tid == id).and_then(|(_, t)| {
            match (&t[0], &t[1]) {
                (Some(a), Some(b)) => Some([a.clone(), b.clone()]),
                _ => None,
            }
        })
    };

    macro_rules! flush {
        () => {
            if let Some(broken) = cur_broken.take() {
                let _ = cur.take(); // discard the partially-built broken case
                out.push(Err(broken));
            } else if let Some(c) = cur.take() {
                out.push(Ok(c));
            }
        };
    }

    for (i, line) in data.lines().enumerate() {
        let ln = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        // While the current battle is marked broken, skip its remaining lines
        // (a new SCEN/INIT flushes it).
        let record = f[0];
        if cur_broken.is_some() && record != "SCEN" && record != "INIT" && record != "TEAM" && record != "FMT" {
            continue;
        }
        match record {
            "SCEN" => {
                if f.len() >= 2 {
                    teams.push((f[1].to_string(), [None, None]));
                }
            }
            "FMT" => {
                // FMT <id> <format> — protocol chunks only; sits before INIT.
                if f.len() == 3 {
                    formats.push((f[1].to_string(), f[2].to_string()));
                }
            }
            "TEAM" => {
                if f.len() != 4 {
                    continue;
                }
                let side = if f[2] == "p1" { 0 } else { 1 };
                if let Some((_, t)) = teams.iter_mut().find(|(tid, _)| tid == f[1]) {
                    t[side] = Some(f[3].to_string());
                }
            }
            "INIT" => {
                flush!();
                if f.len() != 4 {
                    out.push(Err(("?".to_string(), format!("INIT needs 4 fields (line {ln})"))));
                    continue;
                }
                let id = f[1].to_string();
                let fmt = formats
                    .iter()
                    .find(|(fid, _)| fid == &id)
                    .map(|(_, fm)| fm.clone())
                    .unwrap_or_else(|| "gen3customgame".to_string());
                match team_for(&teams, &id) {
                    Some(t) => {
                        cur = Some(CaseExpect {
                            id,
                            teams: t,
                            init_seed: f[2].to_string(),
                            decisions: Vec::new(),
                            ended: false,
                            winner: WinTok::None,
                            format: fmt,
                            lines: Vec::new(),
                        });
                    }
                    None => {
                        cur_broken = Some((id, format!("missing TEAM lines (line {ln})")));
                    }
                }
            }
            "DEC" => {
                let Some(c) = cur.as_mut() else { continue };
                // 36 = the pre-batch-5-floor format (saved repro dirs stay replayable);
                // 38 = the batch-5 format (+ fixedMove/batch5Move coverage flags);
                // 39 = the current batch-6 format (+ the batch6Move flag). The trailing
                // flags are e2e-gate-only (`gen_e2e_fuzz.js` emitBattle) — the replayer
                // ignores them.
                if f.len() != 36 && f.len() != 38 && f.len() != 39 {
                    cur_broken = Some((c.id.clone(), format!("DEC needs 36, 38 or 39 fields, got {} (line {ln})", f.len())));
                    continue;
                }
                let parsed: Result<DecExpect, String> = (|| {
                    let req = match f[3] {
                        "move" => ReqTok::Move,
                        "switch" => ReqTok::Switch,
                        other => return Err(format!("bad request {other:?}")),
                    };
                    let force = [f[4] == "1", f[5] == "1"];
                    let choice = [parse_choice(f[6])?, parse_choice(f[7])?];
                    let g = |i: usize| -> Result<u16, String> {
                        f[i].parse::<u16>().map_err(|e| format!("bad num f[{i}]: {e}"))
                    };
                    let gi = |i: usize| -> Result<i8, String> {
                        f[i].parse::<i8>().map_err(|e| format!("bad int f[{i}]: {e}"))
                    };
                    let side = |base: usize, spikes_i: usize| -> Result<SideExpect, String> {
                        Ok(SideExpect {
                            species: f[base].to_string(),
                            hp: g(base + 1)?,
                            maxhp: g(base + 2)?,
                            fainted: f[base + 3] == "1",
                            status: parse_status(f[base + 4])?,
                            boosts: [gi(base + 5)?, gi(base + 6)?, gi(base + 7)?, gi(base + 8)?, gi(base + 9)?],
                            confusion: g(base + 10)? as u8,
                            left: g(base + 11)? as usize,
                            spikes: g(spikes_i)? as u8,
                        })
                    };
                    Ok(DecExpect {
                        request: req,
                        force,
                        choice,
                        seed_after: f[8].to_string(),
                        p1: side(9, 34)?,
                        p2: side(21, 35)?,
                        first_mover: f[33].to_string(),
                    })
                })();
                match parsed {
                    Ok(d) => c.decisions.push(d),
                    Err(why) => cur_broken = Some((c.id.clone(), format!("{why} (line {ln})"))),
                }
            }
            "L" => {
                // L <id> <lineNo> <RAW protocol line — MAY contain tabs/pipes; LAST field>.
                // splitn(4) keeps the raw payload intact (mirrors protocol_test.rs's L-parse).
                let Some(c) = cur.as_mut() else { continue };
                let lf: Vec<&str> = line.splitn(4, '\t').collect();
                if lf.len() == 4 {
                    c.lines.push(lf[3].to_string());
                } else {
                    // A truly empty raw line ("L\t<id>\t<n>\t") splits to 3 fields — keep "".
                    c.lines.push(String::new());
                }
            }
            "END" => {
                let Some(c) = cur.as_mut() else { continue };
                if f.len() != 4 {
                    cur_broken = Some((c.id.clone(), format!("END needs 4 fields (line {ln})")));
                    continue;
                }
                c.ended = f[2] == "1";
                c.winner = match f[3] {
                    "p1" => WinTok::P1,
                    "p2" => WinTok::P2,
                    "tie" => WinTok::Tie,
                    "none" => WinTok::None,
                    other => {
                        cur_broken = Some((c.id.clone(), format!("bad winner {other:?} (line {ln})")));
                        continue;
                    }
                };
            }
            other => {
                if let Some(c) = cur.as_mut() {
                    cur_broken = Some((c.id.clone(), format!("unknown record {other:?} (line {ln})")));
                }
            }
        }
    }
    flush!();
    out
}

// ── Verdicts ─────────────────────────────────────────────────────────────────

struct Verdict {
    battle: String,
    verdict: &'static str, // "ok" | "diverged" | "panic" | "parse_error"
    kind: Option<&'static str>,
    decision: Option<usize>,
    expected: Option<String>,
    got: Option<String>,
    detail: Option<String>,
    decisions_compared: usize,
}

impl Verdict {
    fn ok(battle: &str, decisions: usize) -> Verdict {
        Verdict {
            battle: battle.to_string(),
            verdict: "ok",
            kind: None,
            decision: None,
            expected: None,
            got: None,
            detail: None,
            decisions_compared: decisions,
        }
    }
    fn diverged(
        battle: &str,
        kind: &'static str,
        decision: Option<usize>,
        expected: String,
        got: String,
        detail: String,
        decisions_compared: usize,
    ) -> Verdict {
        Verdict {
            battle: battle.to_string(),
            verdict: "diverged",
            kind: Some(kind),
            decision,
            expected: Some(expected),
            got: Some(got),
            detail: Some(detail),
            decisions_compared,
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
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
    out
}

fn emit(v: &Verdict) {
    let mut s = format!(
        "{{\"battle\":\"{}\",\"verdict\":\"{}\",\"decisions\":{}",
        json_escape(&v.battle),
        v.verdict,
        v.decisions_compared
    );
    if let Some(k) = v.kind {
        s.push_str(&format!(",\"kind\":\"{k}\""));
    }
    if let Some(d) = v.decision {
        s.push_str(&format!(",\"decision\":{d}"));
    }
    if let Some(e) = &v.expected {
        s.push_str(&format!(",\"expected\":\"{}\"", json_escape(e)));
    }
    if let Some(g) = &v.got {
        s.push_str(&format!(",\"got\":\"{}\"", json_escape(g)));
    }
    if let Some(d) = &v.detail {
        s.push_str(&format!(",\"detail\":\"{}\"", json_escape(d)));
    }
    s.push('}');
    println!("{s}");
}

// ── The replay + compare (mirrors the e2e gate's checks, verdict-returning) ──

fn species_id(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

fn replay_case(case: &CaseExpect, dex: &Dex, protocol: bool) -> Verdict {
    let opts = BattleOptions {
        // The chunk's format (default gen3customgame — no clauses → no SetStatus
        // handler-sort shuffle; `gen3ou` turns the clause shuffle ON so the state seeds
        // match a gen3ou-captured golden, and the byte path reframes the OU tier/rules).
        format_id: case.format.clone(),
        seed: Some(case.init_seed.clone()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(case.teams[0].clone()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(case.teams[1].clone()) },
    };
    // Byte mode only when requested AND the chunk actually carries captured lines.
    let byte_mode = protocol && !case.lines.is_empty();

    let mut battle = match Battle::start_with_switchins(&opts, dex) {
        Ok(b) => b,
        Err(e) => {
            return Verdict::diverged(
                &case.id,
                "start_error",
                None,
                "battle starts".to_string(),
                format!("start failed: {e}"),
                e,
                0,
            );
        }
    };

    let got_init = battle.state().unwrap().prng_seed();
    if got_init != case.init_seed {
        return Verdict::diverged(
            &case.id,
            "init_seed",
            None,
            case.init_seed.clone(),
            got_init,
            "constructed PRNG seed != sim's pre-first-decision seed".to_string(),
            0,
        );
    }

    let script: Vec<ScriptDecision> =
        case.decisions.iter().map(|d| ScriptDecision { p1: d.choice[0], p2: d.choice[1] }).collect();
    // Byte mode replays through the EMITTING engine (`run_full_battle_logged` — same
    // bit-for-bit `run_full_battle` + the PRNG-free protocol side-output), so the SAME
    // state/seed compare runs first (a draw bug still reports `seed`, the root cause),
    // then the byte-diff surfaces an emission-only divergence as `protocol`.
    let (outcome, emitted): (BattleOutcome, Vec<String>) = if byte_mode {
        let (o, lines) = battle.state_mut().unwrap().run_full_battle_logged(&script, dex);
        // gen3ou: rewrite the gen3customgame framing tier/rules to OU before diffing.
        let raw: Vec<String> = if case.format.contains("gen3ou") {
            pokesim::bridge::reframe(&lines, &case.format)
        } else {
            lines.into_iter().map(|l| l.0).collect()
        };
        (o, raw)
    } else {
        (battle.state_mut().unwrap().run_full_battle(&script, dex), Vec::new())
    };

    let n = outcome.decisions.len().min(case.decisions.len());
    for di in 0..n {
        let rec = &outcome.decisions[di];
        let exp = &case.decisions[di];

        // Priority 1: the SEED (a draw bug dominates any downstream state noise).
        if rec.seed_after != exp.seed_after {
            return Verdict::diverged(
                &case.id,
                "seed",
                Some(di),
                exp.seed_after.clone(),
                rec.seed_after.clone(),
                "a draw is mis-ordered/missing/extra (RNG-consumption divergence)".to_string(),
                di,
            );
        }
        // Priority 2: the request boundary kind.
        let req_ok = match (&rec.request, exp.request) {
            (RequestKind::Move, ReqTok::Move) => true,
            (RequestKind::ForceSwitch { force }, ReqTok::Switch) => *force == exp.force,
            _ => false,
        };
        if !req_ok {
            return Verdict::diverged(
                &case.id,
                "request",
                Some(di),
                format!("{:?} force={:?}", exp.request, exp.force),
                format!("{:?}", rec.request),
                "move-vs-forceSwitch boundary mismatch".to_string(),
                di,
            );
        }
        // Priority 3+: per-side snapshots.
        for (idx, (snap, e, sp)) in [
            (0usize, (&rec.active[0], &exp.p1, &rec.active_species[0])),
            (1usize, (&rec.active[1], &exp.p2, &rec.active_species[1])),
        ] {
            if species_id(sp) != species_id(&e.species) {
                return Verdict::diverged(
                    &case.id,
                    "species",
                    Some(di),
                    e.species.clone(),
                    sp.clone(),
                    format!("side {idx} active species mismatch"),
                    di,
                );
            }
            let left = rec.pokemon_left[idx];
            if snap.hp != e.hp || snap.maxhp != e.maxhp || snap.fainted != e.fainted || left != e.left {
                return Verdict::diverged(
                    &case.id,
                    "state",
                    Some(di),
                    format!("hp={}/{} fainted={} left={}", e.hp, e.maxhp, e.fainted, e.left),
                    format!("hp={}/{} fainted={} left={}", snap.hp, snap.maxhp, snap.fainted, left),
                    format!("side {idx} hp/maxhp/fainted/pokemon_left mismatch (seed matches)"),
                    di,
                );
            }
            if !e.fainted {
                if !status_variant_eq(snap.status, e.status) {
                    return Verdict::diverged(
                        &case.id,
                        "status",
                        Some(di),
                        format!("{:?}", e.status),
                        format!("{:?}", snap.status),
                        format!("side {idx} major-status mismatch (seed matches)"),
                        di,
                    );
                }
                if snap.boosts[0..5] != e.boosts[..] {
                    return Verdict::diverged(
                        &case.id,
                        "boost",
                        Some(di),
                        format!("{:?}", e.boosts),
                        format!("{:?}", &snap.boosts[0..5]),
                        format!("side {idx} stat-stage mismatch (seed matches)"),
                        di,
                    );
                }
                let rust_conf = snap.confusion.unwrap_or(0);
                if rust_conf != e.confusion {
                    return Verdict::diverged(
                        &case.id,
                        "confusion",
                        Some(di),
                        e.confusion.to_string(),
                        rust_conf.to_string(),
                        format!("side {idx} confusion-counter mismatch (seed matches)"),
                        di,
                    );
                }
            }
            if rec.spikes[idx] != e.spikes {
                return Verdict::diverged(
                    &case.id,
                    "spikes",
                    Some(di),
                    e.spikes.to_string(),
                    rec.spikes[idx].to_string(),
                    format!("side {idx} spikes-layer mismatch (seed matches)"),
                    di,
                );
            }
        }
        // First mover (move turns; the golden may say "none" for a no-action turn).
        if exp.request == ReqTok::Move {
            let sim_first: Option<usize> = match exp.first_mover.as_str() {
                "p1" => Some(0),
                "p2" => Some(1),
                _ => None,
            };
            if sim_first.is_some() && rec.first_mover != sim_first {
                return Verdict::diverged(
                    &case.id,
                    "firstmover",
                    Some(di),
                    exp.first_mover.clone(),
                    format!("{:?}", rec.first_mover),
                    "wrong first mover on a move turn (seed matches)".to_string(),
                    di,
                );
            }
        }
    }

    if outcome.decisions.len() != case.decisions.len() {
        return Verdict::diverged(
            &case.id,
            "decision_count",
            Some(n),
            case.decisions.len().to_string(),
            outcome.decisions.len().to_string(),
            "decision-count mismatch (all common decisions matched)".to_string(),
            n,
        );
    }

    if outcome.ended != case.ended {
        return Verdict::diverged(
            &case.id,
            "ended",
            None,
            case.ended.to_string(),
            outcome.ended.to_string(),
            "game-end flag mismatch".to_string(),
            n,
        );
    }
    let rust_win = match outcome.winner {
        Some(0) => WinTok::P1,
        Some(1) => WinTok::P2,
        Some(_) => WinTok::None,
        None if outcome.ended => WinTok::Tie,
        None => WinTok::None,
    };
    if rust_win != case.winner {
        return Verdict::diverged(
            &case.id,
            "winner",
            None,
            format!("{:?}", case.winner),
            format!("{:?}", rust_win),
            "final winner mismatch".to_string(),
            n,
        );
    }

    // ── The BYTE differential (protocol mode) ────────────────────────────────
    // State + seed + winner all matched → the engine is bit-for-bit; now diff the
    // literal OMNISCIENT `|...|` bytes. Both sides run through the identical DENYLIST
    // filter (drop debug/error, normalize |t:|), then a first-divergence line diff
    // (the protocol_test.rs:343-363 loop pattern). Reports a NEW `protocol` kind.
    if byte_mode {
        let golden = filter_bytes(&case.lines);
        let engine = filter_bytes(&emitted);
        let m = golden.len().max(engine.len());
        for i in 0..m {
            let g = golden.get(i).map(String::as_str);
            let e = engine.get(i).map(String::as_str);
            if g != e {
                let lo = i.saturating_sub(2);
                let ctx = |v: &[String]| -> String {
                    (lo..(i + 2).min(v.len())).map(|k| format!("[{k}] {}", v[k])).collect::<Vec<_>>().join(" ¦ ")
                };
                return Verdict::diverged(
                    &case.id,
                    "protocol",
                    Some(i),
                    g.unwrap_or("<none>").to_string(),
                    e.unwrap_or("<none>").to_string(),
                    format!(
                        "byte divergence at filtered line {i} (fmt={}) | golden ctx: {} | engine ctx: {}",
                        case.format, ctx(&golden), ctx(&engine)
                    ),
                    n,
                );
            }
        }
    }

    Verdict::ok(&case.id, n)
}

// ── Panic capture (the engine fail-louds on unmodeled mechanics) ─────────────

thread_local! {
    static PANIC_MSG: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn install_panic_capture() {
    panic::set_hook(Box::new(|info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "non-string panic payload".to_string()
        };
        let loc = info.location().map(|l| format!(" at {}:{}", l.file(), l.line())).unwrap_or_default();
        PANIC_MSG.with(|m| *m.borrow_mut() = Some(format!("{msg}{loc}")));
    }));
}

fn take_panic_msg() -> String {
    PANIC_MSG.with(|m| m.borrow_mut().take()).unwrap_or_else(|| "unknown panic".to_string())
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    // `--protocol` turns on the omniscient BYTE differential (gen3_omniscient_byte_fuzz_v1):
    // replay through `run_full_battle_logged` + diff the captured `|...|` lines. It also
    // auto-enables per-case when the chunk carries L-rows, so a saved repro dir replays in
    // byte mode automatically. Also honored via POKESIM_AB_PROTOCOL=1.
    let flag_protocol = raw_args.iter().any(|a| a == "--protocol")
        || std::env::var("POKESIM_AB_PROTOCOL").map(|v| v == "1").unwrap_or(false);
    let args: Vec<String> = raw_args.into_iter().filter(|a| !a.starts_with("--")).collect();
    if args.is_empty() {
        eprintln!("usage: ab_replay [--protocol] <chunk-file | repro-dir> [...]");
        std::process::exit(2);
    }
    install_panic_capture();
    let dex = Dex::for_gen(3);

    let mut ok = 0usize;
    let mut diverged = 0usize;
    let mut panicked = 0usize;
    let mut parse_errors = 0usize;

    for arg in &args {
        // A repro dir holds its single-battle chunk as battle.txt.
        let mut path = std::path::PathBuf::from(arg);
        if path.is_dir() {
            path = path.join("battle.txt");
        }
        let data = match std::fs::read_to_string(&path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("cannot read {}: {e}", path.display());
                std::process::exit(2);
            }
        };
        for entry in parse_chunk(&data) {
            match entry {
                Err((id, why)) => {
                    parse_errors += 1;
                    let v = Verdict {
                        battle: id,
                        verdict: "parse_error",
                        kind: None,
                        decision: None,
                        expected: None,
                        got: None,
                        detail: Some(why),
                        decisions_compared: 0,
                    };
                    emit(&v);
                }
                Ok(case) => {
                    // Auto-enable byte mode when the chunk carries L-rows (a saved repro
                    // dir), else honor the explicit --protocol flag.
                    let proto = flag_protocol || !case.lines.is_empty();
                    let result = panic::catch_unwind(AssertUnwindSafe(|| replay_case(&case, &dex, proto)));
                    let v = match result {
                        Ok(v) => v,
                        Err(_) => {
                            let msg = take_panic_msg();
                            Verdict {
                                battle: case.id.clone(),
                                verdict: "panic",
                                kind: Some("panic"),
                                decision: None,
                                expected: None,
                                got: None,
                                detail: Some(msg),
                                decisions_compared: 0,
                            }
                        }
                    };
                    match v.verdict {
                        "ok" => ok += 1,
                        "panic" => panicked += 1,
                        _ => diverged += 1,
                    }
                    emit(&v);
                }
            }
        }
    }

    println!(
        "{{\"chunk_summary\":{{\"ok\":{ok},\"diverged\":{diverged},\"panic\":{panicked},\"parse_error\":{parse_errors}}}}}"
    );
}
