//! `sim_bridge` — a drop-in Rust replacement for
//! `src/utils/bridge/local_sim_bridge.js`, speaking the EXACT stdin/stdout protocol so
//! it can run behind the Python bridge (`local_battle_runner.py`) with zero protocol
//! change: replace `node local_sim_bridge.js` with this binary.
//!
//! # Protocol (mirrors `local_sim_bridge.js` byte-for-byte)
//!
//! stdin (newline-delimited commands):
//! - `START <json>`  `{formatid, seed?, persistent?, resumeReseed?, p1:{name,team}, p2:{name,team}}`
//! - `CHOOSE <side> <choice>`  e.g. `CHOOSE p1 move 1` / `CHOOSE p2 switch 3`
//! - `FORCELOSE <side>`  e.g. `FORCELOSE p1`  (poke-env `/forfeit` path)
//! - `END`  tear down and exit
//!
//! stdout (newline-delimited frames):
//! - `p1 <base64(chunk)>` / `p2 <base64(chunk)>`  one protocol chunk that side saw
//! - `__END__`  battle over, both side streams closed (persistent → reset for the next START)
//! - `__ERR__ <base64(msg)>`  fatal error
//! - `__RECON__ <base64(json)>`  the reconstruction record — **DEFERRED** (see below)
//!
//! Base64 per chunk because protocol text contains `\n`, `|`, and arbitrary JSON in
//! `|request|` — one stdout line == exactly one side-tagged chunk, so the Python side
//! can demux unambiguously.
//!
//! # Emission engine — replay-from-genesis (same pattern as `BattleStream::write_line`)
//!
//! The bridge does NOT keep a paused incremental engine. It accumulates the one-sided
//! `CHOOSE` command stream and, on every START/CHOOSE, re-runs
//! [`run_full_battle_bridge_chunked`] over the whole accumulated stream, then emits the
//! NEW per-side CHUNK suffix (past what has already been written) as `pN <base64>` lines.
//! Deterministic (the engine is bit-for-bit → each replay reproduces every prior chunk
//! byte-identically) and cheap at bridge scale. The chunk boundaries + the HP-privacy
//! fold + the `|request|` frames + the trapped state machine are all produced by the
//! shared [`crate::bridge`] emitter (byte-gated by `tests/bridge_test.rs` at line level,
//! and by `harness/gen_sim_bridge_diff.js` at the chunk/stdout level vs the real Node
//! bridge).
//!
//! # DEFERRED — honest scope
//!
//! - **`__RECON__`** (the reconstruction record: `{v, format_id, prng_seed, input_log,
//!   commands}`) serves the search / counterfactual layer, NOT core training/eval. The
//!   Node version dumps the sim's internal `battle.inputLog`; the port has no
//!   byte-identical `input_log` (a separate concern). So this binary emits **NO
//!   `__RECON__`**. The Python side degrades gracefully — `local_battle_runner.py`'s
//!   `_offer_recon` swallows any capture failure and the demux simply never sees the
//!   frame, so no `__RECON__` is a no-op for the core path (only the forensic
//!   reconstruction sibling is affected). We do NOT fake `__RECON__` content parity.
//! - **`resumeReseed`** (`{turn, seed}` — swap the battle's PRNG at the start of a
//!   divergence turn for the counterfactual Monte-Carlo re-roll) needs a mid-battle
//!   `Battle::reseed`, which the replay-from-genesis driver has no hook for (and
//!   `Battle::reseed` is still `todo!()`). It is IGNORED with a one-line stderr note; a
//!   `START` that carries it runs the ordinary battle under its base seed. Again this
//!   serves the search layer only, never core training/eval.

use std::io::{self, BufRead, Write};

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{
    advance_seed_for_construction, parse_choice, run_full_battle_bridge_chunked_ended, Cmd,
};
use pokesim::dex::Dex;
use pokesim::json::Json;

fn main() {
    let dex = Dex::for_gen(3);
    let mut sess = Session::new();

    let stdin = io::stdin();
    let mut out = io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // stdin closed / read error → exit (mirrors `stdin.on('end')`)
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Crash-don't-drop: a handler panic becomes `__ERR__`, matching the Node bridge's
        // `uncaughtException` handler (which does NOT exit on a per-line failure unless
        // the panic is unrecoverable — here we report and keep the loop alive).
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handle_line(&mut sess, line, &dex, &mut out)
        }));
        match res {
            Ok(Ok(LineResult::Continue)) => {}
            Ok(Ok(LineResult::Exit)) => return,
            Ok(Err(msg)) => emit_err(&mut out, &msg),
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic".to_string());
                emit_err(&mut out, &msg);
            }
        }
    }
}

enum LineResult {
    Continue,
    Exit,
}

/// One live bridge session — the accumulated battle setup + command stream, plus how
/// many chunks have already been flushed to stdout (the replay-suffix cursor).
struct Session {
    /// Sticky: once any START asks for it, the process survives battle ends.
    persistent: bool,
    /// The current battle's setup (`None` between battles in persistent mode).
    setup: Option<Setup>,
    /// The accumulated `CHOOSE` commands, in processing order.
    cmds: Vec<Cmd>,
    /// How many chunks of the concatenated per-side stream have been emitted.
    emitted: usize,
    /// Whether the current battle has already emitted `__END__` (guard).
    ended: bool,
}

struct Setup {
    format_id: String,
    seed: Option<String>,
    p1: PlayerOptions,
    p2: PlayerOptions,
}

impl Session {
    fn new() -> Self {
        Session {
            persistent: false,
            setup: None,
            cmds: Vec::new(),
            emitted: 0,
            ended: false,
        }
    }

    /// Reset for the next battle on the SAME process (persistent mode) — a fresh START
    /// rebuilds a clean battle (mirrors the Node bridge's `streams = null; …` reset).
    fn reset(&mut self) {
        self.setup = None;
        self.cmds.clear();
        self.emitted = 0;
        self.ended = false;
    }

    fn opts(&self) -> Result<BattleOptions, String> {
        let s = self.setup.as_ref().ok_or("no battle in progress (missing START)")?;
        Ok(BattleOptions {
            format_id: s.format_id.clone(),
            seed: s.seed.clone(),
            p1: s.p1.clone(),
            p2: s.p2.clone(),
        })
    }
}

fn handle_line(
    sess: &mut Session,
    line: &str,
    dex: &Dex,
    out: &mut impl Write,
) -> Result<LineResult, String> {
    let (cmd, rest) = match line.find(' ') {
        Some(i) => (&line[..i], &line[i + 1..]),
        None => (line, ""),
    };
    match cmd {
        "START" => {
            handle_start(sess, rest)?;
            // A fresh START emits the framing chunks + the initial request(s).
            flush_new_chunks(sess, dex, out)?;
            Ok(LineResult::Continue)
        }
        "CHOOSE" => {
            handle_choose(sess, rest)?;
            flush_new_chunks(sess, dex, out)?;
            Ok(LineResult::Continue)
        }
        "FORCELOSE" => {
            handle_forcelose(sess, rest.trim(), dex, out)?;
            Ok(LineResult::Continue)
        }
        "END" => Ok(LineResult::Exit),
        other => Err(format!("unknown command: {other}")),
    }
}

/// `START <json>` — parse `{formatid, seed?, persistent?, resumeReseed?, p1, p2}` and
/// build a fresh session (a fresh battle in persistent mode).
fn handle_start(sess: &mut Session, json: &str) -> Result<(), String> {
    let v = Json::parse(json).map_err(|e| format!("START JSON: {e}"))?;
    if v.get("persistent").and_then(|p| p.as_bool()).unwrap_or(false) {
        sess.persistent = true;
    }
    if v.get("resumeReseed").map(|r| !r.is_null()).unwrap_or(false) {
        // DEFERRED — the replay-from-genesis driver has no mid-battle reseed hook (see
        // the module docs). Note it once; run the ordinary battle under the base seed.
        eprintln!(
            "[sim_bridge] resumeReseed is not supported (search-layer only) — ignoring; \
             running under the base seed"
        );
    }
    let format_id = v
        .str_at("formatid")
        .ok_or("START: missing formatid")?
        .to_string();
    // A given `>start` seed is the RAW seed; advance it by the sim's turn-0 construction
    // draw (the Quick Claw) so the port's draw-free replay-from-genesis lines up with the
    // real sim's post-`>start` PRNG state — the "pre-first-decision seed convention" the
    // engine's own draw suites use. `None` seed → the port picks its default (no reference).
    let seed = v.get("seed").and_then(|s| s.as_array()).map(|a| {
        let raw = a
            .iter()
            .map(|x| format!("{}", x.as_f64().unwrap_or(0.0) as u64))
            .collect::<Vec<_>>()
            .join(",");
        advance_seed_for_construction(&raw)
    });
    let p1 = parse_player(&v, "p1")?;
    let p2 = parse_player(&v, "p2")?;

    // A new battle: reset the per-battle state (persistent keeps `persistent`).
    sess.reset();
    sess.setup = Some(Setup { format_id, seed, p1, p2 });
    Ok(())
}

fn parse_player(v: &Json, key: &str) -> Result<PlayerOptions, String> {
    let p = v.get(key).ok_or_else(|| format!("START: missing {key}"))?;
    let name = p
        .str_at("name")
        .ok_or_else(|| format!("START: {key}.name"))?
        .to_string();
    let team = p
        .str_at("team")
        .ok_or_else(|| format!("START: {key}.team"))?
        .to_string();
    Ok(PlayerOptions { name, team: PackedTeam(team) })
}

/// `CHOOSE <side> <choice>` — accumulate the one-sided command. An unknown side / choice
/// token is a hard error (a malformed driver). If no battle is live it is dropped
/// silently (mirrors the Node bridge's `if (streams && streams[side])` guard).
fn handle_choose(sess: &mut Session, rest: &str) -> Result<(), String> {
    if sess.setup.is_none() || sess.ended {
        // No live battle (mirrors the Node bridge's guard: a stray CHOOSE is ignored).
        return Ok(());
    }
    let (side_tok, choice_tok) = match rest.find(' ') {
        Some(i) => (&rest[..i], &rest[i + 1..]),
        None => (rest, ""),
    };
    let side = match side_tok {
        "p1" => 0usize,
        "p2" => 1usize,
        other => return Err(format!("CHOOSE: bad side {other:?}")),
    };
    let choice = parse_choice(choice_tok)
        .ok_or_else(|| format!("CHOOSE: unsupported choice {choice_tok:?}"))?;
    sess.cmds.push(Cmd { side, choice });
    Ok(())
}

/// `FORCELOSE <side>` — the poke-env `/forfeit` path. The port has no `>forcelose`
/// engine hook (`run_full_battle` plays a scripted battle to its natural end), so this
/// bridge treats a forfeit as an immediate battle end: it flushes whatever chunks the
/// accumulated stream produced so far, then emits `__END__` (+ persistent reset). This
/// keeps the Python side (which only needs the battle to TERMINATE on a forfeit) working
/// — the forfeiting side simply gets no further protocol.
fn handle_forcelose(
    sess: &mut Session,
    _side: &str,
    dex: &Dex,
    out: &mut impl Write,
) -> Result<(), String> {
    if sess.setup.is_none() || sess.ended {
        return Ok(());
    }
    // Flush any pending chunks from the stream so far, then terminate.
    flush_new_chunks(sess, dex, out)?;
    end_battle(sess, out);
    Ok(())
}

/// Re-run the chunked emitter over the accumulated command stream and write the NEW
/// per-side chunk suffix as `pN <base64>` lines. Emits `__END__` when the battle ends.
fn flush_new_chunks(sess: &mut Session, dex: &Dex, out: &mut impl Write) -> Result<(), String> {
    if sess.ended {
        return Ok(());
    }
    let opts = sess.opts()?;
    let (chunks, ended) = run_full_battle_bridge_chunked_ended(&opts, &sess.cmds, dex)?;
    // Emit every chunk past the cursor, in flush order (both sides interleaved).
    for c in chunks.chunks.iter().skip(sess.emitted) {
        emit_chunk(out, c.side, &c.lines);
    }
    sess.emitted = chunks.chunks.len();

    if ended {
        end_battle(sess, out);
    }
    Ok(())
}

/// Emit `__END__` (+ persistent reset / non-persistent exit-arm). In persistent mode the
/// next `START` rebuilds a fresh battle; otherwise the process would exit — but this
/// binary keeps its loop alive and simply refuses further work until the next START/END
/// (the Python non-persistent path sends `END` right after `__END__`).
fn end_battle(sess: &mut Session, out: &mut impl Write) {
    if sess.ended {
        return;
    }
    sess.ended = true;
    writeln!(out, "__END__").ok();
    out.flush().ok();
    if sess.persistent {
        sess.reset();
    }
}

// ===========================================================================
// stdout framing.
// ===========================================================================

fn emit_chunk(out: &mut impl Write, side: usize, lines: &[String]) {
    let tag = if side == 0 { "p1" } else { "p2" };
    let payload = lines.join("\n");
    let b64 = base64_encode(payload.as_bytes());
    writeln!(out, "{tag} {b64}").ok();
    out.flush().ok();
}

fn emit_err(out: &mut impl Write, msg: &str) {
    let b64 = base64_encode(msg.as_bytes());
    writeln!(out, "__ERR__ {b64}").ok();
    out.flush().ok();
}

// ===========================================================================
// Base64 (std-only; standard alphabet, `=` padding — matches Node's Buffer.toString).
// ===========================================================================

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64[((n >> 18) & 0x3f) as usize] as char);
        out.push(B64[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}
