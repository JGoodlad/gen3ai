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
//! - `__RECON__ <base64(json)>`  the reconstruction record, once per battle, just before
//!   `__END__` (see [`emit_recon`] for the honest scope of its `input_log`)
//!
//! # Seeds (`gen3_bridge_seed_forms_v1` / `gen3_bridge_seedless_fixed_seed_v1`)
//!
//! `seed` and `resumeReseed.seed` accept EVERY form `new PRNG()` does — `[m,n,o,p]`,
//! `"m,n,o,p"`, `"gen5,<hex16>"`, `"sodium,<hex>"` — via the one [`parse_seed_field`]
//! parser. A seed that is PRESENT but unusable is a LOUD `__ERR__`, never a fall-back.
//! NO seed at all MINTS a fresh `sodium,<hex>` ([`pokesim::prng::Prng::generate_seed`]),
//! exactly like the sim's own `PRNG` constructor.
//!
//! Base64 per chunk because protocol text contains `\n`, `|`, and arbitrary JSON in
//! `|request|` — one stdout line == exactly one side-tagged chunk, so the Python side
//! can demux unambiguously.
//!
//! # Emission engine — TRULY INCREMENTAL (`gen3_bridge_incremental_replay_v1`)
//!
//! The session holds ONE PERSISTENT live-battle [`pokesim::bridge::BridgeSession`] across
//! CHOOSE lines. On START it builds the battle + emits the framing + the first request; on
//! each CHOOSE it feeds ONE command to the session, which advances the live battle to its
//! next request boundary and appends ONLY the new per-side chunk(s) — **O(1) per CHOOSE,
//! O(N) per battle, no re-simulation of a prior turn EVER**. (The previous design
//! re-ran the whole accumulated stream from genesis on every CHOOSE — O(N²)/battle — which
//! WEDGED long staller battles under `--use-bridge=rust`: each late step took minutes and
//! the `SubprocVecEnv` step-barrier deadlocked.) The chunk boundaries + the HP-privacy fold
//! + the `|request|` frames + the trapped state machine are all produced by the shared
//! [`pokesim::bridge`] emitter (byte-gated by `tests/bridge_test.rs` at line level, a
//! bit-for-bit parity test vs the genesis-replay reference core in `tests/bridge_test.rs`,
//! and by `harness/gen_sim_bridge_diff.js` at the chunk/stdout level vs the real Node bridge).
//!
//! # SUPPORTED — honest scope
//!
//! - **`__RECON__`** (the reconstruction record: `{v, format_id, prng_seed, input_log,
//!   commands}`) serves the search / counterfactual layer. It IS emitted, once per battle,
//!   as of `gen3_bridge_recon_record_v1` — and, since `gen3_bridge_seedless_fixed_seed_v1`,
//!   on a SEEDLESS battle too (the resolved seed is the minted one). The Node version dumps
//!   the sim's internal `battle.inputLog`; the port renders the COMMITTED-choice lines from
//!   its own script, so `input_log` is replay-EQUIVALENT rather than guaranteed
//!   byte-identical for exotic wire spellings. We do NOT fake `__RECON__` content parity —
//!   see [`emit_recon`].
//! - **`resumeReseed`** (`{turn, seed}` — swap the battle's PRNG at the start of a divergence
//!   turn for the counterfactual Monte-Carlo re-roll) is **SUPPORTED** as of
//!   `gen3_bridge_resume_reseed_v1`, via `BridgeSession::{turn, reseed}` (the incremental
//!   session owns a live `BattleState` whose `prng` is a plain field, so the bridge's need did
//!   not have to wait on the clone-and-branch snapshot surface — which is now simply `Clone`,
//!   via `BridgeSession::snapshot`). Applied at the START of the
//!   divergence turn, BEFORE its choices commit, ONCE — mirroring the node bridge — so the
//!   recorded prefix keeps its dice and only the post-divergence resolution re-rolls. A
//!   malformed spec is a hard error, not a silent fall-back to the base seed — a counterfactual
//!   that quietly answers the WRONG question is worse than one that fails.

use std::io::{self, BufRead, Write};

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::prng::{normalize_seed, Prng};

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

/// One live bridge session — the PERSISTENT incremental battle engine
/// ([`BridgeSession`], the live `Battle` + stepping primitive), plus how many chunks
/// have already been flushed to stdout (the emit cursor). `None` between battles in
/// persistent mode.
struct Session {
    /// Sticky: once any START asks for it, the process survives battle ends.
    persistent: bool,
    /// The live incremental battle engine (`None` before START / between battles).
    bridge: Option<BridgeSession>,
    /// How many chunks of the session's per-side stream have been emitted.
    emitted: usize,
    /// Whether the current battle has already emitted `__END__` (guard).
    ended: bool,
    /// Counterfactual Monte-Carlo re-roll (`gen3_bridge_resume_reseed_v1`): the
    /// `{turn, seed}` from START, and the once-latch that mirrors the node bridge's
    /// `reseeded`. Cleared per battle.
    resume_reseed: Option<(u32, String)>,
    reseeded: bool,
    /// The reconstruction record's inputs (`gen3_bridge_recon_record_v1`), captured at
    /// START and as CHOOSE lines arrive. Cleared per battle by `reset`.
    recon: Recon,
}

/// The `__RECON__` record's raw materials. See `emit_recon`.
#[derive(Default)]
struct Recon {
    format_id: String,
    /// The RESOLVED `>start` seed as a `Prng::new`-acceptable string — the caller's
    /// (`"1,2,3,4"` / `"gen5,…"` / `"sodium,…"`, an array form comma-joined) or, when the
    /// caller supplied none, the one this process MINTED. Never empty
    /// (`gen3_bridge_seedless_fixed_seed_v1`), so every battle has a replayable record.
    seed: String,
    /// The two `>player` payloads exactly as received, so the record round-trips.
    p1_json: String,
    p2_json: String,
    /// Every CHOOSE the child processed, in order — INCLUDING attempts the engine
    /// refused. Protocol-faithful (`commands`).
    cmds: Vec<(String, String)>,
    /// Whether this battle already emitted its record (once per battle).
    emitted: bool,
}

impl Session {
    fn new() -> Self {
        Session {
            persistent: false,
            resume_reseed: None,
            reseeded: false,
            recon: Recon::default(),
            bridge: None,
            emitted: 0,
            ended: false,
        }
    }

    /// Reset for the next battle on the SAME process (persistent mode) — a fresh START
    /// rebuilds a clean battle (mirrors the Node bridge's `streams = null; …` reset).
    fn reset(&mut self) {
        self.bridge = None;
        self.emitted = 0;
        self.ended = false;
        // A persistent child must NOT leak one battle's record into the next.
        self.recon = Recon::default();
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
            // A fresh START builds the live session (framing + first request into its chunks).
            handle_start(sess, rest, dex)?;
            flush_new_chunks(sess, out)?;
            Ok(LineResult::Continue)
        }
        "CHOOSE" => {
            handle_choose(sess, rest, dex)?;
            // A FATAL session condition (today: a NAME-form choice that resolves against
            // nothing) must become a LOUD `__ERR__`, never an endless re-request spin —
            // `gen3_bridge_unresolvable_choice_failloud_v1`.
            if let Some(msg) = sess.bridge.as_ref().and_then(|b| b.fatal()) {
                return Err(msg.to_string());
            }
            flush_new_chunks(sess, out)?;
            Ok(LineResult::Continue)
        }
        "FORCELOSE" => {
            handle_forcelose(sess, rest.trim(), out)?;
            Ok(LineResult::Continue)
        }
        "END" => Ok(LineResult::Exit),
        other => Err(format!("unknown command: {other}")),
    }
}

/// `START <json>` — parse `{formatid, seed?, persistent?, resumeReseed?, p1, p2}` and
/// BUILD a fresh live [`BridgeSession`] (the framing + first request are emitted into its
/// chunks, flushed by the caller). A fresh battle in persistent mode.
fn handle_start(sess: &mut Session, json: &str, dex: &Dex) -> Result<(), String> {
    let v = Json::parse(json).map_err(|e| format!("START JSON: {e}"))?;
    if v.get("persistent").and_then(|p| p.as_bool()).unwrap_or(false) {
        sess.persistent = true;
    }
    // --- resumeReseed (`gen3_bridge_resume_reseed_v1`) — the counterfactual Monte-Carlo
    //     re-roll, now SUPPORTED (it used to be parsed, warned about, and ignored). `{turn,
    //     seed}`: at the START of `turn`, BEFORE that turn's choices commit, the live battle's
    //     PRNG is swapped for a fresh one — so the recorded PREFIX keeps its original dice and
    //     only the post-divergence resolution re-rolls. Applied in `handle_choose`; both the
    //     spec and the latch reset per battle so a PERSISTENT child re-arms on the next START
    //     (and, just as important, does NOT inherit a previous battle's reseed). ---
    sess.resume_reseed = None;
    sess.reseeded = false;
    if let Some(rr) = v.get("resumeReseed").filter(|r| !r.is_null()) {
        let turn = rr.get("turn").and_then(|t| t.as_f64()).map(|t| t as u32);
        // EVERY form `new PRNG()` accepts — the sole producer
        // (`main.prober.falsifier.fresh_seeds`) emits the `"a,b,c,d"` STRING form, which an
        // array-only parse rejected outright (`gen3_bridge_seed_forms_v1`).
        let seed = parse_seed_field(rr.get("seed"), "resumeReseed.seed")?;
        match (turn, seed) {
            (Some(t), Some(sd)) => sess.resume_reseed = Some((t, sd)),
            // FAIL-LOUD on a malformed spec rather than silently running under the base seed:
            // a counterfactual that quietly answers the WRONG question is worse than an error.
            _ => return Err("START: resumeReseed needs both `turn` and `seed`".to_string()),
        }
    }
    let format_id = v
        .str_at("formatid")
        .ok_or("START: missing formatid")?
        .to_string();
    // A given `>start` seed is the RAW seed, passed THROUGH unmodified: the turn-0
    // CONSTRUCTION WINDOW is now modeled in the engine (`gen3_turn0_construction_v1`,
    // via `BridgeSession::new_construct_turn0` below), so the port reproduces the sim's
    // gender samples + speed-tie shuffles + Quick Claw from the raw seed bit-for-bit
    // (the old pure `advance_seed_for_construction` seed hack modeled ONLY the Quick
    // Claw → it desynced a speed-TIED lead or an unspecified-gender mon).
    //
    // NO seed → MINT one (`gen3_bridge_seedless_fixed_seed_v1`), exactly like Showdown's
    // `PRNG` constructor (`if (!seed) seed = PRNG.generateSeed()`). This used to fall
    // through to the engine's `DEFAULT_CONSTRUCT_SEED` ("0,0,0,0"), so EVERY seedless
    // battle — i.e. every training episode and every eval game, since neither
    // `bridge_session` nor `run_local_battles` passes a seed — replayed ONE dice stream,
    // identically across every parallel env worker. Resolving it HERE (rather than letting
    // the engine default) also means the record always has a real `prng_seed`, so
    // `__RECON__` is emitted for a seedless battle too.
    let seed = parse_seed_field(v.get("seed"), "seed")?.unwrap_or_else(Prng::generate_seed);
    let seed = Some(seed);
    let p1 = parse_player(&v, "p1")?;
    let p2 = parse_player(&v, "p2")?;

    // A new battle: reset the per-battle state (persistent keeps `persistent`), then build
    // the live incremental engine (advances to + emits the first request boundary).
    sess.reset();
    // Capture the record's materials (`gen3_bridge_recon_record_v1`). The `>player` payloads
    // are re-serialized from the parsed values rather than sliced out of the raw START JSON,
    // so the record is well-formed regardless of the caller's spacing/key order.
    sess.recon = Recon {
        format_id: format_id.clone(),
        seed: seed.clone().unwrap_or_default(),
        p1_json: format!(
            "{{\"name\":{},\"team\":{}}}",
            json_quote(&p1.name),
            json_quote(&p1.team.0)
        ),
        p2_json: format!(
            "{{\"name\":{},\"team\":{}}}",
            json_quote(&p2.name),
            json_quote(&p2.team.0)
        ),
        cmds: Vec::new(),
        emitted: false,
    };
    let opts = BattleOptions { format_id, seed, p1, p2 };
    sess.bridge = Some(BridgeSession::new_construct_turn0(&opts, dex)?);
    Ok(())
}

/// Parse a START seed field into a [`Prng::new`]-acceptable seed string
/// (`gen3_bridge_seed_forms_v1`). The ONE seed parser for both `seed` and
/// `resumeReseed.seed`.
///
/// Accepts every form the Node bridge does, because `local_sim_bridge.js` hands
/// `msg.seed` to the sim verbatim and `new PRNG(seed)` takes:
///   * `[m,n,o,p]` — a JSON array (`PRNG`: `if (Array.isArray(seed)) seed = seed.join(",")`);
///   * `"m,n,o,p"` / `"gen5,<hex16>"` / `"sodium,<hex>"` — a string (`setSeed`'s three cases).
/// Absent / `null` → `Ok(None)` (the caller decides: mint for `seed`, error for
/// `resumeReseed`).
///
/// FAILS LOUD on a seed that is PRESENT but unusable — a number, a bool, a non-numeric
/// array element, an unrecognized string. That is the whole point: the pre-fix parser
/// read only the array form and silently `None`d everything else, so a string seed ran a
/// DIFFERENT battle than the caller asked for and nothing said so. A wrong input that is
/// silently accepted is worse than a crash.
fn parse_seed_field(field: Option<&Json>, what: &str) -> Result<Option<String>, String> {
    let Some(j) = field.filter(|s| !s.is_null()) else {
        return Ok(None);
    };
    let raw = if let Some(a) = j.as_array() {
        let mut parts = Vec::with_capacity(a.len());
        for x in a {
            let n = x
                .as_f64()
                .ok_or_else(|| format!("START: `{what}` array holds a non-numeric element"))?;
            parts.push(format!("{}", n as u64));
        }
        parts.join(",")
    } else if let Some(s) = j.as_str() {
        normalize_seed(s)
    } else {
        return Err(format!(
            "START: `{what}` must be a seed string or a number array, got {j:?}"
        ));
    };
    Prng::validate_seed(&raw).map_err(|e| format!("START: `{what}` — {e}"))?;
    Ok(Some(raw))
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

/// `CHOOSE <side> <choice>` — feed the one-sided command to the live session, which
/// advances the battle O(1) to its next request boundary. An unknown side / choice token
/// is a hard error (a malformed driver). If no battle is live it is dropped silently
/// (mirrors the Node bridge's `if (streams && streams[side])` guard).
fn handle_choose(sess: &mut Session, rest: &str, dex: &Dex) -> Result<(), String> {
    if sess.bridge.is_none() || sess.ended {
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
    // `commands` is PROTOCOL-faithful: every CHOOSE this child processed, in order,
    // INCLUDING attempts the engine refused (a refused maybe-trapped probe never commits, so
    // it is absent from `input_log`, but its `|error|` + re-request round IS part of the
    // per-side stream the agent saw).
    sess.recon
        .cmds
        .push((side_tok.to_string(), choice_tok.to_string()));
    // --- resumeReseed (`gen3_bridge_resume_reseed_v1`) — mirrors the node bridge exactly:
    //     swap the PRNG at the START of the divergence turn (the battle's `turn` has already
    //     advanced to it once the prior turn resolved), BEFORE this turn's choices commit, and
    //     ONCE. That split is the whole point of the counterfactual: the recorded PREFIX keeps
    //     its original dice and only the post-divergence resolution draws from the fresh
    //     stream, so a re-rolled win-% estimates THAT board rather than a different game. ---
    if let Some((turn, seed)) = sess.resume_reseed.clone() {
        if !sess.reseeded {
            if let Some(b) = sess.bridge.as_mut() {
                if b.turn() == turn {
                    b.reseed(&seed);
                    sess.reseeded = true;
                }
            }
        }
    }
    sess.bridge
        .as_mut()
        .expect("bridge present")
        .feed_cmd(Cmd { side, choice }, dex);
    Ok(())
}

/// `FORCELOSE <side>` — the poke-env `/forfeit` path. The Node bridge writes `>forcelose
/// <side>` into the sim, which runs a real `win(otherSide)`: BOTH players receive the
/// deciding `|` + `|win|<name>` batch, and only then do the streams close. So this must
/// emit that win batch too, via [`BridgeSession::forfeit`], BEFORE `__END__`.
///
/// Emitting a bare `__END__` (what this used to do) is what made the Rust bridge HANG the
/// trainer: poke-env's `Battle` never sees `|win|`, so `battle.finished` stays False and the
/// env's next `reset()` waits forever on a result that can never arrive. The training seam
/// forfeits whenever `reset()` lands mid-battle, so every episode boundary could wedge.
fn handle_forcelose(sess: &mut Session, side: &str, out: &mut impl Write) -> Result<(), String> {
    if sess.bridge.is_none() || sess.ended {
        return Ok(());
    }
    let side_idx = match side {
        "p1" => 0,
        "p2" => 1,
        other => return Err(format!("FORCELOSE: bad side {other:?}")),
    };
    // Run the forfeit through the live session so the win lines land in its chunk stream,
    // then flush everything past the cursor (the pending batch AND the win batch). The
    // forfeit marks the session ended, so the flush emits `__END__` on its own — exactly like
    // a natural end reached through CHOOSE. Calling `end_battle` again here would emit a
    // SECOND `__END__`, because a persistent `end_battle` resets the `ended` guard for the
    // next battle.
    if let Some(bridge) = sess.bridge.as_mut() {
        bridge.forfeit(side_idx);
    }
    flush_new_chunks(sess, out)?;
    Ok(())
}

/// Write the NEW per-side chunk suffix the live session produced (past the emit cursor)
/// as `pN <base64>` lines. Emits `__END__` when the battle ends.
fn flush_new_chunks(sess: &mut Session, out: &mut impl Write) -> Result<(), String> {
    if sess.ended {
        return Ok(());
    }
    let (len, is_ended) = {
        let bridge = sess
            .bridge
            .as_ref()
            .ok_or("no battle in progress (missing START)")?;
        let all = &bridge.chunks().chunks;
        // Emit every chunk past the cursor, in flush order (both sides interleaved).
        for c in all.iter().skip(sess.emitted) {
            emit_chunk(out, c.side, &c.lines);
        }
        (all.len(), bridge.is_ended())
    };
    sess.emitted = len;
    if is_ended {
        end_battle(sess, out);
    }
    Ok(())
}

/// Emit `__END__` (+ persistent reset / non-persistent exit-arm). In persistent mode the
/// next `START` rebuilds a fresh battle; otherwise the process would exit — but this
/// binary keeps its loop alive and simply refuses further work until the next START/END
/// (the Python non-persistent path sends `END` right after `__END__`).

/// Minimal JSON string escaping for the record (the crate is std-only; the payloads here are
/// player names and packed teams, so only the mandatory escapes can occur).
fn json_quote(v: &str) -> String {
    let mut out = String::with_capacity(v.len() + 2);
    out.push('"');
    for c in v.chars() {
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

/// Emit `__RECON__ <base64(json)>` — the battle's reconstruction record
/// (`gen3_bridge_recon_record_v1`), once per battle, just before `__END__`.
///
/// `{v, format_id, prng_seed, input_log, commands}`, mirroring the node bridge:
///   * `input_log` — `>start` with the RESOLVED seed, both `>player` lines, then the
///     COMMITTED choices. STATE-faithful: replaying it rebuilds the battle.
///   * `commands`  — every CHOOSE processed, refusals included. PROTOCOL-faithful.
///
/// HONEST SCOPE. The `>start`/`>player` lines are exact, and they are the only part any
/// consumer currently reads (`ReconstructionRecord.start_options()` / `.players()`; the
/// replay path drives `commands`). The COMMITTED-CHOICE lines are rendered from the
/// engine's own committed `script`, so they are replay-EQUIVALENT rather than guaranteed
/// byte-identical to the sim's `inputLog` normalization for exotic wire spellings (e.g. a
/// forced `move struggle`). That distinction is deliberate: the module's rule is that we do
/// NOT fake `__RECON__` content parity, so this claims semantic faithfulness only.
///
/// The `seed.is_empty()` skip below is now UNREACHABLE by construction — `handle_start`
/// always resolves a seed (the caller's, or a minted one) — and is kept only as a
/// belt-and-braces guard against a record that could not be replayed. Before
/// `gen3_bridge_seedless_fixed_seed_v1` it fired on EVERY seedless battle, which is why a
/// rust eval wrote no `*_reconstruction.json` and the prober's forensic commands went dark.
fn emit_recon(sess: &mut Session, out: &mut impl Write) {
    if sess.recon.emitted || sess.recon.seed.is_empty() {
        return;
    }
    sess.recon.emitted = true;
    let r = &sess.recon;
    // The seed is a JSON STRING here, mirroring the sim's own `inputLog[0]`
    // (`JSON.stringify({formatid, seed: this.prngSeed})`, where `prngSeed` is
    // `PRNG.startingSeed` — a string, since the constructor `join(",")`s an array form).
    // It must NOT be rendered as a bare `[...]` array: a minted `sodium,<hex>` seed is not
    // a number list, so the array spelling produced INVALID JSON on the seedless path.
    let mut input_log: Vec<String> = vec![format!(
        ">start {{\"formatid\":\"{}\",\"seed\":{}}}",
        r.format_id,
        json_quote(&r.seed)
    )];
    input_log.push(format!(">player p1 {}", r.p1_json));
    input_log.push(format!(">player p2 {}", r.p2_json));
    // The COMMITTED choices, from the engine's own script.
    if let Some(b) = sess.bridge.as_ref() {
        for dec in b.script() {
            for (side, ch) in [(1usize, dec.p1), (2usize, dec.p2)] {
                if let Some(c) = ch {
                    let tok = match c {
                        pokesim::turn::Choice::Move(k) => format!("move {}", k + 1),
                        pokesim::turn::Choice::Switch(n) => format!("switch {}", n + 1),
                    };
                    input_log.push(format!(">p{side} {tok}"));
                }
            }
        }
    }
    let il = input_log.iter().map(|l| json_quote(l)).collect::<Vec<_>>().join(",");
    let cmds = r
        .cmds
        .iter()
        .map(|(s, c)| format!("[{},{}]", json_quote(s), json_quote(c)))
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(
        "{{\"v\":1,\"format_id\":{},\"prng_seed\":{},\"input_log\":[{}],\"commands\":[{}]}}",
        json_quote(&r.format_id),
        json_quote(&r.seed),
        il,
        cmds
    );
    writeln!(out, "__RECON__ {}", base64_encode(json.as_bytes())).ok();
}

fn end_battle(sess: &mut Session, out: &mut impl Write) {
    if sess.ended {
        return;
    }
    sess.ended = true;
    emit_recon(sess, out);
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
