//! `sim_bridge` — a drop-in Rust replacement for
//! `src/utils/bridge/local_sim_bridge.js`, speaking the EXACT stdin/stdout protocol so
//! it can run behind the Python bridge (`local_battle_runner.py`) with zero protocol
//! change: replace `node local_sim_bridge.js` with this binary.
//!
//! # Protocol (mirrors `local_sim_bridge.js` byte-for-byte)
//!
//! stdin (newline-delimited commands):
//! - `START <json>`  `{formatid, seed?, persistent?, resumeReseed?, core_obs?, p1:{name,team}, p2:{name,team}}`
//! - `CHOOSE <side> <choice>`  e.g. `CHOOSE p1 move 1` / `CHOOSE p2 switch 3`
//! - `FORCELOSE <side>`  e.g. `FORCELOSE p1`  (poke-env `/forfeit` path)
//! - `END`  tear down and exit
//!
//! stdout (newline-delimited frames):
//! - `p1 <base64(chunk)>` / `p2 <base64(chunk)>`  one protocol chunk that side saw
//! - `__OBS__ <p1|p2> <json>`  the side's observation row built by the CORE — ONLY when the
//!   battle's START carried `core_obs` (see "Core observation mode" below)
//! - `__END__`  battle over, both side streams closed (persistent → reset for the next START)
//! - `__ERR__ <base64(msg)>`  fatal error
//! - `__RECON__ <base64(json)>`  the reconstruction record, once per battle, just before
//!   `__END__` (see [`emit_recon`] for the honest scope of its `input_log`)
//!
//! # Core observation mode (`gen3_bridge_core_obs_v1`) — OPT-IN, default OFF
//!
//! `START`'s optional `core_obs` key,
//! `{"sides": ["p1"] | ["p2"] | ["p1","p2"], "decision_tense": bool, "switch_freeze": bool}`
//! (all three REQUIRED when the key is present; an unknown key, an empty / repeated / unknown
//! side or a non-boolean flag is a LOUD `__ERR__`), turns on the Rust core's observation for the
//! named sides. The two booleans are the progress clock's `ClockConfig` (training's
//! `--progress-decision-tense` / `--progress-switch-freeze`). ABSENT (or `null`) ⇒ this binary's
//! stdout is BYTE-IDENTICAL to the mode's absence (pinned by `tests/sim_bridge_core_obs_test.rs`).
//!
//! Per requested side the child keeps a PARSE-built version chain with trackers
//! (`BattleVersion::parse_root_with`) — the observation comes THROUGH THE PARSER, the program's
//! §6c one observation path — advanced by exactly the lines that side was newly shipped in each
//! write (incremental; the whole stream is never re-parsed). Every `CHOOSE` of a requested side
//! is noted on its chain (`note_choice`, the raw choice token) BEFORE it is fed, as
//! `core_events` does. For each write (START's first emission, every CHOOSE / FORCELOSE) each
//! requested side whose chain took a DECISION at the write's boundary gets ONE frame, written
//! BEFORE that write's chunk frames (p1's before p2's) — the parent fires each chunk as an
//! un-awaited task, so the row must be stashed before the request chunk is dispatched:
//!
//! ```text
//! __OBS__ p1 {"frame":{"dtype":"<f4","shape":[2501],"b64":…},"mask":[11 × 0|1],
//!             "tokens":{"<idx>":"<choice>",…},"turn":<int>,"line":<int>,"rqid":<int>|null,"n":<int>}
//! ```
//!
//! `frame` is `encoder::wire::frame` of `BattleVersion::encode` (NaN-prefilled in test /
//! self-check builds, zero-filled in release); `mask` is `present::mask`; `tokens` is
//! `present::choice_tokens` (the real mapper's choice string per legal action); `turn` is the
//! reading's turn; `line` is the side's stream index of the `|request|` it decided on; `rqid` is
//! the request JSON's `rqid` when it carries one (the engine never writes one: `null` today); `n`
//! is the frame's 0-based index among THIS side's frames in the current battle (0 at every START)
//! — the alignment key a consumer counts its own decisions against, since `rqid` cannot be.
//! A decision's request must be the LAST line the side was shipped in the write (the `core_events`
//! alignment), and a write opens at most one decision per side. A battle that ends emits no
//! frame for its terminal board (no decision). Any parse / fold / encode / alignment failure is a
//! FATAL `__ERR__`, written IN PLACE of that write's chunk frames — never a skipped frame, never a
//! fall-back — and the mode stays failed for the rest of the battle. The chains are dropped at every battle reset (persistent recycling keeps
//! nothing). This builds the core's own reading on the parse chain: the transport's reveal fold
//! and core source recording stay OFF (`tests/view_fold_opt_in_test.rs`).
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
use pokesim::core_events::jsonval::Val;
use pokesim::dex::Dex;
use pokesim::encoder::{self, OBS_DIM};
use pokesim::json::Json;
use pokesim::present;
use pokesim::prng::{normalize_seed, Prng};
use pokesim::trackers::clock::ClockConfig;
use pokesim::version::BattleVersion;

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
                // A self-check failure is never one bad line: exit (compiled out of `--release`).
                #[cfg(any(debug_assertions, feature = "emission-selfcheck"))]
                pokesim::emission_check::exit_if_failure(&msg);
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
    /// The OPT-IN core observation (`gen3_bridge_core_obs_v1`): `None` unless this battle's
    /// START carried `core_obs`. Dropped by `reset` (a new battle rebuilds it from its START).
    core_obs: Option<CoreObs>,
}

/// A parsed `core_obs` START key.
struct CoreObsSpec {
    sides: [bool; 2],
    cfg: ClockConfig,
}

/// Parse START's `core_obs` key. Absent or `null` → `None` (the mode is OFF and nothing about
/// this child's output changes). Present → every field is REQUIRED and checked: a mode flag is
/// read, never assumed, so a typo or a missing clock boolean is a loud error rather than a
/// silently-default observation.
fn parse_core_obs(v: &Json) -> Result<Option<CoreObsSpec>, String> {
    let Some(c) = v.get("core_obs").filter(|c| !c.is_null()) else {
        return Ok(None);
    };
    let obj = c.as_object().ok_or("START: `core_obs` must be an object")?;
    for k in obj.keys() {
        if !matches!(k.as_str(), "sides" | "decision_tense" | "switch_freeze") {
            return Err(format!("START: `core_obs` has an unknown key {k:?}"));
        }
    }
    let arr = c
        .get("sides")
        .and_then(Json::as_array)
        .ok_or("START: `core_obs.sides` must be an array of \"p1\" / \"p2\"")?;
    if arr.is_empty() {
        return Err("START: `core_obs.sides` is empty".to_string());
    }
    let mut sides = [false; 2];
    for s in arr {
        let i = match s.as_str() {
            Some("p1") => 0,
            Some("p2") => 1,
            _ => return Err(format!("START: `core_obs.sides` holds {s:?}, not \"p1\" / \"p2\"")),
        };
        if sides[i] {
            return Err(format!("START: `core_obs.sides` names p{} twice", i + 1));
        }
        sides[i] = true;
    }
    let flag = |k: &str| -> Result<bool, String> {
        c.get(k)
            .and_then(Json::as_bool)
            .ok_or_else(|| format!("START: `core_obs.{k}` must be present and a boolean"))
    };
    let cfg = ClockConfig { decision_tense: flag("decision_tense")?, switch_freeze: flag("switch_freeze")? };
    Ok(Some(CoreObsSpec { sides, cfg }))
}

/// The core observation state of ONE battle: per requested side, the PARSE-built version chain
/// (trackers on) and how far it has read.
struct CoreObs {
    /// The side's parse chain; `None` for a side that was not requested.
    chains: [Option<BattleVersion>; 2],
    /// Lines of the side's stream each chain has folded (the incremental cursor).
    folded: [usize; 2],
    /// Decisions each chain has taken (one `__OBS__` frame each).
    decided: [u32; 2],
    /// Set while a step is in flight and kept on failure: once a step failed (or panicked, which
    /// leaves the chain taken), every later write of this battle is refused too — the chain no
    /// longer holds the stream, and a skipped frame must never pass for a quiet one.
    failed: Option<String>,
}

impl CoreObs {
    fn new(spec: &CoreObsSpec, players: [&PlayerOptions; 2]) -> Result<CoreObs, String> {
        let mut chains = [None, None];
        for side in 0..2 {
            if spec.sides[side] {
                let p = players[side];
                chains[side] = Some(
                    BattleVersion::parse_root_with(side, &p.name, Some(&p.team.0), Some(spec.cfg))
                        .map_err(|e| format!("core_obs: p{} root: {}", side + 1, e.message()))?,
                );
            }
        }
        Ok(CoreObs { chains, folded: [0, 0], decided: [0, 0], failed: None })
    }

    /// Note a requested side's choice token on its chain, BEFORE the command is fed (a denied
    /// own action keeps it — `core_events` feeds its step chain the same way).
    fn note_choice(&mut self, side: usize, token: &str) {
        if let Some(c) = self.chains[side].as_mut() {
            c.note_choice(side, token);
        }
    }

    /// Advance every chain over the lines its side was shipped in `chunks[from..]` (this write)
    /// and emit one `__OBS__` frame per decision taken at the write's boundary.
    fn step(&mut self, bridge: &BridgeSession, from: usize, out: &mut impl Write) -> Result<(), String> {
        if let Some(m) = &self.failed {
            return Err(format!("core_obs: refused after an earlier failure in this battle: {m}"));
        }
        self.failed = Some("a core_obs step did not complete (panic)".to_string());
        match self.step_inner(bridge, from, out) {
            Ok(()) => {
                self.failed = None;
                Ok(())
            }
            Err(e) => {
                self.failed = Some(e.clone());
                Err(e)
            }
        }
    }

    fn step_inner(&mut self, bridge: &BridgeSession, from: usize, out: &mut impl Write) -> Result<(), String> {
        let chunks = &bridge.chunks().chunks[from..];
        for side in 0..2 {
            let Some(chain) = self.chains[side].take() else { continue };
            let tag = side + 1;
            let new: Vec<&str> = chunks
                .iter()
                .filter(|c| c.side == side)
                .flat_map(|c| c.lines.iter().map(String::as_str))
                .collect();
            let next = chain
                .parse_advance(&new)
                .map_err(|e| format!("core_obs: parse p{tag}: {}", e.message()))?;
            self.folded[side] += new.len();
            let s = next.stream(side).ok_or_else(|| format!("core_obs: p{tag}: the chain lost its stream"))?;
            if s.lines != self.folded[side] {
                return Err(format!("core_obs: p{tag}: the chain folded {} lines, the cursor says {}", s.lines, self.folded[side]));
            }
            // The incremental cursor == the whole shipped stream (O(chunks); test / self-check
            // builds only — the cursor is the same one that emits the chunks).
            #[cfg(any(debug_assertions, feature = "emission-selfcheck"))]
            if self.folded[side] != bridge.side_line_count(side) {
                return Err(format!(
                    "core_obs: p{tag}: folded {} lines but the side was shipped {}",
                    self.folded[side],
                    bridge.side_line_count(side)
                ));
            }
            let decisions = next.trackers(side).map_or(0, |t| t.decisions);
            let opened = decisions.checked_sub(self.decided[side]).ok_or_else(|| format!("core_obs: p{tag}: the decision count went backwards"))?;
            match (next.decision(side), opened) {
                (None, 0) => {}
                (Some(d), 1) => {
                    // The core_events alignment: the decision's request is the LAST line the side
                    // was shipped in this write (the live player decides after the request chunk).
                    if d.line + 1 != s.lines {
                        return Err(format!(
                            "core_obs: [ALIGN] p{tag} decided at stream line {} but the write shipped {} lines",
                            d.line, s.lines
                        ));
                    }
                    // `n` = this frame's index among this side's frames in this battle (the chain
                    // is rebuilt at every START, so it restarts at 0).
                    let json = obs_json(&next, side, d.line, self.decided[side])?;
                    writeln!(out, "__OBS__ p{tag} {json}").ok();
                    out.flush().ok();
                }
                (d, n) => {
                    return Err(format!(
                        "core_obs: p{tag}: one write opened {n} decisions (a decision at the boundary: {}) — \
                         exactly one frame per decision cannot be kept",
                        d.is_some()
                    ));
                }
            }
            self.decided[side] = decisions;
            self.chains[side] = Some(next);
        }
        Ok(())
    }
}

/// The `__OBS__` JSON of `side`'s decision on `v` (the request at stream line `line`), the side's
/// `n`-th frame (0-based) of the battle.
fn obs_json(v: &BattleVersion, side: usize, line: usize, n: u32) -> Result<String, String> {
    let tag = side + 1;
    let mut row = [0.0f32; OBS_DIM];
    v.encode(side, &mut row).map_err(|e| format!("core_obs: encode p{tag}: {}", e.message()))?;
    let legal = v.legal(side).ok_or_else(|| format!("core_obs: p{tag}: a decision with no legality"))?;
    let reading = &v.stream(side).ok_or_else(|| format!("core_obs: p{tag}: no stream"))?.board_reading;
    let tokens = present::choice_tokens(reading, &legal).map_err(|e| format!("core_obs: tokens p{tag}: {}", e.message()))?;
    let mask = present::mask(&legal);
    let rqid = match reading.last_request.as_ref().and_then(|r| r.get("rqid")) {
        None | Some(Val::Null) => "null".to_string(),
        Some(Val::Int(i)) => i.to_string(),
        Some(other) => return Err(format!("core_obs: p{tag}: the request's rqid is not an integer: {other:?}")),
    };
    let mask_json = mask.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(",");
    Ok(format!(
        "{{\"frame\":{},\"mask\":[{mask_json}],\"tokens\":{},\"turn\":{},\"line\":{line},\"rqid\":{rqid},\"n\":{n}}}",
        encoder::wire::frame(&row),
        present::tokens_json(&tokens),
        reading.turn
    ))
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
            core_obs: None,
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
        // …nor its core observation chains (dropped here; the next START builds its own).
        self.core_obs = None;
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
            // A CHOOSE with NO LIVE BATTLE is a no-op, exactly as in the Node bridge
            // (`local_sim_bridge.js`: `if (streams && streams[side]) { … }`). In PERSISTENT
            // mode the child resets itself at `__END__` (`end_battle` → `Session::reset` →
            // `bridge = None`), and a late answer to the ending battle's last `|request|`
            // lands here routinely: `BridgeSession._dispatch` fires poke-env's feeds as
            // UN-AWAITED tasks, so one can resolve after `__END__` but before the parent has
            // dropped the old battle tag. `handle_choose` already drops such a CHOOSE, but the
            // arm then fell through to `flush_new_chunks`, which has no bridge and returned
            // `Err("no battle in progress (missing START)")` → `__ERR__`. That is FATAL to the
            // parent: an `__ERR__` retires `_persistent_read_loop` and trips
            // `_signal_transport_dead()`, so every in-flight `step()` raises
            // `ShowdownException` and the whole run dies. Gate the flush too.
            if sess.bridge.is_none() || sess.ended {
                return Ok(LineResult::Continue);
            }
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
    // The OPT-IN core observation (`gen3_bridge_core_obs_v1`): parsed and its chains built
    // BEFORE anything is reset, so a malformed key refuses the START without touching state.
    let core_obs = match parse_core_obs(&v)? {
        Some(spec) => Some(CoreObs::new(&spec, [&p1, &p2])?),
        None => None,
    };

    // A new battle: reset the per-battle state (persistent keeps `persistent`), then build
    // the live incremental engine (advances to + emits the first request boundary).
    sess.reset();
    sess.core_obs = core_obs;
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
    // The core observation's chain takes the RAW choice token BEFORE the command is fed —
    // including a choice the engine will refuse (a denied own action keeps it), exactly the
    // order `core_events` uses (`note_choice`, then `feed_cmd`).
    if let Some(o) = sess.core_obs.as_mut() {
        o.note_choice(side, choice_tok);
    }
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
    // Record the forfeit in `commands` BEFORE running it, exactly where the Node bridge does
    // (`cmdLog.push(['forcelose', rest])`). `commands` is the ordered log the offline replay
    // path drives (`search::feed_recorded_cmd` has a `"forcelose"` arm, and
    // `recorded_turn_choices` stops at one), so a rust record that omitted it replayed a
    // FORFEITED battle as if it had played on — and, downstream,
    // `cf_producer.record_is_full_replay_anchorable`'s forfeit exclusion scans this very field,
    // so under `--impl rust` it was INERT and a census of forfeits read a false 0. Recording it
    // here is the smaller of the two honest fixes (one push, at the site that already knows the
    // side) and it fixes the record itself rather than one of its readers.
    sess.recon
        .cmds
        .push(("forcelose".to_string(), side.to_string()));
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
/// as `pN <base64>` lines — preceded, in core observation mode, by each requested side's
/// `__OBS__` frame for a decision at this boundary. Emits `__END__` when the battle ends.
fn flush_new_chunks(sess: &mut Session, out: &mut impl Write) -> Result<(), String> {
    if sess.ended {
        return Ok(());
    }
    let from = sess.emitted;
    // The core observation frames go FIRST, BEFORE this write's chunk frames: the parent's reader
    // fires each chunk as an un-awaited task, and the trainee's embed can run on the request chunk
    // before the reader has read any later line — so the row must already be stashed when the
    // request chunk is dispatched. A failure here is an `__ERR__` in place of the write's chunks
    // (the battle is dead; nothing of this write is shipped).
    if let (Some(obs), Some(bridge)) = (sess.core_obs.as_mut(), sess.bridge.as_ref()) {
        obs.step(bridge, from, out)?;
    }
    let (len, is_ended) = {
        let bridge = sess
            .bridge
            .as_ref()
            .ok_or("no battle in progress (missing START)")?;
        let all = &bridge.chunks().chunks;
        // Emit every chunk past the cursor, in flush order (both sides interleaved).
        for c in all.iter().skip(from) {
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
///   * `commands`  — every CHOOSE processed, refusals included, PLUS a
///     `["forcelose", <side>]` entry for a forfeit (mirroring the node bridge's `cmdLog`).
///     PROTOCOL-faithful. The offline replay path drives this field, and a forfeit that is
///     absent from it replays as a battle that played on.
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
