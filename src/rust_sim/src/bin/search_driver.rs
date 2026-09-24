//! `search_driver` — the Rust drop-in for **both** offline driver families
//! (`gen3_rust_search_driver_v1` + `gen3_rust_replay_driver_v1`).
//!
//! Node splits these across TWO scripts; the port serves both from this ONE binary, which
//! is exactly what `utils/bridge/sim_bridge_bin.py::search_driver_spawn_argv` assumes:
//!
//! | family | node script | verbs | shape |
//! |---|---|---|---|
//! | SEARCH | `search_driver.js` | `open_root` / `expand_many` / `close` | PERSISTENT, `{id, cmd}` |
//! | REPLAY | `replay_driver.js` | `replay` / `reroll` / `reroll_many` | ONE-SHOT, `{mode}` |
//!
//! **Dispatch is on the KEY, not on a flag:** a request carrying `mode` is a one-shot
//! replay-family request (answer, then exit like the Node driver does — one JSON object on
//! stdout with NO trailing newline, exit 0, or `{"error": …}` + exit 1); anything else runs
//! the persistent `{id, cmd}` loop below. The two families cannot collide because no caller
//! sends both keys, and `utils/bridge/reconstruction.py` / `search_session.py` each speak
//! exactly one of them.
//!
//! A WARM, PERSISTENT clone-and-branch SEARCH server. Unlike the one-shot
//! `sim_bridge`, this stays alive and holds a node-snapshot cache, so a multi-ply BEAM
//! over candidate lines (the prober's `better_line` probe) can branch a search TREE
//! from any explored node WITHOUT re-replaying the battle from turn 1 per node.
//!
//! # THE LEVER — why the port is structurally simpler than the Node original
//!
//! Node clones a mid-battle state with `State.serializeBattle`/`deserializeBattle`,
//! which forces a whole dance per arm: deserialize, `restart()` a fresh `BattleStream`
//! with the same `send` wiring, `sendUpdates()` to flush the WHOLE re-emitted historical
//! log, and mark a BASELINE index so only this ply's suffix is returned (the re-emitted
//! prefix is useless — it lacks the `|request|` lines, which are sent out of band and
//! never stored in `battle.log`, so a materializer parsing it would never see the team).
//!
//! The port needs none of that. [`BridgeSession`] is plain owned data all the way down,
//! so [`BridgeSession::snapshot`] is a derived `Clone` — a deep, independent paused
//! battle — and [`BridgeSession::clear_chunks`] makes the branch's chunk stream contain
//! exactly its own suffix BY CONSTRUCTION rather than by index arithmetic. Same wire
//! contract, no byte format, no baseline.
//!
//! # PROTOCOL (newline-delimited JSON, one request → exactly one response line)
//!
//! ```text
//! {id, cmd:"open_root", record, turn}
//!   → {id, ok, node_id, requests, recorded_choices, pre_state,
//!      prefix_p1_chunks, prefix_p2_chunks}
//! {id, cmd:"expand_many", side?:"p1"|"p2", arms:[{node_id, p1_action, p2_action, seed,
//!                                                 label, recorded_exact?, followup?}]}
//!   → {id, ok, arms:[{label, node_id, ended, stuck, outcome, requests,
//!                     choices_used, p1_chunks, p2_chunks, view_p1, view_p2}]}
//! {id, cmd:"close"} → {id, ok, bye:true}, then exit 0
//! ```
//!
//! **`side` ELIDES the one-sided payload the caller will not read** (`SideWant`,
//! `gen3_expand_many_side_elision_v1`): with `side:"p1"` the `p2_chunks` / `view_p2` fields are
//! OMITTED, which is 43.0% of the reply bytes on the banked search decisions. Omitting `side`
//! is the default and renders the historical body byte-for-byte.
//!
//! # THE ONE-SHOT REPLAY PROTOCOL (`replay_driver.js`)
//!
//! ```text
//! {mode:"replay", record}
//!   → {p1_chunks, p2_chunks, outcome}
//! {mode:"reroll", record, turn, seeds:[…], p1_action, p2_action, followup}
//!   → {turn, pre_state, requests, recorded_choices, prefix_p1_chunks, prefix_p2_chunks,
//!      rerolls:[{seed, choices_used, outcome, turn_log, p1_chunks, p2_chunks}]}
//! {mode:"reroll_many", record, turn, followup,
//!  arms:[{p1_action, p2_action, seed, label}]}
//!   → the same head, with `arms:[{label, seed, …same per-arm fields}]`
//! ```
//!
//! Every arm runs in its OWN FRESH session rebuilt from turn 1 — NOT a clone of the root.
//! That is deliberate and load-bearing: `reroll_many` is the INDEPENDENT ORACLE the
//! clone-and-branch path above is checked against (`search_clone_parity_fuzz_test`), and a
//! shared prefix would destroy the independence that makes the check mean anything.
//!
//! Unparseable JSON answers `{"error": "..."}` with NO id (there is none to echo);
//! an unknown cmd or any failure answers `{id, ok:false, error:"..."}`. A panic is
//! caught and reported the same way — a search server that dies mid-beam would strand
//! the caller, so the loop always survives (the `sim_bridge` `catch_unwind` pattern).
//!
//! `p*_chunks` are this ply's one-sided SUFFIX; the Python caller composes the
//! (request/team-complete) ROOT prefix + each ply's suffix, the same shape
//! `reroll_many` produces. `pre_state` / `outcome` are OMNISCIENT (referee view) —
//! never fed to the obs encoder.

use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::Arc;

use pokesim::bridge::BridgeSession;
use pokesim::dex::Dex;
use pokesim::driver_timing::{ArmClock, ArmTimings};
use pokesim::json::Json;
use pokesim::search::{
    aux_rng_from_seed, build_to_turn, json_quote, log_len, outcome_of, pre_state, recorded_queues,
    recorded_turn_choices, resolve_turn, resolve_turn_capturing, resolve_turn_exact, resolve_turn_sourced,
    session_from_record, session_from_record_core, side_chunk_strings, turn_log, write_cmd, ActionSpec,
    Capture, Record, Resolved, TurnSource, RECORDED_QUEUE_CAP,
};
use pokesim::version::{self, BattleVersion};
use pokesim::view::one_sided_view;

/// One explored node: a paused session, plus (root only) the record + the index of the
/// first command turn T did not consume. Only the root carries those, so a depth-1
/// expand can reproduce the realized turn EXACTLY (`recorded_exact`) for the `value_crn`
/// faithfulness anchor — off-root there is no alignment to a recorded command stream.
struct Node {
    state: NodeState,
    record: Option<Record>,
    rest_idx: usize,
}

/// A node's paused battle: a bare engine (the `protocol` / `view` roads), or a
/// [`BattleVersion`] (`materializer=core`, `gen3_core_search_v1`) — the engine plus each side's
/// folded stream, so a successor's view is the version's, not a projection beside it.
enum NodeState {
    Plain(BridgeSession),
    Core(Arc<BattleVersion>, CorePath),
}

/// How a core node's successors fold their lines: TYPED at the source (the shortcut the Rust
/// Core Program's §6c licenses by `parse(emit(step)) == step`) or from the side's TEXT (the
/// stream-only path every other observation takes). Chosen at `open_root`; the whole tree runs on
/// one path so a search number is attributable to one.
#[derive(Clone, Copy, PartialEq)]
enum CorePath {
    Typed,
    Text,
}

impl Node {
    fn engine(&self) -> &BridgeSession {
        match &self.state {
            NodeState::Plain(s) => s,
            NodeState::Core(v, _) => v.engine().expect("a step-built version always has an engine"),
        }
    }
}

struct Server {
    nodes: HashMap<String, Node>,
    counter: u64,
    /// Core arms seen by the INTEGRITY sampler (1-in-N is `count % N == 0`), process lifetime.
    integrity_seen: u64,
}

impl Server {
    fn new() -> Server {
        Server { nodes: HashMap::new(), counter: 0, integrity_seen: 0 }
    }

    /// `n0`, `n1`, … — MONOTONIC for the process lifetime. A fresh `open_root` drops the
    /// previous tree (so a warm process reused across many battles stays memory-bounded)
    /// but deliberately does NOT reset the counter, so a stale node id from an earlier
    /// tree can never silently resolve to a new node.
    fn fresh_id(&mut self) -> String {
        let id = format!("n{}", self.counter);
        self.counter += 1;
        id
    }
}

fn main() {
    let dex = Dex::for_gen(3);
    let mut srv = Server::new();
    let stdin = io::stdin();
    // A record is ~100 KB, so one request line is large; size the reader for it.
    let mut reader = io::BufReader::with_capacity(1 << 20, stdin.lock());
    let mut out = io::stdout();
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return, // stdin closed — mirrors the Node `rl.on('close')` exit
            Ok(_) => {}
            Err(_) => return,
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        let req = match Json::parse(text) {
            Ok(v) => v,
            Err(e) => {
                respond(&mut out, &format!("{{\"error\":{}}}", json_quote(&format!("bad request JSON: {e}"))));
                continue;
            }
        };
        // THE FAMILY SWITCH. A `mode` key means this is a one-shot replay-family request:
        // answer it and EXIT, exactly as `replay_driver.js` does (its whole lifecycle is one
        // request). Everything else falls through to the persistent search loop.
        if req.get("mode").is_some() {
            one_shot(&mut out, &req, &dex);
        }
        let id = render_id(req.get("id"));
        let cmd = req.str_at("cmd").unwrap_or("").to_string();
        if cmd == "close" {
            respond(&mut out, &format!("{{\"id\":{id},\"ok\":true,\"bye\":true}}"));
            return;
        }
        // Crash-don't-drop: a handler panic becomes an `ok:false` error rather than a
        // dead process, so one bad arm cannot strand the whole beam.
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            handle(&mut srv, &cmd, &req, &dex)
        }));
        let body = match res {
            Ok(Ok(out_body)) => format!("{{\"id\":{id},\"ok\":true,{out_body}}}"),
            Ok(Err(msg)) => format!("{{\"id\":{id},\"ok\":false,\"error\":{}}}", json_quote(&msg)),
            Err(panic) => format!(
                "{{\"id\":{id},\"ok\":false,\"error\":{}}}",
                json_quote(&format!("panic: {}", panic_message(&panic)))
            ),
        };
        respond(&mut out, &body);
    }
}

fn respond(out: &mut impl Write, body: &str) {
    writeln!(out, "{body}").ok();
    out.flush().ok();
}

/// The message a caught panic carried (`&str` and `String` payloads cover every `panic!`
/// / `assert!` / `unwrap` in this crate; anything else reports the bare word).
fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic".to_string())
}

// ===========================================================================
// The ONE-SHOT replay family (`replay_driver.js`).
// ===========================================================================

/// Answer ONE replay-family request and EXIT — the whole lifecycle of `replay_driver.js`.
///
/// Byte-shape parity with the Node driver: the body is a single JSON object with **no
/// trailing newline** (Node's `process.stdout.write(JSON.stringify(obj))`), exit 0 on
/// success and `{"error": …}` + exit 1 on failure. `reconstruction.py::_run_driver` checks
/// the EXIT CODE before it parses stdout, so the code is the load-bearing half of the
/// contract and the message is informational.
///
/// The `catch_unwind` guard is kept for the same reason the persistent loop has one: a panic
/// must become a reportable error, never a child that dies with an empty stdout (which the
/// caller could only report as "failed (rc=101)").
fn one_shot(out: &mut impl Write, req: &Json, dex: &Dex) -> ! {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match req.str_at("mode").unwrap_or("") {
            "replay" => run_replay(req, dex),
            "reroll" => run_reroll(req, dex),
            "reroll_many" => run_reroll_many(req, dex),
            other => Err(format!("unknown mode {other}")),
        }
    }));
    let (body, code) = match res {
        Ok(Ok(body)) => (body, 0),
        Ok(Err(msg)) => (format!("{{\"error\":{}}}", json_quote(&msg)), 1),
        Err(panic) => (
            format!("{{\"error\":{}}}", json_quote(&format!("panic: {}", panic_message(&panic)))),
            1,
        ),
    };
    write!(out, "{body}").ok();
    out.flush().ok();
    std::process::exit(code);
}

/// The record every replay-family verb starts from.
fn record_of(req: &Json) -> Result<Record, String> {
    Record::parse(req.get("record").ok_or("request has no record")?)
}

/// `turn`, validated exactly as Node's `if (!Number.isInteger(T) || T < 1)`.
fn turn_of(req: &Json) -> Result<u32, String> {
    let raw = req.get("turn").and_then(Json::as_f64);
    match raw {
        Some(t) if t.fract() == 0.0 && t >= 1.0 => Ok(t as u32),
        _ => Err(format!("invalid turn {}", raw.map_or("undefined".to_string(), render_num))),
    }
}

/// `mode:"replay"` — re-run the whole recorded command stream and return the regenerated
/// per-side chunks + the final omniscient outcome (Node's `runReplay`).
fn run_replay(req: &Json, dex: &Dex) -> Result<String, String> {
    let rec = record_of(req)?;
    let mut sess = session_from_record(&rec, dex)?;
    for cmd in &rec.commands {
        write_cmd(&mut sess, cmd, dex)?;
    }
    if !sess.is_ended() {
        // Node's exact wording — a truncated / corrupt record is the one thing this verb can
        // detect, and reporting the turn it stalled at is what makes it diagnosable.
        return Err(format!(
            "replayed all {} commands but battle has not ended (turn {}) — corrupt or truncated record?",
            rec.commands.len(),
            sess.turn()
        ));
    }
    Ok(format!(
        "{{\"p1_chunks\":{},\"p2_chunks\":{},\"outcome\":{}}}",
        chunk_array(&sess, 0),
        chunk_array(&sess, 1),
        outcome_of(&sess, false)
    ))
}

/// The decision-point half both re-roll verbs share: ONE inspection pass that reports the
/// board, the choice surfaces, the original picks, and the prefix protocol.
///
/// Returns the rendered head WITHOUT its closing brace, so the caller appends its own
/// `rerolls` / `arms` array. Node builds the same head twice (`runReroll` / `runRerollMany`).
fn reroll_head(rec: &Record, turn: u32, dex: &Dex) -> Result<String, String> {
    let mut inspect = session_from_record(rec, dex)?;
    let rest_idx = build_to_turn(&mut inspect, rec, turn, dex)?;
    Ok(format!(
        "{{\"turn\":{},\"pre_state\":{},\"requests\":{},\"recorded_choices\":{},\
         \"prefix_p1_chunks\":{},\"prefix_p2_chunks\":{}",
        turn,
        pre_state(&inspect),
        requests_json(&inspect),
        recorded_choices_json(rec, rest_idx),
        chunk_array(&inspect, 0),
        chunk_array(&inspect, 1)
    ))
}

/// Resolve ONE arm of turn `turn` in a FRESH session — Node's `resolveArm`, and the inner
/// loop of `runReroll` too (they are the same code there; keeping ONE copy here is why the
/// batched and per-seed paths cannot drift).
///
/// Returns the rendered per-arm TAIL (`choices_used` … `p2_chunks`); the caller prefixes the
/// arm's own identity fields (`seed`, and for `reroll_many` a `label`).
fn resolve_arm(
    rec: &Record,
    turn: u32,
    p1_action: &str,
    p2_action: &str,
    seed: &str,
    followup: &str,
    dex: &Dex,
) -> Result<String, String> {
    let mut sess = session_from_record(rec, dex)?;
    let from_idx = build_to_turn(&mut sess, rec, turn, dex)?;
    // The port's answer to Node's `sess.chunks.p1.slice(prefixCounts.p1)`: drop the prefix
    // so what the arm emits from here IS its suffix, by construction rather than by index.
    sess.clear_chunks();
    let log_start = log_len(&sess);

    let is_original = seed == "original";
    if !is_original {
        sess.reseed(seed); // THE SWAP: every die from here routes through the new PRNG.
    }
    let mut rng = aux_rng_from_seed(seed);
    let resolved: Resolved = if is_original && p1_action == "recorded" && p2_action == "recorded" {
        // The realized line: original dice, original interleaving, original follow-ups.
        resolve_turn_exact(&mut sess, rec, from_idx, dex)?
    } else {
        let q = recorded_queues(rec, from_idx, RECORDED_QUEUE_CAP);
        let mut sources = [
            TurnSource::from_replay_spec(p1_action, &q[0]),
            TurnSource::from_replay_spec(p2_action, &q[1]),
        ];
        resolve_turn_sourced(&mut sess, &mut sources, followup, &mut rng, dex)
    };

    Ok(format!(
        "\"choices_used\":{},\"outcome\":{},\"turn_log\":{},\"p1_chunks\":{},\"p2_chunks\":{}",
        format_args!(
            "{{\"p1\":{},\"p2\":{}}}",
            string_array(&resolved.used[0]),
            string_array(&resolved.used[1])
        ),
        outcome_of(&sess, resolved.stuck),
        string_array(&turn_log(&sess, log_start)),
        chunk_array(&sess, 0),
        chunk_array(&sess, 1)
    ))
}

/// `mode:"reroll"` — one action spec, N seeds (Node's `runReroll`).
fn run_reroll(req: &Json, dex: &Dex) -> Result<String, String> {
    let rec = record_of(req)?;
    let turn = turn_of(req)?;
    let followup = req.str_at("followup").unwrap_or("random").to_string();
    let p1 = req.str_at("p1_action").unwrap_or("recorded").to_string();
    let p2 = req.str_at("p2_action").unwrap_or("recorded").to_string();
    let empty: Vec<Json> = Vec::new();
    let seeds: Vec<String> = req
        .get("seeds")
        .and_then(Json::as_array)
        .unwrap_or(&empty)
        .iter()
        .filter_map(Json::as_str)
        .map(str::to_string)
        .collect();

    let head = reroll_head(&rec, turn, dex)?;
    let mut rerolls = Vec::with_capacity(seeds.len());
    for seed in &seeds {
        let tail = resolve_arm(&rec, turn, &p1, &p2, seed, &followup, dex)?;
        rerolls.push(format!("{{\"seed\":{},{}}}", json_quote(seed), tail));
    }
    Ok(format!("{head},\"rerolls\":[{}]}}", rerolls.join(",")))
}

/// `mode:"reroll_many"` — N independent ARMS (each its own action pair + seed) resolved in
/// ONE process (Node's `runRerollMany`). Purely a batching of `reroll`: each arm still gets
/// a fresh session, so an arm's suffix is byte-identical to the same single `reroll`.
fn run_reroll_many(req: &Json, dex: &Dex) -> Result<String, String> {
    let rec = record_of(req)?;
    let turn = turn_of(req)?;
    let followup = req.str_at("followup").unwrap_or("random").to_string();
    let empty: Vec<Json> = Vec::new();
    let arms = req.get("arms").and_then(Json::as_array).unwrap_or(&empty);

    let head = reroll_head(&rec, turn, dex)?;
    let mut out = Vec::with_capacity(arms.len());
    for arm in arms {
        let p1 = arm.str_at("p1_action").unwrap_or("recorded").to_string();
        let p2 = arm.str_at("p2_action").unwrap_or("recorded").to_string();
        // Node has no default here: a seedless arm reaches `auxRngFromSeed(undefined)` and
        // throws a TypeError, so the request FAILS. Fail with a message instead of inventing
        // a default — inventing one would silently answer a different question.
        let seed = arm.str_at("seed").ok_or("arm: missing seed")?.to_string();
        let tail = resolve_arm(&rec, turn, &p1, &p2, &seed, &followup, dex)?;
        out.push(format!(
            "{{\"label\":{},\"seed\":{},{}}}",
            render_id(arm.get("label")),
            json_quote(&seed),
            tail
        ));
    }
    Ok(format!("{head},\"arms\":[{}]}}", out.join(",")))
}

/// Echo the request's `id` verbatim. Node writes `req.id != null ? req.id : null` and
/// lets `JSON.stringify` render it, so a number stays a number and a string stays
/// quoted. A container id has no producer; it renders as `null` rather than being
/// half-serialized.
fn render_id(v: Option<&Json>) -> String {
    match v {
        None | Some(Json::Null) => "null".to_string(),
        Some(Json::Num(n)) => render_num(*n),
        Some(Json::Str(s)) => json_quote(s),
        Some(Json::Bool(b)) => b.to_string(),
        Some(_) => "null".to_string(),
    }
}

/// `JSON.stringify` number rendering for the values that reach here (integers, and
/// the occasional float label): an integral value prints without a decimal point.
fn render_num(n: f64) -> String {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 9e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

fn handle(srv: &mut Server, cmd: &str, req: &Json, dex: &Dex) -> Result<String, String> {
    match cmd {
        "open_root" => open_root(srv, req, dex),
        "expand_many" => expand_many(srv, req, dex),
        other => Err(format!("unknown cmd {other}")),
    }
}

// ===========================================================================
// open_root
// ===========================================================================

fn open_root(srv: &mut Server, req: &Json, dex: &Dex) -> Result<String, String> {
    let t_raw = req.get("turn").and_then(Json::as_f64);
    let turn = match t_raw {
        Some(t) if t.fract() == 0.0 && t >= 1.0 => t as u32,
        _ => {
            let shown = t_raw.map_or("undefined".to_string(), render_num);
            return Err(format!("invalid turn {shown}"));
        }
    };
    let rec = Record::parse(req.get("record").ok_or("open_root: missing record")?)?;
    // `core` (`gen3_core_search_v1`): build the tree of BattleVersions instead of bare sessions,
    // folding each successor's lines TYPED at the source ("typed") or from the side's TEXT
    // ("text"). Absent = the historical body, byte for byte.
    let core = match req.str_at("core") {
        None => None,
        Some("typed") => Some(CorePath::Typed),
        Some("text") => Some(CorePath::Text),
        Some(other) => return Err(format!("open_root: core must be \"typed\" or \"text\", got \"{other}\"")),
    };
    // `side` (core only): the one side whose stream the tree folds — a search reads one side, so
    // the other's fold would be pure cost at every version. Absent = both.
    let want = match req.str_at("side") {
        None => [true, true],
        Some("p1") => [true, false],
        Some("p2") => [false, true],
        Some(other) => return Err(format!("open_root: side must be \"p1\" or \"p2\", got \"{other}\"")),
    };
    // A fresh root starts a fresh tree; drop the previous search's nodes. ids stay
    // monotonic (see `Server::fresh_id`).
    srv.nodes.clear();
    let mut sess = if core == Some(CorePath::Typed) {
        session_from_record_core(&rec, dex)?
    } else {
        session_from_record(&rec, dex)?
    };
    let rest_idx = build_to_turn(&mut sess, &rec, turn, dex)?;

    let requests = requests_json(&sess);
    let recorded = recorded_choices_json(&rec, rest_idx);
    let ps = pre_state(&sess);
    let p1 = chunk_array(&sess, 0);
    let p2 = chunk_array(&sess, 1);

    let node_id = srv.fresh_id();
    // The core road has no use for the port's projection (its view is the version's own), so a
    // core root renders none.
    let views = if core.is_some() {
        String::new()
    } else {
        format!(",\"view_p1\":{},\"view_p2\":{}", one_sided_view(&sess, 0, dex), one_sided_view(&sess, 1, dex))
    };
    let body = format!(
        "\"node_id\":{},\"requests\":{},\"recorded_choices\":{},\"pre_state\":{},\
         \"prefix_p1_chunks\":{},\"prefix_p2_chunks\":{}{}",
        json_quote(&node_id),
        requests,
        recorded,
        ps,
        p1,
        p2,
        views
    );
    let state = match core {
        None => NodeState::Plain(sess),
        Some(path) => {
            let names = [rec.p1.name.clone(), rec.p2.name.clone()];
            let teams = [rec.p1.team.0.clone(), rec.p2.team.0.clone()];
            let n = [names[0].as_str(), names[1].as_str()];
            let t = [Some(teams[0].as_str()), Some(teams[1].as_str())];
            let v = match path {
                CorePath::Typed => BattleVersion::root(sess, n, t, true, want)?,
                CorePath::Text => BattleVersion::root_text(sess, n, t, want)?,
            };
            NodeState::Core(Arc::new(v), path)
        }
    };
    srv.nodes.insert(node_id, Node { state, record: Some(rec), rest_idx });
    Ok(body)
}

// ===========================================================================
// expand_many
// ===========================================================================

/// Which side's ONE-SIDED payload (`view_pN` + `pN_chunks`) the caller wants back —
/// `gen3_expand_many_side_elision_v1`.
///
/// # Why a request can ask for less than the driver knows
///
/// Both are emitted because both are cheap to produce; neither is cheap to SHIP. A search runs
/// for exactly one side, and `search::_expand_ply` reads `view_p1 if side == "p1" else view_p2`
/// and the matching chunk array and nothing else — so the other side's ~13 KB per arm was
/// rendered, quoted, piped, `json.loads`-ed and dropped. Measured on 864 banked arms it was
/// **43.0% of the reply bytes**
/// (`designs/research_state/measurements/expand_many_2026-09-22/README.md`).
///
/// 🚨 **The default is [`SideWant::Both`], and Both is BYTE-IDENTICAL to the historical body** —
/// same fields, same order, same bytes. A request that does not carry `side` cannot tell this
/// code exists, which is what keeps `harness/search_impl_parity.py` (whose golden requests carry
/// no such key) comparing the node driver against this one field-for-field.
///
/// An UNRECOGNISED `side` is an ERROR, not a fallback to Both: a typo that silently returned
/// everything would read as a working elision that saved nothing, and the only symptom would be
/// a benchmark that failed to move.
#[derive(Clone, Copy, PartialEq)]
enum SideWant {
    Both,
    Only(usize),
}

impl SideWant {
    fn parse(req: &Json) -> Result<SideWant, String> {
        match req.str_at("side") {
            None => Ok(SideWant::Both),
            Some("p1") => Ok(SideWant::Only(0)),
            Some("p2") => Ok(SideWant::Only(1)),
            Some(other) => Err(format!(
                "expand_many: side must be \"p1\" or \"p2\", got \"{other}\""
            )),
        }
    }

    fn wants(self, side: usize) -> bool {
        match self {
            SideWant::Both => true,
            SideWant::Only(s) => s == side,
        }
    }
}

/// `,"<name>":<value>` when the side was asked for, and **nothing at all** when it was not.
///
/// Omission rather than `null` on purpose: the point of the field is its BYTES, and the Python
/// half (`utils/bridge/search_session.py`) distinguishes an absent key from an empty one — an
/// elided side comes back as a sentinel that RAISES on read, so a consumer that wanted the side
/// it declined to ask for fails loudly instead of encoding an empty board.
fn opt_field(name: &str, value: Option<&str>) -> String {
    match value {
        Some(v) => format!(",\"{name}\":{v}"),
        None => String::new(),
    }
}

fn expand_many(srv: &mut Server, req: &Json, dex: &Dex) -> Result<String, String> {
    let empty: Vec<Json> = Vec::new();
    let arms = req.get("arms").and_then(Json::as_array).unwrap_or(&empty).to_vec();
    let want = SideWant::parse(req)?;
    // `integrity` (`gen3_core_search_v1`, core nodes only): every Nth core arm is folded BOTH ways
    // — typed at the source and from the side's text — and the two versions must agree on the
    // whole stream state and the view, or the request FAILS naming the arm and the field. 0 = off.
    let integrity = match req.get("integrity") {
        None | Some(Json::Null) => 0u64,
        Some(v) => v.as_f64().filter(|n| *n >= 0.0 && n.fract() == 0.0).ok_or("expand_many: integrity must be a non-negative integer")? as u64,
    };
    let mut out = Vec::with_capacity(arms.len());
    // `gen3_expand_many_timing_v1` — zero-cost and ZERO-BYTE unless `POKESIM_SEARCH_TIMING=1`.
    let mut timings = ArmTimings::default();
    let whole = ArmClock::start();
    for arm in &arms {
        let node_id = arm.str_at("node_id").ok_or("arm: missing node_id")?;
        let is_core = matches!(srv.nodes.get(node_id).map(|n| &n.state), Some(NodeState::Core(..)));
        if is_core {
            let check = integrity > 0 && srv.integrity_seen % integrity == 0;
            srv.integrity_seen += 1;
            out.push(expand_arm_core(srv, arm, dex, want, check, &mut timings)?);
        } else {
            out.push(expand_arm(srv, arm, dex, want, &mut timings)?);
        }
    }
    whole.stop(&mut timings.total_us);
    Ok(format!("\"arms\":[{}]{}", out.join(","), timings.render_field()))
}

fn expand_arm(
    srv: &mut Server,
    arm: &Json,
    dex: &Dex,
    want: SideWant,
    timings: &mut ArmTimings,
) -> Result<String, String> {
    let node_id = arm.str_at("node_id").ok_or("arm: missing node_id")?.to_string();
    let followup = arm.str_at("followup").unwrap_or("random").to_string();
    let seed = arm.str_at("seed").unwrap_or("original").to_string();
    let recorded_exact = arm.get("recorded_exact").and_then(Json::as_bool).unwrap_or(false);
    let p1_action = arm.str_at("p1_action").unwrap_or("recorded").to_string();
    let p2_action = arm.str_at("p2_action").unwrap_or("recorded").to_string();
    let label = render_id(arm.get("label"));

    let clk_sim = ArmClock::start();
    let (mut sess, resolved) = {
        let node = srv
            .nodes
            .get(&node_id)
            .ok_or_else(|| format!("unknown node {node_id}"))?;
        // Clone the parent and drop its chunk history: what the branch emits from here
        // IS this ply's suffix (the port's answer to Node's `sendUpdates()` + baseline).
        let mut sess = node.engine().snapshot();
        sess.clear_chunks();
        let resolved: Resolved = if recorded_exact {
            // Reproduce the realized turn EXACTLY (the value_crn anchor). Only the root
            // carries the record alignment; never swap the PRNG (realized dice).
            let rec = node
                .record
                .as_ref()
                .ok_or("recorded_exact requested on a non-root node")?;
            resolve_turn_exact(&mut sess, rec, node.rest_idx, dex)?
        } else {
            if seed != "original" {
                sess.reseed(&seed); // THE SWAP: fresh dice for this arm
            }
            let mut rng = aux_rng_from_seed(&seed);
            let spec = [ActionSpec::parse(&p1_action), ActionSpec::parse(&p2_action)];
            resolve_turn(&mut sess, &spec, &followup, &mut rng, dex)
        };
        (sess, resolved)
    };
    clk_sim.stop(&mut timings.sim_us);

    let clk_render = ArmClock::start();
    let ended = sess.is_ended();
    let outcome = outcome_of(&sess, resolved.stuck);
    let requests = if ended { "null".to_string() } else { requests_json(&sess) };
    clk_render.stop(&mut timings.render_us);

    let clk_chunks = ArmClock::start();
    let p1_chunks = want.wants(0).then(|| chunk_array(&sess, 0));
    let p2_chunks = want.wants(1).then(|| chunk_array(&sess, 1));
    clk_chunks.stop(&mut timings.chunks_us);

    // `gen3_one_sided_view_v1` — the arm's resulting board, PROJECTED per side, beside the
    // protocol text that used to be the only way to reach it. Rendered BEFORE `clear_chunks`
    // below (which is chunk-only anyway; the reveal fold is cumulative and survives it).
    let clk_view = ArmClock::start();
    let view_p1 = want.wants(0).then(|| one_sided_view(&sess, 0, dex));
    let view_p2 = want.wants(1).then(|| one_sided_view(&sess, 1, dex));
    // D10 (`gen3_view_at_intermediate_v1`) — the boards at the decisions this ply resolved
    // INTERNALLY, in order, per side. Empty on the ordinary arm; one entry when the ply's
    // faint forced a replacement round, which the two roads otherwise describe one decision
    // apart. Already rendered by `resolve_turn_sourced`; this only splices them in.
    //
    // They are ONE-SIDED payloads like the two above, so `side` elides them on the same terms:
    // a caller that reads `view_p1_at` reads `view_p1`, and neither is shipped for the side it
    // did not ask for.
    let view_p1_at = want.wants(0).then(|| format!("[{}]", resolved.views_at[0].join(",")));
    let view_p2_at = want.wants(1).then(|| format!("[{}]", resolved.views_at[1].join(",")));
    clk_view.stop(&mut timings.view_us);

    let clk_tail = ArmClock::start();
    let used = format!(
        "{{\"p1\":{},\"p2\":{}}}",
        string_array(&resolved.used[0]),
        string_array(&resolved.used[1])
    );
    // A finished battle is a leaf — no child node, so a caller cannot branch past the end.
    let child_id = if ended {
        None
    } else {
        let id = srv.fresh_id();
        sess.clear_chunks();
        srv.nodes.insert(id.clone(), Node { state: NodeState::Plain(sess), record: None, rest_idx: 0 });
        Some(id)
    };
    clk_tail.stop(&mut timings.render_us);

    // 🚨 The four one-sided fields keep their HISTORICAL POSITIONS and order, so a `SideWant::Both`
    // arm renders the same bytes it always did; only an elided side's field vanishes entirely.
    Ok(format!(
        "{{\"label\":{},\"node_id\":{},\"ended\":{},\"stuck\":{},\"outcome\":{},\"requests\":{},\
         \"choices_used\":{}{}{}{}{}{}{}}}",
        label,
        child_id.as_deref().map_or("null".to_string(), json_quote),
        ended,
        resolved.stuck,
        outcome,
        requests,
        used,
        opt_field("p1_chunks", p1_chunks.as_deref()),
        opt_field("p2_chunks", p2_chunks.as_deref()),
        opt_field("view_p1", view_p1.as_deref()),
        opt_field("view_p2", view_p2.as_deref()),
        opt_field("view_p1_at", view_p1_at.as_deref()),
        opt_field("view_p2_at", view_p2_at.as_deref()),
    ))
}

/// ONE arm on the CORE road (`gen3_core_search_v1`): the parent node is a [`BattleVersion`], and
/// the successor is a version too.
///
/// The arm resolves exactly as on the other roads (same clone, same reseed, same follow-up
/// policy, same draws). What differs is what comes back per wanted side, as `core_pN`:
///
/// * `view` / `legal` / `request` — the LEAF version's `present()` view, its legality and its raw
///   request: the version at the first decision this ply opened for the side (a D10 arm: the
///   replacement round inside the turn) or at the end of the turn;
/// * `events` — the readings of the leaf's transition (the ply's `BattleEvent`s, the tracker
///   input), and `mid` — whether the leaf is an intermediate decision;
/// * with the INTEGRITY check sampled for this arm, `text_view` — the same leaf folded the OTHER
///   way (text if the tree is typed, typed if it is text), after the two were asserted equal.
///
/// The child NODE is the leaf version of the side the request asked for (`side`), so a deeper ply
/// branches from the board its leaf describes — including a D10 leaf, which on the other roads
/// sits beside a node one decision past it. With both sides asked for, the child is the end of
/// the turn.
fn expand_arm_core(
    srv: &mut Server,
    arm: &Json,
    dex: &Dex,
    want: SideWant,
    check: bool,
    timings: &mut ArmTimings,
) -> Result<String, String> {
    let node_id = arm.str_at("node_id").ok_or("arm: missing node_id")?.to_string();
    let followup = arm.str_at("followup").unwrap_or("random").to_string();
    let seed = arm.str_at("seed").unwrap_or("original").to_string();
    if arm.get("recorded_exact").and_then(Json::as_bool).unwrap_or(false) {
        return Err("recorded_exact is not served on the core road (the prober's value_crn anchor stays \
                    on the protocol road until it moves to the core, M7)".into());
    }
    let p1_action = arm.str_at("p1_action").unwrap_or("recorded").to_string();
    let p2_action = arm.str_at("p2_action").unwrap_or("recorded").to_string();
    let label = render_id(arm.get("label"));

    let clk_sim = ArmClock::start();
    let (parent, path) = match srv.nodes.get(&node_id).map(|n| &n.state) {
        Some(NodeState::Core(v, p)) => (Arc::clone(v), *p),
        _ => return Err(format!("unknown core node {node_id}")),
    };
    let mut sess = parent.engine().ok_or("core node without an engine")?.snapshot();
    if seed != "original" {
        sess.reseed(&seed);
    }
    let mut rng = aux_rng_from_seed(&seed);
    let spec = [ActionSpec::parse(&p1_action), ActionSpec::parse(&p2_action)];
    let mut resolved =
        resolve_turn_capturing(&mut sess, &spec, &followup, &mut rng, dex, Capture { views: false, sessions: true });
    let mut at = std::mem::take(&mut resolved.sessions_at);
    clk_sim.stop(&mut timings.sim_us);

    let ended = sess.is_ended();
    let outcome = outcome_of(&sess, resolved.stuck);
    let clk_chunks = ArmClock::start();
    let p1_chunks = want.wants(0).then(|| chunk_array(&sess, 0));
    let p2_chunks = want.wants(1).then(|| chunk_array(&sess, 1));
    clk_chunks.stop(&mut timings.chunks_us);

    let fold = |engine: BridgeSession, p: CorePath| -> Result<Arc<BattleVersion>, String> {
        match p {
            CorePath::Typed => parent.child(engine),
            CorePath::Text => parent.child_text(engine),
        }
    };
    let other = |p: CorePath| if p == CorePath::Typed { CorePath::Text } else { CorePath::Typed };
    let clk_core = ArmClock::start();
    let end_engine = if check { Some(sess.snapshot()) } else { None };
    let end = fold(sess, path)?;
    let mut leaves: [Option<(Arc<BattleVersion>, bool)>; 2] = [None, None];
    // The engine each INTERMEDIATE leaf was folded from, kept only for the integrity check's twin.
    let mut mid_engines: [Option<BridgeSession>; 2] = [None, None];
    for s in 0..2 {
        if !want.wants(s) {
            continue;
        }
        leaves[s] = Some(if at[s].is_empty() {
            (Arc::clone(&end), false)
        } else {
            let e = at[s].remove(0);
            if check {
                mid_engines[s] = Some(e.snapshot());
            }
            (fold(e, path)?, true)
        });
    }
    clk_core.stop(&mut timings.core_us);

    // INTEGRITY: the same leaf folded the other way must be the same version, field for field.
    let clk_check = ArmClock::start();
    let mut text_views: [Option<String>; 2] = [None, None];
    if check {
        for s in 0..2 {
            let Some((leaf, mid)) = &leaves[s] else { continue };
            let engine = if *mid {
                mid_engines[s].take().expect("kept when checking")
            } else {
                end_engine.as_ref().expect("taken when checking").snapshot()
            };
            let twin = fold(engine, other(path))?;
            version::streams_equal(leaf, &twin, s).map_err(|e| {
                format!("INTEGRITY: arm {label} (node {node_id}, p{}, {} leaf): the typed and the text path disagree — {e}",
                        s + 1, if *mid { "intermediate" } else { "end-of-turn" })
            })?;
            text_views[s] = Some(twin.view(s)?.json());
        }
    }
    clk_check.stop(&mut timings.integrity_us);

    let clk_render = ArmClock::start();
    let mut core_fields = String::new();
    for s in 0..2 {
        let Some((leaf, mid)) = &leaves[s] else { continue };
        let legal = leaf.legal(s).map_or("null".to_string(), |l| l.json());
        let mut events = String::from("[");
        let mut first = true;
        for ev in leaf.events(s) {
            for r in &ev.readings {
                if !first {
                    events.push(',');
                }
                first = false;
                r.json_into(&mut events);
            }
        }
        events.push(']');
        let mut o = format!(",\"core_p{}\":{{\"view\":{},\"legal\":{},\"request\":", s + 1, leaf.view(s)?.json(), legal);
        pokesim::core_events::json_out::opt_str_into(&mut o, leaf.request(s));
        o.push_str(&format!(",\"events\":{events},\"mid\":{mid}"));
        if let Some(t) = &text_views[s] {
            o.push_str(&format!(",\"text_view\":{t}"));
        }
        o.push('}');
        core_fields.push_str(&o);
    }
    clk_render.stop(&mut timings.core_render_us);

    let used = format!(
        "{{\"p1\":{},\"p2\":{}}}",
        string_array(&resolved.used[0]),
        string_array(&resolved.used[1])
    );
    // The child node: the asked-for side's LEAF (the board its leaf describes), or the end of
    // the turn when both sides were asked for. A finished battle is a leaf with no child.
    let node = match want {
        SideWant::Only(s) => leaves[s].as_ref().map(|(v, _)| Arc::clone(v)).expect("wanted"),
        SideWant::Both => end,
    };
    let requests = if ended { "null".to_string() } else { requests_json(node.engine().expect("step-built")) };
    let child_id = if ended {
        None
    } else {
        let id = srv.fresh_id();
        srv.nodes.insert(id.clone(), Node { state: NodeState::Core(node, path), record: None, rest_idx: 0 });
        Some(id)
    };
    Ok(format!(
        "{{\"label\":{},\"node_id\":{},\"ended\":{},\"stuck\":{},\"outcome\":{},\"requests\":{},\
         \"choices_used\":{}{}{}{}}}",
        label,
        child_id.as_deref().map_or("null".to_string(), json_quote),
        ended,
        resolved.stuck,
        outcome,
        requests,
        used,
        opt_field("p1_chunks", p1_chunks.as_deref()),
        opt_field("p2_chunks", p2_chunks.as_deref()),
        core_fields,
    ))
}

// ===========================================================================
// Shared renderers.
// ===========================================================================

/// `{p1: <request object>, p2: <request object>}`.
///
/// The request payloads are SPLICED IN RAW: [`BridgeSession::active_request_json`]
/// returns the exact `|request|{...}` bytes the wire carried, which is byte-identical
/// to what Showdown's `JSON.stringify(side.activeRequest)` produces (that identity is
/// what `tests/bridge_test.rs` byte-gates). Re-parsing and re-serializing them here
/// could only introduce drift.
fn requests_json(sess: &BridgeSession) -> String {
    let one = |side: usize| -> String {
        match sess.active_request_json(side) {
            Some(line) => line.strip_prefix("|request|").unwrap_or(line).to_string(),
            None => "null".to_string(),
        }
    };
    format!("{{\"p1\":{},\"p2\":{}}}", one(0), one(1))
}

/// `{p1: <choice|null>, p2: <choice|null>}` — the ORIGINAL turn-T picks. Shared by
/// `open_root` and both re-roll verbs (Node builds the same object in three places).
fn recorded_choices_json(rec: &Record, rest_idx: usize) -> String {
    let rc = recorded_turn_choices(rec, rest_idx);
    format!(
        "{{\"p1\":{},\"p2\":{}}}",
        rc.p1.as_deref().map_or("null".to_string(), json_quote),
        rc.p2.as_deref().map_or("null".to_string(), json_quote)
    )
}

fn chunk_array(sess: &BridgeSession, side: usize) -> String {
    string_array(&side_chunk_strings(sess, side))
}

fn string_array(items: &[String]) -> String {
    let parts: Vec<String> = items.iter().map(|s| json_quote(s)).collect();
    format!("[{}]", parts.join(","))
}
