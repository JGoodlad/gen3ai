//! `core_events` — replay recorded battles through the CORE and print its events
//! (`gen3_core_events_v1`, the Rust Core Program's M1). NOT a production transport: nothing in
//! training spawns it. It is the Rust side of the parity harness
//! (`src/agents/battle/rust_core_parity.py`) and the writer of the persisted records.
//!
//! stdin, one or more battles, each a command script:
//!
//! ```text
//! START <json>            {"label","formatid","seed","p1":{name,team},"p2":{…},
//!                          "init_seed":false,"quick_claw":false}
//! CHOOSE <p1|p2> <choice> (every command the live child processed, in order)
//! CHOOSEIF <p1|p2> <choice> (a capture golden's per-decision choice: skipped if the side's
//!                          choice is already accepted at this boundary)
//! FORCELOSE <p1|p2>
//! END
//! ```
//!
//! `seed` is the RAW `>start` seed (the live `sim_bridge` path, which runs the turn-0
//! construction window) unless `init_seed` is true, in which case it is the POST-construction
//! seed a capture golden records (`quick_claw` then carries turn 1's roll).
//!
//! stdout, one JSON line per battle:
//!
//! ```text
//! {"label":…,"ok":true,"error":null,"ended":true,"winner":"p1"|"p2"|null,
//!  "chunks":[[0,[lines…]],…],               // the per-side stream, for the caller's byte check
//!  "viewers":[[<CoreEvent>…],[<CoreEvent>…]]}
//! ```
//!
//! For every battle it REFUSES (`ok:false`) unless: every source record is canonical and
//! renders the log's bytes (one per line), each side's step events re-derive the shipped bytes
//! with per-side conservation, and `parse(side text) == step` on both sides.
//!
//! `--check-records FILE…` instead checks persisted records (round trip + re-parse), one line each.
//!
//! `--views` also captures, at the end of every write that shipped a new `|request|`, BOTH
//! sides' `one_sided_view` and the engine truth (`"views":[{"after","new_request","p1","p2",
//! "truth"},…]`) — slice V of the parity harness (`agents.battle.rust_core_parity_views`).
//!
//! `--record-dir DIR [--commit SHA]` also writes each side's persisted record
//! (`DIR/<label>.p1.jsonl`, `…p2.jsonl`, `core_events::record`), re-reading every file it wrote
//! and refusing unless it round-trips byte-identically and re-parses from its text.

use std::io::{self, BufRead, Write};

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::core_events::json_out;
use pokesim::core_events::parse::{parse, parse_matches_step};
use pokesim::core_events::record::{self, Header, Path, Record};
use pokesim::core_events::{CoreEvent, Line};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::prng::normalize_seed;
use pokesim::version::{self, BattleVersion};
use pokesim::{search, view};

struct Battle {
    label: String,
    opts: BattleOptions,
    init_seed: bool,
    quick_claw: bool,
    cmds: Vec<Script>,
}

enum Script {
    Choose(Cmd),
    /// A capture golden's per-decision choice: fed only if the side's choice is still open
    /// (the golden repeats an ACCEPTED side's choice after the other side's was rejected; the
    /// sim's `side.choose` keeps the first accepted one).
    ChooseIfOpen(Cmd),
    ForceLose(usize),
}

fn player(v: &Json, key: &str) -> Result<PlayerOptions, String> {
    let p = v.get(key).ok_or_else(|| format!("START: missing {key}"))?;
    Ok(PlayerOptions {
        name: p.str_at("name").ok_or("START: name")?.to_string(),
        team: PackedTeam(p.str_at("team").ok_or("START: team")?.to_string()),
    })
}

fn seed_of(v: &Json) -> Result<String, String> {
    let j = v.get("seed").filter(|s| !s.is_null()).ok_or("START: an explicit seed is required")?;
    if let Some(a) = j.as_array() {
        let parts: Vec<String> = a
            .iter()
            .map(|x| x.as_f64().map(|n| format!("{}", n as u64)).ok_or("seed element"))
            .collect::<Result<_, _>>()?;
        Ok(parts.join(","))
    } else {
        Ok(normalize_seed(j.as_str().ok_or("seed must be a string or array")?))
    }
}

fn side_of(tok: &str) -> Result<usize, String> {
    match tok {
        "p1" => Ok(0),
        "p2" => Ok(1),
        o => Err(format!("bad side {o}")),
    }
}

/// One DECISION-BOARD capture (`--views`, slice V of the parity harness), taken at the end of a
/// write that shipped a new `|request|` to at least one side. `after` = the number of per-side
/// chunks flushed so far, so the Python reference can feed each viewer exactly the chunks that
/// preceded the board. Per side: the port's [`one_sided_view`] projection (the view road's), the
/// CORE's `present()` view (the stream-built reading, M2) and its legality, and the core's TRUTH
/// AUDIT of that view against the board.
struct ViewCap {
    after: usize,
    new_request: [bool; 2],
    views: [String; 2],
    truth: String,
    core: [String; 2],
    legal: [String; 2],
    audit: [String; 2],
}

/// The engine facts a one-sided view does NOT carry but the truth audit compares against: per
/// side, per mon (engine roster order), the SIM's volatile set (`search::volatile_names` — the
/// typed fields) and its raw status counter. Read by `agents.battle.rust_core_parity_views`.
fn truth_json(sess: &BridgeSession, dex: &Dex) -> String {
    use pokesim::state::Status;
    let Some(st) = sess.battle_state() else { return "null".into() };
    let mut o = String::from("[");
    for (s, sd) in st.sides.iter().enumerate() {
        if s > 0 {
            o.push(',');
        }
        o.push('[');
        for (i, m) in sd.pokemon.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            let (status, counter) = match m.status {
                Some(Status::Sleep(n)) => ("slp", n as i64),
                Some(Status::Toxic(n)) => ("tox", n as i64),
                Some(Status::Burn) => ("brn", 0),
                Some(Status::Paralysis) => ("par", 0),
                Some(Status::Freeze) => ("frz", 0),
                Some(Status::Poison) => ("psn", 0),
                None => ("", 0),
            };
            o.push_str("{\"name\":");
            json_out::str_into(&mut o, &view::display_name(m, dex));
            o.push_str(",\"species\":");
            json_out::str_into(&mut o, &m.species_id);
            o.push_str(&format!(
                ",\"active\":{},\"status\":\"{status}\",\"status_n\":{counter},\"sleep_skipped\":{},\"vol\":[",
                i == sd.active && !m.fainted,
                m.sleep_skipped
            ));
            for (k, v) in search::volatile_names(m).iter().enumerate() {
                if k > 0 {
                    o.push(',');
                }
                json_out::str_into(&mut o, v);
            }
            o.push_str("]}");
        }
        o.push(']');
    }
    o.push(']');
    o
}

/// `|request|` lines shipped to each side in `chunks[from..]`.
fn requests_since(sess: &BridgeSession, from: usize) -> [usize; 2] {
    let mut n = [0usize; 2];
    for c in sess.chunks().chunks.iter().skip(from) {
        n[c.side] += c.lines.iter().filter(|l| l.starts_with("|request|")).count();
    }
    n
}


fn capture(v: &BattleVersion, dex: &Dex, seen: &mut usize, caps: &mut Vec<ViewCap>) -> Result<(), String> {
    let sess = v.engine().ok_or("a step-built version without an engine")?;
    let total = sess.chunks().chunks.len();
    let n = requests_since(sess, *seen);
    *seen = total;
    if n == [0, 0] {
        return Ok(());
    }
    let side_json = |s: usize| -> Result<(String, String, String), String> {
        let core = v.view(s)?.json();
        let legal = v.legal(s).map_or("null".to_string(), |l| l.json());
        let audit = v.audit(s, dex)?.json();
        Ok((core, legal, audit))
    };
    let (c0, l0, a0) = side_json(0)?;
    let (c1, l1, a1) = side_json(1)?;
    caps.push(ViewCap {
        after: total,
        new_request: [n[0] > 0, n[1] > 0],
        views: [view::one_sided_view(sess, 0, dex)?, view::one_sided_view(sess, 1, dex)?],
        truth: truth_json(sess, dex),
        core: [c0, c1],
        legal: [l0, l1],
        audit: [a0, a1],
    });
    Ok(())
}

type Run = (BridgeSession, [Vec<CoreEvent>; 2], Vec<ViewCap>);

/// The M2 gate at ONE transition: each side's parse-built version, fed the same new TEXT the
/// step-built one folded typed, must agree with it on the whole reading board, the transition's
/// events and the view (`version::parse_matches_step`).
fn parse_gate(v: &BattleVersion, parsed: &mut [Option<BattleVersion>; 2]) -> Result<(), String> {
    let sess = v.engine().ok_or("no engine")?;
    for side in 0..2 {
        let p = parsed[side].take().ok_or("parse chain lost")?;
        let from = p.stream(side).map_or(0, |s| s.lines);
        let text = sess.side_lines(side);
        let next = p.parse_advance(&text[from..]).map_err(|e| format!("parse p{}: {e}", side + 1))?;
        version::parse_matches_step(v, &next, side).map_err(|e| format!("version parse != step: {e}"))?;
        parsed[side] = Some(next);
    }
    Ok(())
}

fn run(b: &Battle, dex: &Dex, record_dir: Option<&str>, commit: &str, views: bool) -> Result<Run, String> {
    let mut sess = if b.init_seed {
        BridgeSession::new_core(&b.opts, b.quick_claw, dex)?
    } else {
        BridgeSession::new_construct_turn0_core(&b.opts, dex)?
    };
    if views {
        // Slice V compares the view road's projection too, so this replay folds reveals
        // (`gen3_view_fold_opt_in_v1` — off unless a reader asks).
        sess.enable_view_fold()?;
    }
    // The battle is replayed as a CHAIN OF VERSIONS (`gen3_core_version_v1`): each command
    // advances the engine and folds each side's new lines, typed at the source, into its stream;
    // a one-side, engine-less parse chain is fed the same text and must agree at every step.
    let names = [b.opts.p1.name.as_str(), b.opts.p2.name.as_str()];
    let teams = [Some(b.opts.p1.team.0.as_str()), Some(b.opts.p2.team.0.as_str())];
    let mut v = BattleVersion::root(sess, names, teams, false, [true, true]).map_err(|e| format!("version root: {e}"))?;
    let mut parsed = [
        Some(BattleVersion::parse_root(0, names[0], teams[0])?),
        Some(BattleVersion::parse_root(1, names[1], teams[1])?),
    ];
    parse_gate(&v, &mut parsed)?;
    let mut caps: Vec<ViewCap> = Vec::new();
    let mut seen = 0usize;
    if views {
        capture(&v, dex, &mut seen, &mut caps)?;
    }
    for c in &b.cmds {
        let e = v.engine().ok_or("no engine")?;
        if e.is_ended() {
            break;
        }
        let skip = matches!(c, Script::ChooseIfOpen(cmd) if e.is_choice_done(cmd.side));
        if skip {
            continue;
        }
        v = v
            .advance_with(|e| {
                match c {
                    Script::Choose(cmd) | Script::ChooseIfOpen(cmd) => e.feed_cmd(cmd.clone(), dex),
                    Script::ForceLose(s) => e.forfeit(*s),
                }
                Ok(())
            })
            .map_err(|e| format!("version step: {e}"))?;
        if let Some(f) = v.engine().and_then(|e| e.fatal()) {
            // A capture golden's blind per-decision script can re-send a rejected choice until
            // the bridge's no-progress cap fails loud (the capture ran into a stall loop): the
            // battle is TRUNCATED there, and everything emitted so far is still checked. A LIVE
            // recording (`CHOOSE`) never does this, so there it is a refusal.
            if matches!(c, Script::ChooseIfOpen(_)) {
                break;
            }
            return Err(format!("bridge fatal: {f}"));
        }
        parse_gate(&v, &mut parsed)?;
        if views {
            capture(&v, dex, &mut seen, &mut caps)?;
        }
    }
    let sess = v.into_engine().ok_or("no engine")?;
    let bs = sess.battle_state().ok_or("no battle state")?;
    let recs = sess.source_recs().ok_or("no source records")?;
    let log = bs.log.lines();
    if recs.len() != log.len() {
        return Err(format!("CONSERVATION: {} source records vs {} log lines", recs.len(), log.len()));
    }
    for (i, (r, l)) in recs.iter().zip(log).enumerate() {
        if r.line.render() != l.0 || Line::parse(&l.0).as_ref() != Ok(&r.line) {
            return Err(format!("source record {i} is not the canonical typing of {:?}", l.0));
        }
    }
    let mut out: [Vec<CoreEvent>; 2] = [Vec::new(), Vec::new()];
    for side in 0..2 {
        let step = sess.core_events(side)?;
        let parsed = parse(&sess.side_lines(side), side).map_err(|e| format!("parse p{}: {e}", side + 1))?;
        parse_matches_step(&step, &parsed).map_err(|e| format!("parse != step (p{}): {e}", side + 1))?;
        if let Some(dir) = record_dir {
            let r = Record::new(
                Header {
                    event_schema: record::EVENT_SCHEMA.into(),
                    core_commit: commit.to_string(),
                    path: Path::Step,
                    viewer: side as u8,
                    format: b.opts.format_id.clone(),
                    showdown_version: None,
                    battle: b.label.clone(),
                },
                &step,
            );
            let bytes = record::write(&r);
            let path = format!("{dir}/{}.p{}.jsonl", b.label, side + 1);
            std::fs::write(&path, &bytes).map_err(|e| format!("{path}: {e}"))?;
            let back = record::read(&std::fs::read_to_string(&path).map_err(|e| e.to_string())?)?;
            if record::write(&back) != bytes {
                return Err(format!("{path}: the record does not round-trip byte-identically"));
            }
            record::check_reparse(&back).map_err(|e| format!("{path}: {e}"))?;
        }
        out[side] = step;
    }
    Ok((sess, out, caps))
}

fn render(b: &Battle, res: Result<Run, String>) -> String {
    let mut o = String::from("{\"label\":");
    json_out::str_into(&mut o, &b.label);
    match res {
        Err(e) => {
            o.push_str(",\"ok\":false,\"error\":");
            json_out::str_into(&mut o, &e);
            o.push('}');
        }
        Ok((sess, viewers, caps)) => {
            o.push_str(",\"ok\":true,\"error\":null,\"ended\":");
            o.push_str(if sess.is_ended() { "true" } else { "false" });
            o.push_str(",\"truncated\":");
            o.push_str(if sess.fatal().is_some() { "true" } else { "false" });
            o.push_str(",\"winner\":");
            match sess.winner() {
                Some(0) => o.push_str("\"p1\""),
                Some(_) => o.push_str("\"p2\""),
                None => o.push_str("null"),
            }
            o.push_str(",\"chunks\":[");
            for (i, c) in sess.chunks().chunks.iter().enumerate() {
                if i > 0 {
                    o.push(',');
                }
                o.push_str(&format!("[{},[", c.side));
                for (j, l) in c.lines.iter().enumerate() {
                    if j > 0 {
                        o.push(',');
                    }
                    json_out::str_into(&mut o, l);
                }
                o.push_str("]]");
            }
            o.push_str("],\"viewers\":[");
            for (s, evs) in viewers.iter().enumerate() {
                if s > 0 {
                    o.push(',');
                }
                o.push('[');
                for (i, e) in evs.iter().enumerate() {
                    if i > 0 {
                        o.push(',');
                    }
                    e.json_into(&mut o);
                }
                o.push(']');
            }
            o.push(']');
            if !caps.is_empty() {
                o.push_str(",\"views\":[");
                for (i, c) in caps.iter().enumerate() {
                    if i > 0 {
                        o.push(',');
                    }
                    o.push_str(&format!(
                        "{{\"after\":{},\"new_request\":[{},{}],\"p1\":{},\"p2\":{},\"truth\":{},\
                         \"core\":[{},{}],\"legal\":[{},{}],\"audit\":[{},{}]}}",
                        c.after, c.new_request[0], c.new_request[1], c.views[0], c.views[1], c.truth,
                        c.core[0], c.core[1], c.legal[0], c.legal[1], c.audit[0], c.audit[1]
                    ));
                }
                o.push(']');
            }
            o.push('}');
        }
    }
    o
}

/// `--present-stream`: ONE side's protocol TEXT on stdin → its `present()` view. The first line is
/// a JSON header `{"viewer":0|1,"username":…,"team":<packed>|null}`; every later line
/// is a protocol line of that side's stream (the parse path, exactly what a server sends). Prints
/// ONE JSON object: `{"ok","error","view","legal","request"}`. The pin of every reading rule
/// against poke-env itself (`agents/battle/rust_core_present_test.py`) runs through it.
fn present_stream() -> i32 {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let head = match lines.next() {
        Some(Ok(h)) => h,
        _ => {
            eprintln!("core_events --present-stream: no header");
            return 2;
        }
    };
    let res = (|| -> Result<String, String> {
        let h = Json::parse(&head).map_err(|e| format!("header: {e}"))?;
        let viewer = h.get("viewer").and_then(|v| v.as_f64()).ok_or("header viewer")? as usize;
        let username = h.str_at("username").ok_or("header username")?.to_string();
        let team = h.str_at("team").map(str::to_string);
        let mut s = version::SideStream::new(viewer, &username, team.as_deref())?;
        for l in lines {
            let l = l.map_err(|e| e.to_string())?;
            s.fold_text(&l)?;
        }
        let view = s.view()?.json();
        let legal = pokesim::present::legal_actions(&s.tracker).map_or("null".to_string(), |l| l.json());
        let mut o = format!("{{\"ok\":true,\"error\":null,\"view\":{view},\"legal\":{legal},\"request\":");
        json_out::opt_str_into(&mut o, s.tracker.last_request_text.as_deref());
        o.push('}');
        Ok(o)
    })();
    match res {
        Ok(o) => println!("{o}"),
        Err(e) => {
            let mut o = String::from("{\"ok\":false,\"error\":");
            json_out::str_into(&mut o, &e);
            o.push('}');
            println!("{o}");
        }
    }
    0
}

/// `--check-records FILE…`: every record must round-trip byte-identically and re-parse from its
/// stored text to its stored typed stream. One line per file: `ok <file>` or `FAIL <file>: why`.
fn check_records(files: &[String]) -> i32 {
    let mut bad = 0;
    for f in files {
        let res = (|| -> Result<(), String> {
            let bytes = std::fs::read_to_string(f).map_err(|e| e.to_string())?;
            let r = record::read(&bytes)?;
            if record::write(&r) != bytes {
                return Err("write(read(bytes)) != bytes".into());
            }
            record::check_reparse(&r)
        })();
        match res {
            Ok(()) => println!("ok {f}"),
            Err(e) => {
                bad += 1;
                println!("FAIL {f}: {e}");
            }
        }
    }
    (bad > 0) as i32
}

/// `--bench-parse ROUNDS`: the cost of `parse` as TRAINING would pay it — ONE side's stream read
/// incrementally, decision by decision (a decision's lines = everything since the previous
/// `|request|`, the request included), through `Line::parse` + the reading fold. Each battle on
/// stdin is replayed once to get its per-side text; the timing loop then touches text only.
/// Prints one JSON object: the median over ROUNDS of the per-decision cost, and the line counts.
fn bench_parse(rounds: usize, battles: &[Battle], dex: &Dex) -> String {
    use pokesim::core_events::reading::Reader;
    use std::hint::black_box;
    use std::time::Instant;
    let mut streams: Vec<(usize, Vec<Vec<String>>)> = Vec::new();
    for b in battles {
        let (sess, _, _) = match run(b, dex, None, "bench", false) {
            Ok(x) => x,
            Err(e) => return format!("{{\"error\":{:?}}}", e),
        };
        for side in 0..2 {
            let mut decisions: Vec<Vec<String>> = vec![Vec::new()];
            for l in sess.side_lines(side) {
                let is_req = l.starts_with("|request|");
                decisions.last_mut().unwrap().push(l);
                if is_req {
                    decisions.push(Vec::new());
                }
            }
            decisions.retain(|d| !d.is_empty());
            streams.push((side, decisions));
        }
    }
    let n_dec: usize = streams.iter().map(|(_, d)| d.len()).sum();
    let n_lines: usize = streams.iter().flat_map(|(_, d)| d.iter()).map(|d| d.len()).sum();
    let n_req: usize = streams.iter().flat_map(|(_, d)| d.iter()).flatten().filter(|l| l.starts_with("|request|")).count();
    let mut per_round: Vec<f64> = Vec::new();
    let mut req_ns: Vec<f64> = Vec::new();
    for _ in 0..rounds {
        // Pass 1 — the headline: one timer per DECISION (no per-line timer overhead).
        let mut total = 0u128;
        for (side, decisions) in &streams {
            let mut reader = Reader::new(*side);
            for d in decisions {
                let t0 = Instant::now();
                for l in d {
                    let line = Line::parse(l).expect("parse");
                    black_box(reader.feed(&line).expect("feed"));
                }
                total += t0.elapsed().as_nanos();
            }
        }
        // Pass 2 — the `|request|` share (a per-line timer; approximate by its own overhead).
        let mut req = 0u128;
        for (side, decisions) in &streams {
            let mut reader = Reader::new(*side);
            for d in decisions {
                for l in d {
                    let t0 = Instant::now();
                    let line = Line::parse(l).expect("parse");
                    black_box(reader.feed(&line).expect("feed"));
                    if l.starts_with("|request|") {
                        req += t0.elapsed().as_nanos();
                    }
                }
            }
        }
        per_round.push(total as f64 / n_dec as f64);
        req_ns.push(req as f64 / n_dec as f64);
    }
    per_round.sort_by(|a, b| a.partial_cmp(b).unwrap());
    req_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = |v: &Vec<f64>| v[v.len() / 2];
    format!(
        "{{\"rounds\":{rounds},\"battles\":{},\"decisions\":{n_dec},\"lines\":{n_lines},\"request_lines\":{n_req},\"ns_per_decision_median\":{:.0},\"ns_per_decision_min\":{:.0},\"ns_per_decision_max\":{:.0},\"request_ns_per_decision_median\":{:.0}}}",
        battles.len(),
        med(&per_round),
        per_round[0],
        per_round[per_round.len() - 1],
        med(&req_ns)
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--check-records") {
        std::process::exit(check_records(&args[2..]));
    }
    if args.get(1).map(String::as_str) == Some("--present-stream") {
        std::process::exit(present_stream());
    }
    let bench_rounds: Option<usize> = if args.get(1).map(String::as_str) == Some("--bench-parse") {
        Some(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5))
    } else {
        None
    };
    let mut bench_battles: Vec<Battle> = Vec::new();
    let mut record_dir: Option<String> = None;
    let mut views = false;
    let mut commit = "unknown".to_string();
    let mut i = if bench_rounds.is_some() { args.len() } else { 1 };
    while i < args.len() {
        match args[i].as_str() {
            "--record-dir" => {
                record_dir = args.get(i + 1).cloned();
                i += 1;
            }
            "--views" => views = true,
            "--commit" => {
                commit = args.get(i + 1).cloned().unwrap_or_default();
                i += 1;
            }
            o => {
                eprintln!("core_events: unknown argument {o}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let dex = Dex::for_gen(3);
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    let mut cur: Option<Battle> = None;
    let mut n = 0usize;
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("core_events: stdin: {e}");
                std::process::exit(1);
            }
        };
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        let (cmd, rest) = line.split_once(' ').unwrap_or((line, ""));
        let step: Result<(), String> = (|| {
            match cmd {
                "START" => {
                    let v = Json::parse(rest).map_err(|e| format!("START JSON: {e}"))?;
                    let label = v.str_at("label").map(str::to_string).unwrap_or_else(|| format!("battle{n}"));
                    cur = Some(Battle {
                        label,
                        opts: BattleOptions {
                            format_id: v.str_at("formatid").ok_or("formatid")?.to_string(),
                            seed: Some(seed_of(&v)?),
                            p1: player(&v, "p1")?,
                            p2: player(&v, "p2")?,
                        },
                        init_seed: v.bool_or("init_seed", false),
                        quick_claw: v.bool_or("quick_claw", false),
                        cmds: Vec::new(),
                    });
                }
                "CHOOSE" | "CHOOSEIF" => {
                    let b = cur.as_mut().ok_or("CHOOSE before START")?;
                    let (side, choice) = rest.split_once(' ').unwrap_or((rest, ""));
                    let choice = parse_choice(choice).ok_or_else(|| format!("bad choice {choice:?}"))?;
                    let c = Cmd { side: side_of(side)?, choice };
                    b.cmds.push(if cmd == "CHOOSE" { Script::Choose(c) } else { Script::ChooseIfOpen(c) });
                }
                "FORCELOSE" => {
                    let b = cur.as_mut().ok_or("FORCELOSE before START")?;
                    b.cmds.push(Script::ForceLose(side_of(rest.trim())?));
                }
                "END" => {
                    let b = cur.take().ok_or("END before START")?;
                    if bench_rounds.is_some() {
                        bench_battles.push(b);
                        return Ok(());
                    }
                    let res = run(&b, &dex, record_dir.as_deref(), &commit, views);
                    let _ = writeln!(out, "{}", render(&b, res));
                    let _ = out.flush();
                    n += 1;
                }
                o => return Err(format!("unknown command {o}")),
            }
            Ok(())
        })();
        if let Err(e) = step {
            eprintln!("core_events: {e}");
            std::process::exit(1);
        }
    }
    if let Some(r) = bench_rounds {
        let _ = writeln!(out, "{}", bench_parse(r, &bench_battles, &dex));
    }
}
