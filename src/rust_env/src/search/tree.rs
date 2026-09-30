//! THE SEARCH TREE — `search_driver`'s CORE road (`open_root` with `core: "text"` + `expand_many`
//! with `rows`), in process. Every node is a step-built [`BattleVersion`] (an `Arc`: a child shares
//! its parent), every arm resolves through the port's own kernels (`pokesim::search`: the same
//! clone, reseed, aux RNG, follow-up policy and D10 capture), and each wanted side's leaf row is
//! ENCODED straight into the caller's buffer instead of a base64 wire frame.
//!
//! BYTE-COMPATIBILITY IS THE CONTRACT (gate: `src/utils/rust_env/successors_parity_test.py`, the
//! depth-3 successor slice): for the same request sequence the root body, and every arm's label,
//! node id, `ended`, `stuck`, `outcome`, `requests`, `choices_used`, `pN_chunks`, `mid`, row BYTES,
//! mask and tokens equal the `search_driver` binary's. Node ids follow its rule exactly: `n0, n1, …`
//! MONOTONIC for the handle's life (the root first, then each non-terminal arm's child in arm
//! order), and a fresh root drops the previous tree, so a stale id never resolves.
//!
//! What the in-process road does NOT serve (refused by name, never silently degraded): a non-core
//! root (`core` absent), the view / events payload (`rows` false), `recorded_exact`, `integrity`.

use std::sync::Arc;

use pokesim::encoder::OBS_DIM;
use pokesim::json::Json;
use pokesim::search::{
    aux_rng_from_seed, build_to_turn, json_quote, outcome_of, pre_state, recorded_turn_choices, resolve_turn_capturing,
    session_from_record, side_chunk_strings, ActionSpec, Capture, Record,
};
use pokesim::trackers::clock::ClockConfig;
use pokesim::version::BattleVersion;

use super::dex;

/// A search tree (see the module docs).
pub struct Tree {
    clock: ClockConfig,
    max_nodes: usize,
    /// The CURRENT tree's nodes; node id `n{base + i}` is `nodes[i]`.
    nodes: Vec<Arc<BattleVersion>>,
    base: u64,
    /// The next id to mint (monotonic for the handle's life).
    counter: u64,
    /// Nodes created (roots + children) and arms expanded, over the handle's life.
    pub nodes_made: u64,
    pub arms: u64,
}

/// Where each arm's rows go: `rows[k]` is slot `k` (`OBS_DIM` f32 each); `cap` slots.
pub struct RowSink<'a> {
    pub rows: &'a mut [f32],
    pub used: usize,
}

impl RowSink<'_> {
    pub fn cap(&self) -> usize {
        self.rows.len() / OBS_DIM
    }
    fn next(&mut self) -> Result<(usize, &mut [f32; OBS_DIM]), String> {
        let k = self.used;
        if k >= self.cap() {
            return Err(format!("expand: the row buffer holds {} rows; this batch needs more (one per arm per wanted side)", self.cap()));
        }
        self.used += 1;
        let row: &mut [f32; OBS_DIM] = (&mut self.rows[k * OBS_DIM..(k + 1) * OBS_DIM]).try_into().expect("row slice");
        Ok((k, row))
    }
}

fn node_index(base: u64, len: usize, id: &str) -> Option<usize> {
    let n: u64 = id.strip_prefix('n')?.parse().ok()?;
    let i = n.checked_sub(base)? as usize;
    (i < len).then_some(i)
}

impl Tree {
    pub fn new(clock: ClockConfig, max_nodes: usize) -> Tree {
        Tree { clock, max_nodes, nodes: Vec::with_capacity(max_nodes), base: 0, counter: 0, nodes_made: 0, arms: 0 }
    }

    fn mint(&mut self, v: Arc<BattleVersion>) -> Result<String, String> {
        if self.nodes.len() >= self.max_nodes {
            return Err(format!(
                "the tree holds its declared max_nodes = {} (the spec's node table is never grown; open a new root or declare more)",
                self.max_nodes
            ));
        }
        let id = format!("n{}", self.counter);
        self.counter += 1;
        self.nodes.push(v);
        self.nodes_made += 1;
        Ok(id)
    }

    /// Nodes in the current tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// `open_root` — `search_driver`'s request object (`turn`, `record`, `core: "text"`, `side`,
    /// `trackers`); returns its reply BODY (the fields between `"ok":true,` and `}`), byte for byte.
    pub fn open_root(&mut self, req: &Json) -> Result<String, String> {
        let t_raw = req.get("turn").and_then(Json::as_f64);
        let turn = match t_raw {
            Some(t) if t.fract() == 0.0 && t >= 1.0 => t as u32,
            _ => return Err(format!("invalid turn {}", t_raw.map_or("undefined".to_string(), render_num))),
        };
        let rec = Record::parse(req.get("record").ok_or("open_root: missing record")?)?;
        match req.str_at("core") {
            Some("text") => {}
            None => return Err("open_root: the in-process road serves the CORE road only — send core \"text\"".into()),
            Some(other) => return Err(format!("open_root: core must be \"text\", got \"{other}\"")),
        }
        let want = match req.str_at("side") {
            None => [true, true],
            Some("p1") => [true, false],
            Some("p2") => [false, true],
            Some(other) => return Err(format!("open_root: side must be \"p1\" or \"p2\", got \"{other}\"")),
        };
        let trackers = req.get("trackers").and_then(Json::as_bool).unwrap_or(false);
        if !trackers {
            return Err("open_root: the in-process road serves ROWS, which need the trackers — send trackers true".into());
        }
        // A fresh root starts a fresh tree (the previous one is dropped; ids stay monotonic).
        self.nodes.clear();
        self.base = self.counter;
        let dex = dex();
        let mut sess = session_from_record(&rec, dex)?;
        let rest_idx = build_to_turn(&mut sess, &rec, turn, dex)?;
        let requests = requests_json(&sess);
        let rc = recorded_turn_choices(&rec, rest_idx);
        let recorded = format!(
            "{{\"p1\":{},\"p2\":{}}}",
            rc.p1.as_deref().map_or("null".to_string(), json_quote),
            rc.p2.as_deref().map_or("null".to_string(), json_quote)
        );
        let ps = pre_state(&sess);
        let p1 = string_array(&side_chunk_strings(&sess, 0));
        let p2 = string_array(&side_chunk_strings(&sess, 1));
        let names = [rec.p1.name.as_str(), rec.p2.name.as_str()];
        let teams = [Some(rec.p1.team.0.as_str()), Some(rec.p2.team.0.as_str())];
        let root = BattleVersion::root_with(sess, names, teams, want, Some(self.clock)).map_err(|e| e.to_string())?;
        let node_id = self.mint(Arc::new(root))?;
        Ok(format!(
            "\"node_id\":{},\"requests\":{},\"recorded_choices\":{},\"pre_state\":{},\"prefix_p1_chunks\":{},\"prefix_p2_chunks\":{}",
            json_quote(&node_id),
            requests,
            recorded,
            ps,
            p1,
            p2
        ))
    }

    /// `expand_many` with `rows` — `search_driver`'s request object (`arms`, `side`, `rows: true`,
    /// and one in-process key, `chunks`: `false` skips rendering `pN_chunks`, which no core-road
    /// reader reads). Returns the reply BODY; each wanted side's row is written into `sink` and its
    /// `core_pN.row` is the SLOT index (`null` when the leaf is not the side's decision).
    pub fn expand(&mut self, req: &Json, sink: &mut RowSink) -> Result<String, String> {
        let empty: Vec<Json> = Vec::new();
        let arms = req.get("arms").and_then(Json::as_array).unwrap_or(&empty);
        let want: Option<usize> = match req.str_at("side") {
            None => None,
            Some("p1") => Some(0),
            Some("p2") => Some(1),
            Some(other) => return Err(format!("expand_many: side must be \"p1\" or \"p2\", got \"{other}\"")),
        };
        if req.get("integrity").is_some_and(|v| !v.is_null()) {
            return Err("expand_many: integrity is DELETED (program §4 M4)".into());
        }
        if !req.get("rows").and_then(Json::as_bool).unwrap_or(false) {
            return Err("expand_many: the in-process road serves ROWS only — send rows true".into());
        }
        let chunks = req.get("chunks").and_then(Json::as_bool).unwrap_or(true);
        let mut out = Vec::with_capacity(arms.len());
        for arm in arms {
            out.push(self.expand_arm(arm, want, chunks, sink)?);
            self.arms += 1;
        }
        Ok(format!("\"arms\":[{}]", out.join(",")))
    }

    /// ONE arm — `search_driver::expand_arm_core`, statement for statement.
    fn expand_arm(&mut self, arm: &Json, want: Option<usize>, chunks: bool, sink: &mut RowSink) -> Result<String, String> {
        let wants = |s: usize| want.is_none_or(|w| w == s);
        let node_id = arm.str_at("node_id").ok_or("arm: missing node_id")?.to_string();
        let followup = arm.str_at("followup").unwrap_or("random").to_string();
        let seed = arm.str_at("seed").unwrap_or("original").to_string();
        if arm.get("recorded_exact").and_then(Json::as_bool).unwrap_or(false) {
            return Err("recorded_exact is not served on the core road".into());
        }
        let p1_action = arm.str_at("p1_action").unwrap_or("recorded").to_string();
        let p2_action = arm.str_at("p2_action").unwrap_or("recorded").to_string();
        let label = render_id(arm.get("label"));
        let parent = match node_index(self.base, self.nodes.len(), &node_id) {
            Some(i) => Arc::clone(&self.nodes[i]),
            None => return Err(format!("unknown core node {node_id}")),
        };
        let dex = dex();
        let mut sess = parent.fork_session().map_err(|e| e.to_string())?;
        if seed != "original" {
            sess.reseed(&seed);
        }
        let mut rng = aux_rng_from_seed(&seed);
        let spec = [ActionSpec::parse(&p1_action), ActionSpec::parse(&p2_action)];
        let mut resolved = resolve_turn_capturing(&mut sess, &spec, &followup, &mut rng, dex, Capture { sessions: true });
        let mut at = std::mem::take(&mut resolved.sessions_at);
        let ended = sess.is_ended();
        let outcome = outcome_of(&sess, resolved.stuck);
        let p1_chunks = (chunks && wants(0)).then(|| string_array(&side_chunk_strings(&sess, 0)));
        let p2_chunks = (chunks && wants(1)).then(|| string_array(&side_chunk_strings(&sess, 1)));
        let end = parent.child(sess).map_err(|e| e.to_string())?;
        let mut leaves: [Option<(Arc<BattleVersion>, bool)>; 2] = [None, None];
        for (s, leaf) in leaves.iter_mut().enumerate() {
            if !wants(s) {
                continue;
            }
            *leaf = Some(if at[s].is_empty() {
                (Arc::clone(&end), false)
            } else {
                let e = at[s].remove(0);
                (parent.child(e).map_err(|e| e.to_string())?, true)
            });
        }
        let mut core_fields = String::new();
        for (s, leaf) in leaves.iter().enumerate() {
            let Some((leaf, mid)) = leaf else { continue };
            core_fields.push_str(&row_payload(leaf, s, *mid, sink)?);
        }
        let used = format!("{{\"p1\":{},\"p2\":{}}}", string_array(&resolved.used[0]), string_array(&resolved.used[1]));
        let node = match want {
            Some(s) => leaves[s].as_ref().map(|(v, _)| Arc::clone(v)).expect("wanted"),
            None => end,
        };
        let requests = if ended {
            "null".to_string()
        } else {
            requests_json_engine(node.engine().ok_or("a core node without an engine")?)
        };
        let child_id = if ended { None } else { Some(self.mint(node)?) };
        let opt = |name: &str, v: Option<String>| v.map_or(String::new(), |v| format!(",\"{name}\":{v}"));
        Ok(format!(
            "{{\"label\":{},\"node_id\":{},\"ended\":{},\"stuck\":{},\"outcome\":{},\"requests\":{},\"choices_used\":{}{}{}{}}}",
            label,
            child_id.as_deref().map_or("null".to_string(), json_quote),
            ended,
            resolved.stuck,
            outcome,
            requests,
            used,
            opt("p1_chunks", p1_chunks),
            opt("p2_chunks", p2_chunks),
            core_fields,
        ))
    }
}

/// `,"core_pN":{"mid":…,"row":<slot>|null,"mask":[…],"tokens":{…}}` — `search_driver`'s
/// `core_row_payload` with the row written to `sink` (slot index in place of the wire frame).
fn row_payload(leaf: &BattleVersion, s: usize, mid: bool, sink: &mut RowSink) -> Result<String, String> {
    let lines = leaf.stream(s).map_or(0, |x| x.lines);
    let decided = leaf.decision(s).is_some_and(|d| d.line + 1 == lines);
    if !decided {
        return Ok(format!(",\"core_p{}\":{{\"mid\":{mid},\"row\":null}}", s + 1));
    }
    let (slot, row) = sink.next()?;
    leaf.encode(s, row).map_err(|e| e.to_string())?;
    let legal = leaf.legal(s).ok_or("a decision with no legality")?;
    let reading = &leaf.stream(s).ok_or("no stream")?.board_reading;
    let tokens = pokesim::present::choice_tokens(reading, &legal).map_err(|e| e.to_string())?;
    Ok(format!(
        ",\"core_p{}\":{{\"mid\":{mid},\"row\":{slot},\"mask\":{:?},\"tokens\":{}}}",
        s + 1,
        pokesim::present::mask(&legal),
        pokesim::present::tokens_json(&tokens)
    ))
}

/// `search_driver::render_id` — a label echoed as JSON (`null` when absent).
fn render_id(v: Option<&Json>) -> String {
    match v {
        None | Some(Json::Null) => "null".to_string(),
        Some(Json::Str(s)) => json_quote(s),
        Some(Json::Num(n)) => render_num(*n),
        Some(Json::Bool(b)) => b.to_string(),
        Some(_) => "null".to_string(),
    }
}

/// `search_driver::render_num` — JS `JSON.stringify` of a number (integers without a fraction).
fn render_num(n: f64) -> String {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 9e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

fn requests_json(sess: &pokesim::bridge::BridgeSession) -> String {
    let one = |side: usize| -> String {
        match sess.active_request_json(side) {
            Some(line) => line.strip_prefix("|request|").unwrap_or(line).to_string(),
            None => "null".to_string(),
        }
    };
    format!("{{\"p1\":{},\"p2\":{}}}", one(0), one(1))
}

fn requests_json_engine(engine: &pokesim::engine::Engine) -> String {
    let one = |side: usize| -> String {
        match engine.request_json(side, dex()) {
            Some(line) => line.strip_prefix("|request|").unwrap_or(&line).to_string(),
            None => "null".to_string(),
        }
    };
    format!("{{\"p1\":{},\"p2\":{}}}", one(0), one(1))
}

fn string_array(items: &[String]) -> String {
    let parts: Vec<String> = items.iter().map(|s| json_quote(s)).collect();
    format!("[{}]", parts.join(","))
}
