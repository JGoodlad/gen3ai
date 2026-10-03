//! SEARCH ON `successors()`, IN-PROCESS (M5 Lane I, `designs/endstate/program_rust_core.md` §2 M5,
//! §6). Two consumers, two shapes, one handle ([`SearchHandle`], built from a [`SearchSpec`]):
//!
//! * [`tree`] — the SEARCH TREE: a tree of step-built `BattleVersion`s opened from a
//!   reconstruction record at a turn, grown one ply per batch of ARMS (`open_root` /
//!   `expand_many`, `search_driver`'s core road — `gen3_core_search_v1` + `rows`), each successor
//!   ENCODED straight into the caller's buffer. It replaces `utils/bridge/search_session.py`'s JSON
//!   process for the in-process road (`src/utils/rust_env/successors.py`); gate: every arm, row,
//!   mask, token, request, outcome and node id BYTE-EQUAL to the `search_driver` binary's to depth 3.
//! * [`playout`] — PLAY OUT TO THE END (Lane S's ground truth): from a decision of a banked core
//!   INPUT LOG, branch each legal action of one side × a set of dice seeds (COMMON RANDOM NUMBERS:
//!   every sibling that shares a seed shares the dice stream from the branch point), then continue
//!   every branch to the end with a caller-supplied policy — the host reads the pending rows, runs
//!   its scorer (a greedy forward, T2's `score()` later) and feeds the actions back. Rows come from
//!   the TRAINING observation path ([`game`]: per-side protocol text → the parse fold → encode,
//!   program §6c), so a playout's policy sees exactly the rows training sees.
//!
//! The FFI surface is `ffi_imp` (forwarded from `crate::ffi`'s generated wrappers; the rows of
//! `ffi.FUNCTIONS` that name `rust_env_search_*` / `rust_env_playout_*` are Lane I's). No battle
//! logic is new here: arms resolve through the port's own kernels (`pokesim::search`), successors
//! are the port's `BattleVersion`s, and a playout's battle is the env core's (`sim_bridge`'s
//! alignment rules, re-stated in [`game`] and gated against the core's own rows).
//!
//! THE DECLARED LIFECYCLE (offline tooling, but honoured): the spec declares `max_nodes` (a tree's
//! node table) and `max_branches` (a playout's branch table); both are reserved at construction and
//! a request that would exceed one is REFUSED by name, never grown.

pub mod ffi_imp;
pub mod game;
pub mod playout;
pub mod tree;

use pokesim::json::Json;
use pokesim::trackers::clock::ClockConfig;

/// The handle's STARTUP declaration — one JSON object, every key required, an unknown key refused
/// (the env spec's rule):
///
/// ```text
/// {"max_nodes": <int >= 1>, "max_branches": <int >= 1>}
/// ```
///
/// (A `clock` key, the progress clock's two flags, was deleted with the trainer's
/// `--progress-decision-tense` / `--progress-switch-freeze`; it is refused as unknown. Every search
/// builds [`ClockConfig::default`], as `search_driver` does.)
#[derive(Clone, Debug, PartialEq)]
pub struct SearchSpec {
    pub max_nodes: usize,
    pub max_branches: usize,
}

pub const SEARCH_SPEC_KEYS: [&str; 2] = ["max_nodes", "max_branches"];

fn count(v: &Json, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Json::as_f64).ok_or_else(|| format!("search spec: `{key}` must be a positive integer"))?;
    if x < 1.0 || x.fract() != 0.0 || x > 1.0e8 {
        return Err(format!("search spec: `{key}` must be a positive integer (<= 1e8), got {x}"));
    }
    Ok(x as usize)
}

impl SearchSpec {
    pub fn from_json(text: &str) -> Result<SearchSpec, String> {
        let v = Json::parse(text).map_err(|e| format!("search spec: not JSON: {e}"))?;
        let obj = v.as_object().ok_or("search spec: must be a JSON object")?;
        for k in obj.keys() {
            if !SEARCH_SPEC_KEYS.contains(&k.as_str()) {
                return Err(format!("search spec: unknown key {k:?}"));
            }
        }
        for k in SEARCH_SPEC_KEYS {
            if !obj.contains_key(k) {
                return Err(format!("search spec: missing key {k:?} (every key is required)"));
            }
        }
        Ok(SearchSpec {
            max_nodes: count(&v, "max_nodes")?,
            max_branches: count(&v, "max_branches")?,
        })
    }
}

/// One in-process search handle: a [`tree::Tree`] and a [`playout::Playouts`] over ONE dex.
pub struct SearchCore {
    pub spec: SearchSpec,
    pub tree: tree::Tree,
    pub playouts: playout::Playouts,
}

impl SearchCore {
    /// STARTUP: the dex, the node table and the branch table are acquired here.
    pub fn new(spec: SearchSpec) -> SearchCore {
        let _ = dex();
        SearchCore {
            tree: tree::Tree::new(ClockConfig::default(), spec.max_nodes),
            playouts: playout::Playouts::new(ClockConfig::default(), spec.max_branches),
            spec,
        }
    }
}

/// The opaque FFI handle (`rust_env_search_new`): one [`SearchCore`] behind a lock + a poison flag,
/// exactly `crate::ffi::Handle`'s contract (single caller; a panic inside poisons it).
pub struct SearchHandle {
    pub(crate) core: std::sync::Mutex<SearchCore>,
    pub(crate) poisoned: std::sync::atomic::AtomicBool,
}

/// The one dex every handle shares (a `Dex` owns ~16 MB; built once per process, on first use).
pub(crate) fn dex() -> &'static pokesim::dex::Dex {
    static D: std::sync::OnceLock<pokesim::dex::Dex> = std::sync::OnceLock::new();
    D.get_or_init(|| pokesim::dex::Dex::for_gen(3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spec_is_strict() {
        let ok = r#"{"max_nodes":10,"max_branches":4}"#;
        let s = SearchSpec::from_json(ok).unwrap();
        assert_eq!((s.max_nodes, s.max_branches), (10, 4));
        assert!(SearchSpec::from_json(r#"{"max_nodes":10}"#).unwrap_err().contains("max_branches"));
        // the deleted `clock` key is refused as unknown
        assert!(SearchSpec::from_json(r#"{"clock":{"decision_tense":false,"switch_freeze":false},"max_nodes":10,"max_branches":4}"#)
            .unwrap_err()
            .contains("unknown key"));
        assert!(SearchSpec::from_json(&ok.replace("\"max_nodes\"", "\"max_node\"")).unwrap_err().contains("unknown key"));
        assert!(SearchSpec::from_json(&ok.replace(":10", ":0")).unwrap_err().contains("positive"));
        assert!(SearchSpec::from_json(&ok.replace("4}", "4,\"x\":1}")).unwrap_err().contains("unknown key"));
    }
}
