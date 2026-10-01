//! PLAY OUT TO THE END — Lane S's ground truth (program §2 M5, Lane S row ④): from a decision of a
//! banked core INPUT LOG, branch each legal action of one side × a set of dice seeds, and continue
//! every branch to the end with a CALLER-SUPPLIED policy.
//!
//! THE LOOP (the host drives it; Rust never runs a network — the T2 shape):
//!
//! ```text
//! open(request)                      -> the root + the branch table
//! k = step([], rows, masks, who)     -> the first pending decisions (k rows written)
//! while k:
//!     actions = policy(rows[:k], masks[:k], who[:k])   # greedy argmax, T2 `score()` later
//!     k = step(actions, rows, masks, who)              # feed, advance, next pending
//! results()                          -> per branch: winner, forfeit, truncated, turn, …
//! ```
//!
//! `who[i] = 2 * branch + side`. Every pending decision of every live branch is in ONE batch, so a
//! policy forward is batched across the whole root. The ROOT decision of the searched side is
//! never pending: it is the branch's forced first action; the other side's root decision (a
//! simultaneous move) IS pending, and — every sibling seeing the same row — a deterministic policy
//! answers it identically across siblings.
//!
//! COMMON RANDOM NUMBERS: a branch's engine is reseeded AT the branch point with its seed, BEFORE
//! any feed, so every sibling that shares a seed shares the dice stream from there (a seed `null`
//! keeps the battle's own stream: the recorded continuation, the gate's anchor). With a
//! deterministic policy a playout is a FUNCTION of (log, at, actions, seeds, policy): the gate
//! (`tests/search_game_test.rs`, `successors_integration_test.py`) holds a rerun byte-identical.
//!
//! THE END: the battle ends (a winner, or a tie — `winner` null); or the STALL FORFEIT (`stall`,
//! training's rule — `crate::episode`: at a decision of `stall.side` whose turn is `>= turn_limit`
//! that side FORCELOSEs before anything is fed); or `max_turns` (< 1000: the port panics at 1,000
//! committed turns, program §0) — the branch stops TRUNCATED, no winner.

use pokesim::encoder::OBS_DIM;
use pokesim::json::Json;
use pokesim::prng::Prng;
use pokesim::search::json_quote;
use pokesim::trackers::clock::ClockConfig;

use super::game::{Game, Log};
use crate::core::columns::ACT;

/// The port panics at 1,000 committed turns (program §0); `max_turns` must stay below it.
pub const MAX_TURNS_CEILING: u32 = 999;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct End {
    pub winner: Option<usize>,
    pub forfeit: bool,
    pub truncated: bool,
    pub turn: u32,
}

pub struct Branch {
    pub game: Game,
    pub action: i32,
    pub seed: usize,
    forced: bool,
    pub decisions: [u32; 2],
    pub end: Option<End>,
}

#[derive(Clone, Copy, Debug)]
struct Stall {
    turn_limit: u32,
    side: usize,
}

/// The playout table of one handle (one root at a time; `open` replaces it).
pub struct Playouts {
    clock: ClockConfig,
    max_branches: usize,
    side: usize,
    stall: Option<Stall>,
    max_turns: u32,
    at: usize,
    seeds: Vec<Option<String>>,
    branches: Vec<Branch>,
    /// The pending list the caller holds (`(branch, side)`), answered by the next `step`.
    pending: Vec<(usize, usize)>,
    /// Each pending decision's frame index `n` (the env core's `dec_n`), parallel to `pending`.
    pending_n: Vec<u32>,
    opened: bool,
    /// Decisions answered by a policy, over the handle's life.
    pub answered: u64,
    /// Branches finished, over the handle's life.
    pub finished: u64,
}

fn side_arg(v: Option<&str>, key: &str) -> Result<usize, String> {
    match v {
        Some("p1") => Ok(0),
        Some("p2") => Ok(1),
        other => Err(format!("playout: `{key}` must be \"p1\" or \"p2\", got {other:?}")),
    }
}

fn uint(v: &Json, key: &str) -> Result<u64, String> {
    let x = v.get(key).and_then(Json::as_f64).ok_or_else(|| format!("playout: `{key}` must be a non-negative integer"))?;
    if x < 0.0 || x.fract() != 0.0 || x > 4.0e9 {
        return Err(format!("playout: `{key}` must be a non-negative integer, got {x}"));
    }
    Ok(x as u64)
}

fn strings(v: Option<&Json>, key: &str) -> Result<Vec<String>, String> {
    v.and_then(Json::as_array)
        .ok_or_else(|| format!("playout: `{key}` must be an array of strings"))?
        .iter()
        .map(|s| s.as_str().map(str::to_string).ok_or_else(|| format!("playout: `{key}` holds a non-string")))
        .collect()
}

/// A core input log as JSON: `{"format_id", "seed", "names": [2], "teams": [2], "cmds": [...]}`.
pub fn log_from_json(v: &Json) -> Result<Log, String> {
    let s = |k: &str| v.str_at(k).map(str::to_string).ok_or_else(|| format!("playout: `log.{k}` must be a string"));
    let two = |k: &str| -> Result<[String; 2], String> {
        let a = strings(v.get(k), &format!("log.{k}"))?;
        <[String; 2]>::try_from(a).map_err(|_| format!("playout: `log.{k}` must hold exactly two strings"))
    };
    Ok(Log { format_id: s("format_id")?, seed: s("seed")?, names: two("names")?, teams: two("teams")?, cmds: strings(v.get("cmds"), "log.cmds")? })
}

const OPEN_KEYS: [&str; 8] = ["log", "at", "side", "actions", "seeds", "stall", "max_turns", "keep_cmds"];

impl Playouts {
    pub fn new(clock: ClockConfig, max_branches: usize) -> Playouts {
        Playouts {
            clock,
            max_branches,
            side: 0,
            stall: None,
            max_turns: MAX_TURNS_CEILING,
            at: 0,
            seeds: Vec::new(),
            branches: Vec::with_capacity(max_branches),
            pending: Vec::with_capacity(2 * max_branches),
            pending_n: Vec::with_capacity(2 * max_branches),
            opened: false,
            answered: 0,
            finished: 0,
        }
    }

    /// OPEN a root: replay `log` to its `at`-th command, then one branch per (action, seed) — the
    /// searched side's legal actions when `actions` is null. Returns the root as JSON.
    pub fn open(&mut self, req: &Json) -> Result<String, String> {
        let obj = req.as_object().ok_or("playout: the request must be an object")?;
        for k in obj.keys() {
            if !OPEN_KEYS.contains(&k.as_str()) {
                return Err(format!("playout: unknown key {k:?}"));
            }
        }
        for k in ["log", "at", "side", "actions", "seeds", "stall", "max_turns"] {
            if !obj.contains_key(k) {
                return Err(format!("playout: missing key {k:?} (every key but keep_cmds is required; null where allowed)"));
            }
        }
        let log = log_from_json(req.get("log").expect("checked"))?;
        let at = uint(req, "at")? as usize;
        let side = side_arg(req.str_at("side"), "side")?;
        let max_turns = uint(req, "max_turns")? as u32;
        if max_turns == 0 || max_turns > MAX_TURNS_CEILING {
            return Err(format!("playout: `max_turns` must be in 1..={MAX_TURNS_CEILING} (the port panics at 1,000 turns), got {max_turns}"));
        }
        let stall = match req.get("stall") {
            Some(Json::Null) => None,
            Some(s) => Some(Stall { turn_limit: uint(s, "turn_limit")? as u32, side: side_arg(s.str_at("side"), "stall.side")? }),
            None => unreachable!("checked"),
        };
        let seeds: Vec<Option<String>> = req
            .get("seeds")
            .and_then(Json::as_array)
            .ok_or("playout: `seeds` must be an array (a string per dice draw; null = the battle's own stream)")?
            .iter()
            .map(|s| match s {
                Json::Null => Ok(None),
                Json::Str(t) => Prng::validate_seed(t).map(|_| Some(t.clone())).map_err(|e| format!("playout: seed {t:?}: {e}")),
                _ => Err("playout: a seed must be a string or null".to_string()),
            })
            .collect::<Result<_, _>>()?;
        if seeds.is_empty() {
            return Err("playout: `seeds` is empty".into());
        }
        // Replay FIRST (a refused log leaves the previous table untouched).
        let root = Game::replay(&log, at, self.clock)?;
        let open = root.open(side).ok_or_else(|| format!("playout: p{} has no open decision after {at} commands", side + 1))?;
        let legal: Vec<i32> = open.tokens.iter().map(|(i, _)| *i as i32).collect();
        let actions: Vec<i32> = match req.get("actions") {
            Some(Json::Null) => legal.clone(),
            Some(Json::Arr(a)) => a
                .iter()
                .map(|x| {
                    let v = x.as_f64().filter(|v| v.fract() == 0.0).ok_or("playout: an action must be an integer")? as i32;
                    if legal.contains(&v) {
                        Ok(v)
                    } else {
                        Err(format!("playout: action {v} is not legal for p{} at the root (legal: {legal:?})", side + 1))
                    }
                })
                .collect::<Result<_, String>>()?,
            _ => return Err("playout: `actions` must be an array of action indices or null".into()),
        };
        let n = actions.len() * seeds.len();
        if n == 0 {
            return Err("playout: no branch to play (no action)".into());
        }
        if n > self.max_branches {
            return Err(format!(
                "playout: {} actions x {} seeds = {n} branches exceed the declared max_branches = {} (never grown)",
                actions.len(),
                seeds.len(),
                self.max_branches
            ));
        }
        let keep_cmds = req.get("keep_cmds").and_then(Json::as_bool).unwrap_or(false);
        let turn = root.turn(side);
        let other_open = root.open(1 - side).is_some();
        let tokens = pokesim::present::tokens_json(&open.tokens);
        // The root decision's frame index (the env core's `dec_n`): a caller keying draws on the
        // parent's decisions checks it (`designs/training/forks.md` §14.3).
        let root_n = open.n;
        self.branches.clear();
        self.pending.clear();
        self.pending_n.clear();
        for &a in &actions {
            for (r, sd) in seeds.iter().enumerate() {
                let mut game = root.branch(sd.as_deref());
                if !keep_cmds {
                    game.cmds = Vec::new();
                }
                self.branches.push(Branch { game, action: a, seed: r, forced: true, decisions: [0, 0], end: None });
            }
        }
        self.side = side;
        self.stall = stall;
        self.max_turns = max_turns;
        self.at = at;
        self.seeds = seeds;
        self.opened = true;
        Ok(format!(
            "{{\"side\":\"p{}\",\"turn\":{turn},\"n\":{root_n},\"other_open\":{other_open},\"tokens\":{tokens},\"actions\":{actions:?},\"n_seeds\":{},\"branches\":{n}}}",
            side + 1,
            self.seeds.len()
        ))
    }

    /// Branches not yet ended.
    pub fn live(&self) -> usize {
        self.branches.iter().filter(|b| b.end.is_none()).count()
    }

    /// FEED `actions` (one per entry of the pending list the last step handed out, in order), advance
    /// every live branch, and write the NEXT pending list into `rows` / `masks` / `who`. Returns its
    /// length; 0 = every branch has ended.
    pub fn step(&mut self, actions: &[i32], rows: &mut [f32], masks: &mut [u8], who: &mut [u32]) -> Result<usize, String> {
        if !self.opened {
            return Err("playout: step before open".into());
        }
        if actions.len() != self.pending.len() {
            return Err(format!("playout: step got {} actions for {} pending decisions", actions.len(), self.pending.len()));
        }
        let mut chosen: Vec<[Option<i32>; 2]> = vec![[None, None]; self.branches.len()];
        for (&(b, s), &a) in self.pending.iter().zip(actions) {
            chosen[b][s] = Some(a);
        }
        let answered_before = !self.pending.is_empty();
        self.answered += self.pending.len() as u64;
        self.pending.clear();
        self.pending_n.clear();
        loop {
            // 1. feed every branch that holds its answers (or only its forced first action).
            for (b, br) in self.branches.iter_mut().enumerate() {
                if br.end.is_some() {
                    continue;
                }
                let mut act = chosen[b];
                if br.forced {
                    act[self.side] = Some(br.action);
                }
                let need: Vec<usize> = (0..2).filter(|&s| br.game.open(s).is_some()).collect();
                if need.is_empty() || need.iter().any(|&s| act[s].is_none()) {
                    continue; // not answered yet (or nothing open: settled below)
                }
                let to_feed = [br.game.open(0).map(|o| o.n), br.game.open(1).map(|o| o.n)];
                for s in 0..2 {
                    if br.game.is_ended() {
                        break;
                    }
                    // A side whose decision an earlier feed of this step closed or replaced is skipped
                    // (`crate::episode`'s STEP rule).
                    if to_feed[s].is_some() && br.game.open(s).map(|o| o.n) == to_feed[s] {
                        br.game.feed(s, act[s].expect("answered")).map_err(|e| format!("branch {b}: {e}"))?;
                        if !(br.forced && s == self.side) {
                            br.decisions[s] += 1;
                        }
                    }
                }
                br.forced = false;
            }
            chosen.iter_mut().for_each(|c| *c = [None, None]);
            // 2. settle ends, then collect the next pending list.
            let mut forced_only = false;
            for (b, br) in self.branches.iter_mut().enumerate() {
                if br.end.is_some() {
                    continue;
                }
                if !br.game.is_ended() {
                    if let Some(st) = self.stall {
                        if br.game.open(st.side).is_some() && br.game.turn(st.side) >= st.turn_limit {
                            br.game.forfeit(st.side).map_err(|e| format!("branch {b}: {e}"))?;
                        }
                    }
                }
                let turn = br.game.turn(0).max(br.game.turn(1));
                if br.game.is_ended() {
                    let forfeit = br.game.cmds.last().is_some_and(|c| c.starts_with("FORCELOSE"));
                    br.end = Some(End { winner: br.game.winner(), forfeit, truncated: false, turn });
                    self.finished += 1;
                    continue;
                }
                if turn >= self.max_turns {
                    br.end = Some(End { winner: None, forfeit: false, truncated: true, turn });
                    self.finished += 1;
                    continue;
                }
                let open: Vec<usize> = (0..2).filter(|&s| br.game.open(s).is_some()).collect();
                if open.is_empty() {
                    return Err(format!("branch {b}: stuck — no decision open and the battle is not over"));
                }
                let mut any = false;
                for s in open {
                    if br.forced && s == self.side {
                        continue;
                    }
                    self.pending.push((b, s));
                    any = true;
                }
                forced_only |= !any; // only the forced root action is open: feed it without a policy
            }
            if !self.pending.is_empty() || !forced_only {
                break;
            }
            let _ = answered_before;
        }
        let k = self.pending.len();
        if rows.len() < k * OBS_DIM || masks.len() < k * ACT || who.len() < k {
            return Err(format!("playout: the buffers hold fewer than the {k} pending rows (allocate 2 x branches)"));
        }
        for (i, &(b, s)) in self.pending.iter().enumerate() {
            let row: &mut [f32; OBS_DIM] = (&mut rows[i * OBS_DIM..(i + 1) * OBS_DIM]).try_into().expect("row");
            let m: &mut [u8; ACT] = (&mut masks[i * ACT..(i + 1) * ACT]).try_into().expect("mask");
            self.branches[b].game.encode(s, row, m).map_err(|e| format!("branch {b}: {e}"))?;
            who[i] = (2 * b + s) as u32;
            self.pending_n.push(self.branches[b].game.open(s).map_or(0, |o| o.n));
        }
        Ok(k)
    }

    /// The frame index `n` of each decision the last `step` handed out (the env core's `dec_n`), in
    /// its order — what a caller keys a decision's draw on (`designs/training/forks.md` §14.3).
    pub fn pending_n(&self, out: &mut [u32]) -> Result<usize, String> {
        let k = self.pending_n.len();
        if out.len() < k {
            return Err(format!("playout: pending_n's buffer holds {} slots, {k} decisions are pending", out.len()));
        }
        out[..k].copy_from_slice(&self.pending_n);
        Ok(k)
    }

    /// Every branch, as JSON (after the last step, or mid-play: `end` null while live).
    pub fn results(&self) -> String {
        let mut out = String::from("[");
        for (b, br) in self.branches.iter().enumerate() {
            if b > 0 {
                out.push(',');
            }
            let end = match br.end {
                None => "null".to_string(),
                Some(e) => format!(
                    "{{\"winner\":{},\"forfeit\":{},\"truncated\":{},\"turn\":{}}}",
                    e.winner.map_or("null".to_string(), |w| w.to_string()),
                    e.forfeit,
                    e.truncated,
                    e.turn
                ),
            };
            let cmds: Vec<String> = br.game.cmds.iter().map(|c| json_quote(c)).collect();
            out.push_str(&format!(
                "{{\"action\":{},\"seed\":{},\"reseed\":{},\"end\":{end},\"decisions\":[{},{}],\"cmds\":[{}]}}",
                br.action,
                br.seed,
                self.seeds[br.seed].as_deref().map_or("null".to_string(), json_quote),
                br.decisions[0],
                br.decisions[1],
                cmds.join(",")
            ));
        }
        out.push(']');
        format!("{{\"side\":\"p{}\",\"at\":{},\"branches\":{out}}}", self.side + 1, self.at)
    }
}
