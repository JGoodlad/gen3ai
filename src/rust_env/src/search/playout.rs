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
//! THE ROOT (`at`): a command INDEX (replay the log's first `at` commands), or a DIVERGENCE TURN
//! `{"turn": T, "other": "recorded" | "policy"}` — replay to the START of turn `T` (both sides at a
//! move request, none of turn `T`'s choices fed: the search tree's `build_to_turn`), then, with
//! `"recorded"`, feed the OTHER side's recorded turn-`T` choice (its first command at or after that
//! point, the tree's `recorded_turn_choices`): the counterfactual's rule that the opponent could not
//! have reacted to our change on the same turn (`gen3_cf_core_playout_v1`, the prober's
//! `replay-counterfactual`). `"policy"` leaves the other side's root decision open (pending). The
//! resolved command index is reported as `at` either way.
//!
//! COMMON RANDOM NUMBERS: a branch's engine is reseeded AT the branch point with its seed, BEFORE
//! any feed, so every sibling that shares a seed shares the dice stream from there (a seed `null`
//! keeps the battle's own stream: the recorded continuation, the gate's anchor). With a
//! deterministic policy a playout is a FUNCTION of (log, at, actions, seeds, policy): the gate
//! (`tests/search_game_test.rs`, `successors_integration_test.py`) holds a rerun byte-identical.
//!
//! AN IN-CORE BOT (`bot`: `{"side", "name", "seed"}`): one side played by a Lane-F scripted bot
//! (`crate::bots`) INSIDE the core, exactly as `crate::opponents`' `Route::Bot` answers p2 — asked
//! only at a REAL decision (never a phantom poll, F-LF-2; never after the other side's stall forfeit
//! closed the battle, F-LF-5), never exposed as pending. Branch `b`'s bot is env `b`'s bot of a
//! route declared with `seed`: stream `k` (choice 0, protect 1) is `random.Random(stream_seed(seed,
//! b, k))` (`crate::opponents::stream_seed`; Python twin `rust_env_opponents.bot_stream_seed`). The
//! bot side must not be the searched side. A bot refusal fails the playout (a CALLER error naming
//! the branch), as the Python bot would raise.
//!
//! THE END: the battle ends (a winner, or a tie — `winner` null); or the STALL FORFEIT (`stall`,
//! training's rule — `crate::episode`: at a decision of a stall side whose turn is `>= turn_limit`
//! that side FORCELOSEs before anything is fed; `"side"` names one side, `"sides"` several — when
//! two are open at the limit p1 forfeits FIRST, the order the bridge processes two stalling poke-env
//! players in); or `max_turns` (< 1000: the port panics at 1,000 committed turns, program §0) — the
//! branch stops TRUNCATED, no winner.
//!
//! TEXT (`text`: `"p1"` / `"p2"`): the results carry that side's protocol lines — the root's
//! `prefix_text` once, each branch's lines after the branch point as its `text` (a narrated
//! play-by-play, the prober's `--narrate`).

use pokesim::encoder::OBS_DIM;
use pokesim::json::Json;
use pokesim::prng::Prng;
use pokesim::search::json_quote;
use pokesim::trackers::clock::ClockConfig;

use super::game::{Game, Log};
use crate::bots::{Bot, Kind};
use crate::core::columns::ACT;
use crate::opponents::stream_seed;

/// The port panics at 1,000 committed turns (program §0); `max_turns` must stay below it.
pub const MAX_TURNS_CEILING: u32 = 999;

/// Seeds must survive a JSON number (f64) exactly (`crate::opponents`' rule).
const MAX_SEED: f64 = 9_007_199_254_740_991.0; // 2^53 - 1

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
    /// The in-core bot of this branch (when the request declared one).
    bot: Option<Bot>,
    /// The bot's answer to its OPEN decision `(ordinal n, token)`, held until the branch is fed.
    bot_answer: Option<(u32, String)>,
}

#[derive(Clone, Debug)]
struct Stall {
    turn_limit: u32,
    /// The stall sides, ascending (p1 forfeits first when both are open at the limit).
    sides: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct BotSpec {
    side: usize,
    kind: Kind,
    seed: u64,
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
    bot: Option<BotSpec>,
    /// The side whose protocol lines the results carry (`text`), and the root's lines.
    text: Option<usize>,
    prefix_text: Vec<String>,
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

const OPEN_KEYS: [&str; 10] = ["log", "at", "side", "actions", "seeds", "stall", "max_turns", "keep_cmds", "bot", "text"];

fn stall_from_json(s: &Json) -> Result<Stall, String> {
    let obj = s.as_object().ok_or("playout: `stall` must be an object or null")?;
    for k in obj.keys() {
        if !["turn_limit", "side", "sides"].contains(&k.as_str()) {
            return Err(format!("playout: `stall`: unknown key {k:?}"));
        }
    }
    let turn_limit = uint(s, "turn_limit")? as u32;
    let mut sides: Vec<usize> = match (s.get("side"), s.get("sides")) {
        (Some(v), None) => vec![side_arg(v.as_str(), "stall.side")?],
        (None, Some(v)) => {
            let a = v.as_array().ok_or("playout: `stall.sides` must be an array of \"p1\" / \"p2\"")?;
            a.iter().map(|x| side_arg(x.as_str(), "stall.sides")).collect::<Result<_, _>>()?
        }
        _ => return Err("playout: `stall` names its side(s) with exactly one of `side` / `sides`".into()),
    };
    sides.sort_unstable();
    sides.dedup();
    if sides.is_empty() {
        return Err("playout: `stall.sides` is empty (pass `stall: null` for no stall forfeit)".into());
    }
    Ok(Stall { turn_limit, sides })
}

fn bot_from_json(v: &Json, searched: usize) -> Result<BotSpec, String> {
    let obj = v.as_object().ok_or("playout: `bot` must be an object {side, name, seed} or null")?;
    for k in obj.keys() {
        if !["side", "name", "seed"].contains(&k.as_str()) {
            return Err(format!("playout: `bot`: unknown key {k:?}"));
        }
    }
    let side = side_arg(v.str_at("side"), "bot.side")?;
    if side == searched {
        return Err(format!("playout: the bot plays p{}, the SEARCHED side (a bot side must be the other side)", side + 1));
    }
    let name = v.str_at("name").ok_or("playout: `bot.name` must be a bot name")?;
    let kind = Kind::from_name(name).ok_or_else(|| format!("playout: bot {name:?} is not one this core plays (Lane F's `bots::Kind` names)"))?;
    let seed = v.get("seed").and_then(Json::as_f64).ok_or("playout: `bot.seed` must be an integer")?;
    if seed < 0.0 || seed.fract() != 0.0 || seed > MAX_SEED {
        return Err(format!("playout: `bot.seed` must be a non-negative integer <= 2^53 - 1, got {seed}"));
    }
    Ok(BotSpec { side, kind, seed: seed as u64 })
}

/// The root of a request's `at`: `(game, resolved command index, the other side's recorded token fed)`.
fn root_of(log: &Log, at: &Json, side: usize, clock: ClockConfig) -> Result<(Game, usize, Option<String>), String> {
    if let Some(obj) = at.as_object() {
        for k in obj.keys() {
            if !["turn", "other"].contains(&k.as_str()) {
                return Err(format!("playout: `at`: unknown key {k:?}"));
            }
        }
        let turn = uint(at, "turn")? as u32;
        let recorded = match at.str_at("other") {
            Some("recorded") => true,
            Some("policy") => false,
            other => return Err(format!("playout: `at.other` must be \"recorded\" or \"policy\", got {other:?}")),
        };
        let (mut g, rest) = Game::replay_to_turn(log, turn, clock)?;
        if !recorded {
            return Ok((g, rest, None));
        }
        let other = 1 - side;
        let prefix = format!("CHOOSE p{} ", other + 1);
        // The other side's FIRST recorded command from turn `turn`'s start; a forfeit ends the scan.
        let mut found = None;
        for (k, c) in log.cmds.iter().enumerate().skip(rest) {
            if c.starts_with("FORCELOSE") {
                break;
            }
            if c.starts_with(&prefix) {
                found = Some((k, c.clone()));
                break;
            }
        }
        let (k, c) = found.ok_or_else(|| format!("playout: p{} has no recorded choice at turn {turn} (the log ends or forfeits first)", other + 1))?;
        g.feed_logged(k, &c)?;
        let tok = c[prefix.len()..].to_string();
        // The resolved index: every command before turn `turn`'s start, plus the other side's choice.
        return Ok((g, rest + 1, Some(tok)));
    }
    let x = at.as_f64().ok_or("playout: `at` must be a command index or {\"turn\", \"other\"}")?;
    if x < 0.0 || x.fract() != 0.0 || x > 4.0e9 {
        return Err(format!("playout: `at` must be a non-negative integer, got {x}"));
    }
    let at = x as usize;
    Ok((Game::replay(log, at, clock)?, at, None))
}

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
            bot: None,
            text: None,
            prefix_text: Vec::new(),
            pending: Vec::with_capacity(2 * max_branches),
            pending_n: Vec::with_capacity(2 * max_branches),
            opened: false,
            answered: 0,
            finished: 0,
        }
    }

    /// OPEN a root: replay `log` to its `at` (a command index or a divergence turn — the module
    /// docs), then one branch per (action, seed) — the searched side's legal actions when `actions`
    /// is null. Returns the root as JSON.
    pub fn open(&mut self, req: &Json) -> Result<String, String> {
        let obj = req.as_object().ok_or("playout: the request must be an object")?;
        for k in obj.keys() {
            if !OPEN_KEYS.contains(&k.as_str()) {
                return Err(format!("playout: unknown key {k:?}"));
            }
        }
        for k in ["log", "at", "side", "actions", "seeds", "stall", "max_turns"] {
            if !obj.contains_key(k) {
                return Err(format!("playout: missing key {k:?} (every key but keep_cmds / bot / text is required; null where allowed)"));
            }
        }
        let log = log_from_json(req.get("log").expect("checked"))?;
        let side = side_arg(req.str_at("side"), "side")?;
        let max_turns = uint(req, "max_turns")? as u32;
        if max_turns == 0 || max_turns > MAX_TURNS_CEILING {
            return Err(format!("playout: `max_turns` must be in 1..={MAX_TURNS_CEILING} (the port panics at 1,000 turns), got {max_turns}"));
        }
        let stall = match req.get("stall") {
            Some(Json::Null) => None,
            Some(s) => Some(stall_from_json(s)?),
            None => unreachable!("checked"),
        };
        let bot = match req.get("bot") {
            None | Some(Json::Null) => None,
            Some(b) => Some(bot_from_json(b, side)?),
        };
        let text = match req.get("text") {
            None | Some(Json::Null) => None,
            Some(t) => Some(side_arg(t.as_str(), "text")?),
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
        let (root, at, other_recorded) = root_of(&log, req.get("at").expect("checked"), side, self.clock)?;
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
                let b = self.branches.len();
                let bot = bot.map(|s| Bot::new(s.kind, stream_seed(s.seed, b, 0), stream_seed(s.seed, b, 1)));
                self.branches.push(Branch { game, action: a, seed: r, forced: true, decisions: [0, 0], end: None, bot, bot_answer: None });
            }
        }
        self.prefix_text = text.map_or_else(Vec::new, |s| root.side_lines(s));
        self.side = side;
        self.stall = stall;
        self.max_turns = max_turns;
        self.at = at;
        self.seeds = seeds;
        self.bot = bot;
        self.text = text;
        self.opened = true;
        Ok(format!(
            "{{\"side\":\"p{}\",\"at\":{at},\"turn\":{turn},\"n\":{root_n},\"other_open\":{other_open},\"other_recorded\":{},\"tokens\":{tokens},\"actions\":{actions:?},\"n_seeds\":{},\"branches\":{n}}}",
            side + 1,
            other_recorded.as_deref().map_or("null".to_string(), json_quote),
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
        let bot_side = self.bot.map(|s| s.side);
        self.answered += self.pending.len() as u64;
        self.pending.clear();
        self.pending_n.clear();
        loop {
            // 1. feed every branch that holds its answers (or only its forced first action / its bot's).
            for (b, br) in self.branches.iter_mut().enumerate() {
                if br.end.is_some() {
                    continue;
                }
                let mut act = chosen[b];
                if br.forced {
                    act[self.side] = Some(br.action);
                }
                let need: Vec<usize> = (0..2).filter(|&s| br.game.open(s).is_some()).collect();
                let answered = |s: usize| {
                    if Some(s) == bot_side {
                        let n = br.game.open(s).map(|o| o.n);
                        br.bot_answer.as_ref().is_some_and(|(an, _)| Some(*an) == n)
                    } else {
                        act[s].is_some()
                    }
                };
                if need.is_empty() || need.iter().any(|&s| !answered(s)) {
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
                        if Some(s) == bot_side {
                            let (_, tok) = br.bot_answer.take().expect("answered");
                            br.game.feed_token(s, &tok).map_err(|e| format!("branch {b}: {e}"))?;
                        } else {
                            br.game.feed(s, act[s].expect("answered")).map_err(|e| format!("branch {b}: {e}"))?;
                        }
                        if !(br.forced && s == self.side) {
                            br.decisions[s] += 1;
                        }
                    }
                }
                br.forced = false;
            }
            chosen.iter_mut().for_each(|c| *c = [None, None]);
            // 2. settle ends, then collect the next pending list (the bot answers its own).
            let mut self_answered = false;
            for (b, br) in self.branches.iter_mut().enumerate() {
                if br.end.is_some() {
                    continue;
                }
                if !br.game.is_ended() {
                    if let Some(st) = &self.stall {
                        for &s in &st.sides {
                            if br.game.open(s).is_some() && br.game.turn(s) >= st.turn_limit {
                                br.game.forfeit(s).map_err(|e| format!("branch {b}: {e}"))?;
                                break;
                            }
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
                    if Some(s) == bot_side {
                        // The bot decides at a REAL decision, once per decision ordinal.
                        let o = br.game.open(s).expect("open");
                        if br.bot_answer.as_ref().is_none_or(|(n, _)| *n != o.n) {
                            let bot = br.bot.as_mut().expect("a bot branch");
                            let d = bot
                                .decide(br.game.reading(s), &o.tokens)
                                .map_err(|e| format!("branch {b}: the {} bot refused: {e}", bot.kind.name()))?;
                            br.bot_answer = Some((o.n, d.token));
                        }
                        continue;
                    }
                    self.pending.push((b, s));
                    any = true;
                }
                self_answered |= !any; // only forced / bot decisions are open: feed them without a policy
            }
            if !self.pending.is_empty() || !self_answered {
                break;
            }
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
        let quote_all = |v: &[String]| v.iter().map(|c| json_quote(c)).collect::<Vec<_>>().join(",");
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
            let text = self.text.map_or(String::new(), |s| format!(",\"text\":[{}]", quote_all(&br.game.side_lines(s))));
            out.push_str(&format!(
                "{{\"action\":{},\"seed\":{},\"reseed\":{},\"end\":{end},\"decisions\":[{},{}],\"cmds\":[{}]{text}}}",
                br.action,
                br.seed,
                self.seeds[br.seed].as_deref().map_or("null".to_string(), json_quote),
                br.decisions[0],
                br.decisions[1],
                quote_all(&br.game.cmds)
            ));
        }
        out.push(']');
        let prefix = self.text.map_or(String::new(), |_| format!(",\"prefix_text\":[{}]", quote_all(&self.prefix_text)));
        format!("{{\"side\":\"p{}\",\"at\":{}{prefix},\"branches\":{out}}}", self.side + 1, self.at)
    }
}
