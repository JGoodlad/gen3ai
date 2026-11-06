//! FRESH DICE, MATCHED SEGMENT BY SEGMENT (`gen3_fresh_dice_match_v1`): the replay of the LEAK-HONEST
//! world builder (`r0s4_honest`, `designs/endstate/design_sprime_honest.md` D3, D4, D6; contract:
//! `designs/rust_sim/search_and_replay_drivers.md` "Fresh dice").
//!
//! WHY. Today's builder replays every candidate world on the battle's TRUE seed, so a world is kept
//! only if it is consistent with the truth's dice, and every value drawn before the root (a sleep
//! counter, the root turn's Quick Claw roll) is the truth's in every world. Here a world never sees
//! the truth: it is replayed on FRESH dice, keyed by the searched side's information set I alone, and
//! kept iff every protocol line the searched side (the VIEWER) was shipped is reproduced byte for
//! byte. One fresh seed reproduces a whole prefix with probability ≈ 10⁻²⁸, so the match is per
//! SEGMENT: the lines one write ships the viewer (I's lines split after each `|request|`; segment 0 is
//! the START write).
//!
//! THE ALGORITHM. Segment 0: the battle is built on `try_seed(dice, 0, t)` for `t = 0, 1, …`, the
//! first `t` whose START write equals I's segment 0 kept. Segment s ≥ 1: the paused battle's open
//! requests say who chooses. The opponent's choice is its INTENT for segment s (`InfoSet::intents`,
//! read by the caller from I's lines: the `|move|` / `|switch|` it was seen to make), or, when I shows
//! it made none (asleep, paralysed, KO'd first), a STAND-IN: each of its offered choices in request
//! order (moves, then switches), the first that reproduces the segment kept. The viewer's choice is
//! its next command. The opponent's is fed first, then the battle's PRNG is replaced by
//! `try_seed(dice, s, t)` and the resolving choice fed; the first `t` (and, for a stand-in, the first
//! choice at it) whose new viewer lines equal segment s is kept. The kept seeds are the RESEED
//! SCHEDULE: replayed (`core_events`' START `"reseed"`, `rust_env`'s `Log::reseed`) they rebuild the
//! same battle with no die of the truth drawn.
//!
//! PERSISTED DICE (D4). A counter drawn at an onset (sleep, confusion, Encore, Disable, a trap, a
//! lock, Uproar, a Future Sight's damage) is the world's own draw, so a later segment can be
//! unmatchable on every try. The caller names, per segment, the earlier segments a failure there
//! may RE-ROLL (`InfoSet::backtrack`: the onsets, in I's lines, of a counter still running; with
//! `prev_backtrack`, the previous segment too: the root turn's Quick Claw roll is drawn at the end of
//! the one before). A segment no try matches re-rolls the latest of them (its next try), up to
//! `rerolls` times a world, before the world fails. The OPPONENT's exact HP is such a value too (the
//! viewer sees a percent, so a damage roll's pick inside it can contradict a later Leftovers or
//! residual line): a failure whose first differing line is an HP line of a mon re-rolls that mon's
//! latest earlier hit (`InfoSet::hp_last`) instead.
//!
//! THE DAMAGE-ROLL CONTROL (`rolls`, `crate::rolls::Mode::Seg`): each damage roll's VALUE is picked to
//! print the segment's own HP line — an efficiency device; every line is still compared.
//!
//! DETERMINISTIC: a world's verdict, schedule and lines are a function of (its teams, I, `dice`,
//! `tries`, `rerolls`, `rolls`). Draw-free outside the battle's own PRNG. Offline only.

use std::sync::Arc;

use crate::battle::BattleOptions;
use crate::bridge::{parse_choice, BridgeSession, Cmd, RequestState};
use crate::dex::Dex;
use crate::json::Json;
use crate::prng::Prng;
use crate::rolls::{Mode, RollCtl};

/// A world's work budget: battle feeds across every try, stand-in and re-roll.
pub const MAX_FEEDS: u64 = 250_000;

/// The searched side's INFORMATION SET as the matcher reads it (module docs).
#[derive(Clone, Debug, Default)]
pub struct InfoSet {
    /// The viewer, 0 = p1, 1 = p2.
    pub viewer: usize,
    /// The viewer's lines, one entry per write (I split after each `|request|`); `[0]` is START's.
    pub segments: Vec<Arc<[String]>>,
    /// The viewer's commands' tokens, in order (each answers its next open request).
    pub ours: Vec<String>,
    /// Per segment, the opponent's choice in the update that resolves into it (`None`: a stand-in).
    /// `[0]` is unused.
    pub intents: Vec<Option<String>>,
    /// Per segment, the earlier segments a failure there may re-roll (module docs, PERSISTED DICE).
    pub backtrack: Vec<Vec<usize>>,
    /// Per segment, `(ident, segments)`: the earlier segments in which the mon `ident` (`p2a: Nick`)
    /// took a damage roll, latest first. A non-owner sees a percent, so the mon's EXACT HP is itself a
    /// persisted die: a failure on an HP line of that mon re-rolls those segments (the latest twice,
    /// then the next, …).
    pub hp_last: Vec<Vec<(String, Vec<usize>)>>,
}

/// One world's match settings.
#[derive(Clone, Debug)]
pub struct MatchCfg {
    /// The base of every try's seed ([`try_seed`]): a digest of I and the world's tag.
    pub dice: String,
    pub tries: u32,
    pub rerolls: u32,
    /// Arm the damage-roll control ([`Mode::Seg`]).
    pub rolls: bool,
    /// Also re-roll the previous segment on a failure (a Quick Claw on either team).
    pub prev_backtrack: bool,
    pub max_feeds: u64,
    /// FIXED opponent commands (the leak test's alternative truths: a log's own commands on other dice):
    /// fed in order wherever the opponent chooses (a refused one is followed by the next), in place of
    /// the intents and stand-ins. `None`: rebuilt from I (the builder).
    pub theirs: Option<Vec<String>>,
    /// Segments before this one keep the battle's own dice (START on `seed0`, no reseed): an alternative
    /// truth whose dice differ from one segment on. 0: every segment on fresh dice (the builder).
    pub from_segment: usize,
    pub seed0: Option<String>,
}

/// The failing segment's best try: its new viewer lines and their common prefix with I's.
#[derive(Clone, Debug)]
pub struct Fail {
    pub segment: usize,
    pub lcp: usize,
    pub lines: Vec<String>,
    /// The commands that try fed (`CHOOSE pN <tok>`).
    pub fed: Vec<String>,
    /// The first try at this segment whose first difference is an HP line of one mon in both: that mon,
    /// then the mon the reference line names `[of]` (a recoil or a drain reads another mon's HP).
    pub hp: Vec<String>,
}

/// A world's verdict (module docs). `ok` iff every segment matched and the battle sits at the
/// viewer's root decision with every one of its commands fed.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub ok: bool,
    /// Segments matched (the START write included).
    pub matched: usize,
    pub segments: usize,
    /// The START seed kept, the reseed schedule and the fed commands (`CHOOSE pN <tok>`); on a failure,
    /// those of the matched prefix (a diagnosis replays them; not a world).
    pub seed: Option<String>,
    pub reseed: Vec<(usize, String)>,
    pub cmds: Vec<String>,
    /// The damage-roll table the kept trajectory took where it differs from the stream's
    /// (`Mode::Force` replays it), and Σ log(k / 16) over the controlled sites.
    pub rolls: Vec<(u32, u8)>,
    pub log_w: Option<f64>,
    /// Per segment, the try kept.
    pub tries: Vec<u32>,
    pub feeds: u64,
    pub rerolls: u32,
    pub fail: Option<Fail>,
    pub error: Option<String>,
}

/// The battle paused at the start of a segment's update, with what the replay fed so far.
#[derive(Clone)]
struct Frame {
    sess: BridgeSession,
    cmds: Vec<String>,
    reseed: Vec<(usize, String)>,
    ours: usize,
    /// FIXED mode: the opponent's commands fed so far.
    theirs: usize,
    seed: String,
}

/// FNV-1a over `bytes`.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed of try `t` at segment `seg` under the base `dice`: `sodium,<32 hex>`, a fixed mix of the
/// three (FNV-1a, then SplitMix64 on two lanes).
pub fn try_seed(dice: &str, seg: usize, t: u32) -> String {
    let h = fnv1a(dice.as_bytes());
    let a = splitmix64(h ^ (seg as u64).wrapping_mul(0xA24B_AED4_963E_E407) ^ u64::from(t).wrapping_mul(0x9FB2_1C65_1E98_DF25));
    let b = splitmix64(a ^ 0xD6E8_FEB8_6659_FD93);
    format!("sodium,{a:016x}{b:016x}")
}

/// A reseed schedule as JSON (`[[k, seed], …]`; absent or null = none): ascending `k`, each named
/// once, every seed one the PRNG takes. Shared by `core_events`' START `"reseed"`.
pub fn schedule_from_json(v: Option<&Json>) -> Result<Vec<(usize, String)>, String> {
    let a = match v {
        None | Some(Json::Null) => return Ok(Vec::new()),
        Some(x) => x.as_array().ok_or("reseed: must be an array of [k, seed]")?,
    };
    let mut out: Vec<(usize, String)> = Vec::with_capacity(a.len());
    for e in a {
        let p = e.as_array().filter(|p| p.len() == 2).ok_or("reseed: an entry is [k, seed]")?;
        let k = p[0].as_f64().filter(|x| *x >= 0.0 && x.fract() == 0.0).ok_or("reseed: a non-integer index")? as usize;
        let seed = p[1].as_str().ok_or("reseed: a seed must be a string")?;
        Prng::validate_seed(seed).map_err(|e| format!("reseed: seed {seed:?}: {e}"))?;
        if out.last().is_some_and(|(j, _)| *j >= k) {
            return Err(format!("reseed: indices must ascend, each named once (command {k})"));
        }
        out.push((k, seed.to_string()));
    }
    Ok(out)
}

fn needs(sess: &BridgeSession, side: usize) -> bool {
    matches!(sess.request_kind(side), Some(RequestState::Move | RequestState::Switch)) && !sess.is_choice_done(side)
}

fn toid(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

/// The choices `side`'s open request offers, as core tokens, in request order: its usable moves
/// (`move <id>`; `move 1` for a recharge, `move struggle` for Struggle), then its switches
/// (`switch <name>`) unless it is trapped.
pub fn offered(sess: &BridgeSession, side: usize) -> Result<Vec<String>, String> {
    let raw = sess.active_request_json(side).ok_or("no open request")?;
    let body = raw.strip_prefix("|request|").unwrap_or(raw);
    let rq = Json::parse(body).map_err(|e| format!("request JSON: {e}"))?;
    let mut out = Vec::new();
    let mut trapped = false;
    let force = rq.get("forceSwitch").is_some_and(|f| !matches!(f, Json::Null));
    if !force {
        if let Some(act) = rq.get("active").and_then(Json::as_array).and_then(|a| a.first()) {
            trapped = act.bool_or("trapped", false);
            for m in act.get("moves").and_then(Json::as_array).unwrap_or(&[]) {
                if m.bool_or("disabled", false) {
                    continue;
                }
                let id = m.str_at("id").ok_or("a request move without an id")?;
                out.push(if id == "recharge" { "move 1".to_string() } else { format!("move {id}") });
            }
        }
    }
    if !trapped {
        for p in rq.get("side").and_then(|s| s.get("pokemon")).and_then(Json::as_array).unwrap_or(&[]) {
            if p.bool_or("active", false) {
                continue;
            }
            if p.str_at("condition").is_some_and(|c| c.starts_with("0 ") || c == "0") {
                continue;
            }
            let ident = p.str_at("ident").ok_or("a request mon without an ident")?;
            let name = ident.split_once(": ").map_or(ident, |(_, n)| n);
            out.push(format!("switch {name}"));
        }
    }
    Ok(out)
}

/// The offered choice an INTENT names: the token itself, a Hidden Power of any type for `move
/// hiddenpower…`, a switch by its name's id; `None` when the world offers no such choice.
fn resolve_intent(offered: &[String], tok: &str) -> Option<String> {
    if let Some(t) = offered.iter().find(|t| *t == tok) {
        return Some(t.clone());
    }
    if tok.starts_with("move hiddenpower") {
        let mut hp = offered.iter().filter(|t| t.starts_with("move hiddenpower"));
        return match (hp.next(), hp.next()) {
            (Some(t), None) => Some(t.clone()),
            _ => None,
        };
    }
    if let Some(name) = tok.strip_prefix("switch ") {
        let want = toid(name);
        return offered.iter().find(|t| t.strip_prefix("switch ").is_some_and(|n| toid(n) == want)).cloned();
    }
    None
}

/// The mon an HP line (`-damage` / `-heal` / `-sethp`) is about, by its ident.
fn hp_line_ident(line: &str) -> Option<&str> {
    let mut p = line.split('|');
    p.next();
    match p.next()? {
        "-damage" | "-heal" | "-sethp" => p.next(),
        _ => None,
    }
}

fn lcp(a: &[String], b: &[String]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// What one segment's tries came to.
enum Seg {
    /// The frame after it, and the (try, pick offset) kept.
    Kept(Frame, u32, usize),
    NoMatch,
}

/// The pick offsets an HP re-roll steps through at one (segment, try) before its next try.
const MAX_OFFSETS: usize = 16;
/// HP re-rolls spent at one hit before the next-earlier hit of the same mon is re-rolled.
const HP_PER_HIT: u32 = 4;

struct Run<'a> {
    base: &'a BattleOptions,
    info: &'a InfoSet,
    cfg: &'a MatchCfg,
    dex: &'a Dex,
    feeds: u64,
    best: Option<Fail>,
}

impl Run<'_> {
    fn spend(&mut self) -> Result<(), String> {
        self.feeds += 1;
        if self.feeds > self.cfg.max_feeds {
            return Err(format!("the world's budget of {} feeds is spent", self.cfg.max_feeds));
        }
        Ok(())
    }

    fn note(&mut self, s: usize, lines: Vec<String>, fed: Vec<String>) {
        let want = &self.info.segments[s];
        let k = lcp(&lines, want);
        let hp: Vec<String> = match (lines.get(k).and_then(|l| hp_line_ident(l)), want.get(k)) {
            (Some(a), Some(w)) if hp_line_ident(w) == Some(a) => {
                let of = w.split('|').filter_map(|t| t.strip_prefix("[of] ")).map(str::to_string);
                std::iter::once(a.to_string()).chain(of).collect()
            }
            _ => Vec::new(),
        };
        let prev_hp = self.best.as_ref().filter(|b| b.segment == s).map(|b| b.hp.clone()).unwrap_or_default();
        if self.best.as_ref().is_none_or(|b| b.segment != s || k > b.lcp) {
            self.best = Some(Fail { segment: s, lcp: k, lines, fed, hp: if prev_hp.is_empty() { hp } else { prev_hp } });
        } else if let Some(b) = self.best.as_mut() {
            if b.hp.is_empty() {
                b.hp = hp;
            }
        }
    }

    fn start(&mut self, t0: u32) -> Result<Seg, String> {
        let me = self.info.viewer;
        let want = self.info.segments.first().ok_or("an information set with no segment")?.clone();
        let own = self.cfg.from_segment > 0;
        let last = if own { t0 + 1 } else { self.cfg.tries };
        for t in t0..last {
            self.spend()?;
            let seed = match (&self.cfg.seed0, own) {
                (Some(s0), true) => s0.clone(),
                (None, true) => return Err("from_segment > 0 needs seed0 (the battle's own seed)".into()),
                _ => try_seed(&self.cfg.dice, 0, t),
            };
            let opts = BattleOptions { seed: Some(seed.clone()), ..self.base.clone() };
            let mut sess = BridgeSession::new_construct_turn0(&opts, self.dex)?;
            if self.cfg.rolls {
                sess.set_roll_ctl(Some(RollCtl::new(me as u8, Mode::Seg { lines: want.clone(), log_start: 0, offset: 0 })))?;
            }
            let lines = sess.side_lines(me);
            if lines[..] == want[..] {
                sess.clear_chunks();
                return Ok(Seg::Kept(Frame { sess, cmds: Vec::new(), reseed: Vec::new(), ours: 0, theirs: 0, seed }, t, 0));
            }
            self.note(0, lines, Vec::new());
        }
        Ok(Seg::NoMatch)
    }

    fn update(&mut self, f: &Frame, s: usize, t0: u32, o0: usize) -> Result<Seg, String> {
        let (me, opp) = (self.info.viewer, 1 - self.info.viewer);
        let want = self.info.segments[s].clone();
        let (me_needs, opp_needs) = (needs(&f.sess, me), needs(&f.sess, opp));
        if !me_needs && !opp_needs {
            self.note(s, Vec::new(), Vec::new());
            return Ok(Seg::NoMatch);
        }
        let ours = if me_needs {
            match self.info.ours.get(f.ours) {
                Some(t) => Some(t.clone()),
                None => {
                    self.note(s, Vec::new(), Vec::new());
                    return Ok(Seg::NoMatch);
                }
            }
        } else {
            None
        };
        let fixed = self.cfg.theirs.as_ref();
        let cands: Vec<Option<String>> = if opp_needs && fixed.is_some() {
            // FIXED mode: the opponent's next command (its later ones follow it while it is refused)
            match fixed.and_then(|th| th.get(f.theirs)) {
                Some(tok) => vec![Some(tok.clone())],
                None => {
                    self.note(s, Vec::new(), Vec::new());
                    return Ok(Seg::NoMatch);
                }
            }
        } else if opp_needs {
            let off = offered(&f.sess, opp)?;
            match self.info.intents.get(s).cloned().flatten() {
                Some(tok) => match resolve_intent(&off, &tok) {
                    Some(t) => vec![Some(t)],
                    None => {
                        self.note(s, Vec::new(), Vec::new());
                        return Ok(Seg::NoMatch);
                    }
                },
                // a stand-in: every offered choice, in request order (moves, then switches)
                None => off.into_iter().map(Some).collect(),
            }
        } else {
            vec![None]
        };
        // the first attempt may carry a PICK OFFSET (an HP re-roll: the same dice, another consistent roll);
        // a segment before `from_segment` keeps the battle's own dice: one attempt, no reseed
        let own = s < self.cfg.from_segment;
        let last_t = if own { t0 + 1 } else { self.cfg.tries };
        let attempts = std::iter::once((t0, o0)).chain((t0 + 1..last_t).map(|t| (t, 0)));
        for (t, offset) in attempts {
            if t >= last_t {
                break;
            }
            let seed = try_seed(&self.cfg.dice, s, t);
            for c in &cands {
                self.spend()?;
                let mut w = f.sess.clone();
                let mut fed: Vec<String> = Vec::with_capacity(2);
                let mut theirs = f.theirs;
                // every feed of the update is preceded by the same reseed (a feed that resolves nothing draws
                // nothing, so only the resolving one's matters; recording each keeps the replay exact)
                let feed = |w: &mut BridgeSession, side: usize, tok: &str, fed: &mut Vec<String>| -> Result<(), String> {
                    if !own {
                        w.reseed(&seed);
                    }
                    if self.cfg.rolls {
                        w.set_roll_segment(want.clone(), offset)?;
                    }
                    let choice = parse_choice(tok).ok_or_else(|| format!("segment {s}: an unparseable token {tok:?}"))?;
                    w.feed_cmd(Cmd { side, choice }, self.dex);
                    fed.push(format!("CHOOSE p{} {tok}", side + 1));
                    Ok(())
                };
                if let Some(c) = c {
                    feed(&mut w, opp, c, &mut fed)?;
                    if fixed.is_some() {
                        theirs += 1;
                        // a refused fixed command: the opponent's next one answers the re-request
                        while needs(&w, opp) && w.fatal().is_none() {
                            match fixed.and_then(|th| th.get(theirs)) {
                                Some(tok) => {
                                    feed(&mut w, opp, tok, &mut fed)?;
                                    theirs += 1;
                                }
                                None => break,
                            }
                        }
                    }
                }
                if let Some(o) = &ours {
                    feed(&mut w, me, o, &mut fed)?;
                }
                let lines = w.side_lines(me);
                if w.fatal().is_none() && lines[..] == want[..] {
                    w.clear_chunks();
                    let mut cmds = f.cmds.clone();
                    let mut reseed = f.reseed.clone();
                    if !own {
                        reseed.extend((cmds.len()..cmds.len() + fed.len()).map(|k| (k, seed.clone())));
                    }
                    cmds.extend(fed);
                    let frame = Frame { sess: w, cmds, reseed, ours: f.ours + usize::from(ours.is_some()), theirs, seed: f.seed.clone() };
                    return Ok(Seg::Kept(frame, t, offset));
                }
                self.note(s, lines, fed);
            }
        }
        Ok(Seg::NoMatch)
    }
}

/// Match one world (`base`: the format, both names and teams; its seed is ignored) against `info`.
pub fn match_world(base: &BattleOptions, info: &InfoSet, cfg: &MatchCfg, dex: &Dex) -> Report {
    let n = info.segments.len();
    let mut rep = Report { segments: n, ..Report::default() };
    let mut run = Run { base, info, cfg, dex, feeds: 0, best: None };
    // frames[s]: the battle at the start of segment s's update (frames[0] is the construction: None)
    let mut frames: Vec<Option<Frame>> = vec![None];
    let mut kept: Vec<(u32, usize)> = vec![(0, 0); n];
    let mut next: Vec<(u32, usize)> = vec![(0, 0); n];
    let mut rerolls = cfg.rerolls;
    // HP re-rolls spent per (failing segment, mon): `HP_PER_HIT` at the latest hit, then the next, …
    let mut hp_depth: std::collections::HashMap<(usize, String), u32> = std::collections::HashMap::new();
    let mut s = 0usize;
    let res: Result<Option<Frame>, String> = loop {
        if s == n {
            break Ok(frames.pop().flatten());
        }
        let got = match &frames[s] {
            None if s == 0 => run.start(next[0].0),
            None => Err(format!("segment {s}: no frame")),
            Some(f) => {
                let f = f.clone();
                run.update(&f, s, next[s].0, next[s].1)
            }
        };
        match got {
            Err(e) => break Err(e),
            Ok(Seg::Kept(frame, t, o)) => {
                kept[s] = (t, o);
                frames.truncate(s + 1);
                frames.push(Some(frame));
                s += 1;
            }
            Ok(Seg::NoMatch) => {
                let mut cands: Vec<usize> = info.backtrack.get(s).cloned().unwrap_or_default();
                if cfg.prev_backtrack && s > 0 {
                    cands.push(s - 1);
                }
                let hp_idents = run.best.as_ref().filter(|b| b.segment == s).map(|b| b.hp.clone()).unwrap_or_default();
                let hp = hp_idents.into_iter().find_map(|ident| {
                    let hits = &info.hp_last.get(s)?.iter().find(|(i, _)| *i == ident)?.1;
                    let d = hp_depth.entry((s, ident)).or_insert(0);
                    let pick = hits.get((*d / HP_PER_HIT) as usize).or(hits.last()).copied();
                    *d += 1;
                    pick
                });
                // an HP re-roll keeps the hit's dice and steps its pick offset; any other re-roll takes the next try
                let target = match hp.filter(|&b| b < s) {
                    Some(b) => Some((b, true)),
                    None => cands.into_iter().filter(|&b| b < s).max().map(|b| (b, false)),
                };
                match target {
                    Some((b, is_hp)) if rerolls > 0 => {
                        rerolls -= 1;
                        rep.rerolls += 1;
                        let (t, o) = kept[b];
                        next[b] = if is_hp && o + 1 < MAX_OFFSETS { (t, o + 1) } else { (t + 1, 0) };
                        for x in next.iter_mut().take(s + 1).skip(b + 1) {
                            *x = (0, 0);
                        }
                        frames.truncate(b + 1);
                        s = b;
                    }
                    _ => break Ok(None),
                }
            }
        }
    };
    rep.feeds = run.feeds;
    rep.tries = kept.iter().map(|(t, _)| *t).collect();
    match res {
        Err(e) => {
            rep.matched = s;
            rep.error = Some(e);
            rep.fail = run.best;
        }
        Ok(None) => {
            rep.matched = s;
            rep.fail = run.best;
            // the matched PREFIX (a diagnosis replays it; `ok` is false, so it is no world)
            if let Some(Some(f)) = frames.get(s) {
                rep.seed = Some(f.seed.clone());
                rep.reseed = f.reseed.clone();
                rep.cmds = f.cmds.clone();
                if let Some(ctl) = f.sess.roll_ctl() {
                    rep.rolls = ctl.sites.iter().filter(|x| x.r != x.r0).map(|x| (x.n, x.r)).collect();
                }
            }
        }
        Ok(Some(f)) => {
            rep.matched = n;
            let at_root = needs(&f.sess, info.viewer) && f.ours == info.ours.len();
            if !at_root {
                rep.error = Some(format!(
                    "every segment matched but the battle is not at the viewer's root decision ({} of {} commands fed, open: {})",
                    f.ours,
                    info.ours.len(),
                    needs(&f.sess, info.viewer)
                ));
            }
            if let Some(ctl) = f.sess.roll_ctl() {
                rep.rolls = ctl.sites.iter().filter(|x| x.r != x.r0).map(|x| (x.n, x.r)).collect();
                rep.log_w = ctl.log_w;
            }
            rep.ok = at_root;
            rep.seed = Some(f.seed);
            rep.reseed = f.reseed;
            rep.cmds = f.cmds;
        }
    }
    rep
}

fn strs(v: Option<&Json>, what: &str) -> Result<Vec<String>, String> {
    v.and_then(Json::as_array)
        .ok_or_else(|| format!("INFOSET: `{what}` must be an array of strings"))?
        .iter()
        .map(|x| x.as_str().map(str::to_string).ok_or_else(|| format!("INFOSET: `{what}` holds a non-string")))
        .collect()
}

impl InfoSet {
    /// `{"viewer":0|1,"segments":[[line,…],…],"ours":[tok,…],"intents":[tok|null,…],"backtrack":[[s,…],…],
    /// "hp_last":[[[ident,[s,…]],…],…]}` (`intents`, `backtrack`, `hp_last` one entry per segment). Unknown keys
    /// are refused.
    pub fn from_json(v: &Json) -> Result<InfoSet, String> {
        let obj = v.as_object().ok_or("INFOSET: must be an object")?;
        for k in obj.keys() {
            if !["viewer", "segments", "ours", "intents", "backtrack", "hp_last"].contains(&k.as_str()) {
                return Err(format!("INFOSET: unknown key {k:?}"));
            }
        }
        let viewer = match v.get("viewer").and_then(Json::as_f64) {
            Some(x) if x == 0.0 || x == 1.0 => x as usize,
            _ => return Err("INFOSET: `viewer` must be 0 or 1".into()),
        };
        let segs = v.get("segments").and_then(Json::as_array).ok_or("INFOSET: `segments` must be an array")?;
        let segments: Vec<Arc<[String]>> =
            segs.iter().map(|s| strs(Some(s), "segments[]").map(Arc::from)).collect::<Result<_, _>>()?;
        if segments.is_empty() {
            return Err("INFOSET: no segment".into());
        }
        let ours = strs(v.get("ours"), "ours")?;
        let ia = v.get("intents").and_then(Json::as_array).ok_or("INFOSET: `intents` must be an array")?;
        let intents: Vec<Option<String>> = ia
            .iter()
            .map(|x| match x {
                Json::Null => Ok(None),
                Json::Str(s) => Ok(Some(s.clone())),
                _ => Err("INFOSET: an intent is a token or null".to_string()),
            })
            .collect::<Result<_, _>>()?;
        let ba = v.get("backtrack").and_then(Json::as_array).ok_or("INFOSET: `backtrack` must be an array")?;
        let backtrack: Vec<Vec<usize>> = ba
            .iter()
            .map(|x| {
                x.as_array()
                    .ok_or("INFOSET: a backtrack entry is an array")?
                    .iter()
                    .map(|j| j.as_f64().filter(|f| *f >= 0.0 && f.fract() == 0.0).map(|f| f as usize).ok_or("INFOSET: a backtrack index"))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()?;
        let ha = v.get("hp_last").and_then(Json::as_array).ok_or("INFOSET: `hp_last` must be an array")?;
        let hp_last: Vec<Vec<(String, Vec<usize>)>> = ha
            .iter()
            .map(|x| {
                x.as_array()
                    .ok_or("INFOSET: an hp_last entry is an array")?
                    .iter()
                    .map(|e| {
                        let p = e.as_array().filter(|p| p.len() == 2).ok_or("INFOSET: an hp_last item is [ident, [segment, …]]")?;
                        let id = p[0].as_str().ok_or("INFOSET: an hp_last ident")?.to_string();
                        let sg = p[1]
                            .as_array()
                            .ok_or("INFOSET: an hp_last item's segments")?
                            .iter()
                            .map(|j| j.as_f64().filter(|f| *f >= 0.0 && f.fract() == 0.0).map(|f| f as usize).ok_or("INFOSET: an hp_last segment"))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok((id, sg))
                    })
                    .collect::<Result<Vec<_>, &str>>()
            })
            .collect::<Result<_, _>>()?;
        if intents.len() != segments.len() || backtrack.len() != segments.len() || hp_last.len() != segments.len() {
            return Err(format!(
                "INFOSET: {} segments, {} intents, {} backtrack, {} hp_last entries (one each per segment)",
                segments.len(),
                intents.len(),
                backtrack.len(),
                hp_last.len()
            ));
        }
        Ok(InfoSet { viewer, segments, ours, intents, backtrack, hp_last })
    }
}

impl MatchCfg {
    /// A MATCH line's settings: `"dice"` (required), `"tries"`, `"rerolls"`, `"rolls"`, `"prev_backtrack"`,
    /// `"max_feeds"`, and the leak test's `"theirs"`, `"from_segment"`, `"seed0"`.
    pub fn from_json(v: &Json) -> Result<MatchCfg, String> {
        let dice = v.str_at("dice").ok_or("MATCH: `dice` (the try seeds' base) is required")?.to_string();
        let num = |k: &str, d: u32| -> Result<u32, String> {
            match v.get(k) {
                None | Some(Json::Null) => Ok(d),
                Some(x) => x.as_f64().filter(|f| *f >= 0.0 && f.fract() == 0.0 && *f <= 1.0e6).map(|f| f as u32).ok_or(format!("MATCH: `{k}` must be a count")),
            }
        };
        let tries = num("tries", 64)?;
        if tries == 0 {
            return Err("MATCH: `tries` must be >= 1".into());
        }
        let theirs = match v.get("theirs") {
            None | Some(Json::Null) => None,
            Some(x) => Some(strs(Some(x), "theirs")?),
        };
        let from_segment = num("from_segment", 0)? as usize;
        let seed0 = v.str_at("seed0").map(str::to_string);
        if let Some(s0) = &seed0 {
            Prng::validate_seed(s0).map_err(|e| format!("MATCH: seed0 {s0:?}: {e}"))?;
        }
        Ok(MatchCfg {
            dice,
            tries,
            rerolls: num("rerolls", 8)?,
            rolls: v.bool_or("rolls", false),
            prev_backtrack: v.bool_or("prev_backtrack", false),
            max_feeds: u64::from(num("max_feeds", MAX_FEEDS as u32)?),
            theirs,
            from_segment,
            seed0,
        })
    }
}

fn q(o: &mut String, s: &str) {
    crate::core_events::json_out::str_into(o, s);
}

impl Report {
    /// One JSON object (the `MATCH` line's answer, after its label).
    pub fn json_into(&self, o: &mut String) {
        o.push_str(&format!(",\"ok\":{},\"matched\":{},\"segments\":{},\"seed\":", self.ok, self.matched, self.segments));
        match &self.seed {
            Some(s) => q(o, s),
            None => o.push_str("null"),
        }
        o.push_str(",\"reseed\":[");
        for (i, (k, s)) in self.reseed.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str(&format!("[{k},"));
            q(o, s);
            o.push(']');
        }
        o.push_str("],\"cmds\":[");
        for (i, c) in self.cmds.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            q(o, c);
        }
        o.push_str("],\"rolls\":[");
        for (i, (n, r)) in self.rolls.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str(&format!("[{n},{r}]"));
        }
        o.push_str("],\"log_w\":");
        match self.log_w {
            Some(w) => o.push_str(&format!("{w:?}")),
            None => o.push_str("null"),
        }
        o.push_str(&format!(",\"tries\":{:?},\"feeds\":{},\"rerolls\":{},\"fail\":", self.tries, self.feeds, self.rerolls));
        match &self.fail {
            None => o.push_str("null"),
            Some(f) => {
                o.push_str(&format!("{{\"segment\":{},\"lcp\":{},\"lines\":[", f.segment, f.lcp));
                for (i, l) in f.lines.iter().enumerate() {
                    if i > 0 {
                        o.push(',');
                    }
                    q(o, l);
                }
                o.push_str("],\"fed\":[");
                for (i, l) in f.fed.iter().enumerate() {
                    if i > 0 {
                        o.push(',');
                    }
                    q(o, l);
                }
                o.push_str("]}");
            }
        }
        o.push_str(",\"error\":");
        match &self.error {
            Some(e) => q(o, e),
            None => o.push_str("null"),
        }
    }
}
