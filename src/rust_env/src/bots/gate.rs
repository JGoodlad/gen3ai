//! THE BOT GATE — replay a banked corpus (`src/utils/rust_env/bot_corpus.py`) through the env
//! core's battle path and hold every Rust bot decision equal to the Python bot's (M5 Lane F).
//!
//! Per episode: the battle is started from the banked inputs (teams, Showdown seed, names) on a
//! [`Game`] — the env core's own row path (Lane I's gate) — and fed the RECORDED actions of both
//! sides. At every p2 decision, BEFORE feeding it, the Rust bot decides on the core's p2 reading:
//!
//! 1. `view` — the view's hash equals the Python view's (the INPUTS are equal);
//! 2. every stream is advanced to the Python stream's offset before the decision (phantom polls in
//!    between are Python's; see `bots/mod.rs`), and after it must sit at the Python offset (the
//!    draw COUNT is equal);
//! 3. `token` and `index` equal the order the Python env sent (the ACTION is equal).
//!
//! The battle itself must replay: the same number of decisions per side and the same winner.

use pokesim::core_events::jsonval::Val;
use pokesim::trackers::clock::ClockConfig;

use super::view::{fnv64, View};
use super::{Bot, Kind, Stream};
use crate::search::game::{Game, Log};

/// Per-bot counts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tally {
    pub episodes: usize,
    pub decisions: usize,
    pub draw_decisions: usize,
    pub view: usize,
    pub action: usize,
    pub rng: usize,
    pub error: usize,
    pub replay: usize,
    /// `(logic.rs line, decisions)` — which return sites the corpus exercised.
    pub branches: Vec<(u32, usize)>,
}

impl Tally {
    pub fn mismatches(&self) -> usize {
        self.view + self.action + self.rng + self.error + self.replay
    }
}

/// The gate's result: per-bot tallies (in first-seen order) and the first examples.
#[derive(Debug, Default)]
pub struct Report {
    pub bots: Vec<(String, Tally)>,
    pub examples: Vec<String>,
}

impl Report {
    pub fn mismatches(&self) -> usize {
        self.bots.iter().map(|(_, t)| t.mismatches()).sum()
    }
    pub fn decisions(&self) -> usize {
        self.bots.iter().map(|(_, t)| t.decisions).sum()
    }
    fn tally(&mut self, bot: &str) -> &mut Tally {
        if let Some(i) = self.bots.iter().position(|(b, _)| b == bot) {
            return &mut self.bots[i].1;
        }
        self.bots.push((bot.to_string(), Tally::default()));
        &mut self.bots.last_mut().expect("pushed").1
    }
    fn example(&mut self, s: String) {
        if self.examples.len() < 12 {
            self.examples.push(s);
        }
    }
    /// One JSON line (the Python harness parses it).
    pub fn json(&self) -> String {
        let mut o = String::from("{\"bots\":{");
        for (i, (b, t)) in self.bots.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str(&format!(
                "\"{b}\":{{\"episodes\":{},\"decisions\":{},\"draw_decisions\":{},\"view\":{},\"action\":{},\"rng\":{},\"error\":{},\"replay\":{},\"branches\":{{{}}}}}",
                t.episodes, t.decisions, t.draw_decisions, t.view, t.action, t.rng, t.error, t.replay,
                t.branches.iter().map(|(l, n)| format!("\"{l}\":{n}")).collect::<Vec<_>>().join(",")
            ));
        }
        o.push_str("},\"examples\":[");
        for (i, e) in self.examples.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            super::view::str_into(&mut o, e);
        }
        o.push_str("]}");
        o
    }
}

fn int(v: &Val, what: &str) -> Result<i64, String> {
    match v {
        Val::Int(n) => Ok(*n),
        other => Err(format!("{what}: {other:?} is not an int")),
    }
}

fn s<'a>(v: &'a Val, key: &str) -> Result<&'a str, String> {
    v.str_at(key).ok_or_else(|| format!("corpus: missing string {key:?}"))
}

fn arr<'a>(v: &'a Val, key: &str) -> Result<&'a [Val], String> {
    match v.get(key) {
        Some(Val::Arr(a)) => Ok(a),
        _ => Err(format!("corpus: missing array {key:?}")),
    }
}

fn offsets(v: &Val, key: &str) -> Result<Vec<(Stream, u64)>, String> {
    let mut out = Vec::new();
    if let Some(Val::Obj(kv)) = v.get(key) {
        for (k, x) in kv {
            let st = match k.as_ref() {
                "choice" => Stream::Choice,
                "protect" => Stream::Protect,
                other => return Err(format!("corpus: unknown stream {other:?}")),
            };
            out.push((st, int(x, key)? as u64));
        }
        Ok(out)
    } else {
        Err(format!("corpus: missing {key:?}"))
    }
}

/// Replay every episode of `corpus` (the JSON text); `Err` only for a malformed corpus.
pub fn run(corpus: &str) -> Result<Report, String> {
    let root = Val::parse(corpus)?;
    if root.str_at("schema") != Some("gen3_bot_corpus_v1") {
        return Err(format!("corpus schema {:?}, expected gen3_bot_corpus_v1", root.str_at("schema")));
    }
    let mut rep = Report::default();
    for (e, ep) in arr(&root, "episodes")?.iter().enumerate() {
        episode(&mut rep, e, ep)?;
    }
    Ok(rep)
}

fn episode(rep: &mut Report, e: usize, ep: &Val) -> Result<(), String> {
    let bot_name = s(ep, "bot")?.to_string();
    let kind = Kind::from_name(&bot_name).ok_or_else(|| format!("corpus: no Rust bot named {bot_name:?}"))?;
    let seeds = ep.get("rng_seeds").ok_or("corpus: missing rng_seeds")?;
    let seed_of = |k: &str| seeds.get(k).map_or(Ok(0), |v| int(v, k)).map(|n| n as u64);
    let mut bot = Bot::new(kind, seed_of("choice")?, seed_of("protect")?);
    let names = arr(ep, "names")?;
    let teams = arr(ep, "teams")?;
    let str_of = |v: &Val| match v {
        Val::Str(x) => Ok(x.to_string()),
        o => Err(format!("corpus: {o:?} is not a string")),
    };
    let log = Log {
        format_id: "gen3ou".into(),
        seed: s(ep, "seed")?.to_string(),
        names: [str_of(&names[0])?, str_of(&names[1])?],
        teams: [str_of(&teams[0])?, str_of(&teams[1])?],
        cmds: Vec::new(),
    };
    let p1: Vec<i64> = arr(ep, "p1")?.iter().map(|v| int(v, "p1")).collect::<Result<_, _>>()?;
    let p2 = arr(ep, "p2")?;
    rep.tally(&bot_name).episodes += 1;
    let mut g = match Game::start(&log, ClockConfig::default()) {
        Ok(g) => g,
        Err(err) => {
            rep.tally(&bot_name).replay += 1;
            rep.example(format!("{bot_name} ep {e}: the battle does not start: {err}"));
            return Ok(());
        }
    };
    let (mut i1, mut i2) = (0usize, 0usize);
    let fail = |rep: &mut Report, what: String| {
        rep.tally(&bot_name).replay += 1;
        rep.example(format!("{bot_name} ep {e}: {what}"));
    };
    while !g.is_ended() {
        let (o1, o2) = (g.open(0).cloned(), g.open(1).cloned());
        if o1.is_none() && o2.is_none() {
            fail(rep, "no decision open and the battle has not ended".into());
            return Ok(());
        }
        let mut a2: Option<i32> = None;
        if let Some(open) = &o2 {
            let Some(d) = p2.get(i2) else {
                fail(rep, format!("the core opened p2 decision {i2}; Python banked {}", p2.len()));
                return Ok(());
            };
            decision(rep, &bot_name, e, i2, &mut bot, &g, &open.tokens, d)?;
            // `idx` null: the stall forfeit's decision — asked, never sent (p1 forfeits below)
            if let Some(Val::Int(idx)) = d.get("idx") {
                a2 = Some(*idx as i32);
            }
            i2 += 1;
        }
        if o1.is_some() {
            let Some(&a) = p1.get(i1) else {
                fail(rep, format!("the core opened p1 decision {i1}; Python banked {}", p1.len()));
                return Ok(());
            };
            i1 += 1;
            if a == -1 {
                if let Err(err) = g.forfeit(0) {
                    fail(rep, format!("forfeit: {err}"));
                    return Ok(());
                }
                continue;
            }
            if let Err(err) = g.feed(0, a as i32) {
                fail(rep, format!("p1 decision {}: {err}", i1 - 1));
                return Ok(());
            }
        }
        if let Some(a) = a2 {
            if let Err(err) = g.feed(1, a) {
                fail(rep, format!("p2 decision {}: {err}", i2 - 1));
                return Ok(());
            }
        }
    }
    if i1 != p1.len() || i2 != p2.len() {
        fail(rep, format!("decisions p1 {i1}/{} p2 {i2}/{} — the battle did not replay", p1.len(), p2.len()));
    }
    let end = ep.get("end").ok_or("corpus: missing end")?;
    let want = match end.get("won") {
        Some(Val::Bool(true)) => Some(0),
        Some(Val::Bool(false)) => Some(1),
        _ => None,
    };
    if g.winner() != want {
        fail(rep, format!("winner {:?}, Python {want:?}", g.winner()));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decision(
    rep: &mut Report,
    bot_name: &str,
    e: usize,
    k: usize,
    bot: &mut Bot,
    g: &Game,
    tokens: &[(usize, String)],
    d: &Val,
) -> Result<(), String> {
    let where_ = format!("{bot_name} ep {e} p2 decision {k} (turn {})", g.turn(1));
    rep.tally(bot_name).decisions += 1;
    let view = match View::build(g.reading(1)) {
        Ok(v) => v,
        Err(err) => {
            rep.tally(bot_name).error += 1;
            rep.example(format!("{where_}: the view refused: {err}"));
            return Ok(());
        }
    };
    match view.render() {
        Ok(text) if fnv64(&text) == s(d, "view")? => {}
        Ok(text) => {
            rep.tally(bot_name).view += 1;
            rep.example(format!("{where_}: VIEW differs — Rust view {text}"));
        }
        Err(err) => {
            rep.tally(bot_name).error += 1;
            rep.example(format!("{where_}: render refused: {err}"));
        }
    }
    let before = offsets(d, "before")?;
    let after = offsets(d, "after")?;
    for (st, w) in &before {
        if let Err(err) = bot.stream_mut(*st).skip_to(*w) {
            rep.tally(bot_name).rng += 1;
            rep.example(format!("{where_}: {err}"));
            return Ok(());
        }
    }
    let drew = before != after;
    if drew {
        rep.tally(bot_name).draw_decisions += 1;
    }
    match bot.decide_on(&view, tokens) {
        Ok(dec) => {
            let line = bot.branch;
            let br = &mut rep.tally(bot_name).branches;
            match br.iter_mut().find(|(l, _)| *l == line) {
                Some((_, n)) => *n += 1,
                None => {
                    br.push((line, 1));
                    br.sort();
                }
            }
            // a SENT decision compares the sent token and its index; the stall forfeit's (never
            // sent) compares the bot's own order
            let (tok, idx) = match d.get("idx") {
                Some(Val::Int(i)) => (s(d, "tok")?, Some(*i)),
                _ => (s(d, "bot")?, dec.index.map(|i| i as i64)),
            };
            if dec.token != tok || dec.index.map(|i| i as i64) != idx {
                rep.tally(bot_name).action += 1;
                rep.example(format!("{where_}: ACTION Rust {:?} / {:?}, Python {tok:?} / {idx:?}", dec.token, dec.index));
            }
        }
        Err(err) => {
            rep.tally(bot_name).error += 1;
            rep.example(format!("{where_}: the bot refused: {err}"));
        }
    }
    let now: Vec<(Stream, u64)> = after.iter().map(|(st, _)| (*st, bot.stream(*st).words)).collect();
    if now != after {
        rep.tally(bot_name).rng += 1;
        rep.example(format!("{where_}: RNG offsets after {now:?}, Python {after:?}"));
    }
    Ok(())
}
