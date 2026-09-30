//! ONE BATTLE ON THE TRAINING OBSERVATION PATH, BRANCHABLE — what a playout steps.
//!
//! A [`Game`] is the env core's battle (`crate::core::pool::Env`) minus the column plumbing, the
//! labels and the episode bookkeeping: a live `BridgeSession` and one PARSE fold per side over
//! exactly the lines each side was shipped by each write (`sim_bridge`'s `core_obs` chain, program
//! §6c — the fold `parse_root_unrecorded` → `parse_advance_lean` performs, trackers without the
//! native record), with `sim_bridge`'s ALIGNMENT rules (one decision per side per write, the
//! decision's request the write's last line). A decision's row is encoded on demand from its side's
//! stream (`BattleVersion::encode`'s inputs), its tokens are the real mapper's
//! (`present::choice_tokens`), and an action is an INDEX the caller chose from the mask.
//!
//! WHY NOT A `BattleVersion` CHAIN: a step-built version keeps its parent (`Arc`), so a 100-decision
//! playout would hold 100 engines per branch; a playout never revisits a boundary, so it folds
//! LINEARLY and keeps nothing behind it. WHY NOT the pool's `Env`: an `Env` is bound to a column
//! set and an episode lifecycle (auto-reset, labels, the stall forfeit on p1 only); a playout
//! branches mid-battle and ends where the battle ends.
//!
//! [`Game::branch`] is the CLONE: the engine is cloned into a fresh transport
//! (`BridgeSession::resume`, the search tree's `fork_session`), the two side streams are cloned
//! (their tracker state is an `Arc`, copied on its next decision). The gate that the clone is exact
//! and that this module IS the env core's row path: `tests/search_game_test.rs` replays the core's
//! own recorded episodes through a `Game` — every row, mask and token byte-equal at every decision,
//! also across a [`Game::branch`] taken at every decision.

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::core_events::Line;
use pokesim::encoder::OBS_DIM;
use pokesim::present::{choice_tokens, legal_actions, mask};
use pokesim::trackers::clock::ClockConfig;
use pokesim::version::SideStream;

use super::dex;
use crate::core::columns::ACT;

/// The battle inputs of one episode — the env core's INPUT LOG (`core::refusal::InputLog`): format,
/// the `ep_seed` string, the two names and packed teams, and the fed commands (`CHOOSE pN <tok>`,
/// `FORCELOSE pN`), in feed order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Log {
    pub format_id: String,
    pub seed: String,
    pub names: [String; 2],
    pub teams: [String; 2],
    pub cmds: Vec<String>,
}

/// An open decision: `(action index, choice token)` per legal action, and its per-side ordinal.
#[derive(Clone, Debug)]
pub struct Open {
    pub tokens: Vec<(usize, String)>,
    pub n: u32,
}

/// See the module docs.
#[derive(Clone)]
pub struct Game {
    sess: BridgeSession,
    streams: [SideStream; 2],
    decided: [u32; 2],
    /// Chunks of `sess` already folded.
    emitted: usize,
    open: [Option<Open>; 2],
    /// The commands fed so far (the input log's `cmds`, prefix included).
    pub cmds: Vec<String>,
}

fn stream(side: usize, name: &str, team: &str, clock: ClockConfig) -> Result<SideStream, String> {
    let mut s = SideStream::new(side, name, Some(team)).map_err(|e| format!("root p{}: {e}", side + 1))?.with_trackers(clock);
    s.trk = s.trk.take().map(|t| t.without_record());
    Ok(s)
}

impl Game {
    /// Start `log`'s battle (its `cmds` are NOT fed — see [`Game::replay`]).
    pub fn start(log: &Log, clock: ClockConfig) -> Result<Game, String> {
        let opts = BattleOptions {
            format_id: log.format_id.clone(),
            seed: Some(log.seed.clone()),
            p1: PlayerOptions { name: log.names[0].clone(), team: PackedTeam(log.teams[0].clone()) },
            p2: PlayerOptions { name: log.names[1].clone(), team: PackedTeam(log.teams[1].clone()) },
        };
        let sess = BridgeSession::new_construct_turn0(&opts, dex())?;
        let streams = [stream(0, &log.names[0], &log.teams[0], clock)?, stream(1, &log.names[1], &log.teams[1], clock)?];
        let mut g = Game { sess, streams, decided: [0, 0], emitted: 0, open: [None, None], cmds: Vec::new() };
        g.advance()?;
        Ok(g)
    }

    /// Start `log`'s battle and feed its first `at` commands (by TOKEN, through the same feed an
    /// index takes — a token the open decision does not offer is refused).
    pub fn replay(log: &Log, at: usize, clock: ClockConfig) -> Result<Game, String> {
        if at > log.cmds.len() {
            return Err(format!("replay: at = {at} is past the log's {} commands", log.cmds.len()));
        }
        let mut g = Game::start(log, clock)?;
        for (k, c) in log.cmds[..at].iter().enumerate() {
            if let Some(rest) = c.strip_prefix("FORCELOSE p") {
                let side = side_of(rest).ok_or_else(|| format!("replay: command {k} {c:?}"))?;
                g.forfeit(side)?;
                continue;
            }
            let rest = c.strip_prefix("CHOOSE p").ok_or_else(|| format!("replay: command {k} {c:?} is not CHOOSE / FORCELOSE"))?;
            let (tag, tok) = rest.split_once(' ').ok_or_else(|| format!("replay: command {k} {c:?}"))?;
            let side = side_of(tag).ok_or_else(|| format!("replay: command {k} {c:?}"))?;
            let idx = g.open[side]
                .as_ref()
                .and_then(|o| o.tokens.iter().find(|(_, t)| t == tok).map(|(i, _)| *i))
                .ok_or_else(|| format!("replay: command {k} {c:?} — p{} has no open decision offering that token", side + 1))?;
            g.feed(side, idx as i32)?;
        }
        Ok(g)
    }

    /// A branch of this game: the same board, stream and open decisions; its own dice from here
    /// (`reseed` = `Some(seed)` swaps them — COMMON RANDOM NUMBERS across the siblings that share a
    /// seed; `None` keeps the battle's own stream, which reproduces the original continuation).
    pub fn branch(&self, reseed: Option<&str>) -> Game {
        let mut sess = BridgeSession::resume(self.sess.engine().clone());
        if let Some(s) = reseed {
            sess.reseed(s);
        }
        Game {
            sess,
            streams: self.streams.clone(),
            decided: self.decided,
            emitted: 0,
            open: self.open.clone(),
            cmds: self.cmds.clone(),
        }
    }

    pub fn open(&self, side: usize) -> Option<&Open> {
        self.open[side].as_ref()
    }

    pub fn is_ended(&self) -> bool {
        self.sess.is_ended()
    }

    /// The winner (0 / 1) of an ended battle; `None` while playing or on a tie.
    pub fn winner(&self) -> Option<usize> {
        self.sess.winner()
    }

    /// `side`'s reading of the turn (what the env's stall forfeit reads).
    pub fn turn(&self, side: usize) -> u32 {
        self.streams[side].board_reading.turn
    }

    /// Encode `side`'s OPEN decision's row into `out` and its mask into `mask_out`.
    pub fn encode(&self, side: usize, out: &mut [f32; OBS_DIM], mask_out: &mut [u8; ACT]) -> Result<(), String> {
        if self.open[side].is_none() {
            return Err(format!("encode: p{} has no open decision", side + 1));
        }
        let s = &self.streams[side];
        let trk = s.trk.as_ref().ok_or("encode: a stream without trackers")?;
        let computed;
        let view = match trk.last.as_ref().and_then(|d| d.view.as_ref()) {
            Some(v) => v,
            None => {
                computed = s.view().map_err(|e| e.to_string())?;
                &computed
            }
        };
        let legal = legal_actions(&s.board_reading);
        let inputs = pokesim::encoder::Inputs { reading: &s.board_reading, view, legal: legal.as_ref(), trackers: &trk.trackers };
        pokesim::encoder::encode(&inputs, out).map_err(|e| format!("encode p{}: {e}", side + 1))?;
        *mask_out = legal.as_ref().map_or([0; ACT], mask);
        Ok(())
    }

    /// Fold every chunk past the write cursor through each side's stream — `pool::Env::advance`,
    /// rule for rule — and open each decision that the write opened.
    fn advance(&mut self) -> Result<(), String> {
        if let Some(f) = self.sess.fatal() {
            return Err(format!("engine fatal: {f}"));
        }
        let chunks = &self.sess.chunks().chunks[self.emitted..];
        for side in 0..2 {
            let tag = side + 1;
            let s = &mut self.streams[side];
            let before = s.lines;
            for ch in chunks.iter().filter(|ch| ch.side == side) {
                for text in &ch.lines {
                    let line = Line::parse(text).map_err(|e| format!("parse p{tag}: line {} {text:?}: {e:?}", s.lines))?;
                    s.fold_lean(line).map_err(|e| format!("parse p{tag}: {e}"))?;
                }
            }
            let decisions = s.trk.as_ref().map_or(0, |t| t.trackers.decisions);
            let opened = decisions
                .checked_sub(self.decided[side])
                .ok_or_else(|| format!("p{tag}: the decision count went backwards"))?;
            let at_boundary = s.trk.as_ref().and_then(|t| t.last.as_ref()).filter(|d| d.line >= before && s.lines > before);
            match (at_boundary, opened) {
                (None, 0) => {}
                (Some(d), 1) => {
                    if d.line + 1 != s.lines {
                        return Err(format!("[ALIGN] p{tag} decided at stream line {} but the write shipped {} lines", d.line, s.lines));
                    }
                    let legal = legal_actions(&s.board_reading).ok_or_else(|| format!("p{tag}: a decision with no legality"))?;
                    let tokens = choice_tokens(&s.board_reading, &legal).map_err(|e| format!("tokens p{tag}: {e}"))?;
                    self.open[side] = Some(Open { tokens, n: self.decided[side] });
                }
                (d, n) => return Err(format!("p{tag}: one write opened {n} decisions (a decision at the boundary: {})", d.is_some())),
            }
            self.decided[side] = decisions;
        }
        self.emitted = self.sess.chunks().chunks.len();
        Ok(())
    }

    /// Feed `side`'s action INDEX (chosen from its open decision's mask).
    pub fn feed(&mut self, side: usize, action: i32) -> Result<(), String> {
        let open = self.open[side].take().ok_or_else(|| format!("p{}: no decision open", side + 1))?;
        let tok = open
            .tokens
            .iter()
            .find(|(i, _)| *i as i64 == action as i64)
            .map(|(_, t)| t.clone())
            .ok_or_else(|| format!("p{} action {action} is not legal here (the mask forbids it)", side + 1))?;
        let choice = parse_choice(&tok).ok_or_else(|| format!("the mapper produced an unparseable token {tok:?}"))?;
        // `sim_bridge`'s order: the stream notes the raw token BEFORE the command is fed.
        if let Some(t) = self.streams[side].trk.as_mut() {
            t.choose(&tok);
        }
        self.cmds.push(format!("CHOOSE p{} {tok}", side + 1));
        self.sess.feed_cmd(Cmd { side, choice }, dex());
        self.advance()
    }

    /// `side` forfeits (`FORCELOSE pN`); every open decision closes.
    pub fn forfeit(&mut self, side: usize) -> Result<(), String> {
        self.cmds.push(format!("FORCELOSE p{}", side + 1));
        self.sess.forfeit(side);
        self.open = [None, None];
        self.advance()
    }
}

fn side_of(tag: &str) -> Option<usize> {
    match tag {
        "1" => Some(0),
        "2" => Some(1),
        _ => None,
    }
}
