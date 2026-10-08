//! A SCRIPTED BOT OVER ONE SIDE'S PROTOCOL STREAM (poke-env retirement P6, `main.anchors`'
//! `bot:<name>` our-side).
//!
//! The env core plays a Lane-F bot ([`crate::bots`]) on the core's own per-side reading
//! (`opponents::Env::bot_token`). An external-anchor read has no env core: its battle runs on a
//! websocket server (our `--server rust` front end, a Node Showdown, an external one) and our side
//! is a CLIENT of it. [`BotSide`] gives that client the same bot on the same reading: the side's
//! shipped lines are folded through [`pokesim::side_reader::SideReader`] — the ONE "protocol text in,
//! the training reading out" chain `sim_bridge`'s core observation mode and P4's `live_reader` share —
//! and at every decision the write opened, [`Bot::decide`] reads that side's `BoardReading` over the
//! decision's own choice tokens (`present::choice_tokens`, the env core's mapper).
//!
//! THE BOT OUTLIVES A BATTLE. A websocket client's bot is ONE `Player` per half-series whose RNG
//! streams run across its battles (the env core's `"streams": "env"` rule — one `Bot` per env for the
//! pool's life). So the bot is installed once ([`BotSide::new`]) and each battle only replaces the
//! READER ([`BotSide::open`]).
//!
//! RANDOMNESS is the env core's DECLARED rule (`opponents::stream_seed`): the caller passes each
//! stream's seed explicitly (`random.Random(seed)` — a seeded Python bot draws the same numbers).
//!
//! FAILURE: every reader refusal (sticky, `SideReader`'s rule) and every bot refusal (where the Python
//! bot would RAISE) is an `Err`; the caller halts the battle, never skips a decision.

use pokesim::present;
use pokesim::side_reader::SideReader;
use pokesim::trackers::clock::ClockConfig;

use crate::bots::{Bot, Decision, Kind};

/// One decision the bot took (the frame's `bot` object).
#[derive(Debug, Clone, PartialEq)]
pub struct BotChoice {
    pub decision: Decision,
    /// The choice stream's 32-bit word offset AFTER the decision (`PyRandom::words`).
    pub choice_words: u64,
    /// The protect stream's offset after the decision (stallers draw from it; 0 otherwise).
    pub protect_words: u64,
    /// The `logic.rs` return site that decided (`Bot::branch`).
    pub branch: u32,
}

/// What one [`BotSide::advance`] produced.
#[derive(Debug, Default)]
pub struct Advanced {
    /// The decision's `__OBS__` frame JSON (`encoder::wire::obs_json_into`) when the write opened one.
    pub frame: Option<Vec<u8>>,
    /// The bot's choice at that decision.
    pub choice: Option<BotChoice>,
}

/// See the module docs.
pub struct BotSide {
    bot: Bot,
    reader: Option<SideReader>,
}

impl BotSide {
    /// The bot `name` (a `bots::Kind` name), its streams seeded `random.Random(choice_seed)` /
    /// `random.Random(protect_seed)`.
    pub fn new(name: &str, choice_seed: u64, protect_seed: u64) -> Result<BotSide, String> {
        let kind = Kind::from_name(name).ok_or_else(|| format!("bot_side: {name:?} is not a bot this core plays"))?;
        Ok(BotSide { bot: Bot::new(kind, choice_seed, protect_seed), reader: None })
    }

    pub fn kind(&self) -> Kind {
        self.bot.kind
    }

    /// A new battle: side `side` (0 = p1) of the player `name` holding `packed_team`. The bot (and its
    /// streams) carries over; the reader is fresh.
    pub fn open(&mut self, side: usize, name: &str, packed_team: Option<&str>) -> Result<(), String> {
        self.reader = None;
        self.reader = Some(SideReader::new(side, name, packed_team, ClockConfig::default())?);
        Ok(())
    }

    /// The open battle's reader (`None` before [`BotSide::open`]).
    pub fn reader(&self) -> Option<&SideReader> {
        self.reader.as_ref()
    }

    /// Note the choice this side sends for its open decision, BEFORE the next write is fed.
    pub fn note_choice(&mut self, token: &str) -> Result<(), String> {
        self.reader.as_mut().ok_or("bot_side: CHOOSE before OPEN")?.note_choice(token);
        Ok(())
    }

    /// Fold one write; when it opened a decision, the frame AND the bot's choice on it.
    pub fn advance(&mut self, lines: &[&str]) -> Result<Advanced, String> {
        let r = self.reader.as_mut().ok_or("bot_side: FEED before OPEN")?;
        let mut frame = Vec::new();
        let adv = r.advance(lines, &mut frame)?;
        if !adv.decided {
            return Ok(Advanced::default());
        }
        let side = r.side();
        let v = r.version().ok_or("bot_side: the reader lost its chain after a decision")?;
        let legal = v.legal(side).ok_or("bot_side: a decision with no legality")?;
        let reading = &v.stream(side).ok_or("bot_side: the reader lost its stream")?.board_reading;
        let tokens = present::choice_tokens(reading, &legal).map_err(|e| format!("bot_side: tokens: {}", e.message()))?;
        let decision = self.bot.decide(reading, &tokens).map_err(|e| format!("{:?} {e}", self.bot.kind))?;
        if decision.index.is_none() {
            // `Order::Default` — the Python bot's `choose_default_move` at a decision whose
            // `valid_orders` is empty. The core opens a decision only with a legal action, so this is
            // a disagreement between the bot's view and the core's legality: refused, never sent.
            return Err(format!("bot_side: {:?} returned the DEFAULT order at an open decision (tokens {tokens:?})", self.bot.kind));
        }
        let choice = BotChoice {
            decision,
            choice_words: self.bot.choice.words,
            protect_words: self.bot.protect.words,
            branch: self.bot.branch,
        };
        Ok(Advanced { frame: Some(frame), choice: Some(choice) })
    }
}

impl BotChoice {
    /// `{"index":<int>,"token":"…","choice_words":<int>,"protect_words":<int>,"branch":<int>}`.
    pub fn json(&self) -> String {
        let mut tok = String::new();
        crate::bots::view::str_into(&mut tok, &self.decision.token);
        format!(
            "{{\"index\":{},\"token\":{tok},\"choice_words\":{},\"protect_words\":{},\"branch\":{}}}",
            self.decision.index.map_or("null".to_string(), |i| i.to_string()),
            self.choice_words,
            self.protect_words,
            self.branch
        )
    }
}
