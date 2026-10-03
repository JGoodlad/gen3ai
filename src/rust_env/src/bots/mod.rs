//! THE SCRIPTED BOTS in the Rust env core (M5 Lane F, `designs/endstate/program_rust_core.md` §2 M5;
//! owner 2026-09-27: port them).
//!
//! One [`Kind`] per inventoried bot class (`src/utils/rust_env/bot_inventory.py`). A [`Bot`] holds
//! its kind and its RNG streams; [`Bot::decide`] reads the bot side's `BoardReading` at an open
//! decision (through [`view::View`], the proven-equal input) and returns the order, its choice
//! token and the 11-dim action INDEX the env core feeds — the index `serialize.order_to_action`
//! gives the Python bot's order.
//!
//! THE GATE: per-decision ACTION EQUALITY against the Python bot on a banked corpus
//! (`src/utils/rust_env/bot_corpus.py` records; `tests/bots_gate_test.rs` replays): at every banked
//! decision the view hash, the action index, the token and every stream's offset after the decision
//! must equal the Python bot's.
//!
//! RANDOMNESS (decided; see [`rng`]): each stream is CPython's MT19937 from the same seed, so a seeded
//! Python bot and the Rust bot draw the same numbers. Production seeding is the host's (Lane E / G):
//! the Python production bots draw from the unseeded process-wide `random`, which no port can
//! reproduce; the Rust env seeds every stream explicitly — a DECLARED change of stream, not of
//! distribution. The Python training wrapper also consumes draws on
//! PHANTOM polls (`choose_move` on a step whose p2 order is never sent); the Rust env asks a bot
//! only at a real decision (finding F-LF-2).

pub mod calc;
pub mod gate;
pub mod logic;
pub mod rng;
pub mod tables;
pub mod view;

use pokesim::present::board_reading::BoardReading;

use rng::PyRandom;
use view::{Order, View};

/// A bot's refusal — where the Python bot would RAISE (an unknown move, a `None` stat compared, an
/// empty `max`). Typed, never guessed around.
#[derive(Debug, Clone, PartialEq)]
pub struct BotError(pub String);

impl std::fmt::Display for BotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bot: {}", self.0)
    }
}

/// Every ported bot, named by its inventory display name ([`Kind::name`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    Random,
    Heuristic,
    HeuristicV2,
    Staller,
    StallerV2,
    Aggressive,
    AggressiveV2,
    SetupSweep,
    SetupSweepV2,
}

impl Kind {
    /// The inventory's display name (`eval_callback.opponent_name`).
    pub fn name(&self) -> &'static str {
        match self {
            Kind::Random => "random",
            Kind::Heuristic => "heuristic",
            Kind::HeuristicV2 => "heuristic2",
            Kind::Staller => "staller",
            Kind::StallerV2 => "staller_v2",
            Kind::Aggressive => "aggressive",
            Kind::AggressiveV2 => "aggressive_v2",
            Kind::SetupSweep => "setup_sweep",
            Kind::SetupSweepV2 => "setup_sweep_v2",
        }
    }

    /// By display name.
    pub fn from_name(name: &str) -> Option<Kind> {
        Some(match name {
            "random" => Kind::Random,
            "heuristic" => Kind::Heuristic,
            "heuristic2" => Kind::HeuristicV2,
            "staller" => Kind::Staller,
            "staller_v2" => Kind::StallerV2,
            "aggressive" => Kind::Aggressive,
            "aggressive_v2" => Kind::AggressiveV2,
            "setup_sweep" => Kind::SetupSweep,
            "setup_sweep_v2" => Kind::SetupSweepV2,
            _ => return None,
        })
    }

    /// The streams this kind draws from (the inventory's `rng` column).
    pub fn streams(&self) -> &'static [Stream] {
        match self {
            Kind::Staller | Kind::StallerV2 => &[Stream::Choice, Stream::Protect],
            _ => &[Stream::Choice],
        }
    }
}

/// A bot RNG stream (`bot_inventory` rows' `rng`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    /// `Player._choice_rng`.
    Choice,
    /// The stallers' `_protect_rng`.
    Protect,
}

impl Stream {
    pub fn name(&self) -> &'static str {
        match self {
            Stream::Choice => "choice",
            Stream::Protect => "protect",
        }
    }
}

/// One decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub order: Order,
    /// `order.message` without `/choose `.
    pub token: String,
    /// The env core's action index (`None` for `default`, which has no index).
    pub index: Option<usize>,
}

/// A bot and its streams.
#[derive(Clone)]
pub struct Bot {
    pub kind: Kind,
    pub choice: PyRandom,
    pub protect: PyRandom,
    /// The `logic.rs` source line of the return site that made the last decision (0 = RandomPlayer,
    /// whose whole policy is one site) — the gate's branch coverage.
    pub branch: u32,
}

impl Bot {
    /// Each stream seeded as `random.Random(seed)` (an unused stream is still built, never drawn).
    pub fn new(kind: Kind, choice_seed: u64, protect_seed: u64) -> Bot {
        Bot { kind, choice: PyRandom::new(choice_seed), protect: PyRandom::new(protect_seed), branch: 0 }
    }

    pub fn stream(&self, s: Stream) -> &PyRandom {
        match s {
            Stream::Choice => &self.choice,
            Stream::Protect => &self.protect,
        }
    }

    pub fn stream_mut(&mut self, s: Stream) -> &mut PyRandom {
        match s {
            Stream::Choice => &mut self.choice,
            Stream::Protect => &mut self.protect,
        }
    }

    /// The bot's order on `view` (`choose_move`).
    pub fn choose(&mut self, v: &View) -> Result<Order, BotError> {
        let br = &mut self.branch;
        *br = 0;
        match self.kind {
            Kind::Random => Ok(logic::random_move(v, &mut self.choice)),
            Kind::Heuristic => logic::heuristic(v, &mut self.choice, br),
            Kind::HeuristicV2 => logic::heuristic_v2(v, &mut self.choice, br),
            Kind::Staller => logic::staller(v, &mut self.choice, &mut self.protect, br),
            Kind::StallerV2 => logic::staller_v2(v, &mut self.choice, &mut self.protect, br),
            Kind::Aggressive => logic::aggressive(v, &mut self.choice, br),
            Kind::AggressiveV2 => logic::aggressive_v2(v, &mut self.choice, br),
            Kind::SetupSweep => logic::setup_sweep(v, &mut self.choice, br),
            Kind::SetupSweepV2 => logic::setup_sweep_v2(v, &mut self.choice, br),
        }
    }

    /// Decide at `reading`'s open decision; `tokens` are the core's `(index, token)` pairs for it
    /// (`present::choice_tokens`). An order whose token the core does not offer is refused — the
    /// Python env would have raised in `order_to_action` or fallen back to a default order.
    pub fn decide(&mut self, reading: &BoardReading, tokens: &[(usize, String)]) -> Result<Decision, BotError> {
        let v = View::build(reading)?;
        self.decide_on(&v, tokens)
    }

    /// [`Bot::decide`] on an already-built view.
    pub fn decide_on(&mut self, v: &View, tokens: &[(usize, String)]) -> Result<Decision, BotError> {
        let order = self.choose(v)?;
        let token = v.token(&order)?;
        let index = match order {
            Order::Default => None,
            _ => Some(
                tokens
                    .iter()
                    .find(|(_, t)| *t == token)
                    .map(|(i, _)| *i)
                    .ok_or_else(|| BotError(format!("the bot chose {token:?}, which the open decision does not offer ({tokens:?})")))?,
            ),
        };
        Ok(Decision { order, token, index })
    }
}
