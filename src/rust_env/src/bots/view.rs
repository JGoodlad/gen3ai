//! THE BOT VIEW — the one input a Rust bot reads, built from the env core's `BoardReading` for the
//! bot's side (M5 Lane F).
//!
//! WHICH STATE A BOT READS (the lane's decision): the core's OWN per-side reading — no second
//! tracker. A Python bot reads a poke-env `Battle` (in training `env.battle2`), and
//! `pokesim::present::BoardReading` is the port of exactly that object, folded from exactly the
//! lines that side is shipped (program §6c). [`View::build`] reads every attribute any bot reads
//! and nothing else; [`View::render`] writes it as the canonical string
//! `src/utils/rust_env/bot_view.py` writes from the Python battle, and the gate compares the two
//! hashes at every banked decision before it compares the action.

use pokesim::present::board_reading::BoardReading;
use pokesim::present::dex::{self as pdex, Status};
use pokesim::present::mon::{PMon, PMove, BOOST_KEYS, STAT_KEYS};

use super::tables::{MoveRow, MOVES};
use super::BotError;

/// A `Move` as a bot reads it.
#[derive(Clone, Debug)]
pub struct MoveV {
    pub id: String,
    pub bp: u32,
    pub typ: &'static str,
    pub cat: &'static str,
    pub acc: f64,
    pub hits: f64,
    pub target: Option<&'static str>,
    pub boosts: Option<&'static [(&'static str, i32)]>,
    pub last: bool,
}

/// A `Pokemon` as a bot reads it.
#[derive(Clone, Debug)]
pub struct MonV {
    pub species: String,
    pub name: Option<String>,
    pub t1: &'static str,
    pub t2: Option<&'static str>,
    pub types: Vec<&'static str>,
    /// hp, atk, def, spa, spd, spe.
    pub base: [u16; 6],
    /// poke-env's boost key order ([`BOOST_KEYS`]).
    pub boosts: [i32; 7],
    pub hp: f64,
    pub status: Option<Status>,
    pub ability: Option<String>,
    pub fainted: bool,
    pub stats: [Option<i64>; 6],
    pub moves: Vec<MoveV>,
}

/// One entry of `battle.valid_orders`.
#[derive(Clone, Debug, PartialEq)]
pub enum Order {
    /// A `SingleBattleOrder(Move)` — an index into `View::moves`.
    Move(usize),
    /// A `SingleBattleOrder(Pokemon)` — an index into `View::team`.
    Switch(usize),
    /// `DefaultBattleOrder`.
    Default,
}

/// Every battle fact a bot reads (see `bot_view.py` for the field list and who reads each).
#[derive(Clone, Debug)]
pub struct View {
    pub wait: bool,
    pub trapped: bool,
    pub force_switch: bool,
    pub team: Vec<MonV>,
    pub active: Option<usize>,
    pub opp_active: Option<MonV>,
    pub opp_fainted: usize,
    pub moves: Vec<MoveV>,
    pub switches: Vec<usize>,
    pub side: Vec<&'static str>,
    pub opp_side: Vec<&'static str>,
}

fn row(id: &str) -> Option<&'static MoveRow> {
    MOVES.binary_search_by(|r| r.id.cmp(id)).ok().map(|i| &MOVES[i])
}

impl MoveV {
    /// `Move(id)`'s properties (`Move.entry` resolution: the id, then a `z`-stripped id).
    pub fn of(m: &PMove) -> Result<MoveV, BotError> {
        let r = row(&m.id)
            .or_else(|| m.id.strip_prefix('z').and_then(row))
            .ok_or_else(|| BotError(format!("Unknown move: {} (ValueError)", m.id)))?;
        Ok(MoveV {
            id: m.id.clone(),
            bp: m.base_power_override.unwrap_or(r.base_power),
            typ: r.typ,
            cat: r.category,
            acc: r.accuracy,
            hits: r.expected_hits,
            target: r.target,
            boosts: r.boosts,
            last: m.last_used,
        })
    }
}

fn status_name(s: Status) -> &'static str {
    match s {
        Status::Brn => "BRN",
        Status::Fnt => "FNT",
        Status::Frz => "FRZ",
        Status::Par => "PAR",
        Status::Psn => "PSN",
        Status::Slp => "SLP",
        Status::Tox => "TOX",
    }
}

impl MonV {
    pub fn of(p: &PMon) -> Result<MonV, BotError> {
        let types = p.types();
        let t1 = *types.first().ok_or_else(|| BotError(format!("{}: no types", p.species)))?;
        Ok(MonV {
            species: p.species.clone(),
            name: p.name.clone(),
            t1,
            t2: types.get(1).copied(),
            types,
            base: p.base_stats(),
            boosts: p.boosts,
            hp: p.hp_fraction(),
            status: p.status,
            ability: p.ability().map(str::to_string),
            fainted: p.fainted(),
            stats: p.stats,
            moves: p.moves.moves_ref().into_iter().map(|(_, m)| MoveV::of(m)).collect::<Result<_, _>>()?,
        })
    }

    /// `base_stats[key]`.
    pub fn base_stat(&self, key: &str) -> f64 {
        self.base[STAT_KEYS.iter().position(|k| *k == key).expect("stat key")] as f64
    }

    /// `boosts[key]`.
    pub fn boost(&self, key: &str) -> i32 {
        self.boosts[BOOST_KEYS.iter().position(|k| *k == key).expect("boost key")]
    }

    /// `stats[key]` of an own mon (`None` would be Python's `TypeError` on the comparison).
    pub fn stat(&self, key: &str) -> Result<i64, BotError> {
        self.stats[STAT_KEYS.iter().position(|k| *k == key).expect("stat key")]
            .ok_or_else(|| BotError(format!("{}.stats[{key}] is None (TypeError)", self.species)))
    }

    /// `Pokemon.last_move` — the first move of `moves.values()` flagged last-used.
    pub fn last_move(&self) -> Option<&MoveV> {
        self.moves.iter().find(|m| m.last)
    }
}

/// `Pokemon.available_moves_from_request`'s objects: the moveset's own `Move` for a known key (a
/// bare `hiddenpower` resolving to the single typed one), else a fresh `Move(id)`.
fn available_move(mon: &PMon, id: &str) -> Result<MoveV, BotError> {
    let moves = mon.moves.moves_ref();
    if let Some((_, m)) = moves.iter().find(|(k, _)| *k == id) {
        return MoveV::of(m);
    }
    if !pdex::is_special_move(id) && id == "hiddenpower" {
        let hps: Vec<&&PMove> = moves.iter().filter(|(k, _)| k.starts_with("hiddenpower")).map(|(_, m)| m).collect();
        if hps.len() == 1 {
            return MoveV::of(hps[0]);
        }
    }
    let fresh = PMove::new(id, None, false).map_err(|e| BotError(format!("Move({id}): {e}")))?;
    MoveV::of(&fresh)
}

impl View {
    /// The view of `r` (the bot's side's reading at its open decision).
    pub fn build(r: &BoardReading) -> Result<View, BotError> {
        let team = r.team.iter().map(|(_, m)| MonV::of(m)).collect::<Result<Vec<_>, _>>()?;
        let active = r.active_index(true);
        let moves = match active {
            Some(ai) => r.available_moves.iter().map(|id| available_move(&r.team[ai].1, id)).collect::<Result<_, _>>()?,
            None => Vec::new(),
        };
        Ok(View {
            wait: r.wait,
            trapped: r.trapped,
            force_switch: r.force_switch,
            team,
            active,
            opp_active: r.active_index(false).map(|i| MonV::of(&r.opp[i].1)).transpose()?,
            opp_fainted: r.opp.iter().filter(|(_, m)| m.fainted()).count(),
            moves,
            switches: r.available_switches.clone(),
            side: r.side_conditions[0].iter().map(|(n, _)| *n).collect(),
            opp_side: r.side_conditions[1].iter().map(|(n, _)| *n).collect(),
        })
    }

    /// `battle.active_pokemon`.
    pub fn active_mon(&self) -> Option<&MonV> {
        self.active.map(|i| &self.team[i])
    }

    /// `battle.valid_orders` (gen 3: no mega / z / dynamax / tera orders).
    pub fn valid_orders(&self) -> Vec<Order> {
        if self.wait {
            return vec![Order::Default];
        }
        let mut out = Vec::new();
        if !self.trapped {
            out.extend(self.switches.iter().map(|&i| Order::Switch(i)));
        }
        if self.active.is_none() || self.force_switch {
            return out;
        }
        out.extend((0..self.moves.len()).map(Order::Move));
        out
    }

    /// `order.message` without the `/choose ` prefix — the choice token.
    pub fn token(&self, o: &Order) -> Result<String, BotError> {
        Ok(match o {
            Order::Default => "default".to_string(),
            Order::Move(i) => {
                let m = &self.moves[*i];
                if m.id == "recharge" {
                    "move 1".to_string()
                } else {
                    format!("move {}", m.id)
                }
            }
            Order::Switch(i) => {
                let name = self.team[*i].name.as_deref().ok_or_else(|| BotError(format!("team slot {i} has no name")))?;
                format!("switch {name}")
            }
        })
    }

    /// The canonical string (`bot_view.view_json`).
    pub fn render(&self) -> Result<String, BotError> {
        let mut o = String::with_capacity(4096);
        o.push_str(&format!("{{\"wait\":{},\"trapped\":{},\"force_switch\":{},\"team\":[", self.wait, self.trapped, self.force_switch));
        for (i, m) in self.team.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            mon_into(&mut o, m);
        }
        o.push_str("],\"active\":");
        match self.active {
            Some(i) => o.push_str(&i.to_string()),
            None => o.push_str("null"),
        }
        o.push_str(",\"opp_active\":");
        match &self.opp_active {
            Some(m) => mon_into(&mut o, m),
            None => o.push_str("null"),
        }
        o.push_str(&format!(",\"opp_fainted\":{},\"moves\":[", self.opp_fainted));
        for (i, m) in self.moves.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            move_into(&mut o, m);
        }
        o.push_str("],\"switches\":[");
        o.push_str(&self.switches.iter().map(usize::to_string).collect::<Vec<_>>().join(","));
        o.push_str("],\"side\":[");
        list_into(&mut o, &self.side);
        o.push_str("],\"opp_side\":[");
        list_into(&mut o, &self.opp_side);
        o.push_str("],\"valid\":[");
        for (i, v) in self.valid_orders().iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            str_into(&mut o, &format!("/choose {}", self.token(v)?));
        }
        o.push_str("]}");
        Ok(o)
    }
}

/// `bot_view.fbits`.
pub fn fbits(x: f64) -> String {
    format!("f:{:016x}", x.to_bits())
}

/// `bot_view._s` — `"` and `\` backslashed, a control char as `\u00XX`, the rest raw.
pub fn str_into(o: &mut String, s: &str) {
    o.push('"');
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                o.push('\\');
                o.push(c);
            }
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
}

fn opt_into(o: &mut String, s: Option<&str>) {
    match s {
        Some(s) => str_into(o, s),
        None => o.push_str("null"),
    }
}

fn list_into(o: &mut String, xs: &[&str]) {
    for (i, x) in xs.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        str_into(o, x);
    }
}

fn move_into(o: &mut String, m: &MoveV) {
    o.push_str("{\"id\":");
    str_into(o, &m.id);
    o.push_str(&format!(",\"bp\":{},\"type\":", m.bp));
    str_into(o, m.typ);
    o.push_str(",\"cat\":");
    str_into(o, m.cat);
    o.push_str(",\"acc\":");
    str_into(o, &fbits(m.acc));
    o.push_str(",\"hits\":");
    str_into(o, &fbits(m.hits));
    o.push_str(",\"target\":");
    opt_into(o, m.target);
    o.push_str(",\"boosts\":");
    match m.boosts {
        None => o.push_str("null"),
        Some(b) => {
            o.push('[');
            for (i, (k, v)) in b.iter().enumerate() {
                if i > 0 {
                    o.push(',');
                }
                o.push('[');
                str_into(o, k);
                o.push_str(&format!(",{v}]"));
            }
            o.push(']');
        }
    }
    o.push_str(&format!(",\"last\":{}}}", m.last));
}

fn mon_into(o: &mut String, m: &MonV) {
    o.push_str("{\"species\":");
    str_into(o, &m.species);
    o.push_str(",\"t1\":");
    str_into(o, m.t1);
    o.push_str(",\"t2\":");
    opt_into(o, m.t2);
    o.push_str(",\"types\":[");
    list_into(o, &m.types);
    o.push_str("],\"base\":[");
    o.push_str(&m.base.iter().map(u16::to_string).collect::<Vec<_>>().join(","));
    o.push_str("],\"boosts\":[");
    o.push_str(&m.boosts.iter().map(i32::to_string).collect::<Vec<_>>().join(","));
    o.push_str("],\"hp\":");
    str_into(o, &fbits(m.hp));
    o.push_str(",\"status\":");
    opt_into(o, m.status.map(status_name));
    o.push_str(",\"ability\":");
    opt_into(o, m.ability.as_deref());
    o.push_str(&format!(",\"fainted\":{},\"stats\":[", m.fainted));
    o.push_str(&m.stats.iter().map(|s| s.map_or("null".to_string(), |v| v.to_string())).collect::<Vec<_>>().join(","));
    o.push_str("],\"moves\":[");
    for (i, mv) in m.moves.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        move_into(o, mv);
    }
    o.push_str("]}");
}

/// FNV-1a-64 of the UTF-8 bytes (`bot_view.fnv64`).
pub fn fnv64(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}
