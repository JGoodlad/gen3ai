//! [`OneSidedView`] and [`present`] — `LiveView.from_battle` over a [`Tracker`], field by field,
//! with the TRUE reading wherever poke-env's is wrong about a sim fact.
//!
//! The view is `agents.battle.live_view.LiveView` in Rust: the same fields, the same spellings,
//! the same ORDER (our mons in `battle.team` order — the first request's roster; theirs in reveal
//! order), so the Python side builds its `LiveView` from [`OneSidedView::json`] with a transport
//! that applies NO rule (`agents.battle.core_view`).

use super::dex;
use super::mon::{PMon, BOOST_KEYS, STAT_KEYS};
use super::tracker::Tracker;
use crate::core_events::json_out;

// 🚨 **`present()` is the TRUE reading** (owner directive, M2): where poke-env's reading is wrong
// about a sim fact the stream can establish, the view carries the truth and the disagreement is a
// named FINDING (`designs/rust_sim/present.md` §3, `agents/battle/poke_env_findings.py`) — never a
// rule that reproduces the mistake. Today: PE-V10 (a fainted mon's stages), PE-R1b (a
// badly-poisoned mon's stage), PE-V16 (Flash Fire after its holder's Fire move). A truth only the
// omniscient board knows cannot reach the view by construction ([`present`] takes the side's
// stream and nothing else): our benched mons' PP after an un-announced Pressure (V15) and a
// Transformed own mon's copied ability (R4) are INFORMATION LIMITS, read as poke-env reads them.

/// `LiveMove`, plus the `Move.id` its key maps to (the D2 encoders sort on that).
#[derive(Debug, Clone, PartialEq)]
pub struct MoveView {
    /// The poke-env moves-dict KEY (`LiveMove.id`).
    pub id: String,
    /// `Move.id` — differs from the key for a typed Hidden Power.
    pub move_id: String,
    pub current_pp: u32,
    pub max_pp: u32,
}

/// `LivePokemon`.
#[derive(Debug, Clone, PartialEq)]
pub struct MonView {
    pub species: String,
    pub active: bool,
    pub fainted: bool,
    pub revealed: bool,
    pub hp_fraction: f64,
    pub status: Option<&'static str>,
    /// Lower-cased `PokemonType` names.
    pub types: Vec<String>,
    pub moves: Vec<MoveView>,
    pub item: Option<String>,
    pub ability: Option<String>,
    /// NONZERO stages, in `Pokemon._boosts` key order.
    pub boosts: Vec<(&'static str, i32)>,
    /// `{_id(effect): counter}` in `_effects` order.
    pub volatiles: Vec<(String, u32)>,
    pub base_stats: [u16; 6],
    pub ivs: Option<Vec<i64>>,
    pub evs: Option<Vec<i64>>,
    pub nature: Option<String>,
    pub spread_known: bool,
    pub consumed_item: Option<String>,
    pub status_counter: u32,
    pub protect_counter: u32,
    /// `dict(mon.stats)` — hp, atk, def, spa, spd, spe (`None` = unknown).
    pub stats: [Option<i64>; 6],
    pub current_hp: u32,
    pub max_hp: u32,
}

/// `LiveSide`.
#[derive(Debug, Clone, PartialEq)]
pub struct SideView {
    pub team_size: usize,
    /// Index into `mons` of the active mon.
    pub active: Option<usize>,
    pub mons: Vec<MonView>,
    /// `{SideCondition name lower-cased: value}`, dict order.
    pub side_conditions: Vec<(String, u32)>,
}

/// `LiveWeather`.
#[derive(Debug, Clone, PartialEq)]
pub struct WeatherView {
    pub weather: Option<String>,
    pub is_permanent: bool,
    pub turns_active: u32,
}

/// `LiveView` — one side's reading of the current board.
#[derive(Debug, Clone, PartialEq)]
pub struct OneSidedView {
    pub turn: u32,
    pub weather: WeatherView,
    pub ours: SideView,
    pub opp: SideView,
    pub finished: bool,
    pub won: Option<bool>,
    pub lost: Option<bool>,
}

type R<T> = Result<T, String>;

/// `present(tracker)` — `LiveView.from_battle` over ONE side's stream, with the true reading where
/// poke-env's is wrong (see above).
///
/// **The signature is the wall.** The only input is the side's stream state (its typed events and
/// its `|request|` snapshots, folded into the [`Tracker`]); there is no board parameter, so a fact
/// only the referee knows cannot reach the view. The step path's typed shortcut and the parse
/// path fold the same lines into the same tracker, which is why they cannot disagree, and the
/// board's only role is to CHECK the result ([`super::audit::check_view`]).
pub fn present(tracker: &Tracker) -> R<OneSidedView> {
    let role = tracker.role as usize;
    let sizes = [tracker.team_size(0), tracker.team_size(1)];
    let side = |own: bool| -> R<SideView> {
        let team = if own { &tracker.team } else { &tracker.opp };
        let act = tracker.active_index(own);
        let declared = if own { role } else { 1 - role };
        let mut mons = Vec::with_capacity(team.len());
        for (i, (_, m)) in team.iter().enumerate() {
            mons.push(mon_view(m, act == Some(i), own)?);
        }
        let conds = &tracker.side_conditions[if own { 0 } else { 1 }];
        Ok(SideView {
            team_size: sizes[declared].unwrap_or(team.len()),
            active: act,
            mons,
            side_conditions: conds.iter().map(|(n, v)| (n.to_ascii_lowercase(), *v)).collect(),
        })
    };
    let w = &tracker.weather;
    Ok(OneSidedView {
        turn: tracker.turn,
        weather: WeatherView {
            weather: w.id.clone(),
            is_permanent: w.id.is_some() && w.permanent,
            turns_active: if w.id.is_some() { tracker.turn.saturating_sub(w.start_turn) } else { 0 },
        },
        ours: side(true)?,
        opp: side(false)?,
        finished: tracker.finished,
        won: tracker.won,
        lost: tracker.won.map(|w| !w),
    })
}

/// `LivePokemon.from_pokemon(mon, active, is_own)`.
fn mon_view(m: &PMon, active: bool, own: bool) -> R<MonView> {
    let mut moves: Vec<MoveView> = m
        .moves
        .moves()
        .into_iter()
        .map(|(k, mv)| Ok(MoveView { id: k, move_id: mv.id.clone(), current_pp: mv.current_pp, max_pp: mv.max_pp()? }))
        .collect::<R<Vec<_>>>()?;
    moves.sort_by(|a, b| a.id.cmp(&b.id));
    let item = match m.item.as_deref() {
        None => None,
        Some(i) if i == super::tables::UNKNOWN_ITEM => None,
        Some(i) => Some(i.to_string()),
    };
    let mut vols: Vec<(String, u32)> = Vec::new();
    for (e, c) in &m.effects {
        let id = dex::effect_live_id(*e);
        match vols.iter_mut().find(|(k, _)| *k == id) {
            Some(slot) => slot.1 = *c,
            None => vols.push((id, *c)),
        }
    }
    Ok(MonView {
        species: m.species.clone(),
        active,
        fainted: m.fainted(),
        revealed: m.revealed,
        hp_fraction: m.hp_fraction(),
        status: m.status.map(|s| s.live()),
        types: m.types().iter().map(|t| t.to_ascii_lowercase()).collect(),
        moves,
        item,
        ability: m.ability().map(str::to_string),
        boosts: BOOST_KEYS.iter().zip(m.boosts.iter()).filter(|(_, v)| **v != 0).map(|(k, v)| (*k, *v)).collect(),
        volatiles: vols,
        base_stats: m.base_stats(),
        ivs: if own { m.ivs.clone() } else { None },
        evs: if own { m.evs.clone() } else { None },
        nature: if own { m.nature.clone() } else { None },
        spread_known: own,
        consumed_item: m.consumed_item.as_deref().filter(|c| !c.is_empty()).map(dex::to_id),
        status_counter: m.status_counter,
        protect_counter: m.protect_counter,
        stats: m.stats,
        current_hp: m.current_hp(),
        max_hp: m.max_hp(),
    })
}

// ---------------------------------------------------------------- JSON

impl OneSidedView {
    /// The view as JSON in `LiveView`'s own shape (`agents.battle.core_view` reads it with no
    /// rule). Floats are Rust's shortest round-trip form, so `hp_fraction` reads back bit-equal.
    pub fn json(&self) -> String {
        let mut o = String::with_capacity(8192);
        o.push_str(&format!("{{\"turn\":{},\"weather\":{{\"weather\":", self.turn));
        json_out::opt_str_into(&mut o, self.weather.weather.as_deref());
        o.push_str(&format!(
            ",\"is_permanent\":{},\"turns_active\":{}}},\"finished\":{},\"won\":{},\"lost\":{},\"ours\":",
            self.weather.is_permanent,
            self.weather.turns_active,
            self.finished,
            opt_bool(self.won),
            opt_bool(self.lost)
        ));
        side_json(&mut o, &self.ours);
        o.push_str(",\"opp\":");
        side_json(&mut o, &self.opp);
        o.push('}');
        o
    }
}

fn opt_bool(b: Option<bool>) -> &'static str {
    match b {
        None => "null",
        Some(true) => "true",
        Some(false) => "false",
    }
}

fn side_json(o: &mut String, s: &SideView) {
    o.push_str(&format!("{{\"team_size\":{},\"active\":", s.team_size));
    match s.active {
        Some(i) => json_out::str_into(o, &s.mons[i].species),
        None => o.push_str("null"),
    }
    o.push_str(",\"side_conditions\":{");
    for (i, (k, v)) in s.side_conditions.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        json_out::str_into(o, k);
        o.push_str(&format!(":{v}"));
    }
    o.push_str("},\"mons\":[");
    for (i, m) in s.mons.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        mon_json(o, m);
    }
    o.push_str("]}");
}

fn int_list(o: &mut String, v: &Option<Vec<i64>>) {
    match v {
        None => o.push_str("null"),
        Some(xs) => {
            o.push('[');
            o.push_str(&xs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            o.push(']');
        }
    }
}

fn mon_json(o: &mut String, m: &MonView) {
    o.push_str("{\"species\":");
    json_out::str_into(o, &m.species);
    o.push_str(&format!(",\"active\":{},\"fainted\":{},\"revealed\":{},\"hp_fraction\":", m.active, m.fainted, m.revealed));
    json_out::f64_into(o, m.hp_fraction);
    o.push_str(",\"status\":");
    json_out::opt_str_into(o, m.status);
    o.push_str(",\"types\":[");
    for (i, t) in m.types.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        json_out::str_into(o, t);
    }
    o.push_str("],\"moves\":[");
    for (i, mv) in m.moves.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str("{\"id\":");
        json_out::str_into(o, &mv.id);
        o.push_str(",\"move_id\":");
        json_out::str_into(o, &mv.move_id);
        o.push_str(&format!(",\"current_pp\":{},\"max_pp\":{}}}", mv.current_pp, mv.max_pp));
    }
    o.push_str("],\"item\":");
    json_out::opt_str_into(o, m.item.as_deref());
    o.push_str(",\"ability\":");
    json_out::opt_str_into(o, m.ability.as_deref());
    o.push_str(",\"boosts\":{");
    for (i, (k, v)) in m.boosts.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&format!("\"{k}\":{v}"));
    }
    o.push_str("},\"volatiles\":{");
    for (i, (k, v)) in m.volatiles.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        json_out::str_into(o, k);
        o.push_str(&format!(":{v}"));
    }
    o.push_str("},\"base_stats\":{");
    for (i, k) in STAT_KEYS.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&format!("\"{k}\":{}", m.base_stats[i]));
    }
    o.push_str("},\"ivs\":");
    int_list(o, &m.ivs);
    o.push_str(",\"evs\":");
    int_list(o, &m.evs);
    o.push_str(",\"nature\":");
    json_out::opt_str_into(o, m.nature.as_deref());
    o.push_str(&format!(",\"spread_known\":{},\"consumed_item\":", m.spread_known));
    json_out::opt_str_into(o, m.consumed_item.as_deref());
    o.push_str(&format!(
        ",\"status_counter\":{},\"protect_counter\":{},\"stats\":{{",
        m.status_counter, m.protect_counter
    ));
    for (i, k) in STAT_KEYS.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        match m.stats[i] {
            Some(v) => o.push_str(&format!("\"{k}\":{v}")),
            None => o.push_str(&format!("\"{k}\":null")),
        }
    }
    o.push_str(&format!("}},\"current_hp\":{},\"max_hp\":{}}}", m.current_hp, m.max_hp));
}
