//! THE ENCODER — the 2501-dim observation row as a method on the version (`gen3_core_encoder_v1`,
//! the Rust Core Program's M4; `designs/rust_sim/encoder.md`).
//!
//! `encode(inputs, out)` is `Gen3ObservationEncoder.encode(battle, hp_tracker, legal,
//! progress_clock, recency, pair_history, event_window)` (`agents/observation/state_encoder.py`)
//! over the core's own reading of ONE side's stream: the [`BoardReading`] (poke-env's `Battle` +
//! `Pokemon`, which the item / type / ability / move sub-encoders read RAW), the view
//! ([`present`](crate::present::present), the `LiveView` the rest reads), the legality, and the
//! side's TRACKERS ([`SideTrackers`], M3). It reproduces the Python encoder's arithmetic ORDER and
//! precision — a Python float (f64) expression evaluated in f64, then rounded ONCE to f32 at the
//! write (`numpy`'s float32 assignment) — and the gate is BYTE equality (slice O,
//! `agents/battle/rust_core_parity_obs.py`).
//!
//! **The encoder never fixes semantics.** Every value is what the reading and the trackers already
//! hold; a fact that is wrong there is fixed there (and in Python, the same day), never here.
//!
//! **Every cell is WRITTEN.** Test and fuzz builds (`debug_assertions` / `emission-selfcheck`)
//! fill the row with NaN before encoding, so a cell a branch forgot to write reads NaN and fails
//! slice O; release builds zero-fill ([`prefill`]).
//!
//! Layout: [`layout`] is GENERATED from `agents/observation/constants.py` (`python -m
//! agents.observation.rust_core_obs_layout --write`). The dex / prior tables: [`data`], read from
//! `data/pokemon/` exactly as `agents.gen3_data` reads them.

pub mod data;
pub mod layout;
mod slot;
pub mod wire;

use crate::core_error::{fault, refuse, CoreError, CoreResult, PyExc};
use crate::present::view::{MonView, SideView};
use crate::present::{BoardReading, LegalActions, OneSidedView};
use crate::trackers::history::EventRecord;
use crate::trackers::SideTrackers;

use layout::*;

pub use layout::OBS_DIM;

/// Everything one side's encode reads — all of it built from that side's stream.
pub struct Inputs<'a> {
    pub reading: &'a BoardReading,
    pub view: &'a OneSidedView,
    pub legal: Option<&'a LegalActions>,
    pub trackers: &'a SideTrackers,
}

/// Fill the row before an encode: NaN in test / fuzz builds (a cell no branch wrote stays
/// visible), 0.0 in release.
#[inline]
pub fn prefill(out: &mut [f32; OBS_DIM]) {
    #[cfg(any(debug_assertions, feature = "emission-selfcheck"))]
    out.fill(f32::NAN);
    #[cfg(not(any(debug_assertions, feature = "emission-selfcheck")))]
    out.fill(0.0);
}

/// The BLOCK a row cell belongs to and its offset inside it (`our_team[slot]+k`, `context+k`, …) —
/// how a byte gate names the first differing cell.
pub fn cell_name(i: usize) -> String {
    let team = |base: usize, name: &str| {
        let k = i - base;
        format!("{name}[slot {}]+{}", k / POKEMON_FULL_DIM, k % POKEMON_FULL_DIM)
    };
    match i {
        _ if i >= OBS_DIM => format!("cell {i} (past the row)"),
        _ if i >= OFFSET_EVENT_WINDOW => format!("event_window+{}", i - OFFSET_EVENT_WINDOW),
        _ if i >= OFFSET_PAIR_HISTORY => format!("pair_history+{}", i - OFFSET_PAIR_HISTORY),
        _ if i >= OFFSET_REACTIVE => format!("reactive+{}", i - OFFSET_REACTIVE),
        _ if i >= OFFSET_GLOBAL => format!("global+{}", i - OFFSET_GLOBAL),
        _ if i >= OFFSET_CONTEXT => format!("context+{}", i - OFFSET_CONTEXT),
        _ if i >= OFFSET_OPP_TEAM => team(OFFSET_OPP_TEAM, "opp_team"),
        _ => team(OFFSET_OUR_TEAM, "our_team"),
    }
}

/// Is this build's prefill the NaN poison? (Reported by the binaries so a gate can check which
/// build it ran.)
pub const NAN_POISON: bool = cfg!(any(debug_assertions, feature = "emission-selfcheck"));

/// A Python `float` written into a float32 cell: one round-to-nearest.
#[inline]
pub(crate) fn put(out: &mut [f32], i: usize, v: f64) {
    out[i] = v as f32;
}

#[inline]
pub(crate) fn zero(out: &mut [f32], lo: usize, n: usize) {
    out[lo..lo + n].fill(0.0);
}

/// A refusal the Python encoder raises with `exc` on the same input.
pub(crate) fn raise(exc: PyExc, msg: impl Into<String>) -> CoreError {
    refuse(exc, msg)
}

/// `LiveSide.get(species)` — the FIRST view mon of that species.
pub(crate) fn live_get<'a>(side: &'a SideView, species: &str) -> Option<&'a MonView> {
    side.mons.iter().find(|m| m.species == species)
}

/// Encode `inputs` into `out` (prefilled first). The whole row, every cell written.
pub fn encode(inp: &Inputs, out: &mut [f32; OBS_DIM]) -> CoreResult<()> {
    prefill(out);
    encode_into(inp, out)
}

/// A checked entry for a caller-owned slice: REFUSED unless it is exactly `OBS_DIM` long.
pub fn encode_slice(inp: &Inputs, out: &mut [f32]) -> CoreResult<()> {
    let len = out.len();
    let row: &mut [f32; OBS_DIM] = out
        .try_into()
        .map_err(|_| fault(format!("encode: the row has {len} cells, the observation has {OBS_DIM}")))?;
    encode(inp, row)
}

fn encode_into(inp: &Inputs, out: &mut [f32; OBS_DIM]) -> CoreResult<()> {
    let t = data::tables();
    // 1 + 2. the two teams' per-mon slots
    slot::team(inp, t, true, out)?;
    slot::team(inp, t, false, out)?;
    // 3. the active contexts
    let ours = inp.view.ours.active.map(|i| &inp.view.ours.mons[i]);
    let opp = inp.view.opp.active.map(|i| &inp.view.opp.mons[i]);
    active_context(ours, &mut out[OFFSET_CONTEXT..OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM])?;
    active_context(opp, &mut out[OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM..OFFSET_CONTEXT + 2 * ACTIVE_CONTEXT_DIM])?;
    // 4. the global environment
    global_env(inp.view, &mut out[OFFSET_GLOBAL..OFFSET_GLOBAL + GLOBAL_ENV_DIM]);
    // 5. the board (reactive) block
    board(inp, t, &mut out[OFFSET_REACTIVE..OFFSET_REACTIVE + REACTIVE_DIM])?;
    // 6. the pair history
    pair_history(inp, &mut out[OFFSET_PAIR_HISTORY..OFFSET_PAIR_HISTORY + PAIR_HISTORY_DIM]);
    // 7. the event window
    event_window(inp, t, &mut out[OFFSET_EVENT_WINDOW..OFFSET_EVENT_WINDOW + EVENT_WINDOW_DIM])?;
    Ok(())
}

// ---------------------------------------------------------------------------- active context

/// `gen3_effects.encode_volatiles` — raises `UnknownVolatileError` on an unclassified id.
fn volatiles(vols: &[(String, u32)], out: &mut [f32]) -> CoreResult<()> {
    let mut v = [0.0f64; VOLATILES_DIM];
    for (vid, _) in vols {
        match VOLATILE_TO_SLOT.binary_search_by(|(k, _, _)| (*k).cmp(vid.as_str())) {
            Ok(i) => {
                let (_, slot, value) = VOLATILE_TO_SLOT[i];
                // `vec[idx] = max(vec[idx], value)` on a float32 cell: compare the stored f32 (as
                // f64) with the f64 value, keep the larger, round once.
                let cur = v[slot] as f32 as f64;
                v[slot] = if value > cur { value } else { cur };
            }
            Err(_) if NOT_A_VOLATILE.binary_search(&vid.as_str()).is_ok() => {}
            Err(_) => {
                return Err(raise(
                    PyExc::UnknownVolatileError,
                    format!("volatile {vid:?} has no gen3 encoding slot (UnknownVolatileError)"),
                ))
            }
        }
    }
    for (i, x) in v.iter().enumerate() {
        put(out, i, *x);
    }
    Ok(())
}

/// `ActiveContextEncoder.encode(live_mon)`: 14 boost dims (positive / negative magnitude per
/// stat) + the volatile block; all zeros without an active mon.
fn active_context(m: Option<&MonView>, out: &mut [f32]) -> CoreResult<()> {
    let Some(m) = m else {
        zero(out, 0, ACTIVE_CONTEXT_DIM);
        return Ok(());
    };
    for (k, stat) in BOOST_ORDER.iter().enumerate() {
        let stage = m.boosts.iter().find(|(s, _)| s == stat).map_or(0, |(_, v)| *v) as i64;
        put(out, 2 * k, stage.max(0) as f64 / 6.0);
        put(out, 2 * k + 1, (-stage).max(0) as f64 / 6.0);
    }
    volatiles(&m.volatiles, &mut out[BOOSTS_DIM..BOOSTS_DIM + VOLATILES_DIM])
}

// ---------------------------------------------------------------------------- global env

/// `GlobalEnvEncoder.encode(live_view)`.
fn global_env(v: &OneSidedView, out: &mut [f32]) {
    zero(out, 0, GLOBAL_ENV_DIM);
    let w = &v.weather;
    let widx = w.weather.as_deref().and_then(|id| WEATHER_IDX.iter().find(|(k, _)| *k == id)).map_or(0, |(_, i)| *i);
    out[widx] = 1.0;
    let mut cur = WEATHER_ONEHOT_DIM;
    put(out, cur, if w.is_permanent { 1.0 } else { 0.0 });
    // `LiveWeather.turns_remaining`: None when absent or permanent, else max(0, 5 - turns_active).
    let tr = if w.weather.is_none() || w.is_permanent { None } else { Some((5i64 - w.turns_active as i64).max(0)) };
    put(out, cur + 1, tr.map_or(0.0, |t| t as f64 / WEATHER_MAX_TURNS));
    cur += 2;
    let spikes = |s: &SideView| s.side_conditions.iter().find(|(k, _)| k == "spikes").map_or(0, |(_, n)| *n);
    put(out, cur, spikes(&v.ours) as f64 / MAX_SPIKES as f64);
    put(out, cur + 1, spikes(&v.opp) as f64 / MAX_SPIKES as f64);
    cur += 2;
    let turn = v.turn as f64;
    let remaining = (MAX_TURNS as f64 - turn).max(0.0);
    put(out, cur, (1.0 + turn).ln() / LOG_MAX_TURNS);
    put(out, cur + 1, remaining / MAX_TURNS as f64);
    put(out, cur + 2, (1.0 + remaining).ln() / LOG_MAX_TURNS);
    cur += CLOCK_DIM;
    for cid in SCREEN_CONDITIONS {
        let has = |s: &SideView| s.side_conditions.iter().any(|(k, _)| k == cid);
        put(out, cur, if has(&v.ours) { 1.0 } else { 0.0 });
        put(out, cur + 1, if has(&v.opp) { 1.0 } else { 0.0 });
        cur += 2;
    }
}

// ---------------------------------------------------------------------------- the board block

/// `ReactiveEncoder.encode(battle, live, legal, progress_clock, wish_pending)`.
fn board(inp: &Inputs, t: &data::Tables, out: &mut [f32]) -> CoreResult<()> {
    zero(out, 0, REACTIVE_DIM);
    let ar = ACTIVE_REQ_MOVES_OFFSET;
    let per = ACTIVE_REQ_MOVES_PER;
    let forced_struggle = match inp.legal {
        Some(l) => l.struggle,
        // the plain-Battle fallback (`available_moves == [struggle]`) — never the trainee's path
        None => inp.reading.available_moves.len() == 1 && inp.reading.available_moves[0] == "struggle",
    };
    if !forced_struggle {
        for (i, mv) in request_slot_moves(inp)?.into_iter().enumerate() {
            if i >= 4 {
                break;
            }
            let Some(mid) = mv else { continue };
            if let Some(md) = t.moves.get(&mid) {
                put(out, ar + i, md.num as f64);
                let ty = if mid == "hiddenpower" { 0 } else { md.type_idx };
                put(out, ar + per + i, ty as f64);
            }
            let legal_now = match inp.legal {
                Some(l) => i < l.move_slots.len() && !l.move_slots[i].disabled,
                None => true,
            };
            put(out, ar + 2 * per + i, if legal_now { 1.0 } else { 0.0 });
        }
    }
    let fainted = |s: &SideView| s.mons.iter().filter(|m| m.fainted).count() as f64 / 6.0;
    put(out, 0, fainted(&inp.view.ours));
    put(out, 1, fainted(&inp.view.opp));
    put(out, 2, inp.trackers.clock.value());
    let wish = |p: bool| if p { WISH_HEAL_FRACTION } else { 0.0 };
    put(out, 3, wish(inp.trackers.wish_pending[0]));
    put(out, 4, wish(inp.trackers.wish_pending[1]));
    Ok(())
}

/// `reactive._request_slot_moves(battle, legal)` → per request slot, the `Move.id` of the move the
/// ACTIVE mon's moveset holds under that slot's request id (a bare `hiddenpower` resolves to the
/// moveset's single `hiddenpower*` key), `None` when unresolved.
fn request_slot_moves(inp: &Inputs) -> CoreResult<Vec<Option<String>>> {
    let active = inp.reading.active_index(true).map(|i| &inp.reading.team[i].1);
    let (Some(legal), Some(active)) = (inp.legal, active) else {
        // `list(battle.available_moves)[:4]` — only reached with no active mon or no legality; a
        // decision always has both, so this is a FAULT rather than a guess at poke-env's list.
        if inp.reading.available_moves.is_empty() {
            return Ok(Vec::new());
        }
        return Err(fault("encode: available moves with no active mon / no legality (unmirrored path)"));
    };
    let moveset = active.moves.moves();
    let mut out = Vec::new();
    for lm in legal.move_slots.iter().take(4) {
        let mut mv = moveset.iter().find(|(k, _)| *k == lm.id).map(|(_, m)| m.id.clone());
        if mv.is_none() && lm.id == "hiddenpower" {
            let hps: Vec<&String> = moveset.iter().filter(|(k, _)| k.starts_with("hiddenpower")).map(|(_, m)| &m.id).collect();
            mv = if hps.len() == 1 { Some(hps[0].clone()) } else { None };
        }
        out.push(mv);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------- pair history

/// `PairHistoryTracker.pair_values(opp_species, our_species)` into 5 cells.
fn pair_cell(p: &crate::trackers::history::PairHistory, opp: Option<&str>, ours: Option<&str>, out: &mut [f32]) {
    let (Some(i), Some(j)) = (opp.filter(|s| !s.is_empty()), ours.filter(|s| !s.is_empty())) else {
        out[..4].fill(0.0);
        out[4] = 1.0;
        return;
    };
    let key = (i.to_string(), j.to_string());
    let cap = PAIR_SAT as i64;
    let rec = match p.shared_last_turn.get(&key) {
        None => 1.0,
        Some(last) => {
            let d = p.turn - last;
            SAT_LUT[if 0 < d && d < cap { d } else if d <= 0 { 0 } else { cap } as usize]
        }
    };
    for (k, m) in [&p.switch_ins, &p.attacks, &p.status_clicks, &p.shared_count].into_iter().enumerate() {
        let n = *m.get(&key).unwrap_or(&0);
        put(out, k, SAT_LUT[if n < cap { n.max(0) } else { cap } as usize]);
    }
    put(out, 4, rec);
}

/// The 6×6×5 block, (opp slot, our slot, cell) row-major, joined by the team-list order.
fn pair_history(inp: &Inputs, out: &mut [f32]) {
    let ph = &inp.trackers.pair;
    fn sp(team: &[(String, crate::present::mon::PMon)], k: usize) -> Option<&str> {
        team.get(k).map(|(_, m)| m.species.as_str())
    }
    for i in 0..TEAM_SIZE {
        for j in 0..TEAM_SIZE {
            let o = (i * TEAM_SIZE + j) * PAIR_HISTORY_CELL_DIM;
            pair_cell(ph, sp(&inp.reading.opp, i), sp(&inp.reading.team, j), &mut out[o..o + PAIR_HISTORY_CELL_DIM]);
        }
    }
}

// ---------------------------------------------------------------------------- the event window

/// `gen3_effects.cant_reason_id` — 1 + the index into `CANT_REASONS_LIVE`, 0 for `None`; raises
/// `UnknownCantReasonError` on an unrecognised reason.
fn cant_reason_id(reason: Option<&str>) -> CoreResult<usize> {
    let Some(raw) = reason else { return Ok(0) };
    let lower = raw.to_lowercase();
    let body = if lower.starts_with("move:") || lower.starts_with("ability:") {
        raw.split_once(':').map_or(raw, |(_, b)| b)
    } else {
        raw
    };
    let rid: String = body.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect();
    CANT_REASONS_LIVE
        .iter()
        .position(|r| *r == rid)
        .map(|i| i + 1)
        .ok_or_else(|| raise(PyExc::UnknownCantReasonError, format!("cant reason {raw:?} (id {rid:?}) is not a known gen3 cause")))
}

/// `turn_view.faint_cause_id`.
fn faint_cause_id(cause: Option<&str>) -> CoreResult<usize> {
    let Some(c) = cause else { return Ok(0) };
    FAINT_CAUSE_VOCAB
        .iter()
        .position(|v| *v == c)
        .map(|i| i + 1)
        .ok_or_else(|| raise(PyExc::ValueError, format!("unknown faint cause {c:?}")))
}

/// `assembler.write_event_row(vec, o, rec, cur_turn)` — one 22-column row.
fn event_row(t: &data::Tables, r: &EventRecord, cur_turn: i64, out: &mut [f32]) -> CoreResult<()> {
    zero(out, 0, EVENT_TOKEN_DIM);
    let species_num = |s: &Option<String>| {
        s.as_deref().filter(|x| !x.is_empty()).and_then(|x| t.species.get(x)).map_or(0.0, |e| e.num)
    };
    put(out, EV_TYPE, r.t as f64);
    put(out, EV_ACTOR_SPECIES, species_num(&r.actor));
    put(
        out,
        EV_ACTOR_SIDE,
        match r.side {
            Some(crate::core_events::Rel::Ours) => 1.0,
            Some(crate::core_events::Rel::Opp) => -1.0,
            None => 0.0,
        },
    );
    put(out, EV_TARGET_SPECIES, species_num(&r.target));
    let mv = r.move_id.as_deref().filter(|m| !m.is_empty()).and_then(|m| t.moves.get(m)).map_or(0.0, |m| m.num as f64);
    put(out, EV_MOVE, mv);
    let is_move = r.t as usize == EVENT_T_MOVE;
    // `max(-1.0, min(1.0, x))`, Python's argument order (the first extreme wins a tie).
    let clip = |x: f64| {
        let m = if x < 1.0 { x } else { 1.0 };
        if m > -1.0 {
            m
        } else {
            -1.0
        }
    };
    // MOVE: the attributed hp fraction; BOOST: the SIGNED stage change / 6; HAZARD: +1 a side
    // condition started, −1 one ended (`gen3_event_window_semantics_fixes_v1`) — only BOOST is scaled.
    let is_boost = r.t as usize == EVENT_T_BOOST;
    put(out, EV_MAGNITUDE, if is_boost { clip(r.hp_delta / 6.0) } else { clip(r.hp_delta) });
    if is_move {
        put(out, EV_OUT_HIT, if r.missed || r.failed { 0.0 } else { 1.0 });
        put(out, EV_OUT_MISS, if r.missed { 1.0 } else { 0.0 });
        put(out, EV_OUT_FAIL, if r.failed { 1.0 } else { 0.0 });
        put(out, EV_CRIT, if r.crit { 1.0 } else { 0.0 });
        out[EV_EFF_NEUTRAL + r.eff as usize] = 1.0;
    }
    put(out, EV_WE_FIRST, if r.we_first { 1.0 } else { 0.0 });
    put(out, EV_STATUS, r.status as f64);
    let cant = match &r.cant {
        Some(reason) => cant_reason_id(reason.as_deref())?,
        None => 0,
    };
    put(out, EV_CANT, cant as f64);
    put(out, EV_FAINT_CAUSE, faint_cause_id(r.faint_cause)? as f64);
    put(out, EV_ITEM_TRANSITION, r.item_tr.unwrap_or(0) as f64);
    let ago = (cur_turn - r.turn).max(0).min(SAT_LUT.len() as i64 - 1) as usize;
    put(out, EV_TURNS_AGO, SAT_LUT[ago]);
    put(out, EV_FORCED_WINDOW, r.forced_window);
    put(out, EV_VALID, 1.0);
    Ok(())
}

/// The last `EVENT_WINDOW_N` records, oldest first, zero rows padding the FRONT.
fn event_window(inp: &Inputs, t: &data::Tables, out: &mut [f32]) -> CoreResult<()> {
    let w = &inp.trackers.window;
    let rows: Vec<&EventRecord> = w.window().collect();
    let rows = &rows[rows.len().saturating_sub(EVENT_WINDOW_N)..];
    let pad = EVENT_WINDOW_N - rows.len();
    zero(out, 0, pad * EVENT_TOKEN_DIM);
    for (k, r) in rows.iter().enumerate() {
        let o = (pad + k) * EVENT_TOKEN_DIM;
        event_row(t, r, w.turn, &mut out[o..o + EVENT_TOKEN_DIM])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
