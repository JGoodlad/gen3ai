//! [`Reader`] — the READING projection: what ONE side's `Gen3Battle` records, line by line.
//!
//! This is `agents.battle.gen3_battle.Gen3Battle` (its `_capture_pre` + `_build_event` +
//! `_move_suffix_events` + `record_choice_rejected`) over the FIVE facts of poke-env's state those
//! methods read — a mon's species, its HP fraction, its status, which mon is active on a side, and
//! which side's move is resolving — plus the turn. Each fact's transitions mirror poke-env's own
//! (`AbstractBattle.parse_message`, `Battle.switch`, `Pokemon.set_hp_status` / `faint` /
//! `cure_status` / `_update_from_details`, `_update_team_from_request`), cited at each branch. The
//! reader sees exactly the lines the side's `Player` routes to the battle, in order (requests
//! included — they write our own team), so its readings are that battle's events verbatim.
//!
//! # The reading rules — where the reading is NOT the simulator's truth
//!
//! Every one is named, applied here, and pinned by a unit test to the poke-env line it mirrors.
//! The core carries the TRUTH beside them (the typed line keeps every `[from]`/`[of]`; the source
//! record keeps the engine scope and exact HP), so a retrain can flip any rule without re-deriving.
//!
//! | rule | the reading | poke-env / gen3ai line |
//! |---|---|---|
//! | R1 | an outcome line's side is the last `\|move\|` line's, reset only at `\|turn\|` | `abstract_battle.py:711` (set), `:1634` (reset); `gen3_battle.py:716,733` |
//! | R2 | an effectiveness line with no open move is owned by the side opposite the defender | `gen3_battle.py:735-738` |
//! | R3 | a MISS/FAIL/CRIT's target is the mon NAMED at index 2 — for `-miss` that is the USER | `gen3_battle.py:717,727` |
//! | R4 | a `[still]` / empty-target move targets the OTHER side's active | `gen3_battle.py:532-539` |
//! | R5 | `\|move\|…\|[miss]` / `[notarget]` adds a second, synthetic MISS / FAIL (`from="move-suffix"`) | `gen3_battle.py:272-274,488-502` |
//! | R6 | HP is the viewer's rendering (own exact, foe `ceil%`) | `Pokemon.current_hp_fraction`; `bridge.rs::hp_percent` |
//! | R7 | an effectiveness event carries only its multiplier (`[from] ability:` dropped) | `gen3_battle.py:744-749` |
//! | R8 | `-cureteam`'s `status` is its field 3 verbatim (gen 3: the `[from] move: Aromatherapy` clause) | `gen3_battle.py:687-691` |
//! | R9 | a DAMAGE/HEAL/SETHP event drops the line's `[of]` source (Recoil, Leech Seed, drain) | `gen3_battle.py:642-652` (`value` has no `of`) |
//! | R10 | a mon named by a line but never introduced is created with `species = to_id(name)` (the NICKNAME) | `abstract_battle.py:421-424` |
//! | R11 | a `-formechange` / `detailschange` never renames the species a later event names | `pokemon.py:469-471` (`store_species=False`) |

use super::line::Line;
use super::schema::{EventKind, Kw};
use super::{to_id, Intercept, Reading, Rel, Route, Value};
use crate::core_error::{fault, malformed, refuse, CoreResult, PyExc};

/// poke-env's `Status` member names.
fn status_name(tok: &str) -> CoreResult<&'static str> {
    Ok(match tok.to_ascii_lowercase().as_str() {
        "brn" => "BRN",
        "fnt" => "FNT",
        "frz" => "FRZ",
        "par" => "PAR",
        "psn" => "PSN",
        "slp" => "SLP",
        "tox" => "TOX",
        other => return Err(refuse(PyExc::KeyError, format!("Status[{other:?}] is not a poke-env status (KeyError)"))),
    })
}

#[derive(Debug, Clone)]
struct Mon {
    species: String,
    /// `Pokemon._current_hp` (`None` until an HP is set).
    cur: Option<u32>,
    max: Option<u32>,
    status: Option<&'static str>,
    active: bool,
    last_details: Option<String>,
}

impl Mon {
    fn new(species: String) -> Mon {
        Mon { species, cur: None, max: None, status: None, active: false, last_details: None }
    }
    /// `Pokemon.current_hp_fraction`: `current_hp / max_hp` if `current_hp` else `0`.
    fn hp_fraction(&self) -> f64 {
        match (self.cur, self.max) {
            (Some(c), Some(m)) if c != 0 => c as f64 / m as f64,
            _ => 0.0,
        }
    }
    /// `Pokemon.faint`.
    fn faint(&mut self) {
        self.cur = Some(0);
        self.status = Some("FNT");
    }
    /// `Pokemon.set_hp_status`.
    fn set_hp_status(&mut self, text: &str) -> CoreResult<()> {
        if text == "0 fnt" {
            self.faint();
            return Ok(());
        }
        let hp = if let Some((hp, st)) = text.split_once(' ') {
            if st.contains(' ') {
                return Err(refuse(PyExc::ValueError, format!("set_hp_status({text:?}): too many values to unpack")));
            }
            self.status = Some(status_name(st)?);
            hp
        } else {
            self.status = None;
            text
        };
        let digits: String = hp.chars().filter(|c| c.is_ascii_digit() || *c == '/').collect();
        let (c, m) = digits.split_once('/').ok_or_else(|| refuse(PyExc::ValueError, format!("set_hp_status({text:?}): no '/' (ValueError)")))?;
        if m.contains('/') {
            return Err(refuse(PyExc::ValueError, format!("set_hp_status({text:?}): too many values to unpack")));
        }
        self.cur = Some(c.parse().map_err(|_| refuse(PyExc::ValueError, format!("set_hp_status({text:?}): bad hp (ValueError)")))?);
        self.max = Some(m.parse().map_err(|_| refuse(PyExc::ValueError, format!("set_hp_status({text:?}): bad max hp (ValueError)")))?);
        Ok(())
    }
    /// `Pokemon._update_from_details` — only the species matters here.
    fn update_from_details(&mut self, details: &str) {
        if self.last_details.as_deref() == Some(details) {
            return;
        }
        self.last_details = Some(details.to_string());
        let d = details.replace(", shiny", "");
        let head = d.split(", ").next().unwrap_or("");
        self.species = to_id(head);
    }
}

/// Is `tok` an identifier by `gen3_battle._is_ident`'s test (`p1…` / `p2…`, length >= 2)?
fn is_ident(tok: &str) -> bool {
    let b = tok.as_bytes();
    b.len() >= 2 && b[0] == b'p' && (b[1] == b'1' || b[1] == b'2')
}

/// `AbstractBattle.get_pokemon`'s key normalisation: `p1a: X` -> `p1: X`.
fn norm_key(tok: &str) -> CoreResult<String> {
    let b = tok.as_bytes();
    if b.len() < 4 {
        return Err(refuse(PyExc::IndexError, format!("get_pokemon({tok:?}): identifier too short (IndexError)")));
    }
    if b[3] != b' ' {
        Ok(format!("{}{}", &tok[..2], &tok[3..]))
    } else {
        Ok(tok.to_string())
    }
}

/// The per-viewer reading fold.
#[derive(Debug, Clone)]
pub struct Reader {
    /// The viewer's role: 0 = p1, 1 = p2.
    viewer: u8,
    turn: u32,
    seq: u32,
    /// R1's state — poke-env's `_current_move_user_side`, as an absolute side.
    mover: Option<u8>,
    /// `_team` (the viewer's) and `_opponent_team`, each insertion-ordered by key `pN: Name`.
    teams: [Vec<(String, Mon)>; 2],
    team_size: [Option<usize>; 2],
}

impl Reader {
    pub fn new(viewer: usize) -> Reader {
        Reader { viewer: viewer as u8, turn: 0, seq: 0, mover: None, teams: [Vec::new(), Vec::new()], team_size: [None, None] }
    }

    /// Which `teams` index holds a side's mons.
    fn team_of(&self, side: u8) -> usize {
        if side == self.viewer {
            0
        } else {
            1
        }
    }

    fn rel(&self, side: u8) -> Rel {
        if side == self.viewer {
            Rel::Ours
        } else {
            Rel::Opp
        }
    }

    fn find(&self, key: &str) -> Option<(usize, usize)> {
        for t in 0..2 {
            if let Some(i) = self.teams[t].iter().position(|(k, _)| k == key) {
                return Some((t, i));
            }
        }
        None
    }

    /// `AbstractBattle.get_pokemon(identifier, details=…)`, creating the mon when unknown.
    fn get(&mut self, tok: &str, details: Option<&str>, force_self: bool) -> CoreResult<(usize, usize)> {
        let key = norm_key(tok)?;
        if let Some(at) = self.find(&key) {
            return Ok(at);
        }
        let side = if key.as_bytes()[1] == b'1' { 0 } else { 1 };
        let t = if force_self { 0 } else { self.team_of(side) };
        if let Some(n) = self.team_size[side as usize] {
            if self.teams[t].len() >= n {
                return Err(refuse(PyExc::ValueError, format!("get_pokemon({tok:?}): team already has {n} pokemons (ValueError)")));
            }
        }
        // R10: with details the species comes from them; without, from the identifier's NAME.
        let mut mon = Mon::new(to_id(key.get(4..).unwrap_or("")));
        if let Some(d) = details {
            mon.update_from_details(d);
        }
        self.teams[t].push((key, mon));
        Ok((t, self.teams[t].len() - 1))
    }

    fn mon(&mut self, tok: &str) -> CoreResult<&mut Mon> {
        let (t, i) = self.get(tok, None, false)?;
        Ok(&mut self.teams[t][i].1)
    }

    /// The side's active mon (`Battle.active_pokemon` / `opponent_active_pokemon`: the FIRST mon
    /// flagged active, in insertion order).
    fn active(&self, rel: Rel) -> Option<&Mon> {
        let t = if rel == Rel::Ours { 0 } else { 1 };
        self.teams[t].iter().map(|(_, m)| m).find(|m| m.active)
    }

    fn active_species(&self, rel: Option<Rel>) -> Option<String> {
        rel.and_then(|r| self.active(r)).map(|m| m.species.clone())
    }

    /// `Gen3Battle._side_of`.
    fn side_of(&self, tok: &str) -> Option<Rel> {
        if !is_ident(tok) {
            return None;
        }
        Some(self.rel(tok.as_bytes()[1] - b'1'))
    }

    /// `Gen3Battle._species_of` (creates, like `get_pokemon`).
    fn species_of(&mut self, tok: &str) -> CoreResult<Option<String>> {
        if !is_ident(tok) {
            return Ok(None);
        }
        Ok(Some(self.mon(tok)?.species.clone()))
    }

    fn hp_fraction(&mut self, tok: &str) -> CoreResult<f64> {
        Ok(self.mon(tok)?.hp_fraction())
    }

    fn new_reading(&mut self, kind: EventKind, sm: &[String], side: Option<Rel>, actor: Option<String>,
                   target: Option<String>, value: Vec<(&'static str, Value)>) -> Reading {
        let r = Reading { seq: self.seq, turn: self.turn, kind, side, actor, target, value, raw: sm.to_vec() };
        self.seq += 1;
        r
    }

    fn from_ident(&mut self, kind: EventKind, sm: &[String], tok: Option<&str>,
                  value: Vec<(&'static str, Value)>) -> CoreResult<Reading> {
        let side = tok.and_then(|t| self.side_of(t));
        let actor = match tok {
            Some(t) => self.species_of(t)?,
            None => None,
        };
        Ok(self.new_reading(kind, sm, side, actor, None, value))
    }

    /// Feed ONE line of this side's stream; return the readings it produced, in order.
    pub fn feed(&mut self, line: &Line) -> CoreResult<Vec<Reading>> {
        match line.kw.route() {
            Route::Plain => Ok(Vec::new()),
            Route::Intercept(i) => self.intercept(i, line),
            Route::Unsupported => Err(refuse(PyExc::UnsupportedMessageType, format!("UnsupportedMessageType: {:?}", line.kw.as_str()))),
            Route::Control | Route::Cosmetic | Route::StateOnly => {
                self.apply(line)?;
                Ok(Vec::new())
            }
            Route::Event(kind) => self.event(kind, line),
        }
    }

    fn intercept(&mut self, i: Intercept, line: &Line) -> CoreResult<Vec<Reading>> {
        let sm = line.split_message();
        match i {
            Intercept::Ignored | Intercept::BigError | Intercept::Win | Intercept::Tie => Ok(Vec::new()),
            Intercept::ShowTeam => Err(malformed("|showteam| is not a gen-3 line")),
            Intercept::Request => {
                if sm.len() > 2 && !sm[2].is_empty() {
                    self.request(&sm[2..].join("|"))?;
                }
                Ok(Vec::new())
            }
            Intercept::Error => {
                // `Player._handle_battle_message`: `[Unavailable choice]` -> the out-of-band
                // `record_choice_rejected` hook (`gen3_battle.py:350-379`); anything else writes
                // nothing to the battle.
                if sm.len() > 2 && sm[2].starts_with("[Unavailable choice]") {
                    let actor = self.active_species(Some(Rel::Ours));
                    let reason = sm[2].clone();
                    let r = self.new_reading(EventKind::ChoiceRejected, &sm, Some(Rel::Ours), actor, None,
                                             vec![("reason", Value::Str(reason))]);
                    return Ok(vec![r]);
                }
                Ok(Vec::new())
            }
        }
    }

    /// `Battle.parse_request` -> `_update_team_from_request`: our side's roster, HP/status (the
    /// `condition`), the active flag and the details-species, for every mon in the request.
    fn request(&mut self, json: &str) -> CoreResult<()> {
        use super::jsonval::Val;
        let v = Val::parse(json).map_err(|e| malformed(format!("request JSON: {e}")))?;
        let Some(Val::Arr(mons)) = v.get("side").and_then(|s| s.get("pokemon")) else {
            return Err(refuse(PyExc::KeyError, "request without side.pokemon"));
        };
        for p in mons {
            let ident = p.str_at("ident").ok_or_else(|| refuse(PyExc::KeyError, "request mon without ident"))?;
            let details = p.str_at("details").unwrap_or("");
            let condition = p.str_at("condition").unwrap_or("");
            let active = matches!(p.get("active"), Some(Val::Bool(true)));
            let key = norm_key(ident)?;
            let (t, i) = match self.find(&key) {
                Some(at) => at,
                None => self.get(ident, Some(details), true)?,
            };
            let m = &mut self.teams[t][i].1;
            // `update_from_request`: `_active`, `set_hp_status(condition)`, `_update_from_details`.
            m.active = active;
            m.set_hp_status(condition)?;
            m.update_from_details(details);
        }
        Ok(())
    }

    /// The state half of `AbstractBattle.parse_message` — only the five facts.
    fn apply(&mut self, line: &Line) -> CoreResult<()> {
        let sm = line.split_message();
        let f = |i: usize| -> &str { sm.get(i).map(String::as_str).unwrap_or("") };
        match line.kw {
            Kw::Turn => {
                // `end_turn`: the new turn number, and R1's reset of the move owner.
                self.turn = f(2).parse().map_err(|_| refuse(PyExc::ValueError, format!("|turn|{}: int() (ValueError)", f(2))))?;
                self.mover = None;
            }
            Kw::Teamsize => {
                let side = if f(2) == "p1" { 0 } else { 1 };
                self.team_size[side] = f(3).parse().ok();
            }
            Kw::Switch | Kw::Drag => {
                // `Battle.switch`: switch the side's current active OUT, get/create the entrant
                // with its details, `switch_in(details)`, `set_hp_status(hp)`.
                let tok = f(2).to_string();
                let side = if tok.starts_with("p1") { 0 } else { 1 };
                let rel = self.rel(side);
                let t = if rel == Rel::Ours { 0 } else { 1 };
                if let Some(m) = self.teams[t].iter_mut().map(|(_, m)| m).find(|m| m.active) {
                    m.active = false;
                }
                let details = f(3).to_string();
                let hp = f(4).to_string();
                let (t, i) = self.get(&tok, Some(&details), false)?;
                let m = &mut self.teams[t][i].1;
                m.active = true;
                m.update_from_details(&details);
                m.set_hp_status(&hp)?;
            }
            Kw::Damage | Kw::Heal | Kw::Sethp => {
                let hp = f(3).to_string();
                self.mon(&f(2).to_string())?.set_hp_status(&hp)?;
            }
            Kw::Faint => self.mon(&f(2).to_string())?.faint(),
            Kw::Status => {
                let st = status_name(f(3))?;
                self.mon(&f(2).to_string())?.status = Some(st);
            }
            Kw::Curestatus => {
                // `Pokemon.cure_status(status)`: only when it IS the current status (and an
                // empty status token is falsy: nothing happens).
                if !f(3).is_empty() {
                    let st = status_name(f(3))?;
                    let m = self.mon(&f(2).to_string())?;
                    if m.status == Some(st) {
                        m.status = None;
                    }
                }
            }
            Kw::Cureteam => {
                // `team.cure_status()` over the NAMED side's team; a fainted mon keeps FNT.
                let tok = f(2);
                let side = if tok.starts_with("p1") { 0 } else { 1 };
                let t = self.team_of(side);
                for (_, m) in self.teams[t].iter_mut() {
                    if m.status != Some("FNT") {
                        m.status = None;
                    }
                }
            }
            Kw::Move => {
                // R1: this move now owns every outcome line until the next `|move|` or `|turn|`.
                let tok = f(2).to_string();
                self.mover = Some(if tok.starts_with("p1") { 0 } else { 1 });
                self.mon(&tok)?;
            }
            Kw::Replace | Kw::Swap => {
                return Err(malformed(format!("|{}| (Illusion / position swap) cannot occur in gen-3 singles", line.kw.as_str())));
            }
            // R11: `forme_change` keeps the species; nothing else here moves the five facts.
            _ => {}
        }
        Ok(())
    }

    fn event(&mut self, kind: EventKind, line: &Line) -> CoreResult<Vec<Reading>> {
        let sm = line.split_message();
        let get = |i: usize| -> Option<&str> { sm.get(i).map(String::as_str) };
        // ---- `_capture_pre`: the facts this line is about to overwrite ----
        let mut pre_target: Option<Option<String>> = None;
        let mut pre_status: Option<&'static str> = None;
        let mut hp_before: Option<f64> = None;
        match line.kw {
            Kw::Move => {
                let actor = get(2).unwrap_or("");
                let side = self.side_of(actor);
                // R4: an explicit identifier target, else the OTHER side's active.
                let tmon: Option<&Mon> = match get(4) {
                    Some(t) if is_ident(t) => {
                        let t = t.to_string();
                        let m = self.mon(&t)?;
                        Some(&*m)
                    }
                    _ => self.active(if side == Some(Rel::Ours) { Rel::Opp } else { Rel::Ours }),
                };
                pre_target = Some(tmon.map(|m| m.species.clone()));
                pre_status = tmon.and_then(|m| m.status);
            }
            Kw::Damage | Kw::Heal | Kw::Sethp => {
                hp_before = Some(self.hp_fraction(get(2).unwrap_or(""))?);
            }
            _ => {}
        }
        // ---- poke-env mutates ----
        self.apply(line)?;
        // ---- `_build_event` ----
        let from_last = || -> Option<String> {
            // `_parse_from`: the LAST `[from]` token wins, stripped.
            sm.iter().skip(3).rev().map(|t| t.trim()).find_map(|t| t.strip_prefix("[from]").map(|r| r.trim().to_string()))
        };
        let of_last = || -> Option<String> {
            sm.iter().skip(3).rev().map(|t| t.trim()).find_map(|t| t.strip_prefix("[of]").map(|r| r.trim().to_string()))
        };
        let mut out = Vec::new();
        match kind {
            EventKind::Move => {
                let move_id = to_id(get(3).unwrap_or(""));
                let mut value = vec![("move_id", Value::Str(move_id.clone())),
                                     ("target_status", pre_status.map(|s| Value::Str(s.into())).unwrap_or(Value::Null))];
                if let Some(src) = delegated_from(&sm, &move_id) {
                    value.push(("from_move", Value::Str(src)));
                }
                let actor_tok = get(2).map(str::to_string);
                let side = actor_tok.as_deref().and_then(|t| self.side_of(t));
                let actor = match actor_tok.as_deref() {
                    Some(t) => self.species_of(t)?,
                    None => None,
                };
                out.push(self.new_reading(kind, &sm, side, actor, pre_target.flatten(), value));
                // R5: the move-suffix synthetics, after the MOVE event.
                let toks: Vec<&str> = sm.iter().skip(3).map(String::as_str).collect();
                if toks.contains(&"[miss]") {
                    out.push(self.from_ident(EventKind::Miss, &sm, actor_tok.as_deref(),
                                             vec![("from", Value::Str("move-suffix".into()))])?);
                }
                if toks.contains(&"[notarget]") {
                    out.push(self.from_ident(EventKind::Fail, &sm, actor_tok.as_deref(),
                                             vec![("from", Value::Str("move-suffix".into()))])?);
                }
            }
            EventKind::Switch | EventKind::Drag | EventKind::Faint | EventKind::Prepare
            | EventKind::Mustrecharge | EventKind::Swap => {
                out.push(self.from_ident(kind, &sm, get(2), Vec::new())?);
            }
            EventKind::Damage | EventKind::Heal => {
                // R6 (the viewer's HP) + R9 (`[of]` dropped).
                let tok = get(2).unwrap_or("").to_string();
                let after = self.hp_fraction(&tok)?;
                let before = hp_before.unwrap_or(after);
                let mut value = vec![("amount", Value::Float(after - before)), ("hp_after", Value::Float(after))];
                if let Some(r) = from_last() {
                    value.push(("reason", Value::Str(r)));
                }
                out.push(self.from_ident(kind, &sm, Some(&tok), value)?);
            }
            EventKind::Sethp => {
                let tok = get(2).unwrap_or("").to_string();
                let after = self.hp_fraction(&tok)?;
                let before = hp_before.unwrap_or(after);
                let mut value = vec![("hp", Value::Float(after)), ("amount", Value::Float(after - before))];
                if let Some(r) = from_last() {
                    value.push(("reason", Value::Str(r)));
                }
                out.push(self.from_ident(kind, &sm, Some(&tok), value)?);
            }
            EventKind::Boost | EventKind::Unboost | EventKind::Setboost => {
                let stat = get(3).map(|s| Value::Str(s.into())).unwrap_or(Value::Null);
                let raw_amt: i64 = match get(4) {
                    Some(a) if !a.trim_start_matches('-').is_empty()
                        && a.trim_start_matches('-').chars().all(|c| c.is_ascii_digit()) => a.parse().unwrap_or(0),
                    _ => 0,
                };
                let signed = if kind == EventKind::Unboost { -raw_amt } else { raw_amt };
                out.push(self.from_ident(kind, &sm, get(2), vec![("stat", stat), ("amount", Value::Int(signed))])?);
            }
            EventKind::Clearboost => {
                let tok = get(2).filter(|t| is_ident(t));
                let op = line.kw.as_str().trim_start_matches('-').to_string();
                out.push(self.from_ident(kind, &sm, tok, vec![("op", Value::Str(op))])?);
            }
            EventKind::Status => {
                let mut value = vec![("status", get(3).map(|s| Value::Str(s.into())).unwrap_or(Value::Null))];
                if let Some(r) = from_last() {
                    value.push(("reason", Value::Str(r)));
                }
                out.push(self.from_ident(kind, &sm, get(2), value)?);
            }
            EventKind::Curestatus => {
                // R8: field 3 verbatim, whatever it is.
                let op = line.kw.as_str().trim_start_matches('-').to_string();
                let value = vec![("status", get(3).map(|s| Value::Str(s.into())).unwrap_or(Value::Null)),
                                 ("op", Value::Str(op))];
                out.push(self.from_ident(kind, &sm, get(2), value)?);
            }
            EventKind::Cant => {
                // `gen3_damp_cant_v1`: filed on the HOLDER; `[of]` names the mon that lost its turn.
                let of = of_last();
                let of_side = of.as_deref().and_then(|t| self.side_of(t));
                let of_actor = match of.as_deref() {
                    Some(t) => self.species_of(t)?,
                    None => None,
                };
                let value = vec![
                    ("reason", get(3).map(|s| Value::Str(s.into())).unwrap_or(Value::Null)),
                    ("move", get(4).map(|s| Value::Str(to_id(s))).unwrap_or(Value::Null)),
                    ("of", of.clone().map(Value::Str).unwrap_or(Value::Null)),
                    ("of_side", of_side.map(|r| Value::Str(r.as_str().into())).unwrap_or(Value::Null)),
                    ("of_actor", of_actor.map(Value::Str).unwrap_or(Value::Null)),
                ];
                out.push(self.from_ident(kind, &sm, get(2), value)?);
            }
            EventKind::Crit | EventKind::Miss | EventKind::Fail => {
                // R1 (owner = the open move's side) + R3 (target = the NAMED mon).
                let mover = self.mover.map(|s| self.rel(s));
                let target_tok = get(2).filter(|t| is_ident(t)).map(str::to_string);
                // `({"from": cause} if cause else None)` — an EMPTY clause is falsy: no key.
                let cause = sm.iter().skip(3).find(|t| t.starts_with("[from]")).map(|t| t[6..].trim().to_string())
                    .filter(|c| !c.is_empty());
                let actor = self.active_species(mover);
                let target = match target_tok.as_deref() {
                    Some(t) => self.species_of(t)?,
                    None => None,
                };
                let value = cause.map(|c| vec![("from", Value::Str(c))]).unwrap_or_default();
                out.push(self.new_reading(kind, &sm, mover, actor, target, value));
            }
            EventKind::Immune | EventKind::Resisted | EventKind::Supereffective => {
                // R1 + R2 (no open move: the side opposite the defender) + R7 (multiplier only).
                let mut mover = self.mover.map(|s| self.rel(s));
                let def_tok = get(2).filter(|t| is_ident(t)).map(str::to_string);
                if mover.is_none() {
                    if let Some(d) = def_tok.as_deref() {
                        mover = self.side_of(d).map(|r| if r == Rel::Ours { Rel::Opp } else { Rel::Ours });
                    }
                }
                let mult = match kind {
                    EventKind::Immune => 0.0,
                    EventKind::Resisted => 0.5,
                    _ => 2.0,
                };
                let actor = self.active_species(mover);
                let target = match def_tok.as_deref() {
                    Some(t) => self.species_of(t)?,
                    None => None,
                };
                out.push(self.new_reading(kind, &sm, mover, actor, target, vec![("multiplier", Value::Float(mult))]));
            }
            EventKind::Item | EventKind::Enditem => {
                let mut value = vec![("item", get(3).map(|s| Value::Str(to_id(s))).unwrap_or(Value::Null))];
                push_cause(&mut value, &sm);
                out.push(self.from_ident(kind, &sm, get(2), value)?);
            }
            EventKind::Ability => {
                let op = if line.kw == Kw::Endability { "end" } else { "reveal" };
                let mut value = vec![("ability", get(3).map(|s| Value::Str(to_id(s))).unwrap_or(Value::Null)),
                                     ("op", Value::Str(op.into()))];
                push_cause(&mut value, &sm);
                out.push(self.from_ident(kind, &sm, get(2), value)?);
            }
            EventKind::Weather => {
                let mut value = vec![("weather", get(2).map(|s| Value::Str(to_id(s))).unwrap_or(Value::Null))];
                push_cause(&mut value, &sm);
                out.push(self.new_reading(kind, &sm, None, None, None, value));
            }
            EventKind::Field => {
                let op = line.kw.as_str().trim_start_matches('-').to_string();
                let value = vec![("effect", get(2).map(|s| Value::Str(s.into())).unwrap_or(Value::Null)),
                                 ("op", Value::Str(op))];
                out.push(self.new_reading(kind, &sm, None, None, None, value));
            }
            EventKind::Side => {
                let side_tok = get(2).unwrap_or("").to_string();
                let op = line.kw.as_str().trim_start_matches('-').to_string();
                let cond = get(3).unwrap_or(&side_tok).to_string();
                let value = vec![("condition", Value::Str(cond)), ("op", Value::Str(op))];
                let side = self.side_of(&side_tok);
                out.push(self.new_reading(kind, &sm, side, None, None, value));
            }
            EventKind::VolatileStart | EventKind::VolatileEnd => {
                let op = line.kw.as_str().trim_start_matches('-').to_string();
                let mut value = vec![("effect", get(3).map(|s| Value::Str(s.into())).unwrap_or(Value::Null)),
                                     ("op", Value::Str(op))];
                push_cause(&mut value, &sm);
                out.push(self.from_ident(kind, &sm, get(2), value)?);
            }
            EventKind::Activate => {
                let mut value = vec![("effect", get(3).map(|s| Value::Str(s.into())).unwrap_or(Value::Null))];
                push_cause(&mut value, &sm);
                out.push(self.from_ident(kind, &sm, get(2), value)?);
            }
            EventKind::Transform => {
                // `_from_ident(…, target=self._species_of(target_ident))` — the target argument is
                // evaluated FIRST (a created mon's insertion order).
                let target_tok = get(3).filter(|t| is_ident(t)).map(str::to_string);
                let target = match target_tok.as_deref() {
                    Some(t) => self.species_of(t)?,
                    None => None,
                };
                let mut r = self.from_ident(kind, &sm, get(2), Vec::new())?;
                r.target = target;
                out.push(r);
            }
            EventKind::Formechange => {
                let op = line.kw.as_str().trim_start_matches('-').to_string();
                out.push(self.from_ident(kind, &sm, get(2), vec![("op", Value::Str(op))])?);
            }
            EventKind::ChoiceRejected | EventKind::Unknown => {
                return Err(fault(format!("{:?} is never a parsed keyword's kind", kind)));
            }
        }
        for r in &out {
            r.check_schema()?;
        }
        Ok(out)
    }
}

/// `**cause` in `_build_event`: the LAST `[from]` as `from` and the LAST `[of]` as `of`, in the
/// order `_parse_from` first inserts them.
fn push_cause(value: &mut Vec<(&'static str, Value)>, sm: &[String]) {
    let mut from: Option<String> = None;
    let mut of: Option<String> = None;
    let mut order: Vec<&'static str> = Vec::new();
    for tok in sm.iter().skip(3) {
        let t = tok.trim();
        if let Some(r) = t.strip_prefix("[from]") {
            if from.is_none() {
                order.push("from");
            }
            from = Some(r.trim().to_string());
        } else if let Some(r) = t.strip_prefix("[of]") {
            if of.is_none() {
                order.push("of");
            }
            of = Some(r.trim().to_string());
        }
    }
    for k in order {
        let v = if k == "from" { from.clone() } else { of.clone() };
        value.push((k, Value::Str(v.unwrap_or_default())));
    }
}

/// `Gen3Battle._delegated_from` over `battle_event.from_clause_move_source`: the FIRST `[from]`
/// token that is not an item/ability cause, `move:` stripped, id-normalised — unless it is the
/// executed move itself (Pursuit's self-tag) or the `lockedmove` marker.
fn delegated_from(sm: &[String], executed: &str) -> Option<String> {
    for tok in sm.iter().skip(3) {
        let t = tok.trim();
        let Some(src) = t.strip_prefix("[from]") else { continue };
        let mut src = src.trim().to_string();
        let low = src.to_lowercase();
        if low.starts_with("item:") || low.starts_with("ability:") {
            continue;
        }
        if low.starts_with("move:") {
            src = src.splitn(2, ':').nth(1).unwrap_or("").trim().to_string();
        }
        let sid = to_id(&src);
        if !sid.is_empty() {
            return (sid != executed && sid != "lockedmove").then_some(sid);
        }
    }
    None
}

/// Fold a whole side stream (typed lines, in order) into its readings.
pub fn read_all(viewer: usize, lines: &[Line]) -> CoreResult<Vec<Vec<Reading>>> {
    let mut r = Reader::new(viewer);
    lines.iter().map(|l| r.feed(l)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREFIX: &[&str] = &[
        "|player|p1|me||",
        "|player|p2|foe||",
        "|teamsize|p1|2",
        "|teamsize|p2|2",
        "|start",
        "|switch|p1a: Metagross|Metagross|301/301",
        "|switch|p2a: Gyarados|Gyarados, M|100/100",
        "|turn|1",
    ];

    fn run(lines: &[&str]) -> Vec<Reading> {
        let mut r = Reader::new(0);
        let mut out = Vec::new();
        for l in PREFIX.iter().chain(lines.iter()) {
            out.extend(r.feed(&Line::parse(l).unwrap()).unwrap());
        }
        out
    }

    fn last(lines: &[&str], kind: EventKind) -> Reading {
        run(lines).into_iter().filter(|e| e.kind == kind).last().expect("reading of that kind")
    }

    /// R1 — `abstract_battle.py:711` sets the owner at `|move|`, `:1634` (`end_turn`) resets it.
    #[test]
    fn r1_the_outcome_owner_is_the_last_move_until_turn() {
        let e = last(&["|move|p1a: Metagross|Meteor Mash|p2a: Gyarados", "|-crit|p2a: Gyarados"], EventKind::Crit);
        assert_eq!((e.side, e.actor.as_deref()), (Some(Rel::Ours), Some("metagross")));
        let e = last(&["|move|p1a: Metagross|Meteor Mash|p2a: Gyarados", "|turn|2",
                       "|switch|p2a: Salamence|Salamence, M|100/100",
                       "|-fail|p1a: Metagross|unboost|[from] ability: Clear Body|[of] p1a: Metagross"], EventKind::Fail);
        assert_eq!((e.side, e.actor.as_deref()), (None, None), "reset at |turn|");
    }

    /// R2 — `gen3_battle.py:735-738`.
    #[test]
    fn r2_an_effectiveness_line_with_no_open_move_is_the_defenders_foes() {
        let e = last(&["|-immune|p2a: Gyarados"], EventKind::Immune);
        assert_eq!((e.side, e.actor.as_deref()), (Some(Rel::Ours), Some("metagross")));
    }

    /// R3 — `gen3_battle.py:717,727`: `-miss|<user>|<target>` names the USER at index 2.
    #[test]
    fn r3_a_miss_targets_the_named_user() {
        let e = last(&["|move|p2a: Gyarados|Hydro Pump|p1a: Metagross|[miss]", "|-miss|p2a: Gyarados|p1a: Metagross"],
                     EventKind::Miss);
        assert_eq!(e.target.as_deref(), Some("gyarados"));
        assert_eq!(e.side, Some(Rel::Opp));
    }

    /// R4 — `gen3_battle.py:532-539`.
    #[test]
    fn r4_a_still_move_targets_the_foe_active() {
        let e = last(&["|move|p1a: Metagross|Protect||[still]"], EventKind::Move);
        assert_eq!(e.target.as_deref(), Some("gyarados"));
    }

    /// R5 — `gen3_battle.py:488-502`.
    #[test]
    fn r5_a_miss_suffix_adds_a_synthetic_miss() {
        let rs = run(&["|move|p2a: Gyarados|Hydro Pump|p1a: Metagross|[miss]"]);
        let tail: Vec<_> = rs.iter().rev().take(2).rev().collect();
        assert_eq!(tail[0].kind, EventKind::Move);
        assert_eq!(tail[1].kind, EventKind::Miss);
        assert_eq!(tail[1].get("from"), Some(&Value::Str("move-suffix".into())));
        assert_eq!(tail[1].target, None);
    }

    /// R6 — the viewer's rendering: own exact, foe percent.
    #[test]
    fn r6_hp_is_the_viewers_rendering() {
        let e = last(&["|-damage|p2a: Gyarados|54/100"], EventKind::Damage);
        assert_eq!(e.get("hp_after"), Some(&Value::Float(0.54)));
        assert_eq!(e.get("amount"), Some(&Value::Float(0.54 - 1.0)));
        let e = last(&["|-damage|p1a: Metagross|200/301"], EventKind::Damage);
        assert_eq!(e.get("hp_after"), Some(&Value::Float(200.0 / 301.0)));
    }

    /// R7 — `gen3_battle.py:744-749`.
    #[test]
    fn r7_effectiveness_drops_the_ability_cause() {
        let e = last(&["|move|p1a: Metagross|Earthquake|p2a: Gyarados", "|-immune|p2a: Gyarados|[from] ability: Levitate"],
                     EventKind::Immune);
        assert_eq!(e.value, vec![("multiplier", Value::Float(0.0))]);
    }

    /// R8 — `gen3_battle.py:687-691`.
    #[test]
    fn r8_cureteam_status_is_the_from_clause() {
        let e = last(&["|-cureteam|p2a: Gyarados|[from] move: Aromatherapy"], EventKind::Curestatus);
        assert_eq!(e.get("status"), Some(&Value::Str("[from] move: Aromatherapy".into())));
        assert_eq!(e.get("op"), Some(&Value::Str("cureteam".into())));
    }

    /// R9 — `gen3_battle.py:642-652`: DAMAGE/HEAL carry `reason`, never `of`.
    #[test]
    fn r9_damage_drops_the_of_source() {
        let e = last(&["|-damage|p1a: Metagross|280/301|[from] Recoil|[of] p2a: Gyarados"], EventKind::Damage);
        assert_eq!(e.get("reason"), Some(&Value::Str("Recoil".into())));
        assert!(e.get("of").is_none());
    }

    /// R10 — `abstract_battle.py:421-424`: an unintroduced mon's species is its NAME, id'd.
    #[test]
    fn r10_an_unintroduced_mon_is_created_from_its_name() {
        let mut r = Reader::new(0);
        for l in ["|player|p1|me||", "|player|p2|foe||", "|switch|p2a: Gyarados|Gyarados, M|100/100",
                  "|switch|p1a: Metagross|Metagross|301/301"] {
            r.feed(&Line::parse(l).unwrap()).unwrap();
        }
        let rs = r.feed(&Line::parse("|-curestatus|p2: Big Fish|slp|[silent]").unwrap());
        // poke-env: `cure_status` of a created mon whose status is None clears nothing; the
        // event names the created species `bigfish`.
        assert_eq!(rs.unwrap()[0].actor.as_deref(), Some("bigfish"));
    }

    /// R11 — `pokemon.py:469-471`.
    #[test]
    fn r11_a_forme_change_keeps_the_species() {
        let e = last(&["|-formechange|p2a: Gyarados|Gyarados-Mega|[msg]", "|-damage|p2a: Gyarados|40/100"],
                     EventKind::Damage);
        assert_eq!(e.actor.as_deref(), Some("gyarados"));
    }

    #[test]
    fn move_target_status_is_sampled_before_the_line() {
        let e = last(&["|-status|p2a: Gyarados|par", "|move|p1a: Metagross|Meteor Mash|p2a: Gyarados"], EventKind::Move);
        assert_eq!(e.get("target_status"), Some(&Value::Str("PAR".into())));
        let e = last(&["|move|p2a: Gyarados|Sleep Talk|p2a: Gyarados", "|move|p2a: Gyarados|Earthquake|p1a: Metagross|[from] Sleep Talk"],
                     EventKind::Move);
        assert_eq!(e.get("from_move"), Some(&Value::Str("sleeptalk".into())));
    }

    #[test]
    fn an_unavailable_choice_is_an_out_of_band_rejection_on_our_log() {
        let rs = run(&["|error|[Unavailable choice] Can't switch: The active Pokémon is trapped"]);
        let e = rs.last().unwrap();
        assert_eq!(e.kind, EventKind::ChoiceRejected);
        assert_eq!((e.side, e.actor.as_deref()), (Some(Rel::Ours), Some("metagross")));
    }

    #[test]
    fn an_unsupported_keyword_is_refused() {
        let mut r = Reader::new(0);
        assert!(r.feed(&Line::parse("|-mega|p1a: X|Y").unwrap()).is_err());
    }
}
