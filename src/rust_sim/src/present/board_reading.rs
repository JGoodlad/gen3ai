//! [`BoardReading`] — poke-env's `Battle` (+ `Gen3Battle`'s weather fold) for ONE side's stream: the
//! state HALF of `AbstractBattle.parse_message`, `Battle.parse_request` and the two terminal hooks,
//! each branch naming the handler it mirrors.
//!
//! It is fed exactly what a live `Player` routes to its battle (`agents.battle.offline_feed` is
//! the Python twin of the dispatch): a `|request|` to [`BoardReading::parse_request`], `|win|` / `|tie|`
//! to the terminal hooks, the player's own ignores dropped, everything else to
//! [`BoardReading::parse_message`]. Where poke-env RAISES on a line, this returns `Err` — a reading
//! that would crash the live player is refused, never guessed at.

use super::dex;
use super::mon::{PMon, PMove, StartDetails, TbMon, R};
use super::tables::BATTLE_IGNORED;
use crate::core_events::jsonval::Val;
use crate::core_events::{Intercept, Line, Route};
use crate::core_error::{malformed, refuse, PyExc};

/// One `side.pokemon[i]` record of a `|request|`, the fields `update_from_request` reads.
#[derive(Debug, Clone, PartialEq)]
pub struct ReqMon {
    pub ident: String,
    pub details: String,
    pub condition: String,
    pub active: bool,
    pub stats: Option<Vec<(String, i64)>>,
    pub moves: Vec<String>,
    pub base_ability: Option<String>,
    pub ability: Option<String>,
    pub item: String,
    pub reviving: Option<bool>,
}

impl ReqMon {
    fn from_val(v: &Val) -> R<ReqMon> {
        let s = |k: &str| v.str_at(k).map(str::to_string);
        let stats = match v.get("stats") {
            Some(Val::Obj(kv)) => Some(
                kv.iter()
                    .map(|(k, x)| match x {
                        Val::Int(n) => Ok((k.clone(), *n)),
                        other => Err(malformed(format!("request stat {k}: {other:?} is not an int"))),
                    })
                    .collect::<R<Vec<_>>>()?,
            ),
            _ => None,
        };
        let moves = match v.get("moves") {
            Some(Val::Arr(a)) => a.iter().filter_map(|m| if let Val::Str(s) = m { Some(s.clone()) } else { None }).collect(),
            _ => return Err(refuse(PyExc::KeyError, "request mon without moves (KeyError)")),
        };
        Ok(ReqMon {
            ident: s("ident").ok_or_else(|| refuse(PyExc::KeyError, "request mon without ident (KeyError)"))?,
            details: s("details").ok_or_else(|| refuse(PyExc::KeyError, "request mon without details (KeyError)"))?,
            condition: s("condition").ok_or_else(|| refuse(PyExc::KeyError, "request mon without condition (KeyError)"))?,
            active: truthy(v.get("active")),
            stats,
            moves,
            base_ability: s("baseAbility"),
            ability: s("ability"),
            item: s("item").ok_or_else(|| refuse(PyExc::KeyError, "request mon without item (KeyError)"))?,
            reviving: v.get("reviving").map(|x| truthy(Some(x))),
        })
    }
}

/// Python truthiness of a JSON value.
pub fn truthy(v: Option<&Val>) -> bool {
    match v {
        None | Some(Val::Null) => false,
        Some(Val::Bool(b)) => *b,
        Some(Val::Int(n)) => *n != 0,
        Some(Val::Float(x)) => *x != 0.0,
        Some(Val::Str(s)) => !s.is_empty(),
        Some(Val::Arr(a)) => !a.is_empty(),
        Some(Val::Obj(o)) => !o.is_empty(),
    }
}

/// `Gen3Battle`'s incrementally folded weather (`_weather_id` / `_weather_permanent` /
/// `_weather_start_turn`), mirroring `_update_weather` over each WEATHER event (rule V12).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WeatherFold {
    pub id: Option<String>,
    pub permanent: bool,
    pub start_turn: u32,
}

/// One side's reading of the battle.
#[derive(Debug, Clone, PartialEq)]
pub struct BoardReading {
    /// The viewer: 0 = p1.
    pub viewer: u8,
    /// `_player_username` — `won_by` compares against it, the `|player|` handler keys the role.
    username: String,
    /// `_player_role` as a side index.
    pub role: u8,
    pub turn: u32,
    /// `_team`, insertion-ordered by key (`p1: Name`).
    pub team: Vec<(String, PMon)>,
    /// `_opponent_team` (reveal order).
    pub opp: Vec<(String, PMon)>,
    /// `_team_size`, keyed by role.
    team_size: [Option<usize>; 2],
    /// `_side_conditions` / `_opponent_side_conditions`: `(SideCondition NAME, value)`, dict order.
    pub side_conditions: [Vec<(&'static str, u32)>; 2],
    /// `_fields` (NAME → start turn).
    fields: Vec<(&'static str, u32)>,
    pub weather: WeatherFold,
    pub finished: bool,
    pub won: Option<bool>,
    // ---- the request half (`Battle.parse_request`) ----
    pub wait: bool,
    pub last_request: Option<Val>,
    /// The raw JSON text of `last_request` (what `LegalActions.last_request` mirrors).
    pub last_request_text: Option<String>,
    pub force_switch: bool,
    pub trapped: bool,
    pub maybe_trapped: bool,
    reviving: bool,
    /// `_available_moves` — only whether one is Struggle is read (`LegalActions.struggle`).
    pub available_moves: Vec<String>,
    /// `_available_switches`, as indices into `team`.
    pub available_switches: Vec<usize>,
    teambuilder: Option<Vec<TbMon>>,
    /// poke-env's PENDING damaging move per mover (0 = ours, 1 = theirs): captured at the `|move|`
    /// of a Physical/Special move (`_pending_{our,opp}_damaging_move`), as `(turn, event)`.
    pub pending_damaging: [Option<(u32, DamagingMoveRead)>; 2],
    /// …PROMOTED when an effectiveness emission for the defender lands in the same turn
    /// (`_set_effectiveness` → `_{our,opp}_last_damaging_move`). Read through
    /// [`BoardReading::last_damaging_move`] (turn-gated). The Hidden-Power belief's input.
    pub last_damaging: [Option<(u32, DamagingMoveRead)>; 2],
}

/// poke-env's `DamagingMoveEvent`.
#[derive(Debug, Clone, PartialEq)]
pub struct DamagingMoveRead {
    pub user_species: String,
    pub target_species: String,
    /// The target's status when the move fired.
    pub target_status: Option<super::dex::Status>,
    pub move_id: String,
    pub effectiveness: f64,
}

fn ident_side(tok: &str) -> Option<u8> {
    match tok.get(..2) {
        Some("p1") => Some(0),
        Some("p2") => Some(1),
        _ => None,
    }
}

impl BoardReading {
    /// `offline_feed.new_battle(viewer, names, packed_team=…)`: a battle named for the viewer's
    /// player, its role set up front, and — when given — the `_teambuilder_team` a `Player`
    /// builds from its packed team (the only source of our own spread in a no-preview format).
    pub fn new(viewer: usize, username: &str, packed_team: Option<&str>) -> R<BoardReading> {
        Ok(BoardReading {
            viewer: viewer as u8,
            username: username.to_string(),
            role: viewer as u8,
            turn: 0,
            team: Vec::new(),
            opp: Vec::new(),
            team_size: [None, None],
            side_conditions: [Vec::new(), Vec::new()],
            fields: Vec::new(),
            weather: WeatherFold::default(),
            finished: false,
            won: None,
            wait: false,
            last_request: None,
            last_request_text: None,
            force_switch: false,
            trapped: false,
            maybe_trapped: false,
            reviving: false,
            available_moves: Vec::new(),
            available_switches: Vec::new(),
            teambuilder: match packed_team {
                Some(p) if !p.is_empty() => Some(TbMon::parse_team(p)?),
                _ => None,
            },
            pending_damaging: [None, None],
            last_damaging: [None, None],
        })
    }

    // ---------------------------------------------------------------- the dispatch

    /// Feed ONE line of this side's stream, routed exactly as `Player._handle_battle_message`
    /// routes it (`offline_feed.feed_line`).
    pub fn feed(&mut self, line: &Line) -> R<()> {
        match line.kw.route() {
            Route::Plain => Ok(()),
            Route::Unsupported => Err(refuse(PyExc::UnsupportedMessageType, format!("UnsupportedMessageType: {:?}", line.kw.as_str()))),
            Route::Intercept(i) => {
                let sm = line.split_message();
                match i {
                    Intercept::Request => {
                        if sm.len() > 2 && !sm[2].is_empty() {
                            self.parse_request(&sm[2..].join("|"))?;
                        }
                        Ok(())
                    }
                    Intercept::Win => {
                        let who = sm.get(2).cloned().unwrap_or_default();
                        self.won_by(&who);
                        Ok(())
                    }
                    Intercept::Tie => {
                        self.finished = true;
                        Ok(())
                    }
                    Intercept::ShowTeam => Err(malformed("|showteam| is not a gen-3 line")),
                    Intercept::Error | Intercept::BigError | Intercept::Ignored => Ok(()),
                }
            }
            Route::Event(_) | Route::Control | Route::Cosmetic | Route::StateOnly => self.parse_message(line),
        }
    }

    /// `_team_size[pN]`, `None` when no `|teamsize|` named that side.
    pub fn team_size(&self, side: usize) -> Option<usize> {
        self.team_size[side]
    }

    /// `AbstractBattle.won_by`.
    pub fn won_by(&mut self, name: &str) {
        self.won = Some(name == self.username);
        self.finished = true;
    }

    // ---------------------------------------------------------------- lookups

    fn team_idx(&self, own: bool) -> &Vec<(String, PMon)> {
        if own {
            &self.team
        } else {
            &self.opp
        }
    }

    /// `Battle.active_pokemon` / `opponent_active_pokemon`: the FIRST mon flagged active.
    pub fn active_index(&self, own: bool) -> Option<usize> {
        self.team_idx(own).iter().position(|(_, m)| m.active)
    }

    fn mon_at(&mut self, at: (bool, usize)) -> &mut PMon {
        if at.0 {
            &mut self.team[at.1].1
        } else {
            &mut self.opp[at.1].1
        }
    }

    fn mon_ref(&self, at: (bool, usize)) -> &PMon {
        if at.0 {
            &self.team[at.1].1
        } else {
            &self.opp[at.1].1
        }
    }

    /// `AbstractBattle.get_pokemon(identifier, force_self_team, details, request)` — `(own team?,
    /// index)`, creating the mon when unknown. A nickname re-key: an entry whose base species
    /// matches the details' species is RE-KEYED to this identifier (the `matches` branch).
    pub fn get_pokemon(&mut self, identifier: &str, force_self: bool, details: &str, request: Option<&ReqMon>)
        -> R<(bool, usize)> {
        let b = identifier.as_bytes();
        if b.len() < 4 {
            return Err(refuse(PyExc::IndexError, format!("get_pokemon({identifier:?}): IndexError")));
        }
        let key = if b[3] != b' ' { format!("{}{}", &identifier[..2], &identifier[3..]) } else { identifier.to_string() };
        if let Some(i) = self.team.iter().position(|(k, _)| *k == key) {
            return Ok((true, i));
        }
        if let Some(i) = self.opp.iter().position(|(k, _)| *k == key) {
            return Ok((false, i));
        }
        let side = ident_side(&key);
        let own = force_self || side == Some(self.role);
        let name = key.get(3..).unwrap_or("").trim().to_string();
        let name_det = details.split(", ").next().unwrap_or("");
        let mut matches = Vec::new();
        for (i, (_, m)) in self.team_idx(own).iter().enumerate() {
            if m.identifies_as(name_det)? {
                matches.push(i);
            }
        }
        if matches.len() >= 2 {
            return Err(refuse(PyExc::AssertionError, format!("get_pokemon({identifier:?}): two team entries identify as {name_det:?} (AssertionError)")));
        }
        if let Some(&i) = matches.first() {
            let team = if own { &mut self.team } else { &mut self.opp };
            team[i].0 = key.clone();
            team[i].1.name = Some(key.get(4..).unwrap_or("").to_string());
            return Ok((own, i));
        }
        let role_side = side.ok_or_else(|| malformed(format!("get_pokemon({identifier:?}): no side")))? as usize;
        let len = self.team_idx(own).len();
        if self.team_size.iter().any(|x| x.is_some()) {
            match self.team_size[role_side] {
                Some(n) if len >= n => {
                    return Err(refuse(PyExc::ValueError, format!("{}'s team already has {n} pokemons: cannot add {key} (ValueError)", &key[..2])));
                }
                None => return Err(refuse(PyExc::KeyError, format!("_team_size[{:?}]: KeyError", &key[..2]))),
                _ => {}
            }
        }
        let mon = if let Some(req) = request {
            PMon::from_request(req, Some(name))?
        } else if !details.is_empty() {
            PMon::from_details(details, Some(name))?
        } else {
            PMon::from_species(key.get(4..).unwrap_or(""), Some(name))?
        };
        let team = if own { &mut self.team } else { &mut self.opp };
        team.push((key, mon));
        Ok((own, team.len() - 1))
    }

    fn mon(&mut self, identifier: &str) -> R<&mut PMon> {
        let at = self.get_pokemon(identifier, false, "", None)?;
        Ok(self.mon_at(at))
    }

    // ---------------------------------------------------------------- the request

    /// `Battle.parse_request(request)` — with the fork's R3 fix (`_sync_active_pp`).
    pub fn parse_request(&mut self, json: &str) -> R<()> {
        let v = Val::parse(json).map_err(|e| malformed(format!("request JSON: {e}")))?;
        self.wait = truthy(v.get("wait"));
        let side = v.get("side").ok_or_else(|| refuse(PyExc::KeyError, "request without side (KeyError)"))?;
        let mons: Vec<Val> = match side.get("pokemon") {
            Some(Val::Arr(a)) => a.clone(),
            _ => return Err(refuse(PyExc::KeyError, "request side without pokemon (KeyError)")),
        };
        let recs = mons.iter().map(ReqMon::from_val).collect::<R<Vec<_>>>()?;
        self.available_moves.clear();
        self.available_switches.clear();
        self.maybe_trapped = false;
        self.reviving = recs.iter().any(|m| m.reviving == Some(true));
        self.trapped = false;
        self.force_switch = match v.get("forceSwitch") {
            Some(Val::Arr(a)) => truthy(a.first()),
            None => false,
            Some(other) => return Err(malformed(format!("forceSwitch {other:?} is not a list"))),
        };
        self.last_request_text = Some(json.to_string());
        if truthy(v.get("teamPreview")) {
            return Err(malformed("teamPreview is not a gen-3 request"));
        }
        if let Some(first) = recs.first() {
            self.role = ident_side(&first.ident).ok_or_else(|| malformed("request ident without side"))?;
        }
        self.update_team_from_request(&recs)?;
        self.backfill_teambuilder_spread()?;
        if let Some(Val::Arr(active)) = v.get("active") {
            let ar = active.first().ok_or_else(|| refuse(PyExc::IndexError, "request active [] (IndexError)"))?;
            if truthy(ar.get("trapped")) {
                self.trapped = true;
            }
            if let Some(ai) = self.active_index(true) {
                self.sync_active_pp(ai, ar)?;
                let avail = available_moves_from_request(&self.team[ai].1, ar)?;
                self.available_moves.extend(avail);
            }
            if truthy(ar.get("maybeTrapped")) {
                self.maybe_trapped = true;
            }
        } else if v.get("active").is_some() {
            return Err(malformed("request active is not a list"));
        }
        if !self.trapped {
            for r in &recs {
                let i = self
                    .team
                    .iter()
                    .position(|(k, _)| *k == r.ident)
                    .ok_or_else(|| refuse(PyExc::KeyError, format!("team[{:?}]: KeyError", r.ident)))?;
                let m = &self.team[i].1;
                if self.reviving {
                    if m.fainted() {
                        self.available_switches.push(i);
                    }
                } else if !m.active && !m.fainted() {
                    self.available_switches.push(i);
                }
            }
        }
        self.last_request = Some(v);
        Ok(())
    }

    /// `AbstractBattle._update_team_from_request(side)`: create unknown mons from the record,
    /// resync the active flags (`was_illusioned` / `switch_in`), then `update_from_request` each.
    fn update_team_from_request(&mut self, recs: &[ReqMon]) -> R<()> {
        let mut falsely = Vec::new();
        let mut truly = Vec::new();
        for r in recs {
            if !self.team.iter().any(|(k, _)| *k == r.ident) {
                self.get_pokemon(&r.ident, true, &r.details, Some(r))?;
            }
            let i = self
                .team
                .iter()
                .position(|(k, _)| *k == r.ident)
                .ok_or_else(|| refuse(PyExc::KeyError, format!("team[{:?}]: KeyError", r.ident)))?;
            let m = &self.team[i].1;
            if r.active && !m.active {
                truly.push(i);
            } else if !r.active && m.active {
                falsely.push(i);
            }
        }
        for i in falsely {
            self.team[i].1.was_illusioned()?;
        }
        for i in truly {
            self.team[i].1.switch_in(None)?;
        }
        for r in recs {
            match self.team.iter().position(|(k, _)| *k == r.ident) {
                Some(i) => self.team[i].1.update_from_request(r)?,
                None => {
                    self.get_pokemon(&r.ident, true, &r.details, Some(r))?;
                }
            }
        }
        Ok(())
    }

    /// `AbstractBattle.backfill_teambuilder_spread()`: each declared mon's IVs / EVs / nature onto
    /// the team member of the same species (`setdefault` — the first one).
    fn backfill_teambuilder_spread(&mut self) -> R<()> {
        let Some(tb) = self.teambuilder.clone() else { return Ok(()) };
        for t in &tb {
            let raw = match (&t.species, &t.nickname) {
                (Some(s), _) => s.clone(),
                (None, Some(n)) => n.clone(),
                (None, None) => continue,
            };
            let want = dex::to_id(&raw);
            if let Some(i) = self.team.iter().position(|(_, m)| dex::to_id(&m.species) == want) {
                self.team[i].1.backfill_spread(t);
            }
        }
        Ok(())
    }

    /// `Battle._sync_active_pp(active_request)` — the fork's R3 fix: our ACTIVE mon's move PP is
    /// the request's (the sim's own word), matched by `Move.retrieve_id`.
    fn sync_active_pp(&mut self, ai: usize, ar: &Val) -> R<()> {
        let Some(Val::Arr(reqs)) = ar.get("moves") else { return Ok(()) };
        for req in reqs {
            let pp = match req.get("pp") {
                Some(Val::Int(n)) => *n,
                None | Some(Val::Null) => continue,
                Some(other) => return Err(malformed(format!("request pp {other:?} is not an int"))),
            };
            let rid = match req.str_at("id") {
                Some(r) if !r.is_empty() && r != "struggle" => r,
                _ => continue,
            };
            let want = dex::retrieve_id(rid);
            let mon = &mut self.team[ai].1;
            let keys: Vec<String> = mon.moves.moves().into_iter().map(|(k, m)| format!("{k}\u{0}{}", m.id)).collect();
            for kk in keys {
                let (k, id) = kk.split_once('\u{0}').expect("joined");
                if dex::retrieve_id(id) == want {
                    if let Some(at) = mon.moves.lookup(k) {
                        if let Some(m) = mon.moves.get_mut(&at) {
                            m.current_pp = pp.max(0) as u32;
                        }
                    }
                    break;
                }
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------- switching

    /// `Battle.switch(pokemon_str, details, hp_status, from_baton_pass)`.
    fn switch(&mut self, pokemon_str: &str, details: &str, hp_status: &str, from_baton_pass: bool) -> R<()> {
        let own = ident_side(pokemon_str) == Some(self.role);
        let outgoing = self.active_index(own);
        let snapshot = match (from_baton_pass, outgoing) {
            (true, Some(o)) => Some(self.mon_ref((own, o)).baton_pass_snapshot()),
            _ => None,
        };
        if let Some(o) = outgoing {
            self.mon_at((own, o)).switch_out();
        }
        let at = self.get_pokemon(pokemon_str, false, details, None)?;
        let m = self.mon_at(at);
        m.switch_in(Some(details))?;
        if m.status == Some(super::dex::Status::Tox) {
            m.status_counter = 0; // `tox.onSwitchIn` resets the stage (`Battle.switch`, the fork's PE-R1b fix)
        }
        m.set_hp_status(hp_status, false)?;
        if let Some(s) = snapshot {
            m.apply_baton_pass(s);
        }
        Ok(())
    }

    /// `AbstractBattle.end_turn(turn)`: the new turn, then `end_turn()` on both actives.
    fn end_turn(&mut self, turn: u32) {
        self.turn = turn;
        for own in [true, false] {
            if let Some(i) = self.active_index(own) {
                self.mon_at((own, i)).end_turn();
            }
        }
    }

    // ---------------------------------------------------------------- side / field

    fn cond_idx(&self, side_tok: &str) -> usize {
        if ident_side(side_tok) == Some(self.role) {
            0
        } else {
            1
        }
    }

    /// `AbstractBattle._side_start`: a stackable condition counts layers, any other stores the
    /// turn it STARTED, only when absent (rule V11).
    fn side_start(&mut self, side: &str, cond: &str) {
        let i = self.cond_idx(side);
        let (name, stack) = dex::side_condition(cond);
        let turn = self.turn;
        let conds = &mut self.side_conditions[i];
        if stack > 0 {
            match conds.iter_mut().find(|(n, _)| *n == name) {
                Some((_, v)) => *v += 1,
                None => conds.push((name, 1)),
            }
        } else if !conds.iter().any(|(n, _)| *n == name) {
            conds.push((name, turn));
        }
    }

    /// `AbstractBattle.side_end`: pops the condition (a `KeyError` when absent), UNKNOWN ignored.
    fn side_end(&mut self, side: &str, cond: &str) -> R<()> {
        let i = self.cond_idx(side);
        let (name, _) = dex::side_condition(cond);
        if name == "UNKNOWN" {
            return Ok(());
        }
        let conds = &mut self.side_conditions[i];
        let before = conds.len();
        conds.retain(|(n, _)| *n != name);
        if conds.len() == before {
            return Err(refuse(PyExc::KeyError, format!("side_conditions.pop({name}): KeyError")));
        }
        Ok(())
    }

    fn field_start(&mut self, s: &str) {
        let f = dex::field(s);
        if f.ends_with("_TERRAIN") {
            self.fields.retain(|(n, _)| !n.ends_with("_TERRAIN"));
        }
        let turn = self.turn;
        match self.fields.iter_mut().find(|(n, _)| *n == f) {
            Some((_, t)) => *t = turn,
            None => self.fields.push((f, turn)),
        }
    }

    fn field_end(&mut self, s: &str) -> R<()> {
        let f = dex::field(s);
        if f == "UNKNOWN" {
            return Ok(());
        }
        let before = self.fields.len();
        self.fields.retain(|(n, _)| *n != f);
        if self.fields.len() == before && f != "NEUTRALIZING_GAS" {
            return Err(refuse(PyExc::KeyError, format!("_fields.pop({f}): KeyError")));
        }
        Ok(())
    }

    // ---------------------------------------------------------------- item / ability clauses

    /// `_check_damage_message_for_item`.
    fn damage_item(&mut self, sm: &[String]) -> R<()> {
        if sm.len() == 6 && sm[4].starts_with("[from] item:") && sm[5].starts_with("[of]") {
            let item = sm[4].rsplit("item:").next().unwrap_or("");
            let pk = sm[5].rsplit("[of]").next().unwrap_or("").trim().to_string();
            let item = dex::to_id(item);
            set_item(self.mon(&pk)?, Some(&item));
        } else if sm.len() == 5 && sm[4].starts_with("[from] item:") {
            let item = dex::to_id(sm[4].rsplit("item:").next().unwrap_or(""));
            let pk = sm[2].clone();
            set_item(self.mon(&pk)?, Some(&item));
        }
        Ok(())
    }

    /// `_check_damage_message_for_ability`: the `[of]` mon's ability.
    fn damage_ability(&mut self, sm: &[String]) -> R<()> {
        if sm.len() == 6 && sm[4].starts_with("[from] ability:") && sm[5].starts_with("[of]") {
            let ab = sm[4].rsplit("ability:").next().unwrap_or("").to_string();
            let pk = sm[5].rsplit("[of]").next().unwrap_or("").trim().to_string();
            self.mon(&pk)?.set_ability(&ab);
        }
        Ok(())
    }

    /// `_check_heal_message_for_item`: only onto a mon whose item is not None, never a berry or
    /// herb (the heal line of an eaten berry follows its `-enditem`).
    fn heal_item(&mut self, sm: &[String]) -> R<()> {
        if sm.len() == 5 && sm[4].starts_with("[from] item:") {
            let item = dex::to_id(sm[4].rsplit("item:").next().unwrap_or(""));
            let pk = sm[2].clone();
            let m = self.mon(&pk)?;
            if m.item.is_some() && !item.contains("berry") && !item.contains("herb") {
                set_item(m, Some(&item));
            }
        }
        Ok(())
    }

    /// `_check_heal_message_for_ability`: the HEALED mon's ability (Hospitality: the `[of]` one).
    fn heal_ability(&mut self, sm: &[String]) -> R<()> {
        if sm.len() == 6 && sm[4].starts_with("[from] ability:") {
            let ab = dex::to_id(sm[4].rsplit("ability:").next().unwrap_or(""));
            let pk = if ab == "hospitality" { sm[5].replace("[of] ", "").trim().to_string() } else { sm[2].clone() };
            self.mon(&pk)?.set_ability(&ab);
        }
        Ok(())
    }

    // ---------------------------------------------------------------- parse_message

    /// The state half of `AbstractBattle.parse_message` (+ `Gen3Battle`'s weather fold).
    pub fn parse_message(&mut self, line: &Line) -> R<()> {
        let sm = line.split_message();
        let kw = sm.get(1).map(String::as_str).unwrap_or("");
        if BATTLE_IGNORED.contains(&kw) {
            return Ok(());
        }
        let f = |i: usize| -> R<&str> {
            sm.get(i).map(String::as_str).ok_or_else(|| refuse(PyExc::ValueError, format!("|{kw}|: field {i} missing (ValueError)")))
        };
        match kw {
            "drag" | "switch" => {
                let (p, d, hp) = (f(2)?.to_string(), f(3)?.to_string(), f(4)?.to_string());
                let bp = sm.iter().skip(5).any(|t| t.starts_with("[from]") && t.to_lowercase().contains("baton pass"));
                self.switch(&p, &d, &hp, bp)?;
            }
            "-damage" => {
                let (p, hp) = (f(2)?.to_string(), f(3)?.to_string());
                let m = self.mon(&p)?;
                m.set_hp_status(&hp, false)?;
                if sm.iter().skip(4).any(|t| t == "[from] psn") {
                    m.note_residual_chip();
                }
                self.damage_item(&sm)?;
                self.damage_ability(&sm)?;
            }
            "move" => self.move_line(&sm)?,
            "cant" => {
                let p = f(2)?.to_string();
                self.mon(&p)?.cant_move();
            }
            "-crit" | "-miss" | "-fail" | "-notarget" | "-nothing" => {}
            "turn" => {
                let n: u32 = f(2)?.trim().parse().map_err(|_| refuse(PyExc::ValueError, format!("|turn|{}: int() (ValueError)", f(2).unwrap_or(""))))?;
                self.end_turn(n);
            }
            "-heal" => {
                let (p, hp) = (f(2)?.to_string(), f(3)?.to_string());
                self.mon(&p)?.set_hp_status(&hp, false)?;
                self.heal_ability(&sm)?;
                self.heal_item(&sm)?;
            }
            "-boost" | "-unboost" => {
                let (p, stat) = (f(2)?.to_string(), f(3)?.to_string());
                let amt: i32 = f(4)?.trim().parse().map_err(|_| refuse(PyExc::ValueError, format!("|{kw}| amount: int() (ValueError)")))?;
                self.mon(&p)?.boost(&stat, if kw == "-boost" { amt } else { -amt })?;
            }
            "-weather" => self.weather(&sm)?,
            "faint" => {
                let p = f(2)?.to_string();
                self.mon(&p)?.faint();
            }
            "-ability" => {
                let (p, cause) = (f(2)?.to_string(), f(3)?.to_string());
                let traced = (sm.len() > 4 && sm[4].starts_with("[from] ability: Trace"))
                    || (sm.len() > 5 && sm[5].starts_with("[from] ability: Trace"));
                if traced {
                    let m = self.mon(&p)?;
                    if m.ability() != Some("trace") {
                        if m.temporary_ability().is_some() {
                            m.set_temporary_ability(None);
                        } else if m.ability().is_some() {
                            clear_base_ability(m);
                        }
                        m.set_ability("trace");
                    }
                    m.set_ability(&cause);
                } else if cause == "Neutralizing Gas" {
                    self.mon(&p)?;
                    self.field_start(&cause);
                } else {
                    self.mon(&p)?.set_ability(&cause);
                }
            }
            "-start" => {
                let (p, effect) = (f(2)?.to_string(), f(3)?.to_string());
                self.mon(&p)?;
                if effect == "ability: Flash Fire" {
                    self.set_effectiveness(&p, 0.0);
                }
                if effect == "typechange" {
                    let types = if sm.len() > 5 && sm[5].starts_with("[of] ") {
                        let other = sm[5][5..].to_string();
                        self.mon(&other)?.types().join("/")
                    } else {
                        f(4)?.to_string()
                    };
                    self.mon(&p)?.start_effect(&effect, Some(StartDetails::Types(types)))?;
                } else {
                    if effect == "Mimic" {
                        let mv = PMove::new(&dex::retrieve_id(f(4)?), None, false)?;
                        self.mon(&p)?.moves.set_mimic(Some(mv));
                    }
                    self.mon(&p)?.start_effect(&effect, None)?;
                }
            }
            "-activate" => self.activate(&sm)?,
            "-status" => {
                let (p, st) = (f(2)?.to_string(), f(3)?.to_string());
                let s = dex::status_from(&st)?;
                self.mon(&p)?.set_status(Some(s));
            }
            "rule" => {}
            "-clearallboost" => {
                for own in [true, false] {
                    if let Some(i) = self.active_index(own) {
                        self.mon_at((own, i)).clear_boosts();
                    }
                }
            }
            "-clearboost" => self.mon(&f(2)?.to_string())?.clear_boosts(),
            "-clearnegativeboost" => self.mon(&f(2)?.to_string())?.clear_negative_boosts(),
            "-clearpositiveboost" => self.mon(&f(2)?.to_string())?.clear_positive_boosts(),
            "-copyboost" => {
                // The fork's R2 fix: the FIRST ident RECEIVES the second's stages.
                let (recv, giver) = (f(2)?.to_string(), f(3)?.to_string());
                let r = self.get_pokemon(&recv, false, "", None)?;
                let g = self.mon(&giver)?.boosts;
                self.mon_at(r).boosts = g;
            }
            "-curestatus" => {
                let (p, st) = (f(2)?.to_string(), f(3)?.to_string());
                self.mon(&p)?.cure_status(Some(&st))?;
            }
            "-cureteam" => {
                let p = f(2)?;
                let own = ident_side(p) == Some(self.role);
                let team = if own { &mut self.team } else { &mut self.opp };
                for (_, m) in team.iter_mut() {
                    m.cure_status(None)?;
                }
            }
            "-end" => {
                let (p, effect) = (f(2)?.to_string(), f(3)?.to_string());
                if effect == "ability: Neutralizing Gas" {
                    self.field_end(&effect)?;
                } else {
                    self.mon(&p)?.end_effect(&effect);
                }
            }
            "-endability" => self.mon(&f(2)?.to_string())?.set_temporary_ability(None),
            "-enditem" => {
                let (p, item) = (f(2)?.to_string(), f(3)?.to_string());
                let m = self.mon(&p)?;
                m.consumed_item = Some(item);
                m.item = None;
            }
            "-fieldend" => self.field_end(f(2)?)?,
            "-fieldstart" => {
                let c = f(2)?.to_string();
                self.field_start(&c);
            }
            "-formechange" | "detailschange" => {
                let (p, species) = (f(2)?.to_string(), f(3)?.to_string());
                self.mon(&p)?.forme_change(&species)?;
            }
            "-invertboost" => self.mon(&f(2)?.to_string())?.invert_boosts(),
            "-item" => self.item_line(&sm)?,
            "-mustrecharge" => {
                self.mon(&f(2)?.to_string())?;
            }
            "-prepare" => {
                if sm.len() >= 5 {
                    let d = sm[4].clone();
                    if d != "[premajor]" {
                        self.mon(&d)?;
                    }
                }
                self.mon(&f(2)?.to_string())?;
            }
            "-setboost" => {
                let (p, stat) = (f(2)?.to_string(), f(3)?.to_string());
                let amt: i32 = f(4)?.trim().parse().map_err(|_| refuse(PyExc::ValueError, "|-setboost| amount: int() (ValueError)"))?;
                self.mon(&p)?.set_boost(&stat, amt)?;
            }
            "-sethp" => {
                let (p, hp) = (f(2)?.to_string(), f(3)?.to_string());
                self.mon(&p)?.set_hp_status(&hp, false)?;
            }
            "-sideend" => {
                let (s, c) = (f(2)?.to_string(), f(3)?.to_string());
                self.side_end(&s, &c)?;
            }
            "-sidestart" => {
                let (s, c) = (f(2)?.to_string(), f(3)?.to_string());
                self.side_start(&s, &c);
            }
            "-singleturn" | "-singlemove" => {
                let (p, effect) = (f(2)?.to_string(), f(3)?.to_string());
                self.mon(&p)?.start_effect(&effect.replace("move: ", ""), None)?;
            }
            "-swapboost" => {
                let (s, t, stats) = (f(2)?.to_string(), f(3)?.to_string(), f(4)?.to_string());
                let a = self.get_pokemon(&s, false, "", None)?;
                let b = self.get_pokemon(&t, false, "", None)?;
                let keys: Vec<String> = if stats.contains("[from]") {
                    ["accuracy", "atk", "def", "evasion", "spa", "spd", "spe"].iter().map(|x| x.to_string()).collect()
                } else {
                    stats.split(", ").map(str::to_string).collect()
                };
                for k in keys {
                    let x = *self.mon_at(a).boost_mut(&k)?;
                    let y = *self.mon_at(b).boost_mut(&k)?;
                    *self.mon_at(a).boost_mut(&k)? = y;
                    *self.mon_at(b).boost_mut(&k)? = x;
                }
            }
            "-transform" => {
                let (p, into) = (f(2)?.to_string(), f(3)?.to_string());
                let a = self.get_pokemon(&p, false, "", None)?;
                if sm.len() > 4 && sm[4] == "[from] ability: Imposter" {
                    self.mon_at(a).add_move("transform")?;
                    self.mon_at(a).set_ability("imposter");
                }
                let b = self.get_pokemon(&into, false, "", None)?;
                let target = self.mon_ref(b).clone();
                self.mon_at(a).transform(&target)?;
            }
            "clearpoke" => {
                for (_, m) in self.team.iter_mut() {
                    m.active = false;
                }
            }
            "gen" => {
                if f(2)?.trim() != "3" {
                    return Err(refuse(PyExc::RuntimeError, format!("Battle Initiated with gen 3 but got: |gen|{}", f(2)?)));
                }
            }
            "tier" | "inactive" | "poke" | "raw" | "start" | "title" | "message" | "-message" => {}
            "player" => self.player(&sm)?,
            "replace" | "swap" => {
                return Err(malformed(format!("|{kw}| (Illusion / position swap) cannot occur in gen-3 singles")));
            }
            "teamsize" => {
                let (p, n) = (f(2)?.to_string(), f(3)?.to_string());
                let side = ident_side(&p).ok_or_else(|| malformed(format!("|teamsize|{p}")))? as usize;
                self.team_size[side] = Some(n.trim().parse().map_err(|_| refuse(PyExc::ValueError, format!("|teamsize| {n}: int() (ValueError)")))?);
            }
            "-supereffective" | "-resisted" => {
                if sm.len() >= 3 {
                    let d = sm[2].clone();
                    self.set_effectiveness(&d, if sm[1] == "-supereffective" { 2.0 } else { 0.5 });
                }
            }
            "-immune" => {
                if sm.len() >= 3 {
                    let d = sm[2].clone();
                    self.set_effectiveness(&d, 0.0);
                }
                if sm.len() == 4 && sm[3].starts_with("[from] ability:") {
                    let cause = sm[3].replace("[from] ability:", "");
                    let p = sm[2].clone();
                    self.mon(&p)?.set_ability(&cause);
                }
            }
            "-swapsideconditions" => self.side_conditions.swap(0, 1),
            other => return Err(refuse(PyExc::NotImplementedError, format!("NotImplementedError: |{other}|"))),
        }
        Ok(())
    }

    /// `Gen3Battle._update_weather` over this line's WEATHER event (rule V12): a fresh set's
    /// `[from] ability:` cause makes it permanent and it starts NOW; an `[upkeep]` tick continues;
    /// `none` clears.
    fn weather(&mut self, sm: &[String]) -> R<()> {
        let wid = sm.get(2).map(|s| dex::to_id(s));
        match wid.as_deref() {
            None | Some("none") => {
                self.weather = WeatherFold::default();
                return Ok(());
            }
            _ => {}
        }
        let wid = wid.expect("checked");
        if sm.iter().any(|t| t == "[upkeep]") {
            if self.weather.id.is_none() {
                self.weather.id = Some(wid);
                self.weather.start_turn = self.turn;
            }
            return Ok(());
        }
        // `_parse_from`: the LAST `[from]` token, stripped.
        let from = sm.iter().skip(3).rev().map(|t| t.trim()).find_map(|t| t.strip_prefix("[from]").map(|r| r.trim().to_string()));
        self.weather.permanent = from.unwrap_or_default().starts_with("ability");
        self.weather.id = Some(wid);
        self.weather.start_turn = self.turn;
        Ok(())
    }

    /// The `player` branch: the role is the side whose name is ours.
    fn player(&mut self, sm: &[String]) -> R<()> {
        match sm.len() {
            5 | 6 => {
                let (player, username) = (&sm[2], &sm[3]);
                let side = ident_side(player).ok_or_else(|| malformed(format!("|player|{player}")))?;
                self.role = if *username == self.username { side } else { 1 - side };
                Ok(())
            }
            4 => {
                if !sm[3].is_empty() {
                    return Err(refuse(PyExc::RuntimeError, format!("Invalid player message: {sm:?}")));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// The `move` branch: the suffix strips, the `[from]` reveal / use rules, the silent
    /// Minimize, Pressure (`_pressure_on`, rule V3), and `Pokemon.moved`.
    fn move_line(&mut self, sm0: &[String]) -> R<()> {
        let mut ev: Vec<String> = sm0.to_vec();
        let who = ev.get(2).cloned().ok_or_else(|| refuse(PyExc::IndexError, "|move|: no user"))?;
        self.mon(&who)?;
        let (mut use_, mut reveal, mut failed, mut spread) = (true, true, false, false);
        let mut overridden: Option<String> = None;
        let last = |ev: &Vec<String>| ev.last().cloned().unwrap_or_default();
        // `_canonical_from_tail` (`gen3_called_move_reading_v1`): a tail of two or more
        // `attrLastMove` flags after the `[from]` clause is put in the order the single-pass strip
        // below consumes, each flag once (`…|[from] Metronome|[miss]|[miss]` is a real gen3 shape).
        canonical_from_tail(&mut ev);
        for suffix in ["[miss]", "[still]", "[notarget]"] {
            if last(&ev) == suffix {
                ev.pop();
                failed = true;
            }
        }
        if last(&ev) == "[notarget]" {
            ev.pop();
        }
        while last(&ev).starts_with("[spread]") {
            spread = true;
            ev.pop();
        }
        if matches!(last(&ev).as_str(), "[from] lockedmove" | "[from]lockedmove" | "[from] Sky Attack") {
            use_ = false;
            reveal = false;
            ev.pop();
        }
        if matches!(last(&ev).as_str(), "[from] Pursuit" | "[from]Pursuit" | "[zeffect]") {
            ev.pop();
        }
        if last(&ev) == "[from] Sleep Talk" {
            *ev.last_mut().expect("nonempty") = "[from] move: Sleep Talk".into();
        }
        if last(&ev).starts_with("[anim]") {
            ev.pop();
        }
        let l = last(&ev);
        if l.starts_with("[from] move: ") || l.starts_with("[from]move: ") {
            ev.pop();
            let o = l.rsplit(": ").next().unwrap_or("").to_string();
            match o.as_str() {
                "Sleep Talk" => overridden = Some(o),
                "Copycat" | "Metronome" | "Nature Power" | "Round" => {
                    reveal = false;
                    overridden = Some(o);
                }
                "Grass Pledge" | "Water Pledge" | "Fire Pledge" => {}
                _ => return Err(refuse(PyExc::ValueError, format!("Unhandled [from] move message - move {o:?} (ValueError)"))),
            }
        }
        if last(&ev) == "null" {
            ev.pop();
        }
        let l = last(&ev);
        if l.starts_with("[from] ability: ") || l.starts_with("[from]ability: ") {
            ev.pop();
            let ab = l.rsplit(": ").next().unwrap_or("").to_string();
            let p = ev[2].clone();
            self.mon(&p)?.set_ability(&ab);
            match ab.as_str() {
                "Magic Bounce" => {
                    use_ = false;
                    reveal = false;
                }
                "Dancer" => return Ok(()),
                _ => return Err(refuse(PyExc::ValueError, format!("Unhandled [from] ability message - ability {ab:?} (ValueError)"))),
            }
        }
        // The RANDOM callers (`GEN3_BARE_MOVE_CALLERS`, `gen3_called_move_reading_v1`) join the
        // class: the called move is not the actor's, so it is neither revealed nor used, and gen3
        // charges no Pressure PP for a sourced move.
        if matches!(last(&ev).as_str(), "[from] Magic Coat" | "[from] Mirror Move" | "[from] Snatch" | "[from]Snatch")
            || is_gen3_bare_move_caller(&last(&ev))
        {
            use_ = false;
            reveal = false;
            ev.pop();
        }
        while last(&ev) == "[still]" {
            ev.pop();
        }
        loop {
            let l = last(&ev);
            if matches!(l.as_str(), "[miss]" | "[notarget]" | "[still]") || l.starts_with("[anim]") || l.starts_with("[spread]") {
                ev.pop();
                if l == "[miss]" || l == "[notarget]" {
                    failed = true;
                }
            } else {
                break;
            }
        }
        let mut presumed: Option<String> = None;
        let (pokemon, mv) = match ev.len() {
            4 => (ev[2].clone(), ev[3].clone()),
            5 => {
                let t = ev[4].clone();
                let ok = t.is_empty()
                    || (t.len() > 4 && ["p1: ", "p2: ", "p1a:", "p1b:", "p2a:", "p2b:"].contains(&&t[..4]));
                if !ok {
                    return Err(refuse(PyExc::ValueError, format!("Unhandled move message format - {ev:?} (ValueError)")));
                }
                presumed = Some(t);
                (ev[2].clone(), ev[3].clone())
            }
            n if n > 5 => {
                if !ev[4].is_empty() {
                    return Err(refuse(PyExc::ValueError, format!("Unhandled move message format - {ev:?} (ValueError)")));
                }
                presumed = Some(String::new());
                (ev[2].clone(), ev[3].clone())
            }
            _ => return Err(refuse(PyExc::ValueError, format!("|move| too short: {ev:?} (ValueError)"))),
        };
        if mv.trim().to_uppercase() == "MINIMIZE" {
            self.mon(&pokemon)?.start_effect("MINIMIZE", None)?;
        }
        if spread || presumed.as_deref() == Some("") {
            presumed = None;
        }
        let pressure = self.pressure_on(&pokemon, &mv, presumed.as_deref())?;
        let m = self.mon(&pokemon)?;
        if let Some(o) = overridden {
            m.moved(&mv, failed, false, reveal, false)?;
            let key = dex::retrieve_id(&o);
            let at = m.moves.lookup(&key).ok_or_else(|| refuse(PyExc::KeyError, format!("moves[{key:?}]: KeyError (overridden move)")))?;
            m.moves.get_mut(&at).expect("looked up").use_move(pressure, true);
        } else if !failed && matches!(mv.as_str(), "Sleep Talk" | "Copycat" | "Metronome" | "Nature Power") {
            m.moved(&mv, failed, use_, reveal, false)?;
        } else {
            m.moved(&mv, failed, use_, reveal, pressure)?;
        }
        // The pending damaging-move capture (after `moved`): `get_pokemon(presumed_target)` for a
        // Physical / Special move — a lookup that CREATES an unseen mon, so its order matters to
        // the reveal order (rule V2).
        if dex::move_row(&dex::to_id(&mv)).is_some_and(|r| r.damaging) {
            let user_species = self.mon(&pokemon)?.species.clone();
            let (target_species, target_status) = match presumed.as_deref() {
                Some(t) => {
                    let tm = self.mon(t)?;
                    (tm.species.clone(), tm.status)
                }
                None => (user_species.clone(), None),
            };
            let mover = usize::from(ident_side(&pokemon) != Some(self.role));
            self.pending_damaging[mover] = Some((
                self.turn,
                DamagingMoveRead { user_species, target_species, target_status, move_id: dex::to_id(&mv), effectiveness: 1.0 },
            ));
        }
        Ok(())
    }

    /// `AbstractBattle._set_effectiveness(defender_side, mult)` — the promotion half.
    fn set_effectiveness(&mut self, defender: &str, mult: f64) {
        // the defender is OURS ⇒ the opponent's move resolved against us
        let mover = usize::from(ident_side(defender) == Some(self.role));
        if let Some((t, ev)) = &self.pending_damaging[mover] {
            if *t == self.turn {
                self.last_damaging[mover] = Some((self.turn, DamagingMoveRead { effectiveness: mult, ..ev.clone() }));
            }
        }
    }

    /// `battle.{our,opp}_last_damaging_move` (`mover` 0 = ours, 1 = theirs) — gated to the turn
    /// that just resolved (`_last_turn_gated`).
    pub fn last_damaging_move(&self, mover: usize) -> Option<&DamagingMoveRead> {
        match &self.last_damaging[mover] {
            Some((t, ev)) if *t + 1 == self.turn => Some(ev),
            _ => None,
        }
    }

    /// `AbstractBattle._pressure_on(pokemon, move, target_str)` with `Battle._get_target_mon`.
    fn pressure_on(&mut self, pokemon: &str, mv: &str, target: Option<&str>) -> R<bool> {
        let id = dex::retrieve_id(mv);
        let Some(row) = dex::move_row(&id) else { return Ok(false) };
        let at = if row.target != "all" && target.is_some() {
            Some(self.get_pokemon(target.expect("checked"), false, "", None)?)
        } else if ident_side(pokemon) == Some(self.role) {
            self.active_index(false).map(|i| (false, i))
        } else {
            self.active_index(true).map(|i| (true, i))
        };
        let Some(at) = at else { return Ok(false) };
        let t = self.mon_ref(at);
        const FOE: [&str; 7] = ["all", "allAdjacent", "allAdjacentFoes", "any", "normal", "randomNormal", "scripted"];
        Ok(t.ability() == Some("pressure") && !t.fainted() && (FOE.contains(&row.target) || row.must_pressure))
    }

    /// The `-activate` branch.
    fn activate(&mut self, sm: &[String]) -> R<()> {
        let target = sm.get(2).cloned().ok_or_else(|| refuse(PyExc::ValueError, "|-activate|: field 2 (ValueError)"))?;
        let effect = sm.get(3).cloned().ok_or_else(|| refuse(PyExc::ValueError, "|-activate|: field 3 (ValueError)"))?;
        let of_src = |sm: &[String], dflt: &str| -> String {
            sm.iter().skip(4).find_map(|t| t.strip_prefix("[of] ").map(str::to_string)).unwrap_or_else(|| dflt.to_string())
        };
        if !target.is_empty() && effect.replace("move: ", "") == "Skill Swap" {
            if sm.len() > 4 {
                let lastf = sm.last().cloned().unwrap_or_default();
                let clean = |a: &String| a.replace("[ability] ", "").replace("[ability2] ", "");
                if let Some(of) = lastf.strip_prefix("[of] ") {
                    let abilities: Vec<String> = sm[4..sm.len() - 1].iter().map(clean).filter(|a| !a.is_empty()).collect();
                    let t = self.get_pokemon(&target, false, "", None)?;
                    let a = self.get_pokemon(of, false, "", None)?;
                    if abilities.len() >= 2 {
                        self.mon_at(t).start_effect(&effect, Some(StartDetails::Abilities(abilities[..2].to_vec())))?;
                        self.mon_at(a).set_temporary_ability(Some(&abilities[1]));
                    } else {
                        let ta = self.mon_ref(t).ability().map(str::to_string);
                        let aa = self.mon_ref(a).ability().map(str::to_string);
                        if let (Some(ta), Some(aa)) = (ta, aa) {
                            self.mon_at(t).set_temporary_ability(Some(&aa));
                            self.mon_at(a).set_temporary_ability(Some(&ta));
                        }
                    }
                } else {
                    let a = self.get_pokemon(&sm[4].clone(), false, "", None)?;
                    let abilities: Vec<String> = sm[5..].iter().map(clean).filter(|x| !x.is_empty()).collect();
                    if abilities.len() >= 2 {
                        let t = self.get_pokemon(&target, false, "", None)?;
                        self.mon_at(t).start_effect(&effect, Some(StartDetails::Abilities(abilities[..2].to_vec())))?;
                        self.mon_at(a).set_temporary_ability(Some(&abilities[1]));
                    }
                }
            }
            return Ok(());
        }
        match effect.as_str() {
            "ability: Dancer" => {
                self.mon(&target)?;
            }
            "ability: Mummy" => {
                let src = of_src(sm, sm.get(4).map(String::as_str).unwrap_or(""));
                self.mon(&src)?.set_temporary_ability(Some("mummy"));
            }
            "ability: Wandering Spirit" => {
                let src = of_src(sm, &target);
                let a4 = sm.get(4).cloned().unwrap_or_default();
                self.mon(&target)?.set_temporary_ability(Some(&a4));
                self.mon(&src)?.set_temporary_ability(Some("wanderingspirit"));
            }
            "ability: Symbiosis" => {
                let src = of_src(sm, &target);
                let it = sm.get(4).cloned().unwrap_or_default().replace("[item] ", "");
                set_item(self.mon(&src)?, Some(&it));
                set_item(self.mon(&target)?, None);
            }
            "item: Leppa Berry" => {
                let key = dex::to_id(sm.get(4).ok_or_else(|| refuse(PyExc::IndexError, "Leppa: field 4 (IndexError)"))?);
                let m = self.mon(&target)?;
                let at = m.moves.lookup(&key).ok_or_else(|| refuse(PyExc::KeyError, format!("moves[{key:?}]: KeyError (Leppa Berry)")))?;
                let mv = m.moves.get_mut(&at).expect("looked up");
                mv.current_pp = (mv.current_pp + 10).min(mv.max_pp()?);
            }
            "move: Mimic" => {
                let mv = PMove::new(&dex::retrieve_id(sm.get(4).ok_or_else(|| refuse(PyExc::IndexError, "Mimic: field 4 (IndexError)"))?), None, false)?;
                self.mon(&target)?.moves.set_mimic(Some(mv));
            }
            "move: Trick" => {
                let src = of_src(sm, sm.get(4).map(String::as_str).unwrap_or(""));
                let a = self.get_pokemon(&target, false, "", None)?;
                let b = self.get_pokemon(&src, false, "", None)?;
                let ia = self.mon_ref(a).item.clone();
                let ib = self.mon_ref(b).item.clone();
                self.mon_at(a).item = ib;
                self.mon_at(b).item = ia;
            }
            _ if !target.is_empty() => {
                if let Some(ab) = effect.strip_prefix("ability: ") {
                    let holder = sm.iter().skip(4).find_map(|t| t.strip_prefix("[of] ").map(str::to_string)).unwrap_or_else(|| target.clone());
                    let h = self.mon(&holder)?;
                    if h.ability().is_none() {
                        h.set_ability(ab);
                    }
                }
                self.mon(&target)?.start_effect(&effect, None)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// The `-item` branch.
    fn item_line(&mut self, sm: &[String]) -> R<()> {
        if sm.len() == 6 {
            let (item, cause) = (sm[3].clone(), sm[4].clone());
            match cause.as_str() {
                "[from] ability: Frisk" => {
                    let p = sm[5].rsplit("[of] ").next().unwrap_or("").to_string();
                    let at = self.get_pokemon(&p, false, "", None)?;
                    let own_act = self.active_index(true).map(|i| (true, i));
                    let opp_act = self.active_index(false).map(|i| (false, i));
                    if Some(at) == own_act {
                        if let Some(o) = opp_act {
                            set_item(self.mon_at(o), Some(&dex::to_id(&item)));
                        }
                    } else if Some(at) == opp_act {
                        if let Some(o) = own_act {
                            set_item(self.mon_at(o), Some(&dex::to_id(&item)));
                        }
                    }
                    self.mon_at(at).set_ability("frisk");
                }
                "[from] ability: Pickpocket" | "[from] ability: Magician" => {
                    let who = sm[2].clone();
                    let victim = sm[5].replace("[of] ", "");
                    let ab = if cause.ends_with("Pickpocket") { "pickpocket" } else { "magician" };
                    let m = self.mon(&who)?;
                    set_item(m, Some(&dex::to_id(&item)));
                    m.set_ability(ab);
                    set_item(self.mon(&victim)?, None);
                }
                "[from] move: Thief" | "[from] move: Covet" => {
                    let who = sm[2].clone();
                    let victim = sm[5].replace("[of] ", "");
                    set_item(self.mon(&who)?, Some(&dex::to_id(&item)));
                    set_item(self.mon(&victim)?, None);
                }
                _ => return Err(refuse(PyExc::ValueError, format!("Unhandled item message: {sm:?} (ValueError)"))),
            }
            return Ok(());
        }
        let p = sm.get(2).cloned().ok_or_else(|| refuse(PyExc::ValueError, "|-item|: field 2 (ValueError)"))?;
        let item = sm.get(3).cloned().ok_or_else(|| refuse(PyExc::ValueError, "|-item|: field 3 (ValueError)"))?;
        if sm.len() > 4
            && matches!(sm[4].as_str(), "[from] ability: Magician" | "[from] move: Switcheroo" | "[from] move: Trick")
        {
            return Ok(());
        }
        set_item(self.mon(&p)?, Some(&dex::to_id(&item)));
        Ok(())
    }
}

/// `Pokemon.item` setter: id-normalised; a truthy item clears `consumed_item`.
fn set_item(m: &mut PMon, item: Option<&str>) {
    m.item = item.map(dex::to_id);
    if m.item.as_deref().is_some_and(|i| !i.is_empty()) {
        m.consumed_item = None;
    }
}

/// The `-ability` Trace case's `mon._ability = None`.
fn clear_base_ability(m: &mut PMon) {
    m.clear_base_ability();
}

/// `Pokemon.available_moves_from_request(request)` — the ids of the moves `_available_moves`
/// would hold (only `struggle` is read downstream), with poke-env's consistency assertion.
fn available_moves_from_request(mon: &PMon, ar: &Val) -> R<Vec<String>> {
    let mut out = Vec::new();
    if mon.has_effect(dex::effect_by_name("COMMANDER").expect("COMMANDER")) {
        return Ok(out);
    }
    let Some(Val::Arr(reqs)) = ar.get("moves") else { return Err(refuse(PyExc::KeyError, "active request without moves (KeyError)")) };
    let keys: Vec<String> = mon.moves.moves().into_iter().map(|(k, _)| k).collect();
    for r in reqs {
        if truthy(r.get("disabled")) {
            continue;
        }
        let id = r.str_at("id").ok_or_else(|| refuse(PyExc::KeyError, "request move without id (KeyError)"))?.to_string();
        if keys.contains(&id) || dex::is_special_move(&id) {
            out.push(id);
        } else if id == "hiddenpower" && keys.iter().filter(|k| k.starts_with("hiddenpower")).count() == 1 {
            out.push(id);
        } else {
            let caller = ["copycat", "metronome", "mefirst", "mirrormove", "assist"].iter().any(|c| keys.iter().any(|k| k == c));
            if !(mon.ability() == Some("dancer") || caller) {
                return Err(refuse(PyExc::AssertionError, format!("Error with move {id}. Expected self.moves to contain copycat, metronome, mefirst, mirrormove, or assist, or to have the ability dancer (AssertionError)")));
            }
            out.push(id);
        }
    }
    Ok(out)
}

/// `GEN3_BARE_MOVE_CALLERS` (`gen3_called_move_reading_v1`): the random move-callers in the BARE
/// form gen3's `useMoveInner` writes — `[from] Metronome` / `[from] Assist` / `[from] Nature
/// Power`, with or without the space.
pub fn is_gen3_bare_move_caller(tag: &str) -> bool {
    let Some(rest) = tag.strip_prefix("[from]") else { return false };
    let name = rest.strip_prefix(' ').unwrap_or(rest);
    matches!(name, "Metronome" | "Assist" | "Nature Power")
}

/// `_canonical_from_tail` (`gen3_called_move_reading_v1`): a multi-flag tail after the `[from]`
/// clause becomes `[notarget]`, `[still]`, `[miss]` (each present flag once); any other line is
/// left as it is.
pub fn canonical_from_tail(ev: &mut Vec<String>) {
    const FLAGS: [&str; 3] = ["[notarget]", "[still]", "[miss]"];
    let Some(j) = (5..ev.len()).find(|&j| ev[j].starts_with("[from]")) else { return };
    let tail = &ev[j + 1..];
    if tail.len() < 2 || !tail.iter().all(|t| FLAGS.contains(&t.as_str())) {
        return;
    }
    let keep: Vec<String> = FLAGS.iter().filter(|f| tail.iter().any(|t| t == *f)).map(|f| f.to_string()).collect();
    ev.truncate(j + 1);
    ev.extend(keep);
}
