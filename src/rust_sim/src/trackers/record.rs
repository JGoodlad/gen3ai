//! The NATIVE per-decision record (`gen3_core_window_record_v1`): what happened in one side's
//! decision window, as an ORDERED list of ACTIONS, each carrying its ordered EFFECTS with their
//! attribution — built from the side's typed lines (the truth the protocol prints: `[from]`,
//! `[of]`, tags) plus the side's reading of the board at each line (what Baton Pass passed, how
//! many Spikes layers an entrant met).
//!
//! It is ONE SIDE'S record — built from the stream that side receives, so nothing the other side
//! cannot see reaches it (the imperfect-information boundary; the omniscient board only GATES it).
//! It does not flatten: a window with two actions of one side (an action and then its forced
//! replacement; a switch and then a drag), several faints each with its cause, a Baton Pass and
//! what it carried, a called move with its caller, a drag with its phazer, residual damage with its
//! source, a refused choice — each stays a record of its own. The frozen `TurnDelta`, the α/β
//! label and the 22-column event window are LOSSY PROJECTIONS of this (the catalogue of what each
//! loses: `designs/research_state/measurements/rust_core_m3_2026-09-24/`); a future reshaped
//! event block can read this directly.
//!
//! Actions are opened by LINE ORDER (the rule a parse-built stream must use): `|move|`, `|switch|`,
//! `|drag|`, `|cant|`, a refused choice, and the end-of-turn RESIDUAL block (the first line whose
//! `[from]` names a residual source, or a `[upkeep]` weather line, after the turn's actions). On the
//! step path every line also carries the engine's action SCOPE, and the record keeps it
//! ([`Action::scope`]) — the gate that line order reproduces the engine's grouping.

use super::ev;
use crate::core_events::json_out;
use crate::core_events::{Cause as LCause, CoreEvent, EventKind as K, Field, Kw, Line, Reading, Rel, Scope};
use crate::present::board_reading::BoardReading;
use crate::present::dex;

/// A mon, as this side reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct Mon {
    pub side: Rel,
    pub species: String,
}

/// Why a mon entered.
#[derive(Debug, Clone, PartialEq)]
pub enum Entry {
    /// A lead at the battle's start.
    Lead,
    /// The player chose it (a voluntary switch).
    Chosen,
    /// The replacement for a mon that fainted (a forced, FREE switch).
    Replacement { fainted: String },
    /// Baton Pass: the passer and what the entrant received.
    BatonPass { passer: String, boosts: Vec<(&'static str, i32)>, volatiles: Vec<String> },
}

/// The CHOICE behind a denied action, as ONE side may know it. 🚨 The information boundary is a TYPE:
/// a viewer knows its own choice (it sent it) and only THAT the opponent was denied — [`Choice::Opp`]
/// carries no data, so no code path can put the opponent's chosen move into a viewer's record.
#[derive(Debug, Clone, PartialEq)]
pub enum Choice {
    /// Our own choice token (`move 2`, `switch 3`), when the caller told the stream what it sent.
    Own(Option<String>),
    /// The opponent chose SOMETHING; what is not this side's to know.
    Opp,
}

/// Why an action that was CHOSEN never happened.
#[derive(Debug, Clone, PartialEq)]
pub enum DenialWhy {
    /// The actor fainted before its turn came: outsped and KOed, Explosion / Self-Destruct, a
    /// recoil-free KO; `by` is the action that KOed it (its mover and move).
    FaintedFirst { cause: Cause, by: Option<(Mon, String)> },
    /// The actor did NOT faint, but a faint earlier in the turn CUT the turn: in gen-3 singles any
    /// faint cancels every remaining queued action (`faintMessages` → `queue.cancelAction` over all
    /// actives, `sim/battle.ts:2606-2616`; the port's `turn/switch.rs`), so a faster Explosion,
    /// a recoil self-KO or a KO of anyone denies every actor still waiting. `by_faint` is the first
    /// faint of the turn and its cause.
    TurnCut { by_faint: Mon, cause: Cause },
}

/// What one action WAS.
#[derive(Debug, Clone, PartialEq)]
pub enum ActionKind {
    Move {
        user: Mon,
        id: String,
        /// The CALLER when a move called this one (Sleep Talk / Metronome / Mirror Move / Assist /
        /// Nature Power / Snatch …); the same id on a Pursuit-on-switch strike (`pursuit_on_switch`).
        called_by: Option<String>,
        target: Option<Mon>,
        /// Pursuit striking a mon that is switching out.
        pursuit_on_switch: bool,
        /// A locked continuation (`[from] lockedmove`: Thrash / Outrage / Petal Dance / a charged
        /// move's release).
        locked: bool,
        /// `[still]` (a charge turn's announcement with no animation, or a miss suffix).
        still: bool,
    },
    Switch { to: Mon, out: Option<String>, entry: Entry },
    /// A FORCED switch (Roar / Whirlwind): who forced it and with what.
    Drag { to: Mon, out: Option<String>, by: Option<Mon>, by_move: Option<String> },
    /// The mon could not act — a DENIED action, REFUSED: the reason (`par`, `slp`, `frz`, `flinch`,
    /// `recharge`, `Focus Punch`, `nopp`, `Taunt`, `Disable`, `damp` …), the move it was prevented
    /// from, who blocked it, and the choice as this side knows it.
    Cant { mon: Mon, reason: String, move_id: Option<String>, blocked_by: Option<Mon>, choice: Choice,
           /// The same mon MOVED later this turn (Sleep Talk / Snore through sleep): the line is the
           /// truth, but the action was not denied. A `Cant` with `then_moved == false` IS a denial.
           then_moved: bool },
    /// A DENIED action that left no line of its own: the actor was chosen for this turn and never
    /// acted (placed right after the action that denied it, in move order).
    Denied { actor: Mon, why: DenialWhy, choice: Choice },
    /// The server REFUSED a choice (`|error|`): trapped, disabled, …
    Refused { reason: Option<String> },
    /// The end-of-turn residual block.
    Residual,
    /// Anything before the first action of a window (a turn's start, a lead's entry abilities).
    Framing,
}

/// The attributed CAUSE of an effect.
#[derive(Debug, Clone, PartialEq)]
pub enum Cause {
    /// The action's own direct result (a move's hit, a switch's entry).
    Direct,
    Move(String),
    Item(String),
    Ability(String),
    /// Spikes on entry, with the layer count the entrant met.
    Spikes { layers: u32 },
    Weather(String),
    /// Residual poison / toxic / burn.
    Status(String),
    Recoil,
    LeechSeed,
    Curse,
    Nightmare,
    Wish { wisher: Option<String> },
    Confusion,
    DestinyBond,
    PerishSong,
    SelfKo,
    Other(String),
}

/// What an effect did.
#[derive(Debug, Clone, PartialEq)]
pub enum What {
    Damage { amount: f64, hp_after: f64 },
    Heal { amount: f64, hp_after: f64 },
    SetHp { hp: f64 },
    /// Substitute took the hit (`|-activate|…|Substitute|[damage]`) or broke (`|-end|…|Substitute`).
    SubstituteHit { broke: bool },
    /// The action's target was LOST to Protect / Detect (`|-activate|<target>|move: Protect`) — a
    /// denial of the move's effect, with the protector.
    Blocked(String),
    Faint,
    Status(String),
    Cure(String),
    Boost { stat: String, n: i64 },
    ClearBoost(String),
    /// An item was revealed, consumed, removed or moved; `to` is the receiver of a Trick / Thief /
    /// Covet (`None` = lost / consumed / merely disclosed).
    Item { item: String, gone: bool, how: String, to: Option<Mon> },
    Ability(String),
    VolatileStart(String),
    VolatileEnd(String),
    Activate(String),
    SideStart { condition: String, layers: u32 },
    SideEnd(String),
    Weather(String),
    Crit,
    Miss,
    Fail,
    Effectiveness(f64),
    /// A charge turn's announcement (`|-prepare|`) and a recharge requirement (`|-mustrecharge|`).
    Prepare(String),
    MustRecharge,
    Transform,
    FormeChange,
    Other(String),
}

/// One effect of an action.
#[derive(Debug, Clone, PartialEq)]
pub struct Effect {
    pub on: Option<Mon>,
    pub what: What,
    pub cause: Cause,
    /// The source mon (`[of]`), when the line names one.
    pub of: Option<Mon>,
}

/// One action and everything it caused.
#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    pub turn: u32,
    pub kind: ActionKind,
    pub effects: Vec<Effect>,
    /// The engine's action scope of the line that opened it (step path; `None` when parsed).
    pub scope: Option<Scope>,
}

/// One side's decision window.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Window {
    pub actions: Vec<Action>,
}

/// Folds a side's lines into [`Window`]s. Carries across windows what the record needs from
/// earlier ones (the nickname → species map, the actives, the Spikes layers per side).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RecordBuilder {
    viewer: u8,
    turn: u32,
    names: Vec<((u8, String), String)>,
    active: [Option<String>; 2],
    /// A faint on this side since its last entry (the next entry is its FREE replacement).
    fainted_pending: [Option<String>; 2],
    /// Spikes layers on each side (absolute side index), from `|-sidestart|` / `|-sideend|`.
    spikes: [u32; 2],
    window: Window,
    /// The last damage cause per (absolute side, species) in the current window (the faint cause).
    last_damage: Vec<((u8, String), Cause)>,
    /// Our choice for the current turn (the move-boundary one; a re-choice after a refusal replaces it).
    own_choice: Option<String>,
    /// Each side's active at the current turn's start — the actors that were CHOSEN for it.
    turn_actors: [Option<String>; 2],
    /// Whether each side's turn actor has acted (moved, chose a switch, or was refused).
    acted: [bool; 2],
    /// A Baton Pass used by each side and not yet completed by its switch (the pass opens a
    /// decision, so the switch arrives in the NEXT window).
    baton_pending: [Option<String>; 2],
    /// Denials found at a faint, placed after the current action once it closes.
    pending_denials: Vec<Action>,
    /// The turn's first faint in its ACTION phase: (the action index holding it, the mon, its cause).
    first_faint: Option<(usize, Mon, Cause)>,
    /// The turn's action phase is over (the residual block opened, `|upkeep`, or a decision).
    actions_closed: bool,
}

fn rel(viewer: u8, side: u8) -> Rel {
    if viewer == side {
        Rel::Ours
    } else {
        Rel::Opp
    }
}

fn residual_cause(c: &LCause) -> bool {
    match c {
        LCause::Bare(s) => {
            let l = s.to_lowercase();
            matches!(l.as_str(), "psn" | "tox" | "brn" | "sandstorm" | "hail" | "leech seed" | "nightmare" | "curse")
        }
        LCause::Item(s) => s == "Leftovers",
        LCause::Move(s) => s == "Wish" || s == "Ingrain" || s == "Future Sight" || s == "Doom Desire",
        _ => false,
    }
}

impl RecordBuilder {
    pub fn new(viewer: usize) -> RecordBuilder {
        RecordBuilder { viewer: viewer as u8, ..Default::default() }
    }

    fn species_of(&self, side: u8, name: &str) -> String {
        self.names
            .iter()
            .find(|((s, n), _)| *s == side && n == name)
            .map(|(_, sp)| sp.clone())
            .unwrap_or_else(|| crate::core_events::to_id(name))
    }

    fn mon_of(&self, id: &crate::core_events::Ident) -> Mon {
        Mon { side: rel(self.viewer, id.side), species: self.species_of(id.side, &id.name) }
    }

    fn open(&mut self, kind: ActionKind, scope: Option<Scope>) {
        self.flush_denials();
        self.window.actions.push(Action { turn: self.turn, kind, effects: Vec::new(), scope });
    }

    fn flush_denials(&mut self) {
        let d = std::mem::take(&mut self.pending_denials);
        self.window.actions.extend(d);
    }

    fn cur(&mut self) -> &mut Action {
        if self.window.actions.is_empty() {
            self.open(ActionKind::Framing, None);
        }
        self.window.actions.last_mut().expect("an action")
    }

    fn cause_of(&self, line: &Line) -> Cause {
        let Some(c) = line.from_cause() else { return Cause::Direct };
        match c {
            LCause::Item(s) => Cause::Item(s.clone()),
            LCause::Ability(s) => Cause::Ability(s.clone()),
            LCause::Move(s) if s == "Wish" => Cause::Wish {
                wisher: line.fields.iter().find_map(|f| match f {
                    Field::Wisher(w) => Some(w.clone()),
                    _ => None,
                }),
            },
            LCause::Move(s) => Cause::Move(s.clone()),
            LCause::Bare(s) => match s.to_lowercase().as_str() {
                "spikes" => Cause::Spikes { layers: 0 },
                "sandstorm" | "hail" => Cause::Weather(s.clone()),
                "psn" | "tox" | "brn" => Cause::Status(s.to_lowercase()),
                "recoil" => Cause::Recoil,
                "leech seed" => Cause::LeechSeed,
                "curse" => Cause::Curse,
                "nightmare" => Cause::Nightmare,
                "confusion" => Cause::Confusion,
                _ => Cause::Other(s.clone()),
            },
        }
    }

    /// Fold ONE line of the side's stream. `before` is the side's board reading BEFORE the line.
    pub fn push(&mut self, ev: &CoreEvent, before: &BoardReading, scope: Option<Scope>) {
        let line = &ev.line;
        let readings: &[Reading] = &ev.readings;
        let first = |k: K| readings.iter().find(|r| r.kind == k);
        let on = line.ident(0).map(|i| self.mon_of(i));
        let of = line.of_source().map(|i| self.mon_of(i));
        match line.kw {
            Kw::Turn => {
                self.close_actions();
                self.close_turn();
                if let Some(Field::Text(t)) = line.field(0) {
                    self.turn = t.parse().unwrap_or(self.turn);
                }
                self.turn_actors = self.active.clone();
                self.acted = [false, false];
                self.own_choice = None;
                self.first_faint = None;
                self.actions_closed = false;
            }
            Kw::Win | Kw::Tie => {
                self.close_actions();
                self.close_turn();
            }
            Kw::Move => {
                let Some(id) = line.ident(0) else { return };
                let user = self.mon_of(id);
                let mv = first(K::Move);
                let mid = mv.and_then(|r| ev::move_id(r)).unwrap_or("").to_string();
                let called_by = mv.and_then(|r| ev::s(r, "from_move")).map(str::to_string).or_else(|| {
                    // a `[from] <Move>` on a move line naming a DIFFERENT move is a call
                    match line.from_cause() {
                        Some(LCause::Bare(s)) | Some(LCause::Move(s)) if crate::core_events::to_id(s) != mid && s != "lockedmove" => {
                            Some(crate::core_events::to_id(s))
                        }
                        _ => None,
                    }
                });
                let pursuit = matches!(line.from_cause(), Some(LCause::Bare(s)) | Some(LCause::Move(s)) if crate::core_events::to_id(s) == mid)
                    && mid == "pursuit";
                let locked = matches!(line.from_cause(), Some(LCause::Bare(s)) if s == "lockedmove");
                let target = line.ident(2).map(|i| self.mon_of(i));
                self.mark_acted(id.side, &user.species);
                let turn = self.turn;
                for a in self.window.actions.iter_mut().rev() {
                    if a.turn != turn {
                        break;
                    }
                    if let ActionKind::Cant { mon, then_moved, .. } = &mut a.kind {
                        if *mon == user {
                            *then_moved = true;
                        }
                    }
                }
                if mid == "batonpass" {
                    self.baton_pending[id.side as usize] = Some(user.species.clone());
                }
                self.open(
                    ActionKind::Move {
                        user,
                        id: mid,
                        called_by,
                        target,
                        pursuit_on_switch: pursuit,
                        locked,
                        still: line.has_tag("[still]"),
                    },
                    scope,
                );
                if line.has_tag("[miss]") {
                    self.cur().effects.push(Effect { on: None, what: What::Miss, cause: Cause::Direct, of: None });
                }
            }
            Kw::Switch | Kw::Drag => {
                let Some(id) = line.ident(0) else { return };
                let species = first(if line.kw == Kw::Switch { K::Switch } else { K::Drag })
                    .and_then(|r| ev::actor(r))
                    .map(str::to_string)
                    .unwrap_or_else(|| crate::core_events::to_id(&id.name));
                let key = (id.side, id.name.clone());
                match self.names.iter_mut().find(|(k, _)| *k == key) {
                    Some(slot) => slot.1 = species.clone(),
                    None => self.names.push((key, species.clone())),
                }
                let s = id.side as usize;
                let mut out = self.active[s].take();
                let mut out_fainted: Option<String> = None;
                let to = Mon { side: rel(self.viewer, id.side), species: species.clone() };
                if line.kw == Kw::Drag {
                    // the phazer: the latest move of the OTHER side this turn
                    let by = self.window.actions.iter().rev().find_map(|a| match &a.kind {
                        ActionKind::Move { user, id, .. } if a.turn == self.turn && user.side != to.side => {
                            Some((user.clone(), id.clone()))
                        }
                        _ => None,
                    });
                    self.open(
                        ActionKind::Drag { to, out, by: by.as_ref().map(|b| b.0.clone()), by_move: by.map(|b| b.1) },
                        scope,
                    );
                } else {
                    let entry = if self.turn == 0 {
                        Entry::Lead
                    } else if let Some(f) = self.fainted_pending[s].take() {
                        out_fainted = Some(f.clone());
                        Entry::Replacement { fainted: f }
                    } else if let Some(passer) = self.baton_pending[s].take() {
                        let (boosts, volatiles) = passed(before, id.side, Some(passer.as_str()));
                        Entry::BatonPass { passer, boosts, volatiles }
                    } else {
                        Entry::Chosen
                    };
                    if out.is_none() {
                        out = out_fainted;
                    }
                    if matches!(entry, Entry::Chosen | Entry::BatonPass { .. }) {
                        if let Some(o) = out.clone() {
                            self.mark_acted(id.side, &o);
                        }
                    }
                    self.open(ActionKind::Switch { to, out, entry }, scope);
                }
                self.fainted_pending[s] = None;
                self.active[s] = Some(species.clone());
                if self.spikes[s] > 0 {
                    // the entrant meets the side's Spikes (its chip, if any, follows as `-damage`)
                    let layers = self.spikes[s];
                    self.last_damage.retain(|(k, _)| *k != (id.side, species.clone()));
                    self.last_damage.push(((id.side, species), Cause::Spikes { layers }));
                }
            }
            Kw::Cant => {
                let (Some(m), Some(id)) = (on, line.ident(0)) else { return };
                let r = first(K::Cant);
                let reason = r.and_then(ev::reason).unwrap_or("").to_string();
                let move_id = r.and_then(ev::cant_move).map(str::to_string);
                // an ability-sourced cant (Damp) is filed against the HOLDER and names the blocked
                // mon in `[of]`; the refused actor is the blocked one
                let (actor, blocked_by) = match (&of, r.and_then(ev::blocked_actor)) {
                    (Some(o), Some(_)) => (o.clone(), Some(m)),
                    _ => (m, of),
                };
                let abs = if actor.side == rel(self.viewer, id.side) { id.side } else { 1 - id.side };
                self.mark_acted(abs, &actor.species);
                let choice = self.choice_for(abs);
                self.open(ActionKind::Cant { mon: actor, reason, move_id, blocked_by, choice, then_moved: false }, scope);
            }
            Kw::Error if first(K::ChoiceRejected).is_some() => {
                let reason = first(K::ChoiceRejected).and_then(ev::reason).map(str::to_string);
                self.open(ActionKind::Refused { reason }, scope);
            }
            Kw::Upkeep => self.close_actions(),
            _ => {
                let cause = self.cause_of(line);
                // The step path KNOWS the residual block (the engine's scope); a parsed stream
                // recognises it by its causes.
                let residual = match scope {
                    Some(sc) => sc == Scope::Residual,
                    None => line.from_cause().is_some_and(residual_cause) || (line.kw == Kw::Weather && line.has_tag("[upkeep]")),
                };
                if residual && !ev.readings.is_empty()
                    && !matches!(self.window.actions.last().map(|a| &a.kind), Some(ActionKind::Residual)) {
                    self.close_actions();
                    self.open(ActionKind::Residual, scope);
                }
                self.effects(ev, on, of, cause);
            }
        }
    }

    /// Our choice token for the coming action (see [`Choice`]).
    pub fn choose(&mut self, token: &str) {
        let refused = matches!(self.window.actions.last().map(|a| &a.kind), Some(ActionKind::Refused { .. }));
        if self.own_choice.is_none() || refused {
            self.own_choice = Some(token.to_string());
        }
    }

    /// The choice as THIS side knows it for absolute side `abs`.
    fn choice_for(&self, abs: u8) -> Choice {
        if abs == self.viewer {
            Choice::Own(self.own_choice.clone())
        } else {
            Choice::Opp
        }
    }

    fn mark_acted(&mut self, abs: u8, species: &str) {
        if self.turn_actors[abs as usize].as_deref() == Some(species) {
            self.acted[abs as usize] = true;
        }
    }

    /// The turn's ACTION phase is over: every turn actor that neither acted nor fainted, in a turn a
    /// faint CUT, was denied by that faint (the gen-3 cancel-all rule) — placed right after the
    /// action that holds the first faint (after any fainted-first denial of that same action).
    fn close_actions(&mut self) {
        if self.actions_closed || self.turn == 0 {
            return;
        }
        self.actions_closed = true;
        self.flush_denials();
        let Some((at, by, cause)) = self.first_faint.clone() else { return };
        for abs in 0..2u8 {
            let s = abs as usize;
            let Some(actor) = self.turn_actors[s].clone() else { continue };
            if self.acted[s] {
                continue;
            }
            self.acted[s] = true;
            let mut i = (at + 1).min(self.window.actions.len());
            while i < self.window.actions.len() && matches!(self.window.actions[i].kind, ActionKind::Denied { .. }) {
                i += 1;
            }
            let denied = Action {
                turn: self.turn,
                kind: ActionKind::Denied {
                    actor: Mon { side: rel(self.viewer, abs), species: actor },
                    why: DenialWhy::TurnCut { by_faint: by.clone(), cause: cause.clone() },
                    choice: self.choice_for(abs),
                },
                effects: Vec::new(),
                scope: None,
            };
            self.window.actions.insert(i, denied);
        }
    }

    /// A turn ends: nothing carries over but what the NEXT turn's actors are (set at `|turn|`).
    fn close_turn(&mut self) {
        self.flush_denials();
    }

    fn effects(&mut self, ev: &CoreEvent, on: Option<Mon>, of: Option<Mon>, cause: Cause) {
        let line = &ev.line;
        let abs = line.ident(0).map(|i| i.side);
        let push = |b: &mut RecordBuilder, what: What, cause: Cause| {
            b.cur().effects.push(Effect { on: on.clone(), what, cause, of: of.clone() });
        };
        for r in &ev.readings {
            let what = match r.kind {
                K::Damage => What::Damage { amount: ev::amount(r).unwrap_or(0.0), hp_after: ev::num(r, "hp_after").unwrap_or(0.0) },
                K::Heal => What::Heal { amount: ev::amount(r).unwrap_or(0.0), hp_after: ev::num(r, "hp_after").unwrap_or(0.0) },
                K::Sethp => What::SetHp { hp: ev::num(r, "hp").unwrap_or(0.0) },
                K::Faint => What::Faint,
                K::Status => What::Status(ev::status(r).unwrap_or("").to_string()),
                K::Curestatus => What::Cure(ev::status(r).unwrap_or("").to_string()),
                K::Boost | K::Unboost | K::Setboost => What::Boost {
                    stat: ev::stat(r).unwrap_or("").to_string(),
                    n: ev::amount(r).unwrap_or(0.0) as i64 * if r.kind == K::Unboost { -1 } else { 1 },
                },
                K::Clearboost => What::ClearBoost(ev::s(r, "op").unwrap_or("").to_string()),
                K::Item | K::Enditem => {
                    let how = ev::from_clause(r).unwrap_or("").to_string();
                    let moved = ["Trick", "Thief", "Covet", "Switcheroo"].iter().any(|w| how.contains(w));
                    let to = if moved && r.kind == K::Enditem { of.clone() } else { None };
                    What::Item { item: ev::item(r).unwrap_or("").to_string(), gone: r.kind == K::Enditem, how, to }
                }
                K::Ability => What::Ability(ev::s(r, "ability").unwrap_or("").to_string()),
                K::VolatileStart => {
                    let e = ev::s(r, "effect").unwrap_or("").to_string();
                    What::VolatileStart(e)
                }
                K::VolatileEnd => {
                    let e = ev::s(r, "effect").unwrap_or("").to_string();
                    if e.eq_ignore_ascii_case("substitute") {
                        What::SubstituteHit { broke: true }
                    } else {
                        What::VolatileEnd(e)
                    }
                }
                K::Activate => {
                    let e = ev::s(r, "effect").unwrap_or("").to_string();
                    let low = e.to_lowercase();
                    if low.contains("substitute") && line.has_tag("[damage]") {
                        What::SubstituteHit { broke: false }
                    } else if low == "protect" || low == "detect" || low == "move: protect" || low == "move: detect" {
                        What::Blocked(e)
                    } else {
                        What::Activate(e)
                    }
                }
                K::Side => {
                    let cond = ev::s(r, "condition").unwrap_or("").to_string();
                    let start = ev::s(r, "op") != Some("sideend");
                    let side_abs = match line.field(0) {
                        Some(Field::SideRef { side, .. }) => Some(*side as usize),
                        _ => None,
                    };
                    if cond.to_lowercase().contains("spikes") {
                        if let Some(s) = side_abs {
                            self.spikes[s] = if start { (self.spikes[s] + 1).min(3) } else { 0 };
                        }
                    }
                    let layers = side_abs.map_or(0, |s| self.spikes[s]);
                    if start {
                        What::SideStart { condition: cond, layers }
                    } else {
                        What::SideEnd(cond)
                    }
                }
                K::Weather => What::Weather(ev::s(r, "weather").unwrap_or("").to_string()),
                K::Crit => What::Crit,
                K::Miss => What::Miss,
                K::Fail => What::Fail,
                K::Immune | K::Resisted | K::Supereffective => What::Effectiveness(ev::multiplier(r).unwrap_or(1.0)),
                K::Prepare => What::Prepare(match line.field(1) {
                    Some(Field::Text(t)) => crate::core_events::to_id(t),
                    _ => String::new(),
                }),
                K::Mustrecharge => What::MustRecharge,
                K::Transform => What::Transform,
                K::Formechange => What::FormeChange,
                K::Move | K::Switch | K::Drag | K::Cant | K::ChoiceRejected => continue,
                _ => What::Other(r.kind.name().to_string()),
            };
            let mut cause = cause.clone();
            if let (Some(side), Some(sp)) = (abs, on.as_ref().map(|m| m.species.clone())) {
                match &what {
                    What::Damage { .. } => {
                        if let Cause::Spikes { .. } = cause {
                            cause = Cause::Spikes { layers: self.spikes[side as usize] };
                        }
                        self.last_damage.retain(|(k, _)| *k != (side, sp.clone()));
                        self.last_damage.push(((side, sp), cause.clone()));
                    }
                    What::Faint => {
                        cause = self.faint_cause(side, &sp);
                        // A turn ACTOR fainting before it acted, outside the residual block, is a
                        // DENIED action — placed right after the action that denied it.
                        let s = side as usize;
                        let residual = matches!(self.window.actions.last().map(|a| &a.kind), Some(ActionKind::Residual));
                        if self.turn > 0 && !residual && !self.actions_closed && self.first_faint.is_none() {
                            let at = self.window.actions.len().saturating_sub(1);
                            self.first_faint = Some((at, Mon { side: rel(self.viewer, side), species: sp.clone() }, cause.clone()));
                        }
                        if self.turn > 0 && !residual && !self.acted[s] && self.turn_actors[s].as_deref() == Some(sp.as_str()) {
                            self.acted[s] = true;
                            let by = match self.window.actions.last().map(|a| &a.kind) {
                                Some(ActionKind::Move { user, id, .. }) => Some((user.clone(), id.clone())),
                                _ => None,
                            };
                            self.pending_denials.push(Action {
                                turn: self.turn,
                                kind: ActionKind::Denied {
                                    actor: Mon { side: rel(self.viewer, side), species: sp.clone() },
                                    why: DenialWhy::FaintedFirst { cause: cause.clone(), by },
                                    choice: self.choice_for(side),
                                },
                                effects: Vec::new(),
                                scope: None,
                            });
                        }
                        self.fainted_pending[side as usize] = Some(sp.clone());
                        if self.active[side as usize].as_deref() == Some(sp.as_str()) {
                            self.active[side as usize] = None;
                        }
                    }
                    _ => {}
                }
            }
            push(self, what, cause);
        }
    }

    /// Why `species` (absolute side) fainted: its own self-KO move, a Destiny Bond / Perish count
    /// just announced, else the last damage it took in this window.
    fn faint_cause(&self, side: u8, species: &str) -> Cause {
        let me = rel(self.viewer, side);
        for a in self.window.actions.iter().rev() {
            if let ActionKind::Move { user, id, .. } = &a.kind {
                if user.side == me && user.species == species && (id == "explosion" || id == "selfdestruct") {
                    return Cause::SelfKo;
                }
            }
            for e in a.effects.iter().rev() {
                match (&e.what, &e.on) {
                    (What::Activate(x), _) if x.to_lowercase().contains("destinybond") || x.to_lowercase().contains("destiny bond") => {
                        return Cause::DestinyBond
                    }
                    (What::VolatileStart(x), Some(m)) if m.side == me && m.species == species && x.to_lowercase().contains("perish0") => {
                        return Cause::PerishSong
                    }
                    _ => {}
                }
            }
        }
        self.last_damage
            .iter()
            .rev()
            .find(|(k, _)| k.0 == side && k.1 == species)
            .map_or(Cause::Direct, |(_, c)| c.clone())
    }

    /// The window still OPEN (the actions since the side's last decision) — a battle's last turn
    /// has no decision to close it.
    pub fn current(&self) -> &Window {
        &self.window
    }

    /// Close the window at a decision: the actions since the previous one.
    pub fn take(&mut self) -> Window {
        // A decision INSIDE a turn ends its action phase only when a faint cut it (the forced
        // replacement); the move request at a turn's START, or a Baton Pass's mid-turn switch
        // decision (the turn goes on), must not.
        if self.first_faint.is_some() {
            self.close_actions();
        }
        self.flush_denials();
        self.last_damage.clear();
        std::mem::take(&mut self.window)
    }
}

/// What a Baton Pass carries from the passer (absolute side `side`, species `passer`), as this side
/// reads the passer at the switch: its nonzero boosts and its passable volatiles.
fn passed(before: &BoardReading, side: u8, passer: Option<&str>) -> (Vec<(&'static str, i32)>, Vec<String>) {
    const PASSABLE: [&str; 12] = [
        "substitute", "confusion", "focusenergy", "leechseed", "curse", "ingrain", "perish3", "perish2", "perish1",
        "lockon", "meanlook", "mindreader",
    ];
    let team = if side == before.role { &before.team } else { &before.opp };
    let Some(p) = passer else { return (Vec::new(), Vec::new()) };
    let Some((_, m)) = team.iter().find(|(_, m)| m.species == p) else { return (Vec::new(), Vec::new()) };
    let boosts = crate::present::mon::BOOST_KEYS
        .iter()
        .zip(m.boosts.iter())
        .filter(|(_, v)| **v != 0)
        .map(|(k, v)| (*k, *v))
        .collect();
    let vols = m
        .effects
        .iter()
        .map(|(id, _)| dex::effect_live_id(*id))
        .filter(|n| PASSABLE.contains(&n.as_str()))
        .collect();
    (boosts, vols)
}

// ---------------------------------------------------------------------------------------- JSON

fn mon_json(out: &mut String, m: &Option<Mon>) {
    match m {
        None => out.push_str("null"),
        Some(m) => {
            out.push('[');
            json_out::str_into(out, m.side.as_str());
            out.push(',');
            json_out::str_into(out, &m.species);
            out.push(']');
        }
    }
}

/// `,"choice":{"own":<token|null>}` or `,"choice":"opp"` — the opponent's carries NO choice.
fn choice_json(out: &mut String, c: &Choice) {
    match c {
        Choice::Own(t) => {
            out.push_str(",\"choice\":{\"own\":");
            json_out::opt_str_into(out, t.as_deref());
            out.push('}');
        }
        Choice::Opp => out.push_str(",\"choice\":\"opp\""),
    }
}

fn cause_json(out: &mut String, c: &Cause) {
    let (k, v): (&str, Option<String>) = match c {
        Cause::Direct => ("direct", None),
        Cause::Move(s) => ("move", Some(s.clone())),
        Cause::Item(s) => ("item", Some(s.clone())),
        Cause::Ability(s) => ("ability", Some(s.clone())),
        Cause::Spikes { layers } => ("spikes", Some(layers.to_string())),
        Cause::Weather(s) => ("weather", Some(s.clone())),
        Cause::Status(s) => ("status", Some(s.clone())),
        Cause::Recoil => ("recoil", None),
        Cause::LeechSeed => ("leechseed", None),
        Cause::Curse => ("curse", None),
        Cause::Nightmare => ("nightmare", None),
        Cause::Wish { wisher } => ("wish", wisher.clone()),
        Cause::Confusion => ("confusion", None),
        Cause::DestinyBond => ("destinybond", None),
        Cause::PerishSong => ("perishsong", None),
        Cause::SelfKo => ("selfko", None),
        Cause::Other(s) => ("other", Some(s.clone())),
    };
    out.push('[');
    json_out::str_into(out, k);
    out.push(',');
    json_out::opt_str_into(out, v.as_deref());
    out.push(']');
}

impl Window {
    /// `[{turn, scope, kind, …, effects: [[on, what, detail, cause, of], …]}, …]`.
    pub fn json_into(&self, out: &mut String) {
        out.push('[');
        for (i, a) in self.actions.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("{{\"turn\":{},\"scope\":", a.turn));
            json_out::opt_str_into(out, a.scope.map(|s| s.label()).as_deref());
            out.push_str(",\"kind\":");
            match &a.kind {
                ActionKind::Move { user, id, called_by, target, pursuit_on_switch, locked, still } => {
                    out.push_str("\"move\",\"user\":");
                    mon_json(out, &Some(user.clone()));
                    out.push_str(",\"id\":");
                    json_out::str_into(out, id);
                    out.push_str(",\"called_by\":");
                    json_out::opt_str_into(out, called_by.as_deref());
                    out.push_str(",\"target\":");
                    mon_json(out, target);
                    out.push_str(&format!(",\"pursuit_on_switch\":{pursuit_on_switch},\"locked\":{locked},\"still\":{still}"));
                }
                ActionKind::Switch { to, out: o, entry } => {
                    out.push_str("\"switch\",\"to\":");
                    mon_json(out, &Some(to.clone()));
                    out.push_str(",\"out\":");
                    json_out::opt_str_into(out, o.as_deref());
                    out.push_str(",\"entry\":");
                    match entry {
                        Entry::Lead => out.push_str("\"lead\""),
                        Entry::Chosen => out.push_str("\"chosen\""),
                        Entry::Replacement { fainted } => {
                            out.push_str("{\"replacement\":");
                            json_out::str_into(out, fainted);
                            out.push('}');
                        }
                        Entry::BatonPass { passer, boosts, volatiles } => {
                            out.push_str("{\"batonpass\":");
                            json_out::str_into(out, passer);
                            out.push_str(",\"boosts\":[");
                            for (j, (k, v)) in boosts.iter().enumerate() {
                                if j > 0 {
                                    out.push(',');
                                }
                                out.push_str(&format!("[\"{k}\",{v}]"));
                            }
                            out.push_str("],\"volatiles\":[");
                            for (j, v) in volatiles.iter().enumerate() {
                                if j > 0 {
                                    out.push(',');
                                }
                                json_out::str_into(out, v);
                            }
                            out.push_str("]}");
                        }
                    }
                }
                ActionKind::Drag { to, out: o, by, by_move } => {
                    out.push_str("\"drag\",\"to\":");
                    mon_json(out, &Some(to.clone()));
                    out.push_str(",\"out\":");
                    json_out::opt_str_into(out, o.as_deref());
                    out.push_str(",\"by\":");
                    mon_json(out, by);
                    out.push_str(",\"by_move\":");
                    json_out::opt_str_into(out, by_move.as_deref());
                }
                ActionKind::Cant { mon, reason, move_id, blocked_by, choice, then_moved } => {
                    out.push_str("\"cant\",\"mon\":");
                    mon_json(out, &Some(mon.clone()));
                    out.push_str(",\"reason\":");
                    json_out::str_into(out, reason);
                    out.push_str(",\"move_id\":");
                    json_out::opt_str_into(out, move_id.as_deref());
                    out.push_str(",\"blocked_by\":");
                    mon_json(out, blocked_by);
                    choice_json(out, choice);
                    out.push_str(&format!(",\"then_moved\":{then_moved}"));
                }
                ActionKind::Denied { actor, why: DenialWhy::TurnCut { by_faint, cause }, choice } => {
                    out.push_str("\"denied\",\"actor\":");
                    mon_json(out, &Some(actor.clone()));
                    out.push_str(",\"why\":\"turn_cut\",\"cause\":");
                    cause_json(out, cause);
                    out.push_str(",\"by_faint\":");
                    mon_json(out, &Some(by_faint.clone()));
                    choice_json(out, choice);
                }
                ActionKind::Denied { actor, why: DenialWhy::FaintedFirst { cause, by }, choice } => {
                    out.push_str("\"denied\",\"actor\":");
                    mon_json(out, &Some(actor.clone()));
                    out.push_str(",\"why\":\"fainted_first\",\"cause\":");
                    cause_json(out, cause);
                    out.push_str(",\"by\":");
                    match by {
                        Some((m, mv)) => {
                            out.push('[');
                            mon_json(out, &Some(m.clone()));
                            out.push(',');
                            json_out::str_into(out, mv);
                            out.push(']');
                        }
                        None => out.push_str("null"),
                    }
                    choice_json(out, choice);
                }
                ActionKind::Refused { reason } => {
                    out.push_str("\"refused\",\"reason\":");
                    json_out::opt_str_into(out, reason.as_deref());
                }
                ActionKind::Residual => out.push_str("\"residual\""),
                ActionKind::Framing => out.push_str("\"framing\""),
            }
            out.push_str(",\"effects\":[");
            for (j, e) in a.effects.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push('[');
                mon_json(out, &e.on);
                out.push(',');
                let (w, d): (&str, String) = match &e.what {
                    What::Damage { amount, hp_after } => ("damage", format!("[{amount:?},{hp_after:?}]")),
                    What::Heal { amount, hp_after } => ("heal", format!("[{amount:?},{hp_after:?}]")),
                    What::SetHp { hp } => ("sethp", format!("{hp:?}")),
                    What::SubstituteHit { broke } => ("substitute", format!("{broke}")),
                    What::Blocked(s) => ("blocked", format!("{s:?}")),
                    What::Faint => ("faint", "null".into()),
                    What::Status(s) => ("status", format!("{s:?}")),
                    What::Cure(s) => ("cure", format!("{s:?}")),
                    What::Boost { stat, n } => ("boost", format!("[{stat:?},{n}]")),
                    What::ClearBoost(s) => ("clearboost", format!("{s:?}")),
                    What::Item { item, gone, how, to } => {
                        let mut t = String::new();
                        mon_json(&mut t, to);
                        ("item", format!("[{item:?},{gone},{how:?},{t}]"))
                    }
                    What::Ability(s) => ("ability", format!("{s:?}")),
                    What::VolatileStart(s) => ("volatile_start", format!("{s:?}")),
                    What::VolatileEnd(s) => ("volatile_end", format!("{s:?}")),
                    What::Activate(s) => ("activate", format!("{s:?}")),
                    What::SideStart { condition, layers } => ("side_start", format!("[{condition:?},{layers}]")),
                    What::SideEnd(s) => ("side_end", format!("{s:?}")),
                    What::Weather(s) => ("weather", format!("{s:?}")),
                    What::Crit => ("crit", "null".into()),
                    What::Miss => ("miss", "null".into()),
                    What::Fail => ("fail", "null".into()),
                    What::Effectiveness(x) => ("effectiveness", format!("{x:?}")),
                    What::Prepare(s) => ("prepare", format!("{s:?}")),
                    What::MustRecharge => ("mustrecharge", "null".into()),
                    What::Transform => ("transform", "null".into()),
                    What::FormeChange => ("formechange", "null".into()),
                    What::Other(s) => ("other", format!("{s:?}")),
                };
                json_out::str_into(out, w);
                out.push(',');
                out.push_str(&d);
                out.push(',');
                cause_json(out, &e.cause);
                out.push(',');
                mon_json(out, &e.of);
                out.push(']');
            }
            out.push_str("]}");
        }
        out.push(']');
    }
}
