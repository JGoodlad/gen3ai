//! The per-decision TRACKERS as state on the version (`gen3_core_trackers_v1`, the Rust Core
//! Program's M3; `designs/rust_sim/trackers.md`).
//!
//! `agents.training.episode_tracker.EpisodeTracker`'s per-decision bookkeeping —
//! `record_context` (the slot registries, the context snapshot, the Hidden-Power belief's two
//! feeders) and `advance_window` (the progress clock, recency, pair history, the 32-row event
//! window) — plus the whole-log wish and sleep folds, the α/β opponent-intent LABEL and the
//! REWARD (the win indicator alone: the shaped terms are not ported, program M3), folded from ONE
//! side's stream at every decision that side takes. A fork shares its parent's trackers (`Arc`) and
//! pays for a copy only when its own transition opens a decision.
//!
//! **What is NOT here.** `TurnDelta` is not a first-class structure: the frozen Python layout
//! survives only as [`delta::DeltaProjection`], the fields its two live consumers read (the clock,
//! the label). What happened in a window, in order and with attribution, is the NATIVE record
//! ([`record`]). The choice-band belief (`ChoiceBandTracker`) is not ported: it has no production
//! reader (a no-op since it landed, `9a37b712`) — a FINDING, not a gap.

pub mod clock;
pub mod delta;
pub mod ev;
pub mod history;
pub mod hp_belief;
pub mod record;
pub mod turnview;

use crate::core_error::CoreResult;
use crate::core_events::json_out;
use crate::core_events::{to_id, Reading, Rel};
use crate::dex::Dex;
use crate::present::{LegalActions, OneSidedView};

use clock::{ClockConfig, ProgressClock};
use delta::{Ctx, DeltaProjection, Slots};
use history::{EventWindow, PairHistory, Recency, SleepFold, WishFold};
use hp_belief::HpBelief;

/// `opp_intent_labels.KIND_*`.
pub const KIND_MOVE: u8 = 0;
pub const KIND_SWITCH: u8 = 1;
pub const KIND_UNKNOWN: u8 = 2;

/// The α/β label for the decision the window CLOSES — `build_opp_intent_label(delta, …)` in ids;
/// the num lookups (`_move_num`, the privileged typed-Hidden-Power num, `_species_num`) are the
/// encoder's tables and cross at M4.
#[derive(Debug, Clone, PartialEq)]
pub struct IntentLabel {
    pub kind: u8,
    pub move_id: Option<String>,
    pub switch_species: Option<String>,
    /// The switch-in's slot in the opponent team list at the PREVIOUS decision (β's frame), `None`
    /// when it was not yet revealed there.
    pub switch_slot: Option<usize>,
    /// The mon that made the decision (`delta.opp_prev_active`, the typed-HP lookup's key) — on a
    /// MOVE label only.
    pub attacker: Option<String>,
}

impl IntentLabel {
    fn unknown() -> IntentLabel {
        IntentLabel { kind: KIND_UNKNOWN, move_id: None, switch_species: None, switch_slot: None, attacker: None }
    }

    /// `build_opp_intent_label(delta, move_num_of, opp_slot_of_species)`, ids only.
    pub fn build(d: &DeltaProjection, prev_frame: &[String]) -> IntentLabel {
        if d.phase_is_forced_switch || d.opp_choice_overridden {
            return Self::unknown();
        }
        // a CALLED move's choice is its CALLER (L4)
        let move_id = d
            .opp_called_via
            .as_deref()
            .filter(|m| !m.is_empty())
            .or_else(|| d.opp_resolved_move_id().filter(|m| !m.is_empty()));
        if let Some(sw) = d.opp_switch_to.as_deref().filter(|s| !s.is_empty()) {
            // a phaze, a same-window replacement, a DRAG (L1 / L2) or a straddling replacement (L3)
            if move_id.is_some() || d.opp_fainted || d.opp_dragged || d.opp_switch_is_replacement {
                return Self::unknown();
            }
            let key = to_id(sw);
            return IntentLabel {
                kind: KIND_SWITCH,
                move_id: None,
                switch_species: Some(sw.to_string()),
                switch_slot: prev_frame.iter().position(|s| to_id(s) == key),
                attacker: None,
            };
        }
        match move_id {
            Some(m) => IntentLabel {
                kind: KIND_MOVE,
                move_id: Some(m.to_string()),
                switch_species: None,
                switch_slot: None,
                attacker: d.opp_prev_active.clone(),
            },
            None => Self::unknown(),
        }
    }

    pub fn json_into(&self, out: &mut String) {
        out.push_str(&format!("{{\"kind\":{},\"move_id\":", self.kind));
        json_out::opt_str_into(out, self.move_id.as_deref());
        out.push_str(",\"switch_species\":");
        json_out::opt_str_into(out, self.switch_species.as_deref());
        out.push_str(",\"switch_slot\":");
        match self.switch_slot {
            Some(i) => out.push_str(&i.to_string()),
            None => out.push_str("null"),
        }
        out.push_str(",\"attacker\":");
        json_out::opt_str_into(out, self.attacker.as_deref());
        out.push('}');
    }
}

/// One side's tracker state at its latest decision.
#[derive(Debug, Clone, PartialEq)]
pub struct SideTrackers {
    /// Decisions recorded so far.
    pub decisions: u32,
    pub our_slots: Slots,
    pub opp_slots: Slots,
    /// `history[-1]` (this decision's context) and `history[-2]`.
    pub last: Option<Ctx>,
    pub prev: Option<Ctx>,
    pub hp: HpBelief,
    pub clock: ProgressClock,
    pub recency: Recency,
    pub pair: PairHistory,
    pub window: EventWindow,
    pub wish: WishFold,
    pub sleep: SleepFold,
    /// At the latest decision: the projection its window folded to, the label, the wish flags.
    pub delta: Option<DeltaProjection>,
    pub label: Option<IntentLabel>,
    pub wish_pending: [bool; 2],
    /// The opponent team list (species, reveal order) at the latest decision — the NEXT label's
    /// β frame.
    pub opp_frame: Vec<String>,
    /// The frame the latest label was resolved against.
    prev_frame: Vec<String>,
}

impl SideTrackers {
    pub fn new(cfg: ClockConfig) -> SideTrackers {
        SideTrackers {
            decisions: 0,
            our_slots: Slots::default(),
            opp_slots: Slots::default(),
            last: None,
            prev: None,
            hp: HpBelief::default(),
            clock: ProgressClock::new(cfg),
            recency: Recency::default(),
            pair: PairHistory::new(),
            window: EventWindow::default(),
            wish: WishFold::default(),
            sleep: SleepFold::default(),
            delta: None,
            label: None,
            wish_pending: [false, false],
            opp_frame: Vec::new(),
            prev_frame: Vec::new(),
        }
    }

    /// A DECISION of this side over `pending` (the readings since its previous decision):
    /// `record_context` then `advance_window`, exactly the live cadence (`Gen3Env.embed_battle`:
    /// `record(battle, mask, legal)` → `update_progress_clock(battle, legal)`).
    pub fn decide(&mut self, live: &OneSidedView, legal: Option<&LegalActions>, pending: &[Reading],
                  opp_last_damaging: Option<turnview::DamagingMove>, dex: &Dex,
                  attempted_switch: Option<&str>) -> CoreResult<()> {
        for r in pending {
            self.wish.fold(r);
            self.sleep.fold(r);
        }
        let window: Vec<&Reading> = pending.iter().collect();
        // --- record_context ---
        let ctx = Ctx::build(live, legal, &mut self.our_slots, &mut self.opp_slots, opp_last_damaging)?;
        self.hp.maybe_observe(dex, live, ctx.forced_switch, ctx.opp_last_damaging.as_ref())?;
        self.hp.scan_no_hp(live);
        self.prev = self.last.take();
        self.last = Some(ctx);
        // --- advance_window ---
        let d = match (&self.prev, &self.last) {
            (Some(p), Some(c)) => DeltaProjection::build(p, c, &window),
            _ => DeltaProjection::empty(),
        };
        self.clock.update(&d, live, legal, dex);
        // The FIRST decision's tracker window is EMPTY (no cursor recorded yet — the Python resync
        // takes an empty window rather than replaying the battle's opening into the counters).
        let win: &[&Reading] = if self.decisions == 0 { &[] } else { &window };
        let turn = live.turn as i64;
        let species = |side: &crate::present::view::SideView, alive_only: bool| -> Option<String> {
            side.active.map(|i| &side.mons[i]).filter(|m| !(alive_only && m.fainted)).map(|m| m.species.clone())
        };
        self.recency.update(turn, win, species(&live.ours, false).as_deref(), species(&live.opp, false).as_deref());
        let (oa, pa) = (species(&live.ours, true), species(&live.opp, true));
        self.pair.update(turn, win, oa.as_deref(), pa.as_deref(), dex);
        self.window.update(turn, win, oa.as_deref(), pa.as_deref(), attempted_switch)?;
        // --- the label (for the decision this window closes) and the obs-side wish flags ---
        self.prev_frame = std::mem::take(&mut self.opp_frame);
        self.label = Some(IntentLabel::build(&d, &self.prev_frame));
        self.opp_frame = live.opp.mons.iter().map(|m| m.species.clone()).collect();
        self.wish_pending = self.wish.pending(turn);
        self.delta = Some(d);
        self.decisions += 1;
        Ok(())
    }

    /// The whole tracker state (slice T's comparison unit).
    pub fn json(&self) -> String {
        let mut o = format!("{{\"decisions\":{},\"our_slots\":", self.decisions);
        self.our_slots.json_into(&mut o);
        o.push_str(",\"opp_slots\":");
        self.opp_slots.json_into(&mut o);
        o.push_str(",\"hp\":");
        self.hp.json_into(&mut o);
        o.push_str(",\"clock\":");
        self.clock.json_into(&mut o);
        o.push_str(",\"clock_value\":");
        json_out::f64_into(&mut o, self.clock.value());
        o.push_str(",\"recency\":");
        self.recency.json_into(&mut o);
        o.push_str(",\"pair\":");
        self.pair.json_into(&mut o);
        o.push_str(",\"window\":");
        self.window.json_into(&mut o);
        o.push_str(&format!(",\"wish\":[{},{}],\"sleep\":", self.wish_pending[0], self.wish_pending[1]));
        self.sleep.json_into(&mut o);
        o.push_str(",\"delta\":");
        match &self.delta {
            Some(d) => d.json_into(&mut o),
            None => o.push_str("null"),
        }
        o.push_str(",\"label\":");
        match &self.label {
            Some(l) => l.json_into(&mut o),
            None => o.push_str("null"),
        }
        o.push('}');
        o
    }
}

/// The REWARD at a boundary — the WIN INDICATOR (`victory_value · 1{win}`, `victory_value` 1.0):
/// every win-prob-era run trains on `1 TERMINAL + 0 PBRS + 0 BIAS` (program M3).
pub fn reward(live: &OneSidedView) -> f64 {
    if live.won == Some(true) {
        1.0
    } else {
        0.0
    }
}

/// `Rel` of an absolute side from `viewer`'s seat.
pub fn rel(viewer: usize, side: usize) -> Rel {
    if viewer == side {
        Rel::Ours
    } else {
        Rel::Opp
    }
}

/// The process-wide gen-3 dex the trackers consult (move flags, the type chart). Loaded once, on
/// first use; a tracker-folding version never takes a `&Dex` so a fork's construction stays
/// argument-compatible with the transport's.
pub fn dex() -> &'static Dex {
    static D: std::sync::OnceLock<Dex> = std::sync::OnceLock::new();
    D.get_or_init(|| Dex::for_gen(3))
}

/// One decision a side took — what the version exposes for it.
#[derive(Debug, Clone)]
pub struct Decision {
    /// The side's stream line index of the `|request|` it decided on.
    pub line: usize,
    /// The native record of the window this decision closed — `None` on a stream built WITHOUT
    /// the record ([`TrackerState::without_record`]: a reader of the row alone, `sim_bridge`).
    pub window: Option<std::sync::Arc<record::Window>>,
    /// The reward of the transition into this decision (the win indicator).
    pub reward: f64,
    /// The side's view at the decision (`present()`), handed to the version's memo so it is
    /// computed once (the request is the side's last line of the transition, so it IS the view at
    /// the version's boundary).
    pub view: Option<crate::present::OneSidedView>,
}

/// The opt-in tracker half of a side's stream (`gen3_core_trackers_v1`). A fork CLONES this
/// struct: the tracker state is an `Arc` (shared, copied only when this fork's own transition
/// opens a decision), the record builder and the pending readings are the fork's own.
#[derive(Debug, Clone)]
pub struct TrackerState {
    pub trackers: std::sync::Arc<SideTrackers>,
    /// The native record's builder; `None` when the stream builds no record
    /// ([`Self::without_record`]) — the trackers, and so the row, never read it.
    pub record: Option<record::RecordBuilder>,
    pending: Vec<Reading>,
    /// The latest decision this stream took.
    pub last: Option<Decision>,
    /// The latest choice token this side sent (E4: a refused switch's target is resolved from it).
    last_choice: Option<String>,
}

impl TrackerState {
    pub fn new(viewer: usize, cfg: ClockConfig) -> TrackerState {
        TrackerState {
            trackers: std::sync::Arc::new(SideTrackers::new(cfg)),
            record: Some(record::RecordBuilder::new(viewer)),
            pending: Vec::new(),
            last: None,
            last_choice: None,
        }
    }

    /// Build NO native record (before the first line): every decision's `window` is `None`. The
    /// trackers, the label, the reward and the encoded row do not read the record, so they are
    /// unchanged — the shape of a consumer that ships only the row (`sim_bridge`'s core
    /// observation mode), which would otherwise build a record per line that nothing reads.
    pub fn without_record(mut self) -> TrackerState {
        self.record = None;
        self
    }

    /// Fold one line's event (before the board reading takes the line — `before` is the board as it
    /// stood, which a Baton Pass reads its passer from).
    pub fn observe(&mut self, ev: &crate::core_events::CoreEvent, before: &crate::present::BoardReading,
                   scope: Option<crate::core_events::Scope>) {
        if let Some(r) = self.record.as_mut() {
            r.push(ev, before, scope);
        }
        self.pending.extend(ev.readings.iter().cloned());
    }

    /// [`Self::observe`]'s record half (the caller hands the readings over with [`Self::pend`]).
    pub fn record_line(&mut self, ev: &crate::core_events::CoreEvent, before: &crate::present::BoardReading,
                       scope: Option<crate::core_events::Scope>) {
        if let Some(r) = self.record.as_mut() {
            r.push(ev, before, scope);
        }
    }

    /// [`Self::observe`]'s pending half, the readings MOVED in (a stream that keeps no event).
    pub fn pend(&mut self, readings: Vec<crate::core_events::Reading>) {
        self.pending.extend(readings);
    }

    /// Our own choice for the coming action — a DENIED own action keeps it (the opponent's never
    /// reaches this side, by type: `record::Denied`).
    pub fn choose(&mut self, token: &str) {
        if let Some(r) = self.record.as_mut() {
            r.choose(token);
        }
        self.last_choice = Some(token.to_string());
    }

    /// After the board took a `|request|` line (stream index `line`): if it opens a DECISION of this
    /// side (a non-empty, non-`wait` request on an unfinished battle with a legal action — the live
    /// player's dispatch, `rust_core_parity_views.decision_points`), fold the trackers.
    pub fn maybe_decide(&mut self, br: &crate::present::BoardReading, request_nonempty: bool, line: usize)
        -> CoreResult<()> {
        if !request_nonempty || br.wait || br.finished {
            return Ok(());
        }
        let Some(legal) = crate::present::legal_actions(br) else { return Ok(()) };
        if crate::present::mask(&legal).iter().all(|m| *m == 0) {
            return Ok(());
        }
        let view = crate::present::present(br)?;
        let pending = std::mem::take(&mut self.pending);
        let dm = br.last_damaging_move(1).map(|d| turnview::DamagingMove {
            user_species: Some(d.user_species.clone()),
            target_species: Some(d.target_species.clone()),
            target_status: d.target_status.map(|s| s.live().to_uppercase()),
            move_id: Some(d.move_id.clone()),
            effectiveness: Some(d.effectiveness),
        });
        // E4 (gen3_event_record_v2): the species our previous choice tried to switch to — the target
        // a refused switch aimed at, which the server's `|error|` does not name.
        let attempted = self.last_choice.as_deref().and_then(|t| attempted_switch_species(br, t));
        std::sync::Arc::make_mut(&mut self.trackers).decide(&view, Some(&legal), &pending, dm, dex(), attempted.as_deref())?;
        let window = self.record.as_mut().map(|r| std::sync::Arc::new(r.take()));
        let reward = reward(&view);
        self.last = Some(Decision { line, window, reward, view: Some(view) });
        Ok(())
    }
}

/// E4: the species a `switch …` choice token names, resolved against this side's reading — `switch N`
/// is the N-th mon of the latest request's `side.pokemon` (Showdown's 1-based order), `switch <name>`
/// the team mon of that nickname (or species). `None` for any other token.
pub fn attempted_switch_species(br: &crate::present::BoardReading, token: &str) -> Option<String> {
    let rest = token.trim().strip_prefix("switch ")?.trim();
    if let Ok(n) = rest.parse::<usize>() {
        let req = br.last_request.as_ref()?;
        let crate::core_events::jsonval::Val::Arr(mons) = req.get("side")?.get("pokemon")? else { return None };
        let ident = mons.get(n.checked_sub(1)?)?.str_at("ident")?;
        let name = ident.split_once(": ").map_or(ident, |(_, nm)| nm);
        return br.team.iter().find(|(_, m)| m.name.as_deref() == Some(name)).map(|(_, m)| m.species.clone());
    }
    br.team
        .iter()
        .find(|(_, m)| m.name.as_deref() == Some(rest))
        .or_else(|| br.team.iter().find(|(_, m)| m.species == crate::core_events::to_id(rest)))
        .map(|(_, m)| m.species.clone())
}
