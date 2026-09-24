//! [`BattleVersion`] — the Rust core's PERSISTENT battle state (`gen3_core_version_v1`, the Rust
//! Core Program's M2; design `designs/endstate/design_three_tier_environment.md` §3.1–3.2).
//!
//! A battle is a chain of immutable versions, one per decision BOUNDARY. Each holds:
//!
//! * **`parent: Option<Arc<BattleVersion>>`** — history is the parent chain; a fork is an
//!   `Arc::clone` of the handle, so K successors of one decision share their past;
//! * **the per-side STREAM state** ([`SideStream`]): the side's typed lines folded into M1's
//!   reading (the `CoreEvent`s) and into the poke-env reading of the board ([`BoardReading`]) — the
//!   ONLY input of the side's view;
//! * **`events`** — the per-side [`CoreEvent`]s of THIS transition;
//! * **the view per side**, computed on demand and memoized ([`BattleVersion::view`]);
//! * **the raw `|request|` per side** (inside the board reading: `last_request_text`), which legality
//!   is derived from ([`BattleVersion::legal`]);
//! * **the engine** — the omniscient battle the step path advances: an [`Engine`] ONLY (the
//!   battle, the turn loop, the open boundary, the typed requests — `gen3_core_engine_split_v1`),
//!   never the wire. It is the REFEREE: nothing in the view reads it ([`present`] takes no board);
//!   [`BattleVersion::audit`] checks the view against it.
//!
//! # Three origins, one state
//!
//! * **step** ([`Origin::Step`]) — the simulator's own transition, a FORK: an engine CLONE is
//!   wrapped in a fresh transport ([`BattleVersion::fork_session`]), driven, and each side's NEW
//!   lines are folded TYPED AT THE SOURCE (the typed shortcut the Rust Core Program's §6c licenses
//!   by `parse(emit(step)) == step`); the child keeps the engine and drops the transport;
//! * **observe** ([`Origin::Observed`]) — a LINEAR replay: the caller drives ONE session and the
//!   chain folds what it shipped, holding no engine (the caller's session is the referee);
//! * **parse** ([`Origin::Parse`]) — one side's protocol TEXT, all a real server sends:
//!   [`BattleVersion::parse_root`] and [`BattleVersion::parse_step`] fold `Line::parse` of each
//!   line into the SAME [`SideStream`]. A parse-built version has one side and no engine (its
//!   omniscient board is partial by construction — the design's §6b).
//!
//! The gate is that the two agree version by version: every side's stream state (the whole
//! reading board), its transition events and its view. `core_events --views` runs it on every
//! corpus battle (`designs/rust_sim/present.md`).

use std::sync::{Arc, OnceLock};
use crate::core_error::{fault, malformed, CoreError, CoreResult};

use crate::bridge::{BridgeSession, Cmd};
use crate::engine::Engine;
use crate::state::BattleState;
use crate::core_events::parse::OwnerScan;
use crate::core_events::reading::Reader;
use crate::core_events::{is_outcome, CoreEvent, Field, Kw, Line, Scope};
use crate::trackers::clock::ClockConfig;
use crate::trackers::{Decision, TrackerState};
use crate::dex::Dex;
use crate::present::{check_view, legal_actions, present, Audit, LegalActions, OneSidedView, BoardReading};

type R<T> = CoreResult<T>;

/// One side's stream state: its typed lines folded, in order, into the reading of the events
/// (M1's [`Reader`] + the outcome-owner scan) and into the reading of the board ([`BoardReading`]).
#[derive(Debug, Clone)]
pub struct SideStream {
    pub board_reading: BoardReading,
    reader: Reader,
    owners: OwnerScan,
    /// Lines of this side folded so far, whole battle.
    pub lines: usize,
    /// The per-decision TRACKERS and the native record (`gen3_core_trackers_v1`, opt-in:
    /// [`SideStream::with_trackers`]). A fork clones the handle; the state is shared until the
    /// fork's own transition opens a decision.
    pub trk: Option<TrackerState>,
}

impl SideStream {
    /// A fresh stream for `viewer` ("p1" = 0), whose player is `username` and — when known —
    /// fights with `packed_team` (the source of our own spread in gen 3).
    pub fn new(viewer: usize, username: &str, packed_team: Option<&str>) -> R<SideStream> {
        Ok(SideStream {
            board_reading: BoardReading::new(viewer, username, packed_team)?,
            reader: Reader::new(viewer),
            owners: OwnerScan::default(),
            lines: 0,
            trk: None,
        })
    }

    /// Turn the trackers ON for this stream (before its first line).
    pub fn with_trackers(mut self, cfg: ClockConfig) -> SideStream {
        let viewer = self.board_reading.viewer as usize;
        self.trk = Some(TrackerState::new(viewer, cfg));
        self
    }

    /// Fold ONE typed line. `scope` is the engine's action scope (the step path's owner truth);
    /// the parse path passes `None` and the owner comes from line order.
    pub fn fold(&mut self, line: Line, src: Option<u32>, scope: Option<Scope>) -> R<CoreEvent> {
        let by_order = self.owners.step(&line);
        let readings = self.reader.feed(&line)?;
        let owner = if is_outcome(line.kw) {
            match scope {
                Some(s) => s.move_side(),
                None => by_order,
            }
        } else {
            None
        };
        let request = line.kw == Kw::Request;
        let request_nonempty = request && matches!(line.field(0), Some(Field::Text(t)) if !t.is_empty());
        let ev = CoreEvent { idx: self.lines as u32, line, src, owner, readings };
        if let Some(t) = self.trk.as_mut() {
            t.observe(&ev, &self.board_reading, scope);
        }
        self.board_reading.feed(&ev.line)?;
        if request {
            if let Some(t) = self.trk.as_mut() {
                t.maybe_decide(&self.board_reading, request_nonempty, self.lines)?;
            }
        }
        self.lines += 1;
        Ok(ev)
    }

    /// Fold ONE line of protocol TEXT (the parse path).
    pub fn fold_text(&mut self, text: &str) -> R<CoreEvent> {
        let line = Line::parse(text).map_err(|e| CoreError::from(e).context(format!("line {} {text:?}: ", self.lines)))?;
        self.fold(line, None, None)
    }

    /// The side's view (`present`).
    pub fn view(&self) -> R<OneSidedView> {
        present(&self.board_reading)
    }
}

/// How a version came to be — which decides what it can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Built by the simulator's own step and OWNING its engine: a fork root or a fork's child
    /// ([`BattleVersion::root`], [`BattleVersion::step`], [`BattleVersion::child`]). Can step.
    Step,
    /// Built by OBSERVING a session the caller drives ([`BattleVersion::observe_root`] /
    /// [`BattleVersion::observe`]) — a linear replay (the parity harness). It holds no engine:
    /// the caller's session is the referee ([`BattleVersion::audit_on`]).
    Observed,
    /// Built from ONE side's protocol text — what a real server sends. No engine, one side.
    Parse,
}

/// One immutable decision boundary of a battle.
pub struct BattleVersion {
    parent: Option<Arc<BattleVersion>>,
    origin: Origin,
    /// The referee and the stepper (a [`Origin::Step`] version only). Never read by a view. The
    /// ENGINE alone (`gen3_core_engine_split_v1`): a fork clones the battle, not the wire.
    engine: Option<Engine>,
    streams: [Option<SideStream>; 2],
    events: [Vec<CoreEvent>; 2],
    views: [OnceLock<R<OneSidedView>>; 2],
    /// Lines of the OBSERVED session's chunk list each side has folded ([`Origin::Observed`]
    /// only; 0 otherwise — a fork's transport starts empty, so it copies the board, never the
    /// battle's text).
    cursor: [usize; 2],
}

impl BattleVersion {
    fn new(parent: Option<Arc<BattleVersion>>, origin: Origin, engine: Option<Engine>, mut streams: [Option<SideStream>; 2],
           events: [Vec<CoreEvent>; 2], cursor: [usize; 2]) -> BattleVersion {
        let views = [OnceLock::new(), OnceLock::new()];
        // A decision taken at THIS boundary already computed the side's view — the memo adopts it.
        for side in 0..2 {
            let Some(s) = streams[side].as_mut() else { continue };
            let first = s.lines - events[side].len();
            if let Some(d) = s.trk.as_mut().and_then(|t| t.last.as_mut()) {
                if d.line >= first && d.line + 1 == s.lines {
                    if let Some(v) = d.view.take() {
                        let _ = views[side].set(Ok(v));
                    }
                }
            }
        }
        BattleVersion { parent, origin, engine, streams, events, views, cursor }
    }

    fn fresh_streams(names: [&str; 2], teams: [Option<&str>; 2], want: [bool; 2], trk: Option<ClockConfig>)
        -> R<[Option<SideStream>; 2]> {
        let mut out = [None, None];
        for side in 0..2 {
            if want[side] {
                let s = SideStream::new(side, names[side], teams[side])?;
                out[side] = Some(match trk {
                    Some(cfg) => s.with_trackers(cfg),
                    None => s,
                });
            }
        }
        Ok(out)
    }

    /// Fold every line `sess` shipped to each wanted side from `from` on, TYPED at the source.
    fn fold_typed(sess: &BridgeSession, streams: &mut [Option<SideStream>; 2], from: [usize; 2])
        -> R<[Vec<CoreEvent>; 2]> {
        let mut events: [Vec<CoreEvent>; 2] = [Vec::new(), Vec::new()];
        for side in 0..2 {
            let Some(s) = streams[side].as_mut() else { continue };
            for (line, src, scope) in sess.typed_side_lines(side, from[side]).map_err(fault)? {
                events[side].push(s.fold(line, src, scope)?);
            }
        }
        Ok(events)
    }

    /// Fold every line `sess` shipped to each wanted side from `from` on, from its TEXT.
    fn fold_text(sess: &BridgeSession, streams: &mut [Option<SideStream>; 2], from: [usize; 2])
        -> R<[Vec<CoreEvent>; 2]> {
        let mut events: [Vec<CoreEvent>; 2] = [Vec::new(), Vec::new()];
        for side in 0..2 {
            let Some(s) = streams[side].as_mut() else { continue };
            for t in sess.side_lines(side).iter().skip(from[side]) {
                events[side].push(s.fold_text(t)?);
            }
        }
        Ok(events)
    }

    /// The ROOT of a FORK tree: `sess` is a CORE session (`BridgeSession::new_core` /
    /// `new_construct_turn0_core`, or a search root built from a record) at a boundary; `names` /
    /// `teams` are the two players' (what each side's `Player` knows about itself). Every line
    /// shipped so far is folded TYPED, then the transport is dropped — the version owns the ENGINE
    /// only. `want` names the sides that carry a stream (a search reads one side, and every
    /// version of its tree then folds that side alone).
    pub fn root(sess: BridgeSession, names: [&str; 2], teams: [Option<&str>; 2], want: [bool; 2]) -> R<BattleVersion> {
        Self::root_with(sess, names, teams, want, None)
    }

    /// [`Self::root`] with the per-decision TRACKERS on (`gen3_core_trackers_v1`); every fork of
    /// the tree inherits them.
    pub fn root_with(sess: BridgeSession, names: [&str; 2], teams: [Option<&str>; 2], want: [bool; 2],
                     trackers: Option<ClockConfig>) -> R<BattleVersion> {
        let mut streams = Self::fresh_streams(names, teams, want, trackers)?;
        let events = Self::fold_typed(&sess, &mut streams, [0, 0])?;
        Ok(Self::new(None, Origin::Step, Some(sess.into_engine()), streams, events, [0, 0]))
    }

    /// [`Self::root`] folded from each side's TEXT (the parse path) — the root of a
    /// `core_path=text` search tree, whose engine needs no source recording.
    pub fn root_text(sess: BridgeSession, names: [&str; 2], teams: [Option<&str>; 2], want: [bool; 2]) -> R<BattleVersion> {
        let mut streams = Self::fresh_streams(names, teams, want, None)?;
        let events = Self::fold_text(&sess, &mut streams, [0, 0])?;
        Ok(Self::new(None, Origin::Step, Some(sess.into_engine()), streams, events, [0, 0]))
    }

    /// The transport a fork of this version is driven through: a CLONE of the engine wrapped in a
    /// fresh transport ([`BridgeSession::resume`]) — no chunk history, no script, no seed anchors;
    /// the outstanding requests' issued bytes are shared with the engine, not re-rendered.
    pub fn fork_session(&self) -> R<BridgeSession> {
        let e = self.engine.as_ref().ok_or_else(|| fault("only a step-built version has an engine to fork"))?;
        Ok(BridgeSession::resume(e.clone()))
    }

    /// STEP — a FORK: the child of this version after `cmds` (the parent is untouched and shared).
    pub fn step(self: &Arc<Self>, cmds: &[Cmd], dex: &Dex) -> R<Arc<BattleVersion>> {
        self.step_with(|e| {
            e.feed_cmds(cmds, dex);
            Ok(())
        })
    }

    /// STEP with an arbitrary drive (a search arm's follow-up loop, a forfeit): `drive` advances a
    /// fork of the engine inside a fresh transport; the child folds whatever each side was shipped.
    pub fn step_with(self: &Arc<Self>, drive: impl FnOnce(&mut BridgeSession) -> R<()>)
        -> R<Arc<BattleVersion>> {
        let mut sess = self.fork_session()?;
        drive(&mut sess)?;
        self.child(sess)
    }

    /// The child of this version whose transport is `sess` — a [`Self::fork_session`] the caller
    /// has already driven (a search arm snapshots it at an intermediate decision and again at the
    /// end of the turn; each is a child). Folds EVERY line the transport shipped (it started empty
    /// at this version's boundary), TYPED, and keeps only the engine.
    pub fn child(self: &Arc<Self>, sess: BridgeSession) -> R<Arc<BattleVersion>> {
        if let Some(f) = sess.engine().fatal_error() {
            return Err(f.clone().context("bridge fatal: "));
        }
        let mut streams = [self.streams[0].clone(), self.streams[1].clone()];
        let events = Self::fold_typed(&sess, &mut streams, [0, 0])?;
        Ok(Arc::new(Self::new(Some(Arc::clone(self)), Origin::Step, Some(sess.into_engine()), streams, events, [0, 0])))
    }

    /// [`Self::child`] folded from each side's TEXT instead of typed at the source: the
    /// stream-only path the Rust Core Program's §6c decided every observation comes through. The
    /// search road runs it as `core_path=text`, and its INTEGRITY mode runs both and asserts the
    /// two children equal ([`streams_equal`]). Needs no source recording in the engine.
    pub fn child_text(self: &Arc<Self>, sess: BridgeSession) -> R<Arc<BattleVersion>> {
        if let Some(f) = sess.engine().fatal_error() {
            return Err(f.clone().context("bridge fatal: "));
        }
        let mut streams = [self.streams[0].clone(), self.streams[1].clone()];
        let events = Self::fold_text(&sess, &mut streams, [0, 0])?;
        Ok(Arc::new(Self::new(Some(Arc::clone(self)), Origin::Step, Some(sess.into_engine()), streams, events, [0, 0])))
    }

    /// Keep only `side`'s stream (a search reads one side; the other's fold is dead weight).
    pub fn only(mut self, side: usize) -> BattleVersion {
        self.streams[1 - side] = None;
        self.events[1 - side].clear();
        self
    }

    /// The ROOT of a LINEAR chain OBSERVING `sess` (a core session the caller drives and keeps —
    /// its full transport history is the caller's). Every line shipped so far is folded TYPED.
    pub fn observe_root(sess: &BridgeSession, names: [&str; 2], teams: [Option<&str>; 2], want: [bool; 2])
        -> R<BattleVersion> {
        Self::observe_root_with(sess, names, teams, want, None)
    }

    /// [`Self::observe_root`] with the per-decision TRACKERS on (`gen3_core_trackers_v1`).
    pub fn observe_root_with(sess: &BridgeSession, names: [&str; 2], teams: [Option<&str>; 2], want: [bool; 2],
                             trackers: Option<ClockConfig>) -> R<BattleVersion> {
        let mut streams = Self::fresh_streams(names, teams, want, trackers)?;
        let events = Self::fold_typed(sess, &mut streams, [0, 0])?;
        let cursor = [sess.side_line_count(0), sess.side_line_count(1)];
        Ok(Self::new(None, Origin::Observed, None, streams, events, cursor))
    }

    /// LINEAR step: consume this version and fold what `sess` (the session it observes, which the
    /// caller has driven since) shipped past its cursor. Keeps no parent and copies no engine — a
    /// replay that never revisits a boundary (the parity harness's recorded battles) pays nothing
    /// per decision but the fold.
    pub fn observe(mut self, sess: &BridgeSession) -> R<BattleVersion> {
        if self.origin != Origin::Observed {
            return Err(fault("observe is for a version built by observe_root"));
        }
        let mut streams = [self.streams[0].take(), self.streams[1].take()];
        let events = Self::fold_typed(sess, &mut streams, self.cursor)?;
        let cursor = [sess.side_line_count(0), sess.side_line_count(1)];
        Ok(Self::new(None, Origin::Observed, None, streams, events, cursor))
    }

    /// The ROOT of a parse-built chain: ONE side's stream and nothing else.
    pub fn parse_root(viewer: usize, username: &str, packed_team: Option<&str>) -> R<BattleVersion> {
        Self::parse_root_with(viewer, username, packed_team, None)
    }

    /// [`Self::parse_root`] with the per-decision TRACKERS on.
    pub fn parse_root_with(viewer: usize, username: &str, packed_team: Option<&str>, trackers: Option<ClockConfig>)
        -> R<BattleVersion> {
        let mut streams = [None, None];
        let s = SideStream::new(viewer, username, packed_team)?;
        streams[viewer] = Some(match trackers {
            Some(cfg) => s.with_trackers(cfg),
            None => s,
        });
        Ok(Self::new(None, Origin::Parse, None, streams, [Vec::new(), Vec::new()], [0, 0]))
    }

    fn only_side(&self) -> R<usize> {
        match (self.origin, &self.streams[0], &self.streams[1]) {
            (Origin::Parse, Some(_), None) => Ok(0),
            (Origin::Parse, None, Some(_)) => Ok(1),
            _ => Err(fault("parse_step is for a parse-built (one-side, engine-less) version")),
        }
    }

    /// PARSE — a fork: the child after this side's next protocol `lines` (text).
    pub fn parse_step<S: AsRef<str>>(self: &Arc<Self>, lines: &[S]) -> R<Arc<BattleVersion>> {
        let side = self.only_side()?;
        let mut s = self.streams[side].clone().expect("only_side");
        let events = lines.iter().map(|l| s.fold_text(l.as_ref())).collect::<R<Vec<_>>>()?;
        let mut streams = [None, None];
        streams[side] = Some(s);
        let mut ev = [Vec::new(), Vec::new()];
        ev[side] = events;
        Ok(Arc::new(Self::new(Some(Arc::clone(self)), Origin::Parse, None, streams, ev, [0, 0])))
    }

    /// PARSE — linear (consumes this version, keeps no parent).
    pub fn parse_advance<S: AsRef<str>>(mut self, lines: &[S]) -> R<BattleVersion> {
        let side = self.only_side()?;
        let mut s = self.streams[side].take().expect("only_side");
        let events = lines.iter().map(|l| s.fold_text(l.as_ref())).collect::<R<Vec<_>>>()?;
        let mut streams = [None, None];
        streams[side] = Some(s);
        let mut ev = [Vec::new(), Vec::new()];
        ev[side] = events;
        Ok(Self::new(None, Origin::Parse, None, streams, ev, [0, 0]))
    }

    /// Tell `side`'s stream the choice it sent for the coming action (a denied own action keeps
    /// it — [`crate::trackers::record::Choice`]). A no-op without trackers.
    pub fn note_choice(&mut self, side: usize, token: &str) {
        if let Some(t) = self.streams[side].as_mut().and_then(|s| s.trk.as_mut()) {
            t.choose(token);
        }
    }

    // ---------------------------------------------------------------- reads

    /// The DECISION `side` took at this version's boundary — `Some` iff the transition INTO this
    /// version ended at one of the side's decision requests (its trackers folded there).
    pub fn decision(&self, side: usize) -> Option<&Decision> {
        let s = self.streams[side].as_ref()?;
        let d = s.trk.as_ref()?.last.as_ref()?;
        let first = s.lines - self.events[side].len();
        (d.line >= first).then_some(d)
    }

    /// `side`'s tracker state at its latest decision.
    pub fn trackers(&self, side: usize) -> Option<&crate::trackers::SideTrackers> {
        Some(&self.streams[side].as_ref()?.trk.as_ref()?.trackers)
    }

    pub fn parent(&self) -> Option<&Arc<BattleVersion>> {
        self.parent.as_ref()
    }
    pub fn origin(&self) -> Origin {
        self.origin
    }
    /// The referee (a [`Origin::Step`] version only). A VIEW never reads it.
    pub fn engine(&self) -> Option<&Engine> {
        self.engine.as_ref()
    }
    /// Consume the version, keeping only its engine.
    pub fn into_engine(self) -> Option<Engine> {
        self.engine
    }
    pub fn stream(&self, side: usize) -> Option<&SideStream> {
        self.streams[side].as_ref()
    }
    /// This transition's events on `side`.
    pub fn events(&self, side: usize) -> &[CoreEvent] {
        &self.events[side]
    }
    /// The raw `|request|` payload `side` holds at this boundary.
    pub fn request(&self, side: usize) -> Option<&str> {
        self.streams[side].as_ref()?.board_reading.last_request_text.as_deref()
    }
    /// `side`'s view (`present`), memoized.
    pub fn view(&self, side: usize) -> R<&OneSidedView> {
        let s = self.streams[side].as_ref().ok_or_else(|| fault(format!("no stream for p{}", side + 1)))?;
        self.views[side].get_or_init(|| s.view()).as_ref().map_err(|e| e.clone())
    }
    /// `side`'s legality at this boundary (`LegalActions.from_battle`).
    pub fn legal(&self, side: usize) -> Option<LegalActions> {
        legal_actions(&self.streams[side].as_ref()?.board_reading)
    }
    /// The TRUTH AUDIT of `side`'s view against this version's own engine board.
    pub fn audit(&self, side: usize, dex: &Dex) -> R<Audit> {
        let board = self.engine.as_ref().and_then(|e| e.battle_state()).ok_or_else(|| fault("this version holds no board (audit_on)"))?;
        self.audit_on(side, board, dex)
    }
    /// The TRUTH AUDIT of `side`'s view against `board` — the observed session's, for an
    /// [`Origin::Observed`] version.
    pub fn audit_on(&self, side: usize, board: &BattleState, dex: &Dex) -> R<Audit> {
        if self.origin == Origin::Parse {
            return Err(fault("a parse-built version has no board"));
        }
        let view = self.view(side)?;
        // Whether the side's CURRENT request carries an `active` block — the one that re-syncs the
        // active mon's PP (fork R3); every other own mon's PP is V15's sighting count.
        let synced = self
            .stream(side)
            .and_then(|s| s.board_reading.last_request.as_ref())
            .is_some_and(|r| matches!(r.get("active"), Some(crate::core_events::jsonval::Val::Arr(_))));
        Ok(check_view(view, board, side, dex, synced))
    }
}

/// Whether two versions of the same boundary agree on `side`'s whole stream state (the reading
/// board AND the transition's events, the source index aside) and its view — the search road's
/// INTEGRITY check between the typed shortcut and the text path. `Err` names the first field.
pub fn streams_equal(a: &BattleVersion, b: &BattleVersion, side: usize) -> R<()> {
    let (x, y) = (a.stream(side).ok_or_else(|| fault("no stream"))?, b.stream(side).ok_or_else(|| fault("no stream"))?);
    if x.board_reading != y.board_reading {
        return Err(fault(first_reading_difference(&x.board_reading, &y.board_reading)));
    }
    let (ea, eb) = (a.events(side), b.events(side));
    if ea.len() != eb.len() {
        return Err(fault(format!("events: {} vs {}", ea.len(), eb.len())));
    }
    for (p, q) in ea.iter().zip(eb) {
        if p.line != q.line || p.owner != q.owner || p.readings != q.readings {
            return Err(fault(format!("events: line {} {:?}", p.idx, p.line.render())));
        }
    }
    let (va, vb) = (a.view(side)?.json(), b.view(side)?.json());
    if va != vb {
        let at = va.bytes().zip(vb.bytes()).position(|(p, q)| p != q).unwrap_or(va.len().min(vb.len()));
        return Err(malformed(format!("view JSON differs at byte {at}: …{}… vs …{}…", &va[at.saturating_sub(40)..(at + 40).min(va.len())],
                           &vb[at.saturating_sub(40)..(at + 40).min(vb.len())])));
    }
    Ok(())
}

/// A readable name for the first board-reading field two readings disagree on.
fn first_reading_difference(a: &BoardReading, b: &BoardReading) -> String {
    if a.turn != b.turn {
        return format!("board_reading.turn {} vs {}", a.turn, b.turn);
    }
    for (own, (ta, tb)) in [(true, (&a.team, &b.team)), (false, (&a.opp, &b.opp))] {
        let who = if own { "ours" } else { "opp" };
        if ta.len() != tb.len() {
            return format!("board_reading.{who}: {} mons vs {}", ta.len(), tb.len());
        }
        for ((ka, ma), (kb, mb)) in ta.iter().zip(tb.iter()) {
            if ka != kb || ma != mb {
                return format!("board_reading.{who}[{ka}]: {ma:?} vs {mb:?}");
            }
        }
    }
    "board_reading (a battle-level field)".to_string()
}

/// The parse-reproduces-step gate for ONE boundary: `step` (a step-built version) and `parsed`
/// (the same side's parse-built version fed the same text) must agree on the whole reading board,
/// the transition's events (typed line, outcome owner, every reading — the source index aside) and
/// the view. `Err` names the first difference.
pub fn parse_matches_step(step: &BattleVersion, parsed: &BattleVersion, side: usize) -> R<()> {
    let (a, b) = (
        step.stream(side).ok_or_else(|| fault("step version lacks the side"))?,
        parsed.stream(side).ok_or_else(|| fault("parsed version lacks the side"))?,
    );
    if a.board_reading != b.board_reading {
        return Err(fault(format!("p{}: the parse-built reading board differs from the step-built one", side + 1)));
    }
    let (ea, eb) = (step.events(side), parsed.events(side));
    if ea.len() != eb.len() {
        return Err(fault(format!("p{}: {} step events vs {} parsed", side + 1, ea.len(), eb.len())));
    }
    for (x, y) in ea.iter().zip(eb) {
        if x.line != y.line || x.owner != y.owner || x.readings != y.readings || x.idx != y.idx {
            return Err(fault(format!("p{} line {}: step {:?} vs parse {:?}", side + 1, x.idx, x.line.render(), y.line.render())));
        }
    }
    if step.view(side)? != parsed.view(side)? {
        return Err(fault(format!("p{}: the parse-built view differs from the step-built one", side + 1)));
    }
    Ok(())
}
