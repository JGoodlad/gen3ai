//! [`BattleVersion`] — the Rust core's PERSISTENT battle state (`gen3_core_version_v1`, the Rust
//! Core Program's M2; design `designs/endstate/design_three_tier_environment.md` §3.1–3.2).
//!
//! A battle is a chain of immutable versions, one per decision BOUNDARY. Each holds:
//!
//! * **`parent: Option<Arc<BattleVersion>>`** — history is the parent chain; a fork is an
//!   `Arc::clone` of the handle, so K successors of one decision share their past;
//! * **the per-side STREAM state** ([`SideStream`]): the side's typed lines folded into M1's
//!   reading (the `CoreEvent`s) and into the poke-env reading of the board ([`Tracker`]) — the
//!   ONLY input of the side's view;
//! * **`events`** — the per-side [`CoreEvent`]s of THIS transition;
//! * **the view per side**, computed on demand and memoized ([`BattleVersion::view`]);
//! * **the raw `|request|` per side** (inside the tracker: `last_request_text`), which legality
//!   is derived from ([`BattleVersion::legal`]);
//! * **the engine** — the omniscient session the step path advances. It is the REFEREE: nothing
//!   in the view reads it ([`present`] takes no board); [`BattleVersion::audit`] checks the view
//!   against it.
//!
//! # Two entry points, one state
//!
//! * **step** — the simulator's own transition: clone (or move) the engine, feed the choices,
//!   fold each side's NEW lines, TYPED AT THE SOURCE (the typed shortcut the Rust Core Program's
//!   §6c licenses by `parse(emit(step)) == step`);
//! * **parse** — one side's protocol TEXT, all a real server sends: [`BattleVersion::parse_root`]
//!   and [`BattleVersion::parse_step`] fold `Line::parse` of each line into the SAME
//!   [`SideStream`]. A parse-built version has one side and no engine (its omniscient board is
//!   partial by construction — the design's §6b).
//!
//! The gate is that the two agree version by version: every side's stream state (the whole
//! reading board), its transition events and its view. `core_events --views` runs it on every
//! corpus battle (`designs/rust_sim/present.md`).

use std::sync::{Arc, OnceLock};

use crate::bridge::{BridgeSession, Cmd};
use crate::core_events::parse::OwnerScan;
use crate::core_events::reading::Reader;
use crate::core_events::{is_outcome, CoreEvent, Line, Scope};
use crate::dex::Dex;
use crate::present::{check_view, legal_actions, present, Audit, LegalActions, OneSidedView, Tracker};

type R<T> = Result<T, String>;

/// One side's stream state: its typed lines folded, in order, into the reading of the events
/// (M1's [`Reader`] + the outcome-owner scan) and into the reading of the board ([`Tracker`]).
#[derive(Debug, Clone)]
pub struct SideStream {
    pub tracker: Tracker,
    reader: Reader,
    owners: OwnerScan,
    /// Lines of this side folded so far, whole battle.
    pub lines: usize,
}

impl SideStream {
    /// A fresh stream for `viewer` ("p1" = 0), whose player is `username` and — when known —
    /// fights with `packed_team` (the source of our own spread in gen 3).
    pub fn new(viewer: usize, username: &str, packed_team: Option<&str>) -> R<SideStream> {
        Ok(SideStream {
            tracker: Tracker::new(viewer, username, packed_team)?,
            reader: Reader::new(viewer),
            owners: OwnerScan::default(),
            lines: 0,
        })
    }

    /// Fold ONE typed line. `scope` is the engine's action scope (the step path's owner truth);
    /// the parse path passes `None` and the owner comes from line order.
    pub fn fold(&mut self, line: Line, src: Option<u32>, scope: Option<Scope>) -> R<CoreEvent> {
        let by_order = self.owners.step(&line);
        let readings = self.reader.feed(&line)?;
        self.tracker.feed(&line)?;
        let owner = if is_outcome(line.kw) {
            match scope {
                Some(s) => s.move_side(),
                None => by_order,
            }
        } else {
            None
        };
        let ev = CoreEvent { idx: self.lines as u32, line, src, owner, readings };
        self.lines += 1;
        Ok(ev)
    }

    /// Fold ONE line of protocol TEXT (the parse path).
    pub fn fold_text(&mut self, text: &str) -> R<CoreEvent> {
        let line = Line::parse(text).map_err(|e| format!("line {} {text:?}: {e}", self.lines))?;
        self.fold(line, None, None)
    }

    /// The side's view (`present`).
    pub fn view(&self) -> R<OneSidedView> {
        present(&self.tracker)
    }
}

/// One immutable decision boundary of a battle.
pub struct BattleVersion {
    parent: Option<Arc<BattleVersion>>,
    /// The referee (step-built versions only). Never read by a view.
    engine: Option<BridgeSession>,
    streams: [Option<SideStream>; 2],
    events: [Vec<CoreEvent>; 2],
    views: [OnceLock<R<OneSidedView>>; 2],
    /// Lines of the ENGINE's current chunk list each side has folded. 0 on a COMPACTED version
    /// (a fork's child), whose engine keeps no chunk history at all — so a fork copies the board,
    /// never the battle's text.
    cursor: [usize; 2],
}

impl BattleVersion {
    fn new(parent: Option<Arc<BattleVersion>>, engine: Option<BridgeSession>, streams: [Option<SideStream>; 2],
           events: [Vec<CoreEvent>; 2], cursor: [usize; 2]) -> BattleVersion {
        BattleVersion { parent, engine, streams, events, views: [OnceLock::new(), OnceLock::new()], cursor }
    }

    /// The ROOT of a step-built chain: `engine` is a CORE session (`BridgeSession::new_core` /
    /// `new_construct_turn0_core`, or a search root built from a record) at a boundary; `names` /
    /// `teams` are the two players' (what each side's `Player` knows about itself). Every line
    /// shipped so far is folded. `compact` drops the engine's chunk history once folded (a search
    /// tree's root); a linear replay keeps it. `want` names the sides that carry a stream (a search
    /// reads one side, and every version of its tree then folds that side alone).
    pub fn root(engine: BridgeSession, names: [&str; 2], teams: [Option<&str>; 2], compact: bool,
                want: [bool; 2]) -> R<BattleVersion> {
        let streams = Self::fresh_streams(names, teams, want)?;
        Self::transition(None, engine, streams, [0, 0], compact)
    }

    fn fresh_streams(names: [&str; 2], teams: [Option<&str>; 2], want: [bool; 2]) -> R<[Option<SideStream>; 2]> {
        let mut out = [None, None];
        for side in 0..2 {
            if want[side] {
                out[side] = Some(SideStream::new(side, names[side], teams[side])?);
            }
        }
        Ok(out)
    }

    /// [`Self::root`] folded from each side's TEXT (the parse path), then COMPACTED — the root of
    /// a `core_path=text` search tree, whose engine needs no source recording.
    pub fn root_text(mut engine: BridgeSession, names: [&str; 2], teams: [Option<&str>; 2],
                     want: [bool; 2]) -> R<BattleVersion> {
        let mut streams = Self::fresh_streams(names, teams, want)?;
        let mut events: [Vec<CoreEvent>; 2] = [Vec::new(), Vec::new()];
        for side in 0..2 {
            let Some(s) = streams[side].as_mut() else { continue };
            for t in engine.side_lines(side) {
                events[side].push(s.fold_text(&t)?);
            }
        }
        engine.clear_chunks();
        Ok(Self::new(None, Some(engine), streams, events, [0, 0]))
    }

    /// Fold every line each side was shipped past `from` (the parent's cursor), TYPED at the
    /// source; `compact` then drops the engine's chunk history.
    fn transition(parent: Option<Arc<BattleVersion>>, mut engine: BridgeSession, mut streams: [Option<SideStream>; 2],
                  from: [usize; 2], compact: bool) -> R<BattleVersion> {
        let mut events: [Vec<CoreEvent>; 2] = [Vec::new(), Vec::new()];
        let mut cursor = [0usize; 2];
        for side in 0..2 {
            cursor[side] = engine.side_line_count(side);
            let Some(s) = streams[side].as_mut() else { continue };
            for (line, src, scope) in engine.typed_side_lines(side, from[side])? {
                events[side].push(s.fold(line, src, scope)?);
            }
        }
        if compact {
            engine.clear_chunks();
            cursor = [0, 0];
        }
        Ok(Self::new(parent, Some(engine), streams, events, cursor))
    }

    /// STEP — a FORK: the child of this version after `cmds` (the parent is untouched and shared).
    pub fn step(self: &Arc<Self>, cmds: &[Cmd], dex: &Dex) -> R<Arc<BattleVersion>> {
        self.step_with(|e| {
            e.feed_cmds(cmds, dex);
            Ok(())
        })
    }

    /// STEP with an arbitrary engine drive (a search arm's follow-up loop, a forfeit): `drive`
    /// advances a CLONE of the engine; the child folds whatever each side was shipped.
    pub fn step_with(self: &Arc<Self>, drive: impl FnOnce(&mut BridgeSession) -> R<()>) -> R<Arc<BattleVersion>> {
        let mut engine = self.engine.as_ref().ok_or("a parse-built version has no engine to step")?.snapshot();
        drive(&mut engine)?;
        self.child(engine)
    }

    /// The child of this version whose engine is `engine` — a CLONE of this version's engine that
    /// the caller has already driven (a search arm snapshots the engine at an intermediate decision
    /// and again at the end of the turn; each is a child). Folds the lines each side (that this
    /// version carries a stream for) was shipped since, and COMPACTS the child.
    pub fn child(self: &Arc<Self>, engine: BridgeSession) -> R<Arc<BattleVersion>> {
        if let Some(f) = engine.fatal() {
            return Err(format!("bridge fatal: {f}"));
        }
        let streams = [self.streams[0].clone(), self.streams[1].clone()];
        Ok(Arc::new(Self::transition(Some(Arc::clone(self)), engine, streams, self.cursor, true)?))
    }

    /// [`Self::child`] folded from each side's TEXT instead of typed at the source: the
    /// stream-only path the Rust Core Program's §6c decided every observation comes through. The
    /// search road runs it as `core_path=text`, and its INTEGRITY mode runs both and asserts the
    /// two children equal ([`streams_equal`]). Needs no source recording in the engine.
    pub fn child_text(self: &Arc<Self>, mut engine: BridgeSession) -> R<Arc<BattleVersion>> {
        if let Some(f) = engine.fatal() {
            return Err(format!("bridge fatal: {f}"));
        }
        let mut streams = [self.streams[0].clone(), self.streams[1].clone()];
        let mut events: [Vec<CoreEvent>; 2] = [Vec::new(), Vec::new()];
        for side in 0..2 {
            let Some(s) = streams[side].as_mut() else { continue };
            let text = engine.side_lines(side);
            for t in text.iter().skip(self.cursor[side]) {
                events[side].push(s.fold_text(t)?);
            }
        }
        engine.clear_chunks();
        Ok(Arc::new(Self::new(Some(Arc::clone(self)), Some(engine), streams, events, [0, 0])))
    }

    /// Keep only `side`'s stream (a search reads one side; the other's fold is dead weight).
    pub fn only(mut self, side: usize) -> BattleVersion {
        self.streams[1 - side] = None;
        self.events[1 - side].clear();
        self
    }

    /// STEP — LINEAR: consume this version (its engine is moved, not cloned) and keep no parent
    /// and the engine's full chunk history. A replay that never revisits a boundary (the parity
    /// harness's recorded battles) pays no engine copy per decision.
    pub fn advance_with(mut self, drive: impl FnOnce(&mut BridgeSession) -> R<()>) -> R<BattleVersion> {
        let mut engine = self.engine.take().ok_or("a parse-built version has no engine to step")?;
        drive(&mut engine)?;
        let streams = [self.streams[0].take(), self.streams[1].take()];
        Self::transition(None, engine, streams, self.cursor, false)
    }

    /// The ROOT of a parse-built chain: ONE side's stream and nothing else.
    pub fn parse_root(viewer: usize, username: &str, packed_team: Option<&str>) -> R<BattleVersion> {
        let mut streams = [None, None];
        streams[viewer] = Some(SideStream::new(viewer, username, packed_team)?);
        Ok(Self::new(None, None, streams, [Vec::new(), Vec::new()], [0, 0]))
    }

    fn only_side(&self) -> R<usize> {
        match (&self.streams[0], &self.streams[1], &self.engine) {
            (Some(_), None, None) => Ok(0),
            (None, Some(_), None) => Ok(1),
            _ => Err("parse_step is for a parse-built (one-side, engine-less) version".into()),
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
        Ok(Arc::new(Self::new(Some(Arc::clone(self)), None, streams, ev, [0, 0])))
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
        Ok(Self::new(None, None, streams, ev, [0, 0]))
    }

    // ---------------------------------------------------------------- reads

    pub fn parent(&self) -> Option<&Arc<BattleVersion>> {
        self.parent.as_ref()
    }
    /// The referee (step-built versions only). A VIEW never reads it.
    pub fn engine(&self) -> Option<&BridgeSession> {
        self.engine.as_ref()
    }
    /// Consume the version, keeping only its engine (a replay's final session).
    pub fn into_engine(self) -> Option<BridgeSession> {
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
        self.streams[side].as_ref()?.tracker.last_request_text.as_deref()
    }
    /// `side`'s view (`present`), memoized.
    pub fn view(&self, side: usize) -> R<&OneSidedView> {
        let s = self.streams[side].as_ref().ok_or_else(|| format!("no stream for p{}", side + 1))?;
        self.views[side].get_or_init(|| s.view()).as_ref().map_err(|e| e.clone())
    }
    /// `side`'s legality at this boundary (`LegalActions.from_battle`).
    pub fn legal(&self, side: usize) -> Option<LegalActions> {
        legal_actions(&self.streams[side].as_ref()?.tracker)
    }
    /// The TRUTH AUDIT of `side`'s view against the engine board.
    pub fn audit(&self, side: usize, dex: &Dex) -> R<Audit> {
        let board = self.engine.as_ref().and_then(|e| e.battle_state()).ok_or("a parse-built version has no board")?;
        let view = self.view(side)?;
        // Whether the side's CURRENT request carries an `active` block — the one that re-syncs the
        // active mon's PP (fork R3); every other own mon's PP is V15's sighting count.
        let synced = self
            .stream(side)
            .and_then(|s| s.tracker.last_request.as_ref())
            .is_some_and(|r| matches!(r.get("active"), Some(crate::core_events::jsonval::Val::Arr(_))));
        Ok(check_view(view, board, side, dex, synced))
    }
}

/// Whether two versions of the same boundary agree on `side`'s whole stream state (the reading
/// board AND the transition's events, the source index aside) and its view — the search road's
/// INTEGRITY check between the typed shortcut and the text path. `Err` names the first field.
pub fn streams_equal(a: &BattleVersion, b: &BattleVersion, side: usize) -> R<()> {
    let (x, y) = (a.stream(side).ok_or("no stream")?, b.stream(side).ok_or("no stream")?);
    if x.tracker != y.tracker {
        return Err(first_tracker_difference(&x.tracker, &y.tracker));
    }
    let (ea, eb) = (a.events(side), b.events(side));
    if ea.len() != eb.len() {
        return Err(format!("events: {} vs {}", ea.len(), eb.len()));
    }
    for (p, q) in ea.iter().zip(eb) {
        if p.line != q.line || p.owner != q.owner || p.readings != q.readings {
            return Err(format!("events: line {} {:?}", p.idx, p.line.render()));
        }
    }
    let (va, vb) = (a.view(side)?.json(), b.view(side)?.json());
    if va != vb {
        let at = va.bytes().zip(vb.bytes()).position(|(p, q)| p != q).unwrap_or(va.len().min(vb.len()));
        return Err(format!("view JSON differs at byte {at}: …{}… vs …{}…", &va[at.saturating_sub(40)..(at + 40).min(va.len())],
                           &vb[at.saturating_sub(40)..(at + 40).min(vb.len())]));
    }
    Ok(())
}

/// A readable name for the first tracker field two readings disagree on.
fn first_tracker_difference(a: &Tracker, b: &Tracker) -> String {
    if a.turn != b.turn {
        return format!("tracker.turn {} vs {}", a.turn, b.turn);
    }
    for (own, (ta, tb)) in [(true, (&a.team, &b.team)), (false, (&a.opp, &b.opp))] {
        let who = if own { "ours" } else { "opp" };
        if ta.len() != tb.len() {
            return format!("tracker.{who}: {} mons vs {}", ta.len(), tb.len());
        }
        for ((ka, ma), (kb, mb)) in ta.iter().zip(tb.iter()) {
            if ka != kb || ma != mb {
                return format!("tracker.{who}[{ka}]: {ma:?} vs {mb:?}");
            }
        }
    }
    "tracker (a battle-level field)".to_string()
}

/// The parse-reproduces-step gate for ONE boundary: `step` (a step-built version) and `parsed`
/// (the same side's parse-built version fed the same text) must agree on the whole reading board,
/// the transition's events (typed line, outcome owner, every reading — the source index aside) and
/// the view. `Err` names the first difference.
pub fn parse_matches_step(step: &BattleVersion, parsed: &BattleVersion, side: usize) -> R<()> {
    let (a, b) = (
        step.stream(side).ok_or("step version lacks the side")?,
        parsed.stream(side).ok_or("parsed version lacks the side")?,
    );
    if a.tracker != b.tracker {
        return Err(format!("p{}: the parse-built reading board differs from the step-built one", side + 1));
    }
    let (ea, eb) = (step.events(side), parsed.events(side));
    if ea.len() != eb.len() {
        return Err(format!("p{}: {} step events vs {} parsed", side + 1, ea.len(), eb.len()));
    }
    for (x, y) in ea.iter().zip(eb) {
        if x.line != y.line || x.owner != y.owner || x.readings != y.readings || x.idx != y.idx {
            return Err(format!("p{} line {}: step {:?} vs parse {:?}", side + 1, x.idx, x.line.render(), y.line.render()));
        }
    }
    if step.view(side)? != parsed.view(side)? {
        return Err(format!("p{}: the parse-built view differs from the step-built one", side + 1));
    }
    Ok(())
}
