//! The ENGINE half of a bridge session (`gen3_core_engine_split_v1`, the Rust Core Program's
//! pre-M3 hand-off; `designs/endstate/program_rust_core.md` §2 "Before M3 starts").
//!
//! A [`crate::bridge::BridgeSession`] used to be ONE struct holding two different things: the
//! battle (the live `Battle`, the [`FullBattleDriver`] turn loop, the open request boundary) and
//! the WIRE (the per-side chunk stream with its HP-privacy fold, the `|request|` JSON strings, the
//! command queue, the committed `script` and the seed anchors). A fork cloned both, so a search
//! successor paid for the wire's history, which grows with battle length.
//!
//! [`Engine`] is the battle alone. It advances from a queue of wire choices the CALLER owns and
//! reports everything it emits, in order, to an [`EngineSink`]: an omniscient log range to flush,
//! a request (TYPED — [`Request`], rendered to its JSON only by a sink that ships it), an
//! `|error|` frame, the forced-Struggle announce, a committed decision. The TRANSPORT
//! (`BridgeSession`) is one sink: it folds those into the per-side chunks `sim_bridge` writes,
//! byte-identically to the pre-split session (the 41-battle transcript gate, `bridge_test`'s
//! incremental-vs-genesis parity). A [`crate::version::BattleVersion`] owns an `Engine` and no
//! transport; a fork wraps a clone in a fresh transport only while it is being driven.
//!
//! **The requests are values.** [`Engine::request`] is the typed `|request|` outstanding to a side
//! (its kind and the four flags that shape its trailing keys), and [`Request::json`] renders it
//! against the paused board. The board does not move while a boundary is open (a reject mutates
//! only the request), so the rendering equals the bytes that went on the wire — pinned by
//! `tests/engine_split_test.rs` at every boundary of a corpus battle.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::battle::{Battle, BattleOptions};
use crate::bridge::{
    boundary_kinds, build_request_with_disabled_source, display_name, has_live_bench, move_disabled, pending_force,
    reject_move_name, resolve_choice, resolve_wire, Cmd, RequestState, SideRequest, WireChoice,
};
use crate::core_error::{fault, malformed, CoreError};
use crate::dex::Dex;
use crate::state::BattleState;
use crate::turn::{Choice, FullBattleDriver, ScriptDecision};

/// A `|request|` the engine issued to one side, as a VALUE: the kind and the flags that decide the
/// JSON's shape. [`Request::json`] renders the exact wire bytes against the paused board.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Request {
    pub kind: RequestState,
    /// A `move` request re-issued after a hidden-trap reject: the active block carries
    /// `"trapped":true`.
    pub trapped: bool,
    /// `,"update":true` — a re-issued request (`emitRequest(_, true)`).
    pub update: bool,
    /// `,"noCancel":true` — a forced replacement only one side is asked for.
    pub no_cancel: bool,
    /// The slot a disabled-move reject marked with `"disabledSource":""`.
    pub disabled_source: Option<u8>,
}

impl Request {
    fn plain(kind: SideRequest, no_cancel: bool) -> Request {
        Request { kind: kind.into(), trapped: false, update: false, no_cancel, disabled_source: None }
    }

    fn side_kind(&self) -> SideRequest {
        match self.kind {
            RequestState::Move => SideRequest::Move,
            RequestState::Switch => SideRequest::ForceSwitch,
            RequestState::Wait => SideRequest::Wait,
        }
    }

    /// The `|request|{…}` line for `side` — byte-identical to what the transport shipped, because
    /// the board a boundary is open on does not change until the boundary closes.
    pub fn json(&self, state: &BattleState, side: usize, dex: &Dex) -> String {
        build_request_with_disabled_source(
            state,
            side,
            self.side_kind(),
            self.trapped,
            self.update,
            self.no_cancel,
            self.disabled_source.map(usize::from),
            dex,
        )
    }
}

/// Where an [`Engine`] reports what it emits, in emission order. The transport
/// ([`crate::bridge::BridgeSession`]) folds each into per-side chunks; a reader that needs none of
/// it passes [`NullSink`].
pub trait EngineSink {
    /// A request boundary opens (before its log flush). The transport records the seed anchor.
    fn boundary_open(&mut self, _state: &BattleState) {}
    /// The omniscient log lines `[from, to)` are flushed (one batch per side).
    fn log(&mut self, state: &BattleState, from: usize, to: usize);
    /// `req` is issued to `side` (a boundary's request, or a re-request after a reject); `json` is
    /// its `|request|` line, rendered ONCE by the engine at issue.
    fn request(&mut self, side: usize, req: &Request, json: &Arc<str>);
    /// An `|error|` frame to `side` (a refused choice).
    fn error(&mut self, side: usize, line: String);
    /// The forced-Struggle `|-activate|` announce to `side` at the commit.
    fn struggle(&mut self, side: usize, line: String);
    /// Nothing is outstanding any more (the boundary closed, or the battle ended).
    fn requests_void(&mut self) {}
    /// A decision committed (both sides' accepted choices) and is about to be fed to the driver.
    fn commit(&mut self, _dec: ScriptDecision) {}
}

/// A sink that keeps nothing.
pub struct NullSink;

impl EngineSink for NullSink {
    fn log(&mut self, _state: &BattleState, _from: usize, _to: usize) {}
    fn request(&mut self, _side: usize, _req: &Request, _json: &Arc<str>) {}
    fn error(&mut self, _side: usize, _line: String) {}
    fn struggle(&mut self, _side: usize, _line: String) {}
}

/// The mid-boundary progress an [`Engine`] persists across advances (a `move` request needs BOTH
/// sides' choices, possibly arriving separately; a trapped reject holds the boundary open).
#[derive(Clone)]
struct BoundaryProgress {
    kinds: [SideRequest; 2],
    got: [Option<Choice>; 2],
    need: [bool; 2],
}

/// A refused choice, in the two shapes `Side.emitChoiceError` produces
/// (`gen3_choice_reject_framing_v1`).
///
/// The distinction is NOT the message and NOT "how many times it was tried" — it is whether the
/// refusal MUTATED the request. A mutation makes the sim re-issue the (now-different) request and
/// tag the error `[Unavailable choice]`; with nothing to re-issue it is `[Invalid choice]` and the
/// client must re-pick from the request it already holds. Model the condition, not a per-message
/// verdict, or the next disabler added here will be classified by hand and get it wrong.
#[derive(Debug, Clone)]
enum RejectClass {
    /// No request change ⇒ `[Invalid choice]`, and NOTHING follows.
    Invalid { message: String },
    /// The request changed (a slot gained `disabledSource`) ⇒ `[Unavailable choice]` plus a
    /// re-issued request carrying `"update":true`.
    Unavailable { message: String, ds_slot: usize },
}

impl RejectClass {
    fn message(&self) -> &str {
        match self {
            RejectClass::Invalid { message } | RejectClass::Unavailable { message, .. } => message,
        }
    }

    /// The exact wire line, byte-for-byte with the sim (probe-measured).
    fn error_line(&self) -> String {
        let tag = match self {
            RejectClass::Invalid { .. } => "[Invalid choice]",
            RejectClass::Unavailable { .. } => "[Unavailable choice]",
        };
        format!("|error|{} {}", tag, self.message())
    }
}

/// Classify a choice this boundary must REFUSE, or `None` when it is legal.
///
/// Mirrors `Side.chooseMove` / `Side.chooseSwitch`'s refusal ladder for the cases the bridge can
/// actually reach. The trapped SWITCH is handled by its own older block in `advance` (it has a
/// two-phase maybeTrapped→trapped machine this classifier deliberately does not duplicate), so it
/// is excluded here by the caller.
///
/// NOTE the message strings carry a non-ASCII `é` in "Pokémon" — that is the sim's own byte
/// sequence and must not be "normalized".
fn classify_reject(
    state: &BattleState,
    side: usize,
    kind: SideRequest,
    wire: &WireChoice,
    resolved: &Choice,
    dex: &Dex,
) -> Option<RejectClass> {
    let s = &state.sides[side];
    let mon = &s.pokemon[s.active];
    // `pass` — Showdown's `Side.choosePass` (`sim/side.ts:1291`). In gen3 SINGLES it is never
    // legal: at a move request the active mon is alive and there is no `commanding` volatile, and
    // at a forced switch `forcedPassesLeft > 0` needs an EMPTY live bench, by which point the
    // side has already lost and the battle is over (`side.ts:530-536`). Both refusals call
    // `emitChoiceError` with no update callback, so they render `[Invalid choice]` with NOTHING
    // following. Checked before the `resolved` match because `Pass` deliberately carries no
    // resolution — it rides the out-of-range fallback, which would otherwise be reported as a
    // bogus "doesn't have a move N".
    if matches!(wire, WireChoice::Pass) {
        return Some(RejectClass::Invalid {
            message: match kind {
                SideRequest::ForceSwitch => format!(
                    "Can't pass: You need to switch in a Pokémon to replace {}",
                    display_name(mon, dex)
                ),
                _ => format!("Can't pass: Your {} must make a move (or switch)", display_name(mon, dex)),
            },
        });
    }
    match resolved {
        Choice::Move(k) => {
            // A NUMERIC slot is bounded by what the REQUEST OFFERED, not by the moveset
            // (`gen3_single_entry_request_slot_reject_v1`). `Side.chooseMove`'s index check runs
            // FIRST — ahead of both substitution branches below — against
            // `getMoveRequestData().moves`, and the two shapes that collapse that array to ONE
            // entry therefore reject `move 2`..`move 4` like any other out-of-range slot.
            //
            // Both halves are node-MEASURED, by two different oracles:
            //   * STRUGGLE — `replay_impl_parity` on a fresh golden, which is a real
            //     `node replay_driver.js` on a real board: an arm feeding `move 2` to a 0-PP
            //     Blissey got a SILENT Struggle substitution from the port, and from node
            //     `|error|[Invalid choice] Can't move: Your Blissey doesn't have a move 2`
            //     followed by a correction (1 vs 2 `choices_used`, a wholly different arm).
            //   * LOCK — `harness/probe_single_entry_request_slot.js`, three arms at a
            //     Solar-Beam CHARGING boundary: `move 1` accepted with NO error, `move 2` and
            //     `move 4` each `|error|[Invalid choice] Can't move: Your Venusaur doesn't
            //     have a move N`.
            //
            // ⚠️ This must never refuse the ONE action the request DID offer — that shape
            // (`gen3_locked_choice_never_rejected_v1`) killed two production launches. It
            // structurally cannot: `offered` is 1 on both branches, so `Move(0)` always passes,
            // and it is the wire's own numeric token that is bounded, never a resolved name.
            if let WireChoice::Move(n) = wire {
                let offered = if mon.move_locked() || mon.must_struggle(dex) { 1 } else { mon.set.moves.len() };
                if *n >= offered {
                    return Some(RejectClass::Invalid {
                        message: format!("Can't move: Your {} doesn't have a move {}", display_name(mon, dex), n + 1),
                    });
                }
            }
            // FORCED STRUGGLE is a SUBSTITUTION, not a refusal. When every usable slot is gone
            // (Taunt / Disable / the Choice lock / 0 PP) the sim's request offers only Struggle
            // and `side.choose` swaps the pick for it — no `|error|`, no re-request. Classifying
            // it as a disabled-move reject made the incremental path emit an error the genesis
            // reference (and node) never send: caught by
            // `bridge_test::bridge_incremental_matches_genesis_replay` on the `taunt_struggle`
            // scenario, which exists precisely because that wedge shipped once before.
            if mon.must_struggle(dex) {
                return None;
            }
            // LOCK-IN is the same shape of substitution, and omitting it cost two production
            // launches. When `move_locked()` (a two-turn move CHARGING, or `must_recharge`) the
            // request builder emits a SINGLE entry with `trapped:true` and no `pp`/`disabled` key
            // — the sim's hardLocked shape. So the request offers exactly one action. Falling
            // through to `move_disabled` below lets the classifier REFUSE that one action, because
            // `move_usable` models the Choice lock, Disable, Encore, Taunt and PP and knows
            // nothing about `two_turn`/`must_recharge` (`state.rs`). Disable landing on the
            // charging slot is enough to trip it.
            //
            // The failure is not a stricter parser, it is rust contradicting ITSELF: it offers X
            // and then rejects X. poke-env re-picks from the same single-entry request, sends the
            // same token, and `REJECT_STREAK_CAP` fires `__ERR__` — which is not in-band, so it
            // retires the reader and crashes the whole run. Observed as "9 consecutive rejects of
            // MoveName(\"solarbeam\")" at ~8 minutes, twice, at load 31 and at load 5 alike.
            //
            // A forced choice is not a refusable one. `resolve_choice` already maps the wire name
            // to `Move(0)` for a locked mon, so the accepted path was always there; only the
            // classifier was cutting it off. One predicate covers the charge family AND the
            // recharge mirror, so this also closes fly/dig/bounce and hyperbeam.
            if mon.move_locked() {
                return None;
            }
            // An out-of-range slot: report the number the CLIENT sent, not the internal
            // out-of-range fallback `resolve_choice` substitutes.
            if *k >= mon.set.moves.len() {
                let shown = match wire {
                    WireChoice::Move(n) => n + 1,
                    _ => k + 1,
                };
                return Some(RejectClass::Invalid {
                    message: format!("Can't move: Your {} doesn't have a move {}", display_name(mon, dex), shown),
                });
            }
            // IMPRISON (`gen3_imprison_choice_reject_v1`): an imprisoned slot is a HIDDEN disable
            // (`disableMove(id, true)` → `'hidden'`). The REQUEST masks it (`getMoves(_, true)`
            // renders `disabled:false`), but `Side.chooseMove` reads `getMoves()` UNRESTRICTED, so
            // the pick is REFUSED exactly like a visible disable: `[Unavailable choice] Can't move:
            // X's Y is disabled` + a re-request with that slot `disabled:true,"disabledSource":""`
            // (and, via `updateDisabledRequest`, no `maybeLocked`) — `harness/probe_imprison.js`
            // Q5, `harness/probe_maybe_flags.js` R1. WRONG (pre-fix): this classifier did not see
            // the FOE-held restriction, so the engine ACCEPTED the pick and the flat driver then
            // dropped the decision (`choice_is_legal` already refuses it), re-opening the
            // boundary with a silent, un-`update`d copy of the same request — no `|error|` at all.
            // When NO slot is usable once the imprisoned ones are counted, the sim instead
            // SUBSTITUTES Struggle (`getMoves()` returns `[]`); that case is left to the existing
            // path (see the FINDING in `designs/rust_sim/ab_fuzzer_findings.md`).
            let imprisoned = |j: usize| {
                state
                    .move_at(side, s.active, j, dex)
                    .is_some_and(|m| state.imprisoned_for(side, &crate::dex::to_id(&m.id), dex))
            };
            let imprisoned_pick = imprisoned(*k)
                && (0..mon.set.moves.len()).any(|j| !move_disabled(mon, j, dex) && !imprisoned(j));
            if move_disabled(mon, *k, dex) || imprisoned_pick {
                let name = reject_move_name(mon, *k, dex);
                return Some(RejectClass::Unavailable {
                    message: format!("Can't move: {}'s {} is disabled", display_name(mon, dex), name),
                    ds_slot: *k,
                });
            }
            None
        }
        Choice::Switch(n) => {
            if *n >= s.pokemon.len() {
                return Some(RejectClass::Invalid {
                    message: format!("Can't switch: You do not have a Pokémon in slot {} to switch to", n + 1),
                });
            }
            if *n == s.active {
                return Some(RejectClass::Invalid {
                    message: "Can't switch: You can't switch to an active Pokémon".to_string(),
                });
            }
            if s.pokemon[*n].fainted || s.pokemon[*n].hp == 0 {
                return Some(RejectClass::Invalid {
                    message: "Can't switch: You can't switch to a fainted Pokémon".to_string(),
                });
            }
            None
        }
    }
}

/// How many consecutive rejects at ONE boundary before the bridge calls it a no-progress LOOP.
///
/// A reject leaves the boundary OPEN and re-issues the request — correct when the client then
/// picks something else. But an RL policy is DETERMINISTIC given the same request, so if its
/// action mask disagrees with the port about legality (e.g. the port thinks the active mon is
/// TRAPPED and the mask does not) it re-sends the SAME switch forever. Measured in a live run:
/// one child spinning at 46 MB/s of re-requests while its env's `step()` never returned, wedging
/// the whole vec-env. Real Showdown has the same reject protocol; only the deterministic client
/// makes it non-terminating, so the BRIDGE must bound it. A handful of rejects is legitimate
/// (trapped-then-move is a normal two-exchange round), so the cap is generous — it exists to turn
/// an unbounded spin into a diagnosable error, not to police ordinary rejects.
const REJECT_STREAK_CAP: u32 = 8;

/// The battle half of a bridge session: the live `Battle`, the [`FullBattleDriver`] turn loop, the
/// open request boundary and the typed requests outstanding. Plain data all the way down, so a
/// derived `Clone` is a deep, independent paused battle — a fork.
#[derive(Clone)]
pub struct Engine {
    battle: Battle,
    driver: FullBattleDriver,
    /// Whether this FORMAT hides exact HP from the non-owner (gen3ou's "HP Percentage Mod").
    report_percent: bool,
    /// Cursor into the LIVE battle log: the lines already flushed to a sink.
    prev_log_len: usize,
    /// The open boundary's per-side progress, or `None` between boundaries.
    boundary: Option<BoundaryProgress>,
    /// The request OUTSTANDING to each side — the port of Showdown's `side.activeRequest`, as a
    /// value (`gen3_bridge_clone_branch_v1`). Set wherever a request is issued (the boundary open,
    /// AND the re-request a reject re-issues), cleared when the boundary closes / the battle ends.
    requests: [Option<Request>; 2],
    /// Each outstanding request's `|request|` line, rendered ONCE when it was issued and SHARED by
    /// every fork (an `Arc`). Not the wire's history — one line per side, replaced at every issue:
    /// a fork's fresh transport needs the bytes its parent's wire shipped, and re-rendering them
    /// per fork measured ≈ 27 µs, more than the whole engine clone saves
    /// (`designs/research_state/measurements/rust_core_m3_2026-09-24/`).
    issued: [Option<Arc<str>>; 2],
    /// The battle reached its natural WIN/LOSS/TIE end.
    ended: bool,
    /// The winner of a [`Engine::forfeit`], if one happened. A forfeit ends the battle through the
    /// PROTOCOL (it writes the `|win|` pair directly) rather than the driver's own win check, so
    /// the driver's phase never becomes `Ended` and [`Engine::winner`] would otherwise report
    /// `None` for a battle that plainly has a winner.
    forfeit_winner: Option<usize>,
    /// An upstream-desync graceful stop (the R20 `!need[s]` case) — no further advance.
    stopped: bool,
    /// A FATAL, non-recoverable condition the live bridge must report as `__ERR__` rather than
    /// silently spin on (`gen3_bridge_unresolvable_choice_failloud_v1`): a NAME-form wire choice
    /// that resolves against nothing, or a boundary that keeps rejecting.
    fatal: Option<CoreError>,
    /// Consecutive REJECTS at the CURRENT boundary (reset whenever a decision commits).
    reject_streak: u32,
}

impl Engine {
    /// A fresh engine over a constructed `battle`: the framing is emitted INTO the live log (kept,
    /// not drained), and nothing is flushed yet — the caller flushes `[0, framing_len())` (the
    /// transport reframes it) and then [`Engine::advance`]s to the first boundary. `core` turns on
    /// the log's typed source recording (`gen3_core_events_v1`).
    pub fn new(mut battle: Battle, opts: &BattleOptions, dex: &Dex, core: bool) -> Result<Engine, String> {
        // Percent HP fold applies only in non-debug formats (gen3ou); a `debug:true` format
        // (gen3customgame) sets `reportExactHP` → both sides see exact HP.
        let report_percent = !crate::bridge::format_is_debug(&opts.format_id);
        let framing_len = {
            let bs = battle.state_mut().ok_or("no state")?;
            bs.log.enable();
            if core {
                bs.log.record_core();
            }
            bs.emit_framing(dex);
            bs.log.lines().len()
        };
        Ok(Engine {
            battle,
            driver: FullBattleDriver::new(),
            report_percent,
            prev_log_len: framing_len,
            boundary: None,
            requests: [None, None],
            issued: [None, None],
            ended: false,
            forfeit_winner: None,
            stopped: false,
            fatal: None,
            reject_streak: 0,
        })
    }

    /// The live battle state (the omniscient referee readout). Always `Some` for an engine built
    /// by [`Engine::new`].
    pub fn battle_state(&self) -> Option<&BattleState> {
        self.battle.state()
    }

    /// Mutable battle state — the counterfactual reseed hook only.
    pub(crate) fn battle_state_mut(&mut self) -> Option<&mut BattleState> {
        self.battle.state_mut()
    }

    /// Whether the log records the typed source of every line (a CORE engine).
    pub fn is_core(&self) -> bool {
        self.battle.state().and_then(|s| s.log.source_recs()).is_some()
    }

    /// Whether this session's FORMAT hides exact HP from the non-owner.
    pub fn report_percent(&self) -> bool {
        self.report_percent
    }

    /// The battle reached game-end.
    pub fn is_ended(&self) -> bool {
        self.ended
    }

    /// A FATAL condition the caller must surface (never spin on) — its message.
    pub fn fatal(&self) -> Option<&str> {
        self.fatal.as_ref().map(CoreError::message)
    }

    /// The same, typed (`gen3_core_error_v1`): an unresolvable choice or a no-progress reject loop
    /// is MALFORMED client input; an upstream desync is an engine FAULT.
    pub fn fatal_error(&self) -> Option<&CoreError> {
        self.fatal.as_ref()
    }

    /// The battle's CURRENT turn. 0 before the battle is built.
    pub fn turn(&self) -> u32 {
        self.battle.state().map(|st| st.turn).unwrap_or(0)
    }

    /// The typed request OUTSTANDING to `side`, or `None` between boundaries / after game-end.
    pub fn request(&self, side: usize) -> Option<&Request> {
        self.requests[side].as_ref()
    }

    /// [`Engine::request`] rendered to its wire bytes against the paused board (a fresh render —
    /// equal to [`Engine::issued_json`], which `tests/engine_split_test.rs` pins).
    pub fn request_json(&self, side: usize, dex: &Dex) -> Option<String> {
        let st = self.battle.state()?;
        self.requests[side].as_ref().map(|r| r.json(st, side, dex))
    }

    /// The `|request|` line [`Engine::request`] was ISSUED as — the exact bytes, shared, never
    /// re-rendered.
    pub fn issued_json(&self, side: usize) -> Option<&Arc<str>> {
        self.issued[side].as_ref()
    }

    /// Issue `req` to `side`: render it once, record both halves, tell the sink.
    fn issue(&mut self, side: usize, req: Request, sink: &mut impl EngineSink, dex: &Dex) {
        let json: Arc<str> = req.json(self.battle.state().expect("state"), side, dex).into();
        sink.request(side, &req, &json);
        self.requests[side] = Some(req);
        self.issued[side] = Some(json);
    }

    /// Nothing is outstanding.
    fn void(&mut self, sink: &mut impl EngineSink) {
        self.requests = [None, None];
        self.issued = [None, None];
        sink.requests_void();
    }

    /// The OPEN request kind for `side` (Showdown's `side.requestState`), `None` when no boundary
    /// is open.
    pub fn request_kind(&self, side: usize) -> Option<RequestState> {
        self.boundary.as_ref().map(|bp| RequestState::from(bp.kinds[side]))
    }

    /// Whether `side` has already supplied an ACCEPTED choice for the open boundary (Showdown's
    /// `side.isChoiceDone()`); `true` also for a `Wait` side and when no boundary is open.
    pub fn is_choice_done(&self, side: usize) -> bool {
        self.boundary.as_ref().map(|bp| !bp.need[side]).unwrap_or(true)
    }

    /// The WINNER's side index once the battle ended (both end paths), else `None` — which is also
    /// a gen-3 TIE; pair with [`Engine::is_ended`].
    pub fn winner(&self) -> Option<usize> {
        self.driver.winner().or(self.forfeit_winner)
    }

    /// Forfeit `side` — the poke-env `/forfeit` → `FORCELOSE <side>` path. Writes the deciding
    /// `|` + `|win|<name>` pair into the log and flushes every line past the cursor as ONE batch
    /// (the natural-end emission: the Node bridge writes `>forcelose` INTO the sim, so Showdown
    /// runs a real `win(otherSide)`).
    pub fn forfeit(&mut self, side: usize, sink: &mut impl EngineSink) {
        if self.ended {
            return;
        }
        let winner = 1 - side;
        {
            let bs = self.battle.state_mut().expect("state");
            if bs.logging() {
                bs.log.separator();
                let name = bs.sides[winner].name.clone();
                bs.log.win(&name);
            }
        }
        let bs = self.battle.state().expect("state");
        let new_len = bs.log.lines().len();
        sink.log(bs, self.prev_log_len, new_len);
        self.prev_log_len = new_len;
        self.ended = true;
        self.forfeit_winner = Some(winner);
        // The battle is over — whatever request was open is void.
        self.boundary = None;
        self.void(sink);
    }

    /// Advance from the current paused state as far as `cmds` allow: at each request boundary
    /// flush the log delta and issue the requests, consume choice(s), announce a forced Struggle,
    /// feed ONE decision to the driver, repeat — pausing when `cmds` runs out mid-boundary.
    /// Byte-identical (via the transport sink: chunks + seeds + script) to the genesis-replay core
    /// fed the same command stream.
    pub fn advance(&mut self, cmds: &mut VecDeque<Cmd>, sink: &mut impl EngineSink, dex: &Dex) {
        // Defensive spin guard (the turn-limit TIE + the driver's `TURN_LIMIT`/`turn_loop` watchdogs
        // are the real runaway protection; this only catches a logic bug in THIS loop).
        let mut guard: u64 = 0;
        loop {
            guard += 1;
            if guard > 100_000_000 {
                panic!("Engine::advance spin guard exceeded (a boundary never resolved)");
            }
            if self.stopped || self.ended {
                return;
            }
            // ── Start a new boundary if not mid-boundary. ──
            if self.boundary.is_none() {
                sink.boundary_open(self.battle.state().expect("state"));
                // The log delta flushed since the previous boundary → ONE chunk per side.
                {
                    let bs = self.battle.state().expect("state");
                    let new_len = bs.log.lines().len();
                    sink.log(bs, self.prev_log_len, new_len);
                    self.prev_log_len = new_len;
                }
                if self.driver.is_ended() {
                    self.ended = true;
                    // Nothing is outstanding once the battle is over (`side.activeRequest` is
                    // cleared at the sim's `win`).
                    self.requests = [None, None];
                    sink.requests_void();
                    return;
                }
                // Determine the pending request kind per side from the paused state.
                let (kinds, no_cancel_forced) = {
                    let bs = self.battle.state().expect("state");
                    let force = pending_force(bs);
                    let is_switch = force[0] || force[1];
                    let kinds = boundary_kinds(bs, &force, is_switch);
                    let non_wait = kinds.iter().filter(|k| **k != SideRequest::Wait).count();
                    (kinds, non_wait < 2)
                };
                // Issue BOTH sides' requests (incl. the `{"wait":true}` frame — the sim issues
                // that too), p1 then p2, one chunk each.
                for side in 0..2 {
                    let no_cancel = kinds[side] == SideRequest::ForceSwitch && no_cancel_forced;
                    self.issue(side, Request::plain(kinds[side], no_cancel), sink, dex);
                }
                self.boundary = Some(BoundaryProgress {
                    kinds,
                    got: [None, None],
                    need: [kinds[0] != SideRequest::Wait, kinds[1] != SideRequest::Wait],
                });
            }

            // ── Consume choice(s) answering this boundary. ──
            loop {
                let (need0, need1) = {
                    let bp = self.boundary.as_ref().expect("boundary");
                    (bp.need[0], bp.need[1])
                };
                if !need0 && !need1 {
                    break; // boundary satisfied
                }
                let cmd = match cmds.front() {
                    Some(c) => c.clone(),
                    None => return, // PAUSE — out of choices; resume on the next one
                };
                let s = cmd.side;
                if !self.boundary.as_ref().expect("boundary").need[s] {
                    // A choice for a side this boundary does NOT request — an UPSTREAM engine
                    // desync (an extra/missing draw shifted a faint/forced-switch onto a side the
                    // recorded game did not have here). Stop gracefully (R20) so the seed anchor
                    // classifies it, instead of crashing.
                    self.stopped = true;
                    // ...but for the LIVE bridge, "stop gracefully" means EMIT NOTHING, EVER — the
                    // child keeps accepting CHOOSE lines and never answers, so poke-env waits on a
                    // battle message that can never arrive and the env's `step()` hangs until its
                    // 120s watchdog kills the whole run. Record it as FATAL so `sim_bridge` reports
                    // `__ERR__` (`gen3_bridge_stopped_failloud_v1`). The OFFLINE replay harnesses
                    // read `stopped`/the streams and ignore `fatal`, so their graceful-
                    // classification behaviour is unchanged.
                    let bp = self.boundary.as_ref().expect("boundary");
                    self.fatal = Some(fault(format!(
                        "upstream desync: CHOOSE for p{} but this boundary does not request it \
                         (kinds p1={:?} p2={:?}, needs p1={} p2={}). The bridge would go silent \
                         forever, so this fails loud.",
                        s + 1,
                        bp.kinds[0],
                        bp.kinds[1],
                        bp.need[0],
                        bp.need[1],
                    )));
                    return;
                }
                // Resolve the wire token (numeric slot OR a NAME) against THIS boundary's state,
                // exactly like Showdown's `side.chooseMove`/`chooseSwitch`.
                //
                // `gen3_bridge_unresolvable_choice_failloud_v1` — a NAME form that resolves
                // against NOTHING is FATAL, not a reject. The out-of-range fallback below makes
                // `choice_is_legal` reject the choice, which leaves the boundary open and
                // re-issues the SAME request; poke-env then re-sends the SAME unmappable name,
                // forever — an unbounded bridge↔Python spin that pins a core and floods stdout
                // (measured: 46 MB/s from one child) while the env's `step()` never returns. The
                // Struggle wedge was ONE instance of this class; rather than wait for the next
                // one, report it. A NUMERIC out-of-range choice is DIFFERENT: it is a legitimate,
                // tested reject-and-re-request (the forced-replacement resume gate and the 0-PP
                // gate both rely on it), and poke-env re-picks from a fresh request, so it cannot
                // spin — that path is untouched.
                if matches!(cmd.choice, WireChoice::MoveName(_) | WireChoice::SwitchSpecies(_)) {
                    let bs = self.battle.state().expect("state");
                    if resolve_choice(bs, s, &cmd.choice).is_none() {
                        let mon = &bs.sides[s].pokemon[bs.sides[s].active];
                        self.fatal = Some(malformed(format!(
                            "unresolvable choice for p{}: {:?} — active {} has moves {:?}; bench {:?}. \
                             Re-requesting would loop forever, so this fails loud.",
                            s + 1,
                            cmd.choice,
                            mon.species_id,
                            mon.set.moves,
                            bs.sides[s]
                                .pokemon
                                .iter()
                                .enumerate()
                                .filter(|(i, m)| *i != bs.sides[s].active && !m.fainted)
                                .map(|(_, m)| m.species_id.clone())
                                .collect::<Vec<_>>(),
                        )));
                        self.stopped = true;
                        return;
                    }
                }
                let kind_s = self.boundary.as_ref().expect("boundary").kinds[s];
                let resolved = {
                    let bs = self.battle.state().expect("state");
                    resolve_wire(bs, s, kind_s == SideRequest::ForceSwitch, &cmd.choice, dex).unwrap_or_else(|| {
                        match cmd.choice {
                            WireChoice::Switch(_) | WireChoice::SwitchSpecies(_) => {
                                Choice::Switch(bs.sides[s].pokemon.len())
                            }
                            _ => Choice::Move(bs.sides[s].pokemon[bs.sides[s].active].set.moves.len()),
                        }
                    })
                };
                cmds.pop_front();
                // Trapped-switch rejection at a MOVE boundary — the `|error|` + (hidden trap) the
                // re-request are SEPARATE chunks; the side still needs a choice.
                let (reject, firm) = {
                    let bs = self.battle.state().expect("state");
                    let locked = bs.sides[s].pokemon[bs.sides[s].active].move_locked();
                    let reject = kind_s == SideRequest::Move
                        && matches!(resolved, Choice::Switch(_))
                        && ((bs.is_trapped(s, dex) && has_live_bench(bs, s)) || locked);
                    let firm = locked || bs.trap_is_firm(s, dex);
                    (reject, firm)
                };
                // Every OTHER reject class (`gen3_choice_reject_framing_v1`). PROBE-MEASURED
                // (`harness/probe_choice_reject_framing.js`, the real sim):
                //   disabled move       -> `[Unavailable choice] Can't move: X's Y is disabled`
                //                          + a re-request TO THAT SIDE ONLY carrying
                //                          `"update":true` + `"disabledSource":""` on the slot
                //   switch into ACTIVE  -> `[Invalid choice] Can't switch: You can't switch to an
                //                          active Pokémon`, and NOTHING follows
                //   out-of-range move   -> `[Invalid choice] Can't move: Your X doesn't have a
                //                          move N`, and NOTHING follows
                // ...and in EVERY class the NON-offending side receives ZERO lines. THE RULE is
                // `Side.emitChoiceError`'s: `[Unavailable choice]` + re-issue IFF its update
                // callback actually CHANGED the request, else `[Invalid choice]` and nothing.
                let general = if reject {
                    None
                } else {
                    let bs = self.battle.state().expect("state");
                    classify_reject(bs, s, kind_s, &cmd.choice, &resolved, dex)
                };
                if let Some(rej) = general {
                    // Same bound as the trapped exchange — a deterministic client re-sending the
                    // same refused choice must fail loud, not spin (see `REJECT_STREAK_CAP`).
                    self.reject_streak += 1;
                    if self.reject_streak > REJECT_STREAK_CAP {
                        self.fatal = Some(malformed(format!(
                            "no-progress reject loop on p{}: {} consecutive rejects of {:?} at one \
                             boundary ({}). The client keeps re-sending a choice this boundary \
                             rejects, so nothing can advance — failing loud instead of spinning.",
                            s + 1,
                            self.reject_streak,
                            cmd.choice,
                            rej.message(),
                        )));
                        self.stopped = true;
                        return;
                    }
                    sink.error(s, rej.error_line());
                    if let RejectClass::Unavailable { ds_slot, .. } = rej {
                        // The ONLY class that re-issues. Emitted to `s` ALONE: the other side's
                        // already-accepted choice stands, so re-asking it would both duplicate a
                        // request the sim never sends and record a phantom extra pick.
                        let req = Request {
                            kind: kind_s.into(),
                            trapped: false,
                            update: true,
                            no_cancel: false,
                            disabled_source: Some(ds_slot as u8),
                        };
                        self.issue(s, req, sink, dex);
                    }
                    continue; // side s still needs a choice; the boundary stays open
                }
                if reject {
                    // Bound the reject↔re-request exchange (`REJECT_STREAK_CAP`).
                    self.reject_streak += 1;
                    if self.reject_streak > REJECT_STREAK_CAP {
                        let bs = self.battle.state().expect("state");
                        let mon = &bs.sides[s].pokemon[bs.sides[s].active];
                        self.fatal = Some(malformed(format!(
                            "no-progress reject loop on p{}: {} consecutive rejects of {:?} at one \
                             boundary (active {}, trapped={}, firm={}, move_locked={}). The client \
                             keeps re-sending a choice this boundary rejects, so nothing can \
                             advance — failing loud instead of spinning.",
                            s + 1,
                            self.reject_streak,
                            cmd.choice,
                            mon.species_id,
                            bs.is_trapped(s, dex),
                            firm,
                            mon.move_locked(),
                        )));
                        self.stopped = true;
                        return;
                    }
                    if firm {
                        sink.error(s, "|error|[Invalid choice] Can't switch: The active Pokémon is trapped".to_string());
                    } else {
                        sink.error(
                            s,
                            "|error|[Unavailable choice] Can't switch: The active Pokémon is trapped".to_string(),
                        );
                        // The RE-ISSUED frame supersedes the one this reject answered: the
                        // boundary is still open and the side must pick again FROM THIS request
                        // (it now carries `trapped:true` + `"update":true`).
                        let req = Request {
                            kind: RequestState::Move,
                            trapped: true,
                            update: true,
                            no_cancel: false,
                            disabled_source: None,
                        };
                        self.issue(s, req, sink, dex);
                    }
                    continue; // side s still needs a choice
                }
                // A choice was ACCEPTED — the boundary is making progress, so the streak resets.
                self.reject_streak = 0;
                let bp = self.boundary.as_mut().expect("boundary");
                bp.got[s] = Some(resolved);
                bp.need[s] = false;
            }

            // ── Boundary satisfied → struggle announce, commit, feed the driver. ──
            let bp = self.boundary.take().expect("boundary");
            // The boundary is CLOSING, so nothing is outstanding until the next one opens.
            self.void(sink);
            {
                let bs = self.battle.state().expect("state");
                for s in 0..2 {
                    if matches!(bp.got[s], Some(Choice::Move(_))) {
                        let mon = &bs.sides[s].pokemon[bs.sides[s].active];
                        if !mon.move_locked() && mon.must_struggle(dex) {
                            let name = display_name(mon, dex);
                            sink.struggle(s, format!("|-activate|p{}a: {}|move: Struggle", s + 1, name));
                        }
                    }
                }
            }
            let mut dec = ScriptDecision::default();
            for s in 0..2 {
                if let Some(c) = bp.got[s] {
                    dec.set_side(s, c);
                }
            }
            sink.commit(dec);
            {
                let driver = &mut self.driver;
                let bs = self.battle.state_mut().expect("state");
                driver.feed(bs, dec, dex);
            }
            // loop back → start a new boundary
        }
    }
}
