//! The Showdown text protocol — the wire format both into and out of the sim.
//!
//! **Bit-for-bit pressure point.** Our poke-env fork parses the `|...|` output
//! lines (`|move|`, `|switch|`, `|-damage|`, `|-crit|`, `|-status|`, `|turn|`,
//! `|win|`, …). A faithful port must reproduce them exactly — same tokens, same
//! order, same HP-fraction formatting. The one documented exception is `|t:|`
//! wall-clock lines, which poke-env ignores.
//!
//! This module carries the line/choice TYPES **and** the emission layer:
//! [`ProtocolBuilder`], an append-only line buffer the engine (`turn.rs`) writes
//! into at each observable event — with ONE sim-mirroring exception,
//! [`ProtocolBuilder::attr_last_move_still`], the port of `attrLastMove('[still]')`
//! (a retro-edit of the last `|move|` line, for fail forms the sim itself decides
//! retroactively). The builder centralizes the fiddly formatting
//! (HP fractions, the `p1a:`/`p1:` identifier split, the `[from]`/`[of]` tags) so
//! it lives in ONE place, not scattered through the engine.
//!
//! # OBSERVATION-ONLY (the load-bearing contract)
//!
//! Emission is a **side output** of an event that ALREADY happened. Pushing a
//! formatted string draws NO PRNG and reads only already-computed state. The whole
//! seed suite (battle/e2e/fullbattle/secondary/…) MUST stay green with IDENTICAL
//! seed assertions after wiring emission — that is the proof the emission did not
//! perturb the engine. See `PROTOCOL_EMISSION_DESIGN.md` §e.
//!
//! # Phase 1 scope (what the engine emits today)
//!
//! The high-frequency CORE: the battle-init framing (`|t:|`/`|gametype|`/`|player|`
//! /`|gen|`/`|tier|`/`|rule|`/`|teamsize|`/`|start`/the blank `|` separator), the
//! turn/phase markers (`|turn|N`, `|upkeep`), `|move|` (+ `[still]`/`[miss]`),
//! `|switch|`/`|drag|`, `|-damage|` (all HP variants + the residual `[from]` tags),
//! `|-heal|` (+ `[from] item:`), `|faint|`, `|-crit|`, `|-supereffective|`,
//! `|-resisted|`, `|-immune|`, `|-miss|`, and `|win|`/`|tie|`. Later phases add the
//! weather/boost/ability/status/volatile/side-condition lines. `protocol_test.rs`
//! is honest about the split: it asserts ONLY the Phase-1 types (deferred types are
//! filtered from BOTH the golden and the engine output — a real subset-equality).

use std::fmt;

use crate::core_events::{Cause as LineCause, Field, Ident, Kw, Line, Scope, SourceRec};

/// Which player a command or output line belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    P1,
    P2,
}

impl Player {
    /// The protocol tag, e.g. `"p1"` / `"p2"`.
    pub fn tag(self) -> &'static str {
        match self {
            Player::P1 => "p1",
            Player::P2 => "p2",
        }
    }

    /// From a 0-based side index (`0 → p1`, `1 → p2`).
    pub fn from_side(side: usize) -> Player {
        if side == 0 {
            Player::P1
        } else {
            Player::P2
        }
    }
}

/// One raw output line, e.g. `"|move|p1a: Tyranitar|Rock Slide|p2a: Starmie"`.
///
/// Kept as an exact string (not a parsed struct) on purpose: the contract is
/// byte-equality with Showdown, so the raw bytes are the source of truth. A
/// typed view can be layered on top later without changing that contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolLine(pub String);

impl fmt::Display for ProtocolLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A side's decision for a turn. Serializes to Showdown's choice grammar
/// (`"move 1"`, `"move 1 mega"`, `"switch 3"`, `"default"`, `"pass"`), which is
/// what the `>p1 …` command line carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Choice {
    /// `move <1-4>` — a move slot (1-indexed, as on the wire).
    Move { slot: u8 },
    /// `switch <1-6>` — switch to a team slot (1-indexed).
    Switch { slot: u8 },
    /// `default` — let the sim pick the forced/only legal action.
    Default,
    /// `pass` — no action this turn (e.g. nothing to do after a faint).
    Pass,
}

impl Choice {
    /// Render to the wire grammar (the text after `>p1 ` / `>p2 `).
    pub fn to_wire(&self) -> String {
        match self {
            Choice::Move { slot } => format!("move {slot}"),
            Choice::Switch { slot } => format!("switch {slot}"),
            Choice::Default => "default".to_string(),
            Choice::Pass => "pass".to_string(),
        }
    }
}

/// A mon reference, rendered `p<N><pos>: <Nickname>` — gen-3 singles is always
/// position `a`. The `name` is the on-field IDENTIFIER = the packed set's NICKNAME
/// (Showdown's `Pokemon.name` = `set.name || species.name`), e.g. `Electhor` for a
/// Zapdos nicknamed `Electhor`, falling back to the species display name only when
/// the set has no nickname. This is the token poke-env keys each mon by, so it MUST
/// be the nickname — never the species (the species belongs in the `|switch|`
/// details field). See `turn.rs::display_name` / `species_name`.
///
/// A **side** reference (side conditions) is `p<N>: <PlayerName>` — no position
/// letter — via [`ProtocolBuilder::side_ref`].
#[derive(Debug, Clone)]
pub struct MonRef {
    pub side: usize,
    pub name: String,
}

impl fmt::Display for MonRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}a: {}", Player::from_side(self.side).tag(), self.name)
    }
}

/// The HP field of an HP-bearing line, rendering the three variants from the
/// inventory (the #1 fiddly rule):
/// - **Healthy:** `224/341`
/// - **With a major status appended (single space):** `116/524 slp`
/// - **Fainted:** `0 fnt` (the literal — NOT `0/341`).
///
/// This is the SINGLE home for HP formatting; every HP-bearing line (`switch`/
/// `drag`/`-damage`/`-heal`) renders through it.
#[derive(Debug, Clone)]
pub struct HpStatus {
    pub hp: u16,
    pub maxhp: u16,
    /// The lowercase status token (`brn`/`par`/`slp`/`frz`/`psn`/`tox`), or `None`.
    /// Ignored when `hp == 0` (a fainted mon renders `0 fnt`).
    pub status: Option<&'static str>,
}

impl fmt::Display for HpStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.hp == 0 {
            return write!(f, "0 fnt");
        }
        match self.status {
            Some(s) => write!(f, "{}/{} {}", self.hp, self.maxhp, s),
            None => write!(f, "{}/{}", self.hp, self.maxhp),
        }
    }
}

/// The `[from] <cause>` provenance tag on a `-damage`/`-heal`/`-status`/`-weather`
/// line. Renders the forms the inventory catalogues (the `item:`/`ability:`/`move:`
/// prefixes, or a bare field cause like `Sandstorm`/`psn`).
#[derive(Debug, Clone)]
pub enum Cause {
    /// `[from] item: <Item>` (e.g. `Leftovers`).
    Item(String),
    /// `[from] ability: <Ability>` (e.g. `Sand Stream`).
    Ability(String),
    /// `[from] move: <Move>` (e.g. `Rest`).
    Move(String),
    /// `[from] <bare>` — a field/residual cause with no prefix (e.g. `Sandstorm`,
    /// `psn`, `brn`, `Leech Seed`, `Spikes`).
    Bare(String),
}

impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cause::Item(name) => write!(f, "[from] item: {name}"),
            Cause::Ability(name) => write!(f, "[from] ability: {name}"),
            Cause::Move(name) => write!(f, "[from] move: {name}"),
            Cause::Bare(text) => write!(f, "[from] {text}"),
        }
    }
}

/// The lowercase protocol token for a boostable stat, as `-boost`/`-unboost` render
/// it — indexed by the [`crate::state::MonState::boosts`] layout `[atk, def, spa, spd,
/// spe, accuracy, evasion]`.
pub const STAT_TOKENS: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

/// Accumulates the omniscient protocol stream as the engine runs. The engine holds
/// ONE of these on `BattleState` (the `log: ProtocolBuilder` field) and pushes
/// lines at the hook points. [`drain`](ProtocolBuilder::drain) hands the
/// accumulated lines to the caller after the battle (or per decision, for a
/// streaming bridge).
///
/// **Append-only, PRNG-free:** every method formats already-decided values and
/// pushes one line; none reads or draws from the PRNG. This is the observation-only
/// guarantee at the type level.
///
/// **Typed at the source** (`gen3_core_events_v1`): every method builds the line as a typed
/// [`Line`] and the text is [`Line::render`] of it, so the typed value and the bytes can never
/// disagree. There is no raw-string escape hatch. With [`Self::record_core`] on, the builder also
/// keeps one [`SourceRec`] per committed line (the typed line, the turn, the engine's action
/// [`Scope`]) — conservation at the source: exactly one record per line, by construction.
#[derive(Debug, Clone, Default)]
pub struct ProtocolBuilder {
    lines: Vec<ProtocolLine>,
    /// Whether emission is enabled. Off by default so the seed suite (which never
    /// drains the log) pays zero formatting cost AND so the observation-only proof
    /// is unmistakable: with `enabled == false` NOTHING is built or pushed.
    /// `run_full_battle` leaves it off; `run_full_battle_logged` turns it on. Either way NO
    /// PRNG is touched (the whole module is draw-free).
    enabled: bool,
    /// The set of `-hint` messages ALREADY emitted this battle. The sim's `Battle.hint()`
    /// dedups against `this.hints` (a Set), so each distinct hint text fires at most ONCE per
    /// battle (`gen3_omniscient_byte_fuzz_v1` FORM 14 — the Knock Off hint over-emission: the
    /// port re-emitted it on every 2nd+ Knock Off). Reset per battle (a fresh `BattleState` =
    /// a fresh builder). PRNG-free like everything here.
    hints_shown: std::collections::HashSet<String>,
    /// A ONE-SHOT `[from] <Effect>` attr the NEXT `|move|` announce carries
    /// (`gen3_move_coverage_batch5_v1` — the Sleep Talk CALLED move's
    /// `|move|<user>|<Picked>|<target>|[from] Sleep Talk`, byte-exact: ONE space, NO
    /// `move:` prefix; the sim's `useMove(…, sourceEffect)` folds it into the announce
    /// ITSELF, so a later `attrLastMove('[miss]')` lands AFTER it — which is why this
    /// is an announce-time fold, not a retro-edit). Set via [`Self::set_next_move_from`]
    /// just before the recursive called-move run; consumed (cleared) by the next
    /// `move_used`. PRNG-free like everything here.
    pending_move_from: Option<String>,
    /// The index (into `lines`) just past the most recent `|turn|N` marker — the
    /// sim's "SEND boundary" (`gen3_omniscient_byte_fuzz_v1`, class A: the spurious
    /// `[miss]` retro-tag). Showdown STREAMS each turn's lines at `makeRequest`, so an
    /// already-sent (prior-turn) `|move|` line can NEVER be observably mutated by a
    /// LATER retro-edit; only the port's single-buffer diff can. A resolving
    /// Future Sight / Doom Desire MISS (turn.rs `apply_future_move`) is the ONE
    /// retro-edit that can cross a turn boundary — its `attrLastMove('[miss]')` would
    /// tag whatever the LAST `|move|` line is, which on a switch-only resolve turn is a
    /// PRIOR-turn move line. Modeling the send boundary makes `attr_last_move_miss` /
    /// `attr_last_move_still` NO-OP on an already-flushed (index < `flush_boundary`)
    /// line, so a cross-turn retro-tag can't corrupt a streamed line. Set in `turn()`
    /// (points just past the `|turn|N` marker = every current-turn move line is after
    /// it, every prior-turn one before it), reset to 0 in `drain()` (a drain is the
    /// bridge/writeline "send"). All same-turn retro-edits (the other 5 miss sites +
    /// every `[still]` site) have their `|move|` line AFTER the current `|turn|` marker
    /// (`idx >= flush_boundary`) so the gate is a no-op for them. PRNG-free,
    /// state-free, emission-only.
    flush_boundary: usize,
    /// The core's source records (`gen3_core_events_v1`), one per committed line, when
    /// [`Self::record_core`] is on. ABSOLUTE indices: a record survives `drain()`.
    recs: Option<Vec<SourceRec>>,
    /// Lines already handed out by `drain()` (a buffer index + this = a record index).
    drained: usize,
    /// The last `|turn|N` marker's N (0 before the first) — a record's `turn`.
    cur_turn: u32,
    /// Which engine action is running — set by the turn loop (`turn/driver.rs`) and around
    /// Pursuit's in-switch strike (`turn/switch.rs`). A record's `scope`: the sim's own answer to
    /// "whose move owns this line", which the protocol never prints. Plain field writes, no draw.
    pub scope: Scope,
}

impl ProtocolBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Arm the ONE-SHOT `[from] <Effect>` attr for the NEXT `|move|` announce (the
    /// Sleep Talk called-move tag). No-op while disabled (the seed suite's zero-cost
    /// invariant — nothing is stored or read).
    pub fn set_next_move_from(&mut self, effect: &str) {
        if !self.enabled {
            return;
        }
        self.pending_move_from = Some(effect.to_string());
    }

    /// Enable emission (the logged battle path calls this). The seed suite never
    /// does, so it keeps an empty, cost-free buffer.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set emission on/off — used to briefly SUPPRESS a nested apply's default emit so a
    /// wrapper can emit the correct form (the Synchronize reflect: the recursive
    /// `try_set_status` would push a bare `|-status|`, but the reflect needs the
    /// `[from] ability: Synchronize|[of]` reveal, which the wrapper emits after restoring).
    /// Save `is_enabled()` first; always restore. Draw-free (no PRNG).
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }

    /// Turn on the core's source recording (`gen3_core_events_v1`): from now on every committed
    /// line also keeps its [`SourceRec`]. Idempotent. Nothing in production turns it on.
    pub fn record_core(&mut self) {
        if self.recs.is_none() {
            self.recs = Some(Vec::new());
        }
    }

    /// The source records so far (absolute order), if recording.
    pub fn source_recs(&self) -> Option<&[SourceRec]> {
        self.recs.as_deref()
    }

    /// The accumulated lines so far (read-only view).
    pub fn lines(&self) -> &[ProtocolLine] {
        &self.lines
    }

    /// Drain everything emitted so far (the per-decision / per-battle batch the
    /// bridge relays). Leaves the buffer empty.
    pub fn drain(&mut self) -> Vec<ProtocolLine> {
        // A drain is the bridge/writeline "send" — every buffered line is now flushed, so
        // reset the send boundary (the next batch has no already-sent prior-turn move line
        // until a fresh `|turn|` marker re-arms it). `gen3_omniscient_byte_fuzz_v1` class A.
        self.flush_boundary = 0;
        self.drained += self.lines.len();
        std::mem::take(&mut self.lines)
    }

    /// The ONE place a line is committed: render it, append it, record it. A method builds its
    /// [`Line`] inside `build` so a disabled builder builds nothing.
    fn emit(&mut self, build: impl FnOnce() -> Line) {
        if !self.enabled {
            return;
        }
        let line = build();
        let text = line.render();
        // The EMISSION SELF-CHECK (`crate::emission_check`): compiled out of `--release`.
        #[cfg(any(debug_assertions, feature = "emission-selfcheck"))]
        crate::emission_check::check_omniscient(&line, &text);
        self.lines.push(ProtocolLine(text));
        if let Some(recs) = &mut self.recs {
            recs.push(SourceRec { line, turn: self.cur_turn, scope: self.scope });
        }
    }

    /// Retro-edit the line at buffer index `idx` through its typed form (the port of
    /// `attrLastMove`): parse the committed text, apply `edit`, re-render — and, when
    /// recording, keep the record in lockstep (a record that does not re-parse from its own
    /// text is a builder bug, refused loudly).
    fn retro_edit(&mut self, idx: usize, edit: impl FnOnce(&mut Line)) {
        let mut line = Line::parse(&self.lines[idx].0).expect("a committed line re-parses");
        if let Some(recs) = &self.recs {
            assert_eq!(recs[self.drained + idx].line, line, "a source record disagrees with its own text");
        }
        edit(&mut line);
        let text = line.render();
        #[cfg(any(debug_assertions, feature = "emission-selfcheck"))]
        crate::emission_check::check_omniscient(&line, &text);
        self.lines[idx].0 = text;
        if let Some(recs) = &mut self.recs {
            recs[self.drained + idx].line = line;
        }
    }

    // ── A side reference `p<N>: <PlayerName>` (side conditions, no position). ──
    pub fn side_ref(side: usize, player_name: &str) -> String {
        format!("{}: {}", Player::from_side(side).tag(), player_name)
    }

    // ── Battle-init framing ────────────────────────────────────────────────────
    // Emitted once at battle start, in the sim's order (see the golden's first
    // ~14 lines). `|t:|` is the wall-clock line — the port emits the NORMALIZED
    // placeholder the golden stores (the value is un-reproducible + poke-env-
    // ignored; the byte test asserts its POSITION, not its timestamp).

    pub fn timestamp(&mut self) {
        self.emit(|| Line::new(Kw::Timestamp, vec![Field::text("<NORMALIZED>")]));
    }
    pub fn gametype_singles(&mut self) {
        self.emit(|| Line::new(Kw::Gametype, vec![Field::text("singles")]));
    }
    /// `|player|p<N>|<name>|<avatar>|<rating>` — avatar+rating empty (two trailing
    /// pipes), matching the capture (`|player|p1|P1||`).
    pub fn player(&mut self, side: usize, name: &str) {
        self.emit(|| {
            Line::new(Kw::Player, vec![Field::text(Player::from_side(side).tag()), Field::text(name), Field::Empty, Field::Empty])
        });
    }
    pub fn gen(&mut self, gen: u8) {
        self.emit(|| Line::new(Kw::Gen, vec![Field::text(gen.to_string())]));
    }
    pub fn tier(&mut self, display: &str) {
        self.emit(|| Line::new(Kw::Tier, vec![text_or_empty(display)]));
    }
    pub fn rule(&mut self, text: &str) {
        self.emit(|| Line::new(Kw::Rule, vec![text_or_empty(text)]));
    }
    pub fn teamsize(&mut self, side: usize, count: usize) {
        self.emit(|| Line::new(Kw::Teamsize, vec![Field::text(Player::from_side(side).tag()), Field::text(count.to_string())]));
    }
    pub fn start(&mut self) {
        self.emit(|| Line::new(Kw::BareStart, vec![]));
    }

    // ── Turn / phase markers ────────────────────────────────────────────────────
    /// The blank `|` separator line (emitted between phases — poke-env parses it as
    /// a no-op). The port MUST emit these for byte-equality.
    pub fn separator(&mut self) {
        self.emit(|| Line::new(Kw::Separator, vec![]));
    }
    pub fn turn(&mut self, n: u32) {
        if self.enabled {
            self.cur_turn = n;
        }
        self.emit(|| Line::new(Kw::Turn, vec![Field::text(n.to_string())]));
        // Model the sim's SEND boundary: the `|turn|N` marker is the point Showdown streams
        // the accumulated batch, so every line up to and including it is "sent". Point the
        // boundary just past it — every current-turn move line is emitted after it (retro-edits
        // still apply), every prior-turn move line is before it (retro-edits no-op). Only
        // meaningful when emission is enabled; `emit` no-ops otherwise but `lines.len()` is
        // then 0 so the boundary stays 0 (harmless). `gen3_omniscient_byte_fuzz_v1` class A.
        self.flush_boundary = self.lines.len();
    }
    pub fn upkeep(&mut self) {
        self.emit(|| Line::new(Kw::Upkeep, vec![]));
    }

    /// Advance the SEND boundary to the current buffer length — the port's model of the
    /// sim's `sendUpdates()` FLUSH at a `makeRequest('switch')` (a mid-turn forced
    /// replacement), which STREAMS every buffered line to the client. Every line up to
    /// now is "sent", so a LATER retro-edit (`attr_last_move_miss` / `attr_last_move_still`)
    /// can no longer observably mutate it. `turn()` already advances the boundary for the
    /// move request; this covers the OTHER `makeRequest` point (the forced-replacement
    /// pause), so a future-move resolve-MISS at the end-of-turn residual does NOT retro-tag
    /// a `|move|` line that a mid-turn faint already flushed (golden ab_20_4 L79: the
    /// Fire Blast line stays untagged; the `-miss` lands on the resolve).
    /// `gen3_omniscient_byte_fuzz_v1` class A. No-op formatting cost when disabled (the
    /// boundary is only consulted by the retro-edits, themselves gated on `enabled`).
    pub fn mark_sent(&mut self) {
        self.flush_boundary = self.lines.len();
    }

    // ── Move / action ───────────────────────────────────────────────────────────
    /// `|move|<user>|<MoveName>|<target>` — the target is the opposing (or self)
    /// active's ident. A `[still]` move (did nothing observable) renders with an
    /// EMPTY target field (two pipes) + the `[still]` tag; a `[miss]` appends the
    /// miss tag (paired with a `|-miss|`).
    pub fn move_used(&mut self, user: &MonRef, move_name: &str, target: Option<&MonRef>, miss: bool, still: bool) {
        if !self.enabled {
            return;
        }
        // Consume the one-shot `[from] <Effect>` attr (the Sleep Talk called move).
        let from = self.pending_move_from.take();
        self.emit(|| {
            let mut f = vec![Field::mon(user), text_or_empty(move_name)];
            if still {
                // Empty target field + [still] (e.g. Protect: `|move|…|Protect||[still]`).
                f.push(Field::Empty);
                f.push(Field::tag("[still]"));
                if let Some(e) = &from {
                    f.push(Field::from_text(e));
                }
            } else if let Some(t) = target {
                f.push(Field::mon(t));
                // The `[from]` fold precedes a `[miss]` (the announce carries the from-attr;
                // the miss is a LATER attrLastMove append).
                if let Some(e) = &from {
                    f.push(Field::from_text(e));
                }
                if miss {
                    f.push(Field::tag("[miss]"));
                }
            } else {
                f.push(Field::Empty);
            }
            Line::new(Kw::Move, f)
        });
    }

    /// Retro-edit the MOST RECENT `|move|` line into the `[still]` did-nothing form —
    /// the port of `Battle.attrLastMove('[still]')` (battle.js: blank the TARGET field,
    /// `parts[4] = ''`, then append `|[still]`), turning
    /// `|move|p1a: Suicune|Disable|p2a: Blissey` into `|move|p1a: Suicune|Disable||[still]`.
    ///
    /// Showdown decides some announce forms RETROACTIVELY — the `|move|` line is pushed
    /// BEFORE the draws that decide the outcome (e.g. Disable's gen4-inherited onStart
    /// 0-PP guard rejects the volatile AFTER the accuracy + `random(2,6)` draws). This
    /// mirrors that mechanism for the arms whose fail form cannot be decided up-front
    /// (`gen3_disable_zero_pp_v1`); the one deliberate exception to the buffer's
    /// otherwise append-only discipline (still PRNG-free, still observation-only).
    /// The current move's announce is always the most recent `|move|` line (nothing
    /// emits another `|move|` between an announce and its fail), matching the sim's
    /// `lastMoveLine` bookkeeping. No-op when disabled or when no `|move|` line exists.
    pub fn attr_last_move_still(&mut self) {
        if !self.enabled {
            return;
        }
        // NO-OP on an already-SENT (prior-turn) `|move|` line — see `flush_boundary`. Every
        // `attr_last_move_still` caller is SAME-TURN (its move line is emitted after the current
        // `|turn|` marker → `idx >= flush_boundary` → the edit still applies), so this gate is
        // load-bearing only for consistency with `attr_last_move_miss`.
        if let Some(idx) = self.lines.iter().rposition(|l| l.0.starts_with("|move|")) {
            if idx < self.flush_boundary {
                return;
            }
            self.retro_edit(idx, |l| {
                // fields: [<user>, <MoveName>, <target>, ...attrs]
                if l.fields.len() >= 3 {
                    l.fields[2] = Field::Empty;
                }
                l.fields.push(Field::tag("[still]"));
            });
        }
    }

    /// The `attrLastMove('[miss]')` retro-edit (Phase 3, `gen3_protocol_phase3_v1`):
    /// append `|[miss]` to the most recent `|move|` line WITHOUT blanking its target
    /// (the sim's accuracy-fail path on a STATUS move whose announce already showed —
    /// Disable/Hypnosis/Will-O-Wisp/Leech Seed: `|move|<user>|<Name>|<target>|[miss]`
    /// then `|-miss|`; byte-verified vs the Phase-3 capture golden). Same append-only
    /// exception discipline as [`Self::attr_last_move_still`]. No-op when disabled.
    pub fn attr_last_move_miss(&mut self) {
        if !self.enabled {
            return;
        }
        // NO-OP on an already-SENT (prior-turn) `|move|` line — see `flush_boundary`. The ONE
        // caller that can cross a turn boundary is the future-move (Future Sight / Doom Desire)
        // resolve MISS in `apply_future_move`: on a switch-only resolve turn the last `|move|`
        // line is a PRIOR-turn move that Showdown already STREAMED and can never re-tag, so
        // tagging it here would corrupt an already-sent line (golden L237: the sim leaves the
        // prior turn's `Ice Beam|p1a: Jirachi` untouched + emits the `-miss` on the resolve
        // turn; the port used to append `|[miss]`). The other 5 miss sites are SAME-TURN
        // (`idx >= flush_boundary`) so the gate is a no-op for them.
        // `gen3_omniscient_byte_fuzz_v1` class A.
        //
        // Matches the last `|move|` OR `|-anim|` line — the sim's `attrLastMove` tracks the
        // last `addMove` line, and a SUN-SKIP Solar Beam's most recent move-family line is the
        // `|-anim|` (emitted after `[still]`+`-prepare`); a missed sun-skip beam appends `[miss]`
        // to THAT anim line (`|-anim|<u>|Solar Beam|<t>|[miss]`), not the `[still]` charge line.
        if let Some(idx) = self
            .lines
            .iter()
            .rposition(|l| l.0.starts_with("|move|") || l.0.starts_with("|-anim|"))
        {
            if idx < self.flush_boundary {
                return;
            }
            self.retro_edit(idx, |l| l.fields.push(Field::tag("[miss]")));
        }
    }

    /// The `attrLastMove('[from] lockedmove')` retro-edit (`gen3_move_coverage_batch4c_v1`
    /// — Solar Beam's FIRE-turn announce: `|move|<user>|Solar Beam|<target>|[from] lockedmove`,
    /// the sim's `useMove` locked-move attr. SIM-PROBED (the omniscient stream, the sim is the
    /// oracle): there IS a SPACE after `[from]` (`|[from] lockedmove`) — the prior port emitted
    /// the no-space form, an omniscient + per-side byte divergence on any real gen3ou Solar Beam /
    /// Shiftry SolarBeam battle. Same append-only exception discipline as
    /// [`Self::attr_last_move_miss`]. No-op when disabled.
    pub fn attr_last_move_from_lockedmove(&mut self) {
        if !self.enabled {
            return;
        }
        if let Some(idx) = self.lines.iter().rposition(|l| l.0.starts_with("|move|")) {
            self.retro_edit(idx, |l| l.fields.push(Field::From(LineCause::Bare("lockedmove".into()))));
        }
    }

    /// `|-prepare|<mon>|<MoveName>` — the two-turn CHARGE announce (Solar Beam's charge
    /// turn, `gen3_move_coverage_batch4c_v1`; probe-observed shape, not yet byte-gated).
    pub fn prepare(&mut self, mon: &MonRef, move_name: &str) {
        self.emit(|| Line::new(Kw::Prepare, vec![Field::mon(mon), text_or_empty(move_name)]));
    }

    /// `|-anim|<mon>|<MoveName>|<target>` — the SUN-SKIP animation line (a charge move
    /// that fires immediately still emits `[still]` + `-prepare` THEN `-anim` before the
    /// damage lines; probe-observed shape, not yet byte-gated).
    pub fn anim(&mut self, mon: &MonRef, move_name: &str, target: &MonRef) {
        self.emit(|| Line::new(Kw::Anim, vec![Field::mon(mon), text_or_empty(move_name), Field::mon(target)]));
    }

    /// `|-mustrecharge|<mon>` — Hyper Beam's recharge-lock announce, printed right after
    /// the `|-damage|`/sub line of a successful hit (`gen3_move_coverage_batch4c_v1`;
    /// probe-observed shape, not yet byte-gated).
    pub fn must_recharge(&mut self, mon: &MonRef) {
        self.emit(|| Line::new(Kw::Mustrecharge, vec![Field::mon(mon)]));
    }

    // ── Switch / drag / faint ───────────────────────────────────────────────────
    /// `|switch|<mon>|<Details>|<HP>`. Details is the pre-built `Pokemon.details`
    /// string (`turn.rs::switch_details`): `<Species>[, L<level>][, <gender>][, shiny]`
    /// — `, L<n>` iff level != 100 (gen3ou is always L100 so it is omitted there),
    /// gender/shiny only when present. A gen-3-singles L100 genderless mon shows just
    /// the species name.
    pub fn switch(&mut self, mon: &MonRef, details: &str, hp: &HpStatus) {
        self.emit(|| Line::new(Kw::Switch, vec![Field::mon(mon), text_or_empty(details), Field::hp(hp)]));
    }
    /// `|switch|<mon>|<Details>|<HP>|[from] <Effect>` — a switch carrying a `[from]` tag
    /// (`gen3_move_coverage_batch3_v1`, the BATON PASS entry: `[from] Baton Pass`).
    pub fn switch_from(&mut self, mon: &MonRef, details: &str, hp: &HpStatus, effect: &str) {
        self.emit(|| {
            Line::new(Kw::Switch, vec![Field::mon(mon), text_or_empty(details), Field::hp(hp), Field::from_text(effect)])
        });
    }
    /// `|drag|<mon>|<Details>|<HP>` — identical grammar to `switch`; the FORCED
    /// (Roar/Whirlwind) entry.
    pub fn drag(&mut self, mon: &MonRef, details: &str, hp: &HpStatus) {
        self.emit(|| Line::new(Kw::Drag, vec![Field::mon(mon), text_or_empty(details), Field::hp(hp)]));
    }
    pub fn faint(&mut self, mon: &MonRef) {
        self.emit(|| Line::new(Kw::Faint, vec![Field::mon(mon)]));
    }

    // ── Damage / heal ───────────────────────────────────────────────────────────
    /// `|-damage|<mon>|<HP>[|[from] <cause>]`. The residual forms carry a `[from]`
    /// cause (`Sandstorm`/`psn`/`brn`/`tox`/`Spikes`/`Leech Seed`).
    pub fn damage(&mut self, mon: &MonRef, hp: &HpStatus, from: Option<&Cause>) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), Field::hp(hp)];
            if let Some(c) = from {
                f.push(Field::from(c));
            }
            Line::new(Kw::Damage, f)
        });
    }
    /// `|-damage|<mon>|<HP>|[from] <cause>|[of] <src>` — a damage line that carries BOTH
    /// a `[from]` cause AND an `[of]` source. The gen-3 **Struggle recoil**
    /// (`gen3_pp_tracking_v1`) emits `|-damage|<user>|<HP>|[from] Recoil|[of] <target>`
    /// (the target of the Struggle is the `[of]` source), matching the golden exactly.
    pub fn damage_of(&mut self, mon: &MonRef, hp: &HpStatus, from: &Cause, of: &MonRef) {
        self.emit(|| Line::new(Kw::Damage, vec![Field::mon(mon), Field::hp(hp), Field::from(from), Field::of(of)]));
    }
    /// `|-heal|<mon>|<HP>[|[from] <cause>]` (Leftovers carries `[from] item:
    /// Leftovers`; a self-heal move / Leech-Seed heal has no `[from]` here in the
    /// Phase-1 set — recovery moves are still deferred).
    pub fn heal(&mut self, mon: &MonRef, hp: &HpStatus, from: Option<&Cause>) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), Field::hp(hp)];
            if let Some(c) = from {
                f.push(Field::from(c));
            }
            Line::new(Kw::Heal, f)
        });
    }
    /// `|-heal|<mon>|<HP>|[from] <cause>|[of] <of>` — a heal that carries BOTH a `[from]`
    /// cause AND an `[of]` source. The DRAIN heal (`gen3_move_coverage_batch1_v1`, Giga
    /// Drain) emits `|-heal|<user>|<HP>|[from] drain|[of] <target>` (the drained mon is the
    /// `[of]` source), matching the sim exactly.
    pub fn heal_of(&mut self, mon: &MonRef, hp: &HpStatus, from: &Cause, of: &MonRef) {
        self.emit(|| Line::new(Kw::Heal, vec![Field::mon(mon), Field::hp(hp), Field::from(from), Field::of(of)]));
    }
    /// `|-heal|<mon>|<HP>|[from] move: Wish|[wisher] <name>` — the WISH delayed heal
    /// (`gen3_move_coverage_batch3_v1`). The `[wisher]` clause carries the CASTER's display
    /// name (stored at cast, so it survives the wisher fainting / switching / phazing).
    pub fn heal_wish(&mut self, mon: &MonRef, hp: &HpStatus, wisher: &str) {
        self.emit(|| {
            Line::new(Kw::Heal, vec![
                Field::mon(mon),
                Field::hp(hp),
                Field::From(LineCause::Move("Wish".into())),
                Field::Wisher(wisher.to_string()),
            ])
        });
    }
    /// `|-heal|<mon>|<HP>|[silent]` — a silent heal: Rest's full heal (`run_rest`) and the
    /// Leech Seed drain's seeder heal (`residuals`). The HP field carries any new status token.
    pub fn heal_silent(&mut self, mon: &MonRef, hp: &HpStatus) {
        self.emit(|| Line::new(Kw::Heal, vec![Field::mon(mon), Field::hp(hp), Field::tag("[silent]")]));
    }

    // ── Effectiveness / crit / miss / immune ────────────────────────────────────
    /// `|-supereffective|<mon>` — `<mon>` is the DEFENDER.
    pub fn supereffective(&mut self, mon: &MonRef) {
        self.emit(|| Line::new(Kw::Supereffective, vec![Field::mon(mon)]));
    }
    /// `|-resisted|<mon>` — the DEFENDER.
    pub fn resisted(&mut self, mon: &MonRef) {
        self.emit(|| Line::new(Kw::Resisted, vec![Field::mon(mon)]));
    }
    /// `|-crit|<mon>` — a critical hit on the DEFENDER.
    pub fn crit(&mut self, mon: &MonRef) {
        self.emit(|| Line::new(Kw::Crit, vec![Field::mon(mon)]));
    }
    /// `|-immune|<mon>` — effectiveness 0. (The ability form carrying `[from]
    /// ability:` is a later phase — the Phase-1 capture immune lines are bare.)
    pub fn immune(&mut self, mon: &MonRef) {
        self.emit(|| Line::new(Kw::Immune, vec![Field::mon(mon)]));
    }
    /// `|-immune|<mon>|[from] ability: <Ability>` — immunity granted by an ABILITY
    /// (`gen3_ability_batch2_v1`, Soundproof blocking a sound move). Observation-only.
    pub fn immune_from_ability(&mut self, mon: &MonRef, ability: &str) {
        self.emit(|| Line::new(Kw::Immune, vec![Field::mon(mon), Field::From(LineCause::Ability(ability.into()))]));
    }
    /// `|-immune|<mon>|<effect>|[from] ability: <Ability>` — an ability blocking a specific
    /// VOLATILE rather than the whole move (`gen3_confuse_ray_v1`: Own Tempo vs confusion emits
    /// `|-immune|<mon>|confusion|[from] ability: Own Tempo`). Probe-settled — the effect segment
    /// sits BETWEEN the mon and the `[from]`, which the plain `immune_from_ability` omits.
    /// Observation-only.
    pub fn immune_effect_from_ability(&mut self, mon: &MonRef, effect: &str, ability: &str) {
        self.emit(|| {
            Line::new(Kw::Immune, vec![Field::mon(mon), text_or_empty(effect), Field::From(LineCause::Ability(ability.into()))])
        });
    }
    /// `|-miss|<user>[|<target>]` — paired with the `|move|…|[miss]`.
    pub fn miss(&mut self, user: &MonRef, target: Option<&MonRef>) {
        self.emit(|| {
            let mut f = vec![Field::mon(user)];
            if let Some(t) = target {
                f.push(Field::mon(t));
            }
            Line::new(Kw::Miss, f)
        });
    }
    /// `|-miss|<user>|<target>` where the USER is a PRE-RENDERED ident string
    /// (`gen3_omniscient_byte_fuzz_v1`, the future-move `-miss` source ref). Showdown
    /// renders every protocol mon via `Pokemon.toString()` — active mons as the slot
    /// form `pNa: <Name>`, but a mon NOT on the field as the SLOT-LESS `pN: <Name>`
    /// (pokemon.ts:532-534). A resolving Future Sight / Doom Desire's CASTER may have
    /// switched out or fainted, so its `-miss` source must render slot-less
    /// (`p1: Jirachi`), NOT the active-slot `p1a: Jirachi` that [`miss`]'s `MonRef`
    /// always produces. The caller renders the correct form via
    /// [`crate::state::BattleState::mon_toref`]. Observation-only.
    pub fn miss_raw_user(&mut self, user: &str, target: &MonRef) {
        self.emit(|| Line::new(Kw::Miss, vec![ident_field(user), Field::mon(target)]));
    }

    // ── Status ───────────────────────────────────────────────────────────────────
    /// `|-status|<mon>|<status>[|[from] <cause>]` — a major status is SET. `status` ∈
    /// `brn`/`par`/`slp`/`frz`/`psn`/`tox`. A self-inflicted status (Rest) carries
    /// `[from] move: Rest`; a foe-inflicted status has no `[from]`.
    pub fn status(&mut self, mon: &MonRef, status: &str, from: Option<&Cause>) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), text_or_empty(status)];
            if let Some(c) = from {
                f.push(Field::from(c));
            }
            Line::new(Kw::Status, f)
        });
    }
    /// `|-status|<mon>|<status>|[from] ability: <Ability>|[of] <src>` — a status inflicted by
    /// a reactive ABILITY on another mon (`gen3_ability_batch2_v1`): the CONTACT_PROC procs
    /// (Static/Poison Point/Flame Body/Effect Spore) statusing the ATTACKER, and Synchronize
    /// reflecting a status back to the SOURCE. `src` is the ability HOLDER (the mon whose
    /// ability caused it). Observation-only.
    pub fn status_from_ability(&mut self, mon: &MonRef, status: &str, ability: &str, src: &MonRef) {
        self.emit(|| {
            Line::new(Kw::Status, vec![
                Field::mon(mon),
                text_or_empty(status),
                Field::From(LineCause::Ability(ability.into())),
                Field::of(src),
            ])
        });
    }
    /// `|-curestatus|<mon>|<status>[|[msg]]` — a status is CURED (wake / thaw / Rest
    /// natural wake). `[msg]` shows the client message (a natural sleep/freeze wake).
    pub fn curestatus(&mut self, mon: &MonRef, status: &str, msg: bool) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), text_or_empty(status)];
            if msg {
                f.push(Field::tag("[msg]"));
            }
            Line::new(Kw::Curestatus, f)
        });
    }
    /// `|-curestatus|<mon>|<status>|[silent]` — a status cured silently, as by Heal Bell's
    /// per-ally `cureStatus(true)` (`gen3_move_coverage_batch2_v1`). A BENCH mon renders
    /// SLOT-LESS (`p<N>: <Name>`, Showdown's off-field `Pokemon.toString()`), the active as a mon
    /// ref (`p<N>a: <Name>`); the caller passes the pre-formatted ident string.
    pub fn curestatus_silent(&mut self, ident: &str, status: &str) {
        self.emit(|| Line::new(Kw::Curestatus, vec![ident_field(ident), text_or_empty(status), Field::tag("[silent]")]));
    }
    /// `|-curestatus|<mon>|<status>|[from] ability: <Ability>|[silent]` — a status cured by
    /// a switch-out ABILITY (`gen3_omniscient_byte_fuzz_v1` FORM 4, Natural Cure): emitted
    /// BEFORE the outgoing mon's replacement `|switch|`/`|drag|` line. Silent (the client
    /// only sees the mon leave). Observation-only.
    pub fn curestatus_from_ability_silent(&mut self, mon: &MonRef, status: &str, ability: &str) {
        self.emit(|| {
            Line::new(Kw::Curestatus, vec![
                Field::mon(mon),
                text_or_empty(status),
                Field::From(LineCause::Ability(ability.into())),
                Field::tag("[silent]"),
            ])
        });
    }
    /// `|-cureteam|<mon>|[from] move: Aromatherapy` — Aromatherapy's team-cure banner
    /// (`gen3_move_coverage_batch2_v1`; unlike Heal Bell, Aromatherapy emits NO per-mon
    /// `-curestatus` line — a single `-cureteam` covers the whole side).
    pub fn cureteam_aromatherapy(&mut self, mon: &MonRef) {
        self.emit(|| Line::new(Kw::Cureteam, vec![Field::mon(mon), Field::From(LineCause::Move("Aromatherapy".into()))]));
    }
    /// `|-sideend|<side>|<Effect>` — a side condition ENDS naturally at its duration expiry
    /// (`gen3_move_coverage_batch2_v1`, Light Screen / Reflect). `<side>` is a side ref.
    pub fn sideend(&mut self, side_ref: &str, effect: &str) {
        self.emit(|| Line::new(Kw::Sideend, vec![side_ref_field(side_ref), text_or_empty(effect)]));
    }
    /// `|-curestatus|<mon>|<status>|[from] move: <Move>` — a status cured BY a move
    /// (`gen3_defrost_v1`: the frozen user of a `flags.defrost` move — Sacred Fire /
    /// Flame Wheel — thaws draw-free via `frz.onModifyMove` after a FAILED 1/5 thaw
    /// roll; the line is emitted BEFORE the user's own `|move|` line. Probe:
    /// `harness/probe_sacredfire_defrost.js`).
    pub fn curestatus_from_move(&mut self, mon: &MonRef, status: &str, move_name: &str) {
        self.emit(|| {
            Line::new(Kw::Curestatus, vec![Field::mon(mon), text_or_empty(status), Field::From(LineCause::Move(move_name.into()))])
        });
    }
    /// `|cant|<mon>|<reason>|<MoveName>|[of] <of>` — a blocked action attributed to another
    /// mon's ability (`gen3_ability_batch2_v1`, Damp cancelling Explosion): the CANT is on the
    /// ABILITY HOLDER (`mon`), the reason is `ability: Damp`, the move name, and `[of]` the
    /// move's user. Mirrors `this.add("cant", damp_holder, "ability: Damp", move, [of] user)`.
    pub fn cant_of_move(&mut self, mon: &MonRef, reason: &str, move_name: &str, of: &MonRef) {
        self.emit(|| Line::new(Kw::Cant, vec![Field::mon(mon), text_or_empty(reason), text_or_empty(move_name), Field::of(of)]));
    }
    /// `|cant|<mon>|<reason>[|<MoveName>]` — a blocked action (full-para / asleep /
    /// frozen / flinch). `reason` ∈ `par`/`slp`/`frz`/`flinch`/…. gen-3 emits no move
    /// name for the status/flinch cases in the capture.
    pub fn cant(&mut self, mon: &MonRef, reason: &str, move_name: Option<&str>) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), text_or_empty(reason)];
            if let Some(m) = move_name {
                f.push(text_or_empty(m));
            }
            Line::new(Kw::Cant, f)
        });
    }
    /// `|-message|<text>` — a clause notice (`Sleep Clause Mod activated.` /
    /// `Freeze Clause activated.`). COSMETIC to poke-env.
    pub fn clause_message(&mut self, text: &str) {
        self.emit(|| Line::new(Kw::Message, vec![text_or_empty(text)]));
    }

    // ── Boost / unboost ────────────────────────────────────────────────────────────
    /// `|-boost|<mon>|<stat>|<amount>` (positive stages) or `|-unboost|<mon>|<stat>|
    /// <amount>` (a drop) — chosen by the SIGN of `stages`. `amount` is the UNSIGNED
    /// magnitude. `stat_idx` indexes [`STAT_TOKENS`]. A zero delta emits nothing (the
    /// sim skips a no-op boost).
    pub fn boost(&mut self, mon: &MonRef, stat_idx: usize, stages: i8) {
        if stages == 0 {
            return;
        }
        let kw = if stages > 0 { Kw::Boost } else { Kw::Unboost };
        self.emit(|| boost_line(kw, mon, STAT_TOKENS[stat_idx], stages.unsigned_abs() as i32));
    }

    /// `|-boost|`/`|-unboost|<mon>|<stat>|<mag>` — a PRIMARY move boost, emitted
    /// UNCONDITIONALLY (INCLUDING a 0-magnitude delta at the ±6 cap —
    /// `gen3_omniscient_byte_fuzz_v1` FORM 10). Unlike [`boost`], this does NOT skip a zero
    /// delta: the sim's `boost()` emits `-boost/-unboost ... 0` for a primary move boost
    /// (`!isSecondary && !isSelf` → the `else if (!isSecondary && !isSelf)` branch) even when
    /// `boostBy == 0` (a Calm Mind / Agility into the +6 cap → `|-boost|…|spa|0`). The `-boost`
    /// vs `-unboost` choice mirrors the sim exactly (`msg = -unboost iff requested < 0 ||
    /// final_boost == -6`, using the REQUESTED sign, not the applied-delta sign). `applied` is
    /// the CLAMPED-applied magnitude (0 at the cap). SECONDARY / self / ability boosts keep the
    /// zero-skipping [`boost`] (they don't emit a 0-delta).
    pub fn boost_applied(&mut self, mon: &MonRef, stat_idx: usize, requested: i8, applied: i8, final_boost: i8) {
        let kw = if requested < 0 || final_boost == -6 { Kw::Unboost } else { Kw::Boost };
        self.emit(|| boost_line(kw, mon, STAT_TOKENS[stat_idx], applied.unsigned_abs() as i32));
    }

    /// `|-unboost|<mon>|atk|<mag>` — the Intimidate Atk-drop line, emitted UNCONDITIONALLY
    /// (INCLUDING `mag == 0`). Unlike [`boost`], this does NOT skip a zero delta: a REQUESTED
    /// stat drop clamped to 0 at the −6 floor STILL reports `|-unboost|<foe>|atk|0` in the sim
    /// (probe `harness/probe_intimidate_floor.js`), whereas [`boost`]'s zero-skip is for a
    /// genuine no-op boost. `mag` is the UNSIGNED applied magnitude (`0` or `1`).
    pub fn unboost_atk_applied(&mut self, mon: &MonRef, mag: u8) {
        self.emit(|| boost_line(Kw::Unboost, mon, STAT_TOKENS[0], mag as i32));
    }
    /// `|-boost|<mon>|<stat>|<n>` with an explicit magnitude, including `0` — the stat-pinch
    /// berry at the +6 cap (`items.rs`, the sim's `boost()` still reporting the 0 delta).
    pub fn boost_raw(&mut self, mon: &MonRef, stat: &str, n: i32) {
        self.emit(|| boost_line(Kw::Boost, mon, stat, n));
    }
    /// `|-boost|<mon>|<stat>|<n>|[from] item: <Item>` — an ITEM boost (the stat-pinch berries).
    pub fn boost_from_item(&mut self, mon: &MonRef, stat: &str, n: i32, item: &str) {
        self.emit(|| {
            let mut l = boost_line(Kw::Boost, mon, stat, n);
            l.fields.push(Field::From(LineCause::Item(item.into())));
            l
        });
    }
    /// `|-clearallboost` — Haze.
    pub fn clearallboost(&mut self) {
        self.emit(|| Line::new(Kw::Clearallboost, vec![]));
    }
    /// `|-clearnegativeboost|<mon>|[silent]` — White Herb's restore (after its `-enditem`).
    pub fn clearnegativeboost_silent(&mut self, mon: &MonRef) {
        self.emit(|| Line::new(Kw::Clearnegativeboost, vec![Field::mon(mon), Field::tag("[silent]")]));
    }

    // ── Weather / ability ────────────────────────────────────────────────────────
    /// `|-weather|<Weather>[|[from] ability: <A>|[of] <src>][|[upkeep]]`. The SET form
    /// (an ability turns the weather on) carries `[from] ability:` + `[of] <src>`; the
    /// per-turn TICK is `|-weather|<Weather>|[upkeep]`.
    pub fn weather(&mut self, weather: &str, from: Option<&Cause>, of: Option<&MonRef>, upkeep: bool) {
        self.emit(|| {
            let mut f = vec![text_or_empty(weather)];
            if let Some(c) = from {
                f.push(Field::from(c));
            }
            if let Some(src) = of {
                f.push(Field::of(src));
            }
            if upkeep {
                f.push(Field::tag("[upkeep]"));
            }
            Line::new(Kw::Weather, f)
        });
    }
    /// `|-ability|<mon>|<Ability>[|<detail>]` — an ability reveals/fires. The Intimidate
    /// switch-in line carries the `boost` detail (`|-ability|p2a: Salamence|Intimidate|
    /// boost`).
    pub fn ability(&mut self, mon: &MonRef, ability: &str, detail: Option<&str>) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), text_or_empty(ability)];
            if let Some(d) = detail {
                f.push(typed_or_text(Kw::Ability, 2, d));
            }
            Line::new(Kw::Ability, f)
        });
    }
    /// `|-formechange|<mon>|<Species>|[msg]|[from] ability: Forecast` — the FORECAST
    /// forme swap (`gen3_forecast_v1`, ROUND 35). `Pokemon.formeChange` is called by
    /// `forecast.onWeatherChange` as `formeChange(forme, this.effect, /*isPermanent*/ false,
    /// '0', '[msg]')`, and the `!isPermanent && source.effectType === 'Ability'` branch of
    /// `sim/pokemon.ts::formeChange` emits exactly
    /// `this.battle.add('-formechange', this, species.name, message, '[from] ability: Forecast')`
    /// — so the FORME's display name (`Castform-Rainy`), then the literal `[msg]`, then the
    /// `[from]` clause. The MON token is the mon's IDENT (its nickname / its BASE species
    /// name), never the forme: `isPermanent` is false, so `Pokemon.details` / `Pokemon.name`
    /// are NOT refreshed. Byte-probed (`harness/probe_r35_forecast_reporting.js` S1).
    pub fn forme_change_forecast(&mut self, mon: &MonRef, species: &str) {
        self.emit(|| {
            Line::new(Kw::Formechange, vec![
                Field::mon(mon),
                text_or_empty(species),
                Field::tag("[msg]"),
                Field::From(LineCause::Ability("Forecast".into())),
            ])
        });
    }
    /// `|-ability|<mon>|<Ability>|[silent]` — the switch-in ability reveal Showdown
    /// adds via `addSplit` (Pressure; the omniscient stream carries the secret line).
    /// Phase 3 (`gen3_protocol_phase3_v1`), byte-verified vs the capture golden.
    pub fn ability_silent(&mut self, mon: &MonRef, ability: &str) {
        self.emit(|| Line::new(Kw::Ability, vec![Field::mon(mon), text_or_empty(ability), Field::tag("[silent]")]));
    }
    /// `|-ability|<mon>|<Copied>|Trace|[from] ability: Trace|[of] <foe>` — the Trace
    /// copy announce at switch-in (`setAbility`'s reveal: the third field is the OLD
    /// ability, always `Trace` here). Phase 3, byte-verified vs the capture golden.
    pub fn ability_traced(&mut self, mon: &MonRef, copied: &str, foe: &MonRef) {
        self.emit(|| {
            Line::new(Kw::Ability, vec![
                Field::mon(mon),
                text_or_empty(copied),
                Field::text("Trace"),
                Field::From(LineCause::Ability("Trace".into())),
                Field::of(foe),
            ])
        });
    }
    /// `|-fail|<mon>|unboost|[<Stat>|][from] ability: <Blocker>|[of] <mon>` — a stat-drop
    /// blocked by the target's own ability. The `[of]` is the BLOCKER itself (its own ability
    /// saved it). The `stat` token MIRRORS the sim's per-ability `onTryBoost` `this.add`:
    /// the WHOLE-table blockers (Clear Body / White Smoke) pass `None` (no stat token — they
    /// delete every negative boost), while the SINGLE-STAT blockers pass the ability's literal
    /// token — **Hyper Cutter → `"Attack"`**, **Keen Eye → `"accuracy"`** (SIM-PROBED forms,
    /// abilities.js: `add("-fail", target, "unboost", "Attack"|"accuracy", …)`). The prior
    /// port dropped Hyper Cutter's `Attack` token → an omniscient byte divergence on any
    /// Intimidate/Charm/Feather-Dance-into-Hyper-Cutter matchup.
    pub fn fail_unboost_from_ability(&mut self, mon: &MonRef, ability: &str, stat: Option<&str>) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), Field::text("unboost")];
            if let Some(s) = stat {
                f.push(text_or_empty(s));
            }
            f.push(Field::From(LineCause::Ability(ability.into())));
            f.push(Field::of(mon));
            Line::new(Kw::Fail, f)
        });
    }
    /// `|-fail|<mon>|[from] ability: <Ability>|[of] <mon>` — a move blocked by the user's OWN
    /// ability with no detail token (`status_moves.rs`, the sleep-immune Rest: Insomnia /
    /// Vital Spirit, `gen3_rest_sleep_immune_v1`).
    pub fn fail_from_ability_of(&mut self, mon: &MonRef, ability: &str) {
        self.emit(|| {
            Line::new(Kw::Fail, vec![Field::mon(mon), Field::From(LineCause::Ability(ability.into())), Field::of(mon)])
        });
    }
    /// `|-hint|<text>` — a client hint (poke-env-ignored, but part of the omniscient
    /// byte stream — e.g. the gen3 Intimidate-vs-Substitute no-op, the Knock Off item note).
    /// DEDUP semantics MIRROR the sim's `Battle.hint(hint, once, side)` (battle.ts:3092):
    /// it returns early if the text is ALREADY in `this.hints`, but only ADDS the text to
    /// `this.hints` **when `once` is true** — so a `once: true` hint (Knock Off, gen4/moves.ts:703)
    /// fires EXACTLY ONCE per battle, while a `once: false` hint (Sleep Clause, rulesets.ts:1395;
    /// the gen3 Intimidate-vs-Substitute + Pursuit notes, no `once` arg) fires on EVERY occurrence.
    /// This is `gen3_omniscient_byte_fuzz_v1` FORM 14 — the Sleep-Clause over-dedup bug the byte
    /// fuzzer surfaced (the port used to dedup ALL hints, so a double sleep-clause block emitted the
    /// hint once where the sim emits it twice). No-op when disabled OR (only for `once`) when this
    /// exact text was already emitted this battle.
    pub fn hint(&mut self, text: &str, once: bool) {
        if !self.enabled {
            return;
        }
        if self.hints_shown.contains(text) {
            return; // already shown this battle (matches the sim's `this.hints.has(...)` early-return)
        }
        if once {
            self.hints_shown.insert(text.to_string());
        }
        self.emit(|| Line::new(Kw::Hint, vec![text_or_empty(text)]));
    }
    /// `|-nothing` — Splash's "But nothing happened!" marker. Phase 3.
    pub fn nothing(&mut self) {
        self.emit(|| Line::new(Kw::Nothing, vec![]));
    }
    /// `|-fieldactivate|move: <Move>` — a field-wide move activation (Pay Day's coin
    /// scatter). Phase 3, byte-verified vs the capture golden.
    pub fn fieldactivate_move(&mut self, move_name: &str) {
        self.emit(|| Line::new(Kw::Fieldactivate, vec![Field::text(format!("move: {move_name}"))]));
    }

    // ── Fail / side-condition / volatile ───────────────────────────────────────────
    /// `|-fail|<mon>[|<detail>][|[weak]]` — a move/effect that failed. `detail` = e.g.
    /// `move: Substitute` (the Substitute-too-weak fail carries the `[weak]` tag).
    pub fn fail(&mut self, mon: &MonRef, detail: Option<&str>, weak: bool) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon)];
            if let Some(d) = detail {
                f.push(typed_or_text(Kw::Fail, 1, d));
            }
            if weak {
                f.push(Field::tag("[weak]"));
            }
            Line::new(Kw::Fail, f)
        });
    }
    /// `|-sidestart|<side>|<Effect>` — a side condition begins (Spikes). `<side>` is a
    /// side ref (`p<N>: <PlayerName>`, no position letter).
    pub fn sidestart(&mut self, side_ref: &str, effect: &str) {
        self.emit(|| Line::new(Kw::Sidestart, vec![side_ref_field(side_ref), text_or_empty(effect)]));
    }
    /// `|-start|<mon>|<Effect>` — a volatile begins (Substitute up). Named
    /// `volatile_start` to avoid clashing with the framing `|start` line.
    pub fn volatile_start(&mut self, mon: &MonRef, effect: &str) {
        self.emit(|| Line::new(Kw::Start, vec![Field::mon(mon), text_or_empty(effect)]));
    }
    /// `|-start|<mon>|<Effect>|<detail>` — a volatile begins with a detail field (Disable's
    /// disabled move: `|-start|<mon>|Disable|<MoveName>`).
    pub fn volatile_start_detail(&mut self, mon: &MonRef, effect: &str, detail: &str) {
        self.emit(|| Line::new(Kw::Start, vec![Field::mon(mon), text_or_empty(effect), typed_or_text(Kw::Start, 2, detail)]));
    }
    /// `|-start|<mon>|<Effect>|[of] <src>` — a volatile begins with a SOURCE clause
    /// (`gen3_move_coverage_batch3_v1`, the GHOST Curse's `curse.onStart`:
    /// `|-start|<foe>|Curse|[of] <user>`). Observation-only.
    pub fn volatile_start_of(&mut self, mon: &MonRef, effect: &str, of: &MonRef) {
        self.emit(|| Line::new(Kw::Start, vec![Field::mon(mon), text_or_empty(effect), Field::of(of)]));
    }
    /// `|-start|<mon>|<Effect>|[from] ability: <Ability>|[of] <src>` — a volatile begun by
    /// the holder's ABILITY against a contact attacker (Cute Charm's infatuation, `status.rs`).
    pub fn volatile_start_from_ability_of(&mut self, mon: &MonRef, effect: &str, ability: &str, of: &MonRef) {
        self.emit(|| {
            Line::new(Kw::Start, vec![
                Field::mon(mon),
                text_or_empty(effect),
                Field::From(LineCause::Ability(ability.into())),
                Field::of(of),
            ])
        });
    }
    /// `|-start|<mon>|typechange|<Types>[|[from] ability: <Ability>]` — a TYPE change
    /// (Conversion's pick in `status_moves.rs`; Color Change's reactive retype in `status.rs`).
    pub fn typechange(&mut self, mon: &MonRef, types: &str, ability: Option<&str>) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), Field::text("typechange"), text_or_empty(types)];
            if let Some(a) = ability {
                f.push(Field::From(LineCause::Ability(a.into())));
            }
            Line::new(Kw::Start, f)
        });
    }
    /// `|-fail|<mon>|slp|[from] Uproar` (+ a trailing `|[msg]` when the blocked mon IS the
    /// uproarer) — UPROAR's field-wide sleep block (`gen3_uproar_v1`). Probe-captured: the
    /// `[msg]` appears ONLY for the uproarer itself, not for anyone else it protects.
    pub fn fail_slp_from_uproar(&mut self, mon: &MonRef, is_uproarer: bool) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), Field::text("slp"), Field::From(LineCause::Bare("Uproar".into()))];
            if is_uproarer {
                f.push(Field::tag("[msg]"));
            }
            Line::new(Kw::Fail, f)
        });
    }
    /// `|-start|<mon>|<effect>|[upkeep]` — UPROAR's per-residual live tick
    /// (`gen3_uproar_v1`). The sim re-emits `-start` with an `[upkeep]` attr each turn the
    /// lock survives, and switches to `-end` only on the tick that expires it.
    pub fn volatile_start_upkeep(&mut self, mon: &MonRef, effect: &str) {
        self.emit(|| Line::new(Kw::Start, vec![Field::mon(mon), text_or_empty(effect), Field::tag("[upkeep]")]));
    }
    /// `|-end|<mon>|<Effect>` — a volatile ends (Substitute breaks).
    pub fn volatile_end(&mut self, mon: &MonRef, effect: &str) {
        self.emit(|| Line::new(Kw::End, vec![Field::mon(mon), text_or_empty(effect)]));
    }
    /// `|-damage|<mon>|<HP>|[from] move: <Move>|[partiallytrapped]` — the PARTIAL-TRAP
    /// residual chip (`gen3_partial_trap_v1`). The sim renders this in `spreadDamage`'s
    /// `case 'partiallytrapped'` arm off `volatiles.partiallytrapped.sourceEffect.fullname`
    /// (`sim/battle.ts:2140`), so the cause is the MOVE's fullname and a bare
    /// `[partiallytrapped]` tag trails it. Probe-verified for all six carriers
    /// (`harness/probe_ptrap_edges.js` section L).
    pub fn damage_partially_trapped(&mut self, mon: &MonRef, hp: &HpStatus, move_name: &str) {
        self.emit(|| {
            Line::new(Kw::Damage, vec![
                Field::mon(mon),
                Field::hp(hp),
                Field::From(LineCause::Move(move_name.into())),
                Field::tag("[partiallytrapped]"),
            ])
        });
    }
    /// `|-end|<mon>|<Move>|[partiallytrapped][|[silent]]` — the PARTIAL-TRAP release
    /// (`gen3_partial_trap_v1`). NOTE the effect token is the BARE move name (`Wrap`), not
    /// `move: Wrap` — the condition passes the sourceEffect OBJECT, which stringifies to
    /// its `name`. `silent` selects the `onResidual` trapper-gone branch (`[silent]`) over
    /// the `onEnd` natural expiry (no tag).
    pub fn partial_trap_end(&mut self, mon: &MonRef, move_name: &str, silent: bool) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), text_or_empty(move_name), Field::tag("[partiallytrapped]")];
            if silent {
                f.push(Field::tag("[silent]"));
            }
            Line::new(Kw::End, f)
        });
    }
    /// `|-end|<mon>|<Effect>|[silent]` — a volatile ends silently (the Attract
    /// onUpdate removal when its source leaves the field, `gen3_ability_batch4_v1`).
    pub fn volatile_end_silent(&mut self, mon: &MonRef, effect: &str) {
        self.emit(|| Line::new(Kw::End, vec![Field::mon(mon), text_or_empty(effect), Field::tag("[silent]")]));
    }
    /// `|-activate|<user>|Skill Swap|||[of] <target>` — the gen&lt;=4 SKILL SWAP form
    /// (`gen3_skill_swap_v1`). TWO EMPTY FIELDS where gen5+ names the two abilities, and NO
    /// `-endability`/`-ability` lines at all. Probe-settled (`harness/probe_skillswap.js`);
    /// `activate()` cannot express the empty pair, hence its own constructor.
    pub fn skill_swap(&mut self, user: &MonRef, target: &MonRef) {
        self.emit(|| {
            Line::new(Kw::Activate, vec![Field::mon(user), Field::text("Skill Swap"), Field::Empty, Field::Empty, Field::of(target)])
        });
    }
    /// `|-activate|<mon>|<Effect>[|<detail>]` — an effect fires without start/end
    /// (Protect blocking, a Substitute absorbing with `[damage]`).
    pub fn activate(&mut self, mon: &MonRef, effect: &str, detail: Option<&str>) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), text_or_empty(effect)];
            if let Some(d) = detail {
                f.push(typed_or_text(Kw::Activate, 2, d));
            }
            Line::new(Kw::Activate, f)
        });
    }
    /// `|-activate|<mon>|item: <Item>|<MoveName>|[consumed]` — a consumed ITEM activating
    /// against a move (`items.rs`, the Leppa-class restore).
    pub fn activate_item_consumed(&mut self, mon: &MonRef, item: &str, move_name: &str) {
        self.emit(|| {
            Line::new(Kw::Activate, vec![
                Field::mon(mon),
                Field::text(format!("item: {item}")),
                text_or_empty(move_name),
                Field::tag("[consumed]"),
            ])
        });
    }
    /// `|-transform|<user>|<target>` — the TRANSFORM copy landed (`gen3_transform_v1`,
    /// `pokemon.transformInto`'s `this.battle.add('-transform', this, pokemon)`). A plain
    /// Transform passes no `effect`, so gen3 never emits the `|[from] <effect>` variant
    /// (Transform is `failencore` + not Sleep-Talk-reachable in the modeled set). The two
    /// idents are the mons' `toString()` forms — for a TRANSFORMED user that is still its
    /// ORIGINAL name (`pokemon.name` is fixed at construction), which is why
    /// `helpers::display_name` reads through the transform overlay. Draw-free.
    pub fn transform(&mut self, user: &MonRef, target: &MonRef) {
        self.emit(|| Line::new(Kw::Transform, vec![Field::mon(user), Field::mon(target)]));
    }
    /// `|-singleturn|<mon>|<Effect>` — a one-turn effect announced (Protect).
    pub fn singleturn(&mut self, mon: &MonRef, effect: &str) {
        self.emit(|| Line::new(Kw::Singleturn, vec![Field::mon(mon), text_or_empty(effect)]));
    }
    /// `|-singlemove|<mon>|<Effect>` — a until-next-move effect announced
    /// (`gen3_move_coverage_batch6_v1`, Destiny Bond's `onStart`).
    pub fn singlemove(&mut self, mon: &MonRef, effect: &str) {
        self.emit(|| Line::new(Kw::Singlemove, vec![Field::mon(mon), text_or_empty(effect)]));
    }
    /// `|-start|<mon>|<Effect>|[silent]` — a volatile begins silently
    /// (`gen3_move_coverage_batch6_v1`, the Perish Song apply's
    /// `|-start|<mon>|perish3|[silent]`).
    pub fn volatile_start_silent(&mut self, mon: &MonRef, effect: &str) {
        self.emit(|| Line::new(Kw::Start, vec![Field::mon(mon), text_or_empty(effect), Field::tag("[silent]")]));
    }
    /// `|-setboost|<mon>|<stat>|<n>|[from] move: <Move>` — a stat stage SET to an
    /// absolute value (`gen3_move_coverage_batch6_v1`, Belly Drum's `battle.boost`
    /// special-case: `|-setboost|<user>|atk|6|[from] move: Belly Drum`).
    pub fn setboost_from_move(&mut self, mon: &MonRef, stat: &str, n: i8, move_name: &str) {
        self.emit(|| {
            Line::new(Kw::Setboost, vec![
                Field::mon(mon),
                text_or_empty(stat),
                Field::text(n.to_string()),
                Field::From(LineCause::Move(move_name.into())),
            ])
        });
    }
    /// `|-sethp|<mon>|<HP>|[from] move: <Move>[|[silent]]` — an HP SET
    /// (`gen3_move_coverage_batch6_v1`, Pain Split: the TARGET line carries
    /// `[silent]`, the USER line does not — the probed order/form).
    pub fn sethp_from_move(&mut self, mon: &MonRef, hp: &HpStatus, move_name: &str, silent: bool) {
        self.emit(|| {
            let mut f = vec![Field::mon(mon), Field::hp(hp), Field::From(LineCause::Move(move_name.into()))];
            if silent {
                f.push(Field::tag("[silent]"));
            }
            Line::new(Kw::Sethp, f)
        });
    }
    /// `|-copyboost|<user>|<target>|[from] move: <Move>` — the user copies the
    /// target's boost stages (`gen3_move_coverage_batch6_v1`, Psych Up).
    pub fn copyboost_from_move(&mut self, user: &MonRef, target: &MonRef, move_name: &str) {
        self.emit(|| {
            Line::new(Kw::Copyboost, vec![Field::mon(user), Field::mon(target), Field::From(LineCause::Move(move_name.into()))])
        });
    }
    /// `|-hitcount|<mon>|<N>` — the number of times a MULTI-STRIKE move hit
    /// (`gen3_move_coverage_batch4b_v1`, Beat Up: one line at the end of the strike loop,
    /// N = the strikes that actually fired).
    pub fn hitcount(&mut self, mon: &MonRef, n: u32) {
        self.emit(|| Line::new(Kw::Hitcount, vec![Field::mon(mon), Field::text(n.to_string())]));
    }
    /// `|-end|<mon>|<Effect>|[from] move: <Move>|[of] <of>` — a volatile removed BY a move
    /// (`gen3_move_coverage_batch1_v1`, Rapid Spin clearing the USER's own Leech Seed:
    /// `|-end|<user>|Leech Seed|[from] move: Rapid Spin|[of] <user>`).
    pub fn volatile_end_from_move(&mut self, mon: &MonRef, effect: &str, move_name: &str, of: &MonRef) {
        self.emit(|| {
            Line::new(Kw::End, vec![
                Field::mon(mon),
                text_or_empty(effect),
                Field::From(LineCause::Move(move_name.into())),
                Field::of(of),
            ])
        });
    }
    /// `|-sideend|<side>|<Effect>|[from] move: <Move>|[of] <of>` — a side condition removed
    /// BY a move (`gen3_move_coverage_batch1_v1`, Rapid Spin clearing the USER's Spikes).
    pub fn sideend_from_move(&mut self, side_ref: &str, effect: &str, move_name: &str, of: &MonRef) {
        self.emit(|| {
            Line::new(Kw::Sideend, vec![
                side_ref_field(side_ref),
                text_or_empty(effect),
                Field::From(LineCause::Move(move_name.into())),
                Field::of(of),
            ])
        });
    }
    /// `|-enditem|<mon>|<Item>` — an item used up (White Herb's restore, `items.rs` /
    /// `helpers.rs`).
    pub fn enditem(&mut self, mon: &MonRef, item: &str) {
        self.emit(|| Line::new(Kw::Enditem, vec![Field::mon(mon), text_or_empty(item)]));
    }
    /// `|-enditem|<mon>|<Item>|[eat]` — a BERRY eaten (`items.rs`).
    pub fn enditem_eat(&mut self, mon: &MonRef, item: &str) {
        self.emit(|| Line::new(Kw::Enditem, vec![Field::mon(mon), text_or_empty(item), Field::tag("[eat]")]));
    }
    /// `|-enditem|<mon>|<Item>|[from] move: Knock Off|[of] <of>` — the KNOCK OFF item removal
    /// (`gen3_move_coverage_batch1_v1`). In gens 3-4 Knock Off only makes the item unusable;
    /// the paired `|-hint|` line follows.
    pub fn enditem_knockoff(&mut self, mon: &MonRef, item: &str, of: &MonRef) {
        self.emit(|| {
            Line::new(Kw::Enditem, vec![
                Field::mon(mon),
                text_or_empty(item),
                Field::From(LineCause::Move("Knock Off".into())),
                Field::of(of),
            ])
        });
    }
    /// `|-enditem|<mon>|<Item>|[silent]|[from] move: Thief|[of] <of>` — the THIEF target-loses
    /// half (a Covet steal has NO `-enditem`, only the `-item` gain). Silent (the client shows
    /// only the attacker's `-item` gain).
    pub fn enditem_thief_silent(&mut self, mon: &MonRef, item: &str, of: &MonRef) {
        self.emit(|| {
            Line::new(Kw::Enditem, vec![
                Field::mon(mon),
                text_or_empty(item),
                Field::tag("[silent]"),
                Field::From(LineCause::Move("Thief".into())),
                Field::of(of),
            ])
        });
    }
    /// `|-item|<mon>|<Item>|[from] move: <Move>|[of] <of>` — the THIEF / COVET item GAIN on the
    /// attacker (`<Move>` = `Thief`/`Covet`, `<of>` = the mon the item came from).
    pub fn item_stolen(&mut self, mon: &MonRef, item: &str, move_name: &str, of: &MonRef) {
        self.emit(|| {
            Line::new(Kw::Item, vec![
                Field::mon(mon),
                text_or_empty(item),
                Field::From(LineCause::Move(move_name.into())),
                Field::of(of),
            ])
        });
    }
    /// `|-item|<mon>|<Item>|[from] move: <Move>` — an item GAINED via a move, with NO `[of]`
    /// clause (`gen3_trick_v1`, the TRICK swap: `setItem` emits `|-item|<mon>|<name>|[from]
    /// move: Trick` for each side that RECEIVES an item). DISTINCT from `item_stolen` (Thief /
    /// Covet), which carries `[of] <target>`.
    pub fn item_from_move(&mut self, mon: &MonRef, item: &str, move_name: &str) {
        self.emit(|| Line::new(Kw::Item, vec![Field::mon(mon), text_or_empty(item), Field::From(LineCause::Move(move_name.into()))]));
    }
    /// `|-enditem|<mon>|<Item>|[silent]|[from] move: <Move>` — an item SILENTLY lost via a move,
    /// with NO `[of]` clause (`gen3_trick_v1`, the TRICK one-sided swap: the ITEMLESS receiver's
    /// counterpart loses its item via `-enditem [silent]`). DISTINCT from `enditem_thief_silent`
    /// (which carries `[of] <of>`).
    pub fn enditem_silent_from_move(&mut self, mon: &MonRef, item: &str, move_name: &str) {
        self.emit(|| {
            Line::new(Kw::Enditem, vec![
                Field::mon(mon),
                text_or_empty(item),
                Field::tag("[silent]"),
                Field::From(LineCause::Move(move_name.into())),
            ])
        });
    }

    // ── End of battle (player layer — the bridge relays these) ──────────────────
    pub fn win(&mut self, player_name: &str) {
        self.emit(|| Line::new(Kw::Win, vec![text_or_empty(player_name)]));
    }
    pub fn tie(&mut self) {
        self.emit(|| Line::new(Kw::Tie, vec![]));
    }
    /// `|message|<text>` — a plain battle message (the turn-limit tie announcement,
    /// `gen3_turn_limit_tie_v1`). Distinct from the `-message` clause notices.
    pub fn message(&mut self, text: &str) {
        self.emit(|| Line::new(Kw::BareMessage, vec![text_or_empty(text)]));
    }
    /// `|bigerror|<text>` — the turn-limit countdown (`gen3_turn_limit_tie_v1`). poke-env's
    /// `Player` intercepts it (never an event).
    pub fn bigerror(&mut self, text: &str) {
        self.emit(|| Line::new(Kw::Bigerror, vec![text_or_empty(text)]));
    }
}

/// A text field, or [`Field::Empty`] for the empty string (the canonical typing). A `|` inside
/// would be TWO fields on the wire — a caller must pass them as two (`volatile_start_detail`).
fn text_or_empty(s: &str) -> Field {
    debug_assert!(!s.contains('|'), "a single protocol field cannot contain '|': {s:?}");
    if s.is_empty() {
        Field::Empty
    } else {
        Field::Text(s.to_string())
    }
}

/// A free-form token in a TYPED keyword's field (a `-fail` / `-activate` / `-ability` detail),
/// typed exactly as [`Line::parse`] would type it at that position — a detail such as
/// `[damage]` is a tag, `move: Substitute` is text.
fn typed_or_text(kw: Kw, idx: usize, s: &str) -> Field {
    let mut probe = String::from("|");
    probe.push_str(kw.as_str());
    for _ in 0..idx {
        probe.push('|');
    }
    probe.push('|');
    probe.push_str(s);
    Line::parse(&probe).ok().and_then(|l| l.fields.into_iter().nth(idx)).unwrap_or_else(|| text_or_empty(s))
}

/// A pre-rendered identifier (`p1: Jirachi` slot-less, or `p1a: …`), typed.
fn ident_field(s: &str) -> Field {
    match Ident::parse(s) {
        Some(i) if i.render() == s => Field::Ident(i),
        _ => text_or_empty(s),
    }
}

/// A pre-rendered side reference `pN: PlayerName`.
fn side_ref_field(s: &str) -> Field {
    let b = s.as_bytes();
    if b.len() >= 4 && b[0] == b'p' && (b[1] == b'1' || b[1] == b'2') && &s[2..4] == ": " {
        Field::SideRef { side: b[1] - b'1', name: s[4..].to_string() }
    } else {
        text_or_empty(s)
    }
}

fn boost_line(kw: Kw, mon: &MonRef, stat: &str, n: i32) -> Line {
    Line::new(kw, vec![Field::mon(mon), text_or_empty(stat), Field::text(n.to_string())])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hp_status_renders_the_three_variants() {
        assert_eq!(HpStatus { hp: 224, maxhp: 341, status: None }.to_string(), "224/341");
        assert_eq!(HpStatus { hp: 116, maxhp: 524, status: Some("slp") }.to_string(), "116/524 slp");
        // Fainted → `0 fnt` regardless of a lingering status token.
        assert_eq!(HpStatus { hp: 0, maxhp: 341, status: None }.to_string(), "0 fnt");
        assert_eq!(HpStatus { hp: 0, maxhp: 341, status: Some("par") }.to_string(), "0 fnt");
    }

    #[test]
    fn mon_ref_and_side_ref_split() {
        assert_eq!(MonRef { side: 0, name: "Tyranitar".into() }.to_string(), "p1a: Tyranitar");
        assert_eq!(MonRef { side: 1, name: "Snorlax".into() }.to_string(), "p2a: Snorlax");
        assert_eq!(ProtocolBuilder::side_ref(1, "P2"), "p2: P2");
    }

    #[test]
    fn cause_renders_item_and_bare() {
        assert_eq!(Cause::Item("Leftovers".into()).to_string(), "[from] item: Leftovers");
        assert_eq!(Cause::Bare("Sandstorm".into()).to_string(), "[from] Sandstorm");
        assert_eq!(Cause::Bare("psn".into()).to_string(), "[from] psn");
    }

    #[test]
    fn disabled_builder_emits_nothing() {
        let mut b = ProtocolBuilder::new();
        b.turn(1);
        b.separator();
        assert!(b.lines().is_empty(), "disabled builder must push nothing");
        b.enable();
        b.turn(2);
        assert_eq!(b.lines().len(), 1);
        assert_eq!(b.lines()[0].0, "|turn|2");
    }

    #[test]
    fn cause_renders_ability_and_move() {
        assert_eq!(Cause::Ability("Sand Stream".into()).to_string(), "[from] ability: Sand Stream");
        assert_eq!(Cause::Move("Rest".into()).to_string(), "[from] move: Rest");
    }

    #[test]
    fn status_from_variants() {
        let mut b = ProtocolBuilder::new();
        b.enable();
        let mon = MonRef { side: 1, name: "Starmie".into() };
        // foe-inflicted: no [from].
        b.status(&mon, "par", None);
        // Rest self-inflict: [from] move: Rest.
        let self_mon = MonRef { side: 0, name: "Snorlax".into() };
        b.status(&self_mon, "slp", Some(&Cause::Move("Rest".into())));
        assert_eq!(b.lines()[0].0, "|-status|p2a: Starmie|par");
        assert_eq!(b.lines()[1].0, "|-status|p1a: Snorlax|slp|[from] move: Rest");
    }

    #[test]
    fn curestatus_and_cant_forms() {
        let mut b = ProtocolBuilder::new();
        b.enable();
        let mon = MonRef { side: 0, name: "Snorlax".into() };
        // natural wake: [msg].
        b.curestatus(&mon, "slp", true);
        // no-msg cure.
        b.curestatus(&mon, "frz", false);
        // cant: par / slp / flinch, no move name.
        b.cant(&mon, "slp", None);
        b.cant(&mon, "flinch", None);
        assert_eq!(b.lines()[0].0, "|-curestatus|p1a: Snorlax|slp|[msg]");
        assert_eq!(b.lines()[1].0, "|-curestatus|p1a: Snorlax|frz");
        assert_eq!(b.lines()[2].0, "|cant|p1a: Snorlax|slp");
        assert_eq!(b.lines()[3].0, "|cant|p1a: Snorlax|flinch");
    }

    #[test]
    fn boost_by_sign_and_stat_token() {
        let mut b = ProtocolBuilder::new();
        b.enable();
        let mon = MonRef { side: 1, name: "Metagross".into() };
        // +1 atk → -boost; -1 spd → -unboost.
        b.boost(&mon, 0, 1); // atk +1
        b.boost(&mon, 3, -1); // spd -1
        b.boost(&mon, 4, 2); // spe +2
        // A zero delta emits nothing.
        b.boost(&mon, 1, 0);
        assert_eq!(b.lines().len(), 3);
        assert_eq!(b.lines()[0].0, "|-boost|p2a: Metagross|atk|1");
        assert_eq!(b.lines()[1].0, "|-unboost|p2a: Metagross|spd|1");
        assert_eq!(b.lines()[2].0, "|-boost|p2a: Metagross|spe|2");
    }

    #[test]
    fn weather_set_and_upkeep_forms() {
        let mut b = ProtocolBuilder::new();
        b.enable();
        let src = MonRef { side: 1, name: "Tyranitar".into() };
        // SET form: [from] ability: + [of].
        b.weather("Sandstorm", Some(&Cause::Ability("Sand Stream".into())), Some(&src), false);
        // upkeep tick.
        b.weather("Sandstorm", None, None, true);
        assert_eq!(
            b.lines()[0].0,
            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Tyranitar"
        );
        assert_eq!(b.lines()[1].0, "|-weather|Sandstorm|[upkeep]");
    }

    #[test]
    fn fail_weak_and_sidestart_start_end_activate_forms() {
        let mut b = ProtocolBuilder::new();
        b.enable();
        let mon = MonRef { side: 0, name: "Suicune".into() };
        // Substitute too-weak fail: detail + [weak].
        b.fail(&mon, Some("move: Substitute"), true);
        // Substitute up / break.
        b.volatile_start(&mon, "Substitute");
        b.volatile_end(&mon, "Substitute");
        // Substitute absorbs damage.
        b.activate(&mon, "Substitute", Some("[damage]"));
        // Spikes side-start.
        b.sidestart(&ProtocolBuilder::side_ref(1, "P2"), "Spikes");
        // Protect singleturn + block.
        let skarm = MonRef { side: 0, name: "Skarmory".into() };
        b.singleturn(&skarm, "Protect");
        b.activate(&skarm, "Protect", None);
        assert_eq!(b.lines()[0].0, "|-fail|p1a: Suicune|move: Substitute|[weak]");
        assert_eq!(b.lines()[1].0, "|-start|p1a: Suicune|Substitute");
        assert_eq!(b.lines()[2].0, "|-end|p1a: Suicune|Substitute");
        assert_eq!(b.lines()[3].0, "|-activate|p1a: Suicune|Substitute|[damage]");
        assert_eq!(b.lines()[4].0, "|-sidestart|p2: P2|Spikes");
        assert_eq!(b.lines()[5].0, "|-singleturn|p1a: Skarmory|Protect");
        assert_eq!(b.lines()[6].0, "|-activate|p1a: Skarmory|Protect");
    }

    #[test]
    fn attr_last_move_still_rewrites_the_most_recent_move_line() {
        // The `attrLastMove('[still]')` retro-edit (`gen3_disable_zero_pp_v1`): the
        // TARGET field of the most recent `|move|` line is BLANKED and `|[still]`
        // appended — the exact battle.js transform (parts[4] = '' then append). Lines
        // pushed after the announce (none in the real disable path, but be robust)
        // are untouched; earlier `|move|` lines are untouched.
        let mut b = ProtocolBuilder::new();
        b.enable();
        let user = MonRef { side: 0, name: "Suicune".into() };
        let target = MonRef { side: 1, name: "Blissey".into() };
        b.move_used(&target, "Struggle", Some(&user), false, false); // an EARLIER move line
        b.move_used(&user, "Disable", Some(&target), false, false);
        b.attr_last_move_still();
        b.fail(&user, None, false);
        assert_eq!(b.lines()[0].0, "|move|p2a: Blissey|Struggle|p1a: Suicune");
        assert_eq!(
            b.lines()[1].0,
            "|move|p1a: Suicune|Disable||[still]",
            "the retro-edit blanks the target + appends [still] (probe: the sim's \
             0-PP-guard fail renders exactly this form)"
        );
        assert_eq!(b.lines()[2].0, "|-fail|p1a: Suicune");
        // Disabled builder: a no-op (no lines, no panic).
        let mut off = ProtocolBuilder::new();
        off.attr_last_move_still();
        assert_eq!(off.lines().len(), 0);
    }

    #[test]
    fn rest_silent_heal_line_with_status_hp() {
        // The Rest full-heal line: `|-heal|<user>|<HP> slp|[silent]` — the HP field
        // carries the new `slp` token (via `HpStatus`) and the trailing `[silent]` tag.
        // (`run_rest` builds this via the typed `heal_silent`; pin the exact bytes here so a
        // format drift is caught.)
        let mut b = ProtocolBuilder::new();
        b.enable();
        let user = MonRef { side: 0, name: "Snorlax".into() };
        let hp = HpStatus { hp: 524, maxhp: 524, status: Some("slp") };
        b.heal_silent(&user, &hp);
        assert_eq!(b.lines()[0].0, "|-heal|p1a: Snorlax|524/524 slp|[silent]");
    }

    #[test]
    fn ability_intimidate_boost_detail() {
        let mut b = ProtocolBuilder::new();
        b.enable();
        let mon = MonRef { side: 1, name: "Salamence".into() };
        b.ability(&mon, "Intimidate", Some("boost"));
        assert_eq!(b.lines()[0].0, "|-ability|p2a: Salamence|Intimidate|boost");
    }

    #[test]
    fn move_still_and_miss_forms() {
        let mut b = ProtocolBuilder::new();
        b.enable();
        let user = MonRef { side: 0, name: "Skarmory".into() };
        let target = MonRef { side: 1, name: "Tyranitar".into() };
        // still: empty target field + [still].
        b.move_used(&user, "Protect", Some(&user), false, true);
        // miss: target + [miss].
        b.move_used(&target, "Rock Slide", Some(&user), true, false);
        // plain: target, no tags.
        b.move_used(&user, "Drill Peck", Some(&target), false, false);
        assert_eq!(b.lines()[0].0, "|move|p1a: Skarmory|Protect||[still]");
        assert_eq!(b.lines()[1].0, "|move|p2a: Tyranitar|Rock Slide|p1a: Skarmory|[miss]");
        assert_eq!(b.lines()[2].0, "|move|p1a: Skarmory|Drill Peck|p2a: Tyranitar");
    }

    #[test]
    fn phase3_forms_and_the_miss_retro_edit() {
        // The Phase-3 constructors (`gen3_protocol_phase3_v1`), byte-pinned to the
        // capture-golden forms.
        let mut b = ProtocolBuilder::new();
        b.enable();
        let user = MonRef { side: 0, name: "Gengar".into() };
        let target = MonRef { side: 1, name: "Snorlax".into() };
        // attr_last_move_miss: append |[miss] WITHOUT blanking the target.
        b.move_used(&user, "Hypnosis", Some(&target), false, false);
        b.attr_last_move_miss();
        assert_eq!(b.lines()[0].0, "|move|p1a: Gengar|Hypnosis|p2a: Snorlax|[miss]");
        // ability_silent (Pressure), ability_traced (Trace), fail_unboost (Clear Body),
        // hint, nothing, fieldactivate.
        b.ability_silent(&user, "Pressure");
        b.ability_traced(&user, "Thick Fat", &target);
        b.fail_unboost_from_ability(&target, "Clear Body", None);
        b.hint("In Gen 3, Intimidate does not activate if every target has a Substitute.", false);
        b.nothing();
        b.fieldactivate_move("Pay Day");
        assert_eq!(b.lines()[1].0, "|-ability|p1a: Gengar|Pressure|[silent]");
        assert_eq!(
            b.lines()[2].0,
            "|-ability|p1a: Gengar|Thick Fat|Trace|[from] ability: Trace|[of] p2a: Snorlax"
        );
        assert_eq!(
            b.lines()[3].0,
            "|-fail|p2a: Snorlax|unboost|[from] ability: Clear Body|[of] p2a: Snorlax"
        );
        assert_eq!(
            b.lines()[4].0,
            "|-hint|In Gen 3, Intimidate does not activate if every target has a Substitute."
        );
        assert_eq!(b.lines()[5].0, "|-nothing");
        assert_eq!(b.lines()[6].0, "|-fieldactivate|move: Pay Day");
        // The taunt/disable residual -end forms + the heal-fail detail.
        b.volatile_end_silent(&target, "move: Taunt");
        b.volatile_end(&target, "Disable");
        b.fail(&user, Some("heal"), false);
        assert_eq!(b.lines()[7].0, "|-end|p2a: Snorlax|move: Taunt|[silent]");
        assert_eq!(b.lines()[8].0, "|-end|p2a: Snorlax|Disable");
        assert_eq!(b.lines()[9].0, "|-fail|p1a: Gengar|heal");
        // Hyper Cutter carries the `unboost|Attack` stat token (the SIM-PROBED single-stat
        // form); Keen Eye carries `accuracy`; the whole-table blockers carry no token.
        b.fail_unboost_from_ability(&user, "Hyper Cutter", Some("Attack"));
        b.fail_unboost_from_ability(&user, "Keen Eye", Some("accuracy"));
        assert_eq!(
            b.lines()[10].0,
            "|-fail|p1a: Gengar|unboost|Attack|[from] ability: Hyper Cutter|[of] p1a: Gengar"
        );
        assert_eq!(
            b.lines()[11].0,
            "|-fail|p1a: Gengar|unboost|accuracy|[from] ability: Keen Eye|[of] p1a: Gengar"
        );
    }

    /// The `-hint` `once` DEDUP semantics (`gen3_omniscient_byte_fuzz_v1`, the byte-fuzzer-surfaced
    /// Sleep-Clause over-dedup): the sim's `Battle.hint(hint, once, side)` returns early if the text
    /// is already in `this.hints` but only ADDS it `if (once)`. So a `once: false` hint (Sleep Clause
    /// Mod / the gen3 Intimidate-vs-Sub / Pursuit notes) fires on EVERY call, while a `once: true`
    /// hint (Knock Off) fires exactly ONCE per battle. WRONG (pre-fix): the port deduped ALL hints, so
    /// a gen3ou double-sleep-clause-block emitted the hint ONCE where the sim emits it twice. Reverting
    /// the `once` gate (dedup unconditionally) fails this pin.
    #[test]
    fn hint_dedup_follows_the_once_param() {
        let sc = "Sleep Clause Mod prevents players from putting more than one of their opponent's \
                  Pok\u{e9}mon to sleep at a time";
        let ko = "In Gens 3-4, Knock Off only makes the target's item unusable; it cannot obtain a new item.";

        // once=false (Sleep Clause) fires on EVERY call — a double block emits the hint TWICE.
        let mut b = ProtocolBuilder::new();
        b.enable();
        b.hint(sc, false);
        b.hint(sc, false);
        let sc_count = b.lines().iter().filter(|l| l.0 == format!("|-hint|{sc}")).count();
        assert_eq!(sc_count, 2, "a `once:false` hint must fire on EVERY call (Sleep Clause block)");

        // once=true (Knock Off) fires exactly ONCE per battle.
        let mut b = ProtocolBuilder::new();
        b.enable();
        b.hint(ko, true);
        b.hint(ko, true);
        let ko_count = b.lines().iter().filter(|l| l.0 == format!("|-hint|{ko}")).count();
        assert_eq!(ko_count, 1, "a `once:true` hint must fire exactly ONCE per battle (Knock Off)");
    }
}
