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
#[derive(Debug, Clone, Default)]
pub struct ProtocolBuilder {
    lines: Vec<ProtocolLine>,
    /// Whether emission is enabled. Off by default so the seed suite (which never
    /// drains the log) pays zero formatting cost AND so the observation-only proof
    /// is unmistakable: with `enabled == false` NOTHING is pushed. `run_full_battle`
    /// leaves it off; `run_full_battle_logged` turns it on. Either way NO PRNG is
    /// touched (the whole module is draw-free).
    enabled: bool,
}

impl ProtocolBuilder {
    pub fn new() -> Self {
        Self { lines: Vec::new(), enabled: false }
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

    /// The accumulated lines so far (read-only view).
    pub fn lines(&self) -> &[ProtocolLine] {
        &self.lines
    }

    /// Drain everything emitted so far (the per-decision / per-battle batch the
    /// bridge relays). Leaves the buffer empty.
    pub fn drain(&mut self) -> Vec<ProtocolLine> {
        std::mem::take(&mut self.lines)
    }

    /// Push a raw, already-formatted line (the escape hatch; prefer the typed
    /// helpers). No-op when emission is disabled.
    pub fn push_raw(&mut self, line: impl Into<String>) {
        if self.enabled {
            self.lines.push(ProtocolLine(line.into()));
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
        self.push_raw("|t:|<NORMALIZED>");
    }
    pub fn gametype_singles(&mut self) {
        self.push_raw("|gametype|singles");
    }
    /// `|player|p<N>|<name>|<avatar>|<rating>` — avatar+rating empty (two trailing
    /// pipes), matching the capture (`|player|p1|P1||`).
    pub fn player(&mut self, side: usize, name: &str) {
        self.push_raw(format!("|player|{}|{}||", Player::from_side(side).tag(), name));
    }
    pub fn gen(&mut self, gen: u8) {
        self.push_raw(format!("|gen|{gen}"));
    }
    pub fn tier(&mut self, display: &str) {
        self.push_raw(format!("|tier|{display}"));
    }
    pub fn rule(&mut self, text: &str) {
        self.push_raw(format!("|rule|{text}"));
    }
    pub fn teamsize(&mut self, side: usize, count: usize) {
        self.push_raw(format!("|teamsize|{}|{}", Player::from_side(side).tag(), count));
    }
    pub fn start(&mut self) {
        self.push_raw("|start");
    }

    // ── Turn / phase markers ────────────────────────────────────────────────────
    /// The blank `|` separator line (emitted between phases — poke-env parses it as
    /// a no-op). The port MUST emit these for byte-equality.
    pub fn separator(&mut self) {
        self.push_raw("|");
    }
    pub fn turn(&mut self, n: u32) {
        self.push_raw(format!("|turn|{n}"));
    }
    pub fn upkeep(&mut self) {
        self.push_raw("|upkeep");
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
        let mut s = format!("|move|{user}|{move_name}|");
        if still {
            // Empty target field + [still] (e.g. Protect: `|move|…|Protect||[still]`).
            s.push_str("|[still]");
        } else if let Some(t) = target {
            s.push_str(&t.to_string());
            if miss {
                s.push_str("|[miss]");
            }
        }
        self.push_raw(s);
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
        if let Some(line) = self.lines.iter_mut().rev().find(|l| l.0.starts_with("|move|")) {
            let mut parts: Vec<&str> = line.0.split('|').collect();
            // parts: ["", "move", "<user>", "<MoveName>", "<target>", ...attrs]
            if parts.len() >= 5 {
                parts[4] = "";
            }
            let mut s = parts.join("|");
            s.push_str("|[still]");
            line.0 = s;
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
        if let Some(line) = self.lines.iter_mut().rev().find(|l| l.0.starts_with("|move|")) {
            line.0.push_str("|[miss]");
        }
    }

    // ── Switch / drag / faint ───────────────────────────────────────────────────
    /// `|switch|<mon>|<Details>|<HP>`. Details = `Species` (L100/genderless omitted,
    /// matching the capture — a gen-3-singles L100 genderless mon shows just the
    /// species name).
    pub fn switch(&mut self, mon: &MonRef, details: &str, hp: &HpStatus) {
        self.push_raw(format!("|switch|{mon}|{details}|{hp}"));
    }
    /// `|drag|<mon>|<Details>|<HP>` — identical grammar to `switch`; the FORCED
    /// (Roar/Whirlwind) entry.
    pub fn drag(&mut self, mon: &MonRef, details: &str, hp: &HpStatus) {
        self.push_raw(format!("|drag|{mon}|{details}|{hp}"));
    }
    pub fn faint(&mut self, mon: &MonRef) {
        self.push_raw(format!("|faint|{mon}"));
    }

    // ── Damage / heal ───────────────────────────────────────────────────────────
    /// `|-damage|<mon>|<HP>[|[from] <cause>]`. The residual forms carry a `[from]`
    /// cause (`Sandstorm`/`psn`/`brn`/`tox`/`Spikes`/`Leech Seed`).
    pub fn damage(&mut self, mon: &MonRef, hp: &HpStatus, from: Option<&Cause>) {
        match from {
            Some(c) => self.push_raw(format!("|-damage|{mon}|{hp}|{c}")),
            None => self.push_raw(format!("|-damage|{mon}|{hp}")),
        }
    }
    /// `|-damage|<mon>|<HP>|[from] <cause>|[of] <src>` — a damage line that carries BOTH
    /// a `[from]` cause AND an `[of]` source. The gen-3 **Struggle recoil**
    /// (`gen3_pp_tracking_v1`) emits `|-damage|<user>|<HP>|[from] Recoil|[of] <target>`
    /// (the target of the Struggle is the `[of]` source), matching the golden exactly.
    pub fn damage_of(&mut self, mon: &MonRef, hp: &HpStatus, from: &Cause, of: &MonRef) {
        self.push_raw(format!("|-damage|{mon}|{hp}|{from}|[of] {of}"));
    }
    /// `|-heal|<mon>|<HP>[|[from] <cause>]` (Leftovers carries `[from] item:
    /// Leftovers`; a self-heal move / Leech-Seed heal has no `[from]` here in the
    /// Phase-1 set — recovery moves are still deferred).
    pub fn heal(&mut self, mon: &MonRef, hp: &HpStatus, from: Option<&Cause>) {
        match from {
            Some(c) => self.push_raw(format!("|-heal|{mon}|{hp}|{c}")),
            None => self.push_raw(format!("|-heal|{mon}|{hp}")),
        }
    }

    // ── Effectiveness / crit / miss / immune ────────────────────────────────────
    /// `|-supereffective|<mon>` — `<mon>` is the DEFENDER.
    pub fn supereffective(&mut self, mon: &MonRef) {
        self.push_raw(format!("|-supereffective|{mon}"));
    }
    /// `|-resisted|<mon>` — the DEFENDER.
    pub fn resisted(&mut self, mon: &MonRef) {
        self.push_raw(format!("|-resisted|{mon}"));
    }
    /// `|-crit|<mon>` — a critical hit on the DEFENDER.
    pub fn crit(&mut self, mon: &MonRef) {
        self.push_raw(format!("|-crit|{mon}"));
    }
    /// `|-immune|<mon>` — effectiveness 0. (The ability form carrying `[from]
    /// ability:` is a later phase — the Phase-1 capture immune lines are bare.)
    pub fn immune(&mut self, mon: &MonRef) {
        self.push_raw(format!("|-immune|{mon}"));
    }
    /// `|-immune|<mon>|[from] ability: <Ability>` — immunity granted by an ABILITY
    /// (`gen3_ability_batch2_v1`, Soundproof blocking a sound move). Observation-only.
    pub fn immune_from_ability(&mut self, mon: &MonRef, ability: &str) {
        self.push_raw(format!("|-immune|{mon}|[from] ability: {ability}"));
    }
    /// `|-miss|<user>[|<target>]` — paired with the `|move|…|[miss]`.
    pub fn miss(&mut self, user: &MonRef, target: Option<&MonRef>) {
        match target {
            Some(t) => self.push_raw(format!("|-miss|{user}|{t}")),
            None => self.push_raw(format!("|-miss|{user}")),
        }
    }

    // ── Status ───────────────────────────────────────────────────────────────────
    /// `|-status|<mon>|<status>[|[from] <cause>]` — a major status is SET. `status` ∈
    /// `brn`/`par`/`slp`/`frz`/`psn`/`tox`. A self-inflicted status (Rest) carries
    /// `[from] move: Rest`; a foe-inflicted status has no `[from]`.
    pub fn status(&mut self, mon: &MonRef, status: &str, from: Option<&Cause>) {
        match from {
            Some(c) => self.push_raw(format!("|-status|{mon}|{status}|{c}")),
            None => self.push_raw(format!("|-status|{mon}|{status}")),
        }
    }
    /// `|-status|<mon>|<status>|[from] ability: <Ability>|[of] <src>` — a status inflicted by
    /// a reactive ABILITY on another mon (`gen3_ability_batch2_v1`): the CONTACT_PROC procs
    /// (Static/Poison Point/Flame Body/Effect Spore) statusing the ATTACKER, and Synchronize
    /// reflecting a status back to the SOURCE. `src` is the ability HOLDER (the mon whose
    /// ability caused it). Observation-only.
    pub fn status_from_ability(&mut self, mon: &MonRef, status: &str, ability: &str, src: &MonRef) {
        self.push_raw(format!("|-status|{mon}|{status}|[from] ability: {ability}|[of] {src}"));
    }
    /// `|-curestatus|<mon>|<status>[|[msg]]` — a status is CURED (wake / thaw / Rest
    /// natural wake). `[msg]` shows the client message (a natural sleep/freeze wake).
    pub fn curestatus(&mut self, mon: &MonRef, status: &str, msg: bool) {
        if msg {
            self.push_raw(format!("|-curestatus|{mon}|{status}|[msg]"));
        } else {
            self.push_raw(format!("|-curestatus|{mon}|{status}"));
        }
    }
    /// `|-curestatus|<mon>|<status>|[from] move: <Move>` — a status cured BY a move
    /// (`gen3_defrost_v1`: the frozen user of a `flags.defrost` move — Sacred Fire /
    /// Flame Wheel — thaws draw-free via `frz.onModifyMove` after a FAILED 1/5 thaw
    /// roll; the line is emitted BEFORE the user's own `|move|` line. Probe:
    /// `harness/probe_sacredfire_defrost.js`).
    pub fn curestatus_from_move(&mut self, mon: &MonRef, status: &str, move_name: &str) {
        self.push_raw(format!("|-curestatus|{mon}|{status}|[from] move: {move_name}"));
    }
    /// `|cant|<mon>|<reason>[|<MoveName>]` — a blocked action (full-para / asleep /
    /// frozen / flinch). `reason` ∈ `par`/`slp`/`frz`/`flinch`/…. gen-3 emits no move
    /// name for the status/flinch cases in the capture.
    /// `|cant|<mon>|<reason>|<MoveName>|[of] <of>` — a blocked action attributed to another
    /// mon's ability (`gen3_ability_batch2_v1`, Damp cancelling Explosion): the CANT is on the
    /// ABILITY HOLDER (`mon`), the reason is `ability: Damp`, the move name, and `[of]` the
    /// move's user. Mirrors `this.add("cant", damp_holder, "ability: Damp", move, [of] user)`.
    pub fn cant_of_move(&mut self, mon: &MonRef, reason: &str, move_name: &str, of: &MonRef) {
        self.push_raw(format!("|cant|{mon}|{reason}|{move_name}|[of] {of}"));
    }
    pub fn cant(&mut self, mon: &MonRef, reason: &str, move_name: Option<&str>) {
        match move_name {
            Some(m) => self.push_raw(format!("|cant|{mon}|{reason}|{m}")),
            None => self.push_raw(format!("|cant|{mon}|{reason}")),
        }
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
        let stat = STAT_TOKENS[stat_idx];
        let mag = stages.unsigned_abs();
        if stages > 0 {
            self.push_raw(format!("|-boost|{mon}|{stat}|{mag}"));
        } else {
            self.push_raw(format!("|-unboost|{mon}|{stat}|{mag}"));
        }
    }

    /// `|-unboost|<mon>|atk|<mag>` — the Intimidate Atk-drop line, emitted UNCONDITIONALLY
    /// (INCLUDING `mag == 0`). Unlike [`boost`], this does NOT skip a zero delta: a REQUESTED
    /// stat drop clamped to 0 at the −6 floor STILL reports `|-unboost|<foe>|atk|0` in the sim
    /// (probe `harness/probe_intimidate_floor.js`), whereas [`boost`]'s zero-skip is for a
    /// genuine no-op boost. `mag` is the UNSIGNED applied magnitude (`0` or `1`).
    pub fn unboost_atk_applied(&mut self, mon: &MonRef, mag: u8) {
        let stat = STAT_TOKENS[0]; // atk
        self.push_raw(format!("|-unboost|{mon}|{stat}|{mag}"));
    }

    // ── Weather / ability ────────────────────────────────────────────────────────
    /// `|-weather|<Weather>[|[from] ability: <A>|[of] <src>][|[upkeep]]`. The SET form
    /// (an ability turns the weather on) carries `[from] ability:` + `[of] <src>`; the
    /// per-turn TICK is `|-weather|<Weather>|[upkeep]`.
    pub fn weather(&mut self, weather: &str, from: Option<&Cause>, of: Option<&MonRef>, upkeep: bool) {
        let mut s = format!("|-weather|{weather}");
        if let Some(c) = from {
            s.push('|');
            s.push_str(&c.to_string());
        }
        if let Some(src) = of {
            s.push_str(&format!("|[of] {src}"));
        }
        if upkeep {
            s.push_str("|[upkeep]");
        }
        self.push_raw(s);
    }
    /// `|-ability|<mon>|<Ability>[|<detail>]` — an ability reveals/fires. The Intimidate
    /// switch-in line carries the `boost` detail (`|-ability|p2a: Salamence|Intimidate|
    /// boost`).
    pub fn ability(&mut self, mon: &MonRef, ability: &str, detail: Option<&str>) {
        match detail {
            Some(d) => self.push_raw(format!("|-ability|{mon}|{ability}|{d}")),
            None => self.push_raw(format!("|-ability|{mon}|{ability}")),
        }
    }
    /// `|-ability|<mon>|<Ability>|[silent]` — the switch-in ability reveal Showdown
    /// adds via `addSplit` (Pressure; the omniscient stream carries the secret line).
    /// Phase 3 (`gen3_protocol_phase3_v1`), byte-verified vs the capture golden.
    pub fn ability_silent(&mut self, mon: &MonRef, ability: &str) {
        self.push_raw(format!("|-ability|{mon}|{ability}|[silent]"));
    }
    /// `|-ability|<mon>|<Copied>|Trace|[from] ability: Trace|[of] <foe>` — the Trace
    /// copy announce at switch-in (`setAbility`'s reveal: the third field is the OLD
    /// ability, always `Trace` here). Phase 3, byte-verified vs the capture golden.
    pub fn ability_traced(&mut self, mon: &MonRef, copied: &str, foe: &MonRef) {
        self.push_raw(format!("|-ability|{mon}|{copied}|Trace|[from] ability: Trace|[of] {foe}"));
    }
    /// `|-fail|<mon>|unboost|[from] ability: <Blocker>|[of] <mon>` — a stat-drop
    /// blocked by the target's own ability (Intimidate into Clear Body / White Smoke /
    /// Hyper Cutter). The `[of]` is the BLOCKER itself (its own ability saved it).
    /// Phase 3, byte-verified vs the capture golden.
    pub fn fail_unboost_from_ability(&mut self, mon: &MonRef, ability: &str) {
        self.push_raw(format!("|-fail|{mon}|unboost|[from] ability: {ability}|[of] {mon}"));
    }
    /// `|-hint|<text>` — a client hint (poke-env-ignored, but part of the omniscient
    /// byte stream — e.g. the gen3 Intimidate-vs-Substitute no-op). Phase 3.
    pub fn hint(&mut self, text: &str) {
        self.push_raw(format!("|-hint|{text}"));
    }
    /// `|-nothing` — Splash's "But nothing happened!" marker. Phase 3.
    pub fn nothing(&mut self) {
        self.push_raw("|-nothing");
    }
    /// `|-fieldactivate|move: <Move>` — a field-wide move activation (Pay Day's coin
    /// scatter). Phase 3, byte-verified vs the capture golden.
    pub fn fieldactivate_move(&mut self, move_name: &str) {
        self.push_raw(format!("|-fieldactivate|move: {move_name}"));
    }

    // ── Fail / side-condition / volatile ───────────────────────────────────────────
    /// `|-fail|<mon>[|<detail>][|[weak]]` — a move/effect that failed. `detail` = e.g.
    /// `move: Substitute` (the Substitute-too-weak fail carries the `[weak]` tag).
    pub fn fail(&mut self, mon: &MonRef, detail: Option<&str>, weak: bool) {
        let mut s = format!("|-fail|{mon}");
        if let Some(d) = detail {
            s.push('|');
            s.push_str(d);
        }
        if weak {
            s.push_str("|[weak]");
        }
        self.push_raw(s);
    }
    /// `|-sidestart|<side>|<Effect>` — a side condition begins (Spikes). `<side>` is a
    /// side ref (`p<N>: <PlayerName>`, no position letter).
    pub fn sidestart(&mut self, side_ref: &str, effect: &str) {
        self.push_raw(format!("|-sidestart|{side_ref}|{effect}"));
    }
    /// `|-start|<mon>|<Effect>` — a volatile begins (Substitute up). Named
    /// `volatile_start` to avoid clashing with the framing `|start` line.
    pub fn volatile_start(&mut self, mon: &MonRef, effect: &str) {
        self.push_raw(format!("|-start|{mon}|{effect}"));
    }
    /// `|-end|<mon>|<Effect>` — a volatile ends (Substitute breaks).
    pub fn volatile_end(&mut self, mon: &MonRef, effect: &str) {
        self.push_raw(format!("|-end|{mon}|{effect}"));
    }
    /// `|-end|<mon>|<Effect>|[silent]` — a volatile ends silently (the Attract
    /// onUpdate removal when its source leaves the field, `gen3_ability_batch4_v1`).
    pub fn volatile_end_silent(&mut self, mon: &MonRef, effect: &str) {
        self.push_raw(format!("|-end|{mon}|{effect}|[silent]"));
    }
    /// `|-activate|<mon>|<Effect>[|<detail>]` — an effect fires without start/end
    /// (Protect blocking, a Substitute absorbing with `[damage]`).
    pub fn activate(&mut self, mon: &MonRef, effect: &str, detail: Option<&str>) {
        match detail {
            Some(d) => self.push_raw(format!("|-activate|{mon}|{effect}|{d}")),
            None => self.push_raw(format!("|-activate|{mon}|{effect}")),
        }
    }
    /// `|-singleturn|<mon>|<Effect>` — a one-turn effect announced (Protect).
    pub fn singleturn(&mut self, mon: &MonRef, effect: &str) {
        self.push_raw(format!("|-singleturn|{mon}|{effect}"));
    }

    // ── End of battle (player layer — the bridge relays these) ──────────────────
    pub fn win(&mut self, player_name: &str) {
        self.push_raw(format!("|win|{player_name}"));
    }
    pub fn tie(&mut self) {
        self.push_raw("|tie");
    }
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
        // (`run_rest` builds this via `push_raw` since the `[silent]` tag has no `[from]`;
        // pin the exact bytes here so a format drift is caught.)
        let mut b = ProtocolBuilder::new();
        b.enable();
        let user = MonRef { side: 0, name: "Snorlax".into() };
        let hp = HpStatus { hp: 524, maxhp: 524, status: Some("slp") };
        b.push_raw(format!("|-heal|{user}|{hp}|[silent]"));
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
        b.fail_unboost_from_ability(&target, "Clear Body");
        b.hint("In Gen 3, Intimidate does not activate if every target has a Substitute.");
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
    }
}
