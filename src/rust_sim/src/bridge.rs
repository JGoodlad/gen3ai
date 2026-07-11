//! Bridge surface — the PER-SIDE (`p1` / `p2`) protocol streams a drop-in
//! replacement for `src/utils/bridge/local_sim_bridge.js` must emit.
//!
//! The omniscient [`crate::protocol::ProtocolBuilder`] stream is the SECRET
//! (channel `-1`) log the crate already produces bit-for-bit. Showdown's
//! `getPlayerStreams` (`deps/pokemon-showdown/dist/sim/battle-stream.js`) SPLITS
//! that log into per-side streams via the `|split|pN` markers + `extractChannelMessages`
//! (`battle.js`) and rides each side's `|request|{...}` JSON on a `|sideupdate|`
//! frame. This module reproduces BOTH gaps ON TOP of the unchanged omniscient
//! stream — it is purely ADDITIVE, like `BattleStream::write_line`:
//!
//! - **G2 — the per-side split + HP-privacy fold** (`derive_side`): the omniscient
//!   log line is folded to each side's view. The rule (derived from
//!   `battle.js::extractChannelMessages` + `pokemon.js::getHealth` +
//!   `mods/gen3/abilities.js`):
//!     * An HP-bearing line (`|switch|`/`|drag|`/`|-damage|`/`|-heal|`/`|-sethp|`)
//!       shows the OWNING side EXACT HP (`463/463`) and the OTHER side a PERCENT
//!       (`ceil(hp*100/maxhp)/100`, clamped so `100` with `hp<maxhp` → `99`;
//!       `0 fnt` unchanged). The owner is the side of the `pNa:` ident.
//!     * An OWNER-ONLY line (empty `shared`, dropped for the other side): gen3's
//!       Pressure switch-in reveal `|-ability|pNa: X|Pressure|[silent]`
//!       (`mods/gen3/abilities.js` `addSplit`), and the Intimidate no-activate
//!       `|-hint|In Gen 3, Intimidate does not activate…` (owner-tagged by the
//!       driver, since a hint carries no `pNa:` prefix).
//!     * Every other line broadcasts verbatim to both sides.
//!
//! - **G1 — the `|request|{...}` JSON** (`RequestBuilder`): serialized from the
//!   crate's EXISTING legality (`is_trapped`, `MonState::move_usable`,
//!   forced-replacement flags) with Showdown's exact field + key order + compact
//!   `JSON.stringify` spacing (no spaces after `:`/`,`), matching
//!   `tests/vectors/bridge_request_schema_samples.json`.
//!
//! The byte target is `tests/vectors/bridge_capture_golden.txt` (30 gen3ou battles)
//! + `tests/vectors/bridge_trapping_golden.txt` (the trapped state machine), gated by
//! `tests/bridge_test.rs`.

use crate::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use crate::dex::Dex;
use crate::prng::{Prng, PrngSeed};
use crate::state::{BattleState, MonState, Status};
use crate::turn::{Choice, ScriptDecision};

// ===========================================================================
// The `>start` construction-window seed advance (the `sim_bridge` drop-in seed
// convention).
// ===========================================================================

/// Advance a RAW `>start` seed by the sim's turn-0 CONSTRUCTION-WINDOW draw so the
/// port's replay-from-genesis (which does NOT model that window) starts from the SAME
/// PRNG state Showdown reaches after `>start`+`>player`. This makes the `sim_bridge`
/// binary byte-identical to the real Node bridge on a seeded battle.
///
/// The construction window consumes exactly ONE PRNG draw — the gen-3 **turn-0 Quick
/// Claw** `randomChance(1, 5)` (fired once when the battle starts) — for a team with
/// EXPLICIT genders. VERIFIED vs the real sim: `[7,11,13,17]` → `[44317,42357,9927,48760]`
/// after `>start`, matched bit-for-bit by one `random_chance(1, 5)` here (see the unit
/// test `construction_seed_advance_matches_the_sim`). The engine's own draw suites use
/// this same "pre-first-decision seed convention" (the golden captures the sim's
/// post-construction `initSeed`; `Battle::start_with_switchins` then replays draw-free
/// from it) — this helper computes that post-construction seed from the raw one.
///
/// HONEST GAP: a mon with an UNSPECIFIED gender RATIO makes the sim draw an ADDITIONAL
/// construction-time `sample(['M','F'])` per such mon (an `addPokemon` draw the port
/// also does not model), which this helper does NOT account for. So the advance is
/// exact only for EXPLICIT-gender teams (the training/eval reality when packed teams
/// carry genders). The differential harness pins genders; the residual gender-sample
/// gap is documented, not faked.
pub fn advance_seed_for_construction(raw: &PrngSeed) -> PrngSeed {
    let mut p = Prng::new(raw);
    let _ = p.random_chance(1, 5); // the turn-0 Quick Claw
    p.get_seed()
}

// ===========================================================================
// Wire choice tokens (the CMD stream the driver replays).
// ===========================================================================

/// One `>pN <choice>` command from a bridge-capture golden's CMD stream, parsed
/// into a side + a 0-based engine [`Choice`]. `switch N` is 1-based on the wire
/// (→ `Switch(N-1)`), targeting the CURRENT `side.pokemon` array position AFTER any
/// prior switch swaps (the crate's array mirrors Showdown's — see `execute_switch`).
#[derive(Debug, Clone, Copy)]
pub struct Cmd {
    pub side: usize,
    pub choice: Choice,
}

/// Parse a wire choice token (`move K` / `switch N`, 1-based) into a 0-based
/// [`Choice`]. `None` for an unsupported token.
pub fn parse_choice(tok: &str) -> Option<Choice> {
    let tok = tok.trim();
    if let Some(k) = tok.strip_prefix("move ") {
        return Some(Choice::Move(k.trim().parse::<usize>().ok()?.checked_sub(1)?));
    }
    if let Some(n) = tok.strip_prefix("switch ") {
        return Some(Choice::Switch(n.trim().parse::<usize>().ok()?.checked_sub(1)?));
    }
    None
}

// ===========================================================================
// The per-side output: the ordered chunk lines each side received.
// ===========================================================================

/// The two per-side streams — each an ordered `Vec` of the raw protocol lines the
/// side would have received (log lines HP-folded + its own `|request|`/`|error|`
/// frames, in emission order). The bridge test asserts these byte-for-byte against
/// the golden's per-side CHUNK rows (flattened, in order).
#[derive(Debug, Clone, Default)]
pub struct BridgeStreams {
    pub p1: Vec<String>,
    pub p2: Vec<String>,
}

// ===========================================================================
// The CHUNKED per-side output — one `Vec<String>` per CHUNK (the `getPlayerStreams`
// flush unit the Node bridge base64-frames as ONE `pN <b64>` stdout line).
// ===========================================================================

/// One flush unit destined for one side — the batch of protocol lines
/// `getPlayerStreams` pushes on one battle `update` / `sideupdate`, folded
/// HP-privately. The Node bridge writes exactly one `pN <base64(chunk)>` stdout line
/// per such flush.
#[derive(Debug, Clone)]
pub struct SideChunk {
    /// 0 = p1, 1 = p2.
    pub side: usize,
    /// The chunk's lines, in order (joined by `\n` for the on-wire payload).
    pub lines: Vec<String>,
}

/// The battle's per-side chunk stream as ONE ordered list of [`SideChunk`]s, in the
/// order the driver flushed them (framing p1/p2 interleaved, then per boundary the
/// turn chunk + requests). This is the drop-in unit for `src/bin/sim_bridge.rs`: the
/// Node bridge writes one `pN <base64(chunk)>` line per chunk, so byte-identity over
/// the wire requires the per-side chunk BOUNDARIES to match, not just the flattened
/// line sequence.
///
/// [`run_full_battle_bridge`] flattens this (its line-level golden gate validates the
/// fold/request logic); the binary consumes the chunk grouping directly.
#[derive(Debug, Clone, Default)]
pub struct BridgeChunks {
    /// Every chunk, in flush order (both sides interleaved).
    pub chunks: Vec<SideChunk>,
}

impl BridgeChunks {
    /// Push ONE chunk (a non-empty batch of lines) to `side`. An empty batch is a
    /// no-op (the sim never flushes an empty chunk to a side).
    fn push_chunk(&mut self, side: usize, lines: Vec<String>) {
        if !lines.is_empty() {
            self.chunks.push(SideChunk { side, lines });
        }
    }
    /// This side's chunks, in order.
    pub fn side_chunks(&self, side: usize) -> impl Iterator<Item = &SideChunk> {
        self.chunks.iter().filter(move |c| c.side == side)
    }
    /// Flatten to line-level [`BridgeStreams`] (each side's chunk lines concatenated in
    /// order) — the backward-compatible view the line-level golden gate asserts.
    pub fn flatten(&self) -> BridgeStreams {
        let mut s = BridgeStreams::default();
        for c in &self.chunks {
            let dst = if c.side == 0 { &mut s.p1 } else { &mut s.p2 };
            dst.extend(c.lines.iter().cloned());
        }
        s
    }
}

// ===========================================================================
// G2 — the per-side split + HP-privacy fold.
// ===========================================================================

/// The percent HP a NON-owner sees, per `pokemon.js::getHealth` (gen3ou =
/// `reportPercentages`): `ceil(100*hp/maxhp)`, clamped so a `100` with `hp<maxhp`
/// shows `99`. `0` HP is handled by the caller (`0 fnt`).
fn hp_percent(hp: u32, maxhp: u32) -> u32 {
    if maxhp == 0 {
        return 0;
    }
    let mut pct = (100 * hp).div_ceil(maxhp);
    if pct == 100 && hp < maxhp {
        pct = 99;
    }
    pct
}

/// Fold ONE omniscient (secret) log line to the view of `for_side`. Returns
/// `None` when the line is OWNER-ONLY and `for_side` is not the owner (the empty-
/// `shared` drop). Otherwise the (possibly HP-%-folded) line for `for_side`.
///
/// `hint_owner` is the owner side of the pending Intimidate `|-hint|` (the one
/// owner-only line with no `pNa:` prefix); `None` unless the driver just flagged one.
/// `report_percent` = the non-owner sees a PERCENT HP (gen3ou's "HP Percentage
/// Mod"). A `debug:true` format (gen3customgame) sets `reportExactHP`, so both sides
/// see exact HP — pass `false` to disable the fold entirely.
fn derive_side(
    line: &str,
    for_side: usize,
    hint_owner: Option<usize>,
    report_percent: bool,
) -> Option<String> {
    // Owner-only: gen3 Pressure switch-in reveal — `|-ability|pNa: X|Pressure|[silent]`.
    if line.starts_with("|-ability|") && line.ends_with("|[silent]") {
        if let Some(owner) = ident_owner(line) {
            return if owner == for_side { Some(line.to_string()) } else { None };
        }
    }
    // Owner-only: the Intimidate no-activate hint (owner supplied by the driver).
    if let Some(owner) = hint_owner {
        if line.starts_with("|-hint|") {
            return if owner == for_side { Some(line.to_string()) } else { None };
        }
    }
    // HP-bearing lines: fold the HP token to a percent for the non-owner (only when
    // the format reports percentages — a debug format shows exact HP to both sides).
    if report_percent {
        if let Some(folded) = fold_hp_line(line, for_side) {
            return Some(folded);
        }
    }
    // Everything else broadcasts verbatim.
    Some(line.to_string())
}

/// The owner side of a line whose first field after the tag is a `pNa: <name>`
/// ident (`|switch|p1a: …`, `|-damage|p2a: …`, `|-ability|p1a: …`). `None` if no
/// such ident.
fn ident_owner(line: &str) -> Option<usize> {
    let parts: Vec<&str> = line.split('|').collect();
    // parts[0]="" parts[1]=tag parts[2]=ident (for the HP/ability lines)
    let ident = parts.get(2)?;
    side_of_ident(ident)
}

/// `p1a: Snorlax` → `Some(0)`, `p2a: …` → `Some(1)`.
fn side_of_ident(ident: &str) -> Option<usize> {
    let id = ident.trim_start();
    if let Some(rest) = id.strip_prefix("p1") {
        if rest.starts_with(|c: char| c == 'a' || c == ':') {
            return Some(0);
        }
    }
    if let Some(rest) = id.strip_prefix("p2") {
        if rest.starts_with(|c: char| c == 'a' || c == ':') {
            return Some(1);
        }
    }
    None
}

/// If `line` is an HP-bearing line (`|switch|`/`|drag|`/`|-damage|`/`|-heal|`/
/// `|-sethp|`), return its `for_side` fold: the owner keeps the exact `x/y`; the
/// other side gets the `ceil%/100` (status suffix preserved; `0 fnt` unchanged).
/// `None` when `line` is not an HP-bearing line.
fn fold_hp_line(line: &str, for_side: usize) -> Option<String> {
    let mut parts: Vec<&str> = line.split('|').collect();
    // parts[0]="" parts[1]=tag parts[2]=ident parts[3]=(switch/drag: details; damage/heal: HP) …
    let tag = *parts.get(1)?;
    let hp_idx = match tag {
        "switch" | "drag" => 4,    // |switch|ident|details|HP
        "-damage" | "-heal" | "-sethp" => 3, // |-damage|ident|HP[|from]
        _ => return None,
    };
    let owner = side_of_ident(parts.get(2)?)?;
    if owner == for_side {
        return Some(line.to_string()); // owner sees the secret (exact) HP already present
    }
    let hp_field = *parts.get(hp_idx)?;
    let folded = fold_hp_field(hp_field);
    let owned = folded.into_owned();
    parts[hp_idx] = &owned;
    Some(parts.join("|"))
}

/// Fold one HP field (`463/463`, `406/463 brn`, `0 fnt`) to the shared-percent
/// form the non-owner sees (`100/100`, `88/100 brn`, `0 fnt`).
fn fold_hp_field(field: &str) -> std::borrow::Cow<'_, str> {
    // Split an optional trailing " <status>".
    let (hp_part, status) = match field.split_once(' ') {
        Some((h, s)) => (h, Some(s)),
        None => (field, None),
    };
    if hp_part == "0" {
        // `0 fnt` — unchanged (both sides).
        return std::borrow::Cow::Borrowed(field);
    }
    let (cur, max) = match hp_part.split_once('/') {
        Some((c, m)) => (c, m),
        None => return std::borrow::Cow::Borrowed(field), // not `x/y` — leave as-is
    };
    let (cur, max) = match (cur.parse::<u32>(), max.parse::<u32>()) {
        (Ok(c), Ok(m)) => (c, m),
        _ => return std::borrow::Cow::Borrowed(field),
    };
    let pct = hp_percent(cur, max);
    let out = match status {
        Some(s) => format!("{pct}/100 {s}"),
        None => format!("{pct}/100"),
    };
    std::borrow::Cow::Owned(out)
}

// ===========================================================================
// G1 — the `|request|{...}` JSON emitter.
// ===========================================================================

/// Minimal JSON string escaper matching `JSON.stringify` for the chars that can
/// appear in gen3 idents/names (quote, backslash, control chars). Non-ASCII UTF-8
/// is emitted verbatim (JSON.stringify does not \u-escape it), matching the golden's
/// raw-UTF-8 nicknames (e.g. `Métalosse`).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// The `moves` array entry id for `side.pokemon[j]` (`getSwitchRequestData`): the
/// mon's move ID (normalized), which for a gen<6 typed Hidden Power is the TYPED id
/// (`hiddenpowergrass`). `set.moves` holds DISPLAY names, so normalize via `to_id`.
fn side_move_id(mv: &str) -> String {
    crate::dex::to_id(mv)
}

/// Split a crate move (a DISPLAY name from `set.moves`) into (bare id, display name)
/// for the `active[].moves[]` request entry. gen<6 Hidden Power shows `id:"hiddenpower"`
/// (BARE) + `move:"Hidden Power <Type> <BP>"`; every other move is `id`/`name`
/// verbatim.
fn active_move_display(mv: &str, dex: &Dex) -> (String, String) {
    let id = crate::dex::to_id(mv);
    let data = dex.moves(&id);
    let name = data.map(|d| d.name.clone()).unwrap_or_else(|| mv.to_string());
    if id.starts_with("hiddenpower") && id != "hiddenpower" {
        // Typed HP: bare id + "<Name> <BP>" (name already reads "Hidden Power <Type>").
        let bp = data.map(|d| d.base_power).unwrap_or(70);
        return ("hiddenpower".to_string(), format!("{name} {bp}"));
    }
    (id, name)
}

/// Whether a move slot is DISABLED for the request (`getMoves`): out of PP, or
/// Disable/Taunt/Choice-lock restricted. `move_usable` folds all of those; a move
/// is `disabled` iff it is NOT usable — EXCEPT when the whole mon must Struggle,
/// where the sim substitutes Struggle (handled by the caller) rather than marking
/// every slot disabled.
fn move_disabled(mon: &MonState, k: usize, dex: &Dex) -> bool {
    !mon.move_usable(k, dex)
}

/// The `condition` string (`getHealth().secret`): `0 fnt` / `x/y` / `x/y <status>`.
fn condition(mon: &MonState) -> String {
    if mon.hp == 0 {
        return "0 fnt".to_string();
    }
    match status_str(mon.status) {
        Some(s) => format!("{}/{} {}", mon.hp, mon.maxhp, s),
        None => format!("{}/{}", mon.hp, mon.maxhp),
    }
}

/// The protocol status token for a major status (mirrors the private
/// `turn::status_token`).
fn status_str(status: Option<Status>) -> Option<&'static str> {
    match status {
        Some(Status::Burn) => Some("brn"),
        Some(Status::Paralysis) => Some("par"),
        Some(Status::Sleep(_)) => Some("slp"),
        Some(Status::Freeze) => Some("frz"),
        Some(Status::Poison) => Some("psn"),
        Some(Status::Toxic(_)) => Some("tox"),
        None => None,
    }
}

/// The mon ident (`getFullDetails`→`fullname` = `pN: <Nickname>`): the nickname when
/// present, else the species display name.
fn mon_ident(side: usize, mon: &MonState, dex: &Dex) -> String {
    let name = display_name(mon, dex);
    format!("p{}: {}", side + 1, name)
}

/// The mon's display name — the nickname (`set.name`) when non-empty, else the
/// species display name.
fn display_name(mon: &MonState, dex: &Dex) -> String {
    if !mon.set.name.is_empty() {
        return mon.set.name.clone();
    }
    dex.species(&mon.species_id)
        .map(|s| s.name.clone())
        .unwrap_or_else(|| mon.species_id.clone())
}

/// The `details` string (`Pokemon.details`): `<Species>` + `, <Gender>` when the
/// mon has a gender ('M'/'F'; 'N'/none omitted). L100 is omitted (gen3 default).
fn details(mon: &MonState, dex: &Dex) -> String {
    let species = dex
        .species(&mon.species_id)
        .map(|s| s.name.clone())
        .unwrap_or_else(|| mon.species_id.clone());
    match mon.gender {
        Some('M') => format!("{species}, M"),
        Some('F') => format!("{species}, F"),
        _ => species,
    }
}

/// Serialize `side.pokemon[j]` (`getSwitchRequestData`) — the exact field order:
/// `ident, details, condition, active, stats{atk,def,spa,spd,spe}, moves[],
/// baseAbility, item, pokeball`.
fn serialize_mon(side: usize, mon: &MonState, active: bool, dex: &Dex) -> String {
    let ident = json_escape(&mon_ident(side, mon, dex));
    let det = json_escape(&details(mon, dex));
    let cond = json_escape(&condition(mon));
    let stats = format!(
        "{{\"atk\":{},\"def\":{},\"spa\":{},\"spd\":{},\"spe\":{}}}",
        mon.stats[1], mon.stats[2], mon.stats[3], mon.stats[4], mon.stats[5]
    );
    let moves = mon
        .set
        .moves
        .iter()
        .map(|m| format!("\"{}\"", json_escape(&side_move_id(m))))
        .collect::<Vec<_>>()
        .join(",");
    // baseAbility = the mon's ORIGINAL ability id (the set's ability, not a
    // Trace-copied one). The crate stores the current ability in `mon.ability`; the
    // request wants the base — read the (immutable) set ability id.
    let base_ability = crate::dex::to_id(&mon.set.ability);
    let item = crate::dex::to_id(&mon.item);
    format!(
        "{{\"ident\":\"{ident}\",\"details\":\"{det}\",\"condition\":\"{cond}\",\"active\":{active},\"stats\":{stats},\"moves\":[{moves}],\"baseAbility\":\"{base_ability}\",\"item\":\"{item}\",\"pokeball\":\"pokeball\"}}"
    )
}

/// Serialize the whole `side` object (`getRequestData`): `{name, id, pokemon:[...]}`.
fn serialize_side(state: &BattleState, side: usize, dex: &Dex) -> String {
    let s = &state.sides[side];
    let name = json_escape(&s.name);
    let id = format!("p{}", side + 1);
    let mons = s
        .pokemon
        .iter()
        .enumerate()
        .map(|(i, m)| serialize_mon(side, m, i == s.active, dex))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"name\":\"{name}\",\"id\":\"{id}\",\"pokemon\":[{mons}]}}")
}

/// Serialize the `active[0]` object (`getMoveRequestData`): `{moves:[...][,
/// maybeTrapped|trapped]}`.
///
/// `trapped_firm` = the mon is TRAPPED and a switch was already rejected this
/// request (→ `trapped:true`, dropping `maybeTrapped`). Otherwise a live-bench
/// trapped mon shows `maybeTrapped:true` (gen3 NEVER emits `trapped` on the first
/// request). No live bench → neither flag.
fn serialize_active(
    state: &BattleState,
    side: usize,
    trapped_firm: bool,
    dex: &Dex,
) -> String {
    let s = &state.sides[side];
    let mon = &s.pokemon[s.active];
    // Struggle substitution: a mon with NO usable move offers only Struggle.
    let must_struggle = mon.must_struggle(dex);
    let moves_json = if must_struggle {
        "[{\"move\":\"Struggle\",\"id\":\"struggle\",\"target\":\"randomNormal\",\"disabled\":false}]"
            .to_string()
    } else {
        let entries = mon
            .set
            .moves
            .iter()
            .enumerate()
            .map(|(k, mv)| {
                let (id, name) = active_move_display(mv, dex);
                let pp = mon.move_pp.get(k).copied().unwrap_or(0);
                let maxpp = mon.move_maxpp.get(k).copied().unwrap_or(0);
                let target = dex
                    .moves(mv)
                    .map(|d| d.target.clone())
                    .unwrap_or_else(|| "normal".to_string());
                let disabled = move_disabled(mon, k, dex);
                format!(
                    "{{\"move\":\"{}\",\"id\":\"{}\",\"pp\":{},\"maxpp\":{},\"target\":\"{}\",\"disabled\":{}}}",
                    json_escape(&name),
                    json_escape(&id),
                    pp,
                    maxpp,
                    json_escape(&target),
                    disabled
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("[{entries}]")
    };
    // The trap flag. gen3 only shows `trapped`/`maybeTrapped` when a live bench
    // exists AND the mon is trapped; Struggle-only (lockedMove) also firms trapped,
    // but our scope has no lockedMove moves.
    //
    // FIRM vs 'hidden' (the request/per-side A/B fuzzer's #1 find): the gen3 mod's
    // **Shadow Tag** sets `pokemon.trapped = true` DIRECTLY (`onFoeTrapPokemon`), so
    // `getMoveRequestData` emits `trapped:true` on the FIRST request — NO `maybeTrapped`
    // phase, no rejection round. Arena Trap / Magnet Pull call `tryTrap(true)` →
    // `trapped = 'hidden'`, so they show `maybeTrapped` until a rejected switch firms
    // them (`trapped_firm`). So a FIRM trap emits `trapped:true` unconditionally;
    // otherwise the `maybeTrapped`→`trapped` (on reject) machine applies.
    // (`state::trap_is_firm`; probe-settled vs the sim, `gen3_shadowtag_firm_trap_v1`.)
    let is_trapped = state.is_trapped(side, dex);
    let has_bench = has_live_bench(state, side);
    let firm = trapped_firm || state.trap_is_firm(side, dex);
    let flag = if is_trapped && has_bench {
        if firm {
            ",\"trapped\":true"
        } else {
            ",\"maybeTrapped\":true"
        }
    } else {
        ""
    };
    format!("{{\"moves\":{moves_json}{flag}}}")
}

/// Whether `side` has ≥1 live, non-active bench mon (mirrors `battle.canSwitch`).
fn has_live_bench(state: &BattleState, side: usize) -> bool {
    let s = &state.sides[side];
    s.pokemon
        .iter()
        .enumerate()
        .any(|(i, m)| i != s.active && !m.fainted)
}

/// A per-boundary request kind for one side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SideRequest {
    Move,
    ForceSwitch,
    Wait,
}

/// Build the `|request|{...}` line for `side` of the given kind. `trapped_firm`
/// only applies to a Move request; `update` appends `,"update":true` (the
/// post-rejection re-request), `no_cancel` appends `,"noCancel":true` to a
/// ForceSwitch.
fn build_request(
    state: &BattleState,
    side: usize,
    kind: SideRequest,
    trapped_firm: bool,
    update: bool,
    no_cancel: bool,
    dex: &Dex,
) -> String {
    let side_json = serialize_side(state, side, dex);
    let mut body = match kind {
        SideRequest::Move => {
            let active = serialize_active(state, side, trapped_firm, dex);
            format!("{{\"active\":[{active}],\"side\":{side_json}}}")
        }
        SideRequest::ForceSwitch => {
            format!("{{\"forceSwitch\":[true],\"side\":{side_json}}}")
        }
        SideRequest::Wait => {
            format!("{{\"wait\":true,\"side\":{side_json}}}")
        }
    };
    // Append the conditional trailing keys in the sim's order: (noCancel) then
    // (update). `getRequests` appends noCancel; `emitRequest(_, true)` appends update.
    if no_cancel {
        // Insert `,"noCancel":true` before the closing brace.
        body.pop();
        body.push_str(",\"noCancel\":true}");
    }
    if update {
        body.pop();
        body.push_str(",\"update\":true}");
    }
    format!("|request|{body}")
}

// ===========================================================================
// The driver — replay a CMD stream and emit both per-side streams.
// ===========================================================================

/// Run a battle from `opts` over the `cmds` command stream and produce the two
/// per-side chunk streams (framing + folded log + per-side `|request|`/`|error|`
/// frames), byte-identical to Showdown's `getPlayerStreams`.
///
/// The driver REPLAYS-from-genesis at each boundary (like `BattleStream::write_line`)
/// to snapshot the pending request state, folds the newly-flushed omniscient log
/// batch to each side, and injects each side's request. Trapped-switch REJECTIONS
/// in the CMD stream produce the `|error|` + `trapped:true` re-request round.
pub fn run_full_battle_bridge(
    opts: &BattleOptions,
    cmds: &[Cmd],
    dex: &Dex,
) -> Result<BridgeStreams, String> {
    Ok(run_full_battle_bridge_chunked(opts, cmds, dex)?.flatten())
}

/// The CHUNK-aware sibling of [`run_full_battle_bridge`]: identical fold/request
/// logic, but the per-side output preserves the `getPlayerStreams` CHUNK BOUNDARIES
/// (each `Vec<String>` = one flush unit → one `pN <base64>` stdout line in the Node
/// bridge). This is the emission engine for `src/bin/sim_bridge.rs`.
///
/// The chunk model, byte-verified against `node local_sim_bridge.js`:
/// - **Framing** → THREE chunks (the `>start` / `>player p1` / `>player p2` writes):
///   `[|t:| , |gametype|singles]`, `[|player|p1|…]`, `[|player|p2|… , … , |turn|1]`.
/// - **A resolved turn** → ONE log chunk per side (the flushed batch ending in
///   `…|upkeep|`+`|turn|N` or `|faint|`), then ONE `|request|` chunk per side.
/// - **A trapped-switch REJECT** → a `|error|` chunk + (for a `'hidden'` trap) a
///   `trapped:true` re-request chunk, on the rejecting side only.
/// - **A forced-Struggle move commit** → a `|-activate|…|move: Struggle` chunk on the
///   struggling side, before that turn's broadcast batch.
pub fn run_full_battle_bridge_chunked(
    opts: &BattleOptions,
    cmds: &[Cmd],
    dex: &Dex,
) -> Result<BridgeChunks, String> {
    Ok(run_full_battle_bridge_chunked_ended(opts, cmds, dex)?.0)
}

/// [`run_full_battle_bridge_chunked`] plus whether the accumulated command stream drove
/// the battle to its natural WIN/LOSS/TIE end (`true` once a terminal boundary was
/// reached — the emitter breaks after flushing the final `|win|`/`|tie|` batch). The
/// `sim_bridge` binary reads this to decide when to write `__END__`, without a second
/// replay.
pub fn run_full_battle_bridge_chunked_ended(
    opts: &BattleOptions,
    cmds: &[Cmd],
    dex: &Dex,
) -> Result<(BridgeChunks, bool), String> {
    let mut chunks = BridgeChunks::default();
    let mut battle_ended = false;
    // Percent HP fold applies only in non-debug formats (gen3ou). A `debug:true`
    // format (gen3customgame) sets `reportExactHP`, so both sides see exact HP.
    let report_percent = !format_is_debug(&opts.format_id);

    // ── Framing (per side, identical structure; gen3 has NO team preview). ──
    // A logged battle with an EMPTY script emits exactly the framing (framing + leads
    // + `|turn|1`) then breaks at the first (absent) decision.
    let raw_framing = {
        let mut b = Battle::start_with_switchins(opts, dex)?;
        let st = b.state_mut().ok_or("no state")?;
        let (_outcome, lines) = st.run_full_battle_logged(&[], dex);
        lines
    };
    let framing = reframe(&raw_framing, &opts.format_id);
    // The framing splits into 3 chunks at the two `|player|` lines (probe-verified vs
    // the Node bridge): [`|t:|`+`|gametype`] from `>start`, [`|player|p1|…`] from
    // `>player p1`, and [`|player|p2|…` … `|turn|1`] from `>player p2`.
    emit_framing_chunks(&mut chunks, &framing, report_percent);

    // ── The decision loop. Replay [0..k] fresh at each boundary to inspect the paused
    //    request state, and diff the omniscient log against the previous replay to get
    //    the batch flushed since. ──
    let mut script: Vec<ScriptDecision> = Vec::new();
    let mut prev_log_len = raw_framing.len();
    let mut cmd_iter = cmds.iter().peekable();
    let mut guard = 0usize;
    let cap = cmds.len() * 4 + 64;

    loop {
        guard += 1;
        if guard > cap {
            return Err(format!("bridge driver exceeded boundary cap ({cap})"));
        }
        // Replay the accumulated script; snapshot the paused state + the full log.
        let (ended, log, state_holder) = replay_snapshot(opts, &script, dex)?;
        let state = state_holder.state().ok_or("paused battle has no state")?;

        // The log batch flushed since the previous boundary → ONE chunk per side.
        emit_log_batch_chunk(&mut chunks, &log[prev_log_len..], report_percent);
        prev_log_len = log.len();

        if ended {
            battle_ended = true;
            break;
        }

        // Determine the pending request kind per side from the paused state.
        let force = pending_force(state);
        let boundary_is_switch = force[0] || force[1];
        let kinds = boundary_kinds(state, &force, boundary_is_switch);
        // noCancel on a forceSwitch iff <2 non-wait requests exist (mirrors
        // getRequests' `multipleRequestsExist`).
        let non_wait = kinds.iter().filter(|k| **k != SideRequest::Wait).count();
        let no_cancel_forced = non_wait < 2;
        // Each side's request for THIS boundary → ONE chunk per side.
        emit_boundary_request_chunks(&mut chunks, state, &kinds, no_cancel_forced, dex);

        // ── Consume the CMD(s) answering this boundary (see the flat driver's docs). ──
        let mut need: [bool; 2] = [
            kinds[0] != SideRequest::Wait,
            kinds[1] != SideRequest::Wait,
        ];
        if cmd_iter.peek().is_none() {
            break;
        }
        let mut got: [Option<Choice>; 2] = [None, None];
        let mut plan_ended = false;
        while need[0] || need[1] {
            let cmd = match cmd_iter.peek() {
                Some(c) => **c,
                None => {
                    plan_ended = true;
                    break;
                }
            };
            let s = cmd.side;
            if !need[s] {
                return Err(format!(
                    "unexpected CMD for side {s} at boundary {guard} (kinds {kinds:?})"
                ));
            }
            cmd_iter.next();
            // Trapped-switch rejection at a MOVE boundary (see the flat driver's docs +
            // `gen3_shadowtag_firm_trap_v1`). The `|error|` and the re-request are SEPARATE
            // chunks (the sim flushes each on the rejected write / the update).
            if kinds[s] == SideRequest::Move
                && matches!(cmd.choice, Choice::Switch(_))
                && state.is_trapped(s, dex)
                && has_live_bench(state, s)
            {
                if state.trap_is_firm(s, dex) {
                    chunks.push_chunk(
                        s,
                        vec!["|error|[Invalid choice] Can't switch: The active Pokémon is trapped"
                            .to_string()],
                    );
                } else {
                    chunks.push_chunk(
                        s,
                        vec!["|error|[Unavailable choice] Can't switch: The active Pokémon is trapped"
                            .to_string()],
                    );
                    let rereq = build_request(state, s, SideRequest::Move, true, true, false, dex);
                    chunks.push_chunk(s, vec![rereq]);
                }
                continue; // side s still needs a choice
            }
            got[s] = Some(cmd.choice);
            need[s] = false;
        }
        if plan_ended {
            break;
        }

        // [EMIT] the per-side OWNER-ONLY STRUGGLE announce (`gen3_struggle_activate_sideupdate_v1`)
        // — a `sideupdate` chunk before the broadcast `|move|` batch flushes next iteration.
        // Order p1 then p2. Only a Move whose active `must_struggle` triggers it.
        for s in 0..2 {
            if matches!(got[s], Some(Choice::Move(_))) {
                let mon = &state.sides[s].pokemon[state.sides[s].active];
                if mon.must_struggle(dex) {
                    let name = display_name(mon, dex);
                    chunks.push_chunk(
                        s,
                        vec![format!("|-activate|p{}a: {}|move: Struggle", s + 1, name)],
                    );
                }
            }
        }

        // Commit the accepted choices into a single ScriptDecision for this boundary.
        let mut dec = ScriptDecision::default();
        for s in 0..2 {
            if let Some(c) = got[s] {
                dec.set_side(s, c);
            }
        }
        script.push(dec);
    }

    Ok((chunks, battle_ended))
}

/// Split the reframed framing lines into the 3 `getPlayerStreams` chunks per side and
/// push them (HP-folded): chunk 1 `[|t:|, |gametype]` (`>start`), chunk 2 `[|player|p1]`
/// (`>player p1`), chunk 3 the rest through `|turn|1` (`>player p2`).
fn emit_framing_chunks(chunks: &mut BridgeChunks, framing: &[String], report_percent: bool) {
    // Boundaries: chunk 2 STARTS at the first `|player|` line, chunk 3 at the second.
    let player_idxs: Vec<usize> = framing
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("|player|"))
        .map(|(i, _)| i)
        .collect();
    // Robust to a non-standard framing: fall back to one chunk if the two `|player|`
    // lines aren't present (never happens for a real gen3 framing).
    let (b1, b2) = match (player_idxs.first(), player_idxs.get(1)) {
        (Some(&a), Some(&b)) => (a, b),
        _ => (framing.len(), framing.len()),
    };
    let segments: [&[String]; 3] = [&framing[..b1], &framing[b1..b2], &framing[b2..]];
    for seg in segments {
        for side in 0..2 {
            let folded: Vec<String> = seg
                .iter()
                .filter_map(|line| derive_side(line, side, None, report_percent))
                .collect();
            chunks.push_chunk(side, folded);
        }
    }
}

/// Fold one flushed omniscient log batch into ONE chunk per side. The Intimidate
/// `|-hint|` owner attribution (the gen3 hint form carries no `pNa:` prefix) tracks the
/// immediately-preceding `|switch|`/`|drag|` owner in the batch.
fn emit_log_batch_chunk(
    chunks: &mut BridgeChunks,
    batch: &[crate::protocol::ProtocolLine],
    report_percent: bool,
) {
    for side in 0..2 {
        let mut out: Vec<String> = Vec::new();
        let mut last_switch_side: Option<usize> = None;
        for line in batch {
            let l = &line.0;
            if l.starts_with("|switch|") || l.starts_with("|drag|") {
                last_switch_side = ident_owner(l);
            }
            let hint_owner = if l.starts_with("|-hint|") && l.contains("Intimidate does not activate")
            {
                last_switch_side
            } else {
                None
            };
            if let Some(folded) = derive_side(l, side, hint_owner, report_percent) {
                out.push(folded);
            }
        }
        chunks.push_chunk(side, out);
    }
}

/// Emit the per-side request frame for both sides at a boundary — ONE chunk each.
fn emit_boundary_request_chunks(
    chunks: &mut BridgeChunks,
    state: &BattleState,
    kinds: &[SideRequest; 2],
    no_cancel_forced: bool,
    dex: &Dex,
) {
    for side in 0..2 {
        let no_cancel = kinds[side] == SideRequest::ForceSwitch && no_cancel_forced;
        let line = build_request(state, side, kinds[side], false, false, no_cancel, dex);
        chunks.push_chunk(side, vec![line]);
    }
}

/// The `|tier|` + `|rule|…` framing lines for gen3ou (the crate's `emit_framing`
/// hard-codes the gen3customgame `[Gen 3] Custom Game` tier + single HP-% rule). The
/// exact OU sequence is captured verbatim from the golden.
const GEN3OU_TIER: &str = "|tier|[Gen 3] OU";
const GEN3OU_RULES: &[&str] = &[
    "|rule|HP Percentage Mod: HP is shown in percentages",
    "|rule|Beat Up Nicknames Mod: Beat Up will not reveal any party members",
    "|rule|Endless Battle Clause: Forcing endless battles is banned",
    "|rule|Sleep Clause Mod: Limit one foe put to sleep",
    "|rule|Switch Priority Clause Mod: Faster Pokémon switch first",
    "|rule|Species Clause: Limit one of each Pokémon",
    "|rule|OHKO Clause: OHKO moves are banned",
    "|rule|Evasion Items Clause: Evasion items are banned",
    "|rule|Evasion Moves Clause: Evasion moves are banned",
    "|rule|One Boost Passer Clause: Limit one Baton Passer that has a way to boost its stats",
    "|rule|Freeze Clause Mod: Limit one foe frozen",
    "|rule|Baton Pass Stat Clause: No Baton Passer may have a way to boost its Speed",
];

/// Rewrite the crate's (gen3customgame) framing to the target format's tier + rule
/// list. For gen3ou this replaces `|tier|[Gen 3] Custom Game` + the single
/// `|rule|HP Percentage Mod…` line with the OU tier + 12-rule list; everything else
/// (players, gen, teamsize, leads, switch-in ability lines, `|turn|1`) is preserved.
/// A non-OU format is passed through unchanged.
fn reframe(raw: &[crate::protocol::ProtocolLine], format_id: &str) -> Vec<String> {
    if !format_id.contains("gen3ou") {
        return raw.iter().map(|l| l.0.clone()).collect();
    }
    let mut out: Vec<String> = Vec::with_capacity(raw.len() + GEN3OU_RULES.len());
    for l in raw {
        let s = &l.0;
        if s == "|tier|[Gen 3] Custom Game" {
            out.push(GEN3OU_TIER.to_string());
        } else if s == "|rule|HP Percentage Mod: HP is shown in percentages" {
            // Replace the lone Custom-Game HP% rule with the full OU rule list.
            for r in GEN3OU_RULES {
                out.push((*r).to_string());
            }
        } else {
            out.push(s.clone());
        }
    }
    out
}

/// Whether `format_id` is a `debug:true` format (the gen3 Custom Game) — such a
/// battle sets `reportExactHP`, so both per-side streams show EXACT HP (no percent
/// fold). Mirrors the ONE relevant flag from `[Gen 3] Custom Game`'s definition.
fn format_is_debug(format_id: &str) -> bool {
    format_id.contains("customgame")
}

/// Which sides are being force-switched at the paused boundary (post-faint
/// replacement). Reads `SideState.switch_flag` (set by the faint protocol).
fn pending_force(state: &BattleState) -> [bool; 2] {
    [state.sides[0].switch_flag, state.sides[1].switch_flag]
}

/// The per-side request kind at this boundary.
fn boundary_kinds(_state: &BattleState, force: &[bool; 2], is_switch: bool) -> [SideRequest; 2] {
    if is_switch {
        [
            if force[0] { SideRequest::ForceSwitch } else { SideRequest::Wait },
            if force[1] { SideRequest::ForceSwitch } else { SideRequest::Wait },
        ]
    } else {
        [SideRequest::Move, SideRequest::Move]
    }
}

/// Replay the battle from genesis through `script`, returning `(ended, full_log,
/// paused_battle)`. The paused battle exposes the boundary state for request
/// serialization.
fn replay_snapshot(
    opts: &BattleOptions,
    script: &[ScriptDecision],
    dex: &Dex,
) -> Result<(bool, Vec<crate::protocol::ProtocolLine>, Battle), String> {
    let mut battle = Battle::start_with_switchins(opts, dex)?;
    let st = battle.state_mut().ok_or("no state")?;
    let (outcome, log) = st.run_full_battle_logged(script, dex);
    Ok((outcome.ended, log, battle))
}

// ===========================================================================
// Golden parser — shared by `tests/bridge_test.rs` + the `bridge_replay` binary.
// ===========================================================================

/// One battle parsed from a bridge-capture golden (`bridge_capture_golden.txt` /
/// `bridge_trapping_golden.txt`): teams + seed + the CMD stream + the expected
/// per-side chunk lines (flattened, in order).
#[derive(Debug, Clone, Default)]
pub struct GoldenBattle {
    pub id: String,
    pub format_id: String,
    pub seed: String,
    pub p1_team: String,
    pub p2_team: String,
    pub cmds: Vec<Cmd>,
    /// Expected p1 / p2 chunk lines (raw payloads, in chunk+line order).
    pub p1_expected: Vec<String>,
    pub p2_expected: Vec<String>,
}

/// Parse a bridge golden's TAB grammar into per-battle records. Grammar
/// (documented in `harness/gen_bridge_capture.js`):
///   `SCEN <id>` / `TEAM <id> <p1|p2> <pack>` / `INIT <id> <bno> <seed m,n,o,p> <fmt>`
///   / `CMD <id> <bno> <cmdNo> <side> <choice>` / `CHUNK <id> <bno> <side> <chunkNo>
///   <lineNo> <raw>` / `END …`. Comment lines (`#`) are skipped.
pub fn parse_bridge_golden(text: &str) -> Result<Vec<GoldenBattle>, String> {
    use std::collections::BTreeMap;
    let mut order: Vec<String> = Vec::new();
    let mut map: BTreeMap<String, GoldenBattle> = BTreeMap::new();
    let ensure = |map: &mut BTreeMap<String, GoldenBattle>, order: &mut Vec<String>, id: &str| {
        if !map.contains_key(id) {
            let mut b = GoldenBattle::default();
            b.id = id.to_string();
            map.insert(id.to_string(), b);
            order.push(id.to_string());
        }
    };
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "SCEN" => {
                ensure(&mut map, &mut order, f[1]);
            }
            "TEAM" => {
                ensure(&mut map, &mut order, f[1]);
                let b = map.get_mut(f[1]).unwrap();
                let pack = f[3..].join("\t"); // pack can't contain a tab, but be safe
                if f[2] == "p1" {
                    b.p1_team = pack;
                } else {
                    b.p2_team = pack;
                }
            }
            "INIT" => {
                ensure(&mut map, &mut order, f[1]);
                let b = map.get_mut(f[1]).unwrap();
                // f: INIT id bno "m,n,o,p" fmt
                b.seed = f[3].to_string();
                b.format_id = f[4].to_string();
            }
            "CMD" => {
                ensure(&mut map, &mut order, f[1]);
                let b = map.get_mut(f[1]).unwrap();
                // f: CMD id bno cmdNo side choice
                let side = if f[4] == "p1" { 0 } else { 1 };
                let choice_tok = f[5..].join("\t");
                let choice = parse_choice(&choice_tok)
                    .ok_or_else(|| format!("bad CMD choice {choice_tok:?} in {}", f[1]))?;
                b.cmds.push(Cmd { side, choice });
            }
            "CHUNK" => {
                ensure(&mut map, &mut order, f[1]);
                let b = map.get_mut(f[1]).unwrap();
                // f: CHUNK id bno side chunkNo lineNo raw…
                let side = f[3];
                let raw = f[6..].join("\t"); // the raw line (may contain tabs? no — but safe)
                if side == "p1" {
                    b.p1_expected.push(raw);
                } else {
                    b.p2_expected.push(raw);
                }
            }
            "END" => {}
            _ => {}
        }
    }
    Ok(order.into_iter().map(|id| map.remove(&id).unwrap()).collect())
}

/// Build the [`BattleOptions`] from packed teams + a seed string.
pub fn bridge_opts(
    format_id: &str,
    seed: PrngSeed,
    p1_team: &str,
    p2_team: &str,
) -> BattleOptions {
    BattleOptions {
        format_id: format_id.to_string(),
        seed: Some(seed),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1_team.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2_team.to_string()) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `advance_seed_for_construction` must reproduce the real sim's post-`>start` PRNG state
    /// for a DISTINCT-SPEED lead matchup (the common case): the sim's turn-0 construction
    /// window is exactly the Quick Claw `random(1,5)` there, matched by one `random_chance(1,5)`.
    /// Ground truth captured from the omniscient `getPlayerStreams` (raw seed → post-`>start`
    /// `battle.prng.getSeed()`): `[7,11,13,17]` → `[44317,42357,9927,48760]`, and a second seed
    /// `[99,88,77,66]` → `[17147,43298,11641,2765]` (see the module docs for the SPEED-TIE gap).
    #[test]
    fn construction_seed_advance_matches_the_sim() {
        assert_eq!(
            advance_seed_for_construction(&"7,11,13,17".to_string()),
            "44317,42357,9927,48760"
        );
        assert_eq!(
            advance_seed_for_construction(&"99,88,77,66".to_string()),
            "17147,43298,11641,2765"
        );
    }

    /// The chunked emitter's `flatten()` must equal the flat `run_full_battle_bridge` output —
    /// the invariant that lets the line-level `bridge_test` golden validate the chunk logic.
    #[test]
    fn chunked_flatten_equals_flat_streams() {
        let dex = Dex::for_gen(3);
        // A tiny explicit-gender battle (Snorlax vs Tauros, both Splash then a move).
        let p1 = "Snorlax||||bodyslam,splash|Serious||M||||";
        let p2 = "Tauros||||bodyslam,splash|Serious||M||||";
        let opts = bridge_opts("gen3customgame", "44317,42357,9927,48760".to_string(), p1, p2);
        let cmds = vec![
            Cmd { side: 0, choice: Choice::Move(1) },
            Cmd { side: 1, choice: Choice::Move(1) },
            Cmd { side: 0, choice: Choice::Move(0) },
            Cmd { side: 1, choice: Choice::Move(0) },
        ];
        let flat = run_full_battle_bridge(&opts, &cmds, &dex).unwrap();
        let chunked = run_full_battle_bridge_chunked(&opts, &cmds, &dex).unwrap();
        assert_eq!(chunked.flatten().p1, flat.p1);
        assert_eq!(chunked.flatten().p2, flat.p2);
        // And the chunk grouping is non-trivial (framing splits into 3 chunks/side + requests).
        assert!(chunked.side_chunks(0).count() >= 4, "p1 must have ≥4 chunks (framing + requests)");
    }
}
