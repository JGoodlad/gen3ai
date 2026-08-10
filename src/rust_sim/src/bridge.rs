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
use crate::prng::PrngSeed;
use crate::state::{BattleState, MonState, Status};
use crate::turn::{Choice, FullBattleDriver, ScriptDecision};

// ===========================================================================
// Wire choice tokens (the CMD stream the driver replays).
// ===========================================================================

/// A wire-level choice token BEFORE it is resolved to a 0-based engine [`Choice`].
///
/// The bridge-capture goldens use the NUMERIC forms (`move K` / `switch N`, 1-based);
/// the REAL poke-env RL runtime (`BattleOrder::message` in `battle_order.py`) serializes
/// choices as NAMES — `/choose move <move_id>` (e.g. `move earthquake`,
/// `move hiddenpowerice`) and `/choose switch <species_name>` (e.g. `switch Salamence`) —
/// exactly like the live Showdown server, which resolves ids/species natively. A NAME
/// cannot be turned into a slot at parse time (slots shift with position swaps), so it is
/// carried here and resolved at the decision boundary (`resolve` below) against the LIVE
/// side state, mirroring how Showdown's `side.chooseMove`/`chooseSwitch` resolve.
///
/// A numeric token ALWAYS parses to `Move`/`Switch` (byte-identical to the old
/// numeric-only `parse_choice`), so the golden / writeline paths are unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireChoice {
    /// `move K` — 1-based move slot on the wire (→ `Choice::Move(K-1)`).
    Move(usize),
    /// `switch N` — 1-based team slot on the wire (→ `Choice::Switch(N-1)`).
    Switch(usize),
    /// `move <move_id>` — the move's ID string (e.g. `earthquake`, `hiddenpowerice`),
    /// resolved against the active mon's ordered moveset.
    MoveName(String),
    /// `switch <species_name>` — the species / pokemon name, resolved against the
    /// team's bench species.
    SwitchSpecies(String),
}

/// One `>pN <choice>` command from a bridge-capture golden's CMD stream OR the live
/// RL runtime — a side + an unresolved [`WireChoice`]. `switch N` is 1-based on the
/// wire (→ `Switch(N-1)`), targeting the CURRENT `side.pokemon` array position AFTER any
/// prior switch swaps (the crate's array mirrors Showdown's — see `execute_switch`).
#[derive(Debug, Clone)]
pub struct Cmd {
    pub side: usize,
    pub choice: WireChoice,
}

/// Parse a wire choice token into an unresolved [`WireChoice`]. A NUMERIC token after
/// `move `/`switch ` yields the 0-based `Move`/`Switch` slot variant (byte-identical to
/// the pre-name parser); a NON-numeric token yields the `MoveName`/`SwitchSpecies`
/// variant (resolved later against the live state, like the real Showdown server).
/// `None` for a token with neither the `move `/`switch ` prefix.
pub fn parse_choice(tok: &str) -> Option<WireChoice> {
    let tok = tok.trim();
    if let Some(rest) = tok.strip_prefix("move ") {
        let rest = rest.trim();
        return Some(match rest.parse::<usize>() {
            Ok(k) => WireChoice::Move(k.checked_sub(1)?),
            Err(_) => WireChoice::MoveName(rest.to_string()),
        });
    }
    if let Some(rest) = tok.strip_prefix("switch ") {
        let rest = rest.trim();
        return Some(match rest.parse::<usize>() {
            Ok(n) => WireChoice::Switch(n.checked_sub(1)?),
            Err(_) => WireChoice::SwitchSpecies(rest.to_string()),
        });
    }
    None
}

/// Resolve a [`WireChoice`] to a 0-based engine [`Choice`] against the LIVE battle
/// state for `side` (the same resolution Showdown's `side.chooseMove`/`chooseSwitch`
/// do at the decision boundary). The numeric variants pass through unchanged.
///
/// - **`MoveName(id)`** — normalize `id` via [`crate::dex::to_id`] and match it against
///   each of the active mon's moveslots (also normalized). A gen<6 TYPED Hidden Power
///   is stored as its TYPED id (`hiddenpowerice`) — [`side_move_id`] normalizes each slot
///   to that same form — so `move hiddenpowerice` matches its slot. `None` (illegal —
///   handled exactly like a numeric out-of-range slot) if no slot matches.
/// - **`SwitchSpecies(name)`** — normalize `name` and match it against each bench mon's
///   species id, returning the FIRST non-active, non-fainted slot. `None` if none match.
pub fn resolve_choice(state: &BattleState, side: usize, choice: &WireChoice) -> Option<Choice> {
    match choice {
        WireChoice::Move(k) => Some(Choice::Move(*k)),
        WireChoice::Switch(n) => Some(Choice::Switch(*n)),
        WireChoice::MoveName(id) => {
            let want = crate::dex::to_id(id);
            let mon = &state.sides[side].pokemon[state.sides[side].active];
            // STRUGGLE (`gen3_bridge_struggle_resolve_v1`, the PP-exhaustion wedge fix): a mon
            // with NO usable move (all slots at 0 PP / disabled / taunted) gets a request
            // offering ONLY the synthetic `{"move":"Struggle","id":"struggle"}` (see
            // `build_request`'s `must_struggle` branch), and poke-env answers `move struggle` —
            // a move-ID that is NEVER present in the REAL moveset. Map it to `Move(0)`:
            // `feed_await_move` then accepts it (the `must_struggle` exception in
            // `choice_is_legal`) and the queue build substitutes Struggle. WITHOUT this, the
            // moveset search below returns `None` → the caller's out-of-range fallback
            // (`Move(moves.len())`) → `choice_is_legal` rejects it → the bridge re-issues the
            // SAME request forever (a bridge↔Python no-progress loop that wedged
            // `--use-bridge=rust` training on any battle reaching Struggle — node resolves it
            // and the game ends). `"struggle"` is never a real moveslot id, so this is
            // unambiguous; a numeric `move 1` already resolves via the `WireChoice::Move` arm.
            if want == "struggle" {
                return Some(Choice::Move(0));
            }
            // MOVE-LOCKED request (`gen3_move_coverage_batch4c_v1`): the request offered a
            // SINGLE pseudo/locked entry, so the wire's `move recharge` / `move solarbeam`
            // resolves to slot 1 of the REQUEST = the engine's Move(0) (the sim accepts
            // both `move 1` and `move recharge` — probed). Matched against the entry's id.
            if mon.move_locked() {
                let locked_id = if mon.must_recharge {
                    "recharge".to_string()
                } else {
                    mon.two_turn
                        .as_ref()
                        .and_then(|t| mon.set.moves.get(t.move_index))
                        .map(|m| crate::dex::to_id(m))
                        .unwrap_or_default()
                };
                return if want == locked_id { Some(Choice::Move(0)) } else { None };
            }
            // Match by move id, collapsing EITHER side's Hidden Power to bare so a wire
            // choice (which poke-env may send TYPED, `move hiddenpowerdark`, or BARE,
            // `move hiddenpower`) resolves against the slot regardless of how the request
            // rendered it — a mon carries at most ONE Hidden Power, so the collapse is
            // unambiguous (`gen3_own_typed_hp_request_roster_v1`; the roster serializer
            // `side_move_id` now emits the TYPED id, but the choice-resolution here must
            // stay lenient to both wire forms).
            let hp_canon = |id: &str| -> String {
                if id.starts_with("hiddenpower") {
                    "hiddenpower".to_string()
                } else {
                    id.to_string()
                }
            };
            let want_c = hp_canon(&want);
            mon.set
                .moves
                .iter()
                .position(|m| hp_canon(&crate::dex::to_id(m)) == want_c)
                .map(Choice::Move)
        }
        WireChoice::SwitchSpecies(name) => {
            // Match the NICKNAME first, then the species (`gen3_bridge_switch_nickname_v1`).
            //
            // poke-env keys every mon by its ON-FIELD IDENT — `p1a: <nickname>` — and serializes a
            // switch as `switch <that name>`. For a nicknamed mon that name is NOT the species:
            // the gen3ou pool ships localized teams like `Airmure (Skarmory)`, so poke-env sends
            // `switch Airmure`. Matching only `species_id` left that unresolvable, and an
            // unresolvable choice never commits the boundary — the bridge re-requests, poke-env
            // re-sends the same name, and the battle makes no progress while the env's `step()`
            // waits on a message that never comes (a 120s silent stall that killed the whole
            // training run). This is the CHOICE-RESOLUTION half of `gen3_nickname_ident_v1`,
            // whose emission half already renders the nickname in every ident.
            //
            // Nickname takes precedence: a mon nicknamed after a DIFFERENT species (a legal,
            // deliberately confusing team) must resolve to the mon actually called that, exactly
            // as Showdown's `side.chooseSwitch` does (it matches `pokemon.name` first).
            let want = crate::dex::to_id(name);
            let s = &state.sides[side];
            let eligible = |i: usize, m: &crate::state::MonState| i != s.active && !m.fainted;
            s.pokemon
                .iter()
                .enumerate()
                .find_map(|(i, m)| {
                    (eligible(i, m) && !m.set.name.is_empty()
                        && crate::dex::to_id(&m.set.name) == want)
                        .then_some(Choice::Switch(i))
                })
                .or_else(|| {
                    s.pokemon.iter().enumerate().find_map(|(i, m)| {
                        (eligible(i, m) && crate::dex::to_id(&m.species_id) == want)
                            .then_some(Choice::Switch(i))
                    })
                })
        }
    }
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
/// (`hiddenpowerdark`). `set.moves` holds DISPLAY names, so normalize via `to_id`.
///
/// THE OWNER-OWN typed-HP RESOLUTION (`gen3_own_typed_hp_request_roster_v1`, the
/// `bab_3_19` per-side request find): the Showdown `Pokemon` constructor resolves a
/// BARE `HiddenPower` move slot to `hiddenpower<hpType>` (via `set.hpType` if the
/// packed set carries the marker, else IV-derived — `getHiddenPower(ivs).type`), and
/// `getSwitchRequestData` returns that TYPED moveSlot id — so the OWNER sees their own
/// bench mon's TYPED HP id. (A set that already stores the explicit typed name
/// [`HiddenPowerGrass`] `to_id`s straight to `hiddenpowergrass`, so this only fires for
/// the bare-stored-plus-marker form Charizard uses: packed `HiddenPower` + `,Dark,,,,`.)
/// The OWNER-facing `active[].moves[]` request ALSO resolves the typed HP — the caller
/// runs `side_move_id` before `active_move_display` (`gen3_own_typed_hp_active_request_v1`,
/// the sibling of this roster fix). Only the opponent HP hiding + the omniscient `|move|`
/// bare-collapse (BF1) stay bare — gen 3 never reveals the opp HP type.
fn side_move_id(mon: &MonState, mv: &str) -> String {
    // Delegates to the ONE resolver (`gen3_transform_v1` moved it to `state.rs` so the
    // Transform copy resolves the TARGET's HP with the same rule — the sim copies
    // `hpType`/`hpPower` for gen<5, and the port encodes the type in the move id).
    crate::state::typed_hp_move_id(mon, mv)
}

/// Split a crate move (a DISPLAY name from `set.moves`) into (bare id, display name)
/// for the `active[].moves[]` request entry. gen<6 Hidden Power shows `id:"hiddenpower"`
/// (BARE) + `move:"Hidden Power <Type> <BP>"`; every other move is `id`/`name`
/// verbatim.
///
/// `hp_bp` is the ACTIVE mon's IV-derived Hidden Power BP (`MonState.hidden_power_bp`,
/// `gen3_iv_derived_hidden_power_bp_v1`). The real sim renders `Hidden Power <Type>
/// <this.hpPower>` (pokemon.ts, `getMoveRequestData`), and `hpPower` is IV-derived — so
/// the request move NAME must use the mon's BP, NOT the flat data 70 (else a BP-68 HP mon
/// shows "Hidden Power Ice 70" and diverges from the sim / confuses poke-env).
fn active_move_display(mv: &str, dex: &Dex, hp_bp: u8) -> (String, String) {
    let id = crate::dex::to_id(mv);
    let data = dex.moves(&id);
    let name = data.map(|d| d.name.clone()).unwrap_or_else(|| mv.to_string());
    if id.starts_with("hiddenpower") && id != "hiddenpower" {
        // Typed HP: bare id + "<Name> <BP>" (name already reads "Hidden Power <Type>").
        return ("hiddenpower".to_string(), format!("{name} {hp_bp}"));
    }
    (id, name)
}

/// Whether a move slot is DISABLED for the request (`getMoves`): out of PP, or
/// Disable/Taunt/Choice-lock restricted. `move_usable` folds all of those; a move
/// is `disabled` iff it is NOT usable — EXCEPT when the whole mon must Struggle,
/// where the sim substitutes Struggle (handled by the caller) rather than marking
/// every slot disabled.
///
/// THE CHOICE LOCK is NOT re-derived here (`gen3_choicelock_after_move_v1`). It rides
/// `move_usable` → `choice_lock_enforced`, i.e. the engine's `choice_locked_move`
/// VOLATILE narrowed by the current-item gate — a faithful model of the sim's
/// `choicelock`, which `choiceband.onAfterMove` adds at the END of runMove.
///
/// This used to fold a LAZY lock of its own (`gen3_choice_lock_request_disabled_v1`, the
/// `bab_3_24` find): "holds a Choice item ∧ has a `last_move` ⇒ disable every other slot".
/// That was a WORKAROUND for the engine setting `choice_locked_move` one phase too early
/// (at PP-deduct, so a Thief that STOLE a Band never locked). It over-locked the mirror
/// case — a mon TRICKED a Band by a slower foe AFTER it had already moved holds a Choice
/// item and has a `last_move`, yet the sim adds NO volatile and keeps all four moves
/// selectable (soak3 `sbd_msb1zfxs_b97`). Fixing the engine's TIMING covers both, so the
/// workaround is gone; see the AfterMove comment in `turn/driver.rs`.
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
    // TRANSFORM (`gen3_transform_v1`): `pokemon.name` is fixed at construction, so a
    // transformed mon's ident stays its ORIGINAL species — read through the overlay.
    let sid = base_species_id(mon);
    dex.species(sid).map(|s| s.name.clone()).unwrap_or_else(|| sid.to_string())
}

/// The mon's CONSTRUCTION species id — `pokemon.baseSpecies`. The request/ident surfaces
/// read THIS because neither `transformInto` (`gen3_transform_v1`, probe C: a Ditto copying
/// Snorlax still serializes `"ident":"p1: Ditto","details":"Ditto"`) nor a non-permanent
/// `formeChange` (`gen3_forecast_v1`, probe S1: a FORMED Castform's request details stay
/// `"Castform, F"`) ever refreshes `pokemon.name` / `pokemon.details`. Reads the
/// construction-fixed [`MonState::base_species_id`] — which also gets the FORMED-Castform-
/// then-Transforms corner right (the overlay's stored species would be the FORME; the sim's
/// `baseSpecies` is the construction species).
fn base_species_id(mon: &MonState) -> &str {
    &mon.base_species_id
}

/// The `details` string (`Pokemon.details`): `<Species>` + `, L<level>` iff level
/// != 100 + `, <Gender>` when the mon has a gender ('M'/'F'; 'N'/none omitted) +
/// `, shiny` for a shiny set. Showdown emits `, L<n>` in the request-JSON `details`
/// field for a non-L100 mon and OMITS it at L100 (gen3 default) — probe-confirmed
/// order `<Species>[, L<level>][, <gender>][, shiny]` (level BEFORE gender),
/// IDENTICAL to the omniscient `|switch|` details form (`turn.rs::switch_details`).
/// gen3ou is always L100 so the pool request goldens never emit it (byte-identical);
/// randbats surface it. The `, shiny` suffix mirrors the omniscient `|switch|`
/// details form (`gen3_shiny_details_v1`, byte-fuzz fixture 08) — the sim's
/// request-JSON carries it too (`"details":"Blissey, F, shiny"`).
fn details(mon: &MonState, dex: &Dex) -> String {
    // `Pokemon.details` is set in the CONSTRUCTOR and refreshed only by `formeChange` —
    // `transformInto` calls `setSpecies` DIRECTLY and never touches it, so a transformed
    // mon's request `details` still reads its own species (`gen3_transform_v1`, probe C).
    let sid = base_species_id(mon);
    let species = dex.species(sid).map(|s| s.name.clone()).unwrap_or_else(|| sid.to_string());
    let mut d = species;
    if mon.level != 100 {
        d.push_str(&format!(", L{}", mon.level));
    }
    match mon.gender {
        Some('M') => d.push_str(", M"),
        Some('F') => d.push_str(", F"),
        _ => {}
    }
    if mon.set.shiny {
        d.push_str(", shiny");
    }
    d
}

/// Serialize `side.pokemon[j]` (`getSwitchRequestData`) — the exact field order:
/// `ident, details, condition, active, stats{atk,def,spa,spd,spe}, moves[],
/// baseAbility, item, pokeball`.
fn serialize_mon(side: usize, mon: &MonState, active: bool, dex: &Dex) -> String {
    let ident = json_escape(&mon_ident(side, mon, dex));
    let det = json_escape(&details(mon, dex));
    let cond = json_escape(&condition(mon));
    // `getSwitchRequestData` serializes **`baseStoredStats`**, and
    // `setSpecies(…, isTransform = true)` deliberately does NOT overwrite those — so a
    // transformed mon's roster `stats` stay its OWN (`gen3_transform_v1`, probe C: a Ditto
    // copying Snorlax still reports `132/132/132/132/132`). Its `moves[]` below DO change,
    // because that field reads the live `moveSlots`.
    let st = match &mon.transform {
        Some(ov) => &ov.base_stats,
        None => &mon.stats,
    };
    let stats = format!(
        "{{\"atk\":{},\"def\":{},\"spa\":{},\"spd\":{},\"spe\":{}}}",
        st[1], st[2], st[3], st[4], st[5]
    );
    let moves = mon
        .set
        .moves
        .iter()
        .map(|m| format!("\"{}\"", json_escape(&side_move_id(mon, m))))
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
    // MOVE-LOCKED request (`gen3_move_coverage_batch4c_v1` — Hyper Beam's mustrecharge /
    // Solar Beam's twoturnmove): the request offers a SINGLE pseudo/locked move entry
    // with ONLY `{move,id}` — NO pp/maxpp/target/disabled keys (Showdown's lockedmove
    // request serialization, probe-verified:
    // `{"moves":[{"move":"Recharge","id":"recharge"}],"trapped":true}` /
    // `{"moves":[{"move":"Solar Beam","id":"solarbeam"}],"trapped":true}`) — plus the
    // FIRM `trapped:true` (below; no maybeTrapped phase — a rejected switch draws
    // `[Invalid choice] Can't switch: The active Pokémon is trapped` with no re-request,
    // like Shadow Tag). HONEST SCOPE: probe-shaped, not yet byte-gated by a bridge
    // capture scenario.
    if mon.move_locked() {
        let entry = if mon.must_recharge {
            "{\"move\":\"Recharge\",\"id\":\"recharge\"}".to_string()
        } else {
            let mid = mon
                .two_turn
                .as_ref()
                .and_then(|t| mon.set.moves.get(t.move_index))
                .cloned()
                .unwrap_or_else(|| "solarbeam".to_string());
            let (id, name) = active_move_display(&mid, dex, mon.hidden_power_bp);
            format!("{{\"move\":\"{}\",\"id\":\"{}\"}}", json_escape(&name), json_escape(&id))
        };
        // BR1 (`gen3_locked_last_mon_trapped_v1`): a move-LOCKED mon is `hardLocked` in
        // getMoveRequestData (pokemon.js:726-727, 744-749): `this.trapped` is set true and
        // `hardLocked || canSwitchIn` is ALWAYS true, so `trapped:true` is emitted
        // UNCONDITIONALLY — even for the LAST mon with no live bench (probe-verified: the
        // recharge / two-turn fire request carries `trapped:true` WITH and WITHOUT a bench;
        // bridge-fuzz BR1). The old `has_live_bench` gate wrongly dropped it on a last-mon
        // recharge/fire turn, so poke-env thought the last mon could switch.
        return format!("{{\"moves\":[{entry}],\"trapped\":true}}");
    }
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
                // Resolve a BARE-stored Hidden Power to its TYPED id first
                // (`gen3_own_typed_hp_active_request_v1`, the sibling of the B3 roster
                // fix `gen3_own_typed_hp_request_roster_v1`): the OWNER's own `active[].
                // moves[]` request shows the TYPED name ("Hidden Power Dark 70"), IV/marker-
                // derived — like the roster — while the id stays bare `hiddenpower`. A set
                // that stores the typed name already `to_id`s to `hiddenpowerdark`; this
                // only fires for the bare-stored-plus-marker form (e.g. Charizard's packed
                // `HiddenPower` + `,Dark,,,,`). Owner-only + request-only, so the opponent HP
                // hiding + the omniscient `|move|` bare-collapse (BF1) are untouched.
                let (id, name) = active_move_display(&side_move_id(mon, mv), dex, mon.hidden_power_bp);
                let pp = mon.move_pp.get(k).copied().unwrap_or(0);
                let maxpp = mon.move_maxpp.get(k).copied().unwrap_or(0);
                let base_target = dex
                    .moves(mv)
                    .map(|d| d.target.clone())
                    .unwrap_or_else(|| "normal".to_string());
                // CURSE's `nonGhostTarget` (`gen3_bridge_curse_request_target_v1`): gen-3
                // `curse.onModifyMove` re-targets a NON-GHOST user's Curse to `self` at
                // runtime, and the sim's `getMoveRequestData` reports that RUNTIME target
                // (`"self"`) while the STATIC dex target is `"normal"`. Render the effective
                // target so the request matches the sim — the SAME runtime read the
                // Pressure-PP path uses (`is_nonghost_curse`, `turn/moves.rs`); a GHOST Curse
                // is foe-directed so it keeps the base `normal`. (This retires the per-side
                // fuzzer's `curse-nonghost-target-self-vs-normal` request-DISPLAY allowlist.)
                let target = if crate::dex::to_id(mv) == "curse"
                    && !crate::turn::mon_types(mon, dex).contains(&crate::dex::Type::Ghost)
                {
                    "self".to_string()
                } else {
                    base_target
                };
                let disabled = move_disabled(mon, k, dex);
                // A MIMIC-ACQUIRED SLOT CARRIES **NO** `target` KEY AT ALL
                // (`gen3_mimic_request_no_target_v1`). gen3 inherits gen4's Mimic, and that
                // override's replacement move-slot literal (`data/mods/gen4/moves.ts:868`)
                // OMITS the `target` field the BASE `data/moves.ts` mimic sets — so
                // `getMoveRequestData`'s `let target = moveSlot.target` reads `undefined` and
                // `JSON.stringify` DROPS the key entirely. Byte form (SIM-CONFIRMED on the
                // fuzz repro): `…"pp":5,"maxpp":24,"disabled":false`. Keyed on the overlay's
                // OWN slot, so the mon's other three moves keep their targets, and the key
                // returns as soon as the overlay is restored on switch-out / faint.
                // ...but NOT while TRANSFORMED (`gen3_transform_v1`): `transformInto`'s own
                // moveslot literal DOES set `target: moveSlot.target` (unlike gen4 Mimic's,
                // which omits it), so every copied slot carries its `target` key — probe C:
                // `{"move":"Curse","id":"curse","pp":5,"maxpp":10,"target":"self",...}`. The
                // Mimic record survives underneath the copy (it is restored after it), so this
                // must be gated or a transformed Mimic-carrier would drop one slot's key.
                let mimicked = mon.transform.is_none()
                    && mon.mimic_overlay.as_ref().is_some_and(|ov| ov.slot == k);
                if mimicked {
                    format!(
                        "{{\"move\":\"{}\",\"id\":\"{}\",\"pp\":{},\"maxpp\":{},\"disabled\":{}}}",
                        json_escape(&name),
                        json_escape(&id),
                        pp,
                        maxpp,
                        disabled
                    )
                } else {
                    format!(
                        "{{\"move\":\"{}\",\"id\":\"{}\",\"pp\":{},\"maxpp\":{},\"target\":\"{}\",\"disabled\":{}}}",
                        json_escape(&name),
                        json_escape(&id),
                        pp,
                        maxpp,
                        json_escape(&target),
                        disabled
                    )
                }
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
    run_full_battle_bridge_with_quick_claw(opts, cmds, false, dex)
}

/// [`run_full_battle_bridge`] with the turn-0 Quick Claw roll supplied
/// (`gen3_turn0_quick_claw_capture_v1`); the plain entry passes `false`, so it stays
/// byte-for-byte the pre-fix behaviour.
pub fn run_full_battle_bridge_with_quick_claw(
    opts: &BattleOptions,
    cmds: &[Cmd],
    quick_claw_roll: bool,
    dex: &Dex,
) -> Result<BridgeStreams, String> {
    let (chunks, _ended, _script, _seeds) =
        run_full_battle_bridge_core_with_quick_claw(opts, cmds, quick_claw_roll, dex)?;
    Ok(chunks.flatten())
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
    let (chunks, ended, _script, _seeds) = run_full_battle_bridge_core(opts, cmds, dex)?;
    Ok((chunks, ended))
}

/// The CORE of the bridge emitter: identical to [`run_full_battle_bridge_chunked_ended`]
/// but ALSO returns the [`ScriptDecision`] list it BUILT while consuming the CMD stream
/// at each boundary AND the per-`|request|`-boundary PRNG seed list (the A2 SEED ANCHOR).
///
/// **The A2 seed-anchor fix (`gen3_perside_seed_anchor_makerequest_align_v1`).** The
/// per-side/request A/B fuzzer's SEED ANCHOR (`src/bin/bridge_replay.rs --ab`) asserts the
/// port's per-decision seed against the omniscient oracle's recorded `seedAfter` BEFORE the
/// per-side byte diff — partitioning a divergence into an upstream engine desync
/// (`kind:"seed"`) vs a genuine per-side/request-serializer bug. It used to re-run
/// `run_full_battle` and read each `DecisionRecord.seed_after`, but that internal checkpoint
/// is captured at a point that, on a **phaze-drag / forced-switch** boundary, is ONE endTurn
/// Quick-Claw `randomChance(1,5)` draw EARLIER than the sim's `makeRequest` flush (the
/// fuzzer's `rec.seeds.push` point) — a per-decision-boundary BOOKKEEPING misalignment (the
/// game is byte-identical to `|win|`), not an engine draw bug. This returns instead the seed
/// read at each **`makeRequest` FLUSH boundary** — the `state.prng_seed()` of the freshly
/// replayed `[0..k]` script paused needing decision `k`'s input, i.e. AFTER decision `k-1`'s
/// endTurn Quick Claw, 1:1 with the sim's `rec.seeds.push`. Reading the seed at a different,
/// makeRequest-aligned checkpoint changes NO PRNG call, so `run_full_battle`'s DecisionRecord
/// semantics (asserted by fullbattle/phaze/e2e) + every committed golden are untouched.
pub fn run_full_battle_bridge_core(
    opts: &BattleOptions,
    cmds: &[Cmd],
    dex: &Dex,
) -> Result<(BridgeChunks, bool, Vec<ScriptDecision>, Vec<PrngSeed>), String> {
    run_full_battle_bridge_core_with_quick_claw(opts, cmds, false, dex)
}

/// [`run_full_battle_bridge_core`] with the turn-0 Quick Claw roll supplied
/// (`gen3_turn0_quick_claw_capture_v1`).
///
/// The plain entry passes `false` — byte-for-byte the pre-fix behaviour, so every existing
/// caller (including the LIVE `sim_bridge`, which models the real construction window via
/// [`BridgeSession::new_construct_turn0`] and so never needs this) is unaffected. The OFFLINE
/// bridge gate (`bin/bridge_replay`) passes the golden's captured `quick_claw_roll`, because
/// its `opts.seed` is the POST-construction seed and the turn-0 `endTurn` roll it skips is
/// otherwise unrecoverable — see [`GoldenBattle::quick_claw_roll`].
pub fn run_full_battle_bridge_core_with_quick_claw(
    opts: &BattleOptions,
    cmds: &[Cmd],
    quick_claw_roll: bool,
    dex: &Dex,
) -> Result<(BridgeChunks, bool, Vec<ScriptDecision>, Vec<PrngSeed>), String> {
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
        st.quick_claw_roll = quick_claw_roll;
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
    // A2 SEED ANCHOR: the port's PRNG seed at each `makeRequest` FLUSH boundary. Captured at
    // the TOP of each iteration whose `script` already holds ≥1 committed decision — that
    // paused-state seed is the seed AFTER the last committed decision's endTurn Quick Claw
    // (the makeRequest boundary), i.e. `rec.seeds[committed - 1]`, 1:1 with the sim's
    // `rec.seeds.push`. Iteration 0 (empty script) is the pre-first-decision `initSeed` and
    // is NOT a `rec.seed`, so it is skipped.
    let mut request_seeds: Vec<PrngSeed> = Vec::new();
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
        let (ended, log, state_holder) = replay_snapshot(opts, &script, quick_claw_roll, dex)?;
        let state = state_holder.state().ok_or("paused battle has no state")?;

        // [A2] Capture the makeRequest-boundary seed (post-Quick-Claw of the prior decision).
        if !script.is_empty() {
            request_seeds.push(state.prng_seed());
        }

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
        let _ = emit_boundary_request_chunks(&mut chunks, state, &kinds, no_cancel_forced, dex);

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
                Some(c) => *c,
                None => {
                    plan_ended = true;
                    break;
                }
            };
            let s = cmd.side;
            if !need[s] {
                // The recorded CMD stream answers a side this boundary does NOT request — the
                // REPLAYED game reached a different request state than the RECORDED game, i.e.
                // the port's engine desynced UPSTREAM (an extra/missing PRNG draw shifted a
                // faint / forced-switch onto a side the sim didn't have here). This is a
                // SYMPTOM of a draw-count divergence, NOT a per-side/request serializer bug.
                // Stop gracefully (R20): return the partial streams + `request_seeds` so
                // `ab_verdict`'s SEED ANCHOR classifies it as `kind:"seed"` (upstream engine →
                // the omniscient fix-queue) — the honest classification — instead of the driver
                // crashing with an opaque "unexpected CMD" error that masked the seed anchor.
                // The seeds already carry the divergence (captured at earlier boundaries), so
                // the anchor reports it at the exact desync decision; if seeds happen to still
                // align, the truncated per-side byte diff catches it as a non-allowlisted
                // divergence — either way the gate stays red, nothing is swallowed.
                plan_ended = true;
                break;
            }
            // Resolve the wire token (numeric slot OR a NAME — `move earthquake` /
            // `switch Salamence`, the live RL runtime's form) against THIS boundary's
            // state, exactly like Showdown's `side.chooseMove`/`chooseSwitch`. An
            // unresolvable NAME (illegal — no matching move slot / bench species) maps to
            // an OUT-OF-RANGE numeric slot so the engine's existing reject-and-re-request
            // gate (`choice_is_legal`) handles it IDENTICALLY to a numeric out-of-range.
            let resolved = resolve_choice(state, s, &cmd.choice).unwrap_or_else(|| match cmd.choice {
                WireChoice::Switch(_) | WireChoice::SwitchSpecies(_) => {
                    Choice::Switch(state.sides[s].pokemon.len())
                }
                _ => Choice::Move(state.sides[s].pokemon[state.sides[s].active].set.moves.len()),
            });
            cmd_iter.next();
            // Trapped-switch rejection at a MOVE boundary (see the flat driver's docs +
            // `gen3_shadowtag_firm_trap_v1`). The `|error|` and the re-request are SEPARATE
            // chunks (the sim flushes each on the rejected write / the update).
            // A MOVE-LOCKED mon (mustrecharge / charging, `gen3_move_coverage_batch4c_v1`)
            // rejects a switch with the FIRM `[Invalid choice]` form + NO re-request
            // (probed — the request already showed `trapped:true`), like Shadow Tag.
            let locked = state.sides[s].pokemon[state.sides[s].active].move_locked();
            if kinds[s] == SideRequest::Move
                && matches!(resolved, Choice::Switch(_))
                && ((state.is_trapped(s, dex) && has_live_bench(state, s)) || locked)
            {
                if locked || state.trap_is_firm(s, dex) {
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
            got[s] = Some(resolved);
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
                // A MOVE-LOCKED mon's single-entry request never Struggle-substitutes
                // (`gen3_move_coverage_batch4c_v1` — the locked move/recharge is offered
                // regardless of PP).
                if !mon.move_locked() && mon.must_struggle(dex) {
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

    Ok((chunks, battle_ended, script, request_seeds))
}

// ===========================================================================
// The INCREMENTAL per-side bridge (`gen3_bridge_incremental_replay_v1`) — a
// PERSISTENT live-battle session that advances ONE request boundary per fed CMD,
// O(1) per input, NEVER re-simulating a prior turn. This is the production path
// (`sim_bridge` holds one across CHOOSE lines); [`run_full_battle_bridge_core`]
// (above) stays the genesis-replay REFERENCE ORACLE the parity test checks against.
// ===========================================================================

/// The mid-boundary progress a [`BridgeSession`] persists across CMD feeds (a `move`
/// request needs BOTH sides' choices, possibly arriving on separate CHOOSE lines; a
/// trapped reject holds the boundary open).
#[derive(Clone)]
struct BoundaryProgress {
    kinds: [SideRequest; 2],
    got: [Option<Choice>; 2],
    need: [bool; 2],
}

/// The agent-visible request surface at an open boundary — the PUBLIC mirror of the
/// private [`SideRequest`], using Showdown's `side.requestState` vocabulary
/// (`'move'` / `'switch'` / `'wait'`).
///
/// A clone-and-branch search kernel branches on exactly `requestState === 'move'`, so
/// the mapping is the load-bearing part: a FORCED replacement is `Switch` (Showdown
/// reports `'switch'` for `forceSwitch`, not some third kind), and a side that is not
/// being asked this boundary is `Wait`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// A top-of-turn `move` request: the side picks a move OR a voluntary switch.
    Move,
    /// A FORCED replacement (`forceSwitch`) after a faint: the side must switch.
    Switch,
    /// This side is not being asked at this boundary (`{"wait":true}`).
    Wait,
}

impl From<SideRequest> for RequestState {
    fn from(r: SideRequest) -> Self {
        match r {
            SideRequest::Move => RequestState::Move,
            SideRequest::ForceSwitch => RequestState::Switch,
            SideRequest::Wait => RequestState::Wait,
        }
    }
}

/// A PERSISTENT per-side bridge session over a LIVE battle. It owns the same
/// `Battle` + [`FullBattleDriver`] stepping primitive [`BattleStream`] uses (the ONE
/// turn-loop), plus the per-side chunk/log/seed bookkeeping the genesis-replay core
/// kept in locals. `feed_cmd` appends one CMD and advances the driver to the next
/// request boundary, emitting ONLY the new chunk — so a battle costs O(N), not O(N²)
/// (the wedge fix): `sim_bridge` builds ONE session per battle and feeds each CHOOSE
/// into it, instead of re-running the whole accumulated stream per CHOOSE.
///
/// **Plain data all the way down — which is what makes [`BridgeSession::snapshot`]
/// work** (`gen3_bridge_clone_branch_v1`): no `Rc`/`RefCell`/`Box<dyn>`/closures/raw
/// pointers/lifetimes anywhere beneath this type, so the derived `Clone` is a DEEP,
/// fully independent paused session — the clone-and-branch primitive a search driver
/// builds a tree from. Shares the caller's `&Dex` (a `Dex` owns ~16 MB of parsed data —
/// a per-battle load would be a real regression), so the bridge threads it per method,
/// unlike the writeline `BattleStream` which owns its Dex for the standalone `new()`
/// surface; a snapshot therefore costs the BATTLE, not the dex.
#[derive(Clone)]
pub struct BridgeSession {
    battle: Battle,
    driver: FullBattleDriver,
    report_percent: bool,
    chunks: BridgeChunks,
    /// The A2 seed anchor — the makeRequest-boundary seed per committed decision.
    request_seeds: Vec<PrngSeed>,
    /// The committed decisions, in order (parity with the genesis core's `script`).
    script: Vec<ScriptDecision>,
    /// Cursor into the LIVE battle log (`prev_log_len` in the genesis core).
    prev_log_len: usize,
    /// The open boundary's per-side progress, or `None` between boundaries.
    boundary: Option<BoundaryProgress>,
    /// The `|request|` line CURRENTLY OUTSTANDING to each side — the port of Showdown's
    /// `side.activeRequest` (`gen3_bridge_clone_branch_v1`). Written wherever a request is
    /// emitted into `chunks` (the boundary open, AND the re-request a hidden-trap reject
    /// re-issues), cleared when the boundary closes / the battle ends. Stored rather than
    /// rebuilt on demand so [`BridgeSession::active_request_json`] returns the EXACT bytes
    /// the wire carried — rebuilding could drift from what was sent (the `update`/`noCancel`
    /// trailing keys depend on how the boundary was reached, not on the current board).
    active_request: [Option<String>; 2],
    /// Unconsumed CMDs (fed but not yet answering a boundary — e.g. a partial
    /// double-replacement, or cmds queued ahead of the boundary that needs them).
    cmd_buf: std::collections::VecDeque<Cmd>,
    /// The battle reached its natural WIN/LOSS/TIE end.
    ended: bool,
    /// The winner of a [`BridgeSession::forfeit`], if one happened. A forfeit ends the
    /// battle through the PROTOCOL (it writes the `|win|` pair directly) rather than the
    /// driver's own win check, so the driver's phase never becomes `Ended` and
    /// [`BridgeSession::winner`] would otherwise report `None` for a battle that plainly
    /// has a winner.
    forfeit_winner: Option<usize>,
    /// An upstream-desync graceful stop (the R20 `!need[s]` case) — no further advance.
    stopped: bool,
    /// A FATAL, non-recoverable condition the live bridge must report as `__ERR__` rather than
    /// silently spin on (`gen3_bridge_unresolvable_choice_failloud_v1`). Today: a NAME-form wire
    /// choice that resolves against nothing, or a boundary that keeps rejecting.
    fatal: Option<String>,
    /// Consecutive REJECTS at the CURRENT boundary (reset whenever a decision commits). A reject
    /// re-issues the same request; if the client re-sends a choice that is rejected the same way,
    /// nothing advances — see `REJECT_STREAK_CAP`.
    reject_streak: u32,
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

impl BridgeSession {
    /// Build a session: construct the live battle, emit + chunk the framing, and advance
    /// to the FIRST request boundary (emitting its request), paused for the first CMD.
    ///
    /// The `opts.seed` is the sim's PRE-FIRST-DECISION (post-construction) seed — the
    /// convention the committed goldens capture. For the LIVE bridge (`sim_bridge`),
    /// which is fed poke-env's RAW `>start` seed, use [`new_construct_turn0`] instead.
    pub fn new(opts: &BattleOptions, dex: &Dex) -> Result<BridgeSession, String> {
        Self::new_with_quick_claw(opts, false, dex)
    }

    /// [`BridgeSession::new`] with the turn-0 Quick Claw roll supplied
    /// (`gen3_turn0_quick_claw_capture_v1`) — the post-construction-seed convention skips the
    /// turn-0 `endTurn` that decides turn 1's roll, and the seed alone cannot carry it. `new`
    /// passes `false` (the pre-fix behaviour), so every existing caller is byte-unaffected.
    pub fn new_with_quick_claw(
        opts: &BattleOptions,
        quick_claw_roll: bool,
        dex: &Dex,
    ) -> Result<BridgeSession, String> {
        let mut battle = Battle::start_with_switchins(opts, dex)?;
        battle.state_mut().ok_or("no state")?.quick_claw_roll = quick_claw_roll;
        Self::new_from_battle(battle, opts, dex)
    }

    /// Build a session from a RAW `>start` seed, running the FULL turn-0 CONSTRUCTION
    /// WINDOW (`gen3_turn0_construction_v1`, [`Battle::start_with_turn0_construction`])
    /// so the live incremental engine lines up with the real sim's post-`>start` PRNG
    /// state on a speed-TIED lead or an unspecified-gender mon. This is the drop-in the
    /// `sim_bridge` binary uses (poke-env sends the raw seed); it REPLACES the old pure
    /// `advance_seed_for_construction` seed hack.
    pub fn new_construct_turn0(opts: &BattleOptions, dex: &Dex) -> Result<BridgeSession, String> {
        Self::new_from_battle(Battle::start_with_turn0_construction(opts, dex)?, opts, dex)
    }

    /// The shared session body: emit + chunk the framing from an already-constructed
    /// battle, then advance to the first request boundary.
    fn new_from_battle(mut battle: Battle, opts: &BattleOptions, dex: &Dex) -> Result<BridgeSession, String> {
        // Percent HP fold applies only in non-debug formats (gen3ou); a `debug:true`
        // format (gen3customgame) sets `reportExactHP` → both sides see exact HP.
        let report_percent = !format_is_debug(&opts.format_id);
        // Emit the framing INTO the live log (kept, not drained — continuous cursor
        // coords), exactly the lines `run_full_battle_logged(&[])` produces.
        let raw_framing: Vec<crate::protocol::ProtocolLine> = {
            let bs = battle.state_mut().ok_or("no state")?;
            bs.log.enable();
            bs.emit_framing(dex);
            bs.log.lines().to_vec()
        };
        let framing = reframe(&raw_framing, &opts.format_id);
        let mut chunks = BridgeChunks::default();
        emit_framing_chunks(&mut chunks, &framing, report_percent);
        let prev_log_len = raw_framing.len();
        let mut sess = BridgeSession {
            battle,
            driver: FullBattleDriver::new(),
            report_percent,
            chunks,
            request_seeds: Vec::new(),
            script: Vec::new(),
            prev_log_len,
            boundary: None,
            active_request: [None, None],
            cmd_buf: std::collections::VecDeque::new(),
            ended: false,
            forfeit_winner: None,
            stopped: false,
            fatal: None,
            reject_streak: 0,
        };
        sess.advance(dex);
        Ok(sess)
    }

    /// Feed ONE command (the `sim_bridge` per-CHOOSE entry) and advance. O(1) amortized.
    pub fn feed_cmd(&mut self, cmd: Cmd, dex: &Dex) {
        self.cmd_buf.push_back(cmd);
        self.advance(dex);
    }

    /// Feed a batch of commands and advance (the single-call oracle/parity path).
    pub fn feed_cmds(&mut self, cmds: &[Cmd], dex: &Dex) {
        for c in cmds {
            self.cmd_buf.push_back(c.clone());
        }
        self.advance(dex);
    }

    /// Forfeit `side` — the poke-env `/forfeit` → `FORCELOSE <side>` path.
    ///
    /// The Node bridge writes `>forcelose <side>` INTO the sim, so Showdown runs a real
    /// `win(otherSide)`: BOTH players receive the deciding `|` + `|win|<name>` batch before the
    /// streams close. Emitting only `__END__` (what this used to do) leaves poke-env's `Battle`
    /// with `finished == False` forever — the env's next `reset()` then waits on a battle result
    /// that can never arrive, which is a HANG, not an error. The training seam forfeits whenever
    /// `reset()` lands mid-battle, so that hang is reachable on any episode boundary.
    ///
    /// Mirrors the natural-end path exactly (`turn::driver`'s `TurnLoopStop::Ended` arm writes the
    /// same `separator()` + `win()` pair into the battle log, and `advance` flushes the delta as
    /// ONE chunk per side) so the wire bytes are byte-for-byte what a real win produces.
    pub fn forfeit(&mut self, side: usize) {
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
        // Flush every log line past the cursor (the win pair, plus any residual the open boundary
        // had not flushed yet) as ONE chunk per side — the natural-end emission.
        let (delta, new_len) = {
            let bs = self.battle.state().expect("state");
            let lines = bs.log.lines();
            (lines[self.prev_log_len..].to_vec(), lines.len())
        };
        emit_log_batch_chunk(&mut self.chunks, &delta, self.report_percent);
        self.prev_log_len = new_len;
        self.ended = true;
        self.forfeit_winner = Some(winner);
        // The battle is over — whatever request was open is void (mirrors the natural-end
        // clear in `advance`).
        self.boundary = None;
        self.active_request = [None, None];
    }

    /// All per-side chunks emitted so far (the `sim_bridge` cursor reads `[emitted..]`).
    pub fn chunks(&self) -> &BridgeChunks {
        &self.chunks
    }

    /// Whether this session's FORMAT hides exact HP from the non-owner (gen3ou's "HP
    /// Percentage Mod"). `false` for a `debug:true` format (gen3customgame sets
    /// `reportExactHP`). Read-only; exposed for [`split_log_lines`], whose shared-form
    /// rendering must agree with the per-side fold.
    pub fn report_percent(&self) -> bool {
        self.report_percent
    }

    /// The battle reached game-end.
    pub fn is_ended(&self) -> bool {
        self.ended
    }

    /// A FATAL condition the caller must surface as `__ERR__` (never spin on). See `fatal`.
    pub fn fatal(&self) -> Option<&str> {
        self.fatal.as_deref()
    }

    /// The A2 seed-anchor list (parity with the genesis core).
    /// The battle's CURRENT turn — the gate for the counterfactual `resumeReseed`
    /// (`gen3_bridge_resume_reseed_v1`). 0 before the battle is built.
    pub fn turn(&self) -> u32 {
        self.battle.state().map(|st| st.turn).unwrap_or(0)
    }

    /// Swap the live battle's PRNG mid-battle (`gen3_bridge_resume_reseed_v1`) — the Rust
    /// equivalent of the node bridge's `rawStream.battle.prng = new PRNG(seed)`.
    ///
    /// This is the counterfactual Monte-Carlo hook: `local_battle_runner` re-runs a recorded
    /// battle with `{"resumeReseed": {turn, seed}}` so the PREFIX keeps its recorded dice and
    /// only the post-divergence resolution draws from a fresh stream — which is what makes a
    /// re-rolled win-probability an estimate of THAT board rather than a different game.
    /// Applied at the START of the divergence turn, BEFORE that turn's choices commit, exactly
    /// once (the caller owns the once-ness, mirroring the node bridge's `reseeded` latch).
    ///
    /// Note this is deliberately NOT `Battle::reseed` (still `todo!()`, alongside
    /// `serialize`/`deserialize` for the clone-and-branch search path): the incremental
    /// session owns a live `BattleState` whose `prng` is a plain field, so the bridge's
    /// counterfactual need does not have to wait on the full snapshot surface.
    pub fn reseed(&mut self, seed: &str) {
        if let Some(st) = self.battle.state_mut() {
            st.prng = crate::prng::Prng::new(seed);
        }
    }

    pub fn request_seeds(&self) -> &[PrngSeed] {
        &self.request_seeds
    }

    /// The committed decisions (parity with the genesis core).
    pub fn script(&self) -> &[ScriptDecision] {
        &self.script
    }

    // =======================================================================
    // The CLONE-AND-BRANCH SEARCH API (`gen3_bridge_clone_branch_v1`).
    //
    // The Node search server clones a paused battle with `State.serializeBattle`
    // and branches it; this is the Rust equivalent, minus the byte format (see
    // [`crate::battle::Battle`] for why none is needed). Everything here is a READ
    // or a snapshot — no method below advances the battle or draws a single number,
    // so wiring a search driver on top cannot perturb the production bridge.
    // =======================================================================

    /// A deep, independent clone of this paused session — **the clone-and-branch
    /// primitive**.
    ///
    /// The clone shares NOTHING with its parent: battle state, PRNG, driver phase,
    /// chunk stream, log cursor and open-boundary progress are all copied, so advancing
    /// either one cannot affect the other. It resumes from exactly this pause point and
    /// rolls the dice the parent would have rolled, which is what makes a branch a
    /// counterfactual of THIS board rather than a different game.
    ///
    /// Cost is the BATTLE, not the dex: the `&Dex` is threaded per method, never owned
    /// here, so a snapshot copies a couple of teams' worth of state rather than ~16 MB.
    ///
    /// A search driver will typically pair this with [`BridgeSession::clear_chunks`],
    /// so the branch's `chunks()` contains only what the branch itself emitted.
    pub fn snapshot(&self) -> BridgeSession {
        self.clone()
    }

    /// Drop every chunk emitted so far and reset the chunk stream, so a resumed clone's
    /// [`BridgeSession::chunks`] contains ONLY what it emits from here — the per-ply
    /// SUFFIX a search driver returns for one branch.
    ///
    /// Deliberately does NOT touch `prev_log_len`: that is a cursor into the BATTLE LOG
    /// (which lines have been folded into chunks yet), not into the chunk stream. Resetting
    /// it would re-emit the whole battle's log into the next chunk. Nor does it touch the
    /// battle, the driver, the open boundary, or `request_seeds`/`script`.
    pub fn clear_chunks(&mut self) {
        self.chunks = BridgeChunks::default();
    }

    /// The OPEN request kind for `side` at the current paused boundary, or `None` when no
    /// boundary is open (the battle ended, was forfeited, or the session stopped on an
    /// upstream desync).
    ///
    /// Mirrors Showdown's `side.requestState`: a forced replacement reports
    /// [`RequestState::Switch`], a top-of-turn request [`RequestState::Move`], and a side
    /// this boundary does not ask reports [`RequestState::Wait`].
    pub fn request_kind(&self, side: usize) -> Option<RequestState> {
        self.boundary.as_ref().map(|bp| RequestState::from(bp.kinds[side]))
    }

    /// Whether `side` has already supplied an ACCEPTED choice for the open boundary — the
    /// equivalent of Showdown's `side.isChoiceDone()`.
    ///
    /// `true` also when the side is not needed at this boundary (a `Wait` request) and when
    /// no boundary is open at all: in both cases the boundary is not waiting on this side.
    /// `false` ONLY while the boundary genuinely wants a choice from it — which includes the
    /// state right after a REJECTED choice, since a reject re-issues the request and leaves
    /// the boundary open.
    pub fn is_choice_done(&self, side: usize) -> bool {
        self.boundary.as_ref().map(|bp| !bp.need[side]).unwrap_or(true)
    }

    /// The EXACT `|request|` line last issued to `side` (the bytes the wire carried), or
    /// `None` when nothing is outstanding (between boundaries / after game-end).
    ///
    /// Byte-identical to what Showdown's `side.activeRequest` serializes to, because it IS
    /// the string that was pushed into the chunk stream — including the `"update":true`
    /// re-request a hidden-trap REJECT re-issues, which is the frame a search kernel must
    /// re-pick from after a rejected switch.
    pub fn active_request_json(&self, side: usize) -> Option<&str> {
        self.active_request[side].as_deref()
    }

    /// Read access to the live battle state — the omniscient referee readout a search
    /// driver scores a branch with (HP, status, boosts, field, the PRNG's current seed).
    /// `None` only if the battle was never constructed, which cannot happen for a session
    /// built by any of the `new*` constructors.
    pub fn battle_state(&self) -> Option<&BattleState> {
        self.battle.state()
    }

    /// The WINNER's side index (0 = p1, 1 = p2) once the battle has ended, else `None`.
    ///
    /// Covers both end paths: the driver's own win check (a side out of mons) and a
    /// [`BridgeSession::forfeit`], which ends the battle through the protocol instead.
    /// `None` therefore means EITHER "still playing" OR a gen-3 double-faint TIE — pair it
    /// with [`BridgeSession::is_ended`] to tell those apart.
    pub fn winner(&self) -> Option<usize> {
        self.driver.winner().or(self.forfeit_winner)
    }

    /// Advance from the current paused state as far as the buffered CMDs allow: at each
    /// request boundary emit the log delta + request, consume CMD(s), emit the struggle
    /// line, feed ONE decision to the driver, repeat — pausing when the CMD buffer runs
    /// out mid-boundary. Byte-identical (chunks + seeds + script) to the genesis-replay
    /// core fed the same CMD stream (asserted by the parity test); the incremental engine
    /// draws the SAME PRNG numbers a genesis replay would, by construction.
    fn advance(&mut self, dex: &Dex) {
        // Defensive spin guard (the driver's `BATTLE_TURN_CAP`/`turn_loop` watchdogs are the
        // real runaway protection; this only catches a logic bug in THIS loop).
        let mut guard: u64 = 0;
        loop {
            guard += 1;
            if guard > 100_000_000 {
                panic!("BridgeSession::advance spin guard exceeded (a boundary never resolved)");
            }
            if self.stopped || self.ended {
                return;
            }
            // ── Start a new boundary if not mid-boundary. ──
            if self.boundary.is_none() {
                // [A2] makeRequest-boundary seed (skip the pre-first-decision boundary).
                if !self.script.is_empty() {
                    let seed = self.battle.state().expect("state").prng_seed();
                    self.request_seeds.push(seed);
                }
                // The log delta flushed since the previous boundary → ONE chunk per side.
                let (delta, new_len) = {
                    let bs = self.battle.state().expect("state");
                    let lines = bs.log.lines();
                    (lines[self.prev_log_len..].to_vec(), lines.len())
                };
                emit_log_batch_chunk(&mut self.chunks, &delta, self.report_percent);
                self.prev_log_len = new_len;
                if self.driver.is_ended() {
                    self.ended = true;
                    // Nothing is outstanding once the battle is over (`side.activeRequest`
                    // is cleared at the sim's `win`), so the search driver's
                    // `active_request_json` reads None rather than a stale frame.
                    self.active_request = [None, None];
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
                {
                    let bs = self.battle.state().expect("state");
                    let issued =
                        emit_boundary_request_chunks(&mut self.chunks, bs, &kinds, no_cancel_forced, dex);
                    // Record what is now OUTSTANDING per side (incl. the `{"wait":true}`
                    // frame — the sim issues that too).
                    let [r0, r1] = issued;
                    self.active_request = [Some(r0), Some(r1)];
                }
                self.boundary = Some(BoundaryProgress {
                    kinds,
                    got: [None, None],
                    need: [kinds[0] != SideRequest::Wait, kinds[1] != SideRequest::Wait],
                });
            }

            // ── Consume CMD(s) answering this boundary. ──
            loop {
                let (need0, need1) = {
                    let bp = self.boundary.as_ref().expect("boundary");
                    (bp.need[0], bp.need[1])
                };
                if !need0 && !need1 {
                    break; // boundary satisfied
                }
                let cmd = match self.cmd_buf.front() {
                    Some(c) => c.clone(),
                    None => return, // PAUSE — out of CMDs; resume on the next fed CMD
                };
                let s = cmd.side;
                if !self.boundary.as_ref().expect("boundary").need[s] {
                    // A CMD for a side this boundary does NOT request — an UPSTREAM engine
                    // desync (an extra/missing draw shifted a faint/forced-switch onto a side
                    // the recorded game did not have here). Stop gracefully (R20) so the seed
                    // anchor classifies it, instead of crashing.
                    self.stopped = true;
                    // ...but for the LIVE bridge, "stop gracefully" means EMIT NOTHING, EVER —
                    // the child keeps accepting CHOOSE lines and never answers, so poke-env waits
                    // on a battle message that can never arrive and the env's `step()` hangs until
                    // its 120s watchdog kills the whole run. Record it as FATAL so `sim_bridge`
                    // reports `__ERR__` (`gen3_bridge_stopped_failloud_v1`). The OFFLINE replay
                    // harnesses read `stopped`/the streams and ignore `fatal`, so their
                    // graceful-classification behaviour is unchanged.
                    let kinds = self.boundary.as_ref().expect("boundary").kinds;
                    self.fatal = Some(format!(
                        "upstream desync: CHOOSE for p{} but this boundary does not request it \
                         (kinds p1={:?} p2={:?}, needs p1={} p2={}). The bridge would go silent \
                         forever, so this fails loud.",
                        s + 1,
                        kinds[0],
                        kinds[1],
                        self.boundary.as_ref().expect("boundary").need[0],
                        self.boundary.as_ref().expect("boundary").need[1],
                    ));
                    return;
                }
                // Resolve the wire token (numeric slot OR a NAME) against THIS boundary's
                // state, exactly like Showdown's `side.chooseMove`/`chooseSwitch`.
                //
                // `gen3_bridge_unresolvable_choice_failloud_v1` — a NAME form that resolves
                // against NOTHING is FATAL, not a reject. The out-of-range fallback below makes
                // `choice_is_legal` reject the choice, which leaves the boundary open and
                // re-issues the SAME request; poke-env then re-sends the SAME unmappable name,
                // forever — an unbounded bridge↔Python spin that pins a core and floods stdout
                // (measured: 46 MB/s from one child) while the env's `step()` never returns.
                // The Struggle wedge was ONE instance of this class; rather than wait for the
                // next one, report it. A NUMERIC out-of-range choice is DIFFERENT: it is a
                // legitimate, tested reject-and-re-request (the forced-replacement resume gate
                // and the 0-PP gate both rely on it), and poke-env re-picks from a fresh request,
                // so it cannot spin — that path is untouched.
                if matches!(cmd.choice, WireChoice::MoveName(_) | WireChoice::SwitchSpecies(_)) {
                    let unresolvable = {
                        let bs = self.battle.state().expect("state");
                        resolve_choice(bs, s, &cmd.choice).is_none()
                    };
                    if unresolvable {
                        let bs = self.battle.state().expect("state");
                        let mon = &bs.sides[s].pokemon[bs.sides[s].active];
                        self.fatal = Some(format!(
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
                        ));
                        self.stopped = true;
                        return;
                    }
                }
                let resolved = {
                    let bs = self.battle.state().expect("state");
                    resolve_choice(bs, s, &cmd.choice).unwrap_or_else(|| match cmd.choice {
                        WireChoice::Switch(_) | WireChoice::SwitchSpecies(_) => {
                            Choice::Switch(bs.sides[s].pokemon.len())
                        }
                        _ => Choice::Move(bs.sides[s].pokemon[bs.sides[s].active].set.moves.len()),
                    })
                };
                self.cmd_buf.pop_front();
                // Trapped-switch rejection at a MOVE boundary — the `|error|` + (hidden trap)
                // the re-request are SEPARATE chunks; the side still needs a choice.
                let (reject, firm) = {
                    let bs = self.battle.state().expect("state");
                    let kind_s = self.boundary.as_ref().expect("boundary").kinds[s];
                    let locked = bs.sides[s].pokemon[bs.sides[s].active].move_locked();
                    let reject = kind_s == SideRequest::Move
                        && matches!(resolved, Choice::Switch(_))
                        && ((bs.is_trapped(s, dex) && has_live_bench(bs, s)) || locked);
                    let firm = locked || bs.trap_is_firm(s, dex);
                    (reject, firm)
                };
                if reject {
                    // Bound the reject↔re-request exchange: a deterministic client re-sends the
                    // same rejected choice forever (`REJECT_STREAK_CAP`).
                    self.reject_streak += 1;
                    if self.reject_streak > REJECT_STREAK_CAP {
                        let bs = self.battle.state().expect("state");
                        let mon = &bs.sides[s].pokemon[bs.sides[s].active];
                        self.fatal = Some(format!(
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
                        ));
                        self.stopped = true;
                        return;
                    }
                    if firm {
                        self.chunks.push_chunk(
                            s,
                            vec!["|error|[Invalid choice] Can't switch: The active Pokémon is trapped"
                                .to_string()],
                        );
                    } else {
                        self.chunks.push_chunk(
                            s,
                            vec!["|error|[Unavailable choice] Can't switch: The active Pokémon is trapped"
                                .to_string()],
                        );
                        let rereq = {
                            let bs = self.battle.state().expect("state");
                            build_request(bs, s, SideRequest::Move, true, true, false, dex)
                        };
                        self.chunks.push_chunk(s, vec![rereq.clone()]);
                        // The RE-ISSUED frame supersedes the one this reject answered: the
                        // boundary is still open and the side must pick again FROM THIS
                        // request (it now carries `trapped:true` + `"update":true`).
                        self.active_request[s] = Some(rereq);
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
            // The boundary is CLOSING, so nothing is outstanding until the next one opens
            // (which re-sets both entries a few lines below, via the loop's top).
            self.active_request = [None, None];
            {
                let bs = self.battle.state().expect("state");
                for s in 0..2 {
                    if matches!(bp.got[s], Some(Choice::Move(_))) {
                        let mon = &bs.sides[s].pokemon[bs.sides[s].active];
                        if !mon.move_locked() && mon.must_struggle(dex) {
                            let name = display_name(mon, dex);
                            self.chunks.push_chunk(
                                s,
                                vec![format!("|-activate|p{}a: {}|move: Struggle", s + 1, name)],
                            );
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
            self.script.push(dec);
            {
                let driver = &mut self.driver;
                let bs = self.battle.state_mut().expect("state");
                driver.feed(bs, dec, dex);
            }
            // loop back → start a new boundary
        }
    }
}

/// The INCREMENTAL sibling of [`run_full_battle_bridge_core`]: drive the whole `cmds`
/// stream through a single [`BridgeSession`] (O(N)) and return the SAME 4-tuple. The
/// parity test asserts this equals the genesis-replay core bit-for-bit.
pub fn run_full_battle_bridge_incremental(
    opts: &BattleOptions,
    cmds: &[Cmd],
    dex: &Dex,
) -> Result<(BridgeChunks, bool, Vec<ScriptDecision>, Vec<PrngSeed>), String> {
    run_full_battle_bridge_incremental_with_quick_claw(opts, cmds, false, dex)
}

/// [`run_full_battle_bridge_incremental`] with the turn-0 Quick Claw roll supplied — the
/// incremental half of `gen3_turn0_quick_claw_capture_v1`, kept in lockstep with
/// [`run_full_battle_bridge_core_with_quick_claw`] so the genesis-vs-incremental parity
/// assertion holds for BOTH values of the flag, not just the `false` default.
pub fn run_full_battle_bridge_incremental_with_quick_claw(
    opts: &BattleOptions,
    cmds: &[Cmd],
    quick_claw_roll: bool,
    dex: &Dex,
) -> Result<(BridgeChunks, bool, Vec<ScriptDecision>, Vec<PrngSeed>), String> {
    let mut sess = BridgeSession::new_with_quick_claw(opts, quick_claw_roll, dex)?;
    sess.feed_cmds(cmds, dex);
    Ok((sess.chunks, sess.ended, sess.script, sess.request_seeds))
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

/// Re-render an omniscient log slice in Showdown's `battle.log` SHAPE — the
/// `|split|pN` / secret / shared triples an `addSplit` line occupies there
/// (`gen3_rust_replay_driver_v1`).
///
/// The port keeps ONE omniscient line per event and derives each side's view at fold time
/// ([`derive_side`]); Showdown instead stores BOTH forms inline, because `Battle.add` sees a
/// FUNCTION argument (`pokemon.getHealth`, which returns `{side, secret, shared}`) and routes
/// to `addSplit`, pushing three entries:
///
/// ```text
/// |split|p2
/// |-damage|p2a: Metagross|133/328     <- secret (the owner's exact HP)
/// |-damage|p2a: Metagross|41/100      <- shared (what the other side sees)
/// ```
///
/// This is the ONLY place the two representations have to be reconciled, and it exists for
/// the replay family's `turn_log` field, which is defined as `battle.log.slice(...)`.
///
/// The classification is exactly the predicate set [`derive_side`] already uses, so the two
/// can never disagree about WHICH lines are split:
///
/// * **HP-bearing** (`switch` / `drag` / `-damage` / `-heal` / `-sethp` with a `pNa:` ident) —
///   always a triple. `shared == secret` in a `reportExactHP` format (`report_percent ==
///   false`) and for `0 fnt`, matching `getHealth`'s own branches; the triple is emitted
///   either way, because the split happens at `add` time regardless of what the two forms
///   contain.
/// * **Owner-only** (`|-ability|…|[silent]` — gen-3 Pressure's explicit `addSplit`, and the
///   Intimidate no-activate `|-hint|`) — a triple whose shared form is the EMPTY string,
///   which is what `addSplit(side, secret)` with no `shared` pushes.
/// * everything else broadcasts as a single entry.
///
/// The `|-hint|` owner is attributed the same way [`emit_log_batch_chunk`] attributes it (the
/// gen-3 hint form carries no `pNa:` prefix, so it tracks the immediately-preceding
/// `|switch|`/`|drag|` owner). A hint with no preceding switch IN THIS SLICE is left
/// un-split — the honest fallback, since the owner is genuinely unknown from the slice alone.
pub fn split_log_lines(lines: &[crate::protocol::ProtocolLine], report_percent: bool) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut last_switch_side: Option<usize> = None;
    for line in lines {
        let l = &line.0;
        if l.starts_with("|switch|") || l.starts_with("|drag|") {
            last_switch_side = ident_owner(l);
        }
        // Owner-only pair 1: the Pressure `addSplit` reveal.
        if l.starts_with("|-ability|") && l.ends_with("|[silent]") {
            if let Some(owner) = ident_owner(l) {
                out.push(format!("|split|p{}", owner + 1));
                out.push(l.clone());
                out.push(String::new());
                continue;
            }
        }
        // Owner-only pair 2: the Intimidate no-activate hint (owner from the driver's
        // preceding switch, mirroring `emit_log_batch_chunk`).
        if l.starts_with("|-hint|") && l.contains("Intimidate does not activate") {
            if let Some(owner) = last_switch_side {
                out.push(format!("|split|p{}", owner + 1));
                out.push(l.clone());
                out.push(String::new());
                continue;
            }
        }
        // HP-bearing: the secret line is what the port already holds; the shared line is the
        // NON-owner fold (identical to the secret in a debug format / for `0 fnt`).
        if let Some(owner) = hp_line_owner(l) {
            let shared = if report_percent {
                fold_hp_line(l, 1 - owner).unwrap_or_else(|| l.clone())
            } else {
                l.clone()
            };
            out.push(format!("|split|p{}", owner + 1));
            out.push(l.clone());
            out.push(shared);
            continue;
        }
        out.push(l.clone());
    }
    out
}

/// The owner side of an HP-bearing line (`switch`/`drag`/`-damage`/`-heal`/`-sethp`), or
/// `None` when the line is not one. Shares [`fold_hp_line`]'s tag set by construction — it
/// simply asks the question without folding.
fn hp_line_owner(line: &str) -> Option<usize> {
    let parts: Vec<&str> = line.split('|').collect();
    match *parts.get(1)? {
        "switch" | "drag" | "-damage" | "-heal" | "-sethp" => {}
        _ => return None,
    }
    side_of_ident(parts.get(2)?)
}

/// Emit the per-side request frame for both sides at a boundary — ONE chunk each.
///
/// Returns the two `|request|` lines it emitted, so a caller can record what is
/// OUTSTANDING per side without rebuilding the JSON (the sim's `side.activeRequest`;
/// see [`BridgeSession::active_request_json`]). The genesis-replay core ignores the
/// return value.
fn emit_boundary_request_chunks(
    chunks: &mut BridgeChunks,
    state: &BattleState,
    kinds: &[SideRequest; 2],
    no_cancel_forced: bool,
    dex: &Dex,
) -> [String; 2] {
    let mut out = [String::new(), String::new()];
    for side in 0..2 {
        let no_cancel = kinds[side] == SideRequest::ForceSwitch && no_cancel_forced;
        let line = build_request(state, side, kinds[side], false, false, no_cancel, dex);
        chunks.push_chunk(side, vec![line.clone()]);
        out[side] = line;
    }
    out
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
///
/// `pub` so the omniscient BYTE fuzzer (`src/bin/ab_replay.rs --protocol`) can rewrite a
/// `run_full_battle_logged` framing to gen3ou before byte-diffing it against the real
/// gen3ou omniscient capture (`gen3_omniscient_byte_fuzz_v1`).
pub fn reframe(raw: &[crate::protocol::ProtocolLine], format_id: &str) -> Vec<String> {
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
    quick_claw_roll: bool,
    dex: &Dex,
) -> Result<(bool, Vec<crate::protocol::ProtocolLine>, Battle), String> {
    let mut battle = Battle::start_with_switchins(opts, dex)?;
    let st = battle.state_mut().ok_or("no state")?;
    // Restore the turn-0 Quick Claw roll on EVERY genesis replay
    // (`gen3_turn0_quick_claw_capture_v1`) — this driver rebuilds from scratch at each
    // boundary, so setting it once at the caller would be silently dropped from decision 1 on.
    st.quick_claw_roll = quick_claw_roll;
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
    /// The omniscient oracle's post-decision PRNG seed at each RESOLVED decision boundary
    /// (`SEED <id> <bno> <decIdx> <m,n,o,p>` rows, one per committed decision, in order) —
    /// the SEED ANCHOR the `--ab` verdict asserts against `run_full_battle`'s per-decision
    /// engine seed before the per-side byte diff. Empty for a golden without SEED rows
    /// (the anchor is then skipped — backward compatible with the pre-anchor grammar).
    pub seeds: Vec<String>,
    /// The sim's `battle.quickClawRoll` at the FIRST request boundary — the optional 6th
    /// `INIT` field (`gen3_turn0_quick_claw_capture_v1`), the SIBLING of `ab_replay`'s
    /// 5th-field capture.
    ///
    /// This golden's `seed` is the PRE-FIRST-DECISION (post-construction) seed, so
    /// [`BridgeSession::new`] resumes DRAW-FREE via `start_with_switchins` and skips the
    /// turn-0 window where gen3 rolls turn 1's Quick Claw (`randomChance(1,5)` at the
    /// completed `endTurn`, read the NEXT turn — `gen3_quick_claw_speed_v1`). Without this
    /// the offline bridge gate mis-orders turn 1 whenever a Quick Claw leads and that roll
    /// was true — the same defect the offline `ab_replay` path carried. (The LIVE bridge is
    /// unaffected: `sim_bridge` uses [`BridgeSession::new_construct_turn0`], which runs the
    /// real construction from the raw seed.)
    ///
    /// `false` when the field is absent = exactly the pre-fix behaviour, so every committed
    /// bridge golden and saved repro replays byte-for-byte.
    pub quick_claw_roll: bool,
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
                // f: INIT id bno "m,n,o,p" fmt [quickClawRoll:0|1]
                b.seed = f[3].to_string();
                b.format_id = f[4].to_string();
                // OPTIONAL 6th field — absent in every pre-`gen3_turn0_quick_claw_capture_v1`
                // golden, which must keep replaying identically.
                b.quick_claw_roll = f.len() >= 6 && f[5] == "1";
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
            "SEED" => {
                ensure(&mut map, &mut order, f[1]);
                let b = map.get_mut(f[1]).unwrap();
                // f: SEED id bno decIdx "m,n,o,p" — the omniscient oracle's post-decision
                // seed, one row per RESOLVED boundary, appended in order.
                b.seeds.push(f[4].to_string());
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
            Cmd { side: 0, choice: WireChoice::Move(1) },
            Cmd { side: 1, choice: WireChoice::Move(1) },
            Cmd { side: 0, choice: WireChoice::Move(0) },
            Cmd { side: 1, choice: WireChoice::Move(0) },
        ];
        let flat = run_full_battle_bridge(&opts, &cmds, &dex).unwrap();
        let chunked = run_full_battle_bridge_chunked(&opts, &cmds, &dex).unwrap();
        assert_eq!(chunked.flatten().p1, flat.p1);
        assert_eq!(chunked.flatten().p2, flat.p2);
        // And the chunk grouping is non-trivial (framing splits into 3 chunks/side + requests).
        assert!(chunked.side_chunks(0).count() >= 4, "p1 must have ≥4 chunks (framing + requests)");
    }

    /// `gen3_bridge_struggle_resolve_v1` — the STRUGGLE / full-PP-exhaustion WEDGE regression.
    /// A mon with NO usable move (every slot at 0 PP) gets a `|request|` offering ONLY the
    /// synthetic `{"move":"Struggle","id":"struggle"}` (see `build_request`'s `must_struggle`
    /// branch), and poke-env answers with the move-ID `move struggle` — a name that is NEVER
    /// present in the REAL moveset. `resolve_choice` MUST map it to `Move(0)` (the single request
    /// entry; the driver then substitutes Struggle via the `must_struggle` exception in
    /// `choice_is_legal`). WITHOUT the mapping, the moveset search returns `None` → the caller's
    /// out-of-range fallback (`Move(moves.len())`) → `choice_is_legal` rejects it → the bridge
    /// re-issues the SAME request forever: a bridge↔Python no-progress loop that wedged
    /// `--use-bridge=rust` training on ANY battle reaching Struggle (node = ~5k lines / completes,
    /// rust = >1M lines / loops; deterministic repro `designs/ai_v8/BUG_rust_struggle_wedge.md`).
    /// The numeric `move 1` already resolved via the `WireChoice::Move` arm; only the NAME form was
    /// broken — which is exactly the form the real RL runtime + poke-env send. Revert the
    /// `want == "struggle"` early-return in `resolve_choice` and this fails (`None != Some(Move(0))`).
    #[test]
    fn struggle_name_choice_resolves_to_move_zero_not_a_wedge() {
        let dex = Dex::for_gen(3);
        let p1 = "Snorlax||||bodyslam,splash|Serious||M||||";
        let p2 = "Tauros||||bodyslam,splash|Serious||M||||";
        let opts = bridge_opts("gen3customgame", "44317,42357,9927,48760".to_string(), p1, p2);
        let mut battle = Battle::start_with_switchins(&opts, &dex).expect("start");
        let st = battle.state_mut().expect("state");
        // Exhaust every PP of p1's active mon → it MUST Struggle (the wedge trigger).
        let a = st.sides[0].active;
        for pp in st.sides[0].pokemon[a].move_pp.iter_mut() {
            *pp = 0;
        }
        assert!(
            st.sides[0].pokemon[a].must_struggle(&dex),
            "setup: the 0-PP mon must Struggle"
        );
        // poke-env sends the move-ID for the offered Struggle; BOTH the NAME and the numeric
        // `move 1` must resolve to the single request entry = the engine's Move(0).
        assert_eq!(
            resolve_choice(st, 0, &WireChoice::MoveName("struggle".to_string())),
            Some(Choice::Move(0)),
            "`move struggle` must resolve to Move(0) for a must-Struggle mon (pre-fix: None → wedge)"
        );
        assert_eq!(
            resolve_choice(st, 0, &WireChoice::Move(0)),
            Some(Choice::Move(0)),
            "the numeric `move 1` path already resolved to Move(0)"
        );
    }

    /// `gen3_bridge_struggle_resolve_v1` — the END-TO-END class gate (closes the coverage hole
    /// that let the Struggle wedge escape: the bridge byte-differential harness only ran SHORT
    /// battles that never reach PP exhaustion). Drive a real battle to Struggle through the
    /// INCREMENTAL path (`run_full_battle_bridge_incremental`, exactly what `sim_bridge` runs) and
    /// answer p1 with `move struggle` (the NAME form poke-env sends). p1's only move is Mean Look
    /// (harmless, no damage) so it exhausts its PP and is forced to Struggle; the ¼-damage recoil
    /// then ends the frail mirror. WITHOUT the fix, `move struggle` never resolves → the boundary
    /// never commits → the battle never ends (`ended=false`; the standalone repro loops forever).
    /// WITH the fix it completes AND Struggle actually runs (its `[from] Recoil` line appears).
    #[test]
    fn a_bridge_battle_reaching_struggle_completes_not_wedges() {
        let dex = Dex::for_gen(3);
        // Two frail single-mon sides. p1's ONLY move is Mean Look (harmless — traps, no damage),
        // so after its PP is spent p1 must Struggle; p2 just Splashes. Struggle then decides it.
        let p1 = "Rattata||||meanlook|Serious||M||||";
        let p2 = "Rattata||||splash|Serious||F||||";
        let opts = bridge_opts("gen3customgame", "44317,42357,9927,48760".to_string(), p1, p2);
        // p1 ALWAYS answers `move struggle` (the NAME): while Mean Look has PP it resolves to
        // Move(0)=Mean Look; once exhausted the request offers only Struggle and this is the exact
        // wire poke-env sends. p2 Splashes. Extra decisions past game-end are ignored.
        let mut cmds = Vec::new();
        for _ in 0..40 {
            cmds.push(Cmd { side: 0, choice: WireChoice::MoveName("struggle".to_string()) });
            cmds.push(Cmd { side: 1, choice: WireChoice::Move(0) });
        }
        let (chunks, ended, _script, _seeds) =
            run_full_battle_bridge_incremental(&opts, &cmds, &dex).expect("incremental");
        assert!(
            ended,
            "the PP-exhaustion battle MUST complete — pre-fix the `move struggle` NAME never \
             resolves → the boundary never commits → the bridge wedges (ended=false)"
        );
        let all: String = chunks
            .chunks
            .iter()
            .flat_map(|c| c.lines.iter().cloned())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            all.contains("[from] Recoil"),
            "Struggle must have EXECUTED (its gen-3 recoil `[from] Recoil` line must appear), \
             not merely been announced"
        );
    }

    /// `gen3_bridge_switch_nickname_v1` — `switch <NICKNAME>` must resolve.
    ///
    /// poke-env keys mons by their on-field ident (`p1a: <nickname>`) and serializes a switch as
    /// `switch <that name>`. The gen3ou pool ships localized teams — `Airmure (Skarmory)` — so it
    /// really does send `switch Airmure`. Matching only `species_id` made that unresolvable, and
    /// an unresolvable choice never commits the boundary: the battle stopped progressing and the
    /// env's `step()` hung until its 120s watchdog killed the run. This is the choice-resolution
    /// half of `gen3_nickname_ident_v1` (whose emission half was already fixed).
    #[test]
    fn switch_by_nickname_resolves_like_showdown_chooseswitch() {
        let dex = Dex::for_gen(3);
        // p1's bench mon is NICKNAMED (the packed form is `NICK|species|...`).
        let p1 = "Rattata||||splash|Serious||M||||]Airmure|skarmory||keeneye|spikes|Impish||M||||";
        let p2 = "Rattata||||splash|Serious||F||||";
        let opts = bridge_opts("gen3customgame", "44317,42357,9927,48760".to_string(), p1, p2);
        let sess = BridgeSession::new(&opts, &dex).expect("session");
        let st = sess.battle.state().expect("state");

        assert_eq!(
            resolve_choice(st, 0, &WireChoice::SwitchSpecies("Airmure".to_string())),
            Some(Choice::Switch(1)),
            "the NICKNAME is what poke-env sends — pre-fix this returned None, the boundary never \
             committed, and the battle silently stopped progressing"
        );
        // The SPECIES form must keep working (poke-env sends it for un-nicknamed mons).
        assert_eq!(
            resolve_choice(st, 0, &WireChoice::SwitchSpecies("Skarmory".to_string())),
            Some(Choice::Switch(1)),
            "the species form must still resolve"
        );
        // A name matching nothing is still unresolvable (the fail-loud guard's input).
        assert_eq!(
            resolve_choice(st, 0, &WireChoice::SwitchSpecies("Blissey".to_string())),
            None,
            "an unknown name must stay unresolvable so the fail-loud guard reports it"
        );
    }

    /// `gen3_bridge_unresolvable_choice_failloud_v1` — an unmappable NAME choice must FAIL LOUD,
    /// not spin.
    ///
    /// The out-of-range fallback makes `choice_is_legal` reject the choice, leaving the boundary
    /// OPEN and re-issuing the SAME request. For a NUMERIC slot that is a legitimate, tested
    /// reject (poke-env re-picks). For a NAME form it is a GUARANTEED infinite loop: poke-env
    /// re-sends the same unmappable name against the same state, forever. Measured in a live
    /// training run: one bridge child emitting 46 MB/s of re-requests while the env's `step()`
    /// never returned, wedging the whole vec-env. The Struggle wedge was one instance of this
    /// class; this makes the CLASS loud instead of waiting for the next member.
    #[test]
    fn an_unresolvable_name_choice_is_fatal_not_an_endless_re_request() {
        let dex = Dex::for_gen(3);
        let p1 = "Rattata||||splash|Serious||M||||";
        let p2 = "Rattata||||splash|Serious||F||||";
        let opts = bridge_opts("gen3customgame", "44317,42357,9927,48760".to_string(), p1, p2);
        let mut sess = BridgeSession::new(&opts, &dex).expect("session");
        assert!(sess.fatal().is_none(), "a fresh session is not fatal");

        // A move the active mon does not have (and which is not Struggle / a locked pseudo-move).
        sess.feed_cmd(
            Cmd { side: 0, choice: WireChoice::MoveName("thunderbolt".to_string()) },
            &dex,
        );
        let msg = sess.fatal().expect(
            "an unmappable NAME choice must be FATAL — pre-fix it fell through to the \
             out-of-range reject and the bridge re-requested forever",
        );
        assert!(
            msg.contains("unresolvable choice") && msg.contains("thunderbolt"),
            "the error must name the offending wire token so the cause is diagnosable: {msg}"
        );

        // A NUMERIC out-of-range choice keeps the LEGITIMATE reject-and-re-request path (the
        // forced-replacement resume + 0-PP gates depend on it) — it must NOT become fatal.
        let mut ok = BridgeSession::new(&opts, &dex).expect("session");
        ok.feed_cmd(Cmd { side: 0, choice: WireChoice::Move(9) }, &dex);
        assert!(
            ok.fatal().is_none(),
            "a numeric out-of-range slot is a normal reject, not a fatal condition"
        );
    }

    /// `gen3_bridge_forfeit_win_v1` — a FORFEIT must emit the deciding `|win|` to BOTH sides.
    ///
    /// The Node bridge writes `>forcelose <side>` into the sim, so Showdown runs a real
    /// `win(otherSide)` and both players receive `|` + `|win|<name>` before the streams close.
    /// The Rust bridge used to emit a bare `__END__` with NO win line, which left poke-env's
    /// `Battle.finished` False forever: the training seam forfeits whenever `reset()` lands
    /// mid-battle, so the env's next `reset()` waited on a result that could never arrive — a
    /// silent HANG that wedged every multi-episode run (`bridge_session_fuzz_test.py --impl rust`
    /// stalled on its first forfeit-reset).
    ///
    /// Pins the OBSERVABLE bytes, not just the ended flag: p1 forfeits, so BOTH sides must see
    /// `|win|` naming p2's player. Reverting `BridgeSession::forfeit` (or the `sim_bridge`
    /// FORCELOSE wiring) drops the win lines and fails here.
    #[test]
    fn a_forfeit_emits_the_win_line_to_both_sides_not_a_bare_end() {
        let dex = Dex::for_gen(3);
        let p1 = "Rattata||||splash|Serious||M||||";
        let p2 = "Rattata||||splash|Serious||F||||";
        let opts = bridge_opts("gen3customgame", "44317,42357,9927,48760".to_string(), p1, p2);
        let mut sess = BridgeSession::new(&opts, &dex).expect("session");
        // Play one ordinary turn so the forfeit lands MID-battle (the reset()-mid-episode shape),
        // not on a fresh board.
        sess.feed_cmd(Cmd { side: 0, choice: WireChoice::Move(0) }, &dex);
        sess.feed_cmd(Cmd { side: 1, choice: WireChoice::Move(0) }, &dex);
        assert!(!sess.is_ended(), "setup: a Splash mirror must still be live");

        sess.forfeit(0); // p1 forfeits → p2 wins
        assert!(sess.is_ended(), "a forfeit must END the session");

        // The win batch must reach BOTH sides — poke-env marks a battle finished per side.
        for side in 0..2 {
            let lines: Vec<String> = sess
                .chunks()
                .side_chunks(side)
                .flat_map(|c| c.lines.iter().cloned())
                .collect();
            assert!(
                lines.iter().any(|l| l == "|win|P2"),
                "side p{} must receive `|win|P2` on a p1 forfeit — pre-fix the bridge emitted a \
                 bare __END__ with no win line, so poke-env never marked the battle finished and \
                 the next reset() hung. got: {:?}",
                side + 1,
                lines
            );
        }
    }

    /// `gen3_bridge_struggle_resolve_v1` — the SHORT forced-Struggle trigger (the named case):
    /// TAUNT, not PP grind. Gengar Taunts a Skarmory whose whole moveset is STATUS (Spikes / Roar
    /// / Whirlwind / Toxic); Taunt makes every slot un-selectable → Skarmory has NO usable move →
    /// `must_struggle` on turn 2, WITHOUT any PP being spent. This exercises the SAME
    /// `resolve_choice("struggle")` bridge path as PP exhaustion but reaches it in ~2 turns — the
    /// class this locks down is "ANY reason a mon is forced to Struggle," not just 0-PP. Skarmory
    /// answers `move struggle` (the wire poke-env sends); pre-fix the name never resolves → the
    /// boundary never commits → the bridge wedges. WITH the fix it completes + Struggle runs.
    #[test]
    fn taunt_forced_struggle_bridge_battle_completes_not_wedges() {
        let dex = Dex::for_gen(3);
        // Gengar (fast) with Taunt + Shadow Ball to finish; a MAX-bulk Skarmory whose 4 moves are
        // ALL status → Taunt strands it into Struggle. (The bridge does not validate learnsets.)
        let p1 = "Gengar|||levitate|taunt,shadowball|Timid|,,,252,4,252||M||||";
        let p2 = "Skarmory|||keeneye|spikes,roar,whirlwind,toxic|Impish|252,,252,,,||M||||";
        let opts = bridge_opts("gen3customgame", "44317,42357,9927,48760".to_string(), p1, p2);
        // Turn 1: Gengar Taunts (move 1); Skarmory's status move is cant'd. Turn 2+: Skarmory is
        // Taunt-stranded → must Struggle → answers `move struggle`; Gengar Shadow Balls it down.
        let mut cmds = vec![
            Cmd { side: 0, choice: WireChoice::Move(0) },                          // Gengar Taunt
            Cmd { side: 1, choice: WireChoice::MoveName("struggle".to_string()) }, // Skarmory (cant)
        ];
        for _ in 0..24 {
            cmds.push(Cmd { side: 0, choice: WireChoice::Move(1) }); // Gengar Shadow Ball
            cmds.push(Cmd { side: 1, choice: WireChoice::MoveName("struggle".to_string()) });
        }
        let (chunks, ended, _script, _seeds) =
            run_full_battle_bridge_incremental(&opts, &cmds, &dex).expect("incremental");
        assert!(
            ended,
            "the Taunt-forced-Struggle battle MUST complete — pre-fix `move struggle` never resolves \
             → the boundary never commits → the bridge wedges (ended=false)"
        );
        let all: String = chunks
            .chunks
            .iter()
            .flat_map(|c| c.lines.iter().cloned())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            all.contains("|Struggle|"),
            "Skarmory must have RUN Struggle (a `|move|…|Struggle|…` line), not looped on the request"
        );
    }

    /// `gen3_sim_bridge_name_choices_v1` — the REAL RL runtime serializes choices as
    /// NAMES (`/choose move <move_id>` / `/choose switch <species_name>`), NOT the
    /// bridge-golden's 1-based numeric slots. The bridge must resolve those names against
    /// the LIVE state to the SAME `Choice` a numeric token would, so a policy plays the
    /// port bit-for-bit like it plays the Node bridge / live server.
    ///
    /// This pins BOTH resolution directions bit-for-bit — INCLUDING the TYPED Hidden Power
    /// case (`move hiddenpowerice`, the exact move that surfaced the bug: its stored id is
    /// the TYPED `hiddenpowerice`, so a bare `to_id`-normalized compare must still match).
    /// A NAME-choice `Cmd` stream must produce output BYTE-IDENTICAL to the equivalent
    /// numeric stream. Revert-verify: if `resolve_choice` stops matching names to slots
    /// (returns `None`), the driver falls back to an OUT-OF-RANGE slot → the battle
    /// diverges from the numeric baseline and this assertion fails.
    #[test]
    fn name_choices_resolve_to_the_same_slots_incl_typed_hidden_power() {
        let dex = Dex::for_gen(3);
        // p1's active has hiddenpowerice at slot 0 (move 1) and thunderbolt at slot 1
        // (move 2); a bench Snorlax it can switch to by species name. p2 is a bulky wall.
        let p1 = "Jolteon||leftovers|voltabsorb|hiddenpowerice,thunderbolt,\
                  batonpass,agility|Timid|,,,252,4,252||M|||]\
                  Snorlax||leftovers|thickfat|bodyslam,earthquake,rest,curse|\
                  Careful|188,,,,252,||M|||";
        let p2 = "Suicune||leftovers|pressure|surf,icebeam,rest,calmmind|\
                  Bold|252,,252,,4,||M|||";
        let opts = bridge_opts("gen3customgame", "44317,42357,9927,48760".to_string(), p1, p2);

        // Unit-level: resolve each name against the fresh lead state → the correct slot.
        {
            let mut b = Battle::start_with_switchins(&opts, &dex).unwrap();
            let st = b.state_mut().unwrap();
            // p1 move hiddenpowerice → slot 0; thunderbolt → slot 1.
            assert_eq!(
                resolve_choice(st, 0, &WireChoice::MoveName("hiddenpowerice".into())),
                Some(Choice::Move(0)),
                "typed Hidden Power must resolve to its moveslot"
            );
            assert_eq!(
                resolve_choice(st, 0, &WireChoice::MoveName("thunderbolt".into())),
                Some(Choice::Move(1))
            );
            // p1 switch Snorlax → its bench slot (1).
            assert_eq!(
                resolve_choice(st, 0, &WireChoice::SwitchSpecies("Snorlax".into())),
                Some(Choice::Switch(1)),
                "switch-by-species must resolve to the bench slot"
            );
            // An unresolvable name → None (the driver then rejects like an out-of-range slot).
            assert_eq!(resolve_choice(st, 0, &WireChoice::MoveName("earthquake".into())), None);
            assert_eq!(resolve_choice(st, 0, &WireChoice::SwitchSpecies("Zapdos".into())), None);
        }

        // End-to-end: a NAME-based CMD stream (what the RL runtime sends) must produce the
        // EXACT same per-side streams as the equivalent NUMERIC stream — proving the
        // resolution feeds the driver identically. Turn 1 p1 uses typed HP (move 1),
        // p2 surfs (move 1); turn 2 p1 switches to Snorlax by species, p2 ice beams (move 2).
        let numeric = vec![
            Cmd { side: 0, choice: WireChoice::Move(0) },
            Cmd { side: 1, choice: WireChoice::Move(0) },
            Cmd { side: 0, choice: WireChoice::Switch(1) },
            Cmd { side: 1, choice: WireChoice::Move(1) },
        ];
        let named = vec![
            Cmd { side: 0, choice: WireChoice::MoveName("hiddenpowerice".into()) },
            Cmd { side: 1, choice: WireChoice::MoveName("surf".into()) },
            Cmd { side: 0, choice: WireChoice::SwitchSpecies("Snorlax".into()) },
            Cmd { side: 1, choice: WireChoice::MoveName("icebeam".into()) },
        ];
        let by_num = run_full_battle_bridge(&opts, &numeric, &dex).unwrap();
        let by_name = run_full_battle_bridge(&opts, &named, &dex).unwrap();
        assert_eq!(by_name.p1, by_num.p1, "name-choice p1 stream must equal numeric");
        assert_eq!(by_name.p2, by_num.p2, "name-choice p2 stream must equal numeric");
        // Sanity: the streams are non-trivial (a real multi-turn battle with a switch).
        assert!(by_num.p1.iter().any(|l| l.contains("Snorlax")), "the switch must have happened");
    }

    /// `gen3_sim_bridge_nickname_ident_v1` — the on-field IDENTIFIER token
    /// (`pNa: <name>`) in every `|switch|` / `|move|` (user AND target) line MUST be
    /// the packed set's NICKNAME (Showdown's `Pokemon.name = set.name || species.name`),
    /// NOT the species — the SPECIES belongs only in the `|switch|` DETAILS field.
    ///
    /// Real localized teams carry nicknames (e.g. a Zapdos nicknamed `Electhor`).
    /// poke-env keys each mon by the ident token: emit the species there and poke-env
    /// fails to match the mon it already tracks and tries to ADD a 7th →
    /// `ValueError: team already has 6 pokemons` (the nicknamed-team crash). The
    /// broadcast `|switch|`/`|move|` idents are rendered by the protocol emitter's
    /// `turn.rs::display_name` (via `MonRef`); the per-side request/sideupdate idents by
    /// `bridge.rs::display_name` + `mon_ident`. THIS pin covers the emitter path (the
    /// omniscient `|switch|`/`|move|` lines the bridge folds to each side).
    ///
    /// PINNED BYTE-FOR-BYTE vs the real sim (`harness/probe_nickname_perside.js`,
    /// `getPlayerStreams` p1 stream, seed `[7,11,13,17]` → the post-`>start`
    /// construction seed `44317,42357,9927,48760`):
    ///   `|switch|p1a: Electhor|Zapdos|321/321`   (ident = NICKNAME, details = SPECIES)
    ///   `|move|p1a: Electhor|Thunderbolt|p2a: Snorlax`  (the `|move|` user token = nickname)
    ///
    /// REVERT-VERIFY (done): reverting `turn.rs::display_name` to return the species
    /// name makes the ident `|switch|p1a: Zapdos|Zapdos|321/321` (and the `|move|` user
    /// `p1a: Zapdos`), and both asserts below FAIL at the exact nickname-vs-species token.
    #[test]
    fn switch_and_move_ident_tokens_use_the_nickname_not_the_species() {
        let dex = Dex::for_gen(3);
        // p1's lead is a Zapdos NICKNAMED `Electhor` (nickname in packed field 1,
        // species in field 2). p2 is a bulky Snorlax (no nickname → ident = species).
        let p1 = "Electhor|Zapdos||Pressure|thunderbolt,roar|Serious||N||||";
        let p2 = "Snorlax|||Immunity|bodyslam,splash|Serious||M||||]\
                  Regice|||ClearBody|icebeam,splash|Serious||N||||";
        let opts = bridge_opts("gen3customgame", "44317,42357,9927,48760".to_string(), p1, p2);
        // Turn 1: both mons attack (p1 Thunderbolt slot 0, p2 Body Slam slot 0).
        let cmds = vec![
            Cmd { side: 0, choice: WireChoice::Move(0) },
            Cmd { side: 1, choice: WireChoice::Move(0) },
        ];
        let streams = run_full_battle_bridge(&opts, &cmds, &dex).unwrap();

        // The `|switch|` ident is the NICKNAME; the DETAILS field is the SPECIES.
        assert!(
            streams.p1.iter().any(|l| l == "|switch|p1a: Electhor|Zapdos|321/321"),
            "the p1 |switch| ident must be the nickname `Electhor` (details `Zapdos`); \
             got the switch line(s): {:?}",
            streams.p1.iter().filter(|l| l.starts_with("|switch|p1")).collect::<Vec<_>>(),
        );
        // The `|move|` USER token is the nickname too.
        assert!(
            streams.p1.iter().any(|l| l == "|move|p1a: Electhor|Thunderbolt|p2a: Snorlax"),
            "the p1 |move| user token must be the nickname `Electhor`; got the move line(s): {:?}",
            streams.p1.iter().filter(|l| l.starts_with("|move|p1")).collect::<Vec<_>>(),
        );
        // GIGO guard: the nickname is genuinely distinct from the species — so the
        // assert can't pass trivially — and the species NEVER appears in an ident slot
        // (`pNa: Zapdos`), only in the `|switch|` details.
        assert_ne!("Electhor", "Zapdos");
        assert!(
            !streams.p1.iter().any(|l| l.contains("p1a: Zapdos")),
            "the species must never appear as the ident (`p1a: Zapdos`) — only in details",
        );
    }
}
