//! Search kernels — the Rust port of `src/utils/bridge/replay_kernels.js`'s
//! search-relevant half (`gen3_rust_search_driver_v1`).
//!
//! These are the pure helpers `src/bin/search_driver.rs` composes into the warm,
//! persistent clone-and-branch SEARCH server that `utils/bridge/search_session.py`
//! drives (the prober's `better_line` beam). Nothing here reads stdin or writes
//! stdout; nothing here owns a node map. The one job is: given a reconstruction
//! RECORD, rebuild the battle to the start of turn T, then resolve ONE joint turn
//! under a chosen action pair + dice — producing the SAME `used` / `stuck` /
//! `outcome` / `pre_state` the Node kernels produce.
//!
//! # Why a port and not a shim
//!
//! The Node driver clones a paused mid-battle state with `State.serializeBattle`;
//! the port's equivalent is [`BridgeSession::snapshot`] — a derived `Clone`, because
//! everything under a session is plain owned data (see the "Clone-and-branch"
//! section of `src/rust_sim/CLAUDE.md`). So the whole per-arm dance Node needs
//! (deserialize → `restart()` a fresh `BattleStream` → `sendUpdates()` → mark a
//! baseline so the re-emitted historical log is not returned) collapses to
//! `snapshot()` + [`BridgeSession::clear_chunks`], and the chunks a branch reports
//! are ITS SUFFIX by construction rather than by index arithmetic.
//!
//! # The ONE structural difference from Node, and why it is unobservable
//!
//! Node writes BOTH sides' choices into the stream and only then lets the battle
//! tick; the port's [`BridgeSession::feed_cmd`] advances synchronously per call.
//! That difference is not observable:
//!
//! * Showdown's `BattleStream._write` is itself synchronous (it calls `_writeLine`
//!   then `battle.sendUpdates()` inline), so Node's first write ALREADY commits the
//!   boundary if the other side had answered — the `await tick()` only drains the
//!   async per-side chunk collectors. Both engines therefore resolve at the moment
//!   the LAST needed side answers.
//! * A boundary that needs both sides does not resolve on the first feed (the port
//!   keeps `need[side]` open), so the second side's choice is picked against the
//!   same board Node picks against.
//! * A REJECTED choice re-issues the request and leaves the boundary open in both
//!   engines ([`BridgeSession::is_choice_done`] stays `false` across a reject), so
//!   the next loop iteration re-picks identically.
//!
//! If a golden ever shows a divergence here, fix it TOWARD Node's semantics.
//!
//! # HONEST SCOPE — `pre_state`
//!
//! [`pre_state`] mirrors `replay_kernels.js::preState`, an OMNISCIENT referee
//! snapshot with **no consumer today** (the Python side records it; nothing reads
//! it). Two fields cannot be rendered from the port's state with full fidelity, and
//! are approximations rather than models:
//!
//! * **`pseudoWeather`** — the port has no pseudo-weather MAP. gen-3 has no
//!   pseudo-weather MOVE (Trick Room / Gravity are gen-4+), so the only entries a
//!   real gen-3 battle carries are the format's two clause rules, which the port
//!   DOES model behind one flag ([`BattleState::sleep_clause`], which gates both the
//!   Sleep Clause Mod and the Freeze Clause Mod). So the list is derived from that
//!   flag: `["freezeclausemod","sleepclausemod"]` in a clause format, `[]` in
//!   `gen3customgame`. **VERIFIED** against the golden (6/6 cases, gen3ou).
//! * **`volatiles`** — the port models volatiles as individual typed fields on
//!   [`MonState`], not as a keyed map, so the NAME list is RECONSTRUCTED by
//!   [`volatile_names`] from those fields using Showdown's condition ids. The golden
//!   gives that mapping exactly ONE piece of coverage, and it was a real find: the
//!   port's `duration: 1` single-turn fields (`focuspunch`, `pursuit`, `protect`,
//!   `flinch`, `beatup`, the Counter/Mirror-Coat record, `endure`, `snatch`) are
//!   cleared at the NEXT turn's top rather than at the residual, so they are STALE-SET
//!   at a move-request boundary where Showdown has already removed them — including
//!   them made 2 of the golden's 12 `pre_state`s diverge. They are excluded; see
//!   [`volatile_names`]. **Every OTHER name in that mapping is UNVERIFIED** (all twelve
//!   golden `pre_state`s end up with an empty list), and it is called out in
//!   `tmp/search_impl_parity.py`'s printed coverage notes for that reason. Do not treat
//!   a green parity run as evidence that e.g. `choicelock` is spelled or timed right.
//!
//! Everything else in `pre_state` — turn, weather, per-mon species/hp/maxhp/status/
//! fainted/position, boosts, item, ability, the bare-`hiddenpower` moveSlot ids and
//! their PP, and the three gen-3 side conditions — is a faithful read of state the
//! port genuinely models.

use crate::battle::{BattleOptions, PackedTeam, PlayerOptions};
use crate::bridge::{parse_choice, BridgeSession, Cmd, RequestState};
use crate::dex::Dex;
use crate::json::Json;
use crate::state::{BattleState, MonState, Status, Weather};

// ===========================================================================
// The aux PRNG — mulberry32 over an FNV-1a hash of the seed string.
// ===========================================================================

/// The aux PRNG's state — a bit-exact port of `replay_kernels.js::auxRngFromSeed`.
///
/// Deliberately INDEPENDENT of the sim's PRNG: drawing our own "random" action /
/// follow-up picks from `battle.prng` would entangle choice randomness with the dice
/// under study, so a re-roll would no longer isolate the dice. Deterministic per
/// re-roll seed, which is what makes an arm reproducible.
///
/// # Bit-exactness vs JavaScript
///
/// The JS is `Math.imul` + `>>>` over int32; a pure-`u32` implementation is
/// bit-identical because `Math.imul(a, b)` IS a wrapping 32-bit multiply, `>>>` on a
/// negative int32 is a logical shift of the same bit pattern, and `t + Math.imul(...)`
/// is coerced back by the following `^` (JS `ToInt32` == wrapping add mod 2^32; both
/// operands are int32 so the intermediate sum is exactly representable in f64).
/// Pinned against a table of JS-produced draws by `tests/search_driver_test.rs`.
#[derive(Debug, Clone)]
pub struct AuxRng {
    a: u32,
}

/// FNV-1a over the seed string's UTF-16 code units, then mulberry32 seeded with it.
///
/// The hash walks UTF-16 code units because JS's `charCodeAt` does; every seed a
/// caller can produce is ASCII (`"original"`, `"sodium,<hex>"`, `"gen5,<hex>"`,
/// `"m,n,o,p"`), for which code units and bytes coincide — `encode_utf16` is used
/// anyway so a non-ASCII label can never silently diverge.
pub fn aux_rng_from_seed(seed: &str) -> AuxRng {
    let mut h: u32 = 2166136261;
    for unit in seed.encode_utf16() {
        h ^= unit as u32;
        h = h.wrapping_mul(16777619);
    }
    AuxRng { a: h }
}

impl AuxRng {
    /// One mulberry32 draw in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        self.a = self.a.wrapping_add(0x6D2B_79F5);
        let a = self.a;
        let mut t = (a ^ (a >> 15)).wrapping_mul(1 | a);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        f64::from(t ^ (t >> 14)) / 4_294_967_296.0
    }
}

/// `items[Math.floor(rng() * items.length)]` — Node's `pickUniform`. Panics on an
/// empty slice (every caller checks first, exactly as Node does).
pub fn pick_uniform<'a>(rng: &mut AuxRng, items: &'a [String]) -> &'a str {
    let idx = (rng.next_f64() * items.len() as f64).floor() as usize;
    &items[idx]
}

// ===========================================================================
// Choice sources.
// ===========================================================================

/// The `switch N` tokens for `side`'s live bench (`!fainted && !isActive`), in team
/// order — Node's `benchSwitches`. 1-based on the wire.
fn bench_switches(st: &BattleState, side: usize) -> Vec<String> {
    let s = &st.sides[side];
    s.pokemon
        .iter()
        .enumerate()
        .filter(|(i, m)| !m.fainted && *i != s.active)
        .map(|(i, _)| format!("switch {}", i + 1))
        .collect()
}

/// Whether `side`'s OUTSTANDING request says the active mon is FIRMLY trapped.
///
/// The REQUEST is the agent-visible truth, so this reads the exact `|request|` bytes
/// the wire carried ([`BridgeSession::active_request_json`]) rather than re-deriving
/// trapping from the board — which is also what Node does (`req.active[0].trapped`).
/// `maybeTrapped` deliberately does NOT block: the sim refuses the attempt and the
/// caller falls through to a re-pick, and modelling that refusal round is the point.
fn request_says_trapped(sess: &BridgeSession, side: usize) -> bool {
    let Some(line) = sess.active_request_json(side) else {
        return false;
    };
    let body = line.strip_prefix("|request|").unwrap_or(line);
    let Ok(v) = Json::parse(body) else {
        return false;
    };
    v.get("active")
        .and_then(Json::as_array)
        .and_then(|a| a.first())
        .and_then(|a| a.get("trapped"))
        .and_then(Json::as_bool)
        .unwrap_or(false)
}

/// A uniform-random LEGAL choice for `side` given its CURRENT request — Node's
/// `randomChoice`, mirrored branch for branch.
///
/// `None` when the side has no open request or is on `wait`. Otherwise:
/// a forced replacement picks from the bench (`"default"` if it is empty); a move
/// request offers every usable move slot plus, when the request does not say
/// `trapped`, the bench — falling back to the sim's `"default"` if nothing is legal.
///
/// **The move-slot source is the BATTLE's slots, not the request's `moves[]` array**
/// (Node reads `side.active[0].moveSlots`). Those differ under Struggle substitution
/// and move-lock, where the request collapses to a single pseudo-entry while
/// `moveSlots` keeps all four; reading the request there would silently diverge on
/// exactly the Struggle / recharge / two-turn turns. The port's per-slot predicate is
/// [`MonState::move_usable`], which is precisely `!disabled && pp > 0`: it folds the
/// same DisableMove sources Showdown's `moveSlot.disabled` does (Choice lock, Disable,
/// Encore, Taunt) plus the PP gate Node applies alongside it.
pub fn random_choice(
    sess: &BridgeSession,
    side: usize,
    rng: &mut AuxRng,
    dex: &Dex,
) -> Option<String> {
    let kind = sess.request_kind(side)?;
    if kind == RequestState::Wait {
        return None;
    }
    let st = sess.battle_state()?;
    let bench = bench_switches(st, side);
    if kind == RequestState::Switch {
        return Some(if bench.is_empty() {
            "default".to_string()
        } else {
            pick_uniform(rng, &bench).to_string()
        });
    }
    let s = &st.sides[side];
    let mon = &s.pokemon[s.active];
    let mut options: Vec<String> = (0..mon.move_pp.len())
        .filter(|&k| mon.move_usable(k, dex))
        .map(|k| format!("move {}", k + 1))
        .collect();
    if !request_says_trapped(sess, side) {
        options.extend(bench);
    }
    Some(if options.is_empty() {
        "default".to_string()
    } else {
        pick_uniform(rng, &options).to_string()
    })
}

/// The FOLLOW-UP policy for a round the arm's configured source does not answer
/// (a mid-turn forced switch, or a re-pick after a refusal) — Node's `followupChoice`.
pub fn followup_choice(
    sess: &BridgeSession,
    side: usize,
    followup: &str,
    rng: &mut AuxRng,
    dex: &Dex,
) -> Option<String> {
    if followup == "default" {
        return Some("default".to_string());
    }
    random_choice(sess, side, rng, dex)
}

/// Resolve the sim's `"default"` choice token into the concrete choice
/// `Side.chooseDefault()` would commit.
///
/// Rendered as a WIRE TOKEN (`"move K"` / `"switch N"`) because [`resolve_turn`] feeds
/// strings; the resolution itself is [`crate::bridge::resolve_auto_choice`], the one
/// `Side.autoChoose()` port, so the search layer and the live bridge cannot drift.
///
/// This used to carry its own copy of the logic, and that copy was WRONG for a
/// MOVE-LOCKED mon (mustrecharge / a charging two-turn move): it scanned the four real
/// moveslots and could return `move 2`, where Showdown's request holds a single
/// pseudo-entry so `autoChoose` commits slot 1 — and `choice_is_legal` accepts only
/// index 0 while locked, so the choice was refused and the boundary re-opened.
///
/// `None` when nothing is legal at all (a forced switch with an empty bench). Node
/// writes `"default"` regardless and the sim refuses it, leaving the boundary open —
/// [`resolve_turn`] reproduces that by counting the round as written and spinning to
/// its guard, so the `stuck` verdict matches.
fn resolve_default(sess: &BridgeSession, side: usize, dex: &Dex) -> Option<String> {
    let kind = sess.request_kind(side)?;
    let st = sess.battle_state()?;
    match crate::bridge::resolve_auto_choice(st, side, kind == RequestState::Switch, dex)? {
        crate::turn::Choice::Move(k) => Some(format!("move {}", k + 1)),
        crate::turn::Choice::Switch(n) => Some(format!("switch {}", n + 1)),
    }
}

// ===========================================================================
// The reconstruction RECORD.
// ===========================================================================

/// The parts of a `*_reconstruction.json` record the search kernels need — the
/// `>start` / `>player` options plus the ordered command stream.
#[derive(Debug, Clone)]
pub struct Record {
    pub format_id: String,
    /// The RESOLVED `>start` seed (a `Prng::new`-acceptable string).
    pub seed: String,
    pub p1: PlayerOptions,
    pub p2: PlayerOptions,
    /// `[side, payload]` pairs; `side` is `"p1"` / `"p2"` / `"forcelose"`.
    pub commands: Vec<(String, String)>,
}

impl Record {
    /// Parse a record from its JSON form — the `{v, format_id, prng_seed, input_log,
    /// commands}` shape both bridges emit as `__RECON__`.
    ///
    /// The battle options come from `input_log` (the `>start` + two `>player` lines),
    /// exactly like Node's `writeStart`, so a record whose top-level `prng_seed`
    /// disagreed with its `>start` line would still replay the battle the log describes.
    pub fn parse(v: &Json) -> Result<Record, String> {
        let log = v
            .get("input_log")
            .and_then(Json::as_array)
            .ok_or("record has no input_log")?;
        let lines: Vec<&str> = log.iter().filter_map(Json::as_str).collect();
        let start = lines
            .iter()
            .find(|l| l.starts_with(">start "))
            .ok_or("record has no >start line")?;
        let players: Vec<&&str> = lines.iter().filter(|l| l.starts_with(">player ")).collect();
        if players.len() != 2 {
            return Err("record does not have exactly two >player lines".to_string());
        }
        let opts = Json::parse(&start[">start ".len()..])
            .map_err(|e| format!("bad >start JSON: {e}"))?;
        let format_id = opts
            .str_at("formatid")
            .ok_or("record >start has no formatid")?
            .to_string();
        let seed = seed_string(opts.get("seed"))?;

        let mut sides: [Option<PlayerOptions>; 2] = [None, None];
        for line in players {
            let rest = &line[">player ".len()..];
            let (tag, blob) = rest.split_once(' ').ok_or("malformed >player line")?;
            let idx = match tag {
                "p1" => 0,
                "p2" => 1,
                other => return Err(format!("unknown player tag {other:?}")),
            };
            let pv = Json::parse(blob).map_err(|e| format!("bad >player JSON: {e}"))?;
            sides[idx] = Some(PlayerOptions {
                name: pv.str_at("name").unwrap_or_default().to_string(),
                team: PackedTeam(pv.str_at("team").unwrap_or_default().to_string()),
            });
        }

        let mut commands = Vec::new();
        for c in v.get("commands").and_then(Json::as_array).unwrap_or(&[]) {
            let pair = c.as_array().ok_or("a command is not a [side, payload] pair")?;
            let side = pair.first().and_then(Json::as_str).ok_or("command side")?;
            let payload = pair.get(1).and_then(Json::as_str).ok_or("command payload")?;
            commands.push((side.to_string(), payload.to_string()));
        }

        Ok(Record {
            format_id,
            seed,
            p1: sides[0].clone().ok_or("missing >player p1")?,
            p2: sides[1].clone().ok_or("missing >player p2")?,
            commands,
        })
    }

    /// The battle options this record's `>start` describes.
    pub fn battle_options(&self) -> BattleOptions {
        BattleOptions {
            format_id: self.format_id.clone(),
            seed: Some(self.seed.clone()),
            p1: self.p1.clone(),
            p2: self.p2.clone(),
        }
    }
}

/// Render a `>start` `seed` field as a `Prng::new`-acceptable string. The sim's own
/// `inputLog[0]` writes it as a STRING; the array spelling is accepted too because
/// older records (and hand-built ones) carry it.
fn seed_string(field: Option<&Json>) -> Result<String, String> {
    match field.filter(|s| !s.is_null()) {
        None => Err("record >start has no seed".to_string()),
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(Json::Arr(a)) => {
            let mut parts = Vec::with_capacity(a.len());
            for x in a {
                let n = x.as_f64().ok_or("non-numeric seed element")?;
                parts.push(format!("{}", n as u64));
            }
            Ok(parts.join(","))
        }
        Some(other) => Err(format!("unusable >start seed {other:?}")),
    }
}

/// A fresh LIVE session at the record's `>start` — the port's `writeStart`.
///
/// Uses [`BridgeSession::new_construct_turn0`] because the record's seed is the RAW
/// `>start` seed poke-env sent, so the turn-0 construction window (gender samples,
/// speed-tie shuffles, the Quick Claw roll) must actually run — the same constructor
/// the production `sim_bridge` binary uses, and the one `tmp/rust_record_replay_check.py`
/// proved replays a node-recorded command stream byte-for-byte.
pub fn session_from_record(rec: &Record, dex: &Dex) -> Result<BridgeSession, String> {
    BridgeSession::new_construct_turn0(&rec.battle_options(), dex)
}

/// Feed ONE recorded command — Node's `writeCmd`. A `forcelose` entry runs the real
/// forfeit (the `|win|` pair), everything else is a per-side choice.
pub fn write_cmd(sess: &mut BridgeSession, cmd: &(String, String), dex: &Dex) -> Result<(), String> {
    let (side, payload) = cmd;
    if side == "forcelose" {
        let loser = side_index(payload).ok_or_else(|| format!("bad forcelose side {payload:?}"))?;
        sess.forfeit(loser);
        return Ok(());
    }
    let idx = side_index(side).ok_or_else(|| format!("bad command side {side:?}"))?;
    let choice = parse_choice(payload).ok_or_else(|| format!("unsupported choice {payload:?}"))?;
    sess.feed_cmd(Cmd { side: idx, choice }, dex);
    Ok(())
}

fn side_index(tag: &str) -> Option<usize> {
    match tag {
        "p1" => Some(0),
        "p2" => Some(1),
        _ => None,
    }
}

// ===========================================================================
// Reconstruction to the start of turn T.
// ===========================================================================

/// The battle is paused at the start-of-turn MOVE round of turn `t` — Node's
/// `atTurnStart`. Both sides must be on a `move` request, which is what makes turn `t`
/// a branchable joint decision rather than a mid-turn forced-switch round.
///
/// The turn an OPEN boundary is about to resolve, which is NOT always `sess.turn()`.
///
/// `BattleState::turn` counts turns that have been *committed*, and the driver increments it
/// in two places: at `commitChoices` for the very first turn, and then EAGERLY at the end of
/// every turn (`turn_already_opened`, mirroring the sim flushing `|turn|N+1` in the completing
/// write). So from turn 2 onwards the field already names the turn whose request is open — but
/// at the FIRST boundary, where nothing has committed yet, it still reads `0` while the wire
/// has already said `|turn|1`.
///
/// Mapping that `0` to `1` is exact rather than a fudge: `turn()` also returns `0` for a
/// battle that was never built, and `at_turn_start`'s `request_kind` conjunct excludes that
/// case (an unbuilt battle has no boundary, so `request_kind` is `None`). For every `t >= 2`
/// this is the identity, which is why the predicate's behaviour is unchanged there.
///
/// **THREE consumers, and each was independently wrong at turn 1** — one cause, three symptoms,
/// which is why fixing only the first would have been worse than fixing none:
/// 1. [`at_turn_start`] — turn 1 could not be OPENED at all (a loud, honest failure).
/// 2. [`pre_state`] — reported turn `0` where Node reports `1`.
/// 3. The loop guards in [`resolve_turn_exact`] and [`resolve_turn_sourced`] — the first commit
///    moved the raw count off `start_turn`, so the loop exited before any mid-turn follow-up
///    (a faint replacement) was fed, returning a HALF-RESOLVED turn as if it were complete.
///
/// (3) is the dangerous one: it is a silently wrong arm, not an error. Opening turn 1 without
/// it would have traded a loud refusal for a quiet lie. Pinned by
/// `tests/replay_driver_test.rs::a_turn_1_faint_still_gets_its_forced_replacement`.
fn open_boundary_turn(sess: &BridgeSession) -> u32 {
    match sess.turn() {
        0 => 1,
        n => n,
    }
}

/// The battle is paused at the start-of-turn MOVE round of turn `t` — Node's
/// `atTurnStart`. Both sides must be on a `move` request, which is what makes turn `t`
/// a branchable joint decision rather than a mid-turn forced-switch round.
///
/// `gen3_search_turn1_open_v1`: this used to compare `sess.turn()` directly, which made it
/// FALSE for `t == 1` on a freshly constructed session — `build_to_turn` then walked the
/// ENTIRE command log and reported `battle never reached the start of turn 1 (ended=true at
/// turn N)`. Node opens turn 1 fine, so the first decision of every battle (~3.35% of move
/// decisions) was rust-uncoverable on BOTH verb families. The cause was `sess.turn()`, not
/// `request_kind`: see [`open_boundary_turn`].
pub fn at_turn_start(sess: &BridgeSession, t: u32) -> bool {
    !sess.is_ended()
        && open_boundary_turn(sess) == t
        && sess.request_kind(0) == Some(RequestState::Move)
        && sess.request_kind(1) == Some(RequestState::Move)
}

/// Replay recorded commands until the battle sits at the start of turn `t`, returning
/// the index of the first UNAPPLIED command — Node's `buildToTurn`.
///
/// The check runs BEFORE each command, so it stops exactly before turn `t`'s first
/// recorded choice: that index is what makes `recorded_exact` able to reproduce the
/// realized turn and `recorded_turn_choices` able to name the original picks.
pub fn build_to_turn(
    sess: &mut BridgeSession,
    rec: &Record,
    t: u32,
    dex: &Dex,
) -> Result<usize, String> {
    let mut i = 0;
    while i < rec.commands.len() {
        if at_turn_start(sess, t) {
            return Ok(i);
        }
        write_cmd(sess, &rec.commands[i], dex)?;
        i += 1;
    }
    if at_turn_start(sess, t) {
        return Ok(i);
    }
    Err(format!(
        "battle never reached the start of turn {} (ended={} at turn {})",
        t,
        sess.is_ended(),
        sess.turn()
    ))
}

/// The first recorded choice per side at or after command index `from` — the ORIGINAL
/// turn-T picks. `None` for a side that never chose again (the battle ended by forfeit
/// that round); the scan stops at a `forcelose`, mirroring Node's `recordedTurnChoices`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecordedChoices {
    pub p1: Option<String>,
    pub p2: Option<String>,
}

pub fn recorded_turn_choices(rec: &Record, from: usize) -> RecordedChoices {
    let mut out = RecordedChoices::default();
    for (side, payload) in rec.commands.iter().skip(from) {
        if side == "forcelose" {
            break;
        }
        let slot = match side.as_str() {
            "p1" => &mut out.p1,
            "p2" => &mut out.p2,
            _ => continue,
        };
        if slot.is_none() {
            *slot = Some(payload.clone());
        }
        if out.p1.is_some() && out.p2.is_some() {
            break;
        }
    }
    out
}

/// Node's `recordedQueues` cap — how many of a side's remaining recorded commands the
/// `"recorded"` action source may pull from. Node writes `cap || 8`.
pub const RECORDED_QUEUE_CAP: usize = 8;

/// Per-side QUEUES of the remaining recorded commands — Node's `recordedQueues`, and the
/// ONE kernel the SEARCH half never needed.
///
/// **Why a queue and not the single-shot latch [`resolve_turn`] uses for the search path.**
/// The `"recorded"` source in the REPLAY family is aligned to a real command stream, and a
/// live turn's stream can contain a REFUSED choice followed by its correction — the
/// maybe-trapped probe: poke-env attempts a switch, the sim answers `|error|[Unavailable
/// choice]` and re-asks, poke-env sends the corrected choice. Replaying that faithfully under
/// a FRESH seed requires pulling the NEXT recorded command on a refusal (the pre-turn state is
/// identical, so the first attempt refuses identically and the recorded correction follows),
/// which a single-shot source cannot do — it would fall to the follow-up POLICY and invent a
/// choice the original battle never made. Entries beyond the ACCEPTED one are never consumed:
/// a follow-up round in a re-rolled timeline (a forced switch after a faint the original
/// timeline may not even contain) is a NEW decision, so the follow-up policy owns it.
///
/// The scan stops at a `forcelose`, mirroring [`recorded_turn_choices`].
pub fn recorded_queues(rec: &Record, from: usize, cap: usize) -> [Vec<String>; 2] {
    let mut out: [Vec<String>; 2] = [Vec::new(), Vec::new()];
    for (side, payload) in rec.commands.iter().skip(from) {
        if side == "forcelose" {
            break;
        }
        let Some(idx) = side_index(side) else { continue };
        if out[idx].len() < cap {
            out[idx].push(payload.clone());
        }
    }
    out
}

// ===========================================================================
// Turn resolution.
// ===========================================================================

/// The result of resolving ONE joint turn: the choices each side actually committed
/// (in order, including follow-up rounds a mid-turn faint forced) and whether the turn
/// failed to settle.
#[derive(Debug, Clone, Default)]
pub struct Resolved {
    pub used: [Vec<String>; 2],
    pub stuck: bool,
}

/// Reproduce turn T EXACTLY as the original battle did — Node's `resolveTurnExact`.
///
/// Feeds the remaining recorded commands verbatim (original interleaving, original
/// follow-ups) until the turn advances or the battle ends. This is the `value_crn`
/// faithfulness anchor: the realized line scored through the same outcome pipeline as
/// the re-rolls, so margins are directly comparable.
pub fn resolve_turn_exact(
    sess: &mut BridgeSession,
    rec: &Record,
    from: usize,
    dex: &Dex,
) -> Result<Resolved, String> {
    // BOUNDARY-turn units, not the raw committed count (`open_boundary_turn`). At turn 1 the
    // raw count is 0 and the first commit moves it to 1, so a raw guard exits the loop before
    // any follow-up (a faint replacement) is fed — the turn silently resolves half-done.
    // Identity for every turn >= 2, where the count is already the open boundary's.
    let start_turn = open_boundary_turn(sess);
    let mut out = Resolved::default();
    let mut i = from;
    while i < rec.commands.len() && !sess.is_ended() && open_boundary_turn(sess) == start_turn {
        let cmd = &rec.commands[i];
        write_cmd(sess, cmd, dex)?;
        if cmd.0 != "forcelose" {
            if let Some(idx) = side_index(&cmd.0) {
                out.used[idx].push(cmd.1.clone());
            }
        }
        i += 1;
    }
    out.stuck = !sess.is_ended() && open_boundary_turn(sess) == start_turn;
    Ok(out)
}

/// Where one side's turn-T choice comes from — the port of `expandArm`'s `spec`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionSpec {
    /// `"recorded"` — meaningful only on the root, where the caller uses
    /// [`resolve_turn_exact`] instead. Off-root there is no record alignment, so the
    /// source yields nothing and every round falls to the follow-up policy.
    Recorded,
    /// `"random"` — one uniform legal pick, SINGLE-SHOT (see [`resolve_turn`]).
    Random,
    /// An explicit wire choice (`"move 3"`, `"switch 2"`, `"move earthquake"`),
    /// SINGLE-SHOT.
    Explicit(String),
}

impl ActionSpec {
    /// Parse the wire spec string a caller sends (`p1_action` / `p2_action`).
    pub fn parse(s: &str) -> ActionSpec {
        match s {
            "recorded" => ActionSpec::Recorded,
            "random" => ActionSpec::Random,
            other => ActionSpec::Explicit(other.to_string()),
        }
    }
}

/// How many loop iterations a turn gets before it is declared `stuck`.
///
/// Node writes `if (guard++ > 40)`, i.e. it compares the value BEFORE the increment —
/// so 41 iterations run and the 42nd trips. The port checks the pre-increment value
/// too, so the two agree exactly; a naive `guard += 1; if guard > 40` would cut one
/// iteration short and could flip a `stuck` verdict.
const RESOLVE_GUARD: u32 = 40;

/// Resolve ONE whole turn — start-of-turn choices plus any follow-up rounds a mid-turn
/// faint forces. Node's `resolveTurn`, including its routing invariant.
///
/// **The routing invariant:** within one turn a side gets at most ONE `move` request —
/// a refused choice re-asks as `move` again, while every mid-turn follow-up (a forced
/// switch after a faint) asks as `switch`. So a `move` request draws from the side's
/// configured SOURCE and anything else from the follow-up policy. An explicit/random
/// source is SINGLE-SHOT: a refusal falls to the follow-up policy rather than
/// re-sending the same rejected choice (callers detect the refusal from the `|error|`
/// line in that side's suffix chunks). A `"recorded"` source is never aligned off-root,
/// so it yields nothing and the follow-up policy answers every round.
pub fn resolve_turn(
    sess: &mut BridgeSession,
    spec: &[ActionSpec; 2],
    followup: &str,
    rng: &mut AuxRng,
    dex: &Dex,
) -> Resolved {
    let mut sources = [TurnSource::from_spec(&spec[0]), TurnSource::from_spec(&spec[1])];
    resolve_turn_sourced(sess, &mut sources, followup, rng, dex)
}

/// Where ONE side's start-of-turn choice comes from, in the general form BOTH driver
/// families need — the port of Node's `sourceChoice` closure.
///
/// [`ActionSpec`] is the SEARCH driver's wire vocabulary; this is what it lowers to, plus the
/// one variant the search half has no use for ([`TurnSource::Queue`], the replay family's
/// record-aligned `"recorded"` source — see [`recorded_queues`]). Keeping ONE resolution loop
/// behind both is deliberate: a second copy of the guard arithmetic / routing invariant is
/// exactly how the two families would drift.
#[derive(Debug, Clone)]
pub enum TurnSource {
    /// Yields nothing, EVER — every round falls to the follow-up policy. This is what
    /// `"recorded"` means OFF-ROOT in the search driver: there is no command stream to align
    /// to, so pretending otherwise would invent choices.
    Silent,
    /// One uniform legal pick, SINGLE-SHOT (a refusal falls to the follow-up policy).
    Random,
    /// One explicit wire choice, SINGLE-SHOT.
    Explicit(String),
    /// The record-aligned `"recorded"` source: pop the FRONT entry per `move` round, so a
    /// refusal pulls the NEXT recorded command. NOT single-shot. Empty ⇒ yields nothing.
    Queue(std::collections::VecDeque<String>),
}

impl TurnSource {
    /// Lower a search-driver [`ActionSpec`] (whose `Recorded` is the off-root no-alignment case).
    pub fn from_spec(spec: &ActionSpec) -> TurnSource {
        match spec {
            ActionSpec::Recorded => TurnSource::Silent,
            ActionSpec::Random => TurnSource::Random,
            ActionSpec::Explicit(s) => TurnSource::Explicit(s.clone()),
        }
    }

    /// Lower a replay-driver action spec string, where `"recorded"` IS record-aligned and so
    /// becomes the queue built by [`recorded_queues`].
    pub fn from_replay_spec(spec: &str, recorded: &[String]) -> TurnSource {
        match spec {
            "recorded" => TurnSource::Queue(recorded.iter().cloned().collect()),
            "random" => TurnSource::Random,
            other => TurnSource::Explicit(other.to_string()),
        }
    }

    /// Whether this source is SINGLE-SHOT (consumed by its first answer). Node keys this off
    /// the spec kind, not off the source object, and the two agree: `recorded` (either flavour)
    /// is never single-shot, `random` / an explicit string always is.
    fn single_shot(&self) -> bool {
        matches!(self, TurnSource::Random | TurnSource::Explicit(_))
    }

    /// The next choice this source offers, or `None` to fall through to the follow-up policy.
    fn next(&mut self, sess: &BridgeSession, side: usize, rng: &mut AuxRng, dex: &Dex) -> Option<String> {
        match self {
            TurnSource::Silent => None,
            TurnSource::Random => random_choice(sess, side, rng, dex),
            TurnSource::Explicit(s) => Some(s.clone()),
            TurnSource::Queue(q) => q.pop_front(),
        }
    }
}

/// [`resolve_turn`] over the general [`TurnSource`] pair — the shared body both driver
/// families run. See [`resolve_turn`] for the routing invariant and the guard arithmetic.
pub fn resolve_turn_sourced(
    sess: &mut BridgeSession,
    sources: &mut [TurnSource; 2],
    followup: &str,
    rng: &mut AuxRng,
    dex: &Dex,
) -> Resolved {
    // BOUNDARY-turn units — see `resolve_turn_exact` for why a raw `sess.turn()` guard
    // truncates turn 1 after its first commit.
    let start_turn = open_boundary_turn(sess);
    let mut out = Resolved::default();
    let mut single_shot = [false, false];
    let mut guard: u32 = 0;
    while !sess.is_ended() && open_boundary_turn(sess) == start_turn {
        let g = guard;
        guard += 1;
        if g > RESOLVE_GUARD {
            out.stuck = true;
            break;
        }
        let mut wrote = false;
        for i in 0..2 {
            let Some(kind) = sess.request_kind(i) else {
                continue;
            };
            if kind == RequestState::Wait || sess.is_choice_done(i) {
                continue;
            }
            let mut c: Option<String> = None;
            if kind == RequestState::Move && !single_shot[i] {
                if sources[i].single_shot() {
                    single_shot[i] = true;
                }
                c = sources[i].next(sess, i, rng, dex);
            }
            if c.is_none() {
                c = followup_choice(sess, i, followup, rng, dex);
            }
            let Some(choice) = c else { continue };
            out.used[i].push(choice.clone());
            // `default` is a sim-side token with no wire-choice variant; resolve it to
            // the concrete pick `Side.chooseDefault()` commits. When nothing is legal
            // we still count the round as written, so the loop spins to its guard and
            // reports `stuck` exactly as Node's refused-write does.
            let concrete = if choice == "default" {
                resolve_default(sess, i, dex)
            } else {
                Some(choice)
            };
            wrote = true;
            let Some(concrete) = concrete else { continue };
            let Some(parsed) = parse_choice(&concrete) else {
                continue;
            };
            sess.feed_cmd(Cmd { side: i, choice: parsed }, dex);
        }
        if !wrote {
            break;
        }
    }
    out
}

// ===========================================================================
// Omniscient snapshots (referee view — for ground truth, NEVER for obs).
// ===========================================================================

/// Minimal JSON string escaper matching `JSON.stringify` for the characters that can
/// appear here (names, nicknames, ids). Non-ASCII UTF-8 is emitted verbatim, exactly
/// as `JSON.stringify` does — matching the goldens' raw-UTF-8 nicknames.
pub fn json_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
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
    out.push('"');
    out
}

/// The protocol status token (`''` when healthy) — Showdown's `pokemon.status`.
fn status_token(mon: &MonState) -> &'static str {
    match mon.status {
        Some(Status::Burn) => "brn",
        Some(Status::Paralysis) => "par",
        Some(Status::Sleep(_)) => "slp",
        Some(Status::Freeze) => "frz",
        Some(Status::Poison) => "psn",
        Some(Status::Toxic(_)) => "tox",
        None => "",
    }
}

/// [`status_token`] for a mon in a SLOT, which is where the sim's `'fnt'` pseudo-status lives.
///
/// Showdown does not model `fnt` as a status the engine can act on — it is written by
/// `Battle.checkFainted` (battle.ts:2541) purely so the readouts show it. Three facts, each
/// VERIFIED against the node driver rather than inferred, pin exactly when it is visible:
///
/// 1. **ACTIVE SLOT ONLY.** `checkFainted` walks `side.active`, and `BattleActions.switchIn`
///    clears it again the moment the corpse is swapped off (`if (oldActive.fainted)
///    oldActive.status = ''`). So a fainted BENCH mon reports `''` — node's `pre_state` shows
///    exactly that for every fainted bench mon.
/// 2. **NOT AFTER THE BATTLE ENDS.** `runAction` reads `this.faintMessages(); if (this.ended)
///    return true;` BEFORE it reaches `checkFainted()` — so the DECIDING faint never gets the
///    token, and the loser's active reports whatever status it actually carried. Node's
///    `replay` outcome shows a fainted active with `''` and, on the other record, with `par`.
///    Getting this wrong is why the first cut of this function was too eager.
/// 3. **OTHERWISE, a fainted active DOES show it** — e.g. the mid-turn `stuck` case, where
///    both sides are fainted with replacements pending and node reports `fnt` for both.
///
/// The port takes the mirror-image modelling decision — [`crate::state::BattleState::
/// check_fainted`] sets the corpse's status to `None`, which is what makes the paralysis
/// ×0.25 stop applying to the replacement sort (`gen3_fnt_clears_status_v1`) — so the token
/// has to be re-derived HERE rather than read off the state. `show_fnt` is the caller's
/// `is_active_slot && !battle_ended`.
///
/// **This is a REAL divergence the widened replay coverage caught** (`gen3_rust_replay_driver_v1`):
/// the search golden never renders an outcome whose active is fainted (its arms all end at a
/// live move boundary), so nothing exercised it until re-rolls of LATE turns started producing
/// arms that KO the active — and until `replay` started reporting a finished battle at all.
fn slot_status_token(mon: &MonState, show_fnt: bool) -> &'static str {
    if show_fnt && mon.fainted {
        return "fnt";
    }
    status_token(mon)
}

/// The resolved Showdown weather id (`''` when clear) — `field.weather`.
fn weather_id(st: &BattleState) -> &'static str {
    match st.field.weather {
        None => "",
        Some(Weather::Sand) => "sandstorm",
        Some(Weather::Rain) => "raindance",
        Some(Weather::Sun) => "sunnyday",
        Some(Weather::Hail) => "hail",
    }
}

/// One side's `sideOutcome` object — the referee readout a search driver scores with.
///
/// The `active_*` fields are non-null whenever a side has an active slot, which in
/// gen-3 singles is always (a fainted active stays in the slot until its replacement
/// commits, and Node reports its values too). Node's `a ? … : null` branch is
/// therefore unreachable here; the JSON shape is identical.
fn side_outcome(st: &BattleState, side: usize, show_fnt: bool) -> String {
    let s = &st.sides[side];
    let a = &s.pokemon[s.active];
    let alive = s.pokemon.iter().filter(|m| !m.fainted).count();
    let team_hp: u32 = s.pokemon.iter().map(|m| u32::from(m.hp)).sum();
    let team_maxhp: u32 = s.pokemon.iter().map(|m| u32::from(m.maxhp)).sum();
    format!(
        "{{\"active_species\":{},\"active_hp\":{},\"active_maxhp\":{},\"active_status\":{},\
         \"active_fainted\":{},\"alive\":{},\"team_hp\":{},\"team_maxhp\":{}}}",
        json_quote(&a.species_id),
        a.hp,
        a.maxhp,
        json_quote(slot_status_token(a, show_fnt)),
        a.fainted,
        alive,
        team_hp,
        team_maxhp
    )
}

/// The arm OUTCOME object — Node's `outcomeOf`. `stuck:true` is appended AFTER `p2`
/// only when set (Node's `Object.assign(base, extra)` insertion order).
pub fn outcome_of(sess: &BridgeSession, stuck: bool) -> String {
    let Some(st) = sess.battle_state() else {
        return "null".to_string();
    };
    let winner = match sess.winner() {
        Some(w) => json_quote(&st.sides[w].name),
        None => "null".to_string(),
    };
    // The `fnt` token exists only while the battle RUNS — see `slot_status_token`.
    let ended = sess.is_ended();
    let mut out = format!(
        "{{\"turn\":{},\"ended\":{},\"winner\":{},\"p1\":{},\"p2\":{}",
        st.turn,
        ended,
        winner,
        side_outcome(st, 0, !ended),
        side_outcome(st, 1, !ended)
    );
    if stuck {
        out.push_str(",\"stuck\":true");
    }
    out.push('}');
    out
}

/// The Showdown condition ids for the volatiles the port models on `mon`, SORTED —
/// **as observed AT A START-OF-TURN MOVE BOUNDARY**, which is the only point
/// [`pre_state`] is ever taken (`open_root` builds to [`at_turn_start`], and nothing
/// else calls it).
///
/// The port keeps volatiles as typed fields rather than a keyed map, so this
/// reconstructs the name list. Two deliberate exclusions, neither an omission:
///
/// * **The `duration: 1` SINGLE-TURN group** — `flinch`, `protect`, `focuspunch`,
///   `pursuit`, `beatup`, `counter`/`mirrorcoat`, `endure`, `snatch`. Showdown removes
///   these at the end-of-turn residual duration tick, so at the NEXT move request they
///   are gone; the port instead clears them at the following turn's TOP
///   (`turn::helpers::clear_flinch`), so its fields are still set at the boundary. That
///   is a byte-neutral internal timing choice (none of them emits an `-end` line), but
///   reading it as a live volatile would report a volatile Showdown does not have.
///   MEASURED, not assumed: including `focuspunch` made two of the golden's twelve
///   `pre_state`s diverge (`turn5 p1 swampert`, `turn9 p2 heracross`), and excluding the
///   group makes all twelve match.
/// * **Non-volatiles** — Transform, Mimic, Truant and Color Change are overlays/flags in
///   the port AND are not volatiles in the sim either (`pokemon.transformed`, a moveSlot
///   replacement, an ability state, `pokemon.types`), so they are absent rather than
///   invented.
///
/// **UNVERIFIED for the names that remain.** Every `pre_state` in the captured golden
/// has an EMPTY volatile list on both actives, so beyond "the duration-1 group must not
/// appear" this mapping has no differential coverage. Do not read a green parity run as
/// evidence that e.g. `choicelock` or `perishsong` is spelled or timed right.
pub fn volatile_names(mon: &MonState) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    if mon.attract.is_some() {
        v.push("attract");
    }
    if mon.charge {
        v.push("charge");
    }
    if mon.choice_locked_move.is_some() {
        v.push("choicelock");
    }
    if mon.confusion.is_some() {
        v.push("confusion");
    }
    if mon.curse.is_some() {
        v.push("curse");
    }
    if mon.destiny_bond {
        v.push("destinybond");
    }
    if mon.disable.is_some() {
        v.push("disable");
    }
    if mon.encore.is_some() {
        v.push("encore");
    }
    if mon.flash_fire {
        v.push("flashfire");
    }
    if mon.focus_energy {
        v.push("focusenergy");
    }
    if mon.leech_seed.is_some() {
        v.push("leechseed");
    }
    if mon.must_recharge {
        v.push("mustrecharge");
    }
    if mon.partial_trap.is_some() {
        v.push("partiallytrapped");
    }
    if mon.perish.is_some() {
        v.push("perishsong");
    }
    if mon.protect_counter > 0 {
        // `stall` is duration 2 and is run down at the RESIDUAL in the port too, so —
        // unlike its `protect` sibling above — it is genuinely live at the boundary.
        v.push("stall");
    }
    if mon.substitute.is_some() {
        v.push("substitute");
    }
    if mon.taunt.is_some() {
        v.push("taunt");
    }
    if mon.trapped_by.is_some() {
        v.push("trapped");
    }
    if mon.two_turn.is_some() {
        v.push("twoturnmove");
    }
    if mon.yawn.is_some() {
        v.push("yawn");
    }
    v.sort_unstable();
    v
}

/// One mon's `moveSlots` entries — `{id, pp}` with the BARE move id.
///
/// Showdown's `Pokemon` constructor rewrites a typed gen-3 Hidden Power slot back to
/// the bare `hiddenpower` move before pushing the slot (it stashes the type on
/// `set.hpType`), so `moveSlot.id` is bare even though the REQUEST's roster reports the
/// typed id. Confirmed by the golden, whose Salamence shows `moveSlots[3].id ==
/// "hiddenpower"` beside `side.pokemon[0].moves[3] == "hiddenpowerflying"`.
fn move_slots_json(mon: &MonState) -> String {
    let entries: Vec<String> = mon
        .set
        .moves
        .iter()
        .enumerate()
        .map(|(k, mv)| {
            let id = crate::dex::to_id(mv);
            let id = if id.starts_with("hiddenpower") { "hiddenpower".to_string() } else { id };
            let pp = mon.move_pp.get(k).copied().unwrap_or(0);
            format!("{{\"id\":{},\"pp\":{}}}", json_quote(&id), pp)
        })
        .collect();
    format!("[{}]", entries.join(","))
}

/// The side conditions the port models, SORTED — `Object.keys(side.sideConditions)`.
/// gen-3 has exactly these three (Spikes is the only entry hazard; Wish and the future
/// moves are SLOT conditions, which `preState` does not report).
fn side_conditions(st: &BattleState, side: usize) -> Vec<&'static str> {
    let s = &st.sides[side];
    let mut v: Vec<&'static str> = Vec::new();
    if s.light_screen > 0 {
        v.push("lightscreen");
    }
    if s.reflect > 0 {
        v.push("reflect");
    }
    if s.spikes > 0 {
        v.push("spikes");
    }
    v.sort_unstable();
    v
}

fn side_snap(st: &BattleState, side: usize, show_fnt: bool) -> String {
    let s = &st.sides[side];
    let a = &s.pokemon[s.active];
    let boosts = format!(
        "{{\"atk\":{},\"def\":{},\"spa\":{},\"spd\":{},\"spe\":{},\"accuracy\":{},\"evasion\":{}}}",
        a.boosts[0], a.boosts[1], a.boosts[2], a.boosts[3], a.boosts[4], a.boosts[5], a.boosts[6]
    );
    let volatiles: Vec<String> = volatile_names(a).iter().map(|n| json_quote(n)).collect();
    let active = format!(
        "{{\"species\":{},\"hp\":{},\"maxhp\":{},\"status\":{},\"fainted\":{},\"boosts\":{},\
         \"item\":{},\"ability\":{},\"volatiles\":[{}],\"moveSlots\":{}}}",
        json_quote(&a.species_id),
        a.hp,
        a.maxhp,
        json_quote(slot_status_token(a, show_fnt)),
        a.fainted,
        boosts,
        json_quote(&crate::dex::to_id(&a.item)),
        json_quote(&crate::dex::to_id(&a.ability)),
        volatiles.join(","),
        move_slots_json(a)
    );
    // `pokemon[i].position` IS the array index in Showdown (the constructor assigns it
    // and `switchIn` reassigns both slots on the swap), and the port mirrors that swap —
    // so the index is the definition, not an approximation.
    let team: Vec<String> = s
        .pokemon
        .iter()
        .enumerate()
        .map(|(i, m)| {
            format!(
                "{{\"species\":{},\"hp\":{},\"maxhp\":{},\"status\":{},\"fainted\":{},\"position\":{}}}",
                json_quote(&m.species_id),
                m.hp,
                m.maxhp,
                // `i == s.active` is the ACTIVE slot, the only place the sim renders `fnt`.
                json_quote(slot_status_token(m, show_fnt && i == s.active)),
                m.fainted,
                i
            )
        })
        .collect();
    let conds: Vec<String> = side_conditions(st, side).iter().map(|c| json_quote(c)).collect();
    format!(
        "{{\"active\":[{}],\"pokemon\":[{}],\"sideConditions\":[{}]}}",
        active,
        team.join(","),
        conds.join(",")
    )
}

/// The comparable pre-turn board snapshot — Node's `preState`. OMNISCIENT (referee
/// view); never feed it to the obs encoder. See the module header for the two
/// approximated fields.
pub fn pre_state(sess: &BridgeSession) -> String {
    let Some(st) = sess.battle_state() else {
        return "null".to_string();
    };
    // See the module header: the port has no pseudo-weather map, and gen-3's only
    // pseudo-weathers are the format's clause rules, which ride one flag.
    let pseudo = if st.sleep_clause {
        "[\"freezeclausemod\",\"sleepclausemod\"]"
    } else {
        "[]"
    };
    format!(
        "{{\"turn\":{},\"weather\":{},\"pseudoWeather\":{},\"p1\":{},\"p2\":{}}}",
        // The BOUNDARY's turn, not the raw committed count — `pre_state` is only ever taken at
        // a start-of-turn move boundary (see this function's doc), and at the FIRST one the
        // committed count is still 0 while Showdown's `battle.turn` is already 1. Identity
        // everywhere else; see `open_boundary_turn`. Caught by `search_impl_parity` only after
        // `gen3_search_turn1_open_v1` let turn 1 be sampled at all.
        open_boundary_turn(sess),
        json_quote(weather_id(st)),
        pseudo,
        side_snap(st, 0, !sess.is_ended()),
        side_snap(st, 1, !sess.is_ended())
    )
}

/// One side's chunk payloads, in order — each chunk's lines joined by `\n`, the same
/// unit `sim_bridge` base64-frames as one `pN <b64>` stdout line.
pub fn side_chunk_strings(sess: &BridgeSession, side: usize) -> Vec<String> {
    sess.chunks()
        .side_chunks(side)
        .map(|c| c.lines.join("\n"))
        .collect()
}

/// How many OMNISCIENT log lines the session has emitted so far — the port's
/// `battle.log.length`, i.e. the cursor a caller takes BEFORE resolving a turn so it can
/// slice that turn's log afterwards ([`turn_log`]). 0 when there is no state.
pub fn log_len(sess: &BridgeSession) -> usize {
    sess.battle_state().map_or(0, |st| st.log.lines().len())
}

/// The OMNISCIENT log slice `[from..]`, rendered in Showdown's `battle.log` SHAPE —
/// Node's `b.log.slice(logStart)`, the `turn_log` field of the replay family.
///
/// **The shape is not the port's raw line list.** Showdown's `battle.log` stores an
/// `addSplit` effect as THREE entries — `|split|pN`, the SECRET line, the SHARED line
/// (empty when there is no shared form) — and every HP-bearing line is one, because
/// `Pokemon.getHealth()` returns a `{side, secret, shared}` object for the `Battle.add`
/// split path (including `0 fnt`, and including a `reportExactHP` format, where `shared`
/// is simply equal to `secret`). The port instead keeps ONE omniscient line and derives
/// each side's view at fold time, so this reconstructs the triples. See
/// [`crate::bridge::split_log_lines`] for the classification, which is the SAME predicate
/// set the production per-side fold uses (`derive_side`).
///
/// OMNISCIENT (referee view) — never feed it to the obs encoder.
pub fn turn_log(sess: &BridgeSession, from: usize) -> Vec<String> {
    let Some(st) = sess.battle_state() else { return Vec::new() };
    let lines = st.log.lines();
    let from = from.min(lines.len());
    crate::bridge::split_log_lines(&lines[from..], sess.report_percent())
}
