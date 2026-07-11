//! Battle — the deterministic engine core and the high-level surface our tooling
//! calls into.
//!
//! This module sketches the callable API. The shapes are chosen to mirror the
//! FIVE ways our Python bridge already drives Showdown today
//! (`src/utils/bridge/`), so a faithful Rust core is a drop-in:
//!
//! | Bridge pattern (today, via Node)            | Rust surface here            |
//! |---------------------------------------------|------------------------------|
//! | streaming battle (`local_sim_bridge.js`)    | [`BattleStream`]             |
//! | mid-battle RNG swap (counterfactual/search) | [`Battle::reseed`]           |
//! | clone-and-branch (`State.serialize…`)       | [`Battle::serialize`] / [`Battle::deserialize`] |
//! | damage oracle (`damage_probe.js`)           | [`Battle::new`] + state accessors (TODO) |
//! | team pack / validate                        | stays a thin shim for now (out of core scope) |
//!
//! Everything below is a SKELETON: signatures + contracts are real, bodies are
//! `todo!()`. The point is to lock the interface before building the engine.

use crate::dex::Dex;
use crate::prng::PrngSeed;
use crate::protocol::{Choice, Player, ProtocolLine};
use crate::state::{BattleState, SideState};

/// A Showdown packed-team string (the `Teams.pack` format). Opaque for now; the
/// engine will parse it against the [`crate::dex::Dex`].
#[derive(Debug, Clone)]
pub struct PackedTeam(pub String);

/// Per-side start options (mirrors the `>player p1 {...}` payload).
#[derive(Debug, Clone)]
pub struct PlayerOptions {
    pub name: String,
    pub team: PackedTeam,
}

/// Everything needed to start a battle (mirrors the `>start {...}` JSON our
/// bridge writes). A `None` seed means "generate one"; `Some` makes the battle
/// reproducible.
#[derive(Debug, Clone)]
pub struct BattleOptions {
    pub format_id: String, // e.g. "gen3ou"
    pub seed: Option<PrngSeed>,
    pub p1: PlayerOptions,
    pub p2: PlayerOptions,
}

/// A serialized snapshot of a paused battle, for clone-and-branch search
/// (mirrors `State.serializeBattle`/`deserializeBattle` — the prober's lookahead
/// hook). The serialization must preserve PRNG continuity so a resumed branch
/// rolls the same dice the live battle would have.
#[derive(Debug, Clone)]
pub struct BattleSnapshot(pub Vec<u8>);

/// A single battle: the authoritative, deterministic core.
///
/// Contract: same [`BattleOptions`] (seed + teams) + same choice sequence ⇒
/// byte-identical [`ProtocolLine`]s to upstream Showdown.
///
/// **Build status.** The in-battle STATE (`state`) is constructed by
/// [`Battle::start`] — the per-mon stats/HP/status/boosts + sides + field that
/// Showdown sets at construction, BEFORE any switch-in event. The event engine
/// (move resolution, switch-in abilities, the protocol stream) is not built yet,
/// so the action/serialization methods below stay `todo!()`.
pub struct Battle {
    /// The constructed in-battle state (sides, field, PRNG, turn). `None` until
    /// [`Battle::start`] (or, later, [`Battle::new`]) builds it.
    state: Option<BattleState>,
}

impl Battle {
    /// Construct a battle's in-battle STATE from `>start` options + two packed
    /// teams: unpack both teams, compute each mon's stats, set
    /// `hp == maxhp == stats[0]`, status empty, boosts 0, not fainted, and record
    /// the gen-3-singles lead (`active = 0`). Builds the PRNG from `opts.seed`,
    /// `turn = 0`, `field.weather = None`.
    ///
    /// Does **NOT** run any switch-in event (no Intimidate boost change, no Sand
    /// Stream weather, no ability `Start`); that is the event engine's job (a
    /// later step). See [`crate::state`] for the construction-vs-event split.
    ///
    /// Crash-don't-drop: returns `Err` on any unpack/stat failure.
    pub fn start(opts: &BattleOptions, dex: &Dex) -> Result<Battle, String> {
        Ok(Battle {
            state: Some(BattleState::start(opts, dex)?),
        })
    }

    /// Construct the in-battle state AND run the `>start` switch-in event
    /// sequence: switch in both leads and fire their switch-in ability events
    /// (Intimidate's foe-Atk drop; Sand Stream / Drizzle / Drought weather).
    ///
    /// Use this when you want the POST-switch-in board ([`Battle::start`] gives
    /// the pristine construction board, before any event). The dispatch mirrors
    /// Showdown's gen-3 path (`runSwitch` → `singleEvent('Start', ability)`,
    /// abilities resolved in raw-Speed order with the speed-tie shuffle drawing
    /// from the PRNG); see [`crate::event`]. Move execution / the turn loop are
    /// later steps, so `turn()` stays 0.
    ///
    /// Crash-don't-drop: returns `Err` on any unpack/stat failure.
    pub fn start_with_switchins(opts: &BattleOptions, dex: &Dex) -> Result<Battle, String> {
        Ok(Battle {
            state: Some(BattleState::start_with_switchins(opts, dex)?),
        })
    }

    /// The constructed state, if [`Battle::start`] has run.
    pub fn state(&self) -> Option<&BattleState> {
        self.state.as_ref()
    }

    /// Mutable access to the constructed state (for driving a turn via
    /// [`BattleState::run_turn`] before the full `choose`/protocol surface is
    /// built). `None` until [`Battle::start`] has run.
    pub fn state_mut(&mut self) -> Option<&mut BattleState> {
        self.state.as_mut()
    }

    /// Read a side's constructed state (0 = p1, 1 = p2). Panics if the battle
    /// has not been started.
    pub fn side(&self, i: usize) -> &SideState {
        self.state
            .as_ref()
            .expect("Battle::side before Battle::start")
            .side(i)
    }

    /// Start a battle from options (the full `>start` → first-protocol path).
    /// Bodies for the action/protocol phase are not built yet; use
    /// [`Battle::start`] for the construction-time state this step provides.
    pub fn new(_opts: BattleOptions) -> Self {
        todo!("engine core (action/protocol phase) not yet implemented")
    }

    /// Submit a side's [`Choice`] (mirrors `>p1 move 1`). Returns the protocol
    /// lines produced; per-side visibility filtering is [`BattleStream`]'s job.
    pub fn choose(&mut self, _who: Player, _choice: &Choice) -> Vec<ProtocolLine> {
        todo!()
    }

    /// Current turn number (`battle.turn`). 0 at pure construction (the start
    /// action that sets turn 1 is not run this step). Panics if not started.
    pub fn turn(&self) -> u32 {
        self.state
            .as_ref()
            .expect("Battle::turn before Battle::start")
            .turn
    }

    /// Whether the battle has ended (`battle.ended`).
    pub fn ended(&self) -> bool {
        todo!()
    }

    /// The winner, once ended (`None` while ongoing or on a tie).
    pub fn winner(&self) -> Option<Player> {
        todo!()
    }

    /// The current RNG seed string (`battle.prngSeed` / `PRNG.getSeed`).
    pub fn seed(&self) -> PrngSeed {
        todo!()
    }

    /// Swap the RNG mid-battle (mirrors `battle.prng = new PRNG(seed)`). This is
    /// the exact hook our counterfactual re-roll and search arms rely on: replay
    /// to a decision point, then resolve the turn under a fresh seed.
    pub fn reseed(&mut self, _seed: PrngSeed) {
        todo!()
    }

    /// Snapshot for cloning (see [`BattleSnapshot`]).
    pub fn serialize(&self) -> BattleSnapshot {
        todo!()
    }

    /// Rebuild a battle from a snapshot, restoring open requests + PRNG continuity.
    pub fn deserialize(_snap: &BattleSnapshot) -> Self {
        todo!()
    }
}

/// Text-protocol driver around the engine — the drop-in for Showdown's
/// `BattleStream` (the OMNISCIENT stream half; `getPlayerStreams`' per-side
/// privacy fold is downstream, out of scope). Accepts `>`-prefixed command
/// lines and returns, PER WRITE, exactly the omniscient `|...|` lines the real
/// `BattleStream` would flush for that write (`gen3_writeline_stream_v1` —
/// byte-gated by `tests/writeline_test.rs` against a real-Node per-write
/// capture, `harness/gen_writeline_capture.js`).
///
/// # Command grammar (the subset the bridge writes)
///
/// - `>start {"formatid":"gen3customgame","seed":[a,b,c,d]}` — the seed must be
///   the **pre-first-decision** seed (the port's replay convention: the sim's
///   turn-0 construction window — gender samples, the turn-0 Quick Claw — is
///   deliberately not modeled, so the raw `>start` seed is NOT reproducible;
///   the differential harness records the sim's `initSeed` and drives this
///   surface with it, exactly like `protocol_test`). Returns the construction
///   chunk (`|t:|` + `|gametype|singles`).
/// - `>player p1 {"name":…,"team":"<packed>"}` — returns that side's `|player|`
///   line; the SECOND player's write additionally returns the whole battle-init
///   framing through `|turn|1` (the real stream's attribution, probe-verified).
/// - `>p1 move K` / `>p2 switch N` (1-based) — returns whatever the engine
///   flushed for that write: nothing while the request boundary is still open,
///   the full turn chunk (ending `…|upkeep`, `|turn|N+1`) on the write that
///   completes it. An ILLEGAL choice (a disabled/0-PP slot, a fainted switch
///   target, a trapped voluntary switch) is rejected exactly like the sim's
///   `side.choose` — no lines, boundary stays open (the per-side pending
///   acceptance in `run_full_battle`).
///
/// # Internals — replay-from-genesis (honest scope)
///
/// The surface does NOT keep a paused incremental engine: on every choice write
/// it re-runs the whole battle from `>start` through the accumulated one-sided
/// [`ScriptDecision`] stream (`run_full_battle_logged` — the SAME request-
/// boundary machinery every golden gate exercises, so the streamed bytes are
/// the gated bytes by construction) and returns the NEW suffix of the
/// concatenated omniscient stream. Deterministic (the engine is bit-for-bit,
/// so each replay reproduces every prior line byte-identically) and cheap
/// (<1ms/battle); the O(turns²) replay cost is irrelevant at bridge scale.
/// `|request|` / `sideupdate` JSON frames are OUT OF SCOPE (the omniscient log
/// stream is the deliverable; the request frames live with the privacy fold).
pub struct BattleStream {
    dex: Dex,
    format_id: Option<String>,
    seed: Option<PrngSeed>,
    players: [Option<PlayerOptions>; 2],
    /// The accumulated one-sided choice decisions (one per `>pN` write, in write
    /// order) — `run_full_battle`'s per-side pending acceptance pairs them.
    script: Vec<crate::turn::ScriptDecision>,
    /// How many lines of the concatenated omniscient stream have been returned.
    emitted: usize,
    /// The battle outcome of the last replay (`ended` short-circuits writes).
    ended: bool,
}

impl Default for BattleStream {
    fn default() -> Self {
        Self::new()
    }
}

impl BattleStream {
    pub fn new() -> Self {
        BattleStream {
            dex: Dex::for_gen(3),
            format_id: None,
            seed: None,
            players: [None, None],
            script: Vec::new(),
            emitted: 0,
            ended: false,
        }
    }

    /// Whether the battle has ended (the last replay produced `|win|`/`|tie`).
    pub fn battle_ended(&self) -> bool {
        self.ended
    }

    /// Feed one command line. Returns the omniscient protocol lines this write
    /// flushes (see the type-level docs for the per-write attribution contract).
    /// Crash-don't-drop: an unparseable command panics (a malformed driver is a
    /// bug, not a battle state).
    pub fn write_line(&mut self, line: &str) -> Vec<ProtocolLine> {
        let line = line.trim_end_matches('\n');
        let rest = line
            .strip_prefix('>')
            .unwrap_or_else(|| panic!("write_line expects a >-prefixed command, got {line:?}"));

        if let Some(json) = rest.strip_prefix("start ") {
            let v = crate::json::Json::parse(json).expect(">start JSON");
            self.format_id = Some(
                v.str_at("formatid").expect(">start formatid").to_string(),
            );
            // The seed array `[a,b,c,d]` → the engine's "a,b,c,d" PrngSeed string.
            let seed = v
                .get("seed")
                .and_then(|s| s.as_array().map(|a| {
                    a.iter()
                        .map(|x| format!("{}", x.as_f64().expect("seed int") as u64))
                        .collect::<Vec<_>>()
                        .join(",")
                }))
                .expect(">start seed");
            self.seed = Some(seed);
            // The construction chunk: `|t:|` + `|gametype|singles` (probe-verified
            // attribution). Byte-identical to `emit_framing`'s first two lines, which
            // the replay skips via `emitted`.
            let mut pre = crate::protocol::ProtocolBuilder::new();
            pre.enable();
            pre.timestamp();
            pre.gametype_singles();
            let out = pre.drain();
            self.emitted = out.len();
            return out;
        }
        if let Some(json) = rest.strip_prefix("player p1 ").map(|j| (0usize, j))
            .or_else(|| rest.strip_prefix("player p2 ").map(|j| (1usize, j)))
        {
            let (side, payload) = json;
            let v = crate::json::Json::parse(payload).expect(">player JSON");
            let name = v.str_at("name").expect(">player name").to_string();
            let team = v.str_at("team").expect(">player team").to_string();
            self.players[side] = Some(PlayerOptions { name: name.clone(), team: PackedTeam(team) });
            if self.players[0].is_some() && self.players[1].is_some() {
                // Both players present → the battle starts: the replay's framing
                // (which re-emits the construction chunk + the first player line —
                // already returned) yields this write's chunk from `emitted` on.
                return self.replay();
            }
            // Only this side's `|player|` line flushes now (probe-verified).
            let mut pre = crate::protocol::ProtocolBuilder::new();
            pre.enable();
            pre.player(side, &name);
            let out = pre.drain();
            self.emitted += out.len();
            return out;
        }
        if let Some(tok) = rest.strip_prefix("p1 ").map(|t| (0usize, t))
            .or_else(|| rest.strip_prefix("p2 ").map(|t| (1usize, t)))
        {
            let (side, choice_str) = tok;
            let choice = parse_choice_token(choice_str)
                .unwrap_or_else(|| panic!("unsupported choice command {line:?}"));
            if self.ended {
                return Vec::new(); // the sim discards post-game writes
            }
            // SCOPE-LIMIT — CHOICE REVISION is FIRST-ACCEPTED-WINS, not the sim's
            // last-write-wins (`gen3_writeline_choice_revision_v1`, review finding F4).
            // Each `>pN` write APPENDS a fresh per-side [`ScriptDecision`]; a SECOND
            // pre-commit `>pN` for the SAME open boundary appends ANOTHER decision instead
            // of REPLACING the pending one. The real sim's `side.choose` CLEARS and
            // re-parses on every subsequent pre-commit write (LAST-write-wins — probe
            // `harness/probe_f4_choice_revision.js`, resolved Dex.mod('gen3'): `>p1 move 1`
            // then `>p1 move 2` executes Earthquake, seed-identical to a single
            // `>p1 move 2`; a 1→2→3 chain executes Rest). This gap is DOCUMENTED-not-fixed
            // BY DESIGN: the replay-from-genesis accumulator carries no open-boundary
            // marker, so a same-side overwrite cannot be distinguished from a same side in
            // an already-COMMITTED prior decision without new boundary tracking — an
            // overwrite rule that fires on the latter DESTABILIZES the writeline gate (a
            // forced-replacement `>p2 switch N` after a completed turn whose last decision
            // already carries a p2 choice — verified: it dropped the replacement chunk).
            // The real bridge sends exactly ONE choice per request, so a revised `>pN`
            // never occurs in production and this scope-limit is unreachable there. See
            // CLAUDE.md → Protocol emission (the write_line scope-limit list).
            let mut dec = crate::turn::ScriptDecision::default();
            dec.set_side(side, choice);
            self.script.push(dec);
            return self.replay();
        }
        panic!("unsupported write_line command {line:?} (supported: >start / >player pN / >pN move K|switch N)");
    }

    /// Replay the battle from genesis through the accumulated script and return
    /// the NEW lines past `emitted` (see the type-level docs).
    fn replay(&mut self) -> Vec<ProtocolLine> {
        let opts = BattleOptions {
            format_id: self.format_id.clone().expect(">start before choices"),
            seed: Some(self.seed.clone().expect(">start seed before choices")),
            p1: self.players[0].clone().expect(">player p1 before choices"),
            p2: self.players[1].clone().expect(">player p2 before choices"),
        };
        let mut battle =
            Battle::start_with_switchins(&opts, &self.dex).expect("write_line battle start");
        let st = battle.state_mut().expect("state");
        let (outcome, lines) = st.run_full_battle_logged(&self.script, &self.dex);
        self.ended = outcome.ended;
        let out: Vec<ProtocolLine> = lines.into_iter().skip(self.emitted).collect();
        self.emitted += out.len();
        out
    }
}

/// Parse a wire choice token (`move K` / `switch N`, 1-based) into a script
/// [`crate::turn::Choice`] (0-based). `None` for an unsupported token.
fn parse_choice_token(tok: &str) -> Option<crate::turn::Choice> {
    if let Some(k) = tok.strip_prefix("move ") {
        let k: usize = k.trim().parse().ok()?;
        return Some(crate::turn::Choice::Move(k.checked_sub(1)?));
    }
    if let Some(n) = tok.strip_prefix("switch ") {
        let n: usize = n.trim().parse().ok()?;
        return Some(crate::turn::Choice::Switch(n.checked_sub(1)?));
    }
    None
}
