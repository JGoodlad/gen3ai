//! THE SIDE READER — one side's PARSE-built observation chain, advanced write by write (P4).
//!
//! The ONE implementation of "protocol text in, the training row out", shared by every road that
//! turns a side's stream into `__OBS__` frames:
//!
//! * `sim_bridge`'s core observation mode (`gen3_bridge_core_obs_v1`) — the chain training's rows are
//!   held byte-equal to (gate ①, the env core);
//! * `live_reader` — the reader SESSION a websocket client feeds a FOREIGN stream to (poke-env
//!   retirement P4, `main.live.reader`): a real Showdown server's lines, our `--server rust` front
//!   end's, or a public replay's (spectator, no decisions);
//! * `rust_env`'s `bot_reader` (`bot_side::BotSide`, P6) — a roster bot deciding on the reading;
//! * `core_events --obs-stream` — the PROBER's batch reader (P6 merged it in, F-P5-8), through
//!   [`SideReader::advance_fold`]: the same fold, rules and sticky failure without the frame, so it
//!   encodes only at the decisions it was asked for.
//!
//! Because both roads call [`SideReader::advance`], "the live reader = the training reader" holds by
//! construction; the P4 gate (a) ("two roads, one row") checks it on real battles anyway.
//!
//! THE CHAIN. `BattleVersion::parse_root_unrecorded(side, name, packed team, cfg)` (trackers on, no
//! native record — the row never reads the record), advanced by `parse_advance_lean` over exactly the
//! lines the side was newly shipped (incremental — the whole stream is never re-parsed). A choice the
//! side sends is noted BEFORE it is fed ([`SideReader::note_choice`], the order `core_events` uses).
//!
//! THE ALIGNMENT (`encoder.md` §5a). A write opens at most ONE decision, and a decision's `|request|`
//! must be the LAST line of the write; either violation, and every parse / fold / encode failure, is
//! an error. FAILURE IS STICKY: after a failed (or panicked) advance every later advance is refused —
//! the chain no longer holds the stream, and a skipped frame must never pass for a quiet one.
use crate::encoder;
use crate::trackers::clock::ClockConfig;
use crate::version::BattleVersion;

/// One side's reader.
pub struct SideReader {
    side: usize,
    /// The chain; `None` only while an advance is in flight or after a failure.
    chain: Option<BattleVersion>,
    /// Lines of the side's stream folded (the incremental cursor).
    folded: usize,
    /// Decisions taken (one frame each; the next frame's `n`).
    decided: u32,
    /// Set while an advance is in flight and kept on failure (sticky).
    failed: Option<String>,
}

/// What one [`SideReader::advance`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Advanced {
    /// A decision opened at this write's boundary (its frame JSON was written to `out`).
    pub decided: bool,
    /// The stream index of the decision's `|request|` (when `decided`).
    pub line: usize,
}

impl SideReader {
    /// A reader for `side` (0 = p1, 1 = p2) of the player `name` holding `packed_team` (`None`: a
    /// spectator-style reading with no own team — it folds, but takes no decisions a row could be
    /// encoded at).
    pub fn new(side: usize, name: &str, packed_team: Option<&str>, cfg: ClockConfig) -> Result<SideReader, String> {
        let chain = BattleVersion::parse_root_unrecorded(side, name, packed_team, cfg)
            .map_err(|e| format!("core_obs: p{} root: {}", side + 1, e.message()))?;
        Ok(SideReader { side, chain: Some(chain), folded: 0, decided: 0, failed: None })
    }

    pub fn side(&self) -> usize {
        self.side
    }
    /// Lines folded so far.
    pub fn folded(&self) -> usize {
        self.folded
    }
    /// Decisions taken so far (= the next frame's `n`).
    pub fn decided(&self) -> u32 {
        self.decided
    }
    /// The failure that stopped this reader, if any.
    pub fn failure(&self) -> Option<&str> {
        self.failed.as_deref()
    }
    /// The chain at the last completed advance (`None` after a failure).
    pub fn version(&self) -> Option<&BattleVersion> {
        self.chain.as_ref()
    }

    /// Note the choice this side sends for its open decision, BEFORE it is fed (a denied own action
    /// keeps it — `trackers::record::Choice`).
    pub fn note_choice(&mut self, token: &str) {
        if let Some(c) = self.chain.as_mut() {
            c.note_choice(self.side, token);
        }
    }

    /// Fold `lines` — everything this side was newly shipped in ONE write — and, when the write
    /// ended at a decision, append that decision's frame JSON (`encoder::wire::obs_json_into`) to `out`.
    pub fn advance(&mut self, lines: &[&str], out: &mut Vec<u8>) -> Result<Advanced, String> {
        self.advance_with(lines, Some(out))
    }

    /// [`SideReader::advance`] WITHOUT the frame: the same fold, alignment rules and sticky failure,
    /// but no row is encoded — for a caller that reads the decision off [`SideReader::version`] and
    /// encodes only where it needs a row (`core_events --obs-stream`'s `encode_at`).
    pub fn advance_fold(&mut self, lines: &[&str]) -> Result<Advanced, String> {
        self.advance_with(lines, None)
    }

    fn advance_with(&mut self, lines: &[&str], out: Option<&mut Vec<u8>>) -> Result<Advanced, String> {
        if let Some(m) = &self.failed {
            return Err(format!("core_obs: refused after an earlier failure in this battle: {m}"));
        }
        self.failed = Some("a core_obs step did not complete (panic)".to_string());
        match self.advance_inner(lines, out) {
            Ok(a) => {
                self.failed = None;
                Ok(a)
            }
            Err(e) => {
                self.failed = Some(e.clone());
                Err(e)
            }
        }
    }

    fn advance_inner(&mut self, lines: &[&str], out: Option<&mut Vec<u8>>) -> Result<Advanced, String> {
        let side = self.side;
        let tag = side + 1;
        let chain = self.chain.take().ok_or_else(|| format!("core_obs: p{tag}: the chain is gone"))?;
        let next = chain
            .parse_advance_lean(lines)
            .map_err(|e| format!("core_obs: parse p{tag}: {}", e.message()))?;
        self.folded += lines.len();
        let s = next.stream(side).ok_or_else(|| format!("core_obs: p{tag}: the chain lost its stream"))?;
        if s.lines != self.folded {
            return Err(format!("core_obs: p{tag}: the chain folded {} lines, the cursor says {}", s.lines, self.folded));
        }
        let decisions = next.trackers(side).map_or(0, |t| t.decisions);
        let opened = decisions
            .checked_sub(self.decided)
            .ok_or_else(|| format!("core_obs: p{tag}: the decision count went backwards"))?;
        let mut adv = Advanced { decided: false, line: 0 };
        match (next.decision(side), opened) {
            (None, 0) => {}
            (Some(d), 1) => {
                // The core_events alignment: the decision's request is the LAST line the side was
                // shipped in this write (the live player decides after the request chunk).
                if d.line + 1 != s.lines {
                    return Err(format!(
                        "core_obs: [ALIGN] p{tag} decided at stream line {} but the write shipped {} lines",
                        d.line, s.lines
                    ));
                }
                // `n` = this frame's index among this side's frames in this battle.
                if let Some(out) = out {
                    encoder::wire::obs_json_into(&next, side, d.line, self.decided, out)?;
                }
                adv = Advanced { decided: true, line: d.line };
            }
            (d, n) => {
                return Err(format!(
                    "core_obs: p{tag}: one write opened {n} decisions (a decision at the boundary: {}) — \
                     exactly one frame per decision cannot be kept",
                    d.is_some()
                ));
            }
        }
        self.decided = decisions;
        self.chain = Some(next);
        Ok(adv)
    }
}
