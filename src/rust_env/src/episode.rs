//! EPISODES AND REWARD — the env's end-of-episode semantics, equal to `Gen3Env`'s (M5 Lane D,
//! `designs/endstate/program_rust_core.md` §2 M5; host half `src/utils/rust_env/episode.py`).
//!
//! * **The reward** is the TERMINAL alone (`Gen3RewardManager.process_turn_reward`; the shaped path
//!   was deleted, `e3ef16db`): [`Terminal::reward`] on p1's reading of the ended battle. `0` on every
//!   op that ends nothing, and for a battle QUARANTINED in progress.
//! * **`terminated` / `truncated`** are `PokeEnv.calc_term_trunc` on p1's reading ([`term_trunc`]):
//!   exactly one side wiped ⇒ terminated, anything else (the stall forfeit, a tie) ⇒ truncated. The
//!   RAW env flags: the learner's re-label (`wrappers.resolve_episode_end`: under `--critic winprob`
//!   a truncation is terminal) is the host's.
//! * **The stall forfeit** is `Gen3Env.action_to_order`: at a p1 (trainee) decision whose turn is
//!   `>= turn_limit` (`StallConfig().threshold` in production) p1 FORFEITS instead of acting, and
//!   p2's pending action is not sent (`PokeEnv.step` skips agent2 after a forfeit). It is decided
//!   BEFORE any feed of the op ([`stall_forfeit_due`]) — finding F-L0-6: the Lane-0 placeholder
//!   checked `turn > limit` AFTER the feeds, so the decisions those feeds opened were encoded and
//!   then discarded, and the forfeit landed one decision late. Only p1 ever forfeits (only the
//!   trainee's `action_to_order` carries the check).
//! * **Ties** end with no winner: indicator ⇒ 0, signed ⇒ `-victory_value` (the pre-cap tie shares
//!   the decisive-loss branch), truncated.
//! * **The terminal observation is NOT produced.** `Gen3Env` returns a Python-encoded row of the
//!   finished battle, and its ONLY consumer is SB3's truncation bootstrap (`TimeLimit.truncated`),
//!   which production never takes: under `--critic winprob` `resolve_episode_end` turns every
//!   truncation terminal. After `done` the `obs` column already holds the NEXT episode's first row
//!   (auto-reset). A host that would bootstrap a truncation (a non-winprob critic) must refuse the
//!   Rust env at startup (Lane G).
//! * **A refused START parks the env** — finding F-L0-2. An episode that ends and whose auto-reset
//!   is REFUSED (a quarantine-class error while constructing the next battle) keeps the ENDED
//!   episode's `reward` / `terminated` / `truncated` / `done`; the refused start is banked
//!   (`refused` = 1) and the env is PARKED (`need` = 0 0, no battle). The next STEP (or RESET) starts
//!   it from the then-staged `ep_team` / `ep_seed`. (Before: the ended episode's outcome was never
//!   written, and the deterministic retry inside the quarantine failed the whole batch.)

use pokesim::json::Json;
use pokesim::present::board_reading::BoardReading;

use crate::core::columns::{EnvCols, SIDES};
use crate::core::pool::{Ctx, Env, Tally};
use crate::core::refusal::{Class, EnvError, InputLog};

/// The TERMINAL-reward declaration (the spec's `terminal` object; `RewardConfig`'s three knobs plus
/// the reward's timeout cap `reward_weights._TIMEOUT_TURN_CAP`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Terminal {
    pub victory_value: f32,
    pub indicator: bool,
    pub draw_penalty: f32,
    pub timeout_turn_cap: u32,
}

/// `episode.TERMINAL_KEYS`.
pub const TERMINAL_KEYS: [&str; 4] = ["victory_value", "terminal_indicator", "draw_penalty", "timeout_turn_cap"];

impl Terminal {
    /// The production terminal (`episode.PRODUCTION_TERMINAL`) — for Rust-side harnesses.
    pub const PRODUCTION: Terminal = Terminal { victory_value: 1.0, indicator: true, draw_penalty: 0.0, timeout_turn_cap: 250 };

    /// Parse the spec's `terminal` object: exactly [`TERMINAL_KEYS`], each typed.
    pub fn from_json(v: Option<&Json>) -> Result<Terminal, String> {
        let v = v.ok_or("spec: missing key \"terminal\"")?;
        let obj = v.as_object().ok_or("spec: `terminal` must be an object")?;
        for k in obj.keys() {
            if !TERMINAL_KEYS.contains(&k.as_str()) {
                return Err(format!("spec: `terminal` has an unknown key {k:?}"));
            }
        }
        for k in TERMINAL_KEYS {
            if !obj.contains_key(k) {
                return Err(format!("spec: `terminal` misses {k:?} (every key is required)"));
            }
        }
        let num = |k: &str| {
            v.get(k).and_then(Json::as_f64).filter(|x| x.is_finite()).ok_or_else(|| format!("spec: `terminal.{k}` must be a finite number"))
        };
        let cap = num("timeout_turn_cap")?;
        if cap < 1.0 || cap.fract() != 0.0 || cap > 4.0e9 {
            return Err(format!("spec: `terminal.timeout_turn_cap` must be a positive integer, got {cap}"));
        }
        let t = Terminal {
            victory_value: num("victory_value")? as f32,
            indicator: v.get("terminal_indicator").and_then(Json::as_bool).ok_or("spec: `terminal.terminal_indicator` must be a boolean")?,
            draw_penalty: num("draw_penalty")? as f32,
            timeout_turn_cap: cap as u32,
        };
        t.validate()?;
        Ok(t)
    }

    /// `main.train.combination_checks`' rule: `draw_penalty` is inapplicable under the indicator.
    pub fn validate(&self) -> Result<(), String> {
        if self.indicator && self.draw_penalty != 0.0 {
            return Err(format!("spec: `terminal.draw_penalty` {} is inapplicable under `terminal_indicator` (must be 0)", self.draw_penalty));
        }
        Ok(())
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"victory_value\":{},\"terminal_indicator\":{},\"draw_penalty\":{},\"timeout_turn_cap\":{}}}",
            self.victory_value, self.indicator, self.draw_penalty, self.timeout_turn_cap
        )
    }

    /// `Gen3RewardManager.process_turn_reward`'s terminal, from p1's view of a FINISHED battle:
    /// `won` = p1 won (`Some(true)`), lost (`Some(false)`) or tied (`None`); `turn` = the reading's
    /// turn at the end (`live.turn`).
    pub fn reward(&self, won: Option<bool>, turn: u32) -> f32 {
        if won == Some(true) {
            self.victory_value
        } else if self.indicator {
            0.0
        } else if turn >= self.timeout_turn_cap {
            self.draw_penalty
        } else {
            -self.victory_value
        }
    }
}

/// `PokeEnv.calc_term_trunc` for a FINISHED battle: `size` is p1's OWN team size (poke-env's
/// `battle.team_size`, used for BOTH sides), `fainted` each side's fainted count in p1's reading.
pub fn term_trunc(size: usize, fainted: [usize; 2]) -> (bool, bool) {
    let wiped = |f: usize| size.saturating_sub(f) == 0;
    let term = wiped(fainted[0]) != wiped(fainted[1]);
    (term, !term)
}

/// The stall forfeit is due before this op's feeds: a p1 decision is open and its turn has reached
/// the threshold (`battle1.turn >= StallConfig().threshold`).
pub fn stall_forfeit_due(turn_limit: Option<u32>, p1_open: bool, p1_turn: u32) -> bool {
    p1_open && turn_limit.is_some_and(|l| p1_turn >= l)
}

/// One ended episode's outcome, from p1's reading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct End {
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

/// The outcome of a finished battle as p1 read it.
pub fn end_of(r: &BoardReading, t: &Terminal) -> Result<End, EnvError> {
    if !r.finished {
        return Err(EnvError::fault("the engine ended the battle but p1's reading saw no |win| / |tie|"));
    }
    let size = r
        .team_size(r.role as usize)
        .ok_or_else(|| EnvError::fault("p1's reading has no |teamsize| for its own side (poke-env's team_size would raise)"))?;
    let fainted = [r.team.iter().filter(|(_, m)| m.fainted()).count(), r.opp.iter().filter(|(_, m)| m.fainted()).count()];
    let (terminated, truncated) = term_trunc(size, fainted);
    Ok(End { reward: t.reward(r.won, r.turn), terminated, truncated })
}

/// A battle banked this op: (the error, its input log, its episode ordinal).
pub type Banked = (EnvError, InputLog, u32);

/// An episode that ENDED this op (M5 Lane H — eval): what an eval host needs about a finished game
/// and cannot read off the columns (after `done` they already describe the NEXT episode, and no
/// column carries the winner or the end turn — a tie and p1's stall forfeit write the same reward
/// and flags under the indicator terminal). Reported for EVERY ended episode, in env order, and held
/// by the core for exactly one op (`Core::finished`, replaced at the next dispatch): a host that
/// wants it reads it after each op; nothing accumulates (the store is reserved at startup).
#[derive(Clone, Debug)]
pub struct Finished {
    pub episode: u32,
    /// 1 = p1 won, 2 = p2 won, 0 = no winner (a tie) — p1's reading's `won`.
    pub winner: u8,
    /// The battle turn at the end (p1's reading's `turn`, poke-env's `battle.turn`).
    pub end_turn: u32,
    /// The side that FORCELOSE'd (1 / 2), 0 = none.
    pub forfeit: u8,
    /// The episode's input log (`InputLog::script` replays it: `core_events`, the record writer).
    pub log: InputLog,
}

impl Finished {
    /// `{"env","episode","winner","end_turn","forfeit","script"}` (`script` = `InputLog::script("")`).
    pub fn json(&self, env: usize) -> String {
        format!(
            "{{\"env\":{env},\"episode\":{},\"winner\":{},\"end_turn\":{},\"forfeit\":{},\"script\":{}}}",
            self.episode,
            self.winner,
            self.end_turn,
            self.forfeit,
            crate::core::refusal::json_str(&self.log.script(""))
        )
    }
}

fn clear_outcome(c: &mut EnvCols) {
    c.reward[0] = 0.0;
    c.done[0] = 0;
    c.terminated[0] = 0;
    c.truncated[0] = 0;
    c.refused[0] = 0;
}

#[cfg(test)]
thread_local! {
    /// TEST-ONLY: refuse the next N starts (after the battle is built) as a quarantine-class error —
    /// the only way to reach a refused START on validated teams (F-L0-2's case).
    pub(crate) static REFUSE_STARTS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

impl Env {
    fn p1_reading(&self) -> Option<&BoardReading> {
        self.chains[0].as_ref().and_then(|c| c.stream(0)).map(|s| &s.board_reading)
    }

    /// Start the next episode; a QUARANTINE-class refusal banks the start and PARKS the env.
    fn start_or_park(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally, banked: &mut Vec<Banked>) -> Result<(), EnvError> {
        self.parked = false;
        let r = self.start(ctx, c, t);
        #[cfg(test)]
        let r = r.and_then(|()| {
            if REFUSE_STARTS.with(|k| k.get()) > 0 {
                REFUSE_STARTS.with(|k| k.set(k.get() - 1));
                return Err(EnvError::engine("injected start refusal (test)"));
            }
            Ok(())
        });
        match r {
            Ok(()) => Ok(()),
            Err(e) if e.class == Class::Quarantine => {
                banked.push((e, self.log.clone(), self.episode()));
                self.sess = None;
                self.chains = [None, None];
                self.open = [None, None];
                self.parked = true;
                c.refused[0] = 1;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// p1 forfeits (`FORCELOSE p1` in its input log — `ForfeitBattleOrder`).
    fn forfeit_p1(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally) -> Result<(), EnvError> {
        self.forfeit_side(ctx, 0, c, t)
    }

    /// `side` forfeits (`FORCELOSE p<side+1>`).
    fn forfeit_side(&mut self, ctx: &Ctx, side: usize, c: &mut EnvCols, t: &mut Tally) -> Result<(), EnvError> {
        self.log.cmds.push(format!("FORCELOSE p{}", side + 1));
        self.sess.as_mut().ok_or_else(|| EnvError::fault("no battle"))?.forfeit(side);
        self.open = [None, None];
        self.advance(ctx, c, t)
    }

    /// M5 Lane E: p2's route forfeits at the stall threshold and its decision has reached it
    /// (`crate::opponents`). Such a decision is NEVER exposed (`need` = 0): today's
    /// `RLPlayer.choose_move` runs its stall check BEFORE any forward, so no row is scored and no
    /// sample drawn — whether its forfeit is then sent (no p1 decision open: [`Env::p2_forfeit_due`])
    /// or dropped because p1 forfeits first (Lane D's rule).
    fn p2_at_stall(&self, ctx: &Ctx) -> bool {
        let p2_turn = self.chains[1].as_ref().and_then(|ch| ch.stream(1)).map_or(0, |s| s.board_reading.turn);
        ctx.spec.opponents.p2_stall_forfeits(self.route) && stall_forfeit_due(ctx.spec.turn_limit, self.open[1].is_some(), p2_turn)
    }

    /// p2 forfeits this STEP: it is at the stall threshold and no p1 decision is open.
    fn p2_forfeit_due(&self, ctx: &Ctx) -> bool {
        self.open[0].is_none() && self.p2_at_stall(ctx)
    }

    fn write_state(&self, ctx: &Ctx, c: &mut EnvCols) -> Result<(), EnvError> {
        let mut any = false;
        let p2_hidden = self.p2_at_stall(ctx);
        for side in 0..SIDES {
            let o = self.open[side].as_ref();
            c.need[side] = (o.is_some() && !(side == 1 && (p2_hidden || self.bot_route(ctx)))) as u8;
            c.dec_n[side] = o.map_or(0, |o| o.n);
            any |= o.is_some();
        }
        c.episode[0] = self.episode();
        c.turn[0] = self.sess.as_ref().map_or(0, |s| s.turn());
        // M5 Lane E: the route this episode's start consumed (a hand-off line: `crate::opponents`).
        c.opp_route[0] = self.route;
        c.opp_slot[0] = ctx.spec.opponents.slot_of(self.route);
        if !any && !self.parked {
            return Err(EnvError::fault("stuck: no decision open and the battle is not over"));
        }
        Ok(())
    }

    /// RESET: start a fresh episode.
    pub(crate) fn reset(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally, banked: &mut Vec<Banked>) -> Result<(), EnvError> {
        clear_outcome(c);
        self.start_or_park(ctx, c, t, banked)?;
        self.write_state(ctx, c)
    }

    /// STEP: the stall forfeit OR feed every open side (p1 first — the script order); then, if the
    /// battle ended, write its outcome and auto-reset.
    pub(crate) fn step(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally, banked: &mut Vec<Banked>) -> Result<(), EnvError> {
        clear_outcome(c);
        if self.parked {
            self.start_or_park(ctx, c, t, banked)?;
            return self.write_state(ctx, c);
        }
        let p1_turn = self.p1_reading().map_or(0, |r| r.turn);
        if stall_forfeit_due(ctx.spec.turn_limit, self.open[0].is_some(), p1_turn) {
            self.forfeit_p1(ctx, c, t)?;
        } else if self.p2_forfeit_due(ctx) {
            // M5 Lane E (hand-off): a POLICY opponent is today's `RLPlayer`, which forfeits at its
            // own decision at the threshold when p1 did not forfeit first (`crate::opponents`).
            self.forfeit_side(ctx, 1, c, t)?;
        } else {
            // Only the decisions open when the op began are fed (the ones the caller answered).
            let act = [c.action[0], c.action[1]];
            let to_feed = [self.open[0].as_ref().map(|o| o.n), self.open[1].as_ref().map(|o| o.n)];
            for side in 0..SIDES {
                if self.sess.as_ref().is_some_and(|s| s.is_ended()) {
                    break;
                }
                // A side whose decision an earlier feed of this op closed or replaced is skipped.
                if to_feed[side].is_some() && self.open[side].as_ref().map(|o| o.n) == to_feed[side] {
                    if side == 1 && self.bot_route(ctx) {
                        // M5 Lane E: the bot answers its own (unexposed) decision, after p1's feed.
                        let tok = self.bot_token()?;
                        self.feed_token(ctx, 1, tok, c, t)?;
                    } else {
                        self.feed(ctx, side, act[side], c, t)?;
                    }
                }
            }
            self.run_bots(ctx, c, t)?;
        }
        if self.sess.as_ref().ok_or_else(|| EnvError::fault("no battle"))?.is_ended() {
            let r = self.p1_reading().ok_or_else(|| EnvError::fault("p1: chain lost"))?;
            let end = end_of(r, &ctx.spec.terminal)?;
            let (winner, end_turn) = (match r.won { Some(true) => 1, Some(false) => 2, None => 0 }, r.turn);
            let forfeit = match self.log.cmds.last().map(String::as_str) {
                Some("FORCELOSE p1") => 1,
                Some("FORCELOSE p2") => 2,
                _ => 0,
            };
            self.finished = Some(Finished { episode: self.episode(), winner, end_turn, forfeit, log: self.log.clone() });
            // The ENDED episode's outcome is written BEFORE the auto-reset, so nothing the reset
            // does (a refused start parks) can lose it (F-L0-2).
            c.reward[0] = end.reward;
            c.terminated[0] = end.terminated as u8;
            c.truncated[0] = end.truncated as u8;
            c.done[0] = 1;
            t.ended += 1;
            self.start_or_park(ctx, c, t, banked)?;
        }
        self.write_state(ctx, c)
    }

    /// QUARANTINE: the battle in progress is dropped (the caller banked it) and the next episode
    /// starts from the staged inputs (or the env parks).
    pub(crate) fn quarantine(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally, banked: &mut Vec<Banked>) -> Result<(), EnvError> {
        clear_outcome(c);
        self.start_or_park(ctx, c, t, banked)?;
        c.done[0] = 1;
        c.refused[0] = 1;
        self.write_state(ctx, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_terminal_is_process_turn_rewards() {
        let ind = Terminal::PRODUCTION;
        assert_eq!(ind.reward(Some(true), 30), 1.0);
        assert_eq!(ind.reward(Some(false), 30), 0.0);
        assert_eq!(ind.reward(None, 30), 0.0, "a tie is a not-win");
        assert_eq!(ind.reward(Some(false), 250), 0.0, "the cap forfeit pays 0 under the indicator");
        let signed = Terminal { victory_value: 30.0, indicator: false, draw_penalty: -35.0, timeout_turn_cap: 250 };
        assert_eq!(signed.reward(Some(true), 251), 30.0);
        assert_eq!(signed.reward(Some(false), 249), -30.0, "a decisive loss");
        assert_eq!(signed.reward(None, 12), -30.0, "the pre-cap tie shares the decisive-loss branch");
        assert_eq!(signed.reward(Some(false), 250), -35.0, "the cap forfeit is the timeout (turn >= cap)");
    }

    #[test]
    fn term_trunc_is_calc_term_trunc() {
        assert_eq!(term_trunc(6, [6, 3]), (true, false), "p1 wiped");
        assert_eq!(term_trunc(6, [0, 6]), (true, false), "p2 wiped");
        assert_eq!(term_trunc(6, [0, 0]), (false, true), "a forfeit: nobody wiped");
        assert_eq!(term_trunc(1, [1, 1]), (false, true), "a tie: both wiped");
    }

    #[test]
    fn the_stall_forfeit_is_p1s_and_at_the_threshold() {
        assert!(!stall_forfeit_due(Some(250), true, 249));
        assert!(stall_forfeit_due(Some(250), true, 250), ">= threshold, not >");
        assert!(!stall_forfeit_due(Some(250), false, 260), "only at a p1 decision");
        assert!(!stall_forfeit_due(None, true, 10_000));
    }

    fn corpus_teams() -> Vec<String> {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../rust_sim/tests/vectors/bridge_corpus");
        let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        let mut out: Vec<String> = Vec::new();
        for f in files.iter().filter(|p| p.extension().is_some_and(|x| x == "txt")) {
            for l in std::fs::read_to_string(f).unwrap().lines().filter(|l| l.starts_with("TEAM\t")) {
                let t = l.split('\t').nth(3).unwrap_or("").to_string();
                if !t.is_empty() && !out.contains(&t) {
                    out.push(t);
                }
            }
        }
        out.truncate(4);
        out
    }

    /// F-L0-2 REGRESSION PIN: an episode that ENDS and whose auto-reset START is refused keeps its
    /// reward / terminated / truncated / done; the refused start is banked (refused = 1) and the env
    /// PARKS (need = 0 0) until the next STEP starts it. WRONG on revert: the reward is lost (0) —
    /// or the deterministic retry inside the quarantine fails the whole batch.
    #[test]
    fn f_l0_2_a_refused_start_keeps_the_ended_episodes_outcome_and_parks() {
        use crate::core::columns::col;
        use crate::core::{Core, OwnedCols, Spec};
        let teams = corpus_teams();
        let spec = Spec {
            n: 1,
            threads: 1,
            format_id: "gen3ou".into(),
            names: ["dpone".into(), "dptwo".into()],
            teams,
            // A forfeit at p1's first decision of turn 2, scored as the timeout (turn >= cap).
            turn_limit: Some(2),
            terminal: Terminal { victory_value: 30.0, indicator: false, draw_penalty: -35.0, timeout_turn_cap: 2 },
            refusal_budget: 4,
            bank_dir: None,
            labels: Vec::new(),
            opponents: crate::opponents::Routes::external(),
        };
        let mut core = Core::new(spec).unwrap();
        let mut cols = OwnedCols::new(1);
        cols.slice_mut::<u32>(col::EP_TEAM).copy_from_slice(&[0, 1]);
        cols.slice_mut::<u32>(col::EP_SEED).copy_from_slice(&[1, 2, 3, 4]);
        let a = cols.addrs();
        core.freeze(a).unwrap();
        assert_eq!(core.dispatch(b'R', a), 0);
        let mut ended = false;
        for _ in 0..64 {
            let (need, mask) = (cols.slice::<u8>(col::NEED).to_vec(), cols.slice::<u8>(col::MASK).to_vec());
            for s in 0..2 {
                let legal = (0..11).find(|&k| mask[s * 11 + k] == 1).map_or(-1, |k| k as i32);
                cols.slice_mut::<i32>(col::ACTION)[s] = if need[s] == 1 { legal } else { -1 };
            }
            let due = need[0] == 1 && cols.slice::<u32>(col::TURN)[0] >= 2;
            if due {
                REFUSE_STARTS.with(|k| k.set(1));
            }
            assert_eq!(core.dispatch(b'S', a), 0, "{:?}", core.last_error().map(|e| e.json()));
            if due {
                assert_eq!(cols.slice::<u8>(col::DONE)[0], 1, "the forfeit ended the episode");
                assert_eq!(cols.slice::<f32>(col::REWARD)[0], -35.0, "the ENDED episode's reward survives the refused start");
                assert_eq!(cols.slice::<u8>(col::TRUNCATED)[0], 1);
                assert_eq!(cols.slice::<u8>(col::TERMINATED)[0], 0);
                assert_eq!(cols.slice::<u8>(col::REFUSED)[0], 1, "the refused start is flagged");
                assert_eq!(cols.slice::<u8>(col::NEED), &[0, 0], "the env is PARKED");
                assert_eq!(core.bank().len(), 1, "the refused start is banked");
                let ep = cols.slice::<u32>(col::EPISODE)[0];
                assert_eq!(core.dispatch(b'S', a), 0, "{:?}", core.last_error().map(|e| e.json()));
                assert_eq!(cols.slice::<u8>(col::DONE)[0], 0);
                assert_eq!(cols.slice::<f32>(col::REWARD)[0], 0.0);
                assert_eq!(cols.slice::<u8>(col::REFUSED)[0], 0);
                assert!(cols.slice::<u8>(col::NEED).contains(&1), "the next STEP started the parked env");
                assert_eq!(cols.slice::<u32>(col::EPISODE)[0], ep + 1);
                ended = true;
                break;
            }
            assert_eq!(cols.slice::<u8>(col::DONE)[0], 0, "no episode ends before the forfeit");
        }
        assert!(ended, "the forfeit never came due");
    }

    #[test]
    fn the_terminal_declaration_parses_strictly() {
        let p = |s: &str| Terminal::from_json(Some(&Json::parse(s).unwrap()));
        let t = p(&Terminal::PRODUCTION.to_json()).unwrap();
        assert_eq!(t, Terminal::PRODUCTION);
        assert!(p(r#"{"victory_value":1,"terminal_indicator":true,"draw_penalty":-35,"timeout_turn_cap":250}"#)
            .unwrap_err()
            .contains("inapplicable"));
        assert!(p(r#"{"victory_value":1,"terminal_indicator":true,"draw_penalty":0}"#).unwrap_err().contains("timeout_turn_cap"));
        assert!(p(r#"{"victory_value":1,"terminal_indicator":true,"draw_penalty":0,"timeout_turn_cap":250,"x":1}"#)
            .unwrap_err()
            .contains("unknown key"));
    }
}
