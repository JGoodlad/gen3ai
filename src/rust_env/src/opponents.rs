//! OPPONENT ROUTING — who answers p2, per EPISODE (M5 Lane E, `designs/endstate/program_rust_core.md`
//! §2 M5; host half `src/agents/training/rust_env_opponents.py`).
//!
//! The core never runs a network (§2 M5 "What crosses"): a POLICY opponent's rows go OUT to the
//! caller, which forwards them to the inference service (T2) and writes the actions back IN through
//! the `action` column — exactly the shape Lane 0 already serves for both sides. A SCRIPTED BOT
//! (Lane F, `crate::bots`) is played INSIDE the core. What this module adds is the per-episode ROUTE:
//!
//! * **The ROUTE TABLE** is part of the STARTUP DECLARATION (the spec's `opponents` array, parsed by
//!   [`Routes::from_json`]). Each row is one of
//!   - `{"kind": "external"}` — the caller answers p2 by whatever means (the harnesses; Lane 0's
//!     default, and the default of `protocol.spec_json`);
//!   - `{"kind": "policy", "slot": k}` — the caller answers p2 from T2 slot `k` (the `opp_slot`
//!     column names it, so the host groups rows by slot without bookkeeping of its own);
//!   - `{"kind": "bot", "bot": "<name>", "seed": s}` — a Lane-F bot the CORE plays: its p2 decisions
//!     are never exposed (`need` = 0) and it is asked ONLY at a real decision (never on a phantom
//!     poll, F-LF-2; never at p1's stall forfeit, F-LF-5).
//!
//!   The table is fixed for the pool's life. Two policy routes naming the same slot are refused (a
//!   slot IS a route: one-to-one, so `opp_route` and `opp_slot` can never disagree about who played).
//! * **Bot RANDOMNESS** (F-LF-3: production Python bots draw from the unseeded process-wide `random`
//!   — no port can reproduce that stream). DECLARED instead: env `e`'s bot on
//!   a route with seed `s` owns one `Bot` for the pool's life (the one `Player` object per roster
//!   class a worker holds today, its streams running across episodes), each stream `k` (choice 0,
//!   protect 1) seeded `random.Random(`[`stream_seed`]`(s, e, k))`. Same distribution, a
//!   declared stream; `rust_env_opponents.bot_stream_seed` is the Python twin.
//! * **The per-episode input** is the caller-staged `ep_opp` column (a route index), read WITH
//!   `ep_team` / `ep_seed` at RESET and at every auto-reset — so it is the NEXT episode's opponent,
//!   and the caller keeps it staged one episode ahead, like the teams.
//! * **The outputs** `opp_route` / `opp_slot` describe the episode the other columns describe (the
//!   route the core consumed at that episode's start — correct across auto-resets and a refused
//!   start's park without any host-side replay of those rules).
//! * **The OPPONENT's stall forfeit** is route-owned ([`Routes::p2_stall_forfeits`]): a POLICY
//!   opponent is today's `RLPlayer`, whose `choose_move` returns `ForfeitBattleOrder` at a decision
//!   whose `battle2.turn >= StallConfig().threshold` (`Gen3Player._handle_stall`), and `PokeEnv.step`
//!   sends it whenever p1 did not forfeit first. So at a p2 decision with turn `>= turn_limit` and NO
//!   p1 decision open (an open p1 decision at that turn forfeits first — Lane D's rule), p2 FORFEITS
//!   (`FORCELOSE p2`; p1 WINS), and that decision is not exposed (`choose_move` forfeits before any
//!   forward: no row scored, no sample drawn). An EXTERNAL route never forfeits (the harnesses script
//!   p2), nor does a BOT (the Python bots carry no stall check — F-LF-5).
//!
//! A route index outside the table is the CALLER's error (`CallerError`), found at the start that
//! reads it — never a silent fallback to route 0. A bot that refuses (where the Python bot would
//! RAISE) QUARANTINES its battle (kind `bot`): confined to that battle, banked and budgeted.

use pokesim::json::Json;

use crate::bots::{Bot, Kind};
use crate::core::columns::EnvCols;
use crate::core::pool::{Ctx, Env, Tally};
use crate::core::refusal::{Class, EnvError};

/// One row of the route table (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub enum Route {
    External,
    Policy { slot: u32 },
    /// `per_episode` (the spec key `"streams": "episode"`, M5 Lane H — eval): the bot's streams are
    /// RE-SEEDED at every episode start from the route seed and that episode's staged battle seed
    /// ([`episode_stream_seed`]), so a game's bot draws are a function of the GAME alone — not of the
    /// env it ran on or the episodes that env played before (the default `"env"` rule: one stream per
    /// env for the pool's life, the one `Player` a training worker holds).
    Bot { kind: Kind, seed: u64, per_episode: bool },
}

/// The kinds a row may name (`rust_env_opponents.ROUTE_KINDS` on the host).
pub const KINDS: [&str; 3] = ["external", "policy", "bot"];

/// Seeds must survive a JSON number (f64) exactly.
const MAX_SEED: f64 = 9_007_199_254_740_991.0; // 2^53 - 1

/// Does this core play the scripted bot `name` (Lane F's `bots::Kind` names)?
pub fn bot_available(name: &str) -> bool {
    Kind::from_name(name).is_some()
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed of env `env`'s bot stream `stream` (choice 0, protect 1) on a route declared with
/// `seed` — `random.Random(stream_seed(...))`. Python twin: `rust_env_opponents.bot_stream_seed`.
pub fn stream_seed(seed: u64, env: usize, stream: u64) -> u64 {
    splitmix64(seed ^ splitmix64(((env as u64) << 2) | stream))
}

/// The four 16-bit battle-seed words of an episode packed low word first (`w0 | w1 << 16 | …`).
pub fn pack_seed_words(words: &[u32]) -> u64 {
    words.iter().take(4).enumerate().fold(0u64, |acc, (i, w)| acc | ((*w as u64 & 0xFFFF) << (16 * i)))
}

/// The seed of stream `stream` (choice 0, protect 1) of a `"streams": "episode"` bot route
/// declared with `seed`, for the episode whose staged battle seed packs to `episode_key`
/// ([`pack_seed_words`]) — `random.Random(episode_stream_seed(...))`. Python twin:
/// `rust_env_opponents.episode_bot_stream_seed`.
pub fn episode_stream_seed(seed: u64, episode_key: u64, stream: u64) -> u64 {
    stream_seed(seed ^ episode_key, 0, stream)
}

/// The declared route table (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct Routes(pub Vec<Route>);

impl Routes {
    /// The table every caller that does not declare one gets: one EXTERNAL route (Lane 0's shape).
    pub fn external() -> Routes {
        Routes(vec![Route::External])
    }

    /// Parse the spec's `opponents` array; the error names the row and the key.
    pub fn from_json(v: Option<&Json>) -> Result<Routes, String> {
        let arr = v.and_then(Json::as_array).ok_or("spec: `opponents` must be an array of route objects")?;
        let mut out = Vec::with_capacity(arr.len());
        for (i, r) in arr.iter().enumerate() {
            let obj = r.as_object().ok_or_else(|| format!("spec: opponents[{i}] must be an object"))?;
            let kind = r.str_at("kind").ok_or_else(|| format!("spec: opponents[{i}] needs a string `kind` (one of {KINDS:?})"))?;
            let bot = r.str_at("bot");
            let allowed: &[&str] = match kind {
                "external" => &["kind"],
                "policy" => &["kind", "slot"],
                "bot" => &["kind", "bot", "seed", "streams"],
                other => return Err(format!("spec: opponents[{i}]: unknown kind {other:?} (one of {KINDS:?})")),
            };
            for k in obj.keys() {
                if !allowed.contains(&k.as_str()) {
                    return Err(format!("spec: opponents[{i}] ({kind}): unknown key {k:?}"));
                }
            }
            for k in allowed.iter().filter(|k| **k != "streams") {
                if !obj.contains_key(*k) {
                    return Err(format!("spec: opponents[{i}] ({kind}): missing key {k:?}"));
                }
            }
            let uint = |key: &str, max: f64| -> Result<u64, String> {
                let s = r.get(key).and_then(Json::as_f64).ok_or_else(|| format!("spec: opponents[{i}].{key} must be an integer"))?;
                if s < 0.0 || s.fract() != 0.0 || s > max {
                    return Err(format!("spec: opponents[{i}].{key} must be a non-negative integer <= {max}, got {s}"));
                }
                Ok(s as u64)
            };
            out.push(match kind {
                "external" => Route::External,
                "policy" => Route::Policy { slot: uint("slot", 1.0e6)? as u32 },
                _ => {
                    let name = bot.ok_or_else(|| format!("spec: opponents[{i}].bot must be a bot name"))?;
                    let k = Kind::from_name(name).ok_or_else(|| {
                        format!("spec: opponents[{i}]: bot {name:?} is not one this core plays (Lane F's `bots::Kind` names)")
                    })?;
                    let per_episode = match r.get("streams") {
                        None => false,
                        Some(v) => match v.as_str() {
                            Some("env") => false,
                            Some("episode") => true,
                            _ => return Err(format!("spec: opponents[{i}].streams must be \"env\" or \"episode\"")),
                        },
                    };
                    Route::Bot { kind: k, seed: uint("seed", MAX_SEED)?, per_episode }
                }
            });
        }
        let routes = Routes(out);
        routes.validate()?;
        Ok(routes)
    }

    /// Non-empty, slots one-to-one.
    pub fn validate(&self) -> Result<(), String> {
        if self.0.is_empty() {
            return Err("spec: `opponents` must declare at least one route".into());
        }
        let mut seen: Vec<u32> = Vec::new();
        for (i, r) in self.0.iter().enumerate() {
            if let Route::Policy { slot } = r {
                if seen.contains(slot) {
                    return Err(format!("spec: opponents[{i}]: slot {slot} is already a route (a slot IS one route)"));
                }
                seen.push(*slot);
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The staged route index, checked against the table (a CALLER error when outside it).
    pub fn check(&self, env: usize, route: u32) -> Result<u32, EnvError> {
        if (route as usize) < self.0.len() {
            Ok(route)
        } else {
            Err(EnvError::caller(format!(
                "env {env}: ep_opp = {route} is outside the declared route table ({} routes)",
                self.0.len()
            )))
        }
    }

    /// Does the p2 side of `route` forfeit at the stall threshold (see the module docs)?
    pub fn p2_stall_forfeits(&self, route: u32) -> bool {
        matches!(self.0.get(route as usize), Some(Route::Policy { .. }))
    }

    /// Is `route` a bot the core plays (p2 never exposed)?
    pub fn is_bot(&self, route: u32) -> bool {
        matches!(self.0.get(route as usize), Some(Route::Bot { .. }))
    }

    /// The `opp_slot` value of a route: its T2 slot, or -1 when the route is not a policy.
    pub fn slot_of(&self, route: u32) -> i32 {
        match self.0.get(route as usize) {
            Some(Route::Policy { slot }) => *slot as i32,
            _ => -1,
        }
    }

    /// At an episode START on `route` (with the episode's staged battle-seed `words`): the fresh bot
    /// of a `"streams": "episode"` route, or None (every other route keeps its bot). The caller
    /// replaces its bot in place — a fixed-size value, nothing acquired.
    pub fn episode_bot(&self, route: u32, words: &[u32]) -> Option<Bot> {
        match self.0.get(route as usize) {
            Some(Route::Bot { kind, seed, per_episode: true }) => {
                let key = pack_seed_words(words);
                Some(Bot::new(*kind, episode_stream_seed(*seed, key, 0), episode_stream_seed(*seed, key, 1)))
            }
            _ => None,
        }
    }

    /// Env `env`'s bots, one per route (None for a non-bot route) — acquired at STARTUP.
    pub fn build_bots(&self, env: usize) -> Vec<Option<Bot>> {
        self.0
            .iter()
            .map(|r| match r {
                Route::Bot { kind, seed, .. } => {
                    Some(Bot::new(*kind, stream_seed(*seed, env, 0), stream_seed(*seed, env, 1)))
                }
                _ => None,
            })
            .collect()
    }

    /// The canonical JSON (round-trips through [`Routes::from_json`]).
    pub fn to_json(&self) -> String {
        let rows: Vec<String> = self
            .0
            .iter()
            .map(|r| match r {
                Route::External => "{\"kind\":\"external\"}".to_string(),
                Route::Policy { slot } => format!("{{\"kind\":\"policy\",\"slot\":{slot}}}"),
                Route::Bot { kind, seed, per_episode } => {
                    format!("{{\"kind\":\"bot\",\"bot\":\"{}\",\"seed\":{seed}{}}}", kind.name(), streams_key(*per_episode))
                }
            })
            .collect();
        format!("[{}]", rows.join(","))
    }
}

fn streams_key(per_episode: bool) -> &'static str {
    if per_episode {
        ",\"streams\":\"episode\""
    } else {
        ""
    }
}

// ------------------------------------------------------------------ the bot, played in the core

impl Env {
    /// The episode's route is a bot.
    pub(crate) fn bot_route(&self, ctx: &Ctx) -> bool {
        ctx.spec.opponents.is_bot(self.route)
    }

    /// The bot's choice token at p2's open decision (Lane F's `Bot::decide` on p2's reading, over the
    /// core's own tokens). A refusal quarantines the battle (kind `bot`).
    pub(crate) fn bot_token(&mut self) -> Result<String, EnvError> {
        let route = self.route as usize;
        let reading = &self
            .chains[1]
            .as_ref()
            .and_then(|ch| ch.stream(1))
            .ok_or_else(|| EnvError::fault("p2: chain lost at a bot decision"))?
            .board_reading;
        let tokens = self.open[1].as_ref().map(|o| o.tokens.as_slice()).ok_or_else(|| EnvError::fault("bot: no p2 decision open"))?;
        let bot = self.bots.get_mut(route).and_then(Option::as_mut).ok_or_else(|| EnvError::fault("bot route without a bot"))?;
        let d = bot
            .decide(reading, tokens)
            .map_err(|e| EnvError { class: Class::Quarantine, kind: "bot", py_class: None, message: format!("{:?} {e}", bot.kind) })?;
        Ok(d.token)
    }

    /// While ONLY the bot has a decision open (p1 waits — a forced switch, a replacement), the bot
    /// plays it: the caller is asked only when p1 decides or the battle ends.
    pub(crate) fn run_bots(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally) -> Result<(), EnvError> {
        let mut guard = 0;
        while self.bot_route(ctx)
            && self.open[1].is_some()
            && self.open[0].is_none()
            && !self.sess.as_ref().is_some_and(|s| s.is_ended())
        {
            guard += 1;
            if guard > 64 {
                return Err(EnvError::fault("bot: 64 bot-only decisions in a row without a p1 decision"));
            }
            let tok = self.bot_token()?;
            self.feed_token(ctx, 1, tok, c, t)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Routes, String> {
        Routes::from_json(Some(&Json::parse(s).unwrap()))
    }

    #[test]
    fn the_table_round_trips_and_names_slots() {
        let r = p(r#"[{"kind":"external"},{"kind":"policy","slot":3},{"kind":"policy","slot":0}]"#).unwrap();
        assert_eq!(r.0, vec![Route::External, Route::Policy { slot: 3 }, Route::Policy { slot: 0 }]);
        assert_eq!(p(&r.to_json()).unwrap(), r);
        assert_eq!([r.slot_of(0), r.slot_of(1), r.slot_of(2), r.slot_of(9)], [-1, 3, 0, -1]);
        assert!(r.check(0, 2).is_ok());
        let e = r.check(5, 3).unwrap_err();
        assert!(e.message.contains("env 5") && e.message.contains("ep_opp = 3"), "{}", e.message);
    }

    #[test]
    fn a_malformed_table_is_refused_by_name() {
        for (text, needle) in [
            ("[]", "at least one route"),
            ("{}", "must be an array"),
            (r#"[{"kind":"nope"}]"#, "unknown kind"),
            (r#"[{"kind":"policy"}]"#, "missing key \"slot\""),
            (r#"[{"kind":"policy","slot":1,"x":2}]"#, "unknown key \"x\""),
            (r#"[{"kind":"policy","slot":-1}]"#, "non-negative"),
            (r#"[{"kind":"policy","slot":1},{"kind":"policy","slot":1}]"#, "already a route"),
            (r#"[{"kind":"external","slot":1}]"#, "unknown key \"slot\""),
        ] {
            let e = p(text).unwrap_err();
            assert!(e.contains(needle), "{text}: {e}");
        }
    }

    /// A refused START parks the env with the refused episode's route; the next STEP starts it from
    /// the THEN-staged `ep_opp` (with the teams) — the rule a host would otherwise have to replay.
    #[test]
    fn a_parked_env_starts_from_the_then_staged_route() {
        use crate::core::columns::col;
        use crate::core::{Core, OwnedCols, Spec};
        use crate::episode::{Terminal, REFUSE_STARTS};
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../rust_sim/tests/vectors/bridge_corpus");
        let mut teams: Vec<String> = Vec::new();
        let mut files: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        for f in files.iter().filter(|p| p.extension().is_some_and(|x| x == "txt")) {
            for l in std::fs::read_to_string(f).unwrap().lines().filter(|l| l.starts_with("TEAM\t")) {
                let t = l.split('\t').nth(3).unwrap_or("").to_string();
                if !t.is_empty() && !teams.contains(&t) {
                    teams.push(t);
                }
            }
        }
        teams.truncate(2);
        let spec = Spec {
            n: 1,
            threads: 1,
            format_id: "gen3ou".into(),
            names: ["eone".into(), "etwo".into()],
            teams,
            turn_limit: None,
            terminal: Terminal::PRODUCTION,
            refusal_budget: 4,
            bank_dir: None,
            labels: Vec::new(),
            opponents: Routes(vec![Route::External, Route::Policy { slot: 4 }, Route::Policy { slot: 1 }]),
            oracle_reveal: crate::core::spec::Reveal::OFF,
        };
        let mut core = Core::new(spec).unwrap();
        let mut cols = OwnedCols::new(1);
        cols.slice_mut::<u32>(col::EP_TEAM).copy_from_slice(&[0, 1]);
        cols.slice_mut::<u32>(col::EP_SEED).copy_from_slice(&[1, 2, 3, 4]);
        cols.slice_mut::<u32>(col::EP_OPP)[0] = 1;
        let a = cols.addrs();
        core.freeze(a).unwrap();
        REFUSE_STARTS.with(|k| k.set(1));
        assert_eq!(core.dispatch(b'R', a), 0, "{:?}", core.last_error().map(|e| e.json()));
        assert_eq!(cols.slice::<u8>(col::NEED), &[0, 0], "the refused start parks");
        assert_eq!(cols.slice::<u8>(col::REFUSED)[0], 1);
        assert_eq!((cols.slice::<u32>(col::OPP_ROUTE)[0], cols.slice::<i32>(col::OPP_SLOT)[0]), (1, 4), "parked: the refused episode's route");
        cols.slice_mut::<u32>(col::EP_OPP)[0] = 2;
        cols.slice_mut::<i32>(col::ACTION).copy_from_slice(&[-1, -1]);
        assert_eq!(core.dispatch(b'S', a), 0, "{:?}", core.last_error().map(|e| e.json()));
        assert!(cols.slice::<u8>(col::NEED).contains(&1), "the STEP started the parked env");
        assert_eq!((cols.slice::<u32>(col::OPP_ROUTE)[0], cols.slice::<i32>(col::OPP_SLOT)[0]), (2, 1), "the THEN-staged route");
    }

    #[test]
    fn a_bot_route_declares_its_seed() {
        let r = p(r#"[{"kind":"bot","bot":"staller","seed":5},{"kind":"bot","bot":"setup_sweep","seed":9}]"#).unwrap();
        assert_eq!(r.0[0], Route::Bot { kind: Kind::Staller, seed: 5, per_episode: false });
        assert_eq!(r.0[1], Route::Bot { kind: Kind::SetupSweep, seed: 9, per_episode: false });
        assert_eq!(p(&r.to_json()).unwrap(), r);
        assert!(r.is_bot(0) && !r.p2_stall_forfeits(0) && r.slot_of(1) == -1);
        for (text, needle) in [
            (r#"[{"kind":"bot","bot":"nope","seed":1}]"#, "not one this core plays"),
            (r#"[{"kind":"bot","bot":"staller"}]"#, "missing key \"seed\""),
            (r#"[{"kind":"bot","bot":"staller","seed":1,"p_bait":0.5}]"#, "unknown key \"p_bait\""),
            (r#"[{"kind":"bot","bot":"baitbot","seed":1}]"#, "not one this core plays"),
        ] {
            let e = p(text).unwrap_err();
            assert!(e.contains(needle), "{text}: {e}");
        }
        let b = r.build_bots(3);
        assert!(b[0].is_some() && b[1].is_some());
        // distinct streams per env and per stream index (the declared seed rule)
        let seeds: std::collections::BTreeSet<u64> = (0..4).flat_map(|e| (0..2).map(move |k| stream_seed(5, e, k))).collect();
        assert_eq!(seeds.len(), 8);
        assert_eq!(stream_seed(5, 3, 1), stream_seed(5, 3, 1));
        // the Python twin (`rust_env_opponents.bot_stream_seed`) pins the same three values (the third is the
        // mixing function at stream index 2, which no bot draws any more — it still keys the splitmix input)
        assert_eq!(
            [stream_seed(5, 0, 0), stream_seed(11, 2, 1), stream_seed((1u64 << 53) - 1, 47, 2)],
            [4517933670823692284, 5390792918547426617, 17462041922349011332]
        );
    }


    #[test]
    fn a_per_episode_bot_route_reseeds_from_the_battle_seed() {
        let r = p(r#"[{"kind":"bot","bot":"staller","seed":5,"streams":"episode"},{"kind":"bot","bot":"random","seed":7,"streams":"env"}]"#).unwrap();
        assert_eq!(r.0[0], Route::Bot { kind: Kind::Staller, seed: 5, per_episode: true });
        assert_eq!(r.0[1], Route::Bot { kind: Kind::Random, seed: 7, per_episode: false });
        assert_eq!(p(&r.to_json()).unwrap(), r);
        assert!(p(r#"[{"kind":"bot","bot":"staller","seed":5,"streams":"game"}]"#).unwrap_err().contains("streams"));
        let w = [1u32, 2, 3, 4];
        assert_eq!(pack_seed_words(&w), 1 | (2 << 16) | (3 << 32) | (4 << 48));
        let b = r.episode_bot(0, &w).unwrap();
        assert_eq!(b.choice.clone().getrandbits(32), crate::bots::rng::PyRandom::new(episode_stream_seed(5, pack_seed_words(&w), 0)).getrandbits(32));
        assert!(r.episode_bot(1, &w).is_none(), "an env-stream route keeps its bot");
        // the Python twin (`rust_env_opponents.episode_bot_stream_seed`) pins the same values
        assert_eq!([episode_stream_seed(5, pack_seed_words(&w), 0), episode_stream_seed(9, 65535, 1)],
                   [14338025463950524205, 12882590778465766730]);
    }
}
