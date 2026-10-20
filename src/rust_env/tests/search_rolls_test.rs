//! A matched world's DAMAGE-ROLL table in the playout tool (`Log.rolls`, `pokesim::rolls`; F1 tier A of
//! `designs/endstate/design_world_sampler.md` §3.3.1): the honest builder may accept a world only on
//! damage rolls other than the battle seed's, so the search's own replay of that world's prefix MUST
//! use the same rolls, or it plays a different battle from the one the gate accepted.
//!
//! 1. An empty table replays exactly the plain log.
//! 2. A table naming a site replays that site's roll: the playout's `Game` and a `pokesim` session
//!    armed with the same Force table print the same lines on both sides, and they differ from the
//!    plain log's (non-vacuity), with the dice stream aligned (the same PRNG state at the end).
//! 3. `log_from_json` reads `rolls` and refuses a bad one by name.

mod common;

use std::collections::BTreeMap;

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::rolls::{Mode, RollCtl};
use pokesim::trackers::clock::ClockConfig;
use pokesim_env::search::game::{Game, Log};
use pokesim_env::search::playout::log_from_json;

const SEED: &str = "11,22,33,44";

/// A few turns of corpus teams, each side's first offered move every decision (a switch when forced).
fn base_log() -> Log {
    let teams = common::corpus_teams();
    let log0 = Log {
        format_id: "gen3ou".into(),
        seed: SEED.into(),
        names: [common::NAMES[0].into(), common::NAMES[1].into()],
        teams: [teams[0].clone(), teams[1].clone()],
        cmds: Vec::new(),
        rolls: Vec::new(),
    };
    let mut g = Game::start(&log0, ClockConfig::default()).expect("start");
    for _ in 0..24 {
        if g.is_ended() {
            break;
        }
        for side in 0..2 {
            let pick = g.open(side).map(|o| {
                o.tokens.iter().find(|(_, t)| t.starts_with("move ")).unwrap_or(&o.tokens[0]).0 as i32
            });
            if let Some(a) = pick {
                g.feed(side, a).expect("feed");
            }
        }
    }
    Log { cmds: g.cmds.clone(), ..log0 }
}

/// The same battle in a bare `pokesim` session, the control armed: (p1 lines, p2 lines, PRNG state, control).
fn session(log: &Log, ctl: RollCtl) -> (Vec<String>, Vec<String>, String, RollCtl) {
    let dex = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: log.format_id.clone(),
        seed: Some(log.seed.clone()),
        p1: PlayerOptions { name: log.names[0].clone(), team: PackedTeam(log.teams[0].clone()) },
        p2: PlayerOptions { name: log.names[1].clone(), team: PackedTeam(log.teams[1].clone()) },
    };
    let mut s = BridgeSession::new_construct_turn0(&opts, &dex).expect("session");
    s.set_roll_ctl(Some(ctl)).expect("arm");
    for c in &log.cmds {
        let rest = c.strip_prefix("CHOOSE p").expect("a CHOOSE");
        let (tag, tok) = rest.split_once(' ').unwrap();
        let side = if tag == "1" { 0 } else { 1 };
        s.feed_cmd(Cmd { side, choice: parse_choice(tok).unwrap() }, &dex);
    }
    let st = s.battle_state().unwrap().prng_seed();
    let c = s.roll_ctl().unwrap().clone();
    (s.side_lines(0), s.side_lines(1), st, c)
}

fn game_lines(log: &Log) -> Option<(Vec<String>, Vec<String>)> {
    let g = Game::replay(log, log.cmds.len(), ClockConfig::default()).ok()?;
    Some((g.side_lines(0), g.side_lines(1)))
}

#[test]
fn a_roll_table_replays_in_the_playout_game() {
    let log = base_log();
    let plain = game_lines(&log).expect("the plain log replays");
    let (p1, p2, st0, rec) = session(&log, RollCtl::new(0, Mode::Record));
    assert_eq!((p1, p2), plain.clone(), "an armed Record session is the plain battle");
    assert!(rec.sites.len() >= 3, "non-vacuity: the log has damage rolls ({})", rec.sites.len());
    // the first site whose other roll still replays the whole log and changes what is printed
    let mut done = false;
    for s in &rec.sites {
        let r = (s.r0 + 8) % 16;
        let lg = Log { rolls: vec![(s.n, r)], ..log.clone() };
        let Some(got) = game_lines(&lg) else { continue };
        if got == plain {
            continue;
        }
        let (q1, q2, st, c) = session(&lg, RollCtl::new(0, Mode::Force(BTreeMap::from([(s.n, r)]))));
        assert_eq!(got, (q1, q2), "site {} roll {r}: the playout Game and the armed session agree", s.n);
        assert_eq!(st, st0, "the dice stream stays aligned with the truth's");
        assert_eq!(c.sites[s.n as usize].r, r);
        done = true;
        break;
    }
    assert!(done, "non-vacuity: some site's other roll changes the battle and still replays");
    // an empty table is the plain log
    assert_eq!(game_lines(&Log { rolls: Vec::new(), ..log.clone() }), Some(plain));
}

#[test]
fn log_from_json_reads_and_checks_rolls() {
    let base = r#"{"format_id":"gen3ou","seed":"1,2,3,4","names":["a","b"],"teams":["x","y"],"cmds":[]"#;
    let p = |tail: &str| log_from_json(&Json::parse(&format!("{base}{tail}}}")).unwrap());
    assert_eq!(p("").unwrap().rolls, Vec::<(u32, u8)>::new());
    assert_eq!(p(r#","rolls":null"#).unwrap().rolls, Vec::<(u32, u8)>::new());
    assert_eq!(p(r#","rolls":[[3,15],[0,2]]"#).unwrap().rolls, vec![(3, 15), (0, 2)]);
    for (bad, why) in [
        (r#","rolls":[[0,16]]"#, "not in 0..16"),
        (r#","rolls":[[1,2],[1,3]]"#, "twice"),
        (r#","rolls":[[1]]"#, "[site, roll]"),
        (r#","rolls":"x""#, "must be an array"),
    ] {
        let e = p(bad).expect_err(bad);
        assert!(e.contains(why), "{bad}: {e}");
    }
}
