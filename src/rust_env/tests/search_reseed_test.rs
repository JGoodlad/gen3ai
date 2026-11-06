//! A leak-honest world's RESEED SCHEDULE in the playout tool (`Log.reseed`; `pokesim::fresh_dice`,
//! `designs/endstate/design_sprime_honest.md` D3, B1). The honest builder keeps a world whose prefix
//! reproduces the searched side's lines on FRESH dice, the PRNG replaced before each resolving command;
//! the search's own replay of that world MUST replace it at the same commands, or it plays another
//! battle than the one the matcher kept.
//!
//! 1. A log whose schedule reseeds at command k replays exactly as the battle branched there onto that
//!    seed (`Game::branch(Some(seed))`, the clone-and-branch primitive): every later line, both sides.
//!    REVERT TOOTH: the same log with its schedule dropped plays another battle.
//! 2. An empty schedule is the plain log, byte for byte.
//! 3. `log_from_json` reads `reseed` and refuses a bad one by name.

mod common;

use pokesim::json::Json;
use pokesim::trackers::clock::ClockConfig;
use pokesim_env::search::game::{Game, Log};
use pokesim_env::search::playout::log_from_json;

const SEED: &str = "11,22,33,44";
const FRESH: &str = "sodium,0f1e2d3c4b5a69788796a5b4c3d2e1f0";

fn pick(g: &Game, side: usize) -> Option<i32> {
    g.open(side).map(|o| o.tokens.iter().find(|(_, t)| t.starts_with("move ")).unwrap_or(&o.tokens[0]).0 as i32)
}

fn play(g: &mut Game, turns: usize) {
    for _ in 0..turns {
        if g.is_ended() {
            break;
        }
        for side in 0..2 {
            if let Some(a) = pick(g, side) {
                g.feed(side, a).expect("feed");
            }
        }
    }
}

/// `turns` turns on the battle's own seed, then the battle BRANCHED onto `FRESH` at that command and
/// played on: (the whole log with its schedule, the branch point, the branch).
fn branched(turns: usize) -> (Log, usize, Game) {
    let teams = common::corpus_teams();
    let log0 = Log {
        format_id: "gen3ou".into(),
        seed: SEED.into(),
        names: [common::NAMES[0].into(), common::NAMES[1].into()],
        teams: [teams[0].clone(), teams[1].clone()],
        cmds: Vec::new(),
        rolls: Vec::new(),
        reseed: Vec::new(),
    };
    let mut g = Game::start(&log0, ClockConfig::default()).expect("start");
    play(&mut g, turns);
    let k = g.cmds.len();
    let mut b = g.branch(Some(FRESH));
    play(&mut b, 12);
    assert!(b.cmds.len() > k + 4, "non-vacuity: the branch plays on");
    (Log { cmds: b.cmds.clone(), reseed: vec![(k, FRESH.to_string())], ..log0 }, k, b)
}

#[test]
fn a_reseed_schedule_replays_the_branched_battle() {
    let (log, k, b) = branched(6);
    let pre = Game::replay(&log, k, ClockConfig::default()).expect("the prefix");
    let full = Game::replay(&log, log.cmds.len(), ClockConfig::default()).expect("the scheduled log replays");
    for side in 0..2 {
        let suffix: Vec<String> = full.side_lines(side)[pre.side_lines(side).len()..].to_vec();
        assert_eq!(suffix, b.side_lines(side), "p{}: the schedule's replay is the branch, line for line", side + 1);
    }
    // the REVERT TOOTH: without its schedule the log is another battle (or does not replay at all)
    let plain = Log { reseed: Vec::new(), ..log.clone() };
    let other = Game::replay(&plain, plain.cmds.len(), ClockConfig::default())
        .map(|g| (g.side_lines(0), g.side_lines(1)))
        .ok();
    assert_ne!(other, Some((full.side_lines(0), full.side_lines(1))), "a dropped schedule must change the battle");
}

#[test]
fn an_empty_schedule_is_the_plain_log() {
    let (log, _k, _b) = branched(4);
    let a = Game::replay(&Log { reseed: Vec::new(), ..log.clone() }, 6, ClockConfig::default()).expect("plain");
    let b = Game::replay(&Log { reseed: vec![(9_999_999, FRESH.into())], ..log.clone() }, 6, ClockConfig::default());
    let b = b.expect("a schedule entry past `at` is never reached");
    assert_eq!((a.side_lines(0), a.side_lines(1)), (b.side_lines(0), b.side_lines(1)));
}

fn parsed(reseed: &str) -> Result<Log, String> {
    let teams = common::corpus_teams();
    let doc = format!(
        "{{\"format_id\":\"gen3ou\",\"seed\":\"{SEED}\",\"names\":[\"a\",\"b\"],\"teams\":[{},{}],\
         \"cmds\":[\"CHOOSE p1 move 1\",\"CHOOSE p2 move 1\",\"CHOOSE p1 move 1\"],\"reseed\":{reseed}}}",
        pokesim::search::json_quote(&teams[0]),
        pokesim::search::json_quote(&teams[1])
    );
    log_from_json(&Json::parse(&doc).expect("json"))
}

#[test]
fn log_from_json_reads_and_refuses_reseed() {
    assert_eq!(parsed("null").expect("null").reseed, Vec::new());
    assert_eq!(parsed(&format!("[[1,\"{FRESH}\"]]")).expect("ok").reseed, vec![(1, FRESH.to_string())]);
    for (bad, why) in [
        (format!("[[2,\"{FRESH}\"],[1,\"{FRESH}\"]]"), "ascend"),
        (format!("[[1,\"{FRESH}\"],[1,\"{FRESH}\"]]"), "ascend"),
        (format!("[[3,\"{FRESH}\"]]"), "past the log"),
        ("[[0,\"nonsense\"]]".to_string(), "seed"),
        ("[[0]]".to_string(), "[k, seed]"),
    ] {
        let e = parsed(&bad).expect_err(&bad);
        assert!(e.contains(why), "{bad}: {e}");
    }
}
