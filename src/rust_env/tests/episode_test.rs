//! EPISODES AND REWARD over real battles (M5 Lane D, `src/episode.rs`).
//!
//! * **F-L0-6 REGRESSION PIN** (`the_stall_forfeit_is_decided_before_any_feed`): with a low stall
//!   threshold, the forfeit happens at EXACTLY the op whose p1 decision was exposed at `turn >=
//!   turn_limit`, p2 is not fed in that op (the log's last command before `FORCELOSE p1` is the
//!   PREVIOUS op's), and every row the core encodes is EXPOSED (`DECISIONS` moves by exactly the
//!   `need` count of each op). WRONG on revert (the Lane-0 placeholder: `turn > limit` AFTER the
//!   feeds): the forfeit comes one decision late, and the forfeit op encodes rows it then discards.
//! * **Natural endings** (`natural_endings_read_calc_term_trunc`): a finished battle with one side
//!   wiped is `terminated`, its reward is the production indicator, and no end is ever both.

mod common;

use pokesim_env::core::columns::{col, counter};
use pokesim_env::core::{Core, OwnedCols};

struct Run {
    forfeits: usize,
    naturals: usize,
    wins: usize,
}

fn drive(n: usize, turn_limit: Option<u32>, steps: usize, seed: u64) -> Run {
    let teams = common::corpus_teams();
    let nt = teams.len();
    let mut spec = common::spec(n, 1, teams);
    spec.turn_limit = turn_limit;
    let mut core = Core::new(spec).unwrap();
    let mut cols = OwnedCols::new(n);
    let mut stage = common::Rng(0xD0 ^ seed);
    let mut act = common::Rng(seed);
    for i in 0..n {
        common::stage(&mut cols, i, &mut stage, nt);
    }
    let a = cols.addrs();
    core.freeze(a).unwrap();
    assert_eq!(core.dispatch(b'R', a), 0);
    let exposed = |c: &OwnedCols| c.slice::<u8>(col::NEED).iter().map(|&x| x as u64).sum::<u64>();
    assert_eq!(cols.slice::<u64>(col::COUNTERS)[counter::DECISIONS], exposed(&cols), "RESET: every encoded row is exposed");
    let mut run = Run { forfeits: 0, naturals: 0, wins: 0 };
    for _ in 0..steps {
        for i in 0..n {
            common::stage(&mut cols, i, &mut stage, nt);
        }
        common::random_actions(&mut cols, &mut act);
        let before_dec = cols.slice::<u64>(col::COUNTERS)[counter::DECISIONS];
        let need0: Vec<u8> = cols.slice::<u8>(col::NEED).to_vec();
        let turn0: Vec<u32> = cols.slice::<u32>(col::TURN).to_vec();
        let ncmds0: Vec<usize> = (0..n).map(|i| core.inline_env(i).unwrap().log.cmds.len()).collect();
        assert_eq!(core.dispatch(b'S', a), 0, "{:?}", core.last_error().map(|e| e.json()));
        assert_eq!(cols.slice::<u8>(col::REFUSED).iter().sum::<u8>(), 0, "no quarantine expected on the corpus");
        assert_eq!(
            cols.slice::<u64>(col::COUNTERS)[counter::DECISIONS] - before_dec,
            exposed(&cols),
            "F-L0-6: a row was encoded and not exposed (a decision opened, then discarded by the forfeit)"
        );
        for i in 0..n {
            let due = turn_limit.is_some_and(|l| need0[2 * i] == 1 && turn0[i] >= l);
            let done = cols.slice::<u8>(col::DONE)[i] == 1;
            let (term, trunc) = (cols.slice::<u8>(col::TERMINATED)[i], cols.slice::<u8>(col::TRUNCATED)[i]);
            let reward = cols.slice::<f32>(col::REWARD)[i];
            if !done {
                assert!(!due, "env {i}: p1 decided at turn {} >= the threshold and was not forfeited", turn0[i]);
                assert_eq!((term, trunc, reward), (0, 0, 0.0), "env {i}: an outcome on an op that ended nothing");
                continue;
            }
            assert_eq!(term + trunc, 1, "env {i}: an end is exactly one of terminated / truncated");
            let log = &core.inline_env(i).unwrap().prev_log;
            let forfeited = log.cmds.last().is_some_and(|c| c == "FORCELOSE p1");
            assert_eq!(forfeited, due, "env {i}: the forfeit came {} (turn {} before the op)", if due { "LATE" } else { "EARLY" }, turn0[i]);
            if forfeited {
                assert_eq!(log.cmds.len(), ncmds0[i] + 1, "env {i}: the forfeit op fed a CHOOSE as well: {:?}", &log.cmds[ncmds0[i]..]);
                assert_eq!((term, trunc, reward), (0, 1, 0.0), "env {i}: a stall forfeit is truncated and pays 0 (indicator)");
                run.forfeits += 1;
            } else {
                assert!(reward == 0.0 || reward == 1.0, "env {i}: the indicator pays 0 or 1, got {reward}");
                if reward == 1.0 {
                    run.wins += 1;
                }
                run.naturals += 1;
            }
        }
    }
    run
}

#[test]
fn the_stall_forfeit_is_decided_before_any_feed() {
    let r = drive(6, Some(4), 300, 21);
    assert!(r.forfeits >= 20, "only {} forfeits — the threshold was not exercised", r.forfeits);
}

#[test]
fn natural_endings_read_calc_term_trunc() {
    let r = drive(6, None, 600, 5);
    assert!(r.naturals >= 5, "only {} natural endings", r.naturals);
    assert!(r.wins >= 1 && r.wins < r.naturals, "both outcomes should occur: {} wins of {}", r.wins, r.naturals);
}
