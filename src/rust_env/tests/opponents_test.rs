//! OPPONENT ROUTING through the real core (M5 Lane E, `crate::opponents`): the staged `ep_opp` is
//! consumed WITH the teams at every start, `opp_route` / `opp_slot` describe the episode the other
//! columns describe (across auto-resets), the columns are thread-count-invariant, and an index
//! outside the declared table is the CALLER's error.

mod common;

use pokesim_env::core::columns::{col, SIDES};
use pokesim_env::core::{Core, OwnedCols};
use pokesim_env::bots::Kind;
use pokesim_env::opponents::{Route, Routes};

fn routes() -> Routes {
    Routes(vec![
        Route::External,
        Route::Policy { slot: 5 },
        Route::Bot { kind: Kind::Staller, seed: 11, per_episode: false },
        Route::Bot { kind: Kind::HeuristicV2, seed: 12, per_episode: false },
        Route::Policy { slot: 2 },
    ])
}

/// The route env `i`'s `k`-th staging writes (a fixed, non-trivial sequence per env).
fn staged_route(i: usize, k: usize) -> u32 {
    ((i * 7 + k * 3 + (k * k) % 5) % 5) as u32
}

struct Run {
    /// Per op: (opp_route, opp_slot, episode) of every env.
    per_op: Vec<(Vec<u32>, Vec<i32>, Vec<u32>)>,
    ended: usize,
}

fn run(n: usize, threads: usize, steps: usize) -> Run {
    let teams = common::corpus_teams();
    let nt = teams.len();
    let mut spec = common::spec(n, threads, teams);
    spec.opponents = routes();
    let mut core = Core::new(spec).expect("core");
    let mut cols = OwnedCols::new(n);
    let mut stage_rng = common::Rng(0xE0E0);
    let mut act_rng = common::Rng(77);
    // `staged[i]` counts env i's stagings; `expect[i][e]` is the route episode e was staged with.
    let mut staged = vec![0usize; n];
    let mut expect: Vec<Vec<u32>> = vec![Vec::new(); n];
    let stage = |cols: &mut OwnedCols, i: usize, staged: &mut Vec<usize>, expect: &mut Vec<Vec<u32>>, rng: &mut common::Rng| {
        common::stage(cols, i, rng, nt);
        let r = staged_route(i, staged[i]);
        cols.slice_mut::<u32>(col::EP_OPP)[i] = r;
        expect[i].push(r);
        staged[i] += 1;
    };
    for i in 0..n {
        stage(&mut cols, i, &mut staged, &mut expect, &mut stage_rng);
    }
    let a = cols.addrs();
    core.freeze(a).expect("freeze");
    assert_eq!(core.dispatch(b'R', a), 0, "{:?}", core.last_error().map(|e| e.json()));
    // The RESET consumed staging 0 of every env; stage episode 1.
    for i in 0..n {
        stage(&mut cols, i, &mut staged, &mut expect, &mut stage_rng);
    }
    let r = routes();
    let mut out = Run { per_op: Vec::new(), ended: 0 };
    let mut check = |cols: &OwnedCols, out: &mut Run, expect: &Vec<Vec<u32>>| {
        let (route, slot, ep) =
            (cols.slice::<u32>(col::OPP_ROUTE).to_vec(), cols.slice::<i32>(col::OPP_SLOT).to_vec(), cols.slice::<u32>(col::EPISODE).to_vec());
        for i in 0..n {
            let e = ep[i] as usize;
            assert_eq!(route[i], expect[i][e], "env {i} episode {e}: opp_route is not the route staged for that episode");
            assert_eq!(slot[i], r.slot_of(route[i]), "env {i}: opp_slot disagrees with the route table");
            if r.is_bot(route[i]) {
                assert_eq!(cols.slice::<u8>(col::NEED)[2 * i + 1], 0, "env {i}: a bot's decision was exposed");
                assert_eq!(cols.slice::<u8>(col::NEED)[2 * i], 1, "env {i}: a bot episode must stop only at a p1 decision");
            }
        }
        out.per_op.push((route, slot, ep));
    };
    check(&cols, &mut out, &expect);
    for _ in 0..steps {
        common::random_actions(&mut cols, &mut act_rng);
        assert_eq!(core.dispatch(b'S', a), 0, "{:?}", core.last_error().map(|e| e.json()));
        check(&cols, &mut out, &expect);
        let done = cols.slice::<u8>(col::DONE).to_vec();
        for i in 0..n {
            if done[i] == 1 {
                out.ended += 1;
                // the auto-reset consumed the staged row: stage the episode after it
                stage(&mut cols, i, &mut staged, &mut expect, &mut stage_rng);
            }
        }
    }
    assert!(core.counters().iter().enumerate().all(|(k, v)| !pokesim_env::core::columns::counter::AFTER_FREEZE.contains(&k) || *v == 0));
    out
}

#[test]
fn the_staged_route_is_consumed_with_the_teams_across_auto_resets() {
    let a = run(6, 1, 500);
    assert!(a.ended >= 12, "too few episodes ended to exercise auto-resets: {}", a.ended);
    let routes_seen: std::collections::BTreeSet<u32> = a.per_op.iter().flat_map(|(r, _, _)| r.iter().copied()).collect();
    assert_eq!(routes_seen.len(), 5, "every declared route must be exercised: {routes_seen:?}");
}

#[test]
fn the_route_columns_are_thread_count_invariant() {
    let a = run(6, 1, 300);
    let b = run(6, 3, 300);
    assert_eq!(a.per_op.len(), b.per_op.len());
    for (k, (x, y)) in a.per_op.iter().zip(&b.per_op).enumerate() {
        assert_eq!(x, y, "op {k}: the route columns differ between T = 1 and T = 3");
    }
}

#[test]
fn a_route_outside_the_table_is_the_callers_error() {
    let teams = common::corpus_teams();
    let mut spec = common::spec(2, 1, teams);
    spec.opponents = routes();
    let mut core = Core::new(spec).unwrap();
    let mut cols = OwnedCols::new(2);
    cols.slice_mut::<u32>(col::EP_TEAM).copy_from_slice(&[0, 1, 1, 0]);
    cols.slice_mut::<u32>(col::EP_SEED).copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    cols.slice_mut::<u32>(col::EP_OPP).copy_from_slice(&[1, 7]);
    let a = cols.addrs();
    core.freeze(a).unwrap();
    let st = core.dispatch(b'R', a);
    assert_eq!(st, 2, "an out-of-table route must be CALLER (2)");
    let e = core.last_error().unwrap().json();
    assert!(e.contains("ep_opp = 7") && e.contains("5 routes"), "{e}");
    let _ = SIDES;
}

#[test]
fn an_unknown_bot_is_refused_at_startup() {
    let e = Routes::from_json(Some(&pokesim::json::Json::parse(r#"[{"kind":"bot","bot":"nobot","seed":1}]"#).unwrap())).unwrap_err();
    assert!(e.contains("nobot"), "{e}");
}
