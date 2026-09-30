//! M5 Lane I — the playout's battle ([`Game`]) IS the env core's row path, and its BRANCH is exact.
//!
//! 1. **Game == the env core.** An inline core plays the bridge corpus with the seeded random policy
//!    (gate ①'s shape); every decision it exposes is recorded (side, `dec_n`, row bytes, mask). Each
//!    finished episode's INPUT LOG is then replayed through a `Game` from turn 0: the decisions it
//!    opens, in order, must equal the core's — row BYTES, mask and ordinal.
//! 2. **The branch is exact.** From a replay of the same log to command `k`, `Game::branch(None)`
//!    (the battle's own dice) fed the log's remaining commands reproduces the same decisions from `k`
//!    on, byte for byte, for every sampled `k`.
//! 3. **Playouts are a function of their inputs**, with COMMON RANDOM NUMBERS: two runs of the same
//!    playout under a deterministic policy are byte-identical; two branches that share (action,
//!    seed) end identically; the recorded continuation (seed null, the recorded first action, a
//!    policy answering the recorded commands) ends the way the recorded episode ended.
//!
//! Teeth: a mutated row byte in the recorded list fails (1).

mod common;

use pokesim::encoder::OBS_DIM;
use pokesim::json::Json;
use pokesim_env::core::columns::{col, ACT, SIDES};
use pokesim_env::core::{Core, OwnedCols};
use pokesim_env::search::game::{Game, Log};
use pokesim_env::search::playout::Playouts;
use pokesim::trackers::clock::ClockConfig;

type Dec = (usize, u32, Vec<u8>, Vec<u8>); // side, ordinal, row bytes, mask

fn row_bytes(r: &[f32]) -> Vec<u8> {
    r.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Run an inline core; return every FINISHED episode as (its input log, its decisions in order).
fn record(n: usize, steps: usize, seed: u64) -> Vec<(Log, Vec<Dec>)> {
    let teams = common::corpus_teams();
    let nt = teams.len();
    let mut core = Core::new(common::spec(n, 1, teams)).unwrap();
    let mut cols = OwnedCols::new(n);
    let mut st = common::Rng(0x0B5E ^ seed);
    let mut act = common::Rng(seed);
    for i in 0..n {
        common::stage(&mut cols, i, &mut st, nt);
    }
    let a = cols.addrs();
    core.freeze(a).unwrap();
    let mut cur: Vec<Vec<Dec>> = vec![Vec::new(); n];
    let mut done_eps = Vec::new();
    let collect = |cols: &OwnedCols, i: usize, into: &mut Vec<Dec>| {
        for s in 0..SIDES {
            let k = i * SIDES + s;
            if cols.slice::<u8>(col::NEED)[k] == 1 {
                let row = &cols.slice::<f32>(col::OBS)[k * OBS_DIM..(k + 1) * OBS_DIM];
                let m = cols.slice::<u8>(col::MASK)[k * ACT..(k + 1) * ACT].to_vec();
                into.push((s, cols.slice::<u32>(col::DEC_N)[k], row_bytes(row), m));
            }
        }
    };
    assert_eq!(core.dispatch(b'R', a), 0);
    for (i, c) in cur.iter_mut().enumerate() {
        collect(&cols, i, c);
    }
    for _ in 0..steps {
        for i in 0..n {
            common::stage(&mut cols, i, &mut st, nt);
        }
        common::random_actions(&mut cols, &mut act);
        assert_eq!(core.dispatch(b'S', a), 0, "{:?}", core.last_error().map(|e| e.json()));
        for i in 0..n {
            if cols.slice::<u8>(col::DONE)[i] == 1 {
                assert_eq!(cols.slice::<u8>(col::REFUSED)[i], 0, "no quarantine expected on the bridge corpus");
                let l = &core.inline_env(i).unwrap().prev_log;
                let log = Log { format_id: l.format_id.clone(), seed: l.seed.clone(), names: l.names.clone(), teams: l.teams.clone(), cmds: l.cmds.clone() };
                done_eps.push((log, std::mem::take(&mut cur[i])));
            }
            collect(&cols, i, &mut cur[i]);
        }
    }
    done_eps
}

/// Every decision `g` opens from here as the log's commands `cmds[from..]` are fed.
fn walk(mut g: Game, cmds: &[String], mut seen: [Option<u32>; 2]) -> Vec<Dec> {
    let mut out = Vec::new();
    let note = |g: &Game, seen: &mut [Option<u32>; 2], out: &mut Vec<Dec>| {
        for s in 0..2 {
            if let Some(o) = g.open(s) {
                if seen[s] != Some(o.n) {
                    seen[s] = Some(o.n);
                    let mut row = [0f32; OBS_DIM];
                    let mut m = [0u8; ACT];
                    g.encode(s, &mut row, &mut m).unwrap();
                    out.push((s, o.n, row_bytes(&row), m.to_vec()));
                }
            }
        }
    };
    note(&g, &mut seen, &mut out);
    for c in cmds {
        if let Some(rest) = c.strip_prefix("FORCELOSE p") {
            g.forfeit(if rest == "1" { 0 } else { 1 }).unwrap();
        } else {
            let rest = c.strip_prefix("CHOOSE p").unwrap();
            let (tag, tok) = rest.split_once(' ').unwrap();
            let s = if tag == "1" { 0 } else { 1 };
            let idx = g.open(s).unwrap().tokens.iter().find(|(_, t)| t == tok).unwrap_or_else(|| panic!("{c} not offered")).0;
            g.feed(s, idx as i32).unwrap();
        }
        note(&g, &mut seen, &mut out);
    }
    assert!(g.is_ended(), "the log's commands must end its battle");
    out
}

/// The ordering of `walk` and of the core differ only between sides WITHIN one write: sort each
/// list by (ordinal, side) — each side's ordinals are 0, 1, 2, … in feed order on both.
fn canon(mut v: Vec<Dec>) -> Vec<Dec> {
    v.sort_by_key(|d| (d.1, d.0));
    v
}

#[test]
fn a_game_is_the_env_cores_row_path_and_its_branch_is_exact() {
    let eps = record(4, 300, 7);
    assert!(eps.len() >= 4, "only {} finished episodes", eps.len());
    let clock = ClockConfig::default();
    let mut n_dec = 0;
    let mut n_branch = 0;
    for (ei, (log, decs)) in eps.iter().enumerate() {
        let got = canon(walk(Game::start(log, clock).unwrap(), &log.cmds, [None, None]));
        let want = canon(decs.clone());
        assert_eq!(got.len(), want.len(), "episode {ei}: decision count");
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g == w, "episode {ei}: decision {k} (p{} #{}) differs from the core's", w.0 + 1, w.1);
        }
        n_dec += got.len();
        // (2) a branch at every 9th command, the battle's own dice, fed the rest of the log.
        for k in (0..log.cmds.len()).step_by(9) {
            let g = Game::replay(log, k, clock).unwrap();
            let seen = [g.open(0).map(|o| o.n), g.open(1).map(|o| o.n)];
            let base = walk(g.branch(None), &log.cmds[k..], seen);
            let lin = walk(Game::replay(log, k, clock).unwrap(), &log.cmds[k..], seen);
            assert!(base == lin, "episode {ei}: a branch at command {k} diverged from the linear replay");
            // (with (1), the branch is therefore the core's continuation too)
            n_branch += 1;
        }
    }
    // Teeth: one flipped byte of one recorded row fails the comparison.
    let (log, decs) = &eps[0];
    let mut bad = canon(decs.clone());
    bad[3].2[17] ^= 1;
    assert!(canon(walk(Game::start(log, clock).unwrap(), &log.cmds, [None, None])) != bad);
    eprintln!("search_game: {} episodes, {n_dec} decisions byte-equal, {n_branch} branches exact", eps.len());
}

/// A deterministic "greedy" policy: the legal action whose index maximises a hash of the row.
fn greedy(row: &[f32], mask: &[u8]) -> i32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in row.iter().step_by(7) {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    let legal: Vec<usize> = (0..ACT).filter(|&a| mask[a] == 1).collect();
    legal[(h % legal.len() as u64) as usize] as i32
}

fn log_json(log: &Log) -> String {
    let q = pokesim::search::json_quote;
    let arr = |v: &[String]| format!("[{}]", v.iter().map(|s| q(s)).collect::<Vec<_>>().join(","));
    format!(
        "{{\"format_id\":{},\"seed\":{},\"names\":{},\"teams\":{},\"cmds\":{}}}",
        q(&log.format_id),
        q(&log.seed),
        arr(&log.names),
        arr(&log.teams),
        arr(&log.cmds)
    )
}

/// Play one root to the end under `policy`; the results JSON.
fn play(req: &str, policy: &mut dyn FnMut(usize, usize, &[f32], &[u8]) -> i32) -> String {
    let mut p = Playouts::new(ClockConfig::default(), 64);
    p.open(&Json::parse(req).unwrap()).unwrap();
    let mut rows = vec![0f32; 128 * OBS_DIM];
    let mut masks = vec![0u8; 128 * ACT];
    let mut who = vec![0u32; 128];
    let mut acts: Vec<i32> = Vec::new();
    loop {
        let k = p.step(&acts, &mut rows, &mut masks, &mut who).unwrap();
        if k == 0 {
            break;
        }
        acts = (0..k)
            .map(|i| policy(who[i] as usize / 2, who[i] as usize % 2, &rows[i * OBS_DIM..(i + 1) * OBS_DIM], &masks[i * ACT..(i + 1) * ACT]))
            .collect();
    }
    assert_eq!(p.live(), 0);
    p.results()
}

#[test]
fn playouts_are_deterministic_with_common_random_numbers_and_the_recorded_continuation_is_reproduced() {
    let eps = record(2, 250, 3);
    let (log, _) = eps.iter().find(|(l, _)| l.cmds.len() > 30 && !l.cmds.iter().any(|c| c.starts_with("FORCELOSE"))).expect("an episode");
    // a p1 decision in the middle of the battle
    let at = (10..log.cmds.len()).find(|&k| log.cmds[k].starts_with("CHOOSE p1")).unwrap();
    let lj = log_json(log);
    let req = format!(
        "{{\"log\":{lj},\"at\":{at},\"side\":\"p1\",\"actions\":null,\"seeds\":[\"sodium,00000000000000000000000000000001\",\"sodium,00000000000000000000000000000001\",\"1,2,3,4\"],\"stall\":{{\"turn_limit\":250,\"side\":\"p1\"}},\"max_turns\":400,\"keep_cmds\":true}}"
    );
    let a = play(&req, &mut |_, _, r, m| greedy(r, m));
    let b = play(&req, &mut |_, _, r, m| greedy(r, m));
    assert_eq!(a, b, "a playout is a function of its inputs");
    let v = Json::parse(&a).unwrap();
    let br = v.get("branches").unwrap().as_array().unwrap();
    assert!(br.len() >= 3 && br.len() % 3 == 0);
    let mut differ = false;
    for t in br.chunks(3) {
        // seeds 0 and 1 are the same seed: CRN makes the two branches ONE outcome, cmds included
        assert_eq!(t[0].get("end"), t[1].get("end"));
        assert_eq!(t[0].get("cmds"), t[1].get("cmds"));
        differ |= t[0].get("cmds") != t[2].get("cmds");
        assert!(t[0].get("end").unwrap().get("turn").is_some(), "every branch ended");
    }
    assert!(differ, "another seed must change at least one continuation (the dice matter)");
    // The recorded continuation: seed null, the recorded first action, a policy answering the log.
    let first_tok = log.cmds[at].strip_prefix("CHOOSE p1 ").unwrap();
    let g = Game::replay(log, at, ClockConfig::default()).unwrap();
    let first = g.open(0).unwrap().tokens.iter().find(|(_, t)| t == first_tok).unwrap().0;
    let req = format!(
        "{{\"log\":{lj},\"at\":{at},\"side\":\"p1\",\"actions\":[{first}],\"seeds\":[null],\"stall\":null,\"max_turns\":999,\"keep_cmds\":true}}"
    );
    // The policy answers each side with its next recorded action INDEX (read off a linear replay).
    let mut idxs: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
    let mut lin = Game::replay(log, at + 1, ClockConfig::default()).unwrap();
    for c in &log.cmds[at + 1..] {
        let (tag, tok) = c.strip_prefix("CHOOSE p").unwrap().split_once(' ').unwrap();
        let s = if tag == "1" { 0 } else { 1 };
        let i = lin.open(s).unwrap().tokens.iter().find(|(_, t)| t == tok).unwrap().0 as i32;
        idxs[s].push(i);
        lin.feed(s, i).unwrap();
    }
    let mut next = [0usize; 2];
    let out = play(&req, &mut |_, s, _, _| {
        next[s] += 1;
        idxs[s][next[s] - 1]
    });
    let v = Json::parse(&out).unwrap();
    let b0 = &v.get("branches").unwrap().as_array().unwrap()[0];
    let cmds: Vec<String> = b0.get("cmds").unwrap().as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
    assert_eq!(cmds, log.cmds, "the recorded continuation replays the recorded battle, command for command");
}
