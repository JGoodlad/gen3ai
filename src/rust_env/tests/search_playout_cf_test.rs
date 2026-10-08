//! The COUNTERFACTUAL playout keys (`gen3_cf_core_playout_v1`, poke-env retirement P6 — the prober's
//! `replay-counterfactual` on the core):
//!
//! 1. **A divergence-turn root** (`at: {"turn": T, "other": "recorded"}`) is the START of turn `T` with
//!    the OTHER side's recorded turn-`T` choice fed: driven by the recorded answers it replays the
//!    recorded battle command for command (turn `T`'s two choices in [other, ours] order) and ends with
//!    its winner; `"other": "policy"` leaves the other side's root decision PENDING.
//! 2. **An in-core bot** (`bot: {side, name, seed}`) is never handed out as pending, and each branch's
//!    every bot command is exactly what a FRESH `Bot` seeded `stream_seed(seed, branch, k)` decides at
//!    that decision of a linear replay — so the bot is asked once per real decision, in order (a
//!    phantom draw would shift its stream and fail this). Reruns are byte-identical.
//! 3. **Stall sides** (`stall.sides`): when both stall sides are open at the limit, p1 forfeits first.
//! 4. **Text** (`text`): `prefix_text` + a branch's `text` is that side's whole protocol, line for line.
//!
//! Plus the refusals, by name.

mod common;

use pokesim::encoder::OBS_DIM;
use pokesim::json::Json;
use pokesim::trackers::clock::ClockConfig;
use pokesim_env::bots::{Bot, Kind};
use pokesim_env::core::columns::{col, ACT};
use pokesim_env::core::{Core, OwnedCols};
use pokesim_env::opponents::stream_seed;
use pokesim_env::search::game::{Game, Log};
use pokesim_env::search::playout::Playouts;

/// Finished episodes of an inline core under the seeded random policy (their input logs).
fn logs(n: usize, steps: usize, seed: u64) -> Vec<Log> {
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
    assert_eq!(core.dispatch(b'R', a), 0);
    let mut out = Vec::new();
    for _ in 0..steps {
        for i in 0..n {
            common::stage(&mut cols, i, &mut st, nt);
        }
        common::random_actions(&mut cols, &mut act);
        assert_eq!(core.dispatch(b'S', a), 0, "{:?}", core.last_error().map(|e| e.json()));
        for i in 0..n {
            if cols.slice::<u8>(col::DONE)[i] == 1 {
                let l = &core.inline_env(i).unwrap().prev_log;
                out.push(Log { format_id: l.format_id.clone(), seed: l.seed.clone(), names: l.names.clone(), teams: l.teams.clone(), cmds: l.cmds.clone() });
            }
        }
    }
    out
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

/// A deterministic policy over the row (a hash, never a model).
fn hashed(row: &[f32], mask: &[u8]) -> i32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in row.iter().step_by(7) {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    let legal: Vec<usize> = (0..ACT).filter(|&a| mask[a] == 1).collect();
    legal[(h % legal.len() as u64) as usize] as i32
}

/// Open `req` and play it to the end; `(root JSON, results JSON, every (branch, side) handed out)`.
fn play(req: &str, policy: &mut dyn FnMut(usize, usize, &[f32], &[u8]) -> i32) -> Result<(Json, Json, Vec<(usize, usize)>), String> {
    let mut p = Playouts::new(ClockConfig::default(), 64);
    let root = p.open(&Json::parse(req).unwrap())?;
    let mut rows = vec![0f32; 128 * OBS_DIM];
    let mut masks = vec![0u8; 128 * ACT];
    let mut who = vec![0u32; 128];
    let mut acts: Vec<i32> = Vec::new();
    let mut seen = Vec::new();
    loop {
        let k = p.step(&acts, &mut rows, &mut masks, &mut who)?;
        if k == 0 {
            break;
        }
        acts = (0..k)
            .map(|i| {
                let (b, s) = (who[i] as usize / 2, who[i] as usize % 2);
                seen.push((b, s));
                policy(b, s, &rows[i * OBS_DIM..(i + 1) * OBS_DIM], &masks[i * ACT..(i + 1) * ACT])
            })
            .collect();
    }
    assert_eq!(p.live(), 0);
    Ok((Json::parse(&root).unwrap(), Json::parse(&p.results()).unwrap(), seen))
}

fn strs(v: &Json) -> Vec<String> {
    v.as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect()
}

fn side_of(c: &str) -> usize {
    if c.starts_with("CHOOSE p1") {
        0
    } else {
        1
    }
}

/// An unforfeited episode and a turn `T` it reaches with at least `margin` turns to spare.
fn episode_and_turn(margin: u32) -> (Log, u32, usize) {
    let eps = logs(2, 300, 11);
    let clock = ClockConfig::default();
    for log in eps.iter().filter(|l| !l.cmds.iter().any(|c| c.starts_with("FORCELOSE"))) {
        let last = Game::replay(log, log.cmds.len(), clock).unwrap().turn(0);
        for t in 4..last.saturating_sub(margin) {
            if let Ok((g, rest)) = Game::replay_to_turn(log, t, clock) {
                assert!(g.at_turn_start(t) && g.open(0).is_some() && g.open(1).is_some());
                return (log.clone(), t, rest);
            }
        }
    }
    panic!("no episode reaches a usable turn");
}

/// The recorded battle with turn `T`'s two choices (at `rest`, `rest + 1`) in [other, ours] order.
fn swapped(log: &Log, rest: usize, ours: usize) -> Vec<String> {
    let mut c = log.cmds.clone();
    if side_of(&c[rest]) == ours {
        c.swap(rest, rest + 1);
    }
    assert!(side_of(&c[rest]) != ours && side_of(&c[rest + 1]) == ours, "turn T's two choices are not adjacent");
    c
}

#[test]
fn a_divergence_turn_root_feeds_the_others_recorded_choice_and_replays_the_recorded_battle() {
    let (log, t, rest) = episode_and_turn(3);
    let clock = ClockConfig::default();
    for ours in [0usize, 1] {
        let cmds = swapped(&log, rest, ours);
        let our_tok = cmds[rest + 1].split_once(' ').unwrap().1.split_once(' ').unwrap().1.to_string();
        let other_tok = cmds[rest].split_once(' ').unwrap().1.split_once(' ').unwrap().1.to_string();
        let lin_log = Log { cmds: cmds.clone(), ..log.clone() };
        let root = Game::replay(&lin_log, rest + 1, clock).unwrap();
        let first = root.open(ours).unwrap().tokens.iter().find(|(_, x)| *x == our_tok).unwrap().0;
        // the recorded answers, read off a linear replay of the swapped log
        let mut idxs: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
        let mut lin = Game::replay(&lin_log, rest + 2, clock).unwrap();
        for c in &cmds[rest + 2..] {
            let s = side_of(c);
            let tok = c.split_once(' ').unwrap().1.split_once(' ').unwrap().1;
            let i = lin.open(s).unwrap().tokens.iter().find(|(_, x)| x == tok).unwrap().0 as i32;
            idxs[s].push(i);
            lin.feed(s, i).unwrap();
        }
        let req = format!(
            "{{\"log\":{},\"at\":{{\"turn\":{t},\"other\":\"recorded\"}},\"side\":\"p{}\",\"actions\":[{first}],\"seeds\":[null],\"stall\":null,\"max_turns\":999,\"keep_cmds\":true}}",
            log_json(&log),
            ours + 1
        );
        let mut next = [0usize; 2];
        let (root_j, res, seen) = play(&req, &mut |_, s, _, _| {
            next[s] += 1;
            idxs[s][next[s] - 1]
        })
        .unwrap();
        assert_eq!(root_j.get("at").and_then(Json::as_f64), Some((rest + 1) as f64), "the resolved command index");
        assert_eq!(root_j.get("other_recorded").and_then(Json::as_str), Some(other_tok.as_str()));
        assert_eq!(root_j.get("other_open").and_then(Json::as_bool), Some(false));
        assert_eq!(root_j.get("turn").and_then(Json::as_f64), Some(t as f64));
        assert!(!seen.is_empty());
        let b0 = &res.get("branches").unwrap().as_array().unwrap()[0];
        assert_eq!(strs(b0.get("cmds").unwrap()), cmds, "p{}: the recorded battle, command for command", ours + 1);
        let want_winner = lin.winner().map_or(Json::Null, |w| Json::Num(w as f64));
        assert_eq!(b0.get("end").unwrap().get("winner"), Some(&want_winner));
        assert_eq!(res.get("at").and_then(Json::as_f64), Some((rest + 1) as f64));
    }
    // "policy" leaves the other side's root decision to the policy: it is handed out first.
    let req = format!(
        "{{\"log\":{},\"at\":{{\"turn\":{t},\"other\":\"policy\"}},\"side\":\"p1\",\"actions\":null,\"seeds\":[null],\"stall\":null,\"max_turns\":999}}",
        log_json(&log)
    );
    let (root_j, _res, seen) = play(&req, &mut |_, _, r, m| hashed(r, m)).unwrap();
    assert_eq!(root_j.get("other_open").and_then(Json::as_bool), Some(true));
    assert_eq!(root_j.get("at").and_then(Json::as_f64), Some(rest as f64));
    assert!(seen.iter().any(|&(_, s)| s == 1));
}

fn bot_req(log: &Log, t: u32, name: &str, seed: u64, seeds: &str, text: bool) -> String {
    format!(
        "{{\"log\":{},\"at\":{{\"turn\":{t},\"other\":\"recorded\"}},\"side\":\"p1\",\"actions\":null,\"seeds\":{seeds},\"stall\":{{\"turn_limit\":250,\"side\":\"p1\"}},\"max_turns\":999,\"keep_cmds\":true,\"bot\":{{\"side\":\"p2\",\"name\":\"{name}\",\"seed\":{seed}}}{}}}",
        log_json(log),
        if text { ",\"text\":\"p1\"" } else { "" }
    )
}

#[test]
fn an_in_core_bot_is_never_pending_and_decides_exactly_at_each_real_decision() {
    let (log, t, rest) = episode_and_turn(3);
    let clock = ClockConfig::default();
    for (name, seed) in [("random", 5u64), ("staller", 9), ("heuristic2", 3)] {
        let req = bot_req(&log, t, name, seed, "[null,\"1,2,3,4\"]", false);
        let (_, res, seen) = play(&req, &mut |_, _, r, m| hashed(r, m)).unwrap();
        assert!(seen.iter().all(|&(_, s)| s == 0), "{name}: the bot's side was handed out as pending");
        let (_, again, _) = play(&req, &mut |_, _, r, m| hashed(r, m)).unwrap();
        assert_eq!(res, again, "{name}: a rerun is byte-identical");
        let kind = Kind::from_name(name).unwrap();
        let mut bot_decisions = 0;
        for (b, br) in res.get("branches").unwrap().as_array().unwrap().iter().enumerate() {
            let cmds = strs(br.get("cmds").unwrap());
            let reseed = br.get("reseed").and_then(Json::as_str).map(str::to_string);
            // a linear replay of the branch: the root (rest + 1 commands), its dice, then its commands
            let lin_log = Log { cmds: cmds.clone(), ..log.clone() };
            let mut g = Game::replay(&lin_log, rest + 1, clock).unwrap().branch(reseed.as_deref());
            let mut fresh = Bot::new(kind, stream_seed(seed, b, 0), stream_seed(seed, b, 1));
            for c in &cmds[rest + 1..] {
                if let Some(rest_c) = c.strip_prefix("FORCELOSE p") {
                    g.forfeit(if rest_c == "1" { 0 } else { 1 }).unwrap();
                    continue;
                }
                let s = side_of(c);
                let tok = c.split_once(' ').unwrap().1.split_once(' ').unwrap().1;
                if s == 1 {
                    let o = g.open(1).unwrap();
                    let d = fresh.decide(g.reading(1), &o.tokens).unwrap();
                    assert_eq!(d.token, tok, "{name} branch {b}: the bot's command is a fresh bot's decision at that decision");
                    bot_decisions += 1;
                    g.feed_token(1, tok).unwrap();
                } else {
                    let i = g.open(0).unwrap().tokens.iter().find(|(_, x)| x == tok).unwrap().0 as i32;
                    g.feed(0, i).unwrap();
                }
            }
            assert!(g.is_ended(), "{name} branch {b}: the branch's commands end its battle");
            assert_eq!(br.get("decisions").unwrap().as_array().unwrap()[1].as_f64().unwrap() as usize > 0, true);
        }
        assert!(bot_decisions > 0, "{name}: the bot decided nothing");
    }
    // Teeth: a bot seeded differently plays (at least one) different command somewhere.
    let a = play(&bot_req(&log, t, "random", 5, "[null]", false), &mut |_, _, r, m| hashed(r, m)).unwrap().1;
    let b = play(&bot_req(&log, t, "random", 6, "[null]", false), &mut |_, _, r, m| hashed(r, m)).unwrap().1;
    assert_ne!(a, b, "the bot's seed must matter (random draws every decision)");
}

#[test]
fn stall_sides_forfeit_p1_first_and_text_is_the_whole_protocol() {
    let (log, t, rest) = episode_and_turn(4);
    let clock = ClockConfig::default();
    let stalled = |stall: &str| {
        let req = format!(
            "{{\"log\":{},\"at\":{{\"turn\":{t},\"other\":\"recorded\"}},\"side\":\"p1\",\"actions\":null,\"seeds\":[null],\"stall\":{stall},\"max_turns\":999}}",
            log_json(&log)
        );
        play(&req, &mut |_, _, r, m| hashed(r, m)).unwrap().1
    };
    let lim = t + 1;
    for (stall, loser) in [
        (format!("{{\"turn_limit\":{lim},\"sides\":[\"p2\",\"p1\"]}}"), 0usize),
        (format!("{{\"turn_limit\":{lim},\"side\":\"p2\"}}"), 1),
        (format!("{{\"turn_limit\":{lim},\"side\":\"p1\"}}"), 0),
    ] {
        let res = stalled(&stall);
        for br in res.get("branches").unwrap().as_array().unwrap() {
            let e = br.get("end").unwrap();
            assert_eq!(e.get("forfeit").and_then(Json::as_bool), Some(true), "{stall}");
            assert_eq!(e.get("winner").and_then(Json::as_f64), Some((1 - loser) as f64), "{stall}");
        }
    }
    // text: prefix_text + a branch's text == the side's lines of a linear replay of the branch
    let req = bot_req(&log, t, "heuristic", 2, "[null]", true);
    let (_, res, _) = play(&req, &mut |_, _, r, m| hashed(r, m)).unwrap();
    let br = &res.get("branches").unwrap().as_array().unwrap()[0];
    let cmds = strs(br.get("cmds").unwrap());
    let mut whole = strs(res.get("prefix_text").unwrap());
    whole.extend(strs(br.get("text").unwrap()));
    let lin = Game::replay(&Log { cmds: cmds.clone(), ..log.clone() }, cmds.len(), clock).unwrap();
    assert_eq!(whole, lin.side_lines(0), "the narrated text is the side's whole protocol");
    assert!(whole.iter().any(|l| l.starts_with("|turn|")) && rest > 0);
}

#[test]
fn the_counterfactual_keys_are_refused_by_name() {
    let (log, t, _) = episode_and_turn(3);
    let base = |extra: &str, at: &str| {
        format!(
            "{{\"log\":{},\"at\":{at},\"side\":\"p1\",\"actions\":null,\"seeds\":[null],\"stall\":null,\"max_turns\":999{extra}}}",
            log_json(&log)
        )
    };
    let turn = format!("{{\"turn\":{t},\"other\":\"recorded\"}}");
    for (req, needle) in [
        (base(",\"bot\":{\"side\":\"p1\",\"name\":\"random\",\"seed\":1}", &turn), "SEARCHED side"),
        (base(",\"bot\":{\"side\":\"p2\",\"name\":\"nope\",\"seed\":1}", &turn), "not one this core plays"),
        (base(",\"bot\":{\"side\":\"p2\",\"name\":\"random\",\"seed\":1,\"x\":1}", &turn), "unknown key"),
        (base(",\"bot\":{\"side\":\"p2\",\"name\":\"random\",\"seed\":-1}", &turn), "bot.seed"),
        (base(",\"text\":\"p3\"", &turn), "`text`"),
        (base("", &format!("{{\"turn\":{t},\"other\":\"maybe\"}}")), "at.other"),
        (base("", &format!("{{\"turn\":{t},\"other\":\"recorded\",\"x\":1}}")), "unknown key"),
        (base("", "{\"turn\":990,\"other\":\"recorded\"}"), "never reaches"),
        (base("", "\"x\""), "command index"),
    ] {
        let mut p = Playouts::new(ClockConfig::default(), 64);
        let e = p.open(&Json::parse(&req).unwrap()).unwrap_err();
        assert!(e.contains(needle), "{needle}: {e}");
    }
    let mut p = Playouts::new(ClockConfig::default(), 64);
    for (stall, needle) in [
        ("{\"turn_limit\":9,\"side\":\"p1\",\"sides\":[\"p1\"]}", "exactly one of"),
        ("{\"turn_limit\":9,\"sides\":[]}", "empty"),
        ("{\"turn_limit\":9,\"sides\":[\"p3\"]}", "stall.sides"),
    ] {
        let req = format!(
            "{{\"log\":{},\"at\":{turn},\"side\":\"p1\",\"actions\":null,\"seeds\":[null],\"stall\":{stall},\"max_turns\":999}}",
            log_json(&log)
        );
        let e = p.open(&Json::parse(&req).unwrap()).unwrap_err();
        assert!(e.contains(needle), "{needle}: {e}");
    }
}
