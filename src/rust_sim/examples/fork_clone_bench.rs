//! `fork_clone_bench` — what a FORK of a paused battle costs to clone, and how much of it the
//! TRANSPORT's growing history (`script` + `request_seeds`) is (`gen3_core_engine_split_v1`, the Rust
//! Core Program's pre-M3 hand-off; the record is
//! `designs/research_state/measurements/rust_core_m3_2026-09-24/`).
//!
//! stdin: a `sim_bridge` transcript (`START <json>` / `CHOOSE <p1|p2> <choice>` / `END`), e.g. the
//! M1 transport record's `transcript_env_seed0.txt`. Every battle is replayed through ONE
//! `BridgeSession`; at every decision boundary the three clones below are timed `REPS` times each
//! (the median per boundary is kept) — then the per-boundary medians are summarised:
//!
//! * `snapshot`      — `BridgeSession::snapshot()`, the whole session (what a fork cloned before the
//!                     split, less the chunk history a compacted version had already dropped: the
//!                     session is `clear_chunks`ed before every sample, as a search node's is);
//! * `script_seeds`  — `script` + `request_seeds` alone, the part the split takes OFF the fork;
//! * `engine`        — (after the split only) `Engine::clone`, what a version's fork clones now;
//! * `resume`        — (after the split only) `BridgeSession::resume(engine.clone())`, the whole
//!                     fork a version drives (`BattleVersion::fork_session`): the engine clone plus
//!                     a fresh transport sharing the engine's issued request bytes.
//!
//! Output: one JSON line — per arm the median / p90 / max over boundaries of the per-boundary
//! median, in ns, plus the boundary count and the mean script / seed lengths at a boundary.
//!
//!     cargo run --release --example fork_clone_bench -- [REPS] < transcript.txt

use std::hint::black_box;
use std::io::{self, BufRead};
use std::time::Instant;

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::json::Json;

fn time_ns(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<u128> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        v.push(t.elapsed().as_nanos());
    }
    v.sort_unstable();
    v[v.len() / 2] as f64
}

fn summary(mut v: Vec<f64>) -> String {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() as f64 - 1.0) * p).round() as usize];
    format!("{{\"median\":{:.0},\"p90\":{:.0},\"max\":{:.0}}}", q(0.5), q(0.9), v[v.len() - 1])
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(201);
    let dex = Dex::for_gen(3);
    let mut snap: Vec<f64> = Vec::new();
    let mut part: Vec<f64> = Vec::new();
    let mut eng: Vec<f64> = Vec::new();
    let mut res: Vec<f64> = Vec::new();
    let (mut script_len, mut seed_len) = (0usize, 0usize);
    let mut sess: Option<BridgeSession> = None;
    let sample = |s: &BridgeSession, snap: &mut Vec<f64>, part: &mut Vec<f64>, eng: &mut Vec<f64>, res: &mut Vec<f64>| {
        let mut c = s.snapshot();
        c.clear_chunks();
        snap.push(time_ns(reps, || {
            black_box(c.snapshot());
        }));
        part.push(time_ns(reps, || {
            black_box((c.script().to_vec(), c.request_seeds().to_vec()));
        }));
        eng.push(time_ns(reps, || {
            black_box(c.engine().clone());
        }));
        res.push(time_ns(reps, || {
            black_box(BridgeSession::resume(c.engine().clone()));
        }));
    };
    for line in io::stdin().lock().lines() {
        let line = line.expect("stdin");
        let (cmd, rest) = line.split_once(' ').unwrap_or((line.as_str(), ""));
        match cmd {
            "START" => {
                let v = Json::parse(rest).expect("START json");
                let p = |k: &str| PlayerOptions {
                    name: v.get(k).and_then(|p| p.str_at("name")).unwrap().to_string(),
                    team: PackedTeam(v.get(k).and_then(|p| p.str_at("team")).unwrap().to_string()),
                };
                let seed = v
                    .get("seed")
                    .and_then(|s| s.as_array())
                    .map(|a| a.iter().map(|x| format!("{}", x.as_f64().unwrap() as u64)).collect::<Vec<_>>().join(","));
                let opts = BattleOptions {
                    format_id: v.str_at("formatid").unwrap().to_string(),
                    seed,
                    p1: p("p1"),
                    p2: p("p2"),
                };
                sess = Some(BridgeSession::new_construct_turn0(&opts, &dex).expect("session"));
            }
            "CHOOSE" => {
                let s = sess.as_mut().expect("CHOOSE before START");
                let (side, choice) = rest.split_once(' ').unwrap();
                let side = if side == "p1" { 0 } else { 1 };
                let before = s.script().len();
                s.feed_cmd(Cmd { side, choice: parse_choice(choice).expect("choice") }, &dex);
                if s.script().len() > before && !s.is_ended() {
                    script_len += s.script().len();
                    seed_len += s.request_seeds().len();
                    sample(s, &mut snap, &mut part, &mut eng, &mut res);
                }
            }
            _ => {}
        }
    }
    let n = snap.len();
    let mut o = format!(
        "{{\"reps\":{reps},\"boundaries\":{n},\"mean_script_len\":{:.1},\"mean_seed_len\":{:.1},\"snapshot_ns\":{},\"script_seeds_ns\":{}",
        script_len as f64 / n as f64,
        seed_len as f64 / n as f64,
        summary(snap),
        summary(part)
    );
    o.push_str(&format!(",\"engine_ns\":{},\"resume_ns\":{}", summary(eng), summary(res)));
    o.push('}');
    println!("{o}");
}
