//! GATE ① — the core's rows are `sim_bridge`'s `__OBS__` rows, BYTE for byte, on the same input
//! log (F-M5-5; M5 Lane 0).
//!
//! The core plays battles between the bridge corpus's real teams under a seeded random policy
//! (one inline pool, both sides' rows out). Every row it writes is recorded with its mask and the
//! mapper's tokens, keyed `(env, episode, side, n)`. Then every episode's INPUT LOG (the finished
//! ones from `prev_log`, the ones still running from `log`) is replayed through THIS checkout's
//! `sim_bridge` with `core_obs` on for both sides and the same clock flags, and every `__OBS__`
//! frame is compared: the base64 frame decoded to bytes == the core's row bytes, the mask equal,
//! the tokens equal, one frame per recorded decision and none extra, no NaN.
//!
//! The binary is `$POKESIM_SIM_BRIDGE_BIN` — REQUIRED (`src/utils/rust_env/core_cargo_test.py`
//! builds the port's self-check `sim_bridge` in this checkout and sets it). A missing binary FAILS:
//! a gate nobody can start is indistinguishable from a gate that passes.

mod common;

use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

use pokesim::core_events::jsonval::Val;
use pokesim_env::core::columns::{col, ACT, OBS_DIM, SIDES};
use pokesim_env::core::refusal::InputLog;
use pokesim_env::core::{Core, OwnedCols};

const N: usize = 6;
const STEPS: usize = 400;

struct Rec {
    row: Vec<u8>,
    mask: Vec<u8>,
    tokens: String,
}

fn b64_decode(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("not base64: {c}"),
        }
    };
    let b: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(b.len() * 3 / 4);
    for ch in b.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in ch.iter().enumerate() {
            n |= val(c) << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if ch.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if ch.len() > 3 {
            out.push(n as u8);
        }
    }
    out
}

fn sim_bridge_frames(bin: &str, log: &InputLog) -> Vec<(usize, Val)> {
    sim_bridge_frames_with(bin, log, false)
}

/// `decision_tense` is the clock flag handed to `sim_bridge` (the core ran with `false`): the teeth
/// test replays with `true` and must see rows differ.
fn sim_bridge_frames_with(bin: &str, log: &InputLog, decision_tense: bool) -> Vec<(usize, Val)> {
    let extra = format!(
        "\"core_obs\":{{\"sides\":[\"p1\",\"p2\"],\"decision_tense\":{decision_tense},\"switch_freeze\":false}},"
    );
    let script = log.script(&extra) + "END\n";
    let mut child = Command::new(bin).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().expect("spawn sim_bridge");
    let mut sin = child.stdin.take().unwrap();
    let w = std::thread::spawn(move || sin.write_all(script.as_bytes()));
    let out = child.wait_with_output().unwrap();
    w.join().unwrap().unwrap();
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(!text.contains("__ERR__"), "sim_bridge refused the core's input log:\n{}", log.script(""));
    text.lines()
        .filter_map(|l| l.strip_prefix("__OBS__ "))
        .map(|l| {
            let side = if l.starts_with("p1 ") { 0 } else { 1 };
            (side, Val::parse(&l[3..]).expect("__OBS__ json"))
        })
        .collect()
}

#[test]
fn the_cores_rows_are_sim_bridges_obs_rows_byte_for_byte() {
    let bin = std::env::var("POKESIM_SIM_BRIDGE_BIN")
        .expect("POKESIM_SIM_BRIDGE_BIN must name THIS checkout's self-check sim_bridge (core_cargo_test.py sets it)");
    let teams = common::corpus_teams();
    let nt = teams.len();
    let mut core = Core::new(common::spec(N, 1, teams)).unwrap();
    let mut cols = OwnedCols::new(N);
    let mut stage_rng = common::Rng(0x0B5E);
    let mut act_rng = common::Rng(11);
    for i in 0..N {
        common::stage(&mut cols, i, &mut stage_rng, nt);
    }
    let a = cols.addrs();
    core.freeze(a).unwrap();
    let mut recs: BTreeMap<(usize, u32, usize, u32), Rec> = BTreeMap::new();
    let mut logs: Vec<(usize, u32, InputLog)> = Vec::new();
    let record = |core: &Core, cols: &OwnedCols, recs: &mut BTreeMap<_, _>| {
        let need = cols.slice::<u8>(col::NEED);
        for i in 0..N {
            let env = core.inline_env(i).unwrap();
            let ep = cols.slice::<u32>(col::EPISODE)[i];
            for s in 0..SIDES {
                if need[i * SIDES + s] == 1 {
                    let n = cols.slice::<u32>(col::DEC_N)[i * SIDES + s];
                    let k = (i * SIDES + s) * OBS_DIM * 4;
                    let rec = Rec {
                        row: cols.bytes(col::OBS)[k..k + OBS_DIM * 4].to_vec(),
                        mask: cols.slice::<u8>(col::MASK)[(i * SIDES + s) * ACT..(i * SIDES + s + 1) * ACT].to_vec(),
                        tokens: pokesim::present::tokens_json(env.open_tokens(s).unwrap()),
                    };
                    assert!(recs.insert((i, ep, s, n), rec).is_none(), "env {i} ep {ep} p{} n {n} recorded twice", s + 1);
                }
            }
        }
    };
    assert_eq!(core.dispatch(b'R', a), 0);
    record(&core, &cols, &mut recs);
    for _ in 0..STEPS {
        for i in 0..N {
            common::stage(&mut cols, i, &mut stage_rng, nt);
        }
        common::random_actions(&mut cols, &mut act_rng);
        let before: Vec<u32> = cols.slice::<u32>(col::EPISODE).to_vec();
        assert_eq!(core.dispatch(b'S', a), 0, "{:?}", core.last_error().map(|e| e.json()));
        for i in 0..N {
            if cols.slice::<u8>(col::DONE)[i] == 1 {
                assert_eq!(cols.slice::<u8>(col::REFUSED)[i], 0, "a quarantine in the parity corpus: change the seed");
                logs.push((i, before[i], core.inline_env(i).unwrap().prev_log.clone()));
            }
        }
        record(&core, &cols, &mut recs);
    }
    for i in 0..N {
        logs.push((i, cols.slice::<u32>(col::EPISODE)[i], core.inline_env(i).unwrap().log.clone()));
    }
    let ended = logs.len() - N;
    assert!(ended >= 5, "only {ended} episodes ended — the corpus does not exercise auto-reset");

    let mut compared = 0usize;
    for (i, ep, log) in &logs {
        let mut per_side = [0u32; 2];
        for (s, v) in sim_bridge_frames(&bin, log) {
            let n = match v.get("n") {
                Some(Val::Int(n)) => *n as u32,
                other => panic!("n: {other:?}"),
            };
            assert_eq!(n, per_side[s], "env {i} ep {ep} p{}: frames out of order", s + 1);
            per_side[s] += 1;
            let rec = recs
                .remove(&(*i, *ep, s, n))
                .unwrap_or_else(|| panic!("sim_bridge shipped a frame the core never wrote: env {i} ep {ep} p{} n {n}", s + 1));
            let b64 = v.get("frame").and_then(|f| f.str_at("b64")).expect("frame.b64");
            let bytes = b64_decode(b64);
            assert_eq!(bytes.len(), OBS_DIM * 4);
            if bytes != rec.row {
                let cell = (0..OBS_DIM).find(|c| bytes[c * 4..c * 4 + 4] != rec.row[c * 4..c * 4 + 4]).unwrap();
                panic!("env {i} ep {ep} p{} n {n}: the row differs first at cell {cell}\nlog:\n{}", s + 1, log.script(""));
            }
            assert!(
                rec.row.chunks(4).all(|c| !f32::from_le_bytes(c.try_into().unwrap()).is_nan()),
                "env {i} ep {ep} p{} n {n}: a NaN cell (an unwritten slot)",
                s + 1
            );
            let mask: Vec<u8> = match v.get("mask") {
                Some(Val::Arr(a)) => a.iter().map(|x| if let Val::Int(k) = x { *k as u8 } else { panic!("mask") }).collect(),
                other => panic!("mask: {other:?}"),
            };
            assert_eq!(mask, rec.mask, "env {i} ep {ep} p{} n {n}: mask", s + 1);
            let tokens = match v.get("tokens") {
                Some(Val::Obj(kv)) => {
                    let pairs: Vec<(usize, String)> = kv
                        .iter()
                        .map(|(k, t)| (k.parse().unwrap(), if let Val::Str(t) = t { t.to_string() } else { panic!("token") }))
                        .collect();
                    pokesim::present::tokens_json(&pairs)
                }
                other => panic!("tokens: {other:?}"),
            };
            assert_eq!(tokens, rec.tokens, "env {i} ep {ep} p{} n {n}: tokens", s + 1);
            compared += 1;
        }
    }
    assert!(recs.is_empty(), "{} rows the core wrote have no sim_bridge frame, e.g. {:?}", recs.len(), recs.keys().next());
    assert!(compared >= 1000, "only {compared} frames compared");
    eprintln!("gate 1: {compared} frames byte-equal over {} input logs ({ended} finished episodes)", logs.len());
}

/// TEETH: the comparison sees the cells a clock flag moves — `sim_bridge` replaying the same logs
/// with `decision_tense = true` (the core ran `false`) must differ in some row.
#[test]
fn the_byte_comparison_has_teeth() {
    let bin = std::env::var("POKESIM_SIM_BRIDGE_BIN").expect("POKESIM_SIM_BRIDGE_BIN");
    let teams = common::corpus_teams();
    let nt = teams.len();
    let mut core = Core::new(common::spec(2, 1, teams)).unwrap();
    let mut cols = OwnedCols::new(2);
    let (mut sr, mut ar) = (common::Rng(5), common::Rng(6));
    for i in 0..2 {
        common::stage(&mut cols, i, &mut sr, nt);
    }
    let a = cols.addrs();
    core.freeze(a).unwrap();
    assert_eq!(core.dispatch(b'R', a), 0);
    let mut ended = false;
    for _ in 0..80 {
        common::random_actions(&mut cols, &mut ar);
        assert_eq!(core.dispatch(b'S', a), 0);
        if cols.slice::<u8>(col::DONE)[0] == 1 {
            ended = true;
            break;
        }
    }
    let env = core.inline_env(0).unwrap();
    let log = if ended { env.prev_log.clone() } else { env.log.clone() };
    assert!(log.cmds.len() >= 20, "too short a battle for the teeth: {} commands", log.cmds.len());
    let same = sim_bridge_frames_with(&bin, &log, false);
    let tense = sim_bridge_frames_with(&bin, &log, true);
    assert_eq!(same.len(), tense.len());
    let differ = same
        .iter()
        .zip(&tense)
        .filter(|((_, a), (_, b))| a.get("frame").and_then(|f| f.str_at("b64")) != b.get("frame").and_then(|f| f.str_at("b64")))
        .count();
    assert!(differ > 0, "decision_tense moved no row over {} frames — the gate could not see a clock divergence", same.len());
    eprintln!("teeth: decision_tense moved {differ} of {} rows", same.len());
}
