//! Shared test harness: the port's bridge-corpus teams, a deterministic staging / action RNG, and a
//! driver that runs a [`Core`] over owned columns.
#![allow(dead_code)]

use pokesim_env::core::columns::{col, counter, ACT, SIDES};
use pokesim_env::core::{Core, OwnedCols, Spec};
use pokesim::trackers::clock::ClockConfig;

pub const NAMES: [&str; 2] = ["m5p1", "m5p2"];

/// Every team of the port's bridge corpus (real gen3ou teams, packed), in file order, deduplicated.
pub fn corpus_teams() -> Vec<String> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../rust_sim/tests/vectors/bridge_corpus");
    let mut files: Vec<_> = std::fs::read_dir(dir).expect("corpus").flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
    files.sort();
    let mut out: Vec<String> = Vec::new();
    for f in files {
        for l in std::fs::read_to_string(&f).unwrap().lines().filter(|l| l.starts_with("TEAM\t")) {
            let t = l.split('\t').nth(3).unwrap_or("").to_string();
            if !t.is_empty() && !out.contains(&t) {
                out.push(t);
            }
        }
    }
    assert!(out.len() >= 10, "the corpus shrank to {} teams", out.len());
    out
}

pub fn spec(n: usize, threads: usize, teams: Vec<String>) -> Spec {
    Spec {
        n,
        threads,
        format_id: "gen3ou".into(),
        names: [NAMES[0].into(), NAMES[1].into()],
        teams,
        clock: ClockConfig::default(),
        turn_limit: Some(300),
        terminal: pokesim_env::episode::Terminal::PRODUCTION,
        refusal_budget: 64,
        bank_dir: None,
        labels: Vec::new(),
    }
}

#[derive(Clone)]
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Stage env `i`'s next episode (two distinct teams, a seed) from `rng`.
pub fn stage(cols: &mut OwnedCols, i: usize, rng: &mut Rng, n_teams: usize) {
    let a = rng.below(n_teams);
    let mut b = rng.below(n_teams - 1);
    if b >= a {
        b += 1;
    }
    cols.slice_mut::<u32>(col::EP_TEAM)[2 * i..2 * i + 2].copy_from_slice(&[a as u32, b as u32]);
    let seed: Vec<u32> = (0..4).map(|_| rng.below(65536) as u32).collect();
    cols.slice_mut::<u32>(col::EP_SEED)[4 * i..4 * i + 4].copy_from_slice(&seed);
}

/// A seeded random policy over the mask for every side with need = 1 (-1 elsewhere).
pub fn random_actions(cols: &mut OwnedCols, rng: &mut Rng) {
    let n = cols.n;
    let need: Vec<u8> = cols.slice::<u8>(col::NEED).to_vec();
    let mask: Vec<u8> = cols.slice::<u8>(col::MASK).to_vec();
    let act = cols.slice_mut::<i32>(col::ACTION);
    for i in 0..n {
        for s in 0..SIDES {
            let k = i * SIDES + s;
            act[k] = -1;
            if need[k] == 1 {
                let legal: Vec<usize> = (0..ACT).filter(|&a| mask[k * ACT + a] == 1).collect();
                assert!(!legal.is_empty(), "env {i} p{} needs an action but its mask is empty", s + 1);
                act[k] = legal[rng.below(legal.len())] as i32;
            }
        }
    }
}

/// FNV-1a-64 over every per-env OUTPUT column and the DETERMINISTIC counters (the wall-clock ones
/// excluded).
pub fn digest(cols: &OwnedCols) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |b: &[u8]| {
        for &x in b {
            h ^= x as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for c in [col::OBS, col::MASK, col::NEED, col::REWARD, col::DONE, col::TERMINATED, col::TRUNCATED, col::REFUSED, col::EPISODE, col::DEC_N, col::TURN] {
        eat(cols.bytes(c));
    }
    let k = cols.slice::<u64>(col::COUNTERS);
    for i in [counter::DISPATCHES, counter::DECISIONS, counter::EPISODES_STARTED, counter::EPISODES_ENDED, counter::REFUSALS] {
        eat(&k[i].to_le_bytes());
    }
    h
}

pub struct Trace {
    pub digests: Vec<u64>,
    pub counters: Vec<u64>,
    pub bank: Vec<String>,
}

/// Build a core, RESET, then `steps` STEPs of the random policy (re-staging every env whose
/// episode ended). Asserts every op returns OK.
pub fn run(n: usize, threads: usize, seed: u64, steps: usize) -> Trace {
    let teams = corpus_teams();
    let nt = teams.len();
    let mut core = Core::new(spec(n, threads, teams)).expect("core");
    let mut cols = OwnedCols::new(n);
    let mut stage_rng = Rng(seed ^ 0x5157_A6E5);
    let mut act_rng = Rng(seed);
    for i in 0..n {
        stage(&mut cols, i, &mut stage_rng, nt);
    }
    let a = cols.addrs();
    core.freeze(a).expect("freeze");
    let mut digests = Vec::with_capacity(steps + 1);
    let st = core.dispatch(b'R', a);
    assert_eq!(st, 0, "RESET: {:?}", core.last_error());
    digests.push(digest(&cols));
    for _ in 0..steps {
        // The first RESET consumed every staged row; re-stage (every env, env order) so the NEXT
        // episode's inputs are always present.
        for i in 0..n {
            stage(&mut cols, i, &mut stage_rng, nt);
        }
        random_actions(&mut cols, &mut act_rng);
        let st = core.dispatch(b'S', a);
        assert_eq!(st, 0, "STEP: {:?}", core.last_error().map(|e| e.json()));
        digests.push(digest(&cols));
    }
    Trace {
        digests,
        counters: cols.slice::<u64>(col::COUNTERS).to_vec(),
        bank: core.bank().items().iter().map(|b| b.json()).collect(),
    }
}
