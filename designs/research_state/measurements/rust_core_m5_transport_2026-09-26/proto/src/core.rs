//! THE ENV CORE — the one implementation both front ends call. It holds every piece of battle
//! logic in the prototype; `ffi.rs` and `bin/m5_envproc.rs` only turn their caller's memory into
//! the [`Cols`] slices below.
//!
//! N battles, each a live `BridgeSession` plus one PARSE chain per side (program §6c: the
//! observation always comes through the parser — the exact road `sim_bridge`'s `core_obs` ships to
//! training today). Workers: T persistent threads, each owning a contiguous block of envs and
//! writing only that block's rows (disjoint, so no lock on the columns).
//!
//! THE COLUMN CONTRACT (row-major, caller-allocated, `E` = env, `S` = side 0/1):
//!   obs    f32 [N][2][OBS_DIM]  the side's row, written iff need[E][S] = 1
//!   mask   u8  [N][2][11]       the side's 11-dim action mask, written iff need = 1
//!   need   u8  [N][2]           1 iff the CALLER must supply that side's action next step
//!   reward f32 [N]              the win indicator (p1's view) of an episode that ENDED this step
//!   done   u8  [N]              1 iff an episode ended this step (the env auto-resets: obs is the
//!                               next episode's first decision — EnvPool semantics)
//!   action i32 [N][2]           input: an action index (0..11) for every side with need = 1
//!
//! Opponent modes: `opp_external` = the T2 shape (both sides' rows go to the caller, which runs
//! both networks); otherwise side 1 is a seeded random player INSIDE Rust (a scripted-bot shape,
//! never encoded).

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::encoder::OBS_DIM;
use pokesim::present;
use pokesim::trackers::clock::ClockConfig;
use pokesim::version::BattleVersion;

pub const ACT: usize = 11;
pub const NAMES: [&str; 2] = ["m5p1", "m5p2"];
/// A battle still running at this turn is forfeited by p1 (a random battle almost never gets here).
pub const TURN_CAP: u32 = 300;

#[derive(Clone, Debug)]
pub struct Spec {
    pub n: usize,
    pub threads: usize,
    pub seed: u64,
    pub opp_external: bool,
}

// ------------------------------------------------------------------ small deterministic RNG

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

// ------------------------------------------------------------------ teams

/// Every team of the port's bridge corpus (real gen3ou teams, packed), in file order.
pub fn corpus_teams() -> Result<Vec<String>, String> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../../../src/rust_sim/tests/vectors/bridge_corpus");
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("corpus {dir}: {e}"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    files.sort();
    let mut out: Vec<String> = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).map_err(|e| e.to_string())?;
        for l in text.lines().filter(|l| l.starts_with("TEAM\t")) {
            let t = l.split('\t').nth(3).unwrap_or("").to_string();
            if !t.is_empty() && !out.contains(&t) {
                out.push(t);
            }
        }
    }
    if out.len() < 2 {
        return Err("corpus holds fewer than two teams".into());
    }
    Ok(out)
}

// ------------------------------------------------------------------ one env

struct Dec {
    tokens: Vec<(usize, String)>,
    mask: [u8; ACT],
}

pub struct Env {
    rng: Rng,
    episode: u64,
    env_seed: u64,
    teams: [String; 2],
    sess: Option<BridgeSession>,
    chains: [Option<BattleVersion>; 2],
    pending: [Option<Dec>; 2],
    emitted: usize,
    pub decisions: u64,
    /// The battle's input log (seed + every command fed), so any refusal is re-runnable alone.
    seed: String,
    cmds: Vec<String>,
}

/// One env's slice of every column.
pub struct EnvCols<'a> {
    pub obs: &'a mut [f32],  // 2 * OBS_DIM
    pub mask: &'a mut [u8],  // 2 * ACT
    pub need: &'a mut [u8],  // 2
    pub reward: &'a mut f32,
    pub done: &'a mut u8,
}

pub struct Ctx {
    pub dex: Dex,
    pub teams: Vec<String>,
    pub opp_external: bool,
    pub cfg: ClockConfig,
}

impl Env {
    pub fn new(pool_seed: u64, id: usize) -> Env {
        let mut r = Rng(pool_seed ^ (0xA5A5_0000_0000_0000u64.wrapping_add(id as u64 * 0x1000_0001)));
        let env_seed = r.next();
        Env {
            rng: Rng(env_seed),
            episode: 0,
            env_seed,
            teams: [String::new(), String::new()],
            sess: None,
            chains: [None, None],
            pending: [None, None],
            emitted: 0,
            decisions: 0,
            seed: String::new(),
            cmds: Vec::new(),
        }
    }

    fn start(&mut self, ctx: &Ctx) -> Result<(), String> {
        let mut r = Rng(self.env_seed ^ self.episode.wrapping_mul(0x2545_F491_4F6C_DD1D));
        self.episode += 1;
        let a = r.below(ctx.teams.len());
        let mut b = r.below(ctx.teams.len() - 1);
        if b >= a {
            b += 1;
        }
        self.teams = [ctx.teams[a].clone(), ctx.teams[b].clone()];
        let seed = format!("{},{},{},{}", r.below(65536), r.below(65536), r.below(65536), r.below(65536));
        self.seed = seed.clone();
        self.cmds.clear();
        let opts = BattleOptions {
            format_id: "gen3ou".into(),
            seed: Some(seed),
            p1: PlayerOptions { name: NAMES[0].into(), team: PackedTeam(self.teams[0].clone()) },
            p2: PlayerOptions { name: NAMES[1].into(), team: PackedTeam(self.teams[1].clone()) },
        };
        self.sess = Some(BridgeSession::new_construct_turn0(&opts, &ctx.dex)?);
        for side in 0..2 {
            // Both chains fold the trackers: a decision is DEFINED by the tracker fold
            // (`BattleVersion::decision`), and an in-Rust opponent needs its decisions too. The
            // in-Rust opponent's row is never encoded.
            // `parse_root_unrecorded`: trackers on, no native window record — exactly the chain
            // `sim_bridge`'s `core_obs` folds for training today.
            self.chains[side] = Some(
                BattleVersion::parse_root_unrecorded(side, NAMES[side], Some(&self.teams[side]), ctx.cfg)
                    .map_err(|e| format!("root p{}: {}", side + 1, e.message()))?,
            );
        }
        self.pending = [None, None];
        self.emitted = 0;
        self.advance(None)?;
        self.drive(ctx)
    }

    /// Fold every line each side was shipped since the cursor through its parse chain, and
    /// update the pending decisions (the `sim_bridge_core_obs_test` take rule).
    fn advance(&mut self, sent: Option<usize>) -> Result<(), String> {
        let sess = self.sess.as_ref().ok_or("no battle")?;
        let chunks = &sess.chunks().chunks[self.emitted..];
        for side in 0..2 {
            let new: Vec<&str> = chunks
                .iter()
                .filter(|c| c.side == side)
                .flat_map(|c| c.lines.iter().map(String::as_str))
                .collect();
            if new.is_empty() {
                if sent == Some(side) {
                    // A choice that shipped nothing yet (the other side still to choose).
                    self.pending[side] = None;
                }
                continue;
            }
            let rejected = new.iter().any(|l| l.starts_with("|error|"));
            let chain = self.chains[side].take().ok_or("chain lost")?;
            let next = chain.parse_advance_lean(&new).map_err(|e| format!("parse p{}: {}", side + 1, e.message()))?;
            if next.decision(side).is_some() {
                let legal = next.legal(side).ok_or("a decision with no legality")?;
                let reading = &next.stream(side).ok_or("no stream")?.board_reading;
                let tokens = present::choice_tokens(reading, &legal).map_err(|e| e.message().to_string())?;
                self.pending[side] = Some(Dec { tokens, mask: present::mask(&legal) });
            } else if sent == Some(side) && !rejected {
                self.pending[side] = None;
            }
            self.chains[side] = Some(next);
        }
        self.emitted = sess.chunks().chunks.len();
        Ok(())
    }

    fn feed(&mut self, ctx: &Ctx, side: usize, tok: &str) -> Result<(), String> {
        let choice = parse_choice(tok).ok_or_else(|| format!("unparseable token {tok:?}"))?;
        if let Some(c) = self.chains[side].as_mut() {
            c.note_choice(side, tok);
        }
        self.cmds.push(format!("CHOOSE p{} {tok}", side + 1));
        let sess = self.sess.as_mut().ok_or("no battle")?;
        sess.feed_cmd(Cmd { side, choice }, &ctx.dex);
        if let Some(f) = sess.fatal() {
            return Err(format!("engine fatal: {f}"));
        }
        self.advance(Some(side))
    }

    /// Run everything that needs no caller: the in-Rust opponent, the turn cap. Returns with the
    /// battle ended or at least one caller-owned decision open.
    fn drive(&mut self, ctx: &Ctx) -> Result<(), String> {
        loop {
            let sess = self.sess.as_ref().ok_or("no battle")?;
            if sess.is_ended() {
                return Ok(());
            }
            if sess.turn() > TURN_CAP {
                self.sess.as_mut().unwrap().forfeit(0);
                self.cmds.push("FORCELOSE p1".into());
                self.advance(None)?;
                continue;
            }
            if !ctx.opp_external {
                if let Some(d) = &self.pending[1] {
                    let tok = d.tokens[self.rng.below(d.tokens.len())].1.clone();
                    self.feed(ctx, 1, &tok)?;
                    continue;
                }
            }
            if self.pending[0].is_some() || (ctx.opp_external && self.pending[1].is_some()) {
                return Ok(());
            }
            return Err("stuck: no decision open and the battle is not over".into());
        }
    }

    fn write(&mut self, ctx: &Ctx, c: &mut EnvCols) -> Result<(), String> {
        for side in 0..2 {
            let need = self.pending[side].is_some() && (side == 0 || ctx.opp_external);
            c.need[side] = need as u8;
            if need {
                let d = self.pending[side].as_ref().unwrap();
                c.mask[side * ACT..(side + 1) * ACT].copy_from_slice(&d.mask);
                let row: &mut [f32; OBS_DIM] = (&mut c.obs[side * OBS_DIM..(side + 1) * OBS_DIM])
                    .try_into()
                    .map_err(|_| "row slice")?;
                self.chains[side]
                    .as_ref()
                    .unwrap()
                    .encode(side, row)
                    .map_err(|e| format!("encode p{}: {}", side + 1, e.message()))?;
                self.decisions += 1;
            }
        }
        Ok(())
    }

    pub fn reset(&mut self, ctx: &Ctx, c: &mut EnvCols) -> Result<(), String> {
        *c.reward = 0.0;
        *c.done = 0;
        self.start(ctx)?;
        self.write(ctx, c)
    }

    pub fn step(&mut self, ctx: &Ctx, act: [i32; 2], c: &mut EnvCols) -> Result<(), String> {
        *c.reward = 0.0;
        *c.done = 0;
        for side in 0..2 {
            if c.need[side] == 0 {
                continue;
            }
            let d = self.pending[side].as_ref().ok_or("need set with no decision open")?;
            let a = act[side];
            let tok = d
                .tokens
                .iter()
                .find(|(i, _)| *i as i32 == a)
                .map(|(_, t)| t.clone())
                .ok_or_else(|| format!("p{}: action {a} is not legal here", side + 1))?;
            self.feed(ctx, side, &tok)?;
        }
        self.drive(ctx)?;
        let sess = self.sess.as_ref().unwrap();
        if sess.is_ended() {
            *c.reward = match sess.winner() {
                Some(0) => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };
            *c.done = 1;
            self.start(ctx)?;
        }
        self.write(ctx, c)
    }

    /// The failing battle as a `core_events` / `sim_bridge` script (START json + commands).
    pub fn repro(&self) -> String {
        let q = |x: &str| format!("{:?}", x);
        format!(
            "START {{\"formatid\":\"gen3ou\",\"seed\":{},\"p1\":{{\"name\":{},\"team\":{}}},\"p2\":{{\"name\":{},\"team\":{}}}}}\n{}",
            q(&self.seed), q(NAMES[0]), q(&self.teams[0]), q(NAMES[1]), q(&self.teams[1]), self.cmds.join("\n")
        )
    }

    /// SEARCH SHAPE — `successors(p1, k)`: fork a step-built root from this env's live session,
    /// step k children (p1's legal tokens round-robin, p2 random when it has a decision open) and
    /// encode p1's row at each child that is a p1 decision. `ok[j]` = 1 iff row j was encoded.
    pub fn successors(&mut self, ctx: &Ctx, k: usize, rows: &mut [f32], ok: &mut [u8]) -> Result<(), String> {
        let d0 = self.pending[0].as_ref().ok_or("successors: p1 has no decision open")?;
        let sess = self.sess.as_ref().ok_or("no battle")?;
        let root = Arc::new(
            BattleVersion::root_with(
                sess.snapshot(),
                NAMES,
                [Some(&self.teams[0]), Some(&self.teams[1])],
                [true, false],
                Some(ctx.cfg),
            )
            .map_err(|e| format!("root: {}", e.message()))?,
        );
        let t0: Vec<String> = d0.tokens.iter().map(|(_, t)| t.clone()).collect();
        let t1: Option<Vec<String>> = self.pending[1].as_ref().map(|d| d.tokens.iter().map(|(_, t)| t.clone()).collect());
        for j in 0..k {
            let mut cmds = vec![Cmd { side: 0, choice: parse_choice(&t0[j % t0.len()]).ok_or("tok")? }];
            if let Some(t1) = &t1 {
                cmds.push(Cmd { side: 1, choice: parse_choice(&t1[self.rng.below(t1.len())]).ok_or("tok")? });
            }
            let child = root.step(&cmds, &ctx.dex).map_err(|e| format!("step: {}", e.message()))?;
            let row: &mut [f32; OBS_DIM] = (&mut rows[j * OBS_DIM..(j + 1) * OBS_DIM]).try_into().map_err(|_| "row")?;
            if child.decision(0).is_some() {
                child.encode(0, row).map_err(|e| format!("encode: {}", e.message()))?;
                ok[j] = 1;
            } else {
                row.fill(0.0);
                ok[j] = 0;
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ the pool

/// Raw column addresses (the caller's memory; a worker touches only its own envs' rows).
#[derive(Clone, Copy)]
pub struct Ptrs {
    pub obs: usize,
    pub mask: usize,
    pub need: usize,
    pub reward: usize,
    pub done: usize,
    pub actions: usize,
}

impl Ptrs {
    /// # Safety
    /// The caller guarantees every column holds N envs' worth of the contract's shape, alive and
    /// unaliased for the duration of the call; distinct `i` give disjoint slices.
    pub unsafe fn env<'a>(&self, i: usize) -> (EnvCols<'a>, [i32; 2]) {
        let obs = std::slice::from_raw_parts_mut((self.obs as *mut f32).add(i * 2 * OBS_DIM), 2 * OBS_DIM);
        let mask = std::slice::from_raw_parts_mut((self.mask as *mut u8).add(i * 2 * ACT), 2 * ACT);
        let need = std::slice::from_raw_parts_mut((self.need as *mut u8).add(i * 2), 2);
        let reward = &mut *(self.reward as *mut f32).add(i);
        let done = &mut *(self.done as *mut u8).add(i);
        let act = if self.actions == 0 {
            [-1, -1]
        } else {
            let a = self.actions as *const i32;
            [*a.add(2 * i), *a.add(2 * i + 1)]
        };
        (EnvCols { obs, mask, need, reward, done }, act)
    }
}

enum Job {
    Reset(Ptrs),
    Step(Ptrs),
}

/// Run one block of envs. A core REFUSAL (a `CoreError` from the fold / encode — a battle the
/// reading cannot explain) QUARANTINES that battle instead of failing the batch: its input log is
/// banked in `refused`, the env starts a fresh episode, and the column reads `done = 1`,
/// `reward = 0` (the caller can tell it from a tie by the refusal counter). A PANIC or a caller
/// error (an illegal action) still fails the batch.
fn run_block(ctx: &Ctx, envs: &mut [Env], lo: usize, job: &Job, refused: &mut Vec<String>) -> Result<(), String> {
    for (k, env) in envs.iter_mut().enumerate() {
        let i = lo + k;
        let r = catch_unwind(AssertUnwindSafe(|| unsafe {
            match job {
                Job::Reset(p) => {
                    let (mut c, _) = p.env(i);
                    env.reset(ctx, &mut c)
                }
                Job::Step(p) => {
                    let (mut c, a) = p.env(i);
                    env.step(ctx, a, &mut c)
                }
            }
        }));
        match r {
            Ok(Ok(())) => {}
            Ok(Err(e)) if !e.contains("is not legal here") => {
                refused.push(format!("env {i}: {e}\nREPRO {}", env.repro()));
                let again = catch_unwind(AssertUnwindSafe(|| unsafe {
                    let (mut c, _) = match job {
                        Job::Reset(p) | Job::Step(p) => p.env(i),
                    };
                    env.reset(ctx, &mut c).map(|()| *c.done = 1)
                }));
                match again {
                    Ok(Ok(())) => {}
                    Ok(Err(e2)) => return Err(format!("env {i}: refused, and its reset refused too: {e2}")),
                    Err(_) => return Err(format!("env {i}: PANIC in the reset after a refusal")),
                }
            }
            Ok(Err(e)) => return Err(format!("env {i}: {e}")),
            Err(_) => return Err(format!("env {i}: PANIC inside the core\nREPRO {}", env.repro())),
        }
    }
    Ok(())
}

struct Worker {
    tx: Sender<Option<Job>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

pub struct Pool {
    pub spec: Spec,
    ctx: Arc<Ctx>,
    /// Inline envs (threads <= 1) — run on the caller's thread.
    inline: Vec<Env>,
    workers: Vec<Worker>,
    done_rx: Option<Receiver<Result<Vec<String>, String>>>,
    /// Every quarantined battle's error + input log, in the order the batches returned them.
    pub refused: Vec<String>,
    /// Wall of the last call INSIDE the core (the caller's wall minus this = the transport).
    pub last_core_ns: u64,
}

impl Pool {
    pub fn new(spec: Spec) -> Result<Pool, String> {
        if spec.n == 0 {
            return Err("n must be >= 1".into());
        }
        let ctx = Arc::new(Ctx { dex: Dex::for_gen(3), teams: corpus_teams()?, opp_external: spec.opp_external, cfg: ClockConfig::default() });
        let envs: Vec<Env> = (0..spec.n).map(|i| Env::new(spec.seed, i)).collect();
        let t = spec.threads.max(1).min(spec.n);
        if t <= 1 {
            return Ok(Pool { spec, ctx, inline: envs, workers: Vec::new(), done_rx: None, refused: Vec::new(), last_core_ns: 0 });
        }
        let (dtx, drx) = channel::<Result<Vec<String>, String>>();
        let mut workers = Vec::new();
        let mut it = envs.into_iter();
        let base = spec.n / t;
        let extra = spec.n % t;
        let mut lo = 0;
        for w in 0..t {
            let cnt = base + usize::from(w < extra);
            let mut block: Vec<Env> = it.by_ref().take(cnt).collect();
            let (tx, rx) = channel::<Option<Job>>();
            let dtx = dtx.clone();
            let ctx = Arc::clone(&ctx);
            let my_lo = lo;
            lo += cnt;
            let handle = std::thread::Builder::new()
                .name(format!("m5w{w}"))
                .spawn(move || {
                    while let Ok(Some(job)) = rx.recv() {
                        let mut refused = Vec::new();
                        let r = run_block(&ctx, &mut block, my_lo, &job, &mut refused).map(|()| refused);
                        if dtx.send(r).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|e| e.to_string())?;
            workers.push(Worker { tx, handle: Some(handle) });
        }
        Ok(Pool { spec, ctx, inline: Vec::new(), workers, done_rx: Some(drx), refused: Vec::new(), last_core_ns: 0 })
    }

    fn run(&mut self, mk: impl Fn() -> Job) -> Result<(), String> {
        let t0 = Instant::now();
        let r = if self.workers.is_empty() {
            let job = mk();
            run_block(&self.ctx, &mut self.inline, 0, &job, &mut self.refused)
        } else {
            for w in &self.workers {
                w.tx.send(Some(mk())).map_err(|_| "worker gone")?;
            }
            let rx = self.done_rx.as_ref().unwrap();
            let mut first_err = Ok(());
            for _ in 0..self.workers.len() {
                match rx.recv().map_err(|_| "worker gone".to_string())? {
                    Ok(mut v) => self.refused.append(&mut v),
                    Err(e) => {
                        if first_err.is_ok() {
                            first_err = Err(e);
                        }
                    }
                }
            }
            first_err
        };
        self.last_core_ns = t0.elapsed().as_nanos() as u64;
        r
    }

    /// # Safety
    /// See [`Ptrs::env`].
    pub unsafe fn reset(&mut self, p: Ptrs) -> Result<(), String> {
        self.run(|| Job::Reset(p))
    }

    /// # Safety
    /// See [`Ptrs::env`].
    pub unsafe fn step(&mut self, p: Ptrs) -> Result<(), String> {
        self.run(|| Job::Step(p))
    }

    /// The search shape on a DEDICATED single env (the caller builds a pool of n = 1, threads = 1).
    pub fn successors(&mut self, k: usize, rows: &mut [f32], ok: &mut [u8]) -> Result<(), String> {
        let t0 = Instant::now();
        let env = self.inline.first_mut().ok_or("successors needs an inline pool (threads = 1)")?;
        let ctx = Arc::clone(&self.ctx);
        let r = catch_unwind(AssertUnwindSafe(|| env.successors(&ctx, k, rows, ok)))
            .unwrap_or_else(|_| Err("PANIC inside the core".into()));
        self.last_core_ns = t0.elapsed().as_nanos() as u64;
        r
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for w in &self.workers {
            let _ = w.tx.send(None);
        }
        for w in &mut self.workers {
            if let Some(h) = w.handle.take() {
                let _ = h.join();
            }
        }
    }
}

/// The build stamp both front ends report (`build.rs`).
pub const STAMP: &str = concat!(
    "commit=", env!("M5_STAMP_COMMIT"),
    ";src=", env!("M5_STAMP_SRC"),
    ";nfiles=", env!("M5_STAMP_NFILES"),
    "\0"
);

/// Buffer sizes in bytes for n envs (the process front end's shm layout uses the same).
pub fn col_bytes(n: usize) -> [usize; 6] {
    [n * 2 * OBS_DIM * 4, n * 2 * ACT, n * 2, n * 4, n, n * 2 * 4]
}
