//! THE POOL — N battles on T persistent worker threads, acquired at startup and FROZEN (M5 Lane 0).
//!
//! Each env holds a live `BridgeSession` (the engine + the per-side transport `sim_bridge` writes)
//! and one PARSE chain per side (`BattleVersion::parse_root_unrecorded` → `parse_advance_lean`):
//! the observation comes THROUGH THE PARSER (program §6c), over exactly the lines each side was
//! shipped by each write — the chain `sim_bridge`'s `core_obs` mode ships to training, with ITS
//! alignment rules (one decision per side per write, the decision's request the write's last line),
//! so gate ① can hold the rows byte-equal to `__OBS__`. A row is encoded straight into the caller's
//! `obs` column the moment its decision opens.
//!
//! THE LIFECYCLE. [`Pool::new`] acquires everything the steady state will use — the dex, the team
//! table (every team unpacked, so a bad team fails at startup, not at its first draw), the envs,
//! the worker threads, the refusal bank's reserved capacity. [`Pool::freeze`] binds the caller's
//! column set and validates it (non-null, aligned). From then on an op acquires nothing: the only
//! thread-spawn and env-construction sites are in `new`, and the `*_AFTER_FREEZE` counters count
//! any that would happen after it (structurally 0), a rebind attempt (refused), and bank growth.
//!
//! DETERMINISM (gate ②). An env's bytes are a function of ITS staged inputs (`ep_team`, `ep_seed`)
//! and the actions fed to it — nothing else: no clock, no RNG outside the battle's own seed, no
//! cross-env state. Workers own contiguous env blocks and write only those rows, and every
//! cross-env reduction (the bank, the first error, the counters) is folded in ENV order, so the
//! thread count and the scheduling are invisible in every column and in the bank.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::present;
use pokesim::prng::Prng;
use pokesim::version::BattleVersion;

use super::columns::{ColAddrs, EnvCols, ACT, OBS_DIM, SEED_WORDS, SIDES};
use super::refusal::{Class, EnvError, InputLog};
use super::spec::Spec;

/// Read-only state every worker shares (acquired at startup).
pub struct Ctx {
    pub dex: Dex,
    pub spec: Spec,
}

/// An open decision: the real mapper's token per legal action index, and its frame index `n`.
struct Open {
    tokens: Vec<(usize, String)>,
    n: u32,
}

/// One env.
pub struct Env {
    id: usize,
    /// Episodes started (the `episode` column is this minus one while one runs).
    started: u32,
    sess: Option<BridgeSession>,
    chains: [Option<BattleVersion>; 2],
    /// Lines of each side's stream its chain has folded.
    folded: [usize; 2],
    /// Decisions each chain has taken (= the next frame's `n`).
    decided: [u32; 2],
    /// Chunks already folded (the write cursor, `sim_bridge`'s `emitted`).
    emitted: usize,
    open: [Option<Open>; 2],
    pub log: InputLog,
    /// The previous episode's input log (moved, not copied, at every start) — a harness reads it
    /// after `done` to replay that episode elsewhere (gate ①).
    pub prev_log: InputLog,
    /// The label families' per-episode state (M5 Lane C), cleared at every start.
    labels: crate::labels::EpisodeState,
}

/// What one env did in one op (folded into the pool counters in env order).
#[derive(Default, Clone, Copy)]
pub struct Tally {
    pub decisions: u64,
    pub started: u64,
    pub ended: u64,
}

fn core_err(at: &str) -> impl Fn(pokesim::core_error::CoreError) -> EnvError + '_ {
    move |e| EnvError::from_core(e, at)
}

impl Env {
    fn new(id: usize) -> Env {
        Env {
            id,
            started: 0,
            sess: None,
            chains: [None, None],
            folded: [0, 0],
            decided: [0, 0],
            emitted: 0,
            open: [None, None],
            log: InputLog::default(),
            prev_log: InputLog::default(),
            labels: crate::labels::EpisodeState::default(),
        }
    }

    /// The open decision's (action index, choice token) pairs for `side` — the mapper's view of
    /// the mask (a harness writes an input log's `CHOOSE` lines from it).
    pub fn open_tokens(&self, side: usize) -> Option<&[(usize, String)]> {
        self.open[side].as_ref().map(|o| o.tokens.as_slice())
    }

    /// The episode ordinal of the battle in progress.
    pub fn episode(&self) -> u32 {
        self.started.saturating_sub(1)
    }

    /// Start the next episode from the staged inputs, fold its first write and encode its first
    /// decisions into `c`.
    fn start(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally) -> Result<(), EnvError> {
        let spec = &ctx.spec;
        let team_idx = [c.ep_team[0] as usize, c.ep_team[1] as usize];
        for (s, &ti) in team_idx.iter().enumerate() {
            if ti >= spec.teams.len() {
                return Err(EnvError::caller(format!(
                    "env {}: ep_team[p{}] = {ti} is outside the declared team table ({} teams)",
                    self.id,
                    s + 1,
                    spec.teams.len()
                )));
            }
        }
        let words: [u32; SEED_WORDS] = c.ep_seed[..SEED_WORDS].try_into().expect("seed row");
        if let Some(w) = words.iter().find(|w| **w >= 65536) {
            return Err(EnvError::caller(format!("env {}: ep_seed word {w} is not < 65536", self.id)));
        }
        let seed = words.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        Prng::validate_seed(&seed).map_err(|e| EnvError::caller(format!("env {}: ep_seed {seed:?}: {e}", self.id)))?;
        self.started += 1;
        t.started += 1;
        self.prev_log = std::mem::take(&mut self.log);
        self.log = InputLog {
            format_id: spec.format_id.clone(),
            seed: seed.clone(),
            names: spec.names.clone(),
            teams: [spec.teams[team_idx[0]].clone(), spec.teams[team_idx[1]].clone()],
            cmds: Vec::new(),
        };
        self.sess = None;
        self.chains = [None, None];
        self.folded = [0, 0];
        self.decided = [0, 0];
        self.emitted = 0;
        self.open = [None, None];
        self.labels.clear();
        for side in 0..SIDES {
            self.chains[side] = Some(
                BattleVersion::parse_root_unrecorded(side, &spec.names[side], Some(&self.log.teams[side]), spec.clock)
                    .map_err(core_err(&format!("root p{}", side + 1)))?,
            );
        }
        let opts = BattleOptions {
            format_id: spec.format_id.clone(),
            seed: Some(seed),
            p1: PlayerOptions { name: spec.names[0].clone(), team: PackedTeam(self.log.teams[0].clone()) },
            p2: PlayerOptions { name: spec.names[1].clone(), team: PackedTeam(self.log.teams[1].clone()) },
        };
        // A battle the engine cannot even construct is the engine's refusal (quarantine class).
        self.sess = Some(BridgeSession::new_construct_turn0(&opts, &ctx.dex).map_err(EnvError::engine)?);
        self.advance(ctx, c, t)
    }

    /// Fold every chunk past the write cursor through each side's chain — `sim_bridge`'s
    /// `CoreObs::step_inner`, rule for rule — and encode each decision that opened into `c`.
    ///
    /// TWO passes (M5 Lane C): every side's chain folds the write FIRST, then each decision that
    /// opened is encoded and labelled — a label reads the OTHER side's chain (its own team, the
    /// Python env's `battle2`), which must already hold the same write. The encode reads only its
    /// own chain, so the rows are unchanged by the order.
    fn advance(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally) -> Result<(), EnvError> {
        let sess = self.sess.as_ref().ok_or_else(|| EnvError::fault("no battle"))?;
        if let Some(f) = sess.fatal() {
            return Err(EnvError::engine(format!("engine fatal: {f}")));
        }
        let chunks = &sess.chunks().chunks[self.emitted..];
        let mut opened_now = [false; SIDES];
        for side in 0..SIDES {
            let tag = side + 1;
            let chain = self.chains[side].take().ok_or_else(|| EnvError::fault(format!("p{tag}: chain lost")))?;
            let new: Vec<&str> =
                chunks.iter().filter(|ch| ch.side == side).flat_map(|ch| ch.lines.iter().map(String::as_str)).collect();
            let next = chain.parse_advance_lean(&new).map_err(core_err(&format!("parse p{tag}")))?;
            self.folded[side] += new.len();
            let s = next.stream(side).ok_or_else(|| EnvError::fault(format!("p{tag}: the chain lost its stream")))?;
            if s.lines != self.folded[side] {
                return Err(EnvError::fault(format!(
                    "p{tag}: the chain folded {} lines, the cursor says {}",
                    s.lines, self.folded[side]
                )));
            }
            let decisions = next.trackers(side).map_or(0, |tr| tr.decisions);
            let opened = decisions
                .checked_sub(self.decided[side])
                .ok_or_else(|| EnvError::fault(format!("p{tag}: the decision count went backwards")))?;
            match (next.decision(side), opened) {
                (None, 0) => {}
                (Some(d), 1) => {
                    if d.line + 1 != s.lines {
                        return Err(EnvError::fault(format!(
                            "[ALIGN] p{tag} decided at stream line {} but the write shipped {} lines",
                            d.line, s.lines
                        )));
                    }
                    opened_now[side] = true;
                }
                (d, n) => {
                    return Err(EnvError::fault(format!(
                        "p{tag}: one write opened {n} decisions (a decision at the boundary: {})",
                        d.is_some()
                    )));
                }
            }
            if !opened_now[side] {
                self.decided[side] = decisions;
            }
            self.chains[side] = Some(next);
        }
        for side in (0..SIDES).filter(|&s| opened_now[s]) {
            let tag = side + 1;
            let chain = self.chains[side].as_ref().ok_or_else(|| EnvError::fault(format!("p{tag}: chain lost")))?;
            let row: &mut [f32; OBS_DIM] =
                (&mut c.obs[side * OBS_DIM..(side + 1) * OBS_DIM]).try_into().map_err(|_| EnvError::fault("obs row slice"))?;
            chain.encode(side, row).map_err(core_err(&format!("encode p{tag}")))?;
            let legal = chain.legal(side).ok_or_else(|| EnvError::fault(format!("p{tag}: a decision with no legality")))?;
            let s = chain.stream(side).ok_or_else(|| EnvError::fault(format!("p{tag}: the chain lost its stream")))?;
            let tokens = present::choice_tokens(&s.board_reading, &legal).map_err(core_err(&format!("tokens p{tag}")))?;
            c.mask[side * ACT..(side + 1) * ACT].copy_from_slice(&present::mask(&legal));
            if !ctx.spec.labels.is_empty() {
                let other = 1 - side;
                let truth = self.chains[other]
                    .as_ref()
                    .and_then(|ch| ch.stream(other))
                    .ok_or_else(|| EnvError::fault(format!("p{tag}: the truth side's chain lost its stream")))?;
                let mut row_now = [0f32; OBS_DIM]; // on the stack: an op allocates nothing new for labels
                row_now.copy_from_slice(&c.obs[side * OBS_DIM..(side + 1) * OBS_DIM]);
                crate::labels::write(&ctx.spec.labels, side, &s.board_reading, &truth.board_reading, &row_now, &mut self.labels, c)
                    .map_err(|e| EnvError::fault(format!("labels p{tag}: {e}")))?;
            }
            let n = self.decided[side];
            self.decided[side] = chain.trackers(side).map_or(0, |tr| tr.decisions);
            self.open[side] = Some(Open { tokens, n });
            t.decisions += 1;
        }
        self.emitted = self.sess.as_ref().map_or(0, |s| s.chunks().chunks.len());
        Ok(())
    }

    /// Feed one side's action (an index the caller chose from the mask).
    fn feed(&mut self, ctx: &Ctx, side: usize, action: i32, c: &mut EnvCols, t: &mut Tally) -> Result<(), EnvError> {
        let open = self.open[side].take().ok_or_else(|| EnvError::fault(format!("p{}: no decision open", side + 1)))?;
        let tok = open
            .tokens
            .iter()
            .find(|(i, _)| *i as i64 == action as i64)
            .map(|(_, tk)| tk.clone())
            .ok_or_else(|| {
                EnvError::caller(format!("env {}: p{} action {action} is not legal here (the mask forbids it)", self.id, side + 1))
            })?;
        let choice = parse_choice(&tok).ok_or_else(|| EnvError::fault(format!("the mapper produced an unparseable token {tok:?}")))?;
        // `sim_bridge`'s order: the chain notes the raw token BEFORE the command is fed.
        if let Some(ch) = self.chains[side].as_mut() {
            ch.note_choice(side, &tok);
        }
        self.log.cmds.push(format!("CHOOSE p{} {tok}", side + 1));
        self.sess.as_mut().ok_or_else(|| EnvError::fault("no battle"))?.feed_cmd(Cmd { side, choice }, &ctx.dex);
        self.advance(ctx, c, t)
    }

    /// The placeholder turn limit: p1 forfeits (`FORCELOSE p1`), as `sim_bridge` runs it.
    fn forfeit_p1(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally) -> Result<(), EnvError> {
        self.log.cmds.push("FORCELOSE p1".into());
        self.sess.as_mut().ok_or_else(|| EnvError::fault("no battle"))?.forfeit(0);
        self.open = [None, None];
        self.advance(ctx, c, t)
    }

    fn write_state(&self, c: &mut EnvCols) -> Result<(), EnvError> {
        let mut any = false;
        for side in 0..SIDES {
            let o = self.open[side].as_ref();
            c.need[side] = o.is_some() as u8;
            c.dec_n[side] = o.map_or(0, |o| o.n);
            any |= o.is_some();
        }
        c.episode[0] = self.episode();
        c.turn[0] = self.sess.as_ref().map_or(0, |s| s.turn());
        if !any {
            return Err(EnvError::fault("stuck: no decision open and the battle is not over"));
        }
        Ok(())
    }

    /// RESET: start a fresh episode.
    pub fn reset(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally) -> Result<(), EnvError> {
        c.reward[0] = 0.0;
        c.done[0] = 0;
        c.refused[0] = 0;
        self.start(ctx, c, t)?;
        self.write_state(c)
    }

    /// STEP: feed every open side (p1 first — the script order), then end / auto-reset.
    pub fn step(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally) -> Result<(), EnvError> {
        c.reward[0] = 0.0;
        c.done[0] = 0;
        c.refused[0] = 0;
        let act = [c.action[0], c.action[1]];
        let to_feed: Vec<usize> = (0..SIDES).filter(|&s| self.open[s].is_some()).collect();
        for side in to_feed {
            if self.sess.as_ref().is_some_and(BridgeSession::is_ended) {
                break;
            }
            // A side whose decision a previous feed of this op closed (none today) is skipped.
            if self.open[side].is_some() {
                self.feed(ctx, side, act[side], c, t)?;
            }
        }
        let sess = self.sess.as_ref().ok_or_else(|| EnvError::fault("no battle"))?;
        if !sess.is_ended() {
            if let Some(limit) = ctx.spec.turn_limit {
                if sess.turn() > limit {
                    self.forfeit_p1(ctx, c, t)?;
                }
            }
        }
        let sess = self.sess.as_ref().ok_or_else(|| EnvError::fault("no battle"))?;
        if sess.is_ended() {
            let r = match sess.winner() {
                Some(0) => 1.0,
                Some(_) => -1.0,
                None => 0.0,
            };
            t.ended += 1;
            self.start(ctx, c, t)?;
            c.reward[0] = r;
            c.done[0] = 1;
        }
        self.write_state(c)
    }

    /// QUARANTINE: the battle in progress is dropped (its log was banked by the caller) and the
    /// next episode starts from the staged inputs.
    pub fn quarantine(&mut self, ctx: &Ctx, c: &mut EnvCols, t: &mut Tally) -> Result<(), EnvError> {
        self.start(ctx, c, t)?;
        c.reward[0] = 0.0;
        c.done[0] = 1;
        c.refused[0] = 1;
        self.write_state(c)
    }
}

// ------------------------------------------------------------------ one block (one worker's envs)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Job {
    Reset,
    Step,
}

/// What one env reports back from an op.
pub struct EnvReport {
    pub env: usize,
    pub tally: Tally,
    /// A quarantine this op (the banked battle's error + input log + episode).
    pub quarantined: Option<(EnvError, InputLog, u32)>,
    /// A batch failure (the class is not `Quarantine`), with the env's input log.
    pub failed: Option<(EnvError, InputLog, u32)>,
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "a panic with a non-string payload".into())
}

fn run_block(ctx: &Ctx, envs: &mut [Env], lo: usize, job: Job, cols: ColAddrs) -> Vec<EnvReport> {
    let mut out = Vec::with_capacity(envs.len());
    for (k, env) in envs.iter_mut().enumerate() {
        let i = lo + k;
        let mut tally = Tally::default();
        // SAFETY: the pool validated `cols` at freeze for n envs; env i's rows belong to this worker.
        let r = catch_unwind(AssertUnwindSafe(|| {
            let mut c = unsafe { cols.env(i) };
            match job {
                Job::Reset => env.reset(ctx, &mut c, &mut tally),
                Job::Step => env.step(ctx, &mut c, &mut tally),
            }
        }))
        .unwrap_or_else(|p| Err(EnvError::panic(format!("PANIC inside the core: {}", panic_msg(p)))));
        let mut rep = EnvReport { env: i, tally, quarantined: None, failed: None };
        match r {
            Ok(()) => {}
            Err(e) if e.class == Class::Quarantine => {
                let banked = (e, env.log.clone(), env.episode());
                let again = catch_unwind(AssertUnwindSafe(|| {
                    let mut c = unsafe { cols.env(i) };
                    env.quarantine(ctx, &mut c, &mut rep.tally)
                }))
                .unwrap_or_else(|p| Err(EnvError::panic(format!("PANIC in the reset after a quarantine: {}", panic_msg(p)))));
                match again {
                    Ok(()) => rep.quarantined = Some(banked),
                    Err(e2) => {
                        // The next episode cannot start either: nothing is left to quarantine INTO.
                        let class = if e2.class == Class::Quarantine { Class::Fault } else { e2.class };
                        let msg = format!("{} — and the reset after its quarantine failed: {}", banked.0.message, e2.message);
                        rep.failed = Some((EnvError { class, kind: e2.kind, py_class: e2.py_class, message: msg }, banked.1, banked.2));
                    }
                }
            }
            Err(e) => rep.failed = Some((e, env.log.clone(), env.episode())),
        }
        out.push(rep);
    }
    out
}

// ------------------------------------------------------------------ the pool

struct Worker {
    tx: Sender<Option<(Job, ColAddrs)>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// See the module docs. Built by [`Pool::new`]; driven by `core::dispatch`.
pub struct Pool {
    pub ctx: Arc<Ctx>,
    inline: Vec<Env>,
    workers: Vec<Worker>,
    done_rx: Option<Receiver<Vec<EnvReport>>>,
    pub(crate) bound: Option<ColAddrs>,
    /// Lifecycle counters that `new` alone may move (kept for the `*_AFTER_FREEZE` columns).
    pub(crate) threads_after_freeze: u64,
    pub(crate) envs_after_freeze: u64,
}

impl Pool {
    /// STARTUP: acquire everything. Every team of the table is unpacked here, so a bad team fails
    /// now rather than at its first draw mid-run.
    pub fn new(spec: Spec) -> Result<Pool, String> {
        spec.validate()?;
        let dex = Dex::for_gen(3);
        // Every team is VALIDATED by use, at startup: it unpacks, the engine constructs a battle
        // with it (a mirror, a fixed seed), and a parse chain roots on it — so a team the core
        // cannot run fails here rather than as a quarantine at its first draw mid-run.
        for (i, t) in spec.teams.iter().enumerate() {
            let sets = pokesim::team::unpack(t, &dex).map_err(|e| format!("spec: team {i} does not unpack: {e}"))?;
            if sets.is_empty() {
                return Err(format!("spec: team {i} is empty"));
            }
            let opts = BattleOptions {
                format_id: spec.format_id.clone(),
                seed: Some("0,0,0,0".into()),
                p1: PlayerOptions { name: spec.names[0].clone(), team: PackedTeam(t.clone()) },
                p2: PlayerOptions { name: spec.names[1].clone(), team: PackedTeam(t.clone()) },
            };
            let sess = BridgeSession::new_construct_turn0(&opts, &dex).map_err(|e| format!("spec: team {i}: the engine refuses it: {e}"))?;
            if let Some(f) = sess.fatal() {
                return Err(format!("spec: team {i}: the engine refuses it: {f}"));
            }
            for side in 0..SIDES {
                BattleVersion::parse_root_unrecorded(side, &spec.names[side], Some(t), spec.clock)
                    .map_err(|e| format!("spec: team {i}: the reading refuses it as p{}: {}", side + 1, e.message()))?;
            }
        }
        if let Some(d) = &spec.bank_dir {
            std::fs::create_dir_all(d).map_err(|e| format!("spec: bank_dir {}: {e}", d.display()))?;
        }
        let n = spec.n;
        let t = spec.effective_threads();
        let ctx = Arc::new(Ctx { dex, spec });
        let envs: Vec<Env> = (0..n).map(Env::new).collect();
        let mut pool = Pool {
            ctx: Arc::clone(&ctx),
            inline: Vec::new(),
            workers: Vec::new(),
            done_rx: None,
            bound: None,
            threads_after_freeze: 0,
            envs_after_freeze: 0,
        };
        if t <= 1 {
            pool.inline = envs;
            return Ok(pool);
        }
        let (dtx, drx) = channel::<Vec<EnvReport>>();
        let mut it = envs.into_iter();
        let (base, extra) = (n / t, n % t);
        let mut lo = 0;
        for w in 0..t {
            let cnt = base + usize::from(w < extra);
            let mut block: Vec<Env> = it.by_ref().take(cnt).collect();
            let (tx, rx) = channel::<Option<(Job, ColAddrs)>>();
            let dtx = dtx.clone();
            let ctx = Arc::clone(&ctx);
            let my_lo = lo;
            lo += cnt;
            let handle = std::thread::Builder::new()
                .name(format!("rust_env_w{w}"))
                .spawn(move || {
                    while let Ok(Some((job, cols))) = rx.recv() {
                        if dtx.send(run_block(&ctx, &mut block, my_lo, job, cols)).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|e| format!("spawning worker {w}: {e}"))?;
            pool.workers.push(Worker { tx, handle: Some(handle) });
        }
        pool.done_rx = Some(drx);
        Ok(pool)
    }

    pub fn n(&self) -> usize {
        self.ctx.spec.n
    }

    /// Env `i` of an INLINE pool (`threads <= 1`) — harness access; a threaded pool's envs live on
    /// their workers and are not readable from here.
    pub fn inline_env(&self, i: usize) -> Option<&Env> {
        self.inline.get(i)
    }

    pub fn is_frozen(&self) -> bool {
        self.bound.is_some()
    }

    /// Run `job` over every env; the reports come back in ENV order whatever the scheduling.
    pub(crate) fn run(&mut self, job: Job, cols: ColAddrs) -> Result<Vec<EnvReport>, String> {
        if self.workers.is_empty() {
            return Ok(run_block(&self.ctx, &mut self.inline, 0, job, cols));
        }
        for w in &self.workers {
            w.tx.send(Some((job, cols))).map_err(|_| "a worker thread is gone")?;
        }
        let rx = self.done_rx.as_ref().ok_or("no result channel")?;
        let mut all = Vec::with_capacity(self.n());
        for _ in 0..self.workers.len() {
            all.extend(rx.recv().map_err(|_| "a worker thread is gone")?);
        }
        all.sort_by_key(|r| r.env);
        Ok(all)
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
