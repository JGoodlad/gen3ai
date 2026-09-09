# Training + launcher — the full chapter

Moved verbatim out of the root `CLAUDE.md` on **2026-09-07** (see `designs/ops/testing.md` for the
same note). **The root keeps the launch commands, the resume/fork hazards and the flag names; this
file is the detail behind them**, and `src/main/launcher/CLAUDE.md` +
`src/agents/training/CLAUDE.md` own the per-flag semantics.

Covers: the launcher and its restart loop, `main.checkargs` and the arch-surface guard, fresh /
resume / fork launches, `--critic`, the in-process bridge transport, the two compile flags,
`--async-rollout`, bot evaluation, the untaught meter, the critic gate, ELO, and exploitability.

---

## Launcher (preferred for long runs)

`src/main/launcher/` wraps `train_rl_agent.py` with **periodic restarts** (reclaim pymalloc
fragmentation; child saves a checkpoint on SIGTERM, launcher relaunches), **crash auto-restart**
(a self-crash relaunches from the last checkpoint after dumping a per-crash
`<run_dir>/crashes/restart_err_<token>.txt`, with a `--max-crash-restarts` circuit-breaker), **git-worktree
isolation** (agent pushes to `main` never affect a running session), a **Textual TUI** (metrics,
FPS, restart countdown, a `↻ N restarts (M crash)` badge; `l` logs, `e` events, `d` dashboard,
`r` restart, `c` checkpoint, `p` plots, `s` status, `f` force eval (confirm-gated; rejected if an
eval cycle is already running), `q`/ctrl-c quit, `v` copy mode
[freeze + native terminal select-and-copy — the portable copy path, works on Terminal.app]),
and **live crash-log
streaming** to `<run_dir>/launcher_child.log`.

The UI is **Textual** (built on the shared `src/main/tui/` base), launched with
`python -m main.launcher …` (or the back-compat alias `python -m main.launcher.tui …`). A closed
terminal (SIGHUP) or external `kill` (SIGTERM) is caught and turned into a clean,
checkpoint-saving shutdown rather than a lost checkpoint.

**A detached launch (`nohup … < /dev/null &`, systemd, cron) runs HEADLESS automatically** — with
no TTY on stdin, Textual's input thread would otherwise busy-loop a whole core forever (measured
on a live run: 96% of a core for 13 h, plus a 982 MB log of full-screen ANSI repaints growing at
17 KB/s into a redirected file). Headless drops the input thread and the repaints, and events are
echoed as plain `[HH:MM:SS] …` lines instead, so `tail -f` on the redirect target still follows
the run. A TTY keeps the full interactive TUI. Detail: `src/main/launcher/CLAUDE.md`.

**Internals — how the UI reconciles the restart loop with Textual's event loop, the
quit/ctrl-c/SIGHUP teardown, crash reporting + auto-restart, exit codes
(`COMPLETE`/`INTERRUPTED`/`CRASH`/`FATAL_CONFIG` — the last gives up without restarting on an
arch/config mismatch instead of looping), the full flag table (`--restart-interval-hours`,
`--restart-grace-minutes`, `--max-crash-restarts`, `--nice`, `--no-pin`, `--sync-to-main`), the
resume contract, and the `:8001` Showdown-port default — live in `src/main/launcher/CLAUDE.md`.**

**The launcher and everything it spawns run at `--nice 10` by default** (`0` disables). A run holds
~940 processes; at nice 0 it competes on equal terms with interactive work sharing the box.
Niceness is inherited across fork/exec, so one call before the first child covers the trainer, its
SubprocVecEnv workers and every eval worker, across periodic and crash restarts alike. **On an idle
box this changes nothing** — niceness only arbitrates under contention.

### Will this command still launch? — `python -m main.checkargs`

A run's recorded `launcher_command` outlives the flags in it, and **argparse reports only the FIRST
unrecognized flag** — so relaunching an old argv after a deletion is a launch-crash-fix loop.
`checkargs` answers offline, in one pass, without touching `models/`:

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.checkargs models/<run>                 # validate that run's recorded command
python -m main.checkargs --argv "--steps 1 --device cuda"
```

Exit 0 = every flag is accepted; exit 1 names each one that would fail. It answers **two
independent questions**, and passing the first says nothing about the second:

1. **Will it launch?** — a flag the parser no longer knows; a COMBINATION the extractor refuses
   (`flag_registry`'s `requires` graph); and every value-conditional rule in
   **`main.train.combination_checks`**, the ONE list the launch path and `checkargs` both read.
   A cross-flag `parser.error` in `config.py` FAILS `combination_checks_test.py` (AST-scanned,
   allowlist EMPTY), so "every" is enforced rather than intended.
2. **Is this the ARCHITECTURE you meant?** — the **ARCH SURFACE vs `designs/production_config.json`**
   diff (`main.train.arch_surface.report`, one function shared by `checkargs`, `--dry-run` and the
   launcher). Its key set is DERIVED from `flag_registry.arch_surface_flags()`, never hand-listed.
   On a FRESH argv a non-empty diff **REFUSES** unless `--allow-nonproduction-arch`; a FORK/RESTART
   is exempt but still printed (it inherits its parent's surface); a PINNED launch is ADVISORY.

🚨 **The two failures are not interchangeable.** A refused flag combination is LOUD and pre-launch —
nothing starts, it costs a minute. **Arch drift is SILENT and post-launch**: everything parses, the
run starts, and the config diff is the only thing that would have told you. On 2026-09-06 an arm
trained a near-bare network for 24.4M steps after three gates passed, all three correctly. So
`checkargs` closes an arch-only failure with its own verdict — *"✗ this command LAUNCHES — and
builds the wrong architecture"*.

**`--arch production`** is the remedy: it applies every ARCH-surface key from
`designs/production_config.json` as if typed, and records `arch_source` in `model_config.json`. It
deliberately does NOT set the CRITIC readouts, `--belief-grad-mode`, or the six SUPERVISION DOSES —
the block lists all of them every time, so its silence is never read as coverage.

Three resolution rules the tool applies, each of which has burned a launch:
- 🚨 **A BARE RUN DIRECTORY MEANS THE RUN'S LAST SNAPSHOT** (`gen3_last_snapshot_resolution_v1`) —
  through the one choke point `fixed_opponent_pool.resolve_model_ref`. Name the `.zip` or `@step`
  to pin a file. Detail: `src/agents/training/CLAUDE.md` → *WHICH FILE a run spec names*.
- 🚨 **AN ARGV IS NOT A CONFIG** — with `--model`, every unnamed flag is INHERITED from the
  checkpoint's `model_config.json`, so the thing that launches is the argv OVERLAID ON THE PARENT.
  `checkargs` builds that effective namespace and labels each finding argv-vs-`INHERITED`.
- 🚨 **A PINNED argv is judged by the PINNED commit's parser** — a flag whose ARITY changed is
  invisible to a presence check. The commit comes from the launcher's own `resolve_pin`.

**`python -m main.launcher --dry-run` is the EXECUTING complement, and the one that is safe on a
same-run RESTART**: it resolves the actual launch on this box (role, run dir, pin, `+X steps`,
inherited config, pool) and exits without creating anything. Reach for it instead of "launch the
real command and kill it" — that habit is harmless on a fork and DESTRUCTIVE on a restart.

Full detail — the four incident narratives behind these guards, the flag tables, and the parser
mechanics — is in `src/main/launcher/CLAUDE.md` (§ *Validating a launch without launching*, § *the
ARCH-SURFACE guard*, § *An argv is validated by the parser of the tree that will RUN it*) and
`designs/research_state/claude_md_archive/checkargs_incidents.md`.
### Starting a fresh run via launcher
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.launcher \
  --restart-interval-hours 3 \
  --steps 15000000 \
  --n-envs 64 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

### Resuming from a checkpoint
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m main.launcher \
  --restart-interval-hours 3 \
  --model models/<run>/checkpoints/checkpoint_NNNN_steps.zip \
  --steps 15000000 \
  --device cuda
```

Periodic + forced checkpoints live in `models/<run>/checkpoints/` (each `.zip` beside its
`.json` sidecar); legacy runs kept them at the run root and still resume. The checkpoint must
carry a `metadata.json` with a `git_hash`; the launcher pins the isolated
worktree to that commit so the resumed run uses the original code (override with
`--sync-to-main` for HEAD, or **`--pin-commit <sha>`** to name the commit outright — the way to
keep a whole batch of arms on ONE commit when something may land on `main` mid-batch, since it
is refused on a same-run restart that would move the pin). All non-launcher flags are forwarded
verbatim to `train_rl_agent.py`.
`python -m main.launcher.tui …` is an alias for the same command.

🚨 **A FORK starts with an EMPTY self-play pool, and an empty pool does not disable `--self-play` —
it falls back to the BOT pool.** A genuine fork with `--self-play` now **auto-seeds** its parent's
pool (every `snapshot_*.zip` PLUS `summary.json` / `win_rate_vs_bots.txt` / `model_config.json` — the
zips alone still read `self_play_fraction=0%`, because the starting fraction comes from the metadata)
and prints `🌱 [SELFPLAY] [pool] seeded N snapshots + metadata from <parent>`; a fork that STILL has
no pool exits `FATAL_CONFIG` naming the three ways out rather than silently training against bots.
A launcher RESTART never re-seeds, a non-empty pool is never touched, and a FRESH run is unchanged.
`--no-fork-pool-seed` opts out of the seeding; `--allow-empty-pool` consents to the bot fallback.
Detail: `src/agents/training/CLAUDE.md` → *A FORK starts POOLLESS*.

🚨 **`--lr`, `--batch-size` and `--n-steps` are INERT on a resume** — the resume path restores the
checkpoint's own optimizer LR and prints `(arg --lr=… ignored on resume)`, so a FORK inherits
whatever rate the PARENT's KL controller had annealed to. Measured (ledger M7): three distillation
folds launched with the same flags ran at a median **5.8e-5 / 2.8e-5 / 1.0e-4**, and the quantity
that predicts a fold's collateral is the **DOSE** = `lr × n_epochs / (batch_size × grad_accum_steps)`
— v8's fold was 2.15e-8, every gen-era fold 2–6.6× higher. **`--fork-lr FLOAT`** pins that: on a
genuine fork it sets the optimizer LR *and* `model.lr_schedule` at load and seeds the KL controller
from it (still clamped into `[--min-lr, --max-lr]`). **`--fork-lr-freeze`** additionally disables
the KL adaptation and the two-phase cosine, so the fold runs at one constant, recordable step size.

**The pin applies ONLY on a genuine fork** — a checkpoint from OUTSIDE the target run dir. A
launcher PERIODIC RESTART re-invokes the same argv into the same run dir every N hours, and
re-pinning there would reset the controller's adapted rate forever; the FREEZE is a property of the
run and does persist (re-read from the pin recorded in `metadata.json`). `--fork-lr` on a fresh run
is refused (use `--lr`). Every metadata write records a **`dose`** block (`lr_now`, `lr_flag`,
`fork_lr`, `lr_frozen`, `effective_batch`, `updates_per_env_step`, `dose_rate_now`,
`kl_controller`), and `train/dose_rate` + `train/effective_batch` ride TensorBoard every rollout.
Read a run's dose — including every run already on disk — with **`python -m main.dose <run>…`**.

---

## Training

Run directly (no restart loop, no worktree isolation):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/main/train_rl_agent.py \
  --model <path/to/checkpoint.zip> \
  --steps 15000000 \
  --n-envs 64 \
  --batch-size 16384 \
  --n-epochs 10 \
  --ent-coef 0.02 \
  --n-steps 2048 \
  --lr 0.0003 \
  --device cuda \
  --log-level periodic
```

Omit `--model` to start a fresh run. Use `--debug` for a single env (DummyVecEnv). Use `--device cpu` on machines without a GPU.

**GPU OOM lever — `--grad-accum-steps K`:** keep a large effective batch when the full minibatch
won't fit. It runs K `--batch-size` micro-batches and steps the optimizer once per group of K,
giving the **exact** gradient of a `batch_size·K` batch at the activation-memory cost of one
micro-batch (stock SB3 steps per-minibatch, so `--batch-size` alone couples effective batch to the
memory peak). E.g. `--batch-size 4096 --grad-accum-steps 4` trains like `--batch-size 16384` at ~¼
the peak. `K=1` (default) is byte-identical to stock; it's a train-loop knob (not version-locked) —
forward it on every resume like `--batch-size`. With `K>=2` it also emits a **`train/noise_scale`**
diagnostic (McCandlish critical batch size) that tells you, as a number, whether your effective batch
is too small / about right / bigger than needed. 🚨 **Read it beside the PER-TERM scalars, never
alone** (`train/noise_scale{,_ratio,_share}_{policy,value,entropy,aux,distill}`, default ON): the
total is measured on the SUM of every loss term, and this tree's dozen dense supervised aux heads
have far lower gradient noise than the clipped surrogate — so a total reading "over-batched" can be
aux DEFLATION rather than a batch that is too big, and the advisor says so explicitly when the two
disagree. **`--adaptive-batch {off,total,policy}`** (OFF by default, byte-identical off) closes that
loop: it holds the chosen term's noise ratio near `--adaptive-batch-target` by doubling/halving K
itself — never `--batch-size`, so every forward shape and the activation peak are unchanged and
`--compile-trainer` never notices — with the moved K persisted in the checkpoint sidecar. It is the
SLOW controller of the two; at fixed `target_kl` a bigger K makes the KL lr loop raise lr, so the
dose is a product of both and `train/dose_rate` is what the operator watches. Details:
`src/agents/training/CLAUDE.md` → Gradient accumulation, and → `--adaptive-batch`.

Checkpoints are saved automatically into `models/run_<timestamp>/checkpoints/` (each `.zip`
beside its per-checkpoint `.json` sidecar); the run-level `model_config.json` / `metadata.json`
/ `latest.txt` and the `final_model*.zip` / `best_model/` stay at the run root.

### WHICH readout is the critic — `--critic {shaped,winprob}` (default `shaped`)

`policy._critic_value` has a MODE. **`shaped` is the default and every generation to date** — the
scalar `value_net` (or distributional `E[Z]`) de-normalized through PopArt into raw shaped-return
units, with the win-prob head an auxiliary BCE. A flagless run is byte-identical.

**`winprob` promotes the head to BE the critic**: `V(s) = sigmoid(win_head logit)` in [0,1], the
value loss IS that head's BCE against the terminal outcome (weighted by `--vf-coef`, **not**
`--win-prob-coef` — one critic, one coefficient), and the reward stream is the TERMINAL **WIN
INDICATOR** alone. At `--victory-value 1.0` and `--gamma 1.0` the undiscounted return from any state
is exactly `1{win}`, so **`V(s) = P(win | s)` with no approximation term** — which is why
`--terminal-indicator` and `--victory-value 1.0` are requirements, not suggestions. PopArt is
refused (a bounded Bernoulli payoff has no scale to track).

🚨 **A critic bounded in [0,1] cannot represent "a timeout is worse than a loss."** That ordering is
not merely unused under `winprob` — it is *unrepresentable*, so `--draw-penalty` is REFUSED. The
anti-stall pressure comes from the obs deadline clock plus **`--arm-no-progress-tax`**. **Stall rate
and mean episode length are PRIMARY endpoints on any `winprob` arm, not monitored ones.**

Three flags are IMPLIED (`--win-prob-mode shaping`, `--gamma 1.0`, `--no-use-popart`) because their
argparse default is a `None` sentinel; four are REQUIRED and named by their own refusal
(`--no-hand-shaping`, `--terminal-indicator`, `--victory-value 1.0`, `--draw-penalty 0`) because
theirs are concrete and an implication could not be told apart from an overwrite. Everything the
mode SUBSUMES is refused rather than ignored. `python -m main.checkargs` reports every one offline.

**`--win-prob-pbrs-frozen <run|zip>` is the exception and is BUILDABLE here**
(`gen3_frozen_phi_actor_only_v1`): a frozen checkpoint's win-prob head supplies an ACTOR-ONLY
potential — `γφ(s′) − φ(s)` reaches the POLICY's advantages and nothing else, so the critic keeps
regressing the unshaped terminal indicator and `V ≡ P(win)` survives bit-for-bit. No coefficient
(φ is already in the value currency). **`--critic` is STRUCTURAL and resume-immutable.**

Design of record:
[`designs/ai_v12/design_winprob_only_critic.md`](designs/ai_v12/design_winprob_only_critic.md);
flag mechanics in `src/agents/model/CLAUDE.md` → *The CRITIC MODE*.

**`--gamma` is a flag** (it was hardcoded at `0.9999`). Its `shaped` default is
`reward_weights.PBRS_GAMMA` itself, so the PBRS invariance premise cannot break on a second copy of
the number, and it is **INERT ON A RESUME** like `--lr`.
### In-process bridge transport (`--use-bridge {off,node,rust}`, **default `rust`**)

`--use-bridge` swaps **both training and eval** between a websocket Showdown server and an
in-process `BattleStream` subprocess — no server, no port, no `/challenge` connection storm,
deterministic delivery (poke-env issue #907). **THE DEFAULT IS `rust`, so a run needs no Showdown
server at all.** It reuses the *entire* obs/reward/mask/wrapper stack unchanged.

| value | transport | when |
|---|---|---|
| **`rust`** | the std-only pokesim `src/rust_sim/src/bin/sim_bridge.rs` | **the DEFAULT** — fastest, smallest child, serverless |
| `node` | the Node `local_sim_bridge.js` | the explicit A/B arm; the parity harness and `gen_sim_bridge_diff.js` need it |
| `off` | websocket to a Showdown server on `--showdown-port` | the ladder / live-server path |

`rust` is a byte-for-byte protocol-compatible drop-in for `node`, so nothing above the transport
changes. Binaries resolve through `sim_bridge_bin.py::resolve_sim_bridge_bin()` /
`resolve_search_driver_bin()` — `$POKESIM_SIM_BRIDGE_BIN` / `$POKESIM_SEARCH_DRIVER_BIN` first,
else a cached `cargo build --release` — with **a clear error and never a silent node fall-back**.
The DEPRECATED `--use-showdown-bridge` alias is **DELETED**.

**Why the default is `rust`, and it is not throughput.** At production `n_envs` the end-to-end FPS
gain over websocket is only ~5%; the case is **operational** — no server, no port, no connection
storm, no RAM-creep, deterministic delivery — and those hold on every run. `rust` over `node` *is*
measured (1.41x at `--n-envs 48`; a ~9 MB child against node's ~224 MB). SEARCH also runs on rust,
though every `impl=` default is still `"node"`.

**Operational facts worth knowing:**
- A bridge child that **dies** mid-run **crashes** the env (launcher restart) — resuming risks a
  corrupted PPO transition. RSS is flat, so the 3 h launcher restart owns the lifecycle and
  `recycle_every` (5000) never fires under it.
- Eval plays **one game at a time per worker** by default (`--eval-concurrency-per-worker 1`);
  raising it is asyncio latency-hiding, not multi-core, and nets negative under training contention.
- The launcher treats an ABSENT `--use-bridge` as a bridge run, so it injects no phantom
  `--showdown-port` (pinned by `default_port_test.py`).

🚨 **THE DURABLE LESSON — a "default" branch that nothing tests is untested however green the suite
is.** Three seed defects shipped on this path because every gate was inherently SEEDED or compared
only aggregates, so the production SEEDLESS branch was never exercised. Two more killed production
launches because the fuzz drives only masked-LEGAL tokens. A related trap: **an allowlist entry can
outlive its own fix and then mislead every reader after it**, including a subagent briefed from it.

Detail — the (all FIXED) seed and CHOOSE-path defects, the parity-gate inventory, the coverage
census and the throughput tables — is in `src/utils/bridge/README.md`, `src/rust_sim/CLAUDE.md`,
`designs/ai_v5/design_local_sim_bridge_transport.md`, and
`designs/research_state/claude_md_archive/bridge_transport_history.md`.
### The two compile flags — CPU opponents vs GPU trainer

`torch.compile` is applied at two independent sites, split by WHO and WHERE. They were one flag
(`--compile-extractor`) until 2026-08-14, which named neither half:

| flag | what | device | **default** | opt-out | on failure |
|---|---|---|---|---|---|
| `--compile-opponents` | each frozen self-play OPPONENT's extractor, in the env workers | **CPU** | **ON** | `--no-compile-opponents` | warn + fall back to eager; `--compile-opponents-strict` opts into raising |
| `--compile-opponents-preload` | the same compile, done ONCE in the forkserver and inherited by fork | **CPU** | **follows `--compile-opponents`** | `--no-compile-opponents-preload` | RAISES at env construction (never a silent wedge) |
| `--compile-trainer` | the LEARNER's extractor — the fwd **and bwd** of the PPO step | **CUDA** | **AUTO — ON when the resolved device is cuda**, OFF on cpu, OFF under `--debug` | `--no-compile-trainer` | **always FATAL** |

**🚨 These default ON (2026-08-17). They are FALLBACKS, not opt-ins** — the flags exist so a run can
be turned back to eager when something is wrong, not so a run can opt into speed. A launch that
types none of them compiles.

`--compile-trainer` is the one that cannot be a flat `True`: it **refuses** a non-cuda device by
design, so a flat default would turn every working CPU invocation into a `FATAL_CONFIG` exit. Its
default is therefore conditioned on the resolved device (`auto` follows the box), and `--debug` is
excluded outright even with an explicit `--device cuda`. **An EXPLICIT `--compile-trainer` on cpu
still refuses, loudly — the default was flipped, not the safety.**

Orthogonal — a run can take either, both, or neither. All are runtime perf knobs: never versioned,
never in `check_compatible`. **They are still not INHERITED on resume — but with the defaults ON
that now cuts the other way**: a flagless resume gets them ON, so it is the OPT-OUT you have to
re-pass each launch, not the flag.

### Compiled CPU opponents (`--compile-opponents`, **default ON**)

`torch.compile`s each frozen self-play OPPONENT's feature extractor in the env workers — the measured
**68% of rollout-worker time**, run on CPU at B=1 where the graph is dispatch-bound. A **runtime perf
knob**: never versioned, never in `check_compatible`. **ON by default**; `--no-compile-opponents`
falls the whole path back to eager, which is what to reach for when the compile is the suspect.

**Measured: B=1 CPU forward 6.371 → 0.976 ms (6.53×)** on the literal production arch (1 graph, 0
graph breaks, max|Δ| vs eager 5.07e-07), and **+33.3% marginal training FPS at `--n-envs 48`**
(406.5 → 541.8, disjoint ranges, 48/48 workers compiled) — the first throughput lever here the
`SubprocVecEnv` barrier does NOT absorb. But the per-forward win has **saturated**: doubling it
(3.6× → 6.53×) moved end-to-end only ~31% → ~33%, so the opponent forward is no longer the rollout
bottleneck and further compiler work on this path is spent effort.

Startup: **`--compile-opponents-preload`** (`gen3_forkserver_preload_v1`, 2026-08-16) compiles ONCE in
the forkserver so every worker inherits the traced graph (~0.12 s each instead of ~30 s). It
**follows `--compile-opponents`**, so it is on by default too; `--no-compile-opponents-preload`
keeps the opponent compile and falls back to `agents.model.compile_prewarm`, which warms the shared
on-disk Inductor cache in the trainer before any worker exists (**59.6 s -> 30.1 s** wall for 16
workers). The two fix DIFFERENT halves — the disk cache removes codegen, the fork removes
per-process dynamo tracing and guard construction.

**It is not the approach that hung a run, and the difference is structural, not a tuning.** The
2026-08 attempt wedged a 48-env run because `fork()` copies every mutex but only the calling thread
and the extractor import started poke-env's global asyncio loop thread. That root cause is FIXED
(the `poke_env` package inits are LAZY, so the extractor's import graph is thread-free — pinned by
`compile_prewarm_test.py`), and the preload additionally **asserts single-threadedness after its own
compile and RAISES**, which kills the forkserver bootstrap and fails `SubprocVecEnv` construction
with a traceback in the parent. The silent wedge is unrepresentable; the worst case is a loud
refusal at startup with a one-flag opt-out. **Untested at 48 workers** (proven live at 4) — see the
training leaf.
Failure is loud on stderr + the launcher event stream, and `--compile-opponents-strict` promotes it to
a hard error (falling back to eager is an invisible ~6.5× regression). "The model still compiles" is a
**default-on test** (`extractor_compiles_test.py`; `GEN3AI_SKIP_COMPILE_TESTS=1` opts out).

Full detail — the four guards, the Inductor crash root-caused to one op, and the startup-cost table —
is in `src/agents/training/CLAUDE.md` → Compiled CPU opponents.

### Compiled GPU trainer (`--compile-trainer`, **default ON for cuda**)

The other half, and the bigger one. **Measured on v76 at the production shape** (batch 4096, PopArt
on, the real `MaskablePPO` path, arms interleaved on an idle box): `policy.evaluate_actions`
forward+backward **155.1 → 88.5 ms = 1.75×**, i.e. **~+62% end-to-end FPS** at the ~89% train share.
Compiling the whole policy instead of just the extractor measured the same to within 0.004×, so the
extractor is what ships — same win, less graph.

**CUDA only, and fail-loud by design.** A silent fall back to eager would be a 1.75× regression that
no metric surfaces (the run trains correctly, just ~38% fewer steps/hour, forever), so a failed,
slower, or numerically-divergent compile is a hard `FATAL_CONFIG` exit rather than a warning — and
`--device cpu` is refused up front, because the CPU backward provably does not lower (Inductor's C++
backend refuses the damage op's `atomic_add` scatter).

**The DEFAULT yields; the REFUSAL does not.** `auto`/`cuda` with a card ⇒ on; `cpu`, any other
explicit device, or `--debug` ⇒ off. The auto path *also* runs `check_shape_stability` and stays
OFF (with a startup line saying why) when the config is one this flag refuses — `--async-rollout`,
or a rollout that does not divide by `--batch-size` — because a default must never turn a command
that works today into a `FATAL_CONFIG`. An explicit `--compile-trainer` on any of those still exits
`FATAL_CONFIG` with the message it always did. **A default yields to the config you typed and says
so; an explicit flag refuses.**

**⚠️ It drops the ObservationDebugger**, which dynamo cannot trace. That debugger is on in
production, so this is a real trade — and **with the default ON, every plain cuda run now makes it
without anyone typing a flag**, which is why it announces itself twice: once at startup when the
auto default resolves to on, and once from `compile_trainer_extractor` when the debugger is actually
dropped. `--no-compile-trainer` is how you keep it. Full detail — the four refusals, the
`state_dict` hazard, the measurement table — is in `src/agents/training/CLAUDE.md` → Compiled GPU
trainer.

### Non-barrier async rollout (`--async-rollout`, opt-in)

Stock `SubprocVecEnv.step()` is a **barrier** — each step waits for the *slowest* of N env workers,
so the latency-bound rollout (py-spy: ~86% wall, GPU ~86% idle) is straggler-gated and the policy
forward never overlaps env stepping. `--async-rollout` swaps in `AsyncSubprocVecEnv` + an on-policy
`collect_rollouts_async` (`src/agents/training/async_vec_env.py`) that keeps every worker
continuously in-flight and forwards whichever envs are **ready**, filling each env's own buffer
column. It stays **exactly on-policy** (PPO freezes the policy during collection — a scheduling
change, not an APPO-style algorithm change). Masks ride natively in the Dict obs (`obs["action_mask"]`);
`env_method` is drain-safe so the eval callback's mid-collection pushes don't desync.
**Measured (bridge, GPU forward, steady-state FPS): +20% at n_envs=16 (=logical cores); +14% at the
production `--n-envs 64` (1489→1695); `--async-rollout --n-envs 32` matches production `sync@64` FPS
with half the envs** (≈half the RAM). Off by default (= stock `SubprocVecEnv`), ignored under
`--debug`. Full design: `designs/ai_v5/design_async_rollout.md`.

### Bot evaluation

Bot eval runs in **frozen-snapshot subprocesses** (`--eval-workers`, default 5) that
**work-steal at battle granularity** from a shared pool and play the live server (or the bridge)
**without pausing training**; results merge into TensorBoard + TUI + best-model and land in
`metadata.json` as a top-level `latest_eval` block. Each opponent's `EVAL_GAMES` are split into
**shard units** (`--eval-shard-games`, default 25 → 4 shards/opponent; per-opponent game count
overridable with `--eval-games`) so any idle worker drains a
straggler's remaining games instead of one worker grinding a whole opponent — the long eval tail
collapses to one shard. The mechanism lives in the well-encapsulated **`eval_sharding/` package**
(deep `ShardedEvalPool` interface; aggregation is **exact** — Σwon/Σfinished etc., raw δ pooled then
one CVaR), with a documented **`rating.py` seam** (`MatchRecord` / `RatingModel` / `BradleyTerryRating`)
ready for a future Glicko-2/TrueSkill without touching the live ELO path. **`--self-play` eval shares
this exact non-blocking pipeline** (with the worker pool doubled to 10, since sentinel matchups infer
for both players) — the workers additionally work-steal the pool sentinels' shards, and a winning
cycle promotes its frozen snapshot into the pool by file-copy (`SnapshotPool.add_from_path`). The full
design (battle-level work-stealing, exact aggregation, graceful-shutdown drain, resume re-publish,
sentinels + promotion, `--eval-workers` / `--eval-shard-games` / `--eval-device`) is in
`src/agents/training/CLAUDE.md`.

### The UNTAUGHT METER — offline off-slice competence (`python -m main.untaught_meter`)

The win rate of a checkpoint **piloting** a fixed team slice against ONE fixed opponent,
cluster-bootstrapped over teams — the number every fold verdict in the ledger rests on. It lived
only as per-batch probe scripts copied between measurement directories (each with its own seed
convention and its own aggregation) until it became `src/main/untaught_meter.py` +
`src/agents/training/untaught_meter.py`. Offline: no training, no launcher, no server, nothing
written under `models/`. `--check` resolves every ref, team and opponent and exits non-zero on any
miss without playing; `--dry-run` prints the plan; `--from-rows` re-reads committed per-team
artifacts with no models at all. Refs resolve through the ONE choke point
(`fixed_opponent_pool.resolve_model_ref`), so a bare run dir means the run's LAST SNAPSHOT exactly
as a launch means it, and the resolved file + rung are printed and stamped in the JSON.

🚨 **A DELTA AGAINST A FROZEN PARENT OVERSTATES A FOLD, and the meter reports a second column for
it.** Ledger 2026-09-06 (cell 2): a plain +1.08M-step continuation of v8's parent — no teacher, no
distillation term, no stable opponents — moved this meter **+3.45pp [+0.46, +6.48]** by itself, so
re-based on a continuation control v8's celebrated +4.64pp becomes ≈ +1.2pp and is not significant.
`--control <arms…>` supplies the continuation arms at matched depth: they are pooled equal-weight,
carry their own **max-pairwise replicate floor**, and every ref gets `Δ vs baseline` *and*
`Δ vs continuation control`, each labelled SIGNIFICANT / WITHIN FLOOR / NOT DETECTED. Run without
`--control` and the report says in print what it is leaving out.

**Reproducibility takes BOTH halves and the meter enforces them**: all five `$GEN3AI_*_SEED` seams
per team plus per-battle sim and policy seeds, and `concurrency > 1` is **REFUSED** (seeded at
concurrency 3, two runs of the same measurement differed by up to +0.043 in level). `--workers N` shards
over TEAMS into single-concurrency processes; a cell is a pure function of (ref, team, battle), and
two `--workers 2` runs being byte-identical is a standing `sim`+`slow` gate. Timeouts are their own
bucket and a run above 25% is INCONCLUSIVE. The recipe, the seed table and the sharding rule are in
`src/agents/training/CLAUDE.md` → *The untaught meter*.

### The TRAINING-SIDE VALUE SIDECAR (`--value-sidecar`, `gen3_value_sidecar_v1`)

**Full topic: [`../training/value_sidecar.md`](../training/value_sidecar.md).**

Every other critic instrument in this runbook reads **eval** battles. The sidecar reads the
**training buffer** — the value PPO actually used, against `win_target`, the label the BCE actually
minimises. Once per rollout a seeded 1/64 of buffer states is appended to
`<run>/value_sidecar/rows.jsonl`; read it with `python -m main.ops.value_sidecar_read <run>
[--out DIR]`.

| Flag | Default |
|---|---|
| `--value-sidecar {auto,on,off}` | `auto` — **ON under `--critic winprob`**, off otherwise |
| `--value-sidecar-fraction` | `0.015625` (1/64 of buffer states) |
| `--value-sidecar-seed` | `0` (the sample is a function of *(seed, rollout index)*) |

It costs **19.3 ms per rollout** at production shape (measured 2026-09-08; 0.016% of a hostile 120 s
rollout, 0.57 MB per rollout) and writes nothing under `models/` beyond the run's own directory.
None of the three flags reaches `model_config.json`.

🚨 **It cannot be added later.** The sidecar reads the rollout buffer, which is gone the moment
`train()` returns — a launched arm either has it or has no training-side read for the rest of its
life. On a win-prob arm you have to opt OUT, which is the intended asymmetry.

🚨 **`main.critic_gate` / `main.ops.critic_read` and `value_sidecar_read` are not substitutes.** The
first two ask whether the critic is calibrated on the **eval** distribution (greedy trainee, fixed
roster, loss-preferring quota); the sidecar asks whether it is calibrated on the distribution it is
being **fit to**. A disagreement is a GENERALISATION finding, not a defect in either.

### The CRITIC GATE — a whole pre-registered read in one command (`python -m main.critic_gate`)

The win-prob-critic arm's read (`designs/ai_v12/design_winprob_only_critic.md` §5.5) is four
instruments, four output formats and a reader holding §4.3's bars in their head.
`python -m main.critic_gate <run> --parent <ref> --control <continuation refs…>` is that
composition — the anchored ladder at **matched SNAPSHOT COUNT and matched FIT SIZE** (the longer
ladder is REFIT on its first n; measured 2026-09-07, the un-refit comparison read a 30-Elo trail
as a 64-Elo lead) against the parent CONTINUED, the
§4.3 calibration gate per checkpoint with **RESOLUTION primary** and `bot`/`pool` never pooled, the
**G7 stall-rate + episode-length KILL condition**, and `main.untaught_meter` with its continuation
control — emitting one markdown report plus a JSON carrying every input path, every resolved file
(through the last-snapshot rule) and every threshold used. It **composes**: the ladder is the BT
fit's, the calibration numbers are `main.scaffolding_gauge --reliability --reliability-reweight`'s
(imported, and the gauge's reweighting REFUSAL is surfaced rather than softened), the meter's
numbers are the meter's. Verified against the committed baseline run: it reproduces §4.1's table
value for value.

🚨 **G1's bar is READ from `measurements/winprob_critic_baseline_2026-09-06/`, never hardcoded**, so
the bar and the record cannot drift apart — move the artifact's `resolution` and the verdict moves
with it (a gated test). A missing or unconverged ladder, a raw (un-reweighted) baseline artifact and
a missing matched stratum all **refuse with exit 2 naming the key**, in `main.exploitability`'s
pattern. **G1 flat with G2–G4 passing prints the design's falsification sentence VERBATIM as the
verdict** — never a pass with a footnote — and `critic_gate_test.py` re-reads the design file and
fails if the two drift apart. G5/G6/G8 are §6 gaps and are printed as NOT RUNNABLE on every report,
because a gate with unrunnable criteria quietly becomes the runnable ones. The G7 threshold is the
TOOL's default (the design registers G7 with no number) and every report says so. `--check` resolves
every input — refs, both ladders, the baseline artifact, the traces, the reweighting, the
episode-length source, and the meter's own `--check` — and exits non-zero naming **every** miss
without computing anything.

### ELO / skill rating

Under self-play pool play, `win_rate_vs_pool` is pinned near 50% by the promotion gate (a
sliding window of recent selves) and `win_rate_vs_bots` saturates — so neither tracks real
progress. The **ELO subsystem** fits an **anchored Bradley-Terry** rating over the win-records
every eval cycle already produces (trainee vs the 9 fixed bots + pool sentinels — no new
battles), giving one absolute number that rises with skill. Each cycle appends a row to an
append-only `<run>/eval_results.jsonl` and records a live `eval/elo` (+CI) to TensorBoard + a
`🏅 ELO` TUI badge. The fixed bots are the anchor; `python -m agents.training.bot_elo_calibration`
plays a one-time bot-vs-bot round-robin (bridge, no server) → `data/gen3_bot_elo_anchors.json`,
making snapshot ELOs **comparable across runs**. Offline: `python -m main.elo <run_dir>` prints a
ladder and plots an Elo-vs-step curve (and can backfill a running run from TensorBoard with
`--source tb`). A rating is a **transitive** model by construction, so the same CLI also prints the
**HodgeRank spine/width split** (`agents/training/hodge.py`) — the cyclic content a scalar ELO is
structurally blind to, tested against the binomial noise floor its own game counts imply, with two
weak per-cycle companions on TensorBoard (`eval/hodge_width_elo`, `eval/hodge_cyclic_fraction`).
Full design: `src/agents/training/CLAUDE.md` → ELO / skill rating.

🚨 **Reporting an ELO has three rules — read them before quoting a number.** The headline is
`<run>/snapshot_ladder/ladder.json` (dense, ±10) rather than `eval/elo` (±29); a rating is only
final once the run is, because BT re-solves every node on every add and the newest one is
**systematically inflated** (gen-10's 12M fell 2089 → 2021 over 12 refits); and a cross-run
comparison must be at matched snapshot COUNT, not matched step. The measured drift table and the
worked example are in `src/agents/training/CLAUDE.md` → *Reading an ELO*.

### Exploitability — the meter a weak opponent cannot inflate (`python -m main.exploitability`)

An ELO says how a generation does against the opponents it was measured on. **Exploitability** says
how much a BEST RESPONSE scores above it — the one progress meter that cannot be inflated by
beating weak opponents, and the quantity "the wheel turns twice" is a claim about (the best response
to generation N+1 should be WEAKER than the best response to generation N). Our exploiters ARE
best-response computations and the fleet admission table IS an exploitability measurement, so the
curve is pure bookkeeping over artifacts that already exist:

```bash
python -m main.exploitability \
  rev-2=designs/research_state/measurements/admission_artifacts/fleet_admission.json \
  rev-3=designs/research_state/measurements/admission_artifacts/r3_admission.json --md
```

The three admission artifacts are **committed** (2026-08-31) at
`designs/research_state/measurements/admission_artifacts/` with a provenance README — they lived
only in a session-scoped job directory until then, which made every exploitability claim in the
ledger un-reproducible the moment it was cleaned.

No battles, no models, no traces. Per generation it emits the best-response **net extraction**
(mean + max over teachers, CIs carried through from the artifact's own per-arm intervals), the
target identity, the ceiling/headroom reframe, the meter-vs-coverage team split, the budget/team
normalization, and a markdown row the ledger can quote. Four caveats ship WITH the number: it is a
**LOWER BOUND** (a finite-budget fork is an imperfect best responder), the teams are **PINNED** (a
subgame restriction — two generations compare only at matched team sets, and partial overlap is
flagged), the **max is selected** and upward-biased, and the two CIs assume different things. A
delta whose interval straddles zero reads "NO DETECTABLE CHANGE", never a direction.

**Schema drift REFUSES** (exit 2, naming the key), and the `net = teacher − reference =
ordered − seniority` identity is RECOMPUTED from the artifact's own per-team cells rather than
trusted — the recorded-vs-derived-key defect class has cost this program dearly enough. Theory
background: `designs/research_state/learning_notes/2026-08-28_nash_exploitability_psro.md`.

---
