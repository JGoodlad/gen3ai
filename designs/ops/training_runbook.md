# Training + launcher — the full chapter

Moved verbatim out of the root `CLAUDE.md` on **2026-09-07** (see `designs/ops/testing.md` for the
same note). **The root keeps the launch commands, the resume/fork hazards and the flag names; this
file is the detail behind them**, and `src/main/launcher/CLAUDE.md` +
`src/agents/training/CLAUDE.md` own the per-flag semantics.

Covers: the launcher and its restart loop, `main.checkargs` and the arch-surface guard, fresh /
resume / fork launches, the win-prob critic, the in-process bridge transport, the compile flag,
bot evaluation, the untaught meter, the critic gate, ELO, and exploitability.

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

🚨 **A restart re-launches from the RESUME role, and a FRESH argv never lands on an existing run**
(2026-09-26). Every periodic/crash restart strips the trainer's FRESH-only flags
(`combination_checks.fresh_only_flags()` — today `--arch`) and sets `--model <latest checkpoint>`, so a
`--arch production` launch survives its first restart (`ai_v14_01_base` did not: exit 2 ×3, ~40
GPU-min). A FRESH launch (no `--model`) whose run dir holds a checkpoint or `model_config.json` is
REFUSED `FATAL_CONFIG` — pass `--model` to continue it, or use a new `--run-name`. `--dry-run` shows
both (`on restart :` line; `REFUSED (run dir)`). Detail: `src/main/launcher/CLAUDE.md`.

🚨 **ERA label (owner, 2026-10-03): `utils/era.py` is the one table** (Rustboro `rb`, Dewford `dw`, Slateport `sp`, Mauville `mv`, Verdanturf `vt`, Fallarbor `fb`, Lavaridge `lv`, Fortree `ft`, Lilycove `lc`, Mossdeep `md`, Sootopolis `st`, Pacifidlog `pd`, Ever Grande `eg` — Hoenn's journey order), `CURRENT_ERA` = Rustboro. Name a run `<code>_<what>` (`rb_x26_s1001`). The DEFAULT names are minted prefixed (`rb_run_<ts>`, `rb_exploiter_vs_<target>`); an explicit `--run-name` is accepted AS TYPED (a silent prefix would make the directory differ from the name quoted in scripts, `--model models/<name>` refs and the ledger; a refusal would break every convention) and the trainer prints `[Era] …` when the name lacks the current era's prefix or carries another's. The RECORD is `metadata.json`'s `era` block (`{code, name, order}`), written ONCE at the run's creation save beside `original_command` and carried verbatim by every later save — a resume or restart never changes it, and a run with no block is **pre-era** (a resume never stamps one onto it). `main.lineage` prints `era=`, `main.elo`'s headline tags `[era: …]`, and the critic gate's ladder section warns `CROSS-ERA COMPARISON` when the two runs' eras differ. The Rustboro era starts with the X5 A/B and X26; earlier runs are pre-era history.

🚨 **Runs ALWAYS land in the MAIN checkout's `models/`, even from a worktree** (2026-10-02; no flag). `--run-name`, the minted `rb_run_<ts>`, a resume and `--dry-run` resolve through `utils.paths.run_archive_dir()` (`$GEN3AI_MODELS_DIR`, else main's `models/`) and the child is handed an absolute `--run-dir`; no archive ⇒ `FATAL_CONFIG` naming `$GEN3AI_MODELS_DIR`. A typed `--run-dir` (or resumed checkpoint) inside a worktree's OWN `models/` is refused `FATAL_CONFIG` — that directory dies silently with the worktree (eight runs, 2026-09-23). `--model models/<run>/…` typed in a worktree resolves to the archive's run.

The UI is **Textual** (built on the shared `src/main/tui/` base), launched with
`python -m main.launcher …` (or the back-compat alias `python -m main.launcher.tui …`). A closed
terminal (SIGHUP) or external `kill` (SIGTERM) is caught and turned into a clean,
checkpoint-saving shutdown rather than a lost checkpoint, at the trainer's next SAFE POINT (after an
in-flight update; a 120 s fallback exits 15 without a save, the launcher SIGKILLs at 150 s; `src/main/launcher/CLAUDE.md`).

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
env-core processes and every eval worker, across periodic and crash restarts alike. **On an idle
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
3. **Is this the RECIPE you meant?** — the **RECIPE SURFACE** diff (`main.train.recipe_surface.report`,
   the same three readers) against the mirror's `recipe` block (`recipe.fresh` = N0's measured
   recipe; `recipe.fork` = E5, the comparison for a fork). On a FRESH
   argv a differing knob the argv did NOT type **REFUSES** unless `--allow-nonproduction-recipe`; a
   TYPED difference is INFO (it is the arm's lever); a FORK is INFO — its recipe is its argv.
   Values and sources: `designs/endstate/design_learner_recipe.md` §3.22.

🚨 **The two failures are not interchangeable.** A refused flag combination is LOUD and pre-launch —
nothing starts, it costs a minute. **Arch drift is SILENT and post-launch**: everything parses, the
run starts, and the config diff is the only thing that would have told you. On 2026-09-06 an arm
trained a near-bare network for 24.4M steps after three gates passed, all three correctly. So
`checkargs` closes an arch-only failure with its own verdict — *"✗ this command LAUNCHES — and
builds the wrong architecture"*.

**`--arch production`** is the remedy: it applies every ARCH-surface key from
`designs/production_config.json` as if typed, and records `arch_source` in `model_config.json`; its
RECIPE half applies the training recipe (incl. the supervision doses; the win-prob critic, its win-indicator terminal and gamma are constants of the namespace, not recipe rows) and records `recipe_source` in `metadata.json`'s `cli_args`. It deliberately does
NOT set the critic READOUTS the win-prob critic (the only critic) implies, or `--belief-grad-mode` — the block lists
them every time, so its silence is never read as coverage. A same-run restart (which strips `--arch`)
resolves each recipe knob by one route (INERT `--lr` / `--batch-size` / `--n-steps`
untouched; recorded fields from `model_config.json`; the rest from the run's `metadata.json:cli_args`),
announced, and REFUSES by name when a value is missing — never a default. It also KEEPS the run's
provenance tags — `arch_source` from the checkpoint's `model_config.json`, `recipe_source` from
`cli_args` (`[Arch] same-run restart: arch_source=… kept`); before 2026-09-30 the first restart
recorded `arch_source: null`. A fork into a new run dir keeps neither (its parent is in `lineage`).

**A NAMED ARM** (`src/main/train/arch_arms.py`, `gen3_static_recovery_v1`) is `--arch <arm>`: the production surface
plus the arm's DECLARED overlay, applied as if typed, with production's RECIPE; the guard judges a fresh argv against
production ⊕ the overlay (no consent needed for the arm itself; a drift from the arm is refused by name), and
`arch_source` records `<arm>@production_config@<12>+overlay@<8>`. The one arm today, **`--arch static_recovery`**, is
`static` + `--mon-hazard-cost on --move-actor-state on --trunk-layers 3 --switch-hazard-cost on --eot-residual on`
(`designs/endstate/design_static_tokens.md` §13). A bisection arm is the named arm plus ONE typed lever and
`--allow-nonproduction-arch`.

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
`--arch production` supplies the architecture AND the production training recipe (the mirror's
`recipe` block, K10(a)). A recipe token typed here OVERRIDES the recipe, so type one only when it is
the arm's lever.
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m main.launcher \
  --restart-interval-hours 3 \
  --steps 15000000 \
  --device cuda \
  --log-level periodic \
  --arch production
```

### Resuming from a checkpoint
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m main.launcher \
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
whatever rate the PARENT's KL controller had annealed to. Measured (ledger M7, on the since-deleted distillation fold): three folds launched with the same flags ran at a median **5.8e-5 / 2.8e-5 / 1.0e-4**, and the quantity
that predicts a fold's collateral is the **DOSE** = `lr × n_epochs × optimizer steps per epoch / rollout rows`
(= `lr × n_epochs / (batch_size × grad_accum_steps)` when the rollout divides evenly; a ragged last
accumulation group is a FULL step — K10(c))
— v8's fold was 2.15e-8, every gen-era fold 2–6.6× higher. **`--fork-lr FLOAT`** pins that: on a
genuine fork it sets the optimizer LR *and* `model.lr_schedule` at load and seeds the KL controller
from it (still clamped into `[--min-lr, --max-lr]`). **`--fork-lr-freeze`** additionally disables
the KL adaptation and the two-phase cosine, so the fold runs at one constant, recordable step size.

**The pin applies ONLY on a genuine fork** — a checkpoint from OUTSIDE the target run dir. A
launcher PERIODIC RESTART re-invokes the same argv (in its RESUME form: FRESH-only flags stripped,
`--model` → the latest checkpoint) into the same run dir every N hours, and
re-pinning there would reset the controller's adapted rate forever; the FREEZE is a property of the
run and does persist (re-read from the pin recorded in `metadata.json`). `--fork-lr` on a fresh run
is refused (use `--lr`).

🚨 **FORKING A FROZEN RUN WITHOUT NAMING YOUR OWN `--fork-lr` IS A STARTUP `[ForkLR] FATAL`**
(`gen3_fork_lr_inherit_guard_v1`). A fork inherits the parent's pinned NUMBER through the optimizer
state and NOT its freeze, so the KL controller starts live at a rate chosen precisely because it
should not move, and the run's dose ends up neither the parent's nor one you selected — and not
stationary within the run. That is the era-2 exploiter defect exactly: the plateau parent's frozen
2.80e-05 annealed to 8.36e-05, median 5.5e-05, **0.39× the v8 reference against era-1's 1.78×**, a
4.5× gap nobody registered. The refusal names the parent's value and the argv that reproduces it;
`--allow-inherited-fork-lr` is the deliberate opt-in, and `python -m main.checkargs` prints the
same verdict offline, before the GPU is touched. Every metadata write records a **`dose`** block (`lr_now`, `lr_flag`,
`fork_lr`, `lr_frozen`, `effective_batch`, `updates_per_env_step`, `dose_rate_now`,
`kl_controller`), and `train/dose_rate` + `train/effective_batch` ride TensorBoard every rollout.
Read a run's dose — including every run already on disk — with **`python -m main.dose <run>…`**.

---

## Training

Run directly (no restart loop, no worktree isolation):

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 src/main/train_rl_agent.py \
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
alone** (`train/noise_scale{,_ratio,_share}_{policy,value,entropy,aux}`, default ON): the
total is measured on the SUM of every loss term, and this tree's dense supervised aux heads
have far lower gradient noise than the clipped surrogate — so a total reading "over-batched" can be
aux DEFLATION rather than a batch that is too big, and the advisor says so explicitly when the two
disagree. **`--adaptive-batch {off,total,policy}`** (OFF by default, byte-identical off) closes that
loop: it holds the chosen term's noise ratio near `--adaptive-batch-target` by doubling/halving K
itself — never `--batch-size`, so every forward shape and the activation peak are unchanged and
`--compile-trainer` never notices — with the moved K persisted in the checkpoint sidecar. It is the
SLOW controller of the two; at fixed `target_kl` a bigger K makes the KL lr loop raise lr, so the
dose is a product of both and `train/dose_rate` is what the operator watches. Details:
`src/agents/training/CLAUDE.md` → Gradient accumulation, and → `--adaptive-batch`.

Checkpoints are saved automatically into `models/rb_run_<timestamp>/checkpoints/` (each `.zip`
beside its per-checkpoint `.json` sidecar); the run-level `model_config.json` / `metadata.json`
/ `latest.txt` and the `final_model*.zip` / `best_model/` stay at the run root.

#### Cadences and N — which interval is TOTAL env steps, which is per UPDATE (F-SZ-3, 2026-10-01)

**The periodic checkpoint is every 2,400,000 TOTAL env steps at every `--n-envs`**
(`main.train.constants.DEFAULT_CHECKPOINT_EVERY_ENV_STEPS`; `--checkpoint-every-steps` overrides, in
the same unit). It used to be a hardcoded 50,000 VEC-ENV CALLS — 2.4M at N = 48 but ~102M at
N = 2048, a run that never checkpoints — and under the (since-deleted) async collector (one callback call per WAVE of
ready envs, < N) it fired early by the mean wave fraction. **The save now lands at the first callback
call whose `num_timesteps` reaches the next multiple of the interval** (`run_io._TrackingCheckpoint
Callback`, `constants.checkpoint_due` — the rule the eval callbacks use), so it is the same under
the Rust collector (ragged host steps; the sync and async-wave collectors were deleted in U3). Each save is at most one call's advance
past its boundary. **A fresh sync run is unchanged** wherever the interval is a multiple of N (N = 48:
the same k × 2.4M steps). **A RESTART changed**: the boundaries are GLOBAL multiples of the interval,
so a run resumed at step S saves next at the first multiple above S, not at S + interval (SB3's call
counter restarted every process), and never re-saves the step it resumed on. Pinned by
`src/main/train/cadence_n_independence_test.py` (N = 48 vs N = 2048 through `build_callbacks`; the real
checkpointer through the real Rust-collector loop) and `src/main/train_rl_agent_test.py` (restart streams).

| cadence | unit | N-independent? |
|---|---|---|
| periodic checkpoint (`--checkpoint-every-steps`) | `num_timesteps` boundary | yes |
| eval cycle (`--eval-freq`, `EVAL_FREQ_STEPS` 2M) — and the snapshot-pool add and opponent-pool refresh it drives | `num_timesteps` | yes |
| plasticity canary (`--canary-reset-steps`) | `num_timesteps` | yes |
| `--diagnostics-every`, compile canary (every 25 updates), `--adaptive-batch-every`, KL lr controller, team win-rate pulls (3 rollouts), `--capacity-*-every`, CUDA memory-trend horizon (25 updates) | UPDATES / rollouts | only while the rollout `n_steps × n_envs` is fixed (production 2048 × 48 = 98,304) — a sizing change of N must restate `n_steps` or `--rollout-target-samples`, or every one of these moves |
| launcher restart, graceful restart, `--log-level periodic` lines | wall clock / episodes | n/a |

### WHICH readout is the critic — the win-prob critic (the only critic; no `--critic` flag)

`policy._critic_value` has a MODE. **`winprob` is the only critic** (a constant of every trainer namespace, `src/main/train/parser/objective.py`; `--critic` was DELETED in P11b batch (b) and a typed one is refused with its reason, `designs/deleted_flags.md`), with its win-indicator terminal (indicator, victory 1.0, draw 0.0) and gamma 1.0 as constants of the namespace too (`--terminal-indicator`, `--victory-value`, `--draw-penalty`, `--gamma` are DELETED, P11b batch (c); a typed one is refused with its reason). **`shaped`** —
every generation through gen-16 — was the scalar `value_net` in raw shaped-return units (the scalar head and the whole SB3 value tower are DELETED since the X5 version break's part 2, audit F1), with the win-prob head an auxiliary BCE (PopArt and the distributional `E[Z]` critic were DELETED, L1, config v131); `shaped` is no longer trainable (`CRITIC_TRAINABLE_MODES` is deleted). An ABSENT record (a pre-v109 config / saved
`policy_kwargs`) still means `shaped` in a RECORD (`critic_mode.CRITIC_UNRECORDED`), but a shaped checkpoint no longer loads at HEAD: every one is below `MIGRATION_FLOOR` 144, and the policy constructor (whose `critic` default is `winprob`) refuses any other value before it builds anything; a resume or fork of one is refused `FATAL_CONFIG` (D4 `PythonEraShapedCheckpoint`). Run it pinned to its own commit.

**`winprob` promotes the head to BE the critic**: `V(s) = sigmoid(win_head logit)` in [0,1], the
value loss IS that head's BCE against the terminal outcome (weighted by `--vf-coef` — one critic, one coefficient; the old `--win-prob-coef` was deleted), and the reward stream is the TERMINAL **WIN
INDICATOR** alone. With victory 1.0 and gamma 1.0 (constants) the undiscounted return from any state
is exactly `1{win}`, so **`V(s) = P(win | s)` with no approximation term** — which is why
the indicator terminal and victory 1.0 are fixed by construction, not options. (A bounded Bernoulli payoff has no scale to track — PopArt is deleted.)

🚨 **A critic bounded in [0,1] cannot represent "a timeout is worse than a loss."** That ordering is
not merely unused under `winprob` — it is *unrepresentable*, so the draw value is fixed at 0.0 (no flag). The
anti-stall pressure comes from the obs deadline clock (the reward has no anti-stall term). **Stall rate
and mean episode length are PRIMARY endpoints on any `winprob` arm, not monitored ones.**

ONE flag is IMPLIED (`--win-prob-mode shaping`) because its
argparse default is a `None` sentinel; the discount and the terminal are constants of the namespace (a resume whose recorded reward is not the production one is REFUSED by `check_reward_config`: run it pinned, or start fresh). Everything the
mode SUBSUMES is refused rather than ignored. `python -m main.checkargs` reports every one offline.

**`--win-prob-pbrs-frozen`, `--win-prob-pbrs-coef` and `--win-prob-pbrs-source` were DELETED** (L1, config v131; the actor-only frozen potential was the one PBRS route buildable under `winprob`). **The recorded `critic` is STRUCTURAL and resume-immutable.**

**`--win-prob-strata-weight` was DELETED** (deletion pass P11c, config v135): the opponent-CLASS inverse-frequency re-pricing of the win-prob BCE's mix read NOT DETECTED on ladder arm 7 and did not replicate; no workflow, backlog row or era step used it. A checkpoint that recorded a non-zero value is refused on a resume / fork (run it pinned to `0da1be4a`).

**The critic-ladder target levers are DELETED (deletion pass L2).** `--win-prob-lambda` / `--win-prob-lambda-truncated` (a λ-return BCE target), `--win-prob-rollout-target` / `-r` / `-mode` / `-weight` (R-rollout Monte-Carlo targets and their anchor weight) and `--win-prob-dense-aux` (25 dense end-of-battle targets) no longer exist: the win-prob BCE's target is always the episode's terminal outcome, and a checkpoint recorded with `dense_aux` ON is refused on every load (a training-only lever refuses a resume/fork naming the pin 475bd817). Why and where recoverable: [`designs/training/critic_and_value_losses.md`](../training/critic_and_value_losses.md).

**`--fork-fraction <0..1>`** (default `0.0` = OFF and bit-identical — no module, no obs key, no
callback, no row; runs on the win-prob critic, the only critic) is **THE FORK ARM**: the
fraction of the buffer's decisions that are FORKED. At a contested decision the episode is replayed
out of the Rust core's finished input log to that turn and `--fork-branches` continuations — the policy's top-2
plus ONE uniformly random legal action — are played to a terminal by the current policy; their
transitions enter the SAME buffer, the fork step masked out of the policy term and the shared prefix
counted once. It is the ONLY knob on this loss that adds STATES rather than re-pricing, re-weighting
or re-aiming the ones collection happened to visit — registered because the promoted head ranks two
successors one move apart at **0.5169**, a coin, while a frozen-trunk refit on exactly this data
reaches **0.6032**.

(The Python arm's ring-cap hazard — `cf-records-keep`, `fork/records_missing` — is history; the Rust arm needs no ring, `designs/training/forks.md` §14.) 🚨 **The cost is real, and the ROW BUDGET is what actually caps it:**
`fork/sim_steps_share` is `forks × branches × remaining decisions` over the trainee's own decisions,
paid as a STALL between collection and `train()` (`fork/seconds`). Fraction 0.02 ASKS for ~2.1×, but
the injection is capped at one buffer's worth of rows (~790 forks at the production shape), so above
~0.008 the fraction is **INERT** — read **`fork/requested` against `fork/forks`** before concluding
anything from a flat `fork/rate`. Read **`fork/rate`** (forks per battle), **`fork/branch_share`** (how
much of the objective is now branch rows played against a SELF-LIKE opponent — the arm's largest
caveat), **`fork/tie_rate`**, **`fork/random_wins`** (the blind-spot rate) and **`fork/pairwise_acc`**
with `fork/pairwise_pairs` beside it (IN-SAMPLE; the registered endpoint is a HELD-OUT read on fresh
forks whose bar is a CI clearing 0.60). `--fork-crn` defaults to `dice_and_draws` and should stay
there; `dice` is the control (the regime of the since-deleted `cf_q_labels` factory). Full chapter:
[`designs/training/forks.md`](../training/forks.md).

Design of record:
[`designs/ai_v12/design_winprob_only_critic.md`](designs/ai_v12/design_winprob_only_critic.md);
flag mechanics in `src/agents/model/CLAUDE.md` → *The CRITIC MODE*.

**`gamma` is not a flag** (`--gamma` was DELETED, P11b batch (c)): the discount is `WINPROB_GAMMA` = 1.0, a constant of the namespace, and SB3 restores a checkpoint's own gamma on a resume like `--lr`.
### In-process bridge transport (the Rust bridge — the ONLY training transport)

There is no transport flag: the trainer's one transport is `rust` (U3 deleted the Python RL transport that took `node` / `off`; P11b deleted the one-valued `--use-bridge`, so a typed one is refused at parse time with the reason). The table below records the transports that still exist as `impl=` values for the harnesses, `play.py` and the ladder. The bridge is an
in-process `BattleStream` subprocess for **training and eval** — no server, no port, no `/challenge` connection storm,
deterministic delivery (poke-env issue #907). **THE DEFAULT IS `rust`, so a run needs no Showdown
server at all.** 

| value | transport | when |
|---|---|---|
| **`rust`** | the std-only pokesim `src/rust_sim/src/bin/sim_bridge.rs` | **the DEFAULT** — fastest, smallest child, serverless |
| `node` (an `impl=` value, not a trainer flag value) | the Node `local_sim_bridge.js` | the explicit A/B arm; the parity harness and `gen_sim_bridge_diff.js` need it |
| websocket (not a trainer transport) | a Showdown server | `play.py` / the ladder only |

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
  corrupted PPO transition.
- Eval plays **one game at a time per worker** (the worker-concurrency flag `--eval-concurrency-per-worker` was deleted by the
  flag census, P11; raising it was asyncio latency-hiding, not multi-core, and netted negative under training contention).
- The launcher injects no `--showdown-port` (the flag was deleted in P11; training and eval run in-process).
- **The trainee's observation comes from the Rust core**, the only source (the obs-source flag and its `python` value were deleted in U3; the `__OBS__` frame is how the rust child ships each decision's row). Detail: `src/agents/training/CLAUDE.md`.

🚨 **THE DURABLE LESSON — a "default" branch that nothing tests is untested however green the suite
is.** Three seed defects shipped on this path because every gate was inherently SEEDED or compared
only aggregates, so the production SEEDLESS branch was never exercised. Two more killed production
launches because the fuzz drives only masked-LEGAL tokens. A related trap: **an allowlist entry can
outlive its own fix and then mislead every reader after it**, including a subagent briefed from it.

Detail — the (all FIXED) seed and CHOOSE-path defects, the parity-gate inventory, the coverage
census and the throughput tables — is in `src/utils/bridge/README.md`, `src/rust_sim/CLAUDE.md`,
`designs/ai_v5/design_local_sim_bridge_transport.md`, and
`designs/research_state/claude_md_archive/bridge_transport_history.md`.
### The compile flag — `--compile-trainer` (GPU learner)

`torch.compile` is applied to the LEARNER (CUDA), through the one flag `--compile-trainer`: **AUTO — ON when the resolved device is cuda**, OFF on cpu, OFF under `--debug`; opt-out `--no-compile-trainer`; on failure **always FATAL**. It is a runtime perf knob: never versioned, never in `check_compatible`, and **not INHERITED on resume** (a flagless resume gets it ON, so the opt-out is what you re-pass). It cannot be a flat `True`: it **refuses** a non-cuda device by design, so its default is conditioned on the resolved device, and `--debug` is excluded even with `--device cuda`. **An EXPLICIT `--compile-trainer` on cpu still refuses, loudly.**

**The CPU opponent compile is gone (deletion pass U3).** The compile-opponents flag, its `-preload` / `-strict` variants and generated `no-` forms, the forkserver preload (`compile_preload.py`), the cache prewarm (`compile_prewarm.py`) and the cross-process revert quorum were deleted: every policy opponent forwards through the T2 inference service, so the trainer compiles none. What remains is `agents.model.compile_opponents.maybe_compile_extractor(model, enabled, label, hide_cuda)` for the OFFLINE readers (prober counterfactual view, snapshot ladder, eval worker, cf producer, search-dividend probe). The recorded design and measurements of the retired path (B=1 CPU forward 6.371 → 0.976 ms, +33.3% marginal FPS at `--n-envs 48`, the synthetic-obs lesson of `ai_v12_14_ladder_truevalue`) are history in `designs/training/compile_flags.md`. The synthetic-obs registry (`agents.model.extra_obs_keys`) still builds every training-path warm-up (`--compile-trainer`'s, `--warmstart-battles`' behaviour-cloning forwards); a new obs-key flag needs one row in that table.

### Compiled GPU trainer (`--compile-trainer`, **default ON for cuda**)

The other half, and the bigger one. **Measured on v76 at the production shape** (batch 4096, PopArt
on (since deleted), the real `MaskablePPO` path, arms interleaved on an idle box): `policy.evaluate_actions`
forward+backward **155.1 → 88.5 ms = 1.75×**, i.e. **~+62% end-to-end FPS** at the ~89% train share.
Compiling the whole policy instead of just the extractor measured the same to within 0.004×, so the
extractor is what ships — same win, less graph.

**CUDA only, and fail-loud by design.** A silent fall back to eager would be a 1.75× regression that
no metric surfaces (the run trains correctly, just ~38% fewer steps/hour, forever), so a failed,
slower, or numerically-divergent compile is a hard `FATAL_CONFIG` exit rather than a warning. The
parity check runs on a committed fixture of REAL obs rows, at the decision level (masked legal
log-probs, V) and on the train graph's gradient, at fp32 `highest` (the only matmul precision; TF32 was
retired, deletion pass K2). 🚨 **Every cuda run with the default `--compile-trainer` from `28eaef29`
(2026-08-17) up to `gen3_inductor_trunk_split_v1` (2026-09-28) trained its LEARNER on a miscompiled
single CUDA graph** (argmax agreement 70.9%, gradient cosine 0.778 vs eager). Its eval, opponents
and traces ran the CPU compile and are clean. The split was torch-2.5.1-only: under torch 2.8 the
unsplit graph passed the gate, and on 2026-10-02 (deletion pass K1) HEAD dropped torch 2.5.1
altogether — the split, the extractor-only compile and every 2.5.1 branch are gone, the learner
compiles only as its declared regions, and the trainer REFUSES torch < 2.8 (`src/utils/torch_floor.py`).
A resume or fork runs on the torch its run recorded — the launcher selects `gen3ai_stable` for a 2.5.1
(or unrecorded) run, which therefore resumes only PINNED to its own commit, and refuses a mismatch
`FATAL_CONFIG` unless `--allow-torch-switch` (`src/main/launcher/CLAUDE.md`). Detail:
`designs/training/compile_flags.md` → "The single-graph CUDA miscompile". And `--device cpu` is
refused up front: the compiled learner is gated on CUDA only.

**The DEFAULT yields; the REFUSAL does not.** `auto`/`cuda` with a card ⇒ on; `cpu`, any other
explicit device, or `--debug` ⇒ off. The auto path *also* runs `check_shape_stability` and stays
OFF (with a startup line saying why) when the config is one this flag refuses — an update (the real size: `--rollout-target-samples`, else `n_steps * n_envs`) that does not divide by `--batch-size` — because a default must never turn a command
that works today into a `FATAL_CONFIG`. An explicit `--compile-trainer` on any of those still exits
`FATAL_CONFIG` with the message it always did. **A default yields to the config you typed and says
so; an explicit flag refuses.**

**With the default ON, every plain cuda run compiles the learner without anyone typing a flag**,
which is why it announces itself at startup when the auto default resolves to on;
`--no-compile-trainer` opts out. Full detail — the four refusals, the
`state_dict` hazard, the measurement table — is in `src/agents/training/CLAUDE.md` → Compiled GPU
trainer.

### Memory headroom — `--grad-checkpointing` (operator knob)

`--grad-checkpointing` is the EXACT memory lever for the learner: it gradient-checkpoints the team transformer's encoder layers and the damage operator (`TeamTransformer`, `DamageOperator`) during the PPO update, trading ONE extra forward in the backward pass — on a GPU that is otherwise idle there — for about **5 GB less activation VRAM**. It is **bit-exact** (dropout 0, non-reentrant checkpointing), so a run on it trains the same numbers as a run without; it is a runtime knob (never in `check_compatible`, never in the saved checkpoint), so it is set fresh from the argv on every launch and **a flagless resume does NOT inherit it** — re-pass it on every resume of an arm that needs it. It is OFF by default and the production recipe does not use it. Reach for it when a launch dies with a CUDA out-of-memory in the update (a larger `--batch-size`, `--n-envs`, or the X5 belief-token architecture, whose tokens grow the transformer's activations), BEFORE shrinking the batch: shrinking the batch changes the dose, this does not. It must run before `--compile-trainer` patches the forward (the launch order handles that); the startup line `[GradCheckpoint] enabled on N transformer block(s)` is the confirmation.

### The Rust env core — THE ONLY ENV CORE (the M5 switch)

The M5 Rust env core (N envs
in one core process; it replaced the Python `SubprocVecEnv` of `Gen3Env` workers, deleted in U3), the trainee and the policy opponents forwarded through the inference service in
one flush, bots in the core, and the COMPLETE-GAME collector (an update fires at
`--rollout-target-samples` completed-game rows; no row dropped for age; `staleness/*` measures the
rest; `--rollout-target-samples` must be a multiple of lcm(`--batch-size`, `--n-envs`) — every
micro-batch is full). Measured at the production mix (N = 48, 95 % self-play): 5.1× the Python
path's trainee decisions/s at 0.04× the CPU per decision, the step 74 % T2 forward. It needs
the win-prob critic (the only critic) and refuses, by name at startup, every flag whose path it does not serve yet
(`src/agents/training/CLAUDE.md` → "The env core"). Validate an argv with `checkargs` first.

**THE SWITCH (`gen3_env_core_switch_v1`, 2026-10-02, ledger *THE M5 SWITCH*; production N\* = 256 since the SIZING verdict, 2026-10-02).** The production core and
every SIZE a run declares at startup live in ONE block, `designs/production_config.json`'s
`recipe.sizing` (N, the n_steps maximum, the collector's update size, T2's slots /
buckets / lanes; `verdict` names the sizing Decision record). There is no `--env-core` flag (deleted, P11b); what a launch does with the checkpoint's RECORDED core is
(`main.train.rust_env_setup.env_core_switch_line` / `refuse_python_era_checkpoint`):

| launch | core |
|---|---|
| fresh (`--arch production` or a bare argv) | **rust** — the only core; the SIZES come from `recipe.sizing` |
| `--model` — a same-run RESTART or a FORK | **rust-era** checkpoint (`metadata.json` / the sidecar's `env_core` stamp says rust): runs on it silently. **Python-era** checkpoint (produced on python, or recorded before the core was stamped): **rust**, printed as `🔀 [ENV CORE] CORE SWITCH` — the data stream changes, the weights and recorded config carry across (deletion pass D4). A checkpoint that trained the **SHAPED critic** (or never recorded one) is **REFUSED**, typed core or not (`FATAL_CONFIG`; `--dry-run` reports it advisory when the launch is pinned) — run it PINNED to its own commit |

A typed `--env-core` is refused at parse time with the reason (the Python core was deleted in U3 and the one-valued flag in P11b). `--dry-run` prints `env core : rust [the only core]`, the core-switch line when one applies, and (with the RECIPE SURFACE block) a `sizing: … verdict` line. **Era boundary:** a run across the switch changes its DATA
stream (the keyed trainee draw, per-game eval seeds, complete-game updates), so throughput and every
core-dependent reading compare only within one core (ledger *THE M5 SWITCH*).

**Under the launcher** (F-LG-6; pre-flight at N = 48 2026-10-01,
`designs/research_state/measurements/m5_switch/`): the trainer builds its own checkout's env core at
startup (`🦀 [ENV CORE BUILD]`, about 7 s cold in the launcher's fresh pin worktree); T2 comes up in
~160 s and the learner's compile gate + prewarm in ~5 min, so startup to the compile LOCK is ~8 min
(N = 256 with the X26 heads, 2026-10-01: T2 ~240 s, LOCK 7.8 min; a same-run resume reuses the compile
cache, T2 ~40 s, LOCK 1.8 min).
The interval and crash restarts re-declare the core, T2 and the eval core. A pin before
the Rust env core's flags existed is refused by name. Detail: `src/main/launcher/CLAUDE.md` → "A Rust-core
run under the launcher"; design, gates and measurements in `designs/training/rust_collector.md`
and `designs/research_state/measurements/m5_laneG/PROGRESS.md`.

### Bot evaluation

Bot eval plays **IN PROCESS on the Rust eval core, BLOCKING** (`rust_eval.launch`, between two host steps
of the collector): the cycle is collected in the same step and costs training its wall time — **~1.5% of wall
at N=256** (`designs/research_state/measurements/m5_sizing/PROGRESS.md` O9, 9.6-16.3 s per cycle at the
production roster). It is NOT a background pipeline: there is no worker pool (`--eval-workers` and the
Python worker branch were deleted in P11 / P10-F2), no skipped cycle and no hung-cycle watchdog; a stop
signal is honoured inside the cycle at its safe points. Results merge into TensorBoard + TUI + best-model
and land in `metadata.json` as a top-level `latest_eval` block. Each opponent's `EVAL_GAMES` are split into
**shard units** (`--eval-shard-games`, default 25 → 4 shards/opponent; per-opponent game count
overridable with `--eval-games`); the same exact aggregation serves the offline Rust eval harness (`rust_eval.offline`, `main.ops.eval_trace_gen`). The mechanism lives in the well-encapsulated **`eval_sharding/` package**
(deep `ShardedEvalPool` interface; aggregation is **exact** — Σwon/Σfinished etc., raw δ pooled then
one CVaR), with a documented **`rating.py` seam** (`MatchRecord` / `RatingModel` / `BradleyTerryRating`)
ready for a future Glicko-2/TrueSkill without touching the live ELO path. **`--self-play` eval shares
this exact blocking cycle** — it additionally plays the pool sentinels' shards, and a winning
cycle promotes its frozen snapshot into the pool by file-copy (`SnapshotPool.add_from_path`). The full
design (exact aggregation, resume re-publish,
sentinels + promotion, `--eval-shard-games`) is in
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
| `--value-sidecar {auto,on,off}` | `auto` — **ON under the win-prob critic (the only critic)**, off otherwise |
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
`🏅 ELO` TUI badge. The fixed bots are the anchor: a one-time bot-vs-bot round-robin → `data/gen3_bot_elo_anchors.json`
makes snapshot ELOs **comparable across runs** (the Python calibration, `bot_elo_calibration`, was deleted in T27 P6
slice 6c; T15's Rust bot-vs-bot calibration is its replacement, `designs/ops/TASK_BACKLOG.md`). Offline: `python -m main.elo <run_dir>` prints a
ladder and plots an Elo-vs-step curve (and can backfill a running run from TensorBoard with
`--source tb`). A rating is a **transitive** model by construction, so the same CLI also prints the
**HodgeRank spine/width split** (`agents/training/hodge.py`) — the cyclic content a scalar ELO is
structurally blind to, tested against the binomial noise floor its own game counts imply, with two
weak per-cycle companions on TensorBoard (`eval/hodge_width_elo`, `eval/hodge_cyclic_fraction`).
Full design: `src/agents/training/CLAUDE.md` → ELO / skill rating.

🚨 **Reporting an ELO has FOUR rules — read them before quoting a number** (the fourth, the fitter's
`recipe` block, and the transport boundary are in "Reporting an ELO — the FOUR rules" at the end of this file). The headline is
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

> 🚨 **`--debug` bypasses `--compile-trainer` and warm-start.** No `--debug` smoke can catch a defect in that layer (2026-09-09: the true-team arm passed the smoke and died in the then-existing forkserver preload). Watch a real launch's first two minutes as the test; the synthetic-obs registry `extra_obs_keys.py` + its AST gate are the standing guard.

---

## Moved from the root `CLAUDE.md` on 2026-10-09 (the lean rewrite)

The root keeps a short card and the rules an agent would otherwise break; this is the detail behind them,
moved verbatim (only the restart-interval note added).

### The smoke test — what it runs, what to look for

Verify the core pipeline before a full run (~2 min, 110 s measured 2026-10-02 beside a bystander job). `--debug` defaults to **CPU**, skips all eval, and is **serverless** (the in-process Rust bridge is the only training transport); to exercise the eval path add `--debug-eval` **AND a short `--eval-freq`** — `--debug-eval` alone fires NO cycle, the default eval interval being 2,000,000 steps. The working form is `--debug --steps 10000 --debug-eval --eval-freq 4000` (measured 2026-10-03: exit 0, 3.6 min, two BLOCKING in-process cycles at steps 4,000 and 8,000 — 900 games each, 61 s and 33 s of wall, `[EVAL] step N: Rust eval core played …` then `aggregate …`); add `--self-play --promote-threshold 0.0` to run `SelfPlayCallback`'s cycle instead (2.2 min, `[SELFPLAY EVAL] step N: Bots …  Pool …`). A bare argv is the production critic on the production core — the win-prob critic (the only critic) + its win-indicator terminal, on the Rust env core (the only core; deletion pass D2, 2026-10-02) — so the smoke runs the Rust env core (it builds this checkout's `src/rust_env` on first use).

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 src/main/train_rl_agent.py --debug --steps 10000
```

🧪 **`--arch production --debug` works too** (2026-10-07): `--debug` runs ONE env, which can never fill the production recipe's 98,304-row update (three agents stalled for hours), so a fresh `--debug` run whose untyped update is above 4,096 rows takes `--rollout-target-samples 2304 --batch-size 384 --n-epochs 1` — each only where you did not type it — and prints `🧪 [DEBUG SHAPE]` (`src/main/train/debug_shape.py`). `--arch production --debug --steps 10000` measured: 5 updates, exit 0, ~6 min on CPU (the production network is the cost).

Look for `🎯 [CRITIC] winprob`, `🦀 [ENV CORE] rust`, `[ModelVersion] Round-trip smoke test PASSED`, a metrics table with a `behaviour/` block per update (5 at 10,000 steps: the Rust core's update fires at ≥ 2,048 completed-game rows, and the last table is `learn()`'s final dump of the last update), `🧊 [LEARNER FREEZE] released — learn() returned; 10 checks passed.`, `Training complete. Model saved to …` and exit 0. (The Rust core prints NO per-episode lines — no `🏁 Episode Finished`, that was the python core's `reward_manager` print; its episode read is the per-update table's `rollout/ep_len_mean`, `rollout/ep_rew_mean` and `rust_env/games_ended`.) A crash before completion is a regression in the env core / collector / learner; `[ModelVersion] FATAL` means the checkpoint's architecture ≠ current code. Run it from the repo root. Its run dir lands in the RUN ARCHIVE like any run's (main's `models/rb_run_<ts>/`, even from a worktree); for a throwaway smoke set `GEN3AI_MODELS_DIR` to an existing scratch directory.

🚨 **`--debug` exercises a STRICTLY SMALLER surface than a real launch.** It is ONE env on CPU with the inference service's EAGER backend (no CUDA graphs), and it BYPASSES `--compile-trainer` and the warm-start layer. A defect that lives there is invisible to this smoke — on 2026-09-09 a privileged-obs-key arm passed it and died two minutes into its real launch (then in the forkserver preload, since deleted — the principle stands for what replaced it). **The first two minutes of a real launch are the only test of that layer**; the synthetic-obs registry (`src/agents/model/extra_obs_keys.py`) and its AST gate are what stand in for a smoke there.

### Launch hazards in full (the root keeps one line each)

- **`--arch production`** applies the production surface as if typed — the ARCH surface AND the TRAINING RECIPE (the mirror's `recipe.fresh`: N0's measured `n_envs`, batch × accumulation, epochs, LR + KL controller, clip, entropy, self-play, the critic, its reward values and the doses; `recipe.fork` = E5, for forks; K10(a)). `checkargs` prints an ARCH SURFACE and a RECIPE SURFACE diff and refuses a fresh argv that differs — a recipe knob only when it was NOT typed (a typed value is the arm's lever; `--allow-nonproduction-recipe` consents). Origin: on 2026-09-06 an arm launched from a design-doc command block with every architecture flag at its OFF default and trained a near-bare network for 24.4M steps; three gates passed and all three were right. Detail: `designs/endstate/design_learner_recipe.md` §3.22.
- **ERAS** are named for Hoenn towns in journey order — the Rustboro era (`rb`) is current — and a run's name carries its two-letter code (`rb_x26_s1001`; `utils/era.py` is the ONE table). The default `run_<ts>` is minted `rb_run_<ts>`; an explicit `--run-name` is accepted AS TYPED with a warning when it lacks the prefix; the immutable `metadata.json` `era` block is the RECORD (a run without it is **pre-era**, never stamped on a resume), and `main.lineage` / the ELO headline / the critic gate's ladder read it and warn across eras (`designs/endstate/era_plan_post_m5.md`).
- **Desktop GPU** (T23, 2026-10-05: `gnome-shell` held 811 MiB of 12 GiB): a CUDA run refuses `FATAL_CONFIG` naming the process, pid, VRAM and the fix — `sudo systemctl stop gdm.service` before the launch, `sudo systemctl start gdm.service` (or reboot) after; NVML unreadable refuses too; `--dry-run` reports the same verdict; `--allow-desktop-gpu` is the recorded dev / short-run opt-out, `--debug` (CPU) is exempt (`src/utils/desktop_gpu.py`; `designs/ops/TRAINING_RUN_SOP.md` §1 0b).
- **Disk** (2026-10-09: the root filesystem hit 100 % under a screen chain, ~4.2 GB per 15M-step run): `FATAL_CONFIG` printing the arithmetic (checkpoints still to write, eval traces, compile cache, logs, margin; derived from the run) from the launcher BEFORE anything exists (so a PINNED child is covered), the trainer, and `--dry-run` (`disk space :` line); in flight, free < 1 x the next checkpoint exits `FATAL_DISK` (8, never restarted). `--allow-low-disk` is the recorded dev / short-run opt-out; `--debug` is exempt from the preflight (`src/utils/disk_guard.py`, `designs/ops/TRAINING_RUN_SOP.md` §1 0c).
- **`--lr`, `--batch-size` and `--n-steps` are INERT on a resume** — SB3 restores the checkpoint's own values (its discount too: `gamma` is a constant of the namespace, not a flag), so a FORK inherits whatever the parent's KL controller had annealed to. **`--fork-lr`** pins it; the quantity that predicts a fold's collateral is the **DOSE** (`lr × n_epochs × optimizer steps per epoch / rollout rows` — `lr × n_epochs / (batch_size × grad_accum_steps)` when the rollout divides evenly; a ragged last group is a FULL step, K10(c)), read with `python -m main.dose <run>`.
- **A FORK starts with an EMPTY self-play pool, and an empty pool does not disable `--self-play` — it falls back to the BOT pool.** A genuine fork auto-seeds its parent's pool and exits `FATAL_CONFIG` if it still has none, and ANY run whose pool is still empty after 3 eval cycles exits `FATAL_SUPPLY` (5) — one of the declared lever supplies, `designs/training/supply_guards.md`.
- **A restart RESUMES; a FRESH argv never lands on a run.** The launcher's interval/crash restart strips the trainer's FRESH-only flags (`--arch`) before adding `--model` (2026-09-26: `ai_v14_01_base` crash-looped out at its first restart), and a FRESH launch into a dir holding a checkpoint or `model_config.json` is REFUSED (`FATAL_CONFIG`) — pass `--model` or a new `--run-name`.

### Defaults worth knowing

The **Rust env core is the only env core** (the Python core was deleted, deletion pass U3, 2026-10-02): the in-process Rust bridge is the only training/eval transport (serverless — no Showdown server), neither is selectable, and a typed `--env-core` / `--use-bridge` is refused at parse time with the reason (`designs/deleted_flags.md`); a `--model` launch of a python-era checkpoint (produced on the Python core, or before the core stamp existed) moves onto `rust`, announced as a CORE SWITCH, and one that trained the SHAPED critic is REFUSED (`FATAL_CONFIG` — run it pinned to its own commit; deletion pass D4); `--compile-trainer` is **ON** (auto-on for cuda; the opponents are served by the inference service, so there is no opponent compile flag); the critic is the win-prob critic, the only critic (a constant of the trainer namespace; `--critic` is DELETED and a typed one is refused with its reason), its win-indicator terminal (indicator, victory 1.0, draw 0.0) and the discount (`gamma` 1.0) are likewise constants of the namespace — `--gamma`, `--victory-value`, `--draw-penalty` and `--terminal-indicator` are DELETED and a typed one is refused with its reason; a resume reads its checkpoint's recorded critic and terminal, and a recorded non-production reward is REFUSED (run it pinned to its own commit, or start a fresh run). Checkpoints land in `models/rb_run_<ts>/checkpoints/`. Everything runs at `--nice 10` by default; a detached launch (`nohup … < /dev/null &`) runs HEADLESS automatically. ⚠️ The launcher's `--restart-interval-hours` CODE default is still `3.0`; the owner's policy since 2026-10-04 is **6** on every launch (backlog T24 sets it from data on the Rust core), so type it.

### The offline meters — the index

**Offline meters** (no training): `main.elo` · `main.untaught_meter` · `main.critic_gate` · `main.exploitability` · `main.scaffolding_gauge` · `main.capacity` · `main.lineage` · `main.dose` · `main.sidecar_audit` · `main.baselines` · `main.tb_curate` · `main.best_response_gap` · `main.policy_drift` · `main.policy_spectrum` · `main.ridealong_read` *(the X26 ride-along heads' reader)* · `main.belief_roles` *(the X5 belief readers, U7: role calibration R1–R4 + the purpose metrics, CPU forwards on the Lane S bank, `infer` = the across-seed t)* · `main.anchors` *(this one PLAYS — see the root's Playing section and `designs/ops/EXTERNAL_ANCHORS_SOP.md`)* · `main.h2h` *(the checkpoint-vs-checkpoint mirrored head-to-head: PLAYS on the Rust eval core, GPU only under `scripts/ops/gpu_lock.sh`; writes eval COUNT-ledger rows, under claims, to the archive's `models/_ledger/` by default or a root YOU name outside `models/`; the player keeps seat p1, so a checkpoint against itself reads 0.5 + a SEAT effect; `play-many` plays many cells on one engine, up to TWO architectures (an A/B's cross); run it from the repo root — `designs/training/eval_and_rating.md`)* · `main.plateau` *(the PLATEAU meter's TIER 1: `tick <run>` PLAYS each due 10M check, the newest node vs the one 50M back, as the registered GSPRT on `main.h2h`'s engine, CPU by default, and writes one decision row per check to the archive's `models/_ledger/`; `status <run>` reads it — Tier 2 is NOT built, so `TIER1_PLATEAU` is a candidate, never a declared plateau; `designs/training/eval_and_rating.md`)* · `main.eval_ledger` *(the COUNT ledger's operator CLI: `audit` / `show` / `verify`; READ-only except the requests stream, `close-stale --apply` and `audit --rebuild-index`, which rewrites only the ledger's CACHE `<root>/.ledger_index/`)*. ⚠️ **Four WRITE under `models/<run>/` by default** (and `main.h2h` appends to the archive-level `models/_ledger/`): `main.elo` (`elo/elo_ratings.json` + `elo/elo_curve.png`; `--out` redirects; `refit --apply` rewrites `snapshot_ladder/ladder.json`), `main.capacity` (`capacity_battery.json`) and `main.scaffolding_gauge` (`scaffolding_gauge.json`), both redirected by `--out`, and `main.lineage --backfill --apply` (the run's `metadata.json` lineage block). None of the others writes there by default; `main.policy_drift` writes to `~/gen3ai_archive/` and refuses `models/`.

`main.policy_spectrum` is the POLICY-SPECTRUM reader (M5 Lane S): any checkpoint `.zip` on a fixed, committed bank of ~20.7k re-encodable turns (`designs/research_state/measurements/m5_laneS/`), forward passes only on CPU — the probability mass on the policy's own 1st / 2nd / 3rd choice, entropy, and per-category mass, per stratum; it refuses `models/` as an output.

`main.belief_roles` is the X5 BELIEF reader (`designs/endstate/design_x5_belief_tokens.md` §4, §7.4): any `.zip` at HEAD's architecture (X5's hypothesis tokens; a blob / pre-break checkpoint is REFUSED with its pinned commit) on the same Lane S bank, CPU forwards through the strict loader — opponent-intent log loss on the common event space (the ADOPTION GATE is Amendment 3's `intent_logloss_conditional`, renormalised over BLOB's named set; a `fixed_mass` read is scored on EVERY blob run of the look via `--reference` (the banked blob `.erow.npz` files), its value the mean; the as-built form, with the MISS rate its own column, never floored, is descriptive), species-presence Brier / log score, OTHER calibration, the Smogon-only role reads R1–R3 — each beside the Smogon prior as a third column; one `per_run` value per metric, and `infer` runs §7.4's two-sample t over seeds against the boundary you pass; it refuses `models/` as an output.

`main.policy_drift` is the refining-vs-new-strategy DESCRIPTOR: it gives per-snapshot KL, margin-bucketed flips and action mix vs the previous, 10M-back and anchor snapshots, and has a detached `watch` mode for a live run (`src/agents/training/CLAUDE.md`).

🚨 **`main.best_response_gap` is the POPULATION loop's meter, and it REFUSES an unmatched comparison.** `gap = (a fresh exploiter's win rate against the generalist G_t it was trained on) − 0.5`, read per ROUND and per team ARCHETYPE from each exploiter run's own recorded vs-target cycles; **the loop is working iff the gap FALLS round over round.** Two exploiters compared at unmatched BUDGET, DOSE or REGIME are a typed refusal naming the cause — the 2026-09-21 era-2/era-1 read was confounded by exactly a 4.5× dose gap (`--fork-lr` unset, the parent's annealed rate inherited) — and `--allow-unmatched` prints it with the confound carried in the header and the JSON. 🚨 **The training-time series it reads is GREEDY-vs-GREEDY** (the eval regime; `eval_sentinel_greedy` does not govern it), while its optional `--play N` defaults to the TRAINING regime, so the two rates are different populations and are never folded together. Detail: `designs/training/exploiter_and_distillation.md`.

**LIVE-run instruments are a different tier — `main.ops.*` and `scripts/ops/`**, and reading a live arm through an offline meter is not the same operation: they read TensorBoard events, the launcher child log and checkpoint mtimes while all of those are still being appended to, and REFUSE when a precondition of the live read is unmet. `designs/ops/TRAINING_RUN_SOP.md` §2 names which layer uses which; `scripts/ops/README.md` lists every one.

### Reporting an ELO — the FOUR rules, the transport boundary, the eval-regime boundary

🚨 **Reporting an ELO has FOUR rules:** the headline is `<run>/snapshot_ladder/ladder.json` (dense, ±10), not `eval/elo` (±29); a rating is only final once the run is (the newest BT node is systematically inflated); a cross-run comparison must be at matched snapshot **COUNT**, not matched step; and **the committed file is quotable only if its `recipe` block matches the current fitter** — every reader refuses otherwise, `python -m main.elo refit <run>` shows what the same nodes read today, and `refit --apply` writes the stamped fit (keeping the old as `ladder.pre_recipe.json`). Neither plays a game. `ladder.json`'s second column `ratings_relative` is Elo above a pinned frozen reference node (the `untaught_meter_opponent_v14` baseline when the ladder holds it, else the first snapshot), fitted from frozen-vs-frozen games alone so it does not drift with the bot anchor — quote it beside the headline, never instead of it. **Every file fitted before `0f230405` is stale**: 68 of 93 move, median max |Δ| 53.4 Elo and 21 of the v9 generation ladder's 153 orderings reverse (`designs/research_state/measurements/ladder_refit_audit_2026-09-22/`). 🚨 **The ladder's TRANSPORT is a regime boundary (2026-10-06):** new edges play on the Rust eval engine (seat-balanced mirrored pairs, Rust rows), every older row on the poke-env `RLPlayer` + Python encoder; each row and `ladder.json` stamps `transport`, and a fit or `main.critic_gate` refuses to mix the two unless told (`designs/training/eval_and_rating.md` "The TRANSPORT boundary").

🚨 **`win_rate_vs_pool` / `eval/elo` carry an OPPONENT-REGIME BOUNDARY at 2026-09-07** and are not comparable across it. Eval pool sentinels are now **GREEDY and draw the trainee's own teams** by default (`--no-eval-sentinel-greedy` opts out; `--promote-threshold` follows, 0.55 / 0.65). The old asymmetry was worth **+8.9 pp** to the trainee, so equal skill reads ~9 pp lower now. **The regime is RECORDED and INHERITED on a flagless resume** — read it (`model_config.json`'s `eval_sentinel_greedy`, or the launch's `⚖️  [EVAL REGIME]` line), never assume it. `ladder.json` and every bot edge are UNAFFECTED. Detail: `designs/training/eval_and_rating.md`.
