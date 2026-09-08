# TRAINING RUN — STANDARD OPERATING PROCEDURE

**This is the standard location for operational procedure (SOP) documents: `designs/ops/`.** This
file is the procedure for launching, watching, killing, relaunching and reading a training run.
It is always-current (same contract as `designs/ARCHITECTURE.md`): state what is true now, no
narrative; the WHY behind a rule is one sentence plus a pointer to the ledger entry that taught it.
Per-era launch commands live in that era's runbook (`designs/ai_v12/launch_runbook.md` today); the
end-of-run batteries live in `designs/research_state/gen*_endofrun_runbook.md`. This document is
era-independent.

Owner rulings are marked **(owner, date)**. Everything else is a standing rule the team adopted.
This file and [`ORCHESTRATOR_SOP.md`](ORCHESTRATOR_SOP.md) are the PROCEDURES OF RECORD; the
project-memory files that used to hold these rules are pointers to them.

---

## 0. Roles, and how sessions find each other

| role | who | does |
|---|---|---|
| **Owner** | the human | the only source of approval; decides kills that are not pre-registered, retention applies, baseline changes |
| **Orchestrator** | one Claude session, renamed per handoff | dispatches agents, LANDS branches, banks ledger entries, keeps `UNDERSTANDING.md` current, relays to peers |
| **Training Run** | one Claude session | RUNS the arm: launch, watchers, crons, kill/relaunch, routine ledger entries for incidents on its run; messages the orchestrator on belief change, blocked, gate failure, batch complete |
| **Agents** | subagents in isolated worktrees | build and measure; never write under `models/`, never launch, never touch the Showdown servers on :8000/:8001 |

**The Training Run session reports to the ORCHESTRATOR, not to the owner.** The orchestrator is its
scrum master (`ORCHESTRATOR_SOP.md` §1): it holds the backlog, unblocks it, and carries owner-facing
traffic. Anything needing the owner's approval is escalated through the orchestrator, never decided
here.

**Standing autonomy grant (owner, 2026-09-03** — *"you can give it more autonomy if needed so it
doesn't wake you unnecessarily"***).** This session may, without asking:

- commit and LAND its own measurement artifacts;
- bank ROUTINE ledger entries — launches, pin and pool checks, calibration passes, housekeeping;
- make small, already-justified instrument fixes.

It messages the orchestrator only for: a belief-changing reading, a blocker or an owner-go item, a
meter-gate failure or a crash, and the completion of a batch. **Tiebreak: message if the ledger's
headline entries would read differently because of it.** Its silence is expected, not a stall.

**Who is the orchestrator right now** is written to one file that every session on this box can
read at any time, because session names change on every handoff and `ListAgents` shows a dozen
similar rows:

```
~/.claude/projects/-home-goodlad-dev-gen3ai/ORCHESTRATOR
```

One line: `<session name> [<ref>] · session <id> · since <ISO timestamp> · <one-line scope>`. The
orchestrator writes it when it takes a handoff and again if its name changes; a session that needs
the orchestrator does `cat` on it, then confirms the name is listed as live by `ListAgents` before
sending. A stale file (name not live) means "no orchestrator; hold routine traffic, push-notify the
owner on anything blocking". Peer sessions cannot grant owner approval — relay, never escalate.

**Alerting reaches the owner through Remote Control only** (owner, 2026-09-06): there is no
session-independent channel, so when every session is down nothing can page them. The layer-1 OS
watcher's status file is therefore the record of the unattended period, and any session taking over
a run reads it before anything else. The orchestrator carries no cap on unasked GPU commitments and
decides the next arm from the data (`ORCHESTRATOR_SOP.md` §6).

**If this session dispatches a subagent**, the dispatch rules are the orchestrator's and are not
duplicated here: Opus only, never a fork, `isolation: "worktree"`, no conditional git-state
instructions in the brief, hazards in the report are findings — `ORCHESTRATOR_SOP.md` §2. The stall
mechanics (`stallMs`, the two watchdogs, resume-then-stop) and the waiting mechanics (the
self-matching `pgrep` trap, the `Monitor` pattern) are `ORCHESTRATOR_SOP.md` §7, and the `pgrep`
trap in particular applies directly to the watchers this session arms in §2 below.

---

## 1. Before launch — "it launches" and "it is the experiment" are INDEPENDENT checks

Ledger `81016942` (2026-09-06): an arm ran ~7 GPU-hours with 31 architecture flags silently at their
OFF defaults while `checkargs`, `resolve_config` and `--dry-run` all passed. Every check below
answers one of the two questions; do both.

**Does it launch?**
1. `python -m main.checkargs --argv "<argv>"` (with `--pin-commit` it judges by that commit's parser).
2. `python -m main.launcher --dry-run <argv>` — resolves role FRESH/FORK/RESTART, run dir, pin,
   inherited-vs-argv config, without creating anything. Safe on a restart; "launch it and kill it
   after the startup lines" is NOT (it is destructive on a same-run restart).
3. A 60-second `--debug --steps 8000` CPU smoke of the actual argv (`feedback: validate by executing`).

**Is it the experiment?**
4. A FRESH argv is the production architecture surface plus the experiment's overrides — built from
   `--arch production` (the launch guard, `gen3_arch_surface_guard_v1`) or the production baseline's
   recorded `original_command` — **never from a design document's abbreviated command block**.
   Expect to strip the flag family the new critic mode SUBSUMES (`checkargs` names them).
5. Diff the RESOLVED `model_config` against `designs/production_config.json` and paste the diff into
   the launch entry. The only differing keys may be the experiment's own.
6. Baselines by NAME: `python -m main.baselines check` must pass; never hand-edit
   `designs/baselines.json` or `designs/production_config.json`. **The launch entry states what this
   arm is measured AGAINST by registry name, and states every MODE flag's value as READ from the
   resolved config** (shaping vs read-only vs detached per head, each coefficient) — "what is the
   baseline?" is answered in writing at launch, never reconstructed later (owner, 2026-09-06).

**Shape and pinning**
7. Every arm of a batch pins its commit explicitly: `--pin-commit <sha>`. Never `--sync-to-main` on
   a batch. A pinned arm is unaffected by later landings on `main`.
8. The micro-batch (`--batch-size`) sets the ACTIVATION PEAK and the peak is a function of the
   architecture surface; `--grad-accum-steps` keeps the effective batch (exact gradient). A fit
   measured on one architecture is not evidence for another (ledger `ce43b4e9`: 4096 fit the stripped
   arm and OOMed at iteration 1 on the full surface; the full-surface shape on the 12 GB card is
   **2048 × K**). The desktop compositor holds ~1.1 GB of that card.
9. On a FORK: `--lr`, `--batch-size`, `--n-steps` are INERT (inherited); pin the rate with
   `--fork-lr`; the self-play pool auto-seeds from the parent. On a RESTART nothing re-pins.
10. Register the read BEFORE launch: endpoints, comparator baseline by name, matched snapshot count,
    the floor (max pairwise difference among replicates), the kill conditions and the successor arm.
    The read is not renegotiated after launch. **A rule that is not in the ledger is not registered:**
    a relayed threshold cites its ledger entry, in one stated unit, or it is written there first
    (ledger 2026-09-06 · *REGISTRATION — the vf_coef restart rule*, where a relayed rule mixed log10
    with ratios and its three clauses overlapped).

---

## 2. At launch — the four layers, set up unasked

**Every long-running job gets all FOUR, at launch, without being asked** — the owner made the cron
half explicit ("set fallback cron (make this standard sop)"):

1. **OS-level watcher** — [`scripts/ops/watch_run.sh`](../../scripts/ops/watch_run.sh) under
   `nohup`: polls progress, ~35 min wedge limit, matches FAILURE words as well as progress,
   checks the arm-INVALIDATING `--sync-to-main` line, writes a status file. Survives a dead
   session.
2. **OS-level chain** for multi-stage work, so stage N+1 starts without a session. Survives too.
3. **`Monitor`** on the status file, filtered to terminal + failure lines only — owned by the
   Training Run session. The orchestrator does NOT watch the run **(owner, 2026-09-02)**; it is
   messaged.
4. **Fallback cron — every 55 MINUTES, never every hour (owner, 2026-09-06)**: the prompt-cache
   time-to-live is 1 h, so a wake inside the window re-uses the cached context and an hourly wake
   re-pays the whole prefix (the root `CLAUDE.md` alone is ~45k tokens). Prefer a drift-free INTERVAL
   (`/loop 55m <prompt>`, or a self-paced wake at 3300 s); if only a cron EXPRESSION is available,
   `3,58 * * * *` keeps every gap ≤ 55 min at the cost of one extra cached wake per hour. Off-minute
   alignment (never :00/:30). Longer gaps are fine when nothing needs checking that often — the rule
   is "never 60 ± a few minutes", the worst point on the cost curve. Scheduling wakes ONLY to keep
   the cache warm is still waste.

**Layers 1–2 are OS processes and keep the machine working if the session dies; layers 3–4 are
session-scoped and are what reach the owner.** Say that asymmetry out loud on every handoff: the
training survives a dead session, the alerting does not.

**The cron is the only layer that can REPAIR rather than notify, so its prompt must carry:**

- what the run is and its argv file path;
- the exact repair command per failure mode (a cron that reports is a slower watcher; a cron that
  relaunches is the actual defense);
- the arm-INVALIDATING conditions — `Pinned to` must read the registered sha, and a `--sync-to-main`
  line voids the arm: stop it, do not let it run. A cron that keeps a scientifically void run alive
  is worse than no cron;
- the notification policy below.

Retire the cron when the work lands (it auto-expires in seven days, but a wake on finished work is
noise). Never use a self-matching `pgrep` pattern in any of these layers (`[w]atch.sh`, not
`watch.sh`) — it has killed this session's own shell repeatedly, and inside a `Monitor` it fails
silently instead. Full trap list: `ORCHESTRATOR_SOP.md` §7.

**Notification policy (owner, 2026-08-23, extended 2026-09-06):** push on COMPLETION (all registered
parts done AND artifacts on disk, not "training stopped"), when BLOCKED on something unrepairable or
on a decision only the owner can make, and on a MAJOR FINDING (0–3 a day, by judgement). Never
routine progress — leg transitions, evals firing, healthy watcher ticks. One line, <200 chars, the
actionable part first. Bake this policy into every cron and watcher prompt, so it holds on fires that
land while the session is idle or mid-turn.

**Every status update carries MARGINAL fps, measured from CHECKPOINT MTIMES** (owner, 2026-08-23 —
the mtime method is the standard meter). SB3's `time/fps` is CUMULATIVE (num_timesteps ÷
elapsed-this-child) so it lags every regime change: a fresh run's bots-only warmup held it at ~950
while the true rate had already fallen to ~525 as self-play ramped. **The obvious fix is a trap** —
recovering elapsed as `T = step / fps` and differencing is exact in algebra, but `fps` is logged as
an INTEGER, so at 7M steps a 1-unit change moves T by ~30 s against the ~200 s between rows; the
derived marginals explode (6,008 fps was observed, physically impossible) and the error grows with
the run, so the numbers look plausible early and rot silently. With `--checkpoint-every-steps N` the
files land every exactly N env-steps, so the wall-clock gap between consecutive checkpoints is a
rounding-free rate. Quote a RANGE while the opponent mix is still ramping. Never reintroduce the
integer-fps derivative.

The launcher runs everything at `--nice 10`; a detached launch (`nohup … < /dev/null &`) is headless
automatically. Never run a Claude session inside the training tmux session (its cgroup).

**THE INSTRUMENTS ARE IN THE REPO, not in a session's scratch directory** (2026-09-07). They were
built here and lived in a session-scoped temporary directory with one arm's name compiled in;
they now take the run as an argument, resolve `models/` the way `utils.paths.main_models_dir()`
does, and REFUSE rather than default. `scripts/ops/` holds the shell layer, `src/main/ops/` the
Python instruments, and [`scripts/ops/README.md`](../../scripts/ops/README.md) lists every one
with what it measures.

| this layer / read | the command |
|---|---|
| layer 1, the OS watcher | `scripts/ops/watch_run.sh <run> --pid-file … --launcher-log …` |
| MARGINAL fps, from checkpoint mtimes | `scripts/ops/marginal_fps.sh <run> [--target N:label]` |
| the deciding famine read | `scripts/ops/famine_read.sh <run> --parent … --famine-comparator … --control "…"` |
| the registered first-restart read | `scripts/ops/restart_read.sh <run> [--baseline …]` |
| the registered scalars, at the SOURCE | `python -m main.ops.tb_read <run> --last 20` |
| the kill bars · the vf_coef framings | `python -m main.ops.killbar` · `python -m main.ops.vf_framings` |
| G7, QUOTED never inferred · the plateau signal | `python -m main.ops.g7_report` · `python -m main.ops.plateau_signal` |
| stall vs sawtooth · per-bot calibration | `python -m main.ops.stall_exhibit` · `main.ops.perbot_r` / `perbot_rank` / `negskill_null` |

A session may still keep a scratch copy while an arm is live; the repo copies are the durable
ones and are what this document names.

---

## 3. During the run

**🤫 THE CADENCE IS SILENCE (owner, 2026-09-07).** *"We don't need so many updates from the training agent.
The training agent can merely wake up on the cron, hit the cache, and no-op. … we definitely don't need
extensive checks just to have back-and-forth conversations."* A cron wake is a cache-hitting NO-OP unless
something CHANGED: a registered verdict flips (a G7 half, the plateau signal, the draw-rate monitor), a
restart read is not KEEP, the run is blocked or crashed, or the batch completes. Otherwise run the
registered reads, bank the routine row in the ledger, send nothing. No new instrument unless a
registered read needs one. Measured 2026-09-06/07: ~55 messages in 18 h, most of them routine reads and
re-derivations that cost a conversation each where a no-op costs nothing.

**Amendment (owner, 2026-09-07): a quick one-liner every once in a while is welcome.** *"I am ok with
quick one liners of progress, eta, highlight or so every once in a while."* Silence is the default
BETWEEN events; an occasional single line — progress, an ETA, one highlight — is wanted. One line, no
reply expected, no back-and-forth, no scheduled digest, and never a reason to invent a check so there
is something to report.

- **Pre-registered kill conditions are executed, not debated.** Under `--critic winprob` stall rate
  and mean episode length are standing KILL conditions (a [0,1] critic cannot rank a timeout below a
  loss). The famine pre-test compares against the named comparator at matched SNAPSHOT COUNT with
  the registry's floor.
- **Never quote a mid-run ELO or delta.** The newest Bradley-Terry node is systematically inflated;
  read `<run>/snapshot_ladder/ladder.json` at run END, and compare runs at matched snapshot count,
  never matched step (`feedback_elo_reading_rules`).
- **First-restart checks:** the decision that is registered for it (today: `vf_coef` from the median
  of the last 20 rollouts' `grad/value_policy_logratio`); `python -m main.sidecar_audit <run>` shows
  the pin is unchanged across the restart. One command: `scripts/ops/restart_read.sh <run>`.
- **Read a scalar from the EVENTS, never the child log's table.** The table is a RENDERING — it
  drops the group prefix (`grad/value_policy_logratio` prints bare) and `--log-level periodic`
  UNDERSAMPLES it, and `launcher_child.log` is a ~1 MiB ring buffer that trims silently, so a
  "last 20" taken from it is the last 20 rows that still FIT. `python -m main.ops.tb_read <run>`
  reads the events, and WARNS on a missing tag instead of defaulting to 0. It has already caught
  two log-rendering misreads: an episode length reported as "falling" that was oscillating, and a
  metric read under the wrong group.
- **Landings during a live cell are announced to the Training Run session before they go to `main`**
  and name what changed in any file the arm's pinned tree also contains.
- **A single-sample reading is never a verdict.** Every registered instrument is read as the median
  (or the series) over the registered window — `grad/value_policy_logratio` swung a full decade
  between consecutive rollouts while its 20-rollout median sat at +0.15; three single-sample readings
  in one day would each have produced the wrong call (ledger 2026-09-06 · *REGISTRATION — the vf_coef
  restart rule*).
- **Composition guard:** `reward/untracked_abs_mean` must read 0; the launch banner must match the
  registered composition (e.g. "1 TERMINAL + 0 PBRS + 0 BIAS").

---

## 4. Killing and relaunching

1. Kill by EXPLICIT pid (`pgrep -af <pattern>`, read the matches, `kill <pid>`). Never `pkill -f`
   (it matches the killer's own shell; it has killed a launch).
2. **An arm is not killed until the pid is gone and the GPU is free** — verify with `ps -p` and
   `nvidia-smi`. A session restart does not kill a launcher child; "killed" was once said for an arm
   that then ran 7 more hours (ledger `81016942`).
3. Never delete the dead run dir. Move it aside with a suffix naming the cause
   (`models/<run>.OOM_4096`) so the incident entry has its evidence; retention policy handles it later.
4. Relaunch under the SAME run name and pin when the fix is a resource shape or an operational
   defect; a SCIENTIFIC change (a flag the read depends on) is a new arm with a new registration.
5. Bank the incident in `designs/research_state/ledger.md` in the same hour: what died, the
   traceback's call site, the evidence that was already on disk, the durable lesson. The Training
   Run session banks routine incidents on its own run; belief-changing ones go to the orchestrator.

---

## 5. Reading and banking

- One command per registered read: `python -m main.critic_gate <run> --parent <name> --control <refs…>`
  for a critic arm; `python -m main.untaught_meter … --control <continuation arms>` for the untaught
  meter (a frozen-parent delta overstates a fold); `python -m main.exploitability` for the
  best-response curve. Every consumer prints which baseline it resolved.
- Every delta carries its confidence interval and one word of the evidence vocabulary:
  SIGNIFICANT / WITHIN FLOOR / NOT DETECTED / EQUIVALENCE SUPPORTED / REFUTED / UNVERIFIED. A floor
  is the max pairwise difference among replicates, never the smaller bar. Cluster over the real unit
  (teams, battles), never pooled rows.
- The ledger is append-only and WINS any disagreement; `designs/research_state/UNDERSTANDING.md` is
  fixed to match it in the same pass whenever a belief changes.
- **A read is validated at the SOURCE before it is quoted** (owner, 2026-09-06 — GIGO vigilance): the
  resolved checkpoint file and rung, the team manifest actually loaded, the opponent actually played, the
  metric's definition in code (units, sign, denominator). An eval once ran against the wrong team set and
  was believed for two rounds; a result that looks impossible is verified at the artifact, not explained.
- **Every code and cell letter carries a human description in the same sentence when written for the
  owner, on EVERY use** (owner, 2026-09-03, repeated 2026-09-06): "C1, the fold with the distillation
  loss switched off"; "R4DOSE3, the fold at double v8's step size". Not just the first use — the owner
  reads on mobile and out of order, and a bare code "means very little" to them. Ledger entries may
  keep the bare codes (they carry the pin tables) but open with the description. Full rule and the
  self-check: `ORCHESTRATOR_SOP.md` §5.
- Reports to the orchestrator lead with the verdict, the evidence tag and one number with its
  interval; the orchestrator carries it to the owner at the design-doc level (`ORCHESTRATOR_SOP.md`
  §5). Detail belongs in the ledger entry the report names.

---

## 6. Never

- Write under `models/` from an agent; start training or the launcher from an agent.
- Touch the Showdown servers on :8000 / :8001; a throwaway server takes a unique 9XXX port.
- `git add` / `commit` / `push` from the main checkout; every edit lands from a worktree branch.
- Import a `__main__` module to inspect it (one did start a training run).
- Quote a measurement taken on a busy box as a measurement; a timeout is never a semantic outcome.
