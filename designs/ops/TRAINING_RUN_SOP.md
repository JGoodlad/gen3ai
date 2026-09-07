# TRAINING RUN — STANDARD OPERATING PROCEDURE

**This is the standard location for operational procedure (SOP) documents: `designs/ops/`.** This
file is the procedure for launching, watching, killing, relaunching and reading a training run.
It is always-current (same contract as `designs/ARCHITECTURE.md`): state what is true now, no
narrative; the WHY behind a rule is one sentence plus a pointer to the ledger entry that taught it.
Per-era launch commands live in that era's runbook (`designs/ai_v12/launch_runbook.md` today); the
end-of-run batteries live in `designs/research_state/gen*_endofrun_runbook.md`. This document is
era-independent.

Owner rulings are marked **(owner, date)**. Everything else is a standing rule the team adopted.

---

## 0. Roles, and how sessions find each other

| role | who | does |
|---|---|---|
| **Owner** | the human | the only source of approval; decides kills that are not pre-registered, retention applies, baseline changes |
| **Orchestrator** | one Claude session, renamed per handoff | dispatches agents, LANDS branches, banks ledger entries, keeps `UNDERSTANDING.md` current, relays to peers |
| **Training Run** | one Claude session | RUNS the arm: launch, watchers, crons, kill/relaunch, routine ledger entries for incidents on its run; messages the orchestrator on belief change, blocked, gate failure, batch complete |
| **Agents** | subagents in isolated worktrees | build and measure; never write under `models/`, never launch, never touch the Showdown servers on :8000/:8001 |

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
   `designs/baselines.json` or `designs/production_config.json`.

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

Every long run gets all four (`feedback_long_run_sop`):

1. **OS-level watcher** (`nohup`): polls progress, ~35 min wedge limit, matches FAILURE words as
   well as progress, writes a status file. Survives a dead session.
2. **OS-level chain** for multi-stage work, so stage N+1 starts without a session.
3. **`Monitor`** on the status file, filtered to terminal + failure lines only — owned by the
   Training Run session. The orchestrator does NOT watch the run **(owner, 2026-09-02)**; it is
   messaged.
4. **Fallback cron — every 55 MINUTES, never every hour (owner, 2026-09-06)**: the prompt-cache
   time-to-live is 1 h, so a wake inside the window re-uses the cached context and an hourly wake
   re-pays the whole prefix. Prefer a drift-free INTERVAL (`/loop 55m <prompt>`, or a self-paced
   wake at 3300 s); if only a cron EXPRESSION is available, `3,58 * * * *` keeps every gap ≤ 55 min
   at the cost of one extra cached wake per hour. Off-minute alignment (never :00/:30). Its prompt
   carries the REPAIR
   steps (the argv file, the exact relaunch command per failure mode), the arm-INVALIDATING
   conditions (`Pinned to` must read the registered sha; a `--sync-to-main` line voids the arm — stop
   it), and the notification policy below.

**Notification policy (owner, 2026-08-23):** push on COMPLETION (artifacts on disk, not "training
stopped") and when BLOCKED on a decision only the owner can make. Never routine progress. Bake it
into every cron and watcher prompt.

**Every status update carries MARGINAL fps from checkpoint mtimes** (owner, 2026-08-23).

The launcher runs everything at `--nice 10`; a detached launch (`nohup … < /dev/null &`) is headless
automatically. Never run a Claude session inside the training tmux session (its cgroup).

---

## 3. During the run

- **Pre-registered kill conditions are executed, not debated.** Under `--critic winprob` stall rate
  and mean episode length are standing KILL conditions (a [0,1] critic cannot rank a timeout below a
  loss). The famine pre-test compares against the named comparator at matched SNAPSHOT COUNT with
  the registry's floor.
- **Never quote a mid-run ELO or delta.** The newest Bradley-Terry node is systematically inflated;
  read `<run>/snapshot_ladder/ladder.json` at run END, and compare runs at matched snapshot count,
  never matched step (`feedback_elo_reading_rules`).
- **First-restart checks:** the decision that is registered for it (today: `vf_coef` from the median
  of the last 20 rollouts' `grad/value_policy_logratio`); `python -m main.sidecar_audit <run>` shows
  the pin is unchanged across the restart.
- **Landings during a live cell are announced to the Training Run session before they go to `main`**
  and name what changed in any file the arm's pinned tree also contains.
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
- Every code and cell letter carries a human description in the same sentence when written for the
  owner **(owner, standing)**.

---

## 6. Never

- Write under `models/` from an agent; start training or the launcher from an agent.
- Touch the Showdown servers on :8000 / :8001; a throwaway server takes a unique 9XXX port.
- `git add` / `commit` / `push` from the main checkout; every edit lands from a worktree branch.
- Import a `__main__` module to inspect it (one did start a training run).
- Quote a measurement taken on a busy box as a measurement; a timeout is never a semantic outcome.
