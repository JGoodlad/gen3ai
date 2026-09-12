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
| **a CRITIC LADDER arm's whole read** | `python -m main.ops.critic_read <arm> --control <control-arm> --step <N> --out <dir>` — identity + G1 + the turn-contrast + **CONDITIONING** as ARM − CONTROL with the delta's CI and its label, one invocation, one report; **QUOTA-MATCHED by default**. Its CONDITIONING section carries the **(A)/(B)** block (v4, 2026-09-10): a critic whose `V` decodes its OWN TEAM is either CONDITIONING on it (within-team resolution not lower, between-team spread up) or SUBSTITUTING it for the board (within-team resolution DOWN), and the two within-cell rows are read WITH their cell census — ~7 episodes per team cell is mostly the binning's own noise, so the coarse 5-stratum row wins where the two disagree. Since **v5** (2026-09-10) it also carries the **CALIBRATION SLOPE** — the outcome regressed on `logit(V)`; `b > 1` means `V` is UNDER-dispersed (SHRUNK toward the base rate), which is what a BOOTSTRAPPED target produces and what an out-of-fold decode, being scale-invariant, cannot see. 🚨 Read it with the LEVER ARM printed beside it (`sd(logit V)`) and with the COMMON-SUPPORT row: the slope's SE scales as `1/sd(logit V)`, so the shrunk side is handed the wider interval by the very effect under test. Since **v6** (2026-09-10) it also carries the **LATE WINDOWS** and the **OPPONENT-IDENTITY SPLIT** of the spread. 🚨 The N-curve (`measurements/winprob_refit_ncurve_2026-09-10/` §3) measured that on a matched-team frame the opponent is **UNOBSERVABLE at turn 1** — Gen 3 has no team preview — so **a turn-1 spread ratio of 0 is BAYES-OPTIMAL, not a defect**, and the window where the question has an answer is **turns 4-10**. Read `cond.spread_ratio.t4_10` / `.t11_24` and `cond.opp_class_auc.t4_10` there; the registered `t1_3` rows stay, bit-identical, because three landed reads were written from them. And read every spread ratio beside its `cond.spread_ratio_optimal.<window>` companion — the part of that spread attributable to the head's own `V` revealing WHICH opponent it faces — because 1.0 is the WRONG reference at early turns. 🚨 It is a DECOMPOSITION TERM, **not a ceiling**: the first read had `V` ABOVE it at every window (hp800 turns 11-24: 0.647 / 0.704 against 0.214 / 0.187), because a per-state conditional mean ATTENUATES, and the excess is spread riding on BOARD state that differs by opponent rather than on opponent IDENTITY. Read the two together; never quote it as "the maximum". 🚨 **The late-window SPREAD rows carry a LARGE run-level floor** — the two hp800 control-vs-control pairs give 0.2179 at `t4_10` and 0.1920 at `t11_24`, bigger than any arm delta measured so far, so a `DETECTED (vs ZERO — NO FLOOR)` on them is NOT a detection once a floor json carries them. `cond.opp_class_auc.t4_10` is the LOW-noise row of the family (floor 0.0219, four runs spanning 0.688-0.709). v6 also prints the **SELF-PLAY CROSSING** of both sides and a `CROSSING MISMATCH` line when they differ: the first crossing is a coin flip on the SEED path (`SELF_PLAY_START = 0.55` / `--self-play-start-wr` against `win_rate_vs_bots`, NOT `--promote-threshold`), six of eight 10M arms crossed at 4,128,768 and two at 2,162,688, and the mismatch is a RAMP from ~3%. ⚠️ DESCRIPTIVE — the crossing is POST-TREATMENT and may be a mediator, so it SELECTS NOTHING |
| **a HIGH-POWER offline re-read** | `python -m main.ops.eval_trace_gen <run>@<step> --games N --sentinels K --out DIR` on EACH side, then `main.ops.critic_read … --arm-traces DIR --control-traces DIR` — for when the ladder read's binding constraint is POWER, not effect size |
| **the TRAINING-SIDE calibration** | `python -m main.ops.value_sidecar_read <run> --out <dir>` — mean V vs mean target, the Murphy decomposition and skill, sliced by turn bucket / opponent class / outcome / 1M step bucket, each with an EPISODE-clustered CI |

🚨 **PIN `critic_read`'s CYCLE WITH `--step` WHENEVER THE ARM'S LAUNCHER MAY STILL BE ALIVE.**
`--on-live skip-newest` is the default and it DROPS the newest cycle when any process still names
the run — so a finished 10M arm whose launcher had not yet exited is read at **8M**, and the
2026-09-09 tdaux read did exactly that. The report now prints WHICH cycle was chosen and WHY, for
both runs, in its header and on stdout before anything expensive runs; `--step N` pins both runs
and `--control-step N` pins the control alone. A run with no `step_N` REFUSES, naming the steps it
has.

🚨 **THE CONDITIONING ROWS ARE THE ONES THE CURRENT LADDER IS READ ON** (2026-09-09). Three offline
reads established that the critic's defect is a CONDITIONING failure in the win head — it emits one
near-marginal win probability regardless of opponent AND of its own team — so the arms built
against it are read on the meters that measure that, not on strength and not only on G1. The two
primaries are the **turn-1-3 between-opponent SPREAD RATIO** of `V` against the outcome (a
calibrated critic's is **1.0**) and the **turn-1 own-team leave-one-battle-out win-rate R²** of
`V`. Both are HEADLINE rows and both appear in the ledger quote. ⚠️ The noise-corrected ratio is
CLAMPED and can sit at or below its own interval's lower bound — **the interval is the read**, and
the unclamped companion is printed beside it. ⚠️ The Elo-slope row is OMITTED WITH A REASON when
the run's snapshot-ladder refit is not bot-anchored; it is never reported on two scales.

🚨 **THE READ IS QUOTA-MATCHED BY DEFAULT, AND `UNMATCHED` IS A REFUSAL TO LABEL** (2026-09-09).
The ladder's arms are traced at different outcome quotas — 40/40/10 against the control's 5/10/5 —
so their read frames differ by 2-4x, and a conditioning row whose estimator is a **FIT on the
frame** (`cond.own_team_r2.*`, `cond.opp_class_auc.t1`) or an **UNCORRECTED second moment**
(`cond.spread_ratio_raw.*`) has an expectation that moves with frame SIZE. Rule 17's
capture-rate reweighting corrects the loss-ENRICHMENT and does **not** correct that. So
`main.ops.critic_read` prints both cycles' **REALIZED per-opponent capture profiles** in its
header — always, matched or not — and where the caps differ it subsamples the RICHER side to the
poorer side's cap (`main.ops.quota_match`, ~8 s), over 21 seeded IN-MEMORY draws with the capture
rates recomputed per draw, and decides that row's label on the **MATCHED** delta. The as-traced
value is printed beside it marked **UNMATCHED** and is never labelled, and a second
**decoder-matched** rung is printed because equal battle counts do not give equal DECODER frames.
Every other row — the noise-corrected spread ratio, the spread delta, the Elo slope, every gate row
and every identity row — is a weighted mean or a regression on cell means, is unaffected by frame
size given correct weights, and is read AS TRACED.

**What to do with each marker.** *Header says SYMMETRIC* — both sides carry the same realized cap;
the report is what the tool printed before matching existed, and nothing needs saying. *A row says
MATCHED* — quote the matched Δ, and never quote the UNMATCHED number beside it as a result.
*A row says `UNMATCHED — not a reading`* — `--no-quota-match` was passed on an unequal pair; on an
unequal frame neither DETECTED nor NOT DETECTED is a claim the report may make, so **re-run without
the flag** rather than reading the row. 🚨 **Match a CONTROL-vs-CONTROL read too**: the replicate
floor carries the same asymmetry with the same sign, and a floor magnitude read off an unmatched
decoder row bakes the artefact into every later arm's bar. The cost of getting this wrong is
measured: `cond.own_team_r2.t1` read **+0.084 [+0.032, +0.183] DETECTED** as traced and **+0.008
[−0.135, +0.082] NOT DETECTED** matched, and the first was WITHDRAWN (ledger 2026-09-09 ·
RETRACTION).

🚨 **WHEN THE BINDING CONSTRAINT IS POWER, RE-READ THE CHECKPOINT — DO NOT RETRAIN** (2026-09-09).
A live eval cycle is sized for a training run, not for a measurement: 100 games against 12
opponents, of which the outcome quota persists ~200 traced battles a side. Five 10M arms were read
that way and registered **zero** detected rows, and the floor read showed why — the turn-1-3
spread ratio's battle-clustered CI is ~±0.4 around a control value of ~0.12 (only 12 opponent
cells), resolution rows resolve only deltas above ~0.03, and the run-to-run floor on the
low-variance rows (bot resolution 0.010, own-team R² 0.012, spread ratio t1-3 0.028) sits **below**
the battle-level CI width. That last fact is the diagnosis: the read was limited by how many
battles it saw, not by how similar the arms are.

Every finished arm still has its 10M checkpoint, so the fix costs CPU and no GPU:

```bash
export PYTHONPATH=$PYTHONPATH:src && export CUDA_VISIBLE_DEVICES=""
# per side — OFFLINE, CPU, niced; a training arm normally shares the box, so cap the workers
nice -n 15 python -m main.ops.eval_trace_gen <run>@10000032 --games 400 --sentinels 6 \
    --out <tmp>/hp_eval/<run> --workers 4 --concurrency 1 --seed 20260909
# then the read, with BOTH sides pointed at generated cycles
python -m main.ops.critic_read <arm> --control <control> --step 10000032 \
    --arm-traces <tmp>/hp_eval/<arm> --control-traces <tmp>/hp_eval/<control> --out <dir>
```

**The four things that make this a measurement and not just a bigger number.**

1. **It is not a re-implementation of eval.** It drives the same `main.eval_worker` over a
   `ShardedEvalPool` plan, so the same LocalBattleRunner / EvalRLPlayer / BattleRecorder path
   writes the same npz keys and the same `selection_schema` 2 manifest. The eval REGIME is READ
   from the run's `eval_sentinel_greedy`, never assumed — that key names an opponent-regime
   boundary worth ~8.9 pp to the trainee, and a run that recorded none is REFUSED.
2. **Capture is ALL by default.** The live quota exists to bound a training run's disk; here the
   traces ARE the measurement, and a loss-enriched subsample is exactly what costs the
   low-variance rows their power. 400 games × 12 opponents fully captured is ~4,800 traced
   battles against a live cycle's ~200.
3. 🚨 **AN OFFLINE FRAME AND A LIVE FRAME ARE DIFFERENT POPULATIONS, and `critic_read` REFUSES to
   difference them.** They differ in games, possibly in opponent count, and decisively in whether
   the traced battles are a random sample or the outcome quota's loss-enriched slice — and every
   conditioning row is a statistic OF its frame. The v3 quota match corrects a difference in
   capture RATE between two frames of the same shape; it cannot turn a 400-game full-capture frame
   into a 100-game quota one. Both sides offline, or both live. Two offline frames must further
   agree on games / opponent set / capture rule / sentinel regime; a differing **seed** is fine and
   is deliberately unchecked, being two draws from one population.
4. **The report says which population it read, and so does the ledger quote.** The header states
   both sides' populations in words — the live one too, because a header that describes only the
   unusual side invites the reader to treat the other as the neutral default — and the one-line
   quote carries an `OFFLINE-GENERATED` marker beside its games/opponent spec, for the same reason
   `CROSS-STEP` exists.

🚨 **NEVER PUT THE HIGH-POWER ROWS IN A TABLE BESIDE THE 100-GAME DELTAS.** They are a different
number of games, possibly a different number of opponent cells, and a different capture rule. Write
them as their own table, against **their own floor pair** — a control-vs-control read generated at
the identical spec. Without that floor a high-power DETECTED is a detection against zero, and the
lesson of the 2026-09-09 RETRACTION is precisely that a bigger frame moves fitted rows on its own.

⚠️ **THE SENTINEL COUNT IS CLAMPED BY WHAT THE RUN SAVED, and is never padded.** `--sentinels 6`
on a 10M arm that kept four snapshots (one of them at the read step, excluded as a
50%-by-construction self-mirror) yields **three** — the same opponent-cell count as the live cycle,
so on these runs the power comes from games and full capture, not from more cells. The manifest
records `sentinels_requested` beside `sentinels_used`; report the gap rather than the request.

🚨 **A GENERATED CYCLE MUST SAY IT FINISHED, AND `critic_read` REFUSES ONE THAT DOES NOT.** A
cycle whose workers die part-way still writes a perfectly well-formed manifest — nominal
`--games`, the full opponent set, a recorded `selection`, capture rates of 1.0 — so nothing else
on disk separates a finished cycle from a truncated one, while the truncated one is a SMALLER
FRAME and frame size moves every fitted conditioning row on its own. The generator therefore
records `battles_expected` beside `battles_played` and a `complete` flag, prints a loud INCOMPLETE
line, and `critic_read` refuses both an incomplete side and one whose completeness is UNRECORDED
(an older producer — unknown is never read as yes). ⚠️ **This is not hypothetical: it happened on
2026-09-09.** A generation was running out of a git worktree, `scripts/land.sh` removed that
worktree on landing, and all four workers died at once with `OSError: failed to make path
absolute`; the cycle landed at 71% of its plan and read as nominal. **Run a long generation from
the MAIN checkout, never from a worktree you intend to land and remove.**

⚠️ **REPRODUCIBILITY IS `--concurrency 1`, NOT `--workers 1`.** A shard's streams are keyed on
`(seed, opponent, shard index)` — deliberately not on the worker id, which work-stealing decides in
a race — so the worker count does not enter the result. Above concurrency 1 several battles of one
shard share the process-global `random` stream the scripted bots draw from, and the order they draw
in is a timing race; the manifest records both numbers and marks the cycle NOT reproducible rather
than letting a number wander without saying so.


🚨 **`critic_read` AND `value_sidecar_read` ANSWER DIFFERENT QUESTIONS AND NEITHER SUPERSEDES THE
OTHER** (`gen3_value_sidecar_v1`, 2026-09-08). `critic_read` reads EVAL battles — a greedy trainee,
a fixed roster, a quota that prefers losses — and asks whether the critic is calibrated on the
**eval** distribution. `value_sidecar_read` reads the run's own **training buffer**, scored against
`win_target`, the label the BCE actually minimises, and asks whether it is calibrated on the
distribution it is being **fit to**. Every probe this project owned before the sidecar read eval
battles, so the second question was simply unmeasurable. A disagreement between the two is a
finding about GENERALISATION, not a defect in either instrument — report it as one.

⚠️ **The sidecar cannot be reconstructed after the fact.** It reads the rollout buffer, which is
gone the moment `train()` returns; a run launched without `--value-sidecar` has no training-side
read available at any later date. It is ON by default under `--critic winprob`, which is the only
regime where `v` is a probability at all.

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

**🚨 AMENDMENT (owner, 2026-09-11) — THE 2026-09-07 ONE-LINER ALLOWANCE IS WITHDRAWN, AND THE ROUNDS
BETWEEN SESSIONS ARE CAPPED.** *"why do you have so many routine wake ups? It seems you are churning
all the time… same thing for the training agent and lots of back of forths that go round after
round."* The per-cycle progress line was read as churn, not as the welcome highlight the 09-07
amendment intended, because it fired on a schedule rather than on an event. **The Training Run
session now sends exactly four things and nothing else:**

1. **the bank line at completion** — and the FULL dose table goes in the ledger entry, not in a
   message;
2. **a watch item firing** — a crash, a void condition, the >25 % timeout line, a G7 breach;
3. **a MAJOR finding**;
4. **a point that needs a DECISION before a number lands** — sent ONCE, carrying the decision it
   needs, and answered once.

No per-cycle status lines. No mid-run dose tables unless a watch item fires. No acknowledgements —
an "accepted" message is itself a round. **The orchestrator's half:** refinements and corrections are
folded into the next read entry rather than landed one per round, and receipt is not acknowledged.

🚨 **A POLLING CRON WITH NOTHING RUNNING IS PURE CHURN — DELETE IT, do not slow it.** The wake itself
is the cost the owner named, so a cron survives only while an arm is live AND only to catch the four
items above; in REPORT-AND-WAIT there is nothing to poll, and the orchestrator's next message is the
wake. Re-create it on the next GO, written to this contract.

- **Pre-registered kill conditions are executed, not debated.** Under `--critic winprob` stall rate
  and mean episode length are standing KILL conditions (a [0,1] critic cannot rank a timeout below a
  loss). The famine pre-test compares against the named comparator at matched SNAPSHOT COUNT with
  the registry's floor.
- **Never quote a mid-run ELO or delta.** The newest Bradley-Terry node is systematically inflated;
  read `<run>/snapshot_ladder/ladder.json` at run END, and compare runs at matched snapshot count,
  never matched step (`feedback_elo_reading_rules`).
- **First-restart checks:** the decision that is registered for it (today: `vf_coef` from the median
  of the last 20 rollouts' `grad/value_policy_logratio`, **read from the TB EVENTS**); `python -m
  main.sidecar_audit <run>` shows the pin is unchanged across the restart. One command:
  `scripts/ops/restart_read.sh <run>` — it reads the statistic from the events and prints the child
  log's figure only as a labelled CROSS-CHECK (warning above 0.10 log10, never deciding).
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
