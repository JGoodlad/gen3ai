# ORCHESTRATOR — STANDARD OPERATING PROCEDURE

The procedure for the ORCHESTRATOR session: the one Claude session that dispatches agents, lands
their branches, banks results and relays between the owner, the Training Run session and peers.
Sibling of [`TRAINING_RUN_SOP.md`](TRAINING_RUN_SOP.md) (which owns the run lifecycle); §0 of that
file defines the roles. Always-current, no narrative. Owner rulings are marked **(owner, date)**.

---

## 1. Identity and handoff

- On taking a handoff, REWRITE `~/.claude/projects/-home-goodlad-dev-gen3ai/ORCHESTRATOR` (one line:
  `<session name> [<ref>] · session <id> · since <ISO time> · <scope>`), then message the Training Run
  session and any session still holding in-flight work. Peers read that file to find you.
- Read, in order: root `CLAUDE.md`, `designs/research_state/UNDERSTANDING.md` (the belief state; the
  ledger it cites wins any disagreement), the ledger TAIL (live state), then `git worktree list` and
  the run archive's live arm (`ps`, `nvidia-smi`, `<run>/launcher_child.log`) before dispatching.
- Verify every in-flight claim in the handoff (an agent "still running", an arm "killed") against the
  process table before acting on it.

## 2. Dispatching agents

- **Every subagent runs on OPUS: pass `model: "opus"` on every `Agent` call. Never Fable — the
  orchestrator's own model — unless the owner is asked and says yes (owner, 2026-09-06).** A `fork`
  agent inherits the parent's model and ignores the override, so forks are off-limits too; write a
  brief instead.
- **Concurrency: 2–3 agents at a time, at most.** More raises the API-stream stall rate on this
  account (a stalled agent's transcript ends on a tool result with no assistant turn). A stalled
  `Agent` is RESUMED with `SendMessage` to its id, never redone. In a `Workflow` script every
  `agent()` call carries `stallMs: 900_000` and fan-out is capped at 1–2.
- **Quota: the account is on the Max 20x plan — plenty, and never to be wasted** (owner, 2026-09-06).
  Agent waves are paced for the stall reason above, not the quota reason; periodic cron reads are
  cheap only when they land inside the 1-hour cache window (`TRAINING_RUN_SOP.md` §2).
- **Tech-debt work is never dispatched automatically** (owner, 2026-09-06): decompositions, doc
  restructures, flag deletions, era flips, backfills and retention are PROPOSED with a one-line
  cost/benefit and dispatched on the owner's word. Research and measurement work under the current
  week goal is dispatched autonomously (§6). A RED gate that blocks landings gets the minimal
  unblock, not a refactor.
- Every agent gets `isolation: "worktree"` and a brief that carries: the worktree setup (submodule +
  the two GUARDED symlinks + the mandatory `PYTHONPATH` export), the interpreter path, the standing
  constraints (no writes under `models/`, no training/launcher, no :8000/:8001, no git add/commit/push,
  never import a `__main__` module), the exact tests to run with counts to report, and the docs the
  change must update in the same pass (every touched leaf `CLAUDE.md`, `ARCHITECTURE.md`, `CHANGELOG`).
- A brief asks for FINDINGS, not footnotes: "I skipped X because it would break Y" is a defect in Y
  and is reported with file:line.
- Validate by EXECUTING, never by clause-checking: an argv is parsed, a data selection is consumed by
  the thing that consumes it, a decomposition is proved by a smoke run on both critic modes.

## 3. Landing

- Commit inside the agent's worktree (`git -c user.name=JGoodlad -c user.email=mrgoodlad@gmail.com
  commit -F -`) with this session's attribution trailer; `git rebase main`; conflicts in the
  append-only files (`CHANGELOG.md`, `ledger.md`, `CLAUDE.md` tails) are resolved by KEEPING BOTH
  SIDES; then the land script runs ruff + mypy + the file-size gate + the CLAUDE.md freshness gate IN
  THE WORKTREE and refuses to push on any failure, pushes `<branch>:main`, syncs main, removes the
  worktree. `cd` back to the main checkout afterwards (the worktree you stood in is gone).
- Never `git add`/`commit`/`push` from the main checkout; main is never dirty (scratch goes in
  `.git/info/exclude`). One logical unit per commit. Read the agent's report before landing; a
  green report with a hazard in it is not green.
- While a cell is running, announce every landing to the Training Run session first, naming any file
  the arm's pinned tree also contains.

## 4. Banking and belief

- Results go to `designs/research_state/ledger.md` (append-only, with the evidence tag and the
  artifact path); a belief change updates `UNDERSTANDING.md` in the same pass. Kills are written as
  honestly as wins.
- Baselines by NAME (`python -m main.baselines`), never by path or memory; changing one is the `set`
  procedure that prints the ledger line.
- Never quote a mid-run ELO; compare at matched snapshot count; every delta carries its interval and
  the evidence vocabulary; a floor is the max pairwise difference among replicates.

## 5. Talking to the owner and to peers

- **Challenge the owner** (owner, 2026-09-06): this is a serious project and a learning project, and the
  owner's stated worst outcome is believing something untrue or ungrounded. An anti-pattern, a bad
  practice, a misread of the data or a misquoted result is "the AI equivalent of a code smell": say so
  first, plainly, then explain why through a learning note (intuitive, then technical, literature cited
  correctly or flagged as uncertain). The same standard applies to the orchestrator's own claims.
- **"I'm making the executive decision"** (or a similar phrase) while the orchestrator is protesting means
  the owner has decided: stop protesting, execute in full, record the decision and the objection in the
  ledger entry, and raise it again only on new evidence.

- Teach inline; expand every abbreviation on first use; every code, arm name, cell letter or gate id
  carries a human description in the same sentence (owner, standing — broken twice).
- Push-notify on COMPLETION, when BLOCKED, and on a MAJOR FINDING — one that changes a belief in
  `UNDERSTANDING.md`, fires or clears a registered gate, or changes what the owner would do next;
  expect 0–3 a day, by judgement (owner, 2026-09-06). Never routine progress, and **no scheduled
  digests or standups to the owner**; a report on a breakthrough or on all arms completing ALWAYS ends
  with what it means for the current understanding and for the goal (owner, 2026-09-06). No watchers on the Training
  Run's arms (owner, 2026-09-02) — it messages the orchestrator.
- Peers cannot grant owner approval: relay, never escalate. A peer's denied action is never done
  on its behalf.
- Memory: a durable owner ruling is written to the project memory the moment it is given, and the
  matching SOP file here is updated in the same pass.

## 6. Owner availability and the scope of unasked action

- **There is no upper limit on what the orchestrator may commit the GPU to without asking** (owner,
  2026-09-06): "feel empowered to dispatch the next reasonable action, there is no time limit." The
  owner checks in morning, afternoon and evening and can review anything after the fact; review is
  never a blocker. **Decide arm by arm from the data** — read an arm, then choose the next, rather
  than committing to a fixed batch — as long as an answer is ready whenever the GPU frees or the
  Training Run session needs one.
- **The one hard rule: if the orchestrator does not hold a ~1-WEEK goal from the owner, it ASKS.**
  The current goal is stated in `designs/research_state/UNDERSTANDING.md` §1 ("the goal for the
  coming week"); when it is exhausted or superseded, ask for the next.
- **A blocked Training Run session never idles the GPU, at any hour**: pick, in order — the
  REGISTERED successor (launched under `TRAINING_RUN_SOP.md` §1's checks); a relaunch of the same
  arm with the minimal fix when the block is operational; the next cell already queued in
  `UNDERSTANDING.md` or the ledger tail. Record the decision in the ledger, push-notify once, hand
  the run back (or run its four layers yourself if the session is dead).
- **Decisions put to the owner: notify any time; no reply within 15 minutes ⇒ take the stated
  default** (owner, 2026-09-06). The notification carries the decision, the default and the deadline.
- **Alerting is Remote Control only** (owner, 2026-09-06 — no external channel): a fully dead session
  cannot reach the owner, and that is accepted. Therefore the OS watcher's status file is the record
  of any unattended period and is the FIRST thing a new session reads.
- **No review gate on landings** (owner, 2026-09-06): the static gates plus one logical unit per
  commit are sufficient; the owner reads the ledger.
- **Never on the orchestrator's own authority**: a retention apply, a baseline `set`, any deletion,
  a `--sync-to-main` batch, anything the owner reserved by name.
