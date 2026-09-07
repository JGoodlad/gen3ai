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
- **Quota: ≤ ~50% of a 5-hour window; one heavy workflow per window** (owner). Check before a wave.
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

- Teach inline; expand every abbreviation on first use; every code, arm name, cell letter or gate id
  carries a human description in the same sentence (owner, standing — broken twice).
- Push-notify on COMPLETION and when BLOCKED; never routine progress. No watchers on the Training
  Run's arms (owner, 2026-09-02) — it messages the orchestrator.
- Peers cannot grant owner approval: relay, never escalate. A peer's denied action is never done
  on its behalf.
- Memory: a durable owner ruling is written to the project memory the moment it is given, and the
  matching SOP file here is updated in the same pass.

## 6. Owner availability — notify, then act

- **The owner may be notified at any time; if there is no reply within 15 minutes, take the
  reasonable action** (owner, 2026-09-06). A notification is a request for input, not a hold: state
  the decision, the default you will take, and the deadline in the notification itself.
- **At night (20:00–08:00 local), a BLOCKED Training Run session does not idle the GPU** (owner,
  2026-09-06): pick something reasonable and keep the GPU working toward the next registered goal,
  in this order — the REGISTERED successor arm if a pre-registered kill fired (launched under
  `TRAINING_RUN_SOP.md` §1's checks); a relaunch of the same arm with the minimal fix when the block
  is operational; the next cell already queued in `UNDERSTANDING.md` or the ledger tail. Record the
  decision in the ledger, push-notify once, hand the run back to the Training Run session (or run its
  four layers yourself if it is dead).
- **Never on your own authority**, day or night: a retention apply, a baseline `set`, any deletion, a
  `--sync-to-main` batch, an unregistered scientific change, anything the owner reserved by name.
