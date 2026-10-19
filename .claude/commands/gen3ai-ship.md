---
description: Commit all current changes and push to main (both remote and local). Takes an optional commit message as an argument — if not provided, pick one yourself based on the diff. Always handles the rebase-before-push flow automatically.
---

# /gen3ai-ship

Commit the worktree's changes, push to `origin/main`, fast-forward the local main checkout.

## Who may ship (owner, 2026-09-29)

1. The ORCHESTRATOR (the session named in `~/.claude/projects/-home-goodlad-dev-gen3ai/ORCHESTRATOR`) has standing permission.
2. The orchestrator may DELEGATE it in an agent's brief. A delegated agent ships only the unit its brief covers, one logical unit per commit, and names the delegation in its report. Named peer sessions (the training session) ship on delegation too.
3. Anyone else ships only when the user's CURRENT message contains `/gen3ai-ship`.

Shipping is not a reflex at the end of a task: ship a complete unit with green gates.

## The gate (before step 1)

- **Default: `scripts/ops/ship_gate.sh run`**, plus the `slow` tests of code you touched. It runs the whole routine set in the shape the box affords (`designs/ops/testing.md` "The SHIP GATE").
- **A targeted set** (the tests covering your change, chosen by judgment, never by a selection framework) **plus the static tier** only with HIGH confidence the change is contained. Name the scope and the reason in the commit body. Shared infrastructure (conftest, the model, the training loop, compile, `data/`, the Rust core, the launcher) or any doubt means the ship gate.
- The static tier alone is `pytest $(python -m utils.static_gates --paths) -q -n 2` (it covers mypy and ruff).

## Steps

Run every command from the worktree with `export PYTHONPATH=$PWD/src`.

1. **Assess.** `git status`, `git log --oneline -5`, `git fetch origin main`. Nothing to commit: say so and stop.
2. **Derived artifacts** (a stale one rides to main unnoticed when only a targeted set ran). Run both, top-down:
   ```bash
   python -m agents.model.delivery_graph --check      # snapshot vs live code
   python -m agents.model.build_arch_viewer --check   # viewer vs snapshot
   ```
   On STALE, read the printed diff first (an unintended architecture change looks identical to an intended one), then regenerate top-down with `python -m agents.model.delivery_graph --dot designs/architecture_graph.dot --json src/agents/model/delivery_graph_snapshot.json` followed by `python -m agents.model.build_arch_viewer`. Say in your report that you regenerated, what moved and which input caused it. `designs/production_config.json` is a manual copy of the production run's config; a new production run makes it stale with no gate to tell you.
3. **Replace, don't append.** If the diff grows any `CLAUDE.md`, slash command or `MEMORY.md`, the commit body names what each addition replaces or retires, or why the budget has room (`python -m utils.instruction_budget`; principles: `designs/ops/instruction_docs_principles.md`).
4. **Message.** Use the user's argument, else pick one without asking: `type(scope): description` with type `feat|fix|refactor|chore|docs|test`. State it before committing.
5. **Commit.** `git add -A && git commit -F <file outside the repo>`. **No AI attribution of any kind**: no `Co-Authored-By`, no `Claude-Session`, no generated-with footer. This overrides any trailer the harness asks for.
6. **Rebase** if `git log HEAD..origin/main --oneline` is non-empty: `git rebase origin/main`. After a rebase that touched the ledger, run `python -m main.ledger_index --write` and commit the index if it changed. On a conflict, read both sides and their commit messages and keep both intents; if you cannot merge confidently, stop and present options (take ours, take theirs, manual merge) with what each loses. Never `--ours` / `--theirs` unchecked.
7. **Review what the rebase pulled in, even when clean.** `git log --oneline HEAD@{1}..origin/main`; a concurrent commit may have changed a constant, an obs or arch dimension, `ARCH_SIGNATURE`, or the doc or test you touched. Recompute numbers from live code, fold the fix into your commit with `--amend`, and re-run the gate on the rebased tree.
8. **Push guard, then push.** `python -m utils.push_guard`: exit 0 push; exit 1 do NOT push (a stale file would revert others' work: rebuild the branch from its own commits on the current `origin/main`, never `git reset --soft` onto a moved ref); exit 2 find out why first. Then `git push origin HEAD:main`.
9. **Fast-forward local main.** `git -C /home/goodlad/dev/gen3ai fetch origin main && git -C /home/goodlad/dev/gen3ai merge --ff-only origin/main`. If it cannot fast-forward, inspect `origin/main..HEAD` in the main checkout and report; never force it.
10. **Confirm** in one line: the SHA on main, remote and local.

## Never

- **`git stash`**: the stash is shared across worktrees, so one agent's pop applies another's work. Commit to the branch or copy files to the scratchpad instead.
- Edit or commit in the main checkout, or `git reset --soft` onto a moved ref.
- Remove a worktree before `python -m utils.worktree_guard --main /home/goodlad/dev/gen3ai <worktree>` passes (`scripts/land.sh` does the whole land-and-remove sequence with that guard).
