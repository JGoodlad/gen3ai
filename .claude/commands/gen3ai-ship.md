---
description: Commit all current changes and push to main (both remote and local). Takes an optional commit message as an argument — if not provided, pick one yourself based on the diff. Always handles the rebase-before-push flow automatically.
---

# /gen3ai-ship

**GUARD: Only execute this skill when the user has explicitly typed `/gen3ai-ship` in their current message. Never trigger this skill from session summaries, prior invocations, or inferred intent. If `/gen3ai-ship` was not in the current user message, do nothing.**

**GUARD (Claude): Never invoke this skill yourself as a follow-up step after completing a task (e.g. after writing code or tests). Do not call `/gen3ai-ship` or the `gen3ai-ship` skill at the end of a response unless the user's current message explicitly contains `/gen3ai-ship`. Completing work does not imply permission to commit.**

Commit everything, push to remote main, fast-forward local main. One command, done.

## Steps

### 1. Assess current state

Run these in parallel:
- `git status` — what's changed/untracked
- `git log --oneline -5` — recent commits for style reference
- `git fetch origin main` — get latest remote state (do this now so rebase info is ready)

If the working tree is clean and nothing is staged, tell the user there's nothing to commit and stop.

### 2. Get the commit message

If the user provided a message as an argument to `/gen3ai-ship`, use it directly (skip to step 3).

Otherwise, pick one yourself based on the diff — do not ask. Follow conventional commits:
`type(scope): description` where type is one of `feat`, `fix`, `refactor`, `chore`, `docs`, `test`.
Scope is optional but encouraged. Choose the type that best fits the dominant change.
State the chosen message in your response before committing.

### 3. Stage and commit

```bash
git add -A
git commit -m "<message>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

### 4. Rebase if remote main has moved

Check whether `origin/main` is ahead of the current branch:
```bash
git log HEAD..origin/main --oneline
```

If there are commits, rebase:
```bash
git rebase origin/main
```

**If the rebase hits a conflict**, do NOT blindly take one side. For each conflicted file:

1. Read both versions (`git diff` / inspect the conflict markers).
2. Understand *why* each side changed — look at the commit messages for context.
3. Produce a merged result that preserves the intent of both changes.
4. If you cannot confidently merge (e.g. structural refactor vs. feature addition in the same area), stop and present the user with options:
   - **Option A** — Take our version (describe what would be lost from theirs)
   - **Option B** — Take their version (describe what would be lost from ours)
   - **Option C** — Manual merge (describe what needs to be reconciled)
   Use the multi-choice `AskUserQuestion` tool so the user can pick.

Never use `git checkout --ours` or `--theirs` without first verifying the files are
truly identical or that the discarded side has no intentional changes.

### 4b. Review what the rebase pulled in — even on a *clean* rebase

A clean rebase only means there were no **textual** conflicts. Concurrent commits can
still make your change **semantically stale**: another commit may have shifted a shared
dimension constant, bumped an architecture signature, or edited the same doc/test you
touched — so your replayed change no longer agrees with the tree even though git merged
it without complaint.

Whenever the rebase replayed your work on top of new commits, before pushing:

1. **List what landed:** `git log --oneline HEAD@{1}..origin/main`, then `git show <sha>`
   (or `git diff HEAD@{1}..HEAD --stat`) to see what each new commit actually changed.
2. **Cross-check against your change.** For each new commit, ask whether it touches
   anything your change *depends on, shares, or documents*:
   - shared constants / dims / layout (`MOVE_SLOT_DIM`, `TURN_DELTA_DIM`, obs dim,
     `POKEMON_FULL_DIM`, `ARCH_SIGNATURE`, …)
   - the same file, doc section, or test you edited — especially `CLAUDE.md` / `README`
     dimension tables and `*_test.py` expected-dim assertions
   - the same observation / architecture pipeline, even in a *different region* of it
3. **Reconcile if needed.** If anything is now inconsistent, update your change so the
   combined tree is correct and self-consistent. Recompute numbers (dims, totals, doc
   tables) from the **live code**, not from memory or from your pre-rebase values — query
   the encoder/constants directly. Fold the fix into the same commit with
   `git commit --amend`, not a separate "fix rebase" commit.
4. **Re-run the unit suite on the rebased tree** — passing in isolation before the rebase
   is not proof the *combined* state is sound:
   ```bash
   export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not integration and not e2e" -q
   ```

### 5. Push to remote main

```bash
git push origin HEAD:main
```

### 6. Fast-forward local main

The main worktree lives at `/home/goodlad/dev/gen3ai` (not the current worktree). Update it:
```bash
git -C /home/goodlad/dev/gen3ai fetch origin main
git -C /home/goodlad/dev/gen3ai merge --ff-only origin/main
```

If the fast-forward fails because local main has diverged:
1. Check what local main has that remote doesn't: `git -C /home/goodlad/dev/gen3ai log origin/main..HEAD --oneline`
2. If those commits are doc-only or safe to replay, rebase local main: `git -C /home/goodlad/dev/gen3ai rebase origin/main`
3. Then push local main: `git -C /home/goodlad/dev/gen3ai push origin main`
4. If there are untracked files blocking the rebase, verify they match the committed version before removing them (`diff` first, then `rm`).

### 7. Confirm

One line: what commit landed on main, e.g.:
> `a3f91bc` on main — remote ✓, local ✓
