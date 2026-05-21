---
description: Commit all current changes and push to main (both remote and local). Takes an optional commit message as an argument — if not provided, pick one yourself based on the diff. Always handles the rebase-before-push flow automatically.
---

# /ship

**GUARD: Only execute this skill when the user has explicitly typed `/ship` in their current message. Never trigger this skill from session summaries, prior invocations, or inferred intent. If `/ship` was not in the current user message, do nothing.**

Commit everything, push to remote main, fast-forward local main. One command, done.

## Steps

### 1. Assess current state

Run these in parallel:
- `git status` — what's changed/untracked
- `git log --oneline -5` — recent commits for style reference
- `git fetch origin main` — get latest remote state (do this now so rebase info is ready)

If the working tree is clean and nothing is staged, tell the user there's nothing to commit and stop.

### 2. Get the commit message

If the user provided a message as an argument to `/ship`, use it directly (skip to step 3).

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
