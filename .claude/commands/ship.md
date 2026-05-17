---
description: Commit all current changes and push to main (both remote and local). Takes an optional commit message as an argument — if not provided, ask the user for one. Always handles the rebase-before-push flow automatically.
---

# /ship

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

Otherwise, show a one-line summary of what changed (from `git status`) and ask:
> What's the commit message? (e.g. `fix(env): correct worktree symlink instructions`)

Follow conventional commits: `type(scope): description` where type is one of `feat`, `fix`, `refactor`, `chore`, `docs`, `test`. Scope is optional but encouraged. If the user gives you a bare description without a prefix, pick the right type yourself and confirm.

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

If the rebase hits a conflict, stop and tell the user clearly which files conflict. Do not attempt to resolve conflicts automatically.

### 5. Push to remote main

```bash
git push origin HEAD:main
```

### 6. Fast-forward local main

The main worktree lives at `/home/goodlad/dev/gen3ai` (not the current worktree). Update it:
```bash
git -C /home/goodlad/dev/gen3ai pull --ff-only
```

If this fails because local main has diverged, report it but don't treat it as a failure — remote is already updated, which is what matters for training runs.

### 7. Confirm

One line: what commit landed on main, e.g.:
> `a3f91bc` on main — remote ✓, local ✓
