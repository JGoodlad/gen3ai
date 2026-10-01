"""The PUSH GUARD (`gen3_push_guard_v1`, 2026-10-01): refuse a push to main whose tree would REVERT
changes the pushing branch never made.

THE INCIDENT IT CLOSES. `aebae9a1` was meant to add one 17-file unit. Its worktree was based on
`7d550baa`; the agent squashed with `git reset --soft origin/main` after origin/main had moved (the refs
are shared across worktrees), so the commit's parent was the new main but its TREE was the worktree's —
carrying the OLD copies of 43 files other commits had changed since. The push deleted the Rust fork-arm
port, the owned-PPO-loop hook table and more, for ~6 minutes (restored by `277f318f`). A merge-base check
alone cannot see it: after a soft reset the merge base IS origin/main.

THE RULE. Let BASE be the branch's CREATION point (its reflog's first entry — what it was created from),
or the merge base with the remote when there is no reflog. The branch OWNS every file in
``git diff BASE HEAD`` — its own changes plus, after an honest rebase, everything main changed since it
was created. A push is refused when ``git diff <remote> HEAD`` touches a file OUTSIDE that set: the
branch's tree differs from main in a file the branch never changed, i.e. it carries a stale copy.

    honest rebase:       diff(main, HEAD) = the unit's files  ⊆  diff(BASE, HEAD) = unit + main's    -> OK
    soft-reset accident: diff(main, HEAD) = unit + 43 stale   ⊄  diff(BASE, HEAD) = the unit's files   -> REFUSED

``python -m utils.push_guard [--branch B] [--remote origin/main] [--repo DIR]`` exits 0 (OK) / 1
(REFUSED, naming the files) / 2 (cannot decide — e.g. no git). `scripts/land.sh` runs it before the push.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Optional, Sequence, Set


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def _files(repo: str, a: str, b: str) -> Set[str]:
    return {f for f in _git(repo, "diff", "--name-only", a, b).splitlines() if f}


def creation_base(repo: str, branch: str, remote: str) -> str:
    """The commit ``branch`` was created from (its reflog's oldest entry), else the merge base."""
    try:
        log = _git(repo, "reflog", "show", "--format=%H", branch).split()
        if log:
            return log[-1]
    except subprocess.CalledProcessError:
        pass
    return _git(repo, "merge-base", remote, branch).strip()


def check(repo: str, branch: str = "HEAD", remote: str = "origin/main") -> List[str]:
    """The files ``branch``'s push would change on ``remote`` WITHOUT the branch ever having changed
    them (sorted; empty = OK)."""
    if branch == "HEAD":
        branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    base = creation_base(repo, branch, remote)
    owned = _files(repo, base, branch)
    pushed = _files(repo, remote, branch)
    return sorted(pushed - owned)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--branch", default="HEAD")
    ap.add_argument("--remote", default="origin/main")
    a = ap.parse_args(argv)
    try:
        stale = check(a.repo, a.branch, a.remote)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[push-guard] CANNOT DECIDE ({exc}) — not pushing", file=sys.stderr)
        return 2
    if stale:
        print(f"[push-guard] REFUSED: the push would change {len(stale)} file(s) on {a.remote} that "
              f"this branch never changed (a stale copy — e.g. a soft reset onto a moved ref): "
              f"{', '.join(stale[:12])}{' …' if len(stale) > 12 else ''}", file=sys.stderr)
        return 1
    print(f"[push-guard] OK: every file the push changes on {a.remote} is one this branch changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
