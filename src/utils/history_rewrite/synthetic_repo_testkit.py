"""A small synthetic repository with invented account names, for the history-rewrite tests.

``make_repo(tmp)`` builds (SHAs in the returned dict)::

    root --- intro --- cite --- msg      (main; annotated tag ``v-intro`` on intro, its message names an account)
       \\
        side                              (a branch from root: must stay untouched by the rewrite)

``intro`` adds ``results.json`` at ``RESULTS_PATH`` (three flagged accounts + a top25 list) and a prose file naming one
of them in another case; ``cite`` adds a doc citing ``root`` and ``intro`` by 7-hex and 40-hex; ``msg`` replaces the names in results.json and prose.md by hashes (as the real history did) and its MESSAGE names an account; one
account (``sidewalker``) also stands alone in ``side.txt``, outside the account-data files, so it is SCOPED and waived there. The invented names below appear in no real data.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from utils.history_rewrite.names import RESULTS_PATH

NAME_A_ID, NAME_A_DISPLAY = "zorblaxpilot", "ZorblaxPilot"
NAME_B_ID, NAME_B_DISPLAY = "quibblefnord", "Quibble Fnord"
NAME_C = "Wumpus Hunter"  # top25 only
NAME_D_ID, NAME_D_DISPLAY = "sidewalker", "SideWalker"  # also stands alone in side.txt, OUTSIDE the account-data files
ALL_NAMES = (NAME_A_ID, NAME_A_DISPLAY, NAME_B_ID, NAME_B_DISPLAY, NAME_C, "wumpushunter", NAME_D_ID, NAME_D_DISPLAY)

_ENV = {"GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "T",
        "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


def git(repo: str | Path, *args: str, env_extra: dict[str, str] | None = None) -> str:
    env = {**os.environ, **_ENV, **(env_extra or {})}
    return subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main", *args],
                          capture_output=True, text=True, check=True, env=env).stdout.strip()


def _commit(repo: Path, msg: str, when: int) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg, env_extra={"GIT_AUTHOR_DATE": f"{when} +0000", "GIT_COMMITTER_DATE": f"{when} +0000"})
    return git(repo, "rev-parse", "HEAD")


def make_repo(tmp: Path, name: str = "src_repo", root_mentions: bool = False) -> dict[str, str]:
    repo = tmp / name
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "README.md").write_text("root readme\n" + (f"early mention of {NAME_A_ID}\n" if root_mentions else ""))
    sha = {"path": str(repo)}
    sha["root"] = _commit(repo, "root", 1_700_000_000)
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "side.txt").write_text(f"side line, {NAME_D_ID} was here\n")
    sha["side"] = _commit(repo, "side branch commit", 1_700_000_100)
    git(repo, "checkout", "-q", "main")
    results = repo / RESULTS_PATH
    results.parent.mkdir(parents=True)
    results.write_text(json.dumps({"accounts": {
        "flagged": {NAME_A_ID: {"name": NAME_A_DISPLAY, "games": 9}, NAME_B_ID: {"name": NAME_B_DISPLAY, "games": 4},
                    NAME_D_ID: {"name": NAME_D_DISPLAY, "games": 2}},
        "top25": [{"name": NAME_A_DISPLAY, "games": 9}, {"name": NAME_C, "games": 7}]}}, indent=1) + "\n")
    (repo / "prose.md").write_text(f"The account {NAME_A_ID.upper()} played 9 games; {NAME_C} 7; fnordless text stays.\n")
    sha["intro"] = _commit(repo, "introduce results", 1_700_001_000)
    git(repo, "tag", "-a", "v-intro", "-m", f"tag naming {NAME_B_DISPLAY}", sha["intro"])
    (repo / "docs").mkdir()
    (repo / "docs" / "cites.md").write_text(
        f"root {sha['root'][:7]} and intro {sha['intro'][:7]}, full {sha['intro']}; pin_{sha['intro'][:8]}; "
        f"tag x-{sha['intro'][:8]}; blob-ish deadbeefcafe0123 stays; {'0' * 40} stays\n")
    sha["cite"] = _commit(repo, f"cite {sha['intro'][:7]}", 1_700_002_000)
    (repo / "later.txt").write_text("later\n")
    results.write_text(json.dumps({"accounts": {"account_ids": "salted sha256", "flagged": {"1f3a": {"games": 9}}}}, indent=1) + "\n")
    (repo / "prose.md").write_text("The account 1f3a played 9 games; the rest stay anonymous.\n")
    sha["msg"] = _commit(repo, f"scrub the names; {NAME_B_ID} was one", 1_700_003_000)  # the tip holds no name
    # a commit that does NOT descend from `intro` yet whose MESSAGE cites it (filter-repo would re-write the message and so its sha)
    git(repo, "checkout", "-q", "side")
    (repo / "side2.txt").write_text("after the introducing commit, on the side line\n")
    sha["side2"] = _commit(repo, f"side line, written after {sha['intro'][:7]}", 1_700_003_500)
    git(repo, "checkout", "-q", "main")
    # a detached commit that NO ref reaches (a worktree pin): it must still get a new sha
    git(repo, "checkout", "-q", "--detach", sha["cite"])
    (repo / "pin.txt").write_text("a pin\n")
    sha["pin"] = _commit(repo, "detached pin", 1_700_004_000)
    git(repo, "checkout", "-q", "main")
    return sha
