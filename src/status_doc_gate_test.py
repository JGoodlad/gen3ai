"""Is `designs/STATUS.md`, the one-page dashboard, still citing real records and still recent?

**Why it exists.** STATUS is the owner's "easy way to reason about our stuff": every goal and verdict carries the
commit that proves it, and a "Running now" section that goes stale silently is worse than none. Prose has no
compiler, so two cheap mechanical checks hold it:

1. **Every cited record exists.** Each backticked hex token of 7-40 characters is a commit SHA. It must resolve to a
   commit that is an ancestor of HEAD, so a typo, an unshipped branch's SHA or a SHA orphaned by a history rewrite
   fails. At least ``_MIN_CITATIONS`` must be found, so a format change cannot make the check vacuous.
2. **"Running now" is fresh.** The ``## N. Running now (as of YYYY-MM-DD)`` heading's date must be no more than
   ``MAX_AGE_DAYS`` before HEAD's committer date, and not after it. The clock is the checkout's newest commit, never
   the wall clock, so a given checkout passes or fails the same way every run (root rule 8).

**Cost:** one ``git rev-list`` and one ``git rev-parse --verify`` per citation, well under a second. Opt out:

    GEN3AI_SKIP_STATUS_DOC_GATE=1 pytest src/ -q
"""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
from typing import List, Optional, Set

import pytest

from utils.paths import repo_path

pytestmark = pytest.mark.static   # the `static` budget tier (conftest._STATIC_BUDGET_BASE_S)

_SKIP = pytest.mark.skipif(os.environ.get("GEN3AI_SKIP_STATUS_DOC_GATE") == "1",
                           reason="GEN3AI_SKIP_STATUS_DOC_GATE=1")

DOC = ("designs", "STATUS.md")
MAX_AGE_DAYS = 3
_MIN_CITATIONS = 20
_SHA = re.compile(r"`([0-9a-f]{7,40})`")
_RUNNING = re.compile(r"^## \d+\. Running now \(as of (\d{4}-\d{2}-\d{2})\)\s*$", re.M)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(repo_path()), *args], check=True, capture_output=True,
                          text=True).stdout


def cited_shas(text: str) -> List[str]:
    return sorted(set(_SHA.findall(text)))


def running_now_date(text: str) -> dt.date:
    found = _RUNNING.findall(text)
    assert len(found) == 1, f"want exactly one '## N. Running now (as of YYYY-MM-DD)' heading, found {found}"
    return dt.date.fromisoformat(found[0])


def staleness_problem(as_of: dt.date, head: dt.date, max_age_days: int = MAX_AGE_DAYS) -> Optional[str]:
    if as_of > head:
        return f"'Running now' is dated {as_of}, after the newest commit ({head}): a typo?"
    if (head - as_of).days > max_age_days:
        return (f"'Running now' is dated {as_of}, {(head - as_of).days} days before the newest commit ({head}); "
                f"the cap is {max_age_days}. Rewrite the section (and the rows it changes) with today's date.")
    return None


def unresolved(shas: List[str], ancestors: Set[str]) -> List[str]:
    bad = []
    for sha in shas:
        try:
            full = _git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}").strip()
        except subprocess.CalledProcessError:
            full = ""
        if full not in ancestors:
            bad.append(sha)
    return bad


def _text() -> str:
    return repo_path(*DOC).read_text()


@_SKIP
def test_every_cited_record_is_a_commit_in_heads_history():
    shas = cited_shas(_text())
    assert len(shas) >= _MIN_CITATIONS, f"only {len(shas)} cited SHAs: did the citation format change?"
    ancestors = set(_git("rev-list", "HEAD").split())
    bad = unresolved(shas, ancestors)
    assert not bad, f"{'/'.join(DOC)} cites commits that are not in HEAD's history: {bad}"


@_SKIP
def test_running_now_is_recent_against_the_newest_commit():
    head = dt.date.fromisoformat(_git("log", "-1", "--format=%cs", "HEAD").strip())
    problem = staleness_problem(running_now_date(_text()), head)
    assert problem is None, problem


def test_meta_the_checks_bite():
    """Revert-proof: a fake SHA, a stale date, a future date and a missing heading each fail."""
    head = _git("rev-parse", "HEAD").strip()
    ancestors = {head}
    assert unresolved([head[:9]], ancestors) == []
    assert unresolved(["deadbeef0"], ancestors) == ["deadbeef0"]
    assert cited_shas("a `1a2b3c4d5` and `--flag` and `abc`") == ["1a2b3c4d5"]
    day = dt.date(2026, 10, 26)
    assert staleness_problem(day, day + dt.timedelta(days=MAX_AGE_DAYS)) is None
    assert staleness_problem(day, day + dt.timedelta(days=MAX_AGE_DAYS + 1)) is not None
    assert staleness_problem(day + dt.timedelta(days=1), day) is not None
    assert running_now_date("## 2. Running now (as of 2026-10-26)\n") == day
    with pytest.raises(AssertionError):
        running_now_date("## 2. Running now\n")
