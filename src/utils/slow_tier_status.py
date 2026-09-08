"""`gen3_slow_tier_status_v1` — THE LAST-KNOWN STATUS OF EVERY `slow` TEST, AS A COMMITTED ARTIFACT.

**The hole this closes.** The routine gate is `-m "not slow and not e2e"`. A `slow` test is
therefore *deselected*, and a deselected test cannot fail — so a red `slow` test is invisible
between full-suite runs, which happen at most once before a ship. That is not hypothetical: on
2026-09-07 `tb_relevance_test::test_a_winprob_run_emits_no_noise_tag_and_every_live_tag` was red on
main for a day behind exactly that marker, and it is the same shape as the obs-golden linchpin
riding main RED three separate times behind the old `-m "not integration"` cut. A gate that reads
GREEN while a test is red is the GIGO class itself.

**The mechanism, in one sentence:** the slow tier WRITES its verdict here, and an unmarked gate test
in the routine suite READS it — so a red `slow` test costs one JSON read to surface instead of a
31-minute run.

* **Writer** — the root `conftest.py` records every `slow` test that actually RAN and merges each
  verdict in **at that test's teardown** (plus a sweep at session finish). It only ever merges: a
  run of one slow test never truncates the other 79 rows, and a routine run (no slow test ran)
  writes nothing at all. Banking per test rather than per session is deliberate — the tier is ~2
  hours beside a live run, and a session interrupted at test 79 of 80 must not throw away 79
  verdicts.
* **Reader** — `src/slow_tier_status_gate_test.py`, unmarked, in every tier, ~free.

**FOUR VERDICT CLASSES, and only one of them is fatal.**

| class | meaning | gate |
|---|---|---|
| `fail` | the slow tier RAN this test and it failed | **FAILS the routine gate**, naming the test and the commit it failed at |
| `inconclusive` | it failed with a TIMEOUT signature | reported, never fatal — *a timeout is never a semantic outcome* |
| unrecorded | collected as `slow` this session, no row here | reported — a new slow test is not a regression |
| stale | the row's commit is far behind HEAD, or not an ancestor of it | reported |

**WHY STALENESS IS REPORTED AND NOT FATAL.** A stale row is a statement about the *schedule*, not
about the code — nobody has run the tier lately. Failing on it would make "the tier has not been run
for 25 commits" indistinguishable from "a test is red", which is precisely the conflation this file
exists to remove, and it would breed a reflex `GEN3AI_SKIP_SLOW_STATUS_GATE=1` that costs the real
signal too. ⚠️ **The honest limitation, stated rather than hidden:** a slow test that broke *since*
the last recorded run still reads `pass` here. Nothing short of running the tier can close that, and
the staleness report is what says how much trust the green is worth.

**WHY THE FILE IS COMMITTED, not gitignored.** Three reasons, and the worktree workflow is the
decisive one:

1. **A gitignored file would be per-worktree, and a fresh worktree is where all the work happens.**
   It would have no file, so the gate would be dead exactly where it is needed — the `models/`
   failure mode (`utils.paths.main_models_dir()` returns `None` in a worktree and every caller must
   turn that into a skip). A gate that skips in every worktree is not a gate.
2. **A red slow test then shows up in the DIFF**, which is a second, human-readable channel that
   costs nothing.
3. It travels with the tree, so the pre-ship full-suite run and the routine gate that follows it
   are talking about the same measurements.

The cost is a merge conflict when two branches both run the tier. The rows are keyed by test id, so
the resolution is always a UNION — keep both sides' rows, and where both edited one row, keep the
one with the newer `at`.

**WHY A MISSING FILE FAILS.** The house rule for a gate is that a missing artifact FAILS and never
skips — a linter that silently opts out reads exactly like a linter that found nothing. Because the
file is COMMITTED, "missing" cannot mean "fresh checkout"; it means the tree is inconsistent (a bad
rebase, a deleted file), which is worth stopping for. The failure message names the command that
produces one and the `GEN3AI_SKIP_SLOW_STATUS_GATE=1` escape hatch, so it can never strand anyone.

Refresh the whole file with the slow tier itself:

    export PYTHONPATH=$PYTHONPATH:src
    python3 -m pytest src/ -m slow -q -n 2
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "gen3_slow_tier_status_v1"

#: Where the committed artifact lives. `$GEN3AI_SLOW_STATUS_FILE` overrides it — used by this
#: module's own meta-tests to plant a red without touching the real file, and by anyone who wants
#: to record a tier run somewhere disposable.
_ENV_PATH = "GEN3AI_SLOW_STATUS_FILE"
_REL_PATH = ("designs", "ops", "slow_tier_status.json")

#: How far behind HEAD a row may be before it is REPORTED as stale. Sized off the observed cadence:
#: this tree lands 10-25 commits on a busy day, so a row older than this predates a day's work and
#: its green is worth correspondingly less. It is a REPORTING threshold, never a failure one.
STALE_COMMITS = 25

#: Verdict precedence within one test: a teardown error beats a passing call, and a timeout beats a
#: pass but loses to a genuine failure elsewhere in the same test.
#:
#: 🚨 **`skip` OUTRANKS `pass`, and that is the whole point of having a rank at all.** A skipped test
#: reports `skipped` in SETUP and then `passed` in TEARDOWN — so with `pass` ranked higher, all 44
#: skips of the first real tier run banked as green (measured 2026-09-07, before this line). A test
#: that did not run must never read as a test that ran and was fine; that is the same absence-is-not-
#: a-zero class the whole artifact exists to prevent.
_RANK = {"fail": 4, "inconclusive": 3, "skip": 2, "pass": 1}

#: A failure whose text matches one of these is recorded INCONCLUSIVE, not FAIL. The list is
#: DECLARED rather than inferred, so it cannot quietly grow into "anything that looks bad is
#: inconclusive". ⚠️ It is a heuristic in one direction only: a genuine assertion failure whose
#: message happens to contain "timed out" is under-reported as inconclusive — which is loud (the
#: gate prints every inconclusive row on every routine run), where the opposite error would be
#: silent.
_TIMEOUT_MARKERS = (
    "TimeoutError", "asyncio.TimeoutError", "timed out", "Timed out", "TIMEOUT",
    "ProgressDeadline", "INCONCLUSIVE",
)


def status_path() -> Path:
    """The status file this process reads and writes."""
    override = os.environ.get(_ENV_PATH)
    if override:
        return Path(override)
    from utils.paths import repo_path
    return repo_path(*_REL_PATH)


# ───────────────────────────────────────────────────────────────────── recording ──


def classify(failed: bool, skipped: bool, text: str) -> str:
    """`pass` / `fail` / `inconclusive` / `skip` for one phase report."""
    if failed:
        return "inconclusive" if any(m in text for m in _TIMEOUT_MARKERS) else "fail"
    return "skip" if skipped else "pass"


def merge_outcome(previous: Optional[str], new: str) -> str:
    """Fold a phase's verdict into the test's verdict, by `_RANK`."""
    if previous is None:
        return new
    return new if _RANK.get(new, 0) > _RANK.get(previous, 0) else previous


def record_results(results: Dict[str, Dict[str, Any]], path: Optional[Path] = None) -> Path:
    """MERGE ``{nodeid: row}`` into the status file and return the path written.

    Merge, never replace: running one slow test must not erase the other rows, and two xdist
    workers finishing at the same moment must not lose each other's results. The read-modify-write
    is done under an exclusive `flock` (in the temp dir, never beside the artifact) and committed
    with an atomic rename,
    so a crashed writer can never leave a half-written artifact behind.
    """
    import fcntl

    target = Path(path) if path is not None else status_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # The lock lives in the system temp dir, keyed by the target's absolute path — NOT beside the
    # artifact. A sidecar `.lock` in `designs/ops/` would be an untracked file the whole tree then
    # has to remember to ignore, and the concurrency this guards (two xdist workers) is always
    # same-box.
    lock = Path(tempfile.gettempdir()) / (
        "gen3_slow_tier_status."
        + hashlib.sha256(str(target.resolve()).encode()).hexdigest()[:16] + ".lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            doc = _read(target) if target.exists() else {"schema": SCHEMA, "tests": {}}
            doc.setdefault("schema", SCHEMA)
            tests = doc.setdefault("tests", {})
            tests.update(results)
            doc["tests"] = dict(sorted(tests.items()))
            doc["last_written"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".slow_status.")
            with os.fdopen(fd, "w") as f:
                json.dump(doc, f, indent=1, sort_keys=False)
                f.write("\n")
            os.replace(tmp, target)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return target


def make_row(status: str, *, commit: str, duration_s: float,
             contention: float, detail: str = "") -> Dict[str, Any]:
    """One test's row. `detail` is the first line of the failure, for the gate's message."""
    return {
        "status": status,
        "commit": commit,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": round(float(duration_s), 2),
        "contention": round(float(contention), 2),
        "detail": detail[:300],
    }


def _read(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or not isinstance(doc.get("tests"), dict):
        raise ValueError(f"{path}: not a {SCHEMA} document (no 'tests' object)")
    return doc


# ───────────────────────────────────────────────────────────────────── reading ──


@dataclass
class Verdict:
    """What the gate found. Only `reds` is fatal; everything else is a report."""

    reds: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    inconclusive: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    unrecorded: List[str] = field(default_factory=list)
    stale: List[Tuple[str, str]] = field(default_factory=list)
    n_rows: int = 0
    n_pass: int = 0
    n_skip: int = 0

    @property
    def ok(self) -> bool:
        return not self.reds

    def red_message(self) -> str:
        lines = [
            f"{len(self.reds)} test(s) in the `slow` tier are RECORDED RED, and the routine gate "
            "deselects them — that is what this gate exists to say:",
        ]
        for nodeid, row in self.reds:
            lines.append(f"  {nodeid}")
            lines.append(f"      failed at commit {row.get('commit', '?')[:12]} "
                         f"on {row.get('at', '?')}  ({row.get('detail', '') or 'no detail'})")
        lines += [
            "",
            "Fix the test, then re-record it:",
            "    export PYTHONPATH=$PYTHONPATH:src && python3 -m pytest <that test> -q",
            "(any run in which the test executes updates its row; the full refresh is "
            "`python3 -m pytest src/ -m slow -q -n 2`).",
            "If the test was RENAMED or DELETED, delete its row from the status file in the same "
            "commit — a row is keyed by test id and nothing else prunes it.",
        ]
        return "\n".join(lines)


def evaluate(doc: Dict[str, Any], collected: Optional[set] = None,
             commit_distance=None) -> Verdict:
    """Turn a status document into a `Verdict`. Pure — the gate test's whole decision lives here.

    ``collected`` is the set of `slow` node ids THIS session collected (the root conftest records
    it before `-m` deselection runs, so the routine gate sees the full slow set even though it
    runs none of them). ``commit_distance(sha) -> Optional[int]`` answers "how many commits is HEAD
    ahead of this one", returning ``None`` when the commit is unknown or not an ancestor.
    """
    v = Verdict()
    tests = doc.get("tests", {})
    v.n_rows = len(tests)
    for nodeid, row in sorted(tests.items()):
        st = row.get("status")
        if st == "fail":
            v.reds.append((nodeid, row))
        elif st == "inconclusive":
            v.inconclusive.append((nodeid, row))
        elif st == "pass":
            v.n_pass += 1
        elif st == "skip":
            v.n_skip += 1
        if commit_distance is not None:
            d = commit_distance(row.get("commit", ""))
            if d is None:
                v.stale.append((nodeid, "commit unknown or not an ancestor of HEAD"))
            elif d > STALE_COMMITS:
                v.stale.append((nodeid, f"{d} commits behind HEAD"))
    if collected:
        v.unrecorded = sorted(n for n in collected if n not in tests)
    return v
