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

* **Writer** — the root `conftest.py` records every `slow` test that actually RAN and, **only when the session is
  told to BANK** (`recording_target`), merges each verdict in **at that test's teardown** (plus a sweep at session
  finish). It only ever merges: a run of one slow test never truncates the other 79 rows. Banking per test rather than
  per session is deliberate — the tier is ~2 hours beside a live run, and a session interrupted at test 79 of 80 must
  not throw away 79 verdicts.
* **An ordinary session NEVER writes the committed file** (2026-11-16). The writer once ran on every session that
  executed a `slow` test (a full `pytest src/`, a targeted re-run of the tests of code you touched), so the tracked
  `designs/ops/slow_tier_status.json` came out of unrelated sessions modified, dirtied every worktree, and rode into
  unrelated commits. Banking is now a DELIBERATE act with two spellings: `GEN3AI_SLOW_STATUS_BANK=1` (the committed
  file) or `GEN3AI_SLOW_STATUS_FILE=<path>` (any file: a scratch one for a GPU window, which `main.ops.gpu_fill_window
  merge-rows` then folds in). A session that measured slow verdicts and banked none says so in its terminal summary.
* **Reader** — `src/slow_tier_status_gate_test.py`, unmarked, in every tier, ~free.

**A SKIP IS NOT A MEASUREMENT, so it never replaces one.** A slow test that skipped (a GPU test run
without `GEN3AI_TEST_ALLOW_GPU`, a missing optional dependency) keeps its banked pass / fail /
inconclusive row — verdict, commit and timestamp unchanged, so staleness keeps reporting the age of
the last REAL run — and gains a `last_skipped_at` note. A skip is banked as a row's status only
where nothing measured is on file. (`merge_row`; 2026-09-29, seven GPU PASS rows lost to SKIP.)

**FOUR VERDICT CLASSES, and only one of them is fatal.**

| class | meaning | gate |
|---|---|---|
| `fail` | the slow tier RAN this test and it failed | **FAILS the routine gate**, naming the test and the commit it failed at |
| `inconclusive` | it failed with a TIMEOUT signature, **or it never finished a CALL phase** (killed or interrupted in flight) | reported, never fatal — *a timeout is never a semantic outcome* |
| unrecorded | collected as `slow` this session, no row here — or a row that has only ever SKIPPED (no measurement) | reported — a new slow test is not a regression |
| stale | the row's test FILE changed since it was measured, or its BASE (below) is far behind HEAD or not an ancestor of it | reported |

**A MEASUREMENT'S IDENTITY: `base` + `test_blob`, never the worktree commit** (2026-11-17). A row used to stamp only
`commit` = the HEAD it was measured at, and a measurement is made in a WORKTREE, whose commits `/gen3ai-ship`'s rebase
then rewrites: the stamp became a SHA that is on no branch of main ("not an ancestor", stale forever; 354 of 730 rows
at 15 such commits on 2026-11-16). Staleness has never been about the code's content, it is a COMMIT COUNT (HEAD more
than `STALE_COMMITS` ahead of the stamp, or the stamp not in HEAD's history) - so the stamp has to be a position in
main's history that a rebase cannot move, and the one thing that cannot move is a commit already on main:

* `base` = `git merge-base HEAD main` at measurement time: the main commit the measured code was built on. Main only
  ever appends (`/gen3ai-ship` rebases then fast-forwards), so `base` is an ancestor of every later HEAD whatever
  happens to the worktree commits on top of it. The distance is counted from it, so the commits main gained between the
  fork and the ship count as unmeasured (honest), and the worktree's own commits count too (conservative by a handful of
  commits against a threshold of 25). Measured ON main, `base` is HEAD itself.
* `test_blob` = the git blob id of the test FILE as measured (the working-tree bytes, so a bank before the commit stamps
  what ran). It is a CONTENT identity: it survives a rebase unchanged when the file is unchanged, and a row whose file
  differs from today's is a measurement of a different test, so it is reported stale however near its base is. It does
  NOT cover the code the test exercises: that stays the commit count's job (the honest limitation below).
* `commit` stays as provenance only (the red message names it); no judgement reads it once `base` exists.

A row with no `base` (an older writer, a GPU window pinned before this) is judged by its `commit` as before and counted
`unstamped`; `python -m utils.slow_tier_prune --adopt-stamps` fills `base` and `test_blob` from the row's own commit
wherever git can still answer (`adopt_stamps`), and REPORTS the rest without touching it.

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

Refresh the whole file with the slow tier itself, told to bank:

    export PYTHONPATH=$PYTHONPATH:src
    GEN3AI_SLOW_STATUS_BANK=1 python3 -m pytest src/ -m slow -q -n 2
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

SCHEMA = "gen3_slow_tier_status_v1"

#: Where the READER looks and where a deliberate bank lands. `$GEN3AI_SLOW_STATUS_FILE` overrides it: this
#: module's own meta-tests plant a red without touching the real file, a GPU window writes a disposable file, and
#: a session given it records there (see `recording_target`).
_ENV_PATH = "GEN3AI_SLOW_STATUS_FILE"
#: `1` makes a session bank its slow verdicts into the COMMITTED file. Unset (or `0`), an ordinary session writes
#: nothing to it: the tracked file changes only when someone meant it to.
BANK_ENV = "GEN3AI_SLOW_STATUS_BANK"
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


def tracked_path() -> Path:
    """The COMMITTED artifact of this checkout, whatever the environment says."""
    from utils.paths import repo_path
    return repo_path(*_REL_PATH)


def status_path() -> Path:
    """The status file this process READS (the gate, the prune tool): `$GEN3AI_SLOW_STATUS_FILE`, else the committed one."""
    override = os.environ.get(_ENV_PATH)
    return Path(override) if override else tracked_path()


def recording_target(environ: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """Where THIS session banks the `slow` verdicts it measures, or ``None`` = nowhere. **The one place that decides.**

    * `$GEN3AI_SLOW_STATUS_FILE` set -> that file (an explicit target is a deliberate act, and it never reaches the
      committed one);
    * else `$GEN3AI_SLOW_STATUS_BANK` set (and not ``0``) -> the committed file;
    * else ``None``: an ordinary session, however many slow tests it runs, leaves the tracked file byte-identical.

    ``environ`` is injectable so the table is pinned without a session (`slow_tier_status_bank_test.py` pins the real
    session too, which is what fails on a revert of the conftest's use of this).
    """
    env = os.environ if environ is None else environ
    explicit = env.get(_ENV_PATH)
    if explicit:
        return Path(explicit)
    if env.get(BANK_ENV, "") not in ("", "0"):
        return tracked_path()
    return None


# ──────────────────────────────────────────────────────────────── a measurement's identity ──

#: The branch a measurement's `base` is taken against. Shipped work only ever reaches it by a fast-forward, so a commit
#: reachable from it is on no branch the ship's rebase can rewrite.
INTEGRATION_BRANCH = "main"


def file_of(nodeid: str) -> str:
    """The repo-relative path of the file a node id lives in."""
    return nodeid.split("::", 1)[0]


def _git(cwd, *argv: str, stdin: Optional[str] = None) -> Optional[str]:
    """``git <argv>`` in ``cwd``: its stdout stripped, ``None`` on any failure (git absent, a nonzero exit, a timeout)."""
    try:
        r = subprocess.run(["git", *argv], cwd=str(cwd), input=stdin, capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def fork_point(cwd, commit: str) -> Optional[str]:
    """The commit of `INTEGRATION_BRANCH` that ``commit`` was built on (``git merge-base``), or ``None`` when git cannot say
    (no such branch, an unknown or garbage-collected commit). ``commit`` itself when it is already on the branch."""
    out = _git(cwd, "merge-base", commit, INTEGRATION_BRANCH)
    return out.splitlines()[0] if out else None


def measurement_base(cwd) -> str:
    """The `base` a row measured NOW, in ``cwd``, stamps (see the module docstring): the fork point of HEAD on main. A
    checkout with no main branch falls back to HEAD itself (the pre-2026-11-17 stamp, not rebase-safe); ``""`` when git
    cannot answer at all."""
    return fork_point(cwd, "HEAD") or _git(cwd, "rev-parse", "HEAD") or ""


def working_blobs(cwd, relpaths: "List[str]") -> Dict[str, str]:
    """``{relpath: git blob id}`` of those files AS THEY ARE ON DISK now, in ONE git call; a path that is not a file is
    simply absent from the answer."""
    present = [p for p in dict.fromkeys(relpaths) if (Path(cwd) / p).is_file()]
    if not present:
        return {}
    out = _git(cwd, "hash-object", "--stdin-paths", stdin="\n".join(present) + "\n")
    ids = out.splitlines() if out else []
    return dict(zip(present, ids)) if len(ids) == len(present) else {}


def blobs_at(cwd, specs: "List[Tuple[str, str]]") -> "List[Optional[str]]":
    """The blob id of ``path`` at ``commit`` for each ``(commit, path)``, in ONE git call; ``None`` where the commit or the
    path does not exist."""
    if not specs:
        return []
    out = _git(cwd, "cat-file", "--batch-check", stdin="".join(f"{c}:{p}\n" for c, p in specs))
    lines = out.splitlines() if out is not None else []
    if len(lines) != len(specs):
        return [None] * len(specs)
    return [ln.split()[0] if len(ln.split()) == 3 and ln.split()[1] == "blob" else None for ln in lines]


def git_distance(sha: str, cwd) -> Optional[int]:
    """How many commits HEAD is ahead of ``sha``; ``None`` when git does not know the commit OR ``sha`` is not an
    ancestor of HEAD.

    ``rev-list --count sha..HEAD`` alone is blind to the second case: for a side-branch commit it counts HEAD's commits
    missing from that branch and reads as "N behind", so a row banked at a commit that never reached main looked fresh
    (2026-11-09: 0 of 680 rows was a verified pass within 25 commits). ``evaluate`` reports ``None`` as stale."""
    try:
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=str(cwd),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if anc.returncode != 0:  # 1 = not an ancestor, 128 = unknown commit
            return None
        out = subprocess.check_output(["git", "rev-list", "--count", f"{sha}..HEAD"], cwd=str(cwd), text=True,
                                      stderr=subprocess.DEVNULL).strip()
        return int(out)
    except Exception:
        return None


def position(row: Dict[str, Any]) -> str:
    """The commit a row's staleness is counted from: its `base` (a commit of main, rebase-proof), else the `commit` an older
    writer stamped (``""`` when neither)."""
    return str(row.get("base") or row.get("commit") or "")


# ───────────────────────────────────────────────────────────────────── recording ──


#: The detail an `inconclusive` row carries when `settle` demoted it: the test never reported a
#: CALL phase, so nothing it asserts was measured.
INTERRUPTED_DETAIL = ("INTERRUPTED in flight: no call-phase report (Ctrl-C / KeyboardInterrupt, a "
                      "SIGTERM'd xdist worker, a killed session) — nothing it asserts was measured")


def settle(status: str, call_reported: bool) -> str:
    """The verdict a test may BANK, given every phase it reported. **`pass` needs a CALL phase.**

    🚨 A test's first report is its SETUP, and a setup that succeeds classifies `pass` — so a test
    that is interrupted mid-CALL is, to the fold above, indistinguishable from one that passed.
    Measured 2026-09-24: a MILESTONE run stopped with SIGTERM banked two in-flight tests as PASS
    with 0.0 s durations (their setup time). Two routes reach the bank with only that setup row: a
    KeyboardInterrupt, which pytest re-raises out of the call WITHOUT making a call report and
    then runs `pytest_sessionfinish`; and a SIGTERM'd xdist worker, whose crash report carries NO
    keywords (so it is not seen as `slow`) while the controller survives to session finish.

    **By class, not by signal:** whatever stopped it, a test with no CALL report did not run to
    completion, so it cannot be green. It is `inconclusive` — reported on every routine run, never
    fatal — rather than dropped, for the same reason a timeout is: the attempt happened and did not
    finish, and silently keeping an older green would hide that. A setup SKIP or setup ERROR has no
    call either and keeps its own class; only the unearned `pass` is demoted.
    """
    if status == "pass" and not call_reported:
        return "inconclusive"
    return status


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


@contextlib.contextmanager
def _exclusive(target: Path):
    """The exclusive `flock` every read-modify-write of the artifact takes. The lock lives in the system temp dir,
    keyed by the target's absolute path - NOT beside the artifact: a sidecar `.lock` in `designs/ops/` would be an
    untracked file the whole tree then has to remember to ignore, and the concurrency this guards (two xdist workers)
    is always same-box."""
    import fcntl

    target.parent.mkdir(parents=True, exist_ok=True)
    lock = Path(tempfile.gettempdir()) / (
        "gen3_slow_tier_status."
        + hashlib.sha256(str(target.resolve()).encode()).hexdigest()[:16] + ".lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _write_atomic(target: Path, doc: Dict[str, Any]) -> None:
    """Stamp `last_written` and commit with an atomic rename, so a crashed writer never leaves half an artifact."""
    doc["last_written"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".slow_status.")
    with os.fdopen(fd, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")
    os.replace(tmp, target)


def record_results(results: Dict[str, Dict[str, Any]], path: Path) -> Path:
    """MERGE ``{nodeid: row}`` into the status file ``path`` and return it. ``path`` is REQUIRED: a default of "the
    committed file" is how an ordinary session came to rewrite it - the caller names its target (`recording_target`).

    Merge, never replace: running one slow test must not erase the other rows, and two xdist
    workers finishing at the same moment must not lose each other's results. Per row, a SKIP never
    replaces a recorded pass / fail / inconclusive - it only annotates it (`merge_row`). The read-modify-write
    is done under an exclusive `flock` (`_exclusive`) and committed with an atomic rename.
    """
    target = Path(path)
    with _exclusive(target):
        doc = _read(target) if target.exists() else {"schema": SCHEMA, "tests": {}}
        doc.setdefault("schema", SCHEMA)
        tests = doc.setdefault("tests", {})
        for nodeid, row in results.items():
            tests[nodeid] = merge_row(tests.get(nodeid), row)
        doc["tests"] = dict(sorted(tests.items()))
        _write_atomic(target, doc)
    return target


def prunable(doc: Dict[str, Any], collected: "set[str]") -> List[str]:
    """The row ids whose test is no longer collected as `slow` (deleted, renamed, re-parametrised or un-marked).
    Pure. A row is keyed by test id and nothing else removes it (`Verdict.red_message`)."""
    return sorted(n for n in doc.get("tests", {}) if n not in collected)


def prune_rows(collected: "set[str]", path: Optional[Path] = None, *, write: bool = True) -> List[Tuple[str, str]]:
    """Delete every row whose id is not in ``collected`` (this session's FULL `slow` set); return ``[(id, status)]`` of
    the rows removed (or, with ``write=False``, that WOULD be). Under the same lock + atomic rename as the writer, so
    a tier run banking at the same moment loses nothing. Refuses an empty ``collected`` (a failed collection must
    never read as 'every test is gone')."""
    if not collected:
        raise ValueError("refusing to prune against an EMPTY collected set (a failed collection is not 'no slow tests')")
    target = Path(path) if path is not None else status_path()
    with _exclusive(target):
        doc = _read(target)
        gone = prunable(doc, collected)
        removed = [(n, str(doc["tests"][n].get("status"))) for n in gone]
        if write and gone:
            for n in gone:
                del doc["tests"][n]
            _write_atomic(target, doc)
    return removed


def orphaned(doc: Dict[str, Any], is_ancestor: "Callable[[str], bool]") -> List[Tuple[str, str]]:
    """``[(id, position)]`` of the rows whose `position` (`base`, else the old `commit` stamp) is NOT an ancestor of HEAD.
    Pure (the git question is ``is_ancestor``). A `base` is a commit of main, so this is what remains for a row an older
    writer stamped with a worktree commit the ship's rebase rewrote (`adopt_stamps` maps those) or whose commit git no
    longer has: the gate reports it STALE, and unlike a dead row it is never deleted - the test still exists, so only a
    re-measure (`GEN3AI_SLOW_STATUS_BANK=1`) replaces it, and deleting a red one would hide the red."""
    verdict: Dict[str, bool] = {}
    out = []
    for nodeid, row in sorted(doc.get("tests", {}).items()):
        sha = position(row)
        if sha not in verdict:
            verdict[sha] = bool(sha) and sha != "unknown" and is_ancestor(sha)
        if not verdict[sha]:
            out.append((nodeid, sha))
    return out


def adopt_stamps(doc: Dict[str, Any], *, fork: "Callable[[str], Optional[str]]",
                 blobs: "Callable[[List[Tuple[str, str]]], List[Optional[str]]]") -> Tuple[List[str], List[Tuple[str, str]]]:
    """Give every row that has NO `base` its `base` and `test_blob`, derived from the row's own `commit`: ``fork(commit)``
    (the commit of main it was built on) and ``blobs([(commit, test file)])`` (the file at that commit - the same
    clean-tree assumption the `commit` stamp always made). **Deterministic or nothing**: a row is stamped only when git
    answers BOTH questions; every other row is left byte-for-byte as it was and returned as ``(id, why)`` so the caller
    reports it - never re-stamped to a guess. Rows that already carry a `base` are never touched (so a partial stamp of
    the new writer is not overwritten with a derivation from a possibly dirty tree). Mutates ``doc`` in place; returns
    ``(adopted ids, [(unmapped id, why)])``."""
    todo = [(n, r) for n, r in sorted(doc.get("tests", {}).items()) if not r.get("base")]
    commits = {n: str(r.get("commit") or "") for n, r in todo}
    blob_of: Dict[str, Optional[str]] = {}
    known = [n for n, _ in todo if commits[n] not in ("", "unknown")]
    for n, b in zip(known, blobs([(commits[n], file_of(n)) for n in known])):
        blob_of[n] = b
    forks: Dict[str, Optional[str]] = {}
    adopted: List[str] = []
    unmapped: List[Tuple[str, str]] = []
    for nodeid, row in todo:
        c = commits[nodeid]
        if c in ("", "unknown"):
            unmapped.append((nodeid, "no commit stamp to derive from"))
            continue
        if c not in forks:
            forks[c] = fork(c)
        if not forks[c]:
            unmapped.append((nodeid, f"commit {c[:12]} is unknown to git (or has no fork point on {INTEGRATION_BRANCH})"))
        elif not blob_of.get(nodeid):
            unmapped.append((nodeid, f"{file_of(nodeid)} does not exist at commit {c[:12]}"))
        else:
            row["base"] = forks[c]
            row["test_blob"] = blob_of[nodeid]
            adopted.append(nodeid)
    return adopted, unmapped


def adopt_rows(path: Path, cwd, *, write: bool = True) -> Tuple[List[str], List[Tuple[str, str]]]:
    """`adopt_stamps` on the status file ``path`` against the git history of ``cwd``, under the writer's flock and atomic
    rename (a bank in flight loses nothing); with ``write=False`` it only reports what it WOULD adopt."""
    target = Path(path)
    with _exclusive(target):
        doc = _read(target)
        adopted, unmapped = adopt_stamps(doc, fork=lambda c: fork_point(cwd, c), blobs=lambda specs: blobs_at(cwd, specs))
        if write and adopted:
            _write_atomic(target, doc)
    return adopted, unmapped


#: The verdicts a later SKIP may NOT replace — everything that is a measurement. See `merge_row`.
_MEASURED = frozenset({"pass", "fail", "inconclusive"})


def merge_row(old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
    """The row to bank for one test, given the row already on file (``old``) and this run's.

    🚨 **A SKIP NEVER REPLACES A MEASUREMENT.** A skip means the test did not run — most often a
    GPU test run without ``GEN3AI_TEST_ALLOW_GPU`` — so it carries no information about whether the
    test passes. Replacing a banked PASS with it throws the measurement away; replacing a banked
    FAIL with it HIDES A RED from the routine gate, which is the one thing this artifact must never
    do. Measured 2026-09-29: one slow-tier run without the GPU flag overwrote seven banked GPU PASS
    rows with SKIP (restored by hand; Lane J's harness had guarded only its own path, F-LJ-5).

    So when ``new`` is a skip and ``old`` holds a measured verdict (pass / fail / inconclusive —
    an inconclusive row records an attempt that did not finish, which a skip does not supersede
    either), the OLD row is kept — its ``status``, ``commit`` and ``at`` untouched, so the
    staleness report still measures the age of the last real measurement — and it gains
    ``last_skipped_at`` / ``last_skipped_commit`` / ``last_skip_detail``, so the file still says
    the tier tried and could not run it. Any non-skip verdict replaces the row outright (dropping
    those notes); a skip over a skip, or over no row, is banked as itself.
    """
    if old is None or new.get("status") != "skip" or old.get("status") not in _MEASURED:
        return dict(new)
    kept = dict(old)
    kept["last_skipped_at"] = new.get("at", "")
    kept["last_skipped_commit"] = new.get("commit", "")
    kept["last_skip_detail"] = new.get("detail", "")
    return kept


def make_row(status: str, *, commit: str, duration_s: float,
             contention: float, detail: str = "", base: str = "", test_blob: str = "") -> Dict[str, Any]:
    """One test's row. `detail` is the first line of the failure, for the gate's message. `commit` is the provenance
    (the HEAD it ran at); `base` and `test_blob` are its IDENTITY (module docstring) and are written only when known."""
    row: Dict[str, Any] = {
        "status": status,
        "commit": commit,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": round(float(duration_s), 2),
        "contention": round(float(contention), 2),
        "detail": detail[:300],
    }
    if base:
        row["base"] = base
    if test_blob:
        row["test_blob"] = test_blob
    return row


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
    #: Measured rows with no `base` or no `test_blob` (an older writer): judged by their `commit` alone, so the content
    #: check and the rebase-proof position are missing. `slow_tier_prune --adopt-stamps` fills them where git can.
    unstamped: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_pass: int = 0
    n_skip: int = 0
    #: Rows whose banked verdict was kept over a later SKIP (they carry `last_skipped_at`).
    skipped_since: List[str] = field(default_factory=list)

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
                         + (f"(on main's {row['base'][:12]}) " if row.get("base") else "")
                         + f"on {row.get('at', '?')}  ({row.get('detail', '') or 'no detail'})")
        lines += [
            "",
            "Fix the test, then re-bank it (a session banks only when told to):",
            f"    export PYTHONPATH=$PYTHONPATH:src && {BANK_ENV}=1 python3 -m pytest <that test> -q",
            f"(a run of the test with {BANK_ENV}=1 updates its row; the full refresh is "
            f"`{BANK_ENV}=1 python3 -m pytest src/ -m slow -q -n 2`).",
            "If the test was RENAMED or DELETED, drop its row in the same commit: "
            "`python -m utils.slow_tier_prune --write` removes every row no longer collected as `slow`.",
        ]
        return "\n".join(lines)


def evaluate(doc: Dict[str, Any], collected: Optional[set] = None,
             commit_distance=None, test_blob_now=None) -> Verdict:
    """Turn a status document into a `Verdict`. Pure — the gate test's whole decision lives here.

    ``collected`` is the set of `slow` node ids THIS session collected (the root conftest records
    it before `-m` deselection runs, so the routine gate sees the full slow set even though it
    runs none of them). ``commit_distance(sha) -> Optional[int]`` answers "how many commits is HEAD
    ahead of this one", returning ``None`` when the commit is unknown or not an ancestor; it is asked about
    the row's `position` (its `base`). ``test_blob_now(relpath) -> Optional[str]`` answers "what is this test
    file's blob id now" (``None``: not a readable file): a row stamped with a `test_blob` that differs is stale
    however near its base is, because it measured a different test.
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
        if st != "skip" and row.get("last_skipped_at"):
            v.skipped_since.append(nodeid)
        if st != "skip" and not (row.get("base") and row.get("test_blob")):
            v.unstamped.append(nodeid)
        reasons: List[str] = []
        if test_blob_now is not None and row.get("test_blob"):
            now = test_blob_now(file_of(nodeid))
            if now is None:
                reasons.append(f"{file_of(nodeid)} is not a readable file now")
            elif now != row["test_blob"]:
                reasons.append(f"{file_of(nodeid)} changed since it was measured ({row['test_blob'][:8]} -> {now[:8]})")
        if commit_distance is not None:
            d = commit_distance(position(row))
            if d is None:
                reasons.append("commit unknown or not an ancestor of HEAD")
            elif d > STALE_COMMITS:
                reasons.append(f"{d} commits behind HEAD")
        if reasons:
            v.stale.append((nodeid, "; ".join(reasons)))
    if collected:
        v.unrecorded = sorted(n for n in collected if n not in tests)
    # A row that has only ever SKIPPED holds no measurement: it is reported with the unrecorded, so a
    # guard that keeps skipping can never read as a guard that passes (2026-10-01).
    v.unrecorded = sorted(set(v.unrecorded) | {n for n, r in tests.items() if r.get("status") == "skip"})
    return v
