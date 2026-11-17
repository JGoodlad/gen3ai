"""A slow-tier row's IDENTITY survives the ship's rebase, and a changed test still makes its row stale.

**The defect (2026-11-16).** A row stamped the WORKTREE commit it was measured at, and `/gen3ai-ship` rebases that commit
onto a moved main: the stamp became a SHA on no branch of main ("not an ancestor", STALE forever; 354 of 730 rows). Staleness
is a COMMIT COUNT, so the stamp must be a position that a rebase cannot move: `base` = the commit of main the measured code was
built on (`git merge-base HEAD main`), plus `test_blob` = the test file's blob id (a content identity that survives a rebase
when the file is unchanged, and differs when the test does). `commit` stays as provenance only.

Real git in `tmp_path` (a main repo, a real `git worktree`, a real `git rebase` onto a moved main, a real fast-forward), judged
by the production functions (`measurement_base`, `working_blobs`, `evaluate`, `git_distance`, `adopt_rows`); the session-level
proof that the conftest stamps them is `src/slow_tier_status_bank_test.py`.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import pytest

from utils import slow_tier_prune as P
from utils import slow_tier_status as ST

NEW = "src/new_test.py::test_added_in_the_worktree"        # a file the worktree adds
OLD = "src/probe_test.py::test_already_on_main"            # a file main already has


def git(cwd: Path, *argv: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
         *argv], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()


def commit_file(cwd: Path, rel: str, text: str, msg: str) -> str:
    (cwd / rel).parent.mkdir(parents=True, exist_ok=True)
    (cwd / rel).write_text(text)
    git(cwd, "add", "-A")
    git(cwd, "commit", "-q", "-m", msg)
    return git(cwd, "rev-parse", "HEAD")


def bank(cwd: Path, nodeid: str, status: str = "pass") -> dict:
    """The row a banking session in ``cwd`` writes: the conftest's own calls, in the conftest's own order."""
    blobs = ST.working_blobs(cwd, [ST.file_of(nodeid)])
    return ST.make_row(status, commit=git(cwd, "rev-parse", "HEAD"), duration_s=1, contention=1,
                       base=ST.measurement_base(cwd), test_blob=blobs.get(ST.file_of(nodeid), ""))


def judge(doc: dict, repo: Path) -> ST.Verdict:
    """What the gate does with a document, against the checkout ``repo``."""
    blobs = ST.working_blobs(repo, sorted({ST.file_of(n) for n in doc["tests"]}))
    return ST.evaluate(doc, commit_distance=lambda sha: ST.git_distance(sha, repo), test_blob_now=blobs.get)


@dataclass
class World:
    main: Path                  # the main checkout (branch `main`)
    wt: Path                    # a real `git worktree` on branch `work`
    m1: str                     # main's commit the worktree forked from
    w1: str                     # the worktree's commit, which the ship's rebase will rewrite
    rows: Dict[str, dict]       # the rows banked in the worktree at w1


@pytest.fixture
def world(tmp_path: Path) -> World:
    """main at m0..m1; a worktree on `work` (forked at m1) adds `src/new_test.py` as w1 and BANKS both rows there."""
    main = tmp_path / "main"
    main.mkdir()
    git(main, "init", "-q", "-b", "main")
    commit_file(main, "src/probe_test.py", "def test_already_on_main():\n    pass\n", "m0")
    m1 = commit_file(main, "other.txt", "one\n", "m1")
    wt = tmp_path / "wt"
    git(main, "worktree", "add", "-q", "-b", "work", str(wt), "main")
    w1 = commit_file(wt, "src/new_test.py", "def test_added_in_the_worktree():\n    pass\n", "w1")
    rows = {NEW: bank(wt, NEW), OLD: bank(wt, OLD)}
    return World(main=main, wt=wt, m1=m1, w1=w1, rows=rows)


def ship(world: World) -> str:
    """What /gen3ai-ship does: main MOVES, the worktree rebases onto it (w1 -> a new SHA), main fast-forwards. Returns w1'."""
    main, wt = world.main, world.wt
    commit_file(main, "other.txt", "two\n", "m2")
    commit_file(main, "other.txt", "three\n", "m3")
    git(wt, "rebase", "-q", "main")
    w1r = git(wt, "rev-parse", "HEAD")
    git(main, "merge", "-q", "--ff-only", "work")
    return w1r


def doc_of(world: World) -> dict:
    return {"tests": json.loads(json.dumps(world.rows))}


# ───────────────────────────────────────────────────────────── the rebase does not orphan a row ──


def test_a_row_banked_in_a_worktree_is_still_judged_correctly_after_the_ship_rebases_it_onto_a_moved_main(world: World) -> None:
    """THE DEFECT. Fails if `measurement_base` stamps the worktree HEAD (the row then cites w1, which the rebase rewrote:
    `git_distance` is None, stale) or if `evaluate` counts from `commit` instead of `base`."""
    main = world.main
    rows = world.rows
    assert rows[NEW]["base"] == world.m1 != rows[NEW]["commit"] == world.w1, \
        "a worktree measurement stamps main's fork point as its base, the worktree commit as provenance only"
    w1r = ship(world)
    assert w1r != world.w1, "the rebase must have rewritten the worktree commit (else this proves nothing)"
    assert ST.git_distance(world.w1, main) is None, "the OLD stamp is what the rebase orphaned"
    v = judge(doc_of(world), main)
    assert v.stale == [], f"a row banked in a worktree was orphaned by the ship's rebase: {v.stale}"
    assert v.unstamped == []
    assert ST.orphaned(doc_of(world), lambda sha: ST.git_distance(sha, main) is not None) == []
    # ...and the same document judged by the OLD stamp alone is what went stale in 354 rows:
    legacy = {n: {k: x for k, x in r.items() if k not in ("base", "test_blob")} for n, r in rows.items()}
    assert sorted(n for n, _ in judge({"tests": legacy}, main).stale) == sorted([NEW, OLD])


def test_a_row_measured_on_main_itself_has_main_s_head_as_its_base(world: World) -> None:
    main = world.main
    assert ST.measurement_base(main) == git(main, "rev-parse", "HEAD") == world.m1


def test_a_checkout_with_no_main_branch_falls_back_to_head(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "trunk")
    head = commit_file(repo, "a.txt", "a\n", "c0")
    assert ST.measurement_base(repo) == head, "no `main`: the old stamp, not an empty or a crashing one"
    assert ST.measurement_base(tmp_path / "not_a_directory") == "", "git cannot answer: nothing to stamp, and no exception"


# ─────────────────────────────────────────────────── a genuinely changed test makes its row stale ──


def test_a_changed_test_file_makes_its_row_stale_and_an_unrelated_commit_does_not(world: World) -> None:
    """Fails if `evaluate` ignores `test_blob`: the changed file's row is ONE commit behind its base, so only the content
    identity can say it measured a different test."""
    main = world.main
    ship(world)
    commit_file(main, "other.txt", "four\n", "m4, touches no test")
    v = judge(doc_of(world), main)
    assert v.stale == [], "a commit that touches no test file leaves both rows fresh"
    commit_file(main, "src/new_test.py", "def test_added_in_the_worktree():\n    assert 1 == 1\n", "m5, edits the test")
    v = judge(doc_of(world), main)
    assert [n for n, _ in v.stale] == [NEW], "only the test whose FILE changed goes stale"
    assert "src/new_test.py changed since it was measured" in v.stale[0][1] and "commits behind" not in v.stale[0][1]
    assert v.ok, "stale is reported, never fatal"


def test_a_row_whose_test_file_is_gone_is_stale_not_silently_fresh(world: World) -> None:
    main = world.main
    ship(world)
    git(main, "rm", "-q", "src/new_test.py")
    git(main, "commit", "-q", "-m", "m4, deletes the test")
    stale = dict(judge(doc_of(world), main).stale)
    assert "not a readable file" in stale[NEW] and OLD not in stale


def test_the_distance_is_counted_from_the_base_so_main_s_new_commits_count_as_unmeasured(world: World) -> None:
    """After the ship HEAD is 3 commits past the base (m2, m3, w1'); 23 more tip it over STALE_COMMITS. A row measured at
    the new tip is fresh."""
    main = world.main
    ship(world)
    for i in range(ST.STALE_COMMITS - 3):
        git(main, "commit", "-q", "--allow-empty", "-m", f"filler {i}")
    assert ST.git_distance(world.m1, main) == ST.STALE_COMMITS
    assert judge(doc_of(world), main).stale == []
    git(main, "commit", "-q", "--allow-empty", "-m", "one more")
    v = judge(doc_of(world), main)
    assert sorted(n for n, _ in v.stale) == sorted([NEW, OLD]) and f"{ST.STALE_COMMITS + 1} commits behind" in v.stale[0][1]
    fresh = {OLD: bank(main, OLD)}
    assert fresh[OLD]["base"] == git(main, "rev-parse", "HEAD") and judge({"tests": fresh}, main).stale == []


# ─────────────────────────────────────────────────────────────────────── evaluate, pure ──


def test_a_row_without_an_identity_stamp_is_reported_unstamped_and_judged_by_its_commit() -> None:
    legacy = {"status": "pass", "commit": "c" * 40, "at": "2026-11-01T00:00:00Z"}
    stamped = dict(legacy, base="b" * 40, test_blob="d" * 40)
    asked = []

    def distance(sha: str):
        asked.append(sha)
        return 0

    v = ST.evaluate({"tests": {"a_test.py::t1": legacy, "b_test.py::t2": stamped, "c_test.py::t3": dict(legacy, status="skip")}},
                    commit_distance=distance, test_blob_now=lambda f: "d" * 40)
    assert v.unstamped == ["a_test.py::t1"], "a skip-only row holds no measurement, so it is not 'unstamped'"
    assert sorted(set(asked)) == ["b" * 40, "c" * 40], "the distance is asked about the BASE when there is one, the commit otherwise"
    assert ST.position(stamped) == "b" * 40 and ST.position(legacy) == "c" * 40 and ST.position({}) == ""


def test_orphaned_judges_the_base_not_the_dead_worktree_commit() -> None:
    doc = {"tests": {"a::t": {"commit": "dead", "base": "good"}, "b::t": {"commit": "dead"}}}
    assert ST.orphaned(doc, lambda sha: sha == "good") == [("b::t", "dead")]


# ─────────────────────────────────────────────────── migrating the rows an older writer stamped ──


def legacy_doc(world: World) -> dict:
    """The rows as the OLD writer left them: only `commit`, a worktree SHA the rebase has since orphaned."""
    return {"schema": ST.SCHEMA, "tests": {
        NEW: {"status": "pass", "commit": world.w1, "at": "2026-11-01T00:00:00Z", "detail": ""},
        OLD: {"status": "pass", "commit": world.m1, "at": "2026-11-01T00:00:00Z", "detail": ""},
        "src/gone_test.py::t_commit_gc_d": {"status": "pass", "commit": "0" * 40, "at": "2026-11-01T00:00:00Z", "detail": ""},
        "src/late_test.py::t_not_yet_there": {"status": "fail", "commit": world.m1, "at": "2026-11-01T00:00:00Z", "detail": "x"},
        "src/nocommit_test.py::t": {"status": "pass", "commit": "unknown", "at": "2026-11-01T00:00:00Z", "detail": ""},
        "src/done_test.py::t_already_stamped": {"status": "pass", "commit": "e" * 40, "base": "f" * 40, "test_blob": "a" * 40,
                                                "at": "2026-11-01T00:00:00Z", "detail": ""},
    }}


def test_adoption_maps_what_git_can_answer_deterministically_and_reports_the_rest_untouched(world: World, tmp_path: Path) -> None:
    main = world.main
    ship(world)
    status = tmp_path / "status.json"
    status.write_text(json.dumps(legacy_doc(world), indent=1) + "\n")
    before = json.loads(status.read_text())["tests"]
    assert ST.git_distance(world.w1, main) is None, "the worktree commit the rebase rewrote: the orphan to map"

    adopted, unmapped = ST.adopt_rows(status, main, write=False)
    assert status.read_text() == json.dumps(legacy_doc(world), indent=1) + "\n", "check mode writes nothing"
    assert sorted(adopted) == sorted([NEW, OLD])
    assert sorted(n for n, _ in unmapped) == sorted(["src/gone_test.py::t_commit_gc_d", "src/late_test.py::t_not_yet_there",
                                                     "src/nocommit_test.py::t"])

    ST.adopt_rows(status, main, write=True)
    after = json.loads(status.read_text())["tests"]
    new_blob = git(main, "rev-parse", f"{world.w1}:src/new_test.py")
    assert after[NEW]["base"] == world.m1, "the orphan maps to the commit of MAIN it was built on, never to itself"
    assert after[NEW]["test_blob"] == new_blob == git(main, "rev-parse", "HEAD:src/new_test.py"), \
        "the rebased file carries the same bytes, so the content identity matches today's"
    assert after[NEW]["commit"] == world.w1, "`commit` stays as provenance, unrewritten"
    assert after[OLD]["base"] == world.m1
    for n in set(before) - {NEW, OLD}:
        assert after[n] == before[n], f"{n}: a row git cannot map must stay byte-for-byte, never re-stamped to a guess"
    assert judge({"tests": {n: after[n] for n in (NEW, OLD)}}, main).stale == []
    assert json.loads(status.read_text())["tests"][NEW] == after[NEW]
    assert ST.adopt_rows(status, main, write=True)[0] == [], "idempotent: a stamped row is never touched again"


def test_adoption_never_stamps_an_unshipped_worktree_commit_as_the_base(world: World, tmp_path: Path) -> None:
    """Before the ship, w1 is on `work` only. The base must still be a commit of MAIN, or the rebase would orphan the new
    stamp too. (Fails if the fork point is taken against HEAD of a worktree that carries unshipped commits.)"""
    status = tmp_path / "status.json"
    status.write_text(json.dumps(legacy_doc(world)) + "\n")
    ST.adopt_rows(status, world.wt, write=True)
    base = json.loads(status.read_text())["tests"][NEW]["base"]
    assert base == world.m1 != world.w1
    assert git(world.main, "merge-base", "--is-ancestor", base, "main") == ""


def test_the_prune_cli_adopts_in_check_mode_then_with_write_and_lists_what_it_cannot_map(world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main = world.main
    ship(world)
    status = tmp_path / "status.json"
    status.write_text(json.dumps(legacy_doc(world), indent=1) + "\n")
    before = status.read_bytes()
    never = lambda: pytest.fail("--adopt-stamps must not run the pytest collection")           # noqa: E731
    assert P.main(["--adopt-stamps"], collect=never, path=status, git_cwd=main) == 0
    out = capsys.readouterr().out
    assert status.read_bytes() == before and "WOULD stamp 2 row(s)" in out and "3 row(s) git cannot map" in out
    assert "is unknown to git" in out and "does not exist at commit" in out and "no commit stamp" in out
    assert P.main(["--adopt-stamps", "--write"], collect=never, path=status, git_cwd=main) == 0
    assert "stamped 2 row(s)" in capsys.readouterr().out
    assert json.loads(status.read_text())["tests"][NEW]["base"] == world.m1
