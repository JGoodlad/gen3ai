"""`utils.xdist_schedule` — the routine gate's cost-ordered `--dist loadfile` schedule.

The unit half pins the table (merge, prune, corrupt-is-empty) and the ordering rule. The end-to-end
half copies the root `conftest.py` and `pytest.ini` VERBATIM beside planted files and runs a real
`-n 2` child session: the file the planted table calls expensive is collected LAST, and must START
first — and with `GEN3AI_COST_SCHEDULE=0` it must not, so a dead hook fails here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from collections import OrderedDict

import pytest

from utils import xdist_schedule as XS
from utils.paths import repo_path, src_root


def test_a_missing_or_corrupt_table_is_empty_never_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(XS, "NOTES", [])        # these notes are planted; keep them out of the summary
    assert XS.load_costs(tmp_path / "absent.json") == {}
    (tmp_path / "bad.json").write_text("{not json")
    assert XS.load_costs(tmp_path / "bad.json") == {}
    (tmp_path / "old.json").write_text(json.dumps({"schema": "something_else", "costs": {"a": [1, 0]}}))
    assert XS.load_costs(tmp_path / "old.json") == {}


def test_record_merges_replaces_and_prunes(tmp_path):
    p = tmp_path / "costs.json"
    day = 86400.0
    XS.record_costs({"a::t": 1.0, "b::t": 2.0}, p, now=100 * day)
    XS.record_costs({"b::t": 5.0}, p, now=101 * day)
    assert XS.load_costs(p) == {"a::t": 1.0, "b::t": 5.0}
    XS.record_costs({"c::t": 0.5}, p, now=(101 + XS.PRUNE_DAYS) * day)   # a::t is PRUNE_DAYS+1 old
    assert XS.load_costs(p) == {"b::t": 5.0, "c::t": 0.5}


def test_units_are_ordered_most_expensive_first_and_ties_keep_collection_order():
    wq = OrderedDict([("a.py", {"a.py::x": False, "a.py::y": False}),
                      ("b.py", {"b.py::x": False}),
                      ("c.py", {"c.py::x": False}),
                      ("d.py", {"d.py::x": False})])
    costs = {"b.py::x": 9.0, "a.py::x": 1.0, "a.py::y": 1.0}
    # c and d are unrecorded: DEFAULT_COST_S each, a tie, so collection order holds between them
    assert list(XS.order_units(wq, costs)) == ["b.py", "a.py", "c.py", "d.py"]


_PLANTED = {
    "a_cheap_test.py": """
        import os, time
        def _stamp(name):
            with open(os.environ["STAMPS"], "a") as f:
                f.write(f"{time.monotonic():.6f} {name}\\n")
        def test_a1(): _stamp("a1"); time.sleep(0.3)
        def test_a2(): _stamp("a2"); time.sleep(0.3)
        def test_a3(): _stamp("a3"); time.sleep(0.3)
        def test_a4(): _stamp("a4"); time.sleep(0.3)
    """,
    "m_cheap_test.py": """
        import os, time
        def _stamp(name):
            with open(os.environ["STAMPS"], "a") as f:
                f.write(f"{time.monotonic():.6f} {name}\\n")
        def test_m1(): _stamp("m1"); time.sleep(0.3)
        def test_m2(): _stamp("m2"); time.sleep(0.3)
    """,
    "z_costly_test.py": """
        import os, time
        def _stamp(name):
            with open(os.environ["STAMPS"], "a") as f:
                f.write(f"{time.monotonic():.6f} {name}\\n")
        def test_z1(): _stamp("z1"); time.sleep(0.3)
    """,
}


def _child_session(tmp_path, *, schedule_on: bool = True, table_state: str = "good"):
    tag = f"{'on' if schedule_on else 'off'}_{table_state}"
    d = tmp_path / tag
    d.mkdir()
    (d / "conftest.py").write_text(repo_path("conftest.py").read_text())
    ini = repo_path("pytest.ini").read_text().replace("testpaths = src tools", "testpaths = .")
    (d / "pytest.ini").write_text(ini)
    for name, body in _PLANTED.items():
        (d / name).write_text(textwrap.dedent(body))
    table = tmp_path / f"costs_{tag}.json"
    if table_state == "good":
        XS.record_costs({"z_costly_test.py::test_z1": 500.0, "a_cheap_test.py::test_a1": 0.01}, table)
    elif table_state == "stale":          # names only tests that no longer exist
        XS.record_costs({"gone_test.py::test_x": 500.0, "z_costly_test.py::test_removed": 9.0}, table)
    elif table_state == "corrupt":
        table.write_text('{"schema": "gen3_test_costs_v1", "costs": {"z_costly_test.py::test_z1": "nan')
    else:
        assert table_state == "missing"
    before = table.read_text() if table.exists() else None
    stamps = d / "stamps.txt"
    env = dict(os.environ, STAMPS=str(stamps), GEN3AI_TEST_COSTS=str(table))
    env["PYTHONPATH"] = os.pathsep.join([str(src_root()), env.get("PYTHONPATH", "")])
    for k in ("GEN3AI_SKIP_DEPS_GUARD", "GEN3AI_SKIP_SLOW_STATUS_RECORD", "GEN3AI_SKIP_TIER_BUDGET"):
        env[k] = "1"
    env.pop("PYTEST_XDIST_WORKER", None)
    if schedule_on:
        env.pop("GEN3AI_COST_SCHEDULE", None)
    else:
        env["GEN3AI_COST_SCHEDULE"] = "0"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--color=no",
                           "-n", "2", str(d)], cwd=str(d), env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-6000:]
    assert "7 passed" in out, out[-3000:]
    order = [ln.split()[1] for ln in sorted(stamps.read_text().splitlines(), key=lambda ln: float(ln.split()[0]))]
    assert sorted(order) == ["a1", "a2", "a3", "a4", "m1", "m2", "z1"], order     # every test ran, once
    after = table.read_text() if table.exists() else None
    return order, before, after, out


@pytest.mark.integration
def test_the_expensive_file_starts_first_and_a_planted_session_teaches_the_table_nothing(tmp_path):
    order, before, after, _ = _child_session(tmp_path)
    assert order.index("z1") < 2, order          # among the two FIRST starts (one per worker)
    # the planted session runs from a temp dir, not this repo: its durations are NOT recorded
    assert after == before
    # a FILE is the unit: a file's tests run in collection order on ONE worker, back to back
    assert [t for t in order if t.startswith("a")] == ["a1", "a2", "a3", "a4"]


@pytest.mark.integration
@pytest.mark.parametrize("table_state", ["missing", "corrupt", "stale"])
def test_the_table_only_orders_work_every_test_runs_whatever_its_state(tmp_path, table_state):
    """ORDER and placement only, never WHICH tests run: a missing, corrupt or stale table runs the
    identical set (all 7, each once — asserted in `_child_session`) and only says so."""
    _order, _before, _after, out = _child_session(tmp_path, table_state=table_state)
    if table_state == "corrupt":
        assert "UNREADABLE" in out and "IGNORED" in out, out[-3000:]
    if table_state == "missing":
        assert "not found" in out, out[-3000:]


@pytest.mark.integration
def test_with_the_schedule_off_collection_order_wins(tmp_path):
    """The revert check: GEN3AI_COST_SCHEDULE=0 is plain loadfile, which hands out files by test COUNT
    (a: 4, m: 2, z: 1), two units per worker — z queues behind m's two tests, so it cannot start in
    the first wave."""
    order, _, _, _ = _child_session(tmp_path, schedule_on=False)
    assert order.index("z1") >= 2, order


def test_a_corrupt_table_is_reported_not_silently_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(XS, "NOTES", [])
    (tmp_path / "bad.json").write_text("{not json")
    assert XS.load_costs(tmp_path / "bad.json") == {}
    assert len(XS.NOTES) == 1 and "UNREADABLE" in XS.NOTES[0]


_CRASHER = """
    import os, signal
    def _stamp(name):
        with open(os.environ["STAMPS"], "a") as f:
            f.write(name + "\\n")
    def test_c1_kills_its_worker():
        _stamp("c1")
        os.kill(os.getpid(), signal.SIGKILL)
    def test_c2_after_the_crash():
        _stamp("c2")
"""


@pytest.mark.integration
def test_a_test_that_kills_its_worker_is_reported_once_and_never_rerun(tmp_path):
    """Upstream `loadscope`/`loadfile` re-queue a crashed unit WITH the crashing test still pending, so a
    restarted worker runs it again (and again, until the restart budget is spent). The schedule must
    report it once, as `load` does, and still run the rest of its file."""
    d = tmp_path / "crash"
    d.mkdir()
    (d / "conftest.py").write_text(repo_path("conftest.py").read_text())
    (d / "pytest.ini").write_text(repo_path("pytest.ini").read_text().replace("testpaths = src tools",
                                                                             "testpaths = ."))
    (d / "crash_test.py").write_text(textwrap.dedent(_CRASHER))
    stamps = d / "stamps.txt"
    env = dict(os.environ, STAMPS=str(stamps), GEN3AI_TEST_COSTS=str(tmp_path / "costs.json"))
    env["PYTHONPATH"] = os.pathsep.join([str(src_root()), env.get("PYTHONPATH", "")])
    for k in ("GEN3AI_SKIP_DEPS_GUARD", "GEN3AI_SKIP_SLOW_STATUS_RECORD", "GEN3AI_SKIP_TIER_BUDGET"):
        env[k] = "1"
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop("GEN3AI_COST_SCHEDULE", None)
    proc = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--color=no",
                           "-n", "2", str(d)], cwd=str(d), env=env, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 1, out[-4000:]
    assert stamps.read_text().split() == ["c1", "c2"], (stamps.read_text(), out[-4000:])
    assert "1 failed, 1 passed" in out, out[-3000:]
