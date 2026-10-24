"""The SHORT LANE of ``utils.heavy_slots`` / ``utils.mem_cap`` (2026-10-23), against REAL processes and never the real slots.

The problem it fixes (measured 10-20..23): jobs of a few minutes waited 1-2 h behind 40-45-min research chunks in the
2-slot FIFO (the D1 corpus soak: 4 min of work after 53 min of waiting). A job that declares itself short
(``mem_cap.sh --max-s S``, S <= 600) may jump ahead of queued long jobs, at most ``SHORT_JUMPS`` times per long job, and
is hard-killed past its bound. Load-bearing pins, each FAILS on revert of what it guards:

* ORDERING: ``grant_order`` is FIFO among shorts, FIFO among longs, with up to K shorts before each queued long; with no
  short ticket it is the plain FIFO of the code before the lane;
* THE STARVATION BOUND: the pass-over count lives in the long ticket's own file; after K shorts were granted ahead of a
  queued long job, it goes next however many shorts are queued, and the next long head starts with a fresh budget;
* A DECLARED-SHORT JOB REALLY BYPASSES QUEUED LONG ONES: real waiters, one slot, the long one first in line - the shorts run
  first, exactly K of them, then the long job (and with no declaration the same arrivals run in plain FIFO order);
* THE DEADLINE binds only a job that TOOK a slot of its own (not a bypass, a re-entrant child of a holder, or a job that needed
  no slot) and is refused above 600 s;
* an old-code process cannot parse (so cannot miscount) a short ticket;
* the release log, the status lines and the declaration's validation.

The overrun KILL itself (a real scope) is ``mem_cap_test.py``. ``integration``: spawns and kills its own processes, by PID.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from utils import heavy_slots as HS
from utils import mem_cap as M
from utils import procfs
from utils.contention import scale_timeout
from utils.paths import src_root

pytestmark = pytest.mark.integration

# a waiter that records WHEN it got the slot (ns), holds briefly, then leaves; argv: marker, declared max_s (0 = long)
_LANE_TAKER = ("import pathlib, sys, time\n"
               "from utils.heavy_slots import heavy_slot\n"
               "max_s = float(sys.argv[2]) or None\n"
               "with heavy_slot(['w', sys.argv[1]], force=True, poll_s=0.02, log=lambda m: None, max_s=max_s):\n"
               "    pathlib.Path(sys.argv[1]).write_text(str(time.time_ns())); time.sleep(0.15)\n")
_HOLDER = ("import pathlib, sys, time\n"
           "from utils.heavy_slots import heavy_slot\n"
           "with heavy_slot(['holder'], force=True, poll_s=0.02, log=lambda m: None):\n"
           "    pathlib.Path(sys.argv[1]).write_text('held'); time.sleep(120)\n")


@pytest.fixture
def slots(tmp_path, monkeypatch):
    d = tmp_path / "heavy"
    monkeypatch.setenv(HS.DIR_ENV, str(d))
    monkeypatch.setenv(HS.SLOTS_ENV, "1")
    for k in (HS.HELD_ENV, HS.BLOCKING_ENV, HS.MAX_S_ENV, HS.SHORT_JUMPS_ENV, HS.SHORT_MARGIN_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEN3AI_GPU_LOCK", str(tmp_path / "gpu.lock"))     # never the box's real lease
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(src_root()), os.environ.get("PYTHONPATH", "")]))
    return d


def _wait_for(pred, what: str, timeout: float = 20.0) -> None:
    end = time.monotonic() + scale_timeout(timeout)
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _kill(*procs) -> None:
    for p in procs:
        if p is not None:
            if p.poll() is None:
                os.kill(p.pid, signal.SIGKILL)
            p.wait()


def _sleepers(n: int):
    return [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]) for _ in range(n)]


def _names(order, who: dict) -> list:
    return [who[t.pid] for t in order]


# ── ORDERING (no timing: tickets of real sleeping pids) ───────────────────────────────────────────────────────────

def test_grant_order_is_fifo_per_lane_with_K_shorts_before_each_queued_long(slots):
    """Arrivals L1 S1 S2 L2 S3 S4 S5 S6: the shorts pass up to K = 3 before EACH long ticket, FIFO among themselves, and the
    longs keep their own FIFO. Fails on a revert to one FIFO (it would be L1 S1 S2 L2 S3 ...)."""
    procs = _sleepers(8)
    try:
        arrivals = [("L1", HS.LANE_LONG), ("S1", HS.LANE_SHORT), ("S2", HS.LANE_SHORT), ("L2", HS.LANE_LONG),
                    ("S3", HS.LANE_SHORT), ("S4", HS.LANE_SHORT), ("S5", HS.LANE_SHORT), ("S6", HS.LANE_SHORT)]
        who = {}
        for (name, lane), p in zip(arrivals, procs):
            HS.take_ticket(slots, p.pid, lane)
            who[p.pid] = name
        assert HS.short_jumps() == HS.SHORT_JUMPS == 3
        got = _names(HS.grant_order(slots, -1), who)
        assert got == ["S1", "S2", "S3", "L1", "S4", "S5", "S6", "L2"], got
        # the ticket's name carries its lane; a legacy (3-part) name is a long ticket
        assert [HS.lane_of(p) for _, _, p in HS._tickets(slots)] == [lane for _, lane in arrivals]
    finally:
        _kill(*procs)


def test_with_no_short_ticket_the_order_is_the_plain_fifo_and_with_no_long_it_is_the_short_fifo(slots):
    procs = _sleepers(4)
    try:
        who = {p.pid: f"T{i}" for i, p in enumerate(procs)}
        for p in procs:
            HS.take_ticket(slots, p.pid)                                     # all long (the default, and every gate's)
        assert _names(HS.grant_order(slots, -1), who) == ["T0", "T1", "T2", "T3"]
        for p in procs:
            HS.drop_ticket(slots, p.pid)
        for p in procs:
            HS.take_ticket(slots, p.pid, HS.LANE_SHORT)
        assert _names(HS.grant_order(slots, -1), who) == ["T0", "T1", "T2", "T3"]
    finally:
        _kill(*procs)


def test_the_starvation_bound_a_queued_long_job_is_passed_over_exactly_K_times(slots):
    """L1 queued, five shorts behind it. Each short grant is charged to L1's own ticket file; after K = 3, L1 is FIRST
    however many shorts are still queued, and the next long head (L2) starts with a fresh budget. Fails on a revert of
    ``note_short_grant`` (L1 would never reach the front) or of the budget (shorts would pass L1 for ever)."""
    procs = _sleepers(7)
    try:
        l1, l2, shorts = procs[0], procs[1], procs[2:]
        who = {l1.pid: "L1", l2.pid: "L2", **{p.pid: f"S{i + 1}" for i, p in enumerate(shorts)}}
        HS.take_ticket(slots, l1.pid)
        for p in shorts:
            HS.take_ticket(slots, p.pid, HS.LANE_SHORT)
        l1_ticket = next(path for _, pid, path in HS._tickets(slots) if pid == l1.pid)
        assert HS.jumped(l1_ticket) == 0                                    # an empty (legacy) ticket reads as 0
        for k in (1, 2, 3):
            assert _names(HS.grant_order(slots, -1), who)[0] == f"S{k}", "a short must lead while the budget lasts"
            assert HS.note_short_grant(slots) == k                          # S<k> took the slot ...
            HS.drop_ticket(slots, shorts[k - 1].pid)                        # ... and left the queue
        assert HS.jumped(l1_ticket) == 3
        assert _names(HS.grant_order(slots, -1), who) == ["L1", "S4", "S5"], "L1 was passed over more than K times"
        HS.take_ticket(slots, l2.pid)                                       # a second long job queues behind L1
        assert _names(HS.grant_order(slots, -1), who) == ["L1", "S4", "S5", "L2"]
        HS.drop_ticket(slots, l1.pid)                                       # L1 got its slot: L2's budget is fresh
        assert _names(HS.grant_order(slots, -1), who) == ["S4", "S5", "L2"]
        assert HS.note_short_grant(slots) == 1 and HS.jumped(
            next(path for _, pid, path in HS._tickets(slots) if pid == l2.pid)) == 1
    finally:
        _kill(*procs)


def test_a_short_grant_with_no_polling_long_charges_nobody_and_never_recreates_a_ticket(slots, monkeypatch):
    procs = _sleepers(2)
    try:
        long_p, short_p = procs
        HS.take_ticket(slots, short_p.pid, HS.LANE_SHORT)
        assert HS.note_short_grant(slots) is None                           # nobody to pass over
        t = HS.take_ticket(slots, long_p.pid)
        now = time.time() + HS.TICKET_STALE_S + 5                           # the long owner is silent: not waiting NOW
        assert HS.note_short_grant(slots, now=now) is None and HS.jumped(t) == 0
        # a ticket whose owner left between the read and the write is NOT recreated (a ghost would block the queue)
        gone = slots / "ticket-000000000099-1.tk"
        monkeypatch.setattr(HS, "_lanes", lambda *a, **k: ([HS.Ticket(99, 1, HS.LANE_LONG, gone)], []))
        assert HS.note_short_grant(slots) is None and not gone.exists()
    finally:
        _kill(*procs)


def test_a_silent_long_ticket_does_not_hold_the_shorts_back_but_keeps_its_place(slots):
    procs = _sleepers(3)
    try:
        l1, s1, s2 = procs
        who = {l1.pid: "L1", s1.pid: "S1", s2.pid: "S2"}
        t = HS.take_ticket(slots, l1.pid)
        HS.take_ticket(slots, s1.pid, HS.LANE_SHORT)
        HS.take_ticket(slots, s2.pid, HS.LANE_SHORT)
        later = time.time() + HS.TICKET_STALE_S + 5
        for p in (s1, s2):                                                  # the shorts keep polling, L1's owner does not
            for _, pid, path in HS._tickets(slots):
                if pid == p.pid:
                    os.utime(path, (later, later))
        assert _names(HS.grant_order(slots, -1, now=later), who) == ["S1", "S2"]
        assert HS.queue_ahead(slots, s1.pid, now=later) == 0
        assert HS.queue_ahead(slots, l1.pid, now=later) == 2, "it keeps its place - behind the shorts that may pass it"
        assert t.exists()                                                   # it is skipped, not dropped
    finally:
        _kill(*procs)


def test_an_old_code_process_cannot_parse_a_short_ticket(slots):
    """Processes started before the lane keep their code: their `_tickets` unpacks ``stem.split('-')`` into THREE parts and
    skips a name it cannot, so they neither see nor miscount a short ticket (the flock stays the only lock). Fails if a
    short ticket's name is ever changed to the legacy three-part shape."""
    [p] = _sleepers(1)
    try:
        short = HS.take_ticket(slots, p.pid, HS.LANE_SHORT)
        with pytest.raises(ValueError):
            _, _seq, _pid = short.stem.split("-")                          # the legacy parser's exact line
        HS.drop_ticket(slots, p.pid)
        long_ = HS.take_ticket(slots, p.pid)
        _, seq_s, pid_s = long_.stem.split("-")                            # ... which still parses a long ticket
        assert int(pid_s) == p.pid and int(seq_s) >= 0
    finally:
        _kill(p)


# ── A DECLARED-SHORT JOB REALLY BYPASSES QUEUED LONG ONES (real waiters, real flock) ───────────────────────────────

@pytest.mark.parametrize("declared", [True, False])
def test_declared_short_jobs_run_ahead_of_a_queued_long_one_exactly_K_times(slots, tmp_path, declared):
    """ONE slot, held. Arrivals: LONG, then five jobs S1..S5. Declared short: S1 S2 S3 run, then LONG, then S4 S5 - the long
    job waited behind exactly K = 3. Declared nothing (control): plain FIFO, LONG first. Fails on a revert of the lane
    (the declared run would be LONG first too) and of the budget (the declared run would be S1..S5 then LONG)."""
    holder_file = tmp_path / "holder"
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(holder_file)])
    procs = []
    try:
        _wait_for(holder_file.exists, "the holder")
        names = ["LONG", "S1", "S2", "S3", "S4", "S5"]
        for k, name in enumerate(names):
            max_s = "60" if (declared and name != "LONG") else "0"
            procs.append(subprocess.Popen([sys.executable, "-c", _LANE_TAKER, str(tmp_path / name), max_s]))
            _wait_for(lambda: len(HS._tickets(slots)) == k + 1, f"{name}'s ticket")       # arrival order is fixed
        _kill(holder)
        for p in procs:
            assert p.wait(timeout=scale_timeout(60)) == 0
        stamps = {n: int((tmp_path / n).read_text()) for n in names}
        ran = sorted(names, key=lambda n: stamps[n])
        want = ["S1", "S2", "S3", "LONG", "S4", "S5"] if declared else ["LONG", "S1", "S2", "S3", "S4", "S5"]
        assert ran == want, ran
        assert not HS._tickets(slots), "a finished waiter's ticket outlived it"
    finally:
        _kill(holder, *procs)


# ── the DEADLINE binds only a job that took a slot of its own ─────────────────────────────────────────────────────

class _FakeRun:
    """Stands in for ``run_capped`` (no systemd): records what ``main`` would have enforced."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, gb, **kw):
        self.calls.append(kw)
        return M.CapResult(0, "u", False, None, 0)


@pytest.fixture
def fake_run(monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(M, "run_capped", fake)
    return fake


def test_a_declared_short_job_that_takes_a_slot_runs_under_declared_plus_margin(slots, fake_run, monkeypatch):
    assert M.main(["--slot", "--max-s", "100", "1", "true"]) == 0
    [kw] = fake_run.calls
    assert kw["deadline_s"] == 100 + HS.SHORT_MARGIN_S and kw["declared_s"] == 100 and kw["slot_fd"] is not None
    monkeypatch.setenv(HS.SHORT_MARGIN_ENV, "1")                           # the margin can only narrow
    assert M.main(["--slot", "--max-s", "100", "1", "true"]) == 0
    assert fake_run.calls[-1]["deadline_s"] == 101
    monkeypatch.setenv(HS.SHORT_MARGIN_ENV, "500")
    assert M.main(["--slot", "--max-s", "100", "1", "true"]) == 0
    assert fake_run.calls[-1]["deadline_s"] == 100 + HS.SHORT_MARGIN_S, "the env widened the margin"
    # the ambient declaration is the same thing as the flag; the flag wins
    monkeypatch.setenv(HS.MAX_S_ENV, "50")
    assert M.main(["--slot", "1", "true"]) == 0 and fake_run.calls[-1]["declared_s"] == 50
    assert M.main(["--slot", "--max-s", "70", "1", "true"]) == 0 and fake_run.calls[-1]["declared_s"] == 70


def test_a_job_with_no_declaration_has_no_deadline(slots, fake_run):
    assert M.main(["--slot", "1", "true"]) == 0
    assert fake_run.calls[-1]["deadline_s"] is None and fake_run.calls[-1]["declared_s"] is None


def test_a_declared_job_that_needs_no_slot_or_bypasses_is_never_killed(slots, fake_run, monkeypatch):
    """The deadline is the price of the lane: no slot taken, no deadline. An ambient ``$GEN3AI_HEAVY_SLOT_MAX_S`` must not be
    able to cut a job that never queued (no run live, no lease) or one the orchestrator marked BLOCKING."""
    monkeypatch.setattr(procfs, "live_training_runs", lambda *a, **k: {})
    monkeypatch.setattr(HS, "lease_reason", lambda: None)
    assert M.main(["--max-s", "100", "1", "true"]) == 0                    # nothing wants a slot
    assert fake_run.calls[-1]["slot_fd"] is None and fake_run.calls[-1]["deadline_s"] is None
    monkeypatch.setattr(procfs, "live_training_runs", lambda *a, **k: {"/m/rb_x": 1})
    monkeypatch.setenv(HS.BLOCKING_ENV, "plateau tick")
    assert M.main(["--max-s", "100", "1", "true"]) == 0                    # a run is live, but BLOCKING bypasses
    assert fake_run.calls[-1]["slot_fd"] is None and fake_run.calls[-1]["deadline_s"] is None


def test_a_declared_job_inside_a_slot_holder_is_re_entrant_and_gets_no_deadline(slots, tmp_path):
    """A child of a holder (the ship gate's mem_cap) takes no second slot - and so must not be killed at 10 min by an ambient
    declaration. A REAL child process of a REAL holder (this test)."""
    child = ("import sys\nfrom utils import mem_cap as M\n"
             "M.run_capped = lambda c, gb, **kw: (print('KW', kw['slot_fd'], kw['deadline_s']), M.CapResult(0, 'u', False, None, 0))[1]\n"
             "sys.exit(M.main(['--slot', '--max-s', '100', '1', 'true']))\n")
    with HS.heavy_slot(["the holder"], force=True, log=lambda _m: None):
        assert HS.verified_holder() is not None
        r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=scale_timeout(60))
    assert r.returncode == 0 and "KW None None" in r.stdout, (r.returncode, r.stdout, r.stderr)


@pytest.mark.parametrize("args,env,needle", [
    (["--max-s", "601", "1", "true"], {}, "at most 600"),
    (["--max-s", "0", "1", "true"], {}, "--max-s"),
    (["--max-s", "-5", "1", "true"], {}, "--max-s"),
    (["--max-s", "nan", "1", "true"], {}, "--max-s"),
    (["--max-s", "abc", "1", "true"], {}, "--max-s"),
    (["1", "true"], {HS.MAX_S_ENV: "700"}, HS.MAX_S_ENV),
    (["1", "true"], {HS.MAX_S_ENV: "soon"}, HS.MAX_S_ENV),
    (["--max-s", "10", "--term-grace", "61", "1", "true"], {}, "stop grace"),
])
def test_a_bad_declaration_is_refused_without_running_anything(slots, fake_run, monkeypatch, capsys, args, env, needle):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert M.main(args) == 2
    err = capsys.readouterr().err
    assert "REFUSING" in err and needle in err, err
    assert not fake_run.calls


def test_the_declaration_is_validated_before_any_queueing(slots):
    for bad in (0, -1, 600.01, float("nan"), float("inf"), "soon"):
        with pytest.raises(HS.ShortLaneError):
            with HS.heavy_slot(["x"], force=True, max_s=bad, log=lambda _m: None):
                pytest.fail("ran with a bad declaration")
    assert HS.parse_max_s(600) == 600.0 and HS.parse_max_s("0.5") == 0.5
    assert not HS._tickets(slots), "a refused declaration left a ticket"


# ── the log and the status ─────────────────────────────────────────────────────────────────────────────────────────

def test_each_release_appends_a_line_with_lane_wait_and_hold(slots):
    with HS.heavy_slot(["a", "short", "job"], force=True, max_s=30, log=lambda _m: None):
        pass
    with HS.heavy_slot(["a", "long", "job"], force=True, log=lambda _m: None):
        pass
    rows = [json.loads(ln) for ln in (slots / HS.EVENTS_FILE).read_text().splitlines()]
    assert [(r["kind"], r["lane"], r["max_s"]) for r in rows] == [("slot", "short", 30.0), ("slot", "long", 0.0)]
    assert all(r["waited_s"] >= 0 and r["held_s"] >= 0 and r["cmd"] for r in rows)


def test_the_event_log_rotates_once_past_its_cap(slots, monkeypatch):
    monkeypatch.setattr(HS, "EVENTS_MAX_BYTES", 200)
    for i in range(6):
        HS.log_event(slots, "slot", i=i, pad="x" * 60)
    assert (slots / (HS.EVENTS_FILE + ".1")).exists() and (slots / HS.EVENTS_FILE).stat().st_size <= 400
    (slots / "afile").write_text("x")
    HS.log_event(slots / "afile" / "sub", "slot")                          # unwritable: a log never fails a job (no raise)


def test_status_marks_a_waiting_short_job_and_the_holders_lane(slots, tmp_path):
    holder_file, marker = tmp_path / "holder", tmp_path / "w"
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(holder_file)])
    waiter = None
    try:
        _wait_for(holder_file.exists, "the holder")
        waiter = subprocess.Popen([sys.executable, "-c", _LANE_TAKER, str(marker), "60"])
        _wait_for(lambda: any(w["pid"] == waiter.pid for w in HS.collect()["waiting"]), "the short waiter's record")
        st = HS.collect()
        [w] = st["waiting"]
        assert w["lane"] == "short" and w["max_s"] == 60.0 and st["slots"][0]["lane"] == "long"
        text = HS.format_human(st)
        assert "SHORT<=60s" in text and "grant order" in text
    finally:
        _kill(holder, waiter)


def test_a_short_holder_far_past_its_worst_case_is_flagged_overdue(slots):
    with HS.heavy_slot(["a", "short", "job"], force=True, max_s=10, log=lambda _m: None):
        ok = HS.collect(now=time.time() + 10 + HS.SHORT_MARGIN_S)["slots"][0]
        assert ok["lane"] == "short" and ok["overdue"] is False
        late = HS.collect(now=time.time() + 10 + HS.SHORT_MARGIN_S + HS.SHORT_GRACE_MAX_S + 60)
        assert late["slots"][0]["overdue"] is True and HS.OVERDUE_TAG in HS.format_human(late)
