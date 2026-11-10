"""The heavy slots' admission BY MEMORY against REAL holders and waiters (``utils.heavy_slots`` + ``utils.heavy_admission``;
the arithmetic alone is ``heavy_admission_test.py``). The box is faked: a 90 GiB ``meminfo`` (budget 62 GiB), an empty
cgroup directory the tests put scopes in, a temporary slot directory, and NO ``$GEN3AI_HEAVY_SLOTS`` (so the box's own
rule is the one under test).

Load-bearing pins, each FAILS on revert of what it guards:
* a third and a fourth capped job run while the caps fit; the fifth meets the hard maximum; a cap that does not fit waits;
* a job with no cap, an unreadable MemTotal and a holder from the old code each restore the fixed count of 2;
* a capped scope that holds no slot (a BLOCKING / GPU job) is committed memory; a holder's own scope is charged to it;
* FIFO holds across the rule: nobody passes a queued job whose cap does not fit, or a queued job with no cap;
* THE SWITCH BACK: ``--admission count`` binds a job already queued; ``memory`` returns; an explicit
  ``$GEN3AI_HEAVY_SLOTS`` is the fixed count with no memory check;
* the routine gate is admitted by its own ``mem_cap.sh 24`` cap, read from the argv ``utils.ship_gate`` builds.
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from utils import gate_lock as GL
from utils import heavy_admission as HA
from utils import heavy_slots as HS
from utils import mem_cap as M
from utils import procfs
from utils import ship_gate as SG
from utils.contention import scale_timeout
from utils.paths import repo_path, src_root

pytestmark = pytest.mark.integration

G = HA.GiB

_TAKER = (
    "import pathlib, sys, time\n"
    "from utils import heavy_slots as HS\n"
    "marker, hold, cap, wait = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4] == 'wait'\n"
    "try:\n"
    "    with HS.heavy_slot(['taker', marker], force=True, wait=wait, poll_s=0.05, report_every_s=0.2,\n"
    "                       cap_gb=None if cap == '-' else float(cap)) as s:\n"
    "        pathlib.Path(marker).write_text(str(s)); time.sleep(hold)\n"
    "except HS.SlotsFull as e:\n"
    "    print(e, file=sys.stderr); sys.exit(HS.EXIT_SLOTS_FULL)\n"
)


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A 90 GiB box (budget 62 GiB) with an empty heavy slice; returns the fake slice's cgroup directory."""
    for k in (HS.SLOTS_ENV, HS.HELD_ENV, HS.BLOCKING_ENV, HS.MAX_S_ENV, HA.MODE_ENV, HA.MAX_SLOTS_ENV, HA.RESERVE_ENV,
              HA.SLICE_ENV, HA.SLICE_GB_ENV, GL.HELD_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(HS.DIR_ENV, str(tmp_path / "slots"))
    monkeypatch.setenv("GEN3AI_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("PYTHONPATH", str(src_root()))
    mi = tmp_path / "meminfo"
    mi.write_text(f"MemTotal:       {90 * G // 1024} kB\n")
    cg = tmp_path / "cg"
    cg.mkdir()
    monkeypatch.setenv(HA.MEMINFO_ENV, str(mi))
    monkeypatch.setenv(HA.CGROUP_ENV, str(cg))
    _PROCS.clear()
    yield cg
    for p in _PROCS:                    # only processes this test started
        if p.poll() is None:
            os.kill(p.pid, signal.SIGKILL)
        p.wait()
    _PROCS.clear()


_PROCS: list = []


def _taker(marker, cap, *, hold=120.0, wait=True) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-c", _TAKER, str(marker), str(hold), "-" if cap is None else str(cap),
                          "wait" if wait else "nowait"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _PROCS.append(p)
    return p


def _wait_for(pred, what: str, timeout: float = 20.0) -> None:
    end = time.monotonic() + scale_timeout(timeout)
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _holders(tmp_path, *caps):
    """Start one holder per cap, IN ORDER (each has its slot before the next starts); returns ``[(proc, slot), …]``."""
    out = []
    for i, cap in enumerate(caps):
        m = tmp_path / f"h{len(_PROCS)}"
        p = _taker(m, cap)
        _wait_for(m.exists, f"holder {i} (cap {cap})")
        out.append((p, int(m.read_text())))
    return out


def _queued(tmp_path, cap) -> subprocess.Popen:
    """Start a job that must QUEUE; returns once its wait record is on disk (the queue then knows its cap)."""
    m = tmp_path / f"q{len(_PROCS)}"
    p = _taker(m, cap, hold=0.0)
    _wait_for(lambda: (HS.lock_dir() / f"wait-{p.pid}.json").exists(), "the job's wait record")
    p.marker = m
    return p


def _try(cap):
    """One ``--no-wait`` take from this process: the slot index, or the refusal's text."""
    try:
        with HS.heavy_slot(["probe"], force=True, wait=False, log=lambda _m: None, cap_gb=cap) as s:
            return s
    except HS.SlotsFull as e:
        return str(e)


def _scope(cg, name, cap_gb, populated=1):
    p = cg / name
    p.mkdir(exist_ok=True)
    (p / "memory.max").write_text(f"{int(cap_gb * G)}\n")
    (p / "cgroup.events").write_text(f"populated {populated}\nfrozen 0\n")


# ── the rule ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_four_capped_jobs_run_at_once_and_the_fifth_meets_the_hard_maximum(box, tmp_path):
    hs = _holders(tmp_path, 12, 12, 12, 12)
    assert [s for _, s in hs] == [3, 2, 1, 0], "capped jobs fill from the top; the fixed count of 2 stopped the third"
    fifth = _try(1)
    assert isinstance(fifth, str) and "all 4 heavy slots are held (the hard maximum)" in fifth, fifth
    st = HS.collect()
    assert st["n_slots"] == 4 and [r["cap_gb"] for r in st["slots"]] == [12.0] * 4
    a = st["admission"]
    assert a["mode"] == "memory" and a["budget_gb"] == 62.0 and a["committed_gb"] == 48.0 and a["unknown_holders"] == 0
    text = HS.format_human(st)
    assert "admission: memory" in text and "<= 62.0 GiB" in text and text.count("cap 12G") == 4, text


def test_a_cap_that_does_not_fit_is_refused_and_a_smaller_one_runs(box, tmp_path):
    _holders(tmp_path, 24, 24)
    big = _try(24)
    assert isinstance(big, str) and "does not fit" in big and "REFUSING (--no-wait)" in big, big      # 72 > 62
    assert _try(14) == 1, "62 of 62 GiB is inside the budget"
    assert isinstance(_try(14.5), str)


def test_a_job_with_no_cap_is_held_to_the_fixed_count_of_two(box, tmp_path):
    (pa, sa), = _holders(tmp_path, 12)
    assert sa == 3
    assert _try(None) == 0, "the second job of the fixed count takes the lowest slot"
    _holders(tmp_path, 12)
    full = _try(None)
    assert isinstance(full, str) and "fixed count of 2" in full and "declares no cap" in full, full
    assert _try(12) == 1, "control: the same box admits a job that declares its cap"


def test_an_unreadable_memtotal_restores_the_fixed_count(box, tmp_path, monkeypatch):
    _holders(tmp_path, 12, 12)
    assert _try(12) == 1
    monkeypatch.setenv(HA.MEMINFO_ENV, str(tmp_path / "absent"))
    full = _try(12)
    assert isinstance(full, str) and "fixed count of 2" in full and "MemTotal" in full, full
    assert "UNKNOWN" in HS.format_human(HS.collect())


def test_a_holder_from_the_old_code_restores_the_fixed_count_until_its_scope_is_seen(box, tmp_path):
    (pa, sa), (pb, sb) = _holders(tmp_path, 12, 12)
    rec = HS.lock_dir() / f"slot{sb}.json"
    raw = json.loads(rec.read_text())
    assert raw.pop("cap_gb") == 12.0                        # the record the code before this rule wrote
    rec.write_text(json.dumps(raw))
    full = _try(12)
    assert isinstance(full, str) and "fixed count of 2" in full and "holder" in full, full
    assert HS.collect()["admission"]["unknown_holders"] == 1
    _scope(box, f"gen3ai-capped-old-{pb.pid}.scope", 12)    # its mem_cap scope carries the enforced cap
    assert _try(12) == 1, "a holder whose scope is readable is a known holder"


def test_a_capped_scope_that_holds_no_slot_is_committed_memory(box, tmp_path):
    _holders(tmp_path, 12)
    _scope(box, "gen3ai-capped-gpuwindow-1.scope", 40)      # pid 1: nobody's child, no slot (a BLOCKING / GPU job)
    full = _try(12)
    assert isinstance(full, str) and "does not fit" in full, full                                    # 64 > 62
    assert HS.collect()["admission"]["unslotted_gb"] == 40.0
    _scope(box, "gen3ai-capped-gpuwindow-1.scope", 40, populated=0)
    assert _try(12) == 2, "an EMPTY scope is not a running job"


def test_a_holders_own_scope_is_charged_to_it_when_it_is_larger_than_what_it_declared(box, tmp_path):
    (pa, _), = _holders(tmp_path, 12)
    _scope(box, f"gen3ai-capped-inner-{pa.pid}.scope", 48)
    assert isinstance(_try(24), str), "48 + 24 > 62: the enforced cap counts, not the smaller declaration"
    assert _try(14) == 2                                    # 48 + 14 = 62: charged once, not 12 + 48


# ── FIFO across the rule ─────────────────────────────────────────────────────────────────────────────────────

def test_a_held_admission_lock_costs_a_poll_never_a_hang(box, monkeypatch):
    """A job stopped mid-admission holds ``admit.lock``: the others are refused for that poll, with the reason, and take
    the slot once it is released (a blocking lock here froze every waiter with no report and defeated ``--no-wait``)."""
    monkeypatch.setattr(HS, "ADMIT_LOCK_WAIT_S", 0.2)
    d = HS.lock_dir()
    d.mkdir(parents=True, exist_ok=True)
    with open(d / HS.ADMIT_LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        refusal = _try(12)
        assert isinstance(refusal, str) and HS.ADMIT_BUSY in refusal, refusal
        assert all(not r["pids"] for r in HS.collect()["slots"]), "nobody is admitted without the lock"
    assert _try(12) == 3


def test_nobody_passes_a_queued_job_whose_cap_does_not_fit(box, tmp_path):
    (pa, _), (pb, _) = _holders(tmp_path, 24, 24)
    q = _queued(tmp_path, 24)                               # 72 > 62: it waits
    assert not q.marker.exists()
    late = _try(12)                                         # 60 would fit, but it is behind q
    assert isinstance(late, str) and "queued behind" in late, late
    os.kill(pa.pid, signal.SIGKILL)                         # a holder ends: 48 fits, the queue's head runs
    _wait_for(q.marker.exists, "the queued job to take the freed budget")
    assert q.wait(timeout=scale_timeout(20)) == 0
    assert not list(HS.lock_dir().glob("wait-*.json")), "a wait record outlived its wait"


def test_nobody_passes_a_queued_job_that_declares_no_cap(box, tmp_path):
    _holders(tmp_path, 12, 12)
    q = _queued(tmp_path, None)                             # the fixed count: it waits for one of the two to end
    assert not q.marker.exists()
    late = _try(12)
    assert isinstance(late, str) and "fixed count of 2" in late and "queued ahead" in late, \
        f"a stream of capped jobs would starve a job with no cap: {late}"


# ── the switch back ──────────────────────────────────────────────────────────────────────────────────────────

def test_the_switch_back_binds_a_queued_job_and_memory_returns(box, tmp_path, capsys):
    _holders(tmp_path, 12, 12)
    assert HS.slot_count() == 4
    assert HS.main(["--admission", "count"]) == 0 and (HS.lock_dir() / HA.MODE_FILE).read_text().strip() == "count"
    assert HS.slot_count() == 2
    full = _try(12)
    assert isinstance(full, str) and "fixed count of 2" in full and "admission is `count`" in full, full
    assert "admission: COUNT" in HS.format_human(HS.collect())
    q = _queued(tmp_path, 12)
    assert HS.admission(HS.lock_dir(), HS.slot_files(), 12.0, pid=q.pid).slot is None and not q.marker.exists()
    assert HS.main(["--admission", "memory"]) == 0          # no restart: the queued job reads the mode at its next poll
    _wait_for(q.marker.exists, "the queued job to be admitted once the box is back on memory")
    assert q.marker.read_text() == "1"
    assert HS.main(["--admission", "sideways"]) == 2
    capsys.readouterr()


def test_the_environment_switches_one_process_back(box, tmp_path, monkeypatch):
    _holders(tmp_path, 12, 12)
    monkeypatch.setenv(HA.MODE_ENV, "count")
    assert isinstance(_try(12), str) and HS.slot_count() == 2
    monkeypatch.setenv(HA.MODE_ENV, "memory")
    assert _try(12) == 1


def test_an_explicit_slot_count_is_the_fixed_count_with_no_memory_check(box, tmp_path, monkeypatch):
    monkeypatch.setenv(HS.SLOTS_ENV, "2")
    assert (HS.fixed_slots(), HS.slot_files(), HS.slot_count()) == (2, 2, 2)
    hs = _holders(tmp_path, 12, 12)
    assert [s for _, s in hs] == [0, 1], "the explicit count is the code path from before this rule: lowest slot first"
    full = _try(12)
    assert isinstance(full, str) and "all 2 heavy slots are held or spoken for" in full, full
    assert "fixed count of 2" in HS.format_human(HS.collect()) and "no memory check" in HS.format_human(HS.collect())
    monkeypatch.setenv(HS.SLOTS_ENV, "3")
    assert _try(80) == 2, "an explicit count checks no memory (the operator's override)"


def test_the_slot_count_a_lane_pool_clamps_to_follows_the_rule(box, monkeypatch):
    assert HS.slot_count() == HA.MAX_SLOTS == 4 and HS.HEAVY_SLOTS == 2
    monkeypatch.setenv(HA.MAX_SLOTS_ENV, "3")
    assert (HS.slot_files(), HS.slot_count()) == (3, 3)
    monkeypatch.setenv(HA.MODE_ENV, "count")
    assert (HS.slot_files(), HS.slot_count()) == (3, 2), "count mode still watches every slot file, and admits 2"


# ── what is recorded ─────────────────────────────────────────────────────────────────────────────────────────

def test_the_slot_event_records_the_cap_and_the_rule(box):
    assert _try(12) == 3 and _try(None) == 0
    ev = [json.loads(x) for x in (HS.lock_dir() / HS.EVENTS_FILE).read_text().splitlines()]
    slots = [(e["slot"], e["cap_gb"], e["rule"]) for e in ev if e["kind"] == "slot"]
    assert slots == [(3, 12.0, HA.RULE_MEMORY), (0, 0.0, HA.RULE_COUNT)], slots


# ── the routine gate ─────────────────────────────────────────────────────────────────────────────────────────

def test_the_caps_a_command_line_declares():
    sh = str(repo_path("scripts", "ops", "mem_cap.sh"))
    assert M.declared_cap_gb([sh, "12", "python", "-m", "x"]) == 12.0
    assert M.declared_cap_gb([sh, "--name", "x35kakuna", "--max-s", "600", "--slot", "--no-wait", "1.5", "true"]) == 1.5
    every = [x for f in M.VALUE_FLAGS for x in (f, "1")] + list(M.BARE_FLAGS)
    assert M.declared_cap_gb([sh, *every, "24", "true"]) == 24.0, "a flag mem_cap takes is not skipped by the cap reader"
    for not_capped in ([], ["python", "-m", "pytest"], ["convoy-gate"], [sh], [sh, "--status"], [sh, "--new-flag", "24", "x"],
                       [sh, "lots", "true"], [sh, "0", "true"], [sh, "-3", "true"]):
        assert M.declared_cap_gb(not_capped) is None, not_capped


@pytest.mark.parametrize("max_run_s", [None, 5400.0])
def test_the_ship_gates_argv_declares_the_gates_cap(max_run_s):
    argv = SG.build_argv(SG.Shape({"/m/rb_x": 1}, [5, 6, 7], 19, 3), sys.executable, max_run_s=max_run_s)
    inner = argv[argv.index(str(repo_path("scripts", "ops", "gate_lock.sh"))) + 1:]      # what gate_lock is handed
    assert M.declared_cap_gb(inner) == float(SG.MEM_CAP_GB) == 24.0


def test_the_gate_is_admitted_by_its_own_cap(box, tmp_path, monkeypatch):
    monkeypatch.setenv(GL.DIR_ENV, str(tmp_path / "gate"))
    monkeypatch.setattr(procfs, "live_training_runs", lambda *a, **k: {"/m/rb_admission_test": 1})
    _holders(tmp_path, 12, 12)
    gate = [str(repo_path("scripts", "ops", "mem_cap.sh")), "--max-run-s", "5400", "24", "python", "-m", "pytest"]
    with GL.gate_slot(log=lambda _m: None, cmd=gate):
        held = HS.verified_holder()
        assert held is not None and held[1] == 1, "the gate queued behind two play units although 48 of 62 GiB fit"
        rec = json.loads((HS.lock_dir() / "slot1.json").read_text())
        assert rec["cap_gb"] == 24.0
        assert _try_outside(12) == 0, "60 of 62 GiB: one more 12 GiB unit still fits beside the gate"
        assert "does not fit" in _try_outside(24)
    said: list = []
    with pytest.raises(TimeoutError):   # control: a gate that declares no cap is a third job of the fixed count
        with GL.gate_slot(log=said.append, cmd=["python", "-m", "pytest"], timeout_s=0.3, poll_s=0.02):
            pytest.fail("a gate that declares no cap ran as a third heavy job")
    waits = [m for m in said if m.startswith("[heavy_slots] waiting")]
    assert waits and all("fixed count of 2" in m and "one of 4" not in m for m in waits), said   # the wait names its rule
    assert not list(HS.lock_dir().glob("wait-*.json")) and not list(HS.lock_dir().glob("ticket-*.tk"))


def _try_outside(cap):
    """`_try` from a process that is NOT inside the gate's slot (a child would re-enter the holder's slot)."""
    r = subprocess.run([sys.executable, "-c", _TAKER, os.devnull, "0", str(cap), "nowait"],
                       env={k: v for k, v in os.environ.items() if k not in (HS.HELD_ENV, GL.HELD_ENV)},
                       capture_output=True, text=True, timeout=scale_timeout(30))
    return 0 if r.returncode == 0 else r.stderr
