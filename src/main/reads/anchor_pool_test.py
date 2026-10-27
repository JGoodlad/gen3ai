"""The anchor lane's bounded concurrency (``main.reads.anchor_pool``): K units at once and never more, each on its own
95xx port, a failed unit retried and an overrun played long exactly as at K = 1, and the assembled cells byte-identical
between K = 1 and K = 3. SCHEDULING ONLY: a stubbed unit runner (or ``kakuna.run_unit`` over a scripted ``Popen``)
plays no game."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from main.reads import anchor_pool as AP
from main.reads import api, layout as LY
from main.reads import runner as RN
from main.reads import supervisors as SV
from main.reads.runner_test import CODE, _spec, _step, _zip
from main.x35 import kakuna as KA

SEEDS = "0-5"
LABELS = ("old", "new")
N_UNITS = 12                                   # 2 models x 1 team set x 6 unit seeds


def _cell() -> Dict[str, Any]:
    return {"battle_format": "gen3ou", "opponent": "metamon:Kakuna", "opponent_version": "Kakuna@ckpt34",
            "our_regime": "greedy", "their_regime": "greedy", "our_team_digest": "d1", "their_team_digest": "d1",
            "server_version": "ws_frontend@aaa", "teamset": "heldout72_hpfix"}


def _summary(name: str) -> Dict[str, Any]:
    wins = int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) % 21     # a fixed function of the unit alone
    return {"status": "OK", "n": 20, "wins": wins, "losses": 20 - wins, "ties": 0, "regime_verified_decisions": True,
            "peer_clean": True, "team_source_asymmetry": False, "cell": _cell(),
            "declared_team_cell": "symmetric", "team_digest_schema": "gen3_team_set_digest_v1"}


def write_played(st: Path, u: Any) -> None:
    """What a counted unit leaves on disk (its ``summary.json`` and ``games.jsonl``), a pure function of the unit."""
    d = st / "cells" / u.name
    d.mkdir(parents=True, exist_ok=True)
    s = _summary(u.name)
    (d / "games.jsonl").write_text("".join(
        json.dumps({"half": h, "index": i, "our_team": f"t{i}", "result": "W" if i < s["wins"] else "L", "turns": 30 + i})
        + "\n" for h in (0, 1) for i in range(10)))
    (d / "summary.json").write_text(json.dumps(s, sort_keys=True))


@pytest.fixture
def lane(tmp_path, run_archive_env, monkeypatch):
    """A planned state of 12 anchor units; ``checkout_code`` is stubbed (no git)."""
    spec = _spec(tmp_path, run_archive_env)
    doc = json.loads(spec.read_text())
    doc["anchor"]["u"]["unit_seeds"] = SEEDS
    spec.write_text(json.dumps(doc))
    _zip(run_archive_env / "rb_new", age=3600)
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=_step)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)
    monkeypatch.setenv("GEN3AI_HEAVY_SLOTS", "3")
    return state


def _sup(state: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in (state / "anchor_supervisor.jsonl").read_text().splitlines()]


class Meter:
    """A stub unit runner: counts the units in flight, holds the first ``wave`` of them at a barrier (so K of them are
    PROVEN to run at once, or the barrier times out and the test fails), then plays each one."""

    def __init__(self, wave: int = 1, delay: float = 0.02, fail: Optional[Dict[str, int]] = None,
                 slow: Optional[Dict[str, float]] = None) -> None:
        self.lock = threading.Lock()
        self.now = self.peak = 0
        self.played: List[str] = []
        self.threads: set = set()
        self.fail = dict(fail or {})                       # {unit name: attempts that leave it uncounted}
        self.slow = dict(slow or {})                       # {unit name: extra seconds after the barrier}
        self.delay = delay
        self.barrier = threading.Barrier(wave, timeout=30) if wave > 1 else None
        self.waited = 0
        self.wave = wave

    def __call__(self, st: Path, u: Any) -> int:
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
            self.played.append(u.name)
            self.threads.add(threading.current_thread().name)
            first_wave = self.barrier is not None and self.waited < self.wave
            self.waited += 1
        try:
            if first_wave:
                assert self.barrier is not None
                self.barrier.wait()
            time.sleep(self.delay + self.slow.get(u.name, 0.0))
            with self.lock:
                left = self.fail.get(u.name, 0)
                if left:
                    self.fail[u.name] = left - 1
            if not left:
                write_played(st, u)
            return 0 if not left else 1
        finally:
            with self.lock:
                self.now -= 1


def _run(state, unit, **kw):
    return RN.run(state, "anchor", time.time() + 3600, run_unit=unit, sleep=lambda s: None, **kw)


def test_k3_runs_three_units_at_once_and_never_more(lane):
    meter = Meter(wave=3)
    assert _run(lane, meter, concurrency=3) == 0
    assert meter.peak == 3, "the barrier held three units at once; a fourth would have raised the peak"
    assert sorted(meter.played) == sorted(f"{lab}__heldout72_hpfix__{s:02d}" for lab in LABELS for s in range(6))
    assert len(meter.played) == N_UNITS, "every unit exactly once"
    assert RN.status(lane)["anchor"]["u"] == {"old": {"counted": 6, "planned": 6}, "new": {"counted": 6, "planned": 6}}
    start = _sup(lane)[0]
    assert (start["event"], start["concurrency"], start["concurrency_requested"]) == ("start", 3, 3)
    assert _sup(lane)[-1] == {**_sup(lane)[-1], "event": "done", "not_counted": []}


def test_the_first_three_units_started_are_the_first_three_of_the_queue(lane):
    """Seed-outermost order is kept: a stopped queue stays balanced by seed whatever K is."""
    meter = Meter(wave=3)
    _run(lane, meter, concurrency=3)
    assert set(meter.played[:3]) == {"old__heldout72_hpfix__00", "new__heldout72_hpfix__00", "old__heldout72_hpfix__01"}


def test_k1_is_the_one_at_a_time_lane_on_the_supervisors_own_thread(lane):
    meter = Meter()
    assert _run(lane, meter) == 0                         # concurrency defaults to 1
    assert meter.peak == 1 and meter.threads == {threading.current_thread().name}
    assert meter.played == [f"{lab}__heldout72_hpfix__{s:02d}" for s in range(6) for lab in LABELS], \
        "unit seeds outermost, as before"
    assert (_sup(lane)[0]["concurrency"], _sup(lane)[0]["concurrency_requested"]) == (1, 1)


def test_concurrency_is_capped_at_the_heavy_slot_count(lane, monkeypatch, capfd):
    monkeypatch.setenv("GEN3AI_HEAVY_SLOTS", "2")
    assert AP.slot_cap() == 2
    meter = Meter(wave=2)
    assert _run(lane, meter, concurrency=5) == 0
    assert meter.peak == 2
    start = _sup(lane)[0]
    assert (start["concurrency"], start["concurrency_requested"]) == (2, 5)
    assert "capped at the box's 2 heavy slots" in capfd.readouterr().out


def test_a_bad_concurrency_is_refused_before_any_lock_or_play(lane):
    meter = Meter()
    for bad in (0, -1):
        with pytest.raises(RN.RunError, match="need an integer >= 1"):
            _run(lane, meter, concurrency=bad)
    with pytest.raises(RN.RunError, match="anchor lane only"):
        RN.anchor_concurrency(2, "s")
    assert RN.anchor_concurrency(1, "s") == 1
    assert meter.played == [] and not (lane / "anchor_supervisor.jsonl").exists(), "nothing started"
    assert _run(lane, Meter(), concurrency=1) == 0, "and the lane was left free to run"


def test_the_cli_takes_concurrency_and_defaults_to_one():
    from main.reads.cli import _parser

    assert _parser().parse_args(["run", "--state", "s", "--lane", "anchor", "--hours", "1"]).concurrency == 1
    a = _parser().parse_args(["resume", "--state", "s", "--lane", "anchor", "--hours", "1", "--concurrency", "3"])
    assert a.concurrency == 3
    # the /proc supervisor scan still reads --state and --lane from an argv that carries --concurrency
    argv = ["python", "-m", "main.reads", "run", "--state", "/x/st", "--lane", "anchor", "--hours", "48",
            "--concurrency", "3"]
    assert SV.parse_argv(argv) == ("run", "/x/st", "anchor")


# ------------------------------------------------------------------------- failures and retries, as at K = 1
def test_a_failed_units_retry_still_works_at_k3(lane):
    bad_once, bad_always = "old__heldout72_hpfix__01", "new__heldout72_hpfix__02"
    meter = Meter(wave=3, fail={bad_once: 1, bad_always: 99})
    assert _run(lane, meter, concurrency=3) == 0
    assert meter.played.count(bad_once) == 2 and KA.is_done(LY.anchor_state(lane, "u", "old"),
                                                            _unit(lane, "old", bad_once))
    assert meter.played.count(bad_always) == KA.MAX_ATTEMPTS, "bounded by MAX_ATTEMPTS, then given up"
    assert len(meter.played) == N_UNITS + 1 + (KA.MAX_ATTEMPTS - 1), "each unit once, the retries on top"
    assert _sup(lane)[-1]["event"] == "done" and _sup(lane)[-1]["not_counted"] == [bad_always]
    assert RN.status(lane)["anchor"]["u"] == {"old": {"counted": 6, "planned": 6}, "new": {"counted": 5, "planned": 6}}


def test_a_failure_is_counted_identically_at_k1_and_k3(lane, tmp_path):
    """The same failing units, the same attempts, at both K: the retry bookkeeping does not depend on K."""
    fails = {"old__heldout72_hpfix__00": 1, "new__heldout72_hpfix__03": 99}
    snap = tmp_path / "snap"
    shutil.copytree(lane, snap, symlinks=True)
    plays = {}
    for k in (1, 3):
        meter = Meter(wave=1, fail=dict(fails))
        assert _run(lane, meter, concurrency=k) == 0
        plays[k] = sorted(meter.played)
        assert _sup(lane)[-1]["not_counted"] == ["new__heldout72_hpfix__03"]
        shutil.rmtree(lane)
        shutil.copytree(snap, lane, symlinks=True)
    assert plays[1] == plays[3]


def _unit(state, label, name):
    return next(u for u in KA.load_plan(LY.anchor_state(state, "u", label)) if u.name == name)


class KeyedPopen:
    """``kakuna.subprocess.Popen`` for ``run_unit`` itself: each unit's attempts end as scripted BY UNIT NAME (``--out``'s
    last part), not by arrival order, so the script is the same at any K. It records the 95xx port each live child was
    given and fails the test if two live children ever share one."""

    script: Dict[str, List[Any]] = {}
    argvs: Dict[str, List[List[str]]] = {}
    live: Dict[int, str] = {}
    clashes: List[str] = []
    lock = threading.Lock()
    barrier: Optional[threading.Barrier] = None
    arrived = 0
    wave = 0
    peak = 0

    def __init__(self, argv, **_kw):
        self.argv = list(argv)
        self.name = Path(self.argv[self.argv.index("--out") + 1]).name
        self.port = int(self.argv[self.argv.index("--port") + 1])
        self.pid = 4242
        with KeyedPopen.lock:
            KeyedPopen.argvs.setdefault(self.name, []).append(self.argv)
            if self.port in KeyedPopen.live:
                KeyedPopen.clashes.append(f"port {self.port}: {KeyedPopen.live[self.port]} and {self.name}")
            KeyedPopen.live[self.port] = self.name
            KeyedPopen.peak = max(KeyedPopen.peak, len(KeyedPopen.live))

    def wait(self):
        with KeyedPopen.lock:
            hold = KeyedPopen.barrier is not None and KeyedPopen.arrived < KeyedPopen.wave
            KeyedPopen.arrived += 1
        if hold:
            assert KeyedPopen.barrier is not None
            KeyedPopen.barrier.wait()
        time.sleep(0.02)
        with KeyedPopen.lock:
            scripted = KeyedPopen.script.get(self.name)
            rc, counted = scripted.pop(0) if scripted else (0, True)
            KeyedPopen.live.pop(self.port, None)
        if counted:
            out = Path(self.argv[self.argv.index("--out") + 1])
            write_played(out.parent.parent, type("U", (), {"name": self.name})())
        return rc


@pytest.fixture
def keyed_play(monkeypatch, tmp_path):
    """The REAL ``kakuna.run_unit`` and port claim (its flocks in a scratch dir), over a scripted ``Popen``."""
    real_claim = KA.claim_port
    for k, v in dict(script={}, argvs={}, live={}, clashes=[], barrier=None, arrived=0, wave=0, peak=0).items():
        setattr(KeyedPopen, k, v)
    monkeypatch.setattr(KA, "preflight", lambda u: None)
    monkeypatch.setattr(KA, "claim_port", lambda *a, **k: real_claim(lock_dir=tmp_path / "ports"))
    monkeypatch.setattr(KA.subprocess, "Popen", KeyedPopen)
    return KeyedPopen


def test_every_unit_in_flight_holds_its_own_95xx_port(lane, keyed_play):
    keyed_play.wave, keyed_play.barrier = 3, threading.Barrier(3, timeout=30)
    assert RN.run(lane, "anchor", time.time() + 3600, sleep=lambda s: None, concurrency=3) == 0     # kakuna.run_unit
    assert keyed_play.peak == 3 and keyed_play.clashes == [], "three live children, never two on one port"
    ports = [int(a[a.index("--port") + 1]) for argvs in keyed_play.argvs.values() for a in argvs]
    assert len(ports) == N_UNITS and all(KA.PORT_RANGE[0] <= p <= KA.PORT_RANGE[1] for p in ports)
    assert all("--max-s" in a for argvs in keyed_play.argvs.values() for a in argvs), "each is a short-lane mem_cap job"
    assert all(any(x.endswith("scripts/ops/mem_cap.sh") for x in a) and "x35kakuna" in a
               for argvs in keyed_play.argvs.values() for a in argvs)
    assert RN.status(lane)["anchor"]["u"] == {"old": {"counted": 6, "planned": 6}, "new": {"counted": 6, "planned": 6}}


CHILD = """
import os, pathlib, sys, time
out, port, summary, shared, want = (pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4]),
                                    int(sys.argv[5]))
out.mkdir(parents=True, exist_ok=True)
fd = os.open(shared / ("live." + port), os.O_CREAT | os.O_EXCL | os.O_WRONLY)   # a live unit on this port: exit 1
(shared / ("started." + out.name)).write_text(port)
end = time.time() + 30
while len(list(shared.glob("started.*"))) < want:                                # wait for the whole first wave
    if time.time() > end:
        sys.exit(7)
    time.sleep(0.02)
time.sleep(0.1)
(out / "summary.json").write_text(summary)
(out / "games.jsonl").write_text("")
os.close(fd)
(shared / ("live." + port)).unlink()
"""


def test_three_real_child_processes_run_at_once_on_distinct_ports(lane, tmp_path, monkeypatch):
    """Real ``Popen`` children, the real port claim: the first three wait for each other (a cross-process rendezvous
    that times out, so a serial lane FAILS), and a child that finds its port already live exits 1 (so two live units on
    one port FAIL)."""
    import sys

    shared = tmp_path / "shared"
    shared.mkdir()
    real_claim = KA.claim_port
    monkeypatch.setattr(KA, "preflight", lambda u: None)
    monkeypatch.setattr(KA, "claim_port", lambda *a, **k: real_claim(lock_dir=tmp_path / "ports"))
    monkeypatch.setattr(KA, "unit_argv", lambda u, out, port, py=sys.executable, *, short=True: [
        sys.executable, "-c", CHILD, str(out), str(port), json.dumps(_summary(u.name)), str(shared), "3"])
    assert RN.run(lane, "anchor", time.time() + 3600, sleep=lambda s: None, concurrency=3) == 0
    started = sorted(p.name for p in shared.glob("started.*"))
    assert len(started) == N_UNITS and list(shared.glob("live.*")) == [], "every child ended and released its port"
    assert RN.status(lane)["anchor"]["u"] == {"old": {"counted": 6, "planned": 6}, "new": {"counted": 6, "planned": 6}}
    log = [json.loads(x) for st in ("old", "new") for x in
           (LY.anchor_state(lane, "u", st) / "units.jsonl").read_text().splitlines()]
    assert all(e["rc"] == 0 for e in log if e["event"] == "end")


def test_the_long_lane_fallback_is_per_unit_at_k3(lane, keyed_play):
    """An overrun unit is played once more long in the same attempt; one whose long retry also failed goes straight to
    the long job on its next attempt; the units around it are unaffected (all as at K = 1)."""
    over = "old__heldout72_hpfix__00"
    keyed_play.script = {over: [(KA.SHORT_LANE_OVERRUN, False), (124, False), (0, True)]}
    assert RN.run(lane, "anchor", time.time() + 3600, sleep=lambda s: None, concurrency=3) == 0
    a = keyed_play.argvs
    assert ["--max-s" in x for x in a[over]] == [True, False, False], "short, long retry, then long straight away"
    assert all(len(v) == 1 and "--max-s" in v[0] for k, v in a.items() if k != over)
    st = LY.anchor_state(lane, "u", "old")
    ev = [json.loads(x) for x in (st / "units.jsonl").read_text().splitlines()]
    assert [e["unit"] for e in ev if e["event"] == "short_lane_overrun"] == [over]
    assert KA.is_done(st, _unit(lane, "old", over)) and _sup(lane)[-1]["not_counted"] == []


# ------------------------------------------------------------------------- the lane's edges
def test_a_unit_that_raises_lets_the_others_finish_before_the_lane_refuses(lane):
    """A preflight refusal (the T28 halt, say) raised by one unit must not leave its siblings running unsupervised: the
    lane waits for them, counts them, then refuses; the claims are held until then."""
    done: List[str] = []

    def unit(st, u):
        if u.name == "old__heldout72_hpfix__00":
            raise KA.KakunaError("a T28 live-parse HALT marker is set")
        time.sleep(0.3)
        write_played(st, u)
        done.append(u.name)
        return 0

    with pytest.raises(KA.KakunaError, match="HALT"):
        _run(lane, unit, concurrency=3)
    assert sorted(done) == ["new__heldout72_hpfix__00", "old__heldout72_hpfix__01"], "both siblings ended first"
    assert _sup(lane)[-1]["event"] == "refused" and "HALT" in _sup(lane)[-1]["why"]
    assert _run(lane, Meter(), concurrency=1) == 0, "the lane's claims were released"


def test_the_registered_stop_starts_no_new_unit_and_waits_for_the_ones_in_flight(lane):
    stop = time.time() + 3600
    now = [stop - 100.0]
    slow = "old__heldout72_hpfix__00"
    meter = Meter(wave=2, slow={slow: 0.5})              # still playing when the other unit ends and the stop has passed

    def unit(st, u):
        now[0] = stop + 1.0                              # the registered stop passes as soon as a unit has started
        return meter(st, u)

    rc = RN.run(lane, "anchor", stop, run_unit=unit, sleep=lambda s: None, clock=lambda: now[0], concurrency=2)
    assert rc == 0 and len(meter.played) == 2 and meter.peak == 2
    ev = _sup(lane)[-1]
    assert ev["event"] == "stop_at" and ev["left"] == N_UNITS - 2, "the two in flight were finished and counted"
    u = RN.status(lane)["anchor"]["u"]
    assert u["old"]["counted"] == 1 and u["new"]["counted"] == 1, "the slow unit was waited for, not abandoned"
    assert KA.is_done(LY.anchor_state(lane, "u", "old"), _unit(lane, "old", slow))


# ------------------------------------------------------------------------- results do not depend on K
def test_the_assembled_cells_are_byte_identical_between_k1_and_k3(lane, tmp_path):
    """One planned state, played twice from the same bytes: K = 1, then K = 3 with the units finishing in a different
    order (a delay that shrinks with the unit seed). ``cells.json`` and every unit's records are equal bytes."""
    snap = tmp_path / "snap"
    shutil.copytree(lane, snap, symlinks=True)
    seen = {}
    for k in (1, 3):
        meter = Meter(wave=min(k, 3))
        late = {u.name: 0.05 * (5 - u.unit_seed) for lab in LABELS for u in KA.load_plan(LY.anchor_state(lane, "u", lab))}

        def unit(st, u, meter=meter, late=late):
            time.sleep(late[u.name] if k > 1 else 0)
            return meter(st, u)

        assert _run(lane, unit, concurrency=k) == 0
        api.assemble(lane)
        cells = LY.cells_file(lane).read_bytes()
        recs = {p.relative_to(lane).as_posix(): p.read_bytes() for p in sorted(lane.glob("anchor/u/*/cells/*/*"))}
        seen[k] = (cells, recs, sorted(meter.played))
        assert meter.peak == k
        shutil.rmtree(lane)
        shutil.copytree(snap, lane, symlinks=True)
    assert seen[1][0] == seen[3][0], "cells.json differs between K = 1 and K = 3"
    assert seen[1][1] == seen[3][1] and len(seen[1][1]) == 2 * N_UNITS, "a unit's records differ with K"
    assert seen[1][2] == seen[3][2], "different units were played"
    cells = json.loads(seen[3][0])
    assert cells["anchor"]["u"]["unplayed"] == 0 and cells["anchor"]["u"]["inconclusive"] == []
    assert cells["anchor"]["u"]["regime_id"], "a regime id is recorded on both"


# ------------------------------------------------------------------------- the pool itself
def test_a_pool_of_one_runs_inline_and_an_error_propagates_at_once():
    ended: List[str] = []
    u = type("U", (), {"name": "a"})()

    def boom(st, unit):
        raise ValueError("x")

    p = AP.UnitPool(1, lambda st, unit: threading.current_thread().name, lambda st, unit: ended.append(unit.name))
    p.submit(Path("/s"), u)
    assert ended == ["a"] and len(p) == 0
    with pytest.raises(ValueError):
        AP.UnitPool(1, boom, lambda *a: None).submit(Path("/s"), u)
    with pytest.raises(ValueError, match="at least one"):
        AP.UnitPool(0, boom, lambda *a: None)


def test_the_same_unit_name_in_two_states_is_two_units():
    gate = threading.Event()
    p = AP.UnitPool(2, lambda st, unit: gate.wait(30), lambda st, unit: None)
    u = type("U", (), {"name": "a"})()
    p.submit(Path("/s1"), u)
    assert p.busy(Path("/s1"), u) and not p.busy(Path("/s2"), u)
    p.submit(Path("/s2"), u)
    assert p.full() and p.free() == 0
    with pytest.raises(RuntimeError, match="full"):
        p.submit(Path("/s3"), u)
    gate.set()
    p.drain()
    assert len(p) == 0
