"""``main.ops.core_split_ab`` on a FAKE process tree and a fake clock: the rotation, the period and the verdict, every
refusal, and that the tree's own CPU list is back after a normal end, an error and a real SIGTERM. The only real
syscalls are on a ``sleep`` tree this file spawns itself; nothing here can reach a training run."""
from __future__ import annotations

import ast
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pytest

from main.ops import core_split_ab as AB
from main.ops import trainer_cpu_profile as T
from main.ops.trainer_cpu_profile_test import COMPILE, ENV, LAUNCHER, TRAINER, FakeProc, _stat
from utils import core_split, procfs
from utils.paths import src_path

CONTROL = list(core_split.DEFAULT.trainer_ids)
ARMS = {arm: core_split.narrowed(CONTROL, drop) for arm, drop in AB.ARMS}
OUTSIDER = 900                                   # a process that is NOT in the launcher's tree
CYCLE_S, ROLLOUT_FROM, ROLLOUT_TO = 30.0, 4.0, 29.0
BURST_S = 0.016


def _table(i: int, wall: float) -> str:
    return f"|    update_wall_s   | {wall} |\n|    collect_ms      | 2.5e+04 |\n|    iterations      | {i} |\n"


class World:
    """The fake clock, the fake affinity table, and the run: a rollout of bursts in every cycle (their period set by
    how many CPUs the workers hold) and a logger table 0.4 s after each rollout, as the real trainer writes it."""

    def __init__(self, root: Path, *, period: Optional[Dict[int, float]] = None, wall: Callable[[int], float] = lambda i: 44.0,
                 late: Tuple[float, ...] = (), eval_at: Optional[float] = None, die_at: Optional[float] = None,
                 vanish_at: Optional[float] = None, pause: Optional[Tuple[float, float]] = None,
                 stall: Optional[Tuple[float, float]] = None, hooks: Optional[Dict[float, Callable[[], None]]] = None,
                 deny: Callable[[int, Set[int]], bool] = lambda tid, cpus: False) -> None:
        self.proc = root / "proc"
        self.fp = FakeProc(self.proc)
        d = self.proc / str(OUTSIDER) / "task" / str(OUTSIDER)
        d.mkdir(parents=True)
        (d / "stat").write_text(_stat(OUTSIDER, "python3", 1, 0, 6))
        (self.proc / str(OUTSIDER) / "stat").write_text(_stat(OUTSIDER, "python3", 1, 0, 6))
        (self.proc / str(OUTSIDER) / "cmdline").write_text("python3\0agent_job.py")
        self.mask: Dict[int, Set[int]] = {tid: set(CONTROL) for tid in FakeProc.THREADS}
        self.mask[OUTSIDER] = {5, 6, 7}
        self.sets: List[Tuple[float, int, Tuple[int, ...]]] = []
        self.t = 0.0
        self.period = period or {10: 0.056, 8: 0.058, 6: 0.062}
        self.wall, self.late, self.eval_at, self.die_at, self.vanish_at = wall, list(late), eval_at, die_at, vanish_at
        self.next: Optional[float] = None
        self.log = root / "run.log"
        self.log.write_text("".join(_table(i, wall(i)) for i in range(3)))
        self.n_tables = 3
        self.marker = root / "state" / core_split.AB_MARKER
        self.on_a_between_rollouts = True
        self.pause, self.stall, self.hooks, self.deny = pause, stall, dict(hooks or {}), deny
        self.last_tree: Optional[AB.Tree] = None

    def clock(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        if self.late and self.t >= self.late[0]:
            dt += 0.030                               # a reading that comes 30 ms late
            self.late.pop(0)
        if self.stall is not None and self.t >= self.stall[0]:
            dt += self.stall[1]                       # the driver itself got no CPU for this long
            self.stall = None
        before, self.t = self.t, self.t + max(dt, 1e-6)
        for at in sorted(self.hooks):
            if before < at <= self.t:
                self.hooks.pop(at)()
        edge = ROLLOUT_TO + 0.4                       # the logger table of every cycle
        if (self.t - edge) // CYCLE_S > (before - edge) // CYCLE_S:
            with open(self.log, "a") as f:
                f.write(_table(self.n_tables, self.wall(self.n_tables)))
            self.n_tables += 1
        if self.eval_at is not None and before < self.eval_at <= self.t:
            with open(self.log, "a") as f:
                f.write("[EVAL] step 2,000,000: Rust eval core played 1,400 games\n")

    def setaff(self, tid: int, cpus: Set[int]) -> None:
        if tid not in self.mask:
            raise ProcessLookupError(tid)
        if self.deny(tid, set(cpus)):
            raise PermissionError(tid)
        self.mask[tid] = set(cpus)
        self.sets.append((self.t, tid, tuple(sorted(cpus))))

    def getaff(self, tid: int) -> Set[int]:
        if tid not in self.mask:
            raise ProcessLookupError(tid)
        return set(self.mask[tid])

    def runnable(self) -> int:
        if self.die_at is not None and self.t >= self.die_at:
            raise RuntimeError("the reader broke")
        if self.vanish_at is not None and self.t >= self.vanish_at:
            return -1
        c = self.t % CYCLE_S
        if self.pause is not None and self.pause[0] <= self.t < self.pause[1]:
            return 0                                  # the rollout itself stands still
        if not ROLLOUT_FROM <= c < ROLLOUT_TO:
            if self.next is not None and c >= ROLLOUT_TO + 0.5 and self.mask[301] != set(CONTROL):
                self.on_a_between_rollouts = False     # the update must run on arm A
            self.next = None
            return 0
        if self.next is None:
            self.next = self.t
        p = self.period[len(self.mask[301])]
        while self.t >= self.next + p:
            self.next += p
        return 6 if self.t < self.next + BURST_S else 0

    def tree(self, launcher: int = LAUNCHER) -> AB.Tree:
        world = self

        class Scripted(AB.Tree):
            def runnable(self) -> int:
                return world.runnable()

        self.last_tree = Scripted(launcher, proc=str(self.proc), setaff=self.setaff, getaff=self.getaff)
        return self.last_tree

    def add_process(self, pid: int, ppid: int, comm: str = "python3", like: int = TRAINER) -> None:
        """A new one-thread process under ``ppid``; it inherits the CPU list ``like`` holds now."""
        d = self.proc / str(pid) / "task" / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(_stat(pid, comm, ppid, 0, 2))
        (self.proc / str(pid) / "stat").write_text(_stat(pid, comm, ppid, 0, 2))
        (self.proc / str(pid) / "cmdline").write_text(comm)
        self.mask[pid] = set(self.mask[like])

    def kill_launcher(self) -> None:
        """The launcher dies; its child is re-parented to pid 1 and lives on with the list it holds."""
        shutil.rmtree(self.proc / str(LAUNCHER))
        (self.proc / str(TRAINER) / "stat").write_text(_stat(TRAINER, "python3", 1, 0, 1))

    def run(self, tmp: Path, **kw: Any) -> Dict[str, Any]:
        kw.setdefault("arms", ARMS)
        return AB.run_ab(self.tree(), kw.pop("arms"), tmp / "out", log=kw.pop("log", self.log), marker=self.marker,
                         clock=self.clock, sleep=self.sleep, handle_signals=False, **kw)

    def all_on_control(self, but: Tuple[int, ...] = ()) -> bool:
        return all(self.mask[tid] == set(CONTROL) for tid in FakeProc.THREADS if tid not in but)


def test_the_rotation_puts_every_arm_in_every_position_twice() -> None:
    assert len(AB.ROTATION) == 6
    for pos in range(3):
        assert sorted(order[pos] for order in AB.ROTATION) == ["A", "A", "B", "B", "C", "C"]
    assert [a for a, _ in AB.ARMS] == ["A", "B", "C"] and [len(ARMS[a]) // 2 for a in "ABC"] == [5, 4, 3]
    assert ARMS["A"] == CONTROL and set(ARMS["C"]) < set(ARMS["B"]) < set(ARMS["A"])


def test_a_full_run_follows_the_rotation_and_reads_the_period(tmp_path: Path) -> None:
    w = World(tmp_path)                               # B is 58 / 56 slower, C 62 / 56
    rec = w.run(tmp_path)
    assert rec["aborted"] is None and rec["restored"] is True and w.all_on_control() and not w.marker.exists()
    assert w.on_a_between_rollouts, "an update ran on a narrowed arm"
    assert [r["order"] for r in rec["rollouts"]] == list(AB.ROTATION)
    assert [(s["rollout"], s["pos"], s["arm"]) for s in rec["segments"]] == [
        (k, p, order[p]) for k, order in enumerate(AB.ROTATION) for p in range(3)]
    # only the tree's own threads were ever set, and only to an arm's list (a subset of the tree's own CPUs)
    assert {tid for _, tid, _ in w.sets} == set(FakeProc.THREADS) and w.mask[OUTSIDER] == {5, 6, 7}
    assert {cpus for _, _, cpus in w.sets} == {tuple(ARMS[a]) for a in "ABC"}
    for s in rec["segments"]:
        assert s["intervals"] >= 100 and s["dropped"] == 0 and s["settled"] is True and s["on_arm"] is True
        assert s["period_s"] == pytest.approx(w.period[len(ARMS[s["arm"]])], abs=0.0002)
    assert all(r["ended_by"] == "segments" and r["segments"] == 3 for r in rec["rollouts"])
    assert rec["restore"] == {"verified": True, "threads": len(FakeProc.THREADS), "off": [], "launcher_alive": True}
    assert rec["window"]["log_to"] > rec["window"]["log_from"] > 0 and rec["env_pid"] == ENV and rec["env_workers"] == 2
    b, c = rec["verdicts"]["B"], rec["verdicts"]["C"]
    assert b["loss"] == pytest.approx(AB.COLLECT_SHARE * (58 / 56 - 1), abs=0.002) and b["verdict"] == "PASS"     # 1.3 %
    assert c["loss"] == pytest.approx(AB.COLLECT_SHARE * (62 / 56 - 1), abs=0.002) and c["verdict"] == "FAIL"     # 3.8 %
    assert rec["adopt_arm"] == "B" and rec["control"]["ok"] and rec["control"]["moved"] == 0.0
    assert len(rec["control"]["before"]) == 3 and len(rec["control"]["after"]) == 6 and rec["evals_in_window"] == 0
    text = AB.render(rec)
    assert "adopt 4+4" in text and "python -m utils.core_split set 4+4 --next-launch" in text
    # the files: the record re-reads to the same verdict, and the recording is `trainer_cpu_profile`'s format
    saved = json.loads((tmp_path / "out" / "record.json").read_text())
    assert saved["verdicts"] == rec["verdicts"] and saved["arms"] == ARMS
    assert AB.main(["verdict", str(tmp_path / "out" / "record.json")]) == 0
    assert T.summarize_burst(T.load(tmp_path / "out" / "burst.jsonl"))["phases"]["rollout"]["seconds"] > 100


@pytest.mark.parametrize("c_period, verdict, adopt", [(0.059, "PASS", "C"), (0.0608, "INCONCLUSIVE", "B")])
def test_the_narrowest_passing_arm_is_adopted(tmp_path: Path, c_period: float, verdict: str, adopt: str) -> None:
    w = World(tmp_path, period={10: 0.056, 8: 0.058, 6: c_period})     # C: 1.9 % (pass) / 3.0 % (between the lines)
    rec = w.run(tmp_path)
    assert rec["verdicts"]["C"]["verdict"] == verdict and rec["adopt_arm"] == adopt and w.all_on_control()
    assert ("adopt 3+5" if adopt == "C" else "adopt 4+4") in AB.render(rec)


def test_an_eval_or_a_moved_control_makes_it_inconclusive(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    rec = World(tmp_path / "a", eval_at=3 * CYCLE_S + 10.0).run(tmp_path / "a")
    assert rec["evals_in_window"] == 1 and rec["adopt_arm"] == "A"
    assert all(v["verdict"] == "INCONCLUSIVE" and "an eval landed" in v["why"][0] for v in rec["verdicts"].values())
    assert "Keep the split as it is (5+3)" in AB.render(rec)
    # table 3 (the first after the start) still holds an update from BEFORE the A/B: tables 4..9 are the A/B's
    rec = World(tmp_path / "b", wall=lambda i: 44.0 if i <= 3 else 50.0).run(tmp_path / "b")
    assert rec["control"]["before"] == [44.0, 44.0, 44.0] and rec["control"]["after"] == [50.0] * 6
    assert rec["control"]["moved"] == pytest.approx(50 / 44 - 1, abs=1e-4) and rec["adopt_arm"] == "A"
    assert all("update_wall_s moved +13.6 %" in v["why"][0] for v in rec["verdicts"].values())


def test_a_late_reading_drops_its_interval_and_too_many_make_the_arm_inconclusive() -> None:
    rows = [[round(0.004 * i, 4), 1 if (i % 14) < 4 else 0] for i in range(2000)]        # a burst every 56 ms
    clean = AB.segment_period(rows, 0.5, 7.5, 0.012)
    assert clean["dropped"] == 0 and clean["period_s"] == pytest.approx(0.056) and clean["intervals"] == 124
    late = [r for r in rows if not 3.0 <= r[0] < 3.1]                                    # 100 ms without a reading
    got = AB.segment_period(late, 0.5, 7.5, 0.012)
    assert got["dropped"] == 1 and got["period_s"] == pytest.approx(0.056), "the late interval is out, not averaged in"
    assert AB.segment_period(rows, 0.5, 0.5, 0.012) == {"intervals": 0, "dropped": 0, "dropped_share": 1.0, "period_s": None}

    def seg(arm: str, k: int, period: float, **kw: Any) -> Dict[str, Any]:
        return {"rollout": k, "arm": arm, "intervals": 120, "dropped_share": 0.0, "period_s": period,
                "settled": True, "on_arm": True, **kw}

    ok = {"ok": True}
    base = [seg(a, k, p) for k in range(6) for a, p in (("A", 0.056), ("B", 0.057), ("C", 0.058))]
    d = AB.decide(base, rollouts=6, control=ok, evals=0)
    assert d["passing"] == ["B", "C"] and d["adopt_arm"] == "C"
    thin = [dict(s, dropped_share=0.3) if (s["arm"], s["rollout"]) == ("C", 2) else s for s in base]
    d = AB.decide(thin, rollouts=6, control=ok, evals=0)
    assert d["verdicts"]["C"]["verdict"] == "INCONCLUSIVE" and "rollout 3 arm C" in d["verdicts"]["C"]["why"][0]
    assert d["adopt_arm"] == "B"
    few = [dict(s, intervals=AB.MIN_INTERVALS - 1) if (s["arm"], s["rollout"]) == ("A", 0) else s for s in base]
    assert AB.decide(few, rollouts=6, control=ok, evals=0)["adopt_arm"] == "A", "arm A is in every ratio"
    assert AB.decide(base[:-1], rollouts=6, control=ok, evals=0)["verdicts"]["C"]["verdict"] == "INCONCLUSIVE"
    assert AB.decide(base, rollouts=6, control=ok, evals=0, aborted="x")["adopt_arm"] == "A"
    # a segment whose CPU list was not verified (at the switch, or at its end) is never a measurement
    for key, bad in (("settled", False), ("on_arm", False), ("on_arm", None)):
        loose = [dict(s, **{key: bad}) if (s["arm"], s["rollout"]) == ("C", 4) else s for s in base]
        d = AB.decide(loose, rollouts=6, control=ok, evals=0)
        assert d["verdicts"]["C"]["verdict"] == "INCONCLUSIVE" and d["adopt_arm"] == "B", (key, bad)
        assert "not verified" in d["verdicts"]["C"]["why"][0] and "rollout 5 arm C" in d["verdicts"]["C"]["why"][0]
    for key in ("settled", "on_arm"):
        unsaid = [{k: v for k, v in s.items() if k != key} for s in base]
        assert AB.decide(unsaid, rollouts=6, control=ok, evals=0)["adopt_arm"] == "A", "a record that does not say is unverified"


def test_the_pass_and_fail_lines_and_the_standard_error() -> None:
    def run(ratios: List[float]) -> Dict[str, Any]:
        segs = [{"rollout": k, "arm": a, "intervals": 120, "dropped_share": 0.0, "period_s": p, "settled": True,
                 "on_arm": True}
                for k, r in enumerate(ratios) for a, p in (("A", 1.0), ("B", r), ("C", r))]
        return AB.decide(segs, rollouts=len(ratios), control={"ok": True}, evals=0)["verdicts"]["B"]

    def ratio(loss: float) -> float:
        return 1.0 + loss / AB.COLLECT_SHARE

    assert (AB.PASS_MAX, AB.FAIL_MIN, AB.SE_MAX, AB.CONTROL_MAX, AB.COLLECT_SHARE) == (0.025, 0.035, 0.005, 0.10, 0.353)
    assert run([ratio(0.024)] * 6)["verdict"] == "PASS" and run([ratio(-0.01)] * 6)["verdict"] == "PASS"
    assert run([ratio(0.036)] * 6)["verdict"] == "FAIL"
    between = run([ratio(0.030)] * 6)
    assert between["verdict"] == "INCONCLUSIVE" and "between the pass and the fail line" in between["why"][0]
    noisy = run([ratio(0.0), ratio(0.04)] * 3)        # mean 2.0 % (a pass), but its standard error is 0.9 %
    assert noisy["verdict"] == "INCONCLUSIVE" and "standard error" in noisy["why"][0] and noisy["loss"] < AB.PASS_MAX
    c = AB.control_of([44.0, 45.0, 43.0], [44.0] * 6, 6)
    assert c["ok"] and c["moved"] == 0.0
    assert not AB.control_of([44.0, 44.0, 44.0], [39.0] * 6, 6)["ok"]
    assert not AB.control_of([44.0, None, 44.0], [44.0] * 6, 6)["ok"] and not AB.control_of([44.0] * 3, [44.0] * 5, 6)["ok"]


def test_it_refuses_before_changing_anything(tmp_path: Path) -> None:
    w = World(tmp_path)
    with pytest.raises(AB.Refusal, match="not a training launcher"):
        w.tree(TRAINER)
    with pytest.raises(AB.Refusal, match="not a training launcher"):
        w.tree(OUTSIDER)
    with pytest.raises(AB.Refusal, match="not running"):
        w.tree(4242)
    with pytest.raises(AB.Refusal, match="not inside the tree's own list"):
        w.run(tmp_path, arms={**ARMS, "B": ARMS["B"][:-1] + [5]})          # CPU 5 is an agent core
    short = tmp_path / "short.log"
    short.write_text(_table(0, 44.0) + _table(1, 44.0))
    with pytest.raises(AB.Refusal, match="fewer than 3 logger tables"):
        w.run(tmp_path, log=short)
    w.marker.parent.mkdir(parents=True)
    w.marker.write_text("{}")
    with pytest.raises(AB.Refusal, match="an A/B marker exists"):
        w.run(tmp_path)
    w.marker.unlink()
    w.mask[302] = set(ARMS["C"])                                           # a narrowing somebody left in place
    with pytest.raises(AB.Refusal, match="not all on arm A"):
        w.run(tmp_path)
    assert w.sets == [] and not w.marker.exists()
    # the setter itself never widens and never leaves the tree's own list
    w.mask[302] = set(CONTROL)
    tree = w.tree()
    with pytest.raises(AB.Refusal, match="not inside the tree's own"):
        tree.apply(CONTROL + [5], CONTROL)
    with pytest.raises(AB.Refusal):
        tree.apply([], CONTROL)
    assert w.sets == []
    assert tree.apply(ARMS["B"], CONTROL) and OUTSIDER not in {tid for _, tid, _ in w.sets}
    assert set(tree.tids()) == set(FakeProc.THREADS) and tree.tids()[301] == ENV


def test_a_launcher_whose_pid_was_reused_is_never_touched(tmp_path: Path) -> None:
    w = World(tmp_path)
    tree = w.tree()
    rest = (w.proc / str(LAUNCHER) / "stat").read_text().split(") ", 1)[1].split()
    rest[19] = "777"                                                       # another start time: another process
    (w.proc / str(LAUNCHER) / "stat").write_text(f"{LAUNCHER} (python3) " + " ".join(rest) + "\n")
    assert not tree.alive() and tree.tids() == {} and w.sets == []
    assert tree.apply(ARMS["C"], CONTROL) is False, "a walk that finds no thread verifies nothing"
    got = AB.restore_tree(tree, CONTROL)
    assert got == {"verified": True, "threads": 0, "off": [], "launcher_alive": False} and w.sets == []
    assert tree.touched == {}, "verified only because this driver narrowed nothing"


def test_an_error_mid_segment_restores_the_tree_and_propagates(tmp_path: Path) -> None:
    w = World(tmp_path, die_at=ROLLOUT_FROM + 3.0)                         # inside rollout 1's first segment (arm A)...
    with pytest.raises(RuntimeError, match="the reader broke"):
        w.run(tmp_path, rotation=("BCA",))                                 # ...which this rotation makes arm B
    assert any(cpus == tuple(ARMS["B"]) for _, _, cpus in w.sets), "it had narrowed"
    assert w.all_on_control() and not w.marker.exists()
    rec = json.loads((tmp_path / "out" / "record.json").read_text())
    assert rec["restored"] is True and "the reader broke" in rec["aborted"]


def test_vanished_workers_and_a_timeout_end_inconclusive_and_restored(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    w = World(tmp_path / "a", vanish_at=CYCLE_S + ROLLOUT_FROM + 10.0)     # rollout 2, second segment (arm C)
    rec = w.run(tmp_path / "a")
    assert "an env worker is gone" in rec["aborted"] and rec["restored"] and w.all_on_control() and not w.marker.exists()
    assert rec["adopt_arm"] == "A" and all(v["verdict"] == "INCONCLUSIVE" for v in rec["verdicts"].values())
    w = World(tmp_path / "b")
    rec = w.run(tmp_path / "b", timing=AB.Timing(max_s=CYCLE_S + ROLLOUT_FROM + 10.0))
    assert "1 of 6 rollouts" in rec["aborted"] and rec["restored"] and w.all_on_control() and rec["adopt_arm"] == "A"


def test_a_blip_inside_an_update_is_not_a_rollout(tmp_path: Path) -> None:
    w = World(tmp_path)
    real = w.runnable
    w.runnable = lambda: 1 if 2.5 <= w.t < 2.51 else real()               # type: ignore[method-assign]
    rec = w.run(tmp_path, rotation=AB.ROTATION[:2])
    assert [round(r["start"]) for r in rec["rollouts"]] == [34, 64], "the blip at 2.5 s started nothing; it only reset the idle clock"


def _make(w: World) -> Callable[[int, str, Dict[int, str], str], AB.Tree]:
    def make(pid: int, key: str, touched: Dict[int, str], proc: str) -> AB.Tree:
        return AB.Tree(pid, proc=proc, setaff=w.setaff, getaff=w.getaff, key=key, touched=touched)
    return make


def test_restore_reads_the_marker(tmp_path: Path) -> None:
    w = World(tmp_path)
    marker, make = w.marker, _make(w)
    assert AB.restore_from_marker(marker, proc=str(w.proc))[0] == AB.EXIT_OK        # no marker: nothing to do
    key = procfs.read_stat(LAUNCHER, str(w.proc)).key                               # type: ignore[union-attr]
    body = {"launcher": LAUNCHER, "launcher_key": key, "original": core_split.format_cpus(CONTROL),
            "driver_pid": 4242, "driver_key": "4242:1"}
    marker.parent.mkdir(parents=True)
    for tid in (TRAINER, 301, 302):
        w.mask[tid] = set(ARMS["C"])                                                # a driver died on arm C
    marker.write_text(json.dumps({**body, "driver_key": f"{TRAINER}:{TRAINER}"}))   # ...or is it still alive?
    code, msg = AB.restore_from_marker(marker, proc=str(w.proc), make_tree=make)
    assert code == AB.EXIT_REFUSED and "still running" in msg and not w.all_on_control() and marker.exists()
    assert "--force" not in msg and w.sets == [], "nothing overrides a live driver: a forced restore would be measured as an arm"
    marker.write_text(json.dumps({**body, "driver_key": f"{TRAINER}:999"}))         # that pid is another process now
    code, msg = AB.restore_from_marker(marker, proc=str(w.proc), make_tree=make)
    assert code == AB.EXIT_OK and w.all_on_control() and not marker.exists() and w.mask[OUTSIDER] == {5, 6, 7}
    assert {tid for _, tid, _ in w.sets} == {TRAINER, 301, 302} and f"{len(FakeProc.THREADS)} threads" in msg
    # a launcher that is gone (or whose pid is another process now) is not touched; a broken marker is kept
    w.sets.clear()
    marker.write_text(json.dumps({**body, "launcher_key": f"{LAUNCHER}:999"}))
    code, msg = AB.restore_from_marker(marker, proc=str(w.proc), make_tree=make)
    assert code == AB.EXIT_OK and "nothing to restore" in msg and w.sets == [] and not marker.exists()
    for text in ("{not json", json.dumps({"launcher": LAUNCHER}), json.dumps({**body, "original": "x-y"}), "[]"):
        marker.write_text(text)
        assert AB.restore_from_marker(marker, proc=str(w.proc), make_tree=make)[0] == AB.EXIT_REFUSED and marker.exists()
    assert core_split.ab_marker_path(marker.parent / "core_split.json") == marker, "`core_split show` reads this file"
    assert AB.main(["restore"], marker=tmp_path / "none.json") == AB.EXIT_OK


def test_a_narrowing_that_does_not_take_is_never_measured(tmp_path: Path) -> None:
    """The reviewer's first blocker: a set the kernel refuses, or a list something else changed mid-segment, used to
    be booked as that arm. Now the switch reports it and the segment's end reads every thread back."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    w = World(tmp_path / "a", deny=lambda tid, cpus: tid == 302 and cpus == set(ARMS["C"]))
    rec = w.run(tmp_path / "a")
    c_segs = [s for s in rec["segments"] if s["arm"] == "C"]
    assert len(c_segs) == 6 and all(s["settled"] is False and s["on_arm"] is False for s in c_segs)
    assert all(s["settled"] and s["on_arm"] for s in rec["segments"] if s["arm"] != "C")
    c = rec["verdicts"]["C"]
    assert c["verdict"] == "INCONCLUSIVE" and "not verified" in c["why"][0] and "ratios" in c and c["ratios"] == []
    assert rec["verdicts"]["B"]["verdict"] == "PASS" and rec["adopt_arm"] == "B" and rec["restored"] and w.all_on_control()
    # rollout 1 is A, B, C: something puts one worker back on the full list 3 s into the B segment
    w = World(tmp_path / "b", hooks={ROLLOUT_FROM + 11.0: lambda: w.mask.__setitem__(301, set(CONTROL))})
    rec = w.run(tmp_path / "b")
    b0 = next(s for s in rec["segments"] if (s["rollout"], s["arm"]) == (0, "B"))
    assert b0["settled"] is True and b0["on_arm"] is False
    assert rec["verdicts"]["B"]["verdict"] == "INCONCLUSIVE" and "rollout 1 arm B" in rec["verdicts"]["B"]["why"][0]
    assert rec["adopt_arm"] == "A" and w.all_on_control()


def test_a_pause_ends_the_rollout_at_once_and_a_stalled_driver_does_not(tmp_path: Path) -> None:
    """The second blocker: a rollout that ended early kept its segment's planned end, so time on arm A was booked to
    the narrow arm; and a driver that got no CPU for 0.3 s read its own stall as the rollout's end."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    w = World(tmp_path / "a", pause=(ROLLOUT_FROM + 5.0, ROLLOUT_FROM + 6.0))        # 1 s without a burst, on arm B
    rec = w.run(tmp_path / "a", rotation=("BCA", "CAB"))
    first = rec["rollouts"][0]
    assert first["ended_by"] == "idle" and first["segments"] == 1 and first["end"] == pytest.approx(ROLLOUT_FROM + 5.3, abs=0.03)
    seg = rec["segments"][0]
    assert seg["arm"] == "B" and seg["t_to"] == first["end"] < ROLLOUT_FROM + 8.0, "not the end planned at the start"
    back = [t for t, tid, cpus in w.sets if tid == 301 and cpus == tuple(CONTROL)]
    assert back and back[0] == pytest.approx(first["end"], abs=0.01), "arm A is back the moment the rollout ends"
    assert [s["rollout"] for s in rec["segments"]] == [0, 1, 1, 1] and rec["rollouts"][1]["ended_by"] == "segments"
    for arm in "BC":
        v = rec["verdicts"][arm]
        assert v["verdict"] == "INCONCLUSIVE" and "rollout 1 arm A" in v["why"][0] and "missing" in v["why"][0]
    assert rec["adopt_arm"] == "A" and w.all_on_control()
    w = World(tmp_path / "b", stall=(ROLLOUT_FROM + 5.0, 0.5))                       # the DRIVER misses 0.5 s
    rec = w.run(tmp_path / "b")
    assert all(r["ended_by"] == "segments" and r["segments"] == 3 for r in rec["rollouts"]), "the stall ended nothing"
    assert rec["segments"][0]["dropped"] == 1 and rec["adopt_arm"] == "B", "the stalled interval is out; the verdict stands"


def test_a_launcher_that_dies_while_narrowed_leaves_nothing_narrowed(tmp_path: Path) -> None:
    """The third blocker: with the launcher gone the tree walk is empty, and an empty walk used to count as a verified
    restore while the trainer and the env process lived on, narrowed. Every process a narrowing touched is named in
    the marker first and restored whether or not it is still in the tree."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    w = World(tmp_path / "a", hooks={ROLLOUT_FROM + 3.0: lambda: w.kill_launcher()})
    rec = w.run(tmp_path / "a", rotation=("BCA",))
    assert rec["aborted"] == "the launcher is gone" and rec["restored"] is True and not w.marker.exists()
    assert rec["restore"] == {"verified": True, "threads": len(FakeProc.THREADS) - 1, "off": [], "launcher_alive": False}
    assert w.all_on_control(but=(LAUNCHER,)), "the trainer, the env process and the compile worker are back"
    assert all(v["verdict"] == "INCONCLUSIVE" for v in rec["verdicts"].values())
    tree = w.last_tree
    assert tree is not None and set(tree.touched) == {LAUNCHER, TRAINER, ENV, COMPILE}
    # ...and when one of them cannot be put back, the run says so and the marker stays, naming it
    stuck = {"on": False}
    w = World(tmp_path / "b", deny=lambda tid, cpus: stuck["on"] and tid == 302,
              hooks={ROLLOUT_FROM + 3.0: lambda: (w.kill_launcher(), stuck.update(on=True))})
    rec = w.run(tmp_path / "b", rotation=("BCA",))
    assert rec["restored"] is False and rec["restore"]["off"] == [302] and rec["restore"]["launcher_alive"] is False
    assert w.mask[302] == set(ARMS["B"]) and "MAY STILL BE NARROWED" in AB.render(rec)
    left = json.loads(w.marker.read_text())
    assert left["touched"][str(ENV)] == f"{ENV}:{ENV}" and left["driver_pid"] == os.getpid()
    stuck["on"] = False
    code, msg = AB.restore_from_marker(w.marker, proc=str(w.proc), make_tree=_make(w))
    assert code == AB.EXIT_OK and "the launcher itself is gone" in msg and w.mask[302] == set(CONTROL) and not w.marker.exists()
    w.mask[302] = set(ARMS["B"])
    assert AB.restore_from_marker(w.marker, proc=str(w.proc), make_tree=_make(w))[0] == AB.EXIT_OK and w.mask[302] == set(ARMS["B"]), \
        "no marker, no touch"


def test_a_process_born_or_orphaned_while_narrowed_is_restored(tmp_path: Path) -> None:
    def orphan() -> None:                             # the compile worker double-forks away from the tree
        (w.proc / str(COMPILE) / "stat").write_text(_stat(COMPILE, "python3", 1, 0, 4))

    w = World(tmp_path, hooks={ROLLOUT_FROM + 3.0: lambda: w.add_process(500, TRAINER), ROLLOUT_FROM + 4.0: orphan})
    rec = w.run(tmp_path, rotation=("BCA",))          # both happen on arm B
    assert rec["aborted"] is None and rec["restored"] is True
    assert w.mask[500] == set(CONTROL) and w.all_on_control(), "the child born on arm B and the orphan are back"
    tree = w.last_tree
    assert tree is not None and tree.touched[500] == "500:500", "a later narrowing walked the new child: the marker named it"
    c = [cpus for _, tid, cpus in w.sets if tid == COMPILE]
    assert c == [tuple(ARMS["B"]), tuple(CONTROL)], "out of the tree it is narrowed no further, and put back at the rollout's end"
    back = next(t for t, tid, cpus in w.sets if tid == COMPILE and cpus == tuple(CONTROL))
    assert back < rec["rollouts"][0]["end"] - 5.0, "by the switch to arm A (the third segment), not only by the last restore"
    assert rec["restore"]["threads"] == len(FakeProc.THREADS) + 1


def test_the_marker_names_a_process_before_it_is_narrowed_and_at_a_narrow_segments_end(tmp_path: Path) -> None:
    """The marker is all a killed driver leaves behind: a process is named in it BEFORE its first narrowing, and a
    child born on a narrow list is named at that segment's end though no later narrowing walks it."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    named: List[bool] = []

    def watch(tid: int, cpus: Set[int]) -> bool:
        if cpus != set(CONTROL):
            named.append(str(FakeProc.THREADS[tid][0]) in json.loads(w.marker.read_text())["touched"])
        return False

    w = World(tmp_path / "a", deny=watch, hooks={ROLLOUT_FROM + 19.0: lambda: w.add_process(500, TRAINER)})
    rec = w.run(tmp_path / "a", rotation=("ABC",))    # the child is born on arm C, the rollout's last segment
    assert len(named) == 2 * len(FakeProc.THREADS) and all(named), "named on disk before the set, on both narrowings"
    tree = w.last_tree
    assert rec["restored"] and tree is not None and tree.touched[500] == "500:500" and w.mask[500] == set(CONTROL)
    w = World(tmp_path / "b", hooks={ROLLOUT_FROM + 19.0: lambda: w.add_process(500, TRAINER)})
    rec = w.run(tmp_path / "b", rotation=("BCA",))    # born on arm A: it never held a narrow list
    tree = w.last_tree
    assert rec["restored"] and tree is not None and 500 not in tree.touched and w.mask[500] == set(CONTROL)


def test_a_marker_pid_that_is_another_process_now_is_never_set(tmp_path: Path) -> None:
    w = World(tmp_path)
    key = procfs.read_stat(LAUNCHER, str(w.proc)).key                               # type: ignore[union-attr]
    w.mask[301] = set(ARMS["C"])
    detail = AB.restore_tree(_make(w)(LAUNCHER, key, {OUTSIDER: f"{OUTSIDER}:1"}, str(w.proc)), CONTROL)
    assert detail["verified"] and w.all_on_control() and w.mask[OUTSIDER] == {5, 6, 7}, "pid 900 began at another time"
    detail = AB.restore_tree(_make(w)(LAUNCHER, key, {OUTSIDER: f"{OUTSIDER}:{OUTSIDER}"}, str(w.proc)), CONTROL)
    assert detail["verified"] and w.mask[OUTSIDER] == set(CONTROL), "the SAME process is the driver's to put back"


def test_a_thread_born_during_a_pass_is_set_before_the_list_counts_as_verified(tmp_path: Path) -> None:
    born: List[int] = []

    def breed(tid: int, cpus: Set[int]) -> bool:      # the first set of the first pass: a child forks, on the old list
        if not born:
            born.append(tid)
            w.add_process(600, TRAINER)
        return False

    w = World(tmp_path, deny=breed)
    tree = w.tree()
    assert tree.apply(ARMS["B"], CONTROL) is True
    assert w.mask[600] == set(ARMS["B"]) and tree.touched[600] == "600:600", "a pass that changed something proves nothing"


def test_the_marker_is_created_whole_and_only_once(tmp_path: Path) -> None:
    marker = tmp_path / "state" / "m.json"
    AB._create_marker(marker, {"n": 1})
    with pytest.raises(AB.Refusal, match="an A/B marker exists"):
        AB._create_marker(marker, {"n": 2})
    assert json.loads(marker.read_text()) == {"n": 1} and os.listdir(marker.parent) == ["m.json"]


def test_an_ignored_signal_stays_ignored_and_every_handler_is_put_back() -> None:
    before = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGUSR2, signal.SIGPROF)}
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)     # as SIGHUP is under `nohup`
    try:
        with AB._signals(True) as sig:
            assert signal.getsignal(signal.SIGUSR1) is signal.SIG_IGN
            assert signal.getsignal(signal.SIGUSR2) == sig.handle and signal.getsignal(signal.SIGPROF) == sig.handle
            os.kill(os.getpid(), signal.SIGPROF)      # its default action would end the driver, narrowed
            assert sig.seen == signal.SIGPROF
        assert signal.getsignal(signal.SIGUSR2) == before[signal.SIGUSR2] and signal.getsignal(signal.SIGPROF) == before[signal.SIGPROF]
    finally:
        signal.signal(signal.SIGUSR1, before[signal.SIGUSR1])


def test_an_eval_that_ends_after_the_last_rollout_still_counts(tmp_path: Path) -> None:
    """An eval prints its line when it ENDS, ~30 s after it began: the window closes with the control's last table."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    rec = World(tmp_path / "a", eval_at=5 * CYCLE_S + ROLLOUT_TO + 15.0).run(tmp_path / "a")
    assert rec["rollouts"][-1]["end"] < 5 * CYCLE_S + ROLLOUT_TO < rec["evals_in_window"] * (5 * CYCLE_S + ROLLOUT_TO + 15.0)
    assert rec["evals_in_window"] == 1 and rec["adopt_arm"] == "A"
    rows = tmp_path / "b" / "eval_results.jsonl"                                     # the eval file's rows count too
    rows.write_text('{"step": 1}\n')
    w = World(tmp_path / "b", hooks={100.0: lambda: rows.write_text('{"step": 1}\n{"step": 2}\n')})
    rec = w.run(tmp_path / "b", evals=rows)
    assert rec["evals_in_window"] == 1 and rec["window"]["eval_rows_from"] == 1 and rec["window"]["eval_rows_to"] == 2
    assert all("an eval landed" in v["why"][0] for v in rec["verdicts"].values())


def test_a_signal_before_the_loop_or_inside_a_switch_stops_and_restores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlib

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    @contextlib.contextmanager
    def early(enabled: bool) -> Any:                  # a SIGTERM that landed before the loop armed the handler
        sig = AB._Signals()
        sig.handle(signal.SIGTERM, None)
        yield sig

    w = World(tmp_path / "a")
    monkeypatch.setattr(AB, "_signals", early)
    with pytest.raises(SystemExit) as e:
        w.run(tmp_path / "a")
    assert e.value.code == 128 + signal.SIGTERM and w.sets == [] and w.t == 0.0, "it never started"
    rec = json.loads((tmp_path / "a" / "out" / "record.json").read_text())
    assert rec["aborted"] == f"signal {int(signal.SIGTERM)}" and rec["restored"] is True and not w.marker.exists()
    monkeypatch.undo()
    hit = {"n": 0}

    def deny(tid: int, cpus: Set[int]) -> bool:       # the handler raises in the middle of the first narrowing
        hit["n"] += 1
        if hit["n"] == 3:
            raise AB._Stop(signal.SIGQUIT)
        return False

    w = World(tmp_path / "b", deny=deny)
    with pytest.raises(SystemExit) as e:
        w.run(tmp_path / "b", rotation=("BCA",))
    assert e.value.code == 128 + signal.SIGQUIT and w.all_on_control() and not w.marker.exists()
    assert any(cpus == tuple(ARMS["B"]) for _, _, cpus in w.sets), "it was half narrowed"
    assert {getattr(signal, n) for n in AB._HANDLED} >= {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT,
                                                         signal.SIGTSTP, signal.SIGUSR1, signal.SIGUSR2}


def test_a_failing_restore_is_retried_and_reported_never_raised(tmp_path: Path) -> None:
    calls = {"n": 0}
    w = World(tmp_path, die_at=ROLLOUT_FROM + 3.0)
    real = AB.restore_tree

    def flaky(tree: AB.Tree, original: Any) -> Dict[str, Any]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("no /proc for a moment")
        return real(tree, original)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(AB, "restore_tree", flaky)
        with pytest.raises(RuntimeError, match="the reader broke"):
            w.run(tmp_path, rotation=("BCA",))
    assert calls["n"] == 3 and w.all_on_control() and not w.marker.exists()
    calls["n"] = -10                                   # it never comes back: reported, the marker kept, the error raised
    (tmp_path / "x").mkdir()
    w = World(tmp_path / "x", die_at=ROLLOUT_FROM + 3.0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(AB, "restore_tree", flaky)
        with pytest.raises(RuntimeError, match="the reader broke"):
            w.run(tmp_path / "x", rotation=("BCA",))
    rec = json.loads((tmp_path / "x" / "out" / "record.json").read_text())
    assert rec["restored"] is False and "OSError" in rec["restore"]["error"] and w.marker.exists() and not w.all_on_control()


def test_the_train_env_is_told_from_the_eval_env_and_a_dead_worker_ends_the_reading(tmp_path: Path) -> None:
    w = World(tmp_path)
    tree = AB.Tree(LAUNCHER, proc=str(w.proc), setaff=w.setaff, getaff=w.getaff)

    def eval_env(ticks: int) -> None:                 # a second env process: the eval env, with a worker of its own
        for tid, comm in ((400, "rust_env_proc"), (401, "rust_env_w0")):
            d = w.proc / "400" / "task" / str(tid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "stat").write_text(_stat(tid, comm, TRAINER, ticks, 3))
            (d / "schedstat").write_text("0 0 7\n")
        (w.proc / "400" / "stat").write_text(_stat(400, "rust_env_proc", TRAINER, ticks, 3))
        (w.proc / "400" / "cmdline").write_text("rust_env_proc\0--fd\x0057")

    w.fp.ticks[ENV] = 1000
    w.fp.write()
    eval_env(30)
    assert tree.open_workers() == 2 and tree.env_pid == ENV, "the env process with the CPU time, not the eval env's worker"
    assert tree.runnable() == 0
    w.fp.state[301] = "R"
    w.fp.write()
    assert tree.runnable() == 1
    (w.proc / str(ENV) / "task" / "302" / "stat").write_text("")                    # one worker's stat reads empty
    assert tree.runnable() == -1, "ONE dead worker is a restarted env, not a quieter rollout"
    tree.close()
    w.fp.write()                                      # worker 302 reads again
    eval_env(499)                                     # under half the train env's 1000 ticks: still clear-cut
    assert tree.open_workers() == 2 and tree.env_pid == ENV
    tree.close()
    eval_env(500)                                     # exactly half: no longer
    with pytest.raises(AB.Refusal, match="cannot tell the train env from the eval env"):
        tree.open_workers()


def test_the_cli_checks_and_refuses_on_a_tree_that_is_not_on_the_default_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    w = World(tmp_path)
    run_dir = tmp_path / "models" / "rb_x"
    run_dir.mkdir(parents=True)
    (w.proc / str(TRAINER) / "cmdline").write_text(f"python3\0train.py\0--run-dir\0{run_dir}\0--steps\x001")
    assert w.tree().run_dir() == run_dir
    (run_dir / "launcher_child.full.log").write_text(w.log.read_text())
    argv = ["run", "--launcher-pid", str(LAUNCHER), "--out-dir", str(tmp_path / "o")]
    assert AB.main(argv + ["--check"], make_tree=w.tree, marker=w.marker) == AB.EXIT_OK
    out = capsys.readouterr().out
    assert "Nothing was changed" in out and str(run_dir / "launcher_child.full.log") in out and "(3 tables" in out
    assert f"2 train-env workers (env process {ENV})" in out and f"C={core_split.format_cpus(ARMS['C'])}" in out
    assert w.sets == [] and not w.marker.exists() and not (tmp_path / "o").exists()
    # --check runs every refusal of the real run: a marker, one thread off the list, too few tables
    w.marker.parent.mkdir(parents=True)
    w.marker.write_text("{}")
    assert AB.main(argv + ["--check"], make_tree=w.tree, marker=w.marker) == AB.EXIT_REFUSED
    assert "an A/B marker exists" in capsys.readouterr().err
    w.marker.unlink()
    w.mask[203] = set(CONTROL[:-1])
    assert AB.main(argv + ["--check"], make_tree=w.tree, marker=w.marker) == AB.EXIT_REFUSED
    assert "not all on arm A" in capsys.readouterr().err
    w.mask[203] = set(CONTROL)
    (run_dir / "launcher_child.full.log").write_text(_table(0, 44.0) + _table(1, float("1e999")) + "| iterations | 1e999 |\n")
    assert AB.main(argv + ["--check"], make_tree=w.tree, marker=w.marker) == AB.EXIT_REFUSED, "too early: one table"
    assert "fewer than 3 logger tables" in capsys.readouterr().err
    # ...and a record of another schema is not re-read into a verdict
    odd = tmp_path / "odd.json"
    odd.write_text(json.dumps({"schema": "something_else", "segments": [], "rotation": []}))
    assert AB.main(["verdict", str(odd)]) == AB.EXIT_REFUSED and "not a record of schema" in capsys.readouterr().err
    assert w.sets == []
    for tid in FakeProc.THREADS:
        w.mask[tid] = set(ARMS["B"])                  # a trainer already on a 4-core list
    assert AB.main(argv, make_tree=w.tree, marker=w.marker) == AB.EXIT_REFUSED
    assert "the A/B is defined for a trainer on the 5+3 split's list" in capsys.readouterr().err
    assert w.sets == [] and not w.marker.exists() and not (tmp_path / "o").exists()


_HARNESS = r"""
import json, os, subprocess, sys, time
from pathlib import Path
from main.ops import core_split_ab as AB

tmp = Path(sys.argv[1])
bash = subprocess.Popen(["bash", "-c", "sleep 300 & sleep 300 & wait"], start_new_session=True)
deadline = time.monotonic() + 20
while len(os.listdir(f"/proc/{bash.pid}/task")) < 1 or len(open(f"/proc/{bash.pid}/task/{bash.pid}/children").read().split()) < 2:
    assert time.monotonic() < deadline
    time.sleep(0.01)
original = sorted(os.sched_getaffinity(bash.pid))
arms = {"A": original, "B": original[:-1]}

LOG = []

def setaff(tid, cpus):                       # what was asked, and what the kernel says right after
    os.sched_setaffinity(tid, cpus)
    LOG.append([tid, sorted(cpus), sorted(os.sched_getaffinity(tid))])
    (tmp / "sets.tmp").write_text(json.dumps(LOG))
    os.replace(tmp / "sets.tmp", tmp / "sets")
    if set(cpus) != set(original) and len({t for t, c, _ in LOG if c == sorted(cpus)}) == 3:
        (tmp / "narrowed.tmp").write_text(json.dumps(sorted(cpus)))
        os.replace(tmp / "narrowed.tmp", tmp / "narrowed")

class Tree(AB.Tree):
    def open_workers(self):
        return 1
    def runnable(self):                      # one endless rollout: a burst every 20 ms
        t = time.monotonic() - T0
        return 1 if t >= 0.4 and (t % 0.020) < 0.008 else 0

tree = Tree(bash.pid, setaff=setaff, is_launcher=lambda argv: True)
(tmp / "pids.tmp").write_text(json.dumps({"pids": sorted(tree.tids()), "original": original}))
os.replace(tmp / "pids.tmp", tmp / "pids")
T0 = time.monotonic()
timing = AB.Timing(period_s=0.002, segment_s=120.0, settle_s=0.05, idle_s=0.2, confirm_s=0.1, confirm_bursts=2, max_s=100.0)
AB.run_ab(tree, arms, tmp / "out", log=tmp / "run.log", marker=tmp / "marker.json", timing=timing, rotation=("BA",))
"""


def _wait_for(path: Path, proc: "subprocess.Popen[bytes]", seconds: float = 60.0) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert proc.poll() is None, f"the harness exited ({proc.returncode}) before {path.name}"
        assert time.monotonic() < deadline, f"no {path.name} after {seconds} s"
        time.sleep(0.02)


def test_sigterm_mid_segment_restores_a_real_tree(tmp_path: Path) -> None:
    """The real thing, on a ``sleep`` tree this test owns: real ``sched_setaffinity``, a real SIGTERM while a
    narrowed arm is held, and the tree's own list is back when the driver has exited."""
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("this process may run on one CPU only: there is no narrower list to set")
    (tmp_path / "run.log").write_text("".join(_table(i, 44.0) for i in range(3)))
    env = {**os.environ, "PYTHONPATH": str(src_path()), "CUDA_VISIBLE_DEVICES": ""}
    harness = subprocess.Popen([sys.executable, "-c", _HARNESS, str(tmp_path)], env=env)
    pids: List[int] = []
    try:
        _wait_for(tmp_path / "pids", harness)
        info = json.loads((tmp_path / "pids").read_text())
        pids, original = info["pids"], set(info["original"])
        assert len(pids) == 3, "bash and its two sleeps"
        _wait_for(tmp_path / "narrowed", harness)
        narrow = set(json.loads((tmp_path / "narrowed").read_text()))
        assert narrow < original
        marker = json.loads((tmp_path / "marker.json").read_text())
        assert sorted(int(p) for p in marker["touched"]) == sorted(pids), "named on disk BEFORE they were narrowed"
        harness.send_signal(signal.SIGTERM)                                # arm B is held (the segment is 120 s long)
        assert harness.wait(timeout=60) == 128 + signal.SIGTERM
        # The lists are read where they were set, in the harness, right after each set: the box's pin enforcer
        # re-pins nice-19 processes every 30 s, so a read from HERE, later, could see ITS list (and it may re-pin the
        # narrowed tree before the SIGTERM, in which case the restore is one more set per thread, not one fewer).
        sets = json.loads((tmp_path / "sets").read_text())
        assert all(asked == seen for _, asked, seen in sets), "every set took"
        first = {tid: asked for tid, asked, _ in reversed(sets)}
        last = {tid: asked for tid, asked, _ in sets}
        assert set(first) == set(pids) and all(set(c) == narrow for c in first.values()), "the whole tree went to arm B"
        assert all(set(c) == original for c in last.values()), "restored on SIGTERM: each thread's last set is its own list"
        assert not (tmp_path / "marker.json").exists()
        rec = json.loads((tmp_path / "out" / "record.json").read_text())
        assert rec["restored"] is True and rec["aborted"] == f"signal {int(signal.SIGTERM)}"
        assert rec["restore"]["verified"] is True and rec["restore"]["threads"] == 3 and rec["restore"]["off"] == []
    finally:
        if harness.poll() is None:
            harness.kill()
            harness.wait(timeout=30)
        for p in pids:                                # the test's own bash and sleeps, by pid
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_one_signal_raises_once_and_the_restore_cannot_be_interrupted() -> None:
    sig = AB._Signals()
    sig.handle(signal.SIGTERM, None)                  # not raising: only written down
    assert sig.seen == signal.SIGTERM
    sig.raising = True
    with pytest.raises(AB._Stop):
        sig.handle(signal.SIGINT, None)
    assert sig.raising is False, "the handler disarms itself before it raises"
    sig.handle(signal.SIGHUP, None)                   # a second signal during the restore does not raise
    assert sig.seen == signal.SIGHUP and not issubclass(AB._Stop, Exception)


def test_the_cli_refuses_a_pid_that_is_no_launcher(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert AB.main(["run", "--launcher-pid", str(os.getpid()), "--out-dir", str(tmp_path / "o")]) == AB.EXIT_REFUSED
    assert "not a training launcher" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_the_module_changes_affinity_and_nothing_else() -> None:
    """It may set CPU affinity (through ``Tree`` only). It holds no way to signal, renice or spawn a process."""
    tree = ast.parse(Path(AB.__file__).read_text(encoding="utf-8"))
    banned = {"kill", "killpg", "pthread_kill", "raise_signal", "setpriority", "nice", "sched_setscheduler", "Popen",
              "system", "run", "check_call", "check_output"}
    called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not (called & banned), sorted(called & banned)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "subprocess" not in imported
    named = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "sched_setaffinity"]
    assert len(named) == 1, "the one affinity setter is Tree's default argument"
