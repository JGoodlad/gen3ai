"""``main.ops.trainer_cpu_profile`` on a FAKED ``/proc``: the thread groups, the interval arithmetic, the phase rule
and its excluded band, the per-core peak, the window's fps, and that the module can only READ."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from main.ops import trainer_cpu_profile as T

HZ = 100
LAUNCHER, TRAINER, ENV, COMPILE = 100, 200, 300, 400


def _stat(pid: int, comm: str, ppid: int, ticks: int, cpu: int, state: str = "S") -> str:
    rest = ["0"] * 50
    rest[0], rest[1], rest[11], rest[12], rest[19], rest[36] = state, str(ppid), str(ticks), "0", str(pid), str(cpu)
    return f"{pid} ({comm}) " + " ".join(rest) + "\n"


class FakeProc:
    """A ``/proc`` under ``tmp_path``: the launcher, the trainer (main + autograd + cuda threads), one Rust env
    process with two workers, and one inductor compile worker."""

    #: tid -> (pid, thread name, parent pid of the PROCESS, cpu it last ran on)
    THREADS: Dict[int, Tuple[int, str, int, int]] = {
        LAUNCHER: (LAUNCHER, "python3", 1, 0),
        TRAINER: (TRAINER, "python3", LAUNCHER, 1),
        201: (TRAINER, "pt_autograd_0", LAUNCHER, 9),
        202: (TRAINER, "cuda-EvtHandlr", LAUNCHER, 2),
        203: (TRAINER, "python3", LAUNCHER, 2),
        ENV: (ENV, "rust_env_proc", TRAINER, 3),
        301: (ENV, "rust_env_w0", TRAINER, 3),
        302: (ENV, "rust_env_w1", TRAINER, 11),
        COMPILE: (COMPILE, "python3", TRAINER, 4),
    }
    CMD = {LAUNCHER: "python3\0-m\0main.launcher", TRAINER: "python3\0train.py", ENV: "rust_env_proc\0--fd\x0053",
           COMPILE: "python3\0torch/_inductor/compile_worker/__main__.py"}

    def __init__(self, root: Path) -> None:
        self.root = root
        self.ticks = {tid: 0 for tid in self.THREADS}
        self.state = {tid: "S" for tid in self.THREADS}
        self.wait_ns = {tid: 0 for tid in self.THREADS}
        self.cpu_busy = {c: 0 for c in range(16)}
        self.cpu_total = {c: 0 for c in range(16)}
        self.write()

    def write(self) -> None:
        for tid, (pid, comm, ppid, cpu) in self.THREADS.items():
            d = self.root / str(pid) / "task" / str(tid)
            d.mkdir(parents=True, exist_ok=True)
            (d / "stat").write_text(_stat(tid, comm, ppid, self.ticks[tid], cpu, self.state[tid]))
            (d / "schedstat").write_text(f"{self.ticks[tid] * 10_000_000} {self.wait_ns[tid]} 7\n")
            if tid == pid:
                (self.root / str(pid) / "stat").write_text(_stat(pid, comm, ppid, self.ticks[tid], cpu))
                (self.root / str(pid) / "cmdline").write_text(self.CMD[pid])
                (self.root / str(pid) / "status").write_text("Name:\tx\nCpus_allowed_list:\t0-4,8-12\n")
        lines = ["cpu  0 0 0 0 0 0 0 0"]
        for c in range(16):
            b, t = self.cpu_busy[c], self.cpu_total[c]
            lines.append(f"cpu{c} {b} 0 0 {t - b} 0 0 0 0 0 0")
        (self.root / "stat").write_text("\n".join(lines) + "\n")

    def advance(self, busy: Dict[int, float], *, dt: float = 1.0, cpus: Optional[Dict[int, float]] = None,
                wait: Optional[Dict[int, float]] = None) -> None:
        """``dt`` seconds pass: thread ``tid`` kept ``busy[tid]`` of a CPU busy; logical CPU ``c`` was ``cpus[c]`` busy."""
        for tid, frac in busy.items():
            self.ticks[tid] += round(frac * HZ * dt)
        for tid, s in (wait or {}).items():
            self.wait_ns[tid] += round(s * 1e9)
        for c in range(16):
            self.cpu_total[c] += round(HZ * dt)
            self.cpu_busy[c] += round((cpus or {}).get(c, 0.0) * HZ * dt)
        self.write()


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_every_thread_group_is_named():
    c = T.classify
    assert c("rust_env_w7", 301, ENV, LAUNCHER, TRAINER, "rust_env_proc --fd 5") == "env_workers"
    assert c("rust_env_proc", ENV, ENV, LAUNCHER, TRAINER, "rust_env_proc --fd 5") == "env_main"
    assert c("python3", TRAINER, TRAINER, LAUNCHER, TRAINER, "python3 train.py") == "learner"
    assert c("python3", 203, TRAINER, LAUNCHER, TRAINER, "python3 train.py") == "other", "only the MAIN thread learns"
    assert c("pt_autograd_0", 201, TRAINER, LAUNCHER, TRAINER, "python3 train.py") == "autograd"
    assert c("cuda-EvtHandlr", 202, TRAINER, LAUNCHER, TRAINER, "python3 train.py") == "cuda"
    assert c("python3", COMPILE, COMPILE, LAUNCHER, TRAINER, "python3 torch/_inductor/compile_worker/__main__.py") == "compile"
    assert c("cuda0000", 401, COMPILE, LAUNCHER, TRAINER, "x/compile_worker/y") == "compile", "the process decides"
    assert c("python3", LAUNCHER, LAUNCHER, LAUNCHER, TRAINER, "python3 -m main.launcher") == "launcher"
    assert set(T.GROUPS) == {"env_workers", "env_main", "learner", "autograd", "cuda", "compile", "launcher", "other"}


def test_the_tree_and_the_trainer_are_found(tmp_path):
    fp = FakeProc(tmp_path)
    fp.advance({TRAINER: 5.0, ENV: 9.0, COMPILE: 7.0})      # the env process and a compile worker have MORE cpu
    pids = T.tree_pids(LAUNCHER, str(tmp_path))
    assert pids[0] == LAUNCHER and set(pids) == {LAUNCHER, TRAINER, ENV, COMPILE}
    assert T.trainer_pid(pids, LAUNCHER, str(tmp_path)) == TRAINER
    assert T.tree_pids(999, str(tmp_path)) == []
    assert T.affinity(TRAINER, str(tmp_path)) == [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]


def test_a_thread_name_with_a_parenthesis_parses(tmp_path):
    d = tmp_path / "7" / "task" / "8"
    d.mkdir(parents=True)
    (d / "stat").write_text(_stat(8, "a) (b", 1, 42, 13))
    (d / "schedstat").write_text("1 2500 3\n")
    assert T.read_thread(7, 8, str(tmp_path)) == ("a) (b", 42, 2500, 13)
    assert T.read_thread(7, 9, str(tmp_path)) is None


def test_an_interval_is_ticks_over_elapsed_ticks(tmp_path):
    fp, proc = FakeProc(tmp_path), str(tmp_path)
    pids = T.tree_pids(LAUNCHER, proc)
    a, ca = T.read_threads(pids, LAUNCHER, TRAINER, proc), T.read_cpus(proc)
    fp.advance({301: 1.0, 302: 0.5, TRAINER: 0.25, 201: 0.5, LAUNCHER: 0.1}, dt=2.0, cpus={3: 1.0, 11: 0.5, 5: 1.0},
               wait={301: 0.4})
    b, cb = T.read_threads(pids, LAUNCHER, TRAINER, proc), T.read_cpus(proc)
    row = T.interval_row(2.0, 2.0, a, b, ca, cb, HZ)
    assert row["groups"] == {"env_workers": 1.5, "env_main": 0.0, "learner": 0.25, "autograd": 0.5, "cuda": 0.0,
                             "compile": 0.0, "launcher": 0.1, "other": 0.0}
    assert row["wait"]["env_workers"] == 0.4 and row["wait"]["learner"] == 0.0
    assert row["cpu"]["3"] == 1.0 and row["cpu"]["11"] == 0.5 and row["cpu"]["5"] == 1.0 and row["cpu"]["0"] == 0.0
    assert row["on"] == {"0": 0.1, "1": 0.25, "3": 1.0, "9": 0.5, "11": 0.5}, "by the cpu each thread last ran on"


def test_a_thread_seen_only_once_never_counts(tmp_path):
    fp, proc = FakeProc(tmp_path), str(tmp_path)
    fp.advance({301: 50.0})                              # 5,000 ticks of history before the first reading
    pids = T.tree_pids(LAUNCHER, proc)
    b = T.read_threads(pids, LAUNCHER, TRAINER, proc)
    a = {tid: th for tid, th in b.items() if tid != 301}   # 301 is NEW in b: its history is not this interval's
    row = T.interval_row(1.0, 1.0, a, b, T.read_cpus(proc), T.read_cpus(proc), HZ)
    assert row["groups"]["env_workers"] == 0.0
    gone = T.interval_row(1.0, 1.0, b, a, T.read_cpus(proc), T.read_cpus(proc), HZ)   # 301 vanished
    assert gone["groups"]["env_workers"] == 0.0


def _row(workers: float, learner: float = 0.5, cpu: Optional[Dict[str, float]] = None, wait: float = 0.0,
         dt: float = 1.0) -> Dict[str, Any]:
    g = {k: 0.0 for k in T.GROUPS}
    g.update(env_workers=workers, learner=learner)
    w = {k: 0.0 for k in T.GROUPS}
    w["env_workers"] = wait
    full = {str(c): 0.0 for c in range(16)}
    full.update(cpu or {})
    return {"t": 0.0, "dt": dt, "groups": g, "wait": w, "cpu": full, "on": {}}


def test_the_phase_rule_and_its_excluded_band():
    assert T.phase_of(_row(T.ROLLOUT_MIN)) == "rollout", "the bound itself is a rollout"
    assert T.phase_of(_row(7.9)) == "rollout"
    assert T.phase_of(_row(T.UPDATE_MAX)) == "update", "the bound itself is an update"
    assert T.phase_of(_row(0.0)) == "update"
    for between in (0.11, 0.5, 0.99):
        assert T.phase_of(_row(between)) == "transition"
    assert T.UPDATE_MAX < T.ROLLOUT_MIN


def test_percentile_is_nearest_rank():
    xs = [float(i) for i in range(1, 21)]            # 1..20
    assert T.percentile(xs, 95) == 19.0
    assert T.percentile(xs, 100) == 20.0
    assert T.percentile(xs, 50) == 10.0
    assert T.percentile([3.0], 95) == 3.0
    assert T.percentile([], 95) == 0.0


def test_summary_splits_phases_and_reads_the_peak_core():
    head = {"header": 1, "affinity": [0, 1, 2, 3, 4, 8, 9, 10, 11, 12], "log_start": [1_000_000, 100.0]}
    foot = {"footer": 1, "log_end": [1_140_000, 200.0]}
    rows = [head]
    rows += [_row(4.0, 1.0, {"0": 1.0, "8": 0.5, "1": 0.9, "5": 1.0, "13": 1.0}, wait=0.2) for _ in range(19)]
    rows += [_row(8.0, 1.0, {"0": 1.0, "8": 1.0, "5": 1.0, "13": 1.0}, wait=0.2)]   # one burst
    rows += [_row(0.0, 1.2) for _ in range(30)]
    rows += [_row(0.5, 1.0) for _ in range(2)]                                        # straddles the boundary
    rows.append(foot)
    s = T.summarize(rows)
    assert s["intervals"] == 52 and s["affinity"] == head["affinity"]
    ro, up, tr = s["phases"]["rollout"], s["phases"]["update"], s["phases"]["transition"]
    assert (ro["intervals"], up["intervals"], tr["intervals"]) == (20, 30, 2)
    assert ro["busy_threads"] == {"mean": 5.2, "p95": 5.0, "max": 9.0}
    assert ro["groups"]["env_workers"] == {"mean": 4.2, "p95": 4.0, "max": 8.0}
    assert up["busy_threads"] == {"mean": 1.2, "p95": 1.2, "max": 1.2}
    assert ro["wait_s_per_s"] == 0.2 and up["wait_s_per_s"] == 0.0
    # cpus 0 and 8 are one physical core; the agent cpus 5 and 13 (2.0 together) are NOT the trainer's own
    assert ro["busiest_physical_core"] == {"mean": 1.525, "p95": 1.5, "max": 2.0}
    assert ro["busiest_logical_cpu"]["max"] == 1.0
    assert ro["own_cpus_busy_all_tenants"]["mean"] == pytest.approx((19 * 2.4 + 2.0) / 20, abs=1e-3)
    assert s["fps"] == {"steps": 140_000, "seconds": 100.0, "fps": 1400.0}
    text = T.render(s)
    assert "[rollout] 20.0 s" in text and "[transition] 2.0 s" in text and "fps over the window (the run's log): 1400" in text


def test_the_log_gives_the_last_table(tmp_path):
    log = tmp_path / "launcher_child.full.log"
    log.write_text("|    time_elapsed      | 100     |\n|    total_timesteps   | 1000    |\n"
                   "noise\n|    fps  | 3 |\n|    time_elapsed                | 7.07e+03 |\n|    total_timesteps    | 289936859  |\n")
    assert T.log_progress(log) == (289936859, 7070.0)
    (tmp_path / "empty.log").write_text("nothing here\n")
    assert T.log_progress(tmp_path / "empty.log") is None
    assert T.log_progress(tmp_path / "absent.log") is None


def test_sample_end_to_end_on_the_fake_tree(tmp_path):
    root = tmp_path / "proc"
    root.mkdir()
    fp, clock = FakeProc(root), Clock()
    log = tmp_path / "run.log"
    log.write_text("| time_elapsed | 10 |\n| total_timesteps | 5000 |\n")
    plan = [{301: 1.0, 302: 1.0, TRAINER: 0.5}] * 3 + [{TRAINER: 0.6, 201: 0.4}] * 3

    def sleep(_s: float) -> None:
        i = int(round(clock.t))
        fp.advance(plan[i], cpus={3: 1.0} if i < 3 else {1: 0.6})
        if i == 5:
            log.write_text("| time_elapsed | 16 |\n| total_timesteps | 13400 |\n")
        clock.t += 1.0

    out = tmp_path / "rows.jsonl"
    n = T.sample(LAUNCHER, 6.0, out, log=log, proc=str(root), clock=clock, sleep=sleep, hz=HZ)
    assert n == 6
    rows = T.load(out)
    assert rows[0]["header"] == 1 and rows[0]["trainer"] == TRAINER and rows[0]["log_start"] == [5000, 10.0]
    assert rows[-1]["footer"] == 1 and rows[-1]["log_end"] == [13400, 16.0]
    s = T.summarize(rows)
    assert s["phases"]["rollout"]["intervals"] == 3 and s["phases"]["update"]["intervals"] == 3
    assert s["phases"]["rollout"]["busy_threads"]["mean"] == 2.5
    assert s["phases"]["update"]["groups"]["autograd"]["mean"] == 0.4
    assert s["fps"]["fps"] == 1400.0
    assert json.loads(json.dumps(s)) == s


def _burst_file(tmp_path: Path, plan: List[Dict[int, str]], period: float) -> Path:
    """Run ``burst`` on the fake tree: reading ``i`` sees the run states ``plan[i]`` (every other thread sleeps)."""
    root = tmp_path / "proc"
    root.mkdir()
    fp, clock = FakeProc(root), Clock()
    i = [0]

    def show() -> None:
        fp.state = {tid: plan[i[0]].get(tid, "S") for tid in fp.THREADS} if i[0] < len(plan) else fp.state
        fp.write()

    def sleep(_s: float) -> None:
        i[0] += 1
        clock.t = round(i[0] * period, 6)
        show()

    show()
    out = tmp_path / "burst.jsonl"
    assert T.burst(LAUNCHER, round(len(plan) * period, 6), out, period=period, proc=str(root), clock=clock,
                   sleep=sleep) == len(plan)
    return out


def test_burst_reads_the_run_state_of_the_hot_threads_only(tmp_path):
    # cuda / compile / launcher threads runnable must not count: only the env workers, the learner and autograd do
    plan = [{301: "R", 302: "R", 202: "R", COMPILE: "R", LAUNCHER: "R"}, {TRAINER: "R"}, {201: "R", 203: "R"}, {}]
    rows = T.load(_burst_file(tmp_path, plan, 0.25))
    assert rows[0]["kind"] == "burst" and rows[0]["threads"] == {"env_workers": 2, "learner": 1, "autograd": 1}
    assert rows[1:] == [[0.0, 2, 0, 0], [0.25, 0, 1, 0], [0.5, 0, 0, 1], [0.75, 0, 0, 0]]


def test_burst_summary_gives_the_instantaneous_demand(tmp_path):
    # second 0 is a rollout: 2 bursts of the 2 workers (6 of 10 readings), the learner runnable only between them;
    # second 1 is an update: no worker, the learner and autograd alternating
    both, lrn, auto = {301: "R", 302: "R"}, {TRAINER: "R"}, {201: "R"}
    plan = [lrn, both, both, both, lrn, lrn, both, both, both, lrn] + [lrn, auto] * 5
    s = T.summarize_burst(T.load(_burst_file(tmp_path, plan, 0.1)))
    r, u = s["phases"]["rollout"], s["phases"]["update"]
    assert (r["seconds"], r["readings"], u["seconds"], u["readings"]) == (1, 10, 1, 10)
    assert r["workers_runnable_share"] == {"0": 0.4, "2": 0.6} and r["workers_runnable_mean"] == 1.2
    assert r["threads_runnable_peak"] == 2 and r["threads_runnable_mean"] == 1.6
    assert r["workers_and_learner_together_share"] == 0.0 and r["learner_or_autograd_runnable_share"] == 0.4
    assert r["bursts_per_s"] == 2.0 and r["workers_active_share"] == 0.6 and r["burst_ms_mean"] == 300.0
    assert u["workers_runnable_share"] == {"0": 1.0} and u["threads_runnable_peak"] == 1 and u["bursts_per_s"] == 0.0
    assert s["phases"]["transition"] == {"seconds": 0, "readings": 0}
    assert "[rollout] 1 s: runnable threads peak 2" in T.render_burst(s)
    assert json.loads(json.dumps(s)) == s


def test_burst_refuses_a_dead_launcher(tmp_path):
    (tmp_path / "proc").mkdir()
    with pytest.raises(SystemExit, match="not running"):
        T.burst(4242, 1.0, tmp_path / "o.jsonl", proc=str(tmp_path / "proc"))


def test_sample_refuses_a_dead_launcher(tmp_path):
    (tmp_path / "proc").mkdir()
    with pytest.raises(SystemExit, match="not running"):
        T.sample(4242, 1.0, tmp_path / "o.jsonl", proc=str(tmp_path / "proc"))


def test_the_module_can_only_read():
    """It runs beside a live training run, so it must hold no way to signal, re-pin or renice a process."""
    tree = ast.parse(Path(T.__file__).read_text(encoding="utf-8"))
    banned = {"kill", "killpg", "sched_setaffinity", "setpriority", "nice", "sched_setscheduler", "Popen", "system", "run"}
    called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not (called & banned), sorted(called & banned)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not (imported & {"signal", "subprocess"}), imported
