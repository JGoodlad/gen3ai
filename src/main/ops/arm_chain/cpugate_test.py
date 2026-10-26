"""Regression tests for the GO's CPU gate (cpugate.py) on a FAKE /proc with a fake sleep and monotonic clock."""
import os

from main.ops.arm_chain import cpugate

HZ = cpugate.HZ
SAMPLE_S = 10.0


class FakeBox:
    """/proc/stat per-CPU rows + per-pid stat/status; ``advance`` rewrites them, as the sample window passes."""

    def __init__(self, root, n_cpus=4):
        self.root = str(root / "proc")
        os.makedirs(self.root)
        self.n = n_cpus
        self.now = 0.0
        self.drop = ()
        self.cpu = {c: [0, 0, 0, 0, 0, 0, 0, 0] for c in range(n_cpus)}  # user nice system idle iowait irq softirq steal
        self.procs = {}   # pid -> [utime, stime, comm, affinity]
        self.write()

    def set_cpu(self, c, *, busy, idle):
        self.cpu[c][0] += busy
        self.cpu[c][3] += idle

    def set_proc(self, pid, utime, comm="python", aff="0-3"):
        self.procs[pid] = [utime, 0, comm, aff]

    def write(self):
        with open(os.path.join(self.root, "stat"), "w") as f:
            f.write("cpu  " + " ".join(str(sum(v[i] for v in self.cpu.values())) for i in range(8)) + " 0 0 0\n")
            for c, v in self.cpu.items():
                if c not in self.drop:
                    f.write(f"cpu{c} " + " ".join(map(str, v)) + " 0 0 0\n")
            f.write("intr 0\n")
        for pid, (ut, st, comm, aff) in self.procs.items():
            d = os.path.join(self.root, str(pid))
            os.makedirs(d, exist_ok=True)
            rest = ["S", "1", "1", "1"] + ["0"] * 7 + [str(ut), str(st), "0", "0", "20", "0", "1", "0", "5"]
            with open(os.path.join(d, "stat"), "w") as f:
                f.write(f"{pid} ({comm}) " + " ".join(rest) + "\n")
            with open(os.path.join(d, "status"), "w") as f:
                f.write(f"Name:\t{comm}\nCpus_allowed_list:\t{aff}\nVoluntary:\t1\n")

    def sampler(self, between, **kw):
        """Run cpugate.sample with ``between()`` (mutating this box) executed in the fake sleep."""
        self.write()          # the FIRST sample sees the processes registered so far

        def sleep(s):
            between()
            self.now += s
            self.write()
        base = dict(trainer_cpus="0-1", agent_cpus="2-3", sample_s=SAMPLE_S, proc=self.root, sleep=sleep,
                    monotonic=lambda: self.now)
        base.update(kw)
        return cpugate.sample(**base)


def test_agent_cores_busy_trainer_cores_idle_passes(tmp_path):
    """CB-03: agent cores at 100 % with the trainer cores idle PASS (load1 would have held the GO)."""
    box = FakeBox(tmp_path)
    box.set_proc(50, 0, "agent-job", aff="2-3")

    def between():
        for c in (0, 1):
            box.set_cpu(c, busy=2, idle=98)
        for c in (2, 3):
            box.set_cpu(c, busy=100, idle=0)
        box.procs[50][0] += 10 * HZ          # a full CPU for the whole window, but inside the agent cores

    r = box.sampler(between)
    assert r.ok and r.measured and r.hogs == ()
    assert r.idle_frac == 0.98 and r.line.startswith("PASS")


def test_process_with_wide_affinity_at_60_percent_fails_naming_pid(tmp_path):
    """A process that may run on trainer cores and used 60 % of a CPU fails the gate, naming its pid."""
    box = FakeBox(tmp_path)
    box.set_proc(4242, 0, "hog", aff="0-15")
    box.set_proc(4243, 0, "calm", aff="0-15")

    def between():
        for c in (0, 1):
            box.set_cpu(c, busy=1, idle=99)
        box.procs[4242][0] += 6 * HZ          # 60 % of one CPU over the 10 s window
        box.procs[4243][0] += 2 * HZ          # 20 %

    r = box.sampler(between)
    assert not r.ok and r.measured
    assert len(r.hogs) == 1 and "pid 4242" in r.hogs[0] and "60%" in r.hogs[0]
    assert "4243" not in r.line
    assert r.line.startswith("FAIL") and "BELOW the bar" not in r.line


def test_trainer_cores_half_busy_fails_below_the_bar(tmp_path):
    """Trainer CPUs 50 % idle against a 70 % bar fails with the 'BELOW the bar' reason, measured."""
    box = FakeBox(tmp_path)

    def between():
        for c in (0, 1):
            box.set_cpu(c, busy=50, idle=50)

    r = box.sampler(between)
    assert not r.ok and r.measured and r.idle_frac == 0.5
    assert "BELOW the bar" in r.line and r.hogs == ()


def test_iowait_counts_as_idle(tmp_path):
    """The meter is idle + iowait over all jiffies (a trainer waiting on disk is not a busy CPU)."""
    box = FakeBox(tmp_path)

    def between():
        for c in (0, 1):
            box.cpu[c][4] += 90          # iowait
            box.cpu[c][0] += 10

    r = box.sampler(between)
    assert r.ok and r.idle_frac == 0.9


def test_proc_stat_lacking_a_trainer_cpu_is_not_measured(tmp_path):
    """A /proc/stat without a trainer CPU cannot be measured: the gate fails closed rather than passing."""
    box = FakeBox(tmp_path)
    box.drop = (1,)
    r = box.sampler(lambda: None, trainer_cpus="0-1")
    assert r.ok is False and r.measured is False
    assert "CANNOT MEASURE" in r.line and "[1]" in r.line


def test_own_pid_is_excluded_and_a_hog_born_during_the_window_is_named(tmp_path):
    """``me`` is never a hog; a pid born inside the window is scored on its whole CPU count (it used to be skipped)."""
    box = FakeBox(tmp_path)
    box.set_proc(7, 0, "me", aff="0-15")

    def between():
        for c in (0, 1):
            box.set_cpu(c, busy=1, idle=99)
        box.procs[7][0] += 9 * HZ
        box.set_proc(8, 9 * HZ, "newborn", aff="0-15")

    r = box.sampler(between, me=7)
    assert not r.ok and len(r.hogs) == 1 and "pid 8" in r.hogs[0] and "newborn" in r.hogs[0]


def test_parse_cpu_list():
    """CPU lists are the kernel's ``a-b,c`` syntax."""
    assert cpugate.parse_cpu_list("0-2,5,8-9") == {0, 1, 2, 5, 8, 9}
    assert cpugate.parse_cpu_list("") == set()
