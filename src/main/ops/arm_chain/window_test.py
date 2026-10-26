"""Regression tests for the GPU window between two arms (window.py): preconditions, one process per step, GPU idle after."""
import os
import subprocess

import pytest

from main.ops.arm_chain import spec as specmod
from main.ops.arm_chain import window

pytestmark = pytest.mark.integration

GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}
LEASE_OK = "lease owner 'trainer' holds the GPU (token abc)"


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True,
                          env=GIT_ENV).stdout.strip()


def make_repo(path):
    path.mkdir(parents=True)
    git(path, "init", "-q")
    (path / "f").write_text("x")
    git(path, "add", "f")
    git(path, "commit", "-q", "-m", "c1")
    return git(path, "rev-parse", "HEAD")


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeRunner:
    """Records argv / env / cwd; returns scripted rcs; writes its out_path; optional per-call side effect."""

    def __init__(self, rcs=(), clock=None, advance_s=0.0, effects=None):
        self.calls, self.rcs, self.clock, self.advance_s, self.effects = [], list(rcs), clock, advance_s, effects or {}

    def run(self, argv, *, cwd, env, timeout_s, out_path):
        i = len(self.calls)
        self.calls.append({"argv": list(argv), "cwd": cwd, "env": dict(env), "timeout_s": timeout_s})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(f"output of call {i}\n")
        if i in self.effects:
            self.effects[i]()
        if self.clock:
            self.clock.t += self.advance_s
        return self.rcs[i] if i < len(self.rcs) else 0

    def spawn(self, *a, **k):  # pragma: no cover - never used by a window
        raise AssertionError("a window never spawns")


class Env:
    """A chain spec + window whose worktree is a temp git repo at a commit."""

    def __init__(self, tmp_path, steps=None, budget_min=30.0):
        self.tmp = tmp_path
        self.repo = tmp_path / "main_repo"
        self.repo.mkdir()
        self.wt = tmp_path / "wt"
        self.sha = make_repo(self.wt)
        self.job = tmp_path / "job"
        self.job.mkdir()
        steps = steps or [{"name": "gpu1", "argv": ["pytest", "-k", "a"], "mem_gb": 8, "timeout_s": 100},
                          {"name": "gpu2", "argv": ["pytest", "-k", "b"], "mem_gb": 12, "timeout_s": 200}]
        d = {"schema": specmod.SCHEMA, "name": "c", "job_dir": str(self.job), "repo": str(self.repo), "python": "/py",
             "backend": {"kind": "x35", "launch_script": "/l", "stage": "/s", "lease_owner": "trainer",
                         "lease_token_file": "/tok/file"},
             "records": {"stage": "/r"},
             "items": [{"kind": "window", "name": "pre", "worktree": str(self.wt), "require_commit": self.sha[:12],
                        "budget_min": budget_min, "on_fail": "hold", "preempt_marker": "PREEMPT", "steps": steps}]}
        self.spec = specmod.parse(d)
        self.w = self.spec.items[0]
        self.marker = self.job / "PREEMPT"
        self.said = []

    def run(self, runner, idle=((True, "idle"),), lease=LEASE_OK, clock=None, **kw):
        seq = list(idle)

        def probe():
            return seq.pop(0) if len(seq) > 1 else seq[0]
        c = clock or FakeClock()
        return window.run_window(self.spec, self.w, runner, {"HOME": "/h", "CUDA_VISIBLE_DEVICES": ""}, probe,
                                 lambda: lease, self.said.append, monotonic=c, sleep=c.sleep, idle_wait_s=0, **kw)


# -- nvidia_idle --------------------------------------------------------------------------------------------

def test_nvidia_smi_failing_is_not_an_idle_gpu(monkeypatch):
    """H-08: a failing nvidia-smi (rc 9) used to read as 'no compute apps' = idle."""
    monkeypatch.setattr(window.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 9, stdout="", stderr="boom"))
    ok, why = window.nvidia_idle()
    assert ok is False and "rc 9" in why


def test_nvidia_smi_missing_is_not_an_idle_gpu(monkeypatch):
    """H-08: no nvidia-smi binary is not an idle GPU."""
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(window.subprocess, "run", boom)
    ok, why = window.nvidia_idle()
    assert ok is False and "FileNotFoundError" in why


def test_nvidia_smi_timeout_is_not_an_idle_gpu(monkeypatch):
    """H-08: a hanging nvidia-smi is not an idle GPU."""
    def hang(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 1)
    monkeypatch.setattr(window.subprocess, "run", hang)
    assert window.nvidia_idle()[0] is False


def test_nvidia_smi_empty_ok_is_idle_and_listed_pids_are_not(monkeypatch):
    """A successful nvidia-smi with no compute apps is idle; with pids it names them."""
    monkeypatch.setattr(window.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="\n"))
    assert window.nvidia_idle() == (True, "idle")
    monkeypatch.setattr(window.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0, stdout="12\n34\n"))
    ok, why = window.nvidia_idle()
    assert ok is False and "12 34" in why


# -- run_window preconditions ---------------------------------------------------------------------------------

def test_preempt_marker_present_runs_nothing(tmp_path):
    """SOP §3: the preempt marker present means the window is NOT run."""
    e = Env(tmp_path)
    e.marker.write_text("")
    r = FakeRunner()
    res = e.run(r)
    assert res.ok is False and r.calls == [] and "preempt marker" in res.lines[0]


def test_lease_of_another_owner_runs_nothing(tmp_path):
    """The lease must be the chain's owner's; another owner's lease is a failed precondition."""
    e = Env(tmp_path)
    r = FakeRunner()
    res = e.run(r, lease="lease owner 'someone_else' holds the GPU")
    assert res.ok is False and r.calls == [] and "not trainer's" in res.lines[0]


def test_dirty_worktree_runs_nothing(tmp_path):
    """A worktree with uncommitted files is not the commit the window registered."""
    e = Env(tmp_path)
    (e.wt / "stray.py").write_text("x")
    r = FakeRunner()
    res = e.run(r)
    assert res.ok is False and r.calls == [] and "not clean" in res.lines[0]


def test_worktree_lacking_required_commit_runs_nothing(tmp_path):
    """A worktree that does not contain the required commit fails the precondition."""
    e = Env(tmp_path)
    e.w = type(e.w)(**{**e.w.__dict__, "require_commit": "1" * 40})
    r = FakeRunner()
    res = e.run(r)
    assert res.ok is False and r.calls == [] and "lacks" in res.lines[0]


def test_gpu_busy_before_runs_nothing(tmp_path):
    """The GPU must be idle before the window (H-08 meter) or nothing runs."""
    e = Env(tmp_path)
    r = FakeRunner()
    res = e.run(r, idle=[(False, "compute process(es) 99")])
    assert res.ok is False and r.calls == [] and "not idle" in res.lines[0]


# -- run_window steps -----------------------------------------------------------------------------------------

def test_gpu_step_argv_and_env_one_process_per_step(tmp_path):
    """CB-19: each GPU step is its OWN process under gpu_lock.sh + mem_cap.sh with its own timeout; env sees the card."""
    e = Env(tmp_path)
    r = FakeRunner()
    res = e.run(r)
    assert res.ok is True and len(r.calls) == 2
    ops = os.path.join(str(e.repo), "scripts", "ops")
    assert r.calls[0]["argv"] == [f"{ops}/gpu_lock.sh", f"{ops}/mem_cap.sh", "8", "timeout", "100", "pytest", "-k", "a"]
    assert r.calls[1]["argv"] == [f"{ops}/gpu_lock.sh", f"{ops}/mem_cap.sh", "12", "timeout", "200", "pytest", "-k", "b"]
    for c in r.calls:
        assert "CUDA_VISIBLE_DEVICES" not in c["env"]
        assert c["env"]["GEN3AI_TEST_ALLOW_GPU"] == "1"
        assert c["env"]["GEN3AI_GPU_LEASE_TOKEN_FILE"] == "/tok/file"
        assert c["cwd"] == str(e.wt) and c["env"]["PYTHONPATH"] == f"{e.wt}/src"
    assert r.calls[0]["timeout_s"] == 700        # the wall timeout plus the 10-min cap-scope grace


def test_cpu_step_keeps_card_hidden_and_no_gpu_wrappers(tmp_path):
    """A gpu=false step runs under timeout only, with CUDA_VISIBLE_DEVICES='' and no GPU allowance (root rule 5)."""
    e = Env(tmp_path, steps=[{"name": "cpu1", "argv": ["echo", "x"], "mem_gb": 1, "timeout_s": 5, "gpu": False,
                              "cwd": "repo"}])
    r = FakeRunner()
    assert e.run(r).ok
    c = r.calls[0]
    assert c["argv"] == ["timeout", "5", "echo", "x"] and c["cwd"] == str(e.repo)
    assert c["env"]["CUDA_VISIBLE_DEVICES"] == "" and "GEN3AI_TEST_ALLOW_GPU" not in c["env"]


def test_failed_step_fails_window_but_later_steps_still_run(tmp_path):
    """One process per step: step 1's rc 3 is reported and step 2 still runs (a failure does not poison the rest)."""
    e = Env(tmp_path)
    r = FakeRunner(rcs=[3, 0])
    res = e.run(r)
    assert res.ok is False and len(r.calls) == 2
    assert "rc=3" in res.lines[0] and "rc=0" in res.lines[1]


def test_past_budget_marks_later_steps_not_run(tmp_path):
    """A step not STARTED inside budget_min is NOT RUN; the window fails."""
    e = Env(tmp_path, budget_min=30)
    clock = FakeClock()
    r = FakeRunner(clock=clock, advance_s=3600)         # step 1 takes 60 min against a 30-min budget
    res = e.run(r, clock=clock)
    assert res.ok is False and len(r.calls) == 1
    assert any("gpu2: NOT RUN (past the 30-min budget)" in ln for ln in res.lines)


def test_marker_created_by_step_one_stops_step_two(tmp_path):
    """The preempt marker is read before EVERY step: one that appears mid-window stops the rest."""
    e = Env(tmp_path)
    r = FakeRunner(effects={0: lambda: e.marker.write_text("")})
    res = e.run(r)
    assert res.ok is False and len(r.calls) == 1
    assert any("preempt marker appeared" in ln for ln in res.lines)


def test_gpu_not_idle_after_last_step_fails_window(tmp_path):
    """A guest left on the GPU after the window would OOM the next arm: the window FAILS."""
    e = Env(tmp_path)
    r = FakeRunner()
    res = e.run(r, idle=[(True, "idle"), (False, "compute process(es) 777")])
    assert len(r.calls) == 2 and res.ok is False
    assert any("not idle after the window" in ln and "777" in ln for ln in res.lines)
