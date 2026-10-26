"""Regression tests for live-run discovery from launcher argv (live.py) on a FAKE /proc tree."""
import os

from main.ops.arm_chain import live
from utils import procfs

TRAINER = "train_rl_agent" + ".py"  # concatenated: root rule 3


class FakeProc:
    """A /proc tree under tmp_path: per pid a stat, cmdline, and optional cwd / fd/1 symlinks; plus uptime."""

    def __init__(self, root, uptime=1000.0):
        self.root = str(root / "proc")
        os.makedirs(self.root)
        with open(os.path.join(self.root, "uptime"), "w") as f:
            f.write(f"{uptime} 0.0\n")

    def add(self, pid, argv, ppid=1, cwd=None, stdout=None, start_s=400, state="S"):
        d = os.path.join(self.root, str(pid))
        os.makedirs(os.path.join(d, "fd"))
        # fields after "(comm)": state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime
        # cutime cstime prio nice threads itrealvalue starttime  (utils.procfs.read_stat indexes these)
        rest = [state, str(ppid), str(pid), str(pid)] + ["0"] * 7 + ["0", "0", "0", "0"] + ["20", "0", "1", "0"]
        rest.append(str(int(start_s * procfs.HZ)))
        rest += ["0"] * 5
        with open(os.path.join(d, "stat"), "w") as f:
            f.write(f"{pid} (python) " + " ".join(rest) + "\n")
        with open(os.path.join(d, "cmdline"), "wb") as f:
            f.write(b"\0".join(a.encode() for a in argv) + b"\0")
        if cwd:
            os.symlink(cwd, os.path.join(d, "cwd"))
        if stdout:
            os.symlink(stdout, os.path.join(d, "fd", "1"))


def launcher_argv(*extra):
    return ["python", "-m", "main.launcher", *extra]


def setup(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    return FakeProc(tmp_path), str(models)


def test_fake_proc_stat_is_read_back_exactly(tmp_path):
    """The fixture itself: utils.procfs reads pid / ppid / starttime from the fake stat as written."""
    fp, _ = setup(tmp_path)
    fp.add(77, ["x"], ppid=5, start_s=123)
    st = procfs.read_stat(77, fp.root)
    assert (st.pid, st.ppid, st.state, st.starttime) == (77, 5, "S", 123 * procfs.HZ)


def test_run_name_of_a_family_no_glob_knew_resolves(tmp_path):
    """CB-22: a brand-new family resolves from the launcher's --run-name, with no name pattern anywhere."""
    fp, models = setup(tmp_path)
    os.makedirs(os.path.join(models, "rb_brandnew_family_s1"))
    fp.add(100, launcher_argv("--run-name", "rb_brandnew_family_s1", "--steps", "9"), start_s=400,
           stdout=str(tmp_path / "out.log"))
    runs = live.discover(models, fp.root)
    assert len(runs) == 1
    r = runs[0]
    assert r.run_dir == os.path.realpath(os.path.join(models, "rb_brandnew_family_s1"))
    assert r.source == "argv:--run-name"
    assert r.elapsed_s == 600 and r.pid == 100
    assert r.stdout == str(tmp_path / "out.log")


def test_run_dir_flag_resolves_absolute_and_relative(tmp_path):
    """--run-dir D resolves; a relative D is joined to the launcher's cwd."""
    fp, models = setup(tmp_path)
    d = tmp_path / "somewhere" / "runA"
    d.mkdir(parents=True)
    fp.add(100, launcher_argv("--run-dir", str(d)))
    fp.add(101, launcher_argv("--run-dir=somewhere/runA"), cwd=str(tmp_path))
    by_pid = {r.pid: r for r in live.discover(models, fp.root)}
    assert by_pid[100].run_dir == os.path.realpath(str(d)) and by_pid[100].source == "argv:--run-dir"
    assert by_pid[101].run_dir == os.path.realpath(str(d))


def test_restart_from_checkpoint_resolves_to_its_run(tmp_path):
    """A RESTART (--model <run>/checkpoints/checkpoint_N_steps.zip) is the checkpoint's run, labelled argv:--model."""
    fp, models = setup(tmp_path)
    ck = os.path.join(models, "R", "checkpoints")
    os.makedirs(ck)
    fp.add(100, launcher_argv("--model", os.path.join(ck, "checkpoint_5_steps.zip"), "--steps", "9"))
    (r,) = live.discover(models, fp.root)
    assert r.run_dir == os.path.realpath(os.path.join(models, "R")) and r.source == "argv:--model"


def test_model_not_checkpoint_shaped_is_unresolved(tmp_path):
    """A --model that is not <run>/checkpoints/<ckpt> names no run: UNRESOLVED, never guessed."""
    fp, models = setup(tmp_path)
    fp.add(100, launcher_argv("--model", os.path.join(models, "R", "final.zip")))
    (r,) = live.discover(models, fp.root)
    assert r.run_dir is None and r.reason


def test_dry_run_launcher_is_ignored(tmp_path):
    """A --dry-run launcher creates nothing and is not a live run."""
    fp, models = setup(tmp_path)
    os.makedirs(os.path.join(models, "R"))
    fp.add(100, launcher_argv("--dry-run", "--run-name", "R"))
    assert live.discover(models, fp.root) == []


def test_trainer_child_wins_over_launcher_argv(tmp_path):
    """The trainer child's --run-dir is authoritative, even when the launcher's own argv says otherwise."""
    fp, models = setup(tmp_path)
    x = tmp_path / "X"
    x.mkdir()
    os.makedirs(os.path.join(models, "other"))
    fp.add(100, launcher_argv("--run-name", "other"))
    fp.add(101, ["python", "src/main/" + TRAINER, "--run-dir", str(x)], ppid=100)
    runs = live.discover(models, fp.root)
    assert [(r.pid, r.run_dir, r.source) for r in runs] == [(100, os.path.realpath(str(x)), "child")]


def test_nothing_resolvable_is_unresolved_with_a_reason_not_dropped(tmp_path):
    """A launcher whose argv names no run is reported UNRESOLVED (a dropped run is an unlogged run)."""
    fp, models = setup(tmp_path)
    fp.add(100, launcher_argv("--steps", "5"))
    (r,) = live.discover(models, fp.root)
    assert r.run_dir is None and "no trainer child yet" in r.reason
    assert "UNRESOLVED" in r.line()


def test_missing_run_dir_is_unresolved(tmp_path):
    """A named run dir that does not exist (yet) is UNRESOLVED with the path in the reason."""
    fp, models = setup(tmp_path)
    fp.add(100, launcher_argv("--run-name", "not_created_yet"))
    (r,) = live.discover(models, fp.root)
    assert r.run_dir is None and "not_created_yet" in r.reason and "not a directory" in r.reason


def test_trainer_without_launcher_is_orphan(tmp_path):
    """A trainer with no live launcher ancestor is reported as an orphan-trainer."""
    fp, models = setup(tmp_path)
    x = tmp_path / "X"
    x.mkdir()
    fp.add(200, ["python", TRAINER, "--run-dir", str(x)], ppid=1, stdout=str(tmp_path / "t.log"))
    (r,) = live.discover(models, fp.root)
    assert r.source == "orphan-trainer" and r.run_dir == os.path.realpath(str(x)) and r.pid == 200
    assert r.stdout == str(tmp_path / "t.log")


def test_trainer_under_dry_run_launcher_is_orphan(tmp_path):
    """A trainer whose only launcher ancestor is a --dry-run (not a launcher here) is still reported, as an orphan."""
    fp, models = setup(tmp_path)
    x = tmp_path / "X"
    x.mkdir()
    fp.add(100, launcher_argv("--dry-run"))
    fp.add(101, ["python", TRAINER, "--run-dir", str(x)], ppid=100)
    runs = live.discover(models, fp.root)
    assert [r.source for r in runs] == ["orphan-trainer"]


def test_stdout_comes_from_fd1(tmp_path):
    """The alert watch follows the launcher's stdout: read from fd/1's target."""
    fp, models = setup(tmp_path)
    os.makedirs(os.path.join(models, "R"))
    log = tmp_path / "launcher.log"
    fp.add(100, launcher_argv("--run-name", "R"), stdout=str(log))
    fp.add(101, launcher_argv("--run-name", "R"))   # no fd/1 at all
    by_pid = {r.pid: r for r in live.discover(models, fp.root)}
    assert by_pid[100].stdout == str(log)
    assert by_pid[101].stdout == ""
