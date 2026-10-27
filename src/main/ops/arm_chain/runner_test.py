"""Regression tests for the environments the chain runs things in, and the subprocess seam (runner.py)."""
import os
import signal

import pytest

from main.ops.arm_chain import procwatch, runner

BASE = {"HOME": "/nonexistent/userhome", "PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "", "GEN3AI_TEST_ALLOW_GPU": "1",
        "PYTHONPATH": "/worktree/src", "GEN3AI_BLOCKING": "stale", "SECRET_TOKEN": "x", "GEN3AI_PYTHON": "/py"}


def test_go_env_drops_cuda_empty_and_allow_gpu_and_pins_pythonpath():
    """CB-02 (toy_x47 3602) + H-19: a GO inherited CUDA_VISIBLE_DEVICES='' from the arm-end CPU steps and saw no GPU."""
    env = runner.go_env(BASE, "/repo", "/tok/file", "/chain/python")
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert "GEN3AI_TEST_ALLOW_GPU" not in env
    assert env["PYTHONPATH"] == "/repo/src"
    assert env["GEN3AI_GPU_LEASE_TOKEN_FILE"] == "/tok/file"
    assert "GEN3AI_BLOCKING" not in env       # the trainer would inherit it and skip the heavy slots
    assert env["HOME"] == "/nonexistent/userhome" and env["PATH"] == "/usr/bin"
    assert env["GEN3AI_PYTHON"] == "/chain/python"   # the chain file's interpreter, never the supervisor's
    assert "SECRET_TOKEN" not in env          # an allow-list: nothing else leaks into a launch
    runner.assert_card_visible(env)


def test_go_env_does_not_mutate_base():
    """The base environment (the supervisor's own) is read, never edited."""
    base = dict(BASE)
    runner.go_env(base, "/repo", "/t", "b")
    assert base == BASE


def test_assert_card_visible():
    """CB-02 belt: an EMPTY CUDA_VISIBLE_DEVICES is refused; unset or '0' passes."""
    with pytest.raises(runner.CardNotVisible):
        runner.assert_card_visible({"CUDA_VISIBLE_DEVICES": ""})
    runner.assert_card_visible({})
    runner.assert_card_visible({"CUDA_VISIBLE_DEVICES": "0"})


def test_cpu_env_hides_the_card_and_sets_blocking_only_when_asked():
    """Root rule 5: a CPU step has CUDA_VISIBLE_DEVICES=''; GEN3AI_BLOCKING appears only when the step blocks the GPU."""
    env = runner.cpu_env(BASE, "/repo")
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["PYTHONPATH"] == "/repo/src"
    assert "GEN3AI_TEST_ALLOW_GPU" not in env
    assert "GEN3AI_BLOCKING" not in env           # the stale inherited value is removed too
    blocking = runner.cpu_env(BASE, "/repo", blocking="arm end")
    assert blocking["GEN3AI_BLOCKING"] == "arm end"
    assert BASE["CUDA_VISIBLE_DEVICES"] == "" and BASE["GEN3AI_BLOCKING"] == "stale"


def test_agent_prefix_is_nice_and_taskset():
    """Root rule 5b: agent CPU work is niced and pinned to the agent cores."""
    assert runner.agent_prefix("5-7,13-15", 19) == ["nice", "-n", "19", "taskset", "-c", "5-7,13-15"]


def test_tail(tmp_path):
    """tail reports the last lines, or a marker when there is no file."""
    f = tmp_path / "o.txt"
    f.write_text("a\nb\nc\n")
    assert runner.tail(str(f), 2) == "b c"
    assert runner.tail(str(tmp_path / "nope")).startswith("(no ")


@pytest.mark.integration
def test_run_returns_rc_and_writes_output(tmp_path):
    """run writes stdout+stderr to out_path and returns the child's own exit code."""
    out = tmp_path / "sub" / "o.txt"
    rc = runner.SubprocessRunner().run(["sh", "-c", "echo hi; echo err >&2; exit 3"], cwd=str(tmp_path),
                                       env={"PATH": os.environ["PATH"]}, timeout_s=10, out_path=str(out))
    assert rc == 3
    assert set(out.read_text().split()) == {"hi", "err"}


@pytest.mark.integration
def test_run_returns_124_on_timeout_and_says_so(tmp_path):
    """A step past its wall timeout returns 124 and the transcript records the TIMEOUT."""
    out = tmp_path / "o.txt"
    rc = runner.SubprocessRunner().run(["sleep", "30"], cwd=str(tmp_path), env={"PATH": os.environ["PATH"]},
                                       timeout_s=0.3, out_path=str(out))
    assert rc == 124
    assert "TIMEOUT" in out.read_text()


@pytest.mark.integration
def test_spawn_is_in_its_own_session_and_identified(tmp_path):
    """spawn detaches with setsid (a supervisor's death cannot signal the run's group) and returns a live identity."""
    ref = runner.SubprocessRunner().spawn(["sleep", "30"], cwd=str(tmp_path), env={"PATH": os.environ["PATH"]},
                                          out_path=str(tmp_path / "x" / "sp.log"))
    try:
        assert os.getsid(ref.pid) == ref.pid
        assert procwatch.alive(ref)
        assert ref == procwatch.ref_of(ref.pid)
    finally:
        os.kill(ref.pid, signal.SIGKILL)       # our own child, by explicit pid
    # spawn's own reaper thread waits it (a waitpid here would race that thread): it must read as gone, reaped
    assert procwatch.wait_gone(ref, 10, poll_s=0.05)
    import time
    for _ in range(200):
        if not os.path.exists(f"/proc/{ref.pid}"):
            break
        time.sleep(0.05)
    assert not os.path.exists(f"/proc/{ref.pid}") or procwatch.ref_of(ref.pid) != ref


@pytest.mark.integration
def test_a_timeout_ends_the_whole_process_group_not_just_the_child(tmp_path):
    """Review 10-26, finding 5: ``subprocess.run``'s timeout SIGKILLed bash only; the kit's python children (a
    launcher ``--dry-run`` holding a CUDA context) survived and failed the next GO's GPU-idle check."""
    import uuid
    from utils import procfs
    mark = f"armchain_orphan_{uuid.uuid4().hex[:10]}"
    rc = runner.SubprocessRunner().run(["bash", "-c", f"(exec -a {mark} sleep 60) & wait"], cwd=str(tmp_path),
                                       env={"PATH": os.environ["PATH"]}, timeout_s=1.0, out_path=str(tmp_path / "o"))
    left = [pid for pid in procfs.snapshot() if procfs.read_text(pid, "cmdline").startswith(mark)]
    for pid in left:                           # this test's own stand-in, by explicit pid, if the fix regressed
        os.kill(pid, signal.SIGKILL)
    assert rc == 124 and left == [], left
    assert "process group" in (tmp_path / "o").read_text()
