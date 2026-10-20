"""The launcher's CUDA-VISIBILITY PREFLIGHT (`main/launcher/cuda_visibility_gate.py`).

THE INCIDENT (2026-10-20 11:50). An X35 launch passed `--dry-run ... --device cuda`, then its child died at startup with
"No CUDA GPUs are available": the launch env carried `CUDA_VISIBLE_DEVICES=""`, inherited from a CPU-only end-of-run
chain. What is pinned here: the real launch path (`_prepare_session`) and `--dry-run` both refuse `FATAL_CONFIG`, naming
the variable, when the device is explicit cuda and the env the child gets hides every GPU - BEFORE any worktree or run
dir exists - and neither refuses a cpu / auto launch, an unset variable, or a list that leaves a device visible. Every
case sets the env EXPLICITLY (the host's own value never decides a verdict).
"""
import ast
import inspect

import pytest

import main.launcher.cuda_visibility_gate as gate
import main.launcher.dry_run as dry_run_mod
from main.exit_codes import TrainExitCode
from main.launcher.submodule_gate_test import _FRESH, _git, _make_repo, _session  # the throwaway-repo harness

HIDING = ["", " ", "-1", "NoDevFiles", ",0", "x,0", "  "]
VISIBLE = ["0", "1", "0,1", "GPU-1234abcd", "MIG-GPU-1/1/0", "0,-1", "1,NoDevFiles"]


def _set_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(gate.ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(gate.ENV_VAR, value)


# ---------------------------------------------------------------------------------------
# the pure verdict
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", HIDING)
def test_a_value_that_hides_every_device_refuses_an_explicit_cuda_device(value):
    for dev in ("cuda", "cuda:0", "CUDA"):
        v = gate.verdict_for_launch(dev, {gate.ENV_VAR: value})
        assert v.hidden and v.refused, (dev, value)
        assert gate.ENV_VAR in v.reason and "No CUDA GPUs are available" in v.reason
        assert len(v.lines()) == 1, "a typed ONE-line reason"


@pytest.mark.parametrize("value", VISIBLE + [None])
def test_a_value_that_leaves_a_device_visible_or_unset_never_refuses(value):
    env = {} if value is None else {gate.ENV_VAR: value}
    assert not gate.verdict_for_launch("cuda", env).refused


@pytest.mark.parametrize("dev", [None, "", "auto", "cpu", "mps"])
def test_a_non_cuda_device_is_never_refused_even_when_the_env_hides_every_gpu(dev):
    v = gate.verdict_for_launch(dev, {gate.ENV_VAR: ""})
    assert v.hidden and not v.refused
    assert v.lines()[0].startswith("cuda devices : ⚠")


def test_the_gate_stays_cuda_free():
    """It reads the env only: no torch, no subprocess (`nvidia-smi`), no NVML."""
    tree = ast.parse(inspect.getsource(gate))
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not imported & {"torch", "subprocess", "pynvml", "utils"}, imported


# ---------------------------------------------------------------------------------------
# the REAL launch path refuses before anything exists
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["", "-1"])
@pytest.mark.parametrize("flag", [["--device", "cuda"], ["--device=cuda"]])
def test_the_real_launch_refuses_FATAL_CONFIG_and_creates_nothing(value, flag, tmp_path, run_archive, monkeypatch, capsys):
    root, sha = _make_repo(tmp_path, "repo", state="ok")
    _set_env(monkeypatch, value)
    created = []
    worktrees_before = _git(root, "worktree", "list")
    with pytest.raises(SystemExit) as e:
        _session(root, sha, monkeypatch, tmp_path, created=created, argv=_FRESH + flag)
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    err = capsys.readouterr().err
    assert gate.ENV_VAR in err and "--device cuda" in err
    assert created == [], "a refusal must come BEFORE the worktree"
    assert list(run_archive.iterdir()) == [], "...and before the run dir"
    assert _git(root, "worktree", "list") == worktrees_before


@pytest.mark.parametrize("value,flag", [
    (None, ["--device", "cuda"]),            # nothing hidden
    ("0", ["--device", "cuda"]),             # a device left visible
    ("", ["--device", "cpu"]),               # hidden, but the launch is cpu
    ("", ["--device", "auto"]),              # hidden, auto falls to cpu in the trainer
    ("", []),                                # hidden, no --device (auto)
])
def test_the_real_launch_proceeds_when_nothing_is_contradicted(value, flag, tmp_path, run_archive, monkeypatch):
    root, sha = _make_repo(tmp_path, "repo", state="ok")
    _set_env(monkeypatch, value)
    created = []
    _session(root, sha, monkeypatch, tmp_path, created=created, argv=_FRESH + flag)
    assert created == [sha]


# ---------------------------------------------------------------------------------------
# --dry-run prints the same verdict and fails
# ---------------------------------------------------------------------------------------

def _dry(root, monkeypatch, tmp_path, flag):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dry_run_mod, "get_repo_root", lambda *a, **k: root)
    lines = []
    # --allow-desktop-gpu: the host's own display state must not decide this file's verdicts.
    code = dry_run_mod.dry_run(
        _FRESH + ["--allow-desktop-gpu"] + flag, interval_hours=0.0, pin=True, sync_to_main=False, pin_commit=None,
        grace_minutes=1.0, max_crash_restarts=0, nice=0, out=lines.append)
    return code, "\n".join(lines)


@pytest.mark.parametrize("value", ["", "-1"])
def test_dry_run_refuses_a_cuda_launch_that_cannot_see_a_gpu(value, tmp_path, run_archive, monkeypatch):
    root, _sha = _make_repo(tmp_path, "repo", state="ok")
    _set_env(monkeypatch, value)
    code, text = _dry(root, monkeypatch, tmp_path, ["--device", "cuda"])
    assert code == int(TrainExitCode.FATAL_CONFIG), text
    assert "cuda devices : ✗ REFUSED" in text and gate.ENV_VAR in text
    assert "would NOT launch" in text


@pytest.mark.parametrize("value,flag", [(None, ["--device", "cuda"]), ("0", ["--device", "cuda"]),
                                        ("", ["--device", "cpu"])])
def test_dry_run_passes_when_nothing_is_contradicted(value, flag, tmp_path, run_archive, monkeypatch):
    root, _sha = _make_repo(tmp_path, "repo", state="ok")
    _set_env(monkeypatch, value)
    code, text = _dry(root, monkeypatch, tmp_path, flag)
    assert code == 0, text
    assert "cuda devices :" in text and "✗ REFUSED" not in text.split("cuda devices :")[1].split("\n")[0]
