"""INTERPRETER ↔ RUN SAFETY (`main.launcher.torch_runtime`, owner 2026-09-30: torch 2.8 is the
default, 2.5.1 legacy) — a resume or fork runs under the torch its run recorded, the legacy env is
SELECTED for an unrecorded run, and a mismatch is a typed FATAL_CONFIG refusal unless consented.

Every interpreter here is a fake: a shell script that prints a torch version for the probe, so the
tests read the same on a box with one env, two, or none. Reverting the guard (the launcher spawning
`resolve_child_python()` regardless of the run's record) fails (c), (d), (e) and (f).
"""
from __future__ import annotations

import importlib
import json
import os
import sys

import pytest

import main.launcher.torch_runtime as tr
from main.exit_codes import TrainExitCode
from main.launcher.checkpoint import _strip_launcher_args
from main.launcher.child import PYTHON_ENV_VAR
from main.launcher.state import LauncherState

#: The module, not the `run` FUNCTION `main.launcher` re-exports under the same name.
launcher_run = importlib.import_module("main.launcher.run")
FATAL = int(TrainExitCode.FATAL_CONFIG)


def _fake_python(path, torch_version):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho {torch_version}\n")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A conda layout: `<base>/envs/gen3ai_torch28` (the launcher's own, torch 2.8) and
    `<base>/envs/gen3ai_stable` (legacy, torch 2.5.1). `sys.prefix` points at the 2.8 env, so the
    sibling lookup runs exactly as on the real box."""
    envs = tmp_path / "conda" / "envs"
    new = _fake_python(envs / "gen3ai_torch28" / "bin" / "python3", "2.8.0+cu126")
    old = _fake_python(envs / "gen3ai_stable" / "bin" / "python3", "2.5.1+cu121")
    monkeypatch.setattr(tr.sys, "prefix", str(envs / "gen3ai_torch28"))
    monkeypatch.setattr(tr, "resolve_child_python", lambda: new)
    monkeypatch.delenv(PYTHON_ENV_VAR, raising=False)
    monkeypatch.setattr(tr, "_probe_cache", {})
    return {"new": new, "old": old, "envs": envs}


def _run(tmp_path, torch_version=None, name="run"):
    run_dir = tmp_path / "models" / name
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / "checkpoint_1000_steps.zip"
    ckpt.write_text("x")
    meta = {"git_hash": "aaaa1111"}
    if torch_version is not None:
        meta["torch_version"] = torch_version
    (run_dir / "metadata.json").write_text(json.dumps(meta))
    (run_dir / "model_config.json").write_text(json.dumps({"obs_dim": 1}))
    (run_dir / "latest.txt").write_text("checkpoints/checkpoint_1000_steps.zip")
    return str(ckpt)


# (a) the record --------------------------------------------------------------------------------

def test_a_an_UNRECORDED_run_reads_as_the_legacy_2_5_1(tmp_path):
    ver, src = tr.recorded_torch(_run(tmp_path))
    assert ver == tr.LEGACY_TORCH == "2.5.1"
    assert "unrecorded" in src


def test_a_a_recorded_run_reads_its_record(tmp_path):
    assert tr.recorded_torch(_run(tmp_path, "2.8.0+cu126"))[0] == "2.8.0+cu126"


def test_a_the_release_ignores_the_cuda_tag():
    assert tr.release("2.5.1+cu121") == tr.release("2.5.1") == "2.5.1"


# (b) a FRESH launch takes the default interpreter -----------------------------------------------

def test_b_fresh_runs_the_default(box):
    res = tr.resolve_for_launch(None)
    assert (res.python, res.torch, res.refusal) == (box["new"], "2.8.0+cu126", None)


# (c) a legacy run is SELECTED onto the legacy env -----------------------------------------------

def test_c_an_unrecorded_run_resumes_on_the_LEGACY_env(box, tmp_path):
    res = tr.resolve_for_launch(_run(tmp_path))
    assert res.refusal is None
    assert res.python == box["old"], "a 2.5.1 run must resume on gen3ai_stable, not the default"
    assert res.torch == "2.5.1+cu121" and "SELECTED" in res.how


def test_c_a_2_8_run_resumes_on_the_default(box, tmp_path):
    res = tr.resolve_for_launch(_run(tmp_path, "2.8.0+cu126"))
    assert (res.python, res.refusal) == (box["new"], None)


# (d) a mismatch REFUSES, and the consent flag turns it into a recorded switch -------------------

def test_d_no_env_carries_the_recorded_torch_REFUSES(box, tmp_path):
    os.remove(box["old"])                               # this box has no legacy env
    res = tr.resolve_for_launch(_run(tmp_path))
    assert res.refusal and "never switches torch silently" in res.refusal
    assert tr.ALLOW_FLAG in res.refusal and "gen3ai_stable" in res.refusal


def test_d_an_explicit_override_with_the_wrong_torch_REFUSES(box, tmp_path, monkeypatch):
    monkeypatch.setenv(PYTHON_ENV_VAR, box["new"])
    monkeypatch.setattr(tr, "resolve_child_python", lambda: box["new"])
    res = tr.resolve_for_launch(_run(tmp_path))         # unrecorded = 2.5.1
    assert res.refusal, "an explicit $GEN3AI_PYTHON is honoured, never silently replaced"


def test_d_consent_runs_the_default_and_says_so(box, tmp_path):
    os.remove(box["old"])
    res = tr.resolve_for_launch(_run(tmp_path), allow_switch=True)
    assert res.refusal is None and res.python == box["new"] and "CONSENTED" in res.how


def test_d_the_consent_flag_is_launcher_owned():
    assert _strip_launcher_args(["--steps", "1", tr.ALLOW_FLAG]) == ["--steps", "1"]
    assert launcher_run.build_launcher_parser().parse_known_args(
        [tr.ALLOW_FLAG])[0].allow_torch_switch is True


# (e) the REAL launch path: refuses FATAL_CONFIG, and holds the selected env for every spawn -----

def _prepare(ckpt, monkeypatch, **kw):
    monkeypatch.setattr(launcher_run.atexit, "register", lambda *a, **k: None)
    return launcher_run._prepare_session(
        LauncherState(0.0), ["--model", ckpt, "--steps", "2000"], interval_hours=0.0, pin=False,
        sync_to_main=False, pin_commit=None, grace_minutes=1.0, max_crash_restarts=0, **kw)


def test_e_prepare_session_REFUSES_a_torch_mismatch(box, tmp_path, monkeypatch, capsys):
    os.remove(box["old"])
    with pytest.raises(SystemExit) as e:
        _prepare(_run(tmp_path), monkeypatch)
    assert e.value.code == FATAL
    assert "never switches torch silently" in capsys.readouterr().err


def test_e_prepare_session_holds_the_SELECTED_env_for_the_child(box, tmp_path, monkeypatch):
    ctx = _prepare(_run(tmp_path), monkeypatch)
    assert ctx.child_env[PYTHON_ENV_VAR] == box["old"]


def test_e_the_spawn_runs_the_session_interpreter(tmp_path, monkeypatch):
    """`_launch_child` takes argv[0] from the session's child_env, not a fresh resolve."""
    from main.launcher import child
    probe = tmp_path / "probe.py"
    probe.write_text("print('unreachable')\n")
    state = LauncherState(interval_hours=0.0)
    state.run_dir = str(tmp_path)
    monkeypatch.delenv(PYTHON_ENV_VAR, raising=False)
    env = {**child._build_child_env(), PYTHON_ENV_VAR: str(tmp_path / "not-a-python")}
    with pytest.raises(OSError):
        child._launch_child([], env, state, str(probe), os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))


# (f) --dry-run prints the interpreter AND its torch, and refuses exactly as the launch does ----

def test_f_dry_run_prints_interpreter_and_torch_and_refuses(box, tmp_path, monkeypatch, capsys):
    os.remove(box["old"])
    ckpt = _run(tmp_path)
    monkeypatch.setattr("sys.argv", ["launcher", "--dry-run", "--no-pin", "--model", ckpt,
                                     "--steps", "2000"])
    with pytest.raises(SystemExit) as e:
        launcher_run.main()
    out = capsys.readouterr().out
    assert e.value.code == FATAL
    assert f"interpreter : {box['new']}  (torch 2.8.0+cu126)" in out
    assert "run torch   : 2.5.1" in out and "REFUSED (torch)" in out


def test_f_dry_run_shows_the_selected_legacy_env(box, tmp_path, monkeypatch, capsys):
    ckpt = _run(tmp_path)
    monkeypatch.setattr("sys.argv", ["launcher", "--dry-run", "--no-pin", "--model", ckpt,
                                     "--steps", "2000"])
    with pytest.raises(SystemExit):
        launcher_run.main()
    out = capsys.readouterr().out
    assert f"interpreter : {box['old']}  (torch 2.5.1+cu121)" in out and "REFUSED (torch)" not in out


# (g) two CUDA BUILDS of one release are two measurements (owner 2026-10-12, the cu128 twin env) ----

@pytest.mark.parametrize("have, want, expected", [
    ("2.8.0+cu126", "2.8.0+cu126", True),
    ("2.8.0+cu128", "2.8.0+cu128", True),
    ("2.8.0+cu128", "2.8.0+cu126", False),     # the case this exists for
    ("2.8.0+cu126", "2.8.0+cu128", False),
    ("2.8.0+cu128", "2.8.0", True),            # a build-less record matches any build of its release
    ("2.8.0", "2.8.0+cu128", True),
    ("2.5.1+cu121", "2.5.1", True),            # the unrecorded legacy run (reads plain 2.5.1)
    ("2.8.0+cu128", "2.5.1", False),           # a different RELEASE never matches
    ("2.8.0+cu128", "2.9.0+cu128", False),
    (None, "2.8.0+cu128", False),              # an unreadable interpreter carries nothing
    ("2.8.0+cu128", None, False),
])
def test_g_carries_compares_release_and_the_cuda_build_when_both_name_one(have, want, expected):
    assert tr.carries(have, want) is expected


def test_g_build_tag():
    assert tr.build_tag("2.8.0+cu128") == "cu128"
    assert tr.build_tag("2.8.0") is None and tr.build_tag(None) is None and tr.build_tag("2.8.0+") is None


@pytest.fixture
def box_cu128(tmp_path, monkeypatch):
    """The launcher runs in the cu128 TWIN; the default env (cu126) and the legacy env are its siblings."""
    envs = tmp_path / "conda" / "envs"
    twin = _fake_python(envs / "gen3ai_torch28_cu128" / "bin" / "python3", "2.8.0+cu128")
    default = _fake_python(envs / "gen3ai_torch28" / "bin" / "python3", "2.8.0+cu126")
    monkeypatch.setattr(tr.sys, "prefix", str(envs / "gen3ai_torch28_cu128"))
    monkeypatch.setattr(tr, "resolve_child_python", lambda: twin)
    monkeypatch.delenv(PYTHON_ENV_VAR, raising=False)
    monkeypatch.setattr(tr, "_probe_cache", {})
    return {"twin": twin, "default": default, "envs": envs}


def test_g_a_cu126_run_resumed_from_the_cu128_env_SELECTS_the_cu126_env(box_cu128, tmp_path):
    res = tr.resolve_for_launch(_run(tmp_path, "2.8.0+cu126"))
    assert res.refusal is None
    assert res.python == box_cu128["default"] and "SELECTED" in res.how and "gen3ai_torch28" in res.how


def test_g_a_cu126_run_with_the_cu126_env_ABSENT_REFUSES_without_the_consent_flag(box_cu128, tmp_path):
    """THE behaviour the owner asked for: no env carries cu126, so the cu128 interpreter does not
    silently take the run. The refusal names the cu126 record and the env that carries it."""
    os.remove(box_cu128["default"])
    res = tr.resolve_for_launch(_run(tmp_path, "2.8.0+cu126"))
    assert res.refusal and "never switches torch silently" in res.refusal
    assert "2.8.0+cu126" in res.refusal and "2.8.0+cu128" in res.refusal
    assert "`gen3ai_torch28` env" in res.refusal and tr.ALLOW_FLAG in res.refusal


def test_g_an_explicit_cu128_override_with_a_cu126_run_REFUSES(box_cu128, tmp_path, monkeypatch):
    """`GEN3AI_PYTHON=<cu128 env>` is honoured, never replaced by a sibling: it simply does not carry the run."""
    monkeypatch.setenv(PYTHON_ENV_VAR, box_cu128["twin"])
    res = tr.resolve_for_launch(_run(tmp_path, "2.8.0+cu126"))
    assert res.refusal and "pinned by $GEN3AI_PYTHON" in res.refusal


def test_g_consent_switches_a_cu126_run_to_the_cu128_env_and_says_so(box_cu128, tmp_path):
    os.remove(box_cu128["default"])
    res = tr.resolve_for_launch(_run(tmp_path, "2.8.0+cu126"), allow_switch=True)
    assert res.refusal is None and res.python == box_cu128["twin"] and "CONSENTED" in res.how


def test_g_a_cu128_run_resumed_from_the_cu126_default_SELECTS_the_twin(box, tmp_path):
    twin = _fake_python(box["envs"] / "gen3ai_torch28_cu128" / "bin" / "python3", "2.8.0+cu128")
    res = tr.resolve_for_launch(_run(tmp_path, "2.8.0+cu128"))
    assert res.refusal is None and res.python == twin and "SELECTED" in res.how


def test_g_a_cu128_run_with_no_twin_env_REFUSES_from_the_cu126_default(box, tmp_path):
    res = tr.resolve_for_launch(_run(tmp_path, "2.8.0+cu128"))
    assert res.refusal and "gen3ai_torch28_cu128" in res.refusal


def test_g_a_cu128_run_resumes_on_the_twin_and_a_buildless_record_takes_the_launchers_own(box_cu128, tmp_path):
    res = tr.resolve_for_launch(_run(tmp_path, "2.8.0+cu128"))
    assert (res.python, res.refusal) == (box_cu128["twin"], None)
    res = tr.resolve_for_launch(_run(tmp_path, "2.8.0", name="r2"))      # a record with no build tag
    assert (res.python, res.refusal) == (box_cu128["twin"], None)


def test_g_a_FRESH_launch_in_the_cu128_env_runs_it_and_records_its_build_at_the_first_save(box_cu128):
    """A fresh run takes the launcher's interpreter whatever its build (the snapshot writes
    `torch.__version__`, `2.8.0+cu128`, as `metadata.json`'s `torch_version` — `snapshot.py`)."""
    res = tr.resolve_for_launch(None)
    assert (res.python, res.torch, res.refusal) == (box_cu128["twin"], "2.8.0+cu128", None)


def test_the_real_box_probe_reads_this_interpreter():
    """The probe itself, unfaked: this interpreter's torch, read without importing torch."""
    import importlib.metadata
    tr._probe_cache.pop(sys.executable, None)
    assert tr.interpreter_torch(sys.executable) == importlib.metadata.version("torch")
