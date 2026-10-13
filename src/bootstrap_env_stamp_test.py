"""`scripts/bootstrap.sh` step 2 — a WORKTREE bootstrap never mutates the SHARED conda env silently.

**The hazard (2026-09-29).** The "env is current" stamp lived in each worktree's own gitdir, so
EVERY fresh worktree saw no stamp and ran `conda env update --prune` against the shared env — the
one env every live run, pinned launch and agent on the box shares. One started under a live
measurement job that day (killed at the pip step; nothing changed that time). Five m5 worktrees had
each run a full prune-update in the two days before.

**The fix, pinned here by EXECUTING the real script** against a throwaway clone + linked worktree,
with a fake `conda` first on `PATH` that logs every argv (no monkeypatch — the script is bash, so
the stub is the executable it actually runs, and each test asserts on the log it writes):

* the stamp lives in the git COMMON dir, keyed by the env file's sha256
  (`environment_torch28.yml` since 2026-09-30) → a worktree whose env is current invokes no
  `conda env` command at all;
* a worktree whose env file differs PRINTS the diff and REFUSES (exit 3), naming
  `--update-shared-env` (the explicit opt-in) and `--skip-env` (leave the env alone);
* the main checkout keeps the old behaviour (it updates);
* the LEGACY `gen3ai_stable` / `environment.yml` is never named by any conda call;
* the newest legacy per-worktree stamp is adopted when it matches, so the migration refuses nothing.

Reverting the stamp location to `--git-dir` makes the first test fail: the worktree sees no stamp
and the fake conda logs an `env update`.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from utils.paths import repo_path

#: THE env file (owner 2026-09-30: torch 2.8 is the default). `environment.yml` is the frozen
#: LEGACY spec of `gen3ai_stable`, which bootstrap must never read or touch.
_ENV_FILE = "environment_torch28.yml"
_ENV_NAME = "gen3ai_torch28"
_ENV_YML = "name: gen3ai_torch28\ndependencies:\n  - python=3.11\n"
_ENV_YML_CHANGED = _ENV_YML + "  - newpkg=1.0\n"
#: The CUDA 12.8 TWIN (owner 2026-10-12): its own env, env file and stamp name.
_ENV_FILE_CU128 = "environment_torch28_cu128.yml"
_ENV_NAME_CU128 = "gen3ai_torch28_cu128"
_ENV_YML_CU128 = "name: gen3ai_torch28_cu128\ndependencies:\n  - python=3.11\n  - cuda-cccl=12.8.90\n"

_FAKE_CONDA = """#!/bin/sh
echo "$*" >> "$FAKE_CONDA_LOG"
case "$1" in
  info) echo "$FAKE_CONDA_BASE" ;;
  --version) echo "conda 0.0.0" ;;
esac
exit 0
"""
_FAKE_TOOL = """#!/bin/sh
case "$1" in --version) echo "v0.0.0" ;; esac
exit 0
"""


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(cwd), text=True,
                                   stderr=subprocess.STDOUT)


@pytest.fixture
def clone(tmp_path):
    """A throwaway MAIN checkout carrying the real bootstrap.sh, one linked WORKTREE, a fake conda
    whose base holds an existing `gen3ai_torch28` (and the legacy `gen3ai_stable`), and every
    Showdown artifact pre-planted so steps 4-5 are no-ops."""
    if shutil.which("git") is None or shutil.which("sha256sum") is None:
        pytest.fail("git and sha256sum are required to execute bootstrap.sh")
    main = tmp_path / "main"
    (main / "scripts").mkdir(parents=True)
    shutil.copy(repo_path("scripts", "bootstrap.sh"), main / "scripts" / "bootstrap.sh")
    (main / _ENV_FILE).write_text(_ENV_YML)
    (main / _ENV_FILE_CU128).write_text(_ENV_YML_CU128)
    # the LEGACY file is present, as in the repo, and must be ignored
    (main / "environment.yml").write_text("name: gen3ai_stable\ndependencies:\n  - python=3.11\n")
    _git(main, "init", "-q", "-b", "main")
    _git(main, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(main, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", str(wt), "-b", "wt")
    for tree in (main, wt):
        sd = tree / "deps" / "pokemon-showdown"
        (sd / "dist" / "sim").mkdir(parents=True)
        (sd / "node_modules").mkdir()
        (sd / "package.json").write_text("{}")
        (sd / "dist" / "sim" / "index.js").write_text("")

    base = tmp_path / "conda"
    for name in (_ENV_NAME, "gen3ai_stable"):
        (base / "envs" / name / "bin").mkdir(parents=True)
        # the fake interpreter also stands in for `env_file.py check` (--adopt-env): rc from $FAKE_PY_RC
        _exe(base / "envs" / name / "bin" / "python3", '#!/bin/sh\necho "$FAKE_PY_OUT"\nexit "${FAKE_PY_RC:-0}"\n')
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _exe(fake_bin / "conda", _FAKE_CONDA)
    for tool in ("node", "npm", "cargo"):
        _exe(fake_bin / tool, _FAKE_TOOL)
    log = tmp_path / "conda.log"

    def run(tree: Path, *flags: str, extra_env: "dict[str, str] | None" = None):
        env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}", NO_COLOR="1", **(extra_env or {}),
                   FAKE_CONDA_LOG=str(log), FAKE_CONDA_BASE=str(base))
        env.pop("GIT_DIR", None)
        env.pop("GIT_WORK_TREE", None)
        p = subprocess.run(["bash", str(tree / "scripts" / "bootstrap.sh"), "--no-rust",
                            "--no-check", *flags], cwd=str(tree), env=env, text=True,
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
        calls = log.read_text().splitlines() if log.exists() else []
        log.unlink(missing_ok=True)
        return p.returncode, p.stdout + p.stderr, calls

    common = Path(_git(wt, "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    return {"main": main, "wt": wt, "run": run, "stamp_dir": common / "gen3ai-bootstrap",
            "common": common, "base": base}


def _env_calls(calls):
    return [c for c in calls if c.startswith("env ")]


def _plant_stamp(stamp_dir: Path, text: str) -> Path:
    stamp_dir.mkdir(parents=True, exist_ok=True)
    p = stamp_dir / f"env-{_sha(text)}"
    p.write_text(text)
    return p


def test_a_worktree_with_a_current_COMMON_DIR_stamp_never_invokes_conda_env(clone):
    """THE regression: before the fix the stamp was per-worktree, so this worktree saw none and ran
    `conda env update --prune` against the shared env."""
    _plant_stamp(clone["stamp_dir"], _ENV_YML)
    rc, out, calls = clone["run"](clone["wt"])
    assert rc == 0, out
    assert _env_calls(calls) == [], f"a current worktree touched the shared env: {calls}\n{out}"
    assert f"is current for this {_ENV_FILE}" in out, out


def test_a_worktree_whose_environment_yml_changed_REFUSES_and_prints_the_diff(clone):
    _plant_stamp(clone["stamp_dir"], _ENV_YML)
    (clone["wt"] / _ENV_FILE).write_text(_ENV_YML_CHANGED)
    rc, out, calls = clone["run"](clone["wt"])
    assert rc == 3, out
    assert _env_calls(calls) == [], calls
    assert "REFUSED" in out and "--update-shared-env" in out and "--skip-env" in out, out
    assert "+  - newpkg=1.0" in out, "the refusal must say what would change\n" + out
    # --force is a request to mutate too, and is refused the same way from a worktree.
    rc, out, calls = clone["run"](clone["wt"], "--force")
    assert rc == 3 and _env_calls(calls) == [], out
    # --dry-run reports the same refusal (it is the plan) and runs nothing.
    rc, out, calls = clone["run"](clone["wt"], "--dry-run")
    assert rc == 3 and "REFUSED" in out and _env_calls(calls) == [], out


def test_a_worktree_with_NO_stamp_at_all_refuses(clone):
    rc, out, calls = clone["run"](clone["wt"])
    assert rc == 3 and _env_calls(calls) == [], out
    assert "NO stamp records" in out, out


def test_the_explicit_opt_in_updates_and_moves_the_one_stamp(clone):
    old = _plant_stamp(clone["stamp_dir"], _ENV_YML)
    (clone["wt"] / _ENV_FILE).write_text(_ENV_YML_CHANGED)
    rc, out, calls = clone["run"](clone["wt"], "--update-shared-env")
    assert rc == 0, out
    assert _env_calls(calls) == [f"env update -n {_ENV_NAME} -f {_ENV_FILE} --prune"], calls
    assert sorted(p.name for p in clone["stamp_dir"].iterdir()) == [f"env-{_sha(_ENV_YML_CHANGED)}"]
    assert not old.exists(), "a stale stamp left behind would call a REVERTED file current"
    # ...and now every worktree with that file is current, with no conda call.
    rc, out, calls = clone["run"](clone["wt"])
    assert rc == 0 and _env_calls(calls) == [], out


def test_skip_env_leaves_the_env_alone_and_finishes(clone):
    (clone["wt"] / _ENV_FILE).write_text(_ENV_YML_CHANGED)
    rc, out, calls = clone["run"](clone["wt"], "--skip-env")
    assert rc == 0 and _env_calls(calls) == [], out
    assert "--skip-env" in out and "Bootstrap complete" in out, out


def test_the_MAIN_checkout_still_updates_a_changed_env(clone):
    _plant_stamp(clone["stamp_dir"], _ENV_YML)
    (clone["main"] / _ENV_FILE).write_text(_ENV_YML_CHANGED)
    rc, out, calls = clone["run"](clone["main"])
    assert rc == 0, out
    assert _env_calls(calls) == [f"env update -n {_ENV_NAME} -f {_ENV_FILE} --prune"], calls
    assert (clone["stamp_dir"] / f"env-{_sha(_ENV_YML_CHANGED)}").exists()


def test_the_newest_legacy_per_worktree_stamp_is_adopted_when_it_matches(clone):
    """Migration: the old stamps lived in `<common>/worktrees/<wt>/gen3ai-bootstrap/`. The NEWEST
    one records the last successful update; if it is for this exact env file, adopt it."""
    legacy = clone["common"] / "worktrees" / "wt" / "gen3ai-bootstrap"
    legacy.mkdir(parents=True)
    (legacy / f"env-{_sha(_ENV_YML)}").touch()
    rc, out, calls = clone["run"](clone["wt"])
    assert rc == 0 and _env_calls(calls) == [], out
    assert "adopting the newest per-worktree stamp" in out, out
    assert (clone["stamp_dir"] / f"env-{_sha(_ENV_YML)}").read_text() == _ENV_YML


def test_the_LEGACY_env_is_never_touched_even_when_its_file_changes(clone):
    """`gen3ai_stable` (torch 2.5.1, `environment.yml`) is FROZEN for legacy resumes: a changed
    `environment.yml` in the MAIN checkout, even with `--force`, issues no conda call naming it."""
    _plant_stamp(clone["stamp_dir"], _ENV_YML)
    (clone["main"] / "environment.yml").write_text("name: gen3ai_stable\n  - changed=1\n")
    for flags in ((), ("--force",)):
        rc, out, calls = clone["run"](clone["main"], *flags)
        assert rc == 0, out
        assert not any("gen3ai_stable" in c or "environment.yml" in c.split() for c in calls), calls


# ---------------------------------------------------------------------------------------------
# --adopt-env: the SAFE re-stamp after an env-file edit that only added pins the env already has
# (2026-10-12: scipy / scikit-learn declared in environment_torch28.yml; every worktree bootstrap
# refused until re-stamped, and `--update-shared-env` mutates the env every live run shares).
# The fake interpreter stands in for `src/utils/env_file.py check` (rc from $FAKE_PY_RC); the real
# checker has its own tests (`utils/env_file_test.py`).
# ---------------------------------------------------------------------------------------------

def test_adopt_env_stamps_a_SATISFIED_env_without_any_conda_call_and_KEEPS_the_old_stamp(clone):
    old = _plant_stamp(clone["stamp_dir"], _ENV_YML)
    (clone["wt"] / _ENV_FILE).write_text(_ENV_YML_CHANGED)
    rc, out, calls = clone["run"](clone["wt"], "--adopt-env", extra_env={"FAKE_PY_OUT": "all 5 pins met"})
    assert rc == 0, out
    assert _env_calls(calls) == [], f"adopting must not run conda: {calls}\n{out}"
    assert "all 5 pins met" in out and "adopted" in out, out
    assert sorted(p.name for p in clone["stamp_dir"].iterdir()) == sorted(
        [old.name, f"env-{_sha(_ENV_YML_CHANGED)}"]), "the OLD stamp stays true: the env did not change"
    # ...so a worktree still at the previous env file is not refused, and the adopted one is current too.
    for text in (_ENV_YML, _ENV_YML_CHANGED):
        (clone["wt"] / _ENV_FILE).write_text(text)
        rc, out, calls = clone["run"](clone["wt"])
        assert rc == 0 and _env_calls(calls) == [], out


def test_adopt_env_REFUSES_an_env_that_does_not_satisfy_the_file_and_stamps_nothing(clone):
    old = _plant_stamp(clone["stamp_dir"], _ENV_YML)
    (clone["wt"] / _ENV_FILE).write_text(_ENV_YML_CHANGED)
    rc, out, calls = clone["run"](clone["wt"], "--adopt-env",
                                  extra_env={"FAKE_PY_RC": "1", "FAKE_PY_OUT": "1 of 3 pins NOT met: newpkg"})
    assert rc == 3, out
    assert _env_calls(calls) == [], calls
    assert "newpkg" in out and "nothing safe to adopt" in out and "--update-shared-env" in out, out
    assert [p.name for p in clone["stamp_dir"].iterdir()] == [old.name], "a refused adopt must stamp nothing"


def test_adopt_env_under_dry_run_stamps_nothing(clone):
    _plant_stamp(clone["stamp_dir"], _ENV_YML)
    (clone["wt"] / _ENV_FILE).write_text(_ENV_YML_CHANGED)
    rc, out, calls = clone["run"](clone["wt"], "--adopt-env", "--dry-run")
    assert rc == 0 and _env_calls(calls) == [], out
    assert len(list(clone["stamp_dir"].iterdir())) == 1 and "[dry-run] add the env stamp" in out, out


def test_the_refusal_names_adopt_env_as_the_non_mutating_way_forward(clone):
    _plant_stamp(clone["stamp_dir"], _ENV_YML)
    (clone["wt"] / _ENV_FILE).write_text(_ENV_YML_CHANGED)
    rc, out, _ = clone["run"](clone["wt"])
    assert rc == 3 and "--adopt-env" in out, out


# --variant cu128: the twin env, its own file and its own stamp --------------------------------

def test_variant_cu128_CREATES_a_missing_twin_env_from_its_own_file_and_leaves_the_default_alone(clone):
    old = _plant_stamp(clone["stamp_dir"], _ENV_YML)
    rc, out, calls = clone["run"](clone["wt"], "--variant", "cu128")
    assert rc == 0, out
    assert _env_calls(calls) == [f"env create -f {_ENV_FILE_CU128}"], calls
    assert sorted(p.name for p in clone["stamp_dir"].iterdir()) == sorted(
        [old.name, f"envcu128-{_sha(_ENV_YML_CU128)}"]), "the twin's stamp must not delete the default's"
    assert f"conda activate {_ENV_NAME_CU128}" in out, out


def _plant_twin_env(clone) -> None:
    py = clone["base"] / "envs" / _ENV_NAME_CU128 / "bin" / "python3"
    py.parent.mkdir(parents=True)
    _exe(py, "#!/bin/sh\nexit 0\n")


def test_variant_cu128_with_a_current_stamp_does_nothing_and_a_default_rewrite_keeps_the_twins_stamp(clone):
    _plant_twin_env(clone)
    _plant_stamp(clone["stamp_dir"], _ENV_YML)
    twin = clone["stamp_dir"] / f"envcu128-{_sha(_ENV_YML_CU128)}"
    twin.write_text(_ENV_YML_CU128)
    rc, out, calls = clone["run"](clone["wt"], "--variant", "cu128")
    assert rc == 0 and _env_calls(calls) == [], out
    # an --update-shared-env of the DEFAULT replaces the default's stamps only
    (clone["main"] / _ENV_FILE).write_text(_ENV_YML_CHANGED)
    rc, out, calls = clone["run"](clone["main"])
    assert rc == 0 and _env_calls(calls) == [f"env update -n {_ENV_NAME} -f {_ENV_FILE} --prune"], calls
    assert twin.exists(), "rewriting the default env's stamp deleted the twin's"


def test_variant_cu128_is_refused_from_a_worktree_when_the_twin_exists_and_its_file_changed(clone):
    _plant_twin_env(clone)
    (clone["stamp_dir"]).mkdir(parents=True, exist_ok=True)
    (clone["stamp_dir"] / f"envcu128-{_sha(_ENV_YML_CU128)}").write_text(_ENV_YML_CU128)
    (clone["wt"] / _ENV_FILE_CU128).write_text(_ENV_YML_CU128 + "  - newpkg=1.0\n")
    rc, out, calls = clone["run"](clone["wt"], "--variant", "cu128")
    assert rc == 3 and _env_calls(calls) == [], out
    assert "--variant cu128" in out and "+  - newpkg=1.0" in out, out


def test_an_unknown_variant_is_a_usage_error(clone):
    rc, out, calls = clone["run"](clone["wt"], "--variant", "cu999")
    assert rc == 2 and "unknown --variant" in out and calls == [], out
    rc, out, calls = clone["run"](clone["wt"], "--variant")
    assert rc == 2 and "--variant needs a value" in out, out
