"""Gates for path discovery — the arithmetic and the worktree reach-across.

Two things are pinned here (the third, the hard-coded-path CLASS, is the static-tier gate
``src/utils/home_path_literal_gate_test.py``: it only scans source, so it runs in the static tier and a
targeted ship set cannot miss it):

  1. **The depth arithmetic is right**, cross-checked against ``git rev-parse`` rather than
     asserted. `utils/paths.py` computes the repo root from ``__file__``; if the file ever moves,
     git disagrees and this fails. That check is the whole reason one module may own the numbers.
  2. **The run archive resolves to the MAIN checkout from inside a worktree**, and its absence
     produces a `None` that every caller must turn into a skip. Exercised by pointing
     ``$GEN3AI_MODELS_DIR`` at an empty directory — the four tests that read `models/` are driven
     through their skip path *here*, on this box, because on this box they otherwise never take
     it. A skip path nothing exercises is a skip path nobody has ever seen work.
"""
import pathlib
import subprocess

import pytest
from _pytest.outcomes import Skipped

from utils.git import get_main_repo_root, get_repo_root
from utils.paths import (
    HARVEST_DIR_ENV_VAR,
    MODELS_DIR_ENV_VAR,
    harvest_dir,
    main_models_dir,
    models_skip_reason,
    repo_path,
    repo_root,
    run_skip_reason,
    src_path,
    src_root,
    trace_glob,
)


# ------------------------------------------------------------------ 1. the arithmetic is right
def test_repo_root_matches_git():
    """The `__file__` arithmetic and git must agree on where the checkout is."""
    assert str(repo_root()) == get_repo_root(cwd=str(repo_root()))


def test_src_root_is_repo_root_over_src_and_holds_this_package():
    assert src_root() == repo_root() / "src"
    assert (src_root() / "utils" / "paths.py").is_file()
    assert (repo_root() / "conftest.py").is_file()


def test_join_helpers():
    assert repo_path("designs", "ARCHITECTURE.md") == repo_root() / "designs" / "ARCHITECTURE.md"
    assert src_path("agents", "model") == repo_root() / "src" / "agents" / "model"
    # A path need not exist — these are pure joins, so callers can test for absence themselves.
    assert not repo_path("no", "such", "thing").exists()


def test_repo_root_is_cwd_independent(tmp_path, monkeypatch):
    """The point of `__file__`-relative discovery: standing somewhere else changes nothing."""
    before = repo_root()
    monkeypatch.chdir(tmp_path)
    assert repo_root() == before
    assert src_root() == before / "src"


# ------------------------------------------------- 2. the run archive, and the reach-across
def test_main_models_dir_resolves_the_main_checkout_not_the_worktree():
    """`models/` is not committed and lives only in the MAIN checkout.

    Inside a linked worktree `repo_root()` is the worktree, which has no archive — so the
    resolver must reach across via git's shared `--git-common-dir`. Where there is no archive at
    all (a fresh clone) the answer is `None`, which is the skip signal.
    """
    got = main_models_dir()
    if got is None:
        pytest.skip(models_skip_reason())
    assert got.is_dir()
    assert got == pathlib.Path(get_main_repo_root(cwd=str(repo_root()))) / "models"


def test_env_override_wins_and_is_authoritative(tmp_path, monkeypatch):
    monkeypatch.setenv(MODELS_DIR_ENV_VAR, str(tmp_path))
    assert main_models_dir() == tmp_path

    # Set-but-missing must be None, NOT a quiet fall-back to the real archive: an explicit
    # override that silently resolves elsewhere is how a "no archive" test run passes anyway.
    monkeypatch.setenv(MODELS_DIR_ENV_VAR, str(tmp_path / "nope"))
    assert main_models_dir() is None


def test_trace_glob_and_its_skip_reason(tmp_path, monkeypatch):
    monkeypatch.setenv(MODELS_DIR_ENV_VAR, str(tmp_path))
    assert trace_glob("some_run") is None                      # archive present, run absent
    (tmp_path / "some_run" / "eval_traces").mkdir(parents=True)
    got = trace_glob("some_run")
    assert got is not None and got.endswith("_states.npz") and "some_run" in got


def test_skip_messages_name_what_is_missing_and_the_escape_hatch():
    """A skip a contributor reads must not send them hunting for a broken test."""
    for msg in (models_skip_reason(), run_skip_reason("some_run")):
        assert "models/" in msg
        assert "not committed" in msg or "NOT committed" in msg
        assert MODELS_DIR_ENV_VAR in msg
    assert "some_run" in run_skip_reason("some_run")


# --------------------------------------- the four archive-reading tests, driven through SKIP
def _empty_archive(monkeypatch, tmp_path):
    monkeypatch.setenv(MODELS_DIR_ENV_VAR, str(tmp_path))


def test_arch_tables_drift_gate_skips_on_an_empty_archive(monkeypatch, tmp_path):
    """The registry-backed drift gate (`gen3_baselines_registry_v1` replaced the newest-run
    heuristic) must still SKIP, not fail, when the archive is empty — and its skip must name the
    escape hatch. Two ways to have no archive, and both are driven here: an env var pointing at a
    non-directory (`main_models_dir()` is None) and an empty one (the baseline's run is absent)."""
    from agents.model import arch_tables_test as m
    _empty_archive(monkeypatch, tmp_path)
    with pytest.raises(Skipped) as ei:
        m.test_production_config_matches_the_registry()
    assert MODELS_DIR_ENV_VAR in str(ei.value)

    monkeypatch.setenv(MODELS_DIR_ENV_VAR, str(tmp_path / "not-a-directory"))
    with pytest.raises(Skipped) as ei2:
        m.test_production_config_matches_the_registry()
    assert MODELS_DIR_ENV_VAR in str(ei2.value)


def test_audit_states_real_trace_gate_skips_on_an_empty_archive(monkeypatch, tmp_path):
    from agents.model import audit_states_test as m
    _empty_archive(monkeypatch, tmp_path)
    for real_trace_test in (m.test_real_core_traces_recover_a_mask_with_illegal_actions,
                            m.test_real_core_traces_recorded_mask_admits_every_played_action):
        with pytest.raises(Skipped) as ei:
            real_trace_test()
        assert m._REAL_TRACE_RUN in str(ei.value)


def test_git_root_helpers_accept_a_cwd_and_are_consistent(tmp_path, monkeypatch):
    """`get_main_repo_root(cwd=…)` must resolve git's RELATIVE `--git-common-dir` against that
    cwd, not the process's — otherwise pinning the cwd silently changes the answer."""
    expected = get_main_repo_root(cwd=str(repo_root()))
    monkeypatch.chdir(tmp_path)
    assert get_main_repo_root(cwd=str(repo_root())) == expected
    assert get_repo_root(cwd=str(repo_root())) == str(repo_root())
    with pytest.raises(subprocess.CalledProcessError):
        get_repo_root(cwd=str(tmp_path))  # not a checkout at all


# ------------------------------------------------------- 5. harvest_dir — a WRITABLE destination


def test_harvest_dir_env_override_wins_and_need_not_exist(tmp_path, monkeypatch):
    """Unlike ``main_models_dir``, this answers a PRODUCER's question, so a path that does not
    exist yet is the normal case and ``None`` would be useless. The override is still
    authoritative — no quiet fall-back to the real archive."""
    target = tmp_path / "not_yet"
    monkeypatch.setenv(HARVEST_DIR_ENV_VAR, str(target))
    assert harvest_dir() == target
    assert not target.exists()


def test_harvest_dir_creates_on_demand(tmp_path, monkeypatch):
    target = tmp_path / "made"
    monkeypatch.setenv(HARVEST_DIR_ENV_VAR, str(target))
    assert harvest_dir(create=True) == target
    assert target.is_dir()
    assert harvest_dir(create=True) == target          # idempotent


def test_harvest_dir_is_not_inside_models(monkeypatch):
    """models/ is read-only by convention — a probe must never be able to corrupt the archive it
    is probing — so the harvest's artifacts live at their own root."""
    monkeypatch.delenv(HARVEST_DIR_ENV_VAR, raising=False)
    got = harvest_dir()
    assert got.name == "harvest"
    assert "models" not in got.parts
