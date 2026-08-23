"""Gates for path discovery — the arithmetic, the worktree reach-across, and the CLASS.

Three things are pinned here, and the third is the durable one:

  1. **The depth arithmetic is right**, cross-checked against ``git rev-parse`` rather than
     asserted. `utils/paths.py` computes the repo root from ``__file__``; if the file ever moves,
     git disagrees and this fails. That check is the whole reason one module may own the numbers.
  2. **The run archive resolves to the MAIN checkout from inside a worktree**, and its absence
     produces a `None` that every caller must turn into a skip. Exercised by pointing
     ``$GEN3AI_MODELS_DIR`` at an empty directory — the four tests that read `models/` are driven
     through their skip path *here*, on this box, because on this box they otherwise never take
     it. A skip path nothing exercises is a skip path nobody has ever seen work.
  3. **No module under `src/agents`, `src/main`, `src/utils` may hardcode a `/home/…` path as a
     VALUE.** This closes the class the four tests were instances of. It is an AST scan, not a
     grep: comments are absent from the AST entirely and docstrings are skipped, so the gate is
     about code that *uses* an absolute path, never about prose that mentions one. Measured
     2026-08-22: exactly one exemption, and it is a scanner naming its own pattern.
"""
import ast
import pathlib
import subprocess

import pytest
from _pytest.outcomes import Skipped

from utils.git import get_main_repo_root, get_repo_root
from utils.paths import (
    MODELS_DIR_ENV_VAR,
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
    from agents.model import arch_tables_test as m
    _empty_archive(monkeypatch, tmp_path)
    assert m._newest_run_config() is None
    with pytest.raises(Skipped) as ei:
        m.test_production_config_matches_newest_run()
    assert MODELS_DIR_ENV_VAR in str(ei.value)


def test_intent_move_cell_real_obs_skips_on_an_empty_archive(monkeypatch, tmp_path):
    from agents.model import intent_move_cell_test as m
    _empty_archive(monkeypatch, tmp_path)
    with pytest.raises(Skipped) as ei:
        m._real_obs(2)
    assert m._TRACE_RUN in str(ei.value)


def test_audit_states_real_trace_gate_skips_on_an_empty_archive(monkeypatch, tmp_path):
    from agents.model import audit_states_test as m
    _empty_archive(monkeypatch, tmp_path)
    with pytest.raises(Skipped) as ei:
        m.test_real_gen17_traces_recover_a_mask_with_illegal_actions()
    assert m._REAL_TRACE_RUN in str(ei.value)


def test_eval_sharding_fuzz_finds_no_checkpoint_on_an_empty_archive(monkeypatch, tmp_path):
    import importlib.util
    _empty_archive(monkeypatch, tmp_path)
    spec = importlib.util.spec_from_file_location(
        "_esf_for_paths_test", str(src_path("agents", "training", "eval_sharding_fuzz_test.py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._find_checkpoint() is None


# ------------------------------------------------------------------- 3. the class, closed
#: Scope mirrors the ruff and file-size gates. `src/poke_env` is a vendored fork and
#: `src/rust_sim` is a Rust crate whose Python is harness scratch — neither is ours to shape.
_SCANNED_ROOTS = ("agents", "main", "utils")

#: The ONE file allowed to hold an absolute-home literal, because its job IS scanning for them.
#: An exemption without a stated reason outlives its own fix and then misleads every reader.
_SCAN_EXEMPT = {
    "src/main/launcher/interpreter_test.py": "the launcher's own path-literal scanner — the "
                                             "literal here is the regex it searches WITH",
    "src/utils/paths_test.py": "this file — same reason",
}


def _string_values(path: pathlib.Path):
    """Every string CONSTANT in `path` that is not a docstring, with its line.

    Comments never reach the AST, so prose mentioning a path is structurally out of scope; only
    a string the code actually *uses* can fail this gate.
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings:
            yield node.lineno, node.value


def test_no_module_hardcodes_an_absolute_home_path():
    """THE CLASS. Four tests read `models/` through a `/home/goodlad/...` literal and therefore
    skipped forever on every other machine — invisible coverage loss, because a skip that is
    supposed to happen looks exactly like a skip that is not. Fixing the four lines closes four
    instances; this closes the class.
    """
    offenders = []
    for pkg in _SCANNED_ROOTS:
        for path in sorted((src_root() / pkg).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(repo_root()).as_posix()
            if rel in _SCAN_EXEMPT:
                continue
            try:
                values = list(_string_values(path))
            except SyntaxError:
                continue  # syntax_test.py owns that failure
            for lineno, value in values:
                if "/home/" in value:
                    offenders.append(f"{rel}:{lineno}: {value[:100]!r}")
    assert not offenders, (
        "absolute home path used as a value — it is correct on exactly one machine, and where it "
        "guards a skip it makes the test vanish silently everywhere else. Resolve it with "
        "`utils.paths` (repo_path/src_path/main_models_dir):\n  " + "\n  ".join(offenders))


def test_the_scan_exemptions_are_all_still_load_bearing():
    """An exemption that stopped being needed must LEAVE the list — the c-family house rule."""
    stale = []
    for rel in _SCAN_EXEMPT:
        path = repo_root() / rel
        assert path.is_file(), f"exempt file {rel} no longer exists — drop the entry"
        if not any("/home/" in v for _, v in _string_values(path)):
            stale.append(rel)
    assert not stale, f"exemptions no longer needed, delete them: {stale}"


def test_the_scan_can_actually_fail(tmp_path):
    """A gate that cannot fail is not a gate (the vacuity family)."""
    bad = tmp_path / "bad.py"
    bad.write_text('X = "/home/someone/models"\n', encoding="utf-8")
    assert any("/home/" in v for _, v in _string_values(bad))
    good = tmp_path / "good.py"
    good.write_text('# /home/someone/models is where it lives\n"""/home/x doc."""\nX = 1\n',
                    encoding="utf-8")
    assert not any("/home/" in v for _, v in _string_values(good))


def test_git_root_helpers_accept_a_cwd_and_are_consistent(tmp_path, monkeypatch):
    """`get_main_repo_root(cwd=…)` must resolve git's RELATIVE `--git-common-dir` against that
    cwd, not the process's — otherwise pinning the cwd silently changes the answer."""
    expected = get_main_repo_root(cwd=str(repo_root()))
    monkeypatch.chdir(tmp_path)
    assert get_main_repo_root(cwd=str(repo_root())) == expected
    assert get_repo_root(cwd=str(repo_root())) == str(repo_root())
    with pytest.raises(subprocess.CalledProcessError):
        get_repo_root(cwd=str(tmp_path))  # not a checkout at all
