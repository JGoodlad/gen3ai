"""The torch global-state guard (`gen3_torch_state_guard_v1`): its diff, and — FAILS ON REVERT — that
the ROOT conftest really fails a leaking test in each of its three windows (test / module / import).

The end-to-end half copies the root `conftest.py` VERBATIM beside planted test files and runs pytest
on them in a SUBPROCESS. A subprocess, because the planted tests leak on purpose: in-process, their
leak would land in this very process and (correctly) fail this test. A verbatim copy, because the
wiring IS the guard — delete the autouse fixture, the module-boundary hook or the collection hook
from the root conftest and the matching planted leak passes, and so this test fails.
"""
import os
import re
import subprocess
import sys
import textwrap

import pytest
import torch

from utils import torch_state_guard as tsg
from utils.paths import repo_path, src_root

# --------------------------------------------------------------------------- the diff, in-process


def test_a_clean_window_has_no_diff():
    before = tsg.snapshot()
    torch.ones(3).sum()
    assert tsg.diff(before, tsg.snapshot()) == []


def test_every_core_row_is_read_and_none_is_unreadable():
    snap = tsg.snapshot()
    core = [g.name for g in tsg.DECLARED if g.module == "torch"]
    assert set(core) <= set(snap), "a core torch row was not read although torch is imported"
    bad = [n for n, v in snap.items() if v == tsg._UNREADABLE]
    assert not bad, f"rows whose reader raised on this torch ({torch.__version__}): {bad}"


def test_a_thread_count_change_is_named_with_both_values_and_restoring_it_clears_it():
    before = tsg.snapshot()
    n = torch.get_num_threads()
    try:
        torch.set_num_threads(n + 1)
        leaks = tsg.diff(before, tsg.snapshot())
    finally:
        torch.set_num_threads(n)
    assert leaks == [("torch.get_num_threads()", n, n + 1)]
    assert tsg.diff(before, tsg.snapshot()) == []
    assert f"torch.get_num_threads(): {n} -> {n + 1}" in tsg.describe("test X", leaks)


def test_a_dynamo_config_change_is_named_by_KEY_and_patch_restores_it():
    import torch._dynamo  # noqa: F401
    cfg = torch._dynamo.config
    before = tsg.snapshot()
    with cfg.patch(suppress_errors=not cfg.suppress_errors):
        leaks = tsg.diff(before, tsg.snapshot())
    assert [name for name, _, _ in leaks] == ["torch._dynamo.config.suppress_errors"]
    assert tsg.diff(before, tsg.snapshot()) == []


def test_torch_globals_restores_even_on_an_exception():
    before = tsg.snapshot()
    other = "high" if torch.get_float32_matmul_precision() == "highest" else "highest"
    with pytest.raises(RuntimeError):
        with tsg.torch_globals(num_threads=torch.get_num_threads() + 1,
                               float32_matmul_precision=other):
            assert tsg.diff(before, tsg.snapshot())
            raise RuntimeError("boom")
    assert tsg.diff(before, tsg.snapshot()) == []


def test_a_module_imported_INSIDE_the_window_is_judged_against_its_default():
    """A config module absent BEFORE is compared to its import-time default — measured equal to its
    live values on a fresh import (the premise this rule rests on, checked here on this torch)."""
    import torch._inductor.config as ind
    live, default = tsg._config_values(ind), tsg._config_defaults(ind)
    moved = {k for k in live if not tsg._same(live[k], default[k])}
    assert not moved, f"inductor config keys already off their defaults in this process: {moved}"
    name = "torch._inductor.config"
    after = {name: dict(live, max_autotune=not live["max_autotune"])}
    leaks = tsg.diff({}, after)
    assert [n for n, _, _ in leaks] == ["torch._inductor.config.max_autotune"]


# --------------------------------------------------------------------------- the wiring, end to end

_PLANTED = {
    # TEST window: a body that leaks, a body that restores (three idioms), and one after the leak.
    "test_body.py": """
        import torch, torch._dynamo
        from utils.torch_state_guard import torch_globals

        def test_leaks_threads():
            torch.set_num_threads(torch.get_num_threads() + 1)

        def test_restores_with_try_finally():
            n = torch.get_num_threads()
            try:
                torch.set_num_threads(n + 2)
            finally:
                torch.set_num_threads(n)

        def test_restores_with_the_context_manager():
            with torch_globals(num_threads=3, float32_matmul_precision="medium"):
                pass

        def test_restores_with_config_patch():
            with torch._dynamo.config.patch(cache_size_limit=99):
                pass

        def test_leaks_dynamo_config():
            torch._dynamo.config.cache_size_limit = 97
    """,
    # MODULE window: a module-scoped fixture that never restores; and one that does (no failure).
    "test_modfix_leak.py": """
        import pytest, torch

        @pytest.fixture(scope="module")
        def pinned():
            torch.set_float32_matmul_precision("medium")

        def test_one(pinned):
            pass

        def test_two(pinned):
            pass
    """,
    "test_modfix_clean.py": """
        import pytest, torch

        @pytest.fixture(scope="module")
        def pinned():
            prev = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("high")
            yield
            torch.set_float32_matmul_precision(prev)

        def test_one(pinned):
            assert torch.get_float32_matmul_precision() == "high"
    """,
    # COLLECTION window: a module-level statement.
    "test_importleak.py": """
        import torch
        torch.set_default_dtype(torch.float64)

        def test_never_reached():
            pass
    """,
}


@pytest.fixture(scope="module")
def planted_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("tsg_planted")
    with open(repo_path("conftest.py")) as f:
        (d / "conftest.py").write_text(f.read())
    (d / "pytest.ini").write_text("[pytest]\npython_files = test_*.py\n")
    for name, body in _PLANTED.items():
        (d / name).write_text(textwrap.dedent(body))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(src_root()), env.get("PYTHONPATH", "")])
    for k in ("GEN3AI_SKIP_DEPS_GUARD", "GEN3AI_SKIP_SLOW_STATUS_RECORD", "GEN3AI_SKIP_TIER_BUDGET"):
        env[k] = "1"
    env.pop("GEN3AI_SKIP_TORCH_STATE_GUARD", None)
    env.pop("PYTEST_XDIST_WORKER", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist", "-rA", "--color=no",
         "--continue-on-collection-errors", "-o", "addopts=", str(d)],
        cwd=str(d), env=env, capture_output=True, text=True, timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def _outcome(out, nodeid_suffix):
    """The `-rA` summary outcomes for one test: e.g. {'PASSED', 'ERROR'}."""
    return set(re.findall(rf"^(PASSED|FAILED|ERROR|SKIPPED) \S*{re.escape(nodeid_suffix)}\b",
                          out, re.M))


def test_the_run_fails_overall(planted_run):
    rc, out = planted_run
    assert rc != 0, f"a run with four planted leaks exited {rc}:\n{out[-4000:]}"


def test_TEST_window_a_leaking_body_fails_naming_the_global_values_and_test(planted_run):
    _, out = planted_run
    assert "ERROR" in _outcome(out, "test_body.py::test_leaks_threads"), out[-4000:]
    assert re.search(r"test test_body\.py::test_leaks_threads left 1 process-global", out), out[-4000:]
    assert re.search(r"torch\.get_num_threads\(\): \d+ -> \d+", out)


def test_TEST_window_a_dynamo_config_leak_is_named_by_key(planted_run):
    _, out = planted_run
    assert "ERROR" in _outcome(out, "test_body.py::test_leaks_dynamo_config"), out[-4000:]
    # 2.8 renamed the key (`cache_size_limit` is an alias of `recompile_limit` there).
    assert re.search(r"torch\._dynamo\.config\.(cache_size_limit|recompile_limit): \d+ -> 97", out), \
        out[-4000:]


def test_restoring_tests_pass_cleanly(planted_run):
    _, out = planted_run
    for t in ("test_body.py::test_restores_with_try_finally",
              "test_body.py::test_restores_with_the_context_manager",
              "test_body.py::test_restores_with_config_patch",
              "test_modfix_clean.py::test_one"):
        assert _outcome(out, t) == {"PASSED"}, (t, out[-4000:])


def test_MODULE_window_a_leaking_module_fixture_fails_at_the_module_boundary(planted_run):
    _, out = planted_run
    assert _outcome(out, "test_modfix_leak.py::test_one") == {"PASSED"}, out[-4000:]
    assert "ERROR" in _outcome(out, "test_modfix_leak.py::test_two"), out[-4000:]
    assert re.search(r"module test_modfix_leak \(a module/class-scoped fixture", out), out[-4000:]
    assert "torch.get_float32_matmul_precision(): 'highest' -> 'medium'" in out, out[-4000:]


def test_COLLECTION_window_an_import_time_leak_is_a_collection_error(planted_run):
    _, out = planted_run
    assert re.search(r"ERROR collecting test_importleak\.py", out), out[-4000:]
    assert "collecting test_importleak.py (IMPORT-TIME" in out, out[-4000:]
    assert "torch.get_default_dtype(): torch.float32 -> torch.float64" in out, out[-4000:]
