"""The root conftest shuts Inductor's compile-worker pool down after each module (gen3_test_compile_pool_teardown_v1).

A child session runs the REAL root conftest over two planted modules in ONE process: the first does a
CPU Inductor compile (which starts the `compile_worker --workers=N` pool), the second asserts that no
compile-worker process is left under its own pid. With `GEN3AI_KEEP_COMPILE_POOL=1` (the hook off) the
second module must FAIL — so the check is shown to see a live pool, and a revert of the hook fails here.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from utils.paths import repo_path, src_root

_COMPILES = """
    import torch

    def test_a_cpu_inductor_compile_starts_the_pool():
        import torch._inductor.async_compile as ac
        f = torch.compile(lambda x: (x.sin() * 2 + 1).cos().sum())
        f(torch.randn(64))
        assert ac._pool_set, "this torch did not start a compile pool: the probe proves nothing"
"""

_CHECKS = """
    import os

    def _children_cmdlines(me):
        # /proc, not `ps`: ps cuts `args` at 80 columns off a tty, which drops "compile_worker"
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                if int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1]) == me:
                    yield d, open(f"/proc/{d}/cmdline").read().replace("\\0", " ")
            except (OSError, ValueError, IndexError):
                pass

    def test_no_compile_worker_outlives_its_module():
        left = [c for c in _children_cmdlines(os.getpid()) if "compile_worker" in c[1]]
        assert not left, left
"""


def _child(tmp_path, *, keep: bool):
    d = tmp_path / ("keep" if keep else "default")
    d.mkdir()
    (d / "conftest.py").write_text(repo_path("conftest.py").read_text())
    (d / "pytest.ini").write_text("[pytest]\npython_files = *_test.py\n")
    (d / "a_compile_test.py").write_text(textwrap.dedent(_COMPILES))
    (d / "b_check_test.py").write_text(textwrap.dedent(_CHECKS))
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = os.pathsep.join([str(src_root()), env.get("PYTHONPATH", "")])
    env.update(GEN3AI_SKIP_DEPS_GUARD="1", GEN3AI_SKIP_SLOW_STATUS_RECORD="1", GEN3AI_SKIP_TIER_BUDGET="1")
    env.pop("TORCHINDUCTOR_COMPILE_THREADS", None)      # the pool exists only above 1 compile thread
    if keep:
        env["GEN3AI_KEEP_COMPILE_POOL"] = "1"
    else:
        env.pop("GEN3AI_KEEP_COMPILE_POOL", None)
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:xdist", "-q",
                           "--color=no", "a_compile_test.py", "b_check_test.py"],
                          cwd=str(d), env=env, capture_output=True, text=True, timeout=600)


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="one cpu: Inductor never starts a compile pool")
def test_the_compile_pool_is_gone_after_its_module_and_the_check_sees_a_live_one(tmp_path):
    r = _child(tmp_path, keep=False)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    kept = _child(tmp_path, keep=True)
    assert kept.returncode == 1 and "test_no_compile_worker_outlives_its_module" in kept.stdout, \
        kept.stdout[-4000:] + kept.stderr[-2000:]
