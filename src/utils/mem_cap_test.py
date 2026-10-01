"""``scripts/ops/mem_cap.sh`` / ``utils.mem_cap`` against REAL processes, REAL scopes and a REAL slice.

Every test runs its jobs in a PRIVATE slice (``$GEN3AI_HEAVY_SLICE`` = ``gen3ai-heavytest<pid>.slice``),
never the shared ``gen3ai-heavy.slice``, and ``systemctl --user revert``s it at teardown (the runtime
drop-ins ``set-property --runtime`` wrote). The load-bearing pins, each FAILS on revert of what it guards:

* a child that allocates past a 200 MB cap is OOM-killed (exit ``EXIT_OOM``) while this test process
  AND a sibling process in the CALLER's own cgroup — the stand-in for the Claude session — survive;
* a well-behaved child passes, exits with its own status and reports a peak >= what it touched;
* two jobs that each fit their own cap but not the SLICE's: the larger is killed (its line names the
  SLICE), the sibling job finishes, the parent survives;
* inside the scope ``oom_score_adj`` is +500, the cgroup is the capped scope, the environment passes
  through, and the GPU lock composes with the wrapper in BOTH orders.

``integration``: needs a systemd user manager with the memory controller delegated; SKIPS (with the
reason) where there is none. ~15 s.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from utils import mem_cap as M
from utils.contention import scale_timeout
from utils.paths import repo_path, src_root

pytestmark = pytest.mark.integration

_SH = str(repo_path("scripts", "ops", "mem_cap.sh"))
_GPU_SH = str(repo_path("scripts", "ops", "gpu_lock.sh"))
_ALLOC = textwrap.dedent("""
    import sys, time
    mb = int(sys.argv[1]); hold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    b = bytearray(mb << 20)
    for i in range(0, len(b), 4096):
        b[i] = 1                      # touch every page: RSS, not just a reservation
    print("allocated", mb, flush=True)
    time.sleep(hold)
""")


def _have_user_manager() -> str:
    if shutil.which("systemd-run") is None:
        return "no systemd-run"
    r = subprocess.run(["systemctl", "--user", "is-system-running"], env=M._env(), capture_output=True, text=True)
    if r.stdout.strip() not in ("running", "degraded"):
        return f"no user manager ({r.stdout.strip() or r.stderr.strip()})"
    ctl = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/cgroup.subtree_control")
    if ctl.exists() and "memory" not in ctl.read_text().split():
        return "the memory controller is not delegated to the user manager"
    return ""


_SKIP = _have_user_manager()
if _SKIP:
    pytestmark = [pytest.mark.integration, pytest.mark.skip(reason=f"mem_cap needs systemd: {_SKIP}")]


@pytest.fixture
def env():
    sl = f"gen3ai-heavytest{os.getpid()}.slice"
    e = {**M._env(), M.SLICE_ENV: sl, M.SLICE_GB_ENV: "4",
         "PYTHONPATH": os.pathsep.join([str(src_root()), os.environ.get("PYTHONPATH", "")])}
    yield e
    subprocess.run(["systemctl", "--user", "revert", sl], env=e, capture_output=True)   # the runtime drop-ins
    subprocess.run(["systemctl", "--user", "stop", sl], env=e, capture_output=True)     # the (empty) slice


def _run(args, env, timeout=60.0, **kw):
    return subprocess.run([_SH, *args], env=env, capture_output=True, text=True, timeout=scale_timeout(timeout), **kw)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    st = Path(f"/proc/{pid}/stat").read_text()
    return st[st.rfind(")") + 2] != "Z"


def test_an_overrun_is_killed_and_the_caller_and_its_session_sibling_survive(env):
    my_cgroup = Path("/proc/self/cgroup").read_text()
    session_sibling = subprocess.Popen(["sleep", "120"])           # lives in the CALLER's cgroup
    try:
        r = _run(["--name", "overrun", "0.2", sys.executable, "-c", _ALLOC, "1024"], env)
        assert r.returncode == M.EXIT_OOM, (r.returncode, r.stdout, r.stderr)
        assert "OOM-KILLED" in r.stderr and "its OWN cap" in r.stderr, r.stderr
        assert "allocated" not in r.stdout, "the child got past its cap"
        assert _alive(session_sibling.pid), "a process in the caller's cgroup died with the capped job"
        assert Path("/proc/self/cgroup").read_text() == my_cgroup
        unit = next(w for w in r.stderr.split() if w.startswith(M.UNIT_PREFIX) and w.endswith(".scope"))
        assert M.unit_props(unit, "LoadState")["LoadState"] == "not-found", "the failed scope was left loaded"
    finally:
        session_sibling.kill()
        session_sibling.wait()


def test_a_well_behaved_child_passes_with_its_own_exit_and_reports_its_peak(env):
    r = _run(["0.5", sys.executable, "-c", _ALLOC + "\nsys.exit(7)", "100"], env)
    assert r.returncode == 7, (r.returncode, r.stderr)
    assert "allocated 100" in r.stdout
    assert "cap 0.50 GB" in r.stderr and "gen3ai-capped-python" in r.stderr     # printed at start
    done = [ln for ln in r.stderr.splitlines() if ln.startswith("[mem_cap] done:")]
    assert done and "not OOM-killed" in done[0], r.stderr
    peak_gb = float(done[0].split("peak ")[1].split(" GB")[0])
    assert 0.09 <= peak_gb < 0.5, f"peak {peak_gb} GB does not cover the 100 MB the child touched"


def test_two_jobs_past_the_slice_cap_kill_the_larger_and_spare_the_sibling(env, tmp_path):
    env = {**env, M.SLICE_GB_ENV: "0.4"}
    go = tmp_path / "go"
    sibling_code = _ALLOC.replace("time.sleep(hold)", textwrap.dedent(f"""
        import os
        while not os.path.exists({str(go)!r}):
            time.sleep(0.05)
        print("sibling done", flush=True)
    """))
    sib = subprocess.Popen([_SH, "--name", "sibling", "1", sys.executable, "-c", sibling_code, "120"],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        t0 = time.monotonic()
        while sib.poll() is None and not list(Path("/sys/fs/cgroup").glob(
                f"user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/gen3ai.slice/"
                f"{env[M.SLICE_ENV]}/gen3ai-capped-sibling-*.scope")):
            assert time.monotonic() - t0 < scale_timeout(30), "the sibling job never started"
            time.sleep(0.05)
        time.sleep(1.0)                                          # let it touch its 120 MB
        big = _run(["--name", "big", "1", sys.executable, "-c", _ALLOC, "800"], env)
        assert big.returncode == M.EXIT_OOM, (big.returncode, big.stderr)
        assert "SLICE's aggregate cap" in big.stderr, big.stderr
        go.write_text("1")
        out, err = sib.communicate(timeout=scale_timeout(60))
        assert sib.returncode == 0, (sib.returncode, err)
        assert "sibling done" in out and "not OOM-killed" in err
    finally:
        if sib.poll() is None:
            sib.kill()
            sib.wait()


def test_inside_the_scope_adj_cgroup_and_env_are_what_the_contract_says(env):
    code = textwrap.dedent("""
        import os
        print(open("/proc/self/oom_score_adj").read().strip())
        print(open("/proc/self/cgroup").read().strip())
        print(os.environ["GEN3AI_MEMCAP_MARKER"])
    """)
    r = _run(["--name", "probe", "0.3", sys.executable, "-c", code], {**env, "GEN3AI_MEMCAP_MARKER": "m-41"})
    assert r.returncode == 0, r.stderr
    adj, cg, marker = r.stdout.split()
    assert adj == str(M.OOM_SCORE_ADJ)
    assert f"/gen3ai.slice/{env[M.SLICE_ENV]}/gen3ai-capped-probe-" in cg and cg.endswith(".scope")
    assert marker == "m-41"


@pytest.mark.parametrize("order", ["gpu_lock_outside", "gpu_lock_inside"])
def test_it_composes_with_the_gpu_lock_in_both_orders(env, tmp_path, order):
    e = {**env, "GEN3AI_GPU_LOCK": str(tmp_path / "gpu.lock")}
    inner = [sys.executable, "-c", textwrap.dedent("""
        from utils.gpu_lock import gpu_lock, verified_holder
        assert verified_holder() is not None, "the GPU lock marker did not survive the wrapper"
        with gpu_lock(timeout_s=10):
            print("ok")
    """)]
    cmd = [_GPU_SH, _SH, "0.5", *inner] if order == "gpu_lock_outside" else [_SH, "0.5", _GPU_SH, *inner]
    r = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=scale_timeout(60))
    assert r.returncode == 0, (r.returncode, r.stderr)
    assert r.stdout.strip() == "ok" and "[mem_cap] done:" in r.stderr


@pytest.mark.parametrize("args", [[], ["abc", "true"], ["0", "true"], ["1"]])
def test_a_bad_invocation_is_refused_without_running_anything(env, args):
    r = _run(args, env)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "REFUSING" in r.stderr


def test_the_default_unit_name_sees_through_the_usual_wrappers():
    assert M.default_name(["timeout", "2h", "python", "-m", "main.compile_inventory", "run"]) == "main.compile_inventory"
    assert M.default_name(["/x/scripts/ops/gpu_lock.sh", "python3", "bench.py"]) == "python3"
    assert M.default_name(["env", "A=1", "nice", "-n", "10", "cargo", "build"]) == "cargo"
    assert M.sanitize("a b/c:d") == "a_b_c_d"
