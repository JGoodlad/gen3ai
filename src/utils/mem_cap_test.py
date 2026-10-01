"""``scripts/ops/mem_cap.sh`` / ``utils.mem_cap`` against REAL processes, REAL scopes and a REAL slice.

Every test runs its jobs in a PRIVATE slice (``$GEN3AI_HEAVY_SLICE`` = ``gen3ai-heavytest<pid>.slice``),
never the shared ``gen3ai-heavy.slice``, and ``systemctl --user revert``s it at teardown (the runtime
drop-ins ``set-property --runtime`` wrote). The load-bearing pins, each FAILS on revert of what it guards:

* a child that allocates past a 200 MB cap is OOM-killed (exit ``EXIT_OOM``) while this test process
  AND a sibling process in the CALLER's own cgroup — the stand-in for the Claude session — survive;
* a well-behaved child passes, exits with its own status and reports a peak >= what it touched;
* two jobs that each fit their own cap but not the SLICE's: the larger is killed (its line names the
  SLICE) and a QUIESCENT sibling (asserted blocked in ``read``) finishes and is reported spared although
  its slice holds the kill; a sibling ALLOCATING through the victim's teardown may die too (a kernel
  memcg race, 2026-10-01) — either way its report is truthful;
* inside the scope ``oom_score_adj`` is +500, the cgroup is the capped scope, the environment passes
  through, and the GPU lock composes with the wrapper in BOTH orders.

``integration``: needs a systemd user manager with the memory controller delegated; SKIPS (with the
reason) where there is none. ~15 s.
"""
from __future__ import annotations

import os
import selectors
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


def _slice_dir(env) -> Path:
    return Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
                f"gen3ai.slice/{env[M.SLICE_ENV]}")


def _events(d: Path) -> dict:
    return {k: int(v) for k, v in (ln.split() for ln in (d / "memory.events").read_text().splitlines())}


def _start_sibling(env, wait_code: str) -> tuple:
    """Start a 120 MB sibling job and return ``(Popen, job pid)`` once it has TOUCHED its memory —
    never on a timer: a sibling still allocating when the big job starts is a second race."""
    code = textwrap.dedent("""
        import os, sys, time
        b = bytearray(120 << 20)
        for i in range(0, len(b), 4096):
            b[i] = 1
        print("allocated", os.getpid(), flush=True)
    """) + textwrap.dedent(wait_code) + '\nprint("sibling done", flush=True)\n'
    sib = subprocess.Popen([_SH, "--name", "sibling", "1", sys.executable, "-c", code], env=env,
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    sel = selectors.DefaultSelector()
    sel.register(sib.stdout, selectors.EVENT_READ)
    assert sel.select(timeout=scale_timeout(60)), "the sibling never reported its allocation"
    line = sib.stdout.readline().split()
    assert line[:1] == ["allocated"], (line, sib.poll())
    return sib, int(line[1])


def _finish(sib) -> tuple:
    """Release a sibling blocked on stdin (one byte; communicate() closes the pipe) and collect it."""
    try:
        out, err = sib.communicate(input="x", timeout=scale_timeout(60))
    except BrokenPipeError:          # it died before reading — communicate() still collects it
        out, err = sib.communicate(timeout=scale_timeout(60))
    return out, err


def test_two_jobs_past_the_slice_cap_kill_the_larger_and_spare_a_QUIESCENT_sibling(env):
    """The guarantee the kernel DOES give: the larger job is the slice OOM's victim, and a sibling
    that is not allocating is never chosen. Its PRECONDITION is asserted, not assumed: the sibling is
    blocked in ``read(0, ...)`` (``/proc/<pid>/syscall``) when the big job runs. The old version of
    this test polled a file every 50 ms — an allocation per wake-up — and on 2026-10-01 two gates saw
    the kernel kill it in the victim's teardown window (see the race test below)."""
    env = {**env, M.SLICE_GB_ENV: "0.4"}
    sib, pid = _start_sibling(env, "sys.stdin.read(1)")
    try:
        t0 = time.monotonic()
        while not Path(f"/proc/{pid}/syscall").read_text().startswith("0 0x0 "):   # read(fd 0, ...)
            assert time.monotonic() - t0 < scale_timeout(30), (
                "the sibling is not quiescent (blocked in read on stdin): "
                + Path(f"/proc/{pid}/syscall").read_text())
            time.sleep(0.02)
        big = _run(["--name", "big", "1", sys.executable, "-c", _ALLOC, "800"], env)
        assert big.returncode == M.EXIT_OOM, (big.returncode, big.stderr)
        assert "SLICE's aggregate cap" in big.stderr, big.stderr
        assert _events(_slice_dir(env))["oom_kill"] >= 1, "the slice recorded no OOM kill"
        out, err = _finish(sib)
        # spared, and REPORTED spared although its slice holds an oom_kill: the verdict is read from
        # the job's OWN scope, never from the slice where every kill under it is counted
        assert sib.returncode == 0, (sib.returncode, err)
        assert "sibling done" in out and "not OOM-killed" in err, (out, err)
    finally:
        if sib.poll() is None:
            sib.kill()
            sib.wait()


def test_a_sibling_ALLOCATING_through_the_victims_teardown_can_die_too_and_is_reported_truthfully(env, tmp_path):
    """The guarantee the kernel does NOT give, pinned as the contract: after the slice OOM kills the
    larger job its memory is released over a few ms, and a sibling that allocates in that window
    finds the slice still full with no eligible victim but itself — it is killed by its OWN
    allocation (kernel log 2026-10-01: both gate failures were exactly this; a 1 MB/ms allocator here
    died in 13 of 20 trials). Either outcome is legal. What must hold is that mem_cap REPORTS it
    truthfully: exit 86 + OOM-KILLED iff the job really died before finishing."""
    env = {**env, M.SLICE_GB_ENV: "0.4"}
    go = tmp_path / "go"
    sib, _ = _start_sibling(env, f"""
        while not os.path.exists({str(go)!r}):
            x = bytearray(1 << 20)
            x[::4096] = b"1" * 256
            del x
            time.sleep(0.001)
    """)
    try:
        big = _run(["--name", "big", "1", sys.executable, "-c", _ALLOC, "800"], env)
        assert big.returncode == M.EXIT_OOM and "SLICE's aggregate cap" in big.stderr, (big.returncode, big.stderr)
        go.write_text("1")
        out, err = _finish(sib)
        if sib.returncode == M.EXIT_OOM:
            assert "OOM-KILLED" in err and "sibling done" not in out, (out, err)
        else:
            assert sib.returncode == 0 and "sibling done" in out and "not OOM-killed" in err, (sib.returncode, out, err)
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
