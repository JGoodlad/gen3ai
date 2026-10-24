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
  through, and the GPU lock composes with the wrapper in BOTH orders;
* STOPPING (``gen3_mem_cap_stop_v1``): a ``timeout`` OUTSIDE the wrapper stops a job that defers SIGTERM
  (the trainer's deferred abort) — the whole tree, a ``setsid``-ed worker included — after the grace; a job
  that exits on SIGTERM gets its detached leftovers swept; the wrapper's own SIGKILL stops the job too;
  a bad ``--term-grace`` is refused. Each runs a REAL process tree in a REAL scope and fails on revert
  (the old wrapper never returns);
* THE SHORT LANE (2026-10-23): a job declared short (``--max-s``) that runs past declared + margin is stopped (a job that
  DEFERS SIGTERM is SIGKILLed after the grace), labelled ``SHORT-LANE OVERRUN`` and exits 88 so the caller re-queues it as
  a long job; a declared job inside its bound passes with its own exit status. The queue's ordering is
  ``heavy_slots_lane_test.py``.

``integration``: needs a systemd user manager with the memory controller delegated; SKIPS (with the
reason) where there is none. ~15 s.
"""
from __future__ import annotations

import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Tuple

import pytest

from utils import heavy_slots as HS
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
def env(tmp_path):
    sl = f"gen3ai-heavytest{os.getpid()}.slice"
    # never the REAL heavy slots (taken while a run is live), and never a wait on them here
    # (an inherited BLOCKING mark - the gate running THIS file may carry one - would bypass the slots under test)
    # (nor an ambient short-lane declaration / margin / budget: a job declared short is killed, and these tests pick their own)
    ambient = (HS.HELD_ENV, HS.BLOCKING_ENV, HS.MAX_S_ENV, HS.SHORT_MARGIN_ENV, HS.SHORT_JUMPS_ENV)
    e = {**{k: v for k, v in M._env().items() if k not in ambient}, M.SLICE_ENV: sl, M.SLICE_GB_ENV: "4",
         HS.DIR_ENV: str(tmp_path / "heavy_slots"), HS.SLOTS_ENV: "16",
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


# ---------------------------------------------------------------------------- STOPPING A JOB
#: A "trainer" tree, run as a script: the root, a worker in the root's process group and a worker in its OWN
#: session (`setsid`, which a `timeout`'s process-group signal never reaches). Every process DEFERS SIGTERM /
#: SIGINT / SIGHUP (the trainer's deferred abort) and records ``<role> <pid>`` once its handlers are in place.
#: ``coop`` makes the ROOT exit on SIGTERM (the default disposition) after spawning one deferring `setsid`
#: grandchild; ``graceful`` makes it exit 3 on SIGTERM after a short delay.
_TREE = textwrap.dedent("""
    import os, signal, subprocess, sys, time
    pidfile, role = sys.argv[1], sys.argv[2]
    def record(r):
        with open(pidfile, "a") as f:
            f.write(f"{r} {os.getpid()}\\n")
    def defer(*_a):
        pass
    def bye(*_a):
        time.sleep(0.3)
        os._exit(3)
    if role != "coop":
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, bye if role == "graceful" else defer)
    if role in ("root", "coop"):
        subprocess.Popen([sys.executable, __file__, pidfile, "setsid_worker"], start_new_session=True)
    if role == "root":
        subprocess.Popen([sys.executable, __file__, pidfile, "worker"])
    record(role)
    while True:
        time.sleep(0.1)
""")


def _tree_cmd(tmp_path: Path, role: str) -> Tuple[List[str], Path]:
    script, pidfile = tmp_path / "tree.py", tmp_path / "pids.txt"
    script.write_text(_TREE)
    pidfile.write_text("")
    return [sys.executable, str(script), str(pidfile), role], pidfile


def _tree_pids(pidfile: Path, want: int) -> dict:
    """Wait until ``want`` processes of the tree have recorded themselves (handlers installed)."""
    t0 = time.monotonic()
    while True:
        rows = [ln.split() for ln in pidfile.read_text().splitlines() if ln.strip()]
        if len(rows) >= want:
            return {role: int(pid) for role, pid in rows}
        assert time.monotonic() - t0 < scale_timeout(60), f"the tree never came up: {rows}"
        time.sleep(0.05)


def _all_gone(pids, within: float) -> list:
    t0 = time.monotonic()
    while True:
        alive = [p for p in pids if _alive(p)]
        if not alive or time.monotonic() - t0 > scale_timeout(within):
            return alive
        time.sleep(0.05)


def _kill_tree(pidfile: Path) -> None:
    """Teardown: only the pids THIS test's tree recorded, by explicit pid."""
    for ln in pidfile.read_text().splitlines():
        try:
            os.kill(int(ln.split()[1]), signal.SIGKILL)
        except (IndexError, ValueError, ProcessLookupError):
            pass


def test_a_timeout_OUTSIDE_the_wrapper_stops_a_job_that_defers_SIGTERM_and_its_whole_tree(env, tmp_path):
    """The U4 finding (a), 2026-10-02: `timeout 590 mem_cap.sh … <trainer>` ran 12 min past its bound and needed a
    SIGKILL by PID — the trainer defers SIGTERM to a safe point. Here the REAL `timeout` binary wraps the
    wrapper around a tree whose three processes all defer it, one of them in its own session. SIGTERM to the
    `timeout` process is what its expiry does (it forwards the term signal to its child and its group). After the
    grace every process is gone and the wrapper returns; the OLD wrapper never returned (the bounded
    `communicate` below fails on revert, and the teardown kills the tree by pid)."""
    cmd, pidfile = _tree_cmd(tmp_path, "root")
    p = subprocess.Popen(["timeout", "600", _SH, "--term-grace", "3", "0.5", *cmd], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        pids = _tree_pids(pidfile, 3)
        assert set(pids) == {"root", "worker", "setsid_worker"}
        t_stop = time.monotonic()
        os.kill(p.pid, signal.SIGTERM)                      # what `timeout`'s expiry does
        _out, err = p.communicate(timeout=scale_timeout(60))
        took = time.monotonic() - t_stop
        assert not _all_gone(pids.values(), 10), f"a process of the stopped tree survived: {pids}"
        assert "SIGKILL to the" in err and "still in the scope" in err, err
        assert took >= 2.5, f"killed after {took:.1f} s: before the 3 s grace (the job got no time to save)"
        assert p.returncode in (137, 143, 124), (p.returncode, err)
    finally:
        _kill_tree(pidfile)
        if p.poll() is None:
            p.kill()
            p.wait()


def test_a_job_that_exits_on_SIGTERM_has_its_DETACHED_leftovers_swept_too(env, tmp_path):
    """Whole tree, not just the child: the root dies on SIGTERM at once, but its `setsid` grandchild defers it
    and is no longer anyone's child to wait for. The scope must end up empty (SIGTERM to it, then SIGKILL at the
    deadline), the wrapper reporting the command's own signal exit (143)."""
    cmd, pidfile = _tree_cmd(tmp_path, "coop")
    p = subprocess.Popen([_SH, "--term-grace", "3", "0.5", *cmd], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        pids = _tree_pids(pidfile, 2)
        os.kill(p.pid, signal.SIGTERM)                      # the wrapper itself: `kill <pid>` / a bare forward
        _out, err = p.communicate(timeout=scale_timeout(60))
        assert p.returncode == 143, (p.returncode, err)
        assert not _all_gone(pids.values(), 10), f"the detached leftover survived the wrapper: {pids}"
        assert "left in the scope: SIGTERM to them" in err, err
    finally:
        _kill_tree(pidfile)
        if p.poll() is None:
            p.kill()
            p.wait()


def test_the_wrappers_own_SIGKILL_stops_the_job_it_must_not_outlive_its_wrapper(env, tmp_path):
    """`timeout -k` and a `kill -9` of the wrapper's PID cannot be caught: the supervisor asks the kernel
    (`PR_SET_PDEATHSIG`) to tell it, and the job then stops as for any stop signal. Without it the scope lives
    on, uncapped by any slot, with nobody to stop it."""
    cmd, pidfile = _tree_cmd(tmp_path, "root")
    p = subprocess.Popen([_SH, "--term-grace", "3", "0.5", *cmd], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        pids = _tree_pids(pidfile, 3)
        os.kill(p.pid, signal.SIGKILL)
        p.wait(timeout=scale_timeout(30))
        assert not _all_gone(pids.values(), 20), f"the job outlived its SIGKILLed wrapper: {pids}"
    finally:
        _kill_tree(pidfile)
        if p.poll() is None:
            p.kill()
            p.wait()


def test_a_job_that_stops_inside_the_grace_keeps_its_own_exit_status_and_is_not_killed(env, tmp_path):
    cmd, pidfile = _tree_cmd(tmp_path, "graceful")
    p = subprocess.Popen([_SH, "--term-grace", "30", "0.5", *cmd], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        _tree_pids(pidfile, 1)
        os.kill(p.pid, signal.SIGTERM)
        _out, err = p.communicate(timeout=scale_timeout(60))
        assert p.returncode == 3, (p.returncode, err)
        assert "SIGKILL to the" not in err and "forwarded to the job" in err, err
    finally:
        _kill_tree(pidfile)
        if p.poll() is None:
            p.kill()
            p.wait()


@pytest.mark.parametrize("grace", ["0", "-1", "abc", "nan", "inf", ""])
def test_a_bad_term_grace_is_refused_without_running_anything(env, grace):
    r = _run(["--term-grace", grace, "1", "true"], env)
    assert r.returncode == 2 and "REFUSING" in r.stderr and "--term-grace" in r.stderr, (r.returncode, r.stderr)
    r = _run(["1", "true"], {**env, M.TERM_GRACE_ENV: grace or "x"})
    assert r.returncode == 2 and M.TERM_GRACE_ENV in r.stderr, (r.returncode, r.stderr)


def _slot_taker_result(e) -> str:
    r = subprocess.run([sys.executable, "-c",
                        "from utils import heavy_slots as H\n"
                        "with H.heavy_slot(['second'], force=True, wait=False): print('TOOK')"],
                       env=e, capture_output=True, text=True, timeout=scale_timeout(30))
    return "TOOK" if "TOOK" in r.stdout else ("FULL" if "heavy slots are held" in r.stderr else r.stderr)


def test_a_SIGKILLed_wrappers_heavy_slot_lives_as_long_as_its_job_tree(env, tmp_path):
    """D-2 (core review 2026-10-20): the slot's flock was held by the wrapper alone, so a SIGKILL / OOM kill of
    the wrapper freed it while the capped job (its own systemd scope) still ran. The supervisor now stops the job
    when its wrapper dies (gen3_mem_cap_stop_v1) - but a job that DEFERS SIGTERM runs on for the grace, and the
    slot must stay held through it. A REAL wrapper, a REAL tree, a REAL SIGKILL of the wrapper; only processes
    this test started. Fails on a revert of the fd hand-down (`slot_fd` / the supervisor's pass_fds)."""
    e = {**env, HS.SLOTS_ENV: "1"}
    cmd, pidfile = _tree_cmd(tmp_path, "root")
    p = subprocess.Popen([sys.executable, "-m", "utils.mem_cap", "--slot", "--term-grace", "8", "0.5", *cmd],
                         env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        pids = _tree_pids(pidfile, 3)
        assert _slot_taker_result(e) == "FULL", "precondition: the wrapper holds the only slot"
        os.kill(p.pid, signal.SIGKILL)                          # the wrapper, by the pid this test started
        p.wait(timeout=scale_timeout(30))
        assert all(_alive(q) for q in pids.values()), "precondition: the SIGTERM-deferring tree is in its grace"
        row = HS.collect(Path(e[HS.DIR_ENV]), 1)["slots"][0]
        assert row["pids"] and "owner gone" in row["why"], row                 # --status says so, not "free"
        assert _slot_taker_result(e) == "FULL", "the wrapper's death freed the slot while its job still runs (D-2)"
        assert not _all_gone(pids.values(), 40), "the grace never ended the tree"
        deadline = time.monotonic() + scale_timeout(30)
        got = _slot_taker_result(e)
        while got != "TOOK" and time.monotonic() < deadline:
            time.sleep(0.2)
            got = _slot_taker_result(e)
        assert got == "TOOK", f"the slot never freed once the tree was gone: {got}"
    finally:
        _kill_tree(pidfile)
        if p.poll() is None:
            p.kill()
            p.wait()


def test_gpu_lock_outside_mem_cap_takes_no_cpu_slot_beside_a_full_house(env, tmp_path):
    """D-3: `gpu_lock.sh mem_cap.sh …` used to hold the GPU while the job queued for a CPU slot. With the only
    slot held by someone else and `--slot --no-wait` (exit 87 when it must queue), the GPU job still RUNS."""
    e = {**env, HS.SLOTS_ENV: "1", "GEN3AI_GPU_LOCK": str(tmp_path / "gpu.lock")}
    holder = subprocess.Popen([sys.executable, "-c",
                               "import time\nfrom utils.heavy_slots import heavy_slot\n"
                               f"with heavy_slot(['holder'], force=True):\n    open({str(tmp_path / 'h')!r}, 'w').write('1')\n"
                               "    time.sleep(120)"], env=e)
    try:
        deadline = time.monotonic() + scale_timeout(20)
        while not (tmp_path / "h").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert (tmp_path / "h").exists()
        ctl = subprocess.run([_SH, "--slot", "--no-wait", "0.5", "true"], env=e, capture_output=True, text=True,
                             timeout=scale_timeout(60))
        assert ctl.returncode == HS.EXIT_SLOTS_FULL, ("control: mem_cap alone must refuse", ctl.stderr)
        r = subprocess.run([_GPU_SH, _SH, "--slot", "--no-wait", "0.5", "true"], env=e, capture_output=True,
                           text=True, timeout=scale_timeout(60))
        assert r.returncode == 0, (r.returncode, r.stderr)
        assert "takes no CPU heavy slot" in r.stderr
    finally:
        holder.kill()
        holder.wait()


_SLEEPER_PY = ("import os, signal, sys, time\n"
               "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
               "if sys.argv[3] == 'defer':\n"
               "    signal.signal(signal.SIGTERM, signal.SIG_IGN)       # a job that defers its stop (the trainer's abort)\n"
               "time.sleep(float(sys.argv[2]))\n"
               "open(sys.argv[1] + '.done', 'w').write('finished')\n")


@pytest.mark.parametrize("how", ["exit", "defer"])
def test_a_declared_short_job_that_overruns_is_killed_and_labelled_for_a_long_requeue(env, tmp_path, how):
    """The short lane's price: ``--max-s 1`` + the 1 s margin = a 2 s hard timeout. A job that sleeps 120 s is stopped, the
    wrapper says SHORT-LANE OVERRUN / re-queue as a LONG job and exits 88 - and the process is really gone, also when it
    DEFERS SIGTERM (SIGKILL after the ``--term-grace``). The control, the SAME job inside a wider bound, finishes with its
    own exit status. Fails on a revert of the supervisor's deadline (the 120 s job would run to its end) or of the label."""
    e = {**env, HS.SHORT_MARGIN_ENV: "1"}
    pidfile = tmp_path / "pid"
    t0 = time.monotonic()
    r = _run(["--slot", "--max-s", "1", "--term-grace", "3", "0.5", sys.executable, "-c", _SLEEPER_PY, str(pidfile), "120",
              how], e, timeout=90.0)
    took = time.monotonic() - t0
    assert r.returncode == HS.EXIT_SHORT_OVERRUN == M.EXIT_SHORT_OVERRUN == 88, (r.returncode, r.stderr)
    assert "SHORT-LANE OVERRUN" in r.stderr and "re-queue it as a LONG job" in r.stderr, r.stderr
    assert "hard timeout" in r.stderr and not (tmp_path / "pid.done").exists()
    assert took < scale_timeout(60), f"the wrapper outlived a 2 s hard timeout by {took:.0f} s"
    assert not _alive(int(pidfile.read_text())), "the overrunning job is still alive"
    rows = [json.loads(ln) for ln in (Path(env[HS.DIR_ENV]) / HS.EVENTS_FILE).read_text().splitlines()]
    assert [(x["kind"], x.get("lane")) for x in rows] == [("overrun", None), ("slot", "short")], rows   # logged, then released
    # control: the same job, finishing inside declared + margin, passes untouched (and is not labelled)
    ok = _run(["--slot", "--max-s", "20", "--term-grace", "3", "0.5", sys.executable, "-c", _SLEEPER_PY,
               str(tmp_path / "pid2"), "1", how], e, timeout=90.0)
    assert ok.returncode == 0 and "OVERRUN" not in ok.stderr, (ok.returncode, ok.stderr)
    assert (tmp_path / "pid2.done").read_text() == "finished"
    assert "SHORT lane: declared <= 20 s, hard timeout 21 s" in ok.stderr, ok.stderr


def test_a_long_job_is_not_given_a_deadline_by_the_short_lane(env, tmp_path):
    """No declaration, no deadline: the same 3 s job beside a 1 s margin runs to its end."""
    e = {**env, HS.SHORT_MARGIN_ENV: "1"}
    r = _run(["--slot", "0.5", sys.executable, "-c", _SLEEPER_PY, str(tmp_path / "p"), "3", "exit"], e, timeout=90.0)
    assert r.returncode == 0 and (tmp_path / "p.done").exists(), (r.returncode, r.stderr)
