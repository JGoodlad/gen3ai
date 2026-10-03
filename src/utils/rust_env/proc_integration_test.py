"""The process front end's GATES on a real build (M5 Lane B, ``designs/endstate/program_rust_core.md`` §2 M5).

Builds THIS checkout's self-check ``cdylib`` AND ``rust_env_proc`` into ``src/rust_env/target`` (never
main's; the 09-09 incident) and spawns the child by absolute path through ``proc.ProcCore``.

* GATE ① FFI == PROCESS, byte for byte: Lane A's recorded in-Rust runs (``tests/ffi_reference_test.rs``
  — the bridge corpus 6 x 400, the ladder COMMIT tier 8 x 600, F-M5-1's real quarantine + 60 steps)
  are replayed through BOTH front ends side by side at threads = 3: after every op every output
  column of the process front end equals the FFI's AND the recording, the counters too (the
  wall-clock ``CORE_NS_*`` cells masked, F-LA-2), and the refusal bank; a typed failure (a bad spec,
  STEP before RESET, an illegal action) carries the same class, message and input log through both.
* GATE ② a dead child: SIGKILL between ops and DURING an op, an abort, a stopped child past
  ``op_timeout`` → a typed ``CoreProcessDied`` / ``CoreProcessTimeout`` in bounded time, and a FRESH
  core (a new pid, re-stamped — a forged binary is refused at the respawn) that plays after a RESET;
  a panic → ``CorePanic``, the child's core poisoned, ``respawn()`` recovers.
* GATE ③ no leaked segment: after a normal close, a startup error, a poisoned core, a SIGKILLed
  child and a SIGKILLed PARENT — no new ``/dev/shm`` entry of ours, no memfd of ours still open or
  mapped by any process of this user, no child left.
* GATE ④ the stamp refuses a mismatched child BEFORE any op: a self-check build demanded as release,
  and a forged child (right wire, wrong sources) that records every byte it is sent — it is sent none.
* GATE ⑤ the declared lifecycle through the process: every ``*_AFTER_FREEZE`` (the core's and the
  front end's ``PROC_SPAWNS_AFTER_FREEZE``) is 0 after the gate-① runs; STEP before RESET and an
  unknown opcode are the core's ``LifecycleViolation`` and do NOT poison it; outputs are read-only.
"""
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from utils.contention import scale_timeout
from utils.paths import src_path
from utils.rust_env import columns as C
from utils.rust_env import ffi, proc
from utils.rust_env import protocol as P
from utils.rust_env import stamp as S

pytestmark = [pytest.mark.sim, pytest.mark.integration]

FEATURES = ("--profile", "selfcheck", "--features", "emission-selfcheck")
WALL = {P.counter_index()[k] for k in ("CORE_NS_LAST", "CORE_NS_TOTAL")}
MEMFD = "memfd:gen3ai_rust_env"
ZERO = {c.name: 0 for c in P.COUNTERS if c.after_freeze} | {"PROC_SPAWNS_AFTER_FREEZE": 0}


def _cargo() -> str:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the process front end's gates cannot run (install rustup)")
    return cargo


def _crate_env():
    return dict(os.environ, CARGO_TARGET_DIR=str(src_path("rust_env", "target")))


@pytest.fixture(scope="module")
def built():
    crate = src_path("rust_env")
    r = subprocess.run([_cargo(), "build", "--lib", "--bin", proc.BIN_NAME, *FEATURES, "--manifest-path",
                        str(crate / "Cargo.toml")], env=_crate_env(), capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"building the cdylib + child failed:\n{r.stderr[-4000:]}"
    for p in (ffi.default_path("selfcheck"), proc.default_path("selfcheck")):
        assert p.is_absolute() and p.exists(), p
    return proc.default_path("selfcheck")


@pytest.fixture(scope="module")
def lib(built):
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True)


def _record(out, extra=None, test="record_the_gate_1_run_for_the_ffi_front_end"):
    """Lane A's in-Rust recorder (``ffi_integration_test._record``'s twin: the recording is
    front-end-agnostic, F-LA-7)."""
    crate = src_path("rust_env")
    env = _crate_env()
    env["RUST_ENV_FFI_REF_OUT"] = str(out)
    env.update(extra or {})
    r = subprocess.run(
        [_cargo(), "test", *FEATURES, "--manifest-path", str(crate / "Cargo.toml"), "--test", "ffi_reference_test",
         "--", "--ignored", "--exact", test, "--nocapture"],
        env=env, capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0, f"the in-Rust reference run failed:\n{(r.stdout + r.stderr)[-4000:]}"
    assert f"{test} ... ok" in r.stdout, "the recorder did not run"


def _replay_both(lib, out, threads=3):
    """Replay the recorded inputs through the FFI AND the process front end; after every op every
    output column must be byte-equal across the three (recording, FFI, process). Returns
    ``(proc_core, ffi_core, ops)``, both open."""
    spec = json.loads((out / "spec.json").read_text())
    spec["threads"] = threads
    text = json.dumps(spec)
    fc = ffi.FfiCore(text, lib=lib)
    pc = proc.ProcCore(text, nan_poison=True)
    try:
        ops = _replay_into(fc, pc, out)
    except BaseException:
        pc.close()  # a failed replay must not leak a child or its mapping (gate ③'s census sees it)
        fc.close()
        raise
    return pc, fc, ops


def _replay_into(fc, pc, out):
    n = pc.n
    assert (fc.n, fc.obs_dim) == (pc.n, pc.obs_dim)
    sizes = C.nbytes(n, pc.obs_dim)
    ins = [c for c in C.COLUMNS if c.direction == "in"]
    outs = [c for c in C.COLUMNS if c.direction == "out"]
    names = {o.code: o.name for o in P.OPS}
    ops = 0
    with open(out / "trace.bin", "rb") as f:
        while True:
            code = f.read(1)
            if not code:
                break
            for c in ins:
                buf = f.read(sizes[c.name])
                for core in (fc, pc):
                    core.cols[c.name][...] = np.frombuffer(buf, dtype=C.numpy_dtype(c)).reshape(core.cols[c.name].shape)
            fc.dispatch(names[code[0]])
            pc.dispatch(names[code[0]])
            for c in outs:
                want = f.read(sizes[c.name])
                assert len(want) == sizes[c.name], "the trace is truncated"
                g_p, g_f = pc.cols[c.name], fc.cols[c.name]
                if c.name == "counters":
                    w = np.frombuffer(want, dtype="<u8")
                    keep = [i for i in range(len(w)) if i not in WALL]
                    assert (g_p[keep] == w[keep]).all() and (g_p[keep] == g_f[keep]).all(), \
                        f"op {ops}: counters proc {g_p.tolist()} ffi {g_f.tolist()} recorded {w.tolist()}"
                elif g_p.tobytes() != want or g_p.tobytes() != g_f.tobytes():
                    w = np.frombuffer(want, dtype=C.numpy_dtype(c)).reshape(g_p.shape)
                    bad = [i for i in range(n) if g_p[i].tobytes() != w[i].tobytes() or g_p[i].tobytes() != g_f[i].tobytes()]
                    pytest.fail(f"op {ops} ({chr(code[0])}): column {c.name} differs (process vs FFI / recording) at envs {bad}")
            ops += 1
    bank = json.loads((out / "bank.json").read_text())
    assert pc.bank() == bank == fc.bank(), "the refusal bank differs"
    return ops


def _gate_1(lib, tmp_path, name, extra, want_ops, threads=3):
    out = tmp_path / name
    _record(out, extra)
    pc, fc, ops = _replay_both(lib, out, threads)
    try:
        assert ops == want_ops, f"{ops} ops replayed"
        k = pc.counters()
        assert k["DECISIONS"] >= want_ops and k["EPISODES_ENDED"] >= 5, k
        # GATE ⑤: the declared lifecycle held through the process boundary for the whole run.
        assert pc.after_freeze() == ZERO, pc.after_freeze()
    finally:
        pc.close()
        fc.close()


def test_gate_1_the_bridge_corpus_ffi_equals_process(lib, tmp_path):
    _gate_1(lib, tmp_path, "bridge", {}, 401)


def test_gate_1_the_ladder_commit_tier_ffi_equals_process(lib, tmp_path):
    from utils.ladder_corpus import teams

    f = tmp_path / "ladder_commit.txt"
    f.write_text("\n".join(teams("commit")) + "\n")
    _gate_1(lib, tmp_path, "ladder", {"RUST_ENV_PARITY_TEAMS": str(f), "RUST_ENV_PARITY_N": "8",
                                      "RUST_ENV_PARITY_STEPS": "600", "RUST_ENV_PARITY_SEED": "11"}, 601)


def test_gate_1_a_quarantine_ffi_equals_process(lib, tmp_path):
    out = tmp_path / "quarantine"
    _record(out, test="record_the_quarantine_run_for_the_ffi_front_end")
    pc, fc, _ = _replay_both(lib, out, threads=1)
    try:
        bank = pc.bank()
        assert len(bank) == 1 and "all candidates eliminated" in json.dumps(bank), bank
        assert pc.counters()["REFUSALS"] == 1
        assert pc.after_freeze() == ZERO
    finally:
        pc.close()
        fc.close()


def test_the_three_way_comparison_has_teeth(lib, tmp_path):
    """One flipped output byte in the recording must fail the replay."""
    out = tmp_path / "teeth"
    _record(out, {"RUST_ENV_PARITY_N": "2", "RUST_ENV_PARITY_STEPS": "20"})
    t = bytearray((out / "trace.bin").read_bytes())
    sizes = C.nbytes(2, int(lib.rust_env_obs_dim()))
    at = 1 + sum(sizes[c.name] for c in C.COLUMNS if c.direction == "in") + 7
    t[at] ^= 0x01
    (out / "trace.bin").write_bytes(bytes(t))
    with pytest.raises(pytest.fail.Exception, match="column obs differs"):
        _replay_both(lib, out)


def _small_spec(n=2, threads=1, turn_limit=300):
    from utils.ladder_corpus import teams

    return P.spec_json(n=n, threads=threads, teams=teams("commit")[:3], names=("la", "lb"), turn_limit=turn_limit, refusal_budget=4, bank_dir=None)


def _stage(core, seed=0):
    n = core.n
    rng = np.random.default_rng(seed)
    core.cols["ep_team"][:] = [[i % 3, (i + 1) % 3] for i in range(n)]
    core.cols["ep_seed"][:] = rng.integers(1, 65536, (n, 4))


def _random_step(core, rng):
    m = core.cols["mask"]
    pick = np.where(m == 1, rng.random(m.shape), -1.0).argmax(-1)
    core.cols["action"][...] = np.where(core.cols["need"] == 1, pick, -1)
    core.step()


def _typed(fn):
    try:
        fn()
    except P.RustEnvError as e:
        return type(e).__name__, e.message, e.kind, e.py_class, e.env, e.script
    return None


def test_gate_1_a_typed_failure_is_the_same_through_both(lib, built):
    """The error channel is part of the contract: class, message, kind, env and input log equal."""
    bad = json.dumps({"n": 1})
    a = _typed(lambda: ffi.FfiCore(bad, lib=lib))
    b = _typed(lambda: proc.ProcCore(bad, nan_poison=True))
    assert a is not None and a == b, (a, b)
    fc, pc = ffi.FfiCore(_small_spec(), lib=lib), proc.ProcCore(_small_spec(), nan_poison=True)
    try:
        a, b = _typed(fc.step), _typed(pc.step)
        assert a is not None and a[0] == "LifecycleViolation" and a == b, (a, b)
        for core in (fc, pc):
            _stage(core)
            core.reset()
            illegal = [int(np.flatnonzero(core.cols["mask"][i, s] == 0)[0]) for i in range(2) for s in range(2)]
            core.cols["action"][...] = np.array(illegal, dtype=np.int32).reshape(2, 2)
        a, b = _typed(fc.step), _typed(pc.step)
        assert a is not None and a[0] == "CallerError" and a == b, (a, b)
        assert a[5], "a caller error carries the env's input log"
        a, b = _typed(fc.step), _typed(pc.step)  # F-LA-1: a non-lifecycle failure poisoned both
        assert a is not None and a[0] == "LifecycleViolation" and "POISONED" in a[1] and a == b, (a, b)
    finally:
        fc.close()
        pc.close()


# ------------------------------------------------------------------ gate ②: a dead child


def test_gate_2_a_sigkill_between_ops_is_a_typed_error_and_a_fresh_core(built):
    with proc.ProcCore(_small_spec(), nan_poison=True) as core:
        rng = np.random.default_rng(1)
        _stage(core)
        core.reset()
        for _ in range(3):
            _random_step(core, rng)
        old = core.pid
        os.kill(old, signal.SIGKILL)
        t0 = time.monotonic()
        with pytest.raises(proc.CoreProcessDied, match="SIGKILL") as ei:
            _random_step(core, rng)
        assert time.monotonic() - t0 < 60, "the death must be seen at once (EOF), never waited on"
        assert ei.value.respawned and ei.value.returncode == -signal.SIGKILL
        assert core.pid != old and _exited(old)
        assert core.after_freeze()["PROC_SPAWNS_AFTER_FREEZE"] == 1, "a respawn is COUNTED, never silent"
        assert not core.cols["need"].any() and not core.cols["counters"].any(), "a respawn zeroes the outputs"
        with pytest.raises(P.LifecycleViolation, match="STEP before the first RESET"):
            core.step()  # a FRESH core
        core.reset()
        assert core.cols["need"].sum() == 4
        for _ in range(5):
            _random_step(core, rng)
        assert core.counters()["DISPATCHES"] == 7


def test_gate_2_a_sigkill_during_an_op(built):
    with proc.ProcCore(_small_spec(n=8, threads=1), nan_poison=True) as core:
        rng = np.random.default_rng(2)
        _stage(core)
        core.reset()
        pid = core.pid
        threading.Timer(0.3, lambda: os.kill(pid, signal.SIGKILL)).start()
        t0 = time.monotonic()
        with pytest.raises(proc.CoreProcessDied) as ei:
            while time.monotonic() - t0 < 120:
                _random_step(core, rng)
        assert ei.value.respawned and ei.value.returncode == -signal.SIGKILL
        _stage(core)
        core.reset()
        _random_step(core, rng)


def test_gate_2_an_abort_takes_only_the_child(built):
    with proc.ProcCore(_small_spec(), nan_poison=True) as core:
        with pytest.raises(proc.CoreProcessDied, match="SIGABRT") as ei:
            core._control("ABORT_PROBE")
        assert ei.value.respawned
        _stage(core)
        core.reset()


def test_gate_2_a_stopped_child_times_out_and_is_replaced(built):
    # The op bound applies to the SIGSTOP'd step ONLY (deletion pass P7): built with `op_timeout=2.0`,
    # the setup `reset()` and the post-respawn `reset()` were bounded by the same unscaled 2 s, so a
    # busy box could time out a HEALTHY child — a flake that is not the property under test. The
    # stopped step's bound is scaled by the measured contention (`utils.contention`): it never
    # answers, so a longer bound only waits longer, it cannot pass a broken timeout.
    with proc.ProcCore(_small_spec(), nan_poison=True) as core:
        _stage(core)
        core.reset()
        pid = core.pid
        core.op_timeout = scale_timeout(2.0)
        os.kill(pid, signal.SIGSTOP)
        with pytest.raises(proc.CoreProcessTimeout, match="KILLED") as ei:
            core.step()
        core.op_timeout = None
        assert ei.value.respawned and core.pid != pid and _exited(pid)
        _stage(core)
        core.reset()


def test_gate_2_a_panic_poisons_the_child_and_respawn_recovers(built):
    with proc.ProcCore(_small_spec(), nan_poison=True) as core:
        _stage(core)
        core.reset()
        with pytest.raises(P.CorePanic, match="poisoned"):
            core._control("PANIC_PROBE")
        with pytest.raises(P.LifecycleViolation, match="POISONED"):
            core.step()
        old = core.pid
        core.respawn()
        assert core.pid != old and _exited(old)
        core.reset()
        assert core.after_freeze()["PROC_SPAWNS_AFTER_FREEZE"] == 1


def test_gate_2_without_auto_respawn_the_core_stays_down_until_asked(built):
    with proc.ProcCore(_small_spec(), nan_poison=True, auto_respawn=False) as core:
        os.kill(core.pid, signal.SIGKILL)
        with pytest.raises(proc.CoreProcessDied) as ei:
            _stage(core)
            core.reset()
        assert not ei.value.respawned
        with pytest.raises(proc.CoreProcessDied, match="no child"):
            core.reset()
        core.respawn()
        core.reset()


def _forged_child(tmp_path, stamp: str, *, wire=None, log=None):
    """An executable that prints a handshake (``stamp`` / ``wire``) and records every byte it is sent."""
    log = log or tmp_path / "forged_stdin.bin"
    hs = (f"{proc.HANDSHAKE_TAG}\twire={wire or proc.wire_id()}\tobs_dim=2761\tn_columns={len(C.COLUMNS)}"
          f"\tschema={C.schema_id()}\tstamp={stamp}\n")
    exe = tmp_path / "forged_rust_env_proc"
    exe.write_text(f"#!{sys.executable}\nimport sys\nsys.stdout.write({hs!r}); sys.stdout.flush()\n"
                   f"open({str(log)!r}, 'wb').write(sys.stdin.buffer.read())\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe, log


def _foreign_stamp(lib):
    st = S.parse_stamp(lib.rust_env_stamp().decode())
    st["src"] = "0123456789abcdef"
    return ";".join(f"{k}={v}" for k, v in st.items())


def test_gate_2_the_respawn_is_re_stamped(lib, built, tmp_path):
    """A binary swapped under a live core is REFUSED at the respawn — the death is still typed."""
    exe = tmp_path / "rust_env_proc"
    shutil.copy2(built, exe)
    with proc.ProcCore(_small_spec(), binary=exe, nan_poison=True, startup_timeout=scale_timeout(20)) as core:
        forged, log = _forged_child(tmp_path, _foreign_stamp(lib))
        os.replace(forged, exe)
        os.kill(core.pid, signal.SIGKILL)
        with pytest.raises(proc.CoreProcessDied, match="RESPAWN FAILED: StampMismatch") as ei:
            core.reset()
        assert not ei.value.respawned and isinstance(ei.value.__cause__, S.StampMismatch)
        assert log.read_bytes() == b"", "the refused child was sent nothing"


# ------------------------------------------------------------------ gate ③: no leaked segment


def _exited(pid, wait=10.0):
    """True once ``pid`` is gone or a zombie (an orphan's reaper is not ours to wait for). ``wait``
    is scaled by the measured contention (deletion pass P7)."""
    t = time.monotonic() + scale_timeout(wait)
    while time.monotonic() < t:
        try:
            with open(f"/proc/{pid}/stat") as f:
                if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return True
        except FileNotFoundError:
            return True
        time.sleep(0.05)
    return False


def _our_memfds(ino=None):
    """(open fds, mappings) of OUR memfds across every process of this user (``ino`` narrows)."""
    fds, maps = [], []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    link = os.readlink(f"/proc/{pid}/fd/{fd}")
                except OSError:
                    continue
                if MEMFD in link and (ino is None or os.stat(f"/proc/{pid}/fd/{fd}").st_ino == ino):
                    fds.append((pid, fd))
            with open(f"/proc/{pid}/maps") as f:
                for ln in f:
                    if MEMFD in ln and (ino is None or int(ln.split()[4]) == ino):
                        maps.append((pid, ln.strip()))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return fds, maps


def _self_memfds():
    fds = [fd for fd in os.listdir("/proc/self/fd") if MEMFD in _readlink(f"/proc/self/fd/{fd}")]
    maps = [ln for ln in open("/proc/self/maps") if MEMFD in ln]
    return fds, maps


def _readlink(p):
    try:
        return os.readlink(p)
    except OSError:
        return ""


def _shm():
    return {e for e in os.listdir("/dev/shm") if "rust_env" in e or "gen3ai" in e}


def test_gate_3_no_leak_after_a_normal_close_an_error_a_poison_and_a_sigkill(built):
    shm0, mine0 = _shm(), _self_memfds()  # a baseline: another test's core may still await GC
    children = []

    core = proc.ProcCore(_small_spec(), nan_poison=True)  # normal
    children.append(core.pid)
    _stage(core)
    core.reset()
    assert len(_self_memfds()[0]) > len(mine0[0]) and len(_self_memfds()[1]) > len(mine0[1]), \
        "the census must SEE the live mapping (no vacuous pass)"
    assert _shm() == shm0, "the transport never names a /dev/shm segment"
    core.close()
    assert _self_memfds() == mine0

    with pytest.raises(P.CallerError):  # a startup error
        proc.ProcCore(json.dumps({"n": 1}), nan_poison=True)
    assert _self_memfds() == mine0

    core = proc.ProcCore(_small_spec(), nan_poison=True)  # a poisoned core
    children.append(core.pid)
    with pytest.raises(P.CorePanic):
        core._control("PANIC_PROBE")
    core.close()
    assert _self_memfds() == mine0

    core = proc.ProcCore(_small_spec(), nan_poison=True)  # a SIGKILLed child, respawned, then closed
    children.append(core.pid)
    os.kill(core.pid, signal.SIGKILL)
    with pytest.raises(proc.CoreProcessDied):
        core.reset()
    children.append(core.pid)
    core.close()
    assert _self_memfds() == mine0
    assert _shm() == shm0
    assert all(_exited(p) for p in children), children


_PARENT = r"""
import os, sys, time
from utils.rust_env import proc, protocol as P
spec = P.spec_json(n=2, threads=1, teams=sys.argv[1:], names=("la", "lb"), turn_limit=300, refusal_budget=4, bank_dir=None)
core = proc.ProcCore(spec, nan_poison=True)
core.cols["ep_team"][:] = [[0, 1], [1, 2]]
core.cols["ep_seed"][:] = [[1, 2, 3, 4], [5, 6, 7, 8]]
core.reset()
print("CHILD", core.pid, os.fstat(core._fd).st_ino, flush=True)
time.sleep(600)
"""


def test_gate_3_no_leak_when_the_parent_is_sigkilled(built):
    """The host dies without cleaning up: the child reads EOF and exits, the memfd goes with the
    last reference — nothing named, nothing mapped, nothing running."""
    from utils.ladder_corpus import teams

    shm0 = _shm()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(src_path()), os.environ.get("PYTHONPATH", "")]))
    parent = subprocess.Popen([sys.executable, "-c", _PARENT, *teams("commit")[:3]], env=env,
                              stdout=subprocess.PIPE, text=True)
    try:
        line = parent.stdout.readline()
        assert line.startswith("CHILD "), f"the parent did not start its core: {line!r}"
        _, child, ino = line.split()
        child, ino = int(child), int(ino)
        fds, maps = _our_memfds(ino)
        assert fds and maps, "the census must SEE the live memfd (no vacuous pass)"
    finally:
        os.kill(parent.pid, signal.SIGKILL)  # our own subprocess, by explicit PID
        parent.wait()
    assert _exited(child), f"the orphaned child {child} did not exit on stdin EOF"
    assert _our_memfds(ino) == ([], []), "a process still holds the dead parent's memfd"
    assert _shm() == shm0


# ------------------------------------------------------------------ gate ④: the stamp


def test_gate_4_a_mismatched_build_kind_is_refused_before_any_op(built):
    shm0, mine0 = _shm(), _self_memfds()
    with pytest.raises(S.StampMismatch, match="nan_poison"):
        proc.ProcCore(_small_spec(), nan_poison=False)
    assert _self_memfds() == mine0 and _shm() == shm0


def test_gate_4_a_foreign_child_is_refused_before_it_is_sent_a_byte(lib, built, tmp_path):
    mine0 = _self_memfds()
    exe, log = _forged_child(tmp_path, _foreign_stamp(lib))
    with pytest.raises(S.StampMismatch, match="sources"):
        proc.ProcCore(_small_spec(), binary=exe, nan_poison=True, startup_timeout=scale_timeout(20))
    assert log.read_bytes() == b"", "the spec (or anything else) was sent to a refused child"
    exe2, log2 = _forged_child(tmp_path, lib.rust_env_stamp().decode(), wire="0" * 16,
                               log=tmp_path / "wire_stdin.bin")
    with pytest.raises(proc.ProcLoadError, match="wire"):
        proc.ProcCore(_small_spec(), binary=exe2, nan_poison=True, startup_timeout=scale_timeout(20))
    assert log2.read_bytes() == b""
    assert _self_memfds() == mine0


# ------------------------------------------------------------------ gate ⑤: the lifecycle


def test_gate_5_the_lifecycle_through_the_process(built):
    with proc.ProcCore(_small_spec(), nan_poison=True) as core:
        assert core.after_freeze() == ZERO
        with pytest.raises(P.LifecycleViolation, match="STEP before the first RESET"):
            core.step()
        with core._lock:
            with pytest.raises(P.LifecycleViolation, match="unknown opcode"):
                core._call(ord("Z"), "Z")
        _stage(core)
        core.reset()  # a LIFECYCLE refusal does not poison (F-LA-1)
        _random_step(core, np.random.default_rng(5))
        assert core.after_freeze() == ZERO, core.after_freeze()
        with pytest.raises(ValueError):
            core.cols["obs"][0, 0, 0] = 1.0  # an output column is read-only on the host
    with pytest.raises(P.LifecycleViolation, match="closed"):
        core.step()
