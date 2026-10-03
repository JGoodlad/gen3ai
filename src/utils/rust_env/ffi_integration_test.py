"""The FFI front end's GATES on a real build (M5 Lane A, ``designs/endstate/program_rust_core.md`` §2 M5).

Builds THIS checkout's self-check ``cdylib`` into ``src/rust_env/target`` (never main's; the 09-09
incident) and loads it by absolute path through ``ffi.load`` (stamp + schema + table checked first).

* GATE ① (byte equality): Lane 0's gate-① runs — the bridge corpus (6 envs x 400 steps) and the
  ladder COMMIT tier (8 x 600) — are RECORDED in Rust (``tests/ffi_reference_test.rs``: the same
  staging RNG and seeded random policy as ``sim_bridge_parity_test.rs``, threads = 1) and REPLAYED
  through the FFI at threads = 3: every output column after every op is byte-equal to the in-Rust
  run's, and so are the deterministic counters and the refusal bank.
* GATE ② (panics): a Rust panic across the boundary is a typed ``CorePanic``; a panic inside a
  handle's locked section poisons that handle (``LifecycleViolation`` afterwards); the process
  survives. Run in a SUBPROCESS, so an abort would read as a failure here rather than kill the worker.
* GATE ③ (the declared lifecycle through FFI): the ``*_AFTER_FREEZE`` counters are readable and 0
  after the gate-① runs; the teeth — a second freeze and a dispatch naming another column set are
  refused with ``LifecycleViolation`` and the rebind is COUNTED.
* The stamp's teeth at load: a build of the wrong kind is refused before any op.
"""
import ctypes
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from utils.paths import src_path
from utils.rust_env import columns as C
from utils.rust_env import ffi
from utils.rust_env import protocol as P
from utils.rust_env import stamp as S

pytestmark = [pytest.mark.sim, pytest.mark.integration]

FEATURES = ("--profile", "selfcheck", "--features", "emission-selfcheck")


def _cargo() -> str:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the FFI gates cannot run (install rustup)")
    return cargo


def _crate_env():
    return dict(os.environ, CARGO_TARGET_DIR=str(src_path("rust_env", "target")))


@pytest.fixture(scope="module")
def lib_path():
    crate = src_path("rust_env")
    r = subprocess.run([_cargo(), "build", "--lib", *FEATURES, "--manifest-path", str(crate / "Cargo.toml")],
                       env=_crate_env(), capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"building the cdylib failed:\n{r.stderr[-4000:]}"
    p = ffi.default_path("selfcheck")
    assert p.is_absolute() and p.exists(), p
    return p


@pytest.fixture(scope="module")
def lib(lib_path):
    return ffi.load(lib_path, nan_poison=True)


def _record(out, extra=None, test="record_the_gate_1_run_for_the_ffi_front_end"):
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


def _replay(lib, out, threads=3):
    """Replay the recorded inputs through the FFI; assert every output byte-equal. Returns the core
    (open) and the op count."""
    spec = json.loads((out / "spec.json").read_text())
    spec["threads"] = threads
    core = ffi.FfiCore(json.dumps(spec), lib=lib)
    n = core.n
    sizes = C.nbytes(n, core.obs_dim)
    ins = [c for c in C.COLUMNS if c.direction == "in"]
    outs = [c for c in C.COLUMNS if c.direction == "out"]
    wall = {P.counter_index()[k] for k in ("CORE_NS_LAST", "CORE_NS_TOTAL")}
    ops = 0
    with open(out / "trace.bin", "rb") as f:
        while True:
            code = f.read(1)
            if not code:
                break
            for c in ins:
                buf = f.read(sizes[c.name])
                core.cols[c.name][...] = np.frombuffer(buf, dtype=C.numpy_dtype(c)).reshape(core.cols[c.name].shape)
            core.dispatch({o.code: o.name for o in P.OPS}[code[0]])
            for c in outs:
                want = f.read(sizes[c.name])
                assert len(want) == sizes[c.name], "the trace is truncated"
                got = core.cols[c.name]
                if c.name == "counters":
                    w = np.frombuffer(want, dtype="<u8")
                    keep = [i for i in range(len(w)) if i not in wall]
                    assert (got[keep] == w[keep]).all(), f"op {ops}: counters {got.tolist()} vs {w.tolist()}"
                elif got.tobytes() != want:
                    w = np.frombuffer(want, dtype=C.numpy_dtype(c)).reshape(got.shape)
                    bad = [i for i in range(n) if got[i].tobytes() != w[i].tobytes()]
                    pytest.fail(f"op {ops} ({chr(code[0])}): column {c.name} differs at envs {bad}")
            ops += 1
    assert json.loads((out / "bank.json").read_text()) == core.bank(), "the refusal bank differs"
    return core, ops


def _gate_1(lib, tmp_path, name, extra, want_ops):
    out = tmp_path / name
    _record(out, extra)
    core, ops = _replay(lib, out)
    try:
        assert ops == want_ops, f"{ops} ops replayed"
        k = core.counters()
        assert k["DECISIONS"] >= want_ops and k["EPISODES_ENDED"] >= 5, k
        # GATE ③: the declared lifecycle held through the FFI for the whole run.
        assert core.after_freeze() == {c.name: 0 for c in P.COUNTERS if c.after_freeze}, core.after_freeze()
    finally:
        core.close()


def test_gate_1_the_bridge_corpus_through_ffi_is_byte_equal(lib, tmp_path):
    _gate_1(lib, tmp_path, "bridge", {}, 401)


def test_gate_1_the_ladder_commit_tier_through_ffi_is_byte_equal(lib, tmp_path):
    from utils.ladder_corpus import teams

    f = tmp_path / "ladder_commit.txt"
    f.write_text("\n".join(teams("commit")) + "\n")
    _gate_1(lib, tmp_path, "ladder", {"RUST_ENV_PARITY_TEAMS": str(f), "RUST_ENV_PARITY_N": "8",
                                      "RUST_ENV_PARITY_STEPS": "600", "RUST_ENV_PARITY_SEED": "11"}, 601)


def test_gate_1_a_quarantine_through_ffi_is_byte_equal(lib, tmp_path):
    """Lane 0's banked F-M5-1 refusal replayed, then 60 more steps: the quarantine's `done` /
    `refused` / auto-reset and the BANK (kind, class, input log) are byte-equal through the FFI."""
    out = tmp_path / "quarantine"
    _record(out, test="record_the_quarantine_run_for_the_ffi_front_end")
    core, ops = _replay(lib, out, threads=1)
    try:
        bank = core.bank()
        assert len(bank) == 1 and "all candidates eliminated" in json.dumps(bank), bank
        assert core.counters()["REFUSALS"] == 1
        assert core.after_freeze() == {c.name: 0 for c in P.COUNTERS if c.after_freeze}
    finally:
        core.close()


def test_the_byte_comparison_has_teeth(lib, tmp_path):
    """One flipped output byte in the recorded trace must fail the replay."""
    out = tmp_path / "teeth"
    _record(out, {"RUST_ENV_PARITY_N": "2", "RUST_ENV_PARITY_STEPS": "20"})
    t = bytearray((out / "trace.bin").read_bytes())
    sizes = C.nbytes(2, int(lib.rust_env_obs_dim()))
    # op 0: [op][inputs][obs…]: flip a byte inside op 0's obs column.
    at = 1 + sum(sizes[c.name] for c in C.COLUMNS if c.direction == "in") + 7
    t[at] ^= 0x01
    (out / "trace.bin").write_bytes(bytes(t))
    with pytest.raises(pytest.fail.Exception, match="column obs differs"):
        _replay(lib, out)


_PANIC_SCRIPT = r"""
import glob, json, sys
from utils.rust_env import ffi, protocol as P
lib = ffi.load(ffi.default_path("selfcheck"), nan_poison=True)
def err():
    return P.error_from_json(lib.rust_env_last_error().decode())
out = {}
for kind in (0, 1):
    out[f"probe{kind}"] = [lib.rust_env_panic_probe(None, kind), type(err()).__name__, err().message]
teams = sys.argv[1:]
spec = P.spec_json(n=2, threads=2, teams=teams, names=("pa", "pb"), turn_limit=300, refusal_budget=4, bank_dir=None)
core = ffi.FfiCore(spec, lib=lib)
core.cols["ep_team"][:] = [[0, 1], [1, 0]]
core.cols["ep_seed"][:] = [[1, 2, 3, 4], [5, 6, 7, 8]]
core.reset()
out["locked"] = [lib.rust_env_panic_probe(core._h, 2), type(err()).__name__, err().message]
try:
    core.step()
    out["after"] = "no error"
except P.RustEnvError as e:
    out["after"] = [type(e).__name__, e.message]
core.close()
print("RESULT " + json.dumps(out))
"""


def test_gate_2_a_panic_across_the_boundary_is_a_typed_error_not_a_crash(lib_path):
    from utils.ladder_corpus import teams

    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(src_path()), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", _PANIC_SCRIPT, *teams("commit")[:2]], env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"the process died (rc {r.returncode}) — a panic crossed the boundary:\n{r.stderr[-3000:]}"
    res = json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("RESULT "))[7:])
    assert res["probe0"][:2] == [3, "CorePanic"] and "deliberate panic" in res["probe0"][2], res
    assert res["probe1"][:2] == [3, "CorePanic"] and "non-string payload" in res["probe1"][2], res
    assert res["locked"][:2] == [3, "CorePanic"] and "now poisoned" in res["locked"][2], res
    assert res["after"][0] == "LifecycleViolation" and "POISONED" in res["after"][1], res
    assert "deliberate panic" in r.stderr, "the panic hook's message is the evidence the panic really happened"


def _small_core(lib, n=2):
    from utils.ladder_corpus import teams

    spec = P.spec_json(n=n, threads=1, teams=teams("commit")[:3], names=("la", "lb"), turn_limit=300, refusal_budget=4, bank_dir=None)
    return ffi.FfiCore(spec, lib=lib)


def test_gate_3_the_lifecycle_teeth_through_ffi(lib):
    core = _small_core(lib)
    try:
        assert core.after_freeze() == {c.name: 0 for c in P.COUNTERS if c.after_freeze}
        with pytest.raises(P.LifecycleViolation, match="STEP before the first RESET"):
            core.step()
    finally:
        core.close()
    core = _small_core(lib)
    try:
        core.cols["ep_team"][:] = [[0, 1], [2, 0]]
        core.cols["ep_seed"][:] = [[9, 8, 7, 6], [1, 1, 1, 1]]
        core.reset()
        with pytest.raises(P.LifecycleViolation, match="already frozen"):
            core._check(lib.rust_env_freeze(core._h, core._addrs_p))
        other = {c.name: np.zeros_like(core.cols[c.name]) for c in C.COLUMNS}
        addrs = (ctypes.c_size_t * len(C.COLUMNS))(*[other[c.name].ctypes.data for c in C.COLUMNS])
        with pytest.raises(P.LifecycleViolation, match="other than the frozen binding"):
            core._check(lib.rust_env_dispatch(core._h, ord("S"), ctypes.cast(addrs, ctypes.c_void_p)))
        assert core.after_freeze()["COLUMN_REBINDS_AFTER_FREEZE"] == 1, "the refused rebind must be COUNTED"
        with pytest.raises(ValueError):
            core.cols["obs"][0, 0, 0] = 1.0  # an output column is read-only on the Python side
    finally:
        core.close()
    with pytest.raises(P.LifecycleViolation, match="closed"):
        core.step()


def test_a_bad_spec_is_a_typed_caller_error(lib):
    with pytest.raises(P.CallerError, match="missing key"):
        ffi.FfiCore(json.dumps({"n": 1}), lib=lib)


def test_the_stamp_is_checked_at_load_before_any_op(lib_path, tmp_path):
    """A self-check build demanded as a release build is REFUSED at load (a copy at a new path:
    ``dlopen`` would hand back the already-loaded image for the same path)."""
    cp = tmp_path / "libpokesim_env_copy.so"
    shutil.copy2(lib_path, cp)
    with pytest.raises(S.StampMismatch, match="nan_poison"):
        ffi.load(cp, nan_poison=False)
    st = ffi.load(cp, nan_poison=True).rust_env_stamp().decode()
    assert S.parse_stamp(st)["schema"] == C.schema_id()
