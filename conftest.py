"""Session-wide pytest config: hide the GPU from the whole test suite.

Our unit/integration tests never *require* CUDA — but SB3's ``MaskablePPO`` and
``load_model_snapshot`` default to ``device="auto"``, which SB3's ``get_device`` resolves to
CUDA whenever ``torch.cuda.is_available()``. A few constructions (e.g. in
``src/agents/model/snapshot_test.py``) omit ``device=``, so on a GPU box they build a policy
on the GPU and allocate a ~300 MB CUDA context — stealing VRAM from, and contending with, a
live training run.

Hiding the GPU here makes the suite deterministic, runnable anywhere, and incapable of touching
the GPU. With CUDA invisible, ``torch.cuda.is_available()`` is False, so every ``device="auto"``
resolves to CPU. (The device-less sites in ``snapshot_test.py`` ALSO pin ``device="cpu"``
explicitly — defense in depth: this conftest is the belt, those pins are the suspenders.)

This must run before torch initializes its CUDA driver. The root ``conftest.py`` is imported by
pytest at startup, before any test module is collected/imported and therefore before the first
``torch.cuda`` call — so setting the env var here is early enough.

Escape hatch: set ``GEN3AI_TEST_ALLOW_GPU=1`` to opt out (e.g. a deliberate GPU perf check).
Note this only affects pytest-collected tests; the ``*_fuzz_test.py`` / ``*_benchmark.py``
scripts are run directly (not via pytest), so they are unaffected and still use the GPU.
"""
import os
import sys

import pytest

if not os.environ.get("GEN3AI_TEST_ALLOW_GPU"):
    # Empty string => no visible CUDA device => torch.cuda.is_available() is False
    # => SB3 device="auto" resolves to CPU. Hard-set (not setdefault) so an already-exported
    # CUDA_VISIBLE_DEVICES can't silently re-expose the GPU; GEN3AI_TEST_ALLOW_GPU is the one
    # opt-out.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- BLAS thread pinning: what makes `pytest -n` a speedup instead of a 6.5x SLOWDOWN ---
#
# Measured on this 16-core box, full unit suite:
#
#     serial, unpinned      167 s
#     serial, pinned        147 s
#     -n 8,   UNPINNED      389 s   <-- 6.5x slower than pinned; `user` time 68 min vs 3 min
#     -n 4,   pinned         56 s
#
# Same cliff, same cause as the one `src/main/thread_pinning_test.py` defends for env workers: at
# the library default of one BLAS thread per core, N pytest workers spawn N x 16 competing threads
# and the box thrashes. Nothing in the suite wants multi-threaded BLAS, and someone trying `-n auto`
# would otherwise measure a slowdown and conclude parallelism does not work here — so this is set
# for them rather than written in a doc they have to find first.
#
# Set before torch is imported (BLAS reads these at init, so setting them later is a no-op) — the
# root conftest is imported by pytest at startup, which is early enough. Escape hatch:
# GEN3AI_TEST_ALLOW_THREADS=1.
if not os.environ.get("GEN3AI_TEST_ALLOW_THREADS"):
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_var] = "1"


# --- The EMISSION SELF-CHECK: every rust child a test spawns is the self-check build -------------
#
# `gen3_core_emission_selfcheck_v1` (`src/rust_sim/src/emission_check.rs`): in the self-check build
# every protocol line the port emits is checked AT THE MOMENT it is emitted — it round-trips through
# the typed `Line`, and each viewer's render is exactly the line that viewer is owed (no secret HP,
# no owner-only line, at the other side) — and a failure kills the child. `cargo test` has it by
# `debug_assertions`; this puts every pytest-spawned `sim_bridge` / `search_driver` / `core_events`
# on it too, through `utils.bridge.sim_bridge_bin`, which then builds `--profile selfcheck` into
# `target/selfcheck/` (never `target/release/`, the directory a live run's binaries live in).
# setdefault: an explicit `POKESIM_EMISSION_SELFCHECK=0` (a deliberate production-binary check) wins.
os.environ.setdefault("POKESIM_EMISSION_SELFCHECK", "1")


# --- Tier budget: a slow test may not hide in the cheap tier -------------------------------------
#
# The cost tiers (`sim`, `browser`, `e2e` — see the root CLAUDE.md) only pay off if the DEFAULT tier
# stays fast, and the default tier only stays fast if nothing slow sneaks into it. Three separate
# obs-golden regressions rode on main undetected because the routine gate excluded the tier that
# would have caught them, so "remember to mark it" has now failed three times as a mechanism.
#
# WHY RUNTIME AND NOT THE IMPORT GRAPH. The obvious static rule — "a test that reaches the battle
# runner is a sim test" — was tried and MEASURED WRONG in both directions: 30 collected test files
# transitively import `run_local_battles`, but nearly all are millisecond unit tests that merely
# import a module which *can* play battles (`snapshot_test`, `thread_pinning_test`, `run_name_test`),
# while the genuinely-slow `gen3_data_obs_parity` test reaches it only transitively and a
# direct-call rule would miss it. Import-reachability is not cost. Runtime is cost, so runtime is
# what this checks.
#
# It fires in whatever run you actually did: run the default tier and something slow is in it, you
# are told immediately. It cannot flag a test it did not run — which is fine, because the full
# pre-ship run is exactly where a mis-tiered test would otherwise reach main.
#
# The budget is SCALED BY THE CONTENTION MEASURED OVER THAT TEST'S OWN CALL WINDOW
# (`utils.cpu_meter`, `gen3_contention_meter_v2`): kernel-integrated /proc/stat occupancy (SMT) and
# /proc/schedstat run-queue counters diffed from just before the test started to just after it
# ended, with the test's own worker subtree subtracted — its own load is its cost, not contention.
# On an idle box the factor is 1.0 and nothing changes. Every over-budget line carries that
# reading, and the summary carries the SESSION's, each naming its source. Escape hatch:
# GEN3AI_SKIP_TIER_BUDGET=1.
#
# ⚠️ ONLY the COST markers exempt a test, never the CAPABILITY ones. A `sim` test is not excused for
# being slow — the six-battle obs-golden is `sim` and runs in ~5 s, and it BELONGS in the routine
# gate. Exempting `sim` wholesale would put it back behind the same wall that hid three obs
# regressions. What a test needs and what it costs are different questions; this guard asks the
# second one.
# SIZED FROM THE MEASURED DISTRIBUTION, with clearance — not set to a round number and hoped for.
# The default tier's slowest legitimate members on a quiet box cluster at 12-20 s
# (`falsify_scan_end_to_end` 12.3/17.6/20.1, the two `watchdog` orphan tests ~15, the compile
# gate 12.3). A 20 s budget therefore sat exactly ON the distribution and flapped: a quiet-box run
# failed on a **0.1 s overshoot** (20.1 vs 20.0), which is noise being reported as a verdict.
#
# 30 s keeps ~50% clearance over that ceiling while still catching what this guard is FOR — a test
# that is minutes long hiding in the routine gate. It would still have caught the case that
# justified the guard (`test_cpu_backward_still_does_not_compile`, 37-59 s, now `slow`).
# A threshold a legitimate test sits on produces flapping, not signal — the same lesson as the
# 1.25 idle line below, learned twice in one day.
_TIER_BUDGET_BASE_S = 30.0
_COST_MARKERS = ("slow", "e2e", "benchmark")
# (nodeid, call duration, the contention reading over that call — None when it could not be read)
_over_budget: "list[tuple]" = []
# Over the BASE budget but within the contention-scaled one: never a verdict, but named in one line
# so a slow test is not silently absorbed by a busy box.
_within_scaled: "list[tuple]" = []


# --- Slow-tier LAST-KNOWN STATUS: a red `slow` test must be visible to the ROUTINE gate ---------
#
# The routine gate is `-m "not slow and not e2e"`, so a `slow` test is DESELECTED — and a deselected
# test cannot fail. A red one is therefore invisible until the next full-suite run, which happens at
# most once before a ship. Measured 2026-09-07: `tb_relevance_test`'s winprob smoke rode main red
# for a day behind that marker, the same shape as the obs-golden linchpin riding red three times
# behind the old `-m "not integration"` cut.
#
# So the slow tier WRITES its verdict and the routine gate READS it: this block records every `slow`
# test that actually RAN, and `src/slow_tier_status_gate_test.py` asserts against the file. The
# whole contract — the four verdict classes, why staleness is reported and never fatal, why the file
# is committed rather than gitignored, and why a MISSING file fails — is in
# `src/utils/slow_tier_status.py`'s module docstring.
#
# It records nothing at all on a run in which no slow test executed, so the routine gate never
# touches the artifact. Escape hatch: GEN3AI_SKIP_SLOW_STATUS_RECORD=1.
_slow_results: "dict[str, dict]" = {}
_slow_collected: "set[str]" = set()
_slow_write_note = []


# --- the TEST TEMP ROOT lives on the REAL DISK, never tmpfs `/tmp` (gen3_test_tmp_on_disk_v1) -----
#
# `/tmp` on this box is tmpfs: RAM, and a FIXED 1,048,576-inode table. On 2026-09-30 inodes went 42% ->
# 61% in 15 minutes: every concurrent `-n 2` routine gate left a `pytest-of-<user>/pytest-NNNN` of ~80k
# files (~900 per test for the frame-building tests), five sessions were alive across worktrees, and
# pytest keeps the last 3 on top of the in-flight ones. Two halves, both needed:
#   * WHERE — `tempfile.tempdir` and `$TMPDIR` point at `_test_scratch_root()` BEFORE anything asks
#     for a temp dir, so `tmp_path`, every bare `tempfile.mkdtemp()` / `TemporaryDirectory()`, every
#     xdist worker (execnet spawns them after this, with this environ) and every subprocess a test
#     starts land on disk. The root must not itself be tmpfs — it is REFUSED.
#   * HOW LONG — `pytest.ini` sets `tmp_path_retention_policy = failed` and `_count = 1`: a PASSING
#     test's `tmp_path` is removed at its teardown and a passing session's basetemp at sessionfinish.
# Pinned by `src/utils/pytest_tmp_on_disk_test.py` (a child session; fails on revert of either half).


def _scratch_module():
    """`src/utils/scratch.py` — the ONE scratch-root helper (shared with `agents.model.compile_cache`),
    loaded BY PATH: this runs before anything guarantees `src/` is importable, and a `sys.path` edit
    here would mask the import-precedence gates. The file is stdlib-only by contract."""
    import importlib
    import importlib.util
    mod = sys.modules.get("_gen3ai_conftest_scratch")
    if mod is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "utils", "scratch.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("_gen3ai_conftest_scratch", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            # a COPY of this conftest in a test's temp dir (the conftest's own tests do that): the
            # same module, reached through the import path the child session was given
            mod = importlib.import_module("utils.scratch")
        sys.modules["_gen3ai_conftest_scratch"] = mod
    return mod


def _test_scratch_root():
    """The REAL-disk temp root (`utils.scratch.scratch_root`); a tmpfs/ramfs root is a usage error."""
    m = _scratch_module()
    try:
        return m.scratch_root()
    except m.ScratchOnRamError as exc:
        raise pytest.UsageError(str(exc)) from exc


def _declare_test_tmp_root():
    import tempfile
    root = _test_scratch_root()
    os.environ["TMPDIR"] = root
    tempfile.tempdir = root


# --- K3: a HERMETIC compile cache per test process (gen3_hermetic_compile_cache_v1) --------------
#
# Every pytest process (the controller and each xdist worker) compiles into a FRESH private temp dir,
# created here and deleted at unconfigure — never torch's shared `/tmp/torchinductor_<user>` and
# never a dir any other run, pin or torch env wrote. Overwrites an inherited value on purpose: a
# test run is its own run. Subprocesses a test spawns inherit it. The names mirror
# `agents.model.compile_cache` (pinned by `compile_cache_test.py`); inlined because this runs before
# anything guarantees `src/` is importable.
_COMPILE_CACHE_ENV = ("GEN3AI_COMPILE_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
_compile_cache_session = {"root": None, "pid": None}


def _sweep_dead_test_compile_caches():
    """Remove this user's `gen3ai_pytest_compile_<pid>_*` / `gen3ai_compile_<pid>_*` dirs whose PID is
    dead (a session that was SIGKILLed never reached unconfigure). Mirrors
    `compile_cache.sweep_stale_temp_caches`."""
    import shutil
    base = _test_scratch_root()
    try:
        names = os.listdir(base)
    except OSError:
        return
    for name in names:
        for prefix in ("gen3ai_pytest_compile_", "gen3ai_compile_"):
            if not name.startswith(prefix):
                continue
            head = name[len(prefix):].split("_", 1)[0]
            if not head.isdigit():
                continue
            try:
                os.kill(int(head), 0)
                continue                                   # alive (or not ours to judge)
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
            path = os.path.join(base, name)
            try:
                if os.lstat(path).st_uid == os.getuid():
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass


def _declare_test_compile_cache():
    import atexit
    import tempfile
    _sweep_dead_test_compile_caches()
    root = tempfile.mkdtemp(prefix=f"gen3ai_pytest_compile_{os.getpid()}_", dir=_test_scratch_root())
    atexit.register(_remove_test_compile_cache)          # backstop if unconfigure never runs
    os.environ[_COMPILE_CACHE_ENV[0]] = root
    os.environ[_COMPILE_CACHE_ENV[1]] = os.path.join(root, "inductor")
    os.environ[_COMPILE_CACHE_ENV[2]] = os.path.join(root, "triton")
    _compile_cache_session.update(root=root, pid=os.getpid())


def _remove_test_compile_cache():
    root = _compile_cache_session["root"]
    if root and _compile_cache_session["pid"] == os.getpid():
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def pytest_unconfigure(config):
    """Session teardown — runs after a FAILING session too. SIGKILL is covered by the dead-PID sweep."""
    _remove_test_compile_cache()


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Point every temp dir at the on-disk scratch root (above), then publish the live `slow`-set
    object on `config`, so the gate test reads it off `request.config` rather than by importing the
    root conftest as a module (which depends on sys.path order). Then declare this process's fresh
    compile cache (K3, above) — on the same on-disk scratch root."""
    _declare_test_tmp_root()
    config._gen3ai_slow_collected = _slow_collected
    _declare_test_compile_cache()


def pytest_itemcollected(item):
    """The FULL `slow` set, captured before `-m` deselection can hide it from the routine gate.

    `pytest_itemcollected` fires per item during collection, ahead of every
    `pytest_collection_modifyitems` (which is where `-m` deselection happens) — so a routine run
    that executes zero slow tests still knows exactly which ones it skipped, and can therefore tell
    "recorded green" apart from "never recorded".
    """
    if "slow" in item.keywords:
        _slow_collected.add(item.nodeid)


def _record_slow_result(report):
    """Fold one phase report of one `slow` test into `_slow_results`."""
    if os.environ.get("GEN3AI_SKIP_SLOW_STATUS_RECORD"):
        return
    if "slow" not in report.keywords:
        return
    try:
        from utils.slow_tier_status import classify, merge_outcome
        text = "" if report.passed else str(getattr(report, "longreprtext", "") or report.longrepr)
        status = classify(bool(report.failed), bool(report.skipped), text)
        prev = _slow_results.get(report.nodeid)
        merged = merge_outcome(prev["status"] if prev else None, status)
        detail = (prev or {}).get("detail", "")
        if status in ("fail", "inconclusive") and merged == status:
            # Prefer pytest's own "E   <exception>" line over the trailing "<file>:<line>: <type>"
            # one: the gate's message is read by someone who has NOT got the traceback in front of
            # them, and "AssertionError" alone tells them nothing.
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            errs = [ln[1:].strip() for ln in lines if ln.startswith("E ")]
            detail = (errs[0] if errs else (lines[-1] if lines else "no detail"))
        contention = (prev or {}).get("contention")
        if report.when == "call" and not _meter["worker"]:
            reading = _window_reading(report)       # this test's own call window
            contention = reading.factor if reading is not None else None
        _slow_results[report.nodeid] = {
            "status": merged,
            "duration_s": (prev or {}).get("duration_s", 0.0) + float(report.duration),
            "detail": detail,
            # A PASS is only earned by a CALL phase — see `slow_tier_status.settle`.
            "call": bool((prev or {}).get("call")) or report.when == "call",
            "contention": contention,
        }
    except Exception as exc:            # never let the recorder break a test run
        _slow_write_note.append(f"slow-tier status NOT recorded for {report.nodeid}: {exc!r}")
        return
    # BANK IT AS SOON AS THE TEST IS OVER, not only at session finish. The slow tier is ~2 hours
    # beside a live run, and a session that is interrupted at test 79 of 80 must not throw away 79
    # verdicts — an artifact you only get by not pressing Ctrl-C is an artifact nobody will have.
    # One flock + rewrite of a ~20 KB JSON per slow test is free next to the test itself.
    if report.when == "teardown":
        _write_slow_results([report.nodeid])


def _write_slow_results(nodeids=None):
    """Merge slow results (all of them, or just ``nodeids``) into the committed status artifact."""
    if not _slow_results or os.environ.get("GEN3AI_SKIP_SLOW_STATUS_RECORD"):
        return
    try:
        from utils.git import get_git_hash
        from utils.slow_tier_status import INTERRUPTED_DETAIL, make_row, record_results, settle
        commit = get_git_hash()
        factor = _contention_factor()
        wanted = _slow_results if nodeids is None else {
            n: _slow_results[n] for n in nodeids if n in _slow_results}
        rows = {}
        for nodeid, r in wanted.items():
            # A test interrupted in flight has only its SETUP row here, which reads `pass`; the
            # session-finish sweep below would bank it GREEN. `settle` demotes it to inconclusive.
            status = settle(r["status"], r.get("call", False))
            detail = INTERRUPTED_DETAIL if status != r["status"] else r.get("detail", "")
            own = r.get("contention")
            rows[nodeid] = make_row(status, commit=commit, duration_s=r["duration_s"],
                                    contention=factor if own is None else own, detail=detail)
        if not rows:
            return
        path = record_results(rows)
        if nodeids is None:
            _slow_write_note.append(f"slow-tier status: {len(_slow_results)} row(s) recorded "
                                    f"into {path}")
    except Exception as exc:
        _slow_write_note.append(
            f"slow-tier status COULD NOT BE WRITTEN ({exc!r}) — the rows this run measured are "
            "lost, so the routine gate will keep reading the previous verdict.")


def _tier_budget_seconds(reading=None):
    """The budget for one test: the base, stretched by ``reading`` (that test's window) or, with
    none, by the session reading so far."""
    try:
        factor = (reading or _session_reading()).factor
    except Exception:
        factor = 1.0                    # never let the guard break the run
    return _TIER_BUDGET_BASE_S * factor


# --- The contention METER: sampled across the session, diffed per window -------------------------
#
# Lives in the process that sees every report — the xdist CONTROLLER, or the sole process of a
# serial run — and takes a sample at session start, at most every 2 s on test-start events, at the
# end of every test long enough to matter, and at session end. No thread: a sampler thread in a
# serial run would sit inside the test process, where fork-safety checks count threads.
_meter = {"m": None, "final": None, "worker": False}


def _is_xdist_worker(config):
    return hasattr(config, "workerinput")


def _start_meter():
    try:
        from utils.cpu_meter import WindowMeter
        _meter["m"] = WindowMeter(lambda: [os.getpid()])
    except Exception:
        _meter["m"] = None              # a meter that cannot start leaves the load1 fallback


def pytest_runtest_logstart(nodeid, location):
    m = _meter["m"]
    if m is not None:
        try:
            m.maybe_sample()
        except Exception:
            pass


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Stamp the pid that ran the test, so the controller can subtract THAT worker's own subtree
    (the attribute rides xdist's report serialization untouched)."""
    rep = yield
    rep.gen3ai_pid = os.getpid()
    return rep


def _window_reading(report):
    """Contention over one test's call, or None when it cannot be measured."""
    from utils.cpu_meter import override_reading
    over = override_reading()
    if over is not None:
        return over
    m = _meter["m"]
    start, stop = getattr(report, "start", None), getattr(report, "stop", None)
    if m is None or start is None or stop is None:
        return None
    try:
        return m.window(start, stop, getattr(report, "gen3ai_pid", None))
    except Exception:
        return None


def _session_reading():
    """The whole session's contention (its own process tree subtracted), or an instantaneous
    fallback when no meter ran (a module loaded outside a session, e.g. by its own tests)."""
    from utils.cpu_meter import loadavg_reading, override_reading
    over = override_reading()
    if over is not None:
        return over
    if _meter["final"] is not None:
        return _meter["final"]
    m = _meter["m"]
    if m is None:
        return loadavg_reading()
    return m.session(os.getpid())


def pytest_runtest_logreport(report):
    """Two jobs: the tier-budget overrun above, and the slow-tier status recording below."""
    _record_slow_result(report)         # every phase, so a setup/teardown error is not lost
    if (os.environ.get("GEN3AI_SKIP_TIER_BUDGET") or report.when != "call" or report.skipped
            or _meter["worker"]):
        return                          # an xdist worker's copy: the CONTROLLER judges budgets
    if any(m in report.keywords for m in _COST_MARKERS):
        return                          # it declared its cost; that is the whole point of a tier
    if report.duration <= _TIER_BUDGET_BASE_S:
        return                          # the factor is >= 1, so it cannot be over; skip the read
    reading = _window_reading(report)
    if report.duration > _tier_budget_seconds(reading):
        _over_budget.append((report.nodeid, report.duration, reading))
    else:
        _within_scaled.append((report.nodeid, report.duration, reading))


# A duration is only worth FAILING on when the box was genuinely quiet. This bar is deliberately
# much tighter than `contention.py`'s own 1.25 "looks idle" wording, and the first cut used that
# 1.25 and was WRONG: this box carries a `--nice 10` trainer essentially always, which parks the
# load average around 16-25 on 16 cpus, i.e. a factor of ~1.0-1.6 straddling 1.25. The guard
# therefore flapped exactly where it should have been quiet — a real run failed at factor 1.24 on
# two tests whose measured idle cost is a third of what they showed, while printing "box looks
# idle" next to a load average of 19.9. A gate whose verdict hinges on which side of a knife edge
# a permanent background load happens to land is the flakiness this whole change set exists to
# remove.
#
# At <1.05 the box is unloaded in a way this one rarely is, so in practice the guard REPORTS here
# and ENFORCES on a quiet box (CI, or a deliberate idle run). That asymmetry is the honest one:
# you cannot take a trustworthy duration measurement on a machine that is always training.
#
# THE METER BEHIND IT WAS BLIND, and a threshold is only as good as its meter. Until
# `gen3_contention_meter_v2` the factor was load1/cpus read ONCE at session end. On 2026-09-30 every
# agent's routine gate went RED on this budget with 0 tests failed: load1 ~9-11 on 16 cpus read
# 1.00, "quiet", while several suites and a 312%-CPU measurement ran and the flagged tests took
# 1.3-2.7x their quiet times. Two blind spots: load1 is a 1-minute EMA sampled at one instant, and
# on this 8-core/16-thread box ten busy threads already share cores (1.75x per thread) with NO
# run-queue wait — which PSI `cpu some` cannot see either. `utils.cpu_meter` diffs kernel counters
# over the whole session AND each flagged test's own window; its docstring holds the calibration.
# A run FAILS only when both read < 1.05: a session average can hide the burst that sat on the test.
_IDLE_FACTOR_MAX = 1.05


def _contention_factor():
    """The session's contention factor so far (see `_session_reading`)."""
    try:
        return _session_reading().factor
    except Exception:
        return 1.0         # can't measure ⇒ treat as idle, i.e. hold the guard to its promise


def _box_is_idle():
    """Was this SESSION quiet enough to fail a duration on?"""
    return _contention_factor() < _IDLE_FACTOR_MAX


def _enforceable(entry):
    """An overrun FAILS the run only when its OWN window was quiet too. The session average can hide
    a burst that sat exactly on the one long test, and a window we could not read is not quiet."""
    reading = entry[2] if len(entry) > 2 else None
    if reading is None:
        from utils.cpu_meter import override_reading
        reading = override_reading()
    return reading is not None and reading.factor < _IDLE_FACTOR_MAX


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    for note in _slow_write_note:
        terminalreporter.write_line(note)
    if _within_scaled:
        top = sorted(_within_scaled, key=lambda x: -x[1])[:3]
        terminalreporter.write_line(
            f"tier budget: {len(_within_scaled)} unmarked test(s) ran over "
            f"{_TIER_BUDGET_BASE_S:.0f}s but within their contention-scaled budget (a quiet-box run "
            "decides): " + ", ".join(
                f"{n.split('::')[-1]} {d:.1f}s @x{r.factor:.2f}" if r is not None
                else f"{n.split('::')[-1]} {d:.1f}s" for n, d, r in top))
    if not _over_budget:
        return
    try:
        from utils.contention import describe_contention
        diag = describe_contention()
    except Exception:
        diag = "contention: unavailable"
    idle = _box_is_idle()
    enforced = [e for e in _over_budget if idle and _enforceable(e)]
    terminalreporter.section("TIER BUDGET", red=bool(enforced))
    verdict = ("exceeded" if enforced else
               "exceeded (ADVISORY — the box was busy, see below)")
    terminalreporter.write_line(
        f"{len(_over_budget)} test(s) {verdict} the {_TIER_BUDGET_BASE_S:.0f}s default-tier "
        f"budget (x each test's own contention factor) while carrying no cost marker:")
    for entry in sorted(_over_budget, key=lambda x: -x[1]):
        nodeid, dur = entry[0], entry[1]
        reading = entry[2] if len(entry) > 2 else None
        tag = "ENFORCED" if entry in enforced else "advisory"
        terminalreporter.write_line(f"  {dur:7.1f}s  {nodeid}  [{tag}]")
        terminalreporter.write_line(
            "           window: " + (reading.describe() if reading is not None
                                     else "contention UNREADABLE for this window (not quiet)"))
    terminalreporter.write_line(
        "Mark each one `slow` (and keep whatever capability marker it already has — `sim`, "
        "`browser`, `integration` say what it NEEDS, `slow` says what it COSTS), or make it "
        "faster. A slow test in the routine gate is how the routine gate stops being routine.")
    # State OUR verdict and the number behind it FIRST, with its SOURCE: the old guard printed
    # "box looks idle" beside the very load that caused the overrun. `describe_contention()`'s
    # load figures and `ps` hint follow as a diagnostic, never as the verdict.
    try:
        session = _session_reading().describe()
    except Exception as exc:
        session = f"contention UNREADABLE ({exc!r})"
    terminalreporter.write_line(
        f"session {session} (fail threshold <{_IDLE_FACTOR_MAX}, session AND the test's own "
        "window) — "
        + ("quiet enough to judge a duration." if enforced else
           "NOT failing the run: a duration measured on a contended box is not a measurement of "
           "the test. Re-run on a quiet box for a verdict."))
    terminalreporter.write_line(diag)


def pytest_sessionfinish(session, exitstatus):
    if _meter["m"] is not None and _meter["final"] is None and not os.environ.get(
            "GEN3AI_TIMEOUT_SCALE"):
        try:
            _meter["final"] = _meter["m"].session(os.getpid())   # freeze: summary + exit agree
        except Exception:
            pass
    _write_slow_results()
    # Fail ONLY on a trustworthy measurement. Scaling the budget is not enough on its own — a
    # core-hungry test slows by multiples of any average factor — so contended ⇒ advisory, and
    # quiet (the session AND the test's own window) ⇒ a real verdict.
    if (_over_budget and exitstatus == 0 and _box_is_idle()
            and any(_enforceable(e) for e in _over_budget)):
        session.exitstatus = 1


# --- Fresh-worktree guard: an unfinished checkout must say so ONCE, not 15 times ------------------
#
# A linked git worktree is created with `deps/pokemon-showdown/` present but EMPTY — git materializes
# the submodule PATH, never its contents — and the build artifacts `dist/` + `node_modules/` are
# gitignored INSIDE the submodule, so even `git submodule update --init` leaves them absent. Every
# bridge-backed test then dies inside Node with
#
#     Error: Cannot find module '.../deps/pokemon-showdown/dist/sim/battle-stream'
#
# which pytest reports as ~15 unrelated-looking failures spread across `utils/bridge`, `agents/battle`
# and the obs-golden linchpin. That reads exactly like a real regression in the code under test. On
# 2026-09-06 three separate agents each lost a cycle to it. The defect is not in the tree; the tree
# was never finished being checked out.
#
# So: check the ONE artifact the bridge actually loads, at session start, and fail with a single
# actionable message naming the fix. `scripts/bootstrap.sh` probes the same `dist/sim/index.js`
# (its step 5), deliberately — one file, one meaning, in both places.
#
# WHY `__file__`-RELATIVE AND NOT `utils.paths.repo_root()`. This runs before anything has put `src/`
# on the import path, and it is not repo-root DISCOVERY in the first place: this conftest IS the repo
# root, so `deps/` sitting beside it is a local fact. (`utils/paths.py`'s own docstring names exactly
# this case as the one its helpers are the wrong tool for.)
#
# WHY A SESSION-LEVEL FAILURE AND NOT A SKIP. A skip is the thing this guard exists to prevent: a
# suite that silently drops its battle-backed coverage looks green, and the obs-golden linchpin has
# already ridden main RED three times behind exactly that kind of hole. A checkout that cannot run
# the suite should refuse to report on it rather than report a partial pass. Escape hatch for a
# pure-unit CI that genuinely has no submodule: GEN3AI_SKIP_DEPS_GUARD=1.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SHOWDOWN_DIR = os.path.join(_REPO_ROOT, "deps", "pokemon-showdown")

# `dist/sim/index.js` is the compiled entry point the bridge children load (`local_sim_bridge.js`,
# `replay_kernels.js` and `damage_probe.js` all `require(psPath + '/dist/sim/...')`); `node_modules/`
# is what that compiled code resolves ITS own imports through. Either one missing breaks the bridge,
# and the two are restored by different steps, so both are checked and reported BY NAME.
_REQUIRED_DEPS = (
    os.path.join("deps", "pokemon-showdown", "dist", "sim", "index.js"),
    os.path.join("deps", "pokemon-showdown", "node_modules"),
)

_DEPS_GUARD_MESSAGE = """\
deps/pokemon-showdown is not usable in this checkout.

MISSING: {missing}

This is a CHECKOUT problem, not a test failure. Left alone it surfaces as ~15 failures across
unrelated subsystems, each dying inside Node on "Cannot find module
'.../deps/pokemon-showdown/dist/sim/...'". Nothing is wrong with the code.

FIX (idempotent, run from this checkout):

    ./scripts/bootstrap.sh

or the two manual steps from the root CLAUDE.md "Git Worktree Setup":

    git submodule update --init
    for n in dist node_modules; do \\
      [ -e "deps/pokemon-showdown/$n" ] || \\
        ln -s "<MAIN CHECKOUT>/deps/pokemon-showdown/$n" "deps/pokemon-showdown/$n"; done

Keep the [ -e ] guard, and run it only from a fresh WORKTREE: from the main checkout `dist/` already
exists, so `ln -s` drops the link INSIDE it as dist/dist -> its own parent and every websocket path
dies with ELOOP.

Set GEN3AI_SKIP_DEPS_GUARD=1 to run anyway (a pure-unit CI with no submodule)."""


def pytest_sessionstart(session):
    """Refuse the session on an unfinished checkout, with ONE message instead of ~15 failures; then
    take the contention meter's first sample (controller / sole process only)."""
    if not os.environ.get("GEN3AI_SKIP_DEPS_GUARD"):
        _refuse_unfinished_checkout()
    _refuse_an_unimportable_torch_state_guard()
    _meter["worker"] = _is_xdist_worker(session.config)
    if not _meter["worker"]:
        _start_meter()


def _refuse_unfinished_checkout():
    missing = [rel for rel in _REQUIRED_DEPS
               if not os.path.exists(os.path.join(_REPO_ROOT, rel))]
    if not missing:
        return
    import pytest
    raise pytest.UsageError(_DEPS_GUARD_MESSAGE.format(missing=", ".join(missing)))


# --- Torch GLOBAL-STATE guard: a test that leaks process-global torch state FAILS ----------------
#
# `gen3_torch_state_guard_v1`. Lane E (`c256dd95`): a harness left `torch.set_num_threads(4)` behind,
# the NEXT test built fresh weights at a different thread count (94 of 721 tensors moved by <=7.5e-6),
# an exact-tie argmax flipped, and a compiled-vs-eager parity test failed only when it ran after
# others. Any leaked global makes test ORDER an input. The declared list, the snapshot and the diff
# live in `src/utils/torch_state_guard.py`; this block only decides WHEN to compare. Three windows,
# because a leak can be made in three places and no single window sees all of them:
#
#   TEST       the autouse fixture below: the test body + its function-scoped fixtures. Module- and
#              class-scoped fixtures set up BEFORE any function fixture, so they are outside it.
#   MODULE     from before the first test of a module is set up to after the last one's teardown
#              (which is when pytest finalizes module/class fixtures): a `scope="module"` fixture
#              that sets a global and never restores it.
#   COLLECTION every collector's `collect()` (a test module's import, a conftest.py load): a global
#              set at import time, which happens before ANY test runs and so escapes both of the above.
#
# A difference FAILS, naming the global, both values and the test/module. The guard NEVER restores
# the value — restoring would hide the leak — and has NO allowlist: fix it at the source.
# Per-test cost is two snapshots + a diff (measured in `designs/ops/testing.md`). Escape hatch:
# GEN3AI_SKIP_TORCH_STATE_GUARD=1.

_tsg = {"module": None, "baseline": None, "reported": set(), "primed": False}


def _refuse_an_unimportable_torch_state_guard():
    """A guard that cannot load must SAY so: silently switching it off reads exactly like a clean
    suite. Checked once at session start (cheap: the module imports nothing heavy)."""
    if os.environ.get("GEN3AI_SKIP_TORCH_STATE_GUARD"):
        return
    try:
        import utils.torch_state_guard  # noqa: F401
    except ImportError as exc:
        raise pytest.UsageError(
            f"the torch global-state guard cannot import utils.torch_state_guard ({exc}). Put this "
            "checkout's src/ on the path — `export PYTHONPATH=$PYTHONPATH:src` from the checkout "
            "root, or an ABSOLUTE src path for a subprocess — or set "
            "GEN3AI_SKIP_TORCH_STATE_GUARD=1 to run without the guard.") from exc


def _tsg_mod():
    """The guard module, or None when switched off. `prime()` imports torch once per process so
    its rows always have a BEFORE (the xdist controller runs none of these hooks, so never pays)."""
    if os.environ.get("GEN3AI_SKIP_TORCH_STATE_GUARD"):
        return None
    import utils.torch_state_guard as tsg
    if not _tsg["primed"]:
        tsg.prime()
        _tsg["primed"] = True
    return tsg


@pytest.fixture(autouse=True)
def _torch_global_state_guard(request):
    tsg = _tsg_mod()
    if tsg is None:
        yield
        return
    before = tsg.snapshot()
    yield
    leaks = tsg.diff(before, tsg.snapshot())
    if leaks:
        _tsg["reported"].update(name for name, _, _ in leaks)
        pytest.fail(tsg.describe(f"test {request.node.nodeid}", leaks), pytrace=False)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    tsg = _tsg_mod()
    module = getattr(item, "module", None)
    if tsg is not None and module is not _tsg["module"]:
        _tsg.update(module=module, baseline=tsg.snapshot(), reported=set())
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    result = yield
    tsg = _tsg_mod()
    module = getattr(item, "module", None)
    if tsg is None or _tsg["baseline"] is None:
        return result
    if nextitem is not None and getattr(nextitem, "module", None) is module:
        return result
    # The module's last test in this process: its module/class fixtures were just finalized.
    leaks = [lk for lk in tsg.diff(_tsg["baseline"], tsg.snapshot())
             if lk[0] not in _tsg["reported"]]
    _tsg.update(module=None, baseline=None, reported=set())
    if leaks:
        where = (f"module {getattr(module, '__name__', module)} (a module/class-scoped fixture, "
                 f"or a test whose own leak was already reported; its last test here: "
                 f"{item.nodeid})")
        pytest.fail(tsg.describe(where, leaks), pytrace=False)
    return result


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    tsg = _tsg_mod()
    before = tsg.snapshot() if tsg is not None else None
    report = yield
    if tsg is not None:
        leaks = tsg.diff(before, tsg.snapshot())
        if leaks:
            report.outcome = "failed"
            report.longrepr = tsg.describe(
                f"collecting {collector.nodeid or collector.name} (IMPORT-TIME: a module-level "
                "statement in this file, or in something it imports)", leaks)
    return report


@pytest.fixture
def restore_torch_globals():
    """For a test that sets `torch.set_num_threads` / `torch.set_float32_matmul_precision` in its
    BODY: both are restored at teardown (before the guard above looks). A module-scoped pin uses
    `utils.torch_state_guard.torch_globals(...)` inside a yield fixture instead."""
    from utils.torch_state_guard import torch_globals
    with torch_globals():
        yield
