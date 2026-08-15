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
# The budget is SCALED BY MEASURED CONTENTION (`cpu_contention_factor`), the same rule the rest of
# the tree uses for wall-clock bounds: on an idle box the factor is 1.0 and nothing changes, and
# beside a live training run the budget stretches instead of producing a false failure. On an
# over-budget test the message carries `describe_contention()`, so a starved run diagnoses itself
# rather than starting an investigation. Escape hatch: GEN3AI_SKIP_TIER_BUDGET=1.
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
_over_budget: "list[tuple[str, float]]" = []


def _tier_budget_seconds():
    try:
        from utils.contention import cpu_contention_factor
        return _TIER_BUDGET_BASE_S * cpu_contention_factor()
    except Exception:
        return _TIER_BUDGET_BASE_S      # never let the guard break collection


def pytest_runtest_logreport(report):
    """Record any default-tier test whose CALL phase overran the budget."""
    if os.environ.get("GEN3AI_SKIP_TIER_BUDGET") or report.when != "call" or report.skipped:
        return
    if any(m in report.keywords for m in _COST_MARKERS):
        return                          # it declared its cost; that is the whole point of a tier
    if report.duration > _tier_budget_seconds():
        _over_budget.append((report.nodeid, report.duration))


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
_IDLE_FACTOR_MAX = 1.05


def _contention_factor():
    """Read FRESH (not off the module's TTL cache): the cached value is up to a minute stale, and a
    verdict printed next to a contradicting load average reads as a bug in the guard."""
    try:
        from utils.contention import cpu_contention_factor
        return cpu_contention_factor(refresh=True)
    except Exception:
        return 1.0         # can't measure ⇒ treat as idle, i.e. hold the guard to its promise


def _box_is_idle():
    """Was this run's timing trustworthy enough to FAIL on?"""
    return _contention_factor() < _IDLE_FACTOR_MAX


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _over_budget:
        return
    try:
        from utils.contention import describe_contention
        diag = describe_contention()
    except Exception:
        diag = "contention: unavailable"
    idle = _box_is_idle()
    terminalreporter.section("TIER BUDGET", red=idle)
    verdict = ("exceeded" if idle else
               "exceeded (ADVISORY — the box was busy, see below)")
    terminalreporter.write_line(
        f"{len(_over_budget)} test(s) {verdict} the {_tier_budget_seconds():.0f}s default-tier "
        f"budget while carrying no cost marker:")
    for nodeid, dur in sorted(_over_budget, key=lambda x: -x[1]):
        terminalreporter.write_line(f"  {dur:7.1f}s  {nodeid}")
    terminalreporter.write_line(
        "Mark each one `slow` (and keep whatever capability marker it already has — `sim`, "
        "`browser`, `integration` say what it NEEDS, `slow` says what it COSTS), or make it "
        "faster. A slow test in the routine gate is how the routine gate stops being routine.")
    # State OUR verdict and the number behind it FIRST. `describe_contention()` calls anything
    # under 1.25 "box looks idle", so on its own it printed "load average 19.90 on 16 cpus (box
    # looks idle)" underneath a failure caused by that very load — the guard appearing to
    # contradict itself. Its load figures and the `ps` hint are still worth having, so it follows.
    factor = _contention_factor()
    terminalreporter.write_line(
        f"contention factor {factor:.2f} (fail threshold <{_IDLE_FACTOR_MAX}) — "
        + ("box quiet enough to judge a duration." if idle else
           "NOT failing the run: a duration measured on a contended box is not a measurement of "
           "the test. Re-run on a quiet box for a verdict — the factor tracks the load average, "
           "and a compile-heavy test competing for every core slows by far more than that "
           "(measured: 12.3s idle -> 65.9s at load 22)."))
    terminalreporter.write_line(diag)


def pytest_sessionfinish(session, exitstatus):
    # Fail ONLY on a trustworthy measurement. Scaling the budget is not enough on its own: the
    # factor is loadavg/cpus (~1.4 at load 22), while the actual slowdown on a core-hungry test is
    # multiples of that, so a scaled budget still false-fails beside a live training run — the
    # exact failure this tree has eaten repeatedly (a starved parity run reporting 39/40 bogus
    # skips as a clean pass). Contended ⇒ advisory; idle ⇒ a real verdict.
    if _over_budget and exitstatus == 0 and _box_is_idle():
        session.exitstatus = 1
