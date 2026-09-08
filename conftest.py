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


def pytest_configure(config):
    """Publish the live `slow`-set object on `config`, so the gate test reads it off `request.config`
    rather than by importing the root conftest as a module (which depends on sys.path order)."""
    config._gen3ai_slow_collected = _slow_collected


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
        _slow_results[report.nodeid] = {
            "status": merged,
            "duration_s": (prev or {}).get("duration_s", 0.0) + float(report.duration),
            "detail": detail,
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
        from utils.slow_tier_status import make_row, record_results
        commit = get_git_hash()
        factor = _contention_factor()
        wanted = _slow_results if nodeids is None else {
            n: _slow_results[n] for n in nodeids if n in _slow_results}
        rows = {nodeid: make_row(r["status"], commit=commit, duration_s=r["duration_s"],
                                 contention=factor, detail=r.get("detail", ""))
                for nodeid, r in wanted.items()}
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


def _tier_budget_seconds():
    try:
        from utils.contention import cpu_contention_factor
        return _TIER_BUDGET_BASE_S * cpu_contention_factor()
    except Exception:
        return _TIER_BUDGET_BASE_S      # never let the guard break collection


def pytest_runtest_logreport(report):
    """Two jobs: the tier-budget overrun above, and the slow-tier status recording below."""
    _record_slow_result(report)         # every phase, so a setup/teardown error is not lost
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
    for note in _slow_write_note:
        terminalreporter.write_line(note)
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
    _write_slow_results()
    # Fail ONLY on a trustworthy measurement. Scaling the budget is not enough on its own: the
    # factor is loadavg/cpus (~1.4 at load 22), while the actual slowdown on a core-hungry test is
    # multiples of that, so a scaled budget still false-fails beside a live training run — the
    # exact failure this tree has eaten repeatedly (a starved parity run reporting 39/40 bogus
    # skips as a clean pass). Contended ⇒ advisory; idle ⇒ a real verdict.
    if _over_budget and exitstatus == 0 and _box_is_idle():
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
    """Refuse the session on an unfinished checkout, with ONE message instead of ~15 failures."""
    if os.environ.get("GEN3AI_SKIP_DEPS_GUARD"):
        return
    missing = [rel for rel in _REQUIRED_DEPS
               if not os.path.exists(os.path.join(_REPO_ROOT, rel))]
    if not missing:
        return
    import pytest
    raise pytest.UsageError(_DEPS_GUARD_MESSAGE.format(missing=", ".join(missing)))
