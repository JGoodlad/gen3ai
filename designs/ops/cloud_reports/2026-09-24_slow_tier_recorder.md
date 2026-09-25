Branch `claude/elegant-faraday-5mmpgi` — final commit: the one adding this report, whose parent is `5b583c5` (a file cannot carry its own commit's hash; `git log -1 origin/claude/elegant-faraday-5mmpgi` gives it). Based on `origin/main` @ `3265ec8`.

# The slow-tier recorder banked a KILLED test as PASS — cloud report, 2026-09-24

Task from the orchestrator (Orchestrator O5.5): backlog §2 (b) P0 "The slow-tier recorder can record
a KILLED test as a PASS", plus the (b) P2 `scaffolding_gauge_test` `generated_at` flake.

| commit | what |
|---|---|
| `b21faf2` | the P0 fix (`slow_tier_status.settle` + the conftest wiring), its regression tests, and the doc updates in the root `CLAUDE.md` and `designs/ops/testing.md` |
| `35d5c81` | the P2 flake (a SEPARATE commit, the test file only) |
| `5b583c5` | backlog: both rows moved to §3 DONE with their commits |
| this one | this report |

## 1. Root cause, with a reproduction

**The mechanism.** The root `conftest.py` folds EVERY phase report of a `slow` test into
`_slow_results` (`_record_slow_result`, called from `pytest_runtest_logreport`). A test's first report
is its **SETUP**, and a setup that succeeds is `report.passed`, so `classify(False, False, "")` returns
**`pass`**, with `duration_s` = the setup time (≈ 0.0 s). A normal test then gets a CALL report and a
TEARDOWN report, and the row is banked at teardown. An interrupted test gets neither, and **two
routes bank the setup-only row anyway**:

1. **KeyboardInterrupt** (SIGINT/Ctrl-C, or anything that raises it). pytest's `call_and_report` puts
   `KeyboardInterrupt` in `reraise`, so it leaves the call **without making a call report**. The
   session then unwinds through `pytest_sessionfinish`, and `_write_slow_results()` (the no-argument
   sweep) banks every folded row, including the in-flight one: **`pass`, 0.0 s**.
2. **A SIGTERM'd xdist worker** (the `-n 2` shape of a real tier run). The worker dies. The controller
   survives, prints `FAILED … worker 'gw1' crashed while running …` and synthesizes a failed report.
   But xdist builds that report with **`keywords=()`** and `when="???"`, so `"slow" not in
   report.keywords` and the recorder never sees it. The controller had already folded the relayed
   SETUP report, and its `pytest_sessionfinish` sweep banks it: **`pass`, 0.0 s**. The run's own
   output says FAILED while the artifact says PASS.

Not a route: a plain SIGTERM to a serial pytest, or to the whole process group. Python's default
SIGTERM action kills the process with no Python code run, so nothing is written and the previous
row stands. No signal handler in `src/` is involved (the only SIGTERM handler, in
`main/train/lifecycle.py`, is training-only).

**Reproduction** (scratch, pre-fix conftest `3265ec8`). A temp tree holds a copy of the real root
conftest and two `slow` tests: one passes at once, the other sleeps 30 s. pytest runs under
`$GEN3AI_SLOW_STATUS_FILE=<temp>` and is signalled 5 s in:

| signal | rows written |
|---|---|
| SIGINT → pytest (serial) | `test_quick pass 0.0`, **`test_sleeps pass 0.0`** (exit 2, "KeyboardInterrupt") |
| SIGINT → pytest (`-n 2`) | `test_quick pass 0.0`, **`test_sleeps pass 0.0`** |
| SIGTERM → the xdist WORKER (`-n 2`) | `test_quick pass 0.0`, **`test_sleeps pass 0.0`**, while pytest's own summary says `FAILED … worker 'gw1' crashed` |
| SIGTERM → pytest, or → its process group | only `test_quick pass 0.0`; the in-flight test is not written |

The observed M3 symptom ("PASS rows with 0.0 s durations for two tests still in flight" after a
SIGTERM) matches route 2 exactly (two workers, two in-flight tests). Route 1 would also match if the
stop arrived as SIGINT. ⚠️ I did not have the M3 run's command or process tree, so which of the two
it was is **UNVERIFIED**. The fix covers both, because it keys on the class rather than the signal.

## 2. The fix, and why

**`utils.slow_tier_status.settle(status, call_reported)`: a PASS needs a CALL phase.** The conftest
now carries a `call` flag per test (true once any `when == "call"` report arrives) and passes every
row through `settle` before banking. A `pass` with no call report becomes **`inconclusive`**, with
`detail = INTERRUPTED_DETAIL` ("INTERRUPTED in flight: no call-phase report …"). Nothing else changes:
a setup SKIP and a setup ERROR have no call either, but they keep their own class, and a test whose
call completed banks exactly as before. The teardown bank is unaffected: a normal test always has its
call by then. Only the session-finish sweep ever meets a callless pass.

**INCONCLUSIVE rather than "not recorded" — the decision.**
- It is the class the file already uses for *a run that did not produce a verdict*, the timeout,
  under the standing rule that *a timeout is never a semantic outcome*. A test killed in flight is the
  same class: someone ran out of time on it.
- "Not recorded" would silently keep the PREVIOUS row. That is usually an older PASS, so the artifact
  would read green on a test whose latest attempt never finished, and nothing on the routine gate
  would say so. Inconclusive is REPORTED as a warning on every routine run. That is the visible
  outcome the owner rule asks for.
- The gate's semantics are unchanged: `evaluate` already treats `inconclusive` as reported and never
  fatal, so no reader changed. Only the wording of its warning and docstring now names both causes.

⚠️ **A limitation I did not change:** because inconclusive overwrites a row, an interrupted re-run of
a test recorded `fail` turns that red into a non-fatal inconclusive. The existing timeout path already
does the same (a timeout after a fail replaces it), so this is not new behaviour. It is still a way a
red can quietly leave the fatal set. A follow-up could have `record_results` refuse to replace a
recorded `fail` with an `inconclusive`. I left it out to keep this change to the P0.

## 3. Tests added, and proof each fails on the old code

| test | what it pins | on the old code |
|---|---|---|
| `src/slow_tier_status_interrupt_test.py::test_an_interrupted_slow_test_banks_INCONCLUSIVE_never_PASS[serial-SIGINT]` | copies the REAL root `conftest.py` into `tmp_path`, runs a subprocess pytest over one finished + one sleeping `slow` test, sends SIGINT once the sleeper has **provably started** (it writes a marker file; no fixed sleep), and asserts on a TEMP `$GEN3AI_SLOW_STATUS_FILE`: sleeper = `inconclusive` with the INTERRUPTED detail, finished test = `pass` | **FAILS**: `an in-flight test was banked 'pass' … {'status': 'pass', …, 'duration_s': 0.0, …}` |
| `…[xdist-worker-SIGTERM]` | same, `-n 2`, SIGTERM sent to the **worker** PID (the sleeper writes its own `os.getpid()`) | **FAILS**, same message, `duration_s: 0.0` |
| `src/slow_tier_status_gate_test.py::test_a_PASS_with_no_CALL_phase_settles_INCONCLUSIVE` | pure: `settle` demotes only a callless `pass`; skip/fail keep their class | **FAILS** with `settle` reverted to `return status` (so do both subprocess cases) |
| `src/main/scaffolding_gauge_test.py::test_a_LEGACY_tree…` (P2, modified) | `generated_at` joins `runtime_sec` in the excluded wall-clock set; both are still asserted PUBLISHED | with a scratch pytest plugin that makes every `%Y-%m-%dT%H:%M:%S` strftime return a new second (forcing the straddle), the **old** comparison **FAILS** on `meta` (`'…12:00:01' != '…12:00:00'`) and the new one passes. Unforced, both pass (the flake needs a real second boundary) |

Proof method for the P0: `git show 3265ec8:conftest.py > conftest.py`, run, restore. Result:
**2 failed**. With `settle`'s body reverted: **3 failed, 13 passed** (both subprocess cases plus the
unit). Determinism: the subprocess test passed 5 of 5 consecutive runs (1.47–1.63 s each) and once
under `-n 2`. The marker file means the signal can only land while the call is in flight. A parent
started in the background inherits SIGINT as IGNORED, which would turn the serial case into a
600 s sleep, so the child restores `SIG_DFL` in a `preexec_fn`.

**Tier:** `integration`. It needs out-of-process pytest (and xdist workers) and plays no battles.
It takes ~1.6 s for both cases, so it rides the routine gate, well under the 30 s unmarked budget.
Its wait and communicate bounds use `utils.contention.scale_timeout(60)`, and a sleeper that never
starts fails with "(INCONCLUSIVE)" rather than hanging.

## 4. Exactly what I ran

Environment: a cloud container, **not** `gen3ai_stable`. It had no conda env, no pytest, no torch,
and no `deps/pokemon-showdown` build. I made a venv (`python3 -m venv`, Python 3.11) with the
`environment.yml` pins `pytest==9.0.3 pytest-asyncio==1.3.0 pytest-xdist==3.8.0 ruff==0.16.3
mypy==2.3.1 numpy==2.4.3 orjson==3.11.9`. Every run used `export PYTHONPATH=$PYTHONPATH:src
GEN3AI_SKIP_DEPS_GUARD=1`.

| command (`python -m pytest … -q -p no:cacheprovider`) | result |
|---|---|
| `src/slow_tier_status_interrupt_test.py src/slow_tier_status_gate_test.py src/main/scaffolding_gauge_test.py` | **53 passed** (branch head) |
| the same two slow-tier files with `-n 2` | 16 passed |
| `src/ruff_gate_test.py src/file_size_gate_test.py src/claude_md_freshness_gate_test.py src/test_stub_vacuity_gate_test.py` + the three files above + `src/agents/model/mypy_gate_test.py` | **91 passed, 1 failed**: the mypy gate (below) |
| `src/deps_guard_test.py` | 1 failed, as expected: `test_this_checkout_satisfies_the_guard` fails honestly under `GEN3AI_SKIP_DEPS_GUARD` because this container has no Showdown build |
| `python -m mypy` on this branch vs a worktree of the untouched base `3265ec8` | **52 errors in 24 files in BOTH**. Its scope is `src/agents/model, src/agents/observation` (`mypy.ini`), which this change does not touch. The errors are this bare venv's (no torch/SB3; `unused-ignore` etc.) |

The slow-tier gate's live reading of the committed artifact passed (no recorded red). It reports all
82 rows as stale ("commit unknown or not an ancestor"), which is expected in this container's shallow
clone. **`designs/ops/slow_tier_status.json` is NOT modified**: every recorder run here wrote to a
temp file.

## 5. Not verified

- **The whole routine gate** (`-m "not slow and not e2e"`) and the full suite. Neither can run here:
  there is no torch, no Showdown build and no rust build. Only the files above were run. The
  orchestrator should run the routine gate on the box before landing.
- **The mypy gate on the real env.** The comparison with the base shows this change adds no mypy
  errors, but a clean mypy run was not possible here.
- **Which route M3's SIGTERM actually took** (§1). Both are fixed and both are pinned.
- **Python 3.11.15 exactly.** The venv is the container's Python 3.11 (3.11.x); the pytest, xdist,
  ruff and mypy pins match `environment.yml`.
- Docs changed with the code: the root `CLAUDE.md` gate-table row ("inconclusive (a timeout, or a
  test killed in flight)"), the `designs/ops/testing.md` writer bullet, class table and meta-test
  list, and the `src/utils/slow_tier_status.py` module docstring table. `CHANGELOG.md` and
  `ARCHITECTURE.md` were not touched, since this is not a model change.
