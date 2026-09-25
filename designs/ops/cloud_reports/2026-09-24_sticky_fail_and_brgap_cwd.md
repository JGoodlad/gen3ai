Branch `claude/elegant-faraday-5mmpgi` — final commit: the one adding this report, whose parent is `89f7277` (a file cannot carry its own commit's hash; `git log -1 origin/claude/elegant-faraday-5mmpgi` gives it). Based on `origin/main` @ `34da722`.

# A sticky recorded FAIL, and `main.best_response_gap`'s cwd default — cloud report, 2026-09-24

Two small units from the orchestrator (Orchestrator O5.5). Same rules as the slow-tier recorder task.

| commit | what |
|---|---|
| `5d3882f` | unit 1: a recorded FAIL is STICKY (`slow_tier_status.merge_row`), plus its tests and the `CLAUDE.md` / `designs/ops/testing.md` wording |
| `ea5bb09` | unit 2: `main.best_response_gap` writes no file unless `--json` / `--md` names one, plus its test |
| `89f7277` | backlog: the (d) P2 row moved to §3 DONE; the sticky-FAIL fix added as a §3 DONE row, since it was never a §2 row |
| this one | this report |

## 1. A recorded FAIL is STICKY

**The defect** (the limitation flagged in `2026-09-24_slow_tier_recorder.md` §2). `record_results`
merged a run's rows into the file with `tests.update(results)`, so the newest row always won.
An `inconclusive` row (a timeout, or a test killed before its CALL finished) or a `skip` therefore
**replaced** a recorded `fail`. The red left the gate's fatal set, and nobody had seen the test pass.
Both paths (a timeout after a fail, and an interrupted re-run of a red) went through that one line.

**The fix, by class** (`utils.slow_tier_status.merge_row(recorded, new)`, which `record_results` now
applies per row under the same `flock`):
- **A `fail` is replaced only by a VERDICT**: `pass` (the fix landing, which is how a red is
  cleared) or a newer `fail`.
- **`skip` is treated like `inconclusive`.** The task named inconclusive; I included skip because it
  is the same class: *this run did not measure the test*. A red test that now skips itself (a missing
  browser, a starved box) would otherwise clear the red just as silently.
- **The unverdicted attempt is not lost.** The kept red row gains `held_over: {status, commit, at,
  detail}`, and `Verdict.red_message` prints it ("re-run since, NOT a verdict: inconclusive at commit
  … on …"), so the file still records that the test was re-run and what happened. A later verdict
  replaces the whole row, and the note goes with it.
- **Every other pairing is unchanged**: newest wins. An inconclusive after a PASS still demotes the
  green, as it should. `evaluate` did not change: only `status` decides fatality, `held_over` is an
  extra key it ignores, and the four classes and their fatal/reported split are untouched.

The fix is in `record_results` rather than `merge_outcome`. `merge_outcome` folds one test's PHASES
within a session, where `fail` already outranks everything. The overwrite happened ACROSS runs, at
the file merge.

**Tests (both FAIL on the old `tests.update`):**

| test | shape | on the old code |
|---|---|---|
| `src/slow_tier_status_gate_test.py::test_a_recorded_FAIL_is_STICKY_until_a_real_verdict_replaces_it` | TEMP status file via `record_results`: fail → inconclusive stays fail (held_over set, gate still red, message names the re-run) → skip likewise → a newer fail replaces it → a pass clears it → an inconclusive after that pass still demotes | **FAILS**: the inconclusive replaced the fail (`'inconclusive' == 'fail'`) |
| `src/slow_tier_status_interrupt_test.py::test_an_interrupted_RERUN_of_a_RED_test_leaves_it_RED` | the REAL root conftest in a subprocess session over a TEMP status file **pre-seeded with a `fail` row** for the sleeper, interrupted in flight (serial SIGINT): the row must still be `fail` at the seeded commit, with `held_over.status == "inconclusive"` and the INTERRUPTED detail | **FAILS**: `an interrupted re-run cleared a recorded red: {'status': 'inconclusive', …, 'detail': "INTERRUPTED in flight: …"}` |

Proof method: I swapped the new two-line merge back to `tests.update(results)` and ran both files.
Result: **2 failed, 16 passed**. Restored, 18 passed. The interrupt file passed 3 of 3 consecutive
runs (2.15–2.25 s for its 3 tests).

**Docs:** in the root `CLAUDE.md` gate table, "A recorded FAIL fails the ROUTINE gate … and is STICKY
— only a later PASS or FAIL replaces it". In `designs/ops/testing.md`, the `fail` row of the class
table and the meta-test list (now thirteen). The `slow_tier_status.py` module docstring's class table
and the `record_results` docstring are updated too.

## 2. `main.best_response_gap` no longer writes into cwd

**The defect** (backlog (d) P2, round-2 registration K-1). `--json` defaulted to
`./best_response_gap.json`, so every read without `--json` left a file wherever the caller stood.
Run from the main checkout, it dirtied main.

**The choice: no default output FILE.** A file is written only where `--json PATH` / `--md PATH`
name one. The report is printed to stdout as before, and stderr gets a note: "no --json PATH given:
the JSON report was not written (nothing lands in cwd)". Why this one of the three options:
- **Not a temp path.** An unrequested file in `/tmp` is litter nobody knows to look for, and the
  thing a caller wants from a no-flag run (the table) is already on stdout.
- **Not a refusal inside a git working tree.** That would make the tool's behaviour depend on cwd,
  and the commonest use, a plain "show me the gap" read, would exit non-zero from the one place
  people run it. It also needs a git probe that can itself fail.
- **Existing callers keep working.** Every registered invocation in `designs/` (round-1 and round-2
  registrations, the r1 read README) and both test files pass an explicit `--json`. `--no-json` is
  kept and still wins over `--json`, so the three existing `--no-json` call sites are unchanged. The
  `DEFAULT_JSON` constant is removed; nothing else referenced it.

**Test (FAILS on the old code):**
`src/agents/training/best_response_gap_test.py::test_cli_writes_NOTHING_into_cwd_without_an_explicit_path`.
It runs the CLI on the file's existing `make_run` fixture from an EMPTY cwd (`monkeypatch.chdir`)
with no `--json`, and asserts cwd is still empty and the report reached stdout. It then asserts an
explicit `--json` writes exactly there and cwd stays empty. On the old CLI (the file restored from
`HEAD~` with `git stash`): **FAILS** with `the CLI wrote into cwd: ['best_response_gap.json']`.

## 3. Exactly what I ran

Environment: the same cloud-container venv as the previous report (Python 3.11 with the
`environment.yml` pins for pytest, pytest-asyncio, pytest-xdist, ruff, mypy, numpy and orjson; **no**
torch, no Showdown build, no `models/` archive). Every run used `export PYTHONPATH=$PYTHONPATH:src
GEN3AI_SKIP_DEPS_GUARD=1` and `-p no:cacheprovider`.

| command | result |
|---|---|
| `pytest src/slow_tier_status_interrupt_test.py src/slow_tier_status_gate_test.py src/agents/training/best_response_gap_test.py src/ruff_gate_test.py src/file_size_gate_test.py src/claude_md_freshness_gate_test.py src/test_stub_vacuity_gate_test.py src/agents/model/mypy_gate_test.py` | **99 passed, 1 failed**: the mypy gate (below) |
| `pytest src/slow_tier_status_interrupt_test.py` ×3 | 3 passed each time |
| `python -m mypy` | 52 errors in 24 files, the same count this venv gave on the untouched base `3265ec8` last time. Its scope (`src/agents/model`, `src/agents/observation`) contains nothing this change touches. The errors come from the environment (no torch or SB3) |
| `pytest src/main/best_response_gap_integration_test.py` | **1 failed, 22 skipped, on this branch AND on the untouched base** (`git stash`, same result). `test_cli_refuses_then_prints` needs the `models/` archive and, unlike its siblings, does not call `_archive()` to skip without it, so it dies on `REFUSAL (run_read) … no such run directory`. Not caused by this change, and it passes explicit `--json`/`--no-json`, so it is unaffected by it. ⚠️ Its missing skip is a small defect of its own, which I did not fix: out of scope |

`designs/ops/slow_tier_status.json` is **not modified**: every recorder run here wrote to a temp file.

## 4. Not verified

- **The routine gate and the full suite.** Neither can run here (no torch, no Showdown build, no rust
  build). The orchestrator runs the routine gate before landing.
- **`best_response_gap_integration_test.py` against the real archive.** It needs `models/`, which
  only the main checkout has. My change touches only the default output path, and that test names
  its paths explicitly.
- **A clean mypy run.** Not possible in this venv. The error count is unchanged from the base, and
  mypy's scope contains nothing this change touches.
