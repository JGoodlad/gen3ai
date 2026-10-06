# Testing — the full chapter

Moved verbatim out of the root `CLAUDE.md` on **2026-09-07**, when the root was reduced to a
constitution + command card + map (~10k tokens, loaded into EVERY session including every
subagent's). **The root keeps the command table, the tier markers, the STATIC-GATE TABLE (seven rows — that
table is the complete list) and the binding hazards; this file is the detail behind them.**

Covers: running beside a live training run (contention-scaled timeouts), the two-axis tier system,
file-naming conventions, which command to run, the static gates, the fuzz-test pattern and its
reproducibility rules, the reward golden, and the benchmarks.

---

## Running Tests

### Running beside a live training run (`gen3_contention_robust_timeouts_v1`)

**The box normally carries a production run**, so any wall-clock timeout measures the spare capacity
as much as the code. This voided three separate investigations and produced one dangerous artifact:
`bridge_impl_parity_test` counted a per-battle TIMEOUT as an "unmodeled move" SKIP, so a starved run
reported 39/40 bogus skips as a **clean pass** that blamed the Rust port for the load average.

Bounds are now **scaled by measured CPU contention** (`src/utils/contention.py`): the per-battle
bridge timeout, every child-REAP timeout, the eval-cycle hung-cycle bound and the poke-env
`race_get` watchdog all read `scale_timeout(...)` at CALL time, where the factor is
`max(1, loadavg / len(sched_getaffinity))`, clamped to 12x. **On an idle box the factor is exactly
1.0, so nothing changes.**

The rules that follow from it:

- **A timeout is never a semantic outcome.** It gets its own bucket, and a run whose timeouts exceed
  25% of attempted battles is declared INCONCLUSIVE rather than reported.
- **Every timeout message self-diagnoses** (`describe_contention()`) — load average plus the `ps`
  command — so a starved failure ends an investigation instead of starting one.
- **Benchmarks get the OPPOSITE treatment: warn, never stretch.** A benchmark's output IS the
  measurement, so scaling its bounds just buys a confidently-reported wrong number.
- **Prefer `ProgressDeadline` (bound the IDLE gap) over a total-duration cap** wherever incremental
  progress is observable — contention stretches duration, but only a real wedge stops progress, and
  a duration cap conflates the two by construction. **Scaling does not rescue a cap**: the factor is
  `loadavg / cpus`, while a starved subprocess slows by a multiple of it. Keep a total as an opt-in
  **livelock** backstop only.
- ⚠️ **A test that PRINTS a diagnostic must not print it in the format of a measurement.** A unit
  test fed synthetic counts into the real warning function and emitted a line indistinguishable from
  a live starvation reading — and it was taken for one during the very investigation that found it.
  Its label is now `SYNTHETIC unit-test sample`.
- **`GEN3AI_TIMEOUT_SCALE=N`** forces the factor. **Run the suite under `GEN3AI_TIMEOUT_SCALE=6`
  after touching any of this** — it proves no test depends on a raw constant the helper scales, and
  it caught two real ones.

The eval cycle is the path MOST exposed, since it runs concurrently with training by design: firing
early does not merely lose a cycle, it collects **partial** results that flow into the curriculum
ramp, the promotion gate and the ELO fit — and a truncated sample is whichever shards got scheduled,
not a random subsample. Measurements + the incident record:
`designs/research_state/claude_md_archive/contention_measurements.md`.
### Test TEMP DIRS live on the real disk, and a PASS leaves nothing (`gen3_test_tmp_on_disk_v1`)

`/tmp` on this box is **tmpfs** — RAM plus a FIXED 1,048,576-inode table. On 2026-09-30 its inodes
went 42% → 61% in 15 minutes: each concurrent `-n 2` routine gate retained a
`pytest-of-<user>/pytest-NNNN` of ~80k files (~900 per frame-building test), five sessions were
alive across worktrees, and pytest's default keeps the last 3 on top of the in-flight ones. Two
halves fix it at the source:

- **WHERE** — the root `conftest.py`'s `pytest_configure` (tryfirst) sets `tempfile.tempdir` and
  `$TMPDIR` to `_test_scratch_root()`: `$GEN3AI_SCRATCH`, else `~/.cache/gen3ai/tmp` — the same
  root the K3 compile caches use. So `tmp_path`, every bare `tempfile.mkdtemp()`, every xdist
  worker and every subprocess a test spawns land on disk. A tmpfs/ramfs root is **REFUSED** with a
  `UsageError` naming `$GEN3AI_SCRATCH`.
- **HOW LONG** — `pytest.ini` sets `tmp_path_retention_policy = failed` and
  `tmp_path_retention_count = 1` (pytest 9.0.3 in both `gen3ai_stable` and `gen3ai_torch28`): a
  passing test's `tmp_path` is removed at its teardown, a passing session's basetemp at
  sessionfinish; a FAILED test's dir is kept for the post-mortem.

Pinned by `src/utils/pytest_tmp_on_disk_test.py`, which runs a CHILD session (`-n 2`, plus a
grandchild subprocess) and fails on a revert of either half. ⚠️ A deep `tmp_path` can exceed a unix
socket's 108-byte `sun_path` — bind by a RELATIVE name from inside the dir (as
`src/main/ops/tmp_sweep_test.py` does). Hard-coded `/tmp` defaults that remain for bulk data:
`main/ladder_drift_scan.py --cache /tmp/psreplays` and
`/tmp` defaults of the remaining fuzz scripts (scripts, not collected).

### Test tiers — TWO AXES, and keeping them apart is the point

A marker says **what a test NEEDS** (capability). A separate marker says **what it COSTS** (`slow`).
Collapsing those into one axis is what the old single `integration` marker did, and it failed in
**both** directions at once.

| Marker | Answers | Values |
|---|---|---|
| capability | *can this run here?* | *(unmarked)* · `integration` · `sim` · `browser` · `e2e` |
| **cost** | *should this run routinely?* | **`slow`** (the one that deselects) · `static` (a budget tier that deselects NOTHING — below) |

| Tier | Needs | Count (2026-08-23) · duration (2026-08-14) |
|---|---|---|
| *(unmarked)* | nothing — pure in-process | 6570 tests, **127 s** serial (~56 s at `-n 4`) |
| `integration` | an out-of-process dep, no battles, no browser | 158 tests, ~16 s total |
| `sim` | real battles in-process via the bridge, no server | 60 tests, ~100 s total |
| `browser` | headless chrome | 57 tests, **~19 s** (re-measured 2026-09-29, quiet box, 5 runs 18.0–20.3 s; was 1426 s) — NOT `slow` since 2026-09-29: it runs in the ROUTINE gate (a busy box skips a timed-out probe as inconclusive) |
| `e2e` | a live Showdown server | run directly as scripts |
| `slow` | *(orthogonal)* minutes, not seconds | 75 tests |
| `static` | *(orthogonal)* a static gate: ~free warm, a bounded COLD-cache cost — its own 180 s budget, still in every tier | the 13 gates' files (below) |

**Counts are dated on purpose — RECOUNT before quoting one** (`pytest -m <tier> --collect-only -q`);
this corpus moves faster than the doc describing it. Durations were taken on a quiet box and have
not been re-taken, because a duration measured beside a live run is not a measurement.

🚨 **MEASURE BEFORE YOU TIER — the intuitive answer was wrong, TWICE.** In 2026-08 the **browser**
tests were 88% of the whole integration tier (1426 s of 1623 s), and this doc blamed "a fresh
headless chrome at ~25 s of cold start" — UNMEASURED. Measured 2026-09-29, both that and the next
guess (the 20 s `--virtual-time-budget`) were wrong: bare chrome starts in **0.37 s**, the budget
(20,000 → 500 ms → none) moved nothing, and the 25 s was a **D-Bus keyring timeout** paid by any page
touching chrome's network stack — `--password-store=basic` took it to 0.65 s per page. With one
browser per module over the DevTools pipe (`src/utils/headless_chrome.py`) the tier is ~19 s. Tier
by the profile (`pytest -m <tier> --durations=0`), not by which subsystem feels riskiest — and
**time the cause before you name it**.

🚨 **Cost tracks battle COUNT, not "does it battle"**, so `sim` cannot be the marker that decides
routine cost — `slow` is. `gen3_data_obs_parity` is battle-backed and CHEAP, and putting it behind
the old `-m "not integration"` gate is exactly how it rode main RED three separate times.

**A tier is DECLARED, never inferred.** Cost arrives transitively, so neither a filename nor an
import graph can classify a test (30 collected files transitively reach the battle runner, nearly
all millisecond unit tests). The root `conftest.py` enforces the only signal that IS cost: **an
unmarked test that overruns a 30 s budget is reported and told which marker to take**
(`GEN3AI_SKIP_TIER_BUDGET=1` opts out). It **FAILS the run only on a QUIET box (factor < 1.05); on
a busy one it is ADVISORY** — a compile-heavy test slows by multiples of the contention factor
(measured: 12.3 s idle → 65.9 s at load 22, against a 1.2× scaled budget), so a scaled-only guard
would go red whenever a run is live. `tier_budget_guard_test.py` pins both halves and that the
guard may only ever ADD a failure, never clear one.

**The `static` budget tier (deletion pass K1 follow-up, 2026-10-02; owner: "checks that pass or fail
deterministically").** The 30 s unmarked budget is the wrong yardstick for a STATIC gate: the mypy gate
is 0.26 s warm and a worktree's first (COLD-cache) run read **32.6 s** (19.6 s on 2026-08-17; 9 s with
a private `MYPY_CACHE_DIR` on a quiet box), the enum-str gate ~24 s cold — so whether the routine gate
passed depended on whether a cache happened to be warm. A static gate DECLARES `pytestmark =
pytest.mark.static` (per file; the declared list is `tier_budget_guard_test._STATIC_GATES`, which fails
a gate that does not declare it) and is held to **its own base, `conftest._STATIC_BUDGET_BASE_S` =
180 s** (~5x the worst cold read), scaled by contention and enforced on a quiet box exactly like the
30 s one. It is a TIER, not an exemption: a gate that grew to minutes still overruns it, and unlike
`slow` the marker deselects nothing — `-m "not slow and not e2e"` still runs every gate. (The other
fix considered, exempting the gates from the budget outright, would have let one grow unnoticed.)
Over-budget lines print each test's own budget.

**The factor the guard reads is the WINDOWED meter, not the load average** (`src/utils/cpu_meter.py`,
`gen3_contention_meter_v2`, 2026-09-30). The old reading was `load1 / cpus` taken once at session
end, and that day every agent's routine gate went RED on the budget with 0 tests failed: load1 ~9-11
on 16 cpus read **1.00, "quiet"**, while the flagged tests ran 1.3-2.7x their quiet times. Two blind
spots — load1 is a 1-minute EMA sampled at one instant, and this box is **8 cores / 16 hardware
threads**, so past ~8 busy threads cores are SHARED (a CPU-bound thread runs **1.70-1.83x** slower
beside a busy sibling) with no run-queue wait at all, which Linux PSI `cpu some` cannot see either.
The meter diffs three kernel-integrated counters across a window:

| Counter | Gives | Term |
|---|---|---|
| `/proc/stat` per-cpu busy ticks | `U`, mean busy logical cpus | SMT: `g(U) = 1 + 0.75·min(1, 2(U−P)/U)` past `P` cores |
| OUR tasks' `/proc/<pid>/task/<tid>/schedstat` (box-wide `/proc/schedstat` when there is no self) | run-queue stretch `1 + wait/run` (does not saturate) | × the stretch's EXTERNAL share |
| `/proc/pressure/cpu` `some` total | stall share — REPORTED; the run-queue term only when schedstat is missing | `1/(1−some)` fallback |

The run-queue term reads OUR OWN tasks because the box-wide counter is not what a test feels: it read
1.06-1.23 with 5-8 of 16 cpus busy (one other agent's process queueing on itself), and once x11.1 at
PSI 1.7 %. `load1 / cpus` is the last resort (no `/proc/stat`), and every reading names its SOURCE. **The pytest
session's own process tree is subtracted** (and per test, that test's own xdist-worker subtree), so a
test that saturates the box by itself is still caught. The controller samples at session start, on
test-start events (≤ one per 2 s, ~3 ms each), at the end of every test over 30 s, and at session end
— no thread. **A run FAILS only if the SESSION and the flagged test's OWN WINDOW both read < 1.05**:
a session average can hide the burst that sat on the one long test. Each over-budget line prints its
window's reading; the summary prints the session's, with the old load1 reading beside it. A test
over 30 s but inside its contention-SCALED budget is never a verdict, but it is still NAMED in one
`tier budget:` line — on a busy box a slow test must not vanish from the report.

Calibration (2026-09-30, pure-Python probe, quiet 0.396 s, K busy `while 1: pass` workers, 6 s windows,
same window for every column):

| K | probe slowdown | OLD load1/cpus | PSI-only `1/(1−some)` | NEW |
|---|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4 | 1.03 | 1.00 | 1.00 | 1.01 |
| 8 | 1.92 | 1.00 | 1.00 | 1.52 |
| 10 | 1.98 | 1.00 | 1.04 | 1.77 |
| 12 | 2.01 | 1.00 | 1.19 | 2.05 |
| 16 | 2.16 | 1.00 | 1.24 | 2.11 |
| 24 | 3.51 | 1.00 | 2.51 | 2.94 |

PSI-only under-reads the SMT band and over-reads saturation (`1/(1−some)` = 8.5 vs a measured 5.65 at
K = 32). The new meter still under-reads the probe by up to ~25 % in the 8-core band, but it never
calls a slowed window quiet — the direction the enforcement decision needs. `cpu_meter_test.py` pins
the mapping (the incident case is a named test) and has one REAL test (`integration`, ~4 s: 12 busy
workers read contended, and read as SELF when they are ours). ⚠️ No real-process test can assert an
idle box reads quiet on a shared machine; that half is pinned with synthetic counters only.
`GEN3AI_TIMEOUT_SCALE` still forces the factor. `scale_timeout` / `ProgressDeadline` still read the
instantaneous load1 factor (`contention.py`) — **they are not on the new meter**.

### COMPOSITION gates — where a whole RUN is the unit under test

Some properties exist only in the JOINS between subsystems, and no leg-level test can see them: the
legs are exercised with fake sessions, hand-built shards and pre-recorded fixtures, so a defect in
the hand-off between two of them is invisible to every one of them at once. Those get a test that
launches the real entry point as a SUBPROCESS at smoke scale and asserts on the ARTIFACTS the run
leaves behind — never on "it did not crash".

**No composition gate exists right now.** The only one (`search_teacher_composition_test.py`, `gen3_search_teacher_composition_rust_v1`: >= 2 search-teacher cycles on `--use-bridge rust`, end to end) was DELETED with the search teacher (deletion pass L3, 2026-10-02). The rules below are the shape the next one follows; add its row here (gate, the composition it closes, tier and measured cost) when one lands.

**A composition gate is `slow` by DECLARATION, not by measurement.** A live training run normally shares this
box, so a duration recorded beside one is a note for planning, never a bound to assert against.

Three rules every composition gate here follows, each of them a project rule applied to a subprocess:

- **Every output path goes to the test's `tmp_path` — 🚨 NEVER under `models/`.** The run archive is
  not a scratch space; `train_rl_agent.py`'s `--run-dir` takes the temp dir. 🚨 **The archive is SEALED
  for every test** (2026-10-02, root `conftest.py` sets `GEN3AI_RUN_ARCHIVE_SEALED`): runs always land in
  the MAIN checkout's `models/` (`utils.paths.run_archive_dir`), so anything that creates or resolves a run
  dir — the trainer's `--run-name`, the launcher, `--dry-run`, a meter's default `--out` — REFUSES (a
  typed `RunArchiveError`) unless `$GEN3AI_MODELS_DIR` is set. Take the **`run_archive` fixture**
  (`$GEN3AI_MODELS_DIR` → `<tmp_path>/models`, created) and a forgetful test FAILS instead of writing the
  owner's real archive; real-archive READERS (`main_models_dir`) are untouched. To test "launched from a
  worktree", build a throwaway repo + `git worktree add` and point `utils.paths._archive_anchor` at it
  (`src/utils/run_archive_test.py`).
- **The child is bounded by a `ProgressDeadline` on its own log, not by a total-duration cap.**
  Contention stretches duration; only a real wedge stops output. A wedge is reported as
  **INCONCLUSIVE**, in wording distinct from a failed assertion and carrying
  `describe_contention()` — a timeout is never a semantic outcome.
- **A missing precondition FAILS, it never skips** — an unbuilt rust binary fails with the exact
  `cargo build` line, and a run that reaches "Training complete" having launched fewer than two
  cycles fails as NO-CYCLE rather than passing green on a run that did nothing.

**What the first one actually found, and it was not a crash** (history: that gate is deleted, see above). The search-teacher composition gate's
opening run produced no failed assertion; what it produced was a MEASUREMENT nobody had — the
candidate SELECTION ran inline in `_on_step` for **48 s over 9 traces and 350 s over the default
60-trace `scan_limit`** (measured 2026-09-08 over a real run's copied traces; the gate's own
smaller run read ~30 s / ~100 s), on the training step, every cycle, while the callback was documented
everywhere as "non-blocking (subprocess workers)". That is the argument for this shape of gate: a
leg-level test measures a leg's correctness, and only a whole run can tell you what a join COSTS.
Where a composition gate can cheaply record such a cost as a scalar, do —
a number in the run's own events survives the next person's assumptions better than a comment.

### PARITY tests on a policy: never on FRESH weights, and prove the check can fail (`gen3_fresh_parity_probe_v1`)

A **parity test** says two paths agree: compiled == eager, served == reference, diagnostics ON == OFF.
It is only as strong as the variation in what it compares. **A freshly built production policy is a
VACUOUS probe.** The pointer head's scorers are zero-init, so every legal log-prob on a row is
`-log(n_legal)` whatever the extractor did. The other zero-init projections output exactly 0, and on
the fresh policy 52 of 232 extractor parameters get zero gradient from a features-only loss.
**Measured** (M5 T2, 2026-09-29): an AOT miscompile read max|dlogp| **0.0 on fresh weights and
0.68 on real ones**. The rules:

1. **Perturb a fresh policy before a parity check.** Use `agents.model.parity_probe`:
   `perturb_(module)` or `perturbed_copy(module)` in a test, or `perturbed_parameters(module)` in a
   gate that must judge its own installed graph (restored bit-exactly, private RNG). The helpers
   already in use are `agents.inference.service.fixtures.perturbed_fresh_policy` and
   `extractor_compiles_test._build_production_extractor()`, which is perturbed by default
   (`fresh=True` opts out).
2. **Feed REAL rows, never zeros.** Use the committed `compile_parity_obs.npz` via
   `compile_parity_fixture.load_parity_rows`. An all-zero obs has no valid move seat. In
   `diagnostics_cadence_test`'s bit-identity check it left **106 of 254** parameters unmoved by the
   update, so "a diagnostic changed parameter k" could not fail for them. Real rows plus the
   perturbation leave 19.
3. **Guard the compared quantity, fail-closed.** `parity_probe.require_informative` raises
   `VacuousParityError` when a quantity's spread is not above the check's own bar.
   `compile_trainer.decision_verdicts` / `train_verdict` (and so the inference service's `judge`)
   apply it by default. `allow_vacuous=True` is reserved for a caller that has already judged the
   same graph on perturbed weights.
4. **Prove the test fails on revert**, as for any edge case. (`parity_probe_test` carried a
   pointer-only miscompile against the extractor-only gate — exact features and gradient, corrupt
   move cells only the pointer head reads — until that gate's deletion, K1 2026-10-02; the region
   gate's fresh-weights pass is pinned by `compile_regions_test`.)
5. **A GRADIENT parity check must reach every path, and be judged per parameter**
   (`gen3_gate_grad_coverage_v1`). A loss over the extractor's two feature outputs never reaches the
   stash that the pointer and aux heads read. On the perturbed production policy it left **65 of 254**
   policy parameters with zero gradient, so a backward-only miscompile there passed on any weights.
   The cure is a loss that IS the production loss: region R1's gate differentiates the micro-step's
   own loss over every policy parameter. (The extractor-only gate's probe loss, `gate_loss`, and its
   coverage guard were deleted with that gate, K1 2026-10-02.) A single GLOBAL cosine is not enough either: it is dominated by the largest gradients. Dropping
   the whole move-cell gradient left it at 1.0000, while the per-parameter relative error read 0.10.
   `compile_trainer`'s per-parameter rule is the one that bites.
   `compile_gate_probe_test` pins the per-parameter rule.

### Test file naming conventions

| Pattern | Requires | Marker |
|---|---|---|
| `*_test.py` | Nothing — pure unit tests with mocks | — |
| `*_integration_test.py` | An out-of-process dependency, no live server. **The name is historical and no longer implies the tier** — these split across `integration` (light), `sim` (bridge battles) and `browser` (headless chrome, ~19 s, routine since 2026-09-29). Read the file's `pytestmark`, not its name | `integration` and/or `sim` / `browser` / `slow` |
| `*_fuzz_test.py` | `deps/pokemon-showdown` — runs **real battles in-process via the local BattleStream bridge** (`utils/bridge/local_battle_runner.py`); **no live server**. The default for fuzzing. | none — run directly as scripts (no `test_*` funcs, so `pytest` imports but collects nothing) |
| `*_fuzz_e2e_test.py` | A **live Showdown server** — fuzz whose checks need real async-server timing (e.g. `effectiveness_fuzz_e2e_test`, whose TurnDelta-vs-BattleContext effectiveness window is decision-timing-sensitive) | run directly as scripts |
| `*_e2e_test.py` | A **live Showdown server** on localhost:8000 | `@pytest.mark.e2e` (scripts only, run directly) |
| `*_benchmark.py` | `deps/pokemon-showdown` bridge (no live server) — **performance profiling, not pass/fail**: plays a real battle in-process, then `cProfile`s a hot path | none — run directly as scripts (no `test_*` funcs → `pytest` collects nothing). Place in a dir with no stdlib-shadowing names (e.g. `training/`, not `observation/`) |

### Which command to run

| When | Command | Count (2026-08-23) · duration (2026-08-14) |
|---|---|---|
| **inner loop** — you want the fastest true/false | `-m "not slow and not e2e and not sim and not integration"` | 6570 tests, **127 s** (~56 s at `-n 4`) |
| **THE ROUTINE GATE** — before a commit | `-m "not slow and not e2e"` | 12,782 tests, **4 m 19 s at `-n 6`** on a quiet box (2026-10-01; table below) |
| **before a `/gen3ai-ship`** | the routine gate, OR a TARGETED set you choose (owner, 2026-10-05: "I prefer more targeted test suites") — see "Targeted sets before a ship" below | — |
| **in CI, or on demand** | `pytest src/` (everything) | 11,578 tests, **~47 m** serial (2026-09-29, nice 19, load ~3; the browser tier is ~19 s of it — the rest is corpus growth since 2026-08) |
| just the bridge | `-m sim` | ~100 s |
| just the browser views | `-m browser` | **~19 s** (2026-09-29) |
| **anything on the GPU** (a `GEN3AI_TEST_ALLOW_GPU=1` test, a cuda benchmark) | **Only under a GPU LEASE** the orchestrator granted you in your brief (owner, 2026-10-03: nobody blocks on the GPU). `scripts/ops/gpu_lease.sh acquire --owner <name>` (immediate typed refusal if leased or busy) prints `export GEN3AI_GPU_LEASE_TOKEN=<token>`; an agent's Bash tool keeps no exports between calls, so put `GEN3AI_GPU_LEASE_TOKEN=<token>` (or `--token-file F` once + `GEN3AI_GPU_LEASE_TOKEN_FILE=F`) in front of each call; `release` when done. Then `scripts/ops/gpu_lock.sh <cmd>` — **never a bare `flock ~/.claude/jobs/gpu.lock`** — passes straight through on your token and runs WITHOUT queueing. Everyone else is refused AT ONCE: exit **6** `GpuLeased` (names the owner, holder pid, since when), exit **5** `GpuBusy` (a one-off holder), never a wait; `--wait` / `wait=True` (kernel-blocked, a lease appearing ends it) is for the orchestrator or a training launch. **An agent with no lease never sets `GEN3AI_TEST_ALLOW_GPU=1`** (GPU tests skip without it, so the routine gate stays CPU-only). The helper (`src/utils/gpu_lock.py`, `src/utils/gpu_lease.py`) also exports `GEN3AI_GPU_LOCK_HELD=<pid>`, so a command that takes the lock itself (`policy_spectrum truth --lock`) re-enters instead of deadlocking on its own ancestor (2026-09-30: 15 min at 0% CPU); a bare-`flock` ancestor raises `GpuLockSelfDeadlock` at once. **A wall timeout goes INSIDE the lock** — `scripts/ops/gpu_lock.sh timeout 3000 <cmd>`; a stale lease (dead holder, reused pid) is detected and cleared by `status` / `acquire`; a crashed owner's lease is ended by the orchestrator (`release --force`), or expires at `--max-hours` (default 12) / when `--watch-pid` dies. Tests: `src/utils/gpu_lease_test.py`, `src/utils/gpu_lock_test.py` | — |
| **any exploratory or one-off HEAVY job** (a trace, a benchmark, a measurement driver, anything that loads many checkpoints) | `scripts/ops/mem_cap.sh <GB> <cmd>` (Python: `utils.mem_cap`), a timeout INSIDE (`mem_cap.sh 24 timeout 2h python …`), composes with `gpu_lock.sh` in either order. The job gets its OWN scope (`MemoryMax=<GB>G`, swap 0, `OOMPolicy=stop`) inside `gen3ai-heavy.slice` (aggregate cap, default 64 GB of 89) at `oom_score_adj` +500, so an overrun kills only that job (exit 86). 2026-09-30: three global OOM kills of a 74-82 GB python3 each tore down the whole tmux scope (Claude + every agent) | — |

```bash
# THE ROUTINE GATE — everything cheap, whatever it needs; -n 6 (-n 4 beside a live training run),
# under THE gate slot (one gate at a time).
export PYTHONPATH=$PYTHONPATH:src && scripts/ops/gate_lock.sh \
  /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m pytest src/ -m "not slow and not e2e" -q -n 6
```

**The gate SEMAPHORE** (`scripts/ops/gate_lock.sh`, `src/utils/gate_lock.py`, 2026-09-30): at most
`GATE_SLOTS` = **1** routine gate at a time since 2026-10-01 (was 2; `$GEN3AI_GATE_SLOTS` overrides) —
the measurement is in "Routine-gate wall time" below. Seven concurrent `-n 2` gates
plus a compile inventory and a GPU benchmark put load ~36 on 8 cores / 16 threads; every gate crawled
(one COMMIT test ran 160.8 s against ~21 s quiet), budgets tripped, benchmarks were contaminated. N
`flock` slot files under `~/.claude/jobs/gate_slots/`: try each non-blocking, else block on one in the
kernel while re-trying the rest; a crashed holder's slot frees when it dies. Re-entrant like the GPU
lock (`GEN3AI_GATE_LOCK_HELD=<pid>:<slot>`, verified against `/proc/locks`); all slots held by
ancestors is `GateLockSelfDeadlock` (exit 3); the timeout is inside (`--timeout-s`, exit 4). Wait and
acquisition time go to stderr; `--status` names the holders. When a `train_rl_agent.py --run-dir`
process is live, every take prints ONE line naming the run and recommending `-n 4`
(`gate_lock.live_run_warning`, from `utils.procfs.live_training_runs` — the rule
the retired cutover governor read); it never changes the command.

### Routine-gate wall time and the `-n` × gate_lock policy (measured 2026-09-30 / 10-01)

Every row is the FULL routine gate (`-m "not slow and not e2e"`) under `gate_lock.sh` + `mem_cap.sh`,
started only when the windowed meter read the box quiet (factor ≤ 1.05, ≤ 5 of 16 logical cpus busy
for 30 s, both gate slots free); the "session" column is that run's own window, its own process tree
subtracted from "others" but not from the SMT term (8 workers + their children on 8 cores read high by
themselves). Raw rows: `~/.cache/gen3ai/test_speed/runs/` (scratch, not committed).

| tree | `-n` | schedule | tests | wall | Σ test time | CPU user | session meter |
|---|---|---|---|---|---|---|---|
| `68a2f85b` (before) | 2 | xdist `load` | 12,580 | **626 s** | 1,211 s | — | 1.02 |
| `68a2f85b` (before) | 8 | xdist `load` | 12,580 | **438 s** (one worker busy 428 s, seven ~175 s) | 1,643 s | 1,918 s | 1.08 |
| `aff426f0` (scheduler only) | 8 | `load` / cost-ordered `loadfile` | 12,699 | 362 s / 320 s | 1,796 / 2,100 s | — | 1.30 / 1.61 |
| `940ce487` (after) | 2 | cost-ordered `loadfile` | 12,782 | 660 s | 1,284 s | — | 1.02 |
| `940ce487` | 4 | 〃 | 12,782 | 352 s | 1,355 s | 1,643 s | 1.01 |
| `940ce487` | **6** | 〃 | 12,782 | **259 s** | 1,477 s | 1,842 s | 1.15 |
| `940ce487` | 8 | 〃 | 12,782 | 241 s (all 8 workers end at +234 s) | 1,819 s | 2,208 s | 1.20 |
| `940ce487`, TWO gates at once | 4 + 4 | 〃 | 12,782 each | **691 s and 694 s each** | — | 3,613 / 3,653 s each | 1.74 |

**Policy: one gate at a time (`GATE_SLOTS` = 1) at `-n 6`.** Concurrency LOSES here: two `-n 4` gates
at once took 11.5 min each and burned 2.2× their solo CPU, so the second of two queued gates finishes
at ~8.6 min under one-at-a-time against 11.5 min side by side. `-n 6` is 93 % of `-n 8`'s speed for
83 % of its CPU and leaves ~2 cores for everything else; past it the gate is SMT-bound (Σ test time
rises 1,355 → 1,819 s from `-n 4` to `-n 8`). ⚠️ The numbers are for a box with NO training run;
with one live, use `-n 4` (gate_lock prints the warning) and re-measure.

**Where the time goes** (the `-n 2` profile, 2026-09-30): the top 1 % of tests (125) hold 68 % of
the test time, the top 5 % hold 93 %, the top 20 % hold 99 %; collection is ~4.5 s per worker and
session teardown ~5-10 s, so the gate is test bodies. An attribution run over the 45 heaviest files
put **45 %** of their time in waits on CHILD processes (rust binaries, node bridges, python
eval/replay children), 6 % in production-size policy construction and 4 % in fixed sleeps; the
heavy hitters are the M5 parity gates (`rust_eval/parity*`,
`rust_core_parity`, `bots_gate`), the `cf_producer`/`cf_audit` integration paths, the anchors smoke,
`extractor_compiles` (compile) and the two mypy-backed static gates on a COLD cache (~45 s on a fresh
worktree's first gate; since 2026-10-02 they sit in the `static` budget tier above, so that cost is
declared, not a flake). The fixes that shipped with this table: `3e766294` (a VACUOUS 30 s watchdog
test), `8dcc32ce` (quota_match −20 s), `aff426f0` (the schedule), `2dcf59c3` (the compile pool),
`a49ddee7` (the anchors smoke), `940ce487` (the Lane H file split). What was profiled and NOT fixed is
in `TECH_DEBT_BACKLOG.md` §2(b).

**The xdist SCHEDULE — whole FILES, the most expensive first** (`gen3_cost_ordered_loadfile_v1`,
`src/utils/xdist_schedule.py`, 2026-09-30). `-n N` with no `--dist` of your own now means `--dist
loadfile` (the root conftest's `pytest_cmdline_main`, before xdist would make it `load` — NOT an
`addopts`, which is a usage error under `-p no:xdist`), and its `pytest_xdist_make_scheduler` hands
xdist a `LoadFileScheduling` whose work queue is ordered, once,
by each file's summed per-test durations, so the long poles start first and every worker pulls the
next file when it runs dry (LPT). Why: under xdist's default `--dist load` the routine gate at `-n 8`
took **437 s wall for 1,782 s of test time** — `load` hands each worker a CONTIGUOUS quarter-share
of the collection up front (~390 tests at `-n 8`) and never takes it back, and the `agents/training/rust_*`
parity gates (~400 s) sat in one worker's block: that worker was busy 426 s, the other seven idled after
~200 s. A FILE is the unit so module- and class-scoped fixtures (the Lane H gate's three ~40-70 s eval
runs) are built once — a file runs on one worker, back to back, exactly as it does serially.

The durations live in a per-USER table, `~/.cache/gen3ai/test_costs.json` (`$GEN3AI_TEST_COSTS`),
keyed by repo-relative node id so every worktree shares it; every session of THIS repo merges what it
measured at session end (flock + atomic replace; 60-day prune; a conftest copied into a test's temp
dir records nothing). 🚨 **The table ORDERS work and does nothing else**: the collection is untouched,
so a missing, stale or corrupt table runs the identical set of tests and costs only wall time — an
unreadable one is IGNORED and named in the session summary. `xdist_schedule_test.py` runs a real
`-n 2` child session per table state (good / missing / corrupt / stale) and asserts all seven planted
tests ran, and that the costly file starts first with the schedule on and cannot with it off.
`GEN3AI_COST_SCHEDULE=0` is plain `loadfile`; an explicit `--dist <mode>` always wins (a test whose
child session needs each test on its own worker passes `--dist load`, as `pytest_tmp_on_disk_test`
does). ⚠️ **Upstream `loadfile`/`loadscope` RE-RUN the test that crashed its worker** (the crashed unit is
re-queued with that test still pending, so a restarted worker runs it again until the restart budget
is spent — found when `slow_tier_status_interrupt_test`'s SIGTERM'd worker re-ran a 600 s sleeper);
the schedule overrides `remove_node` to report it once, as `load` does, and re-queue only the rest of
its file. Pinned by `xdist_schedule_test`'s planted SIGKILL, which fails on the upstream method.

**Do not use the old `-m "not integration and not e2e"`.** It is what let the obs-golden linchpin
rot on main three times: `integration` now spans a ~100x cost range, so excluding it throws away
cheap, high-value coverage (bridge battles, data parity, mechanics) to avoid the browser suite. Cut
on **`slow`** instead — that is the marker that means "expensive".

### The two STATIC gates (mypy + ruff) — default-on, in every tier

Static checking is enforced by **tests**, not by habit, because there is no CI on this box: the
routine suite is the only thing that runs on every change, so a check outside it is advisory and
rots. Both carry `static` (the budget tier above — they run even in the fast inner loop) and both are
~free warm:

| Gate | Runs | Scope | Measured |
|---|---|---|---|
| `src/agents/model/mypy_gate_test.py` | `python -m mypy` (**no path argument** — the scope comes from `mypy.ini`) | `src/agents/model` **+ `src/agents/observation`**, per `mypy.ini`'s `files =` | **0.28 s warm**, 19.6 s cold (32.6 s on a loaded box, 2026-10-02) |
| `src/ruff_gate_test.py` | `ruff check src/agents src/main src/utils --select F,E9 --exclude src/poke_env --exclude src/rust_sim` | `agents/` + `main/` + `utils/` | **0.10 s** |

They are complementary, not overlapping: mypy is deep over a **declared short list** of packages
(`mypy.ini` sets `files = src/agents/model, src/agents/observation` with `follow_imports = silent`,
so the rest of the tree is read for types but not reported), while ruff is shallow over everything.
**Widening is a `mypy.ini` edit and the test follows it** — the gate invokes bare `python -m mypy`
with no path argument precisely so the config is the only scope declaration, and its
`_CHECKED_PACKAGES` assertion fails if `files =` is ever shrunk by accident. `--select F,E9` is
pyflakes + syntax errors only — findings that mean the code is **wrong**, never a style opinion, so
the gate cannot degrade into a formatting argument.

**A missing tool FAILS, it does not skip** (both are pinned in `environment_torch28.yml`) — a linter that
silently opts out reads exactly like a linter that found nothing. Opt out explicitly with
`GEN3AI_SKIP_MYPY_GATE=1` / `GEN3AI_SKIP_RUFF_GATE=1`.

**Known ruff findings are per-file entries in `ruff.toml`, never a blanket exclude**, and that file
keeps two categories apart: a PERMANENT one (the model package's declared re-export hubs, which
import names solely so other modules can import them back out) and a TEMPORARY handoff list of
ordinary dead code. The second was meant to shrink to nothing and **has** — it is CLOSED and holds
zero entries, so a new finding has nowhere to be parked: fix it at the source, or give it an inline
`# noqa: <code>` naming the reason. `features_extractor.py`'s file-wide `F401` is the only surviving
entry and it is PERMANENT (measured 2026-08-23: **73** findings without it — up from 33, because the
2026-08-23 class split turned the hub into pure re-export, so nearly every import there is now a name
another module reaches back out through). It is keyed to the HUB alone; the five siblings the class
split into inherit nothing and need nothing. `/gen3ai-ship` runs both gates before
staging (step 1c), so the ship path does not depend on whether the suite was run.

### The FILE-SIZE ratchet (`src/file_size_gate_test.py`) — the third static gate

**Under 1,000 lines is the TARGET for a core source file; under 2,000 is the STRONG bound.** The
gate encodes that asymmetry rather than flattening it: over 2,000 **hard-fails**, while the
1,000–2,000 band fails nobody and is **reported** (run with `-s` for the census — the pool the next
decomposition should come from). A target that fails the build is a bound; a target nothing ever
prints is a wish. Same scope as ruff, unmarked, **0.04 s**, `GEN3AI_SKIP_SIZE_GATE=1` opts out.

**Test files are EXEMPT when they exercise a single subject** — a fuzz test, or an `X_test.py` with
a source sibling named `X`. A long test that pins one module is that module's specification and
splitting it scatters the spec; a test that sprawls across six subsystems takes the same 2,000
bound. The criterion is the **NAME**, deliberately, because an import graph classifies almost every
test as cross-cutting. When it misfires, rename or split the test — do not park it in the allowlist.

🎉 **THE ALLOWLIST IS EMPTY — every source file in the tree is under the 2,000-line bound**
(2026-08-23). That is a MEASUREMENT, not a claim: a meta-test walks the real tree and fails if the
list and the set of oversized files disagree in *either* direction.

🚨 **A new entry is not a legal move, and there is nowhere to park an oversized file: decompose it.**
A decomposition takes one of two shapes — a package (`main/train/`, `prober/engine/`,
`instrumented_ppo/`), or a base-class CHAIN behind the same hub where a re-export hub already exists
(`features_extractor.py`, which keeps every `state_dict` key and every `inspect.signature` reader
byte-identical). A listed file may shrink freely but **fails if it grows ≥10%** past its recorded
count, and **fails when it drops back under 2,000 without being removed** — the list may only
shrink, because a stale entry misleads every reader after it.

**Keeping it empty is always-welcome piecemeal work** — no design doc, no coordination; the
1,000–2,000 census is where the next one comes from. The five original entries and how each was
paid off: `designs/research_state/claude_md_archive/file_size_paydown.md`.
### The CLAUDE.md FRESHNESS gate (`src/claude_md_freshness_gate_test.py`) — the fourth static gate

**Prose has no compiler, and a `CLAUDE.md` is loaded into every session and read as fact.** A path
that moved and a flag that was deleted do not merely go stale — they are *actively believed*, and
the believer then reports a false result or types a command that cannot launch. So the same two
things a scanner CAN check are checked, over every `CLAUDE.md` in the tree: **every repo-relative
path must exist**, and **every `--flag` must resolve in some parser in this tree**. Unmarked,
**0.84 s**, opt out with `GEN3AI_SKIP_CLAUDE_MD_GATE=1`.

The flag surface is an AST scan of `add_argument` literals (plus hand-rolled `sys.argv` flags)
across `src/`, `tools/`, the Rust binaries, the Node harness and `scripts/*.sh` — **not**
`build_parser()`, because importing the training parser alone costs 1.15 s across ~30 parsers and
because several entry points live in `__main__.py`, where an import once *started a training run*
(see `src/main/entry_point_guard_test.py`). It was validated against the live parser: of 588 live
options, every one the scan missed was an auto-generated `--no-` negation.

**A flag or path named deliberately as HISTORY goes in `designs/deleted_flags.md`** — DELETED,
DEMOTED off the CLI, or PROPOSED-but-unbuilt, each with the version, signature or date that made it
so. That list is a ratchet, not a bin: an uncited row fails, and a row whose flag returns to the
live surface fails too. External tools' flags (`pytest`, `ruff`, `pip`, `cargo`, `git`, chrome) live
in the test's own allowlist, **each named with its tool**, so it cannot quietly absorb one of ours.
The census behind it — what each big `CLAUDE.md` is made of and what size is right — is
`designs/research_state/claude_md_census_2026-09-06.md`.

### The SLOW-TIER LAST-KNOWN-STATUS gate (`src/slow_tier_status_gate_test.py`)

**The routine gate is `-m "not slow and not e2e"`, so a `slow` test is DESELECTED — and a
deselected test cannot fail.** A red one is therefore invisible until the next 31-minute full-suite
run, which happens at most once before a ship. Measured, not imagined: on **2026-09-07**
`tb_relevance_test::test_a_winprob_run_emits_no_noise_tag_and_every_live_tag` was red on main for a
day behind that marker, unrecorded in both the ledger and the backlog. It is the same shape as the
obs-golden linchpin riding main RED three separate times behind the old `-m "not integration"` cut —
one marker further out. **A gate that reads GREEN while a test is red is the GIGO class itself.**

So the tier that RUNS the slow tests writes its verdict, and the gate that runs on every commit
reads it:

* **Writer** — the root `conftest.py`. Every phase report of every `slow` test that actually RAN is
  folded into one verdict per test id, and merged into the artifact at session finish. It **merges,
  never replaces**: running one slow test does not erase the other rows, and two `xdist` workers
  finishing together cannot lose each other's results (an exclusive `flock` plus an atomic rename).
  A run in which no slow test executed writes **nothing** — the routine gate never touches the file.
  Each verdict is **banked at that test's teardown**, not only at session finish: the slow tier is
  ~2 hours beside a live run, and a session interrupted at test 79 of 80 must not throw away 79
  verdicts. An artifact you only get by not pressing Ctrl-C is an artifact nobody will have.
  🚨 **A PASS needs a CALL phase** (`slow_tier_status.settle`): a setup that succeeds classifies
  `pass`, so a test interrupted mid-call once banked as PASS at 0.0 s through the session-finish
  sweep (2026-09-24, a MILESTONE run stopped with SIGTERM). A test with no call report now banks
  `inconclusive`; `src/slow_tier_status_interrupt_test.py` interrupts a real session to pin it.
  🚨 **A SKIP never replaces a measurement** (`slow_tier_status.merge_row`): a skipped test did not
  run, so over a banked `pass` / `fail` / `inconclusive` the old row is KEPT — verdict, commit and
  `at` untouched, so staleness still ages the last real run — and gains `last_skipped_at` /
  `last_skipped_commit` / `last_skip_detail`. On 2026-09-29 a slow-tier run without
  `GEN3AI_TEST_ALLOW_GPU` overwrote seven banked GPU PASS rows with SKIP; over a FAIL the same
  replacement would have hidden a red. A skip banks as a row's status only where nothing measured
  is on file.
* **Artifact** — `designs/ops/slow_tier_status.json`, **committed**.
* **Reader** — `src/slow_tier_status_gate_test.py`, unmarked, in every tier, ~0.03 s plus one
  `git rev-list` per distinct recorded commit.

| class | meaning | gate |
|---|---|---|
| `fail` | the tier ran it and it failed | **FAILS the routine gate**, naming the test id and the commit it failed at |
| `inconclusive` | it failed with a TIMEOUT signature, **or it never finished a CALL phase** (Ctrl-C, a SIGTERM'd xdist worker — killed in flight) | reported — *a timeout is never a semantic outcome* |
| unrecorded | collected as `slow` this session, no row | reported — a new slow test is not a regression |
| stale | the row is >25 commits behind HEAD, or its commit is unknown | reported |

The three non-fatal classes are emitted as **warnings**, so they survive `-q` in the warnings
summary, plus a one-line census on `-s`.

**WHY ONLY A RECORDED RED IS FATAL.** Staleness is a statement about the *schedule*, not about the
code. Failing on it would make "nobody has run the tier for 25 commits" indistinguishable from "a
test is red" — precisely the conflation this gate exists to remove, and the same rule as *a TIMEOUT
is never a semantic outcome*. It would also breed a reflex `GEN3AI_SKIP_SLOW_STATUS_GATE=1`, which
costs the real signal too.

⚠️ **THE HONEST LIMITATION, stated rather than hidden: a slow test that broke SINCE the last
recorded run still reads `pass` here.** Nothing short of running the tier closes that. The staleness
report is what says how much a green is worth, and the full refresh is one command:

```bash
export PYTHONPATH=$PYTHONPATH:src && "${GEN3AI_PYTHON:-/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3}" -m pytest src/ -m slow -q -n 2
```

**WHY THE ARTIFACT IS COMMITTED and not gitignored.** The worktree workflow decides it: a gitignored
file is per-worktree, and a fresh worktree — where essentially all work here happens — would have
none, so the gate would be dead exactly where it is needed. That is the `models/` failure mode
(`utils.paths.main_models_dir()` returns `None` in a worktree and every caller must turn it into a
skip); a gate that skips in every worktree is not a gate. Committed, it also puts a red slow test in
the **diff**, a second human-readable channel that costs nothing, and it keeps the pre-ship
full-suite run and the routine gate that follows it talking about the same measurements. The price
is a merge conflict when two branches both run the tier — the rows are keyed by test id, so the
resolution is always a **union** (keep both sides; where both edited one row, keep the newer `at`).

**WHY A MISSING FILE FAILS RATHER THAN WARNS.** The house rule for a gate is that a missing artifact
FAILS and never skips. Because the file is committed, "missing" cannot mean "fresh checkout" — it
means a bad rebase or a stray delete, which is worth stopping for. The message names both the
restore command and `GEN3AI_SKIP_SLOW_STATUS_GATE=1`, so it can never strand anyone. A file that is
*present but empty* fails the same way, with **NO SLOW-TIER STATUS RECORDED**: a gate reading zero
rows is reading nothing at all.

The recording side can be turned off on its own with `GEN3AI_SKIP_SLOW_STATUS_RECORD=1` (a run whose
verdict should not be banked — a deliberate experiment, a starved box). Thirteen meta-tests in the gate
file plant each condition — a red, an inconclusive, an unrecorded, a stale row, an unknown commit,
a merge that must not truncate, a setup-only pass that must not bank green, a skip that must not
replace a banked verdict — so the gate's
behaviour is pinned rather than described. Contract: `src/utils/slow_tier_status.py`.

### A FRESH compile cache per test process (root `conftest.py`; K3, `gen3_hermetic_compile_cache_v1`)

Every pytest process — the controller and each xdist worker — compiles into its OWN fresh dir
(`gen3ai_pytest_compile_<pid>_*`: `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`,
`GEN3AI_COMPILE_CACHE_DIR`) on the REAL DISK — `$GEN3AI_SCRATCH`, else `~/.cache/gen3ai/tmp`, never tmpfs
`/tmp`, resolved by the ONE helper `src/utils/scratch.py` that the pytest temp root uses too —
declared in `pytest_configure` and deleted at `pytest_unconfigure`; subprocesses a test spawns inherit
it. An inherited value is OVERWRITTEN on purpose: no test reads an artifact another run, pin or torch
env wrote (the 2026-09-29 K1b fault was a donated-buffer backward served from the box-wide cache).
Benchmarks and every other process that compiles without a declared cache get a fresh PRIVATE one,
deleted by its creator at exit (`agents.model.compile_cache.ensure_hermetic_cache`); to measure a WARM
start, export the variables yourself. Every such dir carries its creator's PID and is SWEPT by the next
declaration once that PID is dead (a SIGKILLed session leaks only until then — `/tmp` is tmpfs, and
on 2026-09-30 it ran out of INODES). Consequence: a session's first compile of each graph is COLD, so
the production extractor's CPU codegen tests are `slow`, and a compile-path change runs the compile
files' `slow` tier under torch 2.8 (`gen3ai_torch28`, the default) before shipping — the
both-torches rule was RETIRED 2026-09-30 (`designs/training/compile_flags.md`).
`compile_cache_test.py` pins all of it; the full rule is `designs/training/compile_flags.md`
"The HERMETIC per-run compile cache".

### Inductor's COMPILE-WORKER POOL is shut down after each module (root `conftest.py`, `gen3_test_compile_pool_teardown_v1`)

One CPU Inductor compile starts `torch._inductor.async_compile`'s pool — a `compile_worker
--workers=16` child plus 16 forked workers (17 processes, **0.34 GB PSS**, 4.46 GB summed RSS; measured
2026-09-30, torch 2.5.1, `compile_threads` = this box's 16 logical cpus) — and it lived for the rest of
the process. Under xdist every worker that ever compiled kept one for the whole session; it was the
root cause of the BIG-RSS gate failures (`ed5945a9` made those tests immune, not the pool gone). The
root conftest's `pytest_runtest_teardown` now calls `shutdown_compile_workers()` after the LAST test of
each module in the process (0.37 s; the next compiling module restarts the pool, it does not
recompile). `compile_threads` itself is untouched — the compile path tested is production's.
`src/utils/compile_pool_teardown_test.py` runs the real conftest over a compiling module and a
checking one, and the check FAILS with `GEN3AI_KEEP_COMPILE_POOL=1` (the escape hatch). ⚠️ Read child
command lines from `/proc/<pid>/cmdline`, not `ps -o args`: off a tty ps cuts `args` at 80 columns,
which drops `compile_worker` — the first version of that check read no pool where there was one.

### The TORCH GLOBAL-STATE guard (root `conftest.py` + `src/utils/torch_state_guard.py`)

`gen3_torch_state_guard_v1`. **A test that leaves process-global torch/numeric state changed FAILS**,
naming the global, its before → after values and the test. Why it exists (Lane E, `c256dd95`,
F-LJ-6): a harness called `torch.set_num_threads(4)` and never restored it. The NEXT test in the
process then built fresh policy weights at a different intra-op thread count. 94 of 721 tensors moved
by up to 7.5e-6, an exact-tie argmax flipped, and a compiled-vs-eager parity test failed only when
it ran AFTER other tests. A leaked global makes test ORDER an input. Under xdist it also makes the
worker's scheduling an input, so the same commit reads green or red depending on how the items were
distributed.

**The declared list is `DECLARED` in `src/utils/torch_state_guard.py`, and nowhere else.** It holds
these globals:
- the intra-op and inter-op thread counts, fp32 matmul precision, default dtype and default device
- deterministic-algorithms mode and its warn-only flag
- grad, inference and anomaly mode
- cuda-matmul TF32 and reduced-precision reductions; cudnn TF32, benchmark, deterministic and
  enabled; mkldnn enabled
- every key of the dynamo, inductor, functorch and `torch.compiler` configs
- the compiler stance, the dynamo compile-callback counts, and the `compile_control` singleton
  (installed, locked)
- `numpy.geterr()`

**The library half for FRESH WEIGHTS:** `single_thread_build()` (same module; `torch_globals(num_threads=1)`) is THE context manager every site that creates and initialises new parameters builds inside — SB3's orthogonal re-init is a thread-count-dependent LAPACK QR, so the same seed gives different bytes at a different count (`gen3_single_thread_init_v1`, `designs/training/learner_gates.md`). Never write a second copy.

A row is read only when its module is already imported, so the guard never imports a subsystem to
inspect it. The exception is `torch`, which it imports once per process so the core rows always
have a "before". A module first imported INSIDE a window is compared against its import-time
default. That default was measured equal to the live values right after import on 2.5.1 and 2.8.0,
and `torch_state_guard_test` re-checks it.

**Three windows, because a leak can be made in three places:**

| window | where | catches | reported as |
|---|---|---|---|
| TEST | the autouse fixture `_torch_global_state_guard` | the body plus its function-scoped fixtures | ERROR at teardown of that test |
| MODULE | `pytest_runtest_setup` / `pytest_runtest_teardown` wrappers, from before a module's first test to after its last test's teardown | a `scope="module"` or class-scoped fixture that never restores. These set up BEFORE any function fixture, so the TEST window cannot see them | ERROR at teardown of the module's last test, naming the module |
| COLLECTION | a `pytest_make_collect_report` wrapper around every collector | a module-level statement, or a `conftest.py` load. Both happen before any test runs | a collection error naming the file |

**The guard NEVER restores the value, and it has NO allowlist.** Restoring would hide the leak, and
an allowlist is where leaks go to live (the file-size gate's allowlist is empty on the same
principle). Fix the leak at its source with one of these idioms:
- `try/finally`
- `utils.torch_state_guard.torch_globals(num_threads=…, float32_matmul_precision=…, allow_tf32=…)`,
  which restores all four globals. A module-scoped pin puts it inside a yield fixture.
- the root conftest's `restore_torch_globals` fixture, for a body that sets them itself (a CLI
  `main()` run in-process)
- `@utils.torch_state_guard.restores_torch_globals` on a LIBRARY function that sets them for its own
  work (`untaught_meter.play_cells`, `harvest.score_candidates`, `human_agreement.run`), so every
  caller gets its own back — not on a function returning a lazy iterator. A loader whose settings
  must outlive it sets NOTHING and gives its callers a scope instead
  (`policy_spectrum.reader.inference_globals`). Pinned by `src/main/torch_globals_restore_test.py`.
- `torch._dynamo.config.patch(...)`
- an autouse fixture that uninstalls what the test installed. `compile_trainer_test` resets the
  `CompileControl` singleton only when the test itself created it, so an earlier leak is never
  masked.

`GEN3AI_SKIP_TORCH_STATE_GUARD=1` switches the whole guard off. When `utils.torch_state_guard`
cannot be imported, the session is REFUSED rather than run unguarded. A subprocess that copies the
conftest therefore needs an ABSOLUTE `src` on `PYTHONPATH` (see `deps_guard_test`).

**Cost.** A snapshot takes about 2 µs with only torch imported, and about 40 µs on 2.5.1 (about
72 µs on 2.8.0) with every config module imported. Wall time was measured over 3,000 trivial tests
under the real conftest, 2026-09-30, beside a running routine gate: **about 0.15 ms per test** with
torch alone and **about 0.12 ms** with the configs imported. Most of that is pytest's cost for one
more generator fixture, not the snapshot. Over the routine gate's ~12k tests it adds about 1.5 s
of serial time.

⚠️ **What it cannot see.**
1. **The global RNGs.** Every test consumes them, so a before/after diff has no meaning. A test that
   relies on an earlier test's seed is the READER's bug: seed your own generator.
2. **A "leak" that writes the value already there.** The conftest pins `OMP_NUM_THREADS=1`, so
   torch's default is 1 thread and a stray `set_num_threads(1)` changes nothing in the suite. It
   becomes a leak only under `GEN3AI_TEST_ALLOW_THREADS=1`. **Run the routine gate under that
   variable after touching thread handling**: it is how the masked sites surface. For example:
   `GEN3AI_TEST_ALLOW_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 pytest src/ -m "not slow and not
   e2e" -n 2`. Its first run (2026-09-30) found two things the 1-thread default had hidden:
   - one more `cf_audit.main` leak
   - `forward_guard_test`, which compared a control forward taken at the default thread count
     against forwards run at 1 thread, and so passed only while the default was 1
3. **An in-place mutation nested more than one container deep** in a config value.

Revert-proof: `src/utils/torch_state_guard_test.py` copies the root conftest verbatim beside planted
leaks (a body, a module fixture, an import, and a dynamo key), runs it in a subprocess, and asserts
each one fails in its own window while the restoring idioms pass. It was verified by removing each
window in turn: every removal fails exactly the test for that window.

**The tie rule that goes with it.** Lane E's root cause was the leak, but what the leak exposed was
an EXACT tie. Real policies produce exact and near ties between their top two actions. So a
cross-path comparison of the greedy action (compiled vs eager, served vs reference, rust vs python)
asserts equality only where the reference's top-2 margin exceeds a DECLARED bound. That bound is 2×
the path's |Δ log-prob| bar in the (since-deleted) `rust_env_opponents_parity.judge_flips`, and `_TIE_BAND` in T2's
`inference/service/parity.judge`. Strict argmax equality with no margin is a flake waiting for a
tie.

### The REWARD GOLDEN (`src/agents/training/reward_golden_test.py`) — `sim`, ~20 s

The reward stream, bit for bit: every field of every `RewardBreakdown` as `float.hex()`, for every
decision of **30 real bridge battles** (5 battles x 6 reward COMPOSITIONS), sha256'd —
`9463dc24…`, recorded in `src/agents/training/reward_golden.json` beside it with the commit it was
produced at and a **per-sweep** hash, so a mismatch names the composition and battle that moved
instead of only saying "the reward changed". It is the BEFORE/AFTER reference the `reward_manager.py`
decomposition (`b0b3a253`) was proved on, promoted out of a scratch directory: a byte-identity
reference that lives in one agent's tmp protects exactly one refactor.

Reproducible by construction, all four clauses (see *What "fuzz test" means* below): fixed teams by
pool index, a per-player RNG, a fixed sim PRNG seed, and **`concurrency=1`**. Verified: the hash
reproduces bit-for-bit in a fresh worktree with an independently checked-out `data/`. The
preconditions are ASSERTED, not branched on — a harness that errored or played a different number of
decisions fails with *that* message, never as a reward change. Regenerate only for an INTENDED
change, and record it in the ledger:

```bash
export PYTHONPATH=$PYTHONPATH:src && python3 src/agents/training/reward_golden_test.py --write
```

### The LEARNER GOLDEN (`src/agents/training/learner_golden_test.py`) — unmarked, ~3 s

What ONE PPO update computes (K9(a), M5 Lane K): a production-surface learner rebuilt from a fixed seed
runs one eager fp32 `train()` on a committed 64-row real buffer (`learner_golden_buffer.npz`, 65 KB);
the post-update parameter BYTES and every logged loss are compared EXACTLY against
`learner_golden.json`, KEYED BY `torch.__version__` (the init is identical across torch builds, the
update is not — each interpreter has its own entry and a missing one FAILS). A mismatch names the losses
and parameter groups that moved. It never records; an INTENDED change is re-recorded under every
interpreter with an entry, with a reason that lands in the file's history:

```bash
export PYTHONPATH=$PYTHONPATH:src && python3 -m agents.training.learner_golden record --reason "..."
```

Detail — what is pinned, why exact bytes, the scope limits: `designs/training/learner_gates.md`.

### Unit tests only (the fast inner loop)
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m pytest src/ -m "not slow and not e2e and not sim and not integration" -q
```

**Add `-n 2` — it is ~1.8x faster** and costs the box only two cores, which matters because a
training run normally shares this machine. `pytest-xdist` is in `environment_torch28.yml`.

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m pytest src/ -m "not integration and not e2e" -q -n 2
```

Measured 2026-08-14, 16-core box, whole unit suite, same 4527 passed. **The two columns are not
comparable to each other** — a benchmark on this box is meaningless while a production run shares
it, so the idle and under-load figures are reported separately rather than blended:

| | idle box | beside a live training run |
|---|---|---|
| serial | 147 s | 160-168 s |
| **`-n 2`** | — | **90.4 / 90.9 s** |
| `-n 4` | 56 s | 72.1 / 73.9 s |
| `-n 8` / `-n 12` / `-n 16` | 59 / 57 / 59 s | — |

`-n 4` is the floor: past it the wall is one long-pole file (the `torch.compile` gates), so more
workers buy nothing. Under real conditions `-n 4` saves only ~17 s over `-n 2` while taking twice
the cores from the run — hence `-n 2` as the default and `-n 4` when the box is yours. Use plain
serial when you need `-s`, a debugger, or a readable single failure.

> ⚠️ **The parallel speedup depends entirely on BLAS thread pinning, and the root `conftest.py` now
> does it for you** (`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`; escape hatch
> `GEN3AI_TEST_ALLOW_THREADS=1`). Unpinned, `-n 8` measured **389 s — 6.5x SLOWER than pinned
> serial**, with `user` time at 68 min vs 3 min: N workers x 16 BLAS threads on 16 cores thrashes the
> box. It is the same ~38x cliff `src/main/thread_pinning_test.py` defends for env workers. Without
> the conftest pin, anyone trying `-n auto` would measure a slowdown and conclude parallelism does
> not work here.

### Targeted sets before a ship (owner, 2026-10-05)
The full serial suite is NOT required before every `/gen3ai-ship` — **"I prefer more targeted test suites."** An agent ships on the routine gate, or on a TARGETED set it chooses by judgment (the tests that cover what it changed, plus the static gates), **naming the scope and the reason in the commit body**. Shared infrastructure (conftest, the model, the training loop, compile, `data/`, the Rust core, the launcher) or any doubt means the full routine gate. The owner's 2026-09-30 rule still holds: the choice is the agent's, never an AUTOMATED test-selection framework.

### Everything, including the slow tiers (requires symlinked deps/pokemon-showdown + chrome)
**Run this in CI, or when a unit's blast radius warrants it** (not required before every `/gen3ai-ship`; see above). ~47 minutes serial (2026-09-29); the browser suite is ~19 s of it.
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 -m pytest src/ -q
```

⚠️ **A FRESH WORKTREE pays for a cargo build on its first rust-backed test**, which
saturates every core and can turn the contention-scaled per-battle timeouts into a wall of
TIMEOUTs. Observed twice, both times misreading as a rust defect: `bridge_impl_parity` reported 8
of 12 battles timed out and one transport error, and `better_line[rust]` reported a candidate
divergence — **both passed on the warm tree with no code change.** Build the binaries first, or
discount the first run in a new worktree. **The suite runs the EMISSION SELF-CHECK build** (the root
`conftest.py` sets `POKESIM_EMISSION_SELFCHECK=1`, so every rust child is `target/selfcheck/<bin>`:
every emitted protocol line checked as it is emitted, a failure kills the child —
`designs/rust_sim/emission_selfcheck.md`), so that is the build to warm:
```bash
cargo build --profile selfcheck --features emission-selfcheck --bin sim_bridge --bin search_driver --bin core_events --manifest-path src/rust_sim/Cargo.toml
```
The fuzz scripts run on it too when run directly (`sim_bridge_bin._auto_selfcheck`: an entry script
named `*fuzz_test.py`). A benchmark runs the PRODUCTION build (`cargo build --release`) — it measures
what training pays.

### Fuzz tests (`*_fuzz_test.py`, run directly as scripts)
Run battles **in-process via the local BattleStream bridge — no `npm run showdown`
needed** (`utils/bridge/local_battle_runner.py`):
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 src/agents/action/fuzz_test.py [n_battles]
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 src/agents/training/poke_env_gaps/transition_fuzz_test.py [n_battles]
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 src/agents/battle/event_log_fuzz_test.py [n_battles]
# also bridge-backed (no server): poke_env_gaps/{abilities,item_consumption,move_outcome,snatch,incoming_damage}_fuzz_test.py
#                                  poke_env_gaps/move_alignment_fuzz_test.py (per-move obs features ↔ legal.move_slots[k] ↔ action 6+k, forces Choice-lock/Disable)
#                                  poke_env_gaps/faint_attribution_fuzz_test.py (a recorded `<side>:<species>:fainted` names the mon)
#                                      PROTOCOL says fainted — the switch-in-dies case the old decision-time-active label got wrong)
#                                  poke_env_gaps/damage_op_probe_fuzz_test.py (AUTHORITATIVE DamageOperator physics gate — CONSTRUCTED single-turn)
#                                      scenarios via the OMNISCIENT BattleStream `utils/bridge/damage_probe.js`: exact both-side HP + the sim's OWN
#                                      stats, zero measurement confounds; one modifier per scenario [type/STAB/SE/resist/4×/immunity/Thick Fat
#                                      Choice Band/item/boosts/burn/screens/weather]) + poke_env_gaps/damage_op_fuzz_test.py (looser random-game net)
#                                  training/hidden_power_tracker_fuzz_test.py
#                                  utils/bridge/reconstruction_fuzz_test.py (battle replay/re-roll invariants)
#                                  utils/bridge/reroll_many_parity_fuzz_test.py (batched reroll_many == per-call reroll_turn, bit-for-bit obs)
#                                  utils/bridge/search_clone_parity_fuzz_test.py (serializeBattle clone == reroll_many, bit-for-bit obs + value_crn anchor + depth-2)
#                                  and training/obs_roundtrip_fuzz_test.py (offline obs == live obs, bit-for-bit)
#                                  battle/rust_core_trackers_fuzz_test.py [--minutes N] (slice T on FRESH battles: the Rust core's
#                                      per-decision trackers / α-β label / reward == EpisodeTracker, + slices E, V and O — the
#                                      Rust ENCODER's 2501-dim row byte-equal to Gen3ObservationEncoder's — pool + mechanic-dense
#                                      + procedural teams; the fold-equivalence fuzz's shape re-pointed at the core)
```

### THREE TEAM SOURCES — the pool, the procedural generator, and the LADDER-USAGE corpus

Every gate that plays teams can play three sources (`utils.team_sources`, `gen3_ladder_usage_corpus_v1`):

| source | what | surface |
|---|---|---|
| `pool` (default everywhere) | the 719 training teams | what training plays; one narrow human meta |
| `procedural` | `src/rust_sim/harness/ou_random_teams.js`, Smogon-derived, `TeamValidator`-legal, seeded | wide, on-format, no human built it |
| `ladder` | the Metamon `hl_05_26` gen3ou PUBLIC-LADDER teams, filtered to what the ENGINE plays (`src/utils/ladder_corpus/`) | the teams people bring — where the Heal Bell crash hid |

**The corpus is a TEST corpus, never a prior**, and it lives under `src/utils/ladder_corpus/`, NOT
`data/` (pinned runs read `data/` from main). Filter: `Teams.import` + six Pokemon +
`TeamValidator('gen3ou')` + every species / move / item / ability RUNS in the engine
(`scan_move_probe`, the oracle — not the fuzz picker's `isModeledMove`, which rejects Sleep Talk).
Measured 2026-09-24: **22,813 of 22,862 kept (99.79%)**; 49 dropped, all engine fail-louds (Metronome
11, Shell Bell 11, Snore 10, Psywave 9, Fly 3, Blast Burn 2, Dig / Grudge / Triple Kick 1); Heal Bell
teams kept (440). `manifest.json` records the filter, the counts and every tier's sha256; every
reader (Python and JS) REFUSES a data file that does not match. Rebuild (deterministic, `--check`
proves the committed bytes reproduce): `python -m utils.ladder_corpus.build`.

| tier | teams | runs in |
|---|---|---|
| `commit` | 16 | the COMMIT-tier recorded battles (`ladder_0`, `ladder_1` of the Rust Core parity fixture) |
| `milestone` | 800 | slice E/V MILESTONE (2 × 150 battles, 600 teams once each) and the fuzzers' default draw |
| `full` | 22,813 | the CUTOVER tier; `--ladder-tier full` soaks |

```bash
node src/rust_sim/harness/ab_fuzz.js --mode ladder [--ladder-tier full] --battles 200      # + --protocol --format gen3ou
node src/rust_sim/harness/bridge_ab_fuzz.js --mode ladder --format gen3ou --battles 100
node src/rust_sim/harness/gen_sim_bridge_diff.js --mode ladder --format gen3ou --persistent --battles 100
python3 src/agents/training/obs_roundtrip_fuzz_test.py 20 20 --team-source ladder         # also: event_log /
#   core_row_parity / live_view_memo fuzz scripts take --team-source {pool,procedural,ladder}
```

The parity harness's hook is `rust_core_parity.play(key, source="ladder")`, so slices T and O get
all three sources by calling it. **The ladder MILESTONE tier has NAMED known divergences**
(`rust_core_parity.LADDER_KNOWN_DIVERGENCES`, each with its backlog row); they run in their own test
and must still fire in exactly their named classes, so an entry that outlives its fix fails.

**The FULL Metamon smoke** (`python -m main.ladder_usage_smoke`, `gen3_ladder_usage_smoke_v1`) plays
every one of the 22,862 teams once (11,431 battles, the registered n) on the NODE bridge with both
players encoding every decision — INCREMENTAL per `ORCHESTRATOR_SOP.md` §2: 50-battle units, rows
written atomically to a durable directory, resumable (a restarted driver skips the units on disk —
pinned by `main/ladder_usage_smoke_test.py` and proven by a real kill + resume), detached
(`run --detach`), `status` for progress, `report` refuses before the registered n.

**The Rust core CUTOVER stress and the M5 milestone harness are DELETED** (`main/rust_core_cutover/`, `main/rust_core_m5/`, deletion pass U3): the cutover is done, and the Python env they compared the core against is gone. Their pre-registered targets and verdicts are history (`designs/endstate/program_rust_core.md` §3, the ledger). The pattern they proved — minutes-long units, one atomic row each, resumable, `nice 19`, a worker cap, a governor on the live arm's fps, run from a `git archive` PIN (never a worktree: a binary compiled in a worktree panics once it is removed) — remains the template for any new long measurement driver.

### E2E tests (`*_e2e_test.py` / `*_fuzz_e2e_test.py`, require a live server)
```bash
# Start server first: npm run showdown
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 src/agents/action/telemetry_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 src/agents/training/poke_env_gaps/effectiveness_fuzz_e2e_test.py [n_battles]
```

### Benchmarks (`*_benchmark.py`, run directly as scripts)

Six profilers, each answering a different question. All print a loud **"THE BOX IS BUSY"** banner
via `warn_if_contended()` when the box is not idle — a benchmark's output IS the measurement, so
its bounds are never scaled, only warned about.

🚨 **An A/B over repeated updates must RESTORE every update counter per repeat and COUNTERBALANCE
the arm order** (T F F T, never T F T F). `_n_updates` drives `--diagnostics-every`, so un-restored
repeats run different optional probes, and a fixed order confounds the arm with its position. The
2026-10-01 cuDNN-flag read had both confounds; its 2.6 % gap was not attributable to the flag, which
ran zero kernels (`research_state/measurements/own_ppo_loop/`). `learner_benchmark.capture_model_state`
/ `restore_model_state` do the restore.

```bash
export PYTHONPATH=$PYTHONPATH:src
# WHERE the obs pipeline's time goes (component breakdown + cProfile ranking)
python3 src/agents/training/obs_build_benchmark.py [--turn 25] [--reps 400] [--top 22] [--battles 200] [--seed 0]
# WHERE a whole trainer turn's CPU goes (parse + obs + reward + mask + map + tracker), GPU-excluded
# (a standalone profiler that MIRRORS the per-decision stages the deleted Python `Gen3Env` ran — it imports no env)
python3 src/agents/training/trainer_turn_benchmark.py [--decisions 150] [--warmup 3] [--seed 0] [--pin-battles] [--bridge rust|node]   # rust is the default (training's)
# A/B one implementation of LiveView.from_battle against the previous one, on ONE frozen board
python3 src/agents/training/live_view_build_benchmark.py [--reps 2500] [--rounds 6] [--turn 12] [--profile]
# WHERE ONE SEARCHED DECISION's wall goes — the real SearchEngine (the Rust-core road) over BANKED
#   eval traces, in stack-accounted phases (port open_root / expand_many / row wrap / glue)
python3 src/main/search_dividend/search_decision_benchmark.py --traces models/<run>/eval_traces \
    [--decisions 12] [--m-opp 3] [--n-actions N] [--arm honest] [--k-worlds 4] [--rust-timing] [--cprofile out.prof]
# WHERE ONE PPO UPDATE's wall goes — the REAL train() of arm C on ONE real rollout buffer (collected by
#   the trainer in-process as a fork into ~/gen3ai_archive/learner_bench/, never models/), K repeats
#   from identical state; phases (sync-bracketed), ablations (incl. `diag_skipped`: an update the
#   --diagnostics-every cadence skips, at the run's own flags), a per-epoch torch.profiler read (T13).
#   cuda REFUSES a GPU with any compute process or a live trainer; --tiny is the CPU code-path check
python3 -m agents.training.learner_benchmark run --device cuda [--k 5] [--warmup 1] [--buffer <pkl>] [--publish <dir>]
python3 -m agents.training.learner_benchmark run --device cpu --tiny
```

🚨 **`learner_benchmark` (and `main.compile_inventory`, which reuses its argv builder) runs K9(b)'s
behaviour check at `--behaviour-check warn`.** A PINNED buffer's stored log-probs come from the
rollout that collected it — another code and torch — so the check compares the learner with a different
program by construction.

🚨 **Both tools drop, from C's recorded command, every flag the trainer's parser no longer knows**
(`learner_benchmark._unknown_to_the_trainer`): each deletion unit removes flags that command still types
(`--matmul-precision`, the entropy boosts, `--no-value-true-team`, …) and one unknown flag is an argparse
exit that kills the worker before it measures anything.

🚨 **Both tools measure the LEARN LOOP's update, never the trainer's STARTUP update.** The trainer runs a real
`train()` BEFORE `learn()` — the CUDA fit check's dry update (`update_fit.dry_update`, on a tiled FIXTURE of the
buffer's declared shape). A worker that replaced `train` for the whole process took that call as its
measurement: the compile inventory's time stage timed a 2-env, 4,096-row fixture (0.32 s) instead of the pinned
98,304-row buffer, wrote its result and exited before `collect_rollouts` ever ran (K2, 2026-10-02: the reason no
fp32 baseline could be banked), and the benchmark's fresh-collection mode would have pickled the fixture as "the
rollout". `learner_benchmark.learn_loop_only` / `install_worker_hooks` are the fix — the tool's `train` is live only
from the loop's first `collect_rollouts` (`_WORKER["learning"]`), and the time stage refuses to run unless
`buffer_restored` — pinned by `learner_benchmark_test::test_the_tool_never_takes_the_trainers_startup_update`
(the REAL dry update on the golden learner; it fails on the revert). Both workers run their repeats inside
`global_rng_guard.isolated_global_rng()` (each repeat SEEDS the global streams to replay the same minibatch
permutations, and a global reseed after the learner froze is `FATAL_CONFIG`). The time stage REQUIRES
`--keep-prewarm`: the K8 regions are installed by the compile sentinel, and the old skip-the-prewarm variant
replaced the sentinel, never installed them, and died "not compiled" after the whole startup.

🚨 **`learner_benchmark`'s bracketed phases are NOT the un-bracketed update cut into pieces.** Every
mark adds a `torch.cuda.synchronize()`, which removes exactly the CPU/GPU overlap an update with
~270 host-blocking scalar reads per micro-batch lives on. So it reports BOTH train_ms sets, and the
launch/sync-bound question is answered by the profiler windows (GPU busy % = device-interval union
over the window's span; launches and syncs per micro-batch), not by the phase table.

🚨 **A cProfile SHARE IS NOT A WALL SHARE, and it has already cost this project a wrong
priority.** `designs/rust_sim/one_sided_view.md` carried "`map_actions_at` is ~24% of the (since
deleted) view road's per-arm wall" off a cProfile run; measured with `search_decision_benchmark`'s wall timer it
is **1.2%**, and re-running that same benchmark UNDER cProfile does not restore the 24% either.
In the same pair, encode reads 9.5% un-profiled and 15.6% profiled and the action mask 14.7%
against 2.2%. Rank work by a wall measurement; use cProfile to find WHICH function, not HOW MUCH.
(`designs/research_state/measurements/search_profile_2026-09-22/README.md`.)

🚨 **Every change under `src/agents/observation/` must run `obs_build_benchmark` before/after and
confirm no meaningful regression.** That gate, the canonical baseline and the load-stable
regression criteria are in `src/agents/observation/CLAUDE.md`. Absolute ms scale with machine load;
the component **ratios** and the cProfile ranking are the load-stable signal.

🚨 **`--pin-battles` is REQUIRED for any before/after or arm-vs-arm claim.** Unpinned, each
invocation walks a fresh RANDOM battle (the `--seed` fixes the team draw and the action picks, NOT
the sim dice), so two runs profile two different boards: measured, the run-to-run spread was LARGER
than the effect being tested and carried the wrong SIGN. It is off by default so the headline share
table still samples the board distribution.

🚨 **The first two benchmarks cannot measure a CHANGE to `LiveView.from_battle`** — they walk a
fresh random battle per invocation, so two consecutive runs profile two different boards. That
mistake was made and briefly believed. Use `live_view_build_benchmark`, which freezes one seeded
board and alternates the arm order.

⚠️ **A memoized value is billed to whichever stage asks for it FIRST.** `battle.live_view()` is
memoized per state-epoch and five stages read it, so the whole board build was charged to
`obs: legal + mask` — which was then named as the next optimization target on the strength of a
number that was 88% someone else's work. **When a stage looks expensive, check whether it is merely
first.** `trainer_turn_benchmark` now times the shared build on its own line.

⚠️ **Reward cost depends on the REWARD COMPOSITION**, so a single baseline is meaningless without
one: `--reward-argv '<train_rl_agent flags>'` times any composition through the launcher's own
parser, and prints the resolved census with the run header.

Baselines with their provenance:
`designs/research_state/measurements/post_paydown_baselines_2026-08-23.{json,md}`; the superseded
figures and the full share tables are in
`designs/research_state/claude_md_archive/benchmark_measurements.md`.
### Performance-shape tests — speed is checked by STRUCTURE, plus a benchmark at milestones (owner, 2026-10-02)

**There is no wall-clock gate.** `compiled_perf_guard_test` (one production update held to a banked time) was RETIRED:
a timing test is noisy, needs an idle box and goes stale — its only baseline was TF32's, which K2 retired and left it
skipping, blind, until lane C re-measured it. Performance is checked two ways:

1. **Deterministic functional tests of the properties that make the update fast** — counts and graph facts, never a
   time, in every routine gate. The audit (existing + new), what each catches:

| property | test | catches |
|---|---|---|
| the compiled inventory EQUALS the declaration (graphs = regions × signatures, one cache entry per region code object) | `compile_regions_test::test_the_compiled_inventory_EQUALS_the_declaration_table` (two real updates + eager rollout forwards of every mask dtype at batch 4 and 1 — which take NO region route — then the count) · `test_the_run_asserts_the_compiled_inventory_equals_the_declaration` | an extra, missing or duplicated region graph; a region that stopped compiling |
| 0 compiles / rejections after the lock | the same test (`compiles_after_lock == 0`) · `compile_control_test::test_b_*` (a new shape / new code object after the lock is the typed FATAL), `test_an_UNDECLARED_signature_on_the_FIRST_iteration_is_FATAL_not_absorbed`, `test_attach_locks_BEFORE_the_first_iteration…`, the cache-limit detector tests · `r1_declared_levers_test` (every lever an update reaches is declared at startup) | a recompile in the steady state (each is seconds to minutes), an undeclared signature |
| no silent fall-back to eager | `compile_regions_test::test_a_region_that_runs_EAGER_on_its_COMPILED_route_is_FATAL`, `test_dynamo_DISABLED_at_startup…`, `test_the_lock_refuses_every_switch_that_makes_dynamo_run_eager_silently`, `test_a_graph_break_INSIDE_a_region…`, `test_a_RAGGED_micro_batch_is_REFUSED_by_R1s_dispatcher_and_compiles_nothing` (fails on revert of the F9 deletion: the old route returned an eager loss), `test_each_update_records_its_compiled_and_eager_region_calls` | a region whose compiled route ran its Python body; a graph break; a ragged micro-batch run eager or compiled as a new signature |
| **every micro-batch of a REAL update takes R1's compiled route, none eager** *(new)* | `update_performance_shape_test::test_a_real_update_runs_every_micro_batch_through_R1_compiled_and_none_eager` (+ its teeth) | R1 routed AROUND its dispatcher — the eager body run uncounted. **The 2.9× regression the benchmark read (116 s vs 40 s, share 0.000) leaves every run-level FATAL silent**; only the route counts of a real update see it |
| **a bounded number of host scalar reads per update** *(new)* | `update_performance_shape_test::test_a_real_update_makes_a_bounded_number_of_host_scalar_reads` (exact pin: 1,050 on the golden's 8-micro-batch update; production 9,942 vs 66,784 before K8) | a `.item()` / `float(tensor)` added to the per-micro-batch path (each is a device sync; ×480 micro-batches per production update) |
| **attention lowers to the FUSED SDPA kernel, not MATH** *(new)* | CPU: `update_performance_shape_test::test_R1s_attention_profile_on_cpu_is_pinned` (R1's AOT forward graph: 2 fused + 2 MATH on the golden) · CUDA (`slow`, `GEN3AI_TEST_ALLOW_GPU=1`): `test_on_cuda_the_trunk_layer_dispatches_a_fused_sdpa_kernel_not_math` (`_scaled_dot_product_efficient_attention`, never bmm + `_safe_softmax`) | a global `enable_*_sdp(False)`, a hand-rolled attention, a forced `sdpa_kernel(MATH)` (+5.2% of R1 fwd+bwd, `measurements/k6_k8/r1_noise/`). The CUDA cell is the trunk's pin (on CPU a grad-requiring bias is MATH by construction); the CPU cell is the routine-gate net |
| the micro-batch is never re-copied whole from the host per micro-step | `instrumented_ppo_device_batches_test`: resident = ONE copy per update (`copies == 1`); staged (production's default) = one copy per micro-batch (`copies == micro-batches`) and **`test_the_staged_gather_moves_one_micro_batch_per_micro_step_never_the_whole_buffer`** *(new: the bytes per copy = one micro-batch, 2 in flight)* | a gather that stages the whole buffer per micro-step (~11 GB of H2D per update at production shape) |
| **an eval-ledger claim / append / request-scoped read costs the same at any archive size** *(new, F-ED-22)* | `eval_ledger/index_test::test_a_cycle_parses_the_same_lines_at_any_archive_size` · `test_a_read_of_one_request_parses_only_its_rows_at_any_archive_size` (EXACT equality of `store.IO` line counts between a 2- and a 12-cycle archive, and the fast path TAKEN) + their teeth `test_the_counters_see_the_pre_index_full_fold` (`GEN3AI_LEDGER_INDEX=0`: the same counter grows) | a regression to a whole-archive fold per claim (7 µs per event: 0.5 s per claim at 50,000 events) or a reader that silently falls back to the scan |
| no CUDA stream / CUDA graph / optimizer / parameter / module built after startup | `learner_lifecycle_test` (`test_healthy_steps_after_the_freeze_acquire_nothing`, `test_a_production_surface_update_acquires_nothing…`, the lazy-acquisition FATALs) · `learner_lifecycle_gate_test` (static: no training-step path constructs one, `cuda_resource` kind included) · `instrumented_ppo_device_batches_test::test_on_cuda_staged_updates_strand_no_cache_on_a_new_stream` (`slow`, CUDA: forty staged updates hold ONE stream's cache and a flat reserved total) | a per-update stream (the +46–66 MiB/update climb of sizing arm B), a lazy optimizer |

   Every new test carries a TEETH case or was run against the planted regression through the same seam (`update_performance_shape_test`'s
   module docstring; the planted failures are listed in the commit). **When a pin moves on purpose** (the host-read count after
   removing a read, the attention profile after a deliberate change) update the number in the same commit and say why — an
   exact count is a reviewed edit, never a threshold that drifts.

2. **The on-demand update BENCHMARK, at milestones** — after a torch upgrade, before a baseline run, on suspicion:

        scripts/ops/gpu_lock.sh timeout 1260 scripts/ops/mem_cap.sh 48 python -m main.compile_inventory run \
            --stage time --device cuda --keep-prewarm --unbracketed --out-root <dir>        # ≈ 6.5 min, idle GPU

   It times the production update on the pinned 98,304-row buffer (arm C's checkpoint) and reports the profiled compiled
   share; `learner_benchmark` gives the phase / kernel breakdown. The pinned buffer is reused on the Rust core since lane C
   fixed the tools to leave the trainer's startup update (the fit check's dry `train()`) alone. Its 2026-10-02 read — 40.17 s,
   compiled share 0.892, n = 5 — and the planted-regression and loaded-box units are in
   `designs/research_state/measurements/k6_k8/update_benchmark_fp32/`; **a read, not a baseline: it goes stale, re-run it.**

### What "fuzz test" means in this project

**Fuzz tests run real battles — by default in-process via the local BattleStream bridge (no server), or against a live server — and validate observations or behaviour against the actual protocol stream.** They are NOT deterministic scenario tests with fixed inputs.

The canonical pattern (see `src/agents/training/poke_env_gaps/`):

1. Subclass `Player` and override `_handle_battle_message` to intercept raw Showdown protocol lines mid-battle.
2. Archive per-turn snapshots of the state you care about (e.g. which items were consumed, which moves were used).
3. In `choose_move()`, validate that the encoded observation vector matches what the archived protocol events say should be there.
4. Run N random battles; any validation failure raises immediately with a detailed error.

This catches poke-env parsing bugs and encoder gaps that unit tests with mocks cannot — the test exercises the Showdown sim → poke-env → encoder pipeline end to end (the bridge feeds the identical protocol stream the live server would). When asked to write a fuzz test, always follow this pattern rather than writing parametrized unit tests with hand-crafted mock objects.

🚨 **A fuzz SCRIPT wants a new battle every run; a pytest-collected TEST wants the same battle
every run.** A fixture seeded from the wall clock plays a different battle every run and
eventually plays the one that *skips its own assertion* — a battle too short to anchor, a tie
that empties the frame, a line the search outruns. Three shipped and were chased as flakes
(`better_line`, `cf_audit`, and the guards this rule was written from). So a collected test
takes its battle from **`obs_roundtrip_fuzz_test.record_fixture_battle(out_dir, key=…)`** —
reproducible across processes, `key` selecting a different deterministic battle when you want
variety — and asserts its precondition rather than branching on it (`if x is not None:` around
the decisive gate is the antipattern's second half; it fails green). ⚠️ **`random.seed(k)` is
NOT enough**: two players share the global `random` and the bridge interleaves their
`choose_move` calls, so the draw order still diverges (`golden_obs_capture` measured the
decision count swinging by hundreds). Reproducibility needs every randomness source *removed* —
fixed teams, a per-player RNG, a fixed sim seed, **and `concurrency=1`**. Where a test needs a
*qualifying* battle (cf_audit's non-tie, an ending branch arm), redraw over a bounded FIXED
sequence of keys.

🚨 **The concurrency clause is the one that gets missed, because the seeds look sufficient.** They
are not: with the sim dice, both players' policy sampling AND the pool sequence all pinned, a
`run_local_battles(..., concurrency=3)` measurement still wandered — two runs produced **1193 vs
1141 captured states with per-arm levels up to +0.043 apart** (measured 2026-09-03, the offline
collateral-KL column). Interleaved battles consume the shared streams in a *scheduling-dependent*
order, so the seeds are consumed in a different sequence each run. At `concurrency=1` the same two
runs are **byte-identical** — same state count, every value equal to 6 decimals, same sha256. A
measurement that needs quotable LEVELS must therefore serialize; one that needs only orderings or
paired differences may keep the concurrency and say so. Prefer REFUSING the unreproducible
configuration over emitting a quietly-wandering number
(`designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/` does this, and
keeps the divergent runs as the evidence).

**The "per-player RNG" half is now a set of five OPT-IN seeds**, one per drawer that used to reach
into a process-wide RNG — `$GEN3AI_{PLAYER,TEAM,POLICY,POOL,STALLER}_SEED`, or the matching ctor
kwargs. Every default is unchanged (unseeded, the RNG *is* the shared module), and any paired-arm
design over battles should set all five. Measured: under the **same fixed sim seed**, unseeded arms
played different games (84/145 turns vs 212/233, different winners); seeded they were identical.
The table, the four seams, and the census that found them are in
`src/agents/training/CLAUDE.md` → *GLOBAL-RANDOM COUPLING* and
`designs/research_state/measurements/global_random_sweep_2026-08-30.md`.

---
