# Testing — the full chapter

Moved verbatim out of the root `CLAUDE.md` on **2026-09-07**, when the root was reduced to a
constitution + command card + map (~10k tokens, loaded into EVERY session including every
subagent's). **The root keeps the command table, the tier markers, the four static gates and the
binding hazards; this file is the detail behind them.**

Covers: running beside a live training run (contention-scaled timeouts), the two-axis tier system,
file-naming conventions, which command to run, the four static gates, the fuzz-test pattern and its
reproducibility rules, and the benchmarks.

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
### Test tiers — TWO AXES, and keeping them apart is the point

A marker says **what a test NEEDS** (capability). A separate marker says **what it COSTS** (`slow`).
Collapsing those into one axis is what the old single `integration` marker did, and it failed in
**both** directions at once.

| Marker | Answers | Values |
|---|---|---|
| capability | *can this run here?* | *(unmarked)* · `integration` · `sim` · `browser` · `e2e` |
| **cost** | *should this run routinely?* | **`slow`** |

| Tier | Needs | Count (2026-08-23) · duration (2026-08-14) |
|---|---|---|
| *(unmarked)* | nothing — pure in-process | 6570 tests, **127 s** serial (~56 s at `-n 4`) |
| `integration` | an out-of-process dep, no battles, no browser | 158 tests, ~16 s total |
| `sim` | real battles in-process via the bridge, no server | 60 tests, ~100 s total |
| `browser` | headless chrome | 53 tests, **1426 s** — ALL of it also `slow` |
| `e2e` | a live Showdown server | run directly as scripts |
| `slow` | *(orthogonal)* minutes, not seconds | 75 tests |

**Counts are dated on purpose — RECOUNT before quoting one** (`pytest -m <tier> --collect-only -q`);
this corpus moves faster than the doc describing it. Durations were taken on a quiet box and have
not been re-taken, because a duration measured beside a live run is not a measurement.

🚨 **MEASURE BEFORE YOU TIER — the intuitive answer was wrong.** The **browser** tests are 88% of
the whole integration tier (1426 s of 1623 s), because each launches a fresh headless chrome at
~25 s of cold start; every bridge test *combined* is ~170 s. Tier by the profile
(`pytest -m <tier> --durations=0`), not by which subsystem feels riskiest.

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

### COMPOSITION gates — where a whole RUN is the unit under test

Some properties exist only in the JOINS between subsystems, and no leg-level test can see them: the
legs are exercised with fake sessions, hand-built shards and pre-recorded fixtures, so a defect in
the hand-off between two of them is invisible to every one of them at once. Those get a test that
launches the real entry point as a SUBPROCESS at smoke scale and asserts on the ARTIFACTS the run
leaves behind — never on "it did not crash".

| Gate | The composition it closes | Tier · measured |
|---|---|---|
| `src/main/train/search_teacher_composition_test.py` (`gen3_search_teacher_composition_rust_v1`) | **>= 2 search-teacher cycles on `--use-bridge rust`**, end to end: eval traces → falsify-gated selection → frozen trainee → worker subprocess on the rust `search_driver` → confirm rollouts → shard → `CorrectionBuffer` → the AWR aux loss inside `train()`, across a cycle boundary. Asserts the per-cycle markers, the worker config's `"impl": "rust"`, the status histogram (no `worker_no_shard`, no `error:*`) and the TB scalars (`teacher/corrections_per_cycle`, `teacher/loss`, `teacher/n`, `grad/searchteacher_share`) | `sim` + `slow` · **10 m 38 s** (2026-09-07, 16-core box, contention factor 1.32; 8 cycles, 34 candidate-shots, 4 corrections) |

**They are `slow` by DECLARATION, not by measurement.** A live training run normally shares this
box, so a duration recorded beside one is a note for planning, never a bound to assert against.

Three rules every composition gate here follows, each of them a project rule applied to a subprocess:

- **Every output path goes to the test's `tmp_path` — 🚨 NEVER under `models/`.** The run archive is
  not a scratch space; `train_rl_agent.py`'s `--run-dir` takes the temp dir.
- **The child is bounded by a `ProgressDeadline` on its own log, not by a total-duration cap.**
  Contention stretches duration; only a real wedge stops output. A wedge is reported as
  **INCONCLUSIVE**, in wording distinct from a failed assertion and carrying
  `describe_contention()` — a timeout is never a semantic outcome.
- **A missing precondition FAILS, it never skips** — an unbuilt rust binary fails with the exact
  `cargo build` line, and a run that reaches "Training complete" having launched fewer than two
  cycles fails as NO-CYCLE rather than passing green on a run that did nothing.

### Test file naming conventions

| Pattern | Requires | Marker |
|---|---|---|
| `*_test.py` | Nothing — pure unit tests with mocks | — |
| `*_integration_test.py` | An out-of-process dependency, no live server. **The name is historical and no longer implies the tier** — these split across `integration` (light), `sim` (bridge battles) and `browser` (headless chrome, always `slow`). Read the file's `pytestmark`, not its name | `integration` and/or `sim` / `browser` / `slow` |
| `*_fuzz_test.py` | `deps/pokemon-showdown` — runs **real battles in-process via the local BattleStream bridge** (`utils/bridge/local_battle_runner.py`); **no live server**. The default for fuzzing. | none — run directly as scripts (no `test_*` funcs, so `pytest` imports but collects nothing) |
| `*_fuzz_e2e_test.py` | A **live Showdown server** — fuzz whose checks need real async-server timing (e.g. `effectiveness_fuzz_e2e_test`, whose TurnDelta-vs-BattleContext effectiveness window is decision-timing-sensitive) | run directly as scripts |
| `*_e2e_test.py` | A **live Showdown server** on localhost:8000 | `@pytest.mark.e2e` (scripts only, run directly) |
| `*_benchmark.py` | `deps/pokemon-showdown` bridge (no live server) — **performance profiling, not pass/fail**: plays a real battle in-process, then `cProfile`s a hot path | none — run directly as scripts (no `test_*` funcs → `pytest` collects nothing). Place in a dir with no stdlib-shadowing names (e.g. `training/`, not `observation/`) |

### Which command to run

| When | Command | Count (2026-08-23) · duration (2026-08-14) |
|---|---|---|
| **inner loop** — you want the fastest true/false | `-m "not slow and not e2e and not sim and not integration"` | 6570 tests, **127 s** (~56 s at `-n 4`) |
| **THE ROUTINE GATE** — before a commit | `-m "not slow and not e2e"` | 6676 tests, **4 m 36 s** |
| **before a `/gen3ai-ship`, and in CI** | `pytest src/` (everything) | 6751 tests, **31 m** |
| just the bridge | `-m sim` | ~100 s |
| just the browser views | `-m browser` | ~24 m |

```bash
# THE ROUTINE GATE — everything cheap, whatever it needs. Add -n 2 (~1.8x, two cores).
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not slow and not e2e" -q
```

**Do not use the old `-m "not integration and not e2e"`.** It is what let the obs-golden linchpin
rot on main three times: `integration` now spans a ~100x cost range, so excluding it throws away
cheap, high-value coverage (bridge battles, data parity, mechanics) to avoid the browser suite. Cut
on **`slow`** instead — that is the marker that means "expensive".

### The two STATIC gates (mypy + ruff) — default-on, in every tier

Static checking is enforced by **tests**, not by habit, because there is no CI on this box: the
routine suite is the only thing that runs on every change, so a check outside it is advisory and
rots. Both are unmarked (they run even in the fast inner loop) and both are ~free:

| Gate | Runs | Scope | Measured |
|---|---|---|---|
| `src/agents/model/mypy_gate_test.py` | `python -m mypy` (**no path argument** — the scope comes from `mypy.ini`) | `src/agents/model` **+ `src/agents/observation`**, per `mypy.ini`'s `files =` | **0.28 s warm**, 19.6 s cold |
| `src/ruff_gate_test.py` | `ruff check src/agents src/main src/utils --select F,E9 --exclude src/poke_env --exclude src/rust_sim` | `agents/` + `main/` + `utils/` | **0.10 s** |

They are complementary, not overlapping: mypy is deep over a **declared short list** of packages
(`mypy.ini` sets `files = src/agents/model, src/agents/observation` with `follow_imports = silent`,
so the rest of the tree is read for types but not reported), while ruff is shallow over everything.
**Widening is a `mypy.ini` edit and the test follows it** — the gate invokes bare `python -m mypy`
with no path argument precisely so the config is the only scope declaration, and its
`_CHECKED_PACKAGES` assertion fails if `files =` is ever shrunk by accident. `--select F,E9` is
pyflakes + syntax errors only — findings that mean the code is **wrong**, never a style opinion, so
the gate cannot degrade into a formatting argument.

**A missing tool FAILS, it does not skip** (both are pinned in `environment.yml`) — a linter that
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

### Unit tests only (the fast inner loop)
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not slow and not e2e and not sim and not integration" -q
```

**Add `-n 2` — it is ~1.8x faster** and costs the box only two cores, which matters because a
training run normally shares this machine. `pytest-xdist` is in `environment.yml`.

```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -m "not integration and not e2e" -q -n 2
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

### Everything, including the slow tiers (requires symlinked deps/pokemon-showdown + chrome)
**Run this before a `/gen3ai-ship`, and in CI.** ~31 minutes, ~24 of them the browser suite.
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 -m pytest src/ -q
```

⚠️ **A FRESH WORKTREE pays for a `cargo build --release` on its first rust-backed test**, which
saturates every core and can turn the contention-scaled per-battle timeouts into a wall of
TIMEOUTs. Observed twice, both times misreading as a rust defect: `bridge_impl_parity` reported 8
of 12 battles timed out and one transport error, and `better_line[rust]` reported a candidate
divergence — **both passed on the warm tree with no code change.** Build the binaries first, or
discount the first run in a new worktree:
```bash
cargo build --release --bin sim_bridge --bin search_driver --manifest-path src/rust_sim/Cargo.toml
```

### Fuzz tests (`*_fuzz_test.py`, run directly as scripts)
Run battles **in-process via the local BattleStream bridge — no `npm run showdown`
needed** (`utils/bridge/local_battle_runner.py`):
```bash
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/fuzz_test.py [n_battles]
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/poke_env_gaps/transition_fuzz_test.py [n_battles]
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/battle/event_log_fuzz_test.py [n_battles]
# also bridge-backed (no server): poke_env_gaps/{abilities,item_consumption,move_outcome,snatch,incoming_damage}_fuzz_test.py
#                                  poke_env_gaps/move_alignment_fuzz_test.py (per-move obs features ↔ legal.move_slots[k] ↔ action 6+k, forces Choice-lock/Disable)
#                                  poke_env_gaps/belief_labels_fuzz_test.py (hidden-opp belief labels == actual opp team + no-leak)
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
```

### E2E tests (`*_e2e_test.py` / `*_fuzz_e2e_test.py`, require a live server)
```bash
# Start server first: npm run showdown
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/action/telemetry_e2e_test.py
export PYTHONPATH=$PYTHONPATH:src && /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 src/agents/training/poke_env_gaps/effectiveness_fuzz_e2e_test.py [n_battles]
```

### Benchmarks (`*_benchmark.py`, run directly as scripts)

Three profilers, each answering a different question. All print a loud **"THE BOX IS BUSY"** banner
via `warn_if_contended()` when the box is not idle — a benchmark's output IS the measurement, so
its bounds are never scaled, only warned about.

```bash
export PYTHONPATH=$PYTHONPATH:src
# WHERE the obs pipeline's time goes (component breakdown + cProfile ranking)
python3 src/agents/training/obs_build_benchmark.py [--turn 25] [--reps 400] [--top 22] [--battles 200] [--seed 0]
# WHERE a whole trainer turn's CPU goes (parse + obs + reward + mask + map + tracker), GPU-excluded
python3 src/agents/training/trainer_turn_benchmark.py [--decisions 150] [--warmup 3] [--seed 0] [--pin-battles] [--reward-argv '…']
# A/B one implementation of LiveView.from_battle against the previous one, on ONE frozen board
python3 src/agents/training/live_view_build_benchmark.py [--reps 2500] [--rounds 6] [--turn 12] [--profile]
```

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
