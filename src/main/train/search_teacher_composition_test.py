"""The search-as-teacher COMPOSITION, end to end on the rust bridge
(``gen3_search_teacher_composition_rust_v1``).

**Why this file exists.** Every *leg* of the search-teacher path was already gated on rust —
``better_line`` node≡rust candidate values bit-identical, ``search_clone_parity`` (clone ≡
``reroll_many`` at the obs), the counterfactual confirm leg, ``search_driver_turn1``,
``teacher/callback_test`` (shard → buffer), ``teacher/produce_test`` (the 3-tier gate against a
fake session), ``instrumented_ppo_test::test_search_teacher_*`` (the AWR fold in a real
``train()``). What was NOT gated is the **composition**: a training run that actually performs
more than one search-teacher cycle on ``--use-bridge rust``, producing labels, folding them into
the loss, and surviving a cycle boundary. ``designs/ops/TECH_DEBT_BACKLOG.md`` carried the row as
*"the first such run will find the seam"*, and both ``config.py``'s startup banner and
``src/rust_sim/CLAUDE.md`` said so in prose. This test is that run.

The seam it would catch is exactly the one no leg-level test can: the legs are exercised with fake
sessions, hand-built shards and pre-recorded fixtures, so a defect that lives in the *joins* —
the callback's selection blocking the SB3 loop, the worker subprocess's environment, the
shard hand-off, the second cycle's directory wipe, the buffer surviving a ``train()`` boundary —
is invisible to all of them and visible only here. **It found one on its first run:** the candidate
SELECTION ran INLINE in ``_on_step`` (~30 s over this run's 9 traces; 48.1 s / 350.2 s over 9 and
60 traces of a real run's archive, 2026-09-08) while the
callback was documented, everywhere, as non-blocking. That is fixed — a cycle is now ``select``
(a child) then ``search`` (the workers) — and the per-step cost is a scalar this run writes,
``teacher/step_block_ms``.

THE ARGV, and why each value (a run of 6,000 timesteps, one env, CPU, no Showdown server)::

    src/main/train_rl_agent.py
      --debug                      DummyVecEnv, 1 env — the smallest thing that is still the real
                                   training entry point, and serverless.
      --debug-eval                 REQUIRED. `--debug` skips ALL eval by default, and `crater`-mode
                                   selection reads the run's OWN eval_traces/. No eval ⇒ no traces
                                   ⇒ `select_candidates` returns [] forever and NO cycle ever
                                   launches — the test would go green on a run that did nothing.
      --device cpu                 the GPU belongs to whatever real run shares this box (the teacher
                                   workers hide CUDA from themselves anyway).
      --use-bridge rust            THE POINT OF THE TEST. Passed explicitly rather than relying on
                                   the default, so a change of default cannot silently retarget it.
      --steps 30000                🚨 SIZED BY THE COLLECT, NOT BY THE LAUNCH, AND BY THE YIELD.
                                   RE-SIZED 2026-09-08 when selection moved off the training step:
                                   the trainer no longer stalls 30-100 s per cycle, so the SAME
                                   18,000 steps that bought 8 collects / 34 candidate-shots now
                                   buy 3 collects / 21 shots (the run finished in 284 s instead of
                                   638 s — steps now advance DURING a selection, so a cycle spans
                                   ~4,800 steps instead of ~2,200). 21 shots at the pooled 15%
                                   per-candidate conversion leaves the "total > 0" bar a ~3% flake;
                                   30,000 restores 4-6 collects and 24-40 shots (measured over two
                                   runs: 6 cycles / 8 corrections and 4 cycles / 7) and still runs
                                   well inside the pre-change duration. The rest of this note is
                                   the original sizing, and its reasoning is unchanged:
                                   Measured: the first eval cycle's traces land ~step 1500-2000
                                   (eval is non-blocking), the first cycle with candidates launches
                                   at 3000, and a launch while a cycle is pending is SKIPPED, so
                                   cycles collect roughly every 3,500 steps. A 6,000-step pilot
                                   launched two cycles and collected ONE — the second was still
                                   pending when training ended, and nothing waits for a per-cycle
                                   worker at `_on_training_end`. 12,000 gave 3 launches / 2 collects
                                   / 13 candidate-shots, and 13 shots at the measured 25%
                                   per-candidate conversion leaves the "total > 0" bar a ~2% flake.
                                   18,000 measured **8 launches, 8 collects and 34 candidate-shots**
                                   — cycles collect FASTER once a run's later eval cycles offer only
                                   ~3 candidates each — which puts P(the run yields zero) at ~0.4%
                                   against the pooled 15% per-candidate conversion.
      --n-steps 512                a train() every 512 steps, so the AWR fold gets several chances
                                   after the buffer is first filled. (The default 2048 would give
                                   two folds in the whole run.)
      --batch-size 128 --n-epochs 2   the cheapest optimizer pass that still exercises the fold.
      --eval-freq 500 --eval-games 1 --eval-battles 1
                                   the smallest eval that still writes loss traces with their
                                   `*_reconstruction.json` siblings (1 game x 9 opponents ⇒ measured
                                   9 falsify-gated candidates per cycle, ~30 s of selection). Eval
                                   is non-blocking and skips a cycle while the previous one runs, so
                                   a small freq self-throttles instead of stalling.
      --search-teacher             the master enable (constructs the callback + the buffer).
      --search-teacher-coef 0.5    NON-ZERO on purpose: at coef 0 the AWR term is skipped entirely
                                   and `teacher/loss` is never recorded, so the loss half of the
                                   composition would go ungated.
      --teacher-search-freq 1500   MINIMUM steps between cycle launches (a launch is skipped while
                                   a cycle is pending, so the realized cadence is longer). Selection
                                   is a CHILD, so this cadence costs the training loop only the
                                   spawn; it still bounds how much wall-clock a cycle has, because
                                   the select child must finish before the search workers start.
      --teacher-search-budget 8    🚨 SIZED TO THE COLLECT, NOT TO THE SUPPLY. Selection offers
                                   9-16 candidates here, and taking all of them is what a
                                   production run wants — but the CONFIRM games are
                                   budget x --teacher-confirm-rollouts battles, and a cycle that
                                   takes longer than the run has left never COLLECTS. Measured: 16
                                   candidates x 16 rollouts on 2 workers had not collected 5,000
                                   timesteps after its launch; 8 collects in ~1,500. Eight
                                   candidates still give the run ~40 shots at the strictly-better
                                   gate across its cycles, which is what the yield needs.
      --teacher-confirm-rollouts 16   🚨 THE ONE VALUE THAT WAS MEASURED RATHER THAN GUESSED. The
                                   Wilson strictly-better gate needs the alternative line to win at
                                   least one confirm game, and a 6k-step policy wins one at
                                   p ~= 0.028 (measured: 1 correction from 14 candidates at
                                   --teacher-confirm-rollouts 2, over two pilot runs). At 2 rollouts
                                   a whole cycle yields nothing ~90% of the time; at 16 the measured
                                   conversion is **~15% per candidate pooled** (4/11 offline, 3/12,
                                   2/13 and 4/34 across the gating runs). That is what makes the RUN's
                                   "total > 0" bar safe once --steps buys enough candidate-shots;
                                   at 2 rollouts no number of shots would. Raising it further costs
                                   ASYNC worker time, which overlaps training but DELAYS the
                                   collect, so it trades against the number of cycles.
      --teacher-search-workers 2   two worker subprocesses, so the ~9 candidates' confirm games
                                   finish inside one cycle interval. Not 3 (the production default):
                                   a real training run normally shares this box.
      --log-level periodic         the SearchTeacher markers print at verbose=1 regardless.
      --run-dir <tmp_path>         🚨 NEVER under models/. The run archive is not a scratch space.

**WHY THE LABEL ASSERTION IS ON THE RUN'S TOTAL, NOT PER CYCLE.** Whether a *given* cycle produces
a label is a property of the POLICY's luck against the Wilson gate, not of the composition: a cycle
that searches 9 candidates, runs every confirm rollout on rust, writes its shard and reports
``status={'gate_failed': 9}`` has exercised every join this test exists to gate. Asserting per-cycle
``> 0`` would be a ~2%-flaky assertion about a 6k-step policy. The run's TOTAL being ``> 0`` is the
composition claim — labels reach the buffer and the AWR term folds — and it is the one asserted.

**MEASURED DURATION: 472 s (7 m 52 s) at contention factor 1.10, 2026-09-08, 16-core box with a
production run live** — 30,000 steps, 4-6 collected cycles and 7-8 corrections across two runs.
(Before selection moved off the training step: 638 s for 18,000 steps at factor 1.32 — 2.4x the
steps in 74% of the wall clock, because the trainer no longer stalls 30-350 s per cycle.) Under two
minutes of that is training; the rest is eval and the async selection/search children — the whole
run now costs the TRAINING STEP **0.26 s**, which the run asserts and prints. It is marked ``slow``
regardless of what any future measurement says, per the root
``CLAUDE.md``: a live training run normally shares this box, so a duration measured here is not a
measurement, and the marker is a DECLARATION of cost, not an inference from one.

**A TIMEOUT IS NOT A FAILED ASSERTION.** The child is bounded by a ``ProgressDeadline`` on its own
log (every appended byte is a sign of life), never by a total-duration cap — contention stretches
duration but only a real wedge stops progress. A wedge raises ``ProgressTimeout``, which this test
re-raises as a distinctly-worded INCONCLUSIVE failure carrying ``describe_contention()``, so a
starved box can never be read as a defect in the composition.

**Preconditions FAIL, they never skip.** A missing rust binary fails with the exact cargo command;
a run that reaches "Training complete" having launched fewer than two cycles fails as
NO-CYCLE rather than passing green on a run that did nothing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from utils.contention import (
    ProgressDeadline, ProgressTimeout, cpu_contention_factor, describe_contention, scale_timeout)
from utils.paths import repo_path, src_path

pytestmark = [pytest.mark.sim, pytest.mark.slow]

#: The cargo line a missing binary is told to run — quoted verbatim from the root ``CLAUDE.md``.
_CARGO_BUILD = ("cargo build --release --bin sim_bridge --bin search_driver "
                "--manifest-path src/rust_sim/Cargo.toml")

#: No output at all for this long (contention-SCALED) means the child is wedged, not slow. Sized to
#: the longest silent stretch the run legitimately has: a cycle's SELECTION child falsifies every
#: loss trace of the newest eval cycle before anything is printed (measured 48 s over 9 traces on a
#: loaded box; 900 s is ~19x that). Training keeps stepping meanwhile — the selection is no longer
#: on the training step — but at this scale the run prints little else, so the bound still holds.
_IDLE_BUDGET_S = 900.0
#: Livelock backstop only — a child that chatters forever without converging keeps resetting the
#: idle bound. Deliberately generous; it is not the detector.
_TOTAL_BUDGET_S = 5400.0

#: THE PER-STEP COST BOUND, and why it is loose. Selection used to block ``_on_step`` for 30-350 s;
#: the two-phase cycle's worst step is a spawn plus ``model.save`` (measured 0.1-76 ms across a whole
#: run). 5 s is two orders of magnitude above the measurement and two orders BELOW the defect, so it
#: cannot flake on a starved box — a bound that can only be crossed by a regression, never by
#: contention, which is the only kind of duration assertion this project allows.
_STEP_BLOCK_BOUND_S = 5.0

_CYCLE_RE = re.compile(r"\[SearchTeacher\] cycle @ ([\d,]+): (\d+) candidates")
_STEP_BLOCK_RE = re.compile(r"\[SearchTeacher\] step-block ([a-z-]+): ([\d.]+) ms")
_COLLECT_RE = re.compile(r"\[SearchTeacher\] collected (\d+)/(\d+) corrections; status=(\{.*\})")


def _rust_binaries_or_fail() -> Tuple[str, str]:
    """The two built rust binaries this composition needs, or a FAILURE naming the build command.

    Deliberately does NOT build them: ``_resolve_rust_bin`` would shell out to cargo, which
    saturates every core and turns the contention-scaled bounds of every other test running beside
    it into a wall of timeouts (observed twice, both times misread as a rust defect — see
    ``designs/ops/testing.md``). A fresh worktree pays for the build ONCE, explicitly.
    """
    out = []
    for name, env_var in (("sim_bridge", "POKESIM_SIM_BRIDGE_BIN"),
                          ("search_driver", "POKESIM_SEARCH_DRIVER_BIN")):
        override = os.environ.get(env_var)
        path = Path(override) if override else src_path("rust_sim") / "target" / "release" / name
        if not path.is_file():
            pytest.fail(
                f"the rust `{name}` binary is not built at {path} — this test gates the "
                f"search-teacher composition ON RUST and cannot substitute node for it.\n"
                f"    Build it:  {_CARGO_BUILD}\n"
                f"    (or set ${env_var} to a pre-built binary.)")
        out.append(str(path))
    return out[0], out[1]


def _run_child(run_dir: Path, log_path: Path) -> subprocess.Popen:
    """Launch the smoke-scale training run. See the module docstring for every argv value."""
    argv = [
        sys.executable, str(repo_path("src", "main", "train_rl_agent.py")),
        "--debug", "--debug-eval",
        "--device", "cpu",
        "--use-bridge", "rust",
        "--steps", "30000",
        "--n-steps", "512", "--batch-size", "128", "--n-epochs", "2",
        "--run-dir", str(run_dir),
        "--eval-freq", "500", "--eval-games", "1", "--eval-battles", "1",
        "--search-teacher", "--search-teacher-coef", "0.5",
        "--teacher-search-freq", "1500",
        "--teacher-search-budget", "8",
        "--teacher-confirm-rollouts", "16",
        "--teacher-search-workers", "2",
        "--log-level", "periodic",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src_path()), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    # HIDE the GPU from the child entirely. `--device cpu` already keeps the learner off it, but a
    # real training run normally owns this box's card and a CUDA context we do not need is a context
    # that can OOM it. (The teacher workers do the same to themselves in `_spawn_worker`.)
    env["CUDA_VISIBLE_DEVICES"] = ""
    log = open(log_path, "wb")
    try:
        return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(repo_path()), env=env)
    except Exception:
        log.close()
        raise


def _wait_with_progress(proc: subprocess.Popen, log_path: Path) -> int:
    """Poll until the child exits, bounding the IDLE gap in its log rather than total duration."""
    deadline = ProgressDeadline(_IDLE_BUDGET_S, total_budget_s=_TOTAL_BUDGET_S,
                                what="search-teacher composition run")
    last_size = -1
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc
        size = log_path.stat().st_size if log_path.exists() else 0
        if size != last_size:
            last_size = size
            deadline.progress()
        try:
            deadline.check()
        except ProgressTimeout as e:
            proc.kill()
            proc.wait(timeout=60)
            raise AssertionError(
                f"INCONCLUSIVE (not a failed assertion): the training child produced no output "
                f"for the whole idle budget, so this run measured the BOX, not the composition. "
                f"{e} {describe_contention()}\n  log tail:\n"
                + _tail(log_path, 40)) from e
        time.sleep(2.0)


def _tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return "  <no log>"
    return "\n".join("  | " + ln for ln in lines[-n:])


def _markers_tail(path: Path) -> str:
    """Every ``[SearchTeacher]`` line, then the last 25 lines of the log.

    A bare tail of this child is useless: the final eval prints a full board dump, so 60 lines of
    tail is 60 lines of Skarmory. The marker lines ARE the story of the composition, so they are
    extracted whole and the tail is kept short enough to still show a crash.
    """
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return "  <no log>"
    markers = [ln for ln in lines if "[SearchTeacher]" in ln]
    out = ["  markers (" + str(len(markers)) + "):"] + ["  * " + m for m in markers]
    out += ["  last 25 lines:"] + ["  | " + ln for ln in lines[-25:]]
    return "\n".join(out)


def _parse_markers(log_text: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, Dict]]]:
    """``([(step, n_candidates)], [(n_collected, n_candidates, status)])`` from the cycle markers."""
    cycles = [(int(m.group(1).replace(",", "")), int(m.group(2)))
              for m in _CYCLE_RE.finditer(log_text)]
    collects = []
    for m in _COLLECT_RE.finditer(log_text):
        # the status dict is printed with python repr (single quotes) — json needs doubles
        try:
            status = json.loads(m.group(3).replace("'", '"'))
        except ValueError:
            status = {}
        collects.append((int(m.group(1)), int(m.group(2)), status))
    return cycles, collects


def _teacher_scalars(run_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Every ``teacher/*`` and ``grad/searchteacher_*`` scalar the run actually WROTE to its events.

    Read from the TB events, never from the child log's rendered table — the table drops the group
    prefix and undersamples under ``--log-level periodic`` (``main/ops/tb_read.py``).
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from main.ops.run_ref import event_dirs

    out: Dict[str, List[Tuple[int, float]]] = {}
    for d in event_dirs(run_dir):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            if tag.startswith("teacher/") or tag.startswith("grad/searchteacher"):
                out.setdefault(tag, []).extend((s.step, s.value) for s in ea.Scalars(tag))
    for tag in out:
        out[tag].sort()
    return out


def test_search_teacher_runs_multiple_cycles_end_to_end_on_rust(tmp_path):
    """>= 2 search-teacher cycles on ``--use-bridge rust``, with labels and the AWR loss term.

    Asserted on ARTIFACTS, never on "it did not crash": the per-cycle markers the callback emits,
    the worker's own status histogram, the worker config that proves the engine was rust, the
    TensorBoard scalars for both the cross-process facts (``teacher/corrections_per_cycle``) and
    the in-process fold (``teacher/loss`` + ``grad/searchteacher_share``), and "Training complete".
    """
    _rust_binaries_or_fail()
    factor = cpu_contention_factor(refresh=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = tmp_path / "child.log"

    started = time.time()
    proc = _run_child(run_dir, log_path)
    try:
        rc = _wait_with_progress(proc, log_path)
    finally:
        # A leaked training child would keep burning cores long after pytest moved on (an interrupt
        # or an assertion inside the wait both reach here).
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=60)
    elapsed = time.time() - started
    text = log_path.read_text(errors="replace")

    assert rc == 0, (f"the training child exited {rc} (contention factor {factor:.2f}); "
                     f"tail:\n{_markers_tail(log_path)}")
    assert "Training complete" in text, (
        "the run never reached 'Training complete' — the composition did not survive to the end.\n"
        + _markers_tail(log_path))

    # -- the engine was rust, at the seam that matters: the WORKER's own config ------------------
    cfgs = sorted((run_dir / "teacher_cycle").glob("config_*.json"))
    assert cfgs, ("no teacher_cycle/config_*.json — no cycle ever spawned a worker.\n"
                  + _markers_tail(log_path))
    impls = {json.loads(p.read_text()).get("impl") for p in cfgs}
    assert impls == {"rust"}, (
        f"the search-teacher workers ran on {impls}, not rust — a --use-bridge=rust run whose "
        f"teacher silently falls back to node is the exact gap this test exists to close.")

    # -- >= 2 DISTINCT cycles, each with candidates ---------------------------------------------
    cycles, collects = _parse_markers(text)
    steps = sorted({s for s, _ in cycles})
    assert len(steps) >= 2, (
        f"NO-CYCLE / ONE-CYCLE: the run launched {len(steps)} search-teacher cycle(s) "
        f"{steps} in {elapsed:.0f}s. This is a PRECONDITION FAILURE, not a pass: a run that "
        f"never reached a second cycle proves nothing about the cycle boundary. Check that eval "
        f"traces landed under {run_dir}/eval_traces and that selection found candidates.\n"
        + _markers_tail(log_path))
    assert all(n > 0 for _, n in cycles), (
        f"a cycle launched with 0 candidates: {cycles} — the callback should not spawn workers "
        f"for an empty selection.")

    # -- every launched cycle COLLECTED (no abort, no hung-cycle timeout) -------------------------
    assert "timed out — aborted" not in text, (
        "a search-teacher cycle hit the 3600 s hung-cycle watchdog and was aborted — the "
        "composition wedged at the worker.\n" + _markers_tail(log_path))
    assert "[SearchTeacher] selection failed" not in text, (
        "the SELECTION child failed (or died without writing a result). Since 2026-09-07 this also "
        "lands as `error:selection` in that cycle's status histogram, which the per-collect check "
        "below catches independently — both are asserted because a failure that reaches only one "
        "of the two is a reporting bug in its own right.\n" + _markers_tail(log_path))
    assert len(collects) >= 2, (
        f"{len(cycles)} cycles launched but only {len(collects)} COLLECTED — the cycle boundary is "
        f"the property under test, and a launch is not a boundary. Two causes look alike here and "
        f"the log tail tells them apart: (a) a worker wedged or died between launch and collect "
        f"(the composition seam), or (b) the run simply ENDED while a cycle was pending — nothing "
        f"waits for a per-cycle worker at _on_training_end, so --steps must leave room for the "
        f"collect, not just for the launch.\n" + _markers_tail(log_path))
    for n_ok, n_cand, status in collects:
        assert "worker_no_shard" not in status, (
            f"a worker died without writing its shard: status={status}. That is a crash in the "
            f"rust-backed search/confirm worker, which no leg-level test can see.")
        bad = sorted(k for k in status if k.startswith("error:"))
        assert not bad, (f"the worker reported hard errors {bad} in status={status} — the "
                         f"produce leg raised on the rust drivers.")

    # -- labels actually reached the buffer, and the AWR term actually folded ---------------------
    total = sum(n for n, _, _ in collects)
    assert total > 0, (
        f"every cycle produced ZERO corrections ({collects}). The composition ran but taught "
        f"nothing, so the loss half is untested — this is the seam, not a flake.\n"
        + _markers_tail(log_path))

    # -- the teacher never blocked the TRAINING STEP -------------------------------------------
    # This is the finding this gate produced on its first run, turned into an assertion: selection
    # ran inline in `_on_step` for 30-350 s per cycle while every document called the callback
    # non-blocking. The callback now times its own launch/collect paths.
    blocks = [(m.group(1), float(m.group(2))) for m in _STEP_BLOCK_RE.finditer(text)]
    assert blocks, (
        "no `[SearchTeacher] step-block` markers in a run that completed "
        f"{len(collects)} cycles — the per-step cost instrument is not wired, so the "
        "non-blocking claim is once again only a claim.\n" + _markers_tail(log_path))
    worst_label, worst_ms = max(blocks, key=lambda b: b[1])
    bound_ms = scale_timeout(_STEP_BLOCK_BOUND_S) * 1000.0
    assert worst_ms < bound_ms, (
        f"the teacher blocked one training step for {worst_ms:.0f} ms in the `{worst_label}` path "
        f"(bound {bound_ms:.0f} ms = {_STEP_BLOCK_BOUND_S:.0f} s x contention scale). Both phases "
        f"of a cycle are supposed to be children: `_on_step` may spawn, poll and read small JSON, "
        f"and nothing else. {describe_contention()}\n" + _markers_tail(log_path))

    tb = _teacher_scalars(run_dir)
    per_cycle = tb.get("teacher/corrections_per_cycle", [])
    assert len(per_cycle) >= 2, (
        f"teacher/corrections_per_cycle has {len(per_cycle)} point(s) in the TB events; the "
        f"callback records one per COLLECTED cycle, so <2 contradicts the {len(collects)} collect "
        f"markers in the log. Tags present: {sorted(tb)}")
    for tag in ("teacher/loss", "teacher/ce", "teacher/n", "grad/searchteacher_share"):
        assert tb.get(tag), (
            f"{tag}: NO SCALAR FOUND in {run_dir}/tb — absence is not a zero. The corrections "
            f"reached the buffer ({total} of them) but the AWR aux loss never folded, so "
            f"--search-teacher-coef bought nothing. Tags present: {sorted(tb)}")
    assert max(v for _, v in tb["teacher/n"]) > 0, (
        "teacher/n is 0 at every point — the AWR fold sampled an empty batch every time.")

    by_label: Dict[str, List[float]] = {}
    for label, ms in blocks:
        by_label.setdefault(label, []).append(ms)
    cost = "  ".join(f"{k} n={len(v)} max={max(v):.1f}ms" for k, v in sorted(by_label.items()))
    print(f"\n[search-teacher composition] {len(steps)} cycles at {steps}, "
          f"{total} corrections, {elapsed:.0f}s wall at contention factor {factor:.2f}"
          f"\n[search-teacher composition] training-step cost: {cost} "
          f"(total {sum(ms for _, ms in blocks) / 1000.0:.2f} s over the whole run)")
