"""X72's entropy floor reader (``main.x72.floor``): the marks, the floor and the comparison against the frozen
registration; ``clean`` against CE-3d's frozen script; the statistic, the two rows of a launch's last step and the
readiness of each mark on real TensorBoard event files. Nothing touches ``models/``."""
from __future__ import annotations

import ast
import json
import math
import random
import runpy
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from main.x72 import finals as F
from main.x72 import floor as FL
from main.x72.finals_test import DIR
from utils.paths import repo_path

RULE = DIR / "x72_rule.py"
DIAG9 = repo_path("designs/research_state/measurements/ce3d_magnet_screen_2026-11-11/record/diag9_ce3d.py")
FORK = 300_061_272               # a banked swap final's step: where a lineage's own points begin
GAP = 100_000                    # steps between two points (20 points ~ 2M steps)
P = FL.Point


@pytest.fixture(scope="module")
def rule():
    return runpy.run_path(str(RULE))


def f32(x: float) -> float:
    """What a TensorBoard ``simple_value`` holds of ``x``."""
    return float(np.float32(x))


def write_tb(run: Path, rows, name: str) -> None:
    """``rows`` = [(step, wall_time, entropy)], written as a real writer writes ``train/entropy_loss`` (the NEGATED
    entropy, ``simple_value``), in the order given."""
    from tensorboard.compat.proto import event_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter

    tb = run / "tb"
    tb.mkdir(parents=True, exist_ok=True)
    with open(tb / name, "wb") as fh:
        w = RecordWriter(fh)
        w.write(event_pb2.Event(wall_time=0.0, file_version="brain.Event:2").SerializeToString())
        for step, wall, ent in rows:
            ev = event_pb2.Event(wall_time=float(wall), step=int(step))
            ev.summary.value.add(tag=FL.TAG, simple_value=-float(ent))
            w.write(ev.SerializeToString())


def make_run(models: Path, seed: int, rows, *, boundary: bool = True) -> Path:
    """A lineage's run dir as the trainer leaves a fork: the parent's curve (steps <= FORK, entropy 0.01: far
    below the floor) in the inherited file, the lineage's own ``rows`` beside it, the boundary in the record."""
    run = models / F.run_name(seed)
    write_tb(run, [(FORK - (60 - i) * GAP, 1.0e9 + i, 0.01) for i in range(61)],
             "events.out.tfevents.0000000000.inherited.0.0")
    write_tb(run, rows, "events.out.tfevents.1800000000.host.1.0")
    (run / "tb" / "INHERITED_FROM.json").write_text(json.dumps({"marker": "gen3_tb_inherit_v1", "fork_step": FORK}))
    lineage = {"schema": 1, "role": "fork", "fork_parent": {"run_dir": str(models / "parent")}}
    if boundary:
        lineage["fork_step"] = FORK
    (run / "metadata.json").write_text(json.dumps({"lineage": lineage}))
    return run


def line(first: int, n: int, ent, *, wall: float = 1.8e9):
    """``n`` points from step ``first``, GAP apart, a second of wall time apart; ``ent`` a number or f(i)."""
    return [(first + i * GAP, wall + i, ent(i) if callable(ent) else ent) for i in range(n)]


@pytest.fixture()
def models(tmp_path):
    m = tmp_path / "models"
    m.mkdir()
    return m


# ---------------------------------------------------------------------------------------- the registered tables
def test_the_marks_the_floor_and_the_band_are_the_registrations(rule):
    text = " ".join((DIR / "PREREG.md").read_text().split())
    assert ("**The statistic:** the mean of a run's last 20 points of −`train/entropy_loss` at or before a MARK"
            in text and FL.POINTS == 20 and FL.TAG == "train/entropy_loss")
    assert ("the steps round start + 5M, + 10M, + 15M (round start = the previous target; 300,000,170 in round 1) "
            "and the round's end") in text
    assert FL.MARKS == ("5M", "10M", "15M", "end")
    assert [FL.mark_step(1, m) for m in FL.MARKS] == [305_000_170, 310_000_170, 315_000_170, None]
    assert [FL.mark_step(k, "5M") for k in (2, 3, 4)] == [325_000_170, 345_000_170, 365_000_170]
    assert FL.mark_step(4, "15M") == 375_000_170 < F.target(4) == 380_000_170
    for k, mark in ((0, "5M"), (5, "5M"), (1, "20M"), (1, "END")):
        with pytest.raises((FL.FloorError, F.FinalsError)):
            FL.mark_step(k, mark)
    assert (FL.FLOOR_FRACTION, FL.ENT_EPS_NATS) == (rule["FLOOR_FRACTION"], rule["ENT_EPS_NATS"]) == (0.5, 1e-9)
    assert "0.5 × 0.4837687760591507 = 0.24188438802957535 nats" in text
    assert FL.floor_nats() == rule["floor_nats"]() == 0.24188438802957535
    assert repo_path(FL.CE3D_RECORD) == rule["CE3D_RECORD"] / "diag9_ce3d.json" and RULE == repo_path(FL.RULE_FILE)
    assert "`x72_rule.py floor <e1> [<e2> <e3>]`" in text


def test_clean_is_ce3ds_frozen_clean():
    """The frozen script reads ``models/`` at import, so its ``clean`` is taken out of its source and run."""
    tree = ast.parse(DIAG9.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "clean")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(DIAG9), "exec"), ns)  # noqa: S102
    rng = random.Random(72)
    for _ in range(300):
        n = rng.randrange(0, 40)
        walls = rng.sample(range(10_000), n)
        pts = [P(rng.randrange(1, 12) * GAP, float(w), rng.random()) for w in walls]
        rng.shuffle(pts)
        assert FL.clean(pts) == ns["clean"](pts)
    assert FL.clean([]) == ([], 0)


def test_of_two_rows_at_one_step_the_later_written_is_kept():
    """A launch's last step: update k-1's row, then update k's (designs/training/telemetry_scalars.md)."""
    pts = [P(100, 1.0, -0.50), P(200, 2.0, -0.40), P(200, 3.0, -0.30), P(300, 4.0, -0.20)]
    assert FL.clean(pts) == ([pts[0], pts[2], pts[3]], 1)
    assert FL.clean(list(reversed(pts))) == ([pts[0], pts[2], pts[3]], 1)            # by wall time, not input order


def test_an_abandoned_attempts_points_past_the_resume_step_are_dropped():
    first = [P(s, float(s), -0.9) for s in (100, 200, 300, 400)]          # the attempt that died after 400
    again = [P(s, 1000.0 + s, -0.5) for s in (300, 400, 500)]             # resumed from a checkpoint at 200
    got, dropped = FL.clean(first + again)
    assert got == first[:2] + again and dropped == 2


def test_the_statistic_is_the_mean_of_the_last_twenty_points_at_or_before_the_step():
    pts = [P(1000 + i, float(i), -(1.0 + i / 7.0)) for i in range(30)]
    got = FL.statistic(pts, 1024)                                          # points 5..24: the step itself counts
    want = [1.0 + i / 7.0 for i in range(5, 25)]
    assert got["stat_nats"] == statistics.mean(want) and (got["first_step"], got["last_step"]) == (1005, 1024)
    assert (got["points"], got["min_nats"], got["max_nats"]) == (20, want[0], want[-1])
    assert FL.statistic(pts, 10**9)["stat_nats"] == statistics.mean([1.0 + i / 7.0 for i in range(10, 30)])
    with pytest.raises(FL.FloorError, match="19 own point"):
        FL.statistic(pts, 1018)
    with pytest.raises(FL.FloorError, match="finite"):
        FL.statistic([P(i, float(i), 0.3) for i in range(20)], 100)       # a NEGATIVE entropy is not a statistic


# -------------------------------------------------------------------------------- a mark on real event files
def test_a_mid_round_mark_is_read_once_passed_and_is_a_function_of_the_mark(models):
    mark = FL.mark_step(1, "5M")
    first = mark - 30 * GAP                                 # 31 points end exactly AT the mark
    ents = [0.60 - 0.001 * i for i in range(40)]
    run = make_run(models, 7341, line(first, 31, lambda i: ents[i]))
    got = FL.read_run(run, 1, "5M")
    assert got["status"] == "NOT-READY" and "has not passed the mark" in got["reason"], got   # AT is not PAST
    run = make_run(models, 7341, line(first, 32, lambda i: ents[i]))
    got = FL.read_run(run, 1, "5M")
    want = statistics.mean([f32(-f32(-e)) for e in ents[11:31]])
    assert (got["status"], got["stat_nats"], got["mark_step"]) == ("READ", want, mark)
    assert (got["first_step"], got["last_step"], got["own_points"], got["superseded"]) == (
        mark - 19 * GAP, mark, 32, 0)
    later = FL.read_run(make_run(models, 7341, line(first, 40, lambda i: ents[i])), 1, "5M")
    assert later["stat_nats"] == want and later["own_points"] == 40                 # the same mark, read later


def test_the_parents_points_never_fill_in_for_the_runs_own(models):
    """19 own points at or before the mark: a refusal, although the inherited prefix holds 61 more."""
    mark = FL.mark_step(1, "5M")
    run = make_run(models, 7341, line(mark - 18 * GAP, 21, 0.5))
    with pytest.raises(FL.FloorError, match="19 own point"):
        FL.read_run(run, 1, "5M")
    run = make_run(models, 7342, line(mark - 19 * GAP, 21, 0.5))
    assert FL.read_run(run, 1, "5M")["stat_nats"] == f32(0.5)


def test_an_unknown_fork_boundary_is_a_refusal(models, capsys):
    run = make_run(models, 7341, line(FL.mark_step(1, "5M") - 25 * GAP, 30, 0.5), boundary=False)
    with pytest.raises(FL.FloorError, match="fork boundary"):
        FL.read_run(run, 1, "5M")
    assert FL.main(["read", "--round", "1", "--mark", "5M", "--seed", "7341", "--models-dir", str(models)]) == 3
    assert "REFUSED" in capsys.readouterr().err


def _round_end(n_before: int = 30, *, final_rows: int = 2, k: int = 1):
    """A round's own curve to its end: ``n_before`` points at 0.5, then the launch's last step (61,000 past the
    target) with update k-1's row (entropy 0.1) and, a minute later, update k's (0.3)."""
    end = F.target(k) + 61_000
    rows = line(end - n_before * GAP, n_before, 0.5)
    rows += [(end, 1.8e9 + n_before, 0.1), (end, 1.8e9 + n_before + 60, 0.3)][:final_rows]
    return end, rows


def test_the_round_end_keeps_the_final_updates_row_and_waits_for_it(models):
    end, rows = _round_end(final_rows=1)
    got = FL.read_run(make_run(models, 7341, rows), 1, "end")
    assert got["status"] == "NOT-READY" and "holds 1 row(s); a launch's last step holds 2" in got["reason"], got
    end, rows = _round_end()
    got = FL.read_run(make_run(models, 7341, rows), 1, "end")
    want = statistics.mean([f32(0.5)] * 19 + [f32(0.3)])              # update k's row; update k-1's 0.1 is dropped
    assert (got["status"], got["stat_nats"], got["mark_step"], got["last_step"]) == ("READ", want, end, end)
    assert (got["rows_at_mark_step"], got["superseded"], got["min_nats"]) == (2, 1, f32(0.3))
    # round 2 has started in the same run dir: round 1's end is the same number
    more = rows + line(end + GAP, 10, 0.2, wall=1.9e9)
    assert FL.read_run(make_run(models, 7341, more), 1, "end")["stat_nats"] == want
    # and round 2's end is not there yet
    got2 = FL.read_run(make_run(models, 7341, more), 2, "end")
    assert got2["status"] == "NOT-READY" and "round 2 has not ended" in got2["reason"]


def test_a_run_that_has_not_reached_the_target_has_no_round_end(models):
    rows = line(F.target(1) - 40 * GAP, 40, 0.5)                     # the last point is GAP short of the target
    got = FL.read_run(make_run(models, 7341, rows), 1, "end")
    assert got["status"] == "NOT-READY" and "has not ended" in got["reason"]
    assert FL.read_run(models / "nowhere", 1, "end")["status"] == "NOT-READY"


def test_reading_changes_nothing_in_the_run_dir(models):
    _end, rows = _round_end(n_before=80)
    run = make_run(models, 7341, rows)
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(run.rglob("*")) if p.is_file()}
    assert [FL.read_run(run, 1, m)["status"] for m in ("15M", "end")] == ["READ", "READ"]
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(run.rglob("*")) if p.is_file()} == before


# ------------------------------------------------------------------------------------------- the comparison
def test_the_comparison_is_the_frozen_rules_at_and_around_the_boundary(rule):
    bar = FL.floor_nats()
    eps = FL.ENT_EPS_NATS
    xs = [0.0, bar / 2, bar - 1e-6, bar - 2 * eps, bar - 1.5 * eps, bar - eps, bar - 0.5 * eps, bar,
          bar + 0.5 * eps, bar + eps, bar + 1.5 * eps, bar + 1e-6, 0.48, 1.01]
    xs += [math.nextafter(x, s) for x in (bar - eps, bar + eps, bar) for s in (0.0, 1.0)]
    triples = [(x, y, z) for x in xs for y in (xs[0], xs[7], xs[-1]) for z in (xs[3], xs[12])] + [(x,) for x in xs]
    for stats in triples:
        want = rule["floor"](list(stats))
        got = FL.compare(list(zip(F.SEEDS, stats)))
        seeds = lambda idx: [F.SEEDS[i - 1] for i in idx]  # noqa: E731
        assert (got["fired"], got["below_floor"], got["within_1e-9_of_the_floor"], got["floor_nats"]) == (
            want["fired"], seeds(want["below_floor"]), seeds(want["within_1e-9_of_the_floor"]), want["floor_nats"])
    assert FL.compare([(7341, bar - 1.5 * eps)])["fired"] is True
    near = FL.compare([(7341, bar - 0.5 * eps), (7342, bar + 0.5 * eps)])
    assert near["fired"] is False and near["within_1e-9_of_the_floor"] == [7341, 7342]


def _near_floor_rows(mark: int):
    """21 points whose last 20 float32 entropies average within 1e-9 nats of the floor, and BELOW it."""
    bar = FL.floor_nats()
    lo = np.float32(bar)
    lo = float(lo if float(lo) <= bar else np.nextafter(lo, np.float32(0.0)))      # the float32 at or below the floor
    hi = float(np.nextafter(np.float32(lo), np.float32(1.0)))
    assert lo <= bar < hi
    for m in range(21):
        vals = [hi] * m + [lo] * (20 - m)
        mean = statistics.mean(vals)
        if abs(mean - bar) <= 0.9 * FL.ENT_EPS_NATS and mean < bar:
            return [(mark - (19 - i) * GAP, 1.8e9 + i, v) for i, v in enumerate(vals)] + [(mark + GAP, 1.8e9 + 99, 0.9)]
    raise AssertionError("no float32 mix lands in the band")


def test_read_compares_the_lineages_and_hands_the_frozen_cli_the_same_numbers(models, rule, tmp_path):
    mark = FL.mark_step(2, "10M")
    make_run(models, 7341, line(mark - 25 * GAP, 30, 0.45))
    make_run(models, 7342, _near_floor_rows(mark))                    # within 1e-9 below the floor: no fire
    make_run(models, 7343, line(mark - 25 * GAP, 30, 0.30))
    rec = FL.read(2, "10M", models_dir=models)
    assert (rec["outcome"], rec["fired"], rec["below_floor"], rec["within_1e-9_of_the_floor"]) == (
        "NOT-FIRED", False, [], [7342])
    assert rec["read"] == [7341, 7342, 7343] and rec["not_ready"] == [] and FL.exit_code(rec) == 0
    assert rec["lineages"][1]["stat_nats"] < rec["floor_nats"]
    argv = rec["rule_command"].split()
    assert argv[:3] == ["python", FL.RULE_FILE, "floor"] and [float(x) for x in argv[3:]] == [
        x["stat_nats"] for x in rec["lineages"]]
    out = subprocess.run([sys.executable, str(RULE), *argv[2:]], capture_output=True, text=True, check=True)
    frozen = json.loads(out.stdout)
    assert (frozen["fired"], frozen["within_1e-9_of_the_floor"], frozen["stats_nats"]) == (
        False, [2], [x["stat_nats"] for x in rec["lineages"]])
    # one lineage below the floor by more than the band: FIRED, whatever the others are
    make_run(models, 7343, line(mark - 25 * GAP, 30, 0.2418))
    rec = FL.read(2, "10M", models_dir=models)
    assert (rec["outcome"], rec["below_floor"], FL.exit_code(rec)) == ("FIRED", [7343], 10)
    assert rule["floor"]([x["stat_nats"] for x in rec["lineages"]])["below_floor"] == [3]
    make_run(models, 7341, line(mark - 25 * GAP, 20, 0.45))           # lineage 1 has not passed the mark
    rec = FL.read(2, "10M", models_dir=models)
    assert (rec["outcome"], rec["not_ready"], rec["read"], FL.exit_code(rec)) == ("FIRED", [7341], [7342, 7343], 10)
    # nothing fired and a lineage is not ready: NOT-READY, the read ones still reported
    rec = FL.read(2, "10M", models_dir=models, seeds=[7342, 7341])
    assert (rec["outcome"], rec["read"], rec["not_ready"], rec["fired"], FL.exit_code(rec)) == (
        "NOT-READY", [7342], [7341], False, 5)
    assert [x["seed"] for x in rec["lineages"]] == [7341, 7342]        # lineage order, whatever was typed
    with pytest.raises(FL.FloorError, match="not X72 lineages"):
        FL.read(2, "10M", models_dir=models, seeds=[7321])


def test_the_cli_prints_the_record_writes_it_and_exits_by_outcome(models, tmp_path, capsys):
    mark = FL.mark_step(1, "15M")
    make_run(models, 7341, line(mark - 25 * GAP, 30, 0.45))
    make_run(models, 7342, line(mark - 25 * GAP, 30, 0.10))
    base = ["read", "--round", "1", "--mark", "15M", "--models-dir", str(models)]
    out = tmp_path / "rec" / "floor.json"
    assert FL.main([*base, "--seed", "7341", "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec == json.loads(capsys.readouterr().out) and rec["outcome"] == "NOT-FIRED" and rec["read"] == [7341]
    assert [p.name for p in out.parent.iterdir()] == ["floor.json"]
    assert FL.main([*base, "--seed", "7341", "--seed", "7342"]) == 10
    assert FL.main([*base, "--seed", "7343"]) == 5                    # no run dir yet
    assert FL.main([*base]) == 10
    assert FL.main(["read", "--round", "5", "--mark", "5M", "--models-dir", str(models)]) == 3
    assert FL.main([*base[:5], "--models-dir", str(models / "nowhere")]) == 3
