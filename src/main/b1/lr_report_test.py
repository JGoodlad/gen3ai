"""BOUNDARY-1's learning-rate report: the windows, the rung, the ratios, the groups and the refusals
(``main.b1.lr_report``), on synthetic event files and tiny state dicts."""
import io
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from main.b1 import arms as A
from main.b1 import lr_report as R

L50 = R.LOOKS["50M"]
#: One run an arm, plus a second control run: five runs, two degrees of freedom.
FIVE: Tuple[Tuple[str, int], ...] = (("c", 5101), ("c", 5102), ("w", 5103), ("br", 5105), ("wb", 5107))


def _f32(x: float) -> float:
    """``x`` as the event file stores it (a float32)."""
    return float(struct.unpack("f", struct.pack("f", x))[0])


def _rung_lr(k: int) -> float:
    return R.LADDER_BASE * R.LADDER_FACTOR ** k


def write_events(run: Path, rows: Sequence[Tuple[int, Optional[float], Optional[float]]], stamp: int) -> None:
    """One event file of ``(step, lr, kl)`` rows (``None``: that tag has no point there), named with ``stamp`` as
    its timestamp, so the order of a run's files is the test's."""
    from torch.utils.tensorboard import SummaryWriter

    tmp = run / "tb" / f"_w{stamp}"
    w = SummaryWriter(log_dir=str(tmp))
    for step, lr, kl in rows:
        if lr is not None:
            w.add_scalar(R.LR_TAG, lr, step)
        if kl is not None:
            w.add_scalar(R.KL_TAG, kl, step)
    w.add_scalar("train/clip_fraction", 0.1, rows[0][0])                     # a tag the report never reads
    w.close()
    (written,) = list(tmp.iterdir())
    written.rename(run / "tb" / f"events.out.tfevents.{stamp:010d}.host.1.0")
    tmp.rmdir()


def steps_to(last: int, every: int = 5_000_000, first: int = 5_000_000) -> List[int]:
    return [*range(first, last, every), last]


def make_run(models: Path, arm: str, seed: int, *, lr: float = _rung_lr(-3), kl: float = 0.02,
             last: int = L50 + 50_000) -> Path:
    """A run with a flat rate and a flat KL, rows every 5M steps up to ``last``."""
    run = models / A.run_name(arm, seed)
    write_events(run, [(s, lr, kl) for s in steps_to(last)], stamp=1)
    return run


# ------------------------------------------------------------------------------------------------ small statistics
def test_the_geometric_mean_is_the_exponential_of_the_mean_log():
    assert R.geomean([1e-4, 4e-4]) == pytest.approx(2e-4, rel=1e-12)
    assert R.geomean([2.0, 2.0, 2.0]) == pytest.approx(2.0, rel=1e-12)
    assert R.geomean([1.0, 10.0, 100.0]) == pytest.approx(10.0, rel=1e-12)
    assert R.geomean([1.0, 10.0, 100.0]) < (1.0 + 10.0 + 100.0) / 3, "not the arithmetic mean"
    for bad in ([], [1e-4, 0.0], [1e-4, -1e-4], [1e-4, math.inf], [math.nan]):
        with pytest.raises(R.ReportError):
            R.geomean(bad)


@pytest.mark.parametrize("k", range(-18, 5))
def test_a_stored_rung_is_that_rung(k):
    """What the event file holds is the float32 of a rate the controller reached by repeated multiplication: within
    float error of the rung, on either side of it, and decided as the rung every time (rule 8)."""
    walked = R.LADDER_BASE
    for _ in range(abs(k)):
        walked = walked * R.LADDER_FACTOR if k > 0 else walked / R.LADDER_FACTOR
    for lr in (_rung_lr(k), _f32(_rung_lr(k)), walked, _f32(walked)):
        got, exact = R.rung(lr)
        assert got == k and abs(exact - k) < 1e-5


def test_a_rung_within_float_error_on_the_low_side_is_not_floored():
    """The real case: the nine runs' stored 1.7361e-4 reads k = -2.9999998..., which truncation would call -2."""
    lr = 0.00017361111531499773
    exact = math.log(lr / R.LADDER_BASE) / math.log(R.LADDER_FACTOR)
    assert exact > -3.0 and int(exact) == -2
    assert R.rung(lr) == (-3, exact)
    below = _f32(_rung_lr(-8))
    assert math.log(below / R.LADDER_BASE) / math.log(R.LADDER_FACTOR) < -8.0, "and one stored just BELOW its rung"
    assert R.rung(below)[0] == -8


def test_a_rate_off_the_ladder_has_no_rung_and_one_in_between_is_refused():
    assert R.rung(1e-5)[0] is None, "the controller's floor is not a rung"
    assert R.rung(6e-4)[0] is None, "nor its ceiling"
    k, exact = R.rung(_rung_lr(-3) * 1.05)
    assert k is None and -3.0 < exact < -2.0
    assert R.rung(_rung_lr(-3) * (1 + 2 * R.RUNG_OFF_REL))[0] is None
    assert R.rung(_rung_lr(-3) * (1 + R.RUNG_ON_REL / 2))[0] == -3
    assert R.RUNG_ON_REL * 10 <= R.RUNG_OFF_REL, "the two verdicts are separated by a band, not by one threshold"
    for rel in (3 * R.RUNG_ON_REL, R.RUNG_OFF_REL / 3):
        for sign in (1, -1):
            with pytest.raises(R.ReportError, match="not decided"):
                R.rung(_rung_lr(-3) * (1 + sign * rel))
    for bad in (0.0, -1e-4, math.inf, math.nan):
        with pytest.raises(R.ReportError):
            R.rung(bad)


def test_factor_levels_follow_the_cross():
    assert [R.factor_levels(a) for a in ("c", "w", "br", "wb")] == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_a_ratio_is_the_factor_with_the_other_matched():
    """An exactly multiplicative table gives each factor's ratio back with a zero-width interval, on an UNBALANCED
    set of runs (a pooled wide / narrow mean would mix breadth in)."""
    arms = ["c", "c", "c", "w", "br", "wb", "wb"]
    vals = [3.0 * 0.5 ** w * 1.3 ** b for w, b in map(R.factor_levels, arms)]
    got = R.factor_ratios(arms, vals)
    for factor, want in (("width", 0.5), ("breadth", 1.3)):
        r = got[factor]
        assert r["ratio"] == pytest.approx(want, rel=1e-9) and r["df"] == 4
        assert r["lo"] == pytest.approx(want, rel=1e-6) and r["hi"] == pytest.approx(want, rel=1e-6)


def test_the_interval_is_students_t_on_runs_minus_three():
    """Two runs a cell: the width coefficient is the difference of the mean logs, its standard error
    sqrt(s2 / 2) with s2 = RSS / 5, and the half-width t(0.95, 5) of them."""
    arms = ["c", "c", "w", "w", "br", "br", "wb", "wb"]
    logs = [0.00, 0.10, -0.70, -0.74, 0.20, 0.16, -0.50, -0.62]
    got = R.factor_ratios(arms, [math.exp(v) for v in logs])["width"]
    wide = [v for a, v in zip(arms, logs) if a in ("w", "wb")]
    narrow = [v for a, v in zip(arms, logs) if a in ("c", "br")]
    b = sum(wide) / 4 - sum(narrow) / 4
    assert got["ratio"] == pytest.approx(math.exp(b), rel=1e-9) and got["df"] == 5
    assert got["lo"] < got["ratio"] < got["hi"]
    half = math.log(got["hi"] / got["ratio"])
    assert half == pytest.approx(math.log(got["ratio"] / got["lo"]), rel=1e-9), "symmetric on the log"
    import numpy as np
    x = np.array([[1.0, *R.factor_levels(a)] for a in arms])
    beta = np.linalg.solve(x.T @ x, x.T @ np.array(logs))
    rss = float(((np.array(logs) - x @ beta) ** 2).sum())
    assert half == pytest.approx(2.015048 * math.sqrt(rss / 5 / 2), rel=1e-5)


def test_ratios_refuse_what_cannot_carry_an_interval():
    with pytest.raises(R.ReportError, match="cannot separate"):
        R.factor_ratios(["c", "w", "br"], [1.0, 2.0, 3.0])                    # no degree of freedom left
    with pytest.raises(R.ReportError, match="cannot separate"):
        R.factor_ratios(["c", "c", "w", "w", "c"], [1.0, 2.0, 3.0, 4.0, 5.0])  # breadth has one level
    with pytest.raises(R.ReportError, match="positive"):
        R.factor_ratios(["c", "w", "br", "wb", "c"], [1.0, 2.0, 0.0, 4.0, 5.0])
    with pytest.raises(R.ReportError):
        R.factor_ratios(["c", "w"], [1.0])


# ------------------------------------------------------------------------------------------------ the runs
def test_a_run_is_named_by_its_directory(tmp_path):
    assert R.arm_of(tmp_path / A.run_name("wb", 5113)) == ("wb", 5113)
    assert R.arm_of(tmp_path / A.run_name("br", A.SPARE_SEED)) == ("br", A.SPARE_SEED)
    for bad in ("rb_b1c_s5103", "rb_x26", "copy_of_" + A.run_name("c", 5101)):
        with pytest.raises(R.ReportError, match="not a BOUNDARY-1 run name"):
            R.arm_of(tmp_path / bad)
    assert [p.name for p in R.default_runs(tmp_path)] == [A.run_name(a, s) for a, s in A.LAUNCH_ORDER]
    assert len(R.default_runs(tmp_path)) == 12


def test_the_windows_are_the_second_half_up_to_the_first_row_at_the_look(tmp_path):
    """Rows at L/2 - 1 (out), L/2 (in), ..., the first row at or after L (in, and the look's rate), a later row (out)."""
    run = tmp_path / A.run_name("w", 5103)
    half = L50 // 2
    rows = [(1_000_000, 3e-4, 0.50), (half - 1, 3e-4, 0.25), (half, 2e-4, 0.04), (40_000_000, 1e-4, 0.01),
            (L50 + 7, _rung_lr(-7), 0.02), (L50 + 900_000, _rung_lr(-9), 9.0)]
    write_events(run, rows, stamp=1)
    r = R.run_row(run, L50)
    assert (r["run"], r["arm"], r["seed"]) == (run.name, "w", 5103)
    assert (r["rows_to_look"], r["rows_second_half"]) == (5, 3)
    assert (r["second_half_first_step"], r["look_row_step"]) == (half, L50 + 7)
    lrs = [_f32(2e-4), _f32(1e-4), _f32(_rung_lr(-7))]
    kls = [_f32(0.04), _f32(0.01), _f32(0.02)]
    assert r["lr_geomean_second_half"] == pytest.approx(math.exp(sum(map(math.log, lrs)) / 3), rel=1e-12)
    assert r["lr_at_look"] == lrs[-1] and r["rung"] == -7
    assert r["kl_sum_second_half"] == pytest.approx(sum(kls), rel=1e-12)
    assert r["kl_sum_to_look"] == pytest.approx(_f32(0.50) + _f32(0.25) + sum(kls), rel=1e-12)
    assert r["sensitivity_median_second_half"] == pytest.approx(sorted(k / v ** 2 for k, v in zip(kls, lrs))[1],
                                                                rel=1e-12)
    assert (r["superseded_rows"], r["rows_lr_only"], r["rows_kl_only"]) == (0, 0, 0)


def test_a_later_file_supersedes_the_rows_a_resume_discarded(tmp_path):
    """The second process resumed from a checkpoint BEFORE the first one's last two updates: those rows are gone,
    whatever order the files are listed in, and a row with one tag alone is counted and not used."""
    run = tmp_path / A.run_name("c", 5101)
    write_events(run, [(10_000_000, 3e-4, 0.01), (20_000_000, 3e-4, 0.02), (30_000_000, 9e-4, 9.0),
                       (30_100_000, 9e-4, 9.0)], stamp=5)
    write_events(run, [(30_050_000, 2e-4, 0.03), (40_000_000, 2e-4, 0.04), (45_000_000, None, 0.5),
                       (L50 + 1, _rung_lr(-3), 0.05)], stamp=30)
    s = R.read_series(run)
    assert s["steps"] == [10_000_000, 20_000_000, 30_000_000, 30_050_000, 40_000_000, L50 + 1]
    assert s["superseded_rows"] == 2, "30.1M's rate and its KL; 30.0M is before the resume and stays"
    assert (s["rows_lr_only"], s["rows_kl_only"]) == (0, 1)
    assert [f.split(".")[3] for f in s["files"]] == ["0000000005", "0000000030"]
    r = R.run_row(run, L50)
    assert r["kl_sum_to_look"] == pytest.approx(sum(map(_f32, (0.01, 0.02, 9.0, 0.03, 0.04, 0.05))), rel=1e-12)
    assert r["rows_second_half"] == 4 and r["rung"] == -3


def test_a_run_short_of_the_look_step_refuses_the_whole_report_by_name(tmp_path, capsys):
    models = tmp_path / "models"
    runs = [make_run(models, a, s) for a, s in FIVE]
    short = make_run(models, "wb", 5108, last=L50 - 232_560)                 # still training
    late = models / A.run_name("br", 5106)                                   # a gap over the look step
    write_events(late, [(s, 2e-4, 0.02) for s in (30_000_000, 49_000_000, L50 + A.CHECKPOINT_EVERY)], stamp=1)
    absent = models / A.run_name("w", 5111)
    assert R.report(runs, "50M")["n_runs"] == 5

    with pytest.raises(R.ReportError, match=f"{short.name} lacks the look step 50,000,000") as e:
        R.report([*runs, short], "50M")
    assert "its last row is at 49,767,440" in str(e.value)
    with pytest.raises(R.ReportError, match=f"{late.name} lacks the look step .* late"):
        R.report([*runs, late], "50M")
    with pytest.raises(R.ReportError, match=f"{absent.name}: no run directory"):
        R.report([*runs, absent], "50M")

    out = tmp_path / "out" / "r.json"
    rc = R.main(["--look", "50M", "--runs", *map(str, [*runs, short, late, absent]), "--out", str(out)])
    err = capsys.readouterr().err
    assert rc == 3 and err.startswith("REFUSED: 3 of 8 runs")
    assert all(n in err for n in (short.name, late.name, absent.name)), "every short run is named, none dropped"
    assert not out.exists()
    assert R.main(["--look", "100M", "--runs", *map(str, runs)]) == 3, "the same runs have not reached 100M"
    assert f"{runs[0].name} lacks the look step 100,000,000" in capsys.readouterr().err


def test_the_default_is_the_twelve_registered_runs_and_none_is_dropped(tmp_path, capsys):
    models = tmp_path / "models"
    for arm, seed in A.LAUNCH_ORDER[:-1]:
        make_run(models, arm, seed)
    assert R.main(["--look", "50M", "--models-dir", str(models)]) == 3
    err = capsys.readouterr().err
    assert "1 of 12 runs" in err and A.run_name(*A.LAUNCH_ORDER[-1]) in err
    make_run(models, *A.LAUNCH_ORDER[-1])
    assert R.main(["--look", "50M", "--models-dir", str(models)]) == 0
    assert "12 runs" in capsys.readouterr().out
    assert R.main(["--look", "50M", "--models-dir", str(models), "--runs", str(models / "x")]) == 3
    twice = [str(models / A.run_name("c", 5101))] * 2
    assert R.main(["--look", "50M", "--runs", *twice]) == 3 and "named twice" in capsys.readouterr().err


def test_out_is_refused_inside_a_models_directory(tmp_path, monkeypatch, capsys):
    models = tmp_path / "models"
    runs = [make_run(models, a, s) for a, s in FIVE]
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(archive))
    argv = ["--look", "50M", "--runs", *map(str, runs), "--out"]
    for bad in (models / "r.json",                       # beside the runs
                models / "reports" / "r.json",           # under the directory that holds them
                runs[0] / "lr_report.json",              # in a run directory
                models / ".." / "models" / "r.json",     # the same place, spelled another way
                archive / "r.json", archive / "sub" / "r.json"):   # the main checkout's archive
        assert R.main([*argv, str(bad)]) == 3, bad
        assert "is written OUTSIDE models/" in capsys.readouterr().err
        assert not bad.exists()
    assert sorted(p.name for p in models.iterdir()) == sorted(r.name for r in runs), "nothing was created there"
    assert list(archive.iterdir()) == []
    link = tmp_path / "elsewhere"
    link.symlink_to(models, target_is_directory=True)
    assert R.main([*argv, str(link / "r.json")]) == 3, "a symlink into models is still models"
    capsys.readouterr()

    good = tmp_path / "out" / "nested" / "r.json"
    assert R.main([*argv, str(good)]) == 0
    doc = json.loads(good.read_text())
    assert doc["schema"] == R.SCHEMA and doc["n_runs"] == 5 and doc["look_step"] == L50
    assert [r["run"] for r in doc["runs"]] == [r.name for r in runs]
    assert set(doc["factors"]) == {"width", "breadth"}
    assert set(doc["factors"]["width"]) == set(R.QUANTITIES)
    assert "displacement" not in doc


def test_the_report_reads_the_ratio_off_the_runs(tmp_path, capsys):
    """Wide runs at rung -7 and narrow at -3, the KL the same: the rate's ratio is 1.2 ** -4, the KL's is 1, and
    the sensitivity's is 1.2 ** 8."""
    models = tmp_path / "models"
    runs = [make_run(models, a, s, lr=_rung_lr(-7 if a in ("w", "wb") else -3)) for a, s in FIVE]
    doc = R.report(runs, "50M")
    w = doc["factors"]["width"]
    assert w["lr_geomean_second_half"]["ratio"] == pytest.approx(1.2 ** -4, rel=1e-6)
    assert w["lr_at_look"]["ratio"] == pytest.approx(1.2 ** -4, rel=1e-6)
    assert w["kl_sum_second_half"]["ratio"] == pytest.approx(1.0, rel=1e-9)
    assert w["kl_sum_to_look"]["ratio"] == pytest.approx(1.0, rel=1e-9)
    assert w["sensitivity_median_second_half"]["ratio"] == pytest.approx(1.2 ** 8, rel=1e-6)
    assert doc["factors"]["breadth"]["lr_geomean_second_half"]["ratio"] == pytest.approx(1.0, rel=1e-6)
    assert {r["run"]: r["rung"] for r in doc["runs"]} == {
        r.name: (-7 if R.arm_of(r)[0] in ("w", "wb") else -3) for r in runs}
    text = R.render(doc)
    assert all(r.name in text for r in runs) and "width ON / OFF" in text and "REPORTED" in text
    assert R.main(["--look", "50M", "--runs", *map(str, runs)]) == 0
    assert capsys.readouterr().out.strip() == text


# ------------------------------------------------------------------------------------------------ the weights
#: name -> (narrow shape, wide shape, the group it must land in).
TENSORS: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], str]] = {
    "features_extractor.embeddings.species.weight": ((6, 4), (6, 4), "embeddings"),
    "features_extractor.projection.weight": ((4, 8), (8, 8), "projection"),
    "features_extractor.projection.bias": ((4,), (8,), "trunk_resized"),
    "features_extractor.blocks.0.attn.weight": ((4, 4), (8, 8), "trunk_resized"),
    "features_extractor.token_norm.scale": ((3,), (3,), "trunk_same_shape"),
    "features_extractor.win_head.0.weight": ((2, 4), (2, 8), "critic_head"),
    "features_extractor.win_head.2.bias": ((1,), (1,), "critic_head"),
    "mlp_extractor.policy_net.0.weight": ((5, 5), (5, 5), "policy_tower"),
    "pointer_head.query.weight": ((3, 4), (3, 8), "pointer_head_resized"),
    "pointer_head.scale": ((2,), (2,), "pointer_head_same_shape"),
}
#: The change each group's weights make between the two checkpoints, on a narrow run; a wide run's is
#: ``WIDE[group]`` times it.
MOVE = {g: 0.01 * (i + 1) for i, g in enumerate(R.GROUPS)}
WIDE = {g: 0.3 + 0.05 * i for i, g in enumerate(R.GROUPS)}


def write_checkpoint(run: Path, step: int, moved: float, wide: bool, *, break_alias: bool = False,
                     extra: Optional[Dict[str, Any]] = None) -> Path:
    """``checkpoints/checkpoint_<step>_steps.zip`` holding a ``policy.pth`` whose every tensor is ``1 + moved * its
    group's move`` (so the displacement between ``moved=0`` and ``moved=1`` is each group's move)."""
    import torch

    sd: Dict[str, Any] = {}
    for name, (narrow, wider, group) in TENSORS.items():
        scale = MOVE[group] * (WIDE[group] if wide else 1.0)
        sd[name] = torch.full(wider if wide else narrow, 1.0 + moved * scale, dtype=torch.float32)
    sd["features_extractor.embeddings.hp_type_idx_map"] = torch.arange(4, dtype=torch.int64) + int(moved * 100)
    for name in [n for n in sd if n.startswith(R.TRUNK)]:
        for alias in R.ALIASES:
            sd[alias + name[len(R.TRUNK):]] = sd[name].clone()
    if break_alias:
        sd["pi_features_extractor.token_norm.scale"] = sd["features_extractor.token_norm.scale"] + 1.0
    sd.update(extra or {})
    buf = io.BytesIO()
    torch.save(sd, buf)
    path = run / "checkpoints" / f"checkpoint_{step}_steps.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("policy.pth", buf.getvalue())
        z.writestr("data", "{}")
    return path


def make_weights(run: Path, **kw: Any) -> None:
    wide = R.arm_of(run)[0] in ("w", "wb")
    write_checkpoint(run, R.DISPLACEMENT_FROM - 1, 5.0, wide)                # before the step: never the one read
    write_checkpoint(run, R.DISPLACEMENT_FROM + 134, 0.0, wide)
    write_checkpoint(run, R.DISPLACEMENT_FROM + A.CHECKPOINT_EVERY + 9, 7.0, wide)   # a later one: not the first
    write_checkpoint(run, L50 + 412, 1.0, wide, **kw)


def test_every_tensor_has_one_group_from_its_own_name():
    for name, (narrow, wider, group) in TENSORS.items():
        assert R.group_of(name, narrow != wider) == group, name
    assert set(g for _n, _w, g in TENSORS.values()) == set(R.GROUPS), "the fixture exercises every group"
    assert set(R.NAMED_GROUPS) < set(R.GROUPS) and len(set(R.GROUPS)) == len(R.GROUPS)
    assert R.group_of("features_extractor.projection.bias", True) == "trunk_resized", "the projection is its WEIGHT"
    with pytest.raises(R.ReportError, match="no module group"):
        R.group_of("value_net.weight", False)
    with pytest.raises(R.ReportError, match="no module group"):
        R.group_of("mlp_extractor.value_net.0.weight", False)


def test_displacement_is_the_rms_change_by_group_and_its_ratio_by_factor(tmp_path):
    models = tmp_path / "models"
    runs = [make_run(models, a, s) for a, s in FIVE]
    for run in runs:
        make_weights(run)
    doc = R.report(runs, "50M", with_displacement=True)
    d = doc["displacement"]
    assert (d["from_step"], d["to_step"]) == (R.DISPLACEMENT_FROM, L50)
    assert d["tensors"] == len(TENSORS), "the integer map and the two alias copies are not counted"
    assert d["resized_tensors"] == sum(n != w for n, w, _g in TENSORS.values())
    for run in runs:
        wide = R.arm_of(run)[0] in ("w", "wb")
        got = d["runs"][run.name]
        assert (got["from_checkpoint_step"], got["to_checkpoint_step"]) == (R.DISPLACEMENT_FROM + 134, L50 + 412)
        for g in R.GROUPS:
            want = MOVE[g] * (WIDE[g] if wide else 1.0)
            assert got["groups"][g]["rms"] == pytest.approx(want, rel=1e-4), (run.name, g)
            assert got["groups"][g]["weights"] == sum(
                math.prod(w if wide else n) for n, w, grp in TENSORS.values() if grp == g)
    for g in R.GROUPS:
        assert d["factors"]["width"][g]["ratio"] == pytest.approx(WIDE[g], rel=1e-4), g
        assert d["factors"]["breadth"][g]["ratio"] == pytest.approx(1.0, rel=1e-4), g
    text = R.render(doc)
    assert "displacement" in text and all(g in text for g in R.GROUPS)


def test_displacement_refuses_rather_than_guess(tmp_path, capsys):
    models = tmp_path / "models"
    runs = [make_run(models, a, s) for a, s in FIVE]
    argv = ["--displacement", "--runs", *map(str, runs)]
    for run in runs[1:]:
        make_weights(run)
    assert R.main(["--look", "50M", *argv]) == 3
    assert f"{runs[0].name}: no checkpoint at or after 30,000,000" in capsys.readouterr().err

    make_weights(runs[0], break_alias=True)
    assert R.main(["--look", "50M", *argv]) == 3
    assert "is not a copy of features_extractor.token_norm.scale" in capsys.readouterr().err

    import torch
    make_weights(runs[0], extra={"value_head.weight": torch.ones(2)})
    assert R.main(["--look", "50M", *argv]) == 3
    assert "do not hold the same tensors" in capsys.readouterr().err

    make_weights(runs[0])
    assert R.main(["--look", "50M", *argv]) == 0
    capsys.readouterr()
    assert R.main(["--look", "100M", *argv]) == 3
    assert "once, at the 50M look" in capsys.readouterr().err


def test_the_report_writes_nothing_in_a_run_directory(tmp_path):
    models = tmp_path / "models"
    runs = [make_run(models, a, s) for a, s in FIVE]
    for run in runs:
        make_weights(run)
    before = sorted((p.relative_to(models), p.stat().st_mtime_ns, p.stat().st_size)
                    for p in models.rglob("*"))
    assert R.main(["--look", "50M", "--displacement", "--runs", *map(str, runs),
                   "--out", str(tmp_path / "out.json")]) == 0
    assert sorted((p.relative_to(models), p.stat().st_mtime_ns, p.stat().st_size)
                  for p in models.rglob("*")) == before
