"""The fresh-cell report's REPORTED training meters (``main.b1.lr_report --fresh-arm``, Q5-N PREREG §3): the meters by
window, the last epoch's clip fraction, the bounds' touches, the dose to the look and the step-time meter with its
exact ratio, on synthetic event files."""
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pytest

from main.b1 import lr_report as R
from main.b1.lr_report_test import BANKED, FRESH_BASE, _f32, arm_file, fresh_doc, make_cell

L50 = R.LOOKS["50M"]
FRESH = ("rb_q5n_s5201", "rb_q5n_s5202", "rb_q5n_s5203")


def write_tags(run: Path, rows: Mapping[str, Mapping[int, float]], stamp: int) -> None:
    """One event file holding ``tag -> {step: value}``, named with ``stamp`` as its timestamp."""
    from torch.utils.tensorboard import SummaryWriter

    tmp = run / "tb" / f"_t{stamp}"
    w = SummaryWriter(log_dir=str(tmp))
    for tag, points in rows.items():
        for step, v in points.items():
            w.add_scalar(tag, v, step)
    w.close()
    (written,) = list(tmp.iterdir())
    written.rename(run / "tb" / f"events.out.tfevents.{stamp:010d}.host.2.0")
    tmp.rmdir()


def timing(steps: Sequence[int], update_s: float, collect_ms: float, train_ms: float) -> Dict[str, Dict[int, float]]:
    return {R.UPDATE_TAG: {s: update_s for s in steps}, R.COLLECT_TAG: {s: collect_ms for s in steps},
            R.TRAIN_MS_TAG: {s: train_ms for s in steps}}


#: Rollout rows every 2M steps from 2M to 52M: 21 in the step-time window, each 2,000,000 steps after the last.
ROLL = list(range(2_000_000, 52_000_001, 2_000_000))


# ------------------------------------------------------------------------------------------------ the meters
def test_a_meter_is_the_median_of_each_5m_window_below_the_look_and_over_the_kl_window():
    pts = {1_000_000: 9.0, 6_000_000: 1.0, 7_000_000: 3.0, 8_000_000: 100.0, 12_000_000: 2.0, 49_999_999: 4.0,
           50_000_000: 7.0, 55_000_000: 8.0}
    m = R.meter(pts, L50)
    assert len(m["by_window"]) == 10 and m["by_window"][0] == 9.0 and m["by_window"][1] == 3.0    # 1, 3, 100
    assert m["by_window"][2] == 2.0 and m["by_window"][3:9] == [None] * 6 and m["by_window"][9] == 4.0
    assert m["by_window_rows"] == [1, 3, 1, 0, 0, 0, 0, 0, 0, 1], "the row at the look step is in no window"
    assert (m["kl_window_median"], m["kl_window_rows"]) == (4.0, 3), "10M <= step <= 50M, both ends in"
    assert len(R.meter(pts, R.LOOKS["100M"])["by_window"]) == 20
    neg = R.meter({12_000_000: -0.6, 13_000_000: 0.0}, L50, -1.0)
    assert neg["by_window"][2] == 0.3 and neg["kl_window_median"] == 0.3
    assert str(R.meter({12_000_000: 0.0}, L50, -1.0)["kl_window_median"]) == "0.0", "never a negative zero"
    empty = R.meter({}, L50)
    assert empty["kl_window_median"] is None and empty["by_window"] == [None] * 10 and empty["rows"] == 0


def test_the_last_epochs_clip_fraction_is_the_highest_epoch_logged_at_each_row():
    series = {f"{R.CLIP_EPOCH_PREFIX}{k}": {10: 0.1 * k, 20: 0.01 * k} for k in range(10)}
    for k in range(5, 10):
        del series[f"{R.CLIP_EPOCH_PREFIX}{k}"][20]                               # row 20 ran five epochs
    series["train/clip_fraction_vf"] = {10: 9.0, 20: 9.0}                         # not an epoch tag
    series[R.CLIP_EPOCH_PREFIX + "x"] = {10: 9.0}
    series[R.CLIP_TAG] = {10: 5.0}
    rows, epochs = R.last_epoch_clip(series)
    assert rows == {10: pytest.approx(0.9), 20: pytest.approx(0.04)} and epochs == [4, 9]
    assert R.last_epoch_clip({}) == ({}, [])
    assert R._wanted(R.CLIP_EPOCH_PREFIX + "12") and not R._wanted("train/clip_fraction_vf")


def test_a_touch_of_the_floor_or_ceiling_is_within_float_error_and_beyond_either_is_counted():
    base = 6e-4
    steps = [1, 2, 3, 4, 5, 6, 7]
    lrs = [_f32(1e-5), 1e-5, _f32(1.2e-3), 1.2e-3 * (1 + 2 * R.RUNG_ON_REL), 1e-5 * (1 - 2 * R.RUNG_ON_REL), base,
           1.2e-3 * (1 + R.RUNG_ON_REL / 2)]
    b = R.bound_touches(steps, lrs, base)
    assert (b["floor"], b["ceiling"]) == (1e-5, 1.2e-3)
    assert (b["at_floor"], b["first_at_floor"], b["below_floor"]) == (2, 1, 1)
    assert (b["at_ceiling"], b["first_at_ceiling"], b["above_ceiling"]) == (2, 3, 1)
    none = R.bound_touches(steps, [base] * 7, R.LADDER_BASE)
    assert none["ceiling"] == 6e-4 and none["at_ceiling"] == 7 and none["first_at_ceiling"] == 1, "C's ceiling"
    assert R.bound_touches(steps, [3e-4] * 7, R.LADDER_BASE)["at_ceiling"] == 0


# ------------------------------------------------------------------------------------------------ the step time
def test_the_step_time_is_the_card_checks_formula_in_exact_arithmetic():
    upd = {s: 40.0 + (s // 2_000_000) % 3 for s in ROLL}            # 41, 42, 40, ... : median 41
    col = {s: 24_500.0 for s in ROLL}
    tms = {s: 39_000.0 for s in ROLL}
    st = R.step_time("r", {R.UPDATE_TAG: upd, R.COLLECT_TAG: col, R.TRAIN_MS_TAG: tms})
    assert st["rows"] == 21 and (st["first_step"], st["last_step"]) == (10_000_000, 50_000_000)
    want = (Fraction(upd[10_000_000] * 0 + 41) + Fraction(245, 10)) / 2_000_000 * 10 ** 6
    assert Fraction(st["sec_per_1M_steps_exact"]) == want == Fraction(655, 20)
    assert st["sec_per_1M_steps"] == 32.75 and st["median_dsteps"] == 2_000_000.0
    assert st["train_ms_sec_per_1M_steps"] == 19.5 and st["train_ms_rows"] == 21
    assert st["update_share"] == pytest.approx(41 / 65.5)
    # only the rows carrying BOTH tags count, and d(steps) is read between them
    gap = dict(col)
    del gap[20_000_000]
    st2 = R.step_time("r", {R.UPDATE_TAG: upd, R.COLLECT_TAG: gap})
    assert st2["rows"] == 20 and st2["train_ms_sec_per_1M_steps"] is None and st2["train_ms_rows"] == 0


def test_the_step_time_refuses_what_does_not_span_its_window_or_is_too_few_rows_or_not_positive():
    full = timing(ROLL, 40.0, 24_500.0, 39_000.0)
    late = {t: {s: v for s, v in p.items() if s >= 12_000_000} for t, p in full.items()}
    with pytest.raises(R.ReportError, match="do not span the step-time window 10,000,000 to 50,000,000"):
        R.step_time("r", late)
    early = {t: {s: v for s, v in p.items() if s <= 48_000_000} for t, p in full.items()}
    with pytest.raises(R.ReportError, match="do not span"):
        R.step_time("r", early)
    with pytest.raises(R.ReportError, match="rollout rows .none."):
        R.step_time("r", {})
    sparse = timing([5_000_000, 10_000_000, 30_000_000, 50_000_000, 51_000_000], 40.0, 1.0, 1.0)
    with pytest.raises(R.ReportError, match="3 rollout rows in the step-time window; the meter needs 10"):
        R.step_time("r", sparse)
    with pytest.raises(R.ReportError, match="not positive"):
        R.step_time("r", timing(ROLL, 0.0, 0.0, 0.0))


def test_rho_is_the_exact_ratio_of_the_cells_mean_step_times():
    def one(sec: Fraction) -> Dict[str, Any]:
        return {"sec_per_1M_steps_exact": R.exact(sec)}

    fresh = [one(Fraction(400)), one(Fraction(401)), one(Fraction(402))]
    banked = [one(Fraction(600)), one(Fraction(600)), one(Fraction(603))]
    assert R.step_time_ratio(fresh, banked) == Fraction(401, 601)
    assert Fraction(R.exact(Fraction(1, 3))) == Fraction(1, 3) and R.exact(Fraction(6, 4)) == "3/2"
    with pytest.raises(R.ReportError, match="no runs"):
        R.cell_mean_exact([])


# ------------------------------------------------------------------------------------------------ the record
def meters_cell(models: Path, *, fresh_update: float = 30.0, banked_update: float = 40.0, sidecars: bool = True) -> None:
    """:func:`make_cell`'s six runs, each with a second event file of the meters (from 2M, so no LR/KL row is
    superseded) and, with ``sidecars``, per-checkpoint sidecars every 10M."""
    make_cell(models)
    for i, name in enumerate([*FRESH, *BANKED]):
        upd = fresh_update if name in FRESH else banked_update
        tags = {**timing(ROLL, upd, 20_000.0, 1000.0 * upd),
                R.ENTROPY_LOSS_TAG: {s: -0.5 - 0.01 * i for s in ROLL},
                R.VALUE_LOSS_TAG: {s: 0.02 for s in ROLL}, R.CLIP_TAG: {s: 0.15 for s in ROLL},
                **{f"{R.CLIP_EPOCH_PREFIX}{k}": {s: 0.1 + 0.01 * k for s in ROLL}
                   for k in range(5 if name in FRESH else 10)}}
        write_tags(models / name, tags, stamp=2)
        if sidecars:
            ck = models / name / "checkpoints"
            ck.mkdir(parents=True, exist_ok=True)
            lr = FRESH_BASE / 2 if name in FRESH else R.LADDER_BASE / 2
            for n in (10_000_005, 20_000_005, 30_000_005, 40_000_005, 50_000_005, 60_000_005):
                (ck / f"checkpoint_{n}_steps.zip").write_bytes(b"z")
                (ck / f"checkpoint_{n}_steps.json").write_text(json.dumps(
                    {"num_timesteps": n, "lr": lr if n <= L50 + 50_000 else 1.0, "batch_size": 2048,
                     "grad_accum_steps": 32, "n_epochs": 5 if name in FRESH else 10,
                     "dose": {"rollout_rows": 98_304}}))


def test_the_fresh_record_carries_every_meter_each_cells_mean_and_the_two_ratios(tmp_path, capsys):
    models = tmp_path / "models"
    meters_cell(models)
    doc = fresh_doc(tmp_path)
    one, banked = doc["runs"][0], doc["runs"][3]
    assert one["meters"]["entropy"]["by_window"][2] == pytest.approx(0.5) and one["meters"]["entropy"]["rows"] == 26
    assert one["meters"]["clip_fraction_last_epoch"]["epochs"] == [4]
    assert banked["meters"]["clip_fraction_last_epoch"]["epochs"] == [9]
    assert one["meters"]["clip_fraction_last_epoch"]["kl_window_median"] == _f32(0.14)
    assert banked["meters"]["clip_fraction_last_epoch"]["kl_window_median"] == _f32(0.19)
    assert doc["cells"]["banked"]["meters"]["entropy"]["kl_window_median_mean"] == pytest.approx(0.54, rel=1e-6)
    assert one["lr_bounds"]["ceiling"] == 2 * FRESH_BASE and banked["lr_bounds"]["ceiling"] == 2 * R.LADDER_BASE
    # the step time: (30 + 20) / 2M x 1e6 = 25 s against (40 + 20) / 2M x 1e6 = 30 s: rho 5/6, exactly
    assert one["step_time"]["sec_per_1M_steps"] == 25.0 and banked["step_time"]["sec_per_1M_steps"] == 30.0
    assert doc["step_time_ratio_exact"] == "5/6" and doc["step_time_window"] == [10_000_000, 50_000_000]
    # the dose to the look: the sidecars at or before the look's checkpoint only (the 60M one, at lr 1.0, is out)
    assert one["dose"]["upto"] == 50_000_005 and one["look_row_step"] == L50 and one["dose"]["n_lr"] == 5
    assert one["dose"]["dose_rate"] == pytest.approx(FRESH_BASE / 2 * 5 * 2 / 98_304)
    assert doc["dose_rate_ratio"] == pytest.approx(1.0), "half the passes at twice the rate: the matched start"
    assert R.main(["--look", "50M", "--fresh-arm", str(arm_file(tmp_path)), "--models-dir", str(models)]) == 0
    text = capsys.readouterr().out
    for line in ("entropy (-train/entropy_loss)", "last epochs seen [4, 9]", "fresh / banked (rho) 0.8333 = 5/6",
                 "fresh / banked 1", "rb_q5n_s5201        25.00", "floor 1.0e-05: 0 at (first None), 0 below"):
        assert line in text, line


def test_an_absent_meter_reads_as_absent_and_never_refuses_the_registered_label(tmp_path, capsys):
    """Runs that carry only the learning rate and the KL: every meter is None, the step time and the dose carry their
    error, the ratios are None, and the label is still read."""
    make_cell(tmp_path / "models")
    doc = fresh_doc(tmp_path)
    assert doc["label"] is None and doc["figure"] == _f32(0.012)
    r = doc["runs"][0]
    assert r["meters"]["entropy"]["kl_window_median"] is None and r["meters"]["value_loss"]["rows"] == 0
    assert "do not span the step-time window" in r["step_time"]["error"] and r["dose"]["error"]
    assert doc["step_time_ratio"] is None and doc["dose_rate_ratio"] is None
    assert doc["cells"]["fresh"]["sec_per_1M_steps_mean"] is None
    assert R.main(["--look", "50M", "--fresh-arm", str(arm_file(tmp_path)), "--models-dir",
                   str(tmp_path / "models")]) == 0
    text = capsys.readouterr().out
    assert "ERROR rb_q5n_s5201: its rollout rows (none)" in text and "fresh / banked (rho) -\n" in text


def test_one_runs_unreadable_step_time_leaves_rho_unread_rather_than_a_two_run_mean(tmp_path):
    models = tmp_path / "models"
    meters_cell(models, sidecars=False)
    write_tags(models / BANKED[1], {R.UPDATE_TAG: {s: 40.0 for s in ROLL if s <= 40_000_000}}, stamp=3)
    doc = fresh_doc(tmp_path)
    assert "do not span" in doc["runs"][4]["step_time"]["error"]
    assert doc["cells"]["banked"]["sec_per_1M_steps_mean_exact"] is None and doc["step_time_ratio"] is None
    assert doc["cells"]["fresh"]["sec_per_1M_steps_mean_exact"] == "25/1"


def test_a_resume_supersedes_the_meters_rows_as_it_does_the_rates(tmp_path):
    models = tmp_path / "models"
    meters_cell(models, sidecars=False)
    write_tags(models / FRESH[0], {R.ENTROPY_LOSS_TAG: {s: -2.0 for s in ROLL if s >= 30_000_000}}, stamp=3)
    m = fresh_doc(tmp_path)["runs"][0]["meters"]["entropy"]
    assert m["by_window"][4] == 0.5 and m["by_window"][6] == 2.0 and m["rows"] == 26
