"""THE PER-ROW CLIP-MARGIN BAR, STAGE 1 (`gen3_canary_row_bar_v1`; `canary_row_bar`), and the bf16 loss envelope's K.

CPU, no learner and no compile (the rung's arms planted through the REAL `_r1_judge` / `_clip_margin_rule`,
`canary_yardstick_test._Rung`; the canary with a planted null). Every test here fails on a revert of the ruling:

* the constants are the ruled ones and RE-DERIVE from the committed basis (`testdata/canary_stage1_basis.json`, from the
  W0 CUDA census and the 159 banked production checks): k = 6.0, the cap = M = 0.72, the loss K = 9.2;
* ``min(k x max(null, 0.02), 0.72)`` per row, a non-finite null at the cap, and every fallback at the cap;
* the compiled arm is refused on a row whose shift passes ITS bar although the shift is far under M, and the yardsticks
  keep the scalar margin;
* THE BACKTEST: the census's compiled arm refuses on 0 of 20 checkpoints (scheduled and both halves) under the rule,
  and a synthetic shift just above a row's bar (still under M) refuses where the global M passes it;
* the canary binds the bars from its null, falls back to the cap when the null cannot bind, and every record names
  ``canary_rule_version``.

Planted values sit far from every bar (rule 8): 0.30 against bars 0.12 / 0.72; the census reads at most 0.52 of its
bars; a synthetic breach is 1.5 x the bar."""
from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

from agents.model import canary_row_bar as RB
from agents.model import canary_row_null as RN
from agents.model import compile_canary as CC
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model.canary_yardstick_test import ROWS, _Rung

BASIS = os.path.join(os.path.dirname(__file__), "testdata", "canary_stage1_basis.json")


def _basis() -> Dict[str, Any]:
    with open(BASIS) as f:
        out: Dict[str, Any] = json.load(f)
    return out


def _round_up_2sig(x: float) -> float:
    """R1's convention for every bar: rounded UP to 2 significant digits."""
    e = math.floor(math.log10(x)) - 1
    return float(f"{math.ceil(x / 10 ** e - 1e-9) * 10 ** e:.10g}")


# ------------------------------------------------------------------------------------------- the ruled constants
def test_the_constants_are_the_ruled_ones_and_the_cap_IS_the_global_margin():
    assert (RB.ROW_BAR_K, RB.ROW_BAR_FLOOR, RB.ROW_BAR_SEEDS, RB.ROW_BAR_CAP) == (6.0, 0.02, 16, 0.72)
    assert RB.ROW_BAR_CAP == cr.R1_CLIP_MARGIN                  # stage 1: C = M (strictly tighter than the global rule)
    assert RB.ROW_BAR_FLOOR == RN.ROW_NULL_FLOOR and RB.ROW_BAR_SEEDS == RN.ROW_NULL_SEEDS
    assert cr.BF16_ENVELOPE_K == {"loss": 9.2, "grad_rel_l2": 6.4, "param": 12.0}   # loss 5.4 -> 9.2; param 12 kept


def test_k_RE_DERIVES_from_the_W0_census():
    """k = 4 x the healthy max of shift / max(16-seed null, 0.02) over the census's compiled AND eager bf16 arms."""
    worst, at = -1.0, None
    for e in _basis()["census"]:
        null = np.maximum(np.asarray(e["null_k16"]), RB.ROW_BAR_FLOOR)
        for arm in ("Cbf", "Ebf"):
            r = np.asarray(e[arm]) / null
            if r.max() > worst:
                worst, at = float(r.max()), (e["label"], arm, int(r.argmax()))
    assert at == ("rb_x26_s2601_380M", "Ebf", 63) and worst == pytest.approx(1.479, abs=1e-3)
    assert _round_up_2sig(cr.R1_BAR_K * worst) == RB.ROW_BAR_K


def test_the_loss_envelope_K_RE_DERIVES_from_the_production_checks_and_clears_the_C0_anchor():
    """K_loss = 4 x the healthy max over the 159 banked production checks. The C0 s3501 final's slice 1 (5.59 x eager's
    floored distance) refused at window D's 5.4 and passes at 9.2 (the X-CHECK, out of sample)."""
    b = _basis()
    prod = [p["ratio"] for p in b["production_loss_envelope"]]
    assert len(prod) == 159 and max(prod) == pytest.approx(2.283, abs=1e-3)
    assert _round_up_2sig(cr.R1_BAR_K * max(prod)) == cr.BF16_ENVELOPE_K["loss"] == 9.2
    (c0,) = [a["ratio"] for a in b["anchors_loss_envelope"] if a["anchor"] == "c0_s3501" and a["slice"] == 1]
    assert 5.4 < c0 < cr.BF16_ENVELOPE_K["loss"]
    assert max(a["ratio"] for a in b["anchors_loss_envelope"]) / cr.BF16_ENVELOPE_K["loss"] < 0.7


# ------------------------------------------------------------------------------------------- the bar itself
def test_the_per_row_bar_is_k_x_the_floored_null_CAPPED_and_a_non_finite_null_sits_at_the_cap():
    got = RB.row_bars([0.0, 0.01, 0.05, 0.1, 0.2, float("nan"), float("inf")])
    assert got.tolist() == pytest.approx([0.12, 0.12, 0.30, 0.60, 0.72, 0.72, 0.72])


@pytest.mark.parametrize("raw,why", [
    (None, "off"), ({"skipped": "projected 9 s"}, "skipped"), ({"error": "RuntimeError: x"}, "error"),
    ({"truncated": "after 5 of 16", "null": np.zeros(ROWS), "per_seed": np.zeros((5, ROWS))}, "truncated"),
    ({"null": np.zeros(ROWS), "per_seed": np.zeros((3, ROWS))}, "3 seeds"), ({"rows": ROWS}, "no rows")])
def test_a_null_that_cannot_bind_FALLS_BACK_to_the_cap_naming_why(raw, why):
    rb = RB.from_null(raw)
    assert not rb.live and rb.source.startswith("fallback") and why in rb.source, rb.source
    bb = rb.for_batch(np.arange(2 * ROWS) % ROWS)
    assert bb.bars.shape == (2 * ROWS,) and np.all(bb.bars == RB.ROW_BAR_CAP)
    assert rb.record()["row_bars_source"] == rb.source and "bar_per_row" not in rb.record()


def test_a_16_seed_null_BINDS_and_tiles_to_the_batch():
    null = np.full(ROWS, 0.2)
    null[20] = 0.01
    rb = RB.from_null({"null": null, "per_seed": np.tile(null, (16, 1))})
    assert rb.live and rb.source == "null"
    bb = rb.for_batch(np.arange(2 * ROWS) % ROWS)
    assert bb.bars[20] == bb.bars[20 + ROWS] == pytest.approx(0.12) and bb.bars[21] == pytest.approx(0.72)
    with pytest.raises(ValueError, match="outside"):
        rb.for_batch([ROWS])


# ------------------------------------------------------------------------------------------- the rule in the rung
def _bars(row: int, null_at_row: float) -> RB.BatchRowBars:
    null = np.full(ROWS, 0.2)
    null[row] = null_at_row
    return RB.from_null({"null": null, "per_seed": np.tile(null, (16, 1))}).for_batch(np.arange(ROWS))


def _judge(watch: Optional[cr.GradWatch] = None, row_bars: Any = None) -> str:
    model = SimpleNamespace(policy=None)                          # every collaborator that reads it is planted
    return cr._r1_judge(model, ("args",), ["pointer_head.w"], "trained", watch=watch, row_bars=row_bars)[2]


def test_the_compiled_arm_is_REFUSED_past_its_rows_bar_though_far_under_M(monkeypatch):
    """Fails on revert: compiled 0.30 on row 20 is under M 0.72 (the global rule passes it), but row 20's null is
    0.01, so its bar is 6 x 0.02 = 0.12."""
    _Rung(monkeypatch, [{"compiled": 0.30, "eager": 0.10}], {"eager": 4, "compiled": 20})
    assert "0 clip-branch switches" in _judge()                                    # the global rule: PASS
    with pytest.raises(ct.CompileTrainerError, match=r"compiled arm moved fixture row 20 \(batch row 20\) by 0\.3 "
                                                     r"from the fp32 eager reference, past that row's bar 0\.12") as e:
        _judge(row_bars=_bars(20, 0.01))
    assert not isinstance(e.value, cr.YardstickRefusal) and RB.CANARY_RULE_VERSION in str(e.value)
    w = cr.GradWatch(())
    line = _judge(w, row_bars=_bars(20, 0.2))                                       # bar min(1.2, 0.72): PASS
    assert "per-row bars [per_row_stage1" in line and "compiled 0.3 / 0.72" in line, line
    rb = w.record()["clip_margin"]["row_bar"]
    assert rb["bars"] == "the null" and rb["arms"]["compiled"]["fixture_row"] == 20
    assert rb["arms"]["compiled"]["fraction"] == pytest.approx(0.30 / 0.72, rel=1e-3)
    assert "eager" not in rb["arms"]                                              # the yardsticks are not per-row


def test_the_YARDSTICKS_keep_the_scalar_margin(monkeypatch):
    """Eager 0.30 on a row whose per-row bar would be 0.12: no refusal (eager is the yardstick, judged at M)."""
    _Rung(monkeypatch, [{"compiled": 0.05, "eager": 0.30}], {"eager": 20, "compiled": 4})
    line = _judge(row_bars=_bars(20, 0.01))
    assert "eager 3.00e-01 (row 20)" in line and "<= 0.720" in line, line


def test_a_raised_margin_never_raises_a_per_row_bar_past_the_cap(monkeypatch):
    """Compiled 0.9 under a yardstick re-run at M' 1.07: the global rule judges it at M' and passes; the per-row rule
    holds it at the cap 0.72 (stage 1)."""
    rung = _Rung(monkeypatch, [{"eager": 0.856, "compiled": 0.90}], {"eager": 4, "compiled": 20})
    assert "M 1.070" in _judge()
    assert rung.margins[-1] == pytest.approx(1.07)
    with pytest.raises(ct.CompileTrainerError, match=r"past that row's bar 0\.72"):
        _judge(row_bars=_bars(20, 0.2))


def test_bar_fractions_read_the_judged_arm_against_its_rows_bar(monkeypatch):
    _Rung(monkeypatch, [{"compiled": 0.06, "eager": 0.10}], {"eager": 4, "compiled": 20})
    w = cr.GradWatch(())
    _judge(w, row_bars=_bars(20, 0.01))
    f = CC.bar_fractions(w)["clip_margin"]
    assert f["arm"] == "compiled" and f["bar"] == pytest.approx(0.12) and f["fraction"] == pytest.approx(0.5, rel=1e-3)


# ------------------------------------------------------------------------------------------- the backtest
def _census_judge(e: Dict[str, Any], shifts: np.ndarray, rows: Any = None) -> Dict[str, Any]:
    """The production `canary_row_bar.judge` on one census checkpoint's rows (``rows``: a confirmation half)."""
    rb = RB.from_null({"null": np.asarray(e["null_k16"]), "per_seed": np.zeros((RB.ROW_BAR_SEEDS, ROWS))})
    idx = np.arange(ROWS) if rows is None else np.asarray(list(rows))
    ref = torch.zeros(idx.size, dtype=torch.float64)
    arm = {"log_prob": torch.as_tensor(shifts[idx], dtype=torch.float64)}
    return RB.judge(ref, {"compiled": arm}, rb.for_batch(idx))["compiled"]


def test_THE_BACKTEST_the_census_refuses_none_and_a_shift_above_a_rows_bar_refuses():
    """The W0 census's compiled bf16 arm (20 checkpoints x 64 rows, CUDA, B 2,048): 0 refusals under the per-row rule,
    on the scheduled rows and on each half (the census reads at most 0.52 of its bars: rule 8). A synthetic shift of
    1.5 x the bar on the row with the TIGHTEST bar (always under M) refuses, where the global M passes it."""
    census = _basis()["census"]
    assert len(census) == 20
    worst = 0.0
    for e in census:
        cbf = np.asarray(e["Cbf"])
        for rows in (None, range(32), range(32, 64)):        # the scheduled check, then `FIXTURE_HALVES` 2 and 1
            j = _census_judge(e, cbf, rows)
            assert not j["fails"], (e["label"], j)
            worst = max(worst, j["fraction"])
        bars = RB.row_bars(e["null_k16"])
        r = int(np.argmin(bars))
        planted = cbf.copy()
        planted[r] = 1.5 * bars[r]
        assert planted[r] < cr.R1_CLIP_MARGIN                # the global rule would pass it
        j = _census_judge(e, planted)
        assert j["fails"] and j["fixture_row"] == r, (e["label"], j)
    assert worst < 0.6


# ------------------------------------------------------------------------------------------- the canary binds them
def _canary(monkeypatch, tmp, null: Optional[np.ndarray], shifts: Dict[str, float]):
    """A canary whose null is planted (16 seeds, or OFF when ``null`` is None) and whose checks run the REAL rung on
    planted arms (B = 64: batch row == fixture row on slice 0; the halves tile their own rows)."""
    monkeypatch.setattr(CC, "_WEIGHT_DUMPS", 0)
    raw = None if null is None else {"k": 16, "rows": ROWS, "seconds": 0.1, "null": null,
                                     "per_seed": np.tile(null, (16, 1))}
    monkeypatch.setattr(RN, "compute", lambda _m: dict(raw))
    seen: List[Any] = []

    def check(self, slice_, out, watch=None, row_bars=None):
        seen.append(row_bars)
        fr = ct.fixture_index(ROWS, ROWS, slice_)
        if watch is not None:
            watch.fixture_rows = [int(x) for x in fr]
        with monkeypatch.context() as mp:
            _Rung(mp, [shifts], {"eager": 4, "compiled": 20})
            return [_judge(watch, None if row_bars is None else row_bars.for_batch(fr))]
    monkeypatch.setattr(CC.CompileCanary, "_check", check)
    model = SimpleNamespace(policy=torch.nn.Module(), behaviour_dump_dir=str(tmp), num_timesteps=500_000_000)
    monkeypatch.setattr(CC, "_dump_weights", lambda *a, **k: None)
    can = CC.CompileCanary(model, batch_size=ROWS, every=1, saver=lambda _p: None, emit=lambda _l: None,
                           row_null=null is not None)
    return can, seen


def _records(tmp) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in (tmp / CC.VERDICTS_FILE).read_text().splitlines()]


def test_the_canary_BINDS_its_nulls_bars_and_every_record_names_the_rule(monkeypatch, tmp_path):
    """Fails on revert: compiled 0.30 on row 20 (null 0.01, bar 0.12) is refused on the scheduled check and its half,
    and the record says why; the global rule passed it."""
    null = np.full(ROWS, 0.2)
    null[20] = 0.01
    can, seen = _canary(monkeypatch, tmp_path, null, {"compiled": 0.30, "eager": 0.10})
    out = can.after_update()
    assert out["compile/canary_ok"] == 0.0 and out["compile/canary_row_bars_live"] == 1.0
    assert len(seen) == 3 and all(s is not None and s.live for s in seen)     # scheduled + both halves, same bars
    (rec,) = _records(tmp_path)
    assert rec["verdict"] == "unconfirmed" and "past that row's bar 0.12" in rec["detail"]
    assert rec["canary_rule_version"] == RB.CANARY_RULE_VERSION == "per_row_stage1"
    rule = rec["canary_rule"]
    assert rule["row_bars_source"] == "null" and rule["k"] == 6.0 and rule["cap"] == 0.72
    assert rule["envelope_loss_k"] == 9.2 and rule["bar_per_row"][20] == pytest.approx(0.12)
    assert rec["row_null"]["judged"] is True and rec["row_null"]["shadow"] is False
    assert rec["bar_fractions"]["clip_margin"]["bar"] == pytest.approx(0.12)


def test_a_null_that_is_OFF_falls_back_to_the_cap_and_says_so(monkeypatch, tmp_path):
    can, seen = _canary(monkeypatch, tmp_path, None, {"compiled": 0.30, "eager": 0.10})
    out = can.after_update()
    assert out["compile/canary_ok"] == 1.0 and out["compile/canary_row_bars_live"] == 0.0
    (rec,) = _records(tmp_path)
    assert rec["verdict"] == "pass" and rec["canary_rule_version"] == RB.CANARY_RULE_VERSION
    assert rec["canary_rule"]["row_bars_source"].startswith("fallback: the per-row null is off")
    assert rec["bar_fractions"]["clip_margin"]["bar"] == pytest.approx(RB.ROW_BAR_CAP)
    assert seen[0] is not None and not seen[0].live


def test_a_PASS_on_a_high_null_row_records_its_bars(monkeypatch, tmp_path):
    can, _seen = _canary(monkeypatch, tmp_path, np.full(ROWS, 0.2), {"compiled": 0.30, "eager": 0.10})
    assert can.after_update()["compile/canary_ok"] == 1.0
    (rec,) = _records(tmp_path)
    assert rec["verdict"] == "pass" and rec["canary_rule"]["row_bars_source"] == "null"
    assert rec["watch"]["scheduled"]["clip_margin"]["row_bar"]["arms"]["compiled"]["bar"] == pytest.approx(0.72)
    assert rec["bar_fractions"]["clip_margin"]["fraction"] == pytest.approx(0.30 / 0.72, rel=1e-3)
