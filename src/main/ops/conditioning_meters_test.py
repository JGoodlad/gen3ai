"""``main.ops.conditioning_meters`` — the SPREAD IDENTITY, the decodes from ``V``, and every
refusal, on a synthetic trace tree with a KNOWN answer.

The meters answer one question — *does the critic's opinion move with who it is playing and whose
team it is holding?* — and the whole point of the identity is that a calibrated head reads 1.0 and
a marginal head reads 0. So the tree is planted twice from the same skeleton: once with a ``V``
that reproduces each opponent's true win rate (a SEPARATING head) and once with a ``V`` that is
the same number everywhere (a MARGINAL head). Recovering ~1 and ~0 respectively, with intervals,
is the test that the arithmetic measures what it claims.

What else is planted, because each of these has cost a wrong reading somewhere in this tree:

* the CLAMPED ratio can sit BELOW its own interval (head-refit hazard 1) — asserted, not
  worked around, and the unclamped companion is asserted to be finite and positive where the
  clamped one is exactly zero;
* the leave-one-battle-out label must not contain the outcome of the battle it labels;
* an UNANCHORED snapshot-ladder refit is REFUSED, never silently used as a strength axis
  (mixture-diagnostic hazard 1: ``fit_ladder`` returns ratings near 1000 from the wrong cwd);
* a cycle with no manifest ``selection`` block is SELECTION UNKNOWN and REFUSES (rule 17).

Pure unit, unmarked (runs in every tier): no ``models/``, no subprocess, no model forward.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np
import pytest

from main.ops import conditioning_meters as CM

_TURNS = (1, 2, 3, 6, 12, 30)


def _plant(tmp_path, *, wr_by_opp: Dict[str, float], v_of: "callable",
           n_battles: int = 20, n_teams: int = 5, step: int = 100,
           seed: int = 0, schema: Optional[int] = 1):
    """A trace cycle with per-opponent win rates EXACTLY ``wr_by_opp`` and a planted ``V``.

    Capture rates are 1.0 on both outcome classes, so the traced sample IS the population and the
    manifest's ``battles_won / battles_played`` is reproduced by the rows — that keeps the test
    about the spread arithmetic rather than about the reweighting, which has its own test.
    """
    rng = np.random.default_rng(seed)
    cyc = tmp_path / "eval_traces" / f"step_{step}"
    sel: Dict[str, dict] = {}
    for opp, wr in wr_by_opp.items():
        odir = cyc / opp
        odir.mkdir(parents=True)
        wins = int(round(wr * n_battles))
        for b in range(n_battles):
            y = 1.0 if b < wins else 0.0
            v = float(np.clip(v_of(opp, wr, rng), 0.001, 0.999))
            n = len(_TURNS)
            np.savez(odir / f"b{b}_states.npz",
                     values=np.full(n, v), win_probs=np.full(n, v),
                     has_state=np.ones(n, dtype=np.int64))
            (odir / f"b{b}_summary.json").write_text(json.dumps({
                "meta": {"result": "WIN" if y else "LOSS", "step": step},
                "teams": {"ours": [{"species": f"mon{(b % n_teams) * 6 + k}"} for k in range(6)]},
                "invocations": [{"i": i, "turn": t} for i, t in enumerate(_TURNS)]}))
        sel[opp] = {"battles_played": n_battles, "battles_won": wins, "battles_drawn": 0,
                    "traces_written": n_battles, "traces_won": wins,
                    "capture_rate_win": 1.0, "capture_rate_loss": 1.0}
    man = {"step": step, "selection_schema": schema, "opponents": sorted(wr_by_opp),
           "selection": ({"opponents": sel} if schema is not None else None)}
    (cyc / "eval_manifest.json").write_text(json.dumps(man))
    return str(cyc)


_WRS = {"heuristic": 0.9, "staller": 0.7, "aggressive": 0.5,
        "sentinel_0": 0.3, "sentinel_1": 0.1}


def _separating(opp, wr, rng):
    """A head that says the truth about each opponent, plus per-battle noise."""
    return wr + rng.normal(0, 0.05)


def _marginal(opp, wr, rng):
    """A head that emits ONE number regardless of opponent — the defect under test."""
    return 0.6 + rng.normal(0, 0.05)


def _ratio(trace_dir, bucket="t1_3", boot=300, seed=7):
    arr, meta = CM.extract_cycle(trace_dir)
    b = CM.rollup(arr)
    opps, cid = CM.cell_index(b)
    n_cells = len(opps)
    true_wr = np.full(n_cells, np.nan)
    n_games = np.full(n_cells, np.nan)
    for o, wr, g in zip(b["opponent"].tolist(), b["true_wr"].tolist(), b["n_games"].tolist()):
        true_wr[opps.index(o)], n_games[opps.index(o)] = wr, g
    sel0 = np.arange(b["y"].size)
    st = CM.cell_stats(b, sel0, cid, n_cells, bucket, true_wr)
    return CM.spread_corrected(st, st["n_battles"] > 0, true_wr, n_games), meta


# --------------------------------------------------------------------------- the identity

def test_a_separating_head_recovers_a_spread_ratio_of_one(tmp_path) -> None:
    sc, meta = _ratio(_plant(tmp_path, wr_by_opp=_WRS, v_of=_separating, seed=1))
    assert meta["n_battles"] == 100 and meta["n_opponents"] == 5
    assert 0.80 < sc["ratio"] < 1.30, sc
    assert abs(sc["delta"]) < 0.06


def test_a_marginal_head_recovers_a_spread_ratio_of_zero(tmp_path) -> None:
    sc, _ = _ratio(_plant(tmp_path, wr_by_opp=_WRS, v_of=_marginal, seed=2))
    assert sc["ratio"] < 0.15, sc
    assert sc["delta"] < -0.20          # sd(V) far below sd(outcome)
    assert sc["sd_y"] > 0.20


def test_the_two_heads_are_separated_by_the_meter_by_a_wide_margin(tmp_path) -> None:
    sep, _ = _ratio(_plant(tmp_path / "sep", wr_by_opp=_WRS, v_of=_separating, seed=3))
    mar, _ = _ratio(_plant(tmp_path / "mar", wr_by_opp=_WRS, v_of=_marginal, seed=3))
    assert sep["ratio"] - mar["ratio"] > 0.6


def test_the_clamped_ratio_can_be_exactly_zero_while_the_raw_companion_is_not(tmp_path) -> None:
    """🚨 head-refit hazard 1. The corrected ratio floors a negative variance at zero, so a
    marginal head reads EXACTLY 0.000 while the unclamped companion is small but positive. A
    reader who takes the 0.000 as "no spread at all" has read a clamp, not a measurement."""
    sc, _ = _ratio(_plant(tmp_path, wr_by_opp=_WRS, v_of=_marginal, seed=4))
    assert sc["ratio"] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < sc["ratio_raw"] < 0.4
    assert np.isfinite(sc["noise_V"]) and sc["noise_V"] > 0


def test_the_interval_of_a_clamped_zero_sits_above_it(tmp_path) -> None:
    """The same hazard, through the whole block: the point is exactly 0.000, it sits AT OR BELOW
    its own interval's lower bound rather than inside it, and the interval reaches well above —
    which is why the render prints both the clamped and the raw ratio and calls the INTERVAL the
    read. A reader who takes the 0.000 at face value concludes "no spread at all"."""
    _plant(tmp_path, wr_by_opp=_WRS, v_of=_marginal, seed=4)
    blk = CM.conditioning_block(str(tmp_path), 100, boot=600, ladder="off", seed=4)
    pt = blk["points"]["cond.spread_ratio.t1_3"]
    lo, hi = np.percentile(blk["_draws"]["cond.spread_ratio.t1_3"], [2.5, 97.5])
    assert pt == pytest.approx(0.0, abs=1e-9)
    assert pt <= lo and hi > 0.02, (pt, lo, hi)   # never an interior point of its own interval
    assert blk["points"]["cond.spread_ratio_raw.t1_3"] > 0.0


def test_the_block_recovers_both_regimes_with_intervals(tmp_path) -> None:
    _plant(tmp_path / "sep", wr_by_opp=_WRS, v_of=_separating, seed=6)
    _plant(tmp_path / "mar", wr_by_opp=_WRS, v_of=_marginal, seed=6)
    sep = CM.conditioning_block(str(tmp_path / "sep"), 100, boot=400, ladder="off", seed=6)
    mar = CM.conditioning_block(str(tmp_path / "mar"), 100, boot=400, ladder="off", seed=6)
    s_lo = np.percentile(sep["_draws"]["cond.spread_ratio.t1_3"], 2.5)
    m_hi = np.percentile(mar["_draws"]["cond.spread_ratio.t1_3"], 97.5)
    assert sep["points"]["cond.spread_ratio.t1_3"] > 0.8
    assert mar["points"]["cond.spread_ratio.t1_3"] < 0.15
    assert s_lo > m_hi, (s_lo, m_hi)          # the two regimes' intervals do not overlap
    assert set(sep["_draws"]) >= {"cond.spread_ratio.all", "cond.opp_class_auc.t1"}


def test_the_elo_slope_row_is_omitted_with_a_reason_when_the_axis_is_off(tmp_path) -> None:
    _plant(tmp_path, wr_by_opp=_WRS, v_of=_marginal, seed=7)
    blk = CM.conditioning_block(str(tmp_path), 100, boot=100, ladder="off", seed=7)
    assert "cond.elo_slope" not in blk["points"]
    assert "DISABLED" in blk["omitted"]["cond.elo_slope"]


def test_the_opponent_class_auc_separates_a_head_that_ranks_the_pool(tmp_path) -> None:
    """`sentinel_*` is the POOL class. A head whose V tracks the true win rate ranks the two
    classes apart (the sentinels here are the strong ones); a marginal head cannot."""
    _plant(tmp_path / "sep", wr_by_opp=_WRS, v_of=_separating, seed=8)
    _plant(tmp_path / "mar", wr_by_opp=_WRS, v_of=_marginal, seed=8)
    sep = CM.conditioning_block(str(tmp_path / "sep"), 100, boot=200, ladder="off", seed=8)
    mar = CM.conditioning_block(str(tmp_path / "mar"), 100, boot=200, ladder="off", seed=8)
    assert sep["points"]["cond.opp_class_auc.t1"] > 0.90
    assert abs(mar["points"]["cond.opp_class_auc.t1"] - 0.5) < 0.20


# --------------------------------------------------------------------------- the own-team target

def test_the_leave_one_out_label_excludes_the_battle_it_labels() -> None:
    """Without the leave-one-out the label carries the outcome of its own battle, and any decoder
    that sees that battle's state scores against a target it was handed."""
    team = np.array(["A"] * 5 + ["B"] * 5)
    y = np.array([1.0, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    w = np.ones(10)
    loo = CM.loo_team_wr(team, y, w, min_battles=4)
    assert loo[0] == pytest.approx(3 / 4)       # A's other four: 1,1,1,0
    assert loo[4] == pytest.approx(4 / 4)       # A's other four: 1,1,1,1
    assert loo[9] == pytest.approx(0.0)         # B's other four: 0,0,0,0


def test_a_team_under_the_minimum_gets_no_label() -> None:
    loo = CM.loo_team_wr(np.array(["A", "A", "B"]), np.array([1.0, 0, 1]), np.ones(3),
                         min_battles=4)
    assert np.isnan(loo).all()


def test_the_out_of_fold_decode_never_trains_and_tests_on_the_same_battle() -> None:
    """A grouped fold map must put every state of a battle on the same side. Broken, a decoder
    memorises the battle and every R² on this page is a memorisation score."""
    groups = np.repeat(np.arange(30), 4)
    folds = CM._fold_of(groups, 5, seed=0)
    for g in range(30):
        assert np.unique(folds[groups == g]).size == 1


def test_a_constant_feature_column_predicts_the_mean_rather_than_dividing_by_zero() -> None:
    x = np.full(200, 0.5)
    y = np.linspace(0, 1, 200)
    p = CM.grouped_oof_scalar(x, y, np.ones(200), np.repeat(np.arange(50), 4),
                              task="r2", seed=0)
    assert np.isfinite(p).all()
    assert CM.score(y, p, np.ones(200), "r2") < 0.05


def test_a_perfectly_informative_scalar_decodes_and_an_uninformative_one_does_not() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(80), 4)
    y = rng.normal(size=80)[groups]
    good = CM.grouped_oof_scalar(y + rng.normal(0, 0.05, y.size), y, np.ones(y.size), groups,
                                 task="r2", seed=1)
    junk = CM.grouped_oof_scalar(rng.normal(size=y.size), y, np.ones(y.size), groups,
                                 task="r2", seed=1)
    assert CM.score(y, good, np.ones(y.size), "r2") > 0.95
    assert CM.score(y, junk, np.ones(y.size), "r2") < 0.10


# --------------------------------------------------------------------------- refusals

def test_a_cycle_without_a_manifest_refuses(tmp_path) -> None:
    (tmp_path / "eval_traces" / "step_100").mkdir(parents=True)
    with pytest.raises(CM.ConditioningRefusal, match="SELECTION UNKNOWN"):
        CM.extract_cycle(str(tmp_path / "eval_traces" / "step_100"))


def test_a_cycle_whose_manifest_has_no_selection_block_refuses(tmp_path) -> None:
    d = _plant(tmp_path, wr_by_opp={"heuristic": 0.5}, v_of=_marginal, schema=None, seed=9)
    with pytest.raises(CM.ConditioningRefusal, match="LOSS-ENRICHED"):
        CM.extract_cycle(d)


def test_a_battle_whose_outcome_class_has_no_capture_rate_is_dropped_never_weighted_one(
        tmp_path) -> None:
    """Rule 17's exact failure mode: mixing corrected rows with uncorrected ones and calling the
    result corrected. The row is dropped and the drop is REPORTED."""
    d = _plant(tmp_path, wr_by_opp={"heuristic": 0.5}, v_of=_marginal, n_battles=10, seed=10)
    man = json.loads(open(f"{d}/eval_manifest.json").read())
    man["selection"]["opponents"]["heuristic"]["capture_rate_loss"] = 0.0
    open(f"{d}/eval_manifest.json", "w").write(json.dumps(man))
    arr, meta = CM.extract_cycle(d)
    assert (arr["y"] == 1.0).all()
    assert sum("no capture rate" in r for r in meta["refusals"]) == 5


def test_an_unanchored_ladder_refit_is_refused_and_the_row_omitted(tmp_path, monkeypatch) -> None:
    """🚨 mixture-diagnostic hazard 1. `fit_ladder` silently returns an UNANCHORED ladder when it
    cannot find the bot pins — ratings near 1000 instead of near 2000, `anchored_to_bots` false,
    no error. Mixing those with the bot anchors puts one strength axis on two scales."""
    import agents.training.snapshot_ladder as SL

    d = _plant(tmp_path, wr_by_opp=_WRS, v_of=_marginal, seed=11)
    (tmp_path / "eval_results.jsonl").write_text(json.dumps(
        {"step": 100, "sentinels": [{"step": 50, "win_rate": 0.3},
                                    {"step": 60, "win_rate": 0.1}]}) + "\n")
    monkeypatch.setattr(SL, "load_games", lambda run: [(50, 60)])
    monkeypatch.setattr(SL, "fit_ladder",
                        lambda run, steps=None, write=False: {
                            "ratings": {"50": 1001.0, "60": 999.0}, "anchored_to_bots": False})
    per_opp = {o: {"true_win_rate": w} for o, w in _WRS.items()}
    axis, why = CM.strength_axis(str(tmp_path), 100, per_opp)
    assert axis is None
    assert "NOT anchored to the bot pins" in why and "OMITTED" in why
    assert d


def test_an_anchored_refit_with_a_verified_sentinel_map_produces_the_axis(
        tmp_path, monkeypatch) -> None:
    import agents.training.snapshot_ladder as SL

    _plant(tmp_path, wr_by_opp=_WRS, v_of=_separating, seed=12)
    (tmp_path / "eval_results.jsonl").write_text(json.dumps(
        {"step": 100, "sentinels": [{"step": 50, "win_rate": 0.3},
                                    {"step": 60, "win_rate": 0.1}]}) + "\n")
    monkeypatch.setattr(SL, "load_games", lambda run: [(50, 60)])
    monkeypatch.setattr(SL, "fit_ladder",
                        lambda run, steps=None, write=False: {
                            "ratings": {"50": 1907.0, "60": 2004.0}, "anchored_to_bots": True})
    per_opp = {o: {"true_win_rate": w} for o, w in _WRS.items()}
    axis, why = CM.strength_axis(str(tmp_path), 100, per_opp)
    assert axis is not None
    assert axis["sentinel_0"] == 1907.0 and axis["sentinel_1"] == 2004.0
    assert axis["heuristic"] > 1000.0            # from the committed bot anchors
    assert "verified" in why


def test_a_mismatched_sentinel_map_refuses_rather_than_guesses(tmp_path, monkeypatch) -> None:
    """The `sentinel_k` -> snapshot map is POSITIONAL and is verified against the manifest's own
    win counts. A mismatch means the positions have moved; guessing would silently label every
    sentinel cell with another snapshot's rating."""
    import agents.training.snapshot_ladder as SL

    _plant(tmp_path, wr_by_opp=_WRS, v_of=_separating, seed=13)
    (tmp_path / "eval_results.jsonl").write_text(json.dumps(
        {"step": 100, "sentinels": [{"step": 50, "win_rate": 0.77},
                                    {"step": 60, "win_rate": 0.1}]}) + "\n")
    monkeypatch.setattr(SL, "load_games", lambda run: [(50, 60)])
    monkeypatch.setattr(SL, "fit_ladder",
                        lambda run, steps=None, write=False: {
                            "ratings": {"50": 1907.0, "60": 2004.0}, "anchored_to_bots": True})
    axis, why = CM.strength_axis(str(tmp_path), 100,
                                 {o: {"true_win_rate": w} for o, w in _WRS.items()})
    assert axis is None and "MISMATCHES" in why


def test_the_frame_records_the_qc_that_v_is_the_win_prob_column(tmp_path) -> None:
    _plant(tmp_path, wr_by_opp=_WRS, v_of=_separating, seed=14)
    blk = CM.conditioning_block(str(tmp_path), 100, boot=100, ladder="off", seed=14)
    assert blk["frame"]["max_abs_values_minus_winprobs"] == 0.0
    assert blk["frame"]["selection_schema"] == 1
    assert "never pools cycles" in blk["frame"]["recorded_v_note"]


def test_every_declared_meter_is_either_reported_or_omitted_with_a_reason(tmp_path) -> None:
    _plant(tmp_path, wr_by_opp=_WRS, v_of=_separating, seed=15)
    blk = CM.conditioning_block(str(tmp_path), 100, boot=100, ladder="off", seed=15)
    for key in CM.METER_KEYS:
        assert (key in blk["points"]) ^ (key in blk["omitted"]), key
