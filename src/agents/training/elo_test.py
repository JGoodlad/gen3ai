"""Unit tests for the Bradley-Terry rating engine (`elo.py`).

Pure (no server, no torch) — generates synthetic eval rows from a KNOWN ground-truth
ladder and asserts the fit recovers it, plus the anchoring / robustness contracts.
"""
import json
import math

import numpy as np
import pytest

from agents.training import elo
from agents.training.elo import (
    EvalRow, fit_elo, load_rows, load_bot_anchors, _S,
    bot_key, snap_key, snapshot_step,
)


def _winrate(r_subject: float, r_opp: float) -> float:
    """Exact BT win probability for `subject` vs `opp` on the Elo scale."""
    return 1.0 / (1.0 + math.exp(-_S * (r_subject - r_opp)))


def _synthetic_rows(bot_elo: dict[str, float], snap_elo: dict[int, float],
                    n_games: int = 4000, with_sentinels: bool = True) -> list[EvalRow]:
    """Build noise-free eval rows: each snapshot plays every bot, and (optionally) every
    earlier snapshot as a sentinel — all at the EXACT BT win probability for the true
    ratings. With large n_games the MLE should recover the true ladder to within a point."""
    steps = sorted(snap_elo)
    rows = []
    for i, step in enumerate(steps):
        bots = {name: _winrate(snap_elo[step], r) for name, r in bot_elo.items()}
        sentinels = []
        if with_sentinels:
            for prev in steps[:i]:
                sentinels.append((prev, _winrate(snap_elo[step], snap_elo[prev])))
        rows.append(EvalRow(step, n_games, bots, sentinels))
    return rows


def test_recovers_known_ladder_with_pinned_bots():
    bot_elo = {"random": 1000.0, "heuristic": 1300.0, "staller": 1250.0,
               "aggressive": 1280.0, "setup_sweep": 1310.0}
    snap_elo = {1_000_000: 1200.0, 5_000_000: 1450.0, 10_000_000: 1620.0,
                20_000_000: 1750.0}
    rows = _synthetic_rows(bot_elo, snap_elo)

    fit = fit_elo(rows, pin_ratings=bot_elo)

    # Bots stay pinned exactly.
    for name, r in bot_elo.items():
        assert fit.ratings[bot_key(name)] == pytest.approx(r, abs=1e-9)
    # Snapshots recovered to within a couple Elo (prior pull is negligible at n=4000).
    for step, r in snap_elo.items():
        assert fit.ratings[snap_key(step)] == pytest.approx(r, abs=3.0)
    # Monotone curve in step order, exact rank.
    curve = fit.snapshot_curve()
    assert [s for s, _e, _se in curve] == sorted(snap_elo)
    elos = [e for _s, e, _se in curve]
    assert elos == sorted(elos)
    assert fit.converged


def test_snapshot_curve_rises_for_improving_agent():
    bot_elo = {"random": 1000.0, "heuristic": 1300.0}
    snap_elo = {n * 1_000_000: 1100.0 + 40.0 * n for n in range(1, 8)}  # strictly rising
    fit = fit_elo(_synthetic_rows(bot_elo, snap_elo), pin_ratings=bot_elo)
    elos = [e for _s, e, _se in fit.snapshot_curve()]
    assert elos == sorted(elos)
    assert elos[-1] - elos[0] > 200  # clear upward progress, the whole point


def test_pin_ratings_held_rigid():
    bot_elo = {"random": 1000.0, "heuristic": 1500.0}
    snap_elo = {1_000_000: 1400.0}
    fit = fit_elo(_synthetic_rows(bot_elo, snap_elo), pin_ratings=bot_elo)
    assert bot_key("random") in fit.pinned and bot_key("heuristic") in fit.pinned
    assert fit.ratings[bot_key("heuristic")] == pytest.approx(1500.0, abs=1e-9)
    assert fit.ratings[snap_key(1_000_000)] == pytest.approx(1400.0, abs=3.0)


def test_perfect_record_stays_finite():
    # A snapshot that beats every bot 100-0: without a prior its rating is +inf.
    bots = {"random": 1000.0, "heuristic": 1300.0}
    rows = [EvalRow(1_000_000, 100, {"random": 1.0, "heuristic": 1.0}, [])]
    fit = fit_elo(rows, pin_ratings=bots)
    r = fit.ratings[snap_key(1_000_000)]
    assert math.isfinite(r)
    assert r > 1300.0  # clearly stronger than the bots it swept
    assert r < 4000.0  # but the prior keeps it bounded


def test_no_anchor_pins_random_at_base():
    # Without anchors, `random` is pinned at `base` to fix the gauge.
    snap_elo = {1_000_000: 1200.0, 5_000_000: 1400.0}
    bot_elo = {"random": 1000.0, "heuristic": 1300.0}
    rows = _synthetic_rows(bot_elo, snap_elo)
    fit = fit_elo(rows, pin_ratings=None, base=1000.0)
    assert bot_key("random") in fit.pinned
    assert fit.ratings[bot_key("random")] == pytest.approx(1000.0, abs=1e-9)
    # heuristic is now FREE (not pinned) and recovered relative to random.
    assert bot_key("heuristic") not in fit.pinned
    assert fit.ratings[bot_key("heuristic")] == pytest.approx(1300.0, abs=15.0)


def test_standard_errors_present_and_shrink_with_games():
    bot_elo = {"random": 1000.0, "heuristic": 1300.0}
    snap_elo = {1_000_000: 1400.0}
    se_small = fit_elo(_synthetic_rows(bot_elo, snap_elo, n_games=100),
                       pin_ratings=bot_elo).se[snap_key(1_000_000)]
    se_large = fit_elo(_synthetic_rows(bot_elo, snap_elo, n_games=4000),
                       pin_ratings=bot_elo).se[snap_key(1_000_000)]
    assert se_small > 0 and se_large > 0
    assert se_large < se_small  # more games → tighter CI


def test_empty_rows():
    fit = fit_elo([], pin_ratings={"random": 1000.0})
    assert fit.ratings == {} and fit.converged


def test_dedup_by_step_last_wins(tmp_path):
    path = tmp_path / "eval_results.jsonl"
    rows = [
        {"step": 1000, "n_games": 100, "bots": {"random": 0.5}, "sentinels": []},
        {"step": 1000, "n_games": 100, "bots": {"random": 0.9}, "sentinels": []},  # newer
        {"step": 2000, "n_games": 100, "bots": {"random": 0.8}, "sentinels": []},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    loaded = load_rows(str(tmp_path), source="log")
    assert [r.step for r in loaded] == [1000, 2000]
    assert loaded[0].bots["random"] == 0.9  # last write for step 1000 wins


def test_log_roundtrip_with_sentinels(tmp_path):
    path = tmp_path / "eval_results.jsonl"
    path.write_text(json.dumps({
        "step": 5_000_000, "n_games": 100,
        "bots": {"random": 0.99, "heuristic": 0.7},
        "sentinels": [{"step": 1_000_000, "win_rate": 0.55}],
    }) + "\n")
    rows = load_rows(str(tmp_path), source="log")
    assert len(rows) == 1
    r = rows[0]
    assert r.step == 5_000_000 and r.n_games == 100
    assert r.bots == {"random": 0.99, "heuristic": 0.7}
    assert r.sentinels == [(1_000_000, 0.55)]


def test_load_bot_anchors(tmp_path):
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({"ratings": {"random": 1000.0, "heuristic": 1320.0},
                             "se": {"random": 5.0}, "base": 1000.0}))
    anchors = load_bot_anchors(str(p))
    assert anchors["ratings"]["heuristic"] == 1320.0
    assert load_bot_anchors(str(tmp_path / "missing.json")) is None


def test_snapshot_step_roundtrip():
    assert snapshot_step(snap_key(12_345_678)) == 12_345_678


def test_elo_from_eval_block():
    # A single metadata `latest_eval` block (no `elo` field) → compute (elo, se) from its win
    # rates. This is the resume-republish path for a checkpoint saved before the elo field.
    block = {
        "step": 5_000_000,
        "opponents": {"random": {"win_rate": 0.99}, "heuristic": {"win_rate": 0.8},
                      "staller": {"win_rate": 0.75}},
        "pool": {"sentinels": [{"step": 1_000_000, "win_rate": 0.6}]},
    }
    # Force the no-anchor fallback so the test doesn't depend on the committed anchor file.
    r = elo.elo_from_eval_block(block, anchors_path="/tmp/__no_such_anchor__.json")
    assert r is not None
    e, se = r
    assert math.isfinite(e) and se >= 0
    assert e > 1000.0  # beats random handily → above the base anchor
    # Degenerate blocks → None (no usable bot data / no step).
    assert elo.elo_from_eval_block({"step": 1, "opponents": {}}) is None
    assert elo.elo_from_eval_block({"opponents": {"random": {"win_rate": 0.5}}}) is None


def test_anchored_fit_is_translation_equivariant():
    # Shifting ALL bot anchors by a constant C shifts every snapshot by C (pairwise win rates
    # are unchanged), and the fit must stay finite — this is the regression guard for the
    # full-step Newton divergence that damped (line-searched) Newton fixes.
    bot_elo = {"random": 1000.0, "heuristic": 1300.0, "staller": 1250.0,
               "aggressive": 1280.0, "setup_sweep": 1310.0}
    snap_elo = {1_000_000: 1200.0, 8_000_000: 1500.0, 20_000_000: 1700.0}
    rows = _synthetic_rows(bot_elo, snap_elo)
    C = 500.0
    shifted = {k: v + C for k, v in bot_elo.items()}
    fit = fit_elo(rows, pin_ratings=shifted)
    assert fit.converged
    for step, r in snap_elo.items():
        val = fit.ratings[snap_key(step)]
        assert math.isfinite(val)
        assert val == pytest.approx(r + C, abs=5.0)  # snapshots track the anchor shift


def test_inconsistent_anchors_stay_finite():
    # Anchors wildly inconsistent with the data must not blow up to ±1e5 (the original bug).
    bot_elo = {"random": 1000.0, "heuristic": 1300.0, "staller": 1250.0}
    snap_elo = {1_000_000: 1180.0, 6_000_000: 1560.0, 20_000_000: 1790.0}
    rows = _synthetic_rows(bot_elo, snap_elo)
    bad = {"random": 1000.0, "heuristic": 1640.0, "staller": 1530.0}  # compressed/reordered
    fit = fit_elo(rows, pin_ratings=bad)
    assert fit.converged
    for step in snap_elo:
        v = fit.ratings[snap_key(step)]
        assert math.isfinite(v) and -5000 < v < 5000
    # Trend preserved despite the mismatched anchor.
    curve = [e for _s, e, _se in fit.snapshot_curve()]
    assert curve == sorted(curve)


def test_fit_pairwise_round_robin_recovers_ratings():
    # Bot-vs-bot round-robin (the anchor calibration primitive): exact win rates from a
    # known ladder, random pinned at base → recover the others.
    true = {"random": 1000.0, "heuristic": 1320.0, "staller": 1250.0, "aggressive": 1400.0}
    n_games = 5000
    results = []
    names = list(true)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wa = int(round(_winrate(true[a], true[b]) * n_games))
            results.append((a, b, wa, n_games))
    ratings, se, converged = elo.fit_pairwise(results, pinned={"random": 1000.0})
    assert converged
    assert ratings["random"] == pytest.approx(1000.0, abs=1e-9)
    for name, r in true.items():
        assert ratings[name] == pytest.approx(r, abs=10.0)
    assert all(se[n] >= 0 for n in names) and se["heuristic"] > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
