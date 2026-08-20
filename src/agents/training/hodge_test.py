"""Tests for the Hodge (transitive/cyclic) decomposition of a rating graph.

Each test pins one property that, if it broke, would turn the width into a number that reads
like a measurement and is not one:

* a transitive matrix must read ZERO width (else every ladder looks non-transitive);
* a real cycle must be FOUND and NAMED (else the instrument cannot see the thing it is for);
* the energy split must be exactly orthogonal (the algebra that lets "spine + width" be a
  decomposition rather than two loosely related numbers);
* an incomplete graph must stay finite (the live graph is always incomplete);
* pure sampling noise must NOT clear the floor, and its p-values must be roughly uniform
  (the whole point of having a floor at all);
* pendant edges must be excluded from the width scope (their residual is identically zero, so
  including them silently deflates every width);
* the live recorder must emit both scalars when triangles exist and OMIT them when they do
  not — a star graph has no cycles, so a star-only width would be a fake instrument.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from agents.training import elo as elo_mod
from agents.training import hodge as H


# ── helpers ───────────────────────────────────────────────────────────────────


def _p(delta: float) -> float:
    return 1.0 / (1.0 + math.exp(-delta))


def _transitive_results(ratings: dict[str, float], games: int = 200, rng=None):
    """Every pair, played ``games`` times, from a purely transitive truth (logit ratings).
    With ``rng`` the wins are binomial samples; without, they are the exact expectation."""
    names = sorted(ratings)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = _p(ratings[a] - ratings[b])
            wins = int(rng.binomial(games, p)) if rng is not None else int(round(p * games))
            out.append((a, b, wins, games))
    return out


# ── the transitive case ───────────────────────────────────────────────────────


def test_a_purely_transitive_matrix_has_no_cyclic_width():
    """The null case. If this ever reads a width, every ladder reads non-transitive."""
    fit = H.hodge_decompose(_transitive_results({"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}),
                            bootstrap=100, seed=0)
    assert fit is not None
    assert fit.cyclic_energy_fraction < 1e-3, fit.cyclic_energy_fraction
    assert fit.width_rms_excess == 0.0
    assert fit.cycles == []
    # The spine must also be RIGHT, not merely non-cyclic: order and gaps recover the truth.
    r = fit.ratings
    assert r["d"] > r["c"] > r["b"] > r["a"]
    assert fit.spine_spread == pytest.approx(3.0, abs=0.05)


def test_the_energy_split_is_orthogonal_on_a_complete_graph():
    """Σw·Y² = Σw·(r_i−r_j)² + Σw·R², to numerical precision.

    This identity is what makes "spine" and "width" a DECOMPOSITION — the two parts are
    w-orthogonal, so the width cannot be traded against the spine by refitting. It follows
    from the normal equations, so it must hold to machine precision, not approximately."""
    rng = np.random.default_rng(3)
    res = _transitive_results({"a": 0.0, "b": 0.7, "c": 1.4, "d": 2.1, "e": 2.8}, 300, rng)
    res += [("a", "c", 250, 300)]                       # deliberately jam one edge cyclic
    fit = H.hodge_decompose(res, bootstrap=0)
    assert fit is not None
    total = fit.energy_transitive + fit.energy_cyclic
    assert fit.energy_total == pytest.approx(total, rel=1e-10), (
        f"the split is not orthogonal: {fit.energy_total} != {total}")


# ── the cyclic case ───────────────────────────────────────────────────────────


def _rps_in_a_transitive_field(games: int = 400):
    """Three equal-strength players in a 75/25 rock-paper-scissors loop, embedded in a
    transitive field (a weak player below, a strong one above). The cycle must survive the
    surrounding transitive structure — a real ladder is mostly spine."""
    res = [("x", "y", int(0.75 * games), games),
           ("y", "z", int(0.75 * games), games),
           ("z", "x", int(0.75 * games), games)]
    for outer, r in (("lo", -2.0), ("hi", 2.0)):
        for mid in ("x", "y", "z"):
            res.append((outer, mid, int(round(_p(r) * games)), games))
    res.append(("lo", "hi", int(round(_p(-4.0) * games)), games))
    return res


def test_an_rps_triple_is_found_by_name_and_clears_the_noise_floor():
    fit = H.hodge_decompose(_rps_in_a_transitive_field(), bootstrap=200, seed=0)
    assert fit is not None
    assert fit.width_rms_excess_elo > 50, fit.width_rms_excess_elo
    assert fit.p_value is not None and fit.p_value < 0.05
    assert fit.p_value_plugin < 0.05
    assert fit.cycles, "the RPS triangle was not reported"
    named = {frozenset(c.players) for c in fit.cycles}
    assert frozenset({"x", "y", "z"}) in named, [c.describe() for c in fit.cycles]
    top = fit.cycles[0]
    assert set(top.players) == {"x", "y", "z"}
    # Reported in BEAT order, so following the names round the loop is a true statement.
    a, b, c = top.players
    assert (a, b, c) in {("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y")}
    assert top.curl_z > 3.0
    assert top.games == (400, 400, 400)


def test_the_transitive_players_around_the_cycle_are_not_reported_as_cycles():
    """A cycle report must not fire on the transitive triangles that surround it — otherwise
    'significant 3-cycles' is a list of everything."""
    fit = H.hodge_decompose(_rps_in_a_transitive_field(), bootstrap=0)
    assert fit is not None
    for c in fit.cycles:
        assert set(c.players) == {"x", "y", "z"}, c.describe()


# ── incomplete graphs ─────────────────────────────────────────────────────────


def test_an_incomplete_graph_stays_finite_and_recovers_the_order():
    """The live graph is never complete. Missing pairs must leave the solve finite and the
    surviving order intact — a pseudo-inverse on a connected graph, not a singular blow-up."""
    ratings = {"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0, "e": 4.0}
    res = [r for r in _transitive_results(ratings, 200)
           if {r[0], r[1]} not in ({"a", "d"}, {"b", "e"}, {"a", "e"})]
    fit = H.hodge_decompose(res, bootstrap=50, seed=0)
    assert fit is not None
    assert fit.n_components == 1
    assert all(math.isfinite(v) for v in fit.ratings.values())
    order = [p for p, _ in sorted(fit.ratings.items(), key=lambda kv: kv[1])]
    assert order == ["a", "b", "c", "d", "e"]
    assert fit.width_rms_excess == 0.0


def test_a_disconnected_graph_reports_its_components_with_a_caveat():
    res = [("a", "b", 150, 200), ("c", "d", 150, 200)]
    fit = H.hodge_decompose(res, width_scope="all", bootstrap=0)
    assert fit is not None
    assert fit.n_components == 2
    assert any("disconnected" in c for c in fit.caveats), fit.caveats


def test_a_pendant_edge_is_excluded_from_the_width_scope():
    """A player with ONE opponent has residual identically zero (its node equation IS that
    edge). Counting it would inflate Σw and silently deflate every width toward zero — which
    is exactly what a live cycle's sentinel edges would do."""
    res = _rps_in_a_transitive_field()
    res.append(("pendant", "x", 30, 100))
    fit = H.hodge_decompose(res, bootstrap=0)
    assert fit is not None
    assert fit.n_width_edges == len(res) - 1, "the pendant edge was counted"
    wide = H.hodge_decompose(res, width_scope="all", bootstrap=0)
    assert wide is not None
    assert wide.width_rms_raw < fit.width_rms_raw, (
        "including the pendant edge did not deflate the width — the scope rule is not doing "
        "the job it exists for")


# ── the noise floor ───────────────────────────────────────────────────────────


def test_pure_sampling_noise_does_not_clear_the_floor():
    """Games simulated FROM a transitive truth: any width here is binomial noise, and the
    instrument must say so. Fixed seeds — this is a regression test, not a study."""
    rng = np.random.default_rng(7)
    truth = {p: 1.2 * i for i, p in enumerate("abcdef")}
    excesses, ps = [], []
    for t in range(12):
        fit = H.hodge_decompose(_transitive_results(truth, 150, rng), bootstrap=150, seed=t)
        assert fit is not None
        excesses.append(fit.width_rms_excess_elo)
        ps.append(fit.p_value)
    assert max(ps) <= 1.0 and min(ps) > 0.05, f"a null draw was called significant: {ps}"
    assert float(np.median(excesses)) < 20.0, excesses
    # For scale: the real cycle above reads > 50 ELO of excess on the same kind of graph.


def test_null_p_values_are_roughly_uniform():
    """A p-value that is not uniform under the null is not a p-value. Two fixed data seeds,
    25 draws each; the bar is deliberately loose (this catches a broken null, not a 3% bias)."""
    truth = {p: 1.2 * i for i, p in enumerate("abcdef")}
    for data_seed in (7, 11):
        rng = np.random.default_rng(data_seed)
        ps = []
        for t in range(25):
            fit = H.hodge_decompose(_transitive_results(truth, 150, rng), bootstrap=120, seed=t)
            assert fit is not None
            ps.append(fit.p_value)
        arr = np.array(ps)
        assert 0.3 < arr.mean() < 0.7, f"seed {data_seed}: mean p {arr.mean():.3f}"
        assert (arr < 0.05).mean() <= 0.08, f"seed {data_seed}: {(arr < 0.05).mean():.2f} < 0.05"
        assert (arr > 0.5).any() and (arr < 0.5).any(), f"seed {data_seed}: degenerate p"


def test_the_plugin_null_agrees_with_the_bootstrap_null():
    """Two independent nulls — an exact-mean analytic one and a full-pipeline resample. They
    must land in the same place; a gap means one of them is wrong."""
    rng = np.random.default_rng(5)
    fit = H.hodge_decompose(_transitive_results({p: 1.0 * i for i, p in enumerate("abcde")},
                                                200, rng), bootstrap=300, seed=0)
    assert fit is not None
    assert fit.width_rms_null == pytest.approx(fit.width_rms_null_plugin, rel=0.25)


def test_the_read_is_deterministic_for_a_fixed_seed():
    res = _rps_in_a_transitive_field()
    a = H.hodge_decompose(res, bootstrap=80, seed=3)
    b = H.hodge_decompose(res, bootstrap=80, seed=3)
    assert a is not None and b is not None
    assert a.to_json() == b.to_json()


# ── the anchor round-robin ────────────────────────────────────────────────────


def test_the_real_anchor_file_yields_measured_bot_edges():
    """The shipped anchor DOES carry the raw round-robin (a 9×9 ``win_matrix`` + per-pair
    game counts), so the live triangles rest on measured edges rather than reconstructed
    ones. If a future anchor drops them, the fallback below is what runs instead."""
    anchors = elo_mod.load_bot_anchors()
    if not anchors or not anchors.get("win_matrix"):
        pytest.skip("no bot anchor file in this checkout")
    edges, caveats = H.anchor_round_robin_edges(anchors)
    assert len(edges) == 36, len(edges)
    assert caveats == []
    assert all(g > 0 and 0 <= w <= g for _a, _b, w, g in edges)


def test_the_reconstruction_fallback_is_transitive_and_says_so():
    """Ratings-only anchors: edges are rebuilt from the ratings, which makes them transitive
    BY CONSTRUCTION. That is usable (it pins the bot frame) but it is a different claim, so
    it must be flagged."""
    edges, caveats = H.anchor_round_robin_edges({"ratings": {"random": 1000.0,
                                                             "heuristic": 1500.0,
                                                             "staller": 1400.0}},
                                                prior_games=500)
    assert len(edges) == 3
    assert any("RECONSTRUCTED" in c for c in caveats), caveats
    fit = H.hodge_decompose(edges, bootstrap=0)
    assert fit is not None
    assert fit.cyclic_energy_fraction < 1e-4


# ── the live recorder ─────────────────────────────────────────────────────────


class _Logger:
    def __init__(self):
        self.recorded: dict[str, float] = {}

    def record(self, key, value):
        self.recorded[key] = value


def _bots(win_rates: dict) -> dict:
    return dict(win_rates)


def _anchor_fixture(tmp_path) -> str:
    """A tiny 3-bot round-robin in the anchor file's real shape."""
    path = os.path.join(tmp_path, "anchors.json")
    with open(path, "w") as f:
        json.dump({"base": 1000.0,
                   "ratings": {"random": 1000.0, "heuristic": 1500.0, "staller": 1400.0},
                   "win_matrix": {"random": {"heuristic": 0.05, "staller": 0.08},
                                  "heuristic": {"random": 0.95, "staller": 0.55},
                                  "staller": {"random": 0.92, "heuristic": 0.45}},
                   "pair_games": {"heuristic vs random": 1000, "random vs staller": 1000,
                                  "heuristic vs staller": 1000}}, f)
    return path


def test_the_live_recorder_emits_both_scalars_when_triangles_exist(tmp_path):
    path = _anchor_fixture(tmp_path)
    log = _Logger()
    block = H.record_live_hodge(log, 12_000_000,
                                _bots({"random": 0.99, "heuristic": 0.62, "staller": 0.80}),
                                [{"step": 8_000_000, "win_rate": 0.55}], 100,
                                anchors_path=path, bootstrap=40)
    assert block["recorded"] is True
    assert set(log.recorded) == {H.WIDTH_TAG, H.FRACTION_TAG}
    assert all(math.isfinite(v) for v in log.recorded.values())
    assert log.recorded[H.WIDTH_TAG] >= 0.0
    assert block["width_elo"] == pytest.approx(log.recorded[H.WIDTH_TAG], abs=0.01)
    assert block["cyclic_fraction"] == pytest.approx(log.recorded[H.FRACTION_TAG], abs=1e-4)
    assert block["n_triangles"] > 0
    # The width is the TRAINEE's: 3 bot edges, not the 3 anchor edges and not the sentinel.
    assert block["n_width_edges"] == 3


def test_the_live_recorder_omits_the_record_when_the_graph_is_a_star(tmp_path):
    """THE fake-instrument guard. Without the bot round-robin the cycle's own games are a
    tree; a tree has no cycles, so a width computed there is identically zero and would read
    as 'no non-transitivity' forever. It must be OMITTED, and the omission must be counted."""
    log = _Logger()
    block = H.record_live_hodge(log, 12_000_000,
                                _bots({"random": 0.99, "heuristic": 0.62, "staller": 0.80}),
                                [{"step": 8_000_000, "win_rate": 0.55}], 100,
                                anchors_path=os.path.join(tmp_path, "absent.json"),
                                bootstrap=20)
    assert block["recorded"] is False
    assert log.recorded == {}, "a star graph produced a width scalar"
    assert block["caveats"], "the omission was not explained"


def test_a_star_alone_has_no_triangle_at_all():
    """The structural fact the recorder rests on, asserted directly."""
    res, subj, _c = H.cycle_edges(12_000_000, {"a": 0.6, "b": 0.7}, [{"step": 4, "win_rate": 0.5}],
                                  100, anchors=None)
    fit = H.hodge_decompose(res, width_incident_to={subj}, bootstrap=0)
    assert fit is None or fit.n_triangles == 0


def test_sentinel_edges_join_the_fit_but_not_the_width(tmp_path):
    """Sentinels carry real spine information and must be fit; they lie on no triangle, so
    they must not enter a width claim."""
    path = _anchor_fixture(tmp_path)
    fit = H.live_read(12_000_000, {"random": 0.99, "heuristic": 0.62, "staller": 0.80},
                      [{"step": 8_000_000, "win_rate": 0.55}], 100,
                      anchors_path=path, bootstrap=0)
    assert fit is not None
    assert elo_mod.snap_key(8_000_000) in fit.ratings          # in the fit
    assert fit.n_width_edges == 3                              # not in the width


def test_the_eval_row_carries_the_hodge_block(tmp_path):
    """The two numbers must survive into ``eval_results.jsonl`` for offline replotting, and a
    suppressed record must be distinguishable there from a cycle that never ran."""
    from agents.model.snapshot import append_eval_result_row
    run = str(tmp_path)
    append_eval_result_row(run, 1000, 100, {"heuristic": 0.8}, [],
                           hodge={"recorded": True, "width_elo": 12.5, "cyclic_fraction": 0.01})
    append_eval_result_row(run, 2000, 100, {"heuristic": 0.8}, [],
                           hodge={"recorded": False, "caveats": ["no testable triangle"]})
    append_eval_result_row(run, 3000, 100, {"heuristic": 0.8}, [])
    rows = [json.loads(x) for x in open(os.path.join(run, "eval_results.jsonl"))]
    assert rows[0]["hodge"]["width_elo"] == 12.5
    assert rows[1]["hodge"]["recorded"] is False
    assert "hodge" not in rows[2], "a row with no read must stay byte-identical to before"
    # The row still parses as an ELO row — the new field is additive.
    assert elo_mod._row_from_block(rows[0]).bots == {"heuristic": 0.8}
