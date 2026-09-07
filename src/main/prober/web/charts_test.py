"""The Vega-Lite specs are dicts, so their ENCODING is unit-testable without a browser.

What is checked here is the half a render test cannot see cheaply: that the spec plots the field
the caller thinks it does, that the data actually reached `data.values`, and that the two specs
carrying a judgement (the crater bracket, the reliability curve) keep the elements that make them
honest — the fixed lever order, and the identity rule the curve's deviation is read against.

`render_integration_test.py` checks the other half: that a browser turns these into marks.
"""

from __future__ import annotations

import json

import pytest

from main.prober.web import charts

_GATE = {"policy_reducible": 0.2, "aleatoric": 0.3, "unattributed": 0.4, "mixed": 0.1,
         "critic_headroom_upper_bound": 0.7}
_BINS = [{"v_lo": -10.0, "v_hi": -5.0, "v_mean": -7.0, "g_mean": -9.0, "n": 12, "gap": 2.0},
         {"v_lo": -5.0, "v_hi": 2.0, "v_mean": -1.0, "g_mean": -1.5, "n": 11, "gap": 0.5},
         {"v_lo": 2.0, "v_hi": 9.0, "v_mean": 5.0, "g_mean": 6.0, "n": 10, "gap": -1.0}]


def _values(spec):
    return spec["data"]["values"]


def _subtitle(spec) -> str:
    """Subtitles are pre-wrapped into a list of lines (Vega does not wrap title text and a long
    one widens the chart past its container). Rejoin for content assertions."""
    sub = spec["title"]["subtitle"]
    return sub if isinstance(sub, str) else " ".join(sub)


def test_every_spec_declares_the_vega_lite_schema():
    """A spec without `$schema` embeds as a *Vega* spec and silently renders nothing useful."""
    for spec in (charts.crater_bracket_spec(_GATE),
                 charts.reliability_curve_spec(_BINS),
                 charts.reliability_gap_spec(_BINS),
                 charts.outcome_by_step_spec([{"step": 1, "totals": {"win": 1, "loss": 2}}]),
                 charts.triage_lever_spec([{"category": "c", "est_recoverable_winrate_pct": 1.0,
                                            "n": 2, "lever": "obs"}]),
                 charts.scan_drop_spec([{"short_id": "a", "opponent": "o",
                                         "worst": {"delta_v": -3.0, "turn": 4, "chosen": "x"}}])):
        assert spec["$schema"].endswith("/vega-lite/v5.json")
        assert spec["width"] == "container", "a fixed width will overflow the card on a phone"


def test_crater_bracket_keeps_the_fixed_lever_order_and_sums_to_one():
    spec = charts.crater_bracket_spec(_GATE)
    assert [v["lever"] for v in _values(spec)] == charts.LEVER_ORDER
    assert sum(v["share"] for v in _values(spec)) == pytest.approx(1.0)
    # `critic_headroom_upper_bound` is a DERIVED total, not a fifth lever — stacking it would
    # double-count aleatoric and unattributed and make the bar sum to 1.7.
    assert "critic_headroom_upper_bound" not in [v["lever"] for v in _values(spec)]
    assert spec["encoding"]["x"]["stack"] == "zero"


def test_crater_bracket_legend_carries_the_interpretation():
    """The raw keys are opaque and two of them are routinely over-read. The legend text is the
    only thing standing between `unattributed` and being quoted as 'critic error'."""
    labels = [v["label"] for v in _values(charts.crater_bracket_spec(_GATE))]
    assert any("PROVEN" in x for x in labels)
    assert any("NOT proven" in x for x in labels)


def test_crater_bracket_tolerates_a_partial_gate():
    """`calibration` passes `falsify_gate`, which carries a different key set."""
    spec = charts.crater_bracket_spec({"policy_reducible": 0.6, "aleatoric": 0.4})
    assert [v["lever"] for v in _values(spec)] == ["policy_reducible", "aleatoric"]


def test_reliability_curve_plots_g_against_v_over_an_identity_rule():
    spec = charts.reliability_curve_spec(_BINS)
    assert _values(spec)[0]["v_mean"] == -7.0 and _values(spec)[0]["g_mean"] == -9.0
    ident, curve, points = spec["layer"]
    # The identity layer is what makes the deviation readable; if it plots v against g it is just
    # a second copy of the curve and the chart says nothing.
    assert ident["encoding"]["x"]["field"] == ident["encoding"]["y"]["field"] == "v"
    assert ident["mark"]["strokeDash"], "the identity rule must be visually distinct from the curve"
    assert curve["encoding"]["y"]["field"] == "g_mean"
    assert points["encoding"]["size"]["field"] == "n", "bin weight must be visible"


def test_reliability_gap_diverges_around_zero():
    spec = charts.reliability_gap_spec(_BINS)
    assert spec["encoding"]["color"]["scale"]["domainMid"] == 0, (
        "a sequential ramp would paint a −1.0 gap and a +0.5 gap as the same kind of thing")
    assert [v["gap"] for v in _values(spec)] == [2.0, 0.5, -1.0]


def test_outcome_by_step_is_labelled_as_capture_counts_not_a_win_rate():
    """Eval capture is loss-weighted, so a stacked win/loss bar is exactly the chart someone
    would otherwise read as progress."""
    spec = charts.outcome_by_step_spec([{"step": 100, "totals": {"win": 3, "loss": 7}}])
    assert "not a win rate" in _subtitle(spec).lower()
    assert _values(spec) == [{"step": 100, "outcome": "win", "n": 3},
                             {"step": 100, "outcome": "loss", "n": 7}]


def test_triage_lever_spec_sorts_by_the_ranking_metric():
    cats = [{"category": "a", "est_recoverable_winrate_pct": 1.0, "n": 3, "lever": "obs"},
            {"category": "b", "est_recoverable_winrate_pct": 5.0, "n": 9, "lever": "critic"}]
    spec = charts.triage_lever_spec(cats)
    assert spec["encoding"]["y"]["sort"] == "-x", "the lever ORDER is the whole point of triage"
    assert "UPPER BOUND" in _subtitle(spec)


def test_scan_drop_spec_follows_the_metric_and_drops_rows_with_no_worst():
    rows = [{"short_id": "a", "opponent": "o", "worst": {"delta_v": -8.0, "td_residual": -9.0,
                                                         "turn": 1, "chosen": "eq"}},
            {"short_id": "b", "opponent": "o", "worst": {}},
            {"short_id": "c", "opponent": "o", "worst": None}]
    by_dv = charts.scan_drop_spec(rows, metric="value_drop")
    assert [v["battle"] for v in _values(by_dv)] == ["a"]
    assert _values(by_dv)[0]["drop"] == -8.0
    by_td = charts.scan_drop_spec(rows, metric="td_residual")
    assert _values(by_td)[0]["drop"] == -9.0


def test_long_subtitles_are_pre_wrapped_so_a_chart_cannot_widen_its_container():
    """Vega does NOT wrap title text. A 150-character subtitle renders as one ~1000px line and
    drags the whole chart past its container — which on a phone is a page that scrolls sideways.
    So anything long arrives as a list of lines, each within the wrap width."""
    spec = charts.reliability_curve_spec(_BINS)
    sub = spec["title"]["subtitle"]
    assert isinstance(sub, list) and len(sub) > 1, "a long subtitle must arrive pre-wrapped"
    assert all(len(line) <= charts._SUBTITLE_WRAP for line in sub)
    assert "OVER-valued" in " ".join(sub), "wrapping must not drop content"


def test_a_short_subtitle_is_left_as_a_plain_string():
    """Wrapping a subtitle that already fits would only add churn to the spec diff."""
    spec = charts.reliability_gap_spec(_BINS)
    short = charts._subtitle("a short one")
    assert isinstance(short, str)
    assert isinstance(spec["title"]["subtitle"], (str, list))


def test_specs_are_json_serializable():
    """They are embedded into a <script> tag; a numpy scalar in here fails at render time only."""
    import json
    json.dumps(charts.crater_bracket_spec(_GATE))
    json.dumps(charts.reliability_curve_spec(_BINS))


# --- the reliability charts follow the CRITIC'S CURRENCY -----------------------------------
# On a win-prob critic the curve is a PROBABILITY CALIBRATION (predicted P(win) vs realized win
# rate), not "V against a return". Labelling it as a return invites the reader to look for a
# shaped scale on axes that run 0..1 — the same units confusion `overvalue_tau` encoded as a
# number, restated in prose.

_WINPROB = {"mode": "winprob", "units": "P(win)", "is_probability": True, "span": 1.0}
_SHAPED = {"mode": "shaped", "units": "shaped return", "is_probability": False, "span": 60.0}


def _axis_titles(spec):
    enc = spec.get("encoding", {})
    return (enc.get("x", {}).get("axis", {}).get("title"),
            enc.get("y", {}).get("axis", {}).get("title"))


def test_the_reliability_curve_is_labelled_a_RETURN_on_a_shaped_critic():
    for cur in (None, _SHAPED):                      # None = the historical call, unchanged
        x, y = _axis_titles(charts.reliability_curve_spec(_BINS, cur))
        assert x == "recorded V(s)" and y == "realized return G(s)"


def test_the_reliability_curve_is_labelled_a_PROBABILITY_on_a_winprob_critic():
    spec = charts.reliability_curve_spec(_BINS, _WINPROB)
    x, y = _axis_titles(spec)
    assert x == "predicted P(win)" and y == "realized win rate"
    blob = json.dumps(spec)
    assert "realized return" not in blob, "a probability calibration must not claim to plot a return"


def test_the_gap_chart_follows_the_same_currency():
    x_s, y_s = _axis_titles(charts.reliability_gap_spec(_BINS, _SHAPED))
    x_w, y_w = _axis_titles(charts.reliability_gap_spec(_BINS, _WINPROB))
    assert x_s == "recorded V(s)" and y_s == "gap V − G"
    # a difference of two probabilities is percentage POINTS, not a return
    assert x_w == "predicted P(win)" and y_w == "gap (pp)"
