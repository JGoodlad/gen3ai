"""Vega-Lite specs, built in Python from a `ProbeSession` result.

WHY SPECS AND NOT PLOTTING CODE. A chart written as imperative JS is only checkable by looking at
it. A Vega-Lite spec is a **dict** — so it diffs, it snapshots, and a unit test can assert "the
reliability curve plots `v_mean` against `g_mean` and carries an identity reference rule" without a
browser. The browser's only job is to turn the spec into pixels, and that half is what
`render_integration_test.py` checks by counting the marks Vega actually emitted.

Every function here is PURE: dict in, dict out, no IO, no session. They are the only place chart
encoding lives, so a page can never disagree with the JSON it was rendered from.

The colour scheme is fixed per chart rather than left to Vega's default, because two of these
charts encode a JUDGEMENT (proven vs unproven, over- vs under-valued) and the default categorical
ramp would give the proven and the unproven leg equally confident colours.
"""

from __future__ import annotations

# The falsify_scan levers, in the order the bracket should read: proven at the bottom, the
# residual the sweep could not pin in the middle, irreducible dice on top. `_LEVER_LABEL` is also
# the legend text, so it carries the interpretation the raw key does not.
LEVER_ORDER = ["policy_reducible", "mixed", "unattributed", "aleatoric"]
LEVER_LABEL = {
    "policy_reducible": "policy_reducible — PROVEN better action",
    "mixed": "mixed — both luck and a better action",
    "unattributed": "unattributed — residual, NOT proven critic error",
    "aleatoric": "aleatoric — bad-tail dice (LUCK)",
}
_LEVER_COLOR = {
    "policy_reducible": "#2a7f62",   # proven: the one confident green
    "mixed": "#7d6b3f",
    "unattributed": "#8a8a86",       # deliberately grey — it is a residual, not a finding
    "aleatoric": "#5b6e9e",
}

_BASE = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json",
         "background": None, "autosize": {"type": "fit", "contains": "padding"}}

# Vega does not wrap title text: a 200-character subtitle renders as one 1400px line and WIDENS
# the whole chart past its container, which on a phone is a page that scrolls sideways. So the
# subtitles are pre-wrapped here into the list-of-lines form Vega-Lite accepts. 52 characters is
# about what fits at the narrowest viewport the layout targets.
_SUBTITLE_WRAP = 52


def _subtitle(text: str) -> "str | list[str]":
    import textwrap
    if len(text) <= _SUBTITLE_WRAP:
        return text
    return textwrap.wrap(text, _SUBTITLE_WRAP)


def _title(text: str, subtitle: str) -> dict:
    return {"text": text, "subtitle": _subtitle(subtitle), "subtitleFontStyle": "italic"}


def _spec(**kw) -> dict:
    out = dict(_BASE)
    out.update(kw)
    return out


def crater_bracket_spec(gate: dict, *, weighting: str = "delta") -> dict:
    """`falsify_scan`'s crater bracket as one normalized stacked bar.

    The four levers are shares of crater MAGNITUDE that sum to 1, which is exactly a
    single-stack bar — and a stack makes the honest reading unavoidable: the proven leg is a
    slice of the whole, not a number standing on its own.
    """
    values = [{"lever": k, "label": LEVER_LABEL[k], "share": float(gate.get(k) or 0.0)}
              for k in LEVER_ORDER if k in gate]
    return _spec(
        title=_title("Crater-magnitude bracket",
                     f"|δ|-weighted share of crater mass ({weighting} weighting)"),
        data={"values": values},
        width="container", height=90,
        mark={"type": "bar", "tooltip": True},
        encoding={
            "x": {"field": "share", "type": "quantitative", "stack": "zero",
                  "axis": {"format": "%", "title": "share of crater magnitude"},
                  "scale": {"domain": [0, 1]}},
            "color": {"field": "label", "type": "nominal",
                      "scale": {"domain": [LEVER_LABEL[k] for k in LEVER_ORDER if k in gate],
                                "range": [_LEVER_COLOR[k] for k in LEVER_ORDER if k in gate]},
                      "legend": {"title": None, "orient": "bottom", "columns": 1}},
            "order": {"field": "lever", "type": "nominal"},
        },
    )


def reliability_curve_spec(bins: "list[dict]") -> dict:
    """The critic reliability curve: recorded V (x) against realized return G (y), per equal-count
    bin, over a dashed identity rule.

    This is the chart that most justifies a browser. In a terminal the curve is a table of ten
    rows and the reader has to hold "is G tracking V?" in their head; here the deviation from the
    identity line IS the finding, and the bin's `gap` rides the point size so a wide bin cannot
    quietly dominate a narrow one.
    """
    values = [{"v_mean": b["v_mean"], "g_mean": b["g_mean"], "gap": b["gap"], "n": b["n"],
               "v_lo": b["v_lo"], "v_hi": b["v_hi"]} for b in bins]
    ident = {"mark": {"type": "line", "strokeDash": [4, 4], "color": "#9a9a95", "opacity": 0.9},
             "encoding": {"x": {"field": "v", "type": "quantitative"},
                          "y": {"field": "v", "type": "quantitative"}},
             "data": {"values": [{"v": b["v_mean"]} for b in bins]}}
    curve = {"mark": {"type": "line", "point": False, "color": "#2a6f97"},
             "encoding": {"x": {"field": "v_mean", "type": "quantitative"},
                          "y": {"field": "g_mean", "type": "quantitative"}}}
    points = {"mark": {"type": "point", "filled": True, "color": "#2a6f97", "tooltip": True},
              "encoding": {
                  "x": {"field": "v_mean", "type": "quantitative"},
                  "y": {"field": "g_mean", "type": "quantitative"},
                  "size": {"field": "n", "type": "quantitative",
                           "legend": {"title": "decisions in bin"}},
                  "tooltip": [{"field": "v_mean", "type": "quantitative", "title": "V (bin mean)"},
                              {"field": "g_mean", "type": "quantitative", "title": "G (realized)"},
                              {"field": "gap", "type": "quantitative", "title": "gap V−G"},
                              {"field": "n", "type": "quantitative", "title": "n"}]}}
    return _spec(
        title=_title("Critic reliability: recorded V vs realized return G",
                     "above the dashed identity ⇒ the critic UNDER-valued; below ⇒ "
                     "OVER-valued · selection-skewed, read the caveats"),
        data={"values": values},
        width="container", height=280,
        layer=[ident, curve, points],
        encoding={"x": {"axis": {"title": "recorded V(s)"}},
                  "y": {"axis": {"title": "realized return G(s)"}}},
    )


def reliability_gap_spec(bins: "list[dict]") -> dict:
    """Per-bin `gap` = V−G as a diverging bar — the same data as the curve, read as a residual.

    The curve answers "does G track V"; this answers "WHERE does it not, and by how much", which
    is the question `calibration` actually splits its unattributed bucket on.
    """
    values = [{"v_mean": b["v_mean"], "gap": b["gap"], "n": b["n"]} for b in bins]
    return _spec(
        title=_title("Reliability gap per V-bin (V − G)",
                     "positive ⇒ the critic systematically over-values at that V level"),
        data={"values": values},
        width="container", height=180,
        mark={"type": "bar", "tooltip": True},
        encoding={
            "x": {"field": "v_mean", "type": "quantitative", "axis": {"title": "recorded V(s)"}},
            "y": {"field": "gap", "type": "quantitative", "axis": {"title": "gap V − G"}},
            "color": {"field": "gap", "type": "quantitative",
                      "scale": {"scheme": "redblue", "reverse": True, "domainMid": 0},
                      "legend": None},
        },
    )


def outcome_by_step_spec(steps: "list[dict]") -> dict:
    """Win/loss counts per eval step from `run_summary` — the run's shape at a glance.

    NOT a win-rate curve: eval traces are LOSS-WEIGHTED by the capture quota, so the ratio here is
    a sampling artifact. The subtitle says so, because a stacked bar of wins and losses is exactly
    the chart someone would otherwise read as progress.
    """
    values = []
    for s in steps:
        t = s.get("totals") or {}
        values.append({"step": s["step"], "outcome": "win", "n": t.get("win", 0)})
        values.append({"step": s["step"], "outcome": "loss", "n": t.get("loss", 0)})
    return _spec(
        title=_title("Captured traces per eval step",
                     "CAPTURE COUNTS, not a win rate — eval trace capture is loss-weighted"),
        data={"values": values},
        width="container", height=170,
        mark={"type": "bar", "tooltip": True},
        encoding={
            "x": {"field": "step", "type": "ordinal", "axis": {"title": "training step",
                                                               "labelAngle": -40}},
            "y": {"field": "n", "type": "quantitative", "stack": "zero",
                  "axis": {"title": "captured battles"}},
            "color": {"field": "outcome", "type": "nominal",
                      "scale": {"domain": ["win", "loss"], "range": ["#2a7f62", "#a4553f"]},
                      "legend": {"title": None, "orient": "top"}},
        },
    )


def triage_lever_spec(categories: "list[dict]") -> dict:
    """`triage`'s failure categories ranked by estimated recoverable win-rate — the lever order."""
    values = [{"category": c["category"], "pct": c["est_recoverable_winrate_pct"],
               "n": c["n"], "lever": c["lever"]} for c in categories]
    return _spec(
        title=_title("Failure levers by estimated recoverable win-rate",
                     "an UPPER BOUND — assumes fixing the lever flips every loss it explains"),
        data={"values": values},
        width="container", height={"step": 24},
        mark={"type": "bar", "tooltip": True, "color": "#2a6f97"},
        encoding={
            "y": {"field": "category", "type": "nominal", "sort": "-x", "axis": {"title": None}},
            "x": {"field": "pct", "type": "quantitative",
                  "axis": {"title": "est. recoverable win-rate (pp)"}},
            "tooltip": [{"field": "category", "type": "nominal"},
                        {"field": "lever", "type": "nominal"},
                        {"field": "pct", "type": "quantitative", "title": "est. recoverable pp"},
                        {"field": "n", "type": "quantitative", "title": "losses"}],
        },
    )


def scan_drop_spec(rows: "list[dict]", *, metric: str = "value_drop") -> dict:
    """One bar per battle: the magnitude of its single worst turning point, ranked."""
    key = "delta_v" if metric == "value_drop" else "td_residual"
    values = [{"battle": r["short_id"], "drop": (r.get("worst") or {}).get(key),
               "opponent": r["opponent"], "turn": (r.get("worst") or {}).get("turn"),
               "chosen": (r.get("worst") or {}).get("chosen")}
              for r in rows if (r.get("worst") or {}).get(key) is not None]
    return _spec(
        title=_title(f"Worst turning point per battle ({metric})",
                     "model-free: the single most negative step per battle, ranked globally"),
        data={"values": values},
        width="container", height={"step": 18},
        mark={"type": "bar", "tooltip": True, "color": "#a4553f"},
        encoding={
            "y": {"field": "battle", "type": "nominal", "sort": "x", "axis": {"title": None}},
            "x": {"field": "drop", "type": "quantitative",
                  "axis": {"title": "ΔV" if metric == "value_drop" else "TD residual"}},
            "tooltip": [{"field": "battle", "type": "nominal"},
                        {"field": "opponent", "type": "nominal"},
                        {"field": "turn", "type": "quantitative"},
                        {"field": "chosen", "type": "nominal"},
                        {"field": "drop", "type": "quantitative"}],
        },
    )
