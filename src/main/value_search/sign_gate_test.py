"""``main.value_search.sign_gate`` (the S-U gate test): the estimator IS the S-U review's ``boot`` with one panel (draw
for draw), the pooled U estimate weighs panels equally, the hashed draw, the acting rule, the per-state values and the
registered GO / STOP rule with rule 8's boundary. Pure numpy, synthetic rows; the bank reproduction itself is the
measurement's ``scripts/bank_check.py`` (the archive's rows)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from main.value_search import sign_gate as SG


def _review_boot(vals: dict, B: int = 1000, seed: int = 20261027):
    """``su_headroom_2026-10-27/scripts/review_checks.py``'s ``boot``, verbatim in substance."""
    keys = sorted(vals)
    arrs = [np.asarray([v for v in vals[k] if not np.isnan(v)], dtype=float) for k in keys]
    s = np.array([a.sum() for a in arrs])
    n = np.array([len(a) for a in arrs])
    rng = np.random.default_rng(seed)
    est = s.sum() / max(n.sum(), 1)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(keys), len(keys))
        bs.append(s[i].sum() / max(n[i].sum(), 1))
    return [float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(n.sum())]


def _vals(rng, k=60, p=0.55, nan=0.6):
    out = {}
    for c in range(k):
        m = int(rng.integers(1, 4))
        out[f"g{c:03d}"] = [float("nan") if rng.random() < nan else float(rng.random() < p) for _ in range(m)]
    return out


def test_one_panel_is_the_reviews_boot_draw_for_draw():
    rng = np.random.default_rng(1)
    for _ in range(5):
        v = _vals(rng)
        got = SG.cluster_ratio_boot({"p": v}, ["p"], B=1000, seed=20261027)
        assert [got["est"], got["lo"], got["hi"], got["n"]] == _review_boot(v)


def test_the_pooled_estimate_weighs_panels_equally_and_is_deterministic():
    a = {"g1": [1.0, 1.0, 1.0, 1.0], "g2": [1.0, 1.0]}        # 6 / 6
    b = {"h1": [0.0], "h2": [float("nan")]}                   # 0 / 1; an all-NaN game still takes part
    r = SG.cluster_ratio_boot({"A": a, "B": b}, ["A", "B"], B=1000, seed=7)
    assert r["est"] == 0.5 and r["n"] == 7 and r["per_panel"] == {"A": [1.0, 6, 2], "B": [0.0, 1, 2]}
    assert r == SG.cluster_ratio_boot({"A": a, "B": b}, ["A", "B"], B=1000, seed=7)


def test_the_draw_is_the_two_smallest_hashes_per_game_inside_the_window():
    recs = []
    for deal in ("d1", "d2"):
        for turn in range(5, 40):
            recs.append({"deal": deal, "turn": turn, "n": turn + 3, "n_legal": 1 if turn == 20 else 4})
    sel = SG.select(recs)
    for deal in ("d1", "d2"):
        mine = [(r["turn"], r["n"]) for r in sel if r["deal"] == deal]
        elig = [(t, t + 3) for t in range(10, 31) if t != 20]
        want = sorted(elig, key=lambda k: SG.select_hash(deal, *k))[:2]
        assert mine == want
    assert all(r["select_hash"] == SG.select_hash(r["deal"], r["turn"], r["n"]) for r in sel)
    with pytest.raises(SG.GateError, match="kept twice"):
        SG.select(recs + [recs[10]])


def test_the_acting_rule_is_pikl_with_ties_to_the_policys_earlier_action():
    lg = [2.0, 1.0, 0.0]
    assert SG.pikl_pick([0.5, 0.5, 0.5], lg, 0.015) == 0
    assert SG.pikl_pick([0.50, 0.52, 0.0], lg, 0.015) == 1        # +0.02 beats 0.015 * (log pi gap of 1)
    assert SG.pikl_pick([0.50, 0.51, 0.0], lg, 0.015) == 0
    assert SG.pikl_pick([0.5, 0.5], [1.0, 1.0], 0.015) == 0      # an exact tie goes to the earlier action


def _row(panel, deal, i, *, dev=True, right=True, excluded=False, tie=False):
    s = ({"searched": False, "why": "excluded", "cpu_s": 3.0} if excluded else
         {"searched": True, "deviation": dev, "played": 1 if dev else 0, "cpu_s": 9.0, "leak": {"rewritten_moves": 0}})
    r = {"panel": panel, "deal": deal, "key": f"{panel}|{deal}|{i}", "search": s,
         "speed_flip": {"ok": True, "speed_mons": 3, "speed_pinned": int(i % 5 == 0)}}
    if dev and not excluded:
        tp, tt = (0.6, 0.6) if tie else ((0.7, 0.4) if right else (0.4, 0.7))
        r["referee"] = {"T_pick": tp, "T_top": tt, "sign": SG.sign_value(tp, tt), "cpu_s": 20.0}
    return r


def _rows(acc_u: float, acc_home: float, n_games=40):
    rows = []
    for p, acc in (("heldout72_hpfix", acc_u), ("forum72", acc_u), ("p72", acc_u), ("home72", acc_home)):
        for g in range(n_games):
            for i in range(2):
                k = g * 2 + i
                rows.append(_row(p, f"{p}{g}", k, dev=k % 5 != 4, right=(k % 10) < acc * 10,
                                 excluded=k % 37 == 0, tie=k % 23 == 0))
    return rows


PARAMS = {"home_panel": "home72", "u_panels": ["heldout72_hpfix", "forum72", "p72"], "bar": 0.5, "rule8_eps": 0.0005,
          "bootstrap": {"B": 2000, "seed": 20261027}}


def test_go_needs_the_pooled_U_lower_bound_above_the_bar_and_home72_is_only_reported():
    go = SG.read(_rows(0.9, 0.5), PARAMS)
    assert go["verdict"] == "GO" and go["primary"]["lo"] > 0.5 and not go["boundary"]
    stop = SG.read(_rows(0.5, 0.9), PARAMS)
    assert stop["verdict"] == "STOP" and stop["reported"]["home72_sign"]["est"] > 0.8
    assert stop["reported"]["u_minus_home72_sign"]["sign"] == "-"
    ps = stop["reported"]["per_set"]["forum72"]
    assert ps["states"] == 80 and ps["excluded"] == 3 and ps["searched"] == 77
    assert ps["countable_deviations"] < ps["deviations"] <= ps["searched"]


def test_a_lower_bound_within_rule8s_eps_of_the_bar_reads_STOP_and_is_flagged(monkeypatch):
    real = SG.cluster_ratio_boot

    def near(groups, panels, **kw):
        r = real(groups, panels, **kw)
        return dict(r, lo=0.5004) if list(panels) == PARAMS["u_panels"] else r

    monkeypatch.setattr(SG, "cluster_ratio_boot", near)
    res = SG.read(_rows(0.9, 0.5), PARAMS)
    assert res["verdict"] == "STOP" and res["boundary"]


def test_a_short_or_doubled_score_file_and_a_deviation_without_its_referee_are_refused():
    rows = _rows(0.6, 0.5)
    with pytest.raises(SG.GateError, match="selection holds"):
        SG.read(rows, {**PARAMS, "expected_states": len(rows) + 1})
    with pytest.raises(SG.GateError, match="twice"):
        SG.read(rows + rows[:1], PARAMS)
    bad = [dict(rows[1])]
    bad[0].pop("referee")
    with pytest.raises(SG.GateError, match="no referee"):
        SG.read(bad, PARAMS)


def test_per_state_values():
    v = SG.state_values(_row("p", "d", 1, dev=False))
    assert v["dev"] == 0.0 and math.isnan(v["sign"]) and v["imp"] == 0.0 and v["excluded"] == 0.0
    v = SG.state_values(_row("p", "d", 1, excluded=True))
    assert v["searched"] == 0.0 and v["excluded"] == 1.0 and math.isnan(v["dev"]) and math.isnan(v["imp"])
    v = SG.state_values(_row("p", "d", 1, right=False))
    assert v["sign"] == 0.0 and v["imp"] == pytest.approx(-0.3)
    assert math.isnan(SG.state_values(_row("p", "d", 1, tie=True))["sign"])


def test_the_referee_seeds_are_the_banks_layout_and_never_the_searchs_dice():
    from main.policy_spectrum.truth import turn_seeds
    from main.value_search.search import seeds_for

    k = "su_live_v1|abc|t12|n7"
    s = SG.referee_seeds(k)
    assert len(s) == SG.REFEREE_SEEDS == 24 and s == turn_seeds(k, 40)[16:40]
    assert not set(s) & set(seeds_for(k, 64))


def test_select_reads_counted_cells_only_and_is_written_once(tmp_path, capsys):
    import json

    from main.anchors.input_log import SCHEMA

    base = tmp_path / "anchor" / "gate" / "pstar"
    units = [{"name": f"pstar__{ts}__100", "teamset": ts, "unit_seed": 100} for ts in ("home72", "p72", "forum72")]
    (base / "cells").mkdir(parents=True)
    (base / "plan.json").write_text(json.dumps({"units": units}))
    ev = [{"event": "end", "unit": "pstar__home72__100", "counted": False},      # an earlier attempt
          {"event": "end", "unit": "pstar__home72__100", "counted": True},
          {"event": "end", "unit": "pstar__p72__100", "counted": False}]       # never counted
    (base / "units.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    for u in units:
        d = base / "cells" / u["name"]
        d.mkdir()
        rows = [{"schema": SCHEMA, "deal": f"{u['teamset']}-g{g}", "turn": t, "n": t * 2, "n_legal": 3}
                for g in range(3) for t in range(10, 16)]
        (d / "input_logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "selected.jsonl"
    assert SG.main(["select", "--state", str(tmp_path), "--group", "gate", "--label", "pstar", "--out", str(out)]) == 0
    sel = [json.loads(x) for x in out.read_text().splitlines()]
    assert {r["panel"] for r in sel} == {"home72"} and len(sel) == 6
    assert all(r["schema"] == SG.SELECT_SCHEMA and r["cell"] == "pstar__home72__100" for r in sel)
    with pytest.raises(SystemExit, match="written once"):
        SG.main(["select", "--state", str(tmp_path), "--group", "gate", "--label", "pstar", "--out", str(out)])
