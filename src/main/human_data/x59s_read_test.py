"""X59-S's registered read rule (``x59s_read.py``) on planted data: classes, the seed guard, the tie rule."""
from __future__ import annotations

import math

import numpy as np
import pytest

from main.human_data import x59s_read as X


def _arm(rates, g=400):
    return [{p: (int(round(r * g)), g) for p in ("H", "F", "P72")} for r in rates]


def test_a_clear_gain_reads_better_and_a_null_does_not():
    arms = {"S-H": _arm([0.30, 0.31, 0.29, 0.30, 0.30]), "S-P": _arm([0.20] * 5), "S-S": _arm([0.25] * 5)}
    r = X.read(arms)
    assert r["delta_labels"]["class"] == "HUMAN-LABELS-BETTER" and r["delta_data"]["class"] == "HUMAN-DATA-BETTER"
    assert r["delta_states"]["class"] == "STATES-HURT"
    assert not r["seed_sd"]["guard_fires"] and r["delta_data"]["rule"].startswith("z")
    same = {a: _arm([0.25] * 5) for a in X.ARMS}
    assert X.read(same)["delta_data"]["class"] == X.NO_DIFF          # a narrow null IS an answer (M6)


def test_the_seed_term_never_narrows_below_the_bound():
    arms = {a: _arm([0.25] * 5) for a in X.ARMS}
    hw = X.read(arms)["delta_data"]["halfwidth"]
    b = 2 * 0.25 * 0.75 / 1200 / 5
    assert hw == pytest.approx(X.Z * math.sqrt(b + 2 * X.SIGMA_MAX ** 2 / 5), rel=1e-6)


def test_a_wide_seed_spread_fires_the_guard_and_reads_welch():
    rng = np.random.default_rng(1)
    arms = {"S-H": _arm(0.30 + rng.normal(0, 0.08, 5)), "S-P": _arm(0.25 + rng.normal(0, 0.08, 5)),
            "S-S": _arm(0.25 + rng.normal(0, 0.08, 5))}
    r = X.read(arms)
    assert r["seed_sd"]["guard_fires"] and r["delta_data"]["rule"].startswith("Welch")


def test_a_bound_on_zero_takes_the_undecided_class():
    assert X.classify(1e-12, 0.05, ("B", "W", "U")) == "U"
    assert X.classify(-0.05, -1e-12, ("B", "W", "U")) == "U"
    assert X.classify(2e-9, 0.05, ("B", "W", "U")) == "B"


def test_inhouse():
    assert X.inhouse([0.56, 0.58, 0.55, 0.57, 0.56], ("B", "W", "U"))["class"] == "B"
    assert X.inhouse([0.52, 0.47, 0.51, 0.49, 0.50], ("B", "W", "U"))["class"] == X.NO_DIFF
    assert X.inhouse([0.58, 0.40, 0.55, 0.45, 0.52], ("B", "W", "U"))["class"] == "U"


def test_unequal_arms_are_refused():
    arms = {"S-H": _arm([0.3] * 5), "S-P": _arm([0.3] * 4), "S-S": _arm([0.3] * 5)}
    with pytest.raises(ValueError):
        X.read(arms)


def test_register_cross_checks_the_three_inputs(tmp_path):
    import json

    from main.human_data import checks

    def man(d, extra):
        d.mkdir()
        (d / "rows_manifest.json").write_text(json.dumps({"rowset": "r", "train_rows": 20, "val_rows": 2,
                                                          "train_hidden": 1,
                                                          "train_shards": [{"rows": 16}, {"rows": 4}], **extra}))
    rows, ss, lab = tmp_path / "rows", tmp_path / "ss", tmp_path / "lab"
    man(rows, {})
    from main.human_data import rowset as RS

    sha = RS.S_sha(rows / "rows_manifest.json")
    man(ss, {"like_rows_manifest_sha256": sha})
    lab.mkdir()
    (lab / "labels_manifest.json").write_text(json.dumps({"rows_manifest_sha256": sha, "pstar_sha256": "x",
                                                         "shards": [{}, {}, {}]}))
    assert checks.register(rows, lab, ss)["pass"]
    (ss / "rows_manifest.json").write_text(json.dumps({"rowset": "r", "train_rows": 20, "val_rows": 2,
                                                       "train_shards": [{"rows": 10}, {"rows": 10}],
                                                       "like_rows_manifest_sha256": sha}))
    assert not checks.register(rows, lab, ss)["pass"]          # other shard sizes: other step budgets
    man(tmp_path / "ss2", {"like_rows_manifest_sha256": sha})
    assert not checks.register(rows, lab, tmp_path / "ss2", require_panel_rule=True)["pass"]   # B1: unruled S-S
    man(tmp_path / "ss3", {"like_rows_manifest_sha256": sha, "panel_rule": {"profile_sha256": "p"}})
    l0 = tmp_path / "l0"
    l0.mkdir()
    (l0 / "labels_manifest.json").write_text(json.dumps({"rows_manifest_sha256": sha, "pstar_sha256": "x",
                                                        "shards": [{}, {}, {}]}))
    r = checks.register(rows, lab, tmp_path / "ss3", l0, require_panel_rule=True)
    assert r["pass"] and r["labels0_manifest_sha256"] and r["selfplay_panel_rule_profile_sha256"] == "p"


def test_no_difference_needs_the_whole_interval_inside_the_bar():
    names = ("B", "W", "U")
    assert X.classify(-0.039, 0.039, names) == X.NO_DIFF
    assert X.classify(-0.041, 0.02, names) == "U" and X.classify(-0.02, 0.041, names) == "U"
    assert X.classify(-0.04 + 5e-10, 0.02, names) == "U"                # within 1e-9 of the bar: undecided
    assert X.classify(0.001, 0.03, names) == "B"


def _read(sh, sp, ss):
    return X.read({"S-H": _arm([sh] * 5), "S-P": _arm([sp] * 5), "S-S": _arm([ss] * 5)})


YARD_LEARNED = {"S-P": [1.42] * 5, "S-S": [1.40] * 5, "S-H": [1.60] * 5}
FLOOR = {"random": 0.0, "aggressive": 0.0458}


def _ih(d, lo=None, hi=None):
    return {"delta": d, "lo": d - 0.05 if lo is None else lo, "hi": d + 0.05 if hi is None else hi, "class": "x"}


def _v(r, ih, *, uninf=None, f=None, yard=YARD_LEARNED, fr2="NO-ARTEFACT", fit=0.001, share=0.0, rlb=-0.02):
    return X.verdict(r, ih, uninf or {"delta_data": False, "delta_labels": False},
                     f or {"delta_data": 0.05, "delta_labels": 0.05}, yard, fr2, fit, share, rlb)


def test_the_gate_is_per_contrast_on_the_measured_references():
    pilot = {"S-H": _arm([0.05] * 5), "S-P": _arm([0.05] * 5), "S-S": _arm([0.20] * 5)}
    g = X.gate(pilot, FLOOR)
    assert g["uninformative"] == {"delta_data": False, "delta_labels": True, "delta_states": False}
    assert g["continue"]
    allfloor = {a: _arm([0.05] * 5) for a in X.ARMS}
    assert not X.gate(allfloor, FLOOR)["continue"]
    at_bar = {a: _arm([0.0958] * 5, g=10000) for a in X.ARMS}                 # 958 / 10000 = the bar: within
    assert X.gate(at_bar, FLOOR)["uninformative"]["delta_data"]


def test_the_interval_widens_to_the_contrasts_own_welch():
    # one noisy arm (S-H SD ~3 pp, the others 0): the pooled guard does not fire, the contrast's own Welch is wider
    noisy = {"S-H": _arm([0.21, 0.25, 0.29, 0.23, 0.27]), "S-P": _arm([0.25] * 5), "S-S": _arm([0.25] * 5)}
    r = X.read(noisy)
    assert not r["seed_sd"]["guard_fires"]
    c = r["delta_data"]
    assert c["halfwidth"] == max(c["halfwidth_registered"], c["halfwidth_welch"])
    assert c["halfwidth_welch"] > c["halfwidth_registered"] and "widened" in c["rule"]


def test_no_difference_needs_the_in_house_interval_too():
    r = _read(0.25, 0.25, 0.25)
    assert r["delta_data"]["class"] == X.NO_DIFF
    ih = {k: _ih(0.0, -0.03, 0.03) for k in ("delta_data", "delta_labels")}
    assert _v(r, ih)["classes"]["delta_data"] == X.NO_DIFF
    ih = {k: _ih(0.0, -0.05, 0.03) for k in ("delta_data", "delta_labels")}
    assert _v(r, ih)["classes"]["delta_data"] == "NOT-RESOLVED"


def test_the_agreement_guard_runs_both_ways_and_f_must_agree():
    r = _read(0.30, 0.20, 0.20)
    assert r["delta_data"]["class"] == "HUMAN-DATA-BETTER"
    ih = {"delta_data": _ih(0.0), "delta_labels": _ih(-0.02)}
    v = _v(r, ih)
    assert v["classes"] == {"delta_data": X.KAKUNA_SPECIFIC, "delta_labels": X.KAKUNA_SPECIFIC}
    ih = {"delta_data": _ih(0.03), "delta_labels": _ih(0.03)}
    v = _v(r, ih, f={"delta_data": 0.0, "delta_labels": 0.02})
    assert v["classes"] == {"delta_data": X.POP_SPECIFIC, "delta_labels": "HUMAN-LABELS-BETTER"}
    r = _read(0.20, 0.30, 0.30)
    assert r["delta_data"]["class"] == "SELF-PLAY-DATA-BETTER"
    v = _v(r, {k: _ih(0.0) for k in ("delta_data", "delta_labels")})
    assert v["classes"]["delta_data"] == X.KAKUNA_SPECIFIC and not any("closes 1300" in a for a in v["actions"])


def test_hbc_needs_the_midpoint_and_the_closure_needs_every_gate():
    r = _read(0.30, 0.20, 0.25)
    ih = {k: _ih(0.03) for k in ("delta_data", "delta_labels")}
    assert _v(r, ih)["actions"][0].startswith("H-BC first")
    far = {"S-P": [1.52] * 5, "S-S": [1.40] * 5, "S-H": [1.60] * 5}           # closer to S-H than to S-S
    assert "NOT routed" in _v(r, ih, yard=far)["actions"][0]
    r = _read(0.20, 0.30, 0.30)
    ih = {k: _ih(-0.05) for k in ("delta_data", "delta_labels")}
    assert any(a.startswith("closes 1300+") for a in _v(r, ih)["actions"])
    for kw in ({"rlb": 0.001}, {"rlb": 5e-10}, {"fit": 0.01}, {"share": 0.0101}):
        assert not any(a.startswith("closes 1300+") for a in _v(r, ih, **kw)["actions"]), kw
    assert any("pending Delta_restriction" in a for a in _v(r, ih, rlb=None)["actions"])
    assert any(a.startswith("closes 1300+") for a in _v(r, ih, fr2="ARTEFACT")["actions"])   # FR-2 gates nothing
    assert any(a.startswith("closes 1300+") for a in _v(r, ih, share=0.01)["actions"])   # exactly 1.0 %: not more
    assert any(">= 1500 follow-up" in a for a in _v(r, ih)["actions"])


def test_an_uninformative_contrast_routes_nothing():
    r = _read(0.30, 0.20, 0.25)
    ih = {k: _ih(0.03) for k in ("delta_data", "delta_labels")}
    v = _v(r, ih, uninf={"delta_data": True, "delta_labels": True})
    assert v["classes"] == {"delta_data": X.UNINF, "delta_labels": X.UNINF}
    assert v["kakuna_classes"]["delta_data"] == "HUMAN-DATA-BETTER"
    assert not any("follow-up" in a for a in v["actions"])


def test_the_verdict_refuses_an_unread_fr2():
    r = _read(0.20, 0.30, 0.30)
    ih = {k: _ih(-0.05) for k in ("delta_data", "delta_labels")}
    with pytest.raises(ValueError, match="FR-2"):
        _v(r, ih, fr2="UNREAD")
