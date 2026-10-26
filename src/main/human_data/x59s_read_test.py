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


def test_floor_and_inhouse():
    arms = {a: _arm([0.10] * 5) for a in X.ARMS}
    assert X.read(arms)["floor"]
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


def test_uninformative_when_every_arm_sits_near_the_floor():
    r = _read(0.20, 0.17, 0.18)
    ih = {k: {"delta": 0.01, "class": "x"} for k in ("delta_data", "delta_labels")}
    v = X.verdict(r, ih, {"random": 0.05, "maxdamage": 0.16}, [1.4] * 5, [1.6] * 5, "UNCLEAR")
    assert v["uninformative"] and "close nothing" in v["actions"][0]
    v = X.verdict(r, ih, {"random": 0.05, "maxdamage": 0.10}, [1.4] * 5, [1.6] * 5, "UNCLEAR")
    assert not v["uninformative"]
    assert X.uninformative({"a": 0.15}, {"f": 0.10}, False)              # exactly 5 pp: within (tie)


def test_a_kakuna_only_gain_is_kakuna_specific_and_routes_nothing():
    r = _read(0.30, 0.20, 0.20)
    assert r["delta_data"]["class"] == "HUMAN-DATA-BETTER"
    ih = {"delta_data": {"delta": 0.0, "class": "x"}, "delta_labels": {"delta": -0.02, "class": "x"}}
    v = X.verdict(r, ih, {"random": 0.02, "maxdamage": 0.05}, [1.4] * 5, [1.6] * 5, "UNCLEAR")
    assert v["classes"] == {"delta_data": "KAKUNA-SPECIFIC", "delta_labels": "KAKUNA-SPECIFIC"}
    assert not any("H-BC first" in a for a in v["actions"])


def test_hbc_needs_sp_to_have_learned_its_teacher_and_closure_needs_fr2():
    r = _read(0.30, 0.20, 0.25)
    ih = {k: {"delta": 0.03, "class": "x"} for k in ("delta_data", "delta_labels")}
    fl = {"random": 0.02, "maxdamage": 0.05}
    yes = X.verdict(r, ih, fl, [1.40, 1.41, 1.39, 1.40, 1.42], [1.60, 1.61, 1.59, 1.62, 1.60], "UNCLEAR")
    no = X.verdict(r, ih, fl, [1.60, 1.61, 1.59, 1.62, 1.60], [1.60, 1.61, 1.59, 1.62, 1.60], "UNCLEAR")
    assert yes["actions"][0].startswith("H-BC first") and "NOT routed" in no["actions"][0]
    r = _read(0.20, 0.30, 0.30)
    assert r["delta_data"]["class"] == "SELF-PLAY-DATA-BETTER" and r["delta_states"]["lo"] <= 0
    ih = {k: {"delta": -0.05, "class": "x"} for k in ("delta_data", "delta_labels")}
    ok = X.verdict(r, ih, fl, [1.4] * 4 + [1.41], [1.6] * 4 + [1.61], "NO-ARTEFACT")
    art = X.verdict(r, ih, fl, [1.4] * 4 + [1.41], [1.6] * 4 + [1.61], "ARTEFACT")
    assert any(a.startswith("closes 1300+") for a in ok["actions"])
    assert not any(a.startswith("closes 1300+") for a in art["actions"])
    assert any(">= 1500 follow-up" in a for a in ok["actions"])


def test_the_verdict_refuses_an_unread_fr2():
    r = _read(0.20, 0.30, 0.30)
    ih = {k: {"delta": -0.05, "class": "x"} for k in ("delta_data", "delta_labels")}
    with pytest.raises(ValueError, match="FR-2"):
        X.verdict(r, ih, {"random": 0.02, "maxdamage": 0.05}, [1.4] * 5, [1.6] * 5, "UNREAD")
