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
    assert X.read(same)["delta_data"]["class"] == "NOT-RESOLVED"


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
    assert X.inhouse([0.52, 0.47, 0.51, 0.49, 0.50], ("B", "W", "U"))["class"] == "U"


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
