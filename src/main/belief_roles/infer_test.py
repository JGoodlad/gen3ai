"""The across-seed two-sample t (§7.4) against a hand computation, its rule-8 boundary and its refusals."""
from __future__ import annotations

import json
import math

import pytest

from main.belief_roles.infer import InferenceRefused, compare_reads, two_sample_t

TREAT = [1.0, 1.2, 0.9]           # X5 runs' log loss (lower is better)
CONTROL = [1.5, 1.4, 1.7]         # blob runs'


def test_t_matches_a_hand_computation():
    # means 1.0333.. / 1.5333..; both sample variances (ddof 1) = 0.07 / 3 = 0.023333..
    # pooled 0.023333.., se = sqrt(0.023333.. * (1/3 + 1/3)) = 0.1247219.., d = 0.5, t = 4.0089186..
    r = two_sample_t(TREAT, CONTROL, direction="lower", boundary=2.683)
    var = 0.07 / 3
    se = math.sqrt(var * (2 / 3))
    assert r.df == 4
    assert abs(r.improvement - 0.5) < 1e-12
    assert abs(r.se - se) < 1e-12
    assert abs(r.t - 0.5 / se) < 1e-9
    assert abs(r.t - 4.008918628686366) < 1e-9
    assert r.crossed
    # scipy's equal-variance Student t on (control - treat) is the same statistic
    from scipy.stats import ttest_ind
    assert abs(ttest_ind(CONTROL, TREAT, equal_var=True).statistic - r.t) < 1e-9
    assert abs(r.p_one_sided - ttest_ind(CONTROL, TREAT, equal_var=True, alternative="greater").pvalue) < 1e-12


def test_direction_margin_and_unequal_n():
    hi = two_sample_t([0.9, 0.8, 0.85, 0.95], [0.7, 0.75], direction="higher", boundary=1.874)
    lo = two_sample_t([-0.9, -0.8, -0.85, -0.95], [-0.7, -0.75], direction="lower", boundary=1.874)
    assert hi.df == 4 and abs(hi.t - lo.t) < 1e-12
    m = two_sample_t(TREAT, CONTROL, direction="lower", boundary=0.0, margin=0.1)
    assert abs(m.t - 0.4 / m.se) < 1e-9


def test_rule8_a_t_within_1e9_of_the_boundary_is_not_a_crossing():
    t = two_sample_t(TREAT, CONTROL, direction="lower", boundary=0.0).t
    assert not two_sample_t(TREAT, CONTROL, direction="lower", boundary=t).crossed
    assert not two_sample_t(TREAT, CONTROL, direction="lower", boundary=t - 5e-10).crossed
    assert two_sample_t(TREAT, CONTROL, direction="lower", boundary=t - 2e-9).crossed


def test_refusals_on_plain_values():
    with pytest.raises(InferenceRefused):
        two_sample_t([1.0], CONTROL, direction="lower", boundary=1.0)
    with pytest.raises(InferenceRefused):
        two_sample_t([1.0, 1.0], [2.0, 2.0], direction="lower", boundary=1.0)
    with pytest.raises(InferenceRefused):
        two_sample_t(TREAT, CONTROL, direction="sideways", boundary=1.0)


def _read(tmp_path, name, arm, value, sha, bank="b" * 64, roles="r" * 64):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"schema": "gen3_belief_purpose_read_v1", "label": name, "arm": arm,
                             "bank_sha256": bank, "role_set_sha256": roles,
                             "checkpoint": {"sha256": sha}, "per_run": {"intent_logloss": value}}))
    return p


def test_compare_reads_and_its_refusals(tmp_path):
    t = [_read(tmp_path, f"x{i}", "fixed_mass", v, f"t{i}") for i, v in enumerate(TREAT)]
    c = [_read(tmp_path, f"b{i}", "blob", v, f"c{i}") for i, v in enumerate(CONTROL)]
    r = compare_reads(t, c, "intent_logloss", boundary=2.683)
    assert abs(r.t - 4.008918628686366) < 1e-9 and r.crossed
    with pytest.raises(InferenceRefused, match="bank_sha256"):
        compare_reads(t, c[:2] + [_read(tmp_path, "bx", "blob", 1.6, "cx", bank="z" * 64)],
                      "intent_logloss", boundary=1.0)
    with pytest.raises(InferenceRefused, match="role_set_sha256"):
        compare_reads(t, c[:2] + [_read(tmp_path, "by", "blob", 1.6, "cy", roles="q" * 64)],
                      "intent_logloss", boundary=1.0)
    with pytest.raises(InferenceRefused, match="more than once"):
        compare_reads(t, c[:2] + [_read(tmp_path, "bz", "blob", 1.6, "t0")], "intent_logloss", boundary=1.0)
    with pytest.raises(InferenceRefused, match="arms"):
        compare_reads(t, c[:2] + [_read(tmp_path, "bw", "fixed_mass", 1.6, "cw")], "intent_logloss",
                      boundary=1.0)
    with pytest.raises(InferenceRefused, match="unknown metric"):
        compare_reads(t, c, "not_a_metric", boundary=1.0)
    with pytest.raises(InferenceRefused, match="no on_pool value"):
        compare_reads(t, c, "presence_brier", boundary=1.0)


def _cread(tmp_path, name, arm, value, sha, ref):
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"schema": "gen3_belief_purpose_read_v2", "label": name, "arm": arm,
                             "bank_sha256": "b" * 64, "role_set_sha256": "r" * 64,
                             "checkpoint": {"sha256": sha}, "eset_reference": ref,
                             "per_run": {"intent_logloss_conditional": value}}))
    return p


def test_the_adoption_gate_metric_needs_a_one_to_one_set_pairing(tmp_path):
    """Amendment 3(b): blob reads on their OWN named set; the fixed_mass reads' reference blob
    checkpoints map one-to-one onto the control group's — else the arms are scored over different
    supports and the comparison is REFUSED."""
    m = "intent_logloss_conditional"
    c = [_cread(tmp_path, f"b{i}", "blob", v, f"c{i}", {"mode": "own", "checkpoint_sha256": f"c{i}"})
         for i, v in enumerate(CONTROL)]

    def treat(refs):
        return [_cread(tmp_path, f"x{i}", "fixed_mass", v, f"t{i}",
                       {"mode": "paired", "checkpoint_sha256": r} if r else {"mode": "none"})
                for i, (v, r) in enumerate(zip(TREAT, refs))]

    r = compare_reads(treat(["c0", "c1", "c2"]), c, m, boundary=2.683)
    assert abs(r.t - 4.008918628686366) < 1e-9 and r.crossed
    assert compare_reads(treat(["c2", "c0", "c1"]), c, m, boundary=2.683).t == r.t
    for bad in (["c0", "c0", "c1"], ["c0", "c1", "cX"], ["c0", "c1", None]):
        with pytest.raises(InferenceRefused, match="one-to-one|paired blob"):
            compare_reads(treat(bad), c, m, boundary=2.683)
    with pytest.raises(InferenceRefused, match="one-to-one"):
        compare_reads(treat(["c0", "c1", "c2"])[:2], c, m, boundary=2.683)       # unequal n
    c_bad = c[:2] + [_cread(tmp_path, "b9", "blob", 1.7, "c9", {"mode": "paired", "checkpoint_sha256": "c0"})]
    with pytest.raises(InferenceRefused, match="OWN named set"):
        compare_reads(treat(["c0", "c1", "c2"]), c_bad, m, boundary=2.683)
