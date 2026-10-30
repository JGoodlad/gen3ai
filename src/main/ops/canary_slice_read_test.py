"""`main.ops.canary_slice_read`: the row-set parser and its refusals, the fixture patch (tiles the set, restores the
real function even on an error), the pre-declared rule (each class and the 1 % boundary band on both sides), the state
and window verdicts, the CLI's refusals, and (slow) the dither read on the real learner on CPU with R1 installed under
dynamo's eager backend (compiled == eager, so the compiled and eager distances agree and the null is non-zero)."""
from __future__ import annotations

import pytest

from main.ops import canary_dump_replay as R
from main.ops import canary_slice_read as S

K = 12.0
P = "features_extractor.spread_belief.nature_head.weight"


# ------------------------------------------------------------------------------------------------ row sets
def test_parse_rows_ranges_and_refusals():
    assert S.parse_rows("0-8,10-31") == list(range(9)) + list(range(10, 32))
    assert S.parse_rows("9,32-63") == [9] + list(range(32, 64))
    for bad in ("9", "", "3-1", "0-3,2", "60-64", "-1"):
        with pytest.raises((R.Refusal, ValueError)):
            S.parse_rows(bad)


def test_parse_row_sets_needs_a_label_and_never_shadows_a_slice():
    assert S.parse_row_sets(["a=0-3", "b=4,5"]) == {"a": [0, 1, 2, 3], "b": [4, 5]}
    for bad in (["0-3"], ["=0-3"], ["a=0-3", "a=4-5"], ["2=0-3"]):
        with pytest.raises(R.Refusal):
            S.parse_row_sets(bad)


def test_custom_rows_tiles_the_set_whatever_slice_and_restores_on_error():
    from agents.model import compile_trainer as ct
    real = ct.fixture_index
    with S.custom_rows([9, 32, 33]):
        assert ct.fixture_index(7, 64, 2).tolist() == [9, 32, 33, 9, 32, 33, 9]
        assert ct.fixture_index(4, 64, 0).tolist() == [9, 32, 33, 9]
    assert ct.fixture_index is real
    with pytest.raises(RuntimeError):
        with S.custom_rows([0, 1]):
            raise RuntimeError("boom")
    assert ct.fixture_index is real
    with S.custom_rows(None):
        assert ct.fixture_index is real
    with S.custom_rows([0, 70]):
        with pytest.raises(R.Refusal):
            ct.fixture_index(4, 64, 0)


# ------------------------------------------------------------------------------------------------ the rule
def _d(null, comp, eager=None):
    return {"null": list(null), "compiled": list(comp), "eager": list(eager or [0.01] * len(comp))}


def test_target_classes_and_the_boundary_band_both_sides():
    n = [0.2, 0.5, 1.0]                                   # max 1.0
    assert S.target_verdict(_d(n, [1.5, 1.6, 1.7]), 0.01, K)["class"] == "REAL"
    assert S.target_verdict(_d(n, [0.5, 0.6, 0.7]), 0.01, K)["class"] == "CONDITIONING"
    assert S.target_verdict(_d(n, [0.9, 1.0, 1.1]), 0.01, K)["class"] == "BOUNDARY"          # median exactly n
    assert S.target_verdict(_d(n, [0.0, 1.0099, 2.0]), 0.01, K)["class"] == "BOUNDARY"       # inside +1 %
    assert S.target_verdict(_d(n, [0.0, 1.0101, 2.0]), 0.01, K)["class"] == "REAL"           # just outside
    assert S.target_verdict(_d(n, [0.0, 0.9899, 2.0]), 0.01, K)["class"] == "CONDITIONING"
    assert S.target_verdict(_d(n, []), 0.01, K)["class"] == "UNREAD"
    assert S.target_verdict(_d(n, [float("nan"), 1.0, 1.0]), 0.01, K)["class"] == "NONFINITE"


def test_target_reports_the_canary_ratio_per_seed_and_eager_against_the_null():
    t = S.target_verdict(_d([0.4, 0.5, 0.6], [0.3, 0.2, 0.25], eager=[0.001, 0.05, 0.02]), 0.01, K)
    assert t["class"] == "CONDITIONING"
    assert t["canary_ratio_per_seed"] == pytest.approx([30.0, 4.0, 12.5])               # floored at 0.01
    assert t["share_over_k"] == pytest.approx(2 / 3)
    assert t["eager_median_over_null_median"] == pytest.approx(0.02 / 0.5)


def _dither(**per_slice):
    return {str(s): {"floor": 0.01, "params": {P: d}} for s, d in per_slice.items()}


def test_state_verdict_real_conditioning_inconclusive_invalid_and_eager_only():
    cond, real = _d([1.0], [0.5]), _d([1.0], [2.0])
    assert S.state_verdict(_dither(**{"0": cond, "2": cond}), [P], K, True, True)["verdict"] == "CONDITIONING"
    assert S.state_verdict(_dither(**{"0": cond, "2": real}), [P], K, True, True)["verdict"] == "REAL"
    assert S.state_verdict(_dither(**{"0": cond, "2": _d([1.0], [1.0])}), [P], K, True, True)["verdict"] \
        == "INCONCLUSIVE"
    v = S.state_verdict(_dither(**{"0": cond}), [P, "other"], K, True, True)
    assert v["verdict"] == "CONDITIONING"                                                # NOT-JUDGED is not a read
    assert {t["class"] for t in v["targets"]} == {"CONDITIONING", "NOT-JUDGED"}
    assert S.state_verdict({}, [P], K, True, True)["verdict"] == "INCONCLUSIVE"
    assert S.state_verdict(_dither(**{"0": real}), [P], K, False, True)["verdict"] == "INVALID-REPLAY"
    assert S.state_verdict(_dither(**{"0": cond}), [P], K, True, False)["verdict"] == "EAGER-ONLY"


def test_window_verdict():
    c, r = {"verdict": "CONDITIONING"}, {"verdict": "REAL"}
    assert S.window_verdict({"a": c, "b": c})["hypothesis_i_real_numerics"] == "RULED OUT AT THESE STATES"
    assert S.window_verdict({"a": c, "b": r})["hypothesis_i_real_numerics"] == "LIVE"
    assert S.window_verdict({"a": c, "b": None})["hypothesis_i_real_numerics"] == "INCONCLUSIVE"
    assert S.window_verdict({"a": c, "b": {"verdict": "INCONCLUSIVE"}})["hypothesis_i_real_numerics"] == "INCONCLUSIVE"


def test_run_verdict_reads_each_state_dir(tmp_path):
    import json
    for name, v in (("a", "CONDITIONING"), ("b", "REAL")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "verdict.json").write_text(json.dumps({"verdict": v}))
    out = tmp_path / "w.json"
    assert S.main(["verdict", "--root", f"{tmp_path / 'a'},{tmp_path / 'b'},{tmp_path / 'c'}",
                   "--verdict-out", str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["hypothesis_i_real_numerics"] == "LIVE" and got["states"]["c"] == "MISSING"


def test_cli_refusals(tmp_path, capsys):
    assert S.main(["--run-dir", str(tmp_path), "--out", str(tmp_path / "o")]) == 2          # neither dump nor anchor
    assert "exactly one" in capsys.readouterr().err
    assert S.main(["verdict"]) == 2
    assert S.main(["--dump", "x", "--anchor", "y", "--run-dir", str(tmp_path), "--out", str(tmp_path / "o")]) == 2


# ------------------------------------------------------------------------------------------------ the real path
@pytest.mark.slow  # the production-surface learner: ~1-2 min on CPU beside a live run
def test_the_dither_read_runs_on_the_real_learner_cpu():
    """`dither_read` on slice 2 at B 64 with R1 under dynamo's eager backend: every judged parameter carries one number
    per seed per arm, compiled == eager (the eager backend IS eager), the fp32 null is non-zero (the dither moved the
    function) and the weights are restored bit-exactly."""
    import torch

    from agents.model import compile_control as cc
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    m = LG.build_learner()
    m.batch_size = 64
    m.behaviour_dump_dir = None
    before = {n: p.detach().clone() for n, p in m.policy.named_parameters()}
    try:
        R.install_compiled(m, backend="eager")
        d = S.dither_read(m, 2, 64, compiled=True, seeds=2)
        assert d["judged"] > 0 and len(d["relocated_per_seed"]) == 2
        some = next(iter(d["params"].values()))
        assert len(some["null"]) == len(some["eager"]) == len(some["compiled"]) == 2
        assert all(c == pytest.approx(e, rel=1e-6, abs=1e-9) for v in d["params"].values()
                   for c, e in zip(v["compiled"], v["eager"]))
        assert max(max(v["null"]) for v in d["params"].values()) > 0
        assert all(torch.equal(p, before[n]) for n, p in m.policy.named_parameters())
    finally:
        from agents.model import compile_regions as cr
        cr.uninstall(m)
        cc._reset_control_for_tests()
        torch._dynamo.reset()
