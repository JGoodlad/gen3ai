"""`main.ops.canary_dump_replay`: the pre-declared summary rule (each class, the ceil(N/2) boundary), the validity
check against the dump's record, anchor mode's self-validity and refusals, the sha256 refusal, the out-dir refusals, and
(slow) the real learner's E32 / Ebf
path on CPU plus the canary's own check through dynamo's eager backend."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from main.ops import canary_dump_replay as R

K = 12.0
EDGE = "features_extractor.edge_bias.c2_map.weight"


def _edge(ratio):
    return {EDGE: {"ratio": ratio, "compiled": ratio * 0.05, "eager": 0.05}}


# ------------------------------------------------------------------------------------------------ the rule
def test_classify_each_class_and_the_boundary_band():
    assert R.classify(True, _edge(23.2), K) == "refused_edge_bias"
    assert R.classify(True, _edge(3.0), K) == "refused_other"          # refused, but not on edge_bias
    assert R.classify(False, _edge(3.0), K) == "pass"
    assert R.classify(False, _edge(K * (1 + 0.5 * R.BOUNDARY_REL)), K) == "boundary"
    assert R.classify(True, _edge(K * (1 - 0.5 * R.BOUNDARY_REL)), K) == "boundary"
    assert R.classify(True, _edge(float("inf")), K) == "refused_edge_bias"
    assert R.classify(True, {EDGE: {"eager": 0.05}}, K) == "refused_other"   # an eager-only row has no ratio


def _reps(n, fired, other=None):
    """n repeats; the first ``fired`` refuse on edge_bias at slice 2; slices 0 / 1 as ``other`` (default pass)."""
    return [{0: other or "pass", 2: "refused_edge_bias" if i < fired else "pass", 1: other or "pass"}
            for i in range(n)]


@pytest.mark.parametrize("n,fired,verdict", [(5, 3, "REPRODUCES"), (5, 2, "MIXED"), (5, 5, "REPRODUCES"),
                                             (4, 2, "REPRODUCES"), (4, 1, "MIXED"), (1, 1, "REPRODUCES"),
                                             (5, 0, "DOES NOT REPRODUCE")])
def test_summary_rule_at_ceil_half(n, fired, verdict):
    assert R.summarize(_reps(n, fired), valid=True, compiled=True)["verdict"] == verdict


def test_summary_any_refusal_or_boundary_elsewhere_is_mixed_never_does_not_reproduce():
    assert R.summarize(_reps(5, 0, other="refused_other"), True, True)["verdict"] == "MIXED"
    assert R.summarize(_reps(5, 0, other="boundary"), True, True)["verdict"] == "MIXED"
    assert R.summarize(_reps(5, 0, other="refused_edge_bias"), True, True)["verdict"] == "MIXED"


def test_summary_invalid_replay_overrides_and_cpu_is_eager_only():
    assert R.summarize(_reps(5, 5), valid=False, compiled=True)["verdict"] == "INVALID-REPLAY"
    assert R.summarize([{2: "pass"}], valid=True, compiled=False)["verdict"] == "EAGER-ONLY"
    assert R.summarize([{2: "pass"}], valid=False, compiled=False)["verdict"] == "INVALID-REPLAY"
    s = R.summarize(_reps(5, 3), True, True)
    assert s["need"] == 3 and s["counts"]["2"]["refused_edge_bias"] == 3 and s["rule"] == R.RULE_TEXT


# ------------------------------------------------------------------------------------------------ validity
def _record():
    cm = {"relocated_rows": 704, "rows": 2048, "nearest_unmoved": 1.036}
    return {"detail": f"x (e.g. {EDGE} compiled 1.27e+00 vs eager 5.47e-02)",
            "watch": {"confirm_slice2": {"clip_margin": cm, "global": {"drift_grad": 0.135},
                                         "worst": {"param": EDGE, "eager": 0.05715},
                                         "params": {"pointer_head.x.weight": {"eager": 0.04}}}}}


def _direct(reloc=704, eager=0.05715, drift=0.135, ph=0.04):
    return {2: {"relocated_rows": reloc, "rows": 2048, "nearest_unmoved": 1.0362,
                "per_param": {EDGE: {"eager": eager}, "pointer_head.x.weight": {"eager": ph}},
                "watch": {"global": {"drift_grad": drift}}}}


def test_validity_passes_on_the_records_numbers_and_fails_on_a_moved_state():
    assert R.validity(_direct(), {}, _record(), "cuda")["valid"]
    bad = R.validity(_direct(reloc=703), {}, _record(), "cuda")
    assert not bad["valid"] and bad["failures"][0]["quantity"] == "relocated_rows"
    assert not R.validity(_direct(eager=0.0625), {}, _record(), "cuda")["valid"]     # 9 % off on CUDA
    assert not R.validity(_direct(drift=0.15), {}, _record(), "cuda")["valid"]
    assert not R.validity(_direct(ph=0.08), {}, _record(), "cuda")["valid"]
    # CPU: the eager numbers are reported (with their ratio), never judged; the STATE still is
    cpu = R.validity(_direct(eager=0.17, ph=0.08, drift=0.18), {}, _record(), "cpu")
    assert cpu["valid"] and any(r.get("ratio") and r["judged"] is False for r in cpu["rows"])
    assert not R.validity(_direct(reloc=703), {}, _record(), "cpu")["valid"]


# ------------------------------------------------------------------------------------------------ anchor mode
def _slice(reloc=40, near=0.75, drift=0.2, eager=0.06, c_reloc=None, c_near=None, c_drift=None, c_eager=None,
           canary=True):
    d = {"relocated_rows": reloc, "rows": 2048, "nearest_unmoved": near, "watch": {"global": {"drift_grad": drift}},
         "per_param": {EDGE: {"eager": eager, "compiled": 1.0}, "pointer_head.x.weight": {"eager": 9.9}}}
    c = {"watch": {"clip_margin": {"relocated_rows": reloc if c_reloc is None else c_reloc, "rows": 2048,
                                   "nearest_unmoved": near if c_near is None else c_near},
                   "global": {"drift_grad": drift if c_drift is None else c_drift},
                   "edge_bias": {EDGE: {"eager": eager if c_eager is None else c_eager, "compiled": 1.0}}}}
    return {"direct": d, "canary": c if canary else None}


def _areps(*per_repeat):
    return [{"slices": {"2": sr}} for sr in per_repeat]


def test_self_validity_passes_on_self_consistent_numbers():
    v = R.self_validity(_areps(_slice(), _slice(eager=0.06 * 1.04, drift=0.2 * 0.97)), (2,))
    assert v["valid"] and v["n_failures"] == 0 and v["n_rows"] > 0


@pytest.mark.parametrize("bad", [dict(reloc=41), dict(near=0.76), dict(eager=0.066), dict(drift=0.23)])
def test_self_validity_fails_when_a_repeat_moves(bad):
    v = R.self_validity(_areps(_slice(), _slice(**bad)), (2,))
    assert not v["valid"] and all("vs repeat 0" in f["quantity"] or "vs direct" in f["quantity"]
                                  for f in v["failures"])


@pytest.mark.parametrize("bad", [dict(c_reloc=39), dict(c_near=0.76), dict(c_eager=0.07), dict(c_drift=0.25)])
def test_self_validity_fails_when_the_canary_and_direct_arms_disagree(bad):
    v = R.self_validity(_areps(_slice(**bad)), (2,))
    assert not v["valid"] and all(f["quantity"].endswith("vs direct") for f in v["failures"])


def test_self_validity_judges_only_edge_bias_and_cpu_without_a_canary_still_checks_repeats():
    assert R.self_validity(_areps(_slice(canary=False), _slice(canary=False)), (2,))["valid"]
    assert not R.self_validity(_areps(_slice(canary=False), _slice(canary=False, reloc=1)), (2,))["valid"]
    assert not R.self_validity([], (2,))["valid"]                  # nothing read is never valid


def _zip(path: Path, members=("policy.pth", "data")) -> Path:
    import zipfile
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for m in members:
            z.writestr(m, b"x")
    return path


def test_anchor_must_be_a_policy_zip_inside_the_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for f in ("model_config.json", "metadata.json"):
        (run / f).write_text("{}")
    good = _zip(run / "checkpoints" / "checkpoint_1_steps.zip")
    assert R.check_anchor(good, run.resolve()) == good.resolve()
    with pytest.raises(R.Refusal, match="not inside"):
        R.check_anchor(_zip(tmp_path / "other" / "final_model.zip"), run.resolve())
    with pytest.raises(R.Refusal, match="policy.pth"):
        R.check_anchor(_zip(run / "nopolicy.zip", members=("data",)), run.resolve())
    with pytest.raises(R.Refusal, match="no anchor"):
        R.check_anchor(run / "missing.zip", run.resolve())
    (run / "metadata.json").unlink()
    with pytest.raises(R.Refusal, match="metadata.json"):
        R.check_anchor(good, run.resolve())


def test_compile_keys_refuses_a_route_the_replay_does_not_compile():
    from types import SimpleNamespace
    ok = SimpleNamespace(learner_cudagraphs="off", grad_checkpointing=False, trunk_precision="bf16", batch_size=4096)
    assert R.compile_keys(ok)["trunk_precision"] == "bf16"
    for k, v in (("learner_cudagraphs", "on"), ("grad_checkpointing", True), ("trunk_precision", "fp32")):
        bad = SimpleNamespace(**{**vars(ok), k: v})
        with pytest.raises(R.Refusal, match=k.replace("_", "-")):
            R.compile_keys(bad)


def test_main_needs_exactly_one_of_dump_or_anchor(tmp_path, capsys):
    base = ["--run-dir", str(tmp_path), "--out", str(tmp_path / "o"), "--device", "cpu"]
    assert R.main(base) == 2 and "exactly one" in capsys.readouterr().err
    assert R.main(["--dump", "a.pt", "--anchor", "b.zip", *base]) == 2 and "exactly one" in capsys.readouterr().err
    assert R.main(["--anchor", "b.zip", "--verdicts", "v.jsonl", *base]) == 2


# ------------------------------------------------------------------------------------------------ refusals
def _dump(tmp_path: Path, good: bool = True) -> Path:
    d = tmp_path / "arch"
    d.mkdir()
    p = d / "canary_u1_unconfirmed_policy.pt"
    p.write_bytes(b"weights")
    h = hashlib.sha256(b"weights" if good else b"other").hexdigest()
    (d / "SHA256SUMS").write_text(f"{h}  {p.name}\n")
    return p


def test_sha_refusal(tmp_path):
    good = _dump(tmp_path)
    assert R.check_dump_sha(good) == hashlib.sha256(b"weights").hexdigest()
    good.write_bytes(b"tampered")
    with pytest.raises(R.Refusal, match="MISMATCH"):
        R.check_dump_sha(good)
    (good.parent / "SHA256SUMS").unlink()
    with pytest.raises(R.Refusal, match="SHA256SUMS"):
        R.check_dump_sha(good)


def test_main_exits_2_on_a_bad_sha(tmp_path, capsys):
    bad = _dump(tmp_path, good=False)
    rc = R.main(["--dump", str(bad), "--run-dir", str(tmp_path), "--out", str(tmp_path / "o"), "--device", "cpu"])
    assert rc == 2 and "MISMATCH" in capsys.readouterr().err


def test_out_dir_refused_inside_models_or_a_checkout_or_non_empty(tmp_path):
    models = tmp_path / "models"
    (models / "run").mkdir(parents=True)
    with pytest.raises(R.Refusal, match="models/"):
        R.check_out_dir(models / "run" / "replay", models_dir=models)
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(R.Refusal, match="checkout"):
        R.check_out_dir(repo / "a" / "b", models_dir=models)
    (repo / "wt").mkdir()
    (repo / "wt" / ".git").write_text("gitdir: x")                # a worktree's .git is a FILE
    with pytest.raises(R.Refusal, match="checkout"):
        R.check_out_dir(repo / "wt" / "o", models_dir=models)
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("1")
    with pytest.raises(R.Refusal, match="not empty"):
        R.check_out_dir(full, models_dir=models)
    assert R.check_out_dir(tmp_path / "fresh", models_dir=models) == (tmp_path / "fresh").resolve()


def test_the_record_is_the_one_naming_the_dump(tmp_path):
    v = tmp_path / "canary_verdicts.jsonl"
    v.write_text("\n".join(json.dumps(r) for r in ({"update": 10, "verdict": "pass"},
                                                   {"update": 200, "verdict": "unconfirmed", "policy_pt": "a.pt"})))
    assert R.recorded(v, "a.pt")["update"] == 200
    with pytest.raises(R.Refusal):
        R.recorded(v, "b.pt")


# ------------------------------------------------------------------------------------------------ the real path
@pytest.mark.slow  # the production-surface learner: ~30-60 s on CPU beside a live run
def test_the_eager_arms_and_the_canary_check_run_on_the_real_learner_cpu():
    """E32 / Ebf through `direct_arms` (Cbf SKIPPED), then the canary's own `_check` with R1 installed under dynamo's
    eager backend (compiled == eager arithmetic, so it PASSES and every edge_bias ratio is ~1)."""
    import torch

    from agents.model import compile_control as cc
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    m = LG.build_learner()
    m.batch_size = 64
    m.behaviour_dump_dir = None
    try:
        d = R.direct_arms(m, 2, 64, compiled=False)
        assert d["grad_sha256"]["Cbf"].startswith("SKIPPED")
        assert d["rows"] == 64 and 0 <= d["relocated_rows"] <= 64
        assert d["watch"]["edge_bias"], "the production surface judges edge_bias.* parameters"
        assert all("compiled" not in v and "eager" in v for v in d["watch"]["edge_bias"].values())
        assert d["grad_sha256"]["E32"] != d["grad_sha256"]["Ebf"]          # the two precisions really differ
        R.install_compiled(m, backend="eager")
        c = R.canary_check(m, 2, 64)
        assert c["refused"] is False, c["text"][:500]
        assert R.classify(c["refused"], c["watch"]["edge_bias"], K) == "pass"
        d2 = R.direct_arms(m, 2, 64, compiled=True)
        assert d2["grad_sha256"]["Cbf"] == d2["grad_sha256"]["Ebf"]        # the eager backend IS eager
        assert d2["grad_sha256"]["E32"] == d["grad_sha256"]["E32"]         # the reference is deterministic
    finally:
        from agents.model import compile_regions as cr
        cr.uninstall(m)
        cc._reset_control_for_tests()
        torch._dynamo.reset()


def test_the_replay_judges_by_the_RULE_its_record_names():
    """gen3_canary_row_bar_v1: a global-M record (no field) replays at the scalar margin; a per-row record replays
    against ITS OWN recorded bars (or the cap when it fell back); an unknown rule refuses."""
    from agents.model import canary_row_bar as RB
    from main.ops import canary_dump_replay as R
    assert R.replay_row_bars({"verdict": "unconfirmed"}) is None
    assert R.replay_row_bars({"canary_rule_version": RB.LEGACY_RULE_VERSION}) is None
    live = R.replay_row_bars({"canary_rule_version": RB.CANARY_RULE_VERSION,
                              "canary_rule": {"bar_per_row": [0.12, 0.72], "row_bars_source": "null"}})
    assert live.live and live.bars.tolist() == [0.12, 0.72]
    fb = R.replay_row_bars({"canary_rule_version": RB.CANARY_RULE_VERSION,
                            "canary_rule": {"row_bars_source": "fallback: the null skipped"}})
    assert not fb.live and "skipped" in fb.source
    import pytest as _pt
    with _pt.raises(R.Refusal, match="per_row_stage9"):
        R.replay_row_bars({"canary_rule_version": "per_row_stage9"})
