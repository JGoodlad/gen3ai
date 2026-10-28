"""THE SHADOW PER-ROW DITHER NULL (`gen3_canary_row_null_shadow_v1`; `canary_row_null`): recorded at every canary,
judging NOTHING.

On CPU, on the K9 golden's real learner (no compile; the scheduled check's arms planted through the REAL `_r1_judge`
and margin rule, `canary_yardstick_test._Rung`):

* the null restores every parameter bit-exactly, the training mode, the fp32 matmul precision and the global RNG, and
  the SAME weights give the SAME null (fixed seeds);
* its dither is the design's (`main.ops.diag_bf16_canary.dithered`, relative 2^-9, name-keyed): seed s's per-row
  value is exactly that dither's;
* the cap SKIPS with its reason, an error is RECORDED (never raised), a malformed cap refuses at construction;
* a null that CANNOT BIND (here K = 3 seeds, not the rule's 16: `canary_row_bar.from_null` falls back to the cap = M)
  leaves the canary's verdict, its TB scalars and every record field but ``row_null`` / ``canary_rule`` IDENTICAL with
  the null on and off — for a PASS, an UNCONFIRMED and a FATAL — and the ON record carries the per-row null, each
  arm's per-row shift and the per-row ratio. The BINDING null is `canary_row_bar_test`'s.

Planted shifts sit far from every bar (rule 8): PASS 0.30 / 0.20 vs M 0.72; the breach 0.90."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from agents.model import canary_row_null as RN
from agents.model import compile_canary as CC
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model.canary_yardstick_test import ROWS, _Rung

K = 3          # seeds per null here (production: `ROW_NULL_SEEDS`): each is one B = 64 forward, ~0.2 s on CPU
CAP = 600.0    # far above the CPU cost, so no read here depends on the clock


@pytest.fixture(scope="module")
def learner():
    from agents.training import learner_golden as LG
    m = LG.build_learner()
    LG.load_buffer_into(m)
    return m


def _params(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.policy.named_parameters()}


def _same(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a)


# ----------------------------------------------------------------------------------- the null itself
def test_the_null_changes_nothing_and_the_same_weights_give_the_same_null(learner):
    before = _params(learner)
    learner.policy.set_training_mode(False)
    prec = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    rng = torch.get_rng_state()
    try:
        a = RN.compute(learner, k=K, cap_s=CAP)
        b = RN.compute(learner, k=K, cap_s=CAP)
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(prec)
    assert torch.equal(torch.get_rng_state(), rng)
    assert learner.policy.training is False
    learner.policy.set_training_mode(True)
    assert _same(before, _params(learner))
    assert "error" not in a and "skipped" not in a and "truncated" not in a, a
    n = a["rows"]
    assert a["per_seed"].shape == (K, n) and a["null"].shape == (n,)
    assert np.array_equal(a["null"], a["per_seed"].max(axis=0))
    assert np.array_equal(a["per_seed"], b["per_seed"])                  # deterministic: fixed seeds
    assert a["ref_repeat_max_abs"] == 0.0                                # the fp32 forward repeats bit-exactly
    assert float(a["null"].max()) > 0.0                                  # the dither moved the function


def test_the_dither_is_the_designs_dither(learner):
    """Seed s's per-row value is exactly `diag_bf16_canary.dithered`'s seed s on the same rows (CPU: the same draws)."""
    from agents.model.trunk_precision import fp32_scope
    from main.ops import diag_bf16_canary as D
    assert RN.ROW_NULL_REL == D.DITHER_REL
    raw = RN.compute(learner, k=2, cap_s=CAP)
    n = raw["rows"]
    b = cr.r1_batch(learner, n, 0)

    def lp() -> torch.Tensor:
        learner.policy.set_training_mode(True)
        with fp32_scope(learner.policy), torch.no_grad():
            out = learner.policy.evaluate_actions_functional(b.obs, b.actions.long().flatten(), b.action_masks)
        return out[1].detach().double().cpu()
    ref = lp()
    for s in range(2):
        with D.dithered([learner], s):
            got = (lp() - ref).abs().numpy()
        assert np.array_equal(got, raw["per_seed"][s]), s


def test_the_cap_SKIPS_with_its_reason_and_zero_turns_it_off(learner, monkeypatch):
    before = _params(learner)
    raw = RN.compute(learner, k=K, cap_s=1e-9)
    assert "projected" in raw["skipped"] and "null" not in raw and raw["ref_forward_s"] > 0
    rec = RN.record(raw)
    assert rec["skipped"] == raw["skipped"] and "null_per_row" not in rec and rec["judged"] is False
    assert RN.compute(learner, cap_s=0.0)["skipped"].startswith("off")
    monkeypatch.setenv(RN.CAP_ENV, "0")
    assert RN.compute(learner)["skipped"].startswith("off")
    assert _same(before, _params(learner))
    monkeypatch.setenv(RN.CAP_ENV, "soon")
    with pytest.raises(ValueError, match=RN.CAP_ENV):
        RN.cap_seconds()
    with pytest.raises(ct.CompileTrainerError, match=RN.CAP_ENV):     # refused at construction, never at a canary
        CC.CompileCanary(learner, batch_size=2 * ROWS, every=1)
    CC.CompileCanary(learner, batch_size=2 * ROWS, every=1, row_null=False)   # off: the cap is not read


def test_an_error_is_RECORDED_and_every_parameter_restored(learner, monkeypatch):
    before = _params(learner)
    calls = {"n": 0}
    real = learner.policy.evaluate_actions_functional

    def flaky(*a: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 4:                                # inside the seed loop (after 2 reference calls), past a rewrite
            raise RuntimeError("planted")
        return real(*a)
    monkeypatch.setattr(learner.policy, "evaluate_actions_functional", flaky)
    raw = RN.compute(learner, k=K, cap_s=CAP)
    assert raw["error"] == "RuntimeError: planted" and "null" not in raw
    monkeypatch.undo()
    assert _same(before, _params(learner))


def test_the_record_carries_each_arms_per_row_ratio_to_the_floored_null():
    null = np.array([0.0, 0.1, 0.5])
    raw = {"k": 2, "rows": 3, "seconds": 0.5, "null": null, "per_seed": np.array([[0.0, 0.1, 0.2], [0.0, 0.05, 0.5]])}
    rec = RN.record(raw, {"compiled": np.array([0.01, 0.3, 0.25]), "eager": np.array([1.0, 2.0])})
    c = rec["arms"]["compiled"]
    assert c["ratio_per_row"] == [0.5, 3.0, 0.5]                          # 0.01 / max(0, 0.02): the floor
    assert c["ratio"]["max"] == 3.0 and c["ratio"]["argmax_row"] == 1
    assert "error" in rec["arms"]["eager"]                                # a shape mismatch is named, not guessed
    assert rec["seed_worst_row"] == [2, 2] and rec["null"]["max"] == 0.5 and rec["seeds_done"] == 2


def test_the_watch_records_each_arms_shift_PER_FIXTURE_ROW_as_the_max_over_its_tiles():
    w = cr.GradWatch(())
    w.fixture_rows = [0, 1, 0, 1]
    ref = torch.tensor([-1.0, -2.0, -1.0, -2.0], dtype=torch.float64)
    fix = cr.clip_relocation(ref, ref.float(), torch.ones(4, dtype=torch.float64), 0.2, 0.72)
    arm = {"log_prob": torch.tensor([-1.1, -2.0, -0.7, float("nan")])}
    w.fill_rows(fix, {"compiled": arm})
    assert np.allclose(w.row_dlogp["compiled"][0], 0.3) and w.row_dlogp["compiled"][1] == float("inf")
    assert "row_dlogp" not in json.dumps(w.record())                      # not part of the watch record


# ----------------------------------------------------------------------------------- the canary: verdict unchanged
def _world(monkeypatch, tmp, learner, row_null: bool) -> Tuple[CC.CompileCanary, Dict[str, Any], List[str]]:
    monkeypatch.setattr(CC, "_WEIGHT_DUMPS", 0)
    plan: Dict[str, Any] = {"checks": [], "rows": {"eager": 4, "compiled": 20}}
    said: List[str] = []

    def check(self, slice_, out, watch=None, row_bars=None):
        shifts = plan["checks"].pop(0)
        fr = ct.fixture_index(ROWS, ROWS, slice_)
        if watch is not None:                              # B = ROWS: batch row == fixture row (slice 0)
            watch.fixture_rows = [int(x) for x in fr]
        with monkeypatch.context() as mp:
            _Rung(mp, shifts, plan["rows"])
            model = SimpleNamespace(policy=None)
            bb = None if row_bars is None else row_bars.for_batch(fr)
            return [cr._r1_judge(model, ("args",), ["pointer_head.w"], "trained", watch=watch, row_bars=bb)[2]]
    monkeypatch.setattr(CC.CompileCanary, "_check", check)
    learner.behaviour_dump_dir = str(tmp)
    learner.num_timesteps = 478_310_831
    can = CC.CompileCanary(learner, batch_size=ROWS, every=1, saver=lambda _p: None, emit=said.append,
                           row_null=row_null)
    return can, plan, said


_VOLATILE = ("compile/canary_seconds", "compile/canary_row_null_seconds")


def _run(monkeypatch, tmp, learner, row_null: bool, checks: List[Any]) -> Tuple[Any, List[Dict[str, Any]]]:
    monkeypatch.setattr(RN, "ROW_NULL_SEEDS", K)
    monkeypatch.setattr(RN, "ROW_NULL_CAP_S", CAP)
    can, plan, _said = _world(monkeypatch, tmp, learner, row_null)
    outs: List[Any] = []
    for c in checks:
        plan["checks"] = list(c)
        try:
            outs.append({k: v for k, v in can.after_update().items() if k not in _VOLATILE})
        except CC.CompileCanaryError as exc:
            outs.append(("FATAL", str(exc).split("Checkpoint:")[0]))
    recs = [json.loads(x) for x in (tmp / CC.VERDICTS_FILE).read_text().splitlines()]
    return outs, recs


_PASS = {"compiled": 0.30, "eager": 0.20}
_BREACH = {"compiled": 0.90, "eager": 0.10}
_UNREPRODUCIBLE = [{"eager": 0.856, "compiled": 0.52}, {"eager": 1.5, "compiled": 0.52}]
SEQUENCE = [[[_PASS]], [_UNREPRODUCIBLE, [_PASS], _UNREPRODUCIBLE], [[_BREACH]] * 3]   # PASS, UNCONFIRMED, FATAL


def test_the_verdict_is_IDENTICAL_with_the_shadow_on_and_off(learner, monkeypatch, tmp_path):
    """The shadow judges nothing: the same planted checks read the same verdicts, TB scalars and records (but
    ``row_null``) whether it runs or not."""
    before = _params(learner)
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    on, recs_on = _run(monkeypatch, tmp_path / "on", learner, True, SEQUENCE)
    off, recs_off = _run(monkeypatch, tmp_path / "off", learner, False, SEQUENCE)
    assert on == off
    assert [r["verdict"] for r in recs_on] == ["pass", "unconfirmed", "fatal"]
    strip = lambda r: {k: v for k, v in r.items() if k not in ("row_null", "canary_rule", "time")}   # noqa: E731
    assert [strip(r) for r in recs_on] == [strip(r) for r in recs_off]
    assert all("row_null" not in r for r in recs_off)
    assert all(r["canary_rule"]["row_bars_source"].startswith("fallback: the null has 3 seeds") for r in recs_on)
    assert all(r["canary_rule"]["row_bars_source"].startswith("fallback: the per-row null is off") for r in recs_off)
    assert _same(before, _params(learner))
    rn = recs_on[0]["row_null"]
    assert rn["judged"] is False and rn["seeds_done"] == K and len(rn["null_per_row"]) == ROWS, rn
    comp = rn["arms"]["compiled"]
    assert comp["dlogp"]["argmax_row"] == 20 and comp["dlogp"]["max"] == pytest.approx(0.30, rel=1e-3)
    assert comp["ratio_per_row"][20] == pytest.approx(0.30 / max(rn["null_per_row"][20], RN.ROW_NULL_FLOOR),
                                                      rel=2e-3)
    assert all("row_null" in r for r in recs_on)                          # UNCONFIRMED and FATAL records carry it too
