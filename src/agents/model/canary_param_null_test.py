"""THE PER-PARAMETER DITHER NULL (`gen3_canary_param_null_v1`; `canary_param_null`): the canary's bf16 per-parameter
bar is K x max(eager bf16's error, the fp32 dither null, the floor), the null computed only when the eager bar alone
would refuse.

CPU. The rule runs through the REAL `compile_regions._r1_verdict_bf16` on planted arms (no learner); the null's own
mechanics on a toy policy with the reference arm planted; the real K9 learner (slow) for the dither's parity with the
replay's census and for an injected miscompile. Every test here fails on a revert of the change:

* THE FALSE ALARM IS GONE: a parameter whose compiled error passes 12 x eager but sits inside 12 x its null PASSES
  (the eager bar alone refused it), and the refusal / rule line / watch / ``bar_fractions`` name the null;
* THE TEETH STAY: an injected miscompile (one parameter's compiled gradient scaled x3, error 2.0) is REFUSED with the
  null bound; on the real learner a DOUBLED gradient on its largest parameter is refused after a real 16-seed null;
* EXACTLY LAZY: no null when every parameter passes the eager bar; one null over every judged parameter otherwise;
* THE FALLBACK: a null that is off, skipped, truncated, in error or short of 16 seeds leaves the eager bar (refuses);
* THE BACKTEST on the 2026-10-30 GPU replay (`testdata/canary_param_null_basis.json`): the 14 refusing reads at the five
  states' own weights PASS (max 1.60 of K's 12; the eager bar refused each at 12.2-52.4), and every one of the census's
  42 dithered reads above K, and its whole new-statistic tail, passes (max 9.18);
* the canary names the rule (``canary_rule_version`` `per_row_stage1_param_null`, ``canary_rule.param_null``), and the
  replay tools judge a record by the rule it names.

Planted values sit far from every bar (rule 8): ratios 3, 30, 200 against K 12."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from agents.model import canary_param_null as PN
from agents.model import canary_row_bar as RB
from agents.model import compile_canary as CC
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct

BASIS = os.path.join(os.path.dirname(__file__), "testdata", "canary_param_null_basis.json")
K = 12.0
FLOOR = 0.00988
NAMES = ["big.w", "carried.w", "plain.w", "value.w"]
BIG = 100.0                    # the big parameter's entries: it holds the global gradient, so only per-param rules bind


def _arms(c_err: List[float], e_err: List[float]) -> Dict[str, Dict[str, torch.Tensor]]:
    """Reference / eager / compiled arms whose per-parameter relative errors are exactly ``e_err`` / ``c_err``."""
    ref = [torch.full((4,), BIG if i == 0 else 1.0) for i in range(len(NAMES))]

    def arm(err: List[float]) -> Dict[str, torch.Tensor]:
        g = torch.cat([r * (1.0 + e) for r, e in zip(ref, err)])
        return {"grad": g, "grad_sizes": torch.tensor([4] * len(NAMES)), "loss": torch.ones(1)}
    return {"ref": arm([0.0] * len(NAMES)), "eager": arm(e_err), "comp": arm(c_err)}


def _verdict(c_err: List[float], e_err: List[float], null_fn: Any = None, watch: Any = None) -> str:
    a = _arms(c_err, e_err)
    return cr._r1_verdict_bf16(a["eager"], a["comp"], a["ref"], NAMES, "trained", watch=watch, param_null=null_fn)


class _Null:
    """A planted null function: counts its calls, returns `compute`'s shape with ``null`` per index."""

    def __init__(self, null: Dict[int, float], **extra: Any) -> None:
        self.null, self.extra, self.calls = null, extra, []

    def __call__(self, idx: List[int]) -> Dict[str, Any]:
        self.calls.append(list(idx))
        out = {"version": PN.PARAM_NULL_VERSION, "seeds": 16, "seeds_done": 16, "seconds": 1.5,
               "null": {i: self.null.get(i, 0.0) for i in idx}}
        out.update(self.extra)
        return out


E = [0.01, 0.01, 0.01, 0.01]
CARRIED = [0.01, 0.30, 0.01, 0.01]            # carried.w: 0.30 / max(0.01, floor) = 30 > 12 under the eager bar
NULL = {0: 0.001, 1: 0.10, 2: 0.002, 3: 0.003}  # carried.w's null 0.10: 0.30 / 0.10 = 3 <= 12


# ------------------------------------------------------------------------------------------- the rule
def test_the_constants_are_the_replays():
    assert (PN.PARAM_NULL_SEEDS, PN.PARAM_NULL_REL) == (16, 2.0 ** -9)
    assert cr.BF16_ENVELOPE_K is not None and cr.BF16_ENVELOPE_K["param"] == K       # K 12 kept (ruled 10-28)
    assert RB.CANARY_RULE_VERSION == "per_row_stage1_param_null" and RB.STAGE1_RULE_VERSION == "per_row_stage1"


def test_a_SINGLE_ROW_CARRIED_parameter_inside_its_null_PASSES_where_the_eager_bar_refused():
    """Fails on revert: the eager bar alone refuses carried.w at 30 x."""
    with pytest.raises(ct.CompileTrainerError, match=r"than 12 x eager bf16's own error on 1 parameter\(s\) "
                                                     r"\(e\.g\. carried\.w compiled 3\.00e-01 vs eager 1\.00e-02\)"):
        _verdict(CARRIED, E)
    null = _Null(NULL)
    w = cr.GradWatch(("carried.",))
    line = _verdict(CARRIED, E, null, w)
    assert null.calls == [[0, 1, 2, 3]]                          # ONE null, over every judged parameter
    assert "per-param worst ratio 3.00 <= 12 (carried.w: compiled 3.00e-01 vs eager 1.00e-02, null 1.00e-01)" in line
    assert "per-parameter null (gen3_canary_param_null_v1): null, 16 seeds" in line, line
    rec = w.record()
    assert rec["worst"]["param"] == "carried.w" and rec["worst"]["ratio"] == pytest.approx(3.0, rel=1e-4)
    p = rec["params"]["carried.w"]
    assert p["ratio"] == pytest.approx(3.0, rel=1e-4) and p["ratio_eager"] == pytest.approx(30.0, rel=1e-4)
    assert p["null"] == pytest.approx(0.10)
    pn = rec["param_null"]
    assert pn["source"] == "null" and pn["seeds_done"] == 16 and set(pn["params"]) == {"carried.w"}
    assert pn["params"]["carried.w"]["ratio"] == pytest.approx(3.0, rel=1e-3)
    frac = CC.bar_fractions(w)["param_ratio"]                    # the near-margin trigger reads the JUDGED ratio
    assert frac["value"] == pytest.approx(3.0, rel=1e-4) and frac["fraction"] == pytest.approx(0.25, rel=1e-4)


def test_an_INJECTED_MISCOMPILE_is_REFUSED_with_the_null_bound():
    """One parameter's compiled gradient scaled x3 (error 2.0): 2.0 / max(0.01, its null 0.01) = 200 > 12."""
    scaled = [0.01, 0.01, 2.0, 0.01]
    null = _Null({0: 0.001, 1: 0.10, 2: 0.01, 3: 0.003})
    w = cr.GradWatch(("plain.",))
    with pytest.raises(ct.CompileTrainerError, match=r"than 12 x max\(eager bf16's own error, its fp32 dither null\) "
                                                     r"on 1 parameter\(s\) \(e\.g\. plain\.w compiled 2\.00e\+00 vs "
                                                     r"eager 1\.00e-02, null 1\.00e-02\).*per-parameter null") as e:
        _verdict(scaled, E, null, w)
    assert not isinstance(e.value, cr.YardstickRefusal) and null.calls == [[0, 1, 2, 3]]
    assert w.record()["param_null"]["params"]["plain.w"]["ratio"] == pytest.approx(200.0, rel=1e-3)
    # the null cannot rescue a parameter whose error is far outside the function's own rounding move, even beside a
    # carried parameter it does rescue
    with pytest.raises(ct.CompileTrainerError, match=r"on 1 parameter\(s\) \(e\.g\. plain\.w"):
        _verdict([0.01, 0.30, 2.0, 0.01], E, null)


def test_EXACTLY_LAZY_no_null_when_every_parameter_passes_the_eager_bar():
    null = _Null(NULL)
    w = cr.GradWatch(("carried.",))
    line = _verdict([0.01, 0.05, 0.01, 0.01], E, null, w)        # 0.05 / 0.01 = 5 <= 12
    assert null.calls == [] and "per-parameter null" not in line
    assert "param_null" not in w.record() and w.record()["params"]["carried.w"]["ratio"] == pytest.approx(5.0, rel=1e-4)


def test_the_new_bar_is_NEVER_below_the_eager_one():
    """A null below eager's error changes nothing: a parameter the eager bar refuses stays refused."""
    with pytest.raises(ct.CompileTrainerError, match="carried.w"):
        _verdict(CARRIED, E, _Null({1: 0.001}))
    j = PN.judge({0: 0.5}, {0: 0.1}, {0: FLOOR}, K, _Null({0: 0.0}))
    assert j.ratios[0] == pytest.approx(5.0) and not j.bad


@pytest.mark.parametrize("extra,why", [
    ({"error": "RuntimeError: boom"}, "the null error"), ({"skipped": "off (cap 0 s)"}, "the null skipped"),
    ({"truncated": "the cap 180 s would pass"}, "the null truncated"), ({"seeds_done": 3}, "the null has 3 seeds")])
def test_a_null_that_CANNOT_BIND_falls_back_to_the_eager_bar_and_says_why(extra, why):
    w = cr.GradWatch(())
    with pytest.raises(ct.CompileTrainerError, match=r"than 12 x eager bf16's own error .*fallback: " + why) as e:
        _verdict(CARRIED, E, _Null(NULL, **extra), w)
    assert "carried.w compiled 3.00e-01 vs eager 1.00e-02)" in str(e.value)
    assert w.record()["param_null"]["source"].startswith("fallback")


def test_a_NON_FINITE_null_reads_zero_so_that_parameter_keeps_its_eager_bar():
    with pytest.raises(ct.CompileTrainerError, match="carried.w"):
        _verdict(CARRIED, E, _Null({1: float("nan")}))
    with pytest.raises(ct.CompileTrainerError, match="carried.w"):
        _verdict(CARRIED, E, _Null({1: float("inf")}))


def test_the_startup_gate_and_every_caller_without_the_flag_keep_the_eager_bar():
    """`param_null` defaults off at every layer: the gate, the perturbed rungs, the replay of an older record."""
    import inspect
    for fn in (cr.r1_rungs, cr._r1_judge):
        assert inspect.signature(fn).parameters["param_null"].default is False
    assert inspect.signature(cr._r1_verdict_bf16).parameters["param_null"].default is None
    with pytest.raises(ct.CompileTrainerError, match="carried.w"):
        _verdict(CARRIED, E, None)


# ------------------------------------------------------------------------------------------- the null itself
class _Toy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.a = torch.nn.Parameter(torch.randn(3, 5, generator=g))
        self.b = torch.nn.Parameter(torch.randn(5, generator=g))


def _plant_reference(monkeypatch, fail_at: int = -1) -> List[int]:
    """`_r1_reference` on the toy: the gradient of a fixed non-linear loss at the policy's CURRENT weights."""
    calls: List[int] = []

    def ref(model: Any, _args: Any) -> Dict[str, torch.Tensor]:
        calls.append(1)
        if len(calls) == fail_at:
            raise RuntimeError("planted")
        p = model.policy
        with torch.enable_grad():
            a, b = p.a.detach().requires_grad_(), p.b.detach().requires_grad_()
            loss = torch.tanh(a @ torch.sin(b * 3.0)).pow(3).sum()
            ga, gb = torch.autograd.grad(loss, [a, b])
        return {"grad": torch.cat([ga.flatten(), gb.flatten()]), "grad_sizes": torch.tensor([15, 5])}
    monkeypatch.setattr(cr, "_r1_reference", ref)
    monkeypatch.setattr(cr, "release_gate_memory", lambda _m: 0)
    return calls


def _toy() -> Any:
    return SimpleNamespace(policy=_Toy())


def test_the_null_is_the_replays_dither_and_restores_every_parameter_bit_exactly(monkeypatch):
    from main.ops.diag_bf16_canary import dithered
    _plant_reference(monkeypatch)
    m = _toy()
    before = {n: p.detach().clone() for n, p in m.policy.named_parameters()}
    rng = torch.get_rng_state()
    ref = cr._r1_reference(m, ())
    raw = PN.compute(m, (), ref, [0, 1], seeds=4, cap_s=600.0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(before[n], p) for n, p in m.policy.named_parameters())
    assert raw["seeds_done"] == 4 and "error" not in raw and "truncated" not in raw
    for s in range(4):                                     # seed s IS `dithered(models, s)`: the census's null
        with dithered([m], s, PN.PARAM_NULL_REL):
            g = torch.split(cr._r1_reference(m, ())["grad"].double(), [15, 5])
        base = torch.split(ref["grad"].double(), [15, 5])
        want = [float((g[i] - base[i]).norm() / base[i].norm()) for i in (0, 1)]
        assert raw["per_seed"][s] == want
    assert raw["null"] == {0: max(r[0] for r in raw["per_seed"]), 1: max(r[1] for r in raw["per_seed"])}
    assert 0.0 < raw["null"][0] < 1.0                      # the dither moved the (ill-conditioned) toy function
    again = PN.compute(m, (), ref, [0, 1], seeds=4, cap_s=600.0)
    assert again["per_seed"] == raw["per_seed"]            # fixed seeds: the same weights give the same null


def test_the_null_CAP_skips_truncates_and_an_error_is_recorded_with_the_weights_restored(monkeypatch):
    _plant_reference(monkeypatch)
    m = _toy()
    before = {n: p.detach().clone() for n, p in m.policy.named_parameters()}
    ref = cr._r1_reference(m, ())
    assert PN.compute(m, (), ref, [0], cap_s=0.0)["skipped"].startswith("off")
    t = iter([0.0, 10.0, 20.0])                    # seed 1 took 10 s: seed 2 would end at 20 s > the cap 15 s
    raw = PN.compute(m, (), ref, [0], seeds=16, cap_s=15.0, clock=lambda: next(t))
    assert raw["seeds_done"] == 1 and "would pass at seed 2" in raw["truncated"]
    assert PN.bind(raw)[0] is None and PN.bind(raw)[1].startswith("fallback: the null truncated")
    _plant_reference(monkeypatch, fail_at=3)       # the third reference call (seed 2) raises
    raw = PN.compute(m, (), ref, [0], seeds=16, cap_s=600.0)
    assert raw["error"] == "RuntimeError: planted" and raw["seeds_done"] == 2
    assert all(torch.equal(before[n], p) for n, p in m.policy.named_parameters())
    assert all(p.grad is None for p in m.policy.parameters())


def test_a_malformed_cap_refuses_at_CONSTRUCTION_and_zero_turns_the_null_off(monkeypatch):
    monkeypatch.setenv(PN.CAP_ENV, "soon")
    with pytest.raises(ct.CompileTrainerError, match="not a number of seconds"):
        CC.CompileCanary(SimpleNamespace(policy=torch.nn.Module()), batch_size=64, row_null=False)
    CC.CompileCanary(SimpleNamespace(policy=torch.nn.Module()), batch_size=64, row_null=False, param_null=False)
    monkeypatch.setenv(PN.CAP_ENV, "0")
    assert PN.cap_seconds() == 0.0


# ------------------------------------------------------------------------------------------- the canary and the replay
def test_the_canary_names_its_rule_and_the_replay_judges_a_record_by_ITS_rule():
    m = SimpleNamespace(policy=torch.nn.Module())
    on = CC.CompileCanary(m, batch_size=64, row_null=False)
    off = CC.CompileCanary(m, batch_size=64, row_null=False, param_null=False)
    assert on.param_null_on and on.rule_version == RB.CANARY_RULE_VERSION
    assert off.rule_version == RB.STAGE1_RULE_VERSION
    assert on._rule_record()["param_null"]["version"] == PN.PARAM_NULL_VERSION
    assert on._rule_record()["version"] == RB.CANARY_RULE_VERSION and off._rule_record()["param_null"] == "off"
    from main.ops import canary_dump_replay as R
    assert R.replay_param_null({"canary_rule_version": RB.CANARY_RULE_VERSION}) is True
    for old in ({}, {"canary_rule_version": RB.STAGE1_RULE_VERSION}, {"canary_rule_version": "global_m"}):
        assert R.replay_param_null(old) is False
    bars = {"canary_rule": {"bar_per_row": [0.72] * 64, "row_bars_source": "null"}}
    for v in (RB.STAGE1_RULE_VERSION, RB.CANARY_RULE_VERSION):
        assert R.replay_row_bars({"canary_rule_version": v, **bars}).live


# ------------------------------------------------------------------------------------------- the GPU backtest
def _basis() -> Dict[str, Any]:
    with open(BASIS) as f:
        out: Dict[str, Any] = json.load(f)
    return out


def test_the_BACKTEST_the_five_states_own_refusing_reads_PASS_under_the_null():
    """The 14 per-parameter refusals of the canary at the five states' own weights (GPU, 2026-10-30): the eager bar
    refused each (12.2-52.4 x); with the parameter's 16-seed null each reads at most 1.60 x (A1-arch
    `mon_hazard_proj.weight`, slice 2)."""
    rows = _basis()["undithered"]
    assert len(rows) == 14 and {r["state"] for r in rows} == {"a_x48kffn_s3502_u200", "b_n1p_s3505_u100",
                                                             "c_n1p_s3501_u200", "e_a1arch_s3503_u200"}
    worst = 0.0
    for r in rows:
        args = ({0: r["compiled"]}, {0: r["eager"]}, {0: r["floor"]}, K)
        old = PN.judge(*args)
        assert old.bad == [0] and old.ratios[0] > K, r
        new = PN.judge(*args, _Null({0: r["null_max"]}), [r["param"]])
        assert not new.bad, r
        worst = max(worst, new.ratios[0])
    assert worst == pytest.approx(1.602, abs=2e-3) and worst / K < 0.14


def test_the_BACKTEST_every_dithered_read_of_the_census_passes_and_the_tail_re_derives():
    """Over 62,880 dithered GPU reads the eager bar refused 42 and the new bar none; the committed tail (every entry
    with an eager-bar read over K or a new-bar read over 3) re-derives both: the verdicts read by read through `judge`
    (lazy: the null only where the eager bar refuses), the new statistic's maximum c / max(e, null, f) directly."""
    b = _basis()
    c = b["census"]
    assert (c["reads"], c["entries"], c["old_over_k"], c["new_over_k"]) == (62880, 3930, 42, 0)
    old_over, worst, at = 0, 0.0, None
    for t in b["tail"]:
        nmax = max(t["null"])
        for e, comp in zip(t["eager"], t["compiled"]):
            args = ({0: comp}, {0: e}, {0: t["floor"]}, K)
            old_over += bool(PN.judge(*args).bad)
            assert not PN.judge(*args, _Null({0: nmax})).bad, t["param"]
            r = PN._ratio(comp, e, nmax, t["floor"])
            if r > worst:
                worst, at = r, (t["state"], t["slice"], t["param"])
    assert old_over == c["old_over_k"] == 42
    assert worst == pytest.approx(c["new_max"][0], rel=1e-3) and list(at) == c["new_max"][1:]
    assert c["new_max"][0] == pytest.approx(9.179, abs=1e-3) and c["old_max"][0] == pytest.approx(27.89, abs=1e-2)


# ------------------------------------------------------------------------------------------- the real learner
@pytest.fixture(scope="module")
def learner():
    from agents.model import compile_control as cc
    from agents.model import trunk_precision as TP
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    model = LG.build_learner()
    LG.load_buffer_into(model)
    TP.set_trunk_precision(model.policy, "bf16")
    cc.control().install()
    cr.install(model, backend="eager")
    try:
        yield model
    finally:
        cr.uninstall(model)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


@pytest.mark.slow  # the K9 learner's build + 16 fp32 arms: ~30 s on CPU
def test_on_the_real_learner_the_null_is_the_censuss_statistic(learner):
    """`compute` on the canary's own relocated rows equals the replay's formula (`dithered` + the fp32 reference arm,
    `main.ops.canary_slice_read.dither_read`'s ``null``), number for number."""
    from main.ops.diag_bf16_canary import dithered
    torch.set_float32_matmul_precision("highest")
    learner.policy.set_training_mode(True)
    b = cr.r1_batch(learner, 16, 0)
    args, _fix = cr.relocate_clip_boundary(learner, cr._r1_args(learner, b))
    ref = cr._r1_reference(learner, args)
    sizes = [int(x) for x in ref["grad_sizes"].tolist()]
    idx = [i for i, g in enumerate(torch.split(ref["grad"], sizes)) if float(g.norm()) > 0][:40]
    raw = PN.compute(learner, args, ref, idx, seeds=2, cap_s=600.0)
    assert raw["seeds_done"] == 2 and "error" not in raw, raw
    base = torch.split(ref["grad"].double(), sizes)
    for s in range(2):
        with dithered([learner], s, PN.PARAM_NULL_REL):
            g = torch.split(cr._r1_reference(learner, args)["grad"].double(), sizes)
        assert raw["per_seed"][s] == [PN._dist(g[i], base[i]) for i in idx]
        cr.release_gate_memory(learner)


@pytest.mark.slow  # the K9 learner's build + a canary check with a real 16-seed null: ~40 s on CPU
def test_on_the_real_learner_a_DOUBLED_gradient_is_REFUSED_after_a_real_null(learner, monkeypatch):
    """The canary's check, compiled = eager bf16 (dynamo's eager backend) except a planted x2 on the largest-gradient
    parameter's compiled gradient (error ~1.0; its CPU null at B 16 is 0.028, so ~35 x against K 12). The global
    gradient's envelope is set out of the way so the per-parameter rule is the one under test; without the plant the
    check passes and computes no null."""
    monkeypatch.setattr(cr, "BF16_DRIFT_BARS", {"loss": 1.0, "grad_rel_l2": 5.0})
    monkeypatch.setattr(cr, "BF16_ENVELOPE_K", {"loss": 9.2, "grad_rel_l2": 1e6, "param": K})
    torch.set_float32_matmul_precision("highest")
    can = CC.CompileCanary(learner, batch_size=16, row_null=False)
    w = cr.GradWatch(())
    can._check(0, {}, w)
    assert w.param_null is None                               # exactly lazy: the healthy check needs no null
    real = cr._r1_arm

    def plant(model: Any, fn: Any, a: Any) -> Dict[str, torch.Tensor]:
        out = real(model, fn, a)
        if fn is model._compiled_micro_step:
            g = out["grad"].clone()
            sizes = [int(x) for x in out["grad_sizes"].tolist()]
            big = max(range(len(sizes)), key=lambda i: float(torch.split(g, sizes)[i].norm()))
            start = sum(sizes[:big])
            g[start:start + sizes[big]] *= 2.0
            out = {**out, "grad": g}
        return out
    monkeypatch.setattr(cr, "_r1_arm", plant)
    w = cr.GradWatch(())
    with pytest.raises(ct.CompileTrainerError, match=r"than 12 x max\(eager bf16's own error, its fp32 dither null\)"):
        can._check(0, {}, w)
    pn = w.param_null
    assert pn is not None and pn["source"] == "null" and pn["seeds_done"] == PN.PARAM_NULL_SEEDS
    (name, d), = pn["params"].items()
    assert d["compiled"] == pytest.approx(1.0, abs=0.05) and d["ratio"] > 2 * K, (name, d)
    assert all(np.isfinite(float(v)) for v in (d["null"], d["eager"]))
