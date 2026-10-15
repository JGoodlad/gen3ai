"""THE R1 GATE'S RESOLUTION under the bf16 PARITY RULE: the smallest planted backward miscompile it refuses, per
parameter, computed from the gate's own arms and then CONFIRMED by planting it (`gen3_r1_bf16_resolution_v1`).

WHY. Since the bf16 trunk became production (`10b3a6b6`), R1 is judged by `compile_regions._r1_verdict_bf16`: a
parameter's compiled gradient is refused only when its distance from the fp32 eager reference exceeds
K (`BF16_ENVELOPE_K["param"]`) x max(eager bf16's own distance, the fp32 bar). bf16's rounding moves a micro-batch
gradient by 1-18 % per parameter, so the rule's RESOLUTION is coarse: a 10 % error on
`pokemon_encoder.move_network.0.weight` is NOT refused (the 2026-10-15 CUDA run of
`compile_regions_fixed_mass_cuda_test`). That is a known, measured limit
(`designs/training/compile_flags.md` "The bf16 gate's resolution"), and this file pins WHAT it is rather than
pretending the fp32 rule's 10 % teeth still hold.

THE ARITHMETIC (exact, for the planted fault `plant`: compiled arm only, parameter p, a relative error m in the
alternating-sign direction s, so the compiled segment C_p becomes C_p + m |C_p| s / |s|). With a = C_p - R_p:

    |a + m |C_p| s^|^2 = T^2   =>   m* = ( -(a . s^) + sqrt( (a . s^)^2 - |a|^2 + T^2 ) ) / |C_p|

for the per-parameter check (T = K_param x max(e_p, floor_p) x |R_p|) and the global gradient check (|a|^2 -> the
whole |C - R|^2, T = K_grad x max(d(E, R), BF16_GRAD_FLOOR) x |R|); the gate refuses at the smallest m* over the
checks and over the RUNGS it runs (the live weights; on fresh weights also the seeded perturbation at the trained
bar; on trained weights with unmoved parameters also their perturbation). The plant does not touch the loss or
the eager arms, so the DRIFT and loss checks cannot fire. `confirm_resolution` then plants 0.5 m* (the gate must
PASS) and 2 m* (it must REFUSE): a factor-2 margin each side, so no input sits near the boundary (rule 8).

CPU here (the K9 golden learner at the production bf16 recipe, B = 16, R1 on dynamo's `eager` backend, so the
compiled arm IS eager bf16 and only the plant separates them): the mechanics. The CUDA counterparts that pin the
real graph's numbers are `compile_regions_fixed_mass_cuda_test` and `compile_regions_trained_cuda_test`, run by
`scripts/ops/verify_compile_gate_cuda.sh`."""
from __future__ import annotations

import contextlib
import json
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model import trunk_precision as TP
from agents.model.compile_gate_probe import per_param_grad_errors

#: One parameter per CLASS of R1's graph (each a `compile_trainer.grad_parameters` name the golden rows judge):
#: the encoder upstream of the bf16 rounds (the fp32 test's own target), a Linear INSIDE a bf16 trunk round, the
#: fp32 projection after the trunk, and the pointer head's move projection (the decision side).
REPRESENTATIVE: Tuple[str, ...] = (
    "features_extractor.pokemon_encoder.move_network.0.weight",
    "features_extractor.team_transformer.transformer_layers.0.linear1.weight",
    "features_extractor.projection.weight",
    "pointer_head.move_proj.weight",
)
#: where a CUDA run writes its resolution record (the verification script sets it; unset = no file)
OUT_ENV = "GEN3AI_GATE_RESOLUTION_OUT"


def plant(monkeypatch: Any, param: str, rel: float = 0.0, *, nan: bool = False) -> None:
    """Corrupt R1's COMPILED arm only: ``param``'s gradient segment by a relative ``rel`` in the alternating-sign
    direction (the independence test's `_plant`, for any parameter), or its first element to NaN."""
    real = cr._r1_arm

    def arm(model: Any, fn: Any, args: Any) -> Dict[str, torch.Tensor]:
        out = real(model, fn, args)
        if fn is not model._compiled_micro_step:
            return out
        names = [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]
        sizes = [int(x) for x in out["grad_sizes"].tolist()]
        i = names.index(param)
        start = sum(sizes[:i])
        g = out["grad"].clone()
        seg = g[start:start + sizes[i]]
        if nan:
            seg[0] = float("nan")
        elif rel:
            sign = torch.where(torch.arange(seg.numel(), device=seg.device) % 2 == 0, 1.0, -1.0).to(seg)
            g[start:start + sizes[i]] = seg + rel * float(seg.norm()) * sign / float(sign.norm())
        return {**out, "grad": g}
    monkeypatch.setattr(cr, "_r1_arm", arm)


def _sign_hat(n: int, like: torch.Tensor) -> torch.Tensor:
    s = torch.where(torch.arange(n, device=like.device) % 2 == 0, 1.0, -1.0).double()
    return s / s.norm()


def _root(a_dot_s: float, a_sq: float, t: float, c: float) -> float:
    """The larger root m of |a + m c s^|^2 = t^2 (``a_sq`` = |a|^2, ``c`` = |C_p|): the smallest refused plant."""
    if c <= 0.0:
        return math.inf
    disc = a_dot_s * a_dot_s - a_sq + t * t
    if disc < 0.0:                       # the honest arm already exceeds t: refused at m = 0 (the honest gate fails)
        return 0.0
    return max(0.0, (-a_dot_s + math.sqrt(disc)) / c)


class Rung:
    """One rung of R1's gate as `compile_regions.r1_rungs` runs it: a label, the regime of its bar, the per-index
    bar override (the live rung's unmoved parameters), and the weights context."""

    def __init__(self, label: str, regime: str, over: Dict[int, float], ctx: Any) -> None:
        self.label, self.regime, self.over, self.ctx = label, regime, over, ctx


def gate_rungs(model: Any, names: List[str], regime: str) -> List[Rung]:
    """The rungs `r1_rungs` runs for ``regime`` (mirrors it: live; fresh -> the seeded perturbation at the trained
    bar; trained with unmoved parameters -> those perturbed, every parameter at the trained bar)."""
    from agents.model.parity_probe import PERTURB_LADDER, perturbed_parameters, rung_seed
    unmoved = cr.unmoved_parameters(model) if regime != "fresh" else []
    idx = {n: i for i, n in enumerate(names)}
    rungs = [Rung("live", regime, {idx[n]: cr.R1_PARAM_BAR["fresh"] for n in unmoved}, contextlib.nullcontext())]
    scale, k = PERTURB_LADDER[0]
    if regime == "fresh":
        rungs.append(Rung(f"perturbed scale={scale:g} seed+{k}", "trained", {},
                          perturbed_parameters(model.policy, seed=rung_seed(k), scale=scale)))
    elif unmoved:
        rungs.append(Rung(f"{len(unmoved)} unmoved perturbed", "trained", {},
                          perturbed_parameters(model.policy, seed=rung_seed(k), scale=scale, only=unmoved)))
    return rungs


def _q(xs: List[float], p: float) -> float:
    ys = sorted(x for x in xs if math.isfinite(x))
    return ys[int(p * (len(ys) - 1))] if ys else math.nan


def bf16_rung_resolution(comp: Dict[str, torch.Tensor], eager: Dict[str, torch.Tensor],
                         ref: Dict[str, torch.Tensor], names: List[str], regime: str,
                         over: Dict[int, float]) -> Dict[str, Any]:
    """One rung's per-parameter RESOLUTION under the bf16 rule (module docstring), plus the same-precision
    statistics (compiled bf16 vs EAGER bf16) the (1b) decision reads. Every judged parameter is covered."""
    assert cr.BF16_ENVELOPE_K is not None
    k_p, k_g = float(cr.BF16_ENVELOPE_K["param"]), float(cr.BF16_ENVELOPE_K["grad_rel_l2"])
    sizes = [int(x) for x in ref["grad_sizes"].tolist()]
    C, E, R = (torch.split(x["grad"].double(), sizes) for x in (comp, eager, ref))
    c_all, r_all = comp["grad"].double(), ref["grad"].double()
    glob_a_sq = float((c_all - r_all).norm()) ** 2
    d_g_e = float((eager["grad"].double() - r_all).norm()) / float(r_all.norm())
    t_g = k_g * max(d_g_e, cr.BF16_GRAD_FLOOR) * float(r_all.norm())
    err_e = dict(per_param_grad_errors(eager["grad"], ref["grad"], sizes, floor_frac=ct._PARAM_GRAD_FLOOR))
    err_ce = dict(per_param_grad_errors(comp["grad"], eager["grad"], sizes, floor_frac=ct._PARAM_GRAD_FLOOR))
    per: Dict[str, Dict[str, Any]] = {}
    for i in range(len(sizes)):
        c = float(C[i].norm())
        if c == 0.0:
            continue                     # a zero segment: the plant (relative to |C_p|) cannot move it
        a = C[i] - R[i]
        sh = _sign_hat(a.numel(), a)
        a_dot_s, a_sq = float(a @ sh), float(a.norm()) ** 2
        floor_p = float(over.get(i, cr.R1_PARAM_BAR[regime]))
        e_p = err_e.get(i)               # None: under the per-parameter floor, so only the GLOBAL check sees it
        m_param = math.inf if e_p is None else _root(a_dot_s, a_sq, k_p * max(e_p, floor_p) * float(R[i].norm()), c)
        m_glob = _root(a_dot_s, glob_a_sq, t_g, c)
        per[names[i]] = {"resolution": min(m_param, m_glob), "by": "param" if m_param <= m_glob else "global",
                         "judged": e_p is not None, "floor": floor_p,
                         "d_eager_bf16_vs_fp32": math.nan if e_p is None else e_p,
                         "d_compiled_vs_fp32": math.sqrt(a_sq) / float(R[i].norm()) if e_p is not None else math.nan,
                         "d_compiled_vs_eager_bf16": err_ce.get(i, math.nan)}
    judged = [v for v in per.values() if v["judged"]]
    res = [v["resolution"] for v in judged]
    ce = [v["d_compiled_vs_eager_bf16"] for v in judged]
    er = [v["d_eager_bf16_vs_fp32"] for v in judged]
    ratio = [x / max(y, v["floor"]) for x, y, v in zip(ce, er, judged) if math.isfinite(x)]
    return {"n_judged": len(judged), "per_param": per,
            "resolution": {"median": _q(res, .5), "p90": _q(res, .9), "max": _q(res, 1.0), "min": _q(res, 0.0)},
            "same_precision": {"d_compiled_vs_eager_bf16": {"median": _q(ce, .5), "p90": _q(ce, .9),
                                                            "max": _q(ce, 1.0)},
                               "d_eager_bf16_vs_fp32": {"median": _q(er, .5), "p90": _q(er, .9), "max": _q(er, 1.0)},
                               "ratio_ce_over_max_er_floor_max": _q(ratio, 1.0)}}


def gate_resolution(model: Any, batch_size: int) -> Dict[str, Any]:
    """The bf16 gate's resolution on ``model``'s current weights: per rung (`bf16_rung_resolution`) and, per
    judged parameter, the gate's (the minimum over its rungs). Arms through the gate's own `_r1_pair` /
    `_r1_reference` on the gate's own rows; deterministic (GPU: bit-identical run to run, window D)."""
    assert TP.trunk_precision_of(model.policy) == "bf16"
    policy = model.policy
    names = [n for n, _ in ct.grad_parameters(model, policy.features_extractor)]
    regime = cr.weights_regime(model)
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    was = policy.training
    out: Dict[str, Any] = {"regime": regime, "batch_size": int(batch_size), "rungs": {}}
    try:
        with torch.random.fork_rng(devices=devices):
            policy.set_training_mode(True)
            args = cr._r1_args(model, cr.r1_batch(model, int(batch_size)))
            for rung in gate_rungs(model, names, regime):
                with rung.ctx:
                    comp, eager = cr._r1_pair(model, args)
                    ref = cr._r1_reference(model, args)
                out["rungs"][rung.label] = bf16_rung_resolution(comp, eager, ref, names, rung.regime, rung.over)
    finally:
        policy.set_training_mode(was)
        for p in policy.parameters():
            p.grad = None
    gate: Dict[str, float] = {}
    judged: set = set()
    for r in out["rungs"].values():
        for n, v in r["per_param"].items():
            gate[n] = min(gate.get(n, math.inf), v["resolution"])
            if v["judged"]:
                judged.add(n)
    out["gate_per_param"] = gate
    out["gate_judged"] = sorted(judged)
    vals = [gate[n] for n in judged]             # over the parameters the per-parameter rule judges in some rung
    out["gate"] = {"n_judged": len(vals), "median": _q(vals, .5), "p90": _q(vals, .9), "max": _q(vals, 1.0),
                   "min": _q(vals, 0.0)}
    return out


def confirm_resolution(model: Any, monkeypatch: Any, batch_size: int, params: Tuple[str, ...]) -> Dict[str, Any]:
    """`gate_resolution`, then for each of ``params`` (each judged in some rung) the gate PASSES a plant at 0.5 m* and
    REFUSES one at 2 m* (`CompileTrainerError`). Returns the resolution record with the confirmations."""
    rec = gate_resolution(model, batch_size)
    rec["confirmed"] = {}
    for p in params:
        assert p in rec["gate_judged"], f"PRECONDITION: {p} is judged by the gate on these rows (in some rung)"
        m = rec["gate_per_param"][p]
        assert 0.0 < m < math.inf, (p, m)
        with monkeypatch.context() as mp:
            plant(mp, p, 0.5 * m)
            cr.gate_regions(model, batch_size=batch_size, say=lambda _m: None)
        with monkeypatch.context() as mp:
            plant(mp, p, 2.0 * m)
            with pytest.raises(ct.CompileTrainerError):
                cr.gate_regions(model, batch_size=batch_size, say=lambda _m: None)
        rec["confirmed"][p] = {"resolution": m, "passes_at": 0.5 * m, "refused_at": 2.0 * m}
    return rec


def write_record(name: str, rec: Dict[str, Any]) -> Optional[str]:
    """Write ``rec`` to ``$GEN3AI_GATE_RESOLUTION_OUT/<name>.json`` when the variable is set (the CUDA
    verification script); None otherwise."""
    d = os.environ.get(OUT_ENV)
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.json")
    meta = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "bars": {"BF16_ENVELOPE_K": cr.BF16_ENVELOPE_K, "BF16_DRIFT_BARS": cr.BF16_DRIFT_BARS,
                     "R1_PARAM_BAR": cr.R1_PARAM_BAR, "BF16_GRAD_FLOOR": cr.BF16_GRAD_FLOOR}}
    with open(path + ".tmp", "w") as f:
        json.dump({**meta, **rec}, f, indent=1, default=float)
    os.replace(path + ".tmp", path)
    return path


# ------------------------------------------------------------------------------------------- CPU mechanics
BATCH = 16


@pytest.fixture(scope="module")
def fresh_bf16() -> Iterator[Any]:
    """The K9 golden learner at the PRODUCTION recipe (bf16 rounds), fresh weights, R1 on dynamo's `eager`
    backend (module-scoped: one trace for the file)."""
    from agents.model import parity_probe as PP
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PP, "PERTURB_SCALE", 0.0)              # a FRESH launch's weights
        m = LG.build_learner()
    LG.load_buffer_into(m, LG.BUFFER_PATH)
    m.batch_size = BATCH
    assert TP.trunk_precision_of(m.policy) == "bf16"      # PRECONDITION: the production recipe is the bf16 trunk
    cc.control().install()
    cr.install(m, backend="eager")
    try:
        yield m
    finally:
        cr.uninstall(m)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


@pytest.mark.integration
def test_the_honest_bf16_gate_PASSES_under_the_bf16_rule(fresh_bf16):
    """PRECONDITION for every plant below (and the file's one dynamo trace)."""
    rules = cr.gate_regions(fresh_bf16, batch_size=BATCH, say=lambda _m: None)
    assert any("bf16 rule:" in r for r in rules), rules


@pytest.mark.integration
def test_the_bf16_gates_resolution_is_what_its_arithmetic_says_planted_both_sides(fresh_bf16, monkeypatch):
    """The computed resolution of the representative parameters (the encoder the fp32 test plants on, a Linear
    inside a bf16 round, the projection, the pointer head) is CONFIRMED: 0.5 m* passes, 2 m* refuses (a one-off at
    0.97 / 1.03 m* also split pass / refuse on all four: the arithmetic is exact). On this eager-backend learner the compiled arm
    IS eager bf16 (only the plant separates them); the arithmetic is the CUDA tests', unchanged."""
    rec = confirm_resolution(fresh_bf16, monkeypatch, BATCH, REPRESENTATIVE)
    assert rec["regime"] == "fresh" and len(rec["rungs"]) == 2, rec["rungs"].keys()
    for r in rec["rungs"].values():
        # the eager backend: compiled == eager bf16 bit for bit, so the same-precision distance is 0
        assert r["same_precision"]["d_compiled_vs_eager_bf16"]["max"] == 0.0


@pytest.mark.integration
def test_a_10pc_encoder_miscompile_is_BELOW_the_bf16_gates_resolution_on_fresh_weights(fresh_bf16, monkeypatch):
    """The known limit, pinned (not a target): under the bf16 rule a 10 % backward error on
    `move_network.0.weight` PASSES the gate — the fp32 rule's teeth (`compile_regions_independence_test`) do not
    hold under bf16. If this starts failing the gate got SHARPER: update the resolution section of
    `designs/training/compile_flags.md` and drop this pin."""
    rec = gate_resolution(fresh_bf16, BATCH)
    assert rec["gate_per_param"][REPRESENTATIVE[0]] > 2 * 0.1, rec["gate_per_param"][REPRESENTATIVE[0]]
    with monkeypatch.context() as mp:
        plant(mp, REPRESENTATIVE[0], 0.1)
        cr.gate_regions(fresh_bf16, batch_size=BATCH, say=lambda _m: None)


@pytest.mark.integration
def test_a_planted_NaN_is_still_NAMED_under_the_bf16_rule(fresh_bf16, monkeypatch):
    plant(monkeypatch, REPRESENTATIVE[0], nan=True)
    with pytest.raises(ct.NonFiniteGateArmError, match=r"COMPILED arm's gradient is NON-FINITE.*move_network"):
        cr.gate_regions(fresh_bf16, batch_size=BATCH, say=lambda _m: None)


def test_the_root_is_the_LARGER_one_and_refuses_beyond_it():
    """`_root` against a brute-force scan of |a + m c s^| (deterministic: the scan points are 1 % away)."""
    g = torch.Generator().manual_seed(7)
    a = torch.randn(64, generator=g, dtype=torch.float64) * 0.03
    sh = _sign_hat(64, a)
    c, t = 1.7, 0.4
    m = _root(float(a @ sh), float(a.norm()) ** 2, t, c)
    assert float((a + 0.99 * m * c * sh).norm()) < t < float((a + 1.01 * m * c * sh).norm())
    assert _root(0.0, 1.0, 0.5, 1.0) == 0.0          # already over the bar: refused at m = 0
    assert _root(0.0, 0.0, 1.0, 0.0) == math.inf     # a zero segment cannot be planted


_WINDOW_D = os.path.join(os.path.dirname(__file__), "..", "..", "..", "designs", "research_state", "measurements",
                         "perf_phase_2026-10-10", "results", "window_D", "E_bf16_envelope.json")


def same_precision_K(reads: Dict[str, Any], *, bar_k: float = 4.0) -> Tuple[float, float]:
    """(the healthy max over the reads of d(compiled bf16, EAGER bf16)_p / max(d(eager bf16, fp32)_p, floor_p) over
    the judged parameters, the median of that ratio) — what a SAME-PRECISION per-parameter rule (compiled bf16 vs
    eager bf16, normalised by the parameter's own bf16 error exactly as the envelope is) would be derived from. Its K
    would be ``bar_k`` x the max, by the bf16 rule's own derivation (`derive_bf16_bars.py`)."""
    ratios: List[float] = []
    for state, s in reads["states"].items():
        floor_p = cr.R1_PARAM_BAR["fresh" if state == "fresh" else "trained"]
        for sl in ("slice0", "slice1"):
            if sl not in s:
                continue
            r = s[sl]
            ce, er = r["pairs"]["Cbf_vs_Ebf"]["per_param_rel"], r["pairs"]["Ebf_vs_E32"]["per_param_rel"]
            ratios += [ce[i] / max(er[i], floor_p) for i, x in enumerate(r["grad_norm_ratio_E32"])
                       if x > ct._PARAM_GRAD_FLOOR]
    ratios.sort()
    return ratios[-1], ratios[len(ratios) // 2]


@pytest.mark.skipif(not os.path.exists(_WINDOW_D), reason="the committed window-D GPU reads are absent")
def test_a_SAME_PRECISION_rule_would_not_be_sharper_on_the_committed_GPU_reads():
    """THE (1b) DECISION's evidence (`designs/training/compile_flags.md` "The bf16 gate's resolution"): judging
    compiled bf16 against EAGER bf16 (a compiler defect, at the same precision) instead of against the fp32
    reference does NOT tighten the per-parameter teeth. On window D's 20 GPU reads (B = 2,048; `Cbf_vs_Ebf` and
    `Ebf_vs_E32` per parameter) Inductor's fused kernels keep fp32 intermediates where eager rounds each op's output
    to bf16, so compiled-vs-eager bf16 differs by bf16 ROUNDING too: median ratio 0.55 of eager bf16's own fp32
    distance, healthy max 2.96 — K = 4 x 2.96 = 11.8, the envelope's 12. "Clearly better" was taken as >= 2x sharper
    teeth (K <= 6); the read is 11.8, so the rule was NOT built."""
    reads = json.load(open(_WINDOW_D))
    k_max, k_med = same_precision_K(reads)
    assert cr.BF16_ENVELOPE_K is not None
    k_pair = 4.0 * k_max
    assert k_pair > 0.5 * cr.BF16_ENVELOPE_K["param"], (k_pair, k_med)      # not >= 2x sharper (11.8 vs 6)
    assert 0.3 < k_med < 1.0, k_med                                        # same order as bf16's own error
