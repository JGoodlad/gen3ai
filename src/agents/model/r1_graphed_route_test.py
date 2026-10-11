"""T25 lever C — R1's GRAPHED route (`--learner-cudagraphs on`, `gen3_r1_graphed_route_v1`), on CPU.

THE PROBLEM (window B, F-PB-2). R1 compiled with Inductor's cudagraph trees passes the startup gate bit-equal and
then dies on the first update: the grad-balance probe's `autograd.grad(retain_graph=True)` traverses R1's graph a
SECOND time, and cudagraph trees record ONE backward per forward. THE FIX: two declared R1 callables of one
signature — the graphed one for every micro-step no retain_graph probe re-enters, the un-graphed compiled one for
the probe micro-steps — routed per micro-step from the diagnostics plan (`diagnostics_cadence.micro_retains_graph`
-> `compile_regions.r1_route_for`), deterministically.

CPU HAS NO CUDA GRAPHS, so the graphed route compiles here with a STAND-IN backend that keeps cudagraph trees' one
load-bearing constraint: a second backward traversal of one forward's graph RAISES (`_one_backward_backend`). Its
numbers are dynamo's `eager` backend's, so a flag-on arm and a flag-off arm train bit-identically. Every test
names the code it pins; each fails on a revert of it.
"""
from __future__ import annotations

import argparse
import contextlib
import io
from typing import Any, Dict, List

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct

BATCH = 16                       # the K9 golden learner's micro-batch (64 rows, accumulation 3)


# ------------------------------------------------------------------------------- the stand-in
class _OneBackward(torch.autograd.Function):
    """Identity on R1's grad-carrying outputs; its backward runs ONCE per forward — a second traversal of the
    same forward's graph (a retain_graph probe, then the real backward) raises, as cudagraph trees do."""

    @staticmethod
    def forward(ctx: Any, token: Dict[str, int], *ts: torch.Tensor) -> Any:   # type: ignore[override]
        ctx.token = token
        return tuple(t.view_as(t) for t in ts)

    @staticmethod
    def backward(ctx: Any, *grads: Any) -> Any:                               # type: ignore[override]
        ctx.token["n"] += 1
        if ctx.token["n"] > 1:
            raise RuntimeError("STAND-IN cudagraph trees: a second backward through one forward's graph "
                               "(F-PB-2 — a retain_graph probe reached the GRAPHED route)")
        return (None, *grads)


def _one_backward_backend(gm: torch.fx.GraphModule, example_inputs: Any) -> Any:
    def run(*args: Any) -> Any:
        outs = gm(*args)
        seq = list(outs) if isinstance(outs, (tuple, list)) else [outs]
        idx = [i for i, o in enumerate(seq) if isinstance(o, torch.Tensor) and o.requires_grad]
        if idx:
            gated = _OneBackward.apply({"n": 0}, *[seq[i] for i in idx])
            for i, g in zip(idx, gated):
                seq[i] = g
        return type(outs)(seq) if isinstance(outs, (tuple, list)) else seq[0]
    return run


# ------------------------------------------------------------------------------------ helpers
def _learner(graphed: bool) -> Any:
    from agents.training import learner_golden as LG
    from agents.training.instrumented_ppo.noise_scale_terms import NOISE_TERM_GROUPS
    model = LG.build_learner()
    LG.load_buffer_into(model)
    model._learner_cudagraphs = bool(graphed)
    # prime the noise-scale EMAs past warm-up so the per-term tags are emitted (as `diagnostics_cadence_test`)
    model._noise_ema_s, model._noise_ema_g2, model._noise_ema_n = 50.0, 2.0, 500
    model._noise_ema_terms = {g: [50.0, 2.0, 500] for g in NOISE_TERM_GROUPS}
    return model


def _install(model: Any) -> List[str]:
    cc.control().install()
    return cr.install(model, backend="eager", graphed_backend=_one_backward_backend)


def _record_routes(model: Any, seq: List[str]) -> None:
    def rec(fn: Any, tag: str) -> Any:
        def w(*a: Any, **k: Any) -> Any:
            seq.append(tag)
            return fn(*a, **k)
        return w
    model._compiled_micro_step = rec(model._compiled_micro_step, "c")
    if getattr(model, "_compiled_micro_step_graphed", None) is not None:
        model._compiled_micro_step_graphed = rec(model._compiled_micro_step_graphed, "g")


def _update(model: Any, *, diagnostic: bool) -> Dict[str, Any]:
    """One `train()` on the golden rows: the process's diagnostic update, or a PLAIN one (rollout index 1)."""
    from agents.training import learner_golden as LG
    LG.load_buffer_into(model)
    model.logger.name_to_value.clear()
    if not diagnostic:
        model._diagnostics_ran_in_process = True
        model.num_timesteps = LG.N_STEPS * LG.N_ENVS           # rollout index 1: not on the cadence (10)
    model.train()
    logged = {k: v for k, v in model.logger.name_to_value.items() if not k.endswith(("_ms", "_s"))}
    return {"logged": logged,
            "params": {k: v.detach().clone() for k, v in model.policy.state_dict().items()}}


@pytest.fixture(scope="module")
def arms():
    """Two arms from the same deterministic build: OFF (one compiled route) and ON (both routes, the full
    startup lifecycle: install, gate, prewarm, inventory, lock), each a diagnostic update then a plain one."""
    out: Dict[str, Any] = {}
    made: List[Any] = []
    try:
        for name, graphed in (("off", False), ("on", True)):
            cc._reset_control_for_tests()
            torch._dynamo.reset()
            model = _learner(graphed)
            made.append(model)
            ctl = cc.control()
            names = _install(model)
            rules = cr.gate_regions(model, batch_size=BATCH, say=lambda _m: None)
            prewarm = ctl.prewarm(cr.prewarm_calls(model, batch_size=BATCH))
            inv = cr.assert_inventory(model)
            ctl.lock("test: the end of startup")
            seqs: List[List[str]] = []
            ups = []
            for diagnostic in (True, False):
                seq: List[str] = []
                saved = (model._compiled_micro_step, getattr(model, "_compiled_micro_step_graphed", None))
                _record_routes(model, seq)
                with ctl.guard("update end"):
                    ups.append(_update(model, diagnostic=diagnostic))
                model._compiled_micro_step = saved[0]
                if saved[1] is not None:
                    model._compiled_micro_step_graphed = saved[1]
                seqs.append(seq)
            ctl.check("end")
            out[name] = {"model": model, "names": names, "rules": rules, "prewarm": prewarm, "inventory": inv,
                         "routes": seqs, "updates": ups,
                         "after_lock": (ctl.compiles_after_lock, ctl.rejected_after_lock)}
            ctl.release("test arm done")
            cc._reset_control_for_tests()        # the process-global compile config back, before the tests
        yield out
    finally:
        for m in made:
            cr.uninstall(m)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


# ------------------------------------------------------------------- the route rule (pure)
def test_the_route_rule_sends_every_retain_graph_micro_step_to_the_compiled_route():
    from agents.training.instrumented_ppo.diagnostics_cadence import micro_retains_graph
    from agents.training.instrumented_ppo.noise_scale_terms import NULL_TAGGER, PerTermNoiseSampler
    assert cr.r1_route_for(retained_graph=True) == "compiled"
    assert cr.r1_route_for(retained_graph=False) == "graphed"
    sampler = PerTermNoiseSampler([torch.nn.Parameter(torch.zeros(2))])
    assert micro_retains_graph(NULL_TAGGER, grad_balance_pending=False) is False
    assert micro_retains_graph(NULL_TAGGER, grad_balance_pending=True) is True
    assert micro_retains_graph(sampler, grad_balance_pending=False) is True
    sampler.failed = True                        # a failed sampler takes no gradient: graphed is safe
    assert micro_retains_graph(sampler, grad_balance_pending=False) is False


# --------------------------------------------------------------- the declaration and install
def test_the_graphed_signature_is_declared_only_when_the_flag_is_on():
    off, on = argparse.Namespace(_learner_cudagraphs=False), argparse.Namespace(_learner_cudagraphs=True)
    assert cr.declared_signature_count() == cr.declared_signature_count(off) == 1
    assert cr.declared_signature_count(on) == 2
    assert [s.route for s in cr.declared_signatures(on)] == ["compiled", "graphed"]
    assert [s.route for s in cr.declared_signatures(off)] == ["compiled"]


@pytest.fixture
def bare():
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    made: List[Any] = []
    try:
        yield made
    finally:
        for m in made:
            cr.uninstall(m)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def _spy_compile(monkeypatch: Any) -> List[Dict[str, Any]]:
    seen: List[Dict[str, Any]] = []
    real = torch.compile

    def spy(fn: Any, **kw: Any) -> Any:
        seen.append(dict(kw))
        return real(fn, fullgraph=True, dynamic=False, backend="eager")
    monkeypatch.setattr(torch, "compile", spy)
    return seen


@pytest.mark.parametrize("graphed,preset,expect", [
    (False, "default", [{"fullgraph": True, "dynamic": False}]),
    (True, "default", [{"fullgraph": True, "dynamic": False},
                       {"fullgraph": True, "dynamic": False, "options": {"triton.cudagraphs": True}}]),
    (True, "coordesc", [{"fullgraph": True, "dynamic": False, "options": {"coordinate_descent_tuning": True}},
                        {"fullgraph": True, "dynamic": False,
                         "options": {"coordinate_descent_tuning": True, "triton.cudagraphs": True}}]),
])
def test_install_compiles_the_production_R1_and_the_graphed_twin_only_when_on(bare, monkeypatch, graphed,
                                                                              preset, expect):
    """Flag OFF: ONE `torch.compile` with exactly the production kwargs (no options) — byte-identical. ON: the same
    compile, then a second one with the same kwargs + `triton.cudagraphs` (+ the preset's options on both)."""
    from agents.training import learner_golden as LG
    seen = _spy_compile(monkeypatch)
    model = LG.build_learner()
    bare.append(model)
    model._learner_cudagraphs = graphed
    model._r1_inductor_preset = preset
    cc.control().install()
    names = cr.install(model)
    assert seen == expect
    assert (getattr(model, "_compiled_micro_step_graphed", None) is not None) is graphed
    assert names == (["R1_learner_micro_step", "R1_learner_micro_step[graphed]"] if graphed
                     else ["R1_learner_micro_step"])


def test_the_flag_refuses_a_preset_that_already_graphs_R1(bare):
    from agents.training import learner_golden as LG
    model = LG.build_learner()
    bare.append(model)
    model._learner_cudagraphs, model._r1_inductor_preset = True, "cudagraphs"
    cc.control().install()
    with pytest.raises(ct.CompileTrainerError, match="drop `cudagraphs` from the preset"):
        cr.install(model, backend="eager")


# -------------------------------------------------------------- the lifecycle, flag ON (CPU)
def test_flag_on_startup_gates_prewarms_and_inventories_BOTH_routes(arms):
    on, off = arms["on"], arms["off"]
    assert on["names"] == ["R1_learner_micro_step", "R1_learner_micro_step[graphed]"]
    assert any(r.startswith("R1[graphed] ") for r in on["rules"]), on["rules"]
    assert any("ACCUMULATED" in r for r in on["rules"]), on["rules"]
    assert "R1[graphed]" in on["prewarm"] and "R1[graphed]" not in off["prewarm"]
    assert "R1 2 graphs" in on["inventory"]
    assert off["inventory"] == "[CompileRegions] inventory == declaration: R1 1 graph (fullgraph=True)"
    assert not any("[graphed]" in r or "ACCUMULATED" in r for r in off["rules"])
    assert on["after_lock"] == (0, 0) and off["after_lock"] == (0, 0)


def test_the_probe_micro_steps_take_the_compiled_route_and_every_other_one_the_graphed(arms):
    """The K9 golden learner: 64 rows, micro 16 x accumulation 3, 2 epochs = 8 micro-steps. On the process's
    first (diagnostic) update the per-term noise sampler collects epoch 0's first group (3 micro-steps) and the
    grad-balance probe takes one of them: COMPILED; the rest GRAPHED. A plain update: all GRAPHED. Flag off: one
    route. The stand-in RAISES on a probe that reaches the graphed route, so this ran each probe un-graphed."""
    diag, plain = arms["on"]["routes"]
    assert diag == ["c", "c", "c", "g", "g", "g", "g", "g"], diag
    assert plain == ["g"] * 8, plain
    assert arms["off"]["routes"] == [["c"] * 8, ["c"] * 8]


def test_the_probes_read_and_learning_are_IDENTICAL_whichever_route_the_plain_micro_steps_take(arms):
    on, off = arms["on"]["updates"], arms["off"]["updates"]
    logged = on[0]["logged"]
    assert any(k.startswith("grad/") for k in logged), "the diagnostic update ran no grad-balance probe"
    assert "train/noise_scale_ratio_policy" in logged, "the per-term noise sampler emitted nothing"
    for a, b in zip(off, on):
        assert a["logged"].keys() == b["logged"].keys()
        bad = [k for k in a["logged"] if not (a["logged"][k] == b["logged"][k]
                                              or (a["logged"][k] != a["logged"][k] and b["logged"][k] != b["logged"][k]))]
        assert not bad, bad
        for k in a["params"]:
            assert torch.equal(a["params"][k], b["params"][k]), k


def test_the_canary_judges_every_declared_route(arms):
    from agents.model.compile_canary import CompileCanary
    for name, want in (("on", ("R1[graphed] ", "R1[compiled] ")), ("off", ("R1 ",))):
        model = arms[name]["model"]
        out: Dict[str, float] = {}
        rules = CompileCanary(model, batch_size=BATCH, every=1)._regions(model, out, 0)
        assert [r[:len(p)] for r, p in zip(rules, want)] == list(want), rules
        assert "compile/canary_grad_cosine" in out


# ----------------------------------------------------------------- teeth: the stand-in and the rung
def test_the_stand_in_has_teeth_a_probe_on_the_graphed_route_raises(bare, monkeypatch):
    """Revert proof for the routing test above: route EVERY micro-step graphed and the diagnostic update dies in
    the stand-in, exactly as F-PB-2 died on the GPU."""
    model = _learner(True)
    bare.append(model)
    _install(model)
    monkeypatch.setattr(cr, "r1_route_for", lambda *, retained_graph: "graphed")
    with pytest.raises(RuntimeError, match="second backward through one forward"):
        _update(model, diagnostic=True)


def test_the_accumulation_rung_refuses_a_graphed_route_that_corrupts_an_accumulated_grad(bare):
    """A graphed route whose SECOND accumulated backward lands wrong (the `.grad`-aliases-the-pool hazard) passes
    every single-step rung and is refused by the accumulation rung."""
    model = _learner(True)
    bare.append(model)
    _install(model)
    names = [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]
    regime = cr.weights_regime(model)
    assert "ACCUMULATED" in cr._r1_accum_rule(model, names, regime, BATCH)      # healthy: passes
    real = model._compiled_micro_step_graphed
    params = [p for _, p in ct.grad_parameters(model, model.policy.features_extractor)]

    def corrupting(*a: Any, **k: Any) -> Any:
        out = real(*a, **k)
        if any(p.grad is not None for p in params):    # accumulating: the second backward doubles
            return out._replace(loss=out.loss * 2.0)
        return out
    model._compiled_micro_step_graphed = corrupting
    model.policy.set_training_mode(True)
    with pytest.raises(ct.CompileTrainerError):
        cr._r1_accum_rule(model, names, regime, BATCH)


def test_a_cudagraph_skip_after_install_is_FATAL_never_a_silent_un_graphed_route(bare):
    from torch._dynamo.utils import counters
    model = _learner(True)
    bare.append(model)
    ctl = cc.control()
    _install(model)
    ctl.graphed_check = lambda: cr.graphed_route_skipped(model)
    assert cr.graphed_route_skipped(model) is None
    ctl.check("healthy")
    counters["inductor"]["cudagraph_skips"] += 1
    try:
        assert "SKIPPED cudagraphs" in (cr.graphed_route_skipped(model) or "")
        with pytest.raises(cc.CompileSentinelError, match="SKIPPED cudagraphs"):
            ctl.check("after a skip")
        with pytest.raises(ct.CompileTrainerError, match="SKIPPED cudagraphs"):
            cr.assert_inventory(model)
    finally:
        counters["inductor"]["cudagraph_skips"] -= 1


# --------------------------------------------------------------------- the flag surface
def _resolved(argv: List[str]) -> Any:
    from main.train.config import resolve_config
    from main.train.parser import build_parser
    p = build_parser()
    a = p.parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(a, p)
    return a


def test_the_flag_parses_defaults_off_and_is_a_runtime_knob_never_recorded():
    from agents.model.model_version.fields import ModelVersionFields
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from main.train.combination_checks import fresh_only_flags
    from main.train.compile_flags import learner_cudagraphs_on
    from main.train.parser import build_parser
    assert build_parser().parse_args([]).learner_cudagraphs == "off"
    assert learner_cudagraphs_on(build_parser().parse_args(["--learner-cudagraphs", "on"]))
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--learner-cudagraphs", "yes"])
    # not a ModelVersion field (nothing to inherit from model_config.json) …
    assert "learner_cudagraphs" not in getattr(ModelVersionFields, "__dataclass_fields__", {})
    # … never pickled into a checkpoint …
    excl = InstrumentedMaskablePPO.__new__(InstrumentedMaskablePPO)._excluded_save_params()
    assert {"_learner_cudagraphs", "_compiled_micro_step_graphed"} <= set(excl)
    # … and NOT fresh-only, so the launcher's restarts forward it with the argv
    assert "--learner-cudagraphs" not in fresh_only_flags()
    # the shared compile cache keys it (it shapes what is compiled)
    from agents.model.compile_cache_shared import COMPILE_ARG_KEYS
    assert "learner_cudagraphs" in COMPILE_ARG_KEYS


@pytest.mark.parametrize("extra", [["--device", "cpu"], ["--no-compile-trainer"], ["--debug"]])
def test_the_flag_is_refused_without_a_compiled_cuda_learner(extra):
    from main.train.combination_checks import failing_checks
    from main.train.parser import build_parser
    a = build_parser().parse_args(["--learner-cudagraphs", "on", *extra])
    assert "learner_cudagraphs_needs_the_compiled_cuda_learner" in [c.name for c in failing_checks(a)]
    ok = build_parser().parse_args(["--learner-cudagraphs", "on", "--device", "cuda", "--compile-trainer"])
    assert "learner_cudagraphs_needs_the_compiled_cuda_learner" not in [c.name for c in failing_checks(ok)]
    off = build_parser().parse_args(["--device", "cpu"])
    assert "learner_cudagraphs_needs_the_compiled_cuda_learner" not in [c.name for c in failing_checks(off)]


def test_an_auto_compile_that_resolved_off_refuses_at_startup():
    """The case the parse-time check cannot see: an AUTO --compile-trainer its own default turned OFF."""
    from main.exit_codes import TrainExitCode
    from main.train.compile_flags import learner_cudagraphs_refusal
    from main.train.lifecycle import _maybe_compile_trainer
    args = argparse.Namespace(learner_cudagraphs="on", compile_trainer=False)
    assert "resolved OFF" in (learner_cudagraphs_refusal(args) or "")
    assert learner_cudagraphs_refusal(argparse.Namespace(learner_cudagraphs="off", compile_trainer=False)) is None
    with pytest.raises(SystemExit) as exc:
        _maybe_compile_trainer(argparse.Namespace(policy=None), args)
    assert exc.value.code == TrainExitCode.FATAL_CONFIG


def test_the_trainer_passes_the_flag_to_the_sentinel(monkeypatch):
    """`_arm_compile_sentinel` hands the RESOLVED flag to `arm_compile_sentinel` (the real wiring the perf bench
    uses too) — flag off passes False."""
    from agents.model import compile_trainer as CT
    from main.train.lifecycle import _arm_compile_sentinel
    seen: List[bool] = []
    monkeypatch.setattr(CT, "arm_compile_sentinel",
                        lambda model, *, batch_size, emit=None, learner_cudagraphs=False: seen.append(learner_cudagraphs))
    m = argparse.Namespace(batch_size=16)
    _arm_compile_sentinel(m, argparse.Namespace(compile_trainer=True, learner_cudagraphs="on"))
    _arm_compile_sentinel(m, argparse.Namespace(compile_trainer=True, learner_cudagraphs="off"))
    _arm_compile_sentinel(m, argparse.Namespace(compile_trainer=True))
    assert seen == [True, False, False]


def test_checkargs_accepts_the_flag_and_reports_its_refusal():
    from main.checkargs import check
    ok = check(["--arch", "production", "--learner-cudagraphs", "on", "--device", "cuda", "--compile-trainer"])
    assert "--learner-cudagraphs" in ok["accepted"] and not ok["unknown"]
    assert not [c.name for c, _ in ok["combinations"] if "cudagraphs" in c.name]
    assert ok["ns"].learner_cudagraphs == "on"
    bad = check(["--arch", "production", "--learner-cudagraphs", "on", "--device", "cpu"])
    assert "learner_cudagraphs_needs_the_compiled_cuda_learner" in [c.name for c, _ in bad["combinations"]]


def test_a_resolved_launch_defaults_off_and_keeps_a_typed_value():
    assert _resolved(["--steps", "100"]).learner_cudagraphs == "off"
    assert _resolved(["--steps", "100", "--learner-cudagraphs", "on", "--device", "cuda",
                      "--compile-trainer"]).learner_cudagraphs == "on"
