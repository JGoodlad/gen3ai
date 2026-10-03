"""K8 — the DECLARED REGIONS, as the routine compile INVENTORY test (`gen3_declared_regions_v1`).

The production-surface learner (the K9 golden's), CPU, torch 2.8, dynamo's `eager` backend (graphs,
breaks, guards and recompiles are dynamo's — independent of the backend; the CUDA Inductor compile is
the GPU tier's and the startup gate's): install R1 `fullgraph=True`, gate it against eager,
prewarm the declared signature, LOCK, then run two REAL updates (every fold step, the eager tail, the
probes) and the learner's EAGER rollout forward (the compiled rollout region R0 was deleted, P10-E).
The inventory must EQUAL the declaration table:

  * graphs == declared regions x signatures (`compile_regions.declared_signature_count`);
  * 0 undeclared breaks — guaranteed by `fullgraph=True` (a break is a compile error), and the teeth
    test below proves a break inside R1 is the typed STARTUP FATAL naming dynamo's reason;
  * 0 compiles / rejections after the lock, one cache entry per region code object.

The regions are the ONLY compiled learner surface: HEAD runs torch >= 2.8 (the 2.5.1 extractor-only
compile was deleted, K1 2026-10-02)."""
from __future__ import annotations

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct

N_ENVS, BATCH = 4, 16


@pytest.fixture
def learner():
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    model = LG.build_learner()
    LG.load_buffer_into(model)
    try:
        yield model
    finally:
        cr.uninstall(model)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def _graphs() -> int:
    from torch._dynamo.utils import counters
    return int(counters["stats"]["unique_graphs"])


def test_the_declaration_table_is_well_formed():
    names = [r.name for r in cr.REGIONS]
    assert len(names) == len(set(names))
    for r in cr.REGIONS:
        assert r.why, r.name
        assert bool(r.signatures) == r.compiled, r.name        # compiled <=> it declares signatures
    assert {r.name for r in cr.compiled_regions()} == {"R1_learner_micro_step"}
    assert cr.declared_signature_count() == 1


def test_the_compiled_inventory_EQUALS_the_declaration_table(learner):
    model = learner
    ctl = cc.control()
    ctl.install()
    g0 = _graphs()
    assert cr.install(model, backend="eager") == ["R1_learner_micro_step"]
    rules = cr.gate_regions(model, batch_size=BATCH, say=lambda _m: None)
    assert rules and all(r.startswith("R1 ") for r in rules), rules      # the ONLY gated region
    ctl.prewarm(cr.prewarm_calls(model, batch_size=BATCH))
    ctl.lock("test: the end of startup")
    assert _graphs() - g0 == cr.declared_signature_count()
    ent = cc.cache_entries_by_code()
    assert {k.split(" ")[0] for k in ent} == {"micro_step"}, ent
    assert all(v == 1 for v in ent.values()), ent
    from agents.training import learner_golden as LG
    for _ in range(2):                                      # two REAL updates: the fold, the tail, the probes
        LG.load_buffer_into(model)
        with ctl.guard("update end"):
            model.train()
    # the learner's rollout forward (`predict`-class callers; the Rust collector's is T2's own) is EAGER
    # at every batch, takes NO region route, and compiles nothing — there is no compiled rollout region
    from agents.model import region_calls as RC
    obs = ct._prewarm_obs(model, N_ENVS)
    _, mask = ct._parity_obs(int(obs["observation"].shape[-1]), N_ENVS, torch.device("cpu"))
    model.policy.set_training_mode(False)
    RC.take()
    with ctl.guard("rollout"), torch.no_grad():
        import numpy as np
        for m in (mask, np.asarray(mask).astype(np.int8), torch.as_tensor(np.asarray(mask)),
                  torch.as_tensor(np.asarray(mask)).float()):
            model.policy(obs, action_masks=m)
        model.policy(ct._prewarm_obs(model, 1), action_masks=mask[:1])   # batch 1
    assert RC.peek() == {}, f"a rollout forward took a region route: {RC.peek()}"
    ctl.check("end")
    assert ctl.compiles_after_lock == 0 and ctl.rejected_after_lock == 0, ctl.after_lock_frames
    assert _graphs() - g0 == cr.declared_signature_count()


def test_a_graph_break_INSIDE_a_region_is_the_typed_startup_FATAL(learner, monkeypatch):
    """`fullgraph=True` is the mechanism: a host read inside R1 must refuse at startup, naming
    dynamo's reason — never a silent split into two graphs. Fails if the region is compiled without
    `fullgraph=True` (the break would be absorbed) or the error escapes untyped (a restartable crash)."""
    from agents.training.instrumented_ppo import micro_step as ms
    real = ms.win_prob_terms

    def with_a_host_read(*a, **k):
        out = real(*a, **k)
        if float(out[0]) > 1e30:                           # a data-dependent Python branch
            return (out[0] * 0.0,) + tuple(out[1:])
        return out
    monkeypatch.setattr(ms, "win_prob_terms", with_a_host_read)
    cc.control().install()
    cr.install(learner, backend="eager")
    with pytest.raises(ct.CompileTrainerError, match="DECLARED REGION does not compile as one graph"):
        cr.gate_regions(learner, batch_size=BATCH, say=lambda _m: None)


def test_an_UNDECLARED_signature_after_the_lock_names_the_failing_guard(learner):
    ctl = cc.control()
    ctl.install()
    cr.install(learner, backend="eager")
    ctl.prewarm(cr.prewarm_calls(learner, batch_size=BATCH))
    ctl.lock("test")
    args = cr._r1_args(learner, cr.r1_batch(learner, BATCH * 2))     # a batch the table never declared
    with pytest.raises(cc.CompileSentinelError, match="UNDECLARED SIGNATURE") as ei:
        with ctl.guard("update end"):
            learner._compiled_micro_step._gen3_compiled(*args)       # the compiled region itself
    assert "size mismatch" in str(ei.value) or "expected 16" in str(ei.value), str(ei.value)[-800:]


def test_a_RAGGED_micro_batch_is_REFUSED_by_R1s_dispatcher_and_compiles_nothing(learner):
    """R1 has ONE declared row count (`gen3_r1_no_ragged_v1`): the collector refuses an update that does
    not divide by the micro-batch, so the former declared EAGER route for the last micro-batch of an
    epoch (unreachable in production) is deleted. A call with another row count is a mis-wired caller:
    the dispatcher raises a typed error naming the cause — it does not run the function eager (a silent
    slowdown), does not hand the compiled region an undeclared shape, and compiles nothing, locked or
    not. Fails on revert: the old route returned `micro_step`'s eager loss instead of raising."""
    from agents.model import region_calls as RC
    ctl = cc.control()
    ctl.install()
    cr.install(learner, backend="eager")
    ctl.prewarm(cr.prewarm_calls(learner, batch_size=BATCH))
    ctl.lock("test")
    RC.take()
    args = cr._r1_args(learner, cr.r1_batch(learner, BATCH - 5))
    with pytest.raises(ct.CompileTrainerError, match=r"a micro-batch of 11 rows, but the declared signature is 16"):
        with ctl.guard("update end"):
            learner._compiled_micro_step(*args)
    ctl.check("end")
    assert ctl.compiles_after_lock == 0 and ctl.rejected_after_lock == 0
    assert RC.peek() == {}, f"a refused call must count no route: {RC.peek()}"


# ------------------------------------------------- R1's per-parameter bar, by weight regime (K8)
#: The healthy readings the bars were derived from (`R1_PARAM_BAR`'s comment): arm C's final weights
#: at the gate's batch read 2.47e-3 (the sizing study's FATAL under the old 1e-3 bar — CUDA EAGER's own
#: fp32 error on `history_events.itemtr_emb`; the compiled gradient was within 1.7e-5 of float64).
C_FINAL_HEALTHY_READING = 2.47e-3
#: The retired extractor gate's per-parameter bar (TF32-era; `compile_trainer` no longer carries it).
RETIRED_EXTRACTOR_BAR = 1e-3


def _arms(eps, k=3, n_params=20):
    """Two gradient arms over ``n_params`` parameters; the compiled one carries a relative error of
    exactly ``eps`` on parameter ``k`` (||c_k - e_k|| / ||e_k|| == eps), the loss identical. Parameter
    ``k`` is a SMALL path (1% of a typical norm, above the gate's floor), so the global cosine cannot
    see the error and the per-parameter rule is what is judged."""
    g = torch.Generator().manual_seed(0)
    sizes = [100] * n_params
    parts = [torch.randn(s, generator=g) * (0.01 if i == k else 1.0) for i, s in enumerate(sizes)]
    eager = torch.cat(parts)
    comp = torch.cat([p * (1.0 + eps) if i == k else p for i, p in enumerate(parts)])
    mk = lambda gr: {"loss": torch.tensor([0.5]), "grad": gr, "grad_sizes": torch.tensor(sizes)}  # noqa: E731
    return mk(eager), mk(comp), [f"p{i}" for i in range(n_params)]


@pytest.mark.parametrize("regime", ["fresh", "trained"])
def test_R1s_per_parameter_bar_is_the_REGIMES_measured_bar(regime):
    """C's measured healthy reading passes the TRAINED regime (it FATAL'd every fp32 resume under the
    extractor gate's fresh bar, 1e-3 — fails on revert); twice the regime's bar is refused in both."""
    bar = cr.R1_PARAM_BAR[regime]
    assert bar > RETIRED_EXTRACTOR_BAR                      # the old bar, which the healthy noise exceeds
    e, c, names = _arms(C_FINAL_HEALTHY_READING)
    line = cr._r1_verdict(e, c, names, regime)
    assert f"[{regime} weights]" in line and f"<= {bar:g}" in line
    e, c, names = _arms(2.0 * bar)
    with pytest.raises(ct.CompileTrainerError, match="DISAGREES with eager on 1 parameter"):
        cr._r1_verdict(e, c, names, regime)


def _fresh_learner(monkeypatch):
    """The production-surface learner as a FRESH launch builds it: the golden's construction without
    its seeded perturbation (the zero-init pointer head left uniform)."""
    import agents.model.parity_probe as pp
    from agents.training import learner_golden as LG
    monkeypatch.setattr(pp, "perturb_", lambda *a, **k: None)
    model = LG.build_learner()
    monkeypatch.undo()
    LG.load_buffer_into(model)
    return model


def test_weights_regime_reads_a_fresh_launch_as_fresh_and_moved_weights_as_trained(learner, monkeypatch):
    assert cr.weights_regime(learner) == "trained"          # the golden's perturbed weights
    assert cr.weights_regime(_fresh_learner(monkeypatch)) == "fresh"


def test_the_GATE_judges_trained_weights_at_the_trained_bar(learner, monkeypatch):
    """Through `gate_regions` (CPU, dynamo `eager` backend, so compiled == eager and the only error is
    the planted one): C's healthy reading planted on one parameter of R1's compiled arm PASSES on
    trained weights (the resume / restart / fork case the sizing study FATAL'd on); an error past the
    trained bar is still the typed startup FATAL."""
    real = cr._r1_arm

    def planted(eps):
        def arm(model, fn, args):
            out = real(model, fn, args)
            if fn is model._compiled_micro_step:
                g = out["grad"].clone()
                sizes = [int(x) for x in out["grad_sizes"].tolist()]
                big = max(range(len(sizes)), key=lambda i: float(torch.split(g, sizes)[i].norm()))
                start = sum(sizes[:big])
                g[start:start + sizes[big]] *= (1.0 + eps)
                out = {**out, "grad": g}
            return out
        return arm
    cc.control().install()
    cr.install(learner, backend="eager")
    monkeypatch.setattr(cr, "_r1_arm", planted(C_FINAL_HEALTHY_READING))
    rules = cr.gate_regions(learner, batch_size=BATCH, say=lambda _m: None)
    assert any("[trained weights]" in r for r in rules), rules
    monkeypatch.setattr(cr, "_r1_arm", planted(3.0 * cr.R1_PARAM_BAR["trained"]))
    with pytest.raises(ct.CompileTrainerError, match="DISAGREES with eager"):
        cr.gate_regions(learner, batch_size=BATCH, say=lambda _m: None)


def test_a_FRESH_launch_is_ALSO_judged_on_a_perturbation_at_the_TRAINED_bar(monkeypatch):
    """On fresh weights R1 runs twice: on the weights at the fresh bar, and on the declared ladder's
    first perturbation rung at the trained bar — so an error the fresh bar would absorb (between the
    two bars) is still refused at a fresh launch. Fails if the perturbed pass is dropped."""
    model = _fresh_learner(monkeypatch)
    real = cr._r1_arm
    eps = 0.5 * (cr.R1_PARAM_BAR["trained"] + cr.R1_PARAM_BAR["fresh"])   # between the two bars

    def planted(m, fn, args):
        out = real(m, fn, args)
        if fn is m._compiled_micro_step:
            g = out["grad"].clone()
            sizes = [int(x) for x in out["grad_sizes"].tolist()]
            big = max(range(len(sizes)), key=lambda i: float(torch.split(g, sizes)[i].norm()))
            start = sum(sizes[:big])
            g[start:start + sizes[big]] *= (1.0 + eps)
            out = {**out, "grad": g}
        return out
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    try:
        cc.control().install()
        cr.install(model, backend="eager")
        assert cr.weights_regime(model) == "fresh"
        monkeypatch.setattr(cr, "_r1_arm", planted)
        with pytest.raises(ct.CompileTrainerError, match="DISAGREES with eager"):
            cr.gate_regions(model, batch_size=BATCH, say=lambda _m: None)
        monkeypatch.setattr(cr, "_r1_arm", real)
        rules = cr.gate_regions(model, batch_size=BATCH, say=lambda _m: None)
        assert any("[fresh weights]" in r for r in rules), rules
        assert any("[fresh weights, seeded perturbation" in r for r in rules), rules
    finally:
        cr.uninstall(model)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


# ----------------------------------------------------- a COLLAPSED critic no longer refuses the gate
#
# A win-prob head saturated at a logit of about -9 (a trainee losing ~97 %; `~/gen3ai_archive/cutover_prep/
# fresh3`, 2026-09-30) has a value spread far under V's 1e-4 bar, so the DECISION readout of the deleted
# compiled rollout region R0 was vacuous on its real weights and the gate had to CLIMB the perturbation
# ladder to judge it (and refused a critic no rung could move). R1's verdict is the loss and the
# gradients, not V's spread, so with R0 gone a collapsed critic is judged on its real weights like any
# other and the climb (`_r0_verdicts`) is deleted with it.

def _gate(model):
    return cr.gate_regions(model, batch_size=BATCH, say=lambda _m: None)


def test_a_COLLAPSED_critic_is_judged_by_R1_like_any_other_weights(learner, monkeypatch):
    """The saturated critic (win logit -12) that the old R0 readout REFUSED ("vacuous on every perturbation
    rung") passes the gate now, judged by R1's loss and gradients — and a gradient fault planted on it is
    still caught, so the pass is not vacuous. Fails if an R0-style vacuity refusal comes back."""
    cc.control().install()
    cr.install(learner, backend="eager")
    fe = learner.policy.features_extractor
    head = fe.win_head.net[3]
    with torch.no_grad():
        head.bias.fill_(-12.0)
        head.weight.mul_(0.01)
    rules = _gate(learner)
    assert rules and all(r.startswith("R1 ") for r in rules), rules
    real = cr._r1_arm

    def planted(model, fn, args):
        out = real(model, fn, args)
        if fn is model._compiled_micro_step:
            g = out["grad"].clone()
            sizes = [int(x) for x in out["grad_sizes"].tolist()]
            big = max(range(len(sizes)), key=lambda i: float(torch.split(g, sizes)[i].norm()))
            start = sum(sizes[:big])
            g[start:start + sizes[big]] *= 1.5
            out = {**out, "grad": g}
        return out
    monkeypatch.setattr(cr, "_r1_arm", planted)
    with pytest.raises(ct.CompileTrainerError, match="DISAGREES with eager"):
        _gate(learner)


# --------------------------------------------- no silent fall-back to eager (gen3_no_silent_eager_v1)
def _locked(learner):
    ctl = cc.control()
    ctl.install()
    cr.install(learner, backend="eager")
    ctl.prewarm(cr.prewarm_calls(learner, batch_size=BATCH))
    ctl.lock("test")
    return ctl


def test_a_region_that_runs_EAGER_on_its_COMPILED_route_is_FATAL(learner):
    """Owner, 2026-10-01: no silent performance regression from a partly uncompiled learner. Under
    the `force_eager` stance the compiled callable silently runs its Python body; the dispatcher sees
    the body execute and refuses. Fails if the dispatcher stops watching."""
    from agents.model import region_calls as RC
    _locked(learner)
    args = cr._r1_args(learner, cr.r1_batch(learner, BATCH))
    learner._compiled_micro_step(*args)                       # healthy: compiled, counted
    assert RC.peek().get("R1_compiled") == 1
    torch.compiler.set_stance("force_eager")
    try:
        with pytest.raises(ct.CompileTrainerError, match="region R1 ran EAGER on its COMPILED route"):
            learner._compiled_micro_step(*args)
    finally:
        torch.compiler.set_stance("fail_on_recompile")
        RC.take()


def test_dynamo_DISABLED_at_startup_is_the_gates_FATAL_not_an_eager_run(learner):
    """TORCHDYNAMO_DISABLE (config.disable) makes `torch.compile` hand back the plain function: the
    regions would 'compile' into eager. The gate's first compiled-route call sees the body run."""
    cc.control().install()
    prev = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        cr.install(learner, backend="eager")
        with pytest.raises(ct.CompileTrainerError, match="ran EAGER on its COMPILED route"):
            cr.gate_regions(learner, batch_size=BATCH, say=lambda _m: None)
    finally:
        torch._dynamo.config.disable = prev


def test_the_lock_refuses_every_switch_that_makes_dynamo_run_eager_silently(learner):
    ctl = _locked(learner)
    ctl.check("healthy")
    for name, bad in (("suppress_errors", True), ("disable", True)):
        prev = getattr(torch._dynamo.config, name)
        setattr(torch._dynamo.config, name, bad)
        try:
            with pytest.raises(cc.CompileSentinelError, match=name):
                ctl.check("after the switch")
        finally:
            setattr(torch._dynamo.config, name, prev)
    torch.compiler.set_stance("default")
    try:
        with pytest.raises(cc.CompileSentinelError, match="not 'fail_on_recompile'"):
            ctl.check("after the stance moved")
    finally:
        torch.compiler.set_stance("fail_on_recompile")
    ctl.check("restored")


def test_the_run_asserts_the_compiled_inventory_equals_the_declaration(learner):
    _locked(learner)
    assert "inventory == declaration" in cr.assert_inventory(learner)
    torch.compiler.set_stance("default")
    try:
        args = cr._r1_args(learner, cr.r1_batch(learner, BATCH * 2))
        learner._compiled_micro_step._gen3_compiled(*args)    # an undeclared second R1 graph
    finally:
        torch.compiler.set_stance("fail_on_recompile")
    with pytest.raises(ct.CompileTrainerError, match="INVENTORY differs"):
        cr.assert_inventory(learner)


def test_each_update_records_its_compiled_and_eager_region_calls(learner):
    """The per-update window: every R1 micro-batch is a compiled-route call; there is NO declared eager
    route any more, so `eager_fallback_calls` / `eager_share` read 0 — and a future declared route
    (`<region>_eager_<why>`, counted by its dispatcher) is what they would carry (synthetic here)."""
    from agents.model import region_calls as RC
    ctl = _locked(learner)
    full = cr._r1_args(learner, cr.r1_batch(learner, BATCH))
    learner._compiled_micro_step(*full)
    learner._compiled_micro_step(*full)

    class _Log:
        name_to_value = {"train/train_ms": 36300.0}
        rec: dict = {}

        def record(self, k, v):
            self.rec[k] = v
    log = _Log()
    learner._logger = log
    ctl.record(learner)
    assert log.rec["lifecycle/compiled_region_calls"] == 2.0
    assert log.rec["lifecycle/eager_fallback_calls"] == 0.0 and log.rec["lifecycle/eager_share"] == 0.0
    assert log.rec["lifecycle/update_wall_s"] == 36.3
    assert RC.peek() == {}                                    # the window was reset
    RC.count("R1_compiled")
    RC.count("RX_eager_synthetic")                            # what a future declared eager route would count
    ctl.record(learner)
    assert abs(log.rec["lifecycle/eager_share"] - 0.5) < 1e-9 and log.rec["lifecycle/eager_fallback_calls"] == 1.0


# ------------------------------------------------ one startup gate per region (gen3_one_gate_per_region_v1)


# ------------------------------------------------ one startup gate per region (gen3_one_gate_per_region_v1)
def test_the_sentinel_installs_the_regions_and_the_preflight_compiles_nothing(learner, monkeypatch, capsys):
    """The trainer's compile step (`preflight_compile_trainer`) compiles NOTHING and the sentinel goes
    straight to the regions, having INSTALLED the control first (the compile config pinned — K1b's
    donated_buffer=False — and the detectors on). Fails if an extractor-level compile comes back
    (an instance `fe.forward`) or if the sentinel stops installing the control before the regions."""
    monkeypatch.setattr(ct, "resolve_device", lambda fe: torch.device("cuda"))   # the CPU refusal's seam
    assert ct.preflight_compile_trainer(learner, True) is None
    assert "forward" not in vars(learner.policy.features_extractor)
    assert "DECLARED REGION (R1" in capsys.readouterr().out
    reached = []

    class _Stop(Exception):
        pass

    def stop(model, **_k):
        assert cc.control().installed
        assert torch._functorch.config.donated_buffer is False
        reached.append(model)
        raise _Stop
    monkeypatch.setattr(cr, "install", stop)
    with pytest.raises(_Stop):
        ct.arm_compile_sentinel(learner, batch_size=BATCH)
    assert reached == [learner]


def test_the_trainer_calls_the_preflight_and_the_sentinel_with_no_fallback_route():
    import inspect
    from main.train import lifecycle
    src = inspect.getsource(lifecycle)
    assert "preflight_compile_trainer(model" in src and "arm_compile_sentinel(model" in src
    assert "compile_trainer_extractor" not in src and "regions_follow" not in src
