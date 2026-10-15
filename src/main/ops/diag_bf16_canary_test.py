"""The bf16-canary DIAGNOSTIC (`main.ops.diag_bf16_canary`) — the CPU-checkable half of the GPU read.

What a GPU window cannot afford to find out: that a read means something other than it says. So the formulas are
held to the code they model, on CPU:
* the IMPLIED logit gradient (`implied_dz`), pushed through the pointer head, reproduces the pointer head's
  gradient of the REAL `micro_step` (the K9 golden learner, its committed rows) — the forward-vs-backward read is only
  meaningful if this holds; and the per-row decomposition sums back to it (kappa >= 1);
* the canary's statistic replayed (`canary_stat`) equals the rule's own numbers (`compile_regions.GradWatch`);
* the dither restores the weights BIT-EXACTLY and applies the same noise to every learner;
* the checkpoint resolution refuses rather than substitutes; the null's share above K; the verdict renders.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from main.ops import diag_bf16_canary as D


# ------------------------------------------------------------------------------------------- pure functions
def _run_dir(tmp_path, steps):
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    for n in steps:
        (ck / f"checkpoint_{n}_steps.zip").write_text("x")
    return tmp_path


def test_a_checkpoint_spec_resolves_to_the_NEAREST_and_refuses_a_far_one(tmp_path):
    run = _run_dir(tmp_path, [15000067, 16000158, 72000094])
    assert D.resolve_checkpoint(run, "72M")[1].name == "checkpoint_72000094_steps.zip"
    assert D.resolve_checkpoint(run, "15M")[1].name == "checkpoint_15000067_steps.zip"
    with pytest.raises(D.DiagError, match="within 0.6M"):
        D.resolve_checkpoint(run, "40M")
    with pytest.raises(D.DiagError, match="final_model_canary_fatal"):
        D.resolve_checkpoint(run, "fatal")
    (run / D.FATAL_ZIP).write_text("x")
    assert D.resolve_checkpoint(run, "fatal") == ("fatal", run / D.FATAL_ZIP)


def test_the_dither_moves_every_learner_by_the_SAME_noise_and_restores_BIT_EXACTLY():
    torch.manual_seed(0)
    a = SimpleNamespace(policy=torch.nn.Linear(16, 8))
    b = SimpleNamespace(policy=torch.nn.Linear(16, 8))
    b.policy.load_state_dict(a.policy.state_dict())
    before = {n: p.detach().clone() for n, p in a.policy.named_parameters()}
    with D.dithered([a, b], seed=3):
        for (n, pa), (_, pb) in zip(a.policy.named_parameters(), b.policy.named_parameters()):
            assert torch.equal(pa, pb)
            rel = ((pa - before[n]) / before[n]).abs().detach()
            assert 0 < float(rel.max()) < 8 * D.DITHER_REL
    with D.dithered([a], seed=4):
        moved4 = a.policy.weight.detach().clone()
    with D.dithered([a], seed=3):
        assert not torch.equal(a.policy.weight, moved4)        # another seed, another draw
    for n, p in a.policy.named_parameters():
        assert torch.equal(p, before[n])


def test_the_null_counts_the_share_of_HEALTHY_pairs_above_K():
    per_seed = [{"Ebf": {"p": 0.02}, "Cbf": {"p": 0.03}}, {"Ebf": {"p": 0.04}, "Cbf": {"p": 0.5}},
                {"Ebf": {"p": 0.02}, "Cbf": {"p": 0.02}}]
    s = D.null_summary(per_seed, k=12.0, floor=0.01)
    assert s["seeds"] == 3
    assert s["ratio_null"]["p"]["Ebf_pairs"]["n"] == 6 and s["ratio_null"]["p"]["Ebf_pairs"]["p_over_k"] == 0.0
    # Cbf / Ebf per seed: 1.5, 12.5, 1.0 -> one of three above 12
    assert s["ratio_null"]["p"]["Cbf_vs_Ebf"]["p_over_k"] == pytest.approx(1 / 3)


def _arm(grad, loss=1.0):
    z = torch.zeros(1)
    return D.ArmOut(loss, torch.tensor(grad), z, z, z, z)


def test_the_canary_statistic_replayed_EQUALS_the_rules_own_numbers():
    """`canary_stat` replays `_r1_verdict_bf16`'s per-parameter statistic; the rule's own watch record (the same
    numbers the run's canary writes) must agree with it exactly, and a 50 % compiled error is over K in both."""
    from agents.model import compile_regions as cr
    names = ["pointer_head.move_proj.weight", "pointer_head.move_score.weight", "trunk.w"]
    sizes = [2, 2, 4]
    ref = _arm([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    eager = _arm([1.02, 1.0, 1.0, 1.03, 1.0, 1.0, 1.01, 1.0])
    comp = _arm([1.5, 1.0, 1.0, 1.02, 1.0, 1.0, 1.01, 1.0])
    st = D.canary_stat(comp, eager, ref, names, sizes, "trained")
    w = cr.GradWatch(("pointer_head.",))
    out = D.canary_verdict(comp, eager, ref, names, sizes, "trained")
    assert out["verdict"] == "refused"
    gs = torch.tensor(sizes)
    with pytest.raises(Exception):
        cr._r1_verdict_bf16({"loss": torch.tensor([1.0]), "grad": eager.grad, "grad_sizes": gs},
                            {"loss": torch.tensor([1.0]), "grad": comp.grad, "grad_sizes": gs},
                            {"loss": torch.tensor([1.0]), "grad": ref.grad, "grad_sizes": gs},
                            names, "trained", watch=w)
    for n, v in w.params.items():
        assert st["watch"][n]["ratio"] == pytest.approx(v["ratio"], rel=1e-9)
    assert "pointer_head.move_proj.weight" in st["over_k"]
    assert "pointer_head.move_score.weight" not in st["over_k"]


def test_the_clip_branch_and_the_relocation_move_ONLY_the_rows_near_their_SWITCHING_boundary():
    """`ppo_active` is `micro_step`'s clip branch; the proposed canary fix (`relocate_old_log_prob`) moves a row only
    when its reference ratio sits within the margin of the boundary that switches ITS gradient (A > 0: 1 + clip;
    A < 0: 1 - clip), and puts it 2 x margin inside the active side; every other row keeps the golden's old
    log-prob."""
    st = SimpleNamespace(clip_range=0.15)
    lp = torch.zeros(4)
    rho = torch.tensor([1.155, 0.846, 0.4, 5.0], dtype=torch.float64)       # near hi, near lo, far, far
    old = (lp.double() - rho.log()).float()
    adv = torch.tensor([1.0, -1.0, 1.0, -1.0])
    ref = D.ArmOut(0.0, torch.zeros(1), torch.zeros(4, 11), lp, torch.zeros(4), adv)
    assert D.ppo_active(ref, old, st).tolist() == [False, False, True, True]
    new, info = D.relocate_old_log_prob(ref, old, st, margin=0.5)
    assert info["moved_batch_rows"] == 2
    assert D.ppo_active(ref, new, st).tolist() == [True, True, True, True]
    rho_new = (lp.double() - new.double()).exp()
    assert float(rho_new[0]) == pytest.approx(1.15 * 2.718281828 ** -1.0, rel=1e-5)
    assert float(rho_new[1]) == pytest.approx(0.85 * 2.718281828, rel=1e-5)
    assert torch.equal(new[2:], old[2:])


def test_the_verdict_template_renders_a_growth_row_from_a_record(tmp_path):
    n = D.WATCH + "move_proj.weight"
    rec = {"label": "fatal", "slices": {"0": {
        "pairwise": {"Ebf|E32": {n: 0.028}, "Cbf|E32": {n: 0.40}, "Cbf|Ebf": {n: 0.39}, "C32|E32": {n: 1e-5}},
        "canary_stat": {"Cbf": {"watch": {n: {"ratio": 14.3}}}},
        "rows": {"params": {n: {"kappa": 2.6}}}}}}
    (tmp_path / "gpu_fatal.json").write_text(json.dumps(rec))
    md = D.render_verdict(tmp_path)
    row = next(line for line in md.splitlines() if line.startswith("| gpu_fatal"))
    assert "0.028" in row and "0.4" in row and "14.3" in row and "2.6" in row
    assert "(i) precision-independent miscompile" in md


# -------------------------------------------------------------- the formulas against the REAL micro-step
@pytest.fixture(scope="module")
def golden_arms():
    """The K9 golden learner (production surface, bf16 trunk), its 64 committed rows untiled, E32 + Ebf."""
    from agents.model import compile_regions as cr
    from agents.training import learner_golden as LG
    from agents.training.instrumented_ppo.micro_step import micro_step
    model = LG.build_learner()
    LG.load_buffer_into(model)
    model.policy.set_training_mode(True)
    names, sizes = D.param_table(model)
    b = cr.r1_batch(model, 64, 0)
    call_args = cr._r1_args(model, b)
    arms = {"E32": D.run_arm(model, micro_step, call_args, fp32_trunk=True),
            "Ebf": D.run_arm(model, micro_step, call_args, fp32_trunk=False)}
    return model, names, sizes, call_args, arms


@pytest.mark.integration
def test_the_IMPLIED_logit_gradient_reproduces_the_real_micro_steps_pointer_head_gradient(golden_arms):
    """The forward-vs-backward read pushes each arm's implied dL/dz through the pointer head: it means something only
    if, for the fp32 reference, that reproduces the REAL micro-step's pointer-head gradient (the PPO clipped surrogate
    and the entropy term are the only terms that reach the head). Fails if the formula drifts from `micro_step`."""
    model, names, sizes, call_args, arms = golden_arms
    read = D.implied_read(model, arms, call_args, names, sizes)
    assert read["judged"], read
    assert read["validity"]["valid"], read["validity"]
    assert read["validity"]["worst_self_E32"] < 1e-4
    # the bf16 arm differs from the reference, and its own forward accounts for most of the head's difference
    assert read["Ebf"]["dz_vs_ref"] > 0


@pytest.mark.slow
def test_EVERY_route_of_the_GPU_read_RUNS_through_a_smoke_backend_on_cpu(monkeypatch, tmp_path):
    """The GPU window is ONE shot, so every route `read_checkpoint` takes with ALL of `GPU_ARMS` — the production
    install, the fp32 second learner, `emulate_precision_casts`'s arm, `aot_eager`, the graph-break arm, the canary's
    own `_check`, the fp32 rule, the relocated reads and the null — must RUN before it is spent. Here on CPU through
    dynamo's `eager` backend (`--smoke-backend`: compiled == eager arithmetic, so every verdict must PASS), on the K9
    golden learner at the fixture's 64 rows."""
    from agents.model import compile_control as cc
    from agents.model import compile_regions as cr
    from agents.training import learner_golden as LG

    def fake_build(zip_path, device, *, precision="bf16", batch=None):
        m = LG.build_learner(trunk_precision=precision)
        m.batch_size = int(batch or D.RUN_BATCH)
        m.policy.set_training_mode(True)
        return m
    monkeypatch.setattr(D, "build_learner", fake_build)
    monkeypatch.setattr(D, "load_weights", lambda models, zp: None)
    monkeypatch.setattr(D, "RUN_BATCH", 64)
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    prev_limit = torch._dynamo.config.cache_size_limit           # `Context` raises it for its many arms
    ctx = None
    try:
        ctx = D.Context(tmp_path / "x.zip", "cpu", D.GPU_ARMS, lambda s: None, smoke_backend="eager")
        rec = D.read_checkpoint(ctx, "golden", tmp_path / "x.zip", dither_n=1, slices=(0,))
    finally:
        for m in (ctx.models if ctx is not None else []):
            if cr.installed(m):
                cr.uninstall(m)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.config.cache_size_limit = prev_limit
        torch._dynamo.reset()
    assert rec["failed_arms"] == {}, rec["failed_arms"]
    s0 = rec["slices"]["0"]
    assert set(s0["loss"]) == set(D.GPU_ARMS)
    assert s0["canary_verdict"]["verdict"] == "pass", s0["canary_verdict"]
    assert s0["canary_replay"]["verdict"] == "pass", s0["canary_replay"]
    assert s0["fp32_rule"]["over_bar"] == [], s0["fp32_rule"]
    assert s0["relocated"]["canary_verdict"]["verdict"] == "pass"
    assert set(s0["implied"]) >= set(D.GPU_ARMS) and s0["implied"]["validity"]["valid"]
    n = D.WATCH + "move_proj.weight"
    assert rec["null"]["ratio_null"][n]["Cbf_vs_Ebf"]["n"] == 1
    assert rec["null"]["relocated"]["ratio_null"][n]["Cbf_vs_Ebf"]["n"] == 1


@pytest.mark.integration
def test_the_per_row_decomposition_SUMS_BACK_to_the_heads_gradient(golden_arms):
    model, names, sizes, call_args, arms = golden_arms
    rows = D.row_decomposition(model, arms, call_args, 0, 64)
    assert rows["rows"] == 64
    for n, rec in rows["params"].items():
        if rec["sum_norm"] > 0:
            assert rec["kappa"] >= 1.0 - 1e-9, (n, rec["kappa"])
            assert sum(r["share_of_abs"] for r in rec["top_rows"]) <= 1.0 + 1e-9
    # the decomposition's total is the implied gradient's: the sum over rows equals the head's E32 gradient
    inputs = D.head_inputs(model, call_args[1], fp32_trunk=True)
    st = call_args[-1]
    dz = D.implied_dz(arms["E32"], call_args[2], call_args[4], call_args[3], st)
    (g,) = D.head_grads(model, inputs, [dz])
    n = D.WATCH + "move_proj.weight"
    assert rows["params"][n]["sum_norm"] == pytest.approx(float(g[n].norm()), rel=1e-6)
