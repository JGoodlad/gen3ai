"""P10-C — R1's startup gate (and so the canary) ALWAYS judges the K9 learner golden's real labelled
rows, and REFUSES a judged parameter set smaller than the golden rows' own
(`gen3_r1_golden_rows_always_v1`, `gen3_r1_judged_set_v1`).

The defect (P10 review, confirmed): when a run's observation keys differed from the golden's — every
`--fork-fraction > 0` run, whose space adds `fork_pg_m` — R1 was judged on the real-obs fixture with
every label at ZERO. The critic, intent and belief losses were then zero, their gradients fell under
the per-parameter floor, and ~48 of their parameters went unjudged (176 judged vs 219 on the golden
rows): a backward miscompile of the win-prob critic would have passed the gate and every canary
check, silently.

CPU, eager arms (the judged set is a property of the EAGER gradient on the rows), plus one gate run
on dynamo's `eager` backend for the refusal path. Each test below FAILS on a revert of the fix.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.training.fork_arm import PG_MASK_KEY

BATCH = 64
N_ENVS = 4


def _learner(fork: bool):
    from agents.training import learner_golden as LG
    if not fork:
        model = LG.build_learner()
    else:
        from main.train.production_args import production_args
        args = production_args()
        args.fork_fraction = 0.1
        model = LG.build_learner(args=args)
    # NOT `load_buffer_into` (it refuses a space other than the golden's): at the startup gate the
    # buffer is EMPTY in a real launch too — R1's gate reads only the buffer's KEY SET (the fold's
    # predicates) and the committed golden file, never the buffer's rows.
    return model


@pytest.fixture(scope="module")
def production():
    return _learner(False)


@pytest.fixture(scope="module")
def fork():
    return _learner(True)


def _eager_arm(model, b):
    from agents.training.instrumented_ppo.micro_step import micro_step
    was = model.policy.training
    model.policy.set_training_mode(True)
    try:
        return cr._r1_arm(model, micro_step, cr._r1_args(model, b))
    finally:
        model.policy.set_training_mode(was)


def _names(model):
    return [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]


def test_a_fork_shaped_space_is_judged_on_the_golden_rows_with_its_mask_at_the_declared_1(fork):
    assert PG_MASK_KEY in fork.policy.observation_space.spaces          # the precondition: fork-shaped
    b = cr.r1_batch(fork, BATCH)
    assert b.source.startswith("the K9 learner golden's real labelled buffer"), b.source
    assert b.filled == (PG_MASK_KEY,)
    assert torch.equal(b.obs[PG_MASK_KEY], torch.ones_like(b.obs[PG_MASK_KEY]))
    assert list(b.obs) == list(fork.policy.observation_space.spaces)


def test_a_fork_shaped_space_judges_the_FULL_golden_parameter_set(production, fork):
    """Same seeds, same weights: the fork learner's judged set on its filled rows must cover every
    parameter the production learner judges clearly on the golden rows (and the counts agree up to
    the band). Reverted, the fork learner is judged on zero labels: ~48 parameters go missing."""
    for (_, p), (_, q) in zip(production.policy.named_parameters(), fork.policy.named_parameters()):
        assert torch.equal(p, q)                                         # the precondition
    ref = _ratios(_eager_arm(production, cr.r1_batch(production, BATCH)))
    run = _ratios(_eager_arm(fork, cr.r1_batch(fork, BATCH)))
    fl = ct._PARAM_GRAD_FLOOR
    names = _names(fork)
    clear = [i for i, r in enumerate(ref) if r > 2.0 * fl]              # judged CLEARLY (rule 8)
    missing = [names[i] for i in clear if not run[i] > fl]
    assert len(clear) > 150, len(clear)                                 # it judged something real
    assert not missing, f"{len(missing)} parameters unjudged on the fork shape: {missing[:10]}"


def _ratios(arm):
    """Each parameter's eager gradient norm over the largest — the per-parameter rule's floor axis
    (computed here, independently of `compile_regions`, so a revert fails on the SUBSTANCE)."""
    sizes = [int(x) for x in arm["grad_sizes"].tolist()]
    norms = [float(g.norm()) for g in torch.split(arm["grad"].float(), sizes)]
    top = max(norms)
    return [n / top for n in norms]


def test_the_gate_REFUSES_a_placeholder_that_shrinks_the_judged_set(fork, monkeypatch):
    """A placeholder that BLANKS a term (the fork mask at 0 deletes the policy gradient) must be the
    typed startup FATAL, not a quieter gate. The positive half: the declared placeholder passes."""
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    # R1's declared signature is the MODEL's batch size (`install` reads it); the gate below judges BATCH
    # rows, so they must agree — a ragged call used to take the eager route silently, and the "compiled"
    # arm of this gate was eager (R1 refuses it now, `gen3_r1_no_ragged_v1`)
    monkeypatch.setattr(fork, "batch_size", BATCH)
    try:
        cc.control().install()
        cr.install(fork, backend="eager")
        rules = cr.gate_regions(fork, batch_size=BATCH, say=lambda _m: None)
        r1 = [r for r in rules if r.startswith("R1 judged set")]
        assert r1 and "[filled ['fork_pg_m']]" in r1[0], rules

        def blanking(key, space):
            return np.zeros(tuple(space.shape), dtype=space.dtype)
        monkeypatch.setattr(cr, "fill_value", blanking)
        with pytest.raises(ct.CompileTrainerError, match="SHRANK the judged parameter set"):
            cr.gate_regions(fork, batch_size=BATCH, say=lambda _m: None)
    finally:
        cr.uninstall(fork)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def test_a_key_with_NO_declared_placeholder_is_REFUSED(fork, monkeypatch):
    """Reverted, an undeclared key silently sent R1 to the zero-label fixture."""
    from gymnasium import spaces
    monkeypatch.setitem(fork.policy.observation_space.spaces, "undeclared_label",
                        spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32))
    with pytest.raises(ct.CompileTrainerError, match=r"undeclared_label.*NO declared placeholder"):
        cr.r1_batch(fork, BATCH)


def test_judged_set_shrinkage_compares_only_OUTSIDE_the_band():
    """Pure rule (`JUDGED_SET_BAND`): a parameter counts as missing only when the reference judges it
    CLEARLY (> band x floor) and the run does not judge it (<= floor) — a hair either side of the
    floor never decides the verdict."""
    fl, band = 1e-3, cr.JUDGED_SET_BAND

    def arm(ratios):
        g = torch.tensor([1.0] + list(ratios))
        return {"grad": g, "grad_sizes": torch.ones(len(g), dtype=torch.long)}
    names = ["top", "clear", "hair", "gone"]
    ref = arm([10 * fl, 1.01 * fl, band * fl * 1.5])
    run_ok = arm([10 * fl, 0.99 * fl, band * fl * 1.5])        # only the in-band one moved
    assert cr.judged_set_shrinkage(ref, run_ok, names, floor=fl) == ([], 3, 3, 1)
    run_bad = arm([10 * fl, 1.01 * fl, 0.0])                    # a clearly-judged one went to zero
    missing, clear, judged, banded = cr.judged_set_shrinkage(ref, run_bad, names, floor=fl)
    assert missing == ["gone"] and clear == 3 and banded == 1
