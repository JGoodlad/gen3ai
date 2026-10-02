"""Tests for `--compile-trainer`'s shared pieces (`agents.model.compile_trainer`): the parity
VERDICTS every compile gate applies, the shape-stability refusal, the committed real-obs fixture and
the startup PREFLIGHT. The declared regions themselves (install, gate, prewarm, lock) are
`compile_regions_test.py`'s.

Every rule this flag enforces guards a failure that would otherwise be SILENT: a fallback to eager
costs ~1.75-2x forever and the run keeps training correctly, so nothing in any metric would surface
it. The verdicts are PURE FUNCTIONS precisely so they can be tested on any box.
"""
import pytest
import torch

import numpy as np

from agents.model.compile_trainer import (_MAX_NUMERIC_DRIFT, _TF32_EPS, _TF32_K, CompileTrainerError,
                                          check_numerics, check_shape_stability, fp32_reference,
                                          parity_verdict, preflight_compile_trainer, resolve_device)


class _FakeFE(torch.nn.Module):
    """Stand-in with the two things the helper reads: parameters (for the device) and a width.

    Deliberately not the real extractor — these are control-flow tests, and building the real one
    would make them slow enough that nobody runs them.
    """

    def __init__(self, device="cpu", obs_dim=32):
        super().__init__()
        self.lin = torch.nn.Linear(obs_dim, 8)
        self.obs_dim = obs_dim
        self.to(device)

    def forward(self, obs):
        h = self.lin(obs["observation"])
        return h, h


def _model(device="cpu", obs_dim=32):
    m = torch.nn.Module()
    m.policy = torch.nn.Module()
    m.policy.features_extractor = _FakeFE(device, obs_dim)
    return m


# --------------------------------------------------------------------------- the no-op


def test_disabled_is_a_true_noop():
    """OFF must not touch the model at all: an off run has to be byte-identical."""
    m = _model()
    fe = m.policy.features_extractor
    assert preflight_compile_trainer(m, False) is None
    assert "forward" not in vars(fe), "OFF must not install an instance attribute at all"


# --------------------------------------------------------------------------- pure verdicts


def test_numerics_drift_is_refused():
    """A faster wrong model is not a win."""
    check_numerics(0.0)
    check_numerics(9e-5)                        # under tolerance: cuBLAS reduction-order noise
    with pytest.raises(CompileTrainerError) as e:
        check_numerics(1e-3)
    assert "DISAGREES" in str(e.value)


def test_fp32_rule_is_UNCHANGED_by_the_precision_aware_gate():
    """gen3_tf32_parity_gate_v1 must not weaken the default path: at 'highest' the rule is the same
    1e-4 on max|compiled - eager|, and a reference — even a wildly wrong one — is never consulted
    (so it cannot LOOSEN the fp32 bar)."""
    assert _MAX_NUMERIC_DRIFT == 1e-4
    check_numerics(9e-5, precision="highest")
    with pytest.raises(CompileTrainerError, match="DISAGREES"):
        check_numerics(1e-3, precision="highest", eager_err=1.0)   # eager_err is IGNORED at fp32
    eager = (torch.zeros(4, 8), torch.zeros(4, 3))
    bogus_ref = (torch.full((4, 8), 5.0), torch.full((4, 3), 5.0))
    ok = parity_verdict(eager=eager, compiled=(eager[0] + 5e-5, eager[1]), reference=bogus_ref,
                        precision="highest")
    assert "fp32" in ok and "< 0.0001" in ok
    with pytest.raises(CompileTrainerError, match="DISAGREES"):
        parity_verdict(eager=eager, compiled=(eager[0] + 2e-4, eager[1]), reference=bogus_ref,
                       precision="highest")


def test_tf32_rule_is_relative_to_what_eager_tf32_itself_pays():
    """At 'high' both arms are measured against an fp32 reference; compiled passes iff
    e_comp <= K*e_eager + EPS. The T32 launch's 7.62e-03 compiled-vs-eager is the kind of number
    this admits; a disagreement well past K x eager's own rounding is refused."""
    ref = (torch.zeros(4, 8), torch.zeros(4, 3))
    eager = (ref[0] + 3e-3, ref[1])                       # eager TF32 rounding: e_eager = 3e-3
    ok = parity_verdict(eager=eager, compiled=(ref[0] - 5e-3, ref[1]), reference=ref,
                        precision="high")                 # |c-e| = 8e-3 > 1e-4, but e_comp 5e-3 ok
    assert "'high'" in ok and "TF32" in ok and "e_eager" in ok and "e_comp" in ok
    bar = _TF32_K * 3e-3 + _TF32_EPS
    with pytest.raises(CompileTrainerError, match="DISAGREES") as e:
        parity_verdict(eager=eager, compiled=(ref[0], ref[1] + 1.5 * bar), reference=ref,
                       precision="high")
    assert "fp32" in str(e.value) and "wrong kernel" in str(e.value)


def test_tf32_rule_is_never_looser_than_fp32_when_eager_is_exact():
    """e_eager == 0 (nothing TF32 touched) collapses the bar to EPS == the fp32 tolerance."""
    assert _TF32_EPS == _MAX_NUMERIC_DRIFT
    check_numerics(9e-5, precision="high", eager_err=0.0)
    with pytest.raises(CompileTrainerError, match="DISAGREES"):
        check_numerics(2e-4, precision="high", eager_err=0.0)


def test_tf32_rule_refuses_without_a_reference_and_on_nan():
    """No fp32 reference => no claim; NaN anywhere => FAIL, never a pass."""
    t = (torch.zeros(2, 2), torch.zeros(2, 1))
    with pytest.raises(CompileTrainerError, match="no fp32 reference"):
        parity_verdict(eager=t, compiled=t, reference=None, precision="high")
    for bad_eager in (None, float("nan"), float("inf"), -1.0):
        with pytest.raises(CompileTrainerError):
            check_numerics(0.0, precision="high", eager_err=bad_eager)
    with pytest.raises(CompileTrainerError):
        check_numerics(float("nan"), precision="high", eager_err=1e-3)


@pytest.fixture
def _restore_matmul_precision():
    prev = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(prev)


def test_fp32_reference_runs_at_highest_and_restores_the_callers_precision(
        _restore_matmul_precision):
    """The reference forward must see 'highest', and the trainer's precision must come back —
    INCLUDING when the forward raises (else a TF32 argv would silently train at fp32, or vice versa)."""
    torch.set_float32_matmul_precision("high")
    seen = []

    def fn(obs):
        seen.append(torch.get_float32_matmul_precision())
        return obs["observation"], obs["observation"]

    fp32_reference(fn, {"observation": torch.ones(2, 3)})
    assert seen == ["highest"] and torch.get_float32_matmul_precision() == "high"

    def boom(obs):
        raise RuntimeError("forward exploded")

    with pytest.raises(RuntimeError):
        fp32_reference(boom, {"observation": torch.ones(2, 3)})
    assert torch.get_float32_matmul_precision() == "high"


# --------------------------------------------------------------------------- the refusals


def test_cpu_is_refused():
    """CPU is rejected up front rather than attempted: the compiled learner is CUDA-gated only."""
    with pytest.raises(CompileTrainerError, match="requires CUDA"):
        preflight_compile_trainer(_model("cpu"), True)


def test_missing_extractor_is_refused():
    m = torch.nn.Module()
    m.policy = torch.nn.Module()
    with pytest.raises(CompileTrainerError, match="no `features_extractor`"):
        preflight_compile_trainer(m, True)


def test_a_learner_without_the_micro_step_is_REFUSED_not_compiled_another_way(monkeypatch):
    """The declared regions are the ONLY compiled learner surface (the torch-2.5.1 extractor-only
    compile was deleted, K1 2026-10-02): a learner without `_micro_static` cannot build R1, so it is
    refused at the compile step rather than silently compiled as something smaller. Fails if a
    fallback compile path comes back."""
    monkeypatch.setattr("agents.model.compile_trainer.resolve_device",
                        lambda fe: torch.device("cuda"))   # past the device gate, not around it
    m = _model("cpu")
    with pytest.raises(CompileTrainerError, match="no micro-step"):
        preflight_compile_trainer(m, True)
    assert "forward" not in vars(m.policy.features_extractor)
    from agents.model.compile_trainer import arm_compile_sentinel
    with pytest.raises(CompileTrainerError, match="instrumented learner"):
        arm_compile_sentinel(m, n_envs=4, batch_size=16)
    lines = []
    m._micro_static = lambda *a: None
    preflight_compile_trainer(m, True, emit=lines.append)
    assert lines and "DECLARED REGIONS" in lines[0]


def test_resolve_device_reads_the_models_real_device():
    assert resolve_device(_FakeFE("cpu")).type == "cpu"


# --------------------------------------------------------------- shape stability (the real hazard)


def _stable(**kw):
    base = dict(n_steps=2048, n_envs=48, batch_size=4096, async_rollout=False)   # gen-10's config
    base.update(kw)
    return base


def test_the_production_config_is_accepted():
    """gen-10: 2048*48 = 98304 = 24 x 4096 exactly. Two shapes total, well under cache_size_limit."""
    check_shape_stability(**_stable())


def test_async_rollout_is_refused():
    """The async collector forwards whichever envs are READY, so the batch VARIES by construction —
    an unbounded shape set, which exhausts dynamo's cache and drops to eager SILENTLY."""
    with pytest.raises(CompileTrainerError) as e:
        check_shape_stability(**_stable(async_rollout=True))
    msg = str(e.value)
    assert "--async-rollout" in msg and "SILENTLY" in msg
    assert "+14%" in msg and "+62%" in msg, "the error should let you pick, with the measured numbers"


def test_a_remainder_minibatch_is_refused_with_a_concrete_suggestion():
    """A remainder is a THIRD shape replayed every epoch, for no benefit. The error must not leave
    the reader doing arithmetic — it names a batch size that actually divides."""
    with pytest.raises(CompileTrainerError) as e:
        check_shape_stability(**_stable(batch_size=5000))
    msg = str(e.value)
    assert "remainder" in msg
    import re
    m = re.search(r"e\.g\. (\d+)", msg)
    assert m, f"no concrete suggestion in: {msg}"
    suggested = int(m.group(1))
    assert (2048 * 48) % suggested == 0, f"suggested {suggested} does not divide the rollout"
    assert suggested <= 5000


def test_the_suggestion_is_the_LARGEST_divisor_that_fits():
    """Suggesting 1 would be technically correct and useless."""
    with pytest.raises(CompileTrainerError) as e:
        check_shape_stability(**_stable(batch_size=5000))
    import re
    assert int(re.search(r"e\.g\. (\d+)", str(e.value)).group(1)) == 4096


def test_a_zero_batch_size_does_not_divide_by_zero():
    check_shape_stability(**_stable(batch_size=0))     # unknown -> not our call to police


# ------------------------------------------------ gen3_compile_parity_real_obs_v1: the fixture + verdicts


def test_the_committed_real_obs_fixture_matches_the_live_layout():
    """A layout change makes the fixture STALE, and a stale fixture must fail HERE (the routine
    gate), not at a launch. Regenerate with `python -m agents.model.compile_parity_fixture --write`."""
    from agents.model.compile_parity_fixture import N_ROWS, load_parity_rows
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    width = Gen3ObservationEncoder(load_mappings()).get_layout()["total_dim"]
    rows, mask = load_parity_rows(width)
    assert rows.shape == (N_ROWS, width) and rows.dtype == np.float32
    assert mask.shape[0] == N_ROWS and mask.any(axis=1).all()
    assert (rows != 0).any(axis=1).all(), "every fixture row must be a REAL state, never a zero row"


def test_a_stale_or_missing_fixture_REFUSES_rather_than_falling_back(monkeypatch, tmp_path):
    """The fallback (zeros) is exactly the probe that hid the miscompile — so there is none."""
    from agents.model import compile_parity_fixture as cpf
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    width = Gen3ObservationEncoder(load_mappings()).get_layout()["total_dim"]
    with pytest.raises(cpf.ParityFixtureError, match="STALE"):
        cpf.load_parity_rows(width + 1)
    monkeypatch.setattr(cpf, "FIXTURE_PATH", tmp_path / "absent.npz")
    with pytest.raises(cpf.ParityFixtureError, match="missing"):
        cpf.load_parity_rows(width)


def test_decision_level_bars_catch_a_value_or_policy_shift_that_features_alone_miss():
    """The gate reads what the rollout and PPO read: masked legal log-probs and V, each at its
    own fp32 bar."""
    from agents.model.compile_trainer import _FP32_TOL, decision_verdicts
    # INFORMATIVE rows (gen3_fresh_parity_probe_v1): a quantity constant across rows / within
    # every row's legal entries is REFUSED as vacuous — `parity_probe_test` pins that.
    g = torch.Generator().manual_seed(0)
    lp = torch.zeros(4, 11)
    lp[:, :3] = torch.log_softmax(torch.randn(4, 3, generator=g), dim=-1)
    base = {"features": torch.randn(4, 8, generator=g), "legal_logprob": lp,
            "value": torch.linspace(0.2, 0.8, 4)}
    ok = decision_verdicts(eager=base, compiled={k: v.clone() for k, v in base.items()},
                           precision="highest")
    assert len(ok) == 3
    bad_v = {k: v.clone() for k, v in base.items()}
    bad_v["value"] += 2 * _FP32_TOL["value"]
    with pytest.raises(CompileTrainerError, match="value"):
        decision_verdicts(eager=base, compiled=bad_v, precision="highest")
    bad_p = {k: v.clone() for k, v in base.items()}
    bad_p["legal_logprob"][0, 3] -= 2 * _FP32_TOL["legal_logprob"]
    with pytest.raises(CompileTrainerError, match="legal_logprob"):
        decision_verdicts(eager=base, compiled=bad_p, precision="highest")
    with pytest.raises(CompileTrainerError, match="no fp32 reference"):
        decision_verdicts(eager=base, compiled=base, precision="high")


def test_train_graph_gradient_cosine_is_gated():
    """The single-graph miscompile's gradient had cosine 0.778 to eager's — PPO stepping the wrong
    way. The train verdict must refuse that at fp32, and apply the relative rule at TF32."""
    from agents.model.compile_trainer import _MIN_GRAD_COSINE, train_verdict
    g = torch.randn(1000, generator=torch.Generator().manual_seed(0))
    f = torch.randn(2, 4, generator=torch.Generator().manual_seed(2))   # informative rows
    assert "grad cosine" in train_verdict(eager={"features": f, "grad": g},
                                          compiled={"features": f, "grad": g.clone()},
                                          precision="highest")
    skew = g + 0.6 * torch.randn(1000, generator=torch.Generator().manual_seed(1))
    with pytest.raises(CompileTrainerError, match="gradient DISAGREES"):
        train_verdict(eager={"features": f, "grad": g}, compiled={"features": f, "grad": skew},
                      precision="highest")
    assert _MIN_GRAD_COSINE == 0.9999
    ref = {"features": f, "grad": g}
    tf32_eager = {"features": f + 1e-3, "grad": g + 1e-3 * torch.randn(1000)}
    with pytest.raises(CompileTrainerError, match="gradient DISAGREES"):
        train_verdict(eager=tf32_eager, compiled={"features": f + 1e-3, "grad": skew},
                      reference=ref, precision="high")
    assert "1-cos" in train_verdict(eager=tf32_eager, compiled=tf32_eager, reference=ref,
                                    precision="high")
